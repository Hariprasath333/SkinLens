"""
SkinLens Clinical web application (FastAPI + Jinja2 + SQLite/PostgreSQL).

Run from the clinical_system folder:
    uvicorn webapp.main:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import base64
import json
import logging
import secrets
import socket
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

import bcrypt
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from . import config, insight, patient
from .analysis import BadImage, analyze, compute, decode_image, run_lime
from .db import Feedback, Patient, Scan, SessionLocal, User, init_db, next_patient_code
from .diseases import ABBR, DISEASES
from .ml.predictor import CLASS_FULL, CLASSES, get_predictor
from .report import build_pdf
from .research import DATA_FILTERS, dashboard_stats, export_csv

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("skinlens")
BODY_SITES = ["unknown", "face", "scalp", "ear", "neck", "chest", "abdomen", "back", "trunk",
              "upper extremity", "hand", "lower extremity", "foot", "acral", "genital", "oral/genital"]
GENDERS = ["unknown", "female", "male", "other"]  # first = default ("Not specified")


def local_ip() -> str | None:
    """This computer's address on the Wi-Fi/LAN (no packet is actually sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def phone_url_message(argv: list[str] | None = None) -> str:
    """What to open on the phone, based on uvicorn's --host/--port options."""
    argv = sys.argv if argv is None else argv

    def opt(name, default):
        for i, a in enumerate(argv):
            if a == name and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return default
    host, port = opt("--host", "127.0.0.1"), opt("--port", "8000")
    if host in ("127.0.0.1", "localhost"):
        return (f"SkinLens is running on this computer only (http://127.0.0.1:{port}). To use it on a phone, "
                f"start it with: uvicorn webapp.main:app --host 0.0.0.0 --port {port}")
    ip = local_ip() if host == "0.0.0.0" else host
    if not ip:
        return f"SkinLens is listening on all addresses, port {port}, but no Wi-Fi/LAN address was found."
    return (f"Open on your phone (same Wi-Fi): http://{ip}:{port}\n"
            f"   On this computer:              http://127.0.0.1:{port}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    try:
        p = get_predictor()  # load the model once at start-up
        log.info("Model loaded: %s on %s", p.name, p.device)
    except Exception:  # noqa: BLE001
        log.exception("Model could not be loaded; analysis pages will show an error")
    if "uvicorn" in " ".join(sys.argv):
        msg = phone_url_message()
        print("\n" + "=" * 70 + "\n   " + msg + "\n" + "=" * 70 + "\n", flush=True)
    yield


app = FastAPI(title="SkinLens Clinical", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=config.SECRET_KEY, max_age=config.SESSION_MAX_AGE,
                   same_site="lax", https_only=config.COOKIE_SECURE, session_cookie="skinlens_session")
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=config.BASE_DIR / "templates")
templates.env.globals["now"] = datetime.now
templates.env.globals["conf_high"] = config.CONFIDENCE_HIGH
templates.env.globals["conf_medium"] = config.CONFIDENCE_MEDIUM


# ----------------------------------------------------------------------------- helpers
class NotLoggedIn(Exception):
    pass


@app.exception_handler(NotLoggedIn)
async def _to_login(request: Request, _exc):
    return RedirectResponse(f"/login?next={request.url.path}", status_code=303)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    uid = request.session.get("user_id")
    user = db.get(User, uid) if uid else None
    if user is None:
        request.session.clear()
        raise NotLoggedIn()
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Admins only")
    return user


def csrf_token(request: Request) -> str:
    tok = request.session.get("csrf")
    if not tok:
        tok = request.session["csrf"] = secrets.token_urlsafe(32)
    return tok


def check_csrf(request: Request, token: str) -> None:
    if not token or not secrets.compare_digest(token, request.session.get("csrf", "")):
        raise HTTPException(400, "Form expired. Please go back, reload the page and try again.")


def render(request: Request, name: str, user: User | None = None, status_code: int = 200, **ctx):
    flash = request.session.pop("flash", None)
    return templates.TemplateResponse(request, name, dict(user=user, csrf=csrf_token(request), flash=flash,
                                                          **ctx), status_code=status_code)


def flash(request: Request, msg: str, kind: str = "info") -> None:
    request.session["flash"] = {"msg": msg, "kind": kind}


def visible_patients(user: User):
    q = select(Patient)
    return q if user.is_admin else q.where(Patient.created_by_id == user.id)


def get_patient(db: Session, user: User, code: str) -> Patient:
    p = db.scalar(visible_patients(user).where(Patient.patient_code == code))
    if p is None:
        raise HTTPException(404, "Patient not found")
    return p


def get_scan(db: Session, user: User, scan_id: int) -> Scan:
    s = db.get(Scan, scan_id)
    if s is None or not (user.is_admin or s.patient.created_by_id == user.id):
        raise HTTPException(404, "Scan not found")
    return s


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt(rounds=12)).decode()


def verify_pw(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode())
    except ValueError:
        return False


def safe_next(url: str | None) -> str:
    return url if url and url.startswith("/") and not url.startswith("//") else "/dashboard"


# ----------------------------------------------------------------------------- auth
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request, db: Session = Depends(get_db)):
    """Start screen: choose doctor mode or patient mode."""
    uid = request.session.get("user_id")
    user = db.get(User, uid) if uid else None
    return render(request, "start.html", None, staff=user)


# ----------------------------------------------------------------------------- PWA
@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return FileResponse(config.BASE_DIR / "static" / "manifest.json", media_type="application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    # served from the root so it may control the whole app; never cached by the browser itself
    return FileResponse(config.BASE_DIR / "static" / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(config.BASE_DIR / "static" / "icons" / "favicon.ico", media_type="image/x-icon",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/offline", response_class=HTMLResponse, include_in_schema=False)
def offline(request: Request):
    return templates.TemplateResponse(request, "offline.html", {})


# ----------------------------------------------------------------------------- patient mode
NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}


@app.get("/check", response_class=HTMLResponse)
def check_page(request: Request):
    return render(request, "patient_check.html", None, sites=BODY_SITES, max_mb=config.MAX_UPLOAD_MB,
                  model=get_predictor().describe())


@app.post("/check")
async def check_submit(request: Request, image: UploadFile | None = File(None),
                       image_gallery: UploadFile | None = File(None), age: str = Form(""), sex: str = Form(""),
                       body_site: str = Form("unknown"), csrf: str = Form("")):
    """Patient mode: no login, nothing stored. The photo is analysed in memory; the upload's temp
    file is closed (deleted) at once, and the result is only sent back to the phone."""
    check_csrf(request, csrf)
    ctx = dict(sites=BODY_SITES, max_mb=config.MAX_UPLOAD_MB, model=get_predictor().describe())
    data = await read_upload(image, image_gallery)
    if not data:
        return render(request, "patient_check.html", None, error="Please take a photo or choose one first.",
                      status_code=400, **ctx)
    allowed, wait = patient.limiter.check(patient.client_ip(request))
    if not allowed:
        return render(request, "patient_check.html", None, status_code=429, **ctx,
                      error=f"You have reached the limit of {patient.limiter.limit} checks per hour. "
                            f"Please try again in about {max(1, wait // 60)} minutes.")
    try:
        age_v = int(age) if age.strip() else None
        age_v = age_v if age_v is None or 0 <= age_v <= 120 else None
    except ValueError:
        age_v = None
    from starlette.concurrency import run_in_threadpool
    try:
        rgb = decode_image(data)
        r = await run_in_threadpool(compute, rgb, age_v, sex if sex in ("female", "male") else None,
                                    body_site if body_site in BODY_SITES else "unknown")
    except BadImage as e:
        return render(request, "patient_check.html", None, error=str(e), status_code=400, **ctx)
    finally:
        del data
    adv, tone = patient.advice(r["code"], r["level"])
    site_v = body_site if body_site in BODY_SITES else "unknown"
    x = insight.build(r["code"], r["probs"], r["features"], r["level"], age=age_v,
                      sex=sex if sex in ("female", "male") else None, site=site_v, uses_metadata=r["uses_metadata"])
    r.update(advice=adv, tone=tone, plain=patient.plain_result(r["code"]),
             heatmap_line=patient.heatmap_line(r["meas"]), x=x)
    pdf = await run_in_threadpool(patient.summary_pdf, r)
    from .ml import xai
    photo = patient.to_data_uri(r["rgb"])
    token = patient.lime_token(photo, age_v, sex, site_v, r["code"])
    resp = render(request, "patient_result.html", None, r=r, x=x, level_words=patient.LEVEL_WORDS[r["level"]],
                  photo=photo, heat=patient.to_data_uri(xai.overlay(r["rgb"], r["cam"])), lime_token=token,
                  quality=r["quality"], pdf_b64=base64.b64encode(pdf).decode(),
                  pdf_name=f"skinlens_summary_{datetime.now():%Y%m%d_%H%M}.pdf")
    resp.headers.update(NO_STORE)
    return resp


@app.post("/check/lime")
async def check_lime(request: Request):
    """Patient mode, second explanation (LIME, about a minute on a CPU). The phone sends back the photo it
    is showing plus the signed token it got with the result; the photo is analysed in memory only."""
    try:
        body = await request.json()
        photo, token = str(body.get("photo", "")), str(body.get("token", ""))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Bad request")
    claims = patient.read_lime_token(token, photo)
    if claims is None:
        raise HTTPException(403, "This result has expired. Please check the photo again.")
    allowed, _wait = patient.lime_limiter.check(patient.client_ip(request))
    if not allowed:
        raise HTTPException(429, "Too many requests. Please try again later.")
    from starlette.concurrency import run_in_threadpool
    data = base64.b64decode(photo.split(",", 1)[1])
    out = await run_in_threadpool(patient.lime_picture, decode_image(data), claims)
    return JSONResponse(out, headers=NO_STORE)


# ----------------------------------------------------------------------------- disease information
def _learn_ctx(request: Request, db: Session):
    uid = request.session.get("user_id")
    user = db.get(User, uid) if uid else None
    return user, ("base.html" if user else "patient_base.html")


@app.get("/learn", response_class=HTMLResponse)
def learn_index(request: Request, db: Session = Depends(get_db)):
    user, layout = _learn_ctx(request, db)
    codes = [c for c in get_predictor().classes if c in DISEASES] or list(DISEASES)
    return render(request, "learn_index.html", user, layout=layout, diseases=DISEASES, codes=codes, abbr=ABBR)


@app.get("/learn/abcde", response_class=HTMLResponse)
def learn_abcde(request: Request, db: Session = Depends(get_db)):
    user, layout = _learn_ctx(request, db)
    return render(request, "learn_abcde.html", user, layout=layout)


@app.get("/learn/{code}", response_class=HTMLResponse)
def learn_disease(code: str, request: Request, db: Session = Depends(get_db)):
    code = code.upper()
    if code not in DISEASES:
        raise HTTPException(404, "Unknown lesion type")
    user, layout = _learn_ctx(request, db)
    return render(request, "learn_disease.html", user, layout=layout, code=code, d=DISEASES[code], abbr=ABBR[code])


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/dashboard"):
    return render(request, "login.html", next=safe_next(next), allow_register=config.ALLOW_REGISTRATION)


# Login lockout: 5 wrong passwords for one username (or from one address) in 15 minutes -> wait.
_LOGIN_FAILS: dict[str, list[float]] = {}
_LOCK_LIMIT, _LOCK_WINDOW = 5, 15 * 60


def _client_ip(request: Request) -> str:
    # behind a Cloudflare tunnel every request comes from 127.0.0.1; the real address is in this header
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _recent_fails(key: str) -> list[float]:
    import time
    now = time.time()
    fails = [t for t in _LOGIN_FAILS.get(key, []) if now - t < _LOCK_WINDOW]
    _LOGIN_FAILS[key] = fails
    return fails


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...),
          next: str = Form("/dashboard"), csrf: str = Form(""), db: Session = Depends(get_db)):
    import time
    check_csrf(request, csrf)
    keys = ["u:" + username.strip().lower(), "ip:" + _client_ip(request)]
    if any(len(_recent_fails(k)) >= _LOCK_LIMIT for k in keys):
        log.warning("login locked for %s", keys)
        return render(request, "login.html", error="Too many wrong attempts. Please wait 15 minutes and try again.",
                      next=safe_next(next), allow_register=config.ALLOW_REGISTRATION, status_code=429)
    user = db.scalar(select(User).where(User.username == username.strip().lower()))
    if user is None or not verify_pw(password, user.password_hash):
        for k in keys:
            _LOGIN_FAILS.setdefault(k, []).append(time.time())
        return render(request, "login.html", error="Wrong username or password.", next=safe_next(next),
                      allow_register=config.ALLOW_REGISTRATION, status_code=401)
    for k in keys:
        _LOGIN_FAILS.pop(k, None)
    request.session.clear()  # new session id contents after login
    request.session["user_id"] = user.id
    flash(request, f"Welcome, {user.full_name}.", "success")
    return RedirectResponse(safe_next(next), status_code=303)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if not config.ALLOW_REGISTRATION:
        raise HTTPException(403, "Registration is disabled. Ask an administrator.")
    return render(request, "register.html")


@app.post("/register")
def register(request: Request, full_name: str = Form(...), username: str = Form(...),
             password: str = Form(...), password2: str = Form(...), csrf: str = Form(""),
             db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    if not config.ALLOW_REGISTRATION:
        raise HTTPException(403, "Registration is disabled.")
    username = username.strip().lower()
    error = None
    if not (3 <= len(username) <= 64) or not username.replace(".", "").replace("_", "").isalnum():
        error = "Username: 3-64 characters, letters, numbers, dot or underscore."
    elif len(password) < 8:
        error = "Password must be at least 8 characters."
    elif password != password2:
        error = "The two passwords do not match."
    elif db.scalar(select(User).where(User.username == username)):
        error = "That username is already taken."
    if error:
        return render(request, "register.html", error=error, full_name=full_name, username=username,
                      status_code=400)
    first = db.scalar(select(func.count(User.id))) == 0
    user = User(username=username, full_name=full_name.strip()[:120], password_hash=hash_pw(password),
                role="admin" if first else "doctor")
    db.add(user)
    db.commit()
    request.session.clear()
    request.session["user_id"] = user.id
    flash(request, "Account created." + (" You are the first user, so you are the Admin." if first else ""),
          "success")
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/logout")
def logout(request: Request, csrf: str = Form("")):
    check_csrf(request, csrf)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ----------------------------------------------------------------------------- pages
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    pats = visible_patients(user).subquery()
    n_pat = db.scalar(select(func.count()).select_from(pats))
    scans_q = select(Scan).join(Patient)
    if not user.is_admin:
        scans_q = scans_q.where(Patient.created_by_id == user.id)
    n_scan = db.scalar(select(func.count()).select_from(scans_q.subquery()))
    n_dis = db.scalar(select(func.count()).select_from(
        scans_q.where(Scan.risk.in_(["malignant", "premalignant"])).subquery()))
    today = db.scalar(select(func.count()).select_from(
        scans_q.where(Scan.created_at >= datetime.combine(date.today(), datetime.min.time())).subquery()))
    recent = db.scalars(scans_q.order_by(Scan.created_at.desc()).limit(8)).all()
    return render(request, "dashboard.html", user, stats=dict(patients=n_pat, scans=n_scan, disease=n_dis,
                                                               today=today), recent=recent,
                  model=get_predictor().describe())


@app.get("/patients/new", response_class=HTMLResponse)
def new_patient_page(request: Request, user: User = Depends(current_user)):
    return render(request, "patient_new.html", user, genders=GENDERS, today=date.today().isoformat())


@app.post("/patients/new")
def new_patient(request: Request, name: str = Form(...), age: str = Form(""), gender: str = Form("unknown"),
                contact: str = Form(""), visit_date: str = Form(""), csrf: str = Form(""),
                user: User = Depends(current_user), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    name = name.strip()
    try:
        age_v = int(age) if age.strip() else None
        if age_v is not None and not 0 <= age_v <= 120:
            raise ValueError
        vdate = date.fromisoformat(visit_date) if visit_date else date.today()
    except ValueError:
        return render(request, "patient_new.html", user, genders=GENDERS, today=date.today().isoformat(),
                      error="Please check the age (0-120) and the date.", form=dict(name=name, contact=contact),
                      status_code=400)
    if not name:
        return render(request, "patient_new.html", user, genders=GENDERS, today=date.today().isoformat(),
                      error="Name is required.", status_code=400)
    for attempt in range(3):  # retry if two users create a patient at the same moment
        p = Patient(patient_code=next_patient_code(db), name=name[:120], age=age_v,
                    gender=gender if gender in GENDERS else "unknown", contact=contact.strip()[:120],
                    visit_date=vdate, created_by_id=user.id)
        db.add(p)
        try:
            db.commit()
            break
        except Exception:  # noqa: BLE001
            db.rollback()
            if attempt == 2:
                raise
    flash(request, f"Patient {p.patient_code} created. You can now upload an image.", "success")
    return RedirectResponse(f"/analyze?patient={p.patient_code}", status_code=303)


@app.get("/patients", response_class=HTMLResponse)
def patients(request: Request, q: str = "", user: User = Depends(current_user), db: Session = Depends(get_db)):
    query = visible_patients(user)
    if q.strip():
        like = f"%{q.strip()}%"
        query = query.where(or_(Patient.patient_code.ilike(like), Patient.name.ilike(like)))
    rows = db.scalars(query.order_by(Patient.created_at.desc()).limit(200)).all()
    return render(request, "patients.html", user, patients=rows, q=q)


@app.get("/patients/{code}", response_class=HTMLResponse)
def patient_detail(code: str, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return render(request, "patient_detail.html", user, patient=get_patient(db, user, code))


@app.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request, patient: str = "", user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    rows = db.scalars(visible_patients(user).order_by(Patient.created_at.desc()).limit(500)).all()
    return render(request, "analyze.html", user, patients=rows, selected=patient, sites=BODY_SITES,
                  model=get_predictor().describe(), max_mb=config.MAX_UPLOAD_MB)


async def read_upload(*files: UploadFile | None) -> bytes:
    """The first non-empty file of the form ("Take photo" or "Choose from gallery"). Upload temp
    files are closed (deleted from disk) straight away."""
    data = b""
    for f in files:
        if f is None:
            continue
        try:
            if not data:
                data = await f.read()
        finally:
            await f.close()
    return data


@app.post("/analyze")
async def analyze_submit(request: Request, background: BackgroundTasks, patient_code: str = Form(...),
                         body_site: str = Form("unknown"), image: UploadFile | None = File(None),
                         image_gallery: UploadFile | None = File(None), csrf: str = Form(""),
                         user: User = Depends(current_user), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    patient = get_patient(db, user, patient_code)
    data = await read_upload(image, image_gallery)
    try:
        from starlette.concurrency import run_in_threadpool
        scan = await run_in_threadpool(analyze, patient, user.id, data,
                                       body_site if body_site in BODY_SITES else "unknown")
    except BadImage as e:
        rows = db.scalars(visible_patients(user).order_by(Patient.created_at.desc()).limit(500)).all()
        return render(request, "analyze.html", user, patients=rows, selected=patient_code, sites=BODY_SITES,
                      model=get_predictor().describe(), max_mb=config.MAX_UPLOAD_MB, error=str(e),
                      status_code=400)
    background.add_task(run_lime, scan.id)
    return RedirectResponse(f"/scans/{scan.id}", status_code=303)


@app.get("/scans/{scan_id}", response_class=HTMLResponse)
def scan_page(scan_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    scan = get_scan(db, user, scan_id)
    return render(request, "result.html", user, scan=scan, x=insight.for_scan(scan), dx_options=DX_OPTIONS,
                  my_fb=scan.feedback_by(user.id))


# ----------------------------------------------------------------------------- doctor feedback
DX_OPTIONS = [(c, CLASS_FULL[c]) for c in CLASSES] + [("OTHER", "Other / not in this list")]
AGREEMENT = {"agree": "Agree", "disagree": "Disagree", "unsure": "Unsure"}


@app.post("/scans/{scan_id}/feedback")
def scan_feedback(scan_id: int, request: Request, agreement: str = Form(""), correct_class: str = Form(""),
                  usefulness: str = Form(""), comment: str = Form(""), csrf: str = Form(""),
                  user: User = Depends(current_user), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    scan = get_scan(db, user, scan_id)
    error = None
    try:
        stars = int(usefulness)
    except ValueError:
        stars = 0
    if agreement not in AGREEMENT:
        error = "Please choose Agree, Disagree or Unsure."
    elif agreement == "disagree" and correct_class not in dict(DX_OPTIONS):
        error = "You disagreed: please pick the diagnosis you think is correct."
    elif agreement == "disagree" and correct_class == scan.predicted_class:
        error = "The diagnosis you picked is the same as the AI result. Choose another one, or select Agree."
    elif not 1 <= stars <= 5:
        error = "Please rate how useful the explanation was (1 to 5 stars)."
    if error:
        draft = dict(agreement=agreement, correct_class=correct_class, usefulness=stars or None, comment=comment)
        return render(request, "result.html", user, scan=scan, x=insight.for_scan(scan), dx_options=DX_OPTIONS,
                      my_fb=draft, fb_error=error, status_code=400)
    fb = scan.feedback_by(user.id)
    if fb is None:
        fb = Feedback(scan_id=scan.id, doctor_id=user.id, agreement=agreement, usefulness=stars)
        db.add(fb)
    fb.agreement, fb.usefulness = agreement, stars
    fb.correct_class = correct_class if agreement == "disagree" else None
    fb.comment = comment.strip()[:2000]
    fb.updated_at = datetime.now()
    db.commit()
    flash(request, "Thank you - your feedback was saved.", "success")
    return RedirectResponse(f"/scans/{scan.id}#feedback", status_code=303)


@app.get("/scans/{scan_id}/lime")
def lime_status(scan_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    s = get_scan(db, user, scan_id)
    return JSONResponse(dict(status=s.lime_status, summary=s.lime_summary,
                             url=f"/media/{s.id}/lime" if s.lime_status == "done" else None))


@app.get("/scans/{scan_id}/report.pdf")
def report(scan_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    s = get_scan(db, user, scan_id)
    pdf = build_pdf(s)
    fname = f"{s.patient.patient_code}_scan{s.id}_{s.created_at:%Y%m%d}.pdf"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/media/{scan_id}/{kind}")
def media(scan_id: int, kind: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    s = get_scan(db, user, scan_id)
    path = {"original": s.image_path, "gradcam": s.gradcam_path, "lime": s.lime_path}.get(kind)
    if not path:
        raise HTTPException(404)
    full = (config.STORAGE_DIR / path).resolve()
    if config.STORAGE_DIR.resolve() not in full.parents or not full.exists():
        raise HTTPException(404)
    return FileResponse(full, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request, user: User = Depends(admin_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(User, func.count(func.distinct(Patient.id)))
        .outerjoin(Patient, Patient.created_by_id == User.id).group_by(User.id).order_by(User.created_at)).all()
    return render(request, "admin_users.html", user, rows=rows)


@app.post("/admin/users/{uid}/role")
def admin_set_role(uid: int, request: Request, role: str = Form(...), csrf: str = Form(""),
                   user: User = Depends(admin_user), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    target = db.get(User, uid)
    if target is None or role not in ("doctor", "admin"):
        raise HTTPException(400)
    if target.id == user.id and role != "admin":
        flash(request, "You cannot remove your own admin role.", "error")
    else:
        target.role = role
        db.commit()
        flash(request, f"{target.full_name} is now {role}.", "success")
    return RedirectResponse("/admin/users", status_code=303)


# ----------------------------------------------------------------------------- research (admin)
@app.get("/admin/research", response_class=HTMLResponse)
def research_dashboard(request: Request, data: str = "all", user: User = Depends(admin_user),
                       db: Session = Depends(get_db)):
    data = data if data in DATA_FILTERS else "all"
    return render(request, "research.html", user, s=dashboard_stats(db, data), data=data, filters=DATA_FILTERS)


@app.get("/admin/export.csv")
def export(include_demo: int = 0, user: User = Depends(admin_user), db: Session = Depends(get_db)):
    body = "﻿" + export_csv(db, include_demo=bool(include_demo))  # BOM: Excel opens UTF-8 correctly
    name = f"skinlens_scans_{datetime.now():%Y%m%d_%H%M}{'_with_demo' if include_demo else ''}.csv"
    return Response(body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ----------------------------------------------------------------------------- model information
def paper_figures() -> list[Path]:
    d = config.PAPER_FIGURES_DIR
    return sorted(p for p in d.glob("*.png") if p.is_file()) if d.is_dir() else []


@app.get("/model-info", response_class=HTMLResponse)
def model_info(request: Request, user: User = Depends(current_user)):
    info = json.loads((config.BASE_DIR / "model_info.json").read_text(encoding="utf-8"))
    figs = [dict(file=p.name, title=p.stem.replace("_", " ").replace("-", " ").capitalize()) for p in paper_figures()]
    return render(request, "model_info.html", user, info=info, figures=figs, model=get_predictor().describe())


@app.get("/paper-figures/{name}")
def paper_figure(name: str, user: User = Depends(current_user)):
    match = next((p for p in paper_figures() if p.name == name), None)  # only files listed in the folder
    if match is None:
        raise HTTPException(404)
    return FileResponse(match, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


# ----------------------------------------------------------------------------- Validation Lab (admin only)
import re as _re  # noqa: E402

RUN_NAME = _re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")
PRESETS = {"all": (["T1", "T2", "T3", "T4", "T5", "T6", "T7"], "sample"),
           "t1_full": (["T1"], "full"), **{t: ([t], "sample") for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]}}


def _runs() -> list:
    from .validation.runner import Run
    root = config.VALIDATION_RESULTS_DIR
    if not root.exists():
        return []
    return [Run(d) for d in sorted(root.iterdir(), reverse=True) if d.is_dir() and RUN_NAME.match(d.name)
            and (d / "run.json").exists()]


def _get_run(name: str):
    from .validation.runner import Run
    if not RUN_NAME.match(name) or not (config.VALIDATION_RESULTS_DIR / name / "run.json").exists():
        raise HTTPException(404, "Validation run not found")
    return Run(config.VALIDATION_RESULTS_DIR / name)


@app.get("/admin/validation", response_class=HTMLResponse)
def validation_lab(request: Request, run: str = "", user: User = Depends(admin_user)):
    from .validation import criteria, data as vdata
    from .validation.report import REPORT_NAME, app_versions, verdicts
    from .validation.runner import is_running
    runs = _runs()
    sel = _get_run(run) if run else (runs[0] if runs else None)
    try:
        check = vdata.check()
    except Exception as e:  # noqa: BLE001
        check = dict(missing=[str(e)], root=str(config.VALIDATION_DATA_DIR))
    ctx = dict(runs=[(r.dir.name, r.info.get("tests"), r.info.get("t1_mode"), is_running(r)) for r in runs], sel=None,
               check=check, criteria=criteria.CRITERIA, any_running=any(is_running(r) for r in runs))
    if sel is not None:
        res = {t: sel.load_json(f"{t}_metrics.json") for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]}
        charts = sorted(p.name for p in sel.charts.glob("*.png"))
        files = sorted(p.name for p in sel.dir.iterdir() if p.suffix in (".csv", ".json", ".xml", ".log"))
        ctx.update(sel=sel.dir.name, info=sel.info, versions=app_versions(sel), state=sel.state(), res=res, charts=charts, files=files,
                   running=is_running(sel), verdicts=verdicts(sel), has_report=(sel.dir / REPORT_NAME).exists())
    return render(request, "validation.html", user, **ctx)


@app.post("/admin/validation/start")
def validation_start(request: Request, preset: str = Form(...), csrf: str = Form(""), user: User = Depends(admin_user)):
    check_csrf(request, csrf)
    from .validation.runner import is_running, new_run, start_background
    if preset not in PRESETS:
        raise HTTPException(400, "Unknown test")
    if any(is_running(r) for r in _runs()):
        flash(request, "A validation run is already running. Wait for it to finish (or stop it) first.", "error")
        return RedirectResponse("/admin/validation", status_code=303)
    tests, mode = PRESETS[preset]
    run = new_run(tests, mode)
    start_background(run)
    flash(request, f"Started {', '.join(tests)}{' (full test set)' if mode == 'full' else ''} in run {run.dir.name}.", "success")
    return RedirectResponse(f"/admin/validation?run={run.dir.name}", status_code=303)


@app.post("/admin/validation/{name}/resume")
def validation_resume(name: str, request: Request, csrf: str = Form(""), user: User = Depends(admin_user)):
    check_csrf(request, csrf)
    from .validation.runner import is_running, start_background
    run = _get_run(name)
    if any(is_running(r) for r in _runs()):
        flash(request, "A validation run is already running.", "error")
    else:
        st = run.state()
        for t, s in st.items():                    # failed or interrupted tests are tried again
            if s.get("status") in ("running", "failed"):
                run.update(t, status="pending")
        start_background(run)
        flash(request, f"Resumed run {name}: finished results are kept.", "success")
    return RedirectResponse(f"/admin/validation?run={name}", status_code=303)


@app.post("/admin/validation/{name}/report")
async def validation_rebuild_report(name: str, request: Request, csrf: str = Form(""), user: User = Depends(admin_user)):
    check_csrf(request, csrf)
    from starlette.concurrency import run_in_threadpool
    from .validation.report import build_report
    await run_in_threadpool(build_report, _get_run(name))
    flash(request, "Report rebuilt.", "success")
    return RedirectResponse(f"/admin/validation?run={name}", status_code=303)


@app.get("/admin/validation/{name}/status")
def validation_status(name: str, user: User = Depends(admin_user)):
    from .validation.runner import is_running
    run = _get_run(name)
    return JSONResponse(dict(state=run.state(), running=is_running(run)), headers={"Cache-Control": "no-store"})


@app.get("/admin/validation/{name}/chart/{chart}")
def validation_chart(name: str, chart: str, user: User = Depends(admin_user)):
    run = _get_run(name)
    match = next((p for p in run.charts.glob("*.png") if p.name == chart), None)
    if match is None:
        raise HTTPException(404)
    return FileResponse(match, media_type="image/png", headers={"Cache-Control": "no-cache"})


@app.get("/admin/validation/{name}/files/{fname}")
def validation_file(name: str, fname: str, user: User = Depends(admin_user)):
    from .validation.report import REPORT_NAME
    run = _get_run(name)
    match = next((p for p in run.dir.iterdir() if p.name == fname and p.is_file() and
                  (p.suffix in (".csv", ".json", ".xml", ".log") or p.name == REPORT_NAME)), None)
    if match is None:
        raise HTTPException(404)
    media = {".csv": "text/csv", ".json": "application/json", ".xml": "application/xml", ".log": "text/plain",
             ".pdf": "application/pdf"}[match.suffix]
    return FileResponse(match, media_type=media, filename=match.name)


# ----------------------------------------------------------------------------- About / Version
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, db: Session = Depends(get_db)):
    from starlette.concurrency import run_in_threadpool
    from .validation.identity import model_identity
    from .version import APP_VERSION, RELEASE_DATE
    uid = request.session.get("user_id")
    user = db.get(User, uid) if uid else None
    model = await run_in_threadpool(model_identity)
    latest = None
    f = config.VALIDATION_RESULTS_DIR / "latest.json"
    if f.exists():
        latest = json.loads(f.read_text(encoding="utf-8"))
    return render(request, "about.html", user, layout="base.html" if user else "patient_base.html",
                  version=APP_VERSION, release=RELEASE_DATE, model=model, latest=latest)


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "model": p.name, "device": str(p.device)}


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith(("/media", "/scans/", "/check/lime")) and request.url.path.endswith(("lime", ".pdf")):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    uid = request.session.get("user_id")
    with SessionLocal() as db:
        user = db.get(User, uid) if uid else None
    return render(request, "error.html", user, code=exc.status_code, detail=exc.detail, status_code=exc.status_code)
