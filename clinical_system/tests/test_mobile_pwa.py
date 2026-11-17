"""PWA (manifest, icons, service worker, offline page), start screen, patient mode (saves nothing,
advice, PDF, rate limit) and phone-access helpers."""
import base64
import re

from PIL import Image

from conftest import ROOT, SAMPLE_IMAGE, csrf_of

STATIC = ROOT / "webapp" / "static"


def db_counts():
    from webapp.db import Feedback, Patient, Scan, SessionLocal, User
    with SessionLocal() as db:
        return tuple(db.query(m).count() for m in (User, Patient, Scan, Feedback))


def storage_files():
    from webapp import config
    return sorted(p.name for p in config.STORAGE_DIR.rglob("*") if p.is_file())


def check(c, data=SAMPLE_IMAGE.read_bytes(), field="image", **form):
    tok = csrf_of(c.get("/check").text)
    return c.post("/check", data=dict(csrf=tok, **form), files={field: ("phone.jpg", data, "image/jpeg")})


def test_manifest_icons_service_worker(client):
    r = client.get("/manifest.webmanifest")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/manifest+json")
    m = r.json()
    assert (m["name"], m["short_name"], m["theme_color"], m["background_color"], m["display"], m["start_url"]) == \
        ("SkinLens", "SkinLens", "#0077B6", "#F5FAFD", "standalone", "/")
    sizes = {(i["sizes"], i.get("purpose", "any")) for i in m["icons"]}
    assert {("192x192", "any"), ("512x512", "any"), ("512x512", "maskable")} <= sizes
    for i in m["icons"]:
        assert client.get(i["src"]).status_code == 200
        w, h = Image.open(STATIC / i["src"].replace("/static/", "")).size
        assert f"{w}x{h}" == i["sizes"]

    sw = client.get("/sw.js")
    assert sw.status_code == 200 and "javascript" in sw.headers["content-type"]
    assert sw.headers["service-worker-allowed"] == "/" and "no-cache" in sw.headers["cache-control"]
    js = sw.text
    shell = re.findall(r'"([^"]+)"', re.search(r"const SHELL = \[(.*?)\];", js, re.S).group(1))
    assert "/offline" in shell and "/static/css/style.css" in shell
    assert not any(p.startswith(("/media", "/scans", "/check", "/patients", "/admin", "/paper-figures")) for p in shell)
    assert "caches.match(\"/offline\")" in js                          # offline fallback for pages
    for p in shell:                                                     # every shell file exists
        assert client.get(p).status_code == 200, p
    assert "You are offline" in client.get("/offline").text


def test_start_screen_and_meta(client):
    page = client.get("/").text
    for t in ("I'm a doctor / clinic staff", "Check my skin (patient)", 'rel="manifest"', 'name="theme-color"',
              "apple-touch-icon", "Install SkinLens", "SkinLens is a research project, not a medical diagnosis."):
        assert t in page, t
    assert 'href="/check"' in page and 'href="/login"' in page
    assert "Back to start" in client.get("/login").text


def test_patient_check_page():
    from fastapi.testclient import TestClient
    from webapp.main import app
    c = TestClient(app)                                  # no login
    page = c.get("/check").text
    assert 'capture="environment"' in page and 'name="image_gallery"' in page and "Take photo" in page
    assert "Choose from gallery" in page and "Good light" in page and "About 10 cm away" in page
    assert "SkinLens is a research project, not a medical diagnosis." in page
    assert "about 10–20 seconds" in page or "about 10-20 seconds" in page


def test_patient_mode_saves_nothing(client):
    from fastapi.testclient import TestClient
    from webapp import patient
    from webapp.main import app
    patient.limiter.reset()
    c = TestClient(app)
    before_db, before_files = db_counts(), storage_files()
    assert c.post("/check", data={"csrf": "x"}, files={"image": ("a.jpg", b"1", "image/jpeg")}).status_code == 400  # CSRF
    r = check(c, field="image_gallery", age="44", sex="female", body_site="back")
    assert r.status_code == 200
    assert "no-store" in r.headers["cache-control"]
    html = r.text
    for t in ("Your result", "What the app found", "This looks most like", "How sure the app is",
              "Where the app looked", "Download summary", "SkinLens is a research project, not a medical diagnosis.",
              "data:image/jpeg;base64,"):
        assert t in html, t
    assert any(a in html for a in (patient.ADVICE_CONCERN, patient.ADVICE_LESS, patient.ADVICE_UNSURE))
    assert "Top 3" not in html and "%</b>" not in html                  # no probability table
    pdf = base64.b64decode(re.search(r'data:application/pdf;base64,([A-Za-z0-9+/=]+)', html).group(1))
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000
    # nothing was stored: same database rows, no new files on disk
    assert db_counts() == before_db and storage_files() == before_files
    # bad files are refused politely, still nothing stored
    r = check(c, data=b"not an image")
    assert r.status_code == 400 and "not a readable image" in r.text and db_counts() == before_db
    assert "take a photo or choose one" in c.post("/check", data={"csrf": csrf_of(c.get("/check").text)}).text


def test_patient_advice_rules():
    from webapp.patient import ADVICE_CONCERN, ADVICE_LESS, ADVICE_UNSURE, advice, plain_result
    for code in ("MEL", "BCC", "AK"):
        assert advice(code, "High") == (ADVICE_CONCERN, "concern")
        assert advice(code, "Medium")[0] == ADVICE_CONCERN
    for code in ("NV", "BKL", "DF", "VASC"):
        assert advice(code, "High") == (ADVICE_LESS, "calm")
    for code in ("MEL", "NV"):
        assert advice(code, "Low") == (ADVICE_UNSURE, "unsure")
    assert ADVICE_CONCERN == "This may need attention. Please see a dermatologist soon."
    assert ADVICE_UNSURE == "The app is not sure. Please see a doctor."
    assert ADVICE_LESS.startswith("This looks less concerning, but only a doctor can confirm.")
    assert "common mole" in plain_result("NV") and "melanoma" in plain_result("MEL")


def test_patient_rate_limit():
    from fastapi.testclient import TestClient
    from webapp import patient
    from webapp.main import app
    c = TestClient(app)
    old = patient.limiter.limit
    patient.limiter.reset()
    patient.limiter.limit = 2
    try:
        assert check(c).status_code == 200 and check(c).status_code == 200
        r = check(c)
        assert r.status_code == 429 and "limit of 2 checks per hour" in r.text
    finally:
        patient.limiter.limit = old
        patient.limiter.reset()


def test_client_ip_trusts_proxy_only_from_localhost():
    from types import SimpleNamespace
    from webapp.patient import client_ip
    req = lambda host, headers: SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)  # noqa: E731
    assert client_ip(req("127.0.0.1", {"cf-connecting-ip": "203.0.113.9"})) == "203.0.113.9"   # cloudflared
    assert client_ip(req("192.168.1.50", {"cf-connecting-ip": "1.2.3.4"})) == "192.168.1.50"   # cannot fake it
    assert client_ip(req("127.0.0.1", {"x-forwarded-for": "198.51.100.7, 10.0.0.1"})) == "198.51.100.7"


def test_doctor_mode_mobile_and_gallery_upload(client, make_client):
    from webapp.db import Patient, SessionLocal, User
    try:
        _doctor_mode_checks(make_client)
    finally:  # leave the shared test database as it was (test_webapp expects "first user = admin")
        with SessionLocal() as db:
            u = db.query(User).filter_by(username="mobile.doc").one_or_none()
            if u:
                for p in db.query(Patient).filter_by(created_by_id=u.id).all():
                    db.delete(p)
                db.delete(u)
                db.commit()


def _doctor_mode_checks(make_client):
    from test_webapp import add_patient, register
    from webapp.db import SessionLocal, User
    c = make_client()
    register(c, "Dr. Mobile", "mobile.doc")
    with SessionLocal() as db:
        db.query(User).filter_by(username="mobile.doc").one().role = "doctor"
        db.commit()
    page = c.get("/dashboard").text
    assert 'class="bottom-nav"' in page and "has-bottom-nav" in page and "More" in page
    code = add_patient(c, "Mobile Person")
    tok = csrf_of(c.get(f"/analyze?patient={code}").text)
    up = c.get(f"/analyze?patient={code}").text
    assert 'capture="environment"' in up and "Choose from gallery" in up
    r = c.post("/analyze", data=dict(patient_code=code, body_site="back", csrf=tok),
               files={"image_gallery": ("g.jpg", SAMPLE_IMAGE.read_bytes(), "image/jpeg")}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/scans/")
    res = c.get(r.headers["location"]).text
    assert 'data-swipe="#r-swipe"' in res and "Grad-CAM" in res
    r = c.post("/analyze", data=dict(patient_code=code, csrf=tok), follow_redirects=False)
    assert r.status_code == 400 and "file is empty" in r.text


def test_phone_url_message():
    from webapp.main import local_ip, phone_url_message
    msg = phone_url_message(["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8080"])
    ip = local_ip()
    if ip:
        assert f"http://{ip}:8080" in msg and "Open on your phone" in msg
    assert "--host 0.0.0.0" in phone_url_message(["uvicorn", "webapp.main:app", "--port", "8080"])
