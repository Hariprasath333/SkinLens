"""End-to-end tests of the web app: login, patients, upload, prediction, explanation, database, PDF."""
import re
from datetime import date

from conftest import SAMPLE_IMAGE, csrf_of

YEAR = date.today().year


def register(c, name, username, pw="Secret123!"):
    tok = csrf_of(c.get("/register").text)
    return c.post("/register", data=dict(full_name=name, username=username, password=pw, password2=pw, csrf=tok),
                  follow_redirects=False)


def login(c, username, pw="Secret123!"):
    tok = csrf_of(c.get("/login").text)
    return c.post("/login", data=dict(username=username, password=pw, csrf=tok, next="/dashboard"),
                  follow_redirects=False)


def add_patient(c, name, age="54", gender="male"):
    tok = csrf_of(c.get("/patients/new").text)
    r = c.post("/patients/new", data=dict(name=name, age=age, gender=gender, contact="+91 90000 00000",
                                          visit_date=date.today().isoformat(), csrf=tok), follow_redirects=False)
    assert r.status_code == 303
    return re.search(r"patient=(PAT-\d{4}-\d{4})", r.headers["location"]).group(1)


def upload(c, code, data, filename="lesion.jpg", site="back"):
    tok = csrf_of(c.get(f"/analyze?patient={code}").text)
    return c.post("/analyze", data=dict(patient_code=code, body_site=site, csrf=tok),
                  files={"image": (filename, data, "image/jpeg")}, follow_redirects=False)


def test_pages_require_login(make_client):
    c = make_client()
    for url in ["/dashboard", "/patients", "/patients/new", "/analyze", "/patients/PAT-2026-0001",
                "/scans/1", "/scans/1/report.pdf", "/media/1/original", "/admin/users"]:
        r = c.get(url, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login"), url
    assert c.get("/login").status_code == 200


def test_full_workflow(client, make_client):
    from webapp.db import Patient, Scan, SessionLocal, User
    from webapp import config

    # --- registration: first user is admin, password stored as bcrypt hash
    assert register(client, "Dr. Admin", "admin").status_code == 303
    with SessionLocal() as db:
        admin = db.query(User).filter_by(username="admin").one()
        assert admin.role == "admin" and admin.password_hash.startswith("$2") and "Secret" not in admin.password_hash

    # --- CSRF and wrong password are rejected
    assert client.post("/patients/new", data=dict(name="x")).status_code == 400
    client.cookies.clear()
    r = client.post("/login", data=dict(username="admin", password="wrong", csrf=csrf_of(client.get("/login").text)))
    assert r.status_code == 401 and "Wrong username or password" in r.text
    assert login(client, "admin").status_code == 303
    assert "Dashboard" in client.get("/dashboard").text

    # --- patients get sequential IDs
    code1 = add_patient(client, "Asha Raman")
    code2 = add_patient(client, "Vikram Singh", age="", gender="unknown")
    assert code1 == f"PAT-{YEAR}-0001" and code2 == f"PAT-{YEAR}-0002"

    # --- invalid uploads are refused politely
    r = upload(client, code1, b"not an image", "x.jpg")
    assert r.status_code == 400 and "not a readable image" in r.text

    # --- real upload -> prediction, Grad-CAM, explanation, LIME, all saved
    img = SAMPLE_IMAGE.read_bytes()
    r = upload(client, code1, img)
    assert r.status_code == 303 and r.headers["location"].startswith("/scans/")
    scan_id = int(r.headers["location"].rsplit("/", 1)[1])
    with SessionLocal() as db:
        s = db.get(Scan, scan_id)
        assert s.patient.patient_code == code1 and s.doctor.username == "admin"
        assert s.predicted_class and 0 <= s.confidence <= 1 and s.confidence_level in ("High", "Medium", "Low")
        assert len(s.top3) == 3 and sum(t["prob"] for t in s.top3) <= 1.0001
        assert s.top3[0]["prob"] >= s.top3[1]["prob"] >= s.top3[2]["prob"]
        assert "The model focused on" in s.explanation and s.recommendation and s.meaning
        assert s.lime_status == "done", s.lime_status  # background task ran
        for p in (s.image_path, s.gradcam_path, s.lime_path):
            assert (config.STORAGE_DIR / p).exists()
        if s.confidence_level == "Low":
            assert "Low confidence" in s.warning

    page = client.get(f"/scans/{scan_id}").text
    for text in ["Top 3 possible classes", "Why the model gave this result", "What this means",
                 "Recommendation", "Grad-CAM", "LIME", "Not a substitute for professional medical diagnosis",
                 "Download Report"]:
        assert text in page, text
    for kind in ("original", "gradcam", "lime"):
        r = client.get(f"/media/{scan_id}/{kind}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    assert client.get(f"/scans/{scan_id}/lime").json()["status"] == "done"

    # --- scan history: a second scan for the same patient
    assert upload(client, code1, img, site="face").status_code == 303
    detail = client.get(f"/patients/{code1}").text
    assert detail.count('class="scan"') == 2

    # --- search by ID and by part of the name
    assert code1 in client.get(f"/patients?q={code1}").text
    res = client.get("/patients?q=vikram").text
    assert code2 in res and code1 not in res

    # --- PDF report
    r = client.get(f"/scans/{scan_id}/report.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF" and len(r.content) > 20_000

    # --- dashboard counts
    dash = client.get("/dashboard").text
    assert "Recent scans" in dash and code1 in dash

    # --- a second doctor cannot see the first doctor's patients; admin can see everything
    other = make_client()
    assert register(other, "Dr. Second", "doctor2").status_code == 303
    with SessionLocal() as db:
        assert db.query(User).filter_by(username="doctor2").one().role == "doctor"
    assert other.get(f"/patients/{code1}").status_code == 404
    assert other.get(f"/scans/{scan_id}").status_code == 404
    assert other.get(f"/media/{scan_id}/original").status_code == 404
    assert other.get("/admin/users").status_code == 403
    own = add_patient(other, "Meera Das")
    assert own == f"PAT-{YEAR}-0003"
    assert own in client.get("/patients").text  # admin sees all records
    assert "doctor2" in client.get("/admin/users").text

    # --- logout protects pages again
    tok = csrf_of(client.get("/dashboard").text)
    assert client.post("/logout", data=dict(csrf=tok), follow_redirects=False).status_code == 303
    assert client.get("/dashboard", follow_redirects=False).status_code == 303
    with SessionLocal() as db:
        assert db.query(Patient).count() == 3 and db.query(Scan).count() == 2


def test_patient_code_sequence():
    from webapp.db import SessionLocal, next_patient_code
    with SessionLocal() as db:
        assert next_patient_code(db, 1999) == "PAT-1999-0001"


def test_explanation_on_synthetic_lesion():
    import numpy as np
    import cv2
    from webapp.ml import xai
    img = np.full((384, 384, 3), (205, 160, 140), np.uint8)              # skin
    pts = np.array([[150, 90], [260, 120], [300, 230], [230, 310], [170, 280], [90, 200], [120, 150]])
    cv2.fillPoly(img, [pts], (70, 45, 35))                                # dark irregular lesion
    cv2.circle(img, (200, 190), 30, (25, 20, 20), -1)                     # very dark centre
    cam = np.zeros((384, 384), np.float32)
    cam[120:260, 120:260] = 1.0
    meas = xai.measurements(img, cam)
    assert meas["lesion_found"] and meas["focus_inside_lesion"] > 0.6
    text = xai.written_explanation("MEL", meas)
    assert "border" in text and "melanoma" in text and "centre" in text
    level, warn = xai.confidence_level(0.41, 0.05)
    assert level == "Low" and "Low confidence" in warn
    assert xai.confidence_level(0.93, 0.85)[0] == "High"
