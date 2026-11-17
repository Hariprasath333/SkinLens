"""New logo/icons, all-class probabilities, "why this result" measurements, photo quality check,
disease information + ABCDE pages, and the PDFs that include them."""
import base64
import json
import re

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import ROOT, SAMPLE_IMAGE, csrf_of

STATIC = ROOT / "webapp" / "static"


def rgb_sample():
    return cv2.cvtColor(cv2.imread(str(SAMPLE_IMAGE)), cv2.COLOR_BGR2RGB)


def jpg(rgb, q=92):
    return cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def new_patient(c, name):
    loc = c.post("/patients/new", data=dict(name=name, csrf=csrf_of(c.get("/patients/new").text)),
                 follow_redirects=False).headers["location"]
    return loc.split("=")[1]


def upload(c, code, data):
    r = c.post("/analyze", data=dict(patient_code=code, body_site="back", csrf=csrf_of(c.get(f"/analyze?patient={code}").text)),
               files={"image": ("a.jpg", data, "image/jpeg")}, follow_redirects=False)
    assert r.status_code == 303, r.text[:500]
    return int(r.headers["location"].rsplit("/", 1)[1])


@pytest.fixture(scope="module")
def doc():
    """A logged-in doctor (not the first user, so the existing tests' admin stays admin)."""
    from fastapi.testclient import TestClient
    from webapp.main import app
    from test_webapp import register
    c = TestClient(app)
    assert register(c, "Dr. Explain", "explain.doc").status_code == 303
    return c


# ----------------------------------------------------------------------------- logo and icons
def test_new_logo_everywhere(client):
    for f, size in [("favicon-16.png", 16), ("favicon-32.png", 32), ("favicon-48.png", 48), ("apple-touch-icon.png", 180),
                    ("icon-192.png", 192), ("icon-512.png", 512), ("icon-maskable-512.png", 512)]:
        assert Image.open(STATIC / "icons" / f).size == (size, size), f
    assert set(Image.open(STATIC / "icons" / "favicon.ico").info["sizes"]) >= {(16, 16), (32, 32), (48, 48)}
    r = client.get("/favicon.ico")
    assert r.status_code == 200 and r.headers["content-type"] == "image/x-icon"
    logo = (ROOT / "branding" / "skinlens_logo.svg").read_text(encoding="utf-8")
    assert (STATIC / "img" / "logo.svg").read_text(encoding="utf-8") == logo
    assert "#1F5FAF" in logo and "#102A47" in logo                   # ring and spot of the new logo
    for page in ("/", "/login", "/check", "/offline"):
        html = client.get(page).text
        assert ("/static/img/logo.svg" in html or "/static/img/app-icon.svg" in html), page
    head = client.get("/login").text
    for t in ("favicon.ico", "favicon-16.png", "favicon-48.png", 'sizes="180x180"'):
        assert t in head
    sw = client.get("/sw.js").text
    assert 'VERSION = "skinlens-shell-v2"' in sw and "/static/icons/favicon.ico" in sw
    # maskable icon: background reaches the corners, logo inside the 80% safe zone
    m = np.asarray(Image.open(STATIC / "icons" / "icon-maskable-512.png").convert("RGB")).astype(int)
    assert abs(m[2, 2] - m[256, 20]).sum() < 10                       # corners = background colour
    dark = np.argwhere(np.abs(m - m[2, 2]).sum(2) > 60)
    r_max = np.hypot(dark[:, 0] - 256, dark[:, 1] - 256).max()
    assert r_max < 0.4 * 512


# ----------------------------------------------------------------------------- probabilities
def test_round_to_total_and_rows():
    from webapp.insight import prob_rows, round_to_total
    assert sum(round_to_total([1 / 3, 1 / 3, 1 / 3], 100)) == 100
    probs = {"AK": .012, "BCC": .021, "BKL": .087, "DF": .004, "MEL": .624, "NV": .233, "VASC": .019}
    rows = prob_rows(probs)
    assert [r["code"] for r in rows] == sorted(probs, key=lambda c: -probs[c])
    assert sum(r["pct"] for r in rows) == 100 and abs(sum(r["pct1"] for r in rows) - 100) < 1e-9
    assert rows[0]["pct1"] == 62.4 and rows[0]["abbr"] == "mel" and rows[0]["group"] == "attention"
    assert {r["code"]: r["group"] for r in rows}["NV"] == "harmless"
    assert rows[1]["plain"].startswith("Mole") and rows[1]["medical"] == "Melanocytic nevus"


def test_patient_mode_bars_reasons_details(client):
    from fastapi.testclient import TestClient
    from webapp import patient
    from webapp.main import app
    patient.limiter.reset()
    c = TestClient(app)
    page = c.get("/check").text
    assert "About you (optional)" in page and "helps the app a little" not in page
    assert "<details" not in page.split("About you (optional)")[0][-400:]         # not collapsed
    assert 'id="age"' in page and 'id="body_site"' in page and "quality-box" in page
    r = c.post("/check", data=dict(csrf=csrf_of(page), age="50", body_site="back"),
               files={"image_gallery": ("p.jpg", SAMPLE_IMAGE.read_bytes(), "image/jpeg")})
    html = r.text
    from webapp.ml.predictor import get_predictor
    k = len(get_predictor().classes)                                   # 7 (SkinLens) or 8 (test model)
    pcts = [int(v) for v in re.findall(r'<span class="pbar-pct">(\d+)%</span>', html)]
    assert len(pcts) == k and sum(pcts) == 100 and pcts == sorted(pcts, reverse=True)
    assert html.count("What is this?") == k and "/learn/" in html and "/learn/abcde" in html
    assert "These percentages show how sure the app is, not your actual risk." in html
    assert "Why the app gave this result" in html and "Show full details" in html
    reasons = re.search(r'<ul class="reasons">(.*?)</ul>', html, re.S).group(1)
    assert 3 <= reasons.count("<li>") <= 5
    assert "Red areas influenced the result most" in html and "Highlighted areas pushed the app towards this result" in html
    assert "Clinical summary" in html and "Grad-CAM explains the CNN branch only; LIME covers the whole model." in html
    assert "Age</dt><dd>50" in html and "Sex</dt><dd>not given" in html
    # the top bar matches the predicted class named in the plain result
    top = re.search(r'<span class="pbar-med">([^<]+)</span>', html).group(1)
    assert top.lower() in html.split("This looks most like", 1)[1].split("</p>", 1)[0].lower()


def test_patient_lime_token(client):
    from fastapi.testclient import TestClient
    from webapp import patient
    from webapp.main import app
    patient.limiter.reset(); patient.lime_limiter.reset()
    c = TestClient(app)
    html = c.post("/check", data=dict(csrf=csrf_of(c.get("/check").text)),
                  files={"image": ("p.jpg", SAMPLE_IMAGE.read_bytes(), "image/jpeg")}).text
    token = re.search(r'data-token="([^"]+)"', html).group(1)
    photo = re.search(r'<img src="(data:image/jpeg;base64,[^"]+)" alt="Your photo">', html).group(1)
    assert c.post("/check/lime", json=dict(photo=photo[:-8] + "AAAAAAA=", token=token)).status_code == 403
    r = c.post("/check/lime", json=dict(photo=photo, token=token))
    assert r.status_code == 200 and r.json()["image"].startswith("data:image/jpeg;base64,")
    assert "no-store" in r.headers["cache-control"]
    assert c.post("/check/lime", json=dict(photo=photo, token=token)).status_code == 403     # used once only
    assert c.post("/check/lime", json=dict(photo=photo, token="forged")).status_code == 403


def test_doctor_result_and_pdf(doc):
    from webapp.db import Scan, SessionLocal
    from webapp.insight import prob_rows
    sid = upload(doc, new_patient(doc, "Explain test"), SAMPLE_IMAGE.read_bytes())
    with SessionLocal() as db:
        s = db.get(Scan, sid)
    from webapp.ml.predictor import get_predictor
    k = len(get_predictor().classes)
    probs = s.probs
    assert len(probs) == k and abs(sum(probs.values()) - 1) < 1e-6
    assert max(probs, key=probs.get) == s.predicted_class and abs(probs[s.predicted_class] - s.confidence) < 1e-9
    assert s.features["lesion_found"] in (True, False) and "issues" in s.quality
    html = doc.get(f"/scans/{s.id}").text
    pcts = [float(v) for v in re.findall(r'<span class="pbar-pct">([\d.]+)%</span>', html)]
    assert len(pcts) == k and abs(sum(pcts) - 100) < 1e-6
    assert pcts[0] == prob_rows(probs)[0]["pct1"]
    for t in (f"Probabilities of all {k} classes", "Clinical summary", "Prediction:", "Asymmetry score",
              "Border irregularity", "Number of colours", "Lesion-to-skin contrast", "Estimated lesion area share",
              "Heat-map energy inside lesion", "Top 3 possible classes", "Patient information used by the model",
              "Grad-CAM explains the CNN branch only", "(mel)", "(nv)"):
        assert t in html, t
    pdf = doc.get(f"/scans/{s.id}/report.pdf").content
    assert pdf[:4] == b"%PDF"
    import pypdfium2 as pdfium
    text = "".join(pg.get_textpage().get_text_range() for pg in pdfium.PdfDocument(pdf))
    for t in ("Probabilities of all classes", "Clinical summary", "Asymmetry score", "Photo quality check",
              "Grad-CAM explains the CNN branch only"):
        assert t in text, t


def test_patient_summary_pdf_has_table():
    from fastapi.testclient import TestClient
    from webapp import patient
    from webapp.main import app
    import pypdfium2 as pdfium
    patient.limiter.reset()
    c = TestClient(app)
    html = c.post("/check", data=dict(csrf=csrf_of(c.get("/check").text)),
                  files={"image": ("p.jpg", SAMPLE_IMAGE.read_bytes(), "image/jpeg")}).text
    pdf = base64.b64decode(re.search(r'data:application/pdf;base64,([A-Za-z0-9+/=]+)', html).group(1))
    text = "".join(pg.get_textpage().get_text_range() for pg in pdfium.PdfDocument(pdf))
    for t in ("How well the photo matches each type", "Why the app gave this result", "not your actual risk",
              "Clinical summary", "Melanocytic nevus"):
        assert t in text, t


# ----------------------------------------------------------------------------- measurements
def test_features_on_synthetic_spots():
    from webapp.ml.features import ASYM_HIGH, BORDER_HIGH, lesion_features
    from webapp.insight import patient_reasons
    bg = np.full((400, 400, 3), (225, 180, 160), np.uint8)
    round_spot = bg.copy(); cv2.circle(round_spot, (200, 200), 80, (150, 100, 70), -1)
    f = lesion_features(round_spot, np.ones((384, 384), np.float32))
    assert f["lesion_found"] and f["asymmetry"] < 0.1 and f["border_irregularity"] < BORDER_HIGH and f["n_colours"] == 1
    assert f["darker_by"] > 0 and 0.08 < f["area_share"] < 0.18
    L = bg.copy(); cv2.rectangle(L, (120, 120), (170, 290), (130, 85, 60), -1); cv2.rectangle(L, (120, 240), (290, 290), (130, 85, 60), -1)
    g = lesion_features(L)
    assert g["asymmetry"] >= ASYM_HIGH and g["border_irregularity"] >= BORDER_HIGH
    multi = bg.copy(); cv2.circle(multi, (200, 200), 90, (120, 80, 55), -1)
    cv2.circle(multi, (170, 190), 32, (35, 25, 25), -1); cv2.circle(multi, (235, 215), 26, (110, 120, 150), -1)
    assert lesion_features(multi)["n_colours"] >= 3
    reasons = patient_reasons(g)
    assert any("do not look the same" in r for r in reasons) and any("uneven" in r for r in reasons)
    # heat-map focus: energy only outside the spot -> honest warning
    cam = np.ones((384, 384), np.float32); cy, cx = np.mgrid[0:384, 0:384]
    cam[np.hypot(cx - 192, cy - 192) < 110] = 0
    h = lesion_features(round_spot, cam)
    assert h["heat_focus"] == "outside"
    assert "less reliable" in " ".join(patient_reasons(h))


# ----------------------------------------------------------------------------- photo quality
def test_quality_check_server():
    from webapp.quality import check
    rgb = rgb_sample()
    assert check(rgb)["ok"]
    codes = lambda img: {i["code"] for i in check(img)["issues"]}  # noqa: E731
    assert "blurry" in codes(cv2.GaussianBlur(rgb, (0, 0), 3))
    assert "dark" in codes((rgb * 0.2).astype(np.uint8))
    assert "bright" in codes(np.clip(rgb.astype(int) + 120, 0, 255).astype(np.uint8))
    assert "small" in codes(cv2.resize(rgb, (250, 188)))
    sky = np.zeros((450, 600, 3), np.uint8); sky[:] = (60, 130, 220)
    assert codes(sky) == {"not_skin"}
    assert "off_centre" in codes(np.roll(rgb, (150, 220), (0, 1)))


def test_quality_saved_and_never_blocks(doc):
    from webapp.db import Scan, SessionLocal
    from webapp import patient
    dark = jpg((rgb_sample() * 0.2).astype(np.uint8))
    sid = upload(doc, new_patient(doc, "Quality test"), dark)        # analysed anyway (redirect to result)
    with SessionLocal() as db:
        q = db.get(Scan, sid).quality
    assert not q["ok"] and "dark" in {i["code"] for i in q["issues"]}
    html = doc.get(f"/scans/{sid}").text
    assert "Photo quality check" in html and "The photo is too dark" in html and "Retake photo" in html
    # patient mode: warned on the result, still analysed
    from fastapi.testclient import TestClient
    from webapp.main import app
    patient.limiter.reset()
    c = TestClient(app)
    blurred = jpg(cv2.GaussianBlur(rgb_sample(), (0, 0), 4))
    res = c.post("/check", data=dict(csrf=csrf_of(c.get("/check").text)), files={"image": ("b.jpg", blurred, "image/jpeg")})
    assert res.status_code == 200 and "The photo looks blurry" in res.text and "Your result" in res.text


def test_old_scans_without_new_columns_still_render(doc):
    from webapp.db import Scan, SessionLocal
    sid = upload(doc, new_patient(doc, "Old scan"), SAMPLE_IMAGE.read_bytes())
    with SessionLocal() as db:
        s = db.get(Scan, sid)
        db.delete(s.extra)                                            # as a scan made before v1.5
        db.commit()
    html = doc.get(f"/scans/{sid}").text
    assert "only the top 3 are shown" in html and "Top 3 possible classes" in html
    assert doc.get(f"/scans/{sid}/report.pdf").content[:4] == b"%PDF"


# ----------------------------------------------------------------------------- disease information
def test_disease_and_abcde_pages():
    from fastapi.testclient import TestClient
    from webapp.diseases import DISEASES
    from webapp.main import app
    c = TestClient(app)                                                # no login needed
    for code in ("mel", "bcc", "ak", "nv", "bkl", "df", "vasc"):
        html = c.get(f"/learn/{code}").text
        d = DISEASES[code.upper()]
        for t in (d["name"], "What it is", "Common signs", "How serious is it?", "When to see a doctor",
                  "General information, not a diagnosis"):
            assert t in html, (code, t)
    assert c.get("/learn/xyz").status_code == 404
    idx = c.get("/learn").text
    assert idx.count('class="learn-item') >= 7
    ab = c.get("/learn/abcde").text
    for t in ("Asymmetry", "Border", "Colour", "Diameter", "Evolving", "6 mm", "<svg"):
        assert t in ab, t
    assert ab.count("<svg") >= 10
    assert 'href="/learn/abcde"' in c.get("/").text


def test_quality_check_skin_coloured_striped_surface():
    """v1.5.2: wood-like surfaces (skin-coloured but striped and very saturated) are 'not skin'; skin is not."""
    from webapp.quality import check
    from webapp.validation.runner import non_skin_images
    woods = [img for name, img in non_skin_images(20, 31337) if "wood" in name]   # not the T6 seed
    assert woods and all("not_skin" in {i["code"] for i in check(w)["issues"]} for w in woods)
    m = check(rgb_sample())
    assert m["ok"] and m["metrics"]["stripes"] < 0.6
