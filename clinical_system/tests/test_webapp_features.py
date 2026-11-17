"""Tests for doctor feedback, analysis timings, research dashboard, CSV export, model info page,
DEMO seeding and the database migration."""
import csv
import io
import os
from pathlib import Path

import cv2
import numpy as np

from conftest import SAMPLE_IMAGE, csrf_of
from test_webapp import add_patient, login, register, upload


def set_role(username, role):
    from webapp.db import SessionLocal, User
    with SessionLocal() as db:
        db.query(User).filter_by(username=username).one().role = role
        db.commit()


def feedback(c, scan_id, **data):
    tok = csrf_of(c.get(f"/scans/{scan_id}").text)
    return c.post(f"/scans/{scan_id}/feedback", data=dict(csrf=tok, **data), follow_redirects=False)


def test_feedback_dashboard_export_modelinfo(client, make_client):
    from webapp.db import Feedback, Scan, SessionLocal
    from webapp.research import EXPORT_COLUMNS

    doc, doc2, adm = make_client(), make_client(), make_client()
    for c, name, u in ((doc, "Dr. Feature", "feat.doc"), (doc2, "Dr. Other", "feat.doc2"), (adm, "Dr. Boss", "feat.admin")):
        assert register(c, name, u).status_code == 303
    set_role("feat.doc", "doctor"); set_role("feat.doc2", "doctor"); set_role("feat.admin", "admin")

    code = add_patient(doc, "Zara Confidential")
    r = upload(doc, code, SAMPLE_IMAGE.read_bytes())
    sid = int(r.headers["location"].rsplit("/", 1)[1])

    # ---- 3. timings saved with the scan (LIME ran as a background task)
    with SessionLocal() as db:
        s = db.get(Scan, sid)
        assert s.time_predict_ms > 0 and s.time_gradcam_ms > 0 and s.time_lime_ms > 0
        assert abs(s.time_total_ms - (s.time_predict_ms + s.time_gradcam_ms + s.time_lime_ms)) < 1e-6
        predicted = s.predicted_class
    page = doc.get(f"/scans/{sid}").text
    assert "Doctor feedback" in page and "Analysis time: prediction" in page

    # ---- 1. feedback validation
    assert doc.post(f"/scans/{sid}/feedback", data=dict(agreement="agree", usefulness="4")).status_code == 400  # CSRF
    assert "choose Agree, Disagree or Unsure" in feedback(doc, sid, usefulness="4").text
    assert "pick the diagnosis" in feedback(doc, sid, agreement="disagree", usefulness="3").text
    assert "same as the AI result" in feedback(doc, sid, agreement="disagree", correct_class=predicted, usefulness="3").text
    r = feedback(doc, sid, agreement="agree", usefulness="9")
    assert r.status_code == 400 and "1 to 5 stars" in r.text

    # valid, then updated (still one row per doctor and scan)
    assert feedback(doc, sid, agreement="agree", usefulness="4").status_code == 303
    other = "MEL" if predicted != "MEL" else "NV"
    assert feedback(doc, sid, agreement="disagree", correct_class=other, usefulness="2",
                    comment="=HYPERLINK(\"x\") heat-map on ruler").status_code == 303
    with SessionLocal() as db:
        fbs = db.query(Feedback).filter_by(scan_id=sid).all()
        assert len(fbs) == 1 and fbs[0].agreement == "disagree" and fbs[0].correct_class == other
        assert fbs[0].usefulness == 2 and fbs[0].doctor.username == "feat.doc"
    page = doc.get(f"/scans/{sid}").text
    assert 'value="disagree"' in page and "checked" in page and "Update feedback" in page
    # another doctor cannot see / rate this scan; the admin can
    assert feedback(doc2, sid, agreement="agree", usefulness="5").status_code == 404 or \
        doc2.get(f"/scans/{sid}").status_code == 404
    assert adm.post(f"/scans/{sid}/feedback", data=dict(agreement="agree", usefulness="5",
                                                         csrf=csrf_of(adm.get(f"/scans/{sid}").text)),
                    follow_redirects=False).status_code == 303
    assert "Feedback from other staff" in adm.get(f"/scans/{sid}").text

    # ---- 2. research dashboard (admin only)
    assert doc.get("/admin/research").status_code == 403
    page = adm.get("/admin/research").text
    for t in ("Research dashboard", "chart.js@4.4.1", "Scans per predicted class", "Confidence level",
              "Doctor agreement", "Average explanation usefulness", "Average analysis time per scan",
              "Cases where the doctor disagreed", f"/scans/{sid}", code, "heat-map on ruler"):
        assert t in page, t
    assert "Zara Confidential" not in page                              # patient names are not shown
    assert adm.get("/admin/research?data=real").status_code == 200
    assert "Research" in adm.get("/dashboard").text and "/admin/research" not in doc.get("/dashboard").text

    # ---- 4. CSV export (admin only, no patient names)
    assert doc.get("/admin/export.csv").status_code == 403
    r = adm.get("/admin/export.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    text = r.content.decode("utf-8-sig")
    assert "Zara" not in text and "Confidential" not in text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0].keys()) == EXPORT_COLUMNS
    mine = [x for x in rows if x["scan_id"] == str(sid)]
    assert len(mine) == 2 and {x["patient_id"] for x in mine} == {code}       # one row per feedback
    assert {x["feedback_agreement"] for x in mine} == {"disagree", "agree"}
    dis = next(x for x in mine if x["feedback_agreement"] == "disagree")
    assert dis["feedback_comment"].startswith("'=")                        # formula injection neutralised
    assert dis["feedback_correct_class"] == other and dis["feedback_usefulness_1to5"] == "2"
    assert float(dis["time_prediction_ms"]) > 0 and dis["top1_class"] == predicted and dis["is_demo"] == "0"

    # ---- 5. model info page (all logged-in users) + paper figures
    anon = make_client()
    assert anon.get("/model-info", follow_redirects=False).status_code == 303
    page = doc.get("/model-info").text
    for t in ("EfficientNetV2-S", "Swin", "HAM10000", "10,015", "lesion", "85.45%", "76.03%", "Dermatofibroma",
              "Intended use", "Limitations", "outside the lesion", "darker skin", "Not a substitute"):
        assert t in page, t
    assert "Figures from the paper" not in page
    figdir = Path(os.environ["PAPER_FIGURES_DIR"])
    figdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(figdir / "roc_curves.png"), np.full((40, 60, 3), 200, np.uint8))
    page = doc.get("/model-info").text
    assert "Figures from the paper" in page and "/paper-figures/roc_curves.png" in page and "Roc curves" in page
    r = doc.get("/paper-figures/roc_curves.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert doc.get("/paper-figures/missing.png").status_code == 404
    assert doc.get("/paper-figures/..%2F..%2Fsecret.key").status_code == 404
    assert anon.get("/paper-figures/roc_curves.png", follow_redirects=False).status_code == 303


def test_seed_demo_and_remove(client, make_client):
    import seed_demo
    from webapp import config
    from webapp.db import Patient, SessionLocal

    created = seed_demo.seed(n_patients=3, max_scans=2, lime_scans=1, seed_value=1)
    assert created["patients"] == 3 and created["scans"] >= 3
    with SessionLocal() as db:
        demo = db.query(Patient).filter(Patient.is_demo.is_(True)).all()
        assert len(demo) == 3 and all(p.name.startswith("DEMO") for p in demo)
        scans = [s for p in demo for s in p.scans]
        assert all(s.time_predict_ms is not None for s in scans)
        assert sum(s.lime_status == "done" for s in scans) == 1
        assert all(f.comment.startswith("[DEMO]") for s in scans for f in s.feedback)
        files = [config.STORAGE_DIR / s.image_path for s in scans]
        codes = [p.patient_code for p in demo]

    adm = make_client()
    register(adm, "Dr. Demo Viewer", "demo.viewer"); set_role("demo.viewer", "admin")
    page = adm.get("/admin/research?data=demo").text
    assert "Includes DEMO data" in page and "DEMO" in page
    assert "Includes DEMO data" not in adm.get("/admin/research?data=real").text
    assert 'badge demo">DEMO' in adm.get("/patients").text
    real = adm.get("/admin/export.csv").content.decode("utf-8-sig")
    assert not any(c in real for c in codes)                               # DEMO left out by default
    full = list(csv.DictReader(io.StringIO(adm.get("/admin/export.csv?include_demo=1").content.decode("utf-8-sig"))))
    assert {r["patient_id"] for r in full if r["is_demo"] == "1"} == set(codes)

    seed_demo.remove_demo()
    with SessionLocal() as db:
        assert db.query(Patient).filter(Patient.is_demo.is_(True)).count() == 0
    assert not any(f.exists() for f in files)


def test_migration_adds_columns(tmp_path):
    from sqlalchemy import create_engine, inspect, text
    from webapp.db import migrate
    eng = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with eng.begin() as con:  # a database from before this update
        con.execute(text("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)"))
        con.execute(text("CREATE TABLE scans (id INTEGER PRIMARY KEY, confidence FLOAT)"))
        con.execute(text("INSERT INTO scans (id, confidence) VALUES (1, 0.9)"))
    assert sorted(migrate(eng)) == ["patients.is_demo", "scans.time_gradcam_ms", "scans.time_lime_ms",
                                    "scans.time_predict_ms"]
    assert migrate(eng) == []                                              # idempotent
    cols = {c["name"] for c in inspect(eng).get_columns("scans")}
    assert {"time_predict_ms", "time_gradcam_ms", "time_lime_ms"} <= cols
    with eng.connect() as con:
        assert con.execute(text("SELECT confidence FROM scans WHERE id = 1")).scalar() == 0.9   # data kept
