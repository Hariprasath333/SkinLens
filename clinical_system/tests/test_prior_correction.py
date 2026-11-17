"""Prior correction: formula, settings file, class re-ordering, confidence levels, and the full
analysis path (result, top-3, level, explanation, database, PDF) with a 7-class model."""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import SAMPLE_IMAGE

HAM_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CODES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]


def settings():
    from webapp.ml.predictor import load_model_settings
    return load_model_settings()


def manual(p, alpha, lp, eps=1e-8):
    s = np.log(np.asarray(p, float) + eps) + alpha * np.asarray(lp, float)
    e = np.exp(s - s.max())
    return e / e.sum(), s


def test_settings_file_and_formula():
    from webapp.ml.predictor import PriorCorrection
    pc = settings()["prior_correction"]
    assert pc["enabled"] and pc["alpha"] == 0.3 and pc["class_order"] == HAM_ORDER and len(pc["log_prior"]) == 7
    corr, note = PriorCorrection.for_classes(settings(), CODES)
    assert note == "applied" and corr.alpha == 0.3
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.dirichlet(np.ones(7) * 0.5)
        got = corr.apply(p)
        want, score = manual(p, 0.3, pc["log_prior"])
        assert np.allclose(got, want, atol=1e-12) and abs(got.sum() - 1) < 1e-12
        assert got.argmax() == score.argmax()                 # prediction = argmax(score)


def test_reordering_and_not_applicable():
    from webapp.ml.predictor import PriorCorrection
    pc = settings()["prior_correction"]
    shuffled = ["NV", "MEL", "AK", "VASC", "DF", "BKL", "BCC"]
    corr, _ = PriorCorrection.for_classes(settings(), shuffled)
    lp = dict(zip(CODES, pc["log_prior"]))
    assert np.allclose(corr.log_prior, [lp[c] for c in shuffled])
    # an 8-class model (with SCC) does not match the 7 prior values -> not applied, with a reason
    none, note = PriorCorrection.for_classes(settings(), CODES + ["SCC"])
    assert none is None and "NOT applied" in note
    off = json.loads(json.dumps(settings()))
    off["prior_correction"]["enabled"] = False
    assert PriorCorrection.for_classes(off, CODES)[0] is None


def test_confidence_levels():
    from webapp.ml.xai import confidence_level
    assert confidence_level(0.81, 0.7)[0] == "High"
    assert confidence_level(0.80, 0.7)[0] == "Medium"          # High is strictly above 80 %
    assert confidence_level(0.50, 0.3)[0] == "Medium"
    lv, warn = confidence_level(0.4999, 0.1)
    assert lv == "Low" and "Low confidence" in warn
    assert "close" in confidence_level(0.55, 0.05)[1]          # close second -> extra warning only


class SevenClass:
    """Tiny 7-class model with the SkinLens class order, plus the prior correction from the settings."""

    @staticmethod
    def make(temperature=1.064):
        from webapp.ml.predictor import Member, Predictor, PriorCorrection
        torch.manual_seed(3)
        net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 16, 3, stride=2, padding=1),
                            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 7))
        with torch.no_grad():  # make 'nv' a close second so the correction can change the answer
            net[-1].bias.copy_(torch.tensor([0.0, 0.2, 0.1, 0.9, 0.0, 0.6, 0.3]))
        net.eval()
        p = Predictor()
        p.members = [Member(net, 64, temperature, net[2])]
        p.classes = CODES
        p.name = "seven-class test model"
        p.prior, p.prior_note = PriorCorrection.for_classes(settings(), CODES)
        return p


def test_predict_uses_corrected_probs_without_temperature():
    pred = SevenClass.make()
    rgb = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)
    res = pred.predict(rgb)
    raw = pred.predict_batch([rgb], calibrated=False)[0]
    want, score = manual(raw, 0.3, settings()["prior_correction"]["log_prior"])
    assert np.allclose([res["probs"][c] for c in CODES], want)
    assert np.allclose([res["raw_probs"][c] for c in CODES], raw)       # raw = plain softmax (T not applied)
    assert res["top_class"] == CODES[int(score.argmax())] and res["prior_corrected"]
    assert [c for c, _ in res["ranked"]] == [CODES[i] for i in np.argsort(-want)]
    assert pred.display_name.endswith("+ prior correction (alpha=0.3)")


def test_full_analysis_path_with_correction(client, monkeypatch):
    """result, top-3, confidence level, explanation, database row and PDF all use corrected values;
    Grad-CAM and LIME explain the predicted (corrected) class."""
    import webapp.analysis as analysis
    from webapp.db import Patient, Scan, SessionLocal, User
    from webapp.main import hash_pw
    from webapp.ml import xai
    from webapp.report import build_pdf

    pred = SevenClass.make()
    monkeypatch.setattr(analysis, "get_predictor", lambda: pred)
    called = {}
    real_gradcam = xai.gradcam
    monkeypatch.setattr(xai, "gradcam", lambda p, rgb, idx, **k: called.setdefault("cam", idx) is not None
                        and real_gradcam(p, rgb, idx, **k))
    real_lime = xai.lime
    monkeypatch.setattr(xai, "lime", lambda p, rgb, idx, *a, **k: called.setdefault("lime", idx) is not None
                        and real_lime(p, rgb, idx, *a, **k))

    with SessionLocal() as db:
        u = User(username="prior.tester", full_name="Dr. Prior", password_hash=hash_pw("x" * 10))
        db.add(u); db.commit()
        pt = Patient(patient_code="PAT-1990-0001", name="Prior Test", age=50, gender="female", created_by_id=u.id)
        db.add(pt); db.commit()
        db.refresh(pt)
    try:
        _check_analysis(analysis, pred, pt, u, called, manual_fn=manual)
    finally:  # leave the shared test database as it was (other tests rely on "first user = admin")
        with SessionLocal() as db:
            db.delete(db.get(Patient, pt.id))
            db.delete(db.get(User, u.id))
            db.commit()


def _check_analysis(analysis, pred, pt, u, called, manual_fn):
    from webapp.db import Scan, SessionLocal
    from webapp.ml import xai
    from webapp.report import build_pdf
    manual = manual_fn
    data = SAMPLE_IMAGE.read_bytes()
    scan = analysis.analyze(pt, u.id, data, "back")
    analysis.run_lime(scan.id)

    rgb = analysis.decode_image(data)
    with pred.session():
        raw = pred.predict_batch([rgb], calibrated=False)[0]
    want, score = manual(raw, 0.3, settings()["prior_correction"]["log_prior"])
    top = int(score.argmax())
    with SessionLocal() as db:
        s = db.get(Scan, scan.id)
        assert s.predicted_class == CODES[top]
        assert s.confidence == pytest.approx(want[top], abs=1e-6)
        assert [t["code"] for t in s.top3] == [CODES[i] for i in np.argsort(-want)[:3]]
        assert [t["prob"] for t in s.top3] == pytest.approx(sorted(want, reverse=True)[:3], abs=1e-6)
        assert s.confidence_level == xai.confidence_level(want[top], want[top] - sorted(want)[-2])[0]
        assert s.predicted_name.lower() in s.explanation.lower()      # explanation names the corrected class
        assert "prior correction (alpha=0.3)" in s.model_name
        assert called["cam"] == top and called["lime"] == top          # explain the predicted class
        assert s.lime_status == "done"
        pdf = build_pdf(s)
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000
