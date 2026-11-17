"""Numbers for the research dashboard and the anonymised CSV export (admin only)."""
from __future__ import annotations

import csv
import io
from statistics import mean

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .db import Feedback, Patient, Scan
from .ml.predictor import CLASSES, class_info

DATA_FILTERS = {"all": "All scans", "real": "Real scans only (no DEMO)", "demo": "DEMO scans only"}


def _scans(db: Session, data: str) -> list[Scan]:
    q = (select(Scan).join(Patient).options(selectinload(Scan.feedback).selectinload(Feedback.doctor),
                                           selectinload(Scan.patient), selectinload(Scan.doctor))
         .order_by(Scan.created_at))
    if data == "real":
        q = q.where(Patient.is_demo.is_not(True))
    elif data == "demo":
        q = q.where(Patient.is_demo.is_(True))
    return list(db.scalars(q).unique())


def _avg(values) -> float | None:
    v = [x for x in values if x is not None]
    return mean(v) if v else None


def dashboard_stats(db: Session, data: str = "all") -> dict:
    scans = _scans(db, data)
    fb = [f for s in scans for f in s.feedback]
    order = CLASSES + sorted({s.predicted_class for s in scans} - set(CLASSES))
    per_class = {c: sum(s.predicted_class == c for s in scans) for c in order}
    levels = {lv: sum(s.confidence_level == lv for s in scans) for lv in ("High", "Medium", "Low")}
    agree = {a: sum(f.agreement == a for f in fb) for a in ("agree", "disagree", "unsure")}
    # agreement per confidence level: does the doctor agree more often when the AI is confident?
    by_level = {}
    for lv in ("High", "Medium", "Low"):
        f_lv = [f for s in scans if s.confidence_level == lv for f in s.feedback]
        by_level[lv] = (100 * sum(f.agreement == "agree" for f in f_lv) / len(f_lv)) if f_lv else None
    disagreements = [
        dict(scan=s, fb=f, ai=class_info(s.predicted_class), doctor_dx=class_info(f.correct_class)
             if f.correct_class else None)
        for s in reversed(scans) for f in s.feedback if f.agreement == "disagree"]
    return dict(
        total_scans=len(scans),
        demo_scans=sum(bool(s.patient.is_demo) for s in scans),
        per_class=[dict(code=c, name=class_info(c)["name"], n=n) for c, n in per_class.items()],
        levels=levels,
        feedback_count=len(fb),
        scans_with_feedback=sum(bool(s.feedback) for s in scans),
        agree=agree,
        agree_rate=100 * agree["agree"] / len(fb) if fb else None,
        usefulness_avg=_avg(f.usefulness for f in fb),
        usefulness_dist={k: sum(f.usefulness == k for f in fb) for k in range(1, 6)},
        agree_by_level=by_level,
        time_predict=_avg(s.time_predict_ms for s in scans),
        time_gradcam=_avg(s.time_gradcam_ms for s in scans),
        time_lime=_avg(s.time_lime_ms for s in scans),
        # what the doctor waits for: prediction + Grad-CAM (LIME follows in the background)
        time_to_result=_avg(s.time_predict_ms + s.time_gradcam_ms for s in scans
                            if s.time_predict_ms is not None and s.time_gradcam_ms is not None),
        time_total=_avg(s.time_total_ms for s in scans if s.time_lime_ms is not None),
        timed_scans=sum(s.time_predict_ms is not None for s in scans),
        disagreements=disagreements,
    )


EXPORT_COLUMNS = [
    "scan_id", "patient_id", "is_demo", "scan_datetime", "body_site", "model",
    "predicted_class", "predicted_name", "risk", "confidence", "confidence_level",
    "top1_class", "top1_prob", "top2_class", "top2_prob", "top3_class", "top3_prob",
    "time_prediction_ms", "time_gradcam_ms", "time_lime_ms", "time_total_ms", "lime_status",
    "scanned_by_doctor_id", "feedback_doctor_id", "feedback_agreement", "feedback_correct_class",
    "feedback_usefulness_1to5", "feedback_comment", "feedback_datetime",
]


def _cell(v):
    """Numbers as numbers; text guarded against spreadsheet formula injection."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if v < 10 else f"{v:.1f}"
    s = str(v).replace("\r", " ").replace("\n", " ")
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t") else s


def export_csv(db: Session, include_demo: bool = False) -> str:
    """One row per scan and doctor feedback (a scan without feedback gives one row with empty
    feedback columns). The patient's name, contact and age are NOT exported - only the patient ID.
    Doctors appear as numeric IDs."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(EXPORT_COLUMNS)
    for s in _scans(db, "all" if include_demo else "real"):
        top = s.top3 + [{}] * (3 - len(s.top3))
        base = [s.id, s.patient.patient_code, int(bool(s.patient.is_demo)), s.created_at.isoformat(timespec="seconds"),
                s.body_site, s.model_name, s.predicted_class, s.predicted_name, s.risk, s.confidence,
                s.confidence_level,
                top[0].get("code"), top[0].get("prob"), top[1].get("code"), top[1].get("prob"),
                top[2].get("code"), top[2].get("prob"),
                s.time_predict_ms, s.time_gradcam_ms, s.time_lime_ms, s.time_total_ms, s.lime_status, s.doctor_id]
        for f in (s.feedback or [None]):
            fbv = ([f.doctor_id, f.agreement, f.correct_class, f.usefulness, f.comment,
                    f.updated_at.isoformat(timespec="seconds")] if f else [None] * 6)
            w.writerow([_cell(v) for v in base + fbv])
    return out.getvalue()
