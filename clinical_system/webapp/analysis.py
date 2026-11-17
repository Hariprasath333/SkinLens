"""Runs one analysis end to end and stores everything in the database."""
from __future__ import annotations

import json
import logging
import time
import uuid

import cv2
import numpy as np

from . import config
from .db import Patient, Scan, ScanExtra, SessionLocal
from . import quality as photo_quality
from .ml import xai
from .ml.features import lesion_features
from .ml.predictor import class_info, get_predictor

log = logging.getLogger("skinlens.analysis")


class BadImage(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise BadImage("The file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise BadImage(f"The image is larger than {config.MAX_UPLOAD_MB} MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise BadImage("This file is not a readable image (use JPG or PNG).")
    if min(img.shape[:2]) < 64:
        raise BadImage("The image is too small (minimum 64 x 64 pixels).")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save(rgb: np.ndarray, name: str) -> str:
    path = config.STORAGE_DIR / name
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return name  # stored relative to STORAGE_DIR


def compute(rgb: np.ndarray, age=None, sex=None, site=None, explain: bool = True) -> dict:
    """Prediction + Grad-CAM + written explanation, in memory only (nothing is saved).
    Used by doctor mode (which then saves the scan), by patient mode (which saves nothing) and by the
    Validation Lab. explain=False skips Grad-CAM and the image measurements only (bulk validation runs);
    preprocessing, metadata, prior correction and confidence levels are exactly the same code."""
    h, w = rgb.shape[:2]
    t_pre = time.perf_counter()
    quality = photo_quality.check(rgb) if explain else None   # on the photo as received (never blocks)
    if max(h, w) > 1600:  # the model uses 224-384 px anyway
        s = 1600 / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    time_pre_ms = 1000 * (time.perf_counter() - t_pre)
    pred = get_predictor()
    cam = None
    with pred.session(age=age, sex=sex, site=site):
        # timed inside the session, so waiting for another user's analysis is not counted
        t0 = time.perf_counter()
        res = pred.predict(rgb)
        t1 = time.perf_counter()
        code = res["top_class"]
        if explain:
            cam = xai.gradcam(pred, rgb, pred.classes.index(code))
        t2 = time.perf_counter()
    meas = xai.measurements(rgb, cam) if explain else {}
    feats = lesion_features(rgb, cam) if explain else {}
    ranked = res["ranked"]
    p1, p2 = ranked[0][1], ranked[1][1] if len(ranked) > 1 else 0.0
    level, warning = xai.confidence_level(p1, p1 - p2)
    status, meaning, rec = xai.meaning_and_recommendation(code, level)
    return dict(rgb=rgb, cam=cam, meas=meas, features=feats, quality=quality, probs=res["probs"],
                raw_probs=res["raw_probs"], uses_metadata=pred.uses_metadata, code=code, info=class_info(code),
                ranked=ranked, confidence=p1, level=level, warning=warning, status=status, meaning=meaning,
                recommendation=rec, model_name=pred.display_name, time_pre_ms=time_pre_ms,
                time_predict_ms=1000 * (t1 - t0), time_gradcam_ms=1000 * (t2 - t1) if explain else None)


def analyze(patient: Patient, doctor_id: int, data: bytes, body_site: str) -> Scan:
    """Doctor mode: analyse and save the scan (images on disk, result in the database)."""
    r = compute(decode_image(data), age=patient.age, sex=patient.gender, site=body_site)
    key = uuid.uuid4().hex
    code, info = r["code"], r["info"]
    scan = Scan(
        patient_id=patient.id, doctor_id=doctor_id, body_site=body_site or "unknown",
        image_path=_save(r["rgb"], f"{key}_original.jpg"),
        gradcam_path=_save(xai.overlay(r["rgb"], r["cam"]), f"{key}_gradcam.jpg"),
        predicted_class=code, predicted_name=info["name"], risk=info["risk"],
        confidence=r["confidence"], confidence_level=r["level"], warning=r["warning"],
        top3_json=json.dumps(xai.top3(r["ranked"])),
        explanation=xai.written_explanation(code, r["meas"]),
        meaning=f"{r['status']}. {r['meaning']}", recommendation=r["recommendation"],
        model_name=r["model_name"], lime_status="pending",
        time_predict_ms=r["time_predict_ms"], time_gradcam_ms=r["time_gradcam_ms"],
        extra=ScanExtra(probs_json=json.dumps(r["probs"]), features_json=json.dumps(r["features"]),
                        quality_json=json.dumps(r["quality"])),
    )
    with SessionLocal() as db:
        db.add(scan)
        db.commit()
        db.refresh(scan)
    return scan


def run_lime(scan_id: int) -> None:
    """Second explanation method; slower, so it runs after the result page is shown."""
    with SessionLocal() as db:
        scan = db.get(Scan, scan_id)
        if scan is None:
            return
        patient = scan.patient
        try:
            bgr = cv2.imread(str(config.STORAGE_DIR / scan.image_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pred = get_predictor()
            with pred.session(age=patient.age, sex=patient.gender, site=scan.body_site):
                t0 = time.perf_counter()
                out = xai.lime(pred, rgb, pred.classes.index(scan.predicted_class),
                               config.LIME_SAMPLES, config.LIME_SEGMENTS)
                scan.time_lime_ms = 1000 * (time.perf_counter() - t0)
            scan.lime_path = _save(out["image"], scan.image_path.replace("_original", "_lime"))
            share = float(out["mask"].mean())
            scan.lime_summary = (
                f"LIME (a second, independent method) hid different parts of the image "
                f"{config.LIME_SAMPLES} times and measured how the answer changed. The highlighted "
                f"regions ({100 * share:.0f}% of the image) are the ones that most supported the answer "
                f"'{scan.predicted_name}'. If they agree with the Grad-CAM heat-map, the explanation is "
                f"more trustworthy.")
            scan.lime_status = "done"
        except Exception:  # noqa: BLE001
            log.exception("LIME failed for scan %s", scan_id)
            scan.lime_status = "failed"
        db.commit()
