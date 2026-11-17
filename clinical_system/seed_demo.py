"""
Fill the web app with clearly marked DEMO data (for demo screenshots of the dashboards).

    python seed_demo.py                       # 15 DEMO patients, 1-2 scans each, simulated feedback
    python seed_demo.py --patients 20 --lime 3
    python seed_demo.py --images C:\\path\\to\\dermoscopy_images     # more varied predictions
    python seed_demo.py --remove              # delete ALL demo data again

What it creates
  * 2 demo doctor accounts (cannot log in: random password) that "made" the scans
  * DEMO patients: name "DEMO Patient NN", is_demo flag set, a DEMO badge everywhere in the app
  * scans made by the REAL analysis pipeline (real model prediction, Grad-CAM, timings) on
    augmented copies of the images in references/test_images (or --images)
  * SIMULATED doctor feedback (comments start with "[DEMO]")

The demo feedback is made up. Never report it as results: the research dashboard has a
"Real scans only" filter, and the CSV export leaves DEMO rows out by default.
Run it with the server stopped or running - both work. Run from the clinical_system folder.
"""
from __future__ import annotations

import argparse
import random
import secrets
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from webapp import config  # noqa: E402
from webapp.analysis import analyze, run_lime  # noqa: E402
from webapp.db import Feedback, Patient, Scan, SessionLocal, User, init_db, next_patient_code  # noqa: E402
from webapp.main import BODY_SITES, hash_pw  # noqa: E402
from webapp.ml.predictor import CLASSES, get_predictor  # noqa: E402

DEMO_DOCTORS = [("demo.doctor1", "DEMO Dr. A. Mehta"), ("demo.doctor2", "DEMO Dr. R. Iyer")]
COMMENTS = {
    "agree": ["Typical dermoscopic pattern.", "Agree; heat-map on the lesion.", "Consistent with my assessment.", ""],
    "disagree": ["Heat-map focused on the border/ruler, not the lesion.", "Would biopsy - atypical network.",
                 "Clinically looks like a seborrhoeic keratosis.", ""],
    "unsure": ["Image quality too low to decide.", "Needs dermoscopy follow-up.", ""],
}
AGREE_P = {"High": 0.85, "Medium": 0.65, "Low": 0.40}   # simulated: doctors agree more with confident results


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A different-looking copy: flip / rotate / zoom / light and colour change."""
    if rng.random() < 0.5:
        img = cv2.flip(img, rng.choice([0, 1]))
    img = np.ascontiguousarray(np.rot90(img, rng.randint(0, 3)))
    h, w = img.shape[:2]
    z = rng.uniform(0.75, 1.0)
    ch, cw = int(h * z), int(w * z)
    y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
    img = img[y0:y0 + ch, x0:x0 + cw]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.25), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.8, 1.15), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def demo_users(db) -> list[User]:
    users = []
    for username, name in DEMO_DOCTORS:
        u = db.query(User).filter_by(username=username).one_or_none()
        if u is None:
            u = User(username=username, full_name=name, role="doctor",
                     password_hash=hash_pw(secrets.token_urlsafe(32)))  # nobody knows it: cannot log in
            db.add(u)
            db.commit()
        users.append(u)
    return users


def remove_demo() -> None:
    init_db()
    with SessionLocal() as db:
        pats = db.query(Patient).filter(Patient.is_demo.is_(True)).all()
        files = [p for pt in pats for s in pt.scans for p in (s.image_path, s.gradcam_path, s.lime_path) if p]
        n_scans = sum(len(p.scans) for p in pats)
        for p in pats:
            db.delete(p)  # scans and their feedback are deleted with the patient
        db.commit()
        for f in files:
            (config.STORAGE_DIR / f).unlink(missing_ok=True)
        removed_users = 0
        for username, _ in DEMO_DOCTORS:
            u = db.query(User).filter_by(username=username).one_or_none()
            if u and not db.query(Patient).filter_by(created_by_id=u.id).count() and \
                    not db.query(Scan).filter_by(doctor_id=u.id).count():
                db.query(Feedback).filter_by(doctor_id=u.id).delete()
                db.delete(u)
                removed_users += 1
        db.commit()
    print(f"Removed {len(pats)} DEMO patients, {n_scans} scans, {len(files)} image files, {removed_users} demo accounts.")


def seed(n_patients: int = 15, max_scans: int = 2, image_dir: Path | None = None, lime_scans: int = 2,
         seed_value: int = 7) -> dict:
    init_db()
    rng = random.Random(seed_value)
    folders = [image_dir] if image_dir else [HERE.parent / "references" / "test_images"]
    pool = [p for d in folders if d and d.exists() for p in sorted(d.rglob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not pool:
        sys.exit(f"No images found in {folders}. Pass --images <folder with dermoscopy images>.")
    pred = get_predictor()
    print(f"Model: {pred.name} | {len(pool)} source image(s) | {n_patients} DEMO patients")
    created = dict(patients=0, scans=0, feedback=0)
    with SessionLocal() as db:
        docs = demo_users(db)
        start = len(db.query(Patient).filter(Patient.is_demo.is_(True)).all())
        scans_done = 0
        for i in range(n_patients):
            doc = rng.choice(docs)
            visit = date.today() - timedelta(days=rng.randint(0, 60))
            p = Patient(patient_code=next_patient_code(db), name=f"DEMO Patient {start + i + 1:02d}",
                        age=rng.randint(22, 86), gender=rng.choice(["female", "male"]),
                        contact="demo data - not a real person", visit_date=visit,
                        created_by_id=doc.id, is_demo=True)
            db.add(p)
            db.commit()
            created["patients"] += 1
            for k in range(rng.randint(1, max_scans)):
                img = augment(cv2.imread(str(rng.choice(pool))), rng)
                ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                scan = analyze(p, doc.id, enc.tobytes(), rng.choice(BODY_SITES[1:]))
                when = datetime.combine(visit, datetime.min.time()) + timedelta(hours=rng.randint(9, 17),
                                                                                 minutes=rng.randint(0, 59), days=30 * k)
                s = db.get(Scan, scan.id)
                s.created_at = min(when, datetime.now())
                if scans_done < lime_scans:
                    db.commit()
                    run_lime(s.id)
                    db.refresh(s)
                else:
                    s.lime_status = "skipped"
                    s.lime_summary = "LIME was not computed for this DEMO scan (seed_demo.py --lime)."
                # simulated clinician feedback for ~85 % of scans
                if rng.random() < 0.85:
                    ag = "agree" if rng.random() < AGREE_P[s.confidence_level] else rng.choice(["disagree", "unsure"])
                    corr = rng.choice([c for c in CLASSES if c != s.predicted_class]) if ag == "disagree" else None
                    stars = rng.choices([2, 3, 4, 5], weights=[1, 3, 5, 4])[0] if ag == "agree" else \
                        rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 2])[0]
                    comment = rng.choice(COMMENTS[ag])
                    db.add(Feedback(scan_id=s.id, doctor_id=rng.choice(docs).id, agreement=ag, correct_class=corr,
                                    usefulness=stars, comment=f"[DEMO] {comment}".strip(),
                                    created_at=s.created_at, updated_at=s.created_at))
                    created["feedback"] += 1
                db.commit()
                created["scans"] += 1
                scans_done += 1
                print(f"  {p.patient_code} {p.name}: {s.predicted_class} {100 * s.confidence:.0f}% "
                      f"({s.confidence_level}) in {s.time_predict_ms + s.time_gradcam_ms:.0f} ms", flush=True)
    print(f"Created {created['patients']} DEMO patients, {created['scans']} scans, {created['feedback']} simulated "
          "feedback entries. Remove them later with: python seed_demo.py --remove")
    return created


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, default=15)
    ap.add_argument("--max-scans", type=int, default=2, help="1..N scans per patient")
    ap.add_argument("--images", type=Path, default=None, help="folder with lesion images to use")
    ap.add_argument("--lime", type=int, default=2, help="compute LIME for this many scans (slow on CPU)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--remove", action="store_true", help="delete all DEMO data")
    a = ap.parse_args()
    if a.remove:
        remove_demo()
    else:
        seed(a.patients, a.max_scans, a.images, a.lime, a.seed)


if __name__ == "__main__":
    main()
