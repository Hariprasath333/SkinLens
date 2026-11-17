"""
Runs the Validation Lab tests in the background (a separate process, so the web app stays responsive)
and can resume an interrupted run: every per-image result is appended to a CSV as soon as it is known.

    # from the clinical_system folder
    .venv\\Scripts\\python.exe -m webapp.validation.runner --new --tests T1,T2,T3,T4,T5,T6,T7          # T1 = 300 sample
    .venv\\Scripts\\python.exe -m webapp.validation.runner --new --tests T1 --t1 full                  # all 2,014 images
    .venv\\Scripts\\python.exe -m webapp.validation.runner --run validation_results/2026-11-15_1530   # resume

Nothing is written to the app database: test images go through webapp.analysis.compute() in memory, the
load test (T5) uses patient mode (which stores nothing) on a separate server with a throw-away data folder.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import config
from . import criteria as C
from . import data as D

ALL_TESTS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


# ============================================================================ run folder + state
class Run:
    def __init__(self, folder: Path):
        self.dir = Path(folder)
        self.charts = self.dir / "charts"
        self.charts.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # files
    @property
    def info(self) -> dict:
        return json.loads((self.dir / "run.json").read_text(encoding="utf-8"))

    def state(self) -> dict:
        f = self.dir / "state.json"
        for attempt in range(50):
            try:
                return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            except (PermissionError, json.JSONDecodeError):
                time.sleep(0.1 * (attempt + 1))
        return {}

    def _write_state(self, st: dict) -> None:
        tmp = self.dir / f"state.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
        for attempt in range(50):          # Windows: the file may be open in another process for a moment
            try:
                os.replace(tmp, self.dir / "state.json")
                return
            except PermissionError:
                time.sleep(0.1 * (attempt + 1))
        raise PermissionError("state.json stayed locked")

    def update(self, test: str, **kw) -> None:
        with self._lock:
            st = self.state()
            st.setdefault(test, {}).update(kw, updated=datetime.now().isoformat(timespec="seconds"))
            self._write_state(st)

    def progress(self, test: str, done: int, total: int, note: str = "") -> None:
        self.update(test, done=done, total=total, note=note)

    def save_json(self, name: str, obj) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")

    def load_json(self, name: str):
        f = self.dir / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def log(self, msg: str) -> None:
        with open(self.dir / "runner.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%H:%M:%S} {msg}\n")
        print(msg, flush=True)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def new_run(tests: list[str], t1_mode: str = "sample", folder: Path | None = None) -> Run:
    """Creates validation_results/<date-time>/ with run.json (criteria frozen BEFORE anything runs)."""
    from .identity import identity
    root = config.VALIDATION_RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    folder = folder or root / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run = Run(folder)
    info = dict(created=datetime.now().isoformat(timespec="seconds"), tests=tests, t1_mode=t1_mode,
                criteria=C.frozen(), identity=identity())
    run.save_json("run.json", info)
    run._write_state({t: dict(status="pending", done=0, total=0) for t in tests})
    return run


# ============================================================================ the app's inference
def load_rgb(path: str) -> np.ndarray:
    """Exactly what the upload does: the file's bytes through webapp.analysis.decode_image()."""
    from ..analysis import decode_image
    return decode_image(Path(path).read_bytes())


def predict(rgb: np.ndarray, row, explain: bool = False) -> dict:
    """THE app's inference function (webapp.analysis.compute) with the patient's metadata."""
    from ..analysis import compute
    age, sex, site = D.meta(row)
    return compute(rgb, age=age, sex=sex, site=site, explain=explain)


class PredCache:
    """predictions.csv: one row per (image, condition). Shared by T1, T2 and T3 of the same run."""
    def __init__(self, run: Run):
        self.f = run.dir / "predictions.csv"
        self.rows: dict[tuple[str, str], dict] = {}
        self.fields = ["image_id", "condition", "true", "pred", "confidence", "level", "ms"] + \
            [f"p_{c}" for c in D.CODES] + [f"raw_{c}" for c in D.CODES]
        if self.f.exists():
            with open(self.f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.rows[(r["image_id"], r["condition"])] = r

    def get(self, image_id: str, cond: str = "original"):
        return self.rows.get((image_id, cond))

    def add(self, row, cond: str, r: dict, ms: float) -> dict:
        rec = dict(image_id=row["image_id"], condition=cond, true=row["code"], pred=r["code"],
                   confidence=f"{r['confidence']:.6f}", level=r["level"], ms=f"{ms:.1f}",
                   **{f"p_{c}": f"{r['probs'][c]:.8f}" for c in D.CODES},
                   **{f"raw_{c}": f"{r['raw_probs'][c]:.8f}" for c in D.CODES})
        new = not self.f.exists()
        with open(self.f, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields)
            if new:
                w.writeheader()
            w.writerow(rec)
        self.rows[(row["image_id"], cond)] = rec
        return rec


def ensure_predictions(run: Run, cache: PredCache, rows, test: str, cond: str = "original", transform=None,
                       offset: int = 0, total: int | None = None) -> None:
    total = total or len(rows)
    done = offset
    for _, row in rows.iterrows():
        if cache.get(row["image_id"], cond) is None:
            rgb = load_rgb(row["path"])
            if transform is not None:
                rgb = transform(rgb)
            t = time.perf_counter()
            r = predict(rgb, row)
            cache.add(row, cond, r, 1000 * (time.perf_counter() - t))
        done += 1
        if done % 5 == 0 or done == total:
            run.progress(test, done, total, f"{cond}: {done}/{total}")


def probs_of(recs: list[dict], prefix: str = "p_") -> np.ndarray:
    return np.array([[float(r[f"{prefix}{c}"]) for c in D.CODES] for r in recs])


# ============================================================================ T1 clinical accuracy
def t1(run: Run, cache: PredCache) -> dict:
    from .metrics import classification
    s = run.info["criteria"]["settings"]
    m = D.manifest()
    mode = run.info.get("t1_mode", "sample")
    rows = m if mode == "full" else D.stratified_sample(m, s["t1_sample_size"], s["seed"])
    rows[["image_id", "dx", "age", "sex", "localization"]].to_csv(run.dir / "T1_images.csv", index=False)
    ensure_predictions(run, cache, rows, "T1")
    recs = [cache.get(i) for i in rows["image_id"]]
    res = classification([r["true"] for r in recs], probs_of(recs), D.CODES, [r["level"] for r in recs],
                         bins=s["ece_bins"])
    ref = run.info["criteria"]["reference"]
    tol = run.info["criteria"]["criteria"]["T1"]["tolerance"]
    res.update(mode=mode, reference=ref, tolerance=tol,
               mean_ms=float(np.mean([float(r["ms"]) for r in recs])))
    if mode == "full":
        diff = res["accuracy"] - ref["accuracy"]
        res["verdict"] = "PASS" if abs(diff) <= tol + 1e-12 else "FAIL"
        res["verdict_text"] = f"accuracy {100 * res['accuracy']:.2f}% vs reference 85.80% (difference {100 * diff:+.2f} points)"
    else:
        lo, hi = res["ci95"]["accuracy"]
        res["verdict"] = "REPORTED"
        res["verdict_text"] = (f"sample accuracy {100 * res['accuracy']:.2f}% (95% CI {100 * lo:.1f}–{100 * hi:.1f}%); "
                               f"the reference 85.80% is {'inside' if lo <= ref['accuracy'] <= hi else 'outside'} "
                               "the interval. Pass/fail needs the full 2,014-image run.")
    run.save_json("T1_metrics.json", res)
    from . import charts
    charts.t1_charts(run, res)
    return res


# ============================================================================ T2 integration agreement
def t2(run: Run, cache: PredCache) -> dict:
    from ..ml.predictor import PriorCorrection, load_model_settings
    m = D.manifest()
    nb = D.notebook_predictions()
    ensure_predictions(run, cache, m, "T2")
    names = [str(c) for c in nb["class_names"]]
    order = [names.index(h) for h in D.HAM_ORDER]                      # notebook columns -> app class order
    nb_ids = [str(i) for i in nb["image_id"]]
    pos = {i: k for k, i in enumerate(nb_ids)}
    matched = [i for i in m["image_id"] if i in pos]
    recs = [cache.get(i) for i in matched]
    idx = np.array([pos[i] for i in matched])
    app_c, app_raw = probs_of(recs), probs_of(recs, "raw_")
    nb_c = nb["probs_corrected"][idx][:, order]
    nb_raw = nb["probs_raw"][idx][:, order]
    nb_pred = [D.HAM_TO_CODE[names[k]] for k in nb["y_pred_corrected"][idx]]
    y_true = [D.HAM_TO_CODE[names[k]] for k in nb["y_true"][idx]]
    app_pred = [r["pred"] for r in recs]
    same = np.array([a == b for a, b in zip(app_pred, nb_pred)])
    diff_c = np.abs(app_c - nb_c)
    # is the notebook's corrected output what the app's prior correction gives for the notebook's raw output?
    settings = load_model_settings()
    pc, _ = PriorCorrection.for_classes(settings, D.CODES)
    nb_recorrected = pc.apply(nb_raw)
    self_consistency = float(np.abs(nb_recorrected - nb_c).max())
    app_lp = dict(zip([D.HAM_TO_CODE.get(str(c).lower(), str(c).upper()) for c in settings["prior_correction"]["class_order"]],
                      settings["prior_correction"]["log_prior"]))
    nb_lp = dict(zip([D.HAM_TO_CODE[n] for n in names], [float(v) for v in nb["log_prior"]]))
    lp_diff = max(abs(app_lp[c] - nb_lp[c]) for c in D.CODES)
    acc_app = float(np.mean([a == t for a, t in zip(app_pred, y_true)]))
    acc_nb = float(np.mean([a == t for a, t in zip(nb_pred, y_true)]))
    import hashlib
    npz_file = D.data_dir() / "test_predictions.npz"
    out = dict(
        npz_sha256=hashlib.sha256(npz_file.read_bytes()).hexdigest(),
        npz_modified=datetime.fromtimestamp(npz_file.stat().st_mtime).isoformat(timespec="seconds"),
        run_at=datetime.now().isoformat(timespec="seconds"),
        n_matched=len(matched), n_manifest=len(m), n_npz=len(nb_ids), agreement=float(same.mean()),
        n_disagree=int((~same).sum()), max_prob_diff=float(diff_c.max()), mean_prob_diff=float(diff_c.mean()),
        max_raw_prob_diff=float(np.abs(app_raw - nb_raw).max()),
        npz_alpha=float(nb["alpha"]), app_alpha=float(settings["prior_correction"]["alpha"]), log_prior_max_diff=lp_diff,
        npz_correction_self_consistency=self_consistency,
        accuracy_app=acc_app, accuracy_npz=acc_nb,
        npz_mean_top_prob=float(nb["probs_corrected"].max(1).mean()), app_mean_top_prob=float(app_c.max(1).mean()),
        npz_pred_counts={c: int(sum(p == c for p in nb_pred)) for c in D.CODES},
        app_pred_counts={c: int(sum(p == c for p in app_pred)) for c in D.CODES},
        compared=("For every test image (matched by image_id) the app's predicted class (argmax of its prior-"
                  "corrected probabilities, alpha 0.3) was compared with y_pred_corrected in the npz, and the "
                  "app's corrected probabilities with probs_corrected. The npz already contains corrected values; "
                  "re-applying the app's prior correction to probs_raw was used only to check that the file is "
                  "internally consistent."),
    )
    thr = run.info["criteria"]["criteria"]["T2"]["min_agreement"]
    out["verdict"] = "PASS" if out["agreement"] >= thr else "FAIL"
    out["verdict_text"] = (f"{100 * out['agreement']:.2f}% identical class ({out['n_disagree']} of {len(matched)} differ); "
                           f"largest probability difference {out['max_prob_diff']:.4f}")
    with open(run.dir / "T2_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "true", "app_pred", "npz_pred", "same", "max_abs_prob_diff"])
        for i, t, a, b, d in zip(matched, y_true, app_pred, nb_pred, diff_c.max(1)):
            w.writerow([i, t, a, b, int(a == b), f"{d:.6f}"])
    run.save_json("T2_metrics.json", out)
    from . import charts
    charts.t2_chart(run, out)
    return out


# ============================================================================ T3 robustness
def conditions(s: dict) -> dict:
    def blur(x):
        return cv2.GaussianBlur(x, (0, 0), s["t3_blur_sigma"])

    def scale(f):
        return lambda x: np.clip(x.astype(np.float32) * f, 0, 255).astype(np.uint8)

    def jpeg(x):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, s["t3_jpeg_quality"]])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def phone(x):
        h, w = x.shape[:2]
        k = s["t3_phone_side"] / max(h, w)
        big = cv2.resize(x, (round(w * k), round(h * k)), interpolation=cv2.INTER_LINEAR if k > 1 else cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(big, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        back = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return cv2.resize(back, (w, h), interpolation=cv2.INTER_AREA if k > 1 else cv2.INTER_LINEAR)

    def crop(x):
        h, w = x.shape[:2]
        f = s["t3_crop"]
        return np.ascontiguousarray(x[h - round(h * f):, w - round(w * f):])   # 90% window, shifted off-centre

    return {"blur": blur, "darker": scale(s["t3_dark_factor"]), "brighter": scale(s["t3_bright_factor"]),
            "rotate90": lambda x: np.ascontiguousarray(np.rot90(x, 1)),
            "rotate180": lambda x: np.ascontiguousarray(np.rot90(x, 2)), "jpeg50": jpeg, "phone_resize": phone,
            "crop90": crop}


def t3(run: Run, cache: PredCache) -> dict:
    s = run.info["criteria"]["settings"]
    m = D.manifest()
    rows = D.stratified_sample(m, s["t3_images"], s["seed"] + 3)
    rows[["image_id", "dx"]].to_csv(run.dir / "T3_images.csv", index=False)
    conds = conditions(s)
    total = len(rows) * (1 + len(conds))
    ensure_predictions(run, cache, rows, "T3", total=total)
    for k, (name, fn) in enumerate(conds.items(), start=1):
        ensure_predictions(run, cache, rows, "T3", name, fn, offset=k * len(rows), total=total)
    orig = {i: cache.get(i)["pred"] for i in rows["image_id"]}
    truth = dict(zip(rows["image_id"], rows["code"]))
    per = {"original": dict(accuracy=float(np.mean([orig[i] == truth[i] for i in orig])), stability=1.0)}
    for name in conds:
        preds = {i: cache.get(i, name)["pred"] for i in rows["image_id"]}
        per[name] = dict(accuracy=float(np.mean([preds[i] == truth[i] for i in preds])),
                         stability=float(np.mean([preds[i] == orig[i] for i in preds])))
    avg = float(np.mean([per[n]["stability"] for n in conds]))
    thr = run.info["criteria"]["criteria"]["T3"]["min_stability"]
    out = dict(n=len(rows), conditions=per, average_stability=avg, verdict="PASS" if avg >= thr else "FAIL",
               verdict_text=f"average stability {100 * avg:.1f}% over {len(conds)} conditions",
               settings={k: v for k, v in s.items() if k.startswith("t3_")})
    run.save_json("T3_metrics.json", out)
    from . import charts
    charts.t3_chart(run, out)
    return out


# ============================================================================ T4 speed
def _transient_scan(r: dict, row, files: dict):
    """A Scan object that is NEVER added to a database session: only used to time the real PDF builder."""
    from ..db import Patient, Scan, ScanExtra, User
    from ..ml import xai
    age, sex, _site = D.meta(row)
    pat = Patient(patient_code="VALIDATION", name="Validation image (not a patient)", age=None if age is None else int(age),
                  gender=sex or "unknown", contact="")
    doc = User(username="validation", full_name="Validation Lab", password_hash="-", role="doctor")
    return Scan(id=0, patient=pat, doctor=doc, created_at=datetime.now(), body_site=_site,
                image_path=files["original"], gradcam_path=files["gradcam"], lime_path=files.get("lime", ""),
                lime_status="done", lime_summary="", predicted_class=r["code"], predicted_name=r["info"]["name"],
                risk=r["info"]["risk"], confidence=r["confidence"], confidence_level=r["level"], warning=r["warning"],
                top3_json=json.dumps(xai.top3(r["ranked"])), explanation=xai.written_explanation(r["code"], r["meas"]),
                meaning=f"{r['status']}. {r['meaning']}", recommendation=r["recommendation"], model_name=r["model_name"],
                extra=ScanExtra(probs_json=json.dumps(r["probs"]), features_json=json.dumps(r["features"]),
                                quality_json=json.dumps(r["quality"])))


def t4(run: Run, cache: PredCache) -> dict:
    from ..analysis import decode_image
    from ..ml import xai
    from ..ml.predictor import get_predictor
    from ..report import build_pdf
    s = run.info["criteria"]["settings"]
    m = D.manifest()
    rows = D.stratified_sample(m, s["t4_images"], s["seed"] + 4)
    f = run.dir / "T4_timings.csv"
    done = {}
    if f.exists():
        with open(f, newline="", encoding="utf-8") as fh:
            done = {r["image_id"]: r for r in csv.DictReader(fh)}
    fields = ["image_id", "decode_ms", "preprocess_ms", "predict_ms", "gradcam_ms", "explain_ms", "lime_ms", "pdf_ms",
              "result_ms", "full_ms"]
    tmp = Path(tempfile.mkdtemp(prefix="skinlens_t4_"))
    pred = get_predictor()
    try:
        for k, (_, row) in enumerate(rows.iterrows(), start=1):
            if row["image_id"] not in done:
                data = Path(row["path"]).read_bytes()
                t0 = time.perf_counter()
                rgb = decode_image(data)
                t1 = time.perf_counter()
                age, sex, site = D.meta(row)
                from ..analysis import compute
                r = compute(rgb, age=age, sex=sex, site=site, explain=True)
                t2 = time.perf_counter()
                files = {}
                for name, img in (("original", r["rgb"]), ("gradcam", xai.overlay(r["rgb"], r["cam"]))):
                    p = tmp / f"{row['image_id']}_{name}.jpg"
                    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    files[name] = str(p)
                t3_ = time.perf_counter()
                with pred.session(age=age, sex=sex, site=site):
                    lime = xai.lime(pred, r["rgb"], pred.classes.index(r["code"]), config.LIME_SAMPLES, config.LIME_SEGMENTS)
                t4_ = time.perf_counter()
                p = tmp / f"{row['image_id']}_lime.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(lime["image"], cv2.COLOR_RGB2BGR))
                files["lime"] = str(p)
                t5 = time.perf_counter()
                pdf = build_pdf(_transient_scan(r, row, files))
                t6 = time.perf_counter()
                assert pdf[:4] == b"%PDF"
                explain_ms = 1000 * (t2 - t1) - r["time_pre_ms"] - r["time_predict_ms"] - r["time_gradcam_ms"]
                rec = dict(image_id=row["image_id"], decode_ms=1000 * (t1 - t0),
                           preprocess_ms=1000 * (t1 - t0) + r["time_pre_ms"], predict_ms=r["time_predict_ms"],
                           gradcam_ms=r["time_gradcam_ms"], explain_ms=explain_ms, lime_ms=1000 * (t4_ - t3_),
                           pdf_ms=1000 * (t6 - t5), result_ms=1000 * (t3_ - t0), full_ms=1000 * (t6 - t0))
                new = not f.exists()
                with open(f, "a", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields)
                    if new:
                        w.writeheader()
                    w.writerow({k2: (f"{v:.1f}" if isinstance(v, float) else v) for k2, v in rec.items()})
                done[row["image_id"]] = {k2: str(v) for k2, v in rec.items()}
            run.progress("T4", k, len(rows), f"image {k}/{len(rows)} (LIME takes about a minute each)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    recs = [done[i] for i in rows["image_id"]]

    def stats(key):
        v = np.array([float(r[key]) for r in recs]) / 1000
        return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()))
    st = {k: stats(k) for k in fields[1:]}
    pg = np.array([float(r["predict_ms"]) + float(r["gradcam_ms"]) for r in recs]) / 1000
    st["predict_plus_gradcam"] = dict(mean=float(pg.mean()), median=float(np.median(pg)), p95=float(np.percentile(pg, 95)),
                                      max=float(pg.max()))
    thr = run.info["criteria"]["criteria"]["T4"]["max_p95_s"]
    p95 = st["predict_plus_gradcam"]["p95"]
    out = dict(n=len(recs), stats=st, lime_samples=config.LIME_SAMPLES, verdict="PASS" if p95 < thr else "FAIL",
               verdict_text=f"95th percentile of prediction + Grad-CAM = {p95:.2f} s (limit {thr:.0f} s)")
    run.save_json("T4_metrics.json", out)
    from . import charts
    charts.t4_chart(run, out)
    return out


# ============================================================================ T5 reliability and load (real HTTP API)
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def t5(run: Run, cache: PredCache) -> dict:
    import httpx
    import psutil
    s = run.info["criteria"]["settings"]
    m = D.manifest()
    rows = D.stratified_sample(m, s["t5_sequential"], s["seed"] + 5)
    data_tmp = Path(tempfile.mkdtemp(prefix="skinlens_t5_"))
    port = _free_port()
    env = {k: v for k, v in os.environ.items()}
    env.update(APP_DATA_DIR=str(data_tmp), PATIENT_RATE_LIMIT="100000", APP_SECRET_KEY="validation-lab-load-test")
    env.pop("DATABASE_URL", None)
    log = open(run.dir / "T5_server.log", "w", encoding="utf-8")
    t_start = time.perf_counter()
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "webapp.main:app", "--host", "127.0.0.1", "--port", str(port)],
                           cwd=str(config.SYSTEM_DIR), env=env, stdout=log, stderr=subprocess.STDOUT)
    not_throttled(srv.pid)
    base = f"http://127.0.0.1:{port}"
    mem, stop = [], threading.Event()
    results = []
    try:
        for _ in range(240):
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if srv.poll() is not None:
                raise RuntimeError("test server exited during start-up (see T5_server.log)")
            time.sleep(1)
        startup_s = time.perf_counter() - t_start
        proc = psutil.Process(srv.pid)

        def sample():
            while not stop.is_set():
                try:
                    rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
                    mem.append((time.perf_counter() - t_start, rss / 2**20))
                except psutil.Error:
                    pass
                stop.wait(0.5)
        threading.Thread(target=sample, daemon=True).start()

        def one(client: httpx.Client, row, phase: str, user: int):
            t = time.perf_counter()
            rec = dict(phase=phase, user=user, image_id=row["image_id"], status=0, ok=False, error="")
            try:
                page = client.get(base + "/check", timeout=60)
                import re
                tok = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
                age, sex, site = D.meta(row)
                r = client.post(base + "/check", timeout=600,
                                data=dict(csrf=tok, age="" if age is None else str(int(age)), sex=sex or "", body_site=site),
                                files={"image": (row["image_id"] + ".jpg", Path(row["path"]).read_bytes(), "image/jpeg")})
                rec["status"] = r.status_code
                rec["ok"] = r.status_code == 200 and "Your result" in r.text and "pbar-pct" in r.text
                if not rec["ok"]:
                    rec["error"] = f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:200]
            rec["latency_s"] = time.perf_counter() - t
            rec["t"] = time.perf_counter() - t_start
            return rec

        with httpx.Client() as c:
            for k, (_, row) in enumerate(rows.iterrows(), start=1):
                results.append(one(c, row, "sequential", 0))
                run.progress("T5", k, s["t5_sequential"] + s["t5_users"] * s["t5_requests_per_user"],
                             f"sequential request {k}/{s['t5_sequential']}")
        seq_lat = [r["latency_s"] for r in results if r["ok"]]
        service = float(np.median(seq_lat)) if seq_lat else float("nan")
        barrier = threading.Barrier(s["t5_users"])
        conc: list[dict] = []
        lock = threading.Lock()
        pool = rows.sample(s["t5_users"] * s["t5_requests_per_user"], random_state=s["seed"], replace=True)

        def user(u: int):
            with httpx.Client() as cu:
                barrier.wait()
                for j in range(s["t5_requests_per_user"]):
                    rec = one(cu, pool.iloc[u * s["t5_requests_per_user"] + j], "concurrent", u + 1)
                    with lock:
                        conc.append(rec)
                        run.progress("T5", s["t5_sequential"] + len(conc),
                                     s["t5_sequential"] + s["t5_users"] * s["t5_requests_per_user"],
                                     f"{s['t5_users']} users at once: {len(conc)} requests done")
        ths = [threading.Thread(target=user, args=(u,)) for u in range(s["t5_users"])]
        [t.start() for t in ths]
        [t.join() for t in ths]
        results += conc
        crashed = srv.poll() is not None
        # patient mode must not have stored anything on the test server
        db_rows = None
        try:
            import sqlite3
            con = sqlite3.connect(data_tmp / "app.db")
            db_rows = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("patients", "scans")}
            con.close()
        except Exception:  # noqa: BLE001
            pass
        stored_files = sum(1 for p in (data_tmp / "storage").rglob("*") if p.is_file()) if (data_tmp / "storage").exists() else 0
    finally:
        stop.set()
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()
        shutil.rmtree(data_tmp, ignore_errors=True)
    with open(run.dir / "T5_requests.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["phase", "user", "image_id", "status", "ok", "error", "latency_s", "t"])
        w.writeheader()
        for r in results:
            w.writerow({**r, "latency_s": f"{r['latency_s']:.3f}", "t": f"{r['t']:.1f}"})

    def lat(phase):
        v = np.array([r["latency_s"] for r in results if r["phase"] == phase])
        return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                    max=float(v.max())) if len(v) else {}
    conc_lat = np.array([r["latency_s"] for r in results if r["phase"] == "concurrent"])
    failures = [r for r in results if not r["ok"]]
    mem_v = [v for _, v in mem]
    out = dict(sequential=lat("sequential"), concurrent=lat("concurrent"), users=s["t5_users"],
               requests=len(results), failed=len(failures), errors=sorted({r["error"] for r in failures}),
               crashed=crashed, startup_s=startup_s, service_time_s=service,
               queue_wait_s=dict(mean=float(max(0.0, conc_lat.mean() - service)) if len(conc_lat) else None,
                                 p95=float(max(0.0, np.percentile(conc_lat, 95) - service)) if len(conc_lat) else None,
                                 max=float(max(0.0, conc_lat.max() - service)) if len(conc_lat) else None),
               memory_mb=dict(start=mem_v[0] if mem_v else None, peak=max(mem_v) if mem_v else None,
                              end=mem_v[-1] if mem_v else None),
               memory_trace=[(round(t, 1), round(v, 1)) for t, v in mem[::4]],
               stored_after_test=dict(database_rows=db_rows, files=stored_files),
               endpoint="POST /check (patient mode: full analysis, Grad-CAM, explanation and PDF; stores nothing)")
    ok = out["failed"] == 0 and not crashed
    out["verdict"] = "PASS" if ok else "FAIL"
    out["verdict_text"] = (f"{out['requests']} requests, {out['failed']} failed, server "
                           f"{'crashed' if crashed else 'stayed up'}; peak memory {out['memory_mb']['peak']:.0f} MB")
    run.save_json("T5_metrics.json", out)
    from . import charts
    charts.t5_chart(run, out, results)
    return out


# ============================================================================ T6 photo quality check
def non_skin_images(n: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Generated pictures that are clearly not skin (sky, leaves, a text page, wood, a blue wall). They are
    synthetic, not photographs; documented as such in the report."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = ["sky", "leaves", "text", "wood", "wall"][i % 5]
        h, w = 450, 600
        if kind == "sky":
            g = np.linspace(0, 1, h)[:, None, None]
            img = (np.array([90, 150, 235]) * (1 - g) + np.array([200, 225, 250]) * g) * np.ones((h, w, 3))
            img += rng.normal(0, 4, (h, w, 3))
        elif kind == "leaves":
            base = rng.normal(0, 1, (h // 8, w // 8))
            base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
            img = np.dstack([40 + 25 * base, 110 + 45 * base, 35 + 20 * base]) + rng.normal(0, 6, (h, w, 3))
        elif kind == "text":
            img = np.full((h, w, 3), 245, np.uint8)
            for y in range(40, h - 20, 28):
                cv2.putText(img, "".join(rng.choice(list("abcdefghij klmnop")) for _ in range(30)), (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        elif kind == "wood":
            x = np.linspace(0, 30, w)[None, :] + rng.normal(0, 0.3, (h, 1))
            ring = (np.sin(x + 3 * np.sin(np.linspace(0, 4, h))[:, None]) + 1) / 2
            img = np.dstack([110 + 60 * ring, 70 + 45 * ring, 30 + 25 * ring])
        else:
            img = np.ones((h, w, 3)) * np.array([70, 120, 180]) + rng.normal(0, 10, (h, w, 3))
        out.append((f"nonskin_{kind}_{i:02d}", np.clip(img, 0, 255).astype(np.uint8)))
    return out


def t6(run: Run, cache: PredCache) -> dict:
    from ..quality import check
    s = run.info["criteria"]["settings"]
    m = D.manifest()
    rows = D.stratified_sample(m, s["t6_images"], s["seed"] + 6)
    groups = {"blurred": "blurry", "dark": "dark", "tiny": "small", "non_skin": "not_skin"}
    recs = []
    k = 0
    total = len(rows) * 4 + s["t6_images"]
    for _, row in rows.iterrows():
        rgb = load_rgb(row["path"])
        h, w = rgb.shape[:2]
        tiny_k = s["t6_tiny_side"] / max(h, w)
        variants = {"original": rgb,
                    "blurred": cv2.GaussianBlur(rgb, (0, 0), s["t6_blur_sigma"]),
                    "dark": (rgb.astype(np.float32) * s["t6_dark_factor"]).astype(np.uint8),
                    "tiny": cv2.resize(rgb, (round(w * tiny_k), round(h * tiny_k)), interpolation=cv2.INTER_AREA)}
        for g, img in variants.items():
            q = check(img)
            codes = [i["code"] for i in q["issues"]]
            recs.append(dict(group=g, image=row["image_id"], codes=" ".join(codes),
                             warned=bool(codes), correct=(groups.get(g) in codes) if g != "original" else not codes))
            k += 1
        run.progress("T6", k, total)
    for name, img in non_skin_images(s["t6_images"], s["seed"] + 66):
        q = check(img)
        codes = [i["code"] for i in q["issues"]]
        recs.append(dict(group="non_skin", image=name, codes=" ".join(codes), warned=bool(codes), correct="not_skin" in codes))
        k += 1
        run.progress("T6", k, total)
    with open(run.dir / "T6_quality.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "image", "codes", "warned", "correct"])
        w.writeheader()
        w.writerows(recs)
    per = {}
    for g in ["blurred", "dark", "tiny", "non_skin", "original"]:
        rr = [r for r in recs if r["group"] == g]
        per[g] = dict(n=len(rr), correct=sum(r["correct"] for r in rr), warned=sum(r["warned"] for r in rr),
                      rate=float(np.mean([r["correct"] for r in rr])))
    thr = run.info["criteria"]["criteria"]["T6"]["min_detection"]
    ok = all(per[g]["rate"] >= thr for g in groups)
    out = dict(groups=per, expected_warning=groups, verdict="PASS" if ok else "FAIL",
               verdict_text=", ".join(f"{g} {per[g]['correct']}/{per[g]['n']}" for g in groups) +
               f"; unchanged test images without a warning {per['original']['correct']}/{per['original']['n']}",
               settings={k2: v for k2, v in s.items() if k2.startswith("t6_")})
    run.save_json("T6_metrics.json", out)
    from . import charts
    charts.t6_chart(run, out)
    return out


# ============================================================================ T7 automated test suite
def t7(run: Run, cache: PredCache) -> dict:
    import xml.etree.ElementTree as ET
    junit = run.dir / "T7_junit.xml"
    env = {k: v for k, v in os.environ.items()
           if k not in ("APP_DATA_DIR", "APP_STORAGE_DIR", "MODEL_KIND", "DATABASE_URL", "PATIENT_RATE_LIMIT",
                        "APP_SECRET_KEY", "VALIDATION_RESULTS_DIR")}
    run.progress("T7", 0, 1, "running the automated test suite (about 15 minutes)")
    t = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                       cwd=str(config.SYSTEM_DIR), env=env, capture_output=True, text=True)
    (run.dir / "T7_pytest.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    files: dict[str, dict] = {}
    failed_names = []
    for case in suite.iter("testcase"):
        parts = case.get("classname", "").split(".")
        f = (parts[1] if len(parts) > 1 and parts[0] == "tests" else parts[0]) or "?"
        d = files.setdefault(f, dict(tests=0, failed=0, skipped=0))
        d["tests"] += 1
        if case.find("failure") is not None or case.find("error") is not None:
            d["failed"] += 1
            failed_names.append(f"{case.get('classname')}::{case.get('name')}")
        if case.find("skipped") is not None:
            d["skipped"] += 1
    n, fails, errs, skips = (int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    out = dict(tests=n, failures=fails, errors=errs, skipped=skips, passed=n - fails - errs - skips,
               duration_s=time.perf_counter() - t, by_file=files, failed=failed_names,
               last_line=(p.stdout.strip().splitlines() or [""])[-1])
    out["verdict"] = "PASS" if fails + errs == 0 else "FAIL"
    out["verdict_text"] = f"{out['passed']} passed, {fails} failed, {errs} errors, {skips} skipped"
    run.progress("T7", 1, 1)
    run.save_json("T7_metrics.json", out)
    return out


TESTS = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7}


# ============================================================================ main loop
def not_throttled(pid: int | None = None) -> None:
    """Windows 11 runs window-less background processes on the slow efficiency cores (measured here: about 3 of
    12 threads busy instead of 6-7, 2-4x slower). Above-normal priority keeps the test process (and the T5 test
    server) on the normal cores, as for the app running in a terminal window."""
    if sys.platform != "win32":
        return
    try:
        import psutil
        psutil.Process(pid).nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass


def execute(run: Run, only: list[str] | None = None) -> None:
    not_throttled()
    pid_file = run.dir / ("runner.pid" if only is None else "runner_rerun.pid")
    pid_file.write_text(str(os.getpid()))
    cache = PredCache(run)
    try:
        for test in only or run.info["tests"]:
            st = run.state().get(test, {})
            if st.get("status") == "done":
                continue
            run.update(test, status="running", started=datetime.now().isoformat(timespec="seconds"), error="")
            run.log(f"{test} started")
            try:
                res = TESTS[test](run, cache)
                from ..version import APP_VERSION
                if isinstance(res, dict):                       # the app version this test ran with
                    res["app_version"] = APP_VERSION
                    run.save_json(f"{test}_metrics.json", res)
                run.update(test, status="done", verdict=res.get("verdict"), summary=res.get("verdict_text"),
                           finished=datetime.now().isoformat(timespec="seconds"))
                run.log(f"{test} done: {res.get('verdict')} - {res.get('verdict_text')}")
            except Exception as e:  # noqa: BLE001
                run.update(test, status="failed", error=f"{type(e).__name__}: {e}")
                run.log(f"{test} FAILED to run:\n{traceback.format_exc()}")
        try:
            from .report import build_report
            path = build_report(run)
            run.log(f"report written: {path}")
        except Exception:  # noqa: BLE001
            run.log("report could not be built:\n" + traceback.format_exc())
    finally:
        pid_file.unlink(missing_ok=True)


def rerun(run: Run, tests: list[str], quiet: bool = False) -> None:
    """Runs the given tests of an existing run again (e.g. T2 after the reference file was replaced). The
    earlier result is kept as <T>_metrics_attempt<k>.json and listed in the report; criteria are unchanged."""
    for t in tests:
        f = run.dir / f"{t}_metrics.json"
        if f.exists():
            k = 1
            while (run.dir / f"{t}_metrics_attempt{k}.json").exists():
                k += 1
            f.rename(run.dir / f"{t}_metrics_attempt{k}.json")
            run.log(f"{t}: earlier result kept as {t}_metrics_attempt{k}.json; running {t} again")
        run.update(t, status="pending", verdict=None, summary="")
    if quiet:                              # no progress writes (another runner is busy in the same folder)
        run.progress = lambda *a, **k: None
    execute(run, only=tests)


def is_running(run: Run) -> bool:
    f = run.dir / "runner.pid"
    if not f.exists():
        return False
    try:
        import psutil
        p = psutil.Process(int(f.read_text()))
        return p.is_running() and "python" in p.name().lower()
    except Exception:  # noqa: BLE001
        return False


def start_background(run: Run) -> None:
    """Starts this module as a separate process for the run (used by the Validation Lab page)."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    out = open(run.dir / "runner.out", "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "-m", "webapp.validation.runner", "--run", str(run.dir)], cwd=str(config.SYSTEM_DIR),
                     stdout=out, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", action="store_true", help="create a new run")
    ap.add_argument("--run", help="resume / continue this run folder")
    ap.add_argument("--tests", default=",".join(ALL_TESTS))
    ap.add_argument("--t1", choices=["sample", "full"], default="sample")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--rerun", help="with --run: run these tests again, e.g. T2 (earlier results are kept)")
    a = ap.parse_args(argv)
    if a.new:
        run = new_run([t.strip() for t in a.tests.split(",") if t.strip()], a.t1)
    elif a.run:
        run = Run(Path(a.run))
    else:
        ap.error("use --new or --run")
    print("run folder:", run.dir, flush=True)
    if a.rerun:
        from .runner import is_running as _busy
        rerun(run, [t.strip() for t in a.rerun.split(",") if t.strip()], quiet=_busy(run))
        return
    if a.report_only:
        from .report import build_report
        print(build_report(run))
        return
    execute(run)


if __name__ == "__main__":
    main()
