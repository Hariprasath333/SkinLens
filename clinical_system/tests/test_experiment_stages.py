"""
Runs the paper-experiment stages of research/kaggle_stage.sh on tiny fake data (CPU, SMOKE=1), each
in its own simulated Kaggle session: audit, data (packed images + audit + naive manifest),
leak_test, focus, lesion_focus_train, calibrate, external. Checks the outputs each stage must
produce and that every session's output stays under Kaggle's ~500-file limit.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_kaggle_stages import BASH, stage, status
from test_pipeline import make_fake_masks, make_fake_raw

pytestmark = pytest.mark.skipif(BASH is None, reason="bash not found")
ROOT = Path(__file__).resolve().parents[1]


def n_files(d: Path) -> int:
    return sum(1 for f in d.rglob("*") if f.is_file())


def test_experiment_stages(tmp_path):
    raw = tmp_path / "raw"
    make_fake_raw(raw)
    masks = make_fake_masks(raw, tmp_path / "fake_masks")
    S = {k: tmp_path / f"s_{k}" for k in ["audit", "data", "train", "ens", "leak", "focus", "lft", "cal", "ext"]}
    local = {"LOCAL_DATA": str(tmp_path / "local_data")}

    # ---- audit (stand-alone)
    txt = stage("audit", S["audit"], raw=raw, **local)
    assert "Data audit" in txt and (S["audit"] / "audit" / "audit_report.md").exists()
    assert (S["audit"] / "audit" / "conflict_examples.tar").exists()          # packed
    assert not (S["audit"] / "audit" / "conflict_examples").exists()

    # ---- data: images packed into one .tar, audit + naive manifest included
    stage("data", S["data"], raw=raw, **local)
    data = S["data"] / "data_processed"
    assert (data / "images_384.tar").exists() and not (data / "images_384").exists()
    assert (data / "manifest_naive.csv").exists() and (data / "audit" / "audit_report.md").exists()
    assert n_files(S["data"]) < 500

    # ---- two trained models + ensemble (training reads the packed data: unpacked to local disk)
    for st in ("train_resnet50", "train_efficientnet_b3"):
        stage(st, S["train"], data=data, **local)
    assert (tmp_path / "local_data" / "images_384").is_dir()
    stage("ensemble", S["ens"], models=str(S["train"]), **local)

    # ---- leak_test: naive model trained, clean ResNet-50 re-used from the train session
    prev = f"{S['train']},{S['ens']}"
    txt = stage("leak_test", S["leak"], models=prev, data=data, **local)
    assert "resnet50/holdout already finished" in txt
    rep = (S["leak"] / "leak_test" / "leak_report.md").read_text()
    assert "Inflation caused by the naive split" in rep and (S["leak"] / "leak_test" / "leak_comparison.png").exists()
    leak = json.loads((S["leak"] / "leak_test" / "leak_summary.json").read_text())
    assert leak["leaked_share"] > 0 and set(leak["inflation"]) == {"accuracy", "f1_macro", "auc_roc_macro"}
    card = json.loads(next((S["leak"] / "leak_test" / "naive_runs").rglob("model_card.json")).read_text())
    assert card["variant"] == "naive_split" and card["manifest"] == "manifest_naive.csv"

    # ---- focus: masks matched to test images, metrics + 20 examples
    txt = stage("focus", S["focus"], models=prev, data=data, MASK_DIRS=str(masks), **local)
    best = json.loads(next(S["ens"].rglob("best_model.json")).read_text())["best_single_model"]
    fdir = S["focus"] / "focus" / best
    per = pd.read_csv(fdir / "focus_per_image.csv")
    assert len(per) > 0 and per.in_lesion_ratio.between(0, 1).all() and per.iou.between(0, 1).all()
    rep = (fdir / "focus_report.md").read_text()
    for t in ("by class", "by correctness", "Images used"):
        assert t in rep
    assert (fdir / "focus_by_class.png").exists() and (fdir / "focus_examples_montage.jpg").exists()
    assert (S["focus"] / "masks" / "masks_index.csv").exists() and (S["focus"] / "masks" / "masks_384.tar").exists()
    assert n_files(S["focus"]) < 500

    # ---- lesion-focused retraining (masks re-used from the focus session's packed output)
    txt = stage("lesion_focus_train", S["lft"], models=f"{prev},{S['focus']}", data=data, **local)
    assert "Lesion masks for" in txt
    cmp_ = pd.read_csv(S["lft"] / "lesion_focus" / "lesion_focus_comparison.csv")
    assert len(cmp_) == 2 and {"test_accuracy", "in_lesion_mean", "iou_mean"} <= set(cmp_.columns)
    hist = pd.read_csv(next((S["lft"] / "lesion_focus" / "runs").rglob("history.csv")))
    assert "train_attn_loss" in hist.columns                                   # mask-guided loss was used

    # ---- calibrate: ECE before/after, reliability diagram, calibrated web-app export
    stage("calibrate", S["cal"], models=prev, **local)
    cal = pd.read_csv(S["cal"] / "calibration" / "calibration.csv")
    assert {"ece_before", "ece_after", "temperature"} <= set(cal.columns)
    assert (S["cal"] / "calibration" / "reliability_diagrams.png").exists()
    exp = json.loads((S["cal"] / "calibration" / "models" / "best" / "export.json").read_text())
    assert "calibration" in exp
    code = ("import sys; sys.path.insert(0, '.'); from pathlib import Path; import numpy as np;"
            "from webapp.ml.predictor import TimmPredictor;"
            f"p = TimmPredictor(Path(r'{S['cal'] / 'calibration' / 'models' / 'best'}'));"
            "x = (np.random.default_rng(0).random((200, 200, 3)) * 255).astype('uint8');"
            "pr = p.predict_batch([x])[0]; assert abs(pr.sum() - 1) < 1e-5; print('T', p.ensemble_temperature)")
    import subprocess
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, APP_DATA_DIR=str(tmp_path / "app")))
    assert r.returncode == 0, r.stderr[-2000:]

    # ---- external: every model (incl. experiment variants) + ensemble
    stage("external", S["ext"], models=f"{prev},{S['leak']},{S['lft']}", data=data, **local)
    ext = pd.read_csv(S["ext"] / "external" / "external_results.csv")
    names = set(ext.model)
    assert {"resnet50", "efficientnet_b3", "resnet50+naive_split", f"{best}+lesion_focus"} <= names
    assert any(n.startswith("ensemble") for n in names)
    assert "no SCC images" in (S["ext"] / "external" / "external_report.md").read_text()

    for s in S.values():
        assert n_files(s) < 500, s
        assert "FAILED" not in status(s)
