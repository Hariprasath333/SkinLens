"""
Runs research/kaggle_stage.sh stage by stage on a tiny fake dataset (CPU, SMOKE=1), simulating
separate Kaggle sessions: each session writes to its own OUT_DIR and only sees earlier sessions'
outputs through MODELS_DIR / DATA_DIR - exactly like attaching a saved notebook output as input.

Covers: stage list, data stage (relative paths, CSVs, size), training paused by the time limit and
resumed in the next session from the per-epoch checkpoint, all five model stages, ensemble +
export, CV interrupted mid-way and continued (finished folds skipped), report.
Needs bash (Git Bash on Windows). Takes a few minutes on CPU.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from test_pipeline import make_fake_raw

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "kaggle_stage.sh"


def find_bash():
    for c in [os.getenv("BASH_EXE"), r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")]:
        if c and Path(c).exists() and "System32" not in c:
            return c
    return None


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not found")


def stage(name, out, models="", data="", raw="", max_hours="11", expect=0, **extra):
    extra.setdefault("LOCAL_DATA", str(Path(out).parent / "local_data"))  # where packed images are unpacked
    env = dict(os.environ, PY=sys.executable, INSTALL="0", SMOKE="1", SKIP_DOWNLOAD="1",
               OUT_DIR=str(out), MODELS_DIR=models, DATA_DIR=str(data), RAW_DIR=str(raw),
               MAX_HOURS=max_hours, **extra)
    r = subprocess.run([BASH, str(SCRIPT), *([name] if name else [])], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    out_text = r.stdout + r.stderr
    assert r.returncode == expect, out_text[-4000:]
    return out_text


def status(out):
    return (Path(out) / "STATUS.txt").read_text()


def test_stage_list_and_bad_name(tmp_path):
    txt = stage("", tmp_path / "o", expect=2)
    for s in ["data", "train_efficientnet_b3", "train_resnet50", "train_densenet121", "train_convnext",
              "train_vit", "ensemble", "cv", "report"]:
        assert s in txt
    assert "Unknown stage" in stage("train_alexnet", tmp_path / "o", expect=2)


def test_all_stages_across_sessions(tmp_path):
    raw = tmp_path / "raw"
    make_fake_raw(raw)
    S = {k: tmp_path / f"session_{k}" for k in "ABCDEFGH"}

    # --- session A: data
    stage("data", S["A"], raw=raw)
    data = S["A"] / "data_processed"
    man = pd.read_csv(data / "manifest.csv")
    assert all(not Path(p).is_absolute() and p.startswith("images_384/") for p in man["path"])
    assert (data / "images_384.tar").exists() and not (data / "images_384").exists()  # packed for Kaggle
    for f in ["train.csv", "val.csv", "test.csv", "external_isic2018_test.csv", "cleaning_report.md"]:
        assert (data / f).exists(), f
    assert "finished - data_processed is" in status(S["A"])
    # the processed folder is portable: move it, as Kaggle does when it becomes an input dataset
    moved = tmp_path / "kaggle_input" / "my-data" / "data_processed"
    shutil.copytree(data, moved)

    # --- session B: training stops at the time limit after one epoch (checkpoint saved)
    txt = stage("train_resnet50", S["B"], data=moved, max_hours="0.000001")
    assert "TIME LIMIT" in txt and "PAUSED" in status(S["B"])
    run_b = S["B"] / "runs" / "resnet50" / "holdout"
    assert (run_b / "last.pt").exists() and not (run_b / "model_card.json").exists()

    # --- session C: same stage again with B's output attached -> resumes at epoch 2 and finishes
    txt = stage("train_resnet50", S["C"], models=str(S["B"]), data=moved)
    assert "Resuming" in txt and "after epoch 1" in txt
    run_c = S["C"] / "runs" / "resnet50" / "holdout"
    assert (run_c / "model_card.json").exists() and not (run_c / "last.pt").exists()
    assert len(pd.read_csv(run_c / "history.csv")) == 2
    for st in ["train_efficientnet_b3", "train_densenet121", "train_convnext", "train_vit"]:
        stage(st, S["C"], data=moved)
    assert "already finished" in stage("train_resnet50", S["C"], data=moved)   # nothing is redone
    for m in ["resnet50", "efficientnet_b3", "densenet121", "convnext_tiny", "vit_b16"]:
        assert (S["C"] / "runs" / m / "holdout" / "model_card.json").exists(), m

    # --- session D: ensemble from the models of session C
    stage("ensemble", S["D"], models=str(S["C"]), data=moved)
    exp = json.loads((S["D"] / "models" / "best" / "export.json").read_text())
    assert exp["members"] and (S["D"] / "models" / "best" / exp["members"][0] / "best.pt").exists()
    res = pd.read_csv(S["D"] / "ensemble" / "results.csv")
    assert res["model"].str.startswith("ensemble").any()
    best = json.loads((S["D"] / "ensemble" / "best_model.json").read_text())["best_single_model"]

    # --- a session that is already past its time limit does not start a new fold at all
    prev = f"{S['C']},{S['D']}"
    stage("cv", tmp_path / "late", models=prev, data=moved, max_hours="0.000001")
    assert "PAUSED" in status(tmp_path / "late") and not (tmp_path / "late" / "runs").exists()

    # --- session E: CV stops at the time limit inside fold 0 (~11 s budget: one epoch, then pause)
    txt = stage("cv", S["E"], models=prev, data=moved, max_hours="0.003", EPOCH_HINT="0")
    assert f"Cross-validating: {best}" in txt and "PAUSED" in status(S["E"])
    assert (S["E"] / "runs" / best / "cv0" / "last.pt").exists()

    # --- session F: CV continues (fold 0 resumed, folds 1-4 trained), saved after every fold
    txt = stage("cv", S["F"], models=f"{prev},{S['E']}", data=moved)
    assert "Resuming" in txt
    for k in range(5):
        assert (S["F"] / "runs" / best / f"cv{k}" / "model_card.json").exists()
        assert f"fold {k} of {best} done" in status(S["F"])
    assert (S["F"] / "label_review" / "label_issues.csv").exists()

    # --- session G: running CV again skips every finished fold (and copies them without weights)
    txt = stage("cv", S["G"], models=f"{prev},{S['F']}", data=moved)
    assert txt.count("already finished - skipping") == 5
    assert not (S["G"] / "runs" / best / "cv0" / "best.pt").exists()

    # --- session H: report from everything
    stage("report", S["H"], models=f"{S['C']},{S['D']},{S['F']}", data=moved)
    r = S["H"] / "results"
    table = pd.read_csv(r / "results.csv")
    assert {"accuracy", "precision_macro", "recall_sensitivity_macro", "specificity_macro", "f1_macro",
            "auc_roc_macro"} <= set(table.columns)
    assert len(table) == 6  # 5 models + ensemble
    cv = pd.read_csv(r / "results_cv.csv")
    assert cv.loc[0, "folds"] == 5 and cv.loc[0, "model"] == best
    for f in ["results_per_class.csv", "results_external.csv", "roc_all_models.png", "confusion_resnet50.png",
              "curves_vit_b16.png", "cleaning_report.md", "class_distribution.png", "summary.md"]:
        assert (r / f).exists(), f
    for s in S.values():  # no half-written files anywhere
        if s.exists():
            assert not list(s.rglob("*.tmp")), s
