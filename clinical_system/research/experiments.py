"""
Analyses for the paper's key experiments (called by kaggle_stage.sh; each sub-command can also be
run by hand). Training itself is done by train.py.

  masks     download public lesion masks (HAM10000 + ISIC 2018 Task 1) and match them to our images
  focus     where does a model look? Grad-CAM energy inside the lesion mask, IoU, examples
  compare-focus   before/after table for the lesion-focus experiment
  leak      naive (leaky) split vs clean split: how much are the numbers inflated?
  calibrate temperature scaling: ECE before/after, reliability diagrams, confidence-level accuracy,
            calibrated export for the web app
  external  every model + ensemble on the external ISIC 2018 test set
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("APP_DATA_DIR", str(Path(tempfile.gettempdir()) / "skinlens_app_data"))

from common import CLASSES, RISK  # noqa: E402
from evaluate import auc, discover_runs, load_run, metrics, softmax  # noqa: E402

ISIC = "https://isic-archive.s3.amazonaws.com/challenges/2018"
MASK_SOURCES = {
    "ham10000_tschandl": dict(dataverse_doi="doi:10.7910/DVN/DBW86T", match="segmentation"),
    "isic2018_task1_train": dict(url=f"{ISIC}/ISIC2018_Task1_Training_GroundTruth.zip"),
    "isic2018_task1_val": dict(url=f"{ISIC}/ISIC2018_Task1_Validation_GroundTruth.zip"),
    "isic2018_task1_test": dict(url=f"{ISIC}/ISIC2018_Task1_Test_GroundTruth.zip"),
}


def md_table(df: pd.DataFrame, floatfmt: str = ".4f", index: bool = True) -> str:
    """Markdown table without the optional 'tabulate' package."""
    d = df.reset_index() if index else df
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(f(v) for v in row) + " |" for row in d.itertuples(index=False)])


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    return p


def roots_of(values) -> list[Path]:
    return [Path(p.strip()) for v in values for p in str(v).split(",") if p.strip() and Path(p.strip()).exists()]


def resolve_manifest_paths(df: pd.DataFrame, manifest: Path) -> pd.DataFrame:
    base = manifest.resolve().parent
    df = df.copy()
    df["path"] = [p if Path(p).is_absolute() else str(base / p) for p in df["path"].astype(str)]
    return df


def norm_id(s) -> str:
    return str(s).replace("_downsampled", "").strip()


def ece_score(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & ((conf < bins[i + 1]) if i < n_bins - 1 else (conf <= bins[i + 1]))
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def nll(probs, y):
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.05), np.log(10), 400))
    return float(grid[int(np.argmin([nll(softmax(logits / t), y) for t in grid]))])


def raw_logits(run_dir: Path) -> dict:
    z = np.load(run_dir / "predictions.npz")
    return {k: z[k] for k in z.files}


# ============================================================================ masks
def _download_masks(raw: Path) -> list[Path]:
    from download_datasets import UA, download
    import requests
    raw.mkdir(parents=True, exist_ok=True)
    got = []
    for name, s in MASK_SOURCES.items():
        dest = raw / f"{name}.zip"
        try:
            if not dest.exists():
                url = s.get("url")
                if url is None:
                    r = requests.get("https://dataverse.harvard.edu/api/datasets/:persistentId/",
                                     params={"persistentId": s["dataverse_doi"]},
                                     headers=dict(UA, Accept="application/json"), timeout=60)
                    r.raise_for_status()
                    fid = next(f["dataFile"]["id"] for f in r.json()["data"]["latestVersion"]["files"]
                               if s["match"] in f["dataFile"].get("filename", "").lower())
                    url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
                print(f"Downloading masks: {name}", flush=True)
                download(url, dest)
            out = raw / name
            if not out.exists():
                with zipfile.ZipFile(dest) as z:
                    z.extractall(out)
            got.append(out)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: mask source {name} not available ({e}); continuing without it", flush=True)
    return got


def cmd_masks(a) -> None:
    """Prints the folder holding masks_index.csv + mask PNGs (reused if an earlier session made it)."""
    for root in roots_of(a.roots):
        for idx in sorted(root.rglob("masks_index.csv")):
            d = idx.parent
            tars = sorted(d.glob("*.tar"))
            if tars and not all((d / t.stem).is_dir() for t in tars):
                local = Path(a.local)
                if not (local / "masks_index.csv").exists():
                    import tarfile
                    shutil.rmtree(local, ignore_errors=True)
                    local.mkdir(parents=True)
                    shutil.copy2(idx, local / "masks_index.csv")
                    for t in tars:
                        with tarfile.open(t) as tf:
                            try:
                                tf.extractall(local, filter="data")
                            except TypeError:
                                tf.extractall(local)
                print(local); return
            print(d); return

    data = Path(a.data)
    man = pd.read_csv(data / "manifest.csv")
    sources = [Path(p) for p in a.mask_dirs] if a.mask_dirs else _download_masks(Path(a.raw))
    found = {}  # isic id -> (path, source); HAM masks take priority (listed first)
    for src in sources:
        for f in sorted(src.rglob("*_segmentation.png")):
            found.setdefault(f.name.replace("_segmentation.png", ""), (f, src.name))
    out = Path(a.out)
    mdir = out / f"masks_{a.size}"
    mdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in man.itertuples():
        hit = found.get(norm_id(r.src_id))
        if hit is None:
            continue
        m = cv2.imread(str(hit[0]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        s = a.size / min(m.shape)
        if s < 1:
            m = cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(mdir / f"{r.image_uid}.png"), ((m > 127) * 255).astype(np.uint8))
        rows.append(dict(image_uid=r.image_uid, mask_path=f"{mdir.name}/{r.image_uid}.png", mask_source=hit[1],
                         split=r.split, label=r.label))
    idx = pd.DataFrame(rows, columns=["image_uid", "mask_path", "mask_source", "split", "label"])
    idx.to_csv(out / "masks_index.csv", index=False)
    summary = idx.groupby(["split", "mask_source"]).size().unstack(fill_value=0) if len(idx) else pd.DataFrame()
    (out / "masks_summary.md").write_text(
        f"# Lesion masks\n\nMask files available: {len(found):,}. Matched to our images: **{len(idx):,}** "
        f"of {len(man):,}.\n\n" + (md_table(summary, ".0f") if len(summary) else "(none)") + "\n", encoding="utf-8")
    print(f"Matched masks for {len(idx):,} of {len(man):,} images", file=sys.stderr)
    print(out)


# ============================================================================ focus
def ensemble_members(roots: list[Path], available) -> tuple[list[str], Path | None]:
    """Members of the ensemble built by the 'ensemble' stage (best_model.json), whether or not the
    ensemble was the model exported for the web app; also returns the export.json path (or None)."""
    export = next((f for r in roots for f in sorted(r.rglob("export.json"))), None)
    bm = next((f for r in roots for f in sorted(r.rglob("best_model.json"))), None)
    members = json.loads(bm.read_text()).get("ensemble_members", []) if bm else []
    if not members and export:
        members = json.loads(export.read_text())["members"]
    return [m for m in members if m in available], export


def build_predictor(run_dir: Path):
    import torch
    import timm
    from webapp.ml.predictor import Member, Predictor, _cam_layer
    card = json.loads((run_dir / "model_card.json").read_text())
    p = Predictor()
    model = timm.create_model(card["timm_name"], pretrained=False, num_classes=len(card["classes"]))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=p.device, weights_only=True))
    model.to(p.device).eval()
    layer, tok = _cam_layer(model, card["cam_layer"])
    p.members = [Member(model, card["img_size"], card.get("temperature", 1.0), layer, tok)]
    p.classes = card["classes"]
    p.name = card["model"] + (f"+{card['variant']}" if card.get("variant") else "")
    return p, card


def focus_metrics(cam: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> dict:
    m = mask.astype(bool)
    tot = cam.sum() + 1e-8
    inside = float(cam[m].sum() / tot)
    hot = cam >= thr
    inter, union = (hot & m).sum(), (hot | m).sum()
    area = float(m.mean())
    return dict(in_lesion_ratio=inside, iou=float(inter / union) if union else 0.0, lesion_area=area,
                lift=inside / area if area > 0 else float("nan"))


def render_example(rgb, cam, mask, caption, size=256):
    from webapp.ml.xai import overlay
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    heat = overlay(img, cv2.resize(cam, (size, size)))
    m = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    mo = img.copy()
    mo[m > 0] = (0.55 * mo[m > 0] + 0.45 * np.array([0, 180, 216])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for panel in (heat, mo):
        cv2.drawContours(panel, cnts, -1, (255, 255, 255), 2)
    gap = np.full((size, 6, 3), 255, np.uint8)
    row = np.hstack([img, gap, heat, gap, mo])
    bar = np.full((40, row.shape[1], 3), 255, np.uint8)
    for k, line in enumerate(caption.split("\n")[:2]):
        cv2.putText(bar, line, (6, 16 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return cv2.cvtColor(np.vstack([row, bar]), cv2.COLOR_RGB2BGR)


def run_focus(run_dir: Path, data: Path, masks_dir: Path, out: Path, split: str, n_examples: int,
              max_images: int = 0) -> dict:
    from build_dataset import montage
    from webapp.ml import xai
    out.mkdir(parents=True, exist_ok=True)
    man = resolve_manifest_paths(pd.read_csv(data / "manifest.csv"), data / "manifest.csv")
    idx = pd.read_csv(masks_dir / "masks_index.csv")
    rows = man[man["split"] == split].merge(idx[["image_uid", "mask_path", "mask_source"]], on="image_uid")
    total = int((man["split"] == split).sum())
    if max_images:
        rows = rows.groupby("label", group_keys=False).head(max(1, max_images // len(CLASSES)))
    pred, card = build_predictor(run_dir)
    res, cache = [], {}
    for k, r in enumerate(rows.itertuples()):
        rgb = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(masks_dir / r.mask_path), cv2.IMREAD_GRAYSCALE)
        probs = pred.predict_batch([rgb])[0]
        pi = int(probs.argmax())
        cam = xai.gradcam(pred, rgb, pi, out_size=256)
        msq = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 127
        fm = focus_metrics(cam, msq)
        res.append(dict(image_uid=r.image_uid, label=r.label, predicted=pred.classes[pi], confidence=float(probs[pi]),
                        correct=pred.classes[pi] == r.label, mask_source=r.mask_source, **fm))
        cache[r.image_uid] = (rgb, cam, msq)
        if len(cache) > 400:  # keep memory bounded; examples are re-computed below if needed
            cache.pop(next(iter(cache)))
        if (k + 1) % 200 == 0:
            print(f"  focus: {k + 1:,}/{len(rows):,}", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(out / "focus_per_image.csv", index=False)

    def agg(g):
        return pd.Series(dict(images=len(g), in_lesion_mean=g.in_lesion_ratio.mean(),
                              in_lesion_median=g.in_lesion_ratio.median(), iou_mean=g.iou.mean(),
                              iou_median=g.iou.median(), lesion_area_mean=g.lesion_area.mean(),
                              lift_median=g.lift.median()))
    summary = {"model": pred.name, "split": split, "images_with_mask": len(df), "images_in_split": total}
    tables = {}
    if len(df):
        tables["overall"] = agg(df).to_frame("all").T
        tables["by_class"] = pd.DataFrame({c: agg(df[df.label == c]) for c in CLASSES if c in set(df.label)}).T
        tables["by_correctness"] = pd.DataFrame({k: agg(df[df.correct == v]) for k, v in
                                                 (("correct", True), ("wrong", False)) if (df.correct == v).any()}).T
        for k2, t in tables.items():
            t.to_csv(out / f"focus_{k2}.csv", float_format="%.4f")
        summary.update({k: float(tables["overall"].iloc[0][k]) for k in tables["overall"].columns})
        summary["accuracy_on_masked_images"] = float(df.correct.mean())
        # figure: in-lesion ratio per class
        p = plt()
        fig, ax = p.subplots(figsize=(7, 3.6))
        cls = list(tables["by_class"].index)
        ax.boxplot([df.loc[df.label == c, "in_lesion_ratio"] for c in cls], showfliers=False)
        ax.set_xticks(range(1, len(cls) + 1), cls)
        ax.axhline(df.lesion_area.mean(), ls="--", color="grey", label="lesion share of image (chance level)")
        ax.set_ylabel("Grad-CAM energy inside lesion"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
        ax.set_title(f"Where does {pred.name} look? ({len(df):,} {split} images with masks)")
        fig.tight_layout(); fig.savefig(out / "focus_by_class.png", dpi=250); p.close(fig)
        # 20 examples: half correct, half wrong, spread over classes and focus values
        ex_dir = out / "focus_examples"
        ex_dir.mkdir(exist_ok=True)
        picks = []
        for flag in (True, False):
            sub = df[df.correct == flag].sort_values("in_lesion_ratio")
            if len(sub):
                take = np.unique(np.linspace(0, len(sub) - 1, min(n_examples // 2, len(sub))).astype(int))
                picks += list(sub.iloc[take].image_uid)
        tiles = []
        for j, uid in enumerate(picks[:n_examples]):
            r = df[df.image_uid == uid].iloc[0]
            if uid in cache:
                rgb, cam, msq = cache[uid]
            else:
                rr = rows[rows.image_uid == uid].iloc[0]
                rgb = cv2.cvtColor(cv2.imread(rr.path), cv2.COLOR_BGR2RGB)
                msq = cv2.resize(cv2.imread(str(masks_dir / rr.mask_path), 0), (256, 256),
                                 interpolation=cv2.INTER_NEAREST) > 127
                cam = xai.gradcam(pred, rgb, pred.classes.index(r.predicted), out_size=256)
            cap = (f"{uid}  true {r.label}  pred {r.predicted} ({100 * r.confidence:.0f}%)  "
                   f"{'correct' if r.correct else 'WRONG'}\n"
                   f"in-lesion {100 * r.in_lesion_ratio:.0f}%  IoU {r.iou:.2f}   | original | Grad-CAM | mask |")
            im = render_example(rgb, cam, msq, cap)
            cv2.imwrite(str(ex_dir / f"example_{j:02d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            tiles.append(im)
        if tiles:
            cv2.imwrite(str(out / "focus_examples_montage.jpg"), montage(tiles, cols=2, width=760),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    lines = [f"# Where does the model look? - {pred.name}", "",
             f"Images used: **{len(df):,}** {split} images that have a public lesion mask "
             f"(of {total:,} {split} images). Mask sources: "
             + (", ".join(f"{k} {v:,}" for k, v in df.mask_source.value_counts().items()) if len(df) else "none"), "",
             "- **In-lesion ratio** = share of the Grad-CAM heat-map energy that falls inside the lesion mask "
             "(1.0 = only the lesion). Compare with the lesion's share of the image (chance level).",
             "- **IoU** = overlap between the hot area (heat-map >= 0.5) and the mask.", ""]
    for name, t in tables.items():
        lines += [f"## {name.replace('_', ' ')}", "", md_table(t, ".3f"), ""]
    lines += ["Files: focus_per_image.csv, focus_by_class.png, focus_examples_montage.jpg, "
              "focus_examples.tar (20 x original | Grad-CAM | mask)."]
    (out / "focus_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "focus_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))
    return summary


def cmd_focus(a) -> None:
    run_focus(Path(a.run), Path(a.data), Path(a.masks), Path(a.out), a.split, a.examples, a.max_images)


def find_holdout(roots, model: str, variant: str = "") -> Path:
    holdout, _ = discover_runs(roots, variants=bool(variant))
    key = model + (f"+{variant}" if variant else "")
    if key not in holdout:
        sys.exit(f"No finished run '{key}' found below: {', '.join(map(str, roots))}")
    return holdout[key]


def cmd_compare_focus(a) -> None:
    out = Path(a.out)
    rows = []
    for name, run, fdir in (("before (normal training)", Path(a.before_run), Path(a.before_focus)),
                            ("after (lesion-focused training)", Path(a.after_run), Path(a.after_focus))):
        _, probs, labels, _ = load_run(run)
        m, _ = metrics(probs["test"], labels["test"])
        f = json.loads((fdir / "focus_summary.json").read_text())
        ext = {}
        if "external_isic2018_test" in probs:
            em, _ = metrics(probs["external_isic2018_test"], labels["external_isic2018_test"])
            ext = dict(external_accuracy=em["accuracy"], external_f1_macro=em["f1_macro"])
        rows.append(dict(model=name, test_accuracy=m["accuracy"], test_f1_macro=m["f1_macro"],
                         test_auc_macro=m["auc_roc_macro"], melanoma_sensitivity=m["melanoma_sensitivity"],
                         in_lesion_mean=f.get("in_lesion_mean"), in_lesion_median=f.get("in_lesion_median"),
                         iou_mean=f.get("iou_mean"), images_with_mask=f.get("images_with_mask"), **ext))
    t = pd.DataFrame(rows)
    t.to_csv(out / "lesion_focus_comparison.csv", index=False, float_format="%.4f")
    lines = ["# Lesion-focused training: before vs after", "", md_table(t, ".4f", index=False), ""]
    if len(t) == 2:
        lines += [f"Change: accuracy {100 * (t.test_accuracy[1] - t.test_accuracy[0]):+.2f} points, macro-F1 "
                  f"{100 * (t.test_f1_macro[1] - t.test_f1_macro[0]):+.2f} points, in-lesion attention "
                  f"{100 * (t.in_lesion_mean[1] - t.in_lesion_mean[0]):+.1f} points."]
    (out / "lesion_focus_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ============================================================================ leak test
def cmd_leak(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, run in (("naive", Path(a.naive)), ("clean", Path(a.clean))):
        _, probs, labels, data = load_run(run)
        m, _ = metrics(probs["test"], labels["test"])
        rows[name] = dict(split=name, test_images=m["n"], accuracy=m["accuracy"], f1_macro=m["f1_macro"],
                          auc_roc_macro=m["auc_roc_macro"], balanced_accuracy=m["balanced_accuracy"])
        if "external_isic2018_test" in probs:
            em, _ = metrics(probs["external_isic2018_test"], labels["external_isic2018_test"])
            rows[name].update(external_accuracy=em["accuracy"], external_f1_macro=em["f1_macro"])
        if name == "naive":
            nprobs, nlabels, nuids = probs["test"], labels["test"], data["test_uids"]
    t = pd.DataFrame(rows.values())
    infl = {k: rows["naive"][k] - rows["clean"][k] for k in ("accuracy", "f1_macro", "auc_roc_macro")}
    t.to_csv(out / "leak_results.csv", index=False, float_format="%.4f")

    # how many naive test images have a duplicate / same-lesion photo in the naive training set?
    nm = pd.read_csv(a.naive_manifest)
    train_groups = set(nm.loc[nm.split == "train", "group"])
    train_dups = set(nm.loc[nm.split == "train", "dup_cluster"]) if "dup_cluster" in nm else set()
    te = nm.set_index("image_uid").loc[list(nuids)]
    leaked = te["group"].isin(train_groups).to_numpy()
    exact = te["dup_cluster"].isin(train_dups).to_numpy() if train_dups else np.zeros(len(te), bool)
    correct = nprobs.argmax(1) == nlabels
    sub = pd.DataFrame([
        dict(subset="test images WITH a leaked copy / same-lesion photo in train", images=int(leaked.sum()),
             accuracy=float(correct[leaked].mean()) if leaked.any() else np.nan),
        dict(subset="  - of which an (near-)identical copy of the same photo", images=int(exact.sum()),
             accuracy=float(correct[exact].mean()) if exact.any() else np.nan),
        dict(subset="test images WITHOUT any leak", images=int((~leaked).sum()),
             accuracy=float(correct[~leaked].mean()) if (~leaked).any() else np.nan)])
    sub.to_csv(out / "leak_naive_test_breakdown.csv", index=False, float_format="%.4f")

    p = plt()
    fig, ax = p.subplots(figsize=(6, 3.6))
    keys = ["accuracy", "f1_macro", "auc_roc_macro"]
    x = np.arange(len(keys))
    for k2, (name, col) in enumerate((("naive", "#E63946"), ("clean", "#0077B6"))):
        vals = [100 * rows[name][k] for k in keys]
        ax.bar(x + (k2 - 0.5) * 0.38, vals, 0.38, label=f"{name} split", color=col)
        for xi, v in zip(x, vals):
            ax.text(xi + (k2 - 0.5) * 0.38, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Accuracy", "Macro-F1", "Macro-AUC"]); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Same model, same settings: naive vs leakage-free split"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "leak_comparison.png", dpi=250); p.close(fig)

    lines = ["# Data-leakage experiment (ResNet-50, identical training settings)", "",
             md_table(t, ".4f", index=False), "",
             "**Inflation caused by the naive split** (naive minus clean): "
             + ", ".join(f"{k} {100 * v:+.2f} points" for k, v in infl.items()), "",
             f"In the naive split, **{100 * leaked.mean():.1f}%** of the test images have a duplicate or another "
             f"photo of the same lesion in the training set ({100 * exact.mean():.1f}% an (near-)identical copy).", "",
             md_table(sub, ".4f", index=False), "",
             "Note: the two test sets are different images, so part of the gap can come from the test set itself; "
             "the breakdown above isolates the effect of leaked images within the naive test set.",
             "", "Figure: leak_comparison.png"]
    (out / "leak_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "leak_summary.json").write_text(json.dumps(dict(inflation=infl, leaked_share=float(leaked.mean()),
                                                           exact_share=float(exact.mean())), indent=2))
    print("\n".join(lines))


# ============================================================================ calibration
def reliability(ax, probs, y, label, n_bins=15):
    conf, corr = probs.max(1), probs.argmax(1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf >= bins[i]) & ((conf < bins[i + 1]) if i < n_bins - 1 else (conf <= bins[i + 1]))
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(corr[m].mean())
    ax.plot(xs, ys, "o-", label=f"{label} (ECE {ece_score(probs, y):.3f})")


def confidence_table(probs, y):
    from webapp.ml.xai import confidence_level
    rows = []
    srt = np.sort(probs, 1)
    lv = [confidence_level(float(s[-1]), float(s[-1] - s[-2]))[0] for s in srt]
    lv = np.array(lv)
    for level in ("High", "Medium", "Low"):
        m = lv == level
        rows.append(dict(level=level, share_of_cases=float(m.mean()),
                         accuracy=float((probs.argmax(1) == y)[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def cmd_calibrate(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = roots_of(a.roots)
    holdout, _ = discover_runs(roots)
    if not holdout:
        sys.exit("No trained models found.")
    rows, probs_cal = [], {}
    p = plt()
    n = len(holdout) + 1
    fig, axes = p.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for k, (name, d) in enumerate(holdout.items()):
        z = raw_logits(d)
        T = fit_temperature(z["val_logits"], z["val_labels"])
        for split in ("val", "test"):
            lg, y = z[f"{split}_logits"], z[f"{split}_labels"]
            before, after = softmax(lg), softmax(lg / T)
            rows.append(dict(model=name, split=split, temperature=T, ece_before=ece_score(before, y),
                             ece_after=ece_score(after, y), nll_before=nll(before, y), nll_after=nll(after, y),
                             accuracy=float((after.argmax(1) == y).mean())))
        probs_cal[name] = {s: softmax(z[f"{s}_logits"] / T) for s in ("val", "test")}
        probs_cal[name]["labels"] = {s: z[f"{s}_labels"] for s in ("val", "test")}
        ax = axes[0, k]
        reliability(ax, softmax(z["test_logits"]), z["test_labels"], "before")
        reliability(ax, softmax(z["test_logits"] / T), z["test_labels"], f"after (T={T:.2f})")
        ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title(name, fontsize=10)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)

    # ensemble (members from the exported model, else best 3 by validation)
    members, export = ensemble_members(roots, probs_cal)
    ens_T = None
    if len(members) > 1:
        lab = probs_cal[members[0]]["labels"]
        mean = {s: np.mean([probs_cal[m][s] for m in members], 0) for s in ("val", "test")}
        ens_T = fit_temperature(np.log(mean["val"] + 1e-12), lab["val"])
        for split in ("val", "test"):
            after = softmax(np.log(mean[split] + 1e-12) / ens_T)
            rows.append(dict(model="ensemble(" + "+".join(members) + ")", split=split, temperature=ens_T,
                             ece_before=ece_score(mean[split], lab[split]), ece_after=ece_score(after, lab[split]),
                             nll_before=nll(mean[split], lab[split]), nll_after=nll(after, lab[split]),
                             accuracy=float((after.argmax(1) == lab[split]).mean())))
        ax = axes[0, -1]
        reliability(ax, mean["test"], lab["test"], "members calibrated, averaged")
        reliability(ax, softmax(np.log(mean["test"] + 1e-12) / ens_T), lab["test"], f"+ ensemble T={ens_T:.2f}")
        ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ensemble", fontsize=10); ax.legend(fontsize=7)
        final_probs, final_y = softmax(np.log(mean["test"] + 1e-12) / ens_T), lab["test"]
    else:
        axes[0, -1].axis("off")
        best = members[0] if members else max(holdout, key=lambda m: next(
            r["accuracy"] for r in rows if r["model"] == m and r["split"] == "val"))
        final_probs, final_y = probs_cal[best]["test"], probs_cal[best]["labels"]["test"]
    fig.tight_layout(); fig.savefig(out / "reliability_diagrams.png", dpi=220); p.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(out / "calibration.csv", index=False, float_format="%.4f")
    conf = confidence_table(final_probs, final_y)
    conf.to_csv(out / "confidence_levels.csv", index=False, float_format="%.4f")

    # calibrated export for the web app
    if export:
        dst = out / "models" / "best"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(export.parent, dst)
        info = json.loads((dst / "export.json").read_text())
        for m in info["members"]:
            card_p = dst / m / "model_card.json"
            card = json.loads(card_p.read_text())
            card["temperature"] = float(t[(t.model == m) & (t.split == "val")].temperature.iloc[0])
            card_p.write_text(json.dumps(card, indent=2))
        # the ensemble temperature only applies if the web app runs exactly this ensemble
        info["ensemble_temperature"] = ens_T if sorted(info["members"]) == sorted(members) and len(members) > 1 else None
        info["calibration"] = dict(method="temperature scaling on the validation set",
                                   confidence_levels_on_test=conf.to_dict(orient="records"))
        (dst / "export.json").write_text(json.dumps(info, indent=2, default=float))
    test = t[t.split == "test"]
    lines = ["# Calibration (temperature scaling fitted on the validation set)", "",
             "ECE = expected calibration error (15 bins; 0 = confidence matches accuracy exactly).", "",
             md_table(test.drop(columns="split"), ".4f", index=False), "",
             "## Confidence levels shown in the web app (calibrated probabilities, test set)", "",
             md_table(conf, ".3f", index=False), "",
             "High = top probability >= 0.80 and >= 0.30 ahead of the 2nd class; Medium >= 0.55; Low otherwise.",
             "", "Figure: reliability_diagrams.png. Web-app model with calibration: models/best/"
             + (" (includes the ensemble temperature)" if ens_T else "")]
    (out / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ============================================================================ external test
def cmd_external(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = roots_of(a.roots)
    holdout, _ = discover_runs(roots, variants=True)
    split = a.split
    data = Path(a.data) if a.data else None
    rows, per_cls, probs_all, y_ref = [], [], {}, None
    for name, d in holdout.items():
        _, probs, labels, _ = load_run(d)
        if split in probs:
            pr, y = probs[split], labels[split]
        elif data is not None:  # older run without external predictions: predict now
            pr, y = predict_split(d, data, split)
        else:
            print(f"skip {name}: no {split} predictions and no DATA_DIR"); continue
        probs_all[name], y_ref = pr, y
        m, pc = metrics(pr, y)
        pm = pr.argmax(1)
        ak, scc = CLASSES.index("AK"), CLASSES.index("SCC")
        merged_ok = (pm == y) | ((y == ak) & (pm == scc))
        rows.append(dict(model=name, **{k: m[k] for k in ("n", "accuracy", "balanced_accuracy", "precision_macro",
                                                          "recall_sensitivity_macro", "specificity_macro",
                                                          "f1_macro", "auc_roc_macro", "melanoma_sensitivity",
                                                          "malignant_vs_benign_auc")},
                         accuracy_ak_scc_merged=float(merged_ok.mean()),
                         predicted_scc_share=float((pm == scc).mean())))
        per_cls.append(pc.assign(model=name))
    members, export = ensemble_members(roots, probs_all)
    if len(members) > 1:
        info = json.loads(export.read_text()) if export else {}
        mean = np.mean([probs_all[m] for m in members], 0)
        if info.get("ensemble_temperature") and sorted(info.get("members", [])) == sorted(members):
            mean = softmax(np.log(mean + 1e-12) / info["ensemble_temperature"])
        m, pc = metrics(mean, y_ref)
        rows.append(dict(model="ensemble(" + "+".join(members) + ")", **{k: m[k] for k in (
            "n", "accuracy", "balanced_accuracy", "precision_macro", "recall_sensitivity_macro", "specificity_macro",
            "f1_macro", "auc_roc_macro", "melanoma_sensitivity", "malignant_vs_benign_auc")}))
        per_cls.append(pc.assign(model="ensemble"))
        probs_all["ensemble"] = mean
    if not rows:
        sys.exit(f"No model has predictions for {split}.")
    t = pd.DataFrame(rows)
    t.to_csv(out / "external_results.csv", index=False, float_format="%.4f")
    pcs = pd.concat(per_cls)
    pcs.to_csv(out / "external_per_class.csv", index=False, float_format="%.4f")
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        p = plt()
        best = t.sort_values("f1_macro").iloc[-1]["model"]
        best = best if best in probs_all else "ensemble"
        for nm in [best] + (["ensemble"] if "ensemble" in probs_all and best != "ensemble" else []):
            fig, ax = p.subplots(figsize=(6.5, 5.5))
            ConfusionMatrixDisplay.from_predictions(y_ref, probs_all[nm].argmax(1), labels=range(len(CLASSES)),
                                                    display_labels=CLASSES, cmap="Blues", ax=ax, colorbar=False)
            ax.set_title(f"External ISIC 2018 test - {nm}"); fig.tight_layout()
            fig.savefig(out / f"external_confusion_{nm.split('(')[0]}.png", dpi=250); p.close(fig)
    except ImportError:
        pass
    counts = pd.Series(y_ref).map(dict(enumerate(CLASSES))).value_counts().reindex(CLASSES, fill_value=0)
    lines = ["# External test: ISIC 2018 Task 3 test set (never used for training or model selection)", "",
             f"Images: {len(y_ref):,} after removing any that duplicate our training data. Per class: "
             + ", ".join(f"{c} {n}" for c, n in counts.items()), "",
             "**Note: this set has no SCC images** (its 7 classes follow HAM10000, where 'akiec' covers both actinic "
             "keratosis and Bowen's disease = SCC in situ). Macro averages use only the 7 classes present; "
             "`accuracy_ak_scc_merged` also counts an SCC prediction for an AK image as correct, and "
             "`predicted_scc_share` shows how often the model chose SCC.", "",
             md_table(t, ".4f", index=False), "",
             "Per-class results: external_per_class.csv. Confusion matrices: external_confusion_*.png."]
    (out / "external_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def predict_split(run_dir: Path, data: Path, split: str):
    import torch
    from torch.utils.data import DataLoader
    from common import get_transforms
    from train import ImageDataset, predict
    import timm
    card = json.loads((run_dir / "model_card.json").read_text())
    man = resolve_manifest_paths(pd.read_csv(data / "manifest.csv"), data / "manifest.csv")
    d = man[man["split"] == split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(card["timm_name"], pretrained=False, num_classes=len(card["classes"]))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.to(device)
    _, tf = get_transforms(card["img_size"])
    lg, y = predict(model, DataLoader(ImageDataset(d, tf), 32), device, device.type == "cuda")
    return softmax(lg / card.get("temperature", 1.0)), y


# ============================================================================ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("masks"); s.add_argument("--data", required=True); s.add_argument("--roots", nargs="*", default=[])
    s.add_argument("--out", required=True); s.add_argument("--raw", default="/tmp/raw_masks")
    s.add_argument("--local", default=str(Path(tempfile.gettempdir()) / "skinlens_masks"))
    s.add_argument("--size", type=int, default=384)
    s.add_argument("--mask-dirs", nargs="*", default=[], help="use these folders instead of downloading")
    s.set_defaults(fn=cmd_masks)
    s = sub.add_parser("focus"); s.add_argument("--run", required=True); s.add_argument("--data", required=True)
    s.add_argument("--masks", required=True); s.add_argument("--out", required=True)
    s.add_argument("--split", default="test"); s.add_argument("--examples", type=int, default=20)
    s.add_argument("--max-images", type=int, default=0)
    s.set_defaults(fn=cmd_focus)
    s = sub.add_parser("compare-focus")
    for k in ("before-run", "after-run", "before-focus", "after-focus", "out"):
        s.add_argument(f"--{k}", required=True)
    s.set_defaults(fn=cmd_compare_focus)
    s = sub.add_parser("leak"); s.add_argument("--naive", required=True); s.add_argument("--clean", required=True)
    s.add_argument("--naive-manifest", required=True); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_leak)
    s = sub.add_parser("calibrate"); s.add_argument("--roots", nargs="+"); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("external"); s.add_argument("--roots", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--data", default=""); s.add_argument("--split", default="external_isic2018_test")
    s.set_defaults(fn=cmd_external)
    s = sub.add_parser("find-run"); s.add_argument("--roots", nargs="+"); s.add_argument("--model", required=True)
    s.add_argument("--variant", default="")
    s.set_defaults(fn=lambda a: print(find_holdout(roots_of(a.roots), a.model, a.variant)))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
