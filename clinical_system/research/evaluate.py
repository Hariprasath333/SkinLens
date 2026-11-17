"""
Compare all trained models, build the ensemble, write the paper tables/figures and
export the best model for the web app.

    python research/evaluate.py --runs runs --results results --export ../clinical_system/models/best

Writes to results/:
  results.csv            one row per model (+ ensemble) on the held-out test set
  results_per_class.csv  per-class precision / recall / specificity / F1 / AUC
  results_cv.csv         mean +/- std over the 5 cross-validation folds (if CV runs exist)
  results_external.csv   models on the external test sets (if present)
  roc_<model>.png, confusion_<model>.png, curves_<model>.png, roc_all_models.png
  summary.md             human-readable summary for the paper
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, RISK  # noqa: E402

MALIGNANT = [CLASSES.index(c) for c, r in RISK.items() if r in ("malignant", "premalignant")]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def auc(pos, score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos, score)) if 0 < pos.sum() < len(pos) else float("nan")


def metrics(probs: np.ndarray, y: np.ndarray) -> tuple[dict, pd.DataFrame]:
    from sklearn.metrics import confusion_matrix
    k = len(CLASSES)
    p = probs.argmax(1)
    cm = confusion_matrix(y, p, labels=range(k))
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    present = cm.sum(1) > 0
    prec = np.divide(tp, tp + fp, out=np.zeros(k), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(k), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.zeros(k), where=(tn + fp) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    aucs = np.array([auc(y == c, probs[:, c]) for c in range(k)])
    mal_true, mal_pred = np.isin(y, MALIGNANT), np.isin(p, MALIGNANT)
    summary = {
        "accuracy": float((p == y).mean()),
        "balanced_accuracy": float(rec[present].mean()),
        "precision_macro": float(prec[present].mean()),
        "recall_sensitivity_macro": float(rec[present].mean()),
        "specificity_macro": float(spec[present].mean()),
        "f1_macro": float(f1[present].mean()),
        "auc_roc_macro": float(np.nanmean(aucs)),
        "melanoma_sensitivity": float(rec[CLASSES.index("MEL")]),
        "melanoma_specificity": float(spec[CLASSES.index("MEL")]),
        "malignant_vs_benign_sensitivity": float((mal_pred & mal_true).sum() / max(mal_true.sum(), 1)),
        "malignant_vs_benign_specificity": float((~mal_pred & ~mal_true).sum() / max((~mal_true).sum(), 1)),
        "malignant_vs_benign_auc": auc(mal_true, probs[:, MALIGNANT].sum(1)),
        "n": int(len(y)),
    }
    per_class = pd.DataFrame(dict(cls=CLASSES, precision=prec, recall_sensitivity=rec, specificity=spec,
                                  f1=f1, auc=aucs, support=cm.sum(1)))
    return summary, per_class


def load_run(d: Path):
    card = json.loads((d / "model_card.json").read_text())
    z = np.load(d / "predictions.npz")
    T = card.get("temperature", 1.0)
    data = {k: z[k] for k in z.files}
    probs = {name[:-7]: softmax(data[name] / T) for name in data if name.endswith("_logits")}
    labels = {name[:-7]: data[name] for name in data if name.endswith("_labels")}
    return card, probs, labels, data


def discover_runs(roots: list[Path], variants: bool = False):
    """Finished runs = folders with model_card.json + predictions.npz, at any depth.
    If the same model/run exists under several roots, the LAST root wins (e.g. OUT_DIR over MODELS_DIR).
    Experiment variants (naive split, lesion-focus training, ...) are skipped unless variants=True;
    then they are returned with the key '<model>+<variant>'."""
    holdout, cv = {}, {}
    for root in roots:
        if not root.exists():
            continue
        for card_path in sorted(root.rglob("model_card.json")):
            d = card_path.parent
            if not (d / "predictions.npz").exists() or d.parent.name == "best":
                continue
            card = json.loads(card_path.read_text())
            run = card.get("run", d.name)
            variant = card.get("variant") or ""
            if variant and not variants:
                continue
            if run == "holdout":
                holdout[card["model"] + (f"+{variant}" if variant else "")] = d
            elif run.startswith("cv"):
                cv.setdefault(card["model"], {})[int(run[2:])] = d
    return dict(sorted(holdout.items())), cv


def plots(name, probs, y, history, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(y, probs.argmax(1), labels=range(len(CLASSES)),
                                            display_labels=CLASSES, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {name}"); fig.tight_layout()
    fig.savefig(out / f"confusion_{name}.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c, cls in enumerate(CLASSES):
        if 0 < (y == c).sum() < len(y):
            f, t, _ = roc_curve(y == c, probs[:, c])
            ax.plot(f, t, label=f"{cls} (AUC {auc(y == c, probs[:, c]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)"); ax.set_title(f"ROC - {name}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"roc_{name}.png", dpi=300); plt.close(fig)
    if history is not None and len(history):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(history["epoch"], history["train_loss"], "o-", label="train")
        axs[0].plot(history["epoch"], history["val_loss"], "o-", label="validation")
        axs[0].set_title("Loss"); axs[0].set_xlabel("epoch"); axs[0].legend()
        axs[1].plot(history["epoch"], history["train_acc"], "o-", label="train")
        axs[1].plot(history["epoch"], history["val_acc"], "o-", label="validation")
        axs[1].set_title("Accuracy"); axs[1].set_xlabel("epoch"); axs[1].legend()
        fig.suptitle(name); fig.tight_layout(); fig.savefig(out / f"curves_{name}.png", dpi=300); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs"],
                    help="one or more folders; finished runs are found anywhere below them")
    ap.add_argument("--results", default="results")
    ap.add_argument("--export", default="models/best", help="folder the web app loads ('' = do not export)")
    ap.add_argument("--ensemble-size", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)

    holdout, cv_dirs = discover_runs([Path(r) for r in args.runs])
    if not holdout:
        sys.exit(f"No finished holdout runs (model_card.json + predictions.npz) below: {', '.join(args.runs)}")
    print("Found holdout runs:", ", ".join(f"{m} ({d})" for m, d in holdout.items()))
    rows, per_cls, ext_rows, loaded = [], [], [], {}
    for name, d in holdout.items():
        card, probs, labels, _ = load_run(d)
        loaded[name] = (card, probs, labels, d)
        m, pc = metrics(probs["test"], labels["test"])
        vm, _ = metrics(probs["val"], labels["val"])
        rows.append(dict(model=name, **m, val_f1_macro=vm["f1_macro"], params_millions=card["params"] / 1e6,
                         inference_ms=card["inference_ms"], best_epoch=card["best_epoch"]))
        per_cls.append(pc.assign(model=name))
        hist = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else None
        plots(name, probs["test"], labels["test"], hist, out)
        for split in probs:
            if split.startswith("external_"):
                em, _ = metrics(probs[split], labels[split])
                ext_rows.append(dict(model=name, dataset=split, **em))

    # ---- ensemble of the best models (chosen on VALIDATION macro-F1, never on test)
    ranked = sorted(rows, key=lambda r: -r["val_f1_macro"])
    members = [r["model"] for r in ranked[:args.ensemble_size]] if len(ranked) >= 2 else []
    if members:
        same = all(np.array_equal(loaded[m][2]["test"], loaded[members[0]][2]["test"]) for m in members)
        if same:
            ens = {s: np.mean([loaded[m][1][s] for m in members], 0) for s in loaded[members[0]][1]
                   if all(s in loaded[m][1] for m in members)}
            lab = loaded[members[0]][2]
            m, pc = metrics(ens["test"], lab["test"])
            vm, _ = metrics(ens["val"], lab["val"])
            rows.append(dict(model="ensemble(" + "+".join(members) + ")", **m, val_f1_macro=vm["f1_macro"],
                             params_millions=sum(loaded[x][0]["params"] for x in members) / 1e6,
                             inference_ms=sum(loaded[x][0]["inference_ms"] for x in members), best_epoch=None))
            per_cls.append(pc.assign(model="ensemble"))
            plots("ensemble", ens["test"], lab["test"], None, out)
            for split in ens:
                if split.startswith("external_"):
                    em, _ = metrics(ens[split], lab[split])
                    ext_rows.append(dict(model="ensemble", dataset=split, **em))

    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False, float_format="%.4f")
    pd.concat(per_cls).to_csv(out / "results_per_class.csv", index=False, float_format="%.4f")
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(out / "results_external.csv", index=False, float_format="%.4f")

    # all-model ROC (malignant vs benign) for one figure in the paper
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for name, (_, probs, labels, _) in loaded.items():
            y = np.isin(labels["test"], MALIGNANT); s = probs["test"][:, MALIGNANT].sum(1)
            f, t, _ = roc_curve(y, s); ax.plot(f, t, label=f"{name} (AUC {auc(y, s):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey"); ax.legend(fontsize=7)
        ax.set_title("Malignant/pre-malignant vs benign"); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
        fig.tight_layout(); fig.savefig(out / "roc_all_models.png", dpi=300); plt.close(fig)
    except ImportError:
        pass

    # ---- cross-validation summary
    cv_rows = []
    for model_name, by_fold in sorted(cv_dirs.items()):
        folds = [by_fold[k] for k in sorted(by_fold)]
        fold_m = []
        for d in folds:
            _, probs, labels, _ = load_run(d)
            fold_m.append(metrics(probs["val"], labels["val"])[0])  # each fold's held-out part
        dfm = pd.DataFrame(fold_m).drop(columns="n")
        cv_rows.append({"model": model_name, "folds": len(folds),
                        **{f"{k}_mean": v for k, v in dfm.mean().items()},
                        **{f"{k}_std": v for k, v in dfm.std(ddof=1).items()}})
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(out / "results_cv.csv", index=False, float_format="%.4f")

    # ---- export best (by validation macro-F1) for the web app
    best = max(rows, key=lambda r: r["val_f1_macro"])
    chosen = members if best["model"].startswith("ensemble") else [best["model"]]
    best_single = max((r for r in rows if not r["model"].startswith("ensemble")), key=lambda r: r["val_f1_macro"])
    (out / "best_model.json").write_text(json.dumps(dict(
        best=best["model"], members=chosen, best_single_model=best_single["model"],
        ensemble_members=members, chosen_by="validation macro-F1"), indent=2))
    exp = Path(args.export) if args.export else None
    if exp:
        exp.mkdir(parents=True, exist_ok=True)
        for m in chosen:
            dst = exp / m
            dst.mkdir(exist_ok=True)
            shutil.copy2(loaded[m][3] / "best.pt", dst / "best.pt")
            shutil.copy2(loaded[m][3] / "model_card.json", dst / "model_card.json")
        (exp / "export.json").write_text(json.dumps(dict(kind="timm", members=chosen, chosen_by="validation macro-F1",
                                                         best_single_model=best_single["model"],
                                                         test_metrics={k: best[k] for k in best if k != "model"}),
                                                    indent=2, default=float))

    cols = ["model", "accuracy", "precision_macro", "recall_sensitivity_macro", "specificity_macro",
            "f1_macro", "auc_roc_macro", "melanoma_sensitivity"]
    lines = ["# Results summary (held-out test set)", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join([r["model"]] + [f"{100 * r[c]:.2f}" for c in cols[1:]]) + " |")
    lines += ["", f"Best model (chosen by validation macro-F1): **{best['model']}**"
              + (f", exported to `{exp}`." if exp else ".")]
    if cv_rows:
        lines += ["", "5-fold cross-validation: see results_cv.csv"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
