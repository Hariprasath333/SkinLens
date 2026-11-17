"""Classification metrics for T1 (all computed from the app's own outputs)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width bins on the top-class probability."""
    edges = np.linspace(0, 1, bins + 1)
    out, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            gap = abs(correct[sel].mean() - conf[sel].mean())
            total += sel.mean() * gap
            out.append(dict(lo=float(lo), hi=float(hi), n=int(sel.sum()), acc=float(correct[sel].mean()),
                            conf=float(conf[sel].mean())))
    return float(total), out


def classification(y_true: list[str], probs: np.ndarray, classes: list[str], levels: list[str],
                   bins: int = 15, bootstrap: int = 1000, seed: int = 0) -> dict:
    """y_true: class codes; probs: (n, k) in `classes` order (prior-corrected, as shown by the app)."""
    y = np.array([classes.index(c) for c in y_true])
    pred = probs.argmax(1)
    n, k = probs.shape
    correct = pred == y
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(k))
    per = []
    for i, c in enumerate(classes):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if tn + fp else float("nan")
        try:
            auc = roc_auc_score((y == i).astype(int), probs[:, i]) if 0 < (y == i).sum() < n else float("nan")
        except ValueError:
            auc = float("nan")
        per.append(dict(cls=c, support=int(s[i]), precision=float(p[i]), recall=float(r[i]), specificity=float(spec),
                        f1=float(f[i]), auc=float(auc)))
    present = [i for i in range(k) if (y == i).any()]
    bal = float(np.mean([r[i] for i in present]))
    macro = float(f1_score(y, pred, labels=present, average="macro", zero_division=0))
    e, rel = ece(probs.max(1), correct.astype(float), bins)
    mel = classes.index("MEL") if "MEL" in classes else None
    mel_recall = float(r[mel]) if mel is not None and (y == mel).any() else float("nan")
    by_level = {}
    for lv in ("High", "Medium", "Low"):
        sel = np.array([x == lv for x in levels])
        by_level[lv] = dict(n=int(sel.sum()), share=float(sel.mean()) if n else 0.0,
                            accuracy=float(correct[sel].mean()) if sel.any() else float("nan"))
    # 95% intervals: Wilson for accuracy, percentile bootstrap for the others
    rng = np.random.default_rng(seed)
    boot = dict(balanced_accuracy=[], macro_f1=[], mel_recall=[])
    for _ in range(bootstrap):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], pred[idx]
        pres = [i for i in range(k) if (yy == i).any()]
        rr = precision_recall_fscore_support(yy, pp, labels=range(k), zero_division=0)[1]
        boot["balanced_accuracy"].append(np.mean([rr[i] for i in pres]))
        boot["macro_f1"].append(f1_score(yy, pp, labels=pres, average="macro", zero_division=0))
        if mel is not None and (yy == mel).any():
            boot["mel_recall"].append(rr[mel])
    ci = {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
          for key, v in boot.items()}
    ci["accuracy"] = list(wilson(int(correct.sum()), n))
    return dict(n=int(n), accuracy=float(correct.mean()), balanced_accuracy=bal, macro_f1=macro, mel_recall=mel_recall,
                ece=e, reliability=rel, per_class=per, confusion=cm.tolist(), classes=classes, by_level=by_level,
                ci95=ci, ece_bins=bins)
