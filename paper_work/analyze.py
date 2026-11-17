"""
SkinLens paper analysis: turns results.npz files (from train_eval.py) into the
tables and figures a reviewer expects. Runs on a normal laptop CPU.

Needs only numpy. If matplotlib is installed, figures are also saved.

What it computes for every run
------------------------------
* accuracy, balanced accuracy, macro / weighted precision-recall-F1, per-class table
* per-class and macro ROC-AUC, melanoma sensitivity & specificity,
  malignant-vs-benign (mel+bcc+akiec) sensitivity / specificity / AUC
* 95% bootstrap confidence intervals (resampling whole lesions, not images)
* calibration: NLL, Brier, ECE (15 bins, same as notebook), class-wise ECE,
  before/after temperature scaling and after "bias-corrected" temperature scaling
  (fixes the class-prior shift caused by balanced sampling)
* selective prediction ("refer to a dermatologist when unsure"): risk-coverage, AURC
* conformal prediction sets with a coverage guarantee (overall and per class)
* effect of removing patient metadata and of test-time augmentation (if saved)
* McNemar significance test between every model and the reference model
* multi-seed mean +/- std when several seeds of one preset are given

Usage
-----
python analyze.py --runs runs_folder/*/results.npz --reference proposed --out paper_outputs
python analyze.py --legacy-npz test_predictions.npz --out paper_outputs   # notebook export, test only
"""
import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = [0, 1, 4]  # akiec, bcc, mel
MEL = 4
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# basic maths
# ----------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, y):
    z = logits - logits.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def confusion(y, p, k=7):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, p), 1)
    return cm


def prf(y, p, k=7):
    cm = confusion(y, p, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return prec, rec, f1, cm.sum(1)


def rankdata(a):
    """Average ranks for ties (same as scipy.stats.rankdata)."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    arr = a[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auc_binary(pos, score):
    pos = pos.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(pos, score):
    order = np.argsort(-score)
    pos = pos[order].astype(float)
    tpr = np.concatenate([[0], np.cumsum(pos) / max(pos.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - pos) / max((1 - pos).sum(), 1)])
    return fpr, tpr


def ece(probs, y, n_bins=15):
    """Same definition as the training notebook."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & ((conf < bins[i + 1]) if i < n_bins - 1 else (conf <= bins[i + 1]))
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def classwise_ece(probs, y, n_bins=15):
    k = probs.shape[1]
    out = []
    for c in range(k):
        pc, yc = probs[:, c], (y == c).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        e = 0.0
        for i in range(n_bins):
            m = (pc >= bins[i]) & ((pc < bins[i + 1]) if i < n_bins - 1 else (pc <= bins[i + 1]))
            if m.any():
                e += m.mean() * abs(yc[m].mean() - pc[m].mean())
        out.append(e)
    return float(np.mean(out))


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, 1)))


# ----------------------------------------------------------------------------
# calibration fits (on VALIDATION only)
# ----------------------------------------------------------------------------
def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    losses = [nll(logits / t, y) for t in grid]
    t = grid[int(np.argmin(losses))]
    lo, hi = t / 1.05, t * 1.05  # golden-section refine
    g = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(logits / a, y) < nll(logits / b, y):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)


def fit_bias_corrected_temperature(logits, y, iters=3000, lr=0.05):
    """Learns  p = softmax(a*z + b)  on validation by minimising NLL (convex in a, b).
    The bias b corrects the class-prior shift created by weighted sampling.
    Reference idea: Alexandari et al., ICML 2020 (BCTS)."""
    k = logits.shape[1]
    a, b = 1.0 / fit_temperature(logits, y), np.zeros(k)
    onehot = np.eye(k)[y]
    m_a = v_a = 0.0
    m_b, v_b = np.zeros(k), np.zeros(k)
    for t in range(1, iters + 1):  # Adam
        p = softmax(a * logits + b)
        g = (p - onehot) / len(y)
        ga, gb = float(np.sum(g * logits)), g.sum(0)
        m_a, v_a = 0.9 * m_a + 0.1 * ga, 0.999 * v_a + 0.001 * ga * ga
        m_b, v_b = 0.9 * m_b + 0.1 * gb, 0.999 * v_b + 0.001 * gb * gb
        a -= lr * (m_a / (1 - 0.9 ** t)) / (math.sqrt(v_a / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)
        b -= b.mean()
    return float(a), b


# ----------------------------------------------------------------------------
# metrics with confidence intervals
# ----------------------------------------------------------------------------
def point_metrics(probs, y):
    p = probs.argmax(1)
    prec, rec, f1, sup = prf(y, p)
    aucs = [auc_binary(y == c, probs[:, c]) for c in range(probs.shape[1])]
    mal_true = np.isin(y, MALIGNANT)
    mal_score = probs[:, MALIGNANT].sum(1)
    mal_pred = np.isin(p, MALIGNANT)
    w = sup / sup.sum()
    return {
        "accuracy": float((p == y).mean()),
        "balanced_accuracy": float(rec[sup > 0].mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * w).sum()),
        "macro_auc": float(np.nanmean(aucs)),
        "mel_sensitivity": float(rec[MEL]),
        "mel_specificity": float(((p != MEL) & (y != MEL)).sum() / max((y != MEL).sum(), 1)),
        "mel_auc": aucs[MEL],
        "malignant_sensitivity": float((mal_pred & mal_true).sum() / max(mal_true.sum(), 1)),
        "malignant_specificity": float((~mal_pred & ~mal_true).sum() / max((~mal_true).sum(), 1)),
        "malignant_auc": auc_binary(mal_true, mal_score),
        "per_class": {CLASS_NAMES[c]: dict(precision=float(prec[c]), recall=float(rec[c]), f1=float(f1[c]),
                                           auc=aucs[c], support=int(sup[c])) for c in range(len(sup))},
    }


CI_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "mel_sensitivity",
           "mel_specificity", "malignant_sensitivity", "malignant_auc"]


def bootstrap_ci(probs, y, groups=None, n_boot=1000):
    """Resamples lesions (groups) with replacement, because several images of one
    lesion are not independent."""
    if groups is None:
        groups = np.arange(len(y)).astype(str)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    vals = defaultdict(list)
    for _ in range(n_boot):
        pick = RNG.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        m = point_metrics(probs[idx], y[idx])
        for k in CI_KEYS:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def calibration_block(probs, y):
    return {"ece": ece(probs, y), "classwise_ece": classwise_ece(probs, y),
            "brier": brier(probs, y), "nll": float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))}


# ----------------------------------------------------------------------------
# selective prediction and conformal prediction
# ----------------------------------------------------------------------------
def risk_coverage(probs, y):
    conf = probs.max(1)
    order = np.argsort(-conf)
    err = (probs.argmax(1) != y)[order].astype(float)
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    out = {"aurc": float(np.mean(risk))}
    for c in (1.0, 0.9, 0.8, 0.7):
        n = max(int(round(c * len(y))), 1)
        sel = order[:n]
        out[f"acc@{int(c * 100)}%"] = float((probs[sel].argmax(1) == y[sel]).mean())
        mel_sel = sel[y[sel] == MEL]
        out[f"mel_sens@{int(c * 100)}%"] = (float((probs[mel_sel].argmax(1) == MEL).mean())
                                           if len(mel_sel) else float("nan"))
    return out, cov, risk


def conformal(val_probs, val_y, test_probs, test_y, alpha=0.1, per_class=False):
    """Split conformal prediction (LAC score = 1 - p_true).
    per_class=True gives class-conditional (Mondrian) coverage, important for rare classes."""
    k = val_probs.shape[1]

    def qhat(scores):
        n = len(scores)
        if n == 0:
            return 1.0
        lvl = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        return float(np.quantile(scores, lvl, method="higher"))

    s_val = 1 - val_probs[np.arange(len(val_y)), val_y]
    if per_class:
        q = np.array([qhat(s_val[val_y == c]) for c in range(k)])
        sets = (1 - test_probs) <= q[None, :]
    else:
        sets = (1 - test_probs) <= qhat(s_val)
    covered = sets[np.arange(len(test_y)), test_y]
    size = sets.sum(1)
    return {
        "alpha": alpha, "per_class": per_class,
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
        "class_coverage": {CLASS_NAMES[c]: float(covered[test_y == c].mean()) for c in range(k)
                           if (test_y == c).any()},
        "mel_in_set_for_true_mel": float(sets[test_y == MEL, MEL].mean()) if (test_y == MEL).any() else None,
    }


def mcnemar(y, pa, pb):
    a_right, b_right = pa == y, pb == y
    n01 = int((a_right & ~b_right).sum())
    n10 = int((~a_right & b_right).sum())
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "chi2": chi2, "p_value": float(math.erfc(math.sqrt(chi2 / 2)))}


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_run(path):
    d = np.load(path, allow_pickle=False)
    run = {k: d[k] for k in d.files}
    name = Path(path).parent.name
    m = re.match(r"(.+?)_seed(\d+)_fold(\d+)(.*)$", name)
    run["name"] = name
    run["group"] = (m.group(1) + m.group(4)) if m else name
    run["fold"] = int(m.group(3)) if m else 0
    return run


def load_legacy(path):
    """Notebook export (test_predictions.npz): test only, no validation logits."""
    d = np.load(path, allow_pickle=False)
    y = d["y_true"] if "y_true" in d.files else d["labels"]
    praw = d["test_probs_raw"] if "test_probs_raw" in d.files else d["probs_raw"]
    return {"name": "notebook_export", "group": "notebook_export", "fold": 0,
            "test_logits": np.log(np.clip(praw, 1e-12, 1)), "test_labels": y.astype(int)}


# ----------------------------------------------------------------------------
# per-run analysis
# ----------------------------------------------------------------------------
def analyze_run(run, n_boot):
    y = run["test_labels"].astype(int)
    tl = run["test_logits"]
    groups = run.get("test_lesion_ids")
    res = {"name": run["name"], "n_test": int(len(y))}
    raw = softmax(tl)
    res["raw"] = point_metrics(raw, y)
    res["raw_ci"] = bootstrap_ci(raw, y, groups, n_boot)
    res["calibration"] = {"raw": calibration_block(raw, y)}
    probs_for_downstream = raw
    if "val_logits" in run:
        vl, vy = run["val_logits"], run["val_labels"].astype(int)
        t = fit_temperature(vl, vy)
        ts = softmax(tl / t)
        a, b = fit_bias_corrected_temperature(vl, vy)
        bc = softmax(a * tl + b)
        res["temperature"] = t
        res["bcts"] = {"a": a, "bias": b.tolist()}
        res["calibration"]["val_raw"] = calibration_block(softmax(vl), vy)
        res["calibration"]["val_temp"] = calibration_block(softmax(vl / t), vy)
        res["calibration"]["temp_scaled"] = calibration_block(ts, y)
        res["calibration"]["bias_corrected"] = calibration_block(bc, y)
        res["bias_corrected_metrics"] = point_metrics(bc, y)
        probs_for_downstream = ts
        val_ts = softmax(vl / t)
        res["conformal"] = [conformal(val_ts, vy, ts, y, a_, pc) for a_ in (0.05, 0.10) for pc in (False, True)]
        if "val_logits_tta" in run and "test_logits_tta" in run:
            t2 = fit_temperature(run["val_logits_tta"], vy)
            res["tta"] = point_metrics(softmax(run["test_logits_tta"] / t2), y)
    res["selective"], cov, risk = risk_coverage(probs_for_downstream, y)
    if "test_logits_nometa" in run:
        res["no_metadata"] = point_metrics(softmax(run["test_logits_nometa"]), y)
    res["_curves"] = {"cov": cov, "risk": risk}
    res["_probs"] = probs_for_downstream
    res["_raw"] = raw
    res["_y"] = y
    return res


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.2f}"


def fmt_ci(m, ci, k):
    return f"{pct(m[k])} [{pct(ci[k][0])}, {pct(ci[k][1])}]"


def write_report(results, groups, reference, out):
    L = ["# SkinLens experiment report (auto-generated by analyze.py)", ""]
    L += ["## Table A - Main results on the held-out test set (95% lesion-level bootstrap CI)", "",
          "| Model | Accuracy | Balanced acc. | Macro-F1 | Macro-AUC | MEL sens. | MEL spec. |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        m, ci = r["raw"], r["raw_ci"]
        L.append(f"| {r['name']} | {fmt_ci(m, ci, 'accuracy')} | {fmt_ci(m, ci, 'balanced_accuracy')} | "
                 f"{fmt_ci(m, ci, 'macro_f1')} | {fmt_ci(m, ci, 'macro_auc')} | "
                 f"{fmt_ci(m, ci, 'mel_sensitivity')} | {fmt_ci(m, ci, 'mel_specificity')} |")
    if any(len(v) > 1 for v in groups.values()):
        L += ["", "## Table B - Mean +/- std over seeds / folds", "",
              "| Group | runs | Accuracy | Balanced acc. | Macro-F1 | Macro-AUC |", "|---|---|---|---|---|---|"]
        for g, rs in groups.items():
            cells = []
            for k in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]:
                v = np.array([r["raw"][k] for r in rs]) * 100
                cells.append(f"{v.mean():.2f} +/- {v.std(ddof=1) if len(v) > 1 else 0:.2f}")
            L.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    L += ["", "## Table C - Per-class results", ""]
    for r in results:
        L += [f"**{r['name']}**", "", "| Class | Precision | Recall | F1 | AUC | Support |", "|---|---|---|---|---|---|"]
        for c, v in r["raw"]["per_class"].items():
            L.append(f"| {c} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {pct(v['auc'])} | {v['support']} |")
        L.append("")
    L += ["## Table D - Calibration (lower is better)", "",
          "| Model | Setting | ECE | Class-wise ECE | Brier | NLL |", "|---|---|---|---|---|---|"]
    for r in results:
        for k, v in r["calibration"].items():
            L.append(f"| {r['name']} | {k} | {v['ece']:.4f} | {v['classwise_ece']:.4f} | {v['brier']:.4f} | {v['nll']:.4f} |")
    L += ["", "Settings: raw = softmax(logits); temp_scaled = global temperature fitted on validation; "
          "bias_corrected = temperature + per-class bias fitted on validation (fixes prior shift); "
          "val_* = the same on the validation set.", ""]
    bc = [r for r in results if "bias_corrected_metrics" in r]
    if bc:
        L += ["## Table E - Effect of bias-corrected calibration on decisions", "",
              "| Model | Accuracy raw -> BC | Macro-F1 raw -> BC | MEL sens. raw -> BC | MEL prec. raw -> BC |",
              "|---|---|---|---|---|"]
        for r in bc:
            a, b = r["raw"], r["bias_corrected_metrics"]
            L.append(f"| {r['name']} | {pct(a['accuracy'])} -> {pct(b['accuracy'])} | {pct(a['macro_f1'])} -> "
                     f"{pct(b['macro_f1'])} | {pct(a['mel_sensitivity'])} -> {pct(b['mel_sensitivity'])} | "
                     f"{pct(a['per_class']['mel']['precision'])} -> {pct(b['per_class']['mel']['precision'])} |")
        L.append("")
    L += ["## Table F - Selective prediction (model keeps only its most confident cases)", "",
          "| Model | AURC | Acc@100% | Acc@90% | Acc@80% | Acc@70% | MEL sens@80% |", "|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["selective"]
        L.append(f"| {r['name']} | {s['aurc']:.4f} | {pct(s['acc@100%'])} | {pct(s['acc@90%'])} | "
                 f"{pct(s['acc@80%'])} | {pct(s['acc@70%'])} | {pct(s['mel_sens@80%'])} |")
    cf = [r for r in results if "conformal" in r]
    if cf:
        L += ["", "## Table G - Conformal prediction sets", "",
              "| Model | alpha | per-class? | Coverage | Avg set size | Singletons | Min class coverage | True MEL kept in set |",
              "|---|---|---|---|---|---|---|---|"]
        for r in cf:
            for c in r["conformal"]:
                L.append(f"| {r['name']} | {c['alpha']} | {c['per_class']} | {pct(c['coverage'])} | "
                         f"{c['avg_set_size']:.2f} | {pct(c['singleton_rate'])} | "
                         f"{pct(min(c['class_coverage'].values()))} | {pct(c['mel_in_set_for_true_mel'])} |")
    nm = [r for r in results if "no_metadata" in r or "tta" in r]
    if nm:
        L += ["", "## Table H - Robustness: metadata removed at test time / test-time augmentation", "",
              "| Model | Macro-F1 (normal) | Macro-F1 (no metadata) | Macro-F1 (TTA) | Bal. acc (normal) | Bal. acc (no metadata) |",
              "|---|---|---|---|---|---|"]
        for r in nm:
            n = r.get("no_metadata", {})
            L.append(f"| {r['name']} | {pct(r['raw']['macro_f1'])} | {pct(n.get('macro_f1'))} | "
                     f"{pct(r.get('tta', {}).get('macro_f1'))} | {pct(r['raw']['balanced_accuracy'])} | "
                     f"{pct(n.get('balanced_accuracy'))} |")
    refs = [r for r in results if r["name"].startswith(reference)] if reference else []
    if refs:
        ref = refs[0]
        L += ["", f"## Table I - McNemar test vs reference `{ref['name']}` (same test images)", "",
              "| Model | ref right / other wrong | ref wrong / other right | p-value |", "|---|---|---|---|"]
        for r in results:
            if r is ref or len(r["_y"]) != len(ref["_y"]) or not np.array_equal(r["_y"], ref["_y"]) or r.get("fold") != ref.get("fold"):
                continue
            t = mcnemar(ref["_y"], ref["_raw"].argmax(1), r["_raw"].argmax(1))
            L.append(f"| {r['name']} | {t['n01']} | {t['n10']} | {t['p_value']:.4g} |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # LaTeX version of Table A for the paper
    T = [r"\begin{table}[t]\centering\caption{Results on the HAM10000 test split (95\% CI).}",
         r"\begin{tabular}{lcccc}\hline", r"Model & Acc. & Bal. Acc. & Macro-F1 & Macro-AUC \\ \hline"]
    for r in results:
        m = r["raw"]
        T.append(f"{r['name'].replace('_', chr(92) + '_')} & {100*m['accuracy']:.2f} & {100*m['balanced_accuracy']:.2f} & "
                 f"{100*m['macro_f1']:.2f} & {100*m['macro_auc']:.2f} \\\\")
    T += [r"\hline\end{tabular}\end{table}"]
    (out / "table_main.tex").write_text("\n".join(T) + "\n", encoding="utf-8")


def make_figures(results, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figures")
        return
    for r in results:
        y, raw, ts = r["_y"], r["_raw"], r["_probs"]
        name = r["name"]
        # confusion matrix (row-normalised)
        cm = confusion(y, raw.argmax(1)).astype(float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(7), CLASS_NAMES); ax.set_yticks(range(7), CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.colorbar(im); fig.tight_layout()
        fig.savefig(out / f"{name}_confusion.png", dpi=300); plt.close(fig)
        # reliability diagram
        fig, ax = plt.subplots(figsize=(5, 5))
        for label, p in [("raw", raw), ("temperature-scaled", ts)]:
            conf, corr = p.max(1), (p.argmax(1) == y)
            bins = np.linspace(0, 1, 16)
            xs, ys_ = [], []
            for i in range(15):
                m = (conf >= bins[i]) & (conf < bins[i + 1] if i < 14 else conf <= bins[i + 1])
                if m.sum() >= 5:
                    xs.append(conf[m].mean()); ys_.append(corr[m].mean())
            ax.plot(xs, ys_, marker="o", label=f"{label} (ECE={ece(p, y):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.legend(); ax.set_title(name)
        fig.tight_layout(); fig.savefig(out / f"{name}_reliability.png", dpi=300); plt.close(fig)
        # ROC per class
        fig, ax = plt.subplots(figsize=(5, 5))
        for c in range(7):
            f, t = roc_curve(y == c, raw[:, c])
            ax.plot(f, t, label=f"{CLASS_NAMES[c]} (AUC={r['raw']['per_class'][CLASS_NAMES[c]]['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(out / f"{name}_roc.png", dpi=300); plt.close(fig)
    # risk-coverage, all models in one plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        ax.plot(r["_curves"]["cov"], r["_curves"]["risk"], label=r["name"])
    ax.set_xlabel("Coverage (fraction of cases the model answers)"); ax.set_ylabel("Error rate")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "risk_coverage.png", dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="results.npz files or glob patterns")
    ap.add_argument("--legacy-npz", default=None, help="test_predictions.npz exported by the notebook")
    ap.add_argument("--reference", default="proposed", help="run-name prefix used as reference in McNemar")
    ap.add_argument("--out", default="paper_outputs")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for pat in args.runs for p in glob.glob(pat)})
    runs = [load_run(p) for p in paths]
    if args.legacy_npz:
        runs.append(load_legacy(args.legacy_npz))
    if not runs:
        raise SystemExit("No runs found. Pass --runs path/to/*/results.npz or --legacy-npz file.")

    results = []
    for run in runs:
        print("Analysing", run["name"])
        r = analyze_run(run, args.n_boot)
        r["fold"] = run["fold"]
        results.append(r)
    groups = defaultdict(list)
    for run, r in zip(runs, results):
        groups[run["group"]].append(r)

    write_report(results, groups, args.reference, out)
    make_figures(results, out)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (out / "metrics.json").write_text(json.dumps(clean, indent=2, default=float), encoding="utf-8")
    print("Wrote", out / "REPORT.md", "and", out / "metrics.json")


if __name__ == "__main__":
    main()
