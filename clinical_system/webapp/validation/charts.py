"""Charts for the Validation Lab page and the report (matplotlib, saved as PNG in <run>/charts/)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, TEAL, GREEN, RED, ORANGE, GREY = "#0077B6", "#00B4D8", "#2A9D8F", "#E63946", "#f08a4b", "#8aa0b0"
ABBR = {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d6df")
    ax.tick_params(colors="#3b4a56", labelsize=9)
    ax.grid(axis="y", color="#e3edf3", linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, run, name):
    fig.savefig(run.charts / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def t1_charts(run, res: dict) -> None:
    classes = [ABBR.get(c, c) for c in res["classes"]]
    cm = np.array(res["confusion"])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.55 else "#1d2b36")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted by the app")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (n = {res['n']}; colour = row share)", fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, run, "T1_confusion.png")

    per = res["per_class"]
    x = np.arange(len(per))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for k, (key, col) in enumerate([("precision", BLUE), ("recall", TEAL), ("f1", GREEN)]):
        ax.bar(x + (k - 1) * 0.26, [p[key] for p in per], 0.26, label=key.capitalize() if key != "f1" else "F1", color=col)
    ax.set_xticks(x, [ABBR.get(p["cls"], p["cls"]) for p in per])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Per-class precision, recall and F1", fontsize=10, pad=24)
    _style(ax)
    _save(fig, run, "T1_per_class.png")

    rel = res["reliability"]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1, label="Perfect calibration")
    conf = [b["conf"] for b in rel]
    acc = [b["acc"] for b in rel]
    sizes = [max(12, 220 * b["n"] / res["n"]) for b in rel]
    ax.plot(conf, acc, "-", color=BLUE, linewidth=1.4)
    ax.scatter(conf, acc, s=sizes, color=BLUE, zorder=3, label=f"App (ECE = {res['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted confidence (top class)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram ({res['ece_bins']} bins; dot size = images)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)
    _save(fig, run, "T1_reliability.png")


def t2_chart(run, out: dict) -> None:
    codes = list(out["app_pred_counts"])
    x = np.arange(len(codes))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - 0.2, [out["app_pred_counts"][c] for c in codes], 0.4, label="App", color=BLUE)
    ax.bar(x + 0.2, [out["npz_pred_counts"][c] for c in codes], 0.4, label="test_predictions.npz", color=ORANGE)
    ax.set_xticks(x, [ABBR.get(c, c) for c in codes])
    ax.set_ylabel("Images predicted")
    ax.set_title("Predicted classes on the test set: app vs. notebook file", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)
    _save(fig, run, "T2_classes.png")


def t3_chart(run, out: dict) -> None:
    names = [n for n in out["conditions"] if n != "original"]
    labels = {"blur": "Slight blur", "darker": "Darker −30%", "brighter": "Brighter +30%", "rotate90": "Rotate 90°",
              "rotate180": "Rotate 180°", "jpeg50": "JPEG q50", "phone_resize": "Phone resize", "crop90": "Crop 90%"}
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.bar(x - 0.2, [out["conditions"][n]["stability"] for n in names], 0.4, label="Same prediction as original", color=BLUE)
    ax.bar(x + 0.2, [out["conditions"][n]["accuracy"] for n in names], 0.4, label="Accuracy", color=TEAL)
    ax.axhline(out["conditions"]["original"]["accuracy"], color=GREEN, linewidth=1, linestyle=":",
               label="Accuracy of original images")
    ax.axhline(0.85, color=RED, linewidth=1, linestyle="--", label="85% criterion (average stability)")
    ax.set_xticks(x, [labels[n] for n in names], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"Robustness on {out['n']} test images", fontsize=10, pad=34)
    _style(ax)
    _save(fig, run, "T3_robustness.png")


def t4_chart(run, out: dict) -> None:
    st = out["stats"]
    keys = [("preprocess_ms", "Pre-processing"), ("predict_ms", "Prediction"), ("gradcam_ms", "Grad-CAM"),
            ("explain_ms", "Measurements"), ("pdf_ms", "PDF report"), ("lime_ms", "LIME"), ("result_ms", "Result shown"),
            ("full_ms", "Full analysis")]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.barh(y, [st[k]["median"] for k, _ in keys], 0.55, color=[TEAL] * 6 + [BLUE, BLUE], label="Median")
    ax.scatter([st[k]["p95"] for k, _ in keys], y, color=RED, marker="|", s=200, zorder=3, label="95th percentile")
    ax.set_yticks(y, [n for _, n in keys])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Seconds (log scale)")
    ax.set_title(f"Time per stage, {out['n']} images, CPU", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", color="#e3edf3")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, run, "T4_timing.png")


def t5_chart(run, out: dict, results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    seq = [r for r in results if r["phase"] == "sequential"]
    con = [r for r in results if r["phase"] == "concurrent"]
    ax.scatter([r["t"] for r in seq], [r["latency_s"] for r in seq], s=10, color=BLUE, label=f"{len(seq)} requests in a row")
    ax.scatter([r["t"] for r in con], [r["latency_s"] for r in con], s=14, color=ORANGE,
               label=f"{out['users']} users at the same time")
    bad = [r for r in results if not r["ok"]]
    if bad:
        ax.scatter([r["t"] for r in bad], [r["latency_s"] for r in bad], s=30, color=RED, marker="x", label="failed")
    ax.set_xlabel("Time since server start (s)")
    ax.set_ylabel("Response time (s)")
    ax2 = ax.twinx()
    if out["memory_trace"]:
        t, mb = zip(*out["memory_trace"])
        ax2.plot(t, mb, color=GREY, linewidth=1, alpha=0.6, label="Server memory (MB)")
        ax2.set_ylabel("Memory (MB)", color=GREY)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Reliability and load through the HTTP API (grey line: server memory)", fontsize=10, pad=22)
    _style(ax)
    _save(fig, run, "T5_load.png")


def t6_chart(run, out: dict) -> None:
    g = ["blurred", "dark", "tiny", "non_skin"]
    labels = ["Blurred", "Dark", "Tiny", "Non-skin"]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    v = [out["groups"][k]["rate"] for k in g]
    ax.bar(labels, v, color=[GREEN if x >= 0.9 else RED for x in v], width=0.55)
    for i, x in enumerate(v):
        ax.text(i, x + 0.02, f"{out['groups'][g[i]]['correct']}/{out['groups'][g[i]]['n']}", ha="center", fontsize=9)
    ax.axhline(0.9, color=RED, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Correctly warned")
    ax.set_title("Photo quality check", fontsize=10)
    _style(ax)
    _save(fig, run, "T6_quality.png")
