"""TRL-4 laboratory validation report (A4 PDF) built from one run folder."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .. import config

BLUE = colors.HexColor("#0077B6")
TEAL = colors.HexColor("#00B4D8")
GREEN = colors.HexColor("#2A9D8F")
RED = colors.HexColor("#E63946")
AMBER = colors.HexColor("#c27c0e")
LIGHT = colors.HexColor("#F5FAFD")
LINE = colors.HexColor("#e3edf3")
TEXT = colors.HexColor("#1d2b36")
MUTED = colors.HexColor("#5f7383")
LOGO = config.BASE_DIR / "static" / "icons" / "logo-256.png"
REPORT_NAME = "SkinLens_TRL4_Lab_Validation_Report.pdf"
TEAM = [("Hari Prasath M", "23BCE8999"), ("Kodela Kailash Nadh", "23BCE7748"), ("Dharun Raj B", "23BCE9271")]
GUIDE = "Dr. B V Gokulnath"
SCHOOL = "School of Computer Science and Engineering (SCOPE), VIT-AP University"
NAMES = {"T1": "Clinical accuracy (lab)", "T2": "Integration agreement", "T3": "Robustness", "T4": "Speed",
         "T5": "Reliability and load", "T6": "Photo quality check", "T7": "Functional and security tests"}
ABBR = {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}


def pct(v, d=1):
    return "–" if v is None or v != v else f"{100 * v:.{d}f}%"


def verdicts(run) -> dict:
    """Verdict per test: PASS | FAIL | REPORTED | NOT RUN | ERROR."""
    st = run.state()
    out = {}
    for t in NAMES:
        res = run.load_json(f"{t}_metrics.json")
        if res is not None:
            out[t] = dict(verdict=res.get("verdict", "?"), text=res.get("verdict_text", ""))
        elif st.get(t, {}).get("status") == "failed":
            out[t] = dict(verdict="ERROR", text=st[t].get("error", ""))
        else:
            out[t] = dict(verdict="NOT RUN", text="not part of this run" if t not in st else st[t].get("status", ""))
    return out


def trl_conclusion(v: dict) -> tuple[bool, str]:
    failed = [t for t, x in v.items() if x["verdict"] in ("FAIL", "ERROR")]
    pending = [t for t, x in v.items() if x["verdict"] in ("NOT RUN", "REPORTED")]
    if not failed and not pending:
        return True, ("All acceptance criteria were met in the laboratory. The integrated SkinLens system "
                      "(model, web application, phone app, explanations, reports) is validated at TRL 4.")
    parts = []
    if failed:
        parts.append("criteria not met: " + ", ".join(f"{t} ({NAMES[t]})" for t in failed))
    if pending:
        parts.append("not yet decided: " + ", ".join(
            f"{t} ({'full 2,014-image run needed' if t == 'T1' and v[t]['verdict'] == 'REPORTED' else 'not run'})"
            for t in pending))
    return False, ("TRL 4 is NOT yet demonstrated by this run, because " + "; ".join(parts) +
                   ". The tests that passed show that the integrated system works in the laboratory, but the "
                   "TRL-4 claim requires every criterion to pass.")


def _arch_diagram() -> Drawing:
    d = Drawing(174 * mm, 62 * mm)
    boxes = [(0, 34, "Phone app (PWA)\nor web browser", TEAL), (0, 6, "Doctor mode /\npatient mode", TEAL),
             (46, 20, "FastAPI web server\nlogin, CSRF, rate limit", BLUE), (92, 34, "Inference\npre-processing + metadata\nSkinLens hybrid + prior corr.", BLUE),
             (92, 6, "Explanations\nGrad-CAM, LIME,\nmeasurements, quality", BLUE), (138, 34, "SQLite database\n+ image storage", GREEN),
             (138, 6, "PDF reports\n(ReportLab)", GREEN)]
    for x, y, txt, col in boxes:
        d.add(Rect(x * mm, y * mm, 36 * mm, 22 * mm, rx=6, ry=6, fillColor=col, strokeColor=None))
        lines = txt.split("\n")
        for i, ln in enumerate(lines):
            d.add(String((x + 18) * mm, (y + 11 + (len(lines) / 2 - i - 0.75) * 4.2) * mm, ln, fontName="Helvetica-Bold"
                         if i == 0 else "Helvetica", fontSize=7.6 if i == 0 else 6.8, fillColor=colors.white,
                         textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=MUTED, strokeWidth=1.2))
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        p = [(x2, y2), (x2 - 2.6 * math.cos(a - 0.4), y2 - 2.6 * math.sin(a - 0.4)),
             (x2 - 2.6 * math.cos(a + 0.4), y2 - 2.6 * math.sin(a + 0.4))]
        d.add(Polygon([c * mm for pt in p for c in pt], fillColor=MUTED, strokeColor=None))
    arrow(36, 45, 46, 34)
    arrow(36, 17, 46, 27)
    arrow(82, 34, 92, 44)
    arrow(82, 27, 92, 17)
    arrow(128, 45, 138, 45)
    arrow(128, 17, 138, 17)
    return d


def app_versions(run) -> str:
    """The app version at the start of the run, plus later changes (CHANGES.json)."""
    v = run.info["identity"]["app_version"]
    changes = [c for c in (run.load_json("CHANGES.json") or []) if c["from_version"] != c["to_version"]]
    if not changes:
        return v
    return f"{v} at the start → " + " → ".join(f"{c['to_version']} from {c['time'][:16].replace('T', ' ')}" for c in changes) +         " (photo quality check only; see Changes during this run)"


def build_report(run) -> Path:
    info = run.info
    ident = info["identity"]
    crit = info["criteria"]
    v = verdicts(run)
    trl_ok, conclusion = trl_conclusion(v)
    out = run.dir / REPORT_NAME
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.2, textColor=TEXT)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=MUTED)
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=BLUE,
                        spaceBefore=4, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=BLUE,
                        spaceBefore=8, spaceAfter=4)
    cell = ParagraphStyle("c", parent=body, fontSize=8.4, leading=10.6)
    cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")
    W = 174 * mm

    def table(rows, widths, head=True, zebra=True, font=8.4):
        t = Table([[Paragraph(str(c), cellb if (head and i == 0) else cell) if not hasattr(c, "wrap") else c
                    for c in r] for i, r in enumerate(rows)], colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
        st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
              ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
        if head:
            st += [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLUE)]
        if zebra:
            st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafdff")])]
        t.setStyle(TableStyle(st))
        return t

    def badge(text):
        col = {"PASS": GREEN, "FAIL": RED, "ERROR": RED, "REPORTED": AMBER}.get(text, MUTED)
        t = Table([[Paragraph(f"<b>{text}</b>", ParagraphStyle("bd", parent=cell, textColor=colors.white,
                                                                alignment=TA_CENTER))]], colWidths=[22 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def chart(name, width_mm, cap=""):
        p = run.charts / name
        if not p.exists():
            return Paragraph(f"(chart {name} not available)", small)
        from PIL import Image as PI
        w, h = PI.open(p).size
        img = Image(str(p), width=width_mm * mm, height=width_mm * mm * h / w)
        return KeepTogether([img, Paragraph(cap, small)]) if cap else img

    def page_deco(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            if LOGO.exists():
                canvas.drawImage(str(LOGO), 18 * mm, A4[1] - 15 * mm, 7 * mm, 7 * mm, mask="auto")
            canvas.setFont("Helvetica-Bold", 8.5)
            canvas.setFillColor(BLUE)
            canvas.drawString(27 * mm, A4[1] - 12.4 * mm, "SkinLens – TRL-4 Laboratory Validation Report")
            canvas.setStrokeColor(LINE)
            canvas.line(18 * mm, A4[1] - 17 * mm, A4[0] - 18 * mm, A4[1] - 17 * mm)
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, "SkinLens – for research and decision support only. Not a medical diagnosis.")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Run {run.dir.name}  ·  page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=22 * mm,
                            bottomMargin=17 * mm, title="SkinLens TRL-4 Laboratory Validation Report",
                            author="SkinLens team, VIT-AP University")
    S = []
    # ------------------------------------------------------------------ cover
    created = datetime.fromisoformat(info["created"])
    S += [Spacer(1, 18 * mm)]
    if LOGO.exists():
        S += [Image(str(LOGO), width=30 * mm, height=30 * mm)]
    S += [Spacer(1, 6 * mm),
          Paragraph("SkinLens", ParagraphStyle("t0", parent=h1, fontSize=30, leading=34, alignment=TA_CENTER)),
          Paragraph("TRL-4 Laboratory Validation Report", ParagraphStyle("t1", parent=h1, fontSize=17, leading=22,
                                                                         alignment=TA_CENTER, textColor=TEXT)),
          Paragraph("Integrated skin-lesion decision-support system validated in a laboratory environment",
                    ParagraphStyle("t2", parent=body, alignment=TA_CENTER, textColor=MUTED, fontSize=10.5)),
          Spacer(1, 12 * mm)]
    col = GREEN if trl_ok else AMBER
    verdict_box = Table([[Paragraph(("<b>Result: TRL 4 demonstrated</b>" if trl_ok else
                                     "<b>Result: TRL 4 not yet demonstrated by this run</b>") +
                                    "<br/>" + " · ".join(f"{t} {x['verdict']}" for t, x in v.items()),
                                    ParagraphStyle("vb", parent=body, textColor=colors.white, alignment=TA_CENTER,
                                                   fontSize=10, leading=15))]], colWidths=[150 * mm])
    verdict_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col), ("TOPPADDING", (0, 0), (-1, -1), 10),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    S += [verdict_box, Spacer(1, 12 * mm)]
    finished = datetime.now()
    date_txt = f"{created:%d %B %Y}" if finished.date() == created.date() else f"{created:%d} – {finished:%d %B %Y}"         if (finished.month, finished.year) == (created.month, created.year) else f"{created:%d %B %Y} – {finished:%d %B %Y}"
    meta = [["Run", run.dir.name], ["Date", date_txt], ["App version", app_versions(run)],
            ["Model", ident["model"]["name"]],
            ["Model file / SHA-256", "<br/>".join(f"{f['file']}<br/><font size=7>{f['sha256']}</font>" for f in ident["model"]["files"]) or "–"],
            ["Team", "<br/>".join(f"{n} ({r})" for n, r in TEAM)], ["Guide", GUIDE], ["Institution", SCHOOL]]
    S += [table(meta, [38, 112], head=False, zebra=False), PageBreak()]

    # ------------------------------------------------------------------ 1 purpose
    S += [Paragraph("1. Purpose and TRL 4 definition", h1),
          Paragraph("This report documents the laboratory validation of the integrated SkinLens system. "
                    "<b>Technology Readiness Level 4</b> is defined as <i>component and/or system validation in a "
                    "laboratory environment</i>: the basic technological components are integrated and shown to work "
                    "together, tested against defined acceptance criteria with representative data in the lab. TRL 5 "
                    "requires validation in a <i>relevant</i> environment (for SkinLens: real clinics and smartphone "
                    "photos).", body),
          Paragraph("All tests call the application's own code: the inference function used for real patients "
                    "(<font face='Courier'>webapp.analysis.compute</font>: same decoding, pre-processing, metadata "
                    "encoding, prior correction alpha = 0.3 and confidence levels), the real HTTP API and the real PDF "
                    "builder. No test image or result was stored as a patient record. The acceptance criteria were "
                    f"fixed before the run and frozen in <font face='Courier'>run.json</font> (SHA-256 "
                    f"<font size=7>{crit['sha256']}</font>).", body)]
    # ------------------------------------------------------------------ 2 system
    S += [Paragraph("2. System description and architecture", h1),
          Paragraph("SkinLens analyses a photo of a skin lesion and returns the most likely of seven HAM10000 classes "
                    "with calibrated, prior-corrected probabilities, a confidence level, visual explanations (Grad-CAM, "
                    "LIME), plain-language reasons from image measurements, a photo-quality check and a PDF report. It runs "
                    "as a web application on a laptop and as an installable phone app (PWA) with a doctor mode (login, "
                    "patient records) and a patient mode (no login, nothing stored).", body),
          Spacer(1, 3 * mm), _arch_diagram(),
          Paragraph("Figure 1. Architecture of the integrated system tested in this report.", small)]
    # ------------------------------------------------------------------ 3 identity
    env = ident["environment"]
    libs = ", ".join(f"{k} {val}" for k, val in env["libraries"].items() if val != "not installed")
    m = ident["model"]
    S += [Paragraph("3. System identity", h1),
          table([["Item", "Value"], ["App version", app_versions(run)],
                 ["Model", m["name"]],
                 ["Model weights", "<br/>".join(f"{f['file']} ({f['size_mb']} MB, modified {f['modified']})<br/>"
                                                f"SHA-256 <font size=7>{f['sha256']}</font>" for f in m["files"])],
                 ["Prior correction", f"alpha = {m['prior_alpha']} (model_settings.json SHA-256 "
                                      f"<font size=7>{m['settings_sha256'][:32]}…</font>)"],
                 ["Confidence levels", f"High &gt; {100 * m['confidence_levels']['high_above']:.0f}%, Medium "
                                       f"{100 * m['confidence_levels']['medium_from']:.0f}–{100 * m['confidence_levels']['high_above']:.0f}%, "
                                       f"Low below"],
                 ["Classes", ", ".join(ABBR.get(c, c) for c in m["classes"])],
                 ["Git commit", ident["git_commit"]], ["Recorded", ident["recorded"]],
                 ["Operating system", env["os"]], ["CPU", f"{env['cpu']} ({env['cpu_cores']} logical cores)"],
                 ["RAM", f"{env['ram_gb']} GB"], ["GPU", env["gpu"]], ["Python", env["python"]],
                 ["Libraries", libs]], [38, 136])]
    # ------------------------------------------------------------------ 4 environment and data
    t1 = run.load_json("T1_metrics.json")
    S += [Paragraph("4. Test environment and data", h1),
          Paragraph("<b>Environment.</b> The application ran on the laptop described above (CPU only), exactly as "
                    "installed for users (same virtual environment, same model file and settings). The load test used "
                    "a separate server process of the same application with a throw-away data folder.", body),
          Paragraph("<b>Data.</b> HAM10000 dermoscopic images (10,015 images of 7,470 lesions, 7 classes). The test set "
                    "is the project's <b>lesion-grouped</b> split: 2,014 images of 1,495 lesions, and no lesion (and so no "
                    "test image) was used in training or model selection. Class mix of the test set: nv 1,363, mel 222, "
                    "bkl 205, bcc 100, akiec 68, vasc 34, df 22. Patient age, sex and body site from the HAM10000 "
                    "metadata were passed to the app as a doctor would enter them. Samples (T1 quick run, T3–T6) are "
                    f"proportional stratified samples with a fixed random seed ({crit['settings']['seed']}).", body)]
    # ------------------------------------------------------------------ 5 protocol
    S += [Paragraph("5. Test protocol and acceptance criteria", h1),
          table([["Test", "What is tested", "Acceptance criterion (fixed before the run)"]] +
                [[t, crit["criteria"][t]["name"], crit["criteria"][t]["criterion"]] for t in NAMES], [12, 40, 122]),
          *([Paragraph("<b>Changes during this run.</b>", body),
             table([["Time", "Version", "Change", "Effect on tests"]] +
                   [[c["time"].replace("T", " ")[:16], (f"{c['from_version']} → {c['to_version']}" if c["from_version"] != c["to_version"]
                                           else f"{c['to_version']} (runner only)"),
                     c["change"] + " " + c["developed_on"], c["affects"]] for c in run.load_json("CHANGES.json")],
                   [22, 20, 82, 50])] if run.load_json("CHANGES.json") else []),
          Paragraph(f"Reference values (training notebook): accuracy 85.80% (85.45% before prior correction), "
                    f"macro-F1 0.763, melanoma recall 78.8%, ECE 0.070 – {crit['reference']['source']}.", small),
          PageBreak()]
    # ------------------------------------------------------------------ 6 results
    S += [Paragraph("6. Results", h1)]
    # T1
    S += [Paragraph(f"T1 – Clinical accuracy ({'full test set' if t1 and t1['mode'] == 'full' else '300-image stratified sample'})", h2)]
    if t1:
        ci = t1["ci95"]
        S += [table([["Metric", "App (this run)", "95% CI", "Notebook reference"],
                     ["Images", str(t1["n"]), "", "2,014"],
                     ["Accuracy", pct(t1["accuracy"], 2), f"{pct(ci['accuracy'][0])} – {pct(ci['accuracy'][1])}", "85.80%"],
                     ["Balanced accuracy", pct(t1["balanced_accuracy"], 2),
                      f"{pct(ci['balanced_accuracy'][0])} – {pct(ci['balanced_accuracy'][1])}", "–"],
                     ["Macro-F1", f"{t1['macro_f1']:.3f}", f"{ci['macro_f1'][0]:.3f} – {ci['macro_f1'][1]:.3f}", "0.763"],
                     ["Melanoma recall", pct(t1["mel_recall"]), f"{pct(ci['mel_recall'][0])} – {pct(ci['mel_recall'][1])}", "78.8%"],
                     ["ECE (15 bins)", f"{t1['ece']:.3f}", "", "0.070"]], [44, 34, 50, 46]),
              Spacer(1, 2 * mm),
              table([["Class", "n", "Precision", "Recall", "Specificity", "F1", "AUC"]] +
                    [[ABBR.get(p["cls"], p["cls"]), p["support"], f"{p['precision']:.3f}", f"{p['recall']:.3f}",
                      f"{p['specificity']:.3f}", f"{p['f1']:.3f}", "–" if p["auc"] != p["auc"] else f"{p['auc']:.3f}"]
                     for p in t1["per_class"]], [24, 16, 26, 26, 28, 26, 28]),
              Spacer(1, 2 * mm),
              table([["Confidence level", "Images", "Share", "Accuracy"]] +
                    [[lv, d["n"], pct(d["share"]), pct(d["accuracy"])] for lv, d in t1["by_level"].items()], [44, 30, 40, 60]),
              Table([[chart("T1_confusion.png", 84), chart("T1_reliability.png", 76)]], colWidths=[90 * mm, 84 * mm]),
              chart("T1_per_class.png", 150, "Figure 2. T1 confusion matrix, reliability diagram and per-class metrics."),
              Paragraph(f"<b>Verdict: {t1['verdict']}.</b> {t1['verdict_text']} Per-class AUC with very few images "
                        "(e.g. df) is imprecise.", body)]
    else:
        S += [Paragraph("Not run.", body)]
    # T2
    t2 = run.load_json("T2_metrics.json")
    S += [Paragraph("T2 – Integration agreement (app vs. training notebook)", h2)]
    if t2:
        S += [Paragraph(t2["compared"], body),
              table([["Quantity", "Value"],
                     ["Images matched by image_id", f"{t2['n_matched']} of {t2['n_manifest']} (npz: {t2['n_npz']})"],
                     ["Identical predicted class", f"{pct(t2['agreement'], 2)} ({t2['n_disagree']} differ)"],
                     ["Largest / mean probability difference (corrected)", f"{t2['max_prob_diff']:.4f} / {t2['mean_prob_diff']:.4f}"],
                     ["Largest difference of raw probabilities", f"{t2['max_raw_prob_diff']:.4f}"],
                     ["alpha (npz / app); max log-prior difference", f"{t2['npz_alpha']} / {t2['app_alpha']}; {t2['log_prior_max_diff']:.2e}"],
                     ["npz self-consistency (app correction of probs_raw vs probs_corrected)", f"max diff {t2['npz_correction_self_consistency']:.2e}"],
                     ["Accuracy vs. true labels: app / npz", f"{pct(t2['accuracy_app'], 2)} / {pct(t2['accuracy_npz'], 2)}"],
                     ["Mean top-class probability: app / npz", f"{t2['app_mean_top_prob']:.3f} / {t2['npz_mean_top_prob']:.3f}"]],
                    [100, 74]),
              chart("T2_classes.png", 120, "Figure 3. Predicted classes on the full test set: app vs. test_predictions.npz."),
              Paragraph(f"<b>Verdict: {t2['verdict']}.</b> {t2['verdict_text']}.", body),
              Paragraph("Note: the test_predictions.npz provided first did not contain the trained model's outputs "
                        "(near-uniform probabilities, 58.4% accuracy). It was replaced by a new export from the trained "
                        "checkpoint (85.45% raw / 85.80% corrected, macro-F1 0.763) before this T2 result; the SHA-256 "
                        "above identifies the file used.", small)]
        attempts = sorted(run.dir.glob("T2_metrics_attempt*.json"))
        if attempts:
            rows = [["Attempt", "Reference file", "Identical class", "npz accuracy", "Verdict"]]
            for f in attempts:
                a = json.loads(f.read_text(encoding="utf-8"))
                rows.append([f.stem.split("_")[-1], (a["npz_sha256"][:16] if a.get("npz_sha256") else "not recorded"),
                             pct(a["agreement"], 2), pct(a["accuracy_npz"]), a["verdict"]])
            rows.append(["final", t2.get("npz_sha256", "")[:16], pct(t2["agreement"], 2), pct(t2["accuracy_npz"]), t2["verdict"]])
            bad = [json.loads(f.read_text(encoding="utf-8")) for f in attempts]
            bad = [a for a in bad if a["accuracy_npz"] < 0.7]
            why = (f"The earlier test_predictions.npz did not contain the trained model's outputs (near-uniform "
                   f"probabilities, {pct(bad[0]['accuracy_npz'])} accuracy). The team exported it again from the trained "
                   "checkpoint and T2 was run again. " if bad else "")
            S += [Paragraph(f"<b>History.</b> {why}The acceptance criterion was not changed. All T2 attempts of this run:",
                            body),
                  table(rows, [20, 40, 34, 34, 46])]
        # earlier runs of an older app version (kept as a record) that led to a change
        for old in sorted(run.dir.parent.iterdir()):
            if old == run.dir or not (old / "SUPERSEDED.txt").exists() or not (old / "T2_metrics.json").exists():
                continue
            o = json.loads((old / "T2_metrics.json").read_text(encoding="utf-8"))
            oi = json.loads((old / "run.json").read_text(encoding="utf-8"))
            od = json.loads((old / "T2_diagnosis.json").read_text(encoding="utf-8")) if (old / "T2_diagnosis.json").exists() else None
            S += [Paragraph(f"<b>Earlier run {old.name} (app {oi['identity']['app_version']}).</b> T2 {o['verdict']}: "
                            f"{pct(o['agreement'], 2)} identical class with the same reference file "
                            f"(SHA-256 {o.get('npz_sha256', '')[:16]}…). " + (od["conclusion"] + " " if od else "") +
                            f"The app was changed accordingly (now app {ident['app_version']}: resize with cv2 INTER_LINEAR "
                            "as in training) and all tests were run again in this run; the acceptance criteria were not changed.",
                            body)]
        diag = run.load_json("T2_diagnosis.json")
        if diag:
            S += [Paragraph("<b>Root-cause analysis (after the result; diagnostic only, the app was not changed).</b> "
                            f"{diag['method']}, on {diag['images']}:", body),
                  table([["Resize to 384 px", "Same class", "Median / max probability difference"]] +
                        [[r["resize"], r["class_agreement"], f"{r['median_max_abs_diff']:.4f} / {r['max_abs_diff']:.4f}"]
                         for r in diag["results"]], [64, 30, 80]),
                  Paragraph(diag["conclusion"], body)]
        if t2["verdict"] == "FAIL" and t2["accuracy_npz"] < t2["accuracy_app"] - 0.1:  # wrong reference file
            S += [Paragraph(
                "<b>Interpretation.</b> The provided test_predictions.npz scores far below the reference accuracy "
                f"({pct(t2['accuracy_npz'])} vs. 85.80%) and its probabilities are almost uniform (mean top-class "
                f"probability {t2['npz_mean_top_prob']:.2f}), while the app reproduces the expected accuracy. The file "
                "therefore does not contain the outputs of the trained model (it looks like the output of a network "
                "whose trained weights were not loaded). The disagreement is a problem of the reference file, not "
                "evidence that the app differs from the trained model; but the criterion is not met with this file. "
                "T2 must be re-run with predictions exported from the trained checkpoint.", body)]
    else:
        S += [Paragraph("Not run.", body)]
    # T3
    t3 = run.load_json("T3_metrics.json")
    S += [Paragraph("T3 – Robustness to realistic photo conditions", h2)]
    if t3:
        lab = {"original": "Original", "blur": f"Slight blur (Gaussian sigma {t3['settings']['t3_blur_sigma']} px)",
               "darker": "Darker (−30%)", "brighter": "Brighter (+30%)", "rotate90": "Rotation 90°",
               "rotate180": "Rotation 180°", "jpeg50": "JPEG quality 50",
               "phone_resize": "Phone resize (1024 px and back)", "crop90": "Off-centre crop (90%)"}
        S += [table([["Condition", "Accuracy", "Same prediction as original"]] +
                    [[lab[k], pct(d["accuracy"]), pct(d["stability"])] for k, d in t3["conditions"].items()], [80, 40, 54]),
              chart("T3_robustness.png", 150, f"Figure 4. Robustness on {t3['n']} stratified test images."),
              Paragraph(f"<b>Verdict: {t3['verdict']}.</b> {t3['verdict_text']} (criterion: at least 85%).", body)]
    else:
        S += [Paragraph("Not run.", body)]
    # T4
    t4 = run.load_json("T4_metrics.json")
    S += [Paragraph("T4 – Speed", h2)]
    if (run.dir / "T4_note.txt").exists():
        S += [Paragraph("<b>Measurement conditions.</b> " + (run.dir / "T4_note.txt").read_text(encoding="utf-8").strip(), body)]
    if t4:
        lab = {"decode_ms": "Image decoding", "preprocess_ms": "Pre-processing (decode, quality check, resize)",
               "predict_ms": "Prediction", "gradcam_ms": "Grad-CAM", "explain_ms": "Measurements and text",
               "lime_ms": f"LIME ({t4['lime_samples']} samples)", "pdf_ms": "PDF report",
               "result_ms": "Result page ready (all of the above except LIME/PDF)", "full_ms": "Full analysis incl. LIME and PDF",
               "predict_plus_gradcam": "Prediction + Grad-CAM (criterion)"}
        S += [table([["Stage", "Mean (s)", "Median (s)", "95th pct (s)", "Max (s)"]] +
                    [[lab[k], f"{d['mean']:.2f}", f"{d['median']:.2f}", f"{d['p95']:.2f}", f"{d['max']:.2f}"]
                     for k, d in t4["stats"].items()], [82, 23, 23, 23, 23]),
              chart("T4_timing.png", 140, f"Figure 5. Time per stage on {t4['n']} images (CPU only)."),
              Paragraph(f"<b>Verdict: {t4['verdict']}.</b> {t4['verdict_text']}. LIME runs in the background after the "
                        "result is shown.", body)]
    else:
        S += [Paragraph("Not run.", body)]
    # T5
    t5 = run.load_json("T5_metrics.json")
    S += [Paragraph("T5 – Reliability and load (real HTTP API)", h2)]
    if t5:
        q = t5["queue_wait_s"]
        S += [Paragraph(f"Endpoint: {t5['endpoint']}. A separate server process was started from the same code, then "
                        f"{t5['sequential'].get('n', 0)} analyses were sent one after another, followed by {t5['users']} users "
                        "sending requests at the same time (httpx).", body),
              table([["Measure", "Value"], ["Requests / failed", f"{t5['requests']} / {t5['failed']}"],
                     ["Server crashed", "yes" if t5["crashed"] else "no"],
                     ["Errors", ", ".join(t5["errors"]) or "none"],
                     ["Response time in a row (median / p95)", f"{t5['sequential']['median']:.2f} s / {t5['sequential']['p95']:.2f} s"],
                     ["Response time, users at once (median / p95 / max)",
                      f"{t5['concurrent']['median']:.2f} / {t5['concurrent']['p95']:.2f} / {t5['concurrent']['max']:.2f} s"],
                     ["Queue waiting time (mean / p95 / max)", f"{q['mean']:.2f} / {q['p95']:.2f} / {q['max']:.2f} s"],
                     ["Server memory (start / peak / end)",
                      f"{t5['memory_mb']['start']:.0f} / {t5['memory_mb']['peak']:.0f} / {t5['memory_mb']['end']:.0f} MB"],
                     ["Stored on the test server afterwards", f"database rows {t5['stored_after_test']['database_rows']}, "
                                                              f"files {t5['stored_after_test']['files']}"]], [90, 84]),
              chart("T5_load.png", 140, "Figure 6. Response times and server memory during the load test."),
              Paragraph(f"<b>Verdict: {t5['verdict']}.</b> {t5['verdict_text']}. The model is shared by all users and "
                        "analyses one image at a time, so simultaneous users wait in a queue (waiting time above).", body)]
    else:
        S += [Paragraph("Not run.", body)]
    # T6
    t6 = run.load_json("T6_metrics.json")
    S += [Paragraph("T6 – Photo quality check", h2)]
    for f in sorted(run.dir.glob("T6_metrics_attempt*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        S += [Paragraph(f"<b>Earlier attempt ({f.stem.split('_')[-1]}, app {a.get('app_version', '1.5.1')}).</b> "
                        f"{a['verdict']}: {a['verdict_text']}. The quality check was then improved (see “Changes during "
                        "this run”) and T6 was run again with the unchanged criterion and the same test images.", body)]
    if t6:
        g = t6["groups"]
        st6 = t6["settings"]
        S += [table([["Image group", "How made", "Expected warning", "Correctly warned"],
                     ["Blurred", f"Gaussian blur sigma {st6['t6_blur_sigma']} px", "blurry", f"{g['blurred']['correct']}/{g['blurred']['n']} ({pct(g['blurred']['rate'], 0)})"],
                     ["Dark", f"brightness x {st6['t6_dark_factor']}", "dark", f"{g['dark']['correct']}/{g['dark']['n']} ({pct(g['dark']['rate'], 0)})"],
                     ["Tiny", f"longest side {st6['t6_tiny_side']} px", "too small", f"{g['tiny']['correct']}/{g['tiny']['n']} ({pct(g['tiny']['rate'], 0)})"],
                     ["Non-skin", "generated sky, leaves, text, wood, wall", "not skin", f"{g['non_skin']['correct']}/{g['non_skin']['n']} ({pct(g['non_skin']['rate'], 0)})"],
                     ["Unchanged test images", "for information", "no warning", f"{g['original']['correct']}/{g['original']['n']} ({pct(g['original']['rate'], 0)})"]],
                    [32, 58, 34, 50]),
              chart("T6_quality.png", 100, "Figure 7. Share of degraded images that received the matching warning."),
              Paragraph(f"<b>Verdict: {t6['verdict']}.</b> {t6['verdict_text']}. The server check and the phone "
                        "check use the same thresholds; this test runs the server check.", body)]
    else:
        S += [Paragraph("Not run.", body)]
    # T7
    t7 = run.load_json("T7_metrics.json")
    S += [Paragraph("T7 – Functional and security tests", h2)]
    if t7:
        desc = {"test_webapp": "login, roles, protected pages, upload, prediction, database, PDF, history",
                "test_webapp_features": "doctor feedback, research dashboard (admin only), CSV export, timings",
                "test_webapp_explain": "logo/icons, probabilities, explanations, quality check, info pages, PDFs",
                "test_mobile_pwa": "PWA, patient mode stores nothing, rate limit, upload limits, proxy IPs",
                "test_prior_correction": "prior correction and confidence levels end to end",
                "test_real_model": "the real SkinLens model and explanation", "test_pipeline": "data pipeline",
                "test_kaggle_stages": "Kaggle training stages", "test_experiment_stages": "paper experiments"}
        S += [table([["Test file", "Covers", "Tests", "Failed", "Skipped"]] +
                    [[f, desc.get(f, ""), d["tests"], d["failed"], d["skipped"]] for f, d in sorted(t7["by_file"].items())],
                    [38, 88, 14, 14, 20]),
              Paragraph(f"<b>Verdict: {t7['verdict']}.</b> {t7['verdict_text']} in {t7['duration_s'] / 60:.1f} minutes."
                        + (" Failed: " + "; ".join(t7["failed"]) if t7["failed"] else ""), body)]
    else:
        S += [Paragraph("Not run.", body)]
    # ------------------------------------------------------------------ 7 summary
    S += [PageBreak(), Paragraph("7. Summary", h1),
          table([["Test", "Name", "Result", "Key figure"]] +
                [[t, NAMES[t], badge(x["verdict"]), x["text"]] for t, x in v.items()], [12, 40, 26, 96])]
    # ------------------------------------------------------------------ 8 limitations
    S += [Paragraph("8. Limitations", h1)]
    for s in ["<b>Single dataset.</b> All accuracy figures come from HAM10000 (one source, Austria and Australia); "
              "performance on other populations and cameras is not shown here.",
              "<b>Mostly light skin.</b> HAM10000 contains very few images of darker skin types; results may be lower there.",
              "<b>Dermoscopic images.</b> The test images are dermoscopy photos. Phone photos taken by patients look "
              "different; T3 only simulates some photo conditions.",
              "<b>Explanations.</b> Grad-CAM often concentrates near the lesion border or partly outside the lesion; it "
              "explains the CNN branch only. The plain-language reasons are image measurements that fit the result, not "
              "the model's decision process.",
              "<b>No clinical study yet.</b> No prospective evaluation with doctors or patients has been done.",
              "<b>Laptop-hosted.</b> The system runs on one laptop CPU; simultaneous users queue for the model (T5).",
              "<b>Reference predictions.</b> T2 compares with test_predictions.npz exported from the training notebook; "
              "its SHA-256 is recorded with the T2 result."]:
        S += [Paragraph("• " + s, body)]
    # ------------------------------------------------------------------ 9 conclusion
    S += [Paragraph("9. Conclusion", h1),
          Table([[Paragraph(conclusion, ParagraphStyle("cc", parent=body, textColor=colors.white))]], colWidths=[W],
                style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREEN if trl_ok else AMBER),
                                  ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 10)])),
          Spacer(1, 3 * mm),
          Paragraph("<b>Towards TRL 5 (validation in a relevant environment)</b>:", body)]
    for s in ["test in a real clinic and with real smartphone photos taken by patients and staff;",
              "evaluate on an external dataset (e.g. ISIC 2019/2020, PAD-UFES-20, Fitzpatrick17k for darker skin);",
              "a structured evaluation by dermatologists of results and explanations (the app already records doctor "
              "agreement and usefulness ratings);",
              "hosting on a server with authentication and audit logging instead of a laptop."]:
        S += [Paragraph("• " + s, body)]
    # ------------------------------------------------------------------ 10 team
    S += [KeepTogether([Paragraph("10. Team", h1),
          table([["Name", "Register number"]] + [[n, r] for n, r in TEAM] + [["Guide", GUIDE]], [90, 84]),
          Paragraph(SCHOOL, body)]),
          Spacer(1, 4 * mm),
          Paragraph(f"Raw results: <font face='Courier'>validation_results/{run.dir.name}/</font> (run.json with the "
                    "frozen criteria and system identity, predictions.csv, T1–T7 metrics JSON and CSV files, charts). "
                    f"Report generated {datetime.now():%d %b %Y %H:%M}.", small)]
    doc.build(S, onFirstPage=page_deco, onLaterPages=page_deco)
    run.save_json("summary.json", dict(verdicts=v, trl4=trl_ok, conclusion=conclusion,
                                       generated=datetime.now().isoformat(timespec="seconds")))
    _update_latest(run, v, trl_ok)
    return out


def _update_latest(run, v: dict, trl_ok: bool) -> None:
    """validation_results/latest.json: read by the About page."""
    f = config.VALIDATION_RESULTS_DIR / "latest.json"
    f.write_text(json.dumps(dict(run=run.dir.name, date=datetime.now().strftime("%Y-%m-%d"), started=run.info["created"][:10], trl4=trl_ok,
                                 verdicts={t: x["verdict"] for t, x in v.items()}), indent=2), encoding="utf-8")
