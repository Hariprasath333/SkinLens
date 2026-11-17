"""PDF report for one scan (ReportLab)."""
from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config, insight
from .db import Scan

BLUE = colors.HexColor("#0077B6")
TEAL = colors.HexColor("#00B4D8")
GREEN = colors.HexColor("#2A9D8F")
RED = colors.HexColor("#E63946")
LIGHT = colors.HexColor("#F5FAFD")
DISCLAIMER = "For research and decision-support only. Not a substitute for professional medical diagnosis."
LOGO_PNG = config.BASE_DIR / "static" / "icons" / "logo-256.png"
ORANGE = colors.HexColor("#f08a4b")


def logo_header(title_par, sub_par, width_mm: float = 178):
    """SkinLens logo next to the report title."""
    logo = Image(str(LOGO_PNG), width=14 * mm, height=14 * mm) if LOGO_PNG.exists() else ""
    t = Table([[logo, [title_par, sub_par]]], colWidths=[17 * mm, (width_mm - 17) * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("LINEBELOW", (0, 0), (-1, 0), 1.2, BLUE), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def prob_table(rows: list[dict], mode: str, body, small, width_mm: float = 178):
    """All class probabilities with a small bar; red/orange = needs attention, green = usually harmless."""
    from reportlab.graphics.shapes import Drawing, Rect
    bar_w = 40 * mm

    def bar(c):
        d = Drawing(bar_w, 8)
        d.add(Rect(0, 1, bar_w, 6, fillColor=colors.HexColor("#edf3f7"), strokeColor=None))
        d.add(Rect(0, 1, max(1, bar_w * c["prob"]), 6, strokeColor=None,
                   fillColor=RED if c["group"] == "attention" else GREEN))
        return d
    if mode == "patient":
        head = ["Type of spot", "", "Match"]
        data = [[Paragraph(f"{c['plain']}<br/><font size=7 color='#5f7383'>{c['medical']}</font>", body), bar(c),
                 f"{c['pct']}%"] for c in rows]
        widths = [width_mm - 62, 44, 18]
    else:
        head = ["Class", "Code", "Group", "", "Probability"]
        data = [[c["name"], c["abbr"], "attention" if c["group"] == "attention" else "benign", bar(c),
                 f"{c['pct1']:.1f}%"] for c in rows]
        widths = [width_mm - 112, 18, 24, 44, 26]
    t = Table([head] + data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5 if mode == "patient" else 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 if mode == "patient" else 3),
                           ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.lightgrey)]))
    return t


def build_pdf(scan: Scan) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"Skin lesion report {scan.patient.patient_code}")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=BLUE, fontSize=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading3"], textColor=BLUE, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)
    p, d = scan.patient, scan.doctor
    accent = RED if scan.is_disease else GREEN

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(16 * mm, 9 * mm, DISCLAIMER)
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {_doc.page}")
        canvas.restoreState()

    x = insight.for_scan(scan)
    story = [logo_header(Paragraph("SkinLens Clinical &mdash; Skin Lesion Analysis Report", h1),
                         Paragraph(f"Generated {scan.created_at:%d %b %Y, %H:%M} &middot; Scan #{scan.id}", small)),
             Spacer(1, 6)]

    info = [["Patient ID", p.patient_code, "Analysed by", d.full_name],
            ["Name", p.name, "Date / time", f"{scan.created_at:%d %b %Y %H:%M}"],
            ["Age / Gender", f"{p.age if p.age is not None else '-'} / {p.gender}", "Body site", scan.body_site],
            ["Contact", p.contact or "-", "Model", Paragraph(scan.model_name, small)]]
    t = Table(info, colWidths=[28 * mm, 58 * mm, 28 * mm, 64 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("TEXTCOLOR", (0, 0), (0, -1), BLUE), ("TEXTCOLOR", (2, 0), (2, -1), BLUE),
                           ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                           ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [t, Spacer(1, 8)]

    result = Table([[Paragraph(f"<b>{scan.predicted_name}</b> ({scan.predicted_class})",
                               ParagraphStyle("r", parent=body, fontSize=13, textColor=colors.white)),
                     Paragraph(f"Confidence: <b>{100 * scan.confidence:.1f}%</b> ({scan.confidence_level})",
                               ParagraphStyle("c", parent=body, textColor=colors.white, alignment=2))]],
                   colWidths=[110 * mm, 68 * mm])
    result.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), accent), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [result]
    if scan.warning:
        story += [Spacer(1, 4), Paragraph(f"<font color='#E63946'><b>Warning:</b> {scan.warning}</font>", body)]

    imgs, caps = [], []
    for path, cap in [(scan.image_path, "Uploaded image"), (scan.gradcam_path, "Grad-CAM heat-map"),
                      (scan.lime_path, "LIME explanation")]:
        if path and (config.STORAGE_DIR / path).exists():
            imgs.append(Image(str(config.STORAGE_DIR / path), width=55 * mm, height=55 * mm))
            caps.append(Paragraph(cap, small))
    if imgs:
        story += [Paragraph("Images", h2), Table([imgs, caps], colWidths=[59 * mm] * len(imgs))]

    q = scan.quality
    if q:
        txt = ("passed (sharpness, brightness, size, spot near the centre, skin-like colours)" if q.get("ok") else
               "; ".join(f"{i['title']} &ndash; {i['fix']}" for i in q.get("issues", [])))
        story += [Spacer(1, 4), Paragraph(f"<b>Photo quality check:</b> {txt}", body)]
    story += [Paragraph("Probabilities of all classes" if scan.probs else "Top 3 possible classes", h2),
              prob_table(x["rows"], "doctor", body, small),
              Paragraph(f"Prior-corrected probabilities (the ones used for the prediction); they add up to 100%. "
                        f"Confidence level <b>{scan.confidence_level}</b>: {x['level_meaning']}", small),
              Paragraph("Why the model gave this result", h2), Paragraph(scan.explanation, body)]
    if scan.lime_summary:
        story += [Spacer(1, 4), Paragraph(scan.lime_summary, body)]
    if scan.features_json:
        meas = [["Measurement", "Value", "How to read it"]] + [
            [a, b, Paragraph(c, small)] for a, b, c in x["table"]]
        mt = Table(meas, colWidths=[46 * mm, 40 * mm, 92 * mm])
        mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("TEXTCOLOR", (0, 0), (-1, 0), BLUE),
                                ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
        story += [Spacer(1, 4), Paragraph(f"<b>Clinical summary.</b> {x['summary']}", body), Spacer(1, 4), mt,
                  Paragraph(x["honest_note"], small),
                  Paragraph("Patient information used by the model: " +
                            ", ".join(f"{k.lower()} {v}" for k, v in x["metadata"]) + ".", small),
                  Paragraph(f"Note: {x['gradcam_note']}", small)]
    story += [Paragraph("What this means", h2), Paragraph(scan.meaning, body),
              Paragraph("Recommendation", h2), Paragraph(f"<b>{scan.recommendation}</b>", body),
              Spacer(1, 10), Paragraph(DISCLAIMER, small)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
