"""
Everything shown to explain one result, in both modes and in the PDF reports:
  * all class probabilities (the prior-corrected ones used for the prediction), rounded so they add up to 100
  * plain-language reasons built from real measurements of the photo (patient mode)
  * a measurement table, metadata used and a short clinical-style summary (doctor mode / "Show full details")
"""
from __future__ import annotations

import math

from . import config
from .diseases import ABBR, ATTENTION, PLAIN_LABEL
from .ml import features as F
from .ml.predictor import class_info

LEVEL_MEANING = {
    "High": "the top class has more than {hi}% probability; the result is consistent, but still confirm clinically.",
    "Medium": "the top class has {med}–{hi}% probability; clinical examination advised.",
    "Low": "the top class has less than {med}% probability; the result is unreliable, examination by a "
           "dermatologist is needed.",
}
GRADCAM_NOTE = "Grad-CAM explains the CNN branch only; LIME covers the whole model."
HONEST_NOTE = ("These are features the app measured in the photo that fit this result. They are not exactly how "
               "the AI decided: the AI looks at the whole image in its own way.")
CALM_NOTE = ("These percentages show how sure the app is, not your actual risk. "
             "Only a doctor can diagnose a skin lesion.")


def level_meaning(level: str) -> str:
    return LEVEL_MEANING[level].format(hi=round(100 * config.CONFIDENCE_HIGH), med=round(100 * config.CONFIDENCE_MEDIUM))


# ----------------------------------------------------------------------------- probabilities
def round_to_total(values: list[float], total: int) -> list[int]:
    """Largest-remainder rounding: integers that add up exactly to `total`."""
    raw = [v * total for v in values]
    out = [math.floor(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


def prob_rows(probs: dict[str, float], complete: bool = True) -> list[dict]:
    """All classes, highest first. pct = whole percent (patient), pct1 = one decimal (doctor); both columns
    add up to exactly 100. complete=False (old scans that stored only the top 3): plain rounding, no rescaling."""
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    if complete:
        s = sum(p for _, p in items) or 1.0
        vals = [p / s for _, p in items]
        whole, tenths = round_to_total(vals, 100), round_to_total(vals, 1000)
    else:
        whole, tenths = [round(100 * p) for _, p in items], [round(1000 * p) for _, p in items]
    rows = []
    for (code, p), w, t in zip(items, whole, tenths):
        plain, medical = PLAIN_LABEL.get(code, (class_info(code)["name"], class_info(code)["name"]))
        rows.append(dict(code=code, abbr=ABBR.get(code, code.lower()), name=class_info(code)["name"],
                         plain=plain, medical=medical, prob=p, pct=w, pct1=t / 10,
                         group="attention" if code in ATTENTION else "harmless"))
    return rows


# ----------------------------------------------------------------------------- words from measurements
def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def heat_sentence(f: dict) -> str | None:
    focus = f.get("heat_focus")
    if focus == "outside":
        return "The app looked mostly at the skin around the spot, so this result is less reliable."
    if focus == "border":
        return "The app mostly looked at the edge of the spot (see the heat-map)."
    if focus == "inside":
        return "The app mostly looked at the inside of the spot (see the heat-map)."
    return None


def patient_reasons(f: dict) -> list[str]:
    """3-5 short sentences, each based on a real measurement."""
    if not f.get("lesion_found"):
        out = ["The app could not find a clear outline of the spot, so its shape and colours could not be measured."]
        return out + ([heat_sentence(f)] if heat_sentence(f) else [])
    out = []
    n, cols = f["n_colours"], f["colours"]
    if n >= 2:
        out.append(f"The spot has {n} different colours ({_join(cols)}).")
    elif n == 1:
        out.append(f"The spot has one main colour ({cols[0]}).")
    b = f["border_irregularity"]
    out.append("The edge of the spot is uneven in some places." if b >= F.BORDER_HIGH else
               "The edge of the spot is slightly uneven." if b >= F.BORDER_SLIGHT else
               "The edge of the spot is smooth and even.")
    a = f["asymmetry"]
    out.append("The two halves of the spot do not look the same." if a >= F.ASYM_HIGH else
               "The two halves of the spot look a little different." if a >= F.ASYM_SLIGHT else
               "The two halves of the spot look about the same.")
    de, dl = f["contrast_delta_e"], f["darker_by"]
    tone = "darker" if dl > 0 else "lighter"
    out.append(f"The spot is much {tone} than the skin around it." if de >= F.CONTRAST_HIGH else
               f"The spot is {tone} than the skin around it." if de >= F.CONTRAST_LOW else
               "The spot is only slightly different in colour from the skin around it.")
    if heat_sentence(f):
        out.append(heat_sentence(f))
    return out[:5]


def clinical_features(f: dict) -> list[str]:
    if not f.get("lesion_found"):
        return ["lesion outline not segmented"]
    out = []
    if f["n_colours"] >= 3:
        out.append(f"multiple colours ({f['n_colours']})")
    elif f["n_colours"] == 2:
        out.append("two colours")
    else:
        out.append("uniform colour")
    out.append("irregular border" if f["border_irregularity"] >= F.BORDER_HIGH else
               "slightly irregular border" if f["border_irregularity"] >= F.BORDER_SLIGHT else "regular border")
    out.append("asymmetry" if f["asymmetry"] >= F.ASYM_HIGH else
               "mild asymmetry" if f["asymmetry"] >= F.ASYM_SLIGHT else "symmetric")
    if f["contrast_delta_e"] >= F.CONTRAST_HIGH:
        out.append("high contrast with surrounding skin")
    return out


FOCUS_WORDS = {"inside": "lesion interior", "border": "lesion border", "outside": "outside the lesion (caution)"}
LEVEL_ACTION = {"High": "confirm clinically", "Medium": "clinical examination advised",
                "Low": "unreliable, dermatology review needed"}


def clinical_summary(name: str, prob: float, f: dict, level: str) -> str:
    focus = FOCUS_WORDS.get(f.get("heat_focus"), "not measured")
    return (f"Prediction: {name.lower()} ({100 * prob:.1f}%). Features: {', '.join(clinical_features(f))}. "
            f"Heat-map focus: {focus}. Confidence: {level} – {LEVEL_ACTION[level]}.")


def feature_table(f: dict) -> list[tuple[str, str, str]]:
    """(measurement, value, how to read it) for the doctor table."""
    if not f.get("lesion_found"):
        rows = [("Lesion segmentation", "not found", "Otsu threshold found no clear spot near the centre")]
    else:
        rows = [
            ("Asymmetry score", f"{f['asymmetry']:.2f}",
             f"0 = symmetric; ≥ {F.ASYM_HIGH:.2f} asymmetric (fold over both principal axes)"),
            ("Border irregularity", f"{f['border_irregularity']:.2f}",
             f"0 = smooth; ≥ {F.BORDER_HIGH:.2f} irregular (notches + ragged edge vs convex hull)"),
            ("Number of colours", f"{f['n_colours']}" + (f" ({', '.join(f['colours'])})" if f["colours"] else ""),
             "dermoscopic colours covering ≥ 5% of the lesion, relative to surrounding skin"),
            ("Lesion-to-skin contrast", f"ΔE {f['contrast_delta_e']:.1f}",
             f"CIE76 colour difference; ≥ {F.CONTRAST_HIGH:.0f} high (L* {'darker' if f['darker_by'] > 0 else 'lighter'} "
             f"by {abs(f['darker_by']):.1f})"),
            ("Estimated lesion area share", f"{100 * f['area_share']:.1f}%", "share of the image covered by the lesion"),
        ]
    if "heat_inside" in f:
        rows.append(("Heat-map energy inside lesion",
                     f"{100 * f['heat_inside']:.1f}%" if f.get("lesion_found") else "–",
                     f"border band {100 * f['heat_border']:.0f}%, outside {100 * f['heat_outside']:.0f}%"
                     if f.get("lesion_found") else "lesion not segmented"))
    return rows


def metadata_used(age, sex, site, uses_metadata: bool) -> list[tuple[str, str]]:
    def given(v):
        return "not given" if v in (None, "", "unknown") else str(v)
    rows = [("Age", given(age)), ("Sex", given(sex)), ("Body site", given(site))]
    if not uses_metadata:
        rows.append(("Note", "this model uses the image only"))
    return rows


def build(code: str, probs: dict[str, float], f: dict, level: str, *, age=None, sex=None, site=None,
          uses_metadata: bool = True, complete: bool = True) -> dict:
    """Everything the templates and PDFs need, for one result."""
    rows = prob_rows(probs, complete)
    info = class_info(code)
    return dict(
        rows=rows, top3=rows[:3], features=f, reasons=patient_reasons(f), table=feature_table(f),
        summary=clinical_summary(info["name"], probs.get(code, rows[0]["prob"]), f, level),
        level=level, level_meaning=level_meaning(level), metadata=metadata_used(age, sex, site, uses_metadata),
        heat_sentence=heat_sentence(f), honest_note=HONEST_NOTE, calm_note=CALM_NOTE, gradcam_note=GRADCAM_NOTE,
    )


def for_scan(scan) -> dict:
    """The same for a saved scan (older scans without stored probabilities fall back to the top 3)."""
    probs = scan.probs or {c["code"]: c["prob"] for c in scan.top3}
    return build(scan.predicted_class, probs, scan.features, scan.confidence_level, age=scan.patient.age,
                 sex=scan.patient.gender, site=scan.body_site, uses_metadata=scan.uses_metadata,
                 complete=bool(scan.probs))
