"""Patient mode ("Check my skin"): calm wording, advice, rate limiting and a short summary PDF.
Nothing here touches the database or the disk."""
from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
from collections import deque
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np

DISCLAIMER = "SkinLens is a research project, not a medical diagnosis."

# plain-language names (lesion codes of the 7/8-class models)
PLAIN = {
    "NV": ("a common mole", "melanocytic nevus"),
    "BKL": ("a harmless skin growth", "benign keratosis"),
    "DF": ("a harmless firm skin bump", "dermatofibroma"),
    "VASC": ("a spot made of small blood vessels", "vascular lesion"),
    "AK": ("a rough sun-damage spot", "actinic keratosis"),
    "BCC": ("a common type of skin cancer that grows slowly", "basal cell carcinoma"),
    "MEL": ("melanoma, a type of skin cancer", "melanoma"),
    "SCC": ("a type of skin cancer", "squamous cell carcinoma"),
}
# mel, bcc and akiec need attention (SCC too, for models that have that class)
CONCERNING = {"MEL", "BCC", "AK", "SCC"}

ADVICE_CONCERN = "This may need attention. Please see a dermatologist soon."
ADVICE_LESS = ("This looks less concerning, but only a doctor can confirm. "
               "See a doctor if it changes, bleeds, itches or grows.")
ADVICE_UNSURE = "The app is not sure. Please see a doctor."

LEVEL_WORDS = {"High": "The app is fairly sure about this.",
               "Medium": "The app is only partly sure about this.",
               "Low": "The app is not sure about this."}


def plain_result(code: str) -> str:
    words, medical = PLAIN.get(code, ("a skin spot", code))
    return f"This looks most like {words} ({medical})."


def advice(code: str, level: str) -> tuple[str, str]:
    """(advice text, tone) where tone is 'unsure' | 'concern' | 'calm'."""
    if level == "Low":
        return ADVICE_UNSURE, "unsure"
    if code in CONCERNING:
        return ADVICE_CONCERN, "concern"
    return ADVICE_LESS, "calm"


def heatmap_line(meas: dict) -> str:
    inside = meas.get("focus_inside_lesion")
    base = "The coloured map shows where the app looked most (red = most attention)"
    if inside is not None and inside < 0.35:
        return base + ". It looked mostly outside the spot, so this result is less reliable."
    if inside is not None and inside >= 0.6:
        return base + ". It looked mainly at the spot itself."
    return base + "."


def to_data_uri(rgb: np.ndarray, size: int = 512) -> str:
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


# ----------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """At most `limit` analyses per `window` seconds per client IP (in memory)."""

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """(allowed, seconds until the next analysis is allowed)."""
        now = time.monotonic()
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, int(self.window - (now - q[0])) + 1
            q.append(now)
            return True, 0

    def reset(self) -> None:
        with self.lock:
            self.hits.clear()


limiter = RateLimiter(int(os.getenv("PATIENT_RATE_LIMIT", 10)), float(os.getenv("PATIENT_RATE_WINDOW", 3600)))
lime_limiter = RateLimiter(int(os.getenv("PATIENT_RATE_LIMIT", 10)), float(os.getenv("PATIENT_RATE_WINDOW", 3600)))


# ----------------------------------------------------------------------------- LIME for patient mode
# The result page asks for LIME afterwards (it takes about a minute on a CPU). It sends back the photo
# it shows and a signed token that is only valid for that exact photo, for 15 minutes, and once.
LIME_TOKEN_AGE = 15 * 60
_used_tokens: dict[str, float] = {}
_used_lock = threading.Lock()


def _serializer():
    from itsdangerous import URLSafeTimedSerializer

    from . import config
    return URLSafeTimedSerializer(config.SECRET_KEY, salt="patient-lime")


def _photo_hash(photo_data_uri: str) -> str:
    return hashlib.sha256(photo_data_uri.encode()).hexdigest()


def lime_token(photo_data_uri: str, age, sex, site, code: str) -> str:
    return _serializer().dumps(dict(h=_photo_hash(photo_data_uri), age=age, sex=sex if sex in ("female", "male") else None,
                                    site=site, code=code))


def read_lime_token(token: str, photo_data_uri: str) -> dict | None:
    """The token's claims if it is valid for this photo and not used before, else None."""
    from itsdangerous import BadSignature
    try:
        claims = _serializer().loads(token, max_age=LIME_TOKEN_AGE)
    except BadSignature:
        return None
    if claims.get("h") != _photo_hash(photo_data_uri):
        return None
    now = time.monotonic()
    with _used_lock:
        for k in [k for k, t in _used_tokens.items() if now - t > LIME_TOKEN_AGE]:
            del _used_tokens[k]
        key = hashlib.sha256(token.encode()).hexdigest()
        if key in _used_tokens:
            return None
        _used_tokens[key] = now
    return claims


def lime_picture(rgb: np.ndarray, claims: dict) -> dict:
    from . import config
    from .ml import xai
    from .ml.predictor import get_predictor
    pred = get_predictor()
    with pred.session(age=claims.get("age"), sex=claims.get("sex"), site=claims.get("site")):
        out = xai.lime(pred, rgb, pred.classes.index(claims["code"]), config.LIME_SAMPLES, config.LIME_SEGMENTS)
    share = float(out["mask"].mean())
    return dict(image=to_data_uri(out["image"]),
                summary=f"Highlighted areas ({100 * share:.0f}% of the photo) pushed the app towards this result.")


def client_ip(request) -> str:
    """The phone's IP. Proxy headers are trusted only when the request comes from this computer
    itself (e.g. a Cloudflare tunnel running on the laptop); otherwise they could be faked."""
    peer = request.client.host if request.client else "unknown"
    if peer in ("127.0.0.1", "::1", "localhost"):
        fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return peer


# ----------------------------------------------------------------------------- summary PDF
def summary_pdf(r: dict) -> bytes:
    """A short summary (about two pages) the patient can show a doctor. No name or personal details."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from .ml import xai
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=11 * mm,
                            bottomMargin=11 * mm, title="SkinLens summary")
    ss = getSampleStyleSheet()
    blue = colors.HexColor("#0077B6")
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=blue, fontSize=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading3"], textColor=blue, spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.6, leading=12.6)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, textColor=colors.grey)
    tone = r["tone"]
    box = {"concern": "#E63946", "unsure": "#c27c0e", "calm": "#2A9D8F"}[tone]

    def img(rgb):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Image(BytesIO(enc.tobytes()), width=52 * mm, height=52 * mm)

    adv = Table([[Paragraph(f"<b>{r['advice']}</b>", ParagraphStyle("a", parent=body, textColor=colors.white,
                                                                     fontSize=12, leading=16))]],
                colWidths=[174 * mm])
    adv.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(box)),
                             ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    from .report import logo_header, prob_table
    x = r["x"]
    story = [
        logo_header(Paragraph("SkinLens - skin check summary", h1),
                    Paragraph("A research project, not a medical diagnosis.", small), 174),
        Paragraph(f"Made on {datetime.now():%d %b %Y, %H:%M} with the SkinLens app. Please show this page to your doctor.",
                  small),
        Spacer(1, 5), adv, Spacer(1, 5),
        Table([[img(r["rgb"]), img(xai.overlay(r["rgb"], r["cam"]))],
               [Paragraph("Your photo", small), Paragraph("Where the app looked (red = most attention)", small)]],
              colWidths=[87 * mm, 87 * mm]),
        Paragraph("What the app found", h2),
        Paragraph(r["plain"], body),
        Paragraph(f"How sure the app is: <b>{r['level']}</b>. {LEVEL_WORDS[r['level']]}", body),
        Paragraph(r["heatmap_line"], body),
        *([Paragraph("<b>Photo quality check:</b> " + "; ".join(i["title"] for i in r["quality"]["issues"])
                     + ". The result may be less reliable.", small)] if r.get("quality", {}).get("issues") else []),
        Paragraph("How well the photo matches each type", h2),
        prob_table(x["rows"], "patient", body, small, 174),
        Paragraph(x["calm_note"], small),
        Paragraph("Why the app gave this result", h2),
        Paragraph(x["honest_note"], small),
        *[Paragraph(f"&bull; {s}", body) for s in x["reasons"]],
        Paragraph("For the doctor", h2),
        Paragraph(f"<b>Clinical summary.</b> {x['summary']}", small),
        Paragraph("Measurements: " + "; ".join(f"{a} {b}" for a, b, _ in x["table"]) + ". "
                  f"{x['gradcam_note']}", small),
        Paragraph(f"Model output: {r['info']['name']} ({r['code']}), confidence {100 * r['confidence']:.0f}% "
                  f"({r['level']}). Model: {r['model_name']}. Grad-CAM heat-map shown above. "
                  "Image taken by the patient with a phone camera (not dermoscopy).", small),
        Spacer(1, 6),
        Paragraph(f"<b>{DISCLAIMER}</b> It cannot replace an examination by a doctor. "
                  "If you are worried about a skin spot, see a doctor even if this summary looks reassuring.", body),
    ]
    doc.build(story)
    return buf.getvalue()
