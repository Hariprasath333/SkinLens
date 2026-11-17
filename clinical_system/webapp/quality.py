"""
Photo quality check, run before the analysis (on the phone in JavaScript, and again here on the server).

It never blocks an analysis: problems are shown as friendly advice ("Retake photo" / "Use anyway") and,
in doctor mode, saved with the scan. The same measurements and thresholds are used in
static/js/app.js (QUALITY), so the phone and the server agree.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CHECK_SIDE = 512            # measurements are made on the photo resized to this longest side
MIN_SIDE = 300              # px, shortest side of the original photo
BLUR_MIN = 10.0             # variance of the Laplacian below this = too blurry (sample photo: 84,
                            # same photo blurred with sigma 2 px: 3)
DARK_MEAN = 60              # mean brightness (0-255) below this = too dark
BRIGHT_MEAN = 225           # mean brightness above this = over-exposed
BRIGHT_CLIP = 0.25          # or more than 25% of pixels almost pure white
CENTRE_MAX = 0.25           # spot centre further than 25% of the image from the middle
SKIN_MIN = 0.35             # less than 35% skin-coloured pixels = does not look like skin
# v1.5.2: skin-coloured surfaces such as wood. "Not skin" also when the whole picture has strong stripes in ONE
# direction (gradient orientation coherence, 0 = no main direction, 1 = parallel lines) AND very saturated colour.
# Chosen on development images only (400 HAM10000 training images: max saturation of the stripe-like ones 112;
# generated wood: coherence >= 0.73, saturation 179); never on the T6 test images.
STRIPES_MIN = 0.60
SATURATION_MIN = 140

MESSAGES = {
    "small": ("The photo is too small", "Use the camera's normal setting or move a little closer (at least 300 px)."),
    "blurry": ("The photo looks blurry", "Hold the phone still and tap the spot on the screen to focus before taking the photo."),
    "dark": ("The photo is too dark", "Move to daylight or a bright room. Avoid shadows over the spot."),
    "bright": ("The photo is too bright", "Avoid the flash and direct sunlight; a little indirect light works best."),
    "no_spot": ("No clear spot was found", "Make sure the spot is visible and fills a good part of the picture."),
    "off_centre": ("The spot is not in the centre", "Place the spot in the middle of the picture."),
    "not_skin": ("This does not look like skin", "Take a close photo of the skin spot itself (about 10 cm away)."),
}


def skin_fraction(rgb: np.ndarray) -> float:
    """Share of pixels whose colour is in the usual skin range (YCrCb rule; wide enough for light and
    dark skin and for pink dermoscopy light)."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return float(((cr >= 133) & (cr <= 180) & (cb >= 70) & (cb <= 135)).mean())


def stripes(gray: np.ndarray) -> float:
    """How much the whole picture is made of parallel lines in one direction (structure tensor coherence)."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    return math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)


def check(rgb: np.ndarray, original_size: tuple[int, int] | None = None) -> dict:
    """{'ok': bool, 'issues': [{'code', 'title', 'fix'}], 'metrics': {...}}"""
    h, w = rgb.shape[:2]
    ow, oh = original_size or (w, h)
    s = CHECK_SIDE / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    m = dict(
        width=int(ow), height=int(oh),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
        clipped=float((gray >= 250).mean()),
        skin=skin_fraction(small),
        stripes=stripes(gray),
        saturation=float(cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1].mean()),
    )
    from .ml.xai import lesion_mask
    sq = cv2.resize(small, (256, 256), interpolation=cv2.INTER_AREA)
    mask = lesion_mask(sq)
    m["spot_found"] = mask is not None
    if mask is not None:
        ys, xs = np.nonzero(mask)
        m["spot_offset"] = float(math.hypot(xs.mean() / 256 - 0.5, ys.mean() / 256 - 0.5))
    codes = []
    if min(ow, oh) < MIN_SIDE:
        codes.append("small")
    if m["brightness"] < DARK_MEAN:
        codes.append("dark")
    elif m["brightness"] > BRIGHT_MEAN or m["clipped"] > BRIGHT_CLIP:
        codes.append("bright")
    skin_like_surface = m["stripes"] >= STRIPES_MIN and m["saturation"] >= SATURATION_MIN   # e.g. wood grain
    not_skin = (m["skin"] < SKIN_MIN or skin_like_surface) and "dark" not in codes and "bright" not in codes
    if m["sharpness"] < BLUR_MIN and "dark" not in codes and not not_skin:
        codes.append("blurry")
    if not_skin:
        codes.append("not_skin")
    elif not m["spot_found"]:
        codes.append("no_spot")
    elif m["spot_offset"] > CENTRE_MAX:
        codes.append("off_centre")
    issues = [dict(code=c, title=MESSAGES[c][0], fix=MESSAGES[c][1]) for c in codes]
    return dict(ok=not issues, issues=issues, metrics=m)
