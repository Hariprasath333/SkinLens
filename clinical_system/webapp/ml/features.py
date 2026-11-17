"""
Simple, transparent measurements of the skin spot in a photo, used to EXPLAIN a result in words.

They are classic image-processing measurements (Otsu segmentation of the lesion, shape, colour) and
are reported as "features the app measured that fit this result". The neural network does not use
them, so they are never presented as the way the AI decided.

All measurements are made on the image resized to SIZE x SIZE (the same size as the Grad-CAM map).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .xai import lesion_mask

SIZE = 384
# dermoscopy colour names (as in the ABCD rule / 7-point checklist), in display order
COLOUR_NAMES = ["light brown", "dark brown", "black", "blue-grey", "red", "white"]
MIN_COLOUR_SHARE = 0.05          # a colour counts if it covers at least 5% of the spot


def _lab(img: np.ndarray) -> np.ndarray:
    """CIELAB in real units: L* 0-100, a* and b* around 0."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def remove_hair(img: np.ndarray) -> np.ndarray:
    """Thin dark hairs are found with a black-hat filter and painted over (DullRazor-style), so they
    do not distort the outline or the colours."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    hair = (hat > 18).astype(np.uint8)
    if hair.mean() < 0.002:
        return img
    return cv2.inpaint(img, cv2.dilate(hair, np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA)


SKIN_AB = np.array([14.0, 17.0], np.float32)   # typical a*, b* of light skin in dermoscopy


def colour_classes(lab_px: np.ndarray) -> dict[str, float]:
    """Share of the spot's pixels in each dermoscopy colour (rough rules in CIELAB)."""
    L, a, b = lab_px[:, 0], lab_px[:, 1], lab_px[:, 2]
    chroma = np.hypot(a, b)
    white = (L > 78) & (chroma < 14)
    black = (L < 22) & ~white
    blue = (b < -2) & (L >= 22) & (L < 72) & ~white
    red = (a > 22) & (a > 1.3 * np.maximum(b, 0)) & (L >= 22) & ~blue & ~white
    rest = ~(white | black | blue | red)
    dark_brown = rest & (L < 42)
    light_brown = rest & (L >= 42) & (L <= 82) & (b > 6)
    n = max(len(L), 1)
    masks = {"light brown": light_brown, "dark brown": dark_brown, "black": black, "blue-grey": blue,
             "red": red, "white": white}
    return {k: float(v.sum() / n) for k, v in masks.items()}


def _asymmetry(mask: np.ndarray) -> float:
    """0 = perfectly symmetric. The spot is turned so its long axis is horizontal, then folded over
    both axes; the score is the average share of the spot that does not overlap its mirror image."""
    m = mask.astype(np.uint8)
    mom = cv2.moments(m, binaryImage=True)
    if mom["m00"] == 0:
        return 0.0
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    angle = 0.5 * math.degrees(math.atan2(2 * mom["mu11"], mom["mu20"] - mom["mu02"]))
    h, w = m.shape
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    r = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST) > 0
    area = max(r.sum(), 1)
    return float(np.mean([(r ^ np.flip(r, ax)).sum() / (2 * area) for ax in (0, 1)]))


def _border_irregularity(mask: np.ndarray) -> float:
    """0 = smooth convex outline. Combines notches (1 - solidity) and a ragged edge (how much longer
    the outline is than its convex hull)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = cv2.contourArea(c) / max(cv2.contourArea(hull), 1e-6)
    ragged = 1 - cv2.arcLength(hull, True) / max(cv2.arcLength(c, True), 1e-6)
    return float(np.clip((1 - solidity) + ragged, 0, 1))


def lesion_features(rgb: np.ndarray, cam: np.ndarray | None = None) -> dict:
    """Measurements of the spot plus where the Grad-CAM heat-map is concentrated."""
    img = remove_hair(cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    mask = lesion_mask(img)
    out: dict = dict(lesion_found=mask is not None)
    if mask is None:
        return out
    lab = _lab(img)
    k = max(3, int(0.04 * SIZE)) | 1                                  # ~15 px band at 384 px
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m8 = mask.astype(np.uint8)
    inner = cv2.erode(m8, ker) > 0
    outer = cv2.dilate(m8, ker) > 0
    ring = (cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k))) > 0) & ~outer

    ys, xs = np.nonzero(mask)
    skin_px = lab[ring] if ring.sum() > 50 else lab[~mask]
    # colours are judged against the surrounding skin: a colour cast of the whole photo (pink
    # dermoscopy light, yellow room light) is removed by moving the skin's a*/b* to typical skin
    shift = np.clip(SKIN_AB - np.median(skin_px[:, 1:], 0), -25, 25)
    les_px = lab[mask].copy()
    les_px[:, 1:] += shift
    shares = colour_classes(les_px)
    colours = [c for c in COLOUR_NAMES if shares[c] >= MIN_COLOUR_SHARE]
    les, skin = lab[mask].mean(0), skin_px.mean(0)
    out.update(
        area_share=float(mask.mean()),
        centre_offset=float(math.hypot(xs.mean() / SIZE - 0.5, ys.mean() / SIZE - 0.5)),
        asymmetry=_asymmetry(mask),
        border_irregularity=_border_irregularity(mask),
        colour_shares=shares,
        colours=colours,
        n_colours=len(colours),
        contrast_delta_e=float(np.linalg.norm(les - skin)),           # CIE76 colour difference
        darker_by=float(skin[0] - les[0]),                            # L* units; > 0 = spot is darker
    )
    if cam is not None:
        cam = cam if cam.shape == mask.shape else cv2.resize(cam, mask.shape[::-1])
        tot = float(cam.sum()) + 1e-8
        out.update(
            heat_inside=float(cam[mask].sum() / tot),                  # whole spot, edge included
            heat_core=float(cam[inner].sum() / tot),
            heat_border=float(cam[outer & ~inner].sum() / tot),
            heat_outside=float(cam[~outer].sum() / tot),
        )
        out["heat_focus"] = heat_focus(out)
    return out


def heat_focus(f: dict) -> str:
    """'inside' | 'border' | 'outside': where most of the heat-map energy is.
    The border band is thin, so it counts as the focus once it holds 40% of the energy."""
    if f["heat_outside"] > 0.5:
        return "outside"
    if f["heat_border"] >= 0.4 or f["heat_border"] > f["heat_core"]:
        return "border"
    return "inside" if f["heat_core"] >= f["heat_outside"] else "outside"


# thresholds used for the words (kept together so the patient reasons, the doctor table and the tests agree).
# Calibrated on synthetic shapes (circle 0.01 / 0.05, L-shape 0.33 / 0.42) and the HAM10000 sample image.
ASYM_SLIGHT, ASYM_HIGH = 0.10, 0.20        # asymmetry score
BORDER_SLIGHT, BORDER_HIGH = 0.10, 0.25    # border irregularity score
CONTRAST_LOW, CONTRAST_HIGH = 12.0, 30.0   # Delta E between spot and skin
