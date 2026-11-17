"""
Explainable AI for one prediction:
  * Grad-CAM heat-map (Selvaraju et al., 2017), averaged over ensemble members
  * LIME super-pixel explanation (Ribeiro et al., 2016), implemented here so it can run
    with a small number of samples on a CPU
  * simple image measurements (lesion outline, border, colour) turned into a plain-English sentence
  * confidence level, top-3 classes, "what this means" and a recommendation
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from .. import config
from .predictor import Predictor, class_info

# ----------------------------------------------------------------------------- Grad-CAM


def gradcam(pred: Predictor, rgb: np.ndarray, class_idx: int, out_size: int = 384) -> np.ndarray:
    """Returns a [0,1] heat-map of shape (out_size, out_size). Call inside predictor.session()."""
    maps = []
    for m in pred.members:
        store = {}

        def fwd_hook(_mod, _inp, output):
            out = output[0] if isinstance(output, tuple) else output
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        h = m.cam_layer.register_forward_hook(fwd_hook)
        try:
            x = pred.to_tensor(rgb, m.size).unsqueeze(0).to(pred.device).requires_grad_(True)
            with torch.enable_grad():
                logits = pred._forward(m, x)
                m.model.zero_grad(set_to_none=True)
                logits[0, class_idx].backward()
        finally:
            h.remove()
        act, grad = store["act"].detach()[0], store["grad"].detach()[0]
        if act.dim() == 2:  # ViT tokens (N, C) -> (C, H, W)
            act, grad = act[m.tokens:], grad[m.tokens:]
            side = int(math.sqrt(act.shape[0]))
            act = act.T.reshape(-1, side, side)
            grad = grad.T.reshape(-1, side, side)
        w =grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * act).sum(0)).cpu().numpy()
        cam = cv2.resize(cam, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        maps.append(cam)
        m.model.zero_grad(set_to_none=True)
    cam = np.mean(maps, 0)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    base = cv2.resize(rgb, cam.shape[::-1])
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * base + alpha * heat, 0, 255))


# ----------------------------------------------------------------------------- LIME
def lime(pred: Predictor, rgb: np.ndarray, class_idx: int, n_samples: int, n_segments: int,
         size: int = 256, seed: int = 0, batch: int = 8) -> dict:
    """Which image regions push the prediction up? Call inside predictor.session()."""
    from skimage.segmentation import mark_boundaries, slic
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    seg = slic(img, n_segments=n_segments, compactness=10, start_label=0)
    n_seg = int(seg.max()) + 1
    blurred = cv2.GaussianBlur(img, (0, 0), 12)
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, size=(n_samples, n_seg)).astype(np.float32)
    Z[0] = 1  # the original image
    y = []
    for i in range(0, n_samples, batch):
        imgs = []
        for z in Z[i:i + batch]:
            mask = z[seg][..., None]
            imgs.append(np.uint8(img * mask + blurred * (1 - mask)))
        y.append(pred.predict_batch(imgs)[:, class_idx])
    y = np.concatenate(y)
    # locally weighted ridge regression (LIME's default kernel, width 0.25)
    cos = Z.sum(1) / (np.sqrt(np.maximum(Z.sum(1), 1e-8)) * math.sqrt(n_seg))
    d = 1 - cos
    w = np.exp(-(d ** 2) / 0.25 ** 2)
    X = np.hstack([Z, np.ones((n_samples, 1), np.float32)])
    A = X.T @ (X * w[:, None]) + 1.0 * np.eye(n_seg + 1)
    A[-1, -1] -= 1.0  # do not penalise the intercept
    coef = np.linalg.solve(A, X.T @ (w * y))[:-1]
    top = [int(s) for s in np.argsort(-coef)[:5] if coef[s] > 0]
    keep = np.isin(seg, top)
    dim = np.uint8(img * 0.35 + 255 * 0.65 * 0.15)
    vis = np.where(keep[..., None], img, dim)
    vis = np.uint8(mark_boundaries(vis, keep.astype(int), color=(0, 0.71, 0.85), mode="thick") * 255)
    return dict(image=vis, mask=keep, coef=coef, segments=seg, top=top)


# ----------------------------------------------------------------------------- image measurements
def lesion_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Rough lesion outline: darker-than-skin region nearest the centre (Otsu threshold)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 3)
    _, th = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(cv2.morphologyEx(th, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab_img, stats, cent = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return None
    h, w = th.shape
    best, best_score = None, -1e9
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.01 * h * w:
            continue
        dist = np.hypot(cent[i][0] - w / 2, cent[i][1] - h / 2) / max(h, w)
        touches = (stats[i, cv2.CC_STAT_LEFT] == 0) + (stats[i, cv2.CC_STAT_TOP] == 0) + \
                  (stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] >= w) + \
                  (stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= h)
        score = area / (h * w) - 2 * dist - 0.15 * touches
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    m = lab_img == best
    return m if 0.02 < m.mean() < 0.9 else None


def measurements(rgb: np.ndarray, cam: np.ndarray) -> dict:
    size = cam.shape[0]
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    mask = lesion_mask(img)
    ys, xs = np.mgrid[0:size, 0:size]
    tot = cam.sum() + 1e-8
    cy, cx = (ys * cam).sum() / tot / size, (xs * cam).sum() / tot / size
    vert = "upper" if cy < 0.38 else "lower" if cy > 0.62 else "central"
    horz = "left" if cx < 0.38 else "right" if cx > 0.62 else "central"
    region = "centre" if vert == horz == "central" else (
        f"{vert}-{horz}" if "central" not in (vert, horz) else (vert if horz == "central" else horz))
    hot = cam >= 0.5
    out = dict(region=region, focus_area=float(hot.mean()), lesion_found=mask is not None)
    if mask is None:
        return out
    out["focus_inside_lesion"] = float((cam * mask).sum() / tot)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    out["border_regularity"] = float(4 * math.pi * area / max(perim ** 2, 1e-8))      # 1 = circle
    hull = cv2.convexHull(c)
    out["solidity"] = float(area / max(cv2.contourArea(hull), 1e-8))
    # asymmetry: overlap of the lesion with its mirror image around its centre
    mom = cv2.moments(mask.astype(np.uint8))
    mx, my = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    M = np.float32([[1, 0, size / 2 - mx], [0, 1, size / 2 - my]])
    centred = cv2.warpAffine(mask.astype(np.uint8), M, (size, size)) > 0
    asym = [1 - (centred & np.flip(centred, a)).sum() / max(centred.sum(), 1) for a in (0, 1)]
    out["asymmetry"] = float(np.mean(asym))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    region_px = lab[mask & hot] if (mask & hot).sum() > 50 else lab[mask]
    out["colour_variation"] = float(np.mean(region_px.std(0)))
    out["dark_fraction"] = float((region_px[:, 0] < 60).mean())
    # colour fractions are compared with the surrounding skin, so a pink/blue colour cast of the
    # whole photo (lighting, camera) is not reported as a lesion feature
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    blue = (hsv[..., 0] > 90) & (hsv[..., 0] < 130) & (hsv[..., 1] > 30)
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 90)
    skin = ~mask
    out["blue_grey_fraction"] = float(blue[mask].mean() - (blue[skin].mean() if skin.any() else 0))
    out["red_fraction"] = float(red[mask].mean() - (red[skin].mean() if skin.any() else 0))
    return out


# ----------------------------------------------------------------------------- written explanation
TYPICAL = {
    "MEL": "asymmetry, an irregular border, several colours (including very dark or blue-grey areas) "
           "and atypical structures are features dermatologists associate with melanoma",
    "NV": "a symmetric shape, a regular border and one or two even brown colours are typical of a benign mole",
    "BCC": "pearly or pink areas, branching (arborizing) blood vessels and blue-grey blobs are typical of "
           "basal cell carcinoma",
    "SCC": "scaly or crusted surfaces, white areas and dotted or looped vessels are typical of "
           "squamous cell carcinoma",
    "AK": "a rough, red-brown 'strawberry' pattern on sun-damaged skin is typical of actinic keratosis",
    "BKL": "a waxy, 'stuck-on' look with milia-like cysts or comedo-like openings is typical of "
           "benign keratosis",
    "DF": "a central white patch with a fine brown network around it is typical of dermatofibroma",
    "VASC": "red, purple or blue lagoon-like areas are typical of vascular lesions",
}


def describe(meas: dict) -> list[str]:
    feats = []
    br = meas.get("border_regularity")
    if br is not None:
        if br < 0.55 or meas.get("solidity", 1) < 0.85:
            feats.append("an irregular border")
        elif br < 0.75:
            feats.append("a slightly uneven border")
        else:
            feats.append("a fairly smooth, regular border")
    a = meas.get("asymmetry")
    if a is not None:
        feats.append("an asymmetric shape" if a > 0.25 else "a fairly symmetric shape")
    cv = meas.get("colour_variation")
    if cv is not None:
        feats.append("strong colour variation" if cv > 22 else "moderate colour variation" if cv > 13
                     else "fairly even colour")
    if meas.get("dark_fraction", 0) > 0.15:
        feats.append("very dark areas")
    if meas.get("blue_grey_fraction", 0) > 0.08:
        feats.append("blue-grey areas")
    if meas.get("red_fraction", 0) > 0.15:
        feats.append("red areas")
    return feats


def written_explanation(code: str, meas: dict, lime_ok: bool | None = None) -> str:
    info = class_info(code)
    spread = meas["focus_area"]
    where = f"the {meas['region']} region of the image"
    focus = "a small, well-defined area" if spread < 0.12 else "a moderate area" if spread < 0.3 \
        else "a large part of the image"
    s = [f"The model focused on {focus} in {where}"]
    inside = meas.get("focus_inside_lesion")
    if inside is not None:
        if inside >= 0.6:
            s[0] += ", mostly on the lesion itself."
        elif inside >= 0.35:
            s[0] += ", partly on the lesion and partly on the surrounding skin."
        else:
            s[0] += (", mostly OUTSIDE the lesion. This can mean the model reacted to something else in "
                     "the picture (hair, a ruler, ink marks or the image border), so treat this result "
                     "with extra caution.")
    else:
        s[0] += ". The lesion outline could not be measured automatically in this image."
    feats = describe(meas)
    if feats:
        s.append("In this image the lesion shows " + ", ".join(feats[:-1]) +
                 (" and " if len(feats) > 1 else "") + feats[-1] + ".")
    s.append(f"For reference, {TYPICAL.get(code, 'its visual features')}. "
             f"The model's most likely answer is {info['name'].lower()}.")
    return " ".join(s)


def confidence_level(p: float, margin: float) -> tuple[str, str]:
    """High > 80 %, Medium 50-80 %, Low < 50 % of the (prior-corrected) top-class probability.
    Thresholds come from model_settings.json. `margin` (1st minus 2nd probability) only adds a warning."""
    if p > config.CONFIDENCE_HIGH:
        return "High", ""
    if p >= config.CONFIDENCE_MEDIUM:
        warn = "" if margin >= 0.15 else ("The first and second answers are close; "
                                           "the model is unsure between them.")
        return "Medium", warn
    return "Low", ("Low confidence: the model is not sure about this image. Do not rely on this result; "
                   "please have the lesion examined by a dermatologist. Image quality (focus, lighting, "
                   "hair, distance) may also affect the result.")


def meaning_and_recommendation(code: str, level: str) -> tuple[str, str, str]:
    info = class_info(code)
    risk = info["risk"]
    if risk == "malignant":
        status = "Possible skin cancer detected"
        meaning = (f"The image looks most similar to {info['name'].lower()}, which is a type of skin cancer. "
                   "This is a computer estimate from one photo, not a diagnosis.")
        rec = ("Consult a specialist: refer the patient to a dermatologist promptly for examination "
               "(dermoscopy and, if needed, biopsy).")
    elif risk == "premalignant":
        status = "Pre-cancerous lesion suspected"
        meaning = (f"The image looks most similar to {info['name'].lower()}, a sun-damage lesion that can "
                   "sometimes turn into skin cancer if left untreated.")
        rec = "Consult a specialist: a dermatology review is recommended to confirm and discuss treatment."
    else:
        status = "Likely benign (no cancer detected)"
        meaning = (f"The image looks most similar to {info['name'].lower()}, which is usually harmless. "
                   "This is a computer estimate from one photo, not a diagnosis.")
        rec = ("Routine care: monitor the lesion. Consult a specialist if it changes in size, shape or "
               "colour, bleeds, itches, or if you are worried.")
    if level == "Low":
        rec = "Consult a specialist: the model is uncertain, so this lesion should be examined by a dermatologist."
    return status, meaning, rec


def top3(ranked: list[tuple[str, float]]) -> list[dict]:
    return [dict(**class_info(c), prob=p) for c, p in ranked[:3]]
