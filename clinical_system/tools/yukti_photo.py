r"""
yukti_photo.jpg: 1600 x 1000 collage of four real app screenshots with the SkinLens logo and title (< 2 MB).

    .venv\Scripts\python.exe tools\yukti_photo.py

Uses clinical_system/screenshots/: 15_doctor_result_desktop.png, 12_patient_result_probability_bars_phone.png,
13_patient_why_this_result_phone.png and 23c_validation_progress_desktop.png (made by tools/feature_screens.py).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parents[1]
SHOTS = HERE / "screenshots"
OUT = HERE / "yukti_photo.jpg"
W, H = 1600, 1000
BLUE, NAVY, BG, MUTED = (0, 119, 182), (16, 42, 71), (243, 246, 250), (95, 115, 131)


def font(size, bold=False):
    for name in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(img: Image.Image, w: int, h: int, top: bool = True) -> Image.Image:
    """Scale to width w and crop to height h (from the top)."""
    s = w / img.width
    img = img.resize((w, round(img.height * s)), Image.LANCZOS)
    if img.height > h:
        img = img.crop((0, 0, w, h)) if top else img.crop((0, (img.height - h) // 2, w, (img.height - h) // 2 + h))
    return img


def card(canvas: Image.Image, img: Image.Image, x: int, y: int, radius: int = 18, label: str = "") -> None:
    w, h = img.size
    sh = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((20, 24, w + 20, h + 24), radius, fill=(16, 42, 71, 60))
    canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)), (x - 20, y - 20))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=255)
    canvas.paste(img.convert("RGBA"), (x, y), mask)
    if label:
        d = ImageDraw.Draw(canvas)
        f = font(19, True)
        tw = d.textlength(label, font=f)
        d.rounded_rectangle((x + 14, y + h - 46, x + 34 + tw, y + h - 12), 17, fill=(0, 119, 182, 235))
        d.text((x + 24, y + h - 43), label, font=f, fill="white")


def phone(img: Image.Image, w: int, h: int) -> Image.Image:
    """Screenshot inside a simple drawn phone frame."""
    bez = 12
    frame = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(frame)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 38, fill=(29, 43, 54))
    scr = fit(img, w - 2 * bez, h - 2 * bez)
    m = Image.new("L", scr.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, scr.width - 1, scr.height - 1), 28, fill=255)
    frame.paste(scr, (bez, bez), m)
    return frame


def main() -> None:
    need = {"desk": "15_doctor_result_desktop.png", "bars": "12_patient_result_probability_bars_phone.png",
            "why": "13_patient_why_this_result_phone.png", "val": "23c_validation_progress_desktop.png"}
    missing = [f for f in need.values() if not (SHOTS / f).exists()]
    if missing:
        sys.exit("missing screenshots: " + ", ".join(missing))
    im = {k: Image.open(SHOTS / f).convert("RGB") for k, f in need.items()}
    c = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(c)
    # soft background shapes
    d.ellipse((1150, -300, 1900, 450), fill=(220, 236, 247, 255))
    d.ellipse((-250, 650, 450, 1300), fill=(226, 243, 241, 255))
    # header: logo + title
    logo = Image.open(HERE / "webapp" / "static" / "icons" / "logo-256.png").convert("RGBA").resize((92, 92), Image.LANCZOS)
    c.alpha_composite(logo, (48, 30))
    d.text((158, 30), "SkinLens", font=font(54, True), fill=NAVY)
    d.text((160, 96), "Explainable AI decision support for skin lesions  ·  web app + phone app  ·  TRL-4 lab validated system",
           font=font(22), fill=MUTED)
    # left: desktop result + validation summary
    card(c, fit(im["desk"], 930, 500), 48, 160, label="Doctor mode: result, probabilities, explanation")
    # validation: the whole T1-T7 panel next to a short summary
    val = im["val"]
    k = 268 / val.height
    val = val.resize((round(val.width * k), 268), Image.LANCZOS)
    panel = Image.new("RGB", (930, 268), "white")
    panel.paste(val, (0, 0))
    pd = ImageDraw.Draw(panel)
    x0 = val.width + 26
    pd.text((x0, 40), "7 / 7", font=font(64, True), fill=(42, 157, 143))
    pd.text((x0, 116), "tests passed", font=font(26, True), fill=NAVY)
    pd.text((x0, 156), "TRL 4 demonstrated", font=font(22), fill=MUTED)
    pd.text((x0, 186), "in the laboratory", font=font(22), fill=MUTED)
    card(c, panel, 48, 700)
    d.text((60, 976), "Validation Lab: accuracy 85.80% on 2,014 test images · 100% agreement with training", font=font(15), fill=MUTED)
    # right: two phones
    ph_w, ph_h = 262, 560
    for k, (key, lab) in enumerate((("bars", "Phone: all 7 probabilities"), ("why", "Patient mode: why this result"))):
        x = 1018 + k * (ph_w + 22)
        fr = phone(im[key], ph_w, ph_h)
        sh = Image.new("RGBA", (ph_w + 40, ph_h + 40), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((20, 26, ph_w + 20, ph_h + 26), 38, fill=(16, 42, 71, 70))
        c.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)), (x - 20, 170))
        c.alpha_composite(fr, (x, 180))
        f = font(16, True)
        tw = d.textlength(lab, font=f)
        d.text((x + (ph_w - tw) / 2, 180 + ph_h + 12), lab, font=f, fill=NAVY)
    d.text((1018, 800), "Hari Prasath M · Kodela Kailash Nadh", font=font(19, True), fill=NAVY)
    d.text((1018, 828), "Dharun Raj B", font=font(19, True), fill=NAVY)
    d.text((1018, 866), "Guide: Dr. B V Gokulnath", font=font(18), fill=MUTED)
    d.text((1018, 892), "SCOPE, VIT-AP University", font=font(18), fill=MUTED)
    d.text((1018, 940), "Research prototype – not a medical diagnosis", font=font(16), fill=MUTED)
    rgb = c.convert("RGB")
    for q in (92, 88, 84, 80):
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
        if buf.tell() < 2 * 1024 * 1024:
            OUT.write_bytes(buf.getvalue())
            print(f"wrote {OUT} ({buf.tell() / 1024:.0f} KB, {W}x{H}, quality {q})")
            return
    raise SystemExit("could not get under 2 MB")


if __name__ == "__main__":
    main()
