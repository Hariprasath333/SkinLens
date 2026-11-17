r"""
Makes every logo / icon file the web app and phone app use, from the two SVGs in this folder:

    skinlens_logo.svg       transparent logo   -> website header, login, favicons, PDF reports
    skinlens_app_icon.svg   logo on a square   -> app icons (PWA 192/512, maskable, apple-touch-icon)

Run from the clinical_system folder (needs: pip install playwright pillow; uses the installed Edge or Chrome):
    .\.venv\Scripts\python.exe branding\make_icons.py

After changing the logo, also raise VERSION in webapp/static/sw.js so installed phone apps fetch the new icons.
"""
from __future__ import annotations

import re
import shutil
from io import BytesIO
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "webapp" / "static"
ICONS, IMG = STATIC / "icons", STATIC / "img"
LOGO = (HERE / "skinlens_logo.svg").read_text(encoding="utf-8")
APP = (HERE / "skinlens_app_icon.svg").read_text(encoding="utf-8")
SQUARE = re.sub(r'rx="\d+(\.\d+)?"', 'rx="0"', APP, count=1)   # full-bleed square (the OS rounds the corners)


def maskable(svg: str) -> str:
    """Square background; the logo shrunk to about 52% so it stays inside the 80% safe zone."""
    return SQUARE.replace("scale(0.62)", "scale(0.52)")


def render(page, svg: str, size: int) -> Image.Image:
    page.set_viewport_size({"width": size, "height": size})
    page.set_content(
        "<html><body style='margin:0;background:transparent'>"
        + svg.replace("<svg ", f'<svg width="{size}" height="{size}" ', 1) + "</body></html>")
    return Image.open(BytesIO(page.screenshot(omit_background=True, clip=dict(x=0, y=0, width=size, height=size))))


def main() -> None:
    from playwright.sync_api import sync_playwright
    ICONS.mkdir(parents=True, exist_ok=True)
    jobs = {  # file -> (svg, size)
        "favicon-16.png": (LOGO, 16), "favicon-32.png": (LOGO, 32), "favicon-48.png": (LOGO, 48),
        "logo-256.png": (LOGO, 256),                       # PDF report header
        "apple-touch-icon.png": (SQUARE, 180),             # iOS: no transparency, iOS rounds the corners
        "icon-192.png": (APP, 192), "icon-512.png": (APP, 512),
        "icon-maskable-512.png": (maskable(APP), 512),
    }
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:  # noqa: BLE001
            browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(device_scale_factor=1)
        out = {}
        for name, (svg, size) in jobs.items():
            im = render(page, svg, size)
            if name in ("apple-touch-icon.png", "icon-maskable-512.png"):
                im = im.convert("RGB")
            im.save(ICONS / name)
            out[name] = im
            print(f"wrote icons/{name}  {im.size[0]}x{im.size[1]}")
        browser.close()
    out["favicon-48.png"].save(ICONS / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)],
                               append_images=[out["favicon-16.png"], out["favicon-32.png"]])
    print("wrote icons/favicon.ico  16, 32, 48 px")
    shutil.copyfile(HERE / "skinlens_logo.svg", IMG / "logo.svg")
    shutil.copyfile(HERE / "skinlens_app_icon.svg", IMG / "app-icon.svg")
    print("copied img/logo.svg, img/app-icon.svg")


if __name__ == "__main__":
    main()
