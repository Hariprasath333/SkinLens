"""
Draw the SkinLens app icons (same design as static/img/logo.svg: a magnifying lens with a
medical cross, medical blue background, teal handle).

    python tools/make_icons.py

Writes webapp/static/icons/: icon-192.png, icon-512.png, icon-maskable-512.png,
apple-touch-icon.png (180), favicon-32.png
"""
from pathlib import Path

from PIL import Image, ImageDraw

BLUE, TEAL, WHITE = (0, 119, 182), (0, 180, 216), (255, 255, 255)
OUT = Path(__file__).resolve().parents[1] / "webapp" / "static" / "icons"


def draw(size: int, maskable: bool = False, rounded: bool = True) -> Image.Image:
    s = 4 * size  # draw 4x larger, then shrink: smooth edges
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if maskable or not rounded:      # maskable: full square, the launcher applies its own shape
        d.rectangle([0, 0, s, s], fill=BLUE)
    else:
        d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=BLUE)
    # content area: maskable icons must keep everything inside the central 80 % "safe zone"
    k = 0.62 if maskable else 0.80
    c, u = s / 2, s * k / 40          # 40-unit design grid (like the SVG)
    ox, oy = c - 20 * u, c - 20 * u

    def P(x, y):
        return ox + x * u, oy + y * u
    r, w = 9.0 * u, 2.8 * u           # lens ring
    cx, cy = P(18, 18)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE, width=int(w))
    x0, y0 = P(24.6, 24.6)            # teal handle
    x1, y1 = P(31, 31)
    d.line([x0, y0, x1, y1], fill=TEAL, width=int(3.6 * u))
    hr = 1.8 * u
    d.ellipse([x1 - hr, y1 - hr, x1 + hr, y1 + hr], fill=TEAL)
    t, a = 2.6 * u, 4.6 * u           # white medical cross
    d.rounded_rectangle([cx - t / 2, cy - a, cx + t / 2, cy + a], radius=t / 2, fill=WHITE)
    d.rounded_rectangle([cx - a, cy - t / 2, cx + a, cy + t / 2], radius=t / 2, fill=WHITE)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw(192).save(OUT / "icon-192.png")
    draw(512).save(OUT / "icon-512.png")
    draw(512, maskable=True).save(OUT / "icon-maskable-512.png")
    draw(180, rounded=False).convert("RGB").save(OUT / "apple-touch-icon.png")  # iOS rounds it itself
    draw(32).save(OUT / "favicon-32.png")
    print("Icons written to", OUT)


if __name__ == "__main__":
    main()
