r"""
SkinLens demo video for the YUKTI portal, recorded automatically from the REAL running app.

    .venv\Scripts\python.exe tools\demo_video.py                  # record + narrate + compose
    .venv\Scripts\python.exe tools\demo_video.py --work DIR --stage compose   # re-compose only

What it does
  * starts its OWN server with a throw-away data folder (never your real database), registers a
    DEMO doctor account and fills it with DEMO data (seed_demo.py)
  * analyses HAM10000 TEST images from validation_data (public dataset)
  * records every scene with Playwright video recording (installed Edge = Chromium), with a visible
    mouse pointer drawn into the page, desktop 1366x768 and phone 390x844
  * narration: Windows built-in text-to-speech (System.Speech, Microsoft Zira)
  * composes 1920x1080, 30 fps, H.264 + AAC with imageio-ffmpeg, captions in a bar under the picture
  * stops the server and deletes the temporary data folder (and the work folder unless --keep)

Writes clinical_system/docs/demo/SkinLens_Demo.mp4, thumbnail.jpg and youtube.txt
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
sys.path.insert(0, str(HERE))
from feature_screens import UA, free_port, start_server  # noqa: E402

OUT = HERE / "docs" / "demo"
W, H = 1920, 1080
BAR_H = 100                     # caption bar under the picture
AREA_H = H - BAR_H              # picture area
BLUE, NAVY, BG, MUTED, TEAL = (0, 119, 182), (16, 42, 71), (243, 246, 250), (95, 115, 131), (20, 160, 143)
USER, PW, DOC_NAME = "demo.doctor", "Demo@12345", "Dr. Demo Doctor"
RUN = None                      # final validation run (from validation_results/latest.json)
DESK = {"width": 1366, "height": 768}
PHONE = {"width": 390, "height": 844}
LOGO_SVG = (HERE / "webapp" / "static" / "img" / "logo.svg").read_text(encoding="utf-8")

# ------------------------------------------------------------------ scenes and narration
# Each narration chunk = (caption text, spoken text, mark name or None). A chunk starts at its mark
# (a moment recorded in the scene) or right after the previous chunk.
SCENES = [
    dict(id="s1_title", kind="card", min_len=5.0, chunks=[
        ("SkinLens is an explainable AI tool that helps doctors and patients analyse skin lesions.",
         "Skin Lens is an explainable A.I. tool that helps doctors and patients analyse skin lesions.", None)]),
    dict(id="s2_login", kind="desk", chunks=[
        ("Doctors log in securely and see their patients and recent scans.", None, None)]),
    dict(id="s3_analyze", kind="desk", chunks=[
        ("A doctor adds a patient and uploads a skin image.", None, None),
        ("The analysis takes about ten seconds.", None, "analyze")]),
    dict(id="s4_result", kind="desk", chunks=[
        ("Every result is explained:", None, None),
        ("a heat-map shows where the AI looked,", "a heat map shows where the A.I. looked,", "gradcam"),
        ("LIME shows which areas mattered,", "lime shows which areas mattered,", "lime"),
        ("and all seven disease probabilities are listed", None, "probs"),
        ("with measured features like colour, border and symmetry.", None, "features")]),
    dict(id="s5_pdf", kind="desk", chunks=[
        ("A PDF report is saved to the patient's record.", "A P.D.F. report is saved to the patient's record.", None)]),
    dict(id="s6_phone", kind="phone", label=("Patient mode", "on a phone · no account"), chunks=[
        ("On a phone, anyone can check a spot without an account.", None, None),
        ("Nothing is stored.", None, "about"),
        ("The app warns about poor photos", None, "blurred"),
        ("and gives clear, calm advice.", None, "result")]),
    dict(id="s7_learn", kind="phone", label=("Learn", "skin conditions · ABCDE"), chunks=[
        ("Patients can also learn about each skin condition", None, None),
        ("and the ABCDE warning signs of melanoma.", "and the A B C D E warning signs of melanoma.", "abcde")]),
    dict(id="s8_validation", kind="desk", chunks=[
        ("In laboratory validation, all seven tests passed:", None, None),
        ("accuracy, agreement with the trained model, robustness,", None, None),
        ("speed, reliability, photo checks and software tests.", None, None),
        ("SkinLens is validated at Technology Readiness Level 4.",
         "Skin Lens is validated at Technology Readiness Level 4.", "report")]),
    dict(id="s9_end", kind="card", min_len=5.0, chunks=[
        ("Decision support only – not a medical diagnosis.",
         "Skin Lens. Decision support only. Not a medical diagnosis.", None)]),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd: list, **kw) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"command failed: {cmd[0]}")


# ------------------------------------------------------------------ pointer drawn into the page
CURSOR_JS = r"""
(() => {
  if (window.__demoCursorInit) return; window.__demoCursorInit = true;
  const touch = !!window.__demoTouch, KEY = '__demoCursor';
  const arrow = '<svg width="28" height="28" viewBox="0 0 28 28"><path d="M4 2 L4 22 L9.5 17 L13 25.5 L16.6 24 L13.1 15.6 L20.5 15.6 Z" fill="#111" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"/></svg>';
  let el = null, pos = null;
  try { pos = JSON.parse(sessionStorage.getItem(KEY) || 'null'); } catch (e) {}
  function place() { if (el && pos) el.style.transform = `translate(${pos.x - (touch ? 17 : 4)}px, ${pos.y - (touch ? 17 : 2)}px)`; }
  function make() {
    if (document.getElementById('demo-cursor')) return;
    el = document.createElement('div'); el.id = 'demo-cursor';
    el.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;will-change:transform;' +
      (touch ? 'width:34px;height:34px;border-radius:50%;background:rgba(16,42,71,.28);border:2px solid rgba(255,255,255,.95);box-shadow:0 1px 6px rgba(0,0,0,.25);'
             : 'width:28px;height:28px;filter:drop-shadow(0 1px 2px rgba(0,0,0,.35));');
    if (!touch) el.innerHTML = arrow;
    if (!pos) el.style.opacity = '0';
    document.documentElement.appendChild(el); place();
  }
  window.__demoRipple = (x, y) => {
    const r = document.createElement('div');
    r.style.cssText = `position:fixed;left:${x - 22}px;top:${y - 22}px;width:44px;height:44px;border-radius:50%;` +
      'border:3px solid rgba(0,119,182,.85);background:rgba(0,119,182,.18);z-index:2147483646;pointer-events:none;' +
      'transform:scale(.3);opacity:1;transition:transform .45s ease-out,opacity .5s ease-out;';
    document.documentElement.appendChild(r);
    requestAnimationFrame(() => requestAnimationFrame(() => { r.style.transform = 'scale(1.25)'; r.style.opacity = '0'; }));
    setTimeout(() => r.remove(), 700);
  };
  document.addEventListener('mousemove', (e) => {
    pos = { x: e.clientX, y: e.clientY };
    try { sessionStorage.setItem(KEY, JSON.stringify(pos)); } catch (err) {}
    if (!el) make(); el.style.opacity = '1'; place();
  }, true);
  document.addEventListener('mousedown', (e) => window.__demoRipple(e.clientX, e.clientY), true);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', make); else make();
})();
"""

SCROLL_JS = """([y, ms]) => new Promise((done) => {
  document.documentElement.style.scrollBehavior = 'auto';
  const max = document.documentElement.scrollHeight - innerHeight;
  y = Math.max(0, Math.min(y, max));
  const y0 = scrollY, t0 = performance.now();
  function f(t) { const k = Math.min(1, (t - t0) / ms), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    scrollTo(0, y0 + (y - y0) * e); if (k < 1) requestAnimationFrame(f); else done(); }
  requestAnimationFrame(f); })"""


class Rec:
    """One recorded scene: a page whose video is cut to the kept segments; named marks on that timeline."""

    def __init__(self, browser, name: str, work: Path, phone: bool = False, storage: str | None = None):
        vp = PHONE if phone else DESK
        opts = dict(viewport=vp, record_video_dir=str(work / "raw" / name), record_video_size=vp,
                    accept_downloads=True, device_scale_factor=1)
        if phone:
            opts.update(is_mobile=True, has_touch=True, user_agent=UA)
        if storage:
            opts["storage_state"] = storage
        self.name, self.work, self.phone = name, work, phone
        self.ctx = browser.new_context(**opts)
        self.ctx.add_init_script(("window.__demoTouch = true;" if phone else "") + CURSOR_JS)
        self.page = self.ctx.new_page()
        self.t0 = time.monotonic()
        self.segs: list[list[float]] = []
        self.raw_marks: dict[str, float] = {}
        self.x, self.y = (vp["width"] * 0.55, vp["height"] * 0.55)

    # --- timeline
    def now(self) -> float:
        return time.monotonic() - self.t0

    def start(self) -> None:
        self.segs.append([self.now(), None])

    def pause(self) -> None:
        self.segs[-1][1] = self.now()

    def resume(self) -> None:
        self.start()

    def mark(self, name: str) -> None:
        self.raw_marks[name] = self.now()

    def finish(self) -> dict:
        if self.segs and self.segs[-1][1] is None:
            self.segs[-1][1] = self.now()
        video = self.page.video
        self.ctx.close()
        src = Path(video.path())
        dst = self.work / "raw" / f"{self.name}.webm"
        shutil.move(str(src), dst)
        kept, marks, acc = [], {}, 0.0
        for a, b in self.segs:
            kept.append([round(a, 3), round(b, 3)])
            for k, t in self.raw_marks.items():
                if a <= t <= b:
                    marks[k] = round(acc + t - a, 3)
            acc += b - a
        info = dict(video=dst.name, segments=kept, marks=marks, length=round(acc, 3), phone=self.phone)
        (self.work / "raw" / f"{self.name}.json").write_text(json.dumps(info, indent=1))
        print(f"  recorded {self.name}: {acc:.1f} s kept, marks {marks}", flush=True)
        return info

    # --- pointer
    def move(self, x: float, y: float, dur: float = 0.55) -> None:
        n = max(8, int(dur * 45))
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            t = i / n
            e = t * t * (3 - 2 * t)
            self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            time.sleep(dur / n)
        self.x, self.y = x, y

    def point(self, sel: str, fx: float = 0.5, fy: float = 0.5) -> tuple[float, float]:
        b = self.page.locator(sel).first.bounding_box()
        if b is None:
            raise SystemExit(f"{self.name}: element not visible: {sel}")
        return b["x"] + b["width"] * fx, b["y"] + b["height"] * fy

    def hover(self, sel: str, pause: float = 1.0, fx: float = 0.5, fy: float = 0.5, dur: float = 0.55) -> None:
        b = self.page.locator(sel).first.bounding_box()
        vh = (PHONE if self.phone else DESK)["height"]
        if b and (b["y"] + b["height"] * fy < 80 or b["y"] + b["height"] * fy > vh - 40):
            self.scroll_to(sel, int(vh * 0.45 - b["height"] * fy), 900, 0.2)
        self.move(*self.point(sel, fx, fy), dur=dur)
        time.sleep(pause)

    def click(self, sel: str, pause: float = 0.6, fx: float = 0.5, fy: float = 0.5) -> None:
        self.hover(sel, 0.3, fx, fy)
        self.page.mouse.click(self.x, self.y)
        time.sleep(pause)

    def tap(self, sel: str, pause: float = 0.6) -> None:
        """Pointer goes there and shows a click ripple, without a real click (for <select> boxes)."""
        self.hover(sel, 0.3)
        self.page.evaluate("([x, y]) => window.__demoRipple(x, y)", [self.x, self.y])
        time.sleep(pause)

    def type(self, sel: str, text: str, pause: float = 0.4) -> None:
        self.click(sel, 0.2)
        self.page.keyboard.type(text, delay=55)
        time.sleep(pause)

    def scroll_to(self, sel: str, offset: int = 80, ms: int = 1100, pause: float = 0.3) -> None:
        y = self.page.locator(sel).first.evaluate("(e, o) => e.getBoundingClientRect().top + scrollY - o", offset)
        self.page.evaluate(SCROLL_JS, [y, ms])
        time.sleep(pause)

    def scroll_by(self, dy: int, ms: int = 1100, pause: float = 0.3) -> None:
        y = self.page.evaluate("() => scrollY") + dy
        self.page.evaluate(SCROLL_JS, [y, ms])
        time.sleep(pause)

    def loaded(self, extra: float = 0.4) -> None:
        self.page.wait_for_load_state("load")
        try:
            self.page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:   # noqa: BLE001  (the PWA service worker can keep the network busy)
            pass
        time.sleep(extra)


def viewer_html(title: str, pages_png: list[bytes], width: int) -> str:
    """A plain PDF-viewer-like page (dark background, centred pages) for showing a downloaded PDF."""
    imgs = "".join(f'<img src="data:image/png;base64,{base64.b64encode(p).decode()}" style="width:{width}px">'
                   for p in pages_png)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
      html,body{{margin:0;background:#525659;font-family:'Segoe UI',sans-serif}}
      .bar{{position:fixed;top:0;left:0;right:0;height:44px;background:#323639;color:#f1f1f1;display:flex;align-items:center;
            padding:0 18px;font-size:15px;gap:14px;z-index:5;box-shadow:0 1px 4px rgba(0,0,0,.4)}}
      .bar b{{font-weight:600}} .pages{{padding:62px 0 40px;display:flex;flex-direction:column;align-items:center;gap:16px}}
      img{{box-shadow:0 2px 10px rgba(0,0,0,.45);background:#fff}}</style></head>
      <body><div class="bar"><b>{title}</b><span style="opacity:.75">{len(pages_png)} page(s) shown</span></div>
      <div class="pages">{imgs}</div><script>window.__demoTouch=false;{CURSOR_JS}</script></body></html>"""


def pdf_png(pdf: bytes, pages=(0,), scale: float = 1.6) -> list[bytes]:
    import io
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf)
    out = []
    for i in pages:
        buf = io.BytesIO()
        doc[i].render(scale=scale).to_pil().save(buf, "PNG")
        out.append(buf.getvalue())
    return out


# ------------------------------------------------------------------ test images (public HAM10000 test set)
def pick_images(work: Path) -> dict:
    from webapp import config, quality
    from webapp.analysis import compute
    from webapp.main import BODY_SITES
    from webapp.validation import data as D
    m = D.manifest().set_index("image_id")
    latest = json.loads((config.VALIDATION_RESULTS_DIR / "latest.json").read_text())
    pred = __import__("pandas").read_csv(config.VALIDATION_RESULTS_DIR / latest["run"] / "predictions.csv")
    pred = pred[pred["condition"] == "original"]

    def choose(code: str, min_conf: float, skip: int = 0) -> dict:
        rows = pred[(pred["true"] == code) & (pred["pred"] == code) & (pred["confidence"] >= min_conf)]
        found = 0
        for _, r in rows.iterrows():
            row = m.loc[r["image_id"]]
            age, sex, site = D.meta(row)
            if age is None or sex is None or site not in BODY_SITES or site == "unknown":
                continue
            rgb = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
            if not quality.check(rgb)["ok"]:
                continue
            res = compute(rgb, age, sex, site)
            if res["features"].get("heat_focus") != "inside" or res["code"] != code:
                continue
            if found < skip:
                found += 1
                continue
            return dict(id=r["image_id"], path=row["path"], age=int(age), sex=sex, site=site, conf=float(r["confidence"]))
        raise SystemExit(f"no suitable {code} test image")

    doc = choose("MEL", 0.85)
    pat = choose("NV", 0.90)
    blurred = work / "blurred_photo.jpg"
    cv2.imwrite(str(blurred), cv2.GaussianBlur(cv2.imread(pat["path"]), (0, 0), 4))
    # a few more test images for the DEMO records made by seed_demo.py (other classes)
    seed_dir = work / "seed_images"
    seed_dir.mkdir(exist_ok=True)
    for code in ("BCC", "BKL", "NV", "AK", "MEL", "VASC", "DF", "NV"):
        rows = pred[(pred["true"] == code) & (pred["pred"] == code)]
        for iid in rows["image_id"].head(3):
            if iid not in (doc["id"], pat["id"]):
                shutil.copy(m.loc[iid]["path"], seed_dir / f"{iid}.jpg")
                break
    info = dict(doctor=doc, patient=pat, blurred=str(blurred), seed_dir=str(seed_dir), run=latest["run"])
    print("images:", json.dumps({k: (v["id"] if isinstance(v, dict) else v) for k, v in info.items()}))
    (work / "images.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------ recording
def record(work: Path) -> None:
    from playwright.sync_api import sync_playwright
    (work / "raw").mkdir(parents=True, exist_ok=True)
    data = work / "data"
    data.mkdir(exist_ok=True)
    # choose the images in a separate process, so its copy of the model is gone before the server runs
    if not (work / "images.json").exists():
        run([sys.executable, str(Path(__file__)), "--work", str(work), "--stage", "pick"], cwd=str(HERE))
    imgs = json.loads((work / "images.json").read_text())
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    srv = start_server(data, port)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge")
            # DEMO doctor (first user = Admin, needed for the Validation Lab), then DEMO data
            c = b.new_context(viewport=DESK)
            pg = c.new_page()
            pg.goto(base + "/register")
            pg.fill("#full_name", DOC_NAME); pg.fill("#username", USER); pg.fill("#password", PW); pg.fill("#password2", PW)
            pg.click("form button[type=submit]"); pg.wait_for_url("**/dashboard")
            c.close()
            env = dict(os.environ, APP_DATA_DIR=str(data), APP_SECRET_KEY="screens")
            env.pop("DATABASE_URL", None)
            print("seeding DEMO data ...", flush=True)
            run([sys.executable, "seed_demo.py", "--patients", "8", "--max-scans", "1", "--lime", "0",
                 "--images", imgs["seed_dir"]], cwd=str(HERE), env=env)
            storage = str(work / "storage.json")
            info = {}

            # ---------- scene 2: login -> dashboard
            r = Rec(b, "s2_login", work)
            r.page.goto(base + "/login"); r.loaded(0.6); r.start()
            time.sleep(0.4)
            r.type("#username", USER)
            r.type("#password", PW)
            r.click("form button[type=submit]", 0.1)
            r.page.wait_for_url("**/dashboard"); r.loaded(0.6)
            r.hover(".card.stat >> nth=0", 0.8, fy=0.5)
            r.scroll_to("h2:has-text('Recent scans')", 110, 1000, 0.2)
            r.hover("table tbody tr >> nth=1", 1.0, fx=0.3)
            r.ctx.storage_state(path=storage)
            info["s2_login"] = r.finish()

            # ---------- scene 3: add patient -> upload & analyze
            d = imgs["doctor"]
            r = Rec(b, "s3_analyze", work, storage=storage)
            r.page.goto(base + "/dashboard"); r.loaded(0.6); r.start()
            time.sleep(0.5)
            r.click("a:has-text('+ Add Patient')", 0.1)
            r.page.wait_for_url("**/patients/new"); r.loaded(0.5)
            r.type("#name", "DEMO Patient - YUKTI demo")
            r.type("#age", str(d["age"]))
            r.tap("#gender"); r.page.select_option("#gender", d["sex"]); time.sleep(0.5)
            r.click("main button[type=submit]", 0.1)
            r.page.wait_for_url("**/analyze?patient=*"); r.loaded(0.6)
            r.tap("#body_site"); r.page.select_option("#body_site", d["site"]); time.sleep(0.6)
            with r.page.expect_file_chooser() as fc:
                r.click("label.pick-btn:has-text('Choose from gallery')", 0.1)
            fc.value.set_files(d["path"])
            r.page.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
            time.sleep(0.3)
            r.hover(".photo-preview img", 1.0)
            r.hover(".preview-info", 0.6, fx=0.2)
            r.mark("analyze")
            r.click("button.submit-btn", 0.1)
            t_click = time.monotonic()
            time.sleep(1.5)
            r.pause()
            r.page.wait_for_url("**/scans/*", timeout=300_000)
            r.loaded(0.2)
            wait_s = time.monotonic() - t_click
            r.resume()
            scan_url = r.page.url
            time.sleep(0.4)
            r.move(*r.point(".result-banner .dx", 0.5, 0.5), dur=0.8)
            time.sleep(1.4)
            info["s3_analyze"] = r.finish()
            info["s3_analyze"]["analysis_wall_s"] = round(wait_s, 1)

            # LIME is computed in the background: wait for it (not recorded)
            c = b.new_context(storage_state=storage)
            for _ in range(300):
                if c.request.get(scan_url + "/lime").json()["status"] != "pending":
                    break
                time.sleep(2)
            timings = c.new_page()
            timings.goto(scan_url)
            info["timings"] = timings.locator("#timings").inner_text()
            print("  ", info["timings"])
            c.close()

            # ---------- scene 4: result page
            r = Rec(b, "s4_result", work, storage=storage)
            r.page.goto(scan_url); r.loaded(0.8); r.start()
            r.hover(".result-banner .dx", 1.4)
            r.hover(".result-banner .conf", 1.6)
            r.hover("p:has-text('Photo quality check passed')", 1.4, fx=0.25)
            r.scroll_to("#r-swipe", 150, 1200, 0.3)
            r.hover("#r-swipe figure >> nth=0", 0.8)
            r.mark("gradcam")
            r.hover("#r-swipe figure >> nth=1", 2.6)
            r.mark("lime")
            r.hover("#r-swipe figure >> nth=2", 2.6)
            r.scroll_to(".pbars.doctor", 150, 1200, 0.2)
            r.mark("probs")
            for k in range(7):
                r.hover(f".pbars.doctor > * >> nth={k}", 0.3, fx=0.35, dur=0.3)
            time.sleep(0.6)
            r.scroll_to(".explain-details", 90, 1300, 0.2)
            r.mark("features")
            r.hover(".clinical-summary", 1.6, fx=0.3)
            r.hover(".meas-table tbody tr >> nth=0", 0.9, fx=0.25)
            r.hover(".meas-table tbody tr >> nth=1", 0.9, fx=0.25)
            r.hover(".meas-table tbody tr >> nth=2", 1.4, fx=0.25)
            info["s4_result"] = r.finish()

            # ---------- scene 5: PDF report
            r = Rec(b, "s5_pdf", work, storage=storage)
            r.page.goto(scan_url); r.loaded(0.8); r.start()
            time.sleep(0.4)
            with r.page.expect_download() as dl:
                r.click("a:has-text('Download Report (PDF)')", 0.2)
            pdf = Path(dl.value.path()).read_bytes()
            fname = dl.value.suggested_filename
            time.sleep(0.8)
            r.page.set_content(viewer_html(fname, pdf_png(pdf, (0,), 1.6), 900))
            r.page.mouse.move(r.x, r.y)
            time.sleep(1.4)
            r.scroll_by(420, 2300, 0.3)
            r.scroll_by(420, 2300, 1.0)
            info["s5_pdf"] = r.finish()

            # ---------- scene 6: phone, patient mode
            pt = imgs["patient"]
            r = Rec(b, "s6_phone", work, phone=True)
            r.page.goto(base + "/"); r.loaded(1.0); r.start()
            r.move(*r.point("a.mode-card[href='/check']", 0.5, 0.5), dur=0.8); time.sleep(0.7)
            r.click("a.mode-card[href='/check']", 0.1)
            r.page.wait_for_url("**/check"); r.loaded(0.6)
            r.scroll_to(".about-you", 70, 1000, 0.3)
            r.mark("about")
            r.type("#age", str(pt["age"]))
            r.tap("#sex"); r.page.select_option("#sex", pt["sex"]); time.sleep(0.4)
            r.tap("#body_site"); r.page.select_option("#body_site", pt["site"]); time.sleep(0.6)
            r.scroll_to(".photo-picker", 90, 900, 0.2)
            with r.page.expect_file_chooser() as fc:
                r.click("label.pick-btn:has-text('Choose from gallery')", 0.1)
            fc.value.set_files(imgs["blurred"])
            r.page.wait_for_selector(".quality-box:not([hidden])", timeout=15000)
            r.scroll_to(".quality-box", 260, 800, 0.1)
            r.mark("blurred")
            r.hover(".quality-list", 1.9, fx=0.4)
            with r.page.expect_file_chooser() as fc:
                r.click(".quality-retake", 0.1)
            fc.value.set_files(pt["path"])
            r.page.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
            r.scroll_to(".photo-preview", 90, 700, 0.3)
            r.hover(".preview-info", 0.5, fx=0.3)
            r.scroll_to("button.submit-btn", 300, 900, 0.2)
            r.click("button.submit-btn", 0.1)
            time.sleep(1.2)
            r.pause()
            r.page.wait_for_selector(".advice", timeout=300_000)
            r.loaded(0.3)
            r.resume()
            r.mark("result")
            r.hover(".advice", 2.2, fx=0.5, fy=0.4)
            r.scroll_to("#all-probs", 60, 1000, 0.2)
            r.hover("#all-probs", 2.2, fx=0.6, fy=0.45)
            r.scroll_to("#why", 60, 1100, 0.3)
            r.pause()
            r.page.wait_for_selector("#why img[alt^='Highlighted']", timeout=300_000)
            time.sleep(0.5)
            r.resume()
            r.hover("#why", 2.6, fx=0.5, fy=0.5)
            info["s6_phone"] = r.finish()

            # ---------- scene 7: disease info + ABCDE (phone)
            r = Rec(b, "s7_learn", work, phone=True)
            r.page.goto(base + "/learn/mel"); r.loaded(0.8); r.start()
            time.sleep(1.0)
            r.scroll_by(700, 2200, 0.6)
            r.page.goto(base + "/learn/abcde"); r.loaded(0.3)
            r.mark("abcde")
            time.sleep(1.2)
            r.scroll_by(650, 2200, 0.8)
            r.scroll_by(650, 2000, 1.0)
            info["s7_learn"] = r.finish()

            # ---------- scene 8: Validation Lab + TRL-4 report
            r = Rec(b, "s8_validation", work, storage=storage)
            r.page.goto(base + f"/admin/validation?run={imgs['run']}"); r.loaded(0.8); r.start()
            time.sleep(0.5)
            r.scroll_to("#run-card", 80, 1100, 0.2)
            r.hover("#run-card", 0.6, fx=0.3, fy=0.35)
            r.hover("#run-card", 1.6, fx=0.3, fy=0.7, dur=1.4)
            r.hover("#run-card a:has-text('Download report')", 0.4)
            r.page.evaluate("([x, y]) => window.__demoRipple(x, y)", [r.x, r.y])
            time.sleep(0.5)
            rep = (HERE / "validation_results" / imgs["run"] / "SkinLens_TRL4_Lab_Validation_Report.pdf").read_bytes()
            r.page.set_content(viewer_html("SkinLens_TRL4_Lab_Validation_Report.pdf", pdf_png(rep, (0,), 1.5), 490))
            r.page.mouse.move(r.x, r.y)
            r.mark("report")
            time.sleep(0.6)
            r.move(683, 330, dur=0.9)
            time.sleep(3.6)
            info["s8_validation"] = r.finish()

            # thumbnail picture: result page top in high resolution (not part of the video)
            c = b.new_context(viewport=DESK, device_scale_factor=2, storage_state=storage)
            pg = c.new_page(); pg.goto(scan_url); pg.wait_for_load_state("networkidle")
            pg.screenshot(path=str(work / "result_top.png"))
            pg.locator("#r-swipe").scroll_into_view_if_needed()
            pg.add_style_tag(content=".topbar { position: static !important; }")
            pg.locator(".card:has(#r-swipe)").screenshot(path=str(work / "result_xai.png"))
            c.close()
            b.close()
        (work / "raw" / "info.json").write_text(json.dumps(info, indent=1))
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        time.sleep(1)
        shutil.rmtree(data, ignore_errors=True)
        print("temporary server stopped and its DEMO database deleted:", not data.exists())


# ------------------------------------------------------------------ narration (Windows System.Speech)
TTS_PS1 = r"""
param([string]$List)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft Zira Desktop')
$s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($line in Get-Content -Encoding UTF8 $List) {
  $parts = $line -split "`t", 2
  $s.SetOutputToWaveFile($parts[0], $fmt)
  $s.Speak($parts[1])
}
$s.SetOutputToNull()
"""
SR = 44100


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    keep = np.where(np.abs(a) > 300)[0]           # trim leading / trailing silence
    return a[max(0, keep[0] - 800): keep[-1] + 2400] if len(keep) else a


def narrate(work: Path) -> dict:
    tts = work / "tts"
    tts.mkdir(exist_ok=True)
    lines = []
    for sc in SCENES:
        for i, (cap, spoken, _) in enumerate(sc["chunks"]):
            lines.append(f"{tts / f'{sc['id']}_{i}.wav'}\t{spoken or cap}")
    (tts / "list.txt").write_text("\n".join(lines), encoding="utf-8")
    (tts / "tts.ps1").write_text(TTS_PS1, encoding="utf-8")
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tts / "tts.ps1"), "-List", str(tts / "list.txt")])
    return {sc["id"]: [read_wav(tts / f"{sc['id']}_{i}.wav") for i in range(len(sc["chunks"]))] for sc in SCENES}


# ------------------------------------------------------------------ graphics
def logo_png(size: int) -> Image.Image:
    return Image.open(HERE / "webapp" / "static" / "icons" / "logo-256.png").convert("RGBA").resize((size, size), Image.LANCZOS)


def caption_png(text: str, path: Path) -> None:
    img = Image.new("RGB", (W, BAR_H), BLUE)
    d = ImageDraw.Draw(img)
    for size in (36, 33, 30):
        f = font(size, True)
        if d.textlength(text, font=f) <= W - 160:
            break
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, BAR_H / 2), text, font=f, fill="white", anchor="lm")
    lg = logo_png(56)
    white = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(white).ellipse((0, 0, 63, 63), fill=(255, 255, 255, 255))
    white.alpha_composite(lg, (4, 4))
    img.paste(white, (24, (BAR_H - 64) // 2), white)
    img.save(path)


def desk_bg(path: Path) -> None:
    img = Image.new("RGB", (W, AREA_H), BG)
    d = ImageDraw.Draw(img)
    vw, vh = 1706, 960
    x, y = (W - vw) // 2, (AREA_H - vh) // 2
    sh = Image.new("RGBA", (W, AREA_H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x - 2, y + 4, x + vw + 2, y + vh + 6), 8, fill=(16, 42, 71, 70))
    img.paste(sh.filter(ImageFilter.GaussianBlur(8)), (0, 0), sh.filter(ImageFilter.GaussianBlur(8)))
    d.rectangle((x - 1, y - 1, x + vw, y + vh), outline=(205, 214, 224))
    img.save(path)


PH_SCALE = 1.08
PH_SW, PH_SH = round(390 * PH_SCALE), round(844 * PH_SCALE)       # 421 x 912 screen
PH_BEZ = 14


def phone_layers(label: tuple[str, str], bg_path: Path, frame_path: Path) -> tuple[int, int]:
    fw, fh = PH_SW + 2 * PH_BEZ, PH_SH + 2 * PH_BEZ
    fx, fy = (W - fw) // 2, (AREA_H - fh) // 2
    sx, sy = fx + PH_BEZ, fy + PH_BEZ
    bg = Image.new("RGBA", (W, AREA_H), BG + (255,))
    d = ImageDraw.Draw(bg)
    d.ellipse((1380, -380, 2240, 480), fill=(220, 236, 247, 255))
    d.ellipse((-320, 560, 520, 1400), fill=(226, 243, 241, 255))
    bg.alpha_composite(logo_png(120), (150, 120))
    d.text((150, 262), "SkinLens", font=font(64, True), fill=NAVY)
    d.text((152, 346), "Explainable AI for", font=font(30), fill=MUTED)
    d.text((152, 386), "skin lesion analysis", font=font(30), fill=MUTED)
    d.rounded_rectangle((150, 470, 150 + 8, 470 + 110), 4, fill=TEAL)
    d.text((182, 466), label[0], font=font(44, True), fill=BLUE)
    d.text((184, 530), label[1], font=font(28), fill=MUTED)
    rx = fx + fw + 120
    for i, t in enumerate(("Real app, recorded live", "HAM10000 public test images", "DEMO data only")):
        d.ellipse((rx, 380 + i * 60 + 10, rx + 14, 380 + i * 60 + 24), fill=TEAL)
        d.text((rx + 30, 380 + i * 60), t, font=font(26), fill=NAVY)
    sh = Image.new("RGBA", (W, AREA_H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((fx, fy + 10, fx + fw, fy + fh + 10), 52, fill=(16, 42, 71, 90))
    bg.alpha_composite(sh.filter(ImageFilter.GaussianBlur(16)))
    ImageDraw.Draw(bg).rounded_rectangle((fx, fy, fx + fw, fy + fh), 52, fill=(29, 43, 54, 255))
    bg.convert("RGB").save(bg_path)
    # frame on top of the video: bezel with a rounded screen hole
    fr = Image.new("RGBA", (W, AREA_H), (0, 0, 0, 0))
    ImageDraw.Draw(fr).rounded_rectangle((fx, fy, fx + fw, fy + fh), 52, fill=(29, 43, 54, 255))
    hole = Image.new("L", (W, AREA_H), 0)
    ImageDraw.Draw(hole).rounded_rectangle((sx, sy, sx + PH_SW - 1, sy + PH_SH - 1), 38, fill=255)
    fr.putalpha(Image.fromarray(np.where(np.array(hole) > 0, 0, np.array(fr.getchannel("A"))).astype(np.uint8)))
    fr.save(frame_path)
    return sx, sy


def card_png(browser, html: str, path: Path) -> None:
    pg = browser.new_page(viewport={"width": W, "height": AREA_H})
    pg.set_content(html)
    pg.wait_for_timeout(300)
    pg.screenshot(path=str(path))
    pg.close()


def card_html(kind: str) -> str:
    style = """<style>html,body{margin:0;height:100%;font-family:'Segoe UI',sans-serif;background:#F3F6FA;color:#102A47;overflow:hidden}
      .wrap{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;position:relative}
      .b1{position:absolute;right:-220px;top:-320px;width:860px;height:860px;border-radius:50%;background:#DCECF7}
      .b2{position:absolute;left:-260px;bottom:-420px;width:840px;height:840px;border-radius:50%;background:#E2F3F1}
      .logo svg{width:190px;height:190px} h1{font-size:76px;margin:18px 0 6px;color:#0077B6;letter-spacing:-.5px}
      h2{font-size:40px;margin:0 0 54px;font-weight:600} .team{font-size:30px;font-weight:600;margin:6px 0}
      .guide{font-size:27px;color:#5F7383;margin:8px 0} .uni{font-size:27px;color:#5F7383;margin-top:6px}
      .pill{display:inline-block;margin-top:30px;padding:10px 26px;border-radius:30px;background:#fff;border:1px solid #CDD6E0;font-size:24px;color:#5F7383}
      .warn{font-size:44px;font-weight:700;margin:26px 0 40px;color:#102A47} .rule{width:120px;height:6px;border-radius:3px;background:#14A08F;margin:0 auto 36px}
      .z{position:relative}</style>"""
    if kind == "s1_title":
        body = f"""<div class="wrap"><div class="b1"></div><div class="b2"></div><div class="z">
          <div class="logo">{LOGO_SVG}</div><h1>SkinLens</h1>
          <h2>SkinLens – Explainable AI for Skin Lesion Analysis</h2>
          <div class="team">Hari Prasath M &nbsp;·&nbsp; Kodela Kailash Nadh &nbsp;·&nbsp; Dharun Raj B</div>
          <div class="guide">Guide: Dr. B V Gokulnath</div>
          <div class="uni">School of Computer Science and Engineering (SCOPE), VIT-AP University</div></div></div>"""
    else:
        body = f"""<div class="wrap"><div class="b1"></div><div class="b2"></div><div class="z">
          <div class="logo">{LOGO_SVG}</div><h1>SkinLens</h1>
          <div class="warn">Decision support only – not a medical diagnosis</div><div class="rule"></div>
          <div class="team">Hari Prasath M &nbsp;·&nbsp; Kodela Kailash Nadh &nbsp;·&nbsp; Dharun Raj B</div>
          <div class="guide">Guide: Dr. B V Gokulnath</div>
          <div class="uni">VIT-AP University</div>
          <div class="pill">Validated: TRL 4 (laboratory)</div></div></div>"""
    return f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body>{body}</body></html>"


# ------------------------------------------------------------------ composing
def compose(work: Path) -> Path:
    from playwright.sync_api import sync_playwright
    raw = work / "raw"
    comp = work / "comp"
    comp.mkdir(exist_ok=True)
    voices = narrate(work)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        for k in ("s1_title", "s9_end"):
            card_png(b, card_html(k), comp / f"{k}.png")
        b.close()
    desk_bg(comp / "desk_bg.png")
    ff = ffmpeg()
    parts, report = [], []
    for sc in SCENES:
        sid = sc["id"]
        meta = json.loads((raw / f"{sid}.json").read_text()) if sc["kind"] != "card" else dict(length=0, marks={})
        # narration timing
        starts, t = [], 0.5
        for i, (_, _, mk) in enumerate(sc["chunks"]):
            if mk and mk in meta["marks"]:
                t = max(t, meta["marks"][mk] + 0.2)
            starts.append(t)
            t += len(voices[sid][i]) / SR + 0.25
        voice_end = t - 0.25
        length = max(meta["length"], voice_end + 0.8, sc.get("min_len", 0))
        hold = max(0.0, length - meta["length"])
        audio = np.zeros(int(length * SR) + SR, np.float32)
        for st, v in zip(starts, voices[sid]):
            i0 = int(st * SR)
            audio[i0:i0 + len(v)] += v
        audio = np.clip(audio[: int(length * SR)], -32767, 32767).astype(np.int16)
        wav = comp / f"{sid}.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(audio.tobytes())
        # captions: each chunk from its start to the next start (last one to the end)
        caps = []
        for i, (cap, _, _) in enumerate(sc["chunks"]):
            cp = comp / f"{sid}_cap{i}.png"
            caption_png(cap, cp)
            a = 0.0 if i == 0 else starts[i]
            e = starts[i + 1] if i + 1 < len(sc["chunks"]) else length + 1
            caps.append((cp, a, e))
        # video graph
        inputs, fl = [], []
        if sc["kind"] == "card":
            inputs += ["-loop", "1", "-framerate", "30", "-t", f"{length:.3f}", "-i", str(comp / f"{sid}.png")]
            fl.append(f"[0:v]fps=30,format=rgb24,pad={W}:{H}:0:0:color=#F3F6FA[base]")
        else:
            inputs += ["-i", str(raw / meta["video"])]
            segs = meta["segments"]
            for j, (a, e) in enumerate(segs):
                fl.append(f"[0:v]trim=start={a}:end={e},setpts=PTS-STARTPTS[g{j}]")
            fl.append("".join(f"[g{j}]" for j in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0,fps=30,"
                      f"tpad=stop_mode=clone:stop_duration={hold + 0.5:.3f}[v]")
            if sc["kind"] == "desk":
                inputs += ["-loop", "1", "-i", str(comp / "desk_bg.png")]
                fl.append(f"[v]scale=1706:960:flags=lanczos[vs]")
                fl.append(f"[1:v]pad={W}:{H}:0:0:color=#F3F6FA[bg]")
                fl.append(f"[bg][vs]overlay={(W - 1706) // 2}:{(AREA_H - 960) // 2}:shortest=1[base]")
            else:
                sx, sy = phone_layers(sc["label"], comp / f"{sid}_bg.png", comp / f"{sid}_frame.png")
                inputs += ["-loop", "1", "-i", str(comp / f"{sid}_bg.png"), "-loop", "1", "-i", str(comp / f"{sid}_frame.png")]
                fl.append(f"[v]scale={PH_SW}:{PH_SH}:flags=lanczos[vs]")
                fl.append(f"[1:v]pad={W}:{H}:0:0:color=#F3F6FA[bg]")
                fl.append(f"[bg][vs]overlay={sx}:{sy}:shortest=1[b0]")
                fl.append(f"[b0][2:v]overlay=0:0[base]")
        n_in = sum(1 for x in inputs if x == "-i")
        last = "base"
        for i, (cp, a, e) in enumerate(caps):
            inputs += ["-loop", "1", "-i", str(cp)]
            fl.append(f"[{last}][{n_in + i}:v]overlay=0:{AREA_H}:enable='between(t,{a:.3f},{e:.3f})'[c{i}]")
            last = f"c{i}"
        fl.append(f"[{last}]fade=t=in:st=0:d=0.35:color=white,fade=t=out:st={length - 0.35:.3f}:d=0.35:color=white,"
                  f"format=yuv420p[out]")
        inputs += ["-i", str(wav)]
        out = comp / f"{sid}.mp4"
        run([ff, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fl), "-map", "[out]",
             "-map", f"{n_in + len(caps)}:a", "-t", f"{length:.3f}", "-r", "30", "-c:v", "libx264", "-preset", "slow",
             "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
             "-ac", "2", str(out)])
        parts.append(out)
        report.append(dict(scene=sid, length=round(length, 2), video=meta["length"], voice_end=round(voice_end, 2),
                           starts=[round(s, 2) for s in starts]))
        print(f"  composed {sid}: {length:.1f} s (recorded {meta['length']:.1f} s, narration ends {voice_end:.1f} s)", flush=True)
    lst = comp / "parts.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / "SkinLens_Demo.mp4"
    run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         "-movflags", "+faststart", str(final)])
    (comp / "timeline.json").write_text(json.dumps(report, indent=1))
    return final


# ------------------------------------------------------------------ thumbnail + youtube text
def thumbnail(work: Path) -> None:
    tw, th = 1280, 720
    c = Image.new("RGBA", (tw, th), BG + (255,))
    d = ImageDraw.Draw(c)
    d.ellipse((820, -300, 1500, 380), fill=(220, 236, 247, 255))
    d.ellipse((-260, 430, 360, 1050), fill=(226, 243, 241, 255))
    c.alpha_composite(logo_png(118), (60, 64))
    d.text((60, 196), "SkinLens", font=font(84, True), fill=BLUE)
    for i, t in enumerate(("Explainable AI for", "Skin Lesion Analysis")):
        d.text((62, 300 + i * 58), t, font=font(46, True), fill=NAVY)
    d.text((62, 438), "Grad-CAM · LIME · 7 classes", font=font(28), fill=MUTED)
    d.text((62, 478), "Doctor web app + patient phone app", font=font(28), fill=MUTED)
    d.rounded_rectangle((60, 560, 470, 620), 30, fill=TEAL)
    d.text((265, 590), "TRL 4 lab validated", font=font(28, True), fill="white", anchor="mm")
    d.text((62, 660), "Decision support only – not a medical diagnosis", font=font(20), fill=MUTED)
    shot = Image.open(work / "result_top.png").convert("RGB")
    shot = shot.crop((0, 0, shot.width, min(shot.height, round(shot.width * 0.62))))
    sw = 690
    shot = shot.resize((sw, round(shot.height * sw / shot.width)), Image.LANCZOS)
    x, y = 545, (th - shot.height) // 2
    sh = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x - 4, y + 8, x + sw + 4, y + shot.height + 12), 16, fill=(16, 42, 71, 90))
    c.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)))
    m = Image.new("L", shot.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, shot.width - 1, shot.height - 1), 14, fill=255)
    c.paste(shot, (x, y), m)
    c.convert("RGB").save(OUT / "thumbnail.jpg", "JPEG", quality=90, optimize=True)


YOUTUBE = """TITLE
SkinLens – Explainable AI for Skin Lesion Analysis (Doctor + Patient App, TRL 4 Validated) | VIT-AP University

DESCRIPTION
SkinLens is an explainable AI decision-support tool that analyses skin-lesion images into 7 classes and shows why: Grad-CAM heat-maps, LIME, all class probabilities and measured features (colour, border, symmetry).
Doctors get patient records and PDF reports; on a phone anyone can check a spot without an account, with photo-quality warnings and calm advice. Validated in the laboratory at TRL 4 (7/7 tests passed).
Decision support only – not a medical diagnosis. Team: Hari Prasath M, Kodela Kailash Nadh, Dharun Raj B · Guide: Dr. B V Gokulnath · SCOPE, VIT-AP University.

TAGS
SkinLens, explainable AI, XAI, skin lesion, dermatology AI, skin cancer detection, melanoma, dermoscopy, HAM10000, Grad-CAM, LIME, deep learning, EfficientNetV2, Swin Transformer, medical imaging, decision support, FastAPI, PWA, TRL 4, YUKTI, VIT-AP University
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=None, help="work folder (default: a new temporary folder)")
    ap.add_argument("--stage", choices=["all", "pick", "record", "compose"], default="all")
    ap.add_argument("--keep", action="store_true", help="keep the work folder (recordings) afterwards")
    a = ap.parse_args()
    work = a.work or Path(tempfile.mkdtemp(prefix="skinlens_demo_"))
    work.mkdir(parents=True, exist_ok=True)
    print("work folder:", work, flush=True)
    if a.stage == "pick":
        pick_images(work)
        return
    if a.stage in ("all", "record"):
        record(work)
    if a.stage in ("all", "compose"):
        final = compose(work)
        thumbnail(work)
        (OUT / "youtube.txt").write_text(YOUTUBE, encoding="utf-8")
        mb = final.stat().st_size / 1e6
        r = subprocess.run([ffmpeg(), "-i", str(final)], capture_output=True, text=True).stderr
        dur = next((ln.split("Duration:")[1].split(",")[0].strip() for ln in r.splitlines() if "Duration:" in ln), "?")
        print(f"wrote {final} ({dur}, {mb:.1f} MB)")
    if not a.keep and a.work is None:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
