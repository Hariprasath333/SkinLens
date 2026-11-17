"""
Build the SkinLens user guide: HTML (A4 pages) -> PDF with Playwright.

    python build_guide.py            # writes ../SkinLens_User_Guide.html and ../SkinLens_User_Guide.pdf

Inputs (all in docs/guide/): illustrations/*.svg, screenshots/*.png + markers*.json, facts.json
(measured during the from-scratch verification), figures/*.png (copied from paper_figures/).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

G = Path(__file__).resolve().parents[1]
ILL = G / "illustrations"
SHOTS = G / "screenshots"
FACTS = json.loads((G / "facts.json").read_text(encoding="utf-8"))
MARK = {}
for f in ("markers_register.json", "markers.json"):
    if (SHOTS / f).exists():
        MARK.update(json.loads((SHOTS / f).read_text()))
FOOT = "SkinLens – For research and decision-support only. Not a medical diagnosis."

# ------------------------------------------------------------------ small line icons (24x24)
ICONS = {
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "bolt": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
    "folder": '<path d="M3 6a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "download": '<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',
    "play": '<circle cx="12" cy="12" r="9"/><path d="m10 8 6 4-6 4z"/>',
    "doctor": '<path d="M6 3v5a4 4 0 0 0 8 0V3M10 12v3a5 5 0 0 0 10 0v-2"/><circle cx="20" cy="11" r="2"/>',
    "phone": '<rect x="6" y="2" width="12" height="20" rx="3"/><path d="M11 18h2"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13.5" r="3.5"/>',
    "wrench": '<path d="M14.7 6.3a4 4 0 0 0 5 5L22 14l-8 8-2.3-2.3a4 4 0 0 0-5-5L2 10l8-8z"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "shield": '<path d="M12 3 20 6v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    "card": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M7 15h4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "warn": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0zM12 9v4M12 17h.01"/>',
    "bulb": '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3z"/>',
    "note": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "disk": '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "ram": '<rect x="2" y="7" width="20" height="10" rx="2"/><path d="M6 17v3M10 17v3M14 17v3M18 17v3M6 11h2M11 11h2M16 11h2"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    "py": '<path d="M8 3h6a3 3 0 0 1 3 3v4H9a3 3 0 0 0-3 3v2H5a2 2 0 0 1-2-2v-3a3 3 0 0 1 3-3h1V6a3 3 0 0 1 1-3z"/><path d="M16 21h-6a3 3 0 0 1-3-3v-4h8a3 3 0 0 0 3-3V9h1a2 2 0 0 1 2 2v3a3 3 0 0 1-3 3h-1v1a3 3 0 0 1-1 3z"/>',
    "nogpu": '<rect x="3" y="7" width="18" height="10" rx="2"/><path d="M7 11h4M3 3l18 18"/>',
    "os": '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "key": '<circle cx="8" cy="15" r="4"/><path d="m10.8 12.2 8.2-8.2M17 6l2 2M15 8l2 2"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1"/>',
}


def icon(name, size=22, cls=""):
    return (f'<svg class="ic {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def ill(name, width="100%", cls=""):
    svg = (ILL / name).read_text(encoding="utf-8")
    return f'<div class="ill {cls}" style="width:{width}">{svg}</div>'


def esc(s):
    return html.escape(s, quote=False)


def cmd(text, label="PowerShell"):
    lines = "".join(f'<div class="cl"><span class="pr">&gt;</span>{esc(t)}</div>' if not t.startswith("#") else
                    f'<div class="cl cm">{esc(t)}</div>' for t in text.strip("\n").split("\n"))
    return f'<div class="cmd"><div class="cmd-h">{label}</div>{lines}</div>'


def out(text, label="Output"):
    return f'<div class="cmd out"><div class="cmd-h">{label}</div><pre>{esc(text.strip())}</pre></div>'


def box(kind, title, body):
    ic = {"tip": "bulb", "note": "note", "warn": "warn", "ok": "check"}[kind]
    return f'<div class="box {kind}"><div class="bx-i">{icon(ic, 20)}</div><div><b>{title}</b> {body}</div></div>'


def step(n, title, body):
    return f'<div class="step"><div class="sn">{n}</div><div class="sb"><h4>{title}</h4>{body}</div></div>'


SIDE = {  # where each numbered label goes, so it never hides its button
    ("d02_dashboard", 1): "left", ("d02_dashboard", 2): "below",
    ("d04_upload", 1): "below", ("d04_upload", 2): "below", ("d10_patient_record", 1): "below", ("d10_patient_record", 2): "right",
    ("m01_start", 1): "tr", ("m01_start", 2): "tr", ("m01_start", 3): "right", ("m02_login", 1): "tr",
    ("m03_check", 1): "below", ("m03_check", 2): "below", ("m05_patient_result", 1): "tr",
    ("m06_patient_heatmap", 1): "below", ("m07_patient_summary", 1): "tr",
    ("m09_dashboard", 1): "above", ("m09_dashboard", 2): "above", ("m10_doctor_gradcam", 1): "below",
}


def markers_html(name):
    hs = []
    for m in MARK.get(name, []):
        if m["y"] < 0 or m["y"] + m["h"] > 1.02:   # target not visible in the screenshot
            continue
        side = SIDE.get((name, m["n"]), m.get("side", "right"))
        x, y, w, h = m["x"], m["y"], m["w"], m["h"]
        cx, cy = {"right": (x + w, y + h / 2), "left": (x, y + h / 2), "below": (x + w / 2, y + h),
                  "above": (x + w / 2, y), "tr": (x + w, y)}[side]
        cx, cy = 100 * cx, 100 * cy
        side = "left" if side == "tr" else side
        hs.append(f'<div class="mk {side}" style="left:{cx:.2f}%;top:{cy:.2f}%"><span class="mk-n">{m["n"]}</span>'
                  f'<span class="mk-l">{esc(m["label"])}</span></div>')
    return "".join(hs)


def laptop(name, cap="", width="100%"):
    return (f'<figure class="laptop" style="width:{width}"><div class="lp-screen"><div class="lp-cam"></div>'
            f'<div class="shot"><img src="screenshots/{name}.png">{markers_html(name)}</div></div>'
            f'<div class="lp-base"></div>{f"<figcaption>{cap}</figcaption>" if cap else ""}</figure>')


def phone(name, cap="", width="62mm"):
    return (f'<figure class="phone" style="width:{width}"><div class="ph-body"><div class="ph-notch"></div>'
            f'<div class="shot"><img src="screenshots/{name}.png">{markers_html(name)}</div></div>'
            f'{f"<figcaption>{cap}</figcaption>" if cap else ""}</figure>')


def plain(name, cap="", width="100%", cls=""):
    return (f'<figure class="plainshot {cls}" style="width:{width}"><div class="shot"><img src="screenshots/{name}.png">'
            f'{markers_html(name)}</div>{f"<figcaption>{cap}</figcaption>" if cap else ""}</figure>')


# ------------------------------------------------------------------ pages
PAGES: list[tuple[str, str, str]] = []   # (section id or "", section title, html)
SECTIONS: list[tuple[str, str]] = []


def page(body, sec=None, icon_name=None, title=None, cls=""):
    head = ""
    if title:
        head = (f'<div class="band"><div class="band-i">{icon(icon_name, 26)}</div>'
                f'<div><div class="band-n">{sec}</div><h1>{title}</h1></div></div>')
        SECTIONS.append((sec, title))
    PAGES.append((sec or "", title or "", f'<section class="page {cls}">{head}<div class="pb{"" if title else " nb"}">{body}</div>'
                  '<footer class="pf"><span>' + FOOT + '</span><span class="pn"></span></footer></section>'))


def cont(body, cls=""):
    PAGES.append(("", "", f'<section class="page {cls}"><div class="pb nb">{body}</div>'
                  '<footer class="pf"><span>' + FOOT + '</span><span class="pn"></span></footer></section>'))


F = FACTS
LOGO = (G.parents[1] / "webapp" / "static" / "img" / "logo.svg").read_text(encoding="utf-8")

# 1 cover --------------------------------------------------------------------------------
COVER = f"""
<section class="page cover">
  <div class="cv-top"><div class="cv-logo">{LOGO}</div><div class="cv-brand">SkinLens</div></div>
  <div class="cv-hero">{(ILL / '01_cover_hero.svg').read_text(encoding='utf-8')}</div>
  <h1 class="cv-title">SkinLens – Installation and User Guide</h1>
  <p class="cv-sub">Explainable skin lesion analysis on web and mobile</p>
  <div class="cv-meta">
    <div><div class="cv-k">Team</div>
      <div>Hari Prasath M <span>(23BCE8999)</span></div>
      <div>Kodela Kailash Nadh <span>(23BCE7748)</span></div>
      <div>Dharun Raj B <span>(23BCE9271)</span></div></div>
    <div><div class="cv-k">Guide</div><div>Dr. B V Gokulnath</div>
      <div class="cv-k" style="margin-top:4mm">Institution</div>
      <div>School of Computer Science and Engineering (SCOPE)</div><div>VIT-AP University</div>
      <div class="cv-k" style="margin-top:4mm">Date</div><div>November 2026</div></div>
  </div>
  <div class="cv-foot">{FOOT}</div>
</section>"""

# 3 what is it -----------------------------------------------------------------------------
page(f"""
<p class="lead">SkinLens looks at a photo of a skin spot (lesion) and suggests which of seven common lesion types it
looks most like. It also shows <b>where</b> it looked (a heat-map), <b>how sure</b> it is (High / Medium / Low) and a
short written explanation. It runs as a website on your own computer, which you can also open and install on a phone.</p>
<div class="two">
  <div class="mini">{icon('doctor')}<div><b>Doctor mode</b><br>Log in, keep patient records and scan history, read the
  full result (Grad-CAM, LIME, top 3), give feedback and download a PDF report.</div></div>
  <div class="mini">{icon('camera')}<div><b>Patient mode</b><br>No login. Take a photo, get a calm, plain-language result
  and advice. <b>Nothing is stored.</b></div></div>
</div>
<h3 class="hd">How it works</h3>
{ill('02_how_it_works.svg')}
<div class="two" style="margin-top:4mm">
  <div><h4>The AI model</h4><p>A hybrid network: <b>EfficientNetV2-S</b> (good at fine texture) + <b>Swin Transformer V2</b>
  (good at overall shape) + a small network for age, sex and body site. Trained on HAM10000 (10,015 images).</p></div>
  <div><h4>The explanation</h4><p><b>Grad-CAM</b> colours the areas the model used most. <b>LIME</b> (doctor mode) is a
  second, independent check. The text describes border, colour and where the model looked.</p></div>
</div>
{box('warn', 'Important:', 'SkinLens is a research project for decision support. It is <b>not</b> a medical diagnosis and not a certified medical device.')}
""", "1", "info", "What SkinLens is")

# 4 quick start ---------------------------------------------------------------------------
page(f"""
<p class="lead">Already have Python 3.12 and the project folder? These five steps get you from zero to the start screen.
Each step is explained in detail later in this guide.</p>
<div class="qs">
  {step(1, 'Open PowerShell in the project folder', cmd('cd C:\\Users\\<you>\\Desktop\\SkinClass\\clinical_system'))}
  {step(2, 'Create a virtual environment', cmd('py -3.12 -m venv .venv'))}
  {step(3, 'Install PyTorch (CPU version) and the requirements', cmd('.\\.venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'))}
  {step(4, 'Start the server', cmd('.\\.venv\\Scripts\\python.exe -m uvicorn webapp.main:app --host 0.0.0.0 --port 8080'))}
  {step(5, 'Open the app and create the first account', '<p>Open <span class="url">http://127.0.0.1:8080</span> in a browser, choose <b>I&#39;m a doctor / clinic staff</b> &rarr; <b>Create an account</b>. The first account becomes the <b>Admin</b>.</p>')}
</div>
{box('tip', 'Timing on our test laptop:', f"installation took about {F['install_minutes']} minutes on home broadband; the first start (loading the model) took {F['first_start_s']} s; one analysis took about {F['analysis_s']} s (LIME adds {F['lime_s']} s in the background).")}
""", "2", "bolt", "Quick start")

# 5 requirements --------------------------------------------------------------------------
cards = [
    ("os", "Operating system", "Windows 10 or 11, 64-bit.<br><small>Tested on Windows 11. macOS / Linux: see the side box in section 5 (not tested).</small>"),
    ("py", "Python 3.12", f"64-bit, from python.org.<br><small>Tested with Python {F['python_version']}.</small>"),
    ("disk", "Disk space", f"about {F['disk_total_gb']} GB free<br><small>virtual environment {F['venv_gb']} GB + model file 0.41 GB + project.</small>"),
    ("ram", "Memory (RAM)", f"8 GB recommended<br><small>The running server used about {F['ram_gb']} GB on our laptop.</small>"),
    ("globe", "Internet", f"only for the first installation<br><small>about {F['download_mb']} MB of downloads; fonts and charts also load from the internet.</small>"),
    ("nogpu", "GPU NOT needed", "Everything runs on a normal CPU.<br><small>A GPU is only needed to <i>train</i> new models (Kaggle).</small>"),
]
page('<p class="lead">What you need on the computer that runs SkinLens (the &ldquo;server&rdquo;). Phones only need a modern browser (Chrome on Android, Safari on iPhone).</p>'
     '<div class="cards3">' + "".join(f'<div class="rq">{icon(i, 30)}<h4>{t}</h4><p>{d}</p></div>' for i, t, d in cards) + '</div>'
     + box('note', 'Short folder path:', 'keep the project in a short path such as <span class="url">C:\\Users\\&lt;you&gt;\\Desktop\\SkinClass</span>. In a very deep folder, installing PyTorch fails on Windows with <i>[WinError 206] The filename or extension is too long</i> (this happened in our test).')
     + '<div class="center" style="margin-top:6mm">' + ill('03_installation.svg', '70%') + '</div>',
     "3", "list", "Requirements")

# 6 folder structure -------------------------------------------------------------------------
TREE = [
    ("SkinClass/", "project root", 0, True),
    ("backend/model/architecture.py", "the network design – the app loads it from here", 1, False),
    ("backup/original_model/", "<b>the trained model</b>", 1, True),
    ("best_skinlens_hybrid.pt", "model weights (414 MB) – must be here", 2, False),
    ("temperature.json", "calibration value", 2, False),
    ("references/test_images/sample.jpg", "sample dermoscopic image (used by seed_demo.py)", 1, False),
    ("clinical_system/", "<b>the app – run all commands in this folder</b>", 1, True),
    ("requirements.txt", "Python packages to install", 2, False),
    ("seed_demo.py", "adds or removes DEMO patients for practice", 2, False),
    ("webapp/", "the website: main.py (pages), ml/ (model, Grad-CAM, LIME)", 2, True),
    ("model_settings.json", "prior correction and High/Medium/Low limits", 3, False),
    ("model_info.json", "text of the Model Info page", 3, False),
    ("static/  templates/", "design, icons, app files (manifest, service worker)", 3, False),
    ("instance/", "created on first start: app.db (database) + storage/ (images)", 2, False),
    ("paper_figures/", "figures shown on the Model Info page", 2, False),
    ("research/", "training pipeline for Kaggle (not needed to run the app)", 2, False),
    ("docs/guide/", "this guide", 2, False),
]
rows = "".join(f'<div class="tr" style="padding-left:{l * 8 + 2}mm"><span class="tn {"dir" if d else ""}">{icon("folder" if d else "note", 15)} {esc(n)}</span>'
               f'<span class="td">{e}</span></div>' for n, e, l, d in TREE)
page(f"""<p class="lead">Only the important folders and files. The app itself is in <b>clinical_system</b>, but it also needs
the model files in <b>backend/model</b> and <b>backup/original_model</b>, so keep the whole <b>SkinClass</b> folder together.</p>
<div class="tree">{rows}</div>
{box('warn', 'Do not move the model:', 'the app looks for <span class="url">..\\backup\\original_model\\best_skinlens_hybrid.pt</span> and <span class="url">..\\backend\\model\\architecture.py</span> next to clinical_system. If you moved them, see &ldquo;Model file not found&rdquo; in Troubleshooting.')}
{box('note', 'Your data:', 'patients and scans are stored in <span class="url">clinical_system\\instance\\</span>. Back up this folder; delete it only if you want to start with an empty database.')}
""", "4", "folder", "Folder structure")

# 7-8 installation ---------------------------------------------------------------------------
page(f"""
<div class="side-wrap"><div class="main-col">
{step(1, 'Install Python 3.12', '<p>Download the Windows installer (64-bit) from python.org and run it. Check that it works:</p>' + cmd('py -3.12 --version') + out(F['py_version_out']))}
{box('tip', 'Why “py” and not “python”?', 'on many Windows PCs, typing <span class="url">python</span> opens the Microsoft Store placeholder and prints <i>“Python was not found”</i> (it did on our laptop). The Python launcher <span class="url">py -3.12</span> always finds the real Python.')}
{step(2, 'Get the project folder', '<p>Copy or unzip the <b>SkinClass</b> folder, e.g. to your Desktop. Then open <b>PowerShell</b> (Start menu &rarr; type PowerShell) and go to the app folder:</p>' + cmd('cd C:\\Users\\<you>\\Desktop\\SkinClass\\clinical_system'))}
{step(3, 'Create a virtual environment', '<p>A private place for SkinLens&#39; packages (takes about ' + str(F['venv_s']) + ' s):</p>' + cmd('py -3.12 -m venv .venv'))}
</div>
<aside class="side"><h4>{icon('os', 18)} macOS / Linux</h4>
<p><b>Not tested on this project.</b> The steps are the same with these differences:</p>
<ul><li>use <span class="url">python3.12</span> instead of <span class="url">py -3.12</span></li>
<li>use <span class="url">.venv/bin/python</span> instead of <span class="url">.\\.venv\\Scripts\\python.exe</span></li>
<li>use <span class="url">/</span> instead of <span class="url">\\</span> in paths</li>
<li>find the IP address with <span class="url">ipconfig getifaddr en0</span> (macOS) or <span class="url">hostname -I</span> (Linux)</li></ul>
<p>If something fails there, check the Troubleshooting section.</p></aside></div>
""", "5", "download", "Installation (Windows)")

page(f"""
{step(4, 'Install PyTorch (CPU version)', '<p>The CPU version is smaller and needs no graphics card (about ' + str(F['torch_s']) + ' s on our laptop):</p>' + cmd('.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip\n.\\.venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu'))}
{step(5, 'Install the other requirements', '<p>FastAPI, timm, OpenCV and the rest (about ' + str(F['req_s']) + ' s):</p>' + cmd('.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt') + out(F['req_last_line']))}
{step(6, 'Check that the model file is found', '<p>The model must be at <span class="url">SkinClass\\backup\\original_model\\best_skinlens_hybrid.pt</span>. This command loads it (takes about ' + str(F['model_check_s']) + ' s):</p>' + cmd('.\\.venv\\Scripts\\python.exe -c "from webapp.ml.predictor import get_predictor; print(get_predictor().display_name)"') + out(F['model_check_out']))}
{box('ok', 'Done!', 'if you see the model name with &ldquo;prior correction (alpha=0.3)&rdquo;, the installation is complete. You do <b>not</b> need to “activate” the virtual environment: every command calls <span class="url">.\\.venv\\Scripts\\python.exe</span> directly.')}
""", "5", "download", "")

# 9 running the website -------------------------------------------------------------------
page(f"""
{step(1, 'Start the server', '<p>In PowerShell, inside <b>clinical_system</b>:</p>' + cmd('.\\.venv\\Scripts\\python.exe -m uvicorn webapp.main:app --host 0.0.0.0 --port 8080') + out(F['start_out'], 'The terminal shows') + '<p>The first start took ' + str(F['first_start_s']) + ' s on our laptop (the model is loaded once). Keep this window open while you use SkinLens.</p>')}
{step(2, 'Open the app', '<p>On the same computer open <span class="url">http://127.0.0.1:8080</span>. You can check the server at <span class="url">http://127.0.0.1:8080/health</span>:</p>' + cmd('Invoke-RestMethod http://127.0.0.1:8080/health') + out(F['health_out']))}
{box('note', 'Port:', '8080 is the port used everywhere in this guide. Without <span class="url">--port</span>, uvicorn uses its default port 8000.')}
""", "6", "play", "Running the website")

page(f"""
{step(3, 'Create the first account', '<p>Choose <b>I&#39;m a doctor / clinic staff</b> &rarr; <b>Create an account</b>. <b>The first account becomes the Admin</b>; later accounts are doctors (the Admin can change roles).</p>')}
{laptop('d00_register', 'Creating the first account (example account used in this guide: demo.doctor)', '88%')}
{step(4, 'Stop the server', '<p>Click the PowerShell window and press <span class="key">Ctrl</span> + <span class="key">C</span>. Your data stays in <span class="url">instance\\</span> and is there again next time.</p>')}
{box('tip', 'Practice data:', 'to try the app with made-up patients, run the command below (server running or not). DEMO patients get a yellow DEMO badge. <span class="url">--remove</span> deletes them again.' + cmd('.\\.venv\\Scripts\\python.exe seed_demo.py --patients 8 --lime 1\n.\\.venv\\Scripts\\python.exe seed_demo.py --remove'))}
""", "6", "play", "")

# 10+ doctor mode --------------------------------------------------------------------------
page(f"""
<div class="intro-ill">{ill('04_doctor_mode.svg', '44%')}<p class="lead">Doctor mode is the full clinic app: patients, scans,
explanations, feedback, reports and (for the Admin) research tools. Everything here works the same on a phone, with a
bottom menu instead of the top bar.</p></div>
<h3 class="hd">Log in</h3>
{laptop('d01_login', 'Enter your username and password, then click Log in', '78%')}
""", "7", "doctor", "Using Doctor mode")

page(f"""
<h3 class="hd">The dashboard</h3>
<p>Totals, possible disease findings, today&#39;s scans and the most recent scans. Start with <b>Add Patient</b> or
<b>Upload &amp; Analyze</b>.</p>
{laptop('d02_dashboard', 'Dashboard (DEMO data)', '86%')}
<h3 class="hd">Add a patient</h3>
<p>Each patient gets an automatic ID such as <b>PAT-2026-0001</b>. Age and sex also help the model.</p>
{laptop('d03_add_patient', 'Add Patient form', '74%')}
""", "7", "doctor", "")

page(f"""
<h3 class="hd">Upload and analyse a photo</h3>
<p>Choose the patient and body site, then <b>Take photo</b> (camera on phones) or <b>Choose from gallery</b>. A preview and
photo tips appear; the photo is shrunk to max 1024 px before it is sent.</p>
{laptop('d04_upload', 'Upload & Analyze', '86%')}
{laptop('d05_analysing', 'While the model works: “Analysing… about 10–20 seconds”', '62%')}
""", "7", "doctor", "")

page(f"""
<h3 class="hd">Reading the result</h3>
{laptop('d06_result_top', 'The coloured banner: the most likely lesion type, the confidence and its level', '86%')}
<table class="lv"><tr><th>Level</th><th>Meaning</th><th>What to do</th></tr>
<tr><td><span class="bdg hi">High</span> &gt; 80%</td><td>the model is fairly sure</td><td>still check it yourself</td></tr>
<tr><td><span class="bdg me">Medium</span> 50–80%</td><td>partly sure; a close 2nd class gets a warning</td><td>look at the top 3</td></tr>
<tr><td><span class="bdg lo">Low</span> &lt; 50%</td><td>not sure (red warning)</td><td>do not rely on it; refer</td></tr></table>
<p class="small">The percentages are prior-corrected probabilities (alpha = 0.3, set in <span class="url">webapp\\model_settings.json</span>).
In our tests, results marked High were correct 90.6% of the time.</p>
""", "7", "doctor", "")

page(f"""
<h3 class="hd">Image, Grad-CAM and LIME</h3>
{plain('d07_result_images', 'Left: the photo. Middle: Grad-CAM – red/yellow = where the model looked most. Right: LIME – the regions that supported the answer (computed in the background, about ' + str(F['lime_s']) + ' s).', '100%', 'framed')}
{box('tip', 'Read the heat-map:', 'if most colour is <b>on the lesion</b>, the result is more trustworthy. If it is on hair, a ruler, ink or the image border, the written explanation warns you – treat that result with extra caution.')}
<h3 class="hd">Explanation and top 3</h3>
{plain('d08_result_explanation', 'Written explanation, “What this means”, the recommendation and the three most likely classes', '100%', 'framed')}
""", "7", "doctor", "")

page(f"""
<h3 class="hd">Doctor feedback</h3>
<p>At the bottom of every result: do you agree (Agree / Disagree + correct diagnosis / Unsure), how useful was the
explanation (1–5 stars), and an optional comment. This is used for the clinician evaluation.</p>
{plain('d09_feedback', 'Feedback form', '82%', 'framed')}
<h3 class="hd">Patient record and history</h3>
{laptop('d10_patient_record', 'Every scan of a patient, newest first, with View and PDF buttons', '80%')}
""", "7", "doctor", "")

page(f"""
<div class="two-60">
<div><h3 class="hd">PDF report</h3>
<p>Click <b>Download Report (PDF)</b> on a result (or <b>PDF</b> in the history). The report contains patient details,
the image, Grad-CAM and LIME, the result and confidence, the top 3, the explanation, the recommendation and the
disclaimer.</p>
<h3 class="hd">Find a patient</h3>
{laptop('d12_patient_list', 'Patient Records: search by ID or name', '100%')}
</div>
<div>{plain('d11_report_pdf', 'Page 1 of a PDF report (DEMO patient)', '100%', 'paper')}</div>
</div>
""", "7", "doctor", "")

page(f"""
<h3 class="hd">Admin: staff accounts</h3>
<p>Only the Admin sees <b>Research</b> and <b>Admin</b>. The Admin can make another account Admin or Doctor. Doctors see
only their own patients; the Admin sees all.</p>
{laptop('d13_admin_users', 'Admin page', '78%')}
<h3 class="hd">Admin: research dashboard and export</h3>
{laptop('d14_research', 'Scans per class, confidence levels, doctor agreement, timings, disagreements; anonymised CSV export (patient ID only)', '86%')}
""", "7", "doctor", "")

# mobile -------------------------------------------------------------------------------------
page(f"""
<div class="intro-ill">{ill('06_phone_install.svg', '32%')}<p class="lead">You can use SkinLens on a phone in two ways:
quickly over the <b>same Wi-Fi</b> (a web page), or as an <b>installed app</b> with its own icon (needs HTTPS – we use a
free Cloudflare tunnel).</p></div>
<h3 class="hd">(a) Same Wi-Fi – the quick way</h3>
<div class="two-60"><div>
{step(1, 'Start the server for the network', cmd('.\\.venv\\Scripts\\python.exe -m uvicorn webapp.main:app --host 0.0.0.0 --port 8080') + '<p>It prints the phone address, e.g. <span class="url">' + F['phone_url_example'] + '</span>.</p>')}
{step(2, 'Or find the IP address yourself', cmd('ipconfig') + out(F['ipconfig_out']) + '<p>Use the IPv4 address of your Wi-Fi adapter: <span class="url">http://&lt;laptop-IP&gt;:8080</span>.</p>')}
{box('tip', 'Firewall:', 'the first time, Windows asks if Python may use the network: tick <b>Private networks</b> &rarr; <b>Allow</b>.')}
{step(3, 'Open it on the phone', '<p>Phone on the <b>same Wi-Fi</b>, type the address in Chrome or Safari. To get an icon: browser menu &rarr; <b>Add to Home screen</b>.</p>')}
</div><div>{phone('m01_start', 'Start screen on a phone', '92%')}</div></div>
""", "8", "phone", "Using the mobile app")

page(f"""
<h3 class="hd">(b) Install as an app – HTTPS with a free Cloudflare quick tunnel</h3>
<p>Phones only offer <b>Install</b> (and the offline page) on secure <b>https://</b> addresses. A Cloudflare quick tunnel
gives your laptop a temporary https address – no account needed.</p>
{step(1, 'Install cloudflared (once)', cmd('winget install --id Cloudflare.cloudflared') + box('note', 'Then open a NEW PowerShell window.', 'In the old window you get <i>“cloudflared is not recognized”</i>. Without a new window you can use the full path:') + cmd('& "C:\\Program Files (x86)\\cloudflared\\cloudflared.exe" --version') + out(F['cloudflared_version']))}
{step(2, 'Start the tunnel (server must be running)', cmd('cloudflared tunnel --url http://localhost:8080') + out(F['tunnel_out'], 'It prints (address changes every time)'))}
{step(3, 'Install on the phone', '<p>Open the <b>https://…trycloudflare.com</b> address. <b>Android (Chrome):</b> tap <b>Install SkinLens</b> on the start screen (or menu &rarr; <i>Install app</i>). <b>iPhone (Safari):</b> <b>Share</b> &rarr; <b>Add to Home Screen</b>. SkinLens then opens full screen from its own icon.</p>')}
{box('warn', 'The tunnel is public:', 'anyone with the address can open the app while the tunnel runs. Create staff accounts first, then restart the server with <span class="url">$env:ALLOW_REGISTRATION="0"; $env:COOKIE_SECURE="1"</span> before the uvicorn command, and stop the tunnel with Ctrl + C when you are done.')}
""", "8", "phone", "")

page(f"""
<h3 class="hd">(c) Patient mode – “Check my skin”</h3>
<div class="intro-ill">{ill('05_patient_mode.svg', '40%')}<div><p><b>What it does:</b> no login; take or choose a photo, see the
most likely type in plain words, the confidence level, the heat-map, clear advice, and download a one-page summary for a doctor.</p>
<p><b>What it does not store:</b> nothing goes into the database. The photo is analysed in memory and the upload&#39;s
temporary file is deleted immediately; the result exists only on the phone screen. Limit: 10 checks per hour per phone.</p></div></div>
<div class="phones3">
{phone('m03_check', '1. Take photo or choose from gallery', '100%')}
{phone('m04_preview_tips', '2. Preview, size and photo tips', '100%')}
{phone('m05_patient_result', '3. Advice and result', '100%')}
</div>
""", "8", "phone", "")

page(f"""
<div class="phones3">
{phone('m06_patient_heatmap', 'Swipe or tap to see where the app looked', '100%')}
{phone('m07_patient_summary', 'Download summary; the disclaimer is on every screen', '100%')}
{plain('m08_patient_summary_pdf', 'The one-page summary PDF', '100%', 'paper')}
</div>
<table class="lv"><tr><th>The app found…</th><th>Advice shown to the patient</th></tr>
<tr><td>melanoma, basal cell carcinoma, actinic keratosis (and SCC)</td><td>“This may need attention. Please see a dermatologist soon.”</td></tr>
<tr><td>mole, benign keratosis, dermatofibroma, vascular lesion</td><td>“This looks less concerning, but only a doctor can confirm. See a doctor if it changes, bleeds, itches or grows.”</td></tr>
<tr><td>any result with <b>Low</b> confidence</td><td>“The app is not sure. Please see a doctor.”</td></tr></table>
""", "8", "phone", "")

page(f"""
<h3 class="hd">Doctor mode on a phone</h3>
<p>Same features as on the laptop. The bottom bar has <b>Home</b>, <b>Patients</b>, the big <b>Scan</b> button, <b>Info</b>
and <b>More</b> (Add Patient, Research, Admin, start screen). On a result, swipe between Image, Grad-CAM and LIME.</p>
<div class="phones3">
{phone('m02_login', 'Doctor login', '100%')}
{phone('m09_dashboard', 'Dashboard with bottom bar', '100%')}
{phone('m10_doctor_gradcam', 'Result: Grad-CAM tab', '100%')}
</div>
""", "8", "phone", "")

# good photo ------------------------------------------------------------------------------
page(f"""
<p class="lead">The model was trained on clear, close-up (dermoscopic) images. A good photo gives a more reliable result.</p>
{ill('07_good_photo.svg')}
<div class="cards4">
<div class="ok-c">{icon('check', 22)}<b>Good light</b><p>Daylight or a bright room. No flash glare or shadows.</p></div>
<div class="ok-c">{icon('check', 22)}<b>In focus</b><p>Tap the screen on the spot to focus before you take the photo.</p></div>
<div class="ok-c">{icon('check', 22)}<b>About 10 cm away</b><p>Close enough that the spot fills a good part of the picture.</p></div>
<div class="ok-c">{icon('check', 22)}<b>Lesion in the centre</b><p>Keep hair, rulers and ink marks off the spot if you can.</p></div>
</div>
{box('warn', 'Ask first:', 'always get the person&#39;s consent before photographing their skin.')}
""", "9", "camera", "Taking a good photo")

# troubleshooting -------------------------------------------------------------------------
TS = [
    ("warn", "“Python was not found”", "Typing <span class='url'>python</span> opens the Microsoft Store placeholder.", "Use the launcher: <span class='url'>py -3.12 …</span> (as in this guide). <span class='tested'>tested</span>"),
    ("warn", "Install fails: [WinError 206] filename too long", "The project is in a very deep folder.", "Move SkinClass to a short path, e.g. the Desktop, and repeat the installation. <span class='tested'>tested</span>"),
    ("warn", "Port already in use", f"Error: <i>{F['port_error']}</i>", "Another server is still running. Close it (Ctrl + C), or find it with <span class='url'>netstat -ano | findstr :8080</span>, or use <span class='url'>--port 8081</span>. <span class='tested'>tested</span>"),
    ("warn", "Requirements install fails", "Network or proxy problem, or a wrong Python version.", "If this happens: check the internet connection, run the pip upgrade command again, and make sure <span class='url'>py -3.12 --version</span> shows 3.12."),
    ("warn", "Model file not found", "The server log says the model could not be loaded.", "If this happens: check that <span class='url'>SkinClass\\backup\\original_model\\best_skinlens_hybrid.pt</span> and <span class='url'>SkinClass\\backend\\model\\architecture.py</span> exist next to clinical_system, then run the model check command (section 5)."),
    ("warn", "LIME is slow", f"LIME took about {F['lime_s']} s on our CPU.", "Normal – it runs in the background and the page shows it when ready. To make it faster, start the server with fewer samples: <span class='url'>$env:LIME_SAMPLES=\"30\"</span> before the uvicorn command (if this happens)."),
    ("warn", "Phone cannot open the link", "Different network, firewall or wrong address.", "Same Wi-Fi on both devices; use the address printed at start-up; allow Python on <b>Private networks</b> in Windows Firewall; or use the Cloudflare tunnel (section 8b). (if this happens)"),
    ("warn", "Camera does not open", "Some browsers only allow the camera on secure pages.", "If this happens: use <b>Choose from gallery</b> (take the photo with the camera app first), or open the app through the https tunnel."),
    ("warn", "No “Install SkinLens” button", "Installing needs HTTPS (or the laptop itself).", "Use the Cloudflare tunnel. On iPhone there is never a button: use Share &rarr; Add to Home Screen."),
    ("warn", "“cloudflared is not recognized”", "The PowerShell window was opened before the install.", "Open a new PowerShell window, or use the full path <span class='url'>C:\\Program Files (x86)\\cloudflared\\cloudflared.exe</span>. <span class='tested'>tested</span>"),
    ("warn", "Login problems / reset demo data", "Wrong password, or you want to start again.", "Usernames are lower-case. The Admin can change roles; there is no password reset page (if a password is lost, create a new account). Remove practice data with <span class='url'>seed_demo.py --remove</span>. To start completely fresh: stop the server and delete the <span class='url'>instance</span> folder (deletes ALL patients)."),
]
def ts_rows(items):
    return "".join(f"<tr><td class='tsp'>{icon('warn', 18)}<b>{p}</b><div class='small'>{c}</div></td><td>{f}</td></tr>" for _, p, c, f in items)


page(f"""<div class="intro-ill">{ill('08_troubleshooting.svg', '30%')}<p class="lead">Common problems and fixes.
<span class="tested">tested</span> means we saw the problem and checked the fix on our laptop; the others are marked
“if this happens”.</p></div>
<table class="ts"><tr><th style="width:40%">Problem</th><th>Fix</th></tr>{ts_rows(TS[:6])}</table>
""", "10", "wrench", "Troubleshooting")
page(f"""<h3 class="hd">Troubleshooting (continued)</h3>
<table class="ts"><tr><th style="width:40%">Problem</th><th>Fix</th></tr>{ts_rows(TS[6:])}</table>
{box('tip', 'Still stuck?', 'read the last lines printed in the server window – they usually name the problem – and check that you ran every command inside the <b>clinical_system</b> folder.')}
""", "10", "wrench")

# model and results -------------------------------------------------------------------------
figs = sorted((G / "figures").glob("*.png"))
fig_html = "".join(f'<figure class="fig"><img src="figures/{p.name}"><figcaption>{p.stem.split("_", 1)[1].replace("_", " ").capitalize()}</figcaption></figure>' for p in figs[:4])
page(f"""
<div class="two">
<div><h4>Model</h4><p>Hybrid of <b>EfficientNetV2-S</b> + <b>Swin Transformer V2-Base</b> + a small network for
age, sex and body site (108 million parameters, input 384 × 384). Prior correction (alpha = 0.3) adjusts the
probabilities for how common each lesion type is.</p></div>
<div><h4>Data</h4><p><b>HAM10000</b>: 10,015 dermoscopic images of 7,470 lesions, 7 classes. <b>Lesion-grouped split</b>:
photos of the same lesion never appear in both training and test (test set: 2,014 images).</p></div>
</div>
<div class="metrics">
<div><b>85.80%</b><span>test accuracy<br><small>85.45% before prior correction</small></span></div>
<div><b>0.763</b><span>macro-F1</span></div>
<div><b>78.8%</b><span>melanoma recall</span></div>
<div><b>90.6%</b><span>of High-confidence results were correct</span></div>
</div>
<div class="figs">{fig_html}</div>
<p class="small" style="margin-top:2mm">Figures from <span class="url">clinical_system\\paper_figures</span> (Kaggle evaluation on the test set). Some curves also show other settings that were compared (e.g. test-time augmentation); the app uses prior correction with alpha = 0.3 and no augmentation.</p>
""", "11", "chart", "Model and results")

# safety -------------------------------------------------------------------------------------
page(f"""
<div class="intro-ill">{ill('09_safety.svg', '42%')}<div>
<ul class="safe">
<li>{icon('shield', 20)}<div><b>Decision support only.</b> SkinLens is a research project, not a certified medical device and not a diagnosis.
A qualified clinician makes every decision; suspicious lesions need a dermatologist.</div></li>
<li>{icon('lock', 20)}<div><b>Patient mode stores nothing.</b> No database entry, no saved photo; the upload&#39;s temporary file is deleted immediately.</div></li>
<li>{icon('key', 20)}<div><b>Passwords are hashed</b> (bcrypt), sessions are signed cookies, every form has a CSRF check; images are shown only to logged-in staff allowed to see that patient.</div></li>
<li>{icon('user', 20)}<div><b>Get consent</b> before photographing anyone, and follow your institution&#39;s rules for patient data.</div></li>
</ul></div></div>
{box('warn', 'Known limits:', 'the model sometimes looks outside the lesion; it was trained mostly on light skin (limited data for darker skin tones); phone photos are not dermoscopy images; results on other clinics&#39; images may be lower.')}
{box('note', 'Public tunnel:', 'the Cloudflare quick tunnel makes the app reachable from the internet while it runs – turn off registration, use secure cookies and stop it when done (section 8b).')}
""", "12", "shield", "Safety and privacy")

# quick reference ----------------------------------------------------------------------------
page(f"""
<div class="qr">
<div class="qr-col">
<h4>{icon('download', 18)} Install (once)</h4>
{cmd('cd C:\\Users\\<you>\\Desktop\\SkinClass\\clinical_system\npy -3.12 -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip\n.\\.venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt')}
<h4>{icon('play', 18)} Start / stop</h4>
{cmd('.\\.venv\\Scripts\\python.exe -m uvicorn webapp.main:app --host 0.0.0.0 --port 8080\n# stop: Ctrl + C in that window')}
<h4>{icon('globe', 18)} HTTPS for installing on a phone</h4>
{cmd('winget install --id Cloudflare.cloudflared\n# open a NEW PowerShell window, then:\ncloudflared tunnel --url http://localhost:8080')}
<h4>{icon('user', 18)} Practice data</h4>
{cmd('.\\.venv\\Scripts\\python.exe seed_demo.py --patients 8 --lime 1\n.\\.venv\\Scripts\\python.exe seed_demo.py --remove')}
</div>
<div class="qr-col">
<h4>{icon('globe', 18)} Addresses</h4>
<table class="qt"><tr><td>This computer</td><td><span class="url">http://127.0.0.1:8080</span></td></tr>
<tr><td>Phone, same Wi-Fi</td><td><span class="url">http://&lt;laptop-IP&gt;:8080</span><br><small>IP: printed at start-up, or <span class="url">ipconfig</span></small></td></tr>
<tr><td>Phone, install</td><td><span class="url">https://….trycloudflare.com</span></td></tr>
<tr><td>Server check</td><td><span class="url">/health</span></td></tr>
<tr><td>Patient mode</td><td><span class="url">/check</span></td></tr></table>
<h4>{icon('key', 18)} Example login (this guide)</h4>
<table class="qt"><tr><td>Username</td><td><b>demo.doctor</b></td></tr><tr><td>Password</td><td><b>Demo@1234</b></td></tr></table>
<p class="small">Only for practice on a test database: it does not exist until you create it with <b>Create an account</b>. Use your own strong password for real use.</p>
<h4>{icon('chart', 18)} Confidence levels</h4>
<table class="qt"><tr><td><span class="bdg hi">High</span></td><td>&gt; 80%</td></tr><tr><td><span class="bdg me">Medium</span></td><td>50–80%</td></tr><tr><td><span class="bdg lo">Low</span></td><td>&lt; 50% – see a doctor</td></tr></table>
<h4>{icon('folder', 18)} Important files</h4>
<p class="small"><span class="url">backup\\original_model\\best_skinlens_hybrid.pt</span> – model<br>
<span class="url">clinical_system\\instance\\</span> – database and images (back up!)<br>
<span class="url">webapp\\model_settings.json</span> – alpha, confidence limits</p>
</div></div>
""", "13", "card", "Quick-reference card")


# ------------------------------------------------------------------ contents page (inserted as page 2)
def contents_html():
    # section -> first page number (cover = 1, contents = 2, content pages start at 3)
    first = {}
    for i, (sec, title, _) in enumerate(PAGES):
        if sec and title and sec not in first:
            first[sec] = (title, i + 3)
    rows = "".join(f'<div class="toc-r"><span class="toc-n">{s}</span><span class="toc-t">{t}</span>'
                   f'<span class="toc-d"></span><span class="toc-p">{p}</span></div>' for s, (t, p) in first.items())
    return (f'<section class="page"><div class="band"><div class="band-i">{icon("list", 26)}</div><div><div class="band-n"></div>'
            f'<h1>Contents</h1></div></div><div class="pb"><div class="toc">{rows}</div>'
            f'<div class="toc-ill">{ill("02_how_it_works.svg", "82%")}</div></div>'
            f'<footer class="pf"><span>{FOOT}</span><span class="pn"></span></footer></section>')


CSS = (G / "build" / "guide.css").read_text(encoding="utf-8")
doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SkinLens – Installation and User Guide</title>'
       f'<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
       f'<style>{CSS}</style></head><body>{COVER}{contents_html()}{"".join(p for _, _, p in PAGES)}'
       '<script>document.querySelectorAll(".page").forEach((p,i)=>{const n=p.querySelector(".pn");if(n)n.textContent=(i+1);});</script>'
       '</body></html>')
(G / "SkinLens_User_Guide.html").write_text(doc, encoding="utf-8")
print("pages:", 2 + len(PAGES))

from playwright.sync_api import sync_playwright  # noqa: E402
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    pg = b.new_page()
    pg.goto((G / "SkinLens_User_Guide.html").as_uri())
    pg.wait_for_load_state("networkidle")
    pg.evaluate("document.fonts.ready")
    over = pg.evaluate("""[...document.querySelectorAll('.page')].map((p,i)=>{const b=p.querySelector('.pb');
        return b && b.scrollHeight > b.clientHeight + 2 ? (i+1)+':'+(b.scrollHeight-b.clientHeight) : null}).filter(Boolean)""")
    print("overflowing pages (page:px):", over or "none")
    pg.pdf(path=str(G / "SkinLens_User_Guide.pdf"), format="A4", print_background=True, prefer_css_page_size=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()
print("wrote", G / "SkinLens_User_Guide.pdf")
