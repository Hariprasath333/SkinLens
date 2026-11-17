"""
Phone check of the running web app: every page in both modes at 390x844 (and 360x800 for
horizontal scrolling), PWA installability, service worker, uploads, PDFs - and screenshots for
the report.

    # 1) start the app (any database with at least one admin account)
    .venv\\Scripts\\python.exe -m uvicorn webapp.main:app --port 8080
    # 2) in another terminal (uses the Edge or Chrome already installed on Windows)
    pip install playwright pypdfium2
    $env:SKINLENS_USER="your.admin"; $env:SKINLENS_PASS="..."
    .venv\\Scripts\\python.exe tools\\mobile_check.py --url http://127.0.0.1:8080

Writes clinical_system/screenshots/*.png and screenshots/mobile_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SAMPLE = HERE.parent / "references" / "test_images" / "sample.jpg"
UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0.0.0 Mobile Safari/537.36")


def pdf_to_png(pdf: bytes, out: Path) -> None:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf)
    doc[0].render(scale=1.6).to_pil().save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default=str(HERE / "screenshots"))
    ap.add_argument("--channel", default="msedge", help="msedge or chrome (installed browser)")
    a = ap.parse_args()
    user, pw = os.environ.get("SKINLENS_USER"), os.environ.get("SKINLENS_PASS")
    if not (user and pw):
        sys.exit("Set SKINLENS_USER and SKINLENS_PASS (an admin account of the running app).")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = a.url.rstrip("/")
    report = {"viewport": "390x844 (mobile, touch)", "pages": {}, "pwa": {}, "patient_mode": {}, "doctor_mode": {}}

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        # a normal (not private) browser profile: installability is only checked outside incognito
        profile = tempfile.mkdtemp(prefix="skinlens_phone_")
        ctx = p.chromium.launch_persistent_context(
            profile, channel=a.channel, headless=True, viewport={"width": 390, "height": 844}, device_scale_factor=2,
            is_mobile=True, has_touch=True, user_agent=UA, accept_downloads=True, service_workers="allow")
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        # a phone-camera-sized photo (4032 x 3024) to check the on-phone resize to max 1024 px
        from PIL import Image
        big = Path(profile) / "phone_photo.jpg"
        Image.open(SAMPLE).resize((4032, 3024)).save(big, quality=95)

        def overflow(path_label):
            res = {}
            for w, h in ((390, 844), (360, 800)):
                page.set_viewport_size({"width": w, "height": h})
                page.wait_for_timeout(150)
                sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                res[f"{w}px"] = "ok" if sw <= iw else f"HORIZONTAL SCROLL ({sw} > {iw})"
            page.set_viewport_size({"width": 390, "height": 844})
            small = page.evaluate("""() => [...document.querySelectorAll('a.btn, button, .bottom-nav a, select, input:not([type=hidden]):not([type=file]):not([type=radio])')]
                .filter(e => e.offsetParent !== null).map(e => e.getBoundingClientRect())
                .filter(r => r.height > 0 && r.height < 40).length""")
            report["pages"][path_label] = dict(overflow=res, small_tap_targets=small)

        def shot(name, full=False):
            page.screenshot(path=str(out / f"{name}.png"), full_page=full)

        # ---------------- start screen + PWA
        page.goto(base + "/")
        page.wait_for_load_state("networkidle")
        shot("01_start_screen")
        overflow("/ (start)")
        reg = page.evaluate("""async () => { if (!('serviceWorker' in navigator)) return null;
            const r = await navigator.serviceWorker.ready; return r.scope; }""")
        cdp = ctx.new_cdp_session(page)
        errs = cdp.send("Page.getInstallabilityErrors").get("installabilityErrors", [])
        manifest = page.evaluate("fetch('/manifest.webmanifest').then(r => r.json())")
        cached = page.evaluate("""async () => { const out = [];
            for (const k of await caches.keys()) { const c = await caches.open(k);
              for (const r of await c.keys()) out.push(new URL(r.url).pathname); } return out; }""")
        report["pwa"] = dict(service_worker_scope=reg, installability_errors=errs, installable=not errs,
                             manifest_name=manifest.get("name"), cached_files=sorted(cached))

        # ---------------- patient mode
        page.goto(base + "/check")
        overflow("/check")
        page.set_input_files("input[name=image_gallery]", str(big))
        page.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
        report["patient_mode"]["photo_file_bytes"] = big.stat().st_size
        report["patient_mode"]["preview"] = page.inner_text(".preview-info")
        page.evaluate("document.querySelector('.photo-preview').scrollIntoView({block: 'center'})")
        shot("02_patient_upload")
        report["patient_mode"]["spinner_shown_on_submit"] = page.evaluate("""() => {
            document.querySelector('button.submit-btn').click();
            const o = document.querySelector('.busy-overlay');
            return !o.hidden && o.innerText.replace(/\\s+/g, ' ').trim(); }""")
        page.wait_for_selector(".advice", timeout=120_000)
        report["patient_mode"]["advice"] = page.inner_text(".advice")
        report["patient_mode"]["level"] = page.inner_text(".big-badge")
        page.evaluate("window.scrollTo(0, 0)")
        shot("03_patient_result")
        shot("03b_patient_result_full", full=True)
        overflow("/check (result)")
        page.click(".tab[data-i='1']")
        page.wait_for_timeout(500)
        report["patient_mode"]["swipe_tab_heatmap"] = page.evaluate(
            "document.querySelector('#p-swipe').scrollLeft > 0")
        with page.expect_download() as dl:
            page.click("a[download]")
        pdf = Path(dl.value.path()).read_bytes()
        pdf_to_png(pdf, out / "06b_patient_summary_pdf.png")
        report["patient_mode"]["summary_pdf_bytes"] = len(pdf)

        # ---------------- doctor mode
        page.goto(base + "/login")
        overflow("/login")
        page.fill("#username", user)
        page.fill("#password", pw)
        page.click("button[type=submit]")
        page.wait_for_url(re.compile(r".*/dashboard$"))
        shot("04_doctor_dashboard")
        for path in ("/dashboard", "/patients", "/patients/new", "/analyze", "/model-info", "/admin/research",
                     "/admin/users"):
            page.goto(base + path)
            page.wait_for_load_state("networkidle")
            overflow(path)
        page.goto(base + "/dashboard")
        page.click(".menu-btn-bottom")
        report["doctor_mode"]["more_menu_opens"] = page.is_visible("#main-nav a[href='/admin/research']")
        page.goto(base + "/patients")
        first = page.locator("table.table-stack tbody tr a").first.get_attribute("href")
        page.goto(base + first)
        overflow("/patients/<id>")
        code = first.rsplit("/", 1)[1]
        page.goto(f"{base}/analyze?patient={code}")
        page.set_input_files("input[name=image_gallery]", str(SAMPLE))
        page.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
        report["doctor_mode"]["preview"] = page.inner_text(".preview-info")
        report["doctor_mode"]["spinner_shown_on_submit"] = page.evaluate("""() => {
            document.querySelector('button.submit-btn').click();
            return !document.querySelector('.busy-overlay').hidden; }""")
        page.wait_for_url(re.compile(r".*/scans/\d+$"), timeout=120_000)
        scan_url = page.url
        page.wait_for_load_state("networkidle")
        shot("05_doctor_result")
        shot("05b_doctor_result_full", full=True)
        overflow("/scans/<id>")
        page.click(".tab[data-i='1']")
        page.wait_for_timeout(500)
        report["doctor_mode"]["swipe_tab_gradcam"] = page.evaluate("document.querySelector('#r-swipe').scrollLeft > 0")
        pdf = page.request.get(scan_url + "/report.pdf").body()
        pdf_to_png(pdf, out / "06_doctor_report_pdf.png")
        report["doctor_mode"]["report_pdf_bytes"] = len(pdf)
        report["doctor_mode"]["scan_url"] = scan_url.replace(base, "")
        ctx.close()

    bad = [k for k, v in report["pages"].items() if any(x != "ok" for x in v["overflow"].values())]
    report["summary"] = dict(pages_checked=len(report["pages"]), pages_with_horizontal_scroll=bad,
                             installable=report["pwa"]["installable"])
    (out / "mobile_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
