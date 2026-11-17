"""
Capture the screenshots used in the user guide from a RUNNING SkinLens server that uses a
temporary demo database (never the real one).

    python capture_screenshots.py register --url http://127.0.0.1:8080      # creates demo.doctor (first = Admin)
    (run seed_demo.py in the server's folder)
    python capture_screenshots.py capture  --url http://127.0.0.1:8080

Writes ../screenshots/*.png and ../screenshots/markers.json (where to draw the numbered click markers).
Needs: pip install playwright pypdfium2 (uses the installed Edge or Chrome).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1]
OUT = GUIDE / "screenshots"
SAMPLE = GUIDE.parents[2] / "references" / "test_images" / "sample.jpg"
USER, NAME, PW = "demo.doctor", "Dr. Demo Doctor", "Demo@1234"
UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0.0.0 Mobile Safari/537.36")
markers: dict[str, list] = {}
timings: dict[str, float] = {}


def box(page, selector, n, label, side="right"):
    b = page.locator(selector).first.bounding_box()
    vw = page.viewport_size["width"]
    vh = page.viewport_size["height"]
    return dict(n=n, label=label, x=b["x"] / vw, y=b["y"] / vh, w=b["width"] / vw, h=b["height"] / vh, side=side)


def shot(page, name, marks=(), full=False, clip_el=None):
    path = OUT / f"{name}.png"
    if clip_el:
        page.locator(clip_el).first.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path), full_page=full)
    markers[name] = list(marks)
    print("saved", path.name)


def register(url):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1366, "height": 768}, device_scale_factor=2)
        pg.goto(url + "/register")
        pg.fill("#full_name", NAME); pg.fill("#username", USER)
        pg.fill("#password", PW); pg.fill("#password2", PW)
        OUT.mkdir(parents=True, exist_ok=True)
        shot(pg, "d00_register", [box(pg, "button[type=submit]", 1, "Create account")])
        pg.click("button[type=submit]")
        pg.wait_for_url("**/dashboard")
        print("registered:", pg.inner_text(".alert"))
        b.close()
    (OUT / "markers_register.json").write_text(json.dumps(markers, indent=2))


def pdf_png(data: bytes, out: Path):
    import pypdfium2 as pdfium
    pdfium.PdfDocument(data)[0].render(scale=2).to_pil().save(out)


def capture(url):
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="guide_shots_")
    from PIL import Image
    big = Path(prof) / "phone_photo.jpg"
    Image.open(SAMPLE).resize((4032, 3024)).save(big, quality=92)
    with sync_playwright() as p:
        # ---------------------------------------------------------------- desktop 1366x768
        b = p.chromium.launch(channel="msedge")
        ctx = b.new_context(viewport={"width": 1366, "height": 768}, device_scale_factor=2, accept_downloads=True)
        pg = ctx.new_page()
        pg.goto(url + "/login")
        pg.fill("#username", USER); pg.fill("#password", PW)
        shot(pg, "d01_login", [box(pg, "#username", 1, "Username"), box(pg, "#password", 2, "Password"),
                               box(pg, "button[type=submit]", 3, "Log in")])
        pg.click("button[type=submit]"); pg.wait_for_url("**/dashboard"); pg.wait_for_load_state("networkidle")
        shot(pg, "d02_dashboard", [box(pg, "a.btn.secondary[href='/patients/new']", 1, "Add Patient", "left"),
                                   box(pg, ".page-head a.btn[href='/analyze']", 2, "Upload & Analyze", "left")])
        pg.goto(url + "/patients/new")
        pg.fill("#name", "DEMO Patient - Guide example"); pg.fill("#age", "45")
        pg.select_option("#gender", "female"); pg.fill("#contact", "demo data - not a real person")
        shot(pg, "d03_add_patient", [box(pg, "#name", 1, "Name"), box(pg, "main button[type=submit]", 2, "Save patient")])
        pg.click("main button[type=submit]"); pg.wait_for_url("**/analyze?patient=*")
        code = pg.url.rsplit("=", 1)[1]
        pg.select_option("#body_site", "back")
        pg.set_input_files("input[name=image_gallery]", str(SAMPLE))
        pg.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
        pg.evaluate("window.scrollTo(0, 0)")
        shot(pg, "d04_upload", [box(pg, "#patient_code", 1, "Choose patient"),
                                box(pg, ".pick-btn:nth-child(2)", 2, "Choose photo", "left"),
                                box(pg, "button.submit-btn", 3, "Analyze image", "left")])
        # the real "Analysing" overlay, shown for the screenshot
        pg.evaluate("document.querySelector('.busy-overlay').hidden = false")
        shot(pg, "d05_analysing")
        pg.evaluate("document.querySelector('.busy-overlay').hidden = true")
        t0 = time.perf_counter()
        pg.click("button.submit-btn")
        pg.wait_for_url("**/scans/*", timeout=180_000)
        timings["analysis_to_result_s"] = round(time.perf_counter() - t0, 1)
        scan_url = pg.url
        t1 = time.perf_counter()
        for _ in range(120):  # LIME runs in the background
            st = pg.request.get(scan_url + "/lime").json()["status"]
            if st != "pending":
                break
            time.sleep(3)
        timings["lime_extra_s"] = round(time.perf_counter() - t1, 1)
        pg.goto(scan_url); pg.wait_for_load_state("networkidle")
        timings["result_timing_text"] = pg.inner_text("#timings") if pg.locator("#timings").count() else ""
        shot(pg, "d06_result_top", [box(pg, "a[href$='report.pdf']", 1, "Download PDF", "left"),
                                    box(pg, ".result-banner .conf", 2, "Confidence level", "left")])
        pg.locator(".card", has_text="Image and explanations").scroll_into_view_if_needed()
        shot(pg, "d07_result_images", clip_el=".card:has-text('Image and explanations')")
        shot(pg, "d08_result_explanation", clip_el=".grid.grid-2:has(.explain)")
        shot(pg, "d09_feedback", clip_el="#feedback")
        pg.goto(url + f"/patients/{code}"); pg.wait_for_load_state("networkidle")
        shot(pg, "d10_patient_record", [box(pg, ".timeline .scan .btn.secondary", 1, "View scan", "left"),
                                        box(pg, ".timeline .scan .btn.ghost", 2, "PDF", "left")])
        pdf = pg.request.get(scan_url + "/report.pdf").body()
        pdf_png(pdf, OUT / "d11_report_pdf.png")
        pg.goto(url + "/patients"); pg.wait_for_load_state("networkidle")
        shot(pg, "d12_patient_list", [box(pg, "input[name=q]", 1, "Search by ID or name")])
        pg.goto(url + "/admin/users"); pg.wait_for_load_state("networkidle")
        shot(pg, "d13_admin_users", [box(pg, "table select", 1, "Change role", "left")])
        pg.goto(url + "/admin/research"); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(1200)
        shot(pg, "d14_research", [box(pg, "a[href='/admin/export.csv']", 1, "Export CSV", "left")])
        pg.goto(url + "/model-info"); pg.wait_for_load_state("networkidle")
        shot(pg, "d15_model_info")
        b.close()

        # ---------------------------------------------------------------- phone 390x844
        ctx = p.chromium.launch_persistent_context(
            prof, channel="msedge", viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True,
            has_touch=True, user_agent=UA, accept_downloads=True)
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        pg.goto(url + "/"); pg.wait_for_load_state("networkidle")
        pg.wait_for_function("!document.querySelector('.install-btn').hidden", timeout=15000)
        shot(pg, "m01_start", [box(pg, ".mode-card[href='/login'], .mode-card[href='/dashboard']", 1, "Doctor / staff"),
                               box(pg, ".mode-card[href='/check']", 2, "Check my skin"),
                               box(pg, ".install-row .install-btn", 3, "Install")])
        pg.goto(url + "/login")
        pg.fill("#username", USER); pg.fill("#password", PW)
        shot(pg, "m02_login", [box(pg, "button[type=submit]", 1, "Log in", "left")])
        pg.goto(url + "/check"); pg.wait_for_load_state("networkidle")
        shot(pg, "m03_check", [box(pg, ".pick-btn:nth-child(1)", 1, "Take photo", "left"),
                               box(pg, ".pick-btn:nth-child(2)", 2, "Gallery", "left")])
        pg.set_input_files("input[name=image_gallery]", str(big))
        pg.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
        pg.evaluate("document.querySelector('.photo-preview').scrollIntoView({block: 'start'}); window.scrollBy(0, -70)")
        shot(pg, "m04_preview_tips")
        pg.click("button.submit-btn")
        pg.wait_for_selector(".advice", timeout=180_000)
        pg.evaluate("window.scrollTo(0, 0)")
        shot(pg, "m05_patient_result", [box(pg, ".advice", 1, "Advice", "left")])
        pg.click(".tab[data-i='1']"); pg.wait_for_timeout(600)
        pg.locator("#p-swipe").scroll_into_view_if_needed()
        pg.evaluate("window.scrollBy(0, -120)")
        shot(pg, "m06_patient_heatmap", [box(pg, ".tab[data-i='1']", 1, "Heat-map tab", "left")])
        pg.locator(".result-actions").scroll_into_view_if_needed()
        pg.evaluate("window.scrollBy(0, 160)")
        shot(pg, "m07_patient_summary", [box(pg, "a[download]", 1, "Download summary", "left")])
        with pg.expect_download() as dl:
            pg.click("a[download]")
        pdf_png(Path(dl.value.path()).read_bytes(), OUT / "m08_patient_summary_pdf.png")
        # doctor result on the phone
        pg.goto(url + "/login")
        pg.fill("#username", USER); pg.fill("#password", PW); pg.click("button[type=submit]")
        pg.wait_for_url("**/dashboard")
        shot(pg, "m09_dashboard", [box(pg, ".bottom-nav .scan-btn", 1, "New scan", "left"),
                                   box(pg, ".menu-btn-bottom", 2, "More", "left")])
        pg.goto(scan_url); pg.wait_for_load_state("networkidle")
        pg.click(".tab[data-i='1']"); pg.wait_for_timeout(600)
        pg.locator("#r-swipe").scroll_into_view_if_needed()
        pg.evaluate("window.scrollBy(0, -150)")
        shot(pg, "m10_doctor_gradcam", [box(pg, ".tab[data-i='1']", 1, "Swipe or tap", "left")])
        ctx.close()
    (OUT / "markers.json").write_text(json.dumps(markers, indent=2))
    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    cmd, url = sys.argv[1], sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:8080"
    {"register": register, "capture": capture}[cmd](url)
