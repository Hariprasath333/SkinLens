r"""
Screenshots of the new features (new logo, all-class probabilities, "why this result", photo quality check,
disease pages, ABCDE, About, Validation Lab) on desktop (1366x768) and phone (390x844).

Starts its OWN server with a throw-away data folder (never your real database), creates a DEMO admin and a
DEMO patient, analyses HAM10000 test images, takes the screenshots, then stops the server and deletes
the temporary folder.

    .venv\Scripts\python.exe tools\feature_screens.py              # app features
    .venv\Scripts\python.exe tools\feature_screens.py --validation # Validation Lab page + report pages

Writes clinical_system/screenshots/*.png
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "screenshots"
UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0.0.0 Mobile Safari/537.36")
USER, PW = "demo.admin", "Demo@12345"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(data_dir: Path, port: int) -> subprocess.Popen:
    import httpx
    env = dict(os.environ, APP_DATA_DIR=str(data_dir), APP_SECRET_KEY="screens", PATIENT_RATE_LIMIT="1000")
    env.pop("DATABASE_URL", None)
    log = open(data_dir / "server.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "webapp.main:app", "--host", "127.0.0.1", "--port", str(port)],
                         cwd=str(HERE), env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            time.sleep(1)
    p.kill()
    raise SystemExit("server did not start, see " + str(data_dir / "server.log"))


def test_image(dx: str, k: int = 0) -> Path:
    sys.path.insert(0, str(HERE))
    from webapp.validation import data as D
    m = D.manifest()
    return Path(m[m["dx"] == dx].iloc[k]["path"])


def shot(page, name, full=False, el=None):
    path = OUT / f"{name}.png"
    if el:
        page.add_style_tag(content=".topbar { position: static !important; }")
        page.locator(el).first.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path), full_page=full)
    print("saved", path.name)


def pdf_pages(pdf: bytes, prefix: str, pages=(0,), scale=1.5):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf)
    for i in pages:
        if i < len(doc):
            doc[i].render(scale=scale).to_pil().save(OUT / f"{prefix}_p{i + 1}.png")
            print("saved", f"{prefix}_p{i + 1}.png")


def features(base: str, tmp: Path) -> None:
    from playwright.sync_api import sync_playwright
    mel = test_image("mel", 0)
    rgb = cv2.imread(str(mel))
    blurred, dark = tmp / "blurred.jpg", tmp / "dark.jpg"
    cv2.imwrite(str(blurred), cv2.GaussianBlur(rgb, (0, 0), 4))
    cv2.imwrite(str(dark), (rgb * 0.2).astype(np.uint8))
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        # ---------------- desktop
        d = b.new_context(viewport={"width": 1366, "height": 768}, device_scale_factor=2, accept_downloads=True)
        pg = d.new_page()
        pg.goto(base + "/register")
        pg.fill("#full_name", "Dr. Demo Admin"); pg.fill("#username", USER); pg.fill("#password", PW); pg.fill("#password2", PW)
        pg.click("form button[type=submit]")
        pg.wait_for_url("**/dashboard")
        ctx2 = b.new_context(viewport={"width": 1366, "height": 768}, device_scale_factor=2)
        lp = ctx2.new_page(); lp.goto(base + "/login"); shot(lp, "07_new_logo_login_desktop"); ctx2.close()
        pg.goto(base + "/patients/new")
        pg.fill("#name", "DEMO Patient - feature screenshots"); pg.fill("#age", "55")
        pg.select_option("#gender", "male")
        pg.click("main button[type=submit]")
        pg.wait_for_url("**/analyze?patient=*")
        pg.select_option("#body_site", "back")
        pg.set_input_files("input[name=image_gallery]", str(blurred))
        pg.wait_for_selector(".quality-box:not([hidden])", timeout=15000)
        pg.evaluate("window.scrollTo(0, 0)")
        shot(pg, "10_doctor_quality_warning_blurred_desktop")
        pg.set_input_files("input[name=image_gallery]", str(mel))
        pg.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
        pg.wait_for_timeout(500)
        pg.click("button.submit-btn")
        pg.wait_for_url("**/scans/*", timeout=300_000)
        scan_url = pg.url
        for _ in range(150):
            if pg.request.get(scan_url + "/lime").json()["status"] != "pending":
                break
            time.sleep(2)
        pg.goto(scan_url); pg.wait_for_load_state("networkidle")
        shot(pg, "15_doctor_result_desktop")
        pg.locator(".card.explain").scroll_into_view_if_needed()
        shot(pg, "15b_doctor_result_explanation_desktop", el=".grid.grid-2:has(.explain)")
        shot(pg, "15c_doctor_result_full_desktop", full=True)
        pdf_pages(pg.request.get(scan_url + "/report.pdf").body(), "19_doctor_report_pdf", pages=(0, 1))
        pg.goto(base + "/about"); shot(pg, "22_about_page_desktop")
        d.close()

        # ---------------- phone
        prof = tempfile.mkdtemp(prefix="screens_phone_")
        ph = p.chromium.launch_persistent_context(prof, channel="msedge", viewport={"width": 390, "height": 844},
                                                  device_scale_factor=2, is_mobile=True, has_touch=True, user_agent=UA,
                                                  accept_downloads=True)
        pp = ph.pages[0] if ph.pages else ph.new_page()
        pp.goto(base + "/"); pp.wait_for_load_state("networkidle")
        pp.wait_for_timeout(1500)
        shot(pp, "07b_new_logo_start_phone")
        pp.goto(base + "/check"); pp.wait_for_load_state("networkidle")
        pp.locator(".about-you").scroll_into_view_if_needed(); pp.evaluate("window.scrollBy(0, -120)")
        shot(pp, "09_patient_about_you_always_open_phone")
        for name, f in (("blurred", blurred), ("dark", dark)):
            pp.goto(base + "/check"); pp.wait_for_load_state("networkidle")
            pp.set_input_files("input[name=image_gallery]", str(f))
            pp.wait_for_selector(".quality-box:not([hidden])", timeout=15000)
            pp.locator(".quality-box").scroll_into_view_if_needed(); pp.evaluate("window.scrollBy(0, 120)")
            shot(pp, f"11_patient_quality_warning_{name}_phone")
        pp.goto(base + "/check"); pp.wait_for_load_state("networkidle")
        pp.fill("#age", "55"); pp.select_option("#sex", "male"); pp.select_option("#body_site", "back")
        pp.set_input_files("input[name=image_gallery]", str(mel))
        pp.wait_for_function("document.querySelector('.preview-info').textContent.startsWith('Ready')")
        pp.click("button.submit-btn")
        pp.wait_for_selector(".advice", timeout=300_000)
        pp.evaluate("window.scrollTo(0, 0)")
        shot(pp, "12a_patient_result_top_phone")
        pp.locator("#all-probs").scroll_into_view_if_needed(); pp.evaluate("window.scrollBy(0, -70)")
        shot(pp, "12_patient_result_probability_bars_phone")
        pp.wait_for_selector("#why img[alt^='Highlighted']", timeout=300_000)
        pp.locator("#why").scroll_into_view_if_needed(); pp.evaluate("window.scrollBy(0, -70)")
        shot(pp, "13_patient_why_this_result_phone")
        pp.click("details.full-details summary"); pp.wait_for_timeout(400)
        pp.locator(".explain-details").scroll_into_view_if_needed(); pp.evaluate("window.scrollBy(0, -120)")
        shot(pp, "14_patient_full_details_phone")
        shot(pp, "14b_patient_result_full_page_phone", full=True)
        with pp.expect_download() as dl:
            pp.click("a[download]")
        pdf_pages(Path(dl.value.path()).read_bytes(), "20_patient_summary_pdf", pages=(0, 1))
        pp.goto(base + "/learn/mel"); shot(pp, "17_disease_page_melanoma_phone", full=True)
        pp.goto(base + "/learn"); shot(pp, "17b_disease_list_phone")
        pp.goto(base + "/learn/abcde"); shot(pp, "18_abcde_page_phone", full=True)
        pp.goto(base + "/login"); pp.fill("#username", USER); pp.fill("#password", PW); pp.click("button[type=submit]")
        pp.wait_for_url("**/dashboard")
        pp.goto(scan_url); pp.wait_for_load_state("networkidle")
        pp.locator(".pbars.doctor").scroll_into_view_if_needed(); pp.evaluate("window.scrollBy(0, -90)")
        shot(pp, "16_doctor_result_bars_phone")
        ph.close()
        b.close()


def validation(base: str) -> None:
    from playwright.sync_api import sync_playwright
    sys.path.insert(0, str(HERE))
    from webapp import config
    latest = json.loads((config.VALIDATION_RESULTS_DIR / "latest.json").read_text())
    run = config.VALIDATION_RESULTS_DIR / latest["run"]
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        d = b.new_context(viewport={"width": 1366, "height": 768}, device_scale_factor=2)
        pg = d.new_page()
        pg.goto(base + "/register")
        pg.fill("#full_name", "Dr. Demo Admin"); pg.fill("#username", USER); pg.fill("#password", PW); pg.fill("#password2", PW)
        pg.click("form button[type=submit]"); pg.wait_for_url("**/dashboard")
        pg.goto(base + f"/admin/validation?run={latest['run']}"); pg.wait_for_load_state("networkidle")
        shot(pg, "23_validation_lab_desktop")
        shot(pg, "23b_validation_lab_full_desktop", full=True)
        pg.locator("#run-card").scroll_into_view_if_needed()
        shot(pg, "23c_validation_progress_desktop", el="#run-card")
        pg.goto(base + "/about"); shot(pg, "22_about_page_desktop")
        b.close()
    pdf = (run / "SkinLens_TRL4_Lab_Validation_Report.pdf").read_bytes()
    import pypdfium2 as pdfium
    pdf_pages(pdf, "24_trl4_report", pages=range(len(pdfium.PdfDocument(pdf))), scale=1.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="skinlens_screens_"))
    port = free_port()
    srv = start_server(tmp, port)
    try:
        (validation if a.validation else lambda b: features(b, tmp))(f"http://127.0.0.1:{port}")
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        time.sleep(1)
        shutil.rmtree(tmp, ignore_errors=True)
        print("temporary server stopped and its data folder deleted")


if __name__ == "__main__":
    main()
