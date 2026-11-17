# SkinLens user guide

**`SkinLens_User_Guide.pdf`** is the finished guide: 29 A4 pages showing how to install SkinLens, run it, and use it on a laptop and on a phone.

## Files

| Path | What it is |
|---|---|
| `SkinLens_User_Guide.pdf` | The guide (A4, 29 pages, printed with backgrounds) |
| `SkinLens_User_Guide.html` | The HTML the PDF was printed from (open it in a browser) |
| `facts.json` | Timings and terminal output measured during the install test (14 Nov 2026, Windows 11, CPU only) |
| `illustrations/01_cover_hero.svg` … `09_safety.svg` | 9 original flat SVG illustrations in the app colours |
| `screenshots/d00–d15_*.png` | Desktop screenshots (1366×768, scale 2) taken from a temporary DEMO database |
| `screenshots/m01–m10_*.png` | Phone screenshots (390×844, scale 2) |
| `screenshots/markers.json`, `markers_register.json` | Where the numbered red click markers are drawn on each screenshot |
| `screenshots/timings.json` | Analysis and LIME times measured while capturing |
| `figures/fig0–fig3_*.png` | Copies of the evaluation figures from `clinical_system/paper_figures` |
| `build/build_guide.py` | Builds the HTML and prints the PDF |
| `build/guide.css` | Styles for the guide: layout, step cards, boxes, device frames, markers |
| `build/capture_screenshots.py` | Playwright script that registers the demo account and captures the screenshots |

All patients in the screenshots are made-up DEMO data from `seed_demo.py`. No real patient data is used.

## Rebuilding

To rebuild the guide you need `playwright` and `pypdfium2` installed in the venv. Playwright uses the Microsoft Edge that is already installed on Windows.

```powershell
cd C:\Users\<you>\Desktop\SkinClass\clinical_system\docs\guide\build
..\..\..\.venv\Scripts\python.exe build_guide.py
```

To take new screenshots, run them against a server that uses a **temporary** database, never your real one:

```powershell
python capture_screenshots.py register --url http://127.0.0.1:8080
python capture_screenshots.py capture  --url http://127.0.0.1:8080
```
