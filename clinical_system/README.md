# SkinLens Clinical

A skin-lesion decision-support system with two parts:

1. **`research/`**: downloads public datasets, merges and cleans them, splits them without patient leakage, trains and compares 5 models plus an ensemble, runs 5-fold cross-validation, and writes the paper tables and graphs.
2. **`webapp/`**: a hospital-style web app. It has staff login, patient records with automatic IDs, image upload and analysis, Grad-CAM + LIME explanations, a plain-English explanation, confidence levels, and PDF reports.

> For research and decision-support only. Not a substitute for professional medical diagnosis.

Your original project (`backend/`, `frontend/`, `references/`, `paper_work/`) is unchanged. The original model is backed up in `../backup/original_model/` with SHA-256 checksums. The web app uses that backup copy until you train and export a new model.

---

## 1. Install (Windows, Python 3.12)

```powershell
cd C:\Users\harip\Desktop\SkinClass\clinical_system
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Run the web app

```powershell
cd C:\Users\harip\Desktop\SkinClass\clinical_system
.\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080 and click **Create an account**. The **first account becomes the Admin**. Later accounts are doctors, and the Admin can promote them on the Admin page.

Typical use:
1. **Add Patient**. The ID is created automatically: `PAT-2026-0001`, `PAT-2026-0002`, and so on.
2. **Upload & Analyze**. Pick the patient and body site, then upload a JPG or PNG.
3. The **Result** page shows the image, the Grad-CAM heat-map and the LIME explanation side by side. It also shows the top 3 classes, the confidence level (High / Medium / Low, with a warning when Low), the plain-English explanation, "What this means" and a recommendation.
4. Everything is saved automatically. **Patient Records** has search by ID or name and the full scan history per patient. **Download Report** gives a PDF.

### Use on a phone (same Wi-Fi)
1. Connect the laptop and the phone to the **same Wi-Fi**.
2. Start the server so that other devices can reach it:
   ```powershell
   cd C:\Users\harip\Desktop\SkinClass\clinical_system
   .\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 0.0.0.0 --port 8080
   ```
   The terminal prints the address, e.g. `Open on your phone (same Wi-Fi): http://192.168.1.23:8080`.
3. The first time, Windows asks whether Python may use the network. Tick **Private networks** and click **Allow**. Without this, the phone cannot connect.
4. Type that address into the phone's browser (Chrome on Android, Safari on iPhone). The start screen offers **I'm a doctor / clinic staff** or **Check my skin (patient)**.

Over plain `http://` the app works fully (camera, upload, results, PDF). However, phones only allow **installing** it as an app and its offline page over **HTTPS**. For that, use the tunnel below.

### Install as an app (HTTPS with a free Cloudflare quick tunnel)
1. Install `cloudflared` (no account needed): `winget install --id Cloudflare.cloudflared`
2. Keep the server running (step 2 above), and in a second terminal run:
   ```powershell
   cloudflared tunnel --url http://localhost:8080
   ```
3. It prints an address like `https://random-words.trycloudflare.com`. Open it on the phone:
   - **Android (Chrome):** tap **Install SkinLens** on the start screen, or use the menu: *Install app* / *Add to Home screen*.
   - **iPhone (Safari):** tap **Share**, then **Add to Home Screen**.
4. SkinLens now opens from its own icon, full screen, like a normal app.

**Important:** a quick tunnel puts the app on the **public internet** for as long as it runs; anyone with the address can open it. For real patient data, first create the staff accounts. Then restart with self-registration switched off and secure cookies on, and stop the tunnel (Ctrl+C) when you are done:
```powershell
$env:ALLOW_REGISTRATION="0"; $env:COOKIE_SECURE="1"
.\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 0.0.0.0 --port 8080
```
Patient-mode checks are limited to 10 per hour per phone (`PATIENT_RATE_LIMIT`, `PATIENT_RATE_WINDOW` in seconds).

### Two modes
| | Doctor / clinic staff | Check my skin (patient) |
|---|---|---|
| Login | yes | no |
| Saves photo / result | yes (patient record, history, PDF report) | **no**: nothing goes into the database. The photo is analysed in memory, the upload's temporary file is deleted immediately, and the result exists only on the phone's screen. |
| Shows | full result, probabilities, top 3, Grad-CAM, LIME, explanation, feedback | top result in plain words, High/Medium/Low, Grad-CAM with one line of explanation, advice, a one-page summary PDF |

The app shell (CSS, JS, icons, offline page) is cached by a service worker (`webapp/static/sw.js`). Pages, results, patient data, images and PDFs are **never** cached.

### Prior correction and confidence levels (`webapp/model_settings.json`)
The app adjusts the model's probabilities for how common each lesion type is ("prior correction", validated on Kaggle):

    p      = softmax(model logits)                       (plain softmax, no temperature scaling)
    score  = log(p + 1e-8) + alpha * log_prior           (alpha = 0.3)
    shown  = softmax(score);  prediction = argmax(score)

`alpha`, `log_prior` (in the order akiec, bcc, bkl, df, mel, nv, vasc) and the confidence thresholds (High > 80 %, Medium 50–80 %, Low < 50 %) live in `webapp/model_settings.json`. Edit that file and restart the app to change them; set `"enabled": false` to switch the correction off.

The corrected probabilities are used everywhere: result, top-3, confidence level, explanation, PDF and database. Each scan's stored model name ends in "+ prior correction (alpha=0.3)". Grad-CAM and LIME explain the predicted class exactly as before.

The correction only applies to a 7-class model in this class order. For a different model (e.g. the 8-class models from `research/`), it is skipped, with a note in the log and on the Model Info page.

### Research features
| Feature | Where | Who |
|---|---|---|
| **Doctor feedback**: Agree / Disagree (+ correct diagnosis) / Unsure, explanation usefulness 1–5 stars, comment. Saved per doctor and scan; submitting again updates it. | bottom of every result page | all staff |
| **Timings**: prediction, Grad-CAM and LIME time are saved with every scan | result page (small print), dashboard, export | – |
| **Research dashboard**: scans per class, confidence split, agreement rate (also per confidence level), average usefulness, average analysis time, table of disagreements; filter All / Real only / DEMO only | **Research** in the menu (`/admin/research`) | admin |
| **CSV export**: one row per scan and feedback, with patient ID only (no name, contact or age) and doctors as numeric IDs. DEMO scans are left out unless you click "incl. DEMO". | button on the research dashboard (`/admin/export.csv`) | admin |
| **Model info**: architecture, dataset, test results, intended use, limitations, disclaimer. Any `*.png` you put in `clinical_system/paper_figures/` is shown there too. | **Model Info** in the menu (`/model-info`); text in `webapp/model_info.json` | all staff |

### Explanations, probabilities and photo checks (v1.5)
| Feature | Where |
|---|---|
| **New logo**: header, login, start screen, favicons (16/32/48 px + `favicon.ico`), apple-touch-icon, PWA icons 192/512 + maskable, PDF headers, offline page. Made from `branding/*.svg` by `branding/make_icons.py`. The service-worker cache is now `skinlens-shell-v2`. | everywhere |
| **All class probabilities** (prior-corrected, as used for the prediction; they add up to 100%), highest first. Patient mode uses plain names with the medical name underneath, red/orange = needs attention, green = usually harmless, plus a calm note. Doctor mode shows names, codes (mel, bcc, …) and one decimal. Each bar has a "What is this?" link. Also in both PDFs. | both result pages, both PDFs |
| **Why this result**: 3–5 plain reasons from real measurements of the photo (colours, edge, symmetry, contrast, where the heat-map is), Grad-CAM and LIME side by side, and **Show full details** (patient mode). Doctors get a measurement table, top 3, confidence level, metadata used and a clinical summary. | result pages, PDFs |
| **Photo quality check**: blur, too dark or bright, too small, no spot or spot off-centre, not skin. Runs on the phone first, then on the server. It shows **Retake photo / Use anyway** and never blocks; in doctor mode it is saved with each scan. | upload screens, result pages |
| **Disease pages** for every class and an **ABCDE self-check** with drawn illustrations | `/learn`, `/learn/abcde` (no login needed) |
| **Validation Lab** (TRL-4 tests T1–T7, report PDF) and **About / version** page (model SHA-256, validation status). See [VALIDATION_LAB.md](VALIDATION_LAB.md). | `/admin/validation` (admin), `/about` |

**Demo data for screenshots:**
```powershell
.\.venv\Scripts\python.exe seed_demo.py                 # 15 DEMO patients, real model predictions, SIMULATED feedback
.\.venv\Scripts\python.exe seed_demo.py --images C:\path\to\lesion_images   # more varied predictions
.\.venv\Scripts\python.exe seed_demo.py --remove        # delete all demo data again
```
Demo patients are named "DEMO Patient NN" and show a yellow **DEMO** badge. Their feedback is made up, so never use it as results. For any number in the paper, use the dashboard's "Real scans only" filter and the default export. Existing databases are upgraded automatically on start (new columns are added and existing data is kept).

Data is stored in `clinical_system/instance/`: `app.db` (SQLite) and `storage/` (images). Back up that folder.

| Setting (environment variable) | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | SQLite in `instance/app.db` | e.g. `postgresql+psycopg://user:pw@host/db` (then `pip install psycopg[binary]`) |
| `MODEL_KIND` | `auto` | `auto` uses `models/best` if it exists, otherwise the original SkinLens model. Also `timm`, `skinlens` |
| `LIME_SAMPLES` | `50` | LIME accuracy vs. speed (about 1.5 s per sample on CPU; LIME runs in the background) |
| `APP_SECRET_KEY` | random, saved in `instance/secret.key` | session-cookie signing key |
| `COOKIE_SECURE` | `0` | set `1` when served over HTTPS |
| `ALLOW_REGISTRATION` | `1` | set `0` to stop self-registration once all staff have accounts |

**Security built in:** bcrypt password hashing; signed, HTTP-only session cookies with an 8-hour lifetime; CSRF tokens on every form; every page except login/register is protected; images are served only to logged-in staff who are allowed to see that patient (doctors see their own patients, the Admin sees all); upload size and type are checked. Before real clinical use you would still need HTTPS, a login rate-limit, audit logs and a data-protection review.

## 3. Train the models (needs a GPU: Kaggle, Colab or a lab PC)

Training five models on about 25,000 images is not practical on a laptop CPU.

**On Kaggle, use the stage-by-stage runner, which is built for the 12-hour session and 20 GB limits.** See **[KAGGLE.md](KAGGLE.md)**:
`bash research/kaggle_stage.sh <stage>` with stages `data`, `train_efficientnet_b3`, `train_resnet50`, `train_densenet121`, `train_convnext`, `train_vit`, `ensemble`, `cv`, `report`, plus the paper experiments `audit`, `leak_test`, `focus`, `lesion_focus_train`, `calibrate`, `external`. **[EXPERIMENTS.md](EXPERIMENTS.md)** explains each one. Upload `kaggle_upload.zip`, built by `python tools/make_kaggle_zip.py`.

On a single machine without time limits (lab PC), you can run everything at once:

```bash
pip install -r requirements.txt
bash research/run_all.sh
```

Or run the steps one at a time:

```bash
python research/download_datasets.py --dry-run          # shows sizes, about 10 GB
python research/download_datasets.py                    # asks before downloading
python research/build_dataset.py --cache-size 512       # merge + clean + de-duplicate + split 70/15/15
python research/train.py --model efficientnet_b3        # also: resnet50 densenet121 convnext_tiny vit_b16
python research/evaluate.py                             # results/ + ensemble + export to models/best
python research/train.py --model efficientnet_b3 --cv-fold 0   # ... up to 4 (5-fold CV)
python research/find_label_issues.py --model efficientnet_b3   # possible wrong labels, for review
```

Then copy `models/best/` back to `clinical_system/models/best/` on your PC. The web app uses the new model automatically on its next start.

### What the pipeline does (for your paper's Methods section)

| Step | Detail |
|---|---|
| Datasets | ISIC 2019 (25,331 images: HAM10000 + BCN20000 + MSK), HAM10000 lesion IDs, ISIC 2018 test as an external test set; PAD-UFES-20 / PH2 / Derm7pt optional. Licences are in `../DATASETS.md`. |
| Labels | 8 classes: AK, BCC, BKL, DF, MEL, NV, SCC, VASC. ISIC "UNK" images are removed. |
| Cleaning | Corrupted, blank or tiny images removed; duplicates found by ISIC ID, MD5 and perceptual hash (catches resized or re-compressed copies); duplicate copies with **conflicting labels** removed; every removal is logged to a CSV. `find_label_issues.py` flags likely mislabelled images (confident learning on 5-fold out-of-fold predictions) for **human review**. |
| Split | 70 / 15 / 15, stratified by class, **grouped by lesion/patient and duplicate cluster**, so the same lesion or a copy of an image is never in both train and test. This is checked automatically and recorded in `data/merged/cleaning_report.md`. |
| Imbalance | Class-weighted cross-entropy (1/√count) by default; `--imbalance sampler` for oversampling; augmentation (flips, rotations, affine, colour jitter, coarse dropout). The test set is **not** rebalanced. |
| Training | ImageNet transfer learning: 2 epochs head-only, then full fine-tuning (backbone LR = head LR / 10), AdamW, warm-up + cosine LR, mixed precision, early stopping on validation macro-F1 (patience 6), temperature scaling on validation. |
| Models | EfficientNet-B3, ResNet-50, DenseNet-121, ConvNeXt-Tiny, ViT-B/16, and an ensemble of the best 3 (chosen on **validation**, never on test). |
| Metrics | Accuracy, precision, recall (sensitivity), specificity, F1, AUC-ROC (macro and per class), melanoma sensitivity/specificity, malignant-vs-benign sensitivity/specificity/AUC, confusion matrices, ROC curves, train/val curves, 5-fold mean ± std, external-test results. |

Output files in `results/`: `results.csv`, `results_per_class.csv`, `results_cv.csv`, `results_external.csv`, `summary.md`, `roc_*.png`, `confusion_*.png`, `curves_*.png`, `roc_all_models.png`. Dataset figures are in `data/merged/`: `class_distribution.png` and `cleaning_report.md`.

## 4. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

| Test file | What it checks | Time (CPU) |
|---|---|---|
| `tests/test_webapp.py` | Login and registration (bcrypt, admin role), wrong password, CSRF, protected pages, patient ID sequence, bad-file upload, prediction + Grad-CAM + LIME + explanation saved to the database and disk, scan history, search, PDF report, doctor/admin access rules, logout. Uses a tiny dummy model. | ~20 s |
| `tests/test_pipeline.py` | Builds a fake dataset with planted problems (exact and near duplicates, corrupted file, UNK label, conflicting-label duplicate, external-test overlap) and checks that each is caught. Trains all 5 architectures briefly, runs CV, evaluation, ensemble, export and label-issue detection, then loads the export in the web app and explains a prediction. | ~7 min |
| `tests/test_kaggle_stages.py` | Runs every Kaggle stage on fake data as separate "sessions": data (portable, resized), training paused by the time limit and resumed from the epoch checkpoint, all 5 models, ensemble/export, CV interrupted and continued with finished folds skipped, and the report. Needs bash (Git Bash on Windows). | ~6 min |
| `tests/test_experiment_stages.py` | The paper-experiment stages (audit, data with packed images, leak_test, focus, lesion_focus_train, calibrate, external) on fake data, each as its own "session". Also checks that every output stays below Kaggle's 500-file limit. | ~6 min |
| `tests/test_real_model.py` | The real trained SkinLens model gives the same probabilities as the original backend, plus Grad-CAM and LIME. Run with `RUN_REAL_MODEL=1`. | ~2 min |

## 5. Folder layout

```
clinical_system/
  research/   common.py  download_datasets.py  build_dataset.py  train.py  evaluate.py
              find_label_issues.py  run_all.sh
  webapp/     main.py (routes, auth)  db.py (models)  analysis.py  report.py (PDF)  config.py
              ml/predictor.py (model loading)  ml/xai.py (Grad-CAM, LIME, explanation)
              templates/  static/
  tests/      test_webapp.py  test_pipeline.py  test_real_model.py
  models/best/        <- exported trained model (created by evaluate.py)
  instance/           <- database + uploaded images (created on first run)
```
