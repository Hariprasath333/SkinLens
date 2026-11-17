# Validation Lab (TRL 4: system validated in a laboratory environment)

The Validation Lab tests the **integrated** SkinLens system against acceptance criteria that are fixed before
anything runs. Every test uses the app's own code:
- **Accuracy and robustness tests** go through `webapp.analysis.compute`, the same function that analyses real patients. That covers the same decoding, preprocessing, metadata encoding, prior correction (alpha = 0.3) and confidence levels.
- **The load test** uses the real HTTP API.
- **The speed test** uses the real PDF builder.

Test images are never stored as patient records.

## Where things are

| What | Where |
|---|---|
| Test data | `validation_data/`: the `HAM10000 dataset/` images, `test_manifest.csv` (2,014 lesion-grouped test images) and `test_predictions.npz` (the notebook's predictions) |
| Code | `webapp/validation/`: `criteria.py` (acceptance criteria), `runner.py` (tests T1–T7, resumable), `metrics.py`, `charts.py`, `report.py` (the TRL-4 PDF), `identity.py` (system identity) |
| Results | `validation_results/<date-time>/` (one folder per run): `run.json`, `state.json`, `predictions.csv`, `T1`–`T7` metrics (JSON and CSV), `charts/`, `SkinLens_TRL4_Lab_Validation_Report.pdf` |
| Page | **Validation Lab** in the admin menu (`/admin/validation`). Admins only. |
| About page | `/about`: app version, model file and SHA-256, and the latest validation status |

`run.json` holds three things recorded when the run starts:
- the frozen acceptance criteria and test settings, with their SHA-256;
- the system identity: model hash, versions, OS, CPU, RAM and GPU;
- the list of tests.

## The tests

| Test | What | Criterion |
|---|---|---|
| T1 | Clinical accuracy: accuracy, balanced accuracy, macro-F1, per-class precision/recall/specificity/F1/AUC, confusion matrix, melanoma recall, ECE (15 bins), accuracy by confidence level | Full set (2,014 images): accuracy within 1.5 points of 85.80%. The 300-image sample only reports the 95% CI. |
| T2 | Integration agreement: the app vs `test_predictions.npz` (`y_pred_corrected`), matched by `image_id` | ≥ 99% identical class; the largest probability difference is reported |
| T3 | Robustness on 200 images: blur, darker −30%, brighter +30%, rotation 90° and 180°, JPEG quality 50, phone resize, 90% off-centre crop | Average "same prediction as original" ≥ 85% |
| T4 | Speed on 50 images: preprocessing, prediction, Grad-CAM, measurements, LIME, PDF, full analysis | 95th percentile of prediction + Grad-CAM < 30 s |
| T5 | 100 analyses in a row, then 5 users at once, through the real HTTP API (patient mode on a separate server with a throw-away data folder) | 0 crashes, 0 failed requests |
| T6 | Photo quality check on blurred, dark, tiny and non-skin images (the non-skin images are generated textures, not photos) | ≥ 90% correctly warned in each group (set by us before the first run) |
| T7 | The complete automated test suite | 0 failures |

## Running

**From the page.** Go to Validation Lab and click **Run all** (T1 sample + T2–T7, about 3 hours on a laptop CPU), **T1 on all 2,014 images**, or a single test. The page shows a progress bar for each test. **Resume** continues an interrupted run without redoing finished images. **Download report** gives the PDF.

**From PowerShell** (inside `clinical_system`):

```powershell
# T1 on the 300 sample + T2-T7
.\.venv\Scripts\python.exe -m webapp.validation.runner --new --tests T1,T2,T3,T4,T5,T6,T7
```

```powershell
# full T1: all 2,014 test images
.\.venv\Scripts\python.exe -m webapp.validation.runner --new --tests T1 --t1 full
```

```powershell
# resume an interrupted run (finished images are kept)
.\.venv\Scripts\python.exe -m webapp.validation.runner --run validation_results\2026-11-15_140000
```

```powershell
# rebuild only the report of a run
.\.venv\Scripts\python.exe -m webapp.validation.runner --run validation_results\2026-11-15_140000 --report-only
```

**How long the full T1 run takes.** Prediction alone takes 1–3 s per image on this laptop (it slows down when it gets warm), so 2,014 images take **40–100 minutes**. Grad-CAM is not needed for T1. Leave the laptop plugged in and don't run other heavy programs at the same time. Speed figures (T4, T5) should only be measured with nothing else running.

## History of T2 (integration agreement)

1. **First `test_predictions.npz`: invalid.** Its probabilities were nearly uniform and it scored 58.4%, so it did not contain the trained model's outputs. It was replaced with a new export from the trained checkpoint: 85.45% raw, 85.80% corrected, macro-F1 0.7628. That file's SHA-256 is recorded in each T2 result.
2. **Run `2026-11-15_141650` (app 1.5.0): T2 FAIL**, 96.92% identical class.
   - Root cause: the app resized images to 384 px with `cv2.INTER_AREA`, while training used `cv2.INTER_LINEAR`. With `INTER_LINEAR` the app reproduced the notebook on every image checked. See `T2_diagnosis.json` in that run.
   - The run is kept as a record. `SUPERSEDED.txt` in its folder explains why its T7 was stopped.
3. **App 1.5.1** resizes with `cv2.INTER_LINEAR`, as in training. Before the final run, 40 images that had not been used in the diagnosis all matched the notebook's class.
4. **The final run** (the next folder in `validation_results/`) ran T1 on all 2,014 images plus T2–T7 on app 1.5.1. The acceptance criteria were unchanged.

If you export `test_predictions.npz` again later, load the trained weights first and keep the order of `test_manifest.csv`. The file must contain `image_id`, `y_true`, `probs_raw`, `y_pred_raw`, `probs_corrected`, `y_pred_corrected`, `class_names`, `alpha` and `log_prior`. A quick check: corrected accuracy should be about 0.858.
