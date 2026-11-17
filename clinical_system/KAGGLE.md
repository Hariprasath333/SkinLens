# Running the research pipeline on Kaggle, stage by stage

Kaggle stops a session after about **12 hours**, and `/kaggle/working` holds about **20 GB**. The pipeline is therefore split into stages. You run **one stage per session**.
- Each stage saves its results to `/kaggle/working/out`.
- The next session reads those results after you attach the earlier session's output as an **input**.
- Training saves a checkpoint after **every epoch**. A stage stopped by the time limit continues where it left off when you run it again.

```
data ──► train_efficientnet_b3 ─┐
     ├─► train_resnet50 ────────┤
     ├─► train_densenet121 ─────┼─► ensemble ─► cv (1-2 sessions) ─► report
     ├─► train_convnext ────────┤
     └─► train_vit ─────────────┘
```

## One-time setup

1. On your PC, rebuild the upload file if you changed any code:
   ```
   cd C:\Users\harip\Desktop\SkinClass\clinical_system
   .\.venv\Scripts\python.exe tools\make_kaggle_zip.py
   ```
   This writes `SkinClass\kaggle_upload.zip`, which is under 1 MB: code only, with no `.venv`, datasets, weights or database.
2. On Kaggle, go to **Datasets → New Dataset**, upload `kaggle_upload.zip` (Kaggle unzips it), and name it e.g. `skinlens-code`.
3. Your Kaggle account must be **phone-verified** so that notebooks can use the Internet. Only the `data` stage downloads anything.

## The notebook (same for every stage)

Create a notebook. In **Settings**, set **Accelerator = GPU T4 x2** (or P100) and **Internet = On**. Then click **Add Input** and add `skinlens-code` plus the inputs listed for the stage in the table below. Use one cell:

```python
STAGE = "data"          # <- change for each session
import os, glob
os.environ["MAX_HOURS"] = "11"           # stop cleanly before Kaggle's 12-hour limit
script = glob.glob("/kaggle/input/**/clinical_system/research/kaggle_stage.sh", recursive=True)[0]
!bash {script} {STAGE}
```

Run it with **Save Version → Save & Run All (Commit)**. It then runs in the background, even if you close the browser. When it finishes, its `/kaggle/working/out` folder becomes that version's **Output**. To use it in a later session, click **Add Input → Your Work → (that notebook)**.

Running the script without a stage name prints the list of valid stages.

| # | Stage | Add as input (besides `skinlens-code`) | Saves to `out/` | Rough T4 time |
|---|---|---|---|---|
| 1 | `data` | nothing (downloads ~10 GB to `/tmp/raw`, which is not saved) | `data_processed/`: `images_384.tar` (images resized to 384 px), `manifest.csv`, `manifest_naive.csv`, `train/val/test/external_*.csv`, `cleaning_report.md`, `audit/` (about 2–4 GB, checked to be under 6 GB) | 1–2 h |
| 2 | `train_efficientnet_b3` | output of **data** | `runs/efficientnet_b3/holdout/`: `best.pt`, `model_card.json`, `history.csv`, `predictions.npz` | 2–4 h |
| 3 | `train_resnet50` | output of **data** | `runs/resnet50/holdout/` | 1–3 h |
| 4 | `train_densenet121` | output of **data** | `runs/densenet121/holdout/` | 1–3 h |
| 5 | `train_convnext` | output of **data** | `runs/convnext_tiny/holdout/` | 2–4 h |
| 6 | `train_vit` | output of **data** | `runs/vit_b16/holdout/` | 3–6 h |
| 7 | `ensemble` | outputs of **all 5 train** sessions | `ensemble/` (results + `best_model.json`), `models/best/` (**download this for the web app**) | 5 min, no GPU needed |
| 8 | `cv` | output of **data** + output of **ensemble** (+ the previous `cv` output if it paused) | `runs/<best>/cv0` … `cv4`, `label_review/label_issues.csv` | 10–20 h, so expect 2 sessions |
| 9 | `report` | outputs of all train sessions + **ensemble** + the **last cv** session | `results/`: `results.csv`, `results_per_class.csv`, `results_cv.csv`, `results_external.csv`, ROC / confusion / curve PNGs, `summary.md`, `cleaning_report.md` | 5 min, no GPU needed |

**Paper experiments.** EXPERIMENTS.md explains each one in plain words, with the table or figure it produces.

| # | Stage | Add as input (besides `skinlens-code`) | Saves to `out/` | Rough T4 time |
|---|---|---|---|---|
| 10 | `calibrate` | all train outputs + **ensemble** | `calibration/`: ECE table, reliability diagrams, confidence-level accuracy, **`calibration/models/best/` = calibrated web-app model** | 5 min |
| 11 | `focus` | **data** + all train outputs + **ensemble** | `focus/<best>/`: attention-in-lesion tables, box plot, 20 examples; `masks/` (downloaded lesion masks, reused later) | 10–30 min |
| 12 | `leak_test` | **data** + `train_resnet50` output | `leak_test/`: naive vs clean split table, inflation, bar chart | 2–6 h |
| 13 | `lesion_focus_train` | **data** + all train outputs + **ensemble** + **focus** | `lesion_focus/`: before/after table, both focus reports, retrained model | 2–5 h |
| 14 | `external` | all train outputs + **ensemble** + calibrate + leak_test + lesion_focus_train | `external/`: every model and the ensemble on the ISIC 2018 test set | 5–15 min |
| – | `audit` | nothing (re-downloads ~10 GB) | `audit/`: lesion-grouping check, label-conflict check, 20 example pairs | 1 h |

> **Your current `data_processed` was built before the duplicate-detection fix.** Re-run `data` first. Its output now also contains `audit/` and `manifest_naive.csv`, and the images are packed in `images_384.tar`. Then retrain the models.

Image folders in every output are packed into one `.tar` (Kaggle keeps only about 500 files per output). The stages unpack them to local disk automatically. A `*_montage.jpg` next to each `.tar` shows the same images in a single file.

Stages 2–6 are independent, so you can run several notebooks at the same time if your GPU quota allows. Kaggle gives about 30 GPU hours per week; check your quota under **Settings**.

## When a stage pauses

If a stage reaches `MAX_HOURS`, it stops **between epochs**. It keeps everything finished so far plus `last.pt` (the epoch checkpoint) and prints:

```
Time limit reached ... run the SAME stage again
```

Commit that version. Then start a new session with **the same inputs plus this paused output** and run the same stage. The new session finds `last.pt` and continues from the next epoch.

For `cv`, finished folds are skipped. It saves after every fold, so a second session only does the remaining folds.

`out/STATUS.txt` records what each session did: finished, paused, or which fold.

## Settings you can change (environment variables, set with `os.environ[...]` before the `!bash` line)

| Variable | Default | Use |
|---|---|---|
| `MAX_HOURS` | 11 | stop earlier/later (keep ≥ 30 min margin for Kaggle to save the output) |
| `IMG_SIZE` | 384 | short side of saved images in the data stage (all models use ≤ 384) |
| `MAX_DATA_GB` | 6 | the data stage fails if `data_processed` is larger |
| `EPOCHS`, `BATCH_SIZE`, `WORKERS` | 30, 32, 4 | training settings (lower `BATCH_SIZE` if you get a CUDA out-of-memory error) |
| `BEST_MODEL` | best on validation | force the model used by `cv`, e.g. `efficientnet_b3` |
| `DATA_DIR` / `MODELS_DIR` / `OUT_DIR` | auto | point to other folders; `MODELS_DIR` accepts several folders separated by commas |
| `INSTALL` | 1 | `0` skips `pip install -r requirements-kaggle.txt` |

`requirements-kaggle.txt` deliberately leaves out torch, torchvision, numpy and pandas. The script also pins the versions Kaggle already has, so installing the other packages cannot replace them and break the GPU setup.

## After Kaggle

1. Download `out/models/best/` from the **ensemble** output and copy it to `clinical_system/models/best/` on your PC. When the web app restarts, it uses this model instead of the original SkinLens model.
2. Download `out/results/` from the **report** output. The tables and graphs in it go into the paper.
