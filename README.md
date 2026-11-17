# SkinLens

**Explainable skin lesion classification with a clinical decision-support web app and an installable phone app.**

Capstone project, B.Tech CSE (Software Engineering), School of Computer Science and Engineering (SCOPE), VIT-AP University.

> For research and decision support only. SkinLens is not a medical device and does not replace a dermatologist.

## Results (HAM10000, held-out test set split by lesion: 2,014 images)

| Setting | Accuracy | Macro-F1 | Melanoma recall | ECE |
|---|---|---|---|---|
| Original model | 85.45% | 0.760 | 0.788 | 0.158 |
| Prior correction, α = 0.3 (deployed in the app) | 85.80% | 0.763 | 0.788 | 0.070 |
| 8-view TTA + prior correction, α = 0.8 (best) | **86.64%** | **0.771** | 0.766 | **0.065** |

- When the app says **High confidence** (90.1% of test images), it is correct **90.6%** of the time.
- **TRL 4:** all 7 laboratory validation tests (T1–T7) passed. See [`clinical_system/VALIDATION_LAB.md`](clinical_system/VALIDATION_LAB.md).

## What is inside

| Folder | What it is |
|---|---|
| [`clinical_system/`](clinical_system/) | **The final system.** FastAPI web app (doctor mode, patient mode, PDF reports, research dashboard, Validation Lab), PWA phone app, research and training scripts, tests. Start with its [README](clinical_system/README.md). |
| `clinical_system/docs/guide/` | User guide (PDF and HTML) |
| `backend/`, `frontend/` | The first prototype (FastAPI + React). See [docs/ORIGINAL_PROTOTYPE.md](docs/ORIGINAL_PROTOTYPE.md). |
| `paper_work/` | Training and analysis scripts used for the paper |
| `references/` | Original training notebook |

## Model

- **Hybrid model:** EfficientNetV2-S (1280 features) + Swin V2-B (1024 features) + metadata MLP (age, sex, body site → 32 features), then a 2336 → 512 → 7 head. 108.3M parameters, 384 px input.
- **Training:** 10 epochs, AdamW + cosine schedule, balanced sampler + 1/√n class weights, label smoothing 0.05, mixed precision, on a free Kaggle GPU.
- **Prior correction:** s = log p + α · log π, which undoes the over-correction from double balancing without retraining.
- **Explanations:** Grad-CAM, LIME and plain-language reasons from lesion measurements.

## Quick start (Windows)

1. **Get the code:**
   ```
   git clone https://github.com/Hariprasath333/SkinLens.git
   cd SkinLens\clinical_system
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. **Get the model:** download `best_skinlens_hybrid.pt` (434 MB) from this repository's **Releases** page and put it in `backup\original_model\` (create the folder at the repository root). `temperature.json` is in the same release.
   - SHA-256 of `best_skinlens_hybrid.pt`: `2d747063394d71f2513ae9a92152e0a4e4b79822ed41faf1cd53453619417b89`
3. **Create your admin account:** `.\.venv\Scripts\python.exe tools\set_password.py <username>`. The first account becomes the Admin.
4. **Run:**
   - On your computer only: `.\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 127.0.0.1 --port 8080`, then open http://127.0.0.1:8080
   - With a free public HTTPS link (Cloudflare quick tunnel): `powershell -ExecutionPolicy Bypass -File .\start_public.ps1`
5. **(Optional) Demo data:** `.\.venv\Scripts\python.exe seed_demo.py`

## Dataset

HAM10000 (Tschandl et al., 2018) is **not** included. Download it from the ISIC Archive or Harvard Dataverse and follow [DATASETS.md](DATASETS.md).
