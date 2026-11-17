# SkinLens: Research Paper Readiness Review

*Review date: 12 Nov 2026. Files reviewed: `README.md`, `docs/`, `backend/` (all Python), `frontend/src/`, `references/skinclass.ipynb` (training notebook with all its outputs), `backend/models/temperature.json`.*

> **Short answer.** The engineering is very good: a clean model, a leakage-safe data split, a working web app, and honest documentation. The **research evidence is thin**. There is one model, one data split, one training run, no baselines, no ablations, and a calibration result that is weaker than the README suggests. Most of what is missing is **experiments, not new code**. A ready-to-run experiment kit is in `paper_work/`.

---

## 1. Understanding the project (in simple words)

### What problem it solves
Doctors look at skin spots through a dermoscope (a magnifying camera) to decide whether a spot is harmless (for example a mole) or dangerous (for example melanoma). Some dangerous and harmless spots look almost the same. SkinLens is a computer system that looks at the photo, also uses three simple facts about the patient (**age, sex, and where on the body the spot is**), and then:
1. says which of **7 types** of skin lesion the spot most likely is,
2. gives a confidence value that has been **calibrated** (adjusted so it is more honest),
3. shows a **heat-map** (Grad-CAM) of the part of the image the model focused on.

### The method
| Part | What it is | Detail from your code |
|---|---|---|
| Image branch 1 | **EfficientNetV2-S** (a CNN, good at small textures) | `tf_efficientnetv2_s.in21k_ft_in1k` → 1,280 features |
| Image branch 2 | **Swin Transformer V2-Base** (a vision transformer, good at global shape) | `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` → 1,024 features |
| Patient-data branch | Small neural network (MLP) on a 21-number vector | age (scaled, plus a "missing" flag), sex one-hot (3), body site one-hot (16) → 32 features |
| Fusion | Joins all features together (concatenation) | 1,280 + 1,024 + 32 = 2,336 → Linear(512) → LayerNorm → GELU → Dropout 0.4 → Linear(7) |
| Extra training trick | **Auxiliary head** on the patient-data branch (weight 0.15) | forces the metadata branch to learn something useful by itself |
| Class imbalance | Weighted random sampler (1/count) **and** class-weighted loss (1/√count) | plus label smoothing 0.05 |
| Training | AdamW, LR 1e-5 (backbones) / 1e-4 (heads), cosine schedule, 10 epochs, batch 2 × 4 accumulation steps, mixed precision, gradient clipping 1.0, Kaggle Tesla T4 | best checkpoint chosen by validation macro-F1 |
| Calibration | Global temperature scaling fitted on the validation set | T = 1.064 |
| Explanation | Grad-CAM on `effnet.conv_head` (12×12 map) | explains the CNN branch only |
| Size | 108.3 M parameters, 414 MB checkpoint | |

### Tech stack
PyTorch, timm, Albumentations, OpenCV, scikit-learn (training on Kaggle). The backend is FastAPI with Pydantic v2 and supports single-image prediction, batch prediction and Grad-CAM endpoints. The frontend is React 18, TypeScript, Vite and Tailwind. Data stays in memory and nothing is stored (good for privacy). There are five `verify_*.py` test scripts.

### Dataset
**HAM10000**: 10,015 dermoscopic images of **7,470 different lesions**, in 7 classes. It is very unbalanced:

| akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|
| 327 | 514 | 1,099 | 115 | 1,113 | **6,705 (67%)** | 142 |

**Split:** the split uses `StratifiedGroupKFold` grouped by **lesion ID**, so photos of the same lesion never appear in both training and test. This is a real strength, because many published HAM10000 papers do not do this.

| Split | Images | Lesions |
|---|---|---|
| Train | 6,007 | 4,482 |
| Validation | 1,994 | 1,493 |
| Test | 2,014 | 1,495 |

### Results that exist so far (test set, N = 2,014, one training run)
| Metric | Value |
|---|---|
| Accuracy | 85.45% |
| Macro-F1 | 76.03% |
| Macro precision / recall (recall = balanced accuracy) | 78.49% / 76.47% |
| ECE (calibration error) raw → after temperature | 0.1587 → 0.1411 |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| akiec | 0.71 | 0.81 | 0.76 | 68 |
| bcc | 0.80 | 0.86 | 0.83 | 100 |
| bkl | 0.76 | 0.81 | 0.78 | 205 |
| df | 0.82 | **0.41** | 0.55 | 22 |
| mel | **0.54** | 0.79 | 0.64 | 222 |
| nv | 0.97 | 0.88 | 0.92 | 1,363 |
| vasc | 0.90 | 0.79 | 0.84 | 34 |

Validation macro-F1 by epoch: 0.655, 0.692, 0.715, 0.732, 0.780, 0.774, 0.778, **0.783 (epoch 8, best)**, 0.778, 0.776.

Figures that exist: a confusion matrix and training curves (inside the notebook), plus one Grad-CAM example (`backend/outputs/gradcam_visualization.png`).

---

## 2. Is it enough for a research paper?

| Item | Rating | Why |
|---|---|---|
| Clear problem statement and motivation | **Strong** | The README and presentation notes explain the melanoma-vs-mole problem well, and using patient data the way a doctor does is a good motivation. Add 3–4 cited facts (incidence, dermatologist accuracy) to finish it. |
| Novelty compared with existing work | **Weak** | CNN + transformer hybrids, metadata fusion, temperature scaling and Grad-CAM have each been published on HAM10000 or ISIC many times. Putting all four together is useful engineering, but a reviewer will not see it as a new idea on its own. |
| Literature review / related work | **Missing** | The project has no related-work text and no reference list. |
| Methodology (clear and reproducible) | **Strong**, with gaps | The code is clear: fixed seeds, saved split manifests, every hyperparameter visible, and the backend matches the notebook exactly. Gaps: (a) the printed training log stops at epoch 7 but the history table shows 10 epochs, and the best is epoch 8; the paper must say which epoch was used. (b) The test set was evaluated on a checkpoint re-loaded from a Kaggle model upload, and `training_history.csv` was not exported. (c) Only one seed. |
| Dataset quality and size | **Weak** | One public dataset of about 10k images. It is mostly light-skinned patients, 67% of it is one class, and the test set has only 22 df and 34 vasc images. There is no external test set, so a reviewer cannot tell whether the model generalises. The lesion-grouped split is a plus. |
| Experiments and evaluation metrics | **Weak** | You have accuracy, precision, recall, F1 and ECE. Missing: AUC, melanoma sensitivity/specificity, confidence intervals, more than one seed, statistical tests, inference time, and calibration detail beyond one ECE number. |
| Comparison with existing methods / baselines | **Missing** | No ResNet, DenseNet, ViT, EfficientNet-only or Swin-only model has been trained. There is no ablation that removes metadata, the auxiliary loss or one backbone. Without these, nobody can tell whether the 108M-parameter hybrid is better than a 21M single CNN. |
| Results tables and graphs | **Weak** | One results table, a confusion matrix, training curves and a single Grad-CAM image exist. Missing: comparison table, ablation table, ROC curves, reliability diagram, a grid of Grad-CAM examples (correct and wrong cases). |
| Limitations and future scope | **Strong** (limitations) / **Weak** (future scope) | The README honestly covers skin-tone bias, class imbalance and the CNN-only Grad-CAM. Future work is not written yet. |

### Problems a reviewer will notice

1. **The calibration claim is weak.** On the **validation** set, temperature scaling changed ECE from **0.15411 to 0.15409**, which is no change. T = 1.064 is almost 1. The test drop (0.159 → 0.141) is small, and an ECE of 0.14 is still large for an 85%-accurate model. The README and demo say temperature scaling "prevents overconfident misclassifications", but your own numbers do not support that. **Likely reason:** the weighted sampler trains the model as if all 7 classes were equally common, while real data is 67% nv. One global temperature cannot fix this class-prior shift. `paper_work/analyze.py` fits a bias-corrected temperature (temperature + per-class bias) and draws a reliability diagram so you can check.
2. **Class imbalance is corrected twice.** The weighted sampler **and** the class-weighted loss both push the model toward rare classes. That fits the low melanoma precision (0.54): about 150 non-melanoma images, mostly moles, are called melanoma. Run an ablation (`--sampler none` and `--class-weights none`) and report it.
3. **A single run on a small test set.** With 22 df images, one image changes df recall by 4.5 points. Report 95% confidence intervals and the mean ± std over at least 3 seeds.
4. **Grad-CAM shows only one image** and explains only the CNN branch. Reviewers now expect a quantitative check, or at least a grid of examples.
5. **Comparison with other papers is hard.** Many HAM10000 papers report 95%+ accuracy, but often with random image-level splits where the same lesion appears in train and test. Your lower but honest number has to be **explained and shown** (see idea 5), or it will look weak.

### Final verdict

**Needs major work** for an IEEE/Springer conference. The work needed is mostly running experiments and writing, not rebuilding the system. Expect about 4–6 weeks.

| Level | Fit right now? |
|---|---|
| College journal / college symposium | Yes, ready now |
| National conference | Yes, after adding related work and a baseline table (about 1–2 weeks) |
| IEEE / Springer international conference (Scopus-indexed proceedings) | After experiment Phases 0–2 and idea 1 (about 4–6 weeks) |
| Scopus-indexed journal | After all of the above plus external validation (idea 3) and multi-seed or 5-fold results (about 2–3 months) |

*Tip: before submitting, check that the journal is really indexed in the Scopus source list. Many "fast Scopus" journals are predatory.*

---

## 3. Improvements that add novelty

### Idea 1: "Know when to refer": calibrated, uncertainty-aware triage ★ Recommended first
- **What to do:** (a) Replace plain temperature scaling with **bias-corrected temperature scaling**, which fixes the class-prior shift from the weighted sampler. (b) Add **conformal prediction**: instead of one label, the model outputs a small *set* of possible diagnoses that contains the true one with a guaranteed probability (for example 95%), with a class-wise version so rare classes are also covered. (c) Add **selective prediction**: the model answers only when confident and sends the rest to a dermatologist. Report accuracy at 90/80/70% coverage and how many melanomas are caught.
- **Why it helps the paper:** It turns the current weakest claim (calibration) into the main contribution. "A system that knows when it is unsure" is clinically meaningful, and conformal triage on a multimodal CNN+Transformer model is much less common than another accuracy number. It also fits the web app directly, which could show "possible diagnoses: {mel, nv} → refer".
- **Difficulty:** Easy–Medium. **No retraining needed.** It needs one evaluation-only GPU run (about 20 minutes) to save validation and test logits. `paper_work/analyze.py` already does the analysis.
- **Time:** 1–2 weeks, including writing.

### Idea 2: Metadata-conditioned fusion and robustness to missing patient data
- **What to do:** Replace plain concatenation with **FiLM fusion**, where patient data scales and shifts the image features. It is already coded as `--preset proposed_film`. Train with **metadata dropout**, where patient data is randomly hidden 30% of the time. Then test with metadata removed, to copy real clinics where age or body site is often not recorded.
- **Why it helps:** It gives a method contribution, and the "missing data" scenario is realistic and rarely tested. The ablation shows how much the metadata really helps.
- **Difficulty:** Medium (needs training runs). **Time:** 2–3 weeks, mostly limited by GPU time.

### Idea 3: External validation (does it work on other hospitals' images?)
- **What to do:** Without retraining, test on the **ISIC 2018 Task 3 official test set** (1,512 images; ground truth has been released, check the ISIC challenge site). If time allows, also test on **PH2** and **Derm7pt** (classes need mapping, for example melanoma vs nevus). If test metadata is missing, use the "missing" flags. That doubles as a missing-metadata test.
- **Why it helps:** "Single dataset, no external test" is the most common reason reviewers reject skin-lesion papers. This also gives a fair comparison with ISIC 2018 leaderboard numbers.
- **Difficulty:** Medium. **Time:** 1–2 weeks.

### Idea 4: Explainability you can measure
- **What to do:** HAM10000 has published **lesion segmentation masks** (Tschandl et al., 2020). Measure how much of the Grad-CAM heat falls inside the lesion (energy-inside-mask, pointing game). Add an explanation for the Swin branch (attention rollout or Grad-CAM on the last Swin stage). Count how often the model focuses on artefacts such as rulers, ink marks or hair, especially on wrong predictions.
- **Why it helps:** It moves explainability from "a nice picture" to a result, and it may find shortcut learning, which reviewers value.
- **Difficulty:** Medium. **Time:** about 2 weeks.

### Idea 5: Data-leakage study (lesion split vs random split)
- **What to do:** Train the same model, or a cheaper single EfficientNet, on a **random image-level split** and compare with your lesion-grouped split.
- **Why it helps:** It shows with numbers why your accuracy is lower than papers claiming 95%+. That protects your result and becomes a contribution about evaluation practice.
- **Difficulty:** Easy. **Time:** 3–5 days (1–2 training runs).

*(Optional, harder: a small reader study with 2–3 dermatologists using the web app with and without AI help. This is very strong for a journal but needs ethics approval and 1–2 months. Another option is distilling the 108M model into a small one for CPU or mobile use and reporting latency.)*

**Recommendation: do Idea 1 first.** It needs no retraining, fixes the weakest existing claim, gives the clearest novelty story, and the code is ready. Then run the Phase 1–2 baselines and ablations below, which every reviewer will require anyway.

---

## 4. Experiment plan

### Phases (in order of priority)

| Phase | Runs | Command (`paper_work/train_eval.py`) | GPU cost (rough, T4) |
|---|---|---|---|
| **0: no training** | Dump val/test logits of the existing model, with TTA and no-metadata versions | `--preset proposed --eval-only --checkpoint ... --tta` | ~0.5 h |
| **1: baselines** (single seed) | ResNet-50, DenseNet-121, ViT-B/16, EfficientNetV2-S only, SwinV2-B only, metadata-only | `--preset resnet50`, `densenet121`, `vit_b16`, `effnet_only`, `swin_only`, `meta_only` | ~1.5–3 h each for single backbones (meta_only takes minutes) |
| **2: ablations** | dual image (no metadata), EffNet+meta, Swin+meta, no aux loss; imbalance: sampler only, weights only | `dual_image_only`, `effnet_meta`, `swin_meta`, `no_aux`, `proposed --class-weights none`, `proposed --sampler none` | ~3–5 h each for dual-backbone runs |
| **3: statistics** | Proposed model + best baseline with 2 more seeds each (3 total). Optionally 5-fold CV of the proposed model. | `--seed 7`, `--seed 123`; `--fold 1..4` | ~4 runs |
| **4: new ideas** | FiLM + metadata dropout (Idea 2); random-split leakage run (Idea 5); external test (Idea 3) | `proposed_film --meta-dropout 0.3` | 1–3 runs |

The GPU-hour figures are estimates. Each epoch prints `minutes=`, so measure your real time after the first run. Kaggle gives roughly 30 GPU hours per week (check your quota), so Phases 1–3 take about 2–3 weeks of quota. Runs resume automatically after the 12-hour limit.

### Baselines to compare against
1. **Your own baselines, trained the same way on the same split** (the most important, because they are a fair comparison): ResNet-50, DenseNet-121, EfficientNetV2-S, ViT-B/16, Swin V2-B.
2. **A published metadata-fusion method:** MetaBlock (Pacheco & Krohling, 2021), or simple late fusion. Your `effnet_meta` preset is the simple late-fusion baseline.
3. **Published numbers** in a separate table marked *"different splits, not directly comparable"*: ISIC 2018 Task 3 top entries (balanced accuracy, from the official leaderboard), and recent HAM10000 papers that state whether they used a lesion-level split.

### Tables the paper should include
| # | Table | Produced by |
|---|---|---|
| T1 | Dataset statistics: class counts per split, lesions vs images | notebook output |
| T2 | **Main comparison**: Accuracy, Balanced accuracy, Macro-F1, Macro-AUC, MEL sensitivity and specificity, with 95% CI, params, inference ms | `analyze.py` Table A + `run_config.json` |
| T3 | **Ablation**: remove metadata / aux loss / each backbone; concat vs FiLM | Table A/B |
| T4 | Class-imbalance strategy ablation | Table A |
| T5 | Per-class precision / recall / F1 / AUC | Table C |
| T6 | **Calibration**: ECE, class-wise ECE, Brier, NLL (raw / temperature / bias-corrected; val and test) | Table D, E |
| T7 | **Triage**: selective prediction (accuracy at coverage) and conformal sets (coverage, set size, per-class coverage) | Table F, G |
| T8 | Robustness: metadata missing at test time; TTA | Table H |
| T9 | Statistical significance (McNemar) and multi-seed mean ± std | Table I, B |
| T10 | Comparison with published results (marked as different protocols) | literature |

### Figures the paper should include
1. Architecture diagram (redraw the README ASCII diagram as a clean figure)
2. Class distribution bar chart
3. Training curves (loss, validation F1 and accuracy)
4. Normalised confusion matrix (`*_confusion.png`)
5. ROC curves per class (`*_roc.png`)
6. **Reliability diagram**, before and after calibration (`*_reliability.png`)
7. **Risk–coverage curve**, all models (`risk_coverage.png`)
8. Grad-CAM grid: 2 correct and 1 wrong example per class
9. Screenshot of the web app results page (shows a working system; good for applied tracks)

### Metrics checklist
Accuracy, **balanced accuracy** (the ISIC 2018 metric), macro precision, recall and F1, macro-AUC, melanoma sensitivity at fixed specificity, malignant-vs-benign sensitivity and specificity, ECE, Brier, NLL, AURC, conformal coverage and set size, parameters, inference time (CPU and GPU), training time. All of these are in `analyze.py`. Report CIs from the **lesion-level bootstrap**.

---

## 5. Paper outline (IEEE format)

### Title (3 options)
1. **SkinLens: Calibrated, Uncertainty-Aware Fusion of CNN, Swin Transformer and Clinical Metadata for Dermoscopic Skin Lesion Classification**
2. **Know When to Refer: Conformal Triage with a Hybrid EfficientNetV2–SwinV2 Metadata-Fusion Network for Skin Cancer Screening**
3. **A Leakage-Free Evaluation of Hybrid CNN–Transformer and Metadata Fusion for Seven-Class Skin Lesion Diagnosis on HAM10000**

### Abstract (150–250 words)
- **Problem and gap:** melanoma vs benign confusion; most models ignore patient data, are over- or under-confident, and are evaluated with lesion leakage.
- **Method:** EfficientNetV2-S and SwinV2-B features fused with a 21-dimensional age/sex/site MLP (auxiliary loss), bias-corrected calibration, and conformal triage that outputs diagnosis sets with guaranteed coverage.
- **Results:** on a lesion-grouped HAM10000 split (2,014 test images), give accuracy, balanced accuracy and macro-F1 with CIs; improvement over the best single-backbone baseline; the ECE reduction; at X% coverage, melanoma sensitivity of Y. Fill these in from `REPORT.md`.

### I. Introduction
- Skin cancer burden; the need for early melanoma detection; dermoscopy is hard even for experts (cite Esteva 2017 and Tschandl 2019 reader studies).
- Gaps: image-only models, unreliable confidence, inflated results from lesion leakage, and no "I'm unsure" option.
- Contributions, as 3–4 bullets: (1) hybrid CNN–Transformer–metadata model; (2) leakage-free evaluation with full baselines and ablations; (3) calibrated, conformal triage; (4) an open, privacy-preserving web system (FastAPI + React) with Grad-CAM.

### II. Related Work
- **Dermoscopic classification:** CNNs (Esteva 2017), the ISIC challenges (Codella 2019), EfficientNet ensembles (Gessert 2020), and vision transformers / Swin for skin lesions.
- **Metadata fusion:** concatenation, MetaBlock (Pacheco & Krohling 2021), and attention-based fusion. Explain how yours differs (dual backbone, auxiliary loss, missing-flag encoding, FiLM).
- **Calibration, uncertainty and explainability:** temperature scaling (Guo 2017), label-shift-aware calibration (Alexandari 2020), conformal prediction (Angelopoulos & Bates 2021), Grad-CAM (Selvaraju 2017), and dataset-bias and leakage studies (Cassidy 2022; Daneshjou 2022 on skin-tone diversity).

### III. Proposed Method
- **Data and preprocessing:** HAM10000 statistics, StratifiedGroupKFold by `lesion_id` (60/20/20), 384×384 resize, ImageNet normalisation, augmentation list (flips, rot90, shift-scale-rotate ±30°, colour jitter), and the 21-dimensional metadata encoding with missing flags.
- **Architecture:** the two backbones (1,280 + 1,024), metadata MLP (21→64→32), fusion head (2,336→512→7), auxiliary metadata head and the loss `L = CE(main) + 0.15·CE(aux)` with class weights and label smoothing. Include the diagram, an equation for the fusion, and the parameter count.
- **Calibration and triage:** temperature / bias-corrected scaling fitted on validation; the conformal set rule `C(x) = {k : p_k(x) ≥ 1 − q̂}` with q̂ chosen on validation; the selective-prediction rule; and Grad-CAM on `effnet.conv_head`.

### IV. Experiments and Results
- **Setup:** Kaggle T4, AdamW (1e-5 / 1e-4), cosine schedule, 10 epochs, effective batch 8, AMP, selection by validation macro-F1, 3 seeds; list the metrics; all baselines trained with the same pipeline.
- **Results:** the main comparison (T2), ablation (T3, T4) and per-class results (T5) with confusion matrix and ROC. Discuss mel (precision 0.54) and df (recall 0.41).
- **Calibration and triage:** reliability diagrams, the ECE table, risk–coverage curve, conformal coverage and set size per class, metadata-missing robustness, and the Grad-CAM grid.

### V. Discussion
- Why fusion helps (or does not): which classes gain from metadata; for example, age and body site are known to help bcc and akiec. Use the ablation numbers.
- Clinical meaning: the cost of false melanoma alarms vs missed melanomas, and how triage turns the low mel precision into a safe "refer" decision.
- Limitations: one dataset and mostly light skin types (Fitzpatrick I–III), small rare classes (22 df test images), 108M parameters, Grad-CAM covering the CNN branch only, no prospective clinical testing.

### VI. Conclusion and Future Work
- Summarise the main numbers and the triage result in 3–4 sentences.
- Future work: external and multi-centre validation (ISIC 2019/2020, PH2, Derm7pt), diverse skin-tone data (DDI, Fitzpatrick17k), model compression for mobile, a reader study with dermatologists.
- Release: code, split manifests and the web app (state what you will publish).

---

## Appendix A: Starting reference list (check every entry before citing)

1. P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset…," *Scientific Data*, 2018.
2. N. Codella et al., "Skin lesion analysis toward melanoma detection 2018: A challenge hosted by ISIC," arXiv:1902.03368, 2019.
3. A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, 2017.
4. P. Tschandl et al., "Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification," *Lancet Oncology*, 2019.
5. M. Tan, Q. Le, "EfficientNetV2: Smaller models and faster training," *ICML*, 2021.
6. Z. Liu et al., "Swin Transformer V2: Scaling up capacity and resolution," *CVPR*, 2022.
7. A. Pacheco, R. Krohling, "An attention-based mechanism to combine images and metadata in deep learning models applied to skin cancer classification," *IEEE JBHI*, 2021.
8. N. Gessert et al., "Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data," *MethodsX*, 2020.
9. C. Guo et al., "On calibration of modern neural networks," *ICML*, 2017.
10. A. Alexandari, A. Kundaje, A. Shrikumar, "Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation," *ICML*, 2020.
11. A. Angelopoulos, S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511, 2021.
12. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *ICCV*, 2017.
13. B. Cassidy et al., "Analysis of the ISIC image datasets: Usage, benchmarks and recommendations," *Medical Image Analysis*, 2022.
14. R. Daneshjou et al., "Disparities in dermatology AI performance on a diverse, curated clinical image set," *Science Advances*, 2022.
15. E. Perez et al., "FiLM: Visual reasoning with a general conditioning layer," *AAAI*, 2018.
16. P. Tschandl et al., "Human–computer collaboration for skin cancer recognition," *Nature Medicine*, 2020. *(The source of the HAM10000 lesion masks: confirm the exact citation on the Harvard Dataverse page.)*

## Appendix B: What was added to the project
- `PAPER_REVIEW.md`: this file.
- `paper_work/train_eval.py`: trains and evaluates all baselines, ablations and new ideas on your exact split. It can load your existing checkpoint, resumes after a Kaggle time-out, and saves logits.
- `paper_work/analyze.py`: makes every table and figure listed above. It runs on a laptop, and its output was checked on synthetic data.
- `paper_work/README.md`: step-by-step commands.

No existing file was changed or deleted.
