# Paper experiments: what each one shows and what it produces

All experiments run as stages of `research/kaggle_stage.sh`, one Kaggle session per stage (see KAGGLE.md). Each stage:
- reads earlier results through `MODELS_DIR` (attached notebook outputs);
- reads the processed data through `DATA_DIR`;
- writes to `OUT_DIR`;
- saves a checkpoint after every epoch when it trains;
- packs every image folder into one `.tar`, so each output stays below Kaggle's ~500-file limit.

A `*_montage.jpg` next to each `.tar` shows the same images in a single file.

| Stage | Needs as input | GPU time (T4, rough) | Paper item |
|---|---|---|---|
| `data` (now includes the audit) | – | 1–2 h | Dataset table, cleaning table, audit figure |
| `audit` | – (re-downloads raw data) | 1 h | Same audit, without rebuilding the dataset |
| `leak_test` | data (+ train_resnet50 output, to reuse it) | 2–6 h | **Table: leakage inflation**, Fig. leak_comparison |
| `focus` | data + train outputs + ensemble | 10–30 min | **Table: attention-in-lesion**, Fig. focus_by_class, Fig. examples |
| `lesion_focus_train` | data + train outputs + ensemble (+ focus, to reuse masks) | 2–5 h | **Table: before vs after lesion-focused training** |
| `calibrate` | train outputs + ensemble | 5 min | **Table: ECE before/after**, Fig. reliability diagrams, confidence-level table |
| `external` | train outputs + ensemble (+ leak_test, lesion_focus_train) | 5–15 min | **Table: external validation** |

Run order: `data` → the 5 `train_*` stages → `ensemble` → `calibrate` → `external`, `focus`, `leak_test`, `lesion_focus_train` (any order) → `external` again (to add the variant models) → `report`.

---

## 0. Data audit: are the label conflicts and the lesion grouping right?

**Why:** the first data run reported "HAM10000 lesion ids attached to 0 images" and removed **922 images as label conflicts**. Both numbers needed checking before the data is used for training.

**What we found in the code:**
- **Lesion grouping: probably fine.** The "attach lesion ids" step only fills images whose `lesion_id` is empty. The ISIC 2019 metadata already contains a `lesion_id` for the HAM10000 images, so the step most likely had nothing to fill. Lesions were most likely still grouped correctly, through the ISIC 2019 metadata. The audit now prints the exact counts: how many images already had an id, how many agree with the HAM file, how many were filled.
- **Label conflicts: most were probably not real duplicates.** The first version called two images duplicates when their 64-bit perceptual hashes differed by ≤ 4 bits.
  - Many dermoscopy photos share a dark circular border around the picture. That border dominates the hash, so **different** lesions can match.
  - Matches were also *chained* (A~B and B~C put A, B and C in one cluster), so a few false matches could merge large groups.
  - When labels in such a cluster disagreed, every image in it was removed. When they agreed, all but one were thrown away.
  - We reproduced this with synthetic images. Two different lesions inside the same border were 4 bits apart (counted as duplicates), but their centres were only 0.50 similar. A real resized copy is 0 bits apart and 1.00 similar.

**Fix (now in `build_dataset.py`):** a hash match is only a *candidate*. It counts as a duplicate only when the **centre 60 %** of both images is near-identical (similarity ≥ 0.92) **and** the colours match. Same ISIC ID and identical files (MD5) still count directly.

**Outputs** (`data_processed/audit/` or `audit/`):

| File | What it shows |
|---|---|
| `audit_report.md` | Lesion-grouping counts with an automatic explanation. Old vs new method: images removed, conflict clusters, copies collapsed, largest cluster. How many conflicting pairs are genuine duplicates vs look-alikes. |
| `conflict_pairs.csv` | Every conflicting pair: why it matched, hash distance, similarity, verdict |
| `conflict_examples_montage.jpg`, `conflict_examples.tar` | **20 side-by-side pairs**: the 10 least similar (likely false matches) and the 10 most similar (likely real duplicates with disagreeing labels) |

**For the paper:** one sentence in the dataset section, e.g. "*Perceptual-hash matching alone flagged N_old images; after verification, N_new were true duplicates*", plus the montage as a supplementary figure.

> **Action:** your current `data_processed` was built with the old method. Re-run the `data` stage, then retrain. The new data also has `manifest_naive.csv`, which `leak_test` needs.

---

## 1. `leak_test`: how much does a careless split inflate the results?

**In simple words:** many papers split HAM10000/ISIC images at random. The same lesion, or even the same photo copied twice, can then be in both training and test, so the model is partly tested on images it has already seen. We train **the same ResNet-50 with identical settings** twice:
- **(a) naive split:** random 70/15/15 by image, **no** duplicate removal, **no** lesion grouping (`manifest_naive.csv`);
- **(b) clean split:** our leakage-free split. A finished `train_resnet50` run is reused, because it uses the same settings.

**Outputs** (`leak_test/`):
- `leak_results.csv`, `leak_report.md`: accuracy, macro-F1 and macro-AUC for both splits, plus the **inflation** (naive − clean, in percentage points). External-test results for both models show that the naive model does not generalise better.
- `leak_naive_test_breakdown.csv`: the share of naive test images that have a copy or a same-lesion photo in naive training, and accuracy on those images vs the rest. This isolates the leak effect, because the two test sets are different images.
- `leak_comparison.png`: grouped bar chart.

**For the paper:** a "Data leakage" table, the bar chart, and one sentence in the Introduction: "*Random image-level splitting inflates accuracy by X points on the same model*".

---

## 2. `focus`: where does the best model look?

**In simple words:** a heat-map is only convincing if it falls on the lesion. We measure this on public expert lesion outlines:
- **HAM10000 segmentation masks** (Tschandl et al., Harvard Dataverse, 10,015 masks);
- **ISIC 2018 Task 1** masks (training, validation and test).

Only test images that have a mask are used. The report states how many.

For each image:
- **In-lesion ratio:** the share of the Grad-CAM heat that lies inside the lesion mask (1.0 means the model looks only at the lesion). It is compared with the lesion's share of the image, which is the chance level.
- **IoU:** the overlap between the hot area (heat ≥ 0.5) and the mask.

**Outputs** (`focus/<model>/`):

| File | What it shows |
|---|---|
| `focus_report.md`, `focus_overall.csv` | Mean and median in-lesion ratio and IoU |
| `focus_by_class.csv` | The same per class |
| `focus_by_correctness.csv` | Correct vs wrong predictions. Do mistakes come with looking off-lesion? |
| `focus_per_image.csv` | Every image's values |
| `focus_by_class.png` | Box plot per class with the chance level line |
| `focus_examples_montage.jpg`, `focus_examples.tar` | **20 examples** shown as *original \| Grad-CAM \| mask overlay*: 10 correct and 10 wrong, from low to high focus |
| `masks/masks_index.csv`, `masks/masks_384.tar` | The matched masks. Later stages reuse them. |

**For the paper:** a "Model attention" table (overall, by class, correct vs wrong) and a figure of 4–6 examples from the montage.

---

## 3. `lesion_focus_train`: can we make the model look at the lesion?

**In simple words:** we retrain the best model with two additions that push its attention onto the lesion. Both are used only during training; using the model afterwards needs no mask.
1. **Mask-guided attention loss:** for images with a mask, the model is penalised when its class-activation map lies outside the lesion. Loss = cross-entropy + 0.5 × (1 − share of the activation inside the mask), set by `ATTN_WEIGHT`. It is computed from the last feature map and the classifier weights, so training stays fast. For ViT (no feature map) only the background augmentation is used.
2. **Background augmentation:** with probability 0.5 (`BG_AUG`), the skin around the lesion is blurred or recoloured. Rulers, ink, hair, skin tone and the dermoscope border then stop being useful shortcuts.

The stage then measures **accuracy** (test and external) and **focus** for the model before and after this training, on the same images.

**Outputs** (`lesion_focus/`):
- `lesion_focus_comparison.md` / `.csv`: before vs after table: test accuracy, macro-F1, AUC, melanoma sensitivity, external accuracy/F1, in-lesion ratio, IoU;
- `focus_before/` and `focus_after/`: the full focus reports for both models;
- `runs/<model>/holdout/`: the retrained model, saved as a separate *variant* so it never replaces the normal model.

**For the paper:** a short "Lesion-focused training" subsection with the before/after table. A model that looks more at the lesion with similar or better accuracy (especially on the external set) is the result to report. If accuracy drops, that is an honest finding too.

---

## 4. `calibrate`: can the confidence be trusted?

**In simple words:** a model that says "90 % sure" should be right about 90 % of the time. **Temperature scaling** divides the model's scores by one number T, chosen on the **validation** set, so that confidence matches accuracy. The test set is used only to report the result. **ECE** (expected calibration error) measures the mismatch; 0 is perfect.

**Outputs** (`calibration/`):

| File | What it shows |
|---|---|
| `calibration.csv`, `calibration_report.md` | Per model and for the ensemble: T, ECE and NLL before/after, on validation and test. The ensemble gets its own extra temperature, fitted after averaging. |
| `reliability_diagrams.png` | Confidence vs accuracy before and after (the diagonal is perfect) |
| `confidence_levels.csv` | How often the web app shows High / Medium / Low, and **how accurate the model is at each level**, e.g. "High: 62 % of cases, 96 % correct" |
| `models/best/` | **The calibrated model for the web app.** It includes per-model temperatures and the `ensemble_temperature`. Copy it to `clinical_system/models/best/`. The web app's High/Medium/Low levels then use calibrated probabilities. |

**For the paper:** an ECE table, the reliability diagram, and the confidence-level table. The last one shows that "High confidence" means something.

---

## 5. `external`: does it work on images from outside the training data?

**In simple words:** all models and the ensemble are tested on the **ISIC 2018 Task 3 test set**. It was never used for training, model selection or calibration, and any image that duplicates our training data was removed.

**Note: it has no SCC images.** Its labels follow HAM10000, where "akiec" covers actinic keratosis and Bowen's disease (SCC in situ). The report therefore:
- averages only over the 7 classes present;
- also gives `accuracy_ak_scc_merged`, which counts an SCC prediction for an AK image as correct;
- gives `predicted_scc_share`, how often the model chose SCC.

**Outputs** (`external/`): `external_results.csv` (accuracy, balanced accuracy, precision, recall, specificity, F1, AUC, melanoma sensitivity, malignant-vs-benign AUC per model and the ensemble, including the naive-split and lesion-focus variants if attached), `external_per_class.csv`, `external_confusion_*.png`, `external_report.md`.

**For the paper:** an "External validation" table next to the internal test results. A smaller drop from internal to external is evidence of generalisation.

---

## Quick check before Kaggle

Every stage was run on tiny fake data on a CPU (`tests/test_experiment_stages.py`, `tests/test_kaggle_stages.py`):
- the planted look-alike pair is kept, while planted real duplicates and conflicts are removed;
- the naive manifest contains the duplicates;
- every output folder stays below 500 files.

These tests show that the pipeline works. They say nothing about the real results, which only the Kaggle runs will give.
