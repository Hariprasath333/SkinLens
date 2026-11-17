# Public datasets for skin lesion classification

These are the datasets the new pipeline (`clinical_system/research/`) can use, with what to cite in the paper.

> **Check before you publish.** I wrote the licences and image counts from the official pages as I know them. Open each link once and confirm them, because datasets are sometimes re-licensed or re-versioned. All of the datasets below allow **non-commercial research use**. None of them allows commercial use of the images.

## Datasets used by default (downloaded by `download_datasets.py`)

| # | Dataset | What it contains | Images | Classes | Licence | Source | Cite |
|---|---|---|---|---|---|---|---|
| 1 | **ISIC 2019 Challenge – Training** | Dermoscopic images from **HAM10000** (Vienna, Australia), **BCN20000** (Barcelona) and **MSK** (New York) | 25,331 (9.8 GB zip) | 8 + UNK: MEL, NV, BCC, AK, BKL, DF, VASC, SCC | CC BY-NC 4.0 | https://challenge.isic-archive.com/data/#2019 (files on `isic-archive.s3.amazonaws.com/challenges/2019/`, checked Nov 2026) | Tschandl 2018 (HAM10000); Codella 2018 (MSK/ISIC 2017); Combalia 2019 (BCN20000, arXiv:1908.02288) |
| 2 | **HAM10000 lesion groupings** (`ISIC2018_Task3_Training_LesionGroupings.csv`) | The `lesion_id` for each HAM10000 image, used to keep different photos of the **same lesion** in the same split. Images are not downloaded twice because ISIC 2019 already contains them. | 10,015 rows (metadata only) | – | CC BY-NC 4.0 | https://challenge.isic-archive.com/data/#2018 (same data as Harvard Dataverse https://doi.org/10.7910/DVN/DBW86T) | Tschandl, Rosendahl, Kittler, *Scientific Data* 5:180161, 2018 |
| 3 | **ISIC 2018 Task 3 – Test set** (external test only) | Held-out dermoscopic test images with released ground truth | 1,512 | 7 (HAM10000 classes) | CC BY-NC 4.0 | https://challenge.isic-archive.com/data/#2018 | Codella et al., arXiv:1902.03368, 2019; Tschandl 2018 |

## Optional datasets (manual download; the merge script supports them)

| # | Dataset | What it contains | Images | Licence / access | Source | How it is used |
|---|---|---|---|---|---|---|
| 4 | **PAD-UFES-20** | **Smartphone (clinical, not dermoscopic)** photos from Brazil, with `patient_id` | 2,298 (1,373 patients) | CC BY 4.0 | https://doi.org/10.17632/zr7vgbcyr2.1 (Mendeley Data) | External test by default, because it comes from a different camera type. ACK→AK, NEV→NV, SEK→BKL; BCC, MEL, SCC unchanged. Cite Pacheco et al., *Data in Brief* 32:106221, 2020. |
| 5 | **PH2** | Dermoscopic images from Portugal | 200 (80 common nevi, 80 atypical nevi, 40 melanoma) | Free for research; you agree to terms on the website | https://www.fc.up.pt/addi/ph2%20database.html | External melanoma-vs-nevus test. Cite Mendonça et al., IEEE EMBC 2013. |
| 6 | **Derm7pt** | Dermoscopic + clinical images with 7-point checklist labels | 1,011 cases | Registration, non-commercial | https://derm.cs.sfu.ca | External test. Cite Kawahara et al., IEEE JBHI 2019. |

Not included on purpose:
- **ISIC 2020** (33,126 images): almost all images are "benign/unknown", it is only melanoma-vs-rest, and it is 20+ GB. It adds little for an 8-class model.
- **DDI** and **Fitzpatrick17k**: clinical photos for skin-tone fairness studies. These are good for future work, but they use different classes and access rules.

## Why de-duplication matters here

ISIC 2019 **already contains all of HAM10000**, and the ISIC 2018 test set comes from the same clinics. If you merge them naively, the same photo appears twice and can end up in both train and test, which makes the reported accuracy dishonest. `build_dataset.py` removes this problem in three ways:
1. It matches identical ISIC image IDs.
2. It finds exact duplicates by file hash (MD5) and near-duplicates (resized or re-compressed copies) by perceptual hash.
3. It puts every image of the same lesion or patient (`lesion_id` / `patient_id`), plus every near-duplicate pair, into one group, and a group is never split across train / validation / test.

`data/merged/cleaning_report.md` records how many images each step removed. Copy those numbers into the paper's dataset section.

## Unified label set (8 classes)

| Code | Name | Group |
|---|---|---|
| MEL | Melanoma | Malignant |
| BCC | Basal cell carcinoma | Malignant |
| SCC | Squamous cell carcinoma | Malignant |
| AK | Actinic keratosis (pre-cancerous) | Pre-malignant |
| NV | Melanocytic nevus | Benign |
| BKL | Benign keratosis (seborrhoeic keratosis, solar lentigo, lichen-planus-like keratosis) | Benign |
| DF | Dermatofibroma | Benign |
| VASC | Vascular lesion | Benign |

HAM10000 / ISIC 2018 `akiec` maps to **AK**. In HAM10000, `akiec` also includes Bowen's disease (SCC in situ). ISIC 2019 separates SCC, so when the same image has AK in one source and SCC in another, the ISIC 2019 label wins. Images labelled `UNK` in ISIC 2019 are removed.
