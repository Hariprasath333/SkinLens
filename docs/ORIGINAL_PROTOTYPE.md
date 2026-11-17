# SkinLens: Multimodal Skin Lesion Classification Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg?style=flat&logo=react)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6+-3178C6.svg?style=flat&logo=typescript)](https://www.typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC.svg?style=flat&logo=tailwind-css)](https://tailwindcss.com)

**SkinLens** is a full-stack, research-grade web application for automated dermoscopic skin lesion classification. Built as an engineering capstone project, it pairs deep convolutional and vision transformer architectures with tabular clinical patient metadata, calibrated confidence estimation, and layer-specific gradient-weighted class activation mapping (Grad-CAM).

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Architecture & Methodology](#3-architecture--methodology)
  - [Verified Model Architecture](#verified-model-architecture)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Metadata Feature Engineering](#metadata-feature-engineering)
  - [Temperature Calibration](#temperature-calibration)
  - [Grad-CAM Interpretability](#grad-cam-interpretability)
- [4. The HAM10000 Benchmark Dataset](#4-the-ham10000-benchmark-dataset)
- [5. Documented Research Evaluation Results](#5-documented-research-evaluation-results)
- [6. Technology Stack](#6-technology-stack)
- [7. Project Folder Structure](#7-project-folder-structure)
- [8. Installation & Setup](#8-installation--setup)
- [9. Running the Application](#9-running-the-application)
- [10. API Specification](#10-api-specification)
- [11. Testing & Verification](#11-testing--verification)
- [12. Privacy, Ethics & Medical Disclaimer](#12-privacy-ethics--medical-disclaimer)

---

## 1. Project Overview

Cutaneous malignancies, particularly malignant melanoma and invasive carcinomas, demand early and accurate diagnosis to maximize clinical intervention efficacy. SkinLens addresses the diagnostic challenge by providing an end-to-end multimodal classification pipeline:
- Ingests dermoscopic images ($384 \times 384$ RGB) and extracts visual features through dual vision backbones: **EfficientNetV2-S** and **Swin Transformer V2**.
- Fuses vision features with clinical patient metadata (age, sex, anatomical location) mapped through a dedicated Multi-Layer Perceptron (MLP).
- Softens raw logit outputs via validation-fitted **Temperature Scaling** ($T = 1.064053$) to generate empirically calibrated class probabilities.
- Highlights spatial diagnostic regions influencing the convolutional feature head via **Grad-CAM**.
- Provides a clean, healthcare-focused user experience adhering to minimal design standards (Apple Health / Linear inspired).

---

## 2. Key Features

- **Multi-Image Batch Upload**: Upload single or multiple dermoscopic images simultaneously with drag-and-drop support.
- **Independent Clinical Metadata**: Modify patient age, sex, and anatomical localization independently per case. Includes an optional "Apply to all" convenience action.
- **Strict Error Isolation**: Corrupted or unreadable image uploads fail independently without terminating or discarding successful items in the batch.
- **Confidence Calibration**: Displays both raw model confidence and calibrated confidence to safeguard against deep learning overconfidence.
- **Grad-CAM Visual Explanations**: Side-by-side comparison of original dermoscopy, activation heatmap, and 60/40 composite overlay.
- **Ephemeral Privacy Guarantee**: No user accounts, no databases, no permanent image saving. All data remains exclusively in browser memory and temporary process RAM.

---

## 3. Architecture & Methodology

### Verified Model Architecture

The `SkinLensHybrid` model integrates complementary visual inductive biases with structured tabular embeddings:

```
                      ┌──────────────────────────────────────────────┐
                      │              Input Image (384x384)           │
                      └──────────────┬────────────────┬──────────────┘
                                     │                │
                                     ▼                ▼
┌─────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
│   timm: tf_efficientnetv2_s.in21k_ft_in1k│  │ timm: swinv2_base_window12to24_192to384.ms_in22k_ft_in1k│
│              (num_classes=0)            │  │                    (num_classes=0)                     │
│         Feature Dimension: 1,280        │  │                Feature Dimension: 1,024                │
└────────────────────┬────────────────────┘  └───────────────────────────┬────────────────────────────┘
                     │                                                   │
                     │                 ┌─────────────────────────────────┼────────────────────────────┐
                     │                 │   Input Metadata (dim = 21)     │                            │
                     │                 └────────────────┬────────────────┘                            │
                     │                                  ▼                                             │
                     │                 ┌─────────────────────────────────┐                            │
                     │                 │          Metadata MLP           │                            │
                     │                 │ Linear(21, 64) -> LayerNorm(64) │                            │
                     │                 │ -> GELU() -> Dropout(0.25)      │                            │
                     │                 │ -> Linear(64, 32) -> GELU()     │                            │
                     │                 │      Feature Dimension: 32      │                            │
                     │                 └────────────────┬────────────────┘                            │
                     │                                  │                                             │
                     ▼                                  ▼                                             ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           Fused Representation: Concat [1280 + 1024 + 32] = 2,336                    │
  └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                           Classifier                                                │
  │                     Linear(2336, 512) -> LayerNorm(512) -> GELU() -> Dropout(0.4)                  │
  │                                            Linear(512, 7)                                           │
  └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                     ▼
                                        Output Logits: [Batch, 7]
```

- **Auxiliary Metadata Loss**: During training, an auxiliary head (`meta_aux_head = nn.Linear(32, 7)`) enforced discriminative representation learning within the metadata MLP.
- **Total Model Parameters**: $108,276,310$ ($108.28\text{M}$ weights).

### Preprocessing Pipeline
- **Input Dimensions**: Resized to $384 \times 384$ pixels via linear interpolation.
- **Color Format**: 3-channel RGB.
- **Normalization (ImageNet Standards)**:
  - Mean: `(0.485, 0.456, 0.406)`
  - Standard Deviation: `(0.229, 0.224, 0.225)`
- **Tensor Format**: `(Batch, 3, 384, 384)` float32.

### Metadata Feature Engineering
The clinical vector encodes 21 features into a fixed-length float32 array:
- **Index 0 (Continuous Age)**: $\text{clip}(\text{age}, 0, 90) / 90.0$ ($0.5$ if missing).
- **Index 1 (Age Missing Indicator)**: $1.0$ if age is absent/NaN, else $0.0$.
- **Indices 2–4 (Sex One-Hot)**: `female` (idx 2), `male` (idx 3), `unknown` (idx 4).
- **Indices 5–20 (Anatomical Localization One-Hot, 16 categories)**:
  `['abdomen', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity', 'acral', 'oral/genital', 'unknown']`. Missing or unlisted sites map to index 20 (`unknown`).

### Temperature Calibration
Uncalibrated neural networks frequently output overconfident probabilities. We apply **Global Temperature Scaling**:
$$P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
Where $T = 1.064053$, optimized on validation set negative log-likelihood (NLL). Calibration reduces the Expected Calibration Error (ECE) from $0.1587$ to $0.1411$ on held-out test data.

### Grad-CAM Interpretability
Grad-CAM computes gradients targeting `model.effnet.conv_head` ($1,280 \times 12 \times 12$ activation maps):
$$\alpha_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}, \quad L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
- **Scope Note**: Grad-CAM visualizes features influencing the **EfficientNetV2-S convolutional branch only**. It does not visualize self-attention maps in Swin Transformer V2 or tabular weights in the metadata branch.

---

## 4. The HAM10000 Benchmark Dataset

The **HAM10000** ("Human Against Machine with 10,000 training images") dataset comprises 10,015 dermoscopic lesion patches:

| Code | Canonical Index | Formal Clinical Name | Pathological Nature |
| :---: | :---: | :--- | :--- |
| **AKIEC** | 0 | Actinic Keratoses / Intraepithelial Carcinoma | Precancerous / In-situ |
| **BCC** | 1 | Basal Cell Carcinoma | Malignant (Non-melanoma) |
| **BKL** | 2 | Benign Keratosis-like Lesions | Benign |
| **DF** | 3 | Dermatofibroma | Benign |
| **MEL** | 4 | Melanoma | Malignant |
| **NV** | 5 | Melanocytic Nevi | Benign |
| **VASC** | 6 | Vascular Lesions | Benign |

---

## 5. Documented Research Evaluation Results

Evaluated on the held-out test split ($N = 2,014$ samples) from the research training session:

| Metric | Score |
| :--- | :---: |
| **Test Accuracy** | **85.45%** |
| **Macro-Averaged F1** | **76.03%** |
| **Macro-Precision** | **78.49%** |
| **Macro-Recall** | **76.47%** |
| **Raw Expected Calibration Error (ECE)** | **0.1587** |
| **Calibrated Expected Calibration Error (ECE)** | **0.1411** |

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **akiec** | 0.71 | 0.81 | 0.76 | 68 |
| **bcc** | 0.80 | 0.86 | 0.83 | 100 |
| **bkl** | 0.76 | 0.81 | 0.78 | 205 |
| **df** | 0.82 | 0.41 | 0.55 | 22 |
| **mel** | 0.54 | 0.79 | 0.64 | 222 |
| **nv** | 0.97 | 0.88 | 0.92 | 1,363 |
| **vasc** | 0.90 | 0.79 | 0.84 | 34 |

---

## 6. Technology Stack

- **Backend Framework**: Python 3.12, FastAPI, Uvicorn, Pydantic v2
- **Deep Learning & Computer Vision**: PyTorch, torchvision, timm, Albumentations, OpenCV, Pillow, SciPy
- **Frontend Framework**: React 18, Vite, TypeScript, Tailwind CSS, React Router 6, Lucide Icons

---

## 7. Project Folder Structure

```
SkinClass/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application, endpoints & CORS
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── preprocessing.py     # Image decoding, 384x384 resize, ImageNet normalization
│   │   ├── metadata.py          # 21-dim metadata vectorizer
│   │   ├── inference.py         # Singleton model manager & temperature scaling
│   │   ├── gradcam.py           # Hook-based Grad-CAM engine for effnet.conv_head
│   │   └── batch.py             # Fault-isolated sequential batch processor
│   ├── model/
│   │   ├── __init__.py
│   │   └── architecture.py      # SkinLensHybrid PyTorch neural network definition
│   ├── models/
│   │   ├── best_skinlens_hybrid.pt # Trained PyTorch checkpoint (108.28M weights, 414MB)
│   │   └── temperature.json     # Calibrated temperature scalar (1.064053)
│   ├── outputs/
│   │   └── gradcam_visualization.png # Sample 3-panel Grad-CAM visualization
│   ├── requirements.txt         # Pinned backend dependencies
│   ├── verify_checkpoint.py     # Standalone checkpoint verification script
│   ├── verify_inference.py      # Single-image inference test
│   ├── verify_gradcam.py        # Grad-CAM engine test
│   ├── verify_batch.py          # Batch processing & fault isolation test
│   └── verify_api.py            # Comprehensive 10-endpoint automated API test suite
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/          # CategoryBadge, ProbabilityBar
│   │   │   ├── layout/          # Navbar, Footer, Layout
│   │   │   ├── upload/          # Dropzone, ImageQueue, MetadataEditor
│   │   │   └── results/         # CaseSummaryCard, GradCamViewer, ProbabilityDistribution
│   │   ├── context/
│   │   │   └── AnalysisContext.tsx # Ephemeral in-memory session manager
│   │   ├── pages/
│   │   │   ├── HomePage.tsx     # Landing page & overview
│   │   │   ├── AnalyzePage.tsx  # Multi-image upload & metadata editor
│   │   │   ├── ResultsPage.tsx  # Multi-case results & Grad-CAM viewer
│   │   │   └── MethodologyPage.tsx # Architecture diagram, calibration, metrics
│   │   ├── services/
│   │   │   └── api.ts           # Centralized HTTP client
│   │   ├── types/
│   │   │   └── index.ts         # TypeScript definitions & class data
│   │   ├── App.tsx              # Router setup
│   │   ├── main.tsx             # Entrypoint
│   │   └── index.css            # Tailwind directives
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   └── .env
│
├── docs/
│   ├── DEMO_WORKFLOW.md         # Guide-facing demonstration script
│   └── CAPSTONE_PRESENTATION_NOTES.md # Viva preparation & technical Q&A
│
├── references/
│   ├── test_images/
│   │   └── sample.jpg           # Real HAM10000 test image (Ground Truth: nv)
│   └── skinclass.ipynb          # Original Kaggle training & evaluation notebook
│
└── README.md
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.12+ (64-bit)
- Node.js v18+ and npm v9+

### Backend Environment Setup
```powershell
# Navigate to backend directory
cd C:\Users\harip\Desktop\SkinClass\backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate

# Install backend dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Frontend Environment Setup
```powershell
# Navigate to frontend directory
cd C:\Users\harip\Desktop\SkinClass\frontend

# Install npm packages
npm install
```

---

## 9. Running the Application

### 1. Launch FastAPI Backend
In Terminal 1:
```powershell
cd C:\Users\harip\Desktop\SkinClass\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Base: `http://127.0.0.1:8000`
- Swagger Documentation: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

### 2. Launch React Frontend
In Terminal 2:
```powershell
cd C:\Users\harip\Desktop\SkinClass\frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 10. API Specification

| Method | Endpoint | Description | Request Format |
| :---: | :--- | :--- | :--- |
| `GET` | `/api/health` | Service status, device, model status, temperature | None |
| `GET` | `/api/classes` | Canonical 7 classes in alphabetical order | None |
| `POST`| `/api/predict` | Single image analysis with optional metadata | `multipart/form-data` (`image`, `age`, `sex`, `localization`, `include_gradcam`) |
| `POST`| `/api/predict/batch` | Batch analysis with independent metadata | `multipart/form-data` (`images`, `metadata_json`, `include_gradcam`) |
| `POST`| `/api/explain` | Dedicated Grad-CAM visual explanation | `multipart/form-data` (`image`, `age`, `sex`, `localization`, `target_class`) |

---

## 11. Testing & Verification

Each phase of development includes an isolated test suite:

```powershell
# 1. Verify model checkpoint loading with strict=True
& backend\.venv\Scripts\python.exe backend\verify_checkpoint.py

# 2. Verify single-image inference against real HAM10000 sample
& backend\.venv\Scripts\python.exe backend\verify_inference.py

# 3. Verify Grad-CAM hook execution and overlay generation
& backend\.venv\Scripts\python.exe backend\verify_gradcam.py

# 4. Verify batch inference, metadata independence, and corrupt image isolation
& backend\.venv\Scripts\python.exe backend\verify_batch.py

# 5. Verify FastAPI HTTP endpoints, validations, and response schemas
& backend\.venv\Scripts\python.exe backend\verify_api.py

# 6. Verify frontend production build and TypeScript contracts
cd frontend; npm run build
```

---

## 12. Privacy, Ethics & Medical Disclaimer

### Privacy Guarantee
SkinLens implements an **ephemeral in-memory architecture**:
- Uploaded dermoscopic images and metadata are never written to disk or recorded in persistent databases.
- All session variables and object URLs are cleared immediately upon selecting **Clear Session**.
- No external third-party AI APIs are used. All computation runs locally.

### Scientific Limitations
- **Demographic Bias**: The HAM10000 dataset is predominantly collected from individuals with light skin pigmentation (Fitzpatrick skin phototypes I–III). Performance on darker skin types (types IV–VI) is not clinically established.
- **Class Imbalance**: Over 67% of the HAM10000 dataset represents benign nevi, resulting in lower sensitivity for rare lesions like dermatofibroma.
- **Explainability Scope**: Grad-CAM visualizes features of the convolutional branch only and does not reflect attention in the vision transformer or tabular branch.

### Medical Disclaimer
> **IMPORTANT NOTICE:** SkinLens is an academic research prototype and educational demonstration developed for an engineering capstone project. It is **not** a certified medical diagnostic device and has not received regulatory clearance (such as FDA 510(k), CE mark, or CDSCO). SkinLens predictions must **never** be used to self-diagnose, replace, or delay formal clinical examination, dermatoscopy, or biopsy by a qualified dermatologist or licensed physician.
