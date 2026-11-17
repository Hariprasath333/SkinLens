# SkinLens — Demonstration Workflow Guide

This guide outlines a structured, professional demonstration workflow for presenting the **SkinLens** capstone project to project guides, faculty evaluators, and academic reviewers.

---

## 1. Pre-Demonstration Setup

### Step 1: Start the FastAPI Backend
Open PowerShell Terminal 1:
```powershell
cd C:\Users\harip\Desktop\SkinClass\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
*Expected terminal output:*
```text
INFO:     Started server process
INFO:     Initializing SkinLens Inference Engine & Preloading Model...
INFO:     Model successfully loaded on compute device: cpu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Start the React Frontend
Open PowerShell Terminal 2:
```powershell
cd C:\Users\harip\Desktop\SkinClass\frontend
npm run dev
```
*Expected terminal output:*
```text
  VITE v5.4.11  ready in 320 ms
  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 3: Open the Application
Open Google Chrome or any modern browser and navigate to:
`http://localhost:5173`

---

## 2. Recommended 6-Stage Demonstration Sequence

### Stage 1: Landing Page & Problem Motivation (~2 Minutes)
1. **Show Navbar Health Pill**:
   - Point out the green **"Model Ready (CPU)"** indicator in the top right.
   - Explain that the trained hybrid model checkpoint (108.28M parameters) was verified and loaded once at backend startup.
2. **Present the Problem Statement**:
   - Highlight the challenge: Dermoscopic skin lesion classification is challenging due to visual similarity between benign nevi and malignant melanoma.
   - Explain the solution: A multimodal deep learning architecture that fuses high-resolution vision backbones (384×384) with tabular patient clinical data.
3. **Review the Seven Supported Classes**:
   - Scroll down to the *Seven Supported Categories* section.
   - Show the pathological distinction between Benign (NV, BKL, DF, VASC), Malignant (MEL, BCC), and Precancerous (AKIEC) conditions.
4. **Highlight the Medical & Privacy Notice**:
   - Emphasize that the system is an educational and research prototype.
   - Highlight that images are processed strictly in ephemeral RAM without permanent storage.

---

### Stage 2: Single-Image Analysis & Independent Metadata (~3 Minutes)
1. Click **"Start Analysis"** in the hero section or the **"Analyze"** link in the navbar.
2. Drag and drop or browse to select:
   `references/test_images/sample.jpg`
3. Point out the **ImageQueue** showing thumbnail `#1` in `Ready` status.
4. In the **Clinical Metadata Editor**:
   - Input **Age**: `45`
   - Select **Sex**: `Male`
   - Select **Anatomical Site**: `upper extremity`
5. Click **"Analyze Image"**.
6. Show the real-time processing indicator. The app seamlessly transitions to the **Results Page**.
7. Walk through the results:
   - **Top Card**: Shows predicted category `Melanocytic Nevi (NV)` with green `Benign` badge.
   - **Confidence Comparison**: Point out **Calibrated Confidence (79.2%)** alongside **Raw Model Confidence (82.3%)**.
   - Explain to the guide: *"The model applies temperature scaling ($T = 1.064053$) fitted on validation data to prevent overconfident misclassifications."*
   - **Horizontal Probability Distribution**: All 7 classes sum to 100%, with `nv` leading followed by `df` and `vasc`.

---

### Stage 3: Grad-CAM Visual Interpretability (~2 Minutes)
1. On the same Results view, navigate to the **Visual Interpretability (Grad-CAM)** panel.
2. Select **"3-Panel Side-by-Side"**:
   - **Panel 1 (Original)**: Shows the 384×384 RGB lesion with central pigment network.
   - **Panel 2 (Heatmap)**: Shows JET colormap activation.
   - **Panel 3 (Overlay)**: Shows $0.6 \times \text{Original} + 0.4 \times \text{Heatmap}$.
3. Explain the scientific nuance to the guide:
   - *"Grad-CAM hooks into `model.effnet.conv_head` to compute the gradients of the predicted class score with respect to the 1,280 feature maps."*
   - *"Crucially, we make clear that this explains the convolutional branch only, not the Swin Transformer or metadata MLP."*

---

### Stage 4: Multi-Image Batch Upload & Fault Isolation (~3 Minutes)
1. Click **"New Upload"** or return to `/analyze`.
2. Select `sample.jpg` multiple times (simulating multi-lesion screening).
3. **Demonstrate Independent Metadata**:
   - Case 1: Age `45`, Male, `upper extremity`
   - Case 2: Age `72`, Female, `face`
   - Case 3: Leave all metadata empty (tests missing-value fallback)
4. **Demonstrate "Apply to All" Convenience Tool**:
   - Click "Apply this metadata to all" on Case 2.
   - Show how all cases update, and then show that you can still individually modify Case 1.
5. Click **"Analyze Images"**.
6. When results load, show the **Multi-Case Switcher Tab Bar**:
   - Click between Case #1, Case #2, and Case #3.
   - Show that each case retains its independent probability distribution and Grad-CAM map.
   - Highlight: *"No probabilities are averaged across cases; each is an independent clinical evaluation."*

---

### Stage 5: Ephemeral Session & Privacy Demonstration (~1 Minute)
1. Click **"Clear Session"**.
2. Show that all browser state and object URLs are immediately revoked and the queue resets to zero.
3. Confirm that no database records, cookies, or tracking logs exist.

---

### Stage 6: Technical Methodology Walkthrough (~2 Minutes)
1. Click **"Methodology"** in the top navigation bar.
2. Review the technical diagram with the guide:
   - **Input**: 384×384 RGB
   - **Vision Branch 1**: `tf_efficientnetv2_s.in21k_ft_in1k` $\rightarrow$ 1,280 features
   - **Vision Branch 2**: `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` $\rightarrow$ 1,024 features
   - **Metadata MLP**: 21 input features $\rightarrow$ 64 $\rightarrow$ 32 features
   - **Fusion**: $1280 + 1024 + 32 = 2,336$ dimensions
   - **Classification Head**: $2,336 \rightarrow 512 \rightarrow 7$ classes
3. Point out the verified held-out evaluation metrics:
   - Test Accuracy: **85.45%** ($N = 2,014$)
   - Macro-F1: **76.03%**
   - Calibrated ECE: **0.1411** (improved from 0.1587 raw)
4. Show the documented confusion matrix and precision/recall table.

---

## 3. Demonstration Checklist

- [ ] Backend running and healthy at `http://127.0.0.1:8000/api/health`
- [ ] Frontend running at `http://localhost:5173`
- [ ] `sample.jpg` ready in `references/test_images/sample.jpg`
- [ ] Swagger docs accessible at `http://127.0.0.1:8000/docs`
- [ ] Single prediction tested
- [ ] Batch processing tested
- [ ] Grad-CAM overlay inspected
- [ ] Clear session verified
