# SkinLens — Capstone Presentation Notes & Viva Preparation

---

## 1. Project Introduction & Problem Statement

- **Project Title**: SkinLens — Multimodal Deep Learning Web Platform for Dermoscopic Skin Lesion Classification
- **Domain**: Medical Computer Vision, Clinical Decision Support Systems, Explainable AI (XAI)
- **Problem Statement**:
  - Skin cancer is among the most prevalent human malignancies globally, with cutaneous melanoma accounting for the vast majority of skin cancer-related fatalities.
  - Dermatologists achieve an average diagnostic accuracy of 65%–80% using clinical examination alone, which increases with dermoscopy.
  - However, differentiating between benign nevi and early-stage melanoma or atypical intraepithelial lesions remains challenging due to overlapping visual features.
  - Automated diagnosis using computer vision often relies strictly on images while ignoring critical patient covariates such as age, sex, and anatomical localization.

---

## 2. Research Motivation & Core Contributions

1. **Multimodal Fusion**: Combines dual vision architectures (Convolutional + Hierarchical Transformer) with structured clinical patient metadata to emulate dermatological diagnostic practice.
2. **Confidence Calibration**: Implements temperature scaling ($T = 1.064053$) to adjust overconfident neural network predictions, reducing Expected Calibration Error (ECE) from 15.87% to 14.11%.
3. **Visual Interpretability**: Employs Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the convolutional feature head (`model.effnet.conv_head`) to provide spatial insight into model attention.
4. **Independent Batch Processing**: Architected for multi-lesion screening where every image maintains its own independent clinical metadata, predictions, and explanations with fault isolation.

---

## 3. Dataset & Preprocessing

- **Dataset**: HAM10000 ("Human Against Machine with 10,000 training images"), curated by ViDIR Group (Medical University of Vienna) and Cliff Rosendahl (Queensland, Australia).
- **Images**: 10,015 dermoscopic image patches across 7 classes.
- **Ground Truth Verification**: Histopathology (53.3%), expert consensus follow-up (33.2%), in-vivo confocal microscopy.
- **Input Dimensions**: $384 \times 384$ pixels (matches Swin Transformer V2 window patch configuration).
- **Normalization**: ImageNet standards (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
- **Metadata Encoding (21-dimensional vector)**:
  - Scaled continuous age: $\text{clip}(\text{age}, 0, 90) / 90.0$ (defaults to $0.5$ if missing).
  - Age missing indicator flag: $1.0$ if missing, $0.0$ if present.
  - Sex: 3-way one-hot vector (`female: 2`, `male: 3`, `unknown: 4`).
  - Anatomical Site: 16-way one-hot vector (`abdomen`, `back`, `chest`, `ear`, `face`, `foot`, `genital`, `hand`, `lower extremity`, `neck`, `scalp`, `trunk`, `upper extremity`, `acral`, `oral/genital`, `unknown`).

---

## 4. Technical Architecture Specifications

```
                       ┌────────────────────────────┐
                       │     Input (3, 384, 384)    │
                       └──────────────┬─────────────┘
                                      │
            ┌─────────────────────────┴────────────────────────┐
            ▼                                                  ▼
┌───────────────────────┐                          ┌───────────────────────┐
│   EfficientNetV2-S    │                          │  Swin Transformer V2  │
│ tf_efficientnetv2_s   │                          │ swinv2_base_w12to24   │
│ Feature Dim: 1,280    │                          │ Feature Dim: 1,024    │
└───────────┬───────────┘                          └───────────┬───────────┘
            │                                                  │
            │           ┌───────────────────────────┐          │
            │           │  Metadata Tensor (1, 21)  │          │
            │           └─────────────┬─────────────┘          │
            │                         ▼                        │
            │           ┌───────────────────────────┐          │
            │           │       Metadata MLP        │          │
            │           │ Linear(21, 64) -> LN ->   │          │
            │           │ GELU -> Dropout(0.25) ->  │          │
            │           │ Linear(64, 32) -> GELU    │          │
            │           │ Feature Dim: 32           │          │
            │           └─────────────┬─────────────┘          │
            │                         │                        │
            ▼                         ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             Concatenation: [1280 + 1024 + 32] = 2,336 features           │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              Classifier Head                             │
│       Linear(2336, 512) -> LayerNorm(512) -> GELU -> Dropout(0.4)        │
│                               Linear(512, 7)                             │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
                        Output Logits (7 classes)
```

- **Auxiliary Training Head**: `Linear(32, 7)` directly on the metadata representation to ensure the tabular branch learned discriminative latent features during training.
- **Total Model Parameters**: $108,276,310$ ($108.28\text{M}$ weights).

---

## 5. Experimental Results (Held-Out Test Set, $N = 2,014$)

| Evaluation Metric | Value | Significance |
| :--- | :---: | :--- |
| **Overall Test Accuracy** | **85.45%** | High baseline across multi-class dermoscopy |
| **Macro-Averaged F1** | **76.03%** | Unweighted mean accounting for rare classes |
| **Macro-Precision** | **78.49%** | Low false positive rate |
| **Macro-Recall** | **76.47%** | High sensitivity across classes |
| **Raw ECE** | **0.1587** | Overconfidence gap before calibration |
| **Calibrated ECE** | **0.1411** | Improved probability calibration ($T = 1.064053$) |

### Per-Class Test Performance

| Class | Full Category Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **akiec** | Actinic keratoses / intraepithelial carcinoma | 0.71 | 0.81 | 0.76 | 68 |
| **bcc** | Basal cell carcinoma | 0.80 | 0.86 | 0.83 | 100 |
| **bkl** | Benign keratosis-like lesions | 0.76 | 0.81 | 0.78 | 205 |
| **df** | Dermatofibroma | 0.82 | 0.41 | 0.55 | 22 |
| **mel** | Melanoma | 0.54 | 0.79 | 0.64 | 222 |
| **nv** | Melanocytic nevi | 0.97 | 0.88 | 0.92 | 1363 |
| **vasc** | Vascular lesions | 0.90 | 0.79 | 0.84 | 34 |

---

## 6. Antagonistic Guide Questions & Accurate Technical Answers

### Q1: Why did you combine EfficientNetV2 and Swin Transformer instead of using just one backbone?
> **Answer**: EfficientNetV2 and Swin Transformer operate on complementary inductive biases. EfficientNetV2 employs depthwise separable convolutions with strong local translation equivariance, making it adept at capturing micro-textures like pigment networks, blue-white veils, and dots. Swin Transformer V2 uses hierarchical shifted windows with self-attention, capturing global structural context and asymmetry across the entire lesion without quadratic computational cost. Combining them yields a feature representation ($1,280 + 1,024 = 2,304\text{D}$) that captures both local texture and global lesion symmetry.

### Q2: What is the purpose of temperature scaling? Does it alter model accuracy?
> **Answer**: Temperature scaling does **not** alter classification accuracy or ranking because it is a strictly monotonic transformation ($T > 0$). Its purpose is **probability calibration**. Modern deep neural networks frequently produce overconfident probabilities (e.g., predicting 99% confidence when accuracy is only 80%). By scaling the pre-softmax logits by $z / T$, where $T = 1.064053$, the probability distribution is softened to accurately match true empirical likelihood, reducing Expected Calibration Error (ECE) from $15.87\%$ to $14.11\%$.

### Q3: Why does your Grad-CAM map target `model.effnet.conv_head` instead of the Swin branch or classifier?
> **Answer**: Grad-CAM requires 2D spatial feature activation maps with preserve-able spatial coordinates ($C \times H \times W$). In our hybrid architecture, `model.effnet.conv_head` outputs a spatial feature map of shape $1,280 \times 12 \times 12$ prior to global pooling, making it suitable for computing spatial gradient weights. Swin Transformer features pass through windowed self-attention, requiring attention rollout or specialized transformer explainers. We explicitly communicate this limitation in the interface: Grad-CAM explains the convolutional branch only.

### Q4: How does your system handle missing patient metadata?
> **Answer**: Clinical metadata is frequently incomplete in real-world triage. Rather than imputing arbitrary values or failing, we engineered the input vector during training with an explicit **age-missing indicator flag** (index 1) and an **unknown** category for sex (index 4) and anatomical site (index 20). If a patient's age is absent, index 0 is assigned $0.5$ (median normalized age) and index 1 is set to $1.0$. The MLP learned during training to down-weight the metadata contribution when missing flags are active, allowing the visual backbones to drive classification.

### Q5: How do you prevent memory exhaustion during multi-image batch uploads?
> **Answer**: The backend implements sequential batch execution via `BatchProcessor`. The 108.28M parameter model is loaded once as an in-memory singleton. For each image in the batch, temporary tensors are processed, Grad-CAM hooks are immediately unregistered in `finally` blocks, and `gc.collect()` is invoked after each item. If any individual image is corrupted or fails, exception boundaries isolate that failure to that specific item, allowing the remainder of the batch to complete successfully.

### Q6: Can this application be deployed for direct clinical diagnosis?
> **Answer**: No. SkinLens is strictly an educational research prototype. Clinical deployment requires rigorous prospective multi-center clinical trials, evaluation across diverse skin types (Fitzpatrick types IV–VI), integration with electronic health records (EHR), and formal regulatory clearance (such as FDA 510(k) or CE mark as Software as a Medical Device).
