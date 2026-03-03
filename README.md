# Keratoconus CXL Prediction Pipeline

A multimodal deep learning pipeline for predicting **corneal cross-linking (CXL) necessity** in keratoconus patients. The system fuses four corneal Pentacam image modalities with clinical numeric features using a hybrid CNN–XGBoost architecture, evaluated through rigorous nested cross-validation.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Image Modalities](#2-image-modalities)
  - [3. CNN Encoder](#3-cnn-encoder)
  - [4. Multimodal Fusion](#4-multimodal-fusion)
  - [5. Training Modes](#5-training-modes)
  - [6. Cross-Validation Strategy](#6-cross-validation-strategy)
  - [7. Class Imbalance Handling](#7-class-imbalance-handling)
- [Interpretability & Explainability](#interpretability--explainability)
- [Publication Figure Generation](#publication-figure-generation)
- [Configuration System](#configuration-system)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

The pipeline addresses the clinical question: *"Given a keratoconus patient's Pentacam corneal maps and clinical measurements, does this patient need CXL treatment?"*

**Input:**
- **4 Pentacam corneal map images** per patient (224×224 RGB)
- **Clinical numeric features** (keratometry, pachymetry, corneal indices, patient demographics)

**Output:**
- Binary CXL prediction probability (0–1)
- Per-fold AUC with 95% confidence intervals
- SHAP feature importance analysis
- Grad-CAM visual explanations

---

## Project Structure

```
algo-pipeline/
├── main.py                      # Entry point — loads config, data, runs CV
├── analyze_results.py           # Post-hoc analysis (embeddings, t-SNE, feature dists)
├── generate_results.py          # Publication-quality figure generation (16 figure types)
├── generate_heatmaps.py         # Batch Grad-CAM & CBAM attention heatmap generation
├── explain_cnn_features.py      # Maps CNN features to modalities + clinical interpretation
├── interpret_cnn_features.py    # Per-feature GradCAM targeting specific CNN activations
├── print_support.py             # Utility to inspect feature selector support indices
│
├── configs/                     # YAML configuration files
│   ├── default_config.yaml      # Default (hybrid finetuned, ResNet18, partial freeze)
│   ├── v5_config.yaml           # ...
│   ├── v6–v14_config.yaml       # Various architecture/training experiments
│   └── v15_config.yaml          # Final: frozen CNN + XGBoost (no fine-tuning)
│
└── src/                         # Core library modules
    ├── __init__.py
    ├── data.py                  # Dataset, transforms, data loading, class balancing
    ├── models.py                # CNN encoders, attention modules, classifiers
    ├── engine.py                # Training loops (end-to-end, hybrid frozen, hybrid finetuned)
    ├── evaluation.py            # Nested CV orchestration, baseline comparison
    ├── utils.py                 # Config loading, logging, seed, wandb init
    └── visualization.py         # Grad-CAM, CBAM attention extraction, heatmap plotting
```

---

## Pipeline Architecture

### 1. Data Loading & Preprocessing

**Module:** `src/data.py`

- **Source data:** `multimodal_dataset.csv` containing patient records with clinical features and a binary CXL label (`y`).
- **Feature list:** Loaded from `multimodal_dataset_features.txt` — specifies which numeric columns to use.
- **Filtering:**
  - Patients must have all 4 image modalities available (`has_all_images` flag).
  - Missing numeric features → dropped.
  - Optional **KMax filter** (e.g., `kmax_filter: 55`) removes patients with extremely high keratometry values, excluding trivially obvious cases.

**Image Transforms:**

| Mode       | Operations                                                                                  |
|------------|---------------------------------------------------------------------------------------------|
| Training   | Resize → RandomCrop → HorizontalFlip → Rotation (±10°) → ColorJitter → Affine → GaussianBlur → Normalize |
| Validation | Resize (224×224) → Normalize (ImageNet stats)                                               |

> Augmentations are deliberately conservative to preserve clinical features in corneal maps. RandomErasing is explicitly excluded to avoid destroying diagnostic signal.

### 2. Image Modalities

Each patient has **4 Pentacam-derived corneal maps**:

| Modality            | Clinical Meaning                                       |
|---------------------|--------------------------------------------------------|
| `corneal_thickness` | Pachymetry map — corneal thickness distribution        |
| `curvature_front`   | Anterior curvature (keratometry) — front surface shape |
| `elevation_front`   | Front elevation relative to best-fit sphere            |
| `elevation_back`    | Posterior elevation — early keratoconus biomarker       |

### 3. CNN Encoder

**Module:** `src/models.py`

The pipeline supports multiple CNN encoder architectures:

#### Backbone Registry

| Backbone           | Feature Dim | Grad-CAM Target Layer |
|--------------------|:-----------:|:---------------------:|
| ResNet-18          |     512     |       `layer4`        |
| EfficientNet-B0    |    1280     |     `features.8`      |
| EfficientNet-B4    |    1792     |     `features.8`      |
| MobileNetV3-Small  |     576     |    `features.12`      |

All backbones are initialized with **ImageNet pre-trained weights**.

#### Encoder Variants

1. **`MultiBackboneCNNEncoder`** — **Separate backbone per modality.** Each of the 4 image types gets its own copy of the backbone, allowing modality-specific feature learning. Higher parameter count but more expressive.

2. **`SharedBackboneCNNEncoder`** — **Single shared backbone** processes all modalities, with lightweight **per-branch adapters** (`BranchAdapter`) on top. Dramatically reduces parameters (1 backbone vs 4).

#### `SingleBranchEncoder` Architecture

Each branch follows: **Backbone → (CBAM Attention) → AdaptiveAvgPool → Flatten → (Adapter)**

- **CBAM** (Convolutional Block Attention Module): Channel attention (squeeze-excitation) + Spatial attention applied after the backbone's feature maps.
- **BranchAdapter**: A 2-layer MLP with LayerNorm, GELU activation, dropout, and optional residual connections. Used for domain adaptation when the backbone is frozen.

#### Freezing Strategies

| `freeze_mode` | Behavior                                                                                   |
|---------------|--------------------------------------------------------------------------------------------|
| `all`         | Entire backbone frozen — only attention / adapters / classifier are trainable              |
| `partial`     | Later backbone layers unfrozen (e.g., `layer3`+`layer4` for ResNet, last 3 blocks for EfficientNet) |
| `none`        | Full backbone fine-tuning (risk of overfitting on small datasets)                          |

#### Cross-Modal Attention Fusion

**`CrossModalAttentionFusion`** — Transformer-style fusion of the 4 modality feature vectors:

1. Stack modality features into a sequence → Multi-head self-attention (8 heads)
2. Residual connection + LayerNorm
3. FFN (feature_dim → 2×feature_dim → feature_dim) with GELU + dropout
4. Learned modality importance weights (softmax-normalized)
5. Weighted sum + concatenated projection → final fused representation

When disabled (`use_cross_modal_fusion: false`), features are simply concatenated.

### 4. Multimodal Fusion

**`MultimodalClassifier`** (for end-to-end and hybrid-finetuned modes):

```
CNN Features (e.g., 2048-dim) → Bottleneck (128-dim) → Concat with Numeric Features
→ FC(128) + LayerNorm + ReLU + Dropout
→ FC(64) + LayerNorm + ReLU + Dropout
→ FC(1) → Sigmoid → P(CXL)
```

The **CNN bottleneck** (configurable via `bottleneck_dim`) compresses the high-dimensional CNN representation before concatenation with numeric features, preventing CNN features from dominating.

### 5. Training Modes

The pipeline supports three training strategies, selected via `training_mode` in the config:

#### Mode A: `end_to_end`

**Module:** `engine.py → train_end_to_end()`

Standard neural network training — CNN + numeric features → classifier head → binary prediction. Uses:

- AdamW optimizer with cosine annealing LR schedule
- Focal Loss with class-balanced alpha
- Gradient clipping (max norm = 1.0)
- Early stopping on validation AUC

#### Mode B: `hybrid_cnn_xgboost` ⭐ (Recommended)

**Module:** `engine.py → train_hybrid_cnn_xgboost()`

The final production pipeline. Avoids fine-tuning entirely to prevent overfitting on ~1000 samples:

1. **Feature Extraction** — Frozen CNN backbone extracts feature vectors for all training and test images (single-pass extraction via `extract_all_features()` to maintain label-feature alignment).
2. **Feature Scaling** — `StandardScaler` applied independently to CNN features and numeric features.
3. **Feature Selection** — `SelectKBest` (ANOVA F-statistic) selects the top-*k* most discriminative CNN features (e.g., `n_select_features: 30`).
4. **Feature Combination** — Selected CNN features concatenated with scaled numeric features.
5. **XGBoost Training** — Gradient-boosted tree classifier trained on the combined feature matrix:
   - `scale_pos_weight` for class imbalance
   - Early stopping on validation AUC (20-round patience)
   - Regularization via `subsample`, `colsample_bytree`, and shallow `max_depth`

**Saved artifacts per fold:** CNN encoder weights, XGBoost model (JSON), StandardScalers (joblib), feature selector (joblib).

#### Mode C: `hybrid_finetuned`

**Module:** `engine.py → train_hybrid_finetuned()`

Two-stage approach — fine-tune the CNN first, then hand off to XGBoost:

- **Stage 1 — CNN Fine-tuning:**
  - Wraps encoder in `MultimodalClassifier` for joint image + numeric training
  - Layer-wise learning rates (lower for backbone, higher for classifier head)
  - Learning rate warmup (configurable warmup epochs)
  - Optional Mixup augmentation (with 50% probability)
  - Optional label smoothing
  - Gradient accumulation for effective larger batch sizes
  - Optional SWA (Stochastic Weight Averaging) with validation-gated acceptance
  - Cosine annealing or ReduceLROnPlateau scheduler
  - Early stopping on validation AUC (with patience)

- **Stage 2 — XGBoost:** Same as Mode B, but using the fine-tuned encoder's features.

### 6. Cross-Validation Strategy

**Module:** `src/evaluation.py → nested_cv_multimodal()`

```
For each fold k in {1, ..., K}:
  1. Split by patient ID (ideye) — StratifiedKFold on unique patients
     → No data leakage between train/test (same patient's eyes stay together)
  2. Create DataLoaders (with/without augmentation)
  3. Train multimodal model (Mode A, B, or C)
  4. Train numeric-only XGBoost baseline for comparison
  5. Save models, scalers, selectors to disk
  6. Record out-of-fold (OOF) predictions
  7. Generate Grad-CAM heatmaps (last fold only)
  8. Log ROC and PR curves to Weights & Biases
```

**Key details:**
- **Patient-level splitting** via `ideye` grouping — both eyes of the same patient are always in the same fold.
- **Stratified** on CXL label to maintain class balance across folds.
- Default: **5-fold** cross-validation.
- OOF predictions saved to `oof_predictions.csv` for downstream result generation.
- Final statistics include mean AUC ± std with **95% confidence intervals** (t-distribution).

### 7. Class Imbalance Handling

Multiple complementary strategies:

| Strategy                   | Implementation                                                       |
|----------------------------|----------------------------------------------------------------------|
| **Focal Loss**             | Down-weights well-classified examples (γ=2.0), with class-balanced α |
| **Weighted Random Sampler**| Over-samples minority class during training                          |
| **XGBoost `scale_pos_weight`** | Inverse class frequency weighting                                |
| **SMOTE** (optional)       | Synthetic minority oversampling (available but not used in final config) |

---

## Interpretability & Explainability

### Grad-CAM Heatmaps

**Module:** `src/visualization.py`

The `GradCAM` class computes gradient-weighted class activation maps to visualize which spatial regions of each corneal map influence predictions:

1. Forward pass to compute activations at the target convolutional layer
2. Backward pass to compute gradients w.r.t. the target class
3. Global average pooling of gradients → per-channel weights
4. Weighted combination of activation maps → ReLU → resize to input dimensions

**`generate_heatmaps.py`** produces 3-row visualizations per patient:
- **Row 1:** Raw Pentacam images (denormalized)
- **Row 2:** CBAM spatial attention overlays
- **Row 3:** Grad-CAM heatmap overlays

Includes clinical metadata (age, K values, pachymetry, ISV, IVA, KI) as a text header.

### CNN Feature Interpretation

**`interpret_cnn_features.py`** — Generates per-feature GradCAM heatmaps by wrapping the encoder in a `FeatureTargetFullModel` that targets a specific CNN feature's activation (rather than the overall class score).

**`explain_cnn_features.py`** — Full interpretive pipeline for top SHAP features:
1. Maps each CNN feature index back to its source image modality (based on backbone feature dimension boundaries)
2. Computes spatial statistics (central vs peripheral activation)
3. Finds top-correlating clinical features
4. Generates a clinical name and interpretation string
5. Produces composite and per-feature GradCAM figures (2 CXL + 2 Normal exemplar patients)

### SHAP Analysis

**`generate_results.py → plot_shap_analysis()`** — Aggregates TreeSHAP values across all CV folds to produce:
- Beeswarm plots showing per-sample feature impact
- Bar plots of mean absolute SHAP values
- Identification of top CNN features for downstream interpretation

---

## Publication Figure Generation

**`generate_results.py`** produces 16 publication-quality figures and tables:

| #  | Figure / Table                           | Description                                                        |
|:--:|------------------------------------------|--------------------------------------------------------------------|
| 1  | Data Flow Diagram                        | Programmatic flow chart of data filtering pipeline                 |
| 2  | ML Pipeline Overview                     | Visual architecture diagram of the multimodal pipeline             |
| 3  | ROC-AUC Curves                           | Per-fold + mean ROC curves with CI bands                           |
| 4  | Precision / Recall / F1 vs Threshold     | Metric trade-offs across classification thresholds                 |
| 5  | SHAP Feature Importance                  | Beeswarm + bar plots aggregated across folds                       |
| 6  | Calibration Curve                        | Reliability diagram assessing model calibration                    |
| 7  | Confusion Matrix                         | At optimal threshold + high-precision variant                      |
| 8  | Training Curves                          | Per-fold training loss/AUC parsed from log files                   |
| 9  | Feature Violin Plots                     | Top clinical features split by CXL/Normal                         |
| 10 | Precision-Recall Curve                   | Per-fold + mean PR curves                                         |
| 11 | Summary Statistics Table                 | LaTeX + CSV: AUC, F1, sensitivity, specificity, PPV, NPV          |
| 12 | Table 1: Patient Demographics            | Stratified by CXL status with p-values                            |
| 13 | Decision Curve Analysis (DCA)            | Net clinical benefit vs threshold probability                      |
| 14 | Subgroup Analysis by KMax Severity       | Model performance stratified by disease severity                   |
| 15 | Image Modality Contribution              | XGBoost importance breakdown by corneal map type                   |
| 16 | Exemplar Case Panels                     | TP/TN/FP/FN cases with their 4 corneal maps                       |

All figures are saved as both **PNG (300 DPI)** and **PDF** in the `publication_figures/` output directory.

Styling uses the [`scienceplots`](https://github.com/garrettj403/SciencePlots) library with the `science` + `nature` themes (LaTeX-free).

---

## Configuration System

All hyperparameters are controlled via **YAML config files** in `configs/`. The config is loaded at startup and can be overridden via CLI arguments.

### Key Configuration Parameters

```yaml
# Data
data_csv: '../data/multimodal_dataset.csv'
features_txt: '../data/multimodal_dataset_features.txt'
image_dir: '../cropped_output_v2'
kmax_filter: 55              # Remove easy cases with KMax > 55D

# Architecture
backbone: 'resnet18'          # CNN backbone (resnet18, efficientnet_b0, etc.)
freeze_mode: 'all'            # Backbone freezing: all | partial | none
shared_backbone: true         # Share one backbone across modalities
adapter_mode: 'none'          # Branch adapters: none | per_branch | fusion_only
use_attention: false           # CBAM attention module
use_cross_modal_fusion: false  # Transformer-style cross-modal fusion
bottleneck_dim: 128           # CNN feature bottleneck width
n_select_features: 30         # Top-k CNN features for XGBoost

# Training
training_mode: 'hybrid_cnn_xgboost'  # end_to_end | hybrid_cnn_xgboost | hybrid_finetuned
batch_size: 32
learning_rate: 0.001
weight_decay: 0.03
dropout: 0.5

# XGBoost
xgb_n_estimators: 500
xgb_max_depth: 3
xgb_learning_rate: 0.01

# CV
n_cv_folds: 5
random_state: 42
```

### Configuration Versions

The `configs/` directory contains multiple experimental configurations documenting the research progression:

| Config | Key Characteristics |
|--------|---------------------|
| `default` | Hybrid finetuned, ResNet-18, partial freeze, adapters |
| `v5–v8` | Early experiments with different architectures and training strategies |
| `v9` | EfficientNet-B0, full backbone unfreezing, ReduceLROnPlateau |
| `v10–v11` | Multimodal fine-tuning with shared backbone |
| `v12b` | Frozen backbone, adapters + attention only |
| `v13–v14` | Hybrid finetuned with SWA, label smoothing, gradient accumulation |
| **`v15`** | **Final: Frozen ResNet-18 + XGBoost (no fine-tuning)** |

---

## Usage

### Training

```bash
# Run with default config
python main.py

# Run with a specific config
python main.py --config configs/v15_config.yaml

# Override parameters via CLI
python main.py --config configs/v15_config.yaml --n_cv_folds 3 --backbone resnet18
```

### Generate Publication Figures

```bash
python generate_results.py \
    --results_dir results/v15_frozen_xgboost_YYYYMMDD_HHMMSS \
    --config configs/v15_config.yaml
```

### Generate Heatmaps

```bash
python generate_heatmaps.py \
    --results_dir results/v15_frozen_xgboost_YYYYMMDD_HHMMSS \
    --config configs/v15_config.yaml \
    --n_total 100 --n_tp 5 --n_fp 5 --n_fn 5 --n_tn 5
```

### Explain CNN Features

```bash
python explain_cnn_features.py \
    --results_dir results/v15_frozen_xgboost_YYYYMMDD_HHMMSS \
    --config configs/v15_config.yaml
```

### Post-Training Analysis

```bash
python analyze_results.py \
    --results_dir results/v15_frozen_xgboost_YYYYMMDD_HHMMSS \
    --config configs/v15_config.yaml
```

---

## Key Design Decisions

1. **Frozen CNN + XGBoost over end-to-end training.** With ~1000 patients, full CNN fine-tuning catastrophically overfits. Frozen ImageNet features fed into a well-regularized XGBoost proved more robust and reproducible.

2. **Patient-level (ideye) splitting.** Both eyes of the same patient share anatomical similarities. Splitting at the eye level rather than the patient level would cause data leakage, inflating performance metrics.

3. **Conservative image augmentation.** Medical corneal maps contain diagnostic patterns in specific spatial locations (e.g., inferior steepening). Aggressive augmentations like random erasing would destroy these clinical signals.

4. **Single-pass feature extraction.** When using `WeightedRandomSampler` or shuffled DataLoaders, extracting CNN features and labels in separate passes leads to misalignment. The `extract_all_features()` function captures everything in a single pass.

5. **SelectKBest on CNN features.** The raw CNN feature vector (e.g., 2048-dim for 4×512 from ResNet-18) contains many irrelevant dimensions for CXL prediction. ANOVA F-test selects the 30 most discriminative features, reducing noise for XGBoost.

6. **Validation on fold's true test set.** Rather than creating an internal train/validation split (which would further reduce the already small training set), both the CNN fine-tuning (Mode C) and XGBoost use the fold's held-out test set for early stopping.

---

## Authors

- **Yamit Cohen** — [yamit.jt@gmail.com](mailto:yamit.jt@gmail.com)
- **Hadar Cohen** — [hal.nls@gmail.com](mailto:hal.nls@gmail.com)
