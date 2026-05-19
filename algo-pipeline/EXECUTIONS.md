# Pipeline Executions

This document contains the commands required to regenerate the final results, tables, and interpretation figures for the publication.

All commands should be run from the `algo-pipeline` directory within the standard python virtual environment or Anaconda environment.

## 0. Full Pipeline Run (Training & Evaluation)

This command executes the full algorithmic pipeline from scratch, reading data, processing models (e.g. frozen CNN + XGBoost), and generating a new unique results directory string with a tracked timestamp.

```bash
python main.py --config configs/v15_config.yaml
```

**Outputs generated:**
Results are saved to a dynamic directory (e.g., `../results/v15_frozen_xgboost_20260413_201200`) and the model snapshots and metrics logs are stored within. You must use this output directory string in the subsequent steps.

## 1. Generate Main Results and Figures

This script evaluates the model, calculates metrics (e.g., ROC-AUC), and produces publication-ready statistical tables and figures (like Subgroup Analysis, DCA, Model Contribution arrays, etc.).

```bash
# Substitute the exact results directory corresponding to your run
export RESULTS_DIR="../results/v15_frozen_xgboost_20260302_075810"

python generate_results.py \
  --results_dir $RESULTS_DIR \
  --config configs/v15_config.yaml
```

**Outputs generated:**
Results are saved to `../results/{run_id}/publication_figures/` (or the directory specified in the config) and include:
- `Table_1_Demographics.csv`
- Decision Curve Analysis plots
- KMax Subgroup Analysis plots
- Feature contribution comparisons

## 2. Generate CNN Interpretability Explanations 

This script relies on the output path constructed during the main evaluation (e.g. `v15_frozen_xgboost_XXXXXXXX_XXXXXX`). It runs SHAP and Grad-CAM analyses to explain what features the CNN is identifying and generates composite visual overlays with actual corneal masks and scientific mapping (Turbo palette).

**Note:** This is a GPU-intensive script that can take a few minutes to process the gradients across all K-Folds.

```bash
# Substitute the exact results directory corresponding to your run
export RESULTS_DIR="../results/v15_frozen_xgboost_20260302_075810"

python explain_cnn_features.py \
  --results_dir $RESULTS_DIR \
  --config configs/v15_config.yaml
```

**Outputs generated:**
Results are saved to `{RESULTS_DIR}/publication_figures/cnn_explanations/` and include:
- `cnn_feature_explanations.csv`: A comprehensive dictionary of the structured clinical names, spatial regions, modalities, SHAP significance, and detailed interpretation sentences explaining importance in classifying Keratoconus.
- `fig_summary_top10_cnn_features.png`: A 2x5 grid showing average GradCAM heatmaps across the top 10 ranked CNN features.
- Individual exemplar grids (e.g., `exemplar_rank01_cnn_3.png`) showing 2 extreme CXL patients and 2 Normal patients for each CNN feature.
