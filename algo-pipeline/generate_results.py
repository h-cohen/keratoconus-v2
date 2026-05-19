"""
Publication-quality results generation script for the Keratoconus ML pipeline.
Generates all figures needed for paper submission.

Usage:
    python generate_results.py --results_dir results/v12b_... --config configs/v12b_config.yaml
"""
import argparse
import sys
import re
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'nature', 'no-latex'])
    # Optional adjustments for no-latex compatibility
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
    })
except ImportError:
    print("Warning: scienceplots not found. Falling back to seaborn style.")
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9
    })
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import joblib
import xgboost as xgb
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))
from src.utils import load_config
from src.data import load_data, IMAGE_TYPES, get_image_transform, KeratoconusDataset, collate_keratoconus
from src.models import MultiBackboneCNNEncoder, SharedBackboneCNNEncoder
from torch.utils.data import DataLoader

# ── Consistent palette ──────────────────────────────────────────────────────
COLOR_MM = '#2166ac'       # Multimodal — blue
COLOR_BL = '#b2182b'       # Baseline — red
COLOR_POS = '#d6604d'      # Positive class
COLOR_NEG = '#4393c3'      # Negative class
FOLD_CMAP = plt.cm.Set2
DPI = 300


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_fig(fig, out_dir, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(out_dir / f'{name}.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(out_dir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}")


def load_fold_models(fold, results_dir, device, config):
    """Load CNN encoder, XGBoost model, and transformers for a fold."""
    models_dir = Path(results_dir) / 'models' / f'fold_{fold}'
    if not models_dir.exists():
        return None

    use_shared = config.get('shared_backbone', False)
    EncoderClass = SharedBackboneCNNEncoder if use_shared else MultiBackboneCNNEncoder

    cnn = EncoderClass(
        image_types=IMAGE_TYPES,
        backbone_name=config['backbone'],
        freeze_mode=config['freeze_mode'],
        use_attention=config.get('use_attention', True),
        use_cross_modal_fusion=config.get('use_cross_modal_fusion', True),
        adapter_mode=config.get('adapter_mode', 'none')
    ).to(device)

    enc_path = models_dir / 'cnn_encoder.pth'
    if enc_path.exists():
        cnn.load_state_dict(torch.load(enc_path, map_location=device))
    cnn.eval()

    xgb_model = None
    xgb_path = models_dir / 'xgb_model.json'
    if xgb_path.exists():
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_path))

    scaler_cnn = joblib.load(models_dir / 'scaler_cnn.joblib') if (models_dir / 'scaler_cnn.joblib').exists() else None
    scaler_num = joblib.load(models_dir / 'scaler_num.joblib') if (models_dir / 'scaler_num.joblib').exists() else None
    selector   = joblib.load(models_dir / 'selector.joblib')   if (models_dir / 'selector.joblib').exists()   else None

    return {'cnn': cnn, 'xgb': xgb_model, 'scaler_cnn': scaler_cnn,
            'scaler_num': scaler_num, 'selector': selector}


def run_oof_inference(df, config, results_dir, device, numeric_features):
    """Run out-of-fold inference across all folds. Returns DataFrame with predictions.
    
    First tries to load saved OOF predictions from training (oof_predictions.csv).
    This guarantees exact match with training-reported AUCs.
    Falls back to re-running inference if the file is missing.
    """
    # ─── Try loading saved OOF predictions from training ───
    oof_path = Path(results_dir) / 'oof_predictions.csv'
    if oof_path.exists():
        print(f"  ✓ Loading saved OOF predictions from {oof_path}")
        oof_df = pd.read_csv(oof_path)
        
        # Verify per-fold AUCs match
        for fold in sorted(oof_df['fold'].unique()):
            sub = oof_df[oof_df['fold'] == fold]
            auc = roc_auc_score(sub['y_true'], sub['y_pred'])
            print(f"    Fold {fold}: AUC = {auc:.4f} ({len(sub)} samples)")
        
        overall_auc = roc_auc_score(oof_df['y_true'], oof_df['y_pred'])
        fold_aucs = [roc_auc_score(oof_df[oof_df['fold']==f]['y_true'], 
                                    oof_df[oof_df['fold']==f]['y_pred']) 
                     for f in sorted(oof_df['fold'].unique())]
        mean_auc = np.mean(fold_aucs)
        print(f"    Mean AUC: {mean_auc:.4f}, Pooled AUC: {overall_auc:.4f}")
        
        # Load XGBoost models for SHAP analysis
        fold_xgb_models = {}
        for fold in sorted(oof_df['fold'].unique()):
            models = load_fold_models(fold, results_dir, device, config)
            if models is not None:
                fold_xgb_models[fold] = models['xgb']
        
        return oof_df, fold_xgb_models
    
    # ─── Fallback: re-run inference ───
    print("  ⚠ No saved OOF predictions found. Re-running inference...")
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    ideye_labels = ideye_to_label.values
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    image_dir = Path(config['image_dir'])
    img_size = config.get('image_size', 224)
    val_transform = get_image_transform(training=False, size=img_size)

    records = []
    fold_xgb_models = {}

    for fold, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_labels), 1):
        print(f"  Fold {fold}/{n_folds} inference...")
        models = load_fold_models(fold, results_dir, device, config)
        if models is None:
            print(f"    ⚠ Fold {fold} models not found, skipping")
            continue

        fold_xgb_models[fold] = models['xgb']
        test_ideyes = set(unique_ideyes[test_idx])
        test_df = df[df['ideye'].isin(test_ideyes)].copy()
        ds = KeratoconusDataset(test_df, image_dir, numeric_features, val_transform, IMAGE_TYPES)
        loader = DataLoader(ds, batch_size=config['batch_size'],
                            collate_fn=collate_keratoconus,
                            num_workers=0)

        all_emb, all_num, all_lbl = [], [], []
        with torch.no_grad():
            for imgs, nums, lbls in tqdm(loader, desc=f"    Fold {fold}", leave=False):
                imgs = {k: v.to(device) for k, v in imgs.items()}
                emb = models['cnn'](imgs).cpu().numpy()
                all_emb.append(emb)
                all_num.append(nums.numpy())
                all_lbl.extend(lbls.numpy())

        emb = np.vstack(all_emb)
        num = np.vstack(all_num)

        emb_proc = models['scaler_cnn'].transform(emb) if models['scaler_cnn'] else emb
        if models['selector']:
            emb_proc = models['selector'].transform(emb_proc)
        num_proc = models['scaler_num'].transform(num) if models['scaler_num'] else num
        X = np.hstack([emb_proc, num_proc])

        preds = models['xgb'].predict_proba(X)[:, 1] if models['xgb'] else np.full(len(all_lbl), 0.5)

        for i, row_idx in enumerate(test_df.index):
            records.append({
                'ideye': test_df.loc[row_idx, 'ideye'],
                'y_true': test_df.loc[row_idx, 'y'],
                'y_pred': preds[i],
                'fold': fold,
            })

    return pd.DataFrame(records), fold_xgb_models


# ═══════════════════════════════════════════════════════════════════════════
#  1. Data Preparation Flow Diagram
# ═══════════════════════════════════════════════════════════════════════════

def plot_data_flow(config, out_dir):
    """Programmatic flow diagram of data filtering pipeline."""
    df_raw = pd.read_csv(config['data_csv'], dtype={'id': str})
    with open(config['features_txt'], 'r') as f:
        num_feats = [l.strip() for l in f.readlines()]

    total = len(df_raw)
    n_pos_raw = int((df_raw['y'] == 1).sum()) if 'y' in df_raw.columns else 0
    n_neg_raw = total - n_pos_raw

    df1 = df_raw[df_raw['has_all_images']]
    n1 = len(df1)
    df2 = df1.dropna(subset=num_feats)
    n2 = len(df2)

    kmax_min_filter = config.get('kmax_min_filter', None)
    kmax_max_filter = config.get('kmax_max_filter', config.get('kmax_filter', None))
    exclusive_kmax_min = config.get('exclusive_kmax_min', True)
    exclusive_kmax_max = config.get('exclusive_kmax_max', False)
    
    kmax_col = 'Km F (D):'
    df3 = df2.copy()
    if kmax_col in df3.columns:
        if kmax_min_filter is not None:
            if exclusive_kmax_min:
                df3 = df3[df3[kmax_col] > kmax_min_filter]
            else:
                df3 = df3[df3[kmax_col] >= kmax_min_filter]
                
        if kmax_max_filter is not None:
            if exclusive_kmax_max:
                df3 = df3[df3[kmax_col] < kmax_max_filter]
            else:
                df3 = df3[df3[kmax_col] <= kmax_max_filter]

    n3 = len(df3)
    n_pos_final = int((df3['y'] == 1).sum()) if len(df3) > 0 and 'y' in df3.columns else 0
    n_neg_final = n3 - n_pos_final

    stages = [
        (f'Raw Dataset\n{total} samples\n({n_pos_raw} CXL / {n_neg_raw} Normal)', '#e0e0e0'),
        (f'has_all_images\n{n1} samples\n(−{total - n1} removed)', '#bbdefb'),
        (f'dropna(numeric)\n{n2} samples\n(−{n1 - n2} removed)', '#90caf9'),
    ]
    if kmax_min_filter is not None or kmax_max_filter is not None:
        filter_str = []
        if kmax_min_filter is not None: filter_str.append(f">{'' if exclusive_kmax_min else '='}{kmax_min_filter}")
        if kmax_max_filter is not None: filter_str.append(f"<{'' if exclusive_kmax_max else '='}{kmax_max_filter}")
        stages.append((f'KMax {", ".join(filter_str)}\n{n3} samples\n(−{n2 - n3} removed)', '#64b5f6'))
    stages.append((f'Final Dataset\n{n3} samples\n({n_pos_final} CXL / {n_neg_final} Normal)', '#42a5f5'))

    fig, ax = plt.subplots(figsize=(4, 1.2 * len(stages) + 0.5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(stages) * 1.2 + 0.3)
    ax.axis('off')

    box_h, box_w = 0.9, 3.2
    x_center = 2.0

    for i, (text, color) in enumerate(stages):
        y = (len(stages) - i) * 1.2
        box = FancyBboxPatch((x_center - box_w/2, y - box_h/2), box_w, box_h,
                             boxstyle="round,pad=0.1", facecolor=color, edgecolor='#333333', linewidth=1)
        ax.add_patch(box)
        ax.text(x_center, y, text, ha='center', va='center', fontsize=7, fontweight='bold')
        if i > 0:
            ax.annotate('', xy=(x_center, y + box_h/2), xytext=(x_center, y + 1.2 - box_h/2),
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    fig.suptitle('Data Preparation Pipeline', fontsize=9, fontweight='bold', y=0.98)
    save_fig(fig, out_dir, 'fig1_data_flow')


# ═══════════════════════════════════════════════════════════════════════════
#  2. ML Pipeline Diagram
# ═══════════════════════════════════════════════════════════════════════════

def plot_pipeline_overview(config, out_dir):
    """Visual overview of the multimodal ML pipeline."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    def box(x, y, w, h, text, subtext=None, color='#e3f2fd', fs=7):
        r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.1", facecolor=color, edgecolor='#333', lw=1.2)
        ax.add_patch(r)
        
        if subtext:
            ax.text(x, y + 0.15, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='#111')
            ax.text(x, y - 0.2, subtext, ha='center', va='center', fontsize=fs-2.5, color='#444', style='italic')
        else:
            ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='#111')

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#222', lw=1.5))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom', fontsize=5, color='#555')

    # Input Images
    modalities = ['Corneal Thickness', 'Curvature Front', 'Elevation Front', 'Elevation Back']
    for i, m in enumerate(modalities):
        y_pos = 2.4 - i * 0.55
        box(0.9, y_pos, 1.4, 0.45, m.replace(' ', '\n'), subtext="(Image Map)", color='#fff9c4', fs=6)
        arrow(1.6, y_pos, 2.3, 1.5)

    # Backbone Feature Extraction
    shared_text = "Shared" if config.get("shared_backbone") else "Independent"
    box(3.1, 1.5, 1.3, 1.8, f'CNN Encoder\n({shared_text})', subtext="Extracts visual\npatterns", color='#bbdefb')
    arrow(3.75, 1.5, 4.4, 1.5)
    
    # Attention Mapping
    box(5.0, 1.5, 1.0, 0.9, 'Attention\nMechanism', subtext="Highlights\ncritical regions", color='#c8e6c9')
    arrow(5.5, 1.5, 6.2, 1.5)
    
    # Vector Pooling
    box(6.7, 1.5, 0.8, 0.9, 'Feature\nPooling', subtext="Compress to\n1D Profile", color='#d1c4e9')
    arrow(7.1, 1.5, 7.8, 1.6)

    # Numeric Features coming in from the bottom
    box(6.4, 0.4, 1.2, 0.5, 'Clinical Metrics', subtext="(e.g. KMax, Pachymetry)", color='#fff9c4', fs=6)
    arrow(7.0, 0.4, 7.8, 1.4)

    # Combine and Select
    box(8.3, 1.5, 0.9, 1.0, 'Feature\nFusion', subtext="Combine top\nindicators", color='#ffe0b2')
    arrow(8.75, 1.5, 9.2, 1.5)
    
    # Final Classifier
    box(9.65, 1.5, 0.8, 0.9, 'AI Classifier\n(XGBoost)', subtext="Predict\nDiagnosis", color='#ef9a9a')
    
    # Final Output
    arrow(10.05, 1.5, 10.6, 1.5)
    ax.text(10.8, 1.5, 'CXL\nProbability', fontsize=8, fontweight='bold', va='center', ha='center', color='#b2182b')

    fig.suptitle('End-to-End Artificial Intelligence Diagnosis Pipeline', fontsize=11, fontweight='bold', y=0.95)
    ax.set_xlim(0, 11)
    save_fig(fig, out_dir, 'fig2_pipeline_enhanced')


# ═══════════════════════════════════════════════════════════════════════════
#  3. ROC-AUC Curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_roc_curves(oof_df, df, out_dir, is_global=True):
    """Per-fold + mean ROC curves with CI bands."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    if is_global:
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []

        folds = sorted(oof_df['fold'].unique())
        for fold in folds:
            sub = oof_df[oof_df['fold'] == fold]
            if len(np.unique(sub['y_true'])) < 2:
                continue
            fpr, tpr, _ = roc_curve(sub['y_true'], sub['y_pred'])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0) if tprs else np.zeros_like(mean_fpr)
        if tprs:
            mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0) if tprs else np.zeros_like(mean_fpr)
        
        valid_folds_df = oof_df.groupby('fold').filter(lambda x: len(np.unique(x['y_true'])) > 1)
        if valid_folds_df.empty:
            valid_folds_df = oof_df # fallback
            print("  ⚠ No valid folds for mean ROC calc, plotting mean overall")
        
        fold_aucs = [roc_auc_score(valid_folds_df[valid_folds_df['fold'] == f]['y_true'],
                                   valid_folds_df[valid_folds_df['fold'] == f]['y_pred']) 
                     for f in valid_folds_df['fold'].unique()]
        mean_auc = np.mean(fold_aucs) if fold_aucs else roc_auc_score(oof_df['y_true'], oof_df['y_pred'])
        std_auc = np.std(fold_aucs) if fold_aucs else 0.0

        ax.plot(mean_fpr, mean_tpr, color=COLOR_MM, lw=2,
                label=f'Mean (AUC={mean_auc:.3f} ± {std_auc:.3f})')
        ax.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1), color=COLOR_MM, alpha=0.15)
        
        # Also plot Pooled
        pooled_fpr, pooled_tpr, _ = roc_curve(oof_df['y_true'], oof_df['y_pred'])
        pooled_auc = roc_auc_score(oof_df['y_true'], oof_df['y_pred'])
        ax.plot(pooled_fpr, pooled_tpr, color=COLOR_BL, lw=1.5, ls='--',
                label=f'Pooled (AUC={pooled_auc:.3f})')
        
        # Baseline KMax ROC
        kmax_candidates = ['KMax Sagittal Front (D)', 'Km F (D):']
        kmax_col = next((c for c in kmax_candidates if c in df.columns), None)
        if kmax_col:
            # Map kmax to oof_df to ensure alignment
            kmax_map = df.groupby('ideye')[kmax_col].first()
            oof_valid = oof_df.copy()
            oof_valid['kmax'] = oof_valid['ideye'].map(kmax_map)
            oof_valid = oof_valid.dropna(subset=['kmax'])
            if not oof_valid.empty:
                bl_fpr, bl_tpr, _ = roc_curve(oof_valid['y_true'], oof_valid['kmax'])
                bl_auc = roc_auc_score(oof_valid['y_true'], oof_valid['kmax'])
                if bl_auc < 0.5: # If negatively correlated, flip it
                    bl_auc = 1 - bl_auc
                    bl_fpr, bl_tpr, _ = roc_curve(oof_valid['y_true'], -oof_valid['kmax'])
                ax.plot(bl_fpr, bl_tpr, color='#888888', lw=1.5, ls=':',
                        label=f'Baseline KMax (AUC={bl_auc:.3f})')
    else:
        # Subgroups: only plot pooled since folds might have very small sample sizes
        pooled_fpr, pooled_tpr, _ = roc_curve(oof_df['y_true'], oof_df['y_pred'])
        pooled_auc = roc_auc_score(oof_df['y_true'], oof_df['y_pred'])
        ax.plot(pooled_fpr, pooled_tpr, color=COLOR_MM, lw=2,
                label=f'Pooled (AUC={pooled_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (5-Fold CV)' if is_global else 'Pooled ROC Curve')
    ax.legend(fontsize=5, loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    save_fig(fig, out_dir, 'fig3_roc_auc')


# ═══════════════════════════════════════════════════════════════════════════
#  4. Precision / Recall / F1 vs Threshold
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics_vs_threshold(oof_df, out_dir):
    """Precision, Recall, F1 as a function of classification threshold."""
    y_true = oof_df['y_true'].values
    y_pred = oof_df['y_pred'].values
    thresholds = np.linspace(0.01, 0.99, 200)

    prec_arr, rec_arr, f1_arr = [], [], []
    for t in thresholds:
        y_bin = (y_pred >= t).astype(int)
        prec_arr.append(precision_score(y_true, y_bin, zero_division=0))
        rec_arr.append(recall_score(y_true, y_bin, zero_division=0))
        f1_arr.append(f1_score(y_true, y_bin, zero_division=0))

    f1_arr = np.array(f1_arr)
    best_idx = np.argmax(f1_arr)
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(thresholds, prec_arr, label='Precision', color='#1b9e77', lw=1.5)
    ax.plot(thresholds, rec_arr, label='Recall', color='#d95f02', lw=1.5)
    ax.plot(thresholds, f1_arr, label='F1 Score', color='#7570b3', lw=1.5)
    ax.axvline(best_t, color='grey', ls='--', lw=0.8, alpha=0.7,
               label=f'Best F1 @ t={best_t:.2f}')
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1 vs Threshold')
    ax.legend(fontsize=5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    save_fig(fig, out_dir, 'fig4_metrics_vs_threshold')
    return best_t


# ═══════════════════════════════════════════════════════════════════════════
#  5. SHAP Feature Importance
# ═══════════════════════════════════════════════════════════════════════════

def plot_shap_analysis(oof_df, fold_xgb_models, config, df_full, numeric_features, out_dir):
    """SHAP beeswarm and bar plots aggregated across folds."""
    try:
        import shap
    except ImportError:
        print("  ⚠ shap not installed — skipping SHAP plots (pip install shap)")
        return

    n_numeric = len(numeric_features)

    # Use the best-performing fold for SHAP visualization
    best_fold = max(fold_xgb_models.keys(),
                    key=lambda f: roc_auc_score(
                        oof_df[oof_df['fold'] == f]['y_true'],
                        oof_df[oof_df['fold'] == f]['y_pred']))

    xgb_model = fold_xgb_models[best_fold]
    if xgb_model is None:
        print("  ⚠ No XGBoost model available for SHAP")
        return

    # Re-run inference for best fold to get feature matrix
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df_full.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    ideye_labels = ideye_to_label.values
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    image_dir = Path(config['image_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = None
    for fold, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_labels), 1):
        if fold != best_fold:
            continue
        models = load_fold_models(fold, Path(config['_results_dir']), device, config)
        if models is None:
            print("  ⚠ Could not load models for SHAP")
            return
        test_ideyes = set(unique_ideyes[test_idx])
        test_df = df_full[df_full['ideye'].isin(test_ideyes)]
        ds = KeratoconusDataset(test_df, image_dir, numeric_features,
                                get_image_transform(training=False, size=config.get('image_size', 224)),
                                IMAGE_TYPES)
        loader = DataLoader(ds, batch_size=config['batch_size'],
                            collate_fn=collate_keratoconus, num_workers=config.get('num_workers', 4))
        embs, nums = [], []
        with torch.no_grad():
            for imgs, nm, lb in loader:
                imgs = {k: v.to(device) for k, v in imgs.items()}
                embs.append(models['cnn'](imgs).cpu().numpy())
                nums.append(nm.numpy())
        emb = np.vstack(embs)
        num = np.vstack(nums)
        emb_proc = models['scaler_cnn'].transform(emb) if models['scaler_cnn'] else emb
        if models['selector']:
            emb_proc = models['selector'].transform(emb_proc)
        num_proc = models['scaler_num'].transform(num) if models['scaler_num'] else num
        X = np.hstack([emb_proc, num_proc])

    if X is None:
        print("  ⚠ Could not extract features for SHAP")
        return

    n_features = X.shape[1]
    n_cnn = n_features - n_numeric
    feature_names = [f'cnn_{i}' for i in range(n_cnn)] + list(numeric_features)

    try:
        # Use KernelExplainer as a robust model-agnostic fallback for XGBoost compatibility issues
        predict_fn = lambda x: xgb_model.predict_proba(x)[:, 1]
        background = shap.kmeans(X, 10)
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Kernel explainer is extremely slow, evaluate on a subset of 150 samples uniformly
        sub_idx = np.linspace(0, len(X) - 1, min(len(X), 150), dtype=int)
        X_sub = X[sub_idx]
        
        shap_values = explainer.shap_values(X_sub)
        X = X_sub  # Update X so the summary plots match the evaluated shaped elements
    except Exception as e:
        print(f"  ⚠ SHAP analysis failed: {e} — skipping SHAP plots")
        return

    # Bar plot — top 20
    fig_bar, ax_bar = plt.subplots(figsize=(4, 4))
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[-20:]
    top_names = [feature_names[i].replace(':','') for i in top_idx]
    top_vals = mean_abs[top_idx]
    colors = [COLOR_MM if 'cnn' in n else COLOR_BL for n in top_names]
    ax_bar.barh(range(len(top_idx)), top_vals, color=colors)
    ax_bar.set_yticks(range(len(top_idx)))
    ax_bar.set_yticklabels(top_names, fontsize=5)
    ax_bar.set_xlabel('Mean |SHAP value|')
    ax_bar.set_title('Top 20 Feature Importance (SHAP)')
    legend_patches = [mpatches.Patch(color=COLOR_MM, label='CNN features'),
                      mpatches.Patch(color=COLOR_BL, label='Numeric features')]
    ax_bar.legend(handles=legend_patches, fontsize=5, loc='lower right')
    save_fig(fig_bar, out_dir, 'fig5a_shap_bar')

    # Beeswarm
    fig_bee, ax_bee = plt.subplots(figsize=(4, 5))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=20, show=False, plot_size=None)
    plt.title('SHAP Beeswarm (Top 20 Features)', fontsize=8)
    plt.tight_layout()
    fig_bee = plt.gcf()
    save_fig(fig_bee, out_dir, 'fig5b_shap_beeswarm')


# ═══════════════════════════════════════════════════════════════════════════
#  6. Calibration Curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_calibration(oof_df, out_dir):
    """Reliability diagram showing model calibration."""
    y_true = oof_df['y_true'].values
    y_pred = oof_df['y_pred'].values
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot(prob_pred, prob_true, 's-', color=COLOR_MM, lw=1.5, markersize=4, label='Model')
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8, label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend(fontsize=5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    save_fig(fig, out_dir, 'fig6_calibration')


# ═══════════════════════════════════════════════════════════════════════════
#  7. Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_mat(oof_df, threshold, out_dir):
    """Confusion matrix at given threshold, plus a high-precision threshold variant."""
    y_true = oof_df['y_true'].values
    y_bin = (oof_df['y_pred'].values >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_bin, labels=[0, 1])
    
    threshold_high = 0.55 # Explicit high-precision cutoff
    y_bin_high = (oof_df['y_pred'].values >= threshold_high).astype(int)
    cm_high = confusion_matrix(y_true, y_bin_high, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
    labels = ['Normal', 'CXL']
    
    # Left: Best F1 Threshold
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=False)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Optimal F1 (t={threshold:.2f})', fontsize=8)

    # Right: High Precision Threshold
    sns.heatmap(cm_high, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=False)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'High Precision (t={threshold_high:.2f})', fontsize=8)

    fig.suptitle('Classification Outcomes (Count)', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig7_confusion_matrices')


# ═══════════════════════════════════════════════════════════════════════════
#  8. Training Curves (parsed from log)
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(results_dir, out_dir):
    """Parse training log and plot per-fold training curves."""
    log_files = list(Path(results_dir).glob('training_*.log'))
    if not log_files:
        print("  ⚠ No training log found — skipping training curves")
        return

    pat = re.compile(
        r'Fine-tune Epoch (\d+)/(\d+): '
        r'Train Loss=([\d.]+), Train AUC=([\d.]+), Val AUC=([\d.]+), LR=([\d.e+-]+)')
    fold_pat = re.compile(r'Fold (\d+)/(\d+)')

    with open(log_files[0]) as f:
        lines = f.readlines()

    fold_data = {}
    current_fold = None
    for line in lines:
        fm = fold_pat.search(line)
        if fm:
            current_fold = int(fm.group(1))
            fold_data.setdefault(current_fold, {'epoch': [], 'train_loss': [],
                                                 'train_auc': [], 'val_auc': []})
        m = pat.search(line)
        if m and current_fold is not None:
            fold_data[current_fold]['epoch'].append(int(m.group(1)))
            fold_data[current_fold]['train_loss'].append(float(m.group(3)))
            fold_data[current_fold]['train_auc'].append(float(m.group(4)))
            fold_data[current_fold]['val_auc'].append(float(m.group(5)))

    if not fold_data:
        print("  ⚠ Could not parse training curves from log")
        return

    n_folds = len(fold_data)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for fold, data in fold_data.items():
        c = FOLD_CMAP(fold - 1)
        axes[0].plot(data['epoch'], data['train_loss'], color=c, lw=0.8, alpha=0.7, label=f'Fold {fold}')
        axes[1].plot(data['epoch'], data['train_auc'], color=c, lw=0.8, alpha=0.5, ls='--')
        axes[1].plot(data['epoch'], data['val_auc'], color=c, lw=1.2, label=f'Fold {fold}')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Fine-tuning Loss')
    axes[0].legend(fontsize=4)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Train (dashed) / Val (solid) AUC')
    axes[1].legend(fontsize=4)

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig8_training_curves')


# ═══════════════════════════════════════════════════════════════════════════
#  9. Feature Violin Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_violins(df, out_dir):
    """Violin plots of top clinical features split by class."""
    candidates = ['Km F (D):', 'K2 F (D):', 'Pachy Min:', 'IHD:',
                  'KMax Sagittal Front (D)', 'Astig F (D):']
    feats = [c for c in candidates if c in df.columns]
    if not feats:
        print("  ⚠ No clinical features found for violin plots")
        return

    n = len(feats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    df_plot = df.copy()
    df_plot['Class'] = df_plot['y'].map({0: 'Normal', 1: 'CXL'})

    for i, feat in enumerate(feats):
        sns.violinplot(data=df_plot, x='Class', y=feat, hue='Class', ax=axes[i],
                       palette={'Normal': COLOR_NEG, 'CXL': COLOR_POS},
                       inner='quartile', linewidth=0.8, legend=False)
        axes[i].set_title(feat, fontsize=7)
        axes[i].set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Feature Distributions by Class', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig9_feature_violins')


# ═══════════════════════════════════════════════════════════════════════════
#  10. Precision-Recall Curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_pr_curve(oof_df, out_dir, is_global=True):
    """Precision-Recall curve with per-fold and mean."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    prec_all, rec_all, _ = precision_recall_curve(oof_df['y_true'], oof_df['y_pred'])
    ap_all = average_precision_score(oof_df['y_true'], oof_df['y_pred'])

    if is_global:
        folds = sorted(oof_df['fold'].unique())
        for fold in folds:
            sub = oof_df[oof_df['fold'] == fold]
            if len(np.unique(sub['y_true'])) < 2:
                continue
            prec, rec, _ = precision_recall_curve(sub['y_true'], sub['y_pred'])
            ap = average_precision_score(sub['y_true'], sub['y_pred'])
            ax.plot(rec, prec, color=FOLD_CMAP(fold - 1), alpha=0.3, lw=0.8,
                    label=f'Fold {fold} (AP={ap:.3f})')

        ax.plot(rec_all, prec_all, color=COLOR_MM, lw=2, label=f'Pooled (AP={ap_all:.3f})')
    else:
        ax.plot(rec_all, prec_all, color=COLOR_MM, lw=2, label=f'Pooled (AP={ap_all:.3f})')

    prevalence = oof_df['y_true'].mean()
    ax.axhline(prevalence, color='grey', ls=':', lw=0.8, alpha=0.5, label=f'Prevalence ({prevalence:.2f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve' if is_global else 'Pooled PR Curve')
    ax.legend(fontsize=5, loc='upper right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    save_fig(fig, out_dir, 'fig10_pr_curve')


# ═══════════════════════════════════════════════════════════════════════════
#  11. Summary Statistics Table
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary_table(oof_df, best_threshold, out_dir, is_global=True):
    """Generate LaTeX and CSV summary table."""
    rows = []
    
    if is_global:
        folds = sorted(oof_df['fold'].unique())
        for fold in folds:
            sub = oof_df[oof_df['fold'] == fold]
            y_t, y_p = sub['y_true'].values, sub['y_pred'].values
            if len(np.unique(y_t)) < 2:
                continue
            y_bin = (y_p >= best_threshold).astype(int)
            auc = roc_auc_score(y_t, y_p)
            f1 = f1_score(y_t, y_bin, zero_division=0)
            sens = recall_score(y_t, y_bin, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_t, y_bin, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            rows.append({'Fold': fold, 'AUC': auc, 'F1': f1,
                         'Sensitivity': sens, 'Specificity': spec,
                         'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn})

        tbl = pd.DataFrame(rows)

        # Add mean ± std row
        mean_row = {'Fold': 'Mean±SD'}
        for col in ['AUC', 'F1', 'Sensitivity', 'Specificity']:
            if not tbl.empty:
                mean_row[col] = f"{tbl[col].mean():.3f}±{tbl[col].std():.3f}"
            else:
                mean_row[col] = "N/A"
        for col in ['TP', 'FP', 'FN', 'TN']:
            mean_row[col] = ''
        tbl = pd.concat([tbl, pd.DataFrame([mean_row])], ignore_index=True)
    else:
        tbl = pd.DataFrame(columns=['Fold', 'AUC', 'F1', 'Sensitivity', 'Specificity', 'TP', 'FP', 'FN', 'TN'])

    # Always add pooled row
    y_t, y_p = oof_df['y_true'].values, oof_df['y_pred'].values
    y_bin = (y_p >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_t, y_bin, labels=[0, 1]).ravel()
    pooled_row = {
        'Fold': 'Pooled',
        'AUC': roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
        'F1': f1_score(y_t, y_bin, zero_division=0),
        'Sensitivity': recall_score(y_t, y_bin, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    }
    
    tbl = pd.concat([tbl, pd.DataFrame([pooled_row])], ignore_index=True)

    tbl.to_csv(out_dir / 'summary_table.csv', index=False)
    # LaTeX
    with open(out_dir / 'summary_table.tex', 'w') as f:
        f.write(tbl.to_latex(index=False))
    print(f"  ✓ summary_table (.csv + .tex)")


# ═══════════════════════════════════════════════════════════════════════════
#  12. Table 1 — Patient Demographics & Clinical Characteristics
# ═══════════════════════════════════════════════════════════════════════════

def generate_table1(df, out_dir):
    """Generate Table 1: Patient demographics stratified by CXL status."""
    from scipy.stats import mannwhitneyu

    normal = df[df['y'] == 0]
    cxl = df[df['y'] == 1]

    clinical_features = {
        'Km F (D):': 'Keratometry Front (D)',
        'K2 F (D):': 'K2 Front (D)',
        'Pachy Min:': 'Min Pachymetry (μm)',
        'IHD:': 'Index of Height Decentration',
        'KMax Sagittal Front (D)': 'KMax (D)',
        'Astig F (D):': 'Astigmatism Front (D)',
        'CKI:': 'Central Keratoconus Index',
        'Rmin:': 'Min Radius of Curvature',
    }

    rows = []
    rows.append({'Feature': 'N (eyes)', 'Normal': str(len(normal)),
                 'CXL': str(len(cxl)), 'p-value': ''})

    n_unique_normal = normal['ideye'].nunique() if 'ideye' in normal.columns else len(normal)
    n_unique_cxl = cxl['ideye'].nunique() if 'ideye' in cxl.columns else len(cxl)
    rows.append({'Feature': 'N (unique eyes)', 'Normal': str(n_unique_normal),
                 'CXL': str(n_unique_cxl), 'p-value': ''})

    for col, label in clinical_features.items():
        if col not in df.columns:
            continue
        n_vals = normal[col].dropna()
        c_vals = cxl[col].dropna()
        if len(n_vals) > 5 and len(c_vals) > 5:
            _, p = mannwhitneyu(n_vals, c_vals, alternative='two-sided')
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            rows.append({
                'Feature': label,
                'Normal': f"{n_vals.mean():.2f} ± {n_vals.std():.2f}",
                'CXL': f"{c_vals.mean():.2f} ± {c_vals.std():.2f}",
                'p-value': p_str
            })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / 'table1_demographics.csv', index=False)

    # Render as figure
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1.2))
    ax.axis('off')
    table_obj = ax.table(
        cellText=tbl.values, colLabels=tbl.columns,
        loc='center', cellLoc='center'
    )
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(7)
    table_obj.scale(1.0, 1.5)
    # Header styling
    for j in range(len(tbl.columns)):
        table_obj[0, j].set_facecolor('#2166ac')
        table_obj[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(tbl.columns)):
            if i % 2 == 0:
                table_obj[i, j].set_facecolor('#f0f0f0')

    fig.suptitle('Table 1: Patient Demographics & Clinical Characteristics',
                 fontsize=10, fontweight='bold', y=0.95)
    plt.tight_layout()
    save_fig(fig, out_dir, 'table1_demographics')

    # LaTeX
    with open(out_dir / 'table1_demographics.tex', 'w') as f:
        f.write(tbl.to_latex(index=False))
    print(f"  ✓ table1_demographics (.csv + .tex + figure)")


# ═══════════════════════════════════════════════════════════════════════════
#  13. Decision Curve Analysis (DCA)
# ═══════════════════════════════════════════════════════════════════════════

def plot_decision_curve(oof_df, out_dir):
    """Decision Curve Analysis showing net benefit vs threshold probability."""
    y_true = oof_df['y_true'].values
    y_pred = oof_df['y_pred'].values
    n = len(y_true)
    prevalence = y_true.mean()
    thresholds = np.linspace(0.01, 0.95, 200)

    # Net benefit: Model
    nb_model = []
    for t in thresholds:
        y_bin = (y_pred >= t).astype(int)
        tp = np.sum((y_bin == 1) & (y_true == 1))
        fp = np.sum((y_bin == 1) & (y_true == 0))
        nb = tp / n - fp / n * (t / (1 - t))
        nb_model.append(nb)

    # Net benefit: Treat All
    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(thresholds, nb_model, color=COLOR_MM, lw=2, label='Multimodal Model')
    ax.plot(thresholds, nb_all, color='#888888', lw=1.5, ls='--', label='Treat All')
    ax.axhline(0, color='black', lw=0.8, ls=':', alpha=0.6, label='Treat None')

    # Shade region where model has net benefit
    nb_model_arr = np.array(nb_model)
    nb_all_arr = np.array(nb_all)
    benefit_region = nb_model_arr > np.maximum(nb_all_arr, 0)
    ax.fill_between(thresholds, 0, nb_model_arr,
                     where=benefit_region, alpha=0.1, color=COLOR_MM)

    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend(fontsize=6, loc='upper right')
    ax.set_xlim([0, 0.8])
    y_max = max(max(nb_model), prevalence) * 1.15
    ax.set_ylim([-0.02, y_max])
    save_fig(fig, out_dir, 'fig_decision_curve')


# ═══════════════════════════════════════════════════════════════════════════
#  14. Subgroup Analysis by KMax Severity
# ═══════════════════════════════════════════════════════════════════════════

def plot_subgroup_analysis(oof_df, df_full, out_dir):
    """Model performance stratified by KMax severity subgroups."""
    # Find KMax column
    kmax_candidates = ['KMax Sagittal Front (D)', 'Km F (D):']
    kmax_col = None
    for c in kmax_candidates:
        if c in df_full.columns:
            kmax_col = c
            break
    if kmax_col is None:
        print("  ⚠ No KMax column found — skipping subgroup analysis")
        return

    # Merge KMax into OOF predictions via ideye
    kmax_map = df_full.groupby('ideye')[kmax_col].first()
    oof = oof_df.copy()
    oof['kmax'] = oof['ideye'].map(kmax_map)
    oof = oof.dropna(subset=['kmax'])

    # Define severity subgroups
    bins = [0, 46.5, 48, 53, 55]
    labels_short = ['<46.5D', '46.5–48D', '48–53D','53-55D']
    labels_long  = ['<46.5\n(Stage0)','46.5-48D\n(Stage 1)', '48–53D\n(Stage 2)', '53–55D\n(Stage 3)']
    oof['severity'] = pd.cut(oof['kmax'], bins=bins, labels=labels_short, include_lowest=True)

    subgroups = []
    for short, long in zip(labels_short, labels_long):
        sub = oof[oof['severity'] == short]
        n_total = len(sub)
        n_pos = int(sub['y_true'].sum())
        if n_total >= 10 and len(sub['y_true'].unique()) > 1:
            auc = roc_auc_score(sub['y_true'], sub['y_pred'])
        else:
            auc = None
        subgroups.append({'Severity': long, 'Short': short,
                          'AUC': auc, 'N': n_total, 'N_CXL': n_pos})

    sg_df = pd.DataFrame(subgroups)
    valid = sg_df.dropna(subset=['AUC'])

    if valid.empty:
        print("  ⚠ No valid subgroups — skipping")
        return

    colors_sg = ['#4393c3', '#92c5de', '#f4a582', '#d6604d']

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = range(len(valid))
    bars = ax.bar(x, valid['AUC'].values,
                  color=[colors_sg[i % len(colors_sg)] for i in range(len(valid))],
                  edgecolor='#333', linewidth=0.8, width=0.6)

    for i, (_, row) in enumerate(valid.iterrows()):
        ax.text(i, row['AUC'] + 0.015,
                f"AUC = {row['AUC']:.3f}",
                ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax.set_xticks(list(x))
    ax.set_xticklabels(valid['Severity'].values, fontsize=7)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model Performance by KMax Severity')
    ax.set_ylim([0, min(1.05, valid['AUC'].max() + 0.15)])
    ax.axhline(0.5, color='grey', ls=':', lw=0.8, alpha=0.5, label='Random')
    ax.legend(fontsize=5)

    sg_df.to_csv(out_dir / 'subgroup_analysis.csv', index=False)
    save_fig(fig, out_dir, 'fig_subgroup_kmax')


# ═══════════════════════════════════════════════════════════════════════════
#  15. Image Modality Contribution
# ═══════════════════════════════════════════════════════════════════════════

def plot_modality_contribution(fold_xgb_models, config, results_dir, out_dir):
    """Analyze contribution of each image modality via XGBoost importance."""
    import joblib

    n_folds = config.get('n_cv_folds', 5)
    n_select = config.get('n_select_features', 30)
    image_types = IMAGE_TYPES  # ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
    n_modalities = len(image_types)

    # Collect importance across folds
    modality_importance = {m: [] for m in image_types}
    modality_importance['numeric'] = []

    for fold in range(1, n_folds + 1):
        models_dir = Path(results_dir) / 'models' / f'fold_{fold}'
        xgb_path = models_dir / 'xgb_model.json'
        selector_path = models_dir / 'selector.joblib'

        if not xgb_path.exists():
            continue

        import xgboost as xgb_lib
        xgb_model = xgb_lib.XGBClassifier()
        xgb_model.load_model(str(xgb_path))
        importances = xgb_model.feature_importances_

        # Determine how many features are CNN vs numeric
        selector = joblib.load(selector_path) if selector_path.exists() else None

        if selector is not None:
            support_mask = selector.get_support()  # Boolean mask over all CNN features
            n_cnn_total = len(support_mask)
            n_cnn_selected = support_mask.sum()
        else:
            n_cnn_selected = len(importances)  # fallback
            n_cnn_total = n_cnn_selected
            support_mask = np.ones(n_cnn_total, dtype=bool)

        n_numeric = len(importances) - n_cnn_selected
        features_per_modality = n_cnn_total // n_modalities

        # Map selected CNN features back to modalities
        selected_indices = np.where(support_mask)[0]
        cnn_importances = importances[:n_cnn_selected]
        num_importance_total = importances[n_cnn_selected:].sum()

        per_modality = {m: 0.0 for m in image_types}
        for feat_i, orig_idx in enumerate(selected_indices):
            if feat_i < len(cnn_importances):
                mod_idx = min(orig_idx // features_per_modality, n_modalities - 1)
                mod_name = image_types[mod_idx]
                per_modality[mod_name] += cnn_importances[feat_i]

        for m in image_types:
            modality_importance[m].append(per_modality[m])
        modality_importance['numeric'].append(num_importance_total)

    if not modality_importance['numeric']:
        print("  ⚠ Could not compute modality contributions")
        return

    # Average across folds
    labels_pretty = {
        'corneal_thickness': 'Corneal\nThickness',
        'curvature_front': 'Curvature\nFront',
        'elevation_front': 'Elevation\nFront',
        'elevation_back': 'Elevation\nBack',
        'numeric': 'Clinical\nMetrics'
    }

    avg_imp = {k: np.mean(v) for k, v in modality_importance.items()}
    total = sum(avg_imp.values())
    pct_imp = {k: v / total * 100 for k, v in avg_imp.items()}

    # Sort by importance
    sorted_items = sorted(pct_imp.items(), key=lambda x: x[1], reverse=True)
    names = [labels_pretty.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors_mod = ['#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', '#fff9c4']

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={'width_ratios': [3, 1.2]})

    # Bar chart
    bars = axes[0].barh(range(len(names)), values,
                        color=colors_mod[:len(names)], edgecolor='#333', linewidth=0.8)
    for i, v in enumerate(values):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=7, fontweight='bold')
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlabel('Relative Importance (%)')
    axes[0].set_title('Feature Source Importance', fontsize=9)
    axes[0].invert_yaxis()

    # Pie chart: Image vs Numeric
    img_total = sum(v for k, v in pct_imp.items() if k != 'numeric')
    num_total = pct_imp['numeric']
    axes[1].pie([img_total, num_total],
                labels=['Image\nFeatures', 'Clinical\nMetrics'],
                colors=[COLOR_MM, COLOR_BL], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 7})
    axes[1].set_title('Image vs Clinical', fontsize=9)

    plt.tight_layout()
    save_fig(fig, out_dir, 'fig_modality_contribution')

    # Save data
    mod_df = pd.DataFrame({'Source': [k for k, _ in sorted_items],
                           'Importance_pct': values})
    mod_df.to_csv(out_dir / 'modality_contribution.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════
#  16. Exemplar Case Panels
# ═══════════════════════════════════════════════════════════════════════════

def plot_exemplar_cases(oof_df, df_full, config, out_dir):
    """Show corneal map images for exemplar TP, TN, FP, FN cases."""
    from PIL import Image as PILImage

    image_dir = Path(config['image_dir'])
    image_types = IMAGE_TYPES
    type_labels = {
        'corneal_thickness': 'Thickness',
        'curvature_front': 'Curvature',
        'elevation_front': 'Elev. Front',
        'elevation_back': 'Elev. Back'
    }

    # Get best threshold (Youden's J)
    from sklearn.metrics import roc_curve as sk_roc_curve
    fpr, tpr, thresh = sk_roc_curve(oof_df['y_true'], oof_df['y_pred'])
    j_scores = tpr - fpr
    best_t = thresh[np.argmax(j_scores)]

    oof = oof_df.copy()
    oof['y_bin'] = (oof['y_pred'] >= best_t).astype(int)
    oof['correct'] = oof['y_bin'] == oof['y_true']

    # Map ideye → file id
    id_map = df_full.groupby('ideye')['id'].first().to_dict()
    oof['id'] = oof['ideye'].map(id_map)
    oof = oof.dropna(subset=['id'])

    # Select exemplar cases
    cases = {}
    # True Positive — highest confidence correct positive
    tp = oof[(oof['y_true'] == 1) & (oof['y_bin'] == 1)].sort_values('y_pred', ascending=False)
    if len(tp) > 0:
        cases['True Positive\n(Highest Confidence)'] = tp.iloc[0]

    # True Negative — highest confidence correct negative
    tn = oof[(oof['y_true'] == 0) & (oof['y_bin'] == 0)].sort_values('y_pred', ascending=True)
    if len(tn) > 0:
        cases['True Negative\n(Highest Confidence)'] = tn.iloc[0]

    # False Negative — missed CXL with highest CXL probability among FN
    fn = oof[(oof['y_true'] == 1) & (oof['y_bin'] == 0)].sort_values('y_pred', ascending=False)
    if len(fn) > 0:
        cases['False Negative\n(Missed CXL)'] = fn.iloc[0]

    # False Positive — false alarm with lowest probability among FP
    fp = oof[(oof['y_true'] == 0) & (oof['y_bin'] == 1)].sort_values('y_pred', ascending=True)
    if len(fp) > 0:
        cases['False Positive\n(False Alarm)'] = fp.iloc[0]

    if not cases:
        print("  ⚠ No exemplar cases found")
        return

    n_cases = len(cases)
    n_types = len(image_types)
    fig, axes = plt.subplots(n_cases, n_types, figsize=(n_types * 2.2, n_cases * 2.5))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    case_colors = {
        'True Positive\n(Highest Confidence)': '#4caf50',
        'True Negative\n(Highest Confidence)': '#2196f3',
        'False Negative\n(Missed CXL)': '#f44336',
        'False Positive\n(False Alarm)': '#ff9800',
    }

    for row_i, (case_label, case_row) in enumerate(cases.items()):
        patient_id = str(case_row['id'])
        pred_score = case_row['y_pred']
        true_label = 'CXL' if case_row['y_true'] == 1 else 'Normal'
        border_color = case_colors.get(case_label, '#333')

        for col_i, img_type in enumerate(image_types):
            ax = axes[row_i, col_i]
            img_path = image_dir / f"{patient_id}_{img_type}.jpg"
            if not img_path.exists():
                img_path = image_dir / f"{patient_id}_{img_type}.png"

            if img_path.exists():
                img = PILImage.open(img_path).convert('RGB')
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='grey')

            ax.set_xticks([])
            ax.set_yticks([])

            # Border color
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

            # Column headers (first row only)
            if row_i == 0:
                ax.set_title(type_labels.get(img_type, img_type), fontsize=8, fontweight='bold')

        # Row label
        axes[row_i, 0].set_ylabel(
            f"{case_label}\nP={pred_score:.2f} | True={true_label}",
            fontsize=6, fontweight='bold', color=border_color,
            rotation=0, ha='right', va='center', labelpad=80
        )

    fig.suptitle('Exemplar Cases: Corneal Topography Maps', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig_exemplar_cases')


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  New Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_tsne_embeddings(oof_df, df_full, config, results_dir, numeric_features, out_dir):
    """t-SNE visualization of the multimodal embeddings before XGBoost."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df_full.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    ideye_labels = ideye_to_label.values
    
    # We will just compute the embeddings for the best fold to visualize the space
    valid_folds = [f for f in range(1, n_folds+1) if f in oof_df['fold'].values]
    if not valid_folds: return
    
    fold_aucs = [roc_auc_score(oof_df[oof_df['fold']==f]['y_true'], oof_df[oof_df['fold']==f]['y_pred']) for f in valid_folds]
    if not fold_aucs: return
    best_fold = valid_folds[np.argmax(fold_aucs)]
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    image_dir = Path(config['image_dir'])
    
    X, Y = None, None
    for fold, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_labels), 1):
        if fold != best_fold: continue
        models = load_fold_models(fold, Path(config['_results_dir']), device, config)
        if models is None: return
        test_ideyes = set(unique_ideyes[test_idx])
        test_df = df_full[df_full['ideye'].isin(test_ideyes)].copy()
        
        # Safe transform loading
        test_transform = get_image_transform(training=False, size=config.get('image_size', 224))
        ds = KeratoconusDataset(test_df, image_dir, numeric_features,
                                test_transform, IMAGE_TYPES)
        loader = DataLoader(ds, batch_size=config.get('batch_size', 16),
                            collate_fn=collate_keratoconus, num_workers=0)
        embs, nums, ys = [], [], []
        with torch.no_grad():
            from tqdm import tqdm
            for imgs, nm, lb in tqdm(loader, desc="Extracting features for t-SNE", leave=False):
                imgs = {k: v.to(device) for k, v in imgs.items()}
                embs.append(models['cnn'](imgs).cpu().numpy())
                nums.append(nm.numpy())
                ys.append(lb.numpy())
        if not embs: return
        
        emb = np.vstack(embs)
        num = np.vstack(nums)
        emb_proc = models['scaler_cnn'].transform(emb) if models['scaler_cnn'] else emb
        if models['selector']: emb_proc = models['selector'].transform(emb_proc)
        num_proc = models['scaler_num'].transform(num) if models['scaler_num'] else num
        X = np.hstack([emb_proc, num_proc])
        Y = np.concatenate(ys)
        
    if X is None or len(X) < 5: return
    
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap=plt.cm.coolwarm, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    legend_elements = [mpatches.Patch(facecolor=plt.cm.coolwarm(0.0), edgecolor='k', label='Normal'),
                       mpatches.Patch(facecolor=plt.cm.coolwarm(1.0), edgecolor='k', label='CXL')]
    ax.legend(handles=legend_elements, loc='best', fontsize=7)
    
    ax.set_title('t-SNE Latent Space Visualization', fontsize=9, fontweight='bold')
    ax.axis('off')
    save_fig(fig, out_dir, 'fig17_tsne_embeddings')

def plot_error_analysis(oof_df, df_full, out_dir):
    """Misclassification profiling (Clinical metrics of TP, TN, FP, FN)."""
    kmax_candidates = ['KMax Sagittal Front (D)', 'Km F (D):']
    kmax_col = next((c for c in kmax_candidates if c in df_full.columns), None)
    if not kmax_col: return
    
    kmax_map = df_full.groupby('ideye')[kmax_col].first()
    
    pachy_candidates = ['Pachy Min:', 'Pachymetry Min']
    pachy_col = next((c for c in pachy_candidates if c in df_full.columns), None)
    pachy_map = df_full.groupby('ideye')[pachy_col].first() if pachy_col else None

    oof = oof_df.copy()
    oof['kmax'] = oof['ideye'].map(kmax_map)
    if pachy_map is not None:
        oof['pachy'] = oof['ideye'].map(pachy_map)
        
    # Get optimal threshold to define TP/FP/TN/FN
    fpr, tpr, thresh = roc_curve(oof['y_true'], oof['y_pred'])
    best_t = thresh[np.argmax(tpr - fpr)]
    oof['y_bin'] = (oof['y_pred'] >= best_t).astype(int)
    
    def get_outcome(row):
        if row['y_true'] == 1 and row['y_bin'] == 1: return 'TP'
        if row['y_true'] == 0 and row['y_bin'] == 0: return 'TN'
        if row['y_true'] == 0 and row['y_bin'] == 1: return 'FP'
        return 'FN'
        
    oof['outcome'] = oof.apply(get_outcome, axis=1)
    
    fig, axes = plt.subplots(1, 2 if pachy_col else 1, figsize=(6 if pachy_col else 3, 3))
    axes = [axes] if not pachy_col else axes
    
    order = ['TN', 'FP', 'FN', 'TP']
    palette = {'TN': '#4393c3', 'FP': '#ff9800', 'FN': '#f44336', 'TP': '#d6604d'}
    
    sns.violinplot(data=oof, x='outcome', y='kmax', order=order, ax=axes[0], palette=palette, inner='quartile', linewidth=0.8)
    axes[0].set_title('KMax Distribution by Outcome', fontsize=8)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('KMax (D)')
    
    if pachy_col:
        sns.violinplot(data=oof, x='outcome', y='pachy', order=order, ax=axes[1], palette=palette, inner='quartile', linewidth=0.8)
        axes[1].set_title('Pachymetry Distribution by Outcome', fontsize=8)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Min Pachymetry (μm)')
        
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig18_error_profiling')

def plot_demographic_fairness(oof_df, df_full, out_dir):
    """Algorithmic fairness across age and gender subgroups."""
    gender_col = 'Gender:' if 'Gender:' in df_full.columns else ('Sex:' if 'Sex:' in df_full.columns else None)
    import re
    age_col = next((c for c in df_full.columns if re.search(r'(?i)^age', c)), None)
    
    if not gender_col and not age_col: return
    
    oof = oof_df.copy()
    if age_col:
        age_map = df_full.groupby('ideye')[age_col].first()
        oof['age'] = pd.to_numeric(oof['ideye'].map(age_map), errors='coerce')
        oof['age_group'] = pd.cut(oof['age'], bins=[0, 30, 45, 120], labels=['<30', '30-45', '>45'])
    if gender_col:
        gender_map = df_full.groupby('ideye')[gender_col].first()
        oof['gender'] = oof['ideye'].map(gender_map)
        
    results = []
    
    def eval_subgroup(sub, group_name):
        if len(sub) < 10 or len(sub['y_true'].unique()) < 2: return None
        return {'Subgroup': group_name, 'AUC': roc_auc_score(sub['y_true'], sub['y_pred']), 'N': len(sub)}
        
    if gender_col:
        for g in oof['gender'].dropna().unique():
            res = eval_subgroup(oof[oof['gender'] == g], f"Gender: {g}")
            if res: results.append(res)
            
    if age_col:
        for a in ['<30', '30-45', '>45']:
            res = eval_subgroup(oof[oof['age_group'] == a], f"Age: {a}")
            if res: results.append(res)
            
    res_df = pd.DataFrame(results)
    if res_df.empty: return
    
    fig, ax = plt.subplots(figsize=(4, len(res_df)*0.5 + 1.5))
    sns.barplot(data=res_df, y='Subgroup', x='AUC', ax=ax, palette='Blues_d', edgecolor='k')
    for i, (_, row) in enumerate(res_df.iterrows()):
        ax.text(row['AUC'] + 0.02, i, f"{row['AUC']:.3f} (N={int(row['N'])})", va='center', fontsize=7)
    
    ax.set_xlim([0, 1.05])
    ax.axvline(0.5, color='grey', ls=':')
    ax.set_title('Demographic Fairness (AUC)', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig19_demographic_fairness')
    res_df.to_csv(out_dir / 'demographic_fairness.csv', index=False)

def generate_clinical_operating_points(oof_df, out_dir):
    """Generates metrics tightly coupled to clinical triage settings."""
    y_true, y_pred = oof_df['y_true'].values, oof_df['y_pred'].values
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    
    # 1. Screening (High Sensitivity >= 0.95)
    idx_sens = np.where(tpr >= 0.95)[0]
    best_screening_idx = idx_sens[0] if len(idx_sens) > 0 else np.argmax(tpr)
    
    # 2. Balanced (Youden's J)
    best_balanced_idx = np.argmax(tpr - fpr)
    
    # 3. Confirmatory (High Specificity >= 0.95)
    spec = 1 - fpr
    idx_spec = np.where(spec >= 0.95)[0]
    best_confirmatory_idx = idx_spec[-1] if len(idx_spec) > 0 else np.argmax(spec)
    
    modes = [
        ('Screening (High Sens)', best_screening_idx),
        ('Balanced (Optimal F1)', best_balanced_idx),
        ('Confirmatory (High Spec)', best_confirmatory_idx)
    ]
    
    rows = []
    for mode, idx in modes:
        t = thresh[idx]
        y_bin = (y_pred >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
        rows.append({
            'Mode': mode, 'Threshold': t,
            'Sensitivity': tp/(tp+fn) if tp+fn >0 else 0,
            'Specificity': tn/(tn+fp) if tn+fp >0 else 0,
            'F1': f1_score(y_true, y_bin),
            'PPV': tp/(tp+fp) if tp+fp>0 else 0,
            'NPV': tn/(tn+fn) if tn+fn>0 else 0
        })
        
    df_op = pd.DataFrame(rows)
    df_op.to_csv(out_dir / 'clinical_operating_points.csv', index=False)
    with open(out_dir / 'clinical_operating_points.tex', 'w') as f:
        f.write(df_op.to_latex(index=False, float_format='%.3f'))
    print("  ✓ clinical_operating_points (.csv + .tex)")


def generate_evaluation_suite(oof_df, df, fold_xgb_models, config, results_dir, numeric_features, out_dir, is_global=True):
    print(f"\n── Generating figures in {out_dir} ──")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3: ROC
    plot_roc_curves(oof_df, df, out_dir, is_global)
    
    # 4: Metrics vs Threshold
    best_t = plot_metrics_vs_threshold(oof_df, out_dir)
    
    # 6: Calibration
    plot_calibration(oof_df, out_dir)
    
    # 7: Confusion Matrix
    plot_confusion_mat(oof_df, best_t, out_dir)
    
    # 9: Feature Violins
    plot_feature_violins(df, out_dir)
    
    # 10: PR Curve
    plot_pr_curve(oof_df, out_dir, is_global)
    
    # 11: Summary Table
    generate_summary_table(oof_df, best_t, out_dir, is_global)
    
    # 12: Table 1 — Demographics
    generate_table1(df, out_dir)
    
    # 13: Decision Curve Analysis
    plot_decision_curve(oof_df, out_dir)
    
    # 16: Exemplar Case Panels
    plot_exemplar_cases(oof_df, df, config, out_dir)
    
    if is_global:
        # 1 & 2: Static diagrams
        plot_data_flow(config, out_dir)
        
        # New Additions
        plot_tsne_embeddings(oof_df, df, config, results_dir, numeric_features, out_dir)
        plot_error_analysis(oof_df, df, out_dir)
        plot_demographic_fairness(oof_df, df, out_dir)
        generate_clinical_operating_points(oof_df, out_dir)

        plot_pipeline_overview(config, out_dir)
        
        # 8: Training Curves
        plot_training_curves(results_dir, out_dir)
        
        # 14: Subgroup Analysis by KMax
        plot_subgroup_analysis(oof_df, df, out_dir)
        
        # 15: Image Modality Contribution
        plot_modality_contribution(fold_xgb_models, config, results_dir, out_dir)
        
        # 5: SHAP
        print("\n── SHAP Analysis ──")
        plot_shap_analysis(oof_df, fold_xgb_models, config, df, numeric_features, out_dir)
        
        # 5.b: CNN Interpretation
        # print("\n── CNN Feature Interpretations ──")
        # try:
        #     from interpret_cnn_features import run_cnn_interpretation
        #     cnn_out_dir = Path(out_dir) / 'cnn_interpret'
        #     cnn_out_dir.mkdir(parents=True, exist_ok=True)
        #     run_cnn_interpretation(config, results_dir, df, numeric_features, oof_df, fold_xgb_models, cnn_out_dir)
        # except ImportError:
        #     print("  ⚠ Interpret CNN module not found, skipping.")

def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality results')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config = load_config(args.config)
    config['_results_dir'] = str(results_dir)

    out_dir = results_dir / 'publication_figures'
    out_dir.mkdir(exist_ok=True)
    print(f"Output → {out_dir}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\n── Loading data ──")
    import logging
    logger = logging.getLogger('results_gen')
    logger.addHandler(logging.StreamHandler())
    df, numeric_features = load_data(config, logger)

    # OOF Inference
    print("\n── Running out-of-fold inference ──")
    oof_df, fold_xgb_models = run_oof_inference(df, config, results_dir, device, numeric_features)
    print(f"  Total OOF predictions: {len(oof_df)}")

    # Stage Groupings
    kmax_candidates = ['KMax Sagittal Front (D)', 'Km F (D):']
    kmax_col = None
    for c in kmax_candidates:
        if c in df.columns:
            kmax_col = c
            break

    if kmax_col is not None:
        kmax_map = df.groupby('ideye')[kmax_col].first()
        oof_df['kmax'] = oof_df['ideye'].map(kmax_map)
        bins = [0, 46.5, 48, 53, 55]
        labels_stage = ['Stage0', 'Stage1', 'Stage2', 'Stage3']
        oof_df['stage'] = pd.cut(oof_df['kmax'], bins=bins, labels=labels_stage, include_lowest=True)
        df['stage'] = pd.cut(df[kmax_col], bins=bins, labels=labels_stage, include_lowest=True)
    else:
        labels_stage = []
        print("  ⚠ No KMax column found for stage segregation.")

    # 1. Global Metrics
    global_dir = out_dir / 'global'
    generate_evaluation_suite(oof_df, df, fold_xgb_models, config, results_dir, numeric_features, global_dir, is_global=True)
    
    # 2. Per-Stage Metrics
    for stage in labels_stage:
        stage_oof = oof_df[oof_df['stage'] == stage].copy()
        stage_df = df[df['stage'] == stage].copy()
        
        # Only run if there is adequate data to actually predict on
        if len(stage_oof) > 10 and len(stage_oof['y_true'].unique()) > 1:
            stage_dir = out_dir / stage
            print(f"\n======== Running Evaluation for {stage} ========")
            generate_evaluation_suite(stage_oof, stage_df, fold_xgb_models, config, results_dir, numeric_features, stage_dir, is_global=False)

    print(f"\n✅ All figures saved to {out_dir}")


if __name__ == '__main__':
    main()

