"""
Script to explain the top CNN features from the Frozen + XGBoost model (v15).
Maps CNN features back to their source image modalities, determines their spatial
activation region (central vs peripheral), computes correlations with clinical features,
and generates a publication-ready interpretation table.
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xgboost as xgb
import shap
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).resolve().parent))
from src.utils import load_config
from src.data import load_data, IMAGE_TYPES, get_image_transform, KeratoconusDataset, collate_keratoconus
from src.visualization import get_target_layer_for_branch, denormalize_image, create_heatmap_overlay
from generate_results import load_fold_models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from interpret_cnn_features import FeatureTargetFullModel

def get_feature_modality(feature_name: str, support_indices: np.ndarray, backbone_dim: int = 512) -> str:
    """Map a CNN feature name to its source image modality."""
    if not feature_name.startswith('cnn_'):
        return "Numeric"
    
    idx = int(feature_name.split('_')[1])
    orig_idx = support_indices[idx]
    
    
    modality_idx = orig_idx // backbone_dim
    if modality_idx < len(IMAGE_TYPES):
        return IMAGE_TYPES[modality_idx]
    return "Unknown"

def compute_spatial_stats(cam: np.ndarray) -> str:
    """Determine the spatial location of the activation."""
    if cam.max() == 0:
        return "Inactive"
        
    threshold = cam.max() * 0.5
    y_coords, x_coords = np.where(cam > threshold)
    
    if len(y_coords) == 0:
        return "Diffuse"
        
    cy, cx = np.mean(y_coords), np.mean(x_coords)
    h, w = cam.shape
    
    # Distance from center normalized to 0-1
    center_y, center_x = h / 2, w / 2
    dist = np.sqrt(((cy - center_y) / h)**2 + ((cx - center_x) / w)**2)
    
    if dist < 0.2:
        return "Central"
    elif dist < 0.35:
        return "Paracentral"
    else:
        return "Peripheral"

def get_clinical_correlates(feature_vals: np.ndarray, numeric_df: pd.DataFrame, num_features: list) -> str:
    """Find top correlating clinical features to help interpretation."""
    corrs = []
    for col in num_features:
        if col in numeric_df.columns:
            r, _ = pearsonr(feature_vals, numeric_df[col].values)
            if not np.isnan(r):
                corrs.append((col, r))
                
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_corrs = [f"{c[0]} (r={c[1]:.2f})" for c in corrs[:2] if abs(c[1]) > 0.1]
    
    if not top_corrs:
        return "None strong"
    return ", ".join(top_corrs)

def generate_clinical_name(modality: str, spatial_region: str, top_correlates: str) -> str:
    """Generate a clinical name."""
    if "KMax" in top_correlates or "Km " in top_correlates:
        return "Extreme Steepening"
    elif "Pachy Min" in top_correlates or "Thinnest" in top_correlates:
        return "Focal Thinning"
    elif "ISV" in top_correlates or "IVA" in top_correlates:
        return "Surface Irregularity"
    elif "Prog" in top_correlates:
        return "Thickness Progression"
    else:
        return f"{spatial_region} {modality.split('_')[-1]}".title()

def generate_interpretation(modality: str, spatial_region: str, top_correlates: str) -> str:
    """Generate a reasonable clinical interpretation based on extracted meta-features."""
    modality_clean = modality.replace('_', ' ').title()
    
    interp = f"{modality_clean}"
    
    if spatial_region != "Inactive":
        interp += f" with {spatial_region.lower()} activation."
        
    if "KMax" in top_correlates or "Km" in top_correlates:
        interp += " Correlates with extreme steepening."
    elif "Pachy Min" in top_correlates or "Thinnest" in top_correlates:
        interp += " Associates with focal thinning areas."
    elif "ISV" in top_correlates or "IVA" in top_correlates:
        interp += " Captures surface irregularity indices."
    elif "Elevation" in modality:
        interp += " Detects posterior or anterior ectatic bulging."
        
    return interp


def compute_single_gradcam(cnn, target_layer, orig_idx, images, modality, device):
    """Compute GradCAM for a single sample, returning (raw_img, cam_resized) or None."""
    images_device = {k: v.unsqueeze(0).to(device).requires_grad_(True) for k, v in images.items()}
    
    full_model = FeatureTargetFullModel(cnn, orig_idx).to(device)
    full_model.eval()
    
    raw_acts = []
    def hook(m, inp, out): raw_acts.append(out)
    h1 = target_layer.register_forward_hook(hook)
    
    score = full_model(images_device).squeeze()
    h1.remove()
    
    if not raw_acts:
        return None
    
    modality_idx = IMAGE_TYPES.index(modality)
    if modality_idx >= len(raw_acts):
        return None
    
    act = raw_acts[modality_idx]
    try:
        grads = torch.autograd.grad(score, act, retain_graph=True, allow_unused=True)[0]
        if grads is None:
            return None
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()
        else:
            return None
        
        img_tensor = images[modality]
        h, w = img_tensor.shape[-2:]
        cam_resized = F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        
        raw_img = denormalize_image(img_tensor.numpy())
        return raw_img, cam_resized
    except Exception:
        return None


def generate_per_feature_figures(
    cnn, target_layer, test_df, image_dir, numeric_features, config,
    top_cnn_features, support, shap_values, sub_idx, y_test, feature_names,
    device, out_dir
):
    """Generate 4-sample exemplar figures per CNN feature (2 CXL + 2 Normal)."""
    print("\nGenerating per-feature exemplar figures...")
    
    for rank, (feat_idx, feat_name) in enumerate(top_cnn_features, 1):
        cnn_sel_idx = int(feat_name.split('_')[1])
        orig_idx = support[cnn_sel_idx]
        modality = get_feature_modality(feat_name, support)
        modality_pretty = modality.replace('_', ' ').title()
        
        f_shap = shap_values[:, feat_idx]
        
        cxl_indices = [i for i, idx in enumerate(sub_idx) if y_test[idx] == 1]
        cxl_sorted = sorted(cxl_indices, key=lambda i: f_shap[i], reverse=True)[:2]
        
        normal_indices = [i for i, idx in enumerate(sub_idx) if y_test[idx] == 0]
        normal_sorted = sorted(normal_indices, key=lambda i: f_shap[i], reverse=False)[:2]
        
        selected = []
        for local_i in cxl_sorted:
            df_i = sub_idx[local_i]
            selected.append((df_i, 1, f_shap[local_i]))
        for local_i in normal_sorted:
            df_i = sub_idx[local_i]
            selected.append((df_i, 0, f_shap[local_i]))
        
        if len(selected) < 4:
            print(f"  ⚠ Not enough samples for {feat_name}, skipping figure.")
            continue
        
        # Build figure: 2 rows x 4 cols (raw | overlay for each of 4 patients)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for col, (df_idx, true_label, shap_val) in enumerate(selected):
            row_df = test_df.iloc[[df_idx]]
            sample_ds = KeratoconusDataset(
                row_df, image_dir, numeric_features,
                get_image_transform(training=False, size=config.get('image_size', 224)),
                IMAGE_TYPES
            )
            images, _, _ = sample_ds[0]
            
            result = compute_single_gradcam(cnn, target_layer, orig_idx, images, modality, device)
            
            ideye = row_df.iloc[0]['ideye']
            label_str = "CXL" if true_label == 1 else "Normal"
            
            if result is not None:
                raw_img, cam_resized = result
                overlay = create_heatmap_overlay(raw_img, cam_resized, alpha=0.5)
                
                axes[0, col].imshow(raw_img)
                axes[1, col].imshow(overlay)
            else:
                img_tensor = images[modality]
                raw_img = denormalize_image(img_tensor.numpy())
                axes[0, col].imshow(raw_img)
                axes[1, col].imshow(raw_img)
                axes[1, col].text(0.5, 0.5, 'GradCAM\nUnavailable',
                                  ha='center', va='center',
                                  transform=axes[1, col].transAxes,
                                  fontsize=9, color='white', fontweight='bold',
                                  bbox=dict(facecolor='black', alpha=0.6))
            
            axes[0, col].set_title(f"{label_str}\n{ideye}\nSHAP={shap_val:.4f}", fontsize=9)
            axes[0, col].axis('off')
            axes[1, col].axis('off')
        
        # Row labels
        axes[0, 0].set_ylabel("Raw Image", fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel("GradCAM Overlay", fontsize=12, fontweight='bold')
        
        fig.suptitle(
            f"Rank {rank}: {feat_name} — {modality_pretty}\n"
            f"(2 CXL patients with highest positive SHAP, 2 Normal patients with most negative SHAP)",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        fig_path = out_dir / f"exemplar_rank{rank:02d}_{feat_name}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fig_path.name}")


def run_explanation(config_path, results_path):
    print("Loading configuration...")
    config = load_config(config_path)
    results_dir = Path(results_path)
    config['_results_dir'] = str(results_dir)
    
    out_dir = results_dir / 'publication_figures' / 'cnn_explanations'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    import logging
    logger = logging.getLogger('explain')
    df, numeric_features = load_data(config, logger)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from generate_results import run_oof_inference
    oof_df, fold_xgb_models = run_oof_inference(df, config, results_dir, device, numeric_features)
    
    # Find best fold
    best_fold = max(fold_xgb_models.keys(),
                    key=lambda f: roc_auc_score(
                        oof_df[oof_df['fold'] == f]['y_true'],
                        oof_df[oof_df['fold'] == f]['y_pred']))
    
    print(f"Using best fold {best_fold} for interpretation.")
    models = load_fold_models(best_fold, results_dir, device, config)
    xgb_model = models['xgb']
    selector = models['selector']
    cnn = models['cnn']
    cnn.eval()
    
    # Feature extraction setup
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    image_dir = Path(config['image_dir'])
    
    best_test_idx = None
    for fold, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_to_label.values), 1):
        if fold == best_fold:
             best_test_idx = test_idx
             break
             
    test_ideyes = set(unique_ideyes[best_test_idx])
    test_df = df[df['ideye'].isin(test_ideyes)]
    ds = KeratoconusDataset(test_df, image_dir, numeric_features,
                            get_image_transform(training=False, size=config.get('image_size', 224)),
                            IMAGE_TYPES)
    loader = DataLoader(ds, batch_size=config.get('batch_size', 16), collate_fn=collate_keratoconus, num_workers=2)
    
    embs, nums = [], []
    print("Extracting features for SHAP...")
    with torch.no_grad():
        for imgs, nm, lb in tqdm(loader, desc="Extraction"):
            imgs = {k: v.to(device) for k, v in imgs.items()}
            embs.append(cnn(imgs).cpu().numpy())
            nums.append(nm.numpy())
            
    emb = np.vstack(embs)
    num = np.vstack(nums)
    
    # Apply scaling
    num_proc = models['scaler_num'].transform(num) if models['scaler_num'] else num
    emb_proc = models['scaler_cnn'].transform(emb) if models['scaler_cnn'] else emb
    
    # We need the support indices
    if selector:
        support = selector.get_support(indices=True)
        emb_proc = selector.transform(emb_proc)
    else:
        support = np.arange(emb.shape[1])
        
    X = np.hstack([emb_proc, num_proc])
    
    n_features = X.shape[1]
    n_numeric = len(numeric_features)
    n_cnn = n_features - n_numeric
    feature_names = [f'cnn_{i}' for i in range(n_cnn)] + list(numeric_features)
    
    print("Computing SHAP values...")
    predict_fn = lambda x: xgb_model.predict_proba(x)[:, 1]
    background = shap.kmeans(X, 10)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    sub_idx = np.linspace(0, len(X) - 1, min(len(X), 150), dtype=int)
    shap_values = explainer.shap_values(X[sub_idx])
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs_shap)[-20:][::-1] # Top 20 descending
    
    top_cnn_features = [(i, feature_names[i]) for i in top_idx if 'cnn_' in feature_names[i]]
    print(f"Found {len(top_cnn_features)} CNN features in top 20.")
    
    # Gather explanations
    explanations = []
    y_test = test_df['y'].values
                            
    # Since it's a SharedBackboneCNNEncoder, we can use the shared_encoder directly for GradCAM hook
    backbone_name = config['backbone']
    branch = cnn.shared_encoder if hasattr(cnn, 'shared_encoder') else cnn
    target_layer = get_target_layer_for_branch(branch, backbone_name)
    
    for rank, (feat_idx, feat_name) in enumerate(top_cnn_features, 1):
        cnn_sel_idx = int(feat_name.split('_')[1])
        orig_idx = support[cnn_sel_idx]
        modality = get_feature_modality(feat_name, support)
        
        # Mean difference (Direction)
        feature_vals = X[:, feat_idx]
        mean_cxl = np.mean(feature_vals[y_test == 1])
        mean_normal = np.mean(feature_vals[y_test == 0])
        direction = "Up in CXL" if mean_cxl > mean_normal else "Down in CXL"
        
        # Select samples that highly activate this specific feature in the correct direction
        # Look at the SHAP values (sub_idx maps to the first 150 items of X/test_df)
        f_shap = shap_values[:, feat_idx]
        
        target_label = 1 if direction == "Up in CXL" else 0
        valid_indices = [i for i, idx in enumerate(sub_idx) if y_test[idx] == target_label]
        
        if direction == "Up in CXL":
            best_local_indices = sorted(valid_indices, key=lambda i: f_shap[i], reverse=True)[:10]
        else:
            best_local_indices = sorted(valid_indices, key=lambda i: f_shap[i], reverse=False)[:10]
            
        best_df_indices = [sub_idx[i] for i in best_local_indices]
        sample_df = test_df.iloc[best_df_indices]
        sample_ds = KeratoconusDataset(sample_df, image_dir, numeric_features,
                                get_image_transform(training=False, size=config.get('image_size', 224)),
                                IMAGE_TYPES)
        
        # Calculate GradCAM over samples to find spatial region
        cams = []
        raw_images = []
        for i in range(len(sample_ds)):
            images, _, label = sample_ds[i]
            
            # Forward pass specifically targeting this feature on the shared backbone
            images_device = {k: v.unsqueeze(0).to(device).requires_grad_(True) for k, v in images.items()}
            
            # Use unified hook pattern from interpret_cnn_features.py
            full_model = FeatureTargetFullModel(cnn, orig_idx).to(device)
            full_model.eval()
            
            raw_acts = []
            def hook(m, i, o): raw_acts.append(o)
            h1 = target_layer.register_forward_hook(hook)
            
            score = full_model(images_device).squeeze()
            h1.remove()
            
            if not raw_acts: continue
            
            # In SharedBackbone, it processes image_types in order
            modality_idx = IMAGE_TYPES.index(modality)
            if modality_idx >= len(raw_acts): continue
            
            act = raw_acts[modality_idx]
            try:
                grads = torch.autograd.grad(score, act, retain_graph=True, allow_unused=True)[0]
                if grads is not None:
                    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                    cam = torch.sum(weights * act, dim=1).squeeze().detach().cpu().numpy()
                    cam = np.maximum(cam, 0)
                    if cam.max() > 0:
                        cam /= cam.max()
                        
                        # resizing to raw image scale
                        img_tensor = images[modality]
                        h, w = img_tensor.shape[-2:]
                        cam_resized = F.interpolate(
                            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().numpy()
                        
                        cams.append(cam_resized)
                        raw_images.append(denormalize_image(img_tensor.numpy()))
            except Exception as e:
                pass
                
        # Average CAMs for spatial region
        spatial_region = "Unknown"
        if cams:
            avg_cam = np.mean(cams, axis=0)
            spatial_region = compute_spatial_stats(avg_cam)
            
        # Correlates
        numeric_df = pd.DataFrame(num, columns=numeric_features)
        correlates = get_clinical_correlates(feature_vals, numeric_df, numeric_features)
        
        interp = generate_interpretation(modality, spatial_region, correlates)
        clin_name = generate_clinical_name(modality, spatial_region, correlates)
        
        explanations.append({
            'Rank': rank,
            'Feature': feat_name,
            'Clinical Name': clin_name,
            'SHAP Importance': f"{mean_abs_shap[feat_idx]:.4f}",
            'Source Modality': modality.replace('_', ' ').title(),
            'Original Channel': orig_idx % 512,
            'Spatial Region': spatial_region,
            'Direction': direction,
            'Top Correlates': correlates,
            'Clinical Interpretation': interp
        })
        
        # Keep averge cam for composite figure
        if cams and rank <= 10:
            avg_cam = np.mean(cams, axis=0)
            avg_raw = np.mean(raw_images, axis=0).astype(np.uint8) if raw_images else None
            explanations[-1]['avg_cam'] = avg_cam
            explanations[-1]['avg_raw'] = avg_raw
        
    explanations_df = pd.DataFrame(explanations)
    
    # Create Composite Figure for top 10 CNN features
    top_10 = [e for e in explanations if 'avg_cam' in e]
    if top_10:
        fig_summary, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for idx, item in enumerate(top_10):
            if idx >= 10: break
            ax = axes[idx]
            cam = item['avg_cam']
            raw = item['avg_raw']
            # Plot raw heatmap overlaid!
            if raw is not None:
                overlay = create_heatmap_overlay(raw, cam, alpha=0.5)
                im = ax.imshow(overlay)
            else:
                im = ax.imshow(cam, cmap='jet')
            ax.set_title(f"Rank {item['Rank']}: {item['Source Modality']}\\n({item['Feature']})\\n[{item['Clinical Name']}]", fontsize=10)
            ax.axis('off')
        # Turn off remaining axes
        for idx in range(len(top_10), 10):
            axes[idx].axis('off')
            
        fig_summary.suptitle("Average GradCAM Activations for Top 10 CNN Features", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_summary.savefig(out_dir / 'fig_summary_top10_cnn_features.png', dpi=300, bbox_inches='tight')
        plt.close(fig_summary)
        print(f"Saved composite summary figure.")
        
    # Generate per-feature exemplar figures (2 CXL + 2 Normal per feature)
    generate_per_feature_figures(
        cnn, target_layer, test_df, image_dir, numeric_features, config,
        top_cnn_features, support, shap_values, sub_idx, y_test, feature_names,
        device, out_dir
    )
        
    # Remove avg_cam from dataframe to save to CSV cleanly
    for col_to_remove in ['avg_cam', 'avg_raw']:
        if col_to_remove in explanations_df.columns:
            explanations_df = explanations_df.drop(columns=[col_to_remove])
            
    # Save CSV
    csv_path = out_dir / 'cnn_feature_explanations.csv'
    explanations_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Save LaTeX table
    tex_path = out_dir / 'cnn_feature_explanations.tex'
    tex_str = explanations_df.to_latex(index=False, caption="Interpretation of Top CNN Features", label="tab:cnn_features")
    with open(tex_path, 'w') as f:
        f.write(tex_str)
    print(f"Saved LaTeX tablet to {tex_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    run_explanation(args.config, args.results_dir)
