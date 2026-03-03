"""
Script to interpret the meaning of top SHAP CNN features.
Generates heatmaps (Raw, Attention, GradCAM) specifically targeting the activation of each CNN feature.
"""
import argparse
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xgboost as xgb
import shap
import json
import tempfile
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
from src.utils import load_config
from src.data import load_data, IMAGE_TYPES, get_image_transform, KeratoconusDataset, collate_keratoconus
from src.visualization import (
    GradCAM, get_target_layer_for_branch, denormalize_image,
    create_heatmap_overlay, extract_attention_maps
)
from generate_results import load_fold_models

class FeatureTargetFullModel(nn.Module):
    """Wrapper that outputs a specific feature index to be used as target for GradCAM."""
    def __init__(self, encoder, orig_idx):
        super().__init__()
        self.encoder = encoder
        self.orig_idx = orig_idx
        
    def forward(self, x):
        features = self.encoder(x)
        return features[:, self.orig_idx].unsqueeze(1)


def generate_feature_heatmaps(
    branch_encoder: nn.Module,
    backbone_name: str,
    images: dict,
    orig_idx: int,
    device: torch.device
):
    results = {}
    
    for img_type, img_tensor in images.items():
        if img_tensor is None:
            continue
            
        try:
            target_layer = get_target_layer_for_branch(branch_encoder, backbone_name)
            full_model = FeatureTargetFullModel(branch_encoder, orig_idx).to(device)
            gradcam = GradCAM(full_model, target_layer)
            
            img_input = img_tensor.unsqueeze(0).to(device)
            cam = gradcam.generate_cam(img_input, target_class=0)
            
            h, w = img_tensor.shape[-2:]
            cam_resized = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            img_np = denormalize_image(img_tensor.cpu().numpy())
            results[img_type] = (img_np, cam_resized)
            gradcam.cleanup()
            
        except Exception as e:
            print(f"    Warning: Heatmap failed for {img_type}: {e}")
            continue
            
    return results


def plot_feature_grid(
    raw_images: dict,
    attention_maps: dict,
    gradcam_maps: dict,
    sample_id: str,
    true_label: int,
    feat_name: str,
    out_path: str
):
    image_types = list(raw_images.keys())
    n_cols = len(image_types)
    if n_cols == 0:
        return
    
    has_attn = len(attention_maps) > 0
    n_rows = 3 if has_attn else 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
        
    row_labels = ['Raw Image']
    if has_attn:
        row_labels.append('Attention (CBAM)')
    row_labels.append(f'GradCAM\n({feat_name})')
    
    for col_idx, img_type in enumerate(image_types):
        # Raw
        if img_type in raw_images:
            axes[0, col_idx].imshow(raw_images[img_type])
        axes[0, col_idx].set_title(img_type.replace('_', ' ').title(), fontsize=10)
        axes[0, col_idx].axis('off')
        
        current_row = 1
        
        # Attn
        if has_attn:
            if img_type in attention_maps:
                overlay_attn = create_heatmap_overlay(raw_images[img_type], attention_maps[img_type], alpha=0.5)
                axes[current_row, col_idx].imshow(overlay_attn)
            else:
                axes[current_row, col_idx].imshow(raw_images[img_type])
            axes[current_row, col_idx].axis('off')
            current_row += 1
            
        # GradCAM
        if img_type in gradcam_maps:
            overlay_gc = create_heatmap_overlay(raw_images[img_type], gradcam_maps[img_type], alpha=0.5)
            axes[current_row, col_idx].imshow(overlay_gc)
        else:
            import numpy as np
            gray = np.dot(raw_images[img_type][...,:3], [0.2989, 0.5870, 0.1140])
            gray_rgb = np.stack([gray]*3, axis=-1).astype(np.uint8)
            axes[current_row, col_idx].imshow(gray_rgb)
            axes[current_row, col_idx].text(0.5, 0.5, 'Feature Inactive\n(Not Extracted Here)', 
                                            ha='center', va='center', 
                                            transform=axes[current_row, col_idx].transAxes, 
                                            fontsize=8, color='white', fontweight='bold',
                                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))
        axes[current_row, col_idx].axis('off')
        
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
        
    label_str = 'CXL' if true_label == 1 else 'Normal'
    fig.suptitle(f'Sample: {sample_id} | Class: {label_str} | Feature: {feat_name}', fontsize=14, fontweight='bold')
    
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.25])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cbar_ax)
    cbar.set_label('Activation', rotation=270, labelpad=15)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def run_cnn_interpretation(config, results_dir, df, numeric_features, oof_df, fold_xgb_models, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_numeric = len(numeric_features)
    
    best_fold = max(fold_xgb_models.keys(),
                    key=lambda f: roc_auc_score(
                        oof_df[oof_df['fold'] == f]['y_true'],
                        oof_df[oof_df['fold'] == f]['y_pred']))
    
    models = load_fold_models(best_fold, results_dir, device, config)
    if models is None or models['xgb'] is None or models['cnn'] is None:
        print("  ⚠ Could not load required models for CNN Interpretations.")
        return
        
    xgb_model = models['xgb']
    selector = models['selector']
    cnn = models['cnn']
    
    # Get Shap
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    image_dir = Path(config['image_dir'])
    
    # Find test indices for best fold
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
    with torch.no_grad():
        for imgs, nm, lb in loader:
            imgs = {k: v.to(device) for k, v in imgs.items()}
            embs.append(cnn(imgs).cpu().numpy())
            nums.append(nm.numpy())
    emb = np.vstack(embs)
    num = np.vstack(nums)
    emb_proc = models['scaler_cnn'].transform(emb) if models['scaler_cnn'] else emb
    if selector:
        emb_proc = selector.transform(emb_proc)
    num_proc = models['scaler_num'].transform(num) if models['scaler_num'] else num
    X = np.hstack([emb_proc, num_proc])
    
    n_features = X.shape[1]
    n_cnn = n_features - n_numeric
    feature_names = [f'cnn_{i}' for i in range(n_cnn)] + list(numeric_features)
    
    predict_fn = lambda x: xgb_model.predict_proba(x)[:, 1]
    background = shap.kmeans(X, 10)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    sub_idx = np.linspace(0, len(X) - 1, min(len(X), 150), dtype=int)
    shap_values = explainer.shap_values(X[sub_idx])
    
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[-20:][::-1]
    
    top_cnn_features = [(i, feature_names[i]) for i in top_idx if 'cnn_' in feature_names[i]]
    
    if not top_cnn_features:
        print("  ⚠ No CNN features in top 20 SHAP.")
        return
        
    if selector:
        support = selector.get_support(indices=True)
    else:
        support = np.arange(emb.shape[1])
        
    df_pos = test_df[test_df['y'] == 1]
    df_neg = test_df[test_df['y'] == 0]
    
    sample_df = pd.concat([
        df_pos.sample(n=min(30, len(df_pos)), random_state=42),
        df_neg.sample(n=min(30, len(df_neg)), random_state=42)
    ])
    
    sample_ds = KeratoconusDataset(sample_df, image_dir, numeric_features,
                            get_image_transform(training=False, size=config.get('image_size', 224)),
                            IMAGE_TYPES)
                            
    backbone = config['backbone']
    
    for feat_idx, feat_name in top_cnn_features:
        cnn_sel_idx = int(feat_name.split('_')[1])
        orig_idx = support[cnn_sel_idx]
        
        feat_dir = out_dir / feat_name
        feat_dir.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(range(len(sample_ds)), desc=f"    {feat_name}", leave=False):
            (images, _, label_tensor) = sample_ds[i]
            label = int(label_tensor.item()) if torch.is_tensor(label_tensor) else int(label_tensor)
            ideye = sample_df.iloc[i]['ideye']
            
            images_device = {k: v.unsqueeze(0).to(device).requires_grad_(True) for k, v in images.items()}
            attn_maps = extract_attention_maps(cnn, images, device)
            
            full_model = FeatureTargetFullModel(cnn, orig_idx).to(device)
            full_model.eval()
            
            # Forward pass to explicitly capture target layer activations
            activations = {}
            hooks = []
            
            if hasattr(cnn, 'shared_encoder'):
                branch = cnn.shared_encoder
            else:
                branch = cnn.encoders[list(images.keys())[0]] if hasattr(cnn, 'encoders') else cnn
                
            target_layer = get_target_layer_for_branch(branch, backbone)
            raw_acts = []
            def unified_hook(module, inp, out):
                raw_acts.append(out)
            
            h1 = target_layer.register_forward_hook(unified_hook)
            
            score = full_model(images_device).squeeze()
            h1.remove()
            
            # Map raw_acts to specific image_types
            if hasattr(cnn, 'shared_encoder') and hasattr(cnn, 'image_types'):
                for i, t in enumerate(cnn.image_types):
                    if i < len(raw_acts):
                        activations[t] = raw_acts[i]
            else:
                activations[list(images.keys())[0]] = raw_acts[0] if raw_acts else None
            
            raw_imgs = {}
            gc_maps = {}
            
            for img_type, img_tensor in images.items():
                act = activations.get(img_type, None)
                if act is None:
                    continue
                else:
                    try:
                        grads = torch.autograd.grad(score, act, retain_graph=True, allow_unused=True)[0]
                        
                        if grads is None:
                            pass
                        else:
                            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                            cam = torch.sum(weights * act, dim=1).squeeze().detach().cpu().numpy()
                            cam = np.maximum(cam, 0) # ReLU
                            if cam.max() > 0:
                                cam = cam / cam.max()
                                
                            h, w = img_tensor.shape[-2:]
                            cam_resized = F.interpolate(
                                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze().numpy()
                            gc_maps[img_type] = cam_resized
                    except Exception as e:
                        print(f"Error computing GradCAM for {img_type}: {e}")
                
                raw_imgs[img_type] = denormalize_image(img_tensor.numpy())
            
            save_path = str(feat_dir / f"class_{label}_id_{ideye.replace('/', '_')}.png")
            plot_feature_grid(raw_imgs, attn_maps, gc_maps, ideye, label, feat_name, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    results_dir = Path(args.results_dir)
    config['_results_dir'] = str(results_dir)
    
    out_dir = results_dir / 'publication_figures' / 'cnn_interpret'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    import logging
    logger = logging.getLogger('interp')
    df, numeric_features = load_data(config, logger)
    from generate_results import run_oof_inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof_df, fold_xgb_models = run_oof_inference(df, config, results_dir, device, numeric_features)
    
    run_cnn_interpretation(config, results_dir, df, numeric_features, oof_df, fold_xgb_models, out_dir)
