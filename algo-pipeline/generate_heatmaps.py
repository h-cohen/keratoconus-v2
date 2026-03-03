#!/usr/bin/env python3
"""
Post-training heatmap generator (version-agnostic).

Produces 3-row heatmap visualizations for 100+ test samples:
  Row 1: Raw pentacam images
  Row 2: CBAM attention overlays
  Row 3: Grad-CAM heatmaps

Each figure includes numeric feature overlays (age, K, pachy, ISV, IVA, KI).

Usage:
    python generate_heatmaps.py --run_dir results/<run_dir> --config configs/<config>.yaml [--fold 5] [--n_samples 100]
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import logging
import random
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature', 'no-latex'])
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging
from src.data import (
    load_data, KeratoconusDataset, get_image_transform,
    collate_keratoconus, IMAGE_TYPES,
)
from src.models import MultiBackboneCNNEncoder, SharedBackboneCNNEncoder
from src.engine import extract_all_features
from src.visualization import (
    denormalize_image, extract_attention_maps,
    plot_heatmaps_3row_grid, get_target_layer_for_branch, GradCAM,
)


# Key numeric features to display on heatmaps
DISPLAY_FEATURES = [
    'age',
    'K1 F (D):',
    'K2 F (D):',
    'KMax Sagittal Front (D)',
    'Pachy Apex:',
    'Pachy Min:',
    'ISV:',
    'IVA:',
    'KI:',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Post-training heatmap generator')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to results run directory (e.g., results/v13_difficult_cases_20260224_...)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML used for this run')
    parser.add_argument('--fold', type=int, default=None,
                        help='Which fold to generate heatmaps for (default: last fold)')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Minimum number of samples to generate heatmaps for (default: 100)')
    parser.add_argument('--n_tp', type=int, default=15,
                        help='Number of True Positive samples')
    parser.add_argument('--n_fp', type=int, default=10,
                        help='Number of False Positive samples')
    parser.add_argument('--n_fn', type=int, default=15,
                        help='Number of False Negative samples')
    parser.add_argument('--n_tn', type=int, default=10,
                        help='Number of True Negative samples')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    return parser.parse_args()


def get_fold_data(df, config, fold_idx, n_folds):
    """Recreate train/test split for a specific fold using ideye-based stratification."""
    ideye_to_label = df.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    ideye_labels = ideye_to_label.values
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                         random_state=config['random_state'])
    
    for i, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_labels)):
        if i == fold_idx:
            train_ideyes = set(unique_ideyes[train_idx])
            test_ideyes = set(unique_ideyes[test_idx])
            
            train_df = df[df['ideye'].isin(train_ideyes)].copy()
            test_df = df[df['ideye'].isin(test_ideyes)].copy()
            return train_df, test_df
    
    raise ValueError(f"Fold {fold_idx} not found")


def select_samples(probs, true_labels, n_total, n_tp, n_fp, n_fn, n_tn):
    """Select diverse samples: targeted TP/FP/FN/TN + random fill."""
    indices = set()
    sorted_idx = np.argsort(probs)
    pred_labels = (probs > 0.5).astype(int)
    
    tp_pool = [i for i in sorted_idx[::-1]
               if true_labels[i] == 1 and pred_labels[i] == 1]
    indices.update(tp_pool[:n_tp])
    
    fp_pool = [i for i in sorted_idx[::-1]
               if true_labels[i] == 0 and pred_labels[i] == 1]
    indices.update(fp_pool[:n_fp])
    
    fn_pool = [i for i in sorted_idx
               if true_labels[i] == 1 and pred_labels[i] == 0]
    indices.update(fn_pool[:n_fn])
    
    tn_pool = [i for i in sorted_idx
               if true_labels[i] == 0 and pred_labels[i] == 0]
    indices.update(tn_pool[:n_tn])
    
    # Fill remaining with random samples
    remaining = list(set(range(len(probs))) - indices)
    n_random = max(0, n_total - len(indices))
    if n_random > 0 and remaining:
        indices.update(random.sample(remaining, min(n_random, len(remaining))))
    
    return sorted(list(indices))


def generate_gradcam_for_sample(encoder, images, device, backbone_name):
    """Generate Grad-CAM heatmaps for all modalities of a single sample."""
    gradcam_maps = {}
    
    for img_type, img_tensor in images.items():
        # Get the branch encoder
        if hasattr(encoder, 'encoders') and img_type in encoder.encoders:
            branch = encoder.encoders[img_type]
        elif hasattr(encoder, 'shared_encoder'):
            branch = encoder.shared_encoder
        else:
            continue
        
        try:
            target_layer = get_target_layer_for_branch(branch, backbone_name)
            
            # Create simple wrapper for single-branch Grad-CAM
            class BranchWrapper(torch.nn.Module):
                def __init__(self, enc):
                    super().__init__()
                    self.encoder = enc
                    # Simple linear head for gradient flow
                    self.head = torch.nn.Linear(enc.feature_dim, 1)
                
                def forward(self, x):
                    features = self.encoder(x)
                    return self.head(features)
            
            wrapper = BranchWrapper(branch).to(device)
            wrapper.eval()
            
            gradcam = GradCAM(wrapper, target_layer)
            
            img_input = img_tensor.unsqueeze(0).to(device)
            cam = gradcam.generate_cam(img_input, target_class=1)
            
            # Resize to image size
            h, w = img_tensor.shape[-2:]
            cam_resized = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            gradcam_maps[img_type] = cam_resized
            gradcam.cleanup()
            
        except Exception as e:
            logging.getLogger('heatmaps').warning(
                f"Grad-CAM failed for {img_type}: {e}")
    
    return gradcam_maps


def main():
    args = parse_args()
    
    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    
    # Setup
    heatmap_dir = run_dir / 'heatmaps_3row'
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('heatmaps')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(heatmap_dir / 'heatmap_generation.log'),
        ]
    )
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config: {args.config}")
    
    # Load data
    df, numeric_features = load_data(config, logger)
    image_dir = Path(config['image_dir'])
    
    n_folds = config.get('n_cv_folds', 5)
    fold_idx = (args.fold - 1) if args.fold else (n_folds - 1)  # Default to last fold
    logger.info(f"Generating heatmaps for fold {fold_idx + 1}/{n_folds}")
    
    # Get fold data
    train_df, test_df = get_fold_data(df, config, fold_idx, n_folds)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create test dataset/loader
    img_size = config.get('image_size', 224)
    test_transform = get_image_transform(training=False, size=img_size)
    test_dataset = KeratoconusDataset(
        test_df, image_dir, numeric_features, test_transform, IMAGE_TYPES
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.get('batch_size', 32),
        shuffle=False, collate_fn=collate_keratoconus,
        num_workers=config.get('num_workers', 0)
    )
    
    # Load CNN encoder
    use_shared = config.get('shared_backbone', False)
    EncoderClass = SharedBackboneCNNEncoder if use_shared else MultiBackboneCNNEncoder
    
    cnn_encoder = EncoderClass(
        image_types=IMAGE_TYPES,
        backbone_name=config['backbone'],
        freeze_mode=config['freeze_mode'],
        use_attention=config.get('use_attention', True),
        use_cross_modal_fusion=config.get('use_cross_modal_fusion', True),
        adapter_mode=config.get('adapter_mode', 'none'),
    ).to(device)
    
    fold_num = fold_idx + 1
    weights_path = run_dir / f'models/fold_{fold_num}/cnn_encoder.pth'
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        sys.exit(1)
    
    logger.info(f"Loading encoder weights from {weights_path}")
    cnn_encoder.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    cnn_encoder.eval()
    
    # Get predictions via XGBoost
    xgb_path = run_dir / f'models/fold_{fold_num}/xgb_model.json'
    probs = None
    true_labels = test_df['y'].values
    
    if xgb_path.exists():
        import xgboost as xgb
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        
        logger.info("Loading XGBoost model and generating predictions...")
        
        # Extract features
        train_transform_noaug = get_image_transform(training=False, size=img_size)
        train_dataset_noaug = KeratoconusDataset(
            train_df, image_dir, numeric_features, train_transform_noaug, IMAGE_TYPES
        )
        train_loader_noaug = torch.utils.data.DataLoader(
            train_dataset_noaug, batch_size=config.get('batch_size', 32),
            shuffle=False, collate_fn=collate_keratoconus,
            num_workers=config.get('num_workers', 0)
        )
        
        X_train_cnn, X_train_num, y_train = extract_all_features(
            cnn_encoder, train_loader_noaug, device
        )
        X_test_cnn, X_test_num, y_test = extract_all_features(
            cnn_encoder, test_loader, device
        )
        true_labels = y_test
        
        # Try to load saved transformers, or re-fit
        scaler_cnn_path = run_dir / f'models/fold_{fold_num}/scaler_cnn.joblib'
        scaler_num_path = run_dir / f'models/fold_{fold_num}/scaler_num.joblib'
        selector_path = run_dir / f'models/fold_{fold_num}/selector.joblib'
        
        if scaler_cnn_path.exists():
            scaler_cnn = joblib.load(scaler_cnn_path)
            scaler_num = joblib.load(scaler_num_path)
            selector = joblib.load(selector_path) if selector_path.exists() else None
            logger.info("Loaded saved transformers.")
        else:
            logger.info("Re-fitting transformers from training data...")
            scaler_cnn = StandardScaler()
            scaler_cnn.fit(X_train_cnn)
            scaler_num = StandardScaler()
            scaler_num.fit(X_train_num)
            
            n_select = config.get('n_select_features', 50)
            if X_train_cnn.shape[1] > n_select:
                selector = SelectKBest(score_func=f_classif, k=n_select)
                selector.fit(scaler_cnn.transform(X_train_cnn), y_train)
            else:
                selector = None
        
        X_test_cnn_scaled = scaler_cnn.transform(X_test_cnn)
        X_test_num_scaled = scaler_num.transform(X_test_num)
        
        if selector is not None:
            X_test_cnn_sel = selector.transform(X_test_cnn_scaled)
        else:
            X_test_cnn_sel = X_test_cnn_scaled
        
        X_test_combined = np.hstack([X_test_cnn_sel, X_test_num_scaled])
        
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_path))
        probs = xgb_model.predict_proba(X_test_combined.astype(np.float32))[:, 1]
        
        logger.info(f"Predictions: range [{probs.min():.4f}, {probs.max():.4f}]")
    else:
        logger.warning(f"XGBoost model not found at {xgb_path}. Using random sampling only.")
    
    # Select samples
    if probs is not None:
        sample_indices = select_samples(
            probs, true_labels, args.n_samples,
            args.n_tp, args.n_fp, args.n_fn, args.n_tn
        )
    else:
        n = min(args.n_samples, len(test_dataset))
        sample_indices = random.sample(range(len(test_dataset)), n)
    
    logger.info(f"Generating heatmaps for {len(sample_indices)} samples...")
    
    # Build a map from dataset index to original dataframe row for numeric features
    test_df_reset = test_df.reset_index(drop=True)
    
    generated = 0
    for idx in sample_indices:
        try:
            images, numeric, label = test_dataset[idx]
            
            # Get numeric feature info for overlay
            row = test_df_reset.iloc[idx]
            numeric_info = {}
            for feat in DISPLAY_FEATURES:
                if feat in row.index:
                    val = row[feat]
                    if pd.notna(val):
                        numeric_info[feat] = float(val)
            
            # Raw images (denormalized)
            raw_images = {}
            for img_type, img_tensor in images.items():
                raw_images[img_type] = denormalize_image(img_tensor.cpu().numpy())
            
            # Attention maps
            attn_maps = extract_attention_maps(cnn_encoder, images, device)
            
            # Grad-CAM maps
            gc_maps = generate_gradcam_for_sample(
                cnn_encoder, images, device, config['backbone']
            )
            
            # Prediction for this sample
            pred_val = float(probs[idx]) if probs is not None else None
            true_val = int(label.item()) if torch.is_tensor(label) else int(label)
            
            # Generate 3-row figure
            fig = plot_heatmaps_3row_grid(
                raw_images=raw_images,
                attention_maps=attn_maps,
                gradcam_maps=gc_maps,
                sample_id=str(idx),
                prediction=pred_val,
                true_label=true_val,
                numeric_info=numeric_info,
                image_types=IMAGE_TYPES,
            )
            
            if fig:
                # Determine category for filename
                if probs is not None:
                    pred_label = 1 if pred_val > 0.5 else 0
                    if true_val == 1 and pred_label == 1:
                        cat = 'TP'
                    elif true_val == 0 and pred_label == 1:
                        cat = 'FP'
                    elif true_val == 1 and pred_label == 0:
                        cat = 'FN'
                    else:
                        cat = 'TN'
                    save_name = f'heatmap_{cat}_sample_{idx}.png'
                else:
                    save_name = f'heatmap_sample_{idx}.png'
                
                save_path = heatmap_dir / save_name
                fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
                plt.close(fig)
                generated += 1
                
                if generated % 10 == 0:
                    logger.info(f"Progress: {generated}/{len(sample_indices)} heatmaps generated")
        
        except Exception as e:
            logger.warning(f"Failed to generate heatmap for sample {idx}: {e}")
            continue
    
    logger.info(f"Done! Generated {generated} heatmaps in {heatmap_dir}")


if __name__ == '__main__':
    main()
