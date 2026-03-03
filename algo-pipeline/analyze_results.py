import argparse
import logging
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature', 'no-latex'])
import seaborn as sns
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils import setup_logging, load_config
from src.data import load_data, IMAGE_TYPES, get_image_transform, KeratoconusDataset, collate_keratoconus
from src.models import MultiBackboneCNNEncoder, MultimodalClassifier
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze V8 Model Results')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to results directory (e.g., results/v8_...)')
    parser.add_argument('--config', type=str, default='configs/v8_config.yaml', help='Path to config file')
    return parser.parse_args()

def load_models_for_fold(fold, results_dir, device, config, numeric_features):
    models_dir = Path(results_dir) / 'models' / f'fold_{fold}'
    
    # Load CNN Encoder
    cnn_encoder = MultiBackboneCNNEncoder(
        image_types=IMAGE_TYPES,
        backbone_name=config['backbone'],
        freeze_mode=config['freeze_mode'],
        use_attention=config.get('use_attention', True),
        use_cross_modal_fusion=config.get('use_cross_modal_fusion', True),
        adapter_mode=config.get('adapter_mode', 'none')
    ).to(device)
    
    encoder_path = models_dir / 'cnn_encoder.pth'
    if encoder_path.exists():
        cnn_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        cnn_encoder.eval()
    else:
        print(f"Warning: CNN encoder not found for fold {fold}")
        return None, None

    # Load Transformers
    scaler_cnn = None
    scaler_num = None
    selector = None
    
    if (models_dir / 'scaler_cnn.joblib').exists():
        scaler_cnn = joblib.load(models_dir / 'scaler_cnn.joblib')
    if (models_dir / 'scaler_num.joblib').exists():
        scaler_num = joblib.load(models_dir / 'scaler_num.joblib')
    if (models_dir / 'selector.joblib').exists():
        selector = joblib.load(models_dir / 'selector.joblib')

    # Load XGBoost
    import xgboost as xgb
    xgb_model = None
    xgb_path = models_dir / 'xgb_model.json'
    if xgb_path.exists():
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_path))
    
    return cnn_encoder, xgb_model, scaler_cnn, scaler_num, selector

def get_embeddings_and_predictions(loader, cnn_encoder, device):
    all_embeddings = []
    all_numerics = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, numerics, labels) in enumerate(tqdm(loader, desc="Inference")):
            # Move to device
            images = {k: v.to(device) for k, v in images.items()}
            
            # Get CNN embeddings
            features = cnn_encoder(images)
            
            # Convert to numpy
            batch_embeddings = features.cpu().numpy()
            batch_numerics = numerics.numpy() # Keep CPU
            
            all_embeddings.append(batch_embeddings)
            all_numerics.append(batch_numerics)
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_embeddings), np.vstack(all_numerics), np.array(all_labels)

def main():
    args = parse_args()
    
    # 1. Setup
    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(analysis_dir)
    logger.info(f"Analyzing results from: {results_dir}")
    
    config = load_config(args.config)
    # Update image dir if relative
    if not Path(config['image_dir']).is_absolute():
        # Assuming run from pipeline root
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # 2. Data
    df, numeric_features = load_data(config, logger)
    image_dir = Path(config['image_dir'])
    
    # 3. K-Fold Inference (Recreate folds)
    from sklearn.model_selection import StratifiedKFold
    
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    ideye_labels = ideye_to_label.values
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    
    # Storage for OOF analysis
    oof_data = [] # Will store dicts: {ideye, label, pred, fold, features...}
    
    # Feature names
    pachy_col = 'Pachy Min:'
    kmax_col = 'KMax Sagittal Front (D)' 
    
    # Additional K features
    additional_k_features = [
        'Km F (D):', 
        'K2 F (D):', 
        'K1 B (D):', 
        'K2 B (D):', 
        'Astig F (D):',
        'IHD:'
    ]
    
    features_of_interest = [pachy_col, kmax_col] + additional_k_features
    
    # Check if columns exist
    valid_features = []
    for col in features_of_interest:
        if col in df.columns:
            valid_features.append(col)
        else:
            logger.warning(f"Feature '{col}' not found in dataframe")
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_labels), 1):
        logger.info(f"Processing Fold {fold}/{n_folds}...")
        
        # Load Model
        cnn_encoder, xgb_model, scaler_cnn, scaler_num, selector = load_models_for_fold(fold, results_dir, device, config, numeric_features)
        if cnn_encoder is None:
            continue
            
        # Prepare Data
        test_ideyes = set(unique_ideyes[test_idx])
        test_df = df[df['ideye'].isin(test_ideyes)].copy()
        
        val_transform = get_image_transform(training=False, size=config.get('image_size', 224))
        test_dataset = KeratoconusDataset(test_df, image_dir, numeric_features, val_transform, IMAGE_TYPES)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                collate_fn=collate_keratoconus, num_workers=config.get('num_workers', 4))
        
        # Run Inference (Extract raw features)
        embeddings, numerics, labels = get_embeddings_and_predictions(test_loader, cnn_encoder, device)
        
        # Apply Transformers
        if scaler_cnn:
            embeddings_processed = scaler_cnn.transform(embeddings)
        else:
            embeddings_processed = embeddings

        if selector:
            embeddings_processed = selector.transform(embeddings_processed)
            
        if scaler_num:
            numerics_processed = scaler_num.transform(numerics)
        else:
            numerics_processed = numerics
            
        # Combine
        X_combined = np.hstack([embeddings_processed, numerics_processed])
        
        # Predict
        preds = []
        if xgb_model:
            preds = xgb_model.predict_proba(X_combined)[:, 1]
        else:
            preds = np.zeros(len(labels))
        
        # Store Results
        for i, row_idx in enumerate(test_df.index):
            row = test_df.loc[row_idx]
            
            item = {
                'ideye': row['ideye'],
                'y_true': row['y'],
                'y_pred': preds[i] if len(preds) > 0 else 0.5,
                'fold': fold,
                'embedding': embeddings[i],
            }
            
            # Add features of interest
            for feat in valid_features:
                if feat in row:
                    item[feat] = row[feat]
                
            # Add all numeric features for t-SNE
            item['numeric_features'] = row[numeric_features].values.astype(float)
            
            oof_data.append(item)
            
    # 4. Analysis
    logger.info("Generating Analysis Plots...")
    
    results_df = pd.DataFrame(oof_data)
    
    # Filter: Label 1 (Keratoconus)
    positives = results_df[results_df['y_true'] == 1].copy()
    
    # Define TP/FN
    # Threshold 0.5
    positives['classification'] = positives['y_pred'].apply(lambda x: 'TP' if x >= 0.5 else 'FN')
    
    logger.info(f"Positive Samples: {len(positives)}")
    logger.info(f"Breakdown: {positives['classification'].value_counts().to_dict()}")
    
    # Plot 1: Feature Distributions (Boxplots & Histograms)
    features_to_plot = [f for f in [pachy_col, kmax_col] if f in positives.columns]
    
    if features_to_plot:
        fig, axes = plt.subplots(len(features_to_plot), 2, figsize=(15, 5 * len(features_to_plot)))
        if len(features_to_plot) == 1:
            axes = np.array([axes]) # Ensure 2D array
            
        for i, feature in enumerate(features_to_plot):
            # Histplot
            sns.histplot(data=positives, x=feature, hue='classification', kde=True, ax=axes[i, 0], palette='Set2')
            axes[i, 0].set_title(f'{feature} Distribution')
            
            # Boxplot
            sns.boxplot(data=positives, x='classification', y=feature, ax=axes[i, 1], palette='Set2')
            axes[i, 1].set_title(f'{feature} Boxplot')
            
        plt.tight_layout()
        plt.savefig(analysis_dir / 'feature_distributions_TP_FN.png')
        plt.close()
        
        # Scatter Plots: Pachy vs Other Features
        if pachy_col in positives.columns:
            for feat in valid_features:
                if feat == pachy_col:
                    continue
                    
                plt.figure(figsize=(10, 8))
                sns.scatterplot(data=positives, x=pachy_col, y=feat, hue='classification', style='classification', s=100, palette='Set2')
                plt.title(f'{pachy_col} vs {feat} (TP vs FN)')
                
                # Sanitize filename
                safe_feat = feat.replace(':', '').replace(' ', '_').replace('(', '').replace(')', '')
                plt.savefig(analysis_dir / f'scatter_pachy_vs_{safe_feat}.png')
                plt.close()
            
    # Plot 2: Dimensionality Reduction (t-SNE)
    
    # Prepare data arrays
    num_data = np.stack(positives['numeric_features'].values)
    emb_data = np.stack(positives['embedding'].values)
    combined_data = np.hstack([num_data, emb_data])
    
    # Helper for reduction plot
    def plot_dim_reduction(data, title, filename):
        # PCA first to reduce noise/dims if high
        if data.shape[1] > 50:
            pca = PCA(n_components=50)
            data_pca = pca.fit_transform(data)
        else:
            data_pca = data
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)-1))
        # Ensure we have enough samples for perplexity
        
        try:
            tsne_res = tsne.fit_transform(data_pca)
            
            plt.figure(figsize=(10, 8))
            # Create a localized dataframe for seaborn
            plot_df = pd.DataFrame({
                'tsne_1': tsne_res[:, 0],
                'tsne_2': tsne_res[:, 1],
                'classification': positives['classification'].values
            })
            
            sns.scatterplot(data=plot_df, x='tsne_1', y='tsne_2', hue='classification', style='classification', s=100, palette='Set2')
            plt.title(title)
            plt.savefig(analysis_dir / filename)
            plt.close()
        except Exception as e:
            logger.error(f"t-SNE failed for {title}: {e}")

    logger.info("Running t-SNE...")
    plot_dim_reduction(num_data, 't-SNE on Numeric Features (TP vs FN)', 'tsne_numeric.png')
    plot_dim_reduction(emb_data, 't-SNE on CNN Embeddings (TP vs FN)', 'tsne_embeddings.png')
    plot_dim_reduction(combined_data, 't-SNE on Combined Features (TP vs FN)', 'tsne_combined.png')
    
    logger.info("Analysis Complete.")

if __name__ == '__main__':
    main()
