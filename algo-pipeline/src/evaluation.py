import pandas as pd
import numpy as np
import logging
import torch
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from .data import KeratoconusDataset, get_image_transform, create_weighted_sampler, collate_keratoconus
from .models import MultiBackboneCNNEncoder, SharedBackboneCNNEncoder, MultimodalClassifier
from .engine import train_end_to_end, train_hybrid_cnn_xgboost, train_hybrid_finetuned
from .visualization import generate_sample_heatmaps
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature', 'no-latex'])
from sklearn.metrics import roc_curve, precision_recall_curve

def train_numeric_baseline(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          config: Dict) -> Dict:
    """Train XGBoost numeric baseline."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    X_train_fit, y_train_fit = X_train_scaled, y_train
    X_val_fit, y_val_fit = X_test_scaled, y_test
    
    class_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        scale_pos_weight=class_weight,
        random_state=config['random_state'],
        eval_metric='auc',
        early_stopping_rounds=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    model.fit(X_train_fit, y_train_fit, eval_set=[(X_val_fit, y_val_fit)], verbose=False)
    
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    
    return {'test_auc': test_auc, 'test_proba': test_proba}

def nested_cv_multimodal(df_data: pd.DataFrame, image_dir: Path,
                        numeric_feature_names: List[str], image_types: List[str],
                        device: torch.device, config: Dict,
                        logger: logging.Logger,
                        wandb_run: Optional[Any] = None) -> Dict:
    """Run nested cross-validation."""
    
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df_data.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    ideye_labels = ideye_to_label.values
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    
    fold_results = []
    oof_records = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_labels), 1):
        logger.info(f"Fold {fold}/{n_folds}")
        
        train_ideyes = set(unique_ideyes[train_idx])
        test_ideyes = set(unique_ideyes[test_idx])
        
        train_df = df_data[df_data['ideye'].isin(train_ideyes)].copy()
        test_df = df_data[df_data['ideye'].isin(test_ideyes)].copy()
        
        # Setup DataLoaders
        img_size = config.get('image_size', 224)
        train_transform = get_image_transform(training=True, use_augmentation=config['use_augmentation'], size=img_size)
        val_transform = get_image_transform(training=False, size=img_size)
        
        train_dataset = KeratoconusDataset(train_df, image_dir, numeric_feature_names, train_transform, image_types)
        test_dataset = KeratoconusDataset(test_df, image_dir, numeric_feature_names, val_transform, image_types)
        
        sampler = create_weighted_sampler(train_dataset.labels.numpy()) if config['use_weighted_sampler'] else None
        
        num_workers = config.get('num_workers', 0)
        pin_memory = config.get('pin_memory', False)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 sampler=sampler, collate_fn=collate_keratoconus,
                                 num_workers=num_workers, pin_memory=pin_memory,
                                 persistent_workers=(num_workers > 0))
        train_dataset_noaug = KeratoconusDataset(train_df, image_dir, numeric_feature_names, val_transform, image_types)
        train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=config['batch_size'], 
                                 shuffle=False, collate_fn=collate_keratoconus,
                                 num_workers=num_workers, pin_memory=pin_memory,
                                 persistent_workers=(num_workers > 0))
        
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                collate_fn=collate_keratoconus,
                                num_workers=num_workers, pin_memory=pin_memory,
                                persistent_workers=(num_workers > 0))
        
        # Setup Model
        use_shared_backbone = config.get('shared_backbone', False)
        EncoderClass = SharedBackboneCNNEncoder if use_shared_backbone else MultiBackboneCNNEncoder
        
        cnn_encoder = EncoderClass(
            image_types=image_types,
            backbone_name=config['backbone'],
            freeze_mode=config['freeze_mode'],
            use_attention=config.get('use_attention', True),
            use_cross_modal_fusion=config.get('use_cross_modal_fusion', True),
            adapter_mode=config.get('adapter_mode', 'none')
        ).to(device)
        
        bottleneck_dim = config.get('bottleneck_dim', 128)
        model = MultimodalClassifier(
            cnn_encoder=cnn_encoder,
            num_numeric_features=len(numeric_feature_names),
            dropout=config['dropout'],
            bottleneck_dim=bottleneck_dim
        ).to(device)
        
        # Train Multimodal
        if config.get('training_mode') == 'hybrid_cnn_xgboost':
            logger.info("Training in HYBRID mode (Frozen CNN + XGBoost)")
            result = train_hybrid_cnn_xgboost(train_loader_noaug, test_loader, cnn_encoder, device, config, logger, wandb_run)
        elif config.get('training_mode') == 'hybrid_finetuned':
            logger.info("Training in HYBRID FINETUNED mode (Fine-tune CNN + XGBoost)")
            result = train_hybrid_finetuned(train_loader, train_loader_noaug, test_loader, cnn_encoder, device, config, logger, wandb_run)
        else:
            logger.info("Training in END-TO-END mode")
            result = train_end_to_end(train_loader, test_loader, model, device, config, logger, wandb_run)
        
        # Train Baseline
        logger.info("Training numeric baseline...")
        X_train_num = train_df[numeric_feature_names].values
        y_train_num = train_df['y'].values
        X_test_num = test_df[numeric_feature_names].values
        y_test_num = test_df['y'].values
        
        baseline_result = train_numeric_baseline(X_train_num, y_train_num, X_test_num, y_test_num, config)
        
        fold_results.append({
            'multimodal': result,
            'baseline': baseline_result
        })
        
        # Save OOF predictions for this fold
        if 'test_proba' in result and 'test_labels' in result:
            test_ideyes_list = list(test_ideyes)
            test_df_ordered = test_df.set_index('ideye').loc[test_df['ideye'].values]
            for i, (_, row) in enumerate(test_df.iterrows()):
                if i < len(result['test_proba']):
                    oof_records.append({
                        'ideye': row['ideye'],
                        'y_true': int(result['test_labels'][i]),
                        'y_pred': float(result['test_proba'][i]),
                        'fold': fold
                    })
        
        logger.info(f"Fold {fold} Results - Multimodal AUC: {result['best_val_auc']:.4f}, Baseline AUC: {baseline_result['test_auc']:.4f}")

        # Save models
        if config.get('save_models', True):
            models_dir = os.path.join(config['output_dir'], 'models', f'fold_{fold}')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save CNN Encoder
            torch.save(cnn_encoder.state_dict(), os.path.join(models_dir, 'cnn_encoder.pth'))
            
            # Save XGBoost Model
            if 'xgb_model' in result and result['xgb_model'] is not None:
                result['xgb_model'].save_model(os.path.join(models_dir, 'xgb_model.json'))
                
            # Save Transformers
            import joblib
            if 'scaler_cnn' in result and result['scaler_cnn'] is not None:
                joblib.dump(result['scaler_cnn'], os.path.join(models_dir, 'scaler_cnn.joblib'))
            if 'scaler_num' in result and result['scaler_num'] is not None:
                joblib.dump(result['scaler_num'], os.path.join(models_dir, 'scaler_num.joblib'))
            if 'selector' in result and result['selector'] is not None:
                joblib.dump(result['selector'], os.path.join(models_dir, 'selector.joblib'))
            
            logger.info(f"Saved models to {models_dir}")
        
        if wandb_run:
            wandb_run.log({
                f'fold_{fold}_multimodal_auc': result['best_val_auc'],
                f'fold_{fold}_baseline_auc': baseline_result['test_auc'],
                f'fold_{fold}_improvement': result['best_val_auc'] - baseline_result['test_auc']
            })
            
            # Log Plots
            # Get predictions and labels
            if 'val_preds' in result: # End-to-end
                preds = result['val_preds']
                labels = result['val_labels']
            elif 'test_proba' in result: # Hybrid
                preds = result['test_proba']
                labels = result.get('test_labels')
                if labels is None:
                    labels = y_test_num
            
            if preds is not None and labels is not None:
                # Baseline data
                baseline_preds = baseline_result['test_proba']
                
                # ROC
                fpr, tpr, _ = roc_curve(labels, preds)
                fpr_bl, tpr_bl, _ = roc_curve(y_test_num, baseline_preds)
                
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, label=f'Multimodal (AUC={result["best_val_auc"]:.4f})', linewidth=2)
                ax_roc.plot(fpr_bl, tpr_bl, label=f'Baseline (AUC={baseline_result["test_auc"]:.4f})', linestyle='--', linewidth=2)
                ax_roc.plot([0, 1], [0, 1], 'k:', alpha=0.5)
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve (Fold {fold})')
                ax_roc.legend(loc='lower right')
                ax_roc.grid(True, alpha=0.3)
                
                # PR
                precision, recall, _ = precision_recall_curve(labels, preds)
                precision_bl, recall_bl, _ = precision_recall_curve(y_test_num, baseline_preds)
                
                fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
                ax_pr.plot(recall, precision, label='Multimodal', linewidth=2)
                ax_pr.plot(recall_bl, precision_bl, label='Baseline', linestyle='--', linewidth=2)
                ax_pr.set_xlabel('Recall')
                ax_pr.set_ylabel('Precision')
                ax_pr.set_title(f'Precision-Recall Curve (Fold {fold})')
                ax_pr.legend()
                ax_pr.grid(True, alpha=0.3)
                
                wandb_run.log({
                    f'fold_{fold}_roc': wandb.Image(fig_roc),
                    f'fold_{fold}_pr': wandb.Image(fig_pr)
                })
                plt.close(fig_roc)
                plt.close(fig_pr)
            
            # Generate CNN feature heatmaps
            if fold == n_folds and config.get('generate_heatmaps', True):
                try:
                    logger.info(f"Generating Grad-CAM heatmaps for Fold {fold}...")
                    
                    # Extract probabilities for confidence sampling
                    probs = None
                    labels = None
                    if 'val_preds' in result:
                        probs = np.array(result['val_preds'])
                        labels = np.array(result['val_labels'])
                    elif 'test_proba' in result:
                        probs = np.array(result['test_proba'])
                        labels = np.array(result.get('test_labels'))
                        
                    generate_sample_heatmaps(
                        encoder=cnn_encoder,
                        test_loader=test_loader,
                        device=device,
                        backbone_name=config['backbone'],
                        n_samples=5,
                        probs=probs,
                        true_labels=labels,
                        wandb_run=wandb_run,
                        logger=logger,
                        output_dir=os.path.join(config['output_dir'], 'heatmaps')
                    )
                except Exception as e:
                    logger.warning(f"Could not generate heatmaps: {e}")
            
    # Save OOF predictions to CSV
    if oof_records:
        oof_df_save = pd.DataFrame(oof_records)
        oof_path = os.path.join(config['output_dir'], 'oof_predictions.csv')
        oof_df_save.to_csv(oof_path, index=False)
        logger.info(f"Saved {len(oof_df_save)} OOF predictions to {oof_path}")
    
    # Calculate AUC statistics with uncertainty
    mm_aucs = np.array([r['multimodal']['best_val_auc'] for r in fold_results])
    bl_aucs = np.array([r['baseline']['test_auc'] for r in fold_results])
    
    avg_auc_mm = np.mean(mm_aucs)
    std_auc_mm = np.std(mm_aucs)
    avg_auc_bl = np.mean(bl_aucs)
    std_auc_bl = np.std(bl_aucs)
    
    # Out-Of-Fold (OOF) Ensemble AUC
    if config.get('use_ensemble', False):
        all_preds = []
        all_labels = []
        for r in fold_results:
            if 'test_proba' in r['multimodal'] and 'test_labels' in r['multimodal']:
                all_preds.append(r['multimodal']['test_proba'])
                all_labels.append(r['multimodal']['test_labels'])
        
        if all_preds:
            oof_preds = np.concatenate(all_preds)
            oof_labels = np.concatenate(all_labels)
            oof_auc = roc_auc_score(oof_labels, oof_preds)
            logger.info(f"OOF Ensemble AUC: {oof_auc:.4f} (aggregated from all folds)")
        else:
            oof_auc = None
            logger.warning("Could not compute OOF ensemble AUC - missing predictions")
    else:
        oof_auc = None
    
    # 95% confidence interval (t-distribution for small samples)
    from scipy import stats
    ci_95_mm = stats.t.ppf(0.975, n_folds - 1) * std_auc_mm / np.sqrt(n_folds)
    ci_95_bl = stats.t.ppf(0.975, n_folds - 1) * std_auc_bl / np.sqrt(n_folds)
    
    logger.info("="*50)
    logger.info(f"FINAL RESULTS ({n_folds}-fold CV)")
    logger.info(f"Multimodal AUC: {avg_auc_mm:.4f} ± {std_auc_mm:.4f} (95% CI: [{avg_auc_mm - ci_95_mm:.4f}, {avg_auc_mm + ci_95_mm:.4f}])")
    logger.info(f"Baseline AUC:   {avg_auc_bl:.4f} ± {std_auc_bl:.4f} (95% CI: [{avg_auc_bl - ci_95_bl:.4f}, {avg_auc_bl + ci_95_bl:.4f}])")
    if oof_auc is not None:
        logger.info(f"OOF Ensemble:   {oof_auc:.4f}")
    logger.info(f"Improvement:    {avg_auc_mm - avg_auc_bl:+.4f}")
    logger.info("="*50)
    
    # Create combined ROC plot for wandb with all folds
    if wandb_run:
        fig_combined, ax = plt.subplots(figsize=(10, 8))
        
        # Colors for folds
        colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
        
        # Collect all ROC curves for mean calculation
        mean_fpr = np.linspace(0, 1, 100)
        tprs_mm = []
        tprs_bl = []
        
        for fold_idx, r in enumerate(fold_results):
            # Get predictions and labels
            if 'val_preds' in r['multimodal']:
                preds = r['multimodal']['val_preds']
                labels = r['multimodal']['val_labels']
            else:
                preds = r['multimodal']['test_proba']
                labels = r['multimodal'].get('test_labels', None)
            
            baseline_preds = r['baseline']['test_proba']
            
            if preds is not None and labels is not None:
                # Multimodal ROC
                fpr, tpr, _ = roc_curve(labels, preds)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs_mm.append(interp_tpr)
                ax.plot(fpr, tpr, color=colors[fold_idx], alpha=0.3, lw=1,
                       label=f'Fold {fold_idx+1} MM (AUC={r["multimodal"]["best_val_auc"]:.3f})')
                
                # Baseline ROC
                fpr_bl, tpr_bl, _ = roc_curve(labels, baseline_preds)
                interp_tpr_bl = np.interp(mean_fpr, fpr_bl, tpr_bl)
                interp_tpr_bl[0] = 0.0
                tprs_bl.append(interp_tpr_bl)
        
        # Plot mean ROC curves
        if tprs_mm:
            mean_tpr_mm = np.mean(tprs_mm, axis=0)
            mean_tpr_mm[-1] = 1.0
            std_tpr_mm = np.std(tprs_mm, axis=0)
            
            ax.plot(mean_fpr, mean_tpr_mm, color='blue', lw=2,
                   label=f'Mean Multimodal (AUC={avg_auc_mm:.3f} ± {std_auc_mm:.3f})')
            ax.fill_between(mean_fpr, 
                           np.maximum(mean_tpr_mm - std_tpr_mm, 0),
                           np.minimum(mean_tpr_mm + std_tpr_mm, 1),
                           color='blue', alpha=0.1)
            
            mean_tpr_bl = np.mean(tprs_bl, axis=0)
            mean_tpr_bl[-1] = 1.0
            std_tpr_bl = np.std(tprs_bl, axis=0)
            
            ax.plot(mean_fpr, mean_tpr_bl, color='red', lw=2, linestyle='--',
                   label=f'Mean Baseline (AUC={avg_auc_bl:.3f} ± {std_auc_bl:.3f})')
            ax.fill_between(mean_fpr,
                           np.maximum(mean_tpr_bl - std_tpr_bl, 0),
                           np.minimum(mean_tpr_bl + std_tpr_bl, 1),
                           color='red', alpha=0.1)
        
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {n_folds}-Fold Cross-Validation', fontsize=14)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        wandb_run.log({
            'final/combined_roc': wandb.Image(fig_combined),
            'final/multimodal_auc_mean': avg_auc_mm,
            'final/multimodal_auc_std': std_auc_mm,
            'final/multimodal_auc_ci95': ci_95_mm,
            'final/baseline_auc_mean': avg_auc_bl,
            'final/baseline_auc_std': std_auc_bl,
            'final/improvement': avg_auc_mm - avg_auc_bl
        })
        plt.close(fig_combined)
    
    return {
        'avg_auc_mm': avg_auc_mm, 
        'std_auc_mm': std_auc_mm,
        'ci_95_mm': ci_95_mm,
        'avg_auc_bl': avg_auc_bl, 
        'std_auc_bl': std_auc_bl,
        'ci_95_bl': ci_95_bl,
        'fold_results': fold_results
    }

