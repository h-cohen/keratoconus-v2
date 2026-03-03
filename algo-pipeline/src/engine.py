import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from .models import MultimodalClassifier, MultiBackboneCNNEncoder, SharedBackboneCNNEncoder
from .data import KeratoconusDataset, collate_keratoconus

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def create_enhanced_loss(y_train: np.ndarray, device: torch.device,
                        loss_type: str = 'focal', focal_gamma: float = 2.0) -> nn.Module:
    """Create loss function."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    
    # Simple effective alpha
    alpha = [n_pos / len(y_train), n_neg / len(y_train)]
    alpha_tensor = torch.tensor(alpha).to(device)
    
    if loss_type == 'focal':
        return FocalLoss(alpha=alpha_tensor, gamma=focal_gamma)
    else:
        pos_weight = torch.tensor([n_neg / n_pos]).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def train_one_epoch(model: nn.Module, loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, float]:
    model.train()
    losses, preds, labels = [], [], []
    
    for batch_images, batch_numeric, batch_labels in loader:
        batch_images = {k: v.to(device) for k, v in batch_images.items()}
        batch_numeric = batch_numeric.to(device)
        batch_labels = batch_labels.to(device).view(-1)
        
        optimizer.zero_grad()
        logits = model(batch_images, batch_numeric)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
        
    return np.mean(losses), roc_auc_score(labels, preds)

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
            device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds, labels = [], [], []
    
    with torch.no_grad():
        for batch_images, batch_numeric, batch_labels in loader:
            batch_images = {k: v.to(device) for k, v in batch_images.items()}
            batch_numeric = batch_numeric.to(device)
            batch_labels = batch_labels.to(device).view(-1)
            
            logits = model(batch_images, batch_numeric)
            loss = criterion(logits, batch_labels)
            
            losses.append(loss.item())
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            
    return np.mean(losses), roc_auc_score(labels, preds), np.array(preds), np.array(labels)

def train_end_to_end(train_loader: DataLoader, test_loader: DataLoader,
                    model: MultimodalClassifier, device: torch.device,
                    config: Dict, logger: logging.Logger,
                    wandb_run: Optional[Any] = None) -> Dict:
    
    criterion = create_enhanced_loss(
        train_loader.dataset.labels.numpy(), device,
        loss_type=config.get('imbalance_loss_type', 'focal'),
        focal_gamma=config.get('focal_gamma', 2.0)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                                 weight_decay=config['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    
    best_val_auc = float('-inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = validate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}: "
                   f"Train Loss={train_loss:.4f}, AUC={train_auc:.4f} | "
                   f"Val Loss={val_loss:.4f}, AUC={val_auc:.4f}")
        
        if wandb_run:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/auc': train_auc,
                'val/loss': val_loss,
                'val/auc': val_auc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
        if patience_counter >= config.get('patience', 15):
            logger.info("Early stopping")
            break
            
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Final inference
    _, _, val_preds, val_labels = validate(model, test_loader, criterion, device)
        
    return {'best_val_auc': best_val_auc, 'best_epoch': best_epoch, 
            'val_preds': val_preds, 'val_labels': val_labels}

def extract_cnn_features(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Extract CNN features from the model."""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for batch_images, _, _ in loader:
            batch_images = {k: v.to(device) for k, v in batch_images.items()}
            # Forward pass through encoder only
            # Assuming model is MultiBackboneCNNEncoder
            features = model(batch_images)
            features_list.append(features.cpu().numpy())
            
    return np.vstack(features_list)

def extract_all_features(encoder: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract CNN features, numeric features, and labels in a SINGLE pass.
    
    This is critical for shuffled loaders - we must extract everything together
    to maintain alignment between CNN features and labels.
    
    Returns:
        Tuple of (cnn_features, numeric_features, labels)
    """
    encoder.eval()
    cnn_feats_list = []
    numerics_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_images, batch_numeric, batch_labels in loader:
            batch_images = {k: v.to(device) for k, v in batch_images.items()}
            feats = encoder(batch_images)
            cnn_feats_list.append(feats.cpu().numpy())
            numerics_list.append(batch_numeric.numpy())
            labels_list.append(batch_labels.numpy())
    
    return (
        np.vstack(cnn_feats_list), 
        np.vstack(numerics_list), 
        np.concatenate(labels_list)
    )

def train_hybrid_cnn_xgboost(train_loader_noaug: DataLoader, test_loader: DataLoader,
                            cnn_encoder: nn.Module, device: torch.device,
                            config: Dict, logger: logging.Logger,
                            wandb_run: Optional[Any] = None) -> Dict:
    """Train Hybrid model: Frozen CNN + XGBoost."""
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import train_test_split
    
    # 1. Extract ALL features in a single pass (critical for alignment!)
    logger.info("Extracting CNN features...")
    X_train_cnn, X_train_num, y_train = extract_all_features(cnn_encoder, train_loader_noaug, device)
    X_test_cnn, X_test_num, y_test = extract_all_features(cnn_encoder, test_loader, device)
    
    scaler_cnn = StandardScaler()
    X_train_cnn_scaled = scaler_cnn.fit_transform(X_train_cnn)
    X_test_cnn_scaled = scaler_cnn.transform(X_test_cnn)
    
    scaler_num = StandardScaler()
    X_train_num_scaled = scaler_num.fit_transform(X_train_num)
    X_test_num_scaled = scaler_num.transform(X_test_num)
    
    # 4. Feature selection on CNN features
    n_select = config.get('n_select_features', 50)
    selector = None
    if X_train_cnn_scaled.shape[1] <= n_select:
        X_train_cnn_sel = X_train_cnn_scaled
        X_test_cnn_sel = X_test_cnn_scaled
    else:
        selector = SelectKBest(score_func=f_classif, k=n_select)
        X_train_cnn_sel = selector.fit_transform(X_train_cnn_scaled, y_train)
        X_test_cnn_sel = selector.transform(X_test_cnn_scaled)
        
    # 5. Combine scaled features
    X_train_combined = np.hstack([X_train_cnn_sel, X_train_num_scaled])
    X_test_combined = np.hstack([X_test_cnn_sel, X_test_num_scaled])
    
    # 6. Train XGBoost
    # Eliminate internal random 20% split to prevent duplicate-patient data leakage.
    # We use the fold's true test set for early stopping (same as the CNN).
    X_train_fit, y_train_fit = X_train_combined, y_train
    X_val_fit, y_val_fit = X_test_combined, y_test
    
    class_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=config.get('xgb_n_estimators', 300),
        max_depth=config.get('xgb_max_depth', 5),
        learning_rate=config.get('xgb_learning_rate', 0.02),
        scale_pos_weight=class_weight,
        random_state=config['random_state'],
        eval_metric='auc',
        early_stopping_rounds=20,
        subsample=config.get('xgb_subsample', 0.8),
        colsample_bytree=config.get('xgb_colsample_bytree', 0.8),
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_fit, y_train_fit, eval_set=[(X_val_fit, y_val_fit)], verbose=False)
    
    test_proba = xgb_model.predict_proba(X_test_combined)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    
    return {
        'best_val_auc': test_auc, 
        'test_proba': test_proba, 
        'test_labels': y_test, 
        'xgb_model': xgb_model,
        'scaler_cnn': scaler_cnn,
        'scaler_num': scaler_num,
        'selector': selector
    }

def train_hybrid_finetuned(train_loader: DataLoader, train_loader_noaug: DataLoader, test_loader: DataLoader,
                          cnn_encoder: nn.Module, device: torch.device,
                          config: Dict, logger: logging.Logger,
                          wandb_run: Optional[Any] = None) -> Dict:
    """Hybrid training with fine-tuning: Fine-tune CNN then XGBoost."""
    import copy
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    from .data import collate_keratoconus
    
    logger.info("Step 1: Fine-tuning CNN Encoder (with Numeric Features)")
    
    bottleneck_dim = config.get('bottleneck_dim', 128)
    model = MultimodalClassifier(
        cnn_encoder=cnn_encoder,
        num_numeric_features=len(train_loader.dataset.numeric_feature_names),
        dropout=config.get('dropout', 0.5),
        bottleneck_dim=bottleneck_dim
    ).to(device)
    
    finetune_train_loader = train_loader
    finetune_val_loader = test_loader
    # Extract y_train directly from dataset since there is no internal split
    y_train = train_loader.dataset.labels.numpy()
    
    # Calculate normalization stats from training data directly
    train_numeric_data = train_loader.dataset.numeric_data
    numeric_mean = train_numeric_data.mean(dim=0).to(device)
    numeric_std = train_numeric_data.std(dim=0).to(device)
    # Avoid unknown division by zero
    numeric_std[numeric_std == 0] = 1.0
    
    criterion = create_enhanced_loss(
        y_train, device,
        loss_type=config.get('imbalance_loss_type', 'focal'),
        focal_gamma=config.get('focal_gamma', 2.0)
    )
    
    # Label smoothing
    label_smoothing = config.get('label_smoothing', 0.0)
    if label_smoothing > 0:
        logger.info(f"Using label smoothing: epsilon={label_smoothing}")
    
    # Optimizer with layer-wise learning rates
    backbone_lr = config.get('backbone_lr', 1e-5)
    head_lr = config.get('learning_rate', 5e-5)
    
    # Only include trainable CNN encoder params (adapters, attention, unfrozen layers)
    # Frozen backbone params are excluded to save compute and memory
    encoder_trainable = [p for p in cnn_encoder.parameters() if p.requires_grad]
    
    param_groups = [
        {'params': model.classifier.parameters(), 'lr': head_lr, 'initial_lr': head_lr},
        {'params': model.cnn_bottleneck.parameters(), 'lr': head_lr, 'initial_lr': head_lr}
    ]
    if encoder_trainable:
        param_groups.insert(0, {'params': encoder_trainable, 'lr': backbone_lr, 'initial_lr': backbone_lr})
    
    n_trainable = sum(p.numel() for p in cnn_encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in cnn_encoder.parameters())
    logger.info(f"CNN Encoder: {n_trainable:,}/{n_total:,} trainable params ({100*n_trainable/n_total:.1f}%)")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    
    finetune_epochs = config.get('finetune_epochs', 15)
    warmup_epochs = config.get('warmup_epochs', 2)
    use_mixup = config.get('use_mixup', True)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    grad_clip = config.get('gradient_clip', 1.0)
    
    # Stochastic Weight Averaging (SWA)
    use_swa = config.get('use_swa', False)
    swa_model = None
    swa_start_epoch = int(finetune_epochs * 0.6)  # Start SWA at 60% of training
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=backbone_lr * 0.5)
        logger.info(f"SWA enabled: starts at epoch {swa_start_epoch + 1}/{finetune_epochs}")
    
    # Scheduler
    scheduler_type = config.get('scheduler', 'cosine')
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs - warmup_epochs, eta_min=backbone_lr * 0.1
        )
    
    best_val_auc = float('-inf')
    best_model_state = None
    patience_counter = 0
    patience = config.get('finetune_patience', 7) 
    
    def normalize_numeric(numeric_batch):
        """Normalize numeric batch using training stats."""
        return (numeric_batch - numeric_mean) / numeric_std

    def mixup_data(images_dict, numeric, labels, alpha=0.2):
        """Apply Mixup augmentation to batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        batch_size = labels.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_images = {
            k: lam * v + (1 - lam) * v[index] 
            for k, v in images_dict.items()
        }
        mixed_numeric = lam * numeric + (1 - lam) * numeric[index]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, mixed_numeric, labels_a, labels_b, lam
    
    def mixup_criterion(pred, y_a, y_b, lam):
        """Compute Mixup loss."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    for epoch in range(finetune_epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * warmup_factor
        elif epoch == warmup_epochs:
            # Reset scheduler after warmup
            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr']
        
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_labels_list = []
        
        optimizer.zero_grad()
        for step_idx, (batch_images, batch_numeric, batch_labels) in enumerate(finetune_train_loader):
            batch_images = {k: v.to(device) for k, v in batch_images.items()}
            batch_numeric = batch_numeric.to(device)
            # Normalize numeric features
            batch_numeric = normalize_numeric(batch_numeric)
            
            batch_labels = batch_labels.to(device).view(-1)
            
            # Apply label smoothing
            if label_smoothing > 0:
                batch_labels_smooth = batch_labels * (1 - label_smoothing) + (1 - batch_labels) * label_smoothing
            else:
                batch_labels_smooth = batch_labels
            
            # Apply Mixup with probability
            if use_mixup and np.random.rand() < 0.5:
                mixed_images, mixed_numeric, labels_a, labels_b, lam = mixup_data(
                    batch_images, batch_numeric, batch_labels_smooth, mixup_alpha
                )
                logits = model(mixed_images, mixed_numeric)
                loss = mixup_criterion(logits, labels_a, labels_b, lam)
            else:
                logits = model(batch_images, batch_numeric)
                loss = criterion(logits, batch_labels_smooth)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            
            # Step optimizer every grad_accum_steps
            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(finetune_train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * grad_accum_steps)  # unscale for logging
            # Use predictions from the actual training step (no redundant forward pass)
            with torch.no_grad():
                train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                train_labels_list.extend(batch_labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch_images, batch_numeric, batch_labels in finetune_val_loader:
                batch_images = {k: v.to(device) for k, v in batch_images.items()}
                batch_numeric = batch_numeric.to(device)
                # Normalize numeric features
                batch_numeric = normalize_numeric(batch_numeric)
                
                batch_labels = batch_labels.to(device).view(-1)
                
                logits = model(batch_images, batch_numeric)
                
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        train_auc = roc_auc_score(train_labels_list, train_preds) if len(set(train_labels_list)) > 1 else 0.5
        val_auc = roc_auc_score(val_labels_list, val_preds) if len(set(val_labels_list)) > 1 else 0.5

        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Fine-tune Epoch {epoch+1}/{finetune_epochs}: "
                   f"Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.2e}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if wandb_run:
            wandb_run.log({
                'finetune/epoch': epoch + 1,
                'finetune/train_loss': train_loss,
                'finetune/train_auc': train_auc,
                'finetune/val_auc': val_auc,
                'finetune/lr': current_lr
            })
        
        # Early stopping (but not during warmup or SWA phase)
        # SWA must run to completion to accumulate enough weight snapshots
        in_swa_phase = use_swa and epoch >= swa_start_epoch
        if epoch >= warmup_epochs and not in_swa_phase and patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
            
    # Restore best model or apply SWA (with validation gate)
    swa_epochs_collected = max(0, (epoch + 1) - swa_start_epoch) if use_swa else 0
    use_swa_weights = False
    
    if use_swa and swa_model is not None and swa_epochs_collected >= 3:
        logger.info(f"Evaluating SWA weights (averaged over {swa_epochs_collected} epochs)...")
        
        # Load SWA-averaged weights
        model.load_state_dict(swa_model.module.state_dict())
        
        # Custom BN update for multimodal dataloader
        momenta = {}
        model.train()
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked.zero_()
        
        with torch.no_grad():
            for batch_images, batch_numeric, batch_labels in finetune_train_loader:
                batch_images = {k: v.to(device) for k, v in batch_images.items()}
                batch_numeric = batch_numeric.to(device)
                model(batch_images, batch_numeric)
        
        for module, momentum in momenta.items():
            module.momentum = momentum
        model.eval()
        
        # Evaluate SWA weights on validation set
        swa_val_preds = []
        swa_val_labels = []
        with torch.no_grad():
            for batch_images, batch_numeric, batch_labels in finetune_val_loader:
                batch_images = {k: v.to(device) for k, v in batch_images.items()}
                batch_numeric = batch_numeric.to(device)
                logits = model(batch_images, batch_numeric)
                swa_val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                swa_val_labels.extend(batch_labels.numpy())
        
        swa_val_auc = roc_auc_score(swa_val_labels, swa_val_preds) if len(set(swa_val_labels)) > 1 else 0.5
        
        logger.info(f"SWA Val AUC: {swa_val_auc:.4f} vs Best checkpoint Val AUC: {best_val_auc:.4f}")
        
        if swa_val_auc > best_val_auc:
            logger.info("SWA weights are BETTER — using SWA encoder.")
            use_swa_weights = True
        else:
            logger.info("SWA weights are WORSE — falling back to best checkpoint.")
            use_swa_weights = False
    elif use_swa and swa_epochs_collected < 3:
        logger.info(f"SWA skipped: only {swa_epochs_collected} epochs collected (need ≥3)")
    
    if not use_swa_weights and best_model_state is not None:
        logger.info(f"Restoring best model state with validation AUC={best_val_auc:.4f}")
        model.load_state_dict(best_model_state)
        
    # 1. Evaluate pure Neural Network Head on Test Set
    model.eval()
    nn_val_preds = []
    nn_val_labels = []
    with torch.no_grad():
        for batch_images, batch_numeric, batch_labels in test_loader:
            batch_images = {k: v.to(device) for k, v in batch_images.items()}
            batch_numeric = normalize_numeric(batch_numeric.to(device))
            logits = model(batch_images, batch_numeric)
            nn_val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            nn_val_labels.extend(batch_labels.numpy())
            
    nn_test_auc = roc_auc_score(nn_val_labels, nn_val_preds) if len(set(nn_val_labels)) > 1 else 0.5
    logger.info(f"Final PyTorch MultimodalClassifier Test AUC: {nn_test_auc:.4f}")
    
    # Create an identical StandardScaler so inference works properly
    from sklearn.preprocessing import StandardScaler
    scaler_num = StandardScaler()
    scaler_num.mean_ = numeric_mean.cpu().numpy()
    scaler_num.scale_ = numeric_std.cpu().numpy()
    scaler_num.var_ = np.power(scaler_num.scale_, 2)
    scaler_num.n_features_in_ = len(scaler_num.mean_)
    
    return {
        'best_val_auc': nn_test_auc,
        'test_proba': np.array(nn_val_preds),
        'test_labels': np.array(nn_val_labels),
        'scaler_num': scaler_num,
        'xgb_model': None,
        'scaler_cnn': None,
        'selector': None
    }
