"""
Visualization utilities for CNN feature heatmaps using Grad-CAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature', 'no-latex'])
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Visualizes which regions of the image are important for predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The CNN model
            target_layer: The layer to compute CAM for (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients_list = []
        self.activations_list = []
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save activations during forward pass."""
        self.activations_list.append(output.detach())
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradients during backward pass."""
        self.gradients_list.append(grad_output[0].detach())
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None, target_index: int = -1) -> np.ndarray:
        """
        Generate CAM for the input.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W] or dict
            target_class: Class to generate CAM for (None = use predicted class)
            target_index: If target layer is called multiple times (e.g. shared backbone), 
                          the index of the forward pass to capture (-1 for last).
            
        Returns:
        """
        self.model.eval()
        self.activations_list = []
        self.gradients_list = []
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item() if output.dim() > 1 else (output > 0).long().item()
        
        # Backward pass
        self.model.zero_grad()
        
        if output.dim() == 1 or output.shape[1] == 1:
            # Binary classification
            loss = output.squeeze()
        else:
            # Multi-class
            loss = output[0, target_class]
        
        loss.backward()
        
        # Compute CAM
        if not self.gradients_list or not self.activations_list:
            gradients, activations = None, None
        else:
            if target_index == -1:
                target_index = len(self.activations_list) - 1
            activations = self.activations_list[target_index]
            rev_idx = len(self.activations_list) - 1 - target_index
            if rev_idx < len(self.gradients_list):
                gradients = self.gradients_list[rev_idx]
            else:
                gradients = None
                
        if gradients is None or activations is None:
            # Fallback if graph is completely frozen and backward hooks did not trigger
            h, w = self.model(input_tensor).shape[-2:] if input_tensor.dim() >= 4 else (224, 224)
            if activations is not None:
                h, w = activations.shape[-2:]
            return np.zeros((h, w), dtype=np.float32)
            
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1).squeeze()  # [H, W]
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def cleanup(self):
        """Remove hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_target_layer(encoder: nn.Module, backbone_name: str) -> nn.Module:
    """Get the appropriate target layer for Grad-CAM based on backbone."""
    if backbone_name == 'resnet18':
        return encoder.features.layer4
    elif backbone_name == 'efficientnet_b0':
        return encoder.features.features[8]
    elif backbone_name == 'mobilenet_v3_small':
        return encoder.features.features[12]
    else:
        # Default: try to get last conv layer
        for module in reversed(list(encoder.features.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError(f"Could not find target layer for backbone: {backbone_name}")


def generate_heatmaps_for_sample(
    encoder: nn.Module,
    classifier_head: nn.Module,
    images: Dict[str, torch.Tensor],
    label: int,
    backbone_name: str,
    device: torch.device
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate heatmaps for all image modalities of a sample.
    
    Returns:
        Dict mapping image type to (original_image, heatmap)
    """
    results = {}
    
    # Create a wrapper that includes the classifier head
    class FullModel(nn.Module):
        def __init__(self, enc, head):
            super().__init__()
            self.encoder = enc
            self.head = head
            
        def forward(self, x):
            # x is a dict with single modality
            features = self.encoder(x)
            return self.head(features)
    
    for img_type, img_tensor in images.items():
        if img_tensor is None:
            continue
            
        # Get the single branch encoder for this modality
        if hasattr(encoder, 'encoders'):
            branch_encoder = encoder.encoders[img_type]
        else:
            branch_encoder = encoder
        
        try:
            target_layer = get_target_layer_for_branch(branch_encoder, backbone_name)
            
            # Create wrapper model for this branch
            full_model = FullModel(branch_encoder, classifier_head).to(device)
            
            gradcam = GradCAM(full_model, target_layer)
            
            # Prepare input
            img_input = img_tensor.unsqueeze(0).to(device) if img_tensor.dim() == 3 else img_tensor.to(device)
            
            # Generate CAM
            cam = gradcam.generate_cam(img_input, target_class=label)
            
            # Resize CAM to match original image size
            h, w = img_tensor.shape[-2:]
            cam_resized = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Get original image (denormalize)
            img_np = denormalize_image(img_tensor.cpu().numpy())
            
            results[img_type] = (img_np, cam_resized)
            
            gradcam.cleanup()
            
        except Exception as e:
            logger.warning(f"Could not generate heatmap for {img_type}: {e}")
            continue
    
    return results


def get_target_layer_for_branch(branch_encoder: nn.Module, backbone_name: str) -> nn.Module:
    """Get target layer for a single branch encoder."""
    if hasattr(branch_encoder, 'features'):
        if backbone_name == 'resnet18':
            if isinstance(branch_encoder.features, nn.Sequential):
                # ResNet18 via models.py uses list(children)[:-1], so ends with avgpool
                return branch_encoder.features[-2]
            return branch_encoder.features.layer4
        elif backbone_name == 'efficientnet_b0':
            return branch_encoder.features.features[8]
        elif backbone_name == 'mobilenet_v3_small':
            return branch_encoder.features.features[12]
    
    # Fallback: find last conv layer
    last_conv = None
    for module in branch_encoder.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        raise ValueError("Could not find conv layer for Grad-CAM")
    
    return last_conv


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """Denormalize ImageNet-normalized image for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    return img


def create_heatmap_overlay(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """Create overlay of heatmap on original image."""
    # Apply colormap to heatmap
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Ensure original is in correct format
    if original.max() > 1:
        original = original / 255.0
    
    # Create overlay
    overlay = alpha * heatmap_colored + (1 - alpha) * original
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def plot_heatmaps_grid(
    heatmap_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sample_id: str,
    prediction: Optional[float],
    true_label: int
) -> plt.Figure:
    """
    Create a grid plot of all modality heatmaps.
    
    Args:
        heatmap_results: Dict from generate_heatmaps_for_sample
        sample_id: Sample identifier
        prediction: Model prediction probability
        true_label: Ground truth label
        
    Returns:
        Matplotlib figure
    """
    n_modalities = len(heatmap_results)
    if n_modalities == 0:
        return None
    
    fig, axes = plt.subplots(2, n_modalities, figsize=(4 * n_modalities, 8))
    
    if n_modalities == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (img_type, (original, heatmap)) in enumerate(heatmap_results.items()):
        # Original image
        axes[0, idx].imshow(original)
        axes[0, idx].set_title(f'{img_type}\n(Original)')
        axes[0, idx].axis('off')
        
        # Heatmap overlay
        overlay = create_heatmap_overlay(original, heatmap)
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'{img_type}\n(Grad-CAM)')
        axes[1, idx].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cbar_ax)
    cbar.set_label('Attention', rotation=270, labelpad=15)
    
    
    if prediction is not None:
        pred_label = 1 if prediction > 0.5 else 0
        correct = "✓" if pred_label == true_label else "✗"
        pred_str = f"{prediction:.3f}"
        title = f'Sample {sample_id} | Pred: {pred_str} | True: {true_label} {correct}'
    else:
        title = f'Sample {sample_id} | Pred: N/A | True: {true_label}'
        
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    return fig


def generate_sample_heatmaps(
    encoder: nn.Module,
    test_loader,
    device: torch.device,
    backbone_name: str,
    wandb_run = None,
    logger = None,
    output_dir: Optional[str] = None,
    n_samples: int = 5,  # Kept for backward compatibility
    n_random: int = 2,
    n_best: int = 2,
    n_worst: int = 2,
    n_tp: int = 0,
    n_fp: int = 0,
    n_fn: int = 0,
    n_tn: int = 0,
    probs: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    selected_channels: Optional[Dict[str, List[int]]] = None
) -> List[plt.Figure]:
    """
    Generate heatmaps for test samples, including random, best (most confident),
    and worst (least confident/lowest prob) predictions.
    
    Args:
        encoder: Trained CNN encoder
        test_loader: DataLoader with test samples
        device: Torch device
        backbone_name: Name of backbone architecture
        n_samples: Legacy argument (maps to n_random if others are default)
        n_random: Number of random samples
        n_best: Number of high confidence positive samples
        n_worst: Number of high confidence negative (low prob) samples
        probs: Array of probabilities for all samples in test_loader (must match order)
        true_labels: Array of true labels
        wandb_run: Optional wandb run for logging
        logger: Optional logger
        output_dir: Directory to save heatmaps locally
    """
    import random
    import numpy as np
    
    encoder.eval()
    figures = []
    dataset = test_loader.dataset
    indices = set()
    
    if probs is not None:
        if logger:
            logger.info("Selecting samples by confidence...")
        sorted_indices = np.argsort(probs)
        
        # Best (highest prob)
        if n_best > 0:
            indices.update(sorted_indices[-n_best:])
            
        # Worst (lowest prob)
        if n_worst > 0:
            indices.update(sorted_indices[:n_worst])
            
        if true_labels is not None:
            # TP: y=1, prob descending (high confidence correct positive)
            if n_tp > 0:
                tp_indices = [i for i in sorted_indices[::-1] if true_labels[i] == 1][:n_tp]
                indices.update(tp_indices)
                
            # FP: y=0, prob descending (high confidence incorrect positive)
            if n_fp > 0:
                fp_indices = [i for i in sorted_indices[::-1] if true_labels[i] == 0][:n_fp]
                indices.update(fp_indices)
                
            # FN: y=1, prob ascending (high confidence incorrect negative/missed)
            if n_fn > 0:
                fn_indices = [i for i in sorted_indices if true_labels[i] == 1][:n_fn]
                indices.update(fn_indices)
                
            # TN: y=0, prob ascending (high confidence correct negative)
            if n_tn > 0:
                tn_indices = [i for i in sorted_indices if true_labels[i] == 0][:n_tn]
                indices.update(tn_indices)
            
        # Random from remaining
        remaining = list(set(range(len(probs))) - indices)
        if n_random > 0 and remaining:
            indices.update(random.sample(remaining, min(n_random, len(remaining))))
            
        count = max(count, 5)
        indices.update(random.sample(range(len(dataset)), min(count, len(dataset))))
            
    # Convert to sorted list
    indices = sorted(list(indices))
    
    if logger:
        logger.info(f"Generating Grad-CAM heatmaps for {len(indices)} samples...")
    
    # Create output directory if provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    for idx in indices:
        try:
            sample = dataset[idx]
            images, numeric, label = sample
            
            # Move images to device
            images_device = {k: v.unsqueeze(0).to(device) for k, v in images.items()}
            
            # Get prediction
            with torch.no_grad():
                features = encoder(images_device)
            
            heatmap_results = {}
            for img_type, img_tensor in images.items():
                if img_tensor is None:
                    continue
                
                # Get feature maps from encoder for this modality
                if hasattr(encoder, 'encoders') and img_type in encoder.encoders:
                    branch = encoder.encoders[img_type]
                    
                    # Use hook to capture activations
                    activations = []
                    def hook_fn(module, input, output):
                        activations.append(output.detach())
                    
                    # Find last conv layer
                    target_layer = get_target_layer_for_branch(branch, backbone_name)
                    hook = target_layer.register_forward_hook(hook_fn)
                    
                    # Forward pass
                    with torch.no_grad():
                        _ = branch(img_tensor.unsqueeze(0).to(device))
                    
                    hook.remove()
                    
                    if activations:
                        # Mean activation across channels
                        act = activations[0].squeeze()  # [C, H, W]
                        
                        # Filter by selected channels if provided
                        if selected_channels and img_type in selected_channels:
                            channels = selected_channels[img_type]
                            if not channels:
                                # No selected channels for this view
                                act = torch.zeros_like(act[0]) # [H, W] zero
                            else:
                                act = act[channels] # subset channels
                        
                        if len(act.shape) == 3:
                             heatmap = torch.mean(act, dim=0).cpu().numpy()
                        else:
                             heatmap = act.cpu().numpy() # already 2d (single channel) or zero
                        
                        # Normalize
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        
                        # Resize to original image size
                        h, w = img_tensor.shape[-2:]
                        heatmap_resized = F.interpolate(
                            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float(),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().numpy()
                        
                        # Get original image
                        img_np = denormalize_image(img_tensor.cpu().numpy())
                        
                        heatmap_results[img_type] = (img_np, heatmap_resized)
            
            if heatmap_results:
                # Get prediction for title if available
                pred_val = None
                if probs is not None:
                     pred_val = probs[idx]

                fig = plot_heatmaps_grid(
                    heatmap_results,
                    sample_id=str(idx),
                    prediction=pred_val,
                    true_label=int(label.item()) if torch.is_tensor(label) else int(label)
                )
                
                if fig:
                    figures.append(fig)
                    
                    # Save locally if directory provided
                    if output_dir:
                        save_path = os.path.join(output_dir, f'heatmap_sample_{idx}.png')
                        fig.savefig(save_path)
                        if logger:
                            logger.info(f"Saved heatmap to {save_path}")
                    
                    if wandb_run:
                        import wandb
                        wandb_run.log({
                            f'heatmaps/sample_{idx}': wandb.Image(fig)
                        })
                    
                    plt.close(fig)
                    
        except Exception as e:
            if logger:
                logger.warning(f"Could not generate heatmap for sample {idx}: {e}")
            continue
    
    return figures


def extract_attention_maps(
    encoder: nn.Module,
    images: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Extract CBAM spatial attention maps from the encoder for each modality.
    
    Hooks into the SpatialAttention module within each SingleBranchEncoder's CBAM
    to capture the spatial attention weights [1, 1, H, W].
    
    Args:
        encoder: Trained CNN encoder (Multi or Shared backbone)
        images: Dict of image tensors per modality (unbatched, [C, H, W])
        device: Torch device
        
    Returns:
        Dict mapping modality name to attention map [H, W] (resized to image size)
    """
    encoder.eval()
    attention_maps = {}
    
    for img_type, img_tensor in images.items():
        # Get the branch encoder
        if hasattr(encoder, 'encoders') and img_type in encoder.encoders:
            branch = encoder.encoders[img_type]
        elif hasattr(encoder, 'shared_encoder'):
            branch = encoder.shared_encoder
        else:
            continue
        
        # Find spatial attention module
        spatial_attn = None
        if hasattr(branch, 'attention') and hasattr(branch.attention, 'spatial_attention'):
            spatial_attn = branch.attention.spatial_attention
        
        if spatial_attn is None:
            continue
        
        # Hook to capture spatial attention output
        captured = []
        def hook_fn(module, input, output, _captured=captured):
            _captured.append(output.detach())
        
        hook = spatial_attn.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = branch(img_tensor.unsqueeze(0).to(device))
            
            if captured:
                attn_map = captured[0].squeeze().cpu().numpy()  # [H, W]
                if attn_map.ndim < 2:
                    continue  # Invalid or 1D attention map, skip it
                    
                # Resize to original image size
                h, w = img_tensor.shape[-2:]
                attn_resized = F.interpolate(
                    torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                attention_maps[img_type] = attn_resized
        finally:
            hook.remove()
    
    return attention_maps


def plot_heatmaps_3row_grid(
    raw_images: Dict[str, np.ndarray],
    attention_maps: Dict[str, np.ndarray],
    gradcam_maps: Dict[str, np.ndarray],
    sample_id: str,
    prediction: Optional[float],
    true_label: int,
    numeric_info: Optional[Dict[str, float]] = None,
    image_types: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create a 3-row grid: Row 1 = Raw images, Row 2 = Attention overlays, Row 3 = Grad-CAM overlays.
    
    Includes a text header with key numeric features (age, K values, pachy, etc.)
    
    Args:
        raw_images: Dict modality -> denormalized image [H, W, 3]
        attention_maps: Dict modality -> CBAM spatial attention [H, W]
        gradcam_maps: Dict modality -> Grad-CAM heatmap [H, W]
        sample_id: Sample identifier
        prediction: Model prediction probability
        true_label: Ground truth label
        numeric_info: Dict of numeric feature name -> value for overlay
        image_types: Ordered list of modalities (defaults to dict keys)
        
    Returns:
        Matplotlib figure
    """
    if image_types is None:
        image_types = list(raw_images.keys())
    
    n_cols = len(image_types)
    if n_cols == 0:
        return None
    
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    row_labels = ['Raw Image', 'Attention (CBAM)', 'Grad-CAM']
    
    for col_idx, img_type in enumerate(image_types):
        # Row 1: Raw image
        if img_type in raw_images:
            axes[0, col_idx].imshow(raw_images[img_type])
        axes[0, col_idx].set_title(img_type.replace('_', ' ').title(), fontsize=10)
        axes[0, col_idx].axis('off')
        
        # Row 2: Attention overlay
        if img_type in raw_images and img_type in attention_maps:
            overlay_attn = create_heatmap_overlay(raw_images[img_type], attention_maps[img_type], alpha=0.5)
            axes[1, col_idx].imshow(overlay_attn)
        elif img_type in raw_images:
            axes[1, col_idx].imshow(raw_images[img_type])
            axes[1, col_idx].text(0.5, 0.5, 'N/A', transform=axes[1, col_idx].transAxes,
                                   ha='center', va='center', fontsize=14, color='white')
        axes[1, col_idx].axis('off')
        
        # Row 3: Grad-CAM overlay
        if img_type in raw_images and img_type in gradcam_maps:
            overlay_gc = create_heatmap_overlay(raw_images[img_type], gradcam_maps[img_type], alpha=0.5)
            axes[2, col_idx].imshow(overlay_gc)
        elif img_type in raw_images:
            axes[2, col_idx].imshow(raw_images[img_type])
            axes[2, col_idx].text(0.5, 0.5, 'N/A', transform=axes[2, col_idx].transAxes,
                                   ha='center', va='center', fontsize=14, color='white')
        axes[2, col_idx].axis('off')
    
    # Row labels on the left
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
    
    # Title with prediction info
    if prediction is not None:
        pred_label = 1 if prediction > 0.5 else 0
        correct = "✓" if pred_label == true_label else "✗"
        title = f'Sample {sample_id} | Pred: {prediction:.3f} | True: {true_label} {correct}'
    else:
        title = f'Sample {sample_id} | True: {true_label}'
    
    # Add numeric feature info as subtitle
    if numeric_info:
        info_parts = []
        # Key features to display
        display_keys = [
            ('age', 'Age'),
            ('K1 F (D):', 'K1'),
            ('K2 F (D):', 'K2'),
            ('KMax Sagittal Front (D)', 'KMax'),
            ('Pachy Apex:', 'Pachy'),
            ('Pachy Min:', 'PachyMin'),
            ('ISV:', 'ISV'),
            ('IVA:', 'IVA'),
            ('KI:', 'KI'),
        ]
        for key, short_name in display_keys:
            if key in numeric_info:
                val = numeric_info[key]
                if isinstance(val, float):
                    info_parts.append(f'{short_name}={val:.1f}')
                else:
                    info_parts.append(f'{short_name}={val}')
        
        info_str = ' | '.join(info_parts)
        title = f'{title}\n{info_str}'
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.25])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cbar_ax)
    cbar.set_label('Activation', rotation=270, labelpad=15)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    return fig

