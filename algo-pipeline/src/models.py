import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple

# Backbone registry
BACKBONE_REGISTRY = {
    'resnet18': {
        'model_fn': models.resnet18,
        'weights': 'IMAGENET1K_V1',
        'feature_dim': 512,
        'layer_name': 'layer4',
    },
    'efficientnet_b0': {
        'model_fn': models.efficientnet_b0,
        'weights': 'IMAGENET1K_V1',
        'feature_dim': 1280,
        'layer_name': 'features.8',
    },
    'mobilenet_v3_small': {
        'model_fn': models.mobilenet_v3_small,
        'weights': 'IMAGENET1K_V1',
        'feature_dim': 576,
        'layer_name': 'features.12',
    },
    'efficientnet_b4': {
        'model_fn': models.efficientnet_b4,
        'weights': 'IMAGENET1K_V1',
        'feature_dim': 1792,
        'layer_name': 'features.8',
    },
}

class ChannelAttention(nn.Module):
    """Channel Attention Module."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion."""
    def __init__(self, feature_dim: int, num_modalities: int = 4, reduction_factor: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.output_dim = (feature_dim * num_modalities) // reduction_factor
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(0.1)
        )
        
        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.LayerNorm(self.output_dim // 2),
            nn.ReLU()
        )
        self.final_dim = self.output_dim // 2
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(features_list, dim=1)
        attended, _ = self.attention(stacked, stacked, stacked)
        attended = self.norm1(attended + stacked)
        ffn_out = self.ffn(attended)
        attended = self.norm2(attended + ffn_out)
        
        weights = torch.softmax(self.modality_weights, dim=0)
        weighted = (attended * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        attended_flat = attended.flatten(1)
        fused = self.projection(attended_flat)
        
        return torch.cat([weighted, fused], dim=1)

class BranchAdapter(nn.Module):
    """Lightweight trainable adapter."""
    def __init__(self, in_features: int, hidden_dim: int = 256, 
                 out_features: Optional[int] = None, dropout: float = 0.3):
        super().__init__()
        if out_features is None:
            out_features = in_features
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = (in_features == out_features)
        
        self.norm1 = nn.LayerNorm(in_features)
        self.adapter = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        adapted = self.adapter(normed)
        out = x + adapted if self.use_residual else adapted
        return self.norm2(out)

class SingleBranchEncoder(nn.Module):
    """Single-branch CNN feature extractor."""
    def __init__(self, backbone_name: str = 'resnet18', freeze_mode: str = 'all',
                 use_attention: bool = True, adapter_mode: str = 'none',
                 adapter_hidden_dim: int = 256, adapter_dropout: float = 0.3):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.use_attention = use_attention
        self.adapter_mode = adapter_mode
        backbone_info = BACKBONE_REGISTRY[backbone_name]
        
        model_fn = backbone_info['model_fn']
        weights = backbone_info['weights']
        self.feature_dim = backbone_info['feature_dim']
        
        # Initialize backbone
        if backbone_name == 'resnet18':
            model = model_fn(weights=weights)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            if use_attention:
                self.attention = CBAM(in_channels=512, reduction=16)
        elif backbone_name == 'efficientnet_b0':
            model = model_fn(weights=weights)
            self.features = model.features
            self.avgpool = model.avgpool
            self.flatten = nn.Flatten()
            if use_attention:
                self.attention = CBAM(in_channels=1280, reduction=32)
        elif backbone_name == 'efficientnet_b4':
            model = model_fn(weights=weights)
            self.features = model.features
            self.avgpool = model.avgpool
            self.flatten = nn.Flatten()
            if use_attention:
                self.attention = CBAM(in_channels=1792, reduction=32)
        elif backbone_name == 'mobilenet_v3_small':
            model = model_fn(weights=weights)
            self.features = model.features
            self.avgpool = model.avgpool
            self.flatten = nn.Flatten()
            if use_attention:
                self.attention = CBAM(in_channels=576, reduction=16)
        
        # Freeze layers
        effective_freeze = 'all' if adapter_mode in ['per_branch', 'fusion_only'] else freeze_mode
        self._apply_freeze_mode(effective_freeze)
        
        self.adapter = None
        if adapter_mode == 'per_branch':
            self.adapter = BranchAdapter(
                in_features=self.feature_dim,
                hidden_dim=adapter_hidden_dim,
                out_features=self.feature_dim,
                dropout=adapter_dropout
            )
    
    def _apply_freeze_mode(self, freeze_mode: str):
        """Apply freezing strategy to backbone layers."""
        if freeze_mode == 'all':
            for param in self.features.parameters():
                param.requires_grad = False
        elif freeze_mode == 'partial':
            # Freeze all backbone layers first
            for param in self.features.parameters():
                param.requires_grad = False
            
            # Unfreeze last few layers based on backbone type
            if self.backbone_name == 'resnet18':
                # Unfreeze layer4 (last residual block)
                for name, param in self.features.named_parameters():
                    if 'layer4' in name or 'layer3' in name:
                        param.requires_grad = True
            elif self.backbone_name == 'efficientnet_b0':
                # Unfreeze last 3 blocks (features.6, 7, 8)
                for name, param in self.features.named_parameters():
                    if any(f'features.{i}' in name for i in [6, 7, 8]):
                        param.requires_grad = True
            elif self.backbone_name == 'mobilenet_v3_small':
                for name, param in self.features.named_parameters():
                    if any(f'features.{i}' in name for i in [10, 11, 12]):
                        param.requires_grad = True
                        
        if self.use_attention and hasattr(self, 'attention'):
            for param in self.attention.parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.use_attention and hasattr(self, 'attention'):
            x = self.attention(x)
        if hasattr(self, 'avgpool'):
            x = self.avgpool(x)
        x = self.flatten(x)
        if self.adapter is not None:
            x = self.adapter(x)
        return x

class MultiBackboneCNNEncoder(nn.Module):
    """Multi-head CNN encoder with separate backbone per image type."""
    def __init__(self, image_types: List[str], backbone_name: str = 'resnet18', 
                 freeze_mode: str = 'all', use_attention: bool = True, 
                 use_cross_modal_fusion: bool = True, adapter_mode: str = 'none', 
                 adapter_hidden_dim: int = 256, adapter_dropout: float = 0.3):
        super().__init__()
        
        self.image_types = image_types
        self.use_cross_modal_fusion = use_cross_modal_fusion
        
        self.encoders = nn.ModuleDict({
            img_type: SingleBranchEncoder(
                backbone_name=backbone_name,
                freeze_mode=freeze_mode,
                use_attention=use_attention,
                adapter_mode=adapter_mode,
                adapter_hidden_dim=adapter_hidden_dim,
                adapter_dropout=adapter_dropout
            )
            for img_type in self.image_types
        })
        
        self.single_branch_dim = self.encoders[self.image_types[0]].feature_dim
        
        if use_cross_modal_fusion:
            self.fusion = CrossModalAttentionFusion(
                feature_dim=self.single_branch_dim,
                num_modalities=len(self.image_types),
                reduction_factor=2
            )
            self.feature_dim = self.fusion.final_dim + self.single_branch_dim
        else:
            self.fusion = None
            self.feature_dim = self.single_branch_dim * len(self.image_types)
    
    def forward(self, images_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features_list = [self.encoders[img_type](images_dict[img_type]) 
                        for img_type in self.image_types]
        
        if self.fusion is not None:
            return self.fusion(features_list)
        return torch.cat(features_list, dim=1)


class SharedBackboneCNNEncoder(nn.Module):
    """Shared-backbone CNN encoder: one backbone processes all image types."""
    def __init__(self, image_types: List[str], backbone_name: str = 'resnet18',
                 freeze_mode: str = 'all', use_attention: bool = True,
                 use_cross_modal_fusion: bool = True, adapter_mode: str = 'per_branch',
                 adapter_hidden_dim: int = 256, adapter_dropout: float = 0.3):
        super().__init__()
        
        self.image_types = image_types
        self.use_cross_modal_fusion = use_cross_modal_fusion
        
        self.shared_encoder = SingleBranchEncoder(
            backbone_name=backbone_name,
            freeze_mode=freeze_mode,
            use_attention=use_attention,
            adapter_mode='none',
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout=adapter_dropout
        )
        
        backbone_dim = self.shared_encoder.feature_dim
        self.single_branch_dim = backbone_dim
        
        # Per-branch adapters (lightweight, trainable)
        if adapter_mode in ['per_branch']:
            self.branch_adapters = nn.ModuleDict({
                img_type: BranchAdapter(
                    in_features=backbone_dim,
                    hidden_dim=adapter_hidden_dim,
                    out_features=backbone_dim,
                    dropout=adapter_dropout
                )
                for img_type in self.image_types
            })
        else:
            self.branch_adapters = None
        
        if use_cross_modal_fusion:
            self.fusion = CrossModalAttentionFusion(
                feature_dim=backbone_dim,
                num_modalities=len(self.image_types),
                reduction_factor=2
            )
            self.feature_dim = self.fusion.final_dim + self.single_branch_dim
        else:
            self.fusion = None
            self.feature_dim = backbone_dim * len(self.image_types)
    
    def forward(self, images_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features_list = []
        for img_type in self.image_types:
            feat = self.shared_encoder(images_dict[img_type])
            if self.branch_adapters is not None:
                feat = self.branch_adapters[img_type](feat)
            features_list.append(feat)
        
        if self.fusion is not None:
            return self.fusion(features_list)
        return torch.cat(features_list, dim=1)

class MultimodalClassifier(nn.Module):
    """End-to-end multimodal classifier."""
    def __init__(self, cnn_encoder, 
                 num_numeric_features: int, dropout: float = 0.4,
                 bottleneck_dim: int = 128):
        super().__init__()
        
        self.cnn_encoder = cnn_encoder
        cnn_dim = cnn_encoder.feature_dim
        
        self.cnn_bottleneck = nn.Sequential(
            nn.Linear(cnn_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        combined_dim = bottleneck_dim + num_numeric_features
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, images: Dict[str, torch.Tensor], 
                numeric_features: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn_encoder(images)
        cnn_projected = self.cnn_bottleneck(cnn_features)
        combined = torch.cat([cnn_projected, numeric_features], dim=1)
        return self.classifier(combined).squeeze(1)
