import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
import logging

# Constants
IMAGE_TYPES = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']

def get_image_transform(training: bool = False, use_augmentation: bool = False, size: int = 224) -> transforms.Compose:
    """Get image preprocessing transform."""
    if training and use_augmentation:
        return transforms.Compose([
            transforms.Resize((int(size*1.1), int(size*1.1))),
            transforms.RandomCrop((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Conservative rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle color changes
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class KeratoconusDataset(Dataset):
    """PyTorch Dataset for keratoconus multimodal data."""
    def __init__(self, df: pd.DataFrame, image_dir: Path, 
                 numeric_feature_names: List[str], transform: transforms.Compose,
                 image_types: List[str] = IMAGE_TYPES):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.numeric_feature_names = numeric_feature_names
        self.transform = transform
        self.image_types = image_types
        
        if numeric_feature_names:
            self.numeric_data = torch.tensor(
                self.df[numeric_feature_names].values, dtype=torch.float32
            )
        else:
            self.numeric_data = torch.zeros(len(self.df), dtype=torch.float32)
        
        # Handle case where 'y' might not exist (inference mode)
        if 'y' in self.df.columns:
            self.labels = torch.tensor(self.df['y'].values, dtype=torch.float32)
        else:
            self.labels = torch.zeros(len(self.df), dtype=torch.float32)
            
        self.filenames = self.df['filename'].values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        base_name = filename.replace('.jpg', '')
        
        images_dict = {}
        for img_type in self.image_types:
            img_filename = f"{base_name}_{img_type}.jpg"
            img_path = self.image_dir / img_filename
            try:
                image = Image.open(str(img_path)).convert('RGB')
                images_dict[img_type] = self.transform(image)
            except Exception as e:
                raise FileNotFoundError(f"Could not load image {img_path}: {e}")
        
        numeric = self.numeric_data[idx] if len(self.numeric_feature_names) > 0 else torch.tensor([])
        label = self.labels[idx]
        
        return images_dict, numeric, label

def collate_keratoconus(batch: List) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Custom collate function for batching."""
    images_dicts, numerics, labels = zip(*batch)
    
    image_types = list(images_dicts[0].keys())
    batched_images = {
        img_type: torch.stack([d[img_type] for d in images_dicts])
        for img_type in image_types
    }
    
    batched_numeric = torch.stack(numerics)
    batched_labels = torch.stack(labels).flatten()
    
    return batched_images, batched_numeric, batched_labels

def load_data(config: Dict, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """Load and preprocess dataset."""
    logger.info("Loading data...")
    
    df = pd.read_csv(config['data_csv'], dtype={'id': str})
    
    with open(config['features_txt'], 'r') as f:
        numeric_features = [line.strip() for line in f.readlines()]
    
    df_clean = df[df['has_all_images']].dropna(subset=numeric_features).copy()
    
    kmax_min_filter = config.get('kmax_min_filter', None)
    kmax_max_filter = config.get('kmax_max_filter', config.get('kmax_filter', None))
    exclusive_kmax_min = config.get('exclusive_kmax_min', True)
    exclusive_kmax_max = config.get('exclusive_kmax_max', False)
    
    if kmax_min_filter is not None or kmax_max_filter is not None:
        kmax_col = 'Km F (D):'
        before = len(df_clean)
        
        if kmax_min_filter is not None:
            if exclusive_kmax_min:
                df_clean = df_clean[df_clean[kmax_col] > kmax_min_filter].copy()
            else:
                df_clean = df_clean[df_clean[kmax_col] >= kmax_min_filter].copy()
                
        if kmax_max_filter is not None:
            if exclusive_kmax_max:
                df_clean = df_clean[df_clean[kmax_col] < kmax_max_filter].copy()
            else:
                df_clean = df_clean[df_clean[kmax_col] <= kmax_max_filter].copy()
                
        removed = before - len(df_clean)
        logger.info(f"KMax filtering boundaries applied: removed {removed} samples out of bounds, {len(df_clean)} remaining.")
    
    logger.info(f"Loaded {len(df_clean)} samples with complete data.")
    return df_clean, numeric_features

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    return {cls: weight for cls, weight in zip(unique_classes, weights)}

def create_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Create WeightedRandomSampler for balanced batch sampling."""
    class_weights = compute_class_weights(y)
    sample_weights = np.array([class_weights[label] for label in y])
    sample_weights = torch.from_numpy(sample_weights).float()
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def apply_smote_oversampling(X: np.ndarray, y: np.ndarray, 
                            sampling_strategy: str = 'auto',
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance classes."""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        print("Warning: imbalanced-learn not installed. Skipping SMOTE.")
        return X, y
