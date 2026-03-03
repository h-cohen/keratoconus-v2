import os
import sys
import logging
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def setup_logging(output_dir: Path, verbose: bool = True) -> logging.Logger:
    """Setup logging with file and console handlers."""
    logger = logging.getLogger('keratoconus_training')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args_and_update_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse command line arguments and update config dictionary."""
    parser = argparse.ArgumentParser(description='Keratoconus Pipeline')
    
    parser.add_argument('--config', type=str, help='Path to config file')
    
    known_args = [
        'learning_rate', 'batch_size', 'num_epochs', 'backbone', 'dropout',
        'weight_decay', 'focal_gamma', 'n_select_features', 'xgb_learning_rate',
        'xgb_max_depth', 'xgb_n_estimators', 'n_cv_folds', 'wandb_run_name',
        'wandb_project', 'wandb_entity', 'training_mode', 'image_size',
        'finetune_epochs'
    ]
    
    for arg in known_args:
        val = config.get(arg)
        arg_type = type(val) if val is not None else str
        parser.add_argument(f'--{arg}', type=arg_type, default=val)

    args, unknown = parser.parse_known_args()
    
    # Update config with args
    for arg in known_args:
        if hasattr(args, arg):
            config[arg] = getattr(args, arg)
            
    return config

def init_wandb(config: Dict, logger: logging.Logger) -> Optional[Any]:
    """Initialize wandb run."""
    if not config.get('wandb_enabled', False):
        return None
    
    try:
        import wandb
        # API key handling
        api_key = os.environ.get('WANDB_API_KEY') or config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key, relogin=True)
            
        run_name = config.get('wandb_run_name')
        if run_name is None:
            run_name = f"{config.get('training_mode', 'experiment')}_{config.get('backbone', 'model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        run = wandb.init(
            entity=config.get('wandb_entity'),
            project=config.get('wandb_project'),
            name=run_name,
            config=config,
            tags=config.get('wandb_tags', []),
            notes=config.get('wandb_notes', ''),
            reinit=True
        )
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None
