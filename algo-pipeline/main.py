import argparse
import logging
import torch
from pathlib import Path
from src.utils import setup_logging, load_config, parse_args_and_update_config, init_wandb, set_seed
from src.data import load_data, IMAGE_TYPES
from src.evaluation import nested_cv_multimodal

def main():
    # Determine config path
    import sys
    config_path = Path('configs/default_config.yaml')
    if '--config' in sys.argv:
        try:
            idx = sys.argv.index('--config')
            if idx + 1 < len(sys.argv):
                config_path = Path(sys.argv[idx + 1])
        except ValueError:
            pass
            
    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Update with CLI args
    config = parse_args_and_update_config(config)
    
    # Set random seed globally
    seed = config.get('random_state', 42)
    set_seed(seed)
    
    # Create run-specific output directory
    from datetime import datetime
    run_name = config.get('wandb_run_name', 'experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir_name = f"{run_name}_{timestamp}"
    
    # Base output dir from config (defaults to 'results')
    base_output_dir = Path(config.get('output_dir', 'results'))
    run_output_dir = base_output_dir / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config
    config['output_dir'] = str(run_output_dir)
    print(f"Output directory set to: {config['output_dir']}")
    
    # Setup
    logger = setup_logging(run_output_dir)
    
    logger.info("Starting Keratoconus Pipeline")
    logger.info(f"Config: {config}")
    
    # Wandb
    wandb_run = init_wandb(config, logger)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data
    df, numeric_features = load_data(config, logger)
    image_dir = Path(config['image_dir'])
    
    # Run CV
    results = nested_cv_multimodal(
        df_data=df,
        image_dir=image_dir,
        numeric_feature_names=numeric_features,
        image_types=IMAGE_TYPES,
        device=device,
        config=config,
        logger=logger,
        wandb_run=wandb_run
    )
    
    logger.info("Pipeline completed.")

if __name__ == '__main__':
    main()
