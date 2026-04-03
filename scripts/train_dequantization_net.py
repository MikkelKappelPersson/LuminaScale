"""Hydra-based training script for Dequantization-Net.

Usage (local development):
    python scripts/train_dequantization_net.py --config-name=default

Usage (HPC via Slurm):
    sbatch scripts/train_dequantization_net.sh

Override config on CLI:
    python scripts/train_dequantization_net.py \
        --config-name=default \
        batch_size=16 \
        epochs=50 \
        lmdb_path=/path/to/training_data.lmdb
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path to allow imports without modifying environment
# This allows 'from luminascale...' to work regardless of where the script is called from.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from luminascale.models import create_dequantization_net
from luminascale.training import DequantizationTrainer
from luminascale.training.dequantization_trainer import OnTheFlyBDEDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main training entry point with Hydra config management."""
    # Prevent 'BrokenPipeError' when piping output (e.g. to 'tee' or 'head')
    from signal import signal, SIGPIPE, SIG_DFL
    try:
        signal(SIGPIPE, SIG_DFL)
    except Exception:
        pass

    # If the config has a 'training' key, use it (handles cases where we nested)
    if "training" in cfg:
        cfg = cfg.training
    
    # Set OCIO environment variable for color management
    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)
        logger.info(f"OCIO config: {ocio_config}")
    else:
        logger.warning(f"OCIO config not found at {ocio_config}")

    device = torch.device(cfg.device)

    logger.info(f"Device: {device}")
    logger.info(f"LMDB path: {cfg.get('lmdb_path', 'None')}")
    logger.info(f"Output directory: {cfg.output_dir}")

    # Create dataset and dataloader
    lmdb_path = Path(cfg.get("lmdb_path")).resolve()
    dataset = OnTheFlyBDEDataset(
        lmdb_path=lmdb_path,
        device=device,
        crop_size=cfg.get("crop_size", 512),
        patches_per_image=cfg.get("patches_per_image", 1)
    )
    
    # For CUDA devices, disable multiprocessing to avoid CUDA re-initialization errors
    num_workers = 0 if "cuda" in str(device) else cfg.get("num_workers", 4)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # Keep patches from same image together to maximize cache hit rate
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory since dataset already returns GPU tensors
    )

    # Create model
    model = create_dequantization_net(
        device=device, base_channels=cfg.model.base_channels
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Create trainer and train
    # Use a timestamp-based run name to separate runs in TensorBoard and checkpoints
    if hasattr(cfg, "output_dir"):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.output_dir).resolve() / run_id
        checkpoint_dir = (run_dir / "checkpoints").resolve()
        log_dir = (run_dir / "tensorboard").resolve()
        
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save absolute copy of current hydra config for reproducibility
        with open(run_dir / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
            
        logger.info(f"Run directory created at: {run_dir}")
    else:
        log_dir = None
        checkpoint_dir = None
        run_id = None
    
    trainer = DequantizationTrainer(
        model=model, device=device, learning_rate=cfg.learning_rate, log_dir=log_dir
    )

    trainer.train(
        train_dataloader=dataloader,
        num_epochs=cfg.epochs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=cfg.checkpoint_freq,
        run_name=run_id,
    )
    
    # Clean up GPU resources
    dataset.cleanup()
    logger.info("Training complete. GPU resources released.")


if __name__ == "__main__":
    main()
