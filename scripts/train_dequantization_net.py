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
        hdr_dir=/custom/path
"""

from __future__ import annotations

import logging
import sys
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
from omegaconf import DictConfig

from luminascale.models import create_dequantization_net
from luminascale.training import DequantizationTrainer
from luminascale.training.trainer import DequantizationDataset

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

    device = torch.device(cfg.device)

    logger.info(f"Device: {device}")
    logger.info(f"HDR directory: {cfg.hdr_dir}")
    logger.info(f"sRGB directory: {cfg.get('srgb_dir', 'Not specified (using default)')}")
    logger.info(f"Output directory: {cfg.output_dir}")

    # Create dataset and dataloader
    dataset = DequantizationDataset(cfg.hdr_dir, srgb_dir=cfg.get("srgb_dir"))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
    )

    # Create model
    model = create_dequantization_net(
        device=device, base_channels=cfg.model.base_channels
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Create trainer and train
    trainer = DequantizationTrainer(
        model=model, device=device, learning_rate=cfg.learning_rate
    )

    trainer.train(
        train_dataloader=dataloader,
        num_epochs=cfg.epochs,
        checkpoint_dir=cfg.output_dir,
        checkpoint_freq=cfg.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
