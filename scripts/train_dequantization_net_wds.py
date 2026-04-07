"""Hydra-based training script for Dequantization-Net (WebDataset Version).

Usage (HPC via Slurm):
    sbatch scripts/train_wds.sh

Override config on CLI:
    python scripts/train_dequantization_net_wds.py \
        --config-name=wds \
        batch_size=32 \
        epochs=50
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from luminascale.models import create_dequantization_net
from luminascale.training.dequantization_trainer import LuminaScaleModule
from luminascale.data.wds_dataset import LuminaScaleWebDataset

# Enable standard logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"},
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO", "propagate": True},
        "luminascale": {"handlers": ["default"], "level": "DEBUG", "propagate": False},
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class DebugCallback(L.Callback):
    """Debug callback to trace training loop."""
    
    def on_train_start(self, trainer, pl_module):
        print(f"\n{'='*80}")
        print(f"[DEBUG_CALLBACK] on_train_start called")
        print(f"[DEBUG_CALLBACK] About to iterate through batches...")
        print(f"{'='*80}\n")
    
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n{'='*80}")
        print(f"[DEBUG_CALLBACK] on_train_epoch_start called (epoch={trainer.current_epoch})")
        print(f"{'='*80}\n")
    
    def on_batch_start(self, trainer, pl_module):
        print(f"[DEBUG_CALLBACK] on_batch_start (step={trainer.global_step})")


@hydra.main(config_path="../configs", config_name="wds", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main training entry point for WebDataset pipeline."""
    
    print(f"\n{'='*80}")
    print(f"[MAIN] Starting training initialization...")
    print(f"{'='*80}\n")
    
    # Set OCIO environment variable
    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)
    
    # Run directory setup
    run_id = datetime.now().strftime("wds_%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[MAIN] Run directory: {run_dir}")
    
    # Log config
    with open(run_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 1. Create WebDataset Loaders
    print(f"[MAIN] Creating WebDataset with shard_path: {cfg.get('shard_path')}")
    print(f"[MAIN] batch_size={cfg.get('batch_size', 32)}, num_workers={cfg.get('num_workers', 4)}")
    
    train_dataset = LuminaScaleWebDataset(
        shard_path=cfg.get("shard_path"),
        batch_size=cfg.get("batch_size", 32),
        shuffle_buffer=cfg.get("shuffle_buffer", 1000),
        is_training=True
    )
    print(f"[MAIN] ✓ WebDataset created")
    
    print(f"[MAIN] Getting WebLoader...")
    train_loader = train_dataset.get_loader(num_workers=cfg.get("num_workers", 4))
    print(f"[MAIN] ✓ WebLoader created")
    
    # 2. Setup Lightning Module
    print(f"[MAIN] Creating model...")
    model = create_dequantization_net(in_channels=3, base_channels=cfg.model.base_channels)
    print(f"[MAIN] ✓ Model created")
    
    print(f"[MAIN] Creating LuminaScaleModule...")
    ls_module = LuminaScaleModule(
        model=model,
        learning_rate=cfg.learning_rate
    )
    print(f"[MAIN] ✓ LuminaScaleModule created")
    
    # 3. Setup Trainer
    print(f"[MAIN] Creating Lightning Trainer...")
    logger_tb = TensorBoardLogger(save_dir=str(cfg.output_dir), name=run_id)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="dequant-{epoch:02d}-{loss:.4f}",
        save_top_k=3,
        monitor="loss", # Adjust based on actual validation metric
        mode="min",
    )

    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.epochs,
        logger=logger_tb,
        callbacks=[checkpoint_callback, DebugCallback()],
        precision=cfg.precision,
        strategy=cfg.strategy,
    )
    print(f"[MAIN] ✓ Lightning Trainer created")

    # 4. Start Training
    print(f"\n{'='*80}")
    print(f"[MAIN] 🚀 Starting WDS Training. Run ID: {run_id}")
    print(f"[MAIN] About to call trainer.fit() with train_loader...")
    print(f"{'='*80}\n")
    
    logger.info(f"🚀 Starting WDS Training. Run ID: {run_id}")
    trainer.fit(ls_module, train_dataloaders=train_loader)
    
    print(f"\n{'='*80}")
    print(f"[MAIN] ✓ Training completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
