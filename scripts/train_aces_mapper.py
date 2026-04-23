"""Hydra-based training script for ACESMapper (WebDataset).

Usage (local development):
    python scripts/train_aces_mapper.py --config-name=mapper

Usage (HPC via Slurm):
    sbatch scripts/train_aces_mapper.sh
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from aim.pytorch_lightning import AimLogger
from omegaconf import DictConfig, OmegaConf
import hydra

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from luminascale.training.aces_trainer import ACESMapperTrainer
from luminascale.data.wds_dataset import LuminaScaleWebDataset

# Register resolvers for OmegaConf
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/task", config_name="mapper", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # 1. Performance Optimizations
    torch.set_float32_matmul_precision('high')
    
    # 2. Precision handling
    precision = cfg.get("precision", "16-mixed")
    
    # Set OCIO environment if needed
    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)

    print(f"\n{'='*80}")
    print(f"[MAIN] Starting ACESMapper Training Initialization...")
    print(f"{'='*80}\n")
    
    # 2. Setup Data Module (WebDataset)
    train_dataset = LuminaScaleWebDataset(
        shard_path=cfg.get("shard_path"),
        batch_size=cfg.get("batch_size", 4),
        shuffle_buffer=cfg.get("shuffle_buffer", 100),
        is_training=True,
    )
    
    train_loader = train_dataset.get_loader(
        num_workers=cfg.get("num_workers", 4),
        prefetch_factor=cfg.get("prefetch_size", 2)
    )

    val_loader = None
    if cfg.get("val_shard_path"):
        val_dataset = LuminaScaleWebDataset(
            shard_path=cfg.get("val_shard_path"),
            batch_size=cfg.get("batch_size", 4),
            is_training=False,
        )
        val_loader = val_dataset.get_loader(num_workers=cfg.get("num_workers", 2))

    # 3. Setup Lightning Module
    # We use the params from the config to initialize the trainer which holds the ACESMapper model
    trainer_module = ACESMapperTrainer(
        num_luts=cfg.model.params.num_luts,
        lut_dim=cfg.model.params.lut_dim,
        num_lap=cfg.model.params.num_lap,
        num_residual_blocks=cfg.model.params.num_residual_blocks,
        lr=cfg.trainer.params.lr,
        weight_decay=cfg.trainer.params.weight_decay,
        crop_size=cfg.get("crop_size", 512)
    )

    # 4. Logger & Callbacks
    aim_logger = AimLogger(
        experiment=cfg.task_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="aces-mapper-{epoch:02d}-{val_loss:.4f}",
            monitor="val/loss" if val_loader else "train/loss_total",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=2),
        RichProgressBar(),
    ]

    # 5. Trainer setup
    trainer = L.Trainer(
        max_epochs=cfg.get("epochs", 100),
        accelerator="gpu",
        devices=1,  # Adjust for multi-GPU if needed
        logger=aim_logger,
        callbacks=callbacks,
        precision=precision, 
        gradient_clip_val=1.0,
    )

    # 6. Start Training
    print(f"[MAIN] Starting fit...")
    trainer.fit(trainer_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
