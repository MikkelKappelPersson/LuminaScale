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
import os
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra
import pytorch_lightning as L
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from luminascale.models import create_dequantization_net
from luminascale.training.dequantization_trainer import OnTheFlyBDEDataset, LuminaScaleModule
from luminascale.training.async_prefetch import AsyncDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main training entry point with Hydra config management."""
    # Prevent 'BrokenPipeError'
    from signal import signal, SIGPIPE, SIG_DFL
    try:
        signal(SIGPIPE, SIG_DFL)
    except Exception:
        pass

    # Hydra nesting handling
    if "training" in cfg:
        cfg = cfg.training
    
    # Set OCIO environment variable
    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)
    
    # Run directory setup
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Create dataset
    lmdb_path = Path(cfg.get("lmdb_path")).resolve()
    
    # Get distributed training info from environment (set by SLURM/DDP)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dataset = OnTheFlyBDEDataset(
        lmdb_path=lmdb_path,
        device=None, # Dataset will detect correct GPU per process in DDP
        crop_size=cfg.get("crop_size", 512),
        patches_per_image=cfg.get("patches_per_image", 1),
        rank=rank,
        world_size=world_size,
    )
    
    # Use DistributedSampler to ensure each GPU gets different samples (reduces LMDB contention)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # Each rank gets deterministic subset
        drop_last=False,
    )
    
    # Choose between standard DataLoader and AsyncDataLoader based on config
    use_async_prefetch = cfg.get("use_async_prefetch", False)
    
    if use_async_prefetch:
        # Async prefetch mode: use CPU threads to prefetch LMDB while GPU computes
        # NOTE: Dataset already handles rank-aware partitioning, so we don't use sampler
        logger.info("=" * 80)
        logger.info("ASYNC PREFETCH MODE ENABLED")
        logger.info(f"  num_workers (CPU threads): {cfg.get('prefetch_workers', 4)}")
        logger.info(f"  prefetch_queue_size: {cfg.get('prefetch_queue_size', 3)}")
        logger.info("=" * 80)
        
        # CRITICAL: Lazy-initialize pair_generator by calling __getitem__ once
        # This ensures cdl_processor and pytorch_transformer are created before workers start
        logger.info("Initializing dataset GPU pipeline (one-time setup)...")
        _ = dataset[0]  # Trigger lazy initialization
        logger.info("✓ Dataset GPU pipeline ready")
        
        dataloader = AsyncDataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.get("prefetch_workers", 4),
            prefetch_device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
            queue_size=cfg.get("prefetch_queue_size", 3),
        )
    else:
        # Standard PyTorch DataLoader with DistributedSampler
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=cfg.get("num_workers", 0),
            pin_memory=False,  # Already on GPU, cannot pin CUDA tensors
        )

    # Create model
    raw_model = create_dequantization_net(
        device="cpu", # Lightning handles moving to GPU
        base_channels=cfg.model.base_channels
    )
    
    # Lightning Module
    model_module = LuminaScaleModule(
        model=raw_model,
        learning_rate=cfg.learning_rate,
        vis_freq=cfg.vis_freq if hasattr(cfg, "vis_freq") else 5
    )

    # Loggers and Callbacks
    tb_logger = TensorBoardLogger(
        save_dir=cfg.output_dir,
        name="tensorboard",
        version=run_id
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="dequant-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=cfg.checkpoint_freq,
        save_top_k=-1, # Save all checkpoints according to frequency
    )

    # Trainer Initialization
    trainer = L.Trainer(
        accelerator=cfg.get("accelerator", "gpu"),
        devices=cfg.get("devices", "auto"),
        strategy=cfg.get("strategy", "auto"),  # Will auto-select DDP for multiple GPUs
        precision=cfg.get("precision", 32),
        max_epochs=cfg.epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=run_dir,
        log_every_n_steps=10,  # Reduce progress bar spam (log every 10 steps instead of 1)
        enable_progress_bar=True,
        num_sanity_val_steps=0,  # Skip sanity check for faster startup
    )

    # Train!
    trainer.fit(model=model_module, train_dataloaders=dataloader)
    
    dataset.cleanup()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
