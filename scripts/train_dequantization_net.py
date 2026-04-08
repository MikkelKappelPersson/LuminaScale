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
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf

from luminascale.models import create_dequantization_net
from luminascale.training.dequantization_trainer import OnTheFlyBDEDataset, LuminaScaleModule

# Enable DEBUG to see per-image timing breakdowns from dataset loading
# Use minimal format and stderr to prevent mixing with stdout progress bar
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    stream=sys.stderr,
    force=True
)
# Configure all luminascale loggers to use minimal format
for logger_name in logging.Logger.manager.loggerDict:
    if isinstance(logging.Logger.manager.loggerDict[logger_name], logging.Logger):
        logger_obj = logging.getLogger(logger_name)
        for handler in logger_obj.handlers:
            handler.setFormatter(logging.Formatter("%(message)s"))
logger = logging.getLogger(__name__)


class CompactProgressBar(TQDMProgressBar):
    """Compact progress bar with minimal redundant information."""
    
    def get_metrics(self, trainer, pl_module):
        """Override to inject batch-specific metrics into progress bar."""
        items = super().get_metrics(trainer, pl_module)
        # Remove redundant info
        items.pop("v_num", None)
        items.pop("train_loss", None)
        
        # Inject batch GPU time and loss from module if available
        if hasattr(pl_module, 'last_batch_gpu_ms') and pl_module.last_batch_gpu_ms is not None:
            items[f"GPU"] = f"{pl_module.last_batch_gpu_ms:.1f}ms"
        if hasattr(pl_module, 'last_batch_loss') and pl_module.last_batch_loss is not None:
            items[f"Loss"] = f"{pl_module.last_batch_loss:.4f}"
        
        return items
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Override epoch start to use compact format and set estimated total."""
        super().on_train_epoch_start(trainer, pl_module)
        # Customize the progress bar description to be more compact
        if hasattr(self, 'train_progress_bar') and self.train_progress_bar is not None:
            self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
            # Set estimated total batches from metadata if available
            if hasattr(pl_module, 'estimated_total_batches') and pl_module.estimated_total_batches is not None:
                self.train_progress_bar.total = pl_module.estimated_total_batches


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
    
    # DataLoader with num_workers=0 for DDP stability
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
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
        callbacks=[checkpoint_callback, CompactProgressBar()],
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
