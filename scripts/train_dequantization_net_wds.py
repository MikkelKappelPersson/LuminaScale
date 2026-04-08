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
import warnings
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

from luminascale.models import create_dequantization_net
from luminascale.training.dequantization_trainer import LuminaScaleModule
from luminascale.data.wds_dataset import LuminaScaleWebDataset

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

# Suppress PyTorch Lightning's litlogger tip
_pt_lightning_logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
_pt_lightning_logger.setLevel(logging.ERROR)


from pytorch_lightning.callbacks import TQDMProgressBar, Callback


class TensorBoardFlushCallback(Callback):
    """Callback to explicitly flush TensorBoard logger after each batch and epoch."""
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Flush logger after each batch to ensure events are written."""
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.flush()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Flush logger after each epoch."""
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.flush()


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



@hydra.main(config_path="../configs", config_name="wds", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main training entry point for WebDataset pipeline."""
    
    # Configure torch for optimal performance with Tensor Cores
    torch.set_float32_matmul_precision('high')
    
    print(f"\n{'='*80}")
    print(f"[MAIN] Starting training initialization...")
    print(f"{'='*80}\n")
    
    # Set OCIO environment variable
    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)
    
    # Run directory setup
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[MAIN] Run directory: {run_dir}")
    
    # Log config
    with open(run_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 1. Create WebDataset Loaders
    # On-the-fly patch generation: WebDataset stores 880 unique images once, then .repeat(patches_per_image)
    # loops through the stream to create 880 * 32 = 28,160 total samples. Each time an image appears,
    # a fresh random crop is generated. This matches the OnTheFlyBDEDataset pattern.
    # Total batches = unique_images * patches_per_image / batch_size
    print(f"[MAIN] Creating WebDataset with shard_path: {cfg.get('shard_path')}")
    print(f"[MAIN] batch_size={cfg.get('batch_size', 32)}, num_workers={cfg.get('num_workers', 4)}")
    print(f"[MAIN] patches_per_image={cfg.get('patches_per_image', 1)} (on-the-fly generation via .repeat())")
    
    train_dataset = LuminaScaleWebDataset(
        shard_path=cfg.get("shard_path"),
        batch_size=cfg.get("batch_size", 32),
        shuffle_buffer=cfg.get("shuffle_buffer", 1000),
        is_training=True,
        metadata_parquet=cfg.get("metadata_parquet"),
        patches_per_image=cfg.get("patches_per_image", 1),
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
    # Store estimated batches for progress bar
    ls_module.estimated_total_batches = train_dataset.get_estimated_batches()
    print(f"[MAIN] ✓ LuminaScaleModule created")
    
    # 3. Setup Trainer
    print(f"[MAIN] Creating Lightning Trainer...")
    logger_tb = TensorBoardLogger(
        save_dir=str(run_dir),
        name="",
        version="",
        default_hp_metric=False  # Disable placeholder hp_metric for cleaner TensorBoard display
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="dequant-{epoch:02d}",
        every_n_epochs=cfg.get("checkpoint_freq", 1),
        save_top_k=-1,  # Save all checkpoints according to frequency
    )

    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.epochs,
        logger=logger_tb,
        callbacks=[checkpoint_callback, CompactProgressBar(), TensorBoardFlushCallback()],
        precision=cfg.precision,
        strategy=cfg.strategy,
        log_every_n_steps=1,  # Log every batch for detailed TensorBoard curves
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
