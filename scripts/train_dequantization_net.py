"""Hydra-based training script for Dequantization-Net (WebDataset).

Usage (local development):
    python scripts/train_dequantization_net.py --config-name=default

Usage (HPC via Slurm):
    sbatch scripts/train_wds.sh

Override config on CLI:
    python scripts/train_dequantization_net.py \
        batch_size=32 \
        epochs=50
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import cast
import torch.nn as nn

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra
import torch
import numpy as np
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

# Use minimal format and stderr to prevent mixing with stdout progress bar
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    stream=sys.stderr,
    force=True
)

# Suppress INFO/DEBUG logs from luminascale modules
for module in ["luminascale.training.dequantization_trainer", 
               "luminascale.utils.dataset_pair_generator",
               "luminascale.utils.pytorch_aces_transformer",
               "luminascale.data.wds_dataset"]:
    logging.getLogger(module).setLevel(logging.WARNING)

# Also suppress other verbose libraries
for module in ["pytorch_lightning.utilities.rank_zero", 
               "pytorch_lightning.callbacks.progress",
               "webdataset"]:
    logging.getLogger(module).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


import subprocess
from pytorch_lightning.callbacks import TQDMProgressBar, Callback
from luminascale.utils.io import read_exr


class SyntheticInferenceVisualizerCallback(Callback):
    """Runs synthetic inference visualization after each epoch using run_inference.py."""
    
    def __init__(self, width: int = 1024, height: int = 512, base_channels: int = 32):
        """Initialize the visualization callback.
        
        Args:
            width: Width of synthetic gradient
            height: Height of synthetic gradient
            base_channels: Model base channels (must match training model)
        """
        super().__init__()
        self.width = width
        self.height = height
        self.base_channels = base_channels
        self.run_inference_script = project_root / "scripts" / "run_inference.py"
    
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run inference at the start of training."""
        logger.info("Running synthetic inference visualization at training start...")
        self._log_synthetic_inference(trainer, pl_module, step=0, epoch_label="start")
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run inference after each epoch."""
        self._log_synthetic_inference(
            trainer, pl_module, 
            step=trainer.current_epoch + 1, 
            epoch_label=f"epoch_{trainer.current_epoch}"
        )
    
    def _log_synthetic_inference(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        step: int, 
        epoch_label: str
    ) -> None:
        """Run synthetic inference via run_inference.py and log results to TensorBoard.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module with the model
            step: Global step for TensorBoard logging
            epoch_label: Label for the logged images
        """
        if trainer.logger is None or trainer.log_dir is None:
            return
        
        temp_checkpoint: Path | None = None
        try:
            # Save temporary checkpoint for run_inference.py
            temp_checkpoint = Path(trainer.log_dir) / f".temp_checkpoint_{epoch_label}.pt"
            model = cast(nn.Module, pl_module.model)
            torch.save(model.state_dict(), temp_checkpoint)
            logger.info(f"Saved temporary checkpoint to {temp_checkpoint}")
            
            # Prepare output paths
            log_dir_str = trainer.log_dir
            if log_dir_str is None:
                return
            output_exr = Path(log_dir_str) / f"synthetic_{epoch_label}.exr"
            output_png = Path(log_dir_str) / f"synthetic_{epoch_label}.png"
            
            # Run run_inference.py
            cmd = [
                "python",
                str(self.run_inference_script),
                "--checkpoint", str(temp_checkpoint),
                "--synthetic",
                "--width", str(self.width),
                "--height", str(self.height),
                "--channels", str(self.base_channels),
                "--output", str(output_exr),
                "--device", str(pl_module.device).split(":")[0],
            ]
            
            logger.info(f"Running inference: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode != 0:
                logger.error(f"Inference failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return
            
            logger.info(f"Inference stdout: {result.stdout}")
            
            # Read generated outputs
            if output_exr.exists():
                logger.info(f"Output EXR found at {output_exr}")
                output_np = read_exr(output_exr)
                output_tensor = torch.from_numpy(output_np).float()
            else:
                logger.error(f"Output EXR not found: {output_exr}")
                logger.error(f"Contents of log_dir: {list(Path(log_dir_str).iterdir())}")
                return
            
            # For TensorBoard, we also need LDR input and HDR reference
            # We'll regenerate them to get the images for logging (must match what run_inference.py uses)
            from luminascale.utils.image_generator import create_primary_gradients, quantize_to_8bit
            
            target_w = (self.width // 64) * 64
            target_h = (self.height // 64) * 64
            
            hdr = create_primary_gradients(width=target_w, height=target_h, dtype="float32")
            hdr_clipped = np.clip(hdr, 0, 1)
            ldr = quantize_to_8bit(hdr_clipped)
            # quantize_to_8bit returns float32 in [0, 1], so convert directly without dividing by 255
            ldr_tensor = torch.from_numpy(ldr).float()
            hdr_tensor = torch.from_numpy(hdr_clipped).float()
            
            # Clip output to [0, 1] range
            output_clipped = torch.clamp(output_tensor, 0, 1)
            
            # Apply 25x contrast stretch to reveal quantization artifacts
            contrast_factor = 25.0
            def apply_contrast(x):
                return torch.clamp((x - 0.5) * contrast_factor + 0.5, 0, 1)
            
            ldr_tensor_contrast = apply_contrast(ldr_tensor)
            output_tensor_contrast = apply_contrast(output_clipped)
            hdr_tensor_contrast = apply_contrast(hdr_tensor)
            
            # Extract run_id from log_dir path (last component)
            run_id = Path(log_dir_str).name
            
            # Log high-contrast model output to TensorBoard
            # Use fixed tag path so images are consolidated into a scrubbable timeline in TensorBoard
            from pytorch_lightning.loggers.tensorboard import TensorBoardLogger as TBLogger
            if isinstance(trainer.logger, TBLogger):
                tb = trainer.logger.experiment
            else:
                return
            tb.add_image(f"primaries_gradient", output_tensor_contrast, global_step=step)

            tb.flush()
            logger.info(f"✓ Synthetic inference visualization logged: {epoch_label}")
            logger.info(f"  Output saved to: {output_exr} and {output_png}")
            
            # Cleanup only temporary checkpoint (keep EXR and PNG outputs)
            if temp_checkpoint is not None:
                temp_checkpoint.unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"Error during synthetic inference visualization: {e}", exc_info=True)
            # Best effort cleanup
            if temp_checkpoint is not None:
                try:
                    temp_checkpoint.unlink(missing_ok=True)
                except:
                    pass


class TensorBoardFlushCallback(Callback):
    """Callback to explicitly flush TensorBoard logger after each batch and epoch."""
    
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int) -> None:
        """Flush logger after each batch to ensure events are written."""
        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger as TBLogger
        if trainer.logger and isinstance(trainer.logger, TBLogger):
            trainer.logger.experiment.flush()
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Flush logger after each epoch."""
        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger as TBLogger
        if trainer.logger and isinstance(trainer.logger, TBLogger):
            trainer.logger.experiment.flush()


class CompactProgressBar(TQDMProgressBar):
    """Compact progress bar with minimal redundant information."""
    
    def get_metrics(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict:
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
    
    def on_train_start(self, *_) -> None:  
        """Set total batches at training start."""
        super().on_train_start(*_)
        if hasattr(self, 'train_progress_bar') and self.train_progress_bar is not None:
            # trainer is passed as first variadic arg
            trainer = _[0] if _ else None
            if trainer and hasattr(trainer, 'lightning_module'):
                pl_module = trainer.lightning_module
                if hasattr(pl_module, 'estimated_total_batches') and pl_module.estimated_total_batches is not None:
                    self.train_progress_bar.total = pl_module.estimated_total_batches
    
    def on_train_epoch_start(self, trainer: L.Trainer, *_) -> None: 
        """Set description and total for each epoch."""
        super().on_train_epoch_start(trainer, *_)
        if hasattr(self, 'train_progress_bar') and self.train_progress_bar is not None:
            self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
            if hasattr(trainer, 'lightning_module'):
                pl_module = trainer.lightning_module
                if hasattr(pl_module, 'estimated_total_batches') and pl_module.estimated_total_batches is not None:
                    self.train_progress_bar.total = pl_module.estimated_total_batches



@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
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
    model = create_dequantization_net(base_channels=cfg.model.base_channels)
    print(f"[MAIN] ✓ Model created")
    
    print(f"[MAIN] Creating LuminaScaleModule...")
    ls_module = LuminaScaleModule(
        model=model,
        learning_rate=cfg.learning_rate
    )
    # Store estimated batches for progress bar
    estimated_batches = train_dataset.get_estimated_batches()
    if estimated_batches is not None:
        ls_module.estimated_total_batches = estimated_batches
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
        callbacks=[
            checkpoint_callback, 
            CompactProgressBar(), 
            TensorBoardFlushCallback(),
            SyntheticInferenceVisualizerCallback(
                width=cfg.get("vis_width", 512),
                height=cfg.get("vis_height", 512),
                base_channels=cfg.model.base_channels
            ),
        ],
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
