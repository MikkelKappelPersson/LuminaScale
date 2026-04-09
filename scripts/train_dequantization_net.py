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

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra
import torch
import warnings
import subprocess
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from omegaconf import DictConfig, OmegaConf

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

from luminascale.models import create_dequantization_net
from luminascale.training.dequantization_trainer import LuminaScaleModule
from luminascale.data.wds_dataset import LuminaScaleWebDataset
from luminascale.utils.io import read_exr

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
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        
        try:
            # Save temporary checkpoint for run_inference.py
            temp_checkpoint = Path(trainer.log_dir) / f".temp_checkpoint_{epoch_label}.pt"
            torch.save(pl_module.model.state_dict(), temp_checkpoint)
            logger.info(f"Saved temporary checkpoint to {temp_checkpoint}")
            
            # Prepare output paths
            output_exr = Path(trainer.log_dir) / f"synthetic_{epoch_label}.exr"
            output_png = Path(trainer.log_dir) / f"synthetic_{epoch_label}.png"
            
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
                logger.error(f"Contents of log_dir: {list(Path(trainer.log_dir).iterdir())}")
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
            run_id = Path(trainer.log_dir).name
            
            # Log high-contrast model output to TensorBoard
            # Use fixed tag path so images are consolidated into a scrubbable timeline in TensorBoard
            tb = trainer.logger.experiment
            tb.add_image(f"primaries_gradient", output_tensor_contrast, global_step=step)
            tb.flush()
            logger.info(f"✓ Synthetic inference visualization logged: {epoch_label}")
            logger.info(f"  Output saved to: {output_exr} and {output_png}")
            
            # Cleanup only temporary checkpoint (keep EXR and PNG outputs)
            temp_checkpoint.unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"Error during synthetic inference visualization: {e}", exc_info=True)
            # Best effort cleanup
            try:
                temp_checkpoint.unlink(missing_ok=True)
            except:
                pass


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


class HparamsMetricsCallback(Callback):
    """Log final training metrics associated with hyperparameters."""
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log final metrics after training completes."""
        if trainer.logger is None:
            return
        
        try:
            # Extract callbacks to get checkpoint info
            checkpoint_callback = None
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    checkpoint_callback = cb
                    break
            
            # Prepare final metrics dict
            final_metrics = {
                "final_epoch": trainer.current_epoch,
                "total_steps": trainer.global_step,
            }
            
            # If we have checkpoint info, log the best model path
            if checkpoint_callback and checkpoint_callback.best_model_path:
                final_metrics["best_checkpoint"] = str(checkpoint_callback.best_model_path)
            
            # Log final metrics with the hparams already logged
            if hasattr(trainer.logger, "log_hyperparams"):
                trainer.logger.log_hyperparams({}, final_metrics)
                logger.info(f"✓ Final metrics logged: {final_metrics}")
        except Exception as e:
            logger.error(f"Error logging final metrics: {e}")


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
        default_hp_metric=True  # Enable hp_metric to display hyperparameters in TensorBoard
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
            HparamsMetricsCallback(),
            SyntheticInferenceVisualizerCallback(
                width=cfg.get("inference_width", 1024),
                height=cfg.get("inference_height", 512),
                base_channels=cfg.model.base_channels
            )
        ],
        precision=cfg.precision,
        strategy=cfg.strategy,
        log_every_n_steps=1,  # Log every batch for detailed TensorBoard curves
    )
    print(f"[MAIN] ✓ Lightning Trainer created")

    # Log hyperparameters to TensorBoard
    hparams_dict = {
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.get("batch_size", 32),
        "epochs": cfg.epochs,
        "optimizer": "Adam",
        "loss_fn": "L2 (unmasked)",
        "model_base_channels": cfg.model.base_channels,
        "patches_per_image": cfg.get("patches_per_image", 1),
        "crop_size": 512,
    }
    
    # TensorBoard requires log_hyperparams to be called after trainer initialization
    # We'll log initial hparams now; final metrics will be appended after training
    logger_tb.log_hyperparams(hparams_dict)
    print(f"[MAIN] ✓ Hyperparameters logged to TensorBoard")

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
