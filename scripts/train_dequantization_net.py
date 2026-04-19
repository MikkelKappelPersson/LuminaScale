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
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch.nn as nn

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure matplotlib to use non-interactive backend (suppresses Fontconfig warnings in headless envs)
import matplotlib
matplotlib.use('Agg')

import hydra
import torch
import warnings
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback, RichModelSummary, RichProgressBar
from pytorch_lightning.profilers import SimpleProfiler
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

from luminascale.models import create_dequantization_net
from luminascale.training.dequantization_trainer import DequantizationTrainer
from luminascale.training.logger import CustomTensorBoardLogger
from luminascale.data.wds_dataset import LuminaScaleWebDataset
from luminascale.utils.io import read_exr

# Use INFO level to suppress verbose DEBUG/INFO messages
# Set to DEBUG to see per-image timing breakdowns from dataset loading
# Use minimal format and stderr to prevent mixing with stdout progress bar
logging.basicConfig(
    level=logging.INFO,
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
        logger.debug("Running synthetic inference visualization at training start...")
        self._log_synthetic_inference(trainer, pl_module, step=0, epoch_label="start")
        logger.debug("Running real image inference at training start...")
        self._log_real_image_inference(trainer, pl_module, step=0, epoch_label="start")
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run inference every 5 epochs."""
        # Run both synthetic and real image inference every 5 epochs (at epochs 4, 9, 14, ...)
        if trainer.current_epoch % 5 == 4:
            epoch_label = f"epoch_{trainer.current_epoch}"
            self._log_synthetic_inference(
                trainer, pl_module, 
                step=trainer.current_epoch, 
                epoch_label=epoch_label
            )
            self._log_real_image_inference(
                trainer, pl_module,
                step=trainer.current_epoch,
                epoch_label=epoch_label
            )
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run inference at the end of training to capture final model state."""
        logger.debug("Running inference at training end...")
        self._log_synthetic_inference(trainer, pl_module, step=trainer.current_epoch, epoch_label="end")
        self._log_real_image_inference(trainer, pl_module, step=trainer.current_epoch, epoch_label="end")
    
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
        
        # Guard against None log_dir
        if trainer.log_dir is None:
            logger.error("Trainer log_dir is None, cannot save checkpoint")
            return
        
        temp_checkpoint: Path | None = None
        try:
            # Save temporary checkpoint for run_inference.py
            temp_checkpoint = Path(trainer.log_dir) / f".temp_checkpoint_{epoch_label}.pt"
            # Cast model as nn.Module to access state_dict
            model = cast(nn.Module, pl_module.model)
            
            # Unwrap torch.compile wrapper if present (torch.compile adds "_orig_mod" layer)
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            
            torch.save(model.state_dict(), temp_checkpoint)
            logger.debug(f"Saved temporary checkpoint to {temp_checkpoint}")
            
            # Prepare output paths
            output_exr = Path(trainer.log_dir) / f"synthetic_{epoch_label}.exr"
            output_png = Path(trainer.log_dir) / f"synthetic_{epoch_label}.png"
            
            # Run run_inference.py
            cmd = [
                "python",
                str(self.run_inference_script),
                "--checkpoint",
                str(temp_checkpoint),
                "--synthetic",
                "--width",
                str(self.width),
                "--height",
                str(self.height),
                "--channels",
                str(self.base_channels),
                "--output",
                str(output_exr),
                "--device",
                str(str(pl_module.device).split(":")[0]),
                "--apply-contrast-to-output",
            ]
            
            logger.debug(f"Running inference: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode != 0:
                logger.error(f"Inference failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return
            
            logger.debug(f"Inference stdout: {result.stdout}")
            
            tb = trainer.logger.experiment  # type: ignore
            
            # Log the comparison PNG directly from run_inference.py (3x3 grid with all analysis)
            if output_png.exists():
                logger.debug(f"Logging comparison grid from {output_png}")
                try:
                    from PIL import Image as PILImage
                    import matplotlib.pyplot as plt
                    
                    img = PILImage.open(output_png)
                    fig = plt.figure(figsize=(14, 12))
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    
                    tb.add_figure("visualization/synthetic", fig, global_step=step)
                    tb.flush()
                    plt.close(fig)
                    logger.debug(f"✓ Logged visualization/synthetic")
                except Exception as e:
                    logger.error(f"Failed to log comparison PNG: {e}", exc_info=True)
            else:
                logger.warning(f"Comparison PNG not found: {output_png}")
            
            # Read and log the EXR output as a high-bit image
            if output_exr.exists():
                logger.debug(f"Output EXR found at {output_exr}")
                output_np = read_exr(output_exr)
                output_tensor = torch.from_numpy(output_np).float()
                
                # Log the model output as an image (normalized to [0, 1])
                output_clipped = torch.clamp(output_tensor, 0, 1)
                tb.add_image(f"inference/synthetic", output_clipped, global_step=step)
                tb.flush()
                logger.debug(f"✓ Logged inference/synthetic")
            else:
                logger.error(f"Output EXR not found: {output_exr}")
                logger.error(f"Contents of log_dir: {list(Path(trainer.log_dir).iterdir())}")
                return
            
            # Cleanup only temporary checkpoint (keep EXR and PNG outputs)
            if temp_checkpoint and temp_checkpoint.exists():
                temp_checkpoint.unlink()
            logger.debug(f"✓ Synthetic inference visualization logged: {epoch_label}")
            logger.debug(f"  - Comparison grid (PNG): {output_png}")
            logger.debug(f"  - Model output (EXR): {output_exr}")
        
        except Exception as e:
            logger.error(f"Error during synthetic inference visualization: {e}", exc_info=True)
            # Best effort cleanup
            if temp_checkpoint is not None:
                try:
                    temp_checkpoint.unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass

    def _log_real_image_inference(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        step: int,
        epoch_label: str
    ) -> None:
        """Run inference on real test images and log outputs to TensorBoard.
        
        Processes three reference images from assets folder and logs only EXR outputs.
        Run every 5 epochs (plus at training start).
        
        Images:
        - assets/grinder_01.jpg
        - assets/mountains_01.jpg
        - assets/woods_1.jpg
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module with the model
            step: Global step for TensorBoard logging
            epoch_label: Label for the logged images
        """
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        
        if trainer.log_dir is None:
            logger.error("Trainer log_dir is None, cannot run real image inference")
            return
        
        test_images = [
            project_root / "assets" / "grinder_01.jpg",
            project_root / "assets" / "mountains_01.jpg",
            project_root / "assets" / "woods_1.jpg",
        ]
        
        temp_checkpoint: Path | None = None
        try:
            # Save temporary checkpoint for run_inference.py (reuse for all images)
            temp_checkpoint = Path(trainer.log_dir) / f".temp_checkpoint_real_{epoch_label}.pt"
            model = cast(nn.Module, pl_module.model)
            
            # Unwrap torch.compile wrapper if present (torch.compile adds "_orig_mod" layer)
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            
            torch.save(model.state_dict(), temp_checkpoint)
            
            tb = trainer.logger.experiment  # type: ignore
            
            # Process each test image
            for test_image in test_images:
                if not test_image.exists():
                    logger.warning(f"Test image not found: {test_image}")
                    continue
                
                # Use image name (without .jpg) as identifier
                image_name = test_image.stem
                output_exr = Path(trainer.log_dir) / f"{image_name}_{epoch_label}.exr"
                
                # Run run_inference.py (no PNG comparison needed)
                cmd = [
                    "python",
                    str(self.run_inference_script),
                    "--checkpoint",
                    str(temp_checkpoint),
                    "--input",
                    str(test_image),
                    "--output",
                    str(output_exr),
                    "--device",
                    str(str(pl_module.device).split(":")[0]),
                ]
                
                logger.debug(f"Running real image inference on {image_name}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
                
                if result.returncode != 0:
                    logger.error(f"Real image inference on {image_name} failed with return code {result.returncode}")
                    logger.error(f"STDERR: {result.stderr}")
                    continue
                
                logger.debug(f"Real image inference on {image_name} stdout: {result.stdout}")
                
                # Read and log the EXR output only
                if output_exr.exists():
                    logger.debug(f"Output EXR found: {output_exr}")
                    output_np = read_exr(output_exr)
                    output_tensor = torch.from_numpy(output_np).float()
                    
                    output_clipped = torch.clamp(output_tensor, 0, 1)
                    tb.add_image(f"inference/{image_name}", output_clipped, global_step=step)
                    tb.flush()
                    logger.debug(f"✓ Logged inference/{image_name}")
                else:
                    logger.warning(f"Output EXR not found: {output_exr}")
            
            # Cleanup temporary checkpoint
            if temp_checkpoint and temp_checkpoint.exists():
                temp_checkpoint.unlink()
            logger.debug(f"✓ Real image inference logged: {epoch_label} (3 images)")
        
        except Exception as e:
            logger.error(f"Error during real image inference: {e}", exc_info=True)
            # Best effort cleanup
            if temp_checkpoint is not None:
                try:
                    temp_checkpoint.unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass


class TensorBoardFlushCallback(Callback):
    """Callback to explicitly flush TensorBoard logger after each batch and epoch."""
    
    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
        outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Flush logger after each batch to ensure events are written."""
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.flush()  # type: ignore
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Flush logger after each epoch."""
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.flush()  # type: ignore


class HparamsMetricsCallback(Callback):
    """Log hparams at training start, PSNR metric at epoch end.
    
    Uses CustomTensorBoardLogger's log_hyperparams_metrics to ensure proper timing.
    
    Metric Strategy:
    - Uses PSNR (Peak Signal-to-Noise Ratio) for weight-independent metric logging
    - PSNR is invariant to loss weight changes (unlike weighted loss values)
    - Falls back to final_loss if PSNR unavailable
    - Perfect for fair hyperparameter comparison across different weight configurations
    
    Timeline:
    1. on_fit_start: Log hparams with initial empty metrics
    2. on_train_epoch_end: Log hparams + updated PSNR after each epoch
    3. on_train_end: Log hparams + final PSNR at training completion
    """
    
    def __init__(self, hparams_dict: dict[str, Any]) -> None:
        super().__init__()
        self.hparams_dict = hparams_dict
        self.final_loss: float | None = None
    
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log hyperparameters at the very start of training."""
        if trainer.logger is None:
            return
        
        try:
            # Log hparams with empty metrics dict initially
            if hasattr(trainer.logger, "log_hyperparams_metrics"):
                trainer.logger.log_hyperparams_metrics(self.hparams_dict, {})
                logger.debug(f"✓ Hyperparameters logged at fit_start:")
                for k, v in self.hparams_dict.items():
                    logger.debug(f"    {k}: {v}")
            else:
                logger.warning("Logger does not have log_hyperparams_metrics method")
        except Exception as e:
            logger.error(f"Error logging hyperparameters at fit_start: {e}", exc_info=True)
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log hparams + metrics after each training epoch."""
        if trainer.logger is None:
            return
        
        try:
            # Capture the epoch's metrics (PSNR or fallback loss)
            metrics_dict = self._get_metrics_dict(trainer)
            
            if metrics_dict and hasattr(trainer.logger, "log_hyperparams_metrics"):
                trainer.logger.log_hyperparams_metrics(self.hparams_dict, metrics_dict)
                # Display both PSNR and loss if available
                psnr = metrics_dict.get("psnr_db")
                final_loss = metrics_dict.get("final_loss")
                if psnr is not None:
                    logger.debug(f"  [Epoch {trainer.current_epoch}] PSNR: {psnr:.2f} dB")
                elif final_loss is not None:
                    logger.debug(f"  [Epoch {trainer.current_epoch}] Loss: {final_loss:.6f}")
        except Exception as e:
            logger.error(f"Error logging metrics at epoch end: {e}", exc_info=True)
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log final hparams + metrics when training completes."""
        if trainer.logger is None:
            return
        
        try:
            # Capture final metrics from training completion
            metrics_dict = self._get_metrics_dict(trainer)
            
            if metrics_dict and hasattr(trainer.logger, "log_hyperparams_metrics"):
                trainer.logger.log_hyperparams_metrics(self.hparams_dict, metrics_dict)
                psnr = metrics_dict.get("psnr_db")
                final_loss = metrics_dict.get("final_loss")
                if psnr is not None:
                    logger.debug(f"✓ Training complete. PSNR: {psnr:.2f} dB")
                elif final_loss is not None:
                    logger.debug(f"✓ Training complete. Final loss: {final_loss:.6f}")
            
            # Log best checkpoint info
            checkpoint_callback: ModelCheckpoint | None = None
            for cb in trainer.callbacks:  # type: ignore
                if isinstance(cb, ModelCheckpoint):
                    checkpoint_callback = cb
                    break
            
            if checkpoint_callback and checkpoint_callback.best_model_path:
                logger.debug(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
        except Exception as e:
            logger.error(f"Error logging final metrics: {e}", exc_info=True)
    
    def _get_metrics_dict(self, trainer: L.Trainer) -> dict[str, Any]:
        """Extract metrics from validation data for hparams logging.
        
        Extracts quality metrics (PSNR, SSIM, ΔE) computed on validation set only:
        - PSNR (Peak Signal-to-Noise Ratio): Reconstruction quality [dB]
        - SSIM (Structural Similarity): Structure preservation [dB]
        - ΔE (Delta-E CIE): Perceptual color accuracy [0-∞)
        
        These validation metrics are:
        - Weight-independent (PSNR, unlike weighted loss)
        - Unbiased (computed on unseen validation data, not training data)
        - Suitable for hyperparameter tuning and model selection
        
        Also logs throughput (samples/sec) for data pipeline optimization.
        
        Falls back to loss if validation metrics unavailable.
        """
        metrics_dict: dict[str, Any] = {}
        
        # Extract quality metrics from VALIDATION set (preferred for hparam tuning)
        metric_keys = [
            "metric_psnr/val",      # Weight-independent, unbiased quality metric
            "metric_loss/val",      # Fallback: validation loss
            "loss_total/train",     # Final fallback: training loss (less reliable)
        ]
        
        for metric_key in metric_keys:
            if metric_key in trainer.callback_metrics:
                value = trainer.callback_metrics[metric_key]
                value_float = value.item() if hasattr(value, "item") else float(value)
                # PSNR is in dB; loss is just numeric loss
                metric_name = "psnr_db" if "psnr" in metric_key else "final_loss"
                metrics_dict[metric_name] = value_float
                break
        
        # Add additional quality metrics from validation
        if "metric_ssim/val" in trainer.callback_metrics:
            ssim = trainer.callback_metrics["metric_ssim/val"]
            metrics_dict["ssim_db"] = ssim.item() if hasattr(ssim, "item") else float(ssim)
        
        if "metric_delta_e/val" in trainer.callback_metrics:
            delta_e = trainer.callback_metrics["metric_delta_e/val"]
            metrics_dict["delta_e"] = delta_e.item() if hasattr(delta_e, "item") else float(delta_e)
        
        # Add throughput metric for data pipeline optimization
        if hasattr(trainer.lightning_module, 'last_epoch_throughput_samples_per_sec'):
            throughput = trainer.lightning_module.last_epoch_throughput_samples_per_sec
            if throughput is not None:
                metrics_dict["throughput_samples_per_sec"] = throughput
        
        return metrics_dict


class CustomRichProgressBar(RichProgressBar):
    """Rich progress bar with custom batch metrics (GPU time and loss).
    
    Shows samples/sec (normalized by batch_size) instead of batches/sec for fair comparison.
    """
    
    def get_metrics(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict[str, Any]:
        """Override to inject batch-specific metrics and show samples/sec."""
        items = super().get_metrics(trainer, pl_module)
        # Remove redundant info
        items.pop("v_num", None)
        
        # Calculate samples/sec from raw speed
        batch_size = getattr(pl_module, 'batch_size', 1)
        if self.progress is not None:
            try:
                for task_id in self.progress.task_ids:
                    task = self.progress._tasks[task_id]
                    # task.speed is in iterations/sec (batches/sec) 
                    if task.speed is not None and isinstance(task.speed, (int, float)) and task.speed > 0:
                        samples_per_sec = task.speed * batch_size
                        # Override the speed display in items
                        items["samples/s"] = f"{samples_per_sec:.2f}"
                        # Also keep the batch speed for reference
                        break
            except Exception:
                pass  # Gracefully fall back if accessing task fails
        
        # Inject batch GPU time and loss from module if available
        if hasattr(pl_module, 'last_batch_gpu_ms') and pl_module.last_batch_gpu_ms is not None:
            items["GPU"] = f"{pl_module.last_batch_gpu_ms:.1f}ms"
        if hasattr(pl_module, 'last_batch_loss') and pl_module.last_batch_loss is not None:
            items["Loss"] = f"{pl_module.last_batch_loss:.4f}"
        
        return items
    
    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Override epoch start to set estimated total batches."""
        super().on_train_epoch_start(trainer, pl_module)
        
        # Set estimated total batches from metadata if available
        if hasattr(pl_module, 'estimated_total_batches') and pl_module.estimated_total_batches is not None:
            # Access the rich progress bar and set the total
            if hasattr(self, 'progress') and self.progress is not None:
                # For RichProgressBar, we need to update the task total
                for task_id in self.progress.task_ids:
                    task = self.progress._tasks[task_id]
                    # Check if this is the training task (usually matches the epoch description)
                    if "Epoch" in str(task.description) or task.total is None or task.total == 0:
                        self.progress.update(task_id, total=pl_module.estimated_total_batches)



@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main training entry point for WebDataset pipeline."""
    
    # Import os (needed for OCIO environment setup and CPU affinity)
    import os
    
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
    print(f"[MAIN] Run ID: {run_id}")

    # 1. Create WebDataset Loaders
    # WebDataset streams unique images from shards, generating random crops on-the-fly
    # Each image appears once; random crops are generated per batch iteration
    print(f"[MAIN] Creating WebDataset with shard_path: {cfg.get('shard_path')}")
    print(f"[MAIN] batch_size={cfg.get('batch_size', 32)}, num_workers={cfg.get('num_workers', 4)}")
    
    train_dataset = LuminaScaleWebDataset(
        shard_path=cfg.get("shard_path"),
        batch_size=cfg.get("batch_size", 32),
        shuffle_buffer=cfg.get("shuffle_buffer", 1000),
        is_training=True,
        metadata_parquet=cfg.get("metadata_parquet"),
    )
    print(f"[MAIN] ✓ WebDataset created")
    
    print(f"[MAIN] Getting WebLoader...")
    prefetch_factor = cfg.get("prefetch_size", 2)  # Reuse prefetch_size config as prefetch_factor
    train_loader = train_dataset.get_loader(
        num_workers=cfg.get("num_workers", 4),
        prefetch_factor=prefetch_factor
    )
    print(f"[MAIN] ✓ WebLoader created (num_workers={cfg.get('num_workers', 4)}, prefetch_factor={prefetch_factor})")
    
    # NOTE: Multiprocessing-based async prefetch doesn't work with WebDataset workers due to
    # "daemonic processes are not allowed to have children" limitation. WebDataset's native
    # num_workers + persistent_workers already provides good I/O efficiency.
    # Best config: batch_size=4, num_workers=2, crop_size=1024, precision="16-mixed" = 7.20 samples/sec
    
    # Create validation dataloader if val_shard_path is specified (best practice for monitoring)
    val_loader = None
    val_shard_path = cfg.get("val_shard_path")
    if val_shard_path:
        print(f"[MAIN] Creating validation WebDataset with shard_path: {val_shard_path}")
        val_dataset = LuminaScaleWebDataset(
            shard_path=val_shard_path,
            batch_size=cfg.get("batch_size", 32),
            shuffle_buffer=cfg.get("shuffle_buffer", 1000),
            is_training=False,  # No shuffling for validation
            metadata_parquet=cfg.get("metadata_parquet"),
        )
        print(f"[MAIN] ✓ Validation WebDataset created")
        
        print(f"[MAIN] Getting validation WebLoader...")
        val_loader = val_dataset.get_loader(
            num_workers=cfg.get("num_workers", 4),
            prefetch_factor=prefetch_factor
        )
        print(f"[MAIN] ✓ Validation WebLoader created")
    else:
        print(f"[MAIN] ⚠ val_shard_path not specified; skipping validation")
    
    # 2. Setup Lightning Module
    print(f"[MAIN] Creating model...")
    model = create_dequantization_net(in_channels=3, base_channels=cfg.model.base_channels)  # type: ignore
    print(f"[MAIN] ✓ Model created")
    
    # Move model to CUDA (we always train on GPUs)
    device = torch.device("cuda")
    model = model.to(device)
    print(f"[MAIN] Model moved to {device}")
    
    # Compile the model for performance (PyTorch 2.0+)
    # See: https://lightning.ai/docs/pytorch/2.5.5/advanced/compile.html
    if cfg.get("enable_compile", True):
        print(f"[MAIN] Compiling model with torch.compile (mode='reduce-overhead' for CUDA Graphs)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[MAIN] ✓ Model compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}. Continuing with uncompiled model.")
            print(f"[MAIN] ⚠ Compilation failed: {e}. Proceeding without compilation.")
    else:
        print(f"[MAIN] Model compilation disabled (enable_compile=False)")
    
    print(f"[MAIN] Creating DequantizationTrainer...")
    ls_module = DequantizationTrainer(
        model=model,
        learning_rate=cfg.learning_rate,
        loss_weights=dict(cfg.get("loss", {})),
        crop_size=cfg.get("crop_size", 512),
        val_crop_size=cfg.get("val_crop_size"),  # Can differ from training crop size
        batch_size=cfg.batch_size,
        num_workers=cfg.get("num_workers", 2),
        precision=cfg.get("precision", "32"),
        enable_profiling=cfg.get("enable_profiling", False),  # Disable CUDA sync by default for speed
        bit_crunch_contrast_min=cfg.get("bit_crunch_contrast_min", 1.0),
        bit_crunch_contrast_max=cfg.get("bit_crunch_contrast_max", 1.0),
    )
    # Store estimated batches for progress bar
    ls_module.estimated_total_batches = train_dataset.get_estimated_batches()  # type: ignore
    print(f"[MAIN] ✓ DequantizationTrainer created")
    
    # Prepare hparams dict for logging at training end
    config_name = cfg.get("config_name", "default")
    
    # Extract loss weights from config
    loss_cfg = cfg.get("loss", {})
    l1_weight = loss_cfg.get("l1_weight", 1.0)
    l2_weight = loss_cfg.get("l2_weight", 0.0)  # Currently unused, but available for future use
    charbonnier_weight = loss_cfg.get("charbonnier_weight", 0.05)
    grad_match_weight = loss_cfg.get("grad_match_weight", 0.5)
    
    # Create dynamic loss_fn string showing the actual formula with weights
    loss_fn_str = (
        f"L1*{l1_weight} + L2*{l2_weight} + "
        f"Charbonnier*{charbonnier_weight} + EdgeAware*{grad_match_weight}"
    )

    # Append a sanitized version of the loss formula to the TensorBoard/run directory name
    run_dir_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", loss_fn_str).strip("_")
    run_dir = Path(cfg.output_dir).resolve() / f"{run_id}_{run_dir_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MAIN] Run directory: {run_dir}")

    # Log config
    with open(run_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 3. Setup Trainer
    print(f"[MAIN] Creating Lightning Trainer...")
    logger_tb = CustomTensorBoardLogger(
        save_dir=str(run_dir),
        name="",
        version="",
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="dequant-{epoch:02d}",
        every_n_epochs=cfg.get("checkpoint_freq", 1),
        save_top_k=-1,  # Save all checkpoints according to frequency
    )
    
    # Create dynamic optimizer string showing the actual optimizer and learning rate
    optimizer_str = f"Adam(lr={cfg.learning_rate})"
    
    # Create dynamic scheduler string showing the actual scheduler and parameters
    num_epochs = cfg.epochs
    eta_min = 1e-6
    scheduler_str = f"CosineAnnealingLR(T_max={num_epochs}, eta_min={eta_min})"
    
    hparams_dict = {
        "config_name": config_name,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.get("batch_size", 32),
        "epochs": cfg.epochs,
        "optimizer": optimizer_str,
        "scheduler": scheduler_str,
        "loss_fn": loss_fn_str,
        "weight_l1": l1_weight,
        "weight_l2": l2_weight,
        "weight_charbonnier": charbonnier_weight,
        "weight_grad_match": grad_match_weight,
        "model_base_channels": cfg.model.base_channels,
        "crop_size": 512,
        "shuffle_buffer": cfg.get("shuffle_buffer", 10),
        "precision": cfg.precision,
    }

    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.epochs,
        num_sanity_val_steps=0,  # Skip sanity check for faster startup
        logger=logger_tb,
        callbacks=[
            checkpoint_callback,
            RichModelSummary(max_depth=2),
            CustomRichProgressBar(refresh_rate=100, leave=True),
            TensorBoardFlushCallback(),
            HparamsMetricsCallback(hparams_dict),
            SyntheticInferenceVisualizerCallback(
                width=cfg.get("inference_width", 128),
                height=cfg.get("inference_height", 64),
                base_channels=cfg.model.base_channels
            )
        ],
        profiler=SimpleProfiler(
            filename="training_profile.txt",
            extended=True,
        ),
        precision=cfg.precision,
        strategy=cfg.strategy,
        log_every_n_steps=1,  # Log every batch for detailed TensorBoard curves
    )
    print(f"[MAIN] ✓ Lightning Trainer created")

    # Note: CustomTensorBoardLogger disables early automatic log_hyperparams call
    # HparamsMetricsCallback logs hparams at fit_start and final_loss at epoch_end
    # This ensures hparams are recorded correctly in TensorBoard

    # 4. Start Training
    print(f"\n{'='*80}")
    print(f"[MAIN] 🚀 Starting WDS Training. Run ID: {run_id}")
    print(f"[MAIN] About to call trainer.fit() with train_loader...")
    print(f"{'='*80}\n")
    
    logger.debug(f"🚀 Starting WDS Training. Run ID: {run_id}")
    
    # Resume from checkpoint if specified
    resume_ckpt_path = cfg.get("resume_ckpt_path", None)
    if resume_ckpt_path:
        print(f"[MAIN] Resuming from checkpoint: {resume_ckpt_path}")
        logger.debug(f"Resuming from checkpoint: {resume_ckpt_path}")
    
    trainer.fit(
        ls_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,  # Validation during training (if configured)
        ckpt_path=resume_ckpt_path
    )
    
    print(f"\n{'='*80}")
    print(f"[MAIN] ✓ Training completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
