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
from typing import Any, cast

import torch.nn as nn

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
import matplotlib.pyplot as plt
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

from luminascale.models import create_dequantization_net
from luminascale.training.dequantization_trainer import DequantizationTrainer
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
            torch.save(model.state_dict(), temp_checkpoint)
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
            tb = trainer.logger.experiment  # type: ignore
            tb.add_image(f"primaries_gradient/eval", output_tensor_contrast, global_step=step)
            
            # Create a combined visualization with both normal and contrast views
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Convert tensors to numpy for plotting [H, W, C]
            ldr_np = ldr_tensor.permute(1, 2, 0).numpy()
            output_np = output_clipped.permute(1, 2, 0).numpy()
            hdr_np = hdr_tensor.permute(1, 2, 0).numpy()
            
            ldr_contrast_np = ldr_tensor_contrast.permute(1, 2, 0).numpy()
            output_contrast_np = output_tensor_contrast.permute(1, 2, 0).numpy()
            hdr_contrast_np = hdr_tensor_contrast.permute(1, 2, 0).numpy()
            
            # Row 0: Normal view
            axes[0, 0].imshow(ldr_np)
            axes[0, 0].set_title("Input (8-bit LDR)")
            axes[0, 1].imshow(output_np)
            axes[0, 1].set_title("Output (Expanded)")
            axes[0, 2].imshow(hdr_np)
            axes[0, 2].set_title("Target (32-bit HDR)")
            
            # Row 1: Contrast-boosted view (25x)
            axes[1, 0].imshow(ldr_contrast_np)
            axes[1, 0].set_title(f"Input {contrast_factor}x Contrast")
            axes[1, 1].imshow(output_contrast_np)
            axes[1, 1].set_title(f"Output {contrast_factor}x Contrast")
            axes[1, 2].imshow(hdr_contrast_np)
            axes[1, 2].set_title(f"Target {contrast_factor}x Contrast")
            
            for ax in axes.ravel():
                ax.axis("off")
            
            plt.tight_layout()
            tb.add_figure(f"primaries_visualisation/eval", fig, global_step=step)
            plt.close(fig)
            
            tb.flush()
            logger.info(f"✓ Synthetic inference visualization logged: {epoch_label}")
            logger.info(f"  Output saved to: {output_exr} and {output_png}")
            
            # Cleanup only temporary checkpoint (keep EXR and PNG outputs)
            temp_checkpoint.unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"Error during synthetic inference visualization: {e}", exc_info=True)
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
    """Log final_loss metric after training completes.
    
    Adds final_loss to the hparams metrics so TensorBoard can correlate
    hyperparameters with final training performance.
    """
    
    def __init__(self, hparams_dict: dict[str, Any]) -> None:
        super().__init__()
        self.hparams_dict = hparams_dict
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log hparams + final_loss metric after training completes."""
        if trainer.logger is None:
            return
        
        try:
            # Extract final loss from logged metrics
            final_loss = trainer.callback_metrics.get("loss_L2/train", None)
            
            # Add final_loss to hparams metrics
            metrics_dict: dict[str, Any] = {}
            if final_loss is not None:
                loss_value = final_loss.item() if hasattr(final_loss, "item") else float(final_loss)
                metrics_dict["final_loss"] = loss_value
            
            # Log hparams + metrics together for TensorBoard hparams dashboard
            if hasattr(trainer.logger, "log_hyperparams"):
                trainer.logger.log_hyperparams(self.hparams_dict, metrics_dict)
                logger.info(f"✓ Hparams + final_loss logged: {metrics_dict.get('final_loss', 'N/A'):.6f}")
            
            # Log best checkpoint info
            checkpoint_callback: ModelCheckpoint | None = None
            for cb in trainer.callbacks:  # type: ignore
                if isinstance(cb, ModelCheckpoint):
                    checkpoint_callback = cb
                    break
            
            if checkpoint_callback and checkpoint_callback.best_model_path:
                logger.info(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
        except Exception as e:
            logger.error(f"Error logging final metrics: {e}")


class CompactProgressBar(TQDMProgressBar):
    """Compact progress bar with minimal redundant information."""
    
    def get_metrics(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict[str, Any]:
        """Override to inject batch-specific metrics into progress bar."""
        items = super().get_metrics(trainer, pl_module)
        # Remove redundant info
        items.pop("v_num", None)
        items.pop("train_loss", None)
        
        # Inject batch GPU time and loss from module if available
        if hasattr(pl_module, 'last_batch_gpu_ms') and pl_module.last_batch_gpu_ms is not None:
            items["GPU"] = f"{pl_module.last_batch_gpu_ms:.1f}ms"
        if hasattr(pl_module, 'last_batch_loss') and pl_module.last_batch_loss is not None:
            items["Loss"] = f"{pl_module.last_batch_loss:.4f}"
        
        return items
    
    def on_train_epoch_start(
        self, trainer: L.Trainer, *args: Any
    ) -> None:
        """Override epoch start to use compact format and set estimated total."""
        super().on_train_epoch_start(trainer)
        # Customize the progress bar description to be more compact
        if hasattr(self, 'train_progress_bar') and self.train_progress_bar is not None:
            self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
            # Set estimated total batches from metadata if available
            # Extract pl_module from args if available (PyTorch Lightning may pass it)
            if len(args) > 0:
                pl_module = args[0]
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
    model = create_dequantization_net(in_channels=3, base_channels=cfg.model.base_channels)  # type: ignore
    print(f"[MAIN] ✓ Model created")
    
    # Move model to CUDA (we always train on GPUs)
    device = torch.device("cuda")
    model = model.to(device)
    print(f"[MAIN] Model moved to {device}")
    
    print(f"[MAIN] Creating DequantizationTrainer...")
    ls_module = DequantizationTrainer(
        model=model,
        learning_rate=cfg.learning_rate,
        loss_weights=dict(cfg.get("loss", {})),
    )
    # Store estimated batches for progress bar
    ls_module.estimated_total_batches = train_dataset.get_estimated_batches()  # type: ignore
    print(f"[MAIN] ✓ DequantizationTrainer created")
    
    # 3. Setup Trainer
    print(f"[MAIN] Creating Lightning Trainer...")
    logger_tb = TensorBoardLogger(
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

    # Prepare hparams dict for logging at training end
    config_name = cfg.get("config_name", "default")
    
    # Extract loss weights from config
    loss_cfg = cfg.get("loss", {})
    l1_weight = loss_cfg.get("l1_weight", 1.0)
    charbonnier_weight = loss_cfg.get("tv_weight", 0.05)  # Note: "tv_weight" key maps to charbonnier
    grad_match_weight = loss_cfg.get("grad_match_weight", 0.5)
    
    hparams_dict = {
        "config_name": config_name,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.get("batch_size", 32),
        "epochs": cfg.epochs,
        "optimizer": "Adam",
        "loss_fn": "L1 + Charbonnier + EdgeAware",
        "l1_weight": l1_weight,
        "charbonnier_weight": charbonnier_weight,
        "grad_match_weight": grad_match_weight,
        "model_base_channels": cfg.model.base_channels,
        "patches_per_image": cfg.get("patches_per_image", 1),
        "crop_size": 512,
        "shuffle_buffer": cfg.get("shuffle_buffer", 10),
        "precision": cfg.precision,
    }

    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.epochs,
        logger=logger_tb,
        callbacks=[
            checkpoint_callback,
            CompactProgressBar(),
            TensorBoardFlushCallback(),
            HparamsMetricsCallback(hparams_dict),
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

    # Note: hparams + final_loss will be logged together at training end in HparamsMetricsCallback
    # (TensorBoard requires a single log_hyperparams call with both params and metrics)

    # 4. Start Training
    print(f"\n{'='*80}")
    print(f"[MAIN] 🚀 Starting WDS Training. Run ID: {run_id}")
    print(f"[MAIN] About to call trainer.fit() with train_loader...")
    print(f"{'='*80}\n")
    
    logger.info(f"🚀 Starting WDS Training. Run ID: {run_id}")
    
    # Resume from checkpoint if specified
    resume_ckpt_path = cfg.get("resume_ckpt_path", None)
    if resume_ckpt_path:
        print(f"[MAIN] Resuming from checkpoint: {resume_ckpt_path}")
        logger.info(f"Resuming from checkpoint: {resume_ckpt_path}")
    
    trainer.fit(ls_module, train_dataloaders=train_loader, ckpt_path=resume_ckpt_path)
    
    print(f"\n{'='*80}")
    print(f"[MAIN] ✓ Training completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
