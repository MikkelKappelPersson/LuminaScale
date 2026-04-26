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
    Callback,
)
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
from luminascale.training.logger import CustomTensorBoardLogger
from luminascale.utils.aces_mapper_inference import build_look
from luminascale.utils.aces_mapper_inference import close_figure, run_aces_mapper_inference

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


class HparamsMetricsCallback(Callback):
    """Explicitly log hparams + metrics with CustomTensorBoardLogger.

    CustomTensorBoardLogger disables early automatic hparams logging. This callback
    logs hparams at fit start and updates them with validation metrics during training.
    """

    def __init__(self, hparams_dict: dict[str, Any]) -> None:
        super().__init__()
        self.hparams_dict = hparams_dict

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.logger is not None and hasattr(trainer.logger, "log_hyperparams_metrics"):
            trainer.logger.log_hyperparams_metrics(self.hparams_dict, {})

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.logger is None or not hasattr(trainer.logger, "log_hyperparams_metrics"):
            return

        validation_total_loss = trainer.callback_metrics.get("loss_total/val")
        validation_loss = trainer.callback_metrics.get("loss_l1/val")
        validation_psnr = trainer.callback_metrics.get("psnr/val")

        metrics_dict: dict[str, float] = {}
        # Keep hparams metrics under separate keys so they don't interfere with scalar curves.
        if validation_psnr is not None:
            metrics_dict["metric/psnr"] = float(validation_psnr.detach().cpu().item())
        if validation_total_loss is not None:
            metrics_dict["metric/loss_total_val"] = float(validation_total_loss.detach().cpu().item())
        if metrics_dict:
            trainer.logger.log_hyperparams_metrics(self.hparams_dict, metrics_dict)


class PeriodicACESMapperInferenceCallback(Callback):
    """Save and log ACES mapper comparison dashboards every N epochs."""

    def __init__(
        self,
        *,
        every_n_epochs: int = 1,
        aces_input_path: Path,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.aces_input_path = aces_input_path
        self.output_dir = output_dir
        self.look = build_look()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        epoch_number = trainer.current_epoch + 1
        if epoch_number % self.every_n_epochs != 0:
            return

        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return

        was_training = pl_module.training
        pl_module.eval()
        figure = None

        try:
            save_path = self.output_dir / f"epoch_{epoch_number:04d}.png"
            figure = run_aces_mapper_inference(
                model=pl_module,
                input=self.aces_input_path,
                output_path=save_path,
                look=self.look,
                pred_aces_output=None,
                input_is_aces=True,
                device=trainer.strategy.root_device,
            )
            trainer.logger.experiment.add_figure(
                "inference/comparison",
                figure,
                global_step=epoch_number,
            )
        finally:
            close_figure(figure)
            if was_training:
                pl_module.train()


@hydra.main(config_path="../configs", config_name="mapper", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # 1. Performance Optimizations
    torch.set_float32_matmul_precision('high')
    
    # 2. Precision handling
    precision = cfg.get("precision", "16-mixed")
    task_name = str(cfg.get("task_name") or "mapper")
    inference_vis_input_path = Path(
        cfg.get(
            "inference_vis_input_path",
            "dataset/full/aces/MIT-Adobe_5K_a0001-jmac_DSC1459.exr",
        )
    )
    
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
        lambda_l1=cfg.trainer.params.get("lambda_l1", 1.0),
        lambda_lpips=cfg.trainer.params.get("lambda_lpips", 0.1),
        lambda_smooth=cfg.trainer.params.get("lambda_smooth", 1e-4),
        lambda_mono=cfg.trainer.params.get("lambda_mono", 1e-4),
        crop_size=cfg.get("crop_size", 512)
    )

    # 4. Logger & Callbacks
    run_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name_prefix = str(cfg.get("task_name") or "aces-mapper")

    logger_tb = CustomTensorBoardLogger(
        save_dir=cfg.output_dir,
        name="",
        version=run_version,
    )

    hparams_dict = {
        "task_name": task_name,
        "batch_size": int(cfg.get("batch_size", 4)),
        "crop_size": int(cfg.get("crop_size", 512)),
        "precision": str(precision),
        "epochs": int(cfg.get("epochs", 100)),
        "lr": float(cfg.trainer.params.lr),
        "weight_decay": float(cfg.trainer.params.weight_decay),
        "lambda_l1": float(cfg.trainer.params.get("lambda_l1", 1.0)),
        "lambda_lpips": float(cfg.trainer.params.get("lambda_lpips", 0.1)),
        "lambda_smooth": float(cfg.trainer.params.get("lambda_smooth", 1e-4)),
        "lambda_mono": float(cfg.trainer.params.get("lambda_mono", 1e-4)),
        "num_luts": int(cfg.model.params.num_luts),
        "lut_dim": int(cfg.model.params.lut_dim),
        "num_lap": int(cfg.model.params.num_lap),
        "num_residual_blocks": int(cfg.model.params.num_residual_blocks),
        "num_workers": int(cfg.get("num_workers", 4)),
        "inference_vis_every_n_epochs": int(cfg.get("inference_vis_every_n_epochs", 1)),
    }

    inference_output_dir = Path(logger_tb.log_dir)
    checkpoint_dir = os.path.join(logger_tb.log_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{checkpoint_name_prefix}-{run_version}-{{epoch:02d}}",
            monitor="loss_total/val" if val_loader else "loss_total/train",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=2),
        RichProgressBar(),
        HparamsMetricsCallback(hparams_dict),
        PeriodicACESMapperInferenceCallback(
            aces_input_path=inference_vis_input_path,
            output_dir=inference_output_dir,
            every_n_epochs=int(cfg.get("inference_vis_every_n_epochs", 1)),
        ),
    ]

    # 5. Trainer setup
    trainer = L.Trainer(
        max_epochs=cfg.get("epochs", 100),
        accelerator="gpu",
        devices=1,  # Adjust for multi-GPU if needed
        logger=logger_tb,
        callbacks=callbacks,
        precision=precision, 
        gradient_clip_val=1.0,
    )

    # 6. Start Training
    print(f"[MAIN] Starting fit...")
    trainer.fit(trainer_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
