"""Hydra-based training script for ACESMapper (WebDataset).

Usage (local development):
    python scripts/train_aces_mapper.py --config-name=mapper

Usage (HPC via Slurm):
    sbatch scripts/train_aces_mapper.sh
"""

from __future__ import annotations

import logging
import os
import hashlib
import re
import subprocess
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
from luminascale.training.progress_bar import CustomRichProgressBar
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


def _slugify(value: str) -> str:
    """Convert label to filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _sha256_file(file_path: Path) -> str:
    """Compute SHA256 for a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_display_luts(project_root: Path, cfg: DictConfig) -> None:
    """Ensure required baked display LUT artifacts exist; auto-bake if configured."""
    cm = cfg.get("color_management", {})
    display = str(cm.get("display", "sRGB - Display"))
    view = str(cm.get("view", "ACES 2.0 - SDR 100 nits (Rec.709)"))
    lut_cube_size = int(cm.get("lut_cube_size", 257))
    auto_bake = bool(cm.get("auto_bake_luts", True))

    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if not ocio_config.exists():
        raise FileNotFoundError(f"OCIO config not found: {ocio_config}")

    config_hash8 = _sha256_file(ocio_config)[:8]
    profile_id = f"{config_hash8}__{_slugify(display)}__{_slugify(view)}__{lut_cube_size}"
    profile_dir = project_root / "assets" / "luts" / profile_id

    required = [
        profile_dir / "manifest.json",
        profile_dir / "domains.json",
        profile_dir / "aces2065_to_srgb_display.pt",
        profile_dir / "acescct_to_srgb_display.pt",
    ]

    missing = [p for p in required if not p.exists()]
    if not missing:
        logger.info(f"[LUT] Using existing baked LUT profile: {profile_dir}")
        return

    if not auto_bake:
        missing_str = "\n  - ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Required LUT artifacts are missing and auto_bake_luts is disabled. Missing:\n"
            f"  - {missing_str}"
        )

    logger.info("[LUT] Missing LUT artifacts detected; auto-baking profile...")
    bake_cmd = [
        sys.executable,
        str(project_root / "scripts" / "bake_display_luts.py"),
        "--config",
        str(ocio_config),
        "--display",
        display,
        "--view",
        view,
        "--cube-size",
        str(lut_cube_size),
        "--aces2065-domain-min",
        str(float(cm.get("aces2065_domain_min", -0.5))),
        "--aces2065-domain-max",
        str(float(cm.get("aces2065_domain_max", 10.0))),
        "--acescct-domain-min",
        str(float(cm.get("acescct_domain_min", -1.0))),
        "--acescct-domain-max",
        str(float(cm.get("acescct_domain_max", 1.0))),
    ]
    subprocess.run(bake_cmd, check=True)

    missing_after = [p for p in required if not p.exists()]
    if missing_after:
        missing_str = "\n  - ".join(str(p) for p in missing_after)
        raise FileNotFoundError(
            "LUT auto-bake finished but required artifacts are still missing:\n"
            f"  - {missing_str}"
        )
    logger.info(f"[LUT] Auto-bake complete: {profile_dir}")


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


class SanitizedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that sanitizes PSNR token formatting in checkpoint filenames.

    Transforms generated names like:
      `...-00_psnr16.15.ckpt`
    into:
      `...-00_psnr16-15.ckpt`
    """

    def format_checkpoint_name(
        self,
        metrics: dict[str, torch.Tensor],
        filename: str | None = None,
        ver: int | None = None,
        prefix: str | None = None,
    ) -> str:
        checkpoint_name = super().format_checkpoint_name(
            metrics=metrics,
            filename=filename,
            ver=ver,
            prefix=prefix,
        )
        return re.sub(r"(_psnr\d+)\.(\d+)", r"\1-\2", checkpoint_name)


class PeriodicACESMapperInferenceCallback(Callback):
    """Save and log ACES mapper comparison dashboards every N epochs.
    
    Converts ACES2065-1 reference data to ACEScct (project working space) internally
    for fair comparison with model output. Visualization handles color space transformations.
    """

    def __init__(
        self,
        *,
        every_n_epochs: int = 1,
        aces_input_path: Path,
        output_dir: Path,
        downsample_size: int = 1024,
        lut_cube_size: int = 257,
        target_color_space: str = "ACEScct",
        display: str = "sRGB - Display",
        view: str = "ACES 2.0 - SDR 100 nits (Rec.709)",
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.aces_input_path = aces_input_path
        self.output_dir = output_dir
        self.downsample_size = max(0, int(downsample_size))
        self.lut_cube_size = max(2, int(lut_cube_size))
        self.target_color_space = str(target_color_space)
        self.display = str(display)
        self.view = str(view)
        # Only build look and enable inference if input path exists
        self.enabled = aces_input_path.exists()
        self.look = build_look() if self.enabled else None
        if not self.enabled:
            logger.warning(
                f"[PeriodicACESMapperInferenceCallback] Input path does not exist: {aces_input_path}. "
                f"Inference visualization will be disabled."
            )
        else:
            logger.info(
                f"[PeriodicACESMapperInferenceCallback] Enabled with input: {aces_input_path}"
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero or not self.enabled:
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
            if trainer.strategy.root_device.type == "cuda":
                torch.cuda.empty_cache()

            save_path = self.output_dir / f"epoch_{epoch_number:04d}.png"
            figure = run_aces_mapper_inference(
                model=pl_module,
                input=self.aces_input_path,
                output_path=save_path,
                look=self.look,
                crop_size=0,
                max_side=self.downsample_size,
                pred_aces_output=None,
                input_is_aces=True,
                device=trainer.strategy.root_device,
                lut_cube_size=self.lut_cube_size,
                target_color_space=self.target_color_space,
                display=self.display,
                view=self.view,
            )
            trainer.logger.experiment.add_figure(
                "inference/comparison",
                figure,
                global_step=epoch_number,
            )
        finally:
            close_figure(figure)
            if trainer.strategy.root_device.type == "cuda":
                torch.cuda.empty_cache()
            if was_training:
                pl_module.train()


@hydra.main(config_path="../configs", config_name="mapper", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # 1. Performance Optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    logger.info("[MAIN] cuDNN benchmark disabled to avoid benchmark cache teardown crashes")
    
    # Enable memory-efficient allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 2. Precision handling
    precision = cfg.get("precision", "16-mixed")
    task_name = str(cfg.get("task_name") or "mapper")
    inference_vis_input_path = Path(
        cfg.get(
            "inference_vis_input_path",
            "dataset/full/aces/MIT-Adobe_5K_a0001-jmac_DSC1459.exr",
        )
    )
    
    # Resolve inference path relative to project root (Hydra changes cwd)
    if not inference_vis_input_path.is_absolute():
        inference_vis_input_path = project_root / inference_vis_input_path
    
    # Validate inference input file exists
    if not inference_vis_input_path.exists():
        logger.warning(
            f"[MAIN] Inference input path does not exist: {inference_vis_input_path}"
        )
        logger.warning(
            f"[MAIN] Inference visualization will be skipped. "
            f"Please check inference_vis_input_path config."
        )
    else:
        logger.info(f"[MAIN] Inference input validated: {inference_vis_input_path}")
    
    # Set OCIO environment if needed
    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)

    # Ensure baked display LUT artifacts exist for configured color management.
    ensure_display_luts(project_root, cfg)

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
        crop_size=cfg.get("crop_size", 512),
        crop_mode=str(cfg.data.crop_mode),
        lr_scheduler_eta_min=float(cfg.trainer.params.lr_scheduler_eta_min),
        max_epochs=int(cfg.epochs),
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
        "lr_scheduler_eta_min": float(cfg.trainer.params.lr_scheduler_eta_min),
        "gradient_clip_val": float(cfg.trainer.params.gradient_clip_val),
        "crop_mode": str(cfg.data.crop_mode),
        "num_luts": int(cfg.model.params.num_luts),
        "lut_dim": int(cfg.model.params.lut_dim),
        "num_lap": int(cfg.model.params.num_lap),
        "num_residual_blocks": int(cfg.model.params.num_residual_blocks),
        "num_workers": int(cfg.get("num_workers", 4)),
        "inference_vis_every_n_epochs": int(cfg.get("inference_vis_every_n_epochs", 1)),
        "inference_vis_downsample_size": int(cfg.get("inference_vis_downsample_size", cfg.get("crop_size", 1024))),
    }

    inference_output_dir = Path(logger_tb.log_dir)
    checkpoint_dir = os.path.join(logger_tb.log_dir, "checkpoints")
    callbacks = [
        SanitizedModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{checkpoint_name_prefix}-{run_version}-{{epoch:02d}}_psnr{{psnr/val:.2f}}",
            monitor="psnr/val",
            mode="max",
            save_top_k=int(cfg.get("save_top_k", 3)),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=2),
        CustomRichProgressBar(batch_size=int(cfg.get("batch_size", 4))),
        HparamsMetricsCallback(hparams_dict),
        PeriodicACESMapperInferenceCallback(
            aces_input_path=inference_vis_input_path,
            output_dir=inference_output_dir,
            every_n_epochs=int(cfg.get("inference_vis_every_n_epochs", 1)),
            downsample_size=int(cfg.get("inference_vis_downsample_size", cfg.get("crop_size", 1024))),
            lut_cube_size=int(cfg.get("color_management", {}).get("lut_cube_size", 257)),
            target_color_space=str(cfg.model.params.get("target_color_space", "ACEScct")),
            display=str(cfg.get("color_management", {}).get("display", "sRGB - Display")),
            view=str(cfg.get("color_management", {}).get("view", "ACES 2.0 - SDR 100 nits (Rec.709)")),
        ),
    ]

    # 5. Trainer setup
    trainer = L.Trainer(
        max_epochs=int(cfg.epochs),
        accelerator="gpu",
        devices=1,  # Adjust for multi-GPU if needed
        logger=logger_tb,
        callbacks=callbacks,
        precision=precision, 
        gradient_clip_val=float(cfg.trainer.params.gradient_clip_val),
    )

    # 6. Start Training
    print(f"[MAIN] Starting fit...")
    trainer.fit(trainer_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    completion_message = "[MAIN] Training loop completed successfully. If the process crashes after this point, the failure happened during teardown/shutdown."
    print(completion_message, flush=True)
    logger.info(completion_message)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
