"""ACESMapper Training Module for LuminaScale.

Implements PyTorch Lightning training for color space mapping.
Uses TensorBoard for logging and an LLF-LUT-style reconstruction objective.
Supports flexible target color spaces (ACEScct, ACES2065-1, etc.).

Mandatory Attribution: Based on LLF-LUT (Zeng et al./Wang et al.)
"""

from __future__ import annotations

import logging
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import lpips

from luminascale.models.aces_mapper import ACESMapper
from luminascale.utils.dataset_pair_generator import DatasetPairGenerator


logger = logging.getLogger(__name__)


class ACESMapperTrainer(L.LightningModule):
    """
    Lightning module for training the ACESMapper head.
    """
    def __init__(
        self,
        num_luts: int = 3,
        lut_dim: int = 33,
        num_lap: int = 3,
        num_residual_blocks: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_l1: float = 1.0,
        lambda_lpips: float = 0.1,
        lambda_smooth: float = 1e-4,
        lambda_mono: float = 1e-4,
        lambda_color: float | None = None,
        crop_size: int = 512,
        crop_mode: str = "random",
        lr_scheduler_eta_min: float = 1e-6,
        max_epochs: int = 100,
        target_color_space: str = "ACEScct",
    ) -> None:
        super().__init__()
        if crop_mode != "random":
            raise ValueError(
                f"Invalid crop_mode='{crop_mode}'. ACES mapper requires crop_mode='random' (fail-fast policy)."
            )
        if max_epochs <= 0:
            raise ValueError(f"Invalid max_epochs={max_epochs}. max_epochs must be > 0.")
        if lr_scheduler_eta_min < 0:
            raise ValueError(
                f"Invalid lr_scheduler_eta_min={lr_scheduler_eta_min}. lr_scheduler_eta_min must be >= 0."
            )

        self.save_hyperparameters()

        if lambda_color is not None:
            self.hparams.lambda_lpips = lambda_color
        
        # 1. Initialize model with target color space
        # Note: Training targets come from WebDataset shards (ACEScct by default).
        # Loss is computed in the target color space specified by config.
        self.model = ACESMapper(
            num_luts=num_luts,
            lut_dim=lut_dim,
            num_lap=num_lap,
            num_residual_blocks=num_residual_blocks,
            target_color_space=target_color_space,
        )
        
        # 2. Loss Functions
        self.loss_l1 = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net="vgg")
        for parameter in self.lpips_loss.parameters():
            parameter.requires_grad = False
        self.lpips_loss.eval()

        # 3. Data Processing (GPU-accelerated)
        self.pair_generator = None
        self.invalid_train_batches = 0
        self.invalid_val_batches = 0
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """WebDataset batches are (list[bytes], list[dict]). Skip device transfer for raw bytes."""
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            if isinstance(batch[0], list) and len(batch[0]) > 0 and isinstance(batch[0][0], bytes):
                return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _process_batch(self, batch: tuple[list[bytes], list[dict]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw WDS batch (bytes) into (Input, Target) pairs on GPU."""
        if self.pair_generator is None:
            self.pair_generator = DatasetPairGenerator(self.device, crop_mode=str(self.hparams.crop_mode))
        
        exr_bytes_list, _ = batch
        # DatasetPairGenerator returns (srgb_input, aces_target, timing)
        x, y, _ = self.pair_generator.generate_aces_mapper_batch_from_bytes(
            exr_bytes_list, 
            crop_size=self.hparams.crop_size
        )
        return x, y

    def _tensor_stats(self, tensor: torch.Tensor) -> str:
        total = int(tensor.numel())
        finite = int(torch.isfinite(tensor).sum().item())
        finite_ratio = (finite / total) if total > 0 else 0.0
        if finite > 0:
            finite_tensor = tensor[torch.isfinite(tensor)]
            min_val = float(finite_tensor.min().item())
            max_val = float(finite_tensor.max().item())
        else:
            min_val = float("nan")
            max_val = float("nan")
        return f"shape={tuple(tensor.shape)}, finite_ratio={finite_ratio:.6f}, min={min_val:.6g}, max={max_val:.6g}"

    def _report_invalid_batch(
        self,
        *,
        stage: str,
        batch_idx: int,
        reason: str,
        input_img: torch.Tensor,
        target_img: torch.Tensor,
        pred_img: torch.Tensor | None = None,
        total_loss: torch.Tensor | None = None,
    ) -> None:
        details = [
            f"[BatchGuard][{stage}] Invalid batch encountered at batch_idx={batch_idx}: {reason}",
            f"input_stats: {self._tensor_stats(input_img)}",
            f"target_stats: {self._tensor_stats(target_img)}",
        ]
        if pred_img is not None:
            details.append(f"pred_stats: {self._tensor_stats(pred_img)}")
        if total_loss is not None:
            details.append(f"total_loss={float(total_loss.detach().cpu().item())}")
        logger.warning(" | ".join(details))

    def _compute_losses(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
        point_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_l1 = self.loss_l1(pred_img, target_img)
        perceptual_loss = self.lpips_loss(pred_img, target_img, normalize=True).mean()
        lut_loss = self._compute_lut_regularization(point_weights, device=pred_img.device, dtype=pred_img.dtype)
        total_loss = (
            self.hparams.lambda_l1 * loss_l1
            + self.hparams.lambda_lpips * perceptual_loss
            + lut_loss
        )
        return loss_l1, perceptual_loss, lut_loss, total_loss

    def _lut_tv_and_mono(self, lut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dif_r = lut[:, :, :, :-1] - lut[:, :, :, 1:]
        dif_g = lut[:, :, :-1, :] - lut[:, :, 1:, :]
        dif_b = lut[:, :-1, :, :] - lut[:, 1:, :, :]

        weight_r = torch.ones_like(dif_r)
        weight_r[:, :, :, 0] *= 2.0
        weight_r[:, :, :, -1] *= 2.0

        weight_g = torch.ones_like(dif_g)
        weight_g[:, :, 0, :] *= 2.0
        weight_g[:, :, -1, :] *= 2.0

        weight_b = torch.ones_like(dif_b)
        weight_b[:, 0, :, :] *= 2.0
        weight_b[:, -1, :, :] *= 2.0

        tv = (
            torch.mean((dif_r ** 2) * weight_r)
            + torch.mean((dif_g ** 2) * weight_g)
            + torch.mean((dif_b ** 2) * weight_b)
        )
        mono = (
            torch.mean(torch.relu(dif_r))
            + torch.mean(torch.relu(dif_g))
            + torch.mean(torch.relu(dif_b))
        )
        return tv, mono

    def _compute_lut_regularization(
        self,
        point_weights: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute LUT regularization losses.

        Matches original LLF-LUT design: weights_norm reuses point_weights
        already computed during the model forward pass (no second SFT pass).
        TV / monotonicity losses operate purely on the LUT parameter tensors.
        """
        weights_norm = torch.mean(point_weights ** 2)

        tv_cons = torch.zeros((), device=device, dtype=dtype)
        mn_cons = torch.zeros((), device=device, dtype=dtype)
        for lut_module in self.model.luts:
            tv_term, mono_term = self._lut_tv_and_mono(lut_module.lut)
            tv_cons = tv_cons + tv_term
            mn_cons = mn_cons + mono_term

        loss_smooth = weights_norm + tv_cons if self.hparams.lambda_smooth > 0 else torch.zeros_like(tv_cons)
        loss_mono = mn_cons if self.hparams.lambda_mono > 0 else torch.zeros_like(mn_cons)
        return self.hparams.lambda_smooth * loss_smooth + self.hparams.lambda_mono * loss_mono
        
    def training_step(self, batch: tuple[list[bytes], list[dict]], batch_idx: int) -> torch.Tensor | None:
        # Process raw bytes to tensors if needed
        if isinstance(batch, (tuple, list)) and isinstance(batch[0], list):
            input_img, target_img = self._process_batch(batch)
        else:
            input_img, target_img = batch

        input_img = cast(torch.Tensor, input_img)
        target_img = cast(torch.Tensor, target_img)
        
        # Guard against zero-sized batch (data loading failure)
        if input_img.shape[0] == 0:
            return None

        if not torch.isfinite(input_img).all() or not torch.isfinite(target_img).all():
            self.invalid_train_batches += 1
            self._report_invalid_batch(
                stage="train",
                batch_idx=batch_idx,
                reason="non-finite input/target tensor",
                input_img=input_img,
                target_img=target_img,
            )
            self.log("batch_guard_invalid/train", 1.0, on_step=True, on_epoch=True, batch_size=1)
            return None
        
        # Forward pass — returns (output_image, point_weights)
        pred_img, point_weights = self(input_img)

        if not torch.isfinite(pred_img).all() or not torch.isfinite(point_weights).all():
            self.invalid_train_batches += 1
            self._report_invalid_batch(
                stage="train",
                batch_idx=batch_idx,
                reason="non-finite model output",
                input_img=input_img,
                target_img=target_img,
                pred_img=pred_img,
            )
            self.log("batch_guard_invalid/train", 1.0, on_step=True, on_epoch=True, batch_size=1)
            return None
        
        loss_l1, perceptual_loss, lut_loss, total_loss = self._compute_losses(pred_img, target_img, point_weights)

        if not torch.isfinite(total_loss):
            self.invalid_train_batches += 1
            self._report_invalid_batch(
                stage="train",
                batch_idx=batch_idx,
                reason="non-finite total loss",
                input_img=input_img,
                target_img=target_img,
                pred_img=pred_img,
                total_loss=total_loss,
            )
            self.log("batch_guard_invalid/train", 1.0, on_step=True, on_epoch=True, batch_size=1)
            return None
        
        # Log metrics
        self.log("loss_l1/train", loss_l1, batch_size=input_img.shape[0])
        self.log("loss_lpips/train", perceptual_loss, batch_size=input_img.shape[0])
        self.log("loss_lut/train", lut_loss, batch_size=input_img.shape[0])
        self.log("loss_total/train", total_loss, prog_bar=True, batch_size=input_img.shape[0])
        self.log("batch_guard_invalid/train", 0.0, on_step=True, on_epoch=True, batch_size=1)
        
        return total_loss
        
    def validation_step(self, batch: tuple[list[bytes], list[dict]], batch_idx: int) -> None:
        if isinstance(batch, (tuple, list)) and isinstance(batch[0], list):
            input_img, target_img = self._process_batch(batch)
        else:
            input_img, target_img = batch

        input_img = cast(torch.Tensor, input_img)
        target_img = cast(torch.Tensor, target_img)

        # Guard against zero-sized batch
        if input_img.shape[0] == 0:
            return

        if not torch.isfinite(input_img).all() or not torch.isfinite(target_img).all():
            self.invalid_val_batches += 1
            self._report_invalid_batch(
                stage="val",
                batch_idx=batch_idx,
                reason="non-finite input/target tensor",
                input_img=input_img,
                target_img=target_img,
            )
            self.log("batch_guard_invalid/val", 1.0, on_step=False, on_epoch=True, batch_size=1)
            return

        pred_img, point_weights = self(input_img)

        if not torch.isfinite(pred_img).all() or not torch.isfinite(point_weights).all():
            self.invalid_val_batches += 1
            self._report_invalid_batch(
                stage="val",
                batch_idx=batch_idx,
                reason="non-finite model output",
                input_img=input_img,
                target_img=target_img,
                pred_img=pred_img,
            )
            self.log("batch_guard_invalid/val", 1.0, on_step=False, on_epoch=True, batch_size=1)
            return
        
        loss_l1, perceptual_loss, lut_loss, total_loss = self._compute_losses(pred_img, target_img, point_weights)

        if not torch.isfinite(total_loss):
            self.invalid_val_batches += 1
            self._report_invalid_batch(
                stage="val",
                batch_idx=batch_idx,
                reason="non-finite total loss",
                input_img=input_img,
                target_img=target_img,
                pred_img=pred_img,
                total_loss=total_loss,
            )
            self.log("batch_guard_invalid/val", 1.0, on_step=False, on_epoch=True, batch_size=1)
            return
        
        # PSNR & MSE for evaluation
        mse = F.mse_loss(pred_img, target_img)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        
        self.log("loss_l1/val", loss_l1, on_step=False, on_epoch=True, batch_size=input_img.shape[0])
        self.log("loss_lpips/val", perceptual_loss, on_step=False, on_epoch=True, batch_size=input_img.shape[0])
        self.log("loss_lut/val", lut_loss, on_step=False, on_epoch=True, batch_size=input_img.shape[0])
        self.log("loss_total/val", total_loss, on_step=False, on_epoch=True, batch_size=input_img.shape[0])
        self.log("psnr/val", psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=input_img.shape[0])
        self.log("batch_guard_invalid/val", 0.0, on_step=False, on_epoch=True, batch_size=1)
        
    def configure_optimizers(self):
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters, 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        t_max = int(self.hparams.max_epochs)
        eta_min = float(self.hparams.lr_scheduler_eta_min)
        if t_max <= 0:
            raise ValueError(f"Invalid scheduler T_max={t_max}. max_epochs must be > 0.")
        if eta_min < 0:
            raise ValueError(f"Invalid scheduler eta_min={eta_min}. eta_min must be >= 0.")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=t_max,
            eta_min=eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
