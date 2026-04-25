"""ACESMapper Training Module for LuminaScale.

Implements PyTorch Lightning training for color space mapping to ACES2065-1.
Uses TensorBoard for logging and an LLF-LUT-style reconstruction objective.

Mandatory Attribution: Based on LLF-LUT (Zeng et al./Wang et al.)
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import lpips

from luminascale.models.aces_mapper import ACESMapper
from luminascale.utils.dataset_pair_generator import DatasetPairGenerator


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
        lambda_recon: float = 1.0,
        lambda_lpips: float = 0.1,
        lambda_smooth: float = 1e-4,
        lambda_mono: float = 1e-4,
        lambda_color: float | None = None,
        crop_size: int = 512,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if lambda_color is not None:
            self.hparams.lambda_lpips = lambda_color
        
        # 1. Initialize model
        self.model = ACESMapper(
            num_luts=num_luts,
            lut_dim=lut_dim,
            num_lap=num_lap,
            num_residual_blocks=num_residual_blocks
        )
        
        # 2. Loss Functions
        self.recon_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net="vgg")
        for parameter in self.lpips_loss.parameters():
            parameter.requires_grad = False
        self.lpips_loss.eval()

        # 3. Data Processing (GPU-accelerated)
        self.pair_generator = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            self.pair_generator = DatasetPairGenerator(self.device)
        
        exr_bytes_list, _ = batch
        # DatasetPairGenerator returns (srgb_input, aces_target, timing)
        x, y, _ = self.pair_generator.generate_aces_mapper_batch_from_bytes(
            exr_bytes_list, 
            crop_size=self.hparams.crop_size
        )
        return x, y

    def _compute_losses(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = self.recon_loss(pred_img, target_img)
        perceptual_loss = self.lpips_loss(pred_img, target_img, normalize=True).mean()
        lut_loss = self._compute_lut_regularization(pred_img)
        total_loss = (
            self.hparams.lambda_recon * recon_loss
            + self.hparams.lambda_lpips * perceptual_loss
            + lut_loss
        )
        return recon_loss, perceptual_loss, lut_loss, total_loss

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

    def _compute_lut_regularization(self, input_img: torch.Tensor) -> torch.Tensor:
        weights = self.model.sft(input_img)
        pred_weight = weights[:, : self.model.num_luts]
        weights_norm = torch.mean(pred_weight ** 2)

        tv_cons = torch.zeros((), device=input_img.device, dtype=input_img.dtype)
        mn_cons = torch.zeros((), device=input_img.device, dtype=input_img.dtype)
        for lut_module in self.model.luts:
            tv_term, mono_term = self._lut_tv_and_mono(lut_module.lut)
            tv_cons = tv_cons + tv_term
            mn_cons = mn_cons + mono_term

        loss_smooth = weights_norm + tv_cons if self.hparams.lambda_smooth > 0 else torch.zeros_like(tv_cons)
        loss_mono = mn_cons if self.hparams.lambda_mono > 0 else torch.zeros_like(mn_cons)
        return self.hparams.lambda_smooth * loss_smooth + self.hparams.lambda_mono * loss_mono
        
    def training_step(self, batch: tuple[list[bytes], list[dict]], batch_idx: int) -> torch.Tensor:
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
        
        # Forward pass
        pred_img = self(input_img)
        
        recon_loss, perceptual_loss, lut_loss, total_loss = self._compute_losses(pred_img, target_img)
        
        # Log metrics
        self.log("loss/recon/train", recon_loss, batch_size=input_img.shape[0])
        self.log("loss/lpips/train", perceptual_loss, batch_size=input_img.shape[0])
        self.log("loss/lut/train", lut_loss, batch_size=input_img.shape[0])
        self.log("loss/total/train", total_loss, prog_bar=True, batch_size=input_img.shape[0])
        
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

        pred_img = self(input_img)
        
        recon_loss, perceptual_loss, lut_loss, total_loss = self._compute_losses(pred_img, target_img)
        
        # PSNR & MSE for evaluation
        mse = F.mse_loss(pred_img, target_img)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        
        self.log("loss/recon/val", recon_loss, batch_size=input_img.shape[0])
        self.log("loss/lpips/val", perceptual_loss, batch_size=input_img.shape[0])
        self.log("loss/lut/val", lut_loss, batch_size=input_img.shape[0])
        self.log("loss/total/val", total_loss, batch_size=input_img.shape[0])
        self.log("psnr/val", psnr, prog_bar=True, batch_size=input_img.shape[0])
        
    def configure_optimizers(self):
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters, 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
