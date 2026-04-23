"""ACESMapper Training Module for LuminaScale.

Implements PyTorch Lightning training for color space mapping to ACES2065-1.
Uses Aim for logging and Charbonnier Loss + Color difference as training objectives.

Mandatory Attribution: Based on LLF-LUT (Zeng et al./Wang et al.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

from luminascale.models.aces_mapper import ACESMapper
from luminascale.utils.dataset_pair_generator import DatasetPairGenerator

class CharbonnierLoss(nn.Module):
    """Robust Charbonnier Loss for linear light reconstruction."""
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


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
        lambda_color: float = 0.5,
        crop_size: int = 512,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Initialize model
        self.model = ACESMapper(
            num_luts=num_luts,
            lut_dim=lut_dim,
            num_lap=num_lap,
            num_residual_blocks=num_residual_blocks
        )
        
        # 2. Loss Functions
        self.criterion_reconstruction = CharbonnierLoss()

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
        
    def training_step(self, batch: tuple[list[bytes], list[dict]], batch_idx: int) -> torch.Tensor:
        # Process raw bytes to tensors if needed
        if isinstance(batch, (tuple, list)) and isinstance(batch[0], list):
            input_img, target_img = self._process_batch(batch)
        else:
            input_img, target_img = batch
        
        # Guard against zero-sized batch (data loading failure)
        if input_img.shape[0] == 0:
            return None
        
        # Forward pass
        pred_img = self(input_img)
        
        # Reconstruction Loss (Charbonnier)
        loss_recon = self.criterion_reconstruction(pred_img, target_img)
        
        # Log metrics
        self.log("train/loss_recon", loss_recon, prog_bar=True, batch_size=input_img.shape[0])
        self.log("train/loss_total", loss_recon, prog_bar=True, batch_size=input_img.shape[0])
        
        return loss_recon
        
    def validation_step(self, batch: tuple[list[bytes], list[dict]], batch_idx: int) -> None:
        if isinstance(batch, (tuple, list)) and isinstance(batch[0], list):
            input_img, target_img = self._process_batch(batch)
        else:
            input_img, target_img = batch

        # Guard against zero-sized batch
        if input_img.shape[0] == 0:
            return

        pred_img = self(input_img)
        
        loss_val = self.criterion_reconstruction(pred_img, target_img)
        
        # PSNR & MSE for evaluation
        mse = F.mse_loss(pred_img, target_img)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        
        self.log("val/loss", loss_val, batch_size=input_img.shape[0])
        self.log("val/psnr", psnr, prog_bar=True, batch_size=input_img.shape[0])
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
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
