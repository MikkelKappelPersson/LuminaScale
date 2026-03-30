"""Core trainer classes for dequantization and bit-depth expansion models."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class DequantizationDataset(Dataset):
    """Dataset for paired sRGB (pre-baked) and HDR images, now supported via LMDB."""

    def __init__(
        self,
        hdr_dir: str | Path | None = None,
        srgb_dir: str | Path | None = None,
        lmdb_path: str | Path | None = None,
        file_pattern: str = "*.exr",
        crop_size: int = 512,
        patches_per_image: int = 1,
    ) -> None:
        self.crop_size = crop_size
        self.patches_per_image = max(1, patches_per_image)
        self.lmdb_path = Path(lmdb_path) if lmdb_path else None
        
        # We'll use a simple cache to store the last loaded image to avoid 
        # re-loading 300MB for every patch within the same image.
        self._last_img_idx = -1
        self._cached_hdr = None
        self._cached_srgb = None

        if self.lmdb_path and self.lmdb_path.exists():
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False, readahead=False, meminit=False,
            )
            with self.env.begin(write=False) as txn:
                self.keys = pickle.loads(txn.get(b"__keys__"))
            
            logger.info(
                f"Initialized LMDB dataset with {len(self.keys)} images. "
                f"Generating {self.patches_per_image} patches per image ({len(self)} total samples)."
            )
        else:
            # Legacy Folder Mode
            from luminascale.utils.io import image_to_tensor
            self.hdr_dir = Path(hdr_dir) if hdr_dir else Path("dataset/temp/aces")
            self.srgb_dir = Path(srgb_dir) if srgb_dir else self.hdr_dir.parent / "srgb_looks"
            self.hdr_files = sorted(self.hdr_dir.glob(file_pattern))
            self.paired_files = [(f, self.srgb_dir / f"{f.stem}.png") for f in self.hdr_files 
                                if (self.srgb_dir / f"{f.stem}.png").exists()]
            self.keys = None
            self.image_to_tensor = image_to_tensor

    def __len__(self) -> int:
        num_images = len(self.keys) if self.keys else len(self.paired_files)
        return num_images * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Map flat index to image, cycling every patches_per_image
        img_idx = (idx // self.patches_per_image) % (len(self.keys) if self.keys else len(self.paired_files))
        
        if self.keys:
            # LMDB mode with cache
            if img_idx != self._last_img_idx:
                key = self.keys[img_idx]
                with self.env.begin(write=False) as txn:
                    data = pickle.loads(txn.get(key.encode("ascii")))
                
                self._cached_hdr = torch.from_numpy(data["hdr"]).float() / 10.0  # Normalize HDR to [0, 1] range
                self._cached_srgb = torch.from_numpy(data["ldr"]).float() / 255.0
                self._last_img_idx = img_idx

            # Random crop from the cached image
            c, h, w = self._cached_srgb.shape
            if h < self.crop_size or w < self.crop_size:
                import torch.nn.functional as F
                srgb_tensor = F.interpolate(self._cached_srgb.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear').squeeze(0)
                hdr_tensor = F.interpolate(self._cached_hdr.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear').squeeze(0)
            else:
                top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
                left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
                srgb_tensor = self._cached_srgb[:, top:top+self.crop_size, left:left+self.crop_size]
                hdr_tensor = self._cached_hdr[:, top:top+self.crop_size, left:left+self.crop_size]
        else:
            # Folder mode (legacy path)
            hdr_path, srgb_path = self.paired_files[img_idx]
            import imageio.v3 as iio
            srgb_pixels = iio.imread(srgb_path)
            srgb_tensor = torch.from_numpy(srgb_pixels).permute(2, 0, 1).float() / 255.0
            hdr_tensor = self.image_to_tensor(hdr_path) / 10.0  # Normalize HDR to [0, 1] range

            # Random crop
            c, h, w = srgb_tensor.shape
            if h < self.crop_size or w < self.crop_size:
                import torch.nn.functional as F
                srgb_tensor = F.interpolate(srgb_tensor.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear').squeeze(0)
                hdr_tensor = F.interpolate(hdr_tensor.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear').squeeze(0)
            else:
                top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
                left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
                srgb_tensor = srgb_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
                hdr_tensor = hdr_tensor[:, top:top+self.crop_size, left:left+self.crop_size]

        return srgb_tensor, hdr_tensor


        # Common Random Crop
        c, h, w = srgb_tensor.shape
        if h < self.crop_size or w < self.crop_size:
            import torch.nn.functional as F
            srgb_tensor = F.interpolate(srgb_tensor.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear').squeeze(0)
            hdr_tensor = F.interpolate(hdr_tensor.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear').squeeze(0)
        else:
            top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
            left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
            srgb_tensor = srgb_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
            hdr_tensor = hdr_tensor[:, top:top+self.crop_size, left:left+self.crop_size]

        return srgb_tensor, hdr_tensor


def exposure_mask(
    img: torch.Tensor, threshold_bright: int = 249, threshold_dark: int = 6
) -> torch.Tensor:
    """Compute mask for well-exposed regions (avoid clipped areas)."""
    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )
    gray_8bit = (gray * 255.0).round()

    mask = (gray_8bit >= threshold_dark) & (gray_8bit <= threshold_bright)
    mask = mask.float()

    return mask


def masked_l2_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked L2 loss to ignore clipped regions."""
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss


class Stage2DequantizationDataset(Dataset):
    """Stage 2: Load 32-bit ACES from LMDB and quantize to 8-bit on GPU (ACES→ACES dequantization)."""

    def __init__(
        self,
        lmdb_path: str | Path,
        device: torch.device,
        crop_size: int = 512,
        patches_per_image: int = 1,
    ) -> None:
        self.crop_size = crop_size
        self.patches_per_image = max(1, patches_per_image)
        self.device = device
        self.lmdb_path = Path(lmdb_path)
        
        # Open LMDB
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get(b"__keys__"))
        
        # Image cache to avoid reloading
        self._last_img_idx = -1
        self._cached_aces_32bit = None
        
        logger.info(
            f"Initialized Stage 2 Dataset with {len(self.keys)} images. "
            f"Generating {self.patches_per_image} patches per image ({len(self)} total samples)."
        )
    
    def __len__(self) -> int:
        return len(self.keys) * self.patches_per_image
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (8-bit ACES quantized, 32-bit ACES target) both on device."""
        img_idx = (idx // self.patches_per_image) % len(self.keys)
        
        # Load 32-bit ACES from LMDB (cache to avoid reloading)
        if img_idx != self._last_img_idx:
            key = self.keys[img_idx]
            with self.env.begin(write=False) as txn:
                data = pickle.loads(txn.get(key.encode("ascii")))
            
            # Load and move to device immediately
            aces_32bit_np = data["hdr"]  # Shape: [H, W, 3], range: [0, ~10+]
            self._cached_aces_32bit = torch.from_numpy(aces_32bit_np).permute(2, 0, 1).float().to(self.device)
            self._last_img_idx = img_idx
        
        aces_32bit = self._cached_aces_32bit
        
        # Random crop
        c, h, w = aces_32bit.shape
        if h < self.crop_size or w < self.crop_size:
            import torch.nn.functional as F
            aces_32bit = F.interpolate(
                aces_32bit.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            top = torch.randint(0, h - self.crop_size + 1, (1,), device=self.device).item()
            left = torch.randint(0, w - self.crop_size + 1, (1,), device=self.device).item()
            aces_32bit = aces_32bit[:, top:top+self.crop_size, left:left+self.crop_size]
        
        # Quantize 32-bit ACES to 8-bit on GPU
        # aces_32bit is raw ACES data [0, ~10+], quantize to 8-bit by scaling to [0, 255]
        aces_8bit_quantized = torch.clamp(aces_32bit * 255.0 / 10.0, 0, 255).round()
        
        # Normalize both to [0, 1] range for stable training
        aces_8bit_normalized = aces_8bit_quantized / 255.0      # [0, 1]
        aces_32bit_normalized = aces_32bit / 10.0                # [0, 1] (approx)
        
        return aces_8bit_normalized, aces_32bit_normalized


class DequantizationTrainer:
    """Orchestrates training for Dequantization-Net."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        log_dir: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = masked_l2_loss
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 100,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_freq: int = 10,
        run_name: str | None = None,
    ) -> None:
        """Main training loop.
        
        TensorBoard logs are written to self.writer if initialized.
        View with: tensorboard --logdir=<log_dir>
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for i, (ldr, hdr) in enumerate(train_dataloader):
                ldr, hdr = ldr.to(self.device), hdr.to(self.device)

                # Compute mask for well-exposed regions
                mask = exposure_mask(ldr)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(ldr)
                
                # Compute loss
                loss = self.criterion(output, hdr, mask)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                
                # Log to TensorBoard
                if self.writer:
                    global_step = epoch * len(train_dataloader) + i
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)
                    
                    # Log learning rate
                    for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar("LearningRate", param_group["lr"], global_step)
                
                if (i + 1) % 5 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                        f"Loss: {loss.item():.6f}"
                    )

            avg_loss = running_loss / len(train_dataloader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
            
            # Log epoch-level metrics to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/epoch_avg", avg_loss, epoch + 1)
                self.writer.flush()

            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_name = (
                    f"{run_name}_dequant_net_epoch_{epoch+1}.pt"
                    if run_name
                    else f"dequant_net_epoch_{epoch+1}.pt"
                )
                torch.save(
                    self.model.state_dict(),
                    checkpoint_path / checkpoint_name,
                )
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
            logger.info(f"TensorBoard logs saved to {self.writer.log_dir}")
