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
    ) -> None:
        self.crop_size = crop_size
        self.lmdb_path = Path(lmdb_path) if lmdb_path else None
        
        if self.lmdb_path and self.lmdb_path.exists():
            # LMDB Mode
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.env.begin(write=False) as txn:
                self.keys = pickle.loads(txn.get(b"__keys__"))
            logger.info(f"Initialized LMDB dataset with {len(self.keys)} images from {lmdb_path}")
        else:
            # Legacy Folder Mode (kept for backward compatibility)
            from luminascale.utils.io import image_to_tensor
            self.hdr_dir = Path(hdr_dir) if hdr_dir else Path("dataset/temp/aces")
            self.srgb_dir = Path(srgb_dir) if srgb_dir else self.hdr_dir.parent / "srgb_looks"
            
            self.hdr_files = sorted(self.hdr_dir.glob(file_pattern))
            if not self.hdr_files:
                raise ValueError(f"No HDR images found in {hdr_dir}")

            self.paired_files = []
            for hdr_path in self.hdr_files:
                srgb_path = self.srgb_dir / f"{hdr_path.stem}.png"
                if srgb_path.exists():
                    self.paired_files.append((hdr_path, srgb_path))
            
            if not self.paired_files:
                raise ValueError(f"No matching sRGB images found in {self.srgb_dir}. Did you run bake/pack scripts?")

            logger.info(f"Initialized Folder dataset with {len(self.paired_files)} images")
            self.image_to_tensor = image_to_tensor
            self.keys = None

    def __len__(self) -> int:
        return len(self.keys) if self.keys else len(self.paired_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys:
            # LMDB Fast Path
            key = self.keys[idx]
            with self.env.begin(write=False) as txn:
                raw_data = txn.get(key.encode("ascii"))
                data = pickle.loads(raw_data)
                
            # Convert raw numpy arrays directly to tensors [C, H, W]
            # hdr: float32 [0, 1], ldr: uint8 [0, 255]
            hdr_tensor = torch.from_numpy(data["hdr"]).float()
            srgb_tensor = torch.from_numpy(data["ldr"]).float() / 255.0
        else:
            # Legacy Folder Slow Path
            hdr_path, srgb_path = self.paired_files[idx]
            import imageio.v3 as iio
            srgb_pixels = iio.imread(srgb_path)
            srgb_tensor = torch.from_numpy(srgb_pixels).permute(2, 0, 1).float() / 255.0
            hdr_tensor = self.image_to_tensor(hdr_path)

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


class DequantizationTrainer:
    """Orchestrates training for Dequantization-Net."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = masked_l2_loss

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 100,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_freq: int = 10,
    ) -> None:
        """Main training loop."""
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
                self.optimizer.step()

                running_loss += loss.item()
                if (i + 1) % 5 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                        f"Loss: {loss.item():.6f}"
                    )

            avg_loss = running_loss / len(train_dataloader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")

            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(
                    self.model.state_dict(),
                    checkpoint_path / f"dequant_net_epoch_{epoch+1}.pt",
                )
