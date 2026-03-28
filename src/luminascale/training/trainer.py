"""Core trainer classes for dequantization and bit-depth expansion models."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DequantizationDataset(Dataset):
    """Dataset for paired LDR (pre-baked) and HDR images."""

    def __init__(
        self,
        hdr_dir: str | Path,
        srgb_dir: str | Path | None = None,
        file_pattern: str = "*.exr",
        crop_size: int = 512,
    ) -> None:
        from luminascale.utils.io import image_to_tensor

        self.hdr_dir = Path(hdr_dir)
        # Default to 'srgb_looks' sibling directory if not provided
        self.srgb_dir = Path(srgb_dir) if srgb_dir else self.hdr_dir.parent / "srgb_looks"
        
        self.hdr_files = sorted(self.hdr_dir.glob(file_pattern))
        self.crop_size = crop_size

        if not self.hdr_files:
            raise ValueError(f"No HDR images found in {hdr_dir}")

        # Verify sRGB files exist
        self.paired_files = []
        for hdr_path in self.hdr_files:
            srgb_path = self.srgb_dir / f"{hdr_path.stem}.png"
            if srgb_path.exists():
                self.paired_files.append((hdr_path, srgb_path))
        
        if not self.paired_files:
            raise ValueError(f"No matching sRGB images found in {self.srgb_dir}. Did you run bake_dataset.py?")

        logger.info(f"Found {len(self.paired_files)} paired sRGB/HDR images")
        self.image_to_tensor = image_to_tensor

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hdr_path, srgb_path = self.paired_files[idx]

        # 1. Load pre-baked sRGB (fast PNG decode)
        import imageio.v3 as iio
        srgb_pixels = iio.imread(srgb_path)
        
        # 2. Convert to tensor [0, 1]
        srgb_tensor = torch.from_numpy(srgb_pixels).permute(2, 0, 1).float() / 255.0

        # 3. Load HDR reference
        hdr_tensor = self.image_to_tensor(hdr_path)

        # 4. Random Crop (common to both)
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
    """Compute masked L2 loss (only on well-exposed regions)."""
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.mean()


class DequantizationTrainer:
    """Trainer for dequantization networks (8-bit → 16-bit reconstruction)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.logger = logger

        # Log system information
        self._log_system_info()

    def _log_system_info(self) -> None:
        """Log CPU, GPU, and CUDA information."""
        import psutil
        
        cpu_count = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        self.logger.info(f"System: {cpu_count} CPU cores ({cpu_threads} threads), {ram_gb:.1f} GB RAM")
        
        if self.device.type == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(self.device)
                vram_free, vram_total = torch.cuda.mem_get_info(self.device)
                self.logger.info(
                    f"CUDA: Using GPU '{gpu_name}' "
                    f"({vram_free / 1024**2:.0f}MB / {vram_total / 1024**2:.0f}MB VRAM free)"
                )
            else:
                self.logger.warning("CUDA device requested but torch.cuda.is_available() is False!")
        else:
            self.logger.info(f"Using CPU device: {self.device}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (ldr_batch, hdr_batch) in enumerate(dataloader):
            ldr_batch = ldr_batch.to(self.device)
            hdr_batch = hdr_batch.to(self.device)

            output = self.model(ldr_batch)
            mask = exposure_mask(ldr_batch)

            loss = masked_l2_loss(output, hdr_batch, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                self.logger.info(
                    f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str | Path | None = None,
        checkpoint_freq: int = 10,
    ) -> None:
        """Train the model for specified number of epochs."""
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting training...")
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_dataloader)
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}"
            )

            if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = checkpoint_dir / f"dequantization_net_epoch_{epoch + 1}.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        self.logger.info("Training complete!")

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f"Loaded checkpoint: {path}")
