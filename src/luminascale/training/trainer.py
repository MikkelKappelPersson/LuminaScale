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
    """Dataset for paired LDR (low-quality) and HDR (high-quality) images."""

    def __init__(
        self, hdr_dir: str | Path, ldr_dir: str | Path, file_pattern: str = "*.png"
    ) -> None:
        from luminascale.utils.io import image_to_tensor

        self.hdr_dir = Path(hdr_dir)
        self.ldr_dir = Path(ldr_dir)

        self.hdr_files = sorted(self.hdr_dir.glob(file_pattern))
        if not self.hdr_files:
            raise ValueError(f"No HDR images found in {hdr_dir}")

        logger.info(f"Found {len(self.hdr_files)} HDR-LDR pairs")
        self.image_to_tensor = image_to_tensor

    def __len__(self) -> int:
        return len(self.hdr_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hdr_path = self.hdr_files[idx]
        ldr_path = self.ldr_dir / hdr_path.name

        if not ldr_path.exists():
            raise FileNotFoundError(f"LDR image not found: {ldr_path}")

        ldr_tensor = self.image_to_tensor(ldr_path)
        hdr_tensor = self.image_to_tensor(hdr_path)

        return ldr_tensor, hdr_tensor


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
                f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}"
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
