"""Simple training script for Dequantization-Net.

Usage:
    python scripts/train_dequantization_net.py \
        --hdr_dir /path/to/hdr/images \
        --ldr_dir /path/to/ldr/images \
        --output_dir ./checkpoints \
        --batch_size 8 \
        --epochs 100 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from luminascale.models import create_dequantization_net
from luminascale.utils.io import image_to_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DequantizationDataset(Dataset):
    def __init__(
        self, hdr_dir: str | Path, ldr_dir: str | Path, file_pattern: str = "*.png"
    ) -> None:
        self.hdr_dir = Path(hdr_dir)
        self.ldr_dir = Path(ldr_dir)

        self.hdr_files = sorted(self.hdr_dir.glob(file_pattern))
        if not self.hdr_files:
            raise ValueError(f"No HDR images found in {hdr_dir}")

        logger.info(f"Found {len(self.hdr_files)} HDR-LDR pairs")

    def __len__(self) -> int:
        return len(self.hdr_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hdr_path = self.hdr_files[idx]
        ldr_path = self.ldr_dir / hdr_path.name

        if not ldr_path.exists():
            raise FileNotFoundError(f"LDR image not found: {ldr_path}")

        ldr_tensor = image_to_tensor(ldr_path)
        hdr_tensor = image_to_tensor(hdr_path)

        return ldr_tensor, hdr_tensor


def exposure_mask(
    img: torch.Tensor, threshold_bright: int = 249, threshold_dark: int = 6
) -> torch.Tensor:
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
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (ldr_batch, hdr_batch) in enumerate(dataloader):
        ldr_batch = ldr_batch.to(device)
        hdr_batch = hdr_batch.to(device)

        output = model(ldr_batch)

        mask = exposure_mask(ldr_batch)

        loss = masked_l2_loss(output, hdr_batch, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Dequantization-Net for 8-bit → 16-bit reconstruction"
    )
    parser.add_argument(
        "--hdr_dir",
        type=str,
        required=True,
        help="Directory containing ground-truth 16-bit HDR images",
    )
    parser.add_argument(
        "--ldr_dir",
        type=str,
        required=True,
        help="Directory containing 8-bit degraded LDR images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Checkpoint frequency (every N epochs)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"HDR directory: {args.hdr_dir}")
    logger.info(f"LDR directory: {args.ldr_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    dataset = DequantizationDataset(args.hdr_dir, args.ldr_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    model = create_dequantization_net(device=device, base_channels=16)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        logger.info(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"dequantization_net_epoch_{epoch + 1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
