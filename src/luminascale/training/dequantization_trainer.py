"""Core trainer classes for dequantization and bit-depth expansion models."""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from luminascale.utils.dataset_pair_generator import DatasetPairGenerator
from luminascale.utils.look_generator import get_single_random_look

logger = logging.getLogger(__name__)


class OnTheFlyBDEDataset(Dataset):
    """On-the-fly BDE (Bit-Depth Expansion) dataset with per-image CDL grading.
    
    Generates training pairs on-the-fly during iteration:
    - Loads ACES from LMDB once per image
    - Applies ONE random CDL to the full image (per-image, not per-patch)
    - Transforms full image to sRGB (both 32-bit and 8-bit) once
    - Generates random crops from the cached graded sRGB pair (64 crops per image)
    
    This is ~10x faster than per-patch CDL since expensive GPU ops happen once per image.
    
    Returns:
        (input, target) where:
        - input: sRGB 8-bit (degraded, to be expanded) [3, H, W]
        - target: sRGB 32-bit (ground truth quality) [3, H, W]
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        device: torch.device,
        crop_size: int = 512,
        patches_per_image: int = 1,
    ) -> None:
        self.lmdb_path = Path(lmdb_path).resolve()
        assert self.lmdb_path.exists(), f"LMDB not found: {self.lmdb_path}"
        
        self.device = device
        self.crop_size = crop_size
        self.patches_per_image = max(1, patches_per_image)
        
        # Open LMDB in read-only mode
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        with self.env.begin(write=False) as txn:
            keys_buf = txn.get(b"__keys__")
            assert keys_buf is not None, "LMDB missing __keys__"
            self.keys = pickle.loads(keys_buf)
        
        # Initialize GPU pipeline (ACES load → CDL → OCIO)
        self.pair_generator = DatasetPairGenerator(self.env, self.device, self.keys)
        
        # Image cache: store (srgb_32f, srgb_8u) after full pipeline (reuse across 64 patches)
        self._last_img_idx = -1
        self._cached_srgb_32f = None
        self._cached_srgb_8u = None
        
        logger.info(
            f"Initialized OnTheFlyBDEDataset with {len(self.keys)} images. "
            f"Generating {patches_per_image} patches per image ({len(self)} total samples)."
        )
    
    def __len__(self) -> int:
        return len(self.keys) * self.patches_per_image
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a random crop from a per-image graded sRGB pair.
        
        Returns:
            (input, target) = (srgb_8u, srgb_32f) with shapes [3, H, W] on GPU, normalized to [0, 1]
        """
        # Map flat index to image index
        img_idx = (idx // self.patches_per_image) % len(self.keys)
        
        # Load ACES, apply CDL + OCIO once per image (cache to reuse across 64 patches)
        if img_idx != self._last_img_idx:
            key = self.keys[img_idx]
            try:
                # One random CDL per image
                cdl_params = get_single_random_look()
                
                # Full pipeline: load ACES → apply CDL → transform to sRGB
                srgb_32f, srgb_8u = self.pair_generator.load_aces_apply_cdl_and_transform(key, cdl_params)
                
                # Cache the full graded sRGB for this image
                self._cached_srgb_32f = srgb_32f  # [H, W, 3]
                self._cached_srgb_8u = srgb_8u    # [H, W, 3]
                self._last_img_idx = img_idx
                
            except Exception as e:
                logger.error(f"Failed to load image {key}: {e}")
                raise
        
        # Random crop from cached sRGB pair
        srgb_32f = self._cached_srgb_32f
        srgb_8u = self._cached_srgb_8u
        assert srgb_32f is not None and srgb_8u is not None, "Failed to load sRGB tensors"
        
        H, W = srgb_32f.shape[0], srgb_32f.shape[1]
        
        # Random crop
        if H < self.crop_size or W < self.crop_size:
            # Upscale if image is smaller than crop
            srgb_32f_reshaped = srgb_32f.permute(2, 0, 1).unsqueeze(0)
            srgb_8u_reshaped = srgb_8u.permute(2, 0, 1).unsqueeze(0)
            
            srgb_32f = F.interpolate(
                srgb_32f_reshaped,
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            
            srgb_8u = F.interpolate(
                srgb_8u_reshaped,
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        else:
            # Random crop
            top = torch.randint(0, H - self.crop_size + 1, (1,), device=self.device).item()
            left = torch.randint(0, W - self.crop_size + 1, (1,), device=self.device).item()
            srgb_32f = srgb_32f[top : top + self.crop_size, left : left + self.crop_size, :]
            srgb_8u = srgb_8u[top : top + self.crop_size, left : left + self.crop_size, :]
        
        # Both are [crop_size, crop_size, 3]
        # Convert to float32, normalize 8-bit to [0, 1], and permute to [3, H, W]
        srgb_8u = (srgb_8u.float() / 255.0).permute(2, 0, 1)  # [0, 255] → [0, 1], then [H, W, 3] → [3, H, W]
        srgb_32f = srgb_32f.float().permute(2, 0, 1)  # [H, W, 3] → [3, H, W]
        
        # Return (input, target) = (8-bit input to expand, 32-bit ground truth)
        return srgb_8u, srgb_32f
    
    def cleanup(self) -> None:
        """Release GPU resources."""
        self.pair_generator.cleanup()


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
            keys_buf = txn.get(b"__keys__")
            assert keys_buf is not None, "LMDB missing __keys__"
            self.keys = pickle.loads(keys_buf)
        
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
                buf = txn.get(key.encode("ascii"))
                assert buf is not None, f"Key not found: {key}"
                data = pickle.loads(buf)
            
            # Load and move to device immediately
            # Note: "hdr" is the LMDB dict key for 32-bit ACES data (created by dataset baker)
            aces_32bit_np = data["hdr"]  # Shape: [H, W, 3], range: [0, ~10+]
            self._cached_aces_32bit = torch.from_numpy(aces_32bit_np).permute(2, 0, 1).float().to(self.device)
            self._last_img_idx = img_idx
        
        aces_32bit = self._cached_aces_32bit
        assert aces_32bit is not None, "Failed to load ACES tensor"
        
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
            
            # Timing tracking
            data_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            batch_start = time.time()
            
            for i, (input, target) in enumerate(train_dataloader):
                data_time += time.time() - batch_start
                
                input, target = input.to(self.device), target.to(self.device)

                # Forward pass
                forward_start = time.time()
                self.optimizer.zero_grad()
                output = self.model(input)
                
                # Compute loss (simple L2 on all pixels; masked loss can be added later with proper thresholds)
                loss = F.mse_loss(output, target)
                forward_time += time.time() - forward_start
                
                # Backward pass
                backward_start = time.time()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                backward_time += time.time() - backward_start

                running_loss += loss.item()
                
                # Log to TensorBoard
                if self.writer:
                    global_step = epoch * len(train_dataloader) + i
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)
                    
                    # Log learning rate
                    for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar("LearningRate", param_group["lr"], global_step)
                
                if (i + 1) % 5 == 0:
                    avg_data_time = data_time / (i + 1)
                    avg_forward_time = forward_time / (i + 1)
                    avg_backward_time = backward_time / (i + 1)
                    avg_total = (data_time + forward_time + backward_time) / (i + 1)
                    
                    logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                        f"Loss: {loss.item():.6f} | "
                        f"Data: {avg_data_time:.3f}s | Forward: {avg_forward_time:.3f}s | "
                        f"Backward: {avg_backward_time:.3f}s | Total: {avg_total:.3f}s/batch"
                    )
                
                batch_start = time.time()

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
