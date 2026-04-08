"""Core trainer module for Dequantization-Net (WebDataset-only)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# from torch.utils.tensorboard import SummaryWriter (handled by Lightning)

from ..utils.dataset_pair_generator import DatasetPairGenerator
from ..utils.image_generator import create_primary_gradients, quantize_to_8bit

logger = logging.getLogger(__name__)


class LuminaScaleModule(L.LightningModule):
    """LightningModule for training Dequantization-Net via WebDataset."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        vis_freq: int = 5,
    ) -> None:
        super().__init__()
        print(f"[LuminaScaleModule] Initializing LightningModule...")
        self.model = model
        self.learning_rate = learning_rate
        self.vis_freq = vis_freq
        self.save_hyperparameters(ignore=["model"])
        self.pair_generator = None  # Lazy initialization for WebDataset batches
        self.crop_size = 512  # Default crop size for WebDataset batches
        
        # Track last batch metrics for progress bar
        self.last_batch_gpu_ms = None
        self.last_batch_loss = None
        self.estimated_total_batches = None  # Set by training script if metadata available
        
        # Cache for on-the-fly patch generation
        # When using .repeat() in WebDataset, consecutive items are the same image
        # We cache the decoded full image to avoid re-decoding for each patch
        self._last_image_id = None
        self._cached_srgb_8u = None
        self._cached_srgb_32f = None
        
        # Diagnostic counters for cache effectiveness
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_evals = 0
        
        # Enable manual optimization to do optimizer.step() after each crop
        self.automatic_optimization = False
        
        print(f"[LuminaScaleModule] ✓ Initialization complete")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Override batch transfer to handle WebDataset raw byte batches.
        
        WebDataset batches are (list[bytes], list[dict]) tuples.
        We skip device transfer for raw bytes since they're decoded on GPU in training_step.
        """
        # Check if this is a WebDataset batch: (list[bytes], list[dict])
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            first_elem = batch[0]
            if isinstance(first_elem, list) and len(first_elem) > 0:
                if isinstance(first_elem[0], bytes):
                    # This is a WebDataset batch - skip device transfer
                    return batch
        
        # Standard tensor batch: use default movement logic
        return batch

    def _process_batch(self, batch: tuple[list[bytes], list[dict]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw WDS batch (bytes) into graded training pairs (LDR, HDR) on GPU.
        
        With WebDataset.repeat(patches_per_image), consecutive items are from the same image.
        Implementation: Load full image once, cache it decoded/graded, reuse for all 32 patches
        by generating random crops from the cached version. Avoids 31 redundant EXR decodes.
        """
        # Lazy initialization of pair_generator
        if self.pair_generator is None:
            print(f"[_PROCESS_BATCH] Initializing DatasetPairGenerator on {self.device}...")
            self.pair_generator = DatasetPairGenerator(self.device)
        
        exr_bytes_list, metadata_list = batch
        
        # Ensure the device matches the model's current device
        if self.pair_generator.device != self.device:
            logger.info(f"Updating DatasetPairGenerator device to match model: {self.device}")
            self.pair_generator.device = self.device
            self.pair_generator.pytorch_transformer.device = self.device
            self.pair_generator.cdl_processor.device = self.device

        # Extract image IDs from metadata to detect repeated images
        image_ids = [m.get('id', f'img_{i}') for i, m in enumerate(metadata_list)]
        current_image_id = image_ids[0] if image_ids else None
        
        # Check if this is a repeated image (WebDataset.repeat() loops through same images)
        is_repeated_image = (current_image_id == self._last_image_id and self._cached_srgb_8u is not None)
        
        self._total_evals += 1
        
        if is_repeated_image:
            # CACHE HIT: Generate random crops from already-decoded/graded full image
            self._cache_hits += 1
            hit_rate = (self._cache_hits / self._total_evals * 100) if self._total_evals > 0 else 0
            logger.info(f"[CACHE HIT] Image '{current_image_id}' (Hit rate: {hit_rate:.1f}%)")
            srgb_8u_batch, srgb_32f_batch = self._generate_crops_from_cached_image(
                num_crops=len(exr_bytes_list),
                crop_size=self.crop_size
            )
        else:
            # CACHE MISS: Decode full image from EXR bytes, apply CDL, transform to sRGB
            self._cache_misses += 1
            hit_rate = (self._cache_hits / self._total_evals * 100) if self._total_evals > 0 else 0
            logger.info(f"[CACHE MISS] Image '{current_image_id}' (Hit rate: {hit_rate:.1f}%)")
            t_start = time.perf_counter()
            
            # Decode full image(s) into graded sRGB pairs
            srgb_8u_batch, srgb_32f_batch = self.pair_generator.generate_batch_from_bytes(
                exr_bytes_list, 
                crop_size=self.crop_size
            )
            decode_ms = (time.perf_counter() - t_start) * 1000
            logger.info(f"[DECODE TIME] Decoded image '{current_image_id}' in {decode_ms:7.1f}ms")
            
            # Cache the decoded/graded images for reuse
            if current_image_id and len(exr_bytes_list) == 1:
                self._cached_srgb_8u = srgb_8u_batch
                self._cached_srgb_32f = srgb_32f_batch
                self._last_image_id = current_image_id
        
        return srgb_8u_batch, srgb_32f_batch
    
    def _generate_crops_from_cached_image(self, num_crops: int, crop_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random crops from cached decoded/graded full image."""
        if self._cached_srgb_8u is None or self._cached_srgb_32f is None:
            raise RuntimeError("Cache miss - no cached image available")
        
        # Get cached full image: format is [batch_size, 3, H, W] or [3, H, W]
        cached_8u = self._cached_srgb_8u
        cached_32f = self._cached_srgb_32f
        
        # Extract single full image if batched
        if cached_8u.ndim == 4:
            full_8u = cached_8u[0]  # [3, H, W]
            full_32f = cached_32f[0]
        else:
            full_8u = cached_8u
            full_32f = cached_32f
        
        # Convert from [3, H, W] to [H, W, 3] for cropping
        full_8u = full_8u.permute(1, 2, 0)
        full_32f = full_32f.permute(1, 2, 0)
        
        H, W = full_8u.shape[0], full_8u.shape[1]
        
        # Generate num_crops random crops
        srgb_8u_crops = []
        srgb_32f_crops = []
        
        for _ in range(num_crops):
            if H <= crop_size or W <= crop_size:
                crop_8u = full_8u
                crop_32f = full_32f
            else:
                top = torch.randint(0, H - crop_size + 1, (1,), device=self.device).item()
                left = torch.randint(0, W - crop_size + 1, (1,), device=self.device).item()
                crop_8u = full_8u[top:top+crop_size, left:left+crop_size, :].clone()
                crop_32f = full_32f[top:top+crop_size, left:left+crop_size, :].clone()
            
            srgb_8u_crops.append(crop_8u)
            srgb_32f_crops.append(crop_32f)
        
        # Stack into batch [num_crops, H, W, 3] then convert to [num_crops, 3, H, W]
        srgb_8u_batch = torch.stack(srgb_8u_crops).permute(0, 3, 1, 2)
        srgb_32f_batch = torch.stack(srgb_32f_crops).permute(0, 3, 1, 2)
        
        return srgb_8u_batch, srgb_32f_batch

    def _train_on_image(
        self,
        x_full: torch.Tensor,
        y_full: torch.Tensor,
        batch_idx: int,
        num_patches: int = 32,
    ) -> tuple[float, float]:
        """Train on a single full image by generating random crops and gradient updates.
        
        Strategy:
        1. Receives full image [3, H, W] (already decoded with CDL applied)
        2. Generates num_patches random crops
        3. For each crop: forward → loss → backward → optimizer.step()
        4. Returns average loss and elapsed time (ms)
        
        Args:
            x_full: Full 8-bit image [3, H, W]
            y_full: Full 32-bit target [3, H, W]
            batch_idx: Batch index for logging
            num_patches: Number of patches to generate (default 32 for WDS)
        
        Returns:
            (avg_loss, elapsed_time_ms)
        """
        t_start = time.perf_counter()
        optimizer = self.optimizers()
        crop_losses = []
        
        C, H, W = x_full.shape
        crop_size = self.crop_size
        
        logger.info(f"[BATCH {batch_idx}] Generating {num_patches} crops and {num_patches} gradient updates...")
        
        for patch_idx in range(num_patches):
            # Generate random crop
            if H <= crop_size or W <= crop_size:
                x_crop = x_full
                y_crop = y_full
            else:
                top = torch.randint(0, H - crop_size + 1, (1,)).item()
                left = torch.randint(0, W - crop_size + 1, (1,)).item()
                x_crop = x_full[:, top:top+crop_size, left:left+crop_size]
                y_crop = y_full[:, top:top+crop_size, left:left+crop_size]
            
            # Add batch dimension [1, 3, H, W]
            x_crop = x_crop.unsqueeze(0)
            y_crop = y_crop.unsqueeze(0)
            
            # Ensure float32
            if x_crop.dtype == torch.uint8:
                x_crop = x_crop.float() / 255.0
            if y_crop.dtype == torch.uint8:
                y_crop = y_crop.float() / 255.0
            
            # Forward pass
            y_hat = self.model(x_crop)
            
            # Compute loss
            mask = exposure_mask(y_crop)
            crop_loss = masked_l2_loss(y_hat, y_crop, mask)
            crop_losses.append(crop_loss.item())
            
            # Backward + optimizer step
            optimizer.zero_grad()
            self.manual_backward(crop_loss)
            optimizer.step()
            
            # DEBUG: Log first crop of first batch
            if batch_idx == 0 and patch_idx == 0:
                mask_ratio = mask.sum().item() / mask.numel()
                logger.info(
                    f"[DEBUG CROP 0] Loss: {crop_loss.item():.4e}, "
                    f"Mask coverage: {mask_ratio*100:.1f}%"
                )
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        avg_loss = sum(crop_losses) / num_patches
        
        logger.info(
            f"[BATCH {batch_idx}] ✓ Completed {num_patches} patches with {num_patches} updates. "
            f"Avg loss: {avg_loss:.4e}, GPU time: {elapsed_ms:.1f}ms"
        )
        
        return avg_loss, elapsed_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[list[bytes], list[dict]], batch_idx: int) -> torch.Tensor:
        """Training step for WebDataset batches.
        
        Each batch from WebDataset contains one image repeated 32 times.
        Strategy: Decode once, generate 32 crops, do 32 gradient updates per image.
        """
        try:
            self.last_batch_gpu_ms = None
            
            # Decode full image once
            x_full, y_full = self._process_batch(batch)  # [1, 3, H, W]
            
            # Unbatch to [3, H, W]
            if x_full.ndim == 4:
                x_full = x_full[0]
                y_full = y_full[0]
            
            # Train on image with 32 patches and gradient updates
            avg_loss, elapsed_ms = self._train_on_image(
                x_full, y_full, batch_idx, num_patches=32
            )
            
            self.last_batch_gpu_ms = elapsed_ms
            self.last_batch_loss = avg_loss
            
            # Log to TensorBoard
            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_scalar('loss/train', avg_loss, self.global_step)
                self.logger.experiment.flush()
            
            return torch.tensor(avg_loss, device=self.device)
            
        except Exception as e:
            logger.error(f"[Batch {batch_idx}] Exception: {type(e).__name__}: {e}")
            print(f"[TRAINING_STEP] Exception: {e}")
            import traceback
            traceback.print_exc()
            raise

    def on_train_epoch_end(self) -> None:
        """Log synthetic visualizations at the end of each epoch."""
        if (self.current_epoch + 1) % self.vis_freq != 0:
            return

        self.model.eval()
        with torch.no_grad():
            # Prepare synthetic test data
            prim_hdr = create_primary_gradients(width=512, height=512, dtype="float32")
            prim_8bit = quantize_to_8bit(prim_hdr)

            # Run inference
            input_tensor = torch.from_numpy(prim_8bit).float().unsqueeze(0).to(self.device)
            output_tensor = self.model(input_tensor).squeeze(0)

            # Prepare for plotting [H, W, 3]
            in_np = np.transpose(prim_8bit, (1, 2, 0))
            out_np = np.transpose(output_tensor.cpu().numpy(), (1, 2, 0))
            gt_np = np.transpose(prim_hdr, (1, 2, 0))

            # Boost contrast to reveal banding
            contrast_factor = 25.0
            in_c = np.clip((in_np - 0.5) * contrast_factor + 0.5, 0, 1)
            out_c = np.clip((out_np - 0.5) * contrast_factor + 0.5, 0, 1)
            gt_c = np.clip((gt_np - 0.5) * contrast_factor + 0.5, 0, 1)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0, 0].imshow(in_np)
            axes[0, 0].set_title("Input (8-bit)")
            axes[0, 1].imshow(out_np)
            axes[0, 1].set_title("Output (Expanded)")
            axes[0, 2].imshow(gt_np)
            axes[0, 2].set_title("GT (32-bit)")

            axes[1, 0].imshow(in_c)
            axes[1, 0].set_title(f"Input {contrast_factor}x Contrast")
            axes[1, 1].imshow(out_c)
            axes[1, 1].set_title(f"Output {contrast_factor}x Contrast")
            axes[1, 2].imshow(gt_c)
            axes[1, 2].set_title(f"GT {contrast_factor}x Contrast")

            for ax in axes.ravel():
                ax.axis("off")

            plt.tight_layout()
            
            # Log to TensorBoard
            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.add_figure("Visualizations", fig, global_step=self.global_step)
            plt.close(fig)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)


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
    return mask.float()


def masked_l2_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked L2 loss to ignore clipped regions."""
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss
