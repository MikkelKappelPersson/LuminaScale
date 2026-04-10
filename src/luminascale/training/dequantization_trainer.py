"""Core trainer classes for dequantization and bit-depth expansion models."""

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

from ..data.wds_dataset import LuminaScaleWebDataset
from ..utils.dataset_pair_generator import DatasetPairGenerator
from ..utils.image_generator import create_primary_gradients, quantize_to_8bit

logger = logging.getLogger(__name__)

class LuminaScaleModule(L.LightningModule):
    """LightningModule for training Dequantization-Net."""

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
        # Note: Do NOT call save_hyperparameters() here - full config is logged at training end via HparamsMetricsCallback
        self.pair_generator = None  # Lazy initialization for WebDataset batches
        self.crop_size = 512  # Default crop size for WebDataset batches
        # Track last batch metrics for progress bar
        self.last_batch_gpu_ms = None
        self.last_batch_loss = None
        self.estimated_total_batches = None  # Set by training script if metadata available
        
        # Device: always CUDA for training
        self.device_cuda = torch.device("cuda")
        
        # Cache for on-the-fly patch generation with WebDataset
        # When using .repeat() in WebDataset, consecutive items are the same image
        # We cache the decoded full image to avoid re-decoding for each patch
        self._last_image_id = None
        self._cached_srgb_8u = None
        self._cached_srgb_32f = None
        
        print(f"[LuminaScaleModule] ✓ Initialization complete")

    def setup(self, stage: str) -> None:
        """Lightning setup hook called before training/validation starts.
        
        Ensures model is in train mode.
        """
        # Force model to be in train mode
        self.model.train()
        logger.info(f"[setup] Model on device: {next(self.model.parameters()).device}")

    def _get_device(self) -> torch.device:
        """Get the device for tensor operations.
        
        Always returns CUDA since we only train on GPUs.
        """
        return self.device_cuda

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
        # For tensor batches, Lightning handles the device transfer automatically
        # Just return the batch as-is for tensors; Lightning will move them
        return batch

    def _process_batch(self, batch: tuple[list[bytes], list[dict]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw WDS batch (bytes) into graded training pairs (LDR, HDR) on GPU.
        
        With WebDataset.repeat(patches_per_image), consecutive items are from the same image.
        Implementation: Load full image once, cache it decoded/graded, reuse for all 32 patches
        by generating random crops from the cached version. Avoids 31 redundant EXR decodes.
        """
        # Lazy initialization of pair_generator
        device = self._get_device()
        
        if self.pair_generator is None:
            print(f"[_PROCESS_BATCH] Initializing DatasetPairGenerator on {device}...")
            self.pair_generator = DatasetPairGenerator(device)
        
        exr_bytes_list, metadata_list = batch
        
        # Ensure the device matches the model's current device
        # Update device if it changed (e.g., model moved to different GPU)
        current_device = self._get_device()
        if self.pair_generator.device != current_device:
            logger.info(f"Updating DatasetPairGenerator device to match model: {current_device}")
            self.pair_generator.device = current_device
            self.pair_generator.pytorch_transformer.device = current_device
            self.pair_generator.cdl_processor.device = current_device

        # Extract image IDs from metadata to detect repeated images
        image_ids = [m.get('id', f'img_{i}') for i, m in enumerate(metadata_list)]
        current_image_id = image_ids[0] if image_ids else None
        
        # Check if this is a repeated image (WebDataset.repeat() loops through same images)
        is_repeated_image = (current_image_id == self._last_image_id and self._cached_srgb_8u is not None)
        
        if is_repeated_image:
            # CACHE HIT: Generate random crops from already-decoded/graded full image
            # This avoids the expensive EXR decode + CDL + color transform pipeline
            logger.debug(f"[PROCESS BATCH] Cache HIT for image '{current_image_id}' - reusing decoded image ({len(exr_bytes_list)} crops)")
            srgb_8u_batch, srgb_32f_batch = self._generate_crops_from_cached_image(
                num_crops=len(exr_bytes_list),
                crop_size=self.crop_size,
                device=current_device
            )
        else:
            # CACHE MISS: Decode full image from EXR bytes, apply CDL, transform to sRGB, then cache
            logger.debug(f"[PROCESS BATCH] Cache MISS - decoding image '{current_image_id}'")
            t_start = time.perf_counter()
            
            # Decode full image(s) into graded sRGB pairs
            srgb_8u_batch, srgb_32f_batch = self.pair_generator.generate_batch_from_bytes(
                exr_bytes_list, 
                crop_size=self.crop_size
            )
            decode_ms = (time.perf_counter() - t_start) * 1000
            logger.debug(f"[PROCESS BATCH] Decoded and generated {len(exr_bytes_list)} image pairs in {decode_ms:7.1f}ms")
            
            # Cache the decoded/graded images for reuse on subsequent patches loops
            if current_image_id and len(exr_bytes_list) == 1:
                self._cached_srgb_8u = srgb_8u_batch
                self._cached_srgb_32f = srgb_32f_batch
                self._last_image_id = current_image_id
                logger.debug(f"[PROCESS BATCH] Cached image '{current_image_id}' for reuse")
        
        return srgb_8u_batch, srgb_32f_batch
    
    def _generate_crops_from_cached_image(self, num_crops: int, crop_size: int = 512, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random crops from cached decoded/graded full image.
        
        This is called when WebDataset.repeat() shows the same image again.
        Avoids expensive EXR decode + CDL + color transform by reusing the cached result.
        
        Args:
            num_crops: Number of random crops to generate
            crop_size: Size of each crop (default 512)
            device: Device to create random tensor on (defaults to model device)
        """
        if device is None:
            device = self._get_device()
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
        full_8u = full_8u.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        full_32f = full_32f.permute(1, 2, 0)
        
        H, W = full_8u.shape[0], full_8u.shape[1]
        
        # Generate num_crops random crops from the full image
        srgb_8u_crops = []
        srgb_32f_crops = []
        
        for _ in range(num_crops):
            if H <= crop_size or W <= crop_size:
                # Image smaller than crop size - use full image
                crop_8u = full_8u
                crop_32f = full_32f
            else:
                # Generate random crop location
                top = torch.randint(0, H - crop_size + 1, (1,), device=device).item()
                left = torch.randint(0, W - crop_size + 1, (1,), device=device).item()
                crop_8u = full_8u[top:top+crop_size, left:left+crop_size, :].clone()
                crop_32f = full_32f[top:top+crop_size, left:left+crop_size, :].clone()
            
            srgb_8u_crops.append(crop_8u)
            srgb_32f_crops.append(crop_32f)
        
        # Stack into batch [num_crops, H, W, 3] then convert to [num_crops, 3, H, W]
        srgb_8u_batch = torch.stack(srgb_8u_crops).permute(0, 3, 1, 2)  # [N, H, W, 3] -> [N, 3, H, W]
        srgb_32f_batch = torch.stack(srgb_32f_crops).permute(0, 3, 1, 2)
        
        return srgb_8u_batch, srgb_32f_batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        try:
            self.last_batch_gpu_ms = None
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    if isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                        x, y = batch
                    # WebDataset case: batch is (exr_bytes_list, metadata_list)
                    elif isinstance(batch[0], list) and len(batch[0]) > 0 and isinstance(batch[0][0], bytes):
                        t_start = time.perf_counter()
                        x, y = self._process_batch(batch)
                        self.last_batch_gpu_ms = (time.perf_counter()-t_start)*1000
                    else:
                        x, y = batch
                else:
                    raise ValueError(f"Batch tuple has {len(batch)} elements, expected 2")
            else:
                raise ValueError(f"Batch type {type(batch)} is not tuple/list")
            
            y_hat = self.model(x)

            # Use unmasked L2 loss
            loss = l2_loss(y_hat, y)

            self.log("loss_L2/train", loss, prog_bar=False, sync_dist=True)  # Log to Lightning logger
            
            # Log current learning rate (supports dynamic LR scheduling)
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", current_lr, prog_bar=False, sync_dist=True)
            
            # Store metrics for progress bar display
            self.last_batch_loss = loss.item()
            return loss
            
        except Exception as e:
            logger.error(f"[Batch {batch_idx:3d}] Exception: {type(e).__name__}: {e}")
            print(f"[TRAINING_STEP] Exception message: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            raise

    def on_train_epoch_end(self) -> None:
        """Log synthetic visualizations at the end of each epoch."""
        if (self.current_epoch + 1) % self.vis_freq != 0:
            return

        self.model.eval()
        with torch.no_grad():
            # 1. Prepare Synthetic Data (RGB Primaries)
            prim_hdr = create_primary_gradients(width=512, height=512, dtype="float32")
            prim_8bit = quantize_to_8bit(prim_hdr)

            # 2. Run Inference
            input_tensor = (
                torch.from_numpy(prim_8bit).float().unsqueeze(0).to(self._get_device())
            )
            output_tensor = self.model(input_tensor).squeeze(0)

            # 3. Prepare for plotting [H, W, 3]
            in_np = np.transpose(prim_8bit, (1, 2, 0))
            out_np = np.transpose(output_tensor.cpu().numpy(), (1, 2, 0))
            gt_np = np.transpose(prim_hdr, (1, 2, 0))

            # Boost contrast to reveal banding (25x)
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
    mask = mask.float()

    return mask


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unmasked L2 (MSE) loss - default loss function.
    
    Provides training signal across the full tonal range without masking.
    This is superior to masked L2 as it trains the model on all pixel values,
    including the extremes where dequantization is most visible.
    """
    return F.mse_loss(pred.float(), target.float())


def masked_l2_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked L2 loss to ignore clipped regions (deprecated in favor of l2_loss)."""
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss