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
from ..utils.image_generator import create_primary_gradients, quantize_to_8bit, apply_s_curve_contrast_torch

logger = logging.getLogger(__name__)

class DequantizationTrainer(L.LightningModule):
    """LightningModule for training Dequantization-Net."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        vis_freq: int = 5,
        loss_weights: dict | None = None,
    ) -> None:
        super().__init__()
        print(f"[DequantizationTrainer] Initializing LightningModule...")
        self.model = model
        self.learning_rate = learning_rate
        self.vis_freq = vis_freq
        
        # Loss weights for three-term loss
        if loss_weights is None:
            loss_weights = {
                "l1_weight": 1.0,
                "l2_weight": 0.0,
                "charbonnier_weight": 3.0,
                "grad_match_weight": 2.0,
            }
        self.l1_weight = loss_weights.get("l1_weight", 1.0)
        self.l2_weight = loss_weights.get("l2_weight", 0.0)
        self.charbonnier_weight = loss_weights.get("charbonnier_weight", 3.0)
        self.grad_match_weight = loss_weights.get("grad_match_weight", 2.0)
        
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
        
        print(f"[DequantizationTrainer] ✓ Initialization complete")

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

            # === Revised two-term loss for smooth dequantization ===
            # Problem: Strict gradient matching forced conflicting constraints → worse outputs (3,845 unique)
            # Solution: Use structural smoothing WITHOUT forcing exact gradient matching
            
            # Total loss = reconstruction + global smoothing + edge awareness
            loss = l1_loss(y_hat, y) * self.l1_weight + l2_loss(y_hat, y) * self.l2_weight + charbonnier_loss(y_hat) * self.charbonnier_weight + edge_aware_smoothing_loss(y_hat, y) * self.grad_match_weight
            
            # Debug: Check loss composition
            if batch_idx == 0 and self.current_epoch % 1 == 0:
                print(f"\n[Loss Composition] L1: {l1_loss(y_hat, y).item():.6f} | L2: {l2_loss(y_hat, y).item():.6f} | Charbonnier: {charbonnier_loss(y_hat).item():.6f} | EdgeAware: {edge_aware_smoothing_loss(y_hat, y).item():.6f}")
                print(f"  [Structure] Charbonnier loss (smooth L1) on output: gentle on edges, strict on banding")
                print(f"  [Reasoning] Hard L1 TV was destroying edges; Charbonnier preserves them")
                print(f"  [Strategy] Softer smoothness + edge preservation + edge-aware constraints")

            # Debug: Verify we're training against correct target (not just reproducing input)
            if batch_idx == 0 and self.current_epoch % 1 == 0:  # Log every epoch, batch 0
                x_mean, x_std = x.mean().item(), x.std().item()
                y_mean, y_std = y.mean().item(), y.std().item()
                yhat_mean, yhat_std = y_hat.mean().item(), y_hat.std().item()
                y_minus_x = (y - x).abs().mean().item()
                yhat_minus_x = (y_hat - x).abs().mean().item()
                yhat_minus_y = (y_hat - y).abs().mean().item()
                
                # Count unique values to see if input and target actually differ
                x_flat = x.flatten().detach().cpu().numpy()
                y_flat = y.flatten().detach().cpu().numpy()
                yhat_flat = y_hat.flatten().detach().cpu().numpy()
                x_unique = len(np.unique(np.round(x_flat, decimals=6)))
                y_unique = len(np.unique(np.round(y_flat, decimals=6)))
                yhat_unique = len(np.unique(np.round(yhat_flat, decimals=6)))
                
                print(f"\n[Epoch {self.current_epoch} Batch {batch_idx}] Loss Breakdown & Target Sanity:")
                print(f"  L1 Loss:        {l1_loss(y_hat, y).item():.6f} (pixel-wise reconstruction)")
                print(f"  L2 Loss:        {l2_loss(y_hat, y).item():.6f} (MSE: mean squared error)")
                print(f"  Charbonnier:    {charbonnier_loss(y_hat).item():.6f} (smooth L1: edge-preserving smoothness)")
                print(f"  EdgeAware:      {edge_aware_smoothing_loss(y_hat, y).item():.6f} (no artifacts in smooth regions)")
                print(f"  Total:          {loss.item():.6f}")
                print(f"  Input (x)  - mean: {x_mean:.5f}, std: {x_std:.5f}, unique values: {x_unique}")
                print(f"  Target (y) - mean: {y_mean:.5f}, std: {y_std:.5f}, unique values: {y_unique}")
                print(f"  Output (ŷ) - mean: {yhat_mean:.5f}, std: {yhat_std:.5f}, unique values: {yhat_unique}")
                print(f"  |y - x|    - mean: {y_minus_x:.6f} (target diff from input)")
                print(f"  |ŷ - x|    - mean: {yhat_minus_x:.6f} (output diff from input)")
                print(f"  |ŷ - y|    - mean: {yhat_minus_y:.6f} (output error vs target)")
                if y_minus_x < 0.001:
                    print(f"  WARNING: Target is TOO SIMILAR to input! (diff={y_minus_x:.6f})")

            self.log("loss_L1/train", l1_loss(y_hat, y), prog_bar=False, sync_dist=True)
            self.log("loss_L2/train", l2_loss(y_hat, y), prog_bar=False, sync_dist=True)
            self.log("loss_Charbonnier/train", charbonnier_loss(y_hat), prog_bar=False, sync_dist=True)
            self.log("loss_EdgeAware/train", edge_aware_smoothing_loss(y_hat, y), prog_bar=False, sync_dist=True)
            self.log("loss_total/train", loss, prog_bar=False, sync_dist=True)
            
            # Log weight-independent metrics (PSNR, SSIM) for fair hparam comparison
            # These remain constant regardless of loss weight changes
            psnr_val = compute_psnr(y_hat, y)
            self.log("metric_psnr/train", psnr_val, prog_bar=False, sync_dist=True)
            
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
        """Log synthetic visualizations at the end of each epoch.
        
        Note: Actual visualization is handled by SyntheticInferenceVisualizerCallback
        in train_dequantization_net.py. This method is a no-op.
        """
        pass

    def configure_optimizers(self) -> dict:
        """Configure optimizer with CosineAnnealingLR scheduler.
        
        Uses PyTorch's standard CosineAnnealingLR for epoch-based learning rate decay.
        Decays smoothly from base_lr to eta_min over training_epochs.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Get total epochs from trainer (set during setup)
        num_epochs = 5  # Default fallback
        if hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "max_epochs"):
            num_epochs = self.trainer.max_epochs or 5
        
        print(f"\n[configure_optimizers] Scheduler Setup:")
        print(f"  Base LR: {self.learning_rate}")
        print(f"  Total Epochs: {num_epochs}")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  Target LR (eta_min): 1e-6")
        
        # Create standard PyTorch CosineAnnealingLR scheduler
        # Decays from base_lr to eta_min=1e-6 over num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unmasked L1 (MAE) loss - more robust to outliers.
    
    L1 loss provides sharper transitions and can be more effective than L2
    for dequantization tasks where we want to preserve edges and avoid
    over-smoothing. Less sensitive to outliers compared to L2.
    """
    return F.l1_loss(pred.float(), target.float())


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


def charbonnier_loss(img: torch.Tensor, reduction: str = "mean", eps: float = 1e-3) -> torch.Tensor:
    """Compute Total Variation using Charbonnier (smooth L1) loss.
    
    Uses Charbonnier instead of hard L1 abs. This penalizes small gradients 
    heavily (quantization artifacts) but allows large gradients (real edges) 
    without over-smoothing.
    
    Charbonnier: sqrt(x² + eps²) is smooth everywhere and differentiable,
    unlike abs(x) which has a sharp corner at x=0.
    
    Args:
        img: Image tensor [B, C, H, W]
        reduction: 'mean' or 'sum'
        eps: Smoothing parameter (smaller = more like L1, larger = more like L2)
    
    Returns:
        Charbonnier total variation loss
    """
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    
    # Charbonnier: sqrt(x² + eps²) - smooth approximation to |x|
    # Gentler on large values (preserves edges), harsh on small values (removes banding)
    dy_smooth = torch.sqrt(dy**2 + eps**2)
    dx_smooth = torch.sqrt(dx**2 + eps**2)
    
    if reduction == "mean":
        return (dy_smooth.mean() + dx_smooth.mean()) / 2.0
    else:
        return (dy_smooth.sum() + dx_smooth.sum()) / 2.0


def edge_aware_smoothing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Penalize inconsistency in edge structure (where gradients exist or don't).
    
    Instead of forcing gradients to match exactly, this checks if there are edges
    where they should be (in target) and no edges where there shouldn't be.
    
    This is softer than gradient matching but stronger than plain TV.
    
    Args:
        pred: Model prediction [B, C, H, W]
        target: Target image [B, C, H, W]
    
    Returns:
        Edge consistency loss
    """
    # Compute gradient magnitude (how "edgy" each location is)
    grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    
    # Where target has NO gradients (smooth regions), penalize pred more heavily
    # This forces smooth regions to stay smooth without over-constraining edges
    smooth_mask_y = (grad_target_y < 0.01).float()  # 0 = smooth region in target
    smooth_mask_x = (grad_target_x < 0.01).float()
    
    # In smooth regions, large gradients are bad
    loss_y = (smooth_mask_y * grad_pred_y).mean()
    loss_x = (smooth_mask_x * grad_pred_x).mean()
    
    return (loss_y + loss_x) / 2.0


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio between prediction and target.
    
    PSNR is weight-independent and suitable for hyperparameter tuning.
    Higher PSNR indicates better reconstruction quality.
    
    PSNR = 20 * log10(max_val / sqrt(MSE))
    
    Args:
        pred: Model prediction [B, C, H, W] in range [0, 1]
        target: Target image [B, C, H, W] in range [0, 1]
        max_val: Maximum pixel value (default 1.0 for normalized images)
    
    Returns:
        PSNR value in dB (higher is better)
    """
    mse = F.mse_loss(pred.float(), target.float())
    # Avoid log(0) by adding small epsilon
    psnr = 20.0 * torch.log10(max_val / (torch.sqrt(mse) + 1e-10))
    return psnr