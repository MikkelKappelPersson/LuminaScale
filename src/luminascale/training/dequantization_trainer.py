"""Core trainer classes for dequantization and bit-depth expansion models."""

from __future__ import annotations

import gc
import logging
import os
import pickle
import time
from pathlib import Path

import pytorch_lightning as L
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter (handled by Lightning)

from ..data.wds_dataset import LuminaScaleWebDataset
from ..utils.dataset_pair_generator import DatasetPairGenerator
from ..utils.look_generator import get_single_random_look
from ..utils.image_generator import create_primary_gradients, create_sky_gradient, quantize_to_8bit

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
        device: torch.device | str | None = None,
        crop_size: int = 512,
        patches_per_image: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.lmdb_path = Path(lmdb_path).resolve()
        assert self.lmdb_path.exists(), f"LMDB not found: {self.lmdb_path}"
        
        self.device = torch.device(device) if device else None
        self.crop_size = crop_size
        self.patches_per_image = max(1, patches_per_image)
        self.rank = rank
        self.world_size = world_size
        
        # Open LMDB in read-only mode with optimized settings for large images
        # readahead=False prevents OS page thrashing on 500MB+ images
        # map_size=512GB accommodates sequential large-value reads
        self.env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,  # Disable to prevent page thrashing with 269MB+ images
            meminit=False,
            map_size=512 * 1024 * 1024 * 1024,  # 512GB: delay degradation further
        )
        
        with self.env.begin(write=False) as txn:
            keys_buf = txn.get(b"__keys__")
            assert keys_buf is not None, "LMDB missing __keys__"
            self.keys = pickle.loads(keys_buf)
        
        # Initialize GPU pipeline (ACES load → CDL → OCIO)
        self.pair_generator = None
        
        # Image cache: store (srgb_32f, srgb_8u) after full pipeline (reuse across 64 patches)
        self._last_img_idx = -1
        self._cached_srgb_32f = None
        self._cached_srgb_8u = None
        
        # Timing statistics (kept minimal to avoid memory bloat)
        self._timings = {"lmdb_load": [], "gpu_transfer": [], "cdl": [], "aces_transform": [], "crop": []}
        self._batch_count = 0
        self._report_interval = 50  # Report timings every 50 batches (without expensive empty_cache calls)
        
        # LMDB connection: Note - refresh doesn't help with large images (>4GB each)
        # LMDB memory mapping thrashes on images this large. Consider pre-loading into RAM
        # or reducing image sizes for acceptable training speed.
        self._lmdb_slow_threshold_batches = 1500  # Track when slowdown begins
        
        # Track LMDB read performance to detect degradation
        self._lmdb_read_times = []  # Recent LMDB read times for diagnostics
        self._lmdb_slow_read_threshold_ms = 1000  # Flag reads slower than 1 second
        
        # GPU memory tracking
        self._peak_gpu_memory_mb = 0
        
        logger.info(
            f"Initialized OnTheFlyBDEDataset with {len(self.keys)} images. "
            f"Generating {patches_per_image} patches per image ({len(self)} total samples)."
        )
    
    def __len__(self) -> int:
        """Return number of samples for this GPU rank (distributed across world_size GPUs)."""
        total_samples = len(self.keys) * self.patches_per_image
        # Each rank gets approximately equal share (last rank may have remainder)
        return (total_samples + self.world_size - 1) // self.world_size
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a random crop from a per-image graded sRGB pair.
        
        Returns:
            (input, target) = (srgb_8u, srgb_32f) with shapes [3, H, W] on GPU, normalized to [0, 1]
        """
        t_start = time.perf_counter()
        is_first_batch = (self._batch_count == 0)
        
        if is_first_batch:
            logger.info(f"[BATCH 0] Starting first batch load (idx={idx})...")
        
        # Lazy initialization of pair_generator to support DDP (where device is set late)
        if self.pair_generator is None:
            # Detect device if not set (fallback for DDP processes)
            if self.device is None:
                try:
                    self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
                except RuntimeError:
                    # Fallback for worker processes without direct CUDA access
                    self.device = torch.device("cuda:0")
                logger.info(f"Dataset detected and using device: {self.device}")
            
            logger.info("Initializing GPU color pipeline (ACESColorTransformer + CDL)...")
            logger.info("  ℹ️  This includes OCIO LUT evaluation (~9 sec one-time cost for 128³ color mapping)")
            init_start = time.perf_counter()
            self.pair_generator = DatasetPairGenerator(self.env, self.device, self.keys, timing_tracker=self._timings)
            init_time_sec = time.perf_counter() - init_start
            logger.info(f"✓ GPU pipeline ready ({init_time_sec:.1f}s). Training will now start normally.")

        # Map local index to global image index (distributed across ranks)
        # When using DistributedSampler, each rank gets different samples
        # We need to map the local idx back to the correct image in LMDB
        global_sample_idx = idx + (self.rank * len(self))
        img_idx = (global_sample_idx // self.patches_per_image) % len(self.keys)
        
        # Load ACES, apply CDL + OCIO once per image (cache to reuse across patches)
        if img_idx != self._last_img_idx:
            # Clear previous cache to free GPU memory before loading next image
            self._cached_srgb_32f = None
            self._cached_srgb_8u = None
            # Don't call torch.cuda.empty_cache() here—it's a synchronous blocking op that causes stalls

            key = self.keys[img_idx]
            try:
                # One random CDL per image
                cdl_params = get_single_random_look()
                
                # Full pipeline: load ACES → apply CDL → transform to sRGB
                # Pass self as reference for LMDB performance tracking
                srgb_32f, srgb_8u = self.pair_generator.load_aces_apply_cdl_and_transform(key, cdl_params, dataset_ref=self)
                
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
        
        # Random crop - IMPORTANT: create copies, not views, to avoid holding full image in memory
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
            # Random crop - CRITICAL FIX: .clone() ensures we don't hold reference to full image
            top = torch.randint(0, H - self.crop_size + 1, (1,), device=self.device).item()
            left = torch.randint(0, W - self.crop_size + 1, (1,), device=self.device).item()
            srgb_32f = srgb_32f[top : top + self.crop_size, left : left + self.crop_size, :].clone()
            srgb_8u = srgb_8u[top : top + self.crop_size, left : left + self.crop_size, :].clone()
        
        # Both are [crop_size, crop_size, 3]
        # Convert to float32, normalize 8-bit to [0, 1], and permute to [3, H, W]
        srgb_8u = (srgb_8u.float() / 255.0).permute(2, 0, 1)  # [0, 255] → [0, 1], then [H, W, 3] → [3, H, W]
        srgb_32f = srgb_32f.float().permute(2, 0, 1)  # [H, W, 3] → [3, H, W]
        
        # Track timing and GPU memory
        t_end = time.perf_counter()
        self._batch_count += 1
        batch_time_ms = (t_end - t_start) * 1000
        self._timings["crop"].append(batch_time_ms)
        
        if is_first_batch:
            logger.info(f"[BATCH 0] ✓ First batch ready in {batch_time_ms:.1f}ms. Training begins now.")
        
        # Log timing stats (don't call expensive cleanup—GPU memory is managed by OS)
        if self._batch_count % self._report_interval == 0 and self._batch_count > 0:
            self._log_timing_stats()
            
            # Also log LMDB read performance trend
            if self._lmdb_read_times:
                recent_reads = self._lmdb_read_times[-20:]  # Last 20 reads
                avg_read_ms = np.mean(recent_reads)
                max_read_ms = np.max(recent_reads)
                slow_count = sum(1 for t in recent_reads if t > self._lmdb_slow_read_threshold_ms)
                
                if slow_count > 0 or max_read_ms > 500:
                    logger.warning(
                        f"[LMDB PERFORMANCE ALERT] Batch {self._batch_count}: "
                        f"Avg read {avg_read_ms:.1f}ms | Max {max_read_ms:.1f}ms | "
                        f"Slow reads (>1s): {slow_count}/20"
                    )
        
        # Return (input, target) = (8-bit input to expand, 32-bit ground truth)
        return srgb_8u, srgb_32f
    
    def _log_timing_stats(self) -> None:
        """Log aggregated timing statistics for bottleneck analysis."""
        import statistics
        
        if not any(self._timings.values()):
            return
        
        # Aggressively trim timing lists to prevent unbounded memory growth
        for key in self._timings.keys():
            if len(self._timings[key]) > self._report_interval:
                self._timings[key] = self._timings[key][-self._report_interval:]
        
        stats_lines = [
            "\n" + "="*80,
            f"BATCH TIMING STATS (last {min(self._report_interval, self._batch_count)} batches, Batch #{self._batch_count}):",
            "-"*80,
        ]
        
        total_ms = 0
        for key, times in self._timings.items():
            if times:
                avg_ms = statistics.mean(times)
                max_ms = max(times)
                min_ms = min(times)
                stats_lines.append(f"  {key:20s}: {avg_ms:7.2f}ms avg | {min_ms:7.2f}ms min | {max_ms:7.2f}ms max")
                total_ms += avg_ms
        
        stats_lines.append("-"*80)
        stats_lines.append(f"  {'TOTAL':20s}: {total_ms:7.2f}ms per batch")
        stats_lines.append("="*80 + "\n")
        
        logger.info("\n".join(stats_lines))
    
    def _log_gpu_memory(self) -> None:
        """Log GPU memory usage for memory leak detection."""
        if not torch.cuda.is_available():
            return
        
        allocated_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        reserved_mb = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        
        if allocated_mb > self._peak_gpu_memory_mb:
            self._peak_gpu_memory_mb = allocated_mb
        
        msg = (
            f"\nGPU MEMORY (Batch {self._batch_count}):\n"
            f"  Allocated: {allocated_mb:.1f} MB\n"
            f"  Reserved:  {reserved_mb:.1f} MB\n"
            f"  Peak:      {self._peak_gpu_memory_mb:.1f} MB\n"
        )
        logger.info(msg)
    
    def cleanup(self) -> None:
        """Release GPU resources and close LMDB."""
        if self.pair_generator:
            self.pair_generator.cleanup()
        if self.env:
            self.env.close()


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
        self.save_hyperparameters(ignore=["model"])
        self.pair_generator = None  # Lazy initialization for WebDataset batches
        self.crop_size = 512  # Default crop size for WebDataset batches
        # Track last batch metrics for progress bar
        self.last_batch_gpu_ms = None
        self.last_batch_loss = None
        self.estimated_total_batches = None  # Set by training script if metadata available
        
        # Cache for on-the-fly patch generation (like OnTheFlyBDEDataset)
        # When using .repeat() in WebDataset, consecutive items are the same image
        # We cache the decoded full image to avoid re-decoding for each patch
        self._last_image_id = None
        self._cached_srgb_8u = None
        self._cached_srgb_32f = None
        
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
        if self.pair_generator is None:
            print(f"[_PROCESS_BATCH] Initializing DatasetPairGenerator on {self.device}...")
            self.pair_generator = DatasetPairGenerator(None, self.device)
        
        exr_bytes_list, metadata_list = batch
        
        # Ensure the device matches the model's current device
        # In Lightning, self.device is automatically updated to the assigned GPU.
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
        
        if is_repeated_image:
            # CACHE HIT: Generate random crops from already-decoded/graded full image
            # This avoids the expensive EXR decode + CDL + color transform pipeline
            logger.debug(f"[PROCESS BATCH] Cache HIT for image '{current_image_id}' - reusing decoded image ({len(exr_bytes_list)} crops)")
            srgb_8u_batch, srgb_32f_batch = self._generate_crops_from_cached_image(
                num_crops=len(exr_bytes_list),
                crop_size=self.crop_size
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
    
    def _generate_crops_from_cached_image(self, num_crops: int, crop_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random crops from cached decoded/graded full image.
        
        This is called when WebDataset.repeat() shows the same image again.
        Avoids expensive EXR decode + CDL + color transform by reusing the cached result.
        """
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
                top = torch.randint(0, H - crop_size + 1, (1,), device=self.device).item()
                left = torch.randint(0, W - crop_size + 1, (1,), device=self.device).item()
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
                    # Try direct assignment first (standard LMDB case)
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

            # Compute mask for well-exposed regions (avoid clipped areas)
            mask = exposure_mask(y)
            loss = masked_l2_loss(y_hat, y, mask)

            self.log("loss_L2/train", loss, prog_bar=False, sync_dist=True)  # Log to Lightning logger
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
                torch.from_numpy(prim_8bit).float().unsqueeze(0).to(self.device)
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
        vis_freq: int = 5,
        crop_size: int = 512,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None # Using Lightning for most logging
        self.log_dir = log_dir
        self.vis_freq = vis_freq
        self.crop_size = crop_size
        
        # Initialize the GPU-accelerated Grading/ACES pipeline
        # This will be used to process raw EXR bytes streamed from WDS
        self.pair_generator = DatasetPairGenerator(None, device, keys=[])

    def _process_batch(self, batch: tuple[list[bytes], list[dict]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw WDS batch (bytes) into graded training pairs (LDR, HDR) on GPU."""
        exr_bytes_list, metadata_list = batch
        
        # Ensure the device matches the model's current device
        # In Lightning, self.device is automatically updated to the assigned GPU.
        if self.pair_generator.device != self.device:
            logger.info(f"Updating DatasetPairGenerator device to match model: {self.device}")
            self.pair_generator.device = self.device
            self.pair_generator.pytorch_transformer.device = self.device
            self.pair_generator.cdl_processor.device = self.device

        # 1. Decode and Grade on GPU
        t_start = time.perf_counter()
        srgb_8u_batch, srgb_32f_batch = self.pair_generator.generate_batch_from_bytes(
            exr_bytes_list, 
            crop_size=self.crop_size
        )
        logger.debug(f"[PROCESS BATCH] Generated {len(exr_bytes_list)} image pairs in {(time.perf_counter()-t_start)*1000:7.1f}ms")
        
        return srgb_8u_batch, srgb_32f_batch

    def train(self):
        """Main training loop placeholder (Lightning handles this via fit)."""
        pass

    def _log_synthetic_visualizations(self) -> None:
        """Log synthetic visualizations (RGB Primaries) to TensorBoard."""

        self.model.eval()
        with torch.no_grad():
            # 1. Prepare Synthetic Data (RGB Primaries)
            prim_hdr = create_primary_gradients(width=512, height=512, dtype="float32")
            prim_8bit = quantize_to_8bit(prim_hdr)

            # 2. Run Inference
            input_tensor = torch.from_numpy(prim_8bit).float().unsqueeze(0).to(self.device)
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

            # 4. Create Comparison Figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"RGB Primary Gradients - Epoch {epoch}", fontsize=16)
            
            # Top Row: Normal
            axes[0, 0].imshow(in_np)
            axes[0, 0].set_title("8-bit Input")
            axes[0, 1].imshow(out_np)
            axes[0, 1].set_title("Model Output")
            axes[0, 2].imshow(gt_np)
            axes[0, 2].set_title("Ground Truth")
            
            # Bottom Row: High Contrast
            axes[1, 0].imshow(in_c)
            axes[1, 0].set_title(f"Input ({contrast_factor}x Contrast)")
            axes[1, 1].imshow(out_c)
            axes[1, 1].set_title(f"Model ({contrast_factor}x Contrast)")
            axes[1, 2].imshow(gt_c)
            axes[1, 2].set_title(f"Target ({contrast_factor}x Contrast)")
            
            for ax in axes.ravel(): 
                ax.axis("off")
            plt.tight_layout()

            self.writer.add_figure("Visuals/PrimaryGradients", fig, global_step=epoch)
            plt.close(fig)

        self.model.train()

    def train(
        self,
        train_dataloader: DataLoader | wds.WebLoader,
        num_epochs: int = 100,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_freq: int = 10,
        run_name: str | None = None,
    ) -> None:
        """Main training loop."""
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
            
            for i, batch in enumerate(train_dataloader):
                data_time += time.time() - batch_start
                
                # If using WebDataset, the batch is (exr_bytes, metadata)
                # We need to process it into (input, target) on GPU
                if isinstance(batch, list) and isinstance(batch[0], list) and isinstance(batch[0][0], bytes):
                    proc_start = time.time()
                    input, target = self._process_batch(batch)
                    data_time += time.time() - proc_start # Counting GPU processing as part of data prep
                else:
                    input, target = batch
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
                
                # Immediate logging for first batches or long intervals
                if i % 10 == 0 or i < 5:
                    logger.info(
                        f"Epoch [{epoch}] Batch [{i}] "
                        f"Loss: {loss.item():.6f} | "
                        f"BatchTime: {time.time()-batch_start:.3f}s | "
                        f"Data: {data_time/(i+1):.3f}s | "
                        f"Fwd: {forward_time/(i+1):.3f}s | Bwd: {backward_time/(i+1):.3f}s"
                    )

                # Log to TensorBoard
                if self.writer:
                    global_step = epoch * len(train_dataloader) + i
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)
                    
                    # Log learning rate
                    for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar("LR", param_group["lr"], global_step)
                
                batch_start = time.time()
                
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
                
                # Run visual validation
                if (epoch + 1) % self.vis_freq == 0:
                    self._log_visualizations(epoch + 1)
                
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
