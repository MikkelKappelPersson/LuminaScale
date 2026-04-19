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
from torch.profiler import profile, record_function, ProfilerActivity
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# from torch.utils.tensorboard import SummaryWriter (handled by Lightning)

console = Console()

from ..data.wds_dataset import LuminaScaleWebDataset
from ..utils.dataset_pair_generator import DatasetPairGenerator
from ..utils.image_generator import create_primary_gradients, quantize_to_8bit, apply_s_curve_contrast_torch
from ..utils.metrics import DeltaEACES

logger = logging.getLogger(__name__)

class DequantizationTrainer(L.LightningModule):
    """LightningModule for training Dequantization-Net."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        vis_freq: int = 5,
        loss_weights: dict | None = None,
        crop_size: int = 512,
        val_crop_size: int | None = None,
        batch_size: int = 1,
        num_workers: int = 2,
        precision: str = "32",
        enable_profiling: bool = False,
        bit_crunch_contrast_min: float = 1.0,
        bit_crunch_contrast_max: float = 1.0,
    ) -> None:
        super().__init__()
        print(f"[DequantizationTrainer] Initializing LightningModule...")
        
        # Disable automatic optimization to use manual_backward() for detailed timing
        self.automatic_optimization = False
        
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
        self.crop_size = crop_size  # Configurable crop size for WebDataset batches (training)
        self.val_crop_size = val_crop_size if val_crop_size is not None else crop_size  # Validation crop size (can differ for speed)
        self.bit_crunch_contrast_min = bit_crunch_contrast_min  # Min bit-crunching factor
        self.bit_crunch_contrast_max = bit_crunch_contrast_max  # Max bit-crunching factor
        self.batch_size = batch_size  # Batch size for throughput calculation (samples/sec)
        self.num_workers = num_workers  # Number of data loading workers
        self.precision = precision  # Mixed precision setting (e.g., "16-mixed", "32")
        self.enable_profiling = enable_profiling  # Enable CUDA synchronization for profiling (slows down training)
        # Track last batch metrics for progress bar
        self.last_batch_gpu_ms = None
        self.estimated_total_batches = None  # Set by training script if metadata available
        self.last_epoch_throughput_samples_per_sec = None  # Store for hparams logging
        
        # Device: always CUDA for training
        self.device_cuda = torch.device("cuda")
        
        # Performance profiling per epoch
        self.epoch_timings = {
            "data_load_ms": [],
            "forward_pass_ms": [],
            "loss_compute_ms": [],
            "backward_pass_ms": [],
            "total_batch_ms": [],
            "gpu_sync_overhead_ms": [],      # Time spent in GPU synchronization
            "gpu_transfer_overhead_ms": [],  # CPU↔GPU memory transfers
            "kernel_launch_overhead_ms": [], # GPU kernel launch overhead
            "other_overhead_ms": [],         # Remaining unaccounted time
        }
        self.epoch_gpu_memory = []  # Peak GPU memory per batch
        self.current_epoch_start_time = None
        
        # GPU kernel profiling (torch.profiler)
        self.gpu_kernel_times: dict[str, list[float]] = {}  # Aggregate kernel times per kernel name
        self.profile_batch_idx = 0  # Profile first N batches to capture representative GPU ops
        
        # Cache for on-the-fly patch generation with WebDataset
        # When using .repeat() in WebDataset, consecutive items are the same image
        # We cache the decoded full image to avoid re-decoding for each patch
        self._last_image_id = None
        self._cached_srgb_8u = None
        self._cached_srgb_32f = None
        
        # Validation metrics (computed on validation set, not training data)
        # NOTE: Quality metrics (PSNR, SSIM, ΔE) are only computed on validation data for unbiased evaluation.
        # Use official torchmetrics classes for better DDP integration and stateful accumulation
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)  # Assumes images normalized to [0, 1]
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # Assumes images normalized to [0, 1]
        self.val_delta_e = DeltaEACES()  # Custom metric for ACES color space ΔE
        self.val_psnr_db = None  # For hparams logging
        self.val_ssim_db = None  # For hparams logging
        self.val_delta_e_mean = None
        
        print(f"[DequantizationTrainer] ✓ Initialization complete")

    def setup(self, stage: str) -> None:
        """Lightning setup hook called before training/validation starts.
        
        Ensures model is in train mode.
        """
        # Force model to be in train mode
        self.model.train()
        logger.debug(f"[setup] Model on device: {next(self.model.parameters()).device}")

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

    def _process_batch(self, batch: tuple[list[bytes], list[dict]], crop_size: int | None = None, is_validation: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Convert raw WDS batch (bytes) into graded training pairs (LDR, HDR) on GPU.
        
        With WebDataset.repeat(patches_per_image), consecutive items are from the same image.
        Implementation: Load full image once, cache it decoded/graded, reuse for all 32 patches
        by generating random crops from the cached version. Avoids 31 redundant EXR decodes.
        
        Args:
            batch: Tuple of (exr_bytes_list, metadata_list)
            crop_size: Optional crop size override. If None, uses self.crop_size
            is_validation: If True, disables contrast augmentation (uses min value without randomization)
        
        Returns: (srgb_8u_batch, srgb_32f_batch, timing_dict)
        timing_dict contains component breakdown for cache misses, or cache_hit_ms for cache hits
        """
        if crop_size is None:
            crop_size = self.crop_size
        t_total_start = time.perf_counter()
        timing = {}
        
        # Lazy initialization of pair_generator
        device = self._get_device()
        
        if self.pair_generator is None:
            t_init = time.perf_counter()
            print(f"[_PROCESS_BATCH] Initializing DatasetPairGenerator on {device}...")
            self.pair_generator = DatasetPairGenerator(device)
            timing["init_pair_generator_ms"] = (time.perf_counter() - t_init) * 1000
        
        exr_bytes_list, metadata_list = batch
        
        # Ensure the device matches the model's current device
        # Update device if it changed (e.g., model moved to different GPU)
        t_device_sync = time.perf_counter()
        current_device = self._get_device()
        if self.pair_generator.device != current_device:
            logger.debug(f"Updating DatasetPairGenerator device to match model: {current_device}")
            self.pair_generator.device = current_device
            self.pair_generator.pytorch_transformer.device = current_device
            self.pair_generator.cdl_processor.device = current_device
        timing["device_sync_ms"] = (time.perf_counter() - t_device_sync) * 1000

        # Extract image IDs from metadata to detect repeated images
        t_extract_meta = time.perf_counter()
        image_ids = [m.get('id', f'img_{i}') for i, m in enumerate(metadata_list)]
        current_image_id = image_ids[0] if image_ids else None
        timing["extract_metadata_ms"] = (time.perf_counter() - t_extract_meta) * 1000
        
        # Check if this is a repeated image (WebDataset.repeat() loops through same images)
        is_repeated_image = (current_image_id == self._last_image_id and self._cached_srgb_8u is not None)
        
        if is_repeated_image:
            # CACHE HIT: Generate random crops from already-decoded/graded full image
            # This avoids the expensive EXR decode + CDL + color transform pipeline
            t_crop_start = time.perf_counter()
            srgb_8u_batch, srgb_32f_batch, crop_timing = self._generate_crops_from_cached_image(
                num_crops=len(exr_bytes_list),
                crop_size=self.crop_size,
                device=current_device
            )
            crop_time_ms = (time.perf_counter() - t_crop_start) * 1000
            logger.debug(f"[PROCESS BATCH] Cache HIT for '{current_image_id}' - generated crops in {crop_time_ms:.1f}ms")
            # Merge timings
            all_timing = {**timing, **crop_timing}
            all_timing["total_process_batch_ms"] = (time.perf_counter() - t_total_start) * 1000
            return srgb_8u_batch, srgb_32f_batch, all_timing
        else:
            # CACHE MISS: Decode full image from EXR bytes, apply CDL, transform to sRGB, then cache
            logger.debug(f"[PROCESS BATCH] Cache MISS - decoding image '{current_image_id}'")
            t_decode_start = time.perf_counter()
            
            # For validation, use consistent bit-crunching (min value, no augmentation)
            # For training, use randomized bit-crunching between min/max
            bit_crunch_min = self.bit_crunch_contrast_min if is_validation else self.bit_crunch_contrast_min
            bit_crunch_max = self.bit_crunch_contrast_min if is_validation else self.bit_crunch_contrast_max
            
            # Decode full image(s) into graded sRGB pairs
            srgb_8u_batch, srgb_32f_batch, batch_timing_breakdown = self.pair_generator.generate_batch_from_bytes(
                exr_bytes_list, 
                crop_size=crop_size,
                bit_crunch_contrast_min=bit_crunch_min,
                bit_crunch_contrast_max=bit_crunch_max,
            )
            total_decode_ms = (time.perf_counter() - t_decode_start) * 1000
            logger.debug(f"[PROCESS BATCH] Decoded and graded {len(exr_bytes_list)} image(s) in {total_decode_ms:.1f}ms")
            
            # Measure caching overhead
            t_cache_start = time.perf_counter()
            # Cache the decoded/graded images for reuse on subsequent patches loops
            if current_image_id and len(exr_bytes_list) == 1:
                self._cached_srgb_8u = srgb_8u_batch
                self._cached_srgb_32f = srgb_32f_batch
                self._last_image_id = current_image_id
                logger.debug(f"[PROCESS BATCH] Cached image '{current_image_id}' for reuse")
            timing["cache_storage_ms"] = (time.perf_counter() - t_cache_start) * 1000
            
            # Merge all timings and add total
            all_timing = {**batch_timing_breakdown, **timing}
            all_timing["total_process_batch_ms"] = (time.perf_counter() - t_total_start) * 1000
            
            # Calculate unaccounted time in process_batch
            component_ms = sum(v for k, v in all_timing.items() 
                              if k not in ["total_process_batch_ms", "unaccounted_process_batch_ms"] and isinstance(v, (int, float)))
            unaccounted_ms = all_timing["total_process_batch_ms"] - component_ms
            if unaccounted_ms > 1.0:
                all_timing["unaccounted_process_batch_ms"] = unaccounted_ms
                logger.warning(f"[PROCESS BATCH] Unaccounted: {unaccounted_ms:.1f}ms of {all_timing['total_process_batch_ms']:.1f}ms")
        
            return srgb_8u_batch, srgb_32f_batch, all_timing
    
    def _generate_crops_from_cached_image(self, num_crops: int, crop_size: int = 512, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Generate random crops from cached decoded/graded full image.
        
        This is called when WebDataset.repeat() shows the same image again.
        Avoids expensive EXR decode + CDL + color transform by reusing the cached result.
        
        Args:
            num_crops: Number of random crops to generate
            crop_size: Size of each crop (default 512)
            device: Device to create random tensor on (defaults to model device)
            
        Returns:
            (srgb_8u_batch, srgb_32f_batch, timing_breakdown_dict)
        """
        t0 = time.perf_counter()
        timing = {}
        
        if device is None:
            device = self._get_device()
        if self._cached_srgb_8u is None or self._cached_srgb_32f is None:
            raise RuntimeError("Cache miss - no cached image available")
        
        # Get cached full image: format is [batch_size, 3, H, W] or [3, H, W]
        cached_8u = self._cached_srgb_8u
        cached_32f = self._cached_srgb_32f
        
        # Extract single full image if batched
        t_extract = time.perf_counter()
        if cached_8u.ndim == 4:
            full_8u = cached_8u[0]  # [3, H, W]
            full_32f = cached_32f[0]
        else:
            full_8u = cached_8u
            full_32f = cached_32f
        
        # Convert from [3, H, W] to [H, W, 3] for cropping
        full_8u = full_8u.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        full_32f = full_32f.permute(1, 2, 0)
        timing["extract_and_permute_ms"] = (time.perf_counter() - t_extract) * 1000
        
        H, W = full_8u.shape[0], full_8u.shape[1]
        
        # Generate num_crops random crops from the full image
        t_crop_gen = time.perf_counter()
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
        
        timing["crop_generation_ms"] = (time.perf_counter() - t_crop_gen) * 1000
        
        # Stack into batch [num_crops, H, W, 3] then convert to [num_crops, 3, H, W]
        t_stack = time.perf_counter()
        srgb_8u_batch = torch.stack(srgb_8u_crops).permute(0, 3, 1, 2)  # [N, H, W, 3] -> [N, 3, H, W]
        srgb_32f_batch = torch.stack(srgb_32f_crops).permute(0, 3, 1, 2)
        timing["stacking_ms"] = (time.perf_counter() - t_stack) * 1000
        
        timing["total_cache_hit_ms"] = (time.perf_counter() - t0) * 1000
        
        return srgb_8u_batch, srgb_32f_batch, timing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        try:
            # Start epoch timing on first batch
            if batch_idx == 0:
                self.current_epoch_start_time = time.perf_counter()
                self.epoch_timings = {
                    "data_load_ms": [], 
                    "forward_pass_ms": [], 
                    "loss_compute_ms": [], 
                    "backward_pass_ms": [], 
                    "optimizer_step_ms": [],
                    "total_batch_ms": [],
                    "gpu_sync_overhead_ms": [],
                    "gpu_transfer_overhead_ms": [],
                    "kernel_launch_overhead_ms": [],
                    "other_overhead_ms": [],
                    "overhead_between_steps_ms": [],  # Time between major ops
                    "framework_overhead_ms": [],      # Lightning/PyTorch overhead
                    "component_timings": [], 
                    "dataloader_fetch_ms": [], 
                    "process_batch_ms": []
                }
                self.epoch_gpu_memory = []
                self.gpu_kernel_times = {}  # Reset GPU kernel profiling data
            
            t_batch_start = time.perf_counter()
            self.last_batch_gpu_ms = None
            batch_timing_breakdown = None  # For component-level timing data
            
            # === DATA LOADING ===
            t_data_start = time.perf_counter()
            
            gpu_transfer_ms = 0.0  # Track CPU→GPU memory transfer time
            
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    if isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                        # Tensors already on device, measure zero-copy time
                        x, y = batch
                    # WebDataset case: batch is (exr_bytes_list, metadata_list)
                    elif isinstance(batch[0], list) and len(batch[0]) > 0 and isinstance(batch[0][0], bytes):
                        t_process_batch_start = time.perf_counter()
                        x, y, batch_timing_breakdown = self._process_batch(batch)
                        process_batch_ms = (time.perf_counter() - t_process_batch_start) * 1000
                        self.epoch_timings["process_batch_ms"].append(process_batch_ms)
                        
                        # Measure GPU transfer overhead: time between tensor creation and GPU availability
                        # This includes PCIe transfer, GPU memory allocation, and synchronization
                        if torch.cuda.is_available() and self.enable_profiling:
                            t_transfer_start = time.perf_counter()
                            torch.cuda.synchronize()  # Wait for GPU memory allocation to complete
                            gpu_transfer_ms = (time.perf_counter() - t_transfer_start) * 1000
                    else:
                        x, y = batch
                else:
                    raise ValueError(f"Batch tuple has {len(batch)} elements, expected 2")
            else:
                raise ValueError(f"Batch type {type(batch)} is not tuple/list")
            
            data_load_ms = (time.perf_counter() - t_data_start) * 1000
            
            # === OVERHEAD: Between Data Load and Forward ===
            t_overhead_start = time.perf_counter()
            
            # === CHECKPOINT 1: Before Forward ===
            t_gpu_sync_before_forward = time.perf_counter()
            if torch.cuda.is_available() and self.enable_profiling:
                torch.cuda.synchronize()  # Ensure GPU is idle before starting forward
            gpu_sync_before_forward_ms = (time.perf_counter() - t_gpu_sync_before_forward) * 1000
            
            overhead_to_forward_ms = (time.perf_counter() - t_overhead_start) * 1000
            
            # === FORWARD PASS ===
            t_forward_start = time.perf_counter()
            
            # Profile GPU kernels on first batch only (if profiling enabled)
            if batch_idx == 0 and torch.cuda.is_available() and self.enable_profiling:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    on_trace_ready=lambda p: None,  # Don't print default output
                ) as prof:
                    with record_function("forward_pass"):
                        y_hat = self.model(x)  # GPU kernel launches (async)
                
                # Extract GPU kernel statistics from profiler
                self._extract_gpu_kernels(prof)
            else:
                y_hat = self.model(x)  # GPU kernel launches (async)
            
            # Don't sync yet - loss computation can overlap with forward on GPU
            forward_ms = (time.perf_counter() - t_forward_start) * 1000
            
            # === LOSS COMPUTATION ===
            t_loss_start = time.perf_counter()
            loss = (l1_loss(y_hat, y) * self.l1_weight + 
                   l2_loss(y_hat, y) * self.l2_weight + 
                   charbonnier_loss(y_hat) * self.charbonnier_weight + 
                   edge_aware_smoothing_loss(y_hat, y) * self.grad_match_weight)
            loss_ms = (time.perf_counter() - t_loss_start) * 1000
            
            # === CHECKPOINT 2: Before Backward (sync to see actual forward+loss time) ===
            t_gpu_sync_before_backward = time.perf_counter()
            if torch.cuda.is_available() and self.enable_profiling:
                torch.cuda.synchronize()  # Wait for forward+loss GPU ops to complete
            gpu_sync_before_backward_ms = (time.perf_counter() - t_gpu_sync_before_backward) * 1000
            
            # === BACKWARD PASS ===
            t_backward_start = time.perf_counter()
            self.manual_backward(loss)  # GPU kernel launches (async)
            backward_ms = (time.perf_counter() - t_backward_start) * 1000
            
            # === OVERHEAD: Between Backward and Optimizer ===
            t_overhead_optim_start = time.perf_counter()
            
            # === OPTIMIZER STEP ===
            t_optim_start = time.perf_counter()
            self.optimizers().step()
            self.optimizers().zero_grad()
            optimizer_step_ms = (time.perf_counter() - t_optim_start) * 1000
            
            overhead_to_optim_ms = (time.perf_counter() - t_overhead_optim_start) * 1000
            
            # === CHECKPOINT 3: After Optimizer (sync to ensure all GPU ops complete) ===
            t_gpu_sync_after_backward = time.perf_counter()
            if torch.cuda.is_available() and self.enable_profiling:
                torch.cuda.synchronize()  # Ensure all GPU operations complete
            gpu_sync_after_backward_ms = (time.perf_counter() - t_gpu_sync_after_backward) * 1000
            
            total_batch_ms = (time.perf_counter() - t_batch_start) * 1000
            
            # Calculate total overhead between major operations
            overhead_between_steps_ms = overhead_to_forward_ms + overhead_to_optim_ms
            
            # Calculate overhead breakdown
            # Total GPU sync = waiting time at 3 checkpoints
            total_gpu_sync_ms = gpu_sync_before_forward_ms + gpu_sync_before_backward_ms + gpu_sync_after_backward_ms
            
            # Kernel launch overhead = implicit sync + async kernel launches (in forward+loss+backward)
            # This is the difference between wall-clock time and actual GPU work time
            measured_compute_ms = forward_ms + loss_ms + backward_ms + optimizer_step_ms
            
            # Framework overhead = PyTorch/Lightning overhead + tensor allocation + other hidden costs
            # This is any remaining time not accounted for by operations or GPU sync
            framework_overhead_ms = max(0, total_batch_ms - 
                                       (data_load_ms + measured_compute_ms + total_gpu_sync_ms + 
                                        gpu_transfer_ms + overhead_between_steps_ms))
            
            # Track metrics per batch
            self.epoch_timings["data_load_ms"].append(data_load_ms)
            self.epoch_timings["forward_pass_ms"].append(forward_ms)
            self.epoch_timings["loss_compute_ms"].append(loss_ms)
            self.epoch_timings["backward_pass_ms"].append(backward_ms)
            self.epoch_timings["optimizer_step_ms"].append(optimizer_step_ms)
            self.epoch_timings["total_batch_ms"].append(total_batch_ms)
            self.epoch_timings["gpu_sync_overhead_ms"].append(total_gpu_sync_ms)
            self.epoch_timings["gpu_transfer_overhead_ms"].append(gpu_transfer_ms)
            self.epoch_timings["overhead_between_steps_ms"].append(overhead_between_steps_ms)
            self.epoch_timings["framework_overhead_ms"].append(framework_overhead_ms)
            self.epoch_timings["kernel_launch_overhead_ms"].append(0.0)  # Not directly measurable
            self.epoch_timings["other_overhead_ms"].append(max(0, total_batch_ms - 
                                       (data_load_ms + measured_compute_ms + total_gpu_sync_ms + gpu_transfer_ms)))
            
            # Store component timing breakdown if available
            if batch_timing_breakdown is not None:
                self.epoch_timings["component_timings"].append(batch_timing_breakdown)
            
            # GPU memory tracking
            if torch.cuda.is_available():
                self.epoch_gpu_memory.append(torch.cuda.memory_allocated() / 1e9)  # GB
            
            # Metrics are logged to TensorBoard via self.log() below (no .item() calls needed)

            self.log("loss_L1/train", l1_loss(y_hat, y), prog_bar=False, sync_dist=True)
            self.log("loss_L2/train", l2_loss(y_hat, y), prog_bar=False, sync_dist=True)
            self.log("loss_Charbonnier/train", charbonnier_loss(y_hat), prog_bar=False, sync_dist=True)
            self.log("loss_EdgeAware/train", edge_aware_smoothing_loss(y_hat, y), prog_bar=False, sync_dist=True)
            self.log("loss_total/train", loss, prog_bar=False, sync_dist=True)
            # Log current learning rate (supports dynamic LR scheduling)
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", current_lr, prog_bar=False, sync_dist=True)
            
            self.last_batch_gpu_ms = total_batch_ms
            return loss
            
        except Exception as e:
            logger.error(f"[Batch {batch_idx:3d}] Exception: {type(e).__name__}: {e}")
            print(f"[TRAINING_STEP] Exception message: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            raise

    def _extract_gpu_kernels(self, prof) -> None:
        """Extract and aggregate GPU kernel execution times from torch.profiler.
        
        Filters for CUDA operations and stores kernel times for later display.
        """
        try:
            key_avgs = prof.key_averages()
            # Filter for CUDA operations (GPU kernels)
            for evt in key_avgs:
                if evt.key not in ["forward_pass", "loss_computation", "backward_pass"]:
                    # Only capture actual GPU kernels
                    if "cuda" in evt.key.lower() or "cudnn" in evt.key.lower():
                        cuda_time_ms = evt.cuda_time_total / 1000  # Convert to ms
                        if cuda_time_ms > 0.1:  # Only track significant kernels (>0.1ms)
                            if evt.key not in self.gpu_kernel_times:
                                self.gpu_kernel_times[evt.key] = []
                            self.gpu_kernel_times[evt.key].append(cuda_time_ms)
            
            logger.debug(f"✓ Extracted GPU kernels: {len(self.gpu_kernel_times)} unique kernels")
        except Exception as e:
            logger.debug(f"Warning: Failed to extract GPU kernels: {e}")

    def on_train_epoch_end(self) -> None:
        """Print epoch performance summary with timing breakdown using rich."""
        if not self.epoch_timings["total_batch_ms"]:
            return  # No batches processed
        
        # Calculate statistics
        data_load_avg = np.mean(self.epoch_timings["data_load_ms"])
        forward_avg = np.mean(self.epoch_timings["forward_pass_ms"])
        loss_avg = np.mean(self.epoch_timings["loss_compute_ms"])
        backward_avg = np.mean(self.epoch_timings["backward_pass_ms"])
        optimizer_step_avg = np.mean(self.epoch_timings["optimizer_step_ms"]) if self.epoch_timings["optimizer_step_ms"] else 0
        total_avg = np.mean(self.epoch_timings["total_batch_ms"])
        total_epoch = (time.perf_counter() - self.current_epoch_start_time) if self.current_epoch_start_time else 0
        
        gpu_memory_max = max(self.epoch_gpu_memory) if self.epoch_gpu_memory else 0
        gpu_memory_avg = np.mean(self.epoch_gpu_memory) if self.epoch_gpu_memory else 0
        
        num_batches = len(self.epoch_timings["total_batch_ms"])
        # Calculate both batches/sec and samples/sec for comparison
        throughput_batches = num_batches / total_epoch if total_epoch > 0 else 0
        throughput_samples = (num_batches * self.batch_size) / total_epoch if total_epoch > 0 else 0
        
        # Store for hparams logging
        self.last_epoch_throughput_samples_per_sec = throughput_samples
        
        # Get overhead averages for accounting table
        gpu_sync_avg = np.mean(self.epoch_timings["gpu_sync_overhead_ms"]) if self.epoch_timings["gpu_sync_overhead_ms"] else 0
        gpu_transfer_avg = np.mean(self.epoch_timings["gpu_transfer_overhead_ms"]) if self.epoch_timings["gpu_transfer_overhead_ms"] else 0
        framework_overhead_avg = np.mean(self.epoch_timings["framework_overhead_ms"]) if self.epoch_timings["framework_overhead_ms"] else 0
        
        # Only print accounting table if profiling is enabled
        if self.enable_profiling:
            # Create consolidated time accounting table
            accounting_table = Table(
                title=f"[bold cyan]Epoch {self.current_epoch} - Batch Time Breakdown[/bold cyan]",
                show_header=True,
                header_style="bold magenta",
                border_style="cyan",
                padding=(0, 1),
            )
            accounting_table.add_column("Operation", style="cyan", no_wrap=True)
            accounting_table.add_column("Time (ms)", justify="right", style="green")
            accounting_table.add_column("% of Total", justify="right", style="yellow")
            accounting_table.add_column("Source", style="dim", no_wrap=True)
            
            # Primary operations with explanations
            accounting_table.add_row(
                "Data Loading",
                f"{data_load_avg:.2f}",
                f"{(data_load_avg/total_avg)*100:.1f}%",
                "EXR decode, CDL, ACES, quantization, GPU transfer"
            )
            
            accounting_table.add_row(
                "Forward Pass",
                f"{forward_avg:.2f}",
                f"{(forward_avg/total_avg)*100:.1f}%",
                "Model inference on GPU"
            )
            
            accounting_table.add_row(
                "Loss Computation",
                f"{loss_avg:.2f}",
                f"{(loss_avg/total_avg)*100:.1f}%",
                "L1 + L2 + Charbonnier + EdgeAware on GPU"
            )
            
            accounting_table.add_row(
                "Backward Pass",
                f"{backward_avg:.2f}",
                f"{(backward_avg/total_avg)*100:.1f}%",
                "Gradient computation on GPU"
            )
            
            accounting_table.add_row(
                "Optimizer Step",
                f"{optimizer_step_avg:.2f}",
                f"{(optimizer_step_avg/total_avg)*100:.1f}%",
                "Adam weight update on GPU"
            )
            
            # Overhead components
            if gpu_sync_avg > 0.1:
                accounting_table.add_row(
                    "GPU Synchronization",
                    f"{gpu_sync_avg:.2f}",
                    f"{(gpu_sync_avg/total_avg)*100:.1f}%",
                    "torch.cuda.synchronize() calls"
                )
            
            if gpu_transfer_avg > 0.1:
                accounting_table.add_row(
                    "GPU Memory Transfer",
                    f"{gpu_transfer_avg:.2f}",
                    f"{(gpu_transfer_avg/total_avg)*100:.1f}%",
                    "PCIe bandwidth + GPU allocation"
                )
            
            if framework_overhead_avg > 0.1:
                accounting_table.add_row(
                    "Framework Overhead",
                    f"{framework_overhead_avg:.2f}",
                    f"{(framework_overhead_avg/total_avg)*100:.1f}%",
                    "PyTorch/Lightning hidden costs"
                )
            
            # Calculate total accounted for (without double-counting overhead_between_steps)
            total_accounted = (data_load_avg + forward_avg + loss_avg + backward_avg + optimizer_step_avg + 
                              gpu_sync_avg + gpu_transfer_avg + framework_overhead_avg)
            unaccounted_ms = total_avg - total_accounted
            
            accounting_table.add_row(
                "[bold]Total Accounted[/bold]",
                f"[bold green]{total_accounted:.2f}[/bold green]",
                f"[bold yellow]{(total_accounted/total_avg)*100:.1f}%[/bold yellow]",
                ""
            )
            
            if unaccounted_ms > 0.1:
                accounting_table.add_row(
                    "[bold red]UNACCOUNTED[/bold red]",
                    f"[bold red]{unaccounted_ms:.2f}[/bold red]",
                    f"[bold red]{(unaccounted_ms/total_avg)*100:.1f}%[/bold red]",
                    "[bold red]MISSING - needs investigation[/bold red]"
                )
            
            accounting_table.add_row(
                "[bold cyan]Total Batch[/bold cyan]",
                f"[bold cyan]{total_avg:.2f}[/bold cyan]",
                "[bold cyan]100.0%[/bold cyan]",
                ""
            )
            
            console.print(accounting_table)
        
        # Create GPU kernel profiling table (if profiling was enabled)
        if self.gpu_kernel_times and len(self.gpu_kernel_times) > 0:
            # Sort kernels by total time and get top 10
            kernel_totals = {}
            for kernel_name, times in self.gpu_kernel_times.items():
                kernel_totals[kernel_name] = sum(times)
            
            top_kernels = sorted(kernel_totals.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_kernels and top_kernels[0][1] > 0.1:  # Only show if significant
                gpu_kernel_table = Table(
                    title="[bold cyan]Top GPU Kernels (GPU 🎯)[/bold cyan]",
                    show_header=True,
                    header_style="bold magenta",
                    border_style="cyan",
                    padding=(0, 1),
                )
                gpu_kernel_table.add_column("GPU Kernel / Operation", style="cyan", no_wrap=False)
                gpu_kernel_table.add_column("Time (ms)", justify="right", style="green")
                gpu_kernel_table.add_column("% of Batch", justify="right", style="yellow")
                
                total_kernel_time = sum(tc for _, tc in top_kernels)
                for kernel_name, kernel_time_ms in top_kernels:
                    # Clean up kernel name for display
                    display_name = kernel_name.replace("cudnn::", "").replace("cuda::", "").strip()
                    kernel_pct = (kernel_time_ms / total_avg) * 100
                    gpu_kernel_table.add_row(
                        display_name[:60],  # Truncate long kernel names
                        f"{kernel_time_ms:.2f}",
                        f"{kernel_pct:.1f}%"
                    )
                
                total_kernel_pct = (total_kernel_time / total_avg) * 100
                gpu_kernel_table.add_row(
                    "[bold]Total (Top Kernels)[/bold]",
                    f"[bold green]{total_kernel_time:.2f}[/bold green]",
                    f"[bold yellow]{total_kernel_pct:.1f}%[/bold yellow]"
                )
                
                console.print(gpu_kernel_table)
        
        # Create dataset metrics table (always shown - no GPU overhead)
        dataset_table = Table(
            title="[bold cyan]Dataset Metrics[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            padding=(0, 1),
        )
        dataset_table.add_column("Metric", style="cyan", no_wrap=True)
        dataset_table.add_column("Value", justify="right", style="green")
        
        dataset_table.add_row("Total Epoch Time", f"{total_epoch:.2f}s")
        dataset_table.add_row("Batches Processed", str(num_batches))
        dataset_table.add_row("Throughput (batches/s)", f"{throughput_batches:.2f}")
        dataset_table.add_row("Throughput (samples/s)", f"{throughput_samples:.2f}")
        dataset_table.add_row("GPU Memory (Current)", f"{gpu_memory_avg:.2f} GB")
        dataset_table.add_row("GPU Memory (Peak)", f"{gpu_memory_max:.2f} GB")
        
        # Add performance hyperparameters
        dataset_table.add_row("[bold]─ Configuration ─[/bold]", "")
        dataset_table.add_row("Batch Size", str(self.batch_size))
        dataset_table.add_row("Num Workers", str(self.num_workers))
        dataset_table.add_row("Crop Size", f"{self.crop_size}×{self.crop_size}")
        dataset_table.add_row("Precision", self.precision)
        
        console.print(dataset_table)

    def validation_step(self, batch: dict | tuple | list, batch_idx: int) -> dict:
        """Validation: forward pass + compute all quality metrics.
        
        Computes PSNR, SSIM, ΔE on validation data only (not training data)
        to prevent overfitting and provide unbiased quality assessment.
        
        Quality metrics logged here are per-batch and automatically aggregated
        by Lightning across all validation batches at epoch end.
        
        Handles both WebDataset format (tuple/list) and dictionary format.
        Uses val_crop_size which can differ from training crop_size for speed.
        """
        # Handle WebDataset format: batch is (exr_bytes_list, metadata_list)
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                # WebDataset case: decode batch
                if isinstance(batch[0], list) and len(batch[0]) > 0 and isinstance(batch[0][0], bytes):
                    # Use val_crop_size for validation crops (can be smaller for speed)
                    # Use is_validation=True to disable contrast augmentation
                    x, y, _ = self._process_batch(batch, crop_size=self.val_crop_size, is_validation=True)
                else:
                    # Standard tuple/list format: (x, y)
                    x, y = batch[0], batch[1]
            else:
                raise ValueError(f"Batch tuple has {len(batch)} elements, expected 2")
        elif isinstance(batch, dict):
            # Dictionary format: {"noisy": x, "reference": y}
            x = batch["noisy"]
            y = batch["reference"]
        else:
            raise ValueError(f"Batch type {type(batch)} is not tuple/list/dict")
        
        y_hat = self.model(x)
        
        # Compute loss (same weighting as training)
        loss = (l1_loss(y_hat, y) * self.l1_weight + 
               l2_loss(y_hat, y) * self.l2_weight + 
               charbonnier_loss(y_hat) * self.charbonnier_weight + 
               edge_aware_smoothing_loss(y_hat, y) * self.grad_match_weight)
        
        # Update stateful metrics (torchmetrics handles aggregation + DDP sync internally)
        # PSNR and SSIM use official torchmetrics implementations
        self.val_psnr.update(y_hat, y)
        self.val_ssim.update(y_hat, y)
        
        # ΔE (Delta-E ACES) uses custom metric with proper DDP support
        self.val_delta_e.update(y_hat, y)
        
        # Log loss per batch (aggregated by Lightning)
        batch_size = x.size(0) if hasattr(x, 'size') else 1
        self.log("metric_loss/val", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        # PSNR, SSIM, ΔE will be logged in on_validation_epoch_end() after compute()
        
        return {"val_loss": loss}

    def on_validation_epoch_end(self) -> None:
        """Compute epoch-level validation metrics and store for hparams logging.
        
        Calls .compute() on stateful metrics (PSNR, SSIM, ΔE from torchmetrics) and logs them,
        then resets metrics for the next epoch.
        
        These metrics measure model quality on unseen data and are suitable for
        hyperparameter comparison, early stopping, and best-model selection.
        """
        # Compute epoch-level metrics from accumulated batches
        psnr_epoch = self.val_psnr.compute()
        ssim_epoch = self.val_ssim.compute()
        delta_e_epoch = self.val_delta_e.compute()
        
        # Log epoch-level metrics
        self.log("metric_psnr/val", psnr_epoch, on_epoch=True, sync_dist=True)
        self.log("metric_ssim/val", ssim_epoch, on_epoch=True, sync_dist=True)
        self.log("metric_delta_e/val", delta_e_epoch, on_epoch=True, sync_dist=True)
        
        # Reset metrics for next epoch
        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_delta_e.reset()
        
        # Store for hparams logging
        self.val_psnr_db = psnr_epoch.item() if hasattr(psnr_epoch, "item") else float(psnr_epoch)
        self.val_ssim_db = ssim_epoch.item() if hasattr(ssim_epoch, "item") else float(ssim_epoch)
        self.val_delta_e_mean = delta_e_epoch.item() if hasattr(delta_e_epoch, "item") else float(delta_e_epoch)

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