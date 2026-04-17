"""GPU-Accelerated CDL (Color Decision List) Processor.

Implements PyTorch/CUDA-based CDL color grading for on-the-fly dataset generation.
Follows the "Lean I/O, Heavy Compute" architecture from the data pipeline spec.

CDL Formula:
    Output = (Input × Slope + Offset) ^ Power
    Then apply saturation as luma-weighted blend.

This module is designed for batch processing during training without CPU bottlenecks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple
import logging

import numpy as np
import torch
import OpenImageIO as oiio

from .look_generator import CDLParameters

logger = logging.getLogger(__name__)

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUCDLProcessor:
    """Batch CDL color grading processor on GPU.
    
    Processes ACES images through CDL transformations using PyTorch CUDA kernels.
    Designed for integration into dataloader pipelines with minimal CPU overhead.
    """

    def __init__(self, device: torch.device | None = None, enable_cache_clearing: bool = False) -> None:
        """Initialize processor.
        
        Args:
            device: torch device (defaults to CUDA if available).
            enable_cache_clearing: Enable torch.cuda.empty_cache() in cleanup() (default: False).
                Set to False during training (GPU cache clearing forces synchronization, blocking pipeline).
                Only enable for post-processing or cleanup after long-running inference sessions.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache_clearing = enable_cache_clearing
        logger.debug(f"GPU CDL Processor initialized on {self.device}, cache_clearing={'enabled' if enable_cache_clearing else 'disabled'}")

    def apply_cdl_gpu(
        self,
        image: torch.Tensor,
        cdl_params: CDLParameters,
    ) -> torch.Tensor:
        """Apply CDL (Color Decision List) transform on GPU using PyTorch.

        Args:
            image: Input tensor [H, W, 3] in linear ACES color space.
            cdl_params: CDL color parameters (slope, offset, power, saturation).

        Returns:
            Graded image tensor [H, W, 3] on GPU.
        """
        # Ensure float32 and move to GPU if not already there
        if image.device.type != "cuda":
            image_gpu = image.to(self.device, dtype=torch.float32)
        else:
            image_gpu = image.to(dtype=torch.float32)

        # Reshape params for broadcasting
        slope_t = torch.tensor(cdl_params.slope, dtype=torch.float32, device=self.device).view(
            1, 1, 3
        )
        offset_t = torch.tensor(cdl_params.offset, dtype=torch.float32, device=self.device).view(
            1, 1, 3
        )
        power_t = torch.tensor(cdl_params.power, dtype=torch.float32, device=self.device).view(
            1, 1, 3
        )

        # CDL: (Input × Slope + Offset) ^ Power
        # Note: Input × Slope uses a lot of memory for big images.
        graded = image_gpu * slope_t
        graded.add_(offset_t)
        
        graded.clamp_(min=1e-6)  # In-place clamp
        graded.pow_(power_t)     # In-place power

        # Apply saturation (luma-weighted blend)
        if cdl_params.saturation != 1.0:
            # Rec.709 luma coefficients
            luma_coeff = torch.tensor(
                [0.2126, 0.7152, 0.0722], dtype=torch.float32, device=self.device
            ).view(1, 1, 3)
            luma = torch.sum(graded * luma_coeff, dim=2, keepdim=True)
            
            # graded = luma + saturation * (graded - luma)
            # Use in-place ops for memory efficiency
            graded.sub_(luma)
            graded.mul_(cdl_params.saturation)
            graded.add_(luma)

        return graded

    def batch_cdl_gpu(
        self,
        images_batch: torch.Tensor,
        cdl_params: CDLParameters,
    ) -> Tuple[torch.Tensor, dict]:
        """Process batch of CDL operations on GPU (highly parallelizable).

        Args:
            images_batch: Batch tensor [B, H, W, 3] on GPU or CPU.
            cdl_params: CDL parameters (applied to entire batch).

        Returns:
            (graded_batch: [B, H, W, 3] on GPU, timing: dict).
        """
        timing = {}
        t0 = time.time()

        # Move batch to GPU
        images_gpu = images_batch.to(self.device, dtype=torch.float32)
        timing["gpu_transfer"] = (time.time() - t0) * 1000

        # Apply CDL to entire batch at once
        t0 = time.time()
        slope_t = torch.tensor(
            cdl_params.slope, dtype=torch.float32, device=self.device
        ).view(1, 1, 1, 3)
        offset_t = torch.tensor(
            cdl_params.offset, dtype=torch.float32, device=self.device
        ).view(1, 1, 1, 3)
        power_t = torch.tensor(
            cdl_params.power, dtype=torch.float32, device=self.device
        ).view(1, 1, 1, 3)

        graded = images_gpu * slope_t + offset_t
        graded = torch.clamp(graded, min=1e-6)
        graded = torch.pow(graded, power_t)

        if cdl_params.saturation != 1.0:
            luma_coeff = torch.tensor(
                [0.2126, 0.7152, 0.0722], dtype=torch.float32, device=self.device
            ).view(1, 1, 1, 3)
            luma = torch.sum(graded * luma_coeff, dim=3, keepdim=True)
            graded = luma + cdl_params.saturation * (graded - luma)

        timing["cdl_gpu"] = (time.time() - t0) * 1000

        return graded, timing

    def process_single_image(
        self,
        aces_image_path: Path | str,
        cdl_params: CDLParameters,
    ) -> Tuple[torch.Tensor, dict]:
        """Process single ACES image with CDL on GPU.

        Args:
            aces_image_path: Path to ACES EXR input image.
            cdl_params: CDL parameters for grading.

        Returns:
            (graded_tensor: [H, W, 3] float32 on GPU, timing: dict).
        """
        timing = {}

        # Load ACES image using OIIO (CPU)
        t0 = time.time()
        buf = oiio.ImageBuf(str(aces_image_path))
        image_np = np.asarray(buf.get_pixels(), dtype=np.float32)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        timing["load_oiio"] = (time.time() - t0) * 1000

        # Convert to PyTorch and move to GPU
        t0 = time.time()
        image_torch = torch.from_numpy(image_np)
        graded = self.apply_cdl_gpu(image_torch, cdl_params)
        timing["cdl_total"] = (time.time() - t0) * 1000

        return graded, timing

    def process_batch_from_files(
        self,
        image_paths: list[Path | str],
        cdl_params: CDLParameters,
    ) -> Tuple[torch.Tensor, dict]:
        """Process batch of ACES images from files.

        Args:
            image_paths: List of ACES EXR file paths.
            cdl_params: CDL parameters (applied to all images).

        Returns:
            (graded_batch: [B, H, W, 3] on GPU, timing: dict).
        """
        timing = {"load_files": 0.0, "batch_cdl": 0.0}

        # Load all images
        t0 = time.time()
        images_list = []
        for img_path in image_paths:
            buf = oiio.ImageBuf(str(img_path))
            img_np = np.asarray(buf.get_pixels(), dtype=np.float32)
            if img_np.shape[2] > 3:
                img_np = img_np[:, :, :3]
            images_list.append(torch.from_numpy(img_np))

        timing["load_files"] = (time.time() - t0) * 1000

        # Stack into batch
        t0 = time.time()
        batch = torch.stack(images_list, dim=0)  # [B, H, W, 3]
        graded_batch, batch_timing = self.batch_cdl_gpu(batch, cdl_params)
        timing.update(batch_timing)
        timing["batch_stack"] = (time.time() - t0) * 1000

        return graded_batch, timing

    def cleanup(self) -> None:
        """Clear GPU cache (optional, disabled by default to avoid pipeline blocking).
        
        Only clears cache if enable_cache_clearing=True was set during initialization.
        Cache clearing forces GPU-CPU synchronization which blocks the training pipeline.
        """
        if self.device.type == "cuda" and self.enable_cache_clearing:
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
        elif self.device.type == "cuda" and not self.enable_cache_clearing:
            logger.debug("GPU cache clearing disabled (enable_cache_clearing=False)")
