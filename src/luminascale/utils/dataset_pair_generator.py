"""Dataset utilities for on-the-fly image generation.

Provides utilities for GPU-accelerated ACES loading and transformation.
Note: CDL grading and patching are handled by Dataset classes (e.g., OnTheFlyBDEDataset).
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import lmdb
import numpy as np
import torch

from .gpu_cdl_processor import GPUCDLProcessor

logger = logging.getLogger(__name__)


class DatasetPairGenerator:
    """Utilities for loading ACES and applying GPU transforms.
    
    This class handles the full ACES→sRGB pipeline:
    - Loading ACES from LMDB
    - Applying CDL grading (optional)
    - GPU-accelerated color transforms (OCIO)
    
    CDL grading and dataset iteration are handled by Dataset classes that use these utilities.
    """

    def __init__(
        self,
        lmdb_env: lmdb.Environment,
        device: torch.device,
        keys_cache: list[str] | None = None,
        timing_tracker: dict | None = None,
    ) -> None:
        """Initialize ACES loader and GPU transform utilities.
        
        Args:
            lmdb_env: Open LMDB environment in read-only mode.
            device: GPU device (e.g., torch.device("cuda")).
            keys_cache: Pre-loaded keys list (optional, auto-loads if None).
            timing_tracker: Optional dict to track operation timings for profiling.
        """
        self.env = lmdb_env
        self.device = device
        self.timing_tracker = timing_tracker or {}
        self._image_load_count = 0  # Track for occasional detailed logging
        
        # PyTorch ACES transformer (primary, with LUT for accuracy)
        from .pytorch_aces_transformer import ACESColorTransformer
        self.pytorch_transformer = ACESColorTransformer(device=device, use_lut=True)
        
        self.cdl_processor = GPUCDLProcessor(device=device)
        self.keys_cache = keys_cache or self._load_keys()

    def _load_keys(self) -> list[str]:
        """Load image keys from LMDB."""
        with self.env.begin(write=False) as txn:
            keys_buf = txn.get(b"__keys__")
            if keys_buf is None:
                raise ValueError("LMDB missing __keys__ - invalid format")
            return pickle.loads(keys_buf)

    def _load_aces_from_lmdb(self, key: str) -> torch.Tensor:
        """Load single ACES image from LMDB and return as [H, W, 3] GPU tensor."""
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode("ascii"))
            if buf is None:
                raise KeyError(f"Key not found: {key}")

        # Parse header: 3 × int32 = 12 bytes
        header = np.frombuffer(buf[:12], dtype=np.uint32)
        H, W, C = int(header[0]), int(header[1]), int(header[2])

        # Extract ACES data: float32 array (make writable copy)
        hdr_size = H * W * C * 4
        hdr_np = np.frombuffer(buf[12 : 12 + hdr_size], dtype=np.float32).reshape(C, H, W).copy()

        # Convert to CPU tensor [H, W, 3] first, then move to GPU only when needed
        aces_tensor = torch.from_numpy(hdr_np).permute(1, 2, 0)  # [C, H, W] → [H, W, 3]
        return aces_tensor

    def load_aces_and_transform(
        self, key: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load ACES from LMDB and return both 32-bit and 8-bit sRGB transforms.
        
        Args:
            key: LMDB key for the image.
        
        Returns:
            Tuple of (srgb_32f, srgb_8u) both [H, W, 3] on GPU.
        
        Raises:
            KeyError: If key not found in LMDB.
        """
        aces_tensor = self._load_aces_from_lmdb(key).to(self.device, non_blocking=True)
        srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(aces_tensor)
        srgb_8u = self.pytorch_transformer.aces_to_srgb_8u(aces_tensor)
        return srgb_32f, srgb_8u

    def load_aces_apply_cdl_and_transform(
        self, key: str, cdl_params: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load ACES, apply CDL grading, and transform to sRGB (full pipeline).
        
        This consolidates the entire color pipeline in one method:
        1. Load ACES from LMDB
        2. Apply CDL grading on GPU
        3. Transform to sRGB via PyTorch
        
        Args:
            key: LMDB key for the image.
            cdl_params: CDL parameters dict from look_generator (e.g., {'slope': [...], 'offset': [...], ...}).
        
        Returns:
            Tuple of (srgb_32f, srgb_8u) both [H, W, 3] on GPU, normalized to [0, 1].
        
        Raises:
            KeyError: If key not found in LMDB.
        """
        import time
        
        t0 = time.perf_counter()
        logger.debug(f"[BATCH TIMING] Starting image load for key: {key}")
        
        # Load ACES to CPU
        t_lmdb_start = time.perf_counter()
        aces_tensor_cpu = self._load_aces_from_lmdb(key)
        t_lmdb = time.perf_counter() - t_lmdb_start
        logger.debug(f"[BATCH TIMING]   LMDB load: {t_lmdb*1000:.1f}ms, shape={aces_tensor_cpu.shape}")
        
        # Move to GPU for CDL
        t_transfer_start = time.perf_counter()
        aces_tensor = aces_tensor_cpu.to(self.device, non_blocking=True)
        # Ensure GPU transfer is complete
        torch.cuda.synchronize()
        t_transfer = time.perf_counter() - t_transfer_start
        logger.debug(f"[BATCH TIMING]   GPU transfer: {t_transfer*1000:.1f}ms")
        
        # Apply CDL
        t_cdl_start = time.perf_counter()
        graded_aces = self.cdl_processor.apply_cdl_gpu(aces_tensor, cdl_params)
        torch.cuda.synchronize()
        t_cdl = time.perf_counter() - t_cdl_start
        logger.debug(f"[BATCH TIMING]   CDL apply: {t_cdl*1000:.1f}ms")
        
        # Explicitly delete intermediate ACES tensor to free memory early
        del aces_tensor
        
        # Transform to sRGB
        t_aces_start = time.perf_counter()
        srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(graded_aces)
        srgb_8u = self.pytorch_transformer.aces_to_srgb_8u(graded_aces)
        torch.cuda.synchronize()
        t_aces = time.perf_counter() - t_aces_start
        logger.debug(f"[BATCH TIMING]   ACES transform: {t_aces*1000:.1f}ms")
        
        # Free graded ACES
        del graded_aces
        
        total_time = time.perf_counter() - t0
        self._image_load_count += 1
        
        # Only log detailed timing every 50 images to avoid spam
        if self._image_load_count % 50 == 0:
            logger.info(
                f"\n{'='*80}\n"
                f"[IMAGE LOAD #{self._image_load_count}] ✓ Completed in {total_time*1000:.1f}ms\n"
                f"  LMDB load     : {t_lmdb*1000:.1f}ms\n"
                f"  GPU transfer  : {t_transfer*1000:.1f}ms\n"
                f"  CDL apply     : {t_cdl*1000:.1f}ms\n"
                f"  ACES transform: {t_aces*1000:.1f}ms\n"
                f"{'='*80}\n"
            )
        
        # Track timings if tracker provided
        if self.timing_tracker is not None:
            self.timing_tracker["lmdb_load"].append(t_lmdb * 1000)
            self.timing_tracker["gpu_transfer"].append(t_transfer * 1000)
            self.timing_tracker["cdl"].append(t_cdl * 1000)
            self.timing_tracker["aces_transform"].append(t_aces * 1000)
        
        return srgb_32f, srgb_8u

    def cleanup(self) -> None:
        """Release GPU resources."""
        self.cdl_processor.cleanup()



