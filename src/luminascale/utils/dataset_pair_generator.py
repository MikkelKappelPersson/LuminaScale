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
from .gpu_torch_processor import GPUTorchProcessor

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
    ) -> None:
        """Initialize ACES loader and GPU transform utilities.
        
        Args:
            lmdb_env: Open LMDB environment in read-only mode.
            device: GPU device (e.g., torch.device("cuda")).
            keys_cache: Pre-loaded keys list (optional, auto-loads if None).
        """
        self.env = lmdb_env
        self.device = device
        self.ocio_processor = GPUTorchProcessor(headless=True)
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
        srgb_32f, srgb_8u = self.ocio_processor.apply_ocio_torch(aces_tensor)
        return srgb_32f, srgb_8u

    def load_aces_apply_cdl_and_transform(
        self, key: str, cdl_params: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load ACES, apply CDL grading, and transform to sRGB (full pipeline).
        
        This consolidates the entire color pipeline in one method:
        1. Load ACES from LMDB
        2. Apply CDL grading on GPU
        3. Transform to sRGB via OCIO
        
        Args:
            key: LMDB key for the image.
            cdl_params: CDL parameters dict from look_generator (e.g., {'slope': [...], 'offset': [...], ...}).
        
        Returns:
            Tuple of (srgb_32f, srgb_8u) both [H, W, 3] on GPU, normalized to [0, 1].
        
        Raises:
            KeyError: If key not found in LMDB.
        """
        # Load ACES to CPU
        aces_tensor_cpu = self._load_aces_from_lmdb(key)
        
        # Move to GPU for CDL
        aces_tensor = aces_tensor_cpu.to(self.device, non_blocking=True)
        
        # Apply CDL
        graded_aces = self.cdl_processor.apply_cdl_gpu(aces_tensor, cdl_params)
        
        # Explicitly delete intermediate ACES tensor to free memory early
        del aces_tensor
        
        # Transform to sRGB
        srgb_32f, srgb_8u = self.ocio_processor.apply_ocio_torch(graded_aces)
        
        # Free graded ACES
        del graded_aces
        
        return srgb_32f, srgb_8u

    def cleanup(self) -> None:
        """Release GPU resources."""
        self.ocio_processor.cleanup()
        self.cdl_processor.cleanup()



