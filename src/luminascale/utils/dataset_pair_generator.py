"""Dataset Pair Generators for BDE and Color Conversion Workflows.

Provides reusable generators for creating training pairs:
- BDE pairs: ACES → (sRGB 32-bit, sRGB 8-bit) with random looks
- Color Convert pairs: ACES → sRGB 32-bit with reference looks
"""

from __future__ import annotations

import logging
import pickle
from typing import Generator, Tuple

import lmdb
import numpy as np
import torch

from luminascale.utils.gpu_cdl_processor import GPUCDLProcessor
from luminascale.utils.gpu_torch_processor import GPUTorchProcessor
from luminascale.utils.look_generator import get_single_random_look

logger = logging.getLogger(__name__)


class DatasetPairGenerator:
    """Generate training pairs for BDE or Color Conversion workflows.
    
    Modes:
    - "bde": Bit-Depth Expansion → (reference_srgb_32f, synthetic_srgb_8u, cdl_params)
               Reference: 32-bit sRGB (ground truth), Synthetic: 8-bit sRGB (degraded input)
    - "cc": Color Convert → (reference_aces, synthetic_srgb, cdl_params)
               Reference: ACES 32-bit (ground truth), Synthetic: sRGB 32-bit (color input to convert)
    
    For each image:
    - Load ACES from LMDB
    - Apply random CDL grading
    - Transform to sRGB
    """

    def __init__(
        self,
        lmdb_env: lmdb.Environment,
        device: torch.device,
        mode: str = "bde",
        keys_cache: list[str] | None = None,
    ) -> None:
        """Initialize dataset pair generator.
        
        Args:
            lmdb_env: Open LMDB environment in read-only mode.
            device: GPU device (e.g., torch.device("cuda")).
            mode: "bde" for Bit-Depth Expansion or "cc" for Color Convert.
            keys_cache: Pre-loaded keys list (optional, auto-loads if None).
        """
        if mode not in ("bde", "cc"):
            raise ValueError(f"Unknown mode: {mode}. Must be 'bde' or 'cc'")
        
        self.env = lmdb_env
        self.device = device
        self.mode = mode
        self.cdl_processor = GPUCDLProcessor(device=device)
        self.ocio_processor = GPUTorchProcessor(headless=True)
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

        # Convert to GPU tensor [H, W, 3]
        aces_tensor = torch.from_numpy(hdr_np).to(self.device, non_blocking=True)
        return aces_tensor.permute(1, 2, 0)  # [C, H, W] → [H, W, 3]

    def generate_pairs(
        self, num_images: int, start_idx: int = 0
    ) -> Generator:
        """Generate training pairs.
        
        Both modes yield: (reference, synthetic, cdl_params)
        
        BDE mode:
            reference: sRGB 32-bit (ground truth quality)
            synthetic: sRGB 8-bit (degraded input to expand)
        
        CC mode:
            reference: ACES 32-bit (ground truth color space)
            synthetic: sRGB 32-bit (input to convert to ACES)
        
        All tensors are [H, W, 3] on GPU.
        """
        for i in range(num_images):
            idx = (start_idx + i) % len(self.keys_cache)
            key = self.keys_cache[idx]

            try:
                # Load ACES
                aces_tensor = self._load_aces_from_lmdb(key)

                # Apply random CDL
                cdl_params = get_single_random_look()
                graded_aces = self.cdl_processor.apply_cdl_gpu(aces_tensor, cdl_params)

                # Transform to sRGB (always get both 32f and 8u)
                srgb_32f, srgb_8u = self.ocio_processor.apply_ocio_torch(graded_aces)

                # Yield (reference, synthetic, cdl_params) for both modes
                if self.mode == "bde":
                    # Reference: high-quality sRGB, Synthetic: quantized 8-bit
                    yield srgb_32f, srgb_8u, cdl_params
                else:  # cc
                    # Reference: ACES (ground truth), Synthetic: sRGB (input to convert)
                    yield aces_tensor, srgb_32f, cdl_params
            except Exception as e:
                logger.warning(f"Failed to process {key}: {e}, skipping")
                continue

    def cleanup(self) -> None:
        """Release GPU resources."""
        self.cdl_processor.cleanup()
        self.ocio_processor.cleanup()


# Backwards compatibility aliases
BDEPairGenerator = lambda env, device, keys_cache=None: DatasetPairGenerator(
    env, device, mode="bde", keys_cache=keys_cache
)
ColorConvertPairGenerator = lambda env, device, keys_cache=None: DatasetPairGenerator(
    env, device, mode="cc", keys_cache=keys_cache
)
