"""Dataset utilities for on-the-fly image generation.

Provides utilities for GPU-accelerated ACES loading and transformation.
Note: CDL grading and patching are handled by Dataset classes (e.g., OnTheFlyBDEDataset).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .gpu_cdl_processor import GPUCDLProcessor

logger = logging.getLogger(__name__)

# Track if we've logged the first sample profile to avoid spam
_first_sample_logged = False


class DatasetPairGenerator:
    """Utilities for loading ACES and applying GPU transforms.
    
    This class handles the full ACES→sRGB pipeline:
    - Decoding EXR from raw bytes (WebDataset)
    - Applying CDL grading
    - GPU-accelerated color transforms (ACES→sRGB)
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize GPU transform utilities for WebDataset batches."""
        self.device = device
        
        # PyTorch ACES transformer
        from .pytorch_aces_transformer import ACESColorTransformer
        self.pytorch_transformer = ACESColorTransformer(device=device, use_lut=True)
        
        self.cdl_processor = GPUCDLProcessor(device=device)
        
        # Track batch generation statistics for debugging disk I/O
        self._batch_count = 0
        self._total_bytes_decoded = 0

    def generate_batch_from_bytes(
        self, 
        exr_bytes_list: list[bytes], 
        crop_size: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a list of raw EXR bytes into graded sRGB 8u/32f pairs on GPU."""
        import OpenImageIO as oiio
        import time
        import tempfile
        import os
        
        self._batch_count += 1
        total_size = sum(len(b) for b in exr_bytes_list)
        self._total_bytes_decoded += total_size
        
        logger.info(f"[BATCH {self._batch_count}] Decoding {len(exr_bytes_list)} samples ({total_size/1e6:.1f}MB total) from bytes")
        
        t0 = time.perf_counter()
        srgb_8u_batch = []
        srgb_32f_batch = []
        
        # logger.info(f"Generating batch for {len(exr_bytes_list)} samples")
        
        for idx, exr_bytes in enumerate(exr_bytes_list):
            t_sample = time.perf_counter()
            # 1. Decode EXR from memory using OIIO
            # This is the most expensive CPU operation
            temp_file = None
            try:
                # Write bytes to temp file since open_mem may not be available
                with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
                    tmp.write(exr_bytes)
                    temp_file = tmp.name
                
                buf_input = oiio.ImageInput.open(temp_file)
                if not buf_input:
                    logger.error(f"OIIO failed to open EXR for sample {idx}: {oiio.geterror()}")
                    continue
                
                pixels = buf_input.read_image("float")
                buf_input.close()
            except Exception as e:
                logger.error(f"OIIO Critical Failure on sample {idx}: {e}")
                continue
            finally:
                # Clean up temp file
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                
            t_decode = time.perf_counter()
            
            # 2. Convert to GPU Tensor [C, H, W]
            # OIIO might return pixels in different formats, so handle it
            if pixels is None or len(pixels) == 0:
                logger.error(f"OIIO read_image returned None/empty for sample {idx}")
                continue
            
            # Ensure pixels are in the right format [H, W, C]
            if pixels.ndim == 1:
                # Flattened - need to reshape
                # Assume square image for now or get from OIIO spec
                spec = buf_input.spec()
                h, w, c = spec.height, spec.width, spec.nchannels
                pixels = pixels.reshape((h, w, c))
            elif pixels.ndim == 3 and pixels.shape[2] != 3:
                # Wrong channel order - permute
                if pixels.shape[0] == 3:
                    pixels = pixels.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            
            # 2a. Crop to patch size if necessary (BEFORE converting to tensor for efficiency)
            h, w = pixels.shape[0], pixels.shape[1]
            if crop_size > 0 and (h > crop_size or w > crop_size):
                # Random crop from full image
                top = max(0, (h - crop_size) // 2)
                left = max(0, (w - crop_size) // 2)
                pixels = pixels[top:top+crop_size, left:left+crop_size, :]
            
            # Keep as [H, W, C] for GPU processing (CDL expects this format)
            aces_tensor = torch.from_numpy(pixels.copy()).to(self.device)
            t_gpu = time.perf_counter()
            
            # 3. Apply Random CDL Look
            from .look_generator import get_single_random_look
            look = get_single_random_look()
            
            # Apply CDL on GPU
            aces_graded = self.cdl_processor.apply_cdl_gpu(aces_tensor, look)
            t_cdl = time.perf_counter()
            
            # 4. ACES -> sRGB Transform (returns [H, W, C] in [0, 1] float)
            # Use aces_to_srgb_32f for true 32-bit precision (ground truth)
            srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(aces_graded.unsqueeze(0)).squeeze(0)
            # Use aces_to_srgb_8u for 8-bit quantized version (degraded input)
            srgb_8u_full = self.pytorch_transformer.aces_to_srgb_8u(aces_graded.unsqueeze(0)).squeeze(0)
            t_aces = time.perf_counter()
            
            # 5. For the 8-bit input, use the 8-bit output directly (no re-quantization)
            # This maintains consistency between the two transform paths
            srgb_8u = srgb_8u_full
            t_quant = time.perf_counter()
            
            # Note: Cropping already done on CPU before GPU transfer (see above)
            # No need to crop again here
            
            # Convert from [H, W, C] to [C, H, W] for PyTorch models
            srgb_32f_batch.append(srgb_32f.permute(2, 0, 1))
            srgb_8u_batch.append(srgb_8u.permute(2, 0, 1))
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Sample {idx}: Decode={(t_decode-t_sample)*1000:.1f}ms | "
                    f"GPU={(t_gpu-t_decode)*1000:.1f}ms | "
                    f"CDL={(t_cdl-t_gpu)*1000:.1f}ms | "
                    f"ACES={(t_aces-t_cdl)*1000:.1f}ms"
                )
            elif idx == 0:
                global _first_sample_logged
                if not _first_sample_logged and logger.isEnabledFor(logging.INFO):
                    logger.info(f"First Sample Profile: Decode={(t_decode-t_sample)*1000:.1f}ms | GPU={(t_gpu-t_decode)*1000:.1f}ms")
                    _first_sample_logged = True
            
        # Stack batches: each tensor is [C, 512, 512] -> result is [N, C, 512, 512]
        res_8u = torch.stack(srgb_8u_batch)
        res_32f = torch.stack(srgb_32f_batch)
        
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[BATCH {self._batch_count}] ✓ Processed in {total_ms:.1f}ms (Total decoded so far: {self._total_bytes_decoded/1e6:.1f}MB)")
            
        return res_8u, res_32f



