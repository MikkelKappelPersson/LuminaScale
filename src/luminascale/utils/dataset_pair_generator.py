"""Dataset utilities for WebDataset batch processing.

Provides GPU-accelerated ACES loading, CDL grading, and color transformation.
Designed for WebDataset pipeline: raw EXR bytes → GPU processing → sRGB pairs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .gpu_cdl_processor import GPUCDLProcessor
from .image_generator import apply_s_curve_contrast_torch

logger = logging.getLogger(__name__)

# Track if we've logged the first sample profile to avoid spam
_first_sample_logged = False


class DatasetPairGenerator:
    """GPU-accelerated ACES→sRGB transformation for WebDataset batches.
    
    Handles the full pipeline for batch processing:
    - Decode EXR bytes from WebDataset shards
    - Apply random CDL grading on GPU
    - Transform ACES to sRGB (8-bit and 32-bit)
    """

    def __init__(
        self,
        device: torch.device,
        timing_tracker: dict[str, list] | None = None,
    ) -> None:
        """Initialize GPU transformer for WebDataset batches.
        
        Args:
            device: GPU device (e.g., torch.device('cuda:0'))
            timing_tracker: Optional dict to collect performance metrics
        """
        self.device = device
        self.timing_tracker = timing_tracker or {}
        
        # PyTorch ACES transformer
        from .pytorch_aces_transformer import ACESColorTransformer
        self.pytorch_transformer = ACESColorTransformer(device=device, use_lut=True)
        
        self.cdl_processor = GPUCDLProcessor(device=device)

    def generate_batch_from_bytes(
        self, 
        exr_bytes_list: list[bytes], 
        crop_size: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Process a list of raw EXR bytes into graded sRGB 8u/32f pairs on GPU.
        
        OPTIMIZED: Single-pass decode (eliminates validation pass).
        Pre-allocates output tensors and writes directly.
        """
        import OpenImageIO as oiio
        import time
        import tempfile
        import os
        
        t0 = time.perf_counter()
        
        decode_times, gpu_times, cdl_times, aces_times, quant_times = [], [], [], [], []
        
        # SINGLE PASS: Process all samples, collect valid results
        # Eliminates validation pass which was opening files twice per sample!
        processed_samples = []
        tensors_8u = []
        tensors_32f = []
        
        for idx, exr_bytes in enumerate(exr_bytes_list):
            temp_file = None
            try:
                t_sample = time.perf_counter()
                
                # Write to temp file (can't avoid this - OIIO needs file path)
                with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
                    tmp.write(exr_bytes)
                    temp_file = tmp.name
                
                # Open and read in ONE operation (no separate validation pass)
                buf_input = oiio.ImageInput.open(temp_file)
                if not buf_input:
                    logger.debug(f"OIIO failed to open EXR for sample {idx}")
                    continue
                
                pixels = buf_input.read_image("float")
                buf_input.close()
                
                t_decode = time.perf_counter()
                decode_times.append((t_decode - t_sample) * 1000)
                
                if pixels is None or len(pixels) == 0:
                    logger.debug(f"OIIO read_image returned None/empty for sample {idx}")
                    continue
                
                # Handle pixel format
                if pixels.ndim == 1:
                    # Need to reshape - get dimensions from ImageInput again
                    # (OIIO already closed, so use a quick re-open with minimal overhead)
                    buf_input = oiio.ImageInput.open(temp_file)
                    spec = buf_input.spec()
                    h, w, c = spec.height, spec.width, spec.nchannels
                    buf_input.close()
                    pixels = pixels.reshape((h, w, c))
                elif pixels.ndim == 3 and pixels.shape[2] != 3:
                    if pixels.shape[0] == 3:
                        pixels = pixels.transpose(1, 2, 0)
                
                # Crop
                h, w = pixels.shape[0], pixels.shape[1]
                if crop_size > 0 and (h > crop_size or w > crop_size):
                    top = max(0, (h - crop_size) // 2)
                    left = max(0, (w - crop_size) // 2)
                    pixels = pixels[top:top+crop_size, left:left+crop_size, :]
                
                # Convert to tensor
                aces_tensor = torch.from_numpy(pixels.copy()).to(self.device)
                t_gpu = time.perf_counter()
                gpu_times.append((t_gpu - t_decode) * 1000)
                
                # CDL
                from .look_generator import get_single_random_look
                look = get_single_random_look()
                t_cdl_start = time.perf_counter()
                aces_graded = self.cdl_processor.apply_cdl_gpu(aces_tensor, look)
                t_cdl = time.perf_counter()
                cdl_times.append((t_cdl - t_cdl_start) * 1000)
                
                # ACES
                t_aces_start = time.perf_counter()
                srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(aces_graded.unsqueeze(0)).squeeze(0)
                t_aces = time.perf_counter()
                aces_times.append((t_aces - t_aces_start) * 1000)
                
                # Quantization
                t_quant_start = time.perf_counter()
                srgb_8u = ((srgb_32f * 255).round().to(torch.uint8)).float() / 255.0
                srgb_8u = apply_s_curve_contrast_torch(srgb_8u, strength=2.5)
                srgb_32f = apply_s_curve_contrast_torch(srgb_32f, strength=2.5)
                t_quant = time.perf_counter()
                quant_times.append((t_quant - t_quant_start) * 1000)
                
                # Store results
                tensors_8u.append(srgb_8u.permute(2, 0, 1))
                tensors_32f.append(srgb_32f.permute(2, 0, 1))
                processed_samples.append(idx)
                
            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue
            finally:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # Stack results (now much fewer intermediate tensors!)
        if not tensors_8u:
            # Fallback for empty batch
            return torch.empty(0, 3, crop_size, crop_size, device=self.device), \
                   torch.empty(0, 3, crop_size, crop_size, device=self.device), {}
        
        srgb_8u_batch = torch.stack(tensors_8u)
        srgb_32f_batch = torch.stack(tensors_32f)
        
        total_time = (time.perf_counter() - t0) * 1000
        
        # Return timing breakdown
        timing_breakdown = {
            "oiio_decode_ms": np.mean(decode_times) if decode_times else 0,
            "gpu_transfer_ms": np.mean(gpu_times) if gpu_times else 0,
            "cdl_ms": np.mean(cdl_times) if cdl_times else 0,
            "aces_transform_ms": np.mean(aces_times) if aces_times else 0,
            "quantization_ms": np.mean(quant_times) if quant_times else 0,
            "total_decode_batch_ms": total_time
        }
        logger.debug(f"Batch({len(processed_samples)} samples) {total_time:.1f}ms: "
                    f"Decode={timing_breakdown['oiio_decode_ms']:.1f}ms, "
                    f"GPU={timing_breakdown['gpu_transfer_ms']:.1f}ms, "
                    f"CDL={timing_breakdown['cdl_ms']:.1f}ms")
            
        return srgb_8u_batch, srgb_32f_batch, timing_breakdown

    def cleanup(self) -> None:
        """Release GPU resources."""
        self.cdl_processor.cleanup()



