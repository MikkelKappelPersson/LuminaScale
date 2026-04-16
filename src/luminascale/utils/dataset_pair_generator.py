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
        
        # Detailed timing buckets
        decode_times, gpu_times, cdl_times, aces_times, quant_times = [], [], [], [], []
        temp_io_times, oiio_open_times, look_gen_times, permute_times = [], [], [], []
        
        # SINGLE PASS: Process all samples, collect valid results
        # Eliminates validation pass which was opening files twice per sample!
        processed_samples = []
        tensors_8u = []
        tensors_32f = []
        
        for idx, exr_bytes in enumerate(exr_bytes_list):
            temp_file = None
            try:
                t_sample = time.perf_counter()
                
                # === TEMP FILE I/O ===
                t_io_start = time.perf_counter()
                with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
                    tmp.write(exr_bytes)
                    temp_file = tmp.name
                t_io = time.perf_counter()
                temp_io_times.append((t_io - t_io_start) * 1000)
                
                # === OIIO OPEN ===
                t_open_start = time.perf_counter()
                buf_input = oiio.ImageInput.open(temp_file)
                t_open = time.perf_counter()
                oiio_open_times.append((t_open - t_open_start) * 1000)
                
                if not buf_input:
                    logger.debug(f"OIIO failed to open EXR for sample {idx}")
                    continue
                
                # Get image spec for ROI calculation (fast, no decode yet)
                spec = buf_input.spec()
                h, w, c = spec.height, spec.width, spec.nchannels
                
                # Calculate crop region to minimize OIIO decode overhead
                # Instead of reading full image and cropping, read only the ROI
                if crop_size > 0 and (h > crop_size or w > crop_size):
                    top = max(0, (h - crop_size) // 2)
                    left = max(0, (w - crop_size) // 2)
                    # ROI decode: only read the crop region from disk
                    # This avoids decoding and loading full image data
                    try:
                        # OIIO read_region: (xbegin, xend, ybegin, yend)
                        pixels = buf_input.read_region("float", left, left + crop_size, top, top + crop_size)
                        if pixels is not None:
                            # Returned shape is (crop_size, crop_size, nchannels)
                            pixels = pixels.reshape((crop_size, crop_size, c))
                    except Exception as e:
                        logger.debug(f"OIIO ROI read failed, falling back to full read: {e}")
                        # Fallback: read full image if ROI fails
                        pixels = buf_input.read_image("float")
                        if pixels is not None and pixels.ndim == 1:
                            pixels = pixels.reshape((h, w, c))
                        elif pixels is not None and pixels.shape[0] == 3:
                            pixels = pixels.transpose(1, 2, 0)
                        # Crop in memory
                        if pixels is not None:
                            pixels = pixels[top:top+crop_size, left:left+crop_size, :]
                else:
                    # Image smaller than crop size or no crop needed - read full
                    pixels = buf_input.read_image("float")
                    if pixels is not None and pixels.ndim == 1:
                        pixels = pixels.reshape((h, w, c))
                    elif pixels is not None and pixels.shape[0] == 3:
                        pixels = pixels.transpose(1, 2, 0)
                
                buf_input.close()
                
                t_decode = time.perf_counter()
                decode_times.append((t_decode - t_open) * 1000)
                
                if pixels is None or len(pixels) == 0:
                    logger.debug(f"OIIO read returned None/empty for sample {idx}")
                    continue
                
                # === GPU TRANSFER ===
                t_gpu_start = time.perf_counter()
                aces_tensor = torch.from_numpy(pixels.copy()).to(self.device)
                t_gpu = time.perf_counter()
                gpu_times.append((t_gpu - t_gpu_start) * 1000)
                
                # === LOOK GENERATION ===
                t_look_start = time.perf_counter()
                from .look_generator import get_single_random_look
                look = get_single_random_look()
                t_look = time.perf_counter()
                look_gen_times.append((t_look - t_look_start) * 1000)
                
                # === CDL ===
                t_cdl_start = time.perf_counter()
                aces_graded = self.cdl_processor.apply_cdl_gpu(aces_tensor, look)
                t_cdl = time.perf_counter()
                cdl_times.append((t_cdl - t_cdl_start) * 1000)
                
                # === ACES TRANSFORM ===
                t_aces_start = time.perf_counter()
                srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(aces_graded.unsqueeze(0)).squeeze(0)
                t_aces = time.perf_counter()
                aces_times.append((t_aces - t_aces_start) * 1000)
                
                # === QUANTIZATION ===
                t_quant_start = time.perf_counter()
                srgb_8u = ((srgb_32f * 255).round().to(torch.uint8)).float() / 255.0
                srgb_8u = apply_s_curve_contrast_torch(srgb_8u, strength=2.5)
                srgb_32f = apply_s_curve_contrast_torch(srgb_32f, strength=2.5)
                t_quant = time.perf_counter()
                quant_times.append((t_quant - t_quant_start) * 1000)
                
                # === PERMUTE & STORE ===
                t_permute_start = time.perf_counter()
                tensors_8u.append(srgb_8u.permute(2, 0, 1))
                tensors_32f.append(srgb_32f.permute(2, 0, 1))
                t_permute = time.perf_counter()
                permute_times.append((t_permute - t_permute_start) * 1000)
                
                processed_samples.append(idx)
                
            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue
            finally:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # Stack results (now much fewer intermediate tensors!)
        t_stack_start = time.perf_counter()
        if not tensors_8u:
            # Fallback for empty batch
            return torch.empty(0, 3, crop_size, crop_size, device=self.device), \
                   torch.empty(0, 3, crop_size, crop_size, device=self.device), {}
        
        srgb_8u_batch = torch.stack(tensors_8u)
        srgb_32f_batch = torch.stack(tensors_32f)
        t_stack = time.perf_counter()
        stack_time_ms = (t_stack - t_stack_start) * 1000
        
        total_time = (time.perf_counter() - t0) * 1000
        
        # Return detailed timing breakdown
        timing_breakdown = {
            "oiio_decode_ms": np.mean(decode_times) if decode_times else 0,
            "gpu_transfer_ms": np.mean(gpu_times) if gpu_times else 0,
            "cdl_ms": np.mean(cdl_times) if cdl_times else 0,
            "aces_transform_ms": np.mean(aces_times) if aces_times else 0,
            "quantization_ms": np.mean(quant_times) if quant_times else 0,
            "temp_io_ms": np.mean(temp_io_times) if temp_io_times else 0,
            "oiio_open_ms": np.mean(oiio_open_times) if oiio_open_times else 0,
            "look_gen_ms": np.mean(look_gen_times) if look_gen_times else 0,
            "permute_ms": np.mean(permute_times) if permute_times else 0,
            "stack_batch_ms": stack_time_ms,
            "total_decode_batch_ms": total_time
        }
        
        # Calculate overhead from new granular timings
        explicit_overhead = timing_breakdown["temp_io_ms"] + timing_breakdown["oiio_open_ms"] + timing_breakdown["look_gen_ms"] + timing_breakdown["permute_ms"]
        logger.debug(f"Batch({len(processed_samples)} samples) {total_time:.1f}ms: "
                    f"Decode={timing_breakdown['oiio_decode_ms']:.1f}ms, "
                    f"GPU={timing_breakdown['gpu_transfer_ms']:.1f}ms, "
                    f"CDL={timing_breakdown['cdl_ms']:.1f}ms, "
                    f"Overhead(TempIO+Open+Look+Perm)={explicit_overhead:.1f}ms")
            
        return srgb_8u_batch, srgb_32f_batch, timing_breakdown

    def cleanup(self) -> None:
        """Release GPU resources."""
        self.cdl_processor.cleanup()



