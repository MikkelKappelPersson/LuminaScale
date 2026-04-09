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

# Track if we've logged the first sample profile to avoid spam
_first_sample_logged = False


class DatasetPairGenerator:
    """Utilities for loading ACES and applying GPU transforms.
    
    This class handles the full ACES→sRGB pipeline:
    - Loading ACES from LMDB or raw bytes (WebDataset)
    - Applying CDL grading (optional)
    - GPU-accelerated color transforms (OCIO)
    """

    def __init__(
        self,
        lmdb_env: lmdb.Environment | None,
        device: torch.device,
        keys_cache: list[str] | None = None,
        timing_tracker: dict | None = None,
        lmdb_path: str | None = None,
    ) -> None:
        """Initialize ACES loader and GPU transform utilities."""
        self.env = lmdb_env
        self.lmdb_path = lmdb_path
        self.device = device
        self.timing_tracker = timing_tracker or {}
        
        # PyTorch ACES transformer
        from .pytorch_aces_transformer import ACESColorTransformer
        self.pytorch_transformer = ACESColorTransformer(device=device, use_lut=True)
        
        self.cdl_processor = GPUCDLProcessor(device=device)
        self.keys_cache = keys_cache or (self._load_keys() if self.env else [])

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
            srgb_32f = self.pytorch_transformer.aces_to_srgb_8u(aces_graded.unsqueeze(0)).squeeze(0)
            t_aces = time.perf_counter()
            
            # 5. Quantize to 8-bit (convert to uint8 then back to float for dequantization)
            # srgb_32f is already in [0, 1], so multiply by 255, cast to uint8, then divide by 255
            srgb_8u = ((srgb_32f * 255).round().to(torch.uint8)).float() / 255.0
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
        
        # logger.info(f"Batch processed in {(time.perf_counter()-t0)*1000:.1f}ms (Size: {len(exr_bytes_list)})")
            
        return res_8u, res_32f
    
    def __del__(self) -> None:
        """Cleanup persistent transaction and cursor on destruction."""
        try:
            if self.persistent_cursor is not None:
                self.persistent_cursor.close()
            if self.persistent_txn is not None:
                self.persistent_txn.commit()
        except Exception as e:
            logger.debug(f"Error during DatasetPairGenerator cleanup: {e}")

    def _load_keys(self) -> list[str]:
        """Load image keys from LMDB."""
        with self.env.begin(write=False) as txn:
            keys_buf = txn.get(b"__keys__")
            if keys_buf is None:
                raise ValueError("LMDB missing __keys__ - invalid format")
            return pickle.loads(keys_buf)

    def _load_aces_from_lmdb(self, key: str) -> torch.Tensor:
        """Load single ACES image from LMDB and return as [H, W, 3] GPU tensor."""
        import time
        t_start = time.perf_counter()
        
        # Lazy-initialize cursor on first read (avoids early initialization overhead)
        if self.persistent_cursor is None:
            logger.debug("[CURSOR INIT] Initializing persistent cursor on first read")
            self.persistent_txn = self.env.begin(write=False)
            self.persistent_cursor = self.persistent_txn.cursor()
        
        # Periodically reset cursor to prevent accumulated OS page cache pressure
        # (cursor state degrades after many sequential reads; lightweight reset only)
        self._cursor_read_count += 1
        if self._cursor_read_count > self._cursor_reset_interval:
            logger.debug(f"[CURSOR RESET] Resetting cursor after {self._cursor_read_count} reads")
            try:
                self.persistent_cursor.close()
                self.persistent_txn.commit()
            except Exception as e:
                logger.debug(f"Error closing old cursor: {e}")
            
            # Create fresh transaction and cursor (lightweight operation, just advances position)
            self.persistent_txn = self.env.begin(write=False)
            self.persistent_cursor = self.persistent_txn.cursor()
            self._cursor_read_count = 0
        
        # Use persistent cursor to read
        key_bytes = key.encode("ascii")
        buf = self.persistent_cursor.get(key_bytes)
        if buf is None:
            raise KeyError(f"Key not found: {key}")

        t_read = time.perf_counter() - t_start
        
        # Parse header: 3 × int32 = 12 bytes
        header = np.frombuffer(buf[:12], dtype=np.uint32)
        H, W, C = int(header[0]), int(header[1]), int(header[2])

        # Extract ACES data: float32 array (make writable copy)
        hdr_size = H * W * C * 4
        hdr_np = np.frombuffer(buf[12 : 12 + hdr_size], dtype=np.float32).reshape(C, H, W).copy()

        # Convert to CPU tensor [H, W, 3] first, then move to GPU only when needed
        aces_tensor = torch.from_numpy(hdr_np).permute(1, 2, 0)  # [C, H, W] → [H, W, 3]
        
        # Log slow LMDB reads (>500ms indicates file handle or cache issues)
        if t_read > 0.5:
            logger.warning(f"[SLOW LMDB READ] {key}: {t_read*1000:.1f}ms for {H}×{W}×{C}")
        
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
        self, key: str, cdl_params: dict[str, Any], dataset_ref: Any | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load ACES, apply CDL grading, and transform to sRGB (full pipeline).
        
        Args:
            key: LMDB key for the image.
            cdl_params: CDL parameters dict from look_generator.
            dataset_ref: Reference to OnTheFlyBDEDataset for tracking LMDB read times.
        
        Returns:
            Tuple of (srgb_32f, srgb_8u) both [H, W, 3] on GPU.
        """
        import time
        
        t0 = time.perf_counter()
        
        # Load ACES to CPU
        t_lmdb_start = time.perf_counter()
        aces_tensor_cpu = self._load_aces_from_lmdb(key)
        t_lmdb = time.perf_counter() - t_lmdb_start
        logger.debug(f"[BATCH TIMING]   LMDB load: {t_lmdb*1000:.1f}ms, shape={aces_tensor_cpu.shape}")
        
        # Track LMDB read time for performance monitoring
        if dataset_ref is not None:
            dataset_ref._lmdb_read_times.append(t_lmdb * 1000)
            # Keep only recent history to avoid unbounded memory growth
            if len(dataset_ref._lmdb_read_times) > 100:
                dataset_ref._lmdb_read_times = dataset_ref._lmdb_read_times[-100:]
        
        # Move to GPU for CDL
        t_transfer_start = time.perf_counter()
        aces_tensor = aces_tensor_cpu.to(self.device, non_blocking=True)
        t_transfer = time.perf_counter() - t_transfer_start
        logger.debug(f"[BATCH TIMING]   GPU transfer: {t_transfer*1000:.1f}ms")
        
        # Apply CDL
        t_cdl_start = time.perf_counter()
        graded_aces = self.cdl_processor.apply_cdl_gpu(aces_tensor, cdl_params)
        t_cdl = time.perf_counter() - t_cdl_start
        logger.debug(f"[BATCH TIMING]   CDL apply: {t_cdl*1000:.1f}ms")
        
        # Explicitly delete intermediate ACES tensor to free memory early
        del aces_tensor
        
        # Transform to sRGB (only sync at END to allow pipeline overlap)
        t_aces_start = time.perf_counter()
        srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(graded_aces)
        srgb_8u = self.pytorch_transformer.aces_to_srgb_8u(graded_aces)
        # Single sync at end to ensure all GPU ops complete before returning
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



