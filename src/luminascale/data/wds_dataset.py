"""WebDataset-native data loading for LuminaScale."""

from __future__ import annotations
import webdataset as wds
import torch
import json
import io
import time
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class ExrDecodeResult:
    """A decoded EXR sample for the decode-in-workers path.

    pixels is the cropped/padded float32 field ready for a GPU transfer —
    a contiguous CPU torch tensor (shared-memory IPC, V3.5) or, historically,
    a numpy array; meta is the original shard metadata dict (with the
    worker-side decode timing appended under "decode_ms").
    """

    pixels: np.ndarray | None
    meta: dict


def decode_exr_to_pixels(exr_file_path: str, metadata: dict, crop_size: int) -> np.ndarray | None:
    """OIIO-decode an EXR file to a cropped + padded float32 array.

    This is the shared CPU-side decode used by BOTH the in-step path
    (DatasetPairGenerator, which calls it per sample from temp files it
    writes itself) and the worker path (decode_in_workers, which calls it
    inside the dataloader worker processes). Keeping one implementation
    guarantees identical pixels for both.

    Crop policy mirrors the historical generator behaviour: centre crop to
    crop_size when the image is larger, otherwise a full read with
    in-memory crop of the residual, then reflect/edge padding back up to
    crop_size (the residual net needs spatially uniform, pooling-divisible
    inputs).

    Args:
        exr_file_path: Path of a readable .exr file (temp files expected).
        metadata: Shard metadata dict (reserved; the authoritative crop_size
            is the explicit keyword).
        crop_size: Target square crop side (<= 0 disables cropping).

    Returns:
        float32 ndarray [H, W, C] or None when OIIO cannot open/read it.
    """
    import OpenImageIO as oiio

    buf_input = oiio.ImageInput.open(exr_file_path)
    if not buf_input:
        logger.debug(f"OIIO failed to open EXR: {exr_file_path}")
        return None

    try:
        spec = buf_input.spec()
        h, w, c = spec.height, spec.width, spec.nchannels
        crop = int(crop_size)

        if crop > 0 and (h > crop or w > crop):
            top = max(0, (h - crop) // 2)
            left = max(0, (w - crop) // 2)
            try:
                # OIIO read_region: (xbegin, xend, ybegin, yend) — reads only the ROI
                pixels = buf_input.read_region("float", left, left + crop, top, top + crop)
                if pixels is not None:
                    pixels = pixels.reshape((crop, crop, c))
            except Exception as e:
                logger.debug(f"OIIO ROI read failed ({e}), falling back to full read")
                pixels = buf_input.read_image("float")
                if pixels is not None and pixels.ndim == 1:
                    pixels = pixels.reshape((h, w, c))
                elif pixels is not None and pixels.shape[0] == 3:
                    pixels = pixels.transpose(1, 2, 0)
                if pixels is not None:
                    pixels = pixels[top: top + crop, left: left + crop, :]
        else:
            pixels = buf_input.read_image("float")
            if pixels is not None and pixels.ndim == 1:
                pixels = pixels.reshape((h, w, c))
            elif pixels is not None and pixels.shape[0] == 3:
                pixels = pixels.transpose(1, 2, 0)
    finally:
        buf_input.close()

    if pixels is None or len(pixels) == 0:
        logger.debug(f"OIIO read returned None/empty: {exr_file_path}")
        return None

    pixels = pixels.astype(np.float32, copy=False)
    if crop > 0 and (pixels.shape[0] < crop or pixels.shape[1] < crop):
        pad_h = max(0, crop - pixels.shape[0])
        pad_w = max(0, crop - pixels.shape[1])
        pad_mode = "reflect" if (pixels.shape[0] > pad_h and pixels.shape[1] > pad_w) else "edge"
        pixels = np.pad(pixels, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
    return pixels


def decode_exr_bytes_to_result(exr_bytes: bytes, metadata: dict, crop_size: int) -> ExrDecodeResult:
    """Worker-side decode: EXR bytes -> ExrDecodeResult (float32 pixels, meta).

    Used inside webdataset worker processes (CPU-only; never initialises
    CUDA). Temp files are written to the OS temp dir (node-local /tmp under
    singularity binds) and removed immediately.
    """
    t0 = time.perf_counter()
    temp_file = None
    pixels = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
            tmp.write(exr_bytes)
            temp_file = tmp.name
        pixels = decode_exr_to_pixels(temp_file, metadata, crop_size)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
    decode_ms = (time.perf_counter() - t0) * 1000.0
    meta = dict(metadata or {})
    meta["decode_ms"] = decode_ms
    if pixels is None:
        meta["decode_ok"] = False
        return ExrDecodeResult(pixels=None, meta=meta)
    # V3.5: ship a contiguous CPU tensor instead of numpy — torch's pickler
    # moves tensors through shared memory (no per-batch pipe copy of ~50 MB,
    # which was the measured serial floor after V3).
    pixels_t = torch.from_numpy(pixels).contiguous()
    return ExrDecodeResult(pixels=pixels_t, meta=meta)


def decode_exr_and_json(sample: dict, decode_in_workers: bool = False, crop_size: int = 512) -> tuple[bytes | ExrDecodeResult, dict]:
    """Decoder for LuminaScale shards.

    Two modes:
    - decode_in_workers=False (historical): pass raw EXR bytes through; the
      trainer decodes them OIIO-side inside training_step.
    - decode_in_workers=True: decode here, in the dataloader worker process
      (CPU-only) and ship the float32 crop instead of the bytes, so the
      step only pays for the GPU-side colour ops.

    Returns:
        (exr_bytes_or_result, metadata_dict)
    """
    exr_data = sample.get("exr")
    json_data = sample.get("json")

    metadata = json.loads(json_data.decode("utf-8")) if json_data else {}

    if decode_in_workers:
        return decode_exr_bytes_to_result(exr_data, metadata, crop_size), metadata
    return exr_data, metadata


def collate_wds_batch(batch) -> tuple[list, list]:
    """Custom collate function for WebDataset batches.

    WebDataset.batched() already groups items, so it returns tuples of lists.
    Input format from .batched(): ([item1, item2, ...], [item1, item2, ...])
    where each item is either a (bytes, dict) pair (raw mode) or an
    (ExrDecodeResult, dict) pair (decode-in-workers mode).

    This collate function flattens the nested structure into separate lists.

    Args:
        batch: Tuple of (batched_items_list, batched_items_list) where each batch
               has gotten pre-batched by .batched()

    Returns:
        (exr_list, metadata_list) suitable for the trainer (bytes in raw
        mode, ExrDecodeResult objects in decode-in-workers mode)
    """
    if not batch or (isinstance(batch, (tuple, list)) and len(batch) == 0):
        return [], []

    # Handle the case where .batched() returns a tuple of (list_of_items, list_of_items)
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        # If first element is a list of tuples (items), unpack here
        if isinstance(batch[0], list) and isinstance(batch[1], list):
            # .batched() format: tuple of two lists containing items
            exr_list = []
            metadata_list = []

            items_list = batch[0]  # List of exr items (bytes or ExrDecodeResult)
            metadata_list_raw = batch[1]  # List of metadata

            # The items are already separated by .batched()
            # Just return both lists directly

            # Handle case where items are themselves lists (nested batching)
            for idx, item in enumerate(items_list):
                if isinstance(item, list) and len(item) > 0:
                    # Flatten nested list
                    exr_list.append(item[0])
                else:
                    exr_list.append(item)

            # Similar for metadata
            for idx, item in enumerate(metadata_list_raw):
                if isinstance(item, list) and len(item) > 0:
                    # Flatten nested list
                    metadata_list.append(item[0] if isinstance(item[0], dict) else item)
                else:
                    metadata_list.append(item)

            return exr_list, metadata_list

    # Fallback for other formats
    return batch if isinstance(batch, tuple) else (batch, [])

def _worker_dump_init(worker_id: int) -> None:
    """Debug helper (module level so it pickles under spawn): each worker dumps
    its stack and exits after LUMINA_WORKER_DUMP seconds — deadlock diagnosis."""
    import faulthandler

    if os.environ.get("LUMINA_WORKER_DUMP"):
        faulthandler.dump_traceback_later(float(os.environ["LUMINA_WORKER_DUMP"]), exit=True)


class _DecoderFn:
    """Picklable decoder callable (spawn-safe): wraps decode_exr_and_json with
    the dataset's decode_in_workers / crop_size settings."""

    def __init__(self, decode_in_workers: bool, crop_size: int) -> None:
        self.decode_in_workers = decode_in_workers
        self.crop_size = crop_size

    def __call__(self, sample: dict):
        return decode_exr_and_json(
            sample, decode_in_workers=self.decode_in_workers, crop_size=self.crop_size
        )



class LuminaScaleWebDataset:
    """WebDataset wrapper for streaming training data on HPC."""
    
    def __init__(
        self,
        shard_path: str | Path,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        is_training: bool = True,
        metadata_parquet: str | Path | None = None,
        split: str = "train",
        patches_per_image: int = 1,
        decode_in_workers: bool = False,
        crop_size: int = 512,
    ):
        # Handle shard_path: can be a string, list, or string representation of a list
        import ast
        import glob as glob_module
        from pathlib import Path as PathlibPath
        from hydra.utils import get_original_cwd
        
        if isinstance(shard_path, list):
            # Already a list
            shards = shard_path
        else:
            shard_path_str = str(shard_path)
            # Try to parse as list literal (e.g., "['file1.tar', 'file2.tar']")
            if shard_path_str.startswith('[') and shard_path_str.endswith(']'):
                try:
                    shards = ast.literal_eval(shard_path_str)
                except (ValueError, SyntaxError):
                    # Fall back to glob expansion
                    shards = sorted(glob_module.glob(shard_path_str))
            else:
                # Check if path is a directory - if so, find all .tar files
                try:
                    orig_cwd_temp = PathlibPath(get_original_cwd())
                except:
                    orig_cwd_temp = PathlibPath.cwd()
                
                dir_path = orig_cwd_temp / shard_path_str
                if dir_path.exists() and dir_path.is_dir():
                    # It's a directory - find all .tar files
                    shards = sorted([str(f) for f in dir_path.glob("*.tar")])
                else:
                    # Try glob expansion for patterns like "train-{000000..000001}.tar" or "*.tar"
                    shards = sorted(glob_module.glob(shard_path_str))
        
        # Convert all paths to absolute, resolving from the original working directory
        # (before Hydra changed it)
        try:
            orig_cwd = PathlibPath(get_original_cwd())
        except:
            # Fallback if get_original_cwd() fails
            orig_cwd = PathlibPath.cwd()
        
        if isinstance(shards, list):
            # Filter to only existing files and convert to absolute paths
            shards = [
                str((orig_cwd / p).resolve()) 
                for p in shards 
                if (orig_cwd / p).exists()
            ]
            self.shard_path = shards if shards else []
        else:
            self.shard_path = str((orig_cwd / shards).resolve())
        self.batch_size = batch_size
        self.split = split
        self.patches_per_image = max(1, patches_per_image)  # Ensure at least 1
        self.decode_in_workers = decode_in_workers
        self.crop_size = crop_size
        
        # Try to load total samples from parquet metadata
        self.total_samples = None
        if metadata_parquet:
            metadata_path = orig_cwd / str(metadata_parquet)
            if metadata_path.exists():
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(str(metadata_path))
                    
                    # Filter by split if 'split' column exists
                    if 'split' in table.column_names:
                        split_col = table['split'].to_pylist()
                        filtered_indices = [i for i, s in enumerate(split_col) if s == split]
                        self.total_samples = len(filtered_indices)
                        logger.debug(f"Filtered metadata: {self.total_samples} samples in '{split}' split from {metadata_parquet}")
                    else:
                        # No split column, use all samples
                        self.total_samples = len(table)
                        logger.debug(f"Loaded metadata: {self.total_samples} total samples from {metadata_parquet} (no split column found)")
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {metadata_parquet}: {e}")
        
        
        # Build the WebDataset pipeline
        # Shuffle shards with buffer during training for distributed randomness; deterministic for validation
        # shardshuffle expects integer buffer size (not boolean)
        shardshuffle_buffer = 100 if is_training else False
        dataset = wds.WebDataset(self.shard_path, resampled=False, empty_check=False, shardshuffle=shardshuffle_buffer)
        
        # Split by worker (instead of .shardselection method)
        dataset = dataset.select(wds.shardlists.split_by_worker)
        
        if is_training:
            # Shuffle the ORIGINAL 880 images first (before repeat)
            # This ensures randomization while keeping consecutive repeats together for caching
            dataset = dataset.shuffle(shuffle_buffer)
        
        # Repeat the dataset patches_per_image times for on-the-fly patch generation
        # After shuffle, we get: [shuffled img1, shuffled img2, ...] repeated 32 times
        # This creates 880 * 32 total samples without duplicating storage
        # Caching in _process_batch() will reuse decoded images across repeats
        if self.patches_per_image > 1:
            dataset = dataset.repeat(self.patches_per_image)
            logger.debug(f"Configured dataset to repeat {self.patches_per_image} times for on-the-fly patch generation")
            
        # Map our custom decoder (worker-side decode when enabled)
        dataset = dataset.map(_DecoderFn(self.decode_in_workers, self.crop_size))
        
        # Batching
        dataset = dataset.batched(batch_size)
        
        # Apply collate function to convert batch format
        dataset = dataset.map(collate_wds_batch)
        
        self.dataset = dataset
        logger.debug(f"Initialized WebDataset with {len(self.shard_path) if isinstance(self.shard_path, list) else 'multiple'} shards")

    def get_loader(self, num_workers: int = 4, prefetch_factor: int = 2):
        """Returns a stable WebLoader (Dataloader equivalent).
        
        Args:
            num_workers: Number of worker processes for parallel data loading
            prefetch_factor: Batches to prefetch per worker (default 2)
                             Higher values increase prefetching but memory usage.
        """
        logger.debug(f"Creating WebLoader with num_workers={num_workers}, prefetch_factor={prefetch_factor}")
        
        loader_kwargs = {
            "batch_size": None,  # Batching already handled by .batched()
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": True if num_workers > 0 else False,
        }
        # Decode-in-workers mode MUST use the spawn context: forked workers
        # inherit the parent's libgomp/OpenMP team state (initialised by CPU
        # side torch work like parameter init) and then deadlock forever
        # inside OIIO's first parallel read (observed on AI-Cloud L40S,
        # 2026-09-09). Spawned workers start a fresh interpreter instead.
        # Requires a picklable map fn (_DecoderFn) and a __main__ guard in
        # the entry script — both in place.
        if self.decode_in_workers and num_workers > 0:
            loader_kwargs["multiprocessing_context"] = "spawn"
        # Debug: LUMINA_WORKER_DUMP=<seconds> makes each worker dump its stack
        # (and exit) after that many seconds — deadlock diagnosis only.
        if os.environ.get("LUMINA_WORKER_DUMP"):
            loader_kwargs["worker_init_fn"] = _worker_dump_init
        
        # Add prefetch_factor if num_workers > 0 (only meaningful with multiprocessing)
        # PyTorch requires prefetch_factor >= 1 when num_workers > 0
        if num_workers > 0:
            effective_prefetch = max(prefetch_factor, 1)
            if prefetch_factor != effective_prefetch:
                logger.warning(
                    f"prefetch_factor={prefetch_factor} invalid with num_workers={num_workers}; "
                    f"PyTorch requires prefetch_factor >= 1. Using prefetch_factor={effective_prefetch}"
                )
            try:
                loader_kwargs["prefetch_factor"] = effective_prefetch
            except TypeError:
                # WebLoader might not support prefetch_factor, skip it
                logger.debug("WebLoader does not support prefetch_factor parameter")
        
        loader = wds.WebLoader(self.dataset, **loader_kwargs)
        return loader
    
    def get_estimated_batches(self) -> int | None:
        """Return estimated total batches if metadata is available.
        
        With on-the-fly patch generation via WebDataset.repeat(patches_per_image),
        the total samples = unique_images_in_metadata × patches_per_image.
        This provides an accurate estimate for progress tracking.
        """
        if self.total_samples is not None:
            # Shards already contain patches_per_image copies of each image
            # So total_samples_with_patches = parquet_unique_images × patches_per_image
            total_samples_with_patches = self.total_samples * self.patches_per_image
            estimated_batches = total_samples_with_patches // self.batch_size
            logger.debug(
                f"Estimated total batches: {estimated_batches} "
                f"({self.total_samples} unique images × {self.patches_per_image} patches_per_image / {self.batch_size} batch_size) "
                f"[shards contain pre-baked patches]"
            )
            return estimated_batches
        return None
