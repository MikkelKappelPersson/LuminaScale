"""WebDataset-native data loading for LuminaScale."""

from __future__ import annotations
import webdataset as wds
import torch
import json
import io
import time
import logging
import torch.nn.functional as F
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

def decode_exr_and_json(sample: dict) -> tuple[bytes, dict]:
    """Custom decoder for LuminaScale Shards.
    
    Transforms:
    - .exr -> Raw bytes (needs processing on GPU)
    - .json -> Decoded metadata dictionary
    
    Returns:
        (exr_bytes, metadata_dict) tuple
    """
    # .exr is raw bytes at this stage
    exr_data = sample.get("exr")
    # .json is also raw bytes
    json_data = sample.get("json")
    
    if json_data:
        metadata = json.loads(json_data.decode("utf-8"))
    else:
        metadata = {}
        
    # We pass the raw EXR bytes to the GPU for decoding via OIIO/Cuda
    # This avoids expensive CPU-side EXR decoding
    return exr_data, metadata
    
    return exr_data, metadata


def collate_wds_batch(batch) -> tuple[list[bytes], list[dict]]:
    """Custom collate function for WebDataset batches.
    
    WebDataset.batched() already groups items, so it returns tuples of lists.
    Input format from .batched(): ([item1, item2, ...], [item1, item2, ...])
    where each item is a (exr_bytes, metadata) tuple.
    
    This collate function flattens the nested structure into separate lists.
    
    Args:
        batch: Tuple of (batched_items_list, batched_items_list) where each batch
               has gotten pre-batched by .batched()
        
    Returns:
        (exr_bytes_list, metadata_list) tuple suitable for GPU processing
    """
    if not batch or (isinstance(batch, (tuple, list)) and len(batch) == 0):
        return [], []
    
    # Handle the case where .batched() returns a tuple of (list_of_items, list_of_items)
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        # If first element is a list of tuples (items), unpack here
        if isinstance(batch[0], list) and isinstance(batch[1], list):
            # .batched() format: tuple of two lists containing items
            exr_bytes_list = []
            metadata_list = []
            
            items_list = batch[0]  # List of exr bytes
            metadata_list_raw = batch[1]  # List of metadata
            
            # The items are already separated by .batched()
            # Just return both lists directly
            
            # Handle case where items are themselves lists (nested batching)
            for idx, item in enumerate(items_list):
                if isinstance(item, list) and len(item) > 0:
                    # Flatten nested list
                    exr_data = item[0] if isinstance(item[0], bytes) else item
                    exr_bytes_list.append(exr_data)
                else:
                    exr_bytes_list.append(item)
            
            # Similar for metadata
            for idx, item in enumerate(metadata_list_raw):
                if isinstance(item, list) and len(item) > 0:
                    # Flatten nested list
                    metadata_list.append(item[0] if isinstance(item[0], dict) else item)
                else:
                    metadata_list.append(item)
            
            return exr_bytes_list, metadata_list
    
    # Fallback for other formats
    return batch if isinstance(batch, tuple) else (batch, [])

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
                        logger.info(f"Filtered metadata: {self.total_samples} samples in '{split}' split from {metadata_parquet}")
                    else:
                        # No split column, use all samples
                        self.total_samples = len(table)
                        logger.info(f"Loaded metadata: {self.total_samples} total samples from {metadata_parquet} (no split column found)")
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {metadata_parquet}: {e}")
        
        
        # Build the WebDataset pipeline
        dataset = wds.WebDataset(self.shard_path, resampled=False)
        
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
            logger.info(f"Configured dataset to repeat {self.patches_per_image} times for on-the-fly patch generation")
            
        # Map our custom decoder
        dataset = dataset.map(decode_exr_and_json)
        
        # Batching
        dataset = dataset.batched(batch_size)
        
        # Apply collate function to convert batch format
        dataset = dataset.map(collate_wds_batch)
        
        self.dataset = dataset
        logger.info(f"Initialized WebDataset with {len(self.shard_path) if isinstance(self.shard_path, list) else 'multiple'} shards")

    def get_loader(self, num_workers: int = 4):
        """Returns a stable WebLoader (Dataloader equivalent)."""
        logger.info(f"Creating WebLoader with num_workers={num_workers}")
        
        loader = wds.WebLoader(
            self.dataset, 
            batch_size=None, # Batching already handled by .batched()
            num_workers=num_workers,
            pin_memory=True
        )
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
            logger.info(
                f"Estimated total batches: {estimated_batches} "
                f"({self.total_samples} unique images × {self.patches_per_image} patches_per_image / {self.batch_size} batch_size) "
                f"[shards contain pre-baked patches]"
            )
            return estimated_batches
        return None
