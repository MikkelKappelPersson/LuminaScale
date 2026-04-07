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
    t_start = time.perf_counter()
    
    print(f"[DECODE] Processing sample: {sample.get('__key__')}")
    
    # .exr is raw bytes at this stage
    exr_data = sample.get("exr")
    # .json is also raw bytes
    json_data = sample.get("json")
    
    if exr_data:
        print(f"[DECODE] EXR data size: {len(exr_data)} bytes")
    else:
        print(f"[DECODE] WARNING: No EXR data found in sample!")
    
    if json_data:
        metadata = json.loads(json_data.decode("utf-8"))
        print(f"[DECODE] Metadata keys: {list(metadata.keys())}")
    else:
        metadata = {}
        print(f"[DECODE] No metadata found")
        
    # We pass the raw EXR bytes to the GPU for decoding via OIIO/Cuda
    # This avoids expensive CPU-side EXR decoding
    decode_time_ms = (time.perf_counter() - t_start) * 1000
    print(f"[DECODE] ✓ Decoded in {decode_time_ms:.2f}ms")
    
    return exr_data, metadata


def collate_wds_batch(batch) -> tuple[list[bytes], list[dict]]:
    """Custom collate function for WebDataset batches.
    
    WebDataset.batched() creates a list of (exr_bytes, metadata) tuples.
    This collate function separates them into two lists for processing.
    
    Args:
        batch: List of (exr_bytes, metadata_dict) tuples from WebDataset
        
    Returns:
        (exr_bytes_list, metadata_list) tuple suitable for GPU processing
    """
    print(f"\n[COLLATE] Collating batch with {len(batch)} samples")
    
    if not batch:
        print(f"[COLLATE] WARNING: Empty batch!")
        return [], []
    
    exr_bytes_list = []
    metadata_list = []
    
    for idx, item in enumerate(batch):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            exr_bytes, metadata = item
            exr_bytes_list.append(exr_bytes)
            metadata_list.append(metadata)
            print(f"[COLLATE]   Sample {idx}: {len(exr_bytes)} bytes")
        else:
            print(f"[COLLATE] WARNING: Unexpected batch item format at index {idx}: {type(item)}")
    
    print(f"[COLLATE] ✓ Batch ready: {len(exr_bytes_list)} samples, returning as (list, list)")
    return exr_bytes_list, metadata_list

class LuminaScaleWebDataset:
    """WebDataset wrapper for streaming training data on HPC."""
    
    def __init__(
        self,
        shard_path: str | Path,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        is_training: bool = True
    ):
        self.shard_path = str(shard_path)
        self.batch_size = batch_size
        
        print(f"[WDS.__init__] Starting initialization...")
        print(f"[WDS.__init__] Shard path: {self.shard_path}")
        logger.info(f"Initializing LuminaScaleWebDataset with shards: {self.shard_path}")
        
        # Build the WebDataset pipeline
        print(f"[WDS.__init__] Creating WebDataset object...")
        dataset = wds.WebDataset(self.shard_path, resampled=False)
        print(f"[WDS.__init__] ✓ WebDataset created")
        
        # Split by worker (instead of .shardselection method)
        print(f"[WDS.__init__] Applying split_by_worker...")
        dataset = dataset.select(wds.shardlists.split_by_worker)
        print(f"[WDS.__init__] ✓ split_by_worker applied")
        
        if is_training:
            # Nodes-level shuffling (shards are randomized)
            print(f"[WDS.__init__] Applying shuffle (buffer={shuffle_buffer})...")
            dataset = dataset.shuffle(shuffle_buffer)
            print(f"[WDS.__init__] ✓ shuffle applied")
            
        # Map our custom decoder
        print(f"[WDS.__init__] Applying custom decoder...")
        dataset = dataset.map(decode_exr_and_json)
        print(f"[WDS.__init__] ✓ decoder applied")
        
        # Batching
        print(f"[WDS.__init__] Applying batching (batch_size={batch_size})...")
        dataset = dataset.batched(batch_size)
        print(f"[WDS.__init__] ✓ batching applied")
        
        # Apply collate function to convert batch format
        print(f"[WDS.__init__] Applying collate function...")
        dataset = dataset.map(collate_wds_batch)
        print(f"[WDS.__init__] ✓ collate function applied")
        
        self.dataset = dataset
        print(f"[WDS.__init__] ✓ WebDataset initialization complete!")

    def get_loader(self, num_workers: int = 4):
        """Returns a stable WebLoader (Dataloader equivalent)."""
        print(f"[WDS.get_loader] Creating WebLoader with num_workers={num_workers}")
        logger.info(f"Creating WebLoader with num_workers={num_workers}")
        
        print(f"[WDS.get_loader] Initializing WebLoader object...")
        loader = wds.WebLoader(
            self.dataset, 
            batch_size=None, # Batching already handled by .batched()
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"[WDS.get_loader] ✓ WebLoader created and returning")
        return loader
