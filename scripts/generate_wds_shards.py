"""
2. Calculate luma/saturation stats for metadata.
3. Save raw .exr files and a Parquet metadata table.
4. (Optional) Bake into .tar shards.
"""

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import tqdm
import json
import webdataset as wds
import pyarrow as pa
import pyarrow.parquet as pq

def create_parquet_manifest(exr_dir: str, output_parquet: str, split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Generate a Parquet manifest for existing ACES2065-1 EXR files.
    Applies an 80/10/10 split (train/val/test).
    """
    exr_path = Path(exr_dir)
    files = sorted(list(exr_path.glob("*.exr")))
    
    data = []
    num_files = len(files)
    
    # Simple deterministic split
    train_end = int(num_files * split_ratios[0])
    val_end = train_end + int(num_files * split_ratios[1])
    
    for i, f in enumerate(files):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
            
        data.append({
            "id": f.stem,
            "source_path": str(f.absolute()),
            "split": split,
            "resolution": None, # Could be added with OIIO call if needed
        })
    
    df = pd.DataFrame(data)
    df.to_parquet(output_parquet)
    print(f"Created manifest with {len(df)} entries. Split: 80/10/10")

import argparse
import webdataset as wds
import json
from pathlib import Path
import pandas as pd
import tqdm

def create_parquet_manifest(exr_dir: str, output_parquet: str, split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Generate a Parquet manifest for existing ACES2065-1 EXR files.
    Applies an 80/10/10 split (train/val/test).
    """
    exr_path = Path(exr_dir)
    # Filter for EXR files
    files = sorted(list(exr_path.glob("*.exr")))
    
    if not files:
        print(f"❌ No EXR files found in {exr_dir}")
        return

    data = []
    num_files = len(files)
    
    # Simple deterministic split based on sorted order
    train_end = int(num_files * split_ratios[0])
    val_end = train_end + int(num_files * split_ratios[1])
    
    for i, f in enumerate(files):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
            
        data.append({
            "id": f.stem,
            "source_path": str(f.absolute()),
            "split": split,
            "format": "exr",
            "filesize_bytes": f.stat().st_size
        })
    
    df = pd.DataFrame(data)
    df.to_parquet(output_parquet)
    print(f"✅ Created manifest [{output_parquet}] with {len(df)} entries. Split: 80/10/10")

def bake_webdataset(metadata_parquet: str, shard_root: str, max_shard_size_gb: float = 3.0):
    """
    Transform existing EXR files + Parquet manifest into WebDataset .tar shards.
    Shards are grouped by split (train/val/test).
    
    Each image is stored ONCE. During training, WebDataset.repeat(patches_per_image) loops through
    the data to enable on-the-fly patch generation (like OnTheFlyBDEDataset).
    """
    df = pd.read_parquet(metadata_parquet)
    max_size_bytes = int(max_shard_size_gb * 1e9)
    
    # Process each split separately so shards are non-overlapping
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        if split_df.empty:
            continue
            
        split_dir = Path(shard_root) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern for ShardWriter: train-000000.tar, etc.
        pattern = str(split_dir / f"{split}-%06d.tar")
        
        print(f"🚀 Baking {len(split_df)} images into {split} shards (on-the-fly patch generation)...")
        
        with wds.ShardWriter(pattern, maxsize=max_size_bytes) as sink:
            for img_idx, (_, row) in tqdm.tqdm(enumerate(split_df.iterrows()), total=len(split_df), desc=f"Baking {split}"):
                img_path = Path(row.source_path)
                if not img_path.exists():
                    print(f"⚠️ Warning: File not found {img_path}")
                    continue
                    
                with open(img_path, "rb") as f:
                    exr_data = f.read()
                
                # Store each image ONCE with metadata
                # During training, WebDataset.repeat(patches_per_image) will loop through the stream
                sample = {
                    "__key__": f"{img_idx:06d}",
                    "exr": exr_data,
                    "json": json.dumps({
                        "id": row.id,
                        "source": str(img_path),
                        "split": row.split,
                    }).encode("utf-8")
                }
                sink.write(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LuminaScale WebDataset Baker")
    parser.add_argument("--mode", choices=["manifest", "bake"], required=True)
    parser.add_argument("--input_dir", type=str, help="Directory containing source EXRs")
    parser.add_argument("--output_parquet", type=str, help="Path to save the Parquet manifest")
    parser.add_argument("--manifest", type=str, help="Path to the Parquet manifest for baking")
    parser.add_argument("--output_dir", type=str, help="Root directory for shards")
    parser.add_argument("--max_shard_size", default=3.0, type=float, help="Max shard size in GB")
    
    args = parser.parse_args()
    
    if args.mode == "manifest":
        create_parquet_manifest(args.input_dir, args.output_parquet)
    elif args.mode == "bake":
        bake_webdataset(args.manifest, args.output_dir, args.max_shard_size)
