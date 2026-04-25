"""
2. Calculate luma/saturation stats for metadata.
3. Save raw .exr files and a Parquet metadata table.
4. (Optional) Bake into .tar shards.
"""

from __future__ import annotations
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
import tqdm
import json
import webdataset as wds
import pyarrow as pa
import pyarrow.parquet as pq
import OpenImageIO as oiio

import argparse
import webdataset as wds
import json
from pathlib import Path
import pandas as pd
import tqdm

def create_parquet_manifest(
    exr_dir: str,
    output_parquet: str,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_samples: int | None = None,
):
    """
    Generate a Parquet manifest for existing ACES2065-1 EXR files.
    Applies an 80/10/10 split (train/val/test).
    
    Safe to re-run: will regenerate manifest if source files differ.
    """
    exr_path = Path(exr_dir)
    # Filter for EXR files
    files = sorted(list(exr_path.glob("*.exr")))

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")
        files = files[:max_samples]
    
    if not files:
        print(f"❌ No EXR files found in {exr_dir}")
        return

    # Check if manifest already exists with same number of files
    output_path = Path(output_parquet)
    if output_path.exists():
        existing_df = pd.read_parquet(output_parquet)
        if len(existing_df) == len(files):
            print(f"⊘ Manifest already exists with {len(existing_df)} entries. Skipping regeneration.")
            return
        else:
            print(f"⚠️  Manifest exists but has {len(existing_df)} entries (found {len(files)} files).")
            print(f"   Regenerating manifest...")

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet)
    sample_suffix = f" (limited to {max_samples})" if max_samples is not None else ""
    print(f"✅ Created manifest [{output_parquet}] with {len(df)} entries{sample_suffix}. Split: 80/10/10")

def _read_and_crop_exr_bytes(img_path: Path, crop_size: int, rng: np.random.Generator) -> tuple[bytes, dict]:
    """Read EXR, apply one random crop (if possible), and return EXR bytes + crop metadata."""
    inp = oiio.ImageInput.open(str(img_path))
    if inp is None:
        raise RuntimeError(f"Failed to open EXR: {img_path}")

    try:
        spec = inp.spec()
        width = spec.width
        height = spec.height
        channels = spec.nchannels
        pixels = inp.read_image(format=oiio.FLOAT)
    finally:
        inp.close()

    if pixels is None:
        raise RuntimeError(f"Failed to read EXR pixels: {img_path}")

    image = np.array(pixels, dtype=np.float32).reshape(height, width, channels)

    crop_applied = False
    x0 = 0
    y0 = 0
    crop_w = width
    crop_h = height

    if crop_size > 0 and width >= crop_size and height >= crop_size:
        x0 = int(rng.integers(0, width - crop_size + 1))
        y0 = int(rng.integers(0, height - crop_size + 1))
        crop_w = crop_size
        crop_h = crop_size
        image = image[y0:y0 + crop_h, x0:x0 + crop_w, :]
        crop_applied = True

    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        out = oiio.ImageOutput.create(str(tmp_path))
        if out is None:
            raise RuntimeError(f"Failed to create EXR output: {tmp_path}")
        out_spec = oiio.ImageSpec(crop_w, crop_h, channels, oiio.FLOAT)
        if not out.open(str(tmp_path), out_spec):
            raise RuntimeError(f"Failed to open EXR output for writing: {tmp_path}")
        if not out.write_image(image):
            raise RuntimeError(f"Failed to write cropped EXR data: {tmp_path}")
        out.close()

        with open(tmp_path, "rb") as f:
            exr_data = f.read()
    finally:
        tmp_path.unlink(missing_ok=True)

    crop_meta = {
        "crop_applied": crop_applied,
        "crop_size": crop_size,
        "crop_x": x0,
        "crop_y": y0,
        "crop_width": crop_w,
        "crop_height": crop_h,
        "source_width": width,
        "source_height": height,
    }
    return exr_data, crop_meta


def bake_webdataset(
    metadata_parquet: str,
    shard_root: str,
    max_shard_size_gb: float = 3.0,
    crop_size: int = 2048,
    crop_seed: int = 42,
):
    """
    Transform existing EXR files + Parquet manifest into WebDataset .tar shards.
    Shards are grouped by split (train/val/test).
    
    Safe to re-run: skips splits that already have shards, resumes from last checkpoint.
    
    Each image is stored ONCE. During training, WebDataset.repeat(patches_per_image) loops through
    the data to enable on-the-fly patch generation (like OnTheFlyBDEDataset).
    """
    df = pd.read_parquet(metadata_parquet)
    max_size_bytes = int(max_shard_size_gb * 1e9)
    
    # Process each split separately so shards are non-overlapping
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        if split_df.empty:
            print(f"⊘ No {split} data in manifest, skipping")
            continue

        split_seed = crop_seed + {"train": 0, "val": 1, "test": 2}[split]
        rng = np.random.default_rng(split_seed)
            
        split_dir = Path(shard_root) / split
        
        # Check if this split already has shards
        existing_shards = list(split_dir.glob(f"{split}-*.tar"))
        if existing_shards:
            print(f"✓ {split} split already has {len(existing_shards)} shards. Skipping.")
            continue
        
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern for ShardWriter: train-000000.tar, etc.
        pattern = str(split_dir / f"{split}-%06d.tar")
        
        print(
            f"🚀 Baking {len(split_df)} images into {split} shards "
            f"(random crop={crop_size}x{crop_size}, seed={split_seed})..."
        )
        
        try:
            with wds.ShardWriter(pattern, maxsize=max_size_bytes) as sink:
                for img_idx, (_, row) in tqdm.tqdm(enumerate(split_df.iterrows()), total=len(split_df), desc=f"Baking {split}"):
                    img_path = Path(row.source_path)
                    if not img_path.exists():
                        print(f"⚠️ Warning: File not found {img_path}")
                        continue
                        
                    exr_data, crop_meta = _read_and_crop_exr_bytes(img_path, crop_size, rng)
                    
                    # Store each image ONCE with metadata
                    # During training, WebDataset.repeat(patches_per_image) will loop through the stream
                    sample = {
                        "__key__": f"{img_idx:06d}",
                        "exr": exr_data,
                        "json": json.dumps({
                            "id": row.id,
                            "source": str(img_path),
                            "split": row.split,
                            "crop": crop_meta,
                        }).encode("utf-8")
                    }
                    sink.write(sample)
            print(f"✓ Completed {split} shards")
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted! Partial {split} shards saved. Resume by re-running.")
            raise
        except Exception as e:
            print(f"\n❌ Error baking {split}: {e}")
            print(f"   Partial {split} shards saved. Clean up incomplete shards and re-run to resume.")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LuminaScale WebDataset Baker")
    parser.add_argument("--mode", choices=["manifest", "bake"], required=True)
    parser.add_argument("--input_dir", type=str, help="Directory containing source EXRs")
    parser.add_argument("--output_parquet", type=str, help="Path to save the Parquet manifest")
    parser.add_argument("--manifest", type=str, help="Path to the Parquet manifest for baking")
    parser.add_argument("--output_dir", type=str, help="Root directory for shards")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit manifest generation to the first N sorted EXR files")
    parser.add_argument("--max_shard_size", default=3.0, type=float, help="Max shard size in GB")
    parser.add_argument("--crop_size", default=2048, type=int, help="Random crop size before packing (0 disables)")
    parser.add_argument("--crop_seed", default=42, type=int, help="Seed for reproducible random crops")
    
    args = parser.parse_args()
    
    if args.mode == "manifest":
        create_parquet_manifest(args.input_dir, args.output_parquet, max_samples=args.max_samples)
    elif args.mode == "bake":
        bake_webdataset(args.manifest, args.output_dir, args.max_shard_size, args.crop_size, args.crop_seed)
