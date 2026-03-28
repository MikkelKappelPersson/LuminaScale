#!/usr/bin/env python3
"""Packer script to convert individual ACES (EXR) and sRGB (PNG) images into a high-performance LMDB.

Usage:
    python scripts/pack_lmdb.py --aces-dir dataset/temp/aces --srgb-dir dataset/temp/srgb_looks --output-path dataset/training_data.lmdb
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import imageio.v3 as iio
import lmdb
import numpy as np
from tqdm import tqdm

# Add src to path for internal imports if needed
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from luminascale.utils.io import image_to_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pack_dataset(aces_dir: Path, srgb_dir: Path, output_path: Path, map_size: int = 10**12) -> None:
    """Pack ACES and sRGB image pairs into an LMDB database.
    
    Args:
        aces_dir: Directory containing .exr ACES ground truth files.
        srgb_dir: Directory containing .png sRGB look/input files.
        output_path: Path where the .lmdb file will be created.
        map_size: Maximum size of the database in bytes (default 1TB).
    """
    aces_files = sorted(list(aces_dir.glob("*.exr")))
    if not aces_files:
        logger.error(f"No EXR files found in {aces_dir}")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open LMDB environment
    # map_size is the maximum allowed size on disk
    env = lmdb.open(str(output_path), map_size=map_size)
    
    keys = []
    
    with env.begin(write=True) as txn:
        for exr_path in tqdm(aces_files, desc="Packing LMDB"):
            img_name = exr_path.stem
            png_path = srgb_dir / f"{img_name}.png"
            
            if not png_path.exists():
                logger.warning(f"No matching PNG found for {exr_path.name}, skipping.")
                continue
                
            try:
                # 1. Load HDR (ACES) - convert to float32 numpy [C, H, W]
                # We use image_to_tensor which already handles EXR via OpenImageIO
                hdr_tensor = image_to_tensor(exr_path)
                hdr_array = hdr_tensor.numpy().astype(np.float32)
                
                # 2. Load LDR (sRGB) - keep as uint8 numpy [C, H, W]
                ldr_pixels = iio.imread(png_path)
                # Convert HWC to CHW to match torch convention
                ldr_array = np.transpose(ldr_pixels, (2, 0, 1)).astype(np.uint8)
                
                # 3. Create a data packet
                # We store shape info to avoid ambiguity during unpacking
                data_packet = {
                    "hdr": hdr_array,
                    "ldr": ldr_array,
                    "shape": hdr_array.shape, # (C, H, W)
                }
                
                # 4. Serialize and store
                txn.put(
                    img_name.encode("ascii"),
                    pickle.dumps(data_packet, protocol=pickle.HIGHEST_PROTOCOL)
                )
                keys.append(img_name)
                
            except Exception as e:
                logger.error(f"Failed to pack {img_name}: {e}")

        # Store the list of keys so the Dataset doesn't have to scan the DB
        txn.put(b"__keys__", pickle.dumps(keys))

    env.close()
    logger.info(f"Successfully packed {len(keys)} pairs into {output_path}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Pack ACES and sRGB images into LMDB")
    parser.add_argument("--aces-dir", type=str, default="dataset/temp/aces", help="Path to ACES EXRs")
    parser.add_argument("--srgb-dir", type=str, default="dataset/temp/srgb_looks", help="Path to sRGB PNGs")
    parser.add_argument("--output-path", type=str, default="dataset/training_data.lmdb", help="Output LMDB path")
    parser.add_argument("--map-size", type=int, default=10**12, help="Max DB size in bytes (1TB default)")
    
    args = parser.parse_args()
    
    pack_dataset(
        Path(args.aces_dir),
        Path(args.srgb_dir),
        Path(args.output_path),
        args.map_size
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
