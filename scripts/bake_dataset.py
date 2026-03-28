#!/usr/bin/env python3
"""Dataset baker script to generate LDR images from ACES EXR files.

This script iterates through ACES EXR files, applies a random CDL look,
and saves the result as an 8-bit sRGB PNG to eliminate the CPU bottleneck during training.

Usage:
    python scripts/bake_dataset.py --input-dir dataset/temp/aces --output-dir dataset/temp/srgb_looks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import imageio

# Add src to path for imports
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from luminascale.utils.look_generator import get_single_random_look
from luminascale.utils.io import aces_to_srgb_with_look, aces_to_display

def main() -> int:
    parser = argparse.ArgumentParser(description="Bake LDR dataset from ACES EXRs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(project_root / "dataset" / "temp" / "aces"),
        help="Input directory for ACES EXR files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "dataset" / "temp" / "srgb_looks"),
        help="Output directory for generated sRGB PNGs",
    )
    parser.add_argument(
        "--natural-ratio",
        type=float,
        default=0.5,
        help="Ratio of images to have NO look (natural) applied (0.0=all graded, 0.5=half natural)",
    )
    
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory {input_dir} does not exist.")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exr_files = sorted(list(input_dir.glob("*.exr")))
    if not exr_files:
        print(f"❌ No EXR files found in {input_dir}")
        return 1
        
    print(f"🚀 Baking {len(exr_files)} images from {input_dir} to {output_dir}")
    print(f"📊 Natural ratio: {args.natural_ratio:.2f} (approx. every {1.0/args.natural_ratio:.1f} image)")
    
    for i, img_path in enumerate(tqdm(exr_files, desc="Baking")):
        try:
            # Decide if this should be a "natural" image or "graded"
            is_natural = (i % int(1.0 / args.natural_ratio)) == 0 if args.natural_ratio > 0 else False
            
            if is_natural:
                # ACES -> sRGB (Natural)
                pixels = aces_to_display(img_path)
            else:
                # ACES -> Graded -> sRGB (Look)
                random_look = get_single_random_look()
                pixels = aces_to_srgb_with_look(img_path, random_look)
            
            # 3. Ensure RGB format and clip
            if pixels.shape[2] == 4:
                pixels = pixels[:, :, :3]
            pixels = np.clip(pixels, 0, 1)
            
            # 4. Convert to 8-bit
            pixels_8bit = (pixels * 255.0).astype(np.uint8)
            
            # 5. Save as PNG
            output_path = output_dir / f"{img_path.stem}.png"
            imageio.imwrite(output_path, pixels_8bit)
            
        except Exception as e:
            print(f"\n❌ Failed to process {img_path.name}: {e}")
            
    print(f"\n✅ Finished! Baked dataset is in {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
