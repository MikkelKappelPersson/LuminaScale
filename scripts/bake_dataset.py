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
import OpenImageIO as oiio
from PIL import Image as PILImage
import imageio
import os

# Add src to path for imports
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Set OCIO environment variable before importing any OCIO-dependent modules
# Use the project-local config
ocio_config_path = project_root / "config" / "aces" / "studio-config.ocio"
if ocio_config_path.exists():
    os.environ["OCIO"] = str(ocio_config_path)

from luminascale.utils.look_generator import get_single_random_look
from luminascale.utils.io import ocio_aces_to_srgb_with_look, ocio_aces_to_display

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
        help="Output directory for generated sRGB files",
    )
    parser.add_argument(
        "--natural-ratio",
        type=float,
        default=0.5,
        help="Ratio of images to have NO look (natural) applied (0.0=all graded, 0.5=half natural)",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Save as 32-bit float EXR instead of 8-bit PNG",
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
    print(f"💾 Output format: {'32-bit EXR' if args.float32 else '8-bit PNG'}")
    
    for i, img_path in enumerate(tqdm(exr_files, desc="Baking")):
        try:
            # Decide if this should be a "natural" image or "graded"
            is_natural = (i % int(1.0 / args.natural_ratio)) == 0 if args.natural_ratio > 0 else False
            
            if is_natural:
                # ACES -> sRGB (Natural) using GPU
                pixels = ocio_aces_to_display(img_path)
            else:
                # ACES -> Graded -> sRGB (Look) using GPU
                random_look = get_single_random_look()
                pixels = ocio_aces_to_srgb_with_look(img_path, random_look)
            
            # 3. Ensure RGB format and clip
            if pixels.shape[2] == 4:
                pixels = pixels[:, :, :3]
            pixels = np.clip(pixels, 0, 1)
            
            if args.float32:
                # 4. Save as 32-bit float EXR using imageio
                # NOTE: This saves as uncompressed EXR. For production dataset compression,
                # use: exrheader -update -c zip <file> or use the compress_dataset.py helper
                output_path = output_dir / f"{img_path.stem}.exr"
                # Convert to imageio format (requires channel order adjustment)
                # imageio auto-detects EXR format from .exr extension
                import imageio
                imageio.imwrite(output_path, pixels.astype(np.float32))
            else:
                # 4. Convert to 8-bit
                pixels_8bit = (pixels * 255.0).astype(np.uint8)
                # 5. Save as PNG using PIL (compressed)
                output_path = output_dir / f"{img_path.stem}.png"
                pil_img = PILImage.fromarray(pixels_8bit, mode="RGB")
                pil_img.save(output_path)
            
        except Exception as e:
            print(f"\n❌ Failed to process {img_path.name}: {e}")
            
    print(f"\n✅ Finished! Baked dataset is in {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
