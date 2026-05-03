from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import sys
from tqdm import tqdm
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from luminascale.utils.io import read_exr, write_exr, oiio_aces_to_display


def generate_image_assets():
    """
    Generates sRGB 32-float and 8-uint image assets from ACES2065-1 EXR images.
    
    Reads from: assets/aces-imgs
    Writes to: assets/srgb_32f (EXR) and assets/srgb_8u (PNG)
    """
    base_path = Path(__file__).parent.parent
    aces_dir = base_path / "assets" / "aces-imgs"
    srgb_32f_dir = base_path / "assets" / "srgb_32f"
    srgb_8u_dir = base_path / "assets" / "srgb_8u"

    # Create output directories
    srgb_32f_dir.mkdir(parents=True, exist_ok=True)
    srgb_8u_dir.mkdir(parents=True, exist_ok=True)

    # Find all EXR files in aces-imgs
    aces_files = list(aces_dir.glob("*.exr"))
    
    if not aces_files:
        print(f"No ACES EXR files found in {aces_dir}")
        return

    print(f"Found {len(aces_files)} ACES images. Processing...")

    for aces_file in tqdm(aces_files):
        try:
            # 1. Transform ACES2065-1 to sRGB display space using OIIO
            # result is [C, H, W] float32
            srgb_32f_np = oiio_aces_to_display(aces_file)
            
            # 2. Save sRGB 32f as EXR
            exr_out_path = srgb_32f_dir / aces_file.name
            write_exr(exr_out_path, srgb_32f_np)

            # 3. Create 8u quantized variant
            # Clipping to [0, 1] before quantization
            srgb_8u_np = np.clip(srgb_32f_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
            
            # Transpose back to [H, W, C] for PIL
            srgb_8u_np = srgb_8u_np.transpose(1, 2, 0)
            
            # 4. Save sRGB 8u as PNG
            png_out_path = srgb_8u_dir / aces_file.with_suffix(".png").name
            Image.fromarray(srgb_8u_np).save(png_out_path)

        except Exception as e:
            print(f"Error processing {aces_file.name}: {e}")

if __name__ == "__main__":
    generate_image_assets()
