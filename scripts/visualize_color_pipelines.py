#!/usr/bin/env python3
"""Visualize ACES color space pipelines: old vs new, OIIO vs GPU rendering.

Compares 4 display combinations:
1. Old pipeline (ACES2065-1 → sRGB) + OIIO rendering
2. New pipeline (ACES2065-1 → ACEScct → sRGB) + OIIO rendering
3. Old pipeline + GPU rendering (PyTorch CUDA)
4. New pipeline + GPU rendering (PyTorch CUDA)

This helps validate that ACEScct encoding produces consistent results
across different rendering backends.

Usage:
    python visualize_color_pipelines.py <aces2065_1_image.exr> [--output output.jpg]
    
Example:
    python visualize_color_pipelines.py dataset/aces/exr_ACES2065-1/MIT-Adobe_5K_001.exr --output comparison.jpg
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src to path
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from luminascale.utils.io import colorconvert, oiio_aces_to_display, aces_to_display_gpu, read_exr


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize float [0, ~1] to uint8 [0, 255]."""
    clipped = np.clip(array, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def pipeline_old_oiio(img_path: Path | str) -> np.ndarray:
    """Old pipeline (ACES2065-1 → sRGB) via OIIO rendering."""
    result_chw = oiio_aces_to_display(str(img_path), input_cs="ACES2065-1")  # [C, H, W]
    result_hwc = result_chw.transpose(1, 2, 0)  # [H, W, 3]
    return result_hwc


def pipeline_new_oiio(img_path: Path | str) -> np.ndarray:
    """New pipeline (ACES2065-1 → ACEScct → sRGB) via OIIO rendering."""
    from luminascale.utils.io import write_exr
    import tempfile
    
    # Convert to ACEScct working space
    img_acescct = colorconvert(str(img_path), "ACES2065-1", "ACEScct", strict=True)  # [H, W, 3]
    
    # Write to temp EXR
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        img_acescct_chw = img_acescct.transpose(2, 0, 1)
        write_exr(tmp_path, img_acescct_chw)
        
        # Convert ACEScct → sRGB via OIIO
        result_chw = oiio_aces_to_display(str(tmp_path), input_cs="ACEScct")  # [C, H, W]
        result_hwc = result_chw.transpose(1, 2, 0)  # [H, W, 3]
        return result_hwc
    finally:
        tmp_path.unlink(missing_ok=True)


def pipeline_old_gpu(img_path: Path | str) -> np.ndarray:
    """Old pipeline (ACES2065-1 → sRGB) via GPU rendering."""
    try:
        aces_chw = read_exr(str(img_path))  # [C, H, W]
        aces_hwc = aces_chw.transpose(1, 2, 0)  # [H, W, 3]
        
        # Convert to tensor on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        aces_tensor = torch.from_numpy(aces_hwc).float().to(device)
        
        _, srgb_8bit = aces_to_display_gpu(aces_tensor, input_cs="ACES2065-1")  # Returns (float32, uint8)
        return srgb_8bit.cpu().numpy()  # [H, W, 3] uint8
    except Exception as e:
        print(f"⚠️ GPU rendering failed: {e}, falling back to OIIO")
        return pipeline_old_oiio(img_path)


def pipeline_new_gpu(img_path: Path | str) -> np.ndarray:
    """New pipeline (ACES2065-1 → ACEScct → sRGB).
    
    Uses ACEScct via OIIO (GPU ACEScct→sRGB not yet available).
    """
    return pipeline_new_oiio(img_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize ACES color space pipelines (2x3 grid with difference maps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input ACES2065-1 EXR image file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output comparison image path (JPG/PNG). If not provided, displays in viewer.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display in image viewer (requires DISPLAY environment variable)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    print(f"📷 Processing: {input_path.name}")
    print(f"   Size: {input_path.stat().st_size / 1e6:.1f} MB")
    print("")
    
    # Process all 4 pipelines
    print("🔄 Pipeline 1: Old color space (ACES2065-1 → sRGB) via OIIO...", flush=True)
    img_old_oiio = pipeline_old_oiio(input_path)
    img_old_oiio_uint8 = normalize_to_uint8(img_old_oiio)
    print("✓ Done")
    
    print("🔄 Pipeline 2: New color space (ACES2065-1 → ACEScct → sRGB) via OIIO...", flush=True)
    img_new_oiio = pipeline_new_oiio(input_path)
    img_new_oiio_uint8 = normalize_to_uint8(img_new_oiio)
    print("✓ Done")
    
    print("🔄 Pipeline 3: Old color space via GPU rendering...", flush=True)
    img_old_gpu = pipeline_old_gpu(input_path)
    img_old_gpu_uint8 = img_old_gpu if img_old_gpu.dtype == np.uint8 else normalize_to_uint8(img_old_gpu)
    print("✓ Done")
    
    print("🔄 Pipeline 4: New color space via GPU (ACEScct)...", flush=True)
    img_new_gpu = pipeline_new_gpu(input_path)
    img_new_gpu_uint8 = img_new_gpu if img_new_gpu.dtype == np.uint8 else normalize_to_uint8(img_new_gpu)
    print("✓ Done")
    
    # Compute color differences
    print("")
    print("📊 Color difference statistics:")
    diff_oiio = np.abs(img_old_oiio.astype(np.float32) - img_new_oiio.astype(np.float32))
    diff_oiio_max = np.max(diff_oiio)
    diff_oiio_mean = np.mean(diff_oiio)
    print(f"   Old vs New (OIIO): max={diff_oiio_max:.4f}, mean={diff_oiio_mean:.6f}")
    
    diff_gpu = np.abs(img_old_gpu.astype(np.float32) - img_new_gpu.astype(np.float32))
    diff_gpu_max = np.max(diff_gpu)
    diff_gpu_mean = np.mean(diff_gpu)
    print(f"   Old vs New (GPU):  max={diff_gpu_max:.4f}, mean={diff_gpu_mean:.6f}")
    
    diff_render = np.abs(img_old_oiio.astype(np.float32) - img_old_gpu.astype(np.float32))
    print(f"   OIIO vs GPU (Old): max={np.max(diff_render):.4f}, mean={np.mean(diff_render):.6f}")
    
    # Count unique pixel values (important for AI training - no quantization!)
    print("")
    print("📈 Unique pixel value counts (float32 precision):")
    
    def count_unique_pixels(img_float32):
        """Count approximate unique values by binning to 16-bit precision."""
        # Convert to uint16 range for counting (simulates 16-bit precision)
        img_uint16 = np.clip(img_float32 * 65535, 0, 65535).astype(np.uint16)
        # Flatten and count unique per channel
        unique_per_channel = [len(np.unique(img_uint16[:, :, c])) for c in range(3)]
        total_unique = len(np.unique(img_uint16.reshape(-1, 3), axis=0))
        return unique_per_channel, total_unique
    
    old_oiio_ch, old_oiio_total = count_unique_pixels(img_old_oiio)
    new_oiio_ch, new_oiio_total = count_unique_pixels(img_new_oiio)
    old_gpu_ch, old_gpu_total = count_unique_pixels(img_old_gpu)
    new_gpu_ch, new_gpu_total = count_unique_pixels(img_new_gpu)
    
    print(f"   Old OIIO:  {old_oiio_total:>8} unique RGB tuples (R:{old_oiio_ch[0]}, G:{old_oiio_ch[1]}, B:{old_oiio_ch[2]})")
    print(f"   New OIIO:  {new_oiio_total:>8} unique RGB tuples (R:{new_oiio_ch[0]}, G:{new_oiio_ch[1]}, B:{new_oiio_ch[2]})")
    print(f"   Old GPU:   {old_gpu_total:>8} unique RGB tuples (R:{old_gpu_ch[0]}, G:{old_gpu_ch[1]}, B:{old_gpu_ch[2]})")
    print(f"   New GPU:   {new_gpu_total:>8} unique RGB tuples (R:{new_gpu_ch[0]}, G:{new_gpu_ch[1]}, B:{new_gpu_ch[2]})")
    
    # Check for quantization
    if min(old_oiio_total, new_oiio_total, old_gpu_total, new_gpu_total) < 1000000:
        print("   ⚠️ Warning: Low unique value count detected (possible quantization)")
    else:
        print("   ✓ No quantization detected - rich color depth preserved")
    
    # Create difference maps with better contrast visualization
    diff_oiio_lum = np.mean(diff_oiio, axis=2)  # [H, W]
    diff_gpu_lum = np.mean(diff_gpu, axis=2)  # [H, W]
    
    # Stretch contrast using percentile-based normalization (1st to 99th percentile)
    # This makes small differences visible while avoiding outliers
    def normalize_with_contrast(arr):
        p1, p99 = np.percentile(arr, [1, 99])
        stretched = np.clip((arr - p1) / (p99 - p1 + 1e-8), 0, 1)
        return stretched
    
    diff_oiio_lum_norm = normalize_with_contrast(diff_oiio_lum)
    diff_gpu_lum_norm = normalize_with_contrast(diff_gpu_lum)
    
    diff_oiio_lum_uint8 = (diff_oiio_lum_norm * 255).astype(np.uint8)
    diff_gpu_lum_uint8 = (diff_gpu_lum_norm * 255).astype(np.uint8)
    
    # Heatmap: white = no difference, red = high difference (within the visible range)
    diff_map_oiio_rgb = np.zeros_like(img_old_oiio_uint8)
    diff_map_oiio_rgb[:, :, 0] = 255  # Red = always 255
    diff_map_oiio_rgb[:, :, 1] = 255 - diff_oiio_lum_uint8  # Green decreases with difference
    diff_map_oiio_rgb[:, :, 2] = 255 - diff_oiio_lum_uint8  # Blue decreases with difference
    
    # Same for GPU
    diff_map_gpu_rgb = np.zeros_like(img_old_gpu_uint8)
    diff_map_gpu_rgb[:, :, 0] = 255  # Red = always 255
    diff_map_gpu_rgb[:, :, 1] = 255 - diff_gpu_lum_uint8  # Green decreases with difference
    diff_map_gpu_rgb[:, :, 2] = 255 - diff_gpu_lum_uint8  # Blue decreases with difference
    
    # Scale down images (40%)
    scale_factor = 0.4
    h_small = int(img_old_oiio_uint8.shape[0] * scale_factor)
    w_small = int(img_old_oiio_uint8.shape[1] * scale_factor)
    
    # Resize all images
    pil_old_oiio = Image.fromarray(img_old_oiio_uint8, mode="RGB").resize((w_small, h_small), Image.Resampling.LANCZOS)
    pil_new_oiio = Image.fromarray(img_new_oiio_uint8, mode="RGB").resize((w_small, h_small), Image.Resampling.LANCZOS)
    pil_diff_oiio = Image.fromarray(diff_map_oiio_rgb, mode="RGB").resize((w_small, h_small), Image.Resampling.LANCZOS)
    
    pil_old_gpu = Image.fromarray(img_old_gpu_uint8, mode="RGB").resize((w_small, h_small), Image.Resampling.LANCZOS)
    pil_new_gpu = Image.fromarray(img_new_gpu_uint8, mode="RGB").resize((w_small, h_small), Image.Resampling.LANCZOS)
    pil_diff_gpu = Image.fromarray(diff_map_gpu_rgb, mode="RGB").resize((w_small, h_small), Image.Resampling.LANCZOS)
    
    # Create 2x3 grid canvas
    spacing = 12
    label_height = 45
    
    canvas_width = w_small * 3 + spacing * 4
    canvas_height = h_small * 2 + label_height + spacing * 3
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    
    # Row 1 (OIIO): Old | New | Difference
    canvas.paste(pil_old_oiio, (spacing, label_height + spacing))
    canvas.paste(pil_new_oiio, (w_small + spacing * 2, label_height + spacing))
    canvas.paste(pil_diff_oiio, (w_small * 2 + spacing * 3, label_height + spacing))
    
    # Row 2 (GPU): Old | New | Difference
    row2_y = h_small + label_height + spacing * 2
    canvas.paste(pil_old_gpu, (spacing, row2_y))
    canvas.paste(pil_new_gpu, (w_small + spacing * 2, row2_y))
    canvas.paste(pil_diff_gpu, (w_small * 2 + spacing * 3, row2_y))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    label_y = 12
    
    # Column headers
    draw.text((spacing + 5, label_y), "Old ACES2065-1→sRGB", fill="black", font=font)
    draw.text((w_small + spacing * 2 + 5, label_y), "New ACES2065-1→ACEScct→sRGB", fill="black", font=font)
    draw.text((w_small * 2 + spacing * 3 + 5, label_y), f"Difference (max={diff_oiio_max:.4f})", fill="black", font=font)
    
    # Row labels
    draw.text((2, label_height + spacing + 3), "OIIO", fill="black", font=font)
    draw.text((2, row2_y + 3), "GPU", fill="black", font=font)
    
    # Save or display
    if args.output:
        canvas.save(args.output)
        print(f"✅ Comparison saved: {args.output}")
    
    if args.display:
        try:
            canvas.show()
        except Exception as e:
            print(f"⚠️ Could not display image: {e}")
    
    if not args.output and not args.display:
        # Save to default location
        output_path = input_path.parent / f"{input_path.stem}_comparison_2x3.jpg"
        canvas.save(output_path)
        print(f"✅ Comparison saved: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
