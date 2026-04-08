#!/usr/bin/env python3
"""Test both oiio_aces_to_display implementations to verify they render identically."""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the current version from io.py
from luminascale.utils.io import oiio_aces_to_display as aces_to_display_current

# Import just to verify dependencies
import OpenImageIO as oiio

# Create the old version inline to test both
import tempfile
import os

def aces_to_display_old(
    aces_image_path,
    looks=None,
    display="sRGB - Display",
    view="ACES 2.0 - SDR 100 nits (Rec.709)",
):
    """Old version (from attachment #sym:oiio_aces_to_display)."""
    assert oiio is not None, (
        "OpenImageIO is required for oiio_aces_to_display. "
        "Install it via: pixi install openimageio"
    )

    aces_path = Path(aces_image_path)
    assert aces_path.exists(), f"ACES image not found: {aces_path}"

    # Load ACES image
    buf_aces = oiio.ImageBuf(str(aces_path))

    # Apply ociodisplay (RRT + ODT transforms) with optional looks
    buf_display = oiio.ImageBufAlgo.ociodisplay(
        buf_aces,
        display=display,
        view=view,
        fromspace="",
        looks=looks or "",
        unpremult=True,
    )

    assert buf_display.initialized, f"ociodisplay conversion failed for {aces_path.name}"
    return np.asarray(buf_display.get_pixels(), dtype=np.float32)


def compare_renders(image_path, looks=None):
    """Compare output of old vs current oiio_aces_to_display."""
    print(f"\n{'='*70}")
    print(f"Testing: {Path(image_path).name}")
    print(f"Looks: {looks or 'None (baseline)'}")
    print(f"{'='*70}")
    
    # Render with old version
    result_old = aces_to_display_old(image_path, looks=looks)
    print(f"✓ Old version rendered: shape={result_old.shape}, dtype={result_old.dtype}")
    print(f"  Value range: [{result_old.min():.4f}, {result_old.max():.4f}]")
    
    # Render with current version
    result_current = aces_to_display_current(image_path, looks=looks)
    print(f"✓ Current version rendered: shape={result_current.shape}, dtype={result_current.dtype}")
    print(f"  Value range: [{result_current.min():.4f}, {result_current.max():.4f}]")
    
    # Compute differences
    diff = np.abs(result_old - result_current)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\n📊 Difference Analysis:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Pixels with diff > 1e-5: {np.sum(diff > 1e-5)}/{diff.size}")
    
    if max_diff < 1e-6:
        print(f"\n✅ PERFECT MATCH - Outputs are identical (within float32 precision)")
        match_status = "PERFECT"
    elif max_diff < 1e-4:
        print(f"\n✅ MATCH - Outputs are functionally identical (max diff << perceived)")
        match_status = "MATCH"
    else:
        print(f"\n⚠️  DIFFERENCE - Outputs differ (max diff = {max_diff})")
        match_status = "DIFFERENT"
    
    return result_old, result_current, match_status


def visualize_comparison(result_old, result_current, output_path):
    """Create visual comparison of old vs current rendering."""
    # Clip to [0, 1] for display
    old_display = np.clip(result_old, 0, 1)
    current_display = np.clip(result_current, 0, 1)
    diff = np.abs(result_old - result_current)
    diff_display = np.clip(diff, 0, 0.01) * 100  # Scale for visibility
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(old_display)
    axes[0, 0].set_title("Old Version", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(current_display)
    axes[0, 1].set_title("Current Version", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(diff_display, cmap='hot')
    axes[1, 0].set_title(f"Absolute Difference (max={diff.max():.2e})", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].hist(diff.flatten(), bins=256, log=True)
    axes[1, 1].set_title("Difference Distribution", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Absolute Difference")
    axes[1, 1].set_ylabel("Frequency (log)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n📸 Saved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Find a test image
    aces_dir = Path(__file__).parent / "dataset" / "temp" / "aces"
    test_images = sorted(list(aces_dir.glob("*.exr")))[:3]  # Test first 3
    
    print(f"\n🔍 Found {len(test_images)} ACES images in {aces_dir}")
    
    output_dir = Path(__file__).parent / "outputs" / "render_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_match = True
    for i, image_path in enumerate(test_images):
        try:
            old, current, status = compare_renders(image_path)
            
            if status != "PERFECT" and status != "MATCH":
                all_match = False
            
            # Visualize
            viz_path = output_dir / f"comparison_{i:02d}_{image_path.stem}.png"
            visualize_comparison(old, current, viz_path)
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    if all_match:
        print("✅ ALL TESTS PASSED - Old and current versions render identically!")
    else:
        print("⚠️  Some outputs differ - check visualizations")
    print(f"{'='*70}\n")
