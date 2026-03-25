#!/usr/bin/env python3
"""Quality-filtered ACES conversion script.

Runs quality checks on RAW images and converts passing images to ACES format.
Images that pass quality checks are converted using rawtoaces.

Usage:
    python quality_filtered_aces_conversion.py [OPTIONS]
    
Options:
    --max-images N           Max images to process (default: all)
    --highlight-clip PCT    Max highlight clipping % (default: 3.0)
    --noise-floor STD       Max shadow noise std (default: 15.0)
    --input-dir DIR         Input directory for RAW images (default: dataset/temp/raw)
    --output-dir DIR        Output directory for ACES files (default: dataset/temp/aces)
    --help                  Show this help message
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add src to path for imports
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from luminascale.utils.quality_check import check_image_quality


def main() -> int:
    """Main entry point."""
    script_dir = Path(__file__).parent.absolute()
    src_dir = script_dir.parent / "src"
    sys.path.insert(0, str(src_dir))
    
    parser = argparse.ArgumentParser(
        description="Quality-filtered ACES conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)",
    )
    parser.add_argument(
        "--highlight-clip",
        type=float,
        default=3.0,
        help="Max highlight clipping %% (default: 3.0)",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=15.0,
        help="Max shadow noise std (default: 15.0)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(script_dir.parent / "dataset" / "temp" / "raw"),
        help="Input directory for RAW images (default: dataset/temp/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir.parent / "dataset" / "temp" / "aces"),
        help="Output directory for ACES files (default: dataset/temp/aces)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    raw_dir = Path(args.input_dir)
    aces_dir = Path(args.output_dir)
    
    # Validate input directory
    if not raw_dir.exists():
        print(f"❌ ERROR: Input directory not found: {raw_dir}")
        return 1
    
    # Create output directory
    aces_dir.mkdir(parents=True, exist_ok=True)
    
    # Find RAW images
    raw_extensions = {".cr2", ".crw", ".nef", ".nrw", ".arw", ".raf", ".dng", ".raw"}
    images = sorted(
        [f for f in raw_dir.iterdir() 
         if f.is_file() and f.suffix.lower() in raw_extensions]
    )
    
    if not images:
        print(f"❌ ERROR: No RAW images found in {raw_dir}")
        return 1
    
    # Limit images if specified
    if args.max_images is not None:
        images = images[:args.max_images]
    
    limit_text = f" (limited to {args.max_images})" if args.max_images is not None else ""
    print(f"\n📷 Found {len(images)} RAW images{limit_text}")
    print(f"📋 Quality thresholds: highlight_clip={args.highlight_clip}%, "
          f"noise_floor={args.noise_floor}\n")
    
    # Check if imageio is available for accurate RAW quality checking
    try:
        import imageio
        print("✓ Using imageio for raw sensor data extraction\n")
    except ImportError:
        print("⚠ WARNING: imageio not found - using OpenImageIO fallback")
        print("  This may underestimate highlight clipping (tone-mapped data)")
        print("  To use imageio, run: pixi run python scripts/quality_filtered_aces_conversion.py\n")
    
    passed_count = 0
    failed_count = 0
    converted_count = 0
    skipped_count = 0
    
    for idx, image_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {image_path.name}...", end=" ", flush=True)
        
        # Check if already processed
        output_file = aces_dir / f"{image_path.stem}.exr"
        if output_file.exists():
            print("⊘ Already processed")
            skipped_count += 1
            continue
        
        # Run quality check (verbose for first image)
        result = check_image_quality(
            image_path,
            max_highlight_clip_pct=args.highlight_clip,
            max_shadow_noise_std=args.noise_floor,
            verbose=(idx == 1),
        )
        
        if result.passed:
            print(f"✓ PASS ({result.highlight_clip_pct:.2f}% clip, {result.shadow_noise_std:.2f} noise) ", end="", flush=True)
            passed_count += 1
            
            # Convert to ACES
            print("→ Converting...", end=" ", flush=True)
            if _convert_to_aces(image_path, aces_dir):
                print("✓ Done")
                converted_count += 1
            else:
                print("✗ Failed")
                failed_count += 1
        else:
            print(f"✗ FAIL ({result.highlight_clip_pct:.2f}% clip, {result.shadow_noise_std:.2f} noise | {result.reason})")
            failed_count += 1
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"📊 Summary:")
    print(f"  ⊘ Already processed:       {skipped_count}/{len(images)}")
    print(f"  ✓ Passed quality check:    {passed_count}/{len(images) - skipped_count}")
    print(f"  ✗ Failed quality check:    {failed_count}/{len(images) - skipped_count}")
    print(f"  ✓ Successfully converted:  {converted_count}/{passed_count} "
          f"(of passing images)")
    print(f"\n📁 Output directory: {aces_dir}")
    print("=" * 70)
    
    return 0 if failed_count == 0 else 1


def _convert_to_aces(image_path: Path, output_dir: Path) -> bool:
    """Convert a single RAW image to ACES format using rawtoaces.
    
    Args:
        image_path: Path to RAW image file
        output_dir: Output directory for ACES file
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        cmd = [
            "rawtoaces",
            "--data-dir", "/usr/local/share/rawtoaces/data",
            "--output-dir", str(output_dir),
            "--create-dirs",
            "--overwrite",
            str(image_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode == 0:
            # rawtoaces adds "_aces" suffix by default; rename to remove it
            aces_file_with_suffix = output_dir / f"{image_path.stem}_aces.exr"
            aces_file_clean = output_dir / f"{image_path.stem}.exr"
            if aces_file_with_suffix.exists():
                aces_file_with_suffix.rename(aces_file_clean)
            return True
        return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
