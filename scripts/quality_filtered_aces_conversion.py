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
    --dataset-prefix STR    Prefix for output filenames (e.g., PPR10K_, MIT-Adobe_5K_)
    --help                  Show this help message
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from datetime import datetime
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
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g., PPR10K_, MIT-Adobe_5K_)",
    )
    parser.add_argument(
        "--summary-log",
        type=str,
        default=None,
        help="Append-only summary .log file path (default: <output-dir>/../quality_summary_<prefix>.log)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    raw_dir = Path(args.input_dir)
    aces_dir = Path(args.output_dir)
    if args.summary_log is not None:
        summary_log = Path(args.summary_log)
    else:
        prefix_for_log = args.dataset_prefix.rstrip("_") or "dataset"
        summary_log = aces_dir.parent / f"quality_summary_{prefix_for_log}.log"
    
    # Validate input directory
    if not raw_dir.exists():
        print(f"❌ ERROR: Input directory not found: {raw_dir}")
        return 1
    
    # Create output directory
    aces_dir.mkdir(parents=True, exist_ok=True)
    summary_log.parent.mkdir(parents=True, exist_ok=True)
    
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
    converted_count = 0
    skipped_count = 0
    quality_failed_count = 0
    conversion_failed_count = 0
    failed_clip_count = 0
    failed_noise_count = 0
    failed_both_count = 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset_prefix.rstrip("_") or "dataset"
    
    for idx, image_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {image_path.name}...", end=" ", flush=True)
        
        # Check if already processed
        output_file = aces_dir / f"{args.dataset_prefix}{image_path.stem}.exr"
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
            if _convert_to_aces(image_path, aces_dir, args.dataset_prefix):
                print("✓ Done")
                converted_count += 1
            else:
                print("✗ Failed")
                conversion_failed_count += 1
        else:
            print(f"✗ FAIL ({result.highlight_clip_pct:.2f}% clip, {result.shadow_noise_std:.2f} noise | {result.reason})")
            quality_failed_count += 1
            clip_failed = result.highlight_clip_pct > args.highlight_clip
            noise_failed = result.shadow_noise_std > args.noise_floor
            if clip_failed and noise_failed:
                failure_category = "quality_both"
                failed_both_count += 1
            elif clip_failed:
                failure_category = "quality_clipping"
                failed_clip_count += 1
            else:
                failure_category = "quality_noise"
                failed_noise_count += 1
    
    # Summary
    processed_total = len(images) - skipped_count
    print(f"\n" + "=" * 70)
    print(f"📊 Summary:")
    print(f"  ⊘ Already processed:       {skipped_count}/{len(images)}")
    print(f"  ✓ Passed quality check:    {passed_count}/{processed_total}")
    print(f"  ✗ Failed quality check:    {quality_failed_count}/{processed_total}")
    print(f"    • Clipping only:         {failed_clip_count}")
    print(f"    • Noise only:            {failed_noise_count}")
    print(f"    • Both clip+noise:       {failed_both_count}")
    print(f"  ✓ Successfully converted:  {converted_count}/{passed_count}")
    print(f"  ✗ Conversion failed:       {conversion_failed_count}/{passed_count}")
    print(f"  📝 Summary log:            {summary_log}")
    print(f"\n📁 Output directory: {aces_dir}")
    print("=" * 70)

    _append_summary_log(
        summary_log=summary_log,
        run_id=run_id,
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        aces_dir=aces_dir,
        threshold_clip=args.highlight_clip,
        threshold_noise=args.noise_floor,
        total_images=len(images),
        processed_total=processed_total,
        skipped_count=skipped_count,
        passed_count=passed_count,
        quality_failed_count=quality_failed_count,
        failed_clip_count=failed_clip_count,
        failed_noise_count=failed_noise_count,
        failed_both_count=failed_both_count,
        converted_count=converted_count,
        conversion_failed_count=conversion_failed_count,
    )

    # Quality failures are expected filtering outcomes; conversion failures are pipeline errors.
    return 0 if conversion_failed_count == 0 else 1


def _append_summary_log(
    summary_log: Path,
    run_id: str,
    dataset_name: str,
    raw_dir: Path,
    aces_dir: Path,
    threshold_clip: float,
    threshold_noise: float,
    total_images: int,
    processed_total: int,
    skipped_count: int,
    passed_count: int,
    quality_failed_count: int,
    failed_clip_count: int,
    failed_noise_count: int,
    failed_both_count: int,
    converted_count: int,
    conversion_failed_count: int,
) -> None:
    """Append one run summary block to a .log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with summary_log.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Input directory: {raw_dir.resolve()}\n")
        f.write(f"Output directory: {aces_dir.resolve()}\n")
        f.write(f"Thresholds: highlight_clip <= {threshold_clip}%, noise_floor <= {threshold_noise}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total images considered: {total_images}\n")
        f.write(f"Already processed (skipped): {skipped_count}\n")
        f.write(f"Processed this run: {processed_total}\n")
        f.write(f"Passed quality check: {passed_count}\n")
        f.write(f"Failed quality check: {quality_failed_count}\n")
        f.write(f"  - Failed due to clipping only: {failed_clip_count}\n")
        f.write(f"  - Failed due to noise only: {failed_noise_count}\n")
        f.write(f"  - Failed due to both: {failed_both_count}\n")
        f.write(f"Successfully converted: {converted_count}\n")
        f.write(f"Conversion failures: {conversion_failed_count}\n")
        f.write("=" * 80 + "\n")


def _convert_to_aces(image_path: Path, output_dir: Path, prefix: str = "") -> bool:
    """Convert a single RAW image to ACES format using rawtoaces.
    
    Args:
        image_path: Path to RAW image file
        output_dir: Output directory for ACES file
        prefix: Optional prefix for the output filename
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        image_path = image_path.resolve()
        output_dir = output_dir.resolve()
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
            aces_file_clean = output_dir / f"{prefix}{image_path.stem}.exr"
            if aces_file_with_suffix.exists():
                aces_file_with_suffix.rename(aces_file_clean)
            return True
        else:
            # Print error details for debugging
            if result.stderr:
                import sys
                print(f"\n❌ rawtoaces error: {result.stderr[:200]}", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        import sys
        print(f"\n❌ rawtoaces not found in PATH", file=sys.stderr)
        return False
    except Exception as e:
        import sys
        print(f"\n❌ Conversion error: {str(e)[:200]}", file=sys.stderr)
        return False


if __name__ == "__main__":
    sys.exit(main())
