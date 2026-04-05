#!/usr/bin/env python3
"""Benchmark PyTorch vs OCIO on diverse set of images to check for systematic saturation issues."""

import sys
from pathlib import Path
import logging
from typing import Optional

import numpy as np
import torch

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import from benchmark_pytorch_vs_ocio
import importlib.util
spec = importlib.util.spec_from_file_location("benchmark_pytorch_vs_ocio", Path(__file__).parent / "benchmark_pytorch_vs_ocio.py")
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)
benchmark_image = benchmark_module.benchmark_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_diverse_test_images(aces_dir: Path, num_images: int = 20) -> list[Path]:
    """Select diverse test images from different scenes.
    
    Args:
        aces_dir: Directory containing ACES images
        num_images: Total number of images to select
        
    Returns:
        List of diverse image paths
    """
    # Group images by scene number (1000, 1001, 1002, etc.)
    images_by_scene = {}
    for image_path in aces_dir.glob("*.exr"):
        scene_num = image_path.stem.split('_')[0]
        if scene_num not in images_by_scene:
            images_by_scene[scene_num] = []
        images_by_scene[scene_num].append(image_path)
    
    logger.info(f"Found {len(images_by_scene)} different scenes")
    
    # Select one image per scene to get diversity
    selected = []
    scenes = sorted(images_by_scene.keys())
    
    # Take evenly spaced scenes
    step = max(1, len(scenes) // num_images)
    for i, scene_num in enumerate(scenes[::step]):
        if len(selected) >= num_images:
            break
        # Take first image from each scene
        selected.append(images_by_scene[scene_num][0])
    
    # If we need more, add additional frames from early scenes
    if len(selected) < num_images:
        for scene_num in sorted(images_by_scene.keys())[:3]:
            for img in images_by_scene[scene_num][1:]:
                if len(selected) >= num_images:
                    break
                selected.append(img)
    
    return sorted(selected[:num_images])


def main():
    """Run benchmark on diverse ACES images."""
    # Find test images
    aces_dir = Path(__file__).parent.parent / "dataset" / "temp" / "aces"
    
    if not aces_dir.exists():
        logger.error(f"ACES directory not found: {aces_dir}")
        sys.exit(1)
    
    test_images = get_diverse_test_images(aces_dir, num_images=20)
    
    if not test_images:
        logger.error(f"No test images found in {aces_dir}")
        sys.exit(1)
    
    logger.info(f"Testing {len(test_images)} diverse images")
    logger.info(f"Scene range: {test_images[0].stem.split('_')[0]} - {test_images[-1].stem.split('_')[0]}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Run benchmarks
    results = []
    for i, image_path in enumerate(test_images, 1):
        try:
            logger.info(f"\n[{i}/{len(test_images)}] Testing {image_path.name}...")
            result = benchmark_image(image_path, device=device, num_runs=3)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary statistics
    if results:
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)
        
        psnrs = [r["psnr"] for r in results]
        ssims = [r["ssim"] for r in results]
        delta_es = [r["delta_e"] for r in results]
        speedups = [r["speedup"] for r in results]
        
        logger.info(f"PSNR:    {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB (range: {np.min(psnrs):.2f}-{np.max(psnrs):.2f})")
        logger.info(f"SSIM:    {np.mean(ssims):.4f} ± {np.std(ssims):.4f} (range: {np.min(ssims):.4f}-{np.max(ssims):.4f})")
        logger.info(f"ΔE:      {np.mean(delta_es):.4f} ± {np.std(delta_es):.4f} (range: {np.min(delta_es):.4f}-{np.max(delta_es):.4f})")
        logger.info(f"Speedup: {np.mean(speedups):.2f}× ± {np.std(speedups):.2f}× (range: {np.min(speedups):.2f}×-{np.max(speedups):.2f}×)")
        
        logger.info("\nDetailed Results:")
        logger.info("-" * 100)
        logger.info(f"{'Image':<20} {'PSNR (dB)':<12} {'SSIM':<10} {'ΔE':<10} {'Speedup':<10}")
        logger.info("-" * 100)
        for result in results:
            logger.info(
                f"{result['image']:<20} {result['psnr']:>10.2f}  {result['ssim']:>8.4f}  "
                f"{result['delta_e']:>8.4f}  {result['speedup']:>8.2f}×"
            )
        logger.info("-" * 100)
        
        logger.info("\n✅ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
