"""Image quality checking for dataset validation and filtering.

Fast, efficient quality metrics for raw images:
- Highlight clipping detection
- Noise floor estimation
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityCheckConfig:
    """Thresholds for image quality acceptance."""
    max_highlight_clip_pct: float = 10.0      # % of pixels at max value
    max_shadow_noise_std: float = 50.0        # Std dev in shadow regions (8-bit scale)


@dataclass
class QualityCheckResult:
    """Results of quality check."""
    passed: bool
    highlight_clip_pct: float
    shadow_noise_std: float
    reason: str = ""


def check_image_quality(
    image_path: Path | str,
    max_highlight_clip_pct: float = 10.0,
    max_shadow_noise_std: float = 50.0,
    verbose: bool = False,
) -> QualityCheckResult:
    """Fast quality check for images.
    
    Args:
        image_path: Path to image file (PNG, JPEG, TIFF, EXR, etc.)
        max_highlight_clip_pct: Threshold for highlight clipping % (default: 10.0)
        max_shadow_noise_std: Threshold for shadow noise std (default: 50.0)
        verbose: Print debug info during loading
        
    Returns:
        QualityCheckResult with pass/fail and detailed metrics
    """
    # Load image as uint8
    image = load_image_as_uint8(image_path, verbose=verbose)
    
    if image is None:
        return QualityCheckResult(
            passed=False, 
            highlight_clip_pct=0, 
            shadow_noise_std=0, 
            reason="Failed to load image"
        )
    
    # Calculate metrics
    highlight_clip_pct = _calc_highlight_clipping(image)
    shadow_noise_std = _calc_shadow_noise(image)
    
    # Check thresholds
    failures = []
    if highlight_clip_pct > max_highlight_clip_pct:
        failures.append(
            f"Highlights clipped: {highlight_clip_pct:.1f}% "
            f"(max {max_highlight_clip_pct}%)"
        )
    if shadow_noise_std > max_shadow_noise_std:
        failures.append(
            f"Shadow noise too high: {shadow_noise_std:.1f} "
            f"(max {max_shadow_noise_std})"
        )
    
    return QualityCheckResult(
        passed=len(failures) == 0,
        highlight_clip_pct=highlight_clip_pct,
        shadow_noise_std=shadow_noise_std,
        reason=" | ".join(failures) if failures else "Passed all checks"
    )


def _calc_highlight_clipping(image: np.ndarray) -> float:
    """Calculate % of pixels at highlight extremes (255).
    
    Args:
        image: Image array (H, W) or (H, W, C)
        
    Returns:
        Percentage of pixels at maximum value (255)
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2)  # Convert to grayscale
    
    total_pixels = image.size
    # Count pixels at absolute maximum (255)
    clipped = np.sum(image >= 255)
    
    return 100.0 * clipped / total_pixels


def _calc_shadow_noise(image: np.ndarray, shadow_threshold: int = 30) -> float:
    """Estimate noise floor from shadow regions.
    
    Args:
        image: Image array (H, W) or (H, W, C)
        shadow_threshold: Pixel value threshold for shadow region
        
    Returns:
        Standard deviation of shadow pixels
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    shadow_pixels = image[image < shadow_threshold]
    
    if len(shadow_pixels) < 100:
        return 0.0  # Not enough shadow data
    
    return float(np.std(shadow_pixels))


def batch_quality_check(
    image_dir: Path | str,
    max_highlight_clip_pct: float = 10.0,
    max_shadow_noise_std: float = 50.0,
    pattern: str = "*.CR2",
    max_images: Optional[int] = None,
) -> tuple[list[dict], int]:
    """Run quality checks on all images in a directory.
    
    Args:
        image_dir: Directory containing images
        max_highlight_clip_pct: Threshold for highlight clipping % (default: 10.0)
        max_shadow_noise_std: Threshold for shadow noise std (default: 50.0)
        pattern: File pattern to match (default: "*.CR2" for raw files)
        max_images: Maximum number of images to process. If None, process all images.
        
    Returns:
        Tuple of (results list, failed count)
    """
    image_dir = Path(image_dir)
    images = sorted([f for f in image_dir.glob(pattern)])
    if max_images is not None:
        images = images[:max_images]
    
    limit_text = f" (limited to {max_images})" if max_images is not None else ""
    print(f"Found {len(images)} images in {image_dir.name}{limit_text}")
    
    results = []
    failed_count = 0
    
    for idx, img_path in enumerate(images, 1):
        result = check_image_quality(img_path, max_highlight_clip_pct, max_shadow_noise_std)
        results.append({
            'filename': img_path.name,
            'passed': result.passed,
            'highlight_clip_pct': result.highlight_clip_pct,
            'shadow_noise_std': result.shadow_noise_std,
            'reason': result.reason,
        })
        status = '✓ PASS' if result.passed else '✗ FAIL'
        print(f"[{idx}/{len(images)}] {img_path.name}: {status}")
        
        if not result.passed:
            failed_count += 1
    
    return results, failed_count


def load_image_as_uint8(path: Path | str, verbose: bool = False) -> Optional[np.ndarray]:
    """Load image and convert to uint8 (0-255) using OpenImageIO.
    
    Args:
        path: Path to image file.
        verbose: Print debug info during loading.
        
    Returns:
        Numpy array [H, W, 3] as uint8, or None if loading fails.
    """
    import OpenImageIO as oiio
    
    buf = oiio.ImageBuf(str(path))
    if not buf.initialized:
        if verbose:
            print(f"  [OpenImageIO FAILED] Could not open {path}")
        return None

    # Get pixels as float32 for normalization
    img_array = buf.get_pixels(oiio.TypeFloat)
    assert img_array is not None, f"Failed to get pixels from {path}"
    
    if verbose:
        print(f"  [OpenImageIO] shape={img_array.shape}, dtype={img_array.dtype}")

    # Ensure RGB
    if img_array.shape[2] > 3:
        img_array = img_array[:, :, :3]
    elif img_array.shape[2] == 1:
        img_array = np.stack([img_array.squeeze(2)] * 3, axis=2)

    # Normalize to 0-255
    img_min, img_max = img_array.min(), img_array.max()
    
    if img_max > img_min:
        # If it's already in [0, 1] (typical for tone-mapped/standard images)
        if img_max <= 1.0:
            normalized = (img_array * 255).astype(np.uint8)
        else:
            # HDR normalization: scale full range to 0-255
            normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = img_array.astype(np.uint8)

    if verbose:
        print(f"  [OpenImageIO SUCCESS]")
        
    return normalized
