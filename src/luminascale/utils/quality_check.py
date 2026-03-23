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
    config: Optional[QualityCheckConfig] = None,
) -> QualityCheckResult:
    """Fast quality check for images.
    
    Args:
        image_path: Path to image file (PNG, JPEG, TIFF, EXR, etc.)
        config: Quality thresholds (uses defaults if None)
        
    Returns:
        QualityCheckResult with pass/fail and detailed metrics
    """
    if config is None:
        config = QualityCheckConfig()
    
    # Load image as uint8
    image = load_image_as_uint8(image_path)
    
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
    if highlight_clip_pct > config.max_highlight_clip_pct:
        failures.append(
            f"Highlights clipped: {highlight_clip_pct:.1f}% "
            f"(max {config.max_highlight_clip_pct}%)"
        )
    if shadow_noise_std > config.max_shadow_noise_std:
        failures.append(
            f"Shadow noise too high: {shadow_noise_std:.1f} "
            f"(max {config.max_shadow_noise_std})"
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


def load_image_as_uint8(path: Path | str) -> Optional[np.ndarray]:
    """Load RAW camera image and convert to uint8 (0-255).
    
    Supports Canon CR2, Nikon NEF, Sony ARW, and other RAW formats via OpenImageIO.
    
    Args:
        path: Path to RAW image file (CR2, NEF, ARW, etc.)
        
    Returns:
        Image as uint8 array (H, W, 3) or None if failed
    """
    path_str = str(path).lower()
    
    try:
        # Handle RAW camera files (CR2, NEF, ARW, etc.)
        raw_extensions = {'.cr2', '.crw', '.nef', '.nrw', '.arw', '.raf', '.dng', '.raw'}
        if any(path_str.endswith(ext) for ext in raw_extensions):
            try:
                # Try OpenImageIO first for native RAW support
                import OpenImageIO as oiio
                inp = oiio.ImageInput.open(str(path))
                if inp:
                    image_spec = inp.spec()
                    img_array = inp.read_image()
                    inp.close()
                    
                    # img_array is a flat array, reshape to HWC
                    h, w, c = image_spec.height, image_spec.width, image_spec.nchannels
                    img_array = img_array.reshape(h, w, c)
                    
                    # Normalize to 0-255
                    if img_array.dtype in [np.float32, np.float64]:
                        img_min, img_max = img_array.min(), img_array.max()
                        if img_max > img_min:
                            normalized = (
                                (img_array - img_min) / (img_max - img_min) * 255
                            ).astype(np.uint8)
                        else:
                            normalized = img_array.astype(np.uint8)
                    else:
                        # Already integer - convert to 8-bit
                        img_min, img_max = img_array.min(), img_array.max()
                        if img_max > img_min:
                            normalized = (
                                (img_array - img_min) / (img_max - img_min) * 255
                            ).astype(np.uint8)
                        else:
                            normalized = img_array.astype(np.uint8)
                    
                    # Handle channels
                    if normalized.ndim == 3:
                        if normalized.shape[2] > 3:
                            normalized = normalized[:, :, :3]  # Take RGB channels
                    elif normalized.ndim == 2:
                        normalized = np.stack([normalized] * 3, axis=2)
                    
                    return normalized
            except Exception:
                pass
            
            # Fallback: Try using imageio with libraw/rawpy
            try:
                import imageio
                img_array = imageio.imread(str(path))
                
                # Normalize to 0-255
                if img_array.dtype in [np.float32, np.float64]:
                    img_min, img_max = img_array.min(), img_array.max()
                    if img_max > img_min:
                        normalized = (
                            (img_array - img_min) / (img_max - img_min) * 255
                        ).astype(np.uint8)
                    else:
                        normalized = img_array.astype(np.uint8)
                else:
                    normalized = img_array.astype(np.uint8)
                
                # Handle channels  
                if normalized.ndim == 3 and normalized.shape[2] > 3:
                    normalized = normalized[:, :, :3]
                elif normalized.ndim == 2:
                    normalized = np.stack([normalized] * 3, axis=2)
                
                return normalized
            except Exception:
                pass
            
            return None
        
        # Only RAW files are supported
        return None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
