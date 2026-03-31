"""Utility functions to generate synthetic test images for dequantization inference."""

from __future__ import annotations

import numpy as np
import torch


def create_sky_gradient(
    width: int = 512,
    height: int = 512,
    dtype: str = "uint8",
) -> np.ndarray:
    """Create a sky-like gradient image from dark blue (bottom) to light blue (top).
    
    This gradient is useful for visualizing 8-bit quantization artifacts:
    - When saturated, banding patterns reveal quantization levels
    - Good dequantization removes these artifacts
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        dtype: Output data type ("uint8" for [0-255], "float32" for [0-1])
    
    Returns:
        Sky gradient image [3, height, width] or [height, width, 3] depending on format
    """
    # Create vertical gradient (top = light, bottom = dark)
    # Normalize from 0 to 1
    y_gradient = np.linspace(1.0, 0.3, height).reshape(-1, 1)
    gradient = np.repeat(y_gradient, width, axis=1)  # [height, width]
    
    # Create RGB image with blue sky color
    # Dark blue: (0.1, 0.3, 0.6) → Light blue: (0.4, 0.6, 0.9)
    r = np.linspace(0.48, 0.52, height).reshape(-1, 1)
    g = np.linspace(0.48, 0.52, height).reshape(-1, 1)
    b = np.linspace(1, 1, height).reshape(-1, 1)
    
    r = np.repeat(r, width, axis=1)
    g = np.repeat(g, width, axis=1)
    b = np.repeat(b, width, axis=1)
    
    # Stack into RGB image [height, width, 3]
    sky_rgb = np.stack([r, g, b], axis=2)
    
    # Convert to requested format
    if dtype == "uint8":
        # Convert to 8-bit: [0, 255]
        sky_rgb = (np.clip(sky_rgb, 0, 1) * 255).astype(np.uint8)
    elif dtype == "float32":
        # Ensure float32: [0, 1]
        sky_rgb = np.clip(sky_rgb, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Return as [3, height, width] for PyTorch compatibility
    return np.transpose(sky_rgb, (2, 0, 1))


def create_gradient_image(
    width: int = 512,
    height: int = 512,
    color_start: tuple[float, float, float] = (0.1, 0.3, 0.6),
    color_end: tuple[float, float, float] = (0.4, 0.6, 0.9),
    dtype: str = "float32",
) -> np.ndarray:
    """Create a smooth vertical gradient between two colors.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        color_start: RGB color at bottom (range [0, 1])
        color_end: RGB color at top (range [0, 1])
        dtype: Output data type ("uint8" for [0-255], "float32" for [0-1])
    
    Returns:
        Gradient image [3, height, width]
    """
    # Create vertical interpolation factor [0, 1] from bottom to top
    t = np.linspace(0, 1, height).reshape(-1, 1)
    t = np.repeat(t, width, axis=1)  # [height, width]
    
    # Interpolate colors
    r = color_start[0] * (1 - t) + color_end[0] * t
    g = color_start[1] * (1 - t) + color_end[1] * t
    b = color_start[2] * (1 - t) + color_end[2] * t
    
    # Stack into RGB [height, width, 3]
    gradient = np.stack([r, g, b], axis=2)
    
    # Convert to requested format
    if dtype == "uint8":
        gradient = (np.clip(gradient, 0, 1) * 255).astype(np.uint8)
    elif dtype == "float32":
        gradient = np.clip(gradient, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Return as [3, height, width]
    return np.transpose(gradient, (2, 0, 1))


def create_primary_gradients(
    width: int = 512,
    height: int = 512,
    dtype: str = "float32",
) -> np.ndarray:
    """Create an image with three vertical gradient columns: Red, Green, Blue primaries.
    
    Each column shows a gradient from dark to light in one primary color:
    - Left: Red primary (high R, low G/B)
    - Middle: Green primary (high G, low R/B)
    - Right: Blue primary (high B, low R/G)
    
    Useful for visualizing quantization artifacts in each channel separately.
    
    Args:
        width: Image width in pixels (will be divided into 3 columns)
        height: Image height in pixels
        dtype: Output data type ("uint8" for [0-255], "float32" for [0-1])
    
    Returns:
        Primary gradient image [3, height, width]
    """
    # Ensure equal column widths by distributing remainder
    col_width = width // 3
    remainder = width % 3
    col_widths = [col_width + (1 if i < remainder else 0) for i in range(3)]
    
    # Create vertical gradient for the other two channels (0.48 → 0.52)
    gradient = np.linspace(0.48, 0.52, height).reshape(-1, 1)
    
    columns = []
    
    # Red column: R=1 (full), G and B vary 0.48→0.52
    red_gradient = np.repeat(gradient, col_widths[0], axis=1)
    red_col = np.stack([
        np.ones_like(red_gradient),  # R = 1.0
        red_gradient,                  # G varies
        red_gradient                   # B varies
    ], axis=2)
    columns.append(red_col)
    
    # Green column: G=1 (full), R and B vary 0.48→0.52
    green_gradient = np.repeat(gradient, col_widths[1], axis=1)
    green_col = np.stack([
        green_gradient,                # R varies
        np.ones_like(green_gradient),  # G = 1.0
        green_gradient                 # B varies
    ], axis=2)
    columns.append(green_col)
    
    # Blue column: B=1 (full), R and G vary 0.48→0.52
    blue_gradient = np.repeat(gradient, col_widths[2], axis=1)
    blue_col = np.stack([
        blue_gradient,                 # R varies
        blue_gradient,                 # G varies
        np.ones_like(blue_gradient)    # B = 1.0
    ], axis=2)
    columns.append(blue_col)
    
    # Concatenate horizontally: [height, width, 3]
    image = np.concatenate(columns, axis=1)
    
    # Convert to requested format
    if dtype == "uint8":
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    elif dtype == "float32":
        image = np.clip(image, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Return as [3, height, width] for PyTorch compatibility
    return np.transpose(image, (2, 0, 1))


def quantize_to_8bit(image: np.ndarray) -> np.ndarray:
    """Simulate 8-bit quantization by converting to uint8 and back to float32.
    
    This introduces quantization artifacts visible in saturated views.
    
    Args:
        image: Input image in [0, 1] range as float32
    
    Returns:
        Quantized image in [0, 1] range as float32
    """
    # Simulate 8-bit quantization
    quantized = ((np.clip(image, 0, 1) * 255).astype(np.uint8).astype(np.float32)) / 255.0
    return quantized
