"""Utility functions to generate synthetic test images for dequantization inference."""

from __future__ import annotations

import numpy as np
import torch



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
    width: int = 128,
    height: int = 21,
    block_width: int = 8,
    dtype: str = "float32",
) -> np.ndarray:
    """Create smooth continuous color gradients with all three primaries (8-bit quantized range).
    
    Creates: Red primary + Green primary + Blue primary + white separator
    Smooth gradients using range based on block_width (e.g., 8 → 120-135, 4 → 124-132, 2 → 126-130).
    With primary at 255 (saturated).
    Total height: 21 + 21 + 21 + 1 = 64 pixels
    
    Args:
        width: Image width in pixels (default 128)
        height: Height of each primary row in pixels (default 21)
        block_width: Block width determining color range (8, 4, or 2)
        dtype: Output data type ("uint8" for [0-255], "float32" for [0-1])
    
    Returns:
        Image with all three primaries stacked [3, 64, width]
    """
    # Get color range based on block_width
    color_min, color_max = get_gradient_range_from_block_width(width, block_width)
    
    # Create smooth gradient ramp across the range
    color_ramp = np.linspace(color_min, color_max, width, dtype=np.float32)  # [width]
    
    def create_primary_row(primary_idx: int) -> np.ndarray:
        """Create a row with smooth color gradient for one primary channel."""
        primary_max = 1.0  # 255 in float32
        row = np.ones((height, width, 3), dtype=np.float32)
        
        for j in range(3):
            if j == primary_idx:
                row[:, :, j] = primary_max  # Primary channel at 255
            else:
                row[:, :, j] = color_ramp  # Other channels: smooth gradient
        
        return row  # [height, width, 3]
    
    # Create rows for each primary: R=0, G=1, B=2
    row1 = create_primary_row(0)  # Red primary (R=255, G/B smooth gradient)
    row2 = create_primary_row(1)  # Green primary (G=255, R/B smooth gradient)
    row3 = create_primary_row(2)  # Blue primary (B=255, R/G smooth gradient)
    white_sep = np.ones((1, width, 3), dtype=np.float32)  # White separator [1, width, 3]
    
    # Stack vertically: [height*3+1, width, 3]
    image = np.concatenate([row1, row2, row3, white_sep], axis=0)
    
    # Convert to requested format
    if dtype == "uint8":
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    elif dtype == "float32":
        image = np.clip(image, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return np.transpose(image, (2, 0, 1))  # [3, height*3+1, width]






def combine_primary_gradients(
    width: int = 128,
    height: int = 21,
    dtype: str = "float32",
) -> np.ndarray:
    """Combine all three gradient variants (8x21, 4x21, 2x21) vertically.
    
    Stacks three complete gradient images (each with all three colors + separator):
    - block_width=8: 16 colors per primary (64px tall: 21+21+21+1)
    - block_width=4: 32 colors per primary (64px tall: 21+21+21+1)
    - block_width=2: 64 colors per primary (64px tall: 21+21+21+1)
    
    Total height: 64*3 = 192 pixels (showing different gradient widths in all three colors)
    
    Args:
        width: Image width (default 128)
        height: Height of each primary row within each variant (default 21)
        dtype: Output data type ("uint8" for [0-255], "float32" for [0-1])
    
    Returns:
        Combined image with all three variants stacked [3, 64*3, width]
    """
    # Generate the three variants (each is [3, 64, width])
    gradient_8 = create_primary_gradients(width=width, height=height, block_width=8, dtype="float32")
    gradient_4 = create_primary_gradients(width=width, height=height, block_width=4, dtype="float32")
    gradient_2 = create_primary_gradients(width=width, height=height, block_width=2, dtype="float32")
    
    # Stack vertically: [3, 64*3, width]
    combined = np.concatenate([gradient_8, gradient_4, gradient_2], axis=1)
    
    # Convert to requested format
    if dtype == "uint8":
        combined = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
    elif dtype == "float32":
        combined = np.clip(combined, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return combined


def create_reference_gradients(
    width: int = 128,
    height: int = 21,
    block_width: int = 8,
    dtype: str = "float32",
) -> np.ndarray:
    """Create smooth reference gradient (same 120-135 range, normalized).
    
    Reference shows the same 120-135 range as the 8-bit input, but as a smooth
    continuous gradient instead of discrete values.
    
    Args:
        width: Image width in pixels (default 128)
        height: Height of each primary row in pixels (default 21)
        block_width: Not used (kept for API compatibility)
        dtype: Output data type ("uint8" for [0-255], "float32" for [0-1])
    
    Returns:
        Reference image [3, 64, width]
    """
    # Reference uses same range as input (120-135) but smooth
    color_min,color_max = get_gradient_range_from_block_width(width, block_width)

    color_ramp = np.linspace(color_min, color_max, width, dtype=np.float32)
    
    def create_reference_row(primary_idx: int) -> np.ndarray:
        """Create a reference row with smooth color gradient."""
        primary_max = 1.0
        row = np.ones((height, width, 3), dtype=np.float32)
        
        for j in range(3):
            if j == primary_idx:
                row[:, :, j] = primary_max
            else:
                row[:, :, j] = color_ramp
        
        return row
    
    row1 = create_reference_row(0)
    row2 = create_reference_row(1)
    row3 = create_reference_row(2)
    white_sep = np.ones((1, width, 3), dtype=np.float32)
    
    image = np.concatenate([row1, row2, row3, white_sep], axis=0)
    
    if dtype == "uint8":
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    elif dtype == "float32":
        image = np.clip(image, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return np.transpose(image, (2, 0, 1))


def combine_reference_gradients(
    width: int = 128,
    height: int = 21,
    dtype: str = "float32",
) -> np.ndarray:
    """Combine all three reference gradient variants (8x21, 4x21, 2x21) vertically.
    
    Stacks three reference images with full 0-255 ranges for comparison.
    
    Args:
        width: Image width (default 128)
        height: Height of each primary row within each variant (default 21)
        dtype: Output data type ("uint8" for [0-255], "float32" for [0-1])
    
    Returns:
        Combined reference image [3, 64*3, width]
    """
    gradient_block_8 = create_reference_gradients(width=width, height=height, block_width=8, dtype="float32")
    gradient_block_4 = create_reference_gradients(width=width, height=height, block_width=4, dtype="float32")
    gradient_block_2 = create_reference_gradients(width=width, height=height, block_width=2, dtype="float32")
    
    combined = np.concatenate([gradient_block_8, gradient_block_4, gradient_block_2], axis=1)
    
    if dtype == "uint8":
        combined = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
    elif dtype == "float32":
        combined = np.clip(combined, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return combined




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


def apply_s_curve_contrast(image: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Apply S-curve contrast to amplify differences without clipping.
    
    Uses a parametric S-curve: S(x) = x^k / (x^k + (1-x)^k)
    This expands mid-tones and compresses extremes, making subtle differences visible.
    
    Args:
        image: Input image in [0, 1] range (numpy array)
        strength: Curve steepness (higher = more aggressive). Typical range 1.0-4.0
    
    Returns:
        Contrast-enhanced image in [0, 1] range
    """
    # Clip to valid range
    x = np.clip(image, 0, 1)
    
    # Avoid division by zero at boundaries
    epsilon = 1e-7
    x_safe = np.clip(x, epsilon, 1 - epsilon)
    
    # S-curve: x^k / (x^k + (1-x)^k)
    # This expands mid-tones (0.5) and compresses extremes (0, 1)
    x_power = np.power(x_safe, strength)
    one_minus_x_power = np.power(1 - x_safe, strength)
    s_curve = x_power / (x_power + one_minus_x_power)
    
    # Smooth boundaries
    result = np.where(x < epsilon, 0, np.where(x > 1 - epsilon, 1, s_curve))
    
    return result


def apply_s_curve_contrast_torch(image: torch.Tensor, strength: float = 2.0) -> torch.Tensor:
    """Apply S-curve contrast using PyTorch (GPU-friendly, memory efficient).
    
    Uses a parametric S-curve: S(x) = x^k / (x^k + (1-x)^k)
    Memory efficient - operates in-place where possible.
    
    Args:
        image: Input tensor in [0, 1] range
        strength: Curve steepness (higher = more aggressive)
    
    Returns:
        Contrast-enhanced tensor in [0, 1] range
    """
    # Clip to valid range
    x = torch.clamp(image, 0, 1)
    
    # Avoid division by zero at boundaries
    epsilon = 1e-7
    x_safe = torch.clamp(x, epsilon, 1 - epsilon)
    
    # S-curve: x^k / (x^k + (1-x)^k)
    x_power = torch.pow(x_safe, strength)
    one_minus_x_power = torch.pow(1 - x_safe, strength)
    s_curve = x_power / (x_power + one_minus_x_power)
    
    # Smooth boundaries
    result = torch.where(x < epsilon, torch.zeros_like(x), 
                        torch.where(x > 1 - epsilon, torch.ones_like(x), s_curve))
    
    return result

def get_gradient_range_from_block_width(img_width: int, block_width: int):
    """Calculate normalized color range from block width.
    
    Args:
        img_width: Width of the image in pixels
        block_width: Width of each block in pixels (e.g., 8, 4, 2)
    
    Returns:
        Tuple of (color_min, color_max) normalized to [0, 1] range
        Based on 8-bit centered around 128 ± block_width/2
    """
    range = img_width / block_width / 2
    norm = 256 / 2  # 128 (center of 8-bit range)
    start = norm - range / 2  # e.g., 128 - 4 = 124
    end = norm + range / 2    # e.g., 128 + 4 = 132
    
    # Normalize to [0, 1] range
    return start / 255.0, end / 255.0
