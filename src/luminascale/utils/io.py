"""Utility helpers for image file I/O and preprocessing.

Stage 0 of the pipeline is non‑ML; the functions here are lightweight helpers
that convert a file path to a float tensor suitable for feeding into the
models.  They belong in the `luminascale.utils` package so they can be
imported from anywhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
import numpy as np


def image_to_tensor(image_path: Path | str) -> torch.Tensor:
    """Load an image and convert it to a normalized float tensor.

    The bit depth is inferred from the numpy dtype of the loaded image.

    Args:
        image_path: Path or string pointing to an image file that PIL can
            open (PNG, JPEG, TIFF, etc.).

    Returns:
        Tensor of shape ``[C, H, W]`` with ``dtype=torch.float32`` and
        values scaled to ``[0.0, 1.0]``.
    """
    # Load image, convert to RGB
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        array = np.array(img)

    # Determine bit-depth from array dtype
    dtype_to_bits = {
        np.dtype("uint8"): 8,
        np.dtype("uint16"): 16,
        np.dtype("uint32"): 32,
    }
    bit_depth = dtype_to_bits.get(array.dtype, 8)

    if bit_depth not in [8, 10, 12, 16]:
        raise ValueError(f"Unsupported bit-depth: {bit_depth}")

    max_value = (2 ** bit_depth) - 1

    # Cast to float and normalize to [0.0, 1.0]
    tensor = torch.from_numpy(array).float() / max_value

    # Permute to [C, H, W]
    return tensor.permute(2, 0, 1)
