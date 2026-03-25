"""Utility helpers for image file I/O and preprocessing.

Stage 0 of the pipeline is non‑ML; the functions here are lightweight helpers
that convert a file path to a float tensor suitable for feeding into the
models.  They belong in the `luminascale.utils` package so they can be
imported from anywhere in the codebase.
"""

from __future__ import annotations

import subprocess
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


def convert_to_aces(
    raw_dir: Path | str,
    aces_dir: Path | str,
    max_images: int | None = None,
    data_dir: str = "/usr/local/share/rawtoaces/data",
    overwrite: bool = True,
) -> dict:
    """Convert RAW camera images to ACES format using rawtoaces.

    Processes images from raw_dir and saves ACES-converted versions to
    aces_dir. Uses the rawtoaces command-line tool for conversion.

    Args:
        raw_dir: Path to directory containing RAW camera files.
        aces_dir: Path to output directory for ACES-converted images.
        max_images: Maximum number of images to process. If None, processes
            all images in raw_dir.
        data_dir: Path to rawtoaces data directory (color matrices, etc.).
        overwrite: Whether to overwrite existing output files.

    Returns:
        Dictionary with conversion results containing:
            - "processed": Number of images successfully converted.
            - "errors": List of (filename, error_message) tuples for failures.
            - "output_dir": Absolute path to output directory.

    Raises:
        FileNotFoundError: If raw_dir does not exist.
        RuntimeError: If rawtoaces command is not available on system.
    """
    raw_dir = Path(raw_dir).resolve()
    aces_dir = Path(aces_dir).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW directory not found: {raw_dir}")

    # Create output directory
    aces_dir.mkdir(parents=True, exist_ok=True)

    # Check if rawtoaces is available
    try:
        subprocess.run(
            ["rawtoaces", "--help"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "rawtoaces command not found or not executable. "
            "Please ensure rawtoaces is installed and in your PATH."
        )

    # Gather files to process
    raw_files = sorted([f for f in raw_dir.iterdir() if f.is_file()])
    if max_images is not None:
        raw_files = raw_files[:max_images]

    processed = 0
    errors = []

    for raw_file in raw_files:
        try:
            cmd = [
                "rawtoaces",
                "--data-dir",
                data_dir,
                "--output-dir",
                str(aces_dir),
                "--create-dirs",
            ]
            if overwrite:
                cmd.append("--overwrite")
            cmd.append(str(raw_file))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            processed += 1
        except subprocess.CalledProcessError as e:
            errors.append((raw_file.name, e.stderr or str(e)))
        except subprocess.TimeoutExpired:
            errors.append((raw_file.name, "Conversion timeout (>120s)"))

    return {
        "processed": processed,
        "errors": errors,
        "output_dir": str(aces_dir),
    }
