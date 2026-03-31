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
import OpenImageIO as oiio


def aces_to_display_gpu(
    aces_tensor: torch.Tensor,
    input_cs: str = "ACES2065-1",
    display: str = "sRGB - Display",
    view: str = "ACES 2.0 - SDR 100 nits (Rec.709)",
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated ACES-to-display color transform using OCIO.

    Applies ACES rendering transform (RRT + ODT) to convert from
    scene-referred linear ACES to display-referred sRGB.

    Args:
        aces_tensor: [H, W, 3] float32 tensor on CUDA in ACES2065-1 color space.
        input_cs: Input color space (default: ACES2065-1).
        display: OCIO display name (default: sRGB - Display).
        view: OCIO view name (default: ACES 2.0 SDR 100 nits).

    Returns:
        Tuple of (srgb_32bit, srgb_8bit):
            - srgb_32bit: [H, W, 3] float32 on GPU, values in [0, 1]
            - srgb_8bit: [H, W, 3] uint8 on GPU, quantized to [0, 255]
    """
    from .gpu_torch_processor import GPUTorchProcessor

    if not aces_tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    processor = GPUTorchProcessor(headless=True)
    srgb_32bit, srgb_8bit = processor.apply_ocio_torch(
        aces_tensor,
        input_cs=input_cs,
        display=display,
        view=view,
    )
    processor.cleanup()

    return srgb_32bit, srgb_8bit


def image_to_tensor(image_path: Path | str) -> torch.Tensor:
    """Load an image and convert it to a normalized float tensor.

    The bit depth is inferred from the numpy dtype of the loaded image.

    Args:
        image_path: Path or string pointing to an image file.
            Supports formats PIL can open, plus EXR using OpenImageIO.

    Returns:
        Tensor of shape ``[C, H, W]`` with ``dtype=torch.float32`` and
        values scaled to ``[0.0, 1.0]``.
    """
    image_path = Path(image_path)
    
    # Handle EXR using OpenImageIO
    if image_path.suffix.lower() == ".exr":
        buf = oiio.ImageBuf(str(image_path))
        assert buf.initialized, f"Could not load EXR file: {image_path}"
        
        # OpenImageIO returns [H, W, C]
        array = np.asarray(buf.get_pixels(), dtype=np.float32)
        
        # Ensure RGB
        if array.shape[2] > 3:
            array = array[:, :, :3]
            
        tensor = torch.from_numpy(array).float()
        return tensor.permute(2, 0, 1)

    # Standard formats via PIL
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

    assert bit_depth in [8, 10, 12, 16], f"Unsupported bit-depth: {bit_depth}"

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

    assert raw_dir.exists(), f"RAW directory not found: {raw_dir}"

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
        assert False, (
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



