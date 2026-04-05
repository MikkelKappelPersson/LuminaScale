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


def read_exr(path: Path | str) -> np.ndarray:
    """Read an EXR file and return as [C, H, W] numpy array."""
    buf = oiio.ImageBuf(str(path))
    if not buf.initialized:
        raise FileNotFoundError(f"Could not load EXR file: {path}")
    
    # Get pixels as float32 [H, W, C]
    array = np.asarray(buf.get_pixels(), dtype=np.float32)
    
    # Ensure RGB
    if array.shape[2] > 3:
        array = array[:, :, :3]
        
    # Transpose to [C, H, W]
    return array.transpose(2, 0, 1)


def write_exr(path: Path | str, array: np.ndarray | torch.Tensor):
    """Write an image (numpy [C, H, W] or tensor) to an EXR file."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
        
    # Transpose to [H, W, C] for OpenImageIO
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
        
    spec = oiio.ImageSpec(array.shape[1], array.shape[0], array.shape[2], oiio.FLOAT)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"Could not create EXR output for: {path}")
        
    out.open(str(path), spec)
    out.write_image(array)
    out.close()


def oiio_aces_to_display(path: Path | str) -> np.ndarray:
    """Read ACES EXR and convert to sRGB display space using OpenImageIO.
    
    This is a CPU alternative to aces_to_display_gpu for simple visualization.
    """
    buf = oiio.ImageBuf(str(path))
    # Apply color space conversion (depends on OCIO config)
    # This assumes OpenImageIO can find the OCIO config from environment or default
    result = oiio.ImageBufAlgo.colorconvert(buf, "aces", "sRGB")
    return np.asarray(result.get_pixels(), dtype=np.float32).transpose(2, 0, 1)


def aces_to_display_gpu(
    aces_tensor: torch.Tensor,
    input_cs: str = "ACES2065-1",
    display: str = "sRGB - Display",
    view: str = "ACES 2.0 - SDR 100 nits (Rec.709)",
    use_pytorch: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated ACES-to-display color transform.

    Applies ACES rendering transform (RRT + ODT) to convert from
    scene-referred linear ACES to display-referred sRGB.
    
    Now supports both PyTorch-native and OCIO backends for comparison.
    PyTorch is faster but uses analytical tone mapping; OCIO is reference.

    Args:
        aces_tensor: [H, W, 3] float32 tensor on device in ACES2065-1.
        input_cs: Input color space (default: ACES2065-1).
        display: OCIO display name (default: sRGB - Display).
        view: OCIO view name (default: ACES 2.0 SDR 100 nits).
        use_pytorch: Use PyTorch-native transform (default: True).

    Returns:
        Tuple of (srgb_32bit, srgb_8bit):
            - srgb_32bit: [H, W, 3] float32 on same device, values in [0, 1]
            - srgb_8bit: [H, W, 3] uint8 on same device, quantized to [0, 255]
    """
    if use_pytorch:
        # PyTorch-native implementation (with LUT for accuracy)
        from .pytorch_aces_transformer import ACESColorTransformer

        device = aces_tensor.device
        transformer = ACESColorTransformer(device=device, use_lut=True)
        
        srgb_32bit = transformer.aces_to_srgb_32f(aces_tensor)
        srgb_8bit = transformer.aces_to_srgb_8u(aces_tensor)
        
    else:
        # OCIO implementation (reference, requires CUDA for GPU)
        from .gpu_torch_processor import GPUTorchProcessor

        if not aces_tensor.is_cuda:
            raise ValueError("OCIO backend requires CUDA device")

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



