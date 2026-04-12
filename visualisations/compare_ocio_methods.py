"""Compare CPU OCIO vs GPU LUT precision loss.

This script evaluates both the official CPU OCIO processor and the GPU LUT-based
method to determine if precision loss comes from:
1. The LUT approximation (discretization error)
2. The OCIO color transformation itself (inherent to the algorithm)

Usage:
    python visualisations/compare_ocio_methods.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import tempfile
import os
import webdataset as wds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_single_aces_sample() -> np.ndarray:
    """Load a single ACES sample from validation shards."""
    logger.info("Loading ACES sample from validation shards...")
    
    shard_path = Path(project_root) / "dataset/temp/shards/val"
    pattern = str(shard_path / "val-{000000..999999}.tar")
    
    dataset = wds.WebDataset(pattern).decode().to_tuple("exr")
    
    for idx, (exr_data,) in enumerate(dataset):
        if idx == 0:
            import OpenImageIO as oiio
            
            # Decode EXR
            with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
                tmp.write(exr_data)
                temp_file = tmp.name
            
            try:
                buf_input = oiio.ImageInput.open(temp_file)
                pixels = buf_input.read_image("float")
                buf_input.close()
                
                if pixels.ndim == 1:
                    spec = buf_input.spec()
                    pixels = pixels.reshape((spec.height, spec.width, spec.nchannels))
                elif pixels.ndim == 3 and pixels.shape[0] == 3:
                    pixels = pixels.transpose(1, 2, 0)
                
                # Crop to reasonable size for fast processing
                h, w = min(pixels.shape[0], 1024), min(pixels.shape[1], 1024)
                aces = pixels[:h, :w, :].astype(np.float32)
                
                logger.info(f"✓ Loaded ACES sample: {aces.shape}")
                return aces
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    raise RuntimeError("No samples found")


def cpu_ocio_transform(aces: np.ndarray) -> np.ndarray:
    """Official OCIO CPU method: ACES → sRGB.
    
    This is the reference/ground truth method.
    """
    import PyOpenColorIO as ocio
    
    logger.info("Running CPU OCIO transformation...")
    
    config = ocio.Config.CreateFromFile(
        str(Path(project_root) / "config/aces/studio-config.ocio")
    )
    
    processor = config.getProcessor(
        "ACES2065-1",
        "sRGB - Display",
        "ACES 2.0 - SDR 100 nits (Rec.709)",
        ocio.TRANSFORM_DIR_FORWARD
    )
    
    cpu_processor = processor.getDefaultCPUProcessor()
    
    # Process image - OCIO expects RGBA
    h, w, c = aces.shape
    aces_rgba = np.zeros((h, w, 4), dtype=np.float32)
    aces_rgba[:, :, :3] = aces
    aces_rgba[:, :, 3] = 1.0  # Alpha channel
    
    cpu_processor.applyRGBA(aces_rgba)
    
    # Extract RGB
    srgb = aces_rgba[:, :, :3].astype(np.float32)
    logger.info(f"✓ CPU OCIO done")
    
    return srgb


def gpu_lut_transform(aces: np.ndarray) -> np.ndarray:
    """GPU LUT-based method (current implementation)."""
    from luminascale.utils.gpu_cdl_processor import GPUCDLProcessor
    from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
    from luminascale.utils.look_generator import get_single_random_look
    
    logger.info("Running GPU LUT-based transformation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Apply CDL first (same as training)
    aces_tensor = torch.from_numpy(aces).to(device)
    
    cdl_processor = GPUCDLProcessor(device=device)
    look = get_single_random_look()
    aces_graded = cdl_processor.apply_cdl_gpu(aces_tensor, look)
    
    # Transform to sRGB via LUT
    aces_transformer = ACESColorTransformer(device=device, use_lut=True)
    srgb_tensor = aces_transformer.aces_to_srgb_32f(aces_graded.unsqueeze(0)).squeeze(0)
    
    srgb = srgb_tensor.cpu().numpy()
    if srgb.ndim == 3 and srgb.shape[0] == 3:
        srgb = srgb.transpose(1, 2, 0)
    
    logger.info(f"✓ GPU LUT done")
    
    return srgb


def analyze_precision(data: np.ndarray, stage_name: str) -> int:
    """Count unique values (indicates precision)."""
    if data.ndim == 3:
        if data.shape[0] == 3:
            data = data.transpose(1, 2, 0)
    
    # Sample center region
    h, w = data.shape[0], data.shape[1]
    center = data[h//4:3*h//4, w//4:3*w//4]
    if center.ndim == 3:
        center = center[:, :, 0]
    
    unique_vals = len(np.unique(np.round(center * 1e6)))
    
    logger.info(f"\n{stage_name}:")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Range: [{data.min():.8f}, {data.max():.8f}]")
    logger.info(f"  Unique values: {unique_vals:,}")
    
    return unique_vals


def main() -> None:
    """Main comparison."""
    logger.info("="*70)
    logger.info("OCIO METHOD COMPARISON: CPU Official vs GPU LUT")
    logger.info("="*70)
    
    # Load ACES sample
    aces = load_single_aces_sample()
    
    # Step 1: Analyze raw ACES
    logger.info("\n" + "="*70)
    logger.info("STEP 1: RAW ACES")
    logger.info("="*70)
    unique_aces = analyze_precision(aces, "Raw ACES")
    
    # Step 2: CPU OCIO (ground truth)
    logger.info("\n" + "="*70)
    logger.info("STEP 2: CPU OCIO DIRECT (Official Method)")
    logger.info("="*70)
    srgb_cpu = cpu_ocio_transform(aces)
    unique_cpu = analyze_precision(srgb_cpu, "sRGB (CPU OCIO)")
    
    # Step 3: GPU LUT
    logger.info("\n" + "="*70)
    logger.info("STEP 3: GPU LUT (Current Method)")
    logger.info("="*70)
    srgb_gpu = gpu_lut_transform(aces)
    unique_gpu = analyze_precision(srgb_gpu, "sRGB (GPU LUT)")
    
    # Step 4: Comparison
    logger.info("\n" + "="*70)
    logger.info("COMPARISON & ANALYSIS")
    logger.info("="*70)
    
    logger.info(f"\nUnique values (center region):")
    logger.info(f"  Raw ACES:       {unique_aces:,}")
    logger.info(f"  CPU OCIO:       {unique_cpu:,}")
    logger.info(f"  GPU LUT:        {unique_gpu:,}")
    
    logger.info(f"\nPrecision loss:")
    logger.info(f"  ACES → CPU OCIO: {unique_aces - unique_cpu:,} values ({100 * (unique_aces - unique_cpu) / unique_aces:.1f}%)")
    logger.info(f"  ACES → GPU LUT:  {unique_aces - unique_gpu:,} values ({100 * (unique_aces - unique_gpu) / unique_aces:.1f}%)")
    
    logger.info(f"\nLUT approximation error:")
    diff = np.abs(srgb_cpu - srgb_gpu)
    logger.info(f"  Max difference:  {diff.max():.8f}")
    logger.info(f"  Mean difference: {diff.mean():.8f}")
    logger.info(f"  Std deviation:   {diff.std():.8f}")
    logger.info(f"  Pixels with diff > 0.001: {np.sum(diff > 0.001) / diff.size * 100:.2f}%")
    
    # Conclusion
    logger.info(f"\n" + "="*70)
    logger.info("INTERPRETATION")
    logger.info("="*70)
    
    if abs(unique_cpu - unique_gpu) < unique_cpu * 0.05:
        logger.info("✓ LUT approximation is ACCURATE (~5% difference or less)")
        logger.info("  → Precision loss is inherent to OCIO color transformation")
        logger.info("  → Not caused by LUT discretization")
    else:
        logger.info("✗ LUT introduces significant additional quantization")
        logger.info(f"  → Higher resolution LUT or direct OCIO might help")
    
    if diff.mean() < 0.002:
        logger.info("\n✓ LUT interpolation error is SMALL (mean < 0.002)")
        logger.info("  → 384³ LUT is sufficient for this task")
    else:
        logger.info("\n✗ LUT interpolation has noticeable error")
        logger.info("  → Consider higher resolution or CPU OCIO for training")


if __name__ == "__main__":
    main()
