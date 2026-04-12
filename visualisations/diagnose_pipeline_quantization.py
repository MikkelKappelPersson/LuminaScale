"""Diagnose quantization artifacts through the training pipeline.

Traces data precision and artifacts at each stage:
1. Raw EXR decode (what bit-depth is it really?)
2. After CDL grading
3. After ACES→sRGB conversion
4. After quantization to 8-bit

This will reveal WHERE the information is being lost.
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
import matplotlib.pyplot as plt
import webdataset as wds
import tempfile
import os

from luminascale.utils.dataset_pair_generator import DatasetPairGenerator
from luminascale.utils.gpu_cdl_processor import GPUCDLProcessor
from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
from luminascale.utils.look_generator import get_single_random_look

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_single_exr_sample(val_shard_path: str) -> bytes:
    """Load a single EXR from validation shards.
    
    Args:
        val_shard_path: Path to validation shards
    
    Returns:
        Raw EXR bytes
    """
    logger.info(f"Loading single EXR sample from {val_shard_path}")
    
    shard_path = Path(project_root) / val_shard_path
    pattern = str(shard_path / "val-{000000..999999}.tar")
    
    dataset = wds.WebDataset(pattern).decode().to_tuple("exr")
    
    for idx, (exr_data,) in enumerate(dataset):
        if idx == 0:
            logger.info(f"✓ Loaded EXR sample (type: {type(exr_data)}, size: {len(exr_data) if isinstance(exr_data, bytes) else 'N/A'} bytes)")
            return exr_data
    
    raise RuntimeError("No samples found in validation shards")


def decode_exr_directly(exr_bytes: bytes) -> np.ndarray:
    """Decode EXR directly using OIIO to check raw bit-depth."""
    import OpenImageIO as oiio
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
        tmp.write(exr_bytes)
        temp_file = tmp.name
    
    try:
        buf_input = oiio.ImageInput.open(temp_file)
        if not buf_input:
            raise RuntimeError(f"OIIO failed to open EXR: {oiio.geterror()}")
        
        # Get specs BEFORE reading
        spec = buf_input.spec()
        logger.info(f"\n=== EXR File Specs (raw from disk) ===")
        logger.info(f"  Dimensions: {spec.width}x{spec.height}")
        logger.info(f"  Channels: {spec.nchannels}")
        logger.info(f"  Data format: {spec.format}")
        logger.info(f"  Depth: {spec.depth if hasattr(spec, 'depth') else 'N/A'}")
        
        # Read pixels
        pixels = buf_input.read_image("float")
        buf_input.close()
        
        # Reshape if needed
        if pixels.ndim == 1:
            pixels = pixels.reshape((spec.height, spec.width, spec.nchannels))
        elif pixels.ndim == 3 and pixels.shape[0] == 3:
            pixels = pixels.transpose(1, 2, 0)
        
        logger.info(f"  Loaded array shape: {pixels.shape}")
        logger.info(f"  Data dtype: {pixels.dtype}")
        logger.info(f"  Value range: [{pixels.min():.6f}, {pixels.max():.6f}]")
        
        return pixels.astype(np.float32)
    
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def analyze_precision(data: np.ndarray, stage_name: str) -> tuple[float, float, int]:
    """Analyze how many unique values exist (indicates quantization level).
    
    Returns:
        (min_val, max_val, num_unique_values)
    """
    # Ensure [H, W, C] format
    if data.ndim == 3:
        if data.shape[0] == 3:
            data = data.transpose(1, 2, 0)
        # else already [H, W, C]
    
    # Sample center region to avoid edge artifacts
    h, w = data.shape[0], data.shape[1]
    center = data[h//4:3*h//4, w//4:3*w//4]
    if center.ndim == 3:
        center = center[:, :, 0]  # Use R channel
    
    unique_vals = len(np.unique(np.round(center * 1e6)))  # Round to 6 decimals
    
    logger.info(f"\n=== {stage_name} ===")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Dtype: {data.dtype}")
    logger.info(f"  Range: [{data.min():.8f}, {data.max():.8f}]")
    logger.info(f"  Unique values (in center region, ~6 decimals): {unique_vals}")
    
    return data.min(), data.max(), unique_vals


def main() -> None:
    """Main diagnostic pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}\n")
    
    # Step 1: Load raw EXR and check its native format
    logger.info("="*60)
    logger.info("STEP 1: RAW EXR DECODE (from disk)")
    logger.info("="*60)
    
    exr_bytes = load_single_exr_sample("dataset/temp/shards/val")
    aces_raw = decode_exr_directly(exr_bytes)
    min_raw, max_raw, unique_raw = analyze_precision(aces_raw, "Raw ACES (float32)")
    
    # Estimate bit depth from unique values
    est_bits = np.log2(unique_raw) if unique_raw > 1 else 0
    logger.info(f"  → Estimated bit-depth: ~{est_bits:.1f} bits")
    
    # Step 2: After CDL grading (GPU processing)
    logger.info("\n" + "="*60)
    logger.info("STEP 2: CDL GRADING (GPU)")
    logger.info("="*60)
    
    aces_tensor = torch.from_numpy(aces_raw).to(device)
    cdl_processor = GPUCDLProcessor(device=device)
    look = get_single_random_look()
    aces_graded = cdl_processor.apply_cdl_gpu(aces_tensor, look)
    aces_graded_np = aces_graded.cpu().numpy()
    
    min_graded, max_graded, unique_graded = analyze_precision(aces_graded_np, "ACES after CDL")
    
    # Step 3: After ACES→sRGB (GPU LUT method)
    logger.info("\n" + "="*60)
    logger.info("STEP 3A: ACES → sRGB CONVERSION (GPU LUT Method)")
    logger.info("="*60)
    
    aces_transformer = ACESColorTransformer(device=device, use_lut=True)
    srgb_32f = aces_transformer.aces_to_srgb_32f(aces_graded.unsqueeze(0)).squeeze(0)
    srgb_32f_np = srgb_32f.cpu().numpy()
    # Handle potential shape issues
    if srgb_32f_np.ndim == 3:
        if srgb_32f_np.shape[0] == 3:
            srgb_32f_np = srgb_32f_np.transpose(1, 2, 0)
        # else already [H, W, C]
    
    min_srgb, max_srgb, unique_srgb = analyze_precision(srgb_32f_np, "sRGB 32-bit float (GPU LUT)")
    
    # Step 3B: CPU OCIO direct transformation (for comparison)
    logger.info("\n" + "="*60)
    logger.info("STEP 3B: ACES → sRGB CONVERSION (CPU OCIO Official Method)")
    logger.info("="*60)
    
    try:
        import PyOpenColorIO as ocio
        
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
        
        # Process graded ACES with CPU OCIO (no CDL, just color transform)
        aces_graded_np = aces_graded.cpu().numpy()
        if aces_graded_np.shape[0] == 3:
            aces_graded_np = aces_graded_np.transpose(1, 2, 0)
        
        h, w, c = aces_graded_np.shape
        aces_rgba = np.zeros((h, w, 4), dtype=np.float32)
        aces_rgba[:, :, :3] = aces_graded_np
        aces_rgba[:, :, 3] = 1.0
        
        cpu_processor.applyRGBA(aces_rgba)
        srgb_32f_cpu = aces_rgba[:, :, :3].astype(np.float32)
        
        min_srgb_cpu, max_srgb_cpu, unique_srgb_cpu = analyze_precision(srgb_32f_cpu, "sRGB 32-bit float (CPU OCIO)")
        
        # Compare GPU LUT vs CPU OCIO
        logger.info("\n" + "="*60)
        logger.info("COMPARISON: GPU LUT vs CPU OCIO")
        logger.info("="*60)
        
        diff_methods = np.abs(srgb_32f_np - srgb_32f_cpu)
        logger.info(f"\nPrecision loss:")
        logger.info(f"  ACES (after CDL) → GPU LUT:    {unique_srgb:,} unique values ({100 * (1162172 - unique_srgb) / 1162172:.1f}% loss)")
        logger.info(f"  ACES (after CDL) → CPU OCIO:   {unique_srgb_cpu:,} unique values ({100 * (1162172 - unique_srgb_cpu) / 1162172:.1f}% loss)")
        
        logger.info(f"\nDifference between methods:")
        logger.info(f"  Max difference:   {diff_methods.max():.8f}")
        logger.info(f"  Mean difference:  {diff_methods.mean():.8f}")
        logger.info(f"  Std deviation:    {diff_methods.std():.8f}")
        logger.info(f"  Pixels > 0.001:   {np.sum(diff_methods > 0.001) / diff_methods.size * 100:.2f}%")
        
        if diff_methods.mean() < 0.002:
            logger.info("\n  ✓ LUT approximation is ACCURATE - precision loss is from OCIO, not LUT")
        else:
            logger.info("\n  ✗ LUT introduces additional error - consider higher resolution or CPU OCIO")
        
        srgb_32f_cpu_copy = srgb_32f_cpu  # Save for visualization later
    
    except ImportError:
        logger.warning("PyOpenColorIO not available, skipping CPU OCIO comparison")
        srgb_32f_cpu_copy = None
    
    # Step 4: After 8-bit quantization
    logger.info("\n" + "="*60)
    logger.info("STEP 4: QUANTIZE TO 8-BIT")
    logger.info("="*60)
    
    srgb_8u = ((srgb_32f * 255).round().to(torch.uint8)).float() / 255.0
    srgb_8u_np = srgb_8u.cpu().numpy()
    # Handle potential shape issues
    if srgb_8u_np.ndim == 3:
        if srgb_8u_np.shape[0] == 3:
            srgb_8u_np = srgb_8u_np.transpose(1, 2, 0)
        # else already [H, W, C]
    
    min_8u, max_8u, unique_8u = analyze_precision(srgb_8u_np, "sRGB 8-bit (quantized)")
    
    # Step 5: Difference analysis
    logger.info("\n" + "="*60)
    logger.info("STEP 5: DIFFERENCE ANALYSIS")
    logger.info("="*60)
    
    # Ensure same shape for difference
    if srgb_32f_np.ndim == 3 and srgb_32f_np.shape[0] == 3:
        srgb_32f_np_vis = srgb_32f_np.transpose(1, 2, 0)
    else:
        srgb_32f_np_vis = srgb_32f_np
    
    if srgb_8u_np.ndim == 3 and srgb_8u_np.shape[0] == 3:
        srgb_8u_np_vis = srgb_8u_np.transpose(1, 2, 0)
    else:
        srgb_8u_np_vis = srgb_8u_np
    
    diff_srgb = np.abs(srgb_32f_np_vis - srgb_8u_np_vis)
    logger.info(f"\nDifference (sRGB 32-bit vs 8-bit):")
    logger.info(f"  Max difference: {diff_srgb.max():.8f}")
    logger.info(f"  Mean difference: {diff_srgb.mean():.8f}")
    logger.info(f"  Pixels with diff > 0.001: {np.sum(diff_srgb > 0.001) / diff_srgb.size * 100:.2f}%")
    
    # Visualize the pipeline
    logger.info("\n" + "="*60)
    logger.info("CREATING VISUALIZATION")
    logger.info("="*60)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Normalize for visualization
    aces_raw_vis = np.clip(aces_raw / aces_raw.max(), 0, 1)
    aces_graded_vis = np.clip(aces_graded_np / aces_graded_np.max(), 0, 1)
    
    axes[0, 0].imshow(aces_raw_vis)
    axes[0, 0].set_title(f"Raw ACES\n({unique_raw} unique values)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(aces_graded_vis)
    axes[0, 1].set_title(f"After CDL\n({unique_graded} unique values)")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(srgb_32f_np_vis)
    axes[0, 2].set_title(f"sRGB 32-bit\n({unique_srgb} unique values)")
    axes[0, 2].axis("off")
    
    axes[0, 3].imshow(srgb_8u_np_vis)
    axes[0, 3].set_title(f"sRGB 8-bit\n({unique_8u} unique values)")
    axes[0, 3].axis("off")
    
    # Difference visualizations (25x contrast for visibility)
    contrast = 25.0
    
    diff_aces_cdl = np.abs(aces_raw_vis - aces_graded_vis)
    diff_aces_cdl_c = np.clip((diff_aces_cdl - 0.5) * contrast + 0.5, 0, 1)
    axes[1, 0].imshow(diff_aces_cdl_c)
    axes[1, 0].set_title("Raw vs CDL\n(contrast boosted)")
    axes[1, 0].axis("off")
    
    diff_cdl_srgb = np.abs(aces_graded_vis - (srgb_32f_np_vis / srgb_32f_np_vis.max()))
    diff_cdl_srgb_c = np.clip((diff_cdl_srgb - 0.5) * contrast + 0.5, 0, 1)
    axes[1, 1].imshow(diff_cdl_srgb_c)
    axes[1, 1].set_title("CDL vs sRGB 32-bit\n(contrast boosted)")
    axes[1, 1].axis("off")
    
    diff_32_8_c = np.clip((diff_srgb - 0.5) * contrast + 0.5, 0, 1)
    axes[1, 2].imshow(diff_32_8_c)
    axes[1, 2].set_title("sRGB 32-bit vs 8-bit\n(contrast boosted)")
    axes[1, 2].axis("off")
    
    # Summary stats heatmap
    axes[1, 3].axis("off")
    summary_text = f"""
PRECISION SUMMARY

Raw ACES: {unique_raw:,} unique
CDL: {unique_graded:,} unique
sRGB 32-bit: {unique_srgb:,} unique
sRGB 8-bit: {unique_8u:,} unique

Quantization loss at each stage:
1. Raw→CDL: {unique_raw - unique_graded:,} values
2. CDL→sRGB: {unique_graded - unique_srgb:,} values
3. sRGB32→8: {unique_srgb - unique_8u:,} values

Total pipeline loss:
{unique_raw:,} → {unique_8u:,}
({100 * (unique_raw - unique_8u) / unique_raw:.1f}% reduction)
    """
    axes[1, 3].text(0.1, 0.5, summary_text, fontsize=10, family="monospace",
                    verticalalignment="center")
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "pipeline_quantization_diagnosis.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    logger.info(f"✓ Saved diagnostic visualization to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
