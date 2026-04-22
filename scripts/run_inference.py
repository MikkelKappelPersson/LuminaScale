#!/usr/bin/env python3
"""LuminaScale - Unified Inference Script for Dequantization-Net.

Supports both synthetic (sky gradient) and local image (PNG/EXR) inference.
Handles input resolution alignment (divisible by 64).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from PIL import Image
try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configure matplotlib to use non-interactive backend (suppresses Fontconfig warnings in headless envs)
import matplotlib
matplotlib.use('Agg')

from luminascale.models import create_dequantization_net
from luminascale.utils.io import read_exr, write_exr, oiio_aces_to_display
from luminascale.utils.image_generator import (
    combine_primary_gradients,
    create_primary_gradients,
    combine_reference_gradients,
    create_reference_gradients,
    quantize_to_8bit,
    apply_s_curve_contrast,
)

def blur_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to a numpy image [H, W, C]."""
    if sigma <= 0:
        return image
    # Apply blur to each channel
    blurred = np.zeros_like(image)
    for c in range(image.shape[2]):
        blurred[:, :, c] = ndimage.gaussian_filter(image[:, :, c], sigma=sigma)
    return blurred

def align_to_model(tensor: torch.Tensor) -> torch.Tensor:
    """Align input tensor to be divisible by 64 (for 6-level U-Net)."""
    h, w = tensor.shape[2], tensor.shape[3]
    new_h = (h // 64) * 64
    new_w = (w // 64) * 64
    
    if h != new_h or w != new_w:
        print(f"Aligning resolution: {h}x{w} -> {new_h}x{new_w}")
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return tensor

def run_synthetic_inference(model: torch.nn.Module, device: torch.device, width: int | None, height: int | None, output_path: Path, gradient_type: str = "combined", contrast_strength: float = 10.0, apply_contrast_to_output: bool = False, target_blur_sigma: float = 0.0):
    """Generate and run inference on a synthetic primary gradients image.
    
    Note on resizing: All processing (quantization, contrast, etc.) happens at native resolution.
    Resizing only occurs in save_comparison for visualization/display.
    
    Args:
        gradient_type: "combined", "8x21", "4x21", or "2x21"
        width: Ignored (kept for API compatibility)
        height: Ignored (kept for API compatibility)
        apply_contrast_to_output: If True, apply S-curve contrast to model output before saving EXR
        target_blur_sigma: Gaussian blur sigma to apply to reference (GT) for comparison
    
    Note:
        - Each gradient variant contains all three colors (Red, Green, Blue) + white separator
        - Single variants (8x21, 4x21, 2x21): [3, 64, width]
        - Combined (all three stacked): [3, 192, width]
    """
    print(f"Generating synthetic primary gradients ({gradient_type})...")
    
    # Generate gradients at native resolution (no resizing during processing)
    if gradient_type == "combined":
        hdr_chw = combine_primary_gradients(width=128, dtype="float32")  # [3, 192, 128]
        reference_chw = combine_reference_gradients(width=128, dtype="float32")  # [3, 192, 128]
    elif gradient_type == "8x21":
        hdr_chw = create_primary_gradients(width=128, block_width=8, dtype="float32")
        reference_chw = create_reference_gradients(width=128, block_width=8, dtype="float32")
    elif gradient_type == "4x21":
        hdr_chw = create_primary_gradients(width=128, block_width=4, dtype="float32")
        reference_chw = create_reference_gradients(width=128, block_width=4, dtype="float32")
    elif gradient_type == "2x21":
        hdr_chw = create_primary_gradients(width=128, block_width=2, dtype="float32")
        reference_chw = create_reference_gradients(width=128, block_width=2, dtype="float32")
    else:
        raise ValueError(f"Unknown gradient_type: {gradient_type}")
    
    # All processing at native resolution
    hdr_hwc = np.transpose(hdr_chw, (1, 2, 0))  # [H, W, 3]
    hdr_clipped = np.clip(hdr_hwc, 0, 1)
    ldr = quantize_to_8bit(hdr_clipped)  # Quantize at native resolution
    ldr_chw = np.transpose(ldr, (2, 0, 1))  # Back to [C, H, W]
    
    # Run model at native resolution
    input_tensor = torch.from_numpy(ldr_chw).float().to(device).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().numpy()
    
    # Clip model output to [0, 1] for visualization and saving consistency
    output_clipped = np.clip(output_tensor, 0, 1)
    
    # Optionally apply contrast to output before saving
    output_to_save = output_clipped
    if apply_contrast_to_output:
        output_to_save = apply_s_curve_contrast(output_clipped.transpose(1, 2, 0), strength=contrast_strength).transpose(2, 0, 1)
    
    # Save as EXR (native resolution, no resizing)
    write_exr(output_path, output_to_save)
    
    # Save comparison: resizing for visualization happens only in save_comparison
    reference_clipped = np.clip(reference_chw, 0, 1)
    reference_hwc = reference_clipped.transpose(1, 2, 0)
    
    if target_blur_sigma > 0:
        print(f"Applying reference blur for comparison: sigma={target_blur_sigma}")
        reference_hwc = blur_image(reference_hwc, sigma=target_blur_sigma)
        
    reference_final = reference_hwc.transpose(2, 0, 1)
    
    save_comparison(ldr_chw, output_clipped, reference_final, output_path.with_suffix('.png'), strength=contrast_strength, synthetic=gradient_type)
    print(f"✓ Synthetic inference complete. Model output: {output_path}")

def run_image_inference(model: torch.nn.Module, device: torch.device, input_path: Path, output_path: Path, contrast_strength: float = 20.0, apply_contrast_to_output: bool = False, target_blur_sigma: float = 0.0):
    """Run inference on a local image file."""
    print(f"Processing image: {input_path}")
    
    if input_path.suffix.lower() == '.exr':
        input_np = read_exr(input_path)
    else:
        img = Image.open(input_path).convert("RGB")
        input_np = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).to(device)
    original_h, original_w = input_tensor.shape[2], input_tensor.shape[3]
    
    input_tensor_aligned = align_to_model(input_tensor)
    aligned_h, aligned_w = input_tensor_aligned.shape[2], input_tensor_aligned.shape[3]
    
    with torch.no_grad():
        output_tensor = model(input_tensor_aligned).squeeze(0).cpu().numpy()
    
    # Clip model output to [0, 1] for visualization and saving consistency
    output_clipped = np.clip(output_tensor, 0, 1)
    
    # Optionally apply contrast to output before saving
    output_to_save = output_clipped
    if apply_contrast_to_output:
        output_to_save = apply_s_curve_contrast(output_clipped.transpose(1, 2, 0), strength=contrast_strength).transpose(2, 0, 1)
    
    if output_path.suffix.lower() == '.exr':
        write_exr(output_path, output_to_save)
    else:
        out_img = Image.fromarray((output_to_save.transpose(1, 2, 0) * 255).astype(np.uint8))
        out_img.save(output_path)
    
    # Save comparison visualization (resize input to match output dimensions for shape compatibility)
    # If input was resized by align_to_model, resize back for comparison visualization
    if (original_h != aligned_h or original_w != aligned_w):
        # Convert CHW back to HWC for PIL resizing
        input_hwc = input_np.transpose(1, 2, 0)  # [H_orig, W_orig, C]
        input_pil = Image.fromarray((np.clip(input_hwc, 0, 1) * 255).astype(np.uint8))
        # Resize to match output dimensions
        input_pil_resized = input_pil.resize((aligned_w, aligned_h), Image.Resampling.LANCZOS)
        # Convert back to CHW float
        input_chw = np.array(input_pil_resized).astype(np.float32).transpose(2, 0, 1) / 255.0
    else:
        input_chw = input_np
        
    # Apply reference blur for comparison visualization
    reference_for_comparison = input_chw
    if target_blur_sigma > 0:
        print(f"Applying reference blur for comparison: sigma={target_blur_sigma}")
        reference_hwc = input_chw.transpose(1, 2, 0)
        reference_hwc = blur_image(reference_hwc, sigma=target_blur_sigma)
        reference_for_comparison = reference_hwc.transpose(2, 0, 1)
    
    save_comparison(input_chw, output_clipped, reference_for_comparison, output_path.with_suffix('.png'), strength=contrast_strength)
    
    print(f"✓ Inference complete. Saved to: {output_path}")

def save_comparison(ldr, model_out, gt, save_path: Path, strength: float = 10.0, synthetic: str | None = None):
    """Save comprehensive 3x3 comparison grid with all analysis plots."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Calculate unique values for each
    ldr_np = ldr.transpose(1, 2, 0) if isinstance(ldr, np.ndarray) else ldr.cpu().numpy().transpose(1, 2, 0)
    model_out_np = np.clip(model_out.transpose(1, 2, 0), 0, 1) if isinstance(model_out, np.ndarray) else np.clip(model_out.cpu().numpy().transpose(1, 2, 0), 0, 1)
    gt_np = gt.transpose(1, 2, 0) if isinstance(gt, np.ndarray) else gt.cpu().numpy().transpose(1, 2, 0)
    
    ldr_unique = len(np.unique(np.round(ldr_np.reshape(-1, 3), decimals=6)))
    model_unique = len(np.unique(np.round(model_out_np.reshape(-1, 3), decimals=6)))
    gt_unique = len(np.unique(np.round(gt_np.reshape(-1, 3), decimals=6)))
    
    # Create 3x3 grid with proper spacing
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35, left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    # Row 1: Standard Comparison
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(ldr_np, interpolation='nearest', aspect='auto')
    ax00.set_title(f"8-bit Input\n{ldr_unique:,} unique", fontsize=10, fontweight="bold", pad=8)
    ax00.axis('off')
    
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(model_out_np, interpolation='nearest', aspect='auto')
    ax01.set_title(f"Model Output\n{model_unique:,} unique", fontsize=10, fontweight="bold", pad=8)
    ax01.axis('off')
    
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(gt_np, interpolation='nearest', aspect='auto')
    ax02.set_title(f"32-bit Reference\n{gt_unique:,} unique", fontsize=10, fontweight="bold", pad=8)
    ax02.axis('off')
    
    # Row 2: S-Curve Contrast
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(apply_s_curve_contrast(ldr_np, strength=strength), interpolation='nearest', aspect='auto')
    ax10.set_title(f"Input S-Curve ({strength})\n{ldr_unique:,} unique", fontsize=10, fontweight="bold", pad=8)
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(apply_s_curve_contrast(model_out_np, strength=strength), interpolation='nearest', aspect='auto')
    ax11.set_title(f"Model S-Curve ({strength})\n{model_unique:,} unique", fontsize=10, fontweight="bold", pad=8)
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.imshow(apply_s_curve_contrast(gt_np, strength=strength), interpolation='nearest', aspect='auto')
    ax12.set_title(f"Reference S-Curve ({strength})\n{gt_unique:,} unique", fontsize=10, fontweight="bold", pad=8)
    ax12.axis('off')
    
    # Row 3: Difference Maps and Gradient Ramp Comparison
    # Use s-curve enhanced versions for difference maps to reveal quantized steps
    ldr_enhanced = apply_s_curve_contrast(ldr_np, strength=strength)
    model_enhanced = apply_s_curve_contrast(model_out_np, strength=strength)
    gt_enhanced = apply_s_curve_contrast(gt_np, strength=strength)
    
    diff_map = np.abs(model_enhanced - ldr_enhanced)
    diff_luma = np.mean(diff_map, axis=2)
    
    model_to_gt = np.abs(model_enhanced - gt_enhanced)
    model_to_gt_luma = np.mean(model_to_gt, axis=2)
    
    # Difference: model vs input
    ax20 = fig.add_subplot(gs[2, 0])
    im0 = ax20.imshow(diff_luma, cmap='hot', interpolation='nearest', aspect='auto')
    ax20.set_title(f"Output - Input (S-Curved)\nMean: {diff_luma.mean():.6f}", fontsize=10, fontweight="bold", pad=8)
    ax20.axis('off')
    cbar0 = plt.colorbar(im0, ax=ax20, fraction=0.046, pad=0.04, shrink=0.8)
    cbar0.ax.tick_params(labelsize=8)
    
    # Difference: model vs GT (error)
    ax21 = fig.add_subplot(gs[2, 1])
    im1 = ax21.imshow(model_to_gt_luma, cmap='hot', interpolation='nearest', aspect='auto')
    ax21.set_title(f"Output - Reference (S-Curved)\nMean: {model_to_gt_luma.mean():.6f}", fontsize=10, fontweight="bold", pad=8)
    ax21.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax21, fraction=0.046, pad=0.04, shrink=0.8)
    cbar1.ax.tick_params(labelsize=8)
    
    # Gradient Ramp Comparison (replacing Histogram)
    ax22 = fig.add_subplot(gs[2, 2])
    
    # Pick a representative horizontal slice (avoiding white separators)
    # For synthetic combined, rows 10, 74, 138 are good (middle of R/G/B blocks)
    slice_y = 10 if ldr_np.shape[0] > 10 else ldr_np.shape[0] // 2
    
    # Use Green channel (usually highest precision/importance) or mean luma
    ldr_slice = ldr_np[slice_y, :, 1]
    model_slice = model_out_np[slice_y, :, 1]
    gt_slice = gt_np[slice_y, :, 1]
    
    ax22.plot(ldr_slice, 'r--', label='8-bit Input', alpha=0.8, linewidth=1)
    ax22.plot(gt_slice, 'g-', label='32-bit Reference', alpha=0.6, linewidth=1.5)
    ax22.plot(model_slice, 'b-', label='Model Output', alpha=0.9, linewidth=1)
    
    ax22.set_title("Gradient Ramp Comparison", fontsize=10, fontweight="bold", pad=8)
    ax22.set_xlabel("Pixel Index (Horizontal)", fontsize=9)
    ax22.set_ylabel("Pixel Value (Green)", fontsize=9)
    ax22.legend(fontsize=8, loc='upper left')
    ax22.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis limits to focus on the active range
    padding = (gt_slice.max() - gt_slice.min()) * 0.1
    ax22.set_ylim(gt_slice.min() - padding, gt_slice.max() + padding)
    ax22.legend(fontsize=8, loc='upper right')
    ax22.set_yscale('log')
    ax22.tick_params(labelsize=8)
    ax22.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comprehensive comparison grid saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="LuminaScale Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input", type=str, help="Path to input image (optional if --synthetic used)")
    parser.add_argument("--output", type=str, default="outputs/inference/result.exr", help="Path to save output")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic primary gradients instead of reading input")
    parser.add_argument("--gradient-type", type=str, default="combined", choices=["combined", "8x21", "4x21", "2x21"], help="Type of primary gradient to generate (default: combined)")
    parser.add_argument("--width", type=int, default=None, help="Width for synthetic gradient (default: native size)")
    parser.add_argument("--height", type=int, default=None, help="Height for synthetic gradient (default: native size)")
    parser.add_argument("--contrast-strength", type=float, default=100.0, help="S-curve contrast strength for visualization (default: 100.0)")
    parser.add_argument("--apply-contrast-to-output", action="store_true", help="Apply S-curve contrast to model output before saving EXR (default: False)")
    parser.add_argument("--target-blur-sigma", type=float, default=0.0, help="Gaussian blur sigma for reference (GT) comparison (default: 0.0)")
    parser.add_argument("--channels", type=int, default=32, help="Model base channels (default: 32)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()
    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print(f"Loading model on {device}...")
    model = create_dequantization_net(device=device, base_channels=args.channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle both Lightning checkpoints and raw state dicts
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # Remove "model." prefix if present (from Lightning wrapping)
    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Run Inference
    if args.synthetic:
        run_synthetic_inference(
            model, 
            device, 
            args.width, 
            args.height, 
            output_path, 
            gradient_type=args.gradient_type, 
            contrast_strength=args.contrast_strength, 
            apply_contrast_to_output=args.apply_contrast_to_output,
            target_blur_sigma=args.target_blur_sigma
        )
    elif args.input:
        run_image_inference(
            model, 
            device, 
            Path(args.input), 
            output_path, 
            contrast_strength=args.contrast_strength, 
            apply_contrast_to_output=args.apply_contrast_to_output,
            target_blur_sigma=args.target_blur_sigma
        )
    else:
        print("Error: You must provide either --input or --synthetic")
        sys.exit(1)

if __name__ == "__main__":
    main()
