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

def align_to_model(tensor: torch.Tensor) -> torch.Tensor:
    """Align input tensor to be divisible by 64 (for 6-level U-Net)."""
    h, w = tensor.shape[2], tensor.shape[3]
    new_h = (h // 64) * 64
    new_w = (w // 64) * 64
    
    if h != new_h or w != new_w:
        print(f"Aligning resolution: {h}x{w} -> {new_h}x{new_w}")
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return tensor

def run_synthetic_inference(model: torch.nn.Module, device: torch.device, width: int | None, height: int | None, output_path: Path, gradient_type: str = "combined", contrast_strength: float = 10.0, apply_contrast_to_output: bool = False):
    """Generate and run inference on a synthetic primary gradients image.
    
    Args:
        gradient_type: "combined", "8x21", "4x21", or "2x21"
        width: Target width (None = native size)
        height: Target height (None = native size)
        apply_contrast_to_output: If True, apply S-curve contrast to model output before saving EXR
    
    Note:
        - Each gradient variant contains all three colors (Red, Green, Blue) + white separator
        - Single variants (8x21, 4x21, 2x21): [3, 64, width]
        - Combined (all three stacked): [3, 192, width]
    """
    print(f"Generating synthetic primary gradients ({gradient_type})...")
    
    # Generate HDR (8-bit quantized input) version
    if gradient_type == "combined":
        hdr = combine_primary_gradients(width=128, dtype="float32")  # [3, 192, 128]
        # Transpose to [H, W, C] for processing
        hdr = np.transpose(hdr, (1, 2, 0))  # [192, 128, 3]
        
        target_w = 128  # Width always stays 128
        target_h = 192  # Combined is always 192px (3 variants × 64)
        
        if height is not None:
            # Scale height if specified
            scale = height / 192.0
            target_h = height
            from PIL import Image as PILImage
            hdr_img = PILImage.fromarray((np.clip(hdr, 0, 1) * 255).astype(np.uint8))
            hdr_img = hdr_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
            hdr = np.array(hdr_img).astype(np.float32) / 255.0
            print(f"  Scaled to {target_w}×{target_h}")
        
        # Reference: full 0-255 gradient for comparison
        reference = combine_reference_gradients(width=128, dtype="float32")  # [3, 192, 128]
        reference = np.transpose(reference, (1, 2, 0))  # [192, 128, 3]
        
        if height is not None:
            ref_img = PILImage.fromarray((np.clip(reference, 0, 1) * 255).astype(np.uint8))
            ref_img = ref_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
            reference = np.array(ref_img).astype(np.float32) / 255.0
            print(f"  Scaled to {target_w}×{target_h}")
    else:
        # Single variants - always [3, 64, 128]
        target_w = 128  # Width always 128
        target_h = 64   # Height always 64 for single variants
        
        if gradient_type == "8x21":
            hdr_tensor = create_primary_gradients(width=128, block_width=8, dtype="float32")
            ref_tensor = create_reference_gradients(width=128, block_width=8, dtype="float32")
        elif gradient_type == "4x21":
            hdr_tensor = create_primary_gradients(width=128, block_width=4, dtype="float32")
            ref_tensor = create_reference_gradients(width=128, block_width=4, dtype="float32")
        elif gradient_type == "2x21":
            hdr_tensor = create_primary_gradients(width=128, block_width=2, dtype="float32")
            ref_tensor = create_reference_gradients(width=128, block_width=2, dtype="float32")
        else:
            raise ValueError(f"Unknown gradient_type: {gradient_type}")
        
        # Transpose from [C,H,W] to [H,W,C]
        hdr = np.transpose(hdr_tensor, (1, 2, 0))
        reference = np.transpose(ref_tensor, (1, 2, 0))
    
    hdr_clipped = np.clip(hdr, 0, 1)
    ldr = quantize_to_8bit(hdr_clipped)
    
    # Convert back to [C,H,W] for model
    ldr_chw = np.transpose(ldr, (2, 0, 1))
    
    input_tensor = torch.from_numpy(ldr_chw).float().to(device).unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().numpy()
    
    # Optionally apply contrast to output before saving
    output_to_save = output_tensor
    if apply_contrast_to_output:
        output_to_save = apply_s_curve_contrast(output_tensor.transpose(1, 2, 0), strength=contrast_strength).transpose(2, 0, 1)
    
    # Save as EXR (normalized model output, possibly with contrast)
    write_exr(output_path, output_to_save)
    
    # Save a comparison: input vs model output vs reference (full range ground truth)
    reference_clipped = np.clip(reference, 0, 1)
    save_comparison(ldr_chw, output_tensor, np.transpose(reference_clipped, (2, 0, 1)), output_path.with_suffix('.png'), strength=contrast_strength, synthetic=gradient_type)
    print(f"✓ Synthetic inference complete. Model output: {output_path}")

def run_image_inference(model: torch.nn.Module, device: torch.device, input_path: Path, output_path: Path, contrast_strength: float = 20.0, apply_contrast_to_output: bool = False):
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
    
    # Optionally apply contrast to output before saving
    output_to_save = output_tensor
    if apply_contrast_to_output:
        output_to_save = apply_s_curve_contrast(output_tensor.transpose(1, 2, 0), strength=contrast_strength).transpose(2, 0, 1)
    
    if output_path.suffix.lower() == '.exr':
        write_exr(output_path, output_to_save)
    else:
        out_img = Image.fromarray((np.clip(output_to_save.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
        out_img.save(output_path)
    
    # Save comparison visualization (resize input to match output dimensions for shape compatibility)
    output_clipped = np.clip(output_tensor, 0, 1)
    
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
    
    save_comparison(input_chw, output_clipped, input_chw, output_path.with_suffix('.png'), strength=contrast_strength)
    
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
    
    # Row 3: Difference Maps and Histogram
    diff_map = np.abs(model_out_np - ldr_np)
    diff_luma = np.mean(diff_map, axis=2)
    
    model_to_gt = np.abs(model_out_np - gt_np)
    model_to_gt_luma = np.mean(model_to_gt, axis=2)
    
    # Difference: model vs input
    ax20 = fig.add_subplot(gs[2, 0])
    im0 = ax20.imshow(diff_luma, cmap='hot', interpolation='nearest', aspect='auto')
    ax20.set_title(f"Output - Input\nMean: {diff_luma.mean():.6f}", fontsize=10, fontweight="bold", pad=8)
    ax20.axis('off')
    cbar0 = plt.colorbar(im0, ax=ax20, fraction=0.046, pad=0.04, shrink=0.8)
    cbar0.ax.tick_params(labelsize=8)
    
    # Difference: model vs GT (error)
    ax21 = fig.add_subplot(gs[2, 1])
    im1 = ax21.imshow(model_to_gt_luma, cmap='hot', interpolation='nearest', aspect='auto')
    ax21.set_title(f"Output - Reference\nMean: {model_to_gt_luma.mean():.6f}", fontsize=10, fontweight="bold", pad=8)
    ax21.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax21, fraction=0.046, pad=0.04, shrink=0.8)
    cbar1.ax.tick_params(labelsize=8)
    
    # Histogram
    ax22 = fig.add_subplot(gs[2, 2])
    
    # Prepare histogram data - omit high values if synthetic is provided
    ldr_hist = ldr_np.flatten()
    model_hist = model_out_np.flatten()
    gt_hist = gt_np.flatten()
    
    if synthetic is not None:
        # Filter out values > 0.9 to better space out lower values for synthetic data
        ldr_hist = ldr_hist[ldr_hist <= 0.9]
        model_hist = model_hist[model_hist <= 0.9]
        gt_hist = gt_hist[gt_hist <= 0.9]
    
    # Compute shared bins across all three datasets
    all_values = np.concatenate([ldr_hist, model_hist, gt_hist])
    bins = np.linspace(all_values.min(), all_values.max(), 51)
    
    ax22.hist(ldr_hist, bins=bins, alpha=0.5, label=f"Input ({ldr_unique})", color='red', density=True)
    ax22.hist(model_hist, bins=bins, alpha=0.5, label=f"Output ({model_unique})", color='blue', density=True)
    ax22.hist(gt_hist, bins=bins, alpha=0.5, label=f"Reference ({gt_unique})", color='green', density=True)
    ax22.set_xlabel("Pixel Value", fontsize=9)
    ax22.set_ylabel("Frequency (density)", fontsize=9)
    ax22.set_title("Value Distribution", fontsize=10, fontweight="bold", pad=8)
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
    parser.add_argument("--contrast-strength", type=float, default=20.0, help="S-curve contrast strength for visualization (default: 20.0)")
    parser.add_argument("--apply-contrast-to-output", action="store_true", help="Apply S-curve contrast to model output before saving EXR (default: False)")
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
        run_synthetic_inference(model, device, args.width, args.height, output_path, gradient_type=args.gradient_type, contrast_strength=args.contrast_strength, apply_contrast_to_output=args.apply_contrast_to_output)
    elif args.input:
        run_image_inference(model, device, Path(args.input), output_path, contrast_strength=args.contrast_strength, apply_contrast_to_output=args.apply_contrast_to_output)
    else:
        print("Error: You must provide either --input or --synthetic")
        sys.exit(1)

if __name__ == "__main__":
    main()
