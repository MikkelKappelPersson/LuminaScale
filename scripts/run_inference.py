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

from luminascale.models import create_dequantization_net
from luminascale.utils.io import read_exr, write_exr, oiio_aces_to_display
from luminascale.utils.image_generator import create_primary_gradients, quantize_to_8bit

def align_to_model(tensor: torch.Tensor) -> torch.Tensor:
    """Align input tensor to be divisible by 64 (for 6-level U-Net)."""
    h, w = tensor.shape[2], tensor.shape[3]
    new_h = (h // 64) * 64
    new_w = (w // 64) * 64
    
    if h != new_h or w != new_w:
        print(f"Aligning resolution: {h}x{w} -> {new_h}x{new_w}")
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return tensor

def run_synthetic_inference(model: torch.nn.Module, device: torch.device, width: int, height: int, output_path: Path):
    """Generate and run inference on a synthetic primary gradients image."""
    print(f"Generating synthetic primary gradients ({width}x{height})...")
    # Align size to 64
    target_w = (width // 64) * 64
    target_h = (height // 64) * 64
    
    hdr = create_primary_gradients(width=target_w, height=target_h, dtype="float32")
    hdr_clipped = np.clip(hdr, 0, 1)
    ldr = quantize_to_8bit(hdr_clipped)
    
    input_tensor = torch.from_numpy(ldr).float().to(device).unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().numpy()
    
    # Save as EXR (normalized model output)
    write_exr(output_path, output_tensor)
    
    # Save a comparison side-by-side if possible
    save_comparison(ldr, output_tensor, hdr_clipped, output_path.with_suffix('.png'))
    print(f"✓ Synthetic inference complete. Model output: {output_path}")

def run_image_inference(model: torch.nn.Module, device: torch.device, input_path: Path, output_path: Path):
    """Run inference on a local image file."""
    print(f"Processing image: {input_path}")
    
    if input_path.suffix.lower() == '.exr':
        input_np = read_exr(input_path)
    else:
        img = Image.open(input_path).convert("RGB")
        input_np = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).to(device)
    input_tensor = align_to_model(input_tensor)
    
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().numpy()
    
    if output_path.suffix.lower() == '.exr':
        write_exr(output_path, output_tensor)
    else:
        out_img = Image.fromarray((np.clip(output_tensor.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
        out_img.save(output_path)
    
    print(f"✓ Inference complete. Saved to: {output_path}")

def save_comparison(ldr, model_out, gt, save_path: Path):
    """Save side-by-side comparison images (original and high-contrast)."""
    import matplotlib.pyplot as plt
    
    # Calculate unique values for each
    ldr_np = ldr.transpose(1, 2, 0) if isinstance(ldr, np.ndarray) else ldr.cpu().numpy().transpose(1, 2, 0)
    model_out_np = np.clip(model_out.transpose(1, 2, 0), 0, 1) if isinstance(model_out, np.ndarray) else np.clip(model_out.cpu().numpy().transpose(1, 2, 0), 0, 1)
    gt_np = gt.transpose(1, 2, 0) if isinstance(gt, np.ndarray) else gt.cpu().numpy().transpose(1, 2, 0)
    
    ldr_unique = len(np.unique(np.round(ldr_np.reshape(-1, 3), decimals=6)))
    model_unique = len(np.unique(np.round(model_out_np.reshape(-1, 3), decimals=6)))
    gt_unique = len(np.unique(np.round(gt_np.reshape(-1, 3), decimals=6)))
    
    # 1. Standard Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ldr_np)
    axes[0].set_title(f"8-bit Input - {ldr_unique:,} unique", fontsize=12, fontweight="bold")
    axes[1].imshow(model_out_np)
    axes[1].set_title(f"Model Output - {model_unique:,} unique", fontsize=12, fontweight="bold")
    axes[2].imshow(gt_np)
    axes[2].set_title(f"32-bit Reference - {gt_unique:,} unique", fontsize=12, fontweight="bold")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # 2. High Contrast Comparison (to reveal banding)
    contrast_factor = 25.0
    def apply_contrast(x):
        return np.clip((x - 0.5) * contrast_factor + 0.5, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(apply_contrast(ldr_np))
    axes[0].set_title(f"Input (Contrast {contrast_factor}x) - {ldr_unique:,} unique", fontsize=12, fontweight="bold")
    axes[1].imshow(apply_contrast(model_out_np))
    axes[1].set_title(f"Model (Contrast {contrast_factor}x) - {model_unique:,} unique", fontsize=12, fontweight="bold")
    axes[2].imshow(apply_contrast(gt_np))
    axes[2].set_title(f"Reference (Contrast {contrast_factor}x) - {gt_unique:,} unique", fontsize=12, fontweight="bold")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    contrast_path = save_path.parent / f"{save_path.stem}_contrast{save_path.suffix}"
    plt.savefig(contrast_path, dpi=150)
    plt.close()
    
    # 3. Difference Map (reveal what the network actually changed)
    # Calculate absolute difference and error
    diff_map = np.abs(model_out_np - ldr_np)  # Show per-pixel changes
    diff_luma = np.mean(diff_map, axis=2)  # Average across RGB
    
    # Also compare model output to ground truth
    model_to_gt = np.abs(model_out_np - gt_np)
    model_to_gt_luma = np.mean(model_to_gt, axis=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Difference: model vs input
    im0 = axes[0].imshow(diff_luma, cmap='hot')
    axes[0].set_title(f"Difference Map (Output - Input)\nMean diff: {diff_luma.mean():.6f}", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0])
    
    # Difference: model vs GT (error)
    im1 = axes[1].imshow(model_to_gt_luma, cmap='hot')
    axes[1].set_title(f"Model Error (Output - Reference)\nMean error: {model_to_gt_luma.mean():.6f}", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1])
    
    # Histogram of unique value distribution
    axes[2].hist(ldr_np.flatten(), bins=50, alpha=0.5, label=f"Input ({ldr_unique} unique)", color='red')
    axes[2].hist(model_out_np.flatten(), bins=50, alpha=0.5, label=f"Output ({model_unique} unique)", color='blue')
    axes[2].hist(gt_np.flatten(), bins=50, alpha=0.5, label=f"Reference ({gt_unique} unique)", color='green')
    axes[2].set_xlabel("Pixel Value")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Value Distribution", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    diff_path = save_path.parent / f"{save_path.stem}_diff{save_path.suffix}"
    plt.savefig(diff_path, dpi=150)
    plt.close()
    print(f"✓ Comparison plots saved: {save_path} and {contrast_path}")
    print(f"✓ Difference map saved: {diff_path}")

def main():
    parser = argparse.ArgumentParser(description="LuminaScale Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input", type=str, help="Path to input image (optional if --synthetic used)")
    parser.add_argument("--output", type=str, default="outputs/inference/result.exr", help="Path to save output")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic sky gradient instead of reading input")
    parser.add_argument("--width", type=int, default=512, help="Width for synthetic gradient")
    parser.add_argument("--height", type=int, default=512, help="Height for synthetic gradient")
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
        run_synthetic_inference(model, device, args.width, args.height, output_path)
    elif args.input:
        run_image_inference(model, device, Path(args.input), output_path)
    else:
        print("Error: You must provide either --input or --synthetic")
        sys.exit(1)

if __name__ == "__main__":
    main()
