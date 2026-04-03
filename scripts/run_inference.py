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
from luminascale.utils.image_generator import create_sky_gradient, quantize_to_8bit

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
    """Generate and run inference on a synthetic sky gradient."""
    print(f"Generating synthetic sky gradient ({width}x{height})...")
    # Align size to 64
    target_w = (width // 64) * 64
    target_h = (height // 64) * 64
    
    hdr = create_sky_gradient(width=target_w, height=target_h, dtype="float32")
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
    
    # 1. Standard Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ldr.transpose(1, 2, 0))
    axes[0].set_title("8-bit Input (Quantized)")
    axes[1].imshow(np.clip(model_out.transpose(1, 2, 0), 0, 1))
    axes[1].set_title("Model Output (Dequantized)")
    axes[2].imshow(gt.transpose(1, 2, 0))
    axes[2].set_title("32-bit Reference")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # 2. High Contrast Comparison (to reveal banding)
    contrast_factor = 25.0
    def apply_contrast(x):
        return np.clip((x - 0.5) * contrast_factor + 0.5, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(apply_contrast(ldr.transpose(1, 2, 0)))
    axes[0].set_title(f"Input (Contrast {contrast_factor}x)")
    axes[1].imshow(apply_contrast(np.clip(model_out.transpose(1, 2, 0), 0, 1)))
    axes[1].set_title(f"Model (Contrast {contrast_factor}x)")
    axes[2].imshow(apply_contrast(gt.transpose(1, 2, 0)))
    axes[2].set_title(f"Reference (Contrast {contrast_factor}x)")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    contrast_path = save_path.parent / f"{save_path.stem}_contrast{save_path.suffix}"
    plt.savefig(contrast_path, dpi=150)
    plt.close()
    print(f"✓ Comparison plots saved: {save_path} and {contrast_path}")

def main():
    parser = argparse.ArgumentParser(description="LuminaScale Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input", type=str, help="Path to input image (optional if --synthetic used)")
    parser.add_argument("--output", type=str, default="outputs/inference/result.exr", help="Path to save output")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic sky gradient instead of reading input")
    parser.add_argument("--width", type=int, default=1024, help="Width for synthetic gradient")
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
    state_dict = torch.load(args.checkpoint, map_location=device)
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
