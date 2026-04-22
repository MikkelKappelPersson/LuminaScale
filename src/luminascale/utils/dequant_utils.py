"""Shared utilities for dequantization model inference and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from ..models import DequantNet


def load_dequant_model(
    checkpoint_path: str | Path,
    device: torch.device,
    base_channels: int = 32,
    num_levels: int = 6,
) -> DequantNet:
    """Load a pretrained dequantization model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on (cuda/cpu)
        base_channels: Base channel count (must match training config)
        num_levels: Number of U-Net levels
    
    Returns:
        Loaded model in eval mode
    """
    model = DequantNet(
        in_channels=3,
        base_channels=base_channels,
        num_levels=num_levels,
    )
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def create_gaussian_kernel(size: int = 5, sigma: float = 1.0, device: torch.device | None = None) -> torch.Tensor:
    """Create a 2D Gaussian kernel for blur baseline.
    
    Args:
        size: Kernel size
        sigma: Gaussian standard deviation
        device: Device to create tensor on
    
    Returns:
        2D Gaussian kernel [1, 1, size, size]
    """
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    if device is not None:
        kernel_2d = kernel_2d.to(device)
    
    return kernel_2d


def apply_gaussian_blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur to an image using convolution.
    
    Args:
        image: Input tensor [3, H, W] or [1, 3, H, W]
        kernel: Gaussian kernel [1, 1, size, size]
    
    Returns:
        Blurred image same shape as input
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, 3, H, W]
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, channels, h, w = image.shape
    blurred_list = []
    
    for c in range(channels):
        blurred = F.conv2d(image[:, c:c+1, :, :], kernel, padding=kernel.shape[-1] // 2)
        blurred_list.append(blurred)
    
    blurred = torch.cat(blurred_list, dim=1)
    
    if squeeze_output:
        blurred = blurred.squeeze(0)
    
    return blurred


def compute_metrics(
    model_output: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Compute quality metrics comparing model output to reference.
    
    Args:
        model_output: Predicted image [3, H, W] or [H, W, 3] in [0, 1]
        reference: Ground truth image same shape
    
    Returns:
        Dict with keys: mse, psnr, ssim
    """
    # Ensure [3, H, W] format
    if model_output.shape[0] != 3 and len(model_output.shape) == 3:
        model_output = np.transpose(model_output, (2, 0, 1))
    if reference.shape[0] != 3 and len(reference.shape) == 3:
        reference = np.transpose(reference, (2, 0, 1))
    
    # Clip to valid range
    model_output = np.clip(model_output, 0, 1)
    reference = np.clip(reference, 0, 1)
    
    # MSE
    mse = np.mean((model_output - reference) ** 2)
    
    # PSNR
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    # SSIM on luminance
    model_lum = 0.299 * model_output[0] + 0.587 * model_output[1] + 0.114 * model_output[2]
    ref_lum = 0.299 * reference[0] + 0.587 * reference[1] + 0.114 * reference[2]
    ssim_val = ssim(model_lum, ref_lum, data_range=1.0)
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim_val),
    }


def compare_with_baseline(
    model_output: torch.Tensor,
    baseline_output: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    """Compare model output against baseline with metrics.
    
    Args:
        model_output: Model prediction [3, H, W]
        baseline_output: Baseline method output [3, H, W]
        reference: Ground truth [3, H, W]
    
    Returns:
        Dict with model, baseline, and improvement metrics
    """
    model_np = model_output.cpu().numpy()
    baseline_np = baseline_output.cpu().numpy()
    reference_np = reference.cpu().numpy()
    
    model_metrics = compute_metrics(model_np, reference_np)
    baseline_metrics = compute_metrics(baseline_np, reference_np)
    
    mse_improvement = (baseline_metrics['mse'] - model_metrics['mse']) / (baseline_metrics['mse'] + 1e-10) * 100
    
    return {
        'model': model_metrics,
        'baseline': baseline_metrics,
        'improvement': {
            'mse_pct': float(mse_improvement),
            'psnr_delta': float(model_metrics['psnr'] - baseline_metrics['psnr']),
            'ssim_delta': float(model_metrics['ssim'] - baseline_metrics['ssim']),
        }
    }


def print_metrics_summary(
    results: list[dict[str, Any]],
    title: str = "Metrics Summary",
) -> None:
    """Print formatted metrics summary.
    
    Args:
        results: List of comparison result dicts from compare_with_baseline()
        title: Title for the summary
    """
    model_mse_avg = np.mean([r['model']['mse'] for r in results])
    model_psnr_avg = np.mean([r['model']['psnr'] for r in results])
    model_ssim_avg = np.mean([r['model']['ssim'] for r in results])
    
    baseline_mse_avg = np.mean([r['baseline']['mse'] for r in results])
    baseline_psnr_avg = np.mean([r['baseline']['psnr'] for r in results])
    baseline_ssim_avg = np.mean([r['baseline']['ssim'] for r in results])
    
    mse_improvement = (baseline_mse_avg - model_mse_avg) / (baseline_mse_avg + 1e-10) * 100
    
    print("\n" + "="*70)
    print(f"{title} (Average over {len(results)} samples)")
    print("="*70)
    
    print(f"\n📊 Model Performance:")
    print(f"  MSE:  {model_mse_avg:.6f}")
    print(f"  PSNR: {model_psnr_avg:.2f} dB")
    print(f"  SSIM: {model_ssim_avg:.4f}")
    
    print(f"\n📊 Baseline (Gaussian Blur) Performance:")
    print(f"  MSE:  {baseline_mse_avg:.6f}")
    print(f"  PSNR: {baseline_psnr_avg:.2f} dB")
    print(f"  SSIM: {baseline_ssim_avg:.4f}")
    
    print(f"\n🎯 Improvement vs Baseline:")
    print(f"  MSE improvement: {mse_improvement:+.1f}%")
    print(f"  PSNR delta: {model_psnr_avg - baseline_psnr_avg:+.2f} dB")
    print(f"  SSIM delta: {model_ssim_avg - baseline_ssim_avg:+.4f}")
    
    if mse_improvement > 5:
        print(f"\n✅ Model shows SIGNIFICANT improvement over baseline!")
    elif mse_improvement > 0:
        print(f"\n➖ Model shows slight improvement over baseline.")
    else:
        print(f"\n⚠️  Model performs WORSE than baseline.")
    
    print("\n" + "="*70)
