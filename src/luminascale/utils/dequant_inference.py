"""Dequantization model inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .dequant_utils import (
    apply_gaussian_blur,
    compare_with_baseline,
    create_gaussian_kernel,
    load_dequant_model,
)


def run_dequant_inference_on_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run model inference on a batch of images.
    
    Args:
        model: Dequantization model in eval mode
        batch: Input tensor [B, 3, H, W]
        device: Device to run on
    
    Returns:
        Model output [B, 3, H, W]
    """
    batch = batch.to(device)
    
    with torch.no_grad():
        output = model(batch)
    
    return output


def run_dequant_inference_on_single_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run model inference on a single image.
    
    Args:
        model: Dequantization model in eval mode
        image: Input tensor [3, H, W]
        device: Device to run on
    
    Returns:
        Model output [3, H, W]
    """
    image_batch = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_batch)
    
    output = output.squeeze(0)
    
    return output


def infer_dataset_with_comparison(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    num_samples: int | None = None,
    use_progress_bar: bool = True,
) -> dict[str, list[Any]]:
    """Run inference on dataset samples and compute metrics vs baseline.
    
    Args:
        model: Dequantization model in eval mode
        dataset: Dataset with __getitem__ returning (ldr, hdr)
        device: Device to run on
        num_samples: Number of samples to process (None = all)
        use_progress_bar: Show progress bar
    
    Returns:
        Dict with keys 'results' (list of metrics), 'indices' (processed indices)
    """
    if num_samples is None:
        num_samples = len(dataset)
    
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    results = []
    gaussian_kernel = create_gaussian_kernel(size=5, sigma=1.0, device=device)
    
    iterator = tqdm(indices, desc="Inference", disable=not use_progress_bar)
    
    for idx in iterator:
        ldr, hdr = dataset[idx]
        
        # Ensure batch dim
        if ldr.dim() == 3:
            ldr = ldr.unsqueeze(0)
        if hdr.dim() == 3:
            hdr = hdr.unsqueeze(0)
        
        # Run model inference
        model_output = run_dequant_inference_on_batch(model, ldr, device)
        model_output = model_output.squeeze(0)
        
        # Apply baseline blur
        ldr_squeezed = ldr.squeeze(0)
        baseline_output = apply_gaussian_blur(ldr_squeezed, gaussian_kernel)
        if baseline_output.dim() == 4:
            baseline_output = baseline_output.squeeze(0)
        
        hdr_squeezed = hdr.squeeze(0)
        
        # Compute metrics
        comparison = compare_with_baseline(model_output, baseline_output, hdr_squeezed)
        results.append(comparison)
    
    return {
        'results': results,
        'indices': indices.tolist(),
    }


def save_inference_results(
    results: dict[str, list[Any]],
    output_path: str | Path,
) -> None:
    """Save inference results to a CSV file.
    
    Args:
        results: Results dict from infer_dataset_with_comparison()
        output_path: Path to save CSV
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Index', 'Model_MSE', 'Model_PSNR', 'Model_SSIM',
            'Baseline_MSE', 'Baseline_PSNR', 'Baseline_SSIM',
            'MSE_Improvement_%', 'PSNR_Delta_dB', 'SSIM_Delta'
        ])
        
        for idx, res in zip(results['indices'], results['results']):
            model = res['model']
            baseline = res['baseline']
            improvement = res['improvement']
            
            writer.writerow([
                idx,
                model['mse'], model['psnr'], model['ssim'],
                baseline['mse'], baseline['psnr'], baseline['ssim'],
                improvement['mse_pct'], improvement['psnr_delta'], improvement['ssim_delta'],
            ])
    
    print(f"✅ Results saved to {output_path}")
