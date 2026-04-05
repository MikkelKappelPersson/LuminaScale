#!/usr/bin/env python3
"""Diagnostic tool to find source of visual saturation difference."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path('.') / 'src'))
sys.path.insert(0, str(Path('.') / 'scripts'))

import importlib.util
spec = importlib.util.spec_from_file_location('benchmark_pytorch_vs_ocio', Path('.') / 'scripts' / 'benchmark_pytorch_vs_ocio.py')
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)

import logging
logging.basicConfig(level=logging.WARNING)

# Test image
image_path = Path('dataset/temp/aces') / '1000_0.exr'
print(f"Analyzing: {image_path.name}\n")

# Load
aces_image = benchmark_module.load_aces_image(image_path)
print(f"Input ACES shape: {aces_image.shape}")
print(f"Input ACES range: [{aces_image.min():.4f}, {aces_image.max():.4f}]")
print(f"Input ACES per-channel ranges:")
for c in range(3):
    print(f"  Ch{c}: [{aces_image[..., c].min():.4f}, {aces_image[..., c].max():.4f}]")

# OCIO
ocio_result = benchmark_module.ocio_aces_to_srgb(aces_image)
print(f"\nOCIO output range: [{ocio_result.min():.6f}, {ocio_result.max():.6f}]")
print(f"OCIO per-channel ranges:")
for c in range(3):
    print(f"  Ch{c}: [{ocio_result[..., c].min():.6f}, {ocio_result[..., c].max():.6f}]")
print(f"OCIO per-channel means: {ocio_result.mean(axis=(0,1))}")

# PyTorch
from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
transformer = ACESColorTransformer(device='cpu', use_lut=True)
aces_tensor = torch.from_numpy(aces_image).to('cpu')
with torch.no_grad():
    pytorch_result_torch, _ = transformer(aces_tensor)
pytorch_result = pytorch_result_torch.cpu().numpy()

print(f"\nPyTorch output range: [{pytorch_result.min():.6f}, {pytorch_result.max():.6f}]")
print(f"PyTorch per-channel ranges:")
for c in range(3):
    print(f"  Ch{c}: [{pytorch_result[..., c].min():.6f}, {pytorch_result[..., c].max():.6f}]")
print(f"PyTorch per-channel means: {pytorch_result.mean(axis=(0,1))}")

# Detailed difference analysis
diff = pytorch_result - ocio_result
print(f"\nPer-channel differences (PyTorch - OCIO):")
print(f"  Ch0 (R): mean={diff[..., 0].mean():.6f}, std={diff[..., 0].std():.6f}, max={diff[..., 0].max():.6f}, min={diff[..., 0].min():.6f}")
print(f"  Ch1 (G): mean={diff[..., 1].mean():.6f}, std={diff[..., 1].std():.6f}, max={diff[..., 1].max():.6f}, min={diff[..., 1].min():.6f}")
print(f"  Ch2 (B): mean={diff[..., 2].mean():.6f}, std={diff[..., 2].std():.6f}, max={diff[..., 2].max():.6f}, min={diff[..., 2].min():.6f}")

# Check if certain regions are more saturated
print(f"\nHot pixels (top 1% brightest in OCIO):")
ocio_brightness = ocio_result.sum(axis=2)
threshold = np.percentile(ocio_brightness, 99)
hot_mask = ocio_brightness > threshold
if hot_mask.any():
    hot_ocio = ocio_result[hot_mask]
    hot_pytorch = pytorch_result[hot_mask]
    hot_diff = hot_pytorch - hot_ocio
    print(f"  OCIO hot range: [{hot_ocio.min():.6f}, {hot_ocio.max():.6f}]")
    print(f"  PyTorch hot range: [{hot_pytorch.min():.6f}, {hot_pytorch.max():.6f}]")
    print(f"  Hot pixel diff mean: {hot_diff.mean():.6f}")
    print(f"  Hot pixel per-channel diffs: R={hot_diff[..., 0].mean():.6f}, G={hot_diff[..., 1].mean():.6f}, B={hot_diff[..., 2].mean():.6f}")
