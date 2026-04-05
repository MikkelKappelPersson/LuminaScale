#!/usr/bin/env python3
"""Find the pixel causing saturation difference."""

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

# Load and process image
image_path = Path('dataset/temp/aces') / '1000_0.exr'
aces_image = benchmark_module.load_aces_image(image_path)

# Get outputs
ocio_result = benchmark_module.ocio_aces_to_srgb(aces_image)

from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
transformer = ACESColorTransformer(device='cpu', use_lut=True)
aces_tensor = torch.from_numpy(aces_image).to('cpu')
with torch.no_grad():
    pytorch_result_torch, _ = transformer(aces_tensor)
pytorch_result = pytorch_result_torch.cpu().numpy()

# Find the pixel with max R in PyTorch
max_r_idx = np.unravel_index(np.argmax(pytorch_result[..., 0]), pytorch_result[..., 0].shape)
print(f"Pixel with max PyTorch R ({pytorch_result[max_r_idx][0]:.6f}):")
print(f"  Location: {max_r_idx}")
print(f"  Input ACES: {aces_image[max_r_idx]}")
print(f"  PyTorch output: {pytorch_result[max_r_idx]}")

# Process this pixel through OCIO directly
img = np.array([[[[aces_image[max_r_idx][0], aces_image[max_r_idx][1], aces_image[max_r_idx][2], 1.0]]]], dtype=np.float32).reshape(1, 1, 4)
import PyOpenColorIO as ocio
config = ocio.Config.CreateFromFile(str(Path('config/aces/studio-config.ocio')))
processor = config.getProcessor("ACES2065-1", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", ocio.TRANSFORM_DIR_FORWARD)
cpu_proc = processor.getDefaultCPUProcessor()
cpu_proc.applyRGBA(img)
ocio_single = img[0, 0, :3].copy()

print(f"  OCIO direct: {ocio_single}")
print(f"  OCIO batch: {ocio_result[max_r_idx]}")

print(f"\nPyTorch - OCIO direct: {pytorch_result[max_r_idx] - ocio_single}")
print(f"PyTorch - OCIO batch: {pytorch_result[max_r_idx] - ocio_result[max_r_idx]}")

# Now find pixel with max OCIO R
ocio_max_r_idx = np.unravel_index(np.argmax(ocio_result[..., 0]), ocio_result[..., 0].shape)
print(f"\n\nPixel with max OCIO R ({ocio_result[ocio_max_r_idx][0]:.6f}):")
print(f"  Location: {ocio_max_r_idx}")
print(f"  Input ACES: {aces_image[ocio_max_r_idx]}")
print(f"  OCIO output: {ocio_result[ocio_max_r_idx]}")
print(f"  PyTorch output: {pytorch_result[ocio_max_r_idx]}")

# Process through single-pixel OCIO
img = np.array([[[[aces_image[ocio_max_r_idx][0], aces_image[ocio_max_r_idx][1], aces_image[ocio_max_r_idx][2], 1.0]]]], dtype=np.float32).reshape(1, 1, 4)
cpu_proc.applyRGBA(img)
ocio_single = img[0, 0, :3].copy()
print(f"  OCIO direct: {ocio_single}")

print(f"\nPyTorch - OCIO: {pytorch_result[ocio_max_r_idx] - ocio_result[ocio_max_r_idx]}")
