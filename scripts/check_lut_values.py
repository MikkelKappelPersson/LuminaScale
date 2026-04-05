#!/usr/bin/env python3
"""Check if the LUT is being sampled and used correctly."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path('.') / 'src'))

# Load the transformer and check the actual LUT values
from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer

transformer = ACESColorTransformer(device='cpu', use_lut=True)
lut_3d = transformer.lut_interpolator.lut_3d  # Access the actual LUT

print(f"LUT shape: {lut_3d.shape}")
print(f"LUT data type: {lut_3d.dtype}")
print(f"LUT value ranges:")
print(f"  R: [{lut_3d[..., 0].min():.6f}, {lut_3d[..., 0].max():.6f}]")
print(f"  G: [{lut_3d[..., 1].min():.6f}, {lut_3d[..., 1].max():.6f}]")
print(f"  B: [{lut_3d[..., 2].min():.6f}, {lut_3d[..., 2].max():.6f}]")

# Check specific LUT corners
lut_size = lut_3d.shape[0]
print(f"\nLUT corner values (size={lut_size}):")
print(f"  LUT[0,0,0] = {lut_3d[0, 0, 0]}")  # (0,0,0)
print(f"  LUT[-1,-1,-1] = {lut_3d[-1, -1, -1]}")  # (8,8,8)
print(f"  LUT[127,0,0] = {lut_3d[-1, 0, 0]}")  # (8,0,0) - pure high red
print(f" LUT[0,127,0] = {lut_3d[0, -1, 0]}")  # (0,8,0) - pure high green
print(f"  LUT[0,0,127] = {lut_3d[0, 0, -1]}")  # (0,0,8) - pure high blue

# Compare with direct OCIO
print(f"\nDirect OCIO for comparison:")
import PyOpenColorIO as ocio
config = ocio.Config.CreateFromFile(str(Path('config/aces/studio-config.ocio')))
processor = config.getProcessor("ACES2065-1", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", ocio.TRANSFORM_DIR_FORWARD)
cpu_proc = processor.getDefaultCPUProcessor()

test_cases = [
    (0.0, 0.0, 0.0, "(0,0,0)"),
    (8.0, 8.0, 8.0, "(8,8,8)"),
    (8.0, 0.0, 0.0, "(8,0,0)"),
    (0.0, 8.0, 0.0, "(0,8,0)"),
    (0.0, 0.0, 8.0, "(0,0,8)"),
]

for r, g, b, label in test_cases:
    img = np.array([[[[r, g, b, 1.0]]]], dtype=np.float32).reshape(1, 1, 4)
    cpu_proc.applyRGBA(img)
    print(f"  OCIO {label} = [{img[0,0,0]:.6f}, {img[0,0,1]:.6f}, {img[0,0,2]:.6f}]")

# Now test a real image value to see where it's being looked up
print(f"\nTest with real image pixel (R=5.9805, G=6.0156, B=5.5156):")
# Direct OCIO
img = np.array([[[[5.9805, 6.0156, 5.5156, 1.0]]]], dtype=np.float32).reshape(1, 1, 4)
cpu_proc.applyRGBA(img)
ocio_out = img[0, 0, :3].copy()
print(f"  OCIO output: [{ocio_out[0]:.6f}, {ocio_out[1]:.6f}, {ocio_out[2]:.6f}]")

# PyTorch LUT lookup
aces_val = torch.tensor([[[5.9805, 6.0156, 5.5156]]], dtype=torch.float32)
with torch.no_grad():
    pytorch_out = transformer.aces_to_srgb_32f(aces_val).numpy()[0, 0]
print(f"  PyTorch output: [{pytorch_out[0]:.6f}, {pytorch_out[1]:.6f}, {pytorch_out[2]:.6f}]")
print(f"  Difference: [{pytorch_out[0]-ocio_out[0]:.6f}, {pytorch_out[1]-ocio_out[1]:.6f}, {pytorch_out[2]-ocio_out[2]:.6f}]")
