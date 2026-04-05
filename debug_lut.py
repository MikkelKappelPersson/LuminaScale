#!/usr/bin/env python
"""Debug script to verify LUT extraction and usage."""

import torch
import numpy as np
from pathlib import Path
from src.luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
import OpenImageIO as oiio

# Load test image
test_image_path = Path("dataset/temp/aces/1000_0.exr")
if not test_image_path.exists():
    print(f"Test image not found: {test_image_path}")
    exit(1)

spec = oiio.ImageInput.open(str(test_image_path))
img = spec.read_image("float")  # Read as float32
spec.close()

# Convert to torch and move to correct channel order (HWC → CHW → HWC)
img_np = np.array(img, dtype=np.float32)  # Shape: (H, W, C)
print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
print(f"Image range: [{img_np.min():.4f}, {img_np.max():.4f}]")

# Select a small patch for debugging
patch = img_np[100:110, 100:110, :3]  # 10x10 RGB patch
print(f"\nPatch shape: {patch.shape}")
print(f"Patch min/max: [{patch.min():.4f}, {patch.max():.4f}]")
print(f"Sample patch values:\n{patch[0, :3]}")

# Test with LUT disabled (analytical)
print("\n--- Testing with LUT disabled (analytical) ---")
transformer_analytical = ACESColorTransformer(use_lut=False)
patch_torch = torch.from_numpy(patch).to('cuda')
output_analytical_torch, _ = transformer_analytical(patch_torch)
output_analytical = output_analytical_torch.cpu().numpy()

print(f"Output shape: {output_analytical.shape}")
print(f"Output range: [{output_analytical.min():.4f}, {output_analytical.max():.4f}]")
print(f"Sample output (analytical):\n{output_analytical[0, :3]}")

# Test with LUT enabled
print("\n--- Testing with LUT enabled ---")
transformer_lut = ACESColorTransformer(use_lut=True)
output_lut_torch, _ = transformer_lut(patch_torch)
output_lut = output_lut_torch.cpu().numpy()

print(f"Output shape: {output_lut.shape}")
print(f"Output range: [{output_lut.min():.4f}, {output_lut.max():.4f}]")
print(f"Sample output (LUT):\n{output_lut[0, :3]}")

# Compare outputs
diff = np.abs(output_analytical - output_lut)
print(f"\n--- Comparison ---")
print(f"Max difference: {diff.max():.4f}")
print(f"Mean difference: {diff.mean():.4f}")
print(f"Differences are {'IDENTICAL' if diff.max() < 1e-5 else 'DIFFERENT'}")

# Check if LUT was initialized
print(f"\n--- LUT Status ---")
print(f"Transformer analytical use_lut: {transformer_analytical.use_lut}")
print(f"Transformer LUT use_lut: {transformer_lut.use_lut}")
print(f"Transformer LUT has lut_interpolator: {transformer_lut.lut_interpolator is not None}")
if transformer_lut.lut_interpolator is not None:
    print(f"LUT shape: {transformer_lut.lut_interpolator.lut_3d.shape}")
    print(f"LUT range: [{transformer_lut.lut_interpolator.lut_3d.min():.4f}, {transformer_lut.lut_interpolator.lut_3d.max():.4f}]")
