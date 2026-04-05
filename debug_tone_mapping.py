#!/usr/bin/env python
"""Debug LUT tone mapping to find the brightness issue."""

import torch
import numpy as np
from pathlib import Path
from src.luminascale.utils.pytorch_aces_transformer import ACESColorTransformer
import OpenImageIO as oiio

# Load test image
test_image_path = Path("dataset/temp/aces/1000_0.exr")
spec = oiio.ImageInput.open(str(test_image_path))
img = spec.read_image("float")
spec.close()

img_np = np.array(img, dtype=np.float32)
print(f"Image shape: {img_np.shape}, range: [{img_np.min():.4f}, {img_np.max():.4f}]")

# Create transformer with LUT
print("\n--- Initializing transformer with LUT ---")
transformer = ACESColorTransformer(device='cuda', use_lut=True)

# Test with a few sample values to understand tone mapping
print("\n--- Testing tone mapping with sample values ---")
test_values = torch.tensor([
    # Mid-tones
    [0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    # Dark tones
    [0.1, 0.1, 0.1],
    [0.05, 0.05, 0.05],
    # Bright tones
    [3.0, 3.0, 3.0],
    [4.0, 4.0, 4.0],
    [6.0, 6.0, 6.0],
    [8.0, 8.0, 8.0],
], dtype=torch.float32).to('cuda')

with torch.no_grad():
    # Get intermediate values through pipeline
    # AP0 to AP1
    ap0_to_ap1 = transformer.M_AP0_TO_AP1.to('cuda')
    test_ap1 = test_values @ ap0_to_ap1.t()
    
    # Check if using LUT
    print(f"use_lut: {transformer.use_lut}")
    print(f"has lut_interpolator: {transformer.lut_interpolator is not None}")
    
    if transformer.use_lut:
        # Apply LUT directly to AP1 values
        print("\nUsing LUT for tone mapping:")
        # Need to normalize to [0, 1] for LUT lookup (which covers [0, 8])
        ap1_clamped = torch.clamp(test_ap1, 0.0, 8.0)
        ap1_normalized = ap1_clamped / 8.0
        display_lut = transformer.lut_interpolator.lookup_trilinear(ap1_normalized)
        
        print("\nInput AP1 → LUT normalized → Display output:")
        for i, val in enumerate(test_values.cpu().numpy()):
            ap1_val = test_ap1[i].cpu().numpy()
            norm_val = ap1_normalized[i].cpu().numpy()
            display_val = display_lut[i].cpu().numpy()
            print(f"  {val} → AP1: {ap1_val} → norm: {norm_val} → display: {display_val}")
    else:
        print("NOT using LUT!")
        
# Now test full pipeline on a patch
print("\n--- Testing full pipeline on image patch ---")
patch = img_np[500:510, 500:510, :3]
patch_torch = torch.from_numpy(patch).to('cuda')

with torch.no_grad():
    srgb_lut, ap1_display = transformer(patch_torch)

srgb_lut_np = srgb_lut.cpu().numpy()
ap1_display_np = ap1_display.cpu().numpy()

print(f"AP1 display range: [{ap1_display_np.min():.4f}, {ap1_display_np.max():.4f}]")
print(f"sRGB output range: [{srgb_lut_np.min():.4f}, {srgb_lut_np.max():.4f}]")

# Compare with analytical tone mapping
print("\n--- Comparing LUT vs Analytical ---")
transformer_analytical = ACESColorTransformer(device='cuda', use_lut=False)
with torch.no_grad():
    srgb_analytical, ap1_display_analytical = transformer_analytical(patch_torch)

srgb_analytical_np = srgb_analytical.cpu().numpy()
ap1_display_analytical_np = ap1_display_analytical.cpu().numpy()

print(f"Analytical sRGB range: [{srgb_analytical_np.min():.4f}, {srgb_analytical_np.max():.4f}]")

# Sample comparisons
print("\nSample pixel comparisons (sRGB):")
for i in range(min(3, len(patch))):
    for j in range(min(3, len(patch[0]))):
        lut_val = srgb_lut_np[i, j]
        analytical_val = srgb_analytical_np[i, j]
        diff = np.abs(lut_val - analytical_val)
        print(f"  [{i},{j}] LUT: {lut_val}  Analytical: {analytical_val}  Diff: {diff}")
