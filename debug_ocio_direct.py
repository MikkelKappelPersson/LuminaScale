#!/usr/bin/env python
"""Verify OCIO tone curve values."""

import numpy as np
import PyOpenColorIO as ocio
from pathlib import Path

config_path = Path("config/aces/studio-config.ocio")

# Load OCIO config
config = ocio.Config.CreateFromFile(str(config_path))

# Get the processor - must match exactly what we're using in PyTorch
processor = config.getProcessor(
    "ACES2065-1",
    "sRGB - Display",
    "ACES 2.0 - SDR 100 nits (Rec.709)",
    ocio.TRANSFORM_DIR_FORWARD
)

cpu_processor = processor.getDefaultCPUProcessor()

print("--- Testing OCIO processor directly ---")
print("Testing tone curve via OCIO CPU processor:\n")

# Test the same values we used in the LUT debug
test_values = np.array([
    [0.5, 0.5, 0.5, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 1.0],
    [3.0, 3.0, 3.0, 1.0],
    [4.0, 4.0, 4.0, 1.0],
    [6.0, 6.0, 6.0, 1.0],
    [8.0, 8.0, 8.0, 1.0],
], dtype=np.float32)

print("Input ACES (unbounded) → OCIO StudioDisplay output:")
for val in test_values:
    # Create a 1x1 image
    img = np.array([[val]], dtype=np.float32).reshape(1, 1, 4)
    cpu_processor.applyRGBA(img)
    output = img[0, 0, :3]
    print(f"  {val[:3]} → {output}")

print("\n--- Checking if ApplyRGBA format is correct ---")
# Try different image formats to see what works
test_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# Method 1: 1x1 image
img1 = np.array([[test_val]], dtype=np.float32).reshape(1, 1, 4)
cpu_processor.applyRGBA(img1)
print(f"Method 1 (1x1 reshape): {img1[0, 0, :3]}")

# Method 2: Direct array
img2 = np.array([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32)
cpu_processor.applyRGBA(img2)
print(f"Method 2 (3D array): {img2[0, 0, :3]}")

# Method 3: 2D array (H=1, W=1)
img3 = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32).reshape(1, 4)
try:
    cpu_processor.applyRGBA(img3)
    print(f"Method 3 (1D→2D): {img3[0, :3]}")
except Exception as e:
    print(f"Method 3 failed: {e}")
