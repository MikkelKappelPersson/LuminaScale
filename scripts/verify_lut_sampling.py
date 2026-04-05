#!/usr/bin/env python3
"""Verify OCIO LUT sampling matches direct OCIO calls."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('.') / 'src'))

# Direct OCIO test
import PyOpenColorIO as ocio

config_path = Path('config/aces/studio-config.ocio')
config = ocio.Config.CreateFromFile(str(config_path))
processor = config.getProcessor("ACES2065-1", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", ocio.TRANSFORM_DIR_FORWARD)
cpu_processor = processor.getDefaultCPUProcessor()

# Test a few specific values
test_values = [
    (1.0, 0.0, 0.0),  # Pure red
    (0.0, 1.0, 0.0),  # Pure green
    (0.0, 0.0, 1.0),  # Pure blue
    (0.5, 0.5, 0.5),  # Grey
    (2.0, 0.0, 0.0),  # High red
    (0.0, 2.0, 0.0),  # High green
]

print("Direct OCIO sampling (per-channel):")
print("-" * 70)
print(f"{'Input R':<12} {'Input G':<12} {'Input B':<12} {'Out R':<12} {'Out G':<12} {'Out B':<12}")
print("-" * 70)

for r, g, b in test_values:
    test_img = np.array([[[[r, g, b, 1.0]]]], dtype=np.float32).reshape(1, 1, 4)
    cpu_processor.applyRGBA(test_img)
    out_r, out_g, out_b = test_img[0, 0, :3]
    print(f"{r:<12.4f} {g:<12.4f} {b:<12.4f} {out_r:<12.6f} {out_g:<12.6f} {out_b:<12.6f}")

print("\n" + "="*70)
print("Check if alpha channel affects output:")
print("="*70)

# Same input, different alpha
r, g, b = 1.0, 0.0, 0.0
for alpha in [0.5, 1.0, 2.0]:
    test_img = np.array([[[[r, g, b, alpha]]]], dtype=np.float32).reshape(1, 1, 4)
    cpu_processor.applyRGBA(test_img)
    out_r, out_g, out_b = test_img[0, 0, :3]
    print(f"Input (R={r}, alpha={alpha}): Out R={out_r:.6f}")

print("\n" + "="*70)
print("Extract LUT sample points to verify sampling:")
print("="*70)

# Get the actual LUT values that should be used
lut_size = 64
indices_to_check = [0, 8, 16, 32, 48, 63]

print(f"\nLUT values at key indices (for R=?, G=0, B=0):")
for idx in indices_to_check:
    r_val = (idx / (lut_size - 1)) * 8.0
    test_img = np.array([[[[r_val, 0.0, 0.0, 1.0]]]], dtype=np.float32).reshape(1, 1, 4)
    cpu_processor.applyRGBA(test_img)
    out_r = test_img[0, 0, 0]
    print(f"  LUT[{idx}] = OCIO({r_val:.4f}) = {out_r:.6f}")
