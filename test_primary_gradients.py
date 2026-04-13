#!/usr/bin/env python3
"""Test script to visualize all primary gradient variants."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from src.luminascale.utils.image_generator import (
    create_primary_gradients_8x21,
    create_primary_gradients_4x21,
    create_primary_gradients_2x21,
    combine_primary_gradients,
)

# Test individual variants
print("Testing individual variants...")
img_8x21 = create_primary_gradients_8x21(dtype="float32")
img_4x21 = create_primary_gradients_4x21(dtype="float32")
img_2x21 = create_primary_gradients_2x21(dtype="float32")

print(f"8x21 (Red) shape: {img_8x21.shape}")
print(f"4x21 (Green) shape: {img_4x21.shape}")
print(f"2x21 (Blue) shape: {img_2x21.shape}")

# Test combined image
print("\nTesting combined image...")
combined = combine_primary_gradients(dtype="float32")
print(f"Combined shape: {combined.shape}")

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# 8x21 variant (Red)
img_8x21_display = np.transpose(img_8x21, (1, 2, 0))
axes[0].imshow(img_8x21_display, aspect='auto')
axes[0].set_title("8x21 (Red Primary): 8 colors × 21 pixels")
axes[0].set_ylabel("Red Channel")

# 4x21 variant (Green)
img_4x21_display = np.transpose(img_4x21, (1, 2, 0))
axes[1].imshow(img_4x21_display, aspect='auto')
axes[1].set_title("4x21 (Green Primary): 4 colors × 21 pixels")
axes[1].set_ylabel("Green Channel")

# 2x21 variant (Blue)
img_2x21_display = np.transpose(img_2x21, (1, 2, 0))
axes[2].imshow(img_2x21_display, aspect='auto')
axes[2].set_title("2x21 (Blue Primary): 2 colors × 21 pixels")
axes[2].set_ylabel("Blue Channel")

# Combined
combined_display = np.transpose(combined, (1, 2, 0))
axes[3].imshow(combined_display, aspect='auto')
axes[3].set_title("Combined: 8x21 + 4x21 + 2x21 + 1px white (128×64)")
axes[3].set_xlabel("Pixel value (0-255)")
axes[3].set_ylabel("Primaries (R/G/B)")

plt.tight_layout()
plt.savefig("outputs/primary_gradients_combined.png", dpi=100, bbox_inches='tight')
print(f"\n✓ Saved combined visualization to outputs/primary_gradients_combined.png")
plt.show()
