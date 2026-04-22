"""Debug script to visualize the loss mask for synthetic gradients."""

import sys
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from luminascale.utils.image_generator import create_primary_gradients, quantize_to_8bit, create_reference_gradients

def visualize_mask():
    # 1. Generate synthetic gradients
    width, height = 512, 128
    ref_32f = create_reference_gradients(width=width, height=height, block_width=8)
    # Convert to torch [C, H, W]
    ref_tensor = torch.from_numpy(ref_32f).float().unsqueeze(0) # [1, 3, H, W]
    
    # 2. Compute gradients of the reference (ground truth)
    # This is what we would use in a loss function
    dy = torch.abs(ref_tensor[:, :, 1:, :] - ref_tensor[:, :, :-1, :])
    dx = torch.abs(ref_tensor[:, :, :, 1:] - ref_tensor[:, :, :, :-1])
    
    # Pad back to original size to align with image for visualization
    grad_y = F.pad(dy, (0, 0, 0, 1))
    grad_x = F.pad(dx, (0, 1, 0, 0))
    grad_mag = grad_y + grad_x # Simple magnitude proxy
    
    # 3. Create different mask strategies
    thresholds = [0.001, 0.0005, 0.0001]
    alphas = [500.0, 1000.0, 2000.0]
    
    fig, axes = plt.subplots(len(thresholds) + len(alphas) + 2, 1, figsize=(12, 18))
    
    # Plot original image
    axes[0].imshow(ref_32f.transpose(1, 2, 0))
    axes[0].set_title("32-bit Reference Gradient")
    axes[0].axis('off')
    
    # Plot Gradient Magnitude (sum of X and Y)
    grad_img = grad_mag[0].mean(dim=0).numpy()
    im = axes[1].imshow(grad_img, cmap='hot')
    axes[1].set_title("Gradient Magnitude (Reference)")
    plt.colorbar(im, ax=axes[1])
    axes[1].axis('off')
    
    # Plot Hard Threshold masks
    for i, thresh in enumerate(thresholds):
        mask = (grad_mag[0].mean(dim=0) < thresh).float().numpy()
        axes[i+2].imshow(mask, cmap='gray')
        axes[i+2].set_title(f"Hard Mask (Thresh < {thresh}) - White = Smooth Zone")
        axes[i+2].axis('off')

    # Plot Soft (Exponential) masks
    # Formula: exp(-grad * alpha)
    for i, alpha in enumerate(alphas):
        idx = i + len(thresholds) + 2
        soft_mask = torch.exp(-grad_mag[0].mean(dim=0) * alpha).numpy()
        axes[idx].imshow(soft_mask, cmap='gray')
        axes[idx].set_title(f"Soft Mask (exp(-grad * {alpha})) - White = Smooth Zone")
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = Path("visualisations/loss_mask_debug_v2.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"Mask visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_mask()
