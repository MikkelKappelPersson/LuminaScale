"""Visualize validation dataset pairs (8-bit quantized vs 32-bit reference).

This script loads 10 images from the validation shards, processes them through
the same pipeline as training, and creates side-by-side visualizations showing:
- 8-bit quantized input
- 32-bit smooth reference (target)
- Both at normal and 25x contrast boost (to reveal quantization artifacts)

Usage:
    python visualisations/validate_training_pairs.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
import webdataset as wds
from matplotlib.gridspec import GridSpec

from luminascale.utils.dataset_pair_generator import DatasetPairGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_validation_samples(val_shard_path: str, num_samples: int = 10) -> list[bytes]:
    """Load raw EXR bytes from validation shards.
    
    Args:
        val_shard_path: Path to validation shards (e.g., "dataset/temp/shards/val")
        num_samples: Number of samples to load
    
    Returns:
        List of raw EXR bytes
    """
    logger.info(f"Loading {num_samples} validation samples from {val_shard_path}")
    
    exr_bytes_list = []
    
    # Open validation shards
    shard_path = Path(project_root) / val_shard_path
    pattern = str(shard_path / "val-{000000..999999}.tar")
    
    dataset = wds.WebDataset(pattern).decode().to_tuple("exr")
    
    # Iterate and collect samples
    for idx, (exr_data,) in enumerate(dataset):
        if idx >= num_samples:
            break
        
        # exr_data is already bytes from .decode()
        if isinstance(exr_data, bytes):
            exr_bytes_list.append(exr_data)
        else:
            logger.warning(f"Sample {idx}: Expected bytes, got {type(exr_data)}")
        
        if (idx + 1) % 5 == 0:
            logger.info(f"  Loaded {idx + 1}/{num_samples} samples")
    
    logger.info(f"✓ Loaded {len(exr_bytes_list)} validation samples")
    return exr_bytes_list


def create_visualization(
    device: torch.device,
    exr_bytes_list: list[bytes],
    output_path: Path,
) -> None:
    """Create side-by-side visualizations of training pairs.
    
    Args:
        device: CUDA device
        exr_bytes_list: List of raw EXR bytes
        output_path: Path to save visualization
    """
    logger.info("Processing samples through training pipeline...")
    
    # Initialize pair generator (same as training)
    pair_generator = DatasetPairGenerator(device=device)
    
    # Process all samples in one batch
    input_8u_batch, target_32f_batch, timing_breakdown = pair_generator.generate_batch_from_bytes(
        exr_bytes_list, 
        crop_size=1024, 
        bit_crunch_contrast_min=1.0,
        bit_crunch_contrast_max=20.0,
    )
    
    logger.info(f"✓ Processed {len(input_8u_batch)} samples")
    logger.debug(f"Timing breakdown: {timing_breakdown}")
    
    # Extract and display per-image crunch factors to validate randomization
    crunch_factors = timing_breakdown.get("crunch_factors", [])
    if crunch_factors:
        logger.info(f"Bit-crunch factors per image: {[f'{f:.2f}' for f in crunch_factors]}")
        if len(set(f"{f:.2f}" for f in crunch_factors)) == 1:
            logger.warning("⚠ All images have the SAME crunch factor (randomization may not be working)")
        else:
            logger.info(f"✓ Randomization confirmed: factors vary across images")
    
    # Create grid visualization
    num_samples = len(input_8u_batch)
    fig = plt.figure(figsize=(22, 4 * num_samples))
    gs = GridSpec(num_samples, 5, figure=fig, hspace=0.3, wspace=0.15)
    
    contrast_factor = 2.0
    crunch_factors = timing_breakdown.get("crunch_factors", [1.0] * num_samples)
    
    for sample_idx in range(num_samples):
        # Extract sample
        input_8u = input_8u_batch[sample_idx]  # [3, 512, 512]
        target_32f = target_32f_batch[sample_idx]  # [3, 512, 512]
        
        # Convert to numpy [H, W, 3]
        input_np = input_8u.permute(1, 2, 0).cpu().numpy()
        target_np = target_32f.permute(1, 2, 0).cpu().numpy()
        
        # Apply contrast boost to reveal banding
        input_boosted = np.clip((input_np - 0.5) * contrast_factor + 0.5, 0, 1)
        target_boosted = np.clip((target_np - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Calculate absolute difference (grayscale)
        diff_np = np.abs(input_np - target_np)
        diff_gray = np.mean(diff_np, axis=2)  # Average across channels for grayscale
        
        # Get crunch factor for this sample (for title display)
        crunch = crunch_factors[sample_idx] if sample_idx < len(crunch_factors) else 1.0
        
        # Plot row: input | input contrast | difference | target | target contrast
        ax1 = fig.add_subplot(gs[sample_idx, 0])
        ax1.imshow(input_np, cmap=None)
        ax1.set_title(f"Sample {sample_idx}: Input (8-bit)\nCrunch={crunch:.2f}")
        ax1.axis("off")
        
        ax2 = fig.add_subplot(gs[sample_idx, 1])
        ax2.imshow(input_boosted, cmap=None)
        ax2.set_title(f"Input ({contrast_factor}x contrast)")
        ax2.axis("off")
        
        ax3 = fig.add_subplot(gs[sample_idx, 2])
        im = ax3.imshow(diff_gray, cmap="hot", vmin=0, vmax=0.1)
        ax3.set_title(f"Difference (abs)")
        ax3.axis("off")
        
        ax4 = fig.add_subplot(gs[sample_idx, 3])
        ax4.imshow(target_np, cmap=None)
        ax4.set_title(f"Target (32-bit ref)")
        ax4.axis("off")
        
        ax5 = fig.add_subplot(gs[sample_idx, 4])
        ax5.imshow(target_boosted, cmap=None)
        ax5.set_title(f"Target ({contrast_factor}x contrast)")
        ax5.axis("off")
    
    # Save figure
    fig.suptitle(
        "Validation Dataset: Input vs Target Reference with Difference Analysis\n"
        "Left: 8-bit quantized input | Center: Absolute difference (shows training signal) | Right: 32-bit reference",
        fontsize=14,
        y=0.995
    )
    
    plt.savefig(output_path, dpi=72, bbox_inches="tight")
    logger.info(f"✓ Saved visualization to {output_path}")
    plt.close()


def main() -> None:
    """Main entry point."""
    output_dir = Path(__file__).parent
    output_path = output_dir / "validation_pairs_comparison.png"
    
    # Validate shards directory exists
    val_shard_path = "dataset/temp/shards/val"
    val_dir = project_root / val_shard_path
    if not val_dir.exists():
        logger.error(f"Validation shards directory not found: {val_dir}")
        logger.info(f"Please ensure training data exists at {val_dir}")
        sys.exit(1)
    
    # Load validation samples
    exr_bytes_list = load_validation_samples(
        val_shard_path=val_shard_path,
        num_samples=10
    )
    
    if not exr_bytes_list:
        logger.error("Failed to load any validation samples")
        sys.exit(1)
    
    # Setup CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create visualization
    create_visualization(device, exr_bytes_list, output_path)
    
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
