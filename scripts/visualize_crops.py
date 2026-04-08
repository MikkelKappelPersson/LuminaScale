"""Visualize crops from WebDataset to debug transform issues."""

import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from luminascale.data.wds_dataset import LuminaScaleWebDataset
from luminascale.utils.dataset_pair_generator import DatasetPairGenerator
from luminascale.training.dequantization_trainer import exposure_mask

# Load config
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="wds")
def visualize(cfg: DictConfig):
    """Load dataset, decode first image, generate crops, visualize."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = LuminaScaleWebDataset(
        shard_path=cfg.shard_path,
        batch_size=1,
        shuffle_buffer=10,
        is_training=True,
        patches_per_image=1,  # No repeat in dataset
    )
    
    # Get loader and first batch
    print("Getting WebLoader...")
    loader = dataset.get_loader(num_workers=0)  # Single worker for determinism
    print("Getting first batch...")
    batch = next(iter(loader))
    print(f"Batch type: {type(batch)}, length: {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
    
    # Decode using DatasetPairGenerator
    decoder = DatasetPairGenerator(device=device)
    
    if isinstance(batch[0], list) and len(batch[0]) > 0 and isinstance(batch[0][0], bytes):
        print("WebDataset batch detected (bytes)")
        exr_bytes = batch[0]
        
        print(f"Decoding {len(exr_bytes)} EXR files...")
        x_full, y_full = decoder.generate_batch_from_bytes(exr_bytes)
        
        print(f"Full image shapes: x={x_full.shape}, y={y_full.shape}")
        print(f"Full image dtypes: x={x_full.dtype}, y={y_full.dtype}")
        print(f"Full image ranges: x=[{x_full.min():.4f}, {x_full.max():.4f}], y=[{y_full.min():.4f}, {y_full.max():.4f}]")
        
        # Unbatch
        if x_full.ndim == 4:
            x_full = x_full[0]
            y_full = y_full[0]
        
        print(f"Unbatched shapes: x={x_full.shape}, y={y_full.shape}")
        
        # Generate 4 random crops
        num_crops = 4
        crop_size = 512
        crops_x = []
        crops_y = []
        masks = []
        
        C, H, W = x_full.shape
        print(f"Image size: {H}x{W}")
        
        for i in range(num_crops):
            if H <= crop_size or W <= crop_size:
                x_crop = x_full
                y_crop = y_full
            else:
                top = torch.randint(0, H - crop_size + 1, (1,)).item()
                left = torch.randint(0, W - crop_size + 1, (1,)).item()
                x_crop = x_full[:, top:top+crop_size, left:left+crop_size]
                y_crop = y_full[:, top:top+crop_size, left:left+crop_size]
            
            x_crop = x_crop.unsqueeze(0)
            y_crop = y_crop.unsqueeze(0)
            
            mask = exposure_mask(y_crop)
            
            print(f"\nCrop {i}:")
            print(f"  x: shape={x_crop.shape}, dtype={x_crop.dtype}, range=[{x_crop.min():.4f}, {x_crop.max():.4f}]")
            print(f"  y: shape={y_crop.shape}, dtype={y_crop.dtype}, range=[{y_crop.min():.4f}, {y_crop.max():.4f}]")
            print(f"  mask: shape={mask.shape}, dtype={mask.dtype}, active pixels={mask.sum().item()}/{mask.numel()} ({100*mask.sum().item()/mask.numel():.2f}%)")
            
            crops_x.append(x_crop)
            crops_y.append(y_crop)
            masks.append(mask)
        
        # Convert to uint8 for visualization (0-255 range)
        def to_uint8(tensor):
            """Convert tensor to uint8 for visualization."""
            if tensor.dtype == torch.uint8:
                return tensor
            tensor = tensor.float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor * 255).byte()
        
        # Plot
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle("Dataset Crops Visualization", fontsize=16)
        
        for i in range(num_crops):
            # x (input)
            x_viz = to_uint8(crops_x[i][0].cpu()).permute(1, 2, 0)
            axes[i, 0].imshow(x_viz)
            axes[i, 0].set_title(f"Crop {i} - Input X")
            axes[i, 0].axis("off")
            
            # y (target)
            y_viz = to_uint8(crops_y[i][0].cpu()).permute(1, 2, 0)
            axes[i, 1].imshow(y_viz)
            axes[i, 1].set_title(f"Crop {i} - Target Y")
            axes[i, 1].axis("off")
            
            # mask
            mask_viz = masks[i][0, 0].cpu()
            axes[i, 2].imshow(mask_viz, cmap="gray")
            active = masks[i].sum().item()
            total = masks[i].numel()
            axes[i, 2].set_title(f"Crop {i} - Mask\n({active}/{total} active)")
            axes[i, 2].axis("off")
        
        plt.tight_layout()
        output_path = Path("outputs/crop_visualization.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100)
        print(f"\nSaved visualization to {output_path}")
        plt.close()


if __name__ == "__main__":
    visualize()
