"""Image quality metrics for training evaluation.

Includes SSIM (structural similarity) and ΔE (perceptual color distance in ACES).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchmetrics import Metric


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute SSIM (Structural Similarity Index) between pred and target.
    
    Args:
        pred: Predicted tensor (B, C, H, W) or (C, H, W), float32, range [0, 1] or [0, 255]
        target: Ground truth tensor, same shape as pred
        data_range: Value range of input (default 1.0 for normalized, 255.0 for uint8)
    
    Returns:
        SSIM value (float), range [-1, 1] (higher is better)
    """
    try:
        # Move to CPU and convert to numpy
        if pred.is_cuda:
            pred = pred.cpu()
        if target.is_cuda:
            target = target.cpu()
        
        pred_np = pred.detach().numpy()
        target_np = target.detach().numpy()
        
        # Ensure shapes match
        if pred_np.shape != target_np.shape:
            raise ValueError(f"Shapes must match: pred {pred_np.shape} vs target {target_np.shape}")
        
        # Handle batch dimension
        if pred_np.ndim == 4:
            # (B, C, H, W) - compute SSIM per image and average
            ssim_values = []
            for b in range(pred_np.shape[0]):
                # For multi-channel, compute per-channel and average
                channel_ssims = []
                for c in range(pred_np.shape[1]):
                    s = ssim(target_np[b, c], pred_np[b, c], data_range=data_range)
                    channel_ssims.append(s)
                ssim_values.append(np.mean(channel_ssims))
            return float(np.mean(ssim_values))
        else:
            # (C, H, W) or (H, W)
            if pred_np.ndim == 3:
                # Multi-channel
                channel_ssims = []
                for c in range(pred_np.shape[0]):
                    s = ssim(target_np[c], pred_np[c], data_range=data_range)
                    channel_ssims.append(s)
                return float(np.mean(channel_ssims))
            else:
                # Single channel
                return float(ssim(target_np, pred_np, data_range=data_range))
    except Exception as e:
        raise RuntimeError(f"SSIM computation failed: pred shape {pred.shape}, target shape {target.shape}: {e}") from e


def compute_delta_e_aces(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean ΔE (CIE L*a*b* Delta E) between predictions and targets in ACES space.
    
    Computes color distance in perceptually uniform LAB color space, which approximates
    human color perception. Lower ΔE is better.
    
    Args:
        pred: Predicted tensor (B, C, H, W) or (C, H, W), float32, RGB range [0, 1]
        target: Ground truth tensor, same shape as pred, RGB range [0, 1]
    
    Returns:
        Mean ΔE value (float), perceptual color distance
    """
    # Move to CPU if needed
    if pred.is_cuda:
        pred = pred.cpu()
    if target.is_cuda:
        target = target.cpu()
    
    pred_np = pred.detach().numpy()
    target_np = target.detach().numpy()
    
    # Ensure shapes match
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Shapes must match: pred {pred_np.shape} vs target {target_np.shape}")
    
    # Ensure RGB range
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # RGB → XYZ (assuming sRGB primaries)
    # Standard sRGB primaries
    pred_xyz = _rgb_to_xyz(pred_np)
    target_xyz = _rgb_to_xyz(target_np)
    
    # XYZ → LAB
    pred_lab = _xyz_to_lab(pred_xyz)
    target_lab = _xyz_to_lab(target_xyz)
    
    # Ensure output shapes match
    if pred_lab.shape != target_lab.shape:
        raise ValueError(f"LAB shape mismatch: pred {pred_lab.shape} vs target {target_lab.shape}")
    
    # Compute ΔE (Euclidean distance in LAB space)
    # Handle both (B, C, H, W) and (C, H, W) formats
    if pred_lab.ndim == 4:  # (B, C, H, W)
        pred_lab_hwc = np.transpose(pred_lab, (0, 2, 3, 1))
        target_lab_hwc = np.transpose(target_lab, (0, 2, 3, 1))
    elif pred_lab.ndim == 3 and pred_lab.shape[0] == 3:  # (C, H, W)
        pred_lab_hwc = np.transpose(pred_lab, (1, 2, 0))
        target_lab_hwc = np.transpose(target_lab, (1, 2, 0))
    else:  # Already (..., C) format
        pred_lab_hwc = pred_lab
        target_lab_hwc = target_lab
    
    # Compute Euclidean distance in LAB space
    delta_e = np.sqrt(
        (pred_lab_hwc[..., 0] - target_lab_hwc[..., 0]) ** 2 +
        (pred_lab_hwc[..., 1] - target_lab_hwc[..., 1]) ** 2 +
        (pred_lab_hwc[..., 2] - target_lab_hwc[..., 2]) ** 2
    )
    
    return float(np.mean(delta_e))


def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (sRGB primaries) to XYZ.
    
    Args:
        rgb: RGB array, can be (B, C, H, W) or (C, H, W) or (H, W, C) or (..., 3)
        
    Returns:
        XYZ array in D65 illuminant, same shape as input
    """
    original_shape = rgb.shape
    
    # Convert (B, C, H, W) or (C, H, W) to (..., 3) format
    if rgb.ndim == 4 and rgb.shape[1] == 3:
        # (B, C, H, W) -> (B, H, W, C)
        rgb = np.transpose(rgb, (0, 2, 3, 1))
    elif rgb.ndim == 3 and rgb.shape[0] == 3:
        # (C, H, W) -> (H, W, C)
        rgb = np.transpose(rgb, (1, 2, 0))
    
    # Now rgb is (..., 3)
    # Linearize sRGB
    mask = rgb > 0.04045
    linear = np.where(mask, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    
    # Transformation matrix sRGB → XYZ (D65)
    transform = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # Apply transformation: (..., 3) @ (3, 3).T = (..., 3)
    xyz = linear @ transform.T
    
    # Reshape back to original format
    if len(original_shape) == 4 and original_shape[1] == 3:
        # Restore (B, C, H, W) from (B, H, W, C)
        xyz = np.transpose(xyz, (0, 3, 1, 2))
    elif len(original_shape) == 3 and original_shape[0] == 3:
        # Restore (C, H, W) from (H, W, C)
        xyz = np.transpose(xyz, (2, 0, 1))
    
    return xyz


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ (D65) to LAB.
    
    Args:
        xyz: XYZ array, same format as produced by _rgb_to_xyz (same shape as original input)
    
    Returns:
        LAB array, same shape as input
    """
    original_shape = xyz.shape
    
    # Convert (B, C, H, W) or (C, H, W) to (..., 3) format if needed
    if xyz.ndim == 4 and xyz.shape[1] == 3:
        # (B, C, H, W) -> (B, H, W, C)
        xyz = np.transpose(xyz, (0, 2, 3, 1))
    elif xyz.ndim == 3 and xyz.shape[0] == 3:
        # (C, H, W) -> (H, W, C)
        xyz = np.transpose(xyz, (1, 2, 0))
    
    # Now xyz is (..., 3)
    # D65 white point
    ref_white = np.array([0.95047, 1.00000, 1.08883])
    
    # Normalize by white point
    xyz_norm = xyz / ref_white
    
    # Nonlinear transformation
    epsilon = 0.008856
    kappa = 903.3
    
    f = np.where(
        xyz_norm > epsilon,
        np.power(xyz_norm, 1 / 3),
        (kappa * xyz_norm + 16) / 116
    )
    
    # LAB
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    
    lab = np.stack([L, a, b], axis=-1)
    
    # Reshape back to original format
    if len(original_shape) == 4 and original_shape[1] == 3:
        # Restore (B, C, H, W) from (B, H, W, C)
        lab = np.transpose(lab, (0, 3, 1, 2))
    elif len(original_shape) == 3 and original_shape[0] == 3:
        # Restore (C, H, W) from (H, W, C)
        lab = np.transpose(lab, (2, 0, 1))
    
    return lab


class DeltaEACES(Metric):
    """Stateful metric for ΔE (CIE L*a*b* Delta E) in ACES color space.
    
    Computes mean perceptual color distance between predictions and targets.
    Accumulates ΔE values across batches and synchronizes across distributed 
    processes (DDP) automatically.
    
    Lower ΔE is better (0 = identical, >100 = very different).
    
    Args:
        dist_sync_on_step: If True, synchronize metrics every step. Default: False
                          (sync only at epoch end for better efficiency).
    """
    
    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # State: accumulate mean ΔE values and count for final averaging
        # Using "mean" for dist_reduce_fx ensures proper distributed averaging
        self.add_state("sum_delta_e", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with batch predictions and targets.
        
        Args:
            pred: Predicted images (B, C, H, W), float32, RGB range [0, 1]
            target: Ground truth images (B, C, H, W), float32, RGB range [0, 1]
        """
        # Compute mean ΔE for this batch
        batch_delta_e = compute_delta_e_aces(pred, target)
        
        # Accumulate: convert to tensor to work with distributed training
        batch_delta_e_tensor = torch.tensor(batch_delta_e, device=self.device, dtype=torch.float32)
        
        # Add to running sum
        self.sum_delta_e += batch_delta_e_tensor
        self.count += 1
    
    def compute(self) -> torch.Tensor:
        """Compute the mean ΔE across all accumulated batches.
        
        Returns:
            Mean ΔE value (scalar tensor), computed from accumulated batches
        """
        # Avoid division by zero
        if self.count == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Return mean: sum_delta_e / count gives mean ΔE value
        return self.sum_delta_e / self.count.float()
