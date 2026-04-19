"""Loss functions for dequantization and bit-depth expansion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unmasked L1 (MAE) loss - more robust to outliers.
    
    L1 loss provides sharper transitions and can be more effective than L2
    for dequantization tasks where we want to preserve edges and avoid
    over-smoothing. Less sensitive to outliers compared to L2.
    """
    return F.l1_loss(pred.float(), target.float())


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unmasked L2 (MSE) loss - default loss function.
    
    Provides training signal across the full tonal range without masking.
    This is superior to masked L2 as it trains the model on all pixel values,
    including the extremes where dequantization is most visible.
    """
    return F.mse_loss(pred.float(), target.float())


def charbonnier_loss(
    img: torch.Tensor, reduction: str = "mean", eps: float = 1e-3
) -> torch.Tensor:
    """Compute Total Variation using Charbonnier (smooth L1) loss.
    
    Uses Charbonnier instead of hard L1 abs. This penalizes small gradients 
    heavily (quantization artifacts) but allows large gradients (real edges) 
    without over-smoothing.
    
    Charbonnier: sqrt(x² + eps²) is smooth everywhere and differentiable,
    unlike abs(x) which has a sharp corner at x=0.
    
    Args:
        img: Image tensor [B, C, H, W]
        reduction: 'mean' or 'sum'
        eps: Smoothing parameter (smaller = more like L1, larger = more like L2)
    
    Returns:
        Charbonnier total variation loss
    """
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    
    # Charbonnier: sqrt(x² + eps²) - smooth approximation to |x|
    # Gentler on large values (preserves edges), harsh on small values (removes banding)
    dy_smooth = torch.sqrt(dy**2 + eps**2)
    dx_smooth = torch.sqrt(dx**2 + eps**2)
    
    if reduction == "mean":
        return (dy_smooth.mean() + dx_smooth.mean()) / 2.0
    else:
        return (dy_smooth.sum() + dx_smooth.sum()) / 2.0


def total_variation_loss(
    img: torch.Tensor, reduction: str = "mean", variant: str = "huber"
) -> torch.Tensor:
    """Compute standard Total Variation (TV) loss - baseline for gradient smoothing.
    
    Standard TV penalizes all gradients uniformly across the image. Unlike
    edge_aware_smoothing_loss, this has no notion of target image structure.
    
    Variants:
    - 'l1': Sum of absolute gradients (aggressive smoothing, can over-blur edges)
    - 'l2': Sum of gradient magnitudes (gentler, better edge preservation)
    - 'charbonnier': Smooth approximation using sqrt(x² + eps²) - differentiable everywhere
    - 'huber': True Huber loss (smooth near 0, absolute elsewhere) - threshold at delta
    
    Args:
        img: Image tensor [B, C, H, W]
        reduction: 'mean' or 'sum'
        variant: 'l1', 'l2', 'charbonnier', or 'huber'
    
    Returns:
        Total variation loss
    """
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    
    if variant == "l1":
        # L1 TV: sum of absolute gradients
        tv = torch.abs(dy).mean() + torch.abs(dx).mean()
    elif variant == "l2":
        # L2 TV: sum of gradient magnitudes (Frobenius norm per pixel)
        # sqrt(dy² + dx²) computed at each location
        tv = (torch.sqrt(dy**2 + dx**2)).mean()
    elif variant == "charbonnier":
        # Charbonnier TV: smooth approximation using sqrt(x² + eps²)
        # Differentiable everywhere, closer to L1 than L2
        eps = 1e-3
        tv = (torch.sqrt(dy**2 + eps**2) + torch.sqrt(dx**2 + eps**2)).mean() / 2.0
    elif variant == "huber":
        # Huber TV: true Huber loss - smooth near zero, L1 elsewhere
        # Combines smoothness of L2 with robustness of L1
        delta = 1.0  # Threshold where we switch from quadratic to linear
        grad_y = torch.abs(dy)
        grad_x = torch.abs(dx)
        # Huber: 0.5 * x² if |x| <= delta, else delta * (|x| - 0.5*delta)
        huber_y = torch.where(
            grad_y <= delta,
            0.5 * grad_y**2,
            delta * (grad_y - 0.5 * delta)
        )
        huber_x = torch.where(
            grad_x <= delta,
            0.5 * grad_x**2,
            delta * (grad_x - 0.5 * delta)
        )
        tv = (huber_y.mean() + huber_x.mean()) / 2.0
    else:
        raise ValueError(f"Unknown TV variant: {variant}. Use 'l1', 'l2', 'charbonnier', or 'huber'")
    
    if reduction == "sum":
        tv = tv * img.numel()
    
    return tv


def edge_aware_smoothing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Penalize inconsistency in edge structure (where gradients exist or don't).
    
    Instead of forcing gradients to match exactly, this checks if there are edges
    where they should be (in target) and no edges where there shouldn't be.
    
    This is softer than gradient matching but stronger than plain TV.
    
    Args:
        pred: Model prediction [B, C, H, W]
        target: Target image [B, C, H, W]
    
    Returns:
        Edge consistency loss
    """
    # Compute gradient magnitude (how "edgy" each location is)
    grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    
    # Where target has NO gradients (smooth regions), penalize pred more heavily
    # This forces smooth regions to stay smooth without over-constraining edges
    smooth_mask_y = (grad_target_y < 0.01).float()  # 0 = smooth region in target
    smooth_mask_x = (grad_target_x < 0.01).float()
    
    # In smooth regions, large gradients are bad
    loss_y = (smooth_mask_y * grad_pred_y).mean()
    loss_x = (smooth_mask_x * grad_pred_x).mean()
    
    return (loss_y + loss_x) / 2.0
