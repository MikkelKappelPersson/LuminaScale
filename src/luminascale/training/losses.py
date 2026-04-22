"""Loss functions for dequantization and bit-depth expansion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from math import ceil


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
    return_mask: bool = False,
    alpha: float = 500.0,
    mask_blur_sigma: float = 2.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Penalize inconsistency in edge structure (where gradients exist or don't).
    
    Instead of forcing gradients to match exactly, this checks if there are edges
    where they should be (in target) and no edges where there shouldn't be.
    
    This is softer than gradient matching but stronger than plain TV.
    
    Args:
        pred: Model prediction [B, C, H, W]
        target: Target image [B, C, H, W]
        return_mask: If True, returns (loss, mask) tuple for visualization
        alpha: Sensitivity of the mask. Higher = more selective for smooth areas.
        mask_blur_sigma: Gaussian sigma to smooth the mask (removes noise/grain).
    
    Returns:
        Edge consistency loss (or tuple with mask)
    """
    # 1. Compute target gradient magnitudes (only from Target/GT)
    grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # 2. Smooth the target gradients BEFORE masking to remove grain/noise
    # This prevents the mask from being "grainy" due to floating point jitter
    if mask_blur_sigma > 0:
        kernel_size = int(2 * ceil(2 * mask_blur_sigma) + 1)
        padding = kernel_size // 2
        # Simple box blur weight for stability
        blur_weight = torch.ones((1, 1, kernel_size, kernel_size), device=target.device) / (kernel_size**2)
        
        B, C, H, W = grad_target_y.shape
        grad_target_y = F.conv2d(grad_target_y.view(-1, 1, H, W), blur_weight, padding=padding).view(B, C, H, W)
        
        B, C, H, W = grad_target_x.shape
        grad_target_x = F.conv2d(grad_target_x.view(-1, 1, H, W), blur_weight, padding=padding).view(B, C, H, W)

    # 3. Soft Continuous Mask: exp(-grad * alpha)
    # This creates a weight between 0.0 (sharp edge) and 1.0 (perfectly smooth).
    smooth_mask_y = torch.exp(-grad_target_y * alpha)
    smooth_mask_x = torch.exp(-grad_target_x * alpha)
    
    # 4. Compute gradients for prediction
    grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    
    # 5. Penalize gradients only in smooth regions (where smooth_mask is high)
    # EXPERIMENT: Penalize the existence of gradients in smooth zones (Zero-Target mode)
    # Instead of abs(grad_pred - grad_target), we use just abs(grad_pred) weighted by mask.
    # This forces the model to actively flatten the output in these areas.
    # ADDITION: Use Squared L2 to penalize small gradients much harder than L1 does.
    loss_y = (smooth_mask_y * (grad_pred_y**2)).mean()
    loss_x = (smooth_mask_x * (grad_pred_x**2)).mean()
    
    loss = (loss_y + loss_x) / 2.0
    
    if return_mask:
        # Pad masks back to original size for visualization
        mask_y = F.pad(smooth_mask_y, (0, 0, 0, 1))
        mask_x = F.pad(smooth_mask_x, (0, 1, 0, 0))
        combined_mask = (mask_y + mask_x) / 2.0
        return loss, combined_mask
        
    return loss
