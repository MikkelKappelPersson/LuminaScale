"""Advanced loss functions for dequantization training.

Combines multiple objectives to improve quantization artifact removal:
- L2 reconstruction loss (unmasked, default)
- Total Variation loss (gradient smoothness)
- Perceptual loss (VGG feature matching)
- Edge-aware regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleL2Loss(nn.Module):
    """Simple unmasked L2 (MSE) loss for dequantization.
    
    This is the recommended default loss function that:
    - Does not mask out any pixel values
    - Provides training signal across the full tonal range
    - Especially important for capturing dequantization at extremes
    """
    
    def __init__(self) -> None:
        """Initialize simple L2 loss."""
        super().__init__()
    
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute unmasked L2 loss.
        
        Args:
            pred: Predicted image [B, C, H, W] in [0, 1]
            target: Target image [B, C, H, W] in [0, 1]
            
        Returns:
            Scalar L2 loss (MSE)
        """
        return F.mse_loss(pred, target)


class TotalVariationLoss(nn.Module):
    """Total Variation loss to encourage smooth outputs and reduce banding."""
    
    def __init__(self, weight: float = 0.1) -> None:
        """Initialize TV loss.
        
        Args:
            weight: Lambda weight for TV loss contribution
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute TV loss (sum of absolute gradients).
        
        Args:
            x: Tensor of shape [B, C, H, W]
            
        Returns:
            Scalar TV loss
        """
        # Compute gradients
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        
        # TV is sum of absolute gradients
        tv = (diff_h.sum() + diff_w.sum()) / x.numel()
        
        return self.weight * tv


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG-19 features."""
    
    def __init__(self, weight: float = 0.1, layer: str = "relu5_1") -> None:
        """Initialize perceptual loss.
        
        Args:
            weight: Lambda weight for perceptual loss
            layer: Which VGG layer to extract features from
                   Options: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        """
        super().__init__()
        self.weight = weight
        self.layer = layer
        
        # Load pre-trained VGG19
        vgg19 = torch.hub.load(
            "pytorch/vision:v0.10.0", "vgg19", pretrained=True
        )
        
        # Map layer names to indices
        layer_name_mapping = {
            "relu1_1": 1,
            "relu2_1": 6,
            "relu3_1": 11,
            "relu4_1": 20,
            "relu5_1": 29,
        }
        
        if layer not in layer_name_mapping:
            raise ValueError(f"Unknown layer: {layer}. Choose from {list(layer_name_mapping.keys())}")
        
        layer_idx = layer_name_mapping[layer]
        
        # Extract features up to the specified layer
        self.feature_extractor = nn.Sequential(
            *list(vgg19.features.children())[:layer_idx + 1]
        )
        
        # Freeze VGG weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # VGG normalization constants (ImageNet)
        self.register_buffer(
            "vgg_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "vgg_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            pred: Predicted image [B, C, H, W] in [0, 1]
            target: Target image [B, C, H, W] in [0, 1]
            
        Returns:
            Scalar perceptual loss
        """
        # Normalize for VGG
        pred_norm = (pred - self.vgg_mean) / self.vgg_std
        target_norm = (target - self.vgg_mean) / self.vgg_std
        
        # Extract features
        pred_feats = self.feature_extractor(pred_norm)
        target_feats = self.feature_extractor(target_norm)
        
        # L2 loss on features
        perc_loss = F.mse_loss(pred_feats, target_feats)
        
        return self.weight * perc_loss


class BandingAwareLoss(nn.Module):
    """Loss that specifically targets quantization banding.
    
    Combines:
    - Laplacian of Gaussian (LoG) filter to detect banding patterns
    - Emphasis on mid-tone regions where banding is most visible
    """
    
    def __init__(self, weight: float = 0.2) -> None:
        """Initialize banding-aware loss.
        
        Args:
            weight: Lambda weight for this loss
        """
        super().__init__()
        self.weight = weight
        
        # Create Laplacian of Gaussian kernel for banding detection
        # Detects rapid changes in intensity (banding edges)
        self.register_buffer(
            "log_kernel",
            torch.tensor(
                [
                    [0.0, -1.0, 0.0],
                    [-1.0, 4.0, -1.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute banding-aware loss.
        
        Args:
            pred: Predicted image [B, C, H, W] in [0, 1]
            target: Target image [B, C, H, W] in [0, 1]
            
        Returns:
            Scalar banding loss
        """
        # Convert to grayscale for analysis
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = (
            0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        )
        
        # Apply Laplacian to detect edges (including banding edges)
        pred_laplacian = F.conv2d(pred_gray, self.log_kernel, padding=1)
        target_laplacian = F.conv2d(target_gray, self.log_kernel, padding=1)
        
        # Banding loss: emphasize where target has strong gradients (banding)
        # We want pred to match target's edge structure
        banding_loss = F.mse_loss(pred_laplacian, target_laplacian)
        
        return self.weight * banding_loss


class CombinedDequantizationLoss(nn.Module):
    """Combines multiple losses for improved dequantization."""
    
    def __init__(
        self,
        use_l2: bool = True,
        use_tv: bool = True,
        use_perceptual: bool = False,
        use_banding_aware: bool = True,
        tv_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        banding_weight: float = 0.2,
    ) -> None:
        """Initialize combined loss.
        
        Args:
            use_l2: Include masked L2 reconstruction loss
            use_tv: Include total variation loss
            use_perceptual: Include VGG perceptual loss (slower on GPU)
            use_banding_aware: Include banding-specific loss
            tv_weight: Weight for TV loss
            perceptual_weight: Weight for perceptual loss
            banding_weight: Weight for banding loss
        """
        super().__init__()
        self.use_l2 = use_l2
        self.use_tv = use_tv
        self.use_perceptual = use_perceptual
        self.use_banding_aware = use_banding_aware
        
        if self.use_tv:
            self.tv_loss = TotalVariationLoss(weight=tv_weight)
        
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss(weight=perceptual_weight)
        
        if self.use_banding_aware:
            self.banding_loss = BandingAwareLoss(weight=banding_weight)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            pred: Predicted image [B, C, H, W] in [0, 1]
            target: Target image [B, C, H, W] in [0, 1]
            mask: Optional mask [B, 1, H, W] for well-exposed regions
            
        Returns:
            Dict with individual loss components and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # L2 reconstruction loss (masked)
        if self.use_l2:
            diff = (pred - target) ** 2
            if mask is not None:
                l2_loss = (diff * mask).sum() / (mask.sum() + 1e-8)
            else:
                l2_loss = diff.mean()
            losses["l2"] = l2_loss
            total_loss = total_loss + l2_loss
        
        # Total variation loss (encourages smoothness)
        if self.use_tv:
            tv_loss = self.tv_loss(pred)
            losses["tv"] = tv_loss
            total_loss = total_loss + tv_loss
        
        # Perceptual loss (feature-based)
        if self.use_perceptual:
            perc_loss = self.perceptual_loss(pred, target)
            losses["perceptual"] = perc_loss
            total_loss = total_loss + perc_loss
        
        # Banding-aware loss
        if self.use_banding_aware:
            band_loss = self.banding_loss(pred, target)
            losses["banding"] = band_loss
            total_loss = total_loss + band_loss
        
        losses["total"] = total_loss
        
        return losses
