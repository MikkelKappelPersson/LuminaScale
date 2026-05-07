"""ACES Color Space Mapper for LuminaScale.

Counterpart to DequantNet. Integrates SFT weight prediction, Adaptive 3D LUTs,
and Laplacian Local Refinement to map images to a target color space (configurable).

Mandatory Attribution: Based on LLF-LUT (Zeng et al./Wang et al.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_frequency_transformer import SpatialFrequencyTransformer
from .adaptive_3d_lut import Adaptive3DLUT, Zero3DLUT
from .local_refiner import LocalRefinementHead


class ACESMapper(nn.Module):
    """
    Top-level integrator for color space mapping (ACES-family color spaces).
    
    Maps from display-referred sRGB to a target color space (ACES2065-1, ACEScct, etc.)
    using LLF-LUT-style global LUT fitting + local Laplacian refinement.
    """
    def __init__(
        self,
        num_luts: int = 3,
        lut_dim: int = 33,
        num_lap: int = 3,
        num_residual_blocks: int = 5,
        sft_embed_dim: int = 64,  # Changed from 96 to be divisible by 16 heads (max head count)
        sft_depths: list[int] = [1, 1, 1, 1, 1, 1, 1, 1],
        sft_num_heads: list[int] = [2, 4, 8, 16, 16, 8, 4, 2],
        target_color_space: str = "ACEScct",
    ) -> None:
        super().__init__()
        self.target_color_space = target_color_space
        
        # 1. Global Fitting Head: SFT Weight Predictor
        self.sft = SpatialFrequencyTransformer(
            in_chans=3,
            embed_dim=sft_embed_dim,
            num_weights=num_luts * 2, # Dual weights for global and point logic
            depths=sft_depths,
            num_heads=sft_num_heads
        )
        
        # 2. Global Fitting Head: Basis 3D LUTs
        self.luts = nn.ModuleList()
        # Initializing first LUT as identity, others as zero-correction
        self.luts.append(Adaptive3DLUT(dim=lut_dim))
        for _ in range(num_luts - 1):
            self.luts.append(Zero3DLUT(dim=lut_dim))
            
        # 3. Local Refiner: Laplacian Engine
        self.refiner = LocalRefinementHead(
            num_residual_blocks=num_residual_blocks,
            num_lap=num_lap
        )
        
        self.num_luts = num_luts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image of shape [B, 3, H, W] in display-referred sRGB space.
            
        Returns:
            Tuple of:
              - Mapped image in target color space of shape [B, 3, H, W].
              - point_weights tensor [B, num_luts] used for LUT regularization
                (returned so the trainer can reuse them without a second SFT pass).
        """
        # 1. Predict weights via SFT on low-resolution pyramid input (LLF-LUT style)
        pyr_input = self.refiner.lap_pyramid.pyramid_decom(x)
        pyr_input_low = pyr_input[-1]
        weights = self.sft(pyr_input_low)  # [B, num_luts * 2]
        
        # Split weights for global and point-wise logic
        global_weights = weights[:, :self.num_luts]
        point_weights = weights[:, self.num_luts:]
        
        # 2. Apply LUTs to the input
        # We calculate the fused output from the LUT bank
        # As per LLF-LUT: enhanced_full = sum(weight_point_i * LUT_i(input))
        enhanced_full = 0
        for i in range(self.num_luts):
            # Applying each LUT to the full image
            # we need to consider batch dimension properly
            lut_out = self.luts[i](x) # [B, 3, H, W]
            
            # Weighted sum across batch
            w_i = point_weights[:, i].view(-1, 1, 1, 1)
            enhanced_full = enhanced_full + (w_i * lut_out)

        # In LLF-LUT, it also calculates enhanced_low from the low-res pyramid input
        # To avoid redundancy, we'll derive enhanced_low for the refiner
        # However, for simplicity in the unified module, we'll pass enhanced_full
        # and let the refiner handle the low-res extraction it needs.
        
        # 3. Local Refinement
        # The refiner uses the source image and the LUT-enhanced image to reconstruct details
        pyr_refined = self.refiner(x, enhanced_full)
        
        # Reconstruct the final image from the refined pyramid
        out = self.refiner.reconstruct(pyr_refined)
        
        # Return output and point_weights so the trainer can compute LUT regularization
        # without a second SFT forward pass (matches original LLF-LUT design).
        return out, point_weights
