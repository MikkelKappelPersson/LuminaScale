"""Adaptive 3D LUT bank and trilinear interpolation for LuminaScale.

Based on LLF-LUT (Zeng et al./Wang et al.) implementation of LUT.py.
Adapted for LuminaScale ACES mapper using PyTorch's native grid_sample for portability.

Mandatory Attribution: Based on LLF-LUT (Zeng et al./Wang et al.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TriLinearPytorch(nn.Module):
    """Trilinear interpolation using PyTorch grid_sample.
    
    This replaces the custom CUDA extension in LLF-LUT for better portability
    while maintaining performance.
    """
    def __init__(self):
        super().__init__()

    def forward(self, lut: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lut: 3D LUT parameter of shape [3, dim, dim, dim]
            img: Input image of shape [B, 3, H, W]
            
        Returns:
            Result of shape [B, 3, H, W]
        """
        # grid_sample expects input (N, C, D, H, W) and grid (N, D, H, W, 3)
        # We treat the LUT as the input (N, 3, dim, dim, dim)
        # and the image as the coordinates (N, 1, H, W, 3)
        
        B, C, H, W = img.shape
        dim = lut.shape[-1]
        
        # Scale image to [-1, 1] for grid_sample
        # Assuming img is in range [0, 1]
        grid = img.permute(0, 2, 3, 1).unsqueeze(1) # [B, 1, H, W, 3]
        grid = (grid * 2.0) - 1.0
        
        # Add batch dimension to LUT
        lut_input = lut.unsqueeze(0).expand(B, -1, -1, -1, -1) # [B, 3, dim, dim, dim]
        
        # Perform 3D sampling
        # grid_sample mode='bilinear' on 5D input performs trilinear interpolation
        result = F.grid_sample(
            lut_input, 
            grid, 
            mode="bilinear", 
            padding_mode="border", 
            align_corners=True
        ) # [B, 3, 1, H, W]
        
        return result.squeeze(2) # [B, 3, H, W]


class Adaptive3DLUT(nn.Module):
    def __init__(self, dim: int = 33):
        super().__init__()
        self.dim = dim
        
        # Initialize with Identity LUT
        # Instead of reading from file, we generate it mathematically
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    buffer[0, i, j, k] = i / (dim - 1)
                    buffer[1, i, j, k] = j / (dim - 1)
                    buffer[2, i, j, k] = k / (dim - 1)
                    
        self.lut = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.trilinear = TriLinearPytorch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trilinear(self.lut, x)


class Zero3DLUT(nn.Module):
    def __init__(self, dim: int = 33):
        super().__init__()
        self.dim = dim
        self.lut = nn.Parameter(torch.zeros(3, dim, dim, dim).requires_grad_(True))
        self.trilinear = TriLinearPytorch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trilinear(self.lut, x)
