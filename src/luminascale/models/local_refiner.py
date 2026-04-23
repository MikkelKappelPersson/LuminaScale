"""Local Refinement Head (PPB & Laplacian Engine) for LuminaScale.

Based on LLF-LUT (Zeng et al./Wang et al.) implementation of PPB.py.
Adapted for LuminaScale ACES mapper to handle spatially-aware radiometric refinement.

Mandatory Attribution: Based on LLF-LUT (Zeng et al./Wang et al.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


def remapping(img_gauss: torch.Tensor, img_lpr: torch.Tensor, sigma: torch.Tensor, fact: torch.Tensor, N: int = 10) -> torch.Tensor:
    """Remapping function for local refinement.
    
    Args:
        img_gauss: Gaussian version of current level
        img_lpr: Laplacian version of current level
        sigma: predicted sigma map
        fact: predicted factor map
        N: number of discretization steps
    """
    discretisation = torch.linspace(0, 1, N, device=img_lpr.device)
    discretisation_step = discretisation[1]
    for ref in discretisation:
        img_remap = fact * (img_lpr - ref) * torch.exp(
            -(img_lpr - ref) * (img_lpr - ref) * (2 * sigma * sigma)
        )
        img_lpr = img_lpr + (torch.abs(img_gauss - ref) < discretisation_step) * img_remap * (
            1 - torch.abs(img_gauss - ref) / discretisation_step
        )
    return img_lpr


class LapPyramidConv(nn.Module):
    def __init__(self, num_high: int = 3, channels: int = 3):
        super().__init__()
        self.num_high = num_high
        self.register_buffer("kernel", self._gauss_kernel(channels))

    def _gauss_kernel(self, channels: int = 3) -> torch.Tensor:
        kernel = torch.tensor([
            [1., 4., 6., 4., 1.],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.]
        ])
        kernel /= 256.
        return kernel.repeat(channels, 1, 1, 1)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, ::2, ::2]

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        # Pytorch's interpolate is cleaner and handles device automatically
        # but maintaining the mathematical consistency of the kernel-based upsample
        B, C, H, W = x.shape
        x_up = torch.zeros(B, C, H * 2, W * 2, device=x.device)
        x_up[:, :, ::2, ::2] = x
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        img = F.pad(img, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(img, kernel, groups=img.shape[1])

    def pyramid_decom(self, img: torch.Tensor) -> list[torch.Tensor]:
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2:] != current.shape[2:]:
                up = F.interpolate(up, size=current.shape[2:], mode="bilinear", align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def gauss_decom(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Returns the Gaussian pyramid (low-pass components) for each level."""
        current = img
        pyr = [current]
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            pyr.append(down)
            current = down
        return pyr

    def pyramid_recons(self, pyr: list[torch.Tensor]) -> torch.Tensor:
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2:] != level.shape[2:]:
                up = F.interpolate(up, size=level.shape[2:], mode="bilinear", align_corners=True)
            image = up + level
        return image


class HFBlock(nn.Module):
    """High-Frequency Block for processing pyramid residuals."""
    def __init__(self, num_residual_blocks: int, lap_layer: int = 3):
        super().__init__()
        self.lap_layer = lap_layer
        
        # Initial head for the first level
        model = [
            nn.Conv2d(10, 64, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        # In LLF-LUT, it uses ResidualBlocks here. 
        # Simplified for now assuming standard 3x3 conv blocks if ResidualBlock is complex.
        # But we'll implement a basic one.
        for _ in range(num_residual_blocks):
            model.append(SimpleResidualBlock(64))
            
        model.append(nn.Conv2d(64, 2, kernel_size=1, bias=True))
        self.main_block = nn.Sequential(*model)

        self.high_freq_blocks = nn.ModuleList()
        for _ in range(lap_layer - 1):
            self.high_freq_blocks.append(nn.Sequential(
                nn.Conv2d(9, 16, kernel_size=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 2, kernel_size=1, bias=True)
            ))

    def forward(self, concat_imgs: torch.Tensor, gauss_input: list[torch.Tensor], 
                pyr_input: list[torch.Tensor], enhanced_low: torch.Tensor) -> list[torch.Tensor]:
        
        pyr_reconstruct_list = []
        
        # First level
        fact_sigma = self.main_block(concat_imgs)
        fact, sigma = fact_sigma.chunk(2, dim=1)
        
        pyr_reconstruct_ori = remapping(gauss_input[-2], pyr_input[-2], sigma, fact, 10)
        
        pyr_reconstruct = pyr_reconstruct_ori
        up_enhanced = enhanced_low
        
        # Cache for reversed reconstruction
        pyr_cache = []

        for i in range(self.lap_layer - 1):
            up = F.interpolate(up_enhanced, size=pyr_input[-2-i].shape[2:], mode="bilinear", align_corners=True)
            up_enhanced = up + pyr_reconstruct
            
            # Prep for next scale
            up_enhanced = F.interpolate(up_enhanced, size=pyr_input[-3-i].shape[2:], mode="bilinear", align_corners=True)
            pyr_reconstruct = F.interpolate(pyr_reconstruct, size=pyr_input[-3-i].shape[2:], mode="bilinear", align_corners=True)
            
            concat_high = torch.cat([up_enhanced, pyr_input[-3-i], pyr_reconstruct], 1)
            fact_sigma = self.high_freq_blocks[i](concat_high)
            fact, sigma = fact_sigma.chunk(2, dim=1)
            
            pyr_reconstruct = remapping(gauss_input[-3-i], pyr_input[-3-i], sigma, fact, 10)
            pyr_cache.append(pyr_reconstruct)

        # Reverse list to match LLF-LUT logic
        pyr_reconstruct_list = list(reversed(pyr_cache))
        pyr_reconstruct_list.append(pyr_reconstruct_ori)
        pyr_reconstruct_list.append(enhanced_low)
        
        return pyr_reconstruct_list


class SimpleResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return x + out


class LocalRefinementHead(nn.Module):
    def __init__(self, num_residual_blocks: int = 5, num_lap: int = 3):
        super().__init__()
        self.num_lap = num_lap
        self.lap_pyramid = LapPyramidConv(num_high=num_lap)
        self.block = HFBlock(num_residual_blocks, num_lap)

    def forward(self, input_image: torch.Tensor, enhanced_low: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            input_image: Original input image
            enhanced_low: LUT-processed version of the image (or low-freq result)
        """
        # 1. Decompose input
        pyr_input = self.lap_pyramid.pyramid_decom(input_image)
        gauss_input = self.lap_pyramid.gauss_decom(input_image)
        
        # 2. Generate edge map for edge-aware refinement
        low_freq_gray = kornia.color.rgb_to_grayscale(enhanced_low)
        # Using kornia.filters.canny - result is tuple (magnitude, edges)
        edge_map = kornia.filters.canny(low_freq_gray)[1]
        
        # 3. Setup initial concat for HFBlock
        target_size = pyr_input[-2].shape[2:]
        low_freq_up = F.interpolate(enhanced_low, size=target_size, mode="bilinear", align_corners=True)
        gauss_input_up = F.interpolate(pyr_input[-1], size=target_size, mode="bilinear", align_corners=True)
        edge_map_up = F.interpolate(edge_map, size=target_size, mode="bilinear", align_corners=True)
        
        concat_imgs = torch.cat([pyr_input[-2], edge_map_up, low_freq_up, gauss_input_up], 1)
        
        # 4. Refine pyramid
        pyr_reconstruct_results = self.block(concat_imgs, gauss_input, pyr_input, enhanced_low)
        
        return pyr_reconstruct_results

    def reconstruct(self, pyr: list[torch.Tensor]) -> torch.Tensor:
        return self.lap_pyramid.pyramid_recons(pyr)
