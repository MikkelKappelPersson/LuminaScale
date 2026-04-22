"""Dequant-Net: 6-level U-Net for structural quantization removal.

Implements the architecture from "Single-Image HDR Reconstruction by Learning
to Reverse the Camera Pipeline" (Liu et al., CVPR 2020). The network learns
to remove 8-bit quantization banding artifacts via residual learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DequantNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        num_levels: int = 6,
        leaky_relu_slope: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.leaky_relu_slope = leaky_relu_slope

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        in_ch = in_channels
        for level in range(num_levels):
            out_ch = base_channels * (2 ** level)
            block = self._make_encoder_block(in_ch, out_ch)
            self.encoder_blocks.append(block)
            in_ch = out_ch

        self.bottleneck = self._make_encoder_block(in_ch, in_ch)

        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        decoder_in_ch = in_ch
        for level in range(num_levels - 1, -1, -1):
            out_ch = base_channels * (2 ** level)
            # Fix: The skip connection from encoder level N has 'out_ch' channels
            block = self._make_decoder_block(decoder_in_ch + out_ch, out_ch)
            self.decoder_blocks.append(block)
            decoder_in_ch = out_ch

        self.final_conv = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1
        )
        self.tanh = nn.Tanh()

    def _make_encoder_block(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
        )

    def _make_decoder_block(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_img = x

        skip_connections = []
        feat = x
        for level, encoder_block in enumerate(self.encoder_blocks):
            feat = encoder_block(feat)
            skip_connections.append(feat)
            feat = self.pool(feat)

        feat = self.bottleneck(feat)

        skip_connections.reverse()
        for level, (decoder_block, skip) in enumerate(
            zip(self.decoder_blocks, skip_connections)
        ):
            feat = self.upsample(feat)
            # Crop skip connection to match upsampled feature spatial dimensions
            # This handles off-by-one errors from bilinear upsampling on non-power-of-2 sizes
            feat_h, feat_w = feat.shape[2:]
            skip = skip[:, :, :feat_h, :feat_w]
            feat = torch.cat([feat, skip], dim=1)
            feat = decoder_block(feat)

        residual = self.final_conv(feat)
        residual = self.tanh(residual)

        output = input_img + residual
        return output


def create_dequant_net(
    device: torch.device | str = "cpu",
    base_channels: int = 16,
    **kwargs: dict,
) -> DequantNet:
    model = DequantNet(base_channels=base_channels, **kwargs)
    model = model.to(device)
    return model
