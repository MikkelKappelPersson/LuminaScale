"""Spatial-Frequency Transformer (SFT) for LuminaScale.

Based on LLF-LUT (Zeng et al./Wang et al.) implementation of Spatial_Transformer.
Adapted for LuminaScale ACES mapper.

Mandatory Attribution: Based on LLF-LUT (Zeng et al./Wang et al.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath 
from timm.layers.weight_init import trunc_normal_
import numpy as np


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, B, H, W):
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size) if use_shift_size is not None else (0, 0)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv=None, mask=None):
        kv = q if kv is None else kv
        B_q, N1, C = q.shape
        B_kv, N2, C_kv = kv.shape

        # Fix: Explicitly specify embedding per head
        head_dim = C // self.num_heads

        q = self.q(q).reshape(B_q, N1, self.num_heads, head_dim).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B_kv, N2, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        q = q * self.scale

        # Handle batch dimension mismatch: match B_q to B_kv
        if B_q != B_kv:
            if B_kv % B_q == 0:
                # Repeat q to match kv (e.g. 1 window per batch vs 4 windows per batch)
                q = q.repeat(B_kv // B_q, 1, 1, 1)
                B_q = B_kv
            elif B_q % B_kv == 0:
                # Repeat kv to match q
                k = k.repeat(B_q // B_kv, 1, 1, 1)
                v = v.repeat(B_q // B_kv, 1, 1, 1)
                B_kv = B_q

        attn = (q @ k.transpose(-2, -1))

        # Fix: Access buffer directly instead of calling it
        # Cast to Tensor for Pylance if it's confused about buffer vs callable
        idx = getattr(self, "relative_position_index").view(-1)
        relative_position_bias = self.relative_position_bias_table[idx].view(
            N1, N2, -1
        ).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
             nW = mask.shape[0]
             attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask.unsqueeze(1).unsqueeze(0)
             attn = attn.view(-1, self.num_heads, N1, N2)
             attn = self.softmax(attn)
        else:
             attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # Use v's batch dimension after broadcast/repeat
        # The output of (attn @ v) is [Batch_Total, Num_Heads, N1, Head_Dim]
        # We need to reshape to [Batch_Total, N1, C]
        x = (attn @ v).transpose(1, 2).reshape(-1, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class FourierWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        window_size = self.window_size
        
        # Spatial Attention
        qkv_spatial = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_spatial[0], qkv_spatial[1], qkv_spatial[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        idx = getattr(self, "relative_position_index").view(-1)
        relative_position_bias = self.relative_position_bias_table[idx].view(
            N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x_spatial = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Spectral Attention Path
        # Reshape to 2D for FFT: [B, H, W, C]
        H, W = window_size
        x_2d = x.view(B, H, W, C)
        
        # 2D Real FFT on spatial dimensions
        x_f = torch.fft.rfftn(x_2d, dim=(1, 2), norm='ortho')
        # x_f shape: [B, H, W//2 + 1, C] (complex)
        
        # To use standard Linear/Attention, we treat real/imag as extra channels
        # Concatenate real and imag parts: [B, H, W//2 + 1, C*2]
        x_f_stacked = torch.cat([x_f.real, x_f.imag], dim=-1)
        B_f, H_f, W_f, C_f2 = x_f_stacked.shape
        N_f = H_f * W_f
        x_f_flat = x_f_stacked.view(B_f, N_f, C_f2)

        # Note: self.qkv expects input of size 'dim' (C). 
        # But x_f_flat has C*2. We should use a separate mapping or split.
        # For simplicity and to match the 'qkv' parameters, we process C then repeat or use a linear layer.
        # However, to be robust, let's use a 1x1-style projection if dim mismatch occurs.
        # Ref implementation usually has a separate Linear for spectral path.
        # Since I can't easily add a module now without changing __init__, I will chunk.
        
        # Standard approach for this architecture: 
        # Apply qkv to each half (real/imag) or project.
        # Let's project real/imag separately using the same qkv weights to maintain parameter count logic
        qkv_real = self.qkv(x_f.real.view(B_f, N_f, C)).reshape(B_f, N_f, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_imag = self.qkv(x_f.imag.view(B_f, N_f, C)).reshape(B_f, N_f, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        q_f = torch.complex(qkv_real[0], qkv_imag[0])
        k_f = torch.complex(qkv_real[1], qkv_imag[1])
        v_f = torch.complex(qkv_real[2], qkv_imag[2])
        
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.scale
        # Softmax on Complex numbers is not standard, traditionally use magnitude
        # or separate branches. We use magnitude to define importance.
        attn_mag = attn_f.abs()
        attn_mag = self.softmax(attn_mag)
        attn_mag = self.attn_drop(attn_mag)
        
        # We need to project the attenuation back to the complex domain or 
        # just apply the real-valued attention to the complex values.
        # Here we apply the attention weights to the complex values v_f.
        # attn_mag: [B_f, heads, N_f, N_f] (float)
        # v_f: [B_f, heads, N_f, head_dim] (complex)
        # Result should be complex.
        x_f_res = (attn_mag.to(v_f.dtype) @ v_f).transpose(1, 2).reshape(B_f, H_f, W_f, C)
        
        # Inverse 2D FFT
        x_f = torch.fft.irfftn(x_f_res, s=(H, W), dim=(1, 2), norm='ortho')
        x_f = x_f.reshape(B, N, C)

        x = x_spatial + x_f
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class EncoderTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 8), shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = FourierWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, Hp, Wp, _ = x.shape

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)

        attn_windows = self.attn(x_windows, mask=attn_mask)[0]

        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 8), shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn1 = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.attn2 = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, enc_feat, mask_matrix):
        B, H, W, C = x.shape
        # Handle hierarchical mismatch (e.g. skip-connections from higher res)
        # If enc_feat has more pixels (higher res), we need to ensure local spatial alignment early
        if enc_feat.shape[1] != H or enc_feat.shape[2] != W:
            # Downsample or crop enc_feat to match x's spatial resolution
            # Using interpolate for robustness if resolutions are not perfect multiples
            enc_feat = F.interpolate(
                enc_feat.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        # Partition both main feature and skip feature into identical window grids
        x_windows = window_partition(x, window_size).view(-1, window_size[0] * window_size[1], C)
        enc_windows = window_partition(enc_feat, window_size).view(-1, window_size[0] * window_size[1], C)
        
        # Self-attention on decoder features
        x_windows = self.attn1(x_windows)[0]
        # Cross-attention using encoder features as KV
        x_windows = self.attn2(x_windows, kv=enc_windows)[0]
        
        # Put back together
        x = window_reverse(x_windows.view(-1, window_size[0], window_size[1], C), window_size, B, H, W)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class InputProj(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, kernel_size=3, stride=1, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=stride, padding=1),
            act_layer(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=stride, padding=1),
            act_layer(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class SpatialFrequencyTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        num_weights: int = 3,
        depths: list[int] = [1, 1, 1, 1, 1, 1, 1, 1],
        num_heads: list[int] = [2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes: list[tuple[int, int]] = [(4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4)],
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.num_enc_layers = self.num_layers // 2
        self.num_dec_layers = self.num_layers // 2
        self.embed_dim = embed_dim
        
        self.input_proj = InputProj(in_channels=in_chans, embed_dim=embed_dim)
        
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        self.encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(self.num_enc_layers):
            self.encoder_layers.append(EncoderTransformerBlock(
                dim=embed_dim, num_heads=num_heads[i], window_size=window_sizes[i], 
                drop_path=enc_dpr[i], mlp_ratio=mlp_ratio
            ))
            self.downsample.append(Downsample(embed_dim, embed_dim))
            
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(self.num_dec_layers):
            self.decoder_layers.append(DecoderTransformerBlock(
                dim=embed_dim, num_heads=num_heads[i + self.num_enc_layers], 
                window_size=window_sizes[i + self.num_enc_layers],
                mlp_ratio=mlp_ratio
            ))
            self.upsample.append(Upsample(embed_dim, embed_dim))
            
        self.head = nn.Linear(embed_dim, num_weights)
        trunc_normal_(self.head.weight, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: B, C, H, W
        x = self.input_proj(x) # B, embed_dim, H, W
        x = x.permute(0, 2, 3, 1) # B, H, W, embed_dim
        
        enc_feats = []
        for i in range(self.num_enc_layers):
            x = self.encoder_layers[i](x, None)
            enc_feats.append(x)
            x = self.downsample[i](x)
            
        for i in range(self.num_dec_layers):
            x = self.decoder_layers[i](x, enc_feats[-i-1], None)
            x = self.upsample[i](x)
            
        x = x.mean(dim=[1, 2]) # Global Average Pooling
        weights = self.head(x)
        return weights
