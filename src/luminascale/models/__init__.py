"""Neural network models for LuminaScale."""

from __future__ import annotations

from .dequant_net import DequantNet, create_dequant_net

__all__ = [
    "DequantNet",
    "create_dequant_net",
]
