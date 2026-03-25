"""Neural network models for LuminaScale."""

from __future__ import annotations

from .dequantization_net import DequantizationNet, create_dequantization_net

__all__ = [
    "DequantizationNet",
    "create_dequantization_net",
]
