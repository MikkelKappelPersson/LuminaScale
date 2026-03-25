"""Generate custom OCIO looks dynamically for training data augmentation.

OCIO looks are named transforms applied in a "process_space". This module
generates looks with randomized color grading parameters.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import PyOpenColorIO as OCIO
except ImportError:
    OCIO = None


@dataclass
class CDLParameters:
    """Color Decision List (CDL) grade parameters.

    CDL is the standard format for looks in OCIO. Parameters:
    - slope: Multiply (lift shadows in linear domain), default [1, 1, 1]
    - offset: Add (brighten/darken), default [0, 0, 0]
    - power: Gamma (exponent), default [1, 1, 1]
    - saturation: Overall saturation, default 1.0
    """

    slope: tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    power: tuple[float, float, float] = (1.0, 1.0, 1.0)
    saturation: float = 1.0

    def to_cdl_xml(self) -> str:
        """Generate CDL XML representation."""
        slope_str = " ".join(f"{v:.6f}" for v in self.slope)
        offset_str = " ".join(f"{v:.6f}" for v in self.offset)
        power_str = " ".join(f"{v:.6f}" for v in self.power)

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<ColorCorrection id="aces_look">
  <SOPNode>
    <Slope>{slope_str}</Slope>
    <Offset>{offset_str}</Offset>
    <Power>{power_str}</Power>
  </SOPNode>
  <SatNode>
    <Saturation>{self.saturation:.6f}</Saturation>
  </SatNode>
</ColorCorrection>
"""

# In look_generator.py


def get_single_random_look() -> CDLParameters:
    return random_cdl()


def random_cdl(variance=0.2, sat_range=(0.4, 1.6)) -> CDLParameters:
    """
    Generates extreme looks using a Master + Variance approach for
    Slope, Offset, and Power.
    """
    # 1. SLOPE (Highlights/Gain)
    master_slope = random.uniform(0.6, 1.3)
    slope = tuple(master_slope + random.uniform(-variance, variance) for _ in range(3))

    # 2. OFFSET (Shadows/Lift)
    # Master offset determines if shadows are 'crushed' or 'faded'
    master_offset = random.uniform(-0.1, 0.1)
    # Variance adds a color tint specifically to the shadows
    offset = tuple(
        master_offset + random.uniform(-variance * 0.3, variance * 0.3)
        for _ in range(3)
    )

    # 3. POWER (Gamma/Contrast)
    # Master power determines the overall punchiness
    master_power = random.uniform(0.8, 1.3)
    # Variance creates 'cross-processed' color curves
    power = tuple(
        master_power + random.uniform(-variance * 0.3, variance * 0.3) for _ in range(3)
    )

    return CDLParameters(
        slope=(float(slope[0]), float(slope[1]), float(slope[2])),
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
        power=(float(power[0]), float(power[1]), float(power[2])),
        saturation=random.uniform(*sat_range),
    )
    return CDLParameters(
        slope=(slope[0], slope[1], slope[2]),
        offset=(offset[0], offset[1], offset[2]),
        power=(power[0], power[1], power[2]),
        saturation=random.uniform(*sat_range),
    )


class LUTManager:
    """Handles discovery and selection of LUT files with a defined input space."""

    def __init__(self, lut_dir: Path | str, input_space: str = "ACEScct"):
        self.lut_dir = Path(lut_dir)
        self.input_space = input_space  # Default to Log/ACEScct
        self.extensions = {".cube", ".3dl", ".look", ".clf"}
        self.luts = self._scan_luts()

    def _scan_luts(self) -> list[Path]:
        if not self.lut_dir.exists():
            return []
        return [
            f for f in self.lut_dir.iterdir() if f.suffix.lower() in self.extensions
        ]

    def get_random_lut(self) -> Path | None:
        return random.choice(self.luts) if self.luts else None
