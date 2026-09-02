"""Generate 1:1 crop comparisons for ablation A (a0001, native resolution).

Two regions of interest, refined vs exposure-matched LUT-only (and input):
- the butte silhouette edge (chromatic ringing / fringing)
- a flat-ish foreground region (colour noise in low-texture areas)

Usage (from LuminaScale/):
    PYTHONPATH=src pixi run python scripts/make_ablation_a_crops.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from luminascale.utils.io import read_exr
from make_ablation_a_figure import exposure_match, load_acescct, to_display

# (label, x0, y0, size) fractions of the native 3040x2014 image
CROPS = [
    ("Butte silhouette edge (ringing)", 0.22, 0.15, 0.16),
    ("Flat ground (colour noise)", 0.28, 0.70, 0.14),
]


def crop(arr: np.ndarray, x0f: float, y0f: float, sf: float) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0 = int(x0f * w), int(y0f * h)
    s = int(sf * w)
    return arr[y0 : y0 + s, x0 : x0 + s]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="outputs/inference/ablationA")
    parser.add_argument("--input-dir", default="assets/srgb_32f")
    parser.add_argument(
        "--output",
        default="../med10-journal/attachments/ablation-a-refiner-20260902-a0001-native-crops.png",
    )
    args = parser.parse_args()

    base, input_dir, out_path = Path(args.base), Path(args.input_dir), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_srgb = np.clip(read_exr(str(input_dir / "MIT-Adobe_5K_a0001-jmac_DSC1459.exr")).transpose(1, 2, 0), 0.0, 1.0)
    refined = load_acescct(base / "a0001_refined_native_out.exr")
    lutonly = load_acescct(base / "a0001_lutonly_native_out.exr")
    lut_matched, _ = exposure_match(lutonly, refined)

    panels = [
        ("Input", input_srgb),
        ("Full model (refined)", to_display(refined)),
        ("LUT-only, exposure-matched", to_display(lut_matched)),
    ]

    dpi = 100
    crop_px = int(CROPS[0][3] * 3040) * 2  # crop size after 2x upsample
    fig_w = crop_px * len(panels) / dpi + 2.5
    fig_h = crop_px * len(CROPS) / dpi + 1.5
    fig, axes = plt.subplots(len(CROPS), len(panels), figsize=(fig_w, fig_h), dpi=dpi)
    for row, (label, x0f, y0f, sf) in enumerate(CROPS):
        for col, (plabel, img) in enumerate(panels):
            ax = axes[row][col]
            c = crop(img, x0f, y0f, sf)
            # nearest-neighbour upsample 2x so 1:1 pixels stay visible at full size
            c = np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)
            ax.imshow(c, interpolation="nearest")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(plabel, fontsize=22, pad=10)
        axes[row][0].text(
            -0.02, 0.5, label, transform=axes[row][0].transAxes, rotation=90,
            ha="right", va="center", fontsize=20,
        )
    fig.suptitle(
        "Ablation A - 1:1 crops (2x nearest), refined vs exposure-matched LUT-only, seed 9",
        fontsize=30, y=0.985,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.01, wspace=0.02, hspace=0.06)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved crop comparison: {out_path}")


if __name__ == "__main__":
    main()
