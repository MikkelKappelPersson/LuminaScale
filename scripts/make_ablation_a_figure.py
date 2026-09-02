"""Generate the unified ablation-A figure at native input resolution.

Single scene (MIT-Adobe 5K a0001, 3040x2014), four panels:
input (32-bit sRGB) | full model (refined) | LUT-only (--no-refiner) |
LUT-only exposure-matched.

Exposure matching: per-channel gain computed in linear space (OCIO:
ACEScct -> ACES2065-1) so the low-frequency mean of the matched LUT-only
image equals the refined image's; high frequencies are untouched. The
matched image is converted back to ACEScct and rendered through the same
OCIO display path as the other panels.

Also prints per-panel high-frequency noise stats (std of image minus
gaussian blur, in linear space) for the journal prose.

Usage (from LuminaScale/):
    PYTHONPATH=src pixi run python scripts/make_ablation_a_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import OpenImageIO as oiio
import torch
from scipy import ndimage

from luminascale.utils.io import aces_to_display_gpu, read_exr

INPUT_NAME = "MIT-Adobe_5K_a0001-jmac_DSC1459.exr"
STEM = "a0001"


def load_acescct(path: Path) -> np.ndarray:
    """Read a model-output EXR as HxWx3 ACEScct float32."""
    return read_exr(str(path)).transpose(1, 2, 0)


def oiio_convert(array_hwc: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
    """OCIO color conversion on an in-memory HxWx3 float32 array."""
    buf = oiio.ImageBuf(array_hwc.astype(np.float32))
    out = oiio.ImageBufAlgo.colorconvert(buf, from_space, to_space, False)
    assert out.initialized, f"OCIO conversion failed: {from_space} -> {to_space}"
    return np.asarray(out.get_pixels(), dtype=np.float32)[:, :, :3]


def to_display(array_acescct: np.ndarray) -> np.ndarray:
    """ACEScct HxWx3 -> display sRGB via the same path as the dashboards."""
    t = torch.from_numpy(array_acescct).to("cuda", dtype=torch.float32)
    srgb, _ = aces_to_display_gpu(t, input_cs="ACEScct")
    return srgb.clamp(0.0, 1.0).cpu().numpy()


def exposure_match(lut_acescct: np.ndarray, ref_acescct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gain-match lut to ref in linear space; returns (matched ACEScct, per-channel gains)."""
    ref_lin = oiio_convert(ref_acescct, "ACEScct", "ACES2065-1")
    lut_lin = oiio_convert(lut_acescct, "ACEScct", "ACES2065-1")
    gains = []
    matched = np.empty_like(lut_lin)
    for c in range(3):
        ref_low = ndimage.gaussian_filter(ref_lin[:, :, c], sigma=32)
        lut_low = ndimage.gaussian_filter(lut_lin[:, :, c], sigma=32)
        gain = float(ref_low.mean() / lut_low.mean())
        gains.append(gain)
        matched[:, :, c] = lut_lin[:, :, c] * gain
    return oiio_convert(matched, "ACES2065-1", "ACEScct"), np.array(gains)


def hf_noise_linear(array_acescct: np.ndarray, sigma: int = 4) -> float:
    """High-frequency noise proxy: std of (image - gaussian blur) in linear space."""
    lin = oiio_convert(array_acescct, "ACEScct", "ACES2065-1")
    residual = lin - ndimage.gaussian_filter(lin, sigma=(0, sigma, sigma))
    return float(residual.std())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="outputs/inference/ablationA")
    parser.add_argument("--input-dir", default="assets/srgb_32f")
    parser.add_argument(
        "--output",
        default="../med10-journal/attachments/ablation-a-refiner-20260902-a0001-native-exposure-matched.png",
    )
    args = parser.parse_args()

    base, input_dir, out_path = Path(args.base), Path(args.input_dir), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_srgb = np.clip(read_exr(str(input_dir / INPUT_NAME)).transpose(1, 2, 0), 0.0, 1.0)
    refined = load_acescct(base / f"{STEM}_refined_native_out.exr")
    lutonly = load_acescct(base / f"{STEM}_lutonly_native_out.exr")
    lut_matched, gains = exposure_match(lutonly, refined)
    print(f"Exposure-match per-channel gains (R, G, B): {gains.round(4)}")

    panels = [
        ("Input (32-bit sRGB)", input_srgb),
        ("Full model (refined)", to_display(refined)),
        ("LUT-only (--no-refiner)", to_display(lutonly)),
        (f"LUT-only, exposure-matched (x{gains.round(2)})", to_display(lut_matched)),
    ]
    for name, arr in [("refined", refined), ("lut-only", lutonly), ("lut-only matched", lut_matched)]:
        print(f"HF noise (linear, sigma=4 residual std) {name}: {hf_noise_linear(arr):.5f}")

    native_h, native_w = panels[0][1].shape[:2]
    dpi = 100
    panel_w_in = native_w / dpi
    fig, axes = plt.subplots(
        1, len(panels), figsize=(panel_w_in * len(panels), native_h / dpi + 2.0), dpi=dpi
    )
    fig.subplots_adjust(left=0.002, right=0.998, top=0.82, bottom=0.005, wspace=0.01)
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(label, fontsize=48, pad=16)
    fig.suptitle(
        "Ablation A - refinement head on/off, same input, same seed 9 (native resolution)",
        fontsize=56,
        y=0.97,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved unified native-resolution figure: {out_path}")


if __name__ == "__main__":
    main()
