#!/usr/bin/env python3
"""LuminaScale full inference pipeline: Dequantization -> ACES mapper."""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from luminascale.models import create_dequant_net
from luminascale.utils.aces_mapper_inference import (
    align_to_multiple_hwc,
    build_look,
    center_crop_hwc,
    load_model_from_checkpoint,
    resize_to_max_side_hwc,
)
from luminascale.utils.gpu_cdl_processor import GPUCDLProcessor
from luminascale.utils.io import image_to_tensor, write_exr
from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer


def align_to_multiple_bchw(image: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Pad image [B, C, H, W] to nearest multiple using edge-value replication."""
    if multiple <= 1:
        return image, image.shape[2], image.shape[3]

    height, width = image.shape[2], image.shape[3]
    new_height = ((height + multiple - 1) // multiple) * multiple
    new_width = ((width + multiple - 1) // multiple) * multiple

    if new_height == height and new_width == width:
        return image, height, width

    pad_height = new_height - height
    pad_width = new_width - width
    padded = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
    return padded, height, width


def load_dequant_model_from_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
    channels: int,
) -> torch.nn.Module:
    """Load DequantNet checkpoint into inference model."""
    model = create_dequant_net(device=device, base_channels=channels)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    assert isinstance(state_dict, dict), "Checkpoint state_dict is not a dictionary"

    if all(str(key).startswith("model.") for key in state_dict.keys()):
        state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}

    if all(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    load_result = model.load_state_dict(state_dict, strict=True)
    assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    model.eval()
    return model


def save_tensor_image(output_chw: torch.Tensor, output_path: Path) -> None:
    """Save [C, H, W] tensor to EXR or 8-bit image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_chw = torch.clamp(output_chw, 0.0, 1.0).detach().cpu()

    if output_path.suffix.lower() == ".exr":
        write_exr(output_path, output_chw)
        return

    output_hwc = (output_chw.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(output_hwc).save(output_path)


def run_dequant_inference(
    model: torch.nn.Module,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    align_multiple: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    quantize_input: bool = False,
    input_chw_override: torch.Tensor | None = None,
) -> tuple[Path, torch.Tensor, torch.Tensor]:
    """Run dequant model and save intermediate output image."""
    if input_chw_override is not None:
        input_chw = input_chw_override.to(device=device, dtype=torch.float32)
    else:
        input_chw = image_to_tensor(input_path).to(device=device, dtype=torch.float32)

    if quantize_input:
        input_chw = torch.clamp(input_chw, 0.0, 1.0)
        input_chw = torch.round(input_chw * 255.0) / 255.0

    input_bchw = input_chw.unsqueeze(0)
    aligned_bchw, original_height, original_width = align_to_multiple_bchw(input_bchw, align_multiple)

    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
    with torch.inference_mode(), amp_ctx:
        output_bchw = model(aligned_bchw)

    output_chw = output_bchw.squeeze(0)[:, :original_height, :original_width]
    save_tensor_image(output_chw, output_path)
    return output_path, input_chw.detach().cpu(), output_chw.detach().cpu()


def resize_chw_to(chw: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize [C, H, W] tensor to target size using bilinear interpolation."""
    if chw.shape[1] == height and chw.shape[2] == width:
        return chw
    bchw = chw.unsqueeze(0)
    resized = F.interpolate(bchw, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def run_mapper_inference_on_srgb(
    model: torch.nn.Module,
    input_srgb_chw: torch.Tensor,
    *,
    crop_size: int,
    align_multiple: int,
    max_side: int,
    keep_aligned_output: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run mapper model on sRGB input and return (pred_aces_chw, pred_srgb_chw)."""
    input_srgb_hwc = input_srgb_chw.permute(1, 2, 0).to(device=device, dtype=torch.float32)
    input_srgb_hwc = center_crop_hwc(input_srgb_hwc, crop_size)
    input_srgb_hwc = resize_to_max_side_hwc(input_srgb_hwc, max_side)
    out_h, out_w, _ = input_srgb_hwc.shape
    input_srgb_hwc = align_to_multiple_hwc(input_srgb_hwc, align_multiple)
    input_srgb_hwc = torch.clamp(input_srgb_hwc, 0.0, 1.0)

    model_input = input_srgb_hwc.permute(2, 0, 1).unsqueeze(0)
    forward_model = model.model if hasattr(model, "model") and isinstance(model.model, torch.nn.Module) else model
    model_input = model_input.to(dtype=torch.float32)
    amp_ctx = nullcontext()

    with torch.inference_mode(), amp_ctx:
        pred_aces_chw = forward_model(model_input).squeeze(0)

    if not keep_aligned_output and align_multiple > 1:
        pred_aces_chw = pred_aces_chw[:, :out_h, :out_w]

    transformer = ACESColorTransformer(device=device, use_lut=True)
    pred_srgb_hwc = transformer.aces_to_srgb_32f(pred_aces_chw.permute(1, 2, 0).unsqueeze(0)).squeeze(0)
    pred_srgb_chw = pred_srgb_hwc.permute(2, 0, 1).detach().cpu()
    return pred_aces_chw.detach().cpu(), pred_srgb_chw


def save_full_inference_dashboard(
    input_srgb_chw: torch.Tensor,
    dequant_srgb_chw: torch.Tensor,
    mapper_srgb_chw: torch.Tensor,
    reference_srgb_chw: torch.Tensor | None,
    dequant_reference_srgb_chw: torch.Tensor,
    save_path: Path,
) -> None:
    """Save custom full-inference dashboard (2x4)."""
    target_h, target_w = mapper_srgb_chw.shape[1], mapper_srgb_chw.shape[2]
    panel_aspect = target_h / max(1, target_w)

    input_match = resize_chw_to(input_srgb_chw, target_h, target_w)
    dequant_match = resize_chw_to(dequant_srgb_chw, target_h, target_w)
    mapper_match = mapper_srgb_chw
    ref_match = resize_chw_to(reference_srgb_chw, target_h, target_w) if reference_srgb_chw is not None else None
    dequant_ref_match = resize_chw_to(dequant_reference_srgb_chw, target_h, target_w)

    input_np = torch.clamp(input_match, 0.0, 1.0).permute(1, 2, 0).numpy()
    dequant_np = torch.clamp(dequant_match, 0.0, 1.0).permute(1, 2, 0).numpy()
    mapper_np = torch.clamp(mapper_match, 0.0, 1.0).permute(1, 2, 0).numpy()
    ref_np = (
        torch.clamp(ref_match, 0.0, 1.0).permute(1, 2, 0).numpy() if ref_match is not None else np.zeros_like(mapper_np)
    )

    dequant_ref_np = torch.clamp(dequant_ref_match, 0.0, 1.0).permute(1, 2, 0).numpy()
    dequant_diff = np.mean(np.abs(dequant_np - dequant_ref_np), axis=2)
    dequant_input_diff = np.mean(np.abs(dequant_np - input_np), axis=2)
    mapper_ref = ref_np if ref_match is not None else dequant_np
    mapper_diff = np.mean(np.abs(mapper_np - mapper_ref), axis=2)

    luma_input = np.mean(input_np, axis=2)
    luma_dequant_ref = np.mean(dequant_ref_np, axis=2)
    luma_dequant = np.mean(dequant_np, axis=2)
    luma_mapper = np.mean(mapper_np, axis=2)
    luma_ref = np.mean(mapper_ref, axis=2)

    def gradient_magnitude(image_luma: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(image_luma.astype(np.float32))
        return np.sqrt((grad_x * grad_x) + (grad_y * grad_y))

    grad_input = gradient_magnitude(luma_input)
    grad_dequant_ref = gradient_magnitude(luma_dequant_ref)
    grad_dequant = gradient_magnitude(luma_dequant)
    grad_ref = gradient_magnitude(luma_ref)

    grad_dequant_err = np.abs(grad_dequant - grad_dequant_ref)

    # Explicitly size the figure so each mosaic cell gets the requested dimensions,
    # while accounting for constrained-layout gaps and outer padding.
    cell_width = input_np.shape[1] / 100.0
    cell_height = cell_width * panel_aspect

    ncols = 4
    nrows = 2
    gap_w = 0.08
    gap_h = 0.12
    pad_w = 0.08
    pad_h = 0.08

    fig_width = (ncols * cell_width) + ((ncols - 1) * gap_w * cell_width) + (2.0 * pad_w)
    fig_height = (nrows * cell_height) + ((nrows - 1) * gap_h * cell_height) + (2.0 * pad_h)

    # Keep exported width bounded to avoid generating extremely large PNG files.
    figure_dpi = float(plt.rcParams.get("figure.dpi", 100.0))
    max_output_width_px = 8192.0
    max_fig_width = max_output_width_px / figure_dpi
    if fig_width > max_fig_width:
        downscale = max_fig_width / fig_width
        fig_width = max_fig_width
        fig_height *= downscale
        pad_w *= downscale
        pad_h *= downscale

    # Scale text with panel size so labels remain readable on larger dashboards.
    effective_cell_width = max(1e-6, (fig_width - 2.0 * pad_w) / ncols)
    text_scale = float(np.clip(effective_cell_width / 3.5, 1.0, 3.0)) * 0.8
    title_fs = 11.0 * text_scale
    axis_label_fs = 9.0 * text_scale
    tick_fs = 8.0 * text_scale
    legend_fs = 8.0 * text_scale

    fig, axd = plt.subplot_mosaic(
        [
            ["input", "dequant", "mapper", "reference"],
            ["gradient_domain", "dequant_diff", "mapper_diff", "luma_parity"],
        ],
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": gap_w, "hspace": gap_h},
    )
    fig.subplots_adjust(
        left=pad_w / fig_width,
        right=1.0 - (pad_w / fig_width),
        bottom=pad_h / fig_height,
        top=1.0 - (pad_h / fig_height),
    )

    axd["input"].imshow(input_np)
    axd["input"].set_title("Input", fontsize=title_fs)
    axd["input"].axis("off")
    axd["input"].set_box_aspect(panel_aspect)

    axd["dequant"].imshow(dequant_np)
    axd["dequant"].set_title("Output Dequant", fontsize=title_fs)
    axd["dequant"].axis("off")
    axd["dequant"].set_box_aspect(panel_aspect)

    axd["mapper"].imshow(mapper_np)
    axd["mapper"].set_title("Output ACES Mapper", fontsize=title_fs)
    axd["mapper"].axis("off")
    axd["mapper"].set_box_aspect(panel_aspect)

    axd["reference"].imshow(ref_np)
    axd["reference"].set_title("Reference (sRGB)", fontsize=title_fs)
    axd["reference"].axis("off")
    axd["reference"].set_box_aspect(panel_aspect)

    dequant_im = axd["dequant_diff"].imshow(dequant_diff, cmap="magma")
    axd["dequant_diff"].set_title(
        f"|Dequant - LookRef | Mean - {float(dequant_diff.mean()):.5f} (max {float(dequant_diff.max()):.3f})",
        fontsize=title_fs,
    )
    axd["dequant_diff"].axis("off")
    dequant_cax = axd["dequant_diff"].inset_axes([0.00, 0.00, 1, 0.05])
    dequant_cbar = fig.colorbar(dequant_im, cax=dequant_cax, orientation="horizontal")
    dequant_cbar.set_label("Abs Error", fontsize=axis_label_fs, labelpad=2)
    dequant_cbar.ax.spines[:].set_visible(False)
    dequant_cbar.ax.xaxis.set_ticks_position("bottom")
    dequant_cbar.ax.xaxis.set_label_position("bottom")
    dequant_cbar.ax.tick_params(labelsize=tick_fs)

    mapper_im = axd["mapper_diff"].imshow(mapper_diff, cmap="magma")
    ref_label = "Reference" if ref_match is not None else "Dequant"
    axd["mapper_diff"].set_title(
        f"|Mapper - {ref_label}| Mean - {float(mapper_diff.mean()):.5f} (max {float(mapper_diff.max()):.3f})",
        fontsize=title_fs,
    )
    axd["mapper_diff"].axis("off")
    mapper_cax = axd["mapper_diff"].inset_axes([0.00, 0.00, 1, 0.05])
    mapper_cbar = fig.colorbar(mapper_im, cax=mapper_cax, orientation="horizontal")
    mapper_cbar.set_label("Abs Error", fontsize=axis_label_fs, labelpad=2)
    mapper_cbar.ax.spines[:].set_visible(False)
    mapper_cbar.ax.xaxis.set_ticks_position("bottom")
    mapper_cbar.ax.xaxis.set_label_position("bottom")
    mapper_cbar.ax.tick_params(labelsize=tick_fs)

    # Focus on median luma +/- 0.01 range to detect quantization steps
    median_luma = np.median(luma_dequant_ref)
    luma_range = 0.0001
    luma_min = max(0.0, median_luma - luma_range)
    luma_max = min(1.0, median_luma + luma_range)
    
    # Create bins in the focused range
    num_bins = 1000
    luma_bins = np.linspace(luma_min, luma_max, num_bins)
    bin_centers = (luma_bins[:-1] + luma_bins[1:]) / 2
    bin_centers = (luma_bins[:-1] + luma_bins[1:]) / 2
    
    # Get RGB channels
    dequant_r = dequant_np[:, :, 0].flatten()
    dequant_g = dequant_np[:, :, 1].flatten()
    dequant_b = dequant_np[:, :, 2].flatten()
    
    ref_r = dequant_ref_np[:, :, 0].flatten()
    ref_g = dequant_ref_np[:, :, 1].flatten()
    ref_b = dequant_ref_np[:, :, 2].flatten()
    
    channels = [
        (ref_r, 'Reference R', 'red'),
        (ref_g, 'Reference G', 'green'),
        (ref_b, 'Reference B', 'blue'),
        (dequant_r, 'Dequant R', 'red', '--'),
        (dequant_g, 'Dequant G', 'green', '--'),
        (dequant_b, 'Dequant B', 'blue', '--'),
    ]
    
    for channel_data, label, color, *style in channels:
        hist, _ = np.histogram(channel_data, bins=luma_bins)
        cumulative = np.cumsum(hist)
        linestyle = style[0] if style else '-'
        axd["gradient_domain"].plot(
            bin_centers,
            cumulative,
            label=label,
            linewidth=1.8,
            color=color,
            linestyle=linestyle,
        )
    
    axd["gradient_domain"].set_title(f"Channel Distributions (Luma {median_luma:.3f} ±{luma_range})", fontsize=title_fs)
    axd["gradient_domain"].set_xlabel("Channel Value", fontsize=axis_label_fs)
    axd["gradient_domain"].set_ylabel("Accumulated Pixel Count", fontsize=axis_label_fs)
    axd["gradient_domain"].set_xlim(luma_min, luma_max)
    axd["gradient_domain"].tick_params(axis="both", labelsize=tick_fs)
    axd["gradient_domain"].grid(True, alpha=0.25)
    axd["gradient_domain"].legend(fontsize=legend_fs)

    flat_ref = luma_ref.reshape(-1)
    flat_mapper = luma_mapper.reshape(-1)
    sample_count = min(10000, flat_ref.size)
    sample_idx = np.linspace(0, flat_ref.size - 1, sample_count, dtype=np.int64)
    s_ref = flat_ref[sample_idx]
    s_mapper = flat_mapper[sample_idx]
    axd["luma_parity"].scatter(s_ref, s_mapper, s=10, alpha=0.4, edgecolors="none")
    min_v = float(min(s_ref.min(), s_mapper.min()))
    max_v = float(max(s_ref.max(), s_mapper.max()))
    axd["luma_parity"].plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.0)
    axd["luma_parity"].set_title(f"Luma Parity (Mapper vs {ref_label})", fontsize=title_fs)
    axd["luma_parity"].set_xlabel(ref_label, fontsize=axis_label_fs)
    axd["luma_parity"].set_ylabel("Mapper", fontsize=axis_label_fs)
    axd["luma_parity"].tick_params(axis="both", labelsize=tick_fs)
    axd["luma_parity"].grid(True, alpha=0.50)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi="figure", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full inference (dequant -> ACES mapper) and save comparison dashboard.",
    )
    parser.add_argument(
        "--mapper-checkpoint",
        dest="mapper_checkpoint",
        type=str,
        default="outputs/training/mapper/20260425_231537/checkpoints/aces-mapper-20260425_231537-epoch=09.ckpt",
        help="Path to model checkpoint (.ckpt/.pt)",
    )
    parser.add_argument(
        "--dequant-checkpoint",
        type=str,
        default="outputs/training/dequant/20260422_120606_L1=1.0_L2=0.0_CB=1.0_EA=0.0_TV-huber=0.0/checkpoints/last.ckpt",
        help="Path to dequant model checkpoint (.ckpt/.pt)",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default="dataset/full/aces/MIT-Adobe_5K_a0001-jmac_DSC1459.exr",
        help="Path to regular (non-ACES) input image by default",
    )
    parser.add_argument(
        "--input-is-aces",
        action="store_true",
        help="Flag: run ACES reference pipeline (expects ACES2065-1 EXR input)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save predicted ACES EXR (default: outputs/inference/<input_stem>_out.exr)",
    )
    parser.add_argument(
        "--no-save-output",
        action="store_true",
        help="Disable saving predicted ACES EXR output",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save comparison dashboard plot",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="Optional center crop size before inference (0 disables crop)",
    )
    parser.add_argument(
        "--align-multiple",
        type=int,
        default=64,
        help="Pad H/W to nearest higher multiple using edge replication (<=1 disables)",
    )
    parser.add_argument(
        "--keep-padding",
        action="store_true",
        help="Keep padded aligned dimensions in generated outputs instead of removing alignment padding",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Cap longest image side before inference to reduce VRAM use (<=0 disables)",
    )
    parser.add_argument(
        "--look-mode",
        type=str,
        default="random",
        choices=["random", "manual"],
        help="Look mode: random CDL or manual CDL params",
    )
    parser.add_argument("--slope", type=str, default="1.0,1.0,1.0", help="Manual CDL slope triplet")
    parser.add_argument("--offset", type=str, default="0.0,0.0,0.0", help="Manual CDL offset triplet")
    parser.add_argument("--power", type=str, default="1.0,1.0,1.0", help="Manual CDL power triplet")
    parser.add_argument("--saturation", type=float, default=1.0, help="Manual CDL saturation")
    parser.add_argument(
        "--seed",
        type=int,
        default=9,
        help="Seed for reproducible random look generation",
    )
    parser.add_argument("--num-luts", type=int, default=3, help="Model num_luts")
    parser.add_argument("--lut-dim", type=int, default=33, help="Model LUT dimension")
    parser.add_argument("--num-lap", type=int, default=3, help="Model Laplacian levels")
    parser.add_argument(
        "--num-residual-blocks",
        type=int,
        default=5,
        help="Model refiner residual blocks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (cuda/cpu)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision during model inference on CUDA",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="AMP dtype when --amp is enabled",
    )
    parser.add_argument(
        "--dequant-output",
        type=str,
        default="",
        help="Intermediate dequantized output filename in output folder (default: <input_stem>_dequant.exr)",
    )
    parser.add_argument(
        "--save-dequant",
        action="store_true",
        help="Save dequantized intermediate output (otherwise kept temporary for mapper stage only)",
    )
    parser.add_argument("--channels", "--dequant-channels", dest="dequant_channels", type=int, default=32, help="Dequant model base channels")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    amp_enabled = bool(args.amp and device.type == "cuda")

    input_path = Path(args.input_path)
    assert input_path.exists(), f"Input does not exist: {input_path}"
    is_exr_reference = bool(args.input_is_aces)
    if is_exr_reference:
        assert input_path.suffix.lower() == ".exr", "--input-is-aces expects an EXR input"
    assert args.dequant_checkpoint, "--dequant-checkpoint is required"

    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)

    default_output_dir = project_root / "outputs" / "inference"
    input_stem = input_path.stem

    if args.output:
        output_candidate = Path(args.output)
        output_dir = output_candidate.parent if str(output_candidate.parent) not in {"", "."} else default_output_dir
        pred_aces_name = output_candidate.name
    else:
        output_dir = default_output_dir
        pred_aces_name = f"{input_stem}_out.exr"

    dequant_name = Path(args.dequant_output).name if args.dequant_output else f"{input_stem}_dequant.exr"
    plot_name = f"{input_stem}_plot.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_dequant_output = output_dir / dequant_name
    resolved_pred_aces_output = output_dir / pred_aces_name
    resolved_plot_output = output_dir / plot_name

    look = None
    dequant_input_override_chw: torch.Tensor | None = None
    clean_reference_chw: torch.Tensor | None = None
    dequant_reference_chw: torch.Tensor | None = None
    if is_exr_reference:
        print(f"Creating CDL look for inference: {args.look_mode}")
        look = build_look(args.look_mode, args.slope, args.offset, args.power, args.saturation)

        print("Preparing dequant input from ACES reference (apply look -> ACES to sRGB -> 8-bit quantize)")
        aces_chw = image_to_tensor(input_path).to(device=device, dtype=torch.float32)
        aces_hwc = aces_chw.permute(1, 2, 0)
        cdl_processor = GPUCDLProcessor(device=device)
        transformer = ACESColorTransformer(device=device, use_lut=True)

        # Clean reference for final dashboard panel: ACES -> sRGB, no look.
        clean_srgb_hwc = transformer.aces_to_srgb_32f(aces_hwc.unsqueeze(0)).squeeze(0)
        clean_srgb_hwc = torch.clamp(clean_srgb_hwc, 0.0, 1.0)
        clean_reference_chw = clean_srgb_hwc.permute(2, 0, 1).detach().cpu()

        aces_graded_hwc = cdl_processor.apply_cdl_gpu(aces_hwc, look)
        srgb_hwc = transformer.aces_to_srgb_32f(aces_graded_hwc.unsqueeze(0)).squeeze(0)
        srgb_hwc = torch.clamp(srgb_hwc, 0.0, 1.0)
        dequant_reference_chw = srgb_hwc.permute(2, 0, 1).detach().cpu()
        srgb_hwc = torch.round(srgb_hwc * 255.0) / 255.0
        dequant_input_override_chw = srgb_hwc.permute(2, 0, 1)

    temporary_dequant_path: Path | None = None
    if args.save_dequant:
        dequant_stage_output = resolved_dequant_output
        temporary_dequant_path = None
    else:
        temporary_dequant_path = Path(
            tempfile.NamedTemporaryFile(
                suffix="_dequant_tmp.exr",
                dir=output_dir,
                delete=False,
            ).name
        )
        dequant_stage_output = temporary_dequant_path

    print(f"Loading dequant model on {device}...")
    dequant_model = load_dequant_model_from_checkpoint(
        checkpoint_path=args.dequant_checkpoint,
        device=device,
        channels=args.dequant_channels,
    )
    if amp_enabled:
        dequant_model = dequant_model.to(dtype=amp_dtype)

    print(f"Running dequant inference for {input_path}...")

    dequant_input_for_mapper, dequant_input_chw_cpu, dequant_output_chw_cpu = run_dequant_inference(
        model=dequant_model,
        input_path=input_path,
        output_path=dequant_stage_output,
        device=device,
        align_multiple=args.align_multiple,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        quantize_input=False,
        input_chw_override=dequant_input_override_chw,
    )

    if args.save_dequant:
        print(f"Saved dequantized intermediate: {dequant_input_for_mapper}")

    print(f"Loading ACES mapper model on {device}...")
    mapper_model = load_model_from_checkpoint(
        checkpoint_path=args.mapper_checkpoint,
        device=device,
        num_luts=args.num_luts,
        lut_dim=args.lut_dim,
        num_lap=args.num_lap,
        num_residual_blocks=args.num_residual_blocks,
    )
    # Keep the mapper in float32: its spatial-frequency transformer uses
    # complex operators that fail under ComplexHalf on CUDA.

    print("Running ACES mapper inference on dequant output...")
    if args.keep_padding:
        print("Keeping aligned padded output dimensions (--keep-padding set)")
    else:
        print("Removing alignment padding from generated outputs (default behavior)")

    pred_aces_chw_cpu, mapper_srgb_chw_cpu = run_mapper_inference_on_srgb(
        model=mapper_model,
        input_srgb_chw=dequant_output_chw_cpu,
        crop_size=args.crop_size,
        align_multiple=args.align_multiple,
        max_side=args.max_side,
        keep_aligned_output=args.keep_padding,
        device=device,
    )

    if not args.no_save_output:
        write_exr(resolved_pred_aces_output, pred_aces_chw_cpu.numpy())

    try:
        if args.save_plot:
            reference_for_dashboard = clean_reference_chw if is_exr_reference else None
            assert dequant_reference_chw is not None, (
                "Dequant reference (look-applied sRGB) is required for dashboard metrics; "
                "run with --input-is-aces to generate it"
            )
            save_full_inference_dashboard(
                input_srgb_chw=dequant_input_chw_cpu,
                dequant_srgb_chw=dequant_output_chw_cpu,
                mapper_srgb_chw=mapper_srgb_chw_cpu,
                reference_srgb_chw=reference_for_dashboard,
                dequant_reference_srgb_chw=dequant_reference_chw,
                save_path=resolved_plot_output,
            )
    finally:
        if not args.save_dequant and temporary_dequant_path is not None and temporary_dequant_path.exists():
            temporary_dequant_path.unlink()

    if look is not None:
        print(f"CDL look used: {look}")
    else:
        print("CDL look skipped: only used for ACES validation with --input-is-aces")

    print("===========================================================================")

    if args.no_save_output:
        print("Predicted ACES EXR not saved (--no-save-output set)")
    else:
        print(f"Saved predicted ACES EXR: {resolved_pred_aces_output}")

    if args.save_dequant:
        print(f"Saved dequantized intermediate: {resolved_dequant_output}")
    else:
        print("Dequantized intermediate not saved (use --save-dequant to keep it)")

    if args.save_plot:
        print(f"Saved comparison dashboard: {resolved_plot_output}")
    else:
        print("Comparison dashboard not saved (--save-plot not set)")


if __name__ == "__main__":
    main()