"""Shared ACES mapper inference and visualization helpers."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from luminascale.models.aces_mapper import ACESMapper
from luminascale.utils.gpu_cdl_processor import GPUCDLProcessor
from luminascale.utils.io import image_to_tensor, read_exr, write_exr
from luminascale.utils.look_generator import CDLParameters, get_single_random_look
from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer


def parse_triplet(value: str) -> tuple[float, float, float]:
    """Parse comma-separated float triplet (e.g. '1.0,1.0,1.0')."""
    parts = [part.strip() for part in value.split(",")]
    assert len(parts) == 3, f"Expected 3 comma-separated values, got: {value}"
    return float(parts[0]), float(parts[1]), float(parts[2])


def center_crop_hwc(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Center-crop image [H, W, C] if crop_size is positive and smaller than image size."""
    if crop_size <= 0:
        return image

    height, width, _ = image.shape
    assert crop_size <= height and crop_size <= width, (
        f"crop_size={crop_size} must be <= image size ({height}x{width})"
    )

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top : top + crop_size, left : left + crop_size, :]


def align_to_multiple_hwc(image: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad image [H, W, C] to nearest multiple using edge-value replication."""
    if multiple <= 1:
        return image

    height, width, _ = image.shape
    new_height = ((height + multiple - 1) // multiple) * multiple
    new_width = ((width + multiple - 1) // multiple) * multiple

    if new_height == height and new_width == width:
        return image

    pad_height = new_height - height
    pad_width = new_width - width

    chw = image.permute(2, 0, 1).unsqueeze(0)
    chw = F.pad(chw, (0, pad_width, 0, pad_height), mode="replicate")
    return chw.squeeze(0).permute(1, 2, 0)


def resize_to_max_side_hwc(image: torch.Tensor, max_side: int) -> torch.Tensor:
    """Resize image [H, W, C] if its longest side exceeds max_side."""
    if max_side <= 0:
        return image

    height, width, _ = image.shape
    current_max = max(height, width)
    if current_max <= max_side:
        return image

    scale = max_side / float(current_max)
    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))

    chw = image.permute(2, 0, 1).unsqueeze(0)
    chw = F.interpolate(chw, size=(new_height, new_width), mode="bilinear", align_corners=False)
    return chw.squeeze(0).permute(1, 2, 0)


def build_look(
    look_mode: str = "random",
    slope: str = "1.0,1.0,1.0",
    offset: str = "0.0,0.0,0.0",
    power: str = "1.0,1.0,1.0",
    saturation: float = 1.0,
) -> CDLParameters:
    """Create a random or manual CDL look."""
    if look_mode == "manual":
        return CDLParameters(
            slope=parse_triplet(slope),
            offset=parse_triplet(offset),
            power=parse_triplet(power),
            saturation=float(saturation),
        )
    return get_single_random_look()


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
    num_luts: int,
    lut_dim: int,
    num_lap: int,
    num_residual_blocks: int,
) -> ACESMapper:
    """Load an ACESMapper checkpoint into an inference model."""
    model = ACESMapper(
        num_luts=num_luts,
        lut_dim=lut_dim,
        num_lap=num_lap,
        num_residual_blocks=num_residual_blocks,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    assert isinstance(state_dict, dict), "Checkpoint state_dict is not a dictionary"

    model_prefixed = [key for key in state_dict.keys() if key.startswith("model.")]
    if model_prefixed:
        state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items() if key.startswith("model.")}

    if all(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    load_result = model.load_state_dict(state_dict, strict=True)
    assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"

    model.eval()
    return model


def to_hwc_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert tensor image [H, W, C] to clipped numpy float32."""
    return torch.clamp(image, 0.0, 1.0).detach().cpu().numpy().astype(np.float32)


def save_comparison_grid(
    input_srgb_hwc: torch.Tensor,
    pred_srgb_hwc: torch.Tensor,
    ref_srgb_hwc: torch.Tensor,
    pred_aces_hwc: torch.Tensor,
    ref_aces_hwc: torch.Tensor,
    save_path: Path | None,
) -> Figure:
    """Build and save a compact 2x3 comparison dashboard for ACES mapper output quality."""
    input_np = to_hwc_numpy(input_srgb_hwc)
    pred_np = to_hwc_numpy(pred_srgb_hwc)
    ref_np = to_hwc_numpy(ref_srgb_hwc)

    pred_aces_np = pred_aces_hwc.detach().cpu().numpy().astype(np.float32)
    ref_aces_np = ref_aces_hwc.detach().cpu().numpy().astype(np.float32)

    diff_aces = np.mean(np.abs(pred_aces_np - ref_aces_np), axis=2)
    residual_disp = np.mean(pred_np - ref_np, axis=2)
    pred_luma = np.mean(pred_np, axis=2)
    ref_luma = np.mean(ref_np, axis=2)

    mse_disp = float(np.mean((pred_np - ref_np) ** 2))
    psnr_disp = 10.0 * np.log10(1.0 / (mse_disp + 1e-8))

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.35,
        wspace=0.30,
        left=0.06,
        right=0.97,
        top=0.92,
        bottom=0.07,
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(input_np, aspect="auto")
    ax00.set_title("Input sRGB", fontsize=10, fontweight="bold")
    ax00.axis("off")

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(pred_np, aspect="auto")
    ax01.set_title("Model Output", fontsize=10, fontweight="bold")
    ax01.axis("off")

    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(ref_np, aspect="auto")
    ax02.set_title("Reference", fontsize=10, fontweight="bold")
    ax02.axis("off")

    ax10 = fig.add_subplot(gs[1, 0])
    im0 = ax10.imshow(diff_aces, cmap="magma", aspect="auto")
    ax10.set_title("ACES difference", fontsize=10, fontweight="bold")
    ax10.axis("off")
    cbar0 = plt.colorbar(im0, ax=ax10, fraction=0.046, pad=0.04, shrink=0.8)
    cbar0.ax.tick_params(labelsize=8)

    ax11 = fig.add_subplot(gs[1, 1])
    residual_limit = float(np.max(np.abs(residual_disp)))
    im1 = ax11.imshow(
        residual_disp,
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,      
        aspect="auto",
    )
    ax11.set_title("Display residual", fontsize=10, fontweight="bold")
    ax11.axis("off")
    cbar1 = plt.colorbar(im1, ax=ax11, fraction=0.046, pad=0.04, shrink=0.8)
    cbar1.ax.tick_params(labelsize=8)

    flat_ref = ref_luma.reshape(-1)
    flat_pred = pred_luma.reshape(-1)
    sample_count = min(10000, flat_ref.size)
    sample_indices = np.linspace(0, flat_ref.size - 1, sample_count, dtype=np.int64)
    sample_ref = flat_ref[sample_indices]
    sample_pred = flat_pred[sample_indices]

    ax12 = fig.add_subplot(gs[1, 2])
    ax12.scatter(sample_ref, sample_pred, s=2, alpha=0.2, color="#2b6cb0", edgecolors="none")
    diag_min = float(min(sample_ref.min(), sample_pred.min()))
    diag_max = float(max(sample_ref.max(), sample_pred.max()))
    ax12.plot([diag_min, diag_max], [diag_min, diag_max], "k--", linewidth=1.0, alpha=0.7)
    ax12.set_xlim(diag_min, diag_max)
    ax12.set_ylim(diag_min, diag_max)
    ax12.set_title("Luma parity", fontsize=10, fontweight="bold")
    ax12.set_xlabel("Reference Luma", fontsize=9)
    ax12.set_ylabel("Predicted Luma", fontsize=9)
    ax12.grid(True, alpha=0.25, linestyle="--")
    ax12.tick_params(labelsize=8)

    fig.suptitle("ACES Mapper Inference Dashboard", fontsize=12, fontweight="bold")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def save_input_output_grid(
    input_srgb_hwc: torch.Tensor,
    pred_srgb_hwc: torch.Tensor,
    save_path: Path | None,
) -> Figure:
    """Build and save a compact 1x2 dashboard when no ACES reference is available."""
    input_np = to_hwc_numpy(input_srgb_hwc)
    pred_np = to_hwc_numpy(pred_srgb_hwc)

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        hspace=0.20,
        wspace=0.20,
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.06,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(input_np, aspect="auto")
    ax0.set_title("Input sRGB", fontsize=10, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(pred_np, aspect="auto")
    ax1.set_title("Model Output", fontsize=10, fontweight="bold")
    ax1.axis("off")

    fig.suptitle("ACES Mapper Inference Dashboard (No Reference)", fontsize=12, fontweight="bold")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def close_figure(figure: Figure | None) -> None:
    """Close a matplotlib figure if one was created."""
    if figure is not None:
        plt.close(figure)


def run_aces_mapper_inference(
    model: torch.nn.Module,
    input: Path | str,
    output_path: Path | str | None,
    look: CDLParameters | None = None,
    *,
    crop_size: int = 0,
    align_multiple: int = 64,
    max_side: int = 1024,
    pred_aces_output: Path | str | None = None,
    input_is_aces: bool = True,
    keep_aligned_output: bool = False,
    device: torch.device | str | None = None,
) -> Figure:
    """Run the ACES mapper visualization pipeline and return the matplotlib figure."""
    input_path = Path(input)
    save_path = Path(output_path) if output_path is not None else None
    has_aces_reference = bool(input_is_aces)
    if has_aces_reference and look is None:
        look = build_look()

    forward_model = model.model if hasattr(model, "model") and isinstance(model.model, torch.nn.Module) else model
    reference_parameter = next(forward_model.parameters())
    inference_device = torch.device(device) if device is not None else reference_parameter.device
    model_dtype = reference_parameter.dtype
    use_autocast = inference_device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}

    transformer = ACESColorTransformer(device=inference_device, use_lut=True)
    if has_aces_reference:
        aces_chw = read_exr(input_path)
        ref_aces_hwc = torch.from_numpy(aces_chw.transpose(1, 2, 0)).to(
            device=inference_device,
            dtype=torch.float32,
        )
        ref_aces_hwc = center_crop_hwc(ref_aces_hwc, crop_size)
        ref_aces_hwc = resize_to_max_side_hwc(ref_aces_hwc, max_side)
        output_height, output_width, _ = ref_aces_hwc.shape
        ref_aces_hwc = align_to_multiple_hwc(ref_aces_hwc, align_multiple)

        cdl_processor = GPUCDLProcessor(device=inference_device)
        aces_graded_hwc = cdl_processor.apply_cdl_gpu(ref_aces_hwc, look)
        input_srgb_hwc = transformer.aces_to_srgb_32f(aces_graded_hwc.unsqueeze(0)).squeeze(0)
        input_srgb_hwc = torch.clamp(input_srgb_hwc, 0.0, 1.0)
    else:
        input_chw = image_to_tensor(input_path)
        input_srgb_hwc = input_chw.permute(1, 2, 0).to(device=inference_device, dtype=torch.float32)
        input_srgb_hwc = center_crop_hwc(input_srgb_hwc, crop_size)
        input_srgb_hwc = resize_to_max_side_hwc(input_srgb_hwc, max_side)
        output_height, output_width, _ = input_srgb_hwc.shape
        input_srgb_hwc = align_to_multiple_hwc(input_srgb_hwc, align_multiple)
        input_srgb_hwc = torch.clamp(input_srgb_hwc, 0.0, 1.0)

    input_chw = input_srgb_hwc.permute(2, 0, 1).unsqueeze(0)
    model_input = input_chw if use_autocast else input_chw.to(dtype=model_dtype)
    amp_ctx = torch.autocast(device_type="cuda", dtype=model_dtype) if use_autocast else nullcontext()

    with torch.inference_mode(), amp_ctx:
        pred_aces_chw = forward_model(model_input).squeeze(0)

    pred_aces_hwc = pred_aces_chw.permute(1, 2, 0)
    pred_srgb_hwc = transformer.aces_to_srgb_32f(pred_aces_hwc.unsqueeze(0)).squeeze(0)

    if not keep_aligned_output and align_multiple > 1:
        input_srgb_hwc = input_srgb_hwc[:output_height, :output_width, :]
        pred_aces_hwc = pred_aces_hwc[:output_height, :output_width, :]
        pred_srgb_hwc = pred_srgb_hwc[:output_height, :output_width, :]
        pred_aces_chw = pred_aces_chw[:, :output_height, :output_width]
        if has_aces_reference:
            ref_aces_hwc = ref_aces_hwc[:output_height, :output_width, :]

    if has_aces_reference:
        ref_srgb_hwc = transformer.aces_to_srgb_32f(ref_aces_hwc.unsqueeze(0)).squeeze(0)
        fig = save_comparison_grid(
            input_srgb_hwc=input_srgb_hwc,
            pred_srgb_hwc=pred_srgb_hwc,
            ref_srgb_hwc=ref_srgb_hwc,
            pred_aces_hwc=pred_aces_hwc,
            ref_aces_hwc=ref_aces_hwc,
            save_path=save_path,
        )
    else:
        fig = save_input_output_grid(
            input_srgb_hwc=input_srgb_hwc,
            pred_srgb_hwc=pred_srgb_hwc,
            save_path=save_path,
        )

    if pred_aces_output:
        pred_exr_path = Path(pred_aces_output)
        pred_exr_path.parent.mkdir(parents=True, exist_ok=True)
        write_exr(pred_exr_path, pred_aces_chw.detach().cpu().numpy())

    return fig