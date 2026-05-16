#!/usr/bin/env python3
"""Evaluate LuminaScale quantitative experiments on the WebDataset test split.

This script implements the report protocol from `main.tex`:
- Quantitative comparison rows:
  1. deterministic baseline (sRGB -> ACES via fixed inverse/OETF + matrix)
  2. Dequant-only
  3. Mapper-only
  4. Full pipeline (Dequant -> Mapper)
- Dequant augmentation ablation:
  - evaluates dequantized 32-bit sRGB output before the mapper stage

Outputs:
- terminal text summary
- text log summary (.log)
- optional sampled PNG dashboards
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import tempfile
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import OpenImageIO as oiio
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from luminascale.data.wds_dataset import LuminaScaleWebDataset
from luminascale.models import create_dequant_net
from luminascale.utils.aces_mapper_inference import (
    align_to_multiple_hwc,
    center_crop_hwc,
    load_model_from_checkpoint,
    resize_to_max_side_hwc,
)
from luminascale.utils.gpu_cdl_processor import GPUCDLProcessor
from luminascale.utils.look_generator import CDLParameters, get_single_random_look
from luminascale.utils.metrics import compute_ssim
from luminascale.utils.pytorch_aces_transformer import ACESColorTransformer, ACESMatrices
from scripts.run_full_inference import save_full_inference_dashboard


DEFAULT_DEQUANT_CHECKPOINTS = [
    "outputs/training/dequant/20260514_173132/checkpoints/dequant-20260514_173132-49_psnr59-95.ckpt",
    "outputs/training/dequant/20260515_003808/checkpoints/dequant-20260515_003808-49_psnr60-00.ckpt",
    "outputs/training/dequant/20260515_073550/checkpoints/dequant-20260515_073550-49_psnr60-14.ckpt",
]
DEFAULT_QUANTITATIVE_DEQUANT_CHECKPOINT = (
    "outputs/training/dequant/20260515_073550/checkpoints/dequant-20260515_073550-49_psnr60-14.ckpt"
)
DEFAULT_MAPPER_CHECKPOINT = "checkpoints/aces-mapper-20260511_214649-96_psnr30-72.ckpt"
DEFAULT_TEST_SHARD_PATH = "dataset/ACEScct/full/shards/test"
DEFAULT_METADATA_PARQUET = "dataset/ACEScct/full/training_metadata.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LuminaScale experiments on the test split.")
    parser.add_argument(
        "--dequant-checkpoints",
        nargs="+",
        default=DEFAULT_DEQUANT_CHECKPOINTS,
        help="One or more Dequant checkpoints for the ablation / full-pipeline evaluation.",
    )
    parser.add_argument(
        "--mapper-checkpoint",
        type=str,
        default=DEFAULT_MAPPER_CHECKPOINT,
        help="Mapper checkpoint used for mapper-only and full-pipeline evaluation.",
    )
    parser.add_argument(
        "--quantitative-dequant-checkpoint",
        type=str,
        default=DEFAULT_QUANTITATIVE_DEQUANT_CHECKPOINT,
        help="Single Dequant checkpoint used for the quantitative comparison rows.",
    )
    parser.add_argument(
        "--test-shard-path",
        type=str,
        default=DEFAULT_TEST_SHARD_PATH,
        help="WebDataset shard directory or pattern for the test split.",
    )
    parser.add_argument(
        "--metadata-parquet",
        type=str,
        default=DEFAULT_METADATA_PARQUET,
        help="Parquet metadata used to estimate the test-set size and resolve split metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/full_pipeline",
        help="Directory for all output artifacts.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="summary.log",
        help="Text summary log file name relative to output-dir.",
    )
    parser.add_argument(
        "--dashboard-samples",
        type=int,
        default=0,
        help="Number of samples per dequant variant to save qualitative dashboards for.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of test samples (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Base seed used for deterministic per-sample random CDL look generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="WebDataset loader batch size. Use 1 for deterministic per-sample evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data loader workers for the test WebDataset loader.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="WebLoader prefetch factor when num-workers > 0.",
    )
    parser.add_argument(
        "--align-multiple",
        type=int,
        default=64,
        help="Pad tensors to the nearest higher multiple before model inference.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Optional cap on longest side before mapper inference (0 disables resizing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device.",
    )
    parser.add_argument(
        "--dequant-channels",
        type=int,
        default=32,
        help="Base channels for DequantNet checkpoint instantiation.",
    )
    return parser.parse_args()


def ensure_ocio_env() -> None:
    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ.setdefault("OCIO", str(ocio_config))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def decode_exr_bytes_to_hwc(exr_bytes: bytes) -> torch.Tensor:
    """Decode EXR bytes to CPU tensor [H, W, 3] float32."""
    temp_file: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as handle:
            handle.write(exr_bytes)
            temp_file = handle.name

        image_input = oiio.ImageInput.open(temp_file)
        assert image_input is not None, "Failed to open EXR bytes with OpenImageIO"
        spec = image_input.spec()
        pixels = image_input.read_image("float")
        image_input.close()
        assert pixels is not None, "Failed to decode EXR bytes"
        array = np.asarray(pixels, dtype=np.float32).reshape((spec.height, spec.width, spec.nchannels))
        if array.shape[2] > 3:
            array = array[:, :, :3]
        return torch.from_numpy(array.copy()).to(torch.float32)
    finally:
        if temp_file is not None:
            Path(temp_file).unlink(missing_ok=True)


def checkpoint_label(checkpoint_path: Path) -> str:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent.name
    return checkpoint_path.stem


def sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def deterministic_random_look(sample_id: str, base_seed: int) -> tuple[CDLParameters, int]:
    seed_value = sample_seed(base_seed, sample_id)
    previous_state = random.getstate()
    try:
        random.seed(seed_value)
        return get_single_random_look(), seed_value
    finally:
        random.setstate(previous_state)


def align_to_multiple_bchw(image: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Pad BCHW tensor to nearest multiple using edge replication."""
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


def apply_matrix_last_dim(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    original_shape = values.shape
    flat = values.reshape(-1, 3)
    transformed = flat @ matrix.t()
    return transformed.reshape(original_shape)


def invert_srgb_oetf(srgb: torch.Tensor) -> torch.Tensor:
    srgb = torch.clamp(srgb, 0.0, 1.0)
    return torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
    )


def srgb_to_aces2065_1(
    srgb_chw: torch.Tensor,
    rec709_to_ap0_matrix: torch.Tensor,
) -> torch.Tensor:
    """Deterministic baseline: inverse sRGB OETF then fixed linear color matrix to AP0."""
    srgb_hwc = srgb_chw.permute(1, 2, 0)
    linear_rec709 = invert_srgb_oetf(srgb_hwc)
    ap0_hwc = apply_matrix_last_dim(linear_rec709, rec709_to_ap0_matrix)
    return ap0_hwc.permute(2, 0, 1)


def ap0_to_acescct(ap0_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert ACES2065-1/AP0 linear tensor [C, H, W] to ACEScct [C, H, W]."""
    matrix_ap0_to_ap1 = ACESMatrices.M_AP0_TO_AP1.to(device)
    ap0_hwc = ap0_chw.permute(1, 2, 0).to(device=device, dtype=torch.float32)
    ap1_linear = apply_matrix_last_dim(ap0_hwc, matrix_ap0_to_ap1)

    linear_break = 0.0078125
    linear_branch = (10.5402377416545 * ap1_linear) + 0.0729055341958355
    safe_for_log = torch.clamp(ap1_linear, min=2.0 ** -16, max=65504.0)
    log_branch = (torch.log2(safe_for_log) + 9.72) / 17.52
    acescct_hwc = torch.where(ap1_linear <= linear_break, linear_branch, log_branch)
    return acescct_hwc.permute(2, 0, 1)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    mse = torch.mean((pred - target) ** 2).item()
    return float(10.0 * np.log10(1.0 / (mse + 1e-10)))


def compute_lpips_metric(
    metric: lpips.LPIPS,
    pred_chw: torch.Tensor,
    target_chw: torch.Tensor,
    device: torch.device,
) -> float:
    pred_bchw = pred_chw.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_bchw = target_chw.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        value = metric(pred_bchw, target_bchw, normalize=True).mean().item()
    return float(value)


def resize_chw_to(chw: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if chw.shape[1] == height and chw.shape[2] == width:
        return chw
    resized = F.interpolate(chw.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def run_dequant_model(
    model: torch.nn.Module,
    input_chw: torch.Tensor,
    device: torch.device,
    align_multiple: int,
) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    input_bchw = input_chw.unsqueeze(0).to(device=device, dtype=torch.float32)
    aligned_bchw, original_height, original_width = align_to_multiple_bchw(input_bchw, align_multiple)
    with torch.inference_mode():
        output_bchw = model(aligned_bchw)
    output_chw = output_bchw.squeeze(0)[:, :original_height, :original_width].detach().cpu()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output_chw, float(elapsed_ms)


def run_mapper_model(
    model: torch.nn.Module,
    input_srgb_chw: torch.Tensor,
    device: torch.device,
    align_multiple: int,
    max_side: int,
    display_transformer: ACESColorTransformer,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    start = time.perf_counter()
    input_srgb_hwc = input_srgb_chw.permute(1, 2, 0).to(device=device, dtype=torch.float32)
    input_srgb_hwc = resize_to_max_side_hwc(input_srgb_hwc, max_side)
    output_height, output_width, _ = input_srgb_hwc.shape
    aligned_hwc = align_to_multiple_hwc(input_srgb_hwc, align_multiple)
    aligned_hwc = torch.clamp(aligned_hwc, 0.0, 1.0)

    forward_model = model.model if hasattr(model, "model") and isinstance(model.model, torch.nn.Module) else model
    model_input = aligned_hwc.permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        pred_aces_bchw, _ = forward_model(model_input)

    pred_aces_chw = pred_aces_bchw.squeeze(0)
    pred_aces_chw = pred_aces_chw[:, :output_height, :output_width].detach().cpu()

    target_color_space = getattr(
        forward_model,
        "target_color_space",
        getattr(model, "target_color_space", "ACEScct"),
    )
    pred_srgb_hwc = display_transformer.aces_to_srgb_32f(
        pred_aces_chw.permute(1, 2, 0).unsqueeze(0).to(device=device),
        input_cs=target_color_space,
    ).squeeze(0)
    pred_srgb_chw = pred_srgb_hwc.permute(2, 0, 1).detach().cpu()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return pred_aces_chw, pred_srgb_chw, float(elapsed_ms)


def build_display_reference_and_input(
    target_acescct_hwc: torch.Tensor,
    sample_id: str,
    base_seed: int,
    cdl_processor: GPUCDLProcessor,
    transformer: ACESColorTransformer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, CDLParameters]:
    """Create deterministic look-applied 8-bit input and clean display reference."""
    look, look_seed = deterministic_random_look(sample_id, base_seed)
    clean_display_hwc = transformer.aces_to_srgb_32f(
        target_acescct_hwc.unsqueeze(0).to(device=device),
        input_cs="ACEScct",
    ).squeeze(0)
    graded_acescct_hwc = cdl_processor.apply_cdl_gpu(target_acescct_hwc.to(device=device), look)
    look_display_hwc = transformer.aces_to_srgb_32f(
        graded_acescct_hwc.unsqueeze(0),
        input_cs="ACEScct",
    ).squeeze(0)
    look_display_hwc = torch.clamp(look_display_hwc, 0.0, 1.0)
    quantized_input_hwc = torch.round(look_display_hwc * 255.0) / 255.0
    return (
        quantized_input_hwc.permute(2, 0, 1).detach().cpu(),
        look_display_hwc.permute(2, 0, 1).detach().cpu(),
        clean_display_hwc.permute(2, 0, 1).detach().cpu(),
        look_seed,
        look,
    )


def make_metric_row(
    *,
    sample_id: str,
    split: str,
    source: str,
    experiment: str,
    variant: str,
    pipeline_row: str,
    look_seed: int,
    psnr_acescct: float | None = None,
    ssim_acescct: float | None = None,
    lpips_srgb: float | None = None,
    psnr_srgb: float | None = None,
    ssim_srgb: float | None = None,
    dequant_time_ms: float | None = None,
    mapper_time_ms: float | None = None,
    total_time_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "source": source,
        "experiment": experiment,
        "variant": variant,
        "pipeline_row": pipeline_row,
        "look_seed": int(look_seed),
        "psnr_acescct": psnr_acescct,
        "ssim_acescct": ssim_acescct,
        "lpips_srgb": lpips_srgb,
        "psnr_srgb": psnr_srgb,
        "ssim_srgb": ssim_srgb,
        "dequant_time_ms": dequant_time_ms,
        "mapper_time_ms": mapper_time_ms,
        "total_time_ms": total_time_ms,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    numeric_keys = [
        "psnr_acescct",
        "ssim_acescct",
        "lpips_srgb",
        "psnr_srgb",
        "ssim_srgb",
        "dequant_time_ms",
        "mapper_time_ms",
        "total_time_ms",
    ]

    for row in rows:
        group_key = (row["experiment"], row["variant"], row["pipeline_row"])
        for metric_key in numeric_keys:
            value = row.get(metric_key)
            if value is None:
                continue
            grouped[group_key][metric_key].append(float(value))

    aggregate: list[dict[str, Any]] = []
    for (experiment, variant, pipeline_row), metric_map in sorted(grouped.items()):
        for metric_name, values in sorted(metric_map.items()):
            values_np = np.asarray(values, dtype=np.float64)
            aggregate.append(
                {
                    "experiment": experiment,
                    "variant": variant,
                    "pipeline_row": pipeline_row,
                    "metric": metric_name,
                    "count": int(values_np.size),
                    "mean": float(values_np.mean()),
                    "std": float(values_np.std(ddof=0)),
                    "min": float(values_np.min()),
                    "max": float(values_np.max()),
                }
            )
    return aggregate


def load_dequant_variant_config(checkpoint_path: Path) -> dict[str, Any]:
    config_path = checkpoint_path.parent.parent / "config.yaml"
    config_data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}

    target_blur = config_data.get("target_blur", {}) if isinstance(config_data, dict) else {}
    return {
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path) if config_path.exists() else "",
        "start_sigma": target_blur.get("start_sigma"),
        "end_sigma": target_blur.get("end_sigma"),
        "bit_crunch_contrast_min": config_data.get("bit_crunch_contrast_min") if isinstance(config_data, dict) else None,
        "bit_crunch_contrast_max": config_data.get("bit_crunch_contrast_max") if isinstance(config_data, dict) else None,
    }


def aggregate_lookup(aggregate_rows_data: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["experiment"], row["variant"], row["pipeline_row"], row["metric"]): row
        for row in aggregate_rows_data
    }


def format_mean_std(value: dict[str, Any] | None, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value['mean']:.{precision}f} ± {value['std']:.{precision}f}"


def render_text_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, col in enumerate(row):
            widths[idx] = max(widths[idx], len(str(col)))

    def format_row(cols: list[str]) -> str:
        return " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(cols))

    header_line = format_row(headers)
    separator_line = "-+-".join("-" * width for width in widths)
    body_lines = [format_row(row) for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def build_summary_text(
    *,
    args: argparse.Namespace,
    processed_samples: int,
    aggregate_rows_data: list[dict[str, Any]],
    dequant_labels: dict[str, dict[str, Any]],
    quantitative_dequant_variant: str,
) -> str:
    lookup = aggregate_lookup(aggregate_rows_data)

    mapper_id_match = re.search(r"aces-mapper-(\d{8}_\d{6})", str(args.mapper_checkpoint))
    mapper_checkpoint_id = mapper_id_match.group(1) if mapper_id_match else Path(args.mapper_checkpoint).stem

    def build_quant_rows(specs: list[tuple[str, str, str, str]]) -> list[list[str]]:
        rows: list[list[str]] = []
        for disp_row, disp_variant, lookup_row, lookup_variant in specs:
            is_dequant_variant = lookup_variant in dequant_labels
            variant_meta = dequant_labels.get(lookup_variant, {})
            psnr = lookup.get(("quantitative_comparison", lookup_variant, lookup_row, "psnr_acescct"))
            ssim = lookup.get(("quantitative_comparison", lookup_variant, lookup_row, "ssim_acescct"))
            lpips_value = lookup.get(("quantitative_comparison", lookup_variant, lookup_row, "lpips_srgb"))
            total_time = lookup.get(("quantitative_comparison", lookup_variant, lookup_row, "total_time_ms"))
            rows.append(
                [
                    disp_row,
                    disp_variant,
                    str(variant_meta.get("start_sigma", "-")) if is_dequant_variant else "-",
                    str(variant_meta.get("bit_crunch_contrast_max", "-")) if is_dequant_variant else "-",
                    format_mean_std(psnr, precision=4),
                    format_mean_std(ssim, precision=4),
                    format_mean_std(lpips_value, precision=4),
                    format_mean_std(total_time, precision=2),
                ]
            )
        return rows

    full_pipeline_rows = build_quant_rows([
        ("full_baseline", "n/a", "full_baseline", "shared"),
        ("full_pipeline", quantitative_dequant_variant, "full_pipeline", quantitative_dequant_variant),
    ])
    dequant_rows = build_quant_rows([
        ("dequant_baseline", "n/a", "dequant_baseline", "shared"),
        ("dequant_only", quantitative_dequant_variant, "dequant_only", quantitative_dequant_variant),
    ])
    mapper_rows = build_quant_rows([
        ("mapper_baseline", "n/a", "mapper_baseline", "shared"),
        ("mapper_only", mapper_checkpoint_id, "mapper_only", "shared"),
    ])

    ablation_table_rows: list[list[str]] = []
    dequant_baseline_psnr = lookup.get(("dequant_ablation", "shared", "dequant_baseline", "psnr_srgb"))
    dequant_baseline_ssim = lookup.get(("dequant_ablation", "shared", "dequant_baseline", "ssim_srgb"))
    dequant_baseline_total = lookup.get(("dequant_ablation", "shared", "dequant_baseline", "total_time_ms"))
    ablation_table_rows.append(
        [
            "dequant_baseline",
            "-",
            "-",
            format_mean_std(dequant_baseline_psnr, precision=4),
            format_mean_std(dequant_baseline_ssim, precision=4),
            format_mean_std(dequant_baseline_total, precision=2),
        ]
    )
    for label in dequant_labels:
        variant_meta = dequant_labels[label]
        psnr = lookup.get(("dequant_ablation", label, "dequant_output_srgb", "psnr_srgb"))
        ssim = lookup.get(("dequant_ablation", label, "dequant_output_srgb", "ssim_srgb"))
        total_time = lookup.get(("dequant_ablation", label, "dequant_output_srgb", "total_time_ms"))
        ablation_table_rows.append(
            [
                label,
                str(variant_meta.get("start_sigma", "-")),
                str(variant_meta.get("bit_crunch_contrast_max", "-")),
                format_mean_std(psnr, precision=4),
                format_mean_std(ssim, precision=4),
                format_mean_std(total_time, precision=2),
            ]
        )

    dequant_checkpoint_lines = []
    for label, meta in dequant_labels.items():
        dequant_checkpoint_lines.append(
            f"- `{label}` → `{meta['checkpoint']}` (start_sigma={meta.get('start_sigma')}, bit_crunch_contrast_max={meta.get('bit_crunch_contrast_max')})"
        )
    dequant_checkpoint_block = "\n".join(dequant_checkpoint_lines)

    content = f"""LuminaScale Evaluation Summary

Run configuration
-----------------

- Samples processed: `{processed_samples}`
- Test shard path: `{args.test_shard_path}`
- Metadata parquet: `{args.metadata_parquet}`
- Mapper checkpoint: `{args.mapper_checkpoint}`
- Seed: `{args.seed}`
- Device: `{args.device}`
- Align multiple: `{args.align_multiple}`
- Max side: `{args.max_side}`

Dequant variants
----------------

{dequant_checkpoint_block}

Stage baselines
---------------

- Dequant stage comparison: `dequant_baseline` compares raw 8-bit looked `sRGB` input directly against 32-bit looked `sRGB` target.
- Mapper stage comparison: `mapper_baseline` compares raw 32-bit looked `sRGB` directly against untouched 32-bit `ACEScct` target.
- Full pipeline comparison: `full_baseline` compares raw 8-bit looked `sRGB` directly against untouched 32-bit `ACEScct` target.

Metric domains
--------------

- All metrics are computed directly on raw candidate vs raw target tensors with no candidate-side color-space alignment.
- Metric labels use `(raw)` to emphasize that no color-space harmonization is applied before scoring.

Quantitative comparison checkpoint
---------------------------------

- Quantitative Dequant variant: `{quantitative_dequant_variant}`

Quantitative comparison: Full pipeline
--------------------------------------

- Candidate: 8-bit looked `sRGB`
- Target: untouched 32-bit `ACEScct`

{render_text_table(
    ["pipeline_row", "variant", "start_sigma", "bit_crunch_max", "PSNR (raw)", "SSIM (raw)", "LPIPS (raw)", "Total ms"],
    full_pipeline_rows,
)}

Quantitative comparison: Dequant
--------------------------------

- Candidate: 8-bit looked `sRGB`
- Target: 32-bit looked `sRGB`

{render_text_table(
    ["pipeline_row", "variant", "start_sigma", "bit_crunch_max", "PSNR (raw)", "SSIM (raw)", "LPIPS (raw)", "Total ms"],
    dequant_rows,
)}

Quantitative comparison: Mapper
-------------------------------

- Candidate: 32-bit looked `sRGB`
- Target: untouched 32-bit `ACEScct`

{render_text_table(
    ["pipeline_row", "variant", "start_sigma", "bit_crunch_max", "PSNR (raw)", "SSIM (raw)", "LPIPS (raw)", "Total ms"],
    mapper_rows,
)}

Dequant augmentation ablation
-----------------------------

{render_text_table(
    ["variant", "start_sigma", "bit_crunch_max", "PSNR (raw)", "SSIM (raw)", "Total ms"],
    ablation_table_rows,
)}
"""
    return content


def maybe_save_dashboard(
    *,
    output_dir: Path,
    variant_label: str,
    sample_id: str,
    input_srgb_chw: torch.Tensor,
    dequant_srgb_chw: torch.Tensor,
    mapper_srgb_chw: torch.Tensor,
    reference_srgb_chw: torch.Tensor,
    look_reference_srgb_chw: torch.Tensor,
) -> None:
    dashboard_path = output_dir / "dashboards" / variant_label / f"{sample_id}_dashboard.png"
    save_full_inference_dashboard(
        input_srgb_chw=input_srgb_chw,
        dequant_srgb_chw=dequant_srgb_chw,
        mapper_srgb_chw=mapper_srgb_chw,
        reference_srgb_chw=reference_srgb_chw,
        dequant_reference_srgb_chw=look_reference_srgb_chw,
        save_path=dashboard_path,
    )


def build_dataset(args: argparse.Namespace) -> Any:
    dataset = LuminaScaleWebDataset(
        shard_path=args.test_shard_path,
        batch_size=args.batch_size,
        shuffle_buffer=1,
        is_training=False,
        metadata_parquet=args.metadata_parquet,
        split="test",
        patches_per_image=1,
    )
    return dataset.get_loader(num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)


def resolve_srgb_to_ap0_matrix(device: torch.device) -> torch.Tensor:
    m_xyz_to_rec709 = ACESMatrices.M_XYZ_TO_REC709.to(device)
    m_ap1_to_xyz = ACESMatrices.M_AP1_TO_XYZ.to(device)
    m_ap1_to_ap0 = ACESMatrices.M_AP1_TO_AP0.to(device)
    m_rec709_to_xyz = torch.linalg.inv(m_xyz_to_rec709)
    m_xyz_to_ap1 = torch.linalg.inv(m_ap1_to_xyz)
    return m_ap1_to_ap0 @ m_xyz_to_ap1 @ m_rec709_to_xyz


def load_dequant_model_from_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
    channels: int,
) -> torch.nn.Module:
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


def main() -> None:
    args = parse_args()
    ensure_ocio_env()
    set_global_seed(args.seed)

    device = torch.device(args.device)
    output_dir = (project_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_log_path = output_dir / args.log_file

    print(f"Loading mapper model on {device}...")
    mapper_model = load_model_from_checkpoint(
        checkpoint_path=project_root / args.mapper_checkpoint if not Path(args.mapper_checkpoint).is_absolute() else Path(args.mapper_checkpoint),
        device=device,
        num_luts=3,
        lut_dim=33,
        num_lap=3,
        num_residual_blocks=5,
    )
    _forward_mapper = mapper_model.model if hasattr(mapper_model, "model") and isinstance(mapper_model.model, torch.nn.Module) else mapper_model

    print("Loading dequant checkpoints...")
    dequant_models: dict[str, torch.nn.Module] = {}
    dequant_label_to_path: dict[str, dict[str, Any]] = {}
    for checkpoint_str in args.dequant_checkpoints:
        checkpoint_path = project_root / checkpoint_str if not Path(checkpoint_str).is_absolute() else Path(checkpoint_str)
        label = checkpoint_label(checkpoint_path)
        dequant_label_to_path[label] = load_dequant_variant_config(checkpoint_path)
        dequant_models[label] = load_dequant_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            channels=args.dequant_channels,
        )

    quantitative_dequant_path = (
        project_root / args.quantitative_dequant_checkpoint
        if not Path(args.quantitative_dequant_checkpoint).is_absolute()
        else Path(args.quantitative_dequant_checkpoint)
    )
    quantitative_dequant_variant = checkpoint_label(quantitative_dequant_path)
    if quantitative_dequant_variant not in dequant_models:
        raise ValueError(
            f"quantitative dequant checkpoint {quantitative_dequant_path} must be included in --dequant-checkpoints"
        )

    lpips_metric = lpips.LPIPS(net="vgg").to(device)
    lpips_metric.eval()
    for parameter in lpips_metric.parameters():
        parameter.requires_grad = False

    cdl_processor = GPUCDLProcessor(device=device)
    display_transformer = ACESColorTransformer(device=device, use_lut=True)
    rec709_to_ap0_matrix = resolve_srgb_to_ap0_matrix(device)

    loader = build_dataset(args)

    collected_rows: list[dict[str, Any]] = []
    dashboard_saved_count: dict[str, int] = defaultdict(int)
    shared_rows_written: set[tuple[str, str]] = set()
    processed_samples = 0

    iterator = tqdm(loader, desc="Evaluating", unit="sample")
    for batch in iterator:
        if args.max_samples > 0 and processed_samples >= args.max_samples:
            break

        exr_bytes_list, metadata_list = batch
        if not exr_bytes_list:
            continue

        for exr_bytes, metadata in zip(exr_bytes_list, metadata_list):
            if args.max_samples > 0 and processed_samples >= args.max_samples:
                break

            target_acescct_hwc = decode_exr_bytes_to_hwc(exr_bytes)
            target_acescct_hwc = center_crop_hwc(target_acescct_hwc, 1024)  # Center-crop to match training dimensions
            sample_id = str(metadata.get("id", f"sample_{processed_samples:06d}"))
            split = str(metadata.get("split", "test"))
            source = str(metadata.get("source", ""))

            input_srgb_chw, look_reference_srgb_chw, clean_reference_srgb_chw, look_seed, _ = build_display_reference_and_input(
                target_acescct_hwc=target_acescct_hwc,
                sample_id=sample_id,
                base_seed=args.seed,
                cdl_processor=cdl_processor,
                transformer=display_transformer,
                device=device,
            )
            target_acescct_chw = target_acescct_hwc.permute(2, 0, 1).detach().cpu()

            baseline_start = time.perf_counter()
            baseline_aces2065_chw = srgb_to_aces2065_1(input_srgb_chw.to(device), rec709_to_ap0_matrix).detach().cpu()
            baseline_acescct_chw = ap0_to_acescct(baseline_aces2065_chw, device=device).detach().cpu()
            baseline_total_ms = (time.perf_counter() - baseline_start) * 1000.0

            # mapper_baseline: look-applied sRGB → ACEScct (no ML mapper)
            mapper_baseline_aces2065_chw = srgb_to_aces2065_1(look_reference_srgb_chw.to(device), rec709_to_ap0_matrix).detach().cpu()
            mapper_baseline_acescct_chw = ap0_to_acescct(mapper_baseline_aces2065_chw, device=device).detach().cpu()

            mapper_only_acescct_chw, mapper_only_display_srgb_chw, mapper_only_time_ms = run_mapper_model(
                model=mapper_model,
                input_srgb_chw=input_srgb_chw,
                device=device,
                align_multiple=args.align_multiple,
                max_side=args.max_side,
                display_transformer=display_transformer,
            )

            shared_rows = [
                make_metric_row(
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    experiment="quantitative_comparison",
                    variant="shared",
                    pipeline_row="dequant_baseline",
                    look_seed=look_seed,
                        psnr_acescct=compute_psnr(input_srgb_chw, look_reference_srgb_chw),
                        ssim_acescct=compute_ssim(input_srgb_chw, look_reference_srgb_chw, data_range=1.0),
                        lpips_srgb=compute_lpips_metric(lpips_metric, input_srgb_chw, look_reference_srgb_chw, device),
                    total_time_ms=baseline_total_ms,
                ),
                make_metric_row(
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    experiment="quantitative_comparison",
                    variant="shared",
                    pipeline_row="full_baseline",
                    look_seed=look_seed,
                    psnr_acescct=compute_psnr(input_srgb_chw, target_acescct_chw),
                    ssim_acescct=compute_ssim(input_srgb_chw, target_acescct_chw, data_range=1.0),
                    lpips_srgb=compute_lpips_metric(lpips_metric, input_srgb_chw, target_acescct_chw, device),
                    total_time_ms=baseline_total_ms,
                ),
                make_metric_row(
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    experiment="quantitative_comparison",
                    variant="shared",
                    pipeline_row="mapper_baseline",
                    look_seed=look_seed,
                    psnr_acescct=compute_psnr(look_reference_srgb_chw, target_acescct_chw),
                    ssim_acescct=compute_ssim(look_reference_srgb_chw, target_acescct_chw, data_range=1.0),
                    lpips_srgb=compute_lpips_metric(lpips_metric, look_reference_srgb_chw, target_acescct_chw, device),
                    total_time_ms=baseline_total_ms,
                ),
                make_metric_row(
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    experiment="quantitative_comparison",
                    variant="shared",
                    pipeline_row="mapper_only",
                    look_seed=look_seed,
                    psnr_acescct=compute_psnr(mapper_only_acescct_chw, target_acescct_chw),
                    ssim_acescct=compute_ssim(mapper_only_acescct_chw, target_acescct_chw, data_range=1.0),
                    lpips_srgb=compute_lpips_metric(lpips_metric, mapper_only_acescct_chw, target_acescct_chw, device),
                    mapper_time_ms=mapper_only_time_ms,
                    total_time_ms=mapper_only_time_ms,
                ),
            ]

            shared_key = ("shared", sample_id)
            if shared_key not in shared_rows_written:
                for row in shared_rows:
                    collected_rows.append(row)
                shared_rows_written.add(shared_key)

            quantitative_start = time.perf_counter()
            quantitative_dequant_model = dequant_models[quantitative_dequant_variant]
            dequant_srgb_chw, dequant_time_ms = run_dequant_model(
                model=quantitative_dequant_model,
                input_chw=input_srgb_chw,
                device=device,
                align_multiple=args.align_multiple,
            )

            dequant_only_aces2065_chw = srgb_to_aces2065_1(dequant_srgb_chw.to(device), rec709_to_ap0_matrix).detach().cpu()
            dequant_only_acescct_chw = ap0_to_acescct(dequant_only_aces2065_chw, device=device).detach().cpu()
            dequant_only_display_srgb_chw = display_transformer.aces_to_srgb_32f(
                dequant_only_aces2065_chw.permute(1, 2, 0).unsqueeze(0).to(device),
                input_cs="ACES2065-1",
            ).squeeze(0).permute(2, 0, 1).detach().cpu()

            full_pipeline_acescct_chw, full_pipeline_display_srgb_chw, mapper_time_ms = run_mapper_model(
                model=mapper_model,
                input_srgb_chw=dequant_srgb_chw,
                device=device,
                align_multiple=args.align_multiple,
                max_side=args.max_side,
                display_transformer=display_transformer,
            )

            total_time_ms = (time.perf_counter() - quantitative_start) * 1000.0

            quantitative_rows = [
                make_metric_row(
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    experiment="quantitative_comparison",
                    variant=quantitative_dequant_variant,
                    pipeline_row="dequant_only",
                    look_seed=look_seed,
                        psnr_acescct=compute_psnr(dequant_srgb_chw, look_reference_srgb_chw),
                        ssim_acescct=compute_ssim(dequant_srgb_chw, look_reference_srgb_chw, data_range=1.0),
                        lpips_srgb=compute_lpips_metric(lpips_metric, dequant_srgb_chw, look_reference_srgb_chw, device),
                    dequant_time_ms=dequant_time_ms,
                    total_time_ms=dequant_time_ms,
                ),
                make_metric_row(
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    experiment="quantitative_comparison",
                    variant=quantitative_dequant_variant,
                    pipeline_row="full_pipeline",
                    look_seed=look_seed,
                    psnr_acescct=compute_psnr(full_pipeline_acescct_chw, target_acescct_chw),
                    ssim_acescct=compute_ssim(full_pipeline_acescct_chw, target_acescct_chw, data_range=1.0),
                    lpips_srgb=compute_lpips_metric(lpips_metric, full_pipeline_acescct_chw, target_acescct_chw, device),
                    dequant_time_ms=dequant_time_ms,
                    mapper_time_ms=mapper_time_ms,
                    total_time_ms=total_time_ms,
                ),
            ]

            ablation_row = make_metric_row(
                sample_id=sample_id,
                split=split,
                source=source,
                experiment="dequant_ablation",
                variant="shared",
                pipeline_row="dequant_baseline",
                look_seed=look_seed,
                psnr_srgb=compute_psnr(input_srgb_chw, look_reference_srgb_chw),
                ssim_srgb=compute_ssim(input_srgb_chw, look_reference_srgb_chw, data_range=1.0),
                total_time_ms=baseline_total_ms,
            )
            collected_rows.append(ablation_row)

            ablation_row = make_metric_row(
                sample_id=sample_id,
                split=split,
                source=source,
                experiment="dequant_ablation",
                variant=quantitative_dequant_variant,
                pipeline_row="dequant_output_srgb",
                look_seed=look_seed,
                psnr_srgb=compute_psnr(dequant_srgb_chw, look_reference_srgb_chw),
                ssim_srgb=compute_ssim(dequant_srgb_chw, look_reference_srgb_chw, data_range=1.0),
                dequant_time_ms=dequant_time_ms,
                total_time_ms=dequant_time_ms,
            )

            for row in quantitative_rows + [ablation_row]:
                collected_rows.append(row)

            if args.dashboard_samples > 0 and dashboard_saved_count[quantitative_dequant_variant] < args.dashboard_samples:
                maybe_save_dashboard(
                    output_dir=output_dir,
                    variant_label=quantitative_dequant_variant,
                    sample_id=sample_id,
                    input_srgb_chw=input_srgb_chw,
                    dequant_srgb_chw=dequant_srgb_chw,
                    mapper_srgb_chw=full_pipeline_display_srgb_chw,
                    reference_srgb_chw=clean_reference_srgb_chw,
                    look_reference_srgb_chw=look_reference_srgb_chw,
                )
                dashboard_saved_count[quantitative_dequant_variant] += 1

            for ablation_variant_label, ablation_model in dequant_models.items():
                if ablation_variant_label == quantitative_dequant_variant:
                    continue
                ablation_srgb_chw, ablation_time_ms = run_dequant_model(
                    model=ablation_model,
                    input_chw=input_srgb_chw,
                    device=device,
                    align_multiple=args.align_multiple,
                )
                ablation_row = make_metric_row(
                    sample_id=sample_id,
                    split=split,
                    source=source,
                    experiment="dequant_ablation",
                    variant=ablation_variant_label,
                    pipeline_row="dequant_output_srgb",
                    look_seed=look_seed,
                    psnr_srgb=compute_psnr(ablation_srgb_chw, look_reference_srgb_chw),
                    ssim_srgb=compute_ssim(ablation_srgb_chw, look_reference_srgb_chw, data_range=1.0),
                    dequant_time_ms=ablation_time_ms,
                    total_time_ms=ablation_time_ms,
                )
                collected_rows.append(ablation_row)

            processed_samples += 1
            iterator.set_postfix(samples=processed_samples)

    aggregate_rows_data = aggregate_rows(collected_rows)
    summary_text = build_summary_text(
        args=args,
        processed_samples=processed_samples,
        aggregate_rows_data=aggregate_rows_data,
        dequant_labels=dequant_label_to_path,
        quantitative_dequant_variant=quantitative_dequant_variant,
    )

    print("\n" + summary_text)
    summary_log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_log_path.write_text(summary_text + "\n", encoding="utf-8")

    print("==========================================================================")
    print(f"Processed samples: {processed_samples}")
    print(f"Summary log: {summary_log_path}")
    if args.dashboard_samples > 0:
        print(f"Dashboard directory: {output_dir / 'dashboards'}")


if __name__ == "__main__":
    main()
