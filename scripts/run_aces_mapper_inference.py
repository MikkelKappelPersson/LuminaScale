#!/usr/bin/env python3
"""LuminaScale ACES Mapper inference and visualization.

Modes:
1. ACES reference mode (EXR input): full dashboard with reference diagnostics.
2. Display image mode (PNG/JPG/etc): compact dashboard with input/output only.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from luminascale.utils.aces_mapper_inference import (
    build_look,
    close_figure,
    load_model_from_checkpoint,
    run_aces_mapper_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACES mapper inference and save comparison dashboard.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="outputs/training/mapper/20260425_231537/checkpoints/aces-mapper-20260425_231537-epoch=09.ckpt",
        help="Path to model checkpoint (.ckpt/.pt)",
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
        "--plot-output",
        type=str,
        default="",
        help="Path to save dashboard plot when --save-plot is set (default: outputs/inference/<input_stem>_plot.png)",
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

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    ocio_config = project_root / "config" / "aces" / "studio-config.ocio"
    if ocio_config.exists():
        os.environ["OCIO"] = str(ocio_config)

    input_path = Path(args.input_path)
    assert input_path.exists(), f"Input does not exist: {input_path}"
    is_exr_reference = bool(args.input_is_aces)
    if is_exr_reference:
        assert input_path.suffix.lower() == ".exr", "--input-is-aces expects an EXR input"

    default_output_dir = project_root / "outputs" / "inference"
    input_stem = input_path.stem
    resolved_pred_aces_output = (
        Path(args.output) if args.output else default_output_dir / f"{input_stem}_out.exr"
    )
    resolved_plot_output = (
        Path(args.plot_output) if args.plot_output else default_output_dir / f"{input_stem}_plot.png"
    )

    print(f"Loading model on {device}...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        num_luts=args.num_luts,
        lut_dim=args.lut_dim,
        num_lap=args.num_lap,
        num_residual_blocks=args.num_residual_blocks,
    )

    look = None
    if is_exr_reference:
        print(f"Creating CDL look for inference: {args.look_mode}")
        look = build_look(args.look_mode, args.slope, args.offset, args.power, args.saturation)
    else:
        print("--input-is-aces not set; skipping CDL look and running compact input/output dashboard")

    print(f"Running model inference for {input_path}...")
    if args.keep_padding:
        print("Keeping aligned padded output dimensions (--keep-padding set)")
    else:
        print("Removing alignment padding from generated outputs (default behavior)")

    output_path = resolved_plot_output if args.save_plot else None
    pred_aces_output = None if args.no_save_output else resolved_pred_aces_output
    figure = run_aces_mapper_inference(
        model=model,
        input=input_path,
        output_path=output_path,
        look=look,
        crop_size=args.crop_size,
        align_multiple=args.align_multiple,
        max_side=args.max_side,
        pred_aces_output=pred_aces_output,
        input_is_aces=is_exr_reference,
        keep_aligned_output=args.keep_padding,
        device=device,
    )

    close_figure(figure)

    if look is not None:
        print(f"CDL look used: {look}")
    else:
        print("CDL look skipped: only used for ACES validation with --input-is-aces")

    print("===========================================================================")

    if args.no_save_output:
        print("Predicted ACES EXR not saved (--no-save-output set)")
    else:
        print(f"Saved predicted ACES EXR: {resolved_pred_aces_output}")

    if args.save_plot:
        print(f"Saved comparison dashboard: {resolved_plot_output}")
    else:
        print("Comparison dashboard not saved (--save-plot not set)")



if __name__ == "__main__":
    main()
