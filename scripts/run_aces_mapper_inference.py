#!/usr/bin/env python3
"""LuminaScale ACES Mapper inference and visualization.

Pipeline:
1. Load ACES2065-1 EXR reference.
2. Apply CDL look in ACES space (input degradation/grade).
3. Convert graded ACES to display-referred sRGB (model input).
4. Run ACES mapper checkpoint (sRGB -> ACES prediction).
5. Save a single comparison image with side-by-side diagnostics.
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
        default="outputs/training/mapper/20260425_170412/checkpoints/aces-mapper-20260425_170412-epoch=02.ckpt",
        help="Path to model checkpoint (.ckpt/.pt)",
    )
    parser.add_argument(
        "--aces-input", 
        type=str, 
        default="dataset/full/aces/MIT-Adobe_5K_a0001-jmac_DSC1459.exr",
        help="Path to ACES2065-1 EXR reference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference/aces_mapper_comparison.png",
        help="Path to save comparison dashboard PNG",
    )
    parser.add_argument(
        "--pred-aces-output",
        type=str,
        default="",
        help="Optional path to save predicted ACES EXR",
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
        "--contrast-strength",
        type=float,
        default=20.0,
        help="S-curve contrast strength for diagnostic views",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    input_path = Path(args.aces_input)
    assert input_path.exists(), f"Input does not exist: {input_path}"
    assert input_path.suffix.lower() == ".exr", "--aces-input must be an EXR file"

    print(f"Loading model on {device}...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        num_luts=args.num_luts,
        lut_dim=args.lut_dim,
        num_lap=args.num_lap,
        num_residual_blocks=args.num_residual_blocks,
    )

    print(f"Creating CDL look for inference: {args.look_mode}")
    look = build_look(args.look_mode, args.slope, args.offset, args.power, args.saturation)

    print(f"Running model inference for {input_path}...")
    output_path = Path(args.output)
    figure = run_aces_mapper_inference(
        model=model,
        aces_input=input_path,
        output_path=output_path,
        look=look,
        crop_size=args.crop_size,
        align_multiple=args.align_multiple,
        max_side=args.max_side,
        pred_aces_output=args.pred_aces_output or None,
        contrast_strength=args.contrast_strength,
        device=device,
    )

    close_figure(figure)

    if args.pred_aces_output:
        print(f"Saved predicted ACES EXR: {Path(args.pred_aces_output)}")

    print(f"CDL look used: {look}")
    print("==================================")
    print(f"Saved comparison dashboard: {output_path}")



if __name__ == "__main__":
    main()
