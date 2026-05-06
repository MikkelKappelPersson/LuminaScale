#!/usr/bin/env python3
"""Bake OCIO display LUT artifacts for LuminaScale runtime.

Produces deterministic file-based LUT artifacts under:
    assets/luts/{config_hash8}__{display_slug}__{view_slug}__{cube_size}/

Outputs:
- manifest.json
- domains.json
- aces2065_to_srgb_display.pt
- acescct_to_srgb_display.pt
- aces2065_to_srgb_display.ctf
- acescct_to_srgb_display.ctf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_processor_to_lut(
    processor,
    cube_size: int,
    domain_min: float,
    domain_max: float,
) -> torch.Tensor:
    """Sample an OCIO CPU processor into a 3D LUT tensor [N, N, N, 3]."""
    cpu_processor = processor.getDefaultCPUProcessor()
    axis = np.linspace(domain_min, domain_max, cube_size, dtype=np.float32)

    # Build full 3D lattice in one tensor: [N, N, N, 4]
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    rgba = np.empty((cube_size, cube_size, cube_size, 4), dtype=np.float32)
    rgba[..., 0] = rr
    rgba[..., 1] = gg
    rgba[..., 2] = bb
    rgba[..., 3] = 1.0

    # OCIO CPU processor accepts an HxWx4 RGBA image; flatten 3D lattice to image.
    flat_rgba = rgba.reshape(cube_size * cube_size, cube_size, 4)
    cpu_processor.applyRGBA(flat_rgba)

    lut = flat_rgba.reshape(cube_size, cube_size, cube_size, 4)[..., :3]
    return torch.from_numpy(lut.copy()).float()


def _bake_with_format(
    ocio,
    config,
    *,
    output_path: Path,
    format_name: str,
    input_space: str,
    display: str,
    view: str,
    cube_size: int,
    shaper_space: str,
) -> None:
    """Bake a LUT artifact using OCIO Baker for a specific format."""
    baker = ocio.Baker()
    baker.setConfig(config)
    baker.setFormat(format_name)
    baker.setInputSpace(input_space)
    baker.setDisplayView(display, view)
    baker.setCubeSize(cube_size)
    if shaper_space:
        baker.setShaperSpace(shaper_space)
    baker.bake(str(output_path))


def _bake_source_artifact(
    ocio,
    config,
    *,
    out_dir: Path,
    base_name: str,
    input_space: str,
    display: str,
    view: str,
    cube_size: int,
    shaper_space: str,
) -> str:
    """Bake a source-of-truth LUT artifact, trying preferred formats in order.

    Returns the produced filename (relative to out_dir).
    """
    candidates = [
        ("ctf", f"{base_name}.ctf"),
        ("clf", f"{base_name}.clf"),
        ("resolve_cube", f"{base_name}.cube"),
        ("iridas_cube", f"{base_name}.iridas.cube"),
    ]

    last_error: Exception | None = None
    for format_name, filename in candidates:
        try:
            _bake_with_format(
                ocio,
                config,
                output_path=out_dir / filename,
                format_name=format_name,
                input_space=input_space,
                display=display,
                view=view,
                cube_size=cube_size,
                shaper_space=shaper_space,
            )
            return filename
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(
        f"Failed to bake source artifact for {input_space} using all supported fallback formats"
    ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake OCIO display LUT artifacts for runtime")
    parser.add_argument(
        "--config",
        type=str,
        default="config/aces/studio-config.ocio",
        help="Path to OCIO config file",
    )
    parser.add_argument(
        "--display",
        type=str,
        default="sRGB - Display",
        help="OCIO display name",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="ACES 2.0 - SDR 100 nits (Rec.709)",
        help="OCIO view name",
    )
    parser.add_argument(
        "--cube-size",
        type=int,
        default=257,
        help="3D LUT cube size for runtime tensors",
    )
    parser.add_argument(
        "--aces2065-domain-min",
        type=float,
        default=-0.5,
        help="Domain min for ACES2065-1 LUT sampling",
    )
    parser.add_argument(
        "--aces2065-domain-max",
        type=float,
        default=10.0,
        help="Domain max for ACES2065-1 LUT sampling",
    )
    parser.add_argument(
        "--acescct-domain-min",
        type=float,
        default=-1.0,
        help="Domain min for ACEScct LUT sampling",
    )
    parser.add_argument(
        "--acescct-domain-max",
        type=float,
        default=1.0,
        help="Domain max for ACEScct LUT sampling",
    )
    parser.add_argument(
        "--shaper-space-aces2065",
        type=str,
        default="",
        help="Optional OCIO shaper space for ACES2065-1 CTF baking",
    )
    parser.add_argument(
        "--shaper-space-acescct",
        type=str,
        default="",
        help="Optional OCIO shaper space for ACEScct CTF baking",
    )
    args = parser.parse_args()

    try:
        import PyOpenColorIO as ocio
    except ImportError as exc:
        raise ImportError("PyOpenColorIO is required. Install with: pixi add opencolorio") from exc

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"OCIO config not found: {config_path}")

    config = ocio.Config.CreateFromFile(str(config_path))
    config_hash = _sha256_file(config_path)
    profile_id = (
        f"{config_hash[:8]}__{_slugify(args.display)}__{_slugify(args.view)}__{int(args.cube_size)}"
    )
    out_dir = repo_root / "assets" / "luts" / profile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build processors for runtime tensor sampling.
    p_aces2065 = config.getProcessor(
        "ACES2065-1",
        args.display,
        args.view,
        ocio.TRANSFORM_DIR_FORWARD,
    )
    p_acescct = config.getProcessor(
        "ACEScct",
        args.display,
        args.view,
        ocio.TRANSFORM_DIR_FORWARD,
    )

    print(f"Sampling ACES2065-1 LUT ({args.cube_size}^3)...", flush=True)
    lut_aces2065 = _sample_processor_to_lut(
        p_aces2065,
        cube_size=args.cube_size,
        domain_min=args.aces2065_domain_min,
        domain_max=args.aces2065_domain_max,
    )
    torch.save({"lut_3d": lut_aces2065}, out_dir / "aces2065_to_srgb_display.pt")

    print(f"Sampling ACEScct LUT ({args.cube_size}^3)...", flush=True)
    lut_acescct = _sample_processor_to_lut(
        p_acescct,
        cube_size=args.cube_size,
        domain_min=args.acescct_domain_min,
        domain_max=args.acescct_domain_max,
    )
    torch.save({"lut_3d": lut_acescct}, out_dir / "acescct_to_srgb_display.pt")

    # Bake source-of-truth artifacts with resilient format fallback.
    aces2065_source_file = _bake_source_artifact(
        ocio,
        config,
        out_dir=out_dir,
        base_name="aces2065_to_srgb_display",
        input_space="ACES2065-1",
        display=args.display,
        view=args.view,
        cube_size=args.cube_size,
        shaper_space=args.shaper_space_aces2065,
    )
    acescct_source_file = _bake_source_artifact(
        ocio,
        config,
        out_dir=out_dir,
        base_name="acescct_to_srgb_display",
        input_space="ACEScct",
        display=args.display,
        view=args.view,
        cube_size=args.cube_size,
        shaper_space=args.shaper_space_acescct,
    )

    domains = {
        "aces2065": {"min": float(args.aces2065_domain_min), "max": float(args.aces2065_domain_max)},
        "acescct": {"min": float(args.acescct_domain_min), "max": float(args.acescct_domain_max)},
    }
    (out_dir / "domains.json").write_text(json.dumps(domains, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "ocio_config_path": str(config_path),
        "ocio_config_sha256": config_hash,
        "display": args.display,
        "view": args.view,
        "quality_cube_size": int(args.cube_size),
        "source_artifacts": {
            "aces2065_to_srgb_display": aces2065_source_file,
            "acescct_to_srgb_display": acescct_source_file,
        },
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "tool_version": "bake_display_luts.py@1",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. Wrote LUT profile: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
