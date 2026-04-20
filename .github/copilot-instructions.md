# LuminaScale: Neural Bit-Depth Expansion & ACES Color Space Normalization

## Project Vision

LuminaScale implements neural models for:

-   **Bit-Depth Expansion (BDE)/Dequantization**: Upsampling low-bit-depth imagery (e.g., 8-bit, 10-bit) to higher fidelity using learned super-resolution.
-   **ACES transformer**: Color space transformation and “blind” normalization to SMPTE Academy Color Encoding System (ACES) without ground-truth targets.

## Architecture

-   Python 3.12+
-   Pytorch Ligtning
-   Cuda optimized

## Platform

The platform is twofold. one local for development and one HPC for large scale training

### 1\. Local

-   Managed with pixi (use: pixi run)  for running commands
-   Local gpu: rtx 3080

### 2\. AI-Cloud HPC

-   Multi gpu
-   Uses slurm (use srun or sbatch)
-   Gpus available: t4, a40, a10, a40, l40s

## Code Style & Language

-   **Type hints** on all function signatures; use `typing` module and `from __future__ import annotations` for forward refs.
-   **Docstrings**: code should be self-documenting through clear naming and type hints.
-   **Formatting**: Black (line length 100), isort, flake8 (ignore E501 if line length exceeds).
-   **Linting**: ruff with default config, ignore: E501 (line length handled by Black).
-   **Fail Fast**: Avoid exceptions and instead fail fast or use asserts if needed.

## Domain Knowledge (Reference)

-   **Bit-depth**: Standard depths are 8, 10, 12, 16-bit; 8-bit is common in web/consumer, 10+ in cinema/HDR.
-   **ACES**: Academy Color Encoding System; working space for color-accurate pipelines; requires RGB→ACES XYZ→ACES AP1.
-   **“Blind” ACES norm**: Inferring ACES color space without paired reference images; rely on statistical or perceptual losses.
-   **Metrics**: PSNR (luminance), SSIM (structure), ΔE (perceptual color distance); use BT.709 or ACES-linear spaces as needed.

---

**Version**: 2.0 | **Last updated**: 2026-03-26