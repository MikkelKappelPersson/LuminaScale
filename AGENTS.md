# LuminaScale: Neural Bit-Depth Expansion & ACES Color Space Normalization

## Project Vision

LuminaScale implements neural models for:

-   **Dequantization**: Upsampling low-bit-depth imagery (e.g., 8-bit, 10-bit) to higher fidelity using learned super-resolution.
-   **ACES2065-1 Color space mapping**: Color space mapping and “blind” normalization to SMPTE Academy Color Encoding System (ACES).

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
-   **ACES**: Academy Color Encoding System; ACES2065-1 is the linear RGB color space used for interchange; ACEScg is a working space for CGI.


---

**Version**: 3.0 | **Last updated**: 2026-03-26