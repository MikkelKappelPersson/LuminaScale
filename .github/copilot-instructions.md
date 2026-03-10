# LuminaScale: Neural Bit-Depth Expansion & ACES Color Space Normalization

## Project Vision

LuminaScale implements neural models for:

-   **Bit-Depth Expansion (BDE)**: Upsampling low-bit-depth imagery (e.g., 8-bit, 10-bit) to higher fidelity using learned super-resolution.
-   **ACES Normalization**: Color space transformation and “blind” normalization to SMPTE Academy Color Encoding System (ACES) without ground-truth targets.

## Code Style & Language

-   **Python 3.12+** (PyTorch ecosystem)
-   **Type hints** on all function signatures; use `typing` module and `from __future__ import annotations` for forward refs.
-   **Docstrings**: Google style (see examples in `src/models/`).
-   **Formatting**: Black (line length 100), isort, flake8 (ignore E501 if docstring exceeds).
-   **Linting**: ruff with default config, ignore: E501 (line length handled by Black).

## Architecture

### Directory Structure

```
LuminaScale/
├── src/
│   └── luminascale/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── bde.py           # Bit-depth expansion models
│       │   └── aces_norm.py      # ACES normalization modules
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py        # PyTorch DataLoaders
│       │   └── transforms.py     # Augmentation and preprocessing
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py        # Training loop, Hydra config loading
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── metrics.py        # PSNR, SSIM, ΔE for color spaces
│       │   └── inference.py      # Batch inference and export
│       └── utils/
│           ├── __init__.py
│           ├── io.py             # File loading/saving
│           └── color_ops.py      # Color space transforms (RGB, ACES, etc.)
├── configs/
│   ├── model/
│   │   ├── bde_unet.yaml
│   │   └── aces_mlp.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── hpc_slurm.yaml        # HPC-tuned settings
│   └── config.yaml               # Root Hydra config
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_metrics.py
├── notebooks/
│   └── exploration.ipynb         # EDA and experiments
├── scripts/
│   ├── train.py                  # Main entry point
│   ├── eval.py                   # Evaluation script
│   └── export.py                 # Model export for inference
├── singularity/
│   └── luminascale.def           # Container for HPC (references skill: singularity-container-ops)
├── pyproject.toml
├── README.md
├── LICENSE
└── .github/
    ├── copilot-instructions.md   # This file
    └── skills/
        └── singularity/
            └── SKILL.md          # Singularity HPC container operations
```

## Domain Knowledge (Reference)

-   **Bit-depth**: Standard depths are 8, 10, 12, 16-bit; 8-bit is common in web/consumer, 10+ in cinema/HDR.
-   **ACES**: Academy Color Encoding System; working space for color-accurate pipelines; requires RGB→ACES XYZ→ACES AP1.
-   **“Blind” ACES norm**: Inferring ACES color space without paired reference images; rely on statistical or perceptual losses.
-   **Metrics**: PSNR (luminance), SSIM (structure), ΔE (perceptual color distance); use BT.709 or ACES-linear spaces as needed.

---

**Version**: 1.0 | **Last updated**: 2026-03-10