# LuminaScale

Machine learning model for bit-depth expansion and ACES color space normalization.

## Quick Start

- **New to LuminaScale?** → See [WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md) for a step-by-step guide on Windows
- **Running locally (Linux)?** → See [Local Setup (Pixi)](#local-setup-pixi)
- **Running on HPC?** → Continue to [AI-Cloud Setup](#hpc-setup-ai-cloud) below

---

## Local Setup (Pixi)

For local development and training on your workstation.

### Train ACES mapper (Pixi)

From the project root:

```bash
pixi run python scripts/train_aces_mapper.py --config-name=mapper_dev
```

For a full run instead of dev:

```bash
pixi run python scripts/train_aces_mapper.py --config-name=mapper
```

### Adding or changing dependencies

Edit `pixi.toml` (conda packages under `[dependencies]`, pip-only ones under
`[pypi-dependencies]`), then:

```bash
pixi add <package>     # or edit pixi.toml by hand
pixi lock              # re-resolve and update pixi.lock
```

Commit both `pixi.toml` and `pixi.lock` together — the lockfile pins the exact
environment used locally **and** inside the HPC container, so they never drift.

To run multiple mapper experiments in parallel (e.g., mapper_2 and mapper_3):
```bash
pixi run python scripts/train_aces_mapper.py --config-name=mapper --multirun mapper_experiments=mapper_1,mapper_2,mapper_3,mapper_4
```

### Start TensorBoard 

```bash
pixi run tensorboard --logdir outputs/training/mapper
```

Examples:

- Mapper logs only (flag): `pixi run tensorboard -- --logdir outputs/training/mapper`
- Custom port (flag): `pixi run tensorboard -- --port 6010`
- Environment variable fallback also works: `TB_LOGDIR=outputs/training/mapper TB_PORT=6010 pixi run tensorboard`

Then open:

- `http://localhost:6008`

---

## HPC Setup (AI-Cloud)

For advanced users running large-scale training on the AI-Cloud HPC cluster:

### login to ai-cloud hpc

```bash
ssh aicloud
```

### How the container works

`luminascale.sif` (built from `singularity/luminascale.def`, see
`singularity/build_singularity.sh`) bakes pixi + `pixi.toml`/`pixi.lock` but
**not** the Python environment. On first use it installs the environment from
the lockfile into `~/.lumina-env` (override with `LUMINA_ENV_HOME`). Until that
is done, `python` inside the container is the base-image PyTorch only — good
enough for a smoke test, but training needs the environment below.

> **CUDA version policy:** the stack is pinned to **CUDA 12.x** on purpose.
> The cluster's V100 nodes (sm_70 / Volta) have local scratch storage we may
> want to use, and CUDA 13 dropped Volta support — a cu13 image would fail
> there with "no kernel image". If the V100s are ever abandoned, upgrade the
> base image and the pixi `cuda-version` pin together (see the note at the
> top of `singularity/luminascale.def`).

### One-time: install the container environment

Run this once per cluster account, on a GPU node (`--nv` is required so pixi
can detect the CUDA driver):

```bash
srun --gpus=1 --time=0:30:00 --mem=8G \
  singularity exec --nv luminascale.sif lumina-env install
```

Notes:

- Apptainer/Singularity sanitize the host environment, so a plain
  `export LUMINA_ENV_HOME=...` is **not** forwarded into the container. Pass it
  as `--env LUMINA_ENV_HOME=...` or `SINGULARITYENV_LUMINA_ENV_HOME=...`.
- Home storage is quota-limited (~11 GB for the environment). Point
  `LUMINA_ENV_HOME` at a scratch/project path if needed.
- Install once up front; don't let parallel/array jobs race on first install.
- Update later with `singularity exec luminascale.sif lumina-env update`.

### Tensorboard

```bash
srun --mem=16G singularity exec luminascale.sif tensorboard --logdir=outputs/training --port=6006 --bind_all
```
### Training with config file
```bash
sbatch scripts/train_dequant_net.sh config-name=train_01
```
### Training with params
```bash
sbatch scripts/train_dequant_net.sh loss.l1_weight=1.0 loss.l2_weight=0.0 loss.charbonnier_weight=2.0 loss.grad_match_weight=0.0
```

### Training with srun
```bash
srun --cpus-per-task=16 --mem=64G --gres=gpu:l40s:1 --time=1:00:00 singularity exec --nv luminascale.sif python scripts/train_dequant_net.py --config-name=dev
```

### Inference

Run inference on a 2K synthetic sky gradient using `srun`:

```bash
srun --gres=gpu:1 --mem=16G singularity exec --nv luminascale.sif python scripts/run_dequant_inference.py --checkpoint dataset/temp/test_run/20260331_164330_dequant_net_epoch_1.pt --synthetic --width 2048 --height 1024 --output outputs/inference/sky_2k.exr
```