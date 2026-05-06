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
srun --cpus-per-task=16 --mem=64G --gres=gpu:l40s:1 --time=1:00:00 singularity exec --nv luminascale.sif python3 scripts/train_dequant_net.py --config-name=dev
```

### Inference

Run inference on a 2K synthetic sky gradient using `srun`:

```bash
srun --gres=gpu:1 --mem=16G singularity exec --nv luminascale.sif python3 scripts/run_dequant_inference.py --checkpoint dataset/temp/test_run/20260331_164330_dequant_net_epoch_1.pt --synthetic --width 2048 --height 1024 --output outputs/inference/sky_2k.exr
```