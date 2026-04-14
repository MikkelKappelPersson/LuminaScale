# LuminaScale

Machine learning model for bit-depth expansion and ACES color space normalization.

## login to ai-cloud hpc

```bash
ssh aicloud
```

## Tensorboard

```bash
srun --mem=16G singularity exec luminascale.sif tensorboard --logdir=outputs/training --port=6006 --bind_all
```

## Training with params
```bash
sbatch scripts/train_dequantization_net.sh loss.l1_weight=1.0 loss.l2_weight=0.0 loss.charbonnier_weight=2.0 loss.grad_match_weight=0.0
```

## Inference

Run inference on a 2K synthetic sky gradient using `srun`:

```bash
srun --gres=gpu:1 --mem=16G singularity exec --nv luminascale.sif python3 scripts/run_inference.py --checkpoint dataset/temp/test_run/20260331_164330_dequant_net_epoch_1.pt --synthetic --width 2048 --height 1024 --output outputs/inference/sky_2k.exr
```