# LuminaScale

Machine learning model for bit-depth expansion and ACES color space normalization.

## Inference

Run inference on a 2K synthetic sky gradient using `srun`:

```bash
srun --partition=normal --gres=gpu:1 --mem=16G singularity exec --nv luminascale.sif python3 scripts/run_inference.py --checkpoint dataset/temp/test_run/20260331_164330_dequant_net_epoch_1.pt --synthetic --width 2048 --height 1024 --output outputs/inference/sky_2k.exr
```