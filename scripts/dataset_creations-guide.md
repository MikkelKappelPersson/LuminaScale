# Dataset Generation Guide (WebDataset)

To create a training dataset, generate ACES EXR files and convert to WebDataset format:

```bash
pixi run python scripts/quality_filtered_aces_conversion.py
pixi run python scripts/generate_wds_shards.py
```

## Step 1: Generate ACES EXR Files

Create high-quality ACES HDR imagery from RAW camera files:

```bash
pixi run python scripts/quality_filtered_aces_conversion.py
```

**Optional:** Limit to a small dataset for testing:
```bash
pixi run python scripts/quality_filtered_aces_conversion.py --max-images=10
```

This generates ACES EXR files to `dataset/temp/aces/`

## Step 2: Generate WebDataset Shards

Convert ACES EXR files into WebDataset shards for efficient training:

```bash
pixi run python scripts/generate_wds_shards.py
```

This creates `.tar` shards in `dataset/wds_shards/` that are:
- **Scalable**: On-the-fly patch generation with random crops per image
- **Fast**: GPU-accelerated EXR decoding and CDL grading during training

## Training with WebDataset

The training pipeline automatically loads WebDataset shards and applies:
1. On-the-fly EXR decoding (OIIO)
2. Random CDL color grading
3. ACES→sRGB transformation
4. Patch-based training (32 crops per image)

Run training with:
```bash
pixi run python scripts/train_dequantization_net.py
```

Configure shard path in `configs/default.yaml`:
```yaml
dataset:
  shard_path: dataset/wds_shards/  # WebDataset .tar shards
  patches_per_image: 32             # Patches per image
```
