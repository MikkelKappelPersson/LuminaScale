#!/usr/bin/env bash
# Local GPU Testing: Convert ACES EXRs → WebDataset Shards using Pixi
# No SBATCH/HPC required, runs directly on local GPU

set -e

# Create output directories
mkdir -p dataset/temp/shards/train dataset/temp/shards/val dataset/temp/shards/test

echo "=================================================="
echo "Starting Dataset Bake: ACES EXR → WebDataset Shards (PIXI LOCAL)"
echo "=================================================="
echo "Source: dataset/temp/aces/"
echo "Target: dataset/temp/shards/"
echo ""

# Step 1: Generate the Parquet Manifest (Split 80/10/10)
echo "[1/2] Generating Parquet manifest..."
echo "      Input:  dataset/temp/aces/"
echo "      Output: dataset/temp/training_metadata.parquet"

pixi run python scripts/generate_wds_shards.py --mode manifest \
    --input_dir dataset/temp/aces \
    --output_parquet dataset/temp/training_metadata.parquet

echo "      ✓ Manifest generated"
echo ""

# Step 2: Bake the Shards (Serial to avoid lock contention)
echo "[2/2] Baking WebDataset shards..."
echo "      Input:  dataset/temp/training_metadata.parquet"
echo "      Output: dataset/temp/shards/{train,val,test}/"
echo "      Max shard size: 3.0 GB"

pixi run python scripts/generate_wds_shards.py --mode bake \
    --manifest dataset/temp/training_metadata.parquet \
    --output_dir dataset/temp/shards \
    --max_shard_size 3.0

echo "      ✓ Shards baked"
echo ""

echo "=================================================="
echo "✓ Bake complete!"
echo "=================================================="
echo ""
echo "Summary:"
ls -lh dataset/temp/shards/train/ | head -5
echo ""
wc -l dataset/temp/training_metadata.parquet 2>/dev/null || echo "Parquet manifest created"
echo ""
echo "Next: Run training with:"
echo "  pixi run python scripts/train_dequantization_net_wds.py --config-name=wds"
