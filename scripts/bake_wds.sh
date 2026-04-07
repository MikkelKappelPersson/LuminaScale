#!/usr/bin/env bash
#SBATCH --job-name=bake_wds_shards
#SBATCH --output=outputs/logs/bake_%j.out
#SBATCH --error=outputs/logs/bake_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal

# Path to the LuminaScale Singularity container
CONTAINER=luminascale.sif

# Output directory for the WebDataset shards
# We store them in dataset/temp/shards/ follow by the split name
mkdir -p dataset/temp/shards/train dataset/temp/shards/val dataset/temp/shards/test

echo "Starting Dataset Bake: ACES EXR -> WebDataset Shards"
echo "Target: dataset/temp/shards/"

# 1. Generate the Parquet Manifest (Split 80/10/10)
# This assumes your EXRs are in dataset/temp/aces/
singularity exec --nv $CONTAINER \
    python scripts/generate_wds_shards.py --mode manifest \
        --input_dir dataset/temp/aces \
        --output_parquet dataset/temp/training_metadata.parquet

# 2. Bake the Shards (Serial process to avoid filesystem lock contention)
# Max shard size set to 3GB (~10-15 large EXRs per shard)
singularity exec --nv $CONTAINER \
    python scripts/generate_wds_shards.py --mode bake \
        --manifest dataset/temp/training_metadata.parquet \
        --output_dir dataset/temp/shards \
        --max_shard_size 3.0

echo "Bake complete. Manifest saved to dataset/temp/training_metadata.parquet"
