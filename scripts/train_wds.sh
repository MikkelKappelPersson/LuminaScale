#!/usr/bin/env bash
#SBATCH --job-name=train_wds
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal

# Path to the LuminaScale Singularity container
# Ensure you rebuilt the container with webdataset, etc.
CONTAINER=luminascale.sif

# Run directory cleanup
mkdir -p outputs/training

echo "🚀 Starting Dequantization Training (WebDataset Pipeline)"
echo "Using Shards in: dataset/temp/shards/train/"

# Run the training script via Singularity
# Overriding shuffle_buffer slightly if needed
singularity exec --nv $CONTAINER \
    python scripts/train_dequantization_net_wds.py \
        --config-name=wds \
        batch_size=32 \
        epochs=100 \
        shard_path="dataset/temp/shards/train/train-{000000..000001}.tar"
