#!/usr/bin/env bash
#SBATCH --job-name=train_dequantization_net
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal

# For running with srun: 
# Path to the LuminaScale Singularity container

CONTAINER=luminascale.sif

# Run directory cleanup
mkdir -p outputs/training

echo "🚀 Starting Dequantization Training (WebDataset Pipeline)"
echo "Using Shards in: dataset/temp/shards/train/"

# Run the training script via Singularity
# Overriding shuffle_buffer slightly if needed
singularity exec --nv $CONTAINER \
    python scripts/train_dequantization_net.py \
        --config-name=default
