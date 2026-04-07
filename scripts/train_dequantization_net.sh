#!/bin/bash

# Slurm batch script for dequantization network training
# 
# Usage:
#   sbatch scripts/train_dequantization_net.sh
#   sbatch --time=48:00:00 scripts/train_dequantization_net.sh  # Override time
#

#SBATCH --job-name=train_dequant
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --output=outputs/logs/train_dequantization_%j.log
#SBATCH --error=outputs/logs/train_dequantization_%j.err


# The container image we want to launch:
CONTAINER_IMAGE="luminascale.sif"

# Ensure output directories exist
mkdir -p outputs/logs

# Run training with Singularity using the test config and 1 GPU
# The --nv flag is required to use the GPU inside the container
singularity exec --nv "$CONTAINER_IMAGE" \
  python scripts/train_dequantization_net.py \
    --config-path=../configs \
    --config-name=test

echo "Training completed with exit code: $?"
