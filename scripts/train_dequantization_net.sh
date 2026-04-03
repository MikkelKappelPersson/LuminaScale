#!/bin/bash

# Slurm batch script for dequantization network training
# 
# Usage:
#   sbatch scripts/train_dequantization_net.sh
#   sbatch --time=48:00:00 scripts/train_dequantization_net.sh  # Override time
#

#SBATCH --job-name=train_dequant
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/train_dequantization_%j.log
#SBATCH --error=logs/train_dequantization_%j.err

# Load modules (adjust based on AAU HPC environment)
module load python/3.12
module load cuda/12.1

# Activate environment
source ~/.venv/bin/activate 2>/dev/null || source ~/miniconda3/bin/activate luminascale

# Set data paths (override with --hdr_dir if needed)
export HDR_DIR="${HDR_DIR:-/lustre/scratch/fs62fb/data/hdr}"
export SRGB_DIR="${SRGB_DIR:-/lustre/scratch/fs62fb/data/srgb_looks}"
export OUTPUT_DIR="${OUTPUT_DIR:-/lustre/scratch/fs62fb/outputs/training}"

# Ensure output directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Run training with HPC-optimized config
python scripts/train_dequantization_net.py \
  --config-path=configs \
  --config-name=hpc_slurm \
  hdr_dir="$HDR_DIR" \
  srgb_dir="$SRGB_DIR" \
  output_dir="$OUTPUT_DIR"

echo "Training completed with exit code: $?"
