#!/usr/bin/env bash
#SBATCH --job-name=smoke_dequant_dev
#SBATCH --output=outputs/logs/smoke_%j.out
#SBATCH --error=outputs/logs/smoke_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal

# M2 smoke run: 1 epoch of Dequant-Net on the ACEScct/dev shards.
# Purpose: prove container + runtime env + data path + GPU end-to-end.
# Not a result run — artifacts are disposable.

set -e

CONTAINER="$HOME/projects/LuminaScale/luminascale.sif"
WORKDIR="$HOME/projects/LuminaScale"

echo "=========================================="
echo "LuminaScale M2 smoke run (Dequant, ACEScct/dev, 1 epoch)"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "=========================================="

singularity exec --nv "$CONTAINER" bash -c "cd $WORKDIR && python scripts/train_dequant_net.py \
    --config-name=dequant_dev \
    epochs=1 \
    shard_path=dataset/shards/ACEScct/dev/shards/train \
    val_shard_path=dataset/shards/ACEScct/dev/shards/val \
    metadata_parquet=dataset/shards/ACEScct/dev/training_metadata.parquet"

echo "=========================================="
echo "✓ Smoke run complete"
echo "=========================================="
