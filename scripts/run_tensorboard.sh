#!/usr/bin/env bash
#SBATCH --job-name=tensorboard
#SBATCH --output=logs/tensorboard_%j.out
#SBATCH --error=logs/tensorboard_%j.err
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00

singularity exec luminascale.sif \
    tensorboard --logdir=outputs/training --port=6006 --bind_all
