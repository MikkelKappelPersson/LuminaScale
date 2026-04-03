#!/usr/bin/env bash
#SBATCH --job-name=pack_lmdb
#SBATCH --output=logs/pack_lmdb_%j.out
#SBATCH --error=logs/pack_lmdb_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal

# The container image we want to launch:
container_image="luminascale.sif"

# Run packing script inside container
singularity exec $container_image \
    python scripts/pack_lmdb.py \
        --aces-dir dataset/temp/aces \
        --output-path dataset/training_data.lmdb \
        --seed 42 \
        --map-size 1000000000000
