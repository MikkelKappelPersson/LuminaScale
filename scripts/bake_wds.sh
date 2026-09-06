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

# Layout (see dataset/README.md):
#   reads  dataset/exr/ACES2065-1/           (ACES EXR sources)
#   writes dataset/shards/<COLORSPACE>/<SPLIT>/{shards/{train,val,test},
#           training_metadata.parquet}
#
# Usage (sbatch passes no args; export the variables before submitting):
#   COLORSPACE=ACEScct SPLIT=full MAX_SAMPLES= sbatch scripts/bake_wds.sh

set -e

COLORSPACE="${COLORSPACE:-ACEScct}"
SPLIT="${SPLIT:-full}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

case "$COLORSPACE" in
    ACEScct)    CS_FLAG="--convert-to-acescct" ;;
    ACES2065-1) CS_FLAG="--no-convert-to-acescct" ;;
    *) echo "Unknown colorspace: $COLORSPACE (use ACEScct or ACES2065-1)" >&2; exit 1 ;;
esac

# Path to the LuminaScale Singularity container
CONTAINER=luminascale.sif

EXR_DIR="dataset/exr/ACES2065-1"
OUT_DIR="dataset/shards/$COLORSPACE/$SPLIT"
MANIFEST="$OUT_DIR/training_metadata.parquet"

MAX_FLAG=""
[ -n "$MAX_SAMPLES" ] && MAX_FLAG="--max_samples $MAX_SAMPLES"

mkdir -p "$OUT_DIR/shards/train" "$OUT_DIR/shards/val" "$OUT_DIR/shards/test"

echo "Starting Dataset Bake: ACES EXR -> WebDataset Shards"
echo "Source:      $EXR_DIR"
echo "Target:      $OUT_DIR"
echo "Colour space: $COLORSPACE   Split: $SPLIT${MAX_SAMPLES:+   max_samples: $MAX_SAMPLES}"

# 1. Generate the Parquet Manifest (Split 80/10/10)
# This assumes your EXRs are in dataset/exr/ACES2065-1/
singularity exec --nv $CONTAINER \
    python scripts/generate_wds_shards.py --mode manifest \
        --input_dir "$EXR_DIR" \
        --output_parquet "$MANIFEST" $MAX_FLAG

# 2. Bake the Shards (Serial process to avoid filesystem lock contention)
# Max shard size set to 3GB (~10-15 large EXRs per shard)
# Each image is stored ONCE. During training, WebDataset.repeat(patches_per_image) loops through
# the data to enable on-the-fly patch generation.
singularity exec --nv $CONTAINER \
    python scripts/generate_wds_shards.py --mode bake \
        --manifest "$MANIFEST" \
        --output_dir "$OUT_DIR" \
        --max_shard_size 3.0 $CS_FLAG

echo "Bake complete. Manifest saved to $MANIFEST"
