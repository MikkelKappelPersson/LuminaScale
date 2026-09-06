#!/usr/bin/env bash
# Local GPU Testing: Convert ACES EXRs → WebDataset Shards using Pixi
# No SBATCH/HPC required, runs directly on local GPU
#
# Layout (see dataset/README.md):
#   reads  dataset/exr/ACES2065-1/           (ACES EXR sources)
#   writes dataset/shards/<COLORSPACE>/<SPLIT>/{shards/{train,val,test},
#           training_metadata.parquet}
#
# Usage: bake_wds_local.sh [COLORSPACE] [SPLIT] [MAX_SAMPLES]
#   COLORSPACE   ACEScct (default) | ACES2065-1
#   SPLIT        full (default) | dev
#   MAX_SAMPLES  limit manifest to first N EXRs (smoke sets; implies dev)

set -e

COLORSPACE="${1:-ACEScct}"
SPLIT="${2:-full}"
MAX_SAMPLES="${3:-}"

case "$COLORSPACE" in
    ACEScct)    CS_FLAG="--convert-to-acescct" ;;
    ACES2065-1) CS_FLAG="--no-convert-to-acescct" ;;
    *) echo "Unknown colorspace: $COLORSPACE (use ACEScct or ACES2065-1)" >&2; exit 1 ;;
esac

EXR_DIR="dataset/exr/ACES2065-1"
OUT_DIR="dataset/shards/$COLORSPACE/$SPLIT"
MANIFEST="$OUT_DIR/training_metadata.parquet"

MAX_FLAG=""
[ -n "$MAX_SAMPLES" ] && MAX_FLAG="--max_samples $MAX_SAMPLES"

mkdir -p "$OUT_DIR/shards/train" "$OUT_DIR/shards/val" "$OUT_DIR/shards/test"

echo "=================================================="
echo "Starting Dataset Bake: ACES EXR → WebDataset Shards (PIXI LOCAL)"
echo "=================================================="
echo "Source:      $EXR_DIR"
echo "Target:      $OUT_DIR"
echo "Colour space: $COLORSPACE   Split: $SPLIT${MAX_SAMPLES:+   max_samples: $MAX_SAMPLES}"
echo ""

# Step 1: Generate the Parquet Manifest (Split 80/10/10)
echo "[1/2] Generating Parquet manifest..."
pixi run python scripts/generate_wds_shards.py --mode manifest \
    --input_dir "$EXR_DIR" \
    --output_parquet "$MANIFEST" $MAX_FLAG
echo "      ✓ Manifest generated"
echo ""

# Step 2: Bake the Shards (Serial to avoid lock contention)
echo "[2/2] Baking WebDataset shards (max shard size 3.0 GB)..."
pixi run python scripts/generate_wds_shards.py --mode bake \
    --manifest "$MANIFEST" \
    --output_dir "$OUT_DIR" \
    --max_shard_size 3.0 $CS_FLAG
echo "      ✓ Shards baked"
echo ""

echo "=================================================="
echo "✓ Bake complete!"
echo "=================================================="
echo ""
echo "Summary:"
ls -lh "$OUT_DIR/shards/train/" | head -5
echo ""
wc -l "$MANIFEST" 2>/dev/null || echo "Parquet manifest created"
echo ""
echo "Next: point training at the new shards, e.g.:"
echo "  pixi run python scripts/train_dequant_net_wds.py --config-name=wds \\"
echo "      shard_path=$OUT_DIR/shards/train \\"
echo "      val_shard_path=$OUT_DIR/shards/val \\"
echo "      metadata_parquet=$MANIFEST"
