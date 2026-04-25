#!/usr/bin/env bash
# Process full dataset: Quality filter + ACES conversion + WebDataset baking
# Processes both MIT-Adobe_5K and PPR10K datasets, consolidates to /dataset/full
# SAFE: Can be interrupted and resumed - checkpoints after each phase

set -e

cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" || exit 1
PROJECT_ROOT=$(pwd)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $1"
}

extract_metric_from_log() {
    local logfile="$1"
    local metric_label="$2"

    if [ ! -f "$logfile" ]; then
        echo "0"
        return
    fi

    local value
    value=$(grep -F "${metric_label}:" "$logfile" | tail -1 | awk -F': ' '{print $2}' | tr -d '[:space:]')
    if [ -z "$value" ]; then
        value="0"
    fi
    echo "$value"
}

build_webdataset_dataset() {
    local dataset_name="$1"
    local input_dir="$2"
    local output_dir="$3"
    local sample_limit="${4:-}"
    local manifest="$output_dir/training_metadata.parquet"
    local shards_dir="$output_dir/shards"

    mkdir -p "$output_dir"
    mkdir -p "$shards_dir"/{train,val,test}

    log_step "[$dataset_name] Generating Parquet manifest..."
    echo "      Input:  $input_dir"
    echo "      Output: $manifest"
    if [ -n "$sample_limit" ]; then
        echo "      Limit:  first $sample_limit images"
    fi
    echo ""

    local manifest_cmd=(
        pixi run python scripts/generate_wds_shards.py --mode manifest
        --input_dir "$input_dir"
        --output_parquet "$manifest"
    )
    if [ -n "$sample_limit" ]; then
        manifest_cmd+=(--max_samples "$sample_limit")
    fi

    "${manifest_cmd[@]}" || {
        log_error "$dataset_name manifest generation failed. Run the command again to resume."
        exit 1
    }

    echo ""
    log_step "[$dataset_name] Baking WebDataset shards..."
    echo "      Manifest: $manifest"
    echo "      Output:   $shards_dir"
    echo "      Max shard size: 3.0 GB"
    echo "      Random crop: 2048x2048 (seed=42)"
    echo ""

    pixi run python scripts/generate_wds_shards.py --mode bake \
      --manifest "$manifest" \
      --output_dir "$shards_dir" \
        --max_shard_size 3.0 \
        --crop_size 2048 \
        --crop_seed 42 || {
        log_error "$dataset_name shard baking failed. Run the command again to resume from where it stopped."
        echo ""
        echo "Partial shards saved in: $shards_dir"
        echo "Re-run this script to continue from the last checkpoint."
        exit 1
    }

    echo ""
    log_step "✓ $dataset_name dataset complete!"
}

log_step "🚀 LuminaScale Full Dataset Pipeline (Resumable)"
log_step "Project root: $PROJECT_ROOT"
echo ""

# Setup directories
OUTPUT_FULL="$PROJECT_ROOT/dataset/full"
OUTPUT_DEV="$PROJECT_ROOT/dataset/dev"
ACES_DIR="$OUTPUT_FULL/aces"
SHARDS_DIR="$OUTPUT_FULL/shards"
MIT_LOG="$OUTPUT_FULL/quality_summary_MIT-Adobe_5K.log"
PPR_LOG="$OUTPUT_FULL/quality_summary_PPR10K.log"
COMBINED_LOG="$OUTPUT_FULL/quality_summary_combined.log"
DEV_SAMPLE_COUNT=50

mkdir -p "$ACES_DIR"
mkdir -p "$SHARDS_DIR"/{train,val,test}

log_step "📁 Output directories:"
echo "   ACES files: $ACES_DIR"
echo "   Shards:     $SHARDS_DIR"
echo "   MIT log:    $MIT_LOG"
echo "   PPR log:    $PPR_LOG"
echo "   Combined:   $COMBINED_LOG"
echo ""

# =============================================================================
# Phase 1: Quality Filter & Convert to ACES
# =============================================================================
log_step "=================================================="
log_step "Phase 1: Quality Filter & ACES Conversion"
log_step "=================================================="
echo ""

# MIT-Adobe_5K
MIT_COUNT=$(find "$ACES_DIR" -name "MIT-Adobe_5K_*.exr" 2>/dev/null | wc -l)
log_step "[1/2] Processing MIT-Adobe_5K (full dataset)..."
echo "      Input:  ../dataset/MIT-Adobe_5K/raw"
echo "      Prefix: MIT-Adobe_5K_"
echo "      Current converted: $MIT_COUNT"
echo ""

pixi run python scripts/quality_filtered_aces_conversion.py \
    --input-dir ../dataset/MIT-Adobe_5K/raw \
    --output-dir "$ACES_DIR" \
    --dataset-prefix "MIT-Adobe_5K_" \
    --highlight-clip 2.0 \
    --noise-floor 10.0 \
    --summary-log "$MIT_LOG" || {
        log_error "MIT-Adobe_5K conversion failed. Run the command again to resume."
        exit 1
}

echo ""

# PPR10K
PPR_COUNT=$(find "$ACES_DIR" -name "PPR10K_*.exr" 2>/dev/null | wc -l)
log_step "[2/2] Processing PPR10K (full dataset)..."
echo "      Input:  ../dataset/PPR10K/raw"
echo "      Prefix: PPR10K_"
echo "      Current converted: $PPR_COUNT"
echo ""

pixi run python scripts/quality_filtered_aces_conversion.py \
    --input-dir ../dataset/PPR10K/raw \
    --output-dir "$ACES_DIR" \
    --dataset-prefix "PPR10K_" \
    --highlight-clip 2.0 \
    --noise-floor 10.0 \
    --summary-log "$PPR_LOG" || {
        log_error "PPR10K conversion failed. Run the command again to resume."
        exit 1
}

echo ""
log_step "✓ Phase 1 complete!"
TOTAL_ACES=$(find "$ACES_DIR" -name "*.exr" | wc -l)
echo "📊 Total ACES files: $TOTAL_ACES"
echo ""

if [ "$TOTAL_ACES" -lt "$DEV_SAMPLE_COUNT" ]; then
    log_error "Cannot build dev dataset: found only $TOTAL_ACES ACES files, need at least $DEV_SAMPLE_COUNT"
    exit 1
fi

# =============================================================================
# Phase 2: Generate Manifest & Bake WebDataset Shards
# =============================================================================
log_step "=================================================="
log_step "Phase 2: WebDataset Manifest & Sharding"
log_step "=================================================="
echo ""

build_webdataset_dataset "full" "$ACES_DIR" "$OUTPUT_FULL"
TRAIN_SHARDS=$(ls "$SHARDS_DIR"/train/*.tar 2>/dev/null | wc -l)
VAL_SHARDS=$(ls "$SHARDS_DIR"/val/*.tar 2>/dev/null | wc -l)
TEST_SHARDS=$(ls "$SHARDS_DIR"/test/*.tar 2>/dev/null | wc -l)

MIT_PASSED=$(extract_metric_from_log "$MIT_LOG" "Passed quality check")
MIT_QFAILED=$(extract_metric_from_log "$MIT_LOG" "Failed quality check")
MIT_CLIP=$(extract_metric_from_log "$MIT_LOG" "  - Failed due to clipping only")
MIT_NOISE=$(extract_metric_from_log "$MIT_LOG" "  - Failed due to noise only")
MIT_BOTH=$(extract_metric_from_log "$MIT_LOG" "  - Failed due to both")
MIT_CONVERTED=$(extract_metric_from_log "$MIT_LOG" "Successfully converted")
MIT_EXCLUDED=$(extract_metric_from_log "$MIT_LOG" "Excluded (no spectral data)")
MIT_CFAILED=$(extract_metric_from_log "$MIT_LOG" "Conversion failures")

PPR_PASSED=$(extract_metric_from_log "$PPR_LOG" "Passed quality check")
PPR_QFAILED=$(extract_metric_from_log "$PPR_LOG" "Failed quality check")
PPR_CLIP=$(extract_metric_from_log "$PPR_LOG" "  - Failed due to clipping only")
PPR_NOISE=$(extract_metric_from_log "$PPR_LOG" "  - Failed due to noise only")
PPR_BOTH=$(extract_metric_from_log "$PPR_LOG" "  - Failed due to both")
PPR_CONVERTED=$(extract_metric_from_log "$PPR_LOG" "Successfully converted")
PPR_EXCLUDED=$(extract_metric_from_log "$PPR_LOG" "Excluded (no spectral data)")
PPR_CFAILED=$(extract_metric_from_log "$PPR_LOG" "Conversion failures")

TOTAL_PASSED=$((MIT_PASSED + PPR_PASSED))
TOTAL_QFAILED=$((MIT_QFAILED + PPR_QFAILED))
TOTAL_CLIP=$((MIT_CLIP + PPR_CLIP))
TOTAL_NOISE=$((MIT_NOISE + PPR_NOISE))
TOTAL_BOTH=$((MIT_BOTH + PPR_BOTH))
TOTAL_CONVERTED=$((MIT_CONVERTED + PPR_CONVERTED))
TOTAL_EXCLUDED=$((MIT_EXCLUDED + PPR_EXCLUDED))
TOTAL_CFAILED=$((MIT_CFAILED + PPR_CFAILED))

cat >> "$COMBINED_LOG" << EOF

================================================================================
Run timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Report type: Combined quality + conversion summary
Output directory: $OUTPUT_FULL
Thresholds: highlight_clip <= 2.0%, noise_floor <= 10.0
--------------------------------------------------------------------------------
MIT-Adobe_5K:
    Passed quality check: $MIT_PASSED
    Failed quality check: $MIT_QFAILED
        - Failed due to clipping only: $MIT_CLIP
        - Failed due to noise only: $MIT_NOISE
        - Failed due to both: $MIT_BOTH
    Successfully converted: $MIT_CONVERTED
    Excluded (no spectral data): $MIT_EXCLUDED
    Conversion failures: $MIT_CFAILED

PPR10K:
    Passed quality check: $PPR_PASSED
    Failed quality check: $PPR_QFAILED
        - Failed due to clipping only: $PPR_CLIP
        - Failed due to noise only: $PPR_NOISE
        - Failed due to both: $PPR_BOTH
    Successfully converted: $PPR_CONVERTED
    Excluded (no spectral data): $PPR_EXCLUDED
    Conversion failures: $PPR_CFAILED

TOTAL:
    ACES files present: $TOTAL_ACES
    Passed quality check: $TOTAL_PASSED
    Failed quality check: $TOTAL_QFAILED
        - Failed due to clipping only: $TOTAL_CLIP
        - Failed due to noise only: $TOTAL_NOISE
        - Failed due to both: $TOTAL_BOTH
    Successfully converted: $TOTAL_CONVERTED
    Excluded (no spectral data): $TOTAL_EXCLUDED
    Conversion failures: $TOTAL_CFAILED
    Train shards: $TRAIN_SHARDS
    Val shards: $VAL_SHARDS
    Test shards: $TEST_SHARDS
================================================================================
EOF

echo "   Combined log:          $COMBINED_LOG"
echo ""
log_step "=================================================="
log_step "Phase 3: Dev Dataset (50 images)"
log_step "=================================================="
echo ""

build_webdataset_dataset "dev" "$ACES_DIR" "$OUTPUT_DEV" "$DEV_SAMPLE_COUNT"

DEV_SHARDS_DIR="$OUTPUT_DEV/shards"
DEV_TRAIN_SHARDS=$(ls "$DEV_SHARDS_DIR"/train/*.tar 2>/dev/null | wc -l)
DEV_VAL_SHARDS=$(ls "$DEV_SHARDS_DIR"/val/*.tar 2>/dev/null | wc -l)
DEV_TEST_SHARDS=$(ls "$DEV_SHARDS_DIR"/test/*.tar 2>/dev/null | wc -l)

echo ""
log_step "=================================================="
log_step "✅ Dataset Builds Complete!"
log_step "=================================================="
echo ""
echo "📊 Full dataset:"
echo "   ACES files:            $TOTAL_ACES"
echo "   Train shards:          $TRAIN_SHARDS files"
echo "   Val shards:            $VAL_SHARDS files"
echo "   Test shards:           $TEST_SHARDS files"
echo ""
echo "📊 Dev dataset:"
echo "   Source ACES files:     first $DEV_SAMPLE_COUNT from $ACES_DIR"
echo "   Train shards:          $DEV_TRAIN_SHARDS files"
echo "   Val shards:            $DEV_VAL_SHARDS files"
echo "   Test shards:           $DEV_TEST_SHARDS files"
echo ""
echo "📁 Full output structure:"
tree -L 2 "$OUTPUT_FULL" 2>/dev/null || find "$OUTPUT_FULL" -type d | sed 's|'$OUTPUT_FULL'||' | sort
echo ""
echo "📁 Dev output structure:"
tree -L 2 "$OUTPUT_DEV" 2>/dev/null || find "$OUTPUT_DEV" -type d | sed 's|'$OUTPUT_DEV'||' | sort
echo ""
echo "📝 Next: Use datasets with WebDataset loader:"
echo "   Full: import webdataset as wds"
echo "         dataset = wds.WebDataset('$OUTPUT_FULL/shards/{train,val,test}-*.tar')"
echo "   Dev:  import webdataset as wds"
echo "         dataset = wds.WebDataset('$OUTPUT_DEV/shards/{train,val,test}-*.tar')"
echo ""
log_step "⏱️  Processing complete! $(date '+%Y-%m-%d %H:%M:%S')"
