#!/usr/bin/env bash
# Process full dataset: Quality filter + ACES conversion + WebDataset baking
# Processes both MIT-Adobe_5K and PPR10K datasets, consolidates to /dataset/aces
# SAFE: Can be interrupted and resumed - checkpoints after each phase

set -e

# Parse command-line arguments
COLOR_SPACE="ACES2065-1"
OUTPUT_FOLDER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --color-space)
            COLOR_SPACE="$2"
            shift 2
            ;;
        --output-folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--color-space {ACES2065-1|ACEScct}] [--output-folder FOLDERNAME]"
            echo ""
            echo "Options:"
            echo "  --color-space       Color space for WebDataset shards (default: ACES2065-1)"
            echo "  --output-folder     Custom folder name (default: same as color-space)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --color-space ACEScct                       # Bake to 'dataset/ACEScct/'"
            echo "  $0 --color-space ACEScct --output-folder v2    # Bake to 'dataset/v2/'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Set output folder name (default to color space name)
if [ -z "$OUTPUT_FOLDER" ]; then
    OUTPUT_FOLDER="$COLOR_SPACE"
fi

# Validate color space
case "$COLOR_SPACE" in
    ACES2065-1|ACEScct)
        ;;
    *)
        echo "Error: Unsupported color space '$COLOR_SPACE'"
        echo "Supported: ACES2065-1, ACEScct"
        exit 1
        ;;
esac

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
    local color_space="$4"
    local sample_limit="${5:-}"
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
    echo "      Target color space: $color_space"
    echo ""

    local bake_cmd=(
        pixi run python scripts/generate_wds_shards.py --mode bake
        --manifest "$manifest"
        --output_dir "$shards_dir"
        --max_shard_size 3.0
        --crop_size 2048
        --crop_seed 42
    )
    
    # Add color space conversion flag based on target
    if [ "$color_space" = "ACEScct" ]; then
        bake_cmd+=(--convert-to-acescct)
    else
        bake_cmd+=(--no-convert-to-acescct)
    fi
    
    "${bake_cmd[@]}" || {
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
log_step "Color space: $COLOR_SPACE → Output folder: $OUTPUT_FOLDER"
echo ""

# Setup directories
EXR_DIR="$PROJECT_ROOT/dataset/exr_ACES2065-1"
OUTPUT_FULL="$PROJECT_ROOT/dataset/$OUTPUT_FOLDER/full"
OUTPUT_DEV="$PROJECT_ROOT/dataset/$OUTPUT_FOLDER/dev"
SHARDS_DIR="$OUTPUT_FULL/shards"
LOG_DIR="$PROJECT_ROOT/log"
MIT_LOG="$LOG_DIR/quality_summary_MIT-Adobe_5K.log"
PPR_LOG="$LOG_DIR/quality_summary_PPR10K.log"
COMBINED_LOG="$LOG_DIR/quality_summary_combined.log"
DEV_SAMPLE_COUNT=50

# Verify source EXR directory exists
if [ ! -d "$EXR_DIR" ]; then
    log_error "Source ACES2065-1 EXR directory not found: $EXR_DIR"
    echo "Please run quality_filtered_aces_conversion.py first to generate ACES2065-1 EXRs"
    exit 1
fi

mkdir -p "$SHARDS_DIR"/{train,val,test}
mkdir -p "$LOG_DIR"

log_step "📁 Directory structure:"
echo "   Project root: $PROJECT_ROOT"
echo "   Source EXRs:  $EXR_DIR"
echo "   Full output:  $OUTPUT_FULL"
echo "   Dev output:   $OUTPUT_DEV"
echo "   Shards dir:   $SHARDS_DIR"
echo "   Log dir:      $LOG_DIR"
echo ""

# =============================================================================
# Phase 1: Verify ACES2065-1 source files exist
# =============================================================================
log_step "=================================================="
log_step "Phase 1: Verify Source ACES2065-1 Files"
log_step "=================================================="
echo ""

TOTAL_ACES=$(find "$EXR_DIR" -name "*.exr" | wc -l)
log_step "✓ Found $TOTAL_ACES ACES2065-1 EXR files in $EXR_DIR"
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

build_webdataset_dataset "full" "$EXR_DIR" "$OUTPUT_FULL" "$COLOR_SPACE"
TRAIN_SHARDS=$(ls "$SHARDS_DIR"/train/*.tar 2>/dev/null | wc -l)
VAL_SHARDS=$(ls "$SHARDS_DIR"/val/*.tar 2>/dev/null | wc -l)
TEST_SHARDS=$(ls "$SHARDS_DIR"/test/*.tar 2>/dev/null | wc -l)

# Log the baking summary
cat >> "$COMBINED_LOG" << EOF

================================================================================
Run timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Report type: WebDataset shard baking summary
Source directory: $EXR_DIR
Target color space: $COLOR_SPACE (output folder: $OUTPUT_FOLDER)
Output directory: $OUTPUT_FULL
Notes: $COLOR_SPACE conversion happens during WebDataset shard baking
--------------------------------------------------------------------------------
FULL DATASET:
    Source ACES2065-1 files: $TOTAL_ACES
    Train shards: $TRAIN_SHARDS
    Val shards: $VAL_SHARDS
    Test shards: $TEST_SHARDS
================================================================================
EOF

echo "   Log file:              $COMBINED_LOG"
echo ""
log_step "=================================================="
log_step "Phase 3: Dev Dataset (50 images)"
log_step "=================================================="
echo ""

build_webdataset_dataset "dev" "$EXR_DIR" "$OUTPUT_DEV" "$COLOR_SPACE" "$DEV_SAMPLE_COUNT"

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
echo "   ACES2065-1 files:      $TOTAL_ACES (stored in $EXR_DIR)"
echo "   Train shards:          $TRAIN_SHARDS files ($COLOR_SPACE encoded)"
echo "   Val shards:            $VAL_SHARDS files ($COLOR_SPACE encoded)"
echo "   Test shards:           $TEST_SHARDS files ($COLOR_SPACE encoded)"
echo ""
echo "📊 Dev dataset:"
echo "   Source ACES2065-1 files: first $DEV_SAMPLE_COUNT from $EXR_DIR"
echo "   Train shards:          $DEV_TRAIN_SHARDS files ($COLOR_SPACE encoded)"
echo "   Val shards:            $DEV_VAL_SHARDS files ($COLOR_SPACE encoded)"
echo "   Test shards:           $DEV_TEST_SHARDS files ($COLOR_SPACE encoded)"
echo ""
echo "📁 Output directory structure:"
tree -L 2 "$PROJECT_ROOT/dataset" 2>/dev/null || find "$PROJECT_ROOT/dataset" -maxdepth 2 -type d | sed "s|$PROJECT_ROOT/dataset|dataset|" | sort
echo ""
echo "📝 Next: Use datasets with WebDataset loader:"
echo "   Full: import webdataset as wds"
echo "         dataset = wds.WebDataset('$OUTPUT_FULL/shards/{train,val,test}-*.tar')"
echo "   Dev:  import webdataset as wds"
echo "         dataset = wds.WebDataset('$OUTPUT_DEV/shards/{train,val,test}-*.tar')"
echo ""
log_step "⏱️  Processing complete! $(date '+%Y-%m-%d %H:%M:%S')"
