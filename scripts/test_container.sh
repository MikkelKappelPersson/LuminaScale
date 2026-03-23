#!/usr/bin/env bash
# Comprehensive container test script
# Tests PyTorch/CUDA and rawtoaces functionality

set -e

echo "LuminaScale Container Test Suite"
echo "=================================="
echo ""

# Test 1: PyTorch and CUDA
echo "[1/2] Testing PyTorch and CUDA..."
/opt/venv/bin/python << 'EOF'
import torch

print(f"  PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"  CUDA available: {cuda_available}")

if cuda_available:
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    # Quick computation test
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f"  GPU computation test: ✓ Success")
else:
    print("  WARNING: CUDA not available, CPU mode only")

print("  PyTorch test: ✓ PASSED")
EOF

echo ""

# Test 2: rawtoaces
echo "[2/2] Testing rawtoaces..."
RAW_IMAGE="/home/student.aau.dk/fs62fb/projects/LuminaScale/dataset/temp/0_0.CR2"

if [ ! -f "$RAW_IMAGE" ]; then
    echo "  ERROR: Raw image not found at $RAW_IMAGE"
    echo "  rawtoaces test: ✗ FAILED"
    exit 1
fi

# Test if rawtoaces is available
if ! command -v rawtoaces &> /dev/null; then
    echo "  ERROR: rawtoaces not found in PATH"
    echo "  rawtoaces test: ✗ FAILED"
    exit 1
fi

echo "  Found rawtoaces: $(which rawtoaces)"
echo "  Raw image: $RAW_IMAGE"

# Create output directory in the project
PROJECT_ROOT="/home/student.aau.dk/fs62fb/projects/LuminaScale"
OUTPUT_DIR="$PROJECT_ROOT/temp/aces_output"
mkdir -p "$OUTPUT_DIR"

# rawtoaces auto-generates output filename: {basename}_aces.exr
INPUT_BASENAME=$(basename "$RAW_IMAGE")
BASE_WITHOUT_EXT="${INPUT_BASENAME%.*}"
OUTPUT_ACES="$OUTPUT_DIR/${BASE_WITHOUT_EXT}_aces.exr"

echo "  Output directory: $OUTPUT_DIR"

# Run rawtoaces on the CR2 file with correct syntax
rawtoaces --wb-method metadata --mat-method metadata --output-dir "$OUTPUT_DIR" --create-dirs --overwrite "$RAW_IMAGE" 2>&1
RAWTOACES_EXIT=$?

# Check what files were created
echo "  Files in output directory:"
ls -lh "$OUTPUT_DIR" 2>&1 || echo "  No files found"

if [ $RAWTOACES_EXIT -eq 0 ] && [ -f "$OUTPUT_ACES" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_ACES" | cut -f1)
    echo "  Output: $OUTPUT_ACES (size: $FILE_SIZE)"
    echo "  rawtoaces test: ✓ PASSED"
else
    # Try to find any output file that might have been created
    FOUND_OUTPUT=$(find "$OUTPUT_DIR" -type f 2>/dev/null | head -1)
    if [ -n "$FOUND_OUTPUT" ]; then
        FILE_SIZE=$(du -h "$FOUND_OUTPUT" | cut -f1)
        echo "  Found output at: $FOUND_OUTPUT (size: $FILE_SIZE)"
        echo "  rawtoaces test: ✓ PASSED"
    else
        echo "  ERROR: rawtoaces exit code: $RAWTOACES_EXIT, no output file found"
        echo "  rawtoaces test: ✗ FAILED"
        exit 1
    fi
fi

# Cleanup
# rm -rf "$TMPDIR"

echo ""
echo "=================================="
echo "✓ All tests passed!"
echo "=================================="
