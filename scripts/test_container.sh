#!/usr/bin/env bash
# Comprehensive container test script
# Tests PyTorch/CUDA and rawtoaces functionality

set -e

echo "LuminaScale Container Test Suite"
echo "=================================="
echo ""

# Test 1: PyTorch and CUDA
echo "[1/2] Testing PyTorch and CUDA..."
python3 << 'EOF'
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

# Create temporary output directory
TMPDIR=$(mktemp -d)
OUTPUT_ACES="$TMPDIR/test_output.aces"

# Run rawtoaces on the CR2 file
if rawtoaces "$RAW_IMAGE" -o "$OUTPUT_ACES" 2>&1; then
    if [ -f "$OUTPUT_ACES" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_ACES" | cut -f1)
        echo "  Output: $OUTPUT_ACES (size: $FILE_SIZE)"
        echo "  rawtoaces test: ✓ PASSED"
    else
        echo "  ERROR: rawtoaces did not produce output file"
        echo "  rawtoaces test: ✗ FAILED"
        exit 1
    fi
else
    echo "  ERROR: rawtoaces command failed"
    echo "  rawtoaces test: ✗ FAILED"
    exit 1
fi

# Cleanup
rm -rf "$TMPDIR"

echo ""
echo "=================================="
echo "✓ All tests passed!"
echo "=================================="
