#!/bin/bash
# Test async prefetch implementation

echo "=== Async Prefetch Implementation Test ==="
echo "Starting single-GPU test with async threading enabled..."
echo ""

# Create a minimal test config in configs/
cp /tmp/test_async_quick.yaml configs/test_async.yaml

# Run with async enabled
srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 \
  --job-name=test_async_prefetch \
  singularity exec --nv luminascale.sif \
  python scripts/train_dequantization_net.py \
    --config-name=test_async \
    2>&1 | tee logs/test_async_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Test Complete ==="
