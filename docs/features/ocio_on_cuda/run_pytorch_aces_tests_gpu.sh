#!/usr/bin/env bash
#SBATCH --job-name=pytorch_aces_gpu_tests
#SBATCH --output=logs/pytorch_aces_gpu_tests_%j.out
#SBATCH --error=logs/pytorch_aces_gpu_tests_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal

set -e

CONTAINER=/home/student.aau.dk/fs62fb/projects/LuminaScale/luminascale.sif
WORKDIR=/home/student.aau.dk/fs62fb/projects/LuminaScale

echo "=========================================="
echo "PyTorch ACES GPU Test Suite"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
nvidia-smi --query-gpu=memory.total --format=csv,noheader
echo "=========================================="

# Run tests in container
echo ""
echo "[1/2] Running Unit Tests (18 tests)..."
echo "------"
singularity exec --nv "$CONTAINER" bash -c "cd $WORKDIR && python -m pytest tests/test_pytorch_aces_transformer.py -v --tb=short"

echo ""
echo "[2/2] Running Benchmark (PyTorch vs OCIO on GPU)..."
echo "------"
singularity exec --nv "$CONTAINER" bash -c "cd $WORKDIR && python scripts/benchmark_pytorch_vs_ocio.py"

echo ""
echo "=========================================="
echo "✅ All tests complete!"
echo "Check outputs/benchmark_visualizations/ for comparisons"
echo "=========================================="
