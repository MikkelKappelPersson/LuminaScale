#!/usr/bin/env bash
#SBATCH --job-name=luminascale_process_dataset
#SBATCH --output=outputs/logs/process_dataset_%j.out
#SBATCH --error=outputs/logs/process_dataset_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

# HPC SBATCH equivalent of process_full_dataset.sh
# Runs inside Singularity container with GPU support
# Usage:
#   sbatch scripts/process_full_dataset_hpc.sh --color-space ACEScct
#   sbatch scripts/process_full_dataset_hpc.sh --color-space ACEScct --output-folder v2

set -e

# Singularity container path
CONTAINER="luminascale.sif"

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Singularity container not found: $CONTAINER"
    echo "Build with: singularity build $CONTAINER singularity/luminascale.def"
    exit 1
fi

echo "================================================================================"
echo "LuminaScale Full Dataset Pipeline - HPC (Singularity)"
echo "================================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Container: $CONTAINER"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPUs: $SLURM_GPUS"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""
echo "Command-line args: $@"
echo "================================================================================"
echo ""

mkdir -p outputs/logs

# Run process_full_dataset.sh inside Singularity container with GPU support
singularity exec --nv \
    --bind "$PWD:$PWD" \
    --workdir "$PWD" \
    "$CONTAINER" \
    bash scripts/process_full_dataset.sh "$@"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Dataset processing complete! (Exit code: $EXIT_CODE)"
else
    echo "❌ Dataset processing failed! (Exit code: $EXIT_CODE)"
fi
echo "================================================================================"
echo "Output logs: outputs/logs/"
echo "Log files created:"
find outputs/logs/ -name "process_dataset_*.out" -newer /proc -2>&1 | sed 's|^|  - |' || echo "  (check outputs/logs/ directory)"
echo ""

exit $EXIT_CODE
