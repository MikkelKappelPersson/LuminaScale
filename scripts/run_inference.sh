#!/bin/bash
#SBATCH --job-name=lumina_inference
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Load Singularity container and run the inference script
# Provide the checkpoint path and input image as arguments

CHECKPOINT=${1:-"/mnt/MKP01/med8_project/LuminaScale/dataset/temp/test_run/20260331_164330_dequant_net_epoch_1.pt"}
INPUT=${2:-"dataset/temp/srgb_looks/10_1.png"}
OUTPUT=${3:-"outputs/inference/result_$(date +%Y%m%d_%H%M%S).exr"}

echo "Starting inference job..."
echo "Checkpoint: $CHECKPOINT"
echo "Input: $INPUT"
echo "Output: $OUTPUT"

mkdir -p outputs/inference logs

# Run via singularity
# Note: Ensure the project root and /mnt mounts are available
singularity exec --nv \
    --bind /home/student.aau.dk/fs62fb/projects/LuminaScale:/app \
    --bind /mnt/MKP01:/mnt/MKP01 \
    /home/student.aau.dk/fs62fb/projects/LuminaScale/singularity/luminascale.sif \
    python3 /app/scripts/run_inference.py \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device cuda
