#!/bin/bash
#SBATCH --job-name=lumina_inference
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/inference_%j.out
#SBATCH --error=outputs/logs/inference_%j.err

# Load Singularity container and run the inference script
# Provide the checkpoint path and input image as arguments

RUN_ID=$(date +%Y%m%d_%H%M%S)
CHECKPOINT=${1:-"outputs/training/LATEST/checkpoints/latest.pt"}
INPUT=${2:-"dataset/temp/srgb_looks/10_1.png"}
OUTPUT_DIR=${3:-"outputs/inference/${RUN_ID}"}
OUTPUT_FILE="${OUTPUT_DIR}/result.exr"

echo "Starting inference job..."
echo "Checkpoint: $CHECKPOINT"
echo "Input: $INPUT"
echo "Output: $OUTPUT_FILE"

mkdir -p "$OUTPUT_DIR" outputs/logs

# Run via singularity
# Note: Ensure the project root and /mnt mounts are available
singularity exec --nv \
    --bind /home/student.aau.dk/fs62fb/projects/LuminaScale:/app \
    /home/student.aau.dk/fs62fb/singularity/luminascale.sif \
    python scripts/run_inference.py \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT" \
    --output "$OUTPUT_FILE"
    --bind /mnt/MKP01:/mnt/MKP01 \
    /home/student.aau.dk/fs62fb/projects/LuminaScale/singularity/luminascale.sif \
    python3 /app/scripts/run_inference.py \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device cuda
