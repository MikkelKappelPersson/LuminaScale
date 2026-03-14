#!/bin/bash
#SBATCH --job-name=extract_dataset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/extract_dataset_%j.log
#SBATCH --error=logs/extract_dataset_%j.err

set -e

echo "Starting dataset extraction..."
cd /home/student.aau.dk/fs62fb/projects/LuminaScale/dataset/temp

echo "Extracting dataset.tar.gz..."
tar -xzf dataset.tar.gz

echo "Extraction complete!"
ls -lh
