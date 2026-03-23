#!/usr/bin/env bash
#SBATCH --job-name=jupyter
#SBATCH --output=logs/jupyter_%j.log
#SBATCH --error=logs/jupyter_%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Get the hostname and port
HOSTNAME=$(hostname)
PORT=8888

echo "=========================================="
echo "Jupyter Server Started"
echo "=========================================="
echo "Node: $HOSTNAME"
echo "Port: $PORT"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "To connect via SSH tunneling:"
echo "  ssh -L $PORT:localhost:$PORT student.aau.dk"
echo ""
echo "Then open in VS Code or browser:"
echo "  http://localhost:$PORT"
echo "=========================================="
echo ""

# Start Jupyter in the container
singularity exec --nv /home/student.aau.dk/fs62fb/projects/LuminaScale/luminascale.sif \
    jupyter notebook \
        --ip=0.0.0.0 \
        --port=$PORT \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password='' \
        ~/projects/LuminaScale/notebooks

echo "Jupyter session ended."
