#!/bin/bash
#SBATCH --job-name=multigpu-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --output=logs/multigpu_test_%j.log
#SBATCH --error=logs/multigpu_test_%j.err

set -e

# Load required modules
module load singularity

# Set environment variables
export OCIO=/home/student.aau.dk/fs62fb/projects/LuminaScale/config/aces/studio-config.ocio
export CUDA_VISIBLE_DEVICES=0,1

# Print diagnostic info
echo "=== Multi-GPU Test ==="
echo "HOSTNAME: $(hostname)"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

# Test 1: Single GPU EGL initialization
echo ""
echo "=== Test 1: Single GPU EGL init ==="
srun --ntasks=1 bash -c 'cd /home/student.aau.dk/fs62fb/projects/LuminaScale && python3 -c "
import torch
torch.cuda.set_device(0)
from luminascale.utils.gpu_torch_processor import GPUTorchProcessor
try:
    processor = GPUTorchProcessor(headless=True, gpu_id=0)
    print(\"✓ GPU 0 EGL initialized successfully\")
    processor.cleanup()
except Exception as e:
    print(f\"✗ GPU 0 EGL failed: {e}\")
    exit(1)
"'

# Test 2: Multi-GPU EGL initialization (RUN SEPARATELY for each GPU)
echo ""
echo "=== Test 2: Multi-GPU EGL init (GPU 1) ==="
srun --ntasks=1 bash -c 'cd /home/student.aau.dk/fs62fb/projects/LuminaScale && python3 -c "
import torch
torch.cuda.set_device(1)
from luminascale.utils.gpu_torch_processor import GPUTorchProcessor
try:
    processor = GPUTorchProcessor(headless=True, gpu_id=1)
    print(\"✓ GPU 1 EGL initialized successfully\")
    processor.cleanup()
except Exception as e:
    print(f\"✗ GPU 1 EGL failed: {e}\")
    exit(1)
"'

# Test 3: Tensor transform on GPU 0
echo ""
echo "=== Test 3: Tensor transform on GPU 0 ==="
srun --ntasks=1 bash -c 'cd /home/student.aau.dk/fs62fb/projects/LuminaScale && python3 -c "
import torch
torch.cuda.set_device(0)
from luminascale.utils.gpu_torch_processor import GPUTorchProcessor

try:
    processor = GPUTorchProcessor(headless=True, gpu_id=0)
    
    # Create test tensor (1x3 image)
    test_tensor = torch.ones(1, 1, 3, dtype=torch.float32, device=\"cuda:0\") * 0.5
    
    # Apply ACES2065-1 -> sRGB transform
    srgb_32f, srgb_8u = processor.apply_ocio_torch(test_tensor)
    
    print(f\"✓ Transform succeeded: input {test_tensor.shape} -> {srgb_32f.shape} (float32), {srgb_8u.shape} (uint8)\")
    print(f\"  Output range (f32): [{srgb_32f.min():.3f}, {srgb_32f.max():.3f}]\")
    print(f\"  Output range (u8):  [{srgb_8u.min()}, {srgb_8u.max()}]\")
    processor.cleanup()
except Exception as e:
    print(f\"✗ Transform failed: {e}\")
    import traceback
    traceback.print_exc()
    exit(1)
"'

# Test 4: DDP multi-GPU training startup (verify LOCAL_RANK mapping)
echo ""
echo "=== Test 4: DDP LOCAL_RANK detection ==="
srun bash -c 'cd /home/student.aau.dk/fs62fb/projects/LuminaScale && python3 -c "
import os
rank = int(os.environ.get(\"LOCAL_RANK\", -1))
print(f\"Process {os.environ.get(\"SLURM_PROCID\")}: LOCAL_RANK={rank}, CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\")}\")
"'

echo ""
echo "=== All tests passed! ==="
