# GPU OpenGL on Multi-GPU HPC with Slurm: Implementation Guide

**Focus**: Hardware-accelerated GPU rendering in multi-GPU environments  
**Date**: April 5, 2026  
**Goal**: Get OCIO GPU transforms working on AAU AI Cloud with Slurm + multi-GPU (DDP)

---

## Problem Summary

✗ Current: `eglInitialize()` fails with `EGL_NOT_INITIALIZED` on HPC nodes  
✓ Solution: Proper EGL/GPU setup for headless multi-GPU rendering

---

## Key Findings from NVIDIA & Standards

### EGL (Khronos Standard): OpenGL Without X Server

NVIDIA Driver 355+ supports full desktop OpenGL on headless systems via EGL (Embedded Graphics Library).  
NVIDIA Driver 358+ adds **multi-GPU support**.

**Why your current code fails**:
```python
# This works on workstation but fails on HPC
display = eglGetDisplay(EGL_DEFAULT_DISPLAY)  # ← Returns NULL on headless nodes
eglInitialize(display, ...)                   # ← EGL_NOT_INITIALIZED error
```

---

## Core Solutions (GPU-Only)

### **Solution 1: Fix EGL_DEFAULT_DISPLAY (Multi-GPU Aware)**

**Root Issue**: `EGL_DEFAULT_DISPLAY` might not map to available GPU on headless nodes.

**Fix**: Use `eglGetPlatformDisplayEXT()` with explicit GPU device selection.

```python
# src/luminascale/utils/gpu_torch_processor.py

from OpenGL.EGL import (
    eglGetDisplay,
    eglQueryDevicesEXT,
    eglGetPlatformDisplayEXT,
    EGL_PLATFORM_DEVICE_EXT,
    # ... other EGL constants
)

def _initialize_egl_multigpu(self, gpu_id=None) -> None:
    """Initialize EGL with multi-GPU support."""
    
    # Step 1: Query available GPU devices
    try:
        eglQueryDevicesEXT = eglGetProcAddress("eglQueryDevicesEXT")
        MAX_DEVICES = 4
        egl_devices = (c_void_p * MAX_DEVICES)()
        num_devices = ctypes.c_int()
        
        if not eglQueryDevicesEXT(MAX_DEVICES, egl_devices, ctypes.byref(num_devices)):
            raise RuntimeError("eglQueryDevicesEXT failed")
        
        logger.info(f"EGL detected {num_devices.value} GPU devices")
        
        # Step 2: Select device (use GPU_ID if specified, else default)
        device_idx = gpu_id if gpu_id is not None else 0
        if device_idx >= num_devices.value:
            device_idx = 0
        
        logger.info(f"Using EGL device {device_idx}")
        
        # Step 3: Get platform display for selected device
        eglGetPlatformDisplayEXT = eglGetProcAddress("eglGetPlatformDisplayEXT")
        self._display = eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT,
            egl_devices[device_idx],
            0  # No attributes
        )
        
        if not self._display or self._display == -1:
            raise RuntimeError(f"eglGetPlatformDisplayEXT failed for device {device_idx}")
        
        # Step 4: Initialize EGL on this display
        major, minor = ctypes.c_int(), ctypes.c_int()
        if not eglInitialize(self._display, major, minor):
            raise RuntimeError(
                f"eglInitialize failed on device {device_idx}"
            )
        
        logger.info(f"EGL initialized on GPU {device_idx}: version {major.value}.{minor.value}")
        self._gl_ready = True
        
    except Exception as e:
        logger.error(f"Multi-GPU EGL initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize EGL: {e}")
```

### **Solution 2: Honor CUDA_VISIBLE_DEVICES in DDP**

When Slurm + PyTorch DDP runs, each process gets assigned GPUs via `CUDA_VISIBLE_DEVICES`.  
Extract device mapping and pass to EGL:

```python
# In dataset_pair_generator.py or TrainingModule.__init__()

import os

def get_local_gpu_id() -> int:
    """Extract local GPU ID from Slurm/DDP environment."""
    # PyTorch Lightning sets LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Sometimes also set by Slurm
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        device_ids = [int(x) for x in cuda_visible.split(",")]
        return device_ids[local_rank] if local_rank < len(device_ids) else local_rank
    
    return local_rank

# Initialize GPU processor with correct device
gpu_id = get_local_gpu_id()
self.ocio_processor = GPUTorchProcessor(headless=True, gpu_id=gpu_id)
```

### **Solution 3: Ensure libglvnd & Driver Support**

**Container/Singularity Setup**:

```singularity
# singularity/luminascale.def

Bootstrap: docker
From: pytorch/pytorch:2.1.0-devel-cuda12.1-cudnn8-devel

%post
    # Critical: Ensure NVIDIA EGL support libraries
    apt-get update && apt-get install -y \
        libglvnd0 \
        libglvnd-dev \
        libglx0 \
        glx-diversification-helper \
        libxext6 \
        libx11-6
    
    # Verify NVIDIA driver has EGL (check driver version >= 355)
    # This will be verified at runtime on compute node
    
    # Install OCIO and OpenGL Python bindings
    pip install --no-cache-dir \
        PyOpenColorIO \
        PyOpenGL \
        PyOpenGL_accelerate \
        OpenImageIO

%environment
    # Critical for EGL on multi-GPU
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export LIBGL_ALWAYS_INDIRECT=0  # Use direct rendering (GPU)

%runscript
    exec bash "$@"
```

### **Solution 4: Slurm Job Configuration**

```bash
#!/bin/bash
#SBATCH --job-name=train_multi_gpu
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules
module load python cuda

# Critical: Tell Slurm to allocate GPUs to tasks
export CUDA_VISIBLE_DEVICES=0,1

# Set OCIO config
export OCIO="${PWD}/config/aces/studio-config.ocio"

# Debug info
echo "GPU Info:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LOCAL_RANK will be set by PyTorch Lightning"

# Run training with DDP
python scripts/train_dequantization_net.py \
    --config-name hpc_slurm \
    strategy=ddp \
    devices=2 \
    accelerator=gpu \
    batch_size=16 \
    epochs=200 \
    output_dir=/lustre/scratch/fs62fb/outputs/training/$(date +%Y%m%d_%H%M%S)
```

---

## Diagnostic Script: Verify Multi-GPU EGL Works

**Create**: `opengl_research/test_egl_multigpu.py`

```python
#!/usr/bin/env python3
"""Test EGL multi-GPU support on HPC."""

import ctypes
import os
import sys
from ctypes import c_void_p

try:
    from OpenGL.EGL import (
        eglGetDisplay,
        eglInitialize,
        eglGetProcAddress,
        EGL_DEFAULT_DISPLAY,
        EGL_PLATFORM_DEVICE_EXT,
    )
    print("[✓] OpenGL.EGL imports OK")
except ImportError as e:
    print(f"[✗] OpenGL import failed: {e}")
    sys.exit(1)

print("\n=== GPU OpenGL Multi-GPU Diagnostic ===\n")

# Step 1: Check CUDA visibility
print("1. CUDA Setup:")
os.system("nvidia-smi --query-gpu=index,name,driver_version --format=csv")

cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible or 'unset (all GPUs)'}")

# Step 2: Query EGL devices
print("\n2. EGL Device Discovery:")
try:
    eglQueryDevicesEXT = eglGetProcAddress("eglQueryDevicesEXT")
    if not eglQueryDevicesEXT:
        print("   [✗] eglQueryDevicesEXT not available (old NVIDIA driver)")
        print("       Required: NVIDIA Driver >= 358")
        sys.exit(1)
    
    MAX_DEVICES = 4
    egl_devices = (c_void_p * MAX_DEVICES)()
    num_devices = ctypes.c_int()
    
    result = eglQueryDevicesEXT(MAX_DEVICES, egl_devices, ctypes.byref(num_devices))
    
    if result:
        print(f"   [✓] EGL detected {num_devices.value} devices")
    else:
        print(f"   [✗] eglQueryDevicesEXT returned False")
        sys.exit(1)
except Exception as e:
    print(f"   [✗] EGL device query failed: {e}")
    sys.exit(1)

# Step 3: Try to initialize EGL on each device
print("\n3. EGL Initialization per Device:")
for device_idx in range(num_devices.value):
    try:
        eglGetPlatformDisplayEXT = eglGetProcAddress("eglGetPlatformDisplayEXT")
        display = eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT,
            egl_devices[device_idx],
            0
        )
        
        if not display or display == -1:
            print(f"   Device {device_idx}: [✗] eglGetPlatformDisplayEXT failed")
            continue
        
        major, minor = ctypes.c_int(), ctypes.c_int()
        result = eglInitialize(display, ctypes.byref(major), ctypes.byref(minor))
        
        if result:
            print(f"   Device {device_idx}: [✓] EGL {major.value}.{minor.value} OK")
        else:
            print(f"   Device {device_idx}: [✗] eglInitialize failed")
    except Exception as e:
        print(f"   Device {device_idx}: [✗] Exception: {e}")

print("\n4. Summary:")
print("   If all devices show [✓], GPU OpenGL should work.")
print("   If failed, check:")
print("     - NVIDIA driver version (nvidia-smi --query-gpu=driver_version --format=csv)")
print("     - If driver < 355: Update required")
print("     - Contact HPC admin if driver is old")
```

**Run on compute node**:
```bash
srun -N 1 --gpus-per-node=2 python opengl_research/test_egl_multigpu.py
```

---

## Implementation Checklist

### Phase 1: Single GPU (Verify EGL Works)

- [ ] Run `test_egl_multigpu.py` on single GPU
  ```bash
  srun -N 1 --gpus-per-node=1 python opengl_research/test_egl_multigpu.py
  ```
  - If `[✓]` on all devices → proceed
  - If `[✗]` → contact HPC admin about driver version

- [ ] Update `gpu_torch_processor.py`
  - Add `_initialize_egl_multigpu()` method (code above)
  - Replace `_initialize_egl()` call in `__init__()`

- [ ] Test single-GPU training
  ```bash
  srun -N 1 --gpus-per-node=1 \
    python scripts/train_dequantization_net.py \
    --config-name default \
    devices=1 \
    accelerator=gpu
  ```
  - Expected: Training runs, GPU OCIO transforms work
  - Monitor: `nvidia-smi` shows GPU with OpenGL context

### Phase 2: Multi-GPU with DDP

- [ ] Update `dataset_pair_generator.py` or `dequantization_trainer.py`
  - Add `get_local_gpu_id()` function (code above)
  - Pass `gpu_id` to `GPUTorchProcessor(gpu_id=gpu_id)`

- [ ] Test 2-GPU training
  ```bash
  sbatch scripts/train_hpc_multigpu.sh  # See template below
  ```
  - Expected: Both GPUs utilized, OCIO transforms on each GPU
  - Monitor logs: `GLOBAL_RANK: 0/2` and `GLOBAL_RANK: 1/2`

- [ ] Verify scaling efficiency
  - Single GPU: T seconds/batch
  - Two GPUs: Should be ~1.8–1.9× faster (close to linear)

---

## Slurm Batch Template

**File**: `scripts/train_hpc_multigpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=train_luminascale_multigpu
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# Load modules
module load python cuda

# Activate venv
source /path/to/venv/bin/activate

# Set environment
export PYTHONUNBUFFERED=1
export OCIO="${PWD}/config/aces/studio-config.ocio"
export CUDA_VISIBLE_DEVICES=0,1

echo "=== Multi-GPU Training with EGL ====="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run diagnostic first
python opengl_research/test_egl_multigpu.py

echo -e "\n=== Starting Training ====="

# Train with DDP
python scripts/train_dequantization_net.py \
    --config-name hpc_slurm \
    strategy=ddp \
    devices=2 \
    accelerator=gpu \
    batch_size=16 \
    epochs=200 \
    num_workers=8 \
    lmdb_path=/lustre/scratch/fs62fb/dataset/training_data.lmdb \
    output_dir=/lustre/scratch/fs62fb/outputs/training/$(date +%Y%m%d_%H%M%S)

echo "Training complete."
```

---

## Expected Behavior

### Success (Everything Works)

```
=== Multi-GPU Training with EGL ===
[✓] OpenGL.EGL imports OK
1. CUDA Setup:
   | 0 | GPU Model | 550  |  ← Driver 550 (>= 355) ✓
   | 1 | GPU Model | 550  |
   CUDA_VISIBLE_DEVICES: 0,1

2. EGL Device Discovery:
   [✓] EGL detected 2 devices

3. EGL Initialization per Device:
   Device 0: [✓] EGL 1.5 OK
   Device 1: [✓] EGL 1.5 OK

=== Starting Training =====
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
Local rank: 0 → EGL GPU 0
Local rank: 1 → EGL GPU 1
GPU OCIO processor initialized on GPU 0
GPU OCIO processor initialized on GPU 1
Batch 1: 0.15 sec | GPU 0: 95% | GPU 1: 95%  ← Both GPUs active
```

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `EGL_NOT_INITIALIZED` | Old NVIDIA driver (< 355) | Query admin: update driver to 550+ |
| `eglQueryDevicesEXT not available` | Driver < 358 (no multi-GPU EGL) | Update driver |
| Only GPU 0 active | `CUDA_VISIBLE_DEVICES` wrong | Verify Slurm job shows 2 GPUs |
| Per-GPU context crash | EGL device mapping mismatch | Check `LOCAL_RANK` vs `gpu_id` |

---

## Files to Modify

| File | Change | Time |
|------|--------|------|
| `src/luminascale/utils/gpu_torch_processor.py` | Add `_initialize_egl_multigpu()` | 45 min |
| `src/luminascale/training/dataset_pair_generator.py` | Add `get_local_gpu_id()` and pass to GPU processor | 30 min |
| `singularity/luminascale.def` | Add libglvnd + EGL libraries | 15 min |
| `scripts/train_hpc_multigpu.sh` | New Slurm batch script | 20 min |
| `opengl_research/test_egl_multigpu.py` | New diagnostic script | Already provided |

**Total effort**: ~2–2.5 hours implementation + testing

---

## References

- **NVIDIA EGL Blog**: https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
- **EGL Specification**: https://www.khronos.org/egl
- **Slurm + GPU**: https://slurm.schedmd.com/gres.html
- **PyTorch DDP**: https://pytorch.org/docs/stable/notes/distributed.html

---

## Next Steps

1. **NOW**: Run `test_egl_multigpu.py` diagnostic on HPC → confirm driver support
2. **TODAY**: Implement `_initialize_egl_multigpu()` in gpu_torch_processor.py
3. **TODAY**: Add `get_local_gpu_id()` to dataset_pair_generator.py
4. **TOMORROW**: Test single-GPU training
5. **TOMORROW**: Test multi-GPU (2 GPU) training with DDP
6. **VERIFY**: GPU utilization ~95% on both GPUs during batch

