# GPU OpenGL Multi-GPU HPC: Quick Start (3-3.5 Hours to Training)

**Current Status**: GPU-only solution research complete  
**Focus**: Hardware-accelerated GPU rendering via EGL on Slurm multi-GPU jobs

---

## 🎯 What You Need to Do

### Phase 1: Verify HPC GPU OpenGL Support (15 min)

```bash
srun -N 1 --gpus-per-node=2 python opengl_research/test_egl_multigpu.py
```

**Expected Success**:
```
✓ EGL detected 2 devices
Device 0: [✓] EGL 1.5 OK
Device 1: [✓] EGL 1.5 OK
```

**If it fails**: Contact HPC admin – "NVIDIA driver too old (need >= 358)"

---

### Phase 2: Implement EGL Multi-GPU (2–2.5 hours)

#### 2a. Multi-GPU EGL Initialization (45 min)
**File**: `src/luminascale/utils/gpu_torch_processor.py`

**Change**: Add this method:
```python
def _initialize_egl_multigpu(self, gpu_id=None) -> None:
    """Initialize EGL with explicit GPU device selection."""
    
    # Step 1: Query EGL devices
    eglQueryDevicesEXT = eglGetProcAddress("eglQueryDevicesEXT")
    MAX_DEVICES = 4
    egl_devices = (c_void_p * MAX_DEVICES)()
    num_devices = ctypes.c_int()
    eglQueryDevicesEXT(MAX_DEVICES, egl_devices, ctypes.byref(num_devices))
    
    # Step 2: Select device
    device_idx = gpu_id if gpu_id is not None else 0
    
    # Step 3: Get platform display for specific device
    eglGetPlatformDisplayEXT = eglGetProcAddress("eglGetPlatformDisplayEXT")
    self._display = eglGetPlatformDisplayEXT(
        EGL_PLATFORM_DEVICE_EXT,
        egl_devices[device_idx],
        0
    )
    
    # Step 4: Initialize
    major, minor = ctypes.c_int(), ctypes.c_int()
    if not eglInitialize(self._display, major, minor):
        raise RuntimeError(f"eglInitialize failed on device {device_idx}")
    
    logger.info(f"EGL GPU {device_idx}: version {major.value}.{minor.value}")
    self._gl_ready = True
```

**Also update `__init__`**:
```python
def __init__(self, headless: bool = True, gpu_id: int = None) -> None:
    # ... existing code ...
    self._target_gpu_id = gpu_id
    # ... replace _initialize_egl() call with:
    self._initialize_egl_multigpu(self._target_gpu_id)
```

👉 Full code: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md) Solution 1

---

#### 2b. DDP GPU ID Mapping (30 min)
**File**: `src/luminascale/training/dataset_pair_generator.py`

**Add this function**:
```python
def get_local_gpu_id() -> int:
    """Extract local GPU ID from DDP environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        device_ids = [int(x) for x in cuda_visible.split(",")]
        return device_ids[local_rank] if local_rank < len(device_ids) else local_rank
    return local_rank
```

**In `__init__`, use it**:
```python
gpu_id = get_local_gpu_id()
self.ocio_processor = GPUTorchProcessor(headless=True, gpu_id=gpu_id)
```

👉 Full code: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md) Solution 2

---

#### 2c. Update Singularity (15 min)
**File**: `singularity/luminascale.def`

**Add to `%post` section**:
```singularity
# EGL libraries (GPU OpenGL support)
apt-get update && apt-get install -y \
    libglvnd0 \
    libglvnd-dev \
    libglx0 \
    libx11-6 \
    libxext6

# OCIO + OpenGL Python
pip install --no-cache-dir \
    PyOpenColorIO \
    PyOpenGL \
    PyOpenGL_accelerate
```

👉 Full code: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md) Solution 3

---

#### 2d. Create Slurm Batch Script (20 min)
**File**: `scripts/train_hpc_multigpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=train_gpu2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=gpu

module load python cuda
source /path/to/venv/bin/activate

export PYTHONUNBUFFERED=1
export OCIO="${PWD}/config/aces/studio-config.ocio"
export CUDA_VISIBLE_DEVICES=0,1

# Verify EGL setup
python opengl_research/test_egl_multigpu.py

# Train with DDP
python scripts/train_dequantization_net.py \
    --config-name hpc_slurm \
    strategy=ddp \
    devices=2 \
    accelerator=gpu \
    batch_size=16 \
    epochs=200 \
    output_dir=/lustre/scratch/fs62fb/outputs/training/$(date +%Y%m%d_%H%M%S)
```

👉 Full code: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md)

---

### Phase 3: Test Single GPU First (30 min)

```bash
srun -N 1 --gpus-per-node=1 \
  python scripts/train_dequantization_net.py \
  --config-name default \
  devices=1 \
  accelerator=gpu \
  epochs=1 \
  batch_size=8

# Expected: 100% GPU, ~0.15 sec/batch, no EGL errors
```

---

### Phase 4: Test Multi-GPU DDP (30-45 min)

```bash
sbatch scripts/train_hpc_multigpu.sh

# Monitor
tail -f logs/train_*.err | grep -i "rank\|gpu\|ocio\|batch"

# Expected output:
# GLOBAL_RANK: 0/2  (Process A on GPU 0)
# GLOBAL_RANK: 1/2  (Process B on GPU 1)
# Batch 1: 0.15 sec | GPU 0: 99% | GPU 1: 99%  ← Both active!
```

---

## ✅ Success Metrics

| Metric | Target |
|--------|--------|
| EGL devices detected | 2 (or your GPU count) |
| Single GPU batch time | < 0.2 sec |
| Multi-GPU speedup | 1.8–1.9× (90% efficient) |
| Both GPUs utilized | 95%+ |

---

## 📚 Complete Documentation

**Implementation Guide**: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md)  
**Troubleshooting**: [GPU_TROUBLESHOOTING_DEEP_DIVE.md](GPU_TROUBLESHOOTING_DEEP_DIVE.md)  
**Diagnostic Tool**: `test_egl_multigpu.py`

---

## 🔧 If Anything Fails

| Error | Check |
|-------|-------|
| `eglQueryDevicesEXT not found` | Driver < 358 (old) – contact admin |
| `eglInitialize failed` | See: GPU_TROUBLESHOOTING_DEEP_DIVE.md Part 1 |
| Only 1 GPU active | See: GPU_TROUBLESHOOTING_DEEP_DIVE.md Part 8 |
| Container can't see GPU | See: GPU_TROUBLESHOOTING_DEEP_DIVE.md Part 7 |

---

## Timeline Summary

```
NOW (15 min)     → Run diagnostic
├─ TODAY (2h)    → Implement EGL multi-GPU + DDP changes
├─ TODAY (30m)   → Update Singularity + create batch script
├─ TODAY (1h)    → Test single-GPU
├─ TOMORROW (1h) → Test multi-GPU with DDP
└─ SUCCESS ✓     → Multi-GPU training ready
```

**Total: 3–3.5 hours + testing**

---

## Next Action

```bash
# 1. Run diagnostic NOW
srun -N 1 --gpus-per-node=2 python opengl_research/test_egl_multigpu.py

# 2. If ✓ on all devices → proceed with implementation (2-2.5h)
# 3. If ✗ → contact HPC admin about driver
```

---

## Referenced Documents (All GPU-Only Solutions)

- ✅ `GPU_OPENGL_MULTIGPU_HPC.md` – Main implementation guide
- ✅ `GPU_TROUBLESHOOTING_DEEP_DIVE.md` – Debugging + tuning
- ✅ `test_egl_multi gpu.py` – HPC diagnostic script
- ✅ `README.md` – Quick reference

**Note**: Older documents (HPC_OCIO_RESEARCH.md, etc.) contained CPU-only fallback approaches which are no longer the focus. New approach is GPU-only with explicit EGL multi-GPU setup.

---

**Status**: Ready to implement. Start with diagnostic script. 🚀

