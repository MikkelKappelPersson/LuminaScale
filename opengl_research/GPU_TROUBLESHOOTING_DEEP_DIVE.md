# GPU OpenGL on HPC: Deep Dive & Troubleshooting

---

## Part 1: Why EGL_DEFAULT_DISPLAY Fails on HPC (Root Cause)

### The Problem Flow

```
Compute Node (Headless)
    ├─ GPUs installed: Yes (nvidia-smi shows them)
    ├─ X11 server: No
    ├─ Display manager: No
    └─ /dev/dri/: Exists, but not exposed to some processes
    
When Python calls eglGetDisplay(EGL_DEFAULT_DISPLAY):
    ├─ Queries X11 connection → NULL (no X server)
    ├─ Queries Wayland → NULL (no Wayland)
    └─ Tries platform default → Returns invalid handle
        └─ eglInitialize() then fails with EGL_NOT_INITIALIZED
```

### Why Your Current Code Receives NULL

**File**: `src/luminascale/utils/gpu_torch_processor.py` (line ~168)

```python
# WRONG on HPC (doesn't specify device)
eglBindAPI(EGL_OPENGL_API)
self._display = eglGetDisplay(EGL_DEFAULT_DISPLAY)  # ← Returns -1 on headless
if not eglInitialize(self._display, ...):
    raise RuntimeError("EGL_NOT_INITIALIZED")
```

**Why it fails**:
- `EGL_DEFAULT_DISPLAY` assumes X11 or display driver at index 0
- On HPC: GPU 0 might not be the "default" display
- In multi-GPU + DDP: Each process scrambles for same display handle

### The Fix: Explicit GPU Device Selection

```python
# CORRECT on HPC (specifies GPU explicitly)
eglQueryDevicesEXT()           # ← Find available GPU devices
eglGetPlatformDisplayEXT(      # ← Ask for display from GPU N
    EGL_PLATFORM_DEVICE_EXT,
    egl_device[0],             # ← GPU 0 for process rank 0
    0
)
eglInitialize(display, ...)    # ← Success: Uses GPU 0's EGL context
```

---

## Part 2: Multi-GPU DDP Challenges

### Challenge 1: GPU Assignment in DDP

**Scenario**: `sbatch` job with `--gpus-per-node=2`

```
SLURM allocates GPUs [0, 1] to job
  │
  ├─ PyTorch Lightning DDP spawns 2 processes:
  │   ├─ Process A: LOCAL_RANK=0 → Should use GPU 0
  │   ├─ Process B: LOCAL_RANK=1 → Should use GPU 1
  │
  Problem: Both processes call eglGetDisplay(EGL_DEFAULT_DISPLAY)
           Both might get same display handle or invalid handles
           Or eglGetDisplay() doesn't respect CUDA_VISIBLE_DEVICES
```

**Solution**: Pass explicit `gpu_id` to each process

```python
# Get your process's GPU ID from environment
gpu_id = int(os.environ.get("LOCAL_RANK", 0))

# Tell EGL to use that specific GPU
processor = GPUTorchProcessor(headless=True, gpu_id=gpu_id)
```

### Challenge 2: Container (Singularity) GPU Exposure

**Problem**: Singularity container might not expose `/dev/dri/` or GPU drivers.

**Solution**: Use `--nv` flag + bind `/dev/dri/`

```bash
# WRONG: GPU driver not visible inside container
srun singularity exec luminascale.sif python train.py

# CORRECT: Expose NVIDIA drivers + DRI devices
srun singularity exec --nv --bind /dev/dri:/dev/dri luminascale.sif python train.py
```

### Challenge 3: EGL Context Per-GPU in Shared Memory

With DDP, each process runs on a separate GPU but shares training model via NCCL.

```
Process 0 (GPU 0)                Process 1 (GPU 1)
├─ EGL context → GPU 0           ├─ EGL context → GPU 1
├─ OCIO transform on data        ├─ OCIO transform on data
├─ Model copy (via NCCL)  ←→  ← Model sync (different GPU)
├─ Local update via   ←→ ← Back-sync gradients
└─ Combined gradient update
```

This is **already handled by PyTorch** if you initialize each EGL context on the correct GPU.

---

## Part 3: EGL Configuration Details

### EGL Pbuffer (Off-Screen Rendering)

```python
# Current implementation uses pbuffer (correct for HPC)
surface_attribs = [
    12344,  # EGL_WIDTH
    1024,
    12345,  # EGL_HEIGHT
    1024,
    EGL_NONE,
]
self._surface = eglCreatePbufferSurface(
    self._display, config, surface_attribs_array
)

# Pbuffer = off-screen framebuffer (no window needed)
# Perfect for HPC (compute-only, no display)
```

### EGL Configuration Attributes

```python
config_attribs = [
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,      # Off-screen rendering
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,    # Desktop OpenGL
    EGL_RED_SIZE, 8,                        # 8-bit color channel
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 24,                     # 24-bit depth buffer
    EGL_NONE,
]
```

---

## Part 4: NVIDIA Driver Support Matrix

| Driver | EGL | Multi-GPU EGL | OpenGL | Notes |
|--------|-----|---|---|---|
| < 355 | ❌ No | – | – | Too old, update required |
| 355–357 | ✅ Yes (ES only) | ❌ No | ❌ Limited | OpenGL ES only |
| 358–499 | ✅ Yes | ✅ Yes | ✅ Full | Desktop OpenGL + multi-GPU |
| 500+ | ✅ Yes | ✅ Yes | ✅ Full | Latest (recommended) |

**Check your driver**:
```bash
nvidia-smi --query-gpu=driver_version --format=csv
# Expected: 500+ for AAU AI Cloud (if not, contact admin)
```

---

## Part 5: Debugging Multi-GPU EGL

### Diagnostic 1: Device Discovery

```python
# Verify EGL sees all GPUs
from OpenGL.EGL import eglGetProcAddress

eglQueryDevicesEXT = eglGetProcAddress("eglQueryDevicesEXT")
MAX_DEVICES = 4
egl_devices = (c_void_p * MAX_DEVICES)()
num_devices = ctypes.c_int()

result = eglQueryDevicesEXT(MAX_DEVICES, egl_devices, ctypes.byref(num_devices))
print(f"EGL sees {num_devices.value} devices")
# Expected: 2+ (based on SLURM allocation)
```

### Diagnostic 2: Per-Device Initialization

```python
# Test initialization on each GPU
for device_idx in range(num_devices.value):
    eglGetPlatformDisplayEXT = eglGetProcAddress("eglGetPlatformDisplayEXT")
    
    display = eglGetPlatformDisplayEXT(
        EGL_PLATFORM_DEVICE_EXT,
        egl_devices[device_idx],
        0
    )
    
    if display and display != -1:
        major, minor = c_int(), c_int()
        if eglInitialize(display, byref(major), byref(minor)):
            print(f"GPU {device_idx}: ✓ EGL {major.value}.{minor.value}")
        else:
            print(f"GPU {device_idx}: ✗ eglInitialize failed")
    else:
        print(f"GPU {device_idx}: ✗ eglGetPlatformDisplayEXT failed")
```

### Diagnostic 3: GPU Memory vs Compute Overlap

```bash
# While training is running, check both GPUs active
watch nvidia-smi
# Should see: Memory ▊▊▊▊▊▊▊  GPU ▊▊▊▊▊▊▊  (both high)
```

### Diagnostic 4: EGL Context Binding

```python
# After eglMakeCurrent(), verify context is active
import subprocess
result = subprocess.run(["glxinfo"], capture_output=True, text=True)
if "OpenGL version" in result.stdout:
    print("✓ OpenGL context active")
else:
    print("✗ OpenGL context not working")
```

---

## Part 6: Performance Tuning for Multi-GPU

### GPU Utilization Target

**Ideal**: ~95% GPU utilization on both GPUs, batch time ~0.1–0.3 sec

**Achievable if**:
- ✅ EGL initialized on correct GPU per process
- ✅ OCIO transforms stay on GPU (no CPU-GPU sync)
- ✅ NCCL communication latency < 5 ms
- ✅ DataLoader workers >= 4 per GPU

### NCCL Tuning (Optional)

```bash
# Enable NCCL debug output
export NCCL_DEBUG=TRACE

# python train.py ...
# Look for: "nccl: allreduce completed in X ms"
# Should be < 5ms for 2 GPUs on same node
```

### Batch Size Scaling

| Batch Size | GPU 0 Memory | GPU 1 Memory | Typical Time |
|---|---|---|---|
| 8 | 3 GB | 3 GB | 0.08 sec |
| 16 | 6 GB | 6 GB | 0.15 sec |
| 32 | 12 GB | 12 GB | 0.28 sec |

Choose batch size for 80–85% GPU memory utilization.

---

## Part 7: Singularity Container Configuration

### Required Libraries in Container

```singularity
%post
    # NVIDIA EGL libraries
    apt-get update && apt-get install -y \
        libglvnd0 \
        libglvnd-dev \
        libglx0 \
        libglu1-mesa \
        libxext6 \
        libx11-6 \
        libxcb1
    
    # OCIO + OpenGL
    pip install --no-cache-dir \
        PyOpenColorIO \
        PyOpenGL \
        PyOpenGL_accelerate
```

### Singularity Run Flags

```bash
# Minimal (if /dev/dri exposed by default)
singularity exec --nv container.sif python script.py

# Safe (explicit GPU + device passthrough)
singularity exec --nv \
    --bind /dev/dri:/dev/dri \
    --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    container.sif python script.py
```

---

## Part 8: Common Failure Modes & Recovery

### Failure: "eglInitialize returned False"

**Cause**: Display driver not ready or GPU unavailable  
**Recovery**:
```bash
# Check GPU physically available
nvidia-smi
# If GPU listed but error persists:
#   1. Restart GPU driver (admin action)
#   2. Try different GPU (--gpu-rank-mapping)
#   3. Check Slurm allocation: squeue --me
```

### Failure: "EGL_BAD_DISPLAY"

**Cause**: Invalid display handle (device not found)  
**Recovery**:
```python
# In test_egl_multigpu.py, check device count matches GPU count
print(f"EGL devices: {num_devices}")
print(f"CUDA GPUs: run nvidia-smi | grep GPU")
# Should match (or EGL detects fewer)
```

### Failure: Only 1 GPU Active During Training

**Cause**: DDP not properly synchronized or EGL on wrong device  
**Recovery**:
```bash
# Check Slurm allocation
squeue -j $SLURM_JOB_ID
# Should show: 2 tasks (one per GPU)

# Check PyTorch Lightning logs
grep "GLOBAL_RANK\|LOCAL_RANK" logs/*.err
# Should show: 0/2, 1/2
```

### Failure: Training Crash After 10 Batches (Random)

**Cause**: Shared OpenGL resource conflict between processes  
**Recovery**:
```python
# In gpu_torch_processor.py, add proper resource cleanup
def __del__(self):
    if self._framebuffer:
        GL.glDeleteFramebuffers(1, [self._framebuffer])
    if self._vao:
        GL.glDeleteVertexArrays(1, [self._vao])
    # Ensure EGL context switched before deletion
```

---

## Part 9: Validation Checklist

Before Production:

- [ ] `test_egl_multigpu.py` shows ✓ on all devices
- [ ] NVIDIA driver >= 358 (check: `nvidia-smi`)
- [ ] Single-GPU training: GPU 100% utilized, batch time < 0.2 sec
- [ ] Multi-GPU training: Both GPUs 95%+ utilized
- [ ] Multi-GPU speedup ~1.8–1.9× (close to 2×)
- [ ] Training converges smoothly (no crashes after 100 batches)
- [ ] NCCL communication time < 5 ms (if using TRACE debug)

---

## References

- **NVIDIA EGL Reference**: https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
- **Khronos EGL Spec**: https://www.khronos.org/egl/
- **PyOpenColorIO GPU**: https://opencolorio.org/usageexamples/basic.html (GPU Shaders section)
- **PyTorch DDP**: https://pytorch.org/docs/stable/notes/distributed.html
- **Slurm GPU Allocation**: https://slurm.schedmd.com/gres.html

