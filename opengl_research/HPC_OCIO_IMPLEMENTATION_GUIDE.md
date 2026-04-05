# HPC OCIO Implementation Guide: Code Examples & Diagnostics

**Date**: April 5, 2026  
**Related**: [HPC_OCIO_RESEARCH.md](HPC_OCIO_RESEARCH.md)

---

## Part 1: HPC Environment Diagnostic Checklist

Run this on an AAU AI Cloud compute node (via `srun` or `salloc`):

### 1.1 Check Display Drivers

```bash
# Step 1: Verify GPU is visible
nvidia-smi
# Expected: Lists GPU(s) with CUDA version
# If fails: NVIDIA driver not installed in container

# Step 2: Check for DRM devices (display drivers)
ls -la /dev/dri/
# Expected: card0, render128, etc. (GPU render devices)
# If fails/empty: Display drivers not exposed to container

# Step 3: Check EGL library availability
ldconfig -p | grep EGL
# Expected: libEGL.so (NVIDIA or Mesa)
# If fails: EGL not installed or not in library path

# Step 4: Query NVIDIA driver capabilities
nvidia-smi --query-gpu=driver_version,compute_cap --format=csv
# Expected: Driver version and compute capability
```

### 1.2 Test EGL Directly

```bash
# Create test script: test_egl.py
cat > test_egl.py << 'EOF'
#!/usr/bin/env python3
import sys
try:
    from OpenGL.EGL import eglGetDisplay, eglInitialize, EGL_DEFAULT_DISPLAY
    print("[✓] OpenGL.EGL imports OK")
    
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    if display == -1:
        print("[✗] eglGetDisplay(EGL_DEFAULT_DISPLAY) returned -1")
        sys.exit(1)
    print(f"[✓] eglGetDisplay OK: {hex(id(display))}")
    
    # Try initialize
    try:
        major, minor = 0, 0
        result = eglInitialize(display, ctypes.byref(ctypes.c_int(major)), ctypes.byref(ctypes.c_int(minor)))
        if result:
            print(f"[✓] eglInitialize OK: EGL {major}.{minor}")
        else:
            print("[✗] eglInitialize returned False (no EGL drivers available)")
    except Exception as e:
        print(f"[✗] eglInitialize failed: {e}")
        sys.exit(1)
        
except ImportError as e:
    print(f"[✗] OpenGL import failed: {e}")
    sys.exit(1)
EOF

# Run on compute node
srun -N 1 --gpus-per-node=1 python test_egl.py
```

### 1.3 Check Singularity Container Bindings

```bash
# Inside Singularity container (if using one):
# Check if /dev/dri is mounted
cat /proc/self/mountinfo | grep dri
# Expected: entry for /dev/dri*
# If missing: Container needs `--nv` flag or bind mount

# Check if NVIDIA libs are visible
ls -la /usr/local/cuda/lib64/libEGL*
# Expected: EGL libraries present
```

### 1.4 Quick Test: Can We Use CPU OCIO?

```bash
# Test CPU OCIO independently
cat > test_cpu_ocio.py << 'EOF'
#!/usr/bin/env python3
import os
import PyOpenColorIO as ocio

# Set OCIO config
os.environ["OCIO"] = "/path/to/config/aces/studio-config.ocio"

try:
    config = ocio.Config.CreateFromFile(os.environ["OCIO"])
    print(f"[✓] OCIO config loaded: {config.getName()}")
    
    # Try a simple CPU transform
    processor = config.getProcessor("ACES2065-1", "sRGB - Display")
    print("[✓] CPU processor created (no GPU needed)")
    
except Exception as e:
    print(f"[✗] OCIO failed: {e}")
EOF

srun -N 1 python test_cpu_ocio.py
```

---

## Part 2: Implementation – Approach B (CPU Fallback)

### 2.1 Modify dataset_pair_generator.py

**File**: `src/luminascale/training/dataset_pair_generator.py`

```python
# BEFORE (line ~45-55):
class DatasetPairGenerator:
    def __init__(self, env, device, keys):
        self.keys = keys
        self.device = device
        self.env = env
        
        # Always creates GPU processor; crashes if EGL fails
        self.ocio_processor = GPUTorchProcessor(headless=True)
        self.cdl_processor = CDLProcessor()


# AFTER (with graceful fallback):
class DatasetPairGenerator:
    def __init__(self, env, device, keys):
        self.keys = keys
        self.device = device
        self.env = env
        
        # Try GPU OCIO; fall back to CPU if unavailable
        self.ocio_processor = None
        self.use_gpu_ocio = False
        
        try:
            self.ocio_processor = GPUTorchProcessor(headless=True)
            self.use_gpu_ocio = True
            logger.info("GPU OCIO processor initialized")
        except RuntimeError as e:
            logger.warning(
                f"Failed to initialize GPU OCIO: {e}. "
                "Falling back to CPU OCIO. Per-image transforms will be slower."
            )
            self.ocio_processor = None
            self.use_gpu_ocio = False
        
        self.cdl_processor = CDLProcessor()
    
    def _apply_ocio_transform(self, aces_array):
        """Apply ACES→sRGB transform via GPU or CPU."""
        if self.use_gpu_ocio and self.ocio_processor is not None:
            # GPU path: tensor in, tensor out
            aces_tensor = torch.from_numpy(aces_array).to(self.device)
            _, srgb_tensor = self.ocio_processor.apply_ocio_torch(
                aces_tensor,
                input_cs="ACES2065-1",
                display="sRGB - Display"
            )
            return srgb_tensor.cpu().numpy()
        else:
            # CPU path: use OpenImageIO OCIO
            # Save to temp EXR, transform via CPU OCIO
            temp_exr = f"/tmp/temp_aces_{uuid.uuid4().hex}.exr"
            write_exr(temp_exr, aces_array)
            
            srgb_array = oiio_aces_to_display(temp_exr)
            os.remove(temp_exr)
            return srgb_array
```

### 2.2 Modify io.py – Enhance CPU Fallback

**File**: `src/luminascale/utils/io.py`

```python
# Add caching layer for CPU OCIO (Lines ~150-200):

from functools import lru_cache
import hashlib

class ACESTransformCache:
    """LRU cache for CPU OCIO transforms to reduce per-image overhead."""
    
    def __init__(self, max_cache_size=1000):
        self.max_size = max_cache_size
        self._cache = {}
        self._access_times = {}
    
    def _hash_array(self, arr: np.ndarray) -> str:
        """Create hash of array content."""
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    def get(self, aces_array: np.ndarray) -> Optional[np.ndarray]:
        """Check cache for ACES→sRGB result."""
        key = self._hash_array(aces_array)
        if key in self._cache:
            return self._cache[key].copy()
        return None
    
    def put(self, aces_array: np.ndarray, srgb_array: np.ndarray) -> None:
        """Cache ACES→sRGB result."""
        if len(self._cache) >= self.max_size:
            # Evict least recently accessed
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        key = self._hash_array(aces_array)
        self._cache[key] = srgb_array.copy()
        self._access_times[key] = time.time()


# Global cache instance
_aces_transform_cache = ACESTransformCache(max_cache_size=500)


def oiio_aces_to_display_cached(aces_array: np.ndarray) -> np.ndarray:
    """CPU OCIO transform with caching."""
    # Check cache first
    cached = _aces_transform_cache.get(aces_array)
    if cached is not None:
        logger.debug("OCIO transform cache hit")
        return cached
    
    # Perform transform
    temp_exr = f"/tmp/aces_{uuid.uuid4().hex}.exr"
    try:
        write_exr(temp_exr, aces_array)
        srgb = oiio_aces_to_display(temp_exr)
        _aces_transform_cache.put(aces_array, srgb)
        return srgb
    finally:
        if os.path.exists(temp_exr):
            os.remove(temp_exr)
```

### 2.3 Update gpu_torch_processor.py – Better Error Handling

**File**: `src/luminascale/utils/gpu_torch_processor.py` (lines ~160-180)

```python
# BEFORE:
def _initialize_egl(self) -> None:
    """Initialize EGL context for headless rendering."""
    try:
        eglBindAPI(EGL_OPENGL_API)
        self._display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        major, minor = ctypes.c_int(), ctypes.c_int()
        if not eglInitialize(self._display, major, minor):
            raise RuntimeError("Standard eglInitialize returned False")
        # ... rest of init
    except Exception as e:
        logger.error(f"EGL initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize EGL: {e}")
        

# AFTER (better diagnostics):
def _initialize_egl(self) -> None:
    """Initialize EGL context for headless rendering."""
    diagnostics = {
        "display_drivers": self._check_display_drivers(),
        "egl_available": self._check_egl_available(),
        "nvidia_driver": self._check_nvidia_driver(),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    
    try:
        eglBindAPI(EGL_OPENGL_API)
        self._display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        
        if not self._display or self._display == -1:
            msg = (
                f"eglGetDisplay failed. Diagnostics:\n{diagnostics}\n"
                "This HPC environment may not support GPU OpenGL rendering. "
                "Use CPU OCIO fallback instead."
            )
            raise RuntimeError(msg)
        
        major, minor = ctypes.c_int(), ctypes.c_int()
        if not eglInitialize(self._display, major, minor):
            msg = (
                f"eglInitialize failed (EGL_NOT_INITIALIZED). Diagnostics:\n{diagnostics}\n"
                "GPUs visible but EGL drivers not initialized. "
                "Try: (1) Check with HPC admin about EGL_MESA_SURFACELESS support, "
                "(2) Use CPU OCIO fallback via --disable-gpu-ocio"
            )
            raise RuntimeError(msg)
        
        logger.info(f"EGL initialized: version {major.value}.{minor.value}")
        self._gl_ready = True
        
    except Exception as e:
        logger.error(f"GPU OCIO initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize EGL: {e}")

def _check_display_drivers(self) -> bool:
    """Check if /dev/dri exists."""
    return os.path.exists("/dev/dri")

def _check_egl_available(self) -> bool:
    """Check if EGL libraries can be loaded."""
    try:
        import OpenGL.EGL
        return True
    except ImportError:
        return False

def _check_nvidia_driver(self) -> Optional[str]:
    """Check NVIDIA driver version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except:
        return None
```

---

## Part 3: Implementation – Approach A (Pre-Baking)

### 3.1 Enhanced bake_dataset.py

**File**: `scripts/bake_dataset.py`

```python
# Add GPU/CPU fallback logic (lines ~80-120):

def bake_aces_to_display(aces_dir, output_dir, use_gpu=True, num_workers=4):
    """
    Bake ACES EXRs to sRGB PNGs.
    
    Args:
        aces_dir: Directory of ACES2065-1 EXR files
        output_dir: Output directory for sRGB PNGs
        use_gpu: Try GPU OCIO first; fall back to CPU if not available
        num_workers: Number of CPU workers for parallel processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to initialize GPU processor once
    gpu_processor = None
    use_gpu_ocio = False
    
    if use_gpu:
        try:
            gpu_processor = GPUTorchProcessor(headless=True)
            use_gpu_ocio = True
            logger.info("GPU OCIO initialized; using GPU transforms")
        except RuntimeError as e:
            logger.warning(
                f"GPU OCIO unavailable: {e}\n"
                "Falling back to CPU OCIO (slower but works everywhere)"
            )
            use_gpu_ocio = False
    
    # List all ACES files
    aces_files = sorted(glob.glob(f"{aces_dir}/**/*.exr", recursive=True))
    logger.info(f"Found {len(aces_files)} ACES EXR files")
    
    # Process each file
    for i, aces_path in enumerate(aces_files):
        output_path = os.path.join(output_dir, f"{Path(aces_path).stem}.png")
        
        try:
            # Load ACES EXR
            aces_array = read_exr(aces_path)
            
            # Transform ACES → sRGB
            if use_gpu_ocio:
                aces_tensor = torch.from_numpy(aces_array).cuda()
                _, srgb_uint8 = gpu_processor.apply_ocio_torch(
                    aces_tensor,
                    input_cs="ACES2065-1",
                    display="sRGB - Display"
                )
                srgb_array = srgb_uint8.cpu().numpy()
            else:
                # CPU path
                srgb_array = oiio_aces_to_display(aces_path)
            
            # Save as PNG
            Image.fromarray(srgb_array, mode="RGB").save(output_path)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(aces_files)} files")
        
        except Exception as e:
            logger.error(f"Failed to process {aces_path}: {e}")
            continue
    
    logger.info(f"Baking complete! Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Path to ACES EXR directory")
    parser.add_argument("--output-dir", required=True, help="Path to output sRGB PNG directory")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Try GPU first")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode")
    args = parser.parse_args()
    
    bake_aces_to_display(
        args.input_dir,
        args.output_dir,
        use_gpu=not args.cpu_only and args.use_gpu
    )
```

### 3.2 Add Pre-Bake Mode to Training Config

**File**: `configs/hpc_slurm.yaml`

```yaml
defaults:
  - default

# Pre-baked dataset mode (True = use pre-baked sRGB, False = on-the-fly GPU/CPU)
use_prebaked_dataset: true
prebaked_dir: ${oc.env:PREBAKED_DIR,/lustre/scratch/fs62fb/dataset/srgb_baked}

# If pre-baking not available, fall back to on-the-fly GPU/CPU
enable_gpu_ocio_fallback: true

batch_size: 16
epochs: 200
learning_rate: 5e-5
num_workers: 8

accelerator: gpu
devices: auto
strategy: ddp
precision: 16-mixed
```

### 3.3 Update Dataset Loader for Pre-Baked Mode

**File**: `src/luminascale/training/dequantization_trainer.py` (lines ~90-110)

```python
# Check if pre-baked directory exists; use differently
if config.use_prebaked_dataset and os.path.isdir(config.prebaked_dir):
    logger.info(f"Using pre-baked sRGB dataset: {config.prebaked_dir}")
    # Load dataset from pre-baked PNG files (no OCIO transform needed)
    dataset = PrebakedDequantizationDataset(
        prebaked_dir=config.prebaked_dir,
        keys=lmdb_keys,
        device=device
    )
else:
    logger.info("Using on-the-fly OCIO transforms (GPU or CPU fallback)")
    # Original dataset with on-the-fly OCIO
    dataset = DequantizationDataset(
        lmdb_path=config.lmdb_path,
        keys=lmdb_keys,
        device=device
    )
```

---

## Part 4: Batch Script for HPC Execution

### 4.1 CPU Fallback Mode (Approach B)

**File**: `scripts/train_hpc_cpu_fallback.sh`

```bash
#!/bin/bash
#SBATCH --job-name=train_luminascale_cpu_fallback
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# Load dependencies
module load python cuda

# Activate environment
source /path/to/venv/bin/activate

# Set Python to print output immediately
export PYTHONUNBUFFERED=1

# Set OCIO environment variable (critical for CPU OCIO fallback)
export OCIO="${PWD}/config/aces/studio-config.ocio"

# Ensure GPU rendering isn't forced
export CUDA_VISIBLE_DEVICES=0,1
export SLURM_GPUS_PER_TASK=1

# Run training with CPU fallback enabled
python scripts/train_dequantization_net.py \
    --config-name hpc_slurm \
    strategy=ddp \
    devices=2 \
    num_workers=8 \
    batch_size=16 \
    epochs=200 \
    lmdb_path=/lustre/scratch/fs62fb/dataset/training_data.lmdb \
    output_dir=/lustre/scratch/fs62fb/outputs/training/$(date +%Y%m%d_%H%M%S)

echo "Job completed."
```

### 4.2 Pre-Bake Mode (Approach A)

**File**: `scripts/train_hpc_prebaked.sh`

```bash
#!/bin/bash
#SBATCH --job-name=bake_and_train
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=logs/prebake_%A_%a.out
#SBATCH --error=logs/prebake_%A_%a.err

module load python cuda
source /path/to/venv/bin/activate

export PYTHONUNBUFFERED=1
export OCIO="${PWD}/config/aces/studio-config.ocio"

# Phase 1: Pre-bake dataset (if not already done)
PREBAKED_DIR="/lustre/scratch/fs62fb/dataset/srgb_baked"
if [ ! -d "$PREBAKED_DIR" ]; then
    echo "Pre-baking dataset..."
    python scripts/bake_dataset.py \
        --input-dir /lustre/scratch/fs62fb/dataset/aces_originals \
        --output-dir "$PREBAKED_DIR"
else
    echo "Pre-baked dataset already exists at $PREBAKED_DIR"
fi

# Phase 2: Train with pre-baked dataset (now we can use 2 GPUs)
srun -K --ntasks-per-node=2 \
    python scripts/train_dequantization_net.py \
    --config-name hpc_slurm \
    use_prebaked_dataset=true \
    prebaked_dir=$PREBAKED_DIR \
    strategy=ddp \
    devices=2 \
    batch_size=16 \
    epochs=200 \
    output_dir=/lustre/scratch/fs62fb/outputs/training/$(date +%Y%m%d_%H%M%S)

echo "Pre-bake and training complete."
```

---

## Part 5: Monitoring & Diagnostics on HPC

### 5.1 Check Real-Time Performance

```bash
# SSH to login node, monitor a running job
watch -n 2 'squeue --me | head -5 && echo "---" && nvidia-smi'

# Or, SSH to compute node directly (while job runs)
srun -p gpu --pty nvidia-smi -l 1
```

### 5.2 Capture Diagnostics During Training

```bash
# Modify training script to log diagnostics on first batch
cat >> src/luminascale/training/dequantization_trainer.py << 'EOF'

def log_hpc_diagnostics(logger):
    """Log HPC environment details for debugging."""
    import socket
    import subprocess
    
    logger.info("=== HPC ENVIRONMENT DIAGNOSTICS ===")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
    logger.info(f"SLURM_GPUS_PER_NODE: {os.environ.get('SLURM_GPUS_PER_NODE')}")
    
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        logger.info(f"NVIDIA Driver: {result.stdout.strip()}")
    except:
        logger.warning("Could not query NVIDIA driver")
    
    logger.info(f"OCIO config: {os.environ.get('OCIO')}")
    logger.info(f"/dev/dri exists: {os.path.exists('/dev/dri')}")
EOF
```

### 5.3 Check for GPU OCIO Fallback Usage

```bash
# Search log file for fallback indicators
tail -f logs/train_*.err | grep -E "(GPU OCIO|CPU OCIO|Falling back|initialized)"
```

---

## Part 6: Expected Outputs & Success Criteria

### **Approach B (CPU Fallback) – Success Indicators**

```
✓ Training starts without EGL errors
✓ Log shows "GPU OCIO processor initialized" OR "Falling back to CPU OCIO"
✓ Training loop runs; batches processed at ~0.5–2 sec/batch (depending on image size)
✓ Multi-GPU DDP training communicates between GPUs
```

### **Approach A (Pre-Baking) – Success Indicators**

```
✓ bake_dataset.py completes without errors
✓ Output directory contains sRGB PNG files matching input ACES count
✓ Training loads from pre-baked directory; batch time ~0.1–0.3 sec (much faster)
✓ Multi-GPU DDP training has minimal OCIO overhead (transforms already done)
```

---

## Part 7: Troubleshooting Decision Tree

```
Training crashes on HPC with EGL error?
│
├─ Yes, GPU OCIO failed (eglInitialize error)
│  ├─ Approach B (CPU Fallback) → Immediate fix
│  │  └─ Verify: log shows "Falling back to CPU OCIO"
│  │
│  └─ Approach A (Pre-Bake) → For production
│     └─ Run: python scripts/bake_dataset.py --input-dir ... --output-dir ...
│
└─ No, training runs
   ├─ Check per-batch time (acceptable < 1 sec?)
   │  ├─ Yes → Use current approach
   │  └─ No → Try Approach A (pre-baking) to speed up OCIO
   │
   └─ Check multi-GPU scaling (linear speedup with 2 GPUs?)
      ├─ Yes → Working correctly
      └─ No → Check NCCL_DEBUG=INFO in logs for communication issues
```

---

## Quick Start: Immediate HPC Deployment

1. **Verify OCIO config is accessible**
   ```bash
   ls -la config/aces/studio-config.ocio
   export OCIO="${PWD}/config/aces/studio-config.ocio"
   ```

2. **Test on single GPU first (Approach B)**
   ```bash
   sbatch scripts/train_hpc_cpu_fallback.sh
   # Monitor: tail -f logs/train_*.err | grep -i ocio
   ```

3. **If >20% performance loss on CPU OCIO → deploy pre-baking**
   ```bash
   # Bake dataset (runs once)
   python scripts/bake_dataset.py \
       --input-dir /path/to/aces \
       --output-dir /path/to/baked
   
   # Then training uses pre-baked dir
   sbatch scripts/train_hpc_prebaked.sh
   ```

---

## Summary Table: Implementation Effort vs. Benefit

| Approach | Implementation | Test Time | Benefit | Risk | Recommended |
|----------|---|---|---|---|---|
| **B: CPU Fallback** | 2–3h | 1h | ✅ Unblocks HPC | Low | ✅ Start here |
| **A: Pre-Bake** | 3–4h | 2h | ⭐⭐⭐ Best speed | Low | Then upgrade |
| **C: GPU Virtual** | 8–12h+ | 4h+ | ⭐⭐ (if works) | High | Last resort |
| **D: CPU-Optimized** | 6–8h | 3h | ⭐⭐ (better CPU) | Medium | If B not enough |

**Final Recommendation**: Implement **B now** (~2–3h), test, then **upgrade to A** (~3–4h more) for production.

