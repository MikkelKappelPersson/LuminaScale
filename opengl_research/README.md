# OpenGL/OCIO on Multi-GPU HPC: GPU-Only Solutions

**Focus**: Hardware-accelerated GPU rendering for multi-GPU Slurm jobs  
**Goal**: Get OCIO GPU transforms working on AAU AI Cloud with 2+ GPUs (DDP)

---

## 📋 What's Here

1. **[GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md)** ← **START HERE**
   - Implementation guide
   - EGL multi-GPU setup (explicit device selection)
   - DDP integration (per-process GPU assignment)
   - Slurm batch script template
   - Diagnostic test script

2. **[GPU_TROUBLESHOOTING_DEEP_DIVE.md](GPU_TROUBLESHOOTING_DEEP_DIVE.md)**
   - Root cause of EGL failures
   - Multi-GPU DDP challenges & solutions
   - NVIDIA driver matrix (which versions support multi-GPU EGL)
   - Debugging techniques
   - Performance tuning
   - Common failure modes + recovery

---

## ⚡ Quick Path: 2–2.5 Hours to Multi-GPU Training

### Step 1: Verify HPC Supports GPU OpenGL (15 min)

```bash
# SSH to HPC, run on compute node
srun -N 1 --gpus-per-node=2 python opengl_research/test_egl_multigpu.py

# Expected output:
# ✓ EGL detected 2 devices
# Device 0: [✓] EGL 1.5 OK
# Device 1: [✓] EGL 1.5 OK
```

**If fails**: Contact HPC admin: "Our NVIDIA driver too old" (need >= 358)

### Step 2: Implement GPU Device Selection (45 min)

**File**: `src/luminascale/utils/gpu_torch_processor.py`  
**Change**: Replace `_initialize_egl()` with multi-GPU aware `_initialize_egl_multigpu()`

👉 **See full code**: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md) Solution 1

### Step 3: Add GPU Device Mapping in DDP (30 min)

**File**: `src/luminascale/training/dataset_pair_generator.py`  
**Change**: Extract `LOCAL_RANK` and pass GPU ID to processor

👉 **See full code**: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md) Solution 2

### Step 4: Update Singularity Container (15 min)

**File**: `singularity/luminascale.def`  
**Change**: Add `libglvnd*` EGL libraries

👉 **See full code**: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md) Solution 3

### Step 5: Create Slurm Batch Script (20 min)

**File**: `scripts/train_hpc_multigpu.sh`  
**Template**: Provided in [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md)

### Step 6: Test Single GPU (30 min)

```bash
srun -N 1 --gpus-per-node=1 \
  python scripts/train_dequantization_net.py \
  --config-name default devices=1 accelerator=gpu epochs=1 batch_size=8

# Expected: GPU 100%, ~0.15 sec/batch, no EGL errors
```

### Step 7: Test Multi-GPU (30 min)

```bash
sbatch scripts/train_hpc_multigpu.sh

# Monitor: tail -f logs/train_*.err | grep -i "rank\|gpu\|ocio"

# Expected:
# GLOBAL_RANK: 0/2, GLOBAL_RANK: 1/2
# Both GPUs 95%+ utilized
# Batch time ~0.15–0.25 sec
```

**Success = 1.8–1.9× speedup on 2 GPUs (near-linear scaling)**

---

## ✅ Success Criteria

| Metric | Single GPU | Multi-GPU (2×) |
|--------|---|---|
| EGL devices | 1+ | 2 |
| Batch time | < 0.2 sec | < 0.25 sec |
| GPU util | 95%+ | 95%+ both |
| Speedup | N/A | ~1.8–1.9× |

---

## 🔧 Troubleshooting

| Issue | Doc |
|-------|-----|
| `EGL_NOT_INITIALIZED` | [GPU_TROUBLESHOOTING_DEEP_DIVE.md](GPU_TROUBLESHOOTING_DEEP_DIVE.md) Part 1 |
| Only 1 GPU active | [GPU_TROUBLESHOOTING_DEEP_DIVE.md](GPU_TROUBLESHOOTING_DEEP_DIVE.md) Part 8 |
| Driver too old | Contact admin (need >= 358) |
| Container GPU issues | [GPU_TROUBLESHOOTING_DEEP_DIVE.md](GPU_TROUBLESHOOTING_DEEP_DIVE.md) Part 7 |

---

## 📚 Full Documentation

**Implementations**: [GPU_OPENGL_MULTIGPU_HPC.md](GPU_OPENGL_MULTIGPU_HPC.md)  
**Troubleshooting**: [GPU_TROUBLESHOOTING_DEEP_DIVE.md](GPU_TROUBLESHOOTING_DEEP_DIVE.md)

---

## 🎯 Next Action

→ Run diagnostic: 
```bash
srun -N 1 --gpus-per-node=2 python opengl_research/test_egl_multigpu.py
```

If ✓ on all devices: 2–2.5h to multi-GPU training  
If ✗: Check driver version, contact admin

