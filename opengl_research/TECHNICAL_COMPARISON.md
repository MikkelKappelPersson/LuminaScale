# Multi-GPU HPC Solutions: Technical Comparison & Architecture Patterns

---

## Part 1: Solution Architecture Comparison

### Option 1: Pre-Baked Dataset (RECOMMENDED – Approach A)

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│ One-Time Pre-Processing Phase (Single GPU, Interactive)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ACES EXR Files (on disk)                                       │
│       │                                                          │
│       └─→ bake_dataset.py                                       │
│            ├─→ GPU OCIO (if available & fast)                   │
│            ├─→ CPU OCIO (if GPU unavailable; slower but works) │
│            └─→ Write sRGB PNG to cache directory                │
│                 │                                                │
│                 └─→ Cached sRGB Dataset (on disk; persistent)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Training Phase (Multi-GPU DDP, Slurm Batch Job)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cached sRGB Dataset                                            │
│       │                                                          │
│       └─→ DataLoader (no OCIO transforms needed)                │
│            │                                                     │
│            ├─→ GPU 0 (Process 0) ──┐                           │
│            │                        ├─→ Model Training (DDP)   │
│            ├─→ GPU 1 (Process 1) ──┤                           │
│            │                        │                          │
│            └─→ (+ optional CDL augmentation on GPU)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Multi-GPU Execution Plan**:
```
sbatch train_hpc_prebaked.sh
  ├─ Phase 1: Single GPU, CPU-bound (bake parallel .png writes)
  │  └─ Duration: 2–4 hours (depends on dataset size)
  │     Output: /lustre/scratch/fs62fb/dataset/srgb_baked/
  │
  └─ Phase 2: Multi-GPU DDP training
     ├─ GPU 0 (LOCAL_RANK=0): Load batch → Model forward → Loss
     ├─ GPU 1 (LOCAL_RANK=1): Load batch → Model forward → Loss
     ├─ NCCL AllReduce(): Synchronize gradients across GPUs
     └─ Optimizer step (each GPU tracks full model copy)
        
Scaling efficiency: ~85–90% (near-linear with hardware)
```

---

### Option 2: CPU Fallback (Approach B – Fast Path)

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Training Phase (Multi-GPU DDP, Slurm Batch Job)                 │
│ [No pre-processing; on-the-fly transforms]                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ACES EXR Dataset (on disk)                                     │
│       │                                                          │
│       └─→ DataLoader (per-batch)                                │
│            │                                                     │
│            └─→ Per-image OCIO Transform [Bottleneck!]           │
│                 │                                                │
│                 ├─→ TRY: GPU OCIO (OpenGL/EGL)                   │
│                 │   └─→ FAILS on HPC → Catch exception          │
│                 │                                                │
│                 └─→ FALLBACK: CPU OCIO (OpenImageIO)            │
│                     └─→ ~1–2 ms per image (slow)                │
│                         │                                        │
│                         └─→ DataFrame → GPU (async)              │
│                              │                                   │
│                              ├─→ GPU 0 (Process 0) ──┐          │
│                              │                       ├─→ Training│
│                              ├─→ GPU 1 (Process 1) ──┤          │
│                              │                       │          │
│                              └─→ Loss + Gradients   │          │
│                                                      │          │
│                                  NCCL AllReduce()───┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Multi-GPU Execution Plan**:
```
sbatch train_hpc_cpu_fallback.sh
  ├─ GPU 0 (GLOBAL_RANK=0):
  │  ├─ DataLoader[i]→ ACES EXR
  │  ├─ CPU Transform (1–2 ms)  ← per image, per batch
  │  ├─ GPU Training Loop
  │  └─ NCCL broadcast gradients
  │
  └─ GPU 1 (GLOBAL_RANK=1):
     ├─ DataLoader[i]→ ACES EXR
     ├─ CPU Transform (1–2 ms)  ← **ALSO RUNNING in parallel**
     ├─ GPU Training Loop
     └─ NCCL broadcast gradients

Scaling efficiency: ~60–75% (CPU OCIO becomes bottleneck)
Batch time: 0.5–1.5 sec (vs 0.1–0.3 sec with pre-baked)
```

**When CPU Transform Bottleneck Triggers**:
```
batch_size=16, num_workers=4
Per-batch cost: 16 images × 1.5 ms = 24 ms
Model forward: 300 ms
GPU utilization: 300 / (24+300) = 92%  ← Still acceptable

But with larger batches or images:
batch_size=32, image_size=2K, num_workers=8
Per-batch OCIO: 32 × 3 ms = 96 ms
Model forward: 400 ms
GPU utilization: 400 / (96+400) = 81%  ← Degraded,  consider pre-baking
```

---

### Option 3: GPU Virtualization (Advanced – Approach C)

**Architecture** (if HPC supports `EGL_MESA_SURFACELESS`):
```
┌─────────────────────────────────────────────────────────────────┐
│ Training Phase (Multi-GPU DDP, with GPU Virtualization)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ACES EXR Dataset                                               │
│       │                                                          │
│       └─→ DataLoader                                            │
│            │                                                     │
│            ├─→ GPU 0 (LOCAL_RANK=0):                            │
│            │   ├─→ OCIO Context [GPU Virtualized]               │
│            │   ├─→ ACES→sRGB via GLSL shader (very fast)       │
│            │   └─→ Model Training                               │
│            │                                                     │
│            ├─→ GPU 1 (LOCAL_RANK=1):                            │
│            │   ├─→ OCIO Context [GPU Virtualized]               │
│            │   ├─→ ACES→sRGB via GLSL shader (very fast)       │
│            │   └─→ Model Training                               │
│            │                                                     │
│            └─→ NCCL AllReduce() [Communication Layer]           │
│                                                                  │
│  **Key**: Each GPU context has separate EGL context via MPS     │
│           or NVIDIA virtualization layer                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Pros & Cons**:
- ✅ GPU acceleration for OCIO transforms maintained
- ✅ Similar performance to single-GPU setup
- ❌ **Requires HPC admin action** (enable EGL_MESA_SURFACELESS extensions)
- ❌ May not be available on all clusters
- ❌ Complex deployment & debugging

---

## Part 2: Performance Comparison

### Scenario: Training on 2 GPUs, Batch Size 16

| Metric | Pre-Bakedistration | CPU Fallback | GPU Virtual (if available) |
|--------|---|---|---|
| **Setup Time** | 2–4h (one-time baking) | 0 min (immediate) | 0 min (if admin enables) |
| **Per-Batch OCIO Time** | 0 ms (disk I/O only) | 24 ms (16 imgs × 1.5ms) | 2 ms (GPU shader) |
| **Per-Batch Model Time** | 300 ms | 300 ms | 300 ms |
| **Total Batch Time** | 300–350 ms | 324–350 ms | 302–320 ms |
| **Throughput** | 4.5–5.3 batches/sec | 3.3–3.1 batches/sec | 4.6–5.1 batches/sec |
| **2-GPU Speedup** | ~1.8–1.9× | ~1.5–1.6× | ~1.8–1.9× |
| **GPU Efficiency** | ~90% | ~75% | ~90% |
| **Scalability** | Linear (independent of setup) | Degrades with larger batches | Linear (if available) |

---

## Part 3: HPC-Specific Considerations

### Multi-GPU Resource Allocation (Slurm/DDP)

**Pre-Baked Approach**:
```bash
# Phase 1: Pre-baking (single GPU, high memory I/O)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16    # For parallel PNG writes
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Phase 2: Training (multi-GPU, lower CPU demand)
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8     # Fewer CPU workers (transforms done)
#SBATCH --mem=64G
#SBATCH --time=24:00:00
```

**CPU Fallback Approach**:
```bash
# Single job with higher CPU allocation (for OCIO transforms)
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16    # MORE CPU workers for concurrent OCIO
#SBATCH --mem=64G
#SBATCH --time=24:00:00
```

**Key Difference**: Pre-baking separates concerns (I/O preprocessing vs. compute training).

---

### Container (Singularity) Considerations

**Pre-Baked Workflow**:
```bash
# Build once: includes both GPU OCIO + CPU OCIO fallback
singularity build luminascale.sif luminascale.def
# (~5 min build time, no special HPC admin action needed)

# Pre-bake can run with or without GPU:
srun singularity exec luminascale.sif \
    python scripts/bake_dataset.py --input-dir ... --output-dir ...
# (Works with CPU OCIO if GPU unavailable)

# Training full with multi-GPU:
srun singularity exec --nv luminascale.sif \
    python scripts/train_dequantization_net.py strategy=ddp devices=2
```

**CPU Fallback Workflow**:
```bash
# Container must include both PyOpenColorIO + OpenImageIO
# (handled in luminascale.def %post section)

srun singularity exec --nv luminascale.sif \
    python scripts/train_dequantization_net.py strategy=ddp devices=2
# (On HPC: GPU OCIO init fails → graceful fallback to CPU OCIO)
```

**GPU Virtualization Workflow** (requires special setup):
```bash
# Container must expose /dev/dri AND have EGL_MESA_SURFACELESS support
singularity build --fakeroot luminascale.sif luminascale.def
# (May need admin intervention; ensure NVIDIA drivers included)

srun singularity exec --nv --bind /dev/dri luminascale.sif \
    python scripts/train_dequantization_net.py strategy=ddp devices=2
```

---

## Part 4: Failure Modes & Recovery

### Pre-Baked: What Can Go Wrong?

| Failure Mode | Symptom | Recovery |
|---|---|---|
| Pre-baking crashes | `bake_dataset.py` dies midway | Restart baking (checkpoints can resume) |
| Disk full during bake | Output dir fills up → write errors | Clear `/lustre/scratch/` (query admin) |
| Cached PNG corrupted | Training reads bad pixels | Re-bake that image; checksum validation |
| Config mismatch | Pre-baked color space != training expectation | Document which OCIO config used for baking; tag output dir |

**Resilience**: Pre-baked dataset is immutable; reusable across training runs.

---

### CPU Fallback: What Can Go Wrong?

| Failure Mode | Symptom | Recovery |
|---|---|---|
| Per-image transform crash | Random crashes on batch N | Identify bad image; skip in dataset. Add try/except |
| OCIO config missing | `$OCIO` env var unset → transform fails | `export OCIO=path/to/config.ocio` in sbatch script |
| Temp file collisions | Multiple workers create `/tmp/aces_UUID.exr` conflicts | Use unique temp dirs per worker: `/tmp/aces_rank$SLURM_PROCID_*.exr` |
| CPU bottleneck too severe | Throughput < 1 batch/sec → very slow training | Consider switching to pre-baking |

**Resilience**: Can recover per-image; but slower, less deterministic.

---

### GPU Virtualization: What Can Go Wrong?

| Failure Mode | Symptom | Recovery |
|---|---|---|
| EGL context clash | Multiple processes hit same context | Use per-GPU EGL virtualization (NVIDIA MPS) |
| Missing driver extension | `eglGetDisplay()` still fails | Check HPC admin did ***not*** enable `EGL_MESA_SURFACELESS` |
| Context switching overhead | More GPU context switches than single-GPU | Monitor `nvidia-smi` for context switch count |

---

## Part 5: Decision Tree for HPC Deployment

```
START
│
├─ Quick prototype (next few days)?
│  └─ YES → Use Approach B (CPU Fallback)
│           └─ Goal: Get training running ASAP
│           └─ Accept 20–30% slower OCIO overhead
│
├─ Production training (long-running multi-GPU)?
│  └─ YES → Use Approach A (Pre-Baking)
│           └─ One-time 2–4h preprocessing
│           └─ Then enjoy 3–4× faster training
│           └─ Better resource utilization
│
├─ Maximize GPU utilization (research priority)?
│  └─ YES → Query HPC admin about EGL_MESA_SURFACELESS
│           ├─ If supported → Approach C (GPU Virtual)
│           └─ If not supported → Approach A (Pre-Baking)
│
└─ Uncertain? → Start with B (minimal risk)
                Test performance
                If CPU OCIO bottleneck > 10% → Switch to A
```

---

## Part 6: Memory & Disk Requirements

### Pre-Baking Storage

```
ACES Dataset:
  - Input: N ACES2065-1 EXRs (16-bit float, 3 channels)
  - Size per image (1080p): ~24 MB (1920×1080×3×4 bytes×2)
  - N=1000 images → 24 GB raw
  
After Baking (sRGB PNG):
  - 8-bit PNG (lossy compression)
  - Size per image (1080p): ~1–2 MB (depending on PNG compression)
  - N=1000 images → 1–2 GB final
  
Total Storage Needed: 24 GB (ACES) + 2 GB (PNG cache) ≈ 26 GB
  (On AAU /lustre with 10 TB quota per user → no problem)

Baking Performance (on single GPU):
  - ~100 images/hour (with GPU OCIO)
  - ~20 images/hour (with CPU OCIO fallback)
  - For N=10,000 images: 100–500 hours
    └─ Distribute across multiple nodes → divide by 4
    └─ Or: Run baking overnight in background
```

### Runtime Memory

**Pre-Baked Training (DDP, 2 GPUs)**:
```
Per GPU:
  - Model weights: ~80 MB
  - Batch (16, 3, 1080, 1920, fp16): ~1.5 GB
  - Optimizer states (Adam): ~160 MB
  - Gradients: ~80 MB
  - Cache/overhead: ~200 MB
  
Total per GPU: ~2 GB (easily fits on V100/A100)
Total DDP job: 2 × 2 GB = 4 GB
```

**CPU Fallback Training (DDP, 2 GPUs)**:
```
Per GPU: (same as above) + worker processes

Worker processes (num_workers=8):
  - Per worker: ~100 MB (Python, OCIO libs, temp buffers)
  - 8 workers × 2 GPUs: ~1.6 GB CPU memory

Total DDP job: 4 GB (GPU) + 1.6 GB (CPU) = 5.6 GB
```

**GPU Virtualization** (same as Pre-Baked, assuming EGL virtualization is lightweight)

---

## Part 7: Recommended Deployment Checklist

### Immediate (This Week)

- [ ] Run diagnostic tests on HPC node
  - [ ] `test_egl.py` → understand what fails
  - [ ] `test_cpu_ocio.py` → verify CPU OCIO works
  - [ ] Query HPC admin: "Do you have `EGL_MESA_SURFACELESS` support?"

- [ ] Implement Approach B (CPU Fallback)
  - [ ] Modify `dataset_pair_generator.py` with try/except
  - [ ] Test on 1 GPU first: `sbatch train_hpc_cpu_fallback.sh`
  - [ ] Monitor logs for fallback message

- [ ] Test DDP multi-GPU with fallback
  - [ ] Run 2-GPU training: `sbatch scripts/train_hpc_cpu_fallback.sh` (2 GPUs)
  - [ ] Measure batch time, GPU efficiency

### Short-term (Next 2 Weeks)

- [ ] Implement Approach A (Pre-Baking)
  - [ ] Enhance `bake_dataset.py` with error handling
  - [ ] Run small bake test (100 images); measure time
  - [ ] Create `PrebakedDequantizationDataset` class

- [ ] Benchmark: Approach B vs. Approach A
  - [ ] Measure per-batch latency for both
  - [ ] Estimate total training time savings
  - [ ] Decide: full pre-bake vs. keep fallback

### Production (Before Full Training Run)

- [ ] Finalize chosen approach (usually A or hybrid A+B)
- [ ] Update documentation & training scripts
- [ ] Test full multi-GPU DDP convergence (3–5 epochs)
- [ ] Set up monitoring (TensorBoard, SlumrJob tracking)
- [ ] Archive this research into project wiki

---

## Summary

| Approach | Time to Implementation | Time to Results | Long-term Performance | Complexity | **Recommendation** |
|---|---|---|---|---|---|
| **B: CPU Fallback** | 2–3 h | 1 day | ⭐⭐ (slow) | ⭐ (simple) | **START HERE** |
| **A: Pre-Baking** | 3–4 h | 2–3 days | ⭐⭐⭐⭐⭐ (best) | ⭐⭐ (medium) | **Then upgrade** |
| **C: GPU Virtual** | 8–12 h+ | 4–5 days | ⭐⭐⭐⭐ (if works) | ⭐⭐⭐ (complex) | **If A not enough** |
| **A+B Hybrid** | 5–7 h | 2–3 days | ⭐⭐⭐⭐⭐ | ⭐⭐ (medium) | **BEST overall** |

**Final Verdict**: Go with **Hybrid Approach A+B** (implement fallback immediately, upgrade to pre-baking for production).

