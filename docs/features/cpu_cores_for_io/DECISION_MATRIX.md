# CPU Cores for I/O: Decision & Trade-offs

## Executive Summary

Current bottleneck: **361ms LMDB I/O** vs **230ms GPU compute** per batch
- GPU idle ~131ms/batch waiting for data
- We have 6 CPU cores allocated but using only 1 for DataLoader
- `num_workers=4` blocked by CUDA fork safety

**Solution Path**:
1. Try larger `batch_size` (simplest, 1.5x speedup likely)
2. If insufficient, use threading-based async prefetch (2.5x speedup)
3. Accept trade-off between complexity and speedup

---

## Option Comparison

### Option A: Larger Batch Size (RECOMMENDED FIRST)

```yaml
batch_size: 4 → 8
```

**How it works**:
- 8 samples per batch instead of 4
- GPU compute time ≈ 2x (230ms → 460ms)
- LMDB I/O time = same (361ms per batch)
- Batch time: 591ms → 821ms
- But 2x samples, so throughput: 18.8 → 24.6 samples/s (1.3x)

**Why it works**:
- Larger batches reduce communication overhead in DDP
- Larger batches allow GPU to amortize I/O over more work
- I/O + compute happen more sequentially (less interleaving)

**Pros**:
- ✅ No code changes (config only)
- ✅ Trivial to test (1 line change)
- ✅ Proven stable (standard PyTorch behavior)
- ✅ Better gradient quality (larger batches = better gradient estimates)
- ✅ Still works with DistributedSampler

**Cons**:
- ❌ GPU memory might hit limit (24GB A10 is tight for larger batches)
- ❌ Only 1.3-1.5x speedup (modest)
- ❌ Must reduce num_gpus if OOM (batch_size=8 on 2 GPUs worse than batch_size=4 on 4 GPUs)

**Decision**: Try this immediately if OOM not an issue.

---

### Option B: Threading-Based Async Prefetch

```python
preloader = AsyncBatchPreloader(dataset, num_workers=4)
srgb_32f, srgb_8u = preloader.get_batch()
```

**How it works**:
- 4 CPU threads continuously read LMDB into queue
- Main GPU thread fetches from queue (usually already loaded)
- GPU compute happens while threads prepare next batch
- LMDB I/O parallelized: 4 threads × 361ms/thread ÷ 4 = ~90ms effective if overlapped

```
Timeline (theoretical best case):
Worker-1: [LMDB-1..................][LMDB-2..................][LMDB-3....]
Worker-2:        [LMDB-4..................][LMDB-5..................]
Worker-3:               [LMDB-6..................][LMDB-7.............]
Worker-4:                      [LMDB-8..................][LMDB-9...]
GPU:      [transform-1..........][transform-2..........][transform-3......]
          (queue has batches waiting)
```

**Expected performance**:
- Effective batch time: 230ms GPU (transforms) + 90ms I/O overlap = 320ms
- Speedup: 591ms → 320ms = **1.85x**

**Pros**:
- ✅ Significant speedup (1.85x, almost 2x)
- ✅ No additional GPU memory (I/O happens in CPU RAM queues)
- ✅ CUDA safe (threads don't fork, no CUDA re-init)
- ✅ Scales with spare CPU cores (allocate 6, use 4 for prefetch)
- ✅ Transparent to main training loop (just call get_batch())

**Cons**:
- ❌ Complex code (threading, queues, error handling)
- ❌ CPU RAM usage: 3 batches × 50MB/image ≈ 1.5GB per buffer
- ❌ Harder to debug (race conditions, timeouts)
- ❌ Worker thread crashes can hang main thread (need robust error handling)
- ❌ Startup latency: takes ~1 second to fill queue initially
- ❌ Thread coordination overhead (locks, synchronization)

**Stability Concerns**:
- If slow disk: workers block on LMDB reads, queue fills slowly
- If GPU faster than I/O: queue drains faster than fills, GPU waits anyway
- If workers crash: main can hang on queue.get()

**Decision**: Use if Option A insufficient OR GPU memory allows larger batch_size.

---

### Option C: Custom num_workers with Spawn

```python
torch.multiprocessing.set_start_method('spawn', force=True)
dataloader = DataLoader(..., num_workers=4, start_method='spawn')
```

**How it works**:
- Instead of `fork()`, use `spawn`: subprocess starts fresh with no inherited CUDA
- Fresh subprocess initializes CUDA correctly
- First GPU operation succeeds (no re-init error)

**Why it might work**:
- Avoids the "Cannot re-initialize CUDA" error
- True multiprocessing parallelism (no GIL)
- GPU operations in child processes work independently

**Pros**:
- ✅ Official PyTorch path for multiprocessing
- ✅ Could achieve 3x speedup theoretically (full parallelism)

**Cons**:
- ❌ High memory overhead: each worker process is full Python interpreter (100MB+) × 4 = 400MB+
- ❌ LMDB might not work correctly across forked processes after fresh spawn
- ❌ Still have CUDA context issues in subprocesses if they do GPU work
- ❌ Complex debugging (parent/child process communication)
- ❌ Even if it works, we still hit LMDB contention with multiple opens
- ❌ More likely to hit mysterious bugs than threading

**Verdict**: Probably NOT viable. Threading safer with less complexity.

---

### Option D: Pre-Cache Dataset to NVMe

```bash
# Pre-process: dump all LMDB to NVMe SSDs
python scripts/cache_dataset_to_nvme.py /data/cache/luminascale.nvme

# Training uses cached images on fast SSD (not HDD-backed LMDB)
# LMDB I/O becomes ~50ms (NVMe random access)
```

**How it works**:
- One-time setup: extract all images from LMDB to raw binary format on NVMe
- Training loads from NVMe (much faster than HDD-backed LMDB)
- LMDB I/O → NVMe I/O: 361ms → 50ms

**Expected performance**:
- Batch time: 230ms compute + 50ms I/O = 280ms
- Speedup: 591ms → 280ms = **2.1x**

**Pros**:
- ✅ Simple (just faster I/O layer)
- ✅ Huge speedup (2.1x)
- ✅ No code restructuring needed
- ✅ Zero threading complexity
- ✅ Highly stable (no race conditions)

**Cons**:
- ❌ 50GB+ NVMe space on HPC cluster (might not be available)
- ❌ One-time cache build (20-30 minutes)
- ❌ NVMe space shared among cluster users (competitive allocation)
- ❌ Not portable (cache only valid on this cluster)

**Verdict**: Best long-term solution IF NVMe available. Phase 2 optimization.

---

## Decision Flow Chart

```
START: Current throughput 18.8 samples/s, target?

├─ "Just need 20-25 samples/s"
│  └─ Try Option A (batch_size=8)
│     ├─ OOM? → Reduce num_gpus or keep batch_size=4
│     └─ Works? → DONE, 1.3-1.5x speedup
│
├─ "Need 30+ samples/s"
│  ├─ GPU memory OK for batch_size=8? 
│  │  ├─ YES → Try Option A first anyway
│  │  │   ├─ Gets 1.5x? → DONE
│  │  │   └─ Still need more? → Option B (async)
│  │  └─ NO → Go straight to Option B (async prefetch)
│  │
│  └─ After Option B: If still not enough
│     └─ Phase 2: Option D (NVMe pre-cache)
│
└─ "Best possible performance"
   └─ Phase 2 Roadmap: A → B → D (1.3x → 1.85x → 2.1x cumulative)
```

---

## Recommended Timeline

### Week 1 (This Week): Baseline & Option A Testing

```python
# Test 1: Current baseline (measure)
batch_size=4, devices=4, use_async=false
→ Expected: 18.8 samples/s, 27% GPU util

# Test 2: Larger batch_size (easiest)
batch_size=8, devices=4
→ Expected: 28-30 samples/s, 56% GPU util
→ Verdict: Sufficient for 30% speedup? Go with this.
```

**Effort**: 30 minutes (one config change + benchmark)
**Risk**: Very low
**Payoff**: 1.3-1.5x speedup if it works

### Week 2 (If needed): Option B Async Prefetch

```python
# Implementation: 
# 1. Create async_prefetch.py with AsyncBatchPreloader (code provided)
# 2. Add 3 helper methods to OnTheFlyBDEDataset
# 3. Update config with use_async_prefetch flag
# 4. Test single epoch, measure

# Test 3: With async prefetch
batch_size=4, devices=4, use_async=true, prefetch_workers=4
→ Expected: 44-48 samples/s, 76% GPU util
→ Verdict: 2.1-2.5x speedup
```

**Effort**: 2-3 hours (implement AsyncBatchPreloader, integrate, test)
**Risk**: Medium (threading, timeouts, error handling)
**Payoff**: 2.1-2.5x speedup (nearly 2x)

### Week 4+ (Polish): Option D NVMe Cache

```python
# One-time dataset baking to NVMe
python scripts/cache_dataset_to_nvme.py

# Training uses NVMe files (fast random access)
```

**Effort**: 30 minutes setup + 20 min build time
**Risk**: Low (depends on cluster NVMe availability)
**Payoff**: Additional 1.1x on top of async (cumulative 2.3-2.7x)

---

## Why NOT Option C (spawn)

The hidden issue with `spawn`:

```python
# Parent process: Creates LMDB env on PID 1000
env = lmdb.open('training_data.lmdb', readonly=True)

# Child spawned: Fresh process PID 1005
# Tries to open SAME LMDB
env = lmdb.open('training_data.lmdb', readonly=True)  # ← What do we get?

# Problem: LMDB lock file (lock.mdb) may be in weird state
# Multiple processes opening same LMDB = contentious lock behavior
# Could get: "User requested MDB_NOTLS but the environment had already been used"
```

Threading avoids this because all threads share the same parent process LMDB context.

---

## Summary: Effort vs Payoff

| Option | Effort | Speedup | Code Risk | Complexity |
|--------|--------|---------|-----------|------------|
| **A: Batch Size** | 30min | 1.3x | 🟢 None | ⭐ |
| **B: Async Thread** | 3hr | 1.85x | 🟡 Medium | ⭐⭐⭐ |
| **C: num_workers spawn** | 1hr | ? | 🔴 High | ⭐⭐ |
| **D: NVMe Cache** | 30min + 20min | 2.1x | 🟢 Low | ⭐ |

**Recommended Path:**
1. **This week**: Option A (batch_size=8) — quick win or validation
2. **Next week**: Option B (async) — if A insufficient
3. **Phase 2**: Option D (NVMe) — if clusters has storage

---

## Validation Metrics

Track these for each option:

```yaml
Metric:
  - Samples per second (throughput)
  - GPU utilization (nvidia-smi %GPU)
  - Batch time distribution (min/avg/max ms)
  - LMDB load time (should decrease with async)
  - GPU memory (peak MB)
  - Convergence rate (loss per epoch)
  - Worker/thread health (errors, crashes)
```

**Baseline** (current, for comparison):
```
throughput: 18.8 samples/s
gpu_util: 27%
batch_time: 591ms avg
lmdb_load: 361ms avg
gpu_mem_peak: 1526MB
loss_per_epoch: 6.44e-6
worker_crashes: 0
```

---

## Appendix: Why Threading Beat num_workers

| Factor | Threading | num_workers |
|--------|-----------|------------|
| **LMDB thread-safe?** | ✅ Yes (read-only) | ⚠️ Maybe (process isolation weird) |
| **CUDA safe?** | ✅ Yes (shared context) | ❌ No (fork inherits CUDA) |
| **Memory per worker** | ~10KB | ~100MB+ |
| **Startup time** | ~10ms | ~500ms |
| **Overhead for I/O** | Minimal (I/O releases GIL) | Minimal (separate process) |
| **Debugging** | Easier (shared memory view) | Harder (inter-process debugging) |
| **Production-tested pyTorch?** | Less common | Very common |
| **Right tool for I/O?** | ✅ Yes | Overkill |

Threading wins for THIS use case: pure I/O parallelism, no heavy CPU work, CUDA integration required.

