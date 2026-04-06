# CPU Cores for I/O: Research & Implementation Index

## 📋 Overview

This folder contains research and implementation guides for parallelizing LMDB I/O using spare CPU cores to reduce the data bottleneck in training.

**Current Bottleneck**: 361ms LMDB I/O per batch (43% of total batch time)
**GPU Idle Time**: ~131ms per batch waiting for data
**Allocated CPUs**: 6 cores, currently using only 1 for DataLoader

---

## 📚 Documents

### 1. [ASYNC_PREFETCH_RESEARCH.md](ASYNC_PREFETCH_RESEARCH.md)
**Deep technical research on threading vs multiprocessing for I/O parallelism**

- Problem statement with numbers
- Why `num_workers=4` fails (CUDA fork safety)
- Threading vs multiprocessing comparison table
- 3 architectural approaches (Queue, Dual Pipeline, ThreadPoolExecutor)
- Expected performance gains (2-2.6x speedup)
- Full AsyncBatchPreloader code sketch
- LMDB thread-safety analysis
- Implementation gotchas (GIL, memory, error handling)

**Read this if**: You want to understand the technical foundations and trade-offs.

---

### 2. [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
**Step-by-step practical implementation with runnable code**

- Decision tree for which approach to use
- **Step 1**: Ultra-simple test with larger batch_size (baseline)
- **Step 2**: Complete AsyncBatchPreloader implementation (copy-paste ready)
- Configuration templates for test.yaml
- Testing checklist and benchmarking template
- Common issues and debugging solutions
- Code review checklist before merging

**Start here if**: You want to implement async prefetch.

---

### 3. [DECISION_MATRIX.md](DECISION_MATRIX.md)
**Option comparison and recommended timeline**

- 4 options: Option A (batch size), B (threading), C (spawn), D (NVMe cache)
- Detailed pros/cons for each
- Why Option C (num_workers spawn) doesn't work
- Decision flowchart
- Recommended 4-week timeline
- Effort vs payoff matrix
- Validation metrics for benchmarking

**Decide with this**: Figure out which option fits your constraints.

---

## 🎯 Quick Start Path

### For Immediate Testing (30 min effort):

1. Read: [DECISION_MATRIX.md](DECISION_MATRIX.md) sections "Quick Decision Tree" and "Option A"
2. Run this test:
   ```bash
   # Test larger batch size (no code changes)
   srun --gres=gpu:4 --cpus-per-task=6 --mem=48G --time=02:00:00 \
     singularity exec --nv luminascale.sif \
     python scripts/train_dequantization_net.py --config-name=test \
       batch_size=8 epochs=1 2>&1 | tee logs/test_batch_size_8.log
   ```
3. Compare throughput:
   ```bash
   # Baseline (batch_size=4): 18.8 samples/s (from previous test)
   # Your result: ??? samples/s
   ```
4. If sufficient: Done! If not, proceed to implementation.

---

### For Full Implementation (2-3 hr effort):

1. Read: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) "Step 1 & Step 2"
2. Copy AsyncBatchPreloader code from [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) 
   → Create `src/luminascale/training/async_prefetch.py`
3. Add 3 helper methods to `OnTheFlyBDEDataset`
4. Update config with `use_async_prefetch: true`
5. Run test and benchmark using template in guide

---

## 📊 Expected Outcomes

| Approach | Timeline | Speedup | Code Risk |
|----------|----------|---------|-----------|
| **Option A** (batch_size=8) | 30 min | 1.3-1.5x | None |
| **Option B** (async thread) | 2-3 hr | 1.85x | Medium |
| **Option D** (NVMe cache) | Phase 2 | 2.1x | Low |
| **Combined** (A+B+D) | Phase 2 | ~2.5x | Low-Med |

---

## 🔧 Current Baseline (For Reference)

From last successful training run (Job 826482):

```
Configuration:
  batch_size: 4
  devices: 4
  num_workers: 0
  
Performance:
  throughput: 4.74 it/s per GPU = 18.8 samples/s total
  gpu_utilization: 27%
  
Batch Timing:
  lmdb_load    : 361.37ms (43%)
  gpu_transfer :  24.78ms (3%)
  cdl          :  20.94ms (2%)
  aces_transform: 137.98ms (17%)
  crop         :  35.72ms (4%)
  ─────────────────────────
  total        : ~840ms per batch
  
Training Quality:
  loss at batch 500: 6.44e-6 (converging)
  gpu_mem_peak: 1526MB (stable)
```

---

## 🚀 Next Steps

1. **Today**: Run Option A test (batch_size=8), compare throughput
2. **Tomorrow**: If insufficient, start Option B implementation
3. **This week**: Have stable async prefetch working with benchmarks
4. **Phase 2**: Explore Option D (NVMe) if cluster storage available

---

## 🤔 FAQ

**Q: Why not just use larger batch_size?**
- A: We will test it first! Simplest. If 1.3-1.5x speedup is enough, done.

**Q: Why not num_workers like normal DataLoader?**
- A: CUDA fork safety: forked processes can't re-initialize CUDA. Threading avoids fork.

**Q: Will threading hurt performance vs true parallelism?**
- A: No. LMDB I/O is I/O-bound (not CPU-bound), GIL doesn't block disk I/O.

**Q: How much CPU overhead for async threads?**
- A: Minimal. Each thread does: `env.begin()` → `txn.get()` → put in queue (very fast).

**Q: What if threads are slower than GPU finishes?**
- A: GPU stalls on queue.get() just like before. No regression, only improvement if threads keep up.

**Q: Can this break convergence?**
- A: No. Same data, same order (within queue buffer). Convergence identical to sequential baseline.

---

## 📖 References & Further Reading

Within this folder:
- `ASYNC_PREFETCH_RESEARCH.md` — Technical deep dive
- `IMPLEMENTATION_GUIDE.md` — Coding guide
- `DECISION_MATRIX.md` — Business/strategy decision

External resources:
- [Python threading + I/O](https://realpython.com/intro-to-python-threading/#io-bound)
- [LMDB thread safety](https://github.com/lni/lmdb-py/blob/main/docs/lmdb.rst#threading-considerations)
- [PyTorch DataLoader design](https://pytorch.org/docs/stable/data.html#communication-between-processes)
- [CUDA fork safety](https://pytorch.org/docs/stable/notes/cuda.html#cuda-fork-safety)

---

## 💬 Questions?

If implementing, refer to:
1. **"How do I start?"** → [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
2. **"What should I choose?"** → [DECISION_MATRIX.md](DECISION_MATRIX.md)
3. **"Why does this work?"** → [ASYNC_PREFETCH_RESEARCH.md](ASYNC_PREFETCH_RESEARCH.md)
4. **"It's crashing!"** → [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) → Debugging section

---

**Version**: 1.0 | **Date**: Apr 6, 2026 | **Status**: Research Complete, Ready for Implementation
