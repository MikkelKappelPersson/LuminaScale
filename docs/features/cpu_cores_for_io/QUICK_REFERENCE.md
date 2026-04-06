# Quick Reference: CPU I/O Optimization

## The Problem (In One Picture)

```
Current Batch Timeline (591ms):
┌──────────────────────────────────────────────────────────────┐
│ LMDB I/O (CPU)        GPU Compute    GPU Waits               │
│ ████████████████░░░░░░░░░░░░░░░░░░ ░░░░░░░░                │
│ 361ms                230ms           131ms idle              │
└──────────────────────────────────────────────────────────────┘

With Async Prefetch (320ms):
┌─────────────────────────────────────────────────┐
│ LMDB (threads) while GPU computes GPU computes  │
│ ████░░░░░░░░░░▓████░░░░░░░░░░░░░░░░▓████───── │
│ Batches prefetched    GPU transforms           │
│ in bg                                           │
└─────────────────────────────────────────────────┘
```

---

## Options at a Glance

### 🟢 Option A: Larger Batch Size
```yaml
Change:
  batch_size: 4 → 8

Cost:
  ⏱️  30 minutes (1 line config)
  💻 None (no code changes)
  🎯 None (standard PyTorch)

Gain:
  📊 1.3-1.5x speedup
  💾 No extra GPU memory
  ✅ Proven stable

When:
  ✅ Try this FIRST
```

---

### 🟡 Option B: Async Threading
```python
preloader = AsyncBatchPreloader(dataset, num_workers=4)
batch = preloader.get_batch()

Cost:
  ⏱️  2-3 hours (implement ThreadPool + queue)
  💻 ~200 lines code
  🐛 Medium complexity (threading edge cases)
  💾 1-2GB CPU RAM for queue buffers

Gain:
  📊 1.85x speedup (nearly 2x!)
  ✅ CUDA safe (threads, not fork)
  ✅ Uses spare 6 CPUs effectively

When:
  ⚠️  Only if Option A insufficient
  ✅ If willing to spend 2-3 hours
```

---

### 🔴 Option C: num_workers (Spawn)
```python
# DON'T DO THIS ❌
dataloader = DataLoader(..., num_workers=4, start_method='spawn')
```

**Why not:**
- ❌ Still inherits CUDA context issues
- ❌ LMDB not safe across spawned processes
- ❌ More complex than async threading
- ❌ Mystery bugs likely

---

### 🟣 Option D: NVMe Cache (Phase 2)
```bash
# One-time: export LMDB to fast NVMe storage
python scripts/cache_to_nvme.py

# Training uses NVMe (50x faster than HDD)

Cost:
  ⏱️  30 min setup + 20 min build
  💾 50GB NVMe space needed
  
Gain:
  📊 2.1x speedup (best!)
  ✅ Simplest code (just faster I/O)
  ✅ Most stable
```

---

## Decision Tree (60 seconds)

```
START
  │
  ├─→ Q: Need >50 samples/s?
  │   ├─ NO → Stick with batch_size=4 or try Option A
  │   └─ YES → Continue
  │
  ├─→ Q: GPU memory available (can batch_size=8)?
  │   ├─ YES → Try Option A first (1.3x)
  │   │        If not enough → Try Option B (1.85x)
  │   └─ NO → Go straight to Option B
  │
  └─→ Q: Got 2-3 hours for implementation?
      ├─ YES → Do Option B (async threading)
      └─ NO → Use Option A or wait for Phase 2
```

---

## Recommended Action Plan

### ✅ This Week

**Step 1 (30 min):** Test Option A
```bash
# Just run with batch_size=8
python scripts/train_dequantization_net.py --config-name=test \
  batch_size=8 epochs=1

# Measure: samples/s (should be 25-30)
```

**Decision Point:**
- If `>= 25 samples/s`: ✅ Good enough, use Option A
- If `< 25 samples/s`: Proceed to Step 2

**Step 2 (2-3 hr if needed):** Implement Option B
```bash
# Copy code from IMPLEMENTATION_GUIDE.md
# Create: src/luminascale/training/async_prefetch.py
# Test: python train_dequantization_net.py --config-name=test \
#   use_async_prefetch=true prefetch_workers=4

# Measure: samples/s (should be 40-50)
```

---

## Performance Targets

| Metric | Baseline | +Option A | +Option B | +Option D |
|--------|----------|-----------|-----------|-----------|
| Throughput | 18.8 s/s | 24-30 | 40-48 | 50+ |
| Speedup | 1x | 1.3x | 1.85x | 2.6x |
| GPU %util | 27% | 56% | 76% | ~90% |
| Risk | none | ✅ low | ⚠️ med | ✅ low |

---

## Testing Checklist (5 minutes)

**Before Running:**
- [ ] Cluster has 4+ GPUs available
- [ ] LMDB data ready in `dataset/training_data.lmdb`
- [ ] Space for logs (~100MB per test)

**During Test:**
- [ ] Watch first 10 batches (queue fills up)
- [ ] Monitor GPU memory (nvidia-smi in another terminal)
- [ ] Watch for worker/thread errors in logs

**After Test:**
- [ ] Extract throughput: `grep "it/s" logs/*.log | tail -1`
- [ ] Compare to baseline: was it faster?
- [ ] Check GPU memory: stable? any spikes?

---

## Red Flags (Stop & Debug)

🚨 **Queue timeout after 1 minute**
- Workers stuck on disk I/O
- Fix: Increase workers or check disk speed

🚨 **GPU memory spike after 100 batches**
- Buffers not being freed
- Fix: Reduce queue_size from 3 → 2

🚨 **No speedup despite async threads**
- Threads not actually running (check logs)
- Fix: Enable DEBUG logging, verify `len(dataset)`

🚨 **Training converges differently**
- Shouldn't happen (same data, same transforms)
- Fix: Check if async is reordering samples unexpectedly

---

## Final Decision (Make This Choice Now)

**Based on your constraints:**

- [ ] **A: Easy (batch size)** — Go do this right now (30 min)
  
- [ ] **B: Medium (async)** — I'll implement after A doesn't work (2-3 hr)
  
- [ ] **Wait:** Neither option appeals, will revisit Phase 2

---

**Document Sources:**
- Full research: `ASYNC_PREFETCH_RESEARCH.md`
- Implementation code: `IMPLEMENTATION_GUIDE.md`
- Full decision matrix: `DECISION_MATRIX.md`
- Navigator: `README.md`
