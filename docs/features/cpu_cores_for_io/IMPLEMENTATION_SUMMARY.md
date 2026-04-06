# Async Prefetch Implementation - Summary

## What Was Implemented

Completed full implementation of threading-based async LMDB prefetching for the LuminaScale training pipeline. This parallelizes I/O (using CPU threads) with GPU compute, reducing the data bottleneck.

### Files Created/Modified

#### New Files:
1. **`src/luminascale/training/async_prefetch.py`**
   - `AsyncDataLoader`: PyTorch Lightning-compatible wrapper for async prefetch
   - `AsyncBatchPreloader`: Core threading implementation
   - Thread pool for parallel LMDB reads (2-4 CPU threads)
   - Queue-based batch buffering
   - Full error handling and logging

#### Modified Files:
1. **`src/luminascale/training/dequantization_trainer.py`**
   - Added `_get_key_for_idx(idx)`: Map dataset index to LMDB key
   - Added `_get_cdl_params_for_idx(idx)`: Generate random CDL parameters
   - Added `_load_aces_from_lmdb(key)`: Load ACES as numpy (for CPU thread usage)

2. **`scripts/train_dequantization_net.py`**
   - Import `AsyncDataLoader`
   - Conditional dataloader selection based on `use_async_prefetch` config flag
   - Proper device assignment for async operations

3. **`configs/test.yaml`**
   - Added `use_async_prefetch: false` (toggle)
   - Added `prefetch_workers: 4` (CPU threads)
   - Added `prefetch_queue_size: 3` (batch buffer size)

4. **`configs/test_async.yaml`** (new test config)
   - Minimal config with async enabled for testing
   - Single GPU, 2 workers, batch_size=2 for quick iteration

---

## Architecture Overview

```
Main Thread (GPU):
  ├─ Trainer loop
  ├─ AsyncDataLoader.__iter__()
  ├─ call get_batch()
  └─ GPU transforms + training step

Worker Threads (CPU):
  ├─ AsyncBatchPreloader._worker_loop()
  ├─ Read LMDB keys
  ├─ Load ACES images (I/O-bound, releases GIL)
  └─ Put in queue

Queue (thread-safe):
  ├─ Holds prefetched ACES numpy arrays
  ├─ Decouples I/O from compute
  └─ Typical buffer size: 2-3 batches
```

### Data Flow

```
Iteration 0:
  Worker-1: [LMDB read batch 0..........]
  Worker-2:             [LMDB read batch 1..........]
  Main:                                    │
                                    [transforms]→[train_step]

Iteration 1:
  Worker-1: [LMDB read batch 2................]
  Worker-2:             [LMDB read batch 3................]
  Main:                    │
                  [get_batch() ← already in queue]
                  [transforms]→[train_step]
```

---

## Configuration

### Enable Async Prefetch

```yaml
# configs/test.yaml
use_async_prefetch: true
prefetch_workers: 4          # 4 CPU threads
prefetch_queue_size: 3       # Buffer 3 batches
```

### Command Line Override

```bash
python scripts/train_dequantization_net.py \
  --config-name=test \
  use_async_prefetch=true \
  prefetch_workers=4 \
  epochs=1
```

---

## Expected Performance

### Baseline (current, no async):
```
Batch time: 591ms
  LMDB I/O:     361ms (61%)
  GPU compute:  230ms (39%)
GPU utilization: 27% (idle while loading)
Throughput: 18.8 samples/s
```

### With Async Prefetch (theoretical best case):
```
Batch time: 320ms
  I/O hidden:   361ms (done by workers)
  GPU compute:  230ms (visible)
  Overhead:     ~90ms (queue, synchronization)
GPU utilization: 76% (less idle time)
Throughput: 44-48 samples/s (2.3-2.5x speedup)
```

**Realistic expectation**: 1.85x speedup (40-48 samples/s)

---

## How to Test

### 1. Quick Single-GPU Test

```bash
# Enable async in config
python scripts/train_dequantization_net.py \
  --config-name=test_async \
  epochs=1
```

### 2. Benchmark Comparison

```bash
# Run WITHOUT async (baseline)
python train_dequantization_net.py --config-name=test \
  use_async_prefetch=false epochs=1 2>&1 | tee baseline.log

# Run WITH async (test)
python train_dequantization_net.py --config-name=test \
  use_async_prefetch=true prefetch_workers=4 epochs=1 2>&1 | tee async.log

# Extract throughput
echo "Baseline:" && grep "it/s" baseline.log | tail -1
echo "Async:"    && grep "it/s" async.log    | tail -1
```

### 3. Monitor During Training

```bash
# In separate terminal, watch GPU/CPU usage
watch nvidia-smi

# Or monitor LMDB timing
grep "BATCH TIMING" logs/*.log | tail -10
```

---

## Key Implementation Details

### Thread Safety

- **LMDB**: Thread-safe for reads (multiple readers can open same env)
- **Queue**: Thread-safe (Python's Queue.Queue)
- **GPU tensors**: All GPU transforms happen in main thread (CUDA context safe)
- **No fork**: Uses threading, not multiprocessing (avoids fork safety issues)

### Error Handling

- **Worker crash**: Puts sentinel value (None) in queue
- **Main thread timeout**: `queue.get(timeout=60s)` raises RuntimeError
- **Graceful shutdown**: `preloader.shutdown()` called in finally block

### Memory Management

- **Per-batch overhead**: ~150-200MB (queue buffer for 2-3 batches)
- **GPU memory**: Same as baseline (tensors still on GPU)
- **Cleanup**: Automatic via context manager pattern

---

## Debugging

### If queuing times out:

```python
# Check dataset length
print(f"Dataset length: {len(dataset)}")

# Check if LMDB is accessible
singularity exec --nv luminascale.sif python -c \
  "import lmdb; env = lmdb.open('dataset/training_data.lmdb'); print(env.stat())"

# Enable debug logging
logging.getLogger('luminascale.training.async_prefetch').setLevel(logging.DEBUG)
```

### If no speedup observed:

1. Check thread startup: look for `[AsyncBatchPreloader] Starting X workers` in logs
2. Verify LMDB is still the bottleneck: check BATCH TIMING output
3. Monitor system: are CPUs actually busy? (`top`, `htop`)
4. Try increasing workers: 4 → 6 or 8

### If GPU memory increases:

1. Reduce queue_size: 3 → 2
2. Reduce batch_size: 4 → 2
3. Check for memory leaks in GPU transforms (should be none)

---

## Integration with PyTorch Lightning

The `AsyncDataLoader` implements the standard Python iterator protocol:
- `__iter__()` returns iterator
- `__len__()` returns number of batches
- Yields tuple `(srgb_8u, srgb_32f)` matching LuminaScale's expected format

PyTorch Lightning calls:
```python
trainer.fit(model, train_dataloaders=async_dataloader)
```

Which internally does:
```python
for batch in async_dataloader:  # ← calls AsyncDataLoader.__iter__()
    loss = model.training_step(batch, ...)
```

---

## Multi-GPU Considerations

### For DistributedSampler-based training (standard DataLoader):
- Each GPU uses DistributedSampler to partition dataset
- AsyncDataLoader NOT used (use regular DataLoader)

### For OnTheFlyBDEDataset with rank/world_size (async DataLoader):
- Each GPU constructs dataset with own `rank` and `world_size`
- Dataset internally handles rank-aware indexing
- No DistributedSampler needed with AsyncDataLoader
- AsyncDataLoader iterates over full dataset (already partitioned per-rank)

**Note**: When using `use_async_prefetch=true`, the training script does NOT create a DistributedSampler (it would duplicate filtering).

---

## Performance Tuning Guidelines

| Scenario | Recommendation |
|----------|---|
| LMDB on HDD (slow I/O) | ↑ prefetch_workers (6-8), ↑ queue_size (4-5) |
| LMDB on NVMe (fast I/O) | ↓ prefetch_workers (2-3), ↓ queue_size (2) |
| Memory constrained | ↓ batch_size, ↓ queue_size |
| CPU bound (slow network) | ↑ prefetch_workers |
| GPU under-utilized | ↑ batch_size or ↑ prefetch_workers |

---

## Known Limitations

1. **Convergence**: Same as baseline (same data, same transforms, just reordered)
2. **Determinism**: Batches may have slight timing variations, but deterministic if using same seeds
3. **Single-epoch warmup**: First batch may have setup overhead
4. **Error recovery**: Worker crash stops epoch, retry requires restart
5. **Profiling**: Standard profilers may count worker thread time separately

---

## Future Improvements

1. **Batch size adaptation**: Auto-scale prefetch_workers based on LMDB latency
2. **NVMe pinning**: Optional pre-cache to fast SSD for 3x speedup
3. **Adaptive queue sizing**: Dynamically adjust based on queue fill rate
4. **Checkpoint integration**: Save/restore prefetch state
5. **Streaming mode**: Support infinite LMDB streaming (beyond epoch boundary)

---

## Testing Checklist

- [x] Async prefetch imports without errors
- [x] AsyncBatchPreloader spawns workers correctly
- [x] AsyncDataLoader yields batches in correct format
- [x] Queue handles batches without deadlock
- [x] Worker threads shut down gracefully
- [ ] Single GPU test runs to completion (in progress)
- [ ] Performance matches ~1.85x expectations
- [ ] Multi-GPU test with 4 GPUs
- [ ] Long-term stability (10 epochs)
- [ ] Convergence same as baseline

---

## References

- Full research: `docs/features/cpu_cores_for_io/ASYNC_PREFETCH_RESEARCH.md`
- Implementation guide: `docs/features/cpu_cores_for_io/IMPLEMENTATION_GUIDE.md`
- Decision matrix: `docs/features/cpu_cores_for_io/DECISION_MATRIX.md`
- Quick reference: `docs/features/cpu_cores_for_io/QUICK_REFERENCE.md`
