# Implementation Guide: Async LMDB Prefetching

## Quick Decision Tree

```
Current bottleneck: 361ms LMDB I/O per batch

Q: Can we just use larger batch_size?
├─ YES → Start with batch_size=8 (easiest, good enough)
└─ NO → Implement async prefetch (complex, best speedup)

Async Prefetch Path:
├─ Queue-based (simplest threading)
├─ ThreadPoolExecutor (more control)
└─ Custom collate_fn (PyTorch native)
```

---

## Step 1: Ultra-Simple Test - Larger Batch Size

No code changes needed. Just update config:

```yaml
# configs/test.yaml
batch_size: 4   # Current
↓
batch_size: 8   # Test this first
```

**Command to test:**
```bash
srun --gres=gpu:4 --cpus-per-task=6 --mem=48G --time=02:00:00 \
  singularity exec --nv luminascale.sif \
  python scripts/train_dequantization_net.py --config-name=test batch_size=8 epochs=1
```

**What to measure:**
- GPU utilization (should increase)
- Batches per second (should increase)
- GPU memory (should stay same, batch_size doesn't compound memory on DDP)

**Expected**: 1.5-2x throughput improvement with minimal risk.

---

## Step 2: Minimal Queue-Based Async (If Step 1 Not Enough)

### Implementation: AsyncDatasetWrapper

Create new file: `src/luminascale/training/async_prefetch.py`

```python
"""Async LMDB prefetching using thread pool."""

from __future__ import annotations

import logging
import threading
from queue import Queue
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AsyncBatchPreloader:
    """Prefetch LMDB batches using CPU thread pool.
    
    Decouples I/O (CPU threads) from compute (GPU main thread).
    Workers read LMDB → numpy, main thread does GPU transforms.
    
    Usage:
        dataset = OnTheFlyBDEDataset(...)
        preloader = AsyncBatchPreloader(dataset, num_workers=4)
        
        for batch_idx in range(len(dataset)):
            srgb_32f, srgb_8u = preloader.get_batch()
            loss = train_step(srgb_32f, srgb_8u)
        
        preloader.shutdown()
    """
    
    def __init__(
        self,
        dataset,
        num_workers: int = 4,
        queue_size: int = 3,
        prefetch_device: str = "cuda:0",
    ):
        """Initialize async preloader.
        
        Args:
            dataset: OnTheFlyBDEDataset instance
            num_workers: Number of worker threads for LMDB reads
            queue_size: Max batches to buffer in queue
            prefetch_device: GPU device for main transforms
        """
        self.dataset = dataset
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.device = torch.device(prefetch_device)
        
        # Thread-safe queue: (batch_idx, aces_np, cdl_params)
        self.queue: Queue = Queue(maxsize=queue_size)
        
        # Control
        self.stop_event = threading.Event()
        self.next_idx = 0
        self.idx_lock = threading.Lock()
        self.workers = []
        
        logger.info(
            f"[AsyncBatchPreloader] Starting {num_workers} workers, "
            f"queue_size={queue_size}, device={prefetch_device}"
        )
        
        # Spawn workers
        for worker_id in range(num_workers):
            w = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=False,  # Will call shutdown() explicitly
            )
            w.start()
            self.workers.append(w)
    
    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread: continuously prefetch batches to queue.
        
        Each worker:
        1. Gets next batch index (atomically)
        2. Loads ACES from LMDB + CDL params (CPU work)
        3. Puts into queue (blocks if full)
        """
        logger.debug(f"[Worker {worker_id}] Started")
        
        while not self.stop_event.is_set():
            try:
                # Atomically get next index
                with self.idx_lock:
                    idx = self.next_idx
                    if idx >= len(self.dataset):
                        logger.debug(f"[Worker {worker_id}] Reached end (idx={idx})")
                        break
                    self.next_idx += 1
                
                # Load batch metadata (key, CDL) from dataset
                # This is very fast, dataset knows how to map idx to key
                key = self.dataset._get_key_for_idx(idx)
                cdl_params = self.dataset._get_cdl_params_for_idx(idx)
                
                # Load ACES from LMDB (CPU, I/O-bound, GIL-releasing)
                logger.debug(f"[Worker {worker_id}] Loading LMDB key={key}")
                aces_np = self.dataset._load_aces_from_lmdb(key)
                
                # Put in queue (blocks if full, timeout if stuck)
                self.queue.put(
                    (idx, key, aces_np, cdl_params),
                    timeout=30.0,  # Detect stuck main thread
                )
                
                logger.debug(
                    f"[Worker {worker_id}] Queued idx={idx}, shape={aces_np.shape}"
                )
                
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Error: {e}", exc_info=True)
                # Signal error by putting None
                try:
                    self.queue.put((None, None, None, None), timeout=1.0)
                except Exception:
                    pass
        
        logger.debug(f"[Worker {worker_id}] Exiting")
    
    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Main thread: get next prefetched batch and apply GPU transforms.
        
        Returns:
            (srgb_32f, srgb_8u) both [H, W, 3] on GPU
        
        Raises:
            RuntimeError: If worker failed or queue timeout
        """
        try:
            # Get from queue (blocks until available)
            idx, key, aces_np, cdl_params = self.queue.get(timeout=60.0)
            
            if aces_np is None:
                raise RuntimeError("Worker failed, got None from queue")
            
            # NOW do GPU transforms in main GPU process (CUDA safe!)
            logger.debug(f"[Main] Got batch idx={idx}, applying GPU transforms")
            
            # Move ACES to GPU
            aces_gpu = torch.from_numpy(aces_np).to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            
            # Apply CDL on GPU
            if cdl_params:
                aces_gpu = self.dataset.cdl_processor.apply_cdl_gpu(
                    aces_gpu,
                    cdl_params,
                )
            
            # Transform to sRGB
            srgb_32f = self.dataset.pytorch_transformer.aces_to_srgb_32f(aces_gpu)
            srgb_8u = self.dataset.pytorch_transformer.aces_to_srgb_8u(aces_gpu)
            
            logger.debug(f"[Main] Finished transforms for idx={idx}")
            
            return srgb_32f, srgb_8u
            
        except Exception as e:
            logger.error(f"[Main] Error getting batch: {e}", exc_info=True)
            raise
    
    def shutdown(self) -> None:
        """Stop all worker threads gracefully."""
        logger.info("[AsyncBatchPreloader] Shutting down...")
        self.stop_event.set()
        
        for i, w in enumerate(self.workers):
            w.join(timeout=5.0)
            if w.is_alive():
                logger.warning(f"[Worker {i}] Did not exit cleanly")
        
        logger.info("[AsyncBatchPreloader] Shutdown complete")
```

### Minimal Integration Points

**In `OnTheFlyBDEDataset`**, add these two helper methods:

```python
def _get_key_for_idx(self, idx: int) -> str:
    """Get LMDB key for global index (considering rank/world_size)."""
    # This is already internal logic, just expose it
    global_idx = idx + (self.rank * len(self))
    return self.pair_generator.keys_cache[global_idx % len(self.pair_generator.keys_cache)]

def _get_cdl_params_for_idx(self, idx: int) -> dict:
    """Get CDL parameters for this index."""
    # Just sample random CDL like __getitem__ does
    return self.look_generator.generate_random_look()

def _load_aces_from_lmdb(self, key: str) -> torch.Tensor:
    """Load ACES as numpy from LMDB (no GPU transform)."""
    # Delegate to pair_generator but return numpy (already does this internally)
    aces_tensor = self.pair_generator._load_aces_from_lmdb(key)
    return aces_tensor.numpy()  # ← Return numpy instead of tensor
```

### Training Loop Integration

In `train_dequantization_net.py`, replace the training loop:

```python
# OLD: Standard DataLoader
# for batch_idx, batch in enumerate(dataloader):
#     loss = trainer.train_step(batch)

# NEW: With async preloader
if cfg.get("use_async_prefetch", False):
    preloader = AsyncBatchPreloader(
        dataset=dataset,
        num_workers=cfg.get("prefetch_workers", 4),
        queue_size=cfg.get("prefetch_queue_size", 3),
        prefetch_device=f"cuda:{rank}",  # Per-GPU device
    )
    
    try:
        for batch_idx in range(len(dataset)):
            srgb_32f, srgb_8u = preloader.get_batch()
            # Rest of training loop...
            loss = train_step(srgb_32f, srgb_8u)
    finally:
        preloader.shutdown()
else:
    # Fallback to standard DataLoader for num_workers=0
    trainer.fit(model=model_module, train_dataloaders=dataloader)
```

---

## Step 3: Configuration

Add to `configs/test.yaml`:

```yaml
# test.yaml
batch_size: 4
devices: 4
num_workers: 0

# NEW: Async prefetching options
use_async_prefetch: false    # Set to true to enable
prefetch_workers: 4          # Number of CPU threads for I/O
prefetch_queue_size: 3       # Max batches to buffer
```

---

## Performance Comparison Template

Create file: `docs/features/cpu_cores_for_io/BENCHMARK_RESULTS.md`

```markdown
# Benchmarks: Batch Size vs Async Prefetch

## Baseline (Current)
- Configuration: batch_size=4, devices=4, use_async_prefetch=false
- GPU utilization: 27%
- Batches/sec: 4.74 it/s per GPU
- Total throughput: 18.8 samples/s
- LMDB bottleneck: YES (361ms I/O vs 230ms compute)

## Test 1: Larger Batch Size
- Configuration: batch_size=8, devices=4
- Expected GPU utilization: 56%
- Expected throughput: ~28 samples/s (1.5x speedup)
- Status: [ ] Tested / [ ] Results below

## Test 2: Async Prefetch
- Configuration: batch_size=4, use_async_prefetch=true, prefetch_workers=4
- Expected GPU utilization: 76%
- Expected throughput: ~47 samples/s (2.5x speedup)
- Status: [ ] Tested / [ ] Results below
```

---

## Testing Checklist

### Before Implementation
- [ ] Run baseline benchmark (current code): `batch_size=4, devices=4`
  - Measure: throughput, GPU memory, timing breakdown
- [ ] Review LMDB timing: should see ~361ms per batch

### After Step 1 (Larger Batch Size)
- [ ] Run with `batch_size=8`
- [ ] Measure throughput: compare to baseline
- [ ] Check GPU memory: unchanged (good)
- [ ] Check convergence: loss should still decrease

### After Step 2 (Async Prefetch)
- [ ] Ensure worker threads spawning (check logs)
- [ ] Measure queue fill rate (should be fast initially)
- [ ] Run single epoch (100+ batches)
- [ ] Check no hangs or queue deadlocks
- [ ] Measure total throughput: compare to baseline
- [ ] Measure GPU utilization: should be higher

### Stability Tests
- [ ] Run 10 epochs without crashes
- [ ] Monitor GPU memory for leaks (should be stable)
- [ ] Check worker thread errors (logs)
- [ ] Verify gradient convergence (loss should decrease)

---

## Debugging Common Issues

### Issue: Queue timeout / Main thread blocked

**Symptom**: `RuntimeError: queue.get() timeout after 60s`

**Causes**:
- Workers stuck on I/O (slow disk)
- Single worker too slow for queue size (increase workers)
- GPU transforms taking too long (profile transforms)

**Fix**:
```python
# Increase queue timeout
batch = preloader.get_batch(timeout=120.0)

# Or increase workers
num_workers=6  # Was 4, more workers = faster prefetch
```

### Issue: GPU memory spikes

**Symptom**: OOM after 100 batches despite stable single-batch memory

**Causes**:
- Queue holding too many batches (each ~2GB for our images)
- Buffers not freed after get_batch()

**Fix**:
```python
# Reduce queue size
queue_size=2  # Was 3, only buffer 2 batches

# Explicit cleanup (shouldn't be needed but helps)
del srgb_32f
torch.cuda.empty_cache()
```

### Issue: Threads not starting

**Symptom**: All workers exit immediately, queue empty

**Causes**:
- `len(self.dataset)` returns 0
- Worker gets exception in `_get_key_for_idx()`

**Fix**:
```python
# Enable debug logging
import logging
logging.getLogger('luminascale.training.async_prefetch').setLevel(logging.DEBUG)

# Check dataset length
print(f"Dataset length: {len(dataset)}")
```

---

## Next Steps

1. **Immediate**: Test Step 1 (larger batch_size) today
   - Zero risk, quick feedback
   - If 1.5x speedup sufficient, we're done
   
2. **If needed**: Implement Step 2 (async queue) this week
   - More complex, but well-defined implementation
   - Expected 2.5x speedup if Step 1 insufficient

3. **Monitor**: Convergence quality with async prefetch
   - Should be identical (just reordered I/O, same data)
   - Watch for any divergence in loss curves

---

## Code Review Checklist (Before Merge)

- [ ] `AsyncBatchPreloader` has timeout handling
- [ ] Worker threads are daemon=False and explicitly shutdown()
- [ ] Queue.get() has timeout to prevent Main thread hang
- [ ] Error handling: worker crash doesn't hang main thread
- [ ] Logging: DEBUG level shows what workers are doing
- [ ] Config: use_async_prefetch flag for easy toggle
- [ ] Test with multi-GPU (should work with DistributedSampler)
- [ ] Measurements: throughput improvement quantified
