# Async Prefetching with Threading for LMDB I/O

## Problem Statement

Current single-GPU DataLoader performance:
- **LMDB I/O**: 361ms per batch (sequential, CPU)
- **GPU compute**: 230ms per batch (ACES + crop)
- **Total batch time**: ~840ms
- **GPU utilization**: ~27% (230ms / 840ms)

With multiple GPUs and DistributedSampler, each GPU still experiences sequential I/O per batch, leaving GPU idle while LMDB reads next batch.

**Goal**: Use spare CPU cores (6 allocated) to prefetch future batches while current batch is on GPU.

---

## Why num_workers Fails (Fork Safety Issue)

### Current Blocker: CUDA in Forked Subprocess

When `num_workers > 0`:
1. PyTorch DataLoader forks worker processes
2. Workers inherit parent's CUDA context
3. First GPU operation in worker tries to re-initialize CUDA
4. Error: `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

### Why This Happens

```python
# Parent process
dataset = OnTheFlyBDEDataset(...)  # Creates CUDA context if device="cuda"
dataloader = DataLoader(dataset, num_workers=4)  # Forks 4 workers

# Worker 0 process
# Inherits CUDA context from parent but can't re-initialize
srgb_32f = pytorch_transformer.aces_to_srgb_32f(aces_tensor)  # ❌ CUDA fail
```

**Root cause**: Our dataset immediately applies GPU transforms (`pytorch_transformer.aces_to_srgb_32f()`), which requires CUDA initialization in worker.

---

## Threading as Alternative (No Fork, No CUDA Issues)

### Threading vs Multiprocessing

| Aspect | Multiprocessing (num_workers) | Threading |
|--------|------|----------|
| **Isolation** | Separate processes, separate CUDA contexts | Shared memory, shared GIL |
| **CUDA Safety** | Fork unsafe | ✅ Thread safe (no fork) |
| **LMDB Safety** | Requires custom fork logic | ✅ LMDB thread-safe by default |
| **Setup** | Complex (spawn mode, context) | Simple (Queue, Thread) |
| **Memory Overhead** | High (full process clones) | Low (threads share memory) |
| **CPU Parallelism** | ✅ True parallelism (bypasses GIL) | ❌ GIL-blocked if CPU-bound |
| **Best Use Case** | CPU-heavy transforms | I/O-bound operations |

### LMDB + Threading

LMDB is **I/O-bound**, not CPU-bound:
- Reading from disk (libmdb atomic transactions)
- No heavy computation during read
- **GIL doesn't block I/O operations**
- Multiple threads can read LMDB safely without GIL contention

**Verdict**: Threading perfect for LMDB prefetch. Each thread can do `env.begin()` → `txn.get()` without blocking others.

---

## Architecture Options

### Option 1: Queue-Based Prefetch (Simplest)

```
┌─────────────────┐
│ Main Training   │           ┌──────────────────────┐
│    (GPU)        │◄─────────┤ Prefetch Queue       │
│  • Compute      │  fetch    │ (thread-safe)        │
│    batch[i]     │           │ [batch[i], batch[i+1]│
└─────────────────┘           │  batch[i+2]]         │
                              └──────────────────────┘
                                     ▲
                                     │ populate
                              ┌──────────────────┐
                              │ CPU Thread Pool  │
                              │ • LMDB read[i+2] │
                              │ • LMDB read[i+3] │
                              │ (no GPU work!)    │
                              └──────────────────┘
```

**Mechanism**:
1. Main thread consumes batch from queue
2. Worker threads asynchronously read LMDB batches
3. Batches sit in queue (CPU RAM) until main thread ready
4. Return pre-loaded batch to DataLoader

**Pros**:
- Simple, thread-safe with standard `Queue.Queue`
- Decouples LMDB I/O from GPU compute
- Easy to debug (separate thread, separate concern)

**Cons**:
- Adds CPU→GPU transfer overhead for prefetched batches
- Queue memory usage for buffered batches (3-4 batches ≈ 2-4GB for 32-patch batches)

---

### Option 2: Dual Pipeline (No GPU Transform in Prefetch)

Instead of returning GPU tensors from prefetch, return **CPU numpy arrays**:

```
Worker Thread:
  aces_np = load_lmdb_aces()  # CPU numpy, no GPU
  return aces_np  # ← Queue this, 25MB per image

Main GPU Thread:
  aces_np = queue.get()
  aces_gpu = pytorch_transformer.aces_to_srgb_32f(aces_np)  # ← GPU transform here
```

**Pros**:
- Prefetch doesn't touch GPU (pure I/O parallelism)
- GPU transforms stay in main process (no CUDA issues)
- Simpler CUDA safety story

**Cons**:
- CPU↔GPU transfer happens on main thread (still blocks GPU from compute)
- Need custom DataLoader collate to handle prefetched numpy arrays

---

### Option 3: Thread Pool with Callbacks (Advanced)

```python
executor = ThreadPoolExecutor(max_workers=4)

def prefetch_batch(batch_indices):
    """Run in thread pool, returns loaded batch"""
    batches = {}
    for idx in batch_indices:
        aces_np = load_lmdb(idx)
        batches[idx] = aces_np
    return batches

# Submit prefetch task for next 2 batches
future = executor.submit(prefetch_batch, [i+2, i+3, i+4])

# Do current GPU compute
gpu_output = train_step(current_batch)

# Wait for prefetch, get next batch
next_batch = future.result()  # Blocks if not ready
future = executor.submit(...)  # Submit next prefetch
```

**Pros**:
- Fine-grained control over prefetch timing
- Can submit multiple prefetch tasks
- Great for pipeline where GPU and prefetch have different schedules

**Cons**:
- More complex error handling
- Manual synchronization needed
- Harder to integrate with PyTorch DataLoader

---

## Specific Implementation Strategy for LuminaScale

### Current Pipeline Bottleneck

```python
# In OnTheFlyBDEDataset.__getitem__()
ACES → CDL (GPU) → ACES-to-sRGB (GPU) → Crop (GPU) → Return

# Current: Sequential I/O before GPU work starts
1. Load from LMDB (CPU, 361ms) 📛 Can be async!
2. GPU transform (230ms)
3. Crop (36ms)
```

### Proposed: Async LMDB, GPU Transforms in Main

```python
# Worker threads: Pure I/O (no GPU)
load_aces_from_lmdb(key) → numpy array → Queue

# Main thread: GPU transforms
aces_np = queue.get()  # Already loaded, minimal wait
aces_gpu = pytorch_transformer.aces_to_srgb_32f(aces_np)
srgb_8u = pytorch_transformer.aces_to_srgb_8u(aces_gpu)
```

### Implementation Plan

**Step 1**: Create `AsyncLMDBDataset` wrapper
- Wraps `OnTheFlyBDEDataset`
- Spawns thread pool for LMDB reads
- Returns queue of prefetched batches

**Step 2**: Modify DataLoader
- Use `num_workers=0` (still, no fork)
- Use custom `collate_fn` to return pre-loaded numpy batches
- Keep GPU transforms in main process

**Step 3**: Integrate with trainer
- Replace standard DataLoader with AsyncLMDBDataset
- GPU still does transforms (CUDA safe)
- Thread pool handles I/O parallelism

---

## Expected Performance Gains

**Current (361ms I/O + 230ms GPU):**
```
Batch 0: |---LMDB(361ms)---|---GPU(230ms)---|
Batch 1:                     |---LMDB(361ms)---|---GPU(230ms)---|
```
Result: 591ms per batch

**With Async Prefetch (2 worker threads):**
```
Worker-1:  |LMDB(361ms)-----|LMDB(361ms)-----|LMDB(361ms)---|...
Worker-2:  |  LMDB(361ms)-----|LMDB(361ms)-----|LMDB(361ms)|...
Main GPU:  |---GPU(230ms)---|---GPU(230ms)---|---GPU(230ms)|...
                              (fetches from queue, typically ready)
```

**Result**: ~230-300ms per batch (GPU compute time)
- **2.6-2.0x speedup** from hiding LMDB I/O
- GPU utilization: 76-87% (vs current 27%)

---

## Implementation Code Sketch

### AsyncBatchPreloader

```python
from queue import Queue
from threading import Thread, Lock
import logging

class AsyncBatchPreloader:
    """Prefetch batches using CPU thread pool while GPU computes."""
    
    def __init__(
        self,
        dataset,
        num_workers=2,
        queue_size=3,
    ):
        self.dataset = dataset
        self.queue = Queue(maxsize=queue_size)
        self.workers = []
        self.stop_event = threading.Event()
        self.next_idx = 0
        self.idx_lock = Lock()
        
        # Start worker threads
        for _ in range(num_workers):
            w = Thread(target=self._worker_loop, daemon=True)
            w.start()
            self.workers.append(w)
    
    def _worker_loop(self):
        """Worker thread: continuously prefetch batches to queue."""
        while not self.stop_event.is_set():
            try:
                # Get next index to process
                with self.idx_lock:
                    idx = self.next_idx
                    self.next_idx += 1
                
                if idx >= len(self.dataset):
                    break
                
                # Call dataset but extract ONLY LMDB load (no GPU)
                # We'll do GPU transforms in main thread
                batch = self._load_batch_cpu_only(idx)
                
                # Put in queue (blocks if full)
                self.queue.put((idx, batch), timeout=1.0)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.queue.put((None, None))
    
    def _load_batch_cpu_only(self, idx):
        """Load batch from LMDB, return as numpy (no GPU transform)."""
        # This is the CPU-only part we want to parallelize
        aces_np, _ = self.dataset._load_raw_lmdb(idx)  # Returns [H,W,3] numpy
        return aces_np
    
    def get_batch(self):
        """Main thread: get next prefetched batch."""
        idx, batch_np = self.queue.get(timeout=5.0)
        
        # NOW do GPU transforms in main process
        aces_gpu = batch_np.to(self.device, non_blocking=True)
        srgb_32f = self.pytorch_transformer.aces_to_srgb_32f(aces_gpu)
        srgb_8u = self.pytorch_transformer.aces_to_srgb_8u(aces_gpu)
        
        return srgb_32f, srgb_8u
    
    def shutdown(self):
        """Stop all worker threads."""
        self.stop_event.set()
        for w in self.workers:
            w.join(timeout=2.0)
```

### Integration with DataLoader

```python
# In training script
dataset = OnTheFlyBDEDataset(...)

# Wrap with async preloader
preloader = AsyncBatchPreloader(
    dataset,
    num_workers=4,      # Use 4 CPU cores for I/O
    queue_size=3,       # Buffer 3 batches in RAM
)

# Standard training loop
for idx in range(len(dataset)):
    batch = preloader.get_batch()
    loss = train_step(batch)
    
    if idx % 100 == 0:
        print(f"idx={idx}, loss={loss}")

preloader.shutdown()
```

---

## Gotchas & Considerations

### 1. GIL (Global Interpreter Lock)
- **I/O operations release the GIL** → Multiple threads can read LMDB concurrently
- **CPU-bound operations hold the GIL** → Threading won't help if we do CPU transforms
- **Solution**: Keep GPU transforms in main thread (they're already GPU-bound)

### 2. Queue Memory Overhead
- Each prefetched batch in queue occupies RAM
- 4 images × 32 patches/image × 50MB/image ≈ 6.4GB per buffer
- With `queue_size=3`, that's ~19GB
- **Solution**: Use smaller queue_size (2-3) or stream individual images

### 3. Thread-Safety of LMDB
- LMDB is thread-safe for **reading** (multiple readers → parallel)
- LMDB is **NOT thread-safe for writing** (but we only read)
- Each thread needs its own `env.begin()` transaction
- **Safe pattern**: `with env.begin(write=False) as txn: txn.get(...)`

### 4. Error Handling
- Worker thread crashes silently if not handled
- Main thread could hang waiting on empty queue
- **Solution**: Use timeout on `queue.get()`, monitor thread health

### 5. Startup Latency
- First few batches wait for workers to fill queue
- If `queue_size=3` and 4 workers, takes ~1.4 seconds to fill initially
- **Solution**: Pre-fill queue in __init__, or warm-up loop

---

## Alternative: Reduce Batch Size + Overlap

Without restructuring, simpler approach:

```yaml
# Instead of threading complexity...
batch_size: 4   → 8      # Larger batches hide more I/O latency
devices: 4      → 2      # Run on fewer GPUs with full batches
```

**Why this might work**: 
- Single GPU with batch_size=8: GPU compute time doubles (230ms → ~460ms)
- LMDB I/O (361ms) now 79% of batch time instead of 43%
- GPU utilization: 56% instead of 27%

**Trade-off**: Less GPU parallelism but simpler, more stable.

---

## Decision Matrix

| Approach | Complexity | Speedup | Stability | Recommend For |
|----------|-----------|---------|-----------|----------------|
| **Larger batch_size** | ⭐ (1/5) | ~1.5x | ✅ High | First try, low risk |
| **Queue-based async** | ⭐⭐⭐ (3/5) | ~2.5x | ⚠️ Medium | If batch_size insufficient |
| **ThreadPoolExecutor** | ⭐⭐⭐⭐ (4/5) | ~2.5x | ⚠️ Medium | Fine-grained control needed |
| **Custom num_workers** | ⭐⭐⭐⭐⭐ (5/5) | ~3x | ❌ Low | Not viable (CUDA fork) |

---

## Recommendation

**Phase 1 (This Week)**: Try batch_size optimization
```yaml
batch_size: 8
devices: 4
```
Expected: 1.5-2x speedup, minimal code changes.

**Phase 2 (If needed)**: Implement Queue-based async preloader
- If Phase 1 doesn't achieve 2x speedup
- Implement AsyncBatchPreloader above
- Expected: 2.5x total speedup from baseline

**Phase 3 (Polish)**: Integrate with PyTorch Lightning
- Replace DataLoader with async-aware version
- Monitor thread health, error rates

---

## References

- [Python threading + LMDB](https://github.com/lni/lmdb-py/blob/main/docs/lmdb.rst#threading-considerations)
- [PyTorch DataLoader num_workers](https://pytorch.org/docs/stable/data.html#communication-between-processes)
- [CUDA fork safety](https://pytorch.org/docs/stable/notes/cuda.html#cuda-fork-safety)
- [GIL and I/O](https://realpython.com/intro-to-python-threading/#daemon-threads)
