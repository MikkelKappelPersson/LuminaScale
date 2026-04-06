"""Async LMDB prefetching using thread pool for I/O parallelization."""

from __future__ import annotations

import logging
import threading
from queue import Queue
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AsyncDataLoader:
    """PyTorch Lightning-compatible DataLoader wrapper using async prefetching.
    
    This is a thin wrapper around AsyncBatchPreloader that:
    1. Prefetches LMDB data using CPU threads
    2. Batches the prefetched samples
    3. Acts like a standard PyTorch DataLoader for Lightning compatibility
    
    Usage:
        dataloader = AsyncDataLoader(
            dataset=dataset,
            batch_size=4,
            num_workers=4,  # CPU threads for I/O
            prefetch_device="cuda:0"
        )
        
        for batch in dataloader:
            srgb_8u, srgb_32f = batch
            loss = train_step(srgb_8u, srgb_32f)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int = 4,
        prefetch_device: str = "cuda:0",
        queue_size: int = 3,
    ) -> None:
        """Initialize async dataloader.
        
        Args:
            dataset: OnTheFlyBDEDataset instance
            batch_size: Batch size for collation
            num_workers: Number of CPU threads for LMDB prefetch
            prefetch_device: GPU device for transforms
            queue_size: Max samples to buffer in prefetch queue
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_device = prefetch_device
        self.queue_size = queue_size
        self.preloader = None
    
    def __iter__(self):
        """Iterate over batches using async prefetch."""
        # Create preloader for this epoch
        self.preloader = AsyncBatchPreloader(
            dataset=self.dataset,
            num_workers=self.num_workers,
            queue_size=self.queue_size,
            prefetch_device=self.prefetch_device,
        )
        
        crop_size = self.dataset.crop_size
        device = torch.device(self.prefetch_device)
        
        batch_srgb_8u = []
        batch_srgb_32f = []
        
        try:
            for sample_idx in range(len(self.dataset)):
                # Get prefetched and GPU-transformed full-size images
                srgb_8u_full, srgb_32f_full = self.preloader.get_batch()
                
                # Random crop (same as in original __getitem__)
                H, W = srgb_32f_full.shape[0], srgb_32f_full.shape[1]
                
                if H < crop_size or W < crop_size:
                    # Upscale if too small
                    srgb_32f_full = srgb_32f_full.permute(2, 0, 1).unsqueeze(0)
                    srgb_8u_full = srgb_8u_full.permute(2, 0, 1).unsqueeze(0)
                    
                    srgb_32f = torch.nn.functional.interpolate(
                        srgb_32f_full,
                        size=(crop_size, crop_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                    
                    srgb_8u = torch.nn.functional.interpolate(
                        srgb_8u_full,
                        size=(crop_size, crop_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                else:
                    # Random crop - use .clone() to avoid keeping full images in memory
                    top = torch.randint(0, H - crop_size + 1, (1,), device=device).item()
                    left = torch.randint(0, W - crop_size + 1, (1,), device=device).item()
                    srgb_32f = srgb_32f_full[top : top + crop_size, left : left + crop_size, :].clone()
                    srgb_8u = srgb_8u_full[top : top + crop_size, left : left + crop_size, :].clone()
                
                # Convert to [C, H, W] format and normalize
                srgb_8u = (srgb_8u.float() / 255.0).permute(2, 0, 1)  # [H,W,3] → [3,H,W]
                srgb_32f = srgb_32f.float().permute(2, 0, 1)  # [H,W,3] → [3,H,W]
                
                # Accumulate into batch
                batch_srgb_8u.append(srgb_8u)
                batch_srgb_32f.append(srgb_32f)
                
                # Yield batch when full
                if len(batch_srgb_8u) == self.batch_size:
                    yield (
                        torch.stack(batch_srgb_8u),
                        torch.stack(batch_srgb_32f),
                    )
                    batch_srgb_8u = []
                    batch_srgb_32f = []
            
            # Yield partial batch at end (if any)
            if batch_srgb_8u:
                yield (
                    torch.stack(batch_srgb_8u),
                    torch.stack(batch_srgb_32f),
                )
        
        finally:
            # Ensure cleanup even if iteration interrupted
            if self.preloader:
                self.preloader.shutdown()
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class AsyncBatchPreloader:
    """Prefetch LMDB batches using CPU thread pool.

    Decouples I/O (CPU threads) from compute (GPU main thread).
    Workers read LMDB → numpy, main thread does GPU transforms.

    This solves the bottleneck where LMDB I/O (361ms) blocks GPU compute (230ms).
    With 4 worker threads, LMDB reads happen in parallel, GPU processes batch
    while next batch is being loaded.

    Usage:
        dataset = OnTheFlyBDEDataset(...)
        preloader = AsyncBatchPreloader(dataset, num_workers=4)

        for batch_idx in range(len(dataset)):
            srgb_8u, srgb_32f = preloader.get_batch()
            loss = train_step(srgb_8u, srgb_32f)

        preloader.shutdown()
    """

    def __init__(
        self,
        dataset,
        num_workers: int = 4,
        queue_size: int = 3,
        prefetch_device: str = "cuda:0",
    ) -> None:
        """Initialize async preloader.

        Args:
            dataset: OnTheFlyBDEDataset instance with _get_key_for_idx, etc
            num_workers: Number of worker threads for LMDB reads
            queue_size: Max batches to buffer in queue (memory constraint)
            prefetch_device: GPU device for main transforms (e.g., "cuda:0")
        """
        self.dataset = dataset
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.device = torch.device(prefetch_device)

        # Thread-safe queue: (batch_idx, key, aces_np, cdl_params)
        self.queue: Queue = Queue(maxsize=queue_size)

        # Control signals
        self.stop_event = threading.Event()
        self.next_idx = 0
        self.idx_lock = threading.Lock()
        self.workers = []

        logger.info(
            f"[AsyncBatchPreloader] Starting {num_workers} workers, "
            f"queue_size={queue_size}, device={prefetch_device}"
        )

        # Spawn worker threads (daemon=False so we control shutdown)
        for worker_id in range(num_workers):
            w = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=False,
            )
            w.start()
            self.workers.append(w)

        logger.info(f"[AsyncBatchPreloader] All {num_workers} workers started")

    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread: continuously prefetch batches to queue.

        Each worker:
        1. Gets next batch index (atomically)
        2. Loads ACES from LMDB + CDL params (CPU work, releases GIL)
        3. Puts into queue (blocks if full)
        """
        logger.info(f"[Worker {worker_id}] Started")

        while not self.stop_event.is_set():
            try:
                # Atomically get next index
                with self.idx_lock:
                    idx = self.next_idx
                    if idx >= len(self.dataset):
                        logger.info(f"[Worker {worker_id}] Reached end (idx={idx})")
                        break
                    self.next_idx += 1

                # Get metadata without loading image yet
                try:
                    key = self.dataset._get_key_for_idx(idx)
                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Failed to get key for idx={idx}: {e}", exc_info=True)
                    break

                try:
                    cdl_params = self.dataset._get_cdl_params_for_idx(idx)
                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Failed to get CDL params for idx={idx}: {e}", exc_info=True)
                    break

                # Load ACES from LMDB (CPU, I/O-bound, GIL-releasing)
                # This is the expensive operation we're parallelizing
                logger.info(f"[Worker {worker_id}] Loading LMDB key={key} (idx={idx})")
                try:
                    aces_np = self.dataset._load_aces_from_lmdb(key)
                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Failed to load LMDB key={key}: {e}", exc_info=True)
                    break

                logger.info(f"[Worker {worker_id}] Loaded batch, shape={aces_np.shape}, putting in queue...")

                # Put in queue (blocks if full, timeout detects stuck main thread)
                try:
                    self.queue.put(
                        (idx, key, aces_np, cdl_params),
                        timeout=30.0,
                    )
                    logger.info(f"[Worker {worker_id}] ✓ Queued idx={idx}")
                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Failed to put in queue: {e}", exc_info=True)
                    break

            except Exception as e:
                logger.error(f"[Worker {worker_id}] Unexpected error in main loop: {e}", exc_info=True)
                # Signal error by putting sentinel value
                try:
                    self.queue.put((None, None, None, None), timeout=1.0)
                except Exception:
                    pass
                break

        logger.info(f"[Worker {worker_id}] Exiting")

    def get_batch(self, timeout: float = 60.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Main thread: get next prefetched batch and apply GPU transforms.

        This is called by the main training loop. It blocks until a prefetched
        batch is available, then applies GPU transforms in the main process
        (which is CUDA-safe).

        Args:
            timeout: Seconds to wait for batch from queue before raising error

        Returns:
            (srgb_8u_input, srgb_32f_target) both [B, C, H, W] on GPU, normalized to [0, 1]

        Raises:
            RuntimeError: If worker failed or queue timeout
        """
        try:
            # Get from queue (blocks until available)
            logger.info(f"[Main] Waiting for batch from queue (timeout={timeout}s)...")
            idx, key, aces_np, cdl_params = self.queue.get(timeout=timeout)

            if aces_np is None:
                raise RuntimeError("Worker failed, got sentinel value (None) from queue - check worker logs")

            logger.info(f"[Main] Got batch idx={idx}, applying GPU transforms")

            # NOW do GPU transforms in main GPU process (CUDA safe!)
            
            # Move ACES to GPU
            aces_gpu = torch.from_numpy(aces_np).to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )

            # Apply CDL on GPU if params provided
            if cdl_params:
                aces_gpu = self.dataset.pair_generator.cdl_processor.apply_cdl_gpu(
                    aces_gpu,
                    cdl_params,
                )
                torch.cuda.synchronize()

            # Transform to sRGB (both 32-bit and 8-bit)
            srgb_32f = self.dataset.pair_generator.pytorch_transformer.aces_to_srgb_32f(aces_gpu)
            srgb_8u = self.dataset.pair_generator.pytorch_transformer.aces_to_srgb_8u(aces_gpu)
            torch.cuda.synchronize()

            logger.info(f"[Main] ✓ Finished GPU transforms for idx={idx}")

            # Return in correct order: (srgb_8u_input, srgb_32f_target) [H,W,3]
            return srgb_8u, srgb_32f

        except Exception as e:
            logger.error(
                f"[Main] Error getting batch: {e}\n"
                f"  Queue size: {self.queue.qsize()}\n"
                f"  Workers alive: {sum(1 for w in self.workers if w.is_alive())}\n"
                f"  Check worker logs above for details",
                exc_info=True
            )
            raise

    def shutdown(self) -> None:
        """Stop all worker threads gracefully."""
        logger.info("[AsyncBatchPreloader] Shutting down...")
        self.stop_event.set()

        for i, w in enumerate(self.workers):
            w.join(timeout=5.0)
            if w.is_alive():
                logger.warning(f"[AsyncBatchPreloader] Worker {i} did not exit cleanly")

        logger.info("[AsyncBatchPreloader] Shutdown complete")
