"""Async prefetching loader to hide WebDataset I/O latency.

Implements a background process (not thread) that prefetches batches while the main thread
processes the current batch. Using multiprocessing avoids Python GIL. CPU affinity pins
the prefetch process to specific cores away from training threads.
"""

from __future__ import annotations
import multiprocessing as mp
import logging
from typing import Iterator, Any
import os

logger = logging.getLogger(__name__)


class AsyncPrefetchLoader:
    """Wraps a DataLoader with async prefetching in a background process.
    
    Uses multiprocessing (not threading) to avoid Python GIL. Prefetch process is pinned
    to specific CPU cores away from main training thread.
    
    Maintains a queue of prefetched batches:
    - Background process: Fetch batches from dataloader → add to queue
    - Main thread: Get batches from queue (already prefetched)
    - Result: GPU processing overlaps with shard I/O
    
    Example:
        loader = AsyncPrefetchLoader(wds_loader, prefetch_size=2, cpu_cores=[4, 5, 6])
        for batch in loader:
            # Process batch while background process fetches next batch
            process_batch(batch)
    """
    
    def __init__(self, dataloader: Iterator, prefetch_size: int = 2, cpu_cores: list[int] | None = None):
        """Initialize async prefetch loader with multiprocessing.
        
        Args:
            dataloader: Iterator (WebLoader or DataLoader) to wrap
            prefetch_size: Number of batches to prefetch in advance
            cpu_cores: CPU core indices for prefetch process (e.g., [4, 5, 6]).
                      If None, uses cores 4+ (assuming cores 0-3 for training).
        """
        self.dataloader = dataloader
        self.prefetch_size = prefetch_size
        
        # Determine CPU cores for prefetch process
        if cpu_cores is None:
            num_cores = os.cpu_count() or 8
            # Use cores 4+ for prefetch (leave 0-3 for training)
            cpu_cores = list(range(min(4, num_cores), num_cores))
        self.cpu_cores = cpu_cores
        
        self.queue: mp.Queue | None = None
        self.process: mp.Process | None = None
        self._stop_event: mp.Event | None = None
        self._exception: Exception | None = None
        
    def _prefetch_worker(self, dataloader: Iterator, queue: mp.Queue, stop_event: mp.Event, cpu_cores: list[int]) -> None:
        """Background worker process that prefetches batches into queue.
        
        Runs in a separate process (no GIL). Pins itself to specified CPU cores.
        """
        try:
            # Pin process to CPU cores
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, set(cpu_cores))
                logger.debug(f"Prefetch worker pinned to CPU cores {cpu_cores}")
            
            for batch in dataloader:
                # Check stop signal
                if stop_event.is_set():
                    logger.debug("Prefetch worker received stop signal")
                    break
                
                # Put batch in queue (blocks if queue is full)
                try:
                    queue.put(batch, timeout=5)
                    logger.debug(f"Prefetched batch, queue size: {queue.qsize()}/{self.prefetch_size}")
                except mp.queues.Full:
                    logger.warning("Prefetch queue full, dropping batch")
                    continue
            
            # Signal end of iteration with None sentinel
            queue.put(None, timeout=5)
            logger.debug("Prefetch worker finished")
            
        except Exception as e:
            logger.error(f"Prefetch worker exception: {e}")
            try:
                queue.put(None, timeout=1)  # Signal end even on error
            except:
                pass
    
    def __iter__(self) -> AsyncPrefetchLoader:
        """Start prefetch worker process and return iterator."""
        logger.debug(f"Starting async prefetch loader (prefetch_size={self.prefetch_size}, cpu_cores={self.cpu_cores})")
        
        # Create queue and stop event
        self.queue = mp.Queue(maxsize=self.prefetch_size)
        self._stop_event = mp.Event()
        self._exception = None
        
        # Start prefetch process
        self.process = mp.Process(
            target=self._prefetch_worker,
            args=(self.dataloader, self.queue, self._stop_event, self.cpu_cores),
            daemon=True
        )
        self.process.start()
        logger.debug(f"Prefetch worker process started (PID={self.process.pid})")
        
        return self
    
    def __next__(self) -> Any:
        """Get next prefetched batch from queue."""
        if self.queue is None:
            raise RuntimeError("AsyncPrefetchLoader not initialized. Call iter() first.")
        
        # Get batch from queue (blocks until available)
        try:
            batch = self.queue.get(timeout=30)
        except mp.queues.Empty:
            raise TimeoutError("Prefetch worker timeout: no batch available")
        
        # None sentinel marks end of iteration
        if batch is None:
            # Clean up process
            if self.process:
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1)
            raise StopIteration
        
        return batch
    
    def stop(self) -> None:
        """Stop the prefetch worker process."""
        if self._stop_event:
            self._stop_event.set()
        if self.process and self.process.is_alive():
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        logger.debug("Async prefetch loader stopped")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()
