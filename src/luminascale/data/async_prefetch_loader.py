"""Async prefetching loader to hide WebDataset I/O latency.

Implements a background thread that prefetches batches while the main thread
processes the current batch. This hides shard reading latency and reduces
overall training time.
"""

from __future__ import annotations
import threading
import queue
import logging
from typing import Iterator, Any

logger = logging.getLogger(__name__)


class AsyncPrefetchLoader:
    """Wraps a DataLoader with async prefetching in a background thread.
    
    Maintains a queue of prefetched batches to hide I/O latency:
    - Background thread: Fetch batches from dataloader → add to queue
    - Main thread: Get batches from queue (already prefetched)
    - Result: GPU processing overlaps with shard I/O
    
    Example:
        loader = AsyncPrefetchLoader(wds_loader, prefetch_size=2)
        for batch in loader:
            # Process batch while background thread fetches next batch
            process_batch(batch)
    """
    
    def __init__(self, dataloader: Iterator, prefetch_size: int = 2):
        """Initialize async prefetch loader.
        
        Args:
            dataloader: Iterator (WebLoader or DataLoader) to wrap
            prefetch_size: Number of batches to prefetch in advance
        """
        self.dataloader = dataloader
        self.prefetch_size = prefetch_size
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self.thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._exception: Exception | None = None
        
    def _prefetch_worker(self) -> None:
        """Background worker thread that prefetches batches into queue."""
        try:
            for batch in self.dataloader:
                # Put batch in queue (blocks if queue is full)
                if self._stop_event.is_set():
                    break
                self.queue.put(batch)
                logger.debug(f"Prefetched batch, queue size: {self.queue.qsize()}/{self.prefetch_size}")
            
            # Signal end of iteration with None sentinel
            self.queue.put(None)
            logger.debug("Prefetch worker finished")
            
        except Exception as e:
            logger.error(f"Prefetch worker exception: {e}")
            self._exception = e
            self.queue.put(None)  # Signal end
    
    def __iter__(self) -> AsyncPrefetchLoader:
        """Start prefetch worker thread and return iterator."""
        self._stop_event.clear()
        self._exception = None
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
        logger.debug(f"Started prefetch worker with prefetch_size={self.prefetch_size}")
        return self
    
    def __next__(self) -> Any:
        """Get next prefetched batch from queue."""
        # Check if worker encountered exception
        if self._exception:
            raise self._exception
        
        # Get batch from queue (blocks until available)
        batch = self.queue.get()
        
        # None sentinel marks end of iteration
        if batch is None:
            # Clean up thread
            if self.thread:
                self.thread.join(timeout=1)
            raise StopIteration
        
        return batch
    
    def stop(self) -> None:
        """Stop the prefetch worker thread."""
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        logger.debug("Async prefetch loader stopped")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()
