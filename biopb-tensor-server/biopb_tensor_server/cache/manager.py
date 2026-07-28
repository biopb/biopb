"""Thread-safe cache manager for Flight server.

Singleton manager that coordinates cache operations across all
resolve_chunk_data calls. Uses future/promise pattern for
safe concurrent computation of virtual chunks.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

import pyarrow as pa

from biopb_tensor_server.cache.base import (
    CacheBackend,
    CacheEntry,
    CacheStats,
    ChunkLocation,
)
from biopb_tensor_server.cache.file_backend import ArrowFileBackend, ArrowFileConfig
from biopb_tensor_server.cache.memory_backend import (
    MemoryCacheBackend,
    MemoryCacheConfig,
)
from biopb_tensor_server.core.config import CacheConfig


class CacheManager:
    """Singleton cache manager with thread-safe operations.

    Provides global cache access for resolve_chunk_data().
    Supports future/promise pattern and reference counting.

    Usage:
        # Initialize at server startup
        CacheManager.initialize(CacheConfig())

        # In resolve_chunk_data
        entry = manager.get_or_acquire(key, compute_fn)
        # ... use entry.data ...
        manager.release(key)
    """

    _instance: Optional[CacheManager] = None
    _init_lock = threading.Lock()

    def __init__(self, config: CacheConfig):
        """Initialize with CacheConfig, selecting backend based on config.backend."""
        if config.backend == "memory":
            self._backend = MemoryCacheBackend(
                MemoryCacheConfig(
                    max_entries=config.memory_max_entries,
                    max_bytes=config.memory_max_bytes,
                )
            )
        elif config.backend == "file":
            self._backend = ArrowFileBackend(
                ArrowFileConfig(
                    cache_dir=config.file_cache_dir,
                    max_segment_bytes=config.file_max_segment_bytes,
                    max_total_bytes=config.file_max_total_bytes,
                )
            )
        else:
            raise ValueError(f"Unknown cache backend: {config.backend}")

    @classmethod
    def initialize(cls, config: CacheConfig) -> CacheManager:
        """Initialize singleton (thread-safe)."""
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = cls(config)
            return cls._instance

    @classmethod
    def get_instance(cls) -> Optional[CacheManager]:
        """Get singleton instance."""
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if initialized."""
        return cls._instance is not None

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None

    @property
    def backend(self) -> CacheBackend:
        """Get the underlying backend."""
        return self._backend

    def get_or_acquire(
        self,
        key: bytes,
        compute_fn: Callable[[], Tuple[pa.RecordBatch, int]],
    ) -> CacheEntry:
        """Get or compute entry with future/promise pattern.

        Entry is acquired (ref_count >= 1). Caller must call release()
        after using the data.

        Args:
            key: Cache key bytes
            compute_fn: Returns (RecordBatch, size_bytes)

        Returns:
            CacheEntry with state READY, ref_count >= 1
        """
        return self._backend.get_or_acquire(key, compute_fn)

    def put(
        self,
        key: bytes,
        data: pa.RecordBatch,
        size_bytes: int,
    ) -> bool:
        """Store an already-computed batch. Returns True if this call stored it.

        The write-side counterpart of :meth:`get_or_acquire`, for a caller that
        holds the batch already and has nothing to compute -- an upload. It
        reserves the entry, commits it, and drops the reference the reservation
        took, all in one call.

        That last step is why this exists rather than a caller driving the
        backend's reserve/commit pair itself (biopb/biopb#545). The reservation's
        reference belongs to the promise protocol, not to the caller: it is what
        stops the entry being evicted mid-write. Leaving it held pins the batch in
        memory for the life of the process and makes the entry permanently
        un-removable, which is exactly what ``ArrowFileBackend.release()`` exists
        to prevent.

        A False return means the key was already present or another writer is
        mid-flight, so this batch is dropped rather than overwriting -- an upload
        that must not collide with a previous one gets a fresh cache namespace
        from its ``content_version`` (biopb/biopb#178), not from an overwrite.

        Args:
            key: Cache key bytes
            data: The batch to store
            size_bytes: Size of data in bytes
        """
        _entry, is_owner = self._backend.start_compute(key)
        try:
            if is_owner:
                self._backend.complete_entry(key, data, size_bytes)
        except BaseException as e:
            # A failed commit must not strand a PENDING entry: readers of this
            # key would block on it until pending_timeout.
            if is_owner:
                err = e if isinstance(e, Exception) else RuntimeError(repr(e))
                self._backend.fail_entry(key, err)
            raise
        finally:
            self._backend.release(key)
        return is_owner

    def release(self, key: bytes) -> int:
        """Release reference to entry after use."""
        return self._backend.release(key)

    def remove(self, key: bytes) -> bool:
        """Remove entry (only if evictable)."""
        return self._backend.remove(key)

    def locate_entry(self, key: bytes) -> Optional[ChunkLocation]:
        """Return the on-disk ChunkLocation for a cached chunk, or None.

        Only the file backend can locate entries on disk (issue #9); the memory
        backend inherits the interface's None default and the caller falls back
        to do_get.
        """
        return self._backend.locate_entry(key)

    def clear(self) -> None:
        """Clear all evictable entries."""
        self._backend.clear()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._backend.stats()

    def release_process_lock(self) -> None:
        """Release the cross-process cache lock + clear the WAL, handles left open.

        Delegates to the backend's fast graceful-shutdown path (no-op on the
        memory backend). Callers guard the singleton for ``None`` themselves
        (``CacheManager.get_instance()``).
        """
        self._backend.release_process_lock()

    def close(self) -> None:
        """Close manager."""
        self._backend.close()
