"""Thread-safe cache manager for Flight server.

Singleton manager that coordinates cache operations across all
resolve_chunk_data calls. Uses future/promise pattern for
safe concurrent computation of virtual chunks.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

import pyarrow as pa

from biopb_tensor_server.cache.base import CacheBackend, CacheEntry, CacheStats
from biopb_tensor_server.cache.file_backend import ArrowFileBackend, ArrowFileConfig
from biopb_tensor_server.cache.memory_backend import (
    MemoryCacheBackend,
    MemoryCacheConfig,
)
from biopb_tensor_server.config import CacheConfig


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
        if config.backend == 'memory':
            self._backend = MemoryCacheBackend(MemoryCacheConfig(
                max_entries=config.memory_max_entries,
                max_bytes=config.memory_max_bytes,
            ))
        elif config.backend == 'file':
            self._backend = ArrowFileBackend(ArrowFileConfig(
                cache_dir=config.file_cache_dir,
                max_segment_bytes=config.file_max_segment_bytes,
                max_total_bytes=config.file_max_total_bytes,
            ))
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
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Get or compute entry with future/promise pattern.

        Entry is acquired (ref_count >= 1). Caller must call release()
        after using the data.

        Args:
            key: Cache key bytes
            compute_fn: Returns (RecordBatch, size_bytes)
            metadata: Optional metadata

        Returns:
            CacheEntry with state READY, ref_count >= 1
        """
        return self._backend.get_or_acquire(key, compute_fn, metadata)

    def start_compute(
        self,
        key: bytes,
        metadata: Optional[dict] = None,
    ) -> Tuple[CacheEntry, bool]:
        """Start compute - returns (entry, is_owner).

        Use this when you want to separate the "check cache" from "compute"
        phases. If is_owner=True, you must call complete_entry() or fail_entry().

        Args:
            key: Cache key bytes
            metadata: Optional metadata

        Returns:
            (CacheEntry, is_owner) - if is_owner, you must complete/fail
        """
        return self._backend.start_compute(key, metadata)

    def complete_entry(
        self,
        key: bytes,
        data: pa.RecordBatch,
        size_bytes: int,
    ) -> None:
        """Complete a pending entry (you were the compute owner)."""
        self._backend.complete_entry(key, data, size_bytes)

    def fail_entry(self, key: bytes, error: Exception) -> None:
        """Fail a pending entry."""
        self._backend.fail_entry(key, error)

    def release(self, key: bytes) -> int:
        """Release reference to entry after use."""
        return self._backend.release(key)

    def remove(self, key: bytes) -> bool:
        """Remove entry (only if evictable)."""
        return self._backend.remove(key)

    def clear(self) -> None:
        """Clear all evictable entries."""
        self._backend.clear()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._backend.stats()

    def close(self) -> None:
        """Close manager."""
        self._backend.close()