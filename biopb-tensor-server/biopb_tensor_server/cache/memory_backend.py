"""Thread-safe in-memory LRU cache with future/promise pattern.

Implements:
- Future/promise: First request creates pending entry, others wait
- Reference counting: Entries with ref_count > 0 cannot be evicted
- LRU with size-aware eviction: Prefer evicting smaller entries from LRU half
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pyarrow as pa

from biopb_tensor_server.cache.base import (
    MAX_ARROW_BATCH_BYTES,
    CacheBackend,
    CacheEntry,
    CacheStats,
    EntryState,
)


@dataclass
class MemoryCacheConfig:
    """Configuration for in-memory cache."""

    max_entries: int = 1024
    max_bytes: int = 512 * 1024 * 1024  # 512 MB
    pending_timeout: float = 300.0  # Max wait time for pending entries


class MemoryCacheBackend(CacheBackend):
    """Thread-safe in-memory cache with future/promise pattern.

    Key features:
    1. Future/Promise: When a key is requested but not cached, first thread
       creates a PENDING entry and computes. Other threads find the pending
       entry and wait on its event.
    2. Reference counting: Entries with ref_count > 0 cannot be evicted.
       This prevents evicting data being actively served to Flight clients.
    3. Size-aware LRU: Prefer evicting smaller entries from LRU half.
    """

    def __init__(self, config: MemoryCacheConfig):
        self._config = config
        # OrderedDict for LRU ordering: oldest at front, newest at end
        self._entries: OrderedDict[bytes, CacheEntry] = OrderedDict()
        self._total_bytes: int = 0
        self._stats_lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._pending_waits: int = 0
        self._ref_held_skips: int = 0
        self._oversized_skips: int = 0
        # Main lock protects _entries and all operations
        self._lock = threading.Lock()

    def _estimate_size(self, data: pa.RecordBatch) -> int:
        """Estimate size in bytes."""
        return sum(col.nbytes for col in data.columns)

    def _evict_if_needed(self, needed_bytes: int) -> None:
        """Evict entries if adding needed_bytes would exceed limits.

        Only evicts entries with ref_count <= 0 (not actively referenced).
        """
        while self._entries:
            # Check entry count limit
            if len(self._entries) >= self._config.max_entries:
                if not self._evict_one_evictable():
                    # Can't evict anything (all have refs), stop trying
                    break
                continue

            # Check byte limit
            if self._total_bytes + needed_bytes <= self._config.max_bytes:
                return

            # Need to evict to make room
            if not self._evict_one_evictable():
                break

    def _evict_one_evictable(self) -> bool:
        """Evict one evictable entry from LRU half.

        Returns True if evicted, False if nothing evictable.
        """
        # Find evictable entries in LRU half
        half_size = len(self._entries) // 2
        if half_size == 0:
            # Very small cache - check if first entry is evictable
            first_key = next(iter(self._entries))
            if self._entries[first_key].is_evictable():
                self._do_evict(first_key)
                return True
            self._ref_held_skips += 1
            return False

        # Get LRU half keys
        lru_keys = list(self._entries.keys())[:half_size]

        # Find smallest evictable entry in LRU half
        evictable = [
            (k, self._entries[k].size_bytes)
            for k in lru_keys
            if self._entries[k].is_evictable()
        ]

        if not evictable:
            self._ref_held_skips += 1
            return False

        # Evict smallest
        smallest_key = min(evictable, key=lambda x: x[1])[0]
        self._do_evict(smallest_key)
        return True

    def _do_evict(self, key: bytes) -> None:
        """Actually evict an entry."""
        entry = self._entries.pop(key)
        self._total_bytes -= entry.size_bytes
        self._evictions += 1

    def get_or_acquire(
        self,
        key: bytes,
        compute_fn: Callable[[], Tuple[pa.RecordBatch, int]],
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Get existing entry or create pending and compute.

        This implements the future/promise pattern:
        1. Check if entry exists (READY) -> acquire, return
        2. Check if entry PENDING -> wait for it, then acquire, return
        3. No entry -> create PENDING, compute, complete, acquire, return

        Returns entry with ref_count >= 1 (acquired).
        """
        is_owner = False
        with self._lock:
            entry = self._entries.get(key)

            if entry is not None:
                if entry.state == EntryState.READY:
                    # Ready entry - acquire and return
                    entry.acquire()
                    self._move_to_end(key)
                    self._hits += 1
                    return entry

                if entry.state == EntryState.PENDING:
                    # Pending - wait outside lock (another thread is computing)
                    self._pending_waits += 1

            else:
                # No entry - create pending, we own the computation
                entry = CacheEntry(
                    state=EntryState.PENDING,
                    created_at=time.time(),
                    metadata=metadata or {},
                )
                self._entries[key] = entry
                self._misses += 1
                is_owner = True

        # If we created the pending entry, we must compute
        if is_owner:
            try:
                data, size_bytes = compute_fn()
                self.complete_entry(key, data, size_bytes)
            except Exception as e:
                self.fail_entry(key, e)
                raise

        # If pending (either waiting on another thread or we just completed), wait
        if entry.state == EntryState.PENDING:
            if not entry.wait_ready(self._config.pending_timeout):
                raise TimeoutError("Cache computation timed out for key")

        # Now acquire and return
        with self._lock:
            entry.acquire()
            if entry.state == EntryState.READY:
                self._move_to_end(key)
            return entry

    def get_or_compute(
        self,
        key: bytes,
        compute_fn: Callable[[], Tuple[pa.RecordBatch, int]],
        metadata: Optional[dict] = None,
    ) -> pa.RecordBatch:
        """Convenience method: get_or_acquire then return data.

        Auto-releases on error.
        """
        entry = self.get_or_acquire(key, compute_fn, metadata)
        return entry.data

    def start_compute(
        self,
        key: bytes,
        metadata: Optional[dict] = None,
    ) -> Tuple[CacheEntry, bool]:
        """Start compute phase - returns (entry, is_owner).

        is_owner=True means this thread should compute and call complete_entry.
        is_owner=False means another thread is computing, just wait.

        The returned entry is ALWAYS acquired (ref_count >= 1).

        Args:
            key: Cache key bytes
            metadata: Optional metadata for new entries

        Returns:
            (CacheEntry, is_owner) - is_owner indicates if caller owns computation
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is not None:
                if entry.state == EntryState.READY:
                    entry.acquire()
                    self._move_to_end(key)
                    self._hits += 1
                    return entry, False  # Already ready, not owner

                if entry.state == EntryState.PENDING:
                    # Someone else computing - acquire and wait
                    entry.acquire()
                    self._pending_waits += 1
                    return entry, False  # Not owner, must wait

            # No entry - create pending, we own computation
            entry = CacheEntry(
                state=EntryState.PENDING,
                created_at=time.time(),
                metadata=metadata or {},
            )
            entry.acquire()  # Acquire for compute owner
            self._entries[key] = entry
            self._misses += 1
            return entry, True  # We own the computation

    def complete_entry(
        self,
        key: bytes,
        data: pa.RecordBatch,
        size_bytes: int,
    ) -> None:
        """Mark pending entry as ready."""
        # Check for oversized chunks
        if size_bytes > MAX_ARROW_BATCH_BYTES:
            self._oversized_skips += 1
            # Log warning (import at top level to avoid circular import issues)
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Skipping cache for oversized chunk: {size_bytes} bytes > {MAX_ARROW_BATCH_BYTES}"
            )
            # Still mark as ready so waiting threads get data
            with self._lock:
                entry = self._entries.get(key)
                if entry is None or entry.state != EntryState.PENDING:
                    return
                entry.set_ready(data, size_bytes)
            return

        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state != EntryState.PENDING:
                return  # Already completed or removed

            # Evict if needed before storing
            self._evict_if_needed(size_bytes)

            entry.set_ready(data, size_bytes)
            self._total_bytes += size_bytes
            self._move_to_end(key)

    def fail_entry(self, key: bytes, error: Exception) -> None:
        """Mark pending entry as failed."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state != EntryState.PENDING:
                return

            entry.set_error(error)
            # Remove failed entry from cache
            self._entries.pop(key, None)

    def release(self, key: bytes) -> int:
        """Release reference to entry."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            return entry.release()

    def remove(self, key: bytes) -> bool:
        """Remove entry only if evictable."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if not entry.is_evictable():
                return False
            self._do_evict(key)
            return True

    def clear(self) -> None:
        """Clear all evictable entries."""
        with self._lock:
            to_remove = [k for k, e in self._entries.items() if e.is_evictable()]
            for key in to_remove:
                self._do_evict(key)

    def _move_to_end(self, key: bytes) -> None:
        """Move entry to end (most recently used)."""
        if key in self._entries:
            self._entries.move_to_end(key)

    def stats(self) -> CacheStats:
        """Return cache statistics."""
        with self._stats_lock:
            return CacheStats(
                total_entries=len(self._entries),
                total_bytes=self._total_bytes,
                max_entries=self._config.max_entries,
                max_bytes=self._config.max_bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                pending_waits=self._pending_waits,
                ref_held_evictions_skipped=self._ref_held_skips,
                oversized_skips=self._oversized_skips,
            )

    def close(self) -> None:
        """Close and release resources."""
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0
