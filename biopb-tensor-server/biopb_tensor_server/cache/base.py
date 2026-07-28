"""Thread-safe cache backend interface with future/promise pattern.

Designed for Arrow Flight servers where multiple threads may request
the same virtual chunk simultaneously. The future/promise pattern ensures
only one thread computes while others wait.

Reference counting prevents eviction of entries being actively served.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import pyarrow as pa

logger = logging.getLogger(__name__)

# Chunk splitting threshold - 64MB for parallel Flight transfers
# (Arrow IPC can handle larger, but we split for throughput optimization)
MAX_ARROW_BATCH_BYTES = 64 * 1024 * 1024


def estimate_batch_bytes(batch: pa.RecordBatch) -> int:
    """In-memory size of a cached batch: the sum of its columns' buffers.

    One definition shared by every backend and every index path, so a chunk's
    recorded size cannot depend on which code path recorded it.
    """
    return sum(col.nbytes for col in batch.columns)


@dataclass
class ChunkLocation:
    """On-disk location of a cached chunk's Arrow IPC message.

    Returned by :meth:`CacheBackend.locate_entry` for the localhost cache-file
    handoff (issue #9): a client on the same host mmaps ``segment_path`` and
    reads the single record-batch message at
    ``[byte_offset, byte_offset + byte_length)``. ``generation_id`` is the
    segment file's inode at locate time, so a client can detect a segment that
    was evicted and recreated at the same path.
    """

    segment_path: str
    byte_offset: int
    byte_length: int
    generation_id: int


class EntryState(Enum):
    """State of a cache entry."""

    PENDING = "pending"  # Being computed, other threads should wait
    READY = "ready"  # Computed and available
    ERROR = "error"  # Computation failed


@dataclass
class CacheEntry:
    """Cached chunk entry with reference counting.

    Attributes:
        data: RecordBatch (None if pending/error)
        state: Entry state (pending, ready, error)
        event: Threading event for pending entries to wait on
        error: Exception if state is ERROR
        ref_count: Number of active references (prevents eviction)
        created_at: Creation timestamp
        size_bytes: Data size in bytes
    """

    data: Optional[pa.RecordBatch] = None
    state: EntryState = EntryState.PENDING
    event: threading.Event = field(default_factory=threading.Event)
    error: Optional[Exception] = None
    ref_count: int = 0
    created_at: float = 0.0
    size_bytes: int = 0

    def acquire(self) -> None:
        """Increment reference count to prevent eviction."""
        self.ref_count += 1

    def release(self) -> int:
        """Decrement reference count. Returns new count."""
        self.ref_count -= 1
        return self.ref_count

    def is_evictable(self) -> bool:
        """Check if entry can be evicted (no active references)."""
        return self.ref_count <= 0 and self.state == EntryState.READY

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait for entry to become ready.

        Returns True if ready, False if timeout, raises if error.
        """
        if self.event.wait(timeout):
            if self.state == EntryState.ERROR:
                raise self.error
            return self.state == EntryState.READY
        return False

    def set_ready(self, data: pa.RecordBatch, size_bytes: int) -> None:
        """Mark entry as ready with computed data."""
        self.data = data
        self.size_bytes = size_bytes
        self.state = EntryState.READY
        self.event.set()

    def set_error(self, error: Exception) -> None:
        """Mark entry as error, wake waiting threads."""
        self.error = error
        self.state = EntryState.ERROR
        self.event.set()


@dataclass
class PoolStats:
    """Per-pool cache statistics."""

    pool_key: str  # e.g., "unified-tiny"
    hits: int = 0
    misses: int = 0
    segments: int = 0
    bytes: int = 0
    hit_rate: float = 0.0


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int = 0
    total_bytes: int = 0
    max_entries: int = 0
    max_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    pending_waits: int = 0  # Threads that waited on pending entries
    ref_held_evictions_skipped: int = 0  # Evictions skipped due to ref_count
    oversized_skips: int = 0  # Chunks skipped due to exceeding Arrow batch size limit
    pool_stats: Dict[str, PoolStats] = field(default_factory=dict)


class CacheBackend(ABC):
    """Thread-safe cache backend with future/promise pattern.

    All implementations must support:
    - Pending entries for compute-once semantics
    - Reference counting to prevent eviction of active entries
    - Thread-safe operations

    The key method is get_or_acquire() which implements the future/promise
    pattern for safe concurrent access.
    """

    @abstractmethod
    def get_or_acquire(
        self,
        key: bytes,
        compute_fn: Callable[[], Tuple[pa.RecordBatch, int]],
    ) -> CacheEntry:
        """Get existing entry or create pending entry for computation.

        This is the primary entry point for the future/promise pattern:
        - If entry exists (READY): acquire and return it
        - If entry PENDING: wait for it to become ready, then acquire
        - If entry missing: create PENDING entry, caller should compute
          then call complete_entry()

        The returned entry has ref_count incremented (acquired).

        Args:
            key: Cache key bytes
            compute_fn: Function to compute data, returns (RecordBatch, size_bytes)

        Returns:
            CacheEntry with ref_count >= 1, state READY
        """

    @abstractmethod
    def start_compute(self, key: bytes) -> Tuple[CacheEntry, bool]:
        """Reserve the key without computing: returns (entry, is_owner).

        The check-cache half of get_or_acquire, split out for a caller that
        already holds the data (CacheManager.put) rather than a compute_fn. The
        returned entry is acquired either way; is_owner=True means it is PENDING
        and this caller must complete_entry() or fail_entry() it.

        Args:
            key: Cache key bytes

        Returns:
            (CacheEntry with ref_count >= 1, is_owner)
        """

    @abstractmethod
    def complete_entry(
        self,
        key: bytes,
        data: pa.RecordBatch,
        size_bytes: int,
    ) -> None:
        """Mark a pending entry as ready with computed data.

        Called by the thread that performed computation after get_or_acquire
        returned a PENDING entry with this thread as the "owner".

        Args:
            key: Cache key bytes
            data: Computed RecordBatch
            size_bytes: Size of data in bytes
        """

    @abstractmethod
    def fail_entry(self, key: bytes, error: Exception) -> None:
        """Mark a pending entry as failed.

        Args:
            key: Cache key bytes
            error: Exception that occurred during computation
        """

    @abstractmethod
    def release(self, key: bytes) -> int:
        """Release a reference to an entry.

        Args:
            key: Cache key bytes

        Returns:
            New reference count
        """

    @abstractmethod
    def remove(self, key: bytes) -> bool:
        """Remove entry (only if ref_count <= 0).

        Args:
            key: Cache key bytes

        Returns:
            True if removed, False if not found or has references
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all evictable entries (ref_count <= 0)."""

    @abstractmethod
    def stats(self) -> CacheStats:
        """Return current cache statistics."""

    @abstractmethod
    def close(self) -> None:
        """Close backend and release resources."""

    def release_process_lock(self) -> None:  # noqa: B027 - concrete no-op default, not abstract
        """Release the cross-process cache lock (+ WAL) without closing handles.

        The graceful-shutdown fast path (biopb/biopb#300): release the lock
        *first*, cheaply and upstream-independently, while segment writers/mmaps
        stay open for any still-draining reads. Backends with no process lock
        (memory) inherit this no-op default; the file backend overrides it.
        """

    def locate_entry(self, key: bytes) -> Optional[ChunkLocation]:
        """Return the on-disk location of a cached chunk, or None.

        Backs the localhost cache-file handoff (issue #9), where a same-host
        client mmaps the segment instead of streaming the chunk over do_get.
        Declared here -- rather than sniffed with ``getattr`` -- because the
        manager drives it on *every* backend, so a backend that cannot locate
        (the memory backend has no segment files) should say so through the
        interface. None means "fall back to do_get", the designed floor of the
        whole path.
        """
        return None

    def _skip_if_oversized(
        self, key: bytes, data: pa.RecordBatch, size_bytes: int
    ) -> bool:
        """Handle a chunk too large to cache; True if the caller should stop.

        An oversized chunk is still handed to the threads waiting on it -- the
        entry goes READY in memory -- it just is not stored. Shared by both
        backends, which supply the ``_lock`` / ``_entries`` / ``_oversized_skips``
        state this reads.
        """
        if size_bytes <= MAX_ARROW_BATCH_BYTES:
            return False
        self._oversized_skips += 1
        logger.warning(
            "Skipping cache for oversized chunk: %d bytes > %d",
            size_bytes,
            MAX_ARROW_BATCH_BYTES,
        )
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.state == EntryState.PENDING:
                entry.set_ready(data, size_bytes)
        return True
