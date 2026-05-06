"""Thread-safe cache backend interface with future/promise pattern.

Designed for Arrow Flight servers where multiple threads may request
the same virtual chunk simultaneously. The future/promise pattern ensures
only one thread computes while others wait.

Reference counting prevents eviction of entries being actively served.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

import pyarrow as pa

# Arrow IPC has ~2GB batch size limit due to 32-bit offsets
# Use a safe threshold slightly below the actual limit
MAX_ARROW_BATCH_BYTES = 2 * 1024 * 1024 * 1024 - 1  # ~2GB


class EntryState(Enum):
    """State of a cache entry."""
    PENDING = "pending"      # Being computed, other threads should wait
    READY = "ready"          # Computed and available
    ERROR = "error"          # Computation failed


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
        metadata: Additional metadata
    """
    data: Optional[pa.RecordBatch] = None
    state: EntryState = EntryState.PENDING
    event: threading.Event = field(default_factory=threading.Event)
    error: Optional[Exception] = None
    ref_count: int = 0
    created_at: float = 0.0
    size_bytes: int = 0
    metadata: dict = field(default_factory=dict)

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


class CacheBackend(ABC):
    """Thread-safe cache backend with future/promise pattern.

    All implementations must support:
    - Pending entries for compute-once semantics
    - Reference counting to prevent eviction of active entries
    - Thread-safe operations

    The key method is get_or_compute() which implements the future/promise
    pattern for safe concurrent access.
    """

    @abstractmethod
    def get_or_acquire(
        self,
        key: bytes,
        compute_fn: Callable[[], Tuple[pa.RecordBatch, int]],
        metadata: Optional[dict] = None,
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
            metadata: Optional metadata for new entries

        Returns:
            CacheEntry with ref_count >= 1, state READY
        """
        pass

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
        pass

    @abstractmethod
    def fail_entry(self, key: bytes, error: Exception) -> None:
        """Mark a pending entry as failed.

        Args:
            key: Cache key bytes
            error: Exception that occurred during computation
        """
        pass

    @abstractmethod
    def release(self, key: bytes) -> int:
        """Release a reference to an entry.

        Args:
            key: Cache key bytes

        Returns:
            New reference count
        """
        pass

    @abstractmethod
    def remove(self, key: bytes) -> bool:
        """Remove entry (only if ref_count <= 0).

        Args:
            key: Cache key bytes

        Returns:
            True if removed, False if not found or has references
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all evictable entries (ref_count <= 0)."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close backend and release resources."""
        pass


def get_or_compute_with_context(
    backend: CacheBackend,
    key: bytes,
    compute_fn: Callable[[], Tuple[pa.RecordBatch, int]],
    metadata: Optional[dict] = None,
) -> Tuple[CacheEntry, bool]:
    """Helper for get-or-compute that returns entry and owns_compute flag.

    This allows the caller to know whether they should call complete_entry()
    or fail_entry() if computation fails.

    Args:
        backend: Cache backend
        key: Cache key bytes
        compute_fn: Compute function
        metadata: Optional metadata

    Returns:
        (CacheEntry, is_owner) where is_owner indicates if caller should
        call complete_entry/fail_entry
    """
    pass  # Implementation in concrete backend