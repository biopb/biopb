"""The cache's value types: what an entry is, what it costs, what it reports.

Designed for Arrow Flight servers where multiple threads may request the same
virtual chunk simultaneously. ``CacheEntry`` carries the future/promise state
that makes that safe -- one thread computes while others wait on its event --
and the reference count that keeps an entry being served from being evicted.

The types only; the protocol that drives them lives on ``ArrowFileBackend``
(``file_backend.py``), which owns storage, eviction, and the boot and shutdown
paths, with ``CacheManager`` as the handle the server holds.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Optional

import pyarrow as pa

# Chunk splitting threshold - 64MB for parallel Flight transfers
# (Arrow IPC can handle larger, but we split for throughput optimization)
MAX_ARROW_BATCH_BYTES = 64 * 1024 * 1024


# What a miss costs. Eviction on the file backend is whole-segment, so this
# cannot be consulted when a victim is chosen -- it picks the segment a chunk is
# written into, and reclamation prefers the cheap ones.
#
#   cheap  -- derived data: a scaled chunk reduced from full-resolution chunks
#             that regenerate it, and that it cannot regenerate.
#   normal -- the default: a source read and a decode.
#   pinned -- never evicted. No producer yet: an upload is the only data with no
#             source to re-read, but nothing deletes a cache entry, so pinning
#             one would leave the cache no way back under its budget.
RetentionClass = Literal["cheap", "normal", "pinned"]

# Reclamation order, lowest first. A class absent from this map is never
# selected for eviction, which is what makes "pinned" pinned.
EVICTION_RANK: Dict[RetentionClass, int] = {"cheap": 0, "normal": 1}


def estimate_batch_bytes(batch: pa.RecordBatch) -> int:
    """In-memory size of a cached batch: the sum of its columns' buffers.

    One definition shared by every backend and every index path, so a chunk's
    recorded size cannot depend on which code path recorded it.
    """
    return sum(col.nbytes for col in batch.columns)


@dataclass
class ChunkLocation:
    """On-disk location of a cached chunk's Arrow IPC message.

    Returned by :meth:`ArrowFileBackend.locate_entry` for the localhost
    cache-file handoff (issue #9): a client on the same host mmaps
    ``segment_path`` and reads the single record-batch message at
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
        retention: What a miss for this entry would cost (see RetentionClass)
    """

    data: Optional[pa.RecordBatch] = None
    state: EntryState = EntryState.PENDING
    event: threading.Event = field(default_factory=threading.Event)
    error: Optional[Exception] = None
    ref_count: int = 0
    created_at: float = 0.0
    size_bytes: int = 0
    retention: RetentionClass = "normal"

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

    pool_key: str  # e.g., "cheap-bulk" (retention class + size class)
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
    deferred_write_bytes: int = 0  # Committed from memory, not yet on disk
    # Deferred writes that never reached disk. Invisible otherwise because the caller
    # was released before the write was attempted.
    deferred_write_failures: int = 0
    pool_stats: Dict[str, PoolStats] = field(default_factory=dict)
