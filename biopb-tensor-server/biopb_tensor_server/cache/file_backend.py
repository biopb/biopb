"""Thread-safe persistent Arrow file cache backend.

Implements segmented storage with:
- Mmap reads for near-memory-speed access
- Segment-level LRU eviction
- Crash recovery via WAL and process lock
- Same future/promise pattern as MemoryCacheBackend
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
import pyarrow as pa

from biopb_tensor_server.cache.base import (
    MAX_ARROW_BATCH_BYTES,
    CacheBackend,
    CacheEntry,
    CacheStats,
    EntryState,
    PoolStats,
)
from biopb_tensor_server.cache.recovery import (
    PoolQueueInfo,
    ProcessLock,
    RecoveryStatus,
    SegmentEntryInfo,
    SegmentInfo,
    SieveKSegmentInfo,
    WriteAheadLog,
)

logger = logging.getLogger(__name__)


# Sieve-K constants
K = 2  # Counter saturates at K (levels: 0, 1, 2)
COLD_THRESHOLD_SECONDS = 300  # 5 minutes without access
COLD_FREQUENCY_THRESHOLD = 0  # frequency == 0
MMAP_LIFECYCLE_THRESHOLD = 100  # Only manage mmaps when segments > 100


# Size class thresholds for pooling (fixed values, not derived from MAX_ARROW_BATCH_BYTES)
# With 64MB max chunk size, we want reasonable pooling buckets
SIZE_CLASS_TINY_THRESHOLD = 1 * 1024 * 1024  # <1MB
SIZE_CLASS_SMALL_THRESHOLD = 8 * 1024 * 1024  # 1-8MB
SIZE_CLASS_MEDIUM_THRESHOLD = 32 * 1024 * 1024  # 8-32MB
# large: >=32MB (still cached, just pooled separately)

SizeClass = Literal["tiny", "small", "medium", "large"]


def _get_size_class(size_bytes: int) -> SizeClass:
    """Classify chunk size for pooling."""
    if size_bytes < SIZE_CLASS_TINY_THRESHOLD:
        return "tiny"
    elif size_bytes < SIZE_CLASS_SMALL_THRESHOLD:
        return "small"
    elif size_bytes < SIZE_CLASS_MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "large"


# =============================================================================
# Buffer-based casting helpers (bypass validation for performance)
# =============================================================================

# Unified schema for all dtypes - enables single cache pool per size_class
UNIFIED_SCHEMA = pa.schema(
    [
        pa.field("data", pa.binary()),
        pa.field("shape", pa.list_(pa.int64())),
        pa.field("dtype", pa.string()),
    ]
)

# Name of the per-batch column carrying the entry's cache key. The key MUST
# travel as a column value, not as schema metadata: an Arrow IPC stream
# serializes the schema exactly once (taken from the first batch written to the
# segment), so per-batch schema metadata is lost on read-back and every batch
# would report the first entry's key. A column value is stored per row and
# round-trips correctly. See _rebuild_index_from_segments.
CACHE_KEY_FIELD = "__biopb_cache_key__"

# On-disk segment format version for the localhost cache-file handoff (issue
# #9). The client mmaps and parses segment messages directly, so the layout is
# a cross-process contract. Bump this whenever the segment message layout or the
# data/shape/dtype/cache_key encoding changes in a way an older client can't
# parse; the server reports it in chunk_locate and a client declines the fast
# path (falls back to do_get) for any version it doesn't understand.
CACHE_FILE_FORMAT_VERSION = 1


def _cast_to_unified_schema(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Cast typed batch to unified binary schema for caching.

    Uses buffer-based casting to bypass Arrow's logical type validation.

    Args:
        batch: RecordBatch with schema [data: list<dtype>, shape: list<int64>, dtype: string]

    Returns:
        RecordBatch with unified schema [data: binary, shape: list<int64>, dtype: string]
    """
    data_col = batch.column("data")

    # Get buffers from the list array
    # ListArray: buffers = [validity, offsets], children = [values]
    # Values: primitive array with buffers = [validity, data]
    values = data_col.values

    # Create binary array from values buffer
    # Binary needs: [validity, offsets, data]
    # For single binary blob: offsets = [0, N]
    values_buf = values.buffers()[1]
    num_bytes = values_buf.size
    offsets_buf = pa.py_buffer(np.array([0, num_bytes], dtype=np.int32))

    binary_arr = pa.Array.from_buffers(
        pa.binary(),
        1,  # one row (single binary blob)
        [None, offsets_buf, values_buf],
    )

    return pa.RecordBatch.from_arrays(
        [binary_arr, batch.column("shape"), batch.column("dtype")],
        ["data", "shape", "dtype"],
    )


def _cast_from_unified_schema(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Cast unified binary batch back to typed schema for client consumption.

    Uses buffer-based casting to bypass Arrow's logical type validation.

    Args:
        batch: RecordBatch with unified schema [data: binary, shape: list<int64>, dtype: string]

    Returns:
        RecordBatch with typed schema [data: list<dtype>, shape: list<int64>, dtype: string]
    """
    dtype_str = batch.column("dtype").to_pylist()[0]
    dtype = np.dtype(dtype_str)
    arrow_dtype = pa.from_numpy_dtype(dtype)

    binary_col = batch.column("data")

    # Binary array: buffers = [validity, offsets, data]
    binary_bufs = binary_col.buffers()
    data_buf = binary_bufs[2]  # the actual bytes

    # Number of elements from byte length
    num_elements = len(data_buf) // dtype.itemsize

    # Create values array from data buffer directly
    values_arr = pa.Array.from_buffers(arrow_dtype, num_elements, [None, data_buf])

    # Create list array: buffers = [validity, offsets], children = [values]
    # For single list containing all elements: offsets = [0, num_elements]
    list_offsets = pa.py_buffer(np.array([0, num_elements], dtype=np.int32))

    list_arr = pa.ListArray.from_buffers(
        pa.list_(arrow_dtype),
        1,  # one row
        [None, list_offsets],
        children=[values_arr],
    )

    return pa.RecordBatch.from_arrays(
        [list_arr, batch.column("shape"), batch.column("dtype")],
        ["data", "shape", "dtype"],
    )


def _copy_batch_off_mmap(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Return a copy of `batch` whose buffers are owned in RAM, not a file mmap.

    Reads from a segment are zero-copy: the returned batch's buffers (data,
    shape and dtype columns alike) point straight into the segment file's
    memory mapping. On POSIX that is harmless -- unlink() removes the directory
    entry and the inode survives to last close, so a segment can be evicted
    while a caller still holds the batch. On Windows an active mapping blocks
    deletion (WinError 32), so eviction and cleanup fail as long as any caller
    keeps the batch alive.

    This materializes the batch into pyarrow-owned memory via an IPC round-trip
    (which copies every column, not just the data buffer), severing the tie to
    the mmap so the segment file can be unlinked. Used only where that matters;
    see issue #5.
    """
    sink = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    reader = pa.RecordBatchStreamReader(pa.BufferReader(sink.getvalue()))
    return reader.read_next_batch()


@dataclass
class ArrowFileConfig:
    """Configuration for Arrow file cache backend."""

    cache_dir: Path
    max_segment_bytes: int = 64 * 1024 * 1024  # 64 MB per segment
    max_total_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GB total
    pending_timeout: float = 300.0  # Max wait time for pending entries

    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class ChunkLocation:
    """On-disk location of a cached chunk's Arrow IPC message.

    Returned by ``locate_entry`` for the localhost cache-file handoff (issue
    #9): a client on the same host mmaps ``segment_path`` and reads the single
    record-batch message at ``[byte_offset, byte_offset + byte_length)``.
    ``generation_id`` is the segment file's inode at locate time, so a client
    can detect a segment that was evicted and recreated at the same path.
    """

    segment_path: str
    byte_offset: int
    byte_length: int
    generation_id: int


class ArrowFileBackend(CacheBackend):
    """Thread-safe persistent file cache with future/promise pattern.

    Directory structure:
        cache_dir/
        ├── segments/
        │   ├── seg_0001.arrow
        │   ├── seg_0002.arrow
        │   └── ...
        ├── wal.json
        └── lock

    Key features:
    1. Future/Promise: Same pattern as MemoryCacheBackend
    2. Mmap reads: OS page cache provides near-memory performance
    3. Segment-level eviction: Delete least-recently-used segment
    4. Crash recovery: WAL detects incomplete writes
    """

    def __init__(self, config: ArrowFileConfig):
        self._config = config
        # ``_lock`` guards the in-memory index (``_entries``, ``_metadata``,
        # ``_pool_*``, ``_segment_mmaps``). It is held ONLY for short in-memory
        # mutations and must NEVER be held across a blocking disk write
        # (``write_batch``/``flush``/``writer.close``) -- doing so once wedged the
        # entire server's read path when a write stalled on a full filesystem
        # (the lock was orphaned when the RPC was torn down). Reads take only
        # ``_lock``, so they stay live even if a write stalls.
        self._lock = threading.Lock()
        # ``_write_lock`` serializes the segment-write critical section
        # (``complete_entry`` and the eviction/segment-close it drives -- the
        # sole mutators of writer/segment state). Readers never take it, so a
        # stalled or orphaned write can block only future writes, not reads.
        self._write_lock = threading.Lock()

        # In-memory entries for pending/ready/error state management
        # Only stores small metadata; actual data is in segment files
        self._entries: Dict[bytes, CacheEntry] = {}

        # Metadata index: key -> SegmentEntryInfo (location in segment)
        self._metadata: Dict[bytes, SegmentEntryInfo] = {}

        # Sieve-K: Per-pool queues with frequency counters
        # Replaces old `_segments: OrderedDict[int, SegmentInfo]`
        self._pool_queues: Dict[Tuple[str, SizeClass], PoolQueueInfo] = {}

        # Legacy segment tracking for backward compatibility during migration
        # Maps segment_id -> SegmentInfo for segments not yet in pool queues
        self._segments_legacy: Dict[int, SegmentInfo] = {}

        # Mmap handles for fast reads
        self._segment_mmaps: Dict[int, pa.MemoryMappedFile] = {}

        # Multiple active writers for pooling: segment_id -> writer
        # This allows keeping multiple segments open for different (schema, size_class) pools
        self._pool_writers: Dict[int, pa.RecordBatchStreamWriter] = {}
        # The OSFile sink behind each writer. RecordBatchStreamWriter.close()
        # does NOT close the sink it was handed, so we must track and close it
        # ourselves -- otherwise the write handle lingers until GC and (on
        # Windows) blocks segment unlink during eviction/cleanup. See issue #5.
        self._pool_sinks: Dict[int, pa.OSFile] = {}
        self._pool_paths: Dict[int, Path] = {}
        self._pool_schemas: Dict[int, pa.Schema] = {}

        # Pool tracking: (schema_key, size_class) -> segment_id for open segments
        self._open_pools: Dict[Tuple[str, SizeClass], int] = {}

        # Double-buffered rotation: pre-created next segment for each pool
        self._next_segment_map: Dict[Tuple[str, SizeClass], int] = {}

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._pending_waits: int = 0
        self._ref_held_skips: int = 0
        self._oversized_skips: int = 0

        # Access counter for periodic mmap cleanup
        self._access_counter: int = 0

        # WAL and process lock
        self._wal: Optional[WriteAheadLog] = None
        self._process_lock: Optional[ProcessLock] = None

        # Recovery status (if recovered from crash)
        self._recovery_status: Optional[RecoveryStatus] = None

        # Copy batches off the segment mmap before returning them to callers.
        # Only needed on Windows, where a live mapping blocks segment unlink
        # during eviction/cleanup; POSIX keeps zero-copy reads (issue #5).
        self._copy_on_read = sys.platform == "win32"

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize backend: create dirs, acquire lock, rebuild index."""
        # Create directories. Restrict the cache root to the owner (0o700):
        # segment files hold decoded chunk payloads and their paths are handed
        # to localhost clients (issue #9), so other users on a shared host must
        # not be able to read them. (mode is masked by umask; re-assert with
        # chmod. No-op hardening on Windows, which ignores POSIX modes.)
        segments_dir = self._config.cache_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._config.cache_dir, 0o700)
        except OSError:
            pass

        # Acquire process lock (detect stale lock for crash recovery)
        lock_path = self._config.cache_dir / "lock"
        self._process_lock = ProcessLock(lock_path)

        # Capture staleness BEFORE acquire() removes the stale lock file
        was_stale = self._process_lock.is_stale()

        if not self._process_lock.acquire():
            raise RuntimeError(
                f"Cannot acquire cache lock at {lock_path}. "
                "Another process is using the cache."
            )

        # Initialize WAL
        wal_path = self._config.cache_dir / "wal.json"
        self._wal = WriteAheadLog(wal_path)

        # Check for crash recovery
        if was_stale or self._wal.has_pending():
            logger.info("Cache recovery: stale lock or pending WAL entries detected")
            self._recovery_status = self._recover()

        # Rebuild metadata index from segment files
        self._rebuild_index_from_segments()

        # Find next segment ID (segments are created lazily when first write happens)
        all_segment_ids = set()
        for pool in self._pool_queues.values():
            all_segment_ids.update(pool.queue)
        all_segment_ids.update(self._segments_legacy.keys())
        self._next_segment_id = max(all_segment_ids, default=0) + 1

    def _recover(self) -> RecoveryStatus:
        """Recover from crash: clean up incomplete writes."""
        errors = []
        recovered_entries = 0
        lost_entries = 0
        recovered_bytes = 0
        lost_bytes = 0

        # Get pending keys from WAL
        pending_keys = self._wal.get_pending_keys()

        # Clear WAL - incomplete writes are lost
        for key in pending_keys:
            lost_entries += 1
            logger.warning(f"Cache recovery: lost pending write for key {key.hex()}")
        self._wal.clear()

        # Scan existing segments - valid entries survive
        segments_dir = self._config.cache_dir / "segments"
        for seg_file in segments_dir.glob("seg_*.arrow"):
            try:
                # Just count what's recoverable - actual rebuild happens next
                with pa.memory_map(str(seg_file), "r") as mmap:
                    reader = pa.RecordBatchStreamReader(mmap)
                    for batch in reader:
                        recovered_entries += 1
                        recovered_bytes += batch.nbytes
            except Exception as e:
                errors.append(f"Error reading {seg_file}: {e}")
                logger.error(f"Cache recovery error: {e}")

        return RecoveryStatus(
            recovered_entries=recovered_entries,
            lost_entries=lost_entries,
            recovered_bytes=recovered_bytes,
            lost_bytes=lost_bytes,
            errors=errors,
        )

    def _rebuild_index_from_segments(self) -> None:
        """Scan all segment files to rebuild metadata index and pool queues."""
        segments_dir = self._config.cache_dir / "segments"

        for seg_file in sorted(segments_dir.glob("seg_*.arrow")):
            try:
                # Extract segment ID from filename
                seg_name = seg_file.stem  # e.g., "seg_0001"
                segment_id = int(seg_name.split("_")[1])

                # Memory-map for reading
                mmap = pa.memory_map(str(seg_file), "r")
                self._segment_mmaps[segment_id] = mmap

                # Read all batches to extract keys and build index
                offset = 0
                entry_count = 0
                segment_size = seg_file.stat().st_size
                segment_created = seg_file.stat().st_mtime

                reader = pa.RecordBatchStreamReader(mmap)
                pool_key: Optional[Tuple[str, SizeClass]] = None
                schema_key = "unified"  # Default for unified schema

                # Segments written before the per-batch key-column fix stored
                # the key only in schema metadata, which an IPC stream collapses
                # to the first entry's key on read-back. Such segments cannot be
                # indexed correctly, so drop them rather than serve wrong data.
                if CACHE_KEY_FIELD not in reader.schema.names:
                    logger.warning(
                        f"Discarding legacy/corrupt cache segment without "
                        f"per-batch key column: {seg_file}"
                    )
                    self._segment_mmaps.pop(segment_id, None)
                    mmap.close()
                    seg_file.unlink()
                    continue

                # Walk the IPC stream message-by-message so we can record the
                # byte offset of each record batch (issue #9: the localhost
                # cache-file handoff seeks straight to a single message). The
                # StreamReader above only gives us the schema; reading messages
                # directly off the mmap lets us bracket each batch with tell().
                schema = reader.schema
                mmap.seek(0)
                pa.ipc.read_message(mmap)  # consume the leading schema message
                while True:
                    pos = mmap.tell()
                    try:
                        msg = pa.ipc.read_message(mmap)
                    except (pa.ArrowInvalid, EOFError, StopIteration):
                        break
                    if msg is None:
                        break
                    msg_len = mmap.tell() - pos
                    batch = pa.ipc.read_record_batch(msg, schema)
                    # Extract cache key from the per-batch column.
                    key = batch.column(CACHE_KEY_FIELD)[0].as_py()
                    if key is not None:
                        batch_size = sum(
                            batch.column(name).nbytes
                            for name in ("data", "shape", "dtype")
                        )

                        # Determine pool key for this entry
                        size_class = _get_size_class(batch_size)
                        pool_key = (schema_key, size_class)

                        entry_info = SegmentEntryInfo(
                            segment_id=segment_id,
                            offset=entry_count,  # Entry index for the sequential reader
                            size_bytes=batch_size,
                            metadata={},
                            created_at=segment_created,
                            last_access_time=segment_created,  # Initialize with file mtime
                            byte_offset=pos,
                            byte_length=msg_len,
                        )
                        self._metadata[key] = entry_info
                        entry_count += 1

                # Initialize pool queue for this segment if we have entries
                if pool_key is not None and entry_count > 0:
                    pool_queue = self._pool_queues.get(pool_key)
                    if pool_queue is None:
                        pool_queue = PoolQueueInfo(pool_key=pool_key)
                        self._pool_queues[pool_key] = pool_queue

                    # Add segment to pool queue (at tail since it's oldest)
                    pool_queue.queue.append(segment_id)
                    pool_queue.segments[segment_id] = SieveKSegmentInfo(
                        segment_id=segment_id,
                        size_bytes=segment_size,
                        created_at=segment_created,
                        last_access_time=segment_created,
                        entry_count=entry_count,
                        frequency=0,  # Start with counter=0
                        mmap_released=False,
                    )
                else:
                    # Legacy tracking for segments without pool info
                    self._segments_legacy[segment_id] = SegmentInfo(
                        size_bytes=segment_size,
                        created_at=segment_created,
                        last_access_time=segment_created,
                        entry_count=entry_count,
                    )

            except Exception as e:
                logger.error(f"Error rebuilding index from {seg_file}: {e}")
                # Remove corrupted segment
                try:
                    seg_file.unlink()
                except OSError:
                    pass

    def _get_schema_key(self, schema: pa.Schema) -> str:
        """Get hashable key from schema (types, not metadata)."""
        # Schema equality ignores metadata values, so use types
        return repr(schema.types)

    def _create_segment_for_pool(
        self,
        pool_key: Tuple[str, SizeClass],
        schema: pa.Schema,
    ) -> int:
        """Create a new segment for a specific pool and return its ID."""
        segment_id = self._next_segment_id
        self._next_segment_id += 1

        segments_dir = self._config.cache_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        segment_path = segments_dir / f"seg_{segment_id:04d}.arrow"

        # Create writer
        sink = pa.OSFile(str(segment_path), "wb")
        writer = pa.RecordBatchStreamWriter(sink, schema)

        # Get or create pool queue
        pool_queue = self._pool_queues.get(pool_key)
        if pool_queue is None:
            pool_queue = PoolQueueInfo(pool_key=pool_key)
            self._pool_queues[pool_key] = pool_queue

        # Track segment in pool queue (at head since newest)
        pool_queue.queue.appendleft(segment_id)
        pool_queue.segments[segment_id] = SieveKSegmentInfo(
            segment_id=segment_id,
            size_bytes=0,
            created_at=time.time(),
            last_access_time=time.time(),
            entry_count=0,
            frequency=0,  # New segments start with counter=0
            mmap_released=False,
        )

        # Register in pool tracking
        self._pool_writers[segment_id] = writer
        self._pool_sinks[segment_id] = sink
        self._pool_paths[segment_id] = segment_path
        self._pool_schemas[segment_id] = schema
        self._open_pools[pool_key] = segment_id

        return segment_id

    def _close_writer(self, segment_id: int) -> None:
        """Close and forget a segment's stream writer and its backing sink.

        Both must be closed: RecordBatchStreamWriter.close() finalizes the IPC
        stream but leaves the OSFile sink open, so the write handle would linger
        and block unlink on Windows (issue #5).
        """
        writer = self._pool_writers.pop(segment_id, None)
        if writer is not None:
            writer.close()
        sink = self._pool_sinks.pop(segment_id, None)
        if sink is not None:
            sink.close()

    def _close_segment(self, segment_id: int) -> None:
        """Close a full segment's writer and reopen it read-only as an mmap.

        Caller must hold ``self._write_lock`` (serializes against other mutators)
        but must NOT hold ``self._lock``: ``writer.close()`` flushes buffered
        bytes (a blocking disk op), so it runs with ``self._lock`` released and
        the lock is taken only for the surrounding in-memory index mutations.
        """
        # Detach writer/sink/path from the index under the lock (in-memory only).
        with self._lock:
            writer = self._pool_writers.pop(segment_id, None)
            sink = self._pool_sinks.pop(segment_id, None)
            path = self._pool_paths.pop(segment_id, None)
            self._pool_schemas.pop(segment_id, None)
            self._open_pools = {
                k: v for k, v in self._open_pools.items() if v != segment_id
            }

        # Flush + close the handles WITHOUT self._lock (this is the blocking I/O;
        # both must close -- close() leaves the OSFile sink open, issue #5).
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()

        # Reopen read-only (mmap is a non-blocking read map) and drop redundant
        # in-memory copies, under the lock.
        with self._lock:
            if path and path.exists():
                mmap = pa.memory_map(str(path), "r")
                self._segment_mmaps[segment_id] = mmap

                # The segment is now re-readable, so the in-memory RecordBatch
                # copies of its entries are redundant. Drop those no longer
                # referenced (the common sweep case: written, served, released
                # while the segment was still filling). Entries still referenced
                # keep their copy and are dropped on their own release() now that
                # the segment is readable. This bounds resident decoded data to
                # the open write segments plus what callers currently hold,
                # instead of every chunk ever cached.
                for k, info in list(self._metadata.items()):
                    if info.segment_id == segment_id:
                        entry = self._entries.get(k)
                        if entry is not None and entry.is_evictable():
                            self._entries.pop(k, None)

    def _get_total_size(self) -> int:
        """Get total size across all segments."""
        total = 0
        # Pool queues
        for pool in self._pool_queues.values():
            for seg_info in pool.segments.values():
                total += seg_info.size_bytes
        # Legacy segments
        for seg_info in self._segments_legacy.values():
            total += seg_info.size_bytes
        return total

    def _get_total_segment_count(self) -> int:
        """Count total segments across all pools."""
        total = 0
        for pool in self._pool_queues.values():
            total += len(pool.queue)
        total += len(self._segments_legacy)
        return total

    def _get_pool_key_for_segment(
        self, segment_id: int
    ) -> Optional[Tuple[str, SizeClass]]:
        """Get the pool key for a given segment ID."""
        for pool_key, pool in self._pool_queues.items():
            if segment_id in pool.segments:
                return pool_key
        return None

    def _segment_is_evictable(self, segment_id: int) -> bool:
        """Check if segment has no entries holding references."""
        for key, entry_info in self._metadata.items():
            if entry_info.segment_id == segment_id:
                entry = self._entries.get(key)
                if entry and entry.ref_count > 0:
                    return False
        return True

    def _do_evict_segment(self, segment_id: int) -> None:
        """Actually evict a segment: remove files, metadata, and mmap."""
        # Remove all entries in this segment from metadata
        keys_to_remove = [
            k for k, e in self._metadata.items() if e.segment_id == segment_id
        ]
        for key in keys_to_remove:
            self._metadata.pop(key, None)
            # Also remove from in-memory entries if present
            self._entries.pop(key, None)

        # Close any open writer (and its sink) for this segment
        self._close_writer(segment_id)
        self._pool_paths.pop(segment_id, None)
        self._pool_schemas.pop(segment_id, None)
        self._open_pools = {
            k: v for k, v in self._open_pools.items() if v != segment_id
        }

        # Close and remove mmap
        mmap = self._segment_mmaps.pop(segment_id, None)
        if mmap:
            mmap.close()

        # Delete segment file
        segments_dir = self._config.cache_dir / "segments"
        seg_file = segments_dir / f"seg_{segment_id:04d}.arrow"
        if seg_file.exists():
            seg_file.unlink()

        # Remove from legacy tracking if present
        self._segments_legacy.pop(segment_id, None)

        self._evictions += 1

    def _select_pool_for_eviction(self) -> Optional[Tuple[str, SizeClass]]:
        """Select pool with lowest aggregate hit rate."""
        if not self._pool_queues:
            return None

        # Calculate hit rates
        pool_rates = []
        for pool_key, pool in self._pool_queues.items():
            total = pool.hits + pool.misses
            if total > 0:
                hit_rate = pool.hits / total
            else:
                hit_rate = 0.0
            pool_rates.append((pool_key, hit_rate, pool.queue))

        # Select lowest hit rate pool with segments
        pool_rates.sort(
            key=lambda x: (x[1], -len(x[2]))
        )  # Lowest rate, then most segments
        for pool_key, rate, queue in pool_rates:
            if queue:
                return pool_key
        return None

    def _evict_segment_sieve_k(self) -> bool:
        """Per-pool Sieve-K sweep following reference algorithm.

        Returns True if evicted, False if nothing evictable.
        """
        # Try pool queues first (Sieve-K)
        target_pool = self._select_pool_for_eviction()
        if target_pool is not None:
            pool_queue = self._pool_queues[target_pool]

            # Sieve-K eviction loop
            max_iterations = len(pool_queue.queue) * 2  # Safety limit
            iterations = 0
            while iterations < max_iterations:
                iterations += 1

                # Wrap hand if it exceeds queue length
                if pool_queue.hand >= len(pool_queue.queue):
                    pool_queue.hand = 0

                if len(pool_queue.queue) == 0:
                    return False

                # Get segment at hand offset from tail
                # deque: newest at left (index 0), oldest at right (index -1 is tail)
                # -1 - hand: tail is index -1, hand=0 → -1, hand=1 → -2, etc.
                idx = -1 - pool_queue.hand

                if idx < -len(pool_queue.queue):
                    # Wrapped around, all segments have counter > 0
                    return False

                seg_id = pool_queue.queue[idx]
                seg_info = pool_queue.segments.get(seg_id)

                if seg_info is None:
                    # Segment info missing, skip
                    pool_queue.hand += 1
                    continue

                # Check if segment is evictable (no ref_count entries)
                if not self._segment_is_evictable(seg_id):
                    pool_queue.hand += 1
                    continue

                # Sieve-K logic
                if seg_info.frequency > 0:
                    # Hot segment: decrement counter, advance hand
                    seg_info.frequency -= 1
                    pool_queue.hand += 1
                else:
                    # Cold segment (frequency == 0): evict
                    self._do_evict_segment(seg_id)
                    pool_queue.segments.pop(seg_id, None)
                    # O(n) deletion from deque, acceptable for <1000 segments
                    del pool_queue.queue[idx]
                    # Hand stays at this position for next eviction
                    return True

        # Fall back to legacy segments (simple LRU)
        if self._segments_legacy:
            # Find oldest evictable legacy segment
            evictable_legacy = []
            for seg_id, seg_info in self._segments_legacy.items():
                if self._segment_is_evictable(seg_id):
                    evictable_legacy.append((seg_id, seg_info.last_access_time))

            if evictable_legacy:
                oldest_seg = min(evictable_legacy, key=lambda x: x[1])
                seg_id = oldest_seg[0]
                self._do_evict_segment(seg_id)
                return True

        # Nothing to evict
        self._ref_held_skips += 1
        return False

    def _reopen_segment_mmap(
        self, segment_id: int, seg_info: SieveKSegmentInfo
    ) -> None:
        """Reopen mmap for cold segment that was accessed."""
        path = self._config.cache_dir / "segments" / f"seg_{segment_id:04d}.arrow"
        if path.exists():
            mmap = pa.memory_map(str(path), "r")
            self._segment_mmaps[segment_id] = mmap
            seg_info.mmap_released = False
            # On access: increment counter
            seg_info.frequency = min(K, seg_info.frequency + 1)

    def _maybe_release_cold_mmaps(self) -> None:
        """Release mmap handles for cold segments.

        Only runs if total segments exceed MMAP_LIFECYCLE_THRESHOLD.
        """
        if self._get_total_segment_count() <= MMAP_LIFECYCLE_THRESHOLD:
            return  # Skip for small caches

        now = time.time()
        for pool_key, pool in self._pool_queues.items():
            for seg_id, seg_info in pool.segments.items():
                age = now - seg_info.last_access_time
                if (
                    seg_info.frequency <= COLD_FREQUENCY_THRESHOLD
                    and age > COLD_THRESHOLD_SECONDS
                ):
                    mmap = self._segment_mmaps.pop(seg_id, None)
                    if mmap:
                        mmap.close()
                    seg_info.mmap_released = True

    def _read_batch_from_segment(self, key: bytes) -> Optional[pa.RecordBatch]:
        """Read a batch from segment file using mmap and cast back to typed schema."""
        entry_info = self._metadata.get(key)
        if entry_info is None:
            return None

        # Get pool info for this segment
        pool_key = self._get_pool_key_for_segment(entry_info.segment_id)

        # Track pool hit if we have pool info
        if pool_key:
            pool_queue = self._pool_queues.get(pool_key)
            if pool_queue:
                seg_info = pool_queue.segments.get(entry_info.segment_id)
                if seg_info:
                    # Reopen mmap if cold segment was released
                    if seg_info.mmap_released:
                        self._reopen_segment_mmap(entry_info.segment_id, seg_info)

                    # Sieve-K: increment counter on hit, saturating at K=2
                    seg_info.frequency = min(K, seg_info.frequency + 1)
                    seg_info.last_access_time = time.time()
                    pool_queue.hits += 1

        mmap = self._segment_mmaps.get(entry_info.segment_id)
        if mmap is None:
            return None

        # Seek back to beginning since reader exhausts the mmap
        mmap.seek(0)

        # Periodic mmap cleanup check
        self._access_counter += 1
        if self._access_counter % 100 == 0:
            self._maybe_release_cold_mmaps()

        # Read all batches and find the right one by index
        # (Arrow IPC stream doesn't support direct offset access easily)
        reader = pa.RecordBatchStreamReader(mmap)
        for i, batch in enumerate(reader):
            if i == entry_info.offset:
                # Detach from the segment mmap on Windows so the file can be
                # unlinked during eviction even while a caller holds the batch
                # (issue #5). POSIX keeps the zero-copy mmap read.
                if self._copy_on_read:
                    batch = _copy_batch_off_mmap(batch)
                # Cast from unified binary schema back to typed schema
                typed_batch = _cast_from_unified_schema(batch)
                return typed_batch

        return None

    def _fill_byte_offsets_for_segment(self, segment_id: int) -> None:
        """Walk a segment file once and record each entry's IPC message range.

        Offsets aren't captured at write time (the stream writer buffers the
        schema until the first batch, so the live sink cursor can't bracket the
        first message). Instead we derive them by walking the flushed file,
        matching messages to keys via the per-batch cache_key column, and fill
        ``byte_offset`` / ``byte_length`` on the existing metadata entries. The
        caller (``locate_entry``) holds both ``self._write_lock`` and
        ``self._lock``: ``_write_lock`` excludes concurrent appends by
        ``complete_entry`` so the walk never races a torn append (since #119 the
        write critical section no longer holds ``_lock``), and ``_lock`` guards
        the ``_metadata`` mutations below. A trailing message can still be torn
        by a *prior* failed/partial write (its slack persists in the segment),
        so the read loop stops on any unreadable message rather than raising.
        """
        path = self._config.cache_dir / "segments" / f"seg_{segment_id:04d}.arrow"
        try:
            mm = pa.memory_map(str(path), "r")
        except (OSError, pa.ArrowInvalid):
            return
        # Always close the mapping: an open handle blocks segment deletion on
        # Windows (biopb/biopb#5) and leaks fds elsewhere.
        try:
            try:
                schema = pa.ipc.open_stream(mm).schema
            except Exception:
                return
            if CACHE_KEY_FIELD not in schema.names:
                return
            mm.seek(0)
            pa.ipc.read_message(mm)  # consume the leading schema message
            while True:
                pos = mm.tell()
                try:
                    msg = pa.ipc.read_message(mm)
                except (pa.ArrowInvalid, EOFError, StopIteration, OSError):
                    # OSError "Expected to be able to read N bytes for message
                    # body" = a torn trailing message (a prior partial write's
                    # slack). Treat it as end-of-readable-region.
                    break
                if msg is None:
                    break
                msg_len = mm.tell() - pos
                try:
                    batch = pa.ipc.read_record_batch(msg, schema)
                except Exception:
                    break
                entry_key = batch.column(CACHE_KEY_FIELD)[0].as_py()
                if entry_key is None:
                    continue
                info = self._metadata.get(entry_key)
                if info is not None and info.segment_id == segment_id:
                    info.byte_offset = pos
                    info.byte_length = msg_len
        finally:
            mm.close()

    def locate_entry(self, key: bytes) -> Optional[ChunkLocation]:
        """Return the on-disk location of a cached chunk, or None.

        Backs the localhost cache-file handoff (issue #9). Returns None when the
        key isn't cached or its byte offset can't be resolved, signalling the
        caller to fall back to do_get.
        """
        # Fast path: an already-indexed entry needs only the read lock.
        with self._lock:
            entry_info = self._metadata.get(key)
            if entry_info is None:
                return None
            # byte_offset == 0 is never a real entry — the schema message always
            # occupies the start of the segment — so 0 means "not yet indexed".
            if entry_info.byte_offset and entry_info.byte_length:
                return self._build_chunk_location(entry_info)

        # Slow path: derive offsets by walking the flushed segment file. This
        # must hold ``_write_lock`` so the walk cannot race a concurrent append
        # by ``complete_entry`` -- a torn append leaves a trailing message whose
        # header advertises a body the file doesn't yet contain, and
        # ``pa.ipc.read_message`` then raises (on Windows: OSError "Expected to
        # be able to read N bytes for message body"). The lock order here is
        # ``_write_lock`` -> ``_lock``, matching ``complete_entry``, so there is
        # no ABBA deadlock; ``_write_lock`` blocks at most concurrent writes,
        # never the cached-read path. (Only the *first* locate of an unindexed
        # entry pays this; the fast path above takes neither write lock.)
        with self._write_lock:
            with self._lock:
                entry_info = self._metadata.get(key)
                if entry_info is None:
                    return None
                if not entry_info.byte_offset:
                    self._fill_byte_offsets_for_segment(entry_info.segment_id)
                    entry_info = self._metadata.get(key)
                if (
                    entry_info is None
                    or not entry_info.byte_offset
                    or not entry_info.byte_length
                ):
                    return None
                return self._build_chunk_location(entry_info)

    def _build_chunk_location(
        self, entry_info: SegmentEntryInfo
    ) -> Optional[ChunkLocation]:
        """Build a ``ChunkLocation`` for an already-indexed entry.

        Caller must hold ``self._lock``. Returns None if the segment file has
        gone away (evicted/unlinked) since indexing.
        """
        segment_path = (
            self._config.cache_dir
            / "segments"
            / f"seg_{entry_info.segment_id:04d}.arrow"
        )
        try:
            generation_id = os.stat(segment_path).st_ino
        except OSError:
            return None

        # The client is about to map this segment; treat the locate as a hit
        # so Sieve-K doesn't evict it out from under the reader.
        self._update_segment_frequency(entry_info.segment_id)

        return ChunkLocation(
            segment_path=str(segment_path),
            byte_offset=entry_info.byte_offset,
            byte_length=entry_info.byte_length,
            generation_id=generation_id,
        )

    def get_or_acquire(
        self,
        key: bytes,
        compute_fn: Callable[[], Tuple[pa.RecordBatch, int]],
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Get existing entry or create pending and compute."""
        is_owner = False
        with self._lock:
            entry = self._entries.get(key)

            if entry is not None:
                if entry.state == EntryState.READY:
                    # Ready entry - acquire and return
                    entry.acquire()
                    self._hits += 1
                    # Update segment frequency if entry is from a segment
                    entry_info = self._metadata.get(key)
                    if entry_info:
                        self._update_segment_frequency(entry_info.segment_id)
                    return entry

                if entry.state == EntryState.PENDING:
                    # Pending - wait outside lock (another thread is computing)
                    self._pending_waits += 1

            else:
                # Check if data exists in segment files
                if key in self._metadata:
                    # Data exists in file - create ready entry
                    batch = self._read_batch_from_segment(key)
                    if batch is not None:
                        size_bytes = sum(col.nbytes for col in batch.columns)
                        entry = CacheEntry(
                            data=batch,
                            state=EntryState.READY,
                            created_at=time.time(),
                            size_bytes=size_bytes,
                            metadata=metadata or {},
                        )
                        entry.acquire()
                        self._entries[key] = entry
                        self._hits += 1
                        return entry

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
            return entry

    def _update_segment_frequency(self, segment_id: int) -> None:
        """Update frequency counter for a segment when accessed."""
        pool_key = self._get_pool_key_for_segment(segment_id)
        if pool_key:
            pool_queue = self._pool_queues.get(pool_key)
            if pool_queue:
                seg_info = pool_queue.segments.get(segment_id)
                if seg_info:
                    # Sieve-K: increment counter on hit, saturating at K=2
                    seg_info.frequency = min(K, seg_info.frequency + 1)
                    seg_info.last_access_time = time.time()
                    pool_queue.hits += 1

    def start_compute(
        self,
        key: bytes,
        metadata: Optional[dict] = None,
    ) -> Tuple[CacheEntry, bool]:
        """Start compute phase - returns (entry, is_owner)."""
        with self._lock:
            entry = self._entries.get(key)

            if entry is not None:
                if entry.state == EntryState.READY:
                    entry.acquire()
                    self._hits += 1
                    # Update segment frequency if entry is from a segment
                    entry_info = self._metadata.get(key)
                    if entry_info:
                        self._update_segment_frequency(entry_info.segment_id)
                    return entry, False

                if entry.state == EntryState.PENDING:
                    # Someone else computing - acquire and wait
                    entry.acquire()
                    self._pending_waits += 1
                    return entry, False

            # Check if data exists in segment files
            if key in self._metadata:
                batch = self._read_batch_from_segment(key)
                if batch is not None:
                    size_bytes = sum(col.nbytes for col in batch.columns)
                    entry = CacheEntry(
                        data=batch,
                        state=EntryState.READY,
                        created_at=time.time(),
                        size_bytes=size_bytes,
                        metadata=metadata or {},
                    )
                    entry.acquire()
                    self._entries[key] = entry
                    self._hits += 1
                    return entry, False

            # No entry - create pending, we own computation
            entry = CacheEntry(
                state=EntryState.PENDING,
                created_at=time.time(),
                metadata=metadata or {},
            )
            entry.acquire()
            self._entries[key] = entry
            self._misses += 1
            return entry, True

    def complete_entry(
        self,
        key: bytes,
        data: pa.RecordBatch,
        size_bytes: int,
    ) -> None:
        """Mark a pending entry as ready and persist it to a segment.

        The blocking disk writes (``write_batch`` / ``flush``, and the rotation
        ``writer.close()``) run WITHOUT ``self._lock`` held -- serialized instead
        by ``self._write_lock``, which readers never take. So a write that stalls
        (e.g. a full filesystem) blocks only future writes, never the read path;
        ``self._lock`` is taken only for short in-memory index mutations.
        """
        # Check for oversized chunks
        if size_bytes > MAX_ARROW_BATCH_BYTES:
            self._oversized_skips += 1
            logger.warning(
                f"Skipping cache for oversized chunk: {size_bytes} bytes > {MAX_ARROW_BATCH_BYTES}"
            )
            # Mark entry as ready in memory but don't persist
            with self._lock:
                entry = self._entries.get(key)
                if entry is None or entry.state != EntryState.PENDING:
                    return
                entry.set_ready(data, size_bytes)
            return

        # Serialize the whole write/evict/close critical section. complete_entry
        # is the only mutator of writer/segment state, so holding _write_lock
        # keeps the selected segment stable across the unlocked write below.
        with self._write_lock:
            # ---- PHASE 1: in-memory bookkeeping + segment selection ----
            # Everything here is in-memory or non-blocking-on-ENOSPC (dict
            # mutations, segment-file create, eviction unlink), so it is safe to
            # hold self._lock. NOTE: a residual blocking flush can occur if the
            # eviction sweep closes the currently-open write segment; Sieve-K
            # keeps the newest (active) segment, so this is rare -- the common,
            # every-chunk write+flush below is the path that matters.
            with self._lock:
                entry = self._entries.get(key)
                if entry is None or entry.state != EntryState.PENDING:
                    return

                size_class = _get_size_class(size_bytes)

                # Evict if needed before storing
                while (
                    self._get_total_size() + size_bytes > self._config.max_total_bytes
                ):
                    if not self._evict_segment_sieve_k():
                        break

                # Cast to unified schema (all dtypes share a pool) and attach the
                # cache key as a per-row column (NOT schema metadata): Arrow IPC
                # persists the schema once per segment, so schema metadata can't
                # identify individual batches on rebuild.
                unified_batch = _cast_to_unified_schema(data)
                key_col = pa.array([key], type=pa.binary())
                batch_with_key = pa.RecordBatch.from_arrays(
                    list(unified_batch.columns) + [key_col],
                    names=list(unified_batch.schema.names) + [CACHE_KEY_FIELD],
                )

                schema_key = "unified"
                pool_key = (schema_key, size_class)
                pool_queue = self._pool_queues.get(pool_key)
                if pool_queue is None:
                    pool_queue = PoolQueueInfo(pool_key=pool_key)
                    self._pool_queues[pool_key] = pool_queue
                pool_queue.misses += 1

                # Find or create the open segment for this pool
                segment_id = self._open_pools.get(pool_key)
                if segment_id is None or segment_id not in self._pool_writers:
                    segment_id = self._create_segment_for_pool(
                        pool_key, batch_with_key.schema
                    )
                writer = self._pool_writers.get(segment_id)
                if writer is None:
                    segment_id = self._create_segment_for_pool(
                        pool_key, batch_with_key.schema
                    )
                    writer = self._pool_writers[segment_id]
                sink = self._pool_sinks.get(segment_id)

            # ---- PHASE 2: blocking disk write, self._lock RELEASED ----
            # Flush so the bytes are durable in the page cache; the localhost
            # cache-file handoff (issue #9) derives byte offsets lazily by walking
            # the flushed file in locate_entry(). A failure here (e.g. ENOSPC)
            # propagates out: get_or_acquire's owner branch turns it into
            # fail_entry() + re-raise. That is now safe -- the `with` blocks
            # release both locks on the way out, so a failed (or stalled) write
            # can no longer leave a lock held across the read path.
            if self._wal:
                self._wal.log_pending(key)
            writer.write_batch(batch_with_key)
            if sink is not None:
                sink.flush()

            # ---- PHASE 3: in-memory commit ----
            need_close = False
            with self._lock:
                entry = self._entries.get(key)
                if entry is None or entry.state != EntryState.PENDING:
                    # Raced away (failed/evicted while we wrote). The written
                    # bytes are harmless slack in the segment.
                    return

                seg_info = pool_queue.segments.get(segment_id)
                if seg_info:
                    seg_info.size_bytes += size_bytes
                    seg_info.entry_count += 1
                    seg_info.last_access_time = time.time()

                self._metadata[key] = SegmentEntryInfo(
                    segment_id=segment_id,
                    offset=seg_info.entry_count - 1 if seg_info else 0,
                    size_bytes=size_bytes,
                    metadata=entry.metadata,
                    created_at=time.time(),
                    last_access_time=time.time(),
                )
                entry.set_ready(data, size_bytes)
                need_close = bool(
                    seg_info and seg_info.size_bytes >= self._config.max_segment_bytes
                )

            # Rotate a full segment. _close_segment flushes the writer (blocking),
            # so it manages its own locking and runs with self._lock NOT held
            # (still under _write_lock, so the segment state is stable).
            if need_close:
                self._close_segment(segment_id)

            if self._wal:
                self._wal.log_committed(key)

    def fail_entry(self, key: bytes, error: Exception) -> None:
        """Mark pending entry as failed."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state != EntryState.PENDING:
                return

            entry.set_error(error)
            self._entries.pop(key, None)

    def release(self, key: bytes) -> int:
        """Release reference to entry.

        When the last reference drops, evict the in-memory ``RecordBatch`` for an
        entry that is redundantly persisted on a *readable* segment. Without this,
        every chunk ever served stays mirrored in memory for the life of the
        process, bounded only by the *disk* ``max_total_bytes`` (e.g. 96 GB),
        which exhausts RAM on a large catalog or a precache sweep.

        ``complete_entry`` writes the batch to its segment *and* keeps it on
        ``entry.data`` so the computing/serving thread can return it without an
        immediate re-read. Once no caller holds the entry, that copy is pure RAM
        cost -- *provided the segment is closed and mmap-readable*, in which case
        ``get_or_acquire``/``start_compute`` re-read it via
        ``_read_batch_from_segment`` on the next hit. While the segment is still
        the active write segment it is not yet re-readable, so ``entry.data`` is
        the only readable copy and must be kept; those entries are dropped later,
        at ``_close_segment`` or on a subsequent ``release`` once readable.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            count = entry.release()
            self._maybe_drop_in_memory_batch(key, entry)
            return count

    def _maybe_drop_in_memory_batch(self, key: bytes, entry: CacheEntry) -> None:
        """Drop an entry's in-memory copy when it is a redundant mirror of a
        readable segment.

        Safe to drop only when the entry is unreferenced (``is_evictable``) AND
        its data is recoverable: persisted (``key in self._metadata``) on a
        segment that is closed and mmap-readable (``segment_id in
        self._segment_mmaps``). A caller that already read ``entry.data`` holds
        its own reference, so the batch survives until it is done; only future
        lookups change, and they rebuild from the segment. Caller holds
        ``self._lock``. No-op for in-memory-only entries (e.g. oversized skips)
        and for entries on the still-open write segment.
        """
        if not entry.is_evictable():
            return
        info = self._metadata.get(key)
        if info is None or info.segment_id not in self._segment_mmaps:
            return
        self._entries.pop(key, None)

    def remove(self, key: bytes) -> bool:
        """Remove entry from in-memory tracking only.

        Space is reclaimed when segment is evicted.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if not entry.is_evictable():
                return False

            self._entries.pop(key, None)
            self._metadata.pop(key, None)
            return True

    def clear(self) -> None:
        """Clear all evictable entries and delete all segments."""
        with self._lock:
            # Close all pool writers (and their sinks)
            for segment_id in list(self._pool_writers.keys() | self._pool_sinks.keys()):
                self._close_writer(segment_id)
            self._pool_paths.clear()
            self._pool_schemas.clear()
            self._open_pools.clear()
            self._next_segment_map.clear()

            # Close all mmaps
            for mmap in self._segment_mmaps.values():
                mmap.close()
            self._segment_mmaps.clear()

            # Delete all segment files
            segments_dir = self._config.cache_dir / "segments"
            for seg_file in segments_dir.glob("seg_*.arrow"):
                seg_file.unlink()

            # Clear tracking
            self._pool_queues.clear()
            self._segments_legacy.clear()
            self._metadata.clear()
            self._entries.clear()

            # Clear WAL
            if self._wal:
                self._wal.clear()

            # Reset segment ID counter
            self._next_segment_id = 1
            self._access_counter = 0

    def stats(self) -> CacheStats:
        """Return cache statistics."""
        total_entries = len(self._metadata)
        total_bytes = self._get_total_size()

        # Build pool-level statistics
        pool_stats = {}
        for pool_key, pool in self._pool_queues.items():
            pool_name = f"{pool_key[0]}-{pool_key[1]}"
            total = pool.hits + pool.misses
            hit_rate = pool.hits / total if total > 0 else 0.0
            pool_stats[pool_name] = PoolStats(
                pool_key=pool_name,
                hits=pool.hits,
                misses=pool.misses,
                segments=len(pool.queue),
                bytes=sum(s.size_bytes for s in pool.segments.values()),
                hit_rate=hit_rate,
            )

        return CacheStats(
            total_entries=total_entries,
            total_bytes=total_bytes,
            max_entries=0,  # File backend doesn't have entry count limit
            max_bytes=self._config.max_total_bytes,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            pending_waits=self._pending_waits,
            ref_held_evictions_skipped=self._ref_held_skips,
            oversized_skips=self._oversized_skips,
            pool_stats=pool_stats,
        )

    def get_recovery_status(self) -> Optional[RecoveryStatus]:
        """Get recovery status from last initialization."""
        return self._recovery_status

    def close(self) -> None:
        """Close backend and release resources.

        This preserves segment files for persistence across restarts.
        Only releases locks and closes handles.
        """
        with self._lock:
            # Close all pool writers (and their sinks)
            for segment_id in list(self._pool_writers.keys() | self._pool_sinks.keys()):
                self._close_writer(segment_id)
            self._pool_paths.clear()
            self._pool_schemas.clear()
            self._open_pools.clear()
            self._next_segment_map.clear()

            # Close all mmap handles
            for mmap in self._segment_mmaps.values():
                mmap.close()
            self._segment_mmaps.clear()

            # Clear WAL (writes complete)
            if self._wal:
                self._wal.clear()

            # Release process lock
            if self._process_lock:
                self._process_lock.release()

            # Clear in-memory tracking (data persists in files)
            self._entries.clear()
            self._metadata.clear()
            self._pool_queues.clear()
            self._segments_legacy.clear()
