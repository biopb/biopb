"""Thread-safe persistent Arrow file cache backend.

Implements segmented storage with:
- Mmap reads for near-memory-speed access
- Segment-level LRU eviction
- Crash recovery via the process lock and a torn-tail-tolerant boot walk
- The future/promise pattern over the value types in cache.types

Everything here is concurrent and lock-disciplined: read the ``_lock`` /
``_write_lock`` notes in ``__init__`` before moving code between methods. The
two parts that are *not* live under those locks live elsewhere --
``cache.bootstrap`` (the single-threaded boot path and the cache directory's
layout) and ``cache.segment_index`` (the ``.arrow`` / ``.idx`` file formats).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import pyarrow as pa

from biopb_tensor_server.cache import bootstrap, segment_index
from biopb_tensor_server.cache.bootstrap import CacheLayout
from biopb_tensor_server.cache.recovery import (
    PoolQueueInfo,
    ProcessLock,
    RecoveryStatus,
    SegmentEntryInfo,
    SieveKSegmentInfo,
)
from biopb_tensor_server.cache.segment_index import (
    IndexRecord,
    SegmentScan,
    batch_at_offset,
    batch_with_key,
    bracket_message,
)
from biopb_tensor_server.cache.types import (
    EVICTION_RANK,
    MAX_ARROW_BATCH_BYTES,
    CacheEntry,
    CacheStats,
    ChunkLocation,
    EntryState,
    PoolStats,
    RetentionClass,
    estimate_batch_bytes,
)

__all__ = ["ArrowFileBackend", "ArrowFileConfig", "ChunkLocation"]

logger = logging.getLogger(__name__)


# Segment mmap lifecycle (release the handle for a segment gone cold+idle).
COLD_THRESHOLD_SECONDS = 300  # 5 minutes without access
MMAP_LIFECYCLE_THRESHOLD = 100  # Only manage mmaps when segments > 100


# Size class thresholds for pooling. A segment holds one class, so what this
# buys is a uniform eviction granularity: a "large" segment loses one entry when
# it goes, a "tiny" one loses ~180. Outliers vs bulk, not four buckets: the old
# 8 MB boundary was byte-identical to PREFERRED_ARROW_BATCH_BYTES, the size the
# transfer grid targets, so it cut the population at its own mode (measured on a
# 25 GB cache: median entry 7.99 MB, 52% either side). Splitting the bulk buys
# nothing and costs a pool, and every pool holds one open unsealed segment.
SIZE_CLASS_TINY_THRESHOLD = 1 * 1024 * 1024  # <1MB
SIZE_CLASS_BULK_THRESHOLD = 32 * 1024 * 1024  # 1-32MB: the transfer grid's range
# large: >=32MB (still cached, just pooled separately)

SizeClass = Literal["tiny", "bulk", "large"]


def _get_size_class(size_bytes: int) -> SizeClass:
    """Classify chunk size for pooling."""
    if size_bytes < SIZE_CLASS_TINY_THRESHOLD:
        return "tiny"
    elif size_bytes < SIZE_CLASS_BULK_THRESHOLD:
        return "bulk"
    else:
        return "large"


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


class ArrowFileBackend:
    """Thread-safe persistent file cache with future/promise pattern.

    Directory structure (see ``cache.bootstrap.CacheLayout``):
        cache_dir/
        ├── segments/
        │   ├── seg_0001.arrow   # the batches
        │   ├── seg_0001.idx     # its seal-time index sidecar
        │   └── ...
        ├── format_version
        └── lock

    Key features:
    1. Future/Promise: get_or_acquire / start_compute below
    2. Mmap reads: OS page cache provides near-memory performance
    3. Segment-level eviction: Delete least-recently-used segment
    4. Crash recovery: a stale process lock triggers it; the boot walk
       detects an interrupted write from the segment's torn tail
    """

    def __init__(self, config: ArrowFileConfig):
        self._config = config
        self._layout = CacheLayout(config.cache_dir)
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

        # Metadata index: key -> SegmentEntryInfo (location in segment), plus the
        # segment_id -> keys inverse. Both are maintained together by
        # ``_index_entry`` / ``_unindex_entry`` -- the only writers -- so no
        # caller has to keep them in step. The inverse exists because every
        # segment-scoped operation (evictability, eviction, seal) needs "the
        # entries in this segment", which otherwise means scanning all of
        # ``_metadata``: ~2.5 ms at 110k entries, paid under ``_lock`` (so it
        # stalls readers) once per Sieve-K hand step.
        self._metadata: Dict[bytes, SegmentEntryInfo] = {}
        self._segment_keys: Dict[int, set] = {}

        # Sieve-K: Per-pool queues with frequency counters
        self._pool_queues: Dict[Tuple[RetentionClass, SizeClass], PoolQueueInfo] = {}

        # Mmap handles for fast reads, and each mapped segment's IPC schema.
        # The schema is what lets a read decode the single message at an entry's
        # recorded byte offset; it is cached because re-reading it per chunk
        # would cost back what the offset saves. Both are maintained only via
        # ``_open_segment_mmap`` / ``_forget_segment_mmap``, so a stale schema
        # can never outlive the mapping it describes (segment ids are reused
        # after ``clear()`` resets the counter).
        self._segment_mmaps: Dict[int, pa.MemoryMappedFile] = {}
        self._segment_schemas: Dict[int, pa.Schema] = {}

        # Multiple active writers for pooling: segment_id -> writer
        # This allows keeping multiple segments open for different (schema, size_class) pools
        self._pool_writers: Dict[int, pa.RecordBatchStreamWriter] = {}
        # The OSFile sink behind each writer. RecordBatchStreamWriter.close()
        # does NOT close the sink it was handed, so we must track and close it
        # ourselves -- otherwise the write handle lingers until GC and (on
        # Windows) blocks segment unlink during eviction/cleanup. See issue #5.
        self._pool_sinks: Dict[int, pa.OSFile] = {}
        self._pool_paths: Dict[int, Path] = {}

        # Pool tracking: (schema_key, size_class) -> segment_id for open segments
        self._open_pools: Dict[Tuple[RetentionClass, SizeClass], int] = {}
        # Reverse index: segment_id -> its pool key, for O(1) lookup instead of
        # scanning every pool (mirrors why _segment_keys exists for _metadata).
        # A segment's pool key is fixed at creation and only cleared on
        # eviction; set in _create_segment_for_pool / _install_segment_records.
        self._segment_pool_key: Dict[int, Tuple[RetentionClass, SizeClass]] = {}

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._pending_waits: int = 0
        self._ref_held_skips: int = 0
        self._oversized_skips: int = 0

        # Access counter for periodic mmap cleanup
        self._access_counter: int = 0

        # Process lock (cross-process ownership of the cache dir)
        self._process_lock: Optional[ProcessLock] = None

        # Recovery status (if recovered from crash)
        self._recovery_status: Optional[RecoveryStatus] = None

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize backend: take the cache dir, recover, rebuild the index.

        Single-threaded and pre-publication: no other thread can reach this
        backend yet, which is why nothing here takes ``_lock``. See
        ``cache.bootstrap`` for the directory-level steps.
        """
        self._process_lock, was_stale = bootstrap.acquire_cache_dir(self._layout)

        # Everything past the acquire releases the lock if it raises: a failed
        # init must not leave the cache dir locked until a later boot reclaims
        # it as stale. Several steps can fail (an ENOSPC marker write, an
        # unreadable segment), so the guarantee belongs here rather than in each.
        try:
            self._initialize_locked(was_stale)
        except Exception:
            self._process_lock.release()
            raise

    def _initialize_locked(self, was_stale: bool) -> None:
        """Init steps that require the process lock; see :meth:`_initialize`."""
        # Enforce the segment format-version contract now that we are the
        # exclusive owner and before anything reads the segments: a missing or
        # mismatched marker wipes the on-disk segments. Must run ahead of
        # recovery and the index rebuild.
        wiped = bootstrap.enforce_format_version(self._layout)

        # A version wipe already dropped the segments, so there is nothing to
        # recover from even if the last owner crashed.
        recovering = was_stale and not wiped
        if recovering:
            logger.info("Cache recovery: the cache dir was not released cleanly")

        # Rebuild the metadata index from the segment files. This is also what
        # detects an interrupted write, so it is the source of both recovery
        # counts -- see bootstrap.recovery_status.
        torn_segments = self._rebuild_index_from_segments()

        if recovering:
            self._recovery_status = bootstrap.recovery_status(
                self._layout, len(self._metadata), torn_segments
            )

        # Find next segment ID (segments are created lazily when first write happens)
        all_segment_ids = set()
        for pool in self._pool_queues.values():
            all_segment_ids.update(pool.queue)
        self._next_segment_id = max(all_segment_ids, default=0) + 1

    def _rebuild_index_from_segments(self) -> int:
        """Rebuild the metadata index and pool queues from segment files at boot.

        Returns the number of segments whose tail was torn -- an interrupted
        write, one lost entry each. Only the unsealed segment can be torn (a
        sealed one is never appended to), and it is also the one without a
        sidecar, so the body walk below always sees it.

        Fast path (biopb/biopb#300): a sealed segment carries a ``seg_NNNN.idx``
        sidecar written at seal time with each entry's key -> byte range. When it
        validates (version + recorded ``.arrow`` size), the index is restored
        from it WITHOUT faulting the segment body -- the walk that once read the
        whole cache from disk (tens of GB, ~52-78 s on a caching proxy) now
        happens once, at seal, off the boot path. A missing / stale / legacy
        sidecar falls back to the pre-#300 body walk, which also backfills a
        fresh sidecar so the next boot is fast.
        """
        seg_files = sorted(self._layout.segments_dir.glob("seg_*.arrow"))
        walked = 0
        torn_segments = 0
        for seg_file in seg_files:
            try:
                segment_id = int(seg_file.stem.split("_")[1])
            except (ValueError, IndexError):
                continue
            try:
                if self._load_segment_from_sidecar(segment_id, seg_file):
                    continue
                # No usable sidecar: walk the body once (the pre-#300 path).
                scan = self._scan_segment_records(seg_file)
                # Count the torn tail before deciding the segment's fate: a torn
                # *first* batch leaves no records, and the drop below must not
                # hide the entry it cost.
                if scan is not None and scan.torn:
                    torn_segments += 1
                # An empty walk means the segment holds no recoverable entry (a
                # torn first batch), so it is dropped rather than tracked -- it
                # occupies disk against max_total_bytes and can serve nothing.
                if not scan or not scan.records:
                    logger.warning(
                        f"Discarding legacy/corrupt cache segment (unreadable or "
                        f"without per-batch key column): {seg_file}"
                    )
                    self._drop_segment_files(segment_id)
                    continue
                self._open_segment_mmap(segment_id, seg_file)
                self._install_segment_records(segment_id, seg_file, scan.records)
                walked += 1
                # Backfill a sidecar from the records we just read -- no second
                # body read -- so the next boot skips this segment's walk.
                self._write_sidecar_from_records(segment_id, scan.records)
            except Exception as e:
                logger.error(f"Error rebuilding index from {seg_file}: {e}")
                self._drop_segment_files(segment_id)

        if walked:
            logger.info(
                "Cache boot: walked %d of %d segment(s) lacking a valid sidecar "
                "index and backfilled them; the next boot skips the walk "
                "(biopb/biopb#300).",
                walked,
                len(seg_files),
            )
        return torn_segments

    def _segment_path(self, segment_id: int) -> Path:
        """Path of a segment's body file (``seg_NNNN.arrow``)."""
        return self._layout.segment_path(segment_id)

    def _index_records_for_segment(
        self, segment_id: int
    ) -> Optional[List[IndexRecord]]:
        """Index records for a sealed segment, taken from the live index.

        A sidecar records exactly what ``_metadata`` already holds, so sealing
        can write it without re-reading the body it just finished. Returns None
        if any entry lacks a byte range, so the caller falls back to the body
        walk, which can still derive one. Caller holds ``_lock``.
        """
        records = []
        for key in self._segment_keys.get(segment_id, ()):
            info = self._metadata.get(key)
            if info is None:
                continue
            if not info.byte_offset or not info.byte_length:
                return None
            records.append(
                IndexRecord(
                    key=key,
                    byte_offset=info.byte_offset,
                    byte_length=info.byte_length,
                    size_bytes=info.size_bytes,
                    offset=info.offset,
                )
            )
        records.sort(key=lambda r: r.byte_offset)
        return records

    def _index_entry(self, key: bytes, info: SegmentEntryInfo) -> None:
        """Record an entry in the metadata index and its segment's key set.

        With ``_unindex_entry``, the only writer of ``_metadata`` /
        ``_segment_keys``, so the two cannot drift. Caller holds ``_lock``.
        """
        old = self._metadata.get(key)
        if old is not None and old.segment_id != info.segment_id:
            self._segment_keys.get(old.segment_id, set()).discard(key)
        self._metadata[key] = info
        self._segment_keys.setdefault(info.segment_id, set()).add(key)

    def _unindex_entry(self, key: bytes) -> None:
        """Drop an entry from the metadata index and its segment's key set.

        Caller holds ``_lock``.
        """
        info = self._metadata.pop(key, None)
        if info is not None:
            keys = self._segment_keys.get(info.segment_id)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._segment_keys.pop(info.segment_id, None)

    def _remove_segment_sidecar(self, segment_id: int) -> None:
        """Best-effort unlink of a segment's sidecar (safe if absent)."""
        try:
            self._layout.sidecar_path(segment_id).unlink(missing_ok=True)
        except OSError:
            pass

    def _open_segment_mmap(self, segment_id: int, path) -> None:
        """Map a sealed segment read-only so its entries can be served."""
        self._segment_mmaps[segment_id] = pa.memory_map(str(path), "r")
        self._segment_schemas.pop(segment_id, None)

    def _forget_segment_mmap(self, segment_id: int) -> None:
        """Close and drop a segment's read mapping, and its cached schema."""
        mmap = self._segment_mmaps.pop(segment_id, None)
        if mmap is not None:
            mmap.close()
        self._segment_schemas.pop(segment_id, None)

    def _segment_schema(self, segment_id: int, mmap) -> Optional[pa.Schema]:
        """Schema of a mapped segment, read from its leading message once.

        Returns None if that message can't be read, which sends the caller down
        the sequential fallback.
        """
        schema = self._segment_schemas.get(segment_id)
        if schema is not None:
            return schema
        try:
            mmap.seek(0)
            schema = pa.ipc.open_stream(mmap).schema
        except (pa.ArrowInvalid, OSError, EOFError, StopIteration):
            return None
        self._segment_schemas[segment_id] = schema
        return schema

    def _drop_segment_files(self, segment_id: int) -> None:
        """Discard a bad segment: close its mmap and unlink both ``.arrow`` and
        ``.idx``. Used for a legacy/corrupt segment on the boot fallback path."""
        self._forget_segment_mmap(segment_id)
        seg_file = self._segment_path(segment_id)
        try:
            seg_file.unlink()
        except OSError:
            pass
        self._remove_segment_sidecar(segment_id)

    def _scan_segment_records(self, seg_file: Path) -> Optional[SegmentScan]:
        """Walk a sealed segment's body for its index records, or None if it
        cannot be indexed (see :func:`segment_index.scan_segment_records`).

        A method rather than a direct call so the single body-reading chokepoint
        stays overridable -- both the boot fallback walk and the seal-time
        sidecar fallback go through it.
        """
        return segment_index.scan_segment_records(seg_file)

    def _install_segment_records(
        self,
        segment_id: int,
        seg_file: Path,
        records: List[IndexRecord],
        retention: RetentionClass = "normal",
    ) -> None:
        """Populate ``_metadata`` + pool queues for one segment from its index
        records. Shared by the sidecar fast path and the body-walk fallback so
        both produce an identical index. Entries in a segment share a size class
        (the pool key), so the last one's class keys the segment's pool -- same as
        the pre-#300 walk. Caller has already rejected an empty record list.

        ``retention`` comes from the sidecar where there is one. A body walk has
        no record of it and takes the default, which costs that segment only its
        place in the reclamation order, until it turns over.
        """
        st = seg_file.stat()
        segment_created = st.st_mtime  # file mtime, matching the pre-#300 walk
        for record in records:
            self._index_entry(
                record.key,
                SegmentEntryInfo(
                    segment_id=segment_id,
                    offset=record.offset,  # entry index for the sequential reader
                    size_bytes=record.size_bytes,
                    created_at=segment_created,
                    last_access_time=segment_created,
                    byte_offset=record.byte_offset,
                    byte_length=record.byte_length,
                ),
            )

        pool_key = (retention, _get_size_class(records[-1].size_bytes))
        pool_queue = self._get_or_create_pool_queue(pool_key)
        self._segment_pool_key[segment_id] = pool_key
        # Oldest at the tail: a rebuilt segment predates this session's writes.
        pool_queue.add_segment(
            SieveKSegmentInfo(
                segment_id=segment_id,
                size_bytes=st.st_size,
                created_at=segment_created,
                last_access_time=segment_created,
                entry_count=len(records),
            ),
            newest=False,
        )

    def _load_segment_from_sidecar(self, segment_id: int, seg_file: Path) -> bool:
        """Restore a segment's index from its ``.idx`` sidecar without reading the
        body. Returns True on the fast path; False (fall back to the walk) if the
        sidecar is absent, the wrong version, or stale. A read-only mmap of the
        segment is still created (cheap, and required to serve reads) -- only the
        per-batch body reads are skipped.
        """
        loaded = segment_index.read_sidecar(
            self._layout.sidecar_path(segment_id), seg_file
        )
        if loaded is None:
            return False
        records, retention = loaded
        self._open_segment_mmap(segment_id, seg_file)
        self._install_segment_records(segment_id, seg_file, records, retention)
        return True

    def _write_segment_sidecar(
        self, segment_id: int, records: Optional[List[IndexRecord]] = None
    ) -> None:
        """Persist a sealed segment's key -> byte-range index to ``seg_NNNN.idx``.

        The seal-time entry point (natural rotation and graceful close). Since
        #541 the caller can pass the records straight from the live index
        (``_index_records_for_segment``), which is the same data the body holds
        -- re-reading the segment it just finished cost 2-83 ms per rotation
        under ``_write_lock``, and ``close()`` paid it per open segment on the
        shutdown deadline #300 exists to protect. Falls back to walking the body
        when the caller has no records (an entry missing its byte range), since
        the walk can still derive one.
        """
        seg_file = self._segment_path(segment_id)
        if not seg_file.exists():
            return  # segment already gone (evicted); nothing to index
        if records is None:
            scan = self._scan_segment_records(seg_file)
            records = scan.records if scan else None
        if records:
            self._write_sidecar_from_records(segment_id, records)

    def _write_sidecar_from_records(
        self, segment_id: int, records: List[IndexRecord]
    ) -> None:
        """Write ``seg_NNNN.idx`` from already-computed index records, tagged with
        the segment's retention class (see :func:`segment_index.write_sidecar`).
        """
        pool_key = self._get_pool_key_for_segment(segment_id)
        segment_index.write_sidecar(
            self._layout.sidecar_path(segment_id),
            self._segment_path(segment_id),
            records,
            pool_key[0] if pool_key else "normal",
        )

    def _create_segment_for_pool(
        self,
        pool_key: Tuple[RetentionClass, SizeClass],
        schema: pa.Schema,
    ) -> int:
        """Create a new segment for a specific pool and return its ID."""
        segment_id = self._next_segment_id
        self._next_segment_id += 1

        self._layout.segments_dir.mkdir(parents=True, exist_ok=True)
        segment_path = self._segment_path(segment_id)

        # Create writer.
        #
        # INVARIANT (load-bearing for the localhost mmap fast path): no segment inode
        # is ever truncated or shrunk while a client may have it mapped. A remote
        # client hands out a zero-copy view onto this file's mapping, so it can fault
        # (SIGBUS) at *any* point in that array's life if the bytes under the mapping
        # vanish. This truncating "wb" open is safe only because `segment_id` is strictly
        # monotonic (`_next_segment_id`, boot-initialized to max+1 and only incremented),
        # so `segment_path` is always freshly allocated -- never an id a live mapping
        # holds. Eviction *unlinks* segments (the inode survives to last close);
        # nothing ever truncates one in place. Do not add a "reuse a segment file"
        # or "truncate on repair" path without breaking that view contract first.
        sink = pa.OSFile(str(segment_path), "wb")
        writer = pa.RecordBatchStreamWriter(sink, schema)

        pool_queue = self._get_or_create_pool_queue(pool_key)

        # Track segment in pool queue (at head since newest)
        pool_queue.add_segment(
            SieveKSegmentInfo(
                segment_id=segment_id,
                size_bytes=0,
                created_at=time.time(),
                last_access_time=time.time(),
                entry_count=0,
            ),
            newest=True,
        )

        # Register in pool tracking
        self._pool_writers[segment_id] = writer
        self._pool_sinks[segment_id] = sink
        self._pool_paths[segment_id] = segment_path
        self._open_pools[pool_key] = segment_id
        self._segment_pool_key[segment_id] = pool_key

        return segment_id

    def _get_or_create_pool_queue(
        self, pool_key: Tuple[RetentionClass, SizeClass]
    ) -> PoolQueueInfo:
        """Return the pool's queue, creating it on first use."""
        pool_queue = self._pool_queues.get(pool_key)
        if pool_queue is None:
            pool_queue = PoolQueueInfo(pool_key=pool_key)
            self._pool_queues[pool_key] = pool_queue
        return pool_queue

    def _close_writer(self, segment_id: int) -> None:
        """Close and forget a segment's stream writer and its backing sink.

        Both must be closed: RecordBatchStreamWriter.close() finalizes the IPC
        stream but leaves the OSFile sink open, so the write handle would linger
        and block unlink on Windows (issue #5).
        """
        writer, sink, _path = self._detach_open_segment(segment_id)
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()

    def _detach_open_segment(self, segment_id: int):
        """Pop every open-segment structure for ``segment_id`` and return its
        ``(writer, sink, path)`` for the caller to close.

        The one place that knows which maps track an open segment, so adding or
        retiring one is a single edit rather than four. Purely in-memory --
        closing the handles is the caller's job, deliberately, so it can happen
        outside ``_lock``.
        """
        writer = self._pool_writers.pop(segment_id, None)
        sink = self._pool_sinks.pop(segment_id, None)
        path = self._pool_paths.pop(segment_id, None)
        self._open_pools = {
            k: v for k, v in self._open_pools.items() if v != segment_id
        }
        return writer, sink, path

    def _open_segment_ids(self) -> list:
        """Segment ids with an open writer (the still-unsealed segments)."""
        return list(self._pool_writers)

    def _close_segment(self, segment_id: int) -> None:
        """Close a full segment's writer and reopen it read-only as an mmap.

        Caller must hold ``self._write_lock`` (serializes against other mutators)
        but must NOT hold ``self._lock``: ``writer.close()`` flushes buffered
        bytes (a blocking disk op), so it runs with ``self._lock`` released and
        the lock is taken only for the surrounding in-memory index mutations.
        """
        # Detach writer/sink/path from the index under the lock (in-memory only).
        with self._lock:
            writer, sink, path = self._detach_open_segment(segment_id)

        # Flush + close the handles WITHOUT self._lock (this is the blocking I/O;
        # both must close -- close() leaves the OSFile sink open, issue #5).
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()

        # Reopen read-only (mmap is a non-blocking read map) and drop redundant
        # in-memory copies, under the lock.
        records = None
        with self._lock:
            if path and path.exists():
                self._open_segment_mmap(segment_id, path)

                # The segment is now re-readable, so the in-memory RecordBatch
                # copies of its entries are redundant. Drop those no longer
                # referenced (the common sweep case: written, served, released
                # while the segment was still filling). Entries still referenced
                # keep their copy and are dropped on their own release() now that
                # the segment is readable. This bounds resident decoded data to
                # the open write segments plus what callers currently hold,
                # instead of every chunk ever cached.
                for k in list(self._segment_keys.get(segment_id, ())):
                    entry = self._entries.get(k)
                    if entry is not None and entry.is_evictable():
                        self._entries.pop(k, None)

                records = self._index_records_for_segment(segment_id)

        # The segment is now sealed and immutable: persist its sidecar index so
        # the next boot restores it without a body walk (biopb/biopb#300). Runs
        # outside self._lock (it writes a small file) so a stalled sidecar write
        # can never wedge the read path; still under the caller's
        # self._write_lock, so no append can race it.
        self._write_segment_sidecar(segment_id, records)

    def _get_total_size(self) -> int:
        """Get total size across all segments."""
        return sum(
            seg_info.size_bytes
            for pool in self._pool_queues.values()
            for seg_info in pool.segments.values()
        )

    def _get_total_segment_count(self) -> int:
        """Count total segments across all pools."""
        return sum(len(pool.queue) for pool in self._pool_queues.values())

    def _get_pool_key_for_segment(
        self, segment_id: int
    ) -> Optional[Tuple[RetentionClass, SizeClass]]:
        """Get the pool key for a given segment ID.

        O(1) via ``_segment_pool_key`` -- a segment's pool is fixed at
        creation, so this used to scan every pool on every cache hit
        (``_update_segment_frequency`` and ``_segment_info`` both call it).
        """
        return self._segment_pool_key.get(segment_id)

    def _segment_is_evictable(self, segment_id: int) -> bool:
        """True if no entry in this segment is currently referenced."""
        for key in self._segment_keys.get(segment_id, ()):
            entry = self._entries.get(key)
            if entry and entry.ref_count > 0:
                return False
        return True

    def _do_evict_segment(self, segment_id: int) -> None:
        """Actually evict a segment: remove files, metadata, and mmap."""
        for key in list(self._segment_keys.get(segment_id, ())):
            self._unindex_entry(key)
            # Also remove from in-memory entries if present
            self._entries.pop(key, None)
        self._segment_keys.pop(segment_id, None)
        self._segment_pool_key.pop(segment_id, None)

        # Close any open writer (and its sink) for this segment
        self._close_writer(segment_id)

        self._forget_segment_mmap(segment_id)

        # Delete segment file and its sidecar index (whole-segment eviction).
        seg_file = self._segment_path(segment_id)
        if seg_file.exists():
            seg_file.unlink()
        self._remove_segment_sidecar(segment_id)

        self._evictions += 1

    def _select_pool_for_eviction(self) -> Optional[Tuple[RetentionClass, SizeClass]]:
        """Select the pool to reclaim from: cheapest retention class first, then
        lowest aggregate hit rate within the class.

        The class outranks the hit rate deliberately -- a cheap chunk is one that
        costs least to produce again, so it is reclaimed before a normal one
        however well it is being hit. A ``pinned`` pool is never selected;
        nothing declares that class today.
        """

        def order(pool_key: Tuple[RetentionClass, SizeClass]):
            pool = self._pool_queues[pool_key]
            return (EVICTION_RANK[pool_key[0]], pool.hit_rate, -len(pool.queue))

        # A class absent from EVICTION_RANK (pinned) is never a victim.
        candidates = [
            pool_key
            for pool_key, pool in self._pool_queues.items()
            if pool.queue and pool_key[0] in EVICTION_RANK
        ]
        return min(candidates, key=order, default=None)

    def _evict_segment_sieve_k(self) -> bool:
        """Per-pool Sieve-K sweep following reference algorithm.

        Returns True if evicted, False if nothing evictable.
        """
        target_pool = self._select_pool_for_eviction()
        if target_pool is not None:
            pool_queue = self._pool_queues[target_pool]
            seg_id = pool_queue.select_victim(self._segment_is_evictable)
            if seg_id is not None:
                self._do_evict_segment(seg_id)
                return True

        # Nothing to evict
        self._ref_held_skips += 1
        return False

    def _reopen_segment_mmap(
        self, segment_id: int, seg_info: SieveKSegmentInfo
    ) -> None:
        """Reopen mmap for cold segment that was accessed.

        Accounting for the access belongs to the caller
        (``_update_segment_frequency``), so this only restores the mapping.
        """
        path = self._segment_path(segment_id)
        if path.exists():
            self._open_segment_mmap(segment_id, path)
            seg_info.mmap_released = False

    def _maybe_release_cold_mmaps(self) -> None:
        """Release mmap handles for cold segments.

        Only runs if total segments exceed MMAP_LIFECYCLE_THRESHOLD.
        """
        if self._get_total_segment_count() <= MMAP_LIFECYCLE_THRESHOLD:
            return  # Skip for small caches

        now = time.time()
        for _pool_key, pool in self._pool_queues.items():
            for seg_id, seg_info in pool.segments.items():
                age = now - seg_info.last_access_time
                if seg_info.is_cold() and age > COLD_THRESHOLD_SECONDS:
                    self._forget_segment_mmap(seg_id)
                    seg_info.mmap_released = True

    def _read_batch_from_segment(
        self, key: bytes, touch: bool = True
    ) -> Optional[pa.RecordBatch]:
        """Read one cached chunk back out of its segment file.

        The read is zero-copy on every platform including Windows: the
        returned batch's buffers point straight into the segment mapping, and
        a caller can keep it alive across eviction of that segment.

        Serve the unified binary schema as-is (biopb/biopb#293); just strip
        the internal cache-key column so the wire batch is the clean
        [data, shape, dtype]. No binary->typed conversion.
        """
        entry_info = self._metadata.get(key)
        if entry_info is None:
            return None

        segment_id = entry_info.segment_id
        seg_info = self._segment_info(segment_id)
        if seg_info is not None and seg_info.mmap_released:
            self._reopen_segment_mmap(segment_id, seg_info)
        # A probe reads the bytes without crediting the segment; see
        # ArrowFileBackend.try_acquire for why. The cold-mmap sweep is driven off the
        # same accounting: a read that declines to be credited must not schedule
        # it either, or a scaled read over a cache warmed minutes ago would walk
        # its own working set and munmap the segments the next unit reads.
        if touch:
            self._update_segment_frequency(segment_id)

        mmap = self._segment_mmaps.get(segment_id)
        if mmap is None:
            return None

        if touch:
            self._access_counter += 1
            if self._access_counter % 100 == 0:
                self._maybe_release_cold_mmaps()

        batch = self._read_batch_at(segment_id, mmap, entry_info, key)
        if batch is None:
            return None

        return pa.RecordBatch.from_arrays(
            [batch.column("data"), batch.column("shape"), batch.column("dtype")],
            names=["data", "shape", "dtype"],
        )

    def _batch_at_offset(
        self, segment_id: int, mmap, entry_info: SegmentEntryInfo, key: bytes
    ) -> Optional[pa.RecordBatch]:
        """The batch at this entry's recorded range, if it really is this entry.

        The check that the range still names *key* is the codec's
        (``segment_index.batch_at_offset``); what is this class's is which
        segment the mapping belongs to, and the warning that says the index and
        the body have drifted. None sends the caller down the sequential walk,
        which is slower but finds the right record.
        """
        schema = self._segment_schema(segment_id, mmap)
        if schema is None:
            return None
        batch = batch_at_offset(mmap, schema, entry_info.byte_offset, key)
        if batch is None:
            logger.warning(
                "segment %s offset %s does not hold the entry it is indexed "
                "under; walking the segment instead",
                segment_id,
                entry_info.byte_offset,
            )
        return batch

    def _read_batch_at(
        self, segment_id: int, mmap, entry_info: SegmentEntryInfo, key: bytes
    ) -> Optional[pa.RecordBatch]:
        """Decode the single record batch this entry points at.

        Seeks straight to the entry's recorded byte range rather than walking
        the IPC stream to reach it: an entry carries its range from birth since
        #541, and the walk cost O(entries before it) per hit -- 0.3 ms into a
        76-entry segment, 8.9 ms into a 3000-entry one. An entry without a range,
        or a segment whose schema can't be read, falls back to walk the segment.
        """
        if entry_info.byte_offset and entry_info.byte_length:
            batch = self._batch_at_offset(segment_id, mmap, entry_info, key)
            if batch is not None:
                return batch

        # No usable range: walk the stream to the entry's index.
        mmap.seek(0)
        for i, batch in enumerate(pa.RecordBatchStreamReader(mmap)):
            if i == entry_info.offset:
                return batch
        return None

    def _segment_info(self, segment_id: int) -> Optional[SieveKSegmentInfo]:
        """Sieve-K bookkeeping for a segment, or None if it has none."""
        pool_key = self._get_pool_key_for_segment(segment_id)
        if pool_key is None:
            return None
        pool_queue = self._pool_queues.get(pool_key)
        if pool_queue is None:
            return None
        return pool_queue.segments.get(segment_id)

    def _bracket_written_message(
        self,
        segment_id: int,
        write_start: int,
        write_end: int,
    ) -> Tuple[int, int]:
        """Byte range of the message just appended (``segment_index``'s codec).

        Only the path lookup is this class's: the bracketing itself is shared
        with the uploaded-member store, so both write ranges a reader resolves
        the same way. Caller holds ``_write_lock`` (which keeps the segment's
        writer/sink state stable), not ``_lock``.
        """
        path = self._pool_paths.get(segment_id)
        if path is None:
            return 0, 0
        return bracket_message(path, write_start, write_end)

    def locate_entry(self, key: bytes) -> Optional[ChunkLocation]:
        """Return the on-disk location of a cached chunk, or None.

        Backs the localhost cache-file handoff (issue #9). Byte ranges are
        recorded when the entry is written and restored at boot from the ``.idx``
        sidecar or the segment walk, so this derives nothing: an index lookup
        under ``_lock``, then one check that the range really names this entry
        (below). Returns None when the key isn't cached, has no recorded range,
        its segment is gone, or the range doesn't hold it -- every case
        signalling the caller to fall back to do_get.
        """
        with self._lock:
            entry_info = self._metadata.get(key)
            if entry_info is None:
                return None
            # byte_offset == 0 is never a real entry -- the schema message always
            # occupies the start of the segment -- so 0 means "no range known".
            if not entry_info.byte_offset or not entry_info.byte_length:
                return None
            location = self._build_chunk_location(entry_info)
            if location is None:
                return None

            # A served locate is a genuine cache hit: the client is about to map
            # the segment. Counting it keeps `stats().hits` meaningful on the
            # single-machine deployment, where hits take this path while misses
            # fall back to do_get and are counted there (biopb/biopb#514).
            self._hits += 1
            self._update_segment_frequency(entry_info.segment_id)

        # A locate hands a byte range to another process, which reads it with
        # the server no longer in the loop, so the range is checked before it is
        # published -- a range that does not name this entry would hand that
        # client someone else's pixels silently. ~1 us against a ~290 us locate
        # RTT, and the client is about to fault the same page anyway.
        #
        # Outside the lock: `_lock` guards the in-memory index only, and I/O
        # under it deadlocks (biopb/biopb#302).
        if not self._range_holds_key(location, entry_info, key):
            return None
        return location

    def _range_holds_key(
        self, location: ChunkLocation, entry_info: SegmentEntryInfo, key: bytes
    ) -> bool:
        """Does the recorded range really hold *key*'s record?

        Uses the live mapping when there is one. An open write segment has none
        and gets a short-lived map of its own. Never called under ``_lock``.
        """
        mmap = self._segment_mmaps.get(entry_info.segment_id)
        if mmap is not None:
            return (
                self._batch_at_offset(entry_info.segment_id, mmap, entry_info, key)
                is not None
            )
        try:
            with pa.memory_map(location.segment_path, "r") as scratch:
                return (
                    self._batch_at_offset(
                        entry_info.segment_id, scratch, entry_info, key
                    )
                    is not None
                )
        except (OSError, pa.ArrowInvalid):
            return False

    def _build_chunk_location(
        self, entry_info: SegmentEntryInfo
    ) -> Optional[ChunkLocation]:
        """Project an already-indexed entry onto its on-disk location.

        Pure -- the caller owns hit accounting. Returns None if the segment file
        has gone away (evicted/unlinked) since indexing. Caller holds ``_lock``.
        """
        segment_path = self._segment_path(entry_info.segment_id)
        try:
            generation_id = os.stat(segment_path).st_ino
        except OSError:
            return None

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
        retention: RetentionClass = "normal",
    ) -> CacheEntry:
        """Get existing entry or create pending and compute.

        Delegates the READY/PENDING/hydrate/miss dispatch to ``start_compute``,
        which leaves the returned entry acquired exactly once on every path
        (including hydrate-from-segment) -- so all that is left to do here is
        run ``compute_fn`` for an owned miss and wait for a PENDING entry
        (ours or another thread's) to become READY.

        ``retention`` is read only when this call is the one that creates the
        entry; a hit keeps the class the first writer declared.

        Returns an entry in state READY with ref_count >= 1. The caller must
        ``release`` it.
        """
        entry, is_owner = self.start_compute(key, retention)

        if is_owner:
            try:
                data, size_bytes = compute_fn()
                self.complete_entry(key, data, size_bytes)
            except Exception as e:
                self.fail_entry(key, e)
                raise

        if entry.state == EntryState.PENDING and not entry.wait_ready(
            self._config.pending_timeout
        ):
            raise TimeoutError("Cache computation timed out for key")

        return entry

    def contains(self, key: bytes) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry.state == EntryState.READY
            # No live entry, but the segment index still knows it: that is a
            # hit this backend can serve, by hydrating in try_acquire.
            return key in self._metadata

    def try_acquire(self, key: bytes, touch: bool = True) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return self._hydrate_from_segment(key, touch=touch)
            if entry.state != EntryState.READY:
                return None
            return self._acquire_ready_entry(entry, key, count_hit=False, touch=touch)

    def _acquire_ready_entry(
        self, entry: CacheEntry, key: bytes, *, count_hit: bool, touch: bool = True
    ) -> CacheEntry:
        """Acquire a READY entry, crediting a hit and/or touching its segment.

        Shared by every READY-entry path (``start_compute``, ``try_acquire``):
        a ``try_acquire`` probe does not count as a hit (see
        ``_hydrate_from_segment``) but still touches the segment when
        ``touch``; every other caller counts the hit. Caller holds ``_lock``.
        """
        entry.acquire()
        if count_hit:
            self._hits += 1
        if touch:
            entry_info = self._metadata.get(key)
            if entry_info:
                self._update_segment_frequency(entry_info.segment_id)
        return entry

    def _hydrate_hit(self, key: bytes) -> Optional[CacheEntry]:
        """``_hydrate_from_segment`` plus the hit credit every caller but the
        ``try_acquire`` probe wants. Caller holds ``_lock``.
        """
        hydrated = self._hydrate_from_segment(key)
        if hydrated is not None:
            self._hits += 1
        return hydrated

    def _hydrate_from_segment(
        self, key: bytes, touch: bool = True
    ) -> Optional[CacheEntry]:
        """Rebuild an acquired READY entry from its persisted segment, or None.

        The path taken when a key is on disk but has no live in-memory entry --
        either it was never in this session or ``release`` dropped the redundant
        mirror. Caller holds ``_lock``, and counts the hit if its call was a
        chunk request: a ``try_acquire`` probe is not one, and reaching the
        segment rather than the mirror must not change that.
        """
        if key not in self._metadata:
            return None
        batch = self._read_batch_from_segment(key, touch=touch)
        if batch is None:
            return None
        entry = CacheEntry(
            data=batch,
            state=EntryState.READY,
            created_at=time.time(),
            size_bytes=estimate_batch_bytes(batch),
        )
        entry.acquire()
        self._entries[key] = entry
        return entry

    def _update_segment_frequency(self, segment_id: int) -> None:
        """Count an access against a segment (Sieve-K counter + pool hit)."""
        pool_key = self._get_pool_key_for_segment(segment_id)
        pool_queue = self._pool_queues.get(pool_key) if pool_key else None
        seg_info = pool_queue.segments.get(segment_id) if pool_queue else None
        if seg_info is None:
            return
        pool_queue.record_hit(seg_info, time.time())

    def start_compute(
        self, key: bytes, retention: RetentionClass = "normal"
    ) -> Tuple[CacheEntry, bool]:
        """Reserve the key without computing: returns (entry, is_owner).

        The check-cache half of ``get_or_acquire``, split out for a caller that
        already holds the data (``CacheManager.put``) rather than a
        ``compute_fn``. ``is_owner=True`` means the returned entry is PENDING
        and this caller MUST go on to ``complete_entry`` or ``fail_entry`` it --
        anything else strands every reader of that key until
        ``pending_timeout``.

        Every returned entry is acquired exactly once, on every path --
        ``get_or_acquire`` relies on this to avoid acquiring a second time.
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is not None:
                if entry.state == EntryState.READY:
                    return (
                        self._acquire_ready_entry(entry, key, count_hit=True),
                        False,
                    )

                if entry.state == EntryState.PENDING:
                    # Someone else computing - acquire and wait
                    entry.acquire()
                    self._pending_waits += 1
                    return entry, False

            hydrated = self._hydrate_hit(key)
            if hydrated is not None:
                return hydrated, False

            # No entry - create pending, we own computation
            entry = CacheEntry(
                state=EntryState.PENDING,
                created_at=time.time(),
                retention=retention,
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
        if self._skip_if_oversized(key, data, size_bytes):
            return
        self._persist_entry(key, data, size_bytes)

    def _skip_if_oversized(
        self, key: bytes, data: pa.RecordBatch, size_bytes: int
    ) -> bool:
        """Handle a chunk too large to cache; True if the caller should stop.

        An oversized chunk is still handed to the threads waiting on it -- the
        entry goes READY in memory -- it just is not stored.
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

    def _persist_entry(
        self,
        key: bytes,
        data: pa.RecordBatch,
        size_bytes: int,
    ) -> None:
        """Write one entry to its segment and index it.

        Runs on the caller's thread: the thread that computed the batch is the
        thread that persists it, and does not return until the bytes are down.
        The ``entry.state == PENDING`` guards below are what stop a write whose
        entry was failed or evicted while the unlocked disk write was in flight.
        """
        # Serialize the whole write/evict/close critical section. This is the
        # only mutator of writer/segment state, so holding _write_lock keeps the
        # selected segment stable across the unlocked write below.
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

                # compute_fn already emits the unified binary schema
                # (biopb/biopb#293), so store it directly -- no typed->binary cast.
                # Attach the cache key as a per-row column (NOT schema metadata):
                # Arrow IPC persists the schema once per segment, so schema
                # metadata can't identify individual batches on rebuild.
                keyed_batch = batch_with_key(data, key)

                # The class the entry was reserved with, not this call's.
                pool_key = (entry.retention, size_class)
                pool_queue = self._get_or_create_pool_queue(pool_key)
                pool_queue.record_miss()

                # Find or create the open segment for this pool.
                # _create_segment_for_pool registers writer and sink together,
                # so both lookups below are total.
                segment_id = self._open_pools.get(pool_key)
                if segment_id not in self._pool_writers:
                    segment_id = self._create_segment_for_pool(
                        pool_key, keyed_batch.schema
                    )
                writer = self._pool_writers[segment_id]
                sink = self._pool_sinks[segment_id]

            # ---- PHASE 2: blocking disk write, self._lock RELEASED ----
            # Flush so the bytes are durable in the page cache, and bracket the
            # message with the sink cursor so the localhost cache-file handoff
            # (issue #9) gets its byte range for free. A failure here (e.g.
            # ENOSPC) propagates out: get_or_acquire's owner branch turns it into
            # fail_entry() + re-raise. That is now safe -- the `with` blocks
            # release both locks on the way out, so a failed (or stalled) write
            # can no longer leave a lock held across the read path.
            write_start = sink.tell()
            writer.write_batch(keyed_batch)
            sink.flush()
            byte_offset, byte_length = self._bracket_written_message(
                segment_id, write_start, sink.tell()
            )

            # ---- PHASE 3: in-memory commit ----
            need_close = False
            with self._lock:
                entry = self._entries.get(key)
                if entry is None or entry.state != EntryState.PENDING:
                    # Raced away (failed/evicted while we wrote). The written
                    # bytes are harmless slack in the segment.
                    return

                now = time.time()
                seg_info = pool_queue.segments.get(segment_id)
                if seg_info:
                    seg_info.size_bytes += size_bytes
                    seg_info.entry_count += 1
                    seg_info.last_access_time = now

                self._index_entry(
                    key,
                    SegmentEntryInfo(
                        segment_id=segment_id,
                        offset=seg_info.entry_count - 1 if seg_info else 0,
                        # The index records the batch's buffer size, not the
                        # caller's `size_bytes`: a boot that has to walk the
                        # segment body can only measure buffers, so this is the
                        # one definition all three index producers (write,
                        # sidecar, walk) can agree on. `size_bytes` still drives
                        # the eviction budget and size class below, which is
                        # in-session state a walk never reconstructs.
                        size_bytes=estimate_batch_bytes(data),
                        created_at=now,
                        last_access_time=now,
                        byte_offset=byte_offset,
                        byte_length=byte_length,
                    ),
                )
                if entry.state == EntryState.PENDING:
                    entry.set_ready(data, size_bytes)
                need_close = bool(
                    seg_info and seg_info.size_bytes >= self._config.max_segment_bytes
                )

            # Rotate a full segment. _close_segment flushes the writer (blocking),
            # so it manages its own locking and runs with self._lock NOT held
            # (still under _write_lock, so the segment state is stable).
            if need_close:
                self._close_segment(segment_id)

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
            self._unindex_entry(key)
            return True

    def clear(self) -> None:
        """Clear all evictable entries and delete all segments."""
        with self._lock:
            # Close all pool writers (and their sinks)
            for segment_id in self._open_segment_ids():
                self._close_writer(segment_id)

            # Close all mmaps
            for segment_id in list(self._segment_mmaps):
                self._forget_segment_mmap(segment_id)

            # Delete all segment files and their sidecar indexes
            segments_dir = self._layout.segments_dir
            for seg_file in segments_dir.glob("seg_*.arrow"):
                seg_file.unlink()
            for idx_file in segments_dir.glob("seg_*.idx"):
                idx_file.unlink()

            # Clear tracking
            self._pool_queues.clear()
            self._metadata.clear()
            self._segment_keys.clear()
            self._segment_pool_key.clear()
            self._entries.clear()

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
            pool_stats[pool_name] = PoolStats(
                pool_key=pool_name,
                hits=pool.hits,
                misses=pool.misses,
                segments=len(pool.queue),
                bytes=sum(s.size_bytes for s in pool.segments.values()),
                hit_rate=pool.hit_rate,
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

    def release_process_lock(self) -> None:
        """Release the process lock, leaving handles OPEN.

        This is the cheap, upstream-independent half of :meth:`close`: it drops
        the cross-process lock so the next boot never sees a stale one, but it
        does NOT close segment writers/mmaps. Doing so mid-flight would race any
        ``do_get`` reads still draining. Completed writes are already flushed to
        their segment files -- every write lands on the thread that made it, so
        there is nothing to drain here -- and a write interrupted after this
        point leaves a torn tail that ``_rebuild_index_from_segments`` handles on
        its own. ``ProcessLock.release()`` is idempotent, so a later
        :meth:`close` re-releasing the lock is a harmless no-op.

        The cost of releasing early is that such an interruption is no longer
        *reported*: recovery keys off a stale lock, and this just released it
        cleanly. The segments stay correct either way.
        """
        with self._lock:
            if self._process_lock:
                self._process_lock.release()

    def close(self) -> None:
        """Close backend and release resources.

        This preserves segment files for persistence across restarts.
        Only releases locks and closes handles.

        Unconditional and unbounded-free: every write landed on the thread that
        made it, so by the time any caller can reach this there is nothing in
        flight to wait for and nothing a timeout could protect against. (When
        writes could be deferred this had to drain a background writer first,
        and on a failed drain leave the segments and the process lock behind
        for the next boot to recover -- biopb/biopb#815, removed.)
        """
        # Seal the still-open write segments (flush their streams), capturing
        # their index records so we can persist their sidecars below.
        # Rotation-sealed segments already have a sidecar and are not open.
        with self._lock:
            open_segment_ids = self._open_segment_ids()
            pending_sidecars = {
                segment_id: self._index_records_for_segment(segment_id)
                for segment_id in open_segment_ids
            }
            for segment_id in open_segment_ids:
                self._close_writer(segment_id)

        # Persist a sidecar for each just-sealed segment so the next boot skips
        # its body walk (biopb/biopb#300). Outside self._lock: writes one small
        # file per segment, and a stalled write must not wedge the read path.
        for segment_id, records in pending_sidecars.items():
            self._write_segment_sidecar(segment_id, records)

        with self._lock:
            self._open_pools.clear()

            # Close all mmap handles
            for segment_id in list(self._segment_mmaps):
                self._forget_segment_mmap(segment_id)

            # Release process lock
            if self._process_lock:
                self._process_lock.release()

            # Clear in-memory tracking (data persists in files)
            self._entries.clear()
            self._metadata.clear()
            self._segment_keys.clear()
            self._segment_pool_key.clear()
            self._pool_queues.clear()
