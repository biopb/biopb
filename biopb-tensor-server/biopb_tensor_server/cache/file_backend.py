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
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple

import pyarrow as pa

from biopb_tensor_server.cache.base import (
    CacheBackend,
    CacheEntry,
    CacheStats,
    ChunkLocation,
    EntryState,
    PoolStats,
    estimate_batch_bytes,
)
from biopb_tensor_server.cache.recovery import (
    PoolQueueInfo,
    ProcessLock,
    RecoveryStatus,
    SegmentEntryInfo,
    SieveKSegmentInfo,
    WriteAheadLog,
)

__all__ = ["ArrowFileBackend", "ArrowFileConfig", "ChunkLocation"]

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

# Name of the on-disk marker file (in the cache root, beside ``lock`` and
# ``wal.json``) recording the CACHE_FILE_FORMAT_VERSION the segments were written
# under. ``_enforce_format_version`` reads it at init and wipes the cache when it
# is missing or mismatched, so a segment-layout / key-composition change can't
# silently reuse incompatible on-disk segments. See _enforce_format_version.
FORMAT_VERSION_MARKER = "format_version"

# Per-segment sidecar index (biopb/biopb#300). Each sealed segment
# ``seg_NNNN.arrow`` gets a ``seg_NNNN.idx`` written at seal time recording every
# entry's key -> byte range, so boot restores the index from these small files
# instead of faulting the whole on-disk cache (tens of GB on a caching proxy)
# just to re-derive it. A sealed segment is immutable, so the sidecar needs no
# manifest or generation counter: a boot trusts a sidecar iff its recorded
# ``.arrow`` size matches the file on disk (a torn or mismatched sidecar falls
# back to the body walk). Purely additive -- an older server ignores ``.idx``
# (it globs ``.arrow``); a newer server on an old cache walks and backfills. The
# tiny ``.idx`` bytes are deliberately NOT counted toward ``max_total_bytes``.
SIDECAR_FORMAT_VERSION = 1
_SIDECAR_VERSION_KEY = b"biopb_sidecar_version"
_SIDECAR_SEGMENT_SIZE_KEY = b"biopb_segment_size"
_SIDECAR_VERSION_BYTES = str(SIDECAR_FORMAT_VERSION).encode()
# The key travels as a real column (not schema-only): a sidecar is an Arrow IPC
# FILE, so this is belt-and-suspenders, but it keeps the record self-describing.
_SIDECAR_SCHEMA = pa.schema(
    [
        pa.field("key", pa.binary()),
        pa.field("byte_offset", pa.int64()),
        pa.field("byte_length", pa.int64()),
        pa.field("size_bytes", pa.int64()),
        pa.field("offset", pa.int64()),
    ]
)


def _schema_message_length(path: Path) -> Optional[int]:
    """Byte length of the leading IPC schema message in a segment file.

    The stream writer buffers the schema until the first batch is written, so a
    segment's *first* append advances the sink cursor across schema + batch
    together; splitting them needs the schema message's own length. Let pyarrow
    read its own framing (rather than decoding the encapsulated-message header
    here) so this can't drift from what the writer emits -- it is the same
    ``read_message`` call the boot walk opens a segment with, on the ~160 bytes
    at the head of the file.

    Returns None if that read fails, which costs the *first* entry of this one
    segment its byte range: ``locate_entry`` reports it unavailable and the
    client transfers that chunk over do_get instead of mmap. Later entries in
    the segment bracket directly off the sink cursor and are unaffected, and the
    seal-time ``.idx`` sidecar re-derives the range from the segment body, so a
    restart restores the fast path for it.
    """
    try:
        with pa.OSFile(str(path), "rb") as f:
            pa.ipc.read_message(f)  # the leading schema message
            length = f.tell()
    except (OSError, pa.ArrowInvalid, EOFError, StopIteration):
        return None
    return length or None


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
        self._pool_queues: Dict[Tuple[str, SizeClass], PoolQueueInfo] = {}

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
        self._open_pools: Dict[Tuple[str, SizeClass], int] = {}

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
        segments_dir = self._segments_dir
        segments_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._config.cache_dir, 0o700)
        except OSError:
            pass

        # Take exclusive ownership of the cache dir. The lock is held on an open
        # descriptor for the life of this process, so it is the acquire itself
        # that answers "is someone else using this?" -- staleness is read after,
        # from the leftover owner record (biopb/biopb#544).
        lock_path = self._config.cache_dir / "lock"
        self._process_lock = ProcessLock(lock_path)

        if not self._process_lock.acquire():
            raise RuntimeError(
                f"Cannot acquire cache lock at {lock_path}. "
                "Another process is using the cache."
            )

        was_stale = self._process_lock.is_stale()
        if was_stale:
            prior = self._process_lock.prior_owner() or {}
            logger.warning(
                "Cache directory was not released cleanly (previous owner pid=%s); "
                "running recovery",
                prior.get("pid", "unknown"),
            )

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
        # mismatched marker wipes the on-disk cache (segments + WAL). Must run
        # ahead of WAL init and the index rebuild.
        wiped = self._enforce_format_version()

        self._wal = WriteAheadLog(self._wal_path)

        # Check for crash recovery. A version wipe already dropped the segments
        # and WAL, so there is nothing to recover from.
        if not wiped and (was_stale or self._wal.has_pending()):
            logger.info("Cache recovery: stale lock or pending WAL entries detected")
            self._recovery_status = self._recover()

        # Rebuild metadata index from segment files
        self._rebuild_index_from_segments()

        # Backfill the recovered-entry count from the rebuilt index: _recover()
        # deliberately skips the segment read (biopb/biopb#300), so the
        # authoritative count comes from the walk that had to happen anyway.
        if self._recovery_status is not None:
            self._recovery_status.recovered_entries = len(self._metadata)

        # Find next segment ID (segments are created lazily when first write happens)
        all_segment_ids = set()
        for pool in self._pool_queues.values():
            all_segment_ids.update(pool.queue)
        self._next_segment_id = max(all_segment_ids, default=0) + 1

    def _enforce_format_version(self) -> bool:
        """Wipe the on-disk cache when its segment format version != code.

        ``CACHE_FILE_FORMAT_VERSION`` is the segment message layout / cache-key
        encoding contract (see the constant). Segments written under one version
        cannot be safely reused after a layout or key-composition change: the
        boot rebuild would index them and the server would then serve mis-decoded
        or mis-keyed (stale) chunks. A marker file records the version the
        on-disk segments were written under, and a missing or mismatched marker
        drops them before the index rebuild.

        A cache dir with segments but no marker counts as a mismatch. Discarding
        a layout-compatible cache once is a safe, one-time re-fetch cost, whereas
        serving incompatible bytes is a silent correctness bug -- so we err
        toward wiping. Idempotent on a matching cache (returns False, no I/O
        beyond the marker read).

        Must run while the process lock is held and before WAL init / recovery /
        index rebuild read the segments. Returns True iff a (re)stamp happened
        (marker missing or mismatched), in which case the segments and WAL were
        just wiped and there is nothing to recover.
        """
        marker_path = self._config.cache_dir / FORMAT_VERSION_MARKER
        try:
            on_disk: Optional[int] = int(marker_path.read_text().strip())
        except (OSError, ValueError):
            # Missing marker (pre-enforcement / fresh dir) or an unparseable one
            # (torn write) -> treat as a mismatch and re-stamp.
            on_disk = None

        if on_disk == CACHE_FILE_FORMAT_VERSION:
            return False

        self._wipe_cache_contents(stale_version=on_disk)

        # Stamp the current version. Write to a temp file and atomically replace
        # so a crash mid-write cannot leave a torn marker that spuriously wipes a
        # good cache on the next boot.
        tmp_path = marker_path.with_suffix(".tmp")
        tmp_path.write_text(f"{CACHE_FILE_FORMAT_VERSION}\n")
        os.replace(tmp_path, marker_path)
        return True

    def _wipe_cache_contents(self, stale_version: Optional[int]) -> None:
        """Delete the version-sensitive on-disk state (segments + WAL), leaving
        the held process lock and marker in place, and recreate an empty
        ``segments`` dir. Logs loudly only when data was actually discarded (a
        fresh dir wipe is a silent no-op). Runs before any mmap is opened, so no
        segment is mapped and the removal is safe on Windows too (issue #5).

        Fail-closed on a partial wipe: ``shutil.rmtree(ignore_errors=True)`` can
        leave files behind without surfacing an error -- an NFS unlink that
        returns ESTALE/EIO, a permission glitch, a handle another process still
        holds. A surviving ``seg_*.arrow`` is exactly the incompatible segment
        we came to drop, and the boot rebuild would re-index and serve it as if
        current. So we verify the segments are gone and raise if any remain,
        refusing to start rather than serving mis-decoded/stale chunks -- a dead
        cache owner's lock is stale-reclaimable, an operator can clear the dir.
        (A stray ``.idx`` sidecar is harmless -- the rebuild globs ``.arrow`` --
        so the check targets bodies only.)
        """
        segments_dir = self._segments_dir
        wal_path = self._wal_path
        had_data = (segments_dir.exists() and any(segments_dir.iterdir())) or (
            wal_path.exists()
        )
        if had_data:
            logger.warning(
                "Cache format-version mismatch (on-disk=%s, code=%d): discarding "
                "the incompatible on-disk cache at %s before rebuild.",
                "unversioned" if stale_version is None else stale_version,
                CACHE_FILE_FORMAT_VERSION,
                self._config.cache_dir,
            )

        shutil.rmtree(segments_dir, ignore_errors=True)
        segments_dir.mkdir(parents=True, exist_ok=True)
        try:
            wal_path.unlink()
        except OSError:
            pass

        leftover = sorted(segments_dir.glob("seg_*.arrow"))
        if leftover:
            # _initialize releases the process lock on the way out, so an
            # aborted init doesn't wedge a retry or the next start.
            raise RuntimeError(
                f"Failed to clear the incompatible cache at {segments_dir}: "
                f"{len(leftover)} segment file(s) survived the wipe "
                f"(e.g. {leftover[0].name}). Refusing to start rather than serve "
                "them; remove the cache directory manually and retry."
            )

    def _recover(self) -> RecoveryStatus:
        """Recover from crash: drop incomplete writes recorded in the WAL.

        The recovered-entry accounting is deliberately cheap and does NOT read
        segment bodies (biopb/biopb#300). Iterating every record batch to count
        entries and sum ``batch.nbytes`` faults the entire cache in from disk --
        tens of GB on a caching-proxy server -- purely for one startup log line,
        and it duplicates the walk ``_rebuild_index_from_segments()`` does next
        anyway. So take the recovered byte total from the segment files' on-disk
        sizes (a ``stat``, no read) and let ``_initialize`` backfill
        ``recovered_entries`` from the rebuilt index.
        """
        lost_entries = 0

        # Clear WAL - pending (incomplete) writes never reached a segment, so lost.
        for key in self._wal.get_pending_keys():
            lost_entries += 1
            logger.warning(f"Cache recovery: lost pending write for key {key.hex()}")
        self._wal.clear()

        # Recovered byte total from file sizes -- no segment read (issue #300).
        # (This is the on-disk footprint; corrupt segments are dropped, and any
        # read errors are surfaced, by the rebuild pass that follows.)
        segments_dir = self._segments_dir
        recovered_bytes = 0
        for seg_file in segments_dir.glob("seg_*.arrow"):
            try:
                recovered_bytes += seg_file.stat().st_size
            except OSError:
                pass

        return RecoveryStatus(
            recovered_entries=0,  # backfilled from the rebuilt index in _initialize
            lost_entries=lost_entries,
            recovered_bytes=recovered_bytes,
            lost_bytes=0,
            errors=[],
        )

    def _rebuild_index_from_segments(self) -> None:
        """Rebuild the metadata index and pool queues from segment files at boot.

        Fast path (biopb/biopb#300): a sealed segment carries a ``seg_NNNN.idx``
        sidecar written at seal time with each entry's key -> byte range. When it
        validates (version + recorded ``.arrow`` size), the index is restored
        from it WITHOUT faulting the segment body -- the walk that once read the
        whole cache from disk (tens of GB, ~52-78 s on a caching proxy) now
        happens once, at seal, off the boot path. A missing / stale / legacy
        sidecar falls back to the pre-#300 body walk, which also backfills a
        fresh sidecar so the next boot is fast.
        """
        segments_dir = self._segments_dir
        seg_files = sorted(segments_dir.glob("seg_*.arrow"))
        walked = 0
        for seg_file in seg_files:
            try:
                segment_id = int(seg_file.stem.split("_")[1])
            except (ValueError, IndexError):
                continue
            try:
                if self._load_segment_from_sidecar(segment_id, seg_file):
                    continue
                # No usable sidecar: walk the body once (the pre-#300 path). An
                # empty walk means the segment holds no recoverable entry (a
                # torn first batch), so it is dropped rather than tracked -- it
                # occupies disk against max_total_bytes and can serve nothing.
                records = self._scan_segment_records(seg_file)
                if not records:
                    logger.warning(
                        f"Discarding legacy/corrupt cache segment (unreadable or "
                        f"without per-batch key column): {seg_file}"
                    )
                    self._drop_segment_files(segment_id)
                    continue
                self._open_segment_mmap(segment_id, seg_file)
                self._install_segment_records(segment_id, seg_file, records)
                walked += 1
                # Backfill a sidecar from the records we just read -- no second
                # body read -- so the next boot skips this segment's walk.
                self._write_sidecar_from_records(segment_id, records)
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

    @property
    def _segments_dir(self) -> Path:
        """Directory holding the segment bodies and their sidecar indexes."""
        return self._config.cache_dir / "segments"

    @property
    def _wal_path(self) -> Path:
        return self._config.cache_dir / "wal.json"

    def _segment_path(self, segment_id: int) -> Path:
        """Path of a segment's body file (``seg_NNNN.arrow``)."""
        return self._segments_dir / f"seg_{segment_id:04d}.arrow"

    def _sidecar_path(self, segment_id: int) -> Path:
        """Path of a segment's sidecar index file (``seg_NNNN.idx``)."""
        return self._segments_dir / f"seg_{segment_id:04d}.idx"

    def _index_records_for_segment(self, segment_id: int) -> Optional[list]:
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
                (
                    key,
                    info.byte_offset,
                    info.byte_length,
                    info.size_bytes,
                    info.offset,
                )
            )
        records.sort(key=lambda r: r[1])
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
            self._sidecar_path(segment_id).unlink(missing_ok=True)
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

    def _scan_segment_records(self, seg_file: Path) -> Optional[list]:
        """Walk a sealed segment's IPC stream, returning one index record per
        entry: ``(key, byte_offset, byte_length, size_bytes, offset)``.

        The single source of truth for how a segment body maps to index entries,
        reused by the boot fallback walk and the seal-time sidecar fallback so
        every index path agrees byte-for-byte. Reads message-by-message off a private
        mmap to bracket each record batch (issue #9 needs the byte range) and
        stops at the first unreadable/torn trailing message -- a prior partial
        write's slack. Returns None for a legacy/corrupt segment (no per-batch key
        column, or unreadable), signalling the caller to drop or skip it. The mmap
        is always closed (an open handle blocks unlink on Windows, issue #5).
        """
        try:
            mm = pa.memory_map(str(seg_file), "r")
        except (OSError, pa.ArrowInvalid):
            return None
        try:
            try:
                schema = pa.ipc.open_stream(mm).schema
            except Exception:
                return None
            # A segment written before the per-batch key-column fix stored the
            # key only in schema metadata, which an IPC stream collapses to the
            # first entry's key on read-back -- it cannot be indexed correctly.
            if CACHE_KEY_FIELD not in schema.names:
                return None
            mm.seek(0)
            pa.ipc.read_message(mm)  # consume the leading schema message
            records = []
            entry_index = 0
            while True:
                pos = mm.tell()
                try:
                    msg = pa.ipc.read_message(mm)
                except (pa.ArrowInvalid, EOFError, StopIteration, OSError):
                    break
                if msg is None:
                    break
                msg_len = mm.tell() - pos
                try:
                    batch = pa.ipc.read_record_batch(msg, schema)
                except Exception:
                    break
                key = batch.column(CACHE_KEY_FIELD)[0].as_py()
                if key is None:
                    continue
                size_bytes = sum(
                    batch.column(name).nbytes for name in ("data", "shape", "dtype")
                )
                records.append((key, pos, msg_len, size_bytes, entry_index))
                entry_index += 1
            return records
        finally:
            mm.close()

    def _install_segment_records(
        self, segment_id: int, seg_file: Path, records: list
    ) -> None:
        """Populate ``_metadata`` + pool queues for one segment from its index
        records. Shared by the sidecar fast path and the body-walk fallback so
        both produce an identical index. Entries in a segment share a size class
        (the pool key), so the last one's class keys the segment's pool -- same as
        the pre-#300 walk. Caller has already rejected an empty record list.
        """
        st = seg_file.stat()
        segment_created = st.st_mtime  # file mtime, matching the pre-#300 walk
        for key, byte_offset, byte_length, size_bytes, offset in records:
            self._index_entry(
                key,
                SegmentEntryInfo(
                    segment_id=segment_id,
                    offset=offset,  # entry index for the sequential reader
                    size_bytes=size_bytes,
                    created_at=segment_created,
                    last_access_time=segment_created,
                    byte_offset=byte_offset,
                    byte_length=byte_length,
                ),
            )

        pool_key = ("unified", _get_size_class(records[-1][3]))
        pool_queue = self._get_or_create_pool_queue(pool_key)
        # Oldest at the tail: a rebuilt segment predates this session's writes.
        pool_queue.queue.append(segment_id)
        pool_queue.segments[segment_id] = SieveKSegmentInfo(
            segment_id=segment_id,
            size_bytes=st.st_size,
            created_at=segment_created,
            last_access_time=segment_created,
            entry_count=len(records),
            frequency=0,
            mmap_released=False,
        )

    def _load_segment_from_sidecar(self, segment_id: int, seg_file: Path) -> bool:
        """Restore a segment's index from its ``.idx`` sidecar without reading the
        body. Returns True on the fast path; False (fall back to the walk) if the
        sidecar is absent, the wrong version, or stale (recorded ``.arrow`` size
        != the file on disk). The sidecar is read fully into RAM and its handle
        closed before we touch the segment, so it never blocks unlink (issue #5).
        A read-only mmap of the segment is still created (cheap, and required to
        serve reads) -- only the per-batch body reads are skipped.
        """
        idx_path = self._sidecar_path(segment_id)
        if not idx_path.exists():
            return False
        try:
            with pa.OSFile(str(idx_path), "rb") as f:
                reader = pa.ipc.open_file(f)
                meta = reader.schema.metadata or {}
                if meta.get(_SIDECAR_VERSION_KEY) != _SIDECAR_VERSION_BYTES:
                    return False
                recorded = meta.get(_SIDECAR_SEGMENT_SIZE_KEY)
                if recorded is None or int(recorded) != seg_file.stat().st_size:
                    return False
                table = reader.read_all()
        except (OSError, ValueError, pa.ArrowInvalid):
            return False  # unreadable / mismatched sidecar -> walk the body

        records = list(
            zip(
                table.column("key").to_pylist(),
                table.column("byte_offset").to_pylist(),
                table.column("byte_length").to_pylist(),
                table.column("size_bytes").to_pylist(),
                table.column("offset").to_pylist(),
                strict=True,  # columns of one table -> equal length
            )
        )
        if not records:
            return False  # nothing to install; let the body walk decide
        self._open_segment_mmap(segment_id, seg_file)
        self._install_segment_records(segment_id, seg_file, records)
        return True

    def _write_segment_sidecar(
        self, segment_id: int, records: Optional[list] = None
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
            records = self._scan_segment_records(seg_file)
        if records:
            self._write_sidecar_from_records(segment_id, records)

    def _write_sidecar_from_records(self, segment_id: int, records: list) -> None:
        """Write ``seg_NNNN.idx`` from already-computed index records, read at boot
        instead of walking the body (biopb/biopb#300). Written atomically (tmp +
        ``os.replace``) and best-effort: a failure (e.g. ENOSPC) only forfeits the
        fast path for this one segment next boot, so it is logged and swallowed
        rather than allowed to fail the write/close path.
        """
        if not records:
            # Empty sealed segment: no sidecar. Boot walks/no-ops it -- harmless.
            return
        seg_file = self._segment_path(segment_id)
        try:
            st = seg_file.stat()
        except OSError:
            return

        schema = _SIDECAR_SCHEMA.with_metadata(
            {
                _SIDECAR_VERSION_KEY: _SIDECAR_VERSION_BYTES,
                _SIDECAR_SEGMENT_SIZE_KEY: str(st.st_size).encode(),
            }
        )
        table = pa.Table.from_arrays(
            [
                pa.array([r[0] for r in records], type=pa.binary()),
                pa.array([r[1] for r in records], type=pa.int64()),
                pa.array([r[2] for r in records], type=pa.int64()),
                pa.array([r[3] for r in records], type=pa.int64()),
                pa.array([r[4] for r in records], type=pa.int64()),
            ],
            schema=schema,
        )

        idx_path = self._sidecar_path(segment_id)
        tmp_path = idx_path.parent / (idx_path.name + ".tmp")
        try:
            with pa.OSFile(str(tmp_path), "wb") as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    writer.write_table(table)
            os.replace(str(tmp_path), str(idx_path))
        except OSError as e:
            logger.warning(f"Could not persist cache sidecar {idx_path.name}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _create_segment_for_pool(
        self,
        pool_key: Tuple[str, SizeClass],
        schema: pa.Schema,
    ) -> int:
        """Create a new segment for a specific pool and return its ID."""
        segment_id = self._next_segment_id
        self._next_segment_id += 1

        self._segments_dir.mkdir(parents=True, exist_ok=True)
        segment_path = self._segment_path(segment_id)

        # Create writer
        sink = pa.OSFile(str(segment_path), "wb")
        writer = pa.RecordBatchStreamWriter(sink, schema)

        pool_queue = self._get_or_create_pool_queue(pool_key)

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
        self._open_pools[pool_key] = segment_id

        return segment_id

    def _get_or_create_pool_queue(
        self, pool_key: Tuple[str, SizeClass]
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
    ) -> Optional[Tuple[str, SizeClass]]:
        """Get the pool key for a given segment ID."""
        for pool_key, pool in self._pool_queues.items():
            if segment_id in pool.segments:
                return pool_key
        return None

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

        # Close any open writer (and its sink) for this segment
        self._close_writer(segment_id)

        self._forget_segment_mmap(segment_id)

        # Delete segment file and its sidecar index (whole-segment eviction).
        seg_file = self._segment_path(segment_id)
        if seg_file.exists():
            seg_file.unlink()
        self._remove_segment_sidecar(segment_id)

        self._evictions += 1

    def _select_pool_for_eviction(self) -> Optional[Tuple[str, SizeClass]]:
        """Select pool with lowest aggregate hit rate."""
        if not self._pool_queues:
            return None

        # Calculate hit rates
        pool_rates = []
        for pool_key, pool in self._pool_queues.items():
            total = pool.hits + pool.misses
            hit_rate = pool.hits / total if total > 0 else 0.0
            pool_rates.append((pool_key, hit_rate, pool.queue))

        # Select lowest hit rate pool with segments
        pool_rates.sort(
            key=lambda x: (x[1], -len(x[2]))
        )  # Lowest rate, then most segments
        for pool_key, _rate, queue in pool_rates:
            if queue:
                return pool_key
        return None

    def _evict_segment_sieve_k(self) -> bool:
        """Per-pool Sieve-K sweep following reference algorithm.

        Returns True if evicted, False if nothing evictable.
        """
        target_pool = self._select_pool_for_eviction()
        if target_pool is not None:
            pool_queue = self._pool_queues[target_pool]

            # One sweep of the hand per candidate, twice around at most: a
            # segment whose counter is still hot after a full pass has had it
            # decremented, so a second pass can reach zero.
            for _ in range(len(pool_queue.queue) * 2):
                # Wrap hand if it exceeds queue length
                if pool_queue.hand >= len(pool_queue.queue):
                    pool_queue.hand = 0

                # Get segment at hand offset from tail
                # deque: newest at left (index 0), oldest at right (index -1 is tail)
                # -1 - hand: tail is index -1, hand=0 → -1, hand=1 → -2, etc.
                idx = -1 - pool_queue.hand
                seg_id = pool_queue.queue[idx]
                seg_info = pool_queue.segments.get(seg_id)

                # Skip a segment with no info, or one still being served.
                if seg_info is None or not self._segment_is_evictable(seg_id):
                    pool_queue.hand += 1
                    continue

                if seg_info.frequency > 0:
                    # Hot segment: decrement counter, advance hand
                    seg_info.frequency -= 1
                    pool_queue.hand += 1
                    continue

                # Cold segment (frequency == 0): evict
                self._do_evict_segment(seg_id)
                pool_queue.segments.pop(seg_id, None)
                # O(n) deletion from deque, acceptable for <1000 segments
                del pool_queue.queue[idx]
                # Hand stays at this position for next eviction
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
                if (
                    seg_info.frequency <= COLD_FREQUENCY_THRESHOLD
                    and age > COLD_THRESHOLD_SECONDS
                ):
                    self._forget_segment_mmap(seg_id)
                    seg_info.mmap_released = True

    def _read_batch_from_segment(self, key: bytes) -> Optional[pa.RecordBatch]:
        """Read one cached chunk back out of its segment file.

        Seeks straight to the entry's recorded byte range rather than walking
        the IPC stream to reach it: an entry carries its range from birth since
        #541, and the walk cost O(entries before it) per hit -- 0.3 ms into a
        76-entry segment, 8.9 ms into a 3000-entry one, on every do_get hit
        (``release`` drops the in-RAM mirror, so hits really do re-read). An
        entry without a range, or a segment whose schema can't be read, falls
        back to that walk.
        """
        entry_info = self._metadata.get(key)
        if entry_info is None:
            return None

        segment_id = entry_info.segment_id
        seg_info = self._segment_info(segment_id)
        if seg_info is not None and seg_info.mmap_released:
            self._reopen_segment_mmap(segment_id, seg_info)
        self._update_segment_frequency(segment_id)

        mmap = self._segment_mmaps.get(segment_id)
        if mmap is None:
            return None

        # Periodic mmap cleanup check
        self._access_counter += 1
        if self._access_counter % 100 == 0:
            self._maybe_release_cold_mmaps()

        batch = self._read_batch_at(segment_id, mmap, entry_info)
        if batch is None:
            return None

        # Detach from the segment mmap on Windows so the file can be unlinked
        # during eviction even while a caller holds the batch (issue #5). POSIX
        # keeps the zero-copy mmap read.
        if self._copy_on_read:
            batch = _copy_batch_off_mmap(batch)
        # Serve the unified binary schema as-is (biopb/biopb#293); just strip
        # the internal cache-key column so the wire batch is the clean
        # [data, shape, dtype]. No binary->typed conversion.
        return pa.RecordBatch.from_arrays(
            [batch.column("data"), batch.column("shape"), batch.column("dtype")],
            names=["data", "shape", "dtype"],
        )

    def _read_batch_at(
        self, segment_id: int, mmap, entry_info: SegmentEntryInfo
    ) -> Optional[pa.RecordBatch]:
        """Decode the single record batch this entry points at.

        Seeks straight to the entry's recorded byte range instead of walking the
        IPC stream to reach it. Since biopb/biopb#541 an entry carries that range
        from birth -- recorded at write time, restored at boot from the ``.idx``
        sidecar or the segment walk -- and it is the same range ``locate_entry``
        already hands to a localhost client. Walking cost O(entries before the
        target) on *every* do_get hit, and hits really do re-read: ``release``
        drops the redundant in-RAM mirror once a segment is mmap-readable.

        Falls back to the walk when the entry has no range (the one case
        ``_bracket_written_message`` can't derive: a failed schema-length read on
        a segment's first append) or the segment's schema can't be read.
        """
        if entry_info.byte_offset and entry_info.byte_length:
            schema = self._segment_schema(segment_id, mmap)
            if schema is not None:
                try:
                    mmap.seek(entry_info.byte_offset)
                    return pa.ipc.read_record_batch(pa.ipc.read_message(mmap), schema)
                except (pa.ArrowInvalid, OSError, EOFError, StopIteration):
                    pass  # fall through to the sequential walk

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
        """Byte range of the message just appended, from the sink cursor.

        Recording the range at write time is what lets every later read and
        locate go straight to the entry instead of walking the segment for it
        (biopb/biopb#541).

        ``write_start == 0`` is the segment's first append, where the writer
        also emitted the schema message it had buffered; the batch starts after
        it, so its length is read back off the file. Returns ``(0, 0)`` for a
        range that can't be derived -- the entry is then served over do_get, the
        designed floor of this path. Caller holds ``_write_lock`` (which keeps
        the segment's writer/sink state stable), not ``_lock``.
        """
        if write_end <= write_start:
            return 0, 0
        if write_start > 0:
            return write_start, write_end - write_start

        path = self._pool_paths.get(segment_id)
        if path is None:
            return 0, 0
        schema_len = _schema_message_length(path)
        if schema_len is None or not 0 < schema_len < write_end:
            return 0, 0
        return schema_len, write_end - schema_len

    def locate_entry(self, key: bytes) -> Optional[ChunkLocation]:
        """Return the on-disk location of a cached chunk, or None.

        Backs the localhost cache-file handoff (issue #9). Byte ranges are
        recorded when the entry is written and restored at boot from the ``.idx``
        sidecar or the segment walk, so this derives nothing -- it is a dict
        lookup under ``_lock``. Returns None when the key isn't cached, has no
        recorded range, or its segment is gone, signalling the caller to fall
        back to do_get.
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
            return location

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
                hydrated = self._hydrate_from_segment(key)
                if hydrated is not None:
                    return hydrated

                # No entry - create pending, we own the computation
                entry = CacheEntry(
                    state=EntryState.PENDING,
                    created_at=time.time(),
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
        if entry.state == EntryState.PENDING and not entry.wait_ready(
            self._config.pending_timeout
        ):
            raise TimeoutError("Cache computation timed out for key")

        # Now acquire and return
        with self._lock:
            entry.acquire()
            return entry

    def _hydrate_from_segment(self, key: bytes) -> Optional[CacheEntry]:
        """Rebuild an acquired READY entry from its persisted segment, or None.

        The path taken when a key is on disk but has no live in-memory entry --
        either it was never in this session or ``release`` dropped the redundant
        mirror. Caller holds ``_lock``.
        """
        if key not in self._metadata:
            return None
        batch = self._read_batch_from_segment(key)
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
        self._hits += 1
        return entry

    def _update_segment_frequency(self, segment_id: int) -> None:
        """Count an access against a segment (Sieve-K counter + pool hit)."""
        pool_key = self._get_pool_key_for_segment(segment_id)
        pool_queue = self._pool_queues.get(pool_key) if pool_key else None
        seg_info = pool_queue.segments.get(segment_id) if pool_queue else None
        if seg_info is None:
            return
        # Sieve-K: increment counter on hit, saturating at K=2
        seg_info.frequency = min(K, seg_info.frequency + 1)
        seg_info.last_access_time = time.time()
        pool_queue.hits += 1

    def start_compute(self, key: bytes) -> Tuple[CacheEntry, bool]:
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

            hydrated = self._hydrate_from_segment(key)
            if hydrated is not None:
                return hydrated, False

            # No entry - create pending, we own computation
            entry = CacheEntry(
                state=EntryState.PENDING,
                created_at=time.time(),
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

                # compute_fn already emits the unified binary schema
                # (biopb/biopb#293), so store it directly -- no typed->binary cast.
                # Attach the cache key as a per-row column (NOT schema metadata):
                # Arrow IPC persists the schema once per segment, so schema
                # metadata can't identify individual batches on rebuild.
                unified_batch = data
                key_col = pa.array([key], type=pa.binary())
                batch_with_key = pa.RecordBatch.from_arrays(
                    list(unified_batch.columns) + [key_col],
                    names=list(unified_batch.schema.names) + [CACHE_KEY_FIELD],
                )

                pool_key = ("unified", size_class)
                pool_queue = self._get_or_create_pool_queue(pool_key)
                pool_queue.misses += 1

                # Find or create the open segment for this pool.
                # _create_segment_for_pool registers writer and sink together,
                # so both lookups below are total.
                segment_id = self._open_pools.get(pool_key)
                if segment_id not in self._pool_writers:
                    segment_id = self._create_segment_for_pool(
                        pool_key, batch_with_key.schema
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
            if self._wal:
                self._wal.log_pending(key)
            write_start = sink.tell()
            writer.write_batch(batch_with_key)
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

                seg_info = pool_queue.segments.get(segment_id)
                if seg_info:
                    seg_info.size_bytes += size_bytes
                    seg_info.entry_count += 1
                    seg_info.last_access_time = time.time()

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
                        created_at=time.time(),
                        last_access_time=time.time(),
                        byte_offset=byte_offset,
                        byte_length=byte_length,
                    ),
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
            segments_dir = self._segments_dir
            for seg_file in segments_dir.glob("seg_*.arrow"):
                seg_file.unlink()
            for idx_file in segments_dir.glob("seg_*.idx"):
                idx_file.unlink()

            # Clear tracking
            self._pool_queues.clear()
            self._metadata.clear()
            self._segment_keys.clear()
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

    def release_process_lock(self) -> None:
        """Release the process lock + clear the WAL, leaving handles OPEN.

        The graceful-shutdown fast path (biopb/biopb#300). This is the cheap,
        upstream-independent half of :meth:`close`: it clears the WAL and drops
        the cross-process lock so the next boot never sees a stale lock, but it
        deliberately does NOT close segment writers/mmaps -- doing so mid-flight
        would race any ``do_get`` reads still draining. Completed writes are
        already flushed to their segment files, and clearing the WAL early is
        safe because ``_rebuild_index_from_segments`` tolerates a torn tail
        message (it breaks on ArrowInvalid/EOF), so torn-write safety does not
        depend on the WAL surviving. ``ProcessLock.release()`` is idempotent, so
        a later :meth:`close` re-releasing the lock is a harmless no-op.
        """
        with self._lock:
            if self._wal:
                self._wal.clear()
            if self._process_lock:
                self._process_lock.release()

    def close(self) -> None:
        """Close backend and release resources.

        This preserves segment files for persistence across restarts.
        Only releases locks and closes handles.
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

            # Clear WAL (writes complete)
            if self._wal:
                self._wal.clear()

            # Release process lock
            if self._process_lock:
                self._process_lock.release()

            # Clear in-memory tracking (data persists in files)
            self._entries.clear()
            self._metadata.clear()
            self._segment_keys.clear()
            self._pool_queues.clear()
