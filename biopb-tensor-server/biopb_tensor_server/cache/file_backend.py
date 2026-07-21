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
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple

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
    together; splitting them needs the schema message's own length. Read it off
    the encapsulated-message header rather than re-serializing the schema, so it
    is exactly what the writer emitted: ``[continuation 0xFFFFFFFF][uint32
    metadata length][metadata, padded]`` with an empty body. Returns None if the
    header can't be read, leaving the entry unindexed for the lazy walk.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if len(head) < 8:
        return None
    first, metadata_len = struct.unpack("<II", head)
    if first == 0xFFFFFFFF:
        return 8 + metadata_len
    # Pre-V5 stream framing: a bare length prefix, no continuation marker.
    return 4 + first


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

        # Enforce the segment format-version contract now that we are the
        # exclusive owner and before anything reads the segments: a missing or
        # mismatched marker wipes the on-disk cache (segments + WAL). Must run
        # ahead of WAL init and the index rebuild.
        wiped = self._enforce_format_version()

        # Initialize WAL
        wal_path = self._config.cache_dir / "wal.json"
        self._wal = WriteAheadLog(wal_path)

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
        all_segment_ids.update(self._segments_legacy.keys())
        self._next_segment_id = max(all_segment_ids, default=0) + 1

    def _enforce_format_version(self) -> bool:
        """Wipe the on-disk cache when its segment format version != code.

        ``CACHE_FILE_FORMAT_VERSION`` is the segment message layout / cache-key
        encoding contract (see the constant). Segments written under one version
        cannot be safely reused after a layout or key-composition change: the
        boot rebuild would index them and the server would then serve mis-decoded
        or mis-keyed (stale) chunks. The constant existed but was inert -- only
        reported in ``cache_stats``, never checked at startup -- so any such
        change silently reused incompatible segments. This makes it enforced: a
        marker file records the version the on-disk segments were written under,
        and a missing or mismatched marker drops them before the index rebuild.

        A pre-enforcement cache dir has segments but no marker; that counts as a
        mismatch, so the first boot after this ships wipes it. Discarding a
        layout-compatible cache once is a safe, one-time re-fetch cost, whereas
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
        segments_dir = self._config.cache_dir / "segments"
        wal_path = self._config.cache_dir / "wal.json"
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
            # Release the lock we hold so a same-process retry (or a stale-lock
            # reclaim on the next start) isn't wedged by this aborted init.
            if self._process_lock is not None:
                self._process_lock.release()
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
        segments_dir = self._config.cache_dir / "segments"
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
        segments_dir = self._config.cache_dir / "segments"
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
                # No usable sidecar: walk the body once (the pre-#300 path).
                records = self._scan_segment_records(seg_file)
                if records is None:
                    logger.warning(
                        f"Discarding legacy/corrupt cache segment (unreadable or "
                        f"without per-batch key column): {seg_file}"
                    )
                    self._drop_segment_files(segment_id)
                    continue
                mmap = pa.memory_map(str(seg_file), "r")
                self._segment_mmaps[segment_id] = mmap
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

    def _sidecar_path(self, segment_id: int) -> Path:
        """Path of a segment's sidecar index file (``seg_NNNN.idx``)."""
        return self._config.cache_dir / "segments" / f"seg_{segment_id:04d}.idx"

    def _remove_segment_sidecar(self, segment_id: int) -> None:
        """Best-effort unlink of a segment's sidecar (safe if absent)."""
        try:
            self._sidecar_path(segment_id).unlink(missing_ok=True)
        except OSError:
            pass

    def _drop_segment_files(self, segment_id: int) -> None:
        """Discard a bad segment: close its mmap and unlink both ``.arrow`` and
        ``.idx``. Used for a legacy/corrupt segment on the boot fallback path."""
        mmap = self._segment_mmaps.pop(segment_id, None)
        if mmap is not None:
            mmap.close()
        seg_file = self._config.cache_dir / "segments" / f"seg_{segment_id:04d}.arrow"
        try:
            seg_file.unlink()
        except OSError:
            pass
        self._remove_segment_sidecar(segment_id)

    def _scan_segment_records(self, seg_file: Path) -> Optional[list]:
        """Walk a sealed segment's IPC stream, returning one index record per
        entry: ``(key, byte_offset, byte_length, size_bytes, offset)``.

        The single source of truth for how a segment body maps to index entries,
        reused by the boot fallback walk and the seal-time sidecar write so every
        index path agrees byte-for-byte. Reads message-by-message off a private
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
        the pre-#300 walk. An empty segment falls back to legacy tracking.
        """
        st = seg_file.stat()
        segment_size = st.st_size
        segment_created = st.st_mtime  # file mtime, matching the pre-#300 walk
        size_class: Optional[SizeClass] = None
        for key, byte_offset, byte_length, size_bytes, offset in records:
            self._metadata[key] = SegmentEntryInfo(
                segment_id=segment_id,
                offset=offset,  # entry index for the sequential reader
                size_bytes=size_bytes,
                metadata={},
                created_at=segment_created,
                last_access_time=segment_created,
                byte_offset=byte_offset,
                byte_length=byte_length,
            )
            size_class = _get_size_class(size_bytes)

        entry_count = len(records)
        if entry_count > 0 and size_class is not None:
            pool_key = ("unified", size_class)
            pool_queue = self._pool_queues.get(pool_key)
            if pool_queue is None:
                pool_queue = PoolQueueInfo(pool_key=pool_key)
                self._pool_queues[pool_key] = pool_queue
            # Oldest at the tail: a rebuilt segment predates this session's writes.
            pool_queue.queue.append(segment_id)
            pool_queue.segments[segment_id] = SieveKSegmentInfo(
                segment_id=segment_id,
                size_bytes=segment_size,
                created_at=segment_created,
                last_access_time=segment_created,
                entry_count=entry_count,
                frequency=0,
                mmap_released=False,
            )
        else:
            self._segments_legacy[segment_id] = SegmentInfo(
                size_bytes=segment_size,
                created_at=segment_created,
                last_access_time=segment_created,
                entry_count=entry_count,
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
        mmap = pa.memory_map(str(seg_file), "r")
        self._segment_mmaps[segment_id] = mmap
        self._install_segment_records(segment_id, seg_file, records)
        return True

    def _write_segment_sidecar(self, segment_id: int) -> None:
        """Persist a sealed segment's key -> byte-range index to ``seg_NNNN.idx``.

        The seal-time entry point (natural rotation and graceful close): derives
        the records from the immutable segment file via ``_scan_segment_records``
        -- so it needs no lock and no live ``_metadata``, and the sidecar is
        guaranteed to match what a boot walk would produce. (The boot backfill
        path already holds the records and calls ``_write_sidecar_from_records``
        directly, avoiding a second body read.)
        """
        seg_file = self._config.cache_dir / "segments" / f"seg_{segment_id:04d}.arrow"
        if not seg_file.exists():
            return  # segment already gone (evicted); nothing to index
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
        seg_file = self._config.cache_dir / "segments" / f"seg_{segment_id:04d}.arrow"
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

        # The segment is now sealed and immutable: persist its sidecar index so
        # the next boot restores it without a body walk (biopb/biopb#300). Runs
        # outside self._lock (it re-reads the file and writes a small one) so a
        # stalled sidecar write can never wedge the read path; still under the
        # caller's self._write_lock, so no append can race it.
        self._write_segment_sidecar(segment_id)

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

        # Delete segment file and its sidecar index (whole-segment eviction).
        segments_dir = self._config.cache_dir / "segments"
        seg_file = segments_dir / f"seg_{segment_id:04d}.arrow"
        if seg_file.exists():
            seg_file.unlink()
        self._remove_segment_sidecar(segment_id)

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
        for _pool_key, pool in self._pool_queues.items():
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
                # Serve the unified binary schema as-is (biopb/biopb#293); just
                # strip the internal cache-key column so the wire batch is the
                # clean [data, shape, dtype]. No binary->typed conversion.
                return pa.RecordBatch.from_arrays(
                    [
                        batch.column("data"),
                        batch.column("shape"),
                        batch.column("dtype"),
                    ],
                    names=["data", "shape", "dtype"],
                )

        return None

    def _bracket_written_message(
        self,
        segment_id: int,
        write_start: Optional[int],
        write_end: Optional[int],
    ) -> Tuple[int, int]:
        """Byte range of the message just appended, from the sink cursor.

        Recording the range here is what keeps a cache *miss* off a segment
        walk: entries used to land unindexed, so the next ``locate_entry``
        re-walked the whole active segment (up to ``max_segment_bytes``) to find
        where it went -- O(entries in the segment), measured at ~5 ms on a
        145 MB segment of 0.87 MB chunks and several times that for small-chunk
        sources, and paid only by the localhost fast path it was meant to
        accelerate (biopb/biopb#541).

        ``write_start == 0`` is the segment's first append, where the writer
        also emitted the schema message it had buffered; the batch starts after
        it, so its length is read back off the file. Returns ``(0, 0)`` for a
        range that can't be derived: ``locate_entry`` then reports the chunk
        unavailable and the client transfers it over do_get. Caller holds
        ``_write_lock`` (which keeps the segment's writer/sink state stable),
        not ``_lock``.
        """
        if write_start is None or write_end is None or write_end <= write_start:
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

        Backs the localhost cache-file handoff (issue #9). Returns None when the
        key isn't cached or has no recorded byte range, signalling the caller to
        fall back to do_get.

        Read-only and ``_lock``-only by construction: byte ranges are recorded
        when the entry is written (``_bracket_written_message``) and restored at
        boot from the ``.idx`` sidecar or the segment walk, so there is nothing
        left to derive here. The former lazy-derivation branch was both the
        O(entries-in-segment) cost on every cache miss (biopb/biopb#541) and the
        only place the read path took ``_write_lock`` -- so a write stalled on a
        full filesystem could block locates. Neither is possible now.
        """
        with self._lock:
            entry_info = self._metadata.get(key)
            if entry_info is None:
                return None
            # byte_offset == 0 is never a real entry -- the schema message always
            # occupies the start of the segment -- so 0 means "no range known",
            # and the client transfers the chunk over do_get instead.
            if not entry_info.byte_offset or not entry_info.byte_length:
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

        # The client is about to map this segment; treat the locate as a hit.
        # A localhost mmap handoff is a genuine cache hit, so count it in the
        # top-level counter too -- otherwise `stats().hits` (which the sidecar
        # surfaces as the cache hit-rate) trends to ~0 on the single-machine
        # deployment, where hits take this fast path but misses fall back to
        # do_get and are counted there (biopb/biopb#514). Bump `self._hits`
        # here, next to the per-pool bump, rather than inside
        # `_update_segment_frequency` -- that helper is also called from the
        # do_get accounting paths, which increment `self._hits` separately, so
        # bumping it in the helper would double-count. `_build_chunk_location`
        # is the sole owner of the locate-hit path, so this fires exactly once
        # per served locate. Caller holds `self._lock`.
        self._hits += 1
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
        if entry.state == EntryState.PENDING and not entry.wait_ready(
            self._config.pending_timeout
        ):
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
            # Flush so the bytes are durable in the page cache, and bracket the
            # message with the sink cursor so the localhost cache-file handoff
            # (issue #9) gets its byte range for free. A failure here (e.g.
            # ENOSPC) propagates out: get_or_acquire's owner branch turns it into
            # fail_entry() + re-raise. That is now safe -- the `with` blocks
            # release both locks on the way out, so a failed (or stalled) write
            # can no longer leave a lock held across the read path.
            if self._wal:
                self._wal.log_pending(key)
            write_start = sink.tell() if sink is not None else None
            writer.write_batch(batch_with_key)
            write_end = None
            if sink is not None:
                sink.flush()
                write_end = sink.tell()
            byte_offset, byte_length = self._bracket_written_message(
                segment_id, write_start, write_end
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

                self._metadata[key] = SegmentEntryInfo(
                    segment_id=segment_id,
                    offset=seg_info.entry_count - 1 if seg_info else 0,
                    size_bytes=size_bytes,
                    metadata=entry.metadata,
                    created_at=time.time(),
                    last_access_time=time.time(),
                    byte_offset=byte_offset,
                    byte_length=byte_length,
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

            # Delete all segment files and their sidecar indexes
            segments_dir = self._config.cache_dir / "segments"
            for seg_file in segments_dir.glob("seg_*.arrow"):
                seg_file.unlink()
            for idx_file in segments_dir.glob("seg_*.idx"):
                idx_file.unlink()

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
        # which they are so we can persist their sidecars below. Rotation-sealed
        # segments already have a sidecar and are not in the writer maps.
        with self._lock:
            open_segment_ids = list(self._pool_writers.keys() | self._pool_sinks.keys())
            for segment_id in open_segment_ids:
                self._close_writer(segment_id)

        # Persist a sidecar for each just-sealed segment so the next boot skips
        # its body walk (biopb/biopb#300). Outside self._lock: writes one small
        # file per segment, and a stalled write must not wedge the read path.
        for segment_id in open_segment_ids:
            self._write_segment_sidecar(segment_id)

        with self._lock:
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
