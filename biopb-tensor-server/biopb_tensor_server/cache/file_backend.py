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
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pyarrow as pa

from biopb_tensor_server.cache.base import (
    CacheBackend,
    CacheEntry,
    CacheStats,
    EntryState,
    MAX_ARROW_BATCH_BYTES,
)
from biopb_tensor_server.cache.recovery import (
    ProcessLock,
    RecoveryStatus,
    SegmentEntryInfo,
    SegmentInfo,
    WriteAheadLog,
)


logger = logging.getLogger(__name__)


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
        self._lock = threading.Lock()

        # In-memory entries for pending/ready/error state management
        # Only stores small metadata; actual data is in segment files
        self._entries: Dict[bytes, CacheEntry] = {}

        # Metadata index: key -> SegmentEntryInfo (location in segment)
        self._metadata: Dict[bytes, SegmentEntryInfo] = {}

        # Segment tracking: segment_id -> SegmentInfo
        self._segments: OrderedDict[int, SegmentInfo] = OrderedDict()

        # Mmap handles for fast reads
        self._segment_mmaps: Dict[int, pa.MemoryMappedFile] = {}

        # Active segment for writing
        self._active_segment_id: int = 0
        self._active_segment_writer: Optional[pa.RecordBatchStreamWriter] = None
        self._active_segment_path: Optional[Path] = None

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._pending_waits: int = 0
        self._ref_held_skips: int = 0
        self._oversized_skips: int = 0

        # WAL and process lock
        self._wal: Optional[WriteAheadLog] = None
        self._process_lock: Optional[ProcessLock] = None

        # Recovery status (if recovered from crash)
        self._recovery_status: Optional[RecoveryStatus] = None

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize backend: create dirs, acquire lock, rebuild index."""
        # Create directories
        segments_dir = self._config.cache_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Acquire process lock (detect stale lock for crash recovery)
        lock_path = self._config.cache_dir / "lock"
        self._process_lock = ProcessLock(lock_path)

        if not self._process_lock.acquire():
            raise RuntimeError(
                f"Cannot acquire cache lock at {lock_path}. "
                "Another process is using the cache."
            )

        # Initialize WAL
        wal_path = self._config.cache_dir / "wal.json"
        self._wal = WriteAheadLog(wal_path)

        # Check for crash recovery
        if self._process_lock.is_stale() or self._wal.has_pending():
            logger.info("Cache recovery: stale lock or pending WAL entries detected")
            self._recovery_status = self._recover()

        # Rebuild metadata index from segment files
        self._rebuild_index_from_segments()

        # Find next segment ID and create active segment if needed
        self._active_segment_id = max(self._segments.keys(), default=0) + 1
        if len(self._segments) == 0:
            self._create_new_segment()

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
                with pa.memory_map(str(seg_file), 'r') as mmap:
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
        """Scan all segment files to rebuild metadata index."""
        segments_dir = self._config.cache_dir / "segments"

        for seg_file in sorted(segments_dir.glob("seg_*.arrow")):
            try:
                # Extract segment ID from filename
                seg_name = seg_file.stem  # e.g., "seg_0001"
                segment_id = int(seg_name.split("_")[1])

                # Memory-map for reading
                mmap = pa.memory_map(str(seg_file), 'r')
                self._segment_mmaps[segment_id] = mmap

                # Read all batches to extract keys and build index
                offset = 0
                entry_count = 0
                segment_size = seg_file.stat().st_size
                segment_created = seg_file.stat().st_mtime

                reader = pa.RecordBatchStreamReader(mmap)
                for batch in reader:
                    # Extract cache key from schema metadata
                    schema_meta = batch.schema.metadata or {}
                    key_hex = schema_meta.get(b'cache_key', None)
                    if key_hex:
                        key = bytes.fromhex(key_hex.decode('utf-8'))
                        batch_size = sum(col.nbytes for col in batch.columns)

                        # Update offset for next batch
                        # Note: Arrow IPC stream format doesn't expose exact offsets easily
                        # We use relative offsets within the segment for simplicity
                        entry_info = SegmentEntryInfo(
                            segment_id=segment_id,
                            offset=entry_count,  # Use entry index as "offset"
                            size_bytes=batch_size,
                            metadata={},
                            created_at=segment_created,
                            last_access_time=segment_created,  # Initialize with file mtime
                        )
                        self._metadata[key] = entry_info
                        entry_count += 1

                # Track segment info
                self._segments[segment_id] = SegmentInfo(
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

    def _create_new_segment(self) -> None:
        """Create a new segment file path for writing.

        Writer is created lazily when first batch is written.
        """
        segments_dir = self._config.cache_dir / "segments"
        self._active_segment_path = segments_dir / f"seg_{self._active_segment_id:04d}.arrow"

        # Ensure directory exists
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Track segment info (writer created on first write)
        self._segments[self._active_segment_id] = SegmentInfo(
            size_bytes=0,
            created_at=time.time(),
            last_access_time=time.time(),
            entry_count=0,
        )
        self._active_segment_writer = None

    def _close_active_segment(self) -> None:
        """Close active segment writer and open mmap for reading."""
        if self._active_segment_writer is not None:
            self._active_segment_writer.close()
            self._active_segment_writer = None

            # Open mmap for the now-complete segment
            if self._active_segment_path and self._active_segment_path.exists():
                mmap = pa.memory_map(str(self._active_segment_path), 'r')
                self._segment_mmaps[self._active_segment_id] = mmap

            self._active_segment_path = None

    def _get_total_size(self) -> int:
        """Get total size across all segments."""
        return sum(seg.size_bytes for seg in self._segments.values())

    def _evict_least_recently_used_segment(self) -> bool:
        """Evict the segment with oldest last_access_time.

        Returns True if evicted, False if nothing evictable.
        """
        if not self._segments:
            return False

        # Find segments with no entries holding references
        evictable_segments = []
        for seg_id, seg_info in self._segments.items():
            # Check if any entries in this segment have ref_count > 0
            has_refs = False
            for key, entry_info in self._metadata.items():
                if entry_info.segment_id == seg_id:
                    entry = self._entries.get(key)
                    if entry and entry.ref_count > 0:
                        has_refs = True
                        break

            if not has_refs:
                evictable_segments.append((seg_id, seg_info.last_access_time, seg_info.size_bytes))

        if not evictable_segments:
            self._ref_held_skips += 1
            return False

        # Evict oldest segment
        oldest_seg = min(evictable_segments, key=lambda x: x[1])
        seg_id = oldest_seg[0]

        # Remove all entries in this segment from metadata
        keys_to_remove = [k for k, e in self._metadata.items() if e.segment_id == seg_id]
        for key in keys_to_remove:
            self._metadata.pop(key, None)
            # Also remove from in-memory entries if present
            self._entries.pop(key, None)

        # Close and remove mmap
        mmap = self._segment_mmaps.pop(seg_id, None)
        if mmap:
            mmap.close()

        # Delete segment file
        segments_dir = self._config.cache_dir / "segments"
        seg_file = segments_dir / f"seg_{seg_id:04d}.arrow"
        if seg_file.exists():
            seg_file.unlink()

        # Remove from segments tracking
        self._segments.pop(seg_id, None)
        self._evictions += 1

        logger.debug(f"Evicted segment {seg_id}, {oldest_seg[2]} bytes")
        return True

    def _read_batch_from_segment(self, key: bytes) -> Optional[pa.RecordBatch]:
        """Read a batch from segment file using mmap."""
        entry_info = self._metadata.get(key)
        if entry_info is None:
            return None

        mmap = self._segment_mmaps.get(entry_info.segment_id)
        if mmap is None:
            return None

        # Seek back to beginning since reader exhausts the mmap
        mmap.seek(0)

        # Read all batches and find the right one by index
        # (Arrow IPC stream doesn't support direct offset access easily)
        reader = pa.RecordBatchStreamReader(mmap)
        for i, batch in enumerate(reader):
            if i == entry_info.offset:
                # Update last_access_time for segment
                seg_info = self._segments.get(entry_info.segment_id)
                if seg_info:
                    seg_info.last_access_time = time.time()
                    # Move to end of OrderedDict (most recently used)
                    self._segments.move_to_end(entry_info.segment_id)
                return batch

        return None

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
        """Mark pending entry as ready and write to segment."""
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

        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state != EntryState.PENDING:
                return

            # Evict if needed before storing
            while self._get_total_size() + size_bytes > self._config.max_total_bytes:
                if not self._evict_least_recently_used_segment():
                    break

            # Log pending write to WAL
            if self._wal:
                self._wal.log_pending(key)

            # Add cache key to batch schema metadata
            key_hex = key.hex()
            schema_meta = {b'cache_key': key_hex.encode('utf-8')}
            new_schema = data.schema.with_metadata(schema_meta)
            batch_with_key = data.cast(new_schema)

            # Write to active segment
            if self._active_segment_writer is None:
                self._create_new_segment()
                # Create writer with schema from first batch
                sink = pa.OSFile(str(self._active_segment_path), 'wb')
                self._active_segment_writer = pa.RecordBatchStreamWriter(sink, batch_with_key.schema)

            self._active_segment_writer.write_batch(batch_with_key)

            # Update segment info
            seg_info = self._segments.get(self._active_segment_id)
            if seg_info:
                seg_info.size_bytes += size_bytes
                seg_info.entry_count += 1
                seg_info.last_access_time = time.time()

            # Update metadata index
            self._metadata[key] = SegmentEntryInfo(
                segment_id=self._active_segment_id,
                offset=seg_info.entry_count - 1 if seg_info else 0,
                size_bytes=size_bytes,
                metadata=entry.metadata,
                created_at=time.time(),
                last_access_time=time.time(),
            )

            # Check if segment exceeds max size
            if seg_info and seg_info.size_bytes >= self._config.max_segment_bytes:
                self._close_active_segment()
                self._active_segment_id += 1
                self._create_new_segment()

            # Mark entry as ready
            entry.set_ready(data, size_bytes)

            # Commit WAL
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
        """Release reference to entry."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            return entry.release()

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
            # Close active segment
            self._close_active_segment()

            # Close all mmaps
            for mmap in self._segment_mmaps.values():
                mmap.close()
            self._segment_mmaps.clear()

            # Delete all segment files
            segments_dir = self._config.cache_dir / "segments"
            for seg_file in segments_dir.glob("seg_*.arrow"):
                seg_file.unlink()

            # Clear tracking
            self._segments.clear()
            self._metadata.clear()
            self._entries.clear()

            # Clear WAL
            if self._wal:
                self._wal.clear()

            # Create new active segment
            self._active_segment_id = 1
            self._create_new_segment()

    def stats(self) -> CacheStats:
        """Return cache statistics."""
        total_entries = len(self._metadata)
        total_bytes = self._get_total_size()

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
            # Close active segment writer if it has data
            if self._active_segment_writer is not None:
                self._active_segment_writer.close()
                self._active_segment_writer = None

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
            self._segments.clear()