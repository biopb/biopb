"""Arrow segments in a directory of their own: the chunk cache's format, no cache.

An uploaded ``cache://`` member keeps the chunks exactly as they were uploaded
(``docs/upload-model.md``, Store formats), and the format it keeps them in is
the one the file cache already writes: one Arrow batch per chunk in
``seg_NNNN.arrow``, with a ``seg_NNNN.idx`` sidecar written at seal so the next
life restores the index without faulting the bodies (biopb/biopb#300).

**The format is shared; the budget is not.** A member's segments are the
tensor, not a cache of it: they sit under the member's own directory, outside
``max_total_bytes``, outside the eviction sweep and outside the retention
classes. What is shared with :mod:`~biopb_tensor_server.cache.file_backend` is
the codec and the index in :mod:`~biopb_tensor_server.cache.segment_index` --
code that writes and reads, nothing that deletes.

The key is the caller's and opaque here. A member keys by **bounds**, never by
chunk id: a chunk id carries the server's serving-semantics epoch, which moves
on an upgrade that changes what the bytes mean, and the bytes on disk do not.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import pyarrow as pa

from biopb_tensor_server.cache.segment_index import (
    IndexRecord,
    batch_at_offset,
    batch_with_key,
    batch_without_key,
    bracket_message,
    read_sidecar,
    scan_segment_records,
    write_sidecar,
)
from biopb_tensor_server.cache.types import ChunkLocation

__all__ = ["SegmentStore"]

logger = logging.getLogger(__name__)

# One segment per this many bytes, matching the cache's own default. An
# uploaded chunk is at most MAX_ARROW_BATCH_BYTES, so a segment holds several.
DEFAULT_MAX_SEGMENT_BYTES = 64 * 1024 * 1024

_SEGMENT_GLOB = "seg_*.arrow"


class _Placed(NamedTuple):
    """Where one entry sits: which segment, and where in it."""

    segment_id: int
    record: IndexRecord


class SegmentStore:
    """The segments under *directory*, appended to and read back by key.

    One writer at a time -- an upload's write path already serializes its
    appends -- but a read may run beside one, so the index and the handles are
    guarded.

    A member store is one tensor's worth of segments and is mapped only once
    something reads it, so there is no cold-mmap sweep here: the cache's exists
    because it maps every chunk anyone has ever read.
    """

    def __init__(
        self,
        directory: Path,
        *,
        max_segment_bytes: int = DEFAULT_MAX_SEGMENT_BYTES,
    ) -> None:
        self._dir = Path(directory)
        self._max_segment_bytes = max_segment_bytes
        self._lock = threading.RLock()
        self._index: Dict[bytes, _Placed] = {}
        self._counts: Dict[int, int] = {}
        self._mmaps: Dict[int, pa.MemoryMappedFile] = {}
        self._schemas: Dict[int, pa.Schema] = {}
        self._next_segment_id = 0
        # The one segment taking appends, if any. Sealing closes it and nothing
        # reopens a sealed segment, so an entry's bytes never move.
        self._open_id: Optional[int] = None
        self._writer: Optional[pa.RecordBatchStreamWriter] = None
        self._sink: Optional[pa.OSFile] = None
        self._open_bytes = 0
        self._closed = False

    # -- opening ---------------------------------------------------------------

    @classmethod
    def open(cls, directory: Path, **kwargs) -> SegmentStore:
        """The store at *directory*, with its index restored from disk.

        Each sealed segment's ``.idx`` sidecar first, its body only when the
        sidecar is absent or stale -- the two-step
        ``ArrowFileBackend._rebuild_index_from_segments`` takes, for the same
        reason: a boot that walks bodies faults every byte it indexes.
        """
        store = cls(directory, **kwargs)
        store._restore_index()
        return store

    def _restore_index(self) -> None:
        for seg_file in sorted(self._dir.glob(_SEGMENT_GLOB)):
            try:
                segment_id = int(seg_file.stem.split("_")[1])
            except (IndexError, ValueError):
                logger.warning(f"segment store: {seg_file} is not a segment; skipped")
                continue
            self._next_segment_id = max(self._next_segment_id, segment_id + 1)
            for record in self._records_of(seg_file):
                self._index[record.key] = _Placed(segment_id, record)
                self._counts[segment_id] = self._counts.get(segment_id, 0) + 1

    def _records_of(self, seg_file: Path) -> List[IndexRecord]:
        """One segment's entries, off its sidecar or, failing that, its body."""
        sidecar = read_sidecar(seg_file.with_suffix(".idx"), seg_file)
        if sidecar is not None:
            return sidecar[0]
        scan = scan_segment_records(seg_file)
        if scan is None:
            logger.warning(f"segment store: {seg_file} is unreadable; skipped")
            return []
        if scan.torn:
            # The last append never landed. Append-only with one writer, so it
            # cost exactly the entry the walk stopped at -- which reads back as
            # a chunk nobody uploaded, not as data.
            logger.warning(f"segment store: {seg_file} has a torn tail")
        return scan.records

    # -- writing ---------------------------------------------------------------

    def append(self, key: bytes, batch: pa.RecordBatch) -> None:
        """Store *batch* under *key*, in the open segment or a fresh one.

        Raises ``KeyError`` if *key* is already stored: a member is put-once,
        and a second write would strand a byte range a reader may already hold.
        Raises ``RuntimeError`` once the store is closed -- what a write that
        passed the upload's own gate a moment before a discard meets, and
        without it that write would mint the directory again behind the
        discard's back.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError(f"segment store {self._dir} is closed")
            if key in self._index:
                raise KeyError(key)
            keyed = batch_with_key(batch, key)
            if self._writer is None:
                self._start_segment(keyed.schema)
            segment_id = self._open_id
            writer, sink = self._writer, self._sink
            if writer is None or sink is None or segment_id is None:
                raise RuntimeError(f"segment store {self._dir} has no open segment")

            write_start = sink.tell()
            writer.write_batch(keyed)
            sink.flush()
            byte_offset, byte_length = bracket_message(
                self._segment_path(segment_id), write_start, sink.tell()
            )

            self._index[key] = _Placed(
                segment_id,
                IndexRecord(
                    key=key,
                    byte_offset=byte_offset,
                    byte_length=byte_length,
                    size_bytes=sum(col.nbytes for col in batch.columns),
                    offset=self._counts.get(segment_id, 0),
                ),
            )
            self._counts[segment_id] = self._counts.get(segment_id, 0) + 1
            self._open_bytes = sink.tell()
            if self._open_bytes >= self._max_segment_bytes:
                self._seal_open_segment()

    def _start_segment(self, schema: pa.Schema) -> None:
        """Open a new segment for appends. Caller holds the lock.

        The id only ever increases, so a segment file is always freshly
        allocated and nothing here truncates one in place: a client that mapped
        a segment is never handed different bytes at the same path.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        segment_id = self._next_segment_id
        self._next_segment_id += 1
        self._sink = pa.OSFile(str(self._segment_path(segment_id)), "wb")
        self._writer = pa.RecordBatchStreamWriter(self._sink, schema)
        self._open_id = segment_id
        self._open_bytes = 0

    def seal(self) -> None:
        """Close the open segment and write its sidecar; what READY does.

        Idempotent, and the point at which every byte of the member is servable
        by range: no segment is open on a sealed store.
        """
        with self._lock:
            self._seal_open_segment()

    def _seal_open_segment(self) -> None:
        """Close the writer, then index what it wrote. Caller holds the lock."""
        segment_id = self._open_id
        writer, sink = self._writer, self._sink
        self._writer = self._sink = self._open_id = None
        self._open_bytes = 0
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()  # both: close() leaves the OSFile open (issue #5)
        if segment_id is None:
            return
        seg_file = self._segment_path(segment_id)
        records = sorted(
            (p.record for p in self._index.values() if p.segment_id == segment_id),
            key=lambda r: r.offset,
        )
        write_sidecar(seg_file.with_suffix(".idx"), seg_file, records, "normal")

    # -- reading ---------------------------------------------------------------

    def stored_keys(self) -> List[bytes]:
        """Every key the store holds, in no particular order."""
        with self._lock:
            return list(self._index)

    def read(self, key: bytes) -> Optional[pa.RecordBatch]:
        """The batch stored under *key*, or None if the store does not hold it.

        What was uploaded, minus the key column and with no decode step.
        Zero-copy: the buffers point into the segment's mapping, which this
        store holds until :meth:`close`.
        """
        with self._lock:
            placed = self._index.get(key)
            if placed is None:
                return None
            mm = self._mapping(placed.segment_id)
            schema = self._schema(placed.segment_id, mm)
            if mm is None or schema is None:
                return None
            batch = self._batch(mm, schema, placed)
            if batch is None:
                return None
            return batch_without_key(batch)

    def _batch(self, mm, schema: pa.Schema, placed: _Placed):
        """The record at *placed*, by byte range if it has one, else by walk."""
        record = placed.record
        if record.byte_offset and record.byte_length:
            batch = batch_at_offset(mm, schema, record.byte_offset, record.key)
            if batch is not None:
                return batch
            logger.warning(
                f"segment store: {self._segment_path(placed.segment_id)} offset "
                f"{record.byte_offset} does not hold the entry indexed there; "
                f"walking the segment instead"
            )
        mm.seek(0)
        for i, batch in enumerate(pa.RecordBatchStreamReader(mm)):
            if i == record.offset:
                return batch
        return None

    def locate(self, key: bytes) -> Optional[ChunkLocation]:
        """Where *key*'s bytes sit on disk, for the localhost handoff (issue #9).

        A member answers this off its own sealed segment: the chunk is stored
        as the batch a reader wants, so there is nothing to resolve first and
        nothing to copy into the chunk cache. None when no byte range can be
        named for it, which sends the caller to do_get.
        """
        with self._lock:
            placed = self._index.get(key)
            if placed is None:
                return None
            record = placed.record
            # Offset 0 is never a real entry -- the schema message occupies the
            # start of a segment -- so 0 reads as "no range known".
            if not record.byte_offset or not record.byte_length:
                return None
            seg_path = self._segment_path(placed.segment_id)
            try:
                generation_id = os.stat(seg_path).st_ino
            except OSError:
                return None
            # A locate hands a byte range to another process, which reads it
            # with the server out of the loop, so the range is checked here
            # rather than trusted -- one that no longer names this entry would
            # hand that client someone else's pixels silently.
            mm = self._mapping(placed.segment_id)
            schema = self._schema(placed.segment_id, mm)
            if mm is None or schema is None:
                return None
            if batch_at_offset(mm, schema, record.byte_offset, key) is None:
                return None
            return ChunkLocation(
                segment_path=str(seg_path),
                byte_offset=record.byte_offset,
                byte_length=record.byte_length,
                generation_id=generation_id,
            )

    def _mapping(self, segment_id: int) -> Optional[pa.MemoryMappedFile]:
        """The segment's read mapping, opened on first use. Caller holds the lock."""
        mm = self._mmaps.get(segment_id)
        if mm is not None:
            return mm
        try:
            mm = pa.memory_map(str(self._segment_path(segment_id)), "r")
        except (OSError, pa.ArrowInvalid) as e:
            logger.warning(f"segment store: cannot map segment {segment_id}: {e}")
            return None
        self._mmaps[segment_id] = mm
        return mm

    def _schema(self, segment_id: int, mm) -> Optional[pa.Schema]:
        """The segment's IPC schema, read once. Caller holds the lock."""
        if mm is None:
            return None
        schema = self._schemas.get(segment_id)
        if schema is not None:
            return schema
        try:
            mm.seek(0)
            schema = pa.ipc.open_stream(mm).schema
        except (pa.ArrowInvalid, OSError) as e:
            logger.warning(f"segment store: cannot read segment {segment_id}: {e}")
            return None
        self._schemas[segment_id] = schema
        return schema

    # -- closing ---------------------------------------------------------------

    def close(self) -> None:
        """Release every handle: seal what is open, then drop the mappings.

        An open handle blocks unlink on Windows (issue #5), so this runs before
        the member's directory is removed and when its adapter is closed.
        """
        with self._lock:
            self._seal_open_segment()
            self._closed = True
            for mm in self._mmaps.values():
                try:
                    mm.close()
                except OSError:
                    pass
            self._mmaps.clear()
            self._schemas.clear()

    @property
    def closed(self) -> bool:
        """Whether this store has been closed and takes no further appends."""
        return self._closed

    def _segment_path(self, segment_id: int) -> Path:
        return self._dir / f"seg_{segment_id:04d}.arrow"
