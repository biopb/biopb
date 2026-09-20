"""How a segment's index is derived: from its body, or from its ``.idx`` sidecar.

Two ways to learn which entries a segment holds and where their bytes are:

- ``scan_segment_records`` walks the segment body's IPC messages. Authoritative,
  and the only thing that works on a segment written by an older build.
- ``read_sidecar`` reads the ``seg_NNNN.idx`` file written when the segment was
  sealed. The same records, without faulting the body -- the point of
  biopb/biopb#300.

Both return the same ``IndexRecord`` shape, so the boot path cannot build a
different index depending on which one it took. Pure functions over paths: they
hold no backend state and take no lock, which is what lets them live outside
``file_backend.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, get_args

import pyarrow as pa

from biopb_tensor_server.cache.types import RetentionClass

__all__ = [
    "CACHE_KEY_FIELD",
    "SIDECAR_FORMAT_VERSION",
    "IndexRecord",
    "SegmentScan",
    "read_sidecar",
    "scan_segment_records",
    "write_sidecar",
]

logger = logging.getLogger(__name__)


# Name of the per-batch column carrying the entry's cache key. The key MUST
# travel as a column value, not as schema metadata: an Arrow IPC stream
# serializes the schema exactly once (taken from the first batch written to the
# segment), so per-batch schema metadata is lost on read-back and every batch
# would report the first entry's key. A column value is stored per row and
# round-trips correctly. Read back by ``scan_segment_records`` and by
# ``file_backend`` to confirm a byte range holds the entry it is indexed under.
CACHE_KEY_FIELD = "__biopb_cache_key__"

# Per-segment sidecar index (biopb/biopb#300). Each sealed segment gets a
# ``seg_NNNN.idx`` sidecar written at seal time, recording every entry's key ->
# byte range, so boot restores the index from these small files instead of
# faulting the whole on-disk cache. The retention class rides in the sidecar's
# schema metadata.
#
# A sealed segment is immutable, so the sidecar needs no manifest or generation
# counter: a boot trusts a sidecar iff its recorded ``.arrow`` size matches the
# file on disk. The tiny ``.idx`` bytes are deliberately NOT counted toward
# ``max_total_bytes``.
#
# The key travels as a real column (not schema-only): a sidecar is an Arrow
# IPC FILE, so this is belt-and-suspenders, but it keeps the record self-describing.
SIDECAR_FORMAT_VERSION = 1
_SIDECAR_VERSION_KEY = b"biopb_sidecar_version"
_SIDECAR_SEGMENT_SIZE_KEY = b"biopb_segment_size"
_SIDECAR_RETENTION_KEY = b"biopb_retention"
_SIDECAR_VERSION_BYTES = str(SIDECAR_FORMAT_VERSION).encode()
_SIDECAR_SCHEMA = pa.schema(
    [
        pa.field("key", pa.binary()),
        pa.field("byte_offset", pa.int64()),
        pa.field("byte_length", pa.int64()),
        pa.field("size_bytes", pa.int64()),
        pa.field("offset", pa.int64()),
    ]
)

_KNOWN_RETENTIONS = frozenset(get_args(RetentionClass))

# Arrow IPC end-of-stream: the 0xFFFFFFFF continuation followed by a zero
# metadata length. ``writer.close()`` emits it, so its presence is how a
# cleanly sealed segment is told apart from one whose writer died mid-message.
_IPC_END_OF_STREAM = b"\xff\xff\xff\xff\x00\x00\x00\x00"


class IndexRecord(NamedTuple):
    """One entry's place in a segment, as every index producer reports it.

    ``size_bytes`` is the batch's buffer size, not the caller's declared size:
    a body walk can only measure buffers, so this is the one definition the
    write path, the sidecar and the walk can all agree on. ``offset`` is the
    entry's index within the segment, used by the sequential-read fallback when
    the byte range is unusable.
    """

    key: bytes
    byte_offset: int
    byte_length: int
    size_bytes: int
    offset: int


class SegmentScan(NamedTuple):
    """What a body walk found: the index records, and whether the tail was torn.

    ``torn`` means a trailing message was present but could not be decoded --
    the on-disk signature of a write interrupted mid-``write_batch``. Since a
    segment is append-only and only the unsealed one is ever being written, that
    is exactly one lost entry, and it is what ``RecoveryStatus.lost_entries``
    reports. A clean tail (the reader simply runs out of messages) is not torn.
    """

    records: List[IndexRecord]
    torn: bool


def scan_segment_records(seg_file: Path) -> Optional[SegmentScan]:
    """Walk a sealed segment's IPC stream, returning one index record per entry.

    The authoritative mapping from a segment body to index entries, reused by
    the boot fallback walk and the seal-time sidecar fallback so every index
    path agrees byte-for-byte. Reads message-by-message off a private mmap to
    bracket each record batch (issue #9 needs the byte range) and stops at the
    first unreadable/torn trailing message -- a prior partial write's slack,
    reported as ``torn`` so the boot can account for the entry it cost.
    Returns None for a legacy/corrupt segment (no per-batch key column, or
    unreadable), signalling the caller to drop or skip it. The mmap is always
    closed (an open handle blocks unlink on Windows, issue #5).
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
        torn = False
        while True:
            pos = mm.tell()
            try:
                msg = pa.ipc.read_message(mm)
            except (pa.ArrowInvalid, EOFError, StopIteration, OSError):
                torn = _tail_is_torn(mm, pos)
                break
            if msg is None:
                torn = _tail_is_torn(mm, pos)
                break
            msg_len = mm.tell() - pos
            try:
                batch = pa.ipc.read_record_batch(msg, schema)
            except Exception:
                torn = True
                break
            key = batch.column(CACHE_KEY_FIELD)[0].as_py()
            if key is None:
                continue
            size_bytes = sum(
                batch.column(name).nbytes for name in ("data", "shape", "dtype")
            )
            records.append(IndexRecord(key, pos, msg_len, size_bytes, entry_index))
            entry_index += 1
        return SegmentScan(records, torn)
    finally:
        mm.close()


def _tail_is_torn(mm, pos: int) -> bool:
    """Is what follows the last readable message a partial write?

    Two endings are clean: nothing at all (the writer died between messages, so
    the stream simply stops) and the end-of-stream marker ``writer.close()``
    emits. Anything else is the head of a message whose body never landed --
    one entry lost, which is what recovery reports.
    """
    remaining = mm.size() - pos
    if remaining <= 0:
        return False
    if remaining == len(_IPC_END_OF_STREAM):
        mm.seek(pos)
        return mm.read(remaining) != _IPC_END_OF_STREAM
    return True


def read_sidecar(
    idx_path: Path, seg_file: Path
) -> Optional[Tuple[List[IndexRecord], RetentionClass]]:
    """Read a segment's index out of its ``.idx`` sidecar, without its body.

    Returns ``(records, retention)`` on the fast path, or None when the caller
    must fall back to ``scan_segment_records``: the sidecar is absent, the wrong
    version, or stale (its recorded ``.arrow`` size != the file on disk). The
    sidecar is read fully into RAM and its handle closed before the caller
    touches the segment, so it never blocks unlink (issue #5).
    """
    if not idx_path.exists():
        return None
    try:
        with pa.OSFile(str(idx_path), "rb") as f:
            reader = pa.ipc.open_file(f)
            meta = reader.schema.metadata or {}
            if meta.get(_SIDECAR_VERSION_KEY) != _SIDECAR_VERSION_BYTES:
                return None
            recorded = meta.get(_SIDECAR_SEGMENT_SIZE_KEY)
            if recorded is None or int(recorded) != seg_file.stat().st_size:
                return None
            table = reader.read_all()
    except (OSError, ValueError, pa.ArrowInvalid):
        return None  # unreadable / mismatched sidecar -> walk the body

    records = [
        IndexRecord(*row)
        for row in zip(
            table.column("key").to_pylist(),
            table.column("byte_offset").to_pylist(),
            table.column("byte_length").to_pylist(),
            table.column("size_bytes").to_pylist(),
            table.column("offset").to_pylist(),
            strict=True,  # columns of one table -> equal length
        )
    ]
    if not records:
        return None  # nothing to install; let the body walk decide

    raw = meta.get(_SIDECAR_RETENTION_KEY)
    retention = raw.decode() if raw else "normal"
    if retention not in _KNOWN_RETENTIONS:
        retention = "normal"
    return records, retention


def write_sidecar(
    idx_path: Path,
    seg_file: Path,
    records: List[IndexRecord],
    retention: RetentionClass,
) -> None:
    """Persist a sealed segment's key -> byte-range index, read at boot instead
    of walking the body (biopb/biopb#300).

    Written atomically (tmp + ``os.replace``) and best-effort: a failure (e.g.
    ENOSPC) only forfeits the fast path for this one segment next boot, so it is
    logged and swallowed rather than allowed to fail the write/close path.
    """
    if not records:
        # Empty sealed segment: no sidecar. Boot walks/no-ops it -- harmless.
        return
    try:
        st = seg_file.stat()
    except OSError:
        return

    schema = _SIDECAR_SCHEMA.with_metadata(
        {
            _SIDECAR_VERSION_KEY: _SIDECAR_VERSION_BYTES,
            _SIDECAR_SEGMENT_SIZE_KEY: str(st.st_size).encode(),
            _SIDECAR_RETENTION_KEY: retention.encode(),
        }
    )
    table = pa.Table.from_arrays(
        [
            pa.array([r.key for r in records], type=pa.binary()),
            pa.array([r.byte_offset for r in records], type=pa.int64()),
            pa.array([r.byte_length for r in records], type=pa.int64()),
            pa.array([r.size_bytes for r in records], type=pa.int64()),
            pa.array([r.offset for r in records], type=pa.int64()),
        ],
        schema=schema,
    )

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
