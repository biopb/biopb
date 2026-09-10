"""The chunk payload's wire form: numpy array <-> Arrow RecordBatch.

One schema serves the do_get wire, the file cache and the memory cache, so a
chunk is packed once and read back by either. It lives here rather than in
``adapter_base`` so :mod:`~.cache_source` can unpack a cached entry without
importing the adapter layer that calls it.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa

# --- unified chunk wire schema (biopb/biopb#293) ----------------------------
# Every chunk crosses the wire as an opaque binary blob plus its numpy dtype
# string, NOT a typed Arrow ``list<T>``. Arrow arrays are always native-endian,
# so a typed encoding cannot carry a big-endian source (FITS is ``>i2``) --
# ``pa.array()`` raises "Byte-swapped arrays not supported". Serializing the raw
# bytes and reconstructing them client-side with ``np.frombuffer(buf, dtype_str)``
# preserves the exact dtype (endianness included) and is zero-copy on both ends:
# the server wraps the numpy buffer with no per-element copy, and the cache
# stores/serves this same schema with no typed<->binary conversion. It is the ONE
# schema for the do_get wire, the file cache, and the memory cache. (The file
# cache appends a per-batch cache-key column; the schema is otherwise identical.)
# (Cross-language clients decode the binary column per the dtype string;
# see the Java client's ``SerializableTensorImg``.)
CHUNK_WIRE_SCHEMA = pa.schema(
    [
        pa.field("data", pa.binary()),
        pa.field("shape", pa.list_(pa.int64())),
        pa.field("dtype", pa.string()),
    ]
)


def pack_chunk_batch(arr: np.ndarray) -> pa.RecordBatch:
    """Pack a chunk's numpy array into the unified binary RecordBatch (one row).

    Zero-copy: the array's bytes are wrapped as an Arrow buffer without a copy
    (made C-contiguous first, which copies only a non-contiguous view). The dtype
    string carries endianness, so byte-swapped sources round-trip losslessly.
    See ``CHUNK_WIRE_SCHEMA`` / biopb/biopb#293.
    """
    arr = np.ascontiguousarray(arr)
    data_buf = pa.py_buffer(arr)  # keeps ``arr`` alive; no element copy
    offsets = pa.py_buffer(np.array([0, data_buf.size], dtype=np.int32))
    data_col = pa.Array.from_buffers(pa.binary(), 1, [None, offsets, data_buf])
    return pa.RecordBatch.from_arrays(
        [data_col, pa.array([list(arr.shape)]), pa.array([arr.dtype.str])],
        schema=CHUNK_WIRE_SCHEMA,
    )


def unpack_chunk_view(batch: pa.RecordBatch) -> np.ndarray:
    """A view of a chunk batch's array -- :func:`unpack_chunk_array`, uncopied.

    Valid only while *batch* is alive, and for a cached entry only while that
    entry is acquired, so this is for a caller that consumes it under that
    reference: :func:`~.cache_source.assemble_from_cache` writes it straight into
    its own unit buffer, where the copy below would be a second one, and
    :func:`~.cache_source.borrow_cached_unit` reduces out of it without copying
    at all. Anything that outlives the entry wants :func:`unpack_chunk_array`.
    """
    dtype = np.dtype(batch.column("dtype")[0].as_py())
    shape = tuple(batch.column("shape").to_pylist()[0])
    count = math.prod(shape) if shape else 0
    data_buf = batch.column("data").buffers()[2]
    return np.frombuffer(data_buf, dtype=dtype, count=count).reshape(shape)


def unpack_chunk_array(batch: pa.RecordBatch) -> np.ndarray:
    """Reconstruct a chunk's numpy array from a unified binary RecordBatch.

    Inverse of :func:`pack_chunk_batch`: reads the raw bytes and reinterprets them
    with the per-chunk dtype string, so numpy applies the correct (possibly
    non-native) byte order. Returns an owned copy so it stays valid after any
    backing mmap is released. (The Python Flight client has its own copy of this
    logic, ``_array_from_unified_batch``, since the core ``biopb`` package cannot
    import server code.)
    """
    return unpack_chunk_view(batch).copy()
