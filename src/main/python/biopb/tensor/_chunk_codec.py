"""Chunk-id wire codec -- the single source of truth for the ``chunk_id`` byte
format, shared by the tensor server (which encodes chunk_ids into Flight
endpoints and decodes them on ``do_get``) and the client (which, in the
compact-grid read path, regenerates chunk_ids arithmetically instead of reading
them off explicit endpoints -- biopb/biopb#346).

``biopb-tensor-server`` depends on ``biopb``, so this lives in the core package
and there is exactly one definition of the format; the two sides can never
disagree (same pattern as ``biopb.tensor._wire_version``). ``struct`` +
``ChunkBounds`` -- no server-only dependencies, cheap to import on every path.
``compact_grid_arrays`` additionally uses ``numpy`` (already a hard dependency of
both sides) to build the whole grid without a per-chunk Python loop.

A ``chunk_id`` identifies a chunk by ``(array_id, bounds)`` and is a *pure
function* of them, so a regular grid's chunk_ids are fully derivable from the
grid without the server enumerating them. Format:

- 4 bytes: array_id length (uint32, big-endian)
- N bytes: array_id (UTF-8)
- 2 bytes: ndim (uint16, big-endian)
- 8*ndim bytes: bounds.start (int64, big-endian)
- 8*ndim bytes: bounds.stop (int64, big-endian)
- [scaled only] 8*ndim bytes: scale_hint (int64) + 2 bytes method length + method
"""

import itertools
import struct
from typing import List, NamedTuple, Tuple

import numpy as np

from biopb.tensor.ticket_pb2 import ChunkBounds


def encode_chunk_id(
    array_id: str,
    bounds: "ChunkBounds",
) -> bytes:
    """Encode array_id and bounds into chunk_id.

    Format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - 2 bytes: ndim (uint16, big-endian)
    - 8*ndim bytes: bounds.start (int64, big-endian)
    - 8*ndim bytes: bounds.stop (int64, big-endian)

    Args:
        array_id: Tensor identifier
        bounds: Chunk bounds (start, stop coordinates)

    Returns:
        Encoded chunk_id bytes
    """
    array_id_bytes = array_id.encode("utf-8")
    ndim = len(bounds.start)

    parts = [
        struct.pack(">I", len(array_id_bytes)),
        array_id_bytes,
        struct.pack(">H", ndim),
    ]

    for val in bounds.start:
        parts.append(struct.pack(">q", int(val)))
    for val in bounds.stop:
        parts.append(struct.pack(">q", int(val)))

    return b"".join(parts)


def rewrite_chunk_id_array_id(chunk_id: bytes, new_array_id: str) -> bytes:
    """Replace only the array_id field of a chunk_id, preserving everything else.

    The array_id is a self-describing length-prefixed field at the very front of
    the chunk_id (``[uint32 len][array_id utf-8]``); every byte after it -- ndim,
    the start/stop bounds, and any scale suffix on a scaled chunk_id -- is
    independent of the array_id string. So a remote-tensor *proxy* can map a
    chunk_id between its local (possibly alias-namespaced) array_id and the
    upstream's array_id with a pure byte splice, without understanding bounds or
    scale encoding ("understands nothing"). The splice round-trips for both
    regular and scaled chunk_ids because ``decode_chunk_id`` / ``is_scaled_chunk``
    / ``decode_scale_info`` all recompute their offsets from the (new) length
    prefix.

    Args:
        chunk_id: An encoded chunk_id (regular or scaled).
        new_array_id: The array_id to substitute in.

    Returns:
        A new chunk_id with the array_id field replaced and all trailing bytes
        (ndim/bounds/scale) preserved verbatim.
    """
    old_len = struct.unpack(">I", chunk_id[:4])[0]
    tail = chunk_id[4 + old_len :]
    new_bytes = new_array_id.encode("utf-8")
    return struct.pack(">I", len(new_bytes)) + new_bytes + tail


def decode_chunk_id(chunk_id: bytes) -> Tuple[str, "ChunkBounds"]:
    """Decode array_id and bounds from chunk_id. Works for both regular
    and virtual chunk_ids (ignores virtual payload).

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        Tuple of (array_id, bounds)
    """
    array_id_len = struct.unpack(">I", chunk_id[:4])[0]
    array_id = chunk_id[4 : 4 + array_id_len].decode("utf-8")

    offset = 4 + array_id_len
    ndim = struct.unpack(">H", chunk_id[offset : offset + 2])[0]
    offset += 2

    start = []
    for _ in range(ndim):
        start.append(struct.unpack(">q", chunk_id[offset : offset + 8])[0])
        offset += 8

    stop = []
    for _ in range(ndim):
        stop.append(struct.unpack(">q", chunk_id[offset : offset + 8])[0])
        offset += 8

    bounds = ChunkBounds(start=start, stop=stop)

    return array_id, bounds


def get_bounds_from_chunk_id(chunk_id: bytes) -> "ChunkBounds":
    """Extract bounds from chunk_id."""
    _, bounds = decode_chunk_id(chunk_id)
    return bounds


def encode_chunk_id_with_scale(
    array_id: str,
    bounds: ChunkBounds,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> bytes:
    """Encode chunk_id with bounds and scale info appended.

    Format:
    - Standard bounds encoding (array_id + ndim + start + stop)
    - 8*ndim bytes: scale_hint (int64, big-endian)
    - 2 bytes: method length (uint16)
    - N bytes: method string

    Detection: if len(chunk_id) > bounds_end, it's a scaled chunk.

    Args:
        array_id: Tensor identifier
        bounds: Chunk bounds (start, stop coordinates)
        scale_hint: Scale factors per axis
        reduction_method: Reduction method string

    Returns:
        Encoded chunk_id bytes with scale info appended
    """
    base = encode_chunk_id(array_id, bounds)

    method_bytes = reduction_method.encode("utf-8")

    scale_payload = b"".join(
        [
            b"".join(struct.pack(">q", s) for s in scale_hint),
            struct.pack(">H", len(method_bytes)),
            method_bytes,
        ]
    )

    return base + scale_payload


def _bounds_end(chunk_id: bytes) -> Tuple[int, int]:
    """``(ndim, bounds_end)`` for a chunk_id.

    ``bounds_end`` is where the standard encoding (array_id + ndim + start +
    stop) ends; any bytes past it are the scale payload of a scaled chunk_id
    (see :func:`encode_chunk_id_with_scale`).
    """
    array_id_len = struct.unpack(">I", chunk_id[:4])[0]
    offset = 4 + array_id_len
    ndim = struct.unpack(">H", chunk_id[offset : offset + 2])[0]
    return ndim, offset + 2 + ndim * 8 + ndim * 8


def is_scaled_chunk(chunk_id: bytes) -> bool:
    """Check if chunk_id has scale info appended after bounds.

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        True if chunk_id contains scale info
    """
    _, bounds_end = _bounds_end(chunk_id)
    return len(chunk_id) > bounds_end


def cache_key_for_chunk_id(chunk_id: bytes) -> bytes:
    """Canonical cache key for a chunk_id: the reduction method is advisory.

    For a scaled chunk_id, returns array_id + bounds + scale_hint with the
    trailing ``(uint16 method_len + method bytes)`` suffix dropped, so requests
    that differ only in reduction_method share one cache entry -- the method
    only decides how a true miss is computed (biopb/biopb#76). Non-scaled
    chunk_ids are returned unchanged.

    The result is an opaque cache key: it is NOT a valid chunk_id and must not
    be fed to :func:`decode_scale_info` or forwarded on the wire.
    """
    ndim, bounds_end = _bounds_end(chunk_id)
    if len(chunk_id) <= bounds_end:
        return chunk_id
    return chunk_id[: bounds_end + ndim * 8]


def decode_scale_info(chunk_id: bytes) -> Tuple[Tuple[int, ...], str]:
    """Decode scale_hint and reduction_method from scaled chunk_id.

    Args:
        chunk_id: Encoded chunk identifier with scale info

    Returns:
        Tuple of (scale_hint, reduction_method)
    """
    ndim, bounds_end = _bounds_end(chunk_id)

    # Decode scale_hint
    scale_hint = []
    for ax in range(ndim):
        scale_hint.append(
            struct.unpack(
                ">q", chunk_id[bounds_end + ax * 8 : bounds_end + ax * 8 + 8]
            )[0]
        )

    # Decode method
    method_offset = bounds_end + ndim * 8
    method_len = struct.unpack(">H", chunk_id[method_offset : method_offset + 2])[0]
    method = chunk_id[method_offset + 2 : method_offset + 2 + method_len].decode(
        "utf-8"
    )

    return tuple(scale_hint), method


def expand_compact_grid(descriptor) -> List[Tuple[bytes, ChunkBounds]]:
    """Regenerate the explicit ``(chunk_id, logical_bounds)`` endpoint list from a
    compact-grid GetFlightInfo descriptor (biopb/biopb#346).

    Inverse of the server's compact response (``serving/server.py`` get_flight_info):
    the client calls this when GetFlightInfo returned **no** endpoints but the
    descriptor carries a ``chunk_array_id``. It reproduces the exact list the
    server would otherwise have enumerated -- the same chunk_ids (byte-for-byte)
    and logical bounds, in the same ``np.ndindex`` (row-major) order -- so the
    downstream dask build is identical to the explicit-endpoint path.

    The descriptor must be the *response* descriptor: ``slice_hint`` carries the
    realized virtual-coordinate bounds [start, stop) (the server always sets it in
    compact mode), and ``chunk_array_id`` is the array_id the chunk_ids are encoded
    with (which is not ``array_id`` on a precompute plan). ``scale_hint`` present
    means the chunk_ids are scale-encoded and the logical grid is downsampled;
    absent means a plain read. This mirrors the server's ``_get_read_plan`` exactly.
    """
    ndim = len(descriptor.shape)
    logical_chunk = [int(c) for c in descriptor.chunk_shape]
    scaled = len(descriptor.scale_hint) > 0
    scale = [int(s) for s in descriptor.scale_hint] if scaled else [1] * ndim
    method = descriptor.reduction_method
    chunk_array_id = descriptor.chunk_array_id
    rstart = [int(s) for s in descriptor.slice_hint.start]
    rstop = [int(s) for s in descriptor.slice_hint.stop]

    # virtual chunk size = logical chunk size * scale. The server's
    # virtual_chunk_size = lcm(safe_chunk_size, scale) is a multiple of scale and
    # logical = virtual // scale, so this recovers it exactly.
    vcs = [logical_chunk[d] * scale[d] for d in range(ndim)]
    n = [_ceil_div(rstop[d] - rstart[d], vcs[d]) for d in range(ndim)]

    out: List[Tuple[bytes, ChunkBounds]] = []
    for idx in itertools.product(*(range(k) for k in n)):
        vstart = [rstart[d] + idx[d] * vcs[d] for d in range(ndim)]
        vstop = [min(vstart[d] + vcs[d], rstop[d]) for d in range(ndim)]
        vbounds = ChunkBounds(start=vstart, stop=vstop)
        if scaled:
            cid = encode_chunk_id_with_scale(
                chunk_array_id, vbounds, tuple(scale), method
            )
            lstart = [(vstart[d] - rstart[d]) // scale[d] for d in range(ndim)]
            lstop = [_ceil_div(vstop[d] - rstart[d], scale[d]) for d in range(ndim)]
        else:
            cid = encode_chunk_id(chunk_array_id, vbounds)
            lstart = [vstart[d] - rstart[d] for d in range(ndim)]
            lstop = [vstop[d] - rstart[d] for d in range(ndim)]
        out.append((cid, ChunkBounds(start=lstart, stop=lstop)))
    return out


class CompactGridArrays(NamedTuple):
    """Vectorized form of a compact-grid read plan (see :func:`compact_grid_arrays`).

    Everything is materialized with numpy in a handful of array operations rather
    than an ``n_chunks``-iteration Python loop, and the columnar arrays are what a
    numpy-backed ``BlockwiseDep`` indexes directly -- no intermediate ``ChunkBounds``
    objects, no ``chunk_map`` dict.
    """

    grid_chunks: Tuple[Tuple[int, ...], ...]  # dask ``chunks=`` (per-axis sizes)
    chunk_ids: np.ndarray  # object array of chunk_id bytes, C-order
    lstarts: np.ndarray  # (n_chunks, ndim) int64, logical (output) bounds start
    lstops: np.ndarray  # (n_chunks, ndim) int64, logical (output) bounds stop
    fingerprint: Tuple  # O(1) injective identity of the whole grid


def compact_grid_arrays(descriptor) -> CompactGridArrays:
    """Vectorized twin of :func:`expand_compact_grid`.

    Produces the *same* chunk_ids (byte-for-byte) and logical bounds, in the same
    C-order, but as columnar numpy arrays built without a per-chunk Python loop.
    The chunk_id encoding is separable: every id shares one ``prefix`` (array_id +
    ndim) and, for a scaled read, one constant ``suffix`` (scale + method); only
    the middle 16*ndim bytes -- the virtual start/stop coordinates -- vary per
    chunk, so all of them are packed at once with a single ``astype('>i8').tobytes``
    and sliced out. The logical bounds are separable per axis, so they are computed
    per axis and gathered by the meshgrid index. Mirrors the server's
    ``_get_read_plan`` exactly, same as ``expand_compact_grid``.

    ``fingerprint`` is an O(1) injective identity of the grid: the chunk_ids and the
    data mapping are a pure function of ``(chunk_array_id, shape, chunk_shape,
    scale_hint, realized bounds, reduction_method)``, so two arrays with equal
    fingerprints have identical chunk_ids -- suitable for the dask layer name
    without hashing the ``n_chunks`` ids.
    """
    ndim = len(descriptor.shape)
    aid = descriptor.chunk_array_id
    logical_chunk = [int(c) for c in descriptor.chunk_shape]
    scaled = len(descriptor.scale_hint) > 0
    scale = [int(s) for s in descriptor.scale_hint] if scaled else [1] * ndim
    method = descriptor.reduction_method
    rstart = [int(s) for s in descriptor.slice_hint.start]
    rstop = [int(s) for s in descriptor.slice_hint.stop]
    vcs = [logical_chunk[d] * scale[d] for d in range(ndim)]
    n = [_ceil_div(rstop[d] - rstart[d], vcs[d]) for d in range(ndim)]

    # Per-axis virtual and logical edges (separable). vstart/vstop are the virtual
    # (base-coordinate) chunk bounds encoded in the chunk_id; lstart/lstop are the
    # logical (output-array) bounds delivered to dask.
    axis_vstart, axis_vstop, axis_lstart, axis_lstop = [], [], [], []
    for d in range(ndim):
        vs = rstart[d] + np.arange(n[d], dtype=np.int64) * vcs[d]
        ve = np.minimum(vs + vcs[d], rstop[d])
        axis_vstart.append(vs)
        axis_vstop.append(ve)
        if scaled:
            axis_lstart.append((vs - rstart[d]) // scale[d])
            axis_lstop.append(-(-(ve - rstart[d]) // scale[d]))  # ceil_div
        else:
            axis_lstart.append(vs - rstart[d])
            axis_lstop.append(ve - rstart[d])

    grid_chunks = tuple(
        tuple((axis_lstop[d] - axis_lstart[d]).tolist()) for d in range(ndim)
    )

    # C-order meshgrid of block indices, then gather per-axis edges into
    # (n_chunks, ndim) columns.
    mesh = np.meshgrid(*[np.arange(k) for k in n], indexing="ij")
    idx = np.stack([m.ravel() for m in mesh], axis=1)  # (n_chunks, ndim)
    vstart = np.stack([axis_vstart[d][idx[:, d]] for d in range(ndim)], axis=1)
    vstop = np.stack([axis_vstop[d][idx[:, d]] for d in range(ndim)], axis=1)
    lstarts = np.stack([axis_lstart[d][idx[:, d]] for d in range(ndim)], axis=1)
    lstops = np.stack([axis_lstop[d][idx[:, d]] for d in range(ndim)], axis=1)

    # Vectorized chunk_id bytes: constant prefix (+ constant scale/method suffix),
    # variable 16*ndim-byte virtual-bounds core packed in one shot.
    aid_b = aid.encode("utf-8")
    prefix = struct.pack(">I", len(aid_b)) + aid_b + struct.pack(">H", ndim)
    suffix = b""
    if scaled:
        method_b = method.encode("utf-8")
        suffix = (
            b"".join(struct.pack(">q", s) for s in scale)
            + struct.pack(">H", len(method_b))
            + method_b
        )
    core = np.concatenate([vstart, vstop], axis=1).astype(">i8").tobytes()
    w = 16 * ndim
    n_chunks = idx.shape[0]
    chunk_ids = np.empty(n_chunks, dtype=object)
    chunk_ids[:] = [
        prefix + core[i * w : (i + 1) * w] + suffix for i in range(n_chunks)
    ]

    fingerprint = (
        aid,
        tuple(int(s) for s in descriptor.shape),
        tuple(logical_chunk),
        tuple(scale) if scaled else None,
        tuple(rstart),
        tuple(rstop),
        method if scaled else None,
    )
    return CompactGridArrays(grid_chunks, chunk_ids, lstarts, lstops, fingerprint)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)
