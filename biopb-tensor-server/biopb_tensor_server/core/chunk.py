"""Utilities for encoding and decoding chunk identifiers (chunk_id) used in Flight endpoints.

This module contains:
- ChunkEndpoint dataclass for chunk metadata
- Chunk ID encoding/decoding functions
- Chunk operations (intersection)
- Read plan helper functions
"""

import logging
import os
import struct
from dataclasses import dataclass
from math import lcm
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import SliceHint
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.axes import samples_axis
from biopb_tensor_server.core.downsample import (
    DEFAULT_REDUCTION_METHOD,
    normalize_reduction_method,
)

logger = logging.getLogger(__name__)


# =============================================================================
# chunk_id byte codec -- a STRICTLY SERVER-SIDE concern.
#
# A chunk_id identifies a chunk by (array_id, bounds) and is a pure function of
# them. The server mints chunk_ids into Flight endpoint tickets and decodes them
# on do_get; clients treat a chunk_id as an OPAQUE token (they echo the ticket
# back and read a chunk's bounds from the endpoint's app_metadata), so this
# format is never regenerated off-server. Keeping the codec here -- not in the
# shared `biopb` core -- lets the server evolve it without a lockstep client/Java
# upgrade (the compact-grid read plan that shared it, biopb/biopb#346, was
# reverted for exactly that coupling cost).
#
# Format:
# - 4 bytes: array_id length (uint32, big-endian)
# - N bytes: array_id (UTF-8)
# - 2 bytes: ndim (uint16, big-endian)
# - 8*ndim bytes: bounds.start (int64, big-endian)
# - 8*ndim bytes: bounds.stop (int64, big-endian)
# - [scaled only] 8*ndim bytes: scale_hint (int64)
# - [scaled + non-default method only] 1 byte: reduction_method code
#
# The chunk_id is IDENTITY (array_id + bounds + scale_hint [+ method]). #178 had
# dropped reduction_method from the wire -- it was advisory and the compute path
# hard-coded the default, which silently served a client's requested method with
# the wrong one (biopb/biopb#578). It is back, but compact and default-free: the
# computed downsample space is binary ("nearest" | "area", area = the default),
# so a non-default method appends ONE code byte and "area"/default appends
# nothing. So an area (default) scaled chunk_id -- and its cache key -- stays
# byte-identical to the pre-#178 form (its cache entries survive), and only a
# genuinely-distinct "nearest" read gets a longer id and its own entry. A
# method-free scaled chunk_id (old server / old cache) decodes to the default,
# exactly as before. This reverses the #76 cache-sharing (nearest and area no
# longer collide) -- the deliberate cost of serving the method the client asked
# for.
# A cold downsample uses the server default; see core.adapter_base.resolve_chunk_data.
# (An older chunk_id that still carries a method suffix stays readable: decode /
# is_scaled / cache_key all ignore the trailing bytes, so no cache wipe is needed.)
#
# content_version wrapper (biopb/biopb#178)
# -----------------------------------------
# An OPTIONAL content-version header may be prepended, folding a source's
# content_version into the chunk_id (and hence the cache key) so a re-registered
# source with new bytes can't be masked by a stale cached chunk. A legacy
# chunk_id always begins with ``struct.pack(">I", array_id_len)`` whose high byte
# is 0x00 (array_id is far under 16 MB), so a leading 0xFF sentinel is an
# unambiguous, backward-compatible discriminator: an UNVERSIONED chunk_id is byte
# -identical to the pre-#178 format (existing cache entries stay valid), and the
# version, when present, is a constant header the read-plan mint precomputes once
# and prepends to every chunk_id (so the per-chunk cost is one concat, not a
# re-encode). The whole codec strips this header first, so decode / scale / cache_key
# operate on the inner legacy chunk_id and only cache_key_for_chunk_id keeps the
# version (that is the point -- a different version -> a different key -> the old
# entry is un-lookupable, not mis-served). Clients treat the whole thing as opaque.
# =============================================================================

_CV_SENTINEL = 0xFF  # leading byte marking a version-wrapped chunk_id
_CV_FORMAT = 1  # wrapper layout version (after the sentinel byte)


def _version_header(content_version: bytes) -> bytes:
    """The constant prefix that wraps a chunk_id with a content_version.

    ``[0xFF sentinel][uint8 fmt][uint32 cv_len][cv bytes]``. Precompute once per
    read plan (content_version is constant across a source's chunks) and prepend.
    """
    return (
        struct.pack(">BBI", _CV_SENTINEL, _CV_FORMAT, len(content_version))
        + content_version
    )


def wrap_content_version(inner_chunk_id: bytes, content_version: bytes) -> bytes:
    """Prepend a content_version header to a legacy (inner) chunk_id."""
    return _version_header(content_version) + inner_chunk_id


def _split_version(chunk_id: bytes) -> Tuple[Optional[bytes], bytes]:
    """Split a chunk_id into ``(content_version | None, inner_legacy_chunk_id)``.

    Unversioned chunk_ids (no 0xFF sentinel) pass through unchanged, so every
    codec function below can strip first and reuse the pre-#178 logic verbatim.
    """
    if not chunk_id or chunk_id[0] != _CV_SENTINEL:
        return None, chunk_id
    cv_len = struct.unpack(">I", chunk_id[2:6])[0]
    inner_offset = 6 + cv_len
    return chunk_id[6:inner_offset], chunk_id[inner_offset:]


def content_version_of(chunk_id: bytes) -> Optional[bytes]:
    """The chunk_id's content_version, or None if it carries no version header."""
    return _split_version(chunk_id)[0]


# =============================================================================
# Proxy envelope (biopb/biopb#178 W1)
# -----------------------------------------------------------------------------
# A remote-tensor proxy wraps the UPSTREAM's chunk_id in an envelope instead of
# decoding/rewriting it (the old opacity violation). The inner upstream chunk_id
# is carried VERBATIM -- the proxy never parses it -- alongside a proxy-owned
# ``route`` (the local array_id, used to dispatch to the proxy adapter without
# decoding the inner) and the upstream's ``content_version`` (may be empty).
#
# Layout: ``[0xFE sentinel][uint8 fmt][uint32 route_len][route][uint32 cv_len][cv]
#          [inner: opaque upstream chunk_id]``
#
# 0xFE is a third discriminator, mutually exclusive with the 0x00 legacy high byte
# and the 0xFF content_version sentinel, so any codec entry point can tell the
# three apart from byte 0. The envelope frames (route, cv, inner) with lengths, so
# it is an injective cache key regardless of what the inner carries -- and since
# the inner now carries the reduction_method byte when it is a non-default scaled
# read (biopb/biopb#578), the envelope key distinguishes methods too, for free,
# without ever parsing the opaque inner -- see cache_key_for_chunk_id.
# =============================================================================

_ENV_SENTINEL = 0xFE  # leading byte marking a proxy-envelope chunk_id
_ENV_FORMAT = 1  # envelope layout version (after the sentinel byte)


def is_proxy_envelope(chunk_id: bytes) -> bool:
    """True if ``chunk_id`` is a proxy envelope (leading 0xFE sentinel)."""
    return bool(chunk_id) and chunk_id[0] == _ENV_SENTINEL


def encode_proxy_envelope(
    inner_chunk_id: bytes, route: str, content_version: Optional[bytes]
) -> bytes:
    """Wrap an opaque upstream ``inner_chunk_id`` in a proxy envelope.

    ``route`` is the proxy's LOCAL array_id (how the server dispatches the chunk
    back to this adapter); ``content_version`` is the upstream source's version
    (``None``/empty when the upstream is unversioned). The inner is stored and
    later forwarded byte-for-byte -- the proxy never interprets it.
    """
    route_bytes = route.encode("utf-8")
    cv = content_version or b""
    return (
        struct.pack(">BBI", _ENV_SENTINEL, _ENV_FORMAT, len(route_bytes))
        + route_bytes
        + struct.pack(">I", len(cv))
        + cv
        + inner_chunk_id
    )


def peel_proxy_envelope(chunk_id: bytes) -> Tuple[str, Optional[bytes], bytes]:
    """Split a proxy envelope into ``(route, content_version | None, inner)``.

    Inverse of :func:`encode_proxy_envelope`. A zero-length content_version field
    decodes back to ``None``. ``inner`` is the verbatim upstream chunk_id.
    """
    route_len = struct.unpack(">I", chunk_id[2:6])[0]
    offset = 6 + route_len
    route = chunk_id[6:offset].decode("utf-8")
    cv_len = struct.unpack(">I", chunk_id[offset : offset + 4])[0]
    offset += 4
    cv = chunk_id[offset : offset + cv_len]
    offset += cv_len
    inner = chunk_id[offset:]
    return route, (cv if cv_len > 0 else None), inner


def routing_array_id(chunk_id: bytes) -> str:
    """The local array_id used to dispatch ``chunk_id`` to its adapter.

    For a proxy envelope the ``route`` token IS the local array_id (the inner is
    opaque and never decoded); otherwise decode it from the (possibly
    version-wrapped) chunk_id. This is the one entry point the server routing uses
    so an envelope never reaches :func:`decode_chunk_id`, which would misparse it.
    """
    if is_proxy_envelope(chunk_id):
        return peel_proxy_envelope(chunk_id)[0]
    return decode_chunk_id(chunk_id)[0]


def content_version_from_path(path: object) -> Optional[bytes]:
    """Best-effort content_version for a local file/dir source (biopb/biopb#178).

    The stat signature ``mtime_ns:size`` -- O(1), no read, already the cheap
    change signal ``build_entry_signature`` uses. For a directory source this is
    the directory's own mtime, which flips on member add/remove/rename (the right
    O(1) signal for multi-file sources). Returns None when the path can't be
    stat'd (e.g. a remote URL / cloud store), leaving the source unversioned.

    Blind spots (documented, best-effort per #178):
    - an in-place edit that preserves mtime+size is undetectable;
    - two changes closer together than the filesystem's mtime resolution coalesce
      into one signal (observed ~sub-20ms on Windows dir mtimes).
    Since content_version is sampled once at (re-)registration -- events that are
    seconds apart -- neither blind spot bites the cache-invalidation use case.
    A source needing byte-exact freshness wants an explicit ``volatile`` /
    content-hash mode, not this signal.
    """
    try:
        st = os.stat(path)
    except (OSError, ValueError, TypeError):
        return None
    return f"{st.st_mtime_ns}:{st.st_size}".encode()


def encode_chunk_id(
    array_id: str,
    bounds: ChunkBounds,
) -> bytes:
    """Encode array_id and bounds into chunk_id."""
    array_id_bytes = array_id.encode("utf-8")
    ndim = len(bounds.start)

    return b"".join(
        [
            struct.pack(">I", len(array_id_bytes)),
            array_id_bytes,
            struct.pack(">H", ndim),
            struct.pack(f">{ndim}q", *map(int, bounds.start)),
            struct.pack(f">{ndim}q", *map(int, bounds.stop)),
        ]
    )


def decode_chunk_id(chunk_id: bytes) -> Tuple[str, ChunkBounds]:
    """Decode array_id and bounds from chunk_id. Works for both regular
    and virtual chunk_ids (ignores virtual payload) and version-wrapped ones."""
    _, chunk_id = _split_version(chunk_id)
    array_id_len = struct.unpack(">I", chunk_id[:4])[0]
    array_id = chunk_id[4 : 4 + array_id_len].decode("utf-8")

    offset = 4 + array_id_len
    ndim = struct.unpack(">H", chunk_id[offset : offset + 2])[0]
    offset += 2

    start = struct.unpack_from(f">{ndim}q", chunk_id, offset)
    offset += ndim * 8
    stop = struct.unpack_from(f">{ndim}q", chunk_id, offset)

    bounds = ChunkBounds(start=start, stop=stop)

    return array_id, bounds


def get_bounds_from_chunk_id(chunk_id: bytes) -> ChunkBounds:
    """Extract bounds from chunk_id."""
    _, bounds = decode_chunk_id(chunk_id)
    return bounds


# Compact reduction_method suffix on a scaled chunk_id (biopb/biopb#578). Only a
# NON-default method is carried, as a single code byte, so an "area"/default
# scaled chunk_id stays byte-identical to the method-free #178 form. The computed
# downsample space is binary ("nearest" | "area"), so one code covers it; the
# reverse map decodes it, and an absent byte means the default.
_SCALED_METHOD_BYTE = {"nearest": b"\x01"}
_SCALED_METHOD_BY_BYTE = {1: "nearest"}


def encode_chunk_id_with_scale(
    array_id: str,
    bounds: ChunkBounds,
    scale_hint: Tuple[int, ...],
    reduction_method: str = DEFAULT_REDUCTION_METHOD,
) -> bytes:
    """Encode a scaled chunk_id: bounds encoding + scale_hint [+ method byte].

    Format: standard bounds encoding, then 8*ndim bytes scale_hint (int64), then
    -- only for a NON-default reduction_method -- one method-code byte. The default
    ("area") appends nothing, so an area scaled chunk_id is byte-identical to the
    pre-#178 identity form (biopb/biopb#578, #178, #76). The method is normalized
    (stride->nearest, mean->area), so in practice only "nearest" adds a byte.
    Detection stays ``len(chunk_id) > bounds_end`` (a scaled chunk always carries
    at least the scale_hint); :func:`decode_reduction_method` reads the byte back.
    """
    base = encode_chunk_id(array_id, bounds)
    scale_payload = struct.pack(f">{len(scale_hint)}q", *scale_hint)
    method_suffix = _SCALED_METHOD_BYTE.get(
        normalize_reduction_method(reduction_method), b""
    )
    return base + scale_payload + method_suffix


def _bounds_end(chunk_id: bytes) -> Tuple[int, int]:
    """``(ndim, bounds_end)`` for an INNER (legacy, version-stripped) chunk_id.

    ``bounds_end`` is where the standard encoding (array_id + ndim + start +
    stop) ends; any bytes past it are the scale payload of a scaled chunk_id
    (see :func:`encode_chunk_id_with_scale`). Callers must pass a version-stripped
    chunk_id (offsets and the length comparison are relative to the inner bytes).
    """
    array_id_len = struct.unpack(">I", chunk_id[:4])[0]
    offset = 4 + array_id_len
    ndim = struct.unpack(">H", chunk_id[offset : offset + 2])[0]
    return ndim, offset + 2 + ndim * 8 + ndim * 8


def is_scaled_chunk(chunk_id: bytes) -> bool:
    """Check if chunk_id has scale info appended after bounds."""
    _, inner = _split_version(chunk_id)
    _, bounds_end = _bounds_end(inner)
    return len(inner) > bounds_end


def cache_key_for_chunk_id(chunk_id: bytes) -> bytes:
    """Canonical cache key for a chunk_id.

    A current chunk_id is identity (array_id + bounds [+ scale_hint [+ method
    byte]]), so the key equals the inner bytes -- INCLUDING the compact one-byte
    reduction_method suffix, so a "nearest" read keys distinctly from "area"
    (biopb/biopb#578). Only a LEGACY trailing method suffix (the pre-#178
    ``uint16 len + bytes`` form, which is more than one byte past the scale) is
    stripped, so a cache entry warmed under that old format still maps to today's
    area identity (biopb/biopb#76). Non-scaled chunk_ids are returned unchanged.

    Because an "area"/default scaled chunk_id carries no method byte, its key is
    byte-identical to the pre-#578 key -- so area entries are NOT invalidated;
    only genuinely-distinct "nearest" reads get a new key.

    The result is an opaque cache key: it is NOT a valid chunk_id and must not
    be fed to :func:`decode_scale_info` or forwarded on the wire.

    A content_version (biopb/biopb#178) is kept in the key -- so a version bump
    yields a distinct key and the stale entry becomes un-lookupable -- while the
    inner projection stays byte-identical to the pre-#178 key for an area read, so
    an UNVERSIONED area chunk_id maps to exactly its old cache entry.

    A proxy envelope is returned as-is: it already frames (route, content_version,
    inner) with lengths, so it is an injective key, and since the inner now carries
    the method byte for a non-default scaled read, the envelope key distinguishes
    methods too -- WITHOUT the proxy ever parsing the opaque inner.
    """
    if is_proxy_envelope(chunk_id):
        return chunk_id
    cv, inner = _split_version(chunk_id)
    ndim, bounds_end = _bounds_end(inner)
    scale_end = bounds_end + ndim * 8
    # Keep array_id+bounds+scale_hint and at most the one-byte method suffix; a
    # longer trailing run is the legacy uint16 method form, stripped for #76.
    base = inner if len(inner) <= scale_end + 1 else inner[:scale_end]
    return wrap_content_version(base, cv) if cv is not None else base


def decode_scale_info(chunk_id: bytes) -> Tuple[int, ...]:
    """Decode the scale_hint from a scaled chunk_id.

    Reads only the ndim int64 scale_hint after the bounds encoding. The
    reduction_method (a trailing byte, biopb/biopb#578) is read separately by
    :func:`decode_reduction_method`; any trailing bytes here are ignored, so a
    legacy method-carrying chunk_id still decodes its scale correctly.
    """
    _, chunk_id = _split_version(chunk_id)
    ndim, bounds_end = _bounds_end(chunk_id)

    return struct.unpack_from(f">{ndim}q", chunk_id, bounds_end)


def decode_reduction_method(chunk_id: bytes) -> str:
    """Decode the reduction_method carried by a scaled chunk_id (biopb/biopb#578).

    Only the compact one-byte code minted by :func:`encode_chunk_id_with_scale`
    (exactly one byte past the scale_hint) is honored. A non-scaled chunk_id, a
    method-free scaled chunk_id (old server / pre-#178 cache), or a legacy
    ``uint16 len + bytes`` method suffix all decode to the default -- so an old
    scaled read is served exactly as before (``area``), never rejected.
    """
    _, inner = _split_version(chunk_id)
    ndim, bounds_end = _bounds_end(inner)
    scale_end = bounds_end + ndim * 8
    if len(inner) == scale_end + 1:
        return _SCALED_METHOD_BY_BYTE.get(inner[scale_end], DEFAULT_REDUCTION_METHOD)
    return DEFAULT_REDUCTION_METHOD


# Constants
# Preferred transfer size and hard Arrow batch ceiling (biopb/biopb#684).
# MAX_ARROW_BATCH_BYTES is a wire fact the server enforces on every grid;
# PREFERRED_ARROW_BATCH_BYTES is the default sizing target, applied only where an
# adapter asks for it via default_transfer_chunk_shape and never over a grid the
# adapter declared itself.
#
# There is deliberately no minimum-endpoint floor. Chunk size is the only knob:
# a tensor that fits in one preferred-size chunk *is* one chunk, and splitting a
# 512x512 snapshot into four to manufacture parallelism costs round trips to
# parallelize work that was never the bottleneck.
PREFERRED_ARROW_BATCH_BYTES = 8 * 1024 * 1024
MAX_ARROW_BATCH_BYTES = 64 * 1024 * 1024


@dataclass(slots=True)
class ChunkEndpoint:
    """A chunk with its metadata for Flight endpoint creation.

    Attributes:
        chunk_id: Backend-specific chunk identifier (bytes)
        bounds: Array coordinates (start, stop) for this chunk
    """

    chunk_id: bytes
    bounds: ChunkBounds


# =============================================================================
# Slice and Scale Normalization
# =============================================================================


def normalized_slice_bounds(
    shape: Tuple[int, ...],
    slice_hint: Optional[SliceHint],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Normalize slice bounds from slice_hint.

    Args:
        shape: Tensor shape
        slice_hint: Optional slice hint from request

    Returns:
        Tuple of (start, stop) coordinates

    Raises:
        ValueError: If slice hint dimensionality mismatch or invalid bounds
    """
    if slice_hint is None:
        return tuple(0 for _ in shape), tuple(int(dim) for dim in shape)

    start = tuple(int(value) for value in slice_hint.start)
    stop = tuple(int(value) for value in slice_hint.stop)

    if len(start) != len(shape) or len(stop) != len(shape):
        raise ValueError(
            f"Slice hint dimensionality mismatch: expected {len(shape)}, "
            f"got start={len(start)} stop={len(stop)}"
        )

    for axis, (axis_start, axis_stop, axis_shape) in enumerate(
        zip(start, stop, shape, strict=True)
    ):
        if axis_start < 0 or axis_stop < 0:
            raise ValueError(f"Slice bounds must be non-negative on axis {axis}")
        if axis_start > axis_stop:
            raise ValueError(f"Slice start must be <= stop on axis {axis}")
        if axis_stop > axis_shape:
            raise ValueError(f"Slice stop exceeds tensor shape on axis {axis}")

    return start, stop


def normalized_scale_hint(
    shape: Tuple[int, ...],
    scale_hint: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Normalize scale hint from request.

    Args:
        shape: Tensor shape
        scale_hint: Optional scale hint from request (repeated int64 field)

    Returns:
        Scale hint tuple if valid and non-trivial, None otherwise

    Raises:
        ValueError: If scale hint dimensionality mismatch or invalid values
    """
    if scale_hint is None or len(scale_hint) == 0:
        return None

    scale_hint_tuple = tuple(int(value) for value in scale_hint)
    if len(scale_hint_tuple) != len(shape):
        raise ValueError(
            f"Scale hint dimensionality mismatch: expected {len(shape)}, got {len(scale_hint_tuple)}"
        )

    for axis, scale in enumerate(scale_hint_tuple):
        if scale <= 0:
            raise ValueError(f"Scale hint must be positive on axis {axis}")

    if all(scale == 1 for scale in scale_hint_tuple):
        return None

    return scale_hint_tuple


# =============================================================================
# Size Estimation Helpers
# =============================================================================


def estimate_chunk_bytes(shape: Tuple[int, ...], dtype: str) -> int:
    """Estimate chunk size in bytes from shape and dtype.

    Args:
        shape: Chunk shape
        dtype: Data type string

    Returns:
        Estimated size in bytes
    """
    num_elements = int(np.prod(shape, dtype=np.int64))
    return num_elements * np.dtype(dtype).itemsize


# Defaults mirroring biopb-mcp's [pyramid] config (build_pyramid_levels). These
# decide the coarsest pyramid level the client requests on open; the precache
# worker must warm exactly that scale or its chunk_ids won't match. Keep in sync
# with biopb-mcp/src/biopb_mcp/_config.py if that is retuned.
PRECACHE_THRESHOLD = 4096
PRECACHE_DOWNSCALE_FACTOR = 4
PRECACHE_PIXEL_BUDGET_CUBIC_ROOT = 512


def compute_safe_chunk_size(
    chunk_size: Tuple[int, ...],
    dtype: str,
    dim_labels: Optional[List[str]],
) -> Tuple[int, ...]:
    """Compute a chunk size that fits within Arrow batch limit.

    Uses hierarchical splitting: split along highest priority axis first,
    then next priority axis if still too large, etc.

    Args:
        chunk_size: Original chunk size tuple
        dtype: Data type string
        dim_labels: Optional dimension labels for semantic axis mapping

    Returns:
        Chunk size tuple guaranteed to fit within MAX_ARROW_BATCH_BYTES
    """
    chunk_bytes = estimate_chunk_bytes(chunk_size, dtype)

    if chunk_bytes <= MAX_ARROW_BATCH_BYTES:
        return chunk_size

    # Hierarchical splitting: iteratively reduce axes by priority
    safe_size = list(chunk_size)
    axes_already_split = set()  # Track axes we've already reduced

    while chunk_bytes > MAX_ARROW_BATCH_BYTES:
        # Calculate how many more splits we need
        n_splits_needed = int(np.ceil(chunk_bytes / MAX_ARROW_BATCH_BYTES))

        # Choose next axis to split (excluding already-split axes)
        split_axis = _choose_split_axis_excluding(
            tuple(safe_size), dim_labels, n_splits_needed, axes_already_split
        )

        if split_axis is None:
            # No more axes can be split - shouldn't happen if MAX_ARROW_BATCH_BYTES > 0
            logger.warning(
                f"Cannot split chunk further: size={safe_size}, "
                f"bytes={chunk_bytes}, target={MAX_ARROW_BATCH_BYTES}"
            )
            break

        # Calculate splits for this axis
        axis_size = safe_size[split_axis]
        # Number of splits on this axis (at least 2, at most axis_size)
        n_axis_splits = min(axis_size, max(2, n_splits_needed))

        # Reduce axis size
        safe_size[split_axis] = axis_size // n_axis_splits
        axes_already_split.add(split_axis)

        # Recalculate bytes
        chunk_bytes = estimate_chunk_bytes(tuple(safe_size), dtype)

    return tuple(safe_size)


# Hard ceiling on the source region a single scaled chunk materializes
# server-side. This is a resident-memory bound, not a throughput target -- the
# read block wants to be as large as the reduction allows, and this is what
# stops it from being unbounded. It bounds a working set rather than an Arrow
# batch, so it is far larger than PREFERRED_ARROW_BATCH_BYTES.
MAX_READ_BLOCK_BYTES = 512 * 1024 * 1024


def scaled_virtual_chunk_size(
    transfer_chunk_size: Tuple[int, ...],
    tensor_shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    dtype: str,
    dim_labels: Optional[List[str]],
    output_dtype: Optional[str] = None,
    max_read_block_bytes: int = MAX_READ_BLOCK_BYTES,
    maximum_bytes: int = MAX_ARROW_BATCH_BYTES,
) -> Tuple[int, ...]:
    """Size the source extent one scaled chunk reads.

    A scaled chunk reads ``virtual`` source elements to deliver ``virtual //
    scale`` of them, so pinning the read extent to the full-resolution transfer
    size shrinks the payload by the scale factor per axis while doing identical
    read work -- a 1/32 read delivered 1/1024 of the transfer target
    (biopb/biopb#805). Growing the extent in whole units restores it.

    The extent is therefore bounded by ``transfer * scale`` per axis, so the
    delivered chunk stays at the transfer target and the endpoint count falls
    with the scale, and separately by ``max_read_block_bytes``, so a deep level
    cannot take that to an unbounded read.

    Growth happens in whole units of ``lcm(transfer, scale)`` -- a multiple of
    the scale, so the logical chunks tile without overlapping, and a multiple of
    the transfer extent, so reads stay aligned to the grid the adapter reports.
    It is clamped by ``max_read_block_bytes``, which exists to bound resident
    memory rather than to tune throughput. Nothing holds a minimum endpoint
    count: ``transfer * scale`` already pins the delivered chunk to the transfer
    target, so a read that fits in one chunk is one chunk.

    Growth uses the generic axis priority, but it moves in whole units, so an
    adapter that folded its physical layout into the transfer grid keeps it
    here: an interleaved ND2's ``[1, C, 1, y, x]`` unit can only be multiplied,
    never cut, so C stays whole through a scaled read too (biopb/biopb#809).
    """
    unit = tuple(
        min(lcm(int(transfer), max(1, int(scale))), int(shape))
        for transfer, scale, shape in zip(
            transfer_chunk_size, scale_hint, tensor_shape, strict=True
        )
    )
    scale_product = 1
    for scale in scale_hint:
        scale_product *= max(1, int(scale))
    # Ceiling 1, and the one that decides the shape of a normal read: enough
    # source pixels to reduce to a transfer-sized chunk, and no more. Without it
    # the block grows to whatever memory allows and the *delivered* chunk
    # overshoots the target compute_transfer_chunk_size picked -- at a 512 MiB
    # memory ceiling a 1/2 read delivered 62.9 MB against an 8 MB target. It is
    # also what makes the endpoint count fall with the scale: a 1/S read has 1/S
    # the output pixels, so at a fixed delivered size it needs 1/S the reads.
    scaled_target = estimate_chunk_bytes(transfer_chunk_size, dtype) * scale_product
    # Ceiling 2: resident memory, the operator's knob.
    # Ceiling 3: a backstop. The reduced block crosses the wire, so it must clear
    # the Arrow bound. scaled_target already implies this while get_output_dtype
    # preserves width, but it takes a reduction_method, so a future widening
    # method would not silently breach the wire.
    source_itemsize = np.dtype(dtype).itemsize
    result_itemsize = np.dtype(output_dtype or dtype).itemsize
    wire_limit = (
        maximum_bytes * scale_product * source_itemsize // max(1, result_itemsize)
    )
    # An lcm unit can exceed the hard limits on its own: nothing bounds a
    # client's scale_hint, and a scale coprime with the transfer extent makes
    # lcm() explode -- scale (5, 7, 11) against a 64x2048x2048 transfer chunk
    # asks for a 129 GB block. Alignment to the transfer grid is a performance
    # nicety; any multiple of the scale tiles correctly. So drop the alignment
    # rather than the memory bound.
    hard_limit = min(max_read_block_bytes, wire_limit)
    if estimate_chunk_bytes(unit, dtype) > hard_limit:
        unit = tuple(
            min(max(1, int(scale)), int(extent))
            for scale, extent in zip(scale_hint, tensor_shape, strict=True)
        )
    # One output element per axis is the floor -- below it a chunk delivers
    # nothing -- so an extreme scale can still exceed the limits here.
    target_bytes = max(
        estimate_chunk_bytes(unit, dtype),
        min(scaled_target, max_read_block_bytes, wire_limit),
    )
    return _coalesce_chunk_size(unit, tensor_shape, dtype, dim_labels, target_bytes)


def default_transfer_chunk_shape(
    tensor_shape: Sequence[int],
    dtype: str,
    dim_labels: Optional[Sequence[str]] = None,
    native: Optional[Sequence[int]] = None,
) -> List[int]:
    """The transfer grid for an adapter with no layout knowledge to apply.

    ``chunk_shape`` is the *transfer* grid and the adapter owns it
    (biopb/biopb#809): the server sizes nothing on the adapter's behalf, it only
    clamps the result to the Arrow ceiling. An adapter that knows how its bytes
    sit on disk -- an interleaved ND2 whose channels are one unit, a page-aligned
    TIFF -- states that grid directly. Every other adapter calls this.

    ``native`` seeds the search with the store's own block (zarr chunks, a TIFF
    tile, one plane) so the grid stays a whole multiple of it; omit it and the
    seed is the whole tensor, divided down. The seed is an *alignment* hint, not
    a read unit: no read is ever issued at it.
    """
    shape = tuple(int(dim) for dim in tensor_shape)
    labels = list(dim_labels) if dim_labels else None
    if native is not None and len(native) == len(shape):
        seed = tuple(max(1, int(dim)) for dim in native)
    else:
        seed = shape
    return list(compute_transfer_chunk_size(seed, shape, dtype, labels))


def compute_transfer_chunk_size(
    native_chunk_size: Tuple[int, ...],
    tensor_shape: Tuple[int, ...],
    dtype: str,
    dim_labels: Optional[List[str]],
    preferred_bytes: Optional[int] = None,
    maximum_bytes: Optional[int] = None,
) -> Tuple[int, ...]:
    """Size a transfer grid around ``native_chunk_size``.

    The engine behind :func:`default_transfer_chunk_shape`, and the sizing policy
    an adapter reuses when it wants the standard treatment of a grid it has
    already shaped. Blocks above ``preferred_bytes`` are divided with the
    established T/unknown -> C -> Z -> Y/X priority; smaller blocks are coalesced
    in whole ``native_chunk_size`` multiples, preferring Y/X -> Z -> C ->
    T/unknown, while retaining enough endpoints for parallel reads and scheduler
    utilization.

    ``maximum_bytes`` is the hard wire ceiling; ``preferred_bytes`` is the
    optimization target. Both default to the module constants, read at call time
    so a sweep can move them. Scaled reads do not run this optimizer a second
    time: :func:`scaled_virtual_chunk_size` derives their read extent from the
    chosen transfer grid, so the reduced chunk they deliver lands on the target.
    """
    preferred_bytes = (
        PREFERRED_ARROW_BATCH_BYTES if preferred_bytes is None else preferred_bytes
    )
    maximum_bytes = MAX_ARROW_BATCH_BYTES if maximum_bytes is None else maximum_bytes
    if len(native_chunk_size) != len(tensor_shape):
        raise ValueError(
            "Native chunk rank must match tensor rank: "
            f"chunk={len(native_chunk_size)} shape={len(tensor_shape)}"
        )
    if preferred_bytes <= 0 or maximum_bytes <= 0:
        raise ValueError("Chunk byte targets must be positive")
    if preferred_bytes > maximum_bytes:
        raise ValueError("Preferred chunk bytes must not exceed the maximum")
    if any(int(dim) <= 0 for dim in tensor_shape):
        raise ValueError(f"Tensor dimensions must be positive: {tensor_shape}")
    if any(int(dim) <= 0 for dim in native_chunk_size):
        raise ValueError(
            f"Native chunk dimensions must be positive: {native_chunk_size}"
        )

    native = tuple(
        min(int(chunk), int(shape))
        for chunk, shape in zip(native_chunk_size, tensor_shape, strict=True)
    )
    native_bytes = estimate_chunk_bytes(native, dtype)

    if native_bytes > preferred_bytes:
        result = _divide_chunk_size(native, dtype, dim_labels, preferred_bytes)
    elif native_bytes < preferred_bytes:
        result = _coalesce_chunk_size(
            native,
            tensor_shape,
            dtype,
            dim_labels,
            preferred_bytes,
        )
    else:
        result = native

    # Retain an explicit final guard so a future preferred-size policy change
    # cannot accidentally weaken the Arrow safety bound.
    if estimate_chunk_bytes(result, dtype) > maximum_bytes:
        result = compute_safe_chunk_size(result, dtype, dim_labels)
    return result


def _divide_chunk_size(
    chunk_size: Tuple[int, ...],
    dtype: str,
    dim_labels: Optional[List[str]],
    target_bytes: int,
) -> Tuple[int, ...]:
    """Divide a chunk toward ``target_bytes`` with the established priority."""
    result = list(chunk_size)
    result_bytes = estimate_chunk_bytes(tuple(result), dtype)
    split_axes: Set[int] = set()
    labels = [str(label).lower() for label in dim_labels] if dim_labels else []

    while result_bytes > target_bytes:
        n_splits = int(np.ceil(result_bytes / target_bytes))
        split_count = n_splits
        axis = _choose_split_axis_excluding(
            tuple(result), dim_labels, n_splits, split_axes
        )
        if axis is None:
            # The desired ratio may exceed every one axis even though dividing
            # several axes successively can reach it.
            axis = _choose_split_axis_excluding(
                tuple(result), dim_labels, 2, split_axes
            )
            split_count = 2
        if axis is None:
            break
        label = labels[axis] if axis < len(labels) else ""
        if label in {"y", "x"}:
            spatial_axes = [
                index
                for index, candidate_label in enumerate(labels)
                if candidate_label in {"y", "x"} and result[index] > 1
            ]
            if len(spatial_axes) == 2:
                scale = (target_bytes / result_bytes) ** 0.5
                for spatial_axis in spatial_axes:
                    result[spatial_axis] = max(1, int(result[spatial_axis] * scale))
                    split_axes.add(spatial_axis)
                result_bytes = estimate_chunk_bytes(tuple(result), dtype)
                continue
        result[axis] = max(1, result[axis] // min(result[axis], split_count))
        split_axes.add(axis)
        result_bytes = estimate_chunk_bytes(tuple(result), dtype)

    return tuple(result)


def _coalesce_chunk_size(
    native: Tuple[int, ...],
    tensor_shape: Tuple[int, ...],
    dtype: str,
    dim_labels: Optional[List[str]],
    target_bytes: int,
) -> Tuple[int, ...]:
    """Grow whole native blocks toward ``target_bytes``.

    Y and X grow **as a pair**, never one at a time: a chunk that is square-ish
    in the plane is what makes a square region read touch few chunks. Growing
    them independently reaches the same byte target with a better sequential
    read -- a full-width band is contiguous on disk -- but turns a 512x512 tile
    into one fetch per band it crosses. Coupling is the middle ground: the plane
    stays roughly square, and the axes that are genuinely free to differ (Z, C,
    T) still grow on their own.

    Growth stops at ``target_bytes`` and at the tensor's own extent -- nothing
    else. A tensor small enough to fit in one chunk becomes one chunk.
    """
    current = list(native)
    max_blocks = [
        int(shape) // int(block)
        for block, shape in zip(native, tensor_shape, strict=True)
    ]
    labels = [str(label).lower() for label in dim_labels] if dim_labels else []

    def priority(axis: int) -> int:
        label = labels[axis] if axis < len(labels) else ""
        if label in {"y", "x"}:
            return 0
        if label == "z":
            return 1
        if label in {"c", "channel", "channels"}:
            return 2
        if label in {"t", "time", "frame", "frames"}:
            return 3
        return 4

    def blocks(axis: int) -> int:
        return current[axis] // native[axis]

    spatial = [axis for axis in range(len(current)) if priority(axis) == 0]
    spatial_set = set(spatial)

    def doubled_spatial() -> Optional[List[int]]:
        """The spatial pair with every growable axis doubled, or None."""
        candidate = list(current)
        for axis in spatial:
            new_blocks = min(max_blocks[axis], blocks(axis) * 2)
            candidate[axis] = native[axis] * new_blocks
        return candidate if candidate != current else None

    while True:
        candidates = []
        for axis in range(len(current)):
            if axis in spatial_set:
                continue
            old_blocks = blocks(axis)
            new_blocks = min(max_blocks[axis], old_blocks * 2)
            if new_blocks <= old_blocks:
                continue
            candidate = list(current)
            candidate[axis] = native[axis] * new_blocks
            candidate_bytes = estimate_chunk_bytes(tuple(candidate), dtype)
            if candidate_bytes <= target_bytes:
                candidates.append(
                    (
                        priority(axis),
                        current[axis],
                        candidate_bytes,
                        axis,
                        candidate,
                    )
                )
        # One candidate for the whole spatial pair, so Y and X move together.
        spatial_candidate = doubled_spatial()
        if spatial_candidate is not None:
            candidate_bytes = estimate_chunk_bytes(tuple(spatial_candidate), dtype)
            if candidate_bytes <= target_bytes:
                candidates.append(
                    (
                        0,
                        current[spatial[0]],
                        candidate_bytes,
                        spatial[0],
                        spatial_candidate,
                    )
                )
        if not candidates:
            break
        _, _, _, _, current = min(
            candidates, key=lambda item: (item[0], item[1], -item[2], item[3])
        )

    # Consume any remaining budget. The spatial pair goes first and stays square:
    # size the plane from the budget's square root, then round each axis down to
    # whole native blocks. Doubling has already brought it within a factor of two,
    # so the back-off below runs a block or two at most.
    if spatial and _fill_spatial(
        current, native, max_blocks, spatial, spatial_set, dtype, target_bytes
    ):
        return tuple(current)

    current_bytes = estimate_chunk_bytes(tuple(current), dtype)
    candidates = []
    for axis in range(len(current)):
        if axis in spatial_set:
            continue
        other_bytes = current_bytes // current[axis]
        affordable_blocks = target_bytes // (other_bytes * native[axis])
        new_blocks = min(max_blocks[axis], int(affordable_blocks))
        if new_blocks <= blocks(axis):
            continue
        candidate = list(current)
        candidate[axis] = native[axis] * new_blocks
        candidate_bytes = estimate_chunk_bytes(tuple(candidate), dtype)
        candidates.append(
            (priority(axis), target_bytes - candidate_bytes, axis, candidate)
        )
    if candidates:
        _, _, _, current = min(candidates, key=lambda item: item[:3])

    return tuple(current)


def _fill_spatial(
    current: List[int],
    native: Tuple[int, ...],
    max_blocks: List[int],
    spatial: List[int],
    spatial_set: Set[int],
    dtype: str,
    target_bytes: int,
) -> bool:
    """Spend the remaining budget on Y/X together, keeping the plane square.

    Mutates ``current`` in place and reports whether it grew. Sizing the plane
    from ``sqrt(budget)`` rather than stepping one axis keeps the two extents
    within a block of each other, which is the whole point of coupling them.
    """
    per_element = estimate_chunk_bytes(
        tuple(
            1 if axis in spatial_set else current[axis] for axis in range(len(current))
        ),
        dtype,
    )
    if per_element <= 0:
        return False
    plane_budget = target_bytes // per_element
    if plane_budget <= 0:
        return False

    side = max(1, int(plane_budget**0.5))
    proposed = list(current)
    for axis in spatial:
        want = max(current[axis] // native[axis], side // native[axis])
        proposed[axis] = native[axis] * min(max_blocks[axis], max(1, want))

    # Whole blocks can overshoot the square: back the longer extent off a block
    # at a time, never below where doubling already got us.
    while estimate_chunk_bytes(tuple(proposed), dtype) > target_bytes:
        shrinkable = [
            axis
            for axis in spatial
            if proposed[axis] // native[axis] > current[axis] // native[axis]
        ]
        if not shrinkable:
            return False
        axis = max(shrinkable, key=lambda a: (proposed[a], a))
        proposed[axis] -= native[axis]

    if proposed == current:
        return False
    current[:] = proposed
    return True


def _choose_split_axis_excluding(
    shape: Tuple[int, ...],
    dim_labels: Optional[List[str]],
    n_splits: int,
    exclude_axes: Set[int],
) -> Optional[int]:
    """Choose axis for splitting, excluding already-split axes.

    Priority (highest first): non-spatial axes (t/v/frame/unlabeled, largest
    wins), then 'c', then 'z', then the larger of 'y'/'x' -- skipping any axis
    in exclude_axes and any that cannot accommodate n_splits.

    Returns None if no eligible axis can accommodate n_splits.
    """
    SPATIAL_LABELS = {"y", "x", "z", "c"}

    # Build label -> axis mapping
    label_to_axis: Dict[str, int] = {}
    if dim_labels:
        for ax, label in enumerate(dim_labels):
            label_to_axis[label.lower()] = ax

    # Interleaved RGB(A) samples are one pixel's indivisible components. Keep
    # them together and divide another axis; an unlabeled trailing size-3/4 axis
    # is deliberately not inferred as samples.
    sample_axis = samples_axis(list(dim_labels or []), shape)

    # Eligible axes: not excluded, not interleaved samples, and splittable.
    eligible = [
        ax
        for ax in range(len(shape))
        if ax not in exclude_axes and ax != sample_axis and shape[ax] >= 2
    ]

    if not eligible:
        return None

    # Priority 1: Non-spatial axes (t, v, frame, etc.)
    non_spatial = []
    if dim_labels:
        for ax in eligible:
            label = dim_labels[ax].lower()
            if label not in SPATIAL_LABELS:
                non_spatial.append(ax)
    else:
        non_spatial = eligible

    if non_spatial:
        return max(non_spatial, key=lambda ax: shape[ax])

    # Priority 2: 'c' (channel)
    if "c" in label_to_axis:
        c_ax = label_to_axis["c"]
        if c_ax in eligible:
            return c_ax

    # Priority 3: 'z' (depth)
    if "z" in label_to_axis:
        z_ax = label_to_axis["z"]
        if z_ax in eligible:
            return z_ax

    # Priority 4: Larger of 'y' or 'x'
    y_ax = label_to_axis.get("y")
    x_ax = label_to_axis.get("x")
    # ``None in eligible`` is safely False, so no guard is needed -- and testing
    # ``y_ax`` directly would wrongly reject axis 0 (a falsy but valid index).
    y_eligible = y_ax in eligible
    x_eligible = x_ax in eligible

    if y_eligible and x_eligible:
        return y_ax if shape[y_ax] >= shape[x_ax] else x_ax
    elif y_eligible:
        return y_ax
    elif x_eligible:
        return x_ax

    # Fallback: largest eligible axis
    return max(eligible, key=lambda ax: shape[ax])
