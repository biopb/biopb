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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import PyramidLevel, SliceHint
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.axes import labeled_axis_index, samples_axis
from biopb_tensor_server.core.downsample import (
    CHUNK_ID_IMPLICIT_REDUCTION_METHOD,
    ceil_div,
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
# - [scaled only] 1 byte: reduction_method code
#
# The chunk_id is IDENTITY (array_id + bounds + scale_hint + method). #178 had
# dropped reduction_method from the wire -- it was advisory and the compute path
# hard-coded the default, which silently served a client's requested method with
# the wrong one (biopb/biopb#578). It is back, and every computed method carries
# its own code byte: the id says which method it is, never which one it is not.
#
# That is the property worth protecting. Spelling one method by the ABSENCE of a
# byte ties the wire format -- and, since these bytes are the cache key, every
# entry already written -- to whichever method happens to be the request default.
# Moving that default then re-reads ids that are already on disk (same key,
# different pixels) and makes the old default unreachable, because an explicit
# request for it encodes byte-free and decodes back as the new one. Both were
# observed when the default moved to "nearest"; a mandatory byte is what makes
# the two questions independent.
#
# The price, taken deliberately: a scaled AREA id is no longer byte-identical to
# the pre-#578 method-free form, so area entries warmed under the old encoding
# are orphaned -- unreachable, reclaimed by ordinary segment LRU, no cache
# format-version bump and no wipe. Reads of them are correct, just cold. This
# also reverses the #76 cache-sharing (nearest and area no longer collide), which
# is the cost of serving the method the client asked for.
#
# A byte-free scaled chunk_id (old server, old cache, or a proxy forwarding from
# an older upstream) stays readable and decodes to area -- see
# CHUNK_ID_IMPLICIT_REDUCTION_METHOD. Current scaled ids carry an explicit method,
# and cold compute uses that decoded method in core.adapter_base.resolve_chunk_data.
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


# Compact reduction_method suffix on a scaled chunk_id (biopb/biopb#578). EVERY
# computed method carries its own code byte -- there is no omitted method -- so
# the chunk_id says what it is rather than what it is not, and neither the wire
# nor the cache key depends on which method happens to be the request default.
#
# "area" is deliberately code 2 rather than a renumbering: an older server that
# knows only code 1 resolves an unknown code through its own absent-byte
# fallback, which is "area", so a mandatory-area id read by an old peer still
# lands on area.
#
# Only the COMPUTED methods are coded. "precompute" is not one: get_read_plan
# intercepts it and re-plans against the native level's own store, an unscaled
# read identified by its array_id (source_id/{level}), so it never reaches this
# encoder. Reaching it anyway is a routing bug and raises -- falling back to a
# byte-free id would mint something indistinguishable from a pre-#578 chunk_id
# and serve it as area.
_SCALED_METHOD_BYTE = {"nearest": b"\x01", "area": b"\x02"}
_SCALED_METHOD_BY_BYTE = {1: "nearest", 2: "area"}


def encode_chunk_id_with_scale(
    array_id: str,
    bounds: ChunkBounds,
    scale_hint: Tuple[int, ...],
    reduction_method: str = CHUNK_ID_IMPLICIT_REDUCTION_METHOD,
) -> bytes:
    """Encode a scaled chunk_id: bounds encoding + scale_hint [+ method byte].

    Format: standard bounds encoding, then 8*ndim bytes scale_hint (int64), then
    one method-code byte -- for every computed method, not just a non-default one
    (biopb/biopb#578, #178, #76). The method is normalized (stride->nearest,
    mean->area) before it is coded.

    This is what decouples the identifier from the request default: no method is
    spelled by its absence, so changing which method an unspecified read resolves
    to cannot re-read an id that is already written. The cost is that area ids
    minted before this change (byte-free) no longer match the ids minted now, so
    their cache entries are orphaned -- unreachable, and reclaimed by ordinary
    segment LRU rather than invalidated.

    Detection stays ``len(chunk_id) > bounds_end`` (a scaled chunk always carries
    at least the scale_hint); :func:`decode_reduction_method` reads the byte back.
    """
    base = encode_chunk_id(array_id, bounds)
    scale_payload = struct.pack(f">{len(scale_hint)}q", *scale_hint)
    normalized = normalize_reduction_method(reduction_method)
    try:
        method_suffix = _SCALED_METHOD_BYTE[normalized]
    except KeyError:
        raise ValueError(
            f"No chunk_id code for reduction_method {normalized!r}: a scaled "
            "chunk_id can only carry a computed method. 'precompute' is routed "
            "to its native level by get_read_plan and must not reach here."
        ) from None
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

    A current chunk_id is identity (array_id + bounds [+ scale_hint + method
    byte]), so the key equals the inner bytes -- INCLUDING the compact one-byte
    reduction_method suffix, so a "nearest" read keys distinctly from an "area"
    one (biopb/biopb#578). Only a LEGACY trailing method suffix (the pre-#178
    ``uint16 len + bytes`` form, which is more than one byte past the scale) is
    stripped. Non-scaled chunk_ids are returned unchanged.

    Since the method byte became mandatory, a scaled area key is no longer
    byte-identical to the pre-#578 method-free key: those entries are orphaned,
    not invalidated. Nothing looks them up and nothing rewrites them, so they sit
    until their segment is chosen by the ordinary size-driven segment LRU. That
    is a knowingly accepted one-time re-warm, taken instead of a cache
    format-version bump; the same is true of the legacy suffix form above, which
    now normalizes to a key this server no longer mints.

    The result is an opaque cache key: it is NOT a valid chunk_id and must not
    be fed to :func:`decode_scale_info` or forwarded on the wire.

    A content_version (biopb/biopb#178) is kept in the key -- so a version bump
    yields a distinct key and the stale entry becomes un-lookupable.

    A proxy envelope is returned as-is: it already frames (route, content_version,
    inner) with lengths, so it is an injective key, and since the inner carries a
    method byte on every scaled read, the envelope key distinguishes methods too
    -- WITHOUT the proxy ever parsing the opaque inner.
    """
    if is_proxy_envelope(chunk_id):
        return chunk_id
    cv, inner = _split_version(chunk_id)
    ndim, bounds_end = _bounds_end(inner)
    scale_end = bounds_end + ndim * 8
    # Keep array_id+bounds+scale_hint and at most the one-byte method suffix; a
    # longer trailing run is the legacy uint16 method form, stripped (#76).
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
    (exactly one byte past the scale_hint) is honored.

    The absent-byte fallback is now purely a compatibility path: this server
    mints a byte for every computed method, so a byte-free scaled chunk_id can
    only predate that -- an old cache entry, an id a client is still holding, or
    one a remote proxy forwarded from an older upstream. Everything minted before
    the byte became mandatory was area, which is what
    ``CHUNK_ID_IMPLICIT_REDUCTION_METHOD`` records. A non-scaled chunk_id and a
    legacy ``uint16 len + bytes`` method suffix resolve the same way, so an old
    scaled read is served exactly as before, never rejected.
    """
    _, inner = _split_version(chunk_id)
    ndim, bounds_end = _bounds_end(inner)
    scale_end = bounds_end + ndim * 8
    if len(inner) == scale_end + 1:
        return _SCALED_METHOD_BY_BYTE.get(
            inner[scale_end], CHUNK_ID_IMPLICIT_REDUCTION_METHOD
        )
    return CHUNK_ID_IMPLICIT_REDUCTION_METHOD


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
PRECACHE_DOWNSCALE_FACTOR = 2
# 448**3 = 90 Mvox. The coarsest level is uploaded whole as one 3-D texture by
# both renderers, and Viv casts it to float32 -- measured on a Quadro P2000,
# 90 Mvox holds 50 fps there where 512**3 (134 Mvox) drops to 17.
# See docs/precache-policy.md 9.1.
PRECACHE_PIXEL_BUDGET_CUBIC_ROOT = 448
# 2048**2. Caps the 2-D rungs, and is chosen to land on the level deck.gl asks
# for at fit-to-view in a ~1500-2000px window, so the warmed level is the one
# the browser actually reads.
PRECACHE_PLANE_MAX_PIXELS = 4_000_000


def _precache_xy_indices(shape: Sequence[int], dim_labels) -> Tuple[int, int]:
    """(y_idx, x_idx), agreeing with biopb-mcp's ``_resolve_axes``.

    Prefers a y/x-labeled axis (by synonym, via :func:`core.axes.labeled_axis_index`);
    falls back to the ``[..., Y, X]`` convention (X last, Y second-to-last) when
    either is unlabeled. The client reads the same two axes purely positionally
    now that the served order is canonical (biopb/biopb#596) -- for an order this
    server advertises, a labeled y/x *is* at that position, so the two agree.
    """
    ndim = len(shape)
    if dim_labels:
        y = labeled_axis_index(dim_labels, "y")
        x = labeled_axis_index(dim_labels, "x")
        if y is not None and x is not None:
            return y, x
    if ndim < 2:
        raise ValueError(f"Cannot identify x/y dimensions: tensor is {ndim}-D")
    return ndim - 2, ndim - 1


def _precache_z_index(shape: Sequence[int], dim_labels) -> Optional[int]:
    """Index of the z axis or None, agreeing with biopb-mcp's ``_resolve_axes``.

    Prefers a z-labeled axis (by synonym; absent label => no depth axis, never a
    positional guess -- an unlabeled leading axis may be T/C and must not be
    downsampled); else the positional ``[..., Z, Y, X]`` convention (third-from-
    last) for 3-D+ tensors.
    """
    ndim = len(shape)
    if dim_labels:
        return labeled_axis_index(dim_labels, "z")
    return ndim - 3 if ndim >= 3 else None


def compute_pyramid_scale_hints(
    shape: Sequence[int],
    dim_labels=None,
    threshold: int = PRECACHE_THRESHOLD,
    downscale_factor: int = PRECACHE_DOWNSCALE_FACTOR,
    pixel_budget_cubic_root: int = PRECACHE_PIXEL_BUDGET_CUBIC_ROOT,
    plane_max_pixels: int = PRECACHE_PLANE_MAX_PIXELS,
) -> List[List[int]]:
    """Per-axis scale_hint for *every* level of a computed pyramid.

    Two phases, because the two renderers want different things (see
    ``docs/precache-policy.md`` §4.1):

    - **2-D rungs**, X and Y only, halving by ``downscale_factor`` until the
      plane fits ``plane_max_pixels`` (and X/Y fit ``threshold``). Z is left
      alone: a 2-D view displays one slice, so scaling Z on these rungs discards
      depth resolution to save nothing.
    - **one final 3-D rung**, continuing from the last 2-D rung and scaling X, Y
      *and* Z until the volume fits ``pixel_budget_cubic_root**3``. napari's 3-D
      mode reads ``len(levels) - 1`` whole (``layers/_scalar_field/_slice.py``),
      so the coarsest level is what a renderer uploads as one texture and the
      budget is what bounds it. Appended only when the volume actually exceeds
      the budget, so a 2-D tensor never grows one.

    Starting the 3-D phase from the last 2-D rung rather than from full
    resolution gives per-axis monotonicity for free, which napari requires of
    ``downsample_factors`` (``layers/utils/layer_utils.py``).

    ``ceil_div(L, s)`` is the server's own ``logical_shape`` (adapter_base.py),
    so each scale matches the client's level and the warmed chunk_ids line up
    exactly. A tensor with no z axis is treated as ``Lz = 1``.

    Returns:
        Non-empty list of per-axis scale vectors, coarsest last.
    """
    ndim = len(shape)

    # A tensor with fewer than two axes has no Y/X plane to downsample, so there
    # is no meaningful pyramid -- advertise a single full-resolution level. This
    # also keeps build_pyramid_plan / get_flight_info from raising on 1-D (or 0-D)
    # tensors, where _precache_xy_indices has no X/Y to resolve.
    if ndim < 2:
        return [[1] * ndim]

    budget = pixel_budget_cubic_root**3
    floor = min(pixel_budget_cubic_root, threshold)

    y_idx, x_idx = _precache_xy_indices(shape, dim_labels)
    z_idx = _precache_z_index(shape, dim_labels)
    # A degenerate label set could map z onto an x/y axis; drop it if so.
    if z_idx is not None and z_idx in (x_idx, y_idx):
        z_idx = None

    def _scale_vector(sx, sy, sz):
        scale = [1] * ndim
        scale[x_idx] = sx
        scale[y_idx] = sy
        if z_idx is not None:
            scale[z_idx] = sz
        return scale

    def _extent(sx, sy, sz):
        return (
            ceil_div(shape[x_idx], sx),
            ceil_div(shape[y_idx], sy),
            ceil_div(shape[z_idx], sz) if z_idx is not None else 1,
        )

    # Phase 1 -- 2-D rungs: X and Y only.
    sx = sy = sz = 1
    scales = [_scale_vector(sx, sy, sz)]  # level 0: full resolution
    while True:
        lx, ly, _lz = _extent(sx, sy, sz)
        if lx * ly <= plane_max_pixels and lx <= threshold and ly <= threshold:
            break
        nsx = sx * downscale_factor if lx > floor else sx
        nsy = sy * downscale_factor if ly > floor else sy
        if (nsx, nsy) == (sx, sy):
            break  # nothing left to shrink; avoid an infinite loop
        sx, sy = nsx, nsy
        scales.append(_scale_vector(sx, sy, sz))

    # Phase 2 -- one 3-D rung, only if the volume still exceeds the budget.
    # Continues from the last 2-D rung, so its factors can only be >= that
    # rung's on every axis.
    tx, ty, tz = sx, sy, sz
    while True:
        lx, ly, lz = _extent(tx, ty, tz)
        if lx * ly * lz <= budget:
            break
        ntx = tx * downscale_factor if lx > floor else tx
        nty = ty * downscale_factor if ly > floor else ty
        ntz = tz * downscale_factor if (z_idx is not None and lz > floor) else tz
        if (ntx, nty, ntz) == (tx, ty, tz):
            break  # every axis is at the floor; the budget cannot be met
        tx, ty, tz = ntx, nty, ntz
    if (tx, ty, tz) != (sx, sy, sz):
        scales.append(_scale_vector(tx, ty, tz))

    return scales


def compute_precache_scale_hint(
    shape: Sequence[int],
    dim_labels=None,
    **kwargs: int,
) -> List[int]:
    """Per-axis scale_hint for the *coarsest* pyramid level a client requests.

    The last entry of :func:`compute_pyramid_scale_hints` (``threshold`` /
    ``downscale_factor`` / ``pixel_budget_cubic_root`` forwarded through) -- a
    named thin wrapper so there is one pyramid loop, not two.
    """
    return compute_pyramid_scale_hints(shape, dim_labels, **kwargs)[-1]


def build_pyramid_plan(
    shape: Sequence[int],
    dim_labels=None,
    reduction_method: str = "nearest",
    threshold: int = PRECACHE_THRESHOLD,
    downscale_factor: int = PRECACHE_DOWNSCALE_FACTOR,
    pixel_budget_cubic_root: int = PRECACHE_PIXEL_BUDGET_CUBIC_ROOT,
    plane_max_pixels: int = PRECACHE_PLANE_MAX_PIXELS,
) -> List[PyramidLevel]:
    """Server-advertised computed pyramid as a list of ``PyramidLevel`` protos.

    Wraps :func:`compute_pyramid_scale_hints` (level 0 = full resolution,
    coarsest last); each level carries its scale_hint, the on-the-fly
    ``reduction_method``, and its logical shape ``ceil_div(base, scale)`` -- the
    same extent ``get_read_plan`` returns for that scale, so a client can size
    the level without a probe read. ``native`` is False (computed, not on-disk).

    For tensors that ship a real pyramid, the adapter overrides this with native
    levels (see ``TensorAdapter.get_native_pyramid_levels``); this is the generic
    fallback for everything else.
    """
    scales = compute_pyramid_scale_hints(
        shape,
        dim_labels,
        threshold=threshold,
        downscale_factor=downscale_factor,
        pixel_budget_cubic_root=pixel_budget_cubic_root,
        plane_max_pixels=plane_max_pixels,
    )
    levels: List[PyramidLevel] = []
    for scale in scales:
        level_shape = [
            ceil_div(int(dim), s) for dim, s in zip(shape, scale, strict=True)
        ]
        levels.append(
            PyramidLevel(
                scale_hint=scale,
                reduction_method=reduction_method,
                shape=level_shape,
                native=False,
            )
        )
    return levels


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


def scaled_virtual_chunk_size(
    transfer_chunk_size: Tuple[int, ...],
    tensor_shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    dtype: str,
    dim_labels: Optional[List[str]] = None,
    output_dtype: Optional[str] = None,
) -> Tuple[int, ...]:
    """Size the source extent one scaled chunk reads: ``transfer * scale``.

    A scaled chunk reads ``extent`` source elements to deliver ``extent //
    scale`` of them, so pinning the extent to the full-resolution transfer size
    shrinks the payload by the scale factor per axis while doing identical read
    work -- a 1/32 read delivered 1/1024 of the transfer target
    (biopb/biopb#805). Multiplying by the scale restores it exactly: the
    delivered chunk is the transfer chunk, and the endpoint count falls with the
    scale because each chunk covers ``scale`` times more source per axis.

    The product is a multiple of both the scale and the transfer extent, so
    chunks tile without splitting a reduction block and stay aligned to the grid
    the adapter reports. Clamping to the tensor keeps that: the clamp can only
    bite on the last chunk of an axis, which ends at the tensor's end anyway.

    **Nothing else shapes it** -- no byte target, no coalescing, no memory
    ceiling -- because ``get_scaled_data`` streams the extent rather than
    materialising it (``core/stream_reduce.py``): residency is one unit, so the
    extent is free to be as large as the tensor. What a byte target used to buy,
    a delivered chunk at the transfer size, ``transfer * scale`` gives exactly;
    what it cost was growth along whatever axis happened to be free once the
    scaled axes saturated against the tensor. On a 12-plane TIFF at scale 32
    that was ten Z planes -- 483 MiB read to deliver one requested plane's
    0.05 MiB, where this reads 48 MiB.

    ``dim_labels`` and ``output_dtype`` are vestigial: they described where the
    coalescing was allowed to grow and how wide the result would land, and
    nothing grows any more.
    """
    return tuple(
        min(max(1, int(transfer)) * max(1, int(scale)), int(dim))
        for transfer, scale, dim in zip(
            transfer_chunk_size, scale_hint, tensor_shape, strict=True
        )
    )


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
