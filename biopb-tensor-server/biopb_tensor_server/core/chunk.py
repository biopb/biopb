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

from biopb_tensor_server.core.axes import labeled_axis_index
from biopb_tensor_server.core.downsample import (
    DEFAULT_REDUCTION_METHOD,
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
# 64MB threshold for chunk splitting - enables parallel Flight transfers
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


def needs_splitting(chunk_shape: Tuple[int, ...], dtype: str) -> bool:
    """Check if chunk exceeds Arrow batch limit.

    Args:
        chunk_shape: Chunk shape
        dtype: Data type string

    Returns:
        True if chunk needs splitting
    """
    return estimate_chunk_bytes(chunk_shape, dtype) > MAX_ARROW_BATCH_BYTES


# Defaults mirroring biopb-mcp's [pyramid] config (build_pyramid_levels). These
# decide the coarsest pyramid level the client requests on open; the precache
# worker must warm exactly that scale or its chunk_ids won't match. Keep in sync
# with biopb-mcp/src/biopb_mcp/_config.py if that is retuned.
PRECACHE_THRESHOLD = 4096
PRECACHE_DOWNSCALE_FACTOR = 4
PRECACHE_PIXEL_BUDGET_CUBIC_ROOT = 512


def _precache_xy_indices(shape: Sequence[int], dim_labels) -> Tuple[int, int]:
    """(y_idx, x_idx), matching biopb-mcp's get_xy_dim_indices.

    Prefers a y/x-labeled axis (by synonym, via :func:`core.axes.labeled_axis_index`);
    falls back to the ``[..., Y, X]`` convention (X last, Y second-to-last) when
    either is unlabeled.
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
    """Index of the z axis or None, matching biopb-mcp's get_z_dim_index.

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
) -> List[List[int]]:
    """Per-axis scale_hint for *every* level of a computed pyramid.

    A faithful port of biopb-mcp's ``build_pyramid_levels`` loop, emitting the
    full sequence of levels (not just the coarsest): level 0 is full resolution
    (all 1s), then X, Y and Z are downsampled individually (all other axes stay
    at 1), each stopping at ``axis_floor = min(pixel_budget_cubic_root,
    threshold)``, until the level satisfies ``Lx*Ly*Lz <=
    pixel_budget_cubic_root**3`` and ``Lx, Ly <= threshold``. ``ceil_div(L, s)``
    is the server's own ``logical_shape`` (adapter_base.py), so each scale matches the
    client's level and the warmed chunk_ids line up exactly.

    A tensor with no z axis is treated as ``Lz = 1`` and never gets a z factor.

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

    sx = sy = sz = 1
    scales = [_scale_vector(sx, sy, sz)]  # level 0: full resolution
    while True:
        lx = ceil_div(shape[x_idx], sx)
        ly = ceil_div(shape[y_idx], sy)
        lz = ceil_div(shape[z_idx], sz) if z_idx is not None else 1
        if lx * ly * lz <= budget and lx <= threshold and ly <= threshold:
            break
        nsx = sx * downscale_factor if lx > floor else sx
        nsy = sy * downscale_factor if ly > floor else sy
        nsz = sz * downscale_factor if (z_idx is not None and lz > floor) else sz
        if (nsx, nsy, nsz) == (sx, sy, sz):
            break  # nothing left to shrink; avoid an infinite loop
        sx, sy, sz = nsx, nsy, nsz
        scales.append(_scale_vector(sx, sy, sz))

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
    reduction_method: str = "area",
    threshold: int = PRECACHE_THRESHOLD,
    downscale_factor: int = PRECACHE_DOWNSCALE_FACTOR,
    pixel_budget_cubic_root: int = PRECACHE_PIXEL_BUDGET_CUBIC_ROOT,
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

    # Eligible axes: not excluded and large enough for splits
    eligible = [
        ax for ax in range(len(shape)) if ax not in exclude_axes and shape[ax] >= 2
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
