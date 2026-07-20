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
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import PyramidLevel, SliceHint
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.downsample import ceil_div

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
#
# The chunk_id is pure IDENTITY (array_id + bounds + scale_hint). The scaled form
# used to carry a trailing reduction_method (uint16 len + bytes), but the cache
# key always stripped it (advisory, biopb/biopb#76) and the compute path is the
# only consumer -- so the method left the wire format entirely (biopb/biopb#178).
# A cold downsample uses the server default; see core.base.resolve_chunk_data.
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
# re-encode). The whole codec strips this header first, so decode / scale / rewrite
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
# three apart from byte 0. Because the inner is method-free (#178 made chunk_ids
# identity-only), the whole envelope is an injective, reduction_method-independent
# cache key -- see cache_key_for_chunk_id.
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
    bounds: "ChunkBounds",
) -> bytes:
    """Encode array_id and bounds into chunk_id."""
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
    """
    cv, inner = _split_version(chunk_id)
    old_len = struct.unpack(">I", inner[:4])[0]
    tail = inner[4 + old_len :]
    new_bytes = new_array_id.encode("utf-8")
    rewritten = struct.pack(">I", len(new_bytes)) + new_bytes + tail
    return wrap_content_version(rewritten, cv) if cv is not None else rewritten


def decode_chunk_id(chunk_id: bytes) -> Tuple[str, "ChunkBounds"]:
    """Decode array_id and bounds from chunk_id. Works for both regular
    and virtual chunk_ids (ignores virtual payload) and version-wrapped ones."""
    _, chunk_id = _split_version(chunk_id)
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
) -> bytes:
    """Encode an identity scaled chunk_id: bounds encoding + scale_hint.

    Format: standard bounds encoding, then 8*ndim bytes scale_hint (int64).
    Detection: if ``len(chunk_id) > bounds_end``, it's a scaled chunk. The
    reduction_method is NOT encoded -- it is advisory and sourced server-side at
    compute time (biopb/biopb#178, #76).
    """
    base = encode_chunk_id(array_id, bounds)
    scale_payload = b"".join(struct.pack(">q", s) for s in scale_hint)
    return base + scale_payload


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

    A current chunk_id is already pure identity (array_id + bounds [+ scale_hint]),
    so the key equals the inner bytes. The projection to ``bounds_end + ndim*8`` is
    retained only to stay byte-compatible with an OLDER chunk_id that still carried
    a trailing reduction_method suffix (biopb/biopb#76): dropping it means a cache
    entry warmed under the old format is still hit under the new one. Non-scaled
    chunk_ids are returned unchanged.

    The result is an opaque cache key: it is NOT a valid chunk_id and must not
    be fed to :func:`decode_scale_info` or forwarded on the wire.

    A content_version (biopb/biopb#178) is kept in the key -- so a version bump
    yields a distinct key and the stale entry becomes un-lookupable -- while the
    inner projection stays byte-identical to the pre-#178 key, so an UNVERSIONED
    chunk_id maps to exactly its old cache entry (no forced invalidation).

    A proxy envelope is returned as-is: it already frames (route, content_version,
    inner) with lengths, so it is an injective key, and since its inner is
    method-free it is reduction_method-independent -- keying by it verbatim keeps
    #76 dedup WITHOUT the proxy ever parsing the opaque inner.
    """
    if is_proxy_envelope(chunk_id):
        return chunk_id
    cv, inner = _split_version(chunk_id)
    ndim, bounds_end = _bounds_end(inner)
    base = inner if len(inner) <= bounds_end else inner[: bounds_end + ndim * 8]
    return wrap_content_version(base, cv) if cv is not None else base


def decode_scale_info(chunk_id: bytes) -> Tuple[int, ...]:
    """Decode the scale_hint from a scaled chunk_id.

    Reads only the ndim int64 scale_hint after the bounds encoding. The
    reduction_method is no longer part of the chunk_id (biopb/biopb#178); any
    trailing bytes from an older method-carrying chunk_id are ignored.
    """
    _, chunk_id = _split_version(chunk_id)
    ndim, bounds_end = _bounds_end(chunk_id)

    scale_hint = []
    for ax in range(ndim):
        scale_hint.append(
            struct.unpack(
                ">q", chunk_id[bounds_end + ax * 8 : bounds_end + ax * 8 + 8]
            )[0]
        )

    return tuple(scale_hint)


# Constants
# 64MB threshold for chunk splitting - enables parallel Flight transfers
MAX_ARROW_BATCH_BYTES = 64 * 1024 * 1024

if TYPE_CHECKING:
    pass


@dataclass
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

    Prefers 'y'/'x' dim_labels; falls back to the ``[..., Y, X]`` convention
    (X last, Y second-to-last).
    """
    ndim = len(shape)
    if dim_labels:
        labels_lower = [str(label).lower() for label in dim_labels]
        try:
            return labels_lower.index("y"), labels_lower.index("x")
        except ValueError:
            pass
    if ndim < 2:
        raise ValueError(f"Cannot identify x/y dimensions: tensor is {ndim}-D")
    return ndim - 2, ndim - 1


def _precache_z_index(shape: Sequence[int], dim_labels) -> Optional[int]:
    """Index of the z axis or None, matching biopb-mcp's get_z_dim_index.

    Prefers a 'z' dim_label (absent label => no depth axis); else the positional
    ``[..., Z, Y, X]`` convention (third-from-last) for 3-D+ tensors.
    """
    ndim = len(shape)
    if dim_labels:
        labels_lower = [str(label).lower() for label in dim_labels]
        return labels_lower.index("z") if "z" in labels_lower else None
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
    is the server's own ``logical_shape`` (base.py), so each scale matches the
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
    threshold: int = PRECACHE_THRESHOLD,
    downscale_factor: int = PRECACHE_DOWNSCALE_FACTOR,
    pixel_budget_cubic_root: int = PRECACHE_PIXEL_BUDGET_CUBIC_ROOT,
) -> List[int]:
    """Per-axis scale_hint for the *coarsest* pyramid level a client requests.

    The coarsest level the precache worker warms -- the last entry of
    :func:`compute_pyramid_scale_hints`, kept as a thin wrapper so there is one
    pyramid loop, not two.
    """
    return compute_pyramid_scale_hints(
        shape,
        dim_labels,
        threshold=threshold,
        downscale_factor=downscale_factor,
        pixel_budget_cubic_root=pixel_budget_cubic_root,
    )[-1]


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
    item_size = np.dtype(dtype).itemsize
    chunk_bytes = int(np.prod(chunk_size)) * item_size

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
        chunk_bytes = int(np.prod(safe_size)) * item_size

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
    y_eligible = y_ax in eligible if y_ax else False
    x_eligible = x_ax in eligible if x_ax else False

    if y_eligible and x_eligible:
        return y_ax if shape[y_ax] >= shape[x_ax] else x_ax
    elif y_eligible:
        return y_ax
    elif x_eligible:
        return x_ax

    # Fallback: largest eligible axis
    return max(eligible, key=lambda ax: shape[ax])
