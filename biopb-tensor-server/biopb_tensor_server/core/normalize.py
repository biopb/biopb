"""Canonical axis-order normalization at the ``TensorAdapter`` seam (#596).

Adapters advertise ``dim_labels`` in whatever order their upstream reader emits.
This module turns that into a **wire guarantee** so no consumer has to re-derive
"which axis is Y/X/Z/S" for itself:

    Z, Y, X and S appear last, in that relative order; every other axis -- T, C,
    and any unrecognized label -- keeps its relative order ahead of them.

The rule lives in :func:`biopb_tensor_server.core.axes.canonical_permutation`;
this module is the *seam* that applies it. :func:`normalize_adapter` wraps a
source adapter whose axes are not already canonical in a
:class:`NormalizingAdapter`, and the registry (the single registration
chokepoint) calls it for every source. An already-canonical adapter -- which is
almost all of them, since ``bioio`` fixes ``TCZYXS`` and OME-TIFF / QPTIFF /
TIFF-sequence / ndtiff / DICOM are compliant by construction -- is returned
untouched, so the common path keeps its exact pre-#596 object identity and cost.

What the wrapper permutes, and what it deliberately does not
------------------------------------------------------------

**chunk_ids stay native and opaque.** They are minted by the wrapped adapter,
never rewritten here. That is what keeps this wrapper blind to the chunk codec:
a versioned id, a scaled id and a precompute level id all pass through
untouched. What gets permuted is the *client-visible* geometry
-- the descriptor (``dim_labels`` / ``shape`` / ``chunk_shape`` / ``slice_hint``
/ ``scale_hint`` / ``physical_scale`` / pyramid level shapes) and each
endpoint's logical ``bounds`` -- plus the pixels themselves on the way out. A
client therefore sees a coherent normalized view: normalized bounds, a
normalized descriptor, and a chunk whose axes match them.

Because the id is unchanged but the *bytes it now resolves to* are transposed,
cached segments written before this change would be served in the wrong order.
``CACHE_FILE_FORMAT_VERSION`` is bumped for exactly that reason (see
``cache.file_backend``); the transpose happens **inside** the cache's
``compute_fn``, so what lands in a segment is the served representation and the
localhost mmap fast path stays valid.

**Plans are delegated, not re-derived.** ``plan_flight_info`` / ``get_read_plan``
call the wrapped adapter and permute its answer, rather than inheriting the base
planner. That is what keeps the native-pyramid ``precompute`` routing working --
an override a primitives-only wrapper would silently bypass -- by letting it run
entirely in native order inside the delegate.

An order this server does not own is refused, not permuted
-----------------------------------------------------------

Permuting reads works only where the server owns the whole read path. Two seams
do not, and both **validate** instead, reporting through the shared
:func:`biopb_tensor_server.core.axes.noncanonical_order`:

**Writes** (#596 Decision 3). A writable source carries the uploader's own
declared order, with ``physical_scale`` and ``chunk_shape`` aligned to it;
silently permuting reads would desynchronize them from what ``put_chunk`` wrote.
``serving.upload_manager`` refuses the order at ``create_source``, so a
non-canonical writable source never exists and this wrapper's ``put_chunk``
branch is unreachable in practice.

**The remote proxy.** Its upstream owns the order in exactly the same sense: that
server mints the chunk_ids, plans the reads (biopb/biopb#295) and sizes the grid.
So ``adapters.remote_tensor`` opts out of wrapping (``_normalizable_axes =
False``) and refuses a non-canonical upstream at its read boundary instead.

Refusing rather than repairing is what keeps the guarantee *unconditional* while
leaving nothing stateful to go stale: the test is re-run against the descriptor
in hand on every open, so an upstream that upgrades is picked up immediately and
one that has not cannot be silently mis-served. It costs an upstream-first
upgrade ordering across a federation -- a legacy upstream's sources stay
catalogued and listed, but reads of them fail with a legible error naming the
order to fix -- which is the trade the write path already makes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import (
    DataSourceDescriptor,
    TensorDescriptor,
)
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.adapter_base import (
    SourceAdapter,
    TensorAdapter,
    pack_chunk_batch,
    unpack_chunk_array,
)
from biopb_tensor_server.core.axes import canonical_permutation
from biopb_tensor_server.core.chunk import (
    ChunkEndpoint,
    cache_key_for_chunk_id,
    is_scaled_chunk,
)
from biopb_tensor_server.core.errors import WriteNotSupportedError

if TYPE_CHECKING:
    from biopb.tensor.descriptor_pb2 import PyramidLevel, TensorReadOption

    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.core.adapter_base import TensorReadPlan
    from biopb_tensor_server.core.config import PyramidConfig, SourceConfig

logger = logging.getLogger(__name__)


# --- pure permutation helpers ------------------------------------------------


def _permute(values: Sequence[Any], perm: Tuple[int, ...]) -> List[Any]:
    """Reorder ``values`` (native order) into canonical order."""
    return [values[p] for p in perm]


def _invert(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """The inverse permutation: applying it undoes ``perm``.

    ``_permute(canonical_vector, _invert(perm))`` is the native-order vector, so
    the same helper serves both directions.
    """
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return tuple(inverse)


def _permute_repeated(field, perm: Tuple[int, ...]) -> None:
    """Permute a repeated proto field in place, iff its length matches the rank.

    A per-axis field that is absent (``chunk_shape`` is documented as optionally
    empty) or of some other rank is left alone rather than guessed at.
    """
    if len(field) == len(perm):
        field[:] = _permute(list(field), perm)


def _permute_bounds(bounds: ChunkBounds, perm: Tuple[int, ...]) -> ChunkBounds:
    """A copy of ``bounds`` with its axes reordered by ``perm``."""
    if len(bounds.start) != len(perm) or len(bounds.stop) != len(perm):
        return bounds
    return ChunkBounds(
        start=_permute(list(bounds.start), perm),
        stop=_permute(list(bounds.stop), perm),
    )


def _permute_descriptor(
    desc: TensorDescriptor, perm: Tuple[int, ...]
) -> TensorDescriptor:
    """A copy of ``desc`` with every per-axis field reordered by ``perm``.

    Covers the whole client-visible geometry, including the fields a read plan
    fills in (``slice_hint`` / ``scale_hint``) and the pyramid levels, each of
    which carries its own per-axis ``shape`` and ``scale_hint``.
    """
    out = TensorDescriptor()
    out.CopyFrom(desc)
    _permute_repeated(out.shape, perm)
    _permute_repeated(out.chunk_shape, perm)
    _permute_repeated(out.dim_labels, perm)
    _permute_repeated(out.scale_hint, perm)
    _permute_repeated(out.physical_scale, perm)
    _permute_repeated(out.physical_unit, perm)
    if out.HasField("slice_hint"):
        _permute_repeated(out.slice_hint.start, perm)
        _permute_repeated(out.slice_hint.stop, perm)
    for level in out.pyramid:
        _permute_repeated(level.shape, perm)
        _permute_repeated(level.scale_hint, perm)
    return out


def _normalize_descriptor(desc: TensorDescriptor) -> TensorDescriptor:
    """Normalize one descriptor by its **own** labels, or return it unchanged.

    Per-descriptor rather than per-source: a multi-tensor source (HCS fields, a
    multi-scene file) may hold tensors of differing rank and labelling, so each
    catalog row is classified on its own.
    """
    perm = canonical_permutation(desc.dim_labels, desc.shape)
    return desc if perm is None else _permute_descriptor(desc, perm)


# --- the wrapper -------------------------------------------------------------


class NormalizingAdapter(TensorAdapter):
    """A ``TensorAdapter`` that presents a wrapped adapter in canonical axis order.

    Delegates everything; permutes the per-axis geometry on the way out and the
    per-axis coordinates on the way in. Construct through
    :func:`normalize_adapter`, which returns the adapter unwrapped when there is
    nothing to normalize.

    Both adapter roles are covered because they nest (biopb/biopb#380): the
    source-level methods normalize each tensor descriptor by its own labels,
    while the tensor-level methods share one permutation derived from this
    object's own descriptor. ``get_tensor_adapter`` / ``get_level_adapter``
    hand back a view wrapping whatever they return, so a per-tensor view of a
    multi-tensor source gets the permutation *its* labels imply -- or, when they
    are already canonical, a pass-through (see :meth:`_view`).
    """

    def __init__(self, inner: SourceAdapter) -> None:
        self._inner = inner

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"NormalizingAdapter({self._inner!r})"

    def __getattr__(self, name: str) -> Any:
        """Forward otherwise-unknown attribute *reads* to the wrapped adapter.

        Guards ``_inner`` explicitly: reaching it through this method (before
        ``__init__`` has bound it, or after an unpickle) would recurse forever.

        A safety net for the private, format-specific attributes some adapter
        families read off each other (``_raw_ome_xml``, ``_active_reads``). It
        cannot shadow anything this class or its bases define -- ``__getattr__``
        only runs after normal lookup fails -- which is why the base's per-source
        fields are re-declared as delegating properties below rather than left to
        it. Writes are not forwarded: everything that pokes an adapter attribute
        does so before registration, on the raw object.
        """
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    # --- identity / per-source fields ---------------------------------------
    # SourceAdapter declares these as class attributes defaulting to None, so
    # they resolve on the wrapper before __getattr__ ever runs. Delegate them
    # explicitly or the wrapper would report None for every one.

    @property
    def source_id(self) -> str:
        return self._inner.source_id

    @property
    def _source_url(self) -> Optional[str]:
        return self._inner._source_url

    @property
    def _source_type(self) -> Optional[str]:
        return self._inner._source_type

    @property
    def _tensor_name(self) -> Optional[str]:
        return self._inner._tensor_name

    @property
    def _catalog_url(self) -> Optional[str]:
        return self._inner._catalog_url

    @_catalog_url.setter
    def _catalog_url(self, value: Optional[str]) -> None:
        self._inner._catalog_url = value

    @property
    def _content_version(self) -> Optional[bytes]:
        return self._inner._content_version

    @property
    def source_url(self) -> Optional[str]:
        return self._inner.source_url

    @property
    def source_type(self) -> Optional[str]:
        return self._inner.source_type

    @property
    def content_version(self) -> Optional[bytes]:
        return self._inner.content_version

    @property
    def array_id(self) -> str:
        return self._inner.array_id

    @property
    def capability_token(self) -> Optional[str]:
        return self._inner.capability_token

    @capability_token.setter
    def capability_token(self, value: Optional[str]) -> None:
        self._inner.capability_token = value

    # --- the permutation ------------------------------------------------------

    @property
    def perm(self) -> Optional[Tuple[int, ...]]:
        """This tensor's native -> canonical permutation, or None for identity.

        Derived from the wrapped adapter's own descriptor, every time, and
        deliberately **not** memoized. An adapter's advertised labels are not
        immutable -- a source can be undescribable now and describable later, or
        describable differently later -- and a cached permutation cannot notice.
        Freezing one is how a mirror of a legacy upstream kept transposing after
        that upstream upgraded; the proxy no longer takes this path at all (it
        refuses instead, see the module docstring), so nothing today can go
        stale, and re-deriving keeps it that way for whatever is added next.

        Cheap enough to prefer that: ``canonical_permutation`` is O(ndim) over a
        descriptor the wrapper's caller is generally building anyway, and callers
        that need it twice pass it down rather than asking twice.
        """
        try:
            desc = self._inner.get_tensor_descriptor()
        except Exception:
            return None
        return canonical_permutation(desc.dim_labels, desc.shape)

    # --- source role ----------------------------------------------------------

    @classmethod
    def create_from_config(
        cls, source: SourceConfig, credentials_config: Optional[Any] = None
    ) -> SourceAdapter:
        raise NotImplementedError(
            "NormalizingAdapter wraps an existing adapter; it is applied by "
            "normalize_adapter() at registration, not built from config"
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [_normalize_descriptor(d) for d in self._inner.list_tensor_descriptors()]

    def get_metadata(self) -> dict:
        # Format metadata (OME-XML, .zattrs) is passed through verbatim: it
        # describes the file, and rewriting a foreign schema's axis order is a
        # different, much larger contract than the descriptor's. The descriptor
        # is what carries the guarantee.
        return self._inner.get_metadata()

    def get_source_descriptor(self) -> DataSourceDescriptor:
        desc = self._inner.get_source_descriptor()
        normalized = [_normalize_descriptor(t) for t in desc.tensors]
        del desc.tensors[:]
        desc.tensors.extend(normalized)
        return desc

    def resolve(self) -> DataSourceDescriptor:
        desc = self._inner.resolve()
        normalized = [_normalize_descriptor(t) for t in desc.tensors]
        del desc.tensors[:]
        desc.tensors.extend(normalized)
        return desc

    def is_resident(self) -> bool:
        return self._inner.is_resident()

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> TensorAdapter:
        return self._view(self._inner.get_tensor_adapter(tensor_id))

    def _view(self, inner: SourceAdapter) -> SourceAdapter:
        """Wrap a tensor-level view of this source, deciding nothing.

        These two lookups (:meth:`get_tensor_adapter`, :meth:`get_level_adapter`)
        sit on the ``do_get`` path -- the server resolves a chunk's adapter
        through them on *every* read -- so running the source-level classifier
        here cost a full ``list_tensor_descriptors`` per chunk.

        Nothing needs to be decided. Whether a view actually permutes is
        :attr:`perm`'s business, re-derived per access from that view's own
        descriptor, and a wrapper whose ``perm`` is ``None`` is a pass-through:
        every method early-returns to the delegate. So wrap unconditionally and
        let a canonical sibling tensor of a mixed source carry a transparent
        wrapper, rather than paying to find out it did not need one.

        This is also why no cache is needed and none is kept: caching the
        decision would reintroduce exactly the staleness :attr:`perm` was
        de-memoized to avoid -- a cached "does not need wrapping" cannot notice
        its source's labels changed -- to save an object allocation.
        """
        return (
            inner
            if isinstance(inner, NormalizingAdapter)
            else NormalizingAdapter(inner)
        )

    def put_chunk(
        self,
        bounds: ChunkBounds,
        data: pa.Array | pa.ChunkedArray,
        expected_shape: Tuple[int, ...],
        dtype: Any,
    ) -> None:
        if self.perm is None:
            self._inner.put_chunk(bounds, data, expected_shape, dtype)
            return
        raise WriteNotSupportedError(
            f"source {self.source_id!r} declares a non-canonical axis order "
            f"and is therefore read-only; upload with axes in canonical "
            f"[..., Z, Y, X, S] order instead (biopb/biopb#596)"
        )

    def close(self) -> None:
        self._inner.close()

    def release_registration_cache(self) -> None:
        # Declared on SourceAdapter, so it resolves on the wrapper and never
        # reaches __getattr__ -- delegate explicitly or the inner adapter would
        # never hear about it.
        self._inner.release_registration_cache()

    # --- tensor role ----------------------------------------------------------

    def get_tensor_descriptor(self) -> TensorDescriptor:
        desc = self._inner.get_tensor_descriptor()
        perm = self.perm
        return desc if perm is None else _permute_descriptor(desc, perm)

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read ``bounds`` -- given in canonical order -- as a canonical array."""
        perm = self.perm
        if perm is None:
            return self._inner.get_data(bounds)
        native_bounds = _permute_bounds(bounds, _invert(perm))
        return self._inner.get_data(native_bounds).transpose(perm)

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        # Reached only by a direct caller: the plan path gets its scale from the
        # delegate's own _fill_physical_scale and is permuted with the rest of
        # the descriptor. A source with no tensor role has no scale to report.
        inner_scale = getattr(self._inner, "_physical_scale", None)
        if inner_scale is None:
            return None
        phys = inner_scale()
        perm = self.perm
        if phys is None or perm is None:
            return phys
        scale, unit = phys
        if len(scale) != len(perm) or len(unit) != len(perm):
            return phys
        return _permute(scale, perm), _permute(unit, perm)

    def get_tensor_metadata(self) -> Optional[dict]:
        return self._inner.get_tensor_metadata()

    def get_native_pyramid_levels(self) -> Optional[List[PyramidLevel]]:
        levels = self._inner.get_native_pyramid_levels()
        perm = self.perm
        if levels is None or perm is None:
            return levels
        out = []
        for level in levels:
            copy = type(level)()
            copy.CopyFrom(level)
            _permute_repeated(copy.shape, perm)
            _permute_repeated(copy.scale_hint, perm)
            out.append(copy)
        return out

    def has_native_pyramid(self) -> bool:
        return self._inner.has_native_pyramid()

    def get_level_adapter(self, path: str) -> Optional[TensorAdapter]:
        level = self._inner.get_level_adapter(path)
        return None if level is None else self._view(level)

    # --- planning (delegate, then permute) -----------------------------------

    def _permute_plan(
        self, plan: TensorReadPlan, perm: Optional[Tuple[int, ...]]
    ) -> TensorReadPlan:
        """Rewrite a native-order read plan into canonical order.

        The descriptor's per-axis fields and each endpoint's logical ``bounds``
        are permuted; ``chunk_id`` is carried verbatim, since it is the wrapped
        adapter's own opaque key and ``resolve_chunk_data`` hands it straight
        back.

        Takes ``perm`` rather than reading :attr:`perm` again: its caller already
        derived one to build the native request, and the two must be the same
        permutation or the round trip does not close.
        """
        if perm is None:
            return plan
        plan.descriptor = _permute_descriptor(plan.descriptor, perm)
        plan.chunk_endpoints = [
            ChunkEndpoint(chunk_id=ce.chunk_id, bounds=_permute_bounds(ce.bounds, perm))
            for ce in plan.chunk_endpoints
        ]
        return plan

    def get_read_plan(
        self,
        request_desc: TensorDescriptor,
        max_read_block_bytes: Optional[int] = None,
    ) -> TensorReadPlan:
        perm = self.perm
        if perm is None:
            return self._inner.get_read_plan(request_desc, max_read_block_bytes)
        native_request = _permute_descriptor(request_desc, _invert(perm))
        return self._permute_plan(
            self._inner.get_read_plan(native_request, max_read_block_bytes), perm
        )

    def plan_flight_info(
        self, read_opt: TensorReadOption, pyramid_config: PyramidConfig
    ) -> TensorReadPlan:
        perm = self.perm
        if perm is None:
            return self._inner.plan_flight_info(read_opt, pyramid_config)
        # The client's hints arrive in canonical order; the delegate plans in
        # native order, so they have to go back the other way first.
        inverse = _invert(perm)
        native_opt = type(read_opt)()
        native_opt.CopyFrom(read_opt)
        _permute_repeated(native_opt.scale_hint, inverse)
        if native_opt.HasField("slice_hint"):
            _permute_repeated(native_opt.slice_hint.start, inverse)
            _permute_repeated(native_opt.slice_hint.stop, inverse)
        return self._permute_plan(
            self._inner.plan_flight_info(native_opt, pyramid_config), perm
        )

    # --- chunk serving --------------------------------------------------------

    def resolve_chunk_data(
        self,
        chunk_id: bytes,
        cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Serve one chunk, transposed into canonical order **before** caching.

        The delegate is called with no cache manager and this wrapper owns the
        cache interaction instead, under the same key and the same gate the
        delegate would have used -- literally the base's gate, since the only
        adapter that mints a different kind of id (the remote proxy's envelope)
        is never wrapped. That ordering is the point: a cached segment must hold
        what the client is served, because the localhost fast path hands the
        client that segment's bytes directly, with the server no longer in the
        loop to transpose them. Existing segments predate the transpose, which is
        what ``CACHE_FILE_FORMAT_VERSION`` is bumped for.
        """
        from biopb_tensor_server.cache import ArrowFileBackend

        perm = self.perm
        if perm is None:
            return self._inner.resolve_chunk_data(chunk_id, cache_manager)

        # Under a real permutation composing does not apply, and nothing has to
        # suppress it: the delegate is handed no cache manager below -- this
        # wrapper owns the caching so that what lands in a segment is what the
        # client is served -- and composing without one is a no-op by
        # construction. It could not work anyway: the chunk_ids on this side are
        # in canonical axis order while the delegate's grid is in its own, so
        # ids minted here would name bounds the delegate never serves. Composing
        # a transposed source means minting them on the delegate's side, which
        # is its own change.

        should_cache = cache_manager is not None and (
            is_scaled_chunk(chunk_id)
            or isinstance(cache_manager.backend, ArrowFileBackend)
        )

        def compute_fn():
            batch = self._inner.resolve_chunk_data(chunk_id, None)
            arr = unpack_chunk_array(batch).transpose(perm)
            return pack_chunk_batch(arr), arr.nbytes

        if should_cache:
            cache_key = cache_key_for_chunk_id(chunk_id)
            entry = cache_manager.get_or_acquire(cache_key, compute_fn)
            data = entry.data
            cache_manager.release(cache_key)
            return data
        data, _ = compute_fn()
        return data


def _noncanonical_tensors(adapter: SourceAdapter) -> List[Tuple[str, list, list]]:
    """``(array_id, native labels, canonical labels)`` per out-of-order tensor.

    The classification behind both :func:`needs_normalization` and the log line
    in :func:`normalize_adapter`, so deciding and reporting share one pass over
    ``list_tensor_descriptors`` rather than listing the source twice.
    """
    out = []
    for d in adapter.list_tensor_descriptors():
        perm = canonical_permutation(d.dim_labels, d.shape)
        if perm is not None:
            labels = list(d.dim_labels)
            out.append((d.array_id, labels, [labels[p] for p in perm]))
    return out


# Sources already reported as reordered, as (source_id, array_id, native order).
# normalize_adapter is re-entered per ``get_tensor_adapter`` -- i.e. per chunk
# read for a wrapped source -- so the INFO below would otherwise repeat on the
# hot path. Keyed by the native order too, so a source whose labels genuinely
# change is reported again rather than silently suppressed. Bounded by the number
# of distinct non-canonical tensors, which is small by construction.
_reported: set = set()


def needs_normalization(adapter: SourceAdapter) -> bool:
    """Whether ``adapter`` must be wrapped to satisfy the canonical-order contract.

    True only when a tensor it advertises is *demonstrably* non-canonical. A
    source advertising no tensors at all -- an unresolved cloud source, a remote
    proxy whose upstream is momentarily down -- is left alone: wrapping on the
    mere absence of evidence would replace the adapter object for every such
    source, and its concrete type is something callers legitimately test and
    switch on.

    The gap that leaves is closed at a closer seam: ``UnresolvedSourceAdapter``
    normalizes the adapter it builds at resolution time, when the labels finally
    exist. Nothing else advertises tensors late -- the remote proxy, which does,
    is not a candidate for wrapping at all (``_normalizable_axes``), so an
    upstream that is down at registration cannot leave a source stranded in a
    stale order here.
    """
    return bool(_noncanonical_tensors(adapter))


def normalize_adapter(adapter: Optional[SourceAdapter]) -> Optional[SourceAdapter]:
    """Return ``adapter`` presented in canonical axis order.

    The seam itself: already-canonical adapters are returned **unchanged**, so
    the overwhelmingly common case keeps its object identity and adds no
    delegation to the hot path. Anything that goes wrong deciding -- a duck-typed
    test double, an adapter that raises while listing -- also returns the adapter
    unchanged, because failing to normalize is a degraded contract while
    misfiring is a correctness bug.

    An adapter whose order is owned upstream (``_normalizable_axes = False``) is
    returned unchanged too, and *not* because it is compliant: it enforces the
    contract itself, by refusing a non-canonical order at its read boundary. See
    the module docstring.
    """
    if adapter is None or isinstance(adapter, NormalizingAdapter):
        return adapter
    if not isinstance(adapter, SourceAdapter):
        return adapter  # duck-typed double: nothing to guarantee, nothing to wrap
    if not adapter._normalizable_axes:
        return adapter  # order owned upstream: validated there, never permuted
    try:
        offenders = _noncanonical_tensors(adapter)
    except Exception:
        logger.debug(
            "axis normalization: could not classify source %r; leaving it as-is",
            getattr(adapter, "source_id", "?"),
            exc_info=True,
        )
        return adapter
    if not offenders:
        return adapter
    # Reordering a source's axes is a visible behavior change -- it is what a
    # client sees, and it costs the zero-copy read for that source -- so say so
    # once, at INFO, naming both orders. Without this the only evidence that a
    # source is being transposed is the transposed data itself.
    source_id = getattr(adapter, "source_id", "?")
    for array_id, native, canonical in offenders:
        key = (source_id, array_id, tuple(native))
        if key in _reported:
            continue
        _reported.add(key)
        logger.info(
            "axis normalization: serving %s as %s (stored as %s)",
            array_id,
            canonical,
            native,
        )
    return NormalizingAdapter(adapter)
