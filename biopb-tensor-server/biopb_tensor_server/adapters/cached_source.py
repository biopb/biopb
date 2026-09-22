"""An uploaded tensor whose chunks are served exactly as they were sent.

One adapter per uploaded tensor, fusing the source and tensor roles the way
``OmeZarrAdapter`` does. It holds the tensor's metadata (shape, dtype, grid,
axes) and a record of which bounds have arrived; the bytes themselves go
wherever the subclass puts them.

**This class puts them in the chunk cache**, which makes it volatile: the cache
entry is the upload's only copy, an eviction is a loss and not a gap, and
nothing survives the process. That is what the embedded result cache wants --
``biopb-image-base`` constructs one directly for a fast-return servicer's
output, which is read once and gone. A ``cache://`` uploaded field wants the
opposite and subclasses it
(``adapters.cache_member.CacheMember``), keeping the same batches in segments
under its own directory.

Upload progress and disposal live on the adapter (``adapters._writable``); a
discarded one stays registered as a tombstone until the sweep reclaims it.
Chunks are stored under the chunk_id a read plan mints, so a client's echoed
id resolves without a translation step.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._writable import WritableSource
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.adapter_base import TensorAdapter, catalog_entry
from biopb_tensor_server.core.chunk import (
    content_version_of,
    decode_chunk_id,
    is_scaled_chunk,
    mint_chunk_id,
)
from biopb_tensor_server.core.chunk_batch import CHUNK_WIRE_SCHEMA, unpack_chunk_array
from biopb_tensor_server.core.errors import StaleChunkError

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig

logger = logging.getLogger(__name__)


class CachedSourceAdapter(WritableSource, TensorAdapter):
    """An uploaded tensor kept in the chunk cache.

    One instance per uploaded tensor, registered in the server's registry. It
    begins an upload in the constructor, since every instance is one -- except
    a member adopted from an earlier life, which is published already
    (``track_upload``).

    The bytes go through :meth:`_store_chunk_batch` and come back through
    :meth:`_read_chunk_batch`, the two methods a durable format overrides
    (``adapters.cache_member``). Everything else -- the record of what arrived,
    the gap-versus-loss rule, the region assembly -- is the same either way.

    Arbitrary chunk bounds are accepted here; no other format takes a write off
    its own grid.
    """

    # Process-monotonic generation clock for the content_version of every
    # upload (see next_content_version). One per process, not per server: a
    # persisted file cache is shared by whatever serves it.
    _generation_lock = threading.Lock()
    _last_generation: int = 0

    # There is no backend to re-read: the cache entry is the upload's only copy,
    # so a read times a memcpy out of the cache being classified. Measured, it
    # would clock at DRAM speed and be marked "cheap" -- the class eviction takes
    # first -- for data that cannot be rebuilt at any price.
    _decode_time_is_rebuild_cost = False

    @classmethod
    def create_from_config(
        cls, source: SourceConfig, credentials_config: Optional[Any] = None
    ) -> CachedSourceAdapter:
        """Cache-backed sources are not created from config.

        Raises NotImplementedError - use direct instantiation via DoPut.
        """
        raise NotImplementedError(
            "CachedSourceAdapter is created via DoPut, not config"
        )

    @staticmethod
    def upload_source_id(name: str) -> str:
        """The ``source_id`` a cache-backed source of this *name* lands on.

        Deterministic for a name, minted for an empty one. No longer reachable
        over the wire -- ``add_tensor`` adds a ``cache://`` tensor to a source
        that already has an id -- but still
        how an in-process caller names one it registers itself, which is what
        ``biopb-image-base``'s embedded cache does for a servicer's result.
        """
        if name:
            return f"cache_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        return f"cache_{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"

    @classmethod
    def next_content_version(cls) -> bytes:
        """A process-monotonic generation token for a biopb-written cache: source.

        cache: sources have deterministic ids (``cache:<name>`` -> a fixed
        ``source_id``), so an upload after a restart reuses the id of one the
        persisted file cache may still hold chunks for. Without a fresh
        namespace, ``CacheManager.put`` finds the prior upload's chunk already
        present and declines to overwrite it -- serving stale data
        (biopb/biopb#178). Folding a distinct token into each upload's chunk_ids
        sidesteps that. The same token is what makes reclaiming a discarded
        name safe within one server lifetime: the re-created source's chunk
        ids never collide with the tombstone's, still sitting in the cache.

        Wall-clock ns keeps the token distinct across a restart, where a persisted
        file cache may still hold the prior upload's chunks; ``max(..., last + 1)``
        keeps it strictly increasing even if the clock steps backwards. This is the
        cache: analogue of the file adapters' stat signature -- those sources have no
        file to stat, so biopb (the writer) supplies the version itself.
        """
        with cls._generation_lock:
            gen = max(time.time_ns(), cls._last_generation + 1)
            cls._last_generation = gen
        return f"gen:{gen}".encode()

    def __init__(
        self,
        source_id: str,
        shape: List[int],
        dtype: str,
        chunk_shape: List[int],
        dim_labels: Optional[List[str]] = None,
        ome_metadata: Optional[dict] = None,
        physical_scale: Optional[List[float]] = None,
        physical_unit: Optional[List[str]] = None,
        content_version: Optional[bytes] = None,
        track_upload: bool = True,
    ):
        """Initialize cache-backed source adapter.

        Args:
            source_id: Unique source identifier (e.g., "cache_abc123")
            shape: Array shape
            dtype: Data type string (numpy format)
            chunk_shape: Nominal chunk size per dimension
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict
            physical_scale: Optional per-dimension physical pixel size, aligned
                1:1 with ``dim_labels`` (as the uploader sent it).
            physical_unit: Optional per-dimension unit string for
                ``physical_scale``, aligned 1:1 with ``dim_labels``.
            content_version: Optional per-upload generation token (biopb/biopb#178).
                cache: sources have deterministic ids, so an upload after a restart
                reuses the id; wrapping every written chunk_id with this token gives
                the new upload a fresh cache namespace instead of colliding with the
                prior upload's persisted chunks (which ``CacheManager.put`` would
                decline to overwrite, serving stale data). None leaves the source
                unversioned (legacy bytes).
            track_upload: Whether this instance begins an upload. False for a
                member adopted from an earlier life (``adapters.cache_member``):
                its record died with the process that filled it, it is
                published already, and nothing may write to it again.
        """
        self.source_id = source_id
        # Optional per-source capability token. When set, reading this source
        # takes either it or the server-wide token (see
        # TensorFlightServer._authorize_read). None = no per-source gate.
        self._capability_token: Optional[str] = None
        self._shape = tuple(shape)
        self._dtype = dtype
        self._chunk_shape = tuple(chunk_shape)
        self._dim_labels = dim_labels or [f"dim{i}" for i in range(len(shape))]
        self._ome_metadata = ome_metadata or {}
        # Client-provided physical calibration, echoed verbatim on the wire. The
        # uploader already aligned these to dim_labels, so unlike the file
        # adapters there is nothing to parse or canonicalise -- storing and
        # surfacing them is exact (issue #272).
        self._physical_scale_vec = list(physical_scale) if physical_scale else []
        self._physical_unit_vec = list(physical_unit) if physical_unit else []

        # The record of what has been uploaded: chunk_id -> the bounds it
        # covers. It is what tells a gap (never uploaded -- reads as zeros)
        # from a loss (uploaded, then evicted -- raises), and the two must not
        # look alike to a reader.
        self._written_chunks: Dict[bytes, ChunkBounds] = {}
        # Whether any write has landed off the declared grid. ``cache:`` is the
        # one kind that allows it, and almost nothing does; while it stays
        # False, a grid-aligned read that misses the record is a gap outright,
        # with no scan of the record needed to prove it (:meth:`get_data`).
        self._off_grid_writes = False

        # Per-upload generation token folded into every written chunk_id so an
        # upload reusing a (deterministic) source_id after a restart lands in a
        # fresh cache namespace (biopb/biopb#178). The base read plan folds the
        # same token into minted endpoints, so reads and writes agree.
        self._content_version = content_version

        # Required fields for the adapter interface
        self._source_url = f"cache://{source_id}"
        self._source_type = "cache"

        if track_upload:
            self.begin_upload(self._shape, self._chunk_shape)

    def adopt_uploaded(self, bounds: Iterable[Optional[ChunkBounds]]) -> None:
        """Record chunks a store already holds, as *this* build's chunk ids.

        The read path is keyed by chunk id and a durable store is keyed by
        bounds (``adapters.cache_member``), so the ids are minted here, once,
        at registration: a chunk id also carries the serving-semantics epoch,
        which moves on an upgrade that changes what the bytes mean, while the
        bytes on disk do not.
        """
        for chunk_bounds in bounds:
            if chunk_bounds is None:
                continue  # a key this build did not write; not servable
            chunk_id = mint_chunk_id(
                self.array_id, chunk_bounds, content_version=self.content_version
            )
            self._record_write(chunk_id, chunk_bounds)

    def _record_write(self, chunk_id: bytes, bounds: ChunkBounds) -> None:
        """Note that *chunk_id* covers *bounds*, and whether that left the grid.

        Shared by adoption (a store's own boot-time replay) and a live upload
        (:meth:`write_chunk_arrow`) -- both are "this id now exists", differing
        only in how the id was minted.
        """
        self._written_chunks[chunk_id] = bounds
        if not self._off_grid_writes and not self._on_grid(bounds):
            self._off_grid_writes = True

    def upload_response(self, desc: TensorDescriptor) -> TensorDescriptor:
        """Echo the uploader's physical calibration on the response.

        A cache source stores it AND re-serves it verbatim on the read path, so
        what is advertised here is exactly what a later read reproduces
        (issue #272). The zarr kind persists scale via the ``.zattrs`` or drops
        it, so it does not echo.
        """
        response = super().upload_response(desc)
        if self._physical_scale_vec:
            response.physical_scale.extend(self._physical_scale_vec)
        if self._physical_unit_vec:
            response.physical_unit.extend(self._physical_unit_vec)
        return response

    def get_transfer_chunk_size(self) -> Tuple[int, ...]:
        """The write grid, verbatim -- not re-split at the wire bound.

        The base clamps a declared grid to ``MAX_ARROW_BATCH_BYTES`` so an
        oversized chunk is fetched in pieces. Here the pieces do not exist as
        chunks: a plan on any other grid asks for bounds no upload stored, and
        every one of them is then reassembled by a scan of the whole record
        (:meth:`get_data`) -- correct, and quadratic in the chunk count. An
        uploader that wrote one 67 MB chunk -- a whole result in one DoPut is
        the ordinary runtime case -- is served that chunk whole; the cache
        keeps an oversized entry in memory rather than on disk, and DoGet
        streams it as it was put. A consumer that replans a handle (every
        fast-return consumer, since its handle carries no endpoints) meets
        this path, where the producer's own embedded endpoints used to hide it.
        """
        shape = self._shape
        return tuple(
            min(max(1, int(chunk)), int(dim))
            for chunk, dim in zip(self._chunk_shape, shape, strict=True)
        )

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this cache source.

        ``chunk_shape`` is the uploader's write grid, verbatim and unnegotiable.
        This source has no backend: its chunks are its only copy, and a read
        planned on any other grid names bounds no upload ever stored. While
        such a read raised, that was loud -- the server re-sizing every
        adapter's grid planned a (1,1,1,1024,1024) upload at
        (1,1,4,1024,1024), and none of the 64 chunk_ids existed
        (biopb/biopb#809). It is answered now, off the record rather than off a
        chunk_id, which makes a re-shaped grid quiet and slow instead of loud.
        Nothing may re-shape this value.
        """
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self._dim_labels,
            shape=list(self._shape),
            chunk_shape=list(self._chunk_shape),
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Cache sources are single-tensor."""
        return [catalog_entry(self.get_tensor_descriptor())]

    def get_metadata(self) -> dict:
        """Return OME metadata."""
        return self._ome_metadata

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Echo the uploader's physical calibration onto the wire descriptor.

        Cache-backed sources carry whatever ``physical_scale`` / ``physical_unit``
        the DoPut request supplied, already aligned 1:1 with ``dim_labels`` (the
        client sent it that way), so there is nothing to parse or map -- return
        the stored vectors verbatim. Returns ``None`` when the upload carried no
        calibration, so the base clears the fields rather than advertising empty
        vectors. See ``TensorAdapter._physical_scale``.
        """
        if not self._physical_scale_vec or not self._physical_unit_vec:
            return None
        return list(self._physical_scale_vec), list(self._physical_unit_vec)

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """The region at *bounds*, assembled from the chunks that were uploaded.

        There is no backend: an upload's cache entries are its only copy, so
        this reads them back. A region nobody uploaded reads as **zeros**,
        which is what makes a published-but-unfinished source servable -- the
        writer of a label set that covers a tenth of the frame sends a tenth of
        the chunks, and the rest is background. It is the read gate
        (``WritableSource.check_readable``), not this, that keeps a hole from
        being read as data before the producer has said the holes are holes.

        A chunk that *was* uploaded and is no longer in the cache raises: that
        is loss, not a hole, and the two must not look alike.

        The exact-chunk case is O(1) and is every read a plan makes -- the
        transfer grid is the uploader's own write grid and nothing may re-shape
        it (biopb/biopb#809). A miss is a gap outright, still without touching
        the record, as long as both ends sat on that grid: a grid-aligned
        request that no grid-aligned write covered is empty by construction.
        The O(n) scan is what answers the two cases where that reasoning does
        not hold -- a request off the grid, or a source that has taken a write
        off it (only the ``cache:`` kind allows one) -- and it is what keeps a
        misplanned read returning the pixels rather than a blank frame.
        """
        # Validate bounds via the base TensorAdapter.get_data contract.
        super().get_data(bounds)

        exact = mint_chunk_id(
            self.array_id, bounds, content_version=self.content_version
        )
        if exact in self._written_chunks:
            return self._uploaded_block(exact, bounds)

        start = [int(v) for v in bounds.start]
        stop = [int(v) for v in bounds.stop]
        out = np.zeros(
            tuple(hi - lo for lo, hi in zip(start, stop, strict=True)),
            dtype=np.dtype(self._dtype),
        )
        if not self._off_grid_writes and self._on_grid(bounds):
            return out
        for chunk_id, written in list(self._written_chunks.items()):
            overlap = _overlap_slices(start, stop, written)
            if overlap is None:
                continue
            dst, src = overlap
            out[dst] = self._uploaded_block(chunk_id, written)[src]
        return out

    def _uploaded_block(self, chunk_id: bytes, bounds: ChunkBounds) -> np.ndarray:
        """One uploaded chunk, read back as an array.

        Copied, not viewed: the caller keeps it past the release, and for the
        assembly above it is a source to blit from rather than the answer.
        """
        return unpack_chunk_array(self._read_chunk_batch(chunk_id, bounds))

    # -- where this kind's bytes live ------------------------------------------
    #
    # The chunk cache, for this class: an upload's cache entry is its only copy,
    # which is what makes it volatile and what an eviction costs. A ``cache://``
    # member overrides the two with a segment store under its own directory
    # (``adapters.cache_member``), so the same read and write paths above serve
    # a tensor that outlives the process.

    def _store_chunk_batch(
        self,
        chunk_id: bytes,
        bounds: ChunkBounds,
        batch: pa.RecordBatch,
        size_bytes: int,
    ) -> None:
        """Store one uploaded chunk's batch under *chunk_id*."""
        self._cache().put(chunk_id, batch, size_bytes)

    def _read_chunk_batch(self, chunk_id: bytes, bounds: ChunkBounds) -> pa.RecordBatch:
        """The batch stored for *chunk_id*; raises if it was stored and is gone.

        Never called for a chunk the record does not hold, so a miss here is a
        loss and not a gap -- the two must not look alike to a reader.
        """
        cache_manager = self._cache()
        entry = cache_manager.get_or_acquire(
            chunk_id, lambda: _raise_evicted(self.array_id, bounds)
        )
        try:
            return entry.data
        finally:
            cache_manager.release(chunk_id)

    @staticmethod
    def _cache() -> CacheManager:
        """The process's cache manager; this kind has nowhere else to put bytes."""
        cache_manager = CacheManager.get_instance()
        if cache_manager is None:
            raise RuntimeError("Cache not initialized")
        return cache_manager

    def write_chunk(self, bounds: ChunkBounds, data: np.ndarray) -> None:
        """Write a NumPy chunk: the in-process producer's entry to ``put_chunk``.

        For cache-backed sources, arbitrary bounds allowed. Goes through
        ``put_chunk`` so the write is counted, and refused once the upload is
        over, the same as one arriving over the wire.

        Args:
            bounds: Chunk start/stop coordinates
            data: Numpy array with chunk data (any shape matching bounds)
        """
        # Pass the flat element values (a primitive Arrow array), NOT a list<T>
        # wrapper -- write_chunk_arrow stores the raw value buffer directly.
        self.put_chunk(bounds, pa.array(data.ravel()), data.shape, data.dtype)

    def _store_chunk(self, bounds, data, expected_shape, dtype) -> None:
        """Arbitrary-bounds write: cache-backed sources accept any bounds.

        Delegates straight to ``write_chunk_arrow`` (no NumPy round-trip -- the
        Arrow value buffer is stored as the unified binary chunk schema).
        """
        self.write_chunk_arrow(bounds, data, expected_shape, dtype)

    def _on_grid(self, bounds: ChunkBounds) -> bool:
        """Whether *bounds* is exactly one cell of the declared transfer grid.

        The grid a read plan mints chunk_ids on, so an on-grid write is one a
        later read asks for by that same id.
        """
        grid = self.get_transfer_chunk_size()
        return all(
            lo % size == 0 and hi == min(lo + size, dim)
            for lo, hi, size, dim in zip(
                bounds.start, bounds.stop, grid, self._shape, strict=True
            )
        )

    def write_chunk_arrow(
        self,
        bounds: ChunkBounds,
        data: pa.Array | pa.ChunkedArray,
        logical_shape: Tuple[int, ...] | List[int],
        dtype: np.dtype | str,
    ) -> None:
        """Write chunk data to cache without converting Arrow payloads through NumPy.

        Args:
            bounds: Chunk start/stop coordinates
            data: The chunk's flattened element *values* as a primitive Arrow
                array (e.g. ``int16``) -- NOT a ``list<T>`` wrapper. Its raw value
                buffer is stored directly as the unified binary chunk schema
                (biopb/biopb#293), the same schema ``resolve_chunk_data`` serves,
                so uploaded chunks read back byte-identically.
            logical_shape: Logical chunk shape matching bounds
            dtype: NumPy dtype or dtype string for the chunk data
        """
        # Stored under the id the base read plan mints, so a client's echoed
        # chunk_id resolves here and a prior upload's chunks are never served.
        # Through mint_chunk_id because the read-side probe
        # (core.cache_source.chunk_cache_keys) mints the same way, and a key
        # that differs by one byte is a probe that never hits.
        chunk_id = mint_chunk_id(
            self.array_id, bounds, content_version=self.content_version
        )

        if isinstance(data, pa.ChunkedArray):
            data = data.combine_chunks()
        if pa.types.is_list(data.type):
            # A list<T> wrapper's buffer[1] is offsets, not the value bytes --
            # storing it would silently corrupt the chunk. The binary schema takes
            # the flat values directly (biopb/biopb#293), so reject the wrapper.
            raise TypeError(
                "write_chunk_arrow expects the flat element values (a primitive "
                "array), not a list<T> wrapper"
            )

        logical_shape = list(logical_shape)
        dtype_str = np.dtype(dtype).str
        values_buf = data.buffers()[1]  # primitive array buffers: [validity, data]
        size_bytes = values_buf.size
        offsets = pa.py_buffer(np.array([0, size_bytes], dtype=np.int32))
        data_col = pa.Array.from_buffers(pa.binary(), 1, [None, offsets, values_buf])

        batch = pa.RecordBatch.from_arrays(
            [data_col, pa.array([logical_shape]), pa.array([dtype_str])],
            schema=CHUNK_WIRE_SCHEMA,
        )

        self._store_chunk_batch(chunk_id, bounds, batch, size_bytes)

        # Recorded even when the store declined (the chunk is already there
        # under this id): resolve_chunk_data gates reads on this map, so a
        # re-upload of an existing chunk must still be readable.
        self._record_write(chunk_id, bounds)

        logger.debug(
            f"write_chunk_arrow: stored {size_bytes} bytes at bounds "
            f"{list(bounds.start)} to {list(bounds.stop)}"
        )

    def resolve_chunk_data(
        self,
        chunk_id: bytes,
        cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Serve an uploaded chunk, or an answer assembled from uploaded chunks.

        An **uploaded** unscaled chunk is handed back verbatim: this source
        has no backend, so what the upload stored *is* the chunk and re-packing
        it would copy it for nothing.

        Everything else goes to the base over :meth:`get_data`, which zero-fills
        what was never uploaded -- an unscaled chunk in a gap, and a scaled
        chunk whose extent is only partly covered. A scaled read that *is* fully
        covered is assembled from the cached full-resolution chunks
        (``core.cache_source``), which is what keeps every rung of an uploaded
        source's pyramid viewable (biopb/biopb#265).

        Caching an answer built over zeros is safe **because READY seals**:
        nothing readable is still writable, so no write can fill a gap an
        answer already stored under its chunk_id stands on. That was not true
        while READY and writable were different states, and it is the reason
        they are one.
        """
        # The gate first, and before the version check: a tombstone owes a
        # reader its reason whatever ticket that reader is holding.
        self.check_readable()

        # A chunk_id from another namespace is not a gap, and this is the one
        # source that would otherwise read it as one: every id that misses the
        # record is answered with zeros, so an id minted against a different
        # ``content_version`` would come back as a blank chunk instead of the
        # stale-ticket error every other source gives it. The base's
        # ``check_chunk_version`` lets a legacy unversioned id through by
        # design, which is exactly the case that hurts here.
        if content_version_of(chunk_id) != self.content_version:
            raise StaleChunkError(
                f"chunk_id for {self.array_id!r} was minted against an upload "
                f"this source no longer serves; re-request the read plan "
                f"(GetFlightInfo) rather than retrying this chunk_id.",
                reason="stale_content_version",
            )

        if not is_scaled_chunk(chunk_id) and chunk_id in self._written_chunks:
            _, bounds = decode_chunk_id(chunk_id)
            return self._read_chunk_batch(chunk_id, bounds)

        # Only the base needs one: it caches what it assembles, and an uploaded
        # chunk above was served out of this kind's own store.
        if cache_manager is None:
            raise flight.FlightServerError(
                f"CacheManager required for cache-backed source {self.array_id}"
            )
        return super().resolve_chunk_data(chunk_id, cache_manager)


def _overlap_slices(
    start: List[int], stop: List[int], written: ChunkBounds
) -> Optional[Tuple[Tuple[slice, ...], Tuple[slice, ...]]]:
    """``(destination, source)`` slices for the part of *written* inside the
    request, or None where the two do not meet."""
    dst: List[slice] = []
    src: List[slice] = []
    for lo, hi, c_lo, c_hi in zip(
        start, stop, written.start, written.stop, strict=True
    ):
        begin, end = max(lo, int(c_lo)), min(hi, int(c_hi))
        if begin >= end:
            return None
        dst.append(slice(begin - lo, end - lo))
        src.append(slice(begin - int(c_lo), end - int(c_lo)))
    return tuple(dst), tuple(src)


def _raise_evicted(source_id: str, bounds: ChunkBounds):
    """The read of a chunk that was uploaded and is no longer in the cache.

    Never a hole: this runs only for a chunk ``_written_chunks`` records, so
    the bytes existed and are gone. An upload's cache entry is its only copy,
    so there is nothing to rebuild them from.
    """
    raise flight.FlightServerError(
        f"Cache-backed source {source_id} has lost the chunk at "
        f"{list(bounds.start)}..{list(bounds.stop)}: it was uploaded, and the "
        f"cache has since evicted it. An upload's cache entry is its only copy."
    )
