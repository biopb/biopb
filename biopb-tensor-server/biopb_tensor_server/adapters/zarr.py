"""Zarr adapter for tensor storage.

Relies on OS page cache for raw data caching.
"""

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._writable import WritableSource
from biopb_tensor_server.core.adapter_base import (
    TensorAdapter,
    catalog_entry,
)
from biopb_tensor_server.core.chunk import (
    content_version_from_path,
    default_transfer_chunk_shape,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import UploadDiscardedError

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

# The upload marker a server-minted store carries in its root ``.zattrs``:
# ``{"biopb": {"upload": {"state": "pending" | "ready"}}}``. Written at create,
# flipped at ``finish``. A store still ``pending`` when a server starts is a
# crashed upload -- ``UploadManager.discard_unfinished_stores`` deletes it, and
# discovery declines it (``is_unfinished_upload``) so a shared root never
# serves a partial store as a source of its own (biopb/biopb#1059).
UPLOAD_ATTR = "biopb"
UPLOAD_PENDING = "pending"
UPLOAD_READY = "ready"


def upload_state(zattrs: Any) -> Optional[str]:
    """The upload marker's state from a parsed ``.zattrs``, or None if unmarked."""
    if not isinstance(zattrs, dict):
        return None
    upload = (zattrs.get(UPLOAD_ATTR) or {}).get("upload")
    if not isinstance(upload, dict):
        return None
    state = upload.get("state")
    return state if isinstance(state, str) else None


def with_upload_state(zattrs: dict, state: str) -> dict:
    """*zattrs* with the upload marker set to *state* (a copy; the input is not touched)."""
    out = dict(zattrs)
    block = dict(out.get(UPLOAD_ATTR) or {})
    block["upload"] = {"state": state}
    out[UPLOAD_ATTR] = block
    return out


def is_unfinished_upload(ctx: ClaimContext) -> bool:
    """Whether the ``.zarr`` directory at *ctx* is an upload that never finished.

    Reads ``.zattrs`` only when it is present and resident: a non-resident
    cloud placeholder is deferred by the claims themselves, and a store nobody
    marked is not an upload.
    """
    zattrs_ctx = ctx.join(".zattrs")
    if not zattrs_ctx.exists() or not zattrs_ctx.is_resident():
        return False
    try:
        return upload_state(json.loads(ctx.read_text(".zattrs"))) == UPLOAD_PENDING
    except Exception:
        return False


class ZarrAdapter(WritableSource, TensorAdapter):
    """Adapter for Zarr/N5 chunked arrays.

    Supports both local filesystem and remote storage (S3, GCS, etc.) via fsspec.
    For remote storage, uses zarr.FSStore with fsspec filesystem.

    Writable: a chunk-aligned ``put_chunk`` lands in the store. Only an adapter
    built by ``OmeZarrAdapter.create_upload`` tracks an upload; a catalogued
    store accepts writes untracked.

    An upload's store is the server's own (minted under ``write_dir``), so the
    upload half here also owns its end: discard removes the directory
    (:meth:`_dispose_store`) and finish clears the pending marker
    (:meth:`_mark_store_finished`). Neither touches a discovered store, which
    never began an upload and so never reaches either.
    """

    # A real store on disk, catalogued: the boundary keeps the row in step.
    durable = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim .zarr directories with .zarray or .zattrs.

        Supports both zarr v2 (.zarray/.zattrs) and zarr v3 (zarr.json).
        Works for both local filesystem and remote storage.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim if this is a plain zarr array, None otherwise
        """
        # Must be a directory ending in .zarr
        if not ctx.is_dir() or not ctx.name.endswith(".zarr"):
            return None
        if is_unfinished_upload(ctx):
            return None

        # Check for zarr structure files
        has_zarray = ctx.join(".zarray").exists()
        has_zarr_json = ctx.join("zarr.json").exists()
        has_zattrs = ctx.join(".zattrs").exists()

        # Zarr v2/v3: has .zarray / zarr.json (array metadata). The marker's
        # existence is a stat (recall-free), but if it is a non-resident cloud
        # placeholder the source must still be deferred -- registering it resolved
        # would let create_from_config read the metadata AND make it eligible for
        # background precache warming, which would recall the whole array. So under
        # non-residency claim it provisionally and let resolution hydrate it.
        if has_zarray or has_zarr_json:
            marker = ctx.join(".zarray") if has_zarray else ctx.join("zarr.json")
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type="zarr",
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
                unresolved=not marker.is_resident(),
            )

        # If only .zattrs exists, check if it's NOT an OME-Zarr
        if has_zattrs:
            # Cloud-storage phase 2: reading .zattrs to disambiguate plain-zarr
            # from OME-Zarr would recall a non-resident sidecar placeholder (or
            # block offline). Defer the read exactly as OmeZarrAdapter does: a
            # .zarr dir with a .zattrs is structurally a zarr store, so claim it
            # provisionally as plain zarr and let resolution re-derive the exact
            # type from the hydrated content. (OmeZarrAdapter runs first and, when
            # it also defers, wins claims[0]; this branch only owns the .zattrs
            # store OmeZarr did not provisionally claim.)
            zattrs_ctx = ctx.join(".zattrs")
            if not zattrs_ctx.is_resident():
                state.try_claim_path(ctx.path_str)
                return SourceClaim(
                    source_type="zarr",
                    primary_path=ctx.path_str,
                    is_remote=ctx.is_remote,
                    unresolved=True,
                )
            try:
                zattrs = json.loads(ctx.read_text(".zattrs"))
                # If no multiscales, it might be a plain zarr group or array
                if "multiscales" not in zattrs:
                    state.try_claim_path(ctx.path_str)
                    return SourceClaim(
                        source_type="zarr",
                        primary_path=ctx.path_str,
                        is_remote=ctx.is_remote,
                    )
            except (json.JSONDecodeError, Exception):
                pass

        return None

    def get_metadata(self):
        return {}

    # get_tensor_adapter: inherit the total single-tensor base -- it returns self
    # for the source's sole tensor and raises TensorNotFound for an unknown
    # nonempty field, so a typo'd array_id no longer silently reads the base
    # tensor (issue #378).

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "ZarrAdapter":
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            ZarrAdapter instance
        """
        import zarr
        from zarr.storage import FSStore

        from biopb_tensor_server.core.remote import RemoteStore

        if source.is_remote:
            # Remote storage: use RemoteStore for filesystem creation
            store = RemoteStore.from_config(
                source.url,
                credentials_config=credentials_config,
                profile_name=source.credentials_profile,
            )
            zarr_store = FSStore(store.path, fs=store.fs)
            arr = zarr.open_array(zarr_store, mode="r")
        else:
            # Local filesystem
            path = Path(source.url)
            arr = zarr.open_array(str(path), mode="r")

        return cls(arr, source.source_id)

    def __init__(
        self,
        zarr_array,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize Zarr adapter.

        Args:
            zarr_array: Zarr array object
            source_id: Unique identifier for this data source
        """
        self.zarr_array = zarr_array
        self.source_id = source_id
        self.dim_labels = dim_labels or [f"dim{i}" for i in range(zarr_array.ndim)]

        # Source-level metadata for DataSourceDescriptor
        self._source_url = str(
            zarr_array.store.path
            if hasattr(zarr_array.store, "path")
            else str(zarr_array.store)
        )
        # Cheap content_version from the store directory's stat signature (#178):
        # O(1) dir mtime, which flips on member add/remove/rename. A remote store
        # path (S3/GCS) can't be stat'd -> None -> the source stays unversioned.
        # Inherited by OmeZarrAdapter / _HcsFieldAdapter via super().__init__.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = "zarr"
        # The directory ``create_upload`` minted, for the two store hooks; None
        # on a discovered store, which is not this adapter's to remove or mark.
        self._upload_store_path: Optional[Path] = None
        # Serializes chunk writes against store disposal. Taken by ``put_chunk``
        # around refuse-and-store, so a write that passed ``_refuse_write``
        # lands before ``_dispose_store`` removes the directory, and a write
        # arriving after is refused -- rather than recreating the directory a
        # moment after it was deleted (zarr's DirectoryStore makes parents on
        # write). Ordered before ``progress.lock``; disposal never holds that.
        self._write_lock = threading.Lock()

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """The store's chunk -- the ``native=`` seed of the grid below.

        Inherited by ``OmeZarrAdapter``, ``_HcsFieldAdapter`` and
        ``_QptiffLevelAdapter``, which all read through a zarr array. A rank that
        does not match the array means this adapter presents a different axis
        space than the store's, so the block is not comparable and none is
        claimed. ``None`` also before the array is bound -- ``_QptiffLevelAdapter``
        re-resolves its array per read -- which reads a tiled store as
        unquantized; reachable only if a native level ever serves a computed
        sub-scale, since ``precompute`` chunk_ids are unscaled and never arrive
        here.
        """
        array = getattr(self, "zarr_array", None)
        chunks = getattr(array, "chunks", None)
        if not chunks or len(chunks) != len(array.shape):
            return None
        return tuple(int(size) for size in chunks)

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from zarr array.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape
        """
        super().get_data(bounds)
        slices = self._bounds_to_slices(bounds)
        return self.zarr_array[slices]

    def resolve_chunk_data(self, chunk_id: bytes, cache_manager: Any = None) -> Any:
        """The read path, refused for a tombstone before the cache is consulted.

        The store is gone once discarded, and zarr would answer fill values
        for its missing chunks -- or the cache would answer the bytes it still
        holds. Either way a reader would see data behind a source that has
        none; it learns the reason instead, as a writer does, mapped to the
        read path's wire error like ``CachedSourceAdapter`` maps it.
        """
        try:
            self._refuse_if_discarded()
        except UploadDiscardedError as e:
            raise flight.FlightServerError(str(e)) from e
        return super().resolve_chunk_data(chunk_id, cache_manager)

    def put_chunk(self, bounds, data, expected_shape, dtype) -> None:
        with self._write_lock:
            super().put_chunk(bounds, data, expected_shape, dtype)

    def _dispose_store(self) -> None:
        """Remove the minted directory; see ``_write_lock`` for the ordering."""
        path = self._upload_store_path
        if path is None:
            return
        with self._write_lock:
            shutil.rmtree(path, ignore_errors=True)
        logger.info(f"Removed the store of discarded upload {self.source_id}: {path}")

    def _mark_store_finished(self) -> None:
        self._write_upload_state(UPLOAD_READY)

    def _write_upload_state(self, state: str) -> None:
        """Rewrite the root ``.zattrs`` with the upload marker set to *state*.

        Atomic (write-then-replace) so a crash mid-write cannot leave a store
        with no ``.zattrs`` at all. A store that is already gone -- discard
        raced finish -- is left alone: the marker only matters on a directory
        that exists.
        """
        path = self._upload_store_path
        if path is None:
            return
        zattrs_path = path / ".zattrs"
        with self._write_lock:
            try:
                zattrs = json.loads(zattrs_path.read_text())
            except OSError:
                return
            tmp = zattrs_path.with_name(".zattrs.tmp")
            tmp.write_text(json.dumps(with_upload_state(zattrs, state)))
            os.replace(tmp, zattrs_path)

    def write_chunk(self, chunk_idx: Tuple[int, ...], data: np.ndarray) -> None:
        """Write chunk data to zarr array.

        Args:
            chunk_idx: Chunk coordinates (e.g., (0, 1, 2))
            data: Numpy array with chunk data
        """
        chunks = self.zarr_array.chunks
        slices = tuple(
            slice(idx * chunks[d], (idx + 1) * chunks[d])
            for d, idx in enumerate(chunk_idx)
        )

        # Handle edge chunks - pad if data smaller than expected
        expected_shape = tuple(s.stop - s.start for s in slices)
        if data.shape != expected_shape:
            padded = np.zeros(expected_shape, dtype=self.zarr_array.dtype)
            src_slices = tuple(
                slice(0, min(d, es))
                for d, es in zip(data.shape, expected_shape, strict=True)
            )
            padded[src_slices] = data[src_slices]
            data = padded

        self.zarr_array[slices] = data

    def _store_chunk(self, bounds, data, expected_shape, dtype) -> None:
        """Chunk-aligned write: ``bounds`` must land on the zarr chunk grid.

        Absorbs the alignment/reshape the DoPut handler used to perform inline,
        then delegates the store to ``write_chunk``. The grid comes straight off
        ``self.zarr_array.chunks`` -- the same grid ``write_chunk`` writes into
        (and equal to the descriptor's ``chunk_shape``) -- so validation and
        storage never disagree, and the method stays purely source-level.
        """
        arr = data.to_numpy()
        if expected_shape:
            arr = arr.reshape(expected_shape)
        chunk_shape = list(self.zarr_array.chunks)

        # start must align to the chunk grid
        for d, (start, chunk_size) in enumerate(
            zip(bounds.start, chunk_shape, strict=True)
        ):
            if start % chunk_size != 0:
                raise ValueError(
                    f"Chunk start[{d}]={start} not aligned to chunk_shape[{d}]={chunk_size}"
                )

        # size may only shrink at the edge, never exceed the nominal chunk
        actual_size = [
            stop - start for start, stop in zip(bounds.start, bounds.stop, strict=True)
        ]
        for d, (actual, expected) in enumerate(
            zip(actual_size, chunk_shape, strict=True)
        ):
            if actual > expected:
                raise ValueError(
                    f"Chunk size[{d}]={actual} exceeds chunk_shape[{d}]={expected}"
                )

        chunk_idx = tuple(
            int(s // cs) for s, cs in zip(bounds.start, chunk_shape, strict=True)
        )
        self.write_chunk(chunk_idx, arr)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self.zarr_array.shape),
            # The store's chunk grid is an alignment seed, not the transfer
            # unit: zarr blocks are routinely far below the transfer target, and
            # shipping one per chunk is what biopb/biopb#684 measured as too many
            # endpoints. Reads are served from a real backend at any bounds, so
            # the grid is free to be a whole multiple of the store's.
            chunk_shape=default_transfer_chunk_shape(
                self.zarr_array.shape,
                self.zarr_array.dtype.str,
                self.dim_labels,
                native=self.zarr_array.chunks,
            ),
            dtype=self.zarr_array.dtype.str,
        )

    def list_tensor_descriptors(self):
        return [catalog_entry(self.get_tensor_descriptor())]
