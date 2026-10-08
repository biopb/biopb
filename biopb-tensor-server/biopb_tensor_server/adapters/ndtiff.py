"""NDTiff adapter for Micro-Manager NDTiff storage format.

Handles newer Micro-Manager NDTiff storage format:
- Binary index file: NDTiff.index
- TIFF files: NDTiffStack_*.tif
- Uses ndtiff package which provides as_array() dask interface

Key characteristics:
- Single tensor source exposing full 5D/6D array
- Uses ndtiff.as_array() for lazy dask array access
- All positions share the same spatial dimensions (Y, X)
- as_array() creates unified dask array with zero-padding for missing coordinates

Remote storage support via RemoteNdTiffFileIO wrapper.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_pool import HandlePool, PooledHandle
from biopb_tensor_server.adapters._handle_reaper import DEFAULT_HANDLE_REAPER_TTL
from biopb_tensor_server.adapters._scale import mm_summary_scale
from biopb_tensor_server.core.adapter_base import (
    TensorAdapter,
    TensorEntry,
    catalog_entry,
)
from biopb_tensor_server.core.chunk import (
    content_version_from_path,
    default_transfer_chunk_shape,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.remote import is_remote_url

if TYPE_CHECKING:
    from ndtiff import NDTiffDataset

    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.core.remote import RemoteStore


# The acquisition's open dataset is the handle, kept warm between reads in a pool
# keyed by file identity: a per-read reopen would reopen the *entire* acquisition
# to serve one plane, since the reopen unit is decoupled from the read unit. A
# rebuilt adapter finds the dataset its predecessor opened, and the pool closes it
# once no one has read the source for the TTL (``ServerConfig.handle_reaper_ttl``
# is its ceiling). The reopen unit is the whole acquisition, so the TTL is the
# long default -- but so is the *pin*: NDTiffDataset eagerly opens every
# NDTiffStack_*.tif, so one warm handle here can hold thousands of file
# descriptors where every other pool holds one. That asymmetry, not the reopen
# cost, sets the cap.
_dataset_pool = HandlePool(DEFAULT_HANDLE_REAPER_TTL, 4, "ndtiff-dataset-pool")


# =============================================================================
# RemoteNdTiffFileIO - NDTiffFileIO wrapper for remote storage
# =============================================================================


class RemoteNdTiffFileIO:
    """NDTiffFileIO wrapper using RemoteStore (fsspec).

    Provides the interface expected by ndtiff's NDTiffDataset for remote
    storage access via fsspec.

    The ndtiff library expects file_io with:
    - open_function: callable(path, mode='rb') -> file-like object
    - listdir_function: callable(path) -> list of filenames
    - path_join_function: callable(a, b) -> joined path
    - isdir_function: callable(path) -> bool
    """

    def __init__(self, store: RemoteStore):
        """Initialize RemoteNdTiffFileIO.

        Args:
            store: RemoteStore instance for fsspec-based remote access
        """
        self._store = store

    def open_function(self, path: str, mode: str = "rb"):
        """Open file for reading.

        Args:
            path: Path relative to store root
            mode: File mode (should be 'rb' for binary read)

        Returns:
            File-like object
        """
        # ndtiff paths may be relative to dataset directory
        # strip any leading path components that match store.path
        return self._store.open(path, mode)

    def listdir_function(self, path: str) -> List[str]:
        """List contents of a directory.

        Args:
            path: Path relative to store root

        Returns:
            List of filenames in the directory
        """
        return self._store.listdir(path)

    def path_join_function(self, a: str, b: str) -> str:
        """Join path components.

        Args:
            a: First path component
            b: Second path component

        Returns:
            Joined path
        """
        # ndtiff uses os.path.join semantics
        # For remote storage, we need to handle path joining carefully
        if not a:
            return b
        return f"{a.rstrip('/')}/{b}"

    def isdir_function(self, path: str) -> bool:
        """Check if path is directory.

        Args:
            path: Path to check

        Returns:
            True if path is a directory
        """
        return self._store.isdir(path)


def _close_dataset(dataset) -> None:
    """Close an NDTiffDataset if it offers close(). Tolerates a test double."""
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _extract_summary(dataset) -> dict:
    """Snapshot a dataset's summary metadata as a plain dict.

    ``summary_metadata`` is parsed from ``NDTiff.index`` at construction and does
    not depend on the per-file readers, but snapshotting it at registration means
    ``get_metadata`` / ``_physical_scale`` never touch the dataset -- which
    the pool may have closed between reads.
    """
    meta = getattr(dataset, "summary_metadata", None)
    if meta is None:
        return {}
    if hasattr(meta, "model_dump"):
        return meta.model_dump(mode="json")
    if hasattr(meta, "dict"):
        return meta.dict()
    if isinstance(meta, dict):
        return meta
    return {}


# =============================================================================
# NdTiffAdapter - Adapter for NDTiff storage format
# =============================================================================


class NdTiffAdapter(TensorAdapter):
    """Adapter for Micro-Manager NDTiff storage format.

    Single-tensor source exposing full 5D/6D array.
    Uses ndtiff.as_array() for lazy dask array access.

    All positions share the same spatial dimensions (Y, X) - the
    as_array() method creates a unified dask array with zero-padding
    for missing coordinates.
    """

    SOURCE_TYPE = "ndtiff"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: DiscoveryState) -> Optional[SourceClaim]:
        """Claim directories containing NDTiff datasets.

        Detects NDTiff.index file in directory - this is the signature
        file for NDTiff storage format.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if NDTiff detected
        """
        # Only directories
        if not ctx.is_dir():
            return None

        # Check for NDTiff.index file
        index_file = ctx.join("NDTiff.index")
        if not index_file.exists():
            return None

        # Dir-claiming policy (biopb/biopb): the directory IS the dataset
        # boundary. Claim the dir (+ the recall-free NDTiff.index marker) only;
        # claiming the dir already prunes its whole subtree, so the interior
        # NDTiffStack_*.tif files are never independently walked. Recording them
        # as members would just duplicate that prune and pin a brittle glob.
        state.try_claim_path(ctx.path_str)
        state.try_claim_path(index_file.path_str)

        return SourceClaim(
            source_type=cls.SOURCE_TYPE,
            primary_path=ctx.path_str,
            is_remote=ctx.is_remote,
        )

    @classmethod
    def create_from_config(
        cls,
        source: SourceConfig,
        credentials_config: Optional[Any] = None,
    ) -> NdTiffAdapter:
        """Create adapter instance from SourceConfig.

        Builds a ``reopen`` thunk capturing the url + credentials so the pool
        can close the acquisition when idle and a later read can reopen it (see
        the module docstring), opens it once for the initial handle, and hands
        both to the adapter.

        Args:
            source: SourceConfig with url, source_id
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            NdTiffAdapter instance
        """
        reopen = cls._dataset_opener(
            url=source.url,
            is_remote=source.is_remote,
            credentials_config=credentials_config,
            credentials_profile=source.credentials_profile,
        )
        return cls(
            dataset=reopen(),
            source_id=source.source_id or "",
            source_url=str(source.url),
            reopen=reopen,
        )

    @staticmethod
    def _dataset_opener(
        url: Any,
        is_remote: bool,
        credentials_config: Optional[Any],
        credentials_profile: Optional[str],
    ) -> Callable[[], NDTiffDataset]:
        """Return a zero-arg thunk that (re)opens the ``NDTiffDataset``.

        The same construction ``create_from_config`` used, replayable by the read
        path after the pool closes the dataset. Imports are deferred to call
        time so an env without ndtiff (or fsspec) still imports this module.
        """

        def _open() -> NDTiffDataset:
            from ndtiff import NDTiffDataset

            if is_remote:
                from biopb_tensor_server.core.remote import RemoteStore

                store = RemoteStore.from_config(
                    url=url,
                    credentials_config=credentials_config,
                    profile_name=credentials_profile,
                )
                # Remote path is empty -- the store root IS the dataset.
                return NDTiffDataset("", file_io=RemoteNdTiffFileIO(store))
            # Local filesystem: resolve so the reopen matches the first open.
            return NDTiffDataset(str(Path(url).resolve()))

        return _open

    def __init__(
        self,
        dataset: NDTiffDataset,
        source_id: str,
        source_url: str,
        reopen: Optional[Callable[[], NDTiffDataset]] = None,
        *,
        structure: Optional[Dict[str, Any]] = None,
        summary: Optional[dict] = None,
    ):
        """Initialize NDTiff adapter.

        Args:
            dataset: NDTiffDataset instance (the initial open handle)
            source_id: Unique identifier for this data source
            source_url: URL or path to the data source
            reopen: Optional zero-arg thunk that reopens the dataset. When set (the
                ``create_from_config`` path), the dataset is pooled: the pool may
                close it between reads and the read path reopens on demand. When
                None (a caller that handed in a bare dataset, e.g. a test), the
                handle is this adapter's own, never pooled, and a read after
                ``close()`` fails loudly.
            structure / summary: a restored source's axes, shape and dtype and its
                summary metadata (``catalog_payload`` and the row), given with no
                *dataset* (``None``) and a *reopen*: nothing is opened until a read
                needs the acquisition, as after the pool closes an idle one.
        """
        self._reopen = reopen
        self.source_id = source_id
        self._source_url = source_url
        # Cheap content_version from the directory's stat signature (#178): O(1)
        # dir mtime, which flips on member add/remove/rename -- the right signal
        # for an NDTiff dataset dir. None (unresolved url) leaves it unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = self.SOURCE_TYPE
        if dataset is not None:
            dask_arr = dataset.as_array()

            # Summary metadata snapshot -- so get_metadata/_physical_scale never
            # reach through the dataset, which the pool may have closed (see the
            # helper).
            self._summary_metadata = _extract_summary(dataset)

            # Get axes from dataset
            # ndtiff uses axis names: position, time, channel, z, row, column
            self._axes = list(dataset.axes.keys()) if hasattr(dataset, "axes") else []
            self._shape = list(dask_arr.shape)
            self._dtype = str(dask_arr.dtype)
        else:
            self._summary_metadata = summary or {}
            self._axes = list(structure["axes"])
            self._shape = [int(s) for s in structure["shape"]]
            self._dtype = structure["dtype"]

        # Map axis names to short labels
        axis_alias = {
            "position": "p",
            "time": "t",
            "channel": "c",
            "z": "z",
            "row": "y",
            "column": "x",
        }

        # Infer from axes - last two are always y, x
        self.dim_labels = []
        for ax in self._axes[:-2]:  # Exclude row, column
            label = axis_alias.get(ax.lower(), ax.lower()[0])
            self.dim_labels.append(label)
        self.dim_labels.extend(["y", "x"])

        # One 2D plane matches ndtiff's tile-based storage; it seeds the
        # transfer grid rather than being it (biopb/biopb#809), so a small plane
        # ships several planes per chunk instead of one endpoint each.
        spatial_shape = self._shape[-2:]  # Y, X
        n_spatial = len(spatial_shape)
        n_non_spatial = len(self._shape) - n_spatial
        self._chunk_shape = default_transfer_chunk_shape(
            self._shape,
            self._dtype,
            self.dim_labels,
            native=[1] * n_non_spatial + spatial_shape,
        )

        # The handle the constructor was given: pooled when the adapter can reopen
        # (a restored source holds none yet; its first read opens one), else this
        # adapter's own for life.
        self._own: Optional[PooledHandle] = None
        if dataset is not None:
            handle = self._handle_for(dataset, dask_arr)
            if self._reopen is not None:
                _dataset_pool.put(handle)
            else:
                self._own = handle

    def catalog_payload(self) -> Optional[Dict[str, Any]]:
        """The acquisition's axes, shape and dtype: what opening it (every stack
        file, and a dask graph over them) is done for before a descriptor can be
        made. The summary metadata is the row's. ``None`` for a remote store."""
        if is_remote_url(self._source_url):
            return None
        return {
            "axes": list(self._axes),
            "shape": [int(s) for s in self._shape],
            "dtype": self._dtype,
        }

    @classmethod
    def create_from_payload(
        cls,
        source: SourceConfig,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        credentials_config: Optional[Any] = None,
    ) -> NdTiffAdapter:
        """Rebuild with no dataset open: the acquisition is opened by the first read
        (``_open``), as it is after the pool has closed an idle one."""
        reopen = cls._dataset_opener(
            url=source.url,
            is_remote=source.is_remote,
            credentials_config=credentials_config,
            credentials_profile=source.credentials_profile,
        )
        return cls(
            dataset=None,
            source_id=source.source_id or "",
            source_url=str(source.url),
            reopen=reopen,
            structure=payload,
            summary=metadata,
        )

    def _native_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this adapter."""
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self._shape,
            chunk_shape=self._chunk_shape,
            dtype=self._dtype,
        )

    def _list_native_tensors(self) -> List[TensorEntry]:
        """List all tensors - single tensor source."""
        return [catalog_entry(self._native_descriptor())]

    @property
    def _native_read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """One plane -- the ``native=`` seed above, and the dask block behind it.

        A read slices the dask array, which materialises whole blocks whatever
        window is asked for.
        """
        return tuple([1] * (len(self._shape) - 2) + list(self._shape[-2:]))

    def _read_native(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from the dask array.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        super()._read_native(bounds)
        slices = self._bounds_to_slices(bounds)

        with self._leased() as handle, handle.lock:
            return handle.value[1][slices].compute()

    def _pool_key(self):
        return (self._source_url, self._content_version)

    def _handle_for(self, dataset, dask_arr) -> PooledHandle:
        return PooledHandle(
            self._pool_key(), (dataset, dask_arr), lambda: _close_dataset(dataset)
        )

    def _open(self) -> PooledHandle:
        """Reopen the whole acquisition as a handle the pool closes."""
        dataset = self._reopen()
        return self._handle_for(dataset, dataset.as_array())

    @contextmanager
    def _leased(self):
        """Lease the open acquisition, reopening it if the pool closed it. An
        adapter handed a bare dataset has nothing to rebuild, so a read after
        ``close()`` fails loudly instead."""
        if self._reopen is None:
            if self._own is None:
                raise RuntimeError(f"NDTiff source {self.source_id!r} is closed")
            yield self._own
            return
        with _dataset_pool.checkout(self._pool_key(), self._open) as handle:
            yield handle

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dim pixel size (µm) from the MicroManager summary metadata.

        ``PixelSize_um`` (isotropic X/Y) and the z-step, projected onto the
        ``x`` / ``y`` / ``z`` axes; position / time / channel axes get
        ``0.0`` / ``""``. Reads the same summary dict :meth:`get_metadata`
        returns.
        """
        return mm_summary_scale(self.get_metadata(), self.dim_labels)

    def get_metadata(self) -> dict:
        """Return dataset summary metadata (MicroManager acquisition settings).

        Served from the snapshot taken at registration, so it stands even after
        the pool has closed the underlying dataset.
        """
        return self._summary_metadata

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the acquisition's per-file readers on teardown (biopb/biopb#71).

        ``NDTiffDataset.__init__`` eagerly opens *every* ``NDTiffStack_*.tif``, so
        one registered source pins as many fds as the acquisition has files --
        routinely hundreds, which on Windows makes the whole folder undeletable.
        The pooled dataset is closed at its last lease; the pool's TTL bounds the
        steady-state pin and this releases it deterministically on
        unregister/shutdown.
        """
        if self._reopen is not None:
            _dataset_pool.drop(self._pool_key())
        elif self._own is not None:
            own, self._own = self._own, None
            own.close()
