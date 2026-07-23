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

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import (
    DEFAULT_HANDLE_REAPER_TTL,
    IdleHandleReaper,
)
from biopb_tensor_server.adapters._scale import mm_summary_scale
from biopb_tensor_server.core.adapter_base import TensorAdapter
from biopb_tensor_server.core.chunk import content_version_from_path
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim

if TYPE_CHECKING:
    from ndtiff import NDTiffDataset

    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.core.remote import RemoteStore


# =============================================================================
# Persistent dataset pool (steady-state fd hygiene, biopb/biopb#71)
# =============================================================================
#
# ``NDTiffDataset.__init__`` eagerly opens *every* ``NDTiffStack_*.tif`` in the
# acquisition, so a catalogued source pins one fd per file -- routinely hundreds,
# which on Windows makes the whole acquisition folder undeletable and on POSIX
# holds disk after an unlink. The handle is kept warm between reads (a
# reopen-per-read would reopen the *entire* acquisition to serve one plane, since
# the reopen unit is decoupled from the read unit), and a shared idle reaper
# closes it once no one has read the source for the TTL -- bounding the pin rather
# than pinning for the catalog's whole lifetime. The next read after a lull pays
# one reopen. The TTL is set from ``ServerConfig.handle_reaper_ttl`` at startup;
# see :mod:`biopb_tensor_server.adapters._handle_reaper`.
_dataset_reaper = IdleHandleReaper(DEFAULT_HANDLE_REAPER_TTL, "ndtiff-dataset-reaper")


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
    ``get_metadata`` / ``_physical_scale`` never touch ``self._dataset`` -- which
    the reaper may have closed and set to ``None`` between reads.
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

        Builds a ``reopen`` thunk capturing the url + credentials so the reaper
        can close the acquisition when idle and a later read can reopen it (see
        the module docstring), opens it once for the initial handle, and hands
        both to the adapter.

        Args:
            source: SourceConfig with url, source_id, dim_labels
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
            dim_labels=source.dim_labels,
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
        path after the reaper closes the dataset. Imports are deferred to call
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
        dim_labels: Optional[List[str]] = None,
        io_lock: Optional[threading.Lock] = None,
        reopen: Optional[Callable[[], NDTiffDataset]] = None,
    ):
        """Initialize NDTiff adapter.

        Args:
            dataset: NDTiffDataset instance (the initial open handle)
            source_id: Unique identifier for this data source
            source_url: URL or path to the data source
            dim_labels: Optional dimension labels (if None, inferred from dataset)
            io_lock: Optional thread lock for IO serialization
            reopen: Optional zero-arg thunk that reopens the dataset. When set (the
                ``create_from_config`` path), the reaper may close the acquisition
                between reads and the read path reopens on demand. When None (a
                caller that handed in a bare dataset, e.g. a test), the handle is
                never reaped and a read after ``close()`` fails loudly.
        """
        self._dataset = dataset
        self._reopen = reopen
        self.source_id = source_id
        self._source_url = source_url
        # Cheap content_version from the directory's stat signature (#178): O(1)
        # dir mtime, which flips on member add/remove/rename -- the right signal
        # for an NDTiff dataset dir. None (unresolved url) leaves it unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = self.SOURCE_TYPE
        self._io_lock = io_lock or threading.Lock()
        # Reaper bookkeeping (see the module docstring). Reads stay entirely under
        # _io_lock, so no lock-free read is ever in flight -- _active_reads is a
        # constant 0 and the reaper's non-blocking _io_lock acquire alone fences a
        # close against a read.
        self._persistent_last_access = time.monotonic()
        self._active_reads = 0

        # Get dask array from dataset
        self._dask_arr = dataset.as_array()

        # Summary metadata snapshot -- so get_metadata/_physical_scale never reach
        # through self._dataset, which the reaper may have closed (see the helper).
        self._summary_metadata = _extract_summary(dataset)

        # Get axes from dataset
        # ndtiff uses axis names: position, time, channel, z, row, column
        self._axes = list(dataset.axes.keys()) if hasattr(dataset, "axes") else []

        # Map axis names to short labels
        axis_alias = {
            "position": "p",
            "time": "t",
            "channel": "c",
            "z": "z",
            "row": "y",
            "column": "x",
        }

        # Build dim_labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            # Infer from axes - last two are always y, x
            self.dim_labels = []
            for ax in self._axes[:-2]:  # Exclude row, column
                label = axis_alias.get(ax.lower(), ax.lower()[0])
                self.dim_labels.append(label)
            # Add spatial dims
            self.dim_labels.extend(["y", "x"])

        # Get shape and dtype from dask array
        self._shape = list(self._dask_arr.shape)
        self._dtype = str(self._dask_arr.dtype)

        # Chunk shape: one 2D plane per chunk (1, 1, 1, 1, Y, X) or subset
        # This matches ndtiff's tile-based storage
        spatial_shape = self._shape[-2:]  # Y, X
        n_spatial = len(spatial_shape)
        n_non_spatial = len(self._shape) - n_spatial
        self._chunk_shape = [1] * n_non_spatial + spatial_shape

        # Only a reopen-capable adapter is worth reaping -- one handed a bare
        # dataset it cannot rebuild must keep it. Registering also lazily starts
        # the reaper thread, so a bare-dataset caller (a test) spawns nothing.
        if self._reopen is not None:
            _dataset_reaper.register(self)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this adapter."""
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self._shape,
            chunk_shape=self._chunk_shape,
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors - single tensor source."""
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from the dask array.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        super().get_data(bounds)
        slices = self._bounds_to_slices(bounds)

        with self._io_lock:
            dask_arr = self._ensure_dask_arr()
            self._persistent_last_access = time.monotonic()
            return dask_arr[slices].compute()

    def _ensure_dask_arr(self):
        """Return the dask array, reopening the acquisition if the reaper closed it.

        Caller holds ``self._io_lock``. A reopen-capable adapter (created via
        ``create_from_config``) rebuilds the whole ``NDTiffDataset`` and re-arms
        the reaper; one handed a bare dataset has nothing to rebuild, so a read
        after ``close()`` fails loudly instead.
        """
        if self._dask_arr is not None:
            return self._dask_arr
        if self._reopen is None:
            raise RuntimeError(f"NDTiff source {self.source_id!r} is closed")
        dataset = self._reopen()
        self._dataset = dataset
        self._dask_arr = dataset.as_array()
        self._persistent_last_access = time.monotonic()
        _dataset_reaper.register(self)
        return self._dask_arr

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
        the reaper has closed the underlying dataset.
        """
        return self._summary_metadata

    # ---- lifecycle ----------------------------------------------------------

    def _release_persistent_handle(self) -> None:
        """Close the acquisition's per-file readers; permit a later reopen.

        ``NDTiffDataset.__init__`` eagerly opens *every* ``NDTiffStack_*.tif`` in
        the acquisition, so one registered source pins as many fds as the
        acquisition has files -- routinely hundreds, which on Windows makes the
        whole folder undeletable. Upstream's ``close()`` closes every reader; the
        dask array must go first because its graph holds the dataset.

        This is the shared reaper's release hook (called under ``_io_lock`` once
        the source has been idle past the TTL) and also backs the explicit
        ``close()``. It leaves ``self._reopen`` intact, so a reaper close is
        transparent to a later read; ``close()`` is the same drop with teardown
        intent. Caller holds ``_io_lock`` (reaper/close) or is the GC finalizer.
        """
        self._dask_arr = None
        dataset = self._dataset
        self._dataset = None
        _dataset_reaper.discard(self)
        _close_dataset(dataset)

    def close(self) -> None:
        """Release the dataset's per-file readers on teardown (biopb/biopb#71).

        The handles stay persistent between reads rather than being reopened per
        read (unlike hdf5/mrc): the reopen unit here is the *whole* acquisition,
        so a per-read reopen would open thousands of files to serve one plane. The
        idle reaper bounds the steady-state pin; this releases it deterministically
        on unregister/shutdown.
        """
        with self._io_lock:
            self._release_persistent_handle()

    def __del__(self):
        # GC backstop: release the fds even without an explicit close(). No lock
        # -- nothing references the adapter, so no read can be in flight.
        try:
            self._release_persistent_handle()
        except Exception:
            pass
