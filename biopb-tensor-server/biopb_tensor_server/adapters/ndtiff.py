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
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.discovery import ClaimContext, SourceClaim

if TYPE_CHECKING:
    from ndtiff import NDTiffDataset

    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState
    from biopb_tensor_server.remote import RemoteStore


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


# =============================================================================
# NdTiffAdapter - Adapter for NDTiff storage format
# =============================================================================


class NdTiffAdapter(SourceAdapter, TensorAdapter):
    """Adapter for Micro-Manager NDTiff storage format.

    Single-tensor source exposing full 5D/6D array.
    Uses ndtiff.as_array() for lazy dask array access.

    All positions share the same spatial dimensions (Y, X) - the
    as_array() method creates a unified dask array with zero-padding
    for missing coordinates.
    """

    _single_tensor_source = True
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

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            NdTiffAdapter instance
        """
        from ndtiff import NDTiffDataset

        if source.is_remote:
            # Remote storage: create RemoteStore and wrap with RemoteNdTiffFileIO
            from biopb_tensor_server.remote import RemoteStore

            store = RemoteStore.from_config(
                url=source.url,
                credentials_config=credentials_config,
                profile_name=source.credentials_profile,
            )
            file_io = RemoteNdTiffFileIO(store)

            # NDTiffDataset expects a path within the file_io context
            # For remote, the path is empty (root of the store)
            dataset = NDTiffDataset("", file_io=file_io)
        else:
            # Local filesystem
            dataset_path = Path(source.url).resolve()
            dataset = NDTiffDataset(str(dataset_path))

        return cls(
            dataset=dataset,
            source_id=source.source_id or "",
            source_url=str(source.url),
            dim_labels=source.dim_labels,
        )

    def __init__(
        self,
        dataset: NDTiffDataset,
        source_id: str,
        source_url: str,
        dim_labels: Optional[List[str]] = None,
        io_lock: Optional[threading.Lock] = None,
    ):
        """Initialize NDTiff adapter.

        Args:
            dataset: NDTiffDataset instance
            source_id: Unique identifier for this data source
            source_url: URL or path to the data source
            dim_labels: Optional dimension labels (if None, inferred from dataset)
            io_lock: Optional thread lock for IO serialization
        """
        self._dataset = dataset
        self.source_id = source_id
        self._source_url = source_url
        self._source_type = self.SOURCE_TYPE
        self._io_lock = io_lock or threading.Lock()

        # Get dask array from dataset
        self._dask_arr = dataset.as_array()

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
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))

        with self._io_lock:
            return self._dask_arr[slices].compute()

    def get_metadata(self) -> dict:
        """Return dataset summary metadata.

        MicroManager acquisition settings from dataset.summary_metadata.
        """
        if hasattr(self._dataset, "summary_metadata"):
            meta = self._dataset.summary_metadata
            if meta is None:
                return {}
            # Convert to dict if needed
            if hasattr(meta, "model_dump"):
                return meta.model_dump(mode="json")
            elif hasattr(meta, "dict"):
                return meta.dict()
            elif isinstance(meta, dict):
                return meta
        return {}
