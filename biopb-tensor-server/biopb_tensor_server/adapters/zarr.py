"""Zarr adapter for tensor storage.

Relies on OS page cache for raw data caching.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.chunk import ChunkEndpoint
from biopb_tensor_server.discovery import ClaimContext, SourceClaim, is_remote_url

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState
    from biopb_tensor_server.remote import RemoteStore


class ZarrAdapter(SourceAdapter, TensorAdapter):
    """Adapter for Zarr/N5 chunked arrays.

    Supports both local filesystem and remote storage (S3, GCS, etc.) via fsspec.
    For remote storage, uses zarr.FSStore with fsspec filesystem.

    """

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
        if not ctx.is_dir() or not ctx.name.endswith('.zarr'):
            return None

        # Check for zarr structure files
        has_zarray = ctx.join('.zarray').exists()
        has_zarr_json = ctx.join('zarr.json').exists()
        has_zattrs = ctx.join('.zattrs').exists()

        # Zarr v2: has .zarray (array metadata)
        if has_zarray or has_zarr_json:
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type="zarr",
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
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
            zattrs_ctx = ctx.join('.zattrs')
            if not zattrs_ctx.is_resident():
                state.try_claim_path(ctx.path_str)
                return SourceClaim(
                    source_type="zarr",
                    primary_path=ctx.path_str,
                    is_remote=ctx.is_remote,
                    unresolved=True,
                )
            try:
                zattrs = json.loads(ctx.read_text('.zattrs'))
                # If no multiscales, it might be a plain zarr group or array
                if 'multiscales' not in zattrs:
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
    
    def get_tensor_adapter(self, tensor_id):
        return self

    @classmethod
    def create_from_config(
        cls,
        source: 'SourceConfig',
        credentials_config: Optional[Any] = None,
    ) -> 'ZarrAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            ZarrAdapter instance
        """
        import zarr
        from zarr.storage import FSStore
        from biopb_tensor_server.remote import RemoteStore

        if source.is_remote:
            # Remote storage: use RemoteStore for filesystem creation
            store = RemoteStore.from_config(
                source.url,
                credentials_config=credentials_config,
                profile_name=source.credentials_profile,
            )
            zarr_store = FSStore(store.path, fs=store.fs)
            arr = zarr.open_array(zarr_store, mode='r')
        else:
            # Local filesystem
            path = Path(source.url)
            arr = zarr.open_array(str(path), mode='r')

        return cls(arr, source.source_id, source.dim_labels)

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
            dim_labels: Optional dimension labels
        """
        self.zarr_array = zarr_array
        self.source_id = source_id
        self.dim_labels = dim_labels or [f"dim{i}" for i in range(zarr_array.ndim)]

        # Source-level metadata for DataSourceDescriptor
        self._source_url = str(zarr_array.store.path if hasattr(zarr_array.store, 'path') else str(zarr_array.store))
        self._source_type = "zarr"

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
        desc = self.get_tensor_descriptor()
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))
        return self.zarr_array[slices]

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
            src_slices = tuple(slice(0, min(d, es)) for d, es in zip(data.shape, expected_shape))
            padded[src_slices] = data[src_slices]
            data = padded

        self.zarr_array[slices] = data

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self.zarr_array.shape),
            chunk_shape=list(self.zarr_array.chunks),
            dtype=self.zarr_array.dtype.str,
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]