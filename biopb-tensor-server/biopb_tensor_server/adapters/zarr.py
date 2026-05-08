"""Zarr adapter for tensor storage.

Relies on OS page cache for raw data caching.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import BackendAdapter
from biopb_tensor_server.chunk import (
    ChunkEndpoint,
    encode_chunk_id,
    get_backend_data,
)
from biopb_tensor_server.discovery import SourceClaim

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig


class ZarrAdapter(BackendAdapter):
    """Adapter for Zarr/N5 chunked arrays.

    Chunk ID format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - M bytes: chunk key (UTF-8, e.g., "0/1/2")

    Relies on OS page cache for raw data caching.
    """

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim .zarr directories with .zarray or .zattrs.

        Supports both zarr v2 (.zarray/.zattrs) and zarr v3 (zarr.json).

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            SourceClaim if this is a plain zarr array, None otherwise
        """
        # Must be a directory ending in .zarr
        if not path.is_dir() or not path.name.endswith('.zarr'):
            return None

        # Check for zarr structure files
        zarray_path = path / '.zarray'
        zattrs_path = path / '.zattrs'
        zarr_json_path = path / 'zarr.json'

        # Zarr v2: has .zarray (array metadata)
        if zarray_path.exists():
            return SourceClaim(
                source_type="zarr",
                primary_path=path,
                claimed_paths={path},
            )

        # Zarr v3: has zarr.json
        if zarr_json_path.exists():
            return SourceClaim(
                source_type="zarr",
                primary_path=path,
                claimed_paths={path},
            )

        # If only .zattrs exists, check if it's NOT an OME-Zarr
        if zattrs_path.exists():
            import json
            try:
                with open(zattrs_path) as f:
                    zattrs = json.load(f)
                # If no multiscales, it might be a plain zarr group or array
                if 'multiscales' not in zattrs:
                    # Could be a zarr group - check for array inside
                    # For simplicity, claim it as zarr (will need to open to determine)
                    return SourceClaim(
                        source_type="zarr",
                        primary_path=path,
                        claimed_paths={path},
                    )
            except (json.JSONDecodeError, IOError):
                pass

        return None

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'ZarrAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels

        Returns:
            ZarrAdapter instance
        """
        import zarr

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

    def get_chunk_array(self, chunk_id: bytes) -> np.ndarray:
        """Read a chunk from zarr as numpy array (no caching - relies on OS page cache)."""
        backend_data = get_backend_data(chunk_id)
        chunk_key = backend_data.decode('utf-8')
        chunk_idx = tuple(int(i) for i in chunk_key.split('/'))
        chunks = self.zarr_array.chunks

        # Compute slice for this chunk
        slices = tuple(
            slice(idx * chunks[d], (idx + 1) * chunks[d])
            for d, idx in enumerate(chunk_idx)
        )

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

    def get_raw_chunk_endpoints(self) -> Iterator[ChunkEndpoint]:
        """Yield all chunk endpoints for this zarr array."""
        shape = self.zarr_array.shape
        chunks = self.zarr_array.chunks
        ndim = len(shape)

        def iter_chunk_indices(dim: int = 0, prefix: tuple = ()):
            if dim == ndim:
                yield prefix
            else:
                n_chunks = (shape[dim] + chunks[dim] - 1) // chunks[dim]
                for i in range(n_chunks):
                    yield from iter_chunk_indices(dim + 1, prefix + (i,))

        for chunk_idx in iter_chunk_indices():
            chunk_start = [idx * chunks[d] for d, idx in enumerate(chunk_idx)]
            chunk_stop = [
                min((idx + 1) * chunks[d], shape[d])
                for d, idx in enumerate(chunk_idx)
            ]

            chunk_key = "/".join(str(i) for i in chunk_idx)
            chunk_id = encode_chunk_id(self.array_id, chunk_key.encode('utf-8'))

            yield ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            )