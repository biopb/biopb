"""HDF5 adapter for tensor storage.

Relies on OS page cache for raw data caching.
"""

import struct
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


def _encode_backend_coords(chunk_idx: Tuple[int, ...]) -> bytes:
    """Encode chunk coordinates to bytes (for HDF5)."""
    parts = [struct.pack('>H', len(chunk_idx))]
    for idx in chunk_idx:
        parts.append(struct.pack('>q', idx))  # int64
    return b''.join(parts)


def _decode_backend_coords(data: bytes) -> Tuple[int, ...]:
    """Decode chunk coordinates from bytes (for HDF5)."""
    ndim = struct.unpack('>H', data[:2])[0]
    indices = []
    offset = 2
    for _ in range(ndim):
        idx = struct.unpack('>q', data[offset:offset+8])[0]
        indices.append(idx)
        offset += 8
    return tuple(indices)


class Hdf5Adapter(BackendAdapter):
    """Adapter for HDF5 chunked datasets.

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - uint16 ndim
    - int64[ndim] chunk indices

    Relies on OS page cache for raw data caching.
    """

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim HDF5 files (requires explicit dataset path in config).

        HDF5 files are detected but NOT auto-expanded to tensors because
        they require explicit configuration with dataset path. The claim
        signals this via extra_config['needs_dataset'] = True.

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            SourceClaim with needs_dataset flag, None if not HDF5 file
        """
        if not path.is_file():
            return None

        name = path.name.lower()
        if not (name.endswith('.h5') or name.endswith('.hdf5')):
            return None

        # HDF5 files are claimed but marked as needing explicit dataset config
        # The discovery system will warn about these
        return SourceClaim(
            source_type="hdf5",
            primary_path=path,
            claimed_paths={path},
            extra_config={'needs_dataset': True},
        )

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'Hdf5Adapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels, dataset

        Returns:
            Hdf5Adapter instance

        Raises:
            ValueError: If dataset is not specified
        """
        import h5py

        if source.dataset is None:
            raise ValueError(
                f"HDF5 source '{source.source_id}' requires 'dataset' path in config"
            )

        f = h5py.File(str(source.url), 'r')
        dataset = f[source.dataset]
        return cls(dataset, source.source_id, source.dim_labels)

    def __init__(
        self,
        h5_dataset,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize HDF5 adapter.

        Args:
            h5_dataset: h5py Dataset object
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels
        """
        self.h5_dataset = h5_dataset
        self.source_id = source_id
        self.dim_labels = dim_labels or [f"dim{i}" for i in range(len(h5_dataset.shape))]

        # Source-level metadata for DataSourceDescriptor
        self._source_url = h5_dataset.file.filename if hasattr(h5_dataset, 'file') else ""
        self._source_type = "hdf5"

    def get_chunk_array(self, chunk_id: bytes) -> np.ndarray:
        """Read a chunk from HDF5 as numpy array (no caching - relies on OS page cache)."""
        backend_data = get_backend_data(chunk_id)
        chunk_idx = _decode_backend_coords(backend_data)
        chunks = self.h5_dataset.chunks

        slices = tuple(
            slice(idx * chunks[d], min((idx + 1) * chunks[d], self.h5_dataset.shape[d]))
            for d, idx in enumerate(chunk_idx)
        )

        return self.h5_dataset[slices]

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self.h5_dataset.shape),
            chunk_shape=list(self.h5_dataset.chunks),
            dtype=self.h5_dataset.dtype.str,
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]

    def get_raw_chunk_endpoints(self) -> Iterator[ChunkEndpoint]:
        """Yield all chunk endpoints for this HDF5 dataset."""
        shape = self.h5_dataset.shape
        chunks = self.h5_dataset.chunks
        ndim = len(shape)

        def iter_chunk_indices(dim: int = 0, prefix: Tuple[int, ...] = ()):
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

            chunk_id = encode_chunk_id(self.array_id, _encode_backend_coords(chunk_idx))

            yield ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            )