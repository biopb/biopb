"""AICSImageIO adapter for vendor microscopy formats (CZI, LIF, ND2, DV, etc.).

This adapter wraps aicsimageio's AICSImage class to provide unified access to
various vendor microscopy formats through the BackendAdapter interface.

Supports:
- Multi-scene files (each scene becomes a separate tensor)
- Lazy loading via dask arrays
- OME-XML metadata conversion

Chunk ID format:
- 4 bytes: array_id length (uint32, big-endian)
- N bytes: array_id (UTF-8)
- M bytes: chunk key (UTF-8, e.g., "0/1/2" for dask chunk indices)

Relies on OS page cache for raw data caching.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

import numpy as np
import pyarrow as pa

from biopb_tensor_server.base import (
    BackendAdapter,
    ChunkEndpoint,
    _decode_chunk_id,
    _encode_chunk_id,
)
from biopb_tensor_server.discovery import SourceClaim
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint, DataSourceDescriptor

if TYPE_CHECKING:
    from aicsimageio import AICSImage
    from biopb_tensor_server.config import SourceConfig


# Extensions supported by aicsimageio
AICS_EXTENSIONS = ['.czi', '.lif', '.nd2', '.dv', '.lsm', '.oif', '.oib', '.xml']


class AicsImageIoAdapter(BackendAdapter):
    """Adapter for aicsimageio-supported vendor formats (CZI, LIF, ND2, DV, etc.).

    Wraps a single scene from an AICSImage instance. Each scene in a multi-scene
    file should be wrapped by a separate adapter instance.

    Supports multifield when scenes have different shapes - in that case,
    each scene becomes a separate tensor within the source.

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - chunk key (UTF-8, e.g., "0/1/2" for dask chunk indices)

    Relies on OS page cache for raw data caching.
    """

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim aicsimageio-supported vendor microscopy format files.

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            SourceClaim if this is a supported vendor format file, None otherwise
        """
        if not path.is_file():
            return None

        name = path.name.lower()
        for ext in AICS_EXTENSIONS:
            if name.endswith(ext):
                return SourceClaim(
                    source_type="aics",
                    primary_path=path,
                    claimed_paths={path},
                )
        return None

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'AicsImageIoAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels

        Returns:
            AicsImageIoAdapter instance
        """
        from aicsimageio import AICSImage

        img = AICSImage(str(source.url))
        # Use first scene as default tensor
        img.set_scene(0)
        return cls(
            img,
            0,  # scene_index
            source.source_id,
            source.dim_labels,
        )

    def __init__(
        self,
        aics_image: "AICSImage",
        scene_index: int,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize AICSImageIO adapter.

        Args:
            aics_image: AICSImage instance, already set to the target scene
            scene_index: Index of the scene this adapter represents
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels (overrides auto-detected dims)
        """
        self._aics_image = aics_image
        self.scene_index = scene_index
        self.array_id = array_id

        # Source-level metadata for DataSourceDescriptor
        self._source_url = str(aics_image.source.path if hasattr(aics_image, 'source') else "")
        self._source_type = "aics"

        # Get dask array for lazy chunk access
        self._dask_data = aics_image.dask_data

        # Dimension labels from dims or use defaults
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            # aicsimageio uses dims like 'TCZYX' or similar
            self.dim_labels = list(aics_image.dims.order)

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        """Read a chunk from the aicsimageio dask array (no caching)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        chunk_key = backend_data.decode('utf-8')
        chunk_idx = tuple(int(i) for i in chunk_key.split('/'))

        # Get chunk sizes from dask array
        chunks = self._dask_data.chunks

        # Compute slice for this chunk
        slices = []
        for d, idx in enumerate(chunk_idx):
            chunk_sizes = chunks[d]
            start = sum(chunk_sizes[:idx])
            stop = start + chunk_sizes[idx]
            slices.append(slice(start, stop))

        # Compute the chunk data from dask array
        data = self._dask_data[tuple(slices)].compute()
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return the TensorDescriptor for this tensor."""
        # Derive chunk shape from first chunk in each dimension
        chunks = self._dask_data.chunks
        chunk_shape = [c[0] if c else 1 for c in chunks]

        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self._dask_data.shape),
            chunk_shape=chunk_shape,
            dtype=self._dask_data.dtype.str,
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        """Get chunk endpoints covering the tensor (or a slice of it).

        Args:
            slice_hint: Optional slice range. If provided, return only chunks
                       that intersect this range.

        Returns:
            List of ChunkEndpoint objects with chunk_id and bounds
        """
        shape = self._dask_data.shape
        chunks = self._dask_data.chunks
        ndim = len(shape)

        endpoints = []

        def iter_chunk_indices(dim: int = 0, prefix: tuple = ()):
            if dim == ndim:
                yield prefix
            else:
                n_chunks = len(chunks[dim])
                for i in range(n_chunks):
                    yield from iter_chunk_indices(dim + 1, prefix + (i,))

        for chunk_idx in iter_chunk_indices():
            # Compute chunk bounds from chunk sizes
            chunk_start = []
            chunk_stop = []
            for d, idx in enumerate(chunk_idx):
                chunk_sizes = chunks[d]
                start = sum(chunk_sizes[:idx])
                stop = start + chunk_sizes[idx]
                chunk_start.append(start)
                chunk_stop.append(min(stop, shape[d]))

            # Filter by slice_hint
            if slice_hint is not None:
                intersects = True
                for d in range(ndim):
                    if chunk_stop[d] <= slice_hint.start[d] or chunk_start[d] >= slice_hint.stop[d]:
                        intersects = False
                        break
                if not intersects:
                    continue

            chunk_key = "/".join(str(i) for i in chunk_idx)
            chunk_id = _encode_chunk_id(self.array_id, chunk_key.encode('utf-8'))

            endpoints.append(ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            ))

        return endpoints

    def get_metadata(self) -> dict:
        """Return OME metadata from the aicsimageio file.

        Uses aicsimageio's ome_metadata property which provides OME-XML
        converted metadata.

        Returns:
            OME metadata as dict, or empty dict if unavailable.
        """
        try:
            ome_meta = self._aics_image.ome_metadata
            if ome_meta is None:
                return {}

            # ome_metadata is typically an OME object from ome-types
            # Convert to dict if it has a model_dump method (pydantic v2)
            # or dict method (pydantic v1)
            if hasattr(ome_meta, 'model_dump'):
                return ome_meta.model_dump()
            elif hasattr(ome_meta, 'dict'):
                return ome_meta.dict()
            elif hasattr(ome_meta, '__dict__'):
                # Fallback: try to extract serializable attributes
                return {
                    k: v for k, v in ome_meta.__dict__.items()
                    if not k.startswith('_')
                }
            return {}
        except Exception:
            return {}