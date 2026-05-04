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
# Now includes .tif and .tiff to handle all TIFF formats (not just OME-TIFF)
AICS_EXTENSIONS = ['.czi', '.lif', '.nd2', '.dv', '.lsm', '.oif', '.oib', '.xml', '.tif', '.tiff']


class AicsImageIoAdapter(BackendAdapter):
    """Adapter for aicsimageio-supported vendor formats (CZI, LIF, ND2, DV, etc.).

    Dual-role adapter:
    - Source-level (scene_index=None): manages metadata, lists all scenes
    - Scene-level (scene_index=int): handles data access for one scene

    Multi-scene files expose each scene as a separate tensor within the source.
    Each scene is identified by its scene_id from img.scenes.

    Supports lazy loading via dask arrays.

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
        """Create source-level adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels

        Returns:
            AicsImageIoAdapter instance (source-level, scene_index=None)
        """
        from aicsimageio import AICSImage

        img = AICSImage(str(source.url))
        return cls(
            img,
            scene_index=None,  # Source-level adapter
            array_id=source.source_id,
            dim_labels=source.dim_labels,
            source_url=str(source.url),
        )

    def __init__(
        self,
        aics_image: "AICSImage",
        scene_index: Optional[int],
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
    ):
        """Initialize AICSImageIO adapter.

        Args:
            aics_image: AICSImage instance
            scene_index: None for source-level, int for scene-level
            array_id: Unique identifier for this tensor/source
            dim_labels: Optional dimension labels (overrides auto-detected dims)
            source_url: Optional source URL
        """
        self._aics_image = aics_image
        self.scene_index = scene_index
        self.array_id = array_id

        # Source-level metadata for DataSourceDescriptor
        if source_url:
            self._source_url = source_url
        elif hasattr(aics_image, 'source') and hasattr(aics_image.source, 'path'):
            self._source_url = str(aics_image.source.path)
        else:
            self._source_url = ""
        self._source_type = "aics"

        # Scene-level: cache dask_data after set_scene (thread-safe)
        if scene_index is not None:
            aics_image.set_scene(scene_index)
            self._dask_data = aics_image.dask_data
            self.dim_labels = dim_labels if dim_labels else list(aics_image.dims.order)
            self._cached_descriptors = None  # Scene-level computes on demand
        else:
            # Source-level: no cached dask_data
            self._dask_data = None
            self.dim_labels = dim_labels  # Used as default for all scenes
            self._cached_descriptors = None  # Cached on first list_tensor_descriptors call

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        """Read a chunk from the aicsimageio dask array (no caching).

        Only valid for scene-level adapters.
        """
        if self._dask_data is None:
            raise ValueError("Cannot get chunk data from source-level adapter")
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
        """Return the TensorDescriptor for this tensor.

        Scene-level: returns descriptor for cached dask_data.
        Source-level: returns descriptor for first scene (default tensor).
        """
        if self._dask_data is not None:
            # Scene-level: use cached dask_data
            chunks = self._dask_data.chunks
            chunk_shape = [c[0] if c else 1 for c in chunks]
            return TensorDescriptor(
                array_id=self.array_id,
                dim_labels=self.dim_labels,
                shape=list(self._dask_data.shape),
                chunk_shape=chunk_shape,
                dtype=self._dask_data.dtype.str,
            )

        # Source-level: return first scene descriptor with source array_id
        descriptors = self.list_tensor_descriptors()
        if descriptors:
            first_desc = descriptors[0]
            # Use source array_id for default tensor
            return TensorDescriptor(
                array_id=self.array_id,
                dim_labels=first_desc.dim_labels,
                shape=first_desc.shape,
                chunk_shape=first_desc.chunk_shape,
                dtype=first_desc.dtype,
            )

        raise ValueError("No scenes available")

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors (scenes) available in this source.

        Scene-level: returns [self.get_tensor_descriptor()].
        Source-level: returns descriptors for all scenes (cached).

        Optimization: Uses OME metadata for shapes without scene switching.
        Chunk info is NOT populated - clients should call get_flight_info
        for accurate per-scene chunk/metadata details.

        Returns:
            List of TensorDescriptor for all scenes in this source
        """
        if self.scene_index is not None:
            return [self.get_tensor_descriptor()]

        # Source-level: use cached descriptors if available
        if self._cached_descriptors is not None:
            return self._cached_descriptors

        descriptors = []
        scene_ids = list(self._aics_image.scenes)

        # Try OME metadata first (much faster - no scene switching)
        try:
            ome_meta = self._aics_image.ome_metadata
            if ome_meta is not None and hasattr(ome_meta, 'images') and len(ome_meta.images) == len(scene_ids):
                # Get dtype from first scene (assumed consistent)
                self._aics_image.set_scene(scene_ids[0])
                dtype = self._aics_image.dask_data.dtype.str

                # Get shapes from OME metadata (no scene switching)
                # OME images are in same order as img.scenes
                for i, im in enumerate(ome_meta.images):
                    px = im.pixels
                    shape = [px.size_t, px.size_c, px.size_z, px.size_y, px.size_x]

                    descriptors.append(TensorDescriptor(
                        array_id=scene_ids[i],  # Use img.scenes ID for get_tensor_adapter
                        dim_labels=self.dim_labels if self.dim_labels else list(self._aics_image.dims.order),
                        shape=shape,
                        chunk_shape=[],  # Not populated - call get_flight_info for chunk info
                        dtype=dtype,
                    ))
        except NotImplementedError:
            # Some formats don't support ome_metadata - fall through to scene switching
            pass

        # Fallback: scene switching (slower but always works)
        if not descriptors:
            for scene_id in scene_ids:
                self._aics_image.set_scene(scene_id)
                dask_data = self._aics_image.dask_data

                descriptors.append(TensorDescriptor(
                    array_id=scene_id,
                    dim_labels=self.dim_labels if self.dim_labels else list(self._aics_image.dims.order),
                    shape=list(dask_data.shape),
                    chunk_shape=[],  # Not populated - call get_flight_info for chunk info
                    dtype=dask_data.dtype.str,
                ))

        # Cache for future calls
        self._cached_descriptors = descriptors
        return descriptors

    def get_tensor_adapter(self, tensor_id: str) -> 'BackendAdapter':
        """Get BackendAdapter for a specific scene within this source.

        Args:
            tensor_id: Scene identifier (scene_id from img.scenes)

        Returns:
            AicsImageIoAdapter for the specified scene
        """
        if self.scene_index is not None:
            # Scene-level: return self if matching, else error
            scene_ids = self._aics_image.scenes
            if self.scene_index < len(scene_ids) and scene_ids[self.scene_index] == tensor_id:
                return self
            raise ValueError(f"Unknown tensor: {tensor_id}")

        # Source-level: find scene index by scene_id
        scene_ids = list(self._aics_image.scenes)
        try:
            scene_idx = scene_ids.index(tensor_id)
        except ValueError:
            raise ValueError(f"Unknown scene: {tensor_id}")

        return AicsImageIoAdapter(
            self._aics_image,
            scene_index=scene_idx,
            array_id=f"{self.array_id}/{tensor_id}",
            dim_labels=self.dim_labels,
            source_url=self._source_url,
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        """Get chunk endpoints covering the tensor (or a slice of it).

        Only valid for scene-level adapters.

        Args:
            slice_hint: Optional slice range. If provided, return only chunks
                       that intersect this range.

        Returns:
            List of ChunkEndpoint objects with chunk_id and bounds
        """
        if self._dask_data is None:
            raise ValueError("Cannot get chunks from source-level adapter")
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
            # Use mode='json' to ensure Enum fields (UnitsElectricPotential, etc.)
            # are serialized to their string representations
            if hasattr(ome_meta, 'model_dump'):
                return ome_meta.model_dump(mode='json')
            elif hasattr(ome_meta, 'dict'):
                return ome_meta.dict(by_alias=False, exclude_none=False)
            elif hasattr(ome_meta, '__dict__'):
                # Fallback: try to extract serializable attributes
                return {
                    k: v for k, v in ome_meta.__dict__.items()
                    if not k.startswith('_')
                }
            return {}
        except Exception:
            return {}