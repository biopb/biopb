"""AICSImageIO adapter for vendor microscopy formats (CZI, LIF, ND2, DV, etc.).

This adapter wraps aicsimageio's AICSImage class to provide unified access to
various vendor microscopy formats through the BackendAdapter interface.

Supports:
- Multi-scene files (each scene becomes a separate tensor)
- Lazy loading via dask arrays
- OME-XML metadata conversion

Chunk ID format:
- array_id + bounds encoding (start, stop coordinates)

Relies on OS page cache for raw data caching.
"""

import importlib
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import BackendAdapter
from biopb_tensor_server.chunk import ChunkEndpoint
from biopb_tensor_server.discovery import SourceClaim

if TYPE_CHECKING:
    from aicsimageio import AICSImage

    from biopb_tensor_server.config import SourceConfig


def _get_available_extensions() -> List[str]:
    """Detect file extensions with at least one importable aicsimageio reader.

    Uses aicsimageio's FORMAT_IMPLEMENTATIONS to check which readers are
    actually available (have their required dependencies installed).

    Returns:
        Sorted list of extensions (e.g., ['.czi', '.lif', '.tif', '.tiff'])
    """
    import aicsimageio.formats as formats

    extensions = []
    for ext, reader_paths in formats.FORMAT_IMPLEMENTATIONS.items():
        for reader_path in reader_paths:
            try:
                module_path, cls_name = reader_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                getattr(module, cls_name)
                extensions.append(f'.{ext}')
                break
            except (ImportError, AttributeError, ModuleNotFoundError):
                continue
    return sorted(set(extensions))


# Build extension list at module load time based on installed readers
AICS_EXTENSIONS = _get_available_extensions()


class AicsImageIoAdapter(BackendAdapter):
    """Adapter for aicsimageio-supported vendor formats (CZI, LIF, ND2, DV, etc.).

    Dual-role adapter:
    - Source-level (scene_index=None): manages metadata, lists all scenes
    - Scene-level (scene_index=int): handles data access for one scene

    Multi-scene files expose each scene as a separate tensor within the source.
    Each scene is identified by its scene_id from img.scenes.

    Multi-tensor source: Each scene is a separate tensor.
    Use get_tensor_adapter(scene_id) to access specific scenes.

    Supports lazy loading via dask arrays.

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - chunk key (UTF-8, e.g., "0/1/2" for dask chunk indices)

    Relies on OS page cache for raw data caching.
    """

    # Multi-tensor source: has multiple scenes
    _single_tensor_source = False

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
            source_id=source.source_id,
            dim_labels=source.dim_labels,
            source_url=str(source.url),
        )

    def __init__(
        self,
        aics_image: "AICSImage",
        scene_index: Optional[int],
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        io_lock: Optional[threading.Lock] = None,
    ):
        """Initialize AICSImageIO adapter.

        Args:
            aics_image: AICSImage instance
            scene_index: None for source-level, int for scene-level
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (overrides auto-detected dims)
            source_url: Optional source URL
            io_lock: Optional thread lock for IO serialization. Source-level
                     adapters create a new lock if None; scene-level adapters
                     receive the lock from the source-level adapter.
        """
        self._aics_image = aics_image
        self.scene_index = scene_index
        self.source_id = source_id

        # Thread lock for serializing IO operations
        # Source-level creates lock, scene-level receives from source
        if io_lock is not None:
            self._io_lock = io_lock
        else:
            self._io_lock = threading.Lock()

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

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from aicsimageio dask array.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape or called on source-level adapter
        """
        if self._dask_data is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))
        with self._io_lock:
            return self._dask_data[slices].compute()

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this adapter.

        For scene-level adapters (scene_index is set): returns descriptor for that scene.
        For source-level adapters (scene_index=None): returns first scene descriptor.
        """
        if self._dask_data is not None:
            # Scene-level: compute from dask array
            chunks = self._dask_data.chunks
            chunk_shape = [max(c) for c in chunks]
            return TensorDescriptor(
                array_id=self.array_id,
                dim_labels=self.dim_labels if self.dim_labels else [],
                shape=list(self._dask_data.shape),
                chunk_shape=chunk_shape,
                dtype=self._dask_data.dtype.str,
            )
        # Source-level: return first scene descriptor
        return self.list_tensor_descriptors()[0]

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors (scenes) available in this source.

        Optimization: Uses OME metadata for shapes without scene switching.
        Chunk info is NOT populated - clients should call get_flight_info
        for accurate per-scene chunk/metadata details.

        Returns:
            List of TensorDescriptor for all scenes in this source
        """
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
            AicsImageIoAdapter for the specified scene, with tensor context set
        """
        # Source-level: lazy initialize tensor level adapters
        scene_ids = list(self._aics_image.scenes)
        try:
            scene_idx = scene_ids.index(tensor_id)
        except ValueError:
            raise ValueError(f"Unknown scene: {tensor_id}")
        
        if hasattr(self, '_tensor_adapters'):
            # Check if adapter already exists for this scene
            if tensor_id in self._tensor_adapters:
                return self._tensor_adapters[tensor_id]
        else:
            self._tensor_adapters = {}

        adapter = AicsImageIoAdapter(
            self._aics_image,
            scene_index=scene_idx,
            source_id=self.source_id,
            dim_labels=self.dim_labels,
            source_url=self._source_url,
            io_lock=self._io_lock,
        )
        _ = super(AicsImageIoAdapter, adapter).get_tensor_adapter(tensor_id)  # Set tensor context in base class
        self._tensor_adapters[tensor_id] = adapter

        return adapter

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