"""OME-Zarr adapter for tensor storage.

Relies on OS page cache for raw data caching.
"""

import json
import os
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa

from biopb_tensor_server.base import (
    BackendAdapter,
    ChunkEndpoint,
    TensorReadPlan,
    _decode_chunk_id,
    _encode_chunk_id,
    _chunks_intersect,
    _normalize_reduction_method,
)
from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint, TensorReadOptions


class OmeZarrAdapter(BackendAdapter):
    """Adapter for OME-Zarr (OME-NGFF) datasets.

    Extends ZarrAdapter with OME metadata support:
    - multiscales: Multiple resolution levels
    - axes: Dimension labels with types (channel, space, time)
    - coordinate_transformations: Physical scales
    - omero: Channel colors, names

    Chunk ID format: Same as ZarrAdapter
    - array_id prefix
    - chunk key (UTF-8, e.g., "0/1/2")

    Relies on OS page cache for raw data caching.
    """

    def __init__(
        self,
        zarr_array,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        resolution_level: int = 0
    ):
        """Initialize OME-Zarr adapter.

        Args:
            zarr_array: Zarr array object (from specific resolution level)
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels (overrides OME metadata)
            resolution_level: Which resolution level to use (default 0)
        """
        self.zarr_array = zarr_array
        self.array_id = array_id
        self.resolution_level = resolution_level

        # Try to read OME metadata from .zattrs
        self.ome_metadata = {}
        self.axes = []
        self.channel_names = []

        # Read .zattrs from the zarr group root
        try:
            store = zarr_array.store
            store_str = str(store)
            if store_str.startswith('file://'):
                store_path = str(urlparse(store_str).path)
            elif hasattr(store, 'path'):
                # DirectoryStore has 'path' attribute
                store_path = str(store.path)
            elif hasattr(store, 'root'):
                store_path = str(store.root)
            else:
                store_path = store_str

            # .zattrs is at the group root level
            zattrs_path = os.path.join(store_path, '.zattrs')
            if not os.path.exists(zattrs_path):
                # Check parent directory (group root)
                parent_path = os.path.dirname(store_path.rstrip('/'))
                zattrs_path = os.path.join(parent_path, '.zattrs')

            if os.path.exists(zattrs_path):
                with open(zattrs_path) as f:
                    zattrs = json.load(f)
                    self.ome_metadata = zattrs
                    if 'multiscales' in zattrs:
                        self.axes = zattrs['multiscales'][0].get('axes', [])
                        if 'omero' in zattrs:
                            channels = zattrs['omero'].get('channels', [])
                            self.channel_names = [ch.get('label', f'ch{i}') for i, ch in enumerate(channels)]
        except (json.JSONDecodeError, KeyError, FileNotFoundError, AttributeError):
            pass

        # Set dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        elif self.axes:
            self.dim_labels = [
                ax.get('name', f'dim{i}') if isinstance(ax, dict) else str(ax)
                for i, ax in enumerate(self.axes)
            ]
        else:
            self.dim_labels = [f"dim{i}" for i in range(zarr_array.ndim)]

        # Cache for level adapters (precomputed pyramid levels)
        self._level_adapters: dict = {}

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        """Read a chunk from zarr (no caching - relies on OS page cache)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        chunk_key = backend_data.decode('utf-8')
        chunk_idx = tuple(int(i) for i in chunk_key.split('/'))
        chunks = self.zarr_array.chunks

        slices = tuple(
            slice(idx * chunks[d], (idx + 1) * chunks[d])
            for d, idx in enumerate(chunk_idx)
        )

        data = self.zarr_array[slices]
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self.zarr_array.shape),
            chunk_shape=list(self.zarr_array.chunks),
            dtype=self.zarr_array.dtype.str,
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        shape = self.zarr_array.shape
        chunks = self.zarr_array.chunks
        ndim = len(shape)

        endpoints = []

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

            if slice_hint is not None:
                if not _chunks_intersect(
                    chunk_start, chunk_stop,
                    list(slice_hint.start), list(slice_hint.stop)
                ):
                    continue

            chunk_key = "/".join(str(i) for i in chunk_idx)
            chunk_id = _encode_chunk_id(self.array_id, chunk_key.encode('utf-8'))

            endpoints.append(ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            ))

        return endpoints

    def get_ome_metadata(self) -> dict:
        """Return OME-Zarr metadata."""
        return self.ome_metadata

    def get_channel_info(self) -> List[dict]:
        """Return channel information from OME metadata."""
        if not self.channel_names:
            return [{'label': f'ch{i}'} for i in range(self.zarr_array.shape[0])]

        omero = self.ome_metadata.get('omero', {})
        channels = omero.get('channels', [])

        result = []
        for i, name in enumerate(self.channel_names):
            ch_info = {'label': name}
            if i < len(channels):
                ch_info.update(channels[i])
            result.append(ch_info)
        return result

    def get_metadata(self) -> dict:
        """Return OME-Zarr .zattrs content directly."""
        return self.ome_metadata

    def get_scaled_read_plan(
        self,
        scale_hint: Tuple[int, ...],
        slice_hint: Optional[SliceHint],
        read_options: Optional[TensorReadOptions],
    ) -> TensorReadPlan:
        """Return read plan for requested scale.

        Supports "precompute" method to use precomputed pyramid levels.
        Falls back to virtual scaling for other methods.
        """
        reduction_method = _normalize_reduction_method(
            read_options.reduction_method if read_options else None
        )

        # "precompute" method: use precomputed level if exact match
        if reduction_method == 'precompute':
            level_path = self._find_level_for_scale(scale_hint)

            if level_path is None:
                raise ValueError(
                    f"No precomputed level matching scale_hint {tuple(scale_hint)}."
                )

            # Get scale for slice conversion
            level_scale = self._get_level_scale(level_path)

            # Convert slice from base coords to level coords
            level_slice = self._convert_slice_to_level(slice_hint, level_scale)

            return self._plan_from_precomputed(level_path, level_slice)

        # Other methods: use default virtual scaling
        return super().get_scaled_read_plan(scale_hint, slice_hint, read_options)

    def _find_level_for_scale(self, scale_hint: Tuple[int, ...]) -> Optional[str]:
        """Find precomputed level with exact scale match."""
        multiscales = self.ome_metadata.get('multiscales', [])
        if not multiscales:
            return None

        for ds in multiscales[0].get('datasets', []):
            for t in ds.get('coordinateTransformations', []):
                if t.get('type') == 'scale':
                    scale = tuple(int(s) for s in t.get('scale', []))
                    if scale == scale_hint:
                        return ds.get('path')

        return None

    def _get_level_scale(self, level_path: str) -> Tuple[int, ...]:
        """Extract scale for a specific level path."""
        multiscales = self.ome_metadata.get('multiscales', [])
        if not multiscales:
            return tuple()

        for ds in multiscales[0].get('datasets', []):
            if ds.get('path') == level_path:
                for t in ds.get('coordinateTransformations', []):
                    if t.get('type') == 'scale':
                        return tuple(int(s) for s in t.get('scale', []))

        return tuple()

    def _convert_slice_to_level(
        self,
        slice_hint: Optional[SliceHint],
        level_scale: Tuple[int, ...],
    ) -> Optional[SliceHint]:
        """Convert slice from base coordinates to level coordinates."""
        if slice_hint is None:
            return None

        level_start = [s // sc for s, sc in zip(slice_hint.start, level_scale)]
        level_stop = [s // sc for s, sc in zip(slice_hint.stop, level_scale)]
        return SliceHint(start=level_start, stop=level_stop)

    def _plan_from_precomputed(
        self,
        level_path: str,
        level_slice: Optional[SliceHint],
    ) -> TensorReadPlan:
        """Create read plan from precomputed level."""
        level_adapter = self.get_level_adapter(level_path)
        endpoints = level_adapter.get_chunk_endpoints(level_slice)

        level_desc = level_adapter.get_tensor_descriptor()

        logical_desc = TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(level_desc.shape),
            chunk_shape=list(level_desc.chunk_shape),
            dtype=level_desc.dtype,
        )

        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=endpoints)

    def get_level_adapter(self, path: str) -> ZarrAdapter:
        """Get adapter for a specific precomputed level.

        Args:
            path: Level path (e.g., "0", "1", "2" for OME-Zarr)

        Returns:
            ZarrAdapter for the level array
        """
        if path in self._level_adapters:
            return self._level_adapters[path]

        # Open the level array
        level_arr = self._open_level_array(path)

        # Create adapter for this level
        level_adapter = ZarrAdapter(
            level_arr,
            array_id=f"{self.array_id}/{path}",
            dim_labels=self.dim_labels,
        )
        self._level_adapters[path] = level_adapter
        return level_adapter

    def _open_level_array(self, path: str):
        """Open the Zarr array at the given level path (relative to group root)."""
        import zarr

        store = self.zarr_array.store

        store_str = str(store.path if hasattr(store, 'path') else store)

        if store_str.startswith('file://'):
            store_path = urlparse(store_str).path
        else:
            store_path = store_str

        # Navigate to the group root
        current_path = store_path.rstrip('/')
        while current_path and current_path != '/':
            if os.path.exists(os.path.join(current_path, '.zattrs')):
                break
            current_path = os.path.dirname(current_path)

        # Join with the target level path
        level_path = os.path.join(current_path, path)

        return zarr.open_array(level_path, mode='r')