"""OME-TIFF adapters for tensor storage."""

import struct
from typing import List, Optional, Tuple
from functools import lru_cache

import numpy as np
import pyarrow as pa

from biopb.tensor.base import (
    BackendAdapter,
    ChunkEndpoint,
    _decode_chunk_id,
    _encode_chunk_id,
    _chunks_intersect,
)
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint


# =============================================================================
# OME-TIFF encoding/decoding helpers
# =============================================================================

def _encode_ome_tile(ifd_index: int, tile_indices: Tuple[int, ...]) -> bytes:
    """Encode IFD index and tile indices to bytes (for OME-TIFF)."""
    parts = [
        struct.pack('>H', ifd_index),
        struct.pack('>H', len(tile_indices))
    ]
    for idx in tile_indices:
        parts.append(struct.pack('>q', idx))
    return b''.join(parts)


def _decode_ome_tile(data: bytes) -> Tuple[int, Tuple[int, ...]]:
    """Decode IFD index and tile indices from bytes (for OME-TIFF)."""
    ifd_index = struct.unpack('>H', data[:2])[0]
    ndim = struct.unpack('>H', data[2:4])[0]
    indices = []
    offset = 4
    for _ in range(ndim):
        idx = struct.unpack('>q', data[offset:offset+8])[0]
        indices.append(idx)
        offset += 8
    return ifd_index, tuple(indices)


def _encode_ome_multifile_tile(file_index: int, ifd_index: int, tile_indices: Tuple[int, ...]) -> bytes:
    """Encode file index, IFD index and tile indices for multi-file OME-TIFF."""
    parts = [
        struct.pack('>H', file_index),
        struct.pack('>H', ifd_index),
        struct.pack('>H', len(tile_indices))
    ]
    for idx in tile_indices:
        parts.append(struct.pack('>q', idx))
    return b''.join(parts)


def _decode_ome_multifile_tile(data: bytes) -> Tuple[int, int, Tuple[int, ...]]:
    """Decode file index, IFD index and tile indices for multi-file OME-TIFF."""
    file_index = struct.unpack('>H', data[:2])[0]
    ifd_index = struct.unpack('>H', data[2:4])[0]
    ndim = struct.unpack('>H', data[4:6])[0]
    indices = []
    offset = 6
    for _ in range(ndim):
        idx = struct.unpack('>q', data[offset:offset+8])[0]
        indices.append(idx)
        offset += 8
    return file_index, ifd_index, tuple(indices)


# =============================================================================
# OmeTiffAdapter - Single file OME-TIFF
# =============================================================================

class OmeTiffAdapter(BackendAdapter):
    """Adapter for OME-TIFF files using tifffile.

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - uint16 ifd_index (page index)
    - uint16 ndim
    - int64[ndim] tile indices

    Uses LRU caching for decoded tiles to avoid repeated decompression.
    """

    def __init__(
        self,
        tiff_file,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        cache_size: int = 256
    ):
        """Initialize OME-TIFF adapter.

        Args:
            tiff_file: tifffile.TiffFile object
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
            cache_size: Number of tiles to cache (default 256)
        """
        self.tiff_file = tiff_file
        self.array_id = array_id
        self.cache_size = cache_size

        # Get series info
        self.series = tiff_file.series[0]
        self.series_shape = self.series.shape
        self.series_dims = self.series.dims

        # Get tile info from first page
        first_page = tiff_file.pages[0]
        if not first_page.is_tiled:
            raise ValueError("OME-TIFF must be tiled")

        self.tile_width = first_page.tilewidth
        self.tile_length = first_page.tilelength
        self.tiles_per_row = (first_page.shape[1] + self.tile_width - 1) // self.tile_width
        self.tiles_per_col = (first_page.shape[0] + self.tile_length - 1) // self.tile_length

        # Derive chunk shape (tile_y, tile_x) for each plane
        self.chunk_shape = [self.tile_length, self.tile_width]

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            # Infer from series dims
            self.dim_labels = [d if isinstance(d, str) else str(d) for d in self.series_dims]

        # Full shape includes all planes
        # series_shape is (n_planes, height, width) or similar
        self.full_shape = list(self.series_shape)

        # Adjust chunk_shape to match full_shape dimensions
        # Non-spatial dimensions (like channel/time) have chunk size = 1
        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self.chunk_shape

        # Initialize LRU cache for decoded tiles
        # Cache key is the chunk_id (bytes, which is hashable)
        self._get_decoded_tile = lru_cache(maxsize=cache_size)(self._get_decoded_tile_uncached)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        # Get dtype from first page
        first_page = self.tiff_file.pages[0]
        dtype = first_page.dtype

        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=str(dtype),
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        endpoints = []

        # Iterate over all IFDs (pages/planes)
        for ifd_index in range(len(self.tiff_file.pages)):
            page = self.tiff_file.pages[ifd_index]

            if not page.is_tiled:
                continue

            # Compute plane offset in full array
            # For XYZCT layout, each IFD corresponds to one (C, T, Z) combo
            # This depends on the specific OME-TIFF structure
            plane_offset = ifd_index

            # Iterate over tiles in this page
            for tile_row in range(self.tiles_per_col):
                for tile_col in range(self.tiles_per_row):
                    # Compute chunk bounds
                    y_start = tile_row * self.tile_length
                    y_stop = min((tile_row + 1) * self.tile_length, page.shape[0])
                    x_start = tile_col * self.tile_width
                    x_stop = min((tile_col + 1) * self.tile_width, page.shape[1])

                    # Full bounds including plane dimension
                    chunk_start = [plane_offset, y_start, x_start]
                    chunk_stop = [plane_offset + 1, y_stop, x_stop]

                    if slice_hint is not None:
                        if not _chunks_intersect(
                            chunk_start, chunk_stop,
                            list(slice_hint.start), list(slice_hint.stop)
                        ):
                            continue

                    chunk_id = _encode_chunk_id(self.array_id, _encode_ome_tile(ifd_index, (tile_row, tile_col)))

                    endpoints.append(ChunkEndpoint(
                        chunk_id=chunk_id,
                        bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
                    ))

        return endpoints

    def _get_decoded_tile_uncached(self, chunk_id: bytes) -> np.ndarray:
        """Decode a tile from the TIFF file (uncached)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        ifd_index, tile_indices = _decode_ome_tile(backend_data)
        tile_row, tile_col = tile_indices

        page = self.tiff_file.pages[ifd_index]

        # Compute tile index
        tile_idx = tile_row * self.tiles_per_row + tile_col

        # Read tile using tifffile's low-level API
        offset = page.dataoffsets[tile_idx]
        bytecount = page.databytecounts[tile_idx]

        fh = self.tiff_file.filehandle
        fh.seek(offset)
        raw_data = fh.read(bytecount)

        # Decode the tile
        decoded = page.decode(raw_data, tile_idx)
        data = decoded[0].squeeze()  # Remove singleton dimensions

        # Ensure 2D output
        if data.ndim == 3:
            data = data[0]

        return data

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        # Use cached tile decoding
        data = self._get_decoded_tile(chunk_id)
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])


# =============================================================================
# MultiFileOmeTiffAdapter - Multi-file OME-TIFF
# =============================================================================

class MultiFileOmeTiffAdapter(BackendAdapter):
    """Adapter for multi-file OME-TIFF datasets (e.g., Micro-Manager format).

    Handles datasets where multiple TIFF files form a single logical image:
    - sample/img_0.ome.tiff (channel 0)
    - sample/img_1.ome.tiff (channel 1)
    - sample/_metadata.txt (OME-XML metadata)

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - uint16 file_index (which file in the series)
    - uint16 ifd_index (page index within file)
    - uint16 ndim
    - int64[ndim] tile indices

    Uses LRU caching for decoded tiles.
    """

    def __init__(
        self,
        directory: str,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        cache_size: int = 256
    ):
        """Initialize multi-file OME-TIFF adapter.

        Args:
            directory: Path to directory containing multi-file dataset
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
            cache_size: Number of tiles to cache (default 256)
        """
        import tifffile
        from pathlib import Path

        self.directory = Path(directory)
        self.array_id = array_id
        self.cache_size = cache_size

        # Find all OME-TIFF files in directory
        patterns = ["img_*.ome.tiff", "img_*.ome.tif", "*.ome.tiff", "*.ome.tif"]
        tiff_files = []
        for pattern in patterns:
            tiff_files.extend(sorted(self.directory.glob(pattern)))

        if not tiff_files:
            raise ValueError(f"No OME-TIFF files found in {directory}")

        # Open first file - tifffile auto-discovers related files via OME-XML
        self.tiff_file = tifffile.TiffFile(str(tiff_files[0]))

        # Get unified series info (spans all files)
        if len(self.tiff_file.series) == 0:
            raise ValueError("No series found in OME-TIFF dataset")

        self.series = self.tiff_file.series[0]
        self.series_shape = self.series.shape
        self.series_dims = self.series.dims

        # Get tile info from first page of first file
        first_page = self.tiff_file.pages[0]

        # Check if tiled - if not, we'll handle as single "tile" per plane
        if first_page.is_tiled:
            self.is_tiled = True
            self.tile_width = first_page.tilewidth
            self.tile_length = first_page.tilelength
            self.tiles_per_row = (first_page.shape[1] + self.tile_width - 1) // self.tile_width
            self.tiles_per_col = (first_page.shape[0] + self.tile_length - 1) // self.tile_length
            self.chunk_shape = [self.tile_length, self.tile_width]
        else:
            # Non-tiled: each IFD is a single chunk (the whole plane)
            self.is_tiled = False
            self.tile_width = first_page.shape[1]
            self.tile_length = first_page.shape[0]
            self.tiles_per_row = 1
            self.tiles_per_col = 1
            self.chunk_shape = [first_page.shape[0], first_page.shape[1]]

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            self.dim_labels = [d if isinstance(d, str) else str(d) for d in self.series_dims]

        # Full shape
        self.full_shape = list(self.series_shape)
        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self.chunk_shape

        # Build file index for IFD access
        # Each file contains multiple IFDs (planes)
        # We need to map global IFD index to (file_index, local_ifd_index)
        self._file_ifd_map = []  # List of (file_path, local_ifd_count)
        self._ifd_to_file = []   # Maps global IFD index to (file_index, local_ifd_index)

        # For simple case: assume each file has same number of IFDs
        # More complex: parse OME-XML to understand C,Z,T distribution
        global_ifd_index = 0
        for file_path in tiff_files:
            with tifffile.TiffFile(str(file_path)) as tf:
                n_pages = len(tf.pages)
                self._file_ifd_map.append((file_path, n_pages))
                for local_idx in range(n_pages):
                    self._ifd_to_file.append((len(self._file_ifd_map) - 1, local_idx))
                global_ifd_index += n_pages

        self._total_ifds = global_ifd_index
        self._tiff_files = tiff_files

        # Initialize LRU cache
        self._get_decoded_tile = lru_cache(maxsize=cache_size)(self._get_decoded_tile_uncached)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        # Get dtype from first page
        first_page = self.tiff_file.pages[0]
        dtype = first_page.dtype

        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=str(dtype),
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        import tifffile

        endpoints = []

        # Iterate over all IFDs across all files
        for global_ifd_index in range(self._total_ifds):
            file_index, local_ifd_index = self._ifd_to_file[global_ifd_index]
            file_path = self._tiff_files[file_index]

            # Open file to get page info
            with tifffile.TiffFile(str(file_path)) as tf:
                page = tf.pages[local_ifd_index]

                # Use pre-computed tile counts (handles both tiled and non-tiled)
                tiles_per_row = self.tiles_per_row
                tiles_per_col = self.tiles_per_col

                for tile_row in range(tiles_per_col):
                    for tile_col in range(tiles_per_row):
                        y_start = tile_row * self.tile_length
                        y_stop = min((tile_row + 1) * self.tile_length, page.shape[0])
                        x_start = tile_col * self.tile_width
                        x_stop = min((tile_col + 1) * self.tile_width, page.shape[1])

                        chunk_start = [global_ifd_index, y_start, x_start]
                        chunk_stop = [global_ifd_index + 1, y_stop, x_stop]

                        if slice_hint is not None:
                            if not _chunks_intersect(
                                chunk_start, chunk_stop,
                                list(slice_hint.start), list(slice_hint.stop)
                            ):
                                continue

                        chunk_id = _encode_chunk_id(
                            self.array_id,
                            _encode_ome_multifile_tile(file_index, local_ifd_index, (tile_row, tile_col))
                        )

                        endpoints.append(ChunkEndpoint(
                            chunk_id=chunk_id,
                            bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
                        ))

        return endpoints

    def _get_decoded_tile_uncached(self, chunk_id: bytes) -> np.ndarray:
        """Decode a tile from the multi-file dataset (uncached)."""
        import tifffile

        _, backend_data = _decode_chunk_id(chunk_id)
        file_index, local_ifd_index, tile_indices = _decode_ome_multifile_tile(backend_data)
        tile_row, tile_col = tile_indices

        file_path = self._tiff_files[file_index]

        with tifffile.TiffFile(str(file_path)) as tf:
            page = tf.pages[local_ifd_index]

            if self.is_tiled:
                # Tiled: read specific tile
                tiles_per_row = (page.shape[1] + self.tile_width - 1) // self.tile_width
                tile_idx = tile_row * tiles_per_row + tile_col

                offset = page.dataoffsets[tile_idx]
                bytecount = page.databytecounts[tile_idx]

                fh = tf.filehandle
                fh.seek(offset)
                raw_data = fh.read(bytecount)

                decoded = page.decode(raw_data, tile_idx)
                data = decoded[0].squeeze()
            else:
                # Non-tiled: read entire page as single "tile"
                data = page.asarray()

            if data.ndim == 3:
                data = data[0]

            return data

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        data = self._get_decoded_tile(chunk_id)
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])