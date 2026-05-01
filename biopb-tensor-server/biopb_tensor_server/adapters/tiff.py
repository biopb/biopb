"""OME-TIFF adapters for tensor storage.

Relies on OS page cache for raw data caching.
"""

import struct
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import pyarrow as pa

from biopb_tensor_server.base import (
    BackendAdapter,
    ChunkEndpoint,
    _decode_chunk_id,
    _encode_chunk_id,
    _chunks_intersect,
)
from biopb_tensor_server.discovery import SourceClaim, get_file_identity
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig


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


def _elementtree_to_dict(element) -> dict:
    """Convert ElementTree element to JSON-serializable dict.

    Handles nested elements, attributes, and text content.
    Returns a dict suitable for json.dumps().

    Args:
        element: xml.etree.ElementTree.Element object

    Returns:
        JSON-serializable dictionary representation
    """
    result = {}

    # Add attributes
    for key, value in element.attrib.items():
        result[f"@{key}"] = value

    # Process child elements
    for child in element:
        child_data = _elementtree_to_dict(child)
        child_tag = child.tag

        # Handle multiple elements with same tag
        if child_tag in result:
            # Convert to list if not already
            if not isinstance(result[child_tag], list):
                result[child_tag] = [result[child_tag]]
            result[child_tag].append(child_data)
        else:
            result[child_tag] = child_data

    # Add text content if present
    if element.text and element.text.strip():
        if result:
            result["#text"] = element.text.strip()
        else:
            return element.text.strip()

    return result if result else ""


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

    Relies on OS page cache for raw data caching.
    """

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim single OME-TIFF files.

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            SourceClaim if this is an OME-TIFF file, None otherwise
        """
        if not path.is_file():
            return None

        name = path.name.lower()
        # OME-TIFF extensions
        if name.endswith('.ome.tiff') or name.endswith('.ome.tif'):
            return SourceClaim(
                source_type="ome-tiff",
                primary_path=path,
                claimed_paths={path},
            )

        # Plain TIFF files - also claim as ome-tiff (tifffile handles them)
        if name.endswith('.tiff') or name.endswith('.tif'):
            return SourceClaim(
                source_type="ome-tiff",
                primary_path=path,
                claimed_paths={path},
            )

        return None

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'OmeTiffAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels

        Returns:
            OmeTiffAdapter instance
        """
        import tifffile

        tiff = tifffile.TiffFile(str(source.url))
        return cls(tiff, source.source_id, source.dim_labels)

    def __init__(
        self,
        tiff_file,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize OME-TIFF adapter.

        Args:
            tiff_file: tifffile.TiffFile object
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
        """
        self.tiff_file = tiff_file
        self.array_id = array_id
        self._io_lock = threading.Lock()

        # Source-level metadata for DataSourceDescriptor
        self._source_url = tiff_file.filename if hasattr(tiff_file, 'filename') else ""
        self._source_type = "ome-tiff"

        # Get series info
        self.series = tiff_file.series[0]
        self.series_shape = self.series.shape
        self.series_dims = self.series.dims

        # Get tile info from first page
        first_page = tiff_file.pages[0]
        if not first_page.is_tiled:
            raise ValueError("OME-TIFF pages must be tiled")

        self.is_tiled = True
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
        n_pages = len(self.tiff_file.pages)

        for ifd_index in range(n_pages):
            # Compute plane offset in full array
            plane_offset = ifd_index

            # Iterate over tiles in this page
            for tile_row in range(self.tiles_per_col):
                for tile_col in range(self.tiles_per_row):
                    # Compute chunk bounds using precomputed tile dimensions
                    y_start = tile_row * self.tile_length
                    y_stop = min((tile_row + 1) * self.tile_length, self.series_shape[-2])
                    x_start = tile_col * self.tile_width
                    x_stop = min((tile_col + 1) * self.tile_width, self.series_shape[-1])

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

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        """Decode a tile from the TIFF file (no caching - relies on OS page cache)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        ifd_index, tile_indices = _decode_ome_tile(backend_data)
        tile_row, tile_col = tile_indices

        # tifffile TiffFile/page access is not safe under concurrent reads.
        # Serialize IO to avoid corrupted page/tag parsing in parallel fetches.
        with self._io_lock:
            page = self.tiff_file.pages[ifd_index]

            # Compute tile index
            tile_idx = tile_row * self.tiles_per_row + tile_col

            if not self.is_tiled:
                data = page.asarray()
                if data.ndim == 3:
                    data = data[0]
                arr = pa.array(data.ravel())
                return pa.RecordBatch.from_arrays([arr], ["data"])

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

            arr = pa.array(data.ravel())
            return pa.RecordBatch.from_arrays([arr], ["data"])

    def get_metadata(self) -> dict:
        """Return OME metadata from TIFF file as JSON-serializable dict.

        Uses tifffile's ome_metadata attribute which contains parsed OME-XML.

        Returns:
            OME-XML as JSON-serializable dict, or empty dict if no metadata.
        """
        if hasattr(self.tiff_file, 'ome_metadata') and self.tiff_file.ome_metadata is not None:
            ome_meta = self.tiff_file.ome_metadata
            # tifffile may return string (raw XML) or ElementTree
            if isinstance(ome_meta, str):
                try:
                    root = ET.fromstring(ome_meta)
                    return _elementtree_to_dict(root)
                except ET.ParseError:
                    return {}
            else:
                return _elementtree_to_dict(ome_meta)
        return {}


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

    Relies on OS page cache for raw data caching.
    """

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim directories with multi-file OME-TIFF structure.

        This is a multi-node claim that claims the directory + all TIFF files
        + metadata file.

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            SourceClaim with multi-node paths if valid structure, None otherwise
        """
        if not path.is_dir():
            return None

        # Check for metadata file or multiple img_*.ome.tiff files
        metadata_file = path / '_metadata.txt'
        img_files = list(path.glob('img_*.ome.tiff')) + list(path.glob('img_*.ome.tif'))

        # Also check for other common patterns
        if not metadata_file.exists() and len(img_files) <= 1:
            # Check for other multi-file patterns
            ome_files = list(path.glob('*.ome.tiff')) + list(path.glob('*.ome.tif'))
            if len(ome_files) <= 1:
                # No multi-file structure detected
                return None

        # Multi-node claim: claim the directory + all TIFF files + metadata
        claimed = {path}

        # Add all OME-TIFF files in the directory
        for pattern in ['img_*.ome.tiff', 'img_*.ome.tif', '*.ome.tiff', '*.ome.tif']:
            for img_file in path.glob(pattern):
                identity = get_file_identity(img_file)
                if identity not in visited_identities:
                    claimed.add(img_file)

        # Add metadata file if exists
        if metadata_file.exists():
            claimed.add(metadata_file)

        # Also check for other metadata patterns
        for meta_pattern in ['metadata.txt', '*_metadata.txt']:
            for meta_file in path.glob(meta_pattern):
                if meta_file.exists():
                    claimed.add(meta_file)

        return SourceClaim(
            source_type="ome-tiff-multifile",
            primary_path=path,
            claimed_paths=claimed,
        )

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'MultiFileOmeTiffAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url (directory), source_id, dim_labels

        Returns:
            MultiFileOmeTiffAdapter instance
        """
        return cls(str(source.url), source.source_id, source.dim_labels)

    def __init__(
        self,
        directory: str,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize multi-file OME-TIFF adapter.

        Args:
            directory: Path to directory containing multi-file dataset
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
        """
        import tifffile
        from pathlib import Path

        self.directory = Path(directory)
        self.array_id = array_id

        # Source-level metadata for DataSourceDescriptor
        self._source_url = str(directory)
        self._source_type = "ome-tiff-multifile"

        # Try to get explicit file list from _metadata.txt OME-XML
        expected_files = self._parse_file_list_from_metadata()
        self._missing_files = []

        if expected_files:
            # Use explicit file list from metadata
            tiff_files = []
            for fname in expected_files:
                fpath = self.directory / fname
                if fpath.exists():
                    tiff_files.append(fpath)
                else:
                    self._missing_files.append(fname)

            if self._missing_files:
                import warnings
                warnings.warn(
                    f"Multi-file OME-TIFF dataset has missing files: {self._missing_files}. "
                    f"Proceeding with {len(tiff_files)} available files."
                )

            if not tiff_files:
                raise ValueError(
                    f"No OME-TIFF files found in {directory}. "
                    f"Expected files: {expected_files}, missing: {self._missing_files}"
                )
        else:
            # Fall back to glob discovery (legacy behavior)
            patterns = [
                "img_*.ome.tiff", "img_*.ome.tif",
                "*.ome.tiff", "*.ome.tif",
                "img_channel*.tif", "img_*.tif",  # Micro-Manager standard format
            ]
            tiff_files = []
            for pattern in patterns:
                tiff_files.extend(sorted(self.directory.glob(pattern)))

            if not tiff_files:
                raise ValueError(f"No OME-TIFF files found in {directory}")

        # Open first file to check for OME metadata and get tile info
        self.tiff_file = tifffile.TiffFile(str(tiff_files[0]))
        self._tiff_sequence = None

        # Check if this is an OME-TIFF with auto-discovery
        has_ome_metadata = hasattr(self.tiff_file, 'ome_metadata') and self.tiff_file.ome_metadata is not None

        # For non-OME multi-file datasets, use TiffSequence to get proper multi-file shape
        if not has_ome_metadata and len(tiff_files) > 1:
            self._tiff_sequence = tifffile.TiffSequence([str(f) for f in tiff_files])
            seq_shape = self._tiff_sequence.shape
            if len(seq_shape) == 1:
                first_page = self.tiff_file.pages[0]
                seq_shape = (len(tiff_files), first_page.shape[0], first_page.shape[1])
            self.series_shape = seq_shape
            self.series_dims = tuple(['plane'] + list(self.tiff_file.series[0].dims))
        else:
            if len(self.tiff_file.series) == 0:
                raise ValueError("No series found in OME-TIFF dataset")
            self.series = self.tiff_file.series[0]
            self.series_shape = self.series.shape
            self.series_dims = self.series.dims

        # Get tile info from first page of first file
        first_page = self.tiff_file.pages[0]

        if first_page.is_tiled:
            self.is_tiled = True
            self.tile_width = first_page.tilewidth
            self.tile_length = first_page.tilelength
            self.tiles_per_row = (first_page.shape[1] + self.tile_width - 1) // self.tile_width
            self.tiles_per_col = (first_page.shape[0] + self.tile_length - 1) // self.tile_length
            self.chunk_shape = [self.tile_length, self.tile_width]
        else:
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
            # Micro-Manager metadata carries non-spatial axis semantics (channel/z/time).
            # For multi-file datasets, tifffile often reports the leading axis as generic
            # "plane" even when it is actually channel. Prefer metadata when available.
            self._apply_metadata_axis_labels()

        # Build file index for IFD access
        self._file_ifd_map = []
        self._ifd_to_file = []

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

        # Full shape (adjust for partial datasets)
        self.full_shape = list(self.series_shape)
        if self._missing_files and len(self.full_shape) >= 2:
            if self._total_ifds < self.full_shape[0]:
                self.full_shape[0] = self._total_ifds

        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self.chunk_shape

    def _apply_metadata_axis_labels(self) -> None:
        """Refine inferred dim labels using Micro-Manager Summary metadata.

        The first non-spatial axis in multi-file datasets can be reported as
        a generic plane index by tifffile. When metadata exposes axis order
        and intended dimensions, map that leading axis to c/z/t/p accordingly.
        """
        if not self.dim_labels:
            return

        lead = str(self.dim_labels[0]).lower()
        if lead not in ("plane", "p", "i", "q"):
            return

        metadata = self.get_metadata()
        if not isinstance(metadata, dict):
            return

        summary = metadata.get("Summary")
        if not isinstance(summary, dict):
            return

        intended = summary.get("IntendedDimensions")
        if not isinstance(intended, dict):
            intended = {}

        def _count(name: str, fallback_key: Optional[str] = None) -> int:
            v = intended.get(name)
            if v is None and fallback_key is not None:
                v = summary.get(fallback_key)
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        axis_counts = {
            "channel": _count("channel", "Channels"),
            "z": _count("z", "Slices"),
            "time": _count("time", "Frames"),
            "position": _count("position", "Positions"),
        }

        axis_alias = {
            "channel": "c",
            "c": "c",
            "z": "z",
            "slice": "z",
            "time": "t",
            "t": "t",
            "frame": "t",
            "position": "p",
            "pos": "p",
            "p": "p",
        }

        leading_label: Optional[str] = None

        axis_order = summary.get("AxisOrder")
        ordered_axes: List[str] = []
        if isinstance(axis_order, list):
            ordered_axes = [str(a).strip().lower() for a in axis_order]
        elif isinstance(axis_order, str):
            ordered_axes = [a.strip().lower() for a in axis_order.split(",")]

        # Prefer the first axis in declared order that has multiplicity > 1.
        for axis_name in ordered_axes:
            n = axis_counts.get(axis_name, 0)
            if n > 1:
                leading_label = axis_alias.get(axis_name)
                if leading_label:
                    break

        # Fallback by common Micro-Manager precedence.
        if leading_label is None:
            if axis_counts["channel"] > 1:
                leading_label = "c"
            elif axis_counts["z"] > 1:
                leading_label = "z"
            elif axis_counts["time"] > 1:
                leading_label = "t"
            elif axis_counts["position"] > 1:
                leading_label = "p"

        if leading_label is not None:
            self.dim_labels[0] = leading_label

        # Normalize common spatial label variants for downstream axis mapping.
        for idx, label in enumerate(self.dim_labels):
            l = str(label).lower()
            if l in ("y", "height"):
                self.dim_labels[idx] = "y"
            elif l in ("x", "width"):
                self.dim_labels[idx] = "x"

    def get_tensor_descriptor(self) -> TensorDescriptor:
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

        for global_ifd_index in range(self._total_ifds):
            file_index, local_ifd_index = self._ifd_to_file[global_ifd_index]
            file_path = self._tiff_files[file_index]

            with tifffile.TiffFile(str(file_path)) as tf:
                page = tf.pages[local_ifd_index]

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

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        """Decode a tile from the multi-file dataset (no caching)."""
        import tifffile

        _, backend_data = _decode_chunk_id(chunk_id)
        file_index, local_ifd_index, tile_indices = _decode_ome_multifile_tile(backend_data)
        tile_row, tile_col = tile_indices

        file_path = self._tiff_files[file_index]

        with tifffile.TiffFile(str(file_path)) as tf:
            page = tf.pages[local_ifd_index]

            if self.is_tiled:
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

            arr = pa.array(data.ravel())
            return pa.RecordBatch.from_arrays([arr], ["data"])

    def get_metadata(self) -> dict:
        """Return OME metadata from multi-file dataset."""
        import xml.etree.ElementTree as ET

        # Check for companion metadata.txt or _metadata.txt file
        metadata_patterns = ["metadata.txt", "_metadata.txt", "*_metadata.txt"]
        for pattern in metadata_patterns:
            metadata_files = list(self.directory.glob(pattern))
            if metadata_files:
                try:
                    return self._parse_metadata_txt(metadata_files[0])
                except Exception:
                    pass

        # Fall back to embedded OME-XML from first TIFF file
        if hasattr(self.tiff_file, 'ome_metadata') and self.tiff_file.ome_metadata is not None:
            ome_meta = self.tiff_file.ome_metadata
            if isinstance(ome_meta, str):
                try:
                    root = ET.fromstring(ome_meta)
                    return _elementtree_to_dict(root)
                except ET.ParseError:
                    return {}
            else:
                return _elementtree_to_dict(ome_meta)

        return {}

    def _parse_metadata_txt(self, metadata_path) -> dict:
        """Parse OME-XML from companion metadata file."""
        import json

        content = metadata_path.read_text()

        try:
            root = ET.fromstring(content)
            return _elementtree_to_dict(root)
        except ET.ParseError:
            pass

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                if "OME" in data and isinstance(data["OME"], str):
                    root = ET.fromstring(data["OME"])
                    return _elementtree_to_dict(root)
                return data
        except json.JSONDecodeError:
            pass

        return {}

    def _parse_file_list_from_metadata(self) -> Optional[List[str]]:
        """Extract ordered file list from metadata.txt or _metadata.txt."""
        import json

        metadata_patterns = ["metadata.txt", "_metadata.txt", "*_metadata.txt"]
        metadata_files = []
        for pattern in metadata_patterns:
            metadata_files.extend(self.directory.glob(pattern))

        if not metadata_files:
            return None

        content = metadata_files[0].read_text()

        try:
            root = ET.fromstring(content)
            files = []
            for tiff_data in root.iter():
                if tiff_data.tag.endswith('TiffData') or tiff_data.tag == 'TiffData':
                    for uuid_elem in tiff_data:
                        if uuid_elem.tag.endswith('UUID') or uuid_elem.tag == 'UUID':
                            filename = uuid_elem.get('FileName')
                            if filename:
                                files.append(filename)
            return files if files else None
        except ET.ParseError:
            pass

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "OME" in data and isinstance(data["OME"], str):
                root = ET.fromstring(data["OME"])
                files = []
                for tiff_data in root.iter():
                    if tiff_data.tag.endswith('TiffData') or tiff_data.tag == 'TiffData':
                        for uuid_elem in tiff_data:
                            if uuid_elem.tag.endswith('UUID') or uuid_elem.tag == 'UUID':
                                filename = uuid_elem.get('FileName')
                                if filename:
                                    files.append(filename)
                return files if files else None
        except (json.JSONDecodeError, ET.ParseError):
            pass

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                files_set = set()
                for key in data.keys():
                    if key.startswith('Coords-') or key.startswith('Metadata-'):
                        if '/' in key:
                            filename = key.split('/', 1)[1]
                            if filename.endswith('.tif') or filename.endswith('.tiff'):
                                files_set.add(filename)
                if files_set:
                    files = sorted(files_set)
                    return files
        except json.JSONDecodeError:
            pass

        return None