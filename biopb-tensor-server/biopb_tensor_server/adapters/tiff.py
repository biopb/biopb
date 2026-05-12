"""OME-TIFF adapters for tensor storage.

We could just use aiscimageio backend. But a separate module makes the claim protcol cleaner.
"""

import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import tifffile
import zarr
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.chunk import ChunkEndpoint
from biopb_tensor_server.discovery import ClaimContext, SourceClaim, get_file_identity, is_remote_url

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState
    from biopb_tensor_server.remote import RemoteStore


# =============================================================================
# OME-XML metadata helpers
# =============================================================================

def _get_namespace(root) -> dict:
    """Extract namespace from root element tag.

    Args:
        root: ElementTree root element

    Returns:
        Dictionary with namespace mapping for OME schema
    """
    tag = root.tag
    if tag.startswith('{'):
        namespace = tag.split('}')[0].strip('{')
        return {'ome': namespace}
    # Fall back to common OME namespace if no namespace found
    return {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}


def _extract_files_from_ome_xml(
    ome_metadata: str,
    source_dir: Path | str,
    store: Optional["RemoteStore"] = None,
) -> Optional[List[Path] | List[str]]:
    """Extract ordered file list from OME-XML TiffData elements.

    Parses OME-XML to find all referenced TIFF files via TiffData/UUID elements.
    Files are returned in order with the first TiffData's file as master.

    Args:
        ome_metadata: OME-XML string (raw XML)
        source_dir: Directory containing the source file (for resolving relative paths)
        store: Optional RemoteStore for remote access. If None, uses local Path operations.

    Returns:
        Ordered list of Path objects (local) or str paths (remote), or None if parsing fails
    """
    try:
        root = ET.fromstring(ome_metadata)
        namespace = _get_namespace(root)

        files = []
        seen_files = set()

        # Find all TiffData elements - order matters (first is master)
        for tiff_data in root.findall('.//ome:TiffData', namespace):
            # UUID element contains FileName attribute
            uuid_elem = tiff_data.find('ome:UUID', namespace)
            if uuid_elem is None:
                # Try without namespace prefix (some OME-XML variants)
                for child in tiff_data:
                    if child.tag.endswith('UUID') or child.tag == 'UUID':
                        uuid_elem = child
                        break

            if uuid_elem is not None:
                filename = uuid_elem.get('FileName')
                if filename and filename not in seen_files:
                    if store is not None:
                        # Remote: use RemoteStore methods
                        if source_dir:
                            file_path = store._join(str(source_dir) + '/' + filename)
                        else:
                            file_path = store._join(filename)
                        exists = store.isfile(file_path)
                    else:
                        # Local: use Path operations
                        file_path = Path(source_dir) / filename
                        exists = file_path.exists()

                    if exists:
                        files.append(file_path)
                        seen_files.add(filename)

        return files if files else None
    except ET.ParseError:
        return None


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
# OmeTiffAdapter - Multi-series OME-TIFF (handles single/multi-file + companion)
# =============================================================================

class OmeTiffAdapter(SourceAdapter, TensorAdapter):
    """Adapter for OME-TIFF files using tifffile with multi-series support.

    Handles all OME-TIFF variants:
    - Case 1: OME-TIFF with embedded OME-XML (.ome.tif/.ome.tiff)
    - Case 2: Companion OME files (.companion.ome)

    This is a multi-tensor source (_single_tensor_source = False) where each
    series (field/position) is a separate tensor. Use get_tensor_adapter(series_id)
    to access a specific series.

    Uses tifffile.aszarr() for tile-level lazy loading via OmeTiffSeriesAdapter.
    """

    # Multi-tensor source - each series is a separate tensor
    _single_tensor_source = False

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim OME-TIFF files (single/multi-file) or companion.ome files.

        Handles:
        - Case 1: OME-TIFF files (.ome.tif/.ome.tiff) - discover related files via embedded OME-XML
        - Case 2: Companion OME files (.companion.ome) - parse XML to find TIFF files

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim if this is an OME-TIFF file or companion, None otherwise
        """
        if not ctx.is_file():
            return None

        name = ctx.name.lower()

        if name.endswith('.companion.ome'):
            ome_metadata = ctx.read_text()
            related_files = _extract_files_from_ome_xml(ome_metadata, ctx.parent.path_str, ctx.store)
            if related_files:
                primary_path = related_files[0]
                for f in related_files:
                    state.try_claim_path(f)
                return SourceClaim(
                    source_type="ome-tiff",
                    primary_path=primary_path,
                    is_remote=ctx.is_remote,
                )

        elif name.endswith('.tif') or name.endswith('.tiff'):
            ome_metadata = cls._get_ome_metadata(ctx)
            if ome_metadata:
                related_files = _extract_files_from_ome_xml(ome_metadata, ctx.parent.path_str, ctx.store)
                if related_files:
                    primary_path = related_files[0]
                    for f in related_files:
                        state.try_claim_path(f)
                    return SourceClaim(
                        source_type="ome-tiff",
                        primary_path=primary_path,
                        is_remote=ctx.is_remote,
                    )

        return None

    @classmethod
    def _get_ome_metadata(cls, ctx: ClaimContext) -> Optional[str]:
        """Extract OME-XML metadata from TIFF file if present.

        For remote files, this returns None since parsing embedded XML
        requires downloading the file.
        """
        if ctx.is_remote:
            return None  # Remote OME-TIFF requires companion.ome
        try:
            with tifffile.TiffFile(ctx.path_str) as tf:
                return tf.ome_metadata
        except Exception:
            return None

    @classmethod
    def create_from_config(
        cls,
        source: 'SourceConfig',
        credentials_config: Optional[Any] = None,
    ) -> 'OmeTiffAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            OmeTiffAdapter instance
        """
        # For remote sources, credentials would be handled by fsspec
        source_url = str(source.url)
        return cls(source_url, source.source_id, source.dim_labels)

    def __init__(
        self,
        source_path: os.PathLike,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize OME-TIFF adapter.

        Handles all OME-TIFF types:
        - Single-file OME-TIFF
        - Multi-file OME-TIFF (discovered via embedded OME-XML)
        - Companion OME file

        Args:
            source_path_or_tiff: Path to OME-TIFF file/companion.ome, or
                tifffile.TiffFile object (for backward compatibility - caller
                manages the file handle and must keep it open)
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (applied to all series)
        """
        self.source_id = source_id
        self._io_lock = threading.Lock()
        self._source_type = "ome-tiff"
        self._source_url = str(source_path)
        self._dim_labels_override = dim_labels
        self._series_adapters: Dict[str, str] = {}

        # Extract and cache OME metadata string
        with tifffile.TiffFile(source_path) as tf:
            self._ome_meta = tf.ome_metadata
            self._dtype = str(tf.pages[0].dtype)
            self._series_names = [series.name for series in tf.series]
            self._series_shapes = [series.shape for series in tf.series]
            self._series_dim_labels = [self._dim_labels_override or list(series.axes) for series in tf.series]

        if len(self._series_names) == 0:
            raise ValueError("No series found in OME-TIFF file")
        elif len(self._series_names) == 1:
            self._series_idx = 0

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all series as tensors.

        Returns lightweight descriptors for all series in this TIFF file.
        For backward compatibility with single-series TIFFs, the array_id is
        just source_id when there's exactly one series.
        """
        descriptors = []

        for idx in range(len(self._series_names)):
            series_shape = self._series_shapes[idx]
            series_dim_label = self._series_dim_labels[idx]

            # Create descriptor with basic info
            desc = TensorDescriptor(
                array_id=self.array_id,
                dim_labels=series_dim_label,
                shape=list(series_shape),
                dtype=self._dtype,
            )

            descriptors.append(desc)

        return descriptors

    def get_tensor_adapter(self, tensor_id: str) -> TensorAdapter:
        """Get adapter for specific series.

        - For single-series TIFFs: tensor_id can be source_id (backward compat)
        - For multi-series TIFFs: tensor_id should be series_name or source_id/series_name

        Args:
            tensor_id: Series name, or 'source_id' for single-series backward compat

        Returns:
            OmeTiffSeriesAdapter for the specified series
        """
        if len(self._series_names) == 1:
            return self
    
        if tensor_id in self._series_names:
            self._series_idx = self._series_names.index(tensor_id)
            self._tensor_name = tensor_id
        else:
            raise ValueError(f"Series '{tensor_id}' not found in OME-TIFF")

        return self

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return descriptor for first series (backward compatibility)."""
        descriptors = self.list_tensor_descriptors()
        if descriptors:
            return descriptors[self._series_idx]
        raise ValueError("No series found in OME-TIFF")


    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from zarr array.

        Uses zarr array slicing for tile-level lazy loading. IO is serialized
        because tifffile codecs may not be thread-safe.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))

        with self._io_lock:
            with tifffile.TiffFile(self._source_url) as tf:
                series = tf.series[self._series_idx]
                zarr_array = zarr.open(series.aszarr())
                return zarr_array[slices]

    def get_metadata(self) -> dict:
        """Return OME metadata from TIFF file as JSON-serializable dict.

        Uses cached OME-XML string extracted during initialization.

        Returns:
            OME-XML as JSON-serializable dict, or empty dict if no metadata.
        """
        if self._ome_meta is None:
            return {}

        # Convert OME-XML string to dict
        if isinstance(self._ome_meta, str):
            try:
                root = ET.fromstring(self._ome_meta)
                return _elementtree_to_dict(root)
            except ET.ParseError:
                return {"raw_xml": self._ome_meta}

        return self._ome_meta


# =============================================================================
# TiffSequenceAdapter - Plain TIFF sequences (no metadata)
# =============================================================================

class TiffSequenceAdapter(SourceAdapter, TensorAdapter):
    """Adapter for plain TIFF file sequences in a directory (no metadata).

    Handles datasets where multiple TIFF files form a single logical image:
    - ND000_aligned.tiff, ND001_aligned.tiff, ND002_aligned.tiff, ...

    This is a single-tensor source that tracks file list and index mapping.
    Uses TiffFile().aszarr() for true tile-level lazy reading.

    Unlike MultiFileOmeTiffAdapter, this does NOT:
    - Parse OME-XML or MicroManager metadata
    - Handle 5D/6D coordinate mapping
    - Discover related files via companion files

    Simple axis inference:
    - (num_files, Y, X) for single-page files -> 't' or 'z'
    - (num_files, pages, Y, X) for multi-page files -> 't', 'z' or 'c'
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories containing plain TIFF file sequences.

        A sequence is detected if:
        1. Directory has 3+ TIFF files (excluding OME/MicroManager patterns)
        2. Filenames contain sequential numbers
        3. All files have the same file size (fastest validation)

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if sequence detected
        """
        # Only support local directories for now (remote glob/stat is expensive)
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Gather all TIFF files (excluding OME/MicroManager patterns)
        all_tiffs = list(ctx.glob('*.tif')) + list(ctx.glob('*.tiff'))
        tiff_files = [t._path for t in all_tiffs]  # Extract underlying Path objects

        # Filter out known OME and MicroManager patterns
        exclude_patterns = {
            '*.ome.tif', '*.ome.tiff',
            'img_*.tif', 'img_*.tiff',
            'img_channel*.tif', 'img_channel*.tiff',
        }

        filtered_files = []
        for f in tiff_files:
            excluded = False
            for pattern in exclude_patterns:
                if f.match(pattern):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append(f)

        # Need at least 3 files to consider it a sequence
        if len(filtered_files) < 3:
            return None

        # Check for metadata files - if present, don't claim (let other adapters handle)
        metadata_patterns = ['metadata.txt', '_metadata.txt', 'DisplaySettings.json', '*.companion.ome']
        for pattern in metadata_patterns:
            if list(ctx._path.glob(pattern)):
                return None

        # Extract numbers from filenames to verify sequence
        numbers = []
        for f in filtered_files:
            nums = re.findall(r'\d+', f.name)
            if nums:
                numbers.append((int(nums[-1]), f))
            else:
                return None

        if not numbers:
            return None

        # Sort by extracted number
        numbers.sort(key=lambda x: x[0])
        sorted_files = [f for _, f in numbers]

        # Validate: all files should have the same size (proxy for same dimensions)
        try:
            file_sizes = [f.stat().st_size for f in sorted_files]
            if len(set(file_sizes)) > 1:
                return None
        except OSError:
            return None

        # Claim directory + all TIFF files
        state.try_claim_path(ctx.path_str)
        for img_file in sorted_files:
            state.try_claim_path(img_file)

        return SourceClaim(
            source_type="tiff-sequence",
            primary_path=ctx.path_str,
        )

    @classmethod
    def create_from_config(cls, source: 'SourceConfig', credentials_config: Optional[Any] = None) -> 'TiffSequenceAdapter':
        """Create adapter instance from SourceConfig."""
        return cls(str(source.url), source.source_id or '', source.dim_labels)

    def __init__(
        self,
        directory: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize TIFF sequence adapter.

        Args:
            directory: Path to directory containing TIFF sequence
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (if None, inferred from file count)
        """
        import re
        import tifffile

        self.directory = Path(directory)
        self.source_id = source_id
        self._source_url = str(directory)
        self._source_type = "tiff-sequence"
        self._io_lock = threading.Lock()

        # Discover TIFF files (excluding OME/MicroManager patterns)
        all_tiffs = list(self.directory.glob('*.tif')) + list(self.directory.glob('*.tiff'))

        exclude_patterns = {
            '*.ome.tif', '*.ome.tiff',
            'img_*.tif', 'img_*.tiff',
            'img_channel*.tif', 'img_channel*.tiff',
        }

        tiff_files = []
        for f in all_tiffs:
            excluded = False
            for pattern in exclude_patterns:
                if f.match(pattern):
                    excluded = True
                    break
            if not excluded:
                tiff_files.append(f)

        # Sort by numeric part of filename
        numbers = []
        for f in tiff_files:
            nums = re.findall(r'\d+', f.name)
            if nums:
                numbers.append((int(nums[-1]), f))
        numbers.sort(key=lambda x: x[0])
        self._tiff_files = [f for _, f in numbers]

        if not self._tiff_files:
            raise ValueError(f"No TIFF sequence files found in {directory}")

        # Open first file to get shape and tile info
        with tifffile.TiffFile(str(self._tiff_files[0])) as tf:
            first_page = tf.pages[0]
            self._dtype = str(first_page.dtype)

            # Tile info
            if first_page.is_tiled:
                self.is_tiled = True
                self.tile_width = first_page.tilewidth
                self.tile_length = first_page.tilelength
                self._spatial_chunk = [self.tile_length, self.tile_width]
            else:
                self.is_tiled = False
                self._spatial_chunk = [first_page.shape[0], first_page.shape[1]]

            # Pages per file
            n_pages_per_file = len(tf.pages)

        # Build file index map: (file_path, n_pages)
        self._file_ifd_map = []
        for file_path in self._tiff_files:
            with tifffile.TiffFile(str(file_path)) as tf:
                n_pages = len(tf.pages)
                self._file_ifd_map.append((file_path, n_pages))

        # Determine shape and dim labels
        # Shape: (num_files, pages_per_file, Y, X) or (num_files, Y, X)
        n_files = len(self._tiff_files)

        # Get spatial shape from first page (not series, which may include page dimension)
        spatial_shape = [first_page.shape[0], first_page.shape[1]]

        if n_pages_per_file > 1:
            # Multi-page files: (num_files, pages, Y, X)
            self.full_shape = [n_files, n_pages_per_file] + spatial_shape
            self.chunk_shape = [1, 1] + self._spatial_chunk
            if dim_labels:
                self.dim_labels = dim_labels
            else:
                self.dim_labels = ['t', 'z', 'y', 'x']
        else:
            # Single-page files: (num_files, Y, X)
            self.full_shape = [n_files] + spatial_shape
            self.chunk_shape = [1] + self._spatial_chunk
            if dim_labels:
                self.dim_labels = dim_labels
            else:
                self.dim_labels = ['t', 'y', 'x']

        # Total IFDs for coordinate mapping
        self._total_ifds = sum(n for _, n in self._file_ifd_map)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds using tile-level lazy access.

        Uses TiffFile().aszarr() for true tile-level lazy reading.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        import tifffile
        import zarr

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))

        with self._io_lock:
            pages_per_file = self._file_ifd_map[0][1] if self._file_ifd_map else 1
            original_ndim = len(slices)

            # Build slice tuple: (file_slice, [page_slice], y_slice, x_slice)
            if original_ndim == 4:
                file_slice, page_slice, y_slice, x_slice = slices
            elif original_ndim == 3:
                file_slice, y_slice, x_slice = slices
                page_slice = slice(0, pages_per_file) if pages_per_file > 1 else slice(0, 1)
            else:
                # Handle other dimensionalities
                file_slice = slices[0]
                page_slice = slice(0, pages_per_file) if len(slices) > 3 else slice(0, 1)
                y_slice = slices[-2] if len(slices) >= 2 else slice(None)
                x_slice = slices[-1] if len(slices) >= 1 else slice(None)

            # Determine which files and pages to read
            file_indices = range(
                file_slice.start or 0,
                min(file_slice.stop or len(self._file_ifd_map), len(self._file_ifd_map))
            )
            page_indices = range(
                page_slice.start or 0,
                min(page_slice.stop or pages_per_file, pages_per_file)
            )

            n_files = (file_slice.stop or len(self._file_ifd_map)) - (file_slice.start or 0)
            n_pages = (page_slice.stop or pages_per_file) - (page_slice.start or 0)

            # Read data via aszarr() for tile-level access
            result_pages = []
            for file_idx in file_indices:
                file_path, n_pages_in_file = self._file_ifd_map[file_idx]
                with tifffile.TiffFile(str(file_path)) as tf:
                    zarr_arr = zarr.open_array(tf.series[0].aszarr(), mode='r')
                    zarr_ndim = len(zarr_arr.shape)
                    for page_idx in page_indices:
                        if page_idx < n_pages_in_file:
                            if zarr_ndim == 2:
                                # 2D array (Y, X) - single page
                                page_data = zarr_arr[y_slice, x_slice]
                            else:
                                # 3D+ array (pages, Y, X)
                                page_data = zarr_arr[page_idx, y_slice, x_slice]
                            result_pages.append(page_data)

            # Stack into result array
            if result_pages:
                result_4d = np.stack(result_pages, axis=0)
                h, w = result_4d.shape[-2:]
                result_4d = result_4d.reshape(n_files, n_pages, h, w)
            else:
                result_4d = np.array([])

            # Reshape to match original ndim
            if original_ndim == 3:
                if pages_per_file > 1:
                    result = result_4d.reshape(n_files * n_pages, h, w)
                else:
                    result = result_4d.squeeze(axis=1)
            elif original_ndim == 4:
                result = result_4d
            else:
                result = result_4d

            return result

    def get_metadata(self) -> dict:
        """Return empty dict (no metadata for plain TIFF sequences)."""
        return {}


# =============================================================================
# MicroManagerLegacyAdapter - Legacy MicroManager datasets with JSON metadata
# =============================================================================

class MicroManagerLegacyAdapter(SourceAdapter, TensorAdapter):
    """Adapter for legacy MicroManager datasets with JSON metadata.

    Handles MicroManager v1 v2 datasets with metadata.txt containing:
    - Summary: IntendedDimensions, AxisOrder, Channels, Slices, Frames, Positions
    - Coords-Default/<filename>: PositionIndex, TimeIndex, ChannelIndex, SliceIndex
    - Metadata-Default/<filename>: UUID, Width, Height

    This adapter supports full 5D/6D datasets: (position, time, channel, z, y, x).

    Uses TiffFile().aszarr() for true tile-level lazy reading.
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories with MicroManager metadata files.

        Handles both:
        - MicroManager v1: metadata.txt directly in directory
        - MicroManager v2: DisplaySettings.json at root, metadata.txt in subdirectory

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if MicroManager format
        """
        # Only support local directories for now
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Check for v1 metadata.txt directly in this directory
        v1_metadata = None
        if ctx.join('metadata.txt').exists():
            v1_metadata = ctx.join('metadata.txt')._path

        # Check for v2 DisplaySettings.json (need to find subdirectory with metadata.txt)
        v2_data_dir = None
        if v1_metadata is None and ctx.join('DisplaySettings.json').exists():
            # Look for subdirectories containing metadata.txt
            for subdir in ctx._path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    candidate_metadata = subdir / 'metadata.txt'
                    if candidate_metadata.exists():
                        v2_data_dir = subdir
                        v1_metadata = candidate_metadata
                        break

        if v1_metadata is None:
            return None

        # Parse metadata to confirm it's MicroManager v1 format
        try:
            content = v1_metadata.read_text()
            data = json.loads(content)

            # Check for MicroManager v1 format markers
            has_coords = any(k.startswith('Coords-') for k in data.keys())
            has_summary = 'Summary' in data

            if not (has_coords or has_summary):
                return None
        except (json.JSONDecodeError, OSError):
            return None

        # Discover TIFF files (in current dir or v2 data subdirectory)
        search_dir = v2_data_dir if v2_data_dir else ctx._path
        tiff_patterns = [
            'img_channel*.tif', 'img_channel*.tiff',
            'img_pos*_channel*.tif', 'img_pos*_channel*.tiff',
            'img_*.tif', 'img_*.tiff',
        ]
        tiff_files = []
        seen = set()
        for pattern in tiff_patterns:
            for f in search_dir.glob(pattern):
                if f not in seen:
                    seen.add(f)
                    tiff_files.append(f)

        if not tiff_files:
            return None

        # Claim root directory + metadata + TIFF files
        state.try_claim_path(ctx.path_str)
        state.try_claim_path(str(v1_metadata))
        if v2_data_dir:
            state.try_claim_path(str(v2_data_dir))
        for img_file in tiff_files:
            state.try_claim_path(img_file)

        return SourceClaim(
            source_type="micromanager-legacy",
            primary_path=ctx.path_str,
        )

    @staticmethod
    def _find_metadata_file(directory: Path) -> Optional[Path]:
        """Find MicroManager metadata file in directory.

        Handles both v1 (metadata.txt directly) and v2 (DisplaySettings.json at root,
        metadata.txt in subdirectory) formats.

        Args:
            directory: Path to directory to search

        Returns:
            Path to metadata.txt if found, None otherwise
        """
        # Check for v1 format: metadata.txt directly in directory
        v1_metadata = directory / 'metadata.txt'
        if v1_metadata.exists():
            return v1_metadata

        # Check for v2 format: DisplaySettings.json at root, metadata.txt in subdirectory
        display_settings = directory / 'DisplaySettings.json'
        if display_settings.exists():
            for subdir in directory.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    candidate_metadata = subdir / 'metadata.txt'
                    if candidate_metadata.exists():
                        return candidate_metadata

        return None

    @classmethod
    def create_from_config(cls, source: 'SourceConfig', credentials_config: Optional[Any] = None) -> 'MicroManagerLegacyAdapter':
        """Create adapter instance from SourceConfig."""
        return cls(str(source.url), source.source_id or '', source.dim_labels)

    def __init__(
        self,
        directory: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize MicroManager legacy adapter.

        Args:
            directory: Path to directory containing MicroManager dataset
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (if None, inferred from metadata)
        """
        import json
        import tifffile

        self.directory = Path(directory)
        self.source_id = source_id
        self._source_url = str(directory)
        self._source_type = "micromanager-legacy"
        self._io_lock = threading.Lock()

        # Find and parse metadata file
        metadata_file = self._find_metadata_file(self.directory)
        if metadata_file is None:
            raise ValueError(f"No MicroManager metadata file found in {directory}")

        with open(metadata_file) as f:
            self._raw_metadata = json.load(f)

        # Parse Summary to get dimensions
        summary = self._raw_metadata.get('Summary', {})
        intended = summary.get('IntendedDimensions', {})

        # Get dimension sizes
        self._n_positions = intended.get('position', summary.get('Positions', 1))
        self._n_times = intended.get('time', summary.get('Frames', 1))
        self._n_channels = intended.get('channel', summary.get('Channels', 1))
        self._n_z = intended.get('z', summary.get('Slices', 1))

        # Build coordinate map: (p, t, c, z) -> filename
        self._coord_map: Dict[Tuple[int, int, int, int], Path] = {}
        self._file_list: List[Path] = []

        for key in self._raw_metadata.keys():
            if key.startswith('Coords-'):
                coords = self._raw_metadata[key]
                # Extract path from key (Coords-Default/<filename> or Coords-<filename>)
                coords_prefix_len = len('Coords-')
                filepath_in_key = key[coords_prefix_len:]

                # Get indices
                pos_idx = coords.get('PositionIndex', 0)
                time_idx = coords.get('TimeIndex', 0)
                chan_idx = coords.get('ChannelIndex', 0)
                slice_idx = coords.get('SliceIndex', 0)

                # Look for file relative to directory (may be in subdirectory for v2)
                file_path = self.directory / filepath_in_key
                if file_path.exists():
                    self._coord_map[(pos_idx, time_idx, chan_idx, slice_idx)] = file_path
                    self._file_list.append(file_path)

        if not self._file_list:
            raise ValueError(f"No MicroManager TIFF files found in {directory}")

        # Sort file list for consistent ordering
        self._file_list = sorted(set(self._file_list))

        # Open first file to get shape and dtype
        with tifffile.TiffFile(str(self._file_list[0])) as tf:
            first_page = tf.pages[0]
            self._dtype = str(first_page.dtype)
            self._height = first_page.shape[0]
            self._width = first_page.shape[1]

            # Tile info
            if first_page.is_tiled:
                self.is_tiled = True
                self.tile_width = first_page.tilewidth
                self.tile_length = first_page.tilelength
                self._spatial_chunk = [self.tile_length, self.tile_width]
            else:
                self.is_tiled = False
                self._spatial_chunk = [self._height, self._width]

        # Get axis order from metadata
        axis_order = summary.get('AxisOrder', ['position', 'time', 'channel', 'z'])
        if isinstance(axis_order, str):
            axis_order = [a.strip() for a in axis_order.split(',')]
        self._axis_order = axis_order

        # Map axis names to their counts
        axis_counts = {
            'position': self._n_positions,
            'time': self._n_times,
            'channel': self._n_channels,
            'z': self._n_z,
        }

        # Build shape following axis_order from metadata
        shape_axes = [a.lower() for a in axis_order]
        initial_shape = [axis_counts.get(a.lower(), 1) for a in axis_order] + [self._height, self._width]

        # Remove singleton dimensions while preserving axis correspondence
        # Only remove from non-spatial dimensions (keep y, x)
        self.full_shape = []
        self._shape_axes = []
        for i, (size, axis) in enumerate(zip(initial_shape, shape_axes)):
            if i < len(shape_axes):  # Non-spatial axis
                if size > 1:  # Keep non-singleton dimensions
                    self.full_shape.append(size)
                    self._shape_axes.append(axis)
            else:  # This shouldn't happen, shape_axes is shorter
                pass

        # Always append spatial dimensions
        self.full_shape.extend([self._height, self._width])

        # Chunk shape
        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self._spatial_chunk

        # Dimension labels (from remaining axes after singleton removal)
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            axis_alias = {
                'position': 'p', 'pos': 'p',
                'time': 't', 'frame': 't',
                'channel': 'c',
                'z': 'z', 'slice': 'z',
            }
            self.dim_labels = []
            for axis in self._shape_axes:
                label = axis_alias.get(axis.lower(), axis.lower()[0])
                self.dim_labels.append(label)
            self.dim_labels.extend(['y', 'x'])

        # Build index for efficient lookups
        self._build_file_index()

    def _build_file_index(self) -> None:
        """Build index mapping for efficient coordinate lookups."""
        # Create reverse lookup: global_index -> file_path
        # For single-position/time/z datasets, this simplifies access
        self._index_to_file: Dict[int, Path] = {}

        if len(self.full_shape) == 3:
            # (channels, y, x) - single index is channel
            for (pos, time, chan, z), file_path in self._coord_map.items():
                if pos == 0 and time == 0 and z == 0:
                    self._index_to_file[chan] = file_path
        elif len(self.full_shape) == 4:
            # (channels, z, y, x) or similar
            for (pos, time, chan, z), file_path in self._coord_map.items():
                if pos == 0 and time == 0:
                    # Index = chan * n_z + z (assuming channel-z order)
                    idx = chan * self._n_z + z
                    self._index_to_file[idx] = file_path

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds using tile-level lazy access.

        Uses TiffFile().aszarr() for true tile-level lazy reading.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        import tifffile
        import zarr

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))

        with self._io_lock:
            ndim = len(self.full_shape)
            original_ndim = len(slices)

            # Extract spatial slices (last 2 axes)
            y_slice = slices[-2] if len(slices) >= 2 else slice(None)
            x_slice = slices[-1] if len(slices) >= 1 else slice(None)

            # Determine coordinate ranges based on shape
            if ndim == 3:
                # (channels, y, x)
                chan_slice = slices[0]
                pos_range = [0]
                time_range = [0]
                chan_range = range(chan_slice.start or 0, chan_slice.stop or self._n_channels)
                z_range = [0]
            elif ndim == 4:
                # (channels, z, y, x) or (time, channels, y, x) depending on axis_order
                if 'z' in self.dim_labels[:2]:
                    # (channels, z, y, x) format
                    chan_slice = slices[0]
                    z_slice = slices[1]
                    pos_range = [0]
                    time_range = [0]
                    chan_range = range(chan_slice.start or 0, chan_slice.stop or self._n_channels)
                    z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
                else:
                    # (time, channels, y, x) format
                    time_slice = slices[0]
                    chan_slice = slices[1]
                    pos_range = [0]
                    time_range = range(time_slice.start or 0, time_slice.stop or self._n_times)
                    chan_range = range(chan_slice.start or 0, chan_slice.stop or self._n_channels)
                    z_range = [0]
            elif ndim == 5:
                # (time, channels, z, y, x) or (position, channels, z, y, x)
                if 'p' in self.dim_labels:
                    pos_slice = slices[0]
                    chan_slice = slices[1]
                    z_slice = slices[2]
                    pos_range = range(pos_slice.start or 0, pos_slice.stop or self._n_positions)
                    time_range = [0]
                    chan_range = range(chan_slice.start or 0, chan_slice.stop or self._n_channels)
                    z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
                else:
                    time_slice = slices[0]
                    chan_slice = slices[1]
                    z_slice = slices[2]
                    pos_range = [0]
                    time_range = range(time_slice.start or 0, time_slice.stop or self._n_times)
                    chan_range = range(chan_slice.start or 0, chan_slice.stop or self._n_channels)
                    z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
            elif ndim == 6:
                # (position, time, channels, z, y, x)
                pos_slice = slices[0]
                time_slice = slices[1]
                chan_slice = slices[2]
                z_slice = slices[3]
                pos_range = range(pos_slice.start or 0, pos_slice.stop or self._n_positions)
                time_range = range(time_slice.start or 0, time_slice.stop or self._n_times)
                chan_range = range(chan_slice.start or 0, chan_slice.stop or self._n_channels)
                z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
            else:
                # Fallback - assume all dimensions except last 2 are 1
                pos_range = [0]
                time_range = [0]
                chan_range = [0]
                z_range = [0]

            # Collect data for each coordinate
            result_pages = []
            for pos_idx in pos_range:
                for time_idx in time_range:
                    for chan_idx in chan_range:
                        for slice_idx in z_range:
                            coord = (pos_idx, time_idx, chan_idx, slice_idx)
                            file_path = self._coord_map.get(coord)

                            if file_path is None:
                                # Create empty array for missing data
                                h = y_slice.stop - (y_slice.start or 0) if y_slice.stop else self._height
                                w = x_slice.stop - (x_slice.start or 0) if x_slice.stop else self._width
                                result_pages.append(np.zeros((h, w), dtype=self._dtype))
                                continue

                            with tifffile.TiffFile(str(file_path)) as tf:
                                zarr_arr = zarr.open_array(tf.series[0].aszarr(), mode='r')
                                zarr_ndim = len(zarr_arr.shape)
                                if zarr_ndim == 2:
                                    page_data = zarr_arr[y_slice, x_slice]
                                else:
                                    page_data = zarr_arr[0, y_slice, x_slice] if zarr_ndim >= 3 else zarr_arr[y_slice, x_slice]
                                result_pages.append(page_data)

            # Build output array with proper shape
            if not result_pages:
                return np.array([])

            # Stack pages and reshape to match the expected shape
            n_pos = len(pos_range)
            n_time = len(time_range)
            n_chan = len(chan_range)
            n_z = len(z_range)

            h = result_pages[0].shape[0] if result_pages else 0
            w = result_pages[0].shape[1] if result_pages else 0

            # Stack all pages and reshape
            result = np.stack(result_pages, axis=0)

            # Reshape to (pos, time, chan, z, h, w) based on actual ranges
            if ndim == 3:
                result = result.reshape(n_chan, h, w)
            elif ndim == 4:
                result = result.reshape(n_chan * n_z, h, w)
                if 'z' in self.dim_labels[:2]:
                    result = result.reshape(n_chan, n_z, h, w)
                else:
                    result = result.reshape(n_time, n_chan, h, w)
            elif ndim == 5:
                if 'p' in self.dim_labels:
                    result = result.reshape(n_pos, n_chan, n_z, h, w)
                else:
                    result = result.reshape(n_time, n_chan, n_z, h, w)
            elif ndim == 6:
                result = result.reshape(n_pos, n_time, n_chan, n_z, h, w)

            return result

    def get_metadata(self) -> dict:
        """Return parsed MicroManager metadata."""
        return self._raw_metadata