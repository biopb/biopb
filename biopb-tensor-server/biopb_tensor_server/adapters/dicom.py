"""DICOM adapters for tensor storage.

Handles single DICOM files and multi-file DICOM series using pydicom.
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import BackendAdapter
from biopb_tensor_server.chunk import ChunkEndpoint
from biopb_tensor_server.discovery import SourceClaim, get_file_identity

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig


# =============================================================================
# DICOM metadata helpers
# =============================================================================


def _dicom_value_to_json(value) -> any:
    """Convert DICOM value to JSON-serializable format."""
    try:
        import pydicom
        if isinstance(value, pydicom.multival.MultiValue):
            return [_dicom_value_to_json(v) for v in value]
        elif isinstance(value, bytes):
            return value.hex()
        elif isinstance(value, pydicom.uid.UID):
            return str(value)
        elif hasattr(value, 'value'):
            # Sequence items or data element wrappers
            return _dicom_value_to_json(value.value)
        else:
            return value
    except Exception:
        return str(value)


def _derive_orientation_from_iop(iop: List[float]) -> str:
    """Derive slice orientation from ImageOrientationPatient.

    IOP is [row_cosine_x, row_cosine_y, row_cosine_z, col_cosine_x, col_cosine_y, col_cosine_z]
    Cross product gives slice normal direction.
    """
    if len(iop) < 6:
        return "unknown"

    row_x, row_y, row_z, col_x, col_y, col_z = iop[:6]

    # Cross product gives slice normal
    normal = [
        row_y * col_z - row_z * col_y,
        row_z * col_x - row_x * col_z,
        row_x * col_y - row_y * col_x
    ]

    # Find dominant component
    max_idx = max(range(3), key=lambda i: abs(normal[i]))

    if max_idx == 0:
        return "sagittal"
    elif max_idx == 1:
        return "coronal"
    else:
        return "axial"


# =============================================================================
# DicomAdapter - Single file
# =============================================================================

class DicomAdapter(BackendAdapter):
    """Adapter for single DICOM files.

    Handles .dcm and .dicom files with pixel data.

    Chunk strategy:
    - Single frame: Single chunk for entire 2D image
    - Multi-frame: One chunk per frame (NumberOfFrames > 1)

    Uses pydicom for DICOM parsing and pixel data access.
    """

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim single DICOM files.

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            SourceClaim if this is a valid DICOM file with pixel data capability, None otherwise
        """
        if not path.is_file():
            return None

        # Check extension
        name = path.name.lower()
        if not (name.endswith('.dcm') or name.endswith('.dicom')):
            # Could also check for DICOM prefix "DICM" at byte 128
            # But extension check is sufficient for most cases
            return None

        try:
            import pydicom
            # Quick check: read metadata only (no pixel data)
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)

            # Check for image-related tags (indicating pixel data capability)
            # PixelData isn't loaded with stop_before_pixels=True, so we check Rows/Columns
            if not (hasattr(ds, 'Rows') and hasattr(ds, 'Columns')):
                return None

            return SourceClaim(
                source_type="dicom",
                primary_path=path,
                claimed_paths={path},
            )
        except Exception:
            return None

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'DicomAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels

        Returns:
            DicomAdapter instance
        """
        import pydicom

        ds = pydicom.dcmread(str(source.url))
        return cls(ds, source.source_id, source.dim_labels)

    def __init__(
        self,
        dicom_dataset,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize DICOM adapter.

        Args:
            dicom_dataset: pydicom Dataset object
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels
        """
        self.ds = dicom_dataset
        self.source_id = source_id
        self._io_lock = threading.Lock()

        # Source-level metadata
        if hasattr(dicom_dataset, 'filename'):
            self._source_url = str(dicom_dataset.filename)
        else:
            self._source_url = ""
        self._source_type = "dicom"

        # Get shape info
        rows = int(self.ds.get('Rows', 0))
        cols = int(self.ds.get('Columns', 0))
        num_frames = int(self.ds.get('NumberOfFrames', 1))

        if num_frames > 1:
            self._shape = (num_frames, rows, cols)
            self._is_multiframe = True
        else:
            self._shape = (rows, cols)
            self._is_multiframe = False

        # Get dtype from pixel representation
        bits_stored = int(self.ds.get('BitsStored', 16))
        pixel_repr = int(self.ds.get('PixelRepresentation', 0))

        if bits_stored <= 8:
            self._dtype = 'uint8' if pixel_repr == 0 else 'int8'
        elif bits_stored <= 16:
            self._dtype = 'uint16' if pixel_repr == 0 else 'int16'
        elif bits_stored <= 32:
            self._dtype = 'uint32' if pixel_repr == 0 else 'int32'
        else:
            self._dtype = 'float32'

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            if self._is_multiframe:
                self.dim_labels = ['frame', 'y', 'x']
            else:
                self.dim_labels = ['y', 'x']

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self._shape),
            chunk_shape=list(self._shape),  # Single chunk
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from DICOM pixel data.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape
        """
        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))

        # Serialize IO for thread safety
        with self._io_lock:
            pixel_data = self.ds.pixel_array
            return pixel_data[slices]

    def get_metadata(self) -> dict:
        """Extract DICOM metadata.

        Returns:
            Dictionary with format identifier, DICOM tags, and derived spatial info
        """
        metadata = {
            "format": "dicom",
            "tags": {},
            "spatial": {},
            "patient": {},
        }

        # Key pixel data tags
        pixel_tags = [
            'PixelSpacing', 'SliceThickness', 'ImageOrientationPatient',
            'ImagePositionPatient', 'SliceLocation', 'InstanceNumber',
            'WindowCenter', 'WindowWidth', 'RescaleSlope', 'RescaleIntercept',
            'BitsStored', 'BitsAllocated', 'PixelRepresentation', 'PhotometricInterpretation',
            'Rows', 'Columns', 'NumberOfFrames', 'SamplesPerPixel',
            'KVP', 'ExposureTime', 'XRayTubeCurrent', 'SliceLocation',
        ]

        for tag_name in pixel_tags:
            if hasattr(self.ds, tag_name):
                value = getattr(self.ds, tag_name)
                metadata["tags"][tag_name] = _dicom_value_to_json(value)

        # Derived spatial info
        if hasattr(self.ds, 'PixelSpacing'):
            ps = self.ds.PixelSpacing
            if len(ps) >= 2:
                metadata["spatial"]["pixel_spacing_mm"] = [float(ps[0]), float(ps[1])]

        if hasattr(self.ds, 'SliceThickness'):
            metadata["spatial"]["slice_spacing_mm"] = float(self.ds.SliceThickness)

        if hasattr(self.ds, 'ImageOrientationPatient'):
            iop = self.ds.ImageOrientationPatient
            metadata["spatial"]["orientation"] = _derive_orientation_from_iop(list(iop))

        if hasattr(self.ds, 'ImagePositionPatient'):
            ipp = self.ds.ImagePositionPatient
            metadata["spatial"]["origin_mm"] = [float(ipp[0]), float(ipp[1]), float(ipp[2])]

        if hasattr(self.ds, 'SliceLocation'):
            metadata["spatial"]["slice_location_mm"] = float(self.ds.SliceLocation)

        # Patient/study info (including all patient fields per user request)
        patient_tags = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'PatientAge', 'PatientWeight', 'PatientSize',
            'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
            'StudyDate', 'SeriesDate', 'AcquisitionDate',
            'StudyDescription', 'SeriesDescription',
            'Modality', 'Manufacturer', 'InstitutionName',
        ]

        for tag_name in patient_tags:
            if hasattr(self.ds, tag_name):
                value = getattr(self.ds, tag_name)
                metadata["patient"][tag_name] = _dicom_value_to_json(value)

        # Windowing parameters (for visualization)
        if hasattr(self.ds, 'WindowCenter') and hasattr(self.ds, 'WindowWidth'):
            try:
                wc = self.ds.WindowCenter
                ww = self.ds.WindowWidth
                # Handle multi-value window settings
                if hasattr(wc, '__iter__') and not isinstance(wc, str):
                    metadata["tags"]["WindowCenter"] = [float(v) for v in wc]
                else:
                    metadata["tags"]["WindowCenter"] = float(wc)
                if hasattr(ww, '__iter__') and not isinstance(ww, str):
                    metadata["tags"]["WindowWidth"] = [float(v) for v in ww]
                else:
                    metadata["tags"]["WindowWidth"] = float(ww)
            except Exception:
                pass

        return metadata


# =============================================================================
# DicomSeriesAdapter - Multi-file series
# =============================================================================

class DicomSeriesAdapter(BackendAdapter):
    """Adapter for multi-file DICOM series forming a 3D volume.

    Handles directories where multiple DICOM files share the same SeriesInstanceUID.
    Each file represents one slice of the volume.

    Multi-node claim: claims directory + all DICOM files in the series.

    Chunk strategy: One chunk per slice (file), shape = [1, H, W]
    """

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim directories containing DICOM series.

        Detects multiple DICOM files sharing the same SeriesInstanceUID.

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            SourceClaim (multi-node) if valid DICOM series directory, None otherwise
        """
        if not path.is_dir():
            return None

        # Find DICOM files
        dcm_files = list(path.glob('*.dcm')) + list(path.glob('*.DICOM')) + list(path.glob('*.dicom'))

        # Need at least 2 files for a series
        if len(dcm_files) < 2:
            return None

        try:
            import pydicom

            # Read first file to get series UID
            first_ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

            if not hasattr(first_ds, 'SeriesInstanceUID'):
                return None

            series_uid = first_ds.SeriesInstanceUID

            # Collect files that belong to this series
            series_files = []
            for f in dcm_files:
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                    if hasattr(ds, 'SeriesInstanceUID') and ds.SeriesInstanceUID == series_uid:
                        # Check for image-related tags (Rows/Columns indicate pixel data capability)
                        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                            series_files.append(f)
                except Exception:
                    continue

            if len(series_files) < 2:
                return None

            # Multi-node claim: directory + all series files
            claimed = {path}
            for f in series_files:
                try:
                    identity = get_file_identity(f)
                    if identity not in visited_identities:
                        claimed.add(f)
                except OSError:
                    pass

            return SourceClaim(
                source_type="dicom-series",
                primary_path=path,
                claimed_paths=claimed,
                extra_config={'num_slices': len(series_files), 'series_uid': str(series_uid)},
            )

        except Exception:
            return None

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'DicomSeriesAdapter':
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url (directory), source_id, dim_labels

        Returns:
            DicomSeriesAdapter instance
        """
        return cls(str(source.url), source.source_id, source.dim_labels)

    def __init__(
        self,
        directory: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize DICOM series adapter.

        Args:
            directory: Path to directory containing DICOM series
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels
        """
        from pathlib import Path
        import pydicom

        self.directory = Path(directory)
        self.source_id = source_id
        self._io_lock = threading.Lock()

        # Source-level metadata
        self._source_url = str(directory)
        self._source_type = "dicom-series"

        # Find and sort DICOM files
        dcm_files = list(self.directory.glob('*.dcm')) + list(self.directory.glob('*.DICOM')) + list(self.directory.glob('*.dicom'))

        # Read first file to get series UID and sort criteria
        first_ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        series_uid = first_ds.SeriesInstanceUID

        # Collect files with metadata for sorting
        file_info = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if hasattr(ds, 'SeriesInstanceUID') and ds.SeriesInstanceUID == series_uid:
                    # Get sorting key
                    instance_num = int(ds.get('InstanceNumber', -1))
                    slice_loc = float(ds.get('SliceLocation', 0)) if hasattr(ds, 'SliceLocation') else 0
                    ipp = ds.get('ImagePositionPatient', [0, 0, 0]) if hasattr(ds, 'ImagePositionPatient') else [0, 0, 0]
                    z_pos = float(ipp[2]) if len(ipp) >= 3 else 0

                    file_info.append({
                        'path': f,
                        'instance_number': instance_num,
                        'slice_location': slice_loc,
                        'z_position': z_pos,
                    })
            except Exception:
                continue

        # Sort files - prefer InstanceNumber, then SliceLocation, then z_position
        if any(info['instance_number'] >= 0 for info in file_info):
            # Sort by InstanceNumber
            file_info.sort(key=lambda x: x['instance_number'])
        elif any(info['slice_location'] != 0 for info in file_info):
            # Sort by SliceLocation
            file_info.sort(key=lambda x: x['slice_location'])
        else:
            # Sort by z_position from ImagePositionPatient
            file_info.sort(key=lambda x: x['z_position'])

        self.dicom_files = [info['path'] for info in file_info]
        self._num_slices = len(self.dicom_files)

        if self._num_slices == 0:
            raise ValueError(f"No valid DICOM files found in series: {directory}")

        # Read first file to get shape and dtype info
        first_full = pydicom.dcmread(str(self.dicom_files[0]))
        rows = int(first_full.Rows)
        cols = int(first_full.Columns)
        bits_stored = int(first_full.get('BitsStored', 16))
        pixel_repr = int(first_full.get('PixelRepresentation', 0))

        if bits_stored <= 8:
            self._dtype = 'uint8' if pixel_repr == 0 else 'int8'
        elif bits_stored <= 16:
            self._dtype = 'uint16' if pixel_repr == 0 else 'int16'
        elif bits_stored <= 32:
            self._dtype = 'uint32' if pixel_repr == 0 else 'int32'
        else:
            self._dtype = 'float32'

        self._shape = (self._num_slices, rows, cols)
        self._rows = rows
        self._cols = cols

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            self.dim_labels = ['z', 'y', 'x']

        # Store first dataset for metadata extraction
        self._first_ds = first_full

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self._shape),
            chunk_shape=[1, self._rows, self._cols],  # One slice per chunk
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from DICOM series.

        Reads multiple slices if needed and stacks them into a contiguous array.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape
        """
        import pydicom

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))
        slice_start = int(bounds.start[0])
        slice_stop = int(bounds.stop[0])

        # Determine output shape
        out_shape = tuple(int(e - s) for s, e in zip(bounds.start, bounds.stop))
        dtype = np.dtype(self._dtype)

        # Serialize IO for thread safety
        with self._io_lock:
            if slice_stop - slice_start == 1:
                # Single slice - read directly
                dcm_file = self.dicom_files[slice_start]
                ds = pydicom.dcmread(str(dcm_file))
                pixel_data = ds.pixel_array
                # Apply spatial slices
                return pixel_data[slices[1:]]
            else:
                # Multiple slices - read each and stack
                result = np.zeros(out_shape, dtype=dtype)
                for i, slice_idx in enumerate(range(slice_start, slice_stop)):
                    dcm_file = self.dicom_files[slice_idx]
                    ds = pydicom.dcmread(str(dcm_file))
                    pixel_data = ds.pixel_array
                    # Apply spatial slices and assign to result
                    result[i] = pixel_data[slices[1:]]
                return result

    def get_metadata(self) -> dict:
        """Extract DICOM series metadata.

        Returns metadata from first file plus series-level info.
        """
        metadata = {
            "format": "dicom",
            "tags": {},
            "spatial": {},
            "patient": {},
            "series": {},
        }

        # Key pixel data tags
        pixel_tags = [
            'PixelSpacing', 'SliceThickness', 'ImageOrientationPatient',
            'ImagePositionPatient', 'SliceLocation',
            'WindowCenter', 'WindowWidth', 'RescaleSlope', 'RescaleIntercept',
            'BitsStored', 'BitsAllocated', 'PixelRepresentation', 'PhotometricInterpretation',
            'Rows', 'Columns', 'SamplesPerPixel',
            'KVP', 'ExposureTime', 'XRayTubeCurrent',
        ]

        for tag_name in pixel_tags:
            if hasattr(self._first_ds, tag_name):
                value = getattr(self._first_ds, tag_name)
                metadata["tags"][tag_name] = _dicom_value_to_json(value)

        # Derived spatial info
        if hasattr(self._first_ds, 'PixelSpacing'):
            ps = self._first_ds.PixelSpacing
            if len(ps) >= 2:
                metadata["spatial"]["pixel_spacing_mm"] = [float(ps[0]), float(ps[1])]

        if hasattr(self._first_ds, 'SliceThickness'):
            metadata["spatial"]["slice_spacing_mm"] = float(self._first_ds.SliceThickness)

        if hasattr(self._first_ds, 'ImageOrientationPatient'):
            iop = self._first_ds.ImageOrientationPatient
            metadata["spatial"]["orientation"] = _derive_orientation_from_iop(list(iop))

        if hasattr(self._first_ds, 'ImagePositionPatient'):
            ipp = self._first_ds.ImagePositionPatient
            metadata["spatial"]["origin_mm"] = [float(ipp[0]), float(ipp[1]), float(ipp[2])]

        # Patient/study info
        patient_tags = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'PatientAge', 'PatientWeight', 'PatientSize',
            'StudyInstanceUID', 'SeriesInstanceUID',
            'StudyDate', 'SeriesDate', 'AcquisitionDate',
            'StudyDescription', 'SeriesDescription',
            'Modality', 'Manufacturer', 'InstitutionName',
        ]

        for tag_name in patient_tags:
            if hasattr(self._first_ds, tag_name):
                value = getattr(self._first_ds, tag_name)
                metadata["patient"][tag_name] = _dicom_value_to_json(value)

        # Series-level info
        metadata["series"]["num_slices"] = self._num_slices
        metadata["series"]["directory"] = str(self.directory)

        return metadata