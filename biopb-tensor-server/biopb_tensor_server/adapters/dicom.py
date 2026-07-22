"""DICOM adapters for tensor storage.

Handles single DICOM files and multi-file DICOM series using pydicom.
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._scale import scale_by_label
from biopb_tensor_server.core.adapter_base import TensorAdapter
from biopb_tensor_server.core.chunk import content_version_from_path
from biopb_tensor_server.core.discovery import (
    ClaimContext,
    SourceClaim,
)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState


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
        elif hasattr(value, "value"):
            # Sequence items or data element wrappers
            return _dicom_value_to_json(value.value)
        else:
            return value
    except Exception:
        return str(value)


def _dicom_dtype(bits_stored: int, pixel_repr: int) -> str:
    """NumPy dtype string for a DICOM image from its ``BitsStored`` /
    ``PixelRepresentation`` (0 = unsigned, 1 = two's-complement signed).

    Widths above 32 bits have no integer DICOM encoding and fall back to
    ``float32``. Shared by the single-file and series adapters.
    """
    if bits_stored <= 8:
        return "uint8" if pixel_repr == 0 else "int8"
    if bits_stored <= 16:
        return "uint16" if pixel_repr == 0 else "int16"
    if bits_stored <= 32:
        return "uint32" if pixel_repr == 0 else "int32"
    return "float32"


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
        row_x * col_y - row_y * col_x,
    ]

    # Find dominant component
    max_idx = max(range(3), key=lambda i: abs(normal[i]))

    if max_idx == 0:
        return "sagittal"
    elif max_idx == 1:
        return "coronal"
    else:
        return "axial"


def _dicom_physical_scale(ds, dim_labels) -> Optional[Tuple[List[float], List[str]]]:
    """Per-dim physical size (mm) + unit from a DICOM dataset's spatial tags.

    Shared by the single-file and series adapters (both read the same tags off
    ``ds`` -- the sole file, or the series' first slice). ``PixelSpacing`` is
    ``[row-spacing, column-spacing]`` in mm, i.e. the Y then X pixel size; the
    through-plane spacing (``z`` / multi-frame ``frame`` axis) comes from
    ``SpacingBetweenSlices`` when present, else ``SliceThickness``. Non-spatial
    axes get ``0.0`` / ``""``. Returns ``None`` when no positive size is present.
    """
    row_sp = col_sp = None
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        try:
            row_sp = float(ps[0])
            col_sp = float(ps[1])
        except (TypeError, ValueError):
            row_sp = col_sp = None

    z_sp = None
    for attr in ("SpacingBetweenSlices", "SliceThickness"):
        val = getattr(ds, attr, None)
        if val is None:
            continue
        try:
            z_sp = float(val)
        except (TypeError, ValueError):
            z_sp = None
        if z_sp and z_sp > 0:
            break

    return scale_by_label(
        dim_labels,
        {"y": row_sp, "x": col_sp, "z": z_sp, "frame": z_sp},
        "mm",
    )


def _dicom_common_metadata(ds) -> dict:
    """Full per-file DICOM metadata (format / tags / spatial / patient) from ``ds``.

    Shared by the single-file and series adapters -- both read the same tags off
    one dataset (the sole file, or the series' first slice). The series adapter
    layers a ``series`` block on top.
    """
    metadata: dict = {
        "format": "dicom",
        "tags": {},
        "spatial": {},
        "patient": {},
    }

    # Key pixel data tags
    pixel_tags = [
        "PixelSpacing",
        "SliceThickness",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "SliceLocation",
        "InstanceNumber",
        "WindowCenter",
        "WindowWidth",
        "RescaleSlope",
        "RescaleIntercept",
        "BitsStored",
        "BitsAllocated",
        "PixelRepresentation",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
        "NumberOfFrames",
        "SamplesPerPixel",
        "KVP",
        "ExposureTime",
        "XRayTubeCurrent",
    ]
    for tag_name in pixel_tags:
        if hasattr(ds, tag_name):
            metadata["tags"][tag_name] = _dicom_value_to_json(getattr(ds, tag_name))

    # Derived spatial info
    if hasattr(ds, "PixelSpacing"):
        ps = ds.PixelSpacing
        if len(ps) >= 2:
            metadata["spatial"]["pixel_spacing_mm"] = [float(ps[0]), float(ps[1])]

    if hasattr(ds, "SliceThickness"):
        metadata["spatial"]["slice_spacing_mm"] = float(ds.SliceThickness)

    if hasattr(ds, "ImageOrientationPatient"):
        iop = ds.ImageOrientationPatient
        metadata["spatial"]["orientation"] = _derive_orientation_from_iop(list(iop))

    if hasattr(ds, "ImagePositionPatient"):
        ipp = ds.ImagePositionPatient
        metadata["spatial"]["origin_mm"] = [
            float(ipp[0]),
            float(ipp[1]),
            float(ipp[2]),
        ]

    if hasattr(ds, "SliceLocation"):
        metadata["spatial"]["slice_location_mm"] = float(ds.SliceLocation)

    # Patient/study info
    patient_tags = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "PatientWeight",
        "PatientSize",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "StudyDescription",
        "SeriesDescription",
        "Modality",
        "Manufacturer",
        "InstitutionName",
    ]
    for tag_name in patient_tags:
        if hasattr(ds, tag_name):
            metadata["patient"][tag_name] = _dicom_value_to_json(getattr(ds, tag_name))

    # Windowing parameters (for visualization)
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        try:
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            # Handle multi-value window settings
            if hasattr(wc, "__iter__") and not isinstance(wc, str):
                metadata["tags"]["WindowCenter"] = [float(v) for v in wc]
            else:
                metadata["tags"]["WindowCenter"] = float(wc)
            if hasattr(ww, "__iter__") and not isinstance(ww, str):
                metadata["tags"]["WindowWidth"] = [float(v) for v in ww]
            else:
                metadata["tags"]["WindowWidth"] = float(ww)
        except Exception:
            pass

    return metadata


# Image/Instance-level attributes that vary slice-to-slice within a series
# (DICOM PS3.3 assigns them to the Image IE, not Patient/Study/Series). Reading
# them off the first slice and reporting them as the series' value is misleading,
# so ``DicomSeriesAdapter.get_metadata`` drops them from the shared first-slice
# metadata. Surfacing the real per-slice values is tracked in #560.
_SERIES_NONINVARIANT_TAGS = frozenset(
    {
        # per-instance identity / ordinal
        "SOPInstanceUID",
        "InstanceNumber",
        # per-slice geometry (position along the stack)
        "ImagePositionPatient",
        "SliceLocation",
        # per-instance frame count
        "NumberOfFrames",
        # acquisition params subject to per-slice (dose) modulation
        "KVP",
        "ExposureTime",
        "XRayTubeCurrent",
    }
)
# Image-level display/calibration params (WindowCenter/WindowWidth, RescaleSlope/
# RescaleIntercept) are technically per-instance but in practice near-constant
# across a series and useful as series-level display defaults, so they are kept
# from the first slice until #560 surfaces real per-slice metadata.
# Derived ``spatial`` keys computed from the per-slice geometry tags above.
_SERIES_NONINVARIANT_SPATIAL = frozenset({"origin_mm", "slice_location_mm"})


# =============================================================================
# DicomAdapter - Single file
# =============================================================================


class DicomAdapter(TensorAdapter):
    """Adapter for single DICOM files.

    Handles .dcm and .dicom files with pixel data.
    Supports remote storage via fsspec (pydicom's dcmread accepts file-like objects).

    Chunk strategy:
    - Single frame: Single chunk for entire 2D image
    - Multi-frame: One chunk per frame (NumberOfFrames > 1)

    Uses pydicom for DICOM parsing and pixel data access.
    """

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim single DICOM files.

        Works for both local and remote files.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim if this is a valid DICOM file with pixel data capability, None otherwise
        """
        if not ctx.is_file():
            return None

        # Check extension
        name = ctx.name.lower()
        if not name.endswith((".dcm", ".dicom")):
            return None

        # Cloud-storage phase 2: reading the DICOM header (even
        # stop_before_pixels) recalls a non-resident placeholder. Defer the
        # header read + Rows/Columns validation; claim by extension and resolve
        # on first access.
        if not ctx.is_resident():
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type="dicom",
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
                unresolved=True,
            )

        try:
            import pydicom

            if ctx.is_remote:
                # Remote: read via file-like object
                with ctx.store.open(ctx._remote_path, mode="rb") as fobj:
                    ds = pydicom.dcmread(fobj, stop_before_pixels=True)
            else:
                # Local: read metadata only (no pixel data)
                ds = pydicom.dcmread(ctx.path_str, stop_before_pixels=True)

            # Check for image-related tags (indicating pixel data capability)
            if not (hasattr(ds, "Rows") and hasattr(ds, "Columns")):
                return None

            state.try_claim_path(ctx.path_str)

            return SourceClaim(
                source_type="dicom",
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
            )
        except Exception:
            return None

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "DicomAdapter":
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            DicomAdapter instance
        """
        import pydicom

        if source.is_remote:
            # Remote storage: use fsspec file-like object
            from fsspec.core import url_to_fs

            # Build storage_options from credentials_config if provided
            storage_options = {}
            if credentials_config:
                profile = credentials_config.get_profile(source.credentials_profile)
                if profile:
                    storage_options = profile.to_storage_options()

            fs, fs_path = url_to_fs(source.url, storage_options=storage_options)
            with fs.open(fs_path, mode="rb") as fobj:
                ds = pydicom.dcmread(fobj)
        else:
            # Local filesystem
            ds = pydicom.dcmread(str(source.url))

        return cls(ds, source.source_id, source.dim_labels, source_url=str(source.url))

    def __init__(
        self,
        dicom_dataset,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
    ):
        """Initialize DICOM adapter.

        Args:
            dicom_dataset: pydicom Dataset object
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels
            source_url: Optional source URL (overrides filename-derived path)
        """
        self.ds = dicom_dataset
        self.source_id = source_id
        self._io_lock = threading.Lock()

        # Source-level metadata
        if source_url:
            self._source_url = source_url
        elif hasattr(dicom_dataset, "filename"):
            self._source_url = str(dicom_dataset.filename)
        else:
            self._source_url = ""
        # Cheap content_version from the file's stat signature (#178): O(1),
        # folded into minted chunk_ids so a re-saved file gets a fresh cache
        # namespace. None (unresolved / non-file url) leaves the source unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = "dicom"

        # Get shape info
        rows = int(self.ds.get("Rows", 0))
        cols = int(self.ds.get("Columns", 0))
        num_frames = int(self.ds.get("NumberOfFrames", 1))

        if num_frames > 1:
            self._shape = (num_frames, rows, cols)
            self._is_multiframe = True
        else:
            self._shape = (rows, cols)
            self._is_multiframe = False

        # Get dtype from pixel representation
        bits_stored = int(self.ds.get("BitsStored", 16))
        pixel_repr = int(self.ds.get("PixelRepresentation", 0))
        self._dtype = _dicom_dtype(bits_stored, pixel_repr)

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            if self._is_multiframe:
                self.dim_labels = ["frame", "y", "x"]
            else:
                self.dim_labels = ["y", "x"]

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
        slices = self._bounds_to_slices(bounds)

        # Serialize IO for thread safety
        with self._io_lock:
            pixel_data = self.ds.pixel_array
            return pixel_data[slices]

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dim pixel size (mm) from this file's ``PixelSpacing`` etc."""
        return _dicom_physical_scale(self.ds, self.dim_labels)

    def get_metadata(self) -> dict:
        """Extract DICOM metadata: format identifier, tags, derived spatial info,
        and patient/study info."""
        return _dicom_common_metadata(self.ds)


# =============================================================================
# DicomSeriesAdapter - Multi-file series
# =============================================================================


class DicomSeriesAdapter(TensorAdapter):
    """Adapter for multi-file DICOM series forming a 3D volume.

    Handles directories where multiple DICOM files share the same SeriesInstanceUID.
    Each file represents one slice of the volume.

    Multi-node claim: claims directory + all DICOM files in the series.

    Chunk strategy: One chunk per slice (file), shape = [1, H, W]
    """

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories containing DICOM series.

        Detects multiple DICOM files sharing the same SeriesInstanceUID.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim if valid DICOM series directory, None otherwise
        """
        # Only support local directories for now
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Cloud-storage policy (biopb/biopb): DICOM-series *membership* is derived
        # by dcmread'ing every slice header and grouping by SeriesInstanceUID --
        # a directory routinely holds several distinct series, so the dir is NOT
        # the dataset boundary and the grouping cannot be deferred without a
        # content read. Under a cloud root we therefore do NOT group: return None
        # so the single-file DicomAdapter claims each .dcm as its own source
        # (deferred to unresolved by its own residency gate). This must hold at
        # resolve too (slices are resident by then), which is why the gate is
        # ctx.cloud_root, not residency. A DICOM series degrades to N single-file
        # sources under cloud (transcode to OME-Zarr for proper support).
        if ctx.cloud_root:
            return None

        # Find DICOM files. Route through ctx.glob (not ctx._path.glob) so the
        # snapshot's cached child listing serves the match without re-reading the
        # directory (biopb/biopb#65); unwrap to the underlying Path objects the
        # rest of this method (dcmread / try_claim_path) consumes.
        dcm_files = [
            c._path
            for c in (ctx.glob("*.dcm") + ctx.glob("*.DICOM") + ctx.glob("*.dicom"))
        ]

        # Need at least 2 files for a series
        if len(dcm_files) < 2:
            return None

        try:
            import pydicom

            # Group the image-capable files by SeriesInstanceUID. Keying off the
            # first globbed file's series (the old approach) made the result depend
            # on glob order: a directory holding a singleton series ahead of a real
            # multi-file series was wrongly rejected whenever the filesystem yielded
            # the singleton first. Grouping every file and then choosing makes the
            # decision independent of discovery order.
            series_to_files = {}
            for f in dcm_files:
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                except Exception:
                    continue
                series_uid = getattr(ds, "SeriesInstanceUID", None)
                # Rows/Columns indicate pixel-data capability.
                if series_uid is None or not (
                    hasattr(ds, "Rows") and hasattr(ds, "Columns")
                ):
                    continue
                series_to_files.setdefault(str(series_uid), []).append(f)

            # Claim the largest series with at least two slices.
            series_uid, series_files = max(
                series_to_files.items(),
                key=lambda kv: len(kv[1]),
                default=(None, []),
            )
            if len(series_files) < 2:
                return None

            # Claim directory + all series files
            state.try_claim_path(ctx.path_str)
            for f in series_files:
                state.try_claim_path(f)

            return SourceClaim(
                source_type="dicom-series",
                primary_path=ctx.path_str,
                extra_config={
                    "num_slices": len(series_files),
                    "series_uid": str(series_uid),
                },
            )

        except Exception:
            return None

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "DicomSeriesAdapter":
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
        import pydicom

        self.directory = Path(directory)
        self.source_id = source_id
        self._io_lock = threading.Lock()

        # Source-level metadata
        self._source_url = str(directory)
        # Cheap content_version from the directory's stat signature (#178): O(1)
        # dir mtime, which flips on member add/remove/rename -- the right signal
        # for a multi-file series. None (unresolved url) leaves it unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = "dicom-series"

        # Find and sort DICOM files
        dcm_files = (
            list(self.directory.glob("*.dcm"))
            + list(self.directory.glob("*.DICOM"))
            + list(self.directory.glob("*.dicom"))
        )

        # Read first file to get series UID and sort criteria
        first_ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        series_uid = first_ds.SeriesInstanceUID

        # Collect files with metadata for sorting
        file_info = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if (
                    hasattr(ds, "SeriesInstanceUID")
                    and ds.SeriesInstanceUID == series_uid
                ):
                    # Get sorting key
                    instance_num = int(ds.get("InstanceNumber", -1))
                    slice_loc = (
                        float(ds.get("SliceLocation", 0))
                        if hasattr(ds, "SliceLocation")
                        else 0
                    )
                    ipp = (
                        ds.get("ImagePositionPatient", [0, 0, 0])
                        if hasattr(ds, "ImagePositionPatient")
                        else [0, 0, 0]
                    )
                    z_pos = float(ipp[2]) if len(ipp) >= 3 else 0

                    file_info.append(
                        {
                            "path": f,
                            "instance_number": instance_num,
                            "slice_location": slice_loc,
                            "z_position": z_pos,
                        }
                    )
            except Exception:
                continue

        # Sort files - prefer InstanceNumber, then SliceLocation, then z_position
        if any(info["instance_number"] >= 0 for info in file_info):
            # Sort by InstanceNumber
            file_info.sort(key=lambda x: x["instance_number"])
        elif any(info["slice_location"] != 0 for info in file_info):
            # Sort by SliceLocation
            file_info.sort(key=lambda x: x["slice_location"])
        else:
            # Sort by z_position from ImagePositionPatient
            file_info.sort(key=lambda x: x["z_position"])

        self.dicom_files = [info["path"] for info in file_info]
        self._num_slices = len(self.dicom_files)

        if self._num_slices == 0:
            raise ValueError(f"No valid DICOM files found in series: {directory}")

        # Read the first slice's header (shape/dtype + the tags get_metadata and
        # _physical_scale later report). Header-only -- everything read from
        # _first_ds lives in the header, so stop_before_pixels avoids loading and
        # retaining a whole slice's pixel data for the source's lifetime.
        first_full = pydicom.dcmread(str(self.dicom_files[0]), stop_before_pixels=True)
        rows = int(first_full.Rows)
        cols = int(first_full.Columns)
        bits_stored = int(first_full.get("BitsStored", 16))
        pixel_repr = int(first_full.get("PixelRepresentation", 0))
        self._dtype = _dicom_dtype(bits_stored, pixel_repr)

        self._shape = (self._num_slices, rows, cols)
        self._rows = rows
        self._cols = cols

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            self.dim_labels = ["z", "y", "x"]

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
        slices = self._bounds_to_slices(bounds)
        slice_start = int(bounds.start[0])
        slice_stop = int(bounds.stop[0])

        # Determine output shape
        out_shape = tuple(
            int(e - s) for s, e in zip(bounds.start, bounds.stop, strict=True)
        )
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

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dim pixel size (mm) from the series' first-slice spatial tags.

        X/Y from ``PixelSpacing``; the ``z`` (slice) axis from
        ``SpacingBetweenSlices`` / ``SliceThickness`` -- both read off
        ``_first_ds``, the same slice ``get_metadata`` reports from.
        """
        return _dicom_physical_scale(self._first_ds, self.dim_labels)

    def get_metadata(self) -> dict:
        """Extract DICOM series metadata: the first slice's *series-invariant*
        per-file metadata plus a series-level block.

        The shared extractor pulls every field from the first slice; the
        Image-level fields that vary slice-to-slice (see
        ``_SERIES_NONINVARIANT_TAGS``) are dropped here rather than advertised as
        series-wide. Full per-slice metadata is tracked in #560.
        """
        metadata = _dicom_common_metadata(self._first_ds)
        for section in ("tags", "patient"):
            for name in _SERIES_NONINVARIANT_TAGS:
                metadata[section].pop(name, None)
        for name in _SERIES_NONINVARIANT_SPATIAL:
            metadata["spatial"].pop(name, None)
        metadata["series"] = {
            "num_slices": self._num_slices,
            "directory": str(self.directory),
        }
        return metadata
