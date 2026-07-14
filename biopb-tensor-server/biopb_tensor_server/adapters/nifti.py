"""NIfTI adapter for tensor storage.

Handles .nii and .nii.gz files using nibabel.
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState


class NiftiAdapter(SourceAdapter, TensorAdapter):
    """Adapter for NIfTI files (.nii and .nii.gz).

    Uses nibabel for lazy loading and header parsing.
    Supports remote storage via temp file download (nibabel doesn't support file-like objects).

    Chunk ID format:
    - array_id prefix (via encode_chunk_id)
    - single byte 'W' for whole array

    Single chunk strategy - base class handles splitting for oversized arrays.
    """

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim NIfTI files (.nii or .nii.gz).

        Works for both local and remote files.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim if this is a NIfTI file, None otherwise
        """
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            return None

        state.try_claim_path(ctx.path_str)

        return SourceClaim(
            source_type="nifti",
            primary_path=ctx.path_str,
            is_remote=ctx.is_remote,
        )

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "NiftiAdapter":
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            NiftiAdapter instance
        """
        import nibabel as nib

        if source.is_remote:
            # Remote storage: download to temp file (nibabel doesn't support file-like objects)
            import tempfile

            from fsspec.core import url_to_fs

            # Build storage_options from credentials_config if provided
            storage_options = {}
            if credentials_config:
                profile = credentials_config.get_profile(source.credentials_profile)
                if profile:
                    storage_options = profile.to_storage_options()

            fs, fs_path = url_to_fs(source.url, storage_options=storage_options)

            # Determine suffix for temp file
            suffix = ".nii.gz" if source.url.lower().endswith(".nii.gz") else ".nii"

            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)

            fs.get_file(fs_path, str(tmp_path))

            # Load NIfTI from temp file
            nifti_img = nib.load(str(tmp_path))

            # Note: temp file is NOT deleted - nibabel caches data internally
            # The OS will clean up temp files eventually

            return cls(
                nifti_img,
                source.source_id,
                source.dim_labels,
                source_url=str(source.url),
                temp_file=tmp_path,
            )
        else:
            # Local filesystem
            nifti_img = nib.load(str(source.url))
            return cls(
                nifti_img,
                source.source_id,
                source.dim_labels,
                source_url=str(source.url),
            )

    def __init__(
        self,
        nifti_img,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        temp_file: Optional[Path] = None,
    ):
        """Initialize NIfTI adapter.

        Args:
            nifti_img: nibabel Nifti1Image or Nifti2Image object
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (overrides header-derived labels)
            source_url: Optional source URL (overrides file_map-derived path)
            temp_file: Optional path to temp file for remote sources (for cleanup tracking)
        """
        self.nifti_img = nifti_img
        self.source_id = source_id
        self.header = nifti_img.header
        self._io_lock = threading.Lock()
        self._temp_file = temp_file  # Track temp file for potential cleanup

        # Source-level metadata for DataSourceDescriptor
        if source_url:
            self._source_url = source_url
        elif hasattr(nifti_img, "file_map"):
            # Try to get file path from nibabel
            files = list(nifti_img.file_map.keys())
            self._source_url = files[0] if files else ""
        else:
            self._source_url = ""
        self._source_type = "nifti"

        # Get shape and dtype from header
        # NIfTI dim array: dim[0] = ndim, dim[1-7] = dimensions
        dim_info = self.header.get("dim", None)
        if dim_info is not None:
            ndim = int(dim_info[0])
            self._shape = tuple(int(dim_info[i]) for i in range(1, ndim + 1))
        else:
            self._shape = tuple(nifti_img.shape)

        # NIfTI uses slope/intercept scaling to represent physical values.
        # Scaled data is always float64, so we report float64 as the dtype
        # and return scaled float64 values via nibabel's lazy slicing.
        self._dtype = "float64"

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            self.dim_labels = self._derive_dim_labels()

    def _derive_dim_labels(self) -> List[str]:
        """Derive dimension labels from NIfTI header."""
        ndim = len(self._shape)

        # Check xyzt_units for dimension semantics
        # nibabel encoding: xyzt_units = spatial_unit + temporal_unit
        # where temporal values are: 8=seconds, 16=ms, 24=us, 32=Hz, etc.
        units = self.header.get("xyzt_units", 0)
        if isinstance(units, np.ndarray):
            units = units.item()
        spatial_unit = units & 0x07  # 1=meter, 2=mm, 3=um
        time_unit = units - spatial_unit  # 8=sec, 16=ms, 24=us

        pixdim = self.header.get("pixdim", [0] * 8)
        # If pixdim[4] (time step) is present and > 0, we likely have a time dimension
        if ndim > 4 and pixdim[4] > 0 and time_unit == 0:
            time_unit = 8  # Treat as seconds for labeling purposes

        # NIfTI dimensions: typically (x, y, z) or (t, x, y, z) or (t, c, x, y, z)
        # Dimension order in nibabel is as stored in file
        labels = []

        if ndim == 1:
            labels = ["x"]
        elif ndim == 2:
            labels = ["x", "y"]
        elif ndim == 3:
            # Could be (x,y,z) or (t,x,y) - check intent and units
            if pixdim[1] == pixdim[2] != pixdim[3] or time_unit == 0:
                labels = ["x", "y", "z"]
            else:
                labels = ["t", "x", "y"]
        elif ndim == 4:
            if time_unit >= 8:
                labels = ["t", "x", "y", "z"]
            else:
                labels = ["c", "x", "y", "z"]
        elif ndim == 5:
            # Could be vector/tensor data
            labels = ["v", "t", "x", "y", "z"]
        else:
            # Generic labels for higher dimensions
            labels = [f"dim{i}" for i in range(ndim)]

        return labels

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
        """Read data within bounds from NIfTI file.

        Returns scaled float64 data (applying slope/intercept if present).

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with scaled float64 data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape
        """
        super().get_data(bounds)
        slices = tuple(
            slice(int(s), int(e))
            for s, e in zip(bounds.start, bounds.stop, strict=True)
        )

        # nibabel dataobj lazy slicing applies slope/intercept scaling.
        # nibabel handles thread safety internally, so no io_lock needed.
        dataobj = self.nifti_img.dataobj
        result = dataobj[slices]
        # Cast to float64 defensively (in case no scaling is applied)
        return np.asanyarray(result, dtype=np.float64)

    def get_metadata(self) -> dict:
        """Extract NIfTI header metadata.

        Returns:
            Dictionary with format identifier, header fields, and affine matrix
        """
        metadata = {
            "format": "nifti",
            "header": {},
            "spatial": {},
        }

        # Key header fields
        header_fields = [
            "dim",
            "pixdim",
            "datatype",
            "bitpix",
            "intent_code",
            "intent_name",
            "intent_p1",
            "intent_p2",
            "intent_p3",
            "sform_code",
            "qform_code",
            "xyzt_units",
            "cal_min",
            "cal_max",
            "slice_code",
            "slice_start",
            "slice_end",
            "slice_duration",
            "toffset",
            "descrip",
            "aux_file",
        ]

        for field in header_fields:
            try:
                value = self.header[field]
                # Handle numpy arrays properly (including 0-d scalars)
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        # Scalar numpy array - use item() to get the value
                        metadata["header"][field] = value.item()
                    else:
                        # Multi-dimensional array - convert to list
                        metadata["header"][field] = list(value)
                else:
                    metadata["header"][field] = value
            except KeyError:
                pass

        # Affine transformation matrix (most important for spatial registration)
        affine = self.nifti_img.affine
        metadata["spatial"]["affine_matrix"] = [list(row) for row in affine]

        # Voxel dimensions from pixdim
        # pixdim[0] is qfac, pixdim[1-7] are actual dimensions
        pixdim = self.header.get("pixdim", None)
        if pixdim is not None:
            # Convert to list if it's a numpy array
            if isinstance(pixdim, np.ndarray):
                pixdim_list = list(pixdim)
            else:
                pixdim_list = list(pixdim)
            if len(pixdim_list) >= 4:
                metadata["spatial"]["voxel_size_mm"] = pixdim_list[1:4]
                if len(pixdim_list) >= 5 and pixdim_list[4] > 0:
                    metadata["spatial"]["time_step"] = float(pixdim_list[4])

        # Units interpretation (nibabel encoding: spatial + temporal)
        # where temporal values are: 8=seconds, 16=ms, 24=us
        units = self.header.get("xyzt_units", 0)
        if isinstance(units, np.ndarray):
            units = units.item()
        spatial_unit = units & 0x07
        time_unit = units - spatial_unit

        unit_map = {
            0: "unknown",
            1: "meter",
            2: "mm",
            3: "um",
            8: "s",
            16: "ms",
            24: "us",
        }
        metadata["spatial"]["units"] = unit_map.get(spatial_unit, "unknown")
        if time_unit >= 8:
            metadata["header"]["time_units"] = unit_map.get(time_unit, "unknown")

        # Intent interpretation (what the data represents)
        intent_code = self.header.get("intent_code", 0)
        if isinstance(intent_code, np.ndarray):
            intent_code = intent_code.item()
        intent_map = {
            0: "none",
            1001: "estimate",
            1002: "label",
            1003: "vector",
            1004: "time_series",
            1005: "mesh",
            1006: "matrix",
            1007: "point_set",
            1008: "triangle",
            1009: "quaternion",
            1010: "dimless",
        }
        metadata["header"]["intent"] = intent_map.get(
            intent_code, f"code_{intent_code}"
        )

        return metadata
