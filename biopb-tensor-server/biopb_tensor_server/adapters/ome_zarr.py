"""OME-Zarr adapter for tensor storage.

Extends ZarrAdapter with OME multiscales metadata support.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from urllib.parse import urlparse

from biopb.tensor.descriptor_pb2 import PyramidLevel, SliceHint, TensorDescriptor

from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.base import TensorReadPlan
from biopb_tensor_server.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.downsample import normalize_reduction_method

if TYPE_CHECKING:
    from biopb_tensor_server.base import BackendAdapter
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState


logger = logging.getLogger(__name__)


class OmeZarrAdapter(ZarrAdapter):
    """Adapter for OME-Zarr (OME-NGFF) datasets.

    Extends ZarrAdapter with OME metadata support:
    - multiscales: Multiple resolution levels
    - axes: Dimension labels with types (channel, space, time)
    - coordinate_transformations: Physical scales
    - omero: Channel colors, names

    Supports both local and remote storage (S3, GCS, etc.) via fsspec.

    For HCS (High-Content Screening) plates:
    - Plate as source, wells/fields as tensors
    - array_id = '{source_id}/{well_name}/{field_index}'
    - _single_tensor_source = False (multi-tensor)

    Chunk ID format: Same as ZarrAdapter
    - array_id prefix
    - chunk key (UTF-8, e.g., "0/1/2")

    Note: This adapter can be used in multiple ways:
    1. Source-level (HCS plate): Manages wells/fields, get_tensor_adapter() returns
       ZarrAdapter instances for specific fields
    2. Source-level (single image): Manages resolution levels, get_level_adapter() returns
       ZarrAdapter instances for specific levels
    3. Level-specific: Created with a specific level array, acts as single-tensor
    """

    # Default: single-tensor (level-specific usage)
    # Set to False for HCS plates in __init__
    _single_tensor_source = True

    # HCS-specific state (populated if _is_hcs_plate is True)
    _is_hcs_plate: bool = False
    _plate_root_path: Optional[str] = None  # Path to plate root directory
    _hcs_well_paths: dict = {}  # well_name -> zarr path (e.g., 'A01' -> 'A01')
    _hcs_field_count: int = 0
    _hcs_well_metadata: dict = {}  # well_name -> well .zattrs
    _field_adapters: dict = {}  # field_key -> cached adapter

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim .zarr directories with OME multiscales metadata.

        Detects both regular OME-Zarr multiscale images and HCS plate datasets.
        HCS plates are detected by 'plate' key in .zattrs (checked before multiscales).

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim if this is an OME-Zarr dataset, None otherwise
        """
        # Must be a directory ending in .zarr
        if not ctx.is_dir() or not ctx.name.endswith(".zarr"):
            return None

        # A top-level .zarray / zarr.json means a bare zarr *array*, never an
        # OME-Zarr multiscales group (those are a .zgroup of arrays). Decline so
        # ZarrAdapter's definitive, read-free claim wins -- otherwise a
        # non-resident bare array would be deferred as a provisional "ome-zarr"
        # guess (the .zattrs residency defer below), shadowing the certain "zarr"
        # type until resolve. Existence is a stat, so this stays recall-free.
        if ctx.join(".zarray").exists() or ctx.join("zarr.json").exists():
            return None

        zattrs_ctx = ctx.join(".zattrs")
        if not zattrs_ctx.exists():
            return None

        # Cloud-storage phase 2: if the .zattrs sidecar is a non-resident cloud
        # placeholder, reading it would trigger a whole-file recall (or block
        # offline). Defer the read: recognize the store structurally (a .zarr dir
        # with a .zattrs) and claim it provisionally as ome-zarr -- the type that
        # owns .zarr dirs. The exact subtype (ome-zarr vs ome-zarr-hcs), dim
        # labels, and any HCS field set are resolved on first access.
        if not zattrs_ctx.is_resident():
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type="ome-zarr",
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
                unresolved=True,
            )

        try:
            zattrs = json.loads(ctx.read_text(".zattrs"))

            # Check for HCS plate metadata first (higher priority)
            if "plate" in zattrs:
                state.try_claim_path(ctx.path_str)
                return SourceClaim(
                    source_type="ome-zarr-hcs",
                    primary_path=ctx.path_str,
                    is_remote=ctx.is_remote,
                )

            # Check for OME multiscales key
            if "multiscales" not in zattrs:
                return None
        except (json.JSONDecodeError, KeyError, Exception):
            return None

        state.try_claim_path(ctx.path_str)
        return SourceClaim(
            source_type="ome-zarr",
            primary_path=ctx.path_str,
            is_remote=ctx.is_remote,
        )

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "OmeZarrAdapter":
        """Create adapter instance from SourceConfig.

        Handles both regular OME-Zarr multiscale images and HCS plate datasets.
        For HCS plates, opens the plate root group.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            OmeZarrAdapter instance
        """
        import json

        import zarr
        from fsspec.core import url_to_fs

        zarr_path = str(source.url)

        if source.is_remote:
            # Remote storage: use fsspec FSStore
            # Build storage_options from credentials_config if provided
            storage_options = {}
            if credentials_config:
                profile = credentials_config.get_profile(source.credentials_profile)
                if profile:
                    storage_options = profile.to_storage_options()

            fs, fs_path = url_to_fs(zarr_path, storage_options=storage_options)
            store = zarr.FSStore(fs, fs_path)

            # Read zattrs to determine type (HCS plate or multiscale)
            zattrs_bytes = fs.cat_file(fs_path.rstrip("/") + "/.zattrs")
            zattrs = json.loads(zattrs_bytes)

            # Check for HCS plate metadata
            if "plate" in zattrs:
                # HCS plate: open group and first field for array reference
                root = zarr.open_group(store, mode="r")

                # Find first well and field for initial array
                plate_meta = zattrs.get("plate", {})
                wells = plate_meta.get("wells", [])
                if wells:
                    first_well_path = wells[0].get("path", "0")

                    # Read well .zattrs
                    well_zattrs_bytes = fs.cat_file(
                        fs_path.rstrip("/")
                        + "/"
                        + first_well_path.rstrip("/")
                        + "/.zattrs"
                    )
                    well_zattrs = json.loads(well_zattrs_bytes)
                    well_info = well_zattrs.get("well", {})
                    images = well_info.get("images", [])
                    if images:
                        first_field_path = images[0].get("path", "0")

                        # Read field .zattrs
                        field_zattrs_bytes = fs.cat_file(
                            fs_path.rstrip("/")
                            + "/"
                            + first_well_path.rstrip("/")
                            + "/"
                            + first_field_path.rstrip("/")
                            + "/.zattrs"
                        )
                        field_zattrs = json.loads(field_zattrs_bytes)

                        # Get first resolution level
                        multiscales = field_zattrs.get("multiscales", [])
                        if multiscales:
                            datasets = multiscales[0].get("datasets", [])
                            if datasets:
                                resolution_path = datasets[0].get("path", "0")
                                arr_path = (
                                    first_well_path.rstrip("/")
                                    + "/"
                                    + first_field_path.rstrip("/")
                                    + "/"
                                    + resolution_path
                                )
                                arr = zarr.open_array(store, path=arr_path, mode="r")
                            else:
                                arr = zarr.open_array(store, mode="r")
                        else:
                            arr = zarr.open_array(store, mode="r")
                    else:
                        arr = zarr.open_array(store, mode="r")
                else:
                    arr = zarr.open_array(store, mode="r")
            else:
                # Regular multiscale image
                resolution_path = "0"
                if "multiscales" in zattrs and zattrs["multiscales"]:
                    datasets = zattrs["multiscales"][0].get("datasets", [])
                    if datasets:
                        resolution_path = datasets[0].get("path", "0")

                # Open the group and then the resolution array
                root = zarr.open_group(store, mode="r")
                if resolution_path in root:
                    arr = root[resolution_path]
                else:
                    arr = zarr.open_array(store, mode="r")
        else:
            # Local filesystem
            store = zarr.DirectoryStore(zarr_path)

            try:
                with open(str(Path(zarr_path) / ".zattrs")) as f:
                    zattrs = json.load(f)

                # Check for HCS plate metadata
                if "plate" in zattrs:
                    # HCS plate: open group and first field for array reference
                    root = zarr.open_group(zarr_path, mode="r")

                    # Find first well and field for initial array
                    plate_meta = zattrs.get("plate", {})
                    wells = plate_meta.get("wells", [])
                    if wells:
                        first_well_path = wells[0].get("path", "0")

                        # Read well .zattrs
                        well_zattrs_path = str(
                            Path(zarr_path) / first_well_path.rstrip("/") / ".zattrs"
                        )
                        if Path(well_zattrs_path).exists():
                            with open(well_zattrs_path) as wf:
                                well_zattrs = json.load(wf)
                                well_info = well_zattrs.get("well", {})
                                images = well_info.get("images", [])
                                if images:
                                    first_field_path = images[0].get("path", "0")

                                    # Read field .zattrs
                                    field_zattrs_path = str(
                                        Path(zarr_path)
                                        / first_well_path.rstrip("/")
                                        / first_field_path.rstrip("/")
                                        / ".zattrs"
                                    )
                                    if Path(field_zattrs_path).exists():
                                        with open(field_zattrs_path) as ff:
                                            field_zattrs = json.load(ff)

                                            # Get first resolution level
                                            multiscales = field_zattrs.get(
                                                "multiscales", []
                                            )
                                            if multiscales:
                                                datasets = multiscales[0].get(
                                                    "datasets", []
                                                )
                                                if datasets:
                                                    resolution_path = datasets[0].get(
                                                        "path", "0"
                                                    )
                                                    arr_path = str(
                                                        Path(zarr_path)
                                                        / first_well_path.rstrip("/")
                                                        / first_field_path.rstrip("/")
                                                        / resolution_path
                                                    )
                                                    arr = zarr.open_array(
                                                        arr_path, mode="r"
                                                    )
                                                else:
                                                    arr = zarr.open_array(
                                                        zarr_path, mode="r"
                                                    )
                                            else:
                                                arr = zarr.open_array(
                                                    zarr_path, mode="r"
                                                )
                                    else:
                                        arr = zarr.open_array(zarr_path, mode="r")
                                else:
                                    arr = zarr.open_array(zarr_path, mode="r")
                        else:
                            arr = zarr.open_array(zarr_path, mode="r")
                    else:
                        arr = zarr.open_array(zarr_path, mode="r")
                else:
                    # Regular multiscale image
                    resolution_path = "0"
                    if "multiscales" in zattrs and zattrs["multiscales"]:
                        datasets = zattrs["multiscales"][0].get("datasets", [])
                        if datasets:
                            resolution_path = datasets[0].get("path", "0")

                    root = zarr.open_group(zarr_path, mode="r")
                    if resolution_path in root:
                        arr = root[resolution_path]
                    else:
                        arr = zarr.open_array(store, mode="r")
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                arr = zarr.open_array(zarr_path, mode="r")

        return cls(arr, source.source_id, source.dim_labels)

    def __init__(
        self,
        zarr_array,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        resolution_level: int = 0,
    ):
        """Initialize OME-Zarr adapter.

        Args:
            zarr_array: Zarr array object (from specific resolution level)
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (overrides OME metadata)
            resolution_level: Which resolution level to use (default 0)
        """
        # Initialize base ZarrAdapter first
        # We'll override dim_labels below if OME metadata provides better ones
        super().__init__(zarr_array, source_id, dim_labels)

        self.resolution_level = resolution_level

        # Try to read OME metadata from .zattrs
        self.ome_metadata = {}
        self.axes = []
        self.channel_names = []

        # Determine store path for reading .zattrs
        store = zarr_array.store
        store_str = str(store)
        if store_str.startswith("file://"):
            store_path = str(urlparse(store_str).path)
        elif hasattr(store, "path"):
            # DirectoryStore has 'path' attribute
            store_path = str(store.path)
        elif hasattr(store, "root"):
            store_path = str(store.root)
        else:
            store_path = store_str

        # Navigate to find the plate root .zattrs (might be multiple levels up for HCS)
        # For HCS plates, array is at plate.zarr/A01/0/0, but plate .zattrs is at plate.zarr/
        # We need to find .zattrs with 'plate' key, not just any .zattrs
        plate_root_path = None
        zattrs = None

        # Navigate up to find .zattrs with plate metadata (HCS) or multiscales (single image)
        current_path = store_path.rstrip("/")
        while current_path:
            candidate_zattrs_path = os.path.join(current_path, ".zattrs")
            if os.path.exists(candidate_zattrs_path):
                try:
                    with open(candidate_zattrs_path) as f:
                        candidate_zattrs = json.load(f)

                    # Check for HCS plate metadata first (highest priority)
                    if "plate" in candidate_zattrs:
                        plate_root_path = current_path
                        zattrs = candidate_zattrs
                        break  # Found plate root, stop searching

                    # If we found multiscales without plate, this might be a single image
                    # But continue searching up to see if there's a plate above
                    if "multiscales" in candidate_zattrs and zattrs is None:
                        # Save this as fallback (single image case)
                        plate_root_path = current_path
                        zattrs = candidate_zattrs
                except (OSError, json.JSONDecodeError):
                    pass

            # Move up one level; stop at the filesystem root. Comparing against
            # the parent (rather than against '/') terminates correctly on both
            # POSIX ('/') and Windows drive roots ('C:\\'), where dirname() is a
            # fixed point -- the old '/' check spun forever on Windows.
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break
            current_path = parent_path

        if zattrs is not None:
            self.ome_metadata = zattrs

            # Check for HCS plate metadata first
            if "plate" in zattrs:
                self._is_hcs_plate = True
                self._single_tensor_source = False  # Multi-tensor!
                self._source_type = "ome-zarr-hcs"
                self._plate_root_path = plate_root_path  # Save for field adapters
                self._parse_hcs_plate_structure(plate_root_path)
            elif "multiscales" in zattrs:
                self._source_type = "ome-zarr"
                self.axes = zattrs["multiscales"][0].get("axes", [])
                if "omero" in zattrs:
                    channels = zattrs["omero"].get("channels", [])
                    self.channel_names = [
                        ch.get("label", f"ch{i}") for i, ch in enumerate(channels)
                    ]

        # Override dimension labels from OME metadata if not explicitly provided
        if dim_labels is None and self.axes:
            self.dim_labels = [
                ax.get("name", f"dim{i}") if isinstance(ax, dict) else str(ax)
                for i, ax in enumerate(self.axes)
            ]

        # Cache for level adapters (precomputed pyramid levels)
        self._level_adapters: dict = {}

    def _parse_hcs_plate_structure(self, store_path: str) -> None:
        """Parse HCS plate metadata from plate.zattrs and well .zattrs files.

        Populates:
        - _hcs_well_paths: dict mapping well_name -> well path
        - _hcs_field_count: number of fields per well
        - _hcs_well_metadata: dict mapping well_name -> well .zattrs

        Args:
            store_path: Path to the plate root directory
        """
        plate_meta = self.ome_metadata.get("plate", {})
        wells = plate_meta.get("wells", [])

        # Parse well paths from plate metadata
        # wells is a list of dicts with 'path' and optionally 'row_index', 'column_index'
        well_paths = {}
        for well_info in wells:
            well_path = well_info.get("path", "")
            if well_path:
                # Extract well name from path (e.g., 'A01' from 'A01' or '0' from '0')
                well_name = well_path.rstrip("/").split("/")[-1]
                well_paths[well_name] = well_path

        self._hcs_well_paths = well_paths

        # Read well metadata for each well to get field count
        field_count = 0
        well_metadata = {}
        for well_name, well_path in well_paths.items():
            well_zattrs_path = os.path.join(
                store_path, well_path.rstrip("/"), ".zattrs"
            )
            if os.path.exists(well_zattrs_path):
                try:
                    with open(well_zattrs_path) as f:
                        well_zattrs = json.load(f)
                        well_metadata[well_name] = well_zattrs

                        # Get field count from well metadata
                        # well metadata has 'well' key with 'images' list
                        well_info = well_zattrs.get("well", {})
                        images = well_info.get("images", [])
                        if len(images) > field_count:
                            field_count = len(images)
                except (OSError, json.JSONDecodeError):
                    pass

        self._hcs_field_count = field_count
        self._hcs_well_metadata = well_metadata

        # Get axes/dim_labels from first field's multiscales metadata
        if well_metadata:
            first_well = list(well_metadata.values())[0]
            well_info = first_well.get("well", {})
            images = well_info.get("images", [])
            if images:
                # Path to first field
                first_field_path = images[0].get("path", "0")
                # Try to read field .zattrs for axes
                first_well_name = list(well_metadata.keys())[0]
                field_zattrs_path = os.path.join(
                    store_path,
                    self._hcs_well_paths[first_well_name].rstrip("/"),
                    first_field_path.rstrip("/"),
                    ".zattrs",
                )
                if os.path.exists(field_zattrs_path):
                    try:
                        with open(field_zattrs_path) as f:
                            field_zattrs = json.load(f)
                            if "multiscales" in field_zattrs:
                                self.axes = field_zattrs["multiscales"][0].get(
                                    "axes", []
                                )
                                if "omero" in field_zattrs:
                                    channels = field_zattrs["omero"].get("channels", [])
                                    self.channel_names = [
                                        ch.get("label", f"ch{i}")
                                        for i, ch in enumerate(channels)
                                    ]
                    except (OSError, json.JSONDecodeError):
                        pass

    def _enumerate_hcs_fields(self) -> List[TensorDescriptor]:
        """Enumerate all fields in HCS plate as flattened tensor list.

        Returns TensorDescriptors with:
        - array_id = '{well_name}/{field_index}'
        - shape from multiscales metadata
        - chunk_shape from multiscales metadata
        - dtype from field array

        Returns:
            List of TensorDescriptor for all fields in the plate
        """
        import zarr

        descriptors = []

        # Use plate root path for navigation
        store_path = self._plate_root_path

        for well_name, well_path in self._hcs_well_paths.items():
            well_meta = self._hcs_well_metadata.get(well_name, {})
            well_info = well_meta.get("well", {})
            images = well_info.get("images", [])

            for field_idx, image_info in enumerate(images):
                field_path = image_info.get("path", str(field_idx))
                field_key = f"{well_name}/{field_idx}"

                # Try to read field .zattrs for shape/dtype
                field_zattrs_path = os.path.join(
                    store_path, well_path.rstrip("/"), field_path.rstrip("/"), ".zattrs"
                )

                shape = []
                chunk_shape = []
                dtype = ""
                dim_labels = self.dim_labels

                if os.path.exists(field_zattrs_path):
                    try:
                        with open(field_zattrs_path) as f:
                            field_zattrs = json.load(f)
                            multiscales = field_zattrs.get("multiscales", [])
                            if multiscales:
                                datasets = multiscales[0].get("datasets", [])
                                if datasets:
                                    # Get shape/chunks from first resolution level
                                    first_level_path = datasets[0].get("path", "0")

                                    # Open the array to get actual shape/chunks/dtype
                                    arr_path = os.path.join(
                                        store_path,
                                        well_path.rstrip("/"),
                                        field_path.rstrip("/"),
                                        first_level_path,
                                    )
                                    try:
                                        arr = zarr.open_array(arr_path, mode="r")
                                        shape = list(arr.shape)
                                        chunk_shape = list(arr.chunks)
                                        dtype = arr.dtype.str
                                    except Exception:
                                        # Fallback: estimate from multiscales metadata
                                        pass

                                # Get axes from multiscales
                                axes = multiscales[0].get("axes", [])
                                if axes:
                                    dim_labels = [
                                        ax.get("name", f"dim{i}")
                                        if isinstance(ax, dict)
                                        else str(ax)
                                        for i, ax in enumerate(axes)
                                    ]
                    except (OSError, json.JSONDecodeError):
                        pass

                descriptors.append(
                    TensorDescriptor(
                        # Globally-unique array_id = source_id/field (identity
                        # policy). The HCS field is itself hierarchical
                        # ("well_name/field_index"), so array_id is
                        # "source_id/well_name/field_index"; source_id is slash-free
                        # and recovered by splitting on the first '/'.
                        array_id=f"{self.source_id}/{field_key}",
                        dim_labels=dim_labels,
                        shape=shape,
                        chunk_shape=chunk_shape,
                        dtype=dtype,
                    )
                )

        return descriptors

    def get_ome_metadata(self) -> dict:
        """Return OME-Zarr metadata."""
        return self.ome_metadata

    def get_channel_info(self) -> List[dict]:
        """Return channel information from OME metadata."""
        if not self.channel_names:
            return [{"label": f"ch{i}"} for i in range(self.zarr_array.shape[0])]

        omero = self.ome_metadata.get("omero", {})
        channels = omero.get("channels", [])

        result = []
        for i, name in enumerate(self.channel_names):
            ch_info = {"label": name}
            if i < len(channels):
                ch_info.update(channels[i])
            result.append(ch_info)
        return result

    def get_metadata(self) -> dict:
        """Return OME-Zarr .zattrs content directly."""
        return self.ome_metadata

    def get_physical_scale(self, tensor_id=None):
        """Per-dim physical pixel size + unit from the multiscales transforms.

        Physical size of an axis = (optional multiscales-level scale transform)
        x (level-0 dataset scale), in that axis' ``unit``. Aligned to the axes
        order (which is also this adapter's ``dim_labels``). Only axes that
        declare a ``unit`` carry a physical size -- the OME-Zarr scale is
        otherwise a dimensionless (relative) downsample factor, so unit-less
        axes (and channel/index axes) get ``0.0`` / ``""``. Returns ``None``
        when no axis declares a unit (no physical information). See
        ``SourceAdapter.get_physical_scale``.
        """
        try:
            multiscales = self.ome_metadata.get("multiscales", [])
            if not multiscales:
                return None  # HCS plate or non-image: no top-level multiscales
            ms = multiscales[0]
            axes = ms.get("axes", []) or self.axes
            datasets = ms.get("datasets", [])
            if not axes or not datasets:
                return None
            ndim = len(axes)

            def _scale_vec(transforms):
                for t in transforms or []:
                    if t.get("type") == "scale":
                        vec = t.get("scale", [])
                        if len(vec) == ndim:
                            return [float(v) for v in vec]
                return [1.0] * ndim

            # Transforms compose: dataset (level-0) scale, then multiscales-level.
            global_scale = _scale_vec(ms.get("coordinateTransformations"))
            level0_scale = _scale_vec(datasets[0].get("coordinateTransformations"))

            scale, unit = [], []
            for i, ax in enumerate(axes):
                u = ax.get("unit", "") if isinstance(ax, dict) else ""
                if u:
                    try:
                        v = global_scale[i] * level0_scale[i]
                    except (TypeError, ValueError, IndexError):
                        v = 0.0
                    if v > 0:
                        scale.append(v)
                        unit.append(str(u))
                        continue
                scale.append(0.0)
                unit.append("")
            if not any(scale):
                return None
            return scale, unit
        except Exception:
            return None

    def has_native_pyramid(self) -> bool:
        """True for a single OME-Zarr image with a real (>=2 level) pyramid.

        Such sources already serve overviews efficiently from their native
        coarse levels, so the precache worker skips them. HCS plates report
        False (their fields are warmed individually, like any other source).
        """
        if self._is_hcs_plate:
            return False
        multiscales = self.ome_metadata.get("multiscales", [])
        if not multiscales:
            return False
        datasets = multiscales[0].get("datasets", [])
        return len(datasets) >= 2

    def get_native_pyramid_levels(
        self, tensor_id: Optional[str] = None
    ) -> Optional[List[PyramidLevel]]:
        """Advertise the OME-Zarr multiscales datasets as native pyramid levels.

        One ``PyramidLevel`` per native dataset, ``native=True`` and
        ``reduction_method="precompute"`` so the client requests the on-disk level
        directly (no on-the-fly downsampling). Each level's ``scale_hint`` is the
        dataset's NGFF ``scale`` transformation via :meth:`_get_level_scale` --
        the exact value :meth:`_find_level_for_scale` matches on -- so an
        advertised level round-trips through ``get_read_plan`` to its dataset.

        Returns ``None`` (-> the server advertises a *computed* pyramid) unless
        this is a single image with a real (>=2 level) native pyramid, mirroring
        :meth:`has_native_pyramid`. HCS plates and single-level images fall
        through to the computed path.

        Note: this reuses the existing integer convention -- NGFF ``scale`` values
        are taken as downsample factors relative to level 0 (level 0 == [1,1,...]).
        Files that instead store physical pixel sizes are not matched here, the
        same pre-existing limitation as ``_find_level_for_scale``.
        """
        if not self.has_native_pyramid():
            return None

        multiscales = self.ome_metadata.get("multiscales", [])
        datasets = multiscales[0].get("datasets", [])
        levels: List[PyramidLevel] = []
        for ds in datasets:
            path = ds.get("path")
            if path is None:
                continue
            scale = self._get_level_scale(path)
            if not scale:
                continue
            try:
                level_shape = list(
                    self.get_level_adapter(path).get_tensor_descriptor().shape
                )
            except Exception:
                logger.exception(
                    "ome-zarr: failed to size native level %s of %s",
                    path,
                    self.source_id,
                )
                continue
            levels.append(
                PyramidLevel(
                    scale_hint=list(scale),
                    reduction_method="precompute",
                    shape=level_shape,
                    native=True,
                )
            )
        return levels or None

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors available in this source.

        For HCS plates: Returns flattened list of field tensors.
        For single images: Returns the single tensor descriptor.

        Returns:
            List of TensorDescriptor for all tensors in this source.
        """
        if self._is_hcs_plate:
            return self._enumerate_hcs_fields()
        else:
            # Single multiscale image
            return [self.get_tensor_descriptor()]

    def get_tensor_adapter(self, tensor_id: str) -> "BackendAdapter":
        """Get adapter for a specific tensor.

        For HCS plates: Returns ZarrAdapter for the specific field.
        For single images: Returns self with tensor context set.

        Args:
            tensor_id: For HCS: 'well_name/field_index', for single image: optional

        Returns:
            BackendAdapter for the specific tensor with tensor context set
        """
        if not self._is_hcs_plate:
            # Single image: use base class behavior
            return super().get_tensor_adapter(tensor_id)

        # Accept either the within-source field ('well/field') or the full
        # source-qualified array_id 'source_id/well/field' (identity policy).
        tensor_id = self._within_source_field(tensor_id)

        # HCS plate: create field-level adapter
        # Parse tensor_id as 'well_name/field_index'
        parts = tensor_id.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"HCS tensor_id must be 'well_name/field_index', got: {tensor_id}"
            )

        well_name, field_idx_str = parts
        try:
            field_idx = int(field_idx_str)
        except ValueError:
            raise ValueError(
                f"HCS field_index must be an integer, got: {field_idx_str}"
            )

        # Check if well exists
        if well_name not in self._hcs_well_paths:
            raise ValueError(f"Unknown well: {well_name}")

        field_key = f"{well_name}/{field_idx}"

        # Cache field adapters for reuse
        if not hasattr(self, "_field_adapters"):
            self._field_adapters = {}

        if field_key in self._field_adapters:
            return self._field_adapters[field_key]

        # Create ZarrAdapter for the field's resolution array
        field_adapter = self._create_field_adapter(well_name, field_idx)

        # Set tensor context directly on the field adapter
        field_adapter._tensor_context = True
        field_adapter._tensor_name = f"{well_name}/{field_idx}"

        self._field_adapters[field_key] = field_adapter
        return field_adapter

    def _create_field_adapter(self, well_name: str, field_idx: int) -> ZarrAdapter:
        """Create ZarrAdapter for a specific field.

        Opens the field's resolution array (level 0 by default).

        Args:
            well_name: Well identifier (e.g., 'A01')
            field_idx: Field index within the well

        Returns:
            ZarrAdapter for the field array
        """
        import zarr

        well_path = self._hcs_well_paths[well_name]
        well_meta = self._hcs_well_metadata.get(well_name, {})
        well_info = well_meta.get("well", {})
        images = well_info.get("images", [])

        if field_idx >= len(images):
            raise ValueError(
                f"Field index {field_idx} out of range for well {well_name} (has {len(images)} fields)"
            )

        field_path = images[field_idx].get("path", str(field_idx))

        # Use plate root path for navigation
        store_path = self._plate_root_path

        # Read field .zattrs to get resolution path
        field_zattrs_path = os.path.join(
            store_path, well_path.rstrip("/"), field_path.rstrip("/"), ".zattrs"
        )

        resolution_path = "0"
        if os.path.exists(field_zattrs_path):
            try:
                with open(field_zattrs_path) as f:
                    field_zattrs = json.load(f)
                    multiscales = field_zattrs.get("multiscales", [])
                    if multiscales:
                        datasets = multiscales[0].get("datasets", [])
                        if datasets:
                            resolution_path = datasets[0].get("path", "0")
            except (OSError, json.JSONDecodeError):
                pass

        # Open the field's resolution array
        arr_path = os.path.join(
            store_path, well_path.rstrip("/"), field_path.rstrip("/"), resolution_path
        )

        arr = zarr.open_array(arr_path, mode="r")

        # Get dim_labels from field multiscales
        dim_labels = None
        if os.path.exists(field_zattrs_path):
            try:
                with open(field_zattrs_path) as f:
                    field_zattrs = json.load(f)
                    multiscales = field_zattrs.get("multiscales", [])
                    if multiscales:
                        axes = multiscales[0].get("axes", [])
                        if axes:
                            dim_labels = [
                                ax.get("name", f"dim{i}")
                                if isinstance(ax, dict)
                                else str(ax)
                                for i, ax in enumerate(axes)
                            ]
            except (OSError, json.JSONDecodeError):
                pass

        # Create adapter for this field
        field_adapter = ZarrAdapter(
            arr,
            source_id=self.source_id,
            dim_labels=dim_labels or self.dim_labels,
        )

        # Set tensor name for array_id computation
        field_adapter._tensor_name = f"{well_name}/{field_idx}"

        return field_adapter

    def get_read_plan(self, request_desc: TensorDescriptor) -> TensorReadPlan:
        """Return read plan for requested scale.

        Supports "precompute" method to use precomputed pyramid levels.
        Falls back to virtual scaling for other methods.
        """
        # Extract parameters from request_desc (scale_hint/reduction_method are now direct fields)
        slice_hint = (
            request_desc.slice_hint if request_desc.HasField("slice_hint") else None
        )

        # Compute scale_hint directly from TensorDescriptor
        base_desc = self.get_tensor_descriptor()
        base_shape = tuple(int(dim) for dim in base_desc.shape)
        from biopb_tensor_server.chunk import normalized_scale_hint

        scale_hint = normalized_scale_hint(base_shape, request_desc.scale_hint)

        reduction_method = normalize_reduction_method(request_desc.reduction_method)

        # "precompute" method: use precomputed level if exact match
        if reduction_method == "precompute" and scale_hint is not None:
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
        return super().get_read_plan(request_desc)

    def _find_level_for_scale(self, scale_hint: Tuple[int, ...]) -> Optional[str]:
        """Find precomputed level with exact scale match."""
        multiscales = self.ome_metadata.get("multiscales", [])
        if not multiscales:
            return None

        for ds in multiscales[0].get("datasets", []):
            for t in ds.get("coordinateTransformations", []):
                if t.get("type") == "scale":
                    scale = tuple(int(s) for s in t.get("scale", []))
                    if scale == scale_hint:
                        return ds.get("path")

        return None

    def _get_level_scale(self, level_path: str) -> Tuple[int, ...]:
        """Extract scale for a specific level path."""
        multiscales = self.ome_metadata.get("multiscales", [])
        if not multiscales:
            return tuple()

        for ds in multiscales[0].get("datasets", []):
            if ds.get("path") == level_path:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        return tuple(int(s) for s in t.get("scale", []))

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
        """Create read plan from precomputed level.

        Delegates to base class get_read_plan() with no scale_hint.
        """
        level_adapter = self.get_level_adapter(level_path)

        # Create request descriptor with slice_hint but NO scale_hint (no downsampling)
        level_desc = level_adapter.get_tensor_descriptor()
        request_desc = TensorDescriptor(
            array_id=self.array_id,  # Use original array_id, not level's
            dim_labels=level_desc.dim_labels,
            shape=list(level_desc.shape),
            chunk_shape=list(level_desc.chunk_shape),
            dtype=level_desc.dtype,
        )

        # Set slice_hint if provided
        if level_slice is not None:
            request_desc.slice_hint.start[:] = level_slice.start
            request_desc.slice_hint.stop[:] = level_slice.stop

        # Delegate to base class get_read_plan (no scale_hint means no downsampling)
        read_plan = level_adapter.get_read_plan(request_desc)

        # Override array_id in the returned descriptor to match original request
        read_plan.descriptor.array_id = self.array_id

        return read_plan

    def get_level_adapter(self, path: str) -> ZarrAdapter:
        """Get adapter for a specific precomputed level.

        Args:
            path: Level path (e.g., "0", "1", "2" for OME-Zarr)

        Returns:
            ZarrAdapter for the level array with tensor context set
        """
        if path in self._level_adapters:
            return self._level_adapters[path]

        # Open the level array
        level_arr = self._open_level_array(path)

        # Create adapter for this level
        level_adapter = ZarrAdapter(
            level_arr,
            source_id=self.source_id,
            dim_labels=self.dim_labels,
        )
        # Set tensor name for multi-tensor context
        level_adapter._tensor_name = path
        level_adapter._tensor_context = True

        self._level_adapters[path] = level_adapter
        return level_adapter

    def _open_level_array(self, path: str):
        """Open the Zarr array at the given level path (relative to group root)."""
        import zarr

        store = self.zarr_array.store

        store_str = str(store.path if hasattr(store, "path") else store)

        if store_str.startswith("file://"):
            store_path = urlparse(store_str).path
        else:
            store_path = store_str

        # Navigate to the group root
        current_path = store_path.rstrip("/")
        while current_path and current_path != "/":
            if os.path.exists(os.path.join(current_path, ".zattrs")):
                break
            current_path = os.path.dirname(current_path)

        # Join with the target level path
        level_path = os.path.join(current_path, path)

        return zarr.open_array(level_path, mode="r")
