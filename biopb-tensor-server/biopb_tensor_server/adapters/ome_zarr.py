"""OME-Zarr adapter for tensor storage.

Extends ZarrAdapter with OME multiscales metadata support.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from urllib.parse import urlparse

from biopb.tensor.descriptor_pb2 import PyramidLevel, TensorDescriptor

from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import InvalidTensorId, TensorNotFound

if TYPE_CHECKING:
    from biopb_tensor_server.core.adapter_base import TensorAdapter
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState


logger = logging.getLogger(__name__)


def _store_filesystem_path(store) -> str:
    """Resolve a zarr store to the local filesystem path it is rooted at.

    One definition, two callers (biopb/biopb#530): ``__init__`` walks up from here
    to find the group/plate ``.zattrs``, and ``_open_level_array`` walks up from
    here to find the group root a pyramid level hangs off. They used to enumerate
    different store shapes, so a store carrying ``root`` but not ``path`` resolved
    correctly in one and degraded to ``str(store)`` -- a repr, not a path -- in the
    other, which silently turns a level read into a CWD-relative open.

    ``path`` is the zarr-2 attribute (``DirectoryStore`` / ``FSStore``); ``root`` is
    the zarr-3 ``LocalStore`` one, unreachable under the current ``zarr<3`` pin and
    kept so the 2->3 port cannot reintroduce the split.
    """
    store_str = str(store)
    if store_str.startswith("file://"):
        return str(urlparse(store_str).path)
    path = getattr(store, "path", None)
    if path is not None:
        return str(path)
    root = getattr(store, "root", None)
    if root is not None:
        return str(root)
    return store_str


def _physical_scale_from_multiscales(
    multiscales: list, fallback_axes: list
) -> Optional[Tuple[List[float], List[str]]]:
    """Per-dim physical pixel size + unit from an OME-Zarr ``multiscales`` block.

    Physical size of an axis = (optional multiscales-level scale transform) x
    (level-0 dataset scale), in that axis' ``unit``, aligned to the axes order.
    Only axes that declare a ``unit`` carry a physical size -- an NGFF scale is
    otherwise a dimensionless (relative) downsample factor, so unit-less axes
    (channel / index) get ``0.0`` / ``""``. Returns ``None`` when there is no
    ``multiscales``, no axes/datasets, or no unit-bearing axis. Shared by the
    single-image :meth:`OmeZarrAdapter._physical_scale` and the per-field HCS
    adapter (:class:`_HcsFieldAdapter`), which read the identical block off their
    own ``.zattrs``.

    ``fallback_axes`` supplies the source-level axes to use when *this* block
    omits its own ``axes`` array (permitted by older NGFF while it still carries
    dataset scale transforms). Callers pass the adapter's ``self.axes`` so a
    field/level missing its axes still resolves units from the source; both
    lists describe the same dimensionality, so the fallback is safe.
    """
    if not multiscales:
        return None
    ms = multiscales[0]
    axes = ms.get("axes", []) or fallback_axes
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


def _axes_to_dim_labels(axes: list) -> Optional[List[str]]:
    """Dimension labels from an OME ``axes`` array, or ``None`` when it is empty.

    An axis is either a dict (``{"name": "y", ...}``) or a bare string; both the
    single-image ``__init__`` and every per-field reader turn axes into labels
    the identical way, so the conversion lives here (issue #558)."""
    if not axes:
        return None
    return [
        ax.get("name", f"dim{i}") if isinstance(ax, dict) else str(ax)
        for i, ax in enumerate(axes)
    ]


def _first_dataset_path(multiscales: list) -> Optional[str]:
    """Path of the first (full-resolution) dataset in a ``multiscales`` block.

    Returns ``datasets[0].path`` (defaulting to ``"0"`` when that entry omits its
    own path), or ``None`` when the block carries no ``multiscales``/``datasets``.
    The one place the "which array is level 0" convention is spelled — the create
    path, the field enumeration, and the field-adapter open all route through it
    (issue #558)."""
    if not multiscales:
        return None
    datasets = multiscales[0].get("datasets", [])
    if not datasets:
        return None
    return datasets[0].get("path", "0")


class _HcsFieldAdapter(ZarrAdapter):
    """A single HCS-plate field, served as a plain Zarr array + its own scale.

    :meth:`OmeZarrAdapter._create_field_adapter` opens a field's level-0 array
    directly (no group navigation), so a plain :class:`ZarrAdapter` handles the
    pixels. This thin subclass additionally carries the ``(scale, unit)`` read
    from the field's ``.zattrs`` multiscales, so a per-field ``GetFlightInfo``
    advertises the field's physical calibration (biopb/biopb#272) -- the plate
    adapter itself returns ``None`` (there is no plate-level scale), and the
    field is where the calibration actually lives.
    """

    def __init__(self, *args, physical_scale=None, field_metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._field_physical_scale = physical_scale
        self._field_metadata = field_metadata

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        return self._field_physical_scale

    def get_tensor_metadata(self) -> Optional[dict]:
        """This field's own ``.zattrs`` (its OME-NGFF metadata) as the delta.

        The plate's source-level catalog row is the *plate* ``.zattrs`` (the
        rows/columns/wells layout); the serve path merges this field's own
        ``.zattrs`` (``multiscales`` / ``omero`` -- disjoint from the plate keys)
        over it, so a field advertises both its plate context and its own metadata
        (biopb/biopb#253). ``None`` when the field ``.zattrs`` is unreadable -- the
        field then carries just the plate row.
        """
        return self._field_metadata


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

    # HCS-specific state (populated if _is_hcs_plate is True)
    _is_hcs_plate: bool = False
    _plate_root_path: Optional[str] = None  # Path to plate root directory
    _hcs_field_count: int = 0
    # The dict-valued state below is annotation-only on purpose -- see the
    # per-instance assignments in __init__.
    _hcs_well_paths: dict  # well_name -> zarr path (e.g., 'A01' -> 'A01')
    _hcs_well_metadata: dict  # well_name -> well .zattrs
    _field_adapters: dict  # field_key -> cached adapter

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
        The plate/well/field ``.zattrs`` navigation is written once against the
        unified :class:`ClaimContext` fs seam (local ``Path`` or remote
        ``RemoteStore``) rather than a hand-forked fsspec/``DirectoryStore`` pair
        (issue #558). The already-parsed top-level ``.zattrs`` and the local root
        are threaded into ``__init__`` so it does not re-walk the store back up to
        the root it just descended from.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            OmeZarrAdapter instance
        """
        import zarr

        zarr_path = str(source.url)
        ctx, zarr_store = cls._source_context_and_store(source, credentials_config)

        # A missing/corrupt sidecar already decodes to None in _read_json_sidecar,
        # so zattrs is always a dict here. The zarr opens raise typed
        # Group/ArrayNotFound errors we let propagate (fail fast on a source whose
        # .zattrs the claim vetted but whose arrays are gone).
        zattrs = cls._read_json_sidecar(ctx, ".zattrs") or {}
        if "plate" in zattrs:
            field_relpath = cls._first_field_relpath(ctx, zattrs)
            arr = cls._open_field_array(source, zarr_store, zarr_path, field_relpath)
        else:
            res = _first_dataset_path(zattrs.get("multiscales", [])) or "0"
            root = zarr.open_group(zarr_store, mode="r")
            arr = root[res] if res in root else zarr.open_array(zarr_store, mode="r")

        # Thread the parsed root .zattrs + local root so __init__ skips the store
        # re-walk. Remote passes no local root -> __init__ falls back to its walk,
        # preserving the prior remote behavior (its store path is not local).
        threaded_root = None if source.is_remote else zarr_path
        return cls(
            arr,
            source.source_id,
            source.dim_labels,
            _threaded_zattrs=zattrs or None,
            _threaded_root=threaded_root,
        )

    @staticmethod
    def _source_context_and_store(
        source: "SourceConfig", credentials_config: Optional[Any]
    ) -> Tuple[ClaimContext, Any]:
        """A ``(ClaimContext, zarr store)`` pair rooted at the source.

        One seam for both storage kinds (was a hand-forked remote/local pair):
        the :class:`ClaimContext` reads the ``.zattrs`` sidecars during the plate
        descent, the store opens the resolution arrays. Remote goes through
        :class:`RemoteStore` — the same fs abstraction discovery and the base
        ``ZarrAdapter`` already use — so credential handling stays in one place.
        """
        import zarr

        if source.is_remote:
            from biopb_tensor_server.core.remote import RemoteStore

            store = RemoteStore.from_config(
                source.url,
                credentials_config=credentials_config,
                profile_name=source.credentials_profile,
            )
            zarr_store = zarr.FSStore(store.path, fs=store.fs)
            return ClaimContext("", store), zarr_store

        zarr_path = str(source.url)
        return ClaimContext(Path(zarr_path)), zarr.DirectoryStore(zarr_path)

    @staticmethod
    def _read_json_sidecar(ctx: ClaimContext, name: str) -> Optional[dict]:
        """Parse the JSON sidecar ``name`` under ``ctx``, or ``None`` if it is
        absent or unreadable. The single fs-seam read behind the create-time
        plate navigation, so local and remote decode identically."""
        try:
            if not ctx.join(name).exists():
                return None
            return json.loads(ctx.read_text(name))
        except (OSError, json.JSONDecodeError):
            return None

    @classmethod
    def _first_field_relpath(
        cls, ctx: ClaimContext, plate_zattrs: dict
    ) -> Optional[str]:
        """``well/field/level`` path of the plate's first field's level-0 array.

        Navigates plate -> first well -> first field -> first dataset through the
        fs seam (the descent that ``create_from_config`` used to carry twice).
        Returns ``None`` when the plate has no wells, the well no images, or the
        field no datasets — the caller then opens the store root, exactly as the
        prior per-branch fallbacks did.
        """
        wells = plate_zattrs.get("plate", {}).get("wells", [])
        if not wells:
            return None
        well_path = wells[0].get("path", "0").rstrip("/")

        well_zattrs = cls._read_json_sidecar(ctx.join(well_path), ".zattrs")
        images = (well_zattrs or {}).get("well", {}).get("images", [])
        if not images:
            return None
        field_path = images[0].get("path", "0").rstrip("/")

        field_ctx = ctx.join(well_path).join(field_path)
        field_zattrs = cls._read_json_sidecar(field_ctx, ".zattrs")
        res = _first_dataset_path((field_zattrs or {}).get("multiscales", []))
        if res is None:
            return None
        return f"{well_path}/{field_path}/{res}"

    @staticmethod
    def _open_field_array(
        source: "SourceConfig", zarr_store, zarr_path: str, relpath: Optional[str]
    ):
        """Open a plate field's resolution array (or the store root fallback).

        Preserves the store rooting each mode had: remote opens through the shared
        ``FSStore`` + ``path=``; local opens the absolute directory so
        ``arr.store`` (hence the catalog ``source_url``) roots where it did before.
        """
        import zarr

        if relpath is None:
            return zarr.open_array(zarr_store, mode="r")
        if source.is_remote:
            return zarr.open_array(zarr_store, path=relpath, mode="r")
        return zarr.open_array(os.path.join(zarr_path, relpath), mode="r")

    def __init__(
        self,
        zarr_array,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        resolution_level: int = 0,
        _threaded_zattrs: Optional[dict] = None,
        _threaded_root: Optional[str] = None,
    ):
        """Initialize OME-Zarr adapter.

        Args:
            zarr_array: Zarr array object (from specific resolution level)
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (overrides OME metadata)
            resolution_level: Which resolution level to use (default 0)
            _threaded_zattrs / _threaded_root: the root ``.zattrs`` and its local
                path, supplied by :meth:`create_from_config` (which already read
                them off the fs seam) so this constructor skips walking the store
                back up to the root. Both must be present to take that shortcut;
                a direct construction leaves them ``None`` and the walk runs
                (issue #558).
        """
        # Initialize base ZarrAdapter first
        # We'll override dim_labels below if OME metadata provides better ones
        super().__init__(zarr_array, source_id, dim_labels)

        # Per-instance caches/state. These MUST be assigned here: a class-level
        # mutable default is shared by every adapter in the process, so two HCS
        # plates with the same well name would serve each other's pixels (#522).
        self._hcs_well_paths: dict = {}
        self._hcs_well_metadata: dict = {}
        self._field_adapters: dict = {}
        # Cache for level adapters (precomputed pyramid levels)
        self._level_adapters: dict = {}

        self.resolution_level = resolution_level

        # Try to read OME metadata from .zattrs
        self.ome_metadata = {}
        self.axes = []
        self.channel_names = []

        # The group/plate root and its .zattrs. create_from_config threads both
        # (it already read them descending the store), so we take them as-is;
        # otherwise walk the store path up to the root.
        if _threaded_zattrs is not None and _threaded_root is not None:
            plate_root_path, zattrs = _threaded_root, _threaded_zattrs
        else:
            plate_root_path, zattrs = self._find_group_root(
                _store_filesystem_path(zarr_array.store)
            )

        if zattrs is not None:
            self.ome_metadata = zattrs

            # Check for HCS plate metadata first
            if "plate" in zattrs:
                self._is_hcs_plate = True
                self._source_type = "ome-zarr-hcs"
                self._plate_root_path = plate_root_path  # Save for field adapters
                self._parse_hcs_plate_structure()
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
            self.dim_labels = _axes_to_dim_labels(self.axes)

    @staticmethod
    def _find_group_root(store_path: str) -> Tuple[Optional[str], Optional[dict]]:
        """Walk up from ``store_path`` to the OME-Zarr group/plate root.

        Returns ``(root_path, zattrs)`` for the nearest ancestor whose ``.zattrs``
        carries ``plate`` (HCS, highest priority) or ``multiscales`` (single
        image); ``(None, None)`` when neither is found. A ``plate`` ancestor wins
        even above a ``multiscales`` one, so a field opened deep inside a plate
        still resolves the plate. Only reached for a direct construction —
        ``create_from_config`` threads the root it already descended from.
        """
        plate_root_path = None
        zattrs = None
        current_path = store_path.rstrip("/")
        while current_path:
            candidate_zattrs_path = os.path.join(current_path, ".zattrs")
            if os.path.exists(candidate_zattrs_path):
                try:
                    with open(candidate_zattrs_path) as f:
                        candidate_zattrs = json.load(f)

                    # HCS plate metadata is highest priority: stop the moment it
                    # is found.
                    if "plate" in candidate_zattrs:
                        return current_path, candidate_zattrs

                    # A multiscales-without-plate is the single-image fallback,
                    # but keep climbing in case a plate sits above it.
                    if "multiscales" in candidate_zattrs and zattrs is None:
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

        return plate_root_path, zattrs

    def _read_zattrs_at(self, *parts: str) -> Optional[dict]:
        """Parse the ``.zattrs`` at ``plate_root/<parts>/.zattrs``, or ``None``.

        The single local reader for a plate's interior sidecars — every per-well
        and per-field ``.zattrs`` read (the parse, the enumeration, the
        field-adapter open) goes through here so "where a field's
        resolution/axes/scale come from" lives in one place (issue #558). HCS
        field serving is local-only, so this reads the local filesystem directly.
        """
        path = os.path.join(
            self._plate_root_path, *[p.rstrip("/") for p in parts], ".zattrs"
        )
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _field_array_path(self, well_path: str, field_path: str, level: str) -> str:
        """Absolute path to a plate field's resolution-``level`` array."""
        return os.path.join(
            self._plate_root_path, well_path.rstrip("/"), field_path.rstrip("/"), level
        )

    def _parse_hcs_plate_structure(self) -> None:
        """Parse HCS plate metadata from plate.zattrs and well .zattrs files.

        Populates:
        - _hcs_well_paths: dict mapping well_name -> well path
        - _hcs_field_count: number of fields per well
        - _hcs_well_metadata: dict mapping well_name -> well .zattrs

        Reads the interior sidecars relative to ``self._plate_root_path`` (set by
        ``__init__`` just before this runs) via :meth:`_read_zattrs_at`.
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
            well_zattrs = self._read_zattrs_at(well_path)
            if well_zattrs is None:
                continue
            well_metadata[well_name] = well_zattrs
            # well metadata has a 'well' key with an 'images' list
            images = well_zattrs.get("well", {}).get("images", [])
            field_count = max(field_count, len(images))

        self._hcs_field_count = field_count
        self._hcs_well_metadata = well_metadata

        # Axes / channel names come from the first field's multiscales metadata.
        if well_metadata:
            first_well_name = next(iter(well_metadata))
            images = well_metadata[first_well_name].get("well", {}).get("images", [])
            if images:
                first_field_path = images[0].get("path", "0")
                field_zattrs = self._read_zattrs_at(
                    self._hcs_well_paths[first_well_name], first_field_path
                )
                multiscales = (field_zattrs or {}).get("multiscales", [])
                if multiscales:
                    self.axes = multiscales[0].get("axes", [])
                    channels = (field_zattrs or {}).get("omero", {}).get("channels", [])
                    self.channel_names = [
                        ch.get("label", f"ch{i}") for i, ch in enumerate(channels)
                    ]

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

        for well_name, well_path in self._hcs_well_paths.items():
            well_meta = self._hcs_well_metadata.get(well_name, {})
            well_info = well_meta.get("well", {})
            images = well_info.get("images", [])

            for field_idx, image_info in enumerate(images):
                field_path = image_info.get("path", str(field_idx))
                field_key = f"{well_name}/{field_idx}"

                shape = []
                chunk_shape = []
                dtype = ""
                dim_labels = self.dim_labels

                field_zattrs = self._read_zattrs_at(well_path, field_path)
                multiscales = (field_zattrs or {}).get("multiscales", [])
                res = _first_dataset_path(multiscales)
                if res is not None:
                    # Open the level-0 array for actual shape/chunks/dtype.
                    try:
                        arr = zarr.open_array(
                            self._field_array_path(well_path, field_path, res),
                            mode="r",
                        )
                        shape = list(arr.shape)
                        chunk_shape = list(arr.chunks)
                        dtype = arr.dtype.str
                    except Exception:
                        # Fallback: leave shape/dtype unfilled (metadata-only).
                        pass
                if multiscales:
                    dim_labels = (
                        _axes_to_dim_labels(multiscales[0].get("axes", []))
                        or self.dim_labels
                    )

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
        """Return OME-Zarr .zattrs content directly.

        For an HCS plate this is the *plate* ``.zattrs`` (rows/columns/wells
        layout), which is the source-level catalog row -- a field's own OME
        metadata is served per-tensor via
        :meth:`_HcsFieldAdapter.get_tensor_metadata` (biopb/biopb#253).
        """
        return self.ome_metadata

    def _physical_scale(self):
        """Per-dim physical pixel size + unit from the multiscales transforms.

        Reads this image's top-level ``multiscales`` (aligned to ``dim_labels``);
        an HCS plate has none at the source level and so returns ``None`` here --
        its per-field scale is carried by :class:`_HcsFieldAdapter`. See
        :func:`_physical_scale_from_multiscales` and ``TensorAdapter._physical_scale``.
        """
        try:
            return _physical_scale_from_multiscales(
                self.ome_metadata.get("multiscales", []), self.axes
            )
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

    def get_native_pyramid_levels(self) -> Optional[List[PyramidLevel]]:
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

    def get_tensor_adapter(self, tensor_id: str) -> "TensorAdapter":
        """Get adapter for a specific tensor.

        For HCS plates: Returns ZarrAdapter for the specific field.
        For single images: Returns self with tensor context set.

        Args:
            tensor_id: For HCS: 'well_name/field_index', for single image: optional

        Returns:
            TensorAdapter for the specific tensor with tensor context set
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
            raise InvalidTensorId(
                f"HCS tensor_id must be 'well_name/field_index', got: {tensor_id}",
                reason="malformed_tensor_id",
            )

        well_name, field_idx_str = parts
        try:
            field_idx = int(field_idx_str)
        except ValueError as e:
            raise InvalidTensorId(
                f"HCS field_index must be an integer, got: {field_idx_str}",
                reason="malformed_tensor_id",
            ) from e

        # Check if well exists
        if well_name not in self._hcs_well_paths:
            raise TensorNotFound(f"Unknown well: {well_name}", reason="unknown_field")

        field_key = f"{well_name}/{field_idx}"

        if field_key in self._field_adapters:
            return self._field_adapters[field_key]

        # Create ZarrAdapter for the field's resolution array
        field_adapter = self._create_field_adapter(well_name, field_idx)

        # Set tensor name on the field adapter
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
            raise TensorNotFound(
                f"Field index {field_idx} out of range for well {well_name} (has {len(images)} fields)",
                reason="unknown_field",
            )

        field_path = images[field_idx].get("path", str(field_idx))

        # Read the field's .zattrs once: the resolution path, dim labels, and
        # the physical scale (biopb/biopb#272) all come from its multiscales
        # block. A corrupt/absent .zattrs degrades to defaults (no scale).
        resolution_path = "0"
        dim_labels = None
        physical_scale = None
        # The field's own .zattrs -- its OME-NGFF metadata, served per-field via
        # _HcsFieldAdapter.get_tensor_metadata (the plate row cannot carry it, #253).
        field_metadata = self._read_zattrs_at(well_path, field_path)
        try:
            multiscales = (field_metadata or {}).get("multiscales", [])
            if multiscales:
                resolution_path = _first_dataset_path(multiscales) or "0"
                dim_labels = _axes_to_dim_labels(multiscales[0].get("axes", []))
                # Fall back to the plate source's axes (populated from the first
                # field, see _parse_hcs_plate_structure) when THIS field's
                # multiscales omits its own -- fields in a plate share axis
                # structure, so the source axes carry the right units. Passing the
                # field's own (possibly empty) axes here would make it a no-op.
                physical_scale = _physical_scale_from_multiscales(
                    multiscales, self.axes
                )
        except Exception:
            logger.debug(
                "ome-zarr HCS: field .zattrs unreadable for %s/%s",
                well_name,
                field_path,
                exc_info=True,
            )

        # Open the field's resolution array
        arr = zarr.open_array(
            self._field_array_path(well_path, field_path, resolution_path), mode="r"
        )

        # Create adapter for this field, carrying the field's own physical scale
        # (the plate level has none -- biopb/biopb#272).
        field_adapter = _HcsFieldAdapter(
            arr,
            source_id=self.source_id,
            dim_labels=dim_labels or self.dim_labels,
            physical_scale=physical_scale,
            field_metadata=field_metadata,
        )

        # Set tensor name for array_id computation
        field_adapter._tensor_name = f"{well_name}/{field_idx}"

        return field_adapter

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
            return ()

        for ds in multiscales[0].get("datasets", []):
            if ds.get("path") == level_path:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        return tuple(int(s) for s in t.get("scale", []))

        return ()

    def _level_downsample_factors(self, level: str) -> List[int]:
        """Downsample factors for level ``level`` -- its NGFF ``scale`` transform.

        The base precompute routing (biopb/biopb#557) uses these to translate a
        base-coordinate slice into the level's grid.
        """
        return list(self._get_level_scale(level))

    def get_level_adapter(self, path: str) -> Optional[ZarrAdapter]:
        """Get adapter for a specific precomputed level (single-image only).

        Returns ``None`` for an HCS plate: a plate has no native pyramid, so a
        within-source suffix on one of its chunks is a ``well/field`` tensor id,
        not a level path. Returning ``None`` routes the server's chunk dispatch
        back to :meth:`get_tensor_adapter` (the field adapter) instead of trying
        to open a non-existent level store (biopb/biopb#557).

        Args:
            path: Level path (e.g., "0", "1", "2" for OME-Zarr)

        Returns:
            ZarrAdapter for the level array with tensor context set, or ``None``
            for an HCS plate.
        """
        if self._is_hcs_plate:
            return None

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

        self._level_adapters[path] = level_adapter
        return level_adapter

    def _open_level_array(self, path: str):
        """Open the Zarr array at the given level path (relative to group root)."""
        import zarr

        store_path = _store_filesystem_path(self.zarr_array.store)

        # Navigate to the group root. Terminate on the dirname fixed point rather
        # than on '/', so a Windows drive root ends the walk instead of spinning
        # forever -- the same termination bug already fixed in __init__.
        current_path = store_path.rstrip("/")
        group_root = None
        while current_path:
            if os.path.exists(os.path.join(current_path, ".zattrs")):
                group_root = current_path
                break
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break
            current_path = parent_path

        if group_root is None:
            # Exhausting the walk used to leave current_path == "", so
            # os.path.join("", path) handed zarr a *relative* path resolved
            # against the process CWD: a level read that fails obscurely, or
            # worse succeeds against an unrelated store (biopb/biopb#530).
            raise FileNotFoundError(
                f"no OME-Zarr group root (a directory holding .zattrs) at or above "
                f"{store_path!r}; cannot open pyramid level {path!r} of source "
                f"{self.source_id!r}"
            )

        return zarr.open_array(os.path.join(group_root, path), mode="r")
