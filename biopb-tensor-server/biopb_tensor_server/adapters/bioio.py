"""bioio adapters for vendor microscopy formats.

This module provides a base class and format-specific subclasses for reading
various microscopy formats through bioio's BioImage class. bioio is the
maintained successor to aicsimageio (its API is a near drop-in); each vendor
reader ships as its own ``bioio-*`` plugin, so a slimmer install pulls only the
formats it needs. See docs/aicsimageio-to-bioio-migration.md.

Format-specific subclasses provide meaningful source_type values:
- ZeissAdapter: "zeiss" (CZI, LSM)
- LeicaAdapter: "leica" (LIF)
- NikonAdapter: "nikon" (ND2)
- DvAdapter: "dv" (DeltaVision)
- OlympusAdapter: "olympus" (OIF, OIB)
- AicsImageIoAdapter: "aics" (fallback for other formats)

Supports:
- Multi-scene files (each scene becomes a separate tensor)
- Lazy loading via dask arrays
- OME-XML metadata conversion
- Remote storage (S3, GCS, etc.) via fsspec (passing fs_kwargs)

Chunk ID format:
- array_id + bounds encoding (start, stop coordinates)
"""

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.discovery import ClaimContext, SourceClaim

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bioio import BioImage

    from biopb_tensor_server.base import BackendAdapter
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState


# Canonical OME dimension order. The scene-listing path uses it to
# detect a plain TCZYX source (which it can shape from OME Pixels) versus an
# RGB/samples one it must defer to scene switching.
_CANONICAL_DIMS = "TCZYX"


GENERIC_IMAGE_EXTENSIONS = frozenset(
    [
        # Standard raster formats
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        # Video formats
        ".avi",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
    ]
)

# Microscopy / scientific image formats not handled by a format-specific subclass
# (those handle .czi, .lsm, .lif, .nd2, .dv, .oif, .oib, .companion.ome). Always
# eligible for a discovery claim. .tif/.tiff stay here: plain TIFFs are
# legitimate microscopy sources, and OME-TIFFs are claimed earlier by
# OmeTiffAdapter.
MICROSCOPY_EXTENSIONS = frozenset(
    [
        # TIFF (plain; OME-TIFF handled by OmeTiffAdapter)
        ".tif",
        ".tiff",
        # Microscopy-specific formats (not handled by specific adapters)
        ".mrc",
        ".mrcs",  # MRC electron microscopy
        ".klb",  # Keller Lab Blockfile
        ".ims",  # Imaris
        ".liff",
        ".lim",  # Other Leica variants
        ".cif",
        ".cxd",  # Cell imaging formats
        ".flex",
        ".fli",  # Flexible image transport
        # Scientific image formats
        ".fits",
        ".fit",
        ".fts",  # FITS astronomical/scientific
        ".nrrd",
        ".nhdr",  # NRRD medical imaging
        ".mhd",
        ".mha",
        ".img",
        ".hdr",  # Analyze/MetaImage format
        ".ics",
        ".ids",  # ICS/IDS format
    ]
)

# Full curated set (microscopy + generic). Retained for back-compat with callers
# that reference the union; the actual claim scope is decided per-call by
# :func:`_claim_extensions`, which honors the generic-images opt-in.
CORE_IMAGE_EXTENSIONS = MICROSCOPY_EXTENSIONS | GENERIC_IMAGE_EXTENSIONS
AICS_EXTENSIONS = CORE_IMAGE_EXTENSIONS


# Whether recursive directory discovery may claim GENERIC_IMAGE_EXTENSIONS. Off
# by default (biopb/biopb#40); set from ``ServerConfig.claim_generic_images`` at
# server startup (see cli.serve). The ``BIOPB_CLAIM_GENERIC_IMAGES`` env var
# seeds the initial default so the toggle also applies to discovery paths that
# never load a ServerConfig (e.g. ad-hoc tooling).
_CLAIM_GENERIC_IMAGES: bool = os.environ.get(
    "BIOPB_CLAIM_GENERIC_IMAGES", ""
).strip().lower() in ("1", "true", "yes", "on")


def set_claim_generic_images(enabled: bool) -> None:
    """Enable/disable claiming generic raster/video during directory discovery.

    Process-wide policy (one ServerConfig per process), mirroring the other
    module-level startup toggles. Off by default: generic raster/video pollute
    the catalog during recursive walks (biopb/biopb#40). Does not affect an
    explicitly configured ``type = "aics"`` source, which never consults claim().
    """
    global _CLAIM_GENERIC_IMAGES
    _CLAIM_GENERIC_IMAGES = bool(enabled)


def _claim_extensions() -> frozenset:
    """Extensions eligible for a discovery claim, honoring the generic-images flag."""
    if _CLAIM_GENERIC_IMAGES:
        return CORE_IMAGE_EXTENSIONS
    return MICROSCOPY_EXTENSIONS


class _BioioAdapterBase(SourceAdapter, TensorAdapter):
    """Base adapter for bioio-supported vendor formats.

    This base class provides full functionality for reading microscopy data
    through bioio's BioImage class. Subclasses implement claim() with
    format-specific detection and provide meaningful source_type values.

    Dual-role adapter:
    - Source-level (scene_index=None): manages metadata, lists all scenes
    - Scene-level (scene_index=int): handles data access for one scene

    Multi-scene files expose each scene as a separate tensor within the source.
    Each scene is identified by its scene_id from img.scenes.

    Supports lazy loading via dask arrays.
    Supports remote storage via fsspec (passes fs_kwargs to BioImage).
    """

    # Class-level source type (override in subclasses)
    SOURCE_TYPE: str = "aics"

    # Multi-tensor source: has multiple scenes
    _single_tensor_source = False

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "_BioioAdapterBase":
        """Create source-level adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            Adapter instance (source-level, scene_index=None)
        """
        from bioio import BioImage

        if source.is_remote:
            # Remote storage: resolve storage_options for fsspec authentication
            storage_options = {}
            if credentials_config:
                profile = credentials_config.get_profile(source.credentials_profile)
                if profile:
                    storage_options = profile.to_storage_options()

            # Note: for OME-Zarr, use OmeZarrAdapter (which threads fs_kwargs
            # through to zarr) rather than bioio's OME-Zarr reader.
            img = BioImage(source.url, fs_kwargs=storage_options)
        else:
            # Local filesystem
            img = BioImage(str(source.url))

        return cls(
            img,
            scene_index=None,  # Source-level adapter
            source_id=source.source_id,
            dim_labels=source.dim_labels,
            source_url=str(source.url),
        )

    def __init__(
        self,
        bio_image: "BioImage",
        scene_index: Optional[int],
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        io_lock: Optional[threading.Lock] = None,
    ):
        """Initialize bioio adapter.

        Args:
            bio_image: BioImage instance
            scene_index: None for source-level, int for scene-level
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (overrides auto-detected dims)
            source_url: Optional source URL
            io_lock: Optional thread lock for IO serialization. Source-level
                     adapters create a new lock if None; scene-level adapters
                     receive the lock from the source-level adapter.
        """
        self._bio_image = bio_image
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
        elif hasattr(bio_image, "source") and hasattr(bio_image.source, "path"):
            self._source_url = str(bio_image.source.path)
        else:
            self._source_url = ""
        self._source_type = self.SOURCE_TYPE

        self._dask_data = None  # scene-level dask array, bound below
        self._cached_descriptors = None  # cached on first list_tensor_descriptors
        if scene_index is not None:
            # Scene-level: bind this scene's bioio dask array eagerly.
            self._bio_image.set_scene(scene_index)
            self._dask_data = self._bio_image.dask_data
            self.dim_labels = (
                dim_labels if dim_labels else list(self._bio_image.dims.order)
            )
        else:
            # Source-level: no bound reader; dim_labels is the default for scenes.
            self.dim_labels = dim_labels

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from this scene's bioio dask array.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape or called on source-level adapter
        """
        if self.scene_index is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))
        with self._io_lock:
            return self._dask_data[slices].compute()

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this adapter (bioio).

        Scene-level (scene_index set): computed from the bioio dask array.
        Source-level (scene_index=None): the first scene's descriptor.
        """
        if self.scene_index is not None:
            dask_data = self._dask_data
            chunk_shape = [max(c) for c in dask_data.chunks]
            return TensorDescriptor(
                array_id=self.array_id,
                dim_labels=self.dim_labels if self.dim_labels else [],
                shape=list(dask_data.shape),
                chunk_shape=chunk_shape,
                dtype=dask_data.dtype.str,
            )
        # Source-level: return first scene descriptor
        return self.list_tensor_descriptors()[0]

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors (scenes) available in this source via bioio.

        Uses OME metadata for shapes without scene switching when possible, else
        falls back to per-scene switching. Chunk info is NOT populated -- clients
        call get_flight_info for accurate per-scene chunk/metadata details.

        Returns:
            List of TensorDescriptor for all scenes in this source
        """
        # Source-level: use cached descriptors if available
        if self._cached_descriptors is not None:
            return self._cached_descriptors

        descriptors = []
        scene_ids = list(self._bio_image.scenes)

        # Try OME metadata first (much faster - no scene switching)
        try:
            ome_meta = self._bio_image.ome_metadata
            if (
                ome_meta is not None
                and hasattr(ome_meta, "images")
                and len(ome_meta.images) == len(scene_ids)
            ):
                labels = (
                    list(self.dim_labels)
                    if self.dim_labels
                    else list(self._bio_image.dims.order)
                )
                # The OME-pixels shape below is canonical 5-D TCZYX. It only
                # agrees with `labels` when the image really is plain TCZYX. An
                # RGB/samples source reports dims.order "TCZYXS" (bioio
                # folds the interleaved samples into a trailing S axis, and its
                # dask shape carries C=1,S=3 where OME reports C=3,no-S), so the
                # 5-D shape would disagree with the 6 labels and yield a
                # malformed descriptor -- get_flight_info then rejects every
                # slice as a dimensionality mismatch, so an RGB OME-TIFF fails to
                # open. Defer those to the authoritative scene-switching fallback
                # below, mirroring `_tczyx_shape`'s rejection of the S axis.
                if labels == list(_CANONICAL_DIMS):
                    # Get dtype from first scene (assumed consistent). Kept inside
                    # the canonical guard so a deferred RGB/samples source does not
                    # pay for a scene switch it will redo in the fallback below.
                    self._bio_image.set_scene(scene_ids[0])
                    dtype = self._bio_image.dask_data.dtype.str

                    # Get shapes from OME metadata (no scene switching)
                    # OME images are in same order as img.scenes
                    for i, im in enumerate(ome_meta.images):
                        px = im.pixels
                        shape = [
                            px.size_t,
                            px.size_c,
                            px.size_z,
                            px.size_y,
                            px.size_x,
                        ]

                        descriptors.append(
                            TensorDescriptor(
                                # Globally-unique array_id = source_id/field (the
                                # scene id is the within-source field). Identity
                                # policy: list_flights, get_flight_info, and the
                                # chunk_id all carry this one qualified form.
                                array_id=f"{self.source_id}/{scene_ids[i]}",
                                dim_labels=list(labels),
                                shape=shape,
                                chunk_shape=[],  # Not populated - call get_flight_info for chunk info
                                dtype=dtype,
                            )
                        )
        except NotImplementedError:
            # Some formats don't support ome_metadata - fall through to scene switching
            pass

        # Fallback: scene switching (slower but always works)
        if not descriptors:
            for scene_id in scene_ids:
                self._bio_image.set_scene(scene_id)
                dask_data = self._bio_image.dask_data

                descriptors.append(
                    TensorDescriptor(
                        # Globally-unique array_id = source_id/field (identity
                        # policy); the scene id is the within-source field.
                        array_id=f"{self.source_id}/{scene_id}",
                        dim_labels=self.dim_labels
                        if self.dim_labels
                        else list(self._bio_image.dims.order),
                        shape=list(dask_data.shape),
                        chunk_shape=[],  # Not populated - call get_flight_info for chunk info
                        dtype=dask_data.dtype.str,
                    )
                )

        # Cache for future calls
        self._cached_descriptors = descriptors
        return descriptors

    def _scene_index_for_field(self, field: Optional[str]) -> int:
        """Resolve a within-source scene field to its integer scene index.

        Prefers the cached descriptor order (biopb/biopb#168) so a read does NOT
        re-enumerate ``BioImage.scenes`` -- which would trigger the OME-XML
        object parse the fast path avoided at registration. The cached
        descriptors are in series/scene order, so the position IS the scene
        index, and bioio's ``set_scene`` takes that int directly. Falls
        back to enumerating scenes when no descriptors are cached (e.g. a read
        without a prior list_tensor_descriptors).
        """
        if self._cached_descriptors is not None:
            for i, d in enumerate(self._cached_descriptors):
                if self._within_source_field(d.array_id) == field:
                    return i
            raise ValueError(f"Unknown scene: {field}")
        scene_ids = list(self._bio_image.scenes)
        try:
            return scene_ids.index(field)
        except ValueError:
            raise ValueError(f"Unknown scene: {field}")

    def get_tensor_adapter(self, tensor_id: str) -> "BackendAdapter":
        """Get BackendAdapter for a specific scene within this source.

        Args:
            tensor_id: Scene identifier (scene_id from img.scenes)

        Returns:
            Adapter for the specified scene, with tensor context set
        """
        # Populate _cached_descriptors before resolving the scene index. Idempotent
        # (cached), and it closes the latent list(self._bio_image.scenes) parse in
        # _scene_index_for_field for a read that skipped registration.
        self.list_tensor_descriptors()

        # Accept either the within-source field (scene id) or the full
        # source-qualified array_id (identity policy: array_id = source_id/field).
        tensor_id = self._within_source_field(tensor_id)

        # Source-level: lazy initialize tensor level adapters
        scene_idx = self._scene_index_for_field(tensor_id)

        if hasattr(self, "_tensor_adapters"):
            # Check if adapter already exists for this scene
            if tensor_id in self._tensor_adapters:
                return self._tensor_adapters[tensor_id]
        else:
            self._tensor_adapters = {}

        adapter = self.__class__(
            self._bio_image,
            scene_index=scene_idx,
            source_id=self.source_id,
            dim_labels=self.dim_labels,
            source_url=self._source_url,
            io_lock=self._io_lock,
        )
        # Set tensor context in the adapter
        adapter._tensor_name = tensor_id
        self._tensor_adapters[tensor_id] = adapter

        return adapter

    def get_metadata(self) -> dict:
        """Return OME metadata as a dict (bioio ``ome_metadata`` model_dump).

        Returns:
            OME metadata as dict, or empty dict if unavailable.
        """
        try:
            ome_meta = self._bio_image.ome_metadata
            if ome_meta is None:
                return {}

            # ome_metadata is typically an OME object from ome-types
            # Convert to dict if it has a model_dump method (pydantic v2)
            # or dict method (pydantic v1)
            # Use mode='json' to ensure Enum fields (UnitsElectricPotential, etc.)
            # are serialized to their string representations
            if hasattr(ome_meta, "model_dump"):
                return ome_meta.model_dump(mode="json")
            elif hasattr(ome_meta, "dict"):
                return ome_meta.dict(by_alias=False, exclude_none=False)
            elif hasattr(ome_meta, "__dict__"):
                # Fallback: try to extract serializable attributes
                return {
                    k: v for k, v in ome_meta.__dict__.items() if not k.startswith("_")
                }
            return {}
        except Exception:
            return {}

    def _physical_scale(self):
        """Per-dim physical pixel size + unit from the bioio OME model.

        Reads ``ome_metadata.images[scene].pixels.physical_size_{x,y,z}`` directly
        (no full ``model_dump``) and maps onto ``dim_labels`` by axis label. T/C
        axes get ``0.0`` / ``""``. Returns ``None`` when no positive size is known.
        See ``TensorAdapter._physical_scale``.
        """
        try:
            ome = self._bio_image.ome_metadata
            if ome is None or not getattr(ome, "images", None):
                return None

            # OME images are in img.scenes order; a tensor-bound adapter knows
            # its scene index directly (callers reach this via get_tensor_adapter).
            idx = self.scene_index if self.scene_index is not None else 0
            if idx >= len(ome.images):
                return None

            px = ome.images[idx].pixels

            def _unit(u):
                if u is None:
                    return ""
                return str(getattr(u, "value", None) or u)

            by_label = {
                "x": (px.physical_size_x, _unit(px.physical_size_x_unit)),
                "y": (px.physical_size_y, _unit(px.physical_size_y_unit)),
                "z": (px.physical_size_z, _unit(px.physical_size_z_unit)),
            }

            labels = self.dim_labels or list(self._bio_image.dims.order)
            scale, unit = [], []
            for lab in labels:
                v, u = by_label.get(str(lab).lower(), (None, ""))
                try:
                    fv = float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    fv = 0.0
                if fv > 0:
                    scale.append(fv)
                    unit.append(u)
                else:
                    scale.append(0.0)
                    unit.append("")
            if not any(scale):
                return None
            return scale, unit
        except Exception:
            return None


# =============================================================================
# Format-specific subclasses
# =============================================================================


class ZeissAdapter(_BioioAdapterBase):
    """Adapter for Zeiss microscopy files (CZI and LSM)."""

    SOURCE_TYPE = "zeiss"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim Zeiss CZI and LSM files."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if name.endswith(".czi") or name.endswith(".lsm"):
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type=cls.SOURCE_TYPE,
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
            )
        return None


class LeicaAdapter(_BioioAdapterBase):
    """Adapter for Leica LIF files."""

    SOURCE_TYPE = "leica"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim Leica LIF files."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if name.endswith(".lif"):
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type=cls.SOURCE_TYPE,
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
            )
        return None


class NikonAdapter(_BioioAdapterBase):
    """Adapter for Nikon ND2 files."""

    SOURCE_TYPE = "nikon"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim Nikon ND2 files."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if name.endswith(".nd2"):
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type=cls.SOURCE_TYPE,
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
            )
        return None


class DvAdapter(_BioioAdapterBase):
    """Adapter for DeltaVision DV files."""

    SOURCE_TYPE = "dv"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim DeltaVision DV files."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if name.endswith(".dv"):
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type=cls.SOURCE_TYPE,
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
            )
        return None


class OlympusAdapter(_BioioAdapterBase):
    """Adapter for Olympus OIF and OIB files."""

    SOURCE_TYPE = "olympus"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim Olympus OIF and OIB files."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if name.endswith(".oif") or name.endswith(".oib"):
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type=cls.SOURCE_TYPE,
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
            )
        return None


class BioformatsAdapter(_BioioAdapterBase):
    """Bio-Formats fallback for legacy formats with no pure-Python reader.

    Handles proprietary/legacy formats that only the Java Bio-Formats library
    can read -- ZVI (Zeiss AxioVision) being the headline case. Claims a file
    only when ``bioio_bioformats`` is importable, so installs without the optional
    ``bioformats`` component skip these files (with a warning) instead of
    failing later at read time.

    Reading goes through bioio's Bio-Formats plugin (``bioio-bioformats``), which
    BioImage auto-selects once the plugin is present. A Java runtime is fetched
    lazily by scyjava/cjdk on first read; it is not a build or system dependency.

    Only claims extensions not already handled by a more specific adapter
    (.oib/.oif -> OlympusAdapter, .ims -> AicsImageIoAdapter).
    """

    SOURCE_TYPE = "bioformats"

    # Bio-Formats-only formats lacking a pure-Python reader and not claimed by
    # another adapter. ZVI is the one users actually lost.
    BIOFORMATS_ONLY_EXTENSIONS = (".zvi", ".lei", ".vsi")

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim legacy Bio-Formats-only files when Bio-Formats is available."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if not any(name.endswith(ext) for ext in cls.BIOFORMATS_ONLY_EXTENSIONS):
            return None

        # Gate on the Bio-Formats plugin (importing it does NOT start a JVM).
        # Without it, skip the file loudly rather than claiming and failing
        # later at read time.
        try:
            import bioio_bioformats  # noqa: F401
        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "Skipping %s: it requires Bio-Formats, which is not installed. "
                "Install the optional component with "
                "`pip install biopb-tensor-server[bioformats]` to enable it "
                "(a Java runtime is downloaded automatically on first use).",
                ctx.path_str,
            )
            return None

        state.try_claim_path(ctx.path_str)
        return SourceClaim(
            source_type=cls.SOURCE_TYPE,
            primary_path=ctx.path_str,
            is_remote=ctx.is_remote,
        )


class AicsImageIoAdapter(_BioioAdapterBase):
    """Fallback adapter for remaining bioio-supported formats.

    Claims microscopy/scientific image files not handled by format-specific
    subclasses. By default the claim set is MICROSCOPY_EXTENSIONS; generic
    raster/video (GENERIC_IMAGE_EXTENSIONS) are claimed during recursive
    discovery only when the ``claim_generic_images`` server config flag is on
    (biopb/biopb#40). Generic file types (txt, csv, cfg, etc.) that bioformats
    technically supports are never claimed.

    Note: Some formats handled by specific adapters:
    - .tif with embedded OME-XML → OmeTiffAdapter (pure tifffile; a remote/exotic
      .tif it declines falls through to this generic adapter)
    - .czi, .lsm → ZeissAdapter
    - .lif → LeicaAdapter
    - .nd2 → NikonAdapter
    - .dv → DvAdapter
    - .oif, .oib → OlympusAdapter
    """

    SOURCE_TYPE = "aics"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim bioio-supported files not handled by other adapters."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()

        # Format-specific extensions this generic adapter must NOT claim: the
        # vendor extensions below are owned by their subclasses, and .companion.ome
        # is a metadata sidecar that is no longer supported (declined, not read).
        specific_extensions = (
            ".companion.ome",
            ".czi",
            ".lsm",
            ".lif",
            ".nd2",
            ".dv",
            ".oif",
            ".oib",
        )
        for ext in specific_extensions:
            if name.endswith(ext):
                return None  # Let the specific adapter handle this

        # Check for remaining bioio-supported extensions. Microscopy/scientific
        # formats are always eligible; generic raster/video are included only
        # when the generic-images opt-in is on (biopb/biopb#40).
        for ext in _claim_extensions():
            if name.endswith(ext):
                state.try_claim_path(ctx.path_str)
                return SourceClaim(
                    source_type=cls.SOURCE_TYPE,
                    primary_path=ctx.path_str,
                    is_remote=ctx.is_remote,
                )

        return None
