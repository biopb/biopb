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
import math
import os
import threading
import time
from itertools import product
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import (
    IdleHandleReaper,
)
from biopb_tensor_server.core import chunk as chunk_policy
from biopb_tensor_server.core.adapter_base import TensorAdapter
from biopb_tensor_server.core.chunk import (
    compute_transfer_chunk_size,
    content_version_from_path,
    default_transfer_chunk_shape,
    estimate_chunk_bytes,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import TensorNotFound

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bioio import BioImage

    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState


# Canonical OME dimension order. The scene-listing path uses it to
# detect a plain TCZYX source (which it can shape from OME Pixels) versus an
# RGB/samples one it must defer to scene switching.
_CANONICAL_DIMS = "TCZYX"

# "Not built yet" for the ND2 frame-index cache. Distinct from None, which is a
# real cached answer meaning this scene's frames cannot be addressed by T/Z.
_FRAME_INDEX_UNCACHED = object()


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
        # Microscopy-specific formats (not handled by specific adapters).
        # NOTE: MRC (.mrc/.mrcs) is deliberately NOT here -- no installed bioio
        # plugin can read a plain cryo-EM MRC (bioio-dv's `mrc` lib is
        # DeltaVision-only), so claiming it produced a claim-then-error. MRC is
        # owned by MrcAdapter (rosettasciio), registered ahead of this adapter.
        # See biopb/biopb#94.
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

# Full curated set (microscopy + generic). The actual claim scope is decided
# per-call by :func:`_claim_extensions`, which honors the generic-images opt-in.
CORE_IMAGE_EXTENSIONS = MICROSCOPY_EXTENSIONS | GENERIC_IMAGE_EXTENSIONS


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


class _BioioAdapterBase(TensorAdapter):
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
    RETAIN_SCENE_DASK = True

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
        metadata_cache: Optional[Any] = None,
        shared_handle: Optional[Any] = None,
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
            metadata_cache: Optional processed metadata shared by scene adapters.
        """
        self._bio_image = bio_image
        self.scene_index = scene_index
        self.source_id = source_id

        # Serializes every touch of the shared BioImage. Reentrant because the
        # guarded methods nest -- list_tensor_descriptors reaches _bio_image
        # both directly and through _metadata_for_listing.
        # Source-level creates lock, scene-level receives from source
        if io_lock is not None:
            self._io_lock = io_lock
        else:
            self._io_lock = threading.RLock()

        # Source-level metadata for DataSourceDescriptor
        if source_url:
            self._source_url = source_url
        elif hasattr(bio_image, "source") and hasattr(bio_image.source, "path"):
            self._source_url = str(bio_image.source.path)
        else:
            self._source_url = ""
        # Cheap content_version from the file's stat signature (#178): O(1),
        # folded into minted chunk_ids so a re-saved file gets a fresh cache
        # namespace. Detached-header formats (.mhd/.hdr) stat only the master --
        # a documented blind spot. None (unresolved url) leaves it unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = self.SOURCE_TYPE
        self._metadata_cache = metadata_cache

        self._dask_data = None  # scene-level dask array, bound below
        self._scene_descriptor = None
        # (generation, frame map) -- the map is per SCENE, but it indexes into a
        # reader shared by every scene of this source, so a reopen invalidates it.
        self._nd2_frame_index_cache: Any = _FRAME_INDEX_UNCACHED
        self._nd2_frame_index_generation = -1
        # A reader/handle shared with this source's other scene adapters, for
        # subclasses that keep one warm. None here means "not shared yet";
        # whoever needs it makes it (see NikonAdapter._reader_handle).
        self._shared_handle: Any = shared_handle
        self._cached_descriptors = None  # cached on first list_tensor_descriptors
        # Per-scene adapter cache, source-level only. Assigned here (not lazily on
        # first get_tensor_adapter) so no code path has to hedge about whether the
        # attribute exists; a per-instance dict, never a class attribute, for the
        # reason spelled out in biopb/biopb#522.
        self._tensor_adapters: dict = {}
        if scene_index is not None:
            # Scene-level: bind this scene's bioio dask array eagerly.
            with self._io_lock:
                self._bio_image.set_scene(scene_index)
                self._dask_data = self._bio_image.dask_data
                self.dim_labels = (
                    dim_labels if dim_labels else list(self._bio_image.dims.order)
                )
                if not self.RETAIN_SCENE_DASK:
                    self._scene_descriptor = self._descriptor_from_dask(self._dask_data)
                    self._dask_data = None
                    self._release_bioio_dask_cache()
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
        slices = self._bounds_to_slices(bounds)
        with self._io_lock:
            return self._dask_data[slices].compute()

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this adapter (bioio).

        Scene-level (scene_index set): computed from the bioio dask array.
        Source-level (scene_index=None): the first scene's descriptor.
        """
        if self.scene_index is not None:
            if self._scene_descriptor is not None:
                result = TensorDescriptor()
                result.CopyFrom(self._scene_descriptor)
                # The snapshot is built in the constructor, before the
                # source-level adapter assigns this scene's ``_tensor_name``.
                # Bind identity at retrieval time so every scene gets its
                # source-qualified array_id rather than the bare source_id.
                result.array_id = self.array_id
                return result
            dask_data = self._dask_data
            return self._descriptor_from_dask(dask_data)
        # Source-level: the default (first) scene -- answered by the adapter
        # bound to it, not read back off the catalog listing, which carries no
        # transfer grid (biopb/biopb#812).
        entries = self.list_tensor_descriptors()
        if not entries:
            raise TensorNotFound(
                f"source {self.source_id!r} exposes no scenes",
                reason="unknown_source",
            )
        return self.get_tensor_adapter(entries[0].array_id).get_tensor_descriptor()

    def _native_block(self, dask_data: Any) -> Optional[List[int]]:
        """The backend's own block, used to align the transfer grid.

        Only an alignment seed: no read is issued at this granularity
        (biopb/biopb#809). An adapter whose bytes sit on disk in a shape BioIO's
        block does not describe overrides :meth:`_transfer_chunk_shape` instead.
        """
        return [max(c) for c in dask_data.chunks]

    def _transfer_chunk_shape(
        self, shape: List[int], dtype: str, dask_data: Any
    ) -> List[int]:
        """This tensor's transfer grid -- the meaning of ``chunk_shape`` (#809).

        Only ever called on a scene-bound adapter, whose ``dask_data`` is that
        scene's own array: the grid is a serving fact and the listing path does
        not compute one (biopb/biopb#812).
        """
        return default_transfer_chunk_shape(
            shape, dtype, self.dim_labels, native=self._native_block(dask_data)
        )

    def _descriptor_from_dask(self, dask_data: Any) -> TensorDescriptor:
        """Snapshot the structural facts exposed by a scene Dask array."""
        shape = list(dask_data.shape)
        dtype = dask_data.dtype.str
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels if self.dim_labels else [],
            shape=shape,
            chunk_shape=self._transfer_chunk_shape(shape, dtype, dask_data),
            dtype=dtype,
        )

    def _release_bioio_dask_cache(self) -> None:
        """Release BioIO's current-scene lazy array without clearing metadata."""
        with self._io_lock:
            self._bio_image._xarray_dask_data = None
            self._bio_image._dims = None
            reader = self._bio_image.reader
            reader._xarray_dask_data = None
            reader._dims = None

    def _metadata_for_listing(self) -> Any:
        """Return metadata used by the cheap multi-scene descriptor path."""
        with self._io_lock:
            return self._bio_image.ome_metadata

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List every scene as a structural catalog entry.

        Uses OME metadata for shapes without scene switching when possible, else
        falls back to per-scene switching. The transfer grid is NOT populated: it
        is a serving fact of the scene ``get_tensor_adapter`` binds, and the
        fast path here has bound no scene to answer for it (biopb/biopb#812).
        Clients call ``GetFlightInfo`` for the grid.

        Returns:
            List of TensorDescriptor for all scenes in this source
        """
        # Source-level: use cached descriptors if available
        if self._cached_descriptors is not None:
            return self._cached_descriptors

        with self._io_lock:
            # Re-check: another thread may have built these while this one
            # waited for the lock.
            if self._cached_descriptors is not None:
                return self._cached_descriptors
            self._cached_descriptors = self._build_tensor_descriptors()
        return self._cached_descriptors

    def _build_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Enumerate every scene's descriptor. Caller holds ``_io_lock``."""
        descriptors = []
        scene_ids = list(self._bio_image.scenes)

        # Try OME metadata first (much faster - no scene switching)
        try:
            ome_meta = self._metadata_for_listing()
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
                labels = (
                    list(self.dim_labels)
                    if self.dim_labels
                    else list(self._bio_image.dims.order)
                )

                descriptors.append(
                    TensorDescriptor(
                        # Globally-unique array_id = source_id/field (identity
                        # policy); the scene id is the within-source field.
                        array_id=f"{self.source_id}/{scene_id}",
                        dim_labels=labels,
                        shape=list(dask_data.shape),
                        dtype=dask_data.dtype.str,
                    )
                )

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
            raise TensorNotFound(f"Unknown scene: {field}", reason="unknown_field")
        with self._io_lock:
            scene_ids = list(self._bio_image.scenes)
        try:
            return scene_ids.index(field)
        except ValueError as e:
            raise TensorNotFound(
                f"Unknown scene: {field}", reason="unknown_field"
            ) from e

    def get_tensor_adapter(self, tensor_id: str) -> "TensorAdapter":
        """Get the tensor adapter for a specific scene within this source.

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

        if tensor_id in self._tensor_adapters:
            return self._tensor_adapters[tensor_id]

        adapter = self.__class__(
            self._bio_image,
            scene_index=scene_idx,
            source_id=self.source_id,
            dim_labels=self.dim_labels,
            source_url=self._source_url,
            io_lock=self._io_lock,
            metadata_cache=self._metadata_cache,
            shared_handle=self._shared_handle_for_scenes(),
        )
        # Set tensor context in the adapter
        adapter._tensor_name = tensor_id
        self._tensor_adapters[tensor_id] = adapter

        return adapter

    def _shared_handle_for_scenes(self) -> Any:
        """The handle this source's scene adapters should share, if any.

        None for every format that reopens per read. A subclass keeping one
        reader warm returns it here so its scenes share one, rather than each
        opening its own against the same file.
        """
        return None

    def get_metadata(self) -> dict:
        """Return OME metadata as a dict (bioio ``ome_metadata`` model_dump).

        Returns:
            OME metadata as dict, or empty dict if unavailable.
        """
        try:
            with self._io_lock:
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
            with self._io_lock:
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

            if self.dim_labels:
                labels = self.dim_labels
            else:
                with self._io_lock:
                    labels = list(self._bio_image.dims.order)
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
        if name.endswith((".czi", ".lsm")):
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


# One pool for ND2 readers, separate from the OME-TIFF and NDTiff pools so each
# is bounded on its own terms. See :mod:`_handle_reaper`.
#
# ND2 is the case that module's "reopen per read" default does not cover, and it
# is also the case its long TTL does not fit. The default is argued from open
# cost, and for ND2 the open really is free: 169 open/close pairs on a 21 GB file
# measure 16 ms. What a held ND2 reader saves is not the open but the *mapping*.
# ``nd2.read_frame`` returns a zero-copy view onto the reader's mmap, so a reopen
# hands every call a fresh mapping with an empty page table, and the crop then
# re-faults every row it touches even though the bytes are already in page cache.
# The bill scales with rows revisited, not bytes delivered: tiling one
# 14234x14234 scene on the 1182 transfer grid touches each row once per column of
# tiles, and 169 reads that way cost 203k minor faults and 1.65 s warm where the
# same pixels through one held reader cost 18.5k and 0.30 s.
#
# That is why the TTL is seconds rather than minutes. A warm page table is worth
# only as long as the read that built it -- across an idle gap the next read
# faults in whatever *it* touches regardless, so holding buys nothing and keeps a
# whole-file mapping pinned. 5 s spans one logical read with margin: a full-scene
# read of ND040 measures 4.4 s wall with the handle held.
_ND2_READER_TTL = 5.0

_nd2_reader_reaper = IdleHandleReaper(
    _ND2_READER_TTL, "nd2-reader-reaper", max_handles=8
)


class _Nd2Reader:
    """One ND2 reader, shared by every scene adapter of one source.

    The :class:`~biopb_tensor_server.adapters._handle_reaper.ReapableHandle` is
    this object rather than the adapter because the handle is per *file* while
    adapters are per scene. ND040 has 18 scenes, and a reader per scene would map
    the same 21.8 GB file 18 times to serve reads that are already serialized
    behind the one ``_io_lock`` they share.
    """

    def __init__(self, source_url: str, io_lock) -> None:
        self._source_url = source_url
        # The source's lock, shared with every scene adapter -- so the fence the
        # reaper takes is the same one reads hold.
        self._io_lock = io_lock
        self._reader: Any = None
        # Reads hold ``_io_lock`` end to end, so none is ever in flight when the
        # reaper takes it.
        self._active_reads = 0
        self._persistent_last_access = 0.0
        # Bumped on every open. A scene's frame map indexes into a specific
        # reader; after a reap the offsets belong to a closed mapping, so the map
        # is rebuilt rather than reused.
        self.generation = 0

    def acquire(self) -> Any:
        """Open (or reuse) the reader. Caller holds ``_io_lock``."""
        # Stamped before register, so this handle sorts newest and a cap eviction
        # triggered by its own register never picks it.
        self._persistent_last_access = time.monotonic()
        if self._reader is None:
            import nd2

            self._reader = nd2.ND2File(self._source_url)
            self.generation += 1
            _nd2_reader_reaper.register(self)
        return self._reader

    def _release_persistent_handle(self) -> None:
        """Close the reader and permit a later reopen.

        Caller holds ``_io_lock`` (read path / reaper) or is the GC finalizer.
        Safe to call repeatedly.
        """
        reader, self._reader = self._reader, None
        _nd2_reader_reaper.discard(self)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                logger.debug("error closing persistent ND2 reader", exc_info=True)

    def __del__(self):
        try:
            self._release_persistent_handle()
        except Exception:
            pass


class NikonAdapter(_BioioAdapterBase):
    """Adapter for Nikon ND2 files."""

    SOURCE_TYPE = "nikon"
    RETAIN_SCENE_DASK = False

    def _processed_metadata(self) -> Any:
        """Return BioIO's processed metadata, degrading to empty on failure."""
        if self._metadata_cache is None:
            with self._io_lock:
                try:
                    self._metadata_cache = self._bio_image.metadata
                except Exception:
                    self._metadata_cache = {}
        return self._metadata_cache

    def _metadata_for_listing(self) -> Any:
        return self._processed_metadata()

    def get_metadata(self) -> dict:
        """Read and retain BioIO's processed metadata once for this ND2 source."""
        return self._metadata_to_dict(self._processed_metadata())

    @staticmethod
    def _metadata_to_dict(metadata: Any) -> dict:
        if metadata is None:
            return {}
        if hasattr(metadata, "model_dump"):
            return metadata.model_dump(mode="json")
        if hasattr(metadata, "dict"):
            return metadata.dict(by_alias=False, exclude_none=False)
        if hasattr(metadata, "__dict__"):
            return {
                key: value
                for key, value in metadata.__dict__.items()
                if not key.startswith("_")
            }
        return {}

    def _physical_scale(self):
        """Derive this scene's scale from the shared processed metadata cache."""
        ome = self._processed_metadata()
        if ome is None or not getattr(ome, "images", None):
            return None
        idx = self.scene_index if self.scene_index is not None else 0
        if idx >= len(ome.images):
            return None
        px = ome.images[idx].pixels

        def unit(value):
            if value is None:
                return ""
            return str(getattr(value, "value", None) or value)

        by_label = {
            "x": (px.physical_size_x, unit(px.physical_size_x_unit)),
            "y": (px.physical_size_y, unit(px.physical_size_y_unit)),
            "z": (px.physical_size_z, unit(px.physical_size_z_unit)),
        }
        scale, units = [], []
        for label in self.dim_labels or []:
            value, axis_unit = by_label.get(str(label).lower(), (None, ""))
            try:
                numeric = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                numeric = 0.0
            scale.append(numeric if numeric > 0 else 0.0)
            units.append(axis_unit if numeric > 0 else "")
        return (scale, units) if any(scale) else None

    def _native_block(self, dask_data: Any) -> Optional[List[int]]:
        """Describe one ND2 sequence frame, preserving its pixel layout.

        ``nd2.read_frame`` indexes the acquisition loops (T/Z and the scene's
        position) and returns the complete C/Y/X[/S] frame. Some BioIO/Dask
        arrays report several sequence frames as one chunk, which is not the
        granularity available from the reader below. Keep any smaller pixel
        tiling BioIO reports, but never combine T or Z frames in this seed.
        """
        block = super()._native_block(dask_data)
        if block is None or len(block) != len(self.dim_labels or []):
            return block
        for axis, label in enumerate(self.dim_labels or []):
            if label.upper() in {"T", "Z"}:
                block[axis] = 1
        return block

    def _transfer_chunk_shape(
        self, shape: List[int], dtype: str, dask_data: Any
    ) -> List[int]:
        """Never split an ND2's component axes -- they are inside the pixel.

        ND2 has no planar variant. Every frame is materialised in one layout,
        ``(Y, X, channel, RGB component)`` (``modern_reader._actual_frame_shape``),
        so both C and S sit *below* X: one channel's bytes are ``itemsize`` of
        every ``componentCount * itemsize``. A per-channel chunk therefore faults
        in every page the other components occupy and discards what it did not
        ask for, then the next channel repeats the read. Measured on a 4-channel
        14234-wide uint16 file, full-width band, pages warm: all four channels in
        one read cost 96.4 ms for 233.2 MB (0.41 ms/MB) against 43.9 ms for
        58.3 MB (0.75 ms/MB) for one -- 4x the pixels for 2.2x the time
        (biopb/biopb#806).

        There is nothing to detect. An earlier version probed ``widthBytes ==
        widthPx * componentCount * itemsize`` and treated a mismatch as planar,
        but that identity is nd2's test for **row padding**, not for layout: when
        it fails the reader keeps the same interleaved shape and uses
        ``widthBytes`` as the row stride. So the probe read a padded row -- an
        odd-width RGB camera -- as planar and handed C back to the generic
        splitter, which is exactly the bug #806 is about.

        The declared *unit* is one pixel's worth of every component. C and S are
        inside it and so cannot be split, and nothing downstream re-shapes a
        declared grid (biopb/biopb#809) -- which is what makes this stick, where
        the old fixed divide/coalesce priority took C apart again. The plane is
        left to the shared sizing, which grows Y and X coupled: a full-width band
        reads better sequentially, but turns a 512x512 tile into one fetch per
        band it crosses, and between C-whole aspect ratios the cold cost is
        within scene-to-scene variance (a paired A/B of 1024x1024 against
        512x2048 on untouched scenes ranged 0.73x to 4.2x in *both* directions).
        Keeping the components together is where the measurable win is.
        """
        labels = [str(label).upper() for label in (self.dim_labels or [])]
        if len(labels) != len(shape):
            return super()._transfer_chunk_shape(shape, dtype, dask_data)
        unit = [
            int(size) if label in {"C", "S"} else 1
            for label, size in zip(labels, shape, strict=True)
        ]
        # componentCount == 1: C and S are absent or singleton, so there is no
        # interleaved run to protect and the backend's own block is the better
        # seed. (nd2 reports C = componentCount for a mono file and C x S =
        # componentCount for an RGB one, so this product IS componentCount.)
        if math.prod(unit) <= 1:
            return super()._transfer_chunk_shape(shape, dtype, dask_data)
        # One pixel of every component already at or above the target leaves
        # nothing to grow; declare it and let the Arrow clamp handle the rest.
        if estimate_chunk_bytes(tuple(unit), dtype) >= (
            chunk_policy.PREFERRED_ARROW_BATCH_BYTES
        ):
            return unit
        return list(
            compute_transfer_chunk_size(tuple(unit), tuple(shape), dtype, labels)
        )

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read requested pixels directly from ND2 sequence frames.

        BioIO remains authoritative for scenes and metadata. Pixel reads skip
        its Dask graph because ``nd2.read_frame`` exposes the same frame-level
        access directly. Only the requested subregion is copied, while the ND2
        file is open, so no reader-backed view escapes this method.
        """
        if self.scene_index is None:
            raise ValueError("Cannot get data from source-level adapter")

        TensorAdapter.get_data(self, bounds)
        if not self._source_url or not os.path.isfile(self._source_url):
            return self._get_data_via_bioio(bounds)

        desc = self.get_tensor_descriptor()
        labels = [label.upper() for label in desc.dim_labels]
        if len(labels) != len(desc.shape) or not {"Y", "X"}.issubset(labels):
            return self._get_data_via_bioio(bounds)
        if any(label not in {"T", "C", "Z", "Y", "X", "S"} for label in labels):
            return self._get_data_via_bioio(bounds)
        shape = tuple(int(dim) for dim in desc.shape)
        starts = tuple(int(value) for value in bounds.start)
        stops = tuple(int(value) for value in bounds.stop)
        output = np.empty(
            tuple(stop - start for start, stop in zip(starts, stops, strict=True)),
            dtype=np.dtype(desc.dtype),
        )

        sequence_axes = [
            axis for axis, label in enumerate(labels) if label in {"T", "Z"}
        ]
        sequence_ranges = [range(starts[axis], stops[axis]) for axis in sequence_axes]
        frame_shape = tuple(
            1 if label in {"T", "Z"} else size
            for label, size in zip(labels, shape, strict=True)
        )

        handle = self._reader_handle()
        with self._io_lock:
            reader = handle.acquire()
            try:
                output = self._read_frames_into(
                    reader,
                    handle,
                    labels,
                    starts,
                    stops,
                    output,
                    sequence_axes,
                    sequence_ranges,
                    frame_shape,
                )
            except Exception:
                # A half-open reader is not reusable; drop it so the next read
                # reopens rather than failing on the same handle.
                handle._release_persistent_handle()
                raise
        fallback_to_bioio = output is None

        if fallback_to_bioio:
            return self._get_data_via_bioio(bounds)
        return output

    def _read_frames_into(
        self,
        reader,
        handle,
        labels,
        starts,
        stops,
        output,
        sequence_axes,
        sequence_ranges,
        frame_shape,
    ):
        """Copy the requested subregion out of ``reader``. None -> use BioIO.

        Caller holds ``_io_lock``, which is also what fences the persistent
        reader against the reaper: every view taken here is copied into
        ``output`` before the lock is released, so no reader-backed memory
        outlives the call even though the reader itself now does.
        """
        frame_indices = self._nd2_frame_indices(reader, handle)
        requested_frames = []
        for coordinates in product(*sequence_ranges):
            coordinate_by_label = {
                labels[axis]: coordinate
                for axis, coordinate in zip(sequence_axes, coordinates, strict=True)
            }
            key = (
                coordinate_by_label.get("T", 0),
                coordinate_by_label.get("Z", 0),
            )
            requested_frames.append((coordinate_by_label, key))

        # BioIO serves what the direct path cannot address: a scene whose
        # frames have no T/Z map, or a coordinate the ND2 experiment never
        # enumerated. Decided here but acted on below, once _io_lock is
        # released -- _get_data_via_bioio takes that same lock.
        fallback_to_bioio = frame_indices is None or any(
            key not in frame_indices for _, key in requested_frames
        )
        if not fallback_to_bioio:
            for coordinate_by_label, key in requested_frames:
                frame_index = frame_indices[key]
                frame = reader.read_frame(frame_index).reshape(frame_shape)

                source_slices = []
                output_slices = []
                for axis, label in enumerate(labels):
                    if label in {"T", "Z"}:
                        coordinate = coordinate_by_label[label]
                        source_slices.append(slice(0, 1))
                        output_slices.append(
                            slice(
                                coordinate - starts[axis],
                                coordinate - starts[axis] + 1,
                            )
                        )
                    else:
                        source_slices.append(slice(starts[axis], stops[axis]))
                        output_slices.append(slice(None))
                output[tuple(output_slices)] = frame[tuple(source_slices)]

        return None if fallback_to_bioio else output

    def _reader_handle(self) -> "_Nd2Reader":
        """This source's shared reader, made on first use.

        Scene adapters receive it from the source-level adapter
        (``_shared_handle_for_scenes``); one constructed directly -- a test, a
        benchmark -- makes its own.
        """
        if self._shared_handle is None:
            self._shared_handle = _Nd2Reader(self._source_url, self._io_lock)
        return self._shared_handle

    def _shared_handle_for_scenes(self) -> Any:
        return self._reader_handle()

    def close(self) -> None:
        """Release this source's reader, including its scene adapters'.

        They share one, so closing it here closes it for all of them; the next
        read on any of them reopens.
        """
        for adapter in list(self._tensor_adapters.values()):
            close = getattr(adapter, "close", None)
            if close is not None:
                close()
        if self._shared_handle is not None:
            with self._io_lock:
                self._shared_handle._release_persistent_handle()

    def _get_data_via_bioio(self, bounds: ChunkBounds) -> np.ndarray:
        """Use an ephemeral BioIO Dask array for unsupported direct-read cases."""
        slices = self._bounds_to_slices(bounds)
        with self._io_lock:
            self._bio_image.set_scene(self.scene_index)
            try:
                return self._bio_image.dask_data[slices].compute()
            finally:
                self._release_bioio_dask_cache()

    def _nd2_frame_indices(
        self, reader: Any, handle: "_Nd2Reader"
    ) -> Optional[dict[tuple[int, int], int]]:
        """Map this BioIO scene's T/Z coordinates to ND2 sequence indices.

        Returns None when two frames share a (T, Z) coordinate. An unknown or
        custom acquisition loop collapses onto one T/Z pair, so the direct path
        cannot tell those frames apart; the caller reads the scene through
        BioIO rather than choosing one of them here.

        Cached against the handle's generation: the map indexes into a specific
        reader, so a reap between reads invalidates it. Rebuilding is one pass
        over ``loop_indices``; reusing it against a remapped file would not fail,
        it would read the wrong frames.
        """
        if (
            self._nd2_frame_index_cache is not _FRAME_INDEX_UNCACHED
            and self._nd2_frame_index_generation == handle.generation
        ):
            return self._nd2_frame_index_cache
        self._nd2_frame_index_generation = handle.generation
        result: dict[tuple[int, int], int] = {}
        for frame_index, coordinates in enumerate(reader.loop_indices):
            if int(coordinates.get("P", 0)) != self.scene_index:
                continue
            key = (int(coordinates.get("T", 0)), int(coordinates.get("Z", 0)))
            if key in result:
                self._nd2_frame_index_cache = None
                return None
            result[key] = frame_index
        self._nd2_frame_index_cache = result
        return result

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
        if name.endswith((".oif", ".oib")):
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
