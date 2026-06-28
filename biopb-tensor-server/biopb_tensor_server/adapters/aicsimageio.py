"""AICSImageIO adapters for vendor microscopy formats.

This module provides a base class and format-specific subclasses for reading
various microscopy formats through aicsimageio's AICSImage class.

Format-specific subclasses provide meaningful source_type values:
- OmeTiffAdapter: "ome-tiff" (embedded OME-XML, companion.ome)
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

Relies on OS page cache for raw data caching.
"""

import logging
import os
import re
import threading
import time
import weakref
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.discovery import ClaimContext, SourceClaim

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aicsimageio import AICSImage

    from biopb_tensor_server.base import BackendAdapter
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState
    from biopb_tensor_server.remote import RemoteStore


# =============================================================================
# OME-XML metadata helpers (for OME-TIFF handling)
# =============================================================================


def _get_namespace(root) -> dict:
    """Extract namespace from root element tag.

    Args:
        root: ElementTree root element

    Returns:
        Dictionary with namespace mapping for OME schema
    """
    tag = root.tag
    if tag.startswith("{"):
        namespace = tag.split("}")[0].strip("{")
        return {"ome": namespace}
    return {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


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

        for tiff_data in root.findall(".//ome:TiffData", namespace):
            uuid_elem = tiff_data.find("ome:UUID", namespace)
            if uuid_elem is None:
                for child in tiff_data:
                    if child.tag.endswith("UUID") or child.tag == "UUID":
                        uuid_elem = child
                        break

            if uuid_elem is not None:
                filename = uuid_elem.get("FileName")
                if filename and filename not in seen_files:
                    if store is not None:
                        if source_dir:
                            file_path = store._join(str(source_dir) + "/" + filename)
                        else:
                            file_path = store._join(filename)
                        exists = store.isfile(file_path)
                    else:
                        file_path = Path(source_dir) / filename
                        exists = file_path.exists()

                    if exists:
                        files.append(file_path)
                        seen_files.add(filename)

        return files if files else None
    except ET.ParseError:
        return None


# Process-wide memoization of the embedded-OME-XML probe (biopb/biopb#56, item 6).
# A steady-state rescan opens every monitored .tif through tifffile just to learn
# whether it carries OME-XML — the dominant cost of the post-#63 claim phase
# (~100 ms / 64 tiffs on a real tree). The result is a pure function of the file's
# bytes, so it is cached keyed on the state walk's content-identity signature
# (st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns): any byte change bumps the
# signature, so a hit provably means identical content. A cached value of ``None``
# ("no OME-XML") is meaningful and is stored too, so membership — not truthiness —
# decides a hit. Bounded LRU; only the snapshot-driven path passes a signature, so
# the single-threaded watcher is the only writer, but the lock keeps it safe if a
# concurrent live walk ever supplies one.
_OME_META_CACHE: "OrderedDict[Tuple[str, Tuple], Optional[str]]" = OrderedDict()
_OME_META_CACHE_MAX = 4096
_OME_META_CACHE_LOCK = threading.Lock()


def _probe_ome_metadata_from_tiff(path: Path) -> Optional[str]:
    """Open the TIFF and return its embedded OME-XML, or None. No caching."""
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tf:
            if hasattr(tf, "ome_metadata") and tf.ome_metadata is not None:
                return tf.ome_metadata
    except Exception:
        return None
    return None


def _get_ome_metadata_from_tiff(
    path: Path, signature: Optional[Tuple] = None
) -> Optional[str]:
    """Extract OME-XML metadata from a TIFF file if present.

    Args:
        path: Path to TIFF file
        signature: Content-identity signature for ``path`` (from the discovery
            state walk). When given, the probe result is memoized on
            ``(path, signature)`` so an unchanged file is not reopened on the next
            rescan. When ``None`` (live walk / ad-hoc call) the probe runs uncached.

    Returns:
        OME-XML string if present, None otherwise
    """
    if signature is None:
        return _probe_ome_metadata_from_tiff(path)

    key = (str(path), signature)
    with _OME_META_CACHE_LOCK:
        if key in _OME_META_CACHE:
            _OME_META_CACHE.move_to_end(key)
            return _OME_META_CACHE[key]

    result = _probe_ome_metadata_from_tiff(path)

    with _OME_META_CACHE_LOCK:
        _OME_META_CACHE[key] = result
        _OME_META_CACHE.move_to_end(key)
        while len(_OME_META_CACHE) > _OME_META_CACHE_MAX:
            _OME_META_CACHE.popitem(last=False)
    return result


# =============================================================================
# tifffile-direct descriptor fast path (biopb/biopb#168)
# =============================================================================
#
# Registering a large OME-TIFF used to materialize the AICSImage OME model
# (``ome_metadata`` -> ome-types/pydantic objects, one per plane) just to learn
# shape/dims/dtype/scenes for the catalog row -- ~97 s for a single 40k-plane
# MMStack, serial across sources, dominating server startup. tifffile already
# walks the IFDs and parses the OME-XML cheaply (~6 s), exposing the same
# shape/axes/dtype per series WITHOUT building the pydantic object graph. These
# helpers build the descriptor straight from tifffile; the AICSImage parse is
# deferred to the first actual *read* of a scene (set_scene), off the startup
# path. Anything outside the validated regime returns None and falls back to the
# authoritative aicsimageio descriptor, so a wrong descriptor is never emitted.

# AICSImage always presents OME-TIFF data as 5-D TCZYX (singleton-padding absent
# axes), so the tifffile-derived descriptor must match that exactly to stay
# byte-identical to the aicsimageio path it replaces.
_CANONICAL_DIMS = "TCZYX"


def _tczyx_shape(series_shape, series_axes) -> Optional[List[int]]:
    """Map a tifffile series (shape + axes string) onto canonical 5-D TCZYX.

    Returns a list of 5 ints, or None if any axis is outside TCZYX (e.g. RGB
    samples ``S``, or an unknown ``Q``/``I``) or the axes/shape lengths disagree
    -- cases where AICSImage's own canonicalization (RGB-sample folding, dim
    expansion) could diverge from a naive mapping, so the caller must fall back.
    """
    axes = str(series_axes or "")
    if not axes or len(axes) != len(series_shape):
        return None
    if any(ax not in _CANONICAL_DIMS for ax in axes):
        return None
    by_axis = {ax: int(n) for ax, n in zip(axes, series_shape)}
    return [by_axis.get(ax, 1) for ax in _CANONICAL_DIMS]


def _ome_scene_ids(ome_xml: Optional[str], n_series: int) -> List[str]:
    """Scene identifiers matching ``AICSImage.scenes`` for an OME-TIFF.

    aicsimageio derives scenes as ``tuple(image.id for image in ome.images)``
    (the OME ``Image`` ``ID`` attribute, in document order), and tifffile's
    series are in that same order. We read the IDs directly from the embedded
    OME-XML with a cheap attribute scan -- NOT an ome-types object build, which
    is the very cost this fix removes. On any mismatch (namespace quirk, missing
    attribute, count disagreement) fall back to the positional ``Image:{i}``
    convention, which is what conformant OME files use anyway.
    """
    if ome_xml:
        ids = re.findall(r'<(?:\w+:)?Image\b[^>]*?\bID="([^"]*)"', ome_xml)
        if len(ids) == n_series:
            return ids
    return [f"Image:{i}" for i in range(n_series)]


# Per-plane OME elements: one <Plane> (timing/stage position) and one <TiffData>
# (IFD->plane map) per plane. These are the O(plane-count) bulk of a big MMStack's
# OME-XML and the sole reason ome-types parsing blows up (40k planes -> ~90 s).
# They carry no catalog-relevant *source* metadata (pixel sizes, channels, dims,
# acquisition annotations all live on Image/Pixels/Channel/StructuredAnnotations),
# so the fast metadata path strips them and parses the tiny remainder.
#
# `(/)?>` captures an optional self-closing slash and the conditional `(?(2)...)`
# then branches on it: a self-closing element (`<Plane .../>`, `<TiffData .../>`)
# matches with NOTHING after the tag, while an open tag consumes up to its OWN
# `</name>` (the \1 backreference). Two correctness/perf properties this buys:
#   * a nested self-closing child (`<TiffData><UUID FileName="f"/></TiffData>`,
#     which some MMStacks emit) cannot end the match at its own `/>` and orphan
#     the parent's `</TiffData>` -- the close form is anchored to the parent name
#     (biopb/biopb#193).
#   * self-closing elements never enter the `.*?</name>` branch, so a file with
#     40k self-closing `<Plane/>` does NOT trigger an O(n^2) scan-to-EOF per plane
#     (an earlier `[^>]*(?:/>|>.*?</\1>)` form took ~87 s on a 10k-plane file;
#     this form is ~0.08 s). `[^>]*?` keeps the attribute scan inside the open tag.
_STRIP_PER_PLANE = re.compile(
    r"<(?:\w+:)?(Plane|TiffData)\b[^>]*?(/)?>(?(2)|.*?</(?:\w+:)?\1>)",
    re.DOTALL,
)


def _fast_ome_metadata(ome_xml: str) -> Optional[dict]:
    """Build the OME metadata dict cheaply by stripping per-plane elements first.

    Parses the *reduced* OME-XML (per-plane ``<Plane>``/``<TiffData>`` removed)
    with the real ome-types parser, so the result is structurally identical to
    ``ome_metadata.model_dump(mode="json")`` EXCEPT that ``planes`` and
    ``tiff_data_blocks`` come back empty -- the deliberate accuracy trade for
    making registration O(structure) instead of O(plane-count) (biopb/biopb#168).
    Returns ``None`` on any failure so the caller falls back to the authoritative
    aicsimageio metadata.
    """
    try:
        from ome_types import from_xml

        reduced = _STRIP_PER_PLANE.sub("", ome_xml)
        ome = from_xml(reduced)
        if hasattr(ome, "model_dump"):
            return ome.model_dump(mode="json")
        if hasattr(ome, "dict"):
            return ome.dict(by_alias=False, exclude_none=False)
        return None
    except Exception:
        logger.debug("fast OME metadata parse failed", exc_info=True)
        return None


# Generic raster/video formats that aicsimageio/bioformats technically read but
# are almost never microscopy *tensor* sources in practice (screenshots, icons,
# UI assets, thumbnails, movies). Claiming these during recursive directory
# discovery floods the catalog with junk (biopb/biopb#40), so they are claimed
# only when explicitly opted in via the ``claim_generic_images`` server config
# flag (see :func:`set_claim_generic_images`). An explicitly configured
# ``type = "aics"`` source bypasses claim() entirely and is unaffected.
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


# =============================================================================
# Persistent aszarr-store pool (fast per-plane reads for tifffile-backed sources)
# =============================================================================
#
# aicsimageio's TiffReader fetches every chunk via _get_image_data, which
# *re-opens the file and rebuilds a tifffile aszarr store on every read* --
# re-parsing the full OME-XML and a per-page dask graph each time. For a
# many-plane OME-TIFF that is ~1-2 s of fixed overhead per single plane,
# independent of bytes requested, so scrubbing a large T axis is dominated by
# repeated OME-XML parsing rather than IO.
#
# The scene-level adapter instead opens the aszarr store *once* (see
# _build_persistent_dask) and slices a persistent ``da.from_zarr`` view, taking
# a single-plane read from ~1.4 s to a few ms. The trade-off is a long-lived
# file handle. Because tifffile's handle is a single seek/read cursor and is
# NOT thread-safe, every read goes through the adapter's ``_io_lock`` AND
# computes with the synchronous scheduler, so the handle is only ever touched
# by one thread at a time (the lock serializes separate calls; ``synchronous``
# serializes pages within a call).
#
# Handles are bounded by a TTL: a daemon reaper closes any store idle longer
# than ``_STORE_TTL_SECONDS`` (interactive scrubbing keeps a stack hot; moving
# on releases it). The reaper closes a store only via a *non-blocking* acquire
# of the owner's ``_io_lock``, so it never pulls a handle out from under an
# in-flight read. FD-exhaustion (or any open failure) degrades gracefully to
# the aicsimageio read path, so the peak handle count never crashes the server.
_STORE_TTL_SECONDS = float(os.environ.get("BIOPB_TIFF_STORE_TTL", "300"))
_open_store_adapters: "weakref.WeakSet" = weakref.WeakSet()
_open_store_lock = threading.Lock()
_reaper_started = False


def _register_store_adapter(adapter: "_AicsImageIoAdapterBase") -> None:
    """Track an adapter holding an open persistent store; start the reaper."""
    global _reaper_started
    if _STORE_TTL_SECONDS <= 0:
        return
    with _open_store_lock:
        _open_store_adapters.add(adapter)
        if not _reaper_started:
            _reaper_started = True
            threading.Thread(
                target=_store_reaper_loop,
                name="tiff-store-reaper",
                daemon=True,
            ).start()


def _store_reaper_loop() -> None:
    """Close persistent stores idle longer than the TTL (best-effort, never
    blocks an in-flight read)."""
    interval = max(1.0, min(_STORE_TTL_SECONDS / 4.0, 30.0))
    while True:
        time.sleep(interval)
        try:
            with _open_store_lock:
                adapters = list(_open_store_adapters)
            now = time.monotonic()
            for adapter in adapters:
                last = getattr(adapter, "_persistent_last_access", now)
                if now - last <= _STORE_TTL_SECONDS:
                    continue
                # Only close when no read is in flight; recheck under the lock.
                if adapter._io_lock.acquire(blocking=False):
                    try:
                        idle = time.monotonic() - adapter._persistent_last_access
                        if idle > _STORE_TTL_SECONDS:
                            adapter._close_persistent_store()
                    finally:
                        adapter._io_lock.release()
        except Exception:  # pragma: no cover - reaper must never die
            logger.debug("tiff-store reaper sweep failed", exc_info=True)


class _AicsImageIoAdapterBase(SourceAdapter, TensorAdapter):
    """Base adapter for aicsimageio-supported vendor formats.

    This base class provides full functionality for reading microscopy data
    through aicsimageio's AICSImage class. Subclasses implement claim() with
    format-specific detection and provide meaningful source_type values.

    Dual-role adapter:
    - Source-level (scene_index=None): manages metadata, lists all scenes
    - Scene-level (scene_index=int): handles data access for one scene

    Multi-scene files expose each scene as a separate tensor within the source.
    Each scene is identified by its scene_id from img.scenes.

    Supports lazy loading via dask arrays.
    Supports remote storage via fsspec (passes fs_kwargs to AICSImage).
    """

    # Class-level source type (override in subclasses)
    SOURCE_TYPE: str = "aics"

    # Multi-tensor source: has multiple scenes
    _single_tensor_source = False

    @classmethod
    def create_from_url(
        cls, url: str, source_id: str, dim_labels: Optional[List[str]] = None
    ) -> "_AicsImageIoAdapterBase":
        """Create source-level adapter instance from URL directly.

        Convenience method for creating adapter without SourceConfig.

        Args:
            url: URL or path to the file
            source_id: Unique identifier for this source
            dim_labels: Optional dimension labels

        Returns:
            Adapter instance (source-level, scene_index=None)
        """
        from aicsimageio import AICSImage

        # Companion.ome files are handled directly by AICSImage when bioformats_jar is available
        # No special handling needed - AICSImage will use BioformatsReader

        img = AICSImage(url)
        return cls(
            img,
            scene_index=None,
            source_id=source_id,
            dim_labels=dim_labels,
            source_url=url,
        )

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "_AicsImageIoAdapterBase":
        """Create source-level adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            Adapter instance (source-level, scene_index=None)
        """
        from aicsimageio import AICSImage

        if source.is_remote:
            # Remote storage: resolve storage_options for fsspec authentication
            storage_options = {}
            if credentials_config:
                profile = credentials_config.get_profile(source.credentials_profile)
                if profile:
                    storage_options = profile.to_storage_options()

            # Note: aicsimageio's OmeZarrReader has a gap where fs_kwargs
            # are not passed to ome-zarr-py. For OME-Zarr, use OmeZarrAdapter instead.
            img = AICSImage(source.url, fs_kwargs=storage_options)
        else:
            # Local filesystem
            img = AICSImage(str(source.url))

        return cls(
            img,
            scene_index=None,  # Source-level adapter
            source_id=source.source_id,
            dim_labels=source.dim_labels,
            source_url=str(source.url),
        )

    def __init__(
        self,
        aics_image: "AICSImage",
        scene_index: Optional[int],
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        io_lock: Optional[threading.Lock] = None,
    ):
        """Initialize AICSImageIO adapter.

        Args:
            aics_image: AICSImage instance
            scene_index: None for source-level, int for scene-level
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (overrides auto-detected dims)
            source_url: Optional source URL
            io_lock: Optional thread lock for IO serialization. Source-level
                     adapters create a new lock if None; scene-level adapters
                     receive the lock from the source-level adapter.
        """
        self._aics_image = aics_image
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
        elif hasattr(aics_image, "source") and hasattr(aics_image.source, "path"):
            self._source_url = str(aics_image.source.path)
        else:
            self._source_url = ""
        self._source_type = self.SOURCE_TYPE

        # Scene-level: cache dask_data after set_scene (thread-safe)
        if scene_index is not None:
            aics_image.set_scene(scene_index)
            self._dask_data = aics_image.dask_data
            self.dim_labels = dim_labels if dim_labels else list(aics_image.dims.order)
            self._cached_descriptors = None  # Scene-level computes on demand
        else:
            # Source-level: no cached dask_data
            self._dask_data = None
            self.dim_labels = dim_labels  # Used as default for all scenes
            self._cached_descriptors = (
                None  # Cached on first list_tensor_descriptors call
            )

        # Persistent aszarr-store fast path (built lazily on first get_data for
        # tifffile-backed local sources; see the module-level pool docs). The
        # descriptor still comes from aicsimageio's dask_data above; only reads
        # use the persistent store.
        self._persistent_dask = None
        self._persistent_store = None
        self._persistent_tiff = None
        self._persistent_attempted = False
        self._persistent_last_access = 0.0

        # Cache of the embedded OME-XML string (biopb/biopb#168), shared by the
        # tifffile descriptor and metadata fast paths so registration opens the
        # file once. ``_raw_ome_xml_probed`` distinguishes "not looked yet" from
        # a probed-but-absent (None) result.
        self._raw_ome_xml = None
        self._raw_ome_xml_probed = False

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from aicsimageio dask array.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape or called on source-level adapter
        """
        if self._dask_data is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))
        with self._io_lock:
            zdask = self._ensure_persistent_dask()
            if zdask is not None:
                self._persistent_last_access = time.monotonic()
                # Single shared file handle: the synchronous scheduler keeps the
                # read on this thread (no per-page thread fan-out), and the
                # io_lock above serializes it against other reads -- so the
                # handle is never touched concurrently.
                return zdask[slices].compute(scheduler="synchronous")
            return self._dask_data[slices].compute()

    def _ensure_persistent_dask(self):
        """Return a persistent aszarr-backed dask view, or None to use the
        aicsimageio read path. Built once; caller must hold ``self._io_lock``."""
        if self._persistent_dask is not None:
            return self._persistent_dask
        if self._persistent_attempted:
            return None
        self._persistent_attempted = True
        try:
            self._persistent_dask = self._build_persistent_dask()
        except Exception as exc:
            # Non-tifffile reader, remote URL, dim mismatch, or FD exhaustion
            # (EMFILE/OSError) -> fall back to aicsimageio for this source.
            logger.debug(
                "persistent aszarr store unavailable for %s: %r; "
                "using aicsimageio read path",
                self._source_url,
                exc,
            )
            self._close_persistent_store()
            self._persistent_dask = None
        if self._persistent_dask is not None:
            self._persistent_last_access = time.monotonic()
            _register_store_adapter(self)
        return self._persistent_dask

    def _build_persistent_dask(self):
        """Open the aszarr store once and return a dask view matching the
        aicsimageio descriptor's shape/dim order, or None if not applicable.

        Raises on open/read errors so the caller can fall back.
        """
        import dask.array as da
        import tifffile

        url = self._source_url or ""
        if "://" in url and not url.startswith("file://"):
            return None  # remote/fsspec source: persistent local handle N/A
        path = url[len("file://") :] if url.startswith("file://") else url
        if not path:
            return None

        series_index = self.scene_index or 0
        tiff = tifffile.TiffFile(path)
        try:
            series = tiff.series[series_index]
            store = series.aszarr(level=0, chunkmode="page")
            axes = series.axes
            z = da.from_zarr(store)

            # Reorder the store's axes into the canonical (aicsimageio) order,
            # then insert singleton axes for the dims tifffile dropped.
            canonical = list(self.dim_labels or [])
            present = [ax for ax in canonical if ax in axes]
            if len(present) != len(axes):
                return None  # store has an axis not in the canonical labels
            z = z.transpose([axes.index(ax) for ax in present])
            for i, ax in enumerate(canonical):
                if ax not in axes:
                    z = da.expand_dims(z, axis=i)

            # Correctness gate: must match the trusted aicsimageio descriptor.
            if (
                tuple(z.shape) != tuple(self._dask_data.shape)
                or z.dtype != self._dask_data.dtype
            ):
                return None
        except Exception:
            tiff.close()
            raise

        self._persistent_tiff = tiff
        self._persistent_store = store
        return z

    def _close_persistent_store(self):
        """Close the persistent store/handle and allow a later reopen.

        Caller holds ``self._io_lock`` (reaper/get_data) or is the GC finalizer
        (no concurrent reads possible). Safe to call repeatedly.
        """
        store = self._persistent_store
        tiff = self._persistent_tiff
        self._persistent_dask = None
        self._persistent_store = None
        self._persistent_tiff = None
        self._persistent_attempted = False  # permit reopen on the next read
        try:
            with _open_store_lock:
                _open_store_adapters.discard(self)
        except Exception:
            pass
        for obj in (store, tiff):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    logger.debug("error closing persistent tiff store", exc_info=True)

    def close(self) -> None:
        """Release the persistent file handle, if any (best-effort teardown).

        Cascades to the lazily-created scene-level adapters, which hold the
        actual handles for a source-level adapter. Scene adapters share this
        adapter's ``_io_lock`` (non-reentrant), so the cascade runs WITHOUT
        holding it.
        """
        with self._io_lock:
            self._close_persistent_store()
        for adapter in list(getattr(self, "_tensor_adapters", {}).values()):
            if adapter is not self:
                try:
                    adapter.close()
                except Exception:
                    logger.debug("error closing scene adapter", exc_info=True)

    def __del__(self):
        # GC backstop: release the handle even without an explicit close().
        try:
            self._close_persistent_store()
        except Exception:
            pass

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this adapter.

        For scene-level adapters (scene_index is set): returns descriptor for that scene.
        For source-level adapters (scene_index=None): returns first scene descriptor.
        """
        if self._dask_data is not None:
            # Scene-level: compute from dask array
            chunks = self._dask_data.chunks
            chunk_shape = [max(c) for c in chunks]
            return TensorDescriptor(
                array_id=self.array_id,
                dim_labels=self.dim_labels if self.dim_labels else [],
                shape=list(self._dask_data.shape),
                chunk_shape=chunk_shape,
                dtype=self._dask_data.dtype.str,
            )
        # Source-level: return first scene descriptor
        return self.list_tensor_descriptors()[0]

    def _local_ome_xml(self) -> Optional[str]:
        """Return the embedded OME-XML string for a local source, or None.

        Cached on the instance (and populated as a side effect of the descriptor
        fast path) so registration opens the file at most once across the
        descriptor and metadata fast paths. Returns None for remote, non-TIFF, or
        non-OME sources.
        """
        if self._raw_ome_xml_probed:
            return self._raw_ome_xml
        self._raw_ome_xml_probed = True
        self._raw_ome_xml = None

        url = self._source_url or ""
        if "://" in url and not url.startswith("file://"):
            return None
        path = url[len("file://") :] if url.startswith("file://") else url
        if not path:
            return None
        try:
            import tifffile

            with tifffile.TiffFile(path) as tiff:
                self._raw_ome_xml = tiff.ome_metadata or None
        except Exception:
            self._raw_ome_xml = None
        return self._raw_ome_xml

    def _tifffile_descriptors(self) -> Optional[List[TensorDescriptor]]:
        """Build per-scene descriptors straight from tifffile (biopb/biopb#168).

        Returns a list of ``TensorDescriptor`` on success, or ``None`` to signal
        the caller to fall back to the authoritative aicsimageio descriptor path.
        Applies only to a local, OME-XML-bearing TIFF with canonical
        (TCZYX-subset) axes and no custom ``dim_labels`` override -- the regime
        validated against AICSImage. Scene IDs match ``AICSImage.scenes`` so the
        catalog array_ids are unchanged, and the AICSImage OME parse is avoided
        entirely here (deferred to the first read's ``set_scene``).
        """
        # An explicit dim_labels override goes through the slower, authoritative
        # aicsimageio path (it owns the non-canonical relabeling).
        if self.dim_labels:
            return None

        url = self._source_url or ""
        if "://" in url and not url.startswith("file://"):
            return None  # remote/fsspec source: no local tifffile handle
        path = url[len("file://") :] if url.startswith("file://") else url
        if not path:
            return None

        import tifffile

        try:
            with tifffile.TiffFile(path) as tiff:
                # Scope to OME-TIFF: the embedded OME-XML object-parse is the cost
                # being removed. A plain TIFF is already cheap in aicsimageio and
                # its dim canonicalization is less predictable, so leave it to
                # aicsimageio.
                ome_xml = tiff.ome_metadata
                # Cache for the metadata fast path so it does not reopen the file.
                self._raw_ome_xml = ome_xml or None
                self._raw_ome_xml_probed = True
                if not ome_xml:
                    return None
                series = tiff.series
                n = len(series)
                if n == 0:
                    return None
                scene_ids = _ome_scene_ids(ome_xml, n)

                descriptors = []
                for i, s in enumerate(series):
                    shape = _tczyx_shape(s.shape, s.axes)
                    if shape is None:
                        # An axis AICSImage would canonicalize differently
                        # (RGB samples, unknown dim): defer the whole source.
                        return None
                    descriptors.append(
                        TensorDescriptor(
                            # Identity policy: array_id = source_id/field; the
                            # field is the aicsimageio scene id (OME Image ID).
                            array_id=f"{self.source_id}/{scene_ids[i]}",
                            dim_labels=list(_CANONICAL_DIMS),
                            shape=shape,
                            chunk_shape=[],  # call get_flight_info for chunk info
                            dtype=s.dtype.str,
                        )
                    )
                return descriptors
        except Exception:
            logger.debug(
                "tifffile descriptor fast path unavailable for %s; "
                "using aicsimageio descriptor path",
                self._source_url,
                exc_info=True,
            )
            return None

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors (scenes) available in this source.

        Optimization: Uses OME metadata for shapes without scene switching.
        Chunk info is NOT populated - clients should call get_flight_info
        for accurate per-scene chunk/metadata details.

        Returns:
            List of TensorDescriptor for all scenes in this source
        """
        # Source-level: use cached descriptors if available
        if self._cached_descriptors is not None:
            return self._cached_descriptors

        # Fast path (biopb/biopb#168): derive shape/dims/dtype/scenes from
        # tifffile, skipping the AICSImage OME-XML object parse that dominates
        # startup. Returns None when not applicable (non-OME, remote, custom
        # dims, exotic axes), in which case we fall through to aicsimageio.
        fast = self._tifffile_descriptors()
        if fast is not None:
            self._cached_descriptors = fast
            return fast

        descriptors = []
        scene_ids = list(self._aics_image.scenes)

        # Try OME metadata first (much faster - no scene switching)
        try:
            ome_meta = self._aics_image.ome_metadata
            if (
                ome_meta is not None
                and hasattr(ome_meta, "images")
                and len(ome_meta.images) == len(scene_ids)
            ):
                # Get dtype from first scene (assumed consistent)
                self._aics_image.set_scene(scene_ids[0])
                dtype = self._aics_image.dask_data.dtype.str

                # Get shapes from OME metadata (no scene switching)
                # OME images are in same order as img.scenes
                for i, im in enumerate(ome_meta.images):
                    px = im.pixels
                    shape = [px.size_t, px.size_c, px.size_z, px.size_y, px.size_x]

                    descriptors.append(
                        TensorDescriptor(
                            # Globally-unique array_id = source_id/field (the
                            # scene id is the within-source field). Identity
                            # policy: list_flights, get_flight_info, and the
                            # chunk_id all carry this one qualified form.
                            array_id=f"{self.source_id}/{scene_ids[i]}",
                            dim_labels=self.dim_labels
                            if self.dim_labels
                            else list(self._aics_image.dims.order),
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
                self._aics_image.set_scene(scene_id)
                dask_data = self._aics_image.dask_data

                descriptors.append(
                    TensorDescriptor(
                        # Globally-unique array_id = source_id/field (identity
                        # policy); the scene id is the within-source field.
                        array_id=f"{self.source_id}/{scene_id}",
                        dim_labels=self.dim_labels
                        if self.dim_labels
                        else list(self._aics_image.dims.order),
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
        re-enumerate ``AICSImage.scenes`` -- which would trigger the OME-XML
        object parse the fast path avoided at registration. The cached
        descriptors are in series/scene order, so the position IS the scene
        index, and aicsimageio's ``set_scene`` takes that int directly. Falls
        back to enumerating scenes when no descriptors are cached (e.g. a read
        without a prior list_tensor_descriptors).
        """
        if self._cached_descriptors is not None:
            for i, d in enumerate(self._cached_descriptors):
                if self._within_source_field(d.array_id) == field:
                    return i
            raise ValueError(f"Unknown scene: {field}")
        scene_ids = list(self._aics_image.scenes)
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
            self._aics_image,
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
        """Return OME metadata as a dict (model_dump of the OME-XML).

        Fast path (biopb/biopb#168): for a local OME-TIFF, parse the OME-XML with
        the per-plane ``<Plane>``/``<TiffData>`` elements stripped, which yields
        the same ome-types structure MINUS the per-plane arrays at a fraction of
        the cost (the per-plane bulk is what makes a big MMStack's parse ~90 s).
        This runs at registration (the metadata-DB sync calls get_metadata), so
        keeping it cheap is what actually moves the OME parse off startup. Any
        non-OME-TIFF source, or any failure, falls back to the authoritative
        aicsimageio ``ome_metadata`` below.

        Returns:
            OME metadata as dict, or empty dict if unavailable.
        """
        ome_xml = self._local_ome_xml()
        if ome_xml:
            fast = _fast_ome_metadata(ome_xml)
            if fast is not None:
                return fast

        try:
            ome_meta = self._aics_image.ome_metadata
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

    def get_physical_scale(self, tensor_id=None):
        """Per-dim physical pixel size + unit from the resident OME model.

        Reads ``ome_metadata.images[scene].pixels.physical_size_{x,y,z}``
        directly (no full ``model_dump``) and maps it onto this tensor's
        ``dim_labels`` by axis label. T/C axes get ``0.0`` / ``""``. Returns
        ``None`` when no positive physical size is known. See
        ``SourceAdapter.get_physical_scale``.
        """
        try:
            ome = self._aics_image.ome_metadata
            if ome is None or not getattr(ome, "images", None):
                return None

            # OME images are in img.scenes order. A scene-level adapter knows
            # its index directly; otherwise resolve the requested tensor_id.
            idx = self.scene_index if self.scene_index is not None else 0
            if tensor_id is not None:
                scene_ids = list(self._aics_image.scenes)
                if tensor_id in scene_ids:
                    idx = scene_ids.index(tensor_id)
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

            labels = self.dim_labels or list(self._aics_image.dims.order)
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


class OmeTiffAdapter(_AicsImageIoAdapterBase):
    """Adapter for OME-TIFF files (embedded OME-XML or companion.ome).

    Handles:
    - .tif/.tiff files with embedded OME-XML metadata
    - .companion.ome files (multi-file OME-TIFF with Bioformats)
    """

    SOURCE_TYPE = "ome-tiff"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim OME-TIFF files with embedded or companion OME-XML."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()

        # Cloud-storage policy (biopb/biopb): OME-TIFF *membership* is derived by
        # reading the OME-XML (a .tif's embedded XML or a .companion.ome), which
        # lists sibling files. Under a cloud root that read is deferred, so the
        # member set would be a guess that can diverge at resolve -- and a single
        # directory can hold several unrelated OME-TIFF sets, so the dir is not
        # the dataset boundary. We therefore do NOT group under cloud: return
        # None so the generic AicsImageIoAdapter claims each .tif as its own
        # single-file source (deferred to unresolved by _claim_is_unresolved),
        # and skip the metadata-only .companion.ome entirely. This must hold at
        # resolve too (the file is resident by then), which is why the gate is
        # ctx.cloud_root, not residency. Multi-file OME-TIFF therefore degrades
        # to N single-file sources under cloud (transcode to OME-Zarr for proper
        # support).
        if ctx.cloud_root:
            return None

        # Case 1: Companion OME file - parse XML to find all TIFF files
        # Requires bioformats_jar dependency
        if name.endswith(".companion.ome"):
            try:
                import bioformats_jar  # noqa: F401  # availability probe
            except ImportError:
                return None

            # Cloud-storage phase 2: a non-resident companion placeholder cannot be
            # read without a recall. A .companion.ome unambiguously marks an
            # OME-TIFF set, so claim it provisionally and defer member enumeration
            # (the multi-file grouping) to first access.
            if not ctx.is_resident():
                state.try_claim_path(ctx.path_str)
                return SourceClaim(
                    source_type=cls.SOURCE_TYPE,
                    primary_path=ctx.path_str,
                    unresolved=True,
                )

            ome_metadata = ctx.read_text()
            if ome_metadata:
                related_files = _extract_files_from_ome_xml(
                    ome_metadata, ctx.parent.path_str, ctx.store
                )
                if related_files:
                    primary_path = related_files[0]
                    state.try_claim_path(ctx.path_str)
                    for f in related_files:
                        state.try_claim_path(f)
                    return SourceClaim(
                        source_type=cls.SOURCE_TYPE,
                        primary_path=primary_path,
                    )
            return None

        # Case 2: TIFF file - check for embedded OME-XML
        # Only works for local files (requires tifffile to extract embedded XML)
        if (
            not ctx.is_remote
            and ctx._path is not None
            and (name.endswith(".tif") or name.endswith(".tiff"))
            # Cloud-storage phase 2: the embedded-OME-XML sniff opens the whole
            # TIFF (a recall on a non-resident placeholder). Skip it when the file
            # is not resident: the generic extension-only AicsImageIoAdapter then
            # claims the .tif as an unresolved image. Any multi-file OME-TIFF
            # grouping degrades to per-file (a documented cloud limitation -- the
            # recommended cloud path is to transcode to OME-Zarr).
            and ctx.is_resident()
        ):
            ome_metadata = _get_ome_metadata_from_tiff(ctx._path, ctx.signature)

            if ome_metadata:
                related_files = _extract_files_from_ome_xml(
                    ome_metadata, ctx.parent.path_str, ctx.store
                )
                if related_files:
                    primary_path = related_files[0]
                    for f in related_files:
                        state.try_claim_path(f)
                    return SourceClaim(
                        source_type=cls.SOURCE_TYPE,
                        primary_path=primary_path,
                    )

        return None


class ZeissAdapter(_AicsImageIoAdapterBase):
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


class LeicaAdapter(_AicsImageIoAdapterBase):
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


class NikonAdapter(_AicsImageIoAdapterBase):
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


class DvAdapter(_AicsImageIoAdapterBase):
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


class OlympusAdapter(_AicsImageIoAdapterBase):
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


class BioformatsAdapter(_AicsImageIoAdapterBase):
    """Bio-Formats fallback for legacy formats with no pure-Python reader.

    Handles proprietary/legacy formats that only the Java Bio-Formats library
    can read -- ZVI (Zeiss AxioVision) being the headline case. Claims a file
    only when ``bioformats_jar`` is importable, so installs without the optional
    ``bioformats`` component skip these files (with a warning) instead of
    failing later at read time.

    Reading goes through AICSImage's BioformatsReader, which is auto-selected
    once bioformats_jar is present. A Java runtime is fetched lazily by
    scyjava/cjdk on first read; it is not a build or system dependency.

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

        # Gate on the Bio-Formats jar (importing it does NOT start a JVM).
        # Without it, skip the file loudly rather than claiming and failing
        # later at read time.
        try:
            import bioformats_jar  # noqa: F401
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


class AicsImageIoAdapter(_AicsImageIoAdapterBase):
    """Fallback adapter for remaining aicsimageio-supported formats.

    Claims microscopy/scientific image files not handled by format-specific
    subclasses. By default the claim set is MICROSCOPY_EXTENSIONS; generic
    raster/video (GENERIC_IMAGE_EXTENSIONS) are claimed during recursive
    discovery only when the ``claim_generic_images`` server config flag is on
    (biopb/biopb#40). Generic file types (txt, csv, cfg, etc.) that bioformats
    technically supports are never claimed.

    Note: Some formats handled by specific adapters:
    - .companion.ome → OmeTiffAdapter
    - .czi, .lsm → ZeissAdapter
    - .lif → LeicaAdapter
    - .nd2 → NikonAdapter
    - .dv → DvAdapter
    - .oif, .oib → OlympusAdapter
    """

    SOURCE_TYPE = "aics"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim aicsimageio-supported files not handled by other adapters."""
        if not ctx.is_file():
            return None

        name = ctx.name.lower()

        # Check against format-specific extensions to avoid double-claiming
        # These are already handled by other subclasses
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

        # Check for remaining aicsimageio extensions. Microscopy/scientific
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
