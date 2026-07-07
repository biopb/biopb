"""Pure-tifffile OME-TIFF adapter.

OME-TIFF is read entirely through tifffile -- descriptors, metadata, and physical
scale come from the embedded OME-XML, and pixels from a persistent ``aszarr``
store. There is **no aicsimageio dependency**: this adapter and its OME-XML
helpers stand on their own (biopb/biopb#168, #213). Canonical ``TCZYX`` and
interleaved RGB(A) (a trailing ``S`` samples axis) are both native.

What this adapter deliberately does NOT handle (there is no aicsimageio fallback):

- **Remote OME-TIFF** -- ``claim`` declines a remote URL, so the generic
  ``AicsImageIoAdapter`` (which claims ``.tif``) picks it up via aicsimageio.
- **``.companion.ome``** (multi-file OME-TIFF with a separate companion metadata
  file, historically read via bioformats) -- no longer claimed at all.
- **Truly non-OME axes** (``Q``/``I``) -- ``_ome_axes_shape`` returns ``None`` and
  the source is declined (these do not occur in valid OME-TIFF).

Chunk ID format: array_id + bounds encoding (start, stop coordinates). Relies on
the OS page cache for raw-data caching.
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
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.discovery import ClaimContext, SourceClaim

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.base import BackendAdapter
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState
    from biopb_tensor_server.remote import RemoteStore


# =============================================================================
# OME-XML metadata helpers
# =============================================================================


def _get_namespace(root) -> dict:
    """Extract namespace from root element tag.

    Returns a dict with the OME schema namespace mapping.
    """
    tag = root.tag
    if tag.startswith("{"):
        namespace = tag.split("}")[0].strip("{")
        return {"ome": namespace}
    return {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _extract_files_from_ome_xml(
    ome_metadata: str,
    source_dir: "Path | str",
    store: Optional["RemoteStore"] = None,
) -> "Optional[List[Path] | List[str]]":
    """Extract the ordered TIFF file list from OME-XML ``TiffData`` elements.

    Files are returned in order with the first TiffData's file as master. Returns
    ``None`` if parsing fails or no referenced file exists.
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
# whether it carries OME-XML -- the dominant cost of the post-#63 claim phase
# (~100 ms / 64 tiffs on a real tree). The result is a pure function of the file's
# bytes, so it is cached keyed on the state walk's content-identity signature
# (st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns): any byte change bumps the
# signature, so a hit provably means identical content. A cached value of ``None``
# ("no OME-XML") is meaningful and is stored too, so membership -- not truthiness --
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

    When ``signature`` (the discovery walk's content-identity signature) is given,
    the probe result is memoized on ``(path, signature)`` so an unchanged file is
    not reopened on the next rescan. When ``None`` the probe runs uncached.
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


# OME dimension order is always a permutation of XYZCT (plus an optional samples
# axis S for RGB), so the canonical descriptor is 5-D TCZYX, singleton-padding
# absent axes.
_CANONICAL_DIMS = "TCZYX"


def _tczyx_shape(series_shape, series_axes) -> Optional[List[int]]:
    """Map a tifffile series (shape + axes string) onto canonical 5-D TCZYX.

    Returns a list of 5 ints, or None if any axis is outside TCZYX (e.g. RGB
    samples ``S``, or an unknown ``Q``/``I``) or the axes/shape lengths disagree.
    """
    axes = str(series_axes or "")
    if not axes or len(axes) != len(series_shape):
        return None
    if any(ax not in _CANONICAL_DIMS for ax in axes):
        return None
    by_axis = {ax: int(n) for ax, n in zip(axes, series_shape)}
    return [by_axis.get(ax, 1) for ax in _CANONICAL_DIMS]


def _ome_axes_shape(series_shape, series_axes) -> Optional[Tuple[List[str], List[int]]]:
    """Map a tifffile OME series onto (dim_labels, shape), or None to decline.

    Canonical series map to 5-D ``TCZYX``. A series carrying an interleaved
    *samples* axis ``S`` (photometric-RGB/RGBA OME-TIFF) maps to 6-D ``TCZYXS``,
    with ``S`` trailing -- the layout the webapp renderer expects
    (``extract_yx_slice`` keys on a trailing S of width 3/4). Returns ``None`` for
    a truly non-OME axis (``Q``/``I``) or an axes/shape length mismatch, so the
    caller declines the source (a remote/exotic file then falls to the generic
    aicsimageio adapter).

    OME dimension order is always a permutation of ``XYZCT`` plus optional ``S``,
    so ``TCZYX(S)`` covers every valid OME-TIFF -- there is no aicsimageio fallback.
    """
    canonical = _tczyx_shape(series_shape, series_axes)
    if canonical is not None:
        return list(_CANONICAL_DIMS), canonical
    axes = str(series_axes or "")
    if not axes or len(axes) != len(series_shape) or "S" not in axes:
        return None
    if any(ax not in _CANONICAL_DIMS + "S" for ax in axes):
        return None
    by_axis = {ax: int(n) for ax, n in zip(axes, series_shape)}
    dims = _CANONICAL_DIMS + "S"
    return list(dims), [by_axis.get(ax, 1) for ax in dims]


def _ome_scene_ids(ome_xml: Optional[str], n_series: int) -> List[str]:
    """Scene identifiers for an OME-TIFF, matching the OME ``Image`` ``ID`` order.

    Reads the IDs directly from the embedded OME-XML with a cheap attribute scan --
    NOT an ome-types object build. tifffile's series are in the same (document)
    order. On any mismatch (namespace quirk, missing attribute, count disagreement)
    fall back to the positional ``Image:{i}`` convention, which conformant OME
    files use anyway.
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
    Returns ``None`` on any failure.
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


# =============================================================================
# Persistent aszarr-store pool (tifffile read path)
# =============================================================================
#
# The read path opens a source's tifffile ``aszarr`` store once and keeps the
# handle warm across chunk reads. A background reaper closes stores idle longer
# than the TTL so a long-lived server does not pin file descriptors for sources
# no one is reading. Only OME-TIFF scene adapters own a store, so the pool holds
# only those instances.
_STORE_TTL_SECONDS = float(os.environ.get("BIOPB_TIFF_STORE_TTL", "300"))
_open_store_adapters: "weakref.WeakSet" = weakref.WeakSet()
_open_store_lock = threading.Lock()
_reaper_started = False


def _register_store_adapter(adapter: "OmeTiffAdapter") -> None:
    """Track an OME-TIFF adapter holding an open persistent store; start the reaper."""
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


# =============================================================================
# OmeTiffAdapter
# =============================================================================


class OmeTiffAdapter(SourceAdapter, TensorAdapter):
    """Pure-tifffile adapter for OME-TIFF (embedded OME-XML), single or multi-file.

    Dual-role, keyed on ``scene_index``:

    - Source-level (``scene_index=None``): lists scenes from tifffile; builds
      per-scene adapters.
    - Scene-level (``scene_index=int``): serves one scene from a persistent
      ``aszarr`` store, trusting its handed-down tifffile descriptor.

    Multi-file OME-TIFF (siblings referenced from the master's OME-XML) is stitched
    by tifffile transparently; the module docstring lists the cases that are
    intentionally declined (no aicsimageio fallback).
    """

    SOURCE_TYPE = "ome-tiff"

    # Multi-tensor source: one OME-TIFF may expose several scenes as tensors.
    _single_tensor_source = False

    def __init__(
        self,
        url: str,
        source_id: str,
        scene_index: Optional[int] = None,
        tensor_descriptor: Optional[TensorDescriptor] = None,
        dim_labels: Optional[List[str]] = None,
        io_lock: Optional[threading.Lock] = None,
    ):
        """Initialize an OME-TIFF adapter.

        Args:
            url: URL/path to the master OME-TIFF file.
            source_id: Unique identifier for this source.
            scene_index: None for source-level, int for a bound scene.
            tensor_descriptor: The scene's authoritative tifffile descriptor
                (scene-level only); its dim_labels become this adapter's.
            dim_labels: Optional dimension-label override (source-level; a set
                value routes off the canonical tifffile path -- see
                ``_tifffile_descriptors``).
            io_lock: Shared IO lock. Source-level creates one if None; scene-level
                receives the source's lock.
        """
        self.source_id = source_id
        self._source_url = url or ""
        self._source_type = self.SOURCE_TYPE
        self.scene_index = scene_index
        self._io_lock = io_lock if io_lock is not None else threading.Lock()
        self._cached_descriptors = None

        self._tifffile_descriptor = tensor_descriptor
        if tensor_descriptor is not None:
            self.dim_labels = list(tensor_descriptor.dim_labels)
        else:
            self.dim_labels = dim_labels

        # Persistent aszarr-store state (opened lazily on first get_data). The
        # read serves regions straight from the zarr array -- no dask.
        self._persistent_zarr = None
        self._persistent_axes = None
        self._persistent_store = None
        self._persistent_tiff = None
        self._persistent_attempted = False
        self._persistent_last_access = 0.0

        # Cache of the embedded OME-XML string (biopb/biopb#168), shared by the
        # descriptor, metadata, and physical-scale paths so registration opens the
        # file once. ``_raw_ome_xml_probed`` distinguishes "not looked yet" from a
        # probed-but-absent (None) result.
        self._raw_ome_xml = None
        self._raw_ome_xml_probed = False

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[object] = None
    ) -> "OmeTiffAdapter":
        """Create a source-level adapter from a SourceConfig."""
        return cls(str(source.url), source.source_id, dim_labels=source.dim_labels)

    # ---- reads --------------------------------------------------------------

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from this scene's tifffile aszarr store.

        Validates bounds against the tensor descriptor, then serves the slice from
        the persistent store under ``_io_lock`` (a single shared file handle, read
        on this thread via the synchronous scheduler).

        Raises:
            ValueError: bad bounds, source-level adapter, or store unavailable.
        """
        if self.scene_index is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)  # validate bounds against the descriptor
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))
        with self._io_lock:
            opened = self._ensure_store()
            if opened is None:
                raise ValueError(
                    f"OME-TIFF aszarr store unavailable for {self._source_url!r} "
                    f"(scene {self.scene_index})"
                )
            za, axes = opened
            self._persistent_last_access = time.monotonic()
            return self._read_region(za, axes, slices)

    # ---- descriptors --------------------------------------------------------

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Scene-level: the handed-down tifffile descriptor. Source-level: scene 0."""
        if self.scene_index is not None:
            return self._tifffile_descriptor
        return self.list_tensor_descriptors()[0]

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Per-scene descriptors derived from tifffile (cached).

        Returns an empty list when the source is not a tifffile-readable local
        OME-TIFF (remote, custom dim_labels, non-OME, exotic axes) -- ``claim``
        keeps those out, so in practice this always yields the real scenes.
        """
        if self._cached_descriptors is not None:
            return self._cached_descriptors
        descriptors = self._tifffile_descriptors()
        self._cached_descriptors = descriptors if descriptors is not None else []
        return self._cached_descriptors

    def get_tensor_adapter(self, tensor_id: str) -> "BackendAdapter":
        """Build (and cache) the scene adapter for a within-source field.

        The scene adapter is handed the scene's tifffile descriptor, so it never
        re-derives it and reads straight from the aszarr store.
        """
        descriptors = self.list_tensor_descriptors()
        field = self._within_source_field(tensor_id)
        scene_idx = self._scene_index_for_field(field)

        if not hasattr(self, "_tensor_adapters"):
            self._tensor_adapters = {}
        if field in self._tensor_adapters:
            return self._tensor_adapters[field]

        adapter = OmeTiffAdapter(
            self._source_url,
            self.source_id,
            scene_index=scene_idx,
            tensor_descriptor=descriptors[scene_idx],
            io_lock=self._io_lock,
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def _scene_index_for_field(self, field: Optional[str]) -> int:
        """Resolve a within-source scene field to its integer scene index.

        The cached descriptors are in series/scene order, so the position IS the
        scene index (and the aszarr ``series[index]`` the read opens).
        """
        for i, d in enumerate(self.list_tensor_descriptors()):
            if self._within_source_field(d.array_id) == field:
                return i
        raise ValueError(f"Unknown scene: {field}")

    # ---- metadata / physical scale -----------------------------------------

    def get_metadata(self) -> dict:
        """OME metadata dict from the stripped OME-XML (biopb/biopb#168), else {}.

        Parses the OME-XML with per-plane ``<Plane>``/``<TiffData>`` elements
        stripped -- the same ome-types structure MINUS the per-plane arrays at a
        fraction of the cost. Runs at registration (the metadata-DB sync calls
        get_metadata), so keeping it cheap is what moves the OME parse off startup.
        """
        ome_xml = self._local_ome_xml()
        if ome_xml:
            fast = _fast_ome_metadata(ome_xml)
            if fast is not None:
                return fast
        return {}

    def _physical_scale(self):
        """Per-dim physical pixel size + unit from the local OME-XML (or None)."""
        return self._physical_scale_from_ome_xml()

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the persistent file handle and cascade to scene adapters.

        Scene adapters share this adapter's ``_io_lock`` (non-reentrant), so the
        cascade runs WITHOUT holding it.
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

    # ---- OME-XML internals --------------------------------------------------

    def _local_ome_xml(self) -> Optional[str]:
        """Return the embedded OME-XML string for a local source, or None.

        Cached on the instance (and populated as a side effect of the descriptor
        path) so registration opens the file at most once across the descriptor,
        metadata, and physical-scale paths. Returns None for remote or non-OME
        sources.
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

        Returns a list of ``TensorDescriptor`` on success, or ``None`` to decline
        (custom ``dim_labels`` override, remote/non-``file://`` URL, non-OME TIFF,
        zero series, or a non-OME axis). Scene IDs match the OME ``Image`` IDs so
        the catalog array_ids are stable, and only the tiny OME-XML header is read
        (no ome-types object graph). Canonical ``TCZYX`` and interleaved RGB(A)
        (``TCZYXS``) are both mapped natively via ``_ome_axes_shape``.
        """
        # An explicit dim_labels override is not supported on the pure-tifffile
        # path (it owned the non-canonical relabeling in the old aicsimageio path).
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
                ome_xml = tiff.ome_metadata
                # Cache for the metadata path so it does not reopen the file.
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
                    mapped = _ome_axes_shape(s.shape, s.axes)
                    if mapped is None:
                        # A non-OME axis (Q/I): decline the whole source.
                        return None
                    dim_labels, shape = mapped
                    descriptors.append(
                        TensorDescriptor(
                            # Identity policy: array_id = source_id/field; the
                            # field is the OME Image ID (scene id).
                            array_id=f"{self.source_id}/{scene_ids[i]}",
                            dim_labels=dim_labels,
                            shape=shape,
                            chunk_shape=[],  # call get_flight_info for chunk info
                            dtype=s.dtype.str,
                        )
                    )
                return descriptors
        except Exception:
            logger.debug(
                "tifffile descriptor path unavailable for %s",
                self._source_url,
                exc_info=True,
            )
            return None

    def _physical_scale_from_ome_xml(self):
        """Physical scale from the local OME-XML, per-plane elements stripped.

        Namespace-agnostic ElementTree scan (NOT an ome-types object build): find
        the ``<Image>`` at this scene's index in document order, read its
        ``<Pixels>`` ``PhysicalSizeX/Y/Z`` (+ ``...Unit``), and map onto
        ``dim_labels`` by lowercased axis label (T/C/S -> ``0.0`` / ``""``).
        Physical sizes live on ``<Pixels>`` and survive ``_STRIP_PER_PLANE``. A
        missing ``*Unit`` defaults to ``"µm"`` (OME spec default). Returns ``None``
        on any failure or when no positive size is present -- never raises.
        """
        try:
            ome_xml = self._local_ome_xml()
            if not ome_xml:
                return None
            reduced = _STRIP_PER_PLANE.sub("", ome_xml)
            root = ET.fromstring(reduced)

            def _local(tag):
                return str(tag).rsplit("}", 1)[-1]

            images = [el for el in root.iter() if _local(el.tag) == "Image"]
            idx = self.scene_index or 0
            if idx >= len(images):
                return None
            pixels = next((c for c in images[idx] if _local(c.tag) == "Pixels"), None)
            if pixels is None:
                return None

            def _size(axis):
                raw = pixels.get(f"PhysicalSize{axis}")
                if raw is None:
                    return 0.0, ""
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    return 0.0, ""
                if v <= 0:
                    return 0.0, ""
                return v, (pixels.get(f"PhysicalSize{axis}Unit") or "µm")

            by_label = {"x": _size("X"), "y": _size("Y"), "z": _size("Z")}
            scale, unit = [], []
            for lab in self.dim_labels or []:
                v, u = by_label.get(str(lab).lower(), (0.0, ""))
                scale.append(v)
                unit.append(u)
            if not any(scale):
                return None
            return scale, unit
        except Exception:
            return None

    # ---- persistent aszarr store -------------------------------------------

    def _ensure_store(self):
        """Open the aszarr store as a zarr array once (caller holds ``_io_lock``).

        Returns ``(zarr_array, axes_str)`` or None. A pure-tifffile read needs no
        dask -- ``zarr`` slices the store's pages directly for the requested region
        (see ``_read_region``).
        """
        if self._persistent_zarr is not None:
            return self._persistent_zarr, self._persistent_axes
        if self._persistent_attempted:
            return None
        self._persistent_attempted = True
        try:
            opened = self._open_store()
        except Exception as exc:
            # Non-tifffile reader, remote URL, dim mismatch, or FD exhaustion
            # (EMFILE/OSError): leave the store unavailable for this scene.
            logger.debug("aszarr store unavailable for %s: %r", self._source_url, exc)
            self._close_persistent_store()
            opened = None
        if opened is not None:
            self._persistent_zarr, self._persistent_axes = opened
            self._persistent_last_access = time.monotonic()
            _register_store_adapter(self)
            return opened
        return None

    def _open_store(self):
        """Open ``series[scene].aszarr`` as a zarr array; validate vs the descriptor.

        Returns ``(zarr_array, axes_str)`` or None. Raises on open/read errors so
        the caller records the store as absent. Stashes the tifffile handle + store
        on the instance for ``_close_persistent_store``.
        """
        import tifffile
        import zarr

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
            za = zarr.open(store, mode="r")
            axes = str(series.axes)

            # Correctness gate: the store must match this scene's descriptor. Its
            # canonical shape is the store shape mapped onto dim_labels (singletons
            # for absent axes); both derive from the same series.
            by_axis = {ax: int(za.shape[i]) for i, ax in enumerate(axes)}
            canonical = tuple(by_axis.get(ax, 1) for ax in self.dim_labels or [])
            if (
                canonical != tuple(self._tifffile_descriptor.shape)
                or za.dtype.str != self._tifffile_descriptor.dtype
            ):
                return None
        except Exception:
            tiff.close()
            raise

        self._persistent_tiff = tiff
        self._persistent_store = store
        return za, axes

    def _read_region(self, za, axes, slices):
        """Read the requested canonical region straight from the zarr store.

        ``zarr`` reads only the pages overlapping ``store_slices``; the result is
        reordered into canonical ``dim_labels`` order with singleton axes inserted
        for the dims tifffile dropped. No dask.
        """
        dim_labels = self.dim_labels
        # Slice the store in its native axis order (drop the canonical singletons).
        store_slices = tuple(slices[dim_labels.index(ax)] for ax in axes)
        sub = np.asarray(za[store_slices])
        # Reorder present axes into canonical order, then re-insert the singletons.
        present = [ax for ax in dim_labels if ax in axes]
        sub = np.transpose(sub, [axes.index(ax) for ax in present])
        for i, ax in enumerate(dim_labels):
            if ax not in axes:
                sub = np.expand_dims(sub, axis=i)
        return sub

    def _close_persistent_store(self):
        """Close the persistent store/handle and allow a later reopen.

        Caller holds ``self._io_lock`` (reaper/get_data) or is the GC finalizer
        (no concurrent reads possible). Safe to call repeatedly.
        """
        store = getattr(self, "_persistent_store", None)
        tiff = getattr(self, "_persistent_tiff", None)
        self._persistent_zarr = None
        self._persistent_axes = None
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

    # ---- claim --------------------------------------------------------------

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a local OME-TIFF with embedded OME-XML (single or multi-file).

        Declines remote URLs and ``.companion.ome`` (see the module docstring):
        the generic ``AicsImageIoAdapter`` picks up a remote/plain ``.tif``, and
        companion sets are no longer supported.
        """
        if not ctx.is_file():
            return None

        name = ctx.name.lower()

        # Cloud-storage policy (biopb/biopb): OME-TIFF *membership* is derived by
        # reading the OME-XML, which lists sibling files. Under a cloud root that
        # read is deferred, so the member set would be a guess that can diverge at
        # resolve -- and a single directory can hold several unrelated OME-TIFF
        # sets, so the dir is not the dataset boundary. We therefore do NOT group
        # under cloud: return None so the generic AicsImageIoAdapter claims each
        # .tif as its own single-file source. Multi-file OME-TIFF degrades to N
        # single-file sources under cloud (transcode to OME-Zarr for proper
        # support).
        if ctx.cloud_root:
            return None

        # TIFF file: check for embedded OME-XML. Local only (requires tifffile to
        # extract the embedded XML). A multi-file set's siblings are consumed here
        # via the master's OME-XML file list.
        if (
            not ctx.is_remote
            and ctx._path is not None
            and (name.endswith(".tif") or name.endswith(".tiff"))
            # Cloud-storage phase 2: the embedded-OME-XML sniff opens the whole
            # TIFF (a recall on a non-resident placeholder). Skip it when the file
            # is not resident: the generic extension-only AicsImageIoAdapter then
            # claims the .tif as an unresolved image.
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
