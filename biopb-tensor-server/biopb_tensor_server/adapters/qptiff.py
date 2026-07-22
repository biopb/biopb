"""QPTIFF adapter for Akoya PhenoImager multiplex whole-slide images.

QPTIFF is the output of Akoya Biosciences' PhenoImager platform (formerly
PerkinElmer Vectra/Polaris/Mantra) -- a pyramidal, multi-channel BigTIFF used for
multiplex-IF / whole-slide imaging. It is *almost* an OME-TIFF, except
channel/marker metadata lives in a PerkinElmer/Akoya XML block in the
``ImageDescription`` tag rather than OME-XML, so ``OmeTiffAdapter`` declines it.

This adapter claims by the ``.qptiff`` extension only. A QPTIFF is sometimes
saved with a plain ``.tif``/``.tiff`` extension, but recognizing that requires
opening the file to sniff the vendor XML on the claim path -- a per-rescan read
that is unsafe under cloud/synced folders and wasteful without a cached result
(biopb/biopb#135). Until claim-time sniffs are cached, a ``.tif``-named QPTIFF
falls through to the generic bioio adapter (which reads it, only without the
native pyramid); rename it to ``.qptiff``, or set an explicit ``type: qptiff``
source, to get the pyramid-preserving path.

Reader: ``tifffile`` directly -- NOT Bio-Formats. Bio-Formats is known to expose
only the base resolution of a QPTIFF and drop the prebuilt pyramid levels;
``tifffile`` surfaces the whole pyramid via ``series[0].levels`` and gives
tile-level lazy access per level via ``series[0].aszarr(level=N)``. Preserving
those native levels is the whole point of this adapter (biopb/biopb#135), so this
is a **native-pyramid** adapter -- only the second after ``OmeZarrAdapter``:

- ``get_native_pyramid_levels()`` advertises one ``precompute`` level per on-disk
  resolution (level 0 = full res).
- ``get_read_plan()`` routes a ``precompute`` + ``scale_hint`` request to the
  matching level's ``aszarr`` store; each level's chunks are encoded with
  ``array_id = source_id/{level}`` so ``DoGet`` dispatches back through
  ``get_level_adapter`` (the same mechanism OME-Zarr uses).

v1 exposes only the baseline pyramidal multichannel image as one tensor
(``c,y,x``); the auxiliary Thumbnail/Overview/Label series are surfaced in
``get_metadata()`` but not as separate tensors (biopb/biopb#135 open question).
"""

import logging
import threading
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import PyramidLevel, SliceHint, TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._scale import MICRON, scale_by_label
from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.core.base import TensorAdapter, TensorReadPlan
from biopb_tensor_server.core.chunk import (
    content_version_from_path,
    normalized_scale_hint,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.downsample import normalize_reduction_method

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

QPTIFF_EXTENSIONS = (".qptiff",)

# The vendor XML block in a QPTIFF's ImageDescription is rooted at
# <PerkinElmer-QPI-ImageDescription>; this substring gates channel/marker-name
# extraction in get_metadata() (the file is already open there, so this is a
# read of in-hand bytes -- not a claim-time recall).
_QPI_XML_MARKER = "PerkinElmer-QPI"

# TIFF ResolutionUnit code -> micrometres per unit, for physical-scale conversion.
_RESUNIT_TO_UM = {2: 25400.0, 3: 10000.0}  # 2 = inch, 3 = centimetre


def _default_dim_labels(ndim: int) -> List[str]:
    """Assign canonical axis labels for a QPTIFF baseline series by rank.

    QPTIFF is 2-D multichannel whole-slide imaging: the leading axis is channels
    and the last two are Y/X. tifffile tags a bare leading plane-axis as ``Q``
    (not ``C``), so the OME axis mapping does not apply -- we label by rank here.
    """
    if ndim == 2:
        return ["y", "x"]
    if ndim == 3:
        return ["c", "y", "x"]
    if ndim == 4:
        return ["c", "z", "y", "x"]
    return [f"dim{i}" for i in range(ndim - 2)] + ["y", "x"]


class QptiffAdapter(TensorAdapter):
    """Adapter for Akoya PhenoImager QPTIFF (pyramidal multiplex BigTIFF).

    Single tensor (the baseline pyramidal multichannel image) served straight from
    ``tifffile`` with its native on-disk pyramid advertised as ``precompute``
    levels.
    """

    SOURCE_TYPE = "qptiff"

    # ---- claim --------------------------------------------------------------

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a QPTIFF by the ``.qptiff`` extension -- suffix only.

        Recognizing a QPTIFF saved with a plain ``.tif``/``.tiff`` extension would
        mean opening the file to sniff the PerkinElmer/Akoya vendor XML on every
        rescan -- a read that is unsafe under cloud/synced folders and wasteful
        without a cached result, so that path is deliberately disabled
        (biopb/biopb#135). Extension matching is recall-free, so a
        cloud/synced-folder placeholder is not recalled here. A ``.tif``-named
        QPTIFF therefore falls through to the generic bioio adapter; use an
        explicit ``type: qptiff`` source (or rename to ``.qptiff``) to force the
        native-pyramid path.
        """
        if not ctx.is_file():
            return None

        if ctx.name.lower().endswith(QPTIFF_EXTENSIONS):
            state.try_claim_path(ctx.path_str)
            return SourceClaim(
                source_type=cls.SOURCE_TYPE,
                primary_path=ctx.path_str,
                is_remote=ctx.is_remote,
            )

        return None

    # ---- construction -------------------------------------------------------

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "QptiffAdapter":
        """Create a source-level adapter (the tifffile handle opens lazily)."""
        return cls(str(source.url), source.source_id, dim_labels=source.dim_labels)

    def __init__(
        self,
        url: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
        io_lock: Optional[threading.RLock] = None,
    ):
        self.source_id = source_id
        self._url = url or ""
        self._source_url = url or ""
        # Cheap content_version from the file's stat signature (#178): O(1),
        # folded into minted chunk_ids so a re-saved file gets a fresh cache
        # namespace. None (unresolved / non-file url) leaves the source unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = self.SOURCE_TYPE
        # One lock serialises the open/cache and metadata paths over the single
        # tifffile handle; reads run lock-free (see _read_level). RLock keeps it
        # safe should any of those paths ever acquire it while already held.
        self._io_lock = io_lock if io_lock is not None else threading.RLock()

        self._dim_labels_override = list(dim_labels) if dim_labels else None
        self.dim_labels: Optional[List[str]] = None

        self._tiff = None
        self._series = None
        self._level_stores: dict = {}  # level -> (zarr_array, store)
        self._level_adapters: dict = {}  # level -> ZarrAdapter (native-level backend)
        self._cached_descriptor: Optional[TensorDescriptor] = None

    # ---- tifffile handle / level stores ------------------------------------

    def _local_path(self) -> str:
        url = self._url
        return url[len("file://") :] if url.startswith("file://") else url

    def _open(self):
        """Open the tifffile handle + baseline series once (caller holds the lock)."""
        if self._series is None:
            import tifffile

            self._tiff = tifffile.TiffFile(self._local_path())
            self._series = self._tiff.series[0]
        return self._series

    def _level_store(self, level: int):
        """Open (and cache) the ``aszarr`` store for one pyramid level as an array.

        Default chunkmode, so the zarr chunks are the QPTIFF's native tile grid --
        the access granularity we advertise as ``chunk_shape``.
        """
        with self._io_lock:
            cached = self._level_stores.get(level)
            if cached is not None:
                return cached
            import zarr

            series = self._open()
            store = series.aszarr(level=level)
            za = zarr.open(store, mode="r")
            self._level_stores[level] = (za, store)
            return za, store

    def _n_levels(self) -> int:
        with self._io_lock:
            return len(self._open().levels)

    def _level_shape(self, level: int) -> Tuple[int, ...]:
        with self._io_lock:
            return tuple(int(x) for x in self._open().levels[level].shape)

    def _read_level(self, level: int, bounds: ChunkBounds) -> np.ndarray:
        slices = self._bounds_to_slices(bounds)
        # _level_store takes the lock only for the lazy open + cache; the read
        # itself runs WITHOUT our lock so parallel do_get chunk reads decode
        # concurrently. tifffile already makes this safe: its aszarr store
        # serializes the raw seek+read on one shared handle lock (fh.lock, the
        # same RLock across all our per-level stores), and the tile decode
        # (imagecodecs: LZW for Akoya component data, JPEG for RGB overviews, ...)
        # is per-tile into a fresh buffer, so concurrent reads cannot race. Copy
        # out so the result is independent of the store.
        za, _ = self._level_store(level)
        return np.asarray(za[slices])

    def close(self) -> None:
        with self._io_lock:
            self._close_handles()

    def _close_handles(self) -> None:
        """Close the tifffile handle + per-level stores; allow a later reopen.

        Caller holds ``self._io_lock`` (the explicit ``close()``) or is the GC
        finalizer (no references left, so no read can be in flight -- no lock
        needed). Nulls the instance refs *before* closing so a concurrent reopen,
        were one possible, sees a clean slate; reads them via ``getattr`` so a
        finalizer running after a half-finished ``__init__`` can't raise. Safe to
        call repeatedly.
        """
        stores = getattr(self, "_level_stores", None) or {}
        tiff = getattr(self, "_tiff", None)
        self._level_stores = {}
        self._level_adapters = {}
        self._tiff = None
        self._series = None
        self._cached_descriptor = None
        for _za, store in list(stores.values()):
            try:
                store.close()
            except Exception:
                logger.debug("error closing qptiff level store", exc_info=True)
        if tiff is not None:
            try:
                tiff.close()
            except Exception:
                logger.debug("error closing qptiff handle", exc_info=True)

    def __del__(self):
        # GC backstop: release the handle even without an explicit close(), but
        # WITHOUT taking _io_lock -- acquiring a lock in a finalizer can deadlock
        # against a thread that holds it, or touch torn-down globals at interpreter
        # shutdown. By the time GC collects this adapter no references remain, so
        # no read can be in flight and the lock is unnecessary (the OmeTiffAdapter
        # pattern). The registry's unregister/shutdown path calls close() for the
        # locked, in-flight-safe release.
        try:
            self._close_handles()
        except Exception:
            pass

    # ---- descriptors --------------------------------------------------------

    def get_tensor_descriptor(self) -> TensorDescriptor:
        if self._cached_descriptor is not None:
            return self._cached_descriptor
        za, _ = self._level_store(0)
        shape = self._level_shape(0)
        labels = self._dim_labels_override or _default_dim_labels(len(shape))
        self.dim_labels = labels
        self._cached_descriptor = TensorDescriptor(
            array_id=self.array_id,
            dim_labels=labels,
            shape=list(shape),
            chunk_shape=list(za.chunks),  # native tile grid
            dtype=za.dtype.str,
        )
        return self._cached_descriptor

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    # ---- reads --------------------------------------------------------------

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read a sub-region of the baseline (full-resolution) image."""
        super().get_data(bounds)  # validate against the base descriptor
        return self._read_level(0, bounds)

    # ---- native pyramid -----------------------------------------------------

    def _scale_for(
        self, base_shape: Tuple[int, ...], level_shape: Tuple[int, ...]
    ) -> List[int]:
        """Per-axis integer downsample factor of a level relative to level 0."""
        return [
            max(1, round(b / s)) for b, s in zip(base_shape, level_shape, strict=True)
        ]

    def has_native_pyramid(self) -> bool:
        try:
            return self._n_levels() >= 2
        except Exception:
            logger.debug("qptiff: level enumeration failed", exc_info=True)
            return False

    def get_native_pyramid_levels(self) -> Optional[List[PyramidLevel]]:
        """One ``precompute`` level per on-disk resolution (level 0 = full res).

        Each level's ``scale_hint`` is its integer downsample factor vs level 0 --
        the exact value ``get_read_plan`` matches on -- so an advertised level
        round-trips to its ``aszarr`` store. Returns ``None`` (-> computed pyramid)
        for a single-level file.
        """
        if not self.has_native_pyramid():
            return None
        base = self._level_shape(0)
        levels: List[PyramidLevel] = []
        for i in range(self._n_levels()):
            lshape = self._level_shape(i)
            levels.append(
                PyramidLevel(
                    scale_hint=self._scale_for(base, lshape),
                    reduction_method="precompute",
                    shape=list(lshape),
                    native=True,
                )
            )
        return levels or None

    def get_read_plan(self, request_desc: TensorDescriptor) -> TensorReadPlan:
        """Route a ``precompute`` + ``scale_hint`` read to the matching level store.

        Other reduction methods (and full-res reads) fall through to the base
        uniform-grid planner, which reads level 0 and downsamples on the fly.
        """
        base_desc = self.get_tensor_descriptor()
        base_shape = tuple(int(d) for d in base_desc.shape)
        scale_hint = normalized_scale_hint(base_shape, request_desc.scale_hint)
        reduction_method = normalize_reduction_method(request_desc.reduction_method)
        slice_hint = (
            request_desc.slice_hint if request_desc.HasField("slice_hint") else None
        )

        if reduction_method == "precompute" and scale_hint is not None:
            level = self._find_level_for_scale(scale_hint)
            if level is None:
                raise ValueError(
                    f"No precomputed level matching scale_hint {tuple(scale_hint)}."
                )
            level_scale = self._scale_for(base_shape, self._level_shape(level))
            level_slice = self._convert_slice_to_level(slice_hint, level_scale)
            return self._plan_from_precomputed(level, level_slice)

        return super().get_read_plan(request_desc)

    def _find_level_for_scale(self, scale_hint: Tuple[int, ...]) -> Optional[int]:
        base = self._level_shape(0)
        target = tuple(scale_hint)
        for i in range(self._n_levels()):
            if tuple(self._scale_for(base, self._level_shape(i))) == target:
                return i
        return None

    def _convert_slice_to_level(
        self, slice_hint: Optional[SliceHint], level_scale: List[int]
    ) -> Optional[SliceHint]:
        if slice_hint is None:
            return None
        level_start = [
            s // sc for s, sc in zip(slice_hint.start, level_scale, strict=True)
        ]
        level_stop = [
            s // sc for s, sc in zip(slice_hint.stop, level_scale, strict=True)
        ]
        return SliceHint(start=level_start, stop=level_stop)

    def _plan_from_precomputed(
        self, level: int, level_slice: Optional[SliceHint]
    ) -> TensorReadPlan:
        """Build a read plan whose chunks target one native level's store.

        The level adapter's descriptor carries ``array_id = source_id/{level}``, so
        the base planner encodes that into every chunk_id and ``DoGet`` dispatches
        back through ``get_level_adapter``. The returned descriptor's ``array_id``
        is reset to the base so the client still sees one tensor.
        """
        level_adapter = self.get_level_adapter(str(level))
        level_desc = level_adapter.get_tensor_descriptor()
        request = TensorDescriptor(
            array_id=level_desc.array_id,
            dim_labels=level_desc.dim_labels,
            shape=list(level_desc.shape),
            chunk_shape=list(level_desc.chunk_shape),
            dtype=level_desc.dtype,
        )
        if level_slice is not None:
            request.slice_hint.start[:] = level_slice.start
            request.slice_hint.stop[:] = level_slice.stop
        read_plan = level_adapter.get_read_plan(request)
        read_plan.descriptor.array_id = self.array_id
        return read_plan

    def get_level_adapter(self, path: str) -> ZarrAdapter:
        """Full backend adapter for a native level, keyed by its integer index.

        Reached by ``DoGet`` for ``precompute`` chunks (``array_id`` suffix
        ``/{level}``) via the server's duck-typed ``get_level_adapter`` dispatch.

        Each level's ``aszarr`` store is already a real ``zarr`` array, so -- like
        ``OmeZarrAdapter`` -- the level adapter is a bare ``ZarrAdapter`` over it
        with ``source_id`` inherited and ``_tensor_name = str(level)``. The base
        ``array_id`` property then yields ``source_id/{level}``; nothing hardcodes
        the identifier. This keeps the level adapter a genuine ``TensorAdapter``
        -- which is itself a ``SourceAdapter`` -- so any caller that treats it as a full
        source -- metadata-DB sync, source-level ops -- finds the attributes it
        expects. All levels share the parent's one open ``tifffile`` handle (the
        ``aszarr`` stores reference it), and ``ZarrAdapter`` holds no handle of its
        own, so the parent's ``close()`` remains the single owner of teardown.
        """
        level = int(path)
        cached = self._level_adapters.get(level)
        if cached is not None:
            return cached
        za, _ = self._level_store(level)
        level_adapter = ZarrAdapter(
            za,
            source_id=self.source_id,
            dim_labels=list(self.get_tensor_descriptor().dim_labels),
        )
        level_adapter._tensor_name = str(level)
        # Point provenance at the real file + this format, not ZarrAdapter's
        # synthetic store repr / "zarr" default (the aszarr store has no path).
        level_adapter._source_url = self._source_url
        level_adapter._source_type = self.SOURCE_TYPE
        self._level_adapters[level] = level_adapter
        return level_adapter

    # ---- metadata / physical scale -----------------------------------------

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dim pixel size (µm) + unit from the TIFF resolution tags.

        QPTIFF stores X/Y pixel density in the standard ``XResolution``/
        ``YResolution`` rationals with a ``ResolutionUnit`` (usually centimetre);
        pixel size = unit / density, converted to micrometres. Returns ``None``
        when no usable resolution is present (e.g. ResolutionUnit "none").
        """
        try:
            with self._io_lock:
                self._open()
                page = self._tiff.pages[0]

            def _density(tag_name):
                tag = page.tags.get(tag_name)
                if tag is None:
                    return None
                val = tag.value
                if isinstance(val, tuple) and len(val) == 2 and val[0]:
                    return val[1] / val[0]  # denom/num = units-per-pixel-density^-1
                if val:
                    return 1.0 / float(val)
                return None

            ru_tag = page.tags.get("ResolutionUnit")
            ru = int(ru_tag.value) if ru_tag is not None else 1
            um_per_unit = _RESUNIT_TO_UM.get(ru)
            if um_per_unit is None:
                return None

            labels = self.get_tensor_descriptor().dim_labels
            sizes = {
                axis: d * um_per_unit
                for axis, d in (
                    ("x", _density("XResolution")),
                    ("y", _density("YResolution")),
                )
                if d and d > 0
            }
            return scale_by_label(labels, sizes, MICRON)
        except Exception:
            logger.debug("qptiff: physical scale unavailable", exc_info=True)
            return None

    def get_metadata(self) -> dict:
        """Marker/channel names + the raw vendor XML, best-effort and JSON-safe.

        Channel markers are read from the per-channel level-0 pages' vendor XML.
        Auxiliary series (thumbnail/overview/label) are listed by name only -- v1
        does not expose them as tensors (biopb/biopb#135).
        """
        meta: dict = {"format": "qptiff"}
        try:
            with self._io_lock:
                self._open()
                base = self.get_tensor_descriptor()
                labels = list(base.dim_labels)
                n_channels = int(base.shape[labels.index("c")]) if "c" in labels else 1
                # One entry per channel, positionally (None where a page has no
                # vendor name). Do NOT drop the gaps: collapsing them shortens the
                # list and misaligns it with the channel axis, so a consumer would
                # attribute names to the wrong channels.
                names = [
                    self._marker_name(pg.description or "")
                    for pg in self._tiff.pages[:n_channels]
                ]
                if any(names):
                    meta["channels"] = names
                # Full page-0 vendor XML -- not truncated. It is fetched only on a
                # metadata request (never in list_flights) and a hard byte cap
                # could sever a multi-KB Akoya block mid-element.
                d0 = self._tiff.pages[0].description or ""
                if d0:
                    meta["image_description"] = d0
                aux = [
                    str(s.name)
                    for s in self._tiff.series[1:]
                    if getattr(s, "name", None)
                ]
                if aux:
                    meta["auxiliary_series"] = aux
        except Exception:
            logger.debug("qptiff: metadata parse failed", exc_info=True)
        return meta

    @staticmethod
    def _marker_name(desc: str) -> Optional[str]:
        """Pull a channel/marker name from a page's PerkinElmer XML, or None."""
        if _QPI_XML_MARKER not in desc:
            return None
        try:
            root = ET.fromstring(desc)
        except ET.ParseError:
            return None
        for tag in ("Name", "Biomarker"):
            el = root.find(f".//{tag}")
            if el is not None and el.text and el.text.strip():
                return el.text.strip()
        return None
