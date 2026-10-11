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
  ``get_tensor_adapter`` (the same mechanism OME-Zarr uses).

v1 exposes only the baseline pyramidal multichannel image as one tensor
(``c,y,x``); the auxiliary Thumbnail/Overview/Label series are surfaced in
``registration_record`` but not as separate tensors (biopb/biopb#135 open question).
"""

import logging
import threading
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import PyramidLevel, TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_pool import HandlePool, PooledHandle
from biopb_tensor_server.adapters._handle_reaper import DEFAULT_HANDLE_REAPER_TTL
from biopb_tensor_server.adapters._scale import MICRON, scale_by_label
from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.core.adapter_base import (
    TensorAdapter,
    TensorEntry,
    bounds_to_slices,
    catalog_entry,
    strip_source_prefix,
)
from biopb_tensor_server.core.chunk import (
    content_version_from_path,
    default_transfer_chunk_shape,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.normalize import canonical_axes
from biopb_tensor_server.core.registration import (
    RegistrationRecord,
    metadata_record,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

QPTIFF_EXTENSIONS = (".qptiff",)

# The vendor XML block in a QPTIFF's ImageDescription is rooted at
# <PerkinElmer-QPI-ImageDescription>; this substring gates channel/marker-name
# extraction in registration_record (the file is already open there, so this is a
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


# One pool for QPTIFF handles, keyed by file identity so a rebuilt adapter finds
# its predecessor's. Its open is the expensive kind -- a pyramidal whole-slide
# BigTIFF's IFD table -- so it keeps the long TTL. The cap is tighter than
# OME-TIFF's because one warm handle is more than a parsed directory: the
# ``TiffFile`` plus one live ``aszarr`` store for every pyramid level read.
_handle_pool = HandlePool(DEFAULT_HANDLE_REAPER_TTL, 16, "qptiff-handle-pool")


class _QptiffFile:
    """One open ``TiffFile``, its baseline series and its per-level stores: what
    the pool keeps open for a file."""

    def __init__(self, path: str):
        import tifffile

        self.lock = threading.RLock()
        self.tiff = tifffile.TiffFile(path)
        self.series = self.tiff.series[0]
        self._level_stores: dict = {}  # level -> (zarr_array, store)

    def level_store(self, level: int):
        """Open (and cache) the ``aszarr`` store for one pyramid level as an array.

        Default chunkmode, so the zarr chunks are the QPTIFF's native tile grid --
        the access granularity we advertise as ``chunk_shape``.
        """
        cached = self._level_stores.get(level)
        if cached is not None:
            return cached
        with self.lock:
            cached = self._level_stores.get(level)
            if cached is not None:
                return cached
            import zarr

            store = self.series.aszarr(level=level)
            opened = (zarr.open(store, mode="r"), store)
            self._level_stores[level] = opened
            return opened

    def close(self) -> None:
        for _za, store in list(self._level_stores.values()):
            try:
                store.close()
            except Exception:
                logger.debug("error closing qptiff level store", exc_info=True)
        self._level_stores = {}
        try:
            self.tiff.close()
        except Exception:
            logger.debug("error closing qptiff handle", exc_info=True)


@canonical_axes
class _QptiffLevelAdapter(ZarrAdapter):
    """A native pyramid level's backend, reading under a lease on its parent's
    pooled handle.

    The zarr array this reads through is a view onto the parent's one ``TiffFile``,
    which the pool may close. The array is re-resolved from a lease on every read
    rather than captured once, so a close since the last read replaced it, and
    this adapter may well have outlived that.
    """

    def __init__(self, parent: "QptiffAdapter", level: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent = parent
        self._level = level

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        super(ZarrAdapter, self).get_data(bounds)  # validate against the level
        slices = bounds_to_slices(bounds)
        with self._parent._file() as handle:
            return np.asarray(handle.level_store(self._level)[0][slices])


@canonical_axes
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
        return cls(str(source.url), source.source_id)

    @classmethod
    def create_from_payload(
        cls,
        source: "SourceConfig",
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        credentials_config: Optional[Any] = None,
    ) -> "QptiffAdapter":
        """Rebuild from the row: descriptor, pyramid shapes, scale and metadata are
        answered from it, and the TIFF handle opens on the first read."""
        adapter = cls(str(source.url), source.source_id)
        shape = tuple(int(s) for s in payload["shape"])
        labels = _default_dim_labels(len(shape))
        adapter.dim_labels = labels
        adapter._stored = {
            "payload": payload,
            "level_shapes": [tuple(int(s) for s in lv) for lv in payload["levels"]],
            "scale": payload["scale"],
            "metadata": metadata,
            "descriptor": TensorDescriptor(
                array_id=adapter.array_id,
                dim_labels=labels,
                shape=list(shape),
                chunk_shape=default_transfer_chunk_shape(
                    shape,
                    payload["dtype"],
                    labels,
                    native=tuple(int(c) for c in payload["chunks"]),
                ),
                dtype=payload["dtype"],
            ),
        }
        return adapter

    def catalog_payload(self) -> Optional[Dict[str, Any]]:
        """The baseline image's shape, dtype and tile grid, every pyramid level's
        shape and the physical scale: what the descriptor, the native pyramid and
        the scale hint are made of."""
        if self._stored is not None:
            return self._stored["payload"]
        za, _ = self._level_store(0)
        scale = self._physical_scale()
        return {
            "shape": [int(s) for s in self._level_shape(0)],
            "dtype": za.dtype.str,
            "chunks": [int(c) for c in za.chunks],
            "levels": [
                [int(s) for s in self._level_shape(i)] for i in range(self._n_levels())
            ],
            "scale": None if scale is None else [list(scale[0]), list(scale[1])],
        }

    def __init__(
        self,
        url: str,
        source_id: str,
    ):
        self.source_id = source_id
        self._url = url or ""
        self._source_url = url or ""
        # Cheap content_version from the file's stat signature (#178): O(1),
        # folded into minted chunk_ids so a re-saved file gets a fresh cache
        # namespace. None (unresolved / non-file url) leaves the source unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = self.SOURCE_TYPE
        self.dim_labels: Optional[List[str]] = None

        self._level_adapters: dict = {}  # level -> ZarrAdapter (native-level backend)
        self._cached_descriptor: Optional[TensorDescriptor] = None
        # What a source rebuilt from its row answers without the file (descriptor,
        # level shapes, scale, metadata); None for one parsed from the file.
        self._stored: Optional[Dict[str, Any]] = None

    # ---- tifffile handle / level stores ------------------------------------

    def _local_path(self) -> str:
        url = self._url
        return url[len("file://") :] if url.startswith("file://") else url

    def _pool_key(self):
        return (self._url, self._content_version)

    @contextmanager
    def _file(self):
        """Lease this file's pooled handle for the duration of a block. A leased
        handle is never closed, so reads decode under it without a lock."""
        with _handle_pool.checkout(self._pool_key(), self._open_file) as handle:
            yield handle.value

    def _open_file(self) -> PooledHandle:
        opened = _QptiffFile(self._local_path())
        return PooledHandle(self._pool_key(), opened, opened.close)

    def _level_store(self, level: int):
        with self._file() as handle:
            return handle.level_store(level)

    def _n_levels(self) -> int:
        if self._stored is not None:
            return len(self._stored["level_shapes"])
        with self._file() as handle:
            return len(handle.series.levels)

    def _level_shape(self, level: int) -> Tuple[int, ...]:
        if self._stored is not None:
            return self._stored["level_shapes"][level]
        with self._file() as handle:
            return tuple(int(x) for x in handle.series.levels[level].shape)

    def _read_level(self, level: int, bounds: ChunkBounds) -> np.ndarray:
        slices = bounds_to_slices(bounds)
        # The read runs under a lease and no lock, so parallel do_get chunk reads
        # decode concurrently. tifffile already makes this safe: its aszarr store
        # serializes the raw seek+read on one shared handle lock (fh.lock, the
        # same RLock across all our per-level stores), and the tile decode
        # (imagecodecs: LZW for Akoya component data, JPEG for RGB overviews, ...)
        # is per-tile into a fresh buffer, so concurrent reads cannot race. Copy
        # out so the result is independent of the store.
        with self._file() as handle:
            return np.asarray(handle.level_store(level)[0][slices])

    def close(self) -> None:
        """Release the pooled handle now (at its last lease) rather than waiting
        for the pool's TTL."""
        _handle_pool.drop(self._pool_key())
        self._level_adapters = {}

    # ---- descriptors --------------------------------------------------------

    def _native_descriptor(self) -> TensorDescriptor:
        if self._stored is not None:
            return self._stored["descriptor"]
        if self._cached_descriptor is not None:
            return self._cached_descriptor
        za, _ = self._level_store(0)
        shape = self._level_shape(0)
        labels = _default_dim_labels(len(shape))
        self.dim_labels = labels
        self._cached_descriptor = TensorDescriptor(
            array_id=self.array_id,
            dim_labels=labels,
            shape=list(shape),
            # Seeded by the native tile grid, sized to the transfer target (#809).
            chunk_shape=default_transfer_chunk_shape(
                shape, za.dtype.str, labels, native=za.chunks
            ),
            dtype=za.dtype.str,
        )
        return self._cached_descriptor

    def list_tensors(self) -> List[TensorEntry]:
        return [catalog_entry(self._native_descriptor())]

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

    def _find_level_for_scale(self, scale_hint: Tuple[int, ...]) -> Optional[int]:
        base = self._level_shape(0)
        target = tuple(scale_hint)
        for i in range(self._n_levels()):
            if tuple(self._scale_for(base, self._level_shape(i))) == target:
                return i
        return None

    def _level_downsample_factors(self, level: int) -> List[int]:
        """Downsample factors for ``level`` -- its shape ratio vs level 0.

        The base precompute routing (biopb/biopb#557) uses these to translate a
        base-coordinate slice into the level's grid.
        """
        return self._scale_for(self._level_shape(0), self._level_shape(level))

    def get_tensor_adapter(self, tensor_id: str | None) -> TensorAdapter:
        """A native level for ``<source>/<level>``, else the base behavior."""
        field = strip_source_prefix(self.source_id, tensor_id)
        if field and field.isdigit() and int(field) < self._n_levels():
            return self._level_adapter(int(field))
        return super().get_tensor_adapter(tensor_id)

    def _level_adapter(self, level: int) -> ZarrAdapter:
        """Full backend adapter for a native level, keyed by its integer index.

        Each level's ``aszarr`` store is already a real ``zarr`` array, so -- like
        ``OmeZarrAdapter`` -- the level adapter is a bare ``ZarrAdapter`` over it
        with ``source_id`` inherited and ``_tensor_name = str(level)``. The base
        ``array_id`` property then yields ``source_id/{level}``; nothing hardcodes
        the identifier. All levels share the parent's one open ``tifffile`` handle
        (the ``aszarr`` stores reference it), and ``ZarrAdapter`` holds no handle
        of its own, so the parent's ``close()`` remains the single owner of
        teardown.
        """
        cached = self._level_adapters.get(level)
        if cached is not None:
            return cached
        za, _ = self._level_store(level)
        level_adapter = _QptiffLevelAdapter(
            self,
            level,
            za,
            source_id=self.source_id,
            dim_labels=list(self._native_descriptor().dim_labels),
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
        if self._stored is not None:
            scale = self._stored["scale"]
            return None if scale is None else (list(scale[0]), list(scale[1]))
        try:
            with self._file() as handle:
                page = handle.tiff.pages[0]

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

            labels = self._native_descriptor().dim_labels
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

    def registration_record(
        self, tensors, *, import_rois=True, max_rois_per_tensor=None
    ) -> RegistrationRecord:
        """Marker/channel names + the raw vendor XML, best-effort and JSON-safe.

        Channel markers are read from the per-channel level-0 pages' vendor XML.
        Auxiliary series (thumbnail/overview/label) are listed by name only -- v1
        does not expose them as tensors (biopb/biopb#135).
        """
        if self._stored is not None:
            return metadata_record(dict(self._stored["metadata"]))
        meta: dict = {"format": "qptiff"}
        try:
            with self._file() as handle:
                base = self._native_descriptor()
                labels = list(base.dim_labels)
                n_channels = int(base.shape[labels.index("c")]) if "c" in labels else 1
                # One entry per channel, positionally (None where a page has no
                # vendor name). Do NOT drop the gaps: collapsing them shortens the
                # list and misaligns it with the channel axis, so a consumer would
                # attribute names to the wrong channels.
                names = [
                    self._marker_name(pg.description or "")
                    for pg in handle.tiff.pages[:n_channels]
                ]
                if any(names):
                    meta["channels"] = names
                # Full page-0 vendor XML -- not truncated. It is fetched only on a
                # metadata request (never in list_flights) and a hard byte cap
                # could sever a multi-KB Akoya block mid-element.
                d0 = handle.tiff.pages[0].description or ""
                if d0:
                    meta["image_description"] = d0
                aux = [
                    str(s.name)
                    for s in handle.tiff.series[1:]
                    if getattr(s, "name", None)
                ]
                if aux:
                    meta["auxiliary_series"] = aux
        except Exception:
            logger.debug("qptiff: metadata parse failed", exc_info=True)
        return metadata_record(meta)

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
