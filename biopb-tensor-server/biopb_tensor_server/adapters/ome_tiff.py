"""Pure-tifffile OME-TIFF adapter.

OME-TIFF is read entirely through tifffile -- descriptors, metadata, and physical
scale come from the embedded OME-XML, and pixels from a persistent ``aszarr``
store. There is **no aicsimageio dependency**: this adapter and its OME-XML
helpers stand on their own (biopb/biopb#168, #213). Canonical ``TCZYX`` and
interleaved RGB(A) (a trailing ``S`` samples axis) are both native.

What this adapter deliberately does NOT handle (there is no aicsimageio fallback):

- **Remote OME-TIFF** -- ``claim`` declines a remote URL, so the generic
  ``AicsImageIoAdapter`` (which claims ``.tif``) picks it up via bioio.
- **``.companion.ome``** (multi-file OME-TIFF with a separate companion metadata
  file, historically read via bioformats) -- no longer claimed at all.
- **Truly non-OME axes** (``Q``/``I``) -- ``_ome_axes_shape`` returns ``None`` and
  the source is declined (these do not occur in valid OME-TIFF).

Chunk ID format: array_id + bounds encoding (start, stop coordinates). Relies on
the OS page cache for raw-data caching.
"""

import base64
import io
import logging
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_pool import HandlePool, PooledHandle
from biopb_tensor_server.adapters._handle_reaper import DEFAULT_HANDLE_REAPER_TTL
from biopb_tensor_server.adapters._ome_rois import (
    OME_SET_NAME,
    imported_annotations,
    tensors_by_field,
)
from biopb_tensor_server.adapters._signature_memo import Signature, SignatureMemo
from biopb_tensor_server.adapters.ome_masks import RasterizedMaskAdapter, masks_by_image
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
from biopb_tensor_server.core.errors import TensorNotFound
from biopb_tensor_server.core.labels import label_extent, label_field
from biopb_tensor_server.core.normalize import canonical_axes

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.core.remote import RemoteStore


# =============================================================================
# OME-XML metadata helpers
# =============================================================================


def _tag_name(tag: str) -> str:
    """An element's tag without its ``{namespace}``."""
    return tag.rsplit("}", 1)[-1]


_UUID_FILENAME = re.compile(rb'UUID FileName="([^"]*)"')


def _files_from_ome_xml(xml: bytes) -> Tuple[str, ...]:
    """The distinct ``<UUID FileName=...>`` values of an OME-XML, in document order.

    A literal scan, not an XML parse: a Micro-Manager stack's XML runs to tens of
    MB and repeats a file name per plane, so the parse cost ~25x the scan for the
    same answer. The scan only knows ``UUID FileName="..."``, so the parser decides
    whenever a ``FileName`` token is anything else (another element's attribute,
    single quotes, spaces around ``=``) or a name has an entity.
    """
    found = _UUID_FILENAME.findall(xml)
    names = list(dict.fromkeys(found))
    if len(found) != xml.count(b"FileName") or any(b"&" in n for n in names):
        return _files_from_ome_xml_parsed(xml)
    # Interned: the members of a multi-file set each cache the same names.
    return tuple(sys.intern(n.decode("utf-8", "replace")) for n in names)


def _files_from_ome_xml_parsed(xml: bytes) -> Tuple[str, ...]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ()
    names: Dict[str, None] = {}
    for tiff_data in root.iter():
        if _tag_name(tiff_data.tag) != "TiffData":
            continue
        for child in tiff_data:
            if _tag_name(child.tag) == "UUID" and child.get("FileName"):
                names[sys.intern(child.get("FileName"))] = None
    return tuple(names)


def _existing_files(
    names: Tuple[str, ...],
    source_dir: "Path | str",
    store: Optional["RemoteStore"] = None,
) -> "List[Path] | List[str]":
    """The files *names* refer to that exist beside the master, in order."""
    files = []
    for filename in names:
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
    return files


# The claim's content probe, memoized: the file names an OME-TIFF's OME-XML refers
# to (``()`` for a single-file one), or ``None`` for a TIFF with no OME-XML.
_OME_PROBE_MEMO = SignatureMemo(100_000)

_TIFF_DESCRIPTION_TAG = 270
_TIFF_ASCII = 2
_OME_TAIL = b"OME>"


def _read_ome_xml(path: Path) -> Optional[bytes]:
    """The first IFD's ImageDescription if it is OME-XML, else ``None``.

    Reads the header, the first IFD and the description, and only the last bytes
    of a description that is not OME-XML. Raises for a layout it does not handle.
    """
    with open(path, "rb") as f:
        head = f.read(16)
        order = {b"II": "<", b"MM": ">"}[head[:2]]
        magic = struct.unpack(order + "H", head[2:4])[0]
        if magic == 42:
            (ifd,) = struct.unpack(order + "I", head[4:8])
            count_fmt, count_size, entry_size, offset_fmt, slot = "H", 2, 12, "I", 4
        elif magic == 43:
            (ifd,) = struct.unpack(order + "Q", head[8:16])
            count_fmt, count_size, entry_size, offset_fmt, slot = "Q", 8, 20, "Q", 8
        else:
            raise ValueError("not a TIFF")
        f.seek(ifd)
        (n,) = struct.unpack(order + count_fmt, f.read(count_size))
        table = f.read(n * entry_size)
        for i in range(n):
            entry = table[i * entry_size : (i + 1) * entry_size]
            tag, kind = struct.unpack(order + "HH", entry[:4])
            if tag != _TIFF_DESCRIPTION_TAG:
                continue
            if kind != _TIFF_ASCII:
                raise ValueError("unexpected description type")
            (size,) = struct.unpack(order + offset_fmt, entry[4 : 4 + slot])
            value = entry[4 + slot : 4 + 2 * slot]
            if size <= slot:
                data = value[:size]
            else:
                (offset,) = struct.unpack(order + offset_fmt, value)
                f.seek(offset + max(0, size - 16))
                if not f.read(16).rstrip(b"\x00 \t\r\n").endswith(_OME_TAIL):
                    return None
                f.seek(offset)
                data = f.read(size)
            return data if data.rstrip(b"\x00 \t\r\n").endswith(_OME_TAIL) else None
    return None


def _read_ome_xml_tifffile(path: Path) -> Optional[bytes]:
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tf:
            xml = tf.ome_metadata
    except OSError:
        raise
    except Exception:
        return None
    return xml.encode("utf-8") if xml else None


def _probe_ome_files(path: Path) -> Optional[Tuple[str, ...]]:
    """What the file's OME-XML refers to, or ``None`` without OME-XML.

    Raises ``OSError`` when the file cannot be read, which says nothing about its
    content.
    """
    try:
        xml = _read_ome_xml(path)
    except OSError:
        raise
    except Exception:
        xml = _read_ome_xml_tifffile(path)
    return None if xml is None else _files_from_ome_xml(xml)


def _get_ome_files(
    path: Path, signature: Optional[Signature] = None, *, memoize: bool = True
) -> Optional[Tuple[str, ...]]:
    """What a TIFF's OME-XML refers to (see :func:`_probe_ome_files`), memoized on
    the file's identity so an unchanged file is not reopened on the next rescan.
    An unreadable file reads as having none, and is not memoized."""
    try:
        return _OME_PROBE_MEMO.get(
            path, lambda: _probe_ome_files(path), signature, memoize=memoize
        )
    except OSError:
        return None


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
    by_axis = {ax: int(n) for ax, n in zip(axes, series_shape, strict=True)}
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
    by_axis = {ax: int(n) for ax, n in zip(axes, series_shape, strict=True)}
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


# Some Micro-Manager MMStacks emit a degenerate BinData placeholder -- a BinData
# with no `Length` attribute and no pixel content -- in either the self-closing
# form `<BinData BigEndian="true"/>` or the open-but-empty form
# `<BinData BigEndian="true"></BinData>`. It carries no catalog data, but
# ome-types/pydantic rejects it ("length Field required"), so `from_xml` raises
# and the whole fast path returns None -> `get_metadata` yields `{}` for these
# files (biopb/biopb#199). Dropping the empty placeholder lets `from_xml` succeed
# and produce the real structural dict.
#
# Matched EMPTY forms only. A genuine inline-pixels BinData is always the open
# form WITH content (`<BinData Length="N">...base64...</BinData>`), which neither
# branch matches: the `/>` branch is self-closing-only, and the close branch
# requires `>\s*</BinData>` (a whitespace-only body), so any real content fails
# it. Crucially that branch uses `\s*`, NOT a `.*?</BinData>` scan-to-close -- it
# stops at the first non-whitespace byte, so the match stays O(n) and cannot
# reintroduce the #193 O(n^2) footgun (nor scan across a large inline-pixel blob).
# `[^>]` bounds the attribute scan to the tag; `\b` stops `BinData` matching a
# longer name like `BinDataset`.
_STRIP_EMPTY_BINDATA = re.compile(
    r"<(?:\w+:)?BinData\b[^>]*?(?:/>|>\s*</(?:\w+:)?BinData>)"
)


def _strip_mask_bindata_payloads(ome_xml: str) -> str:
    """Redact inline ``Mask/BinData`` bytes while retaining mask geometry.

    ``Length=0`` keeps the element valid for ome-types if metadata is read
    after registration; removing the element entirely makes ome-types discard
    the containing ROI.  This is called only after the label adapters have
    copied the decoded payloads they need for rasterization.
    """
    root = ET.fromstring(ome_xml)
    for mask in root.iter():
        if mask.tag.rsplit("}", 1)[-1] != "Mask":
            continue
        for child in mask:
            if child.tag.rsplit("}", 1)[-1] == "BinData":
                child.text = None
                child.set("Length", "0")
    return ET.tostring(root, encoding="unicode")


def _b64_encode_mask_bindata(ome: Any) -> None:
    """Base64-encode every ``<Mask>``'s ``bin_data.value`` in place.

    A mask's bitmap is arbitrary binary, and pydantic's ``mode="json"`` dump of
    a ``bytes`` field decodes it as UTF-8 -- which raises on the overwhelming
    majority of real bitmaps (they are not valid UTF-8) and, since there is no
    partial ``model_dump``, would fail the WHOLE metadata parse over one mask,
    silently costing the file every other piece of its OME metadata too. Base64
    is ASCII, so it survives the dump intact; the label rasterizer
    (``adapters/ome_masks.py``) decodes it back off the resulting dict.

    Best-effort: a shape this cannot walk is left alone, and the dump either
    succeeds anyway (no real bitmap) or fails exactly as it would have without
    this call -- never worse.
    """
    for roi in getattr(ome, "rois", None) or ():
        for mask in getattr(getattr(roi, "union", None), "masks", None) or ():
            value = getattr(getattr(mask, "bin_data", None), "value", None)
            if isinstance(value, (bytes, bytearray)):
                mask.bin_data.value = base64.b64encode(bytes(value))


def _fast_ome_metadata(
    ome_xml: str, *, already_reduced: bool = False
) -> Optional[dict]:
    """Build the OME metadata dict cheaply by stripping per-plane elements first.

    Parses the *reduced* OME-XML (per-plane ``<Plane>``/``<TiffData>`` removed)
    with the real ome-types parser, so the result is structurally identical to
    ``ome_metadata.model_dump(mode="json")`` EXCEPT that ``planes`` and
    ``tiff_data_blocks`` come back empty -- the deliberate accuracy trade for
    making registration O(structure) instead of O(plane-count) (biopb/biopb#168)
    -- and that a ``<Mask>``'s ``bin_data.value`` is base64 text rather than raw
    bytes (:func:`_b64_encode_mask_bindata`). Returns ``None`` on any failure.
    """
    try:
        from ome_types import from_xml

        reduced = (
            ome_xml
            if already_reduced
            else _STRIP_EMPTY_BINDATA.sub("", _STRIP_PER_PLANE.sub("", ome_xml))
        )
        ome = from_xml(reduced)
        if hasattr(ome, "model_dump"):
            _b64_encode_mask_bindata(ome)
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
# A scene's tifffile ``aszarr`` store is opened once and kept warm across chunk
# reads in a pool keyed by file identity, not by adapter, so an adapter that is
# dropped and rebuilt finds the store its predecessor opened. The pool closes a
# store idle longer than the TTL (``ServerConfig.handle_reaper_ttl`` is its
# ceiling) and the least recently used beyond the cap. Reopening is the expensive
# case the pool is written for: open is linear in IFD count (~615 ms extrapolated
# at 50k pages), so the TTL is the long default and the cap generous.
_store_pool = HandlePool(DEFAULT_HANDLE_REAPER_TTL, 32, "tiff-store-pool")


def _parallel_read_enabled() -> bool:
    """Whether OME-TIFF chunk reads decode lock-free (biopb/biopb#473).

    Default **off**: readers of one store serialize on its handle lock. Set
    ``BIOPB_OMETIFF_PARALLEL_READ=1`` to serve reads lock-free -- tifffile
    serializes the raw seek+read on the store's own shared handle lock and the tile
    decode is per-tile into a fresh buffer, so concurrent decodes run in parallel
    (the lease keeps the store open for each). Read at call time so a
    process (or a test) can toggle it without reimport; the cost is one dict lookup
    per chunk, negligible against a tile read.
    """
    return os.environ.get("BIOPB_OMETIFF_PARALLEL_READ", "0") == "1"


# =============================================================================
# OmeTiffAdapter
# =============================================================================


_UNSET: Any = object()


@canonical_axes
class OmeTiffAdapter(TensorAdapter):
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

    # Set on an adapter rebuilt from a stored payload (``create_from_payload``): what
    # it was rebuilt from, the physical scale each scene had (``_UNSET`` until
    # then, since ``None`` is a legitimate "no calibration"), and the row's
    # metadata where that is the whole of it (a file with no ROIs).
    _hydrated_payload: Optional[dict] = None
    _seeded_scale: Any = _UNSET
    _hydrated_metadata: Optional[dict] = None
    _seeded_scales: dict = {}

    def __init__(
        self,
        url: str,
        source_id: str,
        scene_index: Optional[int] = None,
        tensor_descriptor: Optional[TensorDescriptor] = None,
    ):
        """Initialize an OME-TIFF adapter.

        Args:
            url: URL/path to the master OME-TIFF file.
            source_id: Unique identifier for this source.
            scene_index: None for source-level, int for a bound scene.
            tensor_descriptor: The scene's authoritative tifffile descriptor
                (scene-level only); its dim_labels become this adapter's.
        """
        self.source_id = source_id
        self._source_url = url or ""
        # Cheap content_version from the master file's stat signature (#178): O(1),
        # folded into minted chunk_ids so a re-saved file gets a fresh cache
        # namespace. None (unresolved / non-file url) leaves the source unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = self.SOURCE_TYPE
        self.scene_index = scene_index
        self._cached_descriptors = None

        self._tifffile_descriptor = tensor_descriptor
        if tensor_descriptor is not None:
            self.dim_labels = list(tensor_descriptor.dim_labels)
        else:
            self.dim_labels = None

        # Cache of the embedded OME-XML string (biopb/biopb#168), shared by the
        # descriptor, metadata, and physical-scale paths so registration opens the
        # file once. ``_raw_ome_xml_probed`` distinguishes "not looked yet" from a
        # probed-but-absent (None) result.
        #
        # The raw string is registration-scope only: it is tens of MB on a
        # per-plane acquisition (one <Plane> + one <TiffData> per T*C*Z), and
        # ``release_registration_cache`` drops it once the catalog owns the
        # metadata (biopb/biopb#783). ``_raw_ome_xml_released`` is the third
        # state -- "there IS XML in the file, we just are not holding it" -- so a
        # later consumer re-reads instead of seeing a false None. Only the
        # plane-stripped ``_reduced_ome_xml`` (hundreds of bytes to a few KB)
        # stays resident; it carries every <Image>/<Pixels> header, which is all
        # the metadata and physical-scale paths read.
        self._raw_ome_xml = None
        self._raw_ome_xml_probed = False
        self._raw_ome_xml_released = False
        self._reduced_ome_xml = None
        self._reduced_ome_xml_probed = False
        # The dict _reduced_ome_xml parses into (biopb/biopb#1059 step 4): a pure
        # function of that already-cached string, so caching it costs nothing in
        # correctness and saves a second ome-types parse when both get_metadata
        # and get_embedded_labels run in the same registration (metadata_db.py
        # calls the former directly; the latter is the registry's one-time call per adapter).
        self._parsed_metadata: Optional[dict] = None
        self._parsed_metadata_probed = False
        # Set only after get_embedded_labels has handed every usable bitmap to
        # its RasterizedMaskAdapter.  release_registration_cache may also run
        # on an adapter that has never entered label discovery, in which case
        # its metadata must remain complete for that later discovery.
        self._mask_payloads_transferred = False

        # Per-scene adapter cache, source-level only. Assigned here (not lazily on
        # first get_tensor_adapter) so no code path has to hedge about whether the
        # attribute exists; a per-instance dict, never a class attribute, for the
        # reason spelled out in biopb/biopb#522.
        self._tensor_adapters: dict = {}

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[object] = None
    ) -> "OmeTiffAdapter":
        """Create a source-level adapter from a SourceConfig."""
        return cls(str(source.url), source.source_id)

    @classmethod
    def create_from_payload(
        cls,
        source: "SourceConfig",
        payload: dict,
        metadata: dict,
        credentials_config: Optional[object] = None,
    ) -> Optional["OmeTiffAdapter"]:
        """Rebuild a source-level adapter from its stored payload: no file is opened.

        The scene descriptors (with their transfer grid) and each scene's physical
        scale come from the payload. The metadata is the row's, which is the whole
        of it for a file with no ROIs; a file that has them keeps ``get_metadata``,
        ``get_embedded_rois`` and the ``@ome`` labels lazy, parsing the file when
        asked, since the row holds neither the ROIs nor the mask bitmaps.

        ``None`` for a payload that predates the scale (the source is parsed).
        """
        if source.is_remote or payload.get("physical_scale") is None:
            return None
        scenes = payload.get("scenes")
        if not scenes:
            return None
        descriptors = [
            TensorDescriptor(
                array_id=s["array_id"],
                dim_labels=s["dim_labels"],
                shape=s["shape"],
                chunk_shape=s["chunk_shape"],
                dtype=s["dtype"],
            )
            for s in scenes
        ]
        adapter = cls._new_hydrated(str(source.url), source.source_id, descriptors)
        adapter._hydrated_payload = payload
        adapter._seeded_scales = {
            array_id: None if scale is None else (list(scale[0]), list(scale[1]))
            for array_id, scale in payload["physical_scale"].items()
        }
        if not payload.get("has_rois"):
            full = cls._parsed_form(metadata)
            adapter._parsed_metadata = full or None
            adapter._parsed_metadata_probed = True
            adapter._hydrated_metadata = full
        return adapter

    @classmethod
    def _parsed_form(cls, row_metadata: dict) -> dict:
        """The metadata a parse would give, from the row's: the row drops the empty
        ``rois`` an OME parse reports."""
        return {**row_metadata, "rois": []} if row_metadata else {}

    @classmethod
    def _new_hydrated(cls, url: str, source_id: str, descriptors) -> "OmeTiffAdapter":
        """A source-level adapter holding *descriptors*, built without reading."""
        adapter = cls(url, source_id)
        adapter._cached_descriptors = descriptors
        return adapter

    def _seed_scene(self, scene: "OmeTiffAdapter", array_id: str) -> None:
        """Hand a scene adapter what this rebuilt source knows, so the scene does not
        open the file for it either."""
        if self._hydrated_payload is None:
            return
        scene._hydrated_payload = self._hydrated_payload
        scales = self._seeded_scales
        if array_id in scales:
            scene._seeded_scale = scales[array_id]
        scene._hydrated_metadata = self._hydrated_metadata
        if self._hydrated_metadata is not None:
            scene._parsed_metadata = self._parsed_metadata
            scene._parsed_metadata_probed = True

    # ---- reads --------------------------------------------------------------

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """One whole page -- the same expression that seeds the grid below.

        The read path opens ``series.aszarr(level=0, chunkmode="page")``, and
        ``ZarrTiffStore._getitem`` serves any request from that mode by calling
        ``page.asarray()``: the entire page is decoded and the window sliced out
        of it. Measured flat in the request -- a 256^2 window costs 108 ms
        against 166 ms for the whole 8192^2 page -- and nothing caches the
        result, so N tiles cost N page decodes.

        Page mode is not a mistake to route around: it exists to coalesce many
        small striles into one buffered pass (9x on one-row strips) and to decode
        striles concurrently (2.4x on JPEG tiles, which the serial per-tile path
        gives up). It is the right mode for reading a page, so the page is the
        block.
        """
        descriptor = self._native_descriptor()
        return tuple(
            int(size) if str(label).upper() in {"Y", "X", "S"} else 1
            for label, size in zip(descriptor.dim_labels, descriptor.shape, strict=True)
        )

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from this scene's tifffile aszarr store.

        The read holds a lease on the pooled store, so it is never closed
        mid-read. By default readers of one store serialize on the handle's lock;
        ``BIOPB_OMETIFF_PARALLEL_READ=1`` (:func:`_parallel_read_enabled`) reads
        without it, since tifffile serializes the raw seek+read on its own handle
        lock and decodes per tile into a fresh buffer (biopb/biopb#473).

        Raises:
            ValueError: bad bounds, source-level adapter, or store unavailable.
        """
        if self.scene_index is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)  # validate bounds against the descriptor
        slices = bounds_to_slices(bounds)

        with self._leased_store() as handle:
            if handle is None:
                raise ValueError(
                    f"OME-TIFF aszarr store unavailable for {self._source_url!r} "
                    f"(scene {self.scene_index})"
                )
            za, axes = handle.value
            if _parallel_read_enabled():
                return self._read_region(za, axes, slices)
            with handle.lock:
                return self._read_region(za, axes, slices)

    def _pool_key(self):
        return (
            type(self).__name__,
            self._source_url,
            self.scene_index,
            self._content_version,
        )

    def _leased_store(self):
        """Lease this scene's store: pooled, or opened for this read alone when
        :meth:`_should_persist_store` says the file is too small to keep open."""
        return _store_pool.checkout(
            self._pool_key(), self._open_pooled, persist=self._should_persist_store()
        )

    def _open_pooled(self) -> Optional[PooledHandle]:
        """Open this scene's store as a handle the pool (or the caller) closes.

        None when the store is unavailable for this scene: a non-tifffile
        reader, a remote URL, a descriptor mismatch, or an open error.
        """
        try:
            opened = self._open_store()
        except Exception as exc:
            logger.debug("aszarr store unavailable for %s: %r", self._source_url, exc)
            return None
        if opened is None:
            return None
        za, axes, store, tiff = opened

        def close():
            for obj in (store, tiff):
                try:
                    obj.close()
                except Exception:
                    logger.debug("error closing aszarr store", exc_info=True)

        return PooledHandle(self._pool_key(), (za, axes), close)

    # ---- descriptors --------------------------------------------------------

    def _native_descriptor(self) -> TensorDescriptor:
        """Scene-level: the handed-down tifffile descriptor. Source-level: scene 0."""
        if self.scene_index is not None:
            return self._tifffile_descriptor
        return self._scene_descriptors()[0]

    def _scene_descriptors(self) -> List[TensorDescriptor]:
        """Per-scene **serving** descriptors derived from tifffile (cached).

        Each carries its own scene's transfer grid, seeded by that scene's page
        geometry, and is handed straight to the scene adapter by
        :meth:`get_tensor_adapter` -- the one object the listing and the read
        agree on. Internal: the catalog surface is
        :meth:`list_tensors`, which projects these.

        Returns an empty list when the source is not a tifffile-readable local
        OME-TIFF (remote, custom dim_labels, non-OME, exotic axes) -- ``claim``
        keeps those out, so in practice this always yields the real scenes.
        """
        if self._cached_descriptors is not None:
            return self._cached_descriptors
        descriptors = self._tifffile_descriptors()
        self._cached_descriptors = descriptors if descriptors is not None else []
        return self._cached_descriptors

    def list_tensors(self) -> List[TensorEntry]:
        """Structural catalog entries for every scene (no grid, #812)."""
        return [catalog_entry(d) for d in self._scene_descriptors()]

    def get_tensor_adapter(self, tensor_id: str) -> "TensorAdapter":
        """Build (and cache) the scene adapter for a within-source field.

        The scene adapter is handed the scene's tifffile descriptor, so it never
        re-derives it and reads straight from the aszarr store.
        """
        descriptors = self._scene_descriptors()
        field = strip_source_prefix(self.source_id, tensor_id)
        scene_idx = self._scene_index_for_field(field)

        if field in self._tensor_adapters:
            return self._tensor_adapters[field]

        adapter = OmeTiffAdapter(
            self._source_url,
            self.source_id,
            scene_index=scene_idx,
            tensor_descriptor=descriptors[scene_idx],
        )
        adapter._tensor_name = field
        self._seed_scene(adapter, descriptors[scene_idx].array_id)
        # Hand the scene the source's already-parsed OME-XML (_scene_descriptors
        # above populated it) so the scene's metadata / physical-scale paths read
        # the cached string instead of re-opening the master file once per scene --
        # the source parses the OME-XML once, every scene inherits it (mirrors how
        # bioio threads its shared _bio_image into scene adapters). Scenes are
        # built lazily at serve time, i.e. normally AFTER the post-registration
        # release, so in practice what they inherit is the stripped form -- which
        # is why physical scale must be derivable from it (biopb/biopb#783).
        if self._raw_ome_xml_probed:
            adapter._raw_ome_xml = self._raw_ome_xml
            adapter._raw_ome_xml_probed = True
            adapter._raw_ome_xml_released = self._raw_ome_xml_released
        if self._reduced_ome_xml_probed:
            adapter._reduced_ome_xml = self._reduced_ome_xml
            adapter._reduced_ome_xml_probed = True
        self._tensor_adapters[field] = adapter
        return adapter

    def _scene_index_for_field(self, field: Optional[str]) -> int:
        """Resolve a within-source scene field to its integer scene index.

        The cached descriptors are in series/scene order, so the position IS the
        scene index (and the aszarr ``series[index]`` the read opens).
        """
        for i, d in enumerate(self._scene_descriptors()):
            if strip_source_prefix(self.source_id, d.array_id) == field:
                return i
        raise TensorNotFound(f"Unknown scene: {field}", reason="unknown_field")

    # ---- metadata / physical scale -----------------------------------------

    def get_metadata(self) -> dict:
        """OME metadata dict from the stripped OME-XML (biopb/biopb#168), else {}.
        The catalog row's producer; the adapter itself reads :meth:`_ome_metadata`.

        Parses the OME-XML with per-plane ``<Plane>``/``<TiffData>`` elements
        stripped -- the same ome-types structure MINUS the per-plane arrays at a
        fraction of the cost. Runs at registration (the metadata-DB sync calls
        get_metadata), so keeping it cheap is what moves the OME parse off startup.

        Goes through ``_reduced_ome_xml_cached()``, not the raw string, so a re-sync
        (an unresolved source resolving) re-parses the stripped form already in
        hand rather than re-opening the file for a string it would strip again --
        and the *dict* that parse produces is itself cached (``_parsed_metadata``),
        since it is a pure function of that same string: a caller that also
        touches ``get_embedded_labels`` in the same registration (the registry's label-set view)
        gets the one parse already done, not a second one.
        """
        return self._ome_metadata()

    def _ome_metadata(self) -> dict:
        """The parsed stripped OME metadata, parsed once and kept."""
        if self._parsed_metadata_probed:
            return self._parsed_metadata or {}
        self._parsed_metadata_probed = True
        reduced = self._reduced_ome_xml_cached()
        if reduced:
            self._parsed_metadata = _fast_ome_metadata(reduced, already_reduced=True)
        return self._parsed_metadata or {}

    def get_embedded_rois(self, metadata, tensors, *, max_per_tensor=None):
        """The OME-XML ``<ROI>`` elements this file carries (see the base).

        Matched by id: ``_ome_scene_ids`` puts the OME image id straight into
        the array_id's field half, so the two id spaces are the same one.
        """
        return imported_annotations(
            metadata,
            tensors_by_field(tensors),
            content_version=self.content_version,
            max_per_tensor=max_per_tensor,
        )

    def get_embedded_labels(self) -> Dict[str, TensorAdapter]:
        """The ``@ome`` set: this file's own ``<Mask>`` ROI shapes, rasterized.

        One tensor per scene that carries at least one mask, keyed
        ``[<scene field>/]labels/@ome`` (see ``adapters/ome_masks.py``). Same
        OME-image-id join as :meth:`get_embedded_rois` (``tensors_by_field``):
        the field half of a scene's ``array_id`` IS the OME image id for this
        format, so the match is string equality, not inference.
        """
        plan = self._mask_label_plan()
        if not plan:
            self._mask_payloads_transferred = True
            return {}
        sets: Dict[str, TensorAdapter] = {}
        for desc, field, dim_labels, shape, masks in plan:
            sets[field] = RasterizedMaskAdapter(
                self.source_id,
                field,
                dim_labels=dim_labels,
                shape=shape,
                masks=masks,
                parent_array_id=desc.array_id,
                content_version=self.content_version,
            )
        self._mask_payloads_transferred = True
        return sets

    def _mask_label_plan(self) -> list:
        """``(scene descriptor, label field, dim_labels, shape, masks)`` for every
        scene that carries a ``<Mask>``, from the metadata alone: no bitmap is
        decoded here, so the catalog can list the label tensors without it."""
        descriptors = self._scene_descriptors()
        by_image = masks_by_image(
            self._ome_metadata(),
            tensors_by_field([(d.array_id, list(d.dim_labels)) for d in descriptors]),
        )
        plan = []
        for desc in descriptors:
            masks = by_image.get(desc.array_id)
            if not masks:
                continue
            dim_labels, shape = label_extent(list(desc.dim_labels), list(desc.shape))
            field = label_field(
                strip_source_prefix(self.source_id, desc.array_id) or "", OME_SET_NAME
            )
            plan.append((desc, field, dim_labels, shape, masks))
        return plan

    def _reduced_ome_xml_cached(self) -> Optional[str]:
        """The plane-stripped OME-XML, computed once and kept for the adapter's life.

        This is the form everything downstream of registration actually reads:
        ``<Plane>``/``<TiffData>`` removed (biopb/biopb#168) plus the degenerate
        ``<BinData>`` placeholder (biopb/biopb#199), so it is O(structure) rather
        than O(plane count) -- hundreds of bytes where the raw string is tens of
        MB. Retaining THIS and dropping the raw is the whole of biopb/biopb#783.
        Returns None for a source with no embedded OME-XML.
        """
        if self._reduced_ome_xml_probed:
            return self._reduced_ome_xml
        ome_xml = self._local_ome_xml()
        if not ome_xml:
            return None  # leave unprobed: nothing to strip, and nothing cached
        self._reduced_ome_xml_probed = True
        self._reduced_ome_xml = _STRIP_EMPTY_BINDATA.sub(
            "", _STRIP_PER_PLANE.sub("", ome_xml)
        )
        return self._reduced_ome_xml

    def _physical_scale(self):
        """Per-dim physical pixel size + unit from the local OME-XML (or None)."""
        if self._seeded_scale is not _UNSET:
            return self._seeded_scale
        return self._physical_scale_from_ome_xml()

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release this scene's pooled store (at its last lease) and cascade to
        the scene adapters."""
        if self.scene_index is not None:
            _store_pool.drop(self._pool_key())
        for adapter in list(self._tensor_adapters.values()):
            if adapter is not self:
                try:
                    adapter.close()
                except Exception:
                    logger.debug("error closing scene adapter", exc_info=True)

    def catalog_payload(self) -> Optional[Dict[str, Any]]:
        """The scene descriptors, which are what a read needs from the file.

        Source-level only. Each is the serving descriptor (``array_id``, axes,
        shape, dtype and the transfer grid seeded from the page geometry), plus
        ``has_rois`` and the embedded mask label tensors. Call it before
        :meth:`release_registration_cache`, which drops the mask bitmaps the
        label plan reads. ``None`` when tifffile declined the source.
        """
        if self.scene_index is not None:
            return None
        scenes = self._scene_descriptors()
        if not scenes:
            return None
        has_rois = bool(self._ome_metadata().get("rois"))
        scales = {}
        for d in scenes:
            scale = self.get_tensor_adapter(d.array_id)._physical_scale()
            scales[d.array_id] = (
                None if scale is None else [list(scale[0]), list(scale[1])]
            )
        return {
            "scenes": [
                {
                    "array_id": d.array_id,
                    "dim_labels": list(d.dim_labels),
                    "shape": [int(s) for s in d.shape],
                    "chunk_shape": [int(c) for c in d.chunk_shape],
                    "dtype": d.dtype,
                }
                for d in scenes
            ],
            # Whether the file carries ``<ROI>`` elements, so a read can import
            # them on first request instead of at registration.
            "has_rois": has_rois,
            # Each scene's calibration, so a restart serves it without the XML.
            "physical_scale": scales,
            # The ``@ome`` label tensors, so the catalog lists them without
            # parsing the masks; the bitmaps are read when the tensor is.
            "masks": [
                {
                    "field": field,
                    "parent_array_id": desc.array_id,
                    "dim_labels": list(dim_labels),
                    "shape": [int(s) for s in shape],
                }
                for desc, field, dim_labels, shape, _ in self._mask_label_plan()
            ],
        }

    def release_registration_cache(self) -> None:
        """Drop the raw OME-XML now that the catalog holds the metadata (#783).

        The raw string exists to build the catalog row; once that row is
        committed it is an uncompressed duplicate of something DuckDB already
        stores in stripped form, resident for as long as the source is
        registered -- i.e. forever, in a serving process. On a per-plane
        acquisition (40,000 timepoints is real) that is tens of MB per source.

        Kept: ``_reduced_ome_xml``, which carries every ``<Image>``/``<Pixels>``
        header and so still answers ``get_metadata`` and ``_physical_scale``
        without touching the file. Also kept is ``_raw_ome_xml_probed`` -- the
        release marks ``_raw_ome_xml_released`` instead of un-probing, or every
        later call would re-open the file and we would have traded a memory leak
        for an I/O one. Recoverable, not lossy: a consumer that genuinely needs
        the full document calls ``_local_ome_xml()`` and pays for it once.

        Only flips the released flag when there was a string to drop, so a
        source with no embedded OME-XML keeps answering None from cache.
        Safe to call twice.

        Derives the stripped form BEFORE dropping the source string, and hands
        it down to every scene, because what a scene inherited in
        ``get_tensor_adapter`` is a snapshot of whatever existed when it was
        built. A scene built between descriptor discovery and ``get_metadata``
        -- the window the reconciler opens by registering a source before
        syncing it, during which a ``GetFlightInfo`` or a precache warm can land
        -- holds the raw string and no stripped one. Releasing that scene
        without settling it first would leave it with nothing but the file, and
        its next physical-scale call would reopen the file AND re-cache the raw
        string for good: the leak back, on a scene nothing releases again.
        """
        if (
            self._hydrated_payload is not None
            and not self._raw_ome_xml_probed
            and not self._reduced_ome_xml_probed
        ):
            # Rebuilt from a payload and never asked for the XML: there is nothing
            # to settle, and settling would open the file.
            for adapter in list(self._tensor_adapters.values()):
                if adapter is not self:
                    adapter.release_registration_cache()
            return
        # Settle first, drop second: a get_tensor_adapter racing this then
        # inherits either (raw, unsettled) and gets cascaded below, or (no raw,
        # settled) and needs nothing.
        reduced = self._reduced_ome_xml_cached()
        if reduced is not None and self._mask_payloads_transferred:
            # The label adapters have already copied the decoded mask bytes.
            # Retaining either the base64 XML or the parsed dict would duplicate
            # a potentially very large payload for the source lifetime. This
            # method must never raise (docstring above), so a reduced XML the
            # stdlib parser rejects is left un-redacted rather than aborting the
            # release below and the cascade to every scene -- worse for memory
            # on that one source, not a correctness problem.
            try:
                reduced = _strip_mask_bindata_payloads(reduced)
            except ET.ParseError:
                logger.debug("could not redact mask payloads", exc_info=True)
            else:
                self._reduced_ome_xml = reduced
                self._parsed_metadata = None
                self._parsed_metadata_probed = False
        if self._raw_ome_xml is not None:
            self._raw_ome_xml = None
            self._raw_ome_xml_released = True
        for adapter in list(self._tensor_adapters.values()):
            if adapter is self:
                continue
            if reduced is not None and (
                self._mask_payloads_transferred or not adapter._reduced_ome_xml_probed
            ):
                adapter._reduced_ome_xml = reduced
                adapter._reduced_ome_xml_probed = True
            if self._mask_payloads_transferred:
                adapter._parsed_metadata = None
                adapter._parsed_metadata_probed = False
                adapter._mask_payloads_transferred = True
            adapter.release_registration_cache()

    # ---- OME-XML internals --------------------------------------------------

    def _local_ome_xml(self) -> Optional[str]:
        """Return the embedded OME-XML string for a local source, or None.

        Cached on the instance (and populated as a side effect of the descriptor
        path) so registration opens the file at most once across the descriptor,
        metadata, and physical-scale paths. Returns None for remote or non-OME
        sources.

        After ``release_registration_cache`` the cache is gone but the file
        still has the XML, so this re-reads it (biopb/biopb#783). That re-read
        is the price of asking for the full document post-registration -- no
        in-tree caller does; both remaining consumers read the stripped form.
        """
        if self._raw_ome_xml_probed and not self._raw_ome_xml_released:
            return self._raw_ome_xml
        self._raw_ome_xml_probed = True
        self._raw_ome_xml_released = False
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
        (remote/non-``file://`` URL, non-OME TIFF, zero series, or a non-OME axis). Scene IDs match the OME ``Image`` IDs so
        the catalog array_ids are stable, and only the tiny OME-XML header is read
        (no ome-types object graph). Canonical ``TCZYX`` and interleaved RGB(A)
        (``TCZYXS``) are both mapped natively via ``_ome_axes_shape``.
        """
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
                self._raw_ome_xml_released = False
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
                    # One whole page seeds the grid --
                    # series.aszarr(chunkmode="page").chunks in canonical order
                    # (full Y/X and RGB samples S, 1 elsewhere). It is the read
                    # path's native unit, so the transfer grid stays a whole
                    # multiple of it rather than straddling pages; a page above
                    # the Arrow ceiling is still re-split by
                    # transfer_chunk_size (biopb/biopb#809).
                    descriptors.append(
                        TensorDescriptor(
                            # Identity policy: array_id = source_id/field; the
                            # field is the OME Image ID (scene id).
                            array_id=f"{self.source_id}/{scene_ids[i]}",
                            dim_labels=dim_labels,
                            shape=shape,
                            chunk_shape=default_transfer_chunk_shape(
                                shape,
                                s.dtype.str,
                                dim_labels,
                                native=[
                                    n if d in ("Y", "X", "S") else 1
                                    for d, n in zip(dim_labels, shape, strict=True)
                                ],
                            ),
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
        """Physical scale from this scene's ``<Pixels>`` header, or None.

        Scans the plane-stripped XML when it is already in hand -- stripping
        removes only ``<Plane>``/``<TiffData>``, so every ``<Image>``/
        ``<Pixels>`` header survives it, and after the post-registration release
        it is the only document left (biopb/biopb#783). Registration always
        computes it (``get_metadata`` does), so post-release it is always there.

        Never *computes* it just for this: ``iterparse`` stops at the requested
        image's ``<Pixels>``, which is cheaper than the whole-document strip that
        would produce the reduced form. Falling back to the raw document is also
        what happens if the stripped one fails to parse -- which would equally
        have failed ``get_metadata``. A stripped document that parses and names
        no physical size is a legitimate ``None``, not a reason to re-read.
        Never raises.
        """
        if self._reduced_ome_xml:
            try:
                return self._scan_physical_scale(self._reduced_ome_xml)
            except Exception:
                logger.debug(
                    "physical-scale scan failed on stripped OME-XML for %s",
                    self._source_url,
                    exc_info=True,
                )
        try:
            ome_xml = self._local_ome_xml()
            return self._scan_physical_scale(ome_xml) if ome_xml else None
        except Exception:
            return None

    def _scan_physical_scale(self, ome_xml: str):
        """Scan one OME-XML document for this scene's physical pixel size.

        Namespace-agnostic ElementTree scan (NOT an ome-types object build): find
        the ``<Image>`` at this scene's index in document order, read its
        ``<Pixels>`` ``PhysicalSizeX/Y/Z`` (+ ``...Unit``), and map onto
        ``dim_labels`` by lowercased axis label (T/C/S -> ``0.0`` / ``""``).
        Physical sizes occur on ``<Pixels>`` before per-plane elements, so stream
        only as far as the requested image's header. A missing ``*Unit`` defaults
        to ``"µm"`` (OME spec default). Returns ``None`` when no positive size is
        present; propagates a parse error so the caller can pick another document.
        """

        def _local(tag):
            return str(tag).rsplit("}", 1)[-1]

        idx = self.scene_index or 0

        def _size(axis):
            raw = attrs.get(f"PhysicalSize{axis}")
            if raw is None:
                return 0.0, ""
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return 0.0, ""
            if v <= 0:
                return 0.0, ""
            return v, (attrs.get(f"PhysicalSize{axis}Unit") or "µm")

        images_seen = -1
        attrs = None
        for _, element in ET.iterparse(io.StringIO(ome_xml), events=("start",)):
            if _local(element.tag) == "Image":
                images_seen += 1
            elif _local(element.tag) == "Pixels" and images_seen == idx:
                attrs = element.attrib
                break
        if attrs is None:
            return None

        by_label = {"x": _size("X"), "y": _size("Y"), "z": _size("Z")}
        scale, unit = [], []
        for lab in self.dim_labels or []:
            v, u = by_label.get(str(lab).lower(), (0.0, ""))
            scale.append(v)
            unit.append(u)
        return (scale, unit) if any(scale) else None

    # ---- persistent aszarr store -------------------------------------------
    def _should_persist_store(self) -> bool:
        """Whether an opened aszarr store should remain open between reads."""
        return True

    def _open_store(self):
        """Open ``series[scene].aszarr`` as a zarr array; validate vs the descriptor.

        Returns ``(zarr_array, axes_str, store, tiff)`` or None; the caller owns
        closing ``store`` and ``tiff``. Raises on open/read errors.
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
                # Not this scene's store: close the fresh store + file handle
                # before bailing. The success path stashes them below for reuse;
                # the reject path must not leak the open fd.
                for obj in (store, tiff):
                    try:
                        obj.close()
                    except Exception:
                        logger.debug(
                            "error closing rejected aszarr store", exc_info=True
                        )
                return None
        except Exception:
            tiff.close()
            raise

        return za, axes, store, tiff

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
            and name.endswith((".tif", ".tiff"))
            # Cloud-storage phase 2: the embedded-OME-XML sniff opens the whole
            # TIFF (a recall on a non-resident placeholder). Skip it when the file
            # is not resident: the generic extension-only AicsImageIoAdapter then
            # claims the .tif as an unresolved image.
            and ctx.is_resident()
        ):
            # Only a monitored root is walked again, so only its probes are kept.
            referenced = _get_ome_files(ctx._path, memoize=ctx.monitored)

            if referenced is not None:
                related_files = _existing_files(
                    referenced, ctx.parent.path_str, ctx.store
                )
                if related_files:
                    primary_path = related_files[0]
                    for f in related_files:
                        state.try_claim_path(f)
                    return SourceClaim(
                        source_type=cls.SOURCE_TYPE,
                        primary_path=primary_path,
                    )
                # Single-file OME-TIFF: embedded OME-XML but no <UUID FileName>
                # references -- the common case, since tifffile and most writers
                # emit a bare <TiffData IFD=.../> with no file list. Claim the file
                # itself as a single-member source so it takes the pure-tifffile
                # path (#168 fast-path descriptors + aszarr reads). Without this it
                # falls through to the generic bioio adapter, which reverts to
                # the full O(planes) OME-model parse #168 exists to avoid.
                return SourceClaim(
                    source_type=cls.SOURCE_TYPE,
                    primary_path=ctx.path_str,
                )

        return None
