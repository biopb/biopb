"""OME-embedded masks, rasterized into the reserved ``@ome`` label set
(biopb/biopb#1059 step 4).

Companion to :mod:`biopb_tensor_server.adapters._ome_rois`, which imports
every OTHER OME-XML shape kind as an annotation and drops a ``<Mask>`` (a
``BinData`` bitmap is not annotation geometry). It IS pixels, though, and
this is where they become the label tensor the design
(``docs/label-tensors.md``, "Rasterizing OME masks") describes: one tensor
per image, id ``<image>/labels/@ome``, whose label value is the 1-based
index of the mask's ROI in that image's own ``roi_refs`` order -- stable,
because it comes from the metadata, exactly as ``_ome_rois`` already relies
on for its own join. Later ROI wins where masks overlap, which the same
order gives for free: shapes are painted in ``roi_refs`` order.

Pure by design, like its sibling: dict in (the OME metadata), a list of
:class:`_MaskShape` per image out (:func:`masks_by_image`).
:class:`RasterizedMaskAdapter` is the tensor adapter that reads one; nothing
here touches a file, and nothing here is registered by discovery -- it is
built only by ``OmeTiffAdapter.get_embedded_labels``.
"""

from __future__ import annotations

import base64
import bz2
import dataclasses
import logging
import math
import zlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._ome_rois import Tensor
from biopb_tensor_server.adapters.labels import NearestPyramidMixin
from biopb_tensor_server.core.adapter_base import TensorAdapter, catalog_entry
from biopb_tensor_server.core.axes import labeled_axis_index
from biopb_tensor_server.core.chunk import default_transfer_chunk_shape
from biopb_tensor_server.core.errors import WriteNotSupportedError

__all__ = ["RasterizedMaskAdapter", "masks_by_image", "strip_mask_bindata"]

logger = logging.getLogger(__name__)

#: The dtype every rasterized set carries -- small next to any real
#: instance count, and the recommendation the design makes for an uploaded
#: set too (``docs/label-tensors.md``, "Dtype").
_DTYPE = "<u4"


@dataclasses.dataclass(frozen=True)
class _MaskShape:
    """One OME ``<Mask>``, as much as painting it needs."""

    label: int  #: 1-based index of its ROI in the image's roi_refs order
    x: float
    y: float
    width: float
    height: float
    the_z: Optional[int]
    the_t: Optional[int]
    raw: bytes  #: ``bin_data.value``, base64-decoded; still possibly compressed
    compression: str  #: ``"none"`` | ``"zlib"`` | ``"bzip2"``


def _decompress(raw: bytes, compression: str) -> bytes:
    if compression == "zlib":
        return zlib.decompress(raw)
    if compression == "bzip2":
        return bz2.decompress(raw)
    return raw


def _bbox(shape: _MaskShape) -> Tuple[int, int, int, int]:
    """*shape*'s pixel-space bounding box, half-open: ``(x0, y0, x1, y1)``.

    The one place ``x``/``y``/``width``/``height`` become integers, so
    :func:`_bitmap`'s shape and the region a read paints (``_paint``) can
    never disagree on it.
    """
    x0, y0 = int(math.floor(shape.x)), int(math.floor(shape.y))
    return x0, y0, x0 + int(round(shape.width)), y0 + int(round(shape.height))


def _bitmap(shape: _MaskShape) -> np.ndarray:
    """*shape*'s own ``(height, width)`` boolean bitmap, unpacked once.

    OME packs a mask 1 bit per pixel, row-major, most-significant-bit-first,
    with no per-row padding: the bit at flat index ``r * width + c`` is pixel
    ``(r, c)``. A bitstream short of ``height * width`` bits (a producer that
    rounds byte-aligned) reads as unset past its end rather than raising.
    """
    x0, y0, x1, y1 = _bbox(shape)
    h, w = max(0, y1 - y0), max(0, x1 - x0)
    n = h * w
    if n == 0:
        return np.zeros((h, w), dtype=bool)
    data = _decompress(shape.raw, shape.compression)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    if bits.size < n:
        bits = np.pad(bits, (0, n - bits.size))
    return bits[:n].reshape(h, w).astype(bool)


def _mask_shape(shape: Mapping[str, Any], label: int) -> Optional[_MaskShape]:
    """One raw OME shape dict as a :class:`_MaskShape`, or None if unreadable."""
    bin_data = shape.get("bin_data")
    if not isinstance(bin_data, Mapping):
        return None
    value = bin_data.get("value") or ""
    raw = base64.b64decode(value) if isinstance(value, str) else bytes(value or b"")

    def optional_int(key: str) -> Optional[int]:
        v = shape.get(key)
        return int(v) if v is not None and int(v) >= 0 else None

    return _MaskShape(
        label=label,
        x=float(shape.get("x") or 0.0),
        y=float(shape.get("y") or 0.0),
        width=float(shape.get("width") or 0.0),
        height=float(shape.get("height") or 0.0),
        the_z=optional_int("the_z"),
        the_t=optional_int("the_t"),
        raw=raw,
        compression=str(bin_data.get("compression") or "none"),
    )


def strip_mask_bindata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """*metadata* with every ``rois[].union.masks[].bin_data.value`` dropped.

    A mask's bitmap is arbitrary binary -- base64 text on the fast metadata
    path (:func:`~biopb_tensor_server.adapters.ome_tiff._b64_encode_mask_bindata`),
    raw bytes elsewhere -- and can be large; it belongs in the rasterized
    ``@ome`` tensor this module builds, never in the SQL-queryable
    ``sources.metadata_json`` column. Unlike ``rois`` as a whole, this runs
    whether or not ROI *annotations* import ran: a mask is not an
    annotation, so it is not covered by that stripping (``metadata_db.py``),
    and its bitmap must not leak into metadata_json regardless.

    Returns *metadata* unchanged (same object) when there is nothing to
    strip, so a caller can skip the JSON re-dump in the common case of no
    masks at all.
    """
    rois = metadata.get("rois")
    if not isinstance(rois, list) or not rois:
        return metadata
    changed = False
    new_rois = []
    for roi in rois:
        union = roi.get("union") if isinstance(roi, Mapping) else None
        masks = union.get("masks") if isinstance(union, Mapping) else None
        if not masks:
            new_rois.append(roi)
            continue
        new_masks = []
        for mask in masks:
            bin_data = mask.get("bin_data") if isinstance(mask, Mapping) else None
            if isinstance(bin_data, Mapping) and "value" in bin_data:
                mask = {
                    **mask,
                    "bin_data": {k: v for k, v in bin_data.items() if k != "value"},
                }
                changed = True
            new_masks.append(mask)
        new_rois.append({**roi, "union": {**union, "masks": new_masks}})
    if not changed:
        return metadata
    return {**metadata, "rois": new_rois}


def masks_by_image(
    metadata: Mapping[str, Any], by_image_id: Mapping[str, Tensor]
) -> Dict[str, List[_MaskShape]]:
    """Every image's ``<Mask>`` shapes, keyed by ``array_id``, in ``roi_refs`` order.

    ``by_image_id`` maps an OME image id to ``(array_id, dim_labels)`` -- the
    same shape :func:`~biopb_tensor_server.adapters._ome_rois.imported_annotations`
    takes, resolved the same way (``tensors_by_field`` / ``tensors_by_image_order``).
    A ROI referenced by no image, or one this source has no tensor for,
    contributes nothing here -- exactly ``_ome_rois``'s ``unreferenced`` /
    ``unmatched_refs`` cases, already counted by the annotation import that
    runs over the same metadata, so not double-counted here.

    One shape at a time, because this reads a file this module did not
    write: unreadable shapes are skipped and logged rather than costing the
    image its other masks.
    """
    rois_by_id: Dict[str, Mapping[str, Any]] = {
        str(r.get("id")): r
        for r in (metadata.get("rois") or [])
        if isinstance(r, Mapping)
    }
    out: Dict[str, List[_MaskShape]] = {}
    images = metadata.get("images") or []
    for image in images if isinstance(images, (list, tuple)) else []:
        if not isinstance(image, Mapping):
            continue
        target = by_image_id.get(str(image.get("id")))
        if target is None:
            continue
        array_id, _ = target
        shapes: List[_MaskShape] = []
        refs = image.get("roi_refs") or []
        for position, ref in enumerate(
            refs if isinstance(refs, (list, tuple)) else [], start=1
        ):
            if not isinstance(ref, Mapping):
                continue
            roi = rois_by_id.get(str(ref.get("id")))
            union = (roi or {}).get("union") if isinstance(roi, Mapping) else None
            if not isinstance(union, Mapping):
                continue
            for raw_shape in union.get("masks") or []:
                if not isinstance(raw_shape, Mapping):
                    continue
                try:
                    shape = _mask_shape(raw_shape, position)
                except Exception:
                    logger.debug(
                        "ome masks: unreadable mask on %s", array_id, exc_info=True
                    )
                    shape = None
                if shape is not None:
                    shapes.append(shape)
        if shapes:
            out[array_id] = shapes
    return out


class RasterizedMaskAdapter(NearestPyramidMixin, TensorAdapter):
    """The ``@ome`` label set: OME ``<Mask>`` shapes painted into one tensor.

    Computed, not stored -- there is no backend to read again, only the
    shapes this adapter was built with (:func:`masks_by_image`). A chunk
    read decodes only the masks whose bounding box and plane pin intersect
    the requested bounds; decoded bitmaps are memoized per shape for this
    adapter's life, so a re-read (or a neighbouring chunk) never re-decodes
    one. Read-only: replacing the set means re-registering the file.

    ``dim_labels`` / ``shape`` are the image's own canonical axes with the
    channel axis dropped (design, "Extent") -- already canonical, since they
    come from the image's own descriptor, so this adapter never needs
    permuting (:attr:`_normalizable_axes`). Y/X are located by *label*
    (:func:`~biopb_tensor_server.core.axes.labeled_axis_index`), never by
    position: an interleaved RGB(A) source keeps its trailing samples axis
    (``S``) here (``label_extent`` drops only the channel axis), so Y/X are
    NOT reliably the last two axes. A pin (``TheZ``/``TheT``) that names an
    axis this image does not have is inert, exactly like ``TheC`` -- the
    channel distinction is never carried here (design, "Extent") -- and so is
    an ``S`` axis, which no mask pins and every mask therefore paints across.
    """

    _normalizable_axes = False

    def __init__(
        self,
        source_id: str,
        field: str,
        *,
        dim_labels: Sequence[str],
        shape: Sequence[int],
        masks: Sequence[_MaskShape],
        parent_array_id: str,
        content_version: Optional[bytes],
    ) -> None:
        self.source_id = source_id
        self._tensor_name = field
        self._dim_labels = [str(d) for d in dim_labels]
        self._shape = [int(s) for s in shape]
        self._masks = list(masks)
        self._parent_array_id = parent_array_id
        self._content_version = content_version
        self._axis = {
            axis: labeled_axis_index(self._dim_labels, axis)
            for axis in ("t", "z", "y", "x")
        }
        self._bitmaps: Dict[int, np.ndarray] = {}
        self._unreadable_bitmap_ids: set[int] = set()

    @property
    def dim_labels(self) -> List[str]:
        return self._dim_labels

    def get_metadata(self) -> dict:
        return {}

    def get_embedded_labels(self) -> Dict[str, TensorAdapter]:
        return {}  # a set has no sets of its own

    @classmethod
    def create_from_config(
        cls, source, credentials_config=None
    ) -> RasterizedMaskAdapter:
        raise NotImplementedError(
            "RasterizedMaskAdapter is built from a parent's OME metadata by "
            "OmeTiffAdapter.get_embedded_labels(), not from config"
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [catalog_entry(self.get_tensor_descriptor())]

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self._dim_labels,
            shape=self._shape,
            chunk_shape=default_transfer_chunk_shape(
                self._shape, _DTYPE, self._dim_labels
            ),
            dtype=_DTYPE,
        )

    def get_tensor_metadata(self) -> Optional[dict]:
        """``image-label`` naming the parent image; no colours -- OME's
        per-shape ``fill_color`` has no NGFF-wide LUT equivalent here."""
        return {
            "image-label": {
                "version": "0.4",
                "source": {"image": self._parent_array_id},
            }
        }

    def put_chunk(self, bounds, data, expected_shape, dtype) -> None:
        raise WriteNotSupportedError(
            f"label set {self.array_id!r} is read-only: it is rasterized from "
            "the file's own OME metadata, not uploaded"
        )

    def _bitmap_of(self, shape: _MaskShape) -> Optional[np.ndarray]:
        """Return a decoded bitmap, or ``None`` for a corrupt one.

        Compression is deliberately deferred until a read intersects the
        shape.  A malformed stream must therefore fail at this per-shape
        boundary, rather than making every chunk of the whole label set
        unreadable.  Cache the failure too, both to avoid repeated decompression
        attempts and to emit one useful warning instead of one per chunk.
        """
        key = id(shape)
        if key in self._unreadable_bitmap_ids:
            return None
        cached = self._bitmaps.get(key)
        if cached is not None:
            return cached
        try:
            cached = _bitmap(shape)
        except Exception:
            logger.warning(
                "ome masks: skipping unreadable compressed mask in %s",
                self.array_id,
                exc_info=True,
            )
            self._unreadable_bitmap_ids.add(key)
            return None
        self._bitmaps[key] = cached
        return cached

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        super().get_data(bounds)
        starts = [int(s) for s in bounds.start]
        stops = [int(e) for e in bounds.stop]
        out = np.zeros(
            tuple(e - s for s, e in zip(starts, stops, strict=True)), dtype=np.uint32
        )
        y_axis, x_axis = self._axis["y"], self._axis["x"]
        if y_axis is None or x_axis is None:
            logger.warning(
                "ome masks: %s has no y/x axis; serving an empty set", self.array_id
            )
            return out
        for shape in self._masks:
            self._paint(out, starts, stops, shape, y_axis, x_axis)
        return out

    def _paint(
        self,
        out: np.ndarray,
        starts: List[int],
        stops: List[int],
        shape: _MaskShape,
        y_axis: int,
        x_axis: int,
    ) -> None:
        """Paint *shape* into *out* wherever its bbox and plane pin intersect
        the requested bounds; a no-op if either misses entirely.

        Every axis stays a ``slice`` (never an int index), including a pinned
        one collapsed to a single position -- so ``region`` always keeps
        *out*'s full rank and axis order, and the bitmap's ``(y, x)`` plane
        can be reshaped straight into *y_axis* / *x_axis* by position,
        wherever they actually sit (never assumed to trail: see the class
        docstring).
        """
        index: List[Any] = [slice(None)] * out.ndim
        for attr, pin in (("z", shape.the_z), ("t", shape.the_t)):
            axis = self._axis.get(attr)
            if axis is None or pin is None:
                continue  # no such axis here, or unpinned: every plane
            if not (starts[axis] <= pin < stops[axis]):
                return  # this chunk holds none of the pinned plane
            rel = pin - starts[axis]
            index[axis] = slice(rel, rel + 1)

        mx0, my0, mx1, my1 = _bbox(shape)
        x0, x1 = max(mx0, starts[x_axis]), min(mx1, stops[x_axis])
        y0, y1 = max(my0, starts[y_axis]), min(my1, stops[y_axis])
        if x0 >= x1 or y0 >= y1:
            return

        bitmap = self._bitmap_of(shape)
        if bitmap is None:
            return
        bitmap = bitmap[y0 - my0 : y1 - my0, x0 - mx0 : x1 - mx0]
        index[y_axis] = slice(y0 - starts[y_axis], y1 - starts[y_axis])
        index[x_axis] = slice(x0 - starts[x_axis], x1 - starts[x_axis])
        region = out[tuple(index)]  # full rank: no axis was int-indexed above
        broadcast_shape = [1] * region.ndim
        broadcast_shape[y_axis] = bitmap.shape[0]
        broadcast_shape[x_axis] = bitmap.shape[1]
        painted = bitmap.reshape(broadcast_shape)
        region[...] = np.where(painted, shape.label, region)
