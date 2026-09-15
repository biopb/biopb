"""OME-embedded ROIs, turned into annotation rows (biopb/biopb#951).

An OME-TIFF carries its own ROIs, and they already reach the catalog: the fast
metadata path strips only ``<Plane>``/``<TiffData>``, so ``<ROI>`` survives into
the ome-types dump that ``SourceAdapter.get_metadata()`` returns. This module is
the pure half of importing them -- dict in, ``RoiAnnotation`` out -- so the store
can file them in the reserved ``@ome`` set that #952 made read-only.

Pure by design: no DB, no adapter, no I/O. Everything it needs is in the metadata
dict and the tensor list the caller already holds, which is what makes the import
free at registration time (no second parse) and the geometry exhaustively
testable without a fixture file.

The join to a tensor is exact rather than inferred::

    rois[].id  <-  images[n].roi_refs[].id  <-  images[n].id  ==  array_id field

``array_id`` is ``f"{source_id}/{scene_id}"`` and ``_ome_scene_ids`` reads those
scene ids from the OME ``Image`` ``ID`` attributes, so the field half of an
array_id IS the OME image id. Its documented fallback -- positional
``Image:{i}`` on a count mismatch -- is the one case the two sides can disagree,
so misses are counted and reported rather than assumed away.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from biopb.image.annotation_pb2 import RoiAnnotation
from biopb.image.roi_pb2 import ROI, Ellipse, Point, Polygon, Polyline, Rectangle

from biopb_tensor_server.core.axes import canonical_axis

logger = logging.getLogger(__name__)

# The set every imported ROI lands in. Reserved (see RESERVED_SET_PREFIX in
# metadata_db): clients may read and clone it, never write it.
OME_SET_NAME = "@ome"

# OME pins a shape to a plane with TheZ/TheT/TheC; the store pins by axis
# POSITION, because a label is neither guaranteed present nor unique. Mapping
# through canonical_axis rather than matching "z" literally keeps a source whose
# dim_labels spell it "depth" or "frame" working.
_PIN_ATTRS = {"the_t": "t", "the_z": "z", "the_c": "c"}


@dataclass
class ImportReport:
    """What the import could not carry, counted rather than swallowed."""

    imported: int = 0
    #: Masks are refused by the store itself (a BinData bitmap is not an
    #: annotation), so they are dropped here -- but loudly.
    dropped_masks: int = 0
    #: A transform with a singular matrix collapses its shape to a segment or a
    #: point. There is no honest geometry to store.
    dropped_degenerate: int = 0
    #: A ROI referenced by an image that no tensor claims. Non-zero means the
    #: `_ome_scene_ids` fallback fired and the two id spaces disagree.
    unmatched_refs: int = 0
    #: ROIs no image references at all. Legal OME, and nothing to attach them to.
    unreferenced: int = 0
    #: Per-tensor cap reached; the remainder of that tensor's ROIs was dropped.
    over_cap: int = 0
    #: A shape this module could not read: a coordinate that is not a number, a
    #: plane index outside uint32, a `union` that is not a mapping. Dropped one
    #: at a time so one bad shape cannot cost a file its other forty.
    malformed: int = 0
    tensors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(
            self.imported
            or self.dropped_masks
            or self.dropped_degenerate
            or self.unmatched_refs
            or self.unreferenced
            or self.over_cap
            or self.malformed
        )

    def summary(self) -> str:
        parts = [f"{self.imported} imported across {len(self.tensors)} tensor(s)"]
        for label, n in (
            ("mask(s) dropped", self.dropped_masks),
            ("degenerate transform(s) dropped", self.dropped_degenerate),
            ("ref(s) matched no tensor", self.unmatched_refs),
            ("ROI(s) referenced by no image", self.unreferenced),
            ("dropped over the per-tensor cap", self.over_cap),
            ("unreadable shape(s) dropped", self.malformed),
        ):
            if n:
                parts.append(f"{n} {label}")
        return ", ".join(parts)


# --- affine ------------------------------------------------------------------
# OME puts an optional AffineTransform on each SHAPE (not on the ROI), a 2x3 in
# pixel space: x' = a00 x + a01 y + a02. We bake it into the coordinates rather
# than carrying it to clients, because the store's contract is "level-0 pixels,
# the server never rescales geometry" and `bbox` is derived server-side -- a
# geometry plus a sidecar matrix would leave bbox the only transformed thing in
# the row. Ignoring it, the third option, puts a shape somewhere plausible and
# wrong with nothing downstream able to tell: exactly the failure biopb/biopb#945
# fixed for ellipse rotation.

_Affine = Tuple[float, float, float, float, float, float]


def _affine(transform: Optional[Mapping[str, Any]]) -> Optional[_Affine]:
    """``(a00, a01, a02, a10, a11, a12)``, or None for absent/identity."""
    if not transform:
        return None
    try:
        m = tuple(
            float(transform[k]) for k in ("a00", "a01", "a02", "a10", "a11", "a12")
        )
    except (KeyError, TypeError, ValueError):
        return None
    return None if m == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0) else m  # type: ignore[return-value]


def _map(m: Optional[_Affine], x: float, y: float) -> Tuple[float, float]:
    if m is None:
        return x, y
    return m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]


def _det(m: _Affine) -> float:
    return m[0] * m[4] - m[1] * m[3]


def _axis_aligned(m: _Affine) -> bool:
    """No rotation or shear, so a rectangle survives as a rectangle."""
    return m[1] == 0.0 and m[3] == 0.0


def _ellipse_axes(
    m: Optional[_Affine], rx: float, ry: float
) -> Tuple[float, float, float]:
    """Semi-axes and rotation of an axis-aligned ellipse under ``m``.

    Returns ``(rx', ry', rotation_radians)``. Exact for any invertible affine:
    an affine image of an ellipse is always an ellipse, and center + two
    semi-axes + rotation describes any ellipse -- so this is lossless only
    because biopb/biopb#945 added ``Ellipse.rotation``.

    ``A = M diag(rx, ry)`` maps the unit circle onto the result, so its singular
    values are the semi-axes and its left singular vectors are their directions.
    Closed-form 2x2 SVD rather than numpy: it is six lines, and this runs inside
    source registration.
    """
    if m is None:
        return rx, ry, 0.0
    a, b = m[0] * rx, m[1] * ry
    c, d = m[3] * rx, m[4] * ry
    e, f = (a + d) / 2.0, (a - d) / 2.0
    g, h = (c + b) / 2.0, (c - b) / 2.0
    q, r = math.hypot(e, h), math.hypot(f, g)
    # A negative second singular value is a reflection; the semi-axis is its
    # magnitude, and the major-axis direction is unaffected. The angle is the
    # LEFT singular vectors' -- the sum, not the difference, which is the right
    # singular vectors' and describes the pre-image. It is correct modulo pi
    # (numpy's SVD may hand back the antipodal column), and an ellipse axis is a
    # line, not a ray, so that ambiguity is not one.
    return q + r, abs(q - r), (math.atan2(g, f) + math.atan2(h, e)) / 2.0


# --- shapes ------------------------------------------------------------------


def _points(spec: Any) -> List[Tuple[float, float]]:
    """Parse an OME ``Points`` string: ``"1,1 3,1 3,3"``."""
    out: List[Tuple[float, float]] = []
    for pair in str(spec or "").split():
        x, _, y = pair.partition(",")
        try:
            out.append((float(x), float(y)))
        except ValueError:
            continue
    return out


def _pt(xy: Tuple[float, float]) -> Point:
    return Point(x=xy[0], y=xy[1])


def _coords(roi: ROI) -> Tuple[float, ...]:
    kind = roi.WhichOneof("shape")
    if kind == "point":
        return (roi.point.x, roi.point.y)
    if kind == "rectangle":
        r = roi.rectangle
        return (r.top_left.x, r.top_left.y, r.bottom_right.x, r.bottom_right.y)
    if kind == "ellipse":
        e = roi.ellipse
        return (e.center.x, e.center.y, e.radius.x, e.radius.y, e.rotation)
    points = roi.polygon.points if kind == "polygon" else roi.polyline.points
    return tuple(v for p in points for v in (p.x, p.y))


def _finite(roi: ROI) -> bool:
    """Whether every coordinate is a real number.

    Checked because nothing downstream would: proto floats accept NaN and inf,
    DuckDB DOUBLE accepts them, and the derived bbox propagates them -- so an
    overflowing affine or a garbage coordinate would land in the SQL surface
    looking like data.
    """
    return all(math.isfinite(v) for v in _coords(roi))


def _geometry(kind: str, shape: Mapping[str, Any]) -> Optional[ROI]:
    """One OME shape as a biopb ROI, with any transform baked in.

    None means the shape has no honest representation here -- a mask, or a
    transform that collapses it. The caller counts which.
    """
    m = _affine(shape.get("transform"))
    if m is not None and _det(m) == 0.0:
        return None

    def f(key: str) -> float:
        return float(shape.get(key) or 0.0)

    if kind in ("points", "labels"):
        return ROI(point=_pt(_map(m, f("x"), f("y"))))

    if kind == "lines":
        # No Line arm in roi.proto, and a 2-point polyline is the same geometry.
        # Only the markers are lost, and those are styling (kept in props).
        return ROI(
            polyline=Polyline(
                points=[
                    _pt(_map(m, f("x1"), f("y1"))),
                    _pt(_map(m, f("x2"), f("y2"))),
                ]
            )
        )

    if kind in ("polygons", "polylines"):
        pts = [_pt(_map(m, x, y)) for x, y in _points(shape.get("points"))]
        if not pts:
            return None
        if kind == "polygons":
            return ROI(polygon=Polygon(points=pts))
        # width stays 0: it is the brush band a scribble covers, and an OME
        # polyline has no such notion -- its stroke_width is display styling.
        return ROI(polyline=Polyline(points=pts))

    if kind == "rectangles":
        x, y, w, h = f("x"), f("y"), f("width"), f("height")
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if m is None or _axis_aligned(m):
            mapped = [_map(m, cx, cy) for cx, cy in corners]
            xs = [p[0] for p in mapped]
            ys = [p[1] for p in mapped]
            # min/max, not corners[0]/[2]: a negative scale flips the rectangle.
            return ROI(
                rectangle=Rectangle(
                    top_left=Point(x=min(xs), y=min(ys)),
                    bottom_right=Point(x=max(xs), y=max(ys)),
                )
            )
        # Rotation or shear makes it a general quadrilateral. Exact as a polygon;
        # only the *kind* is lost, and Rectangle has nowhere to put an angle.
        return ROI(polygon=Polygon(points=[_pt(_map(m, cx, cy)) for cx, cy in corners]))

    if kind == "ellipses":
        rx, ry, rot = _ellipse_axes(m, f("radius_x"), f("radius_y"))
        cx, cy = _map(m, f("x"), f("y"))
        return ROI(
            ellipse=Ellipse(
                center=Point(x=cx, y=cy), radius=Point(x=rx, y=ry), rotation=rot
            )
        )

    return None  # masks


def _plane(shape: Mapping[str, Any], axis_index: Mapping[str, int]) -> Dict[int, int]:
    """TheZ/TheT/TheC as {axis position: index}, omitting what is unset.

    An axis absent from the map applies at every index of it, which is exactly
    OME's own "unset means all planes".
    """
    pin: Dict[int, int] = {}
    for attr, axis in _PIN_ATTRS.items():
        value = shape.get(attr)
        idx = axis_index.get(axis)
        if value is not None and idx is not None and int(value) >= 0:
            pin[idx] = int(value)
    return pin


def _props(roi: Mapping[str, Any], shape: Mapping[str, Any]) -> Optional[str]:
    """Provenance and styling, namespaced so client props cannot collide."""
    ome = {
        "roi_id": roi.get("id"),
        "shape_id": shape.get("id"),
        "text": shape.get("text"),
        "stroke_color": shape.get("stroke_color"),
        "fill_color": shape.get("fill_color"),
    }
    ome = {k: v for k, v in ome.items() if v is not None}
    return json.dumps({"ome": ome}, sort_keys=True) if ome else None


# --- the import --------------------------------------------------------------


def _axis_index(dim_labels: Sequence[str]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for i, label in enumerate(dim_labels):
        axis = canonical_axis(label)
        # First wins: a duplicate label cannot be addressed unambiguously, and
        # guessing the later one would silently pin to the wrong axis.
        if axis is not None and axis not in index:
            index[axis] = i
    return index


Tensor = Tuple[str, Sequence[str]]


def tensors_by_field(tensors: Sequence[Tensor]) -> Dict[str, Tensor]:
    """Index tensors by the field half of their ``array_id``.

    For OME-TIFF that field IS the OME image id: ``_ome_scene_ids`` reads the
    ``Image`` ``ID`` attributes verbatim and ``array_id`` is
    ``f"{source_id}/{scene_id}"``. So the match is string equality, with no
    inference. Its one failure mode is that function's documented fallback to
    positional ``Image:{i}`` on a count mismatch, where the two id spaces
    disagree -- the misses are then counted, not assumed away.
    """
    return {array_id.partition("/")[2]: (array_id, dims) for array_id, dims in tensors}


def tensors_by_image_order(
    metadata: Mapping[str, Any], tensors: Sequence[Tensor]
) -> Dict[str, Tensor]:
    """Pair OME images to tensors by position.

    For a reader whose field names are its OWN scene names rather than the OME
    image ids -- bioio, whose ``array_id`` field comes from
    ``BioImage.scenes`` and can be a CZI scene label or an ND2 point name --
    equality on the id would match nothing at all. Position is the relation the
    adapter already relies on: ``_build_tensor_descriptors`` pairs
    ``ome_meta.images[i]`` with ``scene_ids[i]`` behind the same length guard
    used here.

    An unequal count means the correspondence is unknown, so nothing is paired
    rather than something being paired wrongly.
    """
    images = metadata.get("images") or []
    if not isinstance(images, (list, tuple)) or len(images) != len(tensors):
        return {}
    return {
        str(image["id"]): tensor
        for image, tensor in zip(images, tensors, strict=True)
        if isinstance(image, Mapping) and image.get("id") is not None
    }


def imported_annotations(
    metadata: Mapping[str, Any],
    by_image_id: Mapping[str, Tensor],
    *,
    content_version: Optional[bytes] = None,
    max_per_tensor: Optional[int] = None,
) -> Tuple[Dict[str, List[RoiAnnotation]], ImportReport]:
    """Annotations a source's OME metadata carries, keyed by ``array_id``.

    ``by_image_id`` maps an OME image id to ``(array_id, dim_labels)``. The
    caller resolves it -- with :func:`tensors_by_field` or
    :func:`tensors_by_image_order` -- because how a format's field names relate
    to its OME image ids is a fact about that format, not something this module
    can infer from the strings.

    ``content_version`` is stamped as ``drawn_against_version``, which for an
    imported row means what it says: the file revision this geometry was read
    from.
    """
    report = ImportReport()
    ome_rois = metadata.get("rois") or []
    if not isinstance(ome_rois, (list, tuple)):
        logger.warning("ome rois: `rois` is %s, not a list", type(ome_rois).__name__)
        return {}, report
    ome_rois = [r for r in ome_rois if isinstance(r, Mapping)]
    if not ome_rois:
        return {}, report

    by_scene = {
        image_id: (array_id, _axis_index(dim_labels))
        for image_id, (array_id, dim_labels) in by_image_id.items()
    }
    # Which images reference which ROI. The edge is per-image, so one OME ROI
    # shared by several images becomes a row on each -- the store anchors to one
    # array_id, and denormalising is the only way to say "on both".
    referenced_by: Dict[str, List[str]] = {}
    images = metadata.get("images") or []
    for image in images if isinstance(images, (list, tuple)) else []:
        if not isinstance(image, Mapping):
            continue
        scene = image.get("id")
        refs = image.get("roi_refs") or []
        for ref in refs if isinstance(refs, (list, tuple)) else []:
            if isinstance(ref, Mapping):
                referenced_by.setdefault(str(ref.get("id")), []).append(str(scene))

    out: Dict[str, List[RoiAnnotation]] = {}
    for ome_roi in ome_rois:
        scenes = referenced_by.get(str(ome_roi.get("id")))
        if not scenes:
            report.unreferenced += 1
            continue
        for scene in scenes:
            target = by_scene.get(scene)
            if target is None:
                report.unmatched_refs += 1
                continue
            array_id, axis_index = target
            rows = out.setdefault(array_id, [])
            for kind, shapes in _shape_lists(ome_roi, report):
                for shape in shapes:
                    if kind == "masks":
                        report.dropped_masks += 1
                        continue
                    if max_per_tensor is not None and len(rows) >= max_per_tensor:
                        report.over_cap += 1
                        continue
                    # One shape at a time, because this reads a file we did not
                    # write. A coordinate that is not a number, a TheZ outside
                    # uint32, a name that is not a string -- each raises, and
                    # this import runs inside source registration, so an
                    # unguarded one would cost the source its pixels over an
                    # annotation. Drop the shape, keep the rest, count it.
                    try:
                        ann = _annotation(
                            kind, ome_roi, shape, array_id, axis_index, content_version
                        )
                    except Exception:
                        logger.debug(
                            "ome rois: unreadable %s on %s",
                            kind,
                            array_id,
                            exc_info=True,
                        )
                        report.malformed += 1
                        continue
                    if ann is None:
                        report.dropped_degenerate += 1
                        continue
                    rows.append(ann)
                    report.imported += 1

    # Tensors whose every shape was dropped leave an empty list behind; an
    # empty import is no import, and the caller writes what it is given.
    out = {array_id: rows for array_id, rows in out.items() if rows}
    report.tensors = sorted(out)
    return out, report


def _shape_lists(
    ome_roi: Mapping[str, Any], report: ImportReport
) -> List[Tuple[str, Sequence[Mapping[str, Any]]]]:
    """``(kind, shapes)`` pairs from a ROI's union, skipping anything malformed.

    Structure is checked rather than assumed: this walks a dict parsed from a
    file, and `union` being a list or a shape list being a bare dict are both
    things a hand-rolled writer produces.
    """
    union = ome_roi.get("union")
    if not isinstance(union, Mapping):
        report.malformed += 1
        return []
    pairs = []
    for kind, shapes in union.items():
        if not isinstance(shapes, (list, tuple)):
            report.malformed += 1
            continue
        good = [s for s in shapes if isinstance(s, Mapping)]
        report.malformed += len(shapes) - len(good)
        if good:
            pairs.append((str(kind), good))
    return pairs


def _annotation(
    kind: str,
    ome_roi: Mapping[str, Any],
    shape: Mapping[str, Any],
    array_id: str,
    axis_index: Mapping[str, int],
    content_version: Optional[bytes],
) -> Optional[RoiAnnotation]:
    """One shape as a stored annotation, or None when it has no geometry here."""
    geometry = _geometry(kind, shape)
    if geometry is None or not _finite(geometry):
        return None
    ann = RoiAnnotation(
        roi_id=_roi_id(ome_roi, shape),
        array_id=array_id,
        set_name=OME_SET_NAME,
        # str(): a name is text in the schema but this dict came from a file,
        # and proto refuses a non-string outright.
        label=str(ome_roi.get("name") or shape.get("text") or ""),
        roi=geometry,
        props_json=_props(ome_roi, shape) or "",
        rev=1,
    )
    ann.plane.update(_plane(shape, axis_index))
    if content_version is not None:
        ann.drawn_against_version = content_version
    return ann


def _roi_id(ome_roi: Mapping[str, Any], shape: Mapping[str, Any]) -> str:
    """The OME shape id, which is unique within the document.

    Commas are replaced because the store refuses them: the sidecar deletes by a
    comma-separated `?ids=` list, so an id carrying one could be created and
    never addressed.
    """
    raw = shape.get("id") or ome_roi.get("id") or "shape"
    return str(raw).replace(",", "_")
