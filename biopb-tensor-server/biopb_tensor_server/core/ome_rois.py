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
    tensors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(
            self.imported
            or self.dropped_masks
            or self.dropped_degenerate
            or self.unmatched_refs
            or self.unreferenced
            or self.over_cap
        )

    def summary(self) -> str:
        parts = [f"{self.imported} imported across {len(self.tensors)} tensor(s)"]
        for label, n in (
            ("mask(s) dropped", self.dropped_masks),
            ("degenerate transform(s) dropped", self.dropped_degenerate),
            ("ref(s) matched no tensor", self.unmatched_refs),
            ("ROI(s) referenced by no image", self.unreferenced),
            ("dropped over the per-tensor cap", self.over_cap),
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


def imported_annotations(
    metadata: Mapping[str, Any],
    tensors: Sequence[Tuple[str, Sequence[str]]],
    *,
    content_version: Optional[bytes] = None,
    max_per_tensor: Optional[int] = None,
) -> Tuple[Dict[str, List[RoiAnnotation]], ImportReport]:
    """Annotations a source's OME metadata carries, keyed by ``array_id``.

    ``tensors`` is ``(array_id, dim_labels)`` per tensor -- both already on the
    source descriptor at registration. ``content_version`` is stamped as
    ``drawn_against_version``, which for an imported row means what it says: the
    file revision this geometry was read from.
    """
    report = ImportReport()
    ome_rois = metadata.get("rois") or []
    if not ome_rois:
        return {}, report

    by_scene = {
        array_id.partition("/")[2]: (array_id, _axis_index(dim_labels))
        for array_id, dim_labels in tensors
    }
    # Which images reference which ROI. The edge is per-image, so one OME ROI
    # shared by several images becomes a row on each -- the store anchors to one
    # array_id, and denormalising is the only way to say "on both".
    referenced_by: Dict[str, List[str]] = {}
    for image in metadata.get("images") or []:
        scene = image.get("id")
        for ref in image.get("roi_refs") or []:
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
            for kind, shapes in (ome_roi.get("union") or {}).items():
                for shape in shapes or []:
                    if kind == "masks":
                        report.dropped_masks += 1
                        continue
                    geometry = _geometry(kind, shape)
                    if geometry is None:
                        report.dropped_degenerate += 1
                        continue
                    if max_per_tensor is not None and len(rows) >= max_per_tensor:
                        report.over_cap += 1
                        continue
                    ann = RoiAnnotation(
                        roi_id=_roi_id(ome_roi, shape, len(rows)),
                        array_id=array_id,
                        set_name=OME_SET_NAME,
                        label=ome_roi.get("name") or shape.get("text") or "",
                        roi=geometry,
                        props_json=_props(ome_roi, shape) or "",
                        rev=1,
                    )
                    ann.plane.update(_plane(shape, axis_index))
                    if content_version is not None:
                        ann.drawn_against_version = content_version
                    rows.append(ann)
                    report.imported += 1

    # Tensors whose every shape was dropped leave an empty list behind; an
    # empty import is no import, and the caller writes what it is given.
    out = {array_id: rows for array_id, rows in out.items() if rows}
    report.tensors = sorted(out)
    return out, report


def _roi_id(ome_roi: Mapping[str, Any], shape: Mapping[str, Any], n: int) -> str:
    """The OME shape id, which is unique within the document.

    Commas are replaced because the store refuses them: the sidecar deletes by a
    comma-separated `?ids=` list, so an id carrying one could be created and
    never addressed.
    """
    raw = shape.get("id") or f"{ome_roi.get('id')}#{n}"
    return str(raw).replace(",", "_")
