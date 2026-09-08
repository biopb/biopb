"""Importing OME-embedded ROIs into the reserved @ome set (biopb/biopb#951).

Two halves. `ome_rois.imported_annotations` is pure -- metadata dict in,
annotations out -- so geometry and the affine bake-in are tested directly
against dicts shaped like an ome-types dump. The store half checks that a
registration files them, a re-registration replaces them, and removing the
source takes them with it, since that lifecycle is the whole design: an imported
set is scan output like the `sources` row, not durable user data.
"""

import json
import math

import pytest
from biopb.image.annotation_pb2 import RoiAnnotation
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor
from biopb_tensor_server.core.adapter_base import SourceAdapter
from biopb_tensor_server.core.metadata_db import MetadataDatabase
from biopb_tensor_server.core.ome_rois import OME_SET_NAME, imported_annotations

SOURCE_ID = "ometiff_a1b2c3"
ARRAY_0 = f"{SOURCE_ID}/Image:0"
ARRAY_1 = f"{SOURCE_ID}/Image:1"
DIMS = ["T", "C", "Z", "Y", "X"]
TENSORS = [(ARRAY_0, DIMS), (ARRAY_1, DIMS)]


def _shape(kind, **fields):
    base = {
        "id": fields.pop("id", "Shape:0"),
        "transform": fields.pop("transform", None),
    }
    base.update(fields)
    return kind, base


def _meta(*shapes, images=("Image:0",), roi_id="ROI:0", name=None, extra_images=()):
    """A metadata dict shaped like ome-types' model_dump(mode="json")."""
    union = {}
    for kind, shape in shapes:
        union.setdefault(kind, []).append(shape)
    return {
        "images": [{"id": image, "roi_refs": [{"id": roi_id}]} for image in images]
        + [{"id": image, "roi_refs": []} for image in extra_images],
        "rois": [{"id": roi_id, "name": name, "union": union}],
    }


def _one(metadata, tensors=TENSORS, **kwargs):
    out, report = imported_annotations(metadata, tensors, **kwargs)
    assert list(out) == [ARRAY_0], out
    (annotation,) = out[ARRAY_0]
    return annotation, report


class TestTheJoin:
    def test_a_ref_lands_on_the_matching_tensor(self):
        annotation, report = _one(_meta(_shape("points", x=3, y=4)))

        assert annotation.array_id == ARRAY_0
        assert annotation.set_name == OME_SET_NAME
        assert report.imported == 1

    def test_one_roi_on_two_images_becomes_a_row_on_each(self):
        out, report = imported_annotations(
            _meta(_shape("points", x=1, y=1), images=("Image:0", "Image:1")), TENSORS
        )

        assert sorted(out) == [ARRAY_0, ARRAY_1]
        assert report.imported == 2

    def test_a_roi_no_image_references_is_counted(self):
        metadata = _meta(_shape("points", x=1, y=1))
        metadata["images"] = [{"id": "Image:0", "roi_refs": []}]

        out, report = imported_annotations(metadata, TENSORS)
        assert out == {}
        assert report.unreferenced == 1

    def test_a_ref_to_an_unknown_scene_is_counted(self):
        """The `_ome_scene_ids` positional fallback is where this comes from."""
        out, report = imported_annotations(
            _meta(_shape("points", x=1, y=1), images=("Image:99",)), TENSORS
        )

        assert out == {}
        assert report.unmatched_refs == 1

    def test_no_rois_is_not_an_import(self):
        out, report = imported_annotations({"images": []}, TENSORS)
        assert out == {}
        assert not report


class TestShapeKinds:
    def test_point(self):
        annotation, _ = _one(_meta(_shape("points", x=3, y=4)))
        assert (annotation.roi.point.x, annotation.roi.point.y) == (3.0, 4.0)

    def test_rectangle(self):
        annotation, _ = _one(_meta(_shape("rectangles", x=1, y=2, width=4, height=6)))
        rect = annotation.roi.rectangle
        assert (rect.top_left.x, rect.top_left.y) == (1.0, 2.0)
        assert (rect.bottom_right.x, rect.bottom_right.y) == (5.0, 8.0)

    def test_ellipse(self):
        annotation, _ = _one(
            _meta(_shape("ellipses", x=10, y=20, radius_x=3, radius_y=4))
        )
        ellipse = annotation.roi.ellipse
        assert (ellipse.center.x, ellipse.center.y) == (10.0, 20.0)
        assert (ellipse.radius.x, ellipse.radius.y) == (3.0, 4.0)
        assert ellipse.rotation == 0.0

    def test_polygon(self):
        annotation, _ = _one(_meta(_shape("polygons", points="1,1 3,1 3,3")))
        assert [(p.x, p.y) for p in annotation.roi.polygon.points] == [
            (1.0, 1.0),
            (3.0, 1.0),
            (3.0, 3.0),
        ]

    def test_a_line_becomes_a_two_point_polyline(self):
        annotation, _ = _one(_meta(_shape("lines", x1=0, y1=0, x2=2, y2=5)))
        assert [(p.x, p.y) for p in annotation.roi.polyline.points] == [
            (0.0, 0.0),
            (2.0, 5.0),
        ]

    def test_a_label_becomes_a_point_carrying_its_text(self):
        annotation, _ = _one(_meta(_shape("labels", x=7, y=8, text="mitotic")))
        assert (annotation.roi.point.x, annotation.roi.point.y) == (7.0, 8.0)
        assert annotation.label == "mitotic"

    def test_a_polyline_has_no_geometric_width(self):
        """OME's stroke_width is display styling, not the band a brush covered."""
        annotation, _ = _one(
            _meta(_shape("polylines", points="0,0 1,1", stroke_width=9))
        )
        assert annotation.roi.polyline.width == 0.0

    def test_a_mask_is_dropped_and_counted(self):
        out, report = imported_annotations(
            _meta(_shape("masks", x=0, y=0, width=4, height=4)), TENSORS
        )
        assert out == {}
        assert report.dropped_masks == 1


class TestPlanePinning:
    def test_the_z_maps_to_an_axis_position(self):
        annotation, _ = _one(_meta(_shape("points", x=1, y=1, the_z=2, the_t=5)))
        # DIMS is TCZYX, so t is axis 0 and z is axis 2.
        assert dict(annotation.plane) == {0: 5, 2: 2}

    def test_an_unset_axis_is_absent_rather_than_zero(self):
        annotation, _ = _one(_meta(_shape("points", x=1, y=1, the_c=1)))
        assert dict(annotation.plane) == {1: 1}

    def test_index_zero_is_a_real_pin(self):
        annotation, _ = _one(_meta(_shape("points", x=1, y=1, the_t=0)))
        assert dict(annotation.plane) == {0: 0}

    def test_an_axis_the_tensor_lacks_is_skipped(self):
        annotation, _ = _one(
            _meta(_shape("points", x=1, y=1, the_z=3)), tensors=[(ARRAY_0, ["Y", "X"])]
        )
        assert dict(annotation.plane) == {}


class TestProvenance:
    def test_ome_ids_and_text_ride_in_props(self):
        annotation, _ = _one(
            _meta(_shape("points", x=1, y=1, id="Shape:7", text="note"))
        )
        assert json.loads(annotation.props_json)["ome"] == {
            "roi_id": "ROI:0",
            "shape_id": "Shape:7",
            "text": "note",
        }

    def test_the_shape_id_becomes_the_roi_id(self):
        annotation, _ = _one(_meta(_shape("points", x=1, y=1, id="Shape:7")))
        assert annotation.roi_id == "Shape:7"

    def test_content_version_is_what_it_was_read_from(self):
        annotation, _ = _one(
            _meta(_shape("points", x=1, y=1)), content_version=b"12345:678"
        )
        assert annotation.drawn_against_version == b"12345:678"

    def test_the_roi_name_wins_over_shape_text_as_the_label(self):
        annotation, _ = _one(
            _meta(_shape("points", x=1, y=1, text="shape"), name="roi")
        )
        assert annotation.label == "roi"


class TestTheCap:
    def test_the_remainder_is_dropped_and_counted(self):
        shapes = [_shape("points", x=i, y=0, id=f"Shape:{i}") for i in range(5)]
        out, report = imported_annotations(_meta(*shapes), TENSORS, max_per_tensor=2)

        assert len(out[ARRAY_0]) == 2
        assert report.imported == 2
        assert report.over_cap == 3


def _rotation(radians, scale_x=1.0, scale_y=1.0, tx=0.0, ty=0.0):
    cos, sin = math.cos(radians), math.sin(radians)
    return {
        "a00": cos * scale_x,
        "a01": -sin * scale_y,
        "a02": tx,
        "a10": sin * scale_x,
        "a11": cos * scale_y,
        "a12": ty,
    }


class TestTransforms:
    def test_a_translation_moves_a_point(self):
        annotation, _ = _one(
            _meta(
                _shape(
                    "points",
                    x=1,
                    y=2,
                    transform={
                        "a00": 1,
                        "a01": 0,
                        "a02": 10,
                        "a10": 0,
                        "a11": 1,
                        "a12": 20,
                    },
                )
            )
        )
        assert (annotation.roi.point.x, annotation.roi.point.y) == (11.0, 22.0)

    def test_an_identity_transform_is_a_no_op(self):
        annotation, _ = _one(
            _meta(_shape("points", x=1, y=2, transform=_rotation(0.0)))
        )
        assert (annotation.roi.point.x, annotation.roi.point.y) == (1.0, 2.0)

    def test_a_rotation_maps_every_polygon_vertex(self):
        annotation, _ = _one(
            _meta(
                _shape("polygons", points="1,0 0,1", transform=_rotation(math.pi / 2))
            )
        )
        got = [(p.x, p.y) for p in annotation.roi.polygon.points]
        assert got[0] == pytest.approx((0.0, 1.0), abs=1e-9)
        assert got[1] == pytest.approx((-1.0, 0.0), abs=1e-9)

    def test_an_axis_aligned_scale_keeps_a_rectangle(self):
        annotation, _ = _one(
            _meta(
                _shape(
                    "rectangles",
                    x=1,
                    y=1,
                    width=2,
                    height=4,
                    transform={
                        "a00": 2,
                        "a01": 0,
                        "a02": 0,
                        "a10": 0,
                        "a11": 3,
                        "a12": 0,
                    },
                )
            )
        )
        rect = annotation.roi.rectangle
        assert (rect.top_left.x, rect.top_left.y) == (2.0, 3.0)
        assert (rect.bottom_right.x, rect.bottom_right.y) == (6.0, 15.0)

    def test_a_flip_is_renormalised_rather_than_inverted(self):
        annotation, _ = _one(
            _meta(
                _shape(
                    "rectangles",
                    x=1,
                    y=1,
                    width=2,
                    height=2,
                    transform={
                        "a00": -1,
                        "a01": 0,
                        "a02": 0,
                        "a10": 0,
                        "a11": 1,
                        "a12": 0,
                    },
                )
            )
        )
        rect = annotation.roi.rectangle
        assert rect.top_left.x < rect.bottom_right.x
        assert (rect.top_left.x, rect.bottom_right.x) == (-3.0, -1.0)

    def test_a_rotated_rectangle_demotes_to_an_exact_polygon(self):
        annotation, _ = _one(
            _meta(
                _shape(
                    "rectangles",
                    x=0,
                    y=0,
                    width=2,
                    height=1,
                    transform=_rotation(math.pi / 2),
                )
            )
        )
        assert annotation.roi.WhichOneof("shape") == "polygon"
        got = [(p.x, p.y) for p in annotation.roi.polygon.points]
        assert got[0] == pytest.approx((0.0, 0.0), abs=1e-9)
        assert got[1] == pytest.approx((0.0, 2.0), abs=1e-9)
        assert got[2] == pytest.approx((-1.0, 2.0), abs=1e-9)

    @pytest.mark.parametrize(
        "transform",
        [
            _rotation(0.4),
            _rotation(0.4, scale_x=2.0, scale_y=0.5),
            _rotation(-1.1, scale_x=3.0, scale_y=3.0, tx=5.0, ty=-2.0),
            {"a00": 1, "a01": 0.6, "a02": 0, "a10": 0, "a11": 1, "a12": 0},  # shear
        ],
    )
    def test_an_ellipse_stays_an_exact_ellipse(self, transform):
        """Every transformed rim point must satisfy the reconstructed ellipse."""
        cx, cy, rx, ry = 10.0, -4.0, 3.0, 7.0
        annotation, _ = _one(
            _meta(
                _shape(
                    "ellipses",
                    x=cx,
                    y=cy,
                    radius_x=rx,
                    radius_y=ry,
                    transform=transform,
                )
            )
        )
        got = annotation.roi.ellipse
        m = transform
        for i in range(24):
            t = 2 * math.pi * i / 24
            px, py = cx + rx * math.cos(t), cy + ry * math.sin(t)
            qx = m["a00"] * px + m["a01"] * py + m["a02"]
            qy = m["a10"] * px + m["a11"] * py + m["a12"]
            # Back into the reconstructed ellipse's own frame.
            dx, dy = qx - got.center.x, qy - got.center.y
            cos, sin = math.cos(-got.rotation), math.sin(-got.rotation)
            u = (dx * cos - dy * sin) / got.radius.x
            v = (dx * sin + dy * cos) / got.radius.y
            assert u * u + v * v == pytest.approx(1.0, abs=1e-6)

    def test_a_singular_transform_drops_the_shape(self):
        out, report = imported_annotations(
            _meta(
                _shape(
                    "polygons",
                    points="0,0 1,1 2,0",
                    transform={
                        "a00": 1,
                        "a01": 1,
                        "a02": 0,
                        "a10": 2,
                        "a11": 2,
                        "a12": 0,
                    },
                )
            ),
            TENSORS,
        )
        assert out == {}
        assert report.dropped_degenerate == 1


class _FakeAdapter:
    """Just the surface sync_source_added touches."""

    def __init__(self, metadata, *, tensors=(("Image:0", DIMS),), version=b"1:2"):
        self._metadata = metadata
        self._tensors = tensors
        self.content_version = version
        self.released = False

    def get_embedded_rois(self, metadata, tensors, *, max_per_tensor=None):
        return imported_annotations(
            metadata,
            tensors,
            content_version=self.content_version,
            max_per_tensor=max_per_tensor,
        )

    def get_source_descriptor(self):
        return DataSourceDescriptor(
            source_id=SOURCE_ID,
            source_url="/data/exp.ome.tif",
            source_type="ome-tiff",
            tensors=[
                TensorDescriptor(
                    array_id=f"{SOURCE_ID}/{scene}",
                    dim_labels=dims,
                    shape=[1, 1, 1, 8, 8],
                    dtype="uint8",
                )
                for scene, dims in self._tensors
            ],
            data_resident=True,
        )

    def get_metadata(self):
        return self._metadata

    def release_registration_cache(self):
        self.released = True


def _reserved(db, array_id=ARRAY_0):
    rois, _ = db.list_rois(array_id, set_name=OME_SET_NAME)
    return sorted(r.roi_id for r in rois)


class TestRegistration:
    def test_a_registration_files_the_file_s_rois(self):
        db = MetadataDatabase()
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=2))))

        assert _reserved(db) == ["Shape:0"]
        (roi,), _ = db.list_rois(ARRAY_0)
        assert (roi.roi.point.x, roi.roi.point.y) == (1.0, 2.0)

    def test_the_rois_are_stripped_from_source_metadata(self):
        db = MetadataDatabase()
        metadata = _meta(_shape("points", x=1, y=2))
        metadata["creator"] = "tifffile"
        db.sync_source_added(SOURCE_ID, _FakeAdapter(metadata))

        stored = db.get_metadata_json(SOURCE_ID)
        assert "rois" not in stored
        assert stored["creator"] == "tifffile"  # everything else survives

    def test_a_re_registration_replaces_rather_than_accumulates(self):
        db = MetadataDatabase()
        db.sync_source_added(
            SOURCE_ID,
            _FakeAdapter(
                _meta(
                    _shape("points", x=1, y=1, id="Shape:0"),
                    _shape("points", x=2, y=2, id="Shape:1"),
                )
            ),
        )
        assert _reserved(db) == ["Shape:0", "Shape:1"]

        db.sync_source_added(
            SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=9, y=9, id="Shape:0")))
        )
        assert _reserved(db) == ["Shape:0"]
        (roi,), _ = db.list_rois(ARRAY_0)
        assert (roi.roi.point.x, roi.roi.point.y) == (9.0, 9.0)

    def test_a_file_that_loses_its_rois_loses_its_rows(self):
        db = MetadataDatabase()
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))
        db.sync_source_added(SOURCE_ID, _FakeAdapter({"images": [], "rois": []}))

        assert _reserved(db) == []

    def test_hand_drawn_rows_are_untouched_by_a_re_registration(self):
        db = MetadataDatabase()
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))
        db.put_rois(
            ARRAY_0, [RoiAnnotation(roi_id="mine", set_name="mine", roi=_point())]
        )

        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=5, y=5))))

        assert _reserved(db) == ["Shape:0"]
        assert [r.roi_id for r in db.list_rois(ARRAY_0, set_name="mine")[0]] == ["mine"]

    def test_removing_the_source_takes_the_import_but_not_the_user_s_work(self):
        db = MetadataDatabase()
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))
        db.put_rois(
            ARRAY_0, [RoiAnnotation(roi_id="mine", set_name="mine", roi=_point())]
        )

        db.sync_source_removed(SOURCE_ID)

        assert _reserved(db) == []
        assert [r.roi_id for r in db.list_rois(ARRAY_0)[0]] == ["mine"]

    def test_an_imported_row_is_read_only_end_to_end(self):
        db = MetadataDatabase()
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))

        with pytest.raises(ValueError, match="Shape:0"):
            db.put_rois(
                ARRAY_0,
                [RoiAnnotation(roi_id="Shape:0", set_name="mine", roi=_point())],
            )
        with pytest.raises(ValueError, match="reserved"):
            db.delete_rois(ARRAY_0, set_name=OME_SET_NAME)

    def test_imported_rows_do_not_spend_the_cap(self):
        db = MetadataDatabase(max_rois_per_tensor=2)
        db.sync_source_added(
            SOURCE_ID,
            _FakeAdapter(
                _meta(
                    _shape("points", x=1, y=1, id="Shape:0"),
                    _shape("points", x=2, y=2, id="Shape:1"),
                )
            ),
        )

        # The whole hand-drawn budget is still available beside them: the old
        # count would have made this the third and fourth row and refused.
        db.put_rois(
            ARRAY_0,
            [
                RoiAnnotation(roi_id="mine-1", set_name="mine", roi=_point()),
                RoiAnnotation(roi_id="mine-2", set_name="mine", roi=_point()),
            ],
        )
        assert _reserved(db) == ["Shape:0", "Shape:1"]
        mine, _ = db.list_rois(ARRAY_0, set_name="mine")
        assert sorted(r.roi_id for r in mine) == ["mine-1", "mine-2"]

    def test_the_cap_still_bounds_hand_drawn_rows(self):
        db = MetadataDatabase(max_rois_per_tensor=2)
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))

        with pytest.raises(ValueError, match="Annotation limit"):
            db.put_rois(
                ARRAY_0,
                [
                    RoiAnnotation(roi_id=f"mine-{i}", set_name="mine", roi=_point())
                    for i in range(3)
                ],
            )


def _point():
    from biopb.image import ROI, Point

    return ROI(point=Point(x=1, y=1))


class TestMalformedInput:
    """A file we did not write must not be able to fail source registration.

    Each of these raised before biopb/biopb#951 review: the import runs inside
    `sync_source_added`, so an unguarded raise costs a source its pixels over an
    annotation. They degrade per shape, so one bad shape cannot cost a file its
    other forty.
    """

    GOOD = {"id": "S:good", "x": 1, "y": 1}

    def _with(self, *shapes):
        union = {"points": list(shapes)}
        return {
            "images": [{"id": "Image:0", "roi_refs": [{"id": "ROI:0"}]}],
            "rois": [{"id": "ROI:0", "union": union}],
        }

    def test_a_coordinate_that_is_not_a_number(self):
        out, report = imported_annotations(
            self._with({"id": "S:0", "x": "left", "y": 1}, self.GOOD), TENSORS
        )
        assert [r.roi_id for r in out[ARRAY_0]] == ["S:good"]
        assert report.malformed == 1

    def test_a_plane_index_outside_uint32(self):
        out, report = imported_annotations(
            self._with({"id": "S:0", "x": 1, "y": 1, "the_z": 2**40}, self.GOOD),
            TENSORS,
        )
        assert [r.roi_id for r in out[ARRAY_0]] == ["S:good"]
        assert report.malformed == 1

    def test_a_name_that_is_not_a_string_is_coerced(self):
        metadata = self._with(self.GOOD)
        metadata["rois"][0]["name"] = 7

        annotation, _ = _one(metadata)
        assert annotation.label == "7"

    def test_an_infinite_coordinate_is_not_stored(self):
        """Nothing downstream would catch it: proto, DuckDB and bbox all take it."""
        out, report = imported_annotations(
            self._with({"id": "S:0", "x": float("inf"), "y": 1}, self.GOOD), TENSORS
        )
        assert [r.roi_id for r in out[ARRAY_0]] == ["S:good"]
        assert report.dropped_degenerate == 1

    @pytest.mark.parametrize(
        "union", [[], "polygons", {"points": {"id": "S:0"}}, {"points": ["nope"]}]
    )
    def test_a_union_that_is_not_shaped_like_one(self, union):
        out, report = imported_annotations(
            {
                "images": [{"id": "Image:0", "roi_refs": [{"id": "ROI:0"}]}],
                "rois": [{"id": "ROI:0", "union": union}],
            },
            TENSORS,
        )
        assert out == {}
        assert report.malformed >= 1

    @pytest.mark.parametrize("rois", [{"ROI:0": {}}, "ROI:0", ["ROI:0"]])
    def test_a_rois_field_that_is_not_a_list_of_mappings(self, rois):
        out, _ = imported_annotations({"images": [], "rois": rois}, TENSORS)
        assert out == {}

    @pytest.mark.parametrize("images", [{"Image:0": {}}, "Image:0", [None]])
    def test_an_images_field_that_is_not_a_list_of_mappings(self, images):
        out, _ = imported_annotations(
            {"images": images, "rois": [{"id": "ROI:0", "union": {}}]}, TENSORS
        )
        assert out == {}


class TestRegistrationSurvivesABadImport:
    def test_a_row_the_store_refuses_does_not_fail_registration(self):
        """An over-long roi_id: _prepare_roi rejects it, inside the write."""
        db = MetadataDatabase()
        db.sync_source_added(
            SOURCE_ID,
            _FakeAdapter(_meta(_shape("points", x=1, y=1, id="S:" + "x" * 400))),
        )

        assert db.get_metadata_json(SOURCE_ID) is not None
        assert _reserved(db) == []

    def test_an_adapter_hook_that_raises_does_not_fail_registration(self):
        """Documented as an adapter bug, and still not allowed to be fatal."""
        db = MetadataDatabase()
        adapter = _FakeAdapter(_meta(_shape("points", x=1, y=1)))
        adapter.get_embedded_rois = lambda *a, **k: 1 / 0

        db.sync_source_added(SOURCE_ID, adapter)

        assert _reserved(db) == []
        # The one remaining copy is not dropped when nothing read it.
        assert "rois" in db.get_metadata_json(SOURCE_ID)

    def test_a_failed_write_leaves_the_previous_set(self, monkeypatch):
        db = MetadataDatabase()
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))
        assert _reserved(db) == ["Shape:0"]

        def boom(*args, **kwargs):
            raise RuntimeError("write failed")

        monkeypatch.setattr(MetadataDatabase, "_replace_imported_locked", boom)
        db.sync_source_added(
            SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=9, y=9, id="Shape:1")))
        )

        assert _reserved(db) == ["Shape:0"]  # rolled back, not half-applied
        assert db.get_metadata_json(SOURCE_ID) is not None


class TestTheFormatDecides:
    """`rois` in a metadata dict means whatever that format meant by it.

    The server does not police get_metadata()'s contents, so reading one as
    OME-XML is the adapter's call, made by implementing `get_embedded_rois`.
    Not the key, and not `source_type` -- that is a name and it lies in both
    directions: `ome-zarr` carries NGFF rather than OME-XML, while `zeiss` /
    `leica` / `nikon` and the rest are ome-types dumps through bioio.
    """

    class _PlainAdapter(_FakeAdapter):
        """A format that stores something else under `rois`."""

        get_embedded_rois = SourceAdapter.get_embedded_rois

    def test_a_format_that_carries_nothing_is_not_parsed(self):
        db = MetadataDatabase()
        db.sync_source_added(
            SOURCE_ID, self._PlainAdapter(_meta(_shape("points", x=1, y=1)))
        )

        assert _reserved(db) == []

    def test_and_its_metadata_is_left_alone(self):
        """Nothing read them, so dropping the key would be plain data loss."""
        db = MetadataDatabase()
        db.sync_source_added(
            SOURCE_ID, self._PlainAdapter(_meta(_shape("points", x=1, y=1)))
        )

        assert "rois" in db.get_metadata_json(SOURCE_ID)

    def test_an_adapter_that_does_not_answer_at_all_is_fine(self, monkeypatch):
        """Most test doubles, and anything predating the hook."""
        db = MetadataDatabase()
        monkeypatch.delattr(_FakeAdapter, "get_embedded_rois")

        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))

        assert _reserved(db) == []

    def test_the_base_carries_nothing(self):
        assert SourceAdapter.get_embedded_rois(object(), {"rois": [1]}, []) == (
            {},
            None,
        )

    def test_which_real_adapters_implement_it(self):
        from biopb_tensor_server.adapters.bioio import _BioioAdapterBase
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        declares = lambda cls: "get_embedded_rois" in vars(cls)  # noqa: E731
        assert declares(OmeTiffAdapter)
        assert declares(_BioioAdapterBase)  # and so every vendor subclass
        # .zattrs is NGFF, not an ome-types dump -- the name is the trap.
        assert not declares(OmeZarrAdapter)
        assert not declares(ZarrAdapter)


class TestAnnotationsDisabled:
    """A server not serving the annotation actions does not import them."""

    def test_nothing_is_parsed(self):
        db = MetadataDatabase(annotations_enabled=False)
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))

        rows = db._get_connection().execute("SELECT count(*) FROM rois").fetchone()
        assert rows[0] == 0

    def test_and_the_metadata_keeps_them(self):
        """Nothing read them, so this is the only copy left."""
        db = MetadataDatabase(annotations_enabled=False)
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))

        assert "rois" in db.get_metadata_json(SOURCE_ID)

    def test_the_source_still_registers(self):
        db = MetadataDatabase(annotations_enabled=False)
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))

        assert db.get_metadata_json(SOURCE_ID) is not None


class TestOpenTimeClear:
    def test_a_catalog_from_a_newer_build_is_left_untouched(self, tmp_path):
        """The refusal promises exactly that, so the clear must come after it."""
        from biopb_tensor_server.core.errors import AnnotationStoreError
        from biopb_tensor_server.core.metadata_db import _ROI_SCHEMA_VERSION

        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))
        assert _reserved(db) == ["Shape:0"]
        db._get_connection().execute(
            "UPDATE catalog_meta SET value = ? WHERE key = 'roi_schema_version'",
            [str(_ROI_SCHEMA_VERSION + 1)],
        )
        db.close()

        with pytest.raises(AnnotationStoreError, match="untouched"):
            MetadataDatabase(store_path=store)._get_connection()

        # Reopened by a build that does understand it: the rows are still there.
        import duckdb

        conn = duckdb.connect(str(store))
        conn.execute(
            "UPDATE catalog_meta SET value = ? WHERE key = 'roi_schema_version'",
            [str(_ROI_SCHEMA_VERSION)],
        )
        rows = conn.execute("SELECT roi_id FROM rois").fetchall()
        conn.close()
        assert [r[0] for r in rows] == ["Shape:0"]

    def test_imported_rows_do_not_survive_a_restart(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.sync_source_added(SOURCE_ID, _FakeAdapter(_meta(_shape("points", x=1, y=1))))
        db.put_rois(
            ARRAY_0, [RoiAnnotation(roi_id="mine", set_name="mine", roi=_point())]
        )
        db.close()

        reopened = MetadataDatabase(store_path=store)
        assert _reserved(reopened) == []
        # Hand-drawn work is what persistence is for.
        assert [r.roi_id for r in reopened.list_rois(ARRAY_0)[0]] == ["mine"]
