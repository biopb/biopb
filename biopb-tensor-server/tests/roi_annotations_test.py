"""ROI annotations: the DuckDB store, the Flight actions, and the sidecar routes.

Design: ``docs/roi-annotations.md``. The store-level cases pin the decisions that
are easy to regress -- geometry acceptance, level-0 bbox derivation, plane pinning,
per-ROI rev/conflict, the cap -- while one gRPC round-trip and one FastAPI pass
exercise the wire in each direction.
"""

import json
import math
import threading
import time
from datetime import datetime, timedelta
from unittest import mock

import duckdb
import pyarrow.flight as flight
import pytest
from biopb.image import ROI, Ellipse, Mask, Point, Polygon, Polyline, Rectangle
from biopb.image.annotation_pb2 import RoiAnnotation, RoiPutRequest
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.core import metadata_db
from biopb_tensor_server.core.errors import AnnotationStoreError
from biopb_tensor_server.core.metadata_db import MetadataDatabase
from google.protobuf import json_format

ARRAY_ID = "zarr_a1b2c3/Image:0"

# Half-extent of a 3x4 ellipse turned 45 degrees: hypot(3, 4) / sqrt(2).
_HALF_DIAG = 5.0 / math.sqrt(2.0)


def _polygon(*pts):
    return ROI(polygon=Polygon(points=[Point(x=x, y=y) for x, y in pts]))


def _register_source(db, source_id, source_url):
    db._get_connection().execute(
        "INSERT INTO sources (source_id, source_url, source_type, tensors) "
        "VALUES (?, ?, ?, ?)",
        [source_id, source_url, "zarr", []],
    )


def _annotation(**kwargs):
    plane = kwargs.pop("plane", None)
    roi = kwargs.pop("roi", None) or _polygon((1, 2), (10, 2), (5, 9))
    ann = RoiAnnotation(roi=roi, **kwargs)
    if plane:
        ann.plane.update(plane)
    return ann


class TestStore:
    """MetadataDatabase.put_rois / list_rois / delete_rois."""

    def test_create_assigns_id_rev_and_default_set(self):
        db = MetadataDatabase()
        stored, conflicts = db.put_rois(ARRAY_ID, [_annotation(label="nucleus")])

        assert not conflicts
        assert len(stored[0].roi_id) == 32  # uuid4 hex
        assert stored[0].rev == 1
        assert stored[0].set_name == "default"
        assert stored[0].array_id == ARRAY_ID
        assert stored[0].created_at_unix_ms > 0

    def test_round_trip_preserves_geometry_and_plane(self):
        db = MetadataDatabase()
        # Keyed by 0-based axis position, not by label: dim_labels[2] and [0].
        ann = _annotation(label="nucleus", set_name="nuclei", plane={2: 12, 0: 0})
        db.put_rois(ARRAY_ID, [ann])

        rois, truncated = db.list_rois(ARRAY_ID)
        assert not truncated
        assert len(rois) == 1
        got = rois[0]
        assert got.label == "nucleus"
        assert dict(got.plane) == {2: 12, 0: 0}
        assert [(p.x, p.y) for p in got.roi.polygon.points] == [
            (1.0, 2.0),
            (10.0, 2.0),
            (5.0, 9.0),
        ]

    def test_update_bumps_rev_and_keeps_created_at(self):
        db = MetadataDatabase()
        (first,), _ = db.put_rois(ARRAY_ID, [_annotation(label="a")])
        first.label = "b"
        (second,), _ = db.put_rois(ARRAY_ID, [first])

        assert second.roi_id == first.roi_id
        assert second.rev == 2
        assert second.label == "b"
        assert second.created_at_unix_ms == first.created_at_unix_ms
        assert len(db.list_rois(ARRAY_ID)[0]) == 1

    def test_check_rev_rejects_a_stale_write_but_lands_the_rest(self):
        db = MetadataDatabase()
        (a,), _ = db.put_rois(ARRAY_ID, [_annotation(label="a")])
        (b,), _ = db.put_rois(ARRAY_ID, [_annotation(label="b")])
        # Someone else advances `a` to rev 2 while this client still holds rev 1.
        db.put_rois(ARRAY_ID, [a])

        a.label = "stale"
        b.label = "fresh"
        stored, conflicts = db.put_rois(ARRAY_ID, [a, b], check_rev=True)

        assert [c.roi_id for c in conflicts] == [a.roi_id]
        assert conflicts[0].stored_rev == 2
        assert [s.roi_id for s in stored] == [b.roi_id]
        by_id = {r.roi_id: r for r in db.list_rois(ARRAY_ID)[0]}
        assert by_id[a.roi_id].label != "stale"
        assert by_id[b.roi_id].label == "fresh"

    def test_last_writer_wins_without_check_rev(self):
        db = MetadataDatabase()
        (a,), _ = db.put_rois(ARRAY_ID, [_annotation(label="a")])
        db.put_rois(ARRAY_ID, [a])  # someone else -> rev 2

        a.label = "clobber"
        stored, conflicts = db.put_rois(ARRAY_ID, [a])
        assert not conflicts
        assert stored[0].label == "clobber"

    @pytest.mark.parametrize(
        "roi, expected",
        [
            (ROI(point=Point(x=3, y=4)), (3.0, 4.0, 3.0, 4.0)),
            (
                ROI(
                    rectangle=Rectangle(
                        top_left=Point(x=10, y=20), bottom_right=Point(x=2, y=8)
                    )
                ),
                (2.0, 8.0, 10.0, 20.0),  # normalized, whichever corner is which
            ),
            (
                ROI(ellipse=Ellipse(center=Point(x=10, y=10), radius=Point(x=3, y=4))),
                (7.0, 6.0, 13.0, 14.0),
            ),
            # A quarter turn exchanges the two half-extents ...
            (
                ROI(
                    ellipse=Ellipse(
                        center=Point(x=10, y=10),
                        radius=Point(x=3, y=4),
                        rotation=math.pi / 2,
                    )
                ),
                (6.0, 7.0, 14.0, 13.0),
            ),
            # ... and at 45 degrees the x extent grows past rx, which a bbox
            # taken from the radii alone would under-report by half a pixel.
            (
                ROI(
                    ellipse=Ellipse(
                        center=Point(x=0, y=0),
                        radius=Point(x=3, y=4),
                        rotation=math.pi / 4,
                    )
                ),
                (-_HALF_DIAG, -_HALF_DIAG, _HALF_DIAG, _HALF_DIAG),
            ),
            (_polygon((1, 2), (10, 2), (5, 9)), (1.0, 2.0, 10.0, 9.0)),
            # A zero-width scribble is its vertex extent ...
            (
                ROI(polyline=Polyline(points=[Point(x=1, y=2), Point(x=9, y=6)])),
                (1.0, 2.0, 9.0, 6.0),
            ),
            # ... and a fat one covers width/2 beyond it, because the stroke
            # width is geometry: it is the band of pixels the brush marked.
            (
                ROI(
                    polyline=Polyline(
                        points=[Point(x=1, y=2), Point(x=9, y=6)], width=4
                    )
                ),
                (-1.0, 0.0, 11.0, 8.0),
            ),
        ],
    )
    def test_bbox_is_derived_per_shape(self, roi, expected):
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(roi=roi)])
        # approx, not ==: a rotated bbox goes through cos/sin, and the float32
        # radii on the wire do not land on the exact decimal either way.
        stored = db._get_cursor().execute("SELECT bbox FROM rois").fetchone()[0]
        assert stored == pytest.approx(expected, abs=1e-6)

    def test_mask_and_mesh_are_refused(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="not accepted"):
            db.put_rois(ARRAY_ID, [_annotation(roi=ROI(mask=Mask()))])

    def test_empty_geometry_is_refused(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="no geometry"):
            db.put_rois(ARRAY_ID, [RoiAnnotation(label="nothing")])

    def test_polyline_round_trips_as_its_own_shape(self):
        """A scribble is not a polygon: it stays an open stroke through the store."""
        db = MetadataDatabase()
        scribble = ROI(
            polyline=Polyline(
                points=[Point(x=1, y=2), Point(x=5, y=3), Point(x=9, y=6)], width=3
            )
        )
        db.put_rois(ARRAY_ID, [_annotation(roi=scribble, label="scribble")])

        (got,), _ = db.list_rois(ARRAY_ID)
        assert got.roi.WhichOneof("shape") == "polyline"
        assert got.roi.polyline.width == 3
        assert len(got.roi.polyline.points) == 3
        assert (
            db._get_cursor().execute("SELECT shape_kind FROM rois").fetchone()[0]
            == "polyline"
        )

    def test_two_point_polyline_is_accepted(self):
        """A stroke needs only two points; the 3-point floor is a polygon rule."""
        db = MetadataDatabase()
        stroke = ROI(polyline=Polyline(points=[Point(x=1, y=1), Point(x=2, y=2)]))
        stored, _ = db.put_rois(ARRAY_ID, [_annotation(roi=stroke)])
        assert stored

    def test_single_point_polyline_is_refused(self):
        db = MetadataDatabase()
        stroke = ROI(polyline=Polyline(points=[Point(x=1, y=1)]))
        with pytest.raises(ValueError, match="at least 2 points"):
            db.put_rois(ARRAY_ID, [_annotation(roi=stroke)])

    def test_degenerate_polygon_is_refused(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="at least 3 points"):
            db.put_rois(ARRAY_ID, [_annotation(roi=_polygon((1, 2), (3, 4)))])

    def test_a_bad_shape_rejects_the_whole_batch(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError):
            db.put_rois(
                ARRAY_ID,
                [_annotation(label="good"), _annotation(roi=ROI(mesh=None))],
            )
        assert db.list_rois(ARRAY_ID)[0] == []

    def test_mismatched_array_id_is_refused(self):
        db = MetadataDatabase()
        ann = _annotation(array_id="zarr_other/Image:0")
        with pytest.raises(ValueError, match="does not match"):
            db.put_rois(ARRAY_ID, [ann])

    def test_cap_is_enforced_on_the_post_write_count(self):
        db = MetadataDatabase(max_rois_per_tensor=2)
        db.put_rois(ARRAY_ID, [_annotation(), _annotation()])
        with pytest.raises(ValueError, match="Annotation limit reached"):
            db.put_rois(ARRAY_ID, [_annotation()])
        # Updating what is already there stays legal at the cap.
        existing = db.list_rois(ARRAY_ID)[0]
        db.put_rois(ARRAY_ID, existing)

    def test_list_filters_by_set_and_delete_drops_a_layer(self):
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(set_name="nuclei")])
        db.put_rois(ARRAY_ID, [_annotation(set_name="scratch")])

        assert len(db.list_rois(ARRAY_ID)[0]) == 2
        assert len(db.list_rois(ARRAY_ID, "nuclei")[0]) == 1

        deleted = db.delete_rois(ARRAY_ID, set_name="scratch")
        assert len(deleted) == 1
        assert [r.set_name for r in db.list_rois(ARRAY_ID)[0]] == ["nuclei"]

    def test_delete_by_id_returns_only_what_was_there(self):
        db = MetadataDatabase()
        (a,), _ = db.put_rois(ARRAY_ID, [_annotation()])
        assert db.delete_rois(ARRAY_ID, [a.roi_id, "not-a-real-id"]) == [a.roi_id]

    def test_a_client_id_reused_on_another_tensor_does_not_clobber_it(self):
        """roi_id is unique per tensor, not globally.

        A client may name its own ids, so two tensors independently choosing
        "roi-1" is ordinary. With roi_id alone as the primary key, writing to one
        tensor silently destroyed the other's row: the create-or-update lookup is
        scoped by array_id and saw no conflict, while INSERT OR REPLACE hit the
        global key.
        """
        db = MetadataDatabase()
        other = "zarr_other/Image:0"
        db.put_rois(other, [_annotation(roi_id="roi-1", label="precious")])
        db.put_rois(ARRAY_ID, [_annotation(roi_id="roi-1", label="new")])

        assert [(r.roi_id, r.label) for r in db.list_rois(other)[0]] == [
            ("roi-1", "precious")
        ]
        assert [(r.roi_id, r.label) for r in db.list_rois(ARRAY_ID)[0]] == [
            ("roi-1", "new")
        ]

    def test_a_client_id_is_an_update_on_its_own_tensor(self):
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(roi_id="roi-1", label="first")])
        (second,), _ = db.put_rois(
            ARRAY_ID, [_annotation(roi_id="roi-1", label="second")]
        )
        assert second.rev == 2
        assert len(db.list_rois(ARRAY_ID)[0]) == 1

    def test_duplicate_ids_in_one_batch_are_refused(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="Duplicate roi_id"):
            db.put_rois(
                ARRAY_ID,
                [_annotation(roi_id="dup"), _annotation(roi_id="dup")],
            )
        assert db.list_rois(ARRAY_ID)[0] == []

    def test_an_overlong_client_id_is_refused(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="longer than"):
            db.put_rois(ARRAY_ID, [_annotation(roi_id="x" * 200)])

    def test_a_comma_in_a_client_id_is_refused(self):
        """The sidecar deletes by a comma-separated ?ids= list.

        An id containing a comma could be created but never addressed: it would
        split into two ids matching nothing, and the delete would report zero
        removals with no error at all.
        """
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="may not contain a comma"):
            db.put_rois(ARRAY_ID, [_annotation(roi_id="a,b")])

    def test_a_blank_client_id_gets_a_minted_one(self):
        db = MetadataDatabase()
        (stored,), _ = db.put_rois(ARRAY_ID, [_annotation(roi_id="   ")])
        assert len(stored.roi_id) == 32

    def test_deleting_by_id_touches_only_its_own_tensor(self):
        db = MetadataDatabase()
        other = "zarr_other/Image:0"
        db.put_rois(other, [_annotation(roi_id="roi-1")])
        db.put_rois(ARRAY_ID, [_annotation(roi_id="roi-1")])

        assert db.delete_rois(ARRAY_ID, ["roi-1"]) == ["roi-1"]
        assert len(db.list_rois(other)[0]) == 1

    @pytest.mark.parametrize("verb", ["put", "list", "delete"])
    def test_a_versioned_array_id_is_refused_at_the_store(self, verb):
        """The store is the layer that requires a bare id, so it enforces it.

        The versioned form (`source@token/field`) exists only above the Flight
        wire (identity policy, descriptor.proto). The read routes get this right
        only as a side effect of resolving a descriptor; a store keyed by
        array_id resolves nothing, so a missed strip used to be silent -- the
        annotation filed itself under a phantom tensor whose id is never minted
        again once content changes, and a bare read found nothing.
        """
        db = MetadataDatabase()
        versioned = "zarr_a1b2c3@9f1c4e2b/Image:0"
        calls = {
            "put": lambda: db.put_rois(versioned, [_annotation()]),
            "list": lambda: db.list_rois(versioned),
            "delete": lambda: db.delete_rois(versioned, ["a"]),
        }
        with pytest.raises(ValueError, match="content-version token"):
            calls[verb]()

    def test_an_at_sign_in_a_field_name_is_still_legal(self):
        """Only the source half is parsed: '@' is payload in a field name."""
        db = MetadataDatabase()
        db.put_rois("zarr_a1b2c3/we@ird", [_annotation()])
        assert len(db.list_rois("zarr_a1b2c3/we@ird")[0]) == 1

    def test_annotations_are_scoped_to_their_tensor(self):
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation()])
        db.put_rois("zarr_a1b2c3/Image:1", [_annotation()])
        assert len(db.list_rois(ARRAY_ID)[0]) == 1

    def test_rois_are_readable_on_the_sql_surface(self):
        db = MetadataDatabase()
        db.put_rois(
            ARRAY_ID, [_annotation(label="mitotic"), _annotation(label="mitotic")]
        )
        db.put_rois(ARRAY_ID, [_annotation(label="interphase")])

        info = db.handle_query(
            "SELECT label, count(*) AS n FROM rois GROUP BY label ORDER BY label"
        )
        ticket = info.endpoints[0].ticket.ticket.decode()
        table = db.get_pending_result(ticket)
        assert table.to_pydict() == {"label": ["interphase", "mitotic"], "n": [1, 2]}

    def test_a_filtered_query_is_not_reported_as_truncated(self):
        """Truncation is the server's own flag, not a difference of counts.

        `total_sources` counts the CATALOG, so differencing it against the rows
        a query returned reported truncation for every filtered query and for
        any query against a table other than `sources` -- 3 rows matching out of
        100 sources read as "97 were dropped".
        """
        db = MetadataDatabase()
        for i in range(20):
            _register_source(db, f"zarr_{i}", f"/data/{i}.zarr")
        db.put_rois(ARRAY_ID, [_annotation(label="nucleus")])

        for sql in (
            "SELECT source_id FROM sources WHERE source_id = 'zarr_1'",
            "SELECT roi_id FROM rois",
            "SELECT roi_id FROM rois WHERE label = 'nothing-matches'",
        ):
            md = db.handle_query(sql).schema.metadata
            assert md[b"truncated"] == b"False", sql

    def test_a_real_truncation_is_reported(self):
        db = MetadataDatabase(max_query_results=5)
        for i in range(12):
            _register_source(db, f"zarr_{i}", f"/data/{i}.zarr")
        md = db.handle_query("SELECT source_id FROM sources").schema.metadata
        assert md[b"truncated"] == b"True"
        assert (md[b"total_rows"], md[b"returned_rows"]) == (b"12", b"5")

    def test_sql_surface_stays_read_only(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="forbidden keyword"):
            db.handle_query("DELETE FROM rois")

    def test_source_url_is_captured_for_a_registered_source(self, tmp_path):
        """The orphan-report anchor: array_id is a hash and cannot be inverted."""
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")
        db.put_rois(ARRAY_ID, [_annotation()])
        row = (
            db._get_cursor()
            .execute("SELECT source_url, last_seen_at IS NOT NULL FROM rois")
            .fetchone()
        )
        assert row == ("/data/exp.zarr", True)

    def test_an_unknown_source_still_accepts_annotations(self):
        """Progressive discovery: catalog absence proves nothing about the image.

        Refusing the write would turn a rescan window into lost work, so the row
        lands without a URL -- and is backfilled once the source appears.
        """
        db = MetadataDatabase()
        stored, _ = db.put_rois(ARRAY_ID, [_annotation()])
        assert stored
        assert (
            db._get_cursor().execute("SELECT source_url FROM rois").fetchone()[0]
            is None
        )

    def test_a_null_source_url_is_backfilled_once_the_source_appears(self):
        """A permanent NULL is the one row the staleness model cannot cope with.

        array_id is a SHA-256, so an orphan report could only name such a row as
        "zarr_a1b2c3" and a re-attach prompt would have nothing to offer. Only
        rows caught in a later batch used to heal, leaving the rest NULL forever.
        """
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(roi_id="a"), _annotation(roi_id="b")])
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")

        # A write to a SIBLING tensor heals every row of the source.
        db.put_rois("zarr_a1b2c3/Image:1", [_annotation(roi_id="c")])
        assert (
            db._get_cursor()
            .execute("SELECT count(*) FROM rois WHERE source_url = '/data/exp.zarr'")
            .fetchone()[0]
            == 3
        )

    def test_an_edit_while_the_source_is_absent_keeps_what_the_row_knew(self):
        """A REPLACE rewrites the whole row, so catalog-derived columns must not
        be written when the catalog cannot answer.

        Editing an annotation during a rescan window (or with the drive
        unmounted) used to wipe a good source_url -- losing the only thing that
        can name the image in an orphan report -- and stamp last_seen_at as if
        the tensor had just been observed, resetting the orphan clock for an
        image that may really be gone.
        """
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")
        db.put_rois(ARRAY_ID, [_annotation(roi_id="a", label="v1")])
        seen_before = (
            db._get_cursor().execute("SELECT last_seen_at FROM rois").fetchone()[0]
        )

        db._get_connection().execute(
            "DELETE FROM sources WHERE source_id = ?", ["zarr_a1b2c3"]
        )
        db.put_rois(ARRAY_ID, [_annotation(roi_id="a", label="v2")])

        url, seen_after = (
            db._get_cursor()
            .execute("SELECT source_url, last_seen_at FROM rois")
            .fetchone()
        )
        assert url == "/data/exp.zarr"
        assert seen_after == seen_before
        assert db.list_rois(ARRAY_ID)[0][0].label == "v2"  # the edit still landed

    def test_a_row_created_without_a_catalog_has_no_false_sighting(self):
        """last_seen_at means "observed in the catalog", so it stays NULL here."""
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation()])
        assert (
            db._get_cursor().execute("SELECT last_seen_at FROM rois").fetchone()[0]
            is None
        )

    def test_a_mixed_batch_creates_and_updates_in_one_call(self):
        """Creates and updates take different statements now; one batch drives both."""
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")
        db.put_rois(ARRAY_ID, [_annotation(roi_id="a", label="v1")])

        stored, conflicts = db.put_rois(
            ARRAY_ID,
            [_annotation(roi_id="a", label="v2"), _annotation(roi_id="b", label="new")],
        )
        assert not conflicts
        assert {(s.roi_id, s.label, s.rev) for s in stored} == {
            ("a", "v2", 2),
            ("b", "new", 1),
        }
        assert len(db.list_rois(ARRAY_ID)[0]) == 2

    def test_an_update_while_present_records_a_fresh_sighting(self):
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")
        db.put_rois(ARRAY_ID, [_annotation(roi_id="a")])
        before = db._get_cursor().execute("SELECT last_seen_at FROM rois").fetchone()[0]

        db.put_rois(ARRAY_ID, [_annotation(roi_id="a")])
        after = db._get_cursor().execute("SELECT last_seen_at FROM rois").fetchone()[0]
        assert after >= before

    def test_a_renamed_source_refreshes_every_row(self):
        """The label follows the catalog; it does not freeze at the first write.

        A source's display url moves without its identity moving -- an `alias`
        re-roots a local source, a drag-drop stamps `dnd://`. `source_id` is
        unchanged, so these rows are the same annotations on the same image, and
        a report that named them by the pre-rename url would name something the
        catalog no longer lists.
        """
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")
        db.put_rois(ARRAY_ID, [_annotation(roi_id="a")])
        db._get_connection().execute(
            "UPDATE sources SET source_url = ? WHERE source_id = ?",
            ["dnd://exp.zarr", "zarr_a1b2c3"],
        )
        db.put_rois(ARRAY_ID, [_annotation(roi_id="b")])

        urls = (
            db._get_cursor()
            .execute("SELECT roi_id, source_url FROM rois ORDER BY roi_id")
            .fetchall()
        )
        assert urls == [("a", "dnd://exp.zarr"), ("b", "dnd://exp.zarr")]

    @pytest.mark.parametrize("unnamed", [None, ""])
    def test_a_catalog_row_that_names_nothing_leaves_the_label_alone(self, unnamed):
        """An unnamed source is not a rename.

        Refreshing from it would wipe the only human-readable thing an orphan
        report has -- and `sources.source_url` is nullable, with the descriptor
        path writing "" when a source carries no url of its own.
        """
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")
        db.put_rois(ARRAY_ID, [_annotation(roi_id="a")])
        db._get_connection().execute(
            "UPDATE sources SET source_url = ? WHERE source_id = ?",
            [unnamed, "zarr_a1b2c3"],
        )
        db.put_rois(ARRAY_ID, [_annotation(roi_id="b")])

        urls = (
            db._get_cursor()
            .execute("SELECT roi_id, source_url FROM rois ORDER BY roi_id")
            .fetchall()
        )
        # "a" keeps what it knew; "b" is created while the catalog names
        # nothing, so it lands NULL for a later sighting to backfill -- a new
        # row cannot inherit a label the catalog is no longer offering.
        assert urls == [("a", "/data/exp.zarr"), ("b", None)]


class TestBatchAtomicity:
    """A batch is all-or-nothing, to a failure and to a concurrent reader.

    The write lock only serializes writers -- DuckDB autocommits each statement,
    so without an explicit transaction a mid-batch failure left the rows written
    so far behind, and a reader (list_rois takes no lock, by design) watched a
    layer appear row by row.
    """

    @staticmethod
    def _fail_on_nth(monkeypatch, n):
        from biopb_tensor_server.core import metadata_db as m

        original = m._PreparedRoi.column_values
        calls = {"n": 0}

        def boom(self, columns):
            calls["n"] += 1
            if calls["n"] == n:
                raise RuntimeError("db hiccup")
            return original(self, columns)

        monkeypatch.setattr(m._PreparedRoi, "column_values", boom)

    def test_a_failed_batch_leaves_nothing_behind(self, monkeypatch):
        db = MetadataDatabase()
        self._fail_on_nth(monkeypatch, 4)
        with pytest.raises(RuntimeError, match="db hiccup"):
            db.put_rois(ARRAY_ID, [_annotation(roi_id=str(i)) for i in range(6)])
        assert db.list_rois(ARRAY_ID)[0] == []

    def test_the_connection_survives_a_rolled_back_batch(self, monkeypatch):
        """A BEGIN left open would poison the shared connection for every writer."""
        db = MetadataDatabase()
        self._fail_on_nth(monkeypatch, 2)
        with pytest.raises(RuntimeError):
            db.put_rois(ARRAY_ID, [_annotation(roi_id=str(i)) for i in range(3)])
        monkeypatch.undo()
        stored, _ = db.put_rois(
            ARRAY_ID, [_annotation(roi_id=str(i)) for i in range(3)]
        )
        assert len(stored) == 3

    def test_a_failed_batch_does_not_disturb_what_was_already_there(self, monkeypatch):
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(roi_id="keep", label="v1")])
        self._fail_on_nth(monkeypatch, 2)
        with pytest.raises(RuntimeError):
            db.put_rois(
                ARRAY_ID,
                [_annotation(roi_id="keep", label="v2"), _annotation(roi_id="new")],
            )
        (survivor,) = db.list_rois(ARRAY_ID)[0]
        assert (survivor.roi_id, survivor.label, survivor.rev) == ("keep", "v1", 1)

    def test_a_reader_never_sees_half_a_batch(self):
        """list_rois uses its own cursor and takes no lock, so isolation has to
        come from the transaction, not from mutual exclusion."""
        import threading

        db = MetadataDatabase()
        rois = [_annotation(roi_id=str(i)) for i in range(40)]
        seen = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                seen.append(len(db.list_rois(ARRAY_ID)[0]))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            db.put_rois(ARRAY_ID, rois)
        finally:
            stop.set()
            t.join(timeout=5)

        assert seen, "reader never ran"
        assert set(seen) <= {0, 40}, f"observed a partial batch: {sorted(set(seen))}"


class TestFlightActions:
    """One full server -> client gRPC round-trip over the three actions."""

    def test_round_trip(self):
        from biopb.tensor import TensorFlightClient

        db = MetadataDatabase()
        server = TensorFlightServer("grpc://localhost:0", metadata_db=db)
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            assert "roi_put" in {a.type for a in client._state.client.list_actions()}

            put = client.put_rois(ARRAY_ID, [_annotation(label="nucleus")])
            assert len(put.stored) == 1 and put.stored[0].rev == 1
            roi_id = put.stored[0].roi_id

            listed = client.list_rois(ARRAY_ID)
            assert not listed.truncated
            assert [r.roi_id for r in listed.rois] == [roi_id]
            assert listed.rois[0].label == "nucleus"

            # A rejected geometry comes back as a server error, not a silent drop.
            with pytest.raises(flight.FlightServerError, match="not accepted"):
                client.put_rois(ARRAY_ID, [_annotation(roi=ROI(mask=Mask()))])

            assert client.delete_rois(ARRAY_ID, [roi_id]).deleted == [roi_id]
            assert client.list_rois(ARRAY_ID).rois == []
            client.close()
        finally:
            server.shutdown()

    def test_disabled_server_reports_unavailable(self):
        from biopb.tensor import TensorFlightClient

        server = TensorFlightServer(
            "grpc://localhost:0",
            metadata_db=MetadataDatabase(),
            annotations_enabled=False,
        )
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(1)
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")
            with pytest.raises(flight.FlightUnavailableError, match="disabled"):
                client.list_rois(ARRAY_ID)
            client.close()
        finally:
            server.shutdown()


class TestSidecarRoutes:
    """The SPA's path: proto3 JSON in and out, version token stripped."""

    @pytest.fixture
    def client_and_app(self, monkeypatch):
        from biopb_tensor_server.serving import http_server
        from fastapi.testclient import TestClient

        db = MetadataDatabase()

        class _FakeFlightClient:
            def list_rois(self, array_id, set_name=""):
                from biopb.image.annotation_pb2 import RoiListResult

                rois, truncated = db.list_rois(array_id, set_name)
                return RoiListResult(rois=rois, truncated=truncated)

            def put_rois(self, array_id, rois, *, check_rev=False):
                from biopb.image.annotation_pb2 import RoiPutResult

                try:
                    stored, conflicts = db.put_rois(array_id, rois, check_rev=check_rev)
                except ValueError as e:
                    raise flight.FlightServerError(str(e))
                return RoiPutResult(stored=stored, conflicts=conflicts)

            def delete_rois(self, array_id, roi_ids=(), set_name=""):
                from biopb.image.annotation_pb2 import RoiDeleteResult

                return RoiDeleteResult(
                    deleted=db.delete_rois(array_id, roi_ids, set_name)
                )

        app = http_server.create_app("grpc://127.0.0.1:1", None, 0, [], None, False)
        monkeypatch.setattr(
            app.state.sidecar, "get_client", lambda: _FakeFlightClient()
        )
        return TestClient(app), db

    def test_post_then_get_round_trips_proto3_json(self, client_and_app):
        client, _db = client_and_app
        body = {
            "rois": [
                {
                    "label": "nucleus",
                    "setName": "nuclei",
                    "plane": {"2": "12"},
                    "roi": {
                        "polygon": {
                            "points": [
                                {"x": 1, "y": 2},
                                {"x": 10, "y": 2},
                                {"x": 5, "y": 9},
                            ]
                        }
                    },
                }
            ]
        }
        resp = client.post(f"/api/rois/{ARRAY_ID}", json=body)
        assert resp.status_code == 200, resp.text
        stored = resp.json()["stored"][0]
        assert stored["label"] == "nucleus" and stored["rev"] == "1"

        got = client.get(f"/api/rois/{ARRAY_ID}").json()
        assert len(got["rois"]) == 1
        # A map key is always a JSON string; a uint32 VALUE is a number, since
        # proto3 JSON stringifies only the 64-bit integer types.
        assert got["rois"][0]["plane"] == {"2": 12}
        assert len(got["rois"][0]["roi"]["polygon"]["points"]) == 3

    def test_version_token_is_stripped_on_write_and_restored_on_read(
        self, client_and_app
    ):
        """Annotations anchor on the unversioned id so they survive a content edit."""
        client, db = client_and_app
        versioned = "zarr_a1b2c3@9f1c4e2b/Image:0"
        body = {
            "rois": [
                {"roi": {"point": {"x": 1, "y": 2}}},
            ]
        }
        resp = client.post(f"/api/rois/{versioned}", json=body)
        assert resp.status_code == 200, resp.text
        # Stored bare ...
        assert [r.array_id for r in db.list_rois(ARRAY_ID)[0]] == [ARRAY_ID]
        # ... and echoed back in the form the SPA addresses tensors with.
        assert resp.json()["stored"][0]["arrayId"] == versioned
        got = client.get(f"/api/rois/{versioned}").json()
        assert got["rois"][0]["arrayId"] == versioned
        # A read at a *different* content version still finds them.
        other = client.get("/api/rois/zarr_a1b2c3@deadbeef/Image:0").json()
        assert len(other["rois"]) == 1

    def test_a_read_edit_write_round_trip_survives_the_version_token(
        self, client_and_app
    ):
        """The shape a real client actually uses: GET, change a field, POST back.

        Responses carry versioned array_ids, so the body handed back on the write
        carries one too -- and the store only ever sees bare ids. Stripping the
        path alone was not enough; the whole round trip 422'd.
        """
        client, _db = client_and_app
        versioned = "zarr_a1b2c3@9f1c4e2b/Image:0"
        client.post(
            f"/api/rois/{versioned}",
            json={"rois": [{"roi": {"point": {"x": 1, "y": 2}}}]},
        )

        fetched = client.get(f"/api/rois/{versioned}").json()["rois"][0]
        assert fetched["arrayId"] == versioned
        fetched["label"] = "edited"

        resp = client.post(f"/api/rois/{versioned}", json={"rois": [fetched]})
        assert resp.status_code == 200, resp.text
        stored = resp.json()["stored"][0]
        assert stored["label"] == "edited"
        assert stored["rev"] == "2"  # an update, not a second row
        assert len(client.get(f"/api/rois/{versioned}").json()["rois"]) == 1

    def test_a_genuinely_wrong_tensor_in_the_body_is_still_refused(
        self, client_and_app
    ):
        """Stripping the version must not blunt the mismatch check itself."""
        client, _db = client_and_app
        resp = client.post(
            f"/api/rois/{ARRAY_ID}",
            json={
                "rois": [
                    {
                        "arrayId": "zarr_somethingelse/Image:0",
                        "roi": {"point": {"x": 1, "y": 2}},
                    }
                ]
            },
        )
        assert resp.status_code == 422
        assert "does not match" in resp.json()["detail"]

    def test_rejected_geometry_is_422(self, client_and_app):
        client, _db = client_and_app
        resp = client.post(
            f"/api/rois/{ARRAY_ID}",
            json={"rois": [{"roi": {"mask": {}}}]},
        )
        assert resp.status_code == 422
        assert "not accepted" in resp.json()["detail"]

    def test_malformed_body_is_422(self, client_and_app):
        client, _db = client_and_app
        resp = client.post(f"/api/rois/{ARRAY_ID}", json={"rois": [{"nope": 1}]})
        assert resp.status_code == 422

    def test_delete_by_ids_and_by_set(self, client_and_app):
        client, db = client_and_app
        (a,), _ = db.put_rois(ARRAY_ID, [_annotation(set_name="nuclei")])
        (b,), _ = db.put_rois(ARRAY_ID, [_annotation(set_name="scratch")])

        resp = client.delete(f"/api/rois/{ARRAY_ID}?ids={a.roi_id}")
        assert resp.json()["deleted"] == [a.roi_id]

        resp = client.delete(f"/api/rois/{ARRAY_ID}?set=scratch")
        assert resp.json()["deleted"] == [b.roi_id]
        assert db.list_rois(ARRAY_ID)[0] == []

    def test_a_string_check_rev_is_refused_not_coerced(self, client_and_app):
        """bool("false") is True, so coercing would silently turn conditional
        writes ON for a client that sent the string."""
        client, _db = client_and_app
        resp = client.post(
            f"/api/rois/{ARRAY_ID}",
            json={"rois": [], "check_rev": "false"},
        )
        assert resp.status_code == 422

    def test_cross_origin_write_is_refused(self, client_and_app):
        client, _db = client_and_app
        resp = client.post(
            f"/api/rois/{ARRAY_ID}",
            json={"rois": []},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert resp.status_code == 403


class TestPositionalPlanePin:
    """The pin is keyed by wire axis index, which is what lets it address an
    axis the labels cannot name (biopb#935's sibling: a TIFF sequence's opaque
    file axis, or two axes sharing a label)."""

    def test_an_unlabelled_axis_can_be_pinned(self):
        db = MetadataDatabase()
        # Axis 0 of a tensor whose labels are ("", "z", "y", "x"): under a
        # label-keyed pin this axis had no key at all and every annotation on it
        # broadcast across every index.
        db.put_rois(ARRAY_ID, [_annotation(plane={0: 3})])
        rois, _ = db.list_rois(ARRAY_ID)
        assert dict(rois[0].plane) == {0: 3}

    def test_two_axes_sharing_a_label_stay_distinct(self):
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(plane={0: 1, 1: 7})])
        rois, _ = db.list_rois(ARRAY_ID)
        assert dict(rois[0].plane) == {0: 1, 1: 7}

    def test_an_empty_pin_still_means_every_index(self):
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(plane={})])
        rois, _ = db.list_rois(ARRAY_ID)
        assert dict(rois[0].plane) == {}

    def test_a_negative_axis_cannot_even_be_built(self):
        # uint32 key: protobuf refuses it at assignment, so the store never
        # needs a check and no binding can smuggle one past.
        with pytest.raises(ValueError):
            _annotation(plane={-1: 0})

    def test_a_negative_axis_over_the_wire_is_a_422(self):
        # The sidecar maps a ParseError to 422, so this is the client-facing
        # rejection: no route needs its own guard.
        with pytest.raises(json_format.ParseError):
            json_format.ParseDict({"rois": [{"plane": {"-1": "0"}}]}, RoiPutRequest())

    def test_a_negative_index_cannot_be_built_either(self):
        # Unsigned on both halves. A negative index is not harmless nonsense:
        # it matches no real index, so an annotation carrying one would be
        # hidden on every plane rather than shown on all of them.
        with pytest.raises(ValueError):
            _annotation(plane={0: -5})

    def test_a_negative_index_over_the_wire_is_a_422(self):
        with pytest.raises(json_format.ParseError):
            json_format.ParseDict({"rois": [{"plane": {"0": "-5"}}]}, RoiPutRequest())

    def test_an_out_of_range_axis_is_accepted_and_simply_matches_nothing(self):
        # The write path binds no tensor, so it has no rank to check against.
        # Storing it is harmless: no reader is ever on axis 99.
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation(plane={99: 0})])
        rois, _ = db.list_rois(ARRAY_ID)
        assert dict(rois[0].plane) == {99: 0}


class TestPersistence:
    """A file-backed catalog: what survives a restart, and what deliberately does not."""

    def test_annotations_survive_a_reopen(self, tmp_path):
        store = tmp_path / "nested" / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        (written,), _ = db.put_rois(
            ARRAY_ID, [_annotation(label="nucleus", set_name="nuclei", plane={2: 12})]
        )
        db.close()

        back, _ = MetadataDatabase(store_path=store).list_rois(ARRAY_ID)
        assert [(r.roi_id, r.label, r.set_name, dict(r.plane)) for r in back] == [
            (written.roi_id, "nucleus", "nuclei", {2: 12})
        ]
        assert back[0].roi == written.roi
        assert back[0].rev == written.rev

    def test_without_a_store_path_nothing_persists(self, tmp_path):
        # The default is unchanged: persistence is something a caller asks for.
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation()])
        db.close()
        assert MetadataDatabase().list_rois(ARRAY_ID) == ([], False)

    def test_sources_are_dropped_on_reopen(self, tmp_path):
        # `sources` rides in the same file only because DuckDB has one database
        # per connection. It is scan output, and a row for a source deleted
        # while the server was down would otherwise advertise an unservable
        # tensor forever.
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        _register_source(db, "zarr_a1b2c3", "file:///data/a.zarr")
        db.put_rois(ARRAY_ID, [_annotation()])
        db.close()

        db = MetadataDatabase(store_path=store)
        assert db._get_cursor().execute("SELECT count(*) FROM sources").fetchone() == (
            0,
        )
        assert len(db.list_rois(ARRAY_ID)[0]) == 1

    def test_derived_columns_are_recomputed_from_the_geometry(self, tmp_path):
        # bbox and shape_kind are _roi_bbox output cached in columns, so a
        # change to the formula has to land on restart rather than needing a
        # migration -- adding rotation to Ellipse moved every rotated one.
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.put_rois(
            ARRAY_ID,
            [
                _annotation(
                    roi=ROI(
                        ellipse=Ellipse(
                            center=Point(x=10, y=10),
                            radius=Point(x=3, y=4),
                            rotation=math.pi / 2,
                        )
                    )
                )
            ],
        )
        # Stand in for a row written by the older formula.
        db._get_connection().execute(
            "UPDATE rois SET bbox = [7.0, 6.0, 13.0, 14.0], shape_kind = 'polygon'"
        )
        db.close()

        db = MetadataDatabase(store_path=store)
        kind, bbox = (
            db._get_cursor().execute("SELECT shape_kind, bbox FROM rois").fetchone()
        )
        assert kind == "ellipse"
        # The quarter turn exchanges the half-extents, as _roi_bbox now says.
        assert bbox == pytest.approx((6.0, 7.0, 14.0, 13.0), abs=1e-6)

    def test_an_unreadable_geometry_keeps_its_row(self, tmp_path):
        # Deleting would turn an annotation we cannot re-derive a bbox for into
        # one the user simply lost.
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.put_rois(ARRAY_ID, [_annotation(label="keep me")])
        db._get_connection().execute("UPDATE rois SET geometry = 'not json'")
        db.close()

        db = MetadataDatabase(store_path=store)
        assert db._get_cursor().execute("SELECT label FROM rois").fetchall() == [
            ("keep me",)
        ]

    def test_an_unopenable_store_is_fatal_and_the_file_is_untouched(self, tmp_path):
        # Never a fallback to memory: the alternative to refusing is serving
        # while every ROI drawn goes somewhere that disappears at the next
        # restart. And never a rename -- a held lock raises the same
        # IOException as corruption, and moving a file another server has open
        # splits the annotations across two catalogs silently.
        store = tmp_path / "catalog.duckdb"
        MetadataDatabase(store_path=store).close()
        store.write_bytes(b"not a duckdb file")

        with pytest.raises(AnnotationStoreError, match="persist"):
            MetadataDatabase(store_path=store).open()
        assert store.read_bytes() == b"not a duckdb file"
        assert list(tmp_path.iterdir()) == [store]

    def test_a_transient_failure_is_retried(self, tmp_path, monkeypatch):
        # A DuckDB lock held by a server on its way down is the one open
        # failure that clears by itself, and it is indistinguishable from
        # corruption except by outliving a retry.
        monkeypatch.setattr(metadata_db, "_OPEN_RETRY_SECONDS", 0)
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)

        real, attempts = db._connect, []

        def _locked_once(target):
            attempts.append(target)
            if len(attempts) == 1:
                raise OSError("Could not set lock on file: Conflicting lock is held")
            return real(target)

        monkeypatch.setattr(db, "_connect", _locked_once)
        db.open()
        assert len(attempts) == 2
        assert db.annotations_persisted

    def test_a_session_only_server_says_so(self):
        assert MetadataDatabase().annotations_persisted is False


class TestStorePathResolution:
    """Which file a server picks, which is what keeps two servers apart."""

    @staticmethod
    def _config(**annotations):
        from biopb_tensor_server.core.config import AnnotationsConfig, ServerConfig

        return ServerConfig(annotations=AnnotationsConfig(**annotations))

    def test_the_default_is_derived_from_the_config_path(self, tmp_path):
        from biopb._locations import tensor_catalog_path
        from biopb_tensor_server.cli import _annotation_store_path

        config = tmp_path / "biopb.json"
        assert _annotation_store_path(self._config(), config) == tensor_catalog_path(
            config
        )

    def test_two_configs_get_two_files(self, tmp_path):
        from biopb._locations import tensor_catalog_path

        assert tensor_catalog_path(tmp_path / "a.json") != tensor_catalog_path(
            tmp_path / "b.json"
        )

    def test_an_explicit_path_wins(self, tmp_path):
        from biopb_tensor_server.cli import _annotation_store_path

        chosen = tmp_path / "somewhere.duckdb"
        assert (
            _annotation_store_path(self._config(store_path=str(chosen)), None) == chosen
        )

    def test_persist_off_stays_in_memory(self, tmp_path):
        from biopb_tensor_server.cli import _annotation_store_path

        assert (
            _annotation_store_path(
                self._config(persist=False, store_path=str(tmp_path / "x.duckdb")),
                tmp_path / "biopb.json",
            )
            is None
        )

    def test_no_config_file_means_no_derived_name(self):
        from biopb_tensor_server.cli import _annotation_store_path

        assert _annotation_store_path(self._config(), None) is None


class TestOrphanClock:
    """`last_seen_at`: what advances it, and what a prune makes of it."""

    @staticmethod
    def _age(db, array_id, *, days):
        """Backdate a tensor's sighting, standing in for elapsed time."""
        db._get_connection().execute(
            "UPDATE rois SET last_seen_at = ?, created_at = ? WHERE array_id = ?",
            [
                datetime.now() - timedelta(days=days),
                datetime.now() - timedelta(days=days),
                array_id,
            ],
        )

    def test_a_sighting_reaches_annotations_nobody_touched(self):
        # The reason the sweep exists: a write only refreshes the tensor being
        # drawn on, so without it an untouched annotation looks unseen however
        # often its source is rescanned.
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "file:///data/a.zarr")
        db.put_rois(ARRAY_ID, [_annotation()])
        self._age(db, ARRAY_ID, days=40)

        assert db.mark_sources_seen() == 1
        (seen,) = db._get_cursor().execute("SELECT last_seen_at FROM rois").fetchone()
        assert datetime.now() - seen < timedelta(seconds=30)

    def test_a_source_the_catalog_cannot_answer_for_is_left_alone(self):
        # Absence is not deletion -- an unmounted drive must register as no
        # news, not as a sighting that failed to happen.
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "file:///data/a.zarr")
        db.put_rois(ARRAY_ID, [_annotation()])
        db.put_rois("zarr_gone/Image:0", [_annotation()])
        self._age(db, "zarr_gone/Image:0", days=40)

        assert db.mark_sources_seen() == 1
        (seen,) = (
            db._get_cursor()
            .execute(
                "SELECT last_seen_at FROM rois WHERE array_id = ?",
                ["zarr_gone/Image:0"],
            )
            .fetchone()
        )
        assert datetime.now() - seen > timedelta(days=39)

    def test_the_sweep_backfills_a_url_the_write_could_not_resolve(self):
        # Written before discovery caught up: the row is unreportable until
        # something fills the URL in, and until now only another write could.
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation()])
        assert db._get_cursor().execute("SELECT source_url FROM rois").fetchone() == (
            None,
        )

        _register_source(db, "zarr_a1b2c3", "file:///data/a.zarr")
        db.mark_sources_seen()
        assert db._get_cursor().execute("SELECT source_url FROM rois").fetchone() == (
            "file:///data/a.zarr",
        )

    def test_the_sweep_refreshes_a_renamed_url(self):
        # Same rule as the write path, in the statement that runs for the whole
        # catalog: the label tracks what the catalog calls the source now.
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "file:///data/a.zarr")
        db.put_rois(ARRAY_ID, [_annotation()])
        db._get_connection().execute(
            "UPDATE sources SET source_url = ? WHERE source_id = ?",
            ["lab/a.zarr", "zarr_a1b2c3"],
        )

        db.mark_sources_seen()

        assert db._get_cursor().execute("SELECT source_url FROM rois").fetchone() == (
            "lab/a.zarr",
        )

    def test_a_vanished_source_is_reported_under_its_last_name(self):
        """The point of refreshing: freeze at the LAST sighting, not the first.

        A source renamed and then unmounted used to be reported under the name it
        had when the first annotation was drawn -- a value the catalog had
        abandoned long before the source went away, shown at the one moment the
        column is load-bearing.
        """
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "/data/exp.zarr")
        db.put_rois(ARRAY_ID, [_annotation()])
        db._get_connection().execute(
            "UPDATE sources SET source_url = ? WHERE source_id = ?",
            ["lab/exp.zarr", "zarr_a1b2c3"],
        )
        db.mark_sources_seen()

        # The source goes away; absence writes nothing, so the label stands.
        db._get_connection().execute("DELETE FROM sources")
        self._age(db, ARRAY_ID, days=40)
        db.mark_sources_seen()

        report = db.unseen_rois(datetime.now() - timedelta(days=30))
        assert [r.source_url for r in report] == ["lab/exp.zarr"]

    def test_a_present_source_with_no_url_still_counts_as_seen(self):
        # in_catalog and source_url are separate answers: presence is what
        # turns a fresh row's clock on, not whether a URL came with it.
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", None)
        db.put_rois(ARRAY_ID, [_annotation()])
        (seen,) = db._get_cursor().execute("SELECT last_seen_at FROM rois").fetchone()
        assert seen is not None

    def test_unseen_rois_groups_per_tensor_and_names_the_image(self):
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "file:///data/a.zarr")
        db.put_rois(ARRAY_ID, [_annotation(), _annotation()])
        db.put_rois("zarr_gone/Image:0", [_annotation()])
        self._age(db, "zarr_gone/Image:0", days=40)

        db.mark_sources_seen()
        report = db.unseen_rois(datetime.now() - timedelta(days=30))
        assert [(r.array_id, r.count, r.source_url) for r in report] == [
            ("zarr_gone/Image:0", 1, None)
        ]

    def test_a_row_never_observed_is_aged_from_its_creation(self):
        # Otherwise the strongest orphan -- a source that has never once
        # appeared -- is the one row a prune can never reach.
        db = MetadataDatabase()
        db.put_rois(ARRAY_ID, [_annotation()])
        db._get_connection().execute(
            "UPDATE rois SET created_at = ?", [datetime.now() - timedelta(days=40)]
        )
        assert db._get_cursor().execute("SELECT last_seen_at FROM rois").fetchone() == (
            None,
        )

        report = db.unseen_rois(datetime.now() - timedelta(days=30))
        assert [r.count for r in report] == [1]

    def test_prune_removes_exactly_what_the_report_named(self):
        db = MetadataDatabase()
        _register_source(db, "zarr_a1b2c3", "file:///data/a.zarr")
        db.put_rois(ARRAY_ID, [_annotation(label="keep")])
        db.put_rois("zarr_gone/Image:0", [_annotation(label="drop")])
        self._age(db, "zarr_gone/Image:0", days=40)
        db.mark_sources_seen()

        cutoff = datetime.now() - timedelta(days=30)
        named = sum(r.count for r in db.unseen_rois(cutoff))
        assert db.prune_unseen(cutoff) == named == 1
        assert db._get_cursor().execute("SELECT label FROM rois").fetchall() == [
            ("keep",)
        ]

    def test_pruning_survives_a_restart_with_the_catalog(self, tmp_path):
        # The clock is only worth anything now that the rows outlive the process.
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.put_rois(ARRAY_ID, [_annotation()])
        self._age(db, ARRAY_ID, days=40)
        db.close()

        db = MetadataDatabase(store_path=store)
        assert (
            sum(r.count for r in db.unseen_rois(datetime.now() - timedelta(days=30)))
            == 1
        )
        assert db.prune_unseen(datetime.now() - timedelta(days=30)) == 1


class TestPruneCli:
    """`prune-annotations`: the escape hatch, and the fact that it needs the server down."""

    @staticmethod
    def _config(tmp_path, store, **annotations):
        path = tmp_path / "biopb.json"
        path.write_text(
            json.dumps(
                {
                    "sources": [],
                    "annotations": {"store_path": str(store), **annotations},
                }
            )
        )
        return path

    @staticmethod
    def _seed(store, *, days_old):
        db = MetadataDatabase(store_path=store)
        db.put_rois(ARRAY_ID, [_annotation(label="stale")])
        db._get_connection().execute(
            "UPDATE rois SET last_seen_at = ?, created_at = ?, source_url = ?",
            [
                datetime.now() - timedelta(days=days_old),
                datetime.now() - timedelta(days=days_old),
                "file:///data/plate3.zarr",
            ],
        )
        db.close()

    def _run(self, *args):
        from biopb_tensor_server.cli import app
        from typer.testing import CliRunner

        return CliRunner().invoke(app, ["prune-annotations", *args])

    def test_it_reports_without_deleting(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        self._seed(store, days_old=90)
        config = self._config(tmp_path, store)

        result = self._run(str(config), "--days", "30")
        assert result.exit_code == 0
        assert "plate3.zarr" in result.output
        assert "would be deleted" in result.output
        assert MetadataDatabase(store_path=store).list_rois(ARRAY_ID)[0]

    def test_apply_deletes(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        self._seed(store, days_old=90)
        config = self._config(tmp_path, store)

        result = self._run(str(config), "--days", "30", "--apply")
        assert result.exit_code == 0
        assert "Deleted 1" in result.output
        assert MetadataDatabase(store_path=store).list_rois(ARRAY_ID) == ([], False)

    def test_a_fresh_annotation_is_left_alone(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        self._seed(store, days_old=2)
        config = self._config(tmp_path, store)

        result = self._run(str(config), "--days", "30", "--apply")
        assert "Nothing unseen" in result.output
        assert MetadataDatabase(store_path=store).list_rois(ARRAY_ID)[0]

    def test_it_takes_the_threshold_from_the_config(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        self._seed(store, days_old=90)
        config = self._config(tmp_path, store, prune_unseen_days=30)

        assert "would be deleted" in self._run(str(config)).output

    def test_no_threshold_anywhere_is_refused(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        self._seed(store, days_old=90)
        config = self._config(tmp_path, store)

        result = self._run(str(config))
        assert result.exit_code == 2
        assert "No age threshold" in result.output

    def test_a_session_only_config_has_nothing_to_prune(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        self._seed(store, days_old=90)
        config = self._config(tmp_path, store, persist=False)

        result = self._run(str(config), "--days", "30")
        assert result.exit_code == 0
        assert "nothing on disk" in result.output

    def test_a_catalog_held_open_says_to_stop_the_server(self, tmp_path):
        # The whole answer to "can I run this while the server is up": no.
        # DuckDB's lock is exclusive for readers too.
        store = tmp_path / "catalog.duckdb"
        self._seed(store, days_old=90)
        config = self._config(tmp_path, store)

        holder = MetadataDatabase(store_path=store)
        holder.open()
        try:
            with mock.patch.object(
                metadata_db.MetadataDatabase,
                "_connect",
                side_effect=OSError("Conflicting lock is held in ... (PID 1)"),
            ):
                result = self._run(str(config), "--days", "30", "--apply")
        finally:
            holder.close()

        assert result.exit_code == 1
        # Rich hard-wraps to the terminal width, so match on flattened text.
        assert "needs the server stopped" in " ".join(result.output.split())
        # And nothing was deleted.
        assert MetadataDatabase(store_path=store).list_rois(ARRAY_ID)[0]


class TestSchemaVersioning:
    """The catalog file outlives the code now, so its shape has to be checked."""

    @staticmethod
    def _older_rois_table(store):
        """A `rois` table from a build before two columns existed."""
        conn = duckdb.connect(str(store), config={"enable_external_access": False})
        conn.execute(
            "CREATE TABLE rois (roi_id TEXT, array_id TEXT, source_id TEXT, "
            "set_name TEXT, label TEXT, shape_kind TEXT, "
            "plane MAP(UINTEGER, UINTEGER), bbox DOUBLE[4], geometry TEXT, "
            "rev BIGINT, created_at TIMESTAMP, updated_at TIMESTAMP, "
            "source_url TEXT, last_seen_at TIMESTAMP, "
            "PRIMARY KEY (array_id, roi_id))"
        )
        conn.close()

    def test_a_stale_rois_shape_is_refused_at_startup(self, tmp_path):
        # Without this the server starts, reports SERVING, and every annotation
        # read and write then dies on a missing column -- CREATE TABLE IF NOT
        # EXISTS against an older file is a silent no-op.
        store = tmp_path / "catalog.duckdb"
        self._older_rois_table(store)

        with pytest.raises(AnnotationStoreError, match="missing column"):
            MetadataDatabase(store_path=store).open()

    def test_a_file_from_a_newer_build_is_refused(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.put_rois(ARRAY_ID, [_annotation()])
        db._get_connection().execute(
            "INSERT OR REPLACE INTO catalog_meta VALUES ('roi_schema_version', '99')"
        )
        db.close()

        with pytest.raises(AnnotationStoreError, match="newer biopb"):
            MetadataDatabase(store_path=store).open()

    def test_a_fresh_catalog_is_stamped(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.open()
        assert db._get_cursor().execute(
            "SELECT value FROM catalog_meta WHERE key = 'roi_schema_version'"
        ).fetchone() == (str(metadata_db._ROI_SCHEMA_VERSION),)

    def test_an_unmarked_catalog_reads_as_version_one(self, tmp_path):
        # The marker shipped in the same release as v1, so a rois table without
        # one can only have been written by that release.
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.put_rois(ARRAY_ID, [_annotation(label="drawn")])
        db._get_connection().execute("DELETE FROM catalog_meta")
        db.close()

        db = MetadataDatabase(store_path=store)
        assert [r.label for r in db.list_rois(ARRAY_ID)[0]] == ["drawn"]

    def test_a_missing_migration_is_refused_rather_than_skipped(
        self, tmp_path, monkeypatch
    ):
        store = tmp_path / "catalog.duckdb"
        MetadataDatabase(store_path=store).open()
        # This build now believes in a v2 it has no way to reach.
        monkeypatch.setattr(metadata_db, "_ROI_SCHEMA_VERSION", 2)

        with pytest.raises(AnnotationStoreError, match="No migration"):
            MetadataDatabase(store_path=store).open()

    def test_a_migration_runs_and_the_stamp_advances(self, tmp_path, monkeypatch):
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.put_rois(ARRAY_ID, [_annotation(label="kept")])
        db.close()

        ran = []

        def _v1_to_v2(conn):
            ran.append(True)
            conn.execute("ALTER TABLE rois ADD COLUMN note TEXT")

        monkeypatch.setattr(metadata_db, "_ROI_SCHEMA_VERSION", 2)
        monkeypatch.setattr(metadata_db, "_ROI_MIGRATIONS", {1: _v1_to_v2})

        db = MetadataDatabase(store_path=store)
        db.open()
        assert ran
        assert db._get_cursor().execute(
            "SELECT value FROM catalog_meta WHERE key = 'roi_schema_version'"
        ).fetchone() == ("2",)
        assert [r.label for r in db.list_rois(ARRAY_ID)[0]] == ["kept"]

    def test_a_stale_sources_shape_is_rebuilt_not_refused(self, tmp_path):
        # `sources` is scan output, so it is exempt from versioning entirely:
        # dropping it makes a change to its columns free.
        store = tmp_path / "catalog.duckdb"
        conn = duckdb.connect(str(store), config={"enable_external_access": False})
        conn.execute("CREATE TABLE sources (source_id TEXT PRIMARY KEY)")
        conn.execute(metadata_db._ROIS_DDL)
        conn.close()

        db = MetadataDatabase(store_path=store)
        db.open()
        columns = {
            r[0] for r in db._get_cursor().execute("DESCRIBE sources").fetchall()
        }
        assert {"data_resident", "tensors", "source_url"} <= columns

    def test_the_expected_columns_come_from_the_ddl_itself(self, tmp_path):
        # The check must not rest on a hand-written column list, because the
        # edit that forgets to bump the version is the same edit that would
        # forget to update the list.
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.open()
        actual = {r[0] for r in db._get_cursor().execute("DESCRIBE rois").fetchall()}
        assert "drawn_against_version" in actual
        assert set(MetadataDatabase._ROI_CLIENT_COLUMNS) <= actual


class TestStorePathIsNotCwdRelative:
    """A relative store_path anchors on the config file, never on the cwd."""

    @staticmethod
    def _config(**annotations):
        from biopb_tensor_server.core.config import AnnotationsConfig, ServerConfig

        return ServerConfig(annotations=AnnotationsConfig(**annotations))

    def test_a_relative_path_resolves_against_the_config(self, tmp_path):
        from biopb_tensor_server.cli import _annotation_store_path

        config = tmp_path / "deploy" / "biopb.json"
        resolved = _annotation_store_path(
            self._config(store_path="rois.duckdb"), config
        )
        assert resolved == tmp_path / "deploy" / "rois.duckdb"

    def test_the_cwd_does_not_change_the_answer(self, tmp_path, monkeypatch):
        # The whole point: a server is started by the control plane, by systemd,
        # or by hand from wherever the user was standing.
        from biopb_tensor_server.cli import _annotation_store_path

        config = tmp_path / "biopb.json"
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        first = _annotation_store_path(self._config(store_path="a.duckdb"), config)
        monkeypatch.chdir(elsewhere)
        assert (
            _annotation_store_path(self._config(store_path="a.duckdb"), config) == first
        )

    def test_an_absolute_path_is_left_alone(self, tmp_path):
        from biopb_tensor_server.cli import _annotation_store_path

        chosen = tmp_path / "somewhere" / "x.duckdb"
        assert (
            _annotation_store_path(
                self._config(store_path=str(chosen)), tmp_path / "c.json"
            )
            == chosen
        )

    def test_a_relative_path_with_no_config_is_refused(self):
        from biopb_tensor_server.cli import _annotation_store_path

        with pytest.raises(AnnotationStoreError, match="relative"):
            _annotation_store_path(self._config(store_path="rois.duckdb"), None)


class TestDisabledAnnotationsTouchNothing:
    """`annotations.enabled = false` means the store is not opened at all."""

    @staticmethod
    def _config(**annotations):
        from biopb_tensor_server.core.config import AnnotationsConfig, ServerConfig

        return ServerConfig(annotations=AnnotationsConfig(**annotations))

    def test_no_store_path_so_no_lock(self, tmp_path):
        # DuckDB's lock is exclusive, so holding the catalog open would block
        # prune-annotations and any other reader for a feature this server is
        # not offering.
        from biopb_tensor_server.cli import _annotation_store_path

        assert (
            _annotation_store_path(
                self._config(enabled=False, store_path=str(tmp_path / "x.duckdb")),
                tmp_path / "biopb.json",
            )
            is None
        )

    def test_a_broken_store_cannot_stop_a_server_that_does_not_serve_it(self, tmp_path):
        # Fail-closed is about a promise of durability. A server told not to
        # serve annotations made no such promise.
        from biopb_tensor_server.cli import _annotation_store_path

        store = tmp_path / "catalog.duckdb"
        store.write_bytes(b"not a duckdb file")
        config = self._config(enabled=False, store_path=str(store))

        path = _annotation_store_path(config, tmp_path / "biopb.json")
        assert path is None
        MetadataDatabase(store_path=path, annotations_enabled=False).open()

    def test_the_sql_surface_drops_rois_too(self):
        # Returning empty rows would be the wrong answer: the table is
        # unserved, not unpopulated, and a result set cannot say which.
        db = MetadataDatabase(annotations_enabled=False)
        # Only `rois` goes: `decode_rates` is cache measurement, not annotation
        # data, and a deployment that turned the annotation actions off did not
        # ask to stop measuring its own cache.
        assert "rois" not in db.allowed_tables
        assert {"sources", "decode_rates"} <= db.allowed_tables
        with pytest.raises(ValueError, match="disallowed table: rois"):
            db._validate_query("SELECT * FROM rois")
        db._validate_query("SELECT * FROM sources")

    def test_rois_stays_queryable_when_enabled(self):
        db = MetadataDatabase()
        db._validate_query("SELECT count(*) FROM rois")
