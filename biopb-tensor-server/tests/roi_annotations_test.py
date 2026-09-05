"""ROI annotations: the DuckDB store, the Flight actions, and the sidecar routes.

Design: ``docs/roi-annotations.md``. The store-level cases pin the decisions that
are easy to regress -- geometry acceptance, level-0 bbox derivation, plane pinning,
per-ROI rev/conflict, the cap -- while one gRPC round-trip and one FastAPI pass
exercise the wire in each direction.
"""

import threading
import time

import pyarrow.flight as flight
import pytest
from biopb.image import ROI, Ellipse, Mask, Point, Polygon, Polyline, Rectangle
from biopb.image.annotation_pb2 import RoiAnnotation
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.core.metadata_db import MetadataDatabase

ARRAY_ID = "zarr_a1b2c3/Image:0"


def _polygon(*pts):
    return ROI(polygon=Polygon(points=[Point(x=x, y=y) for x, y in pts]))


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
        ann = _annotation(label="nucleus", set_name="nuclei", plane={"z": 12, "t": 0})
        db.put_rois(ARRAY_ID, [ann])

        rois, truncated = db.list_rois(ARRAY_ID)
        assert not truncated
        assert len(rois) == 1
        got = rois[0]
        assert got.label == "nucleus"
        assert dict(got.plane) == {"z": 12, "t": 0}
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
        assert (
            db._get_cursor().execute("SELECT bbox FROM rois").fetchone()[0] == expected
        )

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

    def test_sql_surface_stays_read_only(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="forbidden keyword"):
            db.handle_query("DELETE FROM rois")

    def test_source_url_is_captured_for_a_registered_source(self, tmp_path):
        """The orphan-report anchor: array_id is a hash and cannot be inverted."""
        db = MetadataDatabase()
        db._get_connection().execute(
            "INSERT INTO sources (source_id, source_url, source_type, tensors) "
            "VALUES (?, ?, ?, ?)",
            ["zarr_a1b2c3", "/data/exp.zarr", "zarr", []],
        )
        db.put_rois(ARRAY_ID, [_annotation()])
        row = (
            db._get_cursor()
            .execute("SELECT source_url, last_seen_at IS NOT NULL FROM rois")
            .fetchone()
        )
        assert row == ("/data/exp.zarr", True)

    def test_an_unknown_source_still_accepts_annotations(self):
        """Progressive discovery: catalog absence proves nothing about the image."""
        db = MetadataDatabase()
        stored, _ = db.put_rois(ARRAY_ID, [_annotation()])
        assert stored
        assert (
            db._get_cursor().execute("SELECT source_url FROM rois").fetchone()[0]
            is None
        )


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
                    "plane": {"z": "12"},
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
        assert got["rois"][0]["plane"] == {"z": "12"}
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

    def test_cross_origin_write_is_refused(self, client_and_app):
        client, _db = client_and_app
        resp = client.post(
            f"/api/rois/{ARRAY_ID}",
            json={"rois": []},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert resp.status_code == 403
