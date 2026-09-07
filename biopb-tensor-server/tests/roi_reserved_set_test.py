"""The reserved set namespace: server-owned annotation layers are read-only.

Design: ``docs/roi-annotations.md`` and biopb/biopb#951. A reserved set is filled
from the source file and replaced wholesale when that file changes, so a client
write landing in one would be discarded at the next re-import rather than stored.
The store refuses it instead of trusting clients to honour the naming rule.

The importer does not exist yet; these plant reserved rows with direct SQL, which
is also the only way to get one past the guard being tested.
"""

from datetime import datetime, timedelta

import pytest
from biopb.image import ROI, Point, Polygon
from biopb.image.annotation_pb2 import RoiAnnotation
from biopb_tensor_server.core.metadata_db import (
    RESERVED_SET_PREFIX,
    MetadataDatabase,
    is_reserved_set,
)
from google.protobuf import json_format

ARRAY_ID = "zarr_a1b2c3/Image:0"
RESERVED = f"{RESERVED_SET_PREFIX}ome"


def _polygon(*pts):
    return ROI(polygon=Polygon(points=[Point(x=x, y=y) for x, y in pts]))


def _annotation(**kwargs):
    kwargs.setdefault("roi", _polygon((1, 2), (10, 2), (5, 9)))
    return RoiAnnotation(**kwargs)


def _plant(db, roi_id, *, set_name=RESERVED, label="from-file", when=None):
    """Insert a row as the importer eventually will -- past the public guard."""
    now = when or datetime.now()
    geometry = json_format.MessageToJson(
        _polygon((0, 0), (4, 0), (4, 4)), indent=0
    ).replace("\n", "")
    db._get_connection().execute(
        "INSERT INTO rois (roi_id, array_id, source_id, set_name, label, "
        "shape_kind, geometry, rev, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 'polygon', ?, 1, ?, ?)",
        [roi_id, ARRAY_ID, ARRAY_ID.split("/")[0], set_name, label, geometry, now, now],
    )


def _sets(db):
    return dict(
        db._get_connection()
        .execute("SELECT roi_id, set_name FROM rois WHERE array_id = ?", [ARRAY_ID])
        .fetchall()
    )


class TestTheNamespaceRule:
    @pytest.mark.parametrize("name", ["@", "@ome", "@ome:0", "@anything"])
    def test_a_leading_marker_reserves(self, name):
        assert is_reserved_set(name)

    @pytest.mark.parametrize("name", ["", "default", "ome", "my@set", "nuclei"])
    def test_everything_else_is_the_client_s(self, name):
        assert not is_reserved_set(name)


class TestWritingIntoAReservedSet:
    @pytest.mark.parametrize("name", ["@", "@ome", "@imagej"])
    def test_is_refused(self, name):
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="reserved"):
            db.put_rois(ARRAY_ID, [_annotation(set_name=name)])

    def test_a_set_merely_containing_the_marker_is_fine(self):
        db = MetadataDatabase()
        (stored,), _ = db.put_rois(ARRAY_ID, [_annotation(set_name="my@set")])
        assert stored.set_name == "my@set"

    def test_nothing_is_written(self):
        db = MetadataDatabase()
        with pytest.raises(ValueError):
            db.put_rois(ARRAY_ID, [_annotation(set_name=RESERVED)])
        assert _sets(db) == {}


class TestReusingAnImportedId:
    """The silent-move path: set_name is a client column, so an UPDATE moves it."""

    def test_is_refused(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")

        with pytest.raises(ValueError, match="ROI:3"):
            db.put_rois(ARRAY_ID, [_annotation(roi_id="ROI:3", set_name="mine")])

    def test_the_imported_row_does_not_move(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")

        with pytest.raises(ValueError):
            db.put_rois(ARRAY_ID, [_annotation(roi_id="ROI:3", set_name="mine")])

        assert _sets(db) == {"ROI:3": RESERVED}
        (row,), _ = db.list_rois(ARRAY_ID, set_name=RESERVED)
        assert row.label == "from-file"

    def test_it_takes_the_whole_batch_with_it(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")

        with pytest.raises(ValueError):
            db.put_rois(
                ARRAY_ID,
                [
                    _annotation(roi_id="mine-1", set_name="mine"),
                    _annotation(roi_id="ROI:3", set_name="mine"),
                ],
            )

        assert _sets(db) == {"ROI:3": RESERVED}

    def test_a_fresh_id_is_how_a_clone_works(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")

        (stored,), conflicts = db.put_rois(
            ARRAY_ID, [_annotation(roi_id="clone-1", set_name="mine")]
        )

        assert not conflicts
        assert stored.set_name == "mine"
        assert _sets(db) == {"ROI:3": RESERVED, "clone-1": "mine"}


class TestDeletingAReservedSet:
    def test_naming_the_set_is_refused(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")

        with pytest.raises(ValueError, match="reserved"):
            db.delete_rois(ARRAY_ID, set_name=RESERVED)
        assert _sets(db) == {"ROI:3": RESERVED}

    def test_an_id_inside_it_is_refused_without_naming_it(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")

        with pytest.raises(ValueError, match="ROI:3"):
            db.delete_rois(ARRAY_ID, ["ROI:3"])
        assert _sets(db) == {"ROI:3": RESERVED}

    def test_one_reserved_id_refuses_the_batch(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")
        db.put_rois(ARRAY_ID, [_annotation(roi_id="mine-1", set_name="mine")])

        with pytest.raises(ValueError, match="ROI:3"):
            db.delete_rois(ARRAY_ID, ["mine-1", "ROI:3"])
        assert _sets(db) == {"ROI:3": RESERVED, "mine-1": "mine"}

    def test_ordinary_ids_still_delete(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")
        db.put_rois(ARRAY_ID, [_annotation(roi_id="mine-1", set_name="mine")])

        assert db.delete_rois(ARRAY_ID, ["mine-1"]) == ["mine-1"]
        assert _sets(db) == {"ROI:3": RESERVED}


class TestClearingATensor:
    """An unqualified delete is scoped, not refused: nothing addressed the set."""

    def test_leaves_the_imported_set_and_takes_the_rest(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")
        db.put_rois(
            ARRAY_ID,
            [
                _annotation(roi_id="mine-1", set_name="mine"),
                _annotation(roi_id="mine-2"),
            ],
        )

        assert sorted(db.delete_rois(ARRAY_ID)) == ["mine-1", "mine-2"]
        assert _sets(db) == {"ROI:3": RESERVED}

    def test_a_tensor_holding_only_an_import_loses_nothing(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3")

        assert db.delete_rois(ARRAY_ID) == []
        assert _sets(db) == {"ROI:3": RESERVED}


class TestTheOrphanClock:
    """A reserved set is a cache, so `unseen_rois` / `prune_unseen` skip it.

    The clock exists to give hand-drawn work a grace period before anything
    deletes it. `prune_unseen` also deletes with raw SQL rather than through
    `delete_rois`, so without this it would remove rows the API refuses to.
    """

    STALE = datetime.now() - timedelta(days=90)
    CUTOFF = datetime.now() - timedelta(days=30)

    def test_an_imported_set_is_not_reported_as_unseen(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3", when=self.STALE)

        assert db.unseen_rois(self.CUTOFF) == []

    def test_an_imported_set_survives_the_prune(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3", when=self.STALE)

        assert db.prune_unseen(self.CUTOFF) == 0
        assert _sets(db) == {"ROI:3": RESERVED}

    def test_hand_drawn_rows_are_still_reported_and_pruned(self):
        db = MetadataDatabase()
        _plant(db, "mine-1", set_name="mine", when=self.STALE)

        (group,) = db.unseen_rois(self.CUTOFF)
        assert group.count == 1
        assert group.array_id == ARRAY_ID
        assert db.prune_unseen(self.CUTOFF) == 1
        assert _sets(db) == {}

    def test_an_imported_row_does_not_inflate_a_real_group(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3", when=self.STALE)
        _plant(db, "mine-1", set_name="mine", when=self.STALE)

        (group,) = db.unseen_rois(self.CUTOFF)
        assert group.count == 1  # the hand-drawn one, not both
        assert db.prune_unseen(self.CUTOFF) == 1
        assert _sets(db) == {"ROI:3": RESERVED}

    def test_the_report_and_the_delete_agree(self):
        db = MetadataDatabase()
        _plant(db, "ROI:3", when=self.STALE)
        _plant(db, "mine-1", set_name="mine", when=self.STALE)
        _plant(db, "mine-2", set_name="mine", when=self.STALE)

        promised = sum(g.count for g in db.unseen_rois(self.CUTOFF))
        assert db.prune_unseen(self.CUTOFF) == promised
