"""The ROI annotation store: schema, migrations, validation and the methods.

User-drawn annotations live in the ``rois`` table of the same DuckDB file as the
source catalog, so the methods here are a mixin of :class:`MetadataDatabase`
rather than a store of their own: they share its connection and its write lock,
and read the catalog's ``sources`` view only to stamp a sighting.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import duckdb
from biopb.image.annotation_pb2 import RoiAnnotation, RoiConflict
from biopb.image.roi_pb2 import ROI
from google.protobuf import json_format

from biopb_tensor_server.core.errors import AnnotationStoreError
from biopb_tensor_server.core.labels import last_named_segment
from biopb_tensor_server.core.registration import MAX_CAPTION_LEN, normalize_caption

logger = logging.getLogger(__name__)

# Shape of the `rois` table. Bump on any change to it and add the matching entry
# to _ROI_MIGRATIONS, whose keys are the version each step upgrades FROM.
#
# `sources` is deliberately exempt: it is scan output, so it is dropped and
# recreated on every open and its columns are always this build's. Only the
# annotations are old enough to need carrying forward.
_ROI_SCHEMA_VERSION = 2


def _mark_label_segment(array_id: str) -> str:
    """*array_id* with its label segment marked: ``labels`` -> ``@labels``.

    The v1 parse, run once over stored ids: the **last** ``labels`` segment of
    the within-source field that has a name after it, which is the one
    ``split_label_field`` used to find. The source half is never touched -- a
    source_id has no ``/`` -- and an id naming no set comes back unchanged.
    """
    head, slash, field = array_id.partition("/")
    if not slash:
        return array_id
    parts = field.split("/")
    i = last_named_segment(parts, "labels")
    if i is None:
        return array_id
    parts[i] = "@labels"
    return head + "/" + "/".join(parts)


def _migrate_rois_v1_to_v2(conn: duckdb.DuckDBPyConnection) -> None:
    """Carry annotations onto the marked label segment.

    A set's wire id became ``<image>/@labels/<name>``, so an annotation filed
    against the old form would otherwise anchor on a tensor id that is never
    minted again -- the silent orphaning ``_require_bare_array_id`` exists to
    prevent, and the reason this table has a ladder rather than being dropped
    like ``decode_rates``.

    In Python rather than SQL because the segment to mark is the one the v1
    parser picked; a ``replace()`` would also hit a *set* named ``labels`` or
    an image field containing one.
    """
    rows = conn.execute(
        "SELECT DISTINCT array_id FROM rois WHERE array_id LIKE '%labels/%'"
    ).fetchall()
    moved = 0
    for (array_id,) in rows:
        marked = _mark_label_segment(array_id)
        if marked == array_id:
            continue
        conn.execute(
            "UPDATE rois SET array_id = ? WHERE array_id = ?", [marked, array_id]
        )
        moved += 1
    if moved:
        logger.info("Moved annotations of %d label set(s) onto '@labels'", moved)


_ROI_MIGRATIONS: Dict[int, Callable[[duckdb.DuckDBPyConnection], None]] = {
    1: _migrate_rois_v1_to_v2,
}

# The `rois` DDL, module level because the schema self-check builds a throwaway
# copy from it (_expected_roi_columns) rather than keeping a second hand-written
# column list that the edit forgetting to bump the version would also forget.
_ROIS_DDL = """
CREATE TABLE IF NOT EXISTS rois (
    -- Unique WITHIN a tensor, not globally: a client may name its
    -- own ids, and two tensors independently choosing "roi-1" is
    -- ordinary, not a conflict. The key must be composite for that
    -- to be safe -- with roi_id alone as the PK, an INSERT OR
    -- REPLACE for one tensor silently overwrote another tensor's
    -- row, because the create-or-update lookup is scoped by
    -- array_id while the key was not.
    roi_id TEXT NOT NULL,
    -- The tensor, in its UNVERSIONED array_id form. Annotations must
    -- outlive an in-place edit of the image, so the sidecar's
    -- `source@token/field` version token never reaches this column.
    array_id TEXT NOT NULL,
    -- array_id split on the first '/': joins, authorization, and the
    -- catalog-presence check last_seen_at is built on.
    source_id TEXT NOT NULL,
    -- What the catalog called this source when it was last SEEN, which
    -- for a row a report can reach is when it was last present. NOT a
    -- liveness probe -- there is no existence oracle spanning file /
    -- proxy / cloud / upload sources -- but array_id is a SHA-256 and
    -- cannot be inverted, so without this an orphan report can only
    -- say "annotations for zarr_a3f2b1c4", which no one can act on.
    -- Also what a re-attach-after-move prompt would key on: a move
    -- changes the path, hence source_id AND array_id, so the name is
    -- what survives to match on. A label, never an identifier --
    -- nothing joins or filters on it.
    source_url TEXT,
    -- Grouping key: the "layer" ("nuclei", "hand-drawn").
    set_name TEXT NOT NULL DEFAULT 'default',
    label TEXT,
    -- point|rectangle|ellipse|polygon, denormalized from the geometry
    -- for filtering. mask/mesh are rejected on write.
    shape_kind TEXT NOT NULL,
    -- Sparse plane pin, WIRE AXIS INDEX -> index on that axis. A
    -- dimension ABSENT from the map applies at every index of it, so
    -- one ROI can follow a z-stack without being duplicated per
    -- plane. Keyed by position, not label: a label is neither
    -- guaranteed present nor unique, so it cannot address every axis
    -- (see annotation.proto). Reading this column against a tensor's
    -- dim_labels is what turns an index back into a name.
    plane MAP(UINTEGER, UINTEGER),
    -- [x0, y0, x1, y1] in level-0 pixels, derived server-side. Unused
    -- by the viewer read path (which fetches a tensor's whole set);
    -- it is what makes the SQL surface useful.
    bbox DOUBLE[4],
    -- biopb.image.ROI as canonical proto3 JSON *text*, not a blob:
    -- the sidecar hands it to the SPA verbatim with no
    -- decode/re-encode, and the row stays legible to the SQL surface.
    geometry TEXT NOT NULL,
    -- Opaque client JSON (colour, score, author).
    props_json TEXT,
    -- content_version at write time -> CONTENT staleness ("the image
    -- changed since this was drawn"), as distinct from the tensor
    -- going away entirely.
    drawn_against_version TEXT,
    rev BIGINT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    -- Last time the source was observed in the catalog -- stamped
    -- by a write that could resolve it, and by the prune sweep (which
    -- additionally gates on a COMPLETE catalog, since only a
    -- conclusion about ABSENCE needs completeness; presence is
    -- presence). Absence is never itself evidence of deletion
    -- (progressive discovery, unmounted drives, a proxy upstream that
    -- is down), so orphan age is measured from here rather than
    -- asserted. NULL means never observed.
    last_seen_at TIMESTAMP,
    PRIMARY KEY (array_id, roi_id)
)
"""


# ---------------------------------------------------------------------------
# ROI annotation helpers
# ---------------------------------------------------------------------------

# Only the 2-D vector arms of biopb.image.ROI are stored. `mask` carries a
# BinData bitmap -- one ROI can be hundreds of KB, so a few thousand of them
# stop being "annotation scale" in the one dimension the row cap is trying to
# bound -- and `mesh` is 3-D, where plane pinning has no meaning. Both belong to
# instance segmentation, which is a label tensor, not this table. The proto
# keeps every arm, so accepting them later is additive.
# A client may name its own roi_id, and it becomes half the primary key. Bound
# it so a pathological key cannot be planted in the catalog.
_MAX_ROI_ID_LEN = 128

# The HTTP sidecar publishes a content-versioned array_id (`source@token[/field]`,
# biopb/biopb#780). Per the identity policy in descriptor.proto that form exists
# ONLY above the Flight wire -- "no adapter, chunk_id, catalog row or descriptor
# ever carries it" -- but nothing used to enforce it, and the read routes only
# get it right as a side effect of resolving a descriptor
# (`_tensor_desc_by_array_id` strips and validates in one step). A store keyed by
# array_id never resolves anything, so a missed strip was silent: the annotation
# filed itself under a phantom tensor whose id is never minted again once content
# changes, with a source_id matching no catalog row.
_VERSION_SEP = "@"


def _require_bare_array_id(array_id: str) -> None:
    """Refuse a content-versioned array_id at the layer that stores one.

    Loud beats silent: a versioned id is not a different tensor, it is a caller
    that forgot to strip, and every such bug so far has been invisible until a
    user's annotations went missing. Only the source half is checked -- a '@' in
    a field name is legal payload, and a source_id can never contain one.
    """
    if _VERSION_SEP in array_id.partition("/")[0]:
        raise ValueError(
            f"array_id {array_id!r} carries a content-version token; annotations "
            f"anchor on the unversioned id (strip it before this layer -- the "
            f"versioned form exists only above the Flight wire)."
        )


_ACCEPTED_SHAPES: Set[str] = {
    "point",
    "rectangle",
    "ellipse",
    "polygon",
    "polyline",
    "text",
}


# Set names in this namespace are server-owned, not client data: an importer
# fills them from the source file and replaces them WHOLESALE when that file
# changes (biopb/biopb#951). A client edit landing there is not merely filed in
# the wrong layer -- it is destroyed at the next re-import, with nothing to show
# for it. So the store refuses the write rather than trusting every client to
# honour a naming convention, the same posture _require_bare_array_id takes.
#
# A prefix, not a fixed name, so a second importer (ImageJ overlays, a GeoJSON
# sidecar) becomes read-only without a client release. Punctuation because the
# namespace has to be one no existing catalog can already be using: a user set
# called "ome" is plausible where "@ome" is not.
RESERVED_SET_PREFIX = "@"


def is_reserved_set(set_name: str) -> bool:
    """Whether ``set_name`` is server-owned, and so read-only to clients."""
    return set_name.startswith(RESERVED_SET_PREFIX)


# The orphan predicate, bound to `(before, RESERVED_SET_PREFIX)`. One constant
# rather than the same SQL written twice, because unseen_rois is the dry run for
# prune_unseen and a difference between them would show a person one set of rows
# and delete another.
#
# Reserved sets are outside the orphan machinery entirely. The clock exists to
# give hand-drawn work a long grace period before anything deletes it; an
# imported set is a cache of the source file, so there is nothing to protect and
# nothing is reclaimed that a re-import would not rebuild. Leaving them in would
# also let this path delete rows that put_rois and delete_rois refuse to touch,
# which is not an invariant (biopb/biopb#951).
_UNSEEN_PREDICATE = (
    "COALESCE(last_seen_at, created_at) < ? AND NOT starts_with(set_name, ?)"
)


@dataclass(frozen=True)
class UnseenRois:
    """One tensor's annotations whose source has gone unobserved.

    What an orphan report or a ``--dry-run`` prints. ``source_url`` is None for
    rows written before their source ever reached the catalog.
    """

    source_id: str
    source_url: Optional[str]
    array_id: str
    count: int
    last_seen_at: Optional[datetime]


@dataclass(frozen=True)
class _PreparedRoi:
    """A validated annotation, normalized into the column values it will occupy.

    Attribute names match the ``rois`` column names, which is what lets
    ``column_values`` build a statement's parameters from a column list instead
    of a hand-maintained positional tuple.
    """

    roi_id: str
    set_name: str
    label: str
    shape_kind: str
    plane: Dict[int, int]
    bbox: List[float]
    geometry: str
    props_json: Optional[str]
    drawn_against_version: Optional[str]
    rev: int
    roi: object

    def column_values(self, columns: Sequence[str]) -> List[object]:
        """Parameters for *columns*, in order."""
        return [getattr(self, name) for name in columns]

    def to_proto(
        self,
        array_id: str,
        rev: int,
        created_at: datetime,
        updated_at: datetime,
    ) -> RoiAnnotation:
        out = RoiAnnotation(
            roi_id=self.roi_id,
            array_id=array_id,
            set_name=self.set_name,
            label=self.label or "",
            roi=self.roi,
            props_json=self.props_json or "",
            rev=rev,
            created_at_unix_ms=_to_unix_ms(created_at),
            updated_at_unix_ms=_to_unix_ms(updated_at),
        )
        out.plane.update(self.plane)
        if self.drawn_against_version is not None:
            out.drawn_against_version = bytes.fromhex(self.drawn_against_version)
        return out


def _prepare_roi(
    array_id: str, roi: RoiAnnotation, *, allow_reserved: bool = False
) -> _PreparedRoi:
    """Validate one annotation and derive its stored columns.

    Raises:
        ValueError: unusable geometry, or an array_id that contradicts the
            batch's.
    """
    if roi.array_id and roi.array_id != array_id:
        raise ValueError(
            f"Annotation array_id {roi.array_id!r} does not match the request's "
            f"{array_id!r}"
        )

    # A client-supplied id becomes half of the primary key, so bound it. Ids are
    # unique per tensor, so no cross-tensor check is needed -- the composite key
    # makes two tensors reusing one id independent rows.
    roi_id = roi.roi_id.strip()
    if len(roi_id) > _MAX_ROI_ID_LEN:
        raise ValueError(
            f"roi_id is longer than {_MAX_ROI_ID_LEN} characters: {roi_id[:32]!r}..."
        )
    # The sidecar deletes by a comma-separated `?ids=` list, so an id containing
    # a comma could be created but never addressed: it would split into two ids
    # matching nothing, and the delete would report zero removals with no error.
    # Refused at the store so no transport can mint an unreachable row.
    if "," in roi_id:
        raise ValueError(f"roi_id may not contain a comma: {roi_id!r}")

    set_name = roi.set_name or "default"
    # allow_reserved is the importer's door, and only the importer's: it is not
    # reachable from put_rois, so no transport can set it.
    if is_reserved_set(set_name) and not allow_reserved:
        raise ValueError(
            f"set_name {set_name!r} is in the reserved {RESERVED_SET_PREFIX!r} "
            f"namespace: the server fills it from the source file and replaces "
            f"it when that file changes, so an edit here would be discarded. "
            f"Copy the set into one of your own to edit it."
        )

    shape_kind = roi.roi.WhichOneof("shape")
    if shape_kind is None:
        raise ValueError("Annotation has no geometry")
    if shape_kind not in _ACCEPTED_SHAPES:
        raise ValueError(
            f"Geometry {shape_kind!r} is not accepted by the annotation store "
            f"(accepted: {', '.join(sorted(_ACCEPTED_SHAPES))}). Instance "
            f"segmentation belongs in a label tensor."
        )

    geometry_msg = roi.roi
    if shape_kind == "text":
        # Normalised on a copy, so the caller's message is not edited under it
        # and every reader sees one newline convention.
        geometry_msg = ROI()
        geometry_msg.CopyFrom(roi.roi)
        text = normalize_caption(geometry_msg.text.text)
        # An empty caption marks nothing and has no extent to select.
        if not text.strip():
            raise ValueError("Caption text is empty")
        if len(text) > MAX_CAPTION_LEN:
            raise ValueError(
                f"Caption text is longer than {MAX_CAPTION_LEN} characters "
                f"({len(text)})"
            )
        geometry_msg.text.text = text

    # The plane pin is unvalidated: uint32 rules out a negative axis or index,
    # and the rank cannot be checked because this path binds no tensor -- a pin
    # naming an axis the tensor lacks simply matches nothing.

    return _PreparedRoi(
        roi_id=roi_id or uuid.uuid4().hex,
        set_name=set_name,
        label=roi.label,
        shape_kind=shape_kind,
        plane=dict(roi.plane),
        bbox=_roi_bbox(geometry_msg, shape_kind),
        # Canonical proto3 JSON, so the SPA and the SQL surface read the same
        # text and the sidecar can pass it through without re-encoding.
        geometry=json_format.MessageToJson(geometry_msg, indent=0).replace("\n", ""),
        props_json=roi.props_json or None,
        # Hex, not raw bytes: the token is opaque and a TEXT column keeps the row
        # legible to the SQL surface.
        drawn_against_version=(
            roi.drawn_against_version.hex()
            if roi.HasField("drawn_against_version")
            else None
        ),
        rev=roi.rev,
        roi=geometry_msg,
    )


def _roi_bbox(roi, shape_kind: str) -> List[float]:
    """Axis-aligned [x0, y0, x1, y1] in level-0 pixels."""
    if shape_kind == "point":
        p = roi.point
        return [p.x, p.y, p.x, p.y]
    if shape_kind == "text":
        # The anchor only: the text's extent is set by the font and the zoom, so
        # it is not image geometry and has no honest pixel box.
        p = roi.text.at
        return [p.x, p.y, p.x, p.y]
    if shape_kind == "rectangle":
        xs = (roi.rectangle.top_left.x, roi.rectangle.bottom_right.x)
        ys = (roi.rectangle.top_left.y, roi.rectangle.bottom_right.y)
        return [min(xs), min(ys), max(xs), max(ys)]
    if shape_kind == "ellipse":
        c, r = roi.ellipse.center, roi.ellipse.radius
        # Half-extents of the rotated ellipse's axis-aligned bounding box. The
        # supporting extent along x is max over t of |rx*cos(t)*cos(rot) -
        # ry*sin(t)*sin(rot)|, which is the hypotenuse of the two terms; y is
        # the same with the angles exchanged. At rot=0 this is (|rx|, |ry|), so
        # the boxes already stored are unchanged.
        cos, sin = math.cos(roi.ellipse.rotation), math.sin(roi.ellipse.rotation)
        hx = math.hypot(r.x * cos, r.y * sin)
        hy = math.hypot(r.x * sin, r.y * cos)
        return [c.x - hx, c.y - hy, c.x + hx, c.y + hy]
    if shape_kind == "polyline":
        points = roi.polyline.points
        if len(points) < 2:
            raise ValueError(f"Polyline needs at least 2 points, got {len(points)}")
        # The stroke width is geometry: a scribble marks the band of pixels the
        # brush covered, so the covered region extends width/2 past the vertices.
        # A bbox taken from the vertices alone would under-report a fat stroke.
        pad = abs(roi.polyline.width) / 2.0
    else:
        points = roi.polygon.points
        if len(points) < 3:
            raise ValueError(f"Polygon needs at least 3 points, got {len(points)}")
        pad = 0.0
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def _row_to_proto(row: Sequence) -> RoiAnnotation:
    """Rebuild a RoiAnnotation from a ``rois`` SELECT row."""
    (
        roi_id,
        array_id,
        set_name,
        label,
        plane,
        geometry,
        props_json,
        drawn_against_version,
        rev,
        created_at,
        updated_at,
    ) = row
    out = RoiAnnotation(
        roi_id=roi_id,
        array_id=array_id,
        set_name=set_name,
        label=label or "",
        props_json=props_json or "",
        rev=rev,
        created_at_unix_ms=_to_unix_ms(created_at),
        updated_at_unix_ms=_to_unix_ms(updated_at),
    )
    json_format.Parse(geometry, out.roi)
    if plane:
        out.plane.update(plane)
    if drawn_against_version:
        out.drawn_against_version = bytes.fromhex(drawn_against_version)
    return out


def _to_unix_ms(value: Optional[datetime]) -> int:
    """Epoch milliseconds for a naive-local DuckDB timestamp; 0 when absent."""
    if value is None:
        return 0
    if value.tzinfo is None:
        value = value.astimezone()
    return int(value.timestamp() * 1000)


class RoiStoreMixin:
    """ROI annotation methods of :class:`MetadataDatabase`.

    Needs ``_get_connection``, ``_get_cursor``, ``_write_lock`` and
    ``_max_rois_per_tensor`` from the class it is mixed into.
    """

    def _reconcile_roi_schema(
        self, conn: duckdb.DuckDBPyConnection, had_rois: bool
    ) -> None:
        """Bring an existing `rois` table up to this build's shape, or refuse.

        Persistence is what makes this necessary: the schema used to be rebuilt
        every boot, so changing it cost nothing. Now the file outlives the code,
        and `CREATE TABLE IF NOT EXISTS` against an older one is a silent no-op
        -- the server starts, reports SERVING, and every annotation read and
        write then fails on a missing column.
        """
        row = conn.execute(
            "SELECT value FROM catalog_meta WHERE key = 'roi_schema_version'"
        ).fetchone()
        if row is not None:
            stored = int(row[0])
        elif had_rois:
            # A file from before the marker existed, which is version 1 by
            # definition -- the marker arrived in the same release as v1.
            stored = 1
        else:
            stored = _ROI_SCHEMA_VERSION

        if stored > _ROI_SCHEMA_VERSION:
            raise AnnotationStoreError(
                f"The annotation catalog was written by a newer biopb "
                f"(rois schema v{stored}; this build understands "
                f"v{_ROI_SCHEMA_VERSION}). Upgrade, or point "
                f"catalog.store_path somewhere else. The file is untouched."
            )

        while stored < _ROI_SCHEMA_VERSION:
            upgrade = _ROI_MIGRATIONS.get(stored)
            if upgrade is None:
                raise AnnotationStoreError(
                    f"No migration from rois schema v{stored} to "
                    f"v{stored + 1}. The file is untouched."
                )
            logger.info("Migrating annotations from schema v%d", stored)
            upgrade(conn)
            stored += 1

        conn.execute(
            "INSERT OR REPLACE INTO catalog_meta VALUES ('roi_schema_version', ?)",
            [str(stored)],
        )
        self._verify_roi_columns(conn)

    @staticmethod
    def _verify_roi_columns(conn: duckdb.DuckDBPyConnection) -> None:
        """Refuse a `rois` table that is not the shape this build writes.

        The version marker only helps if every schema change remembers to bump
        it, and the edit that forgets is the same edit a hand-written column
        list would not have been updated in either. So the expectation is built
        from _ROIS_DDL itself, in a throwaway in-memory database: there is
        nothing here to keep in sync, and a forgotten bump surfaces at startup
        rather than as a Binder Error on the first annotation.
        """
        probe = duckdb.connect(":memory:")
        try:
            probe.execute(_ROIS_DDL)
            expected = {r[0] for r in probe.execute("DESCRIBE rois").fetchall()}
        finally:
            probe.close()
        actual = {r[0] for r in conn.execute("DESCRIBE rois").fetchall()}
        if missing := expected - actual:
            raise AnnotationStoreError(
                f"The annotation catalog is missing column(s) "
                f"{', '.join(sorted(missing))} that this build requires. Its "
                f"schema version claims to be current, so a migration is "
                f"missing rather than un-run. The file is untouched."
            )

    def _reset_derived_state(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Recompute what a formula owns, once the tables exist.

        Clearing `sources` used to happen here; _create_schema drops and
        recreates the table instead, which does the same job and additionally
        keeps its columns current.

        Runs on every open, so a fresh in-memory catalog and a reopened file
        take one path -- against an empty table it does nothing.
        """
        self._rederive_roi_columns(conn)

    def _rederive_roi_columns(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Recompute `shape_kind` and `bbox` from each row's stored geometry.

        `geometry` is the annotation; these two are `_roi_bbox` output cached in
        columns for the SQL surface. Deriving them again at open is what keeps
        that cache honest across a change to the formula -- adding rotation to
        Ellipse (biopb#935) moved every rotated ellipse's box, and this is the
        difference between that landing on restart and it needing a migration.
        """
        rows = conn.execute("SELECT array_id, roi_id, geometry FROM rois").fetchall()
        if not rows:
            return

        updates: List[List[object]] = []
        unreadable = 0
        for array_id, roi_id, geometry in rows:
            try:
                shape = ROI()
                json_format.Parse(geometry, shape)
                kind = shape.WhichOneof("shape")
                if kind is None:
                    raise ValueError("row carries no geometry")
                updates.append([kind, _roi_bbox(shape, kind), array_id, roi_id])
            except Exception as exc:
                # Leave the row alone rather than dropping it: a stale bounding
                # box costs a SQL filter, and the geometry the viewer draws from
                # is the column we could not read, so deleting would turn an
                # unreadable annotation into a missing one.
                unreadable += 1
                logger.warning(
                    "Kept ROI %s/%s with its stored bbox: %s", array_id, roi_id, exc
                )

        if updates:
            # One transaction, not one per row: on a file-backed catalog the
            # difference is an fsync per annotation at every startup.
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany(
                    "UPDATE rois SET shape_kind = ?, bbox = ? "
                    "WHERE array_id = ? AND roi_id = ?",
                    updates,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        logger.info(
            "Restored %d annotation(s)%s",
            len(rows),
            f", {unreadable} with an unreadable geometry" if unreadable else "",
        )

    # ------------------------------------------------------------------
    # ROI annotations
    # ------------------------------------------------------------------

    # Columns a client owns: rewritten verbatim by every update. Everything not
    # in this list is either identity (roi_id / array_id / source_id), set once
    # at creation (created_at), server-derived (rev / updated_at), or
    # catalog-derived (source_url / last_seen_at) -- and an UPDATE that does not
    # name a column cannot corrupt it. That is the point of splitting create
    # from update rather than doing one full-row INSERT OR REPLACE: the
    # "don't touch this on an update" rule is expressed by the statement itself
    # instead of by reconstruction logic that has to get every column right.
    _ROI_CLIENT_COLUMNS = (
        "set_name",
        "label",
        "shape_kind",
        "plane",
        "bbox",
        "geometry",
        "props_json",
        "drawn_against_version",
    )

    def put_rois(
        self,
        array_id: str,
        rois: Sequence[RoiAnnotation],
        *,
        check_rev: bool = False,
    ) -> Tuple[List[RoiAnnotation], List[RoiConflict]]:
        """Create or update a batch of annotations on one tensor.

        The whole batch is applied under the write lock so a client's "save this
        layer" lands as a unit -- that is how row-per-ROI storage still gives
        layer-level atomicity.

        ``check_rev`` makes each write conditional: an annotation whose ``rev``
        differs from the stored one is returned as a conflict and NOT applied,
        while the rest of the batch still lands. Without it, last writer wins.

        Args:
            array_id: Unversioned array_id every annotation belongs to.
            rois: Annotations to store. An empty ``roi_id`` mints a new uuid4.
            check_rev: Enable optimistic concurrency.

        Returns:
            ``(stored, conflicts)`` -- the stored records carry the server's
            roi_id / rev / timestamps.

        Raises:
            ValueError: On an empty array_id, a geometry this store does not
                accept, a mismatched per-ROI array_id, a duplicate roi_id in the
                batch, or a write that would push the tensor past
                ``max_rois_per_tensor``.
        """
        if not array_id:
            raise ValueError("array_id is required")
        _require_bare_array_id(array_id)

        # Validate and normalize everything BEFORE taking the lock: a batch is
        # all-or-nothing on validity, so a bad shape in the tenth annotation must
        # not leave the first nine written.
        prepared = [_prepare_roi(array_id, roi) for roi in rois]

        # A batch naming one roi_id twice is a client bug: the writes would
        # collapse to whichever came last, and the caller would get two "stored"
        # records for one row. Say so rather than silently keeping one.
        seen: Set[str] = set()
        for prep in prepared:
            if prep.roi_id in seen:
                raise ValueError(f"Duplicate roi_id in one batch: {prep.roi_id!r}")
            seen.add(prep.roi_id)

        conn = self._get_connection()
        source_id = array_id.split("/")[0]

        # The write lock serializes writers; it does NOT make the batch atomic,
        # because DuckDB autocommits each statement. Without an explicit
        # transaction a failure partway left the rows written so far behind, and
        # a concurrent reader (list_rois uses its own cursor and takes no lock)
        # watched a layer appear row by row. Cursors see the pre-commit snapshot,
        # so wrapping the whole body gives both all-or-nothing recovery and an
        # all-or-nothing view.
        with self._write_lock:
            conn.execute("BEGIN TRANSACTION")
            try:
                return self._put_rois_locked(
                    conn, array_id, source_id, prepared, check_rev
                )
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # pragma: no cover - rollback of a dead conn
                    # Never mask the original failure with a rollback error.
                    logger.exception("put_rois: ROLLBACK failed for %s", array_id)
                raise

    def _put_rois_locked(
        self,
        conn,
        array_id: str,
        source_id: str,
        prepared: List[_PreparedRoi],
        check_rev: bool,
    ) -> Tuple[List[RoiAnnotation], List[RoiConflict]]:
        """The body of :meth:`put_rois`, inside the lock and the transaction.

        Split out so the transaction is a plain try/except around one call rather
        than a second level of indentation over the whole method.
        """
        # Sampled under the lock, not before it: a writer that waited would
        # otherwise stamp times from before the wait, so a batch committing
        # LATER could carry an earlier updated_at than one that committed first
        # -- and last_seen_at, which the orphan clock reads, could move
        # backwards.
        now = datetime.now()

        stored: List[RoiAnnotation] = []
        conflicts: List[RoiConflict] = []
        in_catalog, source_url = self._observe_source(conn, source_id, now)

        # created_at is read for the RESPONSE only -- the update statement
        # does not carry it, so an existing row's value is preserved by not
        # being mentioned.
        existing = {
            roi_id: (rev, created_at, set_name)
            for roi_id, rev, created_at, set_name in conn.execute(
                "SELECT roi_id, rev, created_at, set_name FROM rois WHERE array_id = ?",
                [array_id],
            ).fetchall()
        }

        # The other half of the reserved-namespace guard: _prepare_roi refuses an
        # incoming reserved set_name, this refuses a write landing on a row that
        # is already in one. Needed because a put naming an existing roi_id takes
        # the UPDATE branch below and set_name is in _ROI_CLIENT_COLUMNS -- so
        # reusing an imported id would not COPY that row into the caller's set, it
        # would MOVE it out of the reserved one, beyond both the importer's reach
        # and the next re-import's replacement, without erroring. Cloning an
        # imported set therefore mints fresh ids (biopb/biopb#951).
        trespass = sorted(
            p.roi_id
            for p in prepared
            if p.roi_id in existing and is_reserved_set(existing[p.roi_id][2])
        )
        if trespass:
            raise ValueError(
                f"{', '.join(trespass)}: already stored in a reserved "
                f"{RESERVED_SET_PREFIX!r} set, which the server owns. Reusing the "
                f"id would move that row out of it rather than copy it -- give "
                f"the copy a new roi_id."
            )

        # Cap on the post-write count, so a batch cannot straddle the limit.
        # Imported rows are excluded: the cap exists to keep this an annotation
        # store rather than a segmentation store, and a user should not be
        # pushed toward it by rows they did not author -- cloning an imported
        # set, which is how editing one works, would be what trips it.
        authored = sum(
            1 for prior in existing.values() if not is_reserved_set(prior[2])
        )
        new_ids = {p.roi_id for p in prepared if p.roi_id not in existing}
        if authored + len(new_ids) > self._max_rois_per_tensor:
            raise ValueError(
                f"Annotation limit reached for {array_id}: "
                f"{authored} stored + {len(new_ids)} new exceeds "
                f"max_rois_per_tensor={self._max_rois_per_tensor}. This is an "
                f"annotation store -- a segmentation belongs in a label tensor."
            )

        assignments = ", ".join(f"{col} = ?" for col in self._ROI_CLIENT_COLUMNS)
        update_sql = (
            f"UPDATE rois SET {assignments}, rev = ?, updated_at = ? "
            f"WHERE array_id = ? AND roi_id = ?"
        )
        insert_sql = (
            "INSERT INTO rois "
            f"(roi_id, array_id, source_id, {', '.join(self._ROI_CLIENT_COLUMNS)}, "
            "rev, created_at, updated_at, source_url, last_seen_at) "
            f"VALUES ({', '.join('?' * (len(self._ROI_CLIENT_COLUMNS) + 8))})"
        )

        for prep in prepared:
            prior = existing.get(prep.roi_id)
            if prior is not None and check_rev and prep.rev != prior[0]:
                conflicts.append(RoiConflict(roi_id=prep.roi_id, stored_rev=prior[0]))
                continue

            values = prep.column_values(self._ROI_CLIENT_COLUMNS)
            if prior is None:
                rev, created_at = 1, now
                conn.execute(
                    insert_sql,
                    [prep.roi_id, array_id, source_id, *values, rev, now, now]
                    # A fresh row is only "seen" if the catalog answered;
                    # inventing a sighting would reset an orphan clock.
                    + [source_url, now if in_catalog else None],
                )
            else:
                rev, created_at = prior[0] + 1, prior[1]
                conn.execute(update_sql, [*values, rev, now, array_id, prep.roi_id])
            stored.append(prep.to_proto(array_id, rev, created_at, now))

        conn.execute("COMMIT")

        logger.debug(
            "put_rois: %s stored, %s conflicts on %s",
            len(stored),
            len(conflicts),
            array_id,
        )
        return stored, conflicts

    @staticmethod
    def _observe_source(
        conn, source_id: str, now: datetime
    ) -> Tuple[bool, Optional[str]]:
        """Record a catalog sighting of *source_id*.

        Returns ``(in_catalog, source_url)``. The two are separate answers: a
        source can be present and carry no URL, and it is *presence* that a
        fresh row's ``last_seen_at`` turns on. Reporting only the URL would make
        the caller decide a sighting from a value that does not mean one.

        Presence in the catalog is evidence; absence is not (progressive
        discovery, an unmounted drive, a proxy upstream that is down). So this
        writes only on presence, and a source the catalog cannot answer for
        leaves every stored row exactly as it was.

        On presence it does two things in one statement, for every row of the
        source:

        * refreshes ``source_url`` to whatever the catalog now calls the source,
          which also backfills the NULL a write could not resolve. The catalog's
          answer wins because this column is a human-readable *label* for a
          source_id, not an identifier -- nothing matches on it (this statement
          joins on source_id) -- and a label is only useful if it says what the
          source is called now. Since the write happens only on presence, the
          stored value freezes by itself at the last sighting, which is the one a
          report should show; it used to freeze at the FIRST, so a source that was
          renamed and later vanished got reported under a name the catalog had
          abandoned long before it went away. A catalog row that names nothing
          (NULL or empty) leaves the stored label alone -- an unnamed source is
          not a rename.
        * stamps ``last_seen_at``. :meth:`mark_sources_seen` is this statement
          over the whole catalog; the difference is only its gate, which a
          *presence* observation does not need -- a conclusion about absence
          does.
        """
        row = conn.execute(
            "SELECT source_url FROM sources WHERE source_id = ? AND source_id IN "
            "(SELECT source_id FROM source_confirmation WHERE confirmed)",
            [source_id],
        ).fetchone()
        if row is None:
            logger.debug(
                "put_rois: %s is not in the catalog; annotations keep whatever "
                "source_url / last_seen_at they already had",
                source_id,
            )
            return False, None
        source_url = row[0] or None  # "" names nothing; treat it as absent
        # One source's half of what mark_sources_seen() does for the whole
        # catalog -- keep the two statements the same shape.
        conn.execute(
            "UPDATE rois SET source_url = COALESCE(?, source_url), last_seen_at = ? "
            "WHERE source_id = ?",
            [source_url, now, source_id],
        )
        return True, source_url

    # ------------------------------------------------------------------
    # The orphan clock
    # ------------------------------------------------------------------

    def mark_sources_seen(self) -> int:
        """Stamp a catalog sighting on every annotation whose source is present.

        :meth:`_observe_source` generalized from one source to all of them, and
        the reason the clock means anything: a write only ever refreshes the
        tensor being drawn on, so without this an untouched annotation would
        look unseen however often its source is rescanned.

        Presence is the only evidence recorded. A source the catalog does not
        list is left entirely alone -- absence is not deletion (progressive
        discovery, an unmounted drive, an upstream that is down), so it must
        register as "no news", not as a sighting that failed to happen.

        The caller owns the gate: this is only meaningful just after a full scan
        completed, which is where SourceManager calls it from.
        """
        conn = self._get_connection()
        with self._write_lock:
            seen = conn.execute(
                "UPDATE rois SET "
                "source_url = COALESCE(NULLIF(s.source_url, ''), rois.source_url), "
                "last_seen_at = ? FROM sources s WHERE rois.source_id = s.source_id "
                "AND s.source_id IN "
                "(SELECT source_id FROM source_confirmation WHERE confirmed) "
                "RETURNING rois.roi_id",
                [datetime.now()],
            ).fetchall()
        logger.debug("mark_sources_seen: %d annotation(s) observed", len(seen))
        return len(seen)

    def unseen_rois(self, before: datetime) -> List[UnseenRois]:
        """Annotations whose source has not been observed since *before*.

        Grouped per tensor, because that is the unit a person confirms: "47
        annotations on /data/plate3.zarr, last seen 12 June". ``source_url`` is
        what makes such a line actionable at all -- ``array_id`` is a SHA-256
        and cannot be inverted -- and it is NULL for rows written before their
        source was ever in the catalog, which is the one orphan nobody can be
        told about.

        Age is ``COALESCE(last_seen_at, created_at)``: a row whose source has
        never once appeared has no sighting to measure from, and treating that
        as "infinitely fresh" would make exactly the strongest orphan immortal.
        """
        rows = (
            self._get_cursor()
            .execute(
                "SELECT source_id, any_value(source_url), array_id, count(*), "
                "max(COALESCE(last_seen_at, created_at)) AS seen FROM rois "
                f"WHERE {_UNSEEN_PREDICATE} "
                "GROUP BY source_id, array_id ORDER BY seen, array_id",
                [before, RESERVED_SET_PREFIX],
            )
            .fetchall()
        )
        return [
            UnseenRois(
                source_id=source_id,
                source_url=source_url,
                array_id=array_id,
                count=count,
                last_seen_at=seen,
            )
            for source_id, source_url, array_id, count, seen in rows
        ]

    def prune_unseen(self, before: datetime) -> int:
        """Delete the annotations :meth:`unseen_rois` reports, returning how many.

        Destructive and unconditional -- every gate (is the catalog complete, is
        auto-prune even on, has a person confirmed) belongs to the caller. Kept
        that way so the dry-run path and the real one share one predicate
        instead of two that can drift.
        """
        conn = self._get_connection()
        with self._write_lock:
            deleted = conn.execute(
                f"DELETE FROM rois WHERE {_UNSEEN_PREDICATE} RETURNING roi_id",
                [before, RESERVED_SET_PREFIX],
            ).fetchall()
        if deleted:
            logger.info("prune_unseen: removed %d annotation(s)", len(deleted))
        return len(deleted)

    def list_rois(
        self, array_id: str, set_name: str = ""
    ) -> Tuple[List[RoiAnnotation], bool]:
        """A tensor's annotations.

        ``set_name`` selects one layer, and is the only way to read a reserved
        set: an unqualified list covers the client-owned sets alone.

        No plane or bbox filter: a client hit-tests and re-renders from the
        resident set. Analytic slicing is the SQL surface's job.

        Returns:
            ``(rois, truncated)``. ``truncated`` is true when the tensor holds
            more rows in scope than the per-tensor cap, in which case the result
            is clipped.
        """
        if not array_id:
            raise ValueError("array_id is required")
        _require_bare_array_id(array_id)

        sql = (
            "SELECT roi_id, array_id, set_name, label, plane, geometry, "
            "props_json, drawn_against_version, rev, created_at, updated_at "
            "FROM rois WHERE array_id = ?"
        )
        params: List[object] = [array_id]
        if set_name:
            sql += " AND set_name = ?"
            params.append(set_name)
        else:
            # Reserved sets are outside the write cap and their rows carry the
            # registration timestamp, so in created_at order a large import
            # fills the read cap ahead of anything a user drew.
            sql += " AND NOT starts_with(set_name, ?)"
            params.append(RESERVED_SET_PREFIX)
        # Stable order so a client diffing two reads sees no spurious churn.
        sql += " ORDER BY created_at, roi_id LIMIT ?"
        params.append(self._max_rois_per_tensor + 1)

        rows = self._get_cursor().execute(sql, params).fetchall()
        truncated = len(rows) > self._max_rois_per_tensor
        if truncated:
            rows = rows[: self._max_rois_per_tensor]
        return [_row_to_proto(row) for row in rows], truncated

    def list_roi_sets(self, array_id: str) -> List[Tuple[str, int]]:
        """Every set on a tensor with its row count, ordered by name.

        Counts the stored rows, not what :meth:`list_rois` returns: an
        unqualified list carries no reserved set and either list can be clipped
        by the read cap, so this is how a client learns a set is there and what
        to name to read it.
        """
        if not array_id:
            raise ValueError("array_id is required")
        _require_bare_array_id(array_id)

        rows = (
            self._get_cursor()
            .execute(
                "SELECT set_name, COUNT(*) FROM rois WHERE array_id = ? "
                "GROUP BY set_name ORDER BY set_name",
                [array_id],
            )
            .fetchall()
        )
        return [(name, int(count)) for name, count in rows]

    def delete_rois(
        self, array_id: str, roi_ids: Iterable[str] = (), set_name: str = ""
    ) -> List[str]:
        """Delete annotations, returning the ids actually removed.

        With ``roi_ids``, deletes exactly those. Without, deletes every
        annotation on ``array_id`` -- narrowed to ``set_name`` when given, which
        is how a client drops a whole layer.

        A reserved set is server-owned and cannot be deleted through here, named
        either directly or by one of its ids. An unqualified "clear this tensor"
        is the one case that neither refuses nor deletes: see below.
        """
        if not array_id:
            raise ValueError("array_id is required")
        _require_bare_array_id(array_id)
        if is_reserved_set(set_name):
            raise ValueError(
                f"set_name {set_name!r} is reserved: the server fills it from "
                f"the source file and replaces it when that file changes, so "
                f"there is nothing here for a client to delete."
            )

        roi_ids = list(roi_ids)
        conn = self._get_connection()
        with self._write_lock:
            if roi_ids:
                placeholders = ", ".join("?" for _ in roi_ids)
                # Ids reach a reserved row without naming its set, so this path
                # needs its own check. Refused, not skipped: a delete that
                # silently drops part of what it was handed reports success for
                # work it did not do.
                reserved = sorted(
                    row[0]
                    for row in conn.execute(
                        f"SELECT roi_id FROM rois WHERE array_id = ? "
                        f"AND roi_id IN ({placeholders}) "
                        f"AND starts_with(set_name, ?)",
                        [array_id, *roi_ids, RESERVED_SET_PREFIX],
                    ).fetchall()
                )
                if reserved:
                    raise ValueError(
                        f"{', '.join(reserved)}: stored in a reserved "
                        f"{RESERVED_SET_PREFIX!r} set, which the server owns and "
                        f"refills from the source file. Nothing was deleted."
                    )
                sql = (
                    f"DELETE FROM rois WHERE array_id = ? "
                    f"AND roi_id IN ({placeholders}) RETURNING roi_id"
                )
                params: List[object] = [array_id, *roi_ids]
            elif set_name:
                sql = (
                    "DELETE FROM rois WHERE array_id = ? AND set_name = ? "
                    "RETURNING roi_id"
                )
                params = [array_id, set_name]
            else:
                # "Clear this tensor" means the caller's own annotations. A
                # reserved set is the file's copy, not theirs, and re-import
                # would restore it regardless -- so it is scoped out rather than
                # deleted. Not a silent partial failure the way the id path
                # would be: nothing here was addressed to it.
                sql = (
                    "DELETE FROM rois WHERE array_id = ? "
                    "AND NOT starts_with(set_name, ?) RETURNING roi_id"
                )
                params = [array_id, RESERVED_SET_PREFIX]
            deleted = [row[0] for row in conn.execute(sql, params).fetchall()]

        logger.debug("delete_rois: removed %s from %s", len(deleted), array_id)
        return deleted

    def discard_array_rois(self, array_id: str) -> int:
        """Delete every annotation on array_id, reserved sets included.

        For an array_id that is gone for good -- a discarded or expired
        upload -- rather than merely unseen in the current catalog scan.
        :meth:`sync_source_removed` deliberately lets hand-drawn annotations
        outlive a *source* going offline, since it may come back
        (biopb/biopb#951); a reclaimed upload's array_id does not come back,
        and its name may be handed to an unrelated tensor next, so nothing
        here should survive to be misread as that tensor's own (#1155).
        """
        _require_bare_array_id(array_id)
        conn = self._get_connection()
        with self._write_lock:
            deleted = conn.execute(
                "DELETE FROM rois WHERE array_id = ? RETURNING roi_id", [array_id]
            ).fetchall()
        logger.debug("discard_array_rois: removed %s from %s", len(deleted), array_id)
        return len(deleted)
