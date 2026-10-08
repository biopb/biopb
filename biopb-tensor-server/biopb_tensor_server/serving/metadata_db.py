"""DuckDB metadata database for efficient source filtering.

Provides indexed SQL queries against source metadata for large catalogs
(>100k sources). Replaces O(n) in-memory scans with indexed DuckDB queries.

Database Schema:
- sources table with indexed fields (source_id, source_url)
- JSON column for full metadata access via DuckDB JSON operators
- Shape summary column for quick size estimates
- rois table: user-drawn ROI annotations, one row per ROI, anchored on the
  unversioned array_id
- decode_rates table: measured full-resolution decode throughput per array_id,
  the input for cache.cheap_decode_mbps (core/retention.py)

Persistence:
- In memory by default. Given a store_path the connection is file-backed, which
  is how annotations and decode rates survive a restart.
- `sources` rides along because DuckDB has one database per connection and the
  sandbox below blocks ATTACH. It is truncated on open, and the rois table's
  derived columns are recomputed there.
- The decode rates live here rather than beside the cache segments they describe
  because the cache directory is the operator's to delete: clearing it should
  cost the bytes, not the measurements that took a run to collect.

Security Model:
- DuckDB connection runs with enable_external_access=False, so all file/network
  access (read_csv, read_text, glob, COPY, ATTACH, extension loading) is blocked
  at the engine level. This is the primary defense against file exfiltration.
- Only the 'sources', 'rois' and 'decode_rates' tables are accessible
  (keyword/table denylist; defense in depth). 'rois' and 'decode_rates' are
  readable here for analysis; every write goes through put_rois/delete_rois or
  save_decode_rates, never through this surface.
- Forbidden keywords: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, EXECUTE
- No subqueries referencing external tables
- Query timeout enforced

Usage:
    db = MetadataDatabase()
    db.sync_source_added(source_id, adapter)
    table = db.query("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import duckdb
import numpy as np
import pyarrow as pa
from biopb.image.annotation_pb2 import RoiAnnotation, RoiConflict
from biopb.image.roi_pb2 import ROI
from biopb.tensor._catalog_rows import SOURCE_ROW_COLUMNS
from google.protobuf import json_format

from biopb_tensor_server.core.adapter_base import to_catalog_url
from biopb_tensor_server.core.attachments import Attachments
from biopb_tensor_server.core.errors import AnnotationStoreError
from biopb_tensor_server.core.labels import last_named_segment

if TYPE_CHECKING:
    from biopb_tensor_server.core.adapter_base import SourceAdapter
    from biopb_tensor_server.core.discovery import SourceClaim
    from biopb_tensor_server.sources.pending_rows import PendingRow

logger = logging.getLogger(__name__)

# Bump when a field of the persisted source row or payload changes meaning (an
# added key needs none). A mismatch drops ``source_catalog`` whole at open.
SOURCE_CATALOG_FORMAT = 4

# The columns ``sources`` publishes, in table order. ``source_catalog`` carries
# them first, then the claim.
_SOURCE_COLUMNS = (
    "source_id, source_url, source_type, indexed_at, metadata_json, "
    "is_resolved, unresolved_reason, unresolved_error, tensors"
)
# The root a source sits under, and its path beneath it, which the view turns into
# ``source_url`` against that root's ``catalog_roots`` row. A source that sits under
# none of the roots the server knows (the scratch source, one registered through the
# API) is under the built-in root with no url of its own, whose rows' ``rel`` is the
# whole url.
INTERNAL_ROOT_ID = "internal"
_LOCATION_COLUMN_NAMES = ("root_id", "rel")
# The claim, the columns a source with no claim leaves NULL (bar the last two).
_CLAIM_COLUMN_NAMES = (
    "primary_path",
    "member_paths",
    "extra_config",
    "signature",
    "payload",
    "last_seen",
    "epoch",
)
# The columns a row is written with, in the order every writer builds its values.
# ``source_url`` is the url the adapter shows: it is only the ``rel`` of a row under
# the built-in root, since every other row's url is its root's.
_ROW_COLUMN_NAMES = (
    "source_id",
    "source_url",
    "source_type",
    "indexed_at",
    "metadata_json",
    "is_resolved",
    "unresolved_reason",
    "tensors",
    "unresolved_error",
)
# What the table stores of those: all but ``source_url``.
_STORED_ROW_COLUMN_NAMES = _ROW_COLUMN_NAMES[:1] + _ROW_COLUMN_NAMES[2:]
_ALL_COLUMN_NAMES = (
    _STORED_ROW_COLUMN_NAMES + _LOCATION_COLUMN_NAMES + _CLAIM_COLUMN_NAMES
)
_ALL_COLUMNS = ", ".join(_ALL_COLUMN_NAMES)
# Everything but the key, for updating a row in place.
_UPDATE_SET = ", ".join(f"{c} = ?" for c in _ALL_COLUMN_NAMES[1:])
# The same, for an ``INSERT ... ON CONFLICT DO UPDATE`` that finds the row there.
_UPSERT_SET = ", ".join(f"{c} = excluded.{c}" for c in _ALL_COLUMN_NAMES[1:])
# A write without a record sets the public columns and leaves the location and the
# claim alone, bar the ``rel`` of a row under the built-in root, which is its url.
_LISTING_COLUMN_NAMES = _ROW_COLUMN_NAMES[2:]
_LISTING_SET = (
    ", ".join(f"{c} = ?" for c in _LISTING_COLUMN_NAMES)
    + f", rel = CASE WHEN root_id = '{INTERNAL_ROOT_ID}' THEN ? ELSE rel END"
)
_LISTING_UPSERT_SET = (
    ", ".join(f"{c} = excluded.{c}" for c in _LISTING_COLUMN_NAMES)
    + f", rel = CASE WHEN source_catalog.root_id = '{INTERNAL_ROOT_ID}' "
    "THEN excluded.rel ELSE source_catalog.rel END"
)
# ``_SOURCE_COLUMNS`` for a ``source_catalog`` row (alias ``c``) joined to its root
# (alias ``r``): the url is the root's, then the row's path beneath it.
_VIEW_SOURCE_COLUMNS = _SOURCE_COLUMNS.replace(
    "source_url",
    "CASE WHEN r.root_url = '' THEN c.rel WHEN c.rel = '.' THEN r.root_url "
    "ELSE r.root_url || '/' || c.rel END AS source_url",
).replace("source_id", "c.source_id", 1)
# A row shows against its root only: one whose root is gone is not listed.
_VIEW_FROM = "FROM source_catalog c JOIN catalog_roots r ON c.root_id = r.root_id"


def _sources_view_ddl() -> str:
    """The published ``sources`` view: the public columns only."""
    return f"CREATE VIEW sources AS SELECT {_VIEW_SOURCE_COLUMNS} {_VIEW_FROM}"


def _array_id(tensor: Dict[str, Any]) -> str:
    """Sort key for a ``tensors`` entry."""
    return tensor["array_id"]


def _confirmation_view_ddl(run_epoch: int) -> str:
    """Whether each source was verified this run, apart from the published view.

    A row under a persisted root is *confirmed* when it was written this run or its
    root's walk finished this run; any other row was seen this run by construction.
    Kept out of ``sources`` because that is a published schema, and out of the
    allowed tables because only the server's own bookkeeping asks.
    """
    return (
        "CREATE VIEW source_confirmation AS "
        "SELECT c.source_id, "
        f"(NOT r.persisted OR greatest(c.epoch, r.epoch) = {int(run_epoch)}) "
        "AS confirmed "
        "FROM source_catalog c JOIN catalog_roots r ON c.root_id = r.root_id"
    )


@dataclass(frozen=True)
class CatalogRecord:
    """Where a source sits in the catalog, and what it persists beside its row.

    ``root_id`` and ``rel``: the root it sits under and its path beneath it.

    ``claim`` and ``signature``: only a source under a persisted root has them. The
    signature is the claim's member signature taken when the claim was made, before
    the parse, in the persisted form (no ``st_dev``): a file that changes during
    registration must not be stamped with its new identity beside the old metadata.

    ``cloud``: the claim sits under a cloud root, whose rows keep no payload (a
    restart never trusts one resolved, so there is nothing to skip).
    """

    claim: Optional[SourceClaim]
    signature: Dict[str, Tuple[Any, ...]]
    root_id: str
    rel: str
    cloud: bool = False


# Opening a persistent catalog is retried this many times: a DuckDB lock held by
# a server on its way down clears in about a second, and a restart race is the
# one open failure that fixes itself.
_OPEN_ATTEMPTS = 3
_OPEN_RETRY_SECONDS = 0.5

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

# Shape of the `decode_rates` table. Bumping this DROPS the table rather than
# migrating it: every row is re-measurable by reading, so a schema change costs
# one warmup, where the migration ladder `rois` needs exists because nothing can
# reproduce an annotation.
#
# Bumped to 2 by the `@labels` marking, which is not a column change but does
# change what a stored array_id means: a label set's rows are keyed on an id
# that is never minted again. Dropped rather than rewritten, since that is what
# this table is for.
_DECODE_RATES_SCHEMA_VERSION = 2

_DECODE_RATES_DDL = """
CREATE TABLE IF NOT EXISTS decode_rates (
    -- The tensor whose full-resolution reads were timed, in the chunk_id's own
    -- array_id form -- which is the native pyramid level's id where there is
    -- one, so a compressed low level and a raw level 0 are separate rows.
    array_id TEXT PRIMARY KEY,
    -- EMA of MB/s over that tensor's full-resolution reads.
    mbps DOUBLE NOT NULL,
    -- How many reads are behind the EMA. Part of the answer, not bookkeeping:
    -- a rate off two samples is still settling and should not be read as a
    -- verdict on the format.
    samples BIGINT NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
"""

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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types, and bytes."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            # Try to decode as UTF-8, otherwise use base64
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                import base64

                return base64.b64encode(obj).decode("ascii")
        # Catch-all: indicate unserializable type
        return f"Unserializable {type(obj).__qualname__}"


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

    # The plane pin is unvalidated: uint32 rules out a negative axis or index,
    # and the rank cannot be checked because this path binds no tensor -- a pin
    # naming an axis the tensor lacks simply matches nothing.

    return _PreparedRoi(
        roi_id=roi_id or uuid.uuid4().hex,
        set_name=set_name,
        label=roi.label,
        shape_kind=shape_kind,
        plane=dict(roi.plane),
        bbox=_roi_bbox(roi.roi, shape_kind),
        # Canonical proto3 JSON, so the SPA and the SQL surface read the same
        # text and the sidecar can pass it through without re-encoding.
        geometry=json_format.MessageToJson(roi.roi, indent=0).replace("\n", ""),
        props_json=roi.props_json or None,
        # Hex, not raw bytes: the token is opaque and a TEXT column keeps the row
        # legible to the SQL surface.
        drawn_against_version=(
            roi.drawn_against_version.hex()
            if roi.HasField("drawn_against_version")
            else None
        ),
        rev=roi.rev,
        roi=roi.roi,
    )


def _roi_bbox(roi, shape_kind: str) -> List[float]:
    """Axis-aligned [x0, y0, x1, y1] in level-0 pixels."""
    if shape_kind == "point":
        p = roi.point
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


class MetadataDatabase:
    """In-memory DuckDB for source metadata filtering.

    Thread-safe: All operations are protected by a lock.
    Lazy initialization: Database created on first access.

    The metadata DB is mandatory (biopb/biopb#225): it is the canonical
    source-browsing surface (``client.query``), so there is no
    off switch -- constructing this object means the catalog is live.

    Args:
        max_query_results: Safety cap on returned rows (truncation signaled via schema metadata)
        query_timeout_ms: Query execution timeout in milliseconds
        max_rois_per_tensor: Cap on stored annotations per tensor. Deliberately
            human-scale: it is the line between an annotation store and an
            object store, and it is what lets the read path be a single
            whole-set fetch.

    Example:
        db = MetadataDatabase()
        db.sync_source_added('plate-001', adapter)
        table = db.query(
            "SELECT source_id FROM sources WHERE tensors[1].dtype = 'uint16'"
        )
    """

    # The public catalog: what the ``catalog`` flight lists and SQL may read.
    # ``rois`` is deliberately absent -- annotations are private data, gated
    # per source on the ``roi`` flight, and a query has no source to authorize
    # against (biopb/biopb#1010). Enforced on DuckDB's own parse of the
    # statement (``_validate_query``), never on the SQL text.
    ALLOWED_TABLES: Set[str] = {"sources", "decode_rates"}

    # Table-valued functions a query may use. ``unnest`` is the documented
    # per-tensor idiom (``FROM sources, UNNEST(tensors)``); the rest generate
    # rows from nothing. Everything else -- above all the file and network
    # readers -- is refused here, and ``enable_external_access=false`` on the
    # connection is the second wall behind that.
    ALLOWED_TABLE_FUNCTIONS: Set[str] = {"unnest", "range", "generate_series"}

    def __init__(
        self,
        max_query_results: int = 100000,
        query_timeout_ms: int = 30000,
        max_rois_per_tensor: int = 5000,
        store_path: Optional[Path] = None,
        annotations_enabled: bool = True,
        checkpoint_threshold_mb: int = 1024,
        restore_sources: bool = False,
    ):
        self._checkpoint_threshold_mb = checkpoint_threshold_mb
        #: Keep ``source_catalog`` across a restart (``catalog.restore``): its rows
        #: are read back by :meth:`restorable_rows`, not cleared at open.
        self.restore_sources = restore_sources and store_path is not None
        #: How many times the file has been opened: what a row's ``epoch`` and its
        #: root's are compared with to say a row was confirmed this run.
        self.run_epoch = 0
        self._max_query_results = max_query_results
        self._query_timeout_ms = query_timeout_ms
        self._max_rois_per_tensor = max_rois_per_tensor
        self._annotations_enabled = annotations_enabled
        # None -> in-memory, and the annotations die with the process.
        self._store_path = Path(store_path) if store_path else None
        # A server not serving the annotation actions does not offer them
        # through the SQL surface either. Empty rows would be the wrong answer:
        # the table is unserved, not unpopulated, and a query cannot tell those
        # apart from a result set.
        self.allowed_tables: Set[str] = set(self.ALLOWED_TABLES)
        if not annotations_enabled:
            self.allowed_tables.discard("rois")

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._write_lock = threading.Lock()  # Lock for write operations only
        self._initialized = False
        # Lists a source's tensors; see bind_registry.
        self._registry: Any = None

        logger.info(
            "MetadataDatabase enabled (DuckDB backend will initialize on first access)"
        )

        # Pending query results for DoGet (stored by ticket)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Lazy initialization of DuckDB connection.

        Returns the shared connection for write operations.
        For reads, use _get_cursor() which returns thread-safe cursors.
        """
        if self._conn is None:
            with self._write_lock:
                if self._conn is None:
                    # Built fully before it is published: the check above is
                    # unlocked, so a reader would otherwise be handed a cursor
                    # on a database mid-open -- a window the derived-column pass
                    # makes long enough to matter.
                    conn = self._open_database()
                    self._create_schema(conn)
                    self._reset_derived_state(conn)
                    self._conn = conn
                    self._initialized = True
                    logger.info(
                        "MetadataDatabase initialized (%s)",
                        self._store_path or "in-memory",
                    )
        return self._conn

    def open(self) -> None:
        """Open the database now, so a bad store fails at startup.

        The connection is otherwise built on first use, which for a persistent
        store would put :class:`AnnotationStoreError` in front of whichever
        request happened to touch the catalog first rather than in front of the
        operator starting the server.

        Raises:
            AnnotationStoreError: a configured store that will not open.
        """
        self._get_connection()

    def _connect(self, target: str) -> duckdb.DuckDBPyConnection:
        """Open *target* with external access disabled.

        Disabling it is the real defense against file exfiltration via read_csv
        / read_text / glob / COPY / ATTACH etc., which the keyword denylist in
        _validate_query cannot reliably cover (e.g. comma-joins like `FROM
        sources, read_text('/etc/passwd')` slip past the FROM-only table check).
        Once disabled it cannot be re-enabled within a running instance, so a
        `SET enable_external_access=true` in a query is rejected. The server
        itself needs no external access: it only does parameterized
        INSERT/DELETE and JSON-operator SELECTs.

        It does not stop DuckDB opening its OWN database file, which is what
        makes a persistent catalog possible without reopening the sandbox.
        """
        return duckdb.connect(
            target,
            config={
                "enable_external_access": False,
                # A checkpoint rewrites the database file and stalls every writer
                # behind it. DuckDB's own 16 MB default fires every few dozen
                # sources of a scan (a row carries up to hundreds of KB of
                # metadata), roughly doubling its time.
                "checkpoint_threshold": f"{self._checkpoint_threshold_mb}MiB",
            },
        )

    def _open_database(self) -> duckdb.DuckDBPyConnection:
        """The connection, from a file when one is configured.

        Two things this deliberately does not do.

        It never **moves the file aside** to start clean. DuckDB raises the same
        ``IOException`` for a corrupt file and for one another process holds,
        and only the first of those wants the file touched. Getting it wrong on
        a lock is the bad case: the rename succeeds while the other server has
        the file open, so it keeps writing to the renamed inode while this one
        starts a fresh catalog at the original path, and the annotations split
        across two files with nothing to say so.

        It never **falls back to memory**. ``catalog.persist`` is a promise
        about durability; serving anyway would keep the server up while every
        ROI drawn on it went to a catalog that disappears at the next restart,
        and that loss surfaces a day later with the work already gone. So this
        raises, and the operator who wants a session-only store asks for one.

        The retry is the only distinction available between the four causes. A
        lock held by a server on its way down clears within a second -- a
        restart race is the one open failure that resolves itself -- while
        corruption, a permission problem and a version mismatch do not.
        """
        if self._store_path is None:
            return self._connect(":memory:")

        for attempt in range(1, _OPEN_ATTEMPTS + 1):
            try:
                self._store_path.parent.mkdir(parents=True, exist_ok=True)
                return self._connect(str(self._store_path))
            except Exception as exc:
                if attempt == _OPEN_ATTEMPTS:
                    raise AnnotationStoreError(
                        f"Could not open the catalog {self._store_path} "
                        f"after {_OPEN_ATTEMPTS} attempts: {exc}. The file has "
                        f"been left untouched. Restore it, fix its permissions, "
                        f"match the DuckDB version that wrote it, or stop the "
                        f"other server holding it -- or set "
                        f'"catalog": {{"persist": false}} to run with a '
                        f"session-only catalog."
                    ) from exc
                logger.warning(
                    "Catalog %s did not open (attempt %d/%d): %s",
                    self._store_path,
                    attempt,
                    _OPEN_ATTEMPTS,
                    exc,
                )
                time.sleep(_OPEN_RETRY_SECONDS)
        raise AssertionError("unreachable")  # pragma: no cover

    @property
    def store_path(self) -> Optional[Path]:
        """The file backing this catalog, or None when it is in memory."""
        return self._store_path

    @property
    def annotations_persisted(self) -> bool:
        """Whether drawn ROIs reach a file, for ``health``.

        The catalog being file-backed is not enough: a server with the
        annotation actions off holds one for `decode_rates` alone, and
        answering True there would promise durability for rows it will not
        accept in the first place.
        """
        return self._store_path is not None and self._annotations_enabled

    def _get_cursor(self) -> duckdb.DuckDBPyConnection:
        """Get a cursor for thread-safe read operations.

        DuckDB cursors (created via conn.cursor()) are thread-safe and can
        execute concurrently. This allows parallel reads without locking.
        """
        return self._get_connection().cursor()

    def _create_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create the sources and rois tables and their indexes.

        `sources` is dropped first. It is scan output -- discovery repopulates
        it, and it was being truncated on open anyway -- so rebuilding it makes
        a change to its columns free, where IF NOT EXISTS against an older file
        would silently keep the old shape.

        `rois` cannot be rebuilt, so it is versioned instead: see
        :meth:`_reconcile_roi_schema`.
        """
        # `sources` is a view over `source_catalog`, so it goes first. An older
        # build left a physical table of that name.
        if conn.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = 'sources'"
        ).fetchone():
            conn.execute("DROP TABLE sources")
        conn.execute("DROP VIEW IF EXISTS sources")
        conn.execute("DROP VIEW IF EXISTS source_confirmation")
        # A build with a separate table for the sources that have no claim left it
        # behind.
        conn.execute("DROP TABLE IF EXISTS sources_volatile")
        self._create_source_catalog(conn)
        conn.execute(_sources_view_ddl())
        conn.execute(_confirmation_view_ddl(self.run_epoch))

        # User-drawn ROI annotations, one row per ROI. A sibling table,
        # deliberately NOT a field inside a source row: sources.metadata_json is
        # adapter-produced and rewritten by the INSERT OR REPLACE in
        # sync_source_added(), so an annotation parked there would be destroyed
        # by the next rescan. Whether the annotations predate this build has to
        # be asked before the CREATE, which is what makes them indistinguishable
        # afterwards.
        had_rois = bool(
            conn.execute(
                "SELECT 1 FROM duckdb_tables() WHERE table_name = 'rois'"
            ).fetchone()
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(_ROIS_DDL)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rois_array ON rois(array_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rois_source ON rois(source_id)")
        self._reconcile_roi_schema(conn, had_rois)

        # Reserved sets are scan output like `sources` itself -- derived from a
        # file and rewritten by sync_source_added -- so they are cleared at open
        # rather than carried forward, or the window before re-registration
        # holds last run's imported rows next to an empty `sources`, possibly
        # derived by code this build no longer runs (biopb/biopb#951).
        #
        # AFTER the schema check, not beside the DROP above, because that check
        # can refuse the catalog -- and it promises "The file is untouched" when
        # it does. A delete before it would have made that a lie, and would have
        # run against a schema this build has not established it understands.
        conn.execute(
            "DELETE FROM rois WHERE starts_with(set_name, ?)", [RESERVED_SET_PREFIX]
        )

        # Measured decode throughput. Versioned like `rois` so an older file
        # cannot silently present the wrong columns, but reconciled by dropping
        # rather than migrating -- see _DECODE_RATES_SCHEMA_VERSION.
        stored_version = conn.execute(
            "SELECT value FROM catalog_meta WHERE key = 'decode_rates_schema_version'"
        ).fetchone()
        if stored_version is not None and stored_version[0] != str(
            _DECODE_RATES_SCHEMA_VERSION
        ):
            logger.info(
                "decode_rates schema %s != %s; dropping the measurements, which "
                "the next run re-collects",
                stored_version[0],
                _DECODE_RATES_SCHEMA_VERSION,
            )
            conn.execute("DROP TABLE IF EXISTS decode_rates")
        conn.execute(_DECODE_RATES_DDL)
        conn.execute(
            "INSERT OR REPLACE INTO catalog_meta VALUES "
            "('decode_rates_schema_version', ?)",
            [str(_DECODE_RATES_SCHEMA_VERSION)],
        )
        logger.debug("Created sources, rois and decode_rates tables and indexes")

    def _create_source_catalog(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create ``source_catalog``: one row per source, whatever kind it is.

        Public row columns first (see ``_SOURCE_COLUMNS``), then where the source
        sits (a root and a path beneath it), then the claim, its claim-time signature
        and the adapter payload. Only a source under a persisted root has a claim
        and could be restored; a mirror, a drop or an API registration sits under a
        root that is not persisted and has none. The private columns stay out of the
        ``sources`` view, since the claim can carry credential profile names and
        paths.

        Dropped whole when ``SOURCE_CATALOG_FORMAT`` differs from the one that
        wrote it, or is missing: the result is today's behaviour, a rebuild.
        Rows are cleared at open unless ``restore_sources`` (showing last run's
        rows with no adapter behind them would be a catalog that lies, so a
        restore registers or marks pending every row it keeps), and the rows under
        roots that are not persisted are cleared either way: nothing could rebuild
        them.
        """
        conn.execute(
            "CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        stored = conn.execute(
            "SELECT value FROM catalog_meta WHERE key = 'source_catalog_format'"
        ).fetchone()
        if stored is None or stored[0] != str(SOURCE_CATALOG_FORMAT):
            conn.execute("DROP TABLE IF EXISTS source_catalog")
            conn.execute("DROP TABLE IF EXISTS catalog_roots")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_catalog (
                source_id TEXT PRIMARY KEY,
                source_type TEXT,
                indexed_at TIMESTAMP,
                metadata_json TEXT,
                -- Does a real, hydrated adapter back this row? Monotonic --
                -- never flips back to FALSE once TRUE -- which is what makes it
                -- storable: a stale copy can only lag harmlessly. TRUE default:
                -- every adapter but the unresolved-cloud proxy is resolved by
                -- construction.
                --
                -- Residency deliberately has no column beside it. It swings both
                -- ways with no event to refresh a row from, so the `is_resident`
                -- read-mask field answers it live (biopb/biopb#1035).
                is_resolved BOOLEAN NOT NULL DEFAULT TRUE,
                -- Why is_resolved is FALSE: 'needs_recall' (a cloud placeholder;
                -- opening it is a consented download), 'pending' (registration
                -- has not run yet; resolving it runs it), 'failed' (it raised; the
                -- error is in unresolved_error). NULL when resolved.
                unresolved_reason VARCHAR,
                -- Why a 'failed' registration raised, as plain text. NULL for
                -- every other row; metadata_json stays the source's own
                -- metadata, never an error.
                unresolved_error VARCHAR,
                -- Full per-tensor structural info (biopb/biopb#224): one struct
                -- per tensor, so multi-field / HCS sources are queryable per
                -- tensor. This is the sole home of shape/dtype -- there is no
                -- scalar projection column -- so a source-wide answer means
                -- `tensors[1].dtype` (DuckDB is 1-indexed), empty on an
                -- unresolved source. Only cheap/structural fields
                -- (already in the lean ListFlights descriptor) are stored
                -- here -- the expensive/lazy fields (metadata_json, pyramid,
                -- physical_scale) are deliberately left out, filled only by
                -- GetFlightInfo. A single nested column (not a
                -- child table) keeps the whole row a single-statement upsert, so
                -- shrinking a source's tensor set can't leave ghost rows and a
                -- read never straddles a torn sources-tensors join. Unresolved
                -- cloud sources carry an empty list. Query per tensor with
                -- UNNEST(tensors) or list_filter(tensors, t -> ...).
                -- The transfer chunk_shape is deliberately NOT here: it is the
                -- read plan of the adapter bound to a specific tensor, not a
                -- catalog fact, and a source-level listing that names one is
                -- guessing for a scene it never selected (biopb/biopb#812).
                -- GetFlightInfo answers it, per resolved tensor.
                tensors STRUCT(
                    array_id VARCHAR,
                    dim_labels VARCHAR[],
                    shape BIGINT[],
                    dtype VARCHAR
                )[],
                -- The root it sits under (`catalog_roots`) and its path beneath
                -- it; the view makes `source_url` of the two.
                root_id TEXT NOT NULL,
                rel TEXT NOT NULL,
                -- The claim, as `SourceClaim` holds it. NULL under a root that is
                -- not persisted.
                primary_path TEXT,
                member_paths VARCHAR[],
                extra_config TEXT,
                -- {member path: [st_ino, size, mtime_ns, ctime_ns]} when the
                -- claim was made. No st_dev: it renumbers across boots.
                signature TEXT,
                -- What the adapter needs to be built without a parse. NULL when
                -- it has none: a restart rebuilds it from the claim.
                payload TEXT,
                last_seen TIMESTAMP,
                -- The run that wrote it (``run_epoch``).
                epoch BIGINT NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS catalog_roots (
                root_id TEXT PRIMARY KEY,
                root_url TEXT NOT NULL,
                -- Whether its sources are kept across a restart. Only a root the
                -- config names is; a drop, an upstream and the built-in root are
                -- gone at open.
                persisted BOOLEAN NOT NULL DEFAULT TRUE,
                -- The last run whose walk of this root finished, and when.
                epoch BIGINT NOT NULL DEFAULT 0,
                last_scanned TIMESTAMP
            )
        """)
        if not self.restore_sources:
            conn.execute("DELETE FROM source_catalog")
            conn.execute("DELETE FROM catalog_roots")
        else:
            conn.execute(
                "DELETE FROM source_catalog WHERE root_id IN "
                "(SELECT root_id FROM catalog_roots WHERE NOT persisted)"
            )
            conn.execute("DELETE FROM catalog_roots WHERE NOT persisted")
        conn.execute(
            "INSERT INTO catalog_roots (root_id, root_url, persisted) "
            "VALUES (?, '', FALSE)",
            [INTERNAL_ROOT_ID],
        )
        row = conn.execute(
            "SELECT value FROM catalog_meta WHERE key = 'run_epoch'"
        ).fetchone()
        self.run_epoch = (int(row[0]) if row else 0) + 1
        conn.execute(
            "INSERT OR REPLACE INTO catalog_meta VALUES ('run_epoch', ?)",
            [str(self.run_epoch)],
        )
        conn.execute(
            "INSERT OR REPLACE INTO catalog_meta VALUES ('source_catalog_format', ?)",
            [str(SOURCE_CATALOG_FORMAT)],
        )

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

    def _validate_query(self, sql: str) -> None:
        """Refuse anything but one SELECT over the public tables.

        Walks DuckDB's own parse of the statement (``json_serialize_sql``), so
        a quoted or schema-qualified name, a comma join, a CTE, a subquery or a
        DESCRIBE all resolve to the tables they actually read; a regex over
        the text cannot see those. The serializer refuses every non-SELECT and
        every multi-statement string, which is what keeps this surface
        read-only.

        Raises:
            ValueError: not a single SELECT, or a table / table function
                outside the allowlists.
        """
        try:
            ast = json.loads(
                self._get_cursor()
                .execute("SELECT json_serialize_sql(?)", [sql])
                .fetchone()[0]
            )
        except duckdb.Error as e:
            raise ValueError(f"SQL query could not be parsed: {e}")
        if ast.get("error"):
            raise ValueError(
                f"SQL query contains a forbidden keyword or statement "
                f"({ast.get('error_message')}). Only one SELECT query is allowed."
            )

        tables: Set[Tuple[str, str]] = set()
        functions: Set[str] = set()
        ctes: Set[str] = set()
        refused: List[str] = []

        def walk(node: object) -> None:
            if isinstance(node, dict):
                kind = node.get("type")
                if kind == "BASE_TABLE":
                    tables.add(
                        (
                            str(node.get("table_name", "")).lower(),
                            str(node.get("schema_name", "")).lower(),
                        )
                    )
                elif kind == "TABLE_FUNCTION":
                    functions.add(
                        str(node.get("function", {}).get("function_name", "")).lower()
                    )
                elif kind == "SHOW_REF":
                    # DESCRIBE / SUMMARIZE / SHOW: schema and statistics of
                    # whatever it names, which may be a private table.
                    refused.append("SHOW/DESCRIBE/SUMMARIZE")
                for key, value in node.items():
                    if key == "cte_map":
                        for entry in value.get("map", []):
                            ctes.add(str(entry.get("key", "")).lower())
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(ast)
        if refused:
            raise ValueError(
                f"SQL query uses {refused[0]}, which is not available here. "
                f"Accessible here: {', '.join(sorted(self.allowed_tables))}."
            )
        for table, schema in sorted(tables):
            if table in ctes and not schema:
                continue
            if schema not in ("", "main") or table not in self.allowed_tables:
                raise ValueError(
                    f"SQL query references disallowed table: {table}. "
                    f"Accessible here: {', '.join(sorted(self.allowed_tables))}."
                )
        for fn in sorted(functions):
            if fn not in self.ALLOWED_TABLE_FUNCTIONS:
                raise ValueError(
                    f"SQL query uses disallowed table function: {fn}. "
                    f"Allowed: {', '.join(sorted(self.ALLOWED_TABLE_FUNCTIONS))}."
                )

    def table_schema(self, table: str) -> pa.Schema:
        """The Arrow schema of one public table (what ListFlights advertises)."""
        if table not in self.allowed_tables:
            raise ValueError(
                f"unknown catalog table: {table}. "
                f"Accessible here: {', '.join(sorted(self.allowed_tables))}."
            )
        return (
            self._get_cursor().execute(f"SELECT * FROM {table} LIMIT 0").arrow().schema
        )

    def query(self, sql: str) -> pa.Table:
        """Execute a safe SQL query; the result is what DoGet streams.

        Truncation is signaled on the table's schema metadata, which Flight
        carries with the stream (``truncated`` / ``total_rows`` /
        ``returned_rows``; ``total_sources`` is the catalog size, informational
        only). Uses cursor() for thread-safe concurrent reads without locking.

        Raises:
            ValueError: If query is invalid or violates security rules
        """
        self._validate_query(sql)

        cursor = self._get_cursor()
        start_time = time.time()
        try:
            arrow_table = cursor.execute(sql).to_arrow_table()
            # Catalog size: context for a caller sizing the browse surface. NOT
            # a truncation denominator -- the query may count something else
            # entirely (a filtered subset, or another table), which is exactly
            # how "3 of 100 matched" got reported as "97 rows were dropped".
            total_sources = cursor.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Query executed in {elapsed_ms:.1f}ms: {sql[:100]}...")
            if elapsed_ms > self._query_timeout_ms:
                logger.warning(
                    f"Query exceeded timeout threshold: {elapsed_ms:.1f}ms > {self._query_timeout_ms}ms"
                )
        except duckdb.Error as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"SQL query failed: {e}")

        # The pre-slice row count is the query's own full size, so truncation
        # is known EXACTLY here and never inferred by comparing counts.
        total_rows = arrow_table.num_rows
        truncated = total_rows > self._max_query_results
        if truncated:
            arrow_table = arrow_table.slice(0, self._max_query_results)
            logger.warning(
                f"Query result truncated: {self._max_query_results} of {total_rows} rows"
            )
        returned_rows = arrow_table.num_rows

        return arrow_table.replace_schema_metadata(
            {
                b"truncated": str(truncated).encode(),
                b"total_rows": str(total_rows).encode(),
                b"returned_rows": str(returned_rows).encode(),
                b"total_sources": str(total_sources).encode(),
                b"query_elapsed_ms": str(int(elapsed_ms)).encode(),
            }
        )

    def sync_source_added(
        self,
        source_id: str,
        adapter: SourceAdapter,
        record: Optional[CatalogRecord] = None,
    ) -> None:
        """Sync a source to the metadata database (INSERT OR REPLACE upsert).

        Called by ``SourceManager`` when a source is registered and, for a
        previously-unresolved cloud source, again when it resolves (the upsert
        overwrites the placeholder row with the concrete one).

        Raises on failure (adapter read, JSON encode, or DB write) rather than
        swallowing, so the caller can react -- the registration path rolls back
        the matching ``register_source`` so the catalog and the registry never
        silently disagree. Logging is the caller's responsibility.

        The row and the ROIs the file carries are one transaction, so a raise
        leaves the previous row and ROIs both as they were.

        Args:
            source_id: Unique source identifier
            adapter: Backend adapter for the source
            record: The claim and its claim-time signature, from a caller that
                registers a source with a claim. The row keeps them, with the
                adapter's ``catalog_payload`` when it is resolved and has one;
                without one the row's claim, if it has one, is left as it was.
        """
        conn = self._get_connection()

        # Read the row's fields off the adapter. This is the ONLY place a
        # source's catalog row is built, so `SourceRegistry.catalog_tensors` is where the
        # "no chunk_shape on a catalog entry" invariant is enforced
        # (biopb/biopb#812).
        source_url = adapter.catalog_url
        source_type = adapter.source_type
        is_resolved = adapter.is_resolved()
        catalog = self._catalog_tensors(source_id, adapter)

        # Full per-tensor structural info (biopb/biopb#224): one struct per
        # tensor, not just tensors[0]. Expensive/lazy fields (metadata_json,
        # pyramid, physical_scale) are omitted -- they belong to the
        # tensor-bound adapter GetFlightInfo binds. Unresolved cloud sources
        # have no tensors -> empty list.
        tensors = self._tensor_rows(catalog)

        # Every source with a claim is persisted; the payload only lets a
        # restart skip the parse, so an adapter without one (or a cloud row, or
        # one that is not resolved) stores NULL and is rebuilt from its claim.
        # Before the record, which is the adapter's cue to drop what it parked
        # for it: the payload reads the same intermediates.
        payload = None
        if (
            record is not None
            and record.claim is not None
            and is_resolved
            and not record.cloud
        ):
            payload = adapter.catalog_payload()

        # The file's metadata and ROIs (#951), built together by the adapter:
        # the FORMAT decides whether its file carries annotations, because the
        # server does not police what get_metadata() returns -- a `rois` key in
        # an EMD's original_metadata or an OME-Zarr's .zattrs means whatever that
        # format meant by it. A server that does not serve the annotation actions
        # does not parse a file's ROIs either: the rows would be unreadable
        # through every surface, so the work and the storage buy nothing.
        registration = adapter.registration_record(
            [(t.array_id, list(t.dim_labels)) for t in catalog],
            import_rois=self._annotations_enabled,
            max_rois_per_tensor=self._max_rois_per_tensor,
        )
        if registration.report:
            logger.info("ome rois for %s: %s", source_id, registration.report.summary())
        indexed_at = datetime.now()
        # Prepared before the write lock: nothing in it needs the connection.
        rois = self._prepare_imported(
            source_id, source_url, registration.rois, indexed_at
        )

        metadata = registration.metadata
        metadata_json = json.dumps(metadata, cls=NumpyEncoder) if metadata else None

        self._upsert_source_row(
            conn,
            [
                source_id,
                source_url,
                source_type,
                indexed_at,
                metadata_json,
                is_resolved,
                None,  # a registered adapter has no reason; ``sync_pending_source`` sets one
                tensors,
                None,
            ],
            record,
            payload,
            # The same transaction as the row, so the two always come from one
            # registration and a raise leaves the previous pair intact. A store
            # that cannot take an annotation cannot be trusted with the row
            # either; what the importer merely refuses is skipped inside.
            also=lambda c: self._replace_imported(c, source_id, rois),
        )

        logger.debug(f"Synced source to metadata database: {source_id}")

    def bind_registry(self, registry: Any) -> None:
        """List tensors through *registry*, which holds the ones attached to a source."""
        self._registry = registry

    def _catalog_tensors(self, source_id: str, adapter: Any) -> List[Any]:
        """The tensors a row lists: the registry's view when bound, else the
        adapter's own (a catalog on its own has no attachments)."""
        if self._registry is not None:
            return self._registry.catalog_tensors(source_id, adapter)
        return Attachments(source_id).catalog_tensors(adapter)

    @staticmethod
    def _tensor_rows(catalog: Sequence[Any]) -> List[Dict[str, Any]]:
        """The ``tensors`` column for *catalog* (``SourceRegistry.catalog_tensors``)."""
        return [
            {
                "array_id": t.array_id,
                "dim_labels": list(t.dim_labels),
                "shape": [int(s) for s in t.shape],
                "dtype": t.dtype,
            }
            for t in catalog
        ]

    def relist_tensors(self, source_id: str, adapter: SourceAdapter) -> bool:
        """Make a row list the tensors *adapter* serves now, and only that.

        For a source rebuilt from its row, whose row was written before the
        uploaded fields and label sets on disk were attached (or before some of
        them went): the row says what was true at its last write. Writes the
        ``tensors`` column and nothing else when it differs -- the metadata the row
        holds is not re-read -- and reports whether it did. The order is not
        compared: a row lists fields in the order they were uploaded and the attach
        scan finds them by name, and neither is a change.
        """
        tensors = self._tensor_rows(self._catalog_tensors(source_id, adapter))
        conn = self._get_connection()
        row = (
            self._get_cursor()
            .execute(
                "SELECT tensors FROM source_catalog WHERE source_id = ?", [source_id]
            )
            .fetchone()
        )
        if row is None or sorted(row[0], key=_array_id) == sorted(
            tensors, key=_array_id
        ):
            return False
        with self._write_lock:
            conn.execute(
                "UPDATE source_catalog SET tensors = ? WHERE source_id = ?",
                [tensors, source_id],
            )
        return True

    def _placement(
        self,
        record: Optional[CatalogRecord],
        url: str,
        payload: Optional[Dict[str, Any]],
        seen: datetime,
    ) -> List[Any]:
        """The ``source_catalog`` columns after the public ones, in table order.

        Without a *record* the source sits under the built-in root and *url* is its
        ``rel``. Without a claim in it the claim columns are NULL.
        """
        if record is None:
            location = [INTERNAL_ROOT_ID, url]
            claim_values = [None] * 5
        else:
            location = [record.root_id, record.rel]
            claim = record.claim
            claim_values = (
                [None] * 5
                if claim is None
                else [
                    claim.primary_path,
                    sorted(claim.member_paths),
                    json.dumps(claim.extra_config, sort_keys=True),
                    json.dumps({k: list(v) for k, v in record.signature.items()}),
                    None if payload is None else json.dumps(payload, sort_keys=True),
                ]
            )
        return location + claim_values + [seen, self.run_epoch]

    @staticmethod
    def _pending_row(
        claim: SourceClaim,
        catalog_url: Optional[str],
        recall: bool,
        error: Optional[str],
        now: datetime,
    ) -> List[Any]:
        """The row of a claimed source that is not registered yet, built from the
        claim alone, in ``_ROW_COLUMN_NAMES`` order."""
        return [
            claim.source_id,
            catalog_url or to_catalog_url(str(claim.primary_path)),
            claim.source_type or "unknown",
            now,
            None,
            False,
            "failed" if error else ("needs_recall" if recall else "pending"),
            [],
            error or None,
        ]

    def _upsert_source_row(
        self,
        conn: duckdb.DuckDBPyConnection,
        row: List[Any],
        record: Optional[CatalogRecord] = None,
        payload: Optional[Dict[str, Any]] = None,
        also: Optional[Callable[[duckdb.DuckDBPyConnection], None]] = None,
    ) -> None:
        """Insert or update a source's row (*row*, in ``_ROW_COLUMN_NAMES`` order),
        serializing writes with the lock.

        One transaction with *also*, which runs after the row is written and
        inside it: a raise from either leaves the previous row and whatever
        *also* wrote exactly as they were.

        A ``source_id`` has one row, so a write cannot list a source twice. With a
        *record* (where the source sits, its claim and signature, and the adapter
        *payload*) the row takes all of it. Without one only the public columns are
        written and the location and claim the row already has are left alone, which
        is what a re-listing wants: a field uploaded to a discovered source changes
        its tensors, not where it sits. A source new to the catalog with no record
        sits under the built-in root, its url the ``rel``.

        A row that exists is updated, not replaced: a registration fills in the
        pending row its claim made, and an ``UPDATE`` costs about two thirds of
        the wall time and a quarter of the CPU of an ``INSERT OR REPLACE`` on the
        indexed table.
        """
        source_id, url, indexed_at = row[0], row[1], row[3]
        # Serialized before the lock: a payload can be large, and every other
        # writer waits while it is held.
        values = row[:1] + row[2:] + self._placement(record, url, payload, indexed_at)
        if record is None:
            update_set, upsert_set = _LISTING_SET, _LISTING_UPSERT_SET
            set_values = row[2:] + [url]
        else:
            update_set, upsert_set = _UPDATE_SET, _UPSERT_SET
            set_values = values[1:]
        with self._write_lock:
            conn.execute("BEGIN TRANSACTION")
            try:
                updated = conn.execute(
                    f"UPDATE source_catalog SET {update_set} WHERE source_id = ?",
                    set_values + [source_id],
                ).fetchone()
                if not updated or not updated[0]:
                    # A row that is there by now (the UPDATE did not see it) is
                    # overwritten, which is what was asked: a registration must
                    # not fail on the pending row its claim made.
                    conn.execute(
                        f"INSERT INTO source_catalog ({_ALL_COLUMNS}) "
                        f"VALUES ({', '.join('?' * len(values))}) "
                        f"ON CONFLICT (source_id) DO UPDATE SET {upsert_set}",
                        values,
                    )
                if also is not None:
                    also(conn)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def sync_pending_source(
        self,
        claim: SourceClaim,
        catalog_url: Optional[str] = None,
        error: Optional[str] = None,
        recall: bool = False,
        record: Optional[CatalogRecord] = None,
    ) -> None:
        """Write the row of a claimed source that is not registered yet.

        Built from the claim alone, so no file is opened: ``is_resolved`` false,
        no tensors, ``unresolved_reason`` ``pending``, or ``needs_recall`` for a
        cloud source (*recall*), whose registration downloads it and waits for a
        client. With *error* (its registration raised) the reason is ``failed``
        and ``unresolved_error`` carries the text, so a client does not wait on
        it. The registered row replaces this one by the same upsert.

        With a *record* the row carries where the source sits and its claim, and
        overwrites what the source had.
        """
        conn = self._get_connection()
        row = self._pending_row(claim, catalog_url, recall, error, datetime.now())
        self._upsert_source_row(conn, row, record)

    # Rows per INSERT: a multi-row statement costs ~0.06 ms a row against ~4 ms for
    # one statement a row, and stays well under DuckDB's parameter limits.
    _PENDING_CHUNK = 500

    def _pending_inserts(
        self, rows: Sequence[PendingRow], now: datetime
    ) -> List[Tuple[str, List[Any]]]:
        """The multi-row ``INSERT ... DO NOTHING`` statements for *rows*, in chunks."""
        statements = []
        for i in range(0, len(rows), self._PENDING_CHUNK):
            chunk = rows[i : i + self._PENDING_CHUNK]
            params: List[Any] = []
            for row in chunk:
                values = self._pending_row(
                    row.claim, row.catalog_url, row.recall, None, now
                )
                params += values[:1] + values[2:]
                params += self._placement(row.record, values[1], None, now)
            width = len(params) // len(chunk)
            statements.append(
                (
                    f"INSERT INTO source_catalog ({_ALL_COLUMNS}) VALUES "
                    + ", ".join([f"({', '.join('?' * width)})"] * len(chunk))
                    + " ON CONFLICT DO NOTHING",
                    params,
                )
            )
        return statements

    def sync_pending_sources(self, rows: Sequence[PendingRow]) -> None:
        """Write the rows of many claimed sources at once (see ``sync_pending_source``).

        Only a source with no row yet gets one. A batch is written some time after
        its claims were made, and in that time a registration may already have
        written the real row, which a pending one must not replace.

        One transaction, serialized with every other writer by the lock; the
        statements are built before it is taken.
        """
        if not rows:
            return
        conn = self._get_connection()
        statements = self._pending_inserts(rows, datetime.now())
        with self._write_lock:
            conn.execute("BEGIN TRANSACTION")
            try:
                for sql, params in statements:
                    conn.execute(sql, params)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def restorable_rows(self) -> List[Dict[str, Any]]:
        """The rows with a claim, which a restore rebuilds claims from, without the
        payload and the metadata (large, and read per source when it is hydrated).

        Empty unless ``restore_sources``: the table was cleared at open.
        """
        if not self.restore_sources:
            return []
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT c.source_id, c.source_type, c.is_resolved, "
            "c.unresolved_reason, c.unresolved_error, c.root_id, c.rel, "
            "c.primary_path, c.member_paths, c.extra_config, c.signature, "
            "c.last_seen, r.last_scanned "
            "FROM source_catalog c JOIN catalog_roots r ON c.root_id = r.root_id "
            "WHERE c.primary_path IS NOT NULL"
        )
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def read_hydration(
        self, source_id: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """``(payload, metadata)`` of a persisted source, or None when it has no
        payload (or it does not decode): what an adapter is rebuilt from without a
        parse. One keyed read, made when the source is hydrated and not before, on
        a cursor because hydrations run on several threads at once."""
        row = (
            self._get_cursor()
            .execute(
                "SELECT payload, metadata_json FROM source_catalog WHERE source_id = ?",
                [source_id],
            )
            .fetchone()
        )
        if row is None or row[0] is None:
            return None
        try:
            return json.loads(row[0]), json.loads(row[1]) if row[1] else {}
        except (TypeError, ValueError):
            logger.warning("unreadable payload for source %s", source_id)
            return None

    def drop_catalog_rows(self, source_ids: Sequence[str]) -> None:
        """Delete persisted rows a restore did not keep, with the reserved ROI rows
        each one's registration derived (see :meth:`sync_source_removed`)."""
        if not source_ids:
            return
        conn = self._get_connection()
        ids = list(source_ids)
        with self._write_lock:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    "DELETE FROM source_catalog WHERE source_id IN (SELECT unnest(?::VARCHAR[]))",
                    [ids],
                )
                conn.execute(
                    "DELETE FROM rois WHERE source_id IN (SELECT unnest(?::VARCHAR[])) "
                    "AND starts_with(set_name, ?)",
                    [ids, RESERVED_SET_PREFIX],
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def rewrite_row_roots(self, rows: Sequence[Tuple[str, str, str]]) -> None:
        """Re-attribute persisted rows to the roots a restore found them under:
        ``(source_id, root_id, rel)`` each."""
        if not rows:
            return
        conn = self._get_connection()
        with self._write_lock:
            conn.execute("BEGIN TRANSACTION")
            try:
                for source_id, root_id, rel in rows:
                    conn.execute(
                        "UPDATE source_catalog SET root_id = ?, rel = ? "
                        "WHERE source_id = ?",
                        [root_id, rel, source_id],
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def confirm_root(self, root_id: str) -> None:
        """Record that this run's walk of a root finished: every row under it that a
        restore brought back is now verified against the disk."""
        conn = self._get_connection()
        with self._write_lock:
            conn.execute(
                "UPDATE catalog_roots SET epoch = ?, last_scanned = ? "
                "WHERE root_id = ?",
                [self.run_epoch, datetime.now(), root_id],
            )

    def sweep_root(self, root_id: str, is_claimed: Callable[[str], bool]) -> int:
        """Delete a root's rows that no claim holds, after a walk of it finished.

        The walk removes a claim that is gone, with its row; this is for the row
        that outlived its claim (a crash between the two writes). Whether a claim
        holds a row is asked of the caller one id at a time, an exact answer, so a
        source being registered as this runs is never taken for an orphan. Returns
        the number deleted.
        """
        conn = self._get_connection()
        held = [
            r[0]
            for r in conn.execute(
                "SELECT source_id FROM source_catalog WHERE root_id = ?", [root_id]
            ).fetchall()
        ]
        orphans = [source_id for source_id in held if not is_claimed(source_id)]
        self.drop_catalog_rows(orphans)
        return len(orphans)

    def sync_roots(self, roots: Sequence[Tuple[str, str]]) -> None:
        """Replace the persisted roots in ``catalog_roots`` with *roots*,
        ``(root_id, root_url)`` each.

        Config is the truth, so a persisted root that is gone is deleted and the
        rest keep their ``epoch``: it says when a restored row's root was last
        walked. Written before any source row is, since the view shows a row only
        against its root. The roots that are not persisted are :meth:`ensure_root`'s.
        """
        conn = self._get_connection()
        ids = [root_id for root_id, _ in roots]
        with self._write_lock:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    "DELETE FROM catalog_roots WHERE persisted AND root_id NOT IN "
                    "(SELECT unnest(?::VARCHAR[]))",
                    [ids],
                )
                for root_id, root_url in roots:
                    conn.execute(
                        "INSERT INTO catalog_roots (root_id, root_url, persisted) "
                        "VALUES (?, ?, TRUE) "
                        "ON CONFLICT (root_id) DO UPDATE SET root_url = excluded.root_url",
                        [root_id, root_url],
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def ensure_root(self, root_id: str, root_url: str) -> None:
        """Record a root that is not persisted (a drop, an upstream) before a source
        under it is written, since the view shows a row only against its root. It
        goes at the next open with the rows under it."""
        conn = self._get_connection()
        with self._write_lock:
            conn.execute(
                "INSERT INTO catalog_roots (root_id, root_url, persisted) "
                "VALUES (?, ?, FALSE) "
                "ON CONFLICT (root_id) DO UPDATE SET root_url = excluded.root_url",
                [root_id, root_url],
            )

    def _prepare_imported(
        self,
        source_id: str,
        source_url: str,
        imported: Mapping[str, List[RoiAnnotation]],
        now: datetime,
    ) -> List[List[Any]]:
        """The ``rois`` rows a file's imported annotations become.

        No rev/created_at carry-forward, unlike :meth:`put_rois`. These rows are
        not edited, they are re-derived -- there is no history to preserve, and
        pretending otherwise would put a monotonic rev on a value that only ever
        restates the file. Needs no connection, so it runs before the write lock.
        """
        rows: List[List[Any]] = []
        for array_id, found in imported.items():
            for annotation in found:
                # allow_reserved: this is the one writer the @ome namespace has.
                # Still through _prepare_roi, so bbox and the canonical geometry
                # JSON are derived exactly as they are for a hand-drawn row --
                # the SQL surface cannot tell the two apart, which is the point.
                try:
                    prep = _prepare_roi(array_id, annotation, allow_reserved=True)
                except ValueError:
                    # _prepare_roi stays the single authority on what is
                    # storable -- an over-long id, say -- so the importer skips
                    # what it refuses instead of carrying a second copy of the
                    # rules.
                    logger.debug(
                        "ome rois: %s rejected by the store", annotation.roi_id
                    )
                    continue
                rows.append(
                    [prep.roi_id, array_id, source_id]
                    + prep.column_values(self._ROI_CLIENT_COLUMNS)
                    # last_seen_at is `now` unconditionally: the source is being
                    # registered, which IS the sighting these rows record.
                    + [1, now, now, source_url, now]
                )
        return rows

    def _replace_imported(
        self,
        conn: duckdb.DuckDBPyConnection,
        source_id: str,
        rows: List[List[Any]],
    ) -> None:
        """Swap a source's reserved rows for *rows*, inside the caller's transaction.

        Delete-then-insert scoped by ``source_id``, not per tensor: a tensor
        whose ROIs were removed upstream has to lose its rows too, and it has no
        entry in the import to drive that from.
        """
        conn.execute(
            "DELETE FROM rois WHERE source_id = ? AND starts_with(set_name, ?)",
            [source_id, RESERVED_SET_PREFIX],
        )
        if rows:
            conn.executemany(
                "INSERT INTO rois "
                f"(roi_id, array_id, source_id, {', '.join(self._ROI_CLIENT_COLUMNS)}, "
                "rev, created_at, updated_at, source_url, last_seen_at) "
                f"VALUES ({', '.join('?' * (len(self._ROI_CLIENT_COLUMNS) + 8))})",
                rows,
            )

    def source_row_ipc(self, source_id: str) -> Optional[bytes]:
        """One source's catalog row as an Arrow IPC stream, or ``None``.

        The row is the only representation of a source that crosses the wire:
        the ``catalog`` flight streams these, and the ``resolve`` action returns
        the single row it just wrote rather than building a second encoding from
        the adapter. Same columns either way (``SOURCE_ROW_COLUMNS``), so a
        client has one decoder.

        Uses ``cursor()`` for a thread-safe read; raises on a DuckDB error.
        """
        cursor = self._get_cursor()
        table = cursor.execute(
            f"SELECT {SOURCE_ROW_COLUMNS} FROM sources WHERE source_id = ?",
            [source_id],
        ).to_arrow_table()
        if table.num_rows == 0:
            return None
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()

    def get_metadata_json(self, source_id: str) -> Optional[dict]:
        """Return a source's stored metadata as a dict, or ``None`` when empty.

        The catalog stores ``json.dumps(adapter.get_metadata())`` -- the **raw**
        dict, no envelope -- so the serve path can read metadata back with a
        cheap local ``SELECT`` instead of recomputing it on the adapter
        (biopb/biopb#253), and for a remote proxy without an upstream RPC (read
        the local mirror row directly, never ``adapter.get_metadata()``). The
        stored JSON is parsed here so callers get a ready dict.

        Returns ``None`` when the source has no usable stored metadata -- which is
        a legitimate answer, not a failure, so the serve path leaves
        ``metadata_json`` empty:
        - the source is absent, or its metadata is SQL NULL (empty is stored as
          NULL),
        - the stored value is not valid JSON / not a JSON object.

        **Raises** on a genuine DuckDB read error. The catalog is the mandatory,
        authoritative source of serve-path metadata (there is no adapter
        fallback), so a read failure must surface as a failed request rather than
        be masked as "no metadata". Uses ``cursor()`` for a thread-safe read.
        """
        try:
            cursor = self._get_cursor()
            row = cursor.execute(
                "SELECT metadata_json FROM sources WHERE source_id = ?", [source_id]
            ).fetchone()
        except Exception as exc:
            logger.warning(
                "metadata_json read failed for source %s: %s", source_id, exc
            )
            raise

        if row is None or not row[0]:
            return None

        try:
            parsed = json.loads(row[0])
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "stored metadata_json for source %s is not valid JSON", source_id
            )
            return None
        return parsed if isinstance(parsed, dict) else None

    def sync_source_removed(self, source_id: str) -> None:
        """Remove a source from the metadata database.

        Called by ``SourceManager`` when a source is unregistered or rolled back.

        Raises on DB failure rather than swallowing, so the caller can react;
        logging is the caller's responsibility.

        Args:
            source_id: Unique source identifier
        """
        conn = self._get_connection()
        with self._write_lock:
            conn.execute("DELETE FROM source_catalog WHERE source_id = ?", [source_id])
            # Reserved rows go with the source row: they are its scan output,
            # re-derived on the next registration. Hand-drawn annotations are
            # deliberately NOT touched here -- outliving their source is the
            # whole of the orphan design (biopb/biopb#951).
            conn.execute(
                "DELETE FROM rois WHERE source_id = ? AND starts_with(set_name, ?)",
                [source_id, RESERVED_SET_PREFIX],
            )
        logger.debug(f"Removed source from metadata database: {source_id}")

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

    def load_decode_rates(self) -> Dict[str, Tuple[float, int]]:
        """Every persisted decode rate, as ``array_id -> (mbps, samples)``.

        The rows of arrays this server no longer serves come back too. Dropping
        them here would cost the measurement of a source that is merely offline
        -- an unmounted share, a folder not yet rescanned -- and a stale row is
        harmless: it is only ever consulted by array_id, which nothing asks for
        unless the array is being read.
        """
        cursor = self._get_cursor()
        rows = cursor.execute(
            "SELECT array_id, mbps, samples FROM decode_rates"
        ).fetchall()
        return {row[0]: (float(row[1]), int(row[2])) for row in rows}

    def save_decode_rates(self, rows: Mapping[str, Tuple[float, int]]) -> None:
        """Upsert measured rates, stamping every row written as observed now.

        Raises like any other write here; the read path's caller
        (:meth:`DecodeRates.flush`) is where a failure is decided to be
        survivable, because that is where the cost of raising is a failed read.
        """
        if not rows:
            return
        now = datetime.now()
        payload = [
            (array_id, float(mbps), int(samples), now)
            for array_id, (mbps, samples) in rows.items()
        ]
        # Connection first, lock second: _get_connection takes _write_lock to
        # build one, and it is not reentrant.
        conn = self._get_connection()
        with self._write_lock:
            conn.executemany(
                "INSERT OR REPLACE INTO decode_rates "
                "(array_id, mbps, samples, updated_at) VALUES (?, ?, ?, ?)",
                payload,
            )

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            with self._write_lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
                    logger.info("MetadataDatabase closed")
