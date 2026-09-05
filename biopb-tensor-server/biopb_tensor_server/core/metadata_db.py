"""DuckDB metadata database for efficient source filtering.

Provides indexed SQL queries against source metadata for large catalogs
(>100k sources). Replaces O(n) in-memory scans with indexed DuckDB queries.

Database Schema:
- sources table with indexed fields (source_id, source_url)
- JSON column for full metadata access via DuckDB JSON operators
- Shape summary column for quick size estimates
- rois table: user-drawn ROI annotations, one row per ROI, anchored on the
  unversioned array_id (docs/roi-annotations.md)

Security Model:
- DuckDB connection runs with enable_external_access=False, so all file/network
  access (read_csv, read_text, glob, COPY, ATTACH, extension loading) is blocked
  at the engine level. This is the primary defense against file exfiltration.
- Only the 'sources' and 'rois' tables are accessible (keyword/table denylist;
  defense in depth). 'rois' is readable here for analysis; every ROI write goes
  through put_rois/delete_rois, never through this surface.
- Forbidden keywords: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, EXECUTE
- No subqueries referencing external tables
- Query timeout enforced

Usage:
    db = MetadataDatabase()
    db.sync_source_added(source_id, adapter)
    flight_info = db.handle_query("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.image.annotation_pb2 import RoiAnnotation, RoiConflict
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor
from google.protobuf import json_format

if TYPE_CHECKING:
    from biopb_tensor_server.core.adapter_base import SourceAdapter

logger = logging.getLogger(__name__)


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
# ROI annotation helpers (docs/roi-annotations.md)
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

_ACCEPTED_SHAPES: Set[str] = {
    "point",
    "rectangle",
    "ellipse",
    "polygon",
    "polyline",
}


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
    plane: Dict[str, int]
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


def _prepare_roi(array_id: str, roi: RoiAnnotation) -> _PreparedRoi:
    """Validate one annotation and derive its stored columns.

    Raises:
        ValueError: unusable geometry, an array_id that contradicts the batch's,
            or a negative plane index.
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

    shape_kind = roi.roi.WhichOneof("shape")
    if shape_kind is None:
        raise ValueError("Annotation has no geometry")
    if shape_kind not in _ACCEPTED_SHAPES:
        raise ValueError(
            f"Geometry {shape_kind!r} is not accepted by the annotation store "
            f"(accepted: {', '.join(sorted(_ACCEPTED_SHAPES))}). Instance "
            f"segmentation belongs in a label tensor."
        )

    for dim, index in roi.plane.items():
        if index < 0:
            raise ValueError(f"Plane index for {dim!r} is negative: {index}")

    return _PreparedRoi(
        roi_id=roi_id or uuid.uuid4().hex,
        set_name=roi.set_name or "default",
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
        return [c.x - abs(r.x), c.y - abs(r.y), c.x + abs(r.x), c.y + abs(r.y)]
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
    source-browsing surface (``client.query_sources``), so there is no
    off switch -- constructing this object means the catalog is live.

    Args:
        max_query_results: Safety cap on returned rows (truncation signaled via schema metadata)
        query_timeout_ms: Query execution timeout in milliseconds
        max_rois_per_tensor: Cap on stored annotations per tensor. Deliberately
            human-scale: it is the line between an annotation store and an
            object store, and it is what lets the read path be a single
            whole-set fetch (see docs/roi-annotations.md).

    Example:
        db = MetadataDatabase()
        db.sync_source_added('plate-001', adapter)
        info = db.handle_query("SELECT source_id FROM sources WHERE dtype='uint16'")
    """

    # Forbidden SQL keywords (write operations, table manipulation)
    FORBIDDEN_KEYWORDS: Set[str] = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "EXECUTE",
        "GRANT",
        "REVOKE",
        "COPY",
        "EXPORT",
        "IMPORT",
        "LOAD",
    }

    # Match forbidden keywords only as whole words. A plain substring test
    # rejects legitimate queries where the keyword appears inside an identifier
    # or string literal (e.g. `LIKE '%/uploads/%'` contains LOAD, `%update%`
    # contains UPDATE). The real defense against file/network access is
    # enable_external_access=False on the connection; this is defense in depth.
    FORBIDDEN_KEYWORD_PATTERN = re.compile(
        r"\b(" + "|".join(sorted(FORBIDDEN_KEYWORDS)) + r")\b"
    )

    # Only these tables can be referenced in queries. ``rois`` is readable here
    # as an ANALYSIS affordance (count labels, join against sources, find
    # annotations overlapping a region); the viewer never composes SQL -- it
    # calls list_rois(), which builds parameterized SQL itself. Writes stay off
    # this surface entirely: FORBIDDEN_KEYWORDS still rejects INSERT/UPDATE/DELETE.
    ALLOWED_TABLES: Set[str] = {"sources", "rois"}

    # Pattern for detecting table references in SQL
    TABLE_REFERENCE_PATTERN = re.compile(
        r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        r"|\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        r"|\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        r"|\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        max_query_results: int = 100000,
        query_timeout_ms: int = 30000,
        max_rois_per_tensor: int = 5000,
    ):
        self._max_query_results = max_query_results
        self._query_timeout_ms = query_timeout_ms
        self._max_rois_per_tensor = max_rois_per_tensor

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._write_lock = threading.Lock()  # Lock for write operations only
        self._initialized = False

        logger.info(
            "MetadataDatabase enabled (DuckDB backend will initialize on first access)"
        )

        # Pending query results for DoGet (stored by ticket)
        self._pending_results: Dict[str, pa.Table] = {}
        self._pending_results_lock = threading.Lock()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Lazy initialization of DuckDB connection.

        Returns the shared connection for write operations.
        For reads, use _get_cursor() which returns thread-safe cursors.
        """
        if self._conn is None:
            with self._write_lock:
                if self._conn is None:
                    # Disable all external file/network access. This is the
                    # real defense against file exfiltration via read_csv /
                    # read_text / glob / COPY / ATTACH etc., which the keyword
                    # denylist in _validate_query cannot reliably cover (e.g.
                    # comma-joins like `FROM sources, read_text('/etc/passwd')`
                    # slip past the FROM-only table check). Once disabled it
                    # cannot be re-enabled within a running instance, so a
                    # `SET enable_external_access=true` in a query is rejected.
                    # The server itself needs no external access: it only does
                    # parameterized INSERT/DELETE and JSON-operator SELECTs.
                    self._conn = duckdb.connect(
                        ":memory:", config={"enable_external_access": False}
                    )
                    self._create_schema()
                    self._initialized = True
                    logger.info("MetadataDatabase initialized with in-memory DuckDB")
        return self._conn

    def _get_cursor(self) -> duckdb.DuckDBPyConnection:
        """Get a cursor for thread-safe read operations.

        DuckDB cursors (created via conn.cursor()) are thread-safe and can
        execute concurrently. This allows parallel reads without locking.
        """
        conn = self._get_connection()
        return conn.cursor()

    def _create_schema(self) -> None:
        """Create sources table and indexes."""
        conn = self._conn
        # Main table
        conn.execute("""
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY,
                source_url TEXT,
                source_type TEXT,
                dtype TEXT,
                indexed_at TIMESTAMP,
                metadata_json TEXT,
                shape_summary TEXT,
                -- NOT NULL DEFAULT FALSE: every source has a residency value
                -- (both insert sites write the descriptor's data_resident bit),
                -- and a non-null column lets `WHERE data_resident` /
                -- `WHERE NOT data_resident` partition ALL rows cleanly -- no
                -- three-valued-logic gap where a NULL row silently drops from
                -- both. FALSE is the conservative default (unknown -> treat as
                -- non-resident; still discoverable via `WHERE NOT data_resident`).
                data_resident BOOLEAN NOT NULL DEFAULT FALSE,
                -- Full per-tensor structural info (biopb/biopb#224): one struct
                -- per tensor, so multi-field / HCS sources are queryable per
                -- tensor instead of via the first-tensor projection only. Only
                -- cheap/structural fields (already in the lean ListFlights
                -- descriptor) are stored here -- the expensive/lazy fields
                -- (metadata_json, pyramid, physical_scale) are deliberately left
                -- out, filled only by GetFlightInfo. A single nested column (not a
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
                )[]
            )
        """)
        # Index on source_url for path filtering
        conn.execute("CREATE INDEX idx_source_url ON sources(source_url)")

        # User-drawn ROI annotations, one row per ROI (design:
        # docs/roi-annotations.md). A sibling table, deliberately NOT a field
        # inside a source row: sources.metadata_json is adapter-produced and
        # rewritten by the INSERT OR REPLACE in sync_source_added(), so an
        # annotation parked there would be destroyed by the next rescan.
        conn.execute(
            """
            CREATE TABLE rois (
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
                -- The catalog URL at write time. NOT a liveness probe -- there is
                -- no existence oracle spanning file / proxy / cloud / upload
                -- sources -- but array_id is a SHA-256 and cannot be inverted, so
                -- without this an orphan report can only say "annotations for
                -- zarr_a3f2b1c4", which no one can act on. Also what a
                -- re-attach-after-move prompt would key on.
                source_url TEXT,
                -- Grouping key: the "layer" ("nuclei", "hand-drawn").
                set_name TEXT NOT NULL DEFAULT 'default',
                label TEXT,
                -- point|rectangle|ellipse|polygon, denormalized from the geometry
                -- for filtering. mask/mesh are rejected on write.
                shape_kind TEXT NOT NULL,
                -- Sparse plane pin, dim_label -> index. A dimension ABSENT from
                -- the map applies at every index of that dimension, so one ROI
                -- can follow a z-stack without being duplicated per plane. Keyed
                -- by label because dim_labels are per-tensor.
                plane MAP(VARCHAR, BIGINT),
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
        )
        conn.execute("CREATE INDEX idx_rois_array ON rois(array_id)")
        conn.execute("CREATE INDEX idx_rois_source ON rois(source_id)")
        logger.debug("Created sources and rois tables and indexes")

    def _validate_query(self, sql: str) -> None:
        """Validate SQL query for security.

        Raises:
            ValueError: If query contains forbidden keywords or references disallowed tables
        """
        # Strip single-quoted string literals before scanning so keywords that
        # appear *inside* a literal (e.g. `LIKE '%update%'`) aren't mistaken for
        # SQL keywords or table names. '' is DuckDB's escaped single quote.
        literal_free = re.sub(r"'(?:''|[^'])*'", "''", sql)
        normalized = literal_free.upper()

        # Check for forbidden keywords (whole-word match, see pattern above)
        match = self.FORBIDDEN_KEYWORD_PATTERN.search(normalized)
        if match:
            raise ValueError(
                f"SQL query contains forbidden keyword: {match.group(1)}. "
                f"Only SELECT queries are allowed."
            )

        # Check for table references
        table_refs = self.TABLE_REFERENCE_PATTERN.findall(literal_free)
        referenced_tables = set()
        for match in table_refs:
            for table_name in match:
                if table_name:
                    referenced_tables.add(table_name.lower())

        # Only allow references to permitted tables
        for table in referenced_tables:
            if table not in self.ALLOWED_TABLES:
                raise ValueError(
                    f"SQL query references disallowed table: {table}. "
                    f"Only the 'sources' table is accessible."
                )

    def handle_query(self, sql: str) -> flight.FlightInfo:
        """Execute a safe SQL query and return FlightInfo.

        The actual query results are stored internally and retrieved via DoGet
        using a ticket that references this query.

        Uses cursor() for thread-safe concurrent reads without locking.

        Args:
            sql: SQL query against the sources table

        Returns:
            FlightInfo with schema and endpoint for DoGet retrieval

        Raises:
            ValueError: If query is invalid or violates security rules
        """
        self._validate_query(sql)

        # Use cursor for thread-safe read (no lock needed for SELECT)
        cursor = self._get_cursor()

        start_time = time.time()
        try:
            # Execute query via cursor - thread-safe, no lock
            result = cursor.execute(sql)
            arrow_table = result.to_arrow_table()

            # Get source count using same cursor
            total_sources = cursor.execute("SELECT COUNT(*) FROM sources").fetchone()[0]

            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Query executed in {elapsed_ms:.1f}ms: {sql[:100]}...")

            # Check timeout
            if elapsed_ms > self._query_timeout_ms:
                logger.warning(
                    f"Query exceeded timeout threshold: {elapsed_ms:.1f}ms > {self._query_timeout_ms}ms"
                )

        except duckdb.Error as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"SQL query failed: {e}")

        # Apply truncation if needed
        returned_rows = arrow_table.num_rows

        if returned_rows > self._max_query_results:
            arrow_table = arrow_table.slice(0, self._max_query_results)
            logger.warning(
                f"Query result truncated: {self._max_query_results} of {returned_rows} rows"
            )

        # Build schema metadata for truncation signaling
        metadata = {
            b"total_sources": str(total_sources).encode(),
            b"returned_sources": str(
                min(returned_rows, self._max_query_results)
            ).encode(),
            b"query_elapsed_ms": str(int(elapsed_ms)).encode(),
        }
        # Tag the TABLE, not just the FlightInfo schema. DoGet streams this very
        # table, and Flight carries a schema's custom metadata with it, so
        # tagging here reaches both kinds of caller: the ones that read the keys
        # off the FlightInfo, and the ones that read them off the result they
        # got from the stream (the sidecar's /api/sources/query).
        arrow_table = arrow_table.replace_schema_metadata(metadata)
        schema = arrow_table.schema

        # Store result for DoGet retrieval
        ticket_id = f"metadata-query-{time.time_ns()}"
        with self._pending_results_lock:
            self._pending_results[ticket_id] = arrow_table

        # Create ticket and endpoint
        ticket = flight.Ticket(ticket_id.encode())
        endpoint = flight.FlightEndpoint(ticket=ticket, locations=[])

        return flight.FlightInfo(
            schema=schema,
            descriptor=flight.FlightDescriptor.for_command(b""),
            endpoints=[endpoint],
            total_records=-1,
            total_bytes=-1,
        )

    def get_pending_result(self, ticket_id: str) -> Optional[pa.Table]:
        """Retrieve pending query result for DoGet.

        Args:
            ticket_id: Ticket identifier from FlightEndpoint

        Returns:
            Arrow Table with query results, or None if not found
        """
        with self._pending_results_lock:
            result = self._pending_results.pop(ticket_id, None)
        return result

    def sync_source_added(self, source_id: str, adapter: SourceAdapter) -> None:
        """Sync a source to the metadata database (INSERT OR REPLACE upsert).

        Called by ``SourceManager`` when a source is registered and, for a
        previously-unresolved cloud source, again when it resolves (the upsert
        overwrites the placeholder row with the concrete descriptor).

        Raises on failure (descriptor read, JSON encode, or DB write) rather
        than swallowing, so the caller can react -- the registration path rolls
        back the matching ``register_source`` so the catalog and ``ListFlights``
        never silently disagree. Logging is the caller's responsibility.

        Once the row is committed this calls
        ``adapter.release_registration_cache()``: the catalog now holds the
        metadata, so the adapter may drop whatever it kept only to produce it.

        Args:
            source_id: Unique source identifier
            adapter: Backend adapter for the source
        """
        conn = self._get_connection()

        # Get source descriptor and metadata
        source_desc = adapter.get_source_descriptor()
        metadata = adapter.get_metadata()

        # Scalar first-tensor projection, kept for back-compat: the MCP guide's
        # `WHERE dtype='uint16'` / `shape_summary` predicates keep working, and
        # since they are written in the SAME upsert as the tensors struct below
        # they can never desync from it.
        shape_summary = None
        dtype = None
        if source_desc.tensors:
            first_tensor = source_desc.tensors[0]
            shape_summary = json.dumps(list(first_tensor.shape))
            dtype = first_tensor.dtype

        # Full per-tensor structural info (biopb/biopb#224): one struct per tensor,
        # not just tensors[0]. Every field here is already populated in the lean
        # source descriptor, so this adds no adapter call and no recall.
        # Expensive/lazy fields (metadata_json, pyramid, physical_scale) are
        # omitted. Unresolved cloud sources have no tensors -> empty list.
        tensors = [
            {
                "array_id": t.array_id,
                "dim_labels": list(t.dim_labels),
                "shape": [int(s) for s in t.shape],
                "dtype": t.dtype,
            }
            for t in source_desc.tensors
        ]

        # Build row data
        indexed_at = datetime.now()
        metadata_json = json.dumps(metadata, cls=NumpyEncoder) if metadata else None

        # Insert or replace (upsert) - serialize writes with lock
        with self._write_lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO sources
                (source_id, source_url, source_type, dtype, indexed_at, metadata_json, shape_summary, data_resident, tensors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    source_id,
                    source_desc.source_url,
                    source_desc.source_type,
                    dtype,
                    indexed_at,
                    metadata_json,
                    shape_summary,
                    source_desc.data_resident,
                    tensors,
                ],
            )

        # The row is committed, so the catalog -- not the adapter -- now owns this
        # source's metadata (biopb/biopb#253). Let the adapter drop whatever it
        # parked on itself only to build the row; OME-TIFF's raw OME-XML is tens
        # of MB on a per-plane acquisition (biopb/biopb#783). Best-effort: a
        # balky release must not fail a registration that already succeeded.
        try:
            adapter.release_registration_cache()
        except Exception:  # pragma: no cover - release is an optimization
            logger.debug(
                "release_registration_cache failed for %s", source_id, exc_info=True
            )

        logger.debug(f"Synced source to metadata database: {source_id}")

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

    def list_source_descriptors(
        self, limit: Optional[int] = None
    ) -> Tuple[List[DataSourceDescriptor], int]:
        """Rebuild the lean ListFlights descriptors from the catalog.

        The DuckDB-backed equivalent of iterating adapters and calling
        ``get_source_descriptor()``. Serving ``ListFlights`` from here makes the
        catalog the single source of truth for browsing, so ``list_sources`` and
        ``query_sources`` cannot drift (biopb/biopb#265).

        Only the cheap/structural fields the lean descriptor carries are
        reconstructed: per-tensor ``array_id``/``dim_labels``/``shape``/``dtype``
        from the ``tensors`` STRUCT[] (biopb/biopb#224). ``chunk_shape`` is left
        empty here and on every ListFlights entry -- the transfer grid belongs to
        the tensor-bound adapter, and ``GetFlightInfo`` is where it is answered
        (biopb/biopb#812). ``metadata_json`` is likewise left empty (filled by
        ``GetFlightInfo``), exactly like the adapter path. ``data_resident`` is the stored snapshot -- the
        field is advisory/volatile by contract (the authoritative gate is a fresh
        ``adapter.is_resident()``), so a point-in-time value is acceptable here.

        Uses ``cursor()`` for a thread-safe read (no lock). The full count is
        carried by a ``COUNT(*) OVER ()`` window in the SAME statement as the
        rows (window functions run before ``LIMIT``), so ``total`` and the
        clipped rows come from one consistent snapshot -- a separate
        ``SELECT COUNT(*)`` could race a concurrent upload and report
        ``returned > total``.

        Args:
            limit: Max rows to return (the ListFlights safety cap). ``None`` =
                no cap.

        Returns:
            ``(descriptors, total)`` where ``total`` is the full catalog row
            count (so the caller can signal truncation when ``limit`` clips it).
        """
        cursor = self._get_cursor()

        sql = (
            "SELECT source_id, source_url, source_type, data_resident, tensors, "
            "COUNT(*) OVER () AS total_count "
            "FROM sources ORDER BY source_id"
        )
        params: list = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = cursor.execute(sql, params).fetchall()

        # COUNT(*) OVER () is identical on every row; no rows -> empty catalog.
        total = rows[0][-1] if rows else 0

        descriptors: List[DataSourceDescriptor] = []
        for source_id, source_url, source_type, data_resident, tensors, _ in rows:
            tensor_descs = [
                TensorDescriptor(
                    array_id=t["array_id"],
                    dim_labels=t["dim_labels"] or [],
                    shape=t["shape"] or [],
                    dtype=t["dtype"] or "",
                )
                for t in (tensors or [])
            ]
            descriptors.append(
                DataSourceDescriptor(
                    source_id=source_id,
                    source_url=source_url or "",
                    source_type=source_type or "",
                    tensors=tensor_descs,
                    metadata_json="",  # lean; filled by GetFlightInfo
                    data_resident=bool(data_resident),
                )
            )
        return descriptors, total

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
            conn.execute("DELETE FROM sources WHERE source_id = ?", [source_id])
        logger.debug(f"Removed source from metadata database: {source_id}")

    # ------------------------------------------------------------------
    # ROI annotations (docs/roi-annotations.md)
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
        now = datetime.now()

        stored: List[RoiAnnotation] = []
        conflicts: List[RoiConflict] = []

        with self._write_lock:
            source_url = self._observe_source(conn, source_id, now)

            # created_at is read for the RESPONSE only -- the update statement
            # does not carry it, so an existing row's value is preserved by not
            # being mentioned.
            existing = {
                roi_id: (rev, created_at)
                for roi_id, rev, created_at in conn.execute(
                    "SELECT roi_id, rev, created_at FROM rois WHERE array_id = ?",
                    [array_id],
                ).fetchall()
            }

            # Cap on the post-write count, so a batch cannot straddle the limit.
            new_ids = {p.roi_id for p in prepared if p.roi_id not in existing}
            if len(existing) + len(new_ids) > self._max_rois_per_tensor:
                raise ValueError(
                    f"Annotation limit reached for {array_id}: "
                    f"{len(existing)} stored + {len(new_ids)} new exceeds "
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
                    conflicts.append(
                        RoiConflict(roi_id=prep.roi_id, stored_rev=prior[0])
                    )
                    continue

                values = prep.column_values(self._ROI_CLIENT_COLUMNS)
                if prior is None:
                    rev, created_at = 1, now
                    conn.execute(
                        insert_sql,
                        [prep.roi_id, array_id, source_id, *values, rev, now, now]
                        # A fresh row is only "seen" if the catalog answered;
                        # inventing a sighting would reset an orphan clock.
                        + [source_url, now if source_url is not None else None],
                    )
                else:
                    rev, created_at = prior[0] + 1, prior[1]
                    conn.execute(update_sql, [*values, rev, now, array_id, prep.roi_id])
                stored.append(prep.to_proto(array_id, rev, created_at, now))

        logger.debug(
            "put_rois: %s stored, %s conflicts on %s",
            len(stored),
            len(conflicts),
            array_id,
        )
        return stored, conflicts

    @staticmethod
    def _observe_source(conn, source_id: str, now: datetime) -> Optional[str]:
        """Record a catalog sighting of *source_id*, returning its URL or None.

        Presence in the catalog is evidence; absence is not (progressive
        discovery, an unmounted drive, a proxy upstream that is down). So this
        writes only on presence, and a source the catalog cannot answer for
        leaves every stored row exactly as it was.

        On presence it does two things in one statement, for every row of the
        source:

        * ``COALESCE`` backfills a ``source_url`` that is still NULL, closing the
          window where annotations were written before discovery caught up. Rows
          that already have one keep it -- the URL is a human-readable label for
          a source_id, not an identifier, so filling a blank is always safe but
          overwriting is not the intent.
        * stamps ``last_seen_at``. This is the same statement the future prune
          sweep runs (docs/roi-annotations.md); the sweep adds only a
          catalog-completeness gate, which a *presence* observation does not
          need -- only a conclusion about absence does.
        """
        row = conn.execute(
            "SELECT source_url FROM sources WHERE source_id = ?", [source_id]
        ).fetchone()
        if row is None:
            logger.debug(
                "put_rois: %s is not in the catalog; annotations keep whatever "
                "source_url / last_seen_at they already had",
                source_id,
            )
            return None
        source_url = row[0]
        conn.execute(
            "UPDATE rois SET source_url = COALESCE(source_url, ?), last_seen_at = ? "
            "WHERE source_id = ?",
            [source_url, now, source_id],
        )
        return source_url

    def list_rois(
        self, array_id: str, set_name: str = ""
    ) -> Tuple[List[RoiAnnotation], bool]:
        """Return a tensor's whole annotation set (optionally one layer).

        No plane or bbox filter by design: a client needs every ROI resident to
        hit-test, drag a vertex and re-render, and a viewport-filtered fetch
        would make the ROI being edited vanish on a pan. Analytic slicing is the
        SQL surface's job.

        Returns:
            ``(rois, truncated)``. ``truncated`` is true only if the row count
            somehow exceeds the per-tensor cap (rows written before a lowered
            cap), in which case the result is clipped.
        """
        if not array_id:
            raise ValueError("array_id is required")

        sql = (
            "SELECT roi_id, array_id, set_name, label, plane, geometry, "
            "props_json, drawn_against_version, rev, created_at, updated_at "
            "FROM rois WHERE array_id = ?"
        )
        params: List[object] = [array_id]
        if set_name:
            sql += " AND set_name = ?"
            params.append(set_name)
        # Stable order so a client diffing two reads sees no spurious churn.
        sql += " ORDER BY created_at, roi_id LIMIT ?"
        params.append(self._max_rois_per_tensor + 1)

        rows = self._get_cursor().execute(sql, params).fetchall()
        truncated = len(rows) > self._max_rois_per_tensor
        if truncated:
            rows = rows[: self._max_rois_per_tensor]
        return [_row_to_proto(row) for row in rows], truncated

    def delete_rois(
        self, array_id: str, roi_ids: Iterable[str] = (), set_name: str = ""
    ) -> List[str]:
        """Delete annotations, returning the ids actually removed.

        With ``roi_ids``, deletes exactly those. Without, deletes every
        annotation on ``array_id`` -- narrowed to ``set_name`` when given, which
        is how a client drops a whole layer.
        """
        if not array_id:
            raise ValueError("array_id is required")

        roi_ids = list(roi_ids)
        conn = self._get_connection()
        with self._write_lock:
            if roi_ids:
                placeholders = ", ".join("?" for _ in roi_ids)
                sql = (
                    f"DELETE FROM rois WHERE array_id = ? "
                    f"AND roi_id IN ({placeholders}) RETURNING roi_id"
                )
                params: List[object] = [array_id, *roi_ids]
            else:
                sql = "DELETE FROM rois WHERE array_id = ?"
                params = [array_id]
                if set_name:
                    sql += " AND set_name = ?"
                    params.append(set_name)
                sql += " RETURNING roi_id"
            deleted = [row[0] for row in conn.execute(sql, params).fetchall()]

        logger.debug("delete_rois: removed %s from %s", len(deleted), array_id)
        return deleted

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            with self._write_lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
                    logger.info("MetadataDatabase closed")
