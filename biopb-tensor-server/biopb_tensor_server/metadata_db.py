"""DuckDB metadata database for efficient source filtering.

Provides indexed SQL queries against source metadata for large catalogs
(>100k sources). Replaces O(n) in-memory scans with indexed DuckDB queries.

Database Schema:
- sources table with indexed fields (source_id, source_url)
- JSON column for full metadata access via DuckDB JSON operators
- Shape summary column for quick size estimates

Security Model:
- DuckDB connection runs with enable_external_access=False, so all file/network
  access (read_csv, read_text, glob, COPY, ATTACH, extension loading) is blocked
  at the engine level. This is the primary defense against file exfiltration.
- Only 'sources' table accessible (keyword/table denylist; defense in depth)
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
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

if TYPE_CHECKING:
    from biopb_tensor_server.base import BackendAdapter

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

    # Only these tables can be referenced in queries
    ALLOWED_TABLES: Set[str] = {"sources"}

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
    ):
        self._max_query_results = max_query_results
        self._query_timeout_ms = query_timeout_ms

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
                tensors STRUCT(
                    array_id VARCHAR,
                    dim_labels VARCHAR[],
                    shape BIGINT[],
                    chunk_shape BIGINT[],
                    dtype VARCHAR
                )[]
            )
        """)
        # Index on source_url for path filtering
        conn.execute("CREATE INDEX idx_source_url ON sources(source_url)")
        logger.debug("Created sources table and indexes")

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
        schema = arrow_table.schema
        metadata = {
            b"total_sources": str(total_sources).encode(),
            b"returned_sources": str(
                min(returned_rows, self._max_query_results)
            ).encode(),
            b"query_elapsed_ms": str(int(elapsed_ms)).encode(),
        }
        schema = schema.with_metadata(metadata)

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

    def sync_source_added(self, source_id: str, adapter: BackendAdapter) -> None:
        """Sync a source to the metadata database (INSERT OR REPLACE upsert).

        Called by ``SourceManager`` when a source is registered and, for a
        previously-unresolved cloud source, again when it resolves (the upsert
        overwrites the placeholder row with the concrete descriptor).

        Raises on failure (descriptor read, JSON encode, or DB write) rather
        than swallowing, so the caller can react -- the registration path rolls
        back the matching ``register_source`` so the catalog and ``ListFlights``
        never silently disagree. Logging is the caller's responsibility.

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
        # ListFlights descriptor (source_desc.tensors), so this adds no adapter
        # call and no recall. Expensive/lazy fields (metadata_json, pyramid,
        # physical_scale) are intentionally omitted -- they are filled only by
        # GetFlightInfo. Unresolved cloud sources have no tensors -> empty list.
        tensors = [
            {
                "array_id": t.array_id,
                "dim_labels": list(t.dim_labels),
                "shape": [int(s) for s in t.shape],
                "chunk_shape": [int(c) for c in t.chunk_shape],
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

        logger.debug(f"Synced source to metadata database: {source_id}")

        # The catalog row is now the metadata cache for this source (biopb#253),
        # so let the adapter release any metadata it retained purely to answer
        # get_metadata(). Skip sources whose serve path still reads per-tensor
        # metadata off the adapter (HCS plates -> metadata_covers_all_tensors()
        # is False, biopb#269). Best-effort: a release failure must never fail
        # registration (the row is already committed). getattr-guarded so a
        # duck-typed adapter without these hooks is simply left alone.
        release = getattr(adapter, "release_retained_metadata", None)
        covers_all = getattr(adapter, "metadata_covers_all_tensors", None)
        if callable(release) and (not callable(covers_all) or covers_all()):
            try:
                release()
            except Exception as exc:
                logger.warning(
                    "release_retained_metadata failed for source %s: %s",
                    source_id,
                    exc,
                )

    def get_metadata_json(self, source_id: str) -> Optional[dict]:
        """Return a source's stored metadata as a dict, or ``None`` to fall back.

        The catalog stores ``json.dumps(adapter.get_metadata())`` -- the **raw**
        dict, no envelope -- so the serve path can read metadata back with a
        cheap local ``SELECT`` instead of recomputing it on the adapter
        (biopb/biopb#253), and for a remote proxy without an upstream RPC (read
        the local mirror row directly, never ``adapter.get_metadata()``). The
        stored JSON is parsed here so callers get a ready dict.

        Returns ``None`` in every "no usable catalog metadata" case -- so the
        caller uniformly falls back to ``adapter.get_metadata()``:
        - the source is absent, or its metadata is SQL NULL (empty is stored as
          NULL),
        - the stored value is not valid JSON / not a JSON object,
        - the DuckDB read itself fails.

        Never raises: a catalog read error degrades to the adapter path rather
        than failing the serve. Uses ``cursor()`` for a thread-safe read.
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
            return None

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
        reconstructed: per-tensor ``array_id``/``dim_labels``/``shape``/
        ``chunk_shape``/``dtype`` from the ``tensors`` STRUCT[] (biopb/biopb#224).
        ``metadata_json`` is left empty (filled by ``GetFlightInfo``), exactly
        like the adapter path. ``data_resident`` is the stored snapshot -- the
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
                    chunk_shape=t["chunk_shape"] or [],
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

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            with self._write_lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
                    logger.info("MetadataDatabase closed")
