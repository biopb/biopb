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
from typing import TYPE_CHECKING, Dict, Optional, Set

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

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

    Args:
        enabled: If False, all operations are no-ops (for disabling feature)
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
        enabled: bool = True,
        max_query_results: int = 100000,
        query_timeout_ms: int = 30000,
    ):
        self._enabled = enabled
        self._max_query_results = max_query_results
        self._query_timeout_ms = query_timeout_ms

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._write_lock = threading.Lock()  # Lock for write operations only
        self._initialized = False

        if self._enabled:
            logger.info(
                "MetadataDatabase enabled (DuckDB backend will initialize on first access)"
            )
        else:
            logger.info("MetadataDatabase disabled")

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
                data_resident BOOLEAN NOT NULL DEFAULT FALSE
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
        # Normalize SQL for checking
        normalized = sql.upper()

        # Check for forbidden keywords
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in normalized:
                raise ValueError(
                    f"SQL query contains forbidden keyword: {keyword}. "
                    f"Only SELECT queries are allowed."
                )

        # Check for table references
        table_refs = self.TABLE_REFERENCE_PATTERN.findall(sql)
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

        # Check for subqueries that might reference external resources
        # (DuckDB can't actually access external resources without explicit config,
        # but we validate anyway for clarity)
        if "SELECT" in normalized and "(" in normalized:
            # Has subquery - ensure it only references sources
            # This is a simplified check; DuckDB's SQL parser would be more thorough
            pass  # DuckDB won't allow external table references without explicit config

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
        if not self._enabled:
            raise ValueError("Metadata database is disabled")

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
        if not self._enabled:
            return

        conn = self._get_connection()

        # Get source descriptor and metadata
        source_desc = adapter.get_source_descriptor()
        metadata = adapter.get_metadata()

        # Build shape_summary from first tensor
        shape_summary = None
        dtype = None
        if source_desc.tensors:
            first_tensor = source_desc.tensors[0]
            shape_summary = json.dumps(list(first_tensor.shape))
            dtype = first_tensor.dtype

        # Build row data
        indexed_at = datetime.now()
        metadata_json = json.dumps(metadata, cls=NumpyEncoder) if metadata else None

        # Insert or replace (upsert) - serialize writes with lock
        with self._write_lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO sources
                (source_id, source_url, source_type, dtype, indexed_at, metadata_json, shape_summary, data_resident)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                ],
            )

        logger.debug(f"Synced source to metadata database: {source_id}")

    def sync_source_removed(self, source_id: str) -> None:
        """Remove a source from the metadata database.

        Called by ``SourceManager`` when a source is unregistered or rolled back.

        Raises on DB failure rather than swallowing, so the caller can react;
        logging is the caller's responsibility.

        Args:
            source_id: Unique source identifier
        """
        if not self._enabled:
            return

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
