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
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import duckdb
import numpy as np
import pyarrow as pa
from biopb.image.annotation_pb2 import RoiAnnotation
from biopb.tensor._catalog_rows import SOURCE_ROW_COLUMNS

from biopb_tensor_server.core.adapter_base import to_catalog_url
from biopb_tensor_server.core.errors import AnnotationStoreError
from biopb_tensor_server.serving import catalog_tree
from biopb_tensor_server.serving.roi_store import (
    _ROIS_DDL,
    RESERVED_SET_PREFIX,
    RoiStoreMixin,
    _prepare_roi,
)
from biopb_tensor_server.serving.tree_paths import tree_columns

if TYPE_CHECKING:
    from biopb_tensor_server.core.adapter_base import SourceAdapter
    from biopb_tensor_server.core.discovery import SourceClaim
    from biopb_tensor_server.sources.pending_rows import PendingRow

logger = logging.getLogger(__name__)

# Bump when a field of the persisted source row or payload changes meaning (an
# added key needs none). A mismatch drops ``source_catalog`` whole at open.
SOURCE_CATALOG_FORMAT = 5

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
#
# ``tree_dir`` and ``tree_name`` are the same url's place in the catalog tree (see
# ``tree_paths``), kept beside it so a level of the tree is a lookup on them rather
# than a rewrite of every url. They are a function of the root's url and ``rel``,
# so they are rewritten when either changes.
INTERNAL_ROOT_ID = "internal"
_LOCATION_COLUMN_NAMES = ("root_id", "rel", "tree_dir", "tree_name")
# The claim, the columns a source with no claim leaves NULL (bar the last).
_CLAIM_COLUMN_NAMES = (
    "primary_path",
    "member_paths",
    "extra_config",
    "signature",
    "payload",
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
# claim alone, bar the ``rel`` of a row under the built-in root, which is its url
# (and so its place in the tree).
_LISTING_COLUMN_NAMES = _ROW_COLUMN_NAMES[2:]
_INTERNAL_PLACE = ("rel", "tree_dir", "tree_name")
_LISTING_SET = ", ".join(f"{c} = ?" for c in _LISTING_COLUMN_NAMES) + "".join(
    f", {c} = CASE WHEN root_id = '{INTERNAL_ROOT_ID}' THEN ? ELSE {c} END"
    for c in _INTERNAL_PLACE
)
_LISTING_UPSERT_SET = ", ".join(
    f"{c} = excluded.{c}" for c in _LISTING_COLUMN_NAMES
) + "".join(
    f", {c} = CASE WHEN source_catalog.root_id = '{INTERNAL_ROOT_ID}' "
    f"THEN excluded.{c} ELSE source_catalog.{c} END"
    for c in _INTERNAL_PLACE
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


def _source_tree_view_ddl() -> str:
    """The published ``source_tree`` view: where each source sits in the catalog
    tree. ``dir`` is the folder as its segments joined by ``\\x01`` (``""`` for the
    top) and ``name`` the last segment (see ``tree_paths``); ``visible`` is whether
    a viewer lists the source, which a resolved source holding no tensor is not.
    Like ``sources``, a row shows only against its root."""
    return (
        "CREATE VIEW source_tree AS SELECT c.source_id, c.tree_dir AS dir, "
        "c.tree_name AS name, (NOT c.is_resolved OR len(c.tensors) > 0) AS visible "
        + _VIEW_FROM
    )


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


class MirroredRow(NamedTuple):
    """A mirrored source's row as :meth:`MetadataDatabase.sync_mirrored_rows` takes
    it: the upstream's, with the ids already local. ``rel`` is the path beneath
    the upstream's root."""

    source_id: str
    rel: str
    metadata_json: Optional[str]
    is_resolved: bool
    tensors: List[Dict[str, Any]]


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


class _ImportedRoisFailed(Exception):
    """The ROI half of a registration write failed; the row's half did not."""


class MetadataDatabase(RoiStoreMixin):
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

    # The queryable catalog: what the ``catalog`` flight lists and SQL may
    # read. Querying is full access (the server-wide token, or open in local
    # mode), and a caller with that can read every annotation on the ``roi``
    # flight anyway; a narrow grant or a seal never reaches this surface. A
    # query permission narrower than full access would need a set-level view
    # (array_id, set_name, count) here instead of ``rois``.
    # Enforced on DuckDB's own parse of the statement (``_validate_query``),
    # never on the SQL text.
    ALLOWED_TABLES: Set[str] = {"sources", "source_tree", "decode_rates", "rois"}

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
        restore_sources: bool = True,
    ):
        self._checkpoint_threshold_mb = checkpoint_threshold_mb
        #: Keep ``source_catalog`` across a restart (``catalog.restore``); off
        #: empties it at open, so the first scan rebuilds it.
        self.restore_sources = restore_sources
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
        # Each root's url, which a row's place in the tree depends on. Kept in step
        # with ``catalog_roots`` by whatever writes it.
        self._root_urls: Dict[str, str] = {}
        # Counts the writes that change what ``sources`` shows; see
        # ``catalog_generation``.
        self._generation = 0
        # ``(generation, count)``: the size of ``sources`` as of a generation.
        self._source_count: Tuple[int, int] = (-1, 0)

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
        conn.execute("DROP VIEW IF EXISTS source_tree")
        conn.execute("DROP VIEW IF EXISTS source_confirmation")
        # A build with a separate table for the sources that have no claim left it
        # behind.
        conn.execute("DROP TABLE IF EXISTS sources_volatile")
        self._create_source_catalog(conn)
        conn.execute(_sources_view_ddl())
        conn.execute(_source_tree_view_ddl())
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
        Rows are kept across an open unless ``restore_sources`` is off, and read
        back by :meth:`restorable_rows` (a restore registers or marks pending every
        row it keeps, so none is listed with no adapter behind it); the rows under
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
                -- Its place in the catalog tree, from the root's url and `rel`
                -- (`tree_paths`): the folder, and the name in it.
                tree_dir TEXT NOT NULL DEFAULT '',
                tree_name TEXT NOT NULL DEFAULT '',
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
                -- The run that wrote it (``run_epoch``).
                epoch BIGINT NOT NULL DEFAULT 0
            )
        """)
        # A file written before the tree columns existed keeps its rows: the
        # columns are added here and filled below, once the roots are known, rather
        # than dropping a catalog that took a long scan to build.
        for column in ("tree_dir", "tree_name"):
            conn.execute(
                f"ALTER TABLE source_catalog ADD COLUMN IF NOT EXISTS {column} "
                "TEXT DEFAULT ''"
            )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS catalog_roots (
                root_id TEXT PRIMARY KEY,
                root_url TEXT NOT NULL,
                -- Whether its sources are kept across a restart. Only a root the
                -- config names is; a drop, an upstream and the built-in root are
                -- gone at open.
                persisted BOOLEAN NOT NULL DEFAULT TRUE,
                -- The last run whose walk of this root finished.
                epoch BIGINT NOT NULL DEFAULT 0
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
        self._root_urls = dict(
            conn.execute("SELECT root_id, root_url FROM catalog_roots").fetchall()
        )
        # Rows the columns were added to, or whose fill an earlier open did not
        # finish: a source with a path always has a name.
        if conn.execute(
            "SELECT 1 FROM source_catalog WHERE tree_name = '' "
            "AND rel NOT IN ('', '.') LIMIT 1"
        ).fetchone():
            for root_id, root_url in self._root_urls.items():
                self._retree_root(conn, root_id, root_url)
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

    def catalog_children(
        self, key: str, limit: int = catalog_tree.DEFAULT_LIMIT
    ) -> Dict[str, Any]:
        """One level of the catalog tree; see :func:`catalog_tree.children`."""
        return catalog_tree.children(self._get_cursor(), key, limit)

    def _count_sources(self, cursor: duckdb.DuckDBPyConnection) -> int:
        """``COUNT(*)`` of ``sources``, counted once per generation: it is on every
        query's reply and a scan of the catalog."""
        generation, count = self._source_count
        if generation != self._generation:
            generation = self._generation
            count = cursor.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            self._source_count = (generation, count)
        return count

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
            total_sources = self._count_sources(cursor)
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
                b"catalog_generation": self.catalog_generation.encode(),
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
        leaves the previous row and ROIs both as they were. Only a failure of the
        ROI write itself is the exception: it is logged, and the row is written
        without them.

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
        catalog = self._catalog_tensors(source_id, adapter)

        # Full per-tensor structural info (biopb/biopb#224): one struct per
        # tensor, not just tensors[0]. Expensive/lazy fields (metadata_json,
        # pyramid, physical_scale) are omitted -- they belong to the
        # tensor-bound adapter GetFlightInfo binds. Unresolved cloud sources
        # have no tensors -> empty list.
        tensors = self._tensor_rows(catalog)

        # Every source with a claim is persisted; the payload only lets a
        # restart skip the parse, so an adapter without one (or a cloud row)
        # stores NULL and is rebuilt from its claim.
        # Before the record, which is the adapter's cue to drop what it parked
        # for it: the payload reads the same intermediates.
        payload = None
        if record is not None and record.claim is not None and not record.cloud:
            payload = adapter.catalog_payload()

        # The file's metadata and ROIs (#951), built together by the adapter:
        # the FORMAT decides whether its file carries annotations, because the
        # server does not police what registration_record returns -- a `rois` key in
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

        row = [
            source_id,
            source_url,
            source_type,
            indexed_at,
            metadata_json,
            True,  # a source with an adapter is resolved; ``sync_pending_source`` writes the rest
            None,  # ... and sets the reason
            tensors,
            None,
        ]

        def replace_rois(c: duckdb.DuckDBPyConnection) -> None:
            try:
                self._replace_imported(c, source_id, rois)
            except Exception as exc:
                raise _ImportedRoisFailed from exc

        # The row and the file's ROIs are one transaction, so they come from the
        # same registration. DuckDB has no savepoints (a failed statement aborts
        # the transaction), so when only the ROIs fail the whole write rolls back
        # and the row goes in alone: no annotation is worth failing a
        # registration, the previous set stays, and the next registration retries.
        try:
            self._upsert_source_row(conn, row, record, payload, also=replace_rois)
        except _ImportedRoisFailed:
            logger.exception(
                "ome rois: could not store the imported set for %s; the source "
                "is registered and serving, and the next registration retries",
                source_id,
            )
            self._upsert_source_row(conn, row, record, payload)

        logger.debug(f"Synced source to metadata database: {source_id}")

    # What makes a mirrored row worth writing again: any column the upstream's
    # row decides. ``indexed_at`` is the write's own clock and is not one.
    _MIRROR_CHANGED = " OR ".join(
        f"source_catalog.{c} IS DISTINCT FROM excluded.{c}"
        for c in (
            "source_type",
            "metadata_json",
            "is_resolved",
            "tensors",
            "root_id",
            "rel",
        )
    )

    def sync_mirrored_rows(
        self, root_id: str, source_type: str, rows: Sequence[MirroredRow]
    ) -> None:
        """Write *rows*, mirrored from an upstream, in one transaction.

        The rows are the upstream's own: nothing is read off an adapter or
        re-encoded, so ``metadata_json`` goes in as the upstream wrote it. A row
        already there with the same content is left alone, ``indexed_at``
        included, so a steady re-list of an unversioned upstream does not churn
        the catalog. The root must be known (:meth:`ensure_root`).
        """
        if not rows:
            return
        conn = self._get_connection()
        now = datetime.now()
        statements = []
        for i in range(0, len(rows), self._PENDING_CHUNK):
            chunk = rows[i : i + self._PENDING_CHUNK]
            params: List[Any] = []
            for r in chunk:
                params += [
                    r.source_id,
                    source_type,
                    now,
                    r.metadata_json,
                    r.is_resolved,
                    None,  # a mirror's reason is the upstream's, and is_resolved says it
                    r.tensors,
                    None,
                ]
                params += self._placement(
                    CatalogRecord(None, {}, root_id, r.rel), r.rel, None
                )
            width = len(params) // len(chunk)
            statements.append(
                (
                    f"INSERT INTO source_catalog ({_ALL_COLUMNS}) VALUES "
                    + ", ".join([f"({', '.join('?' * width)})"] * len(chunk))
                    + f" ON CONFLICT (source_id) DO UPDATE SET {_UPSERT_SET}"
                    + f" WHERE {self._MIRROR_CHANGED}",
                    params,
                )
            )
        with self._write_lock:
            with self._catalog_txn(conn):
                for sql, params in statements:
                    conn.execute(sql, params)

    def sync_mirrored_removed(self, source_ids: Sequence[str]) -> None:
        """Drop the rows of mirrored sources in one transaction.

        :meth:`sync_source_removed` for a batch, and only the row: a mirror's
        registration imports no ROIs (:meth:`sync_mirrored_rows` writes none), so
        there are no reserved sets of its own to delete.
        """
        if not source_ids:
            return
        conn = self._get_connection()
        with self._write_lock:
            with self._catalog_txn(conn):
                for i in range(0, len(source_ids), self._PENDING_CHUNK):
                    chunk = list(source_ids[i : i + self._PENDING_CHUNK])
                    conn.execute(
                        "DELETE FROM source_catalog WHERE source_id IN "
                        f"({', '.join('?' * len(chunk))})",
                        chunk,
                    )

    def bind_registry(self, registry: Any) -> None:
        """List tensors through *registry*, which holds the ones attached to a source,
        and keep each row's listing in step with them."""
        self._registry = registry
        registry.set_attachment_listener(self._attachments_changed)

    def _attachments_changed(self, source_id: str) -> None:
        """Re-list a source whose attached tensors changed (:meth:`relist_tensors`).

        Best-effort for the reason every catalog write on the upload path is: the
        row is how a source is *browsable*, and it must not be able to fail the
        change that made a tensor *readable*. A source with no adapter or no row
        has nothing to re-list."""
        adapter = self._registry.resolve(source_id)
        if adapter is None:
            return
        try:
            self.relist_tensors(source_id, adapter)
        except Exception as e:
            logger.warning(
                f"Failed to re-list {source_id} in the catalog "
                f"(readable by id, not listed): {e}"
            )

    def _catalog_tensors(self, source_id: str, adapter: Any) -> List[Any]:
        """The tensors a row lists: the registry's view when bound, else the
        adapter's own (a catalog on its own has no attachments)."""
        if self._registry is not None:
            return self._registry.catalog_tensors(source_id, adapter)
        return list(adapter.list_tensors())

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

        For a source whose attached fields and label sets are not what its row
        lists: one rebuilt from its row, written before those on disk were attached
        (or before some of them went), or one whose attachments just changed
        (:meth:`_attachments_changed`). Writes the ``tensors`` column and nothing
        else when it differs -- the metadata the row holds is not re-read -- and
        reports whether it did. A source with no row is left without one. The order
        is not compared: a row lists fields in the order they were uploaded and the
        attach scan finds them by name, and neither is a change.
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
            self._catalog_written()
        return True

    @property
    def catalog_generation(self) -> str:
        """Names the catalog's content: it differs whenever a source row, or the
        root a row is shown against, was written since it was last read, and the
        run is part of it so it never repeats across a restart. A reader that read
        it before its query holds a result at least that fresh, so equal
        generations mean equal content.
        """
        return f"{self.run_epoch}.{self._generation}"

    @contextmanager
    def _catalog_txn(self, conn: duckdb.DuckDBPyConnection) -> Iterator[None]:
        """A write transaction on the catalog, for a caller holding the write lock:
        rolled back on an error, committed and then counted in the generation."""
        conn.execute("BEGIN TRANSACTION")
        try:
            yield
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        self._catalog_written()

    def _catalog_written(self) -> None:
        """Advance the generation. After the write is committed, never before: a
        generation read ahead of the data would label old content as new."""
        self._generation += 1

    def _stored_root_url(
        self, conn: duckdb.DuckDBPyConnection, root_id: str
    ) -> Optional[str]:
        """The url ``catalog_roots`` holds for *root_id*, or None."""
        row = conn.execute(
            "SELECT root_url FROM catalog_roots WHERE root_id = ?", [root_id]
        ).fetchone()
        return None if row is None else row[0]

    def _root_url(self, root_id: str) -> str:
        """A root's url, for the place its rows take in the tree; "" for a root the
        catalog does not hold, whose rows the view does not show."""
        url = self._root_urls.get(root_id)
        if url is None:
            url = self._stored_root_url(self._get_cursor(), root_id)
            if url is None:
                return ""
            self._root_urls[root_id] = url
        return url

    @staticmethod
    def _retree_root(
        conn: duckdb.DuckDBPyConnection, root_id: str, root_url: str
    ) -> None:
        """Rewrite the tree columns of every row under a root whose url changed (an
        alias edited, a root that arrived after its rows). Inside the caller's
        transaction; a no-op for a root with no rows."""
        rows = conn.execute(
            "SELECT source_id, rel FROM source_catalog WHERE root_id = ?", [root_id]
        ).fetchall()
        if not rows:
            return
        ids, dirs, names = [], [], []
        for source_id, rel in rows:
            tree_dir, tree_name = tree_columns(root_url, rel)
            ids.append(source_id)
            dirs.append(tree_dir)
            names.append(tree_name)
        conn.execute(
            "UPDATE source_catalog SET tree_dir = v.d, tree_name = v.n FROM "
            "(SELECT unnest(?::VARCHAR[]) AS id, unnest(?::VARCHAR[]) AS d, "
            "unnest(?::VARCHAR[]) AS n) v WHERE source_catalog.source_id = v.id "
            "AND (tree_dir IS DISTINCT FROM v.d OR tree_name IS DISTINCT FROM v.n)",
            [ids, dirs, names],
        )

    def _placement(
        self,
        record: Optional[CatalogRecord],
        url: str,
        payload: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """The ``source_catalog`` columns after the public ones, in table order.

        Without a *record* the source sits under the built-in root and *url* is its
        ``rel``. Without a claim in it the claim columns are NULL.
        """
        if record is None:
            location = [INTERNAL_ROOT_ID, url, *tree_columns("", url)]
            claim_values = [None] * 5
        else:
            location = [
                record.root_id,
                record.rel,
                *tree_columns(self._root_url(record.root_id), record.rel),
            ]
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
        return location + claim_values + [self.run_epoch]

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
        source_id, url = row[0], row[1]
        # Serialized before the lock: a payload can be large, and every other
        # writer waits while it is held.
        values = row[:1] + row[2:] + self._placement(record, url, payload)
        if record is None:
            update_set, upsert_set = _LISTING_SET, _LISTING_UPSERT_SET
            set_values = row[2:] + [url, *tree_columns("", url)]
        else:
            update_set, upsert_set = _UPDATE_SET, _UPSERT_SET
            set_values = values[1:]
        with self._write_lock:
            with self._catalog_txn(conn):
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
                params += self._placement(row.record, values[1], None)
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
            with self._catalog_txn(conn):
                for sql, params in statements:
                    conn.execute(sql, params)

    def restorable_rows(self) -> List[Dict[str, Any]]:
        """The rows with a claim, which a restore rebuilds claims from, without the
        payload and the metadata (large, and read per source when it is hydrated).

        ``idle_runs`` is how many earlier runs neither wrote the row nor finished a
        walk of its root.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT c.source_id, c.source_type, c.is_resolved, "
            "c.unresolved_reason, c.unresolved_error, c.root_id, c.rel, "
            "c.primary_path, c.member_paths, c.extra_config, c.signature, "
            "? - 1 - greatest(c.epoch, r.epoch) AS idle_runs "
            "FROM source_catalog c JOIN catalog_roots r ON c.root_id = r.root_id "
            "WHERE c.primary_path IS NOT NULL",
            [self.run_epoch],
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

    def read_mirrored(
        self, source_id: str
    ) -> Optional[Tuple[str, List[Dict[str, Any]], bool]]:
        """``(source_url, tensors, is_resolved)`` of a catalogued source, or None:
        what a mirror's adapter is rebuilt from, in the local ids the row was
        written with. ``is_resolved`` is the row's, as the upstream last said."""
        row = (
            self._get_cursor()
            .execute(
                "SELECT source_url, tensors, is_resolved FROM sources "
                "WHERE source_id = ?",
                [source_id],
            )
            .fetchone()
        )
        if row is None:
            return None
        return row[0] or "", list(row[1] or []), bool(row[2])

    def drop_catalog_rows(self, source_ids: Sequence[str]) -> None:
        """Delete persisted rows a restore did not keep, with the reserved ROI rows
        each one's registration derived (see :meth:`sync_source_removed`)."""
        if not source_ids:
            return
        conn = self._get_connection()
        ids = list(source_ids)
        with self._write_lock:
            with self._catalog_txn(conn):
                conn.execute(
                    "DELETE FROM source_catalog WHERE source_id IN (SELECT unnest(?::VARCHAR[]))",
                    [ids],
                )
                conn.execute(
                    "DELETE FROM rois WHERE source_id IN (SELECT unnest(?::VARCHAR[])) "
                    "AND starts_with(set_name, ?)",
                    [ids, RESERVED_SET_PREFIX],
                )

    def rewrite_row_roots(self, rows: Sequence[Tuple[str, str, str]]) -> None:
        """Re-attribute persisted rows to the roots a restore found them under:
        ``(source_id, root_id, rel)`` each."""
        if not rows:
            return
        conn = self._get_connection()
        with self._write_lock:
            with self._catalog_txn(conn):
                for source_id, root_id, rel in rows:
                    tree_dir, tree_name = tree_columns(self._root_url(root_id), rel)
                    conn.execute(
                        "UPDATE source_catalog SET root_id = ?, rel = ?, "
                        "tree_dir = ?, tree_name = ? WHERE source_id = ?",
                        [root_id, rel, tree_dir, tree_name, source_id],
                    )

    def confirm_root(self, root_id: str) -> None:
        """Record that this run's walk of a root finished: every row under it that a
        restore brought back is now verified against the disk."""
        conn = self._get_connection()
        with self._write_lock:
            conn.execute(
                "UPDATE catalog_roots SET epoch = ? WHERE root_id = ?",
                [self.run_epoch, root_id],
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
            with self._catalog_txn(conn):
                conn.execute(
                    "DELETE FROM catalog_roots WHERE persisted AND root_id NOT IN "
                    "(SELECT unnest(?::VARCHAR[]))",
                    [ids],
                )
                for root_id, root_url in roots:
                    self._upsert_root(conn, root_id, root_url, persisted=True)
            self._root_urls = {**self._root_urls, **dict(roots)}

    def ensure_root(self, root_id: str, root_url: str) -> None:
        """Record a root that is not persisted (a drop, an upstream) before a source
        under it is written, since the view shows a row only against its root. It
        goes at the next open with the rows under it."""
        conn = self._get_connection()
        with self._write_lock:
            with self._catalog_txn(conn):
                self._upsert_root(conn, root_id, root_url, persisted=False)
            self._root_urls[root_id] = root_url

    def _upsert_root(
        self,
        conn: duckdb.DuckDBPyConnection,
        root_id: str,
        root_url: str,
        *,
        persisted: bool,
    ) -> None:
        """Write a ``catalog_roots`` row and re-tree its rows if its url moved.

        The caller holds the write lock and a catalog transaction. A conflict
        updates only the url: ``persisted`` and ``epoch`` stay as they were.
        """
        old = self._stored_root_url(conn, root_id)
        conn.execute(
            "INSERT INTO catalog_roots (root_id, root_url, persisted) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (root_id) DO UPDATE SET root_url = excluded.root_url",
            [root_id, root_url, persisted],
        )
        if old != root_url:
            self._retree_root(conn, root_id, root_url)

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

        The catalog stores ``json.dumps(adapter.registration_record)`` -- the **raw**
        dict, no envelope -- so the serve path can read metadata back with a
        cheap local ``SELECT`` instead of recomputing it on the adapter
        (biopb/biopb#253), and for a remote proxy without an upstream RPC (read
        the local mirror row directly, never ``adapter.registration_record``). The
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

        return self._loads_metadata(row[0], source_id) if row else None

    @staticmethod
    def _loads_metadata(raw: Optional[str], source_id: str) -> Optional[dict]:
        """A stored ``metadata_json`` as a dict, or ``None`` for empty or invalid."""
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
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
            self._catalog_written()
            # Reserved rows go with the source row: they are its scan output,
            # re-derived on the next registration. Hand-drawn annotations are
            # deliberately NOT touched here -- outliving their source is the
            # whole of the orphan design (biopb/biopb#951).
            conn.execute(
                "DELETE FROM rois WHERE source_id = ? AND starts_with(set_name, ?)",
                [source_id, RESERVED_SET_PREFIX],
            )
        logger.debug(f"Removed source from metadata database: {source_id}")

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
