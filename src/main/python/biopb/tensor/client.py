"""Python client for TensorFlight server.

This module provides a lazy numpy-like array interface using dask.array
for accessing tensors stored in a Flight server.

Features:
- Lazy chunk loading via dask.array
- LRU caching via cachey
- Numpy-compatible slicing and operations
"""

import json
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import pyarrow.flight as flight

# The pickle-safe connection/cache pool + cache-file fast path + chunk-fetch /
# dask-array builder subsystem lives in biopb.tensor._pool (issue #278 item C).
# Import only what TensorFlightClient uses directly below, plus ``configure_cache``
# -- re-exported (redundant `as` alias) for biopb-mcp's dask worker-init plugin,
# which pins each worker's cache budget via ``biopb.tensor.client.configure_cache``.
# The rest of _pool's internals are deliberately NOT re-exported here: their tests
# and benchmarks import them from ``biopb.tensor._pool`` directly. A client
# re-export would be a footgun -- ``_reset_pools_after_fork`` rebinds the module's
# locks (``_POOL_LOCK`` etc.), so a name bound here at import time goes stale after
# a fork, and patching a re-export never lands on the binding _pool actually
# resolves.
from biopb.image.annotation_pb2 import (
    RoiAnnotation,
    RoiDeleteResult,
    RoiListResult,
    RoiPruneResult,
    RoiPutResult,
)
from biopb.tensor._pool import (
    _CACHE_POOL,
    _VIEW_CACHE,
    _build_call_options,
    _clear_view_cache,
    _default_cache_bytes,
    _resolve_cache_bytes,
    configure_cache as configure_cache,
)
from biopb.tensor._session import (
    CatalogClient,
    ChunkFetcher,
    ResolveCancelled as ResolveCancelled,
    _check_wire_protocol as _check_wire_protocol,
    _ClientState,
    _dask_from_flight_info,
    _extract_schema_metadata as _extract_schema_metadata,
    _parse_version as _parse_version,
    _refetch_flight_info,
    _requested_slice,
    _split_array_id as _split_array_id,
)
from biopb.tensor._tls import resolve_tls_trust
from biopb.tensor._upload import UploadRefused as UploadRefused, UploadSession
from biopb.tensor.descriptor_pb2 import (
    AddSourceProgress,
    AddSourceResult,
    DataSourceDescriptor,
    RemoveSourceResult,
    ResolveProgress,
    TensorDescriptor,
    WarmProgress,
)
from biopb.tensor.serialized_pb2 import SerializedTensor
from biopb.tensor.ticket_pb2 import ChunkBounds

logger = logging.getLogger(__name__)


def _normalize_location(location: str) -> str:
    """Normalize location URI for Arrow Flight.

    Converts grpcs:// to grpc+tls:// (Arrow Flight's TLS scheme).
    """
    if location.startswith("grpcs://"):
        return "grpc+tls://" + location[8:]
    return location


class TensorFlightClient:
    """Client for accessing tensors from a TensorFlightServer.

    This client provides lazy, cached access to multi-dimensional arrays
    stored in a Flight server, with support for multifield acquisitions
    where tensors within a source have different shapes.

    Example:
        ```python
        client = TensorFlightClient('grpc://localhost:8815')

        # Browse the catalog (SQL over the server's DuckDB)
        rows = client.query_sources("SELECT * FROM sources", format="records")

        # Get source-level metadata
        metadata = client.get_source_metadata('my-source')

        # Access a tensor by its globally-unique array_id (identity policy):
        # 'source_id/field' for a multi-tensor source, or 'source_id' for a
        # single-tensor one. See proto/biopb/tensor/descriptor.proto.
        arr = client.get_tensor('my-source/tensor-0')  # Returns dask.array
        data = arr[0:100, 0:100].compute()   # Load slice
        ```

    Note:
        The dask arrays returned by get_tensor() are picklable and work with
        dask.distributed: each worker fetches chunks over its own connection,
        so you can scatter an array across a cluster and compute on it.
    """

    # The arrays are pickle-safe because the fetch functions hold no FlightClient
    # in their closure -- connections, caches, and call options are recreated
    # lazily per worker process from module-level pools keyed by (location, token).

    def __init__(
        self,
        location: str = "grpc://localhost:8815",
        cache_bytes: Optional[int] = None,
        token: Optional[str] = None,
        tls_ca_pem: Optional[bytes] = None,
        tls_fingerprint: Optional[str] = None,
    ):
        """Initialize the Flight client.

        Args:
            location: Flight server location
            cache_bytes: Maximum bytes for the chunk cache. ``None`` (the default)
                resolves ``BIOPB_TENSOR_CACHE_LIMIT`` (a size string like ``"2GiB"``,
                or a bare byte count) and falls back to 1 GB; a value passed here
                overrides the env. ``0`` disables the cache.
            token: Bearer token for server authentication.  ``None`` disables auth.
            tls_ca_pem: PEM bytes to trust for a ``grpcs://`` location (a private
                CA, or the server's own certificate), instead of pinning whatever
                the server presents on first connect. Bytes rather than a path:
                which file a cert came from is the caller's policy, not the SDK's.
            tls_fingerprint: Expected SHA-256 of the server's certificate for a
                ``grpcs://`` location, colon-grouped or bare hex. Checked on every
                connect, so unlike trust-on-first-use it also rejects an attacker
                who is already in the path the first time. Ignored when
                *tls_ca_pem* is given.
        """
        if cache_bytes is None:
            cache_bytes = _default_cache_bytes()
        logger.info(
            f"Connecting to Flight server at {location}, cache={cache_bytes}B, auth={token is not None}"
        )
        # Normalize location for Arrow Flight (grpcs:// -> grpc+tls://)
        normalized = _normalize_location(location)
        # For a TLS location, resolve the trust -- a caller-supplied CA or
        # fingerprint, else TOFU (once per process, memoized in _tls) -- and carry
        # it through the connection so every dask worker trusts the same root
        # without touching the pin store or needing the credentials itself
        # (biopb/biopb#604). NO_TLS for plaintext.
        tls_trust = resolve_tls_trust(
            normalized, ca_pem=tls_ca_pem, expected_fingerprint=tls_fingerprint
        )
        # Pickle-safe connection parameters (callers read client._client etc.)
        self._location = normalized
        self._token = token
        self._cache_bytes = cache_bytes
        self._tls_trust = tls_trust
        self._client = flight.FlightClient(normalized, **tls_trust.client_kwargs())
        self._call_options = _build_call_options(token)
        # One shared _ClientState holds the connection, and only the connection:
        # the collaborators (#278 item C) cache no descriptors, so a caller that
        # wants one memoized owns that policy.
        self._state = _ClientState(
            raw_client=self._client,
            call_options=self._call_options,
            location=self._location,
            token=self._token,
            cache_bytes=self._cache_bytes,
            tls_trust=self._tls_trust,
        )
        self._catalog = CatalogClient(self._state)
        self._fetcher = ChunkFetcher(self._state, self._catalog)
        self._upload = UploadSession(self._state)

    # ---- Catalog / metadata / source lifecycle (delegated to CatalogClient) ----

    def list_sources(self) -> Dict[str, DataSourceDescriptor]:
        """List available data sources.

        Deprecated:
            Use :meth:`query_sources`, which hands back rows in the format
            you ask for and leaves the structure to you. This is a thin
            wrapper around
            ``SELECT ... FROM sources`` that inherits the server's query row
            cap, so a large catalog comes back silently truncated -- and a
            browse is exactly where that matters.

        Returns:
            Dictionary mapping source_id to DataSourceDescriptor.
            Each DataSourceDescriptor.tensors carries the *structural* entry for
            every tensor in that source -- array_id, dim_labels, shape, dtype.
            The transfer ``chunk_shape`` is empty here by contract; ask
            :meth:`get_descriptor` for the grid of a specific tensor
            (biopb/biopb#812). ``is_resolved`` is not carried at all -- the
            message has no field for it (biopb/biopb#1032).
        """
        warnings.warn(
            "TensorFlightClient.list_sources() is deprecated and is capped by "
            "the server's query row limit; use query_sources(), which returns "
            "rows in the format you ask for.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._catalog.list_sources()

    def get_source(self, source_id: str) -> Optional[DataSourceDescriptor]:
        """One source's ``DataSourceDescriptor`` by id, or ``None``.

        Deprecated:
            Use :meth:`query_sources` with a ``WHERE source_id = ...``.

        The catalog is public: a source whose pixels need a capability token
        still has its descriptor here. Knowing its id is not authority to read
        it -- that is what the token gates, on :meth:`get_tensor` and
        :meth:`list_rois`.

        Args:
            source_id: The source's id, e.g. ``"zarr_a3f2"``. This is a *source*
                id, not an array_id: pass the routing prefix, not
                ``"aics_7f3/Image:0"``.

        Returns:
            The ``DataSourceDescriptor``, or ``None`` when nothing answers to
            that id.
        """
        warnings.warn(
            "TensorFlightClient.get_source() is deprecated; use query_sources() "
            "with a WHERE source_id = ... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._catalog.get_source(source_id)

    def query_sources(self, sql: str, *, format: str = "arrow") -> Any:  # noqa: A002 - public, documented keyword API (mirrors DuckDB/pandas `format`)
        """Execute SQL query against server's source metadata database.

        The server-side metadata database is mandatory (biopb/biopb#225), so any
        standard tensor-server supports this. Only an embedded server explicitly
        constructed without a metadata database rejects the query.

        Args:
            sql: SQL query (e.g., "SELECT source_id, source_type FROM sources WHERE dtype='uint16'")
            format: Shape of the returned result:

                - ``"arrow"`` (default) — a ``pyarrow.Table``. This is the
                  historical return type; the default is unchanged for backward
                  compatibility. Zero-copy, and the only format that preserves
                  the schema metadata described under *Note*.
                - ``"pandas"`` — a ``pandas.DataFrame`` (requires pandas).
                - ``"records"`` — a ``list[dict]``, one dict per row.

        Returns:
            The query result in the requested ``format``; an empty query
            returns an empty object of that same type. For ``"pandas"`` and
            ``"records"`` the usual Arrow->Python coercion applies (list
            columns such as ``shape_summary`` become Python lists / object
            dtype, and nullable integer columns may widen to float). For
            ``"pandas"``, NULLs in string columns (e.g. ``metadata_json``) are
            normalized to ``None`` rather than the truthy float ``NaN`` Arrow
            would otherwise produce, so ``if row.metadata_json:`` behaves as
            expected.

        Note:
            The server reports truncation via schema metadata
            (``total_sources`` / ``returned_sources``). Those keys survive only
            on the ``"arrow"`` result; for every format truncation is also
            surfaced via a logged INFO line.

        Raises:
            ValueError: If *format* is not one of the supported values. (SQL
                validation -- forbidden keywords / disallowed tables -- happens
                server-side and surfaces as a Flight error, below, not a
                client-side ValueError.)
            ImportError: If ``format="pandas"`` but pandas is not installed.
            FlightServerError: If the server has no metadata database enabled,
                or rejects the query (e.g. forbidden keywords / disallowed
                tables).

        Example:
            ```python
            >>> client = TensorFlightClient('grpc://localhost:8815')
            >>> table = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
            >>> table.to_pandas()  # or pass format="pandas" to get a DataFrame
            ```
        """
        return self._catalog.query_sources(sql, format=format)

    @staticmethod
    def _format_query_result(table, format):  # noqa: A002 - public, documented keyword API (mirrors DuckDB/pandas `format`)
        """Coerce a query result to the requested format. See :meth:`CatalogClient._format_query_result`."""
        return CatalogClient._format_query_result(table, format)

    def get_source_metadata(self, source_id: str) -> dict:
        """Get source-level OME/vendor metadata as a dict.

        Source-scoped, and read from the source's own catalog row: this is the
        metadata the format carries for the whole container. A *field's* own
        extras (an OME-Zarr HCS field's OME block, an EMD signal's
        ``original_metadata``) are per-tensor and come back on a tensor-bound
        :meth:`get_descriptor` with ``with_metadata=True``.

        Args:
            source_id: Source identifier

        Returns:
            The source's metadata dict (the format-specific OME/vendor metadata),
            or an empty dict if the source carries none.

        Raises:
            ValueError: If the source is unknown, or unresolved (cloud /
                synced-folder) -- call `resolve` first.
        """
        return self._catalog.get_source_metadata(source_id)

    def get_physical_scale(
        self, array_id: str
    ) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dimension physical pixel size + unit for a tensor.

        Returns ``(scale, unit)``: two lists aligned with the tensor's
        ``dim_labels`` (source axis order), or ``None`` when no physical sizes
        are known (an older server, or a format that carries none).

        ``physical_scale``/``physical_unit`` are ``TensorDescriptor`` fields the
        server fills on every ``GetFlightInfo`` (issue #31), so this describes the
        tensor and reads them off the answer, never requesting the opt-in
        ``metadata_json`` field on that same descriptor. (Contrast
        `get_source_metadata`, which ships the
        whole OME tree; do not dig physical sizes out of that -- this is the
        compact projection meant for display scale.)

        Args:
            array_id: Globally-unique tensor id (identity policy) -- e.g.
                ``"zarr_a3f2"`` or ``"aics_7f3/Image:0"``. A bare single-tensor
                source id resolves to its sole tensor. A bare *multi*-tensor
                source id anchors on the source's default (first) tensor --
                unlike ``get_tensor``, which requires the field be named; pass the
                qualified ``source_id/field`` to target a specific scene.

        Returns:
            ``(scale, unit)`` lists, or ``None`` if no physical scale is known.
        """
        return self._catalog.get_physical_scale(array_id)

    def get_descriptor(
        self,
        array_id: str,
        with_metadata: bool = False,
        with_pyramid: bool = True,
        with_read_plan: bool = False,
        with_residency: bool = False,
        with_upload_status: bool = False,
    ) -> TensorDescriptor:
        """Fetch one tensor's ``TensorDescriptor`` by its globally-unique array_id.

        A tensor is identified by its ``array_id`` alone (see the tensor identity
        policy at the top of ``proto/biopb/tensor/descriptor.proto``), so this
        takes that one identifier rather than a ``(source_id, tensor_id)`` pair.
        Works even when the source is beyond the server's query row cap.
        **This is the only call that answers the transfer ``chunk_shape``**: the
        grid belongs to the tensor the server binds here, and a catalog row
        carries it empty (biopb/biopb#812). Every call fetches and nothing is
        stored, so what you get back always reflects the masks you passed.
        Passing a bare
        ``source_id`` (single-tensor source, or to anchor on a multi-tensor
        source's default/first tensor) is accepted. To enumerate ALL
        tensors/scenes of a source, read its catalog row's ``tensors`` column
        -- NOT this method.

        This is a cheap probe -- it does NOT resolve. On an unresolved (cloud /
        synced-folder) source it raises an error pointing at `resolve`,
        never triggering a download. Call `resolve` first to read such a
        source.

        The ``with_*`` flags are the ``GetFlightInfo`` response field masks
        (biopb/biopb#563). This is a *describe* call -- the per-tensor facts, not
        a read -- so it defaults to returning shape/dtype/dim_labels/chunk_shape,
        the resolution **pyramid**, and physical_scale, while
        **skipping the read plan** (``with_read_plan=False`` -- the endpoints are
        the per-request O(chunks) half a describe discards) and the **heavy OME
        metadata tree** (``with_metadata=False``, opt-in). Set ``with_metadata=True``
        for ``metadata_json``; set ``with_pyramid=False`` to skip pyramid sizing
        when only the bare structure is needed.

        Args:
            array_id: Globally-unique tensor id, e.g. ``"zarr_a3f2"`` (single-
                tensor source) or ``"aics_7f3/Image:0"`` (multi-tensor source).
            with_metadata: fill ``metadata_json`` (the full OME tree). Default
                ``False`` -- opt in when you need it.
            with_pyramid: advertise the resolution pyramid on the descriptor.
                Default ``True`` (the primary describe consumer reads it).
            with_upload_status: fill ``upload_status`` for a source backed by
                an upload. Cheap (an in-memory record read), but off by default
                like every other optional part.
            with_residency: ask whether this source's bytes are local right
                now, answered on ``is_resident``. Off by default: the answer is
                a bounded stat walk of the source, so ask it for a source you
                are about to read, never in a loop over a listing
                (biopb/biopb#1048). Unset on the response means nobody asked.
            with_read_plan: enumerate the per-request chunk endpoints. Default
                ``False``; a describe discards them, so the plan is skipped.

        Returns:
            The ``TensorDescriptor`` for that tensor.
        """
        return self._catalog.get_descriptor(
            array_id,
            with_metadata=with_metadata,
            with_pyramid=with_pyramid,
            with_read_plan=with_read_plan,
            with_residency=with_residency,
            with_upload_status=with_upload_status,
        )

    def resolve(
        self,
        source_id: str,
        *,
        on_progress: Optional[Callable[[ResolveProgress], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Resolve an unresolved source and return its ``sources`` catalog row.

        Note:
            Experimental. Cloud / remote source support (unresolved sources,
            resolve, and `warm`) is experimental and its behavior may change.
            This returned a ``DataSourceDescriptor`` before biopb/biopb#1032
            and now returns the row itself -- the same information, without
            the SDK picking a structure for it.

        An *unresolved* source is catalogued by URL only -- its shape/dtype/field
        list are unknown until first access (its catalog row has
        ``is_resolved`` false and an empty ``tensors``). The canonical case is
        a cloud / synced-folder ("Files-On-Demand") source.

        Resolving asks the server to hydrate it. For a dehydrated placeholder this
        **downloads the whole file** -- a recall that can take minutes, consume
        local disk, and fail when offline -- then reads its real shape, dtype, and
        field list. This is the heavyweight, *consenting* operation that catalog
        browsing (`query_sources`) deliberately
        avoids; call it only when you intend to read the data. After it returns,
        `get_tensor` and friends work normally.

        Idempotent: resolving an already-resolved source just re-fetches it.

        Args:
            source_id: The source to resolve (e.g. ``"onedrive_a3f2"``).
            on_progress: Optional callback invoked with a ``ResolveProgress``
                (elapsed seconds, target name, target size in bytes) on each
                server heartbeat, so a caller can display progress. Called on the
                calling thread; keep it cheap and non-blocking.
            should_cancel: Optional predicate polled on each heartbeat; when it
                returns True the client stops consuming the stream and raises
                `ResolveCancelled`. The server-side recall continues to
                completion and is cached, so a later ``resolve`` reuses it.

        Returns:
            The source's ``sources`` row, shaped exactly like one element of
            ``query_sources(..., format="records")`` -- ``SOURCE_ROW_COLUMNS``,
            with every tensor enumerated under ``tensors``. A row, not a
            bespoke type: every client can already decode one, and the choice
            of what to decode it into stays yours (biopb/biopb#1032).

            It is the row the server just wrote, so it agrees with a following
            `query_sources` exactly, and returning it closes the transition in
            one call rather than leaving a window in which a rescan could
            re-register the source underneath you.

            Unlike `warm`, which returns a *status* because residency is not a
            durable catalog fact (biopb/biopb#1035) and its file counts exist
            nowhere else, this returns the *result*: resolving is defined by
            what it writes to the row. The recall's elapsed time and target
            size ride ``on_progress`` instead -- both are things a caller can
            already measure or derive, where `warm`'s counts are not.

        Raises:
            ResolveCancelled: if ``should_cancel`` asked to stop mid-resolve.
        """
        return self._catalog.resolve(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )

    def warm(
        self,
        source_id: str,
        *,
        on_progress: Optional[Callable[[WarmProgress], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> WarmProgress:
        """Hydrate-ahead: recall a resolved source's member files on the server.

        Note:
            Experimental. Cloud / remote source support (`resolve` and this hydrate-
            ahead path) is experimental and its behavior may change.

        `resolve` populates a source's *metadata* but, for a multi-file
        cloud source (zarr / ome-zarr / ndtiff / tiff-sequence / micromanager),
        leaves the bulk pixel data dehydrated -- each member file then recalls
        one-at-a-time, slowly, the first time a read touches it (the viewer
        scrubbing planes is the worst case). ``warm`` opts into pulling them all
        resident up front so later reads never stall.

        The recall happens **entirely server-side** (the server walks the source
        directory and reads each file to force the sync engine's recall); no
        pixels cross the wire, only progress. It is idempotent -- already-resident
        files are cheap local reads -- so a ``warm`` re-run after a cancel simply
        finishes the remainder. Only meaningful for multi-file sources; a
        single-file source returns immediately (resolve already recalled it), and
        a remote-url source (an object store, or a ``grpc://`` mirror) raises --
        nothing on the serving machine can be made resident.

        Args:
            source_id: The (already-resolved) source to warm.
            on_progress: Optional callback invoked with a ``WarmProgress``
                (files/bytes done vs total, current file name, elapsed) on each
                progress message. Called on the calling thread; keep it cheap.
            should_cancel: Optional predicate polled per message; when it returns
                True the client closes the stream -- which the server observes and
                stops the recall promptly -- and this raises
                `ResolveCancelled`. Files already recalled stay resident.

        Returns:
            The terminal ``WarmProgress`` snapshot (``files_done`` /
            ``bytes_done`` reflect what was made resident). ``files_total == 0``
            means the source was local and had nothing to warm -- how a client
            learns it is single-file. It never means "not applicable"; that
            case raises (biopb/biopb#1035).

        Raises:
            ResolveCancelled: if ``should_cancel`` asked to stop mid-warm.
            RuntimeError: if the server predates the ``warm`` action (too old for
                hydrate-ahead), or closes the stream without a terminal status.
            FlightServerError: if the source's url is remote. Warm it on the
                server that holds the data.
        """
        return self._catalog.warm(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )

    def add_source(
        self,
        url: str,
        *,
        source_type: str = "",
        on_progress: Optional[Callable[[AddSourceProgress], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> AddSourceResult:
        """Register a local path on the SERVER as a served source at runtime.

        This is the wire entrypoint behind the tensor-browser's drag-drop: it
        hands the server a filesystem path (or directory) that it interprets on
        *its own* filesystem, and the server routes it through the same claim ->
        adapter -> catalog pipeline the directory watcher uses. A dropped
        directory that is not itself a dataset is walked recursively and may
        register several sources, so the action streams progress and a final
        tally rather than returning a single source.

        The path must exist on the server. Because a dropped directory's walk has
        no known size up front, there is no percentage -- progress is a running
        count of sources registered so far.

        Args:
            url: Absolute path (or directory) on the server's filesystem.
            source_type: Explicit adapter type (e.g. ``"zarr"``, ``"ome-zarr"``);
                empty means auto-detect via the adapters' claim protocol.
            on_progress: Optional callback invoked with an ``AddSourceProgress``
                (count + current path) per source as it registers. Called on the
                calling thread; keep it cheap.
            should_cancel: Optional predicate polled per message; when it returns
                True the client closes the stream, which the server observes and
                stops discovery -- sources already registered stay registered.

        Returns:
            The terminal ``AddSourceResult``: ``added`` / ``already_present`` /
            ``refreshed`` / ``removed`` source_ids, and ``failed``
            ``(path, reason)`` pairs. A directory dropped above the large-scan
            threshold comes back as a ``failed`` entry, not a special flag.
            Registration wrote each source's catalog row, so anything beyond the
            ids is one `query_sources` away.

            Re-adding a path that is already registered REBUILDS it against the
            file as it is now -- that is what ``refreshed`` reports, and it is
            how a source picks up an in-place edit, since both its descriptor
            and the content_version that namespaces the chunk cache are sampled
            when its adapter is built. ``refreshed`` is a subset of
            ``already_present``, which keeps its original meaning. Registered
            sources under the path whose files are gone are deregistered and
            listed in ``removed``.

        Raises:
            flight.FlightServerError: whole-request failure (path not found /
                unreadable on the server, or the server declines the request).
            RuntimeError: the server predates the ``add_source`` action, or
                closed the stream without a terminal result.
        """
        return self._catalog.add_source(
            url,
            source_type=source_type,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    def remove_source(self, root_url: str) -> RemoveSourceResult:
        """Deregister a drag-dropped source branch on the SERVER at runtime.

        The narrow counterpart to `add_source`: it removes ONLY
        drag-dropped sources, which the server identifies by the ``dnd://``
        origin scheme on their catalog ``source_url``. ``root_url`` is such a
        branch root (a ``dnd://...`` value); every source at or under it is
        removed as a unit. A non-``dnd://`` ``root_url`` is refused by the server.

        Args:
            root_url: The ``dnd://`` branch root to remove (from the browser's
                dropped-root node).

        Returns:
            A ``RemoveSourceResult`` with ``removed`` (source_ids) and ``failed``
            (``AddSourceFailure`` whose ``path`` carries the source_id).

        Raises:
            flight.FlightServerError: the server refused the request (e.g. a
                non-``dnd://`` root, or removal not enabled).
            RuntimeError: the server predates the ``remove_source`` action, or
                returned no result.
        """
        return self._catalog.remove_source(root_url)

    # ---- ROI annotations ----

    def list_rois(self, array_id: str, set_name: str = "") -> RoiListResult:
        """Fetch a tensor's ROI annotations.

        There is no plane or bbox filter: a client hit-tests and re-renders
        from the resident set. Annotations are private data, gated by the
        tensor's source like its pixels, so they are not on the SQL surface.

        Args:
            array_id: Unversioned array_id of the tensor.
            set_name: Restrict to one layer, and the only way to read a
                reserved (``@``) set. Empty means the client-owned sets.

        Returns:
            ``RoiListResult`` with ``rois``, a ``truncated`` flag, and ``sets``
            -- every set on the tensor with its stored row count, whatever
            ``rois`` covers.

        Raises:
            flight.FlightUnavailableError: annotations disabled, or no metadata DB.
        """
        return self._catalog.list_rois(array_id, set_name)

    def put_rois(
        self,
        array_id: str,
        rois: Sequence[RoiAnnotation],
        *,
        check_rev: bool = False,
    ) -> RoiPutResult:
        """Create or update ROI annotations on a tensor, as one batch.

        Geometry is ``biopb.image.ROI`` in LEVEL-0 pixel coordinates -- a shape
        drawn on a downsampled level must be scaled up by the caller. Only the
        2-D vector arms are accepted (point / rectangle / ellipse / polygon /
        polyline -- the scribble stroke, whose ``width`` is geometry and widens
        its bounding box); a mask or mesh is refused, because instance
        segmentation belongs in a label tensor.

        An annotation with an empty ``roi_id`` is created (the server mints a
        uuid4); one that names an existing id is updated. The batch is applied
        in a single transaction.

        Args:
            array_id: Unversioned array_id every annotation belongs to.
            rois: The annotations to store.
            check_rev: Make each write conditional on ``rev`` matching what is
                stored. Mismatches come back in ``conflicts`` and are not
                applied; the rest of the batch still lands. Default is last
                writer wins.

        Returns:
            ``RoiPutResult`` with ``stored`` (server-assigned roi_id / rev /
            timestamps) and ``conflicts``.

        Raises:
            flight.FlightServerError: rejected geometry, a mismatched array_id,
                or the per-tensor cap would be breached.
        """
        return self._catalog.put_rois(array_id, rois, check_rev=check_rev)

    def delete_rois(
        self,
        array_id: str,
        roi_ids: Sequence[str] = (),
        set_name: str = "",
    ) -> RoiDeleteResult:
        """Delete ROI annotations.

        With ``roi_ids``, deletes exactly those. Without, deletes every
        annotation on the tensor -- narrowed to ``set_name`` when given, which
        is how a whole layer is dropped.

        Returns:
            ``RoiDeleteResult.deleted`` -- the ids actually removed.
        """
        return self._catalog.delete_rois(array_id, roi_ids, set_name)

    def prune_rois(self, unseen_days: int, *, apply: bool = False) -> RoiPruneResult:
        """Report, and with ``apply`` delete, annotations whose image is gone.

        An annotation is unseen when the catalog has not held its source for
        ``unseen_days`` (a row whose source never appeared counts from its
        creation). Reserved, server-owned sets are never pruned. Grouped per
        tensor in ``unseen``; ``deleted`` is the row count removed, 0 on a
        report. Requires the server-wide token: orphans have no source to
        authorize against.
        """
        return self._catalog.prune_rois(unseen_days, apply=apply)

    # ---- Reads (delegated to ChunkFetcher) ----

    def get_tensor(
        self,
        array_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> da.Array:
        """Get a lazy dask array for a tensor, addressed by its array_id.

        Args:
            array_id: Globally-unique tensor id (identity policy) -- e.g.
                ``"zarr_a3f2"`` for a single-tensor source or
                ``"aics_7f3/Image:0"`` for a multi-tensor source.
            slice_hint: Optional slice tuple to filter chunks
            scale_hint: Optional per-dimension integer downsampling factors
            reduction_method: Optional dynamic reduction method for scaled reads

        Returns:
            dask.array with lazy chunk loading

        Raises:
            ValueError: If source not found, tensor not found, or a bare
                multi-tensor source id is given without a within-source field
        """
        return self._fetcher.get_tensor(
            array_id, slice_hint, scale_hint, reduction_method
        )

    def get_tensor_pb(
        self,
        array_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> SerializedTensor:
        """Get a SerializedTensor protobuf for cross-process transfer.

        The planned read (a serialized Arrow ``FlightInfo``) plus this
        connection's location and token. It can be serialized to bytes and
        broadcast to worker processes, where each worker calls
        ``tensor_from_pb()`` to reconstruct the lazy dask array.

        Args:
            array_id: Globally-unique tensor id (identity policy) -- e.g.
                ``"zarr_a3f2"`` or ``"aics_7f3/Image:0"``.
            slice_hint: Optional slice tuple to filter chunks
            scale_hint: Optional per-dimension integer downsampling factors
            reduction_method: Optional dynamic reduction method for scaled reads

        Returns:
            SerializedTensor protobuf object
        """
        return self._fetcher.get_tensor_pb(
            array_id, slice_hint, scale_hint, reduction_method
        )

    @staticmethod
    def descriptor_from_pb(pb: SerializedTensor) -> TensorDescriptor:
        """The resolved descriptor a SerializedTensor's plan names, without
        building the array: shape, dtype, labels, for a reader that only
        needs to describe what it was handed."""
        info = flight.FlightInfo.deserialize(pb.flight_info)
        return TensorDescriptor.FromString(info.descriptor.command)

    @staticmethod
    def tensor_from_pb(
        pb: SerializedTensor,
        cache_bytes: Optional[int] = None,
    ) -> da.Array:
        """The lazy dask array a SerializedTensor describes.

        The one consumer-side helper: the handle is a FlightInfo plus where and
        as whom to read it, so this decodes the plan and builds the same
        chunk-fetching array ``get_tensor`` builds on a live connection. Each
        worker process maintains its own connection pool and LRU cache keyed
        by (location, auth_token).

        A handle with no endpoints -- a source declared before its chunks
        existed -- is planned here with a GetFlightInfo on the embedded
        descriptor; the crop the producer asked for is kept from the handle.

        Args:
            pb: SerializedTensor protobuf object
            cache_bytes: Maximum bytes for the chunk cache. ``None`` (the default)
                resolves ``BIOPB_TENSOR_CACHE_LIMIT`` (or 1 GB); a value passed
                here overrides the env. Only effective for the first tensor
                created in a process for a given (location, auth_token) pair.

        Returns:
            dask.array with lazy chunk loading
        """
        if cache_bytes is None:
            cache_bytes = _default_cache_bytes()
        token = pb.auth_token or None
        info = flight.FlightInfo.deserialize(pb.flight_info)
        requested = _requested_slice(info)
        if not info.endpoints:
            logger.debug("tensor_from_pb: no endpoints, calling GetFlightInfo")
            info = _refetch_flight_info(
                TensorDescriptor.FromString(info.descriptor.command), pb.location, token
            )
        return _dask_from_flight_info(
            info,
            pb.location,
            token,
            cache_bytes,
            resolve_tls_trust(pb.location),
            requested,
        )

    # ====================
    # Upload API (EXPERIMENTAL) -- thin delegators onto the UploadSession
    # collaborator (see biopb.tensor._upload); #278 item C.
    # ====================

    def create_tensor(
        self,
        source_name: str,
        template: Any,
        *,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> TensorDescriptor:
        """Declare a single-tensor source to fill: the first half of an upload.

        Note:
            Experimental. The upload / writable-source API (tensor creation,
            chunk upload, and upload-status polling) is experimental and may
            change.

        Declare, then fill. The returned descriptor is the server's echo --
        ``array_id``, ``shape``, ``dtype``, ``chunk_shape``, ``dim_labels`` --
        and is what ``upload_array``, ``upload_chunk`` and ``finish_upload``
        take. A name is single-use for the life of the server: a second create
        under a name that exists -- pending, finished or discarded -- is
        refused. ``finish_upload`` is what marks the upload complete.

        Args:
            source_name: "cache:name" → cache-backed; "ome_zarr:name" →
                zarr-backed; "cache:" or "ome_zarr:" → server-generated name
            template: Anything with ``.shape`` and ``.dtype`` -- the array to be
                uploaded, or one shaped like it. A dask array also supplies the
                chunk grid (its chunk size per axis).
            chunk_shape: The upload grid, overriding the template's. Required
                to get anything but one chunk from a non-dask template.
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict

        Returns:
            The new source's descriptor.

        Raises:
            pyarrow.flight.FlightServerError: the name is already taken.
        """
        return self._upload.create_tensor(
            source_name,
            template,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
            ome_metadata=ome_metadata,
        )

    def upload_array(self, desc: TensorDescriptor, arr: Any) -> Dict[str, Any]:
        """Fill a declared tensor with an array, and seal it.

        Note:
            Experimental. The upload / writable-source API (tensor creation,
            chunk upload, and upload-status polling) is experimental and may
            change.

        *arr* must match the descriptor's shape and dtype; it is rechunked onto
        the descriptor's chunk grid, every block is sent as one chunk, and the
        source is finished. A numpy array is accepted and chunked on the grid.

        Args:
            desc: The descriptor ``create_tensor`` returned
            arr: The array to upload (dask or numpy)

        Returns:
            The sealed upload status, as ``get_upload_status`` reports it.

        Raises:
            ValueError: *arr* does not match the declared shape or dtype.
            UploadRefused: the upload is over -- sealed or discarded.
        """
        return self._upload.upload_array(desc, arr)

    def upload_chunk(
        self,
        desc: TensorDescriptor,
        bounds: ChunkBounds,
        data: np.ndarray,
    ) -> None:
        """Upload one chunk of a declared tensor.

        Note:
            Experimental. The upload / writable-source API (tensor creation,
            chunk upload, and upload-status polling) is experimental and may
            change.

        The manual half of ``upload_array``: a caller writing chunks itself
        calls this per chunk and ``finish_upload`` when done.

        Args:
            desc: The descriptor ``create_tensor`` returned
            bounds: Chunk start/stop coordinates
            data: Numpy array with chunk data

        Raises:
            UploadRefused: the upload is over -- sealed or discarded.
        """
        self._upload.upload_chunk(desc, bounds, data)

    def finish_upload(self, desc: TensorDescriptor) -> Dict[str, Any]:
        """Seal an upload: the source is complete and takes no further chunks.

        Note:
            Experimental. The upload / writable-source API (tensor creation,
            chunk upload, and upload-status polling) is experimental and may
            change.

        The only route to READY, which is the state a consumer waiting on this
        result polls for. ``upload_array``, which writes every chunk itself,
        calls it for you.

        Args:
            desc: The descriptor ``create_tensor`` returned

        Returns:
            The sealed upload status, as ``get_upload_status`` reports it.

        Raises:
            UploadRefused: the upload was discarded.
        """
        return self._upload.finish_upload(desc)

    def close(self):
        """Close the Flight client."""
        logger.info("Closing Flight client")
        self._client.close()

    def health_check(self) -> Dict[str, Any]:
        """Check server health status via Flight action.

        Returns:
            Dictionary with health status information:

            - `status`: "SERVING" or other status string. Note: with
                progressive discovery, SERVING means "up and serving the
                possibly-still-populating catalog," not "catalog complete" --
                use the freshness fields below to tell whether indexing is
                still in progress.
            - `source_count`: Number of registered sources
            - `metadata_db_enabled`: Whether the server offers a catalog.
                False means it serves its sources by id alone and every
                catalog surface (list_sources, query_sources, resolve,
                annotations) refuses
            - `writable`: Whether server accepts uploads
            - `uptime_seconds`: Server uptime in seconds
            - `full_scan_in_progress`: Whether a full catalog rescan is
                running now (absent on older servers)
            - `last_full_scan_finished_at`: Epoch seconds when a full scan
                last succeeded, or None until the first one does (absent on
                older servers)

        Raises:
            FlightError: If server is unreachable or action fails
        """
        action = flight.Action("health", b"")
        results = self._client.do_action(action, options=self._call_options)
        for result in results:
            return json.loads(result.body.to_pybytes())
        return {"status": "UNKNOWN"}

    def cache_stats(self) -> Dict[str, Any]:
        """Fetch server-side cache statistics via Flight action.

        Returns:
            Dictionary of CacheStats fields: total_entries, total_bytes,
            max_entries, max_bytes, hits, misses, evictions, pending_waits,
            ref_held_evictions_skipped, oversized_skips, and (file backend)
            per-pool stats under "pool_stats".

        Raises:
            FlightError: If server is unreachable or action fails
        """
        action = flight.Action("cache_stats", b"")
        results = self._client.do_action(action, options=self._call_options)
        for result in results:
            return json.loads(result.body.to_pybytes())
        return {}

    def get_upload_status(self, source_id: str) -> Dict[str, Any]:
        """Get upload status for a writable source.

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.

        Args:
            source_id: The ``array_id`` of the descriptor ``create_tensor`` returned

        Returns:
            Dictionary with source_id, state, expected_chunks, and uploaded_chunks.
        """
        return self._catalog.get_upload_status(source_id)

    def cache_info(self) -> Dict:
        """Return cache statistics for this connection.

        The ``size_bytes``/``max_bytes``/``item_count`` fields describe the
        **strong** copy cache (cachey) -- ``do_get`` results and over-budget
        copies, the only chunks that cost client RAM. mmap views live in the weak
        view cache, which costs no RAM and has no byte budget; ``view_items``
        reports how many are currently live (a lower bound -- entries self-prune
        as their arrays are collected).

        Returns:
            Dictionary with copy-cache size/item_count plus ``view_items``.
        """
        key = (self._location, self._token)
        wvd = _VIEW_CACHE.get(key)
        view_items = len(wvd) if wvd is not None else 0

        # Describe the strong copy cache only; the weak view cache is reported
        # separately via view_items.
        if key not in _CACHE_POOL:
            # No copy cache allocated yet. Report the resolved size so a
            # not-yet-created copy cache truthfully shows what it would allow.
            size_bytes, max_bytes, item_count = (
                0,
                _resolve_cache_bytes(self._location, self._cache_bytes),
                0,
            )
        else:
            cache = _CACHE_POOL[key]  # Cache, or None when pinned off
            if cache is None:
                # Pinned off by configure_cache(): report max_bytes == 0 truthfully.
                size_bytes, max_bytes, item_count = 0, 0, 0
            else:
                size_bytes, max_bytes, item_count = (
                    cache.total_bytes,
                    cache.available_bytes,
                    len(cache.data),
                )
        return {
            "size_bytes": size_bytes,
            "max_bytes": max_bytes,
            "item_count": item_count,
            "view_items": view_items,
        }

    def cache_clear(self):
        """Clear both the strong copy cache and the weak view cache for this
        connection namespace (the latter drops only weak references)."""
        key = (self._location, self._token)
        cache = _CACHE_POOL.get(key)
        if cache is not None:
            cache.clear()
        _clear_view_cache(self._location, self._token)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
