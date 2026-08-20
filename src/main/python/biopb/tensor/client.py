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
from biopb.tensor._pool import (
    _CACHE_POOL,
    _VIEW_CACHE,
    _build_call_options,
    _build_dask_array_from_chunk_map,
    _chunk_map_from_endpoints,
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
    _extract_schema_metadata as _extract_schema_metadata,
    _fetch_endpoints_via_get_flight_info,
    _parse_version as _parse_version,
    _request_crop_slices,
    _split_array_id as _split_array_id,
    _TensorContext,
)
from biopb.tensor._tls import resolve_tls_trust
from biopb.tensor._upload import UploadSession
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


def _make_debug_serialized_tensor(
    arr: da.Array, array_id: str = "debug"
) -> SerializedTensor:
    """Create a SerializedTensor with debug_pickled_array for testing.

    Eagerly computes the array and pickles it, bypassing Flight server.
    Preserves original chunk structure for testing chunk-related behavior.
    Populates inferable tensor_descriptor fields.

    Args:
        arr: Dask array to serialize
        array_id: Optional array identifier

    Returns:
        SerializedTensor with debug_pickled_array populated
    """
    import pickle

    # Eager compute
    np_arr = arr.compute()

    # Rechunk to original chunk structure (preserves chunk boundaries for testing)
    computed_da = da.from_array(np_arr, chunks=arr.chunksize)

    descriptor = TensorDescriptor(
        array_id=array_id,
        shape=list(arr.shape),
        dtype=np.dtype(arr.dtype).str,
        chunk_shape=list(arr.chunksize),
    )

    return SerializedTensor(
        tensor_descriptor=descriptor,
        debug_pickled_array=pickle.dumps(computed_da),
    )


class TensorFlightClient:
    """Client for accessing tensors from a TensorFlightServer.

    This client provides lazy, cached access to multi-dimensional arrays
    stored in a Flight server, with support for multifield acquisitions
    where tensors within a source have different shapes.

    Example:
        ```python
        client = TensorFlightClient('grpc://localhost:8815')

        # List data sources (each may contain multiple tensors)
        sources = client.list_sources()

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
        # The connection + the two catalog caches live in one shared _ClientState.
        # The collaborators (#278 item C) read/write it; this facade exposes the
        # caches back-compatibly via the _sources/_descriptors properties below.
        self._state = _ClientState(
            client=self._client,
            call_options=self._call_options,
            location=self._location,
            token=self._token,
            cache_bytes=self._cache_bytes,
            tls_trust=self._tls_trust,
        )
        self._catalog = CatalogClient(self._state)
        self._fetcher = ChunkFetcher(self._state, self._catalog)
        self._upload = UploadSession(self._client, self._call_options)

    # The catalog caches live on the shared _ClientState; expose them here so a
    # caller's reads, in-place mutation, AND reassignment (client._sources = {})
    # all reach the one shared dict the collaborators use (#278 item C).
    @property
    def _sources(self) -> Dict[str, DataSourceDescriptor]:
        return self._state.sources

    @_sources.setter
    def _sources(self, value: Dict[str, DataSourceDescriptor]) -> None:
        self._state.sources = value

    @property
    def _descriptors(self) -> Dict[str, TensorDescriptor]:
        return self._state.descriptors

    @_descriptors.setter
    def _descriptors(self, value: Dict[str, TensorDescriptor]) -> None:
        self._state.descriptors = value

    # ---- Catalog / metadata / source lifecycle (delegated to CatalogClient) ----

    def list_sources(self) -> Dict[str, DataSourceDescriptor]:
        """List available data sources.

        Returns:
            Dictionary mapping source_id to DataSourceDescriptor.
            Each DataSourceDescriptor.tensors contains TensorDescriptor info
            with shape/dtype for all tensors in that source.

        Note:
            Results may be truncated if server has max_list_flights_results configured.
            Check schema metadata for truncation info (truncated=True indicates
            more sources exist on server than were returned).
        """
        return self._catalog.list_sources()

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
        server fills on every ``GetFlightInfo`` (issue #31), so this reads the
        descriptor a prior `get_tensor` already cached -- no extra RPC when
        it is cached, and it never requests the opt-in ``metadata_json`` field on
        that same descriptor. (Contrast `get_source_metadata`, which forces
        ``with_metadata`` to ship the whole OME tree; do not dig physical sizes
        out of that -- this is the compact projection meant for display scale.)

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
    ) -> TensorDescriptor:
        """Fetch one tensor's ``TensorDescriptor`` by its globally-unique array_id.

        A tensor is identified by its ``array_id`` alone (see the tensor identity
        policy at the top of ``proto/biopb/tensor/descriptor.proto``), so this
        takes that one identifier rather than a ``(source_id, tensor_id)`` pair.
        Works even when the source is beyond the (truncatable) ``list_sources()``
        cap. Every call fetches -- the client caches only the *structural* part of
        the answer (shape/dtype/dim_labels/chunk_shape plus physical scale) for
        its own addressing, never ``metadata_json`` or ``pyramid``, so what you
        get back always reflects the masks you passed. Passing a bare
        ``source_id`` (single-tensor source, or to anchor on a multi-tensor
        source's default/first tensor) is accepted. To enumerate ALL
        tensors/scenes of a source, use ``list_sources()[source_id].tensors``
        -- NOT this method.

        This is a cheap probe -- it does NOT resolve. On an unresolved (cloud /
        synced-folder) source it raises an error pointing at `resolve`,
        never triggering a download. Call `resolve` first to read such a
        source.

        The ``with_*`` flags are the ``GetFlightInfo`` response field masks
        (biopb/biopb#563). This is a *describe* call -- the stable per-tensor
        facts, not a read -- so it defaults to returning shape/dtype/dim_labels/
        chunk_shape, the resolution **pyramid**, and physical_scale, while
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
        )

    def resolve(
        self,
        source_id: str,
        *,
        on_progress: Optional[Callable[[ResolveProgress], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> DataSourceDescriptor:
        """Resolve an unresolved source and return its full ``DataSourceDescriptor``.

        Note:
            Experimental. Cloud / remote source support (unresolved sources,
            resolve, and `warm`) is experimental and its behavior may change.

        An *unresolved* source is catalogued by URL only -- its shape/dtype/field
        list are unknown until first access (it lists with ``data_resident`` False
        and an empty ``list_sources()[source_id].tensors``). The canonical case is
        a cloud / synced-folder ("Files-On-Demand") source.

        Resolving asks the server to hydrate it. For a dehydrated placeholder this
        **downloads the whole file** -- a recall that can take minutes, consume
        local disk, and fail when offline -- then reads its real shape, dtype, and
        field list. This is the heavyweight, *consenting* operation that catalog
        browsing (`list_sources` / `query_sources`) deliberately
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
            The full ``DataSourceDescriptor`` with every tensor/field enumerated
            -- the complete field set in one call, regardless of catalog size.

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
        single-file source returns immediately (resolve already recalled it).

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
            ``bytes_done`` reflect what was made resident; on a no-op source
            ``files_total == 0``).

        Raises:
            ResolveCancelled: if ``should_cancel`` asked to stop mid-warm.
            RuntimeError: if the server predates the ``warm`` action (too old for
                hydrate-ahead), or closes the stream without a terminal status.
        """
        return self._catalog.warm(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )

    def add_source(
        self,
        url: str,
        *,
        source_type: str = "",
        dim_labels: Optional[List[str]] = None,
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
        tally rather than returning a single descriptor.

        The path must exist on the server. Because a dropped directory's walk has
        no known size up front, there is no percentage -- progress is a running
        count of sources registered so far.

        Args:
            url: Absolute path (or directory) on the server's filesystem.
            source_type: Explicit adapter type (e.g. ``"zarr"``, ``"ome-zarr"``);
                empty means auto-detect via the adapters' claim protocol.
            dim_labels: Optional dimension labels for the registered tensor(s).
            on_progress: Optional callback invoked with an ``AddSourceProgress``
                (count + current path + last descriptor) per source as it
                registers. Called on the calling thread; keep it cheap.
            should_cancel: Optional predicate polled per message; when it returns
                True the client closes the stream, which the server observes and
                stops discovery -- sources already registered stay registered.

        Returns:
            The terminal ``AddSourceResult`` (``added`` descriptors,
            ``already_present`` source_ids, ``failed`` ``(path, reason)`` pairs).
            A directory dropped above the large-scan threshold comes back as a
            ``failed`` entry, not a special flag.

        Raises:
            flight.FlightServerError: whole-request failure (path not found /
                unreadable on the server, or the server declines the request).
            RuntimeError: the server predates the ``add_source`` action, or
                closed the stream without a terminal result.
        """
        return self._catalog.add_source(
            url,
            source_type=source_type,
            dim_labels=dim_labels,
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

    # ---- Reads (delegated to ChunkFetcher) ----

    def _get_tensor_context(
        self,
        array_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> _TensorContext:
        """See :meth:`ChunkFetcher._get_tensor_context`."""
        return self._fetcher._get_tensor_context(
            array_id, slice_hint, scale_hint, reduction_method
        )

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

        Returns a protobuf containing connection info and chunk tickets
        for lazy reconstruction. The protobuf can be serialized to bytes
        and broadcast to worker processes, where each worker can call
        tensor_from_pb() to reconstruct a lazy dask array.

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

    def _build_dask_array(
        self,
        desc: TensorDescriptor,
        chunks: List[bytes],
        chunk_bounds: List[ChunkBounds],
        schema_metadata: Optional[Dict[str, str]] = None,
    ) -> da.Array:
        """See :meth:`ChunkFetcher._build_dask_array`."""
        return self._fetcher._build_dask_array(
            desc, chunks, chunk_bounds, schema_metadata
        )

    @staticmethod
    def tensor_from_pb(
        pb: SerializedTensor,
        cache_bytes: Optional[int] = None,
    ) -> da.Array:
        """Reconstruct a lazy dask array from SerializedTensor protobuf.

        Creates a dask array that fetches chunks from the Flight server
        independently. Each worker process maintains its own connection
        pool and LRU cache keyed by (location, auth_token).

        If endpoints field is empty, calls GetFlightInfo on the server
        to rebuild the endpoint list.

        If debug_pickled_array is populated, unpickles directly (bypasses server).

        Args:
            pb: SerializedTensor protobuf object
            cache_bytes: Maximum bytes for the chunk cache. ``None`` (the default)
                resolves ``BIOPB_TENSOR_CACHE_LIMIT`` (or 1 GB); a value passed
                here overrides the env. Only effective for the first tensor
                created in a process for a given (location, auth_token) pair.

        Returns:
            dask.array with lazy chunk loading
        """
        import pickle

        if cache_bytes is None:
            cache_bytes = _default_cache_bytes()

        # Debug path: unpickle directly if debug_pickled_array is present
        if pb.debug_pickled_array:
            return pickle.loads(pb.debug_pickled_array)

        descriptor = pb.tensor_descriptor
        shape = tuple(descriptor.shape)
        dtype = np.dtype(descriptor.dtype)

        # Parse endpoints - if empty, fetch from GetFlightInfo
        chunks = []
        chunk_bounds_list = []

        if pb.endpoints:
            # Use serialized endpoints directly
            for ep in pb.endpoints:
                chunks.append(ep.ticket.chunk_id)
                chunk_bounds_list.append(ep.chunk_bounds)
        else:
            # Endpoints not provided - call GetFlightInfo to rebuild
            logger.debug("tensor_from_pb: endpoints empty, calling GetFlightInfo")
            chunks, chunk_bounds_list = _fetch_endpoints_via_get_flight_info(pb)

        # Build the block-index -> (chunk_id, bounds) map + grid shape for lazy
        # chunk fetching (shared with ChunkFetcher._build_dask_array).
        chunk_map, grid_shape = _chunk_map_from_endpoints(
            chunks, chunk_bounds_list, shape
        )

        # Extract schema_metadata from pb for SHM transfer
        schema_metadata = dict(pb.schema_metadata) if pb.schema_metadata else None

        dask_arr = _build_dask_array_from_chunk_map(
            chunk_map,
            grid_shape,
            shape,
            dtype,
            pb.location,
            pb.auth_token if pb.auth_token else None,
            cache_bytes,
            schema_metadata,
            resolve_tls_trust(pb.location),
        )

        # Crop to the originally requested region if original_slice_hint present
        if pb.HasField("original_slice_hint") and descriptor.HasField("slice_hint"):
            dask_arr = dask_arr[
                _request_crop_slices(
                    len(descriptor.shape),
                    pb.original_slice_hint,
                    descriptor.slice_hint,
                    list(descriptor.scale_hint) if descriptor.scale_hint else None,
                )
            ]

        return dask_arr

    # ====================
    # Upload API (EXPERIMENTAL) -- thin delegators onto the UploadSession
    # collaborator (see biopb.tensor._upload); #278 item C.
    # ====================

    def upload_array(
        self,
        arr: da.Array,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Upload dask array to server.

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.

        Args:
            arr: Dask array to upload
            source_name: Source identifier format:
                - "cache:my-name" → cache-backed (ephemeral)
                - "cache:" → cache-backed with server-generated name
                - "ome_zarr:my-name" → zarr-backed (persistent)
                - "ome_zarr:" → zarr-backed with server-generated name
            chunk_shape: Override chunk shape. If None, uses arr.chunksize with
                         automatic rechunking if chunks are non-uniform.
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict

        Returns:
            source_id of created source (e.g., "cache_abc123" or "ome_zarr_def456")
        """
        return self._upload.upload_array(
            arr, source_name, chunk_shape, dim_labels, ome_metadata
        )

    def upload_zarr(
        self,
        zarr_path: str,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Upload local zarr to server.

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.

        Args:
            zarr_path: Path to local zarr directory
            source_name: Source identifier format:
                - "cache:my-name" → cache-backed (ephemeral)
                - "cache:" → cache-backed with server-generated name
                - "ome_zarr:my-name" → zarr-backed (persistent)
                - "ome_zarr:" → zarr-backed with server-generated name
            chunk_shape: Override chunk shape. If None, uses zarr's chunk shape.
            dim_labels: Optional dimension labels (read from zarr if not provided)
            ome_metadata: Optional OME metadata (read from zarr if not provided)

        Returns:
            source_id of created source (e.g., "cache_abc123" or "ome_zarr_def456")
        """
        return self._upload.upload_zarr(
            zarr_path, source_name, chunk_shape, dim_labels, ome_metadata
        )

    def create_source(
        self,
        source_name: str,
        shape: Sequence[int],
        dtype: str,
        chunk_shape: Sequence[int],
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Create source on server (internal).

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.

        Args:
            source_name: "cache:name" → cache-backed; "ome_zarr:name" → zarr-backed
                         "cache:" or "ome_zarr:" → server-generated name
            shape: Array shape
            dtype: Data type string (numpy format)
            chunk_shape: Chunk size per dimension
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict

        Returns:
            source_id assigned by server
        """
        return self._upload.create_source(
            source_name, shape, dtype, chunk_shape, dim_labels, ome_metadata
        )

    def upload_chunk(
        self,
        source_id: str,
        bounds: ChunkBounds,
        data: np.ndarray,
    ) -> None:
        """Upload single chunk (internal).

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.

        Args:
            source_id: Source identifier
            bounds: Chunk start/stop coordinates
            data: Numpy array with chunk data
        """
        self._upload.upload_chunk(source_id, bounds, data)

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
            - `metadata_db_enabled`: Whether metadata database is enabled
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
            source_id: Source identifier returned by create_source()

        Returns:
            Dictionary with source_id, state, expected_chunks, and uploaded_chunks.
        """
        return self._upload.get_upload_status(source_id)

    def get_upload_status_pb(self, pb: SerializedTensor) -> Dict[str, Any]:
        """Get upload status for a registration-first SerializedTensor handle.

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.

        This helper is intended for cache-backed handles returned before upload
        completion, where tensor_descriptor.array_id is the source identifier.

        Args:
            pb: SerializedTensor handle returned by a registration-first flow.

        Returns:
            Dictionary with source_id, state, expected_chunks, and uploaded_chunks.
        """
        return self._upload.get_upload_status_pb(pb)

    def wait_for_upload_ready(
        self,
        source_id: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll upload status until the source reports READY.

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.

        Applies only to sources created by ``create_source()`` /
        ``upload_array()``. A source the server tracks no upload for reports
        UNKNOWN, and that is rejected on the first poll rather than waited out:
        either the source was never an upload target (a catalog source, on disk
        or in the cloud, has no upload to wait for), or its record was dropped
        when the source was removed or the server restarted. Neither reading
        resolves by polling.

        Args:
            source_id: Source identifier returned by create_source().
            timeout_seconds: Maximum time to wait before timing out.
            poll_interval_seconds: Delay between status checks.

        Returns:
            Final upload status dictionary when READY.

        Raises:
            ValueError: If the server tracks no upload for the source (UNKNOWN).
            TimeoutError: If the upload does not reach READY within the timeout.
            RuntimeError: If the upload reports FAILED.
        """
        return self._upload.wait_for_upload_ready(
            source_id, timeout_seconds, poll_interval_seconds
        )

    def wait_for_upload_ready_pb(
        self,
        pb: SerializedTensor,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll upload status until a registration-first SerializedTensor is READY.

        Note:
            Experimental. The upload / writable-source API (source creation, chunk
            upload, and upload-status polling) is experimental and may change.
        """
        return self._upload.wait_for_upload_ready_pb(
            pb, timeout_seconds, poll_interval_seconds
        )

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
