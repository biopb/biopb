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
# Re-exported here (redundant `as` aliases mark intentional re-exports) so
# ``biopb.tensor.client.<name>`` stays a stable import surface for existing
# callers: biopb-mcp's ``configure_cache`` dask worker plugin, the cachefile /
# connection-pool tests, and the benchmarks. A few (``_build_dask_array_from_
# chunk_map``, ``_CACHE_POOL``, ``_resolve_cache_bytes``) are also used directly
# by TensorFlightClient below.
from biopb.tensor._pool import (
    _CACHE_POOL as _CACHE_POOL,
    _CACHEFILE_SUPPORTED_FORMAT as _CACHEFILE_SUPPORTED_FORMAT,
    _CALL_OPTS_POOL as _CALL_OPTS_POOL,
    _CONNECTION_REGISTRY as _CONNECTION_REGISTRY,
    _POOL_LOCK as _POOL_LOCK,
    _REGISTRY_LOCK as _REGISTRY_LOCK,
    _THREAD_LOCAL as _THREAD_LOCAL,
    _array_from_unified_batch as _array_from_unified_batch,
    _build_dask_array_from_chunk_map as _build_dask_array_from_chunk_map,
    _cache_local_enabled as _cache_local_enabled,
    _cachefile_support as _cachefile_support,
    _cachefile_support_lock as _cachefile_support_lock,
    _cachefile_supported as _cachefile_supported,
    _cleanup_connection_pool as _cleanup_connection_pool,
    _evict_dead_threads as _evict_dead_threads,
    _fetch_chunk_block as _fetch_chunk_block,
    _fetch_chunk_distributed as _fetch_chunk_distributed,
    _get_shared_cache as _get_shared_cache,
    _get_shared_call_options as _get_shared_call_options,
    _get_thread_client as _get_thread_client,
    _get_worker_resources as _get_worker_resources,
    _is_cachefile_disabled_by_env as _is_cachefile_disabled_by_env,
    _is_localhost_location as _is_localhost_location,
    _regular_grid_chunks as _regular_grid_chunks,
    _resolve_cache_bytes as _resolve_cache_bytes,
    _set_cachefile_supported as _set_cachefile_supported,
    _should_try_cachefile as _should_try_cachefile,
    _try_cachefile_transfer as _try_cachefile_transfer,
    configure_cache as configure_cache,
)
from biopb.tensor._session import (
    CatalogClient,
    ChunkFetcher,
    ResolveCancelled as ResolveCancelled,
    _check_wire_protocol as _check_wire_protocol,
    _ClientState,
    _descriptor_key,
    _extract_schema_metadata as _extract_schema_metadata,
    _fetch_endpoints_via_get_flight_info,
    _parse_version as _parse_version,
    _request_crop_slices,
    _resolve_array_id as _resolve_array_id,
    _TensorContext,
    _unresolved_source_error as _unresolved_source_error,
)
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


def make_debug_serialized_tensor(
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

    Usage:
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
        cache_bytes: int = 1_000_000_000,  # 1GB default
        token: Optional[str] = None,
    ):
        """Initialize the Flight client.

        Args:
            location: Flight server location
            cache_bytes: Maximum bytes for chunk cache (default 1GB)
            token: Bearer token for server authentication.  ``None`` disables auth.
        """
        logger.info(
            f"Connecting to Flight server at {location}, cache={cache_bytes}B, auth={token is not None}"
        )
        # Normalize location for Arrow Flight (grpcs:// -> grpc+tls://)
        normalized = _normalize_location(location)
        # Pickle-safe connection parameters (callers read client._client etc.)
        self._location = normalized
        self._token = token
        self._cache_bytes = cache_bytes
        self._client = flight.FlightClient(normalized)
        self._call_options = (
            flight.FlightCallOptions(
                headers=[(b"authorization", f"Bearer {token}".encode())]
            )
            if token
            else flight.FlightCallOptions()
        )
        # The connection + the two catalog caches live in one shared _ClientState.
        # The collaborators (#278 item C) read/write it; this facade exposes the
        # caches back-compatibly via the _sources/_descriptors properties below.
        self._state = _ClientState(
            client=self._client,
            call_options=self._call_options,
            location=self._location,
            token=self._token,
            cache_bytes=self._cache_bytes,
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
    def _descriptors(self) -> Dict[Tuple[str, str], TensorDescriptor]:
        return self._state.descriptors

    @_descriptors.setter
    def _descriptors(self, value: Dict[Tuple[str, str], TensorDescriptor]) -> None:
        self._state.descriptors = value

    @staticmethod
    def _descriptor_key(source_id: str, array_id: str) -> Tuple[str, str]:
        """Composite source-unique descriptor-cache key. See :func:`_descriptor_key`."""
        return _descriptor_key(source_id, array_id)

    # ---- Catalog / metadata / source lifecycle (delegated to CatalogClient) ----

    def list_sources(self) -> Dict[str, DataSourceDescriptor]:
        """List available data sources. See :meth:`CatalogClient.list_sources`."""
        return self._catalog.list_sources()

    def query_sources(self, sql: str, *, format: str = "arrow") -> Any:
        """Query the server catalog (DuckDB). See :meth:`CatalogClient.query_sources`."""
        return self._catalog.query_sources(sql, format=format)

    @staticmethod
    def _format_query_result(table, format):
        """Coerce a query result to the requested format. See :meth:`CatalogClient._format_query_result`."""
        return CatalogClient._format_query_result(table, format)

    def get_source_metadata(self, source_id: str) -> dict:
        """Source-level OME/vendor metadata. See :meth:`CatalogClient.get_source_metadata`."""
        return self._catalog.get_source_metadata(source_id)

    def get_physical_scale(
        self,
        array_id: Optional[str] = None,
        tensor_id: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
    ) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dimension physical size + unit. See :meth:`CatalogClient.get_physical_scale`."""
        return self._catalog.get_physical_scale(
            array_id, tensor_id, source_id=source_id
        )

    def _fetch_tensor_descriptor(
        self, source_id: str, tensor_id: Optional[str] = None
    ) -> TensorDescriptor:
        """See :meth:`CatalogClient._fetch_tensor_descriptor`."""
        return self._catalog._fetch_tensor_descriptor(source_id, tensor_id)

    def get_descriptor(self, array_id: str) -> TensorDescriptor:
        """Fetch one tensor's descriptor by array_id. See :meth:`CatalogClient.get_descriptor`."""
        return self._catalog.get_descriptor(array_id)

    def resolve(
        self,
        source_id: str,
        *,
        on_progress: Optional[Callable[[ResolveProgress], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> DataSourceDescriptor:
        """Resolve an unresolved (cloud) source. See :meth:`CatalogClient.resolve`."""
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
        """Hydrate-ahead a resolved multi-file source. See :meth:`CatalogClient.warm`."""
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
        """Register a server-side path as a source. See :meth:`CatalogClient.add_source`."""
        return self._catalog.add_source(
            url,
            source_type=source_type,
            dim_labels=dim_labels,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    def remove_source(self, root_url: str) -> RemoveSourceResult:
        """Deregister a drag-dropped source branch. See :meth:`CatalogClient.remove_source`."""
        return self._catalog.remove_source(root_url)

    def get_source(
        self, source_id: str, tensor_id: Optional[str] = None
    ) -> DataSourceDescriptor:
        """DEPRECATED single-tensor probe. See :meth:`CatalogClient.get_source`."""
        return self._catalog.get_source(source_id, tensor_id)

    # ---- Reads (delegated to ChunkFetcher) ----

    def _get_tensor_context(
        self,
        source_id: str,
        tensor_id: Optional[str] = None,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> _TensorContext:
        """See :meth:`ChunkFetcher._get_tensor_context`."""
        return self._fetcher._get_tensor_context(
            source_id, tensor_id, slice_hint, scale_hint, reduction_method
        )

    def get_tensor(
        self,
        array_id: Optional[str] = None,
        tensor_id: Optional[str] = None,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
    ) -> da.Array:
        """Lazy dask array for a tensor. See :meth:`ChunkFetcher.get_tensor`."""
        return self._fetcher.get_tensor(
            array_id,
            tensor_id,
            slice_hint,
            scale_hint,
            reduction_method,
            source_id=source_id,
        )

    def get_tensor_pb(
        self,
        array_id: Optional[str] = None,
        tensor_id: Optional[str] = None,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
    ) -> SerializedTensor:
        """SerializedTensor handle for cross-process reads. See :meth:`ChunkFetcher.get_tensor_pb`."""
        return self._fetcher.get_tensor_pb(
            array_id,
            tensor_id,
            slice_hint,
            scale_hint,
            reduction_method,
            source_id=source_id,
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
        cache_bytes: int = 1_000_000_000,
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
            cache_bytes: Maximum bytes for chunk cache (default 1GB).
                Only effective for the first tensor created in a process
                for a given (location, auth_token) pair.

        Returns:
            dask.array with lazy chunk loading
        """
        import pickle

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

        # Build chunk map
        chunk_map = {}
        axis_starts = [
            sorted({int(bounds.start[axis]) for bounds in chunk_bounds_list})
            for axis in range(len(shape))
        ]
        axis_index_maps = [
            {start: index for index, start in enumerate(starts)}
            for starts in axis_starts
        ]
        for chunk_id, bounds in zip(chunks, chunk_bounds_list):
            chunk_idx = tuple(
                axis_index_maps[d][int(bounds.start[d])] for d in range(len(shape))
            )
            chunk_map[chunk_idx] = (chunk_id, bounds)

        # Build dask array with lazy chunk fetching
        ndim = len(shape)
        grid_shape = tuple(max(idx[d] + 1 for idx in chunk_map) for d in range(ndim))

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
    # Upload API -- thin delegators onto the UploadSession collaborator
    # (see biopb.tensor._upload); #278 item C.
    # ====================

    def upload_array(
        self,
        arr: da.Array,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Upload a dask array as a new source. See :meth:`UploadSession.upload_array`."""
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
        """Upload a local zarr as a new source. See :meth:`UploadSession.upload_zarr`."""
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
        """Create a writable source on the server. See :meth:`UploadSession.create_source`."""
        return self._upload.create_source(
            source_name, shape, dtype, chunk_shape, dim_labels, ome_metadata
        )

    def upload_chunk(
        self,
        source_id: str,
        bounds: ChunkBounds,
        data: np.ndarray,
    ) -> None:
        """Upload a single chunk. See :meth:`UploadSession.upload_chunk`."""
        self._upload.upload_chunk(source_id, bounds, data)

    def close(self):
        """Close the Flight client."""
        logger.info("Closing Flight client")
        self._client.close()

    def health_check(self) -> Dict[str, Any]:
        """Check server health status via Flight action.

        Returns:
            Dictionary with health status information:
            - status: "SERVING" or other status string. Note: with progressive
              discovery, SERVING means "up and serving the possibly-still-
              populating catalog," not "catalog complete" -- use the freshness
              fields below to tell whether indexing is still in progress.
            - source_count: Number of registered sources
            - metadata_db_enabled: Whether metadata database is enabled
            - writable: Whether server accepts uploads
            - uptime_seconds: Server uptime in seconds
            - full_scan_in_progress: Whether a full catalog rescan is running now
              (absent on older servers)
            - last_full_scan_finished_at: Epoch seconds when a full scan last
              succeeded, or None until the first one does (absent on older
              servers)

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
        """Upload status for a writable source. See :meth:`UploadSession.get_upload_status`."""
        return self._upload.get_upload_status(source_id)

    def get_upload_status_pb(self, pb: SerializedTensor) -> Dict[str, Any]:
        """Upload status for a SerializedTensor handle. See :meth:`UploadSession.get_upload_status_pb`."""
        return self._upload.get_upload_status_pb(pb)

    def wait_for_upload_ready(
        self,
        source_id: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll until the source reports READY. See :meth:`UploadSession.wait_for_upload_ready`."""
        return self._upload.wait_for_upload_ready(
            source_id, timeout_seconds, poll_interval_seconds
        )

    def wait_for_upload_ready_pb(
        self,
        pb: SerializedTensor,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll until a SerializedTensor handle is READY. See :meth:`UploadSession.wait_for_upload_ready_pb`."""
        return self._upload.wait_for_upload_ready_pb(
            pb, timeout_seconds, poll_interval_seconds
        )

    def cache_info(self) -> Dict:
        """Return cache statistics from the pooled cache for this connection.

        Returns:
            Dictionary with cache size and item count
        """
        key = (self._location, self._token)
        pool_entry = _CACHE_POOL.get(key)
        if pool_entry is None:
            # No cache allocated -- either nothing fetched yet, or caching is
            # disabled for this connection (e.g. localhost). Report the resolved
            # size so a disabled cache truthfully shows max_bytes == 0.
            resolved = _resolve_cache_bytes(self._location, self._cache_bytes)
            return {"size_bytes": 0, "max_bytes": resolved, "item_count": 0}
        cache = pool_entry[1]  # Extract cache from (pid, cache) tuple
        if cache is None:
            # Pinned off by configure_cache(): report max_bytes == 0 truthfully.
            return {"size_bytes": 0, "max_bytes": 0, "item_count": 0}
        return {
            "size_bytes": cache.total_bytes,
            "max_bytes": cache.available_bytes,
            "item_count": len(cache.data),
        }

    def cache_clear(self):
        """Clear the pooled cache for this connection namespace."""
        key = (self._location, self._token)
        pool_entry = _CACHE_POOL.get(key)
        if pool_entry is not None and pool_entry[1] is not None:
            pool_entry[1].clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
