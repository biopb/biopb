"""Python client for TensorFlight server.

This module provides a lazy numpy-like array interface using dask.array
for accessing tensors stored in a Flight server.

Features:
- Lazy chunk loading via dask.array
- LRU caching via cachey
- Numpy-compatible slicing and operations
"""

import logging
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import threading

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import dask.array as da
from dask.delayed import delayed
from cachey import Cache

from biopb.tensor.ticket_pb2 import TensorTicket, ChunkBounds, ChunkUpload
from biopb.tensor.descriptor_pb2 import (
    TensorDescriptor,
    SliceHint,
    FlightCmd,
    TensorReadOption,
    MetadataQueryOption,
    DataSourceDescriptor,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Module-level pools for worker-local connection caching (pickle-safe)
# ==============================================================================
#
# Each entry stores (pid, resource) to detect forked processes. When a process
# forks after pool was populated, the child detects pid mismatch and creates
# fresh connections (inherited gRPC sockets are broken).
#
_CONNECTION_POOL: Dict[Tuple[str, Optional[str]], Tuple[int, flight.FlightClient]] = {}
_CACHE_POOL: Dict[Tuple[str, Optional[str]], Tuple[int, Cache]] = {}
_CALL_OPTS_POOL: Dict[Tuple[str, Optional[str]], flight.FlightCallOptions] = {}
_POOL_LOCK = threading.Lock()


def _get_worker_resources(location: str, token: Optional[str], cache_bytes: int):
    """Get cached FlightClient, Cache, and CallOptions for a connection namespace.

    Creates resources lazily on first call per (location, token) key.
    Thread-safe via lock. Fork-safe: stale entries from parent process
    are detected and cleaned up before use.

    Each worker process has its own pool after unpickle. If a process
    forks after pool was populated, the child detects pid mismatch and
    creates fresh connections.

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)
        cache_bytes: Cache size for worker-local cache

    Returns:
        Tuple of (FlightClient, Cache, FlightCallOptions)
    """
    key = (location, token)
    current_pid = os.getpid()

    with _POOL_LOCK:
        # Fork-safety: detect and clean up inherited stale connections
        if key in _CONNECTION_POOL:
            pool_pid, client = _CONNECTION_POOL[key]
            if pool_pid != current_pid:
                # Forked child - inherited gRPC socket is broken, close it
                try:
                    client.close()
                except Exception:
                    pass
                del _CONNECTION_POOL[key]

        if key in _CACHE_POOL:
            pool_pid, cache = _CACHE_POOL[key]
            if pool_pid != current_pid:
                del _CACHE_POOL[key]

        # Create fresh resources for current process
        if key not in _CONNECTION_POOL:
            _CONNECTION_POOL[key] = (current_pid, flight.FlightClient(location))
        if key not in _CACHE_POOL:
            _CACHE_POOL[key] = (current_pid, Cache(available_bytes=cache_bytes))
        if key not in _CALL_OPTS_POOL:
            if token:
                _CALL_OPTS_POOL[key] = flight.FlightCallOptions(
                    headers=[(b"authorization", f"Bearer {token}".encode())]
                )
            else:
                _CALL_OPTS_POOL[key] = flight.FlightCallOptions()

    return (
        _CONNECTION_POOL[key][1],  # client
        _CACHE_POOL[key][1],  # cache
        _CALL_OPTS_POOL[key],  # call_options
    )


def _fetch_chunk_distributed(
    location: str,
    token: Optional[str],
    chunk_id: bytes,
    bounds_start: Tuple[int, ...],
    bounds_stop: Tuple[int, ...],
    cache_bytes: int,
) -> np.ndarray:
    """Fetch a chunk from Flight server using worker-local resources.

    This function is pickle-safe because it has no closure references to
    non-serializable objects (FlightClient). Connection and cache are
    obtained from module-level pools at runtime.

    Args:
        location: Flight server location string
        token: Bearer token (or None for no auth)
        chunk_id: Chunk identifier bytes
        bounds_start: Chunk start coordinates as tuple
        bounds_stop: Chunk stop coordinates as tuple
        cache_bytes: Cache size for worker-local cache

    Returns:
        numpy array with chunk data
    """
    client, cache, call_options = _get_worker_resources(location, token, cache_bytes)

    # Cache lookup
    cache_key = chunk_id.hex()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug(f"fetch_chunk_distributed: cache hit for {cache_key[:16]}")
        return cached

    logger.debug(f"fetch_chunk_distributed: fetching {cache_key[:16]} from server")

    # Fetch from server
    ticket = TensorTicket(chunk_id=chunk_id)
    reader = client.do_get(
        flight.Ticket(ticket.SerializeToString()), options=call_options
    )
    table = reader.read_all()
    arr = table.column("data").to_numpy()[0]  # First row's data list

    # Get shape from shape column (list<int64>)
    shape = tuple(table.column("shape").to_pylist()[0])
    arr = arr.reshape(shape)

    # Cache the result
    cache.put(cache_key, arr, cost=arr.nbytes)

    return arr


def _parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string to (major, minor, patch) tuple."""
    # Handle dev versions like "0.3.1.dev43+g..."
    base = version_str.split(".dev")[0].split("+")[0]
    parts = base.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (major, minor, patch)


def _check_schema_version(schema: pa.Schema) -> None:
    """Check schema metadata version and warn if client version is older."""
    if schema.metadata is None:
        return

    server_version_bytes = schema.metadata.get(b"tensor_schema_version")
    if server_version_bytes is None:
        return

    server_version = server_version_bytes.decode("utf-8")
    try:
        client_version = importlib.metadata.version("biopb")
    except importlib.metadata.PackageNotFoundError:
        return

    server_parsed = _parse_version(server_version)
    client_parsed = _parse_version(client_version)

    if client_parsed < server_parsed:
        logger.warning(
            f"Client version {client_version} is older than server schema version {server_version}. "
            f"Consider upgrading biopb client for compatibility."
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

        # Access a specific tensor
        arr = client.get_tensor('my-source', 'tensor-0')  # Returns dask.array
        data = arr[0:100, 0:100].compute()   # Load slice

    Note:
        This client IS compatible with dask.distributed. The dask arrays
        returned by get_tensor() use a pickle-safe design where FlightClient
        connections are created lazily in each worker from stored connection
        parameters. Each worker maintains its own connection pool and LRU
        cache keyed by (location, token).
    """

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
        # Store pickle-safe connection parameters
        self._location = location
        self._token = token
        self._cache_bytes = cache_bytes
        # Create FlightClient for direct API calls (list_flights, get_flight_info, uploads)
        self._client = flight.FlightClient(location)
        self._call_options = (
            flight.FlightCallOptions(
                headers=[(b"authorization", f"Bearer {token}".encode())]
            )
            if token
            else flight.FlightCallOptions()
        )
        # Cache descriptors for metadata
        self._sources: Dict[str, DataSourceDescriptor] = {}
        self._descriptors: Dict[str, TensorDescriptor] = {}

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
        source_descriptors = {}
        truncated = False
        total_sources = None

        for info in self._client.list_flights(options=self._call_options):
            source_desc = DataSourceDescriptor.FromString(info.descriptor.command)
            source_descriptors[source_desc.source_id] = source_desc
            # Cache tensor descriptors
            for tensor_desc in source_desc.tensors:
                self._descriptors[tensor_desc.array_id] = tensor_desc

            # Check schema metadata for truncation info
            if info.schema.metadata:
                truncated_bytes = info.schema.metadata.get(b"truncated")
                if truncated_bytes:
                    truncated = truncated_bytes.decode() == "True"
                total_sources_bytes = info.schema.metadata.get(b"total_sources")
                if total_sources_bytes:
                    total_sources = int(total_sources_bytes.decode())

        self._sources = source_descriptors

        if truncated and total_sources:
            logger.warning(
                f"list_sources: returned {len(source_descriptors)} of {total_sources} sources (truncated)"
            )
        else:
            logger.info(f"list_sources: returned {len(source_descriptors)} sources")

        return source_descriptors

    def query_sources(self, sql: str) -> pa.Table:
        """Execute SQL query against server's source metadata database.

        Returns Arrow Table with query results. Schema metadata may contain
        'total_sources' and 'returned_sources' keys if result was truncated.

        Requires server to have metadata_db.enabled=True in config.

        Args:
            sql: SQL query (e.g., "SELECT source_id, source_type FROM sources WHERE dtype='uint16'")

        Returns:
            pyarrow.Table with query results

        Raises:
            FlightServerError: If server does not have metadata database enabled
            ValueError: If query contains forbidden keywords or references disallowed tables

        Example:
            >>> client = TensorFlightClient('grpc://localhost:8815')
            >>> table = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
            >>> print(table.to_pandas())
        """
        cmd = FlightCmd(
            source_id="__metadata_query__",
            metadata_query=MetadataQueryOption(sql=sql),
        )
        descriptor = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = self._client.get_flight_info(descriptor, options=self._call_options)

        # Check schema metadata for truncation info
        if info.schema.metadata:
            total_sources = info.schema.metadata.get(b"total_sources")
            if total_sources:
                total = int(total_sources.decode())
                returned = info.schema.metadata.get(b"returned_sources")
                if returned:
                    returned_count = int(returned.decode())
                    if returned_count < total:
                        logger.info(
                            f"query_sources: returned {returned_count} of {total} sources (truncated)"
                        )
                    else:
                        logger.info(f"query_sources: returned {returned_count} sources")

        # Fetch results via DoGet
        if info.endpoints:
            reader = self._client.do_get(
                info.endpoints[0].ticket, options=self._call_options
            )
            return reader.read_all()
        else:
            # Empty result
            return info.schema.empty_table()

    def get_source_metadata(self, source_id: str) -> dict:
        """Get source-level OME/vendor metadata.

        Fetches metadata via GetFlightInfo for the first tensor in the source,
        since metadata_json is populated in the response TensorDescriptor.

        Args:
            source_id: Source identifier

        Returns:
            Parsed metadata from GetFlightInfo response.
            The server wraps metadata in {"type": ..., "dim_label": [...], "metadata": {...}},
            this method returns the inner "metadata" dict,
            or empty dict if no metadata.
        """
        import json

        if source_id not in self._sources:
            self.list_sources()

        source_desc = self._sources.get(source_id)
        if source_desc is None:
            raise ValueError(f"Source not found: {source_id}")

        if not source_desc.tensors:
            return {}

        # Get metadata from first tensor via GetFlightInfo
        first_tensor = source_desc.tensors[0]
        cmd = FlightCmd(
            source_id=source_id,
            tensor_read=TensorReadOption(
                tensor_id=first_tensor.array_id,
                with_metadata=True,
            ),
        )
        flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = self._client.get_flight_info(flight_desc, options=self._call_options)
        response_desc = TensorDescriptor.FromString(info.descriptor.command)

        if response_desc.metadata_json:
            wrapped = json.loads(response_desc.metadata_json)
            # Unwrap to return just the metadata dict
            return wrapped.get("metadata", {})
        return {}

    def get_tensor(
        self,
        source_id: str,
        tensor_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> da.Array:
        """Get a lazy dask array for a tensor within a data source.

        Args:
            source_id: Data source identifier
            tensor_id: Tensor identifier within the source
            slice_hint: Optional slice tuple to filter chunks
            scale_hint: Optional per-dimension integer downsampling factors
            reduction_method: Optional dynamic reduction method for scaled reads

        Returns:
            dask.array with lazy chunk loading
        """
        logger.debug(f"get_tensor: source_id={source_id}, tensor_id={tensor_id}")
        # Get tensor descriptor from cached source info
        if source_id not in self._sources:
            self.list_sources()

        source_desc = self._sources.get(source_id)
        if source_desc is None:
            raise ValueError(f"Source not found: {source_id}")

        # Find tensor descriptor
        tensor_desc = None
        for desc in source_desc.tensors:
            if desc.array_id == tensor_id:
                tensor_desc = desc
                break

        if tensor_desc is None:
            raise ValueError(f"Tensor '{tensor_id}' not found in source '{source_id}'")

        # Convert slice_hint to SliceHint proto
        slice_hint_proto = None
        if slice_hint is not None:
            starts = []
            stops = []
            for s in slice_hint:
                starts.append(s.start if s.start is not None else 0)
                stops.append(
                    s.stop if s.stop is not None else tensor_desc.shape[len(starts) - 1]
                )
            slice_hint_proto = SliceHint(start=starts, stop=stops)

        # Build TensorReadOption with flattened fields
        read_opt = TensorReadOption(
            tensor_id=tensor_id,
            with_metadata=False,
        )
        if slice_hint_proto is not None:
            read_opt.slice_hint.CopyFrom(slice_hint_proto)
        if scale_hint is not None:
            read_opt.scale_hint[:] = list(scale_hint)
        if reduction_method is not None:
            read_opt.reduction_method = reduction_method

        # Build FlightCmd for the request
        cmd = FlightCmd(
            source_id=source_id,
            tensor_read=read_opt,
        )

        # Get flight info
        flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = self._client.get_flight_info(flight_desc, options=self._call_options)
        response_desc = TensorDescriptor.FromString(info.descriptor.command)

        # Check schema version compatibility
        _check_schema_version(info.schema)

        # Cache the response descriptor
        self._descriptors[response_desc.array_id] = response_desc

        # Parse endpoints into chunk info
        chunks = []
        chunk_bounds_list = []

        for endpoint in info.endpoints:
            # Parse ticket
            ticket = TensorTicket.FromString(endpoint.ticket.ticket)

            # Parse chunk bounds from app_metadata
            bounds = ChunkBounds.FromString(endpoint.app_metadata)

            chunks.append(ticket.chunk_id)
            chunk_bounds_list.append(bounds)

        # Build dask array from chunks
        dask_arr = self._build_dask_array(
            desc=response_desc,
            chunks=chunks,
            chunk_bounds=chunk_bounds_list,
        )

        # Crop to the originally requested region.
        # The server snaps slice_hint outward to lcm-aligned chunk boundaries, so
        # the returned descriptor.shape may be larger than what was requested.
        # We crop the dask array back to the exact requested region here.
        if slice_hint_proto is not None and response_desc.HasField("slice_hint"):
            realized = response_desc.slice_hint
            ndim = len(response_desc.shape)
            # Use scale_hint from read_opt (flattened field)
            scale = list(read_opt.scale_hint) if read_opt.scale_hint else None
            crop = []
            for ax in range(ndim):
                req_start = int(slice_hint_proto.start[ax])
                req_stop = int(slice_hint_proto.stop[ax])
                ret_start = int(realized.start[ax])
                s = int(scale[ax]) if scale and ax < len(scale) else 1
                if s > 1:
                    logical_start = (req_start - ret_start) // s
                    logical_stop = (req_stop - ret_start + s - 1) // s
                else:
                    logical_start = req_start - ret_start
                    logical_stop = req_stop - ret_start
                crop.append(slice(logical_start, logical_stop))
            dask_arr = dask_arr[tuple(crop)]

        return dask_arr

    def _build_dask_array(
        self,
        desc: TensorDescriptor,
        chunks: List[bytes],
        chunk_bounds: List[ChunkBounds],
    ) -> da.Array:
        """Build a dask array from chunk info.

        Args:
            desc: Tensor descriptor
            chunks: List of chunk IDs
            chunk_bounds: List of chunk bounds

        Returns:
            dask.array with lazy chunk loading
        """
        shape = tuple(desc.shape)
        dtype = np.dtype(desc.dtype)

        # Create a mapping from chunk index to chunk_id and bounds
        chunk_map = {}
        axis_starts = [
            sorted({int(bounds.start[axis]) for bounds in chunk_bounds})
            for axis in range(len(shape))
        ]
        axis_index_maps = [
            {start: index for index, start in enumerate(starts)}
            for starts in axis_starts
        ]
        for chunk_id, bounds in zip(chunks, chunk_bounds):
            chunk_idx = tuple(
                axis_index_maps[d][int(bounds.start[d])] for d in range(len(shape))
            )
            chunk_map[chunk_idx] = (chunk_id, bounds)

        # Create delayed function to fetch chunk
        # Cache lookup happens INSIDE this function at compute time
        def fetch_chunk(chunk_id: bytes, bounds: ChunkBounds) -> np.ndarray:
            """Fetch and cache a chunk from the server."""
            cache_key = chunk_id.hex()
            cached = self._cache.get(cache_key)
            if cached is not None:
                with self._cache_lock:
                    self._cache_hits += 1
                logger.debug(f"fetch_chunk: cache hit for {cache_key[:16]}")
                return cached

            with self._cache_lock:
                self._cache_misses += 1

            logger.debug(f"fetch_chunk: fetching {cache_key[:16]} from server")
            # Fetch from server
            ticket = TensorTicket(chunk_id=chunk_id)
            reader = self._client.do_get(
                flight.Ticket(ticket.SerializeToString()), options=self._call_options
            )

            # Read all data from the stream
            table = reader.read_all()

            # Extract data from list column (each row is one chunk's flattened data)
            # Data column is list<dtype>, we get the first row's list
            arr = table.column("data").to_numpy()[0]  # First row's data list

            # Get shape from shape column (list<int64>)
            shape = tuple(table.column("shape").to_pylist()[0])
            arr = arr.reshape(shape)

            # Cache the result
            self._cache.put(cache_key, arr, cost=arr.nbytes)

            return arr

        # Build dask array using da.from_delayed for each chunk
        # The fetch_chunk function is called at compute time, not graph-build time
        # Find the grid dimensions
        ndim = len(shape)
        grid_shape = tuple(
            max(idx[d] + 1 for idx in chunk_map.keys()) for d in range(ndim)
        )

        # Create block array
        blocks = np.empty(grid_shape, dtype=object)
        for chunk_idx, (chunk_id, bounds) in chunk_map.items():
            chunk_shape = tuple(
                stop - start for start, stop in zip(bounds.start, bounds.stop)
            )
            # Use pickle-safe module-level function with connection params
            delayed_arr = da.from_delayed(
                delayed(_fetch_chunk_distributed)(
                    self._location,
                    self._token,
                    chunk_id,
                    tuple(bounds.start),
                    tuple(bounds.stop),
                    self._cache_bytes,
                ),
                shape=chunk_shape,
                dtype=dtype,
            )
            blocks[chunk_idx] = delayed_arr

        if blocks.size == 0:
            raise ValueError("No chunks found")

        # Use da.block to combine chunks into a single array
        return da.block(blocks.tolist())

    # ====================
    # Upload API
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
        # Determine target chunk shape
        if chunk_shape is None:
            chunk_shape = arr.chunksize

            # Check if dask chunks are non-uniform
            needs_rechunk = not all(
                len(set(arr.chunks[d])) == 1 for d in range(arr.ndim)
            )

            if needs_rechunk:
                uniform_chunks = tuple(
                    max(arr.chunks[d]) if arr.chunks[d] else arr.shape[d]
                    for d in range(arr.ndim)
                )
                arr = arr.rechunk(uniform_chunks)
                chunk_shape = uniform_chunks
        else:
            if tuple(chunk_shape) != tuple(arr.chunksize):
                arr = arr.rechunk(tuple(chunk_shape))

        # Create source
        source_id = self.create_source(
            source_name=source_name,
            shape=arr.shape,
            dtype=arr.dtype.str,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
            ome_metadata=ome_metadata,
        )

        # Upload chunks
        ndim = arr.ndim
        chunk_shape_tuple = tuple(chunk_shape)
        chunks_per_dim = [
            (arr.shape[d] + chunk_shape_tuple[d] - 1) // chunk_shape_tuple[d]
            for d in range(ndim)
        ]

        from itertools import product

        for chunk_idx in product(*(range(n) for n in chunks_per_dim)):
            chunk_start = [
                idx * chunk_shape_tuple[d] for d, idx in enumerate(chunk_idx)
            ]
            chunk_stop = [
                min((idx + 1) * chunk_shape_tuple[d], arr.shape[d])
                for d, idx in enumerate(chunk_idx)
            ]

            bounds = ChunkBounds(start=chunk_start, stop=chunk_stop)

            slices = tuple(
                slice(chunk_start[d], chunk_stop[d]) for d in range(arr.ndim)
            )
            chunk_data = arr[slices].compute()
            self.upload_chunk(source_id, bounds, chunk_data)

        return source_id

    def upload_zarr(
        self,
        zarr_path: str,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Upload local zarr to server.

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
        import zarr

        arr = zarr.open_array(zarr_path, mode="r")

        # Read metadata from local zarr if not provided
        zattrs_path = Path(zarr_path) / ".zattrs"
        if zattrs_path.exists():
            with open(zattrs_path) as f:
                zattrs = json.load(f)
            if ome_metadata is None and "multiscales" in zattrs:
                ome_metadata = zattrs
            if dim_labels is None and "multiscales" in zattrs:
                axes = zattrs["multiscales"][0].get("axes", [])
                dim_labels = [
                    ax.get("name") if isinstance(ax, dict) else str(ax) for ax in axes
                ]

        dask_arr = da.from_zarr(zarr_path)
        effective_chunk_shape = chunk_shape or arr.chunks

        return self.upload_array(
            dask_arr,
            source_name=source_name,
            chunk_shape=effective_chunk_shape,
            dim_labels=dim_labels,
            ome_metadata=ome_metadata,
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
        req_desc = TensorDescriptor(
            array_id=source_name,
            shape=list(shape),
            dtype=dtype,
            chunk_shape=list(chunk_shape),
            dim_labels=list(dim_labels or []),
            metadata_json=json.dumps(ome_metadata) if ome_metadata else "",
        )

        desc = flight.FlightDescriptor.for_command(req_desc.SerializeToString())
        writer, reader = self._client.do_put(
            desc, pa.schema([]), options=self._call_options
        )
        writer.close()

        metadata = reader.read()
        response_desc = TensorDescriptor.FromString(metadata.to_pybytes())
        logger.info(f"create_source: created {response_desc.array_id}")
        return response_desc.array_id

    def upload_chunk(
        self,
        source_id: str,
        bounds: ChunkBounds,
        data: np.ndarray,
    ) -> None:
        """Upload single chunk (internal).

        Args:
            source_id: Source identifier
            bounds: Chunk start/stop coordinates
            data: Numpy array with chunk data
        """
        upload = ChunkUpload(
            source_id=source_id,
            bounds=bounds,
        )

        desc = flight.FlightDescriptor.for_command(upload.SerializeToString())
        schema = pa.schema([pa.field("data", pa.from_numpy_dtype(data.dtype))])

        writer, reader = self._client.do_put(desc, schema, options=self._call_options)
        batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])
        writer.write_batch(batch)
        writer.close()
        reader.read()
        logger.debug(f"upload_chunk: uploaded {data.nbytes} bytes to {source_id}")

    def close(self):
        """Close the Flight client."""
        logger.info("Closing Flight client")
        self._client.close()

    def health_check(self) -> Dict[str, Any]:
        """Check server health status via Flight action.

        Returns:
            Dictionary with health status information:
            - status: "SERVING" or other status string
            - source_count: Number of registered sources
            - metadata_db_enabled: Whether metadata database is enabled
            - writable: Whether server accepts uploads
            - uptime_seconds: Server uptime in seconds

        Raises:
            FlightError: If server is unreachable or action fails
        """
        action = flight.Action("health", b"")
        results = self._client.do_action(action, options=self._call_options)
        for result in results:
            return json.loads(result.body.to_pybytes())
        return {"status": "UNKNOWN"}

    def cache_info(self) -> Dict:
        """Return cache statistics from the pooled cache for this connection.

        Returns:
            Dictionary with cache size and item count
        """
        key = (self._location, self._token)
        pool_entry = _CACHE_POOL.get(key)
        if pool_entry is None:
            return {"size_bytes": 0, "max_bytes": self._cache_bytes, "item_count": 0}
        cache = pool_entry[1]  # Extract cache from (pid, cache) tuple
        return {
            "size_bytes": cache.total_bytes,
            "max_bytes": cache.available_bytes,
            "item_count": len(cache.data),
        }

    def cache_clear(self):
        """Clear the pooled cache for this connection namespace."""
        key = (self._location, self._token)
        pool_entry = _CACHE_POOL.get(key)
        if pool_entry is not None:
            cache = pool_entry[1]
            cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
