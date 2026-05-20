"""Server creation helper for biopb services.

Provides a simplified interface for creating gRPC servers with
health checks, interceptors, and standard configuration.
Optionally starts an embedded tensor cache server for lazy data handling.
"""

import logging
import os
import secrets
import threading
from concurrent import futures
from pathlib import Path
from typing import Optional, Union

import numpy as np
import dask.array as da
import biopb.image as proto
import biopb.tensor as tensor_proto
import grpc

from biopb_image_base.common import TokenValidationInterceptor, _MAX_MSG_SIZE
from biopb_image_base.health import HealthServicer, add_health_servicer
from biopb_image_base.logging_config import setup_logging, LogLevel
from biopb_image_base.debug import get_system_info

logger = logging.getLogger(__name__)

_TENSOR_SERVER_URL_ENV = "TENSOR_SERVER_URL"


def _resolve_tensor_external_location(
    ip: str,
    local: bool,
    tensor_port: int,
    tensor_external_location: Optional[str],
) -> str:
    """Resolve the client-visible tensor server URL for the embedded server."""
    if tensor_external_location:
        # Warn if localhost used with 0.0.0.0 binding (external clients won't reach it)
        if "localhost" in tensor_external_location and ip == "0.0.0.0" and not local:
            logger.warning(
                "tensor-external-location uses 'localhost' while binding to 0.0.0.0. "
                "External clients cannot reach localhost. Use hostname or IP instead "
                "(e.g., 'grpc://hostname:8817')"
            )
        return tensor_external_location
    if local:
        return f"grpc://localhost:{tensor_port}"
    if ip == "0.0.0.0":
        raise ValueError(
            "tensor_external_location is required when binding to 0.0.0.0. "
            "Set it to the externally reachable address "
            "(e.g., 'grpc://hostname:8817')"
        )
    return f"grpc://{ip}:{tensor_port}"


class EmbeddedTensorCache:
    """Wrapper for embedded TensorFlightServer with location rewriting.

    Provides direct source creation without client SDK, and rewrites
    the location field in SerializedTensor to the external URL.
    """

    def __init__(
        self,
        tensor_server,
        external_location: str,
    ):
        """Initialize wrapper.

        Args:
            tensor_server: TensorFlightServer instance
            external_location: External URL for SerializedTensor (e.g., "grpc://hostname:8817")
        """
        self._server = tensor_server
        self._external_location = external_location

    def create_source(
        self,
        array: Union[np.ndarray, da.Array],
        source_name: Optional[str] = None,
        dim_labels: Optional[list] = None,
    ) -> str:
        """Create a cache-backed source from array.

        Args:
            array: Numpy or dask array to upload
            source_name: Optional source name (auto-generated if None)
            dim_labels: Optional dimension labels

        Returns:
            Source ID for use with to_serialized_tensor()
        """
        import hashlib

        from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
        from biopb_tensor_server.chunk import encode_chunk_id
        from biopb.tensor.ticket_pb2 import ChunkBounds

        # Handle dask arrays - compute to numpy
        if isinstance(array, da.Array):
            array = array.compute()

        # Generate source ID
        if source_name:
            source_id = f"cache_{hashlib.sha256(source_name.encode()).hexdigest()[:12]}"
        else:
            import os
            source_id = f"cache_{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"

        # Compute chunking
        shape = list(array.shape)
        dtype = str(array.dtype)

        # Use array's chunks if dask, otherwise single chunk
        if hasattr(array, 'chunks') and array.chunks:
            chunk_shape = [c[0] if isinstance(c, tuple) else c for c in array.chunks]
        else:
            chunk_shape = shape

        # Create adapter
        adapter = CachedSourceAdapter(
            source_id=source_id,
            shape=shape,
            dtype=dtype,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
        )

        # Register with server
        self._server.register_source(source_id, adapter)

        # Write chunks
        ndim = len(shape)
        chunk_starts = []
        for ax, (sh, ch) in enumerate(zip(shape, chunk_shape)):
            starts = list(range(0, sh, ch))
            chunk_starts.append(starts)

        # Iterate through all chunk start positions
        from itertools import product
        for start_coords in product(*chunk_starts):
            # Compute stop coords
            stop_coords = [min(s + chunk_shape[ax], shape[ax]) for ax, s in enumerate(start_coords)]

            # Extract chunk data
            slices = tuple(slice(s, st) for s, st in zip(start_coords, stop_coords))
            chunk_data = array[slices]

            # Create bounds and write
            bounds = ChunkBounds(start=list(start_coords), stop=stop_coords)
            adapter.write_chunk(bounds, chunk_data)

        logger.debug(f"Created cache source {source_id}: shape={shape}, dtype={dtype}")
        return source_id

    def to_serialized_tensor(
        self,
        source_id: str,
        tensor_id: Optional[str] = None,
    ) -> tensor_proto.SerializedTensor:
        """Get SerializedTensor for a source with rewritten location.

        Args:
            source_id: Source identifier
            tensor_id: Tensor ID (optional for single-tensor sources)

        Returns:
            SerializedTensor protobuf with external location
        """
        from biopb.tensor.serialized_pb2 import SerializedTensor, SerializedEndpoint
        from biopb.tensor.ticket_pb2 import TensorTicket

        # Get adapter from server
        adapter = self._server._get_source_adapter(source_id)
        if adapter is None:
            raise ValueError(f"Source not found: {source_id}")

        # Get descriptor
        descriptor = adapter.get_tensor_descriptor()

        # Build endpoints from written chunks
        endpoints = []
        for chunk_id, bounds in adapter._written_chunks.items():
            ticket = TensorTicket(chunk_id=chunk_id)
            ep = SerializedEndpoint(ticket=ticket, chunk_bounds=bounds)
            endpoints.append(ep)

        # Build SerializedTensor with external location
        serialized = SerializedTensor(
            tensor_descriptor=descriptor,
            location=self._external_location,
            auth_token="",  # No auth for embedded cache
            endpoints=endpoints,
        )

        return serialized


class ExternalTensorCache:
    """Upload lazy results to an existing writable tensor server."""

    def __init__(self, location: str):
        from biopb.tensor import TensorFlightClient

        self._location = location
        self._client = TensorFlightClient(location)

    def create_source(
        self,
        array: Union[np.ndarray, da.Array],
        source_name: Optional[str] = None,
        dim_labels: Optional[list] = None,
    ) -> str:
        import hashlib
        import uuid

        if isinstance(array, np.ndarray):
            chunk_shape = tuple(array.shape)
            array = da.from_array(array, chunks=chunk_shape)
        else:
            chunk_shape = array.chunksize

        normalized_name = source_name or ""
        if normalized_name.startswith("cache:"):
            normalized_name = normalized_name[6:]
        if not normalized_name:
            normalized_name = f"upload-{uuid.uuid4().hex}"

        upload_name = f"cache:{normalized_name}"
        source_id = self._client.upload_array(
            array,
            source_name=upload_name,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
        )
        return source_id

    def to_serialized_tensor(
        self,
        source_id: str,
        tensor_id: Optional[str] = None,
    ) -> tensor_proto.SerializedTensor:
        return self._client.get_tensor_pb(source_id, tensor_id=tensor_id)

    def close(self) -> None:
        self._client.close()


def _start_embedded_tensor_cache(
    cache_dir: Path,
    cache_size: int,
    tensor_port: int = 8817,
    tensor_host: str = "0.0.0.0",
) -> tuple[object, str]:
    """Start embedded TensorFlightServer for ephemeral cache.

    Runs in a background thread. Returns (server, location_url).

    Args:
        cache_dir: Directory for cache files
        cache_size: Maximum cache size in bytes
        tensor_port: Port for Flight server
        tensor_host: Host to bind (default 0.0.0.0 for external access)

    Returns:
        Tuple of (tensor_server, location_url)
    """
    from biopb_tensor_server.server import TensorFlightServer
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig

    # Clean stale lock file (from previous run/crash)
    lock_path = cache_dir / "lock"
    if lock_path.exists():
        try:
            lock_path.unlink()
            logger.debug(f"Removed stale cache lock: {lock_path}")
        except Exception as e:
            logger.warning(f"Could not remove stale lock: {e}")

    # Initialize cache manager singleton with file backend
    cache_config = CacheConfig(
        backend="file",
        file_cache_dir=cache_dir,
        file_max_segment_bytes=64 * 1024 * 1024,  # 64MB segments
        file_max_total_bytes=cache_size,
    )
    CacheManager.initialize(cache_config)

    # Bind to specified host (0.0.0.0 for external access)
    location = f"grpc://{tensor_host}:{tensor_port}"

    # Create server with writable mode (cache already initialized)
    tensor_server = TensorFlightServer(
        location,
        writable=True,
    )

    # Start in background thread
    def _run_tensor_server():
        logger.info(f"Embedded tensor cache server started at {location}")
        tensor_server.serve()

    thread = threading.Thread(target=_run_tensor_server, daemon=True)
    thread.start()

    return tensor_server, location


def create_server(
    servicer,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: Optional[bool] = None,
    log_level: LogLevel = "INFO",
    compression: bool = True,
    health_check: bool = True,
    readiness_check: Optional[callable] = None,
    tensor_cache: Optional[EmbeddedTensorCache | ExternalTensorCache] = None,
) -> tuple[grpc.Server, Optional[str], Optional[HealthServicer]]:
    """Create a configured gRPC server with standard features.

    This creates a gRPC server with:
    - ObjectDetection and ProcessImage services registered
    - Optional token authentication
    - Health check service (standard grpc.health.v1.Health)
    - Configurable compression
    - Proper message size limits

    Args:
        servicer: The main servicer implementing ObjectDetection and ProcessImage
        port: Port to listen on (default 50051)
        workers: Thread pool size (default 10)
        ip: IP to bind to (default "0.0.0.0")
        local: Use local server credentials for secure local-only access
        token: Enable token authentication (None = auto based on local flag)
        log_level: Log level for server (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        compression: Enable gzip compression
        health_check: Enable gRPC health check service
        readiness_check: Optional callable for readiness probe
        tensor_cache: Optional tensor cache for lazy data handling

    Returns:
        Tuple of (server, token_string, health_servicer)
        - server: The configured gRPC server (not started)
        - token_string: The auth token if enabled, None otherwise
        - health_servicer: Health servicer for status updates, None if disabled
    """
    # Inject tensor_cache into servicer if provided
    if tensor_cache is not None and hasattr(servicer, '_tensor_cache'):
        servicer._tensor_cache = tensor_cache

    # Determine token setting
    if token is None:
        token = not local

    # Generate token if needed
    token_str = None
    if token:
        token_str = secrets.token_urlsafe(64)
        print()
        print("COPY THE TOKEN BELOW FOR ACCESS.")
        print("=======================================================================")
        print(f"{token_str}")
        print("=======================================================================")
        print()

    # Create server with interceptors
    interceptors = [TokenValidationInterceptor(token_str)]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        compression=grpc.Compression.Gzip if compression else grpc.Compression.NoCompression,
        interceptors=tuple(interceptors),
        options=(
            ("grpc.max_receive_message_length", _MAX_MSG_SIZE),
            ("grpc.max_send_message_length", _MAX_MSG_SIZE),
        ),
    )

    # Register main services
    proto.add_ObjectDetectionServicer_to_server(servicer, server)
    proto.add_ProcessImageServicer_to_server(servicer, server)

    # Register health check service
    health_servicer = None
    if health_check:
        health_servicer = add_health_servicer(server, readiness_check)

    # Add port
    if local:
        server.add_secure_port(f"127.0.0.1:{port}", grpc.local_server_credentials())
        logger.info(f"Server configured with local credentials on 127.0.0.1:{port}")
    else:
        server.add_insecure_port(f"{ip}:{port}")
        logger.info(f"Server configured on {ip}:{port}")

    return server, token_str, health_servicer


def run_server(
    servicer,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: Optional[bool] = None,
    log_level: LogLevel = "INFO",
    compression: bool = True,
    health_check: bool = True,
    readiness_check: Optional[callable] = None,
    cache_dir: Optional[str] = None,
    cache_size: str = "32GB",
    tensor_port: int = 8817,
    tensor_external_location: Optional[str] = None,
) -> None:
    """Create and run a gRPC server (blocking).

    Optionally starts an embedded tensor cache server for lazy data handling.
    When cache_dir is provided, creates an EmbeddedTensorCache wrapper and injects it
    into the servicer for returning large/lazy results.

    Args:
        servicer: The main servicer implementing ObjectDetection and ProcessImage
        port: Port to listen on (default 50051)
        workers: Thread pool size (default 10)
        ip: IP to bind to (default "0.0.0.0")
        local: Use local server credentials for secure local-only access
        token: Enable token authentication (None = auto based on local flag)
        log_level: Log level for server (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        compression: Enable gzip compression
        health_check: Enable gRPC health check service
        readiness_check: Optional callable for readiness probe
        cache_dir: Directory for tensor cache files (enables embedded tensor server)
        cache_size: Maximum cache size (e.g., "32GB", "100GB")
        tensor_port: Port for embedded tensor Flight server (default 8817)
        tensor_external_location: External URL for tensor server in SerializedTensor
            (e.g., "grpc://hostname:8817"). Defaults to "grpc://<ip>:<tensor_port>".
            The tensor server binds to 0.0.0.0 for external access.
    """
    # Setup logging
    setup_logging(log_level)

    # Log system info
    sys_info = get_system_info()
    logger.info(f"System: {sys_info.get('platform', 'unknown')}, "
                f"Python {sys_info.get('python_version', 'unknown')}, "
                f"CPU {sys_info.get('cpu_count', 'unknown')}")
    if "memory_total_mb" in sys_info:
        logger.info(f"Memory: {sys_info['memory_total_mb']:.0f}MB total, "
                    f"{sys_info.get('memory_available_mb', 0):.0f}MB available")
    if "gpu" in sys_info:
        gpu = sys_info["gpu"]
        logger.info(f"GPU: {gpu['device']}, {gpu['total_mb']:.0f}MB total, "
                    f"{gpu['free_mb']:.0f}MB free")

    tensor_cache = None
    external_tensor_server = os.environ.get(_TENSOR_SERVER_URL_ENV)
    if external_tensor_server:
        tensor_cache = ExternalTensorCache(external_tensor_server)
        logger.info(f"Using external tensor server for lazy uploads: {external_tensor_server}")
    elif cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Parse size string
        size_str = cache_size.upper()
        if size_str.endswith("GB"):
            cache_bytes = int(size_str[:-2]) * 1024 * 1024 * 1024
        elif size_str.endswith("MB"):
            cache_bytes = int(size_str[:-2]) * 1024 * 1024
        else:
            cache_bytes = int(cache_size)

        logger.info(f"Starting embedded tensor cache at {cache_dir} (size: {cache_size})")

        # Determine external location for SerializedTensor
        external_location = _resolve_tensor_external_location(
            ip=ip,
            local=local,
            tensor_port=tensor_port,
            tensor_external_location=tensor_external_location,
        )

        # Start embedded tensor server (binds to 0.0.0.0 for external access)
        tensor_server, bind_location = _start_embedded_tensor_cache(
            cache_dir=cache_path,
            cache_size=cache_bytes,
            tensor_port=tensor_port,
            tensor_host="0.0.0.0",
        )

        # Create wrapper with location rewriting
        tensor_cache = EmbeddedTensorCache(
            tensor_server=tensor_server,
            external_location=external_location,
        )
        logger.info(f"Tensor server listening on {bind_location}")
        logger.info(f"Tensor server advertised at {external_location}")

    logger.info("server starting ...")

    server, token_str, health_servicer = create_server(
        servicer=servicer,
        port=port,
        workers=workers,
        ip=ip,
        local=local,
        token=token,
        log_level=log_level,
        compression=compression,
        health_check=health_check,
        readiness_check=readiness_check,
        tensor_cache=tensor_cache,
    )

    logger.info("server ready")

    server.start()
    server.wait_for_termination()