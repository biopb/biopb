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
from itertools import product
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import biopb.image as proto
import biopb.tensor as tensor_proto
import dask.array as da
import grpc
import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.serialized_pb2 import SerializedTensor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_image_base.common import _MAX_MSG_SIZE, TokenValidationInterceptor
from biopb_image_base.debug import get_system_info
from biopb_image_base.health import HealthServicer, add_health_servicer
from biopb_image_base.logging_config import LogLevel, setup_logging

logger = logging.getLogger(__name__)

_NON_UNIFORM_CHUNKS_ERROR = "Non-uniform dask chunks are not supported; rechunk to a uniform grid before uploading."


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


def _normalize_dim_labels(
    dim_labels: Optional[Sequence[str]],
    ndim: int,
) -> list[str]:
    if dim_labels is not None:
        return list(dim_labels)
    return [f"dim{i}" for i in range(ndim)]


def _as_dask_array(array: Union[np.ndarray, da.Array]) -> da.Array:
    if isinstance(array, da.Array):
        return array
    return da.from_array(array, chunks=array.shape)


def _uniform_chunk_shape(array: da.Array) -> tuple[int, ...]:
    if not all(len(set(axis_chunks)) == 1 for axis_chunks in array.chunks):
        raise ValueError(_NON_UNIFORM_CHUNKS_ERROR)
    return tuple(int(axis_chunks[0]) for axis_chunks in array.chunks)


def _iter_chunk_bounds(
    shape: Sequence[int],
    chunk_shape: Sequence[int],
) -> Iterator[ChunkBounds]:
    chunk_starts = [
        range(0, int(dim), int(chunk))
        for dim, chunk in zip(shape, chunk_shape, strict=True)
    ]
    for start_coords in product(*chunk_starts):
        stop_coords = [
            min(start + int(chunk_shape[axis]), int(shape[axis]))
            for axis, start in enumerate(start_coords)
        ]
        yield ChunkBounds(start=list(start_coords), stop=stop_coords)


def _bounds_to_slices(bounds: ChunkBounds) -> tuple[slice, ...]:
    return tuple(
        slice(start, stop)
        for start, stop in zip(bounds.start, bounds.stop, strict=True)
    )


def _build_registration_tensor(
    source_id: str,
    shape: Sequence[int],
    dtype: str,
    chunk_shape: Sequence[int],
    dim_labels: Optional[Sequence[str]],
    location: str,
    auth_token: str = "",
) -> SerializedTensor:
    descriptor = TensorDescriptor(
        array_id=source_id,
        dim_labels=_normalize_dim_labels(dim_labels, len(shape)),
        shape=list(shape),
        chunk_shape=list(chunk_shape),
        dtype=dtype,
    )
    return SerializedTensor(
        tensor_descriptor=descriptor,
        location=location,
        auth_token=auth_token,
        endpoints=[],
    )


def _normalize_cache_source_name(source_name: Optional[str]) -> str:
    normalized_name = source_name or ""
    if normalized_name.startswith("cache:"):
        normalized_name = normalized_name[6:]
    return normalized_name


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

    def _register_array_template(
        self,
        array_template: da.Array,
        source_name: Optional[str] = None,
        dim_labels: Optional[Sequence[str]] = None,
    ) -> tuple[str, da.Array, tuple[int, ...]]:
        import hashlib

        from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter

        chunk_shape = _uniform_chunk_shape(array_template)
        normalized_name = _normalize_cache_source_name(source_name)

        if normalized_name:
            source_id = (
                f"cache_{hashlib.sha256(normalized_name.encode()).hexdigest()[:12]}"
            )
        else:
            source_id = f"cache_{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"

        adapter = CachedSourceAdapter(
            source_id=source_id,
            shape=list(array_template.shape),
            dtype=array_template.dtype.str,
            chunk_shape=list(chunk_shape),
            dim_labels=list(dim_labels) if dim_labels is not None else None,
        )
        # Per-source capability token: the result is readable only by the caller
        # that receives this SerializedTensor (carried in its auth_token). The
        # embedded server runs writable=False and writes happen in-process, so
        # this token gates read-back without a server-wide secret.
        adapter.capability_token = secrets.token_urlsafe(32)
        self._server.register_source(source_id, adapter)
        self._server.uploads.initialize(source_id, array_template.shape, chunk_shape)
        return source_id, array_template, chunk_shape

    def create_array(
        self,
        source_name: Optional[str],
        dim_labels: Optional[list],
        array_template: da.Array,
    ) -> tensor_proto.SerializedTensor:
        source_id, normalized_array, chunk_shape = self._register_array_template(
            array_template=array_template,
            source_name=source_name,
            dim_labels=dim_labels,
        )
        return _build_registration_tensor(
            source_id=source_id,
            shape=normalized_array.shape,
            dtype=normalized_array.dtype.str,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
            location=self._external_location,
            auth_token=self._server.sources.get(source_id).capability_token,
        )

    def upload_array_chunks(
        self,
        source_id: str,
        endpoint: ChunkBounds,
        chunk: np.ndarray,
    ) -> None:
        adapter = self._server.sources.get(source_id)
        if adapter is None:
            raise ValueError(f"Source not found: {source_id}")
        adapter.write_chunk(endpoint, chunk)
        self._server.uploads.mark_chunk(source_id, endpoint)

    def get_upload_status(self, source_id: str) -> dict:
        return self._server.uploads.status(source_id)

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
        dask_array = _as_dask_array(array)
        source_id, normalized_array, chunk_shape = self._register_array_template(
            array_template=dask_array,
            source_name=source_name,
            dim_labels=dim_labels,
        )

        for bounds in _iter_chunk_bounds(normalized_array.shape, chunk_shape):
            chunk_data = normalized_array[_bounds_to_slices(bounds)].compute()
            self.upload_array_chunks(source_id, bounds, chunk_data)

        logger.debug(
            "Created cache source %s: shape=%s, dtype=%s",
            source_id,
            list(normalized_array.shape),
            normalized_array.dtype.str,
        )
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
        from biopb.tensor.serialized_pb2 import SerializedEndpoint, SerializedTensor
        from biopb.tensor.ticket_pb2 import TensorTicket

        # Get adapter from server
        adapter = self._server.sources.get(source_id)
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

        # Build SerializedTensor with external location. auth_token carries the
        # per-source capability token so only this caller can read the result.
        serialized = SerializedTensor(
            tensor_descriptor=descriptor,
            location=self._external_location,
            auth_token=adapter.capability_token or "",
            endpoints=endpoints,
        )

        return serialized


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
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.core.config import CacheConfig
    from biopb_tensor_server.serving.server import TensorFlightServer

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

    # Read-only over Flight: results are written in-process (adapter.write_chunk),
    # so the Flight write path (do_put / create_source) is pure attack surface here.
    # Read-back is gated by per-source capability tokens (adapter.capability_token).
    tensor_server = TensorFlightServer(
        location,
        writable=False,
    )

    # This embedded server is a *bypass* of the normal tensor-server lifecycle:
    # it hijacks a TensorFlightServer purely as scratch-pad storage for op
    # results and has no data-folder scan / source-registration stage at all
    # (sources appear in-process via adapter.write_chunk). It is therefore ready
    # to serve the instant its Flight port binds. The CLI launcher is the
    # authoritative path that defers mark_ready() until after its scan; here
    # there is nothing to wait for, so mark ready immediately -- otherwise the
    # health action would report STARTING forever and readiness-gating clients
    # (e.g. biopb-mcp) would wait indefinitely.
    tensor_server.mark_ready()

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
    tensor_cache: Optional[EmbeddedTensorCache] = None,
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
    if tensor_cache is not None and hasattr(servicer, "_tensor_cache"):
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
        compression=grpc.Compression.Gzip
        if compression
        else grpc.Compression.NoCompression,
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


def _pyarrow_available() -> bool:
    """True if pyarrow can be imported.

    The tensor cache (lazy data side channel) is built on Arrow Flight and needs
    pyarrow. On builds for old CPUs without SSE4.2/AVX, pyarrow is removed (its
    wheels SIGILL on import there -- see cellpose/BUILD_NO_SSE42.md), so the side
    channel must not be started. find_spec only locates the module; it does not
    import it, so this is safe even on a CPU that cannot run pyarrow.
    """
    import importlib.util

    return importlib.util.find_spec("pyarrow") is not None


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
    logger.info(
        f"System: {sys_info.get('platform', 'unknown')}, "
        f"Python {sys_info.get('python_version', 'unknown')}, "
        f"CPU {sys_info.get('cpu_count', 'unknown')}"
    )
    if "memory_total_mb" in sys_info:
        logger.info(
            f"Memory: {sys_info['memory_total_mb']:.0f}MB total, "
            f"{sys_info.get('memory_available_mb', 0):.0f}MB available"
        )
    if "gpu" in sys_info:
        gpu = sys_info["gpu"]
        logger.info(
            f"GPU: {gpu['device']}, {gpu['total_mb']:.0f}MB total, "
            f"{gpu['free_mb']:.0f}MB free"
        )

    tensor_cache = None
    if cache_dir is not None and not _pyarrow_available():
        # No-SSE4.2/AVX build: pyarrow (hence the lazy/Flight side channel) is
        # unavailable. Do not start the tensor server -- it would crash. Lazy
        # (dask) requests will be cleanly rejected by the servicer instead.
        logger.warning(
            "Tensor cache (lazy data side channel) was requested (cache_dir) "
            "but pyarrow is not available -- this looks like a build for a CPU "
            "without SSE4.2/AVX. Disabling the tensor server; only eager image "
            "data is supported and lazy (dask) input/output will be rejected."
        )
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

        logger.info(
            f"Starting embedded tensor cache at {cache_dir} (size: {cache_size})"
        )

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
