"""The embedded tensor cache: a server returns large results through its own
TensorFlight server, from a file-based cache with a TTL."""

import logging
import os
import re
import secrets
import shutil
import threading
from itertools import product
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import biopb.tensor as tensor_proto
import dask.array as da
import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.serialized_pb2 import SerializedTensor
from biopb.tensor.ticket_pb2 import ChunkBounds
from dask.utils import parse_bytes

from biopb_image_base.common import _pyarrow_available

logger = logging.getLogger(__name__)

_NON_UNIFORM_CHUNKS_ERROR = "Non-uniform dask chunks are not supported; rechunk to a regular grid before uploading."


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


def _as_dask_array(array: Union[np.ndarray, da.Array]) -> da.Array:
    if isinstance(array, da.Array):
        return array
    return da.from_array(array, chunks=array.shape)


def _uniform_chunk_shape(array: da.Array) -> tuple[int, ...]:
    """The chunk grid of *array*: one size per axis, a smaller last chunk allowed."""
    for axis_chunks in array.chunks:
        if len(set(axis_chunks[:-1])) > 1 or axis_chunks[-1] > axis_chunks[0]:
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


def _handle(
    descriptor: TensorDescriptor, location: str, auth_token: str
) -> SerializedTensor:
    """A SerializedTensor for a source this process serves.

    Describe-only: a FlightInfo carrying the descriptor and no endpoints, which
    the consumer plans with its own GetFlightInfo when it reads. One shape for
    a result still being filled and a finished one, and no second copy of the
    read plan to keep in step with the server's.
    """
    import pyarrow as pa
    import pyarrow.flight as flight

    info = flight.FlightInfo(
        schema=pa.schema([]),
        descriptor=flight.FlightDescriptor.for_command(descriptor.SerializeToString()),
        endpoints=[],
        total_records=-1,
        total_bytes=-1,
    )
    return SerializedTensor(
        location=location, auth_token=auth_token, flight_info=info.serialize()
    )


#: The tensor server's scratch source, which every server with a ``write_dir``
#: serves at this fixed id. Spelled out rather than imported so this module
#: stays importable without the tensor server installed.
SCRATCH_SOURCE_ID = "scratch"

#: How long a result is kept on it. Half an hour: a fast-return consumer reads
#: its result as soon as the job reports done, so anything still here long
#: after that is one nobody came back for.
RESULT_TTL_S = 30 * 60

_UNSAFE_IN_A_FIELD_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def _result_field_name(source_name: Optional[str]) -> str:
    """A field name for one result: the caller's word for it, plus a salt.

    Salted because a field is taken for as long as its tensor is served, so a
    second result under one name is refused rather than replacing it.
    """
    stem = _UNSAFE_IN_A_FIELD_NAME.sub(
        "-", (source_name or "").removeprefix("cache:")
    ).strip("-")
    return f"{stem or 'result'}-{os.urandom(4).hex()}"


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
        """Declare one result; answer the ``array_id`` it will be filled under.

        A result is a tensor added to the scratch source, through the same
        ``add_tensor`` boundary a remote client uses -- called in-process, so
        the Flight write path stays refused. Adding rather than registering a
        source of its own is what gives it a deadline (``RESULT_TTL_S``) and a
        producer: a source is shared, and the grant below covers this tensor
        alone.
        """
        chunk_shape = _uniform_chunk_shape(array_template)
        descriptor = TensorDescriptor(
            array_id=(
                f"cache://{SCRATCH_SOURCE_ID}/@fields/{_result_field_name(source_name)}"
            ),
            shape=list(array_template.shape),
            dtype=array_template.dtype.str,
            chunk_shape=list(chunk_shape),
            dim_labels=list(dim_labels) if dim_labels is not None else [],
        )
        answer = self._server.uploads.add_tensor(descriptor)
        # The capability: the result is readable by the caller that receives
        # this SerializedTensor (it rides in the auth_token) and by nobody
        # else. It gates read-back without a server-wide secret, which this
        # server has none of.
        self._result(answer.array_id).capability_token = secrets.token_urlsafe(32)
        return answer.array_id, array_template, chunk_shape

    def _result(self, array_id: str):
        """The adapter serving one result, by the id ``add_tensor`` answered.

        A result is a tensor *attached* to the scratch source rather than a
        source of its own, so it is reached through the registry's
        attachments for its source.
        """
        source_id, _, field = array_id.partition("/")
        adapter = self._server.sources.attached(source_id, field)
        if adapter is None:
            raise ValueError(f"Result not found: {array_id}")
        return adapter

    def create_array(
        self,
        source_name: Optional[str],
        dim_labels: Optional[list],
        array_template: da.Array,
    ) -> tensor_proto.SerializedTensor:
        array_id, _, _ = self._register_array_template(
            array_template=array_template,
            source_name=source_name,
            dim_labels=dim_labels,
        )
        return self.to_serialized_tensor(array_id)

    def upload_array_chunks(
        self,
        array_id: str,
        endpoint: ChunkBounds,
        chunk: np.ndarray,
    ) -> None:
        """Write one chunk into a result this process declared."""
        import pyarrow.flight as flight
        from biopb_tensor_server.core.errors import UploadClosedError

        # The adapter counts the chunk and refuses once the upload is over; both
        # refusals surface as the exception type the wire path raises, so a
        # servicer's job discriminates on it the way a remote client would.
        try:
            self._result(array_id).write_chunk(endpoint, chunk)
        except UploadClosedError as e:
            raise flight.FlightCancelledError(str(e)) from e

    def finish(self, array_id: str) -> dict:
        """Seal a result: the output is complete and takes no further chunks.

        The counterpart to :meth:`discard`, and the only route to READY, which
        is what a consumer polling for the result waits on. A fast-return
        servicer calls this when its job succeeds, exactly as it calls
        ``discard`` when the job dies (biopb/biopb#1048).

        READY is one rung of the upload ladder rather than an operation of its
        own, so it goes through ``set_status``; the name stays because sealing
        is the only rung a servicer ever asks for.
        """
        from biopb_tensor_server.adapters._writable import UploadStatus

        return self._server.uploads.set_status(array_id, UploadStatus.READY)

    def get_upload_status(self, array_id: str) -> dict:
        return self._server.uploads.status(array_id)

    def discard(self, array_id: str, reason: str = "") -> dict:
        """Give up on a result: drop the source, leave a tombstone saying why.

        For a servicer whose background job died or was told to stop. Stopping
        the job itself is the servicer's own concern -- this disposes of the
        output it was going to fill, and makes any write still in flight fail
        with *reason* rather than with a missing source (biopb/biopb#1).
        """
        return self._server.uploads.discard(array_id, reason)

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
            The result's array_id, for use with to_serialized_tensor()
        """
        dask_array = _as_dask_array(array)
        array_id, normalized_array, chunk_shape = self._register_array_template(
            array_template=dask_array,
            source_name=source_name,
            dim_labels=dim_labels,
        )

        for bounds in _iter_chunk_bounds(normalized_array.shape, chunk_shape):
            chunk_data = normalized_array[_bounds_to_slices(bounds)].compute()
            self.upload_array_chunks(array_id, bounds, chunk_data)
        # Synchronous: every chunk is written by the time we get here, so this
        # is the one caller that can seal on its own behalf. `create_array`'s
        # producer fills the result later and finishes for itself.
        self.finish(array_id)

        logger.debug(
            "Created result %s: shape=%s, dtype=%s",
            array_id,
            list(normalized_array.shape),
            normalized_array.dtype.str,
        )
        return array_id

    def to_serialized_tensor(
        self,
        array_id: str,
        tensor_id: Optional[str] = None,
    ) -> tensor_proto.SerializedTensor:
        """Get SerializedTensor for a result, with rewritten location.

        auth_token carries the result's own capability token, so only this
        caller can read it.

        Args:
            array_id: The id ``create_source`` / ``create_array`` answered
            tensor_id: Unused; a result is one tensor

        Returns:
            SerializedTensor protobuf with external location
        """
        adapter = self._result(array_id)
        descriptor = adapter.get_tensor_descriptor()
        remaining = adapter.remaining_ttl()
        if remaining is not None:
            # What the handle is for: the consumer reads this result later, and
            # the one thing it cannot work out for itself is how much later.
            descriptor.ttl_seconds = remaining
        return _handle(
            descriptor,
            self._external_location,
            adapter.capability_token or "",
        )


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

    # Clear the previous run's results since a result's capability
    # token is minted in memory and would not survive restart.
    write_dir = cache_dir / "uploads"
    shutil.rmtree(write_dir, ignore_errors=True)

    # Clean stale lock file (from previous run/crash)
    lock_path = cache_dir / "lock"
    if lock_path.exists():
        try:
            lock_path.unlink()
            logger.debug(f"Removed stale cache lock: {lock_path}")
        except Exception as e:
            logger.warning(f"Could not remove stale lock: {e}")

    # Initialize cache manager singleton with the on-disk Arrow file cache
    cache_config = CacheConfig(
        file_cache_dir=cache_dir,
        file_max_segment_bytes=64 * 1024 * 1024,  # 64MB segments
        file_max_total_bytes=cache_size,
    )
    CacheManager.initialize(cache_config)

    # Bind to specified host (0.0.0.0 for external access)
    location = f"{tensor_host}:{tensor_port}"

    # The server-wide token is minted and *kept*. It exists so ``_authorize``
    # fails closed on every action arm except ``health`` and ``chunk_locate``.
    # Read-back is gated per result by its own capability token and unaffected.
    #
    # Read-only over Flight: Setting a ``write_dir`` is sufficient for the
    # in-process path to reaches through ``uploads`` with the wire verbs
    # still refused.
    #
    # No catalog (metadata_db=None): instead a result is addressed by the
    # array_id its SerializedTensor carries.
    tensor_server = TensorFlightServer(
        location,
        token=secrets.token_urlsafe(32),
        writable=False,
        write_dir=write_dir,
        scratch_ttl=RESULT_TTL_S,
        annotations_enabled=False,
    )

    # Unlike the CLI launcher, which defers mark_ready() until after catalog
    # scan; the embedded server has no catalog to scan, so it marks ready
    # immediately.
    tensor_server.mark_ready()

    # Start in background thread
    def _run_tensor_server():
        logger.info(f"Embedded tensor cache server started at grpc://{location}")
        tensor_server.serve()

    thread = threading.Thread(target=_run_tensor_server, daemon=True)
    thread.start()

    return tensor_server, f"grpc://{location}"


def start_embedded_cache(
    cache_dir: str,
    cache_size: str = "32GB",
    *,
    ip: str = "0.0.0.0",
    local: bool = False,
    tensor_port: int = 8817,
    tensor_external_location: Optional[str] = None,
) -> Optional[EmbeddedTensorCache]:
    """Start the embedded tensor server a remote deployment returns results through.

    Answers ``None`` where pyarrow cannot run, so the server serves inline
    results only.
    """
    if not _pyarrow_available():
        # No-SSE4.2/AVX build: pyarrow (hence the lazy/Flight side channel) is
        # unavailable. Do not start the tensor server -- it would crash. Lazy
        # (dask) requests will be cleanly rejected by the servicer instead.
        logger.warning(
            "Tensor cache (lazy data side channel) was requested (cache_dir) "
            "but pyarrow is not available -- this looks like a build for a CPU "
            "without SSE4.2/AVX. Disabling the tensor server; only eager image "
            "data is supported and lazy (dask) input/output will be rejected."
        )
        return None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # NOTE: parse_bytes reads "GB" as decimal 1e9 -- use "GiB" for the binary
    # 2**30 the old ad-hoc parser assumed.
    cache_bytes = parse_bytes(cache_size)

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

    logger.info(f"Tensor server listening on {bind_location}")
    logger.info(f"Tensor server advertised at {external_location}")
    # Create wrapper with location rewriting
    return EmbeddedTensorCache(
        tensor_server=tensor_server,
        external_location=external_location,
    )
