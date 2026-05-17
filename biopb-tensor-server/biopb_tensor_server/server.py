"""Arrow Flight server for tensor storage.

This module implements a Flight server that exposes chunked multi-dimensional
arrays through the BackendAdapter interface.

The server supports:
- ListFlights: Browse available tensors
- GetFlightInfo: Get tensor metadata and chunk endpoints
- DoGet: Fetch individual chunk data
- DoPut: Upload data (when writable mode enabled)
- Metadata queries: SQL queries against source catalog (via DuckDB)
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import threading
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor, FlightCmd, TensorReadOption, MetadataQueryOption
from biopb.tensor.ticket_pb2 import ChunkBounds, ChunkUpload, TensorTicket

from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.base import SourceAdapter, TensorAdapter, decode_chunk_id
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.metadata_db import MetadataDatabase, NumpyEncoder

logger = logging.getLogger(__name__)


class _NoopMiddleware(flight.ServerMiddleware):
    """Middleware instance that does nothing after a successful auth check."""

    def sending_headers(self) -> dict:
        return {}

    def call_completed(self, exception: Optional[Exception]) -> None:
        pass


class BearerAuthMiddlewareFactory(flight.ServerMiddlewareFactory):
    """Reject calls whose Authorization header does not match the Bearer token.

    Header value must be exactly ``Bearer <token>`` (case-sensitive).
    When *token* is ``None`` or empty the factory is a no-op (auth disabled).
    """

    def __init__(self, token: Optional[str]) -> None:
        self._expected = f"Bearer {token}" if token else None

    def start_call(
        self,
        info: flight.CallInfo,
        headers: dict,
    ) -> Optional[flight.ServerMiddleware]:
        if self._expected is None:
            return _NoopMiddleware()
        # Header values are lists; gRPC lowercases header names.
        values: List[str] = headers.get("authorization", [])
        bearer = values[0] if values else ""
        if bearer != self._expected:
            raise flight.FlightUnauthenticatedError("Invalid or missing Bearer token")
        return _NoopMiddleware()


class TensorFlightServer(flight.FlightServerBase):
    """Arrow Flight server for tensor storage.

    This server exposes multi-dimensional arrays through the Flight protocol,
    with each chunk represented as a separate FlightEndpoint.

    Supports multifield acquisitions where tensors within a data source
    have different shapes (e.g., MicroManager multi-position datasets).

    Usage:
        # Create an adapter for your data
        import zarr
        arr = zarr.open_array('data.zarr', mode='r')
        adapter = ZarrAdapter(arr, 'my-tensor')

        # Start the server
        server = TensorFlightServer('grpc://0.0.0.0:8815')
        server.register_source('my-tensor', adapter)
        server.serve()
    """

    def __init__(
        self,
        location: str = "grpc://0.0.0.0:8815",
        token: Optional[str] = None,
        writable: bool = False,
        write_dir: Optional[Path] = None,
        metadata_db: Optional[MetadataDatabase] = None,
        max_list_flights_results: int = 100000,
        **kwargs,
    ):
        """Initialize the Flight server.

        Args:
            location: Server location (e.g., 'grpc://0.0.0.0:8815')
            token: Bearer token required on every call.  ``None`` disables auth.
            writable: Enable write mode for source creation and data upload
            write_dir: Directory for zarr-backed uploaded sources (required if writable)
            metadata_db: MetadataDatabase instance for source filtering queries (optional)
            max_list_flights_results: Safety cap on list_flights() returned sources
            **kwargs: Additional arguments passed to FlightServerBase
        """
        middleware = kwargs.pop("middleware", {})
        middleware.setdefault("auth", BearerAuthMiddlewareFactory(token))
        super().__init__(location, middleware=middleware, **kwargs)
        self._sources: Dict[str, SourceAdapter] = {}
        self._sources_lock = threading.RLock()
        self._writable = writable
        self._write_dir = write_dir
        self._metadata_db: Optional[MetadataDatabase] = metadata_db
        self._max_list_flights_results = max_list_flights_results
        self._start_time: float = time.time()

    def register_source(self, source_id: str, adapter: SourceAdapter) -> None:
        """Register a data source with the server.

        Args:
            source_id: Unique identifier for the data source
            adapter: Source adapter for the data source
        """
        with self._sources_lock:
            self._sources[source_id] = adapter
        logger.debug(f"Registered source: {source_id}")

    def unregister_source(self, source_id: str) -> None:
        """Unregister a data source from the server.

        Args:
            source_id: Unique identifier for the data source
        """
        with self._sources_lock:
            self._sources.pop(source_id, None)
        logger.debug(f"Unregistered source: {source_id}")

    def _get_source_adapter(self, source_id: str) -> Optional[SourceAdapter]:
        """Thread-safe source lookup."""
        with self._sources_lock:
            return self._sources.get(source_id)

    def _get_sources_snapshot(self) -> List[Tuple[str, SourceAdapter]]:
        """Return a stable snapshot of registered sources for iteration."""
        with self._sources_lock:
            return list(self._sources.items())

    def _parse_ticket(self, ticket: flight.Ticket) -> TensorTicket:
        """Parse a TensorTicket from a Flight Ticket.

        Args:
            ticket: Flight ticket with ticket bytes

        Returns:
            Parsed TensorTicket
        """
        return TensorTicket.FromString(ticket.ticket)

    def _encode_metadata(self, bounds: ChunkBounds) -> bytes:
        """Encode ChunkBounds to bytes for app_metadata.

        Args:
            bounds: Chunk bounds to encode

        Returns:
            Serialized bytes
        """
        return bounds.SerializeToString()

    def _get_adapter_for_tensor(
        self, source_id: str, tensor_id: str
    ) -> Optional[TensorAdapter]:
        """Get adapter for a specific tensor within a source.

        Args:
            source_id: The data source identifier
            tensor_id: The tensor identifier within the source

        Returns:
            TensorAdapter for the specified tensor, or None if not found
        """
        source_adapter = self._get_source_adapter(source_id)
        if source_adapter is None:
            return None

        return source_adapter.get_tensor_adapter(tensor_id)

    def _get_adapter_for_chunk(self, chunk_id: bytes) -> Optional[TensorAdapter]:
        """Get adapter for a specific chunk based on its chunk_id.

        Args:
            chunk_id: The chunk identifier bytes

        Returns:
            TensorAdapter responsible for the chunk, or None if not found
        """
        array_id, *_ = decode_chunk_id(chunk_id)
        source_id, *rest = array_id.split("/")
        rest = "/".join(rest) if rest else None

        source_adapter = self._get_source_adapter(source_id)
        if source_adapter is None:
            return None

        # Check for level adapter (OME-Zarr) only when there's an explicit level path
        if rest is not None and hasattr(source_adapter, "get_level_adapter"):
            return source_adapter.get_level_adapter(rest)

        # Otherwise get tensor adapter (for virtual scaling or single-tensor sources)
        return source_adapter.get_tensor_adapter(rest)

    def list_actions(
        self,
        context: flight.ServerCallContext,
    ) -> List[flight.ActionType]:
        """List available actions on this server.

        Returns:
            List of ActionType objects describing available actions.
        """
        return [
            flight.ActionType("health", "Health check - returns server status JSON"),
            flight.ActionType("create_source", "Create a writable source from a TensorDescriptor request"),
        ]

    def do_action(
        self,
        context: flight.ServerCallContext,
        action: flight.Action,
    ) -> Iterator[bytes]:
        """Execute a custom action.

        Args:
            context: Server call context
            action: Action to execute

        Yields:
            Result bytes (JSON-encoded for health action)
        """
        if action.type == "health":
            uptime_seconds = int(time.time() - self._start_time)
            health_status = {
                "status": "SERVING",
                "source_count": len(self._get_sources_snapshot()),
                "metadata_db_enabled": self._metadata_db is not None,
                "writable": self._writable,
                "uptime_seconds": uptime_seconds,
            }
            yield json.dumps(health_status).encode("utf-8")
        elif action.type == "create_source":
            if not self._writable:
                raise flight.FlightUnauthenticatedError("Server not in write mode")

            req_desc = TensorDescriptor.FromString(action.body.to_pybytes())
            response_desc = self._create_source(req_desc)
            yield response_desc.SerializeToString()
        else:
            raise flight.FlightServerError(f"Unknown action: {action.type}")

    def list_flights(
        self, context: flight.ServerCallContext, criteria: bytes
    ) -> Iterator[flight.FlightInfo]:
        """List all available data sources.

        Each flight represents a data source (which may contain multiple tensors).

        Results are capped at `max_list_flights_results` for safety. Truncation
        is signaled via schema metadata on all returned FlightInfos.

        Args:
            context: Server call context
            criteria: Unused criteria bytes

        Yields:
            FlightInfo for each registered data source (up to max_list_flights_results)
        """
        source_items = self._get_sources_snapshot()
        total_sources = len(source_items)
        max_sources = self._max_list_flights_results
        returned_count = min(total_sources, max_sources)
        truncated = total_sources > max_sources

        if truncated:
            logger.warning(
                f"list_flights truncated: returning {max_sources} of {total_sources} sources"
            )

        # Build base schema metadata for truncation signaling
        base_metadata = {
            b"total_sources": str(total_sources).encode(),
            b"max_sources": str(max_sources).encode(),
            b"returned_sources": str(returned_count).encode(),
            b"truncated": str(truncated).encode(),
        }

        count = 0
        for source_id, adapter in source_items:
            if count >= max_sources:
                break

            source_desc = adapter.get_source_descriptor()

            # Build schema with truncation metadata
            schema = pa.schema([], metadata=base_metadata)

            # Create a FlightDescriptor for this source
            flight_descriptor = flight.FlightDescriptor.for_command(
                source_desc.SerializeToString()
            )

            # Create a single endpoint for listing (no specific tensor selected)
            endpoint = flight.FlightEndpoint(
                ticket=flight.Ticket(b""),  # Empty ticket for listing
                locations=[],
            )

            yield flight.FlightInfo(
                schema=schema,
                descriptor=flight_descriptor,
                endpoints=[endpoint],
                total_records=-1,
                total_bytes=-1,
            )
            count += 1

    def get_flight_info(
        self, context: flight.ServerCallContext, descriptor: flight.FlightDescriptor
    ) -> flight.FlightInfo:
        """Get metadata and chunk endpoints for a tensor.

        Args:
            context: Server call context
            descriptor: Flight descriptor with FlightCmd

        Returns:
            FlightInfo with schema and chunk endpoints
        """
        import json

        cmd = FlightCmd.FromString(descriptor.command)

        # Dispatch based on source_id
        if cmd.source_id == "__metadata_query__":
            # Metadata SQL query branch
            if cmd.HasField("metadata_query"):
                sql = cmd.metadata_query.sql
                logger.debug(
                    f"get_flight_info: metadata_query sql={sql[:100]}..."
                )
                if self._metadata_db is None:
                    raise flight.FlightServerError(
                        "Metadata database not enabled. Set metadata_db config to enable SQL queries."
                    )
                try:
                    return self._metadata_db.handle_query(sql)
                except ValueError as e:
                    # Query validation or execution failure
                    raise flight.FlightInternalError(f"Metadata query failed: {e}") from e
            else:
                raise flight.FlightServerError(
                    "Metadata query source_id but no MetadataQueryOption"
                )

        # Tensor read branch
        if not cmd.HasField("tensor_read"):
            raise flight.FlightServerError(
                f"Tensor read source_id '{cmd.source_id}' but no TensorReadOption"
            )

        read_opt = cmd.tensor_read
        source_id = cmd.source_id
        tensor_id = read_opt.tensor_id

        logger.debug(
            f"get_flight_info: source_id={source_id}, tensor_id={tensor_id}"
        )

        # Get tensor adapter for the specified source and tensor
        tensor_adapter = self._get_adapter_for_tensor(
            source_id, tensor_id
        )
        if tensor_adapter is None:
            logger.warning(
                f"Tensor not found: {source_id}/{tensor_id}"
            )
            raise flight.FlightServerError(
                f"Tensor not found: {source_id}/{tensor_id}"
            )

        # Build request descriptor for the specific tensor
        try:
            base_desc = tensor_adapter.get_tensor_descriptor()
            tensor_desc = TensorDescriptor(
                array_id=tensor_id,
                dim_labels=base_desc.dim_labels,
                shape=base_desc.shape,
                chunk_shape=base_desc.chunk_shape,
                dtype=base_desc.dtype,
            )
            if read_opt.HasField("slice_hint"):
                tensor_desc.slice_hint.CopyFrom(read_opt.slice_hint)
                logger.debug(
                    f"get_flight_info: slice_hint={list(read_opt.slice_hint.start)}-{list(read_opt.slice_hint.stop)}"
                )
            # Apply scale_hint and reduction_method directly to TensorDescriptor
            if read_opt.scale_hint:
                tensor_desc.scale_hint[:] = list(read_opt.scale_hint)
            if read_opt.reduction_method:
                tensor_desc.reduction_method = read_opt.reduction_method

            read_plan = tensor_adapter.get_read_plan(tensor_desc)
            schema = tensor_adapter.get_arrow_schema(read_plan.descriptor)

            # Populate metadata_json in response descriptor if requested
            if read_opt.with_metadata:
                source_adapter = self._get_source_adapter(source_id)
                raw_metadata = tensor_adapter.get_metadata()
                if raw_metadata and source_adapter is not None:
                    wrapped_metadata = {
                        "type": source_adapter._source_type,
                        "dim_label": list(read_plan.descriptor.dim_labels),
                        "metadata": raw_metadata,
                    }
                    read_plan.descriptor.metadata_json = json.dumps(
                        wrapped_metadata, cls=NumpyEncoder
                    )
        except (OSError, IOError, ValueError, json.JSONDecodeError) as e:
            raise flight.FlightInternalError(
                f"Metadata error for {source_id}: {e}"
            ) from e

        # Convert to FlightEndpoints
        endpoints = []
        for ce in read_plan.chunk_endpoints:
            ticket = TensorTicket(chunk_id=ce.chunk_id)
            endpoint = flight.FlightEndpoint(
                ticket=flight.Ticket(ticket.SerializeToString()),
                locations=[],
                app_metadata=self._encode_metadata(ce.bounds),
            )
            endpoints.append(endpoint)

        logger.debug(f"get_flight_info: returning {len(endpoints)} chunk endpoints")
        return flight.FlightInfo(
            schema=schema,
            descriptor=flight.FlightDescriptor.for_command(
                read_plan.descriptor.SerializeToString()
            ),
            endpoints=endpoints,
            total_records=-1,
            total_bytes=-1,
        )

    def do_get(
        self, context: flight.ServerCallContext, ticket: flight.Ticket
    ) -> flight.FlightDataStream:
        """Fetch a chunk's data or metadata query result.

        Args:
            context: Server call context
            ticket: Flight ticket with TensorTicket or metadata query ID

        Returns:
            FlightDataStream with the chunk data or query result
        """
        # Check for metadata query result by checking the ticket prefix
        ticket_bytes = ticket.ticket
        metadata_prefix = b"metadata-query-"
        if ticket_bytes.startswith(metadata_prefix):
            ticket_id = ticket_bytes.decode()
            logger.debug(f"do_get: metadata query result ticket={ticket_id}")
            if self._metadata_db is None:
                raise flight.FlightInternalError("Metadata database not enabled")
            result = self._metadata_db.get_pending_result(ticket_id)
            if result is None:
                # Internal error: pending result should exist if ticket was valid
                raise flight.FlightInternalError(
                    f"Metadata query result not found: {ticket_id}"
                )
            return flight.RecordBatchStream(result)

        tensor_ticket = self._parse_ticket(ticket)
        logger.debug(f"do_get: chunk_id={tensor_ticket.chunk_id[:16]}...")

        adapter = self._get_adapter_for_chunk(tensor_ticket.chunk_id)
        if adapter is None:
            raise flight.FlightServerError(
                f"Adapter not found for chunk_id: {tensor_ticket.chunk_id[:16]}..."
            )

        # Get cache manager singleton (if initialized)
        cache_manager = CacheManager.get_instance()

        # Read the chunk, using the configured cache backend when applicable.
        try:
            record_batch = adapter.resolve_chunk_data(
                tensor_ticket.chunk_id, cache_manager
            )
        except (OSError, IOError, ValueError) as e:
            # ValueError can be raised by bounds validation or parsing failures
            raise flight.FlightInternalError(
                f"I/O error reading chunk data: {e}"
            ) from e

        batch_size = sum(col.nbytes for col in record_batch.columns)
        logger.debug(f"do_get: returning {batch_size} bytes")

        # zoer copy wrapper - do _not_ convert to pa.Table!
        reader = pa.RecordBatchReader.from_batches(record_batch.schema, [record_batch])
        return flight.RecordBatchStream(reader)

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.MetadataRecordBatchReader,
        writer: flight.FlightMetadataWriter,
    ) -> None:
        """Handle source creation and chunk upload.

        Args:
            context: Server call context
            descriptor: Flight descriptor with command bytes
            reader: Flight data stream reader
            writer: Flight metadata writer for responses

        Raises:
            FlightUnauthenticatedError: If server not in write mode
            FlightServerError: If source/chunk creation fails
        """
        if not self._writable:
            raise flight.FlightUnauthenticatedError("Server not in write mode")

        command = descriptor.command

        # Try TensorDescriptor (source creation)
        try:
            req_desc = TensorDescriptor.FromString(command)
            if req_desc.shape and req_desc.dtype:
                self._handle_create_source(req_desc, writer)
                return
        except Exception:
            pass

        # Chunk upload - use ChunkUpload wrapper
        try:
            upload = ChunkUpload.FromString(command)
            self._handle_chunk_upload(upload, reader)
            return
        except Exception as e:
            raise flight.FlightServerError(f"Invalid upload command: {e}")

    def _create_source(self, req_desc: TensorDescriptor) -> TensorDescriptor:
        """Create source from TensorDescriptor and return its resolved descriptor.

        array_id format in request:
        - "cache:name" → cache-backed with given name
        - "cache:" → cache-backed with server-generated name
        - "ome_zarr:name" → zarr-backed with given name
        - "ome_zarr:" → zarr-backed with server-generated name

        Args:
            req_desc: TensorDescriptor request
        Returns:
            TensorDescriptor with resolved server-side source_id
        """
        array_id = req_desc.array_id

        if array_id.startswith("cache:"):
            # Cache-backed source
            provided_name = array_id[6:]  # After 'cache:'
            if provided_name:
                source_id = (
                    f"cache_{hashlib.sha256(provided_name.encode()).hexdigest()[:12]}"
                )
            else:
                source_id = f"cache_{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"

            ome_metadata = (
                json.loads(req_desc.metadata_json) if req_desc.metadata_json else {}
            )

            adapter = CachedSourceAdapter(
                source_id=source_id,
                shape=list(req_desc.shape),
                dtype=req_desc.dtype,
                chunk_shape=list(req_desc.chunk_shape),
                dim_labels=list(req_desc.dim_labels) if req_desc.dim_labels else None,
                ome_metadata=ome_metadata,
            )
            self.register_source(source_id, adapter)

            logger.info(f"Created cache-backed source: {source_id}")

        elif array_id.startswith("ome_zarr:"):
            # Zarr-backed source
            import zarr

            provided_name = array_id[9:]  # After 'ome_zarr:'
            zarr_name = (
                provided_name
                or f"upload_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}"
            )

            if self._write_dir is None:
                raise flight.FlightServerError(
                    "write_dir not configured for zarr-backed sources"
                )

            zarr_path = self._write_dir / f"{zarr_name}.zarr"
            zarr_path.mkdir(parents=True, exist_ok=True)

            store = zarr.DirectoryStore(str(zarr_path))
            arr = zarr.create(
                store=store,
                shape=req_desc.shape,
                dtype=req_desc.dtype,
                chunks=req_desc.chunk_shape,
            )

            # Write OME metadata
            if req_desc.metadata_json:
                zattrs = json.loads(req_desc.metadata_json)
            else:
                zattrs = self._build_minimal_ome_metadata(req_desc)

            with open(zarr_path / ".zattrs", "w") as f:
                json.dump(zattrs, f)

            source_id = f"ome_zarr_{hashlib.sha256(str(zarr_path.resolve()).encode()).hexdigest()[:12]}"

            adapter = OmeZarrAdapter(
                arr,
                source_id,
                list(req_desc.dim_labels) if req_desc.dim_labels else None,
            )

            self.register_source(source_id, adapter)

            logger.info(f"Created zarr-backed source: {source_id} at {zarr_path}")

        else:
            raise flight.FlightServerError(
                f"Invalid array_id format: {array_id}. Use 'cache:' or 'ome_zarr:' prefix"
            )

        return TensorDescriptor(
            array_id=source_id,
            dim_labels=req_desc.dim_labels,
            shape=req_desc.shape,
            chunk_shape=req_desc.chunk_shape,
            dtype=req_desc.dtype,
        )

    def _handle_create_source(
        self, req_desc: TensorDescriptor, writer: flight.FlightMetadataWriter
    ) -> None:
        """Backward-compatible DoPut handler for source creation."""
        writer.write(self._create_source(req_desc).SerializeToString())

    def _handle_chunk_upload(
        self, upload: ChunkUpload, reader: flight.MetadataRecordBatchReader
    ) -> None:
        """Write chunk data.

        For OmeZarr-backed sources: enforces chunk_bounds aligns with chunk_shape.
        For cache-backed sources: arbitrary bounds allowed.
        """
        table = reader.read_all()
        data_column = table.column(0)

        adapter = self._sources.get(upload.source_id)
        if adapter is None:
            raise flight.FlightServerError(f"Source not found: {upload.source_id}")

        bounds = upload.bounds
        expected_shape = tuple(
            stop - start for start, stop in zip(bounds.start, bounds.stop)
        )

        # Check chunk alignment for OmeZarr-backed sources
        if hasattr(adapter, "zarr_array"):
            data = data_column.to_numpy()
            if expected_shape:
                data = data.reshape(expected_shape)
            desc = adapter.get_tensor_descriptor()
            chunk_shape = list(desc.chunk_shape)

            # Verify start aligns to chunk_shape grid
            for d, (start, chunk_size) in enumerate(zip(bounds.start, chunk_shape)):
                if start % chunk_size != 0:
                    raise flight.FlightServerError(
                        f"Chunk start[{d}]={start} not aligned to chunk_shape[{d}]={chunk_size}"
                    )

            # Verify chunk size matches (or is edge chunk)
            actual_size = [
                stop - start for start, stop in zip(bounds.start, bounds.stop)
            ]
            for d, (actual, expected) in enumerate(zip(actual_size, chunk_shape)):
                if actual != expected and actual > expected:
                    raise flight.FlightServerError(
                        f"Chunk size[{d}]={actual} exceeds chunk_shape[{d}]={expected}"
                    )

            # Convert bounds to chunk_idx for zarr
            chunk_idx = tuple(int(s // cs) for s, cs in zip(bounds.start, chunk_shape))
            if hasattr(adapter, "write_chunk"):
                adapter.write_chunk(chunk_idx, data)
            else:
                raise flight.FlightServerError(f"Source does not support writes")

        elif isinstance(adapter, CachedSourceAdapter):
            # Cache-backed sources accept arbitrary bounds
            dtype = table.schema.field(0).type.to_pandas_dtype()
            adapter.write_chunk_arrow(bounds, data_column, expected_shape, dtype)

        else:
            raise flight.FlightServerError(f"Source type does not support writes")

        logger.debug(
            f"Uploaded chunk to {upload.source_id}: bounds={list(bounds.start)}-{list(bounds.stop)}"
        )

    def _build_minimal_ome_metadata(self, desc: TensorDescriptor) -> dict:
        """Build minimal OME-Zarr metadata from TensorDescriptor."""
        dim_labels = (
            list(desc.dim_labels)
            if desc.dim_labels
            else [f"dim{i}" for i in range(len(desc.shape))]
        )

        axes = []
        for i, label in enumerate(dim_labels):
            if label.lower() in ("x", "y", "z"):
                axes.append({"name": label, "type": "space"})
            elif label.lower() in ("c", "channel"):
                axes.append({"name": label, "type": "channel"})
            elif label.lower() in ("t", "time"):
                axes.append({"name": label, "type": "time"})
            else:
                axes.append({"name": label})

        return {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0] * len(desc.shape)}
                            ],
                        }
                    ],
                }
            ]
        }


def serve(
    adapters: Dict[str, SourceAdapter], location: str = "grpc://0.0.0.0:8815", **kwargs
) -> None:
    """Start a Flight server with the given adapters.

    Args:
        adapters: Dictionary mapping source_id to SourceAdapter
        location: Server location
        **kwargs: Additional arguments passed to FlightServerBase
    """
    server = TensorFlightServer(location, **kwargs)
    for source_id, adapter in adapters.items():
        server.register_source(source_id, adapter)

    print(f"Starting Flight server at {location}")
    server.serve()
