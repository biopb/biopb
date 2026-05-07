"""Arrow Flight server for tensor storage.

This module implements a Flight server that exposes chunked multi-dimensional
arrays through the BackendAdapter interface.

The server supports:
- ListFlights: Browse available tensors
- GetFlightInfo: Get tensor metadata and chunk endpoints
- DoGet: Fetch individual chunk data
"""

import logging
from typing import Dict, Iterator, List, Optional

import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor, TensorSelection
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket

from biopb_tensor_server.base import BackendAdapter, decode_chunk_id
from biopb_tensor_server.cache import CacheManager

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
        location: str = 'grpc://0.0.0.0:8815',
        token: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Flight server.

        Args:
            location: Server location (e.g., 'grpc://0.0.0.0:8815')
            token: Bearer token required on every call.  ``None`` disables auth.
            **kwargs: Additional arguments passed to FlightServerBase
        """
        middleware = kwargs.pop("middleware", {})
        middleware.setdefault("auth", BearerAuthMiddlewareFactory(token))
        super().__init__(location, middleware=middleware, **kwargs)
        self._sources: Dict[str, BackendAdapter] = {}

    def register_source(self, source_id: str, adapter: BackendAdapter) -> None:
        """Register a data source with the server.

        Args:
            source_id: Unique identifier for the data source
            adapter: Backend adapter for the data source
        """
        self._sources[source_id] = adapter
        logger.debug(f"Registered source: {source_id}")

    def unregister_source(self, source_id: str) -> None:
        """Unregister a data source from the server.

        Args:
            source_id: Unique identifier for the data source
        """
        self._sources.pop(source_id, None)
        logger.debug(f"Unregistered source: {source_id}")

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

    def _get_adapter_for_source_id(self, source_id: str) -> Optional[BackendAdapter]:
        """Get adapter for a source_id, handling nested paths for level adapters.

        For nested source_ids like "parent/level", finds the parent adapter
        and delegates to its get_level_adapter method if available.

        Args:
            source_id: The source identifier (may contain "/" for nested paths)

        Returns:
            BackendAdapter for the source, or None if not found
        """
        # First try exact match
        adapter = self._sources.get(source_id)
        if adapter is not None:
            return adapter

        # Check for nested path (e.g., "ome-zarr/1" for level 1)
        if '/' in source_id:
            # Split to find parent and level path
            parts = source_id.split('/')
            parent_id = parts[0]
            level_path = '/'.join(parts[1:])

            parent_adapter = self._sources.get(parent_id)
            if parent_adapter is not None and hasattr(parent_adapter, 'get_level_adapter'):
                # Delegate to parent adapter's level adapter method
                try:
                    return parent_adapter.get_level_adapter(level_path)
                except Exception:
                    return None

        return None

    def _get_adapter_for_tensor(
        self,
        source_id: str,
        tensor_id: str
    ) -> Optional[BackendAdapter]:
        """Get adapter for a specific tensor within a source.

        Args:
            source_id: The data source identifier
            tensor_id: The tensor identifier within the source

        Returns:
            BackendAdapter for the specified tensor, or None if not found
        """
        source_adapter = self._get_adapter_for_source_id(source_id)
        if source_adapter is None:
            return None

        return source_adapter.get_tensor_adapter(tensor_id)

    def list_flights(
        self,
        context: flight.ServerCallContext,
        criteria: bytes
    ) -> Iterator[flight.FlightInfo]:
        """List all available data sources.

        Each flight represents a data source (which may contain multiple tensors).

        Args:
            context: Server call context
            criteria: Unused criteria bytes

        Yields:
            FlightInfo for each registered data source, with DataSourceDescriptor
        """
        for source_id, adapter in self._sources.items():
            source_desc = adapter.get_source_descriptor()
            schema = adapter.get_arrow_schema(source_desc.tensors[0] if source_desc.tensors else adapter.get_tensor_descriptor())

            # Create a FlightDescriptor for this source
            flight_descriptor = flight.FlightDescriptor.for_command(
                source_desc.SerializeToString()
            )

            # Create a single endpoint for listing (no specific tensor selected)
            endpoint = flight.FlightEndpoint(
                ticket=flight.Ticket(b''),  # Empty ticket for listing
                locations=[],
            )

            yield flight.FlightInfo(
                schema=schema,
                descriptor=flight_descriptor,
                endpoints=[endpoint],
                total_records=-1,
                total_bytes=-1,
            )

    def get_flight_info(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor
    ) -> flight.FlightInfo:
        """Get metadata and chunk endpoints for a tensor.

        Args:
            context: Server call context
            descriptor: Flight descriptor with TensorSelection

        Returns:
            FlightInfo with schema and chunk endpoints
        """
        import json

        selection = TensorSelection.FromString(descriptor.command)
        logger.debug(
            f"get_flight_info: source_id={selection.source_id}, tensor_id={selection.tensor_id}"
        )

        # Get tensor adapter for the specified source and tensor
        tensor_adapter = self._get_adapter_for_tensor(selection.source_id, selection.tensor_id)
        if tensor_adapter is None:
            logger.warning(f"Tensor not found: {selection.source_id}/{selection.tensor_id}")
            raise flight.FlightServerError(f"Tensor not found: {selection.source_id}/{selection.tensor_id}")

        # Build request descriptor for the specific tensor
        base_desc = tensor_adapter.get_tensor_descriptor()
        tensor_desc = TensorDescriptor(
            array_id=selection.tensor_id,
            dim_labels=base_desc.dim_labels,
            shape=base_desc.shape,
            chunk_shape=base_desc.chunk_shape,
            dtype=base_desc.dtype,
        )
        if selection.HasField('slice_hint'):
            tensor_desc.slice_hint.CopyFrom(selection.slice_hint)
            logger.debug(f"get_flight_info: slice_hint={list(selection.slice_hint.start)}-{list(selection.slice_hint.stop)}")
        if selection.HasField('read_options'):
            tensor_desc.read_options.CopyFrom(selection.read_options)

        read_plan = tensor_adapter.get_read_plan(tensor_desc)
        schema = tensor_adapter.get_arrow_schema(read_plan.descriptor)

        # Populate metadata_json in response descriptor
        metadata = tensor_adapter.get_metadata()
        if metadata:
            read_plan.descriptor.metadata_json = json.dumps(metadata)

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
            descriptor=flight.FlightDescriptor.for_command(read_plan.descriptor.SerializeToString()),
            endpoints=endpoints,
            total_records=-1,
            total_bytes=-1,
        )

    def do_get(
        self,
        context: flight.ServerCallContext,
        ticket: flight.Ticket
    ) -> flight.FlightDataStream:
        """Fetch a chunk's data.

        Args:
            context: Server call context
            ticket: Flight ticket with TensorTicket

        Returns:
            FlightDataStream with the chunk data
        """
        tensor_ticket = self._parse_ticket(ticket)
        logger.debug(f"do_get: chunk_id={tensor_ticket.chunk_id[:16]}...")

        # Decode array_id (source_id/tensor_id) from chunk_id
        array_id, *_ = decode_chunk_id(tensor_ticket.chunk_id)

        # Find the adapter for this chunk
        # The array_id in chunk_id may be:
        # - Just source_id (for single tensor sources)
        # - source_id/level_path (for OME-Zarr precomputed levels)
        # - source_id/tensor_id (for multifield sources)

        if '/' in array_id:
            parts = array_id.split('/')
            source_id = parts[0]
            rest = '/'.join(parts[1:])
            logger.debug(f"do_get: nested path source_id={source_id}, rest={rest}")

            source_adapter = self._sources.get(source_id)
            if source_adapter is None:
                logger.warning(f"Source not found: {source_id}")
                raise flight.FlightServerError(f"Source not found: {source_id}")

            # Check if this is a level adapter path (OME-Zarr)
            if hasattr(source_adapter, 'get_level_adapter'):
                # Use get_level_adapter for nested paths
                adapter = source_adapter.get_level_adapter(rest)
            else:
                # Use get_tensor_adapter for multifield paths
                adapter = source_adapter.get_tensor_adapter(rest)
        else:
            # Single source_id - find the adapter
            adapter = self._sources.get(array_id)
            if adapter is None:
                logger.warning(f"Tensor not found: {array_id}")
                raise flight.FlightServerError(f"Tensor not found: {array_id}")

        if adapter is None:
            logger.warning(f"Tensor not found: {array_id}")
            raise flight.FlightServerError(f"Tensor not found: {array_id}")

        # Get cache manager singleton (if initialized)
        cache_manager = CacheManager.get_instance()

        # Read the chunk (with caching for virtual chunks)
        record_batch = adapter.resolve_chunk_data(tensor_ticket.chunk_id, cache_manager)
        batch_size = sum(col.nbytes for col in record_batch.columns)
        logger.debug(f"do_get: returning {batch_size} bytes")
        return flight.RecordBatchStream(pa.Table.from_batches([record_batch]))


def serve(
    adapters: Dict[str, BackendAdapter],
    location: str = 'grpc://0.0.0.0:8815',
    **kwargs
) -> None:
    """Start a Flight server with the given adapters.

    Args:
        adapters: Dictionary mapping source_id to BackendAdapter
        location: Server location
        **kwargs: Additional arguments passed to FlightServerBase
    """
    server = TensorFlightServer(location, **kwargs)
    for source_id, adapter in adapters.items():
        server.register_source(source_id, adapter)

    print(f"Starting Flight server at {location}")
    server.serve()