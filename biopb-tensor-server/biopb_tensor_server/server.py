"""Arrow Flight server for tensor storage.

This module implements a Flight server that exposes chunked multi-dimensional
arrays through the BackendAdapter interface.

The server supports:
- ListFlights: Browse available tensors
- GetFlightInfo: Get tensor metadata and chunk endpoints
- DoGet: Fetch individual chunk data
"""

from typing import Dict, Iterator, Optional

import pyarrow as pa
import pyarrow.flight as flight

from biopb.tensor.ticket_pb2 import (
    TensorTicket, ChunkBounds
)
from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.base import BackendAdapter, plan_tensor_read, resolve_chunk_data, _decode_chunk_id


class TensorFlightServer(flight.FlightServerBase):
    """Arrow Flight server for tensor storage.

    This server exposes multi-dimensional arrays through the Flight protocol,
    with each chunk represented as a separate FlightEndpoint.

    Usage:
        # Create an adapter for your data
        import zarr
        arr = zarr.open_array('data.zarr', mode='r')
        adapter = ZarrAdapter(arr, 'my-tensor')

        # Start the server
        server = TensorFlightServer('grpc://0.0.0.0:8815')
        server.register_tensor('my-tensor', adapter)
        server.serve()
    """

    def __init__(self, location: str = 'grpc://0.0.0.0:8815', **kwargs):
        """Initialize the Flight server.

        Args:
            location: Server location (e.g., 'grpc://0.0.0.0:8815')
            **kwargs: Additional arguments passed to FlightServerBase
        """
        super().__init__(location, **kwargs)
        self._tensors: Dict[str, BackendAdapter] = {}

    def register_tensor(self, array_id: str, adapter: BackendAdapter) -> None:
        """Register a tensor with the server.

        Args:
            array_id: Unique identifier for the tensor
            adapter: Backend adapter for the tensor
        """
        self._tensors[array_id] = adapter

    def unregister_tensor(self, array_id: str) -> None:
        """Unregister a tensor from the server.

        Args:
            array_id: Unique identifier for the tensor
        """
        self._tensors.pop(array_id, None)

    def _parse_descriptor(self, descriptor: flight.FlightDescriptor) -> TensorDescriptor:
        """Parse a TensorDescriptor from a FlightDescriptor.

        Args:
            descriptor: Flight descriptor with cmd bytes

        Returns:
            Parsed TensorDescriptor
        """
        return TensorDescriptor.FromString(descriptor.command)

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

    def list_flights(
        self,
        context: flight.ServerCallContext,
        criteria: bytes
    ) -> Iterator[flight.FlightInfo]:
        """List all available tensors.

        Args:
            context: Server call context
            criteria: Unused criteria bytes

        Yields:
            FlightInfo for each registered tensor
        """
        for array_id, adapter in self._tensors.items():
            descriptor = adapter.get_tensor_descriptor()
            schema = adapter.get_arrow_schema()

            # Create a FlightDescriptor for this tensor
            flight_descriptor = flight.FlightDescriptor.for_command(
                descriptor.SerializeToString()
            )

            # Create a single endpoint for the tensor (all chunks)
            # Clients will call GetFlightInfo with slice_hint to get chunk endpoints
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
            descriptor: Flight descriptor with TensorDescriptor

        Returns:
            FlightInfo with schema and chunk endpoints
        """
        import json

        tensor_desc = self._parse_descriptor(descriptor)

        adapter = self._tensors.get(tensor_desc.array_id)
        if adapter is None:
            raise flight.FlightServerError(f"Tensor not found: {tensor_desc.array_id}")

        read_plan = plan_tensor_read(adapter, tensor_desc)
        schema = adapter.get_arrow_schema(read_plan.descriptor)

        # Populate metadata_json in response descriptor
        metadata = adapter.get_metadata()
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

        # Decode array_id from chunk_id
        array_id, _ = _decode_chunk_id(tensor_ticket.chunk_id)

        # Get the adapter for this array
        adapter = self._tensors.get(array_id)
        if adapter is None:
            raise flight.FlightServerError(f"Tensor not found: {array_id}")

        # Read the chunk
        record_batch = resolve_chunk_data(adapter, tensor_ticket.chunk_id)
        return flight.RecordBatchStream(pa.Table.from_batches([record_batch]))


def serve(
    adapters: Dict[str, BackendAdapter],
    location: str = 'grpc://0.0.0.0:8815',
    **kwargs
) -> None:
    """Start a Flight server with the given adapters.

    Args:
        adapters: Dictionary mapping array_id to BackendAdapter
        location: Server location
        **kwargs: Additional arguments passed to FlightServerBase
    """
    server = TensorFlightServer(location, **kwargs)
    for array_id, adapter in adapters.items():
        server.register_tensor(array_id, adapter)

    print(f"Starting Flight server at {location}")
    server.serve()