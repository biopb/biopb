"""Tensor storage framework on Arrow Flight.

This package provides a TensorStore-like framework built on Apache Arrow Flight
for efficient multi-dimensional array storage and retrieval.

Key components:
- BackendAdapter: Abstract interface for storage backends
- ZarrAdapter, Hdf5Adapter, OmeTiffAdapter: Concrete implementations
- TensorFlightServer: Flight server for tensor storage
- Proto messages: TensorTicket, ChunkBounds, TensorDescriptor, SliceHint
"""

# Import proto-generated classes with explicit paths
from biopb.tensor.ticket_pb2 import TensorTicket, ChunkBounds
from biopb.tensor.descriptor_pb2 import (
    TensorDescriptor,
    SliceHint,
    TensorCriteria,
    TensorReadOptions,
    TensorWriteOptions,
)

from biopb.tensor.adapter import (
    BackendAdapter,
    ChunkEndpoint,
    ZarrAdapter,
    Hdf5Adapter,
    OmeTiffAdapter,
)
from biopb.tensor.server import (
    TensorFlightServer,
    serve,
)
from biopb.tensor.client import (
    TensorFlightClient,
)

__all__ = [
    # Proto messages
    'TensorTicket',
    'ChunkBounds',
    'TensorDescriptor',
    'SliceHint',
    'TensorCriteria',
    'TensorReadOptions',
    'TensorWriteOptions',
    # Adapters
    'BackendAdapter',
    'ChunkEndpoint',
    'ZarrAdapter',
    'Hdf5Adapter',
    'OmeTiffAdapter',
    # Server
    'TensorFlightServer',
    'serve',
    # Client
    'TensorFlightClient',
]