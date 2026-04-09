"""Tensor storage framework on Arrow Flight.

This package provides TensorStore-like framework built on Apache Arrow Flight
for efficient multi-dimensional array storage and retrieval.

Key components:
- TensorFlightClient: Client for accessing tensors from a TensorFlightServer
- Proto messages: TensorTicket, ChunkBounds, TensorDescriptor, SliceHint

Note: Server components have been moved to the biopb-tensor-server package.
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

# Import client
from biopb.tensor.client import TensorFlightClient

__all__ = [
    # Proto messages
    'TensorTicket',
    'ChunkBounds',
    'TensorDescriptor',
    'SliceHint',
    'TensorCriteria',
    'TensorReadOptions',
    'TensorWriteOptions',
    # Client
    'TensorFlightClient',
]