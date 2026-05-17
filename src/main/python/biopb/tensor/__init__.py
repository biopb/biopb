"""Tensor storage framework on Arrow Flight.

This package provides TensorStore-like framework built on Apache Arrow Flight
for efficient multi-dimensional array storage and retrieval.

Key components:
- TensorFlightClient: Client for accessing tensors from a TensorFlightServer
- Proto messages: TensorTicket, ChunkBounds, TensorDescriptor, SliceHint
- CLI diagnostics: `biopb tensor` command for inspecting sources and tensors

The CLI module provides the `biopb tensor` command with four subcommands:
- query: List sources and tensors from a running server
- metadata: Inspect source metadata and tensor descriptors
- get: Download tensor data to file or stdout
- stats: Compute min/max/mean statistics for a tensor

Note: Server components have been moved to the biopb-tensor-server package.
"""

# Import proto-generated classes with explicit paths
from biopb.tensor.ticket_pb2 import TensorTicket, ChunkBounds
from biopb.tensor.descriptor_pb2 import (
    TensorDescriptor,
    SliceHint,
    TensorCriteria,
    TensorWriteOptions,
    FlightCmd,
    TensorReadOption,
    MetadataQueryOption,
)
from biopb.tensor.serialized_pb2 import SerializedTensor, SerializedEndpoint

# Import client
from biopb.tensor.client import TensorFlightClient

__all__ = [
    # Proto messages
    'TensorTicket',
    'ChunkBounds',
    'TensorDescriptor',
    'SliceHint',
    'TensorCriteria',
    'TensorWriteOptions',
    'FlightCmd',
    'TensorReadOption',
    'MetadataQueryOption',
    'SerializedTensor',
    'SerializedEndpoint',
    # Client
    'TensorFlightClient',
]