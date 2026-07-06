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
from biopb.tensor.descriptor_pb2 import (
    FlightCmd,
    MetadataQueryOption,
    ResolveProgress,
    SliceHint,
    TensorCriteria,
    TensorDescriptor,
    TensorReadOption,
    TensorWriteOptions,
    WarmProgress,
)
from biopb.tensor.serialized_pb2 import SerializedEndpoint, SerializedTensor
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket

# Import client lazily. biopb.tensor.client imports pyarrow at module load, and
# pyarrow's compiled SSE4.2 baseline raises SIGILL on pre-SSE4.2 CPUs (e.g. old
# AMD Opterons). PEP 562 module __getattr__ keeps `from biopb.tensor import
# TensorFlightClient` working while deferring the pyarrow import until the client
# (i.e. the lazy/Flight data path) is actually used.
_LAZY_CLIENT_EXPORTS = (
    "TensorFlightClient",
    "make_debug_serialized_tensor",
    "ResolveCancelled",
)


def __getattr__(name):
    if name in _LAZY_CLIENT_EXPORTS:
        from biopb.tensor import client

        for attr in _LAZY_CLIENT_EXPORTS:
            globals()[attr] = getattr(client, attr)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Proto messages
    "TensorTicket",
    "ChunkBounds",
    "TensorDescriptor",
    "SliceHint",
    "TensorCriteria",
    "TensorWriteOptions",
    "FlightCmd",
    "TensorReadOption",
    "MetadataQueryOption",
    "ResolveProgress",
    "WarmProgress",
    "SerializedTensor",
    "SerializedEndpoint",
    # Client
    "TensorFlightClient",
    "make_debug_serialized_tensor",
    "ResolveCancelled",
]
