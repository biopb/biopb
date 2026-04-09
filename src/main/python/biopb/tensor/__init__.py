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

# Import base classes and utilities from base module
from biopb.tensor.base import (
    BackendAdapter,
    ChunkEndpoint,
    TensorReadPlan,
    ComputeBackendOptions,
    build_arrow_schema,
    configure_compute_backend,
    get_compute_backend_options,
    plan_tensor_read,
    resolve_chunk_data,
    _chunks_intersect,
    _encode_chunk_id,
    _decode_chunk_id,
)

# Import concrete adapters from their respective modules
from biopb.tensor.zarr import ZarrAdapter
from biopb.tensor.hdf5 import Hdf5Adapter
from biopb.tensor.tiff import OmeTiffAdapter, MultiFileOmeTiffAdapter
from biopb.tensor.ome_zarr import OmeZarrAdapter

# Import server and client
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
    # Base classes and utilities
    'BackendAdapter',
    'ChunkEndpoint',
    'TensorReadPlan',
    'ComputeBackendOptions',
    'build_arrow_schema',
    'configure_compute_backend',
    'get_compute_backend_options',
    'plan_tensor_read',
    'resolve_chunk_data',
    '_chunks_intersect',
    '_encode_chunk_id',
    '_decode_chunk_id',
    # Concrete adapters
    'ZarrAdapter',
    'Hdf5Adapter',
    'OmeTiffAdapter',
    'MultiFileOmeTiffAdapter',
    'OmeZarrAdapter',
    # Server
    'TensorFlightServer',
    'serve',
    # Client
    'TensorFlightClient',
]