"""Python client for TensorFlight server.

This module provides a lazy numpy-like array interface using dask.array
for accessing tensors stored in a Flight server.

Features:
- Lazy chunk loading via dask.array
- LRU caching via cachey
- Numpy-compatible slicing and operations
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import threading

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import dask.array as da
from dask.delayed import delayed
from cachey import Cache

from biopb.tensor.ticket_pb2 import TensorTicket, ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint, TensorReadOptions


class TensorFlightClient:
    """Client for accessing tensors from a TensorFlightServer.

    This client provides lazy, cached access to multi-dimensional arrays
    stored in a Flight server.

    Usage:
        client = TensorFlightClient('grpc://localhost:8815')
        arr = client.get_array('my-tensor')  # Returns dask.array
        data = arr[0:100, 0:100].compute()   # Load slice

    Note:
        This client is NOT compatible with dask.distributed. The FlightClient
        is not pickle-serializable, which causes task serialization to fail
        when the scheduler tries to send tasks to workers. Use with the local
        dask scheduler only.

        For distributed use, consider:
        - Creating connections inside delayed functions (per-task overhead)
        - Using a shared cache layer (e.g., Redis, memcached)
        - Implementing a custom dask WorkerPlugin for connection pooling
    """

    def __init__(
        self,
        location: str = 'grpc://localhost:8815',
        cache_bytes: int = 1_000_000_000,  # 1GB default
    ):
        """Initialize the Flight client.

        Args:
            location: Flight server location
            cache_bytes: Maximum bytes for chunk cache (default 1GB)
        """
        self._client = flight.FlightClient(location)
        self._cache = Cache(available_bytes=cache_bytes)
        self._descriptors: Dict[str, TensorDescriptor] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_lock = threading.Lock()

    def list_tensors(self) -> List[str]:
        """List available tensor IDs.

        Returns:
            List of tensor identifiers
        """
        tensor_ids = []
        for info in self._client.list_flights():
            desc = TensorDescriptor.FromString(info.descriptor.command)
            tensor_ids.append(desc.array_id)
            self._descriptors[desc.array_id] = desc
        return tensor_ids

    def get_descriptor(self, array_id: str) -> TensorDescriptor:
        """Get tensor metadata.

        Args:
            array_id: Tensor identifier

        Returns:
            TensorDescriptor with shape, dtype, chunk_shape
        """
        if array_id in self._descriptors:
            return self._descriptors[array_id]

        # Fetch via list_flights
        for info in self._client.list_flights():
            desc = TensorDescriptor.FromString(info.descriptor.command)
            self._descriptors[desc.array_id] = desc
            if desc.array_id == array_id:
                return desc

        raise ValueError(f"Tensor not found: {array_id}")

    def get_metadata(self, array_id: str) -> dict:
        """Get OME-compatible metadata for a tensor.

        Args:
            array_id: Tensor identifier

        Returns:
            Parsed metadata_json (multiscales, axes, omero, etc.)
            per OME-NGFF schema, or empty dict if no metadata.
        """
        import json

        desc = self.get_descriptor(array_id)
        if desc.metadata_json:
            return json.loads(desc.metadata_json)
        return {}

    def get_array(
        self,
        array_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
        read_options: Optional[TensorReadOptions] = None,
    ) -> da.Array:
        """Get a lazy dask array for a tensor.

        Args:
            array_id: Tensor identifier
            slice_hint: Optional slice tuple to filter chunks
            scale_hint: Optional per-dimension integer downsampling factors
            reduction_method: Optional dynamic reduction method for scaled reads
            read_options: Optional explicit read options proto

        Returns:
            dask.array with lazy chunk loading
        """
        # Get tensor descriptor
        desc = self.get_descriptor(array_id)

        # Convert slice_hint to SliceHint proto
        slice_hint_proto = None
        if slice_hint is not None:
            starts = []
            stops = []
            for s in slice_hint:
                starts.append(s.start if s.start is not None else 0)
                stops.append(s.stop if s.stop is not None else desc.shape[len(starts) - 1])
            slice_hint_proto = SliceHint(start=starts, stop=stops)

        if read_options is not None and (scale_hint is not None or reduction_method is not None):
            raise ValueError("Pass either read_options or scale_hint/reduction_method, not both")

        if read_options is None and (scale_hint is not None or reduction_method is not None):
            read_options = TensorReadOptions(
                scale_hint=list(scale_hint or []),
                reduction_method=reduction_method or '',
            )

        # Build descriptor with request-scoped read hints
        desc_with_hint = TensorDescriptor(
            array_id=desc.array_id,
            dim_labels=desc.dim_labels,
            shape=desc.shape,
            chunk_shape=desc.chunk_shape,
            dtype=desc.dtype,
        )
        if slice_hint_proto is not None:
            desc_with_hint.slice_hint.CopyFrom(slice_hint_proto)
        if read_options is not None:
            desc_with_hint.read_options.CopyFrom(read_options)

        # Get flight info
        flight_desc = flight.FlightDescriptor.for_command(desc_with_hint.SerializeToString())
        info = self._client.get_flight_info(flight_desc)
        response_desc = TensorDescriptor.FromString(info.descriptor.command)

        # Parse endpoints into chunk info
        chunks = []
        chunk_bounds_list = []

        for endpoint in info.endpoints:
            # Parse ticket
            ticket = TensorTicket.FromString(endpoint.ticket.ticket)

            # Parse chunk bounds from app_metadata
            bounds = ChunkBounds.FromString(endpoint.app_metadata)

            chunks.append(ticket.chunk_id)
            chunk_bounds_list.append(bounds)

        # Build dask array from chunks
        return self._build_dask_array(
            desc=response_desc,
            chunks=chunks,
            chunk_bounds=chunk_bounds_list,
        )

    def _build_dask_array(
        self,
        desc: TensorDescriptor,
        chunks: List[bytes],
        chunk_bounds: List[ChunkBounds],
    ) -> da.Array:
        """Build a dask array from chunk info.

        Args:
            desc: Tensor descriptor
            chunks: List of chunk IDs
            chunk_bounds: List of chunk bounds

        Returns:
            dask.array with lazy chunk loading
        """
        shape = tuple(desc.shape)
        dtype = np.dtype(desc.dtype)

        # Determine dask chunk sizes from the actual chunk bounds
        # We need to build a grid of chunks

        # Create a mapping from chunk index to chunk_id and bounds
        chunk_map = {}
        axis_starts = [sorted({int(bounds.start[axis]) for bounds in chunk_bounds}) for axis in range(len(shape))]
        axis_index_maps = [
            {start: index for index, start in enumerate(starts)}
            for starts in axis_starts
        ]
        for chunk_id, bounds in zip(chunks, chunk_bounds):
            # Compute chunk index from bounds
            chunk_idx = tuple(
                axis_index_maps[d][int(bounds.start[d])]
                for d in range(len(shape))
            )
            chunk_map[chunk_idx] = (chunk_id, bounds)

        # Create delayed function to fetch chunk
        # Cache lookup happens INSIDE this function at compute time
        def fetch_chunk(chunk_id: bytes, bounds: ChunkBounds) -> np.ndarray:
            """Fetch and cache a chunk from the server."""
            cache_key = chunk_id.hex()
            cached = self._cache.get(cache_key)
            if cached is not None:
                with self._cache_lock:
                    self._cache_hits += 1
                return cached

            with self._cache_lock:
                self._cache_misses += 1

            # Fetch from server
            ticket = TensorTicket(chunk_id=chunk_id)
            reader = self._client.do_get(flight.Ticket(ticket.SerializeToString()))

            # Read all data from the stream
            table = reader.read_all()

            # Convert to numpy
            arr = table.column(0).to_numpy()

            # Reshape to chunk shape
            chunk_shape = tuple(stop - start for start, stop in zip(bounds.start, bounds.stop))
            arr = arr.reshape(chunk_shape)

            # Cache the result
            self._cache.put(cache_key, arr, cost=arr.nbytes)

            return arr

        # Build dask array using da.from_delayed for each chunk
        # The fetch_chunk function is called at compute time, not graph-build time
        # Find the grid dimensions
        ndim = len(shape)
        grid_shape = tuple(
            max(idx[d] + 1 for idx in chunk_map.keys())
            for d in range(ndim)
        )

        # Create block array
        blocks = np.empty(grid_shape, dtype=object)
        for chunk_idx, (chunk_id, bounds) in chunk_map.items():
            chunk_shape = tuple(stop - start for start, stop in zip(bounds.start, bounds.stop))
            # Use dask.delayed to wrap the fetch function
            # The actual fetch + cache lookup happens at .compute() time
            delayed_arr = da.from_delayed(
                delayed(fetch_chunk)(chunk_id, bounds),
                shape=chunk_shape,
                dtype=dtype,
            )
            blocks[chunk_idx] = delayed_arr

        if blocks.size == 0:
            raise ValueError("No chunks found")

        # Use da.block to combine chunks into a single array
        return da.block(blocks.tolist())

    def close(self):
        """Close the Flight client."""
        self._client.close()

    def cache_info(self) -> Dict:
        """Return cache statistics.

        Returns:
            Dictionary with cache size and item count
        """
        import threading
        # cachey doesn't expose hits/misses, so we track them ourselves
        return {
            "size_bytes": self._cache.total_bytes,
            "max_bytes": self._cache.available_bytes,
            "item_count": len(self._cache.data),
            "hits": getattr(self, "_cache_hits", 0),
            "misses": getattr(self, "_cache_misses", 0),
        }

    def cache_clear(self):
        """Clear the chunk cache."""
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()