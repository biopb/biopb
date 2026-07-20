"""Upload lifecycle for a TensorFlightClient connection.

Extracted from :mod:`biopb.tensor.client` (issue #278 item C): source creation,
chunk writing, and upload-status polling are a self-contained concern that
depends only on the Flight connection (client + call options) -- not on the
catalog / descriptor caches the read path shares. :class:`UploadSession` owns
that concern; ``TensorFlightClient`` holds one and delegates its public upload
methods to it.
"""

import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.serialized_pb2 import SerializedTensor
from biopb.tensor.ticket_pb2 import ChunkBounds, ChunkUpload

logger = logging.getLogger(__name__)


def _upload_source_id_from_pb(pb: SerializedTensor) -> str:
    """Extract upload-status source_id from a registration-first SerializedTensor."""
    source_id = pb.tensor_descriptor.array_id
    if not source_id:
        raise ValueError("SerializedTensor tensor_descriptor.array_id is required")
    return source_id


class UploadSession:
    """Source creation, chunk upload, and upload-status polling over one Flight
    connection.

    Holds only the connection handles (``FlightClient`` + ``FlightCallOptions``);
    it never touches the catalog / descriptor caches. ``TensorFlightClient``
    constructs one in its ``__init__`` and delegates its public upload API here.
    """

    def __init__(
        self, client: flight.FlightClient, call_options: flight.FlightCallOptions
    ):
        self._client = client
        self._call_options = call_options

    def upload_array(
        self,
        arr: da.Array,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Upload dask array to server.

        Args:
            arr: Dask array to upload
            source_name: Source identifier format:
                - "cache:my-name" → cache-backed (ephemeral)
                - "cache:" → cache-backed with server-generated name
                - "ome_zarr:my-name" → zarr-backed (persistent)
                - "ome_zarr:" → zarr-backed with server-generated name
            chunk_shape: Override chunk shape. If None, uses arr.chunksize with
                         automatic rechunking if chunks are non-uniform.
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict

        Returns:
            source_id of created source (e.g., "cache_abc123" or "ome_zarr_def456")
        """
        # Determine target chunk shape
        if chunk_shape is None:
            chunk_shape = arr.chunksize

            # Check if dask chunks are non-uniform
            needs_rechunk = not all(
                len(set(arr.chunks[d])) == 1 for d in range(arr.ndim)
            )

            if needs_rechunk:
                uniform_chunks = tuple(
                    max(arr.chunks[d]) if arr.chunks[d] else arr.shape[d]
                    for d in range(arr.ndim)
                )
                arr = arr.rechunk(uniform_chunks)
                chunk_shape = uniform_chunks
        else:
            if tuple(chunk_shape) != tuple(arr.chunksize):
                arr = arr.rechunk(tuple(chunk_shape))

        # Create source
        source_id = self.create_source(
            source_name=source_name,
            shape=arr.shape,
            dtype=arr.dtype.str,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
            ome_metadata=ome_metadata,
        )

        # Upload chunks
        ndim = arr.ndim
        chunk_shape_tuple = tuple(chunk_shape)
        chunks_per_dim = [
            (arr.shape[d] + chunk_shape_tuple[d] - 1) // chunk_shape_tuple[d]
            for d in range(ndim)
        ]

        for chunk_idx in product(*(range(n) for n in chunks_per_dim)):
            chunk_start = [
                idx * chunk_shape_tuple[d] for d, idx in enumerate(chunk_idx)
            ]
            chunk_stop = [
                min((idx + 1) * chunk_shape_tuple[d], arr.shape[d])
                for d, idx in enumerate(chunk_idx)
            ]

            bounds = ChunkBounds(start=chunk_start, stop=chunk_stop)

            slices = tuple(
                slice(chunk_start[d], chunk_stop[d]) for d in range(arr.ndim)
            )
            chunk_data = arr[slices].compute()
            self.upload_chunk(source_id, bounds, chunk_data)

        return source_id

    def upload_zarr(
        self,
        zarr_path: str,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Upload local zarr to server.

        Args:
            zarr_path: Path to local zarr directory
            source_name: Source identifier format:
                - "cache:my-name" → cache-backed (ephemeral)
                - "cache:" → cache-backed with server-generated name
                - "ome_zarr:my-name" → zarr-backed (persistent)
                - "ome_zarr:" → zarr-backed with server-generated name
            chunk_shape: Override chunk shape. If None, uses zarr's chunk shape.
            dim_labels: Optional dimension labels (read from zarr if not provided)
            ome_metadata: Optional OME metadata (read from zarr if not provided)

        Returns:
            source_id of created source (e.g., "cache_abc123" or "ome_zarr_def456")
        """
        import zarr

        arr = zarr.open_array(zarr_path, mode="r")

        # Read metadata from local zarr if not provided
        zattrs_path = Path(zarr_path) / ".zattrs"
        if zattrs_path.exists():
            with open(zattrs_path) as f:
                zattrs = json.load(f)
            if ome_metadata is None and "multiscales" in zattrs:
                ome_metadata = zattrs
            if dim_labels is None and "multiscales" in zattrs:
                axes = zattrs["multiscales"][0].get("axes", [])
                dim_labels = [
                    ax.get("name") if isinstance(ax, dict) else str(ax) for ax in axes
                ]

        dask_arr = da.from_zarr(zarr_path)
        effective_chunk_shape = chunk_shape or arr.chunks

        return self.upload_array(
            dask_arr,
            source_name=source_name,
            chunk_shape=effective_chunk_shape,
            dim_labels=dim_labels,
            ome_metadata=ome_metadata,
        )

    def create_source(
        self,
        source_name: str,
        shape: Sequence[int],
        dtype: str,
        chunk_shape: Sequence[int],
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> str:
        """Create source on server (internal).

        Args:
            source_name: "cache:name" → cache-backed; "ome_zarr:name" → zarr-backed
                         "cache:" or "ome_zarr:" → server-generated name
            shape: Array shape
            dtype: Data type string (numpy format)
            chunk_shape: Chunk size per dimension
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict

        Returns:
            source_id assigned by server
        """
        req_desc = TensorDescriptor(
            array_id=source_name,
            shape=list(shape),
            dtype=dtype,
            chunk_shape=list(chunk_shape),
            dim_labels=list(dim_labels or []),
            metadata_json=json.dumps(ome_metadata) if ome_metadata else "",
        )

        action = flight.Action("create_source", req_desc.SerializeToString())
        results = self._client.do_action(action, options=self._call_options)
        try:
            result = next(results)
        except StopIteration as exc:
            raise RuntimeError("create_source: server returned no result") from exc

        response_desc = TensorDescriptor.FromString(result.body.to_pybytes())
        logger.info(f"create_source: created {response_desc.array_id}")
        return response_desc.array_id

    def upload_chunk(
        self,
        source_id: str,
        bounds: ChunkBounds,
        data: np.ndarray,
    ) -> None:
        """Upload single chunk (internal).

        Args:
            source_id: Source identifier
            bounds: Chunk start/stop coordinates
            data: Numpy array with chunk data
        """
        upload = ChunkUpload(
            source_id=source_id,
            bounds=bounds,
        )

        desc = flight.FlightDescriptor.for_command(upload.SerializeToString())
        schema = pa.schema([pa.field("data", pa.from_numpy_dtype(data.dtype))])

        writer, reader = self._client.do_put(desc, schema, options=self._call_options)
        batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])
        writer.write_batch(batch)
        writer.done_writing()
        writer.close()
        reader.read()
        logger.debug(f"upload_chunk: uploaded {data.nbytes} bytes to {source_id}")

    def get_upload_status(self, source_id: str) -> Dict[str, Any]:
        """Get upload status for a writable source.

        Args:
            source_id: Source identifier returned by create_source()

        Returns:
            Dictionary with source_id, state, expected_chunks, and uploaded_chunks.
        """
        action = flight.Action("upload_status", source_id.encode("utf-8"))
        results = self._client.do_action(action, options=self._call_options)
        for result in results:
            return json.loads(result.body.to_pybytes())
        return {
            "source_id": source_id,
            "state": "UNKNOWN",
            "expected_chunks": 0,
            "uploaded_chunks": 0,
        }

    def get_upload_status_pb(self, pb: SerializedTensor) -> Dict[str, Any]:
        """Get upload status for a registration-first SerializedTensor handle.

        This helper is intended for cache-backed handles returned before upload
        completion, where tensor_descriptor.array_id is the source identifier.

        Args:
            pb: SerializedTensor handle returned by a registration-first flow.

        Returns:
            Dictionary with source_id, state, expected_chunks, and uploaded_chunks.
        """
        return self.get_upload_status(_upload_source_id_from_pb(pb))

    def wait_for_upload_ready(
        self,
        source_id: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll upload status until the source reports READY.

        Args:
            source_id: Source identifier returned by create_source().
            timeout_seconds: Maximum time to wait before timing out.
            poll_interval_seconds: Delay between status checks.

        Returns:
            Final upload status dictionary when READY.

        Raises:
            TimeoutError: If the upload does not reach READY within the timeout.
            RuntimeError: If the upload reports FAILED.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = self.get_upload_status(source_id)
            state = status.get("state")
            if state == "READY":
                return status
            if state == "FAILED":
                raise RuntimeError(f"Upload failed for source '{source_id}'")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for upload readiness for source '{source_id}'"
                )
            time.sleep(poll_interval_seconds)

    def wait_for_upload_ready_pb(
        self,
        pb: SerializedTensor,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll upload status until a registration-first SerializedTensor is READY."""
        return self.wait_for_upload_ready(
            _upload_source_id_from_pb(pb),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
