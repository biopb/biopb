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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.serialized_pb2 import SerializedTensor
from biopb.tensor.ticket_pb2 import ChunkBounds, ChunkUpload

logger = logging.getLogger(__name__)

#: Chunk uploads in flight at once, when the caller names no ``max_workers``.
#:
#: Each *running* task holds one materialized chunk, so this doubles as the
#: memory bound: ``workers x chunk_nbytes``, or ~64 MiB at the 8 MiB transfer
#: cap. The work is a ``do_put`` round trip per chunk, so the useful setting
#: tracks link latency rather than core count -- eight is enough to keep a
#: remote server busy and costs a loopback one nothing.
_DEFAULT_UPLOAD_WORKERS = 8


def _chunk_bounds(
    shape: Sequence[int], chunk_shape: Sequence[int], chunk_idx: Sequence[int]
) -> ChunkBounds:
    """The half-open bounds of one chunk, clipped to *shape* at the far edge."""
    start: List[int] = [idx * chunk_shape[d] for d, idx in enumerate(chunk_idx)]
    stop: List[int] = [
        min((idx + 1) * chunk_shape[d], shape[d]) for d, idx in enumerate(chunk_idx)
    ]
    return ChunkBounds(start=start, stop=stop)


def _upload_source_id_from_pb(pb: SerializedTensor) -> str:
    """Extract upload-status source_id from a registration-first SerializedTensor."""
    source_id = pb.tensor_descriptor.array_id
    if not source_id:
        raise ValueError("SerializedTensor tensor_descriptor.array_id is required")
    return source_id


class UploadSession:
    """Source creation, chunk upload, and upload-status polling over one Flight
    connection.

    .. note:: Experimental. This whole API -- ``create_source`` / ``upload_array``
       / ``upload_zarr``, chunk upload, and upload-status polling -- is
       experimental and its behavior may change.

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
        max_workers: Optional[int] = None,
    ) -> str:
        """Backs TensorFlightClient.upload_array; see that method for the full
        documentation."""
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

        self._upload_chunks(
            source_id, arr, chunk_shape_tuple, chunks_per_dim, max_workers
        )

        return source_id

    def _upload_chunks(
        self,
        source_id: str,
        arr: da.Array,
        chunk_shape: Tuple[int, ...],
        chunks_per_dim: Sequence[int],
        max_workers: Optional[int],
    ) -> None:
        """Compute and upload every chunk of *arr*, several at a time.

        One task per chunk, each doing its own ``.compute()`` and its own
        ``do_put``. Both halves belong in the task: uploading serially pays a
        full round trip (open/write/done/close/read) per chunk with nothing
        overlapping it, and computing serially would then leave the link idle
        for the length of a graph execution. That serialization was the whole
        of biopb/biopb#590, and what it wastes is latency, not bandwidth.

        Submission is windowed rather than issued all at once. A *queued* task
        holds only its bounds -- ``arr`` is lazy -- so the window costs no
        memory and exists to keep a worker's next chunk waiting rather than
        idling for the submit that follows a completion. Memory is bounded by
        the *running* tasks, one materialized chunk each.

        Chunks may land out of order. The server counts them into a set keyed
        by chunk id (``UploadManager.mark_chunk``) and each one writes a
        disjoint region, so arrival order is not observable.
        """
        workers = (
            _DEFAULT_UPLOAD_WORKERS if max_workers is None else max(1, int(max_workers))
        )
        pending: Dict[Future, ChunkBounds] = {}

        def drain(limit: int) -> None:
            """Block until fewer than *limit* tasks are outstanding.

            Raises the first failure seen. The exception is re-raised as-is
            rather than wrapped: a caller's ``except flight.FlightError`` has to
            keep working across this change, so the type and traceback pass
            through untouched and the log line is what names the chunk.
            """
            while len(pending) >= limit:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for fut in done:
                    bounds = pending.pop(fut)
                    exc = fut.exception()
                    if exc is not None:
                        logger.error(
                            "upload_array: chunk at %s of %s failed",
                            list(bounds.start),
                            source_id,
                        )
                        raise exc

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="biopb-upload"
        ) as pool:
            try:
                for chunk_idx in product(*(range(n) for n in chunks_per_dim)):
                    drain(workers * 2)
                    bounds = _chunk_bounds(arr.shape, chunk_shape, chunk_idx)
                    pending[
                        pool.submit(self._compute_and_put, source_id, arr, bounds)
                    ] = bounds
                drain(1)
            except BaseException:
                # Drop the queued backlog before the pool's own shutdown waits on
                # it: a failed upload should stop, not finish uploading into a
                # source the caller is about to hear has failed. Tasks already
                # running cannot be cancelled, and are waited for.
                for fut in pending:
                    fut.cancel()
                raise

    def _compute_and_put(
        self, source_id: str, arr: da.Array, bounds: ChunkBounds
    ) -> None:
        """One chunk, end to end: materialize it, then ship it."""
        slices = tuple(
            slice(start, stop)
            for start, stop in zip(bounds.start, bounds.stop, strict=True)
        )
        self.upload_chunk(source_id, bounds, arr[slices].compute())

    def upload_zarr(
        self,
        zarr_path: str,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
        max_workers: Optional[int] = None,
    ) -> str:
        """Backs TensorFlightClient.upload_zarr; see that method for the full
        documentation."""
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
            max_workers=max_workers,
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
        """Backs TensorFlightClient.create_source; see that method for the full
        documentation."""
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
        """Backs TensorFlightClient.upload_chunk; see that method for the full
        documentation."""
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
        """Backs TensorFlightClient.get_upload_status; see that method for the full
        documentation."""
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
        """Backs TensorFlightClient.get_upload_status_pb; see that method for the full
        documentation."""
        return self.get_upload_status(_upload_source_id_from_pb(pb))

    def wait_for_upload_ready(
        self,
        source_id: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        """Backs TensorFlightClient.wait_for_upload_ready; see that method for the full
        documentation."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = self.get_upload_status(source_id)
            state = status.get("state")
            if state == "READY":
                return status
            if state == "UNKNOWN":
                # Nothing to wait for, so fail now instead of polling to the
                # timeout (biopb/biopb#109). The server records upload progress
                # when create_source() hands out the id, so UNKNOWN never means
                # "not started yet" -- it means there is no upload record at
                # all, which no amount of polling will change.
                raise ValueError(
                    f"The server tracks no upload for source '{source_id}' "
                    "(not an upload target, or its record was dropped by a "
                    "server restart or source removal)."
                )
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
        """Backs TensorFlightClient.wait_for_upload_ready_pb; see that method for the full
        documentation."""
        return self.wait_for_upload_ready(
            _upload_source_id_from_pb(pb),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
