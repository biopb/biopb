"""Upload lifecycle for a TensorFlightClient connection.

Extracted from :mod:`biopb.tensor.client` (issue #278 item C): source creation,
chunk writing, and upload-status polling are a self-contained concern that
reads the shared ``_ClientState`` for its connection and touches none of the
catalog / descriptor caches the read path keeps there. :class:`UploadSession`
owns that concern; ``TensorFlightClient`` holds one and delegates its public
upload methods to it.

It needs both halves of that state: the live connection for its own foreground
RPCs, and the plain ``(location, token, trust)`` triple for anything that rides
into a dask graph -- which is the read path's arrangement too (``_session``).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

from biopb.tensor._pool import _get_shared_call_options, _get_thread_client
from biopb.tensor._tls import NO_TLS, TlsTrust
from biopb.tensor.descriptor_pb2 import TensorDescriptor, UploadStatus as UploadStatusPb
from biopb.tensor.ticket_pb2 import (
    ChunkBounds,
    ChunkUpload,
    CreateSourceResult,
    FinishUpload,
    PutCommand,
)

if TYPE_CHECKING:  # import-time cycle-free; _session never imports this module
    from biopb.tensor._session import _ClientState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadHandle:
    """What ``create_source`` returns: the source, and the attempt at filling it.

    Two ids because they answer different questions. ``source_id`` names the
    object and is what a *reader* is given; ``session_id`` names this attempt
    and is what a write must carry, so that a second ``create_source`` for the
    same ``cache:`` name -- which takes the name over -- makes this session's
    writes fail loudly instead of landing in someone else's source
    (biopb/biopb#1048).

    Plain data, so it pickles into a dask graph with the rest of the target.
    """

    source_id: str
    session_id: str

    def __str__(self) -> str:
        return self.source_id


def _put_chunk(
    client: flight.FlightClient,
    call_options: flight.FlightCallOptions,
    source_id: str,
    bounds: ChunkBounds,
    data: np.ndarray,
    session_id: str = "",
) -> None:
    """One ``do_put``: open, write the batch, close, read the ack.

    Free of any session state, so the same code serves
    :meth:`UploadSession.upload_chunk` and a target that has been unpickled in a
    dask worker with no session to hand.
    """
    cmd = PutCommand(
        chunk=ChunkUpload(source_id=source_id, bounds=bounds, session_id=session_id)
    )
    desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
    schema = pa.schema([pa.field("data", pa.from_numpy_dtype(data.dtype))])

    writer, reader = client.do_put(desc, schema, options=call_options)
    batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])
    writer.write_batch(batch)
    writer.done_writing()
    writer.close()
    reader.read()
    logger.debug(f"upload_chunk: uploaded {data.nbytes} bytes to {source_id}")


class _UploadTarget:
    """A ``da.store`` target that ships each block as one ``do_put``.

    ``store`` hands over a block and the slices it occupies, which is exactly
    the ``ChunkBounds`` an upload wants -- so the whole adapter is that one
    translation, and the scheduling, the memory ordering and the sharing of
    common ancestors between blocks all stay dask's.

    **Holds connection parameters, never a connection.** ``store`` puts the
    target *into the graph*, so under a distributed scheduler it is pickled out
    to the workers -- and a ``FlightClient`` cannot be pickled at all. This is
    the read path's own arrangement (``_session._fetch_endpoints...``): carry
    the plain ``(location, token, trust)`` triple, and let each worker resolve
    it against this module's per-thread connection pool. A worker then dials
    once and every later block on that thread rides the same connection.

    A worker's writes then issue from its own process, beside the compute that
    produced the block.
    """

    __slots__ = (
        "_location",
        "_token",
        "_trust",
        "_source_id",
        "_session_id",
        "shape",
        "dtype",
    )

    def __init__(
        self,
        location: str,
        token: Optional[str],
        trust: Optional[TlsTrust],
        handle: "UploadHandle",
        shape: Sequence[int],
        dtype: np.dtype,
    ):
        self._location = location
        self._token = token
        self._trust = trust or NO_TLS
        self._source_id = handle.source_id
        self._session_id = handle.session_id
        # ``store`` reads these off the target to check it can hold the array.
        self.shape = tuple(shape)
        self.dtype = dtype

    def __setitem__(self, index: Tuple[slice, ...], value: np.ndarray) -> None:
        client = _get_thread_client(self._location, self._token, self._trust)
        call_options = _get_shared_call_options(self._location, self._token)
        bounds = ChunkBounds(
            start=[s.start for s in index], stop=[s.stop for s in index]
        )
        _put_chunk(
            client, call_options, self._source_id, bounds, value, self._session_id
        )


class UploadSession:
    """Source creation and chunk upload over one Flight connection.

    .. note:: Experimental. This whole API -- ``create_source`` / ``upload_array``
       / ``upload_zarr``, and chunk upload -- is experimental and its behavior
       may change. Upload-status polling lives on ``CatalogClient`` (it is a
       read of one descriptor field, not an upload operation).

    Takes the shared ``_ClientState`` its two sibling collaborators take
    (``CatalogClient``, ``ChunkFetcher``) and reads only the connection fields
    from it -- never the catalog / descriptor caches. ``TensorFlightClient``
    constructs one in its ``__init__`` and delegates its public upload API here.
    """

    def __init__(self, state: "_ClientState"):
        self._state = state

    def upload_array(
        self,
        arr: da.Array,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
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
        handle = self.create_source(
            source_name=source_name,
            shape=arr.shape,
            dtype=arr.dtype.str,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
            ome_metadata=ome_metadata,
        )

        self._store_chunks(handle, arr)
        # Sealing is what makes the source readable, so a whole-array upload
        # does it on the caller's behalf -- it is the one caller that knows,
        # from having written every block itself, that there is nothing more to
        # send. A caller driving `create_source` / `upload_chunk` by hand does
        # not, and finishes explicitly.
        self.finish_upload(handle)

        return handle.source_id

    def _store_chunks(self, handle: UploadHandle, arr: da.Array) -> None:
        """Hand the whole upload to dask as one graph.

        ``upload_array`` has already rechunked *arr* onto the upload grid, so
        one dask block is one chunk and ``store`` needs no alignment help.

        Why ``store`` rather than a loop that computes and ships each chunk
        itself (biopb/biopb#590): a per-chunk loop pays a graph optimization
        per chunk, and separate ``.compute()`` calls share no cache, so any
        task two chunks need is computed once *per chunk*. That is the ordinary
        case rather than a corner -- the rechunk above is exactly what makes
        one output chunk draw on a coarser shared source block.

        ``lock=False`` because the default exists for targets that cannot take
        a concurrent ``__setitem__`` (an h5py dataset). Each block writes a
        disjoint region and the server counts arrivals into a set keyed by
        chunk id, so locking would serialize the uploads and buy nothing.

        Nothing here names a scheduler. That is the caller's, as it is for
        every other dask surface this SDK returns: an attached distributed
        client gets the writes, and
        ``dask.config.set(scheduler="threads", num_workers=N)`` around the call
        is how a link that wants exactly N in flight -- or one -- says so.
        """
        target = _UploadTarget(
            self._state.location,
            self._state.token,
            self._state.tls_trust,
            handle,
            arr.shape,
            arr.dtype,
        )
        da.store(arr, target, lock=False)

    def upload_zarr(
        self,
        zarr_path: str,
        source_name: str,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
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
        )

    def create_source(
        self,
        source_name: str,
        shape: Sequence[int],
        dtype: str,
        chunk_shape: Sequence[int],
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> UploadHandle:
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
        results = self._state.client.do_action(action, options=self._state.call_options)
        try:
            result = next(results)
        except StopIteration as exc:
            raise RuntimeError("create_source: server returned no result") from exc

        created = CreateSourceResult.FromString(result.body.to_pybytes())
        logger.info(f"create_source: created {created.tensor_descriptor.array_id}")
        return UploadHandle(created.tensor_descriptor.array_id, created.session_id)

    def finish_upload(self, handle: UploadHandle) -> UploadStatusPb:
        """Backs TensorFlightClient.finish_upload; see that method for the full
        documentation."""
        req = FinishUpload(source_id=handle.source_id, session_id=handle.session_id)
        action = flight.Action("finish", req.SerializeToString())
        results = self._state.client.do_action(action, options=self._state.call_options)
        try:
            result = next(results)
        except StopIteration as exc:
            raise RuntimeError("finish: server returned no result") from exc
        status = UploadStatusPb.FromString(result.body.to_pybytes())
        logger.info(f"finish: sealed {handle.source_id}")
        return status

    def upload_chunk(
        self,
        handle: UploadHandle,
        bounds: ChunkBounds,
        data: np.ndarray,
    ) -> None:
        """Backs TensorFlightClient.upload_chunk; see that method for the full
        documentation."""
        _put_chunk(
            self._state.client,
            self._state.call_options,
            handle.source_id,
            bounds,
            data,
            handle.session_id,
        )
