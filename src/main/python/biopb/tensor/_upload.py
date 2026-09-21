"""Upload lifecycle for a TensorFlightClient connection.

Extracted from :mod:`biopb.tensor.client` (issue #278 item C): declaring a
tensor and writing its chunks are a self-contained concern that reads the
shared ``_ClientState`` for its connection. :class:`UploadSession` owns that
concern; ``TensorFlightClient`` holds one and delegates its public upload
methods to it. Status polling is a read of one descriptor field and lives on
``CatalogClient``.

It needs both halves of that state: the live connection for its own foreground
RPCs, and the plain ``(location, token, trust)`` triple for anything that rides
into a dask graph -- which is the read path's arrangement too (``_session``).
"""

import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

from biopb.tensor._pool import _get_shared_call_options, _get_thread_client
from biopb.tensor._session import (
    _read_option,
    _tensor_read_cmd,
    _unknown_upload_status,
    _upload_status_dict,
    do_action_one_result,
    extra_info,
)
from biopb.tensor._tls import NO_TLS, TlsTrust
from biopb.tensor.descriptor_pb2 import (
    SliceHint,
    TensorDescriptor,
    UploadStatus as UploadStatusPb,
)
from biopb.tensor.ticket_pb2 import (
    ChunkBounds,
    PutCommand,
    RegisterSource,
    RegisterSourceResult,
    SetUploadStatus,
)

if TYPE_CHECKING:  # import-time cycle-free; _session never imports this module
    from biopb.tensor._session import CatalogClient, _ClientState

logger = logging.getLogger(__name__)


class UploadRefused(Exception):
    """A write or a transition reached an upload that takes no more chunks.

    ``state`` is the state the source is in -- ``"DISCARDED"`` (its producer
    gave up; ``reason`` says why) or ``"READY"`` (it was published, which seals
    it against further chunks). ``source_id`` names which, because under
    ``upload_array`` the chunks are in flight concurrently and this is one of
    N.

    Plain data in ``args``, so under a distributed scheduler it is raised on a
    worker and pickles back intact. A sibling of :class:`ResolveCancelled`.
    """

    def __init__(self, source_id: str, state: str, reason: str = "") -> None:
        super().__init__(source_id, state, reason)
        self.source_id = source_id
        self.state = state
        self.reason = reason

    def __str__(self) -> str:
        what = "discarded" if self.state == "DISCARDED" else "already finished"
        why = f": {self.reason}" if self.reason else ""
        return f"Upload {what} for source '{self.source_id}'{why}"


def _refused_from(exc: flight.FlightCancelledError) -> Optional[UploadRefused]:
    """The typed refusal a Flight error carries, or None if it is not one.

    The server puts ``{"reason": "upload_*", "state", "source_id", "detail"}``
    in ``extra_info`` for a discarded or sealed upload; any other cancelled
    call is somebody else's and passes through untouched.
    """
    info = extra_info(exc)
    if info is None or not str(info.get("reason", "")).startswith("upload_"):
        return None
    return UploadRefused(
        str(info.get("source_id", "")),
        str(info.get("state", "")),
        str(info.get("detail", "")),
    )


def _is_label_set(array_id: str) -> bool:
    """Whether *array_id* names a label set rather than a source of its own.

    The one upload kind whose unwritten chunks are meaningful *by declaration*:
    a label set is a zarr with fill value 0, so a chunk that never arrives reads
    back as background and skipping it is free (biopb/biopb#1059). Every
    published upload now reads its gaps as zeros, so skipping would be safe for
    the other kinds too -- but it would also stop reporting them: an all-zero
    array would upload nothing at all and land READY with ``uploaded_chunks``
    at 0. A set is where the caller is already writing a sparse mask and means
    it; a ``cache:`` tensor is not.
    """
    prefixed = ":" in array_id.partition("/")[0]
    return not prefixed and "/labels/" in array_id


def _state_value(state: Any) -> int:
    """The wire enum for a state given as a name, or already as the enum.

    Names, because that is the shape the status dicts this SDK returns already
    use (``status["state"] == "READY"``), so a caller reads a state and writes
    the same string back.
    """
    if isinstance(state, str):
        try:
            return UploadStatusPb.State.Value(state.upper())
        except ValueError as exc:
            raise ValueError(
                f"set_upload_status: {state!r} is not an upload state"
            ) from exc
    return int(state)


def _uniform_chunk_shape(arr: da.Array) -> Tuple[int, ...]:
    """The upload grid a dask array implies: its chunk size, per axis.

    Max per axis, so a ragged trailing chunk (the ordinary case) or an
    irregular chunking both yield one grid the array can be rechunked onto.
    """
    return tuple(
        int(max(axis_chunks)) if axis_chunks else int(dim)
        for axis_chunks, dim in zip(arr.chunks, arr.shape, strict=True)
    )


class _PlannedChunk(NamedTuple):
    """One endpoint of a write plan: where it goes, and the ticket that says so.

    ``start`` / ``stop`` are in the **tensor's** coordinates. An endpoint's
    ``app_metadata`` states them relative to the realized region instead --
    that is what a reader wants, since it is assembling an array of just that
    region -- so :func:`_plan_write` shifts them back by the realized origin.

    ``ticket`` is the endpoint's ticket bytes verbatim -- opaque, and the same
    bytes ``do_get`` takes once the upload is READY.
    """

    start: Tuple[int, ...]
    stop: Tuple[int, ...]
    ticket: bytes


def _plan_write(
    state: "_ClientState", array_id: str, slice_hint: Optional[SliceHint]
) -> List[_PlannedChunk]:
    """The chunks a write must send, as the server plans them.

    The server is the only authority on what a chunk of this tensor is, so a
    write asks for the same plan a read does -- ``GetFlightInfo`` with the
    ``endpoints`` mask and no ``scale_hint``. The mask is the plan alone
    because the rest of a describe (the pyramid especially) costs I/O a write
    has no use for.

    A ``slice_hint`` plans only the box that will be written; the server snaps
    it outward to its grid, so the returned bounds may cover more than was
    asked for. Planning is a metadata read and is answered while the tensor is
    still PENDING, which is what lets an upload be planned at all -- and, being
    idempotent, re-planned to resume after a client crash.
    """
    read_opt = _read_option(endpoints=True)
    if slice_hint is not None:
        read_opt.slice_hint.CopyFrom(slice_hint)
    cmd = _tensor_read_cmd(array_id, read_opt)
    info = state.client.get_flight_info(
        flight.FlightDescriptor.for_command(cmd.SerializeToString()),
        options=state.call_options,
    )
    if not info.endpoints:
        raise ValueError(
            f"upload: the server planned no chunks for {array_id}"
            + ("" if slice_hint is None else " in the requested region")
        )
    # The realized region the plan snapped to; its start is the origin the
    # endpoints' bounds are stated against.
    realized = TensorDescriptor.FromString(info.descriptor.command).slice_hint
    origin = tuple(realized.start) if realized.start else None
    plan = []
    for endpoint in info.endpoints:
        bounds = ChunkBounds.FromString(endpoint.app_metadata)
        if origin is None:
            start, stop = tuple(bounds.start), tuple(bounds.stop)
        else:
            start = tuple(o + v for o, v in zip(origin, bounds.start, strict=True))
            stop = tuple(o + v for o, v in zip(origin, bounds.stop, strict=True))
        plan.append(_PlannedChunk(start, stop, endpoint.ticket.ticket))
    return plan


def _plan_grid(
    plan: Sequence[_PlannedChunk],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    """``(origin, end, chunks)``: the dask chunking the plan's bounds imply.

    A plan covers a rectangular box of a regular grid, so each axis's distinct
    ``(start, stop)`` intervals -- sorted, contiguous -- are that axis's dask
    chunk sizes, and the box corners are the region to slice out of the
    caller's array. Deriving the grid from the plan rather than from
    ``chunk_shape`` is what keeps one authority: a server that snapped the
    request outward, or that chunks differently from the declaration, is
    followed rather than second-guessed.

    Raises ``ValueError`` if the endpoints are not one full rectangular grid --
    a plan this SDK cannot map onto ``da.store`` blocks, which is better said
    than half-uploaded.
    """
    ndim = len(plan[0].start)
    axes = [sorted({(c.start[ax], c.stop[ax]) for c in plan}) for ax in range(ndim)]
    cells = 1
    for intervals in axes:
        cells *= len(intervals)
    contiguous = all(
        hi == lo
        for intervals in axes
        for (_, hi), (lo, _) in zip(intervals, intervals[1:], strict=False)
    )
    if cells != len(plan) or not contiguous:
        raise ValueError(
            "upload: the server's plan is not one rectangular grid of chunks "
            f"({len(plan)} endpoints over a {cells}-cell box); upload it a "
            "chunk at a time with upload_chunk."
        )
    return (
        tuple(intervals[0][0] for intervals in axes),
        tuple(intervals[-1][1] for intervals in axes),
        tuple(tuple(hi - lo for lo, hi in intervals) for intervals in axes),
    )


def _slice_hint_of(
    slice_hint: Sequence[slice], shape: Sequence[int], array_id: str
) -> SliceHint:
    """A ``SliceHint`` for *slice_hint* against a tensor of *shape*.

    An open-ended ``stop`` is filled from the declared shape, which the upload
    path already holds -- unlike the read path, which has to resolve for it.
    """
    if len(slice_hint) != len(shape):
        raise ValueError(
            f"upload: slice_hint has {len(slice_hint)} axes, but {array_id} "
            f"has {len(shape)}"
        )
    return SliceHint(
        start=[0 if s.start is None else int(s.start) for s in slice_hint],
        stop=[
            int(dim) if s.stop is None else int(s.stop)
            for s, dim in zip(slice_hint, shape, strict=True)
        ],
    )


def _put_chunk(
    client: flight.FlightClient,
    call_options: flight.FlightCallOptions,
    array_id: str,
    ticket: bytes,
    data: np.ndarray,
) -> None:
    """One ``do_put``: open, write the batch, close, read the ack.

    *ticket* is the plan endpoint's ticket, echoed back whole; it is what names
    the chunk, so nothing here describes the write but the bytes themselves.
    *array_id* is only for the log line.

    Free of any session state, so the same code serves
    :meth:`UploadSession.upload_chunk` and a target that has been unpickled in a
    dask worker with no session to hand. A refusal surfaces as
    :class:`UploadRefused`.
    """
    cmd = PutCommand(chunk_ticket=ticket)
    desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
    schema = pa.schema([pa.field("data", pa.from_numpy_dtype(data.dtype))])

    try:
        writer, reader = client.do_put(desc, schema, options=call_options)
        with writer:
            writer.write_batch(
                pa.record_batch(
                    [pa.array(data.ravel(), type=schema.field("data").type)],
                    schema=schema,
                )
            )
            writer.done_writing()
            reader.read()
    except flight.FlightCancelledError as exc:
        refused = _refused_from(exc)
        if refused is None:
            raise
        raise refused from exc
    logger.debug(f"upload_chunk: uploaded {data.nbytes} bytes to {array_id}")


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

    **Holds the plan, not a planner.** The tickets are looked up by the block's
    bounds, so the target carries the whole plan as plain bytes rather than
    re-planning per block: one ``GetFlightInfo`` for the upload, and a worker
    that never needs to know what a chunk is.
    """

    __slots__ = (
        "_location",
        "_token",
        "_trust",
        "_array_id",
        "_tickets",
        "_origin",
        "_skip_empty",
        "shape",
        "dtype",
    )

    def __init__(
        self,
        location: str,
        token: Optional[str],
        trust: Optional[TlsTrust],
        array_id: str,
        plan: Sequence["_PlannedChunk"],
        origin: Sequence[int],
        shape: Sequence[int],
        dtype: np.dtype,
        skip_empty: bool = False,
    ):
        self._location = location
        self._token = token
        self._trust = trust or NO_TLS
        self._array_id = array_id
        self._tickets = {(c.start, c.stop): c.ticket for c in plan}
        # Where the stored sub-array sits in the tensor: a plan for a slice
        # starts somewhere other than the origin, and ``store`` indexes the
        # array it was handed.
        self._origin = tuple(int(v) for v in origin)
        # Drop an all-zero block instead of sending it -- only for a label
        # set, where an unwritten chunk is meant to read back as background.
        self._skip_empty = skip_empty
        # ``store`` reads these off the target to check it can hold the array.
        self.shape = tuple(shape)
        self.dtype = dtype

    def __setitem__(self, index: Tuple[slice, ...], value: np.ndarray) -> None:
        if self._skip_empty and not value.any():
            return
        start = tuple(o + s.start for o, s in zip(self._origin, index, strict=True))
        stop = tuple(o + s.stop for o, s in zip(self._origin, index, strict=True))
        ticket = self._tickets.get((start, stop))
        if ticket is None:
            # The array was rechunked onto the plan's own grid, so a miss is a
            # bug here rather than a caller's mistake -- say so with the bounds.
            raise ValueError(
                f"upload_array: no planned chunk at {list(start)}-{list(stop)} "
                f"of {self._array_id}"
            )
        client = _get_thread_client(self._location, self._token, self._trust)
        call_options = _get_shared_call_options(self._location, self._token)
        _put_chunk(client, call_options, self._array_id, ticket, value)


class UploadSession:
    """Tensor declaration and chunk upload over one Flight connection.

    .. note:: Experimental. This whole API -- ``create_tensor`` /
       ``upload_array`` / ``upload_chunk`` / ``set_upload_status`` -- is
       experimental and its behavior may change.

    Declare, then fill: ``create_tensor`` returns the server's descriptor for
    the new source, and that descriptor is what every write takes.

    Takes the shared ``_ClientState`` its two sibling collaborators take
    (``CatalogClient``, ``ChunkFetcher``). ``TensorFlightClient`` constructs one
    in its ``__init__`` and delegates its public upload API here.
    """

    def __init__(self, state: "_ClientState", catalog: "CatalogClient"):
        self._state = state
        # For the one thing status polling is: a partial upload answers with
        # where it stands, and that read lives on the catalog client.
        self._catalog = catalog

    def register_source(self, name: str = "", metadata: Optional[dict] = None) -> str:
        """Backs TensorFlightClient.register_source; see that method."""
        request = RegisterSource(
            name=name or "",
            metadata_json=json.dumps(metadata) if metadata else "",
        )
        body = do_action_one_result(
            self._state,
            flight.Action("register_source", request.SerializeToString()),
            unavailable_hint="Registering a source is unavailable",
        )
        source_id = RegisterSourceResult.FromString(body).source_id
        logger.info(f"register_source: registered {source_id}")
        return source_id

    def create_tensor(
        self,
        source_name: str,
        template: Any,
        *,
        chunk_shape: Optional[Sequence[int]] = None,
        dim_labels: Optional[Sequence[str]] = None,
        ome_metadata: Optional[dict] = None,
    ) -> TensorDescriptor:
        """Backs TensorFlightClient.create_tensor; see that method for the full
        documentation."""
        shape = tuple(int(n) for n in template.shape)
        dtype = np.dtype(template.dtype)
        if chunk_shape is None:
            chunk_shape = (
                _uniform_chunk_shape(template)
                if isinstance(template, da.Array)
                else shape
            )
        req_desc = TensorDescriptor(
            array_id=source_name,
            shape=list(shape),
            dtype=dtype.str,
            chunk_shape=[int(c) for c in chunk_shape],
            dim_labels=list(dim_labels or []),
            metadata_json=json.dumps(ome_metadata) if ome_metadata else "",
        )

        action = flight.Action("create_tensor", req_desc.SerializeToString())
        results = self._state.client.do_action(action, options=self._state.call_options)
        try:
            result = next(results)
        except StopIteration as exc:
            raise RuntimeError("create_tensor: server returned no result") from exc

        desc = TensorDescriptor.FromString(result.body.to_pybytes())
        logger.info(f"create_tensor: created {desc.array_id}")
        return desc

    def upload_array(
        self,
        desc: TensorDescriptor,
        arr: Any,
        slice_hint: Optional[Tuple[slice, ...]] = None,
    ) -> Dict[str, Any]:
        """Backs TensorFlightClient.upload_array; see that method for the full
        documentation."""
        shape = tuple(desc.shape)
        if tuple(arr.shape) != shape:
            raise ValueError(
                f"upload_array: array shape {tuple(arr.shape)} does not match "
                f"the declared shape {shape} of {desc.array_id}"
            )
        if np.dtype(arr.dtype) != np.dtype(desc.dtype):
            raise ValueError(
                f"upload_array: array dtype {np.dtype(arr.dtype)} does not match "
                f"the declared dtype {np.dtype(desc.dtype)} of {desc.array_id}"
            )

        # One plan for the whole upload, and the grid comes back with it: the
        # server decides what a chunk is, so *arr* is rechunked onto the plan's
        # own bounds rather than onto the declared chunk_shape.
        plan = _plan_write(
            self._state,
            desc.array_id,
            None
            if slice_hint is None
            else _slice_hint_of(slice_hint, shape, desc.array_id),
        )
        origin, end, chunks = _plan_grid(plan)
        region = arr[tuple(slice(lo, hi) for lo, hi in zip(origin, end, strict=True))]
        if not isinstance(region, da.Array):
            region = da.from_array(np.asarray(region), chunks=chunks)
        else:
            region = region.rechunk(chunks)  # a no-op when already on the grid

        # An all-zero block of a label set is not sent at all: its unwritten
        # chunks read back as background, so one labelled frame of a thousand
        # costs one frame (biopb/biopb#1059).
        self._store_chunks(desc.array_id, region, plan, origin)
        if slice_hint is not None:
            # A region is by definition not the whole tensor, so this caller
            # cannot know the upload is done. It says so itself, later, with
            # set_upload_status.
            return self._catalog.get_upload_status(desc.array_id)
        # Publishing is what marks the source complete, so a whole-array upload
        # does it on the caller's behalf -- it is the one caller that knows,
        # from having written every block itself, that there is nothing more to
        # send. A caller driving `upload_chunk` by hand does not know when it
        # is done, and moves the upload explicitly.
        return self.set_upload_status(desc, UploadStatusPb.READY)

    def _store_chunks(
        self,
        array_id: str,
        arr: da.Array,
        plan: Sequence["_PlannedChunk"],
        origin: Sequence[int],
    ) -> None:
        """Hand the whole upload to dask as one graph.

        ``upload_array`` has already rechunked *arr* onto the planned grid, so
        one dask block is one planned chunk and ``store`` needs no alignment
        help.

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
            array_id,
            plan,
            origin,
            arr.shape,
            arr.dtype,
            skip_empty=_is_label_set(array_id),
        )
        da.store(arr, target, lock=False)

    def set_upload_status(
        self,
        target: Any,
        state: Any,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Backs TensorFlightClient.set_upload_status; see that method for the
        full documentation."""
        array_id = target if isinstance(target, str) else target.array_id
        req = SetUploadStatus(
            array_id=array_id, state=_state_value(state), reason=reason
        )
        action = flight.Action("set_upload_status", req.SerializeToString())
        try:
            results = self._state.client.do_action(
                action, options=self._state.call_options
            )
            result = next(results)
        except StopIteration as exc:
            raise RuntimeError("set_upload_status: server returned no result") from exc
        except flight.FlightCancelledError as exc:
            refused = _refused_from(exc)
            if refused is None:
                raise
            raise refused from exc
        status = UploadStatusPb.FromString(result.body.to_pybytes())
        if status.state == UploadStatusPb.STATE_UNSPECIFIED:
            # The server leaves the whole message unset for an id it tracks no
            # upload for, which DISCARDED answers with rather than raising.
            return _unknown_upload_status(array_id)
        logger.info(
            f"set_upload_status: {array_id} -> {UploadStatusPb.State.Name(req.state)}"
        )
        return _upload_status_dict(array_id, status)

    def upload_chunk(
        self,
        desc: TensorDescriptor,
        bounds: ChunkBounds,
        data: np.ndarray,
    ) -> None:
        """Backs TensorFlightClient.upload_chunk; see that method for the full
        documentation."""
        want = (tuple(bounds.start), tuple(bounds.stop))
        plan = _plan_write(
            self._state,
            desc.array_id,
            SliceHint(start=list(bounds.start), stop=list(bounds.stop)),
        )
        # The server snaps a slice outward to its grid, so a plan of one chunk
        # whose bounds are the ones asked for is the only proof that *bounds*
        # names a chunk. Refused here rather than sent, because a write of part
        # of a chunk has nowhere to land: the id the planner mints covers the
        # whole cell.
        if len(plan) != 1 or (plan[0].start, plan[0].stop) != want:
            snapped = ", ".join(f"{list(c.start)}-{list(c.stop)}" for c in plan[:4])
            raise ValueError(
                f"upload_chunk: {list(bounds.start)}-{list(bounds.stop)} is not "
                f"one chunk of {desc.array_id}; the server's grid puts it in "
                f"{snapped}{' ...' if len(plan) > 4 else ''}. Write a chunk of "
                f"that grid, or use upload_array."
            )
        _put_chunk(
            self._state.client,
            self._state.call_options,
            desc.array_id,
            plan[0].ticket,
            data,
        )
