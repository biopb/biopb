"""Per-connection catalog + read core for TensorFlightClient.

Extracted from :mod:`biopb.tensor.client` (issue #278 item C). The two
collaborators share the connection via :class:`_ClientState`:

- :class:`CatalogClient` -- discovery / metadata / resolve / warm / source
  registration (``list_sources`` / ``query_sources`` / ``resolve`` / ... RPCs).
- :class:`ChunkFetcher` -- tensor reads: plan a read with GetFlightInfo and
  build the lazy dask chunk-fetching array.

Neither caches a descriptor: every one is fetched per call, and a caller that
wants one memoized owns that policy (see :class:`_ClientState`).

``TensorFlightClient`` holds one of each and delegates its public API to them;
``client.py`` re-exports the module helpers that external callers still import
from ``biopb.tensor.client``.
"""

import json
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

from biopb.image._roi_rows import (
    ROI_ID_SCHEMA,
    ROI_ROW_SCHEMA,
    roi_ids_to_table,
    rois_to_table,
    table_to_rois,
)
from biopb.image.annotation_pb2 import (
    RoiAnnotation,
    RoiDeleteResult,
    RoiListResult,
    RoiPruneRequest,
    RoiPruneResult,
    RoiPutResult,
    RoiSetInfo,
)
from biopb.tensor._catalog_rows import (
    SOURCE_ROW_COLUMNS,
    _descriptor_from_row,
    sql_literal,
    tensor_descriptors_from_row,
)
from biopb.tensor._labels import LABELS_SEGMENT
from biopb.tensor._pool import (
    _build_dask_array_from_chunk_map,
    _chunk_map_from_endpoints,
    _get_shared_call_options,
    _get_thread_client,
)
from biopb.tensor._tls import TlsTrust, resolve_tls_trust
from biopb.tensor.descriptor_pb2 import (
    AddSourceProgress,
    AddSourceRequest,
    AddSourceResult,
    AddSourceStreamMessage,
    CatalogQuery,
    DataSourceDescriptor,
    FlightRequest,
    RemoveSourceRequest,
    RemoveSourceResult,
    ResolveProgress,
    ResolveStreamMessage,
    SliceHint,
    TensorDescriptor,
    TensorReadOption,
    UploadStatus as UploadStatusPb,
    WarmProgress,
    WarmStreamMessage,
)
from biopb.tensor.serialized_pb2 import SerializedTensor
from biopb.tensor.ticket_pb2 import (
    ChunkBounds,
    PutCommand,
    RoiDelete,
    RoiPut,
    RoiRead,
    TensorTicket,
)

logger = logging.getLogger(__name__)


@dataclass
class _ClientState:
    """Per-connection state shared by CatalogClient / ChunkFetcher / the facade.

    The Flight connection handles and nothing else. **The SDK caches no
    descriptors**: every one is fetched per call. Memoizing is a caller's policy,
    since only the caller knows what invalidates it --
    ``biopb-tensor-server/docs/source-descriptor-retirement.md``.
    """

    raw_client: flight.FlightClient
    call_options: flight.FlightCallOptions
    location: str
    token: Optional[str]
    cache_bytes: int
    # Resolved TLS trust for a grpc+tls:// location (the TOFU-pinned server cert
    # plus any hostname override), else NO_TLS. Carried into the lazy chunk-fetch
    # graph so every dask worker's FlightClient trusts the same pinned root
    # without re-running TOFU (biopb/biopb#604, biopb/biopb#606).
    tls_trust: Optional[TlsTrust] = None
    # Set once the server's Flight protocol shape has been checked (or when
    # the check is bypassed, e.g. for a test double).
    protocol_checked: bool = False

    @property
    def client(self) -> flight.FlightClient:
        """The connection, checked once against the server's protocol shape.

        The first use of a client runs one ``health`` action and reads the
        server's ``protocol``. A server that speaks another shape (or none --
        a pre-v2 server has no key) is refused here with an actionable message
        rather than letting a ``FlightRequest`` reach a server that will parse
        it as something else and answer with a confusing error.
        """
        if not self.protocol_checked:
            _check_flight_protocol(self.raw_client, self.call_options, self.location)
            self.protocol_checked = True
        return self.raw_client

    @client.setter
    def client(self, value: flight.FlightClient) -> None:
        """A connection handed in directly is taken as already checked -- this
        is how tests inject a double, and a double has no health to probe."""
        self.raw_client = value
        self.protocol_checked = True


class ResolveCancelled(Exception):
    """Raised by :meth:`TensorFlightClient.resolve` when its ``should_cancel``
    callback asks it to stop.

    The client stops consuming the resolve stream and unwinds; the server's
    recall daemon thread runs to completion and caches its result, so a later
    :meth:`resolve` coalesces onto the finished work rather than re-downloading.
    """


def _upload_status_dict(source_id: str, status: UploadStatusPb) -> Dict[str, Any]:
    """The one shape an upload status takes on the SDK, for the poll and for
    ``set_upload_status`` alike."""
    return {
        "source_id": source_id,
        "state": UploadStatusPb.State.Name(status.state),
        "expected_chunks": status.expected_chunks,
        "uploaded_chunks": status.uploaded_chunks,
        "reason": status.reason,
    }


def _unknown_upload_status(source_id: str) -> Dict[str, Any]:
    """The answer for a source the server tracks no upload for.

    Mirrors the server's own ``unknown_upload_status`` so the two ends agree on
    the shape, and never means "not started yet": the record exists from the
    moment ``add_tensor`` hands out the id.
    """
    return {
        "source_id": source_id,
        "state": "UNKNOWN",
        "expected_chunks": 0,
        "uploaded_chunks": 0,
        "reason": "",
    }


def _request_crop_slices(
    ndim: int,
    original_slice_hint: SliceHint,
    realized_slice_hint: SliceHint,
    scale: Optional[Sequence[int]],
    keep_axes: Tuple[int, ...] = (),
) -> Tuple[slice, ...]:
    """Per-axis crop mapping the requested region onto the realized array.

    The server snaps a slice_hint outward to lcm-aligned chunk boundaries, so
    the realized (returned) bounds can exceed what was requested. This maps the
    requested world-coordinate bounds onto the realized array's logical indices,
    accounting for the applied per-axis downsampling ``scale``. Axes listed in
    ``keep_axes`` are left full (``slice(None)``) -- the websocket render path
    keeps Y/X uncropped so the rendered tile covers the whole loaded region.
    """
    crop = []
    for ax in range(ndim):
        if ax in keep_axes:
            crop.append(slice(None))
            continue
        req_start = int(original_slice_hint.start[ax])
        req_stop = int(original_slice_hint.stop[ax])
        ret_start = int(realized_slice_hint.start[ax])
        s = int(scale[ax]) if scale and ax < len(scale) else 1
        logical_start = (req_start - ret_start) // s
        logical_stop = (req_stop - ret_start + s - 1) // s
        crop.append(slice(logical_start, logical_stop))
    return tuple(crop)


def _parse_flight_endpoints(
    info: "flight.FlightInfo",
) -> Tuple[List[bytes], List[ChunkBounds]]:
    """Decode a FlightInfo's endpoints into parallel ``(chunk_ids, bounds)`` lists.

    chunk_id is an opaque server-minted token (echoed back to do_get); a chunk's
    bounds ride on the endpoint's app_metadata, so the client never decodes the
    chunk_id byte format. Shared by every GetFlightInfo read planner.
    """
    chunks: List[bytes] = []
    chunk_bounds_list: List[ChunkBounds] = []
    for endpoint in info.endpoints:
        ticket = TensorTicket.FromString(endpoint.ticket.ticket)
        chunks.append(ticket.chunk_id)
        chunk_bounds_list.append(ChunkBounds.FromString(endpoint.app_metadata))
    return chunks, chunk_bounds_list


def _refetch_flight_info(
    descriptor: TensorDescriptor, location: str, token: Optional[str]
) -> "flight.FlightInfo":
    """GetFlightInfo for the read a descriptor already describes.

    For a handle that carries no endpoints -- a source declared before its
    chunks existed, or one whose producer chose not to embed a plan. The
    request is rebuilt from the descriptor's realized slice, scale and
    reduction, so the answer is the same plan its producer would have got.

    Reuses the worker's pooled per-thread connection (with its tuned gRPC
    message-size options) rather than dialing a throwaway client; a later
    chunk fetch to the same (location, token) then rides the same connection.
    The TLS resolve is memoized per process, so evaluating it eagerly here --
    even when the pooled client already exists and discards the value -- costs
    a dict lookup rather than a handshake.
    """
    # `endpoints` is explicit: this call exists to get them, and nothing is
    # implied by the mask any more.
    read_opt = _read_option(endpoints=True)
    if descriptor.HasField("slice_hint"):
        read_opt.slice_hint.CopyFrom(descriptor.slice_hint)
    if descriptor.scale_hint:
        read_opt.scale_hint[:] = list(descriptor.scale_hint)
    if descriptor.reduction_method:
        read_opt.reduction_method = descriptor.reduction_method
    cmd = _tensor_read_cmd(descriptor.array_id, read_opt)

    client = _get_thread_client(location, token, resolve_tls_trust(location))
    call_options = _get_shared_call_options(location, token)
    flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
    info = client.get_flight_info(flight_desc, options=call_options)
    logger.debug(f"_refetch_flight_info: got {len(info.endpoints)} endpoints")
    return info


def _requested_slice(info: "flight.FlightInfo") -> Optional[SliceHint]:
    """The slice a plan was asked for, off ``FlightInfo.app_metadata``.

    The server stamps the request's ``slice_hint`` there verbatim, beside the
    chunk-aligned realized one in the descriptor. None for an unsliced read.
    """
    raw = info.app_metadata
    return SliceHint.FromString(raw) if raw else None


def _dask_from_flight_info(
    info: "flight.FlightInfo",
    location: str,
    token: Optional[str],
    cache_bytes: int,
    tls_trust: Optional[TlsTrust],
    requested: Optional[SliceHint] = None,
) -> da.Array:
    """The lazy array a planned read describes.

    The one reconstruction, whether the FlightInfo came from this connection's
    GetFlightInfo (``get_tensor``) or arrived serialized from another process
    (``tensor_from_pb``): decode the descriptor and endpoints, build the
    chunk-fetching array, crop the realized region back to *requested* (the
    plan's own ``app_metadata`` unless the caller kept an earlier one).
    """
    _check_wire_protocol(info.schema)
    descriptor = TensorDescriptor.FromString(info.descriptor.command)
    chunk_ids, bounds_list = _parse_flight_endpoints(info)
    shape = tuple(descriptor.shape)
    chunk_map, grid_shape = _chunk_map_from_endpoints(chunk_ids, bounds_list, shape)
    dask_arr = _build_dask_array_from_chunk_map(
        chunk_map,
        grid_shape,
        shape,
        np.dtype(descriptor.dtype),
        location,
        token,
        cache_bytes,
        tls_trust,
    )
    if requested is None:
        requested = _requested_slice(info)
    if requested is not None and descriptor.HasField("slice_hint"):
        dask_arr = dask_arr[
            _request_crop_slices(
                len(shape),
                requested,
                descriptor.slice_hint,
                list(descriptor.scale_hint) if descriptor.scale_hint else None,
            )
        ]
    return dask_arr


def _check_flight_protocol(
    client: flight.FlightClient, call_options: flight.FlightCallOptions, location: str
) -> None:
    """Refuse a server whose Flight protocol shape is not this SDK's.

    Reads the ``protocol`` key the ``health`` action reports. A server without
    the key predates it and speaks v1 (sentinel-routed descriptors,
    prefix-sniffed tickets), which this SDK no longer does. An unreachable
    server is not this check's concern: its error propagates as it always did.
    """
    from biopb.tensor._wire_version import FLIGHT_PROTOCOL_VERSION

    try:
        results = client.do_action(flight.Action("health", b""), options=call_options)
        body = next(iter(results), None)
    except flight.FlightUnauthenticatedError:
        # A server that gates health, which this one need not do -- the private
        # call this client is about to make authorizes itself either way.
        return
    if body is None:
        return  # not a biopb server at all; let the first real call say so
    try:
        server_ver = int(json.loads(body.body.to_pybytes()).get("protocol", 1))
    except (ValueError, TypeError, AttributeError):
        server_ver = 1
    if server_ver != FLIGHT_PROTOCOL_VERSION:
        stale = "server" if server_ver < FLIGHT_PROTOCOL_VERSION else "client"
        raise RuntimeError(
            f"Incompatible biopb Flight protocol: the server at {location} speaks "
            f"v{server_ver}, this client speaks v{FLIGHT_PROTOCOL_VERSION}. "
            f"Upgrade the {stale} so both sides match."
        )


def _check_wire_protocol(schema: pa.Schema) -> None:
    """Fail fast if the server's chunk wire-protocol version is incompatible.

    The chunk ``RecordBatch`` encoding is a hard contract (biopb/biopb#293): a
    version mismatch means the client would misread every chunk (e.g. decode the
    v2 binary blob as a v1 typed list). We reject at ``GetFlightInfo`` -- before
    any ``do_get`` -- with an actionable message rather than let a cryptic decode
    error surface deep in the read path. The version constant lives in ``biopb``
    core, which both the client and the server import, so there is one source of
    truth (see ``biopb.tensor._wire_version``).
    """
    from biopb.tensor._wire_version import (
        TENSOR_WIRE_PROTOCOL_VERSION,
        WIRE_PROTOCOL_METADATA_KEY,
    )

    meta = schema.metadata or {}
    raw = meta.get(WIRE_PROTOCOL_METADATA_KEY.encode("utf-8"))
    # An unstamped schema is a pre-#293 server, which speaks the v1 typed-list
    # encoding this client can no longer read.
    try:
        server_ver = int(raw.decode("utf-8")) if raw is not None else 1
    except (ValueError, AttributeError):
        server_ver = 1

    if server_ver != TENSOR_WIRE_PROTOCOL_VERSION:
        stale = "server" if server_ver < TENSOR_WIRE_PROTOCOL_VERSION else "client"
        raise RuntimeError(
            f"Incompatible biopb tensor wire protocol: the server speaks v{server_ver}, "
            f"this client speaks v{TENSOR_WIRE_PROTOCOL_VERSION}. The chunk encoding is a "
            f"breaking contract (biopb/biopb#293); upgrade the {stale} so both sides match."
        )


def extra_info(exc: Exception) -> Optional[Mapping[str, Any]]:
    """The structured payload the server attaches to a Flight error, or ``None``.

    pyarrow exposes only a subset of gRPC's status codes as typed exceptions --
    there is no ``FlightNotFoundError`` -- so the server puts the canonical code
    and a machine ``reason`` in ``extra_info`` for clients to switch on (see its
    ``to_flight_error``). This is the one place that decode lives.
    """
    raw = getattr(exc, "extra_info", None)
    if not raw:
        return None
    try:
        return json.loads(bytes(raw).decode())
    except (ValueError, UnicodeDecodeError):
        return None


class _AddressingError(ValueError):
    """An id that names no source, or no such tensor within one.

    Its own type so :meth:`CatalogClient.get_upload_status` can absorb it
    without also swallowing the unresolved steer, which is the other
    ``ValueError`` a descriptor probe raises.
    """


def _addressing_error(exc: flight.FlightError) -> Optional[_AddressingError]:
    """The refusal a terminal addressing error deserves, else ``None``.

    Keeps the server's wording: it knows whether the source is missing or the
    source is there and the field is not, which no client-side reconstruction
    can tell apart without a catalog round trip.
    """
    info = extra_info(exc)
    if info is None or info.get("code") not in ("NOT_FOUND", "INVALID_ARGUMENT"):
        return None
    # pyarrow appends its own ". Detail: ..." to the server's message.
    return _AddressingError(str(exc).split(". Detail:")[0])


def _ambiguous_default_error(source_id: str, count: int) -> ValueError:
    """The #75 refusal: a bare id naming a source that holds several tensors.

    Two paths reach it -- the pre-RPC resolve an open-ended slice needs, and the
    post-response check -- and a caller must not be able to tell which refused.
    """
    return ValueError(
        f"Source '{source_id}' has multiple tensors ({count}), "
        "tensor_id must be specified"
    )


def _raise_read_refusal(exc: flight.FlightError, array_id: str) -> None:
    """Restate a server refusal as the error this SDK promises, or return.

    Shared by the describe (:meth:`CatalogClient._fetch_tensor_descriptor`) and
    the plan (:meth:`ChunkFetcher._plan_read`): the same ``GetFlightInfo`` under
    different masks, so they refuse identically.
    """
    if (
        isinstance(exc, flight.FlightUnavailableError)
        and "unresolved" in str(exc).lower()
    ):
        # Taken from the server rather than a local catalog check, so it also
        # holds for a capability-token holder, who cannot browse the catalog.
        raise _unresolved_source_error(split_array_id(array_id)[0]) from exc
    addressing = _addressing_error(exc)
    if addressing is not None:
        raise addressing from exc


def _unresolved_source_error(source_id: str) -> ValueError:
    """Directive error for reading an *unresolved* (cloud / synced-folder) source.

    Shared by every read entry point so the guidance is uniform: name the cure
    (``client.resolve``) instead of leaking a bare internal "no tensors", and --
    critically for methods like ``get_physical_scale`` -- raise this rather than
    silently recalling (downloading) the whole file just to answer a metadata
    query. Resolving is the heavyweight, *consenting* act; reads must not trigger
    it implicitly."""
    return ValueError(
        f"Source '{source_id}' is unresolved (no tensors listed yet). If this "
        f"is a cloud / synced-folder source, call client.resolve('{source_id}') "
        f"first to download and resolve it, then read it."
    )


def split_array_id(array_id: str) -> Tuple[str, Optional[str]]:
    """Split a tensor's globally-unique ``array_id`` into ``(source_id,
    array_id-or-None)``.

    A tensor is identified by its ``array_id`` ALONE (see the policy at the top
    of ``proto/biopb/tensor/descriptor.proto``); ``source_id`` is its slash-free
    prefix. A bare id (no '/') yields ``None`` -- the server's documented
    "default (first) tensor" request (#44). Whether a bare *multi*-tensor id is
    acceptable is the caller's policy, not this function's: see
    :meth:`CatalogClient._resolve_descriptor`, which refuses it (#75), versus
    :meth:`CatalogClient.get_descriptor`, which anchors on the default.

    Public, unlike its neighbours in this private module: callers outside the
    package honour the same policy (biopb-mcp through ``client.py``'s
    re-export, the tensor server directly).
    """
    if "/" in array_id:
        return array_id.split("/", 1)[0], array_id
    return array_id, None


def _read_option(
    *,
    endpoints: bool = False,
    metadata_json: bool = False,
    pyramid: bool = False,
    upload_status: bool = False,
    is_resident: bool = False,
) -> TensorReadOption:
    """A ``TensorReadOption`` whose field mask names the parts asked for.

    The wire takes a ``FieldMask``; this keeps the SDK's
    own surface named booleans, because hand-assembling paths is a poor API and
    a mistyped one is a server-side refusal rather than a type error.

    Every part is opt-in, including ``endpoints`` -- the O(chunks) read plan.
    The bools this replaced defaulted that one *on*, so a describe had to
    remember to switch it off and a read got it by saying nothing. Now the read
    path asks, and saying nothing is the cheap call.
    """
    opt = TensorReadOption()
    opt.fields.paths.extend(
        name
        for name, wanted in (
            ("endpoints", endpoints),
            ("metadata_json", metadata_json),
            ("pyramid", pyramid),
            ("upload_status", upload_status),
            ("is_resident", is_resident),
        )
        if wanted
    )
    return opt


def _tensor_read_cmd(array_id: str, read_opt: TensorReadOption) -> FlightRequest:
    """Address ``read_opt`` at ``array_id`` as the ``data`` flight's request."""
    read_opt.array_id = array_id
    return FlightRequest(tensor_read=read_opt)


def _catalog_ticket(sql: str) -> flight.Ticket:
    """A DoGet ticket that runs ``sql`` on the ``catalog`` flight."""
    ticket = TensorTicket(catalog_query=CatalogQuery(sql=sql))
    return flight.Ticket(ticket.SerializeToString())


def do_action_one_result(
    state: "_ClientState", action: flight.Action, *, unavailable_hint: str
) -> bytes:
    """Run a single-result ``do_action``, with the same "old server" remap
    :meth:`CatalogClient._iter_action_messages` gives the streaming actions.

    A module function rather than a method because both sessions need it: an
    upload action is as likely to meet a server that predates it as a catalog
    one is -- more so, since the upload surface moves inside wire v2 rather
    than minting a version per revision (``_wire_version``).

    *unavailable_hint* is the feature-specific lead-in for the "Unknown action"
    case (e.g. "Source removal is unavailable"); a genuinely empty result
    stream (a server that never sends one) raises a plain ``RuntimeError``
    naming the action.
    """
    try:
        results = state.client.do_action(action, options=state.call_options)
        result = next(results)
    except flight.FlightError as exc:
        if "Unknown action" in str(exc):
            raise RuntimeError(
                f"{unavailable_hint}: the tensor server is too old to "
                f"support the '{action.type}' action. Upgrade the server."
            ) from exc
        raise
    except StopIteration as exc:
        raise RuntimeError(f"{action.type} returned no result") from exc
    return result.body.to_pybytes()


class CatalogClient:
    """Catalog, metadata, and source-lifecycle RPCs over one Flight connection.

    Owns discovery (``list_sources`` / ``query_sources``), per-tensor metadata
    probes, the experimental cloud ``resolve`` / ``warm`` streams, and runtime
    source registration. Reads and writes the shared ``_ClientState`` caches.
    """

    def __init__(self, state: "_ClientState"):
        self._state = state

    _SOURCES_SQL = f"SELECT {SOURCE_ROW_COLUMNS} FROM sources"

    def list_sources(self) -> Dict[str, DataSourceDescriptor]:
        """Backs TensorFlightClient.list_sources; see that method for the full
        documentation."""
        table = self._query_table(self._SOURCES_SQL + " ORDER BY source_id")
        source_descriptors = {}
        for row in table.to_pylist():
            source_desc = _descriptor_from_row(row)
            source_descriptors[source_desc.source_id] = source_desc
        logger.info(f"list_sources: returned {len(source_descriptors)} sources")
        return source_descriptors

    def get_source(self, source_id: str) -> Optional[DataSourceDescriptor]:
        """Backs TensorFlightClient.get_source; see that method for the full
        documentation."""
        table = self._query_table(
            f"{self._SOURCES_SQL} WHERE source_id = {sql_literal(source_id)}"
        )
        for row in table.to_pylist():
            return _descriptor_from_row(row)
        return None

    def query_sources(self, sql: str, *, format: str = "arrow") -> Any:  # noqa: A002 - public, documented keyword API (mirrors DuckDB/pandas `format`)
        """Backs TensorFlightClient.query_sources; see that method for the full
        documentation."""
        if format not in ("pandas", "arrow", "records"):
            raise ValueError(
                f"query_sources: unknown format {format!r}; "
                "expected 'pandas', 'arrow', or 'records'"
            )

        return self._format_query_result(self._query_table(sql), format)

    def _query_table(self, sql: str) -> pa.Table:
        """One DoGet on the ``catalog`` flight: the ticket carries the SQL, the
        stream's schema metadata carries the truncation flags."""
        reader = self._state.client.do_get(
            _catalog_ticket(sql), options=self._state.call_options
        )
        table = reader.read_all()

        # Truncation comes from the server's flag, not from differencing counts:
        # `total_sources` is the catalog size, so a filtered query (or one
        # against another catalog table) legitimately returns fewer rows without
        # anything having been dropped.
        metadata = table.schema.metadata or {}
        returned = metadata.get(b"returned_rows")
        if returned:
            returned_count = int(returned.decode())
            if metadata.get(b"truncated", b"").decode() == "True":
                total = metadata.get(b"total_rows")
                logger.info(
                    "query_sources: returned %s of %s rows (truncated)",
                    returned_count,
                    int(total.decode()) if total else "?",
                )
            else:
                logger.debug("query_sources: returned %s rows", returned_count)
        return table

    @staticmethod
    def _format_query_result(table: pa.Table, format: str):  # noqa: A002 - public, documented keyword API (mirrors DuckDB/pandas `format`)
        """Convert a query result Arrow table to the caller-requested format.

        ``"arrow"`` (the default) returns the Table unchanged -- backward
        compatible and the only zero-copy / metadata-preserving option.
        ``"pandas"``/``"records"`` are opt-in conveniences; pandas is imported
        lazily so it is required only when ``format="pandas"`` is requested.
        """
        if format == "arrow":
            return table
        if format == "records":
            return table.to_pylist()
        # format == "pandas" (validated by the caller)
        try:
            import pandas  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "query_sources(format='pandas') requires pandas; install "
                "pandas, or call with format='arrow' / format='records'."
            ) from exc
        df = table.to_pandas()
        # Arrow->pandas turns a NULL in a string column into a float NaN, which
        # is *truthy* -- so `if row.metadata_json:` silently passes and then the
        # downstream `json.loads(...)` blows up on a float (issue #47). Normalize
        # missing text cells back to None (falsy, pd.notna-clean). Target by the
        # Arrow schema so genuine numeric NaN in real float columns is untouched.
        # Go through object dtype: pandas' str dtype re-coerces a None put back
        # in via .where() to NaN, but an object column preserves None.
        for col_field in table.schema:
            if pa.types.is_string(col_field.type) or pa.types.is_large_string(
                col_field.type
            ):
                col = df[col_field.name].astype(object)
                df[col_field.name] = col.where(col.notna(), None)
        return df

    def get_source_metadata(self, source_id: str) -> dict:
        """Backs TensorFlightClient.get_source_metadata; see that method for the full
        documentation."""
        # One addressed catalog row. The column IS the answer: get_metadata() is
        # called once at registration to fill it, and the serve path reads it
        # back rather than recomputing (biopb/biopb#253). Going through a
        # tensor-bound GetFlightInfo instead used to overlay the *first* field's
        # get_tensor_metadata() delta, so a multi-field source reported one
        # arbitrary field's extras as the source's metadata.
        table = self._query_table(
            "SELECT is_resolved, metadata_json FROM sources "
            f"WHERE source_id = {sql_literal(source_id)}"
        )
        rows = table.to_pylist()
        if not rows:
            raise ValueError(f"Source not found: {source_id}")
        row = rows[0]

        if not row.get("is_resolved", True):
            # Unresolved (cloud / synced-folder) source: tensors are unknown
            # until resolve. Don't silently return {} -- that conflates
            # "unresolved" with "resolved, no metadata" (the line below). Steer
            # the caller to the explicit, consented resolve() instead, matching
            # get_physical_scale / get_tensor (#108). The flag, not an empty
            # tensor list -- a source can resolve cleanly and hold nothing
            # readable (biopb/biopb#1032).
            raise _unresolved_source_error(source_id)

        raw = row.get("metadata_json")
        return json.loads(raw) if raw else {}

    def get_physical_scale(
        self, array_id: str
    ) -> Optional[Tuple[List[float], List[str]]]:
        """Backs TensorFlightClient.get_physical_scale; see that method for the full
        documentation."""
        # Never the opt-in OME tree, so a compact scale probe does not depend on
        # the server having a metadata catalog. An unresolved source refuses in
        # the probe rather than recalling a whole cloud file to read its pixel
        # size, and a fetch error propagates -- it has to stay distinguishable
        # from "no physical scale recorded", the only case that yields None.
        desc = self._fetch_tensor_descriptor(array_id, with_metadata=False)
        if not desc.physical_scale:
            return None
        return list(desc.physical_scale), list(desc.physical_unit)

    def _fetch_tensor_descriptor(
        self,
        array_id: str,
        with_metadata: bool = False,
        with_pyramid: bool = False,
        with_read_plan: bool = False,
        with_upload_status: bool = False,
        with_residency: bool = False,
    ) -> "TensorDescriptor":
        """Fetch one tensor's descriptor directly from the server (internal).

        Backs the public ``get_descriptor`` (the array_id-keyed primitive). Uses
        the per-tensor ``GetFlightInfo`` RPC, which works even when the source is
        beyond the (truncatable) ``list_sources()`` cap. A bare source_id ->
        the source's default (first) tensor (#44). This is a CHEAP probe: it
        does NOT resolve. An unresolved (cloud / synced-folder) source raises
        the directive ``_unresolved_source_error`` steering the caller to
        :meth:`resolve`, rather than triggering a download.

        The three ``with_*`` flags are the ``GetFlightInfo`` response field masks
        (biopb/biopb#563); each selects one optional part of the response:

        - ``with_metadata`` -- fill ``metadata_json`` (the full OME tree).
        - ``with_pyramid`` -- advertise the resolution pyramid on the descriptor.
        - ``with_read_plan`` -- enumerate the per-request chunk endpoints.
        - ``with_upload_status`` -- progress of the upload backing this tensor.
        - ``with_residency`` -- whether the bytes are local right now. A bounded
          stat walk of the source, so never ask for it in a loop over a
          catalog: that is the shape biopb/biopb#1048 removed.

        This primitive returns only the ``TensorDescriptor`` (never the endpoints),
        so all three masks **default off** -- the cheapest structural probe. With
        ``with_read_plan=False`` the O(chunks) plan the caller would discard is
        skipped; ``with_pyramid=False`` skips the (per-level, potentially remote)
        pyramid sizing; ``with_metadata=False`` skips the heavy OME tree. Callers
        that need any of those parts opt in. A server too old for a path refuses
        the request rather than quietly omitting it, so a caller never plans
        around a field that silently did not arrive.

        One RPC every call, storing nothing, so the masks are honoured each time
        and a live field (``upload_status``, ``is_resident``) is current.
        """
        cmd = _tensor_read_cmd(
            array_id,
            _read_option(
                endpoints=with_read_plan,
                metadata_json=with_metadata,
                pyramid=with_pyramid,
                upload_status=with_upload_status,
                is_resident=with_residency,
            ),
        )
        fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        try:
            info = self._state.client.get_flight_info(
                fd, options=self._state.call_options
            )
        except flight.FlightError as exc:
            # GetFlightInfo does not resolve on serve, so an unresolved (cloud /
            # synced-folder) source refuses here instead of silently downloading.
            # This stays a cheap steering probe.
            _raise_read_refusal(exc, array_id)
            raise
        return TensorDescriptor.FromString(info.descriptor.command)

    def get_descriptor(
        self,
        array_id: str,
        with_metadata: bool = False,
        with_pyramid: bool = True,
        with_read_plan: bool = False,
        with_residency: bool = False,
        with_upload_status: bool = False,
    ) -> "TensorDescriptor":
        """Backs TensorFlightClient.get_descriptor; see that method for the full
        documentation."""
        return self._fetch_tensor_descriptor(
            array_id,
            with_metadata=with_metadata,
            with_residency=with_residency,
            with_upload_status=with_upload_status,
            with_pyramid=with_pyramid,
            with_read_plan=with_read_plan,
        )

    def _resolve_descriptor(self, array_id: str) -> "TensorDescriptor":
        """The structural ``TensorDescriptor`` for ``array_id``: catalog row
        first, then a direct per-tensor probe.

        Called only for an open-ended slice ``stop``, which needs the tensor's
        extent before the request can be built. Holding the row, it also makes
        the addressing refusals, so the post-response check can be skipped.

        The probe is the fallback because it is the only step open to a
        capability-token holder, who may read the source but not browse the
        catalog.
        """
        source_id, tensor_id = split_array_id(array_id)
        row = self._source_tensors_row(source_id)

        if row is not None:
            # The flag, not an empty tensor list: a source can resolve cleanly
            # and hold nothing readable, and steering that owner to resolve()
            # sends them round a loop.
            if not row.get("is_resolved", True):
                raise _unresolved_source_error(source_id)
            tensors = tensor_descriptors_from_row(row)
            if not tensors:
                raise ValueError(
                    f"Source '{source_id}' is resolved but lists no readable "
                    "tensors: nothing in it could be opened as an array."
                )
            if tensor_id is None:
                if len(tensors) > 1:
                    raise _ambiguous_default_error(source_id, len(tensors))
                return tensors[0]
            for candidate in tensors:
                if candidate.array_id == array_id:
                    return candidate

        return self._fetch_tensor_descriptor(array_id, with_metadata=False)

    def _refuse_ambiguous_default(self, array_id: str) -> None:
        """Raise when a bare id named a source holding more than one tensor.

        The server answers a bare id with the source's default (first) tensor
        (#44), which is right for a single-tensor source and the #75 ambiguity
        otherwise. Only the catalog can tell those apart.

        **Costs a query on every bare-id read.** The trigger -- an echoed
        array_id differing from the one asked for -- is not as narrow as it
        looks: every scene-based adapter (nd2, lif, czi, bioio, ome_zarr,
        ome_tiff) qualifies its ids even for a single-scene file. Hence the
        count rather than the column, which runs to tens of kilobytes on a
        plate. It goes away once ``GetFlightInfo`` reports the substitution
        itself, which all four bindings must adopt in step.
        """
        source_id = split_array_id(array_id)[0]
        count = self._source_tensor_count(source_id)
        if count is not None and count > 1:
            raise _ambiguous_default_error(source_id, count)

    def _source_tensor_count(self, source_id: str) -> Optional[int]:
        """How many tensors the catalog lists for a source, or ``None`` if it
        cannot say -- no row matched, or there is no catalog to ask."""
        row = self._addressed_row("len(tensors) AS tensor_count", source_id)
        return row["tensor_count"] if row else None

    def _source_tensors_row(self, source_id: str) -> Optional[Mapping[str, Any]]:
        """One source's addressing columns: the resolved flag and the tensor list.

        Not ``SOURCE_ROW_COLUMNS`` -- the source's url and type are bytes on the
        wire nobody here reads.
        """
        return self._addressed_row("is_resolved, tensors", source_id)

    def _addressed_row(
        self, columns: str, source_id: str
    ) -> Optional[Mapping[str, Any]]:
        """One source's row, projected to ``columns``, or ``None``.

        ``None`` covers both "no such row" and "no catalog to ask" (a capability
        token reads one source's pixels, not the catalog; an embedded server may
        have none). Every caller here treats them alike: the catalog cannot
        answer, so the per-tensor probe is the path left.
        """
        try:
            table = self._query_table(
                f"SELECT {columns} FROM sources "
                f"WHERE source_id = {sql_literal(source_id)}"
            )
        except flight.FlightError:
            return None
        rows = table.to_pylist()
        return rows[0] if rows else None

    def _iter_action_messages(self, action, msg_cls, *, unknown_action_msg=None):
        """Iterate a streaming ``do_action``, yielding ``(which, msg, body)`` per
        non-empty message.

        The loop shared by :meth:`resolve` / :meth:`warm` / :meth:`add_source`:
        the ``do_action`` call, the empty-body heartbeat skip, the envelope parse
        into ``msg_cls`` (a bad parse yields ``which=None``, which every caller
        ignores -- the SDK refuses a pre-v2 server at connect), and the old-server
        ``"Unknown action"`` -> :class:`RuntimeError` remap -- applied only when
        ``unknown_action_msg`` is given; otherwise the ``FlightServerError``
        propagates unchanged.

        Cancellation is deliberately NOT handled here: its semantics differ per
        caller (resolve/warm raise, add_source returns what it has), and the poll
        must run *after* a message is consumed so a terminal already in hand is
        never discarded by a cancel landing on it (issue #4). Each caller polls
        ``should_cancel`` around its own dispatch.
        """
        try:
            for result in self._state.client.do_action(
                action, options=self._state.call_options
            ):
                body = result.body.to_pybytes()
                if not body:
                    continue  # legacy empty-body heartbeat (server predating progress)
                msg = msg_cls()
                try:
                    msg.ParseFromString(body)
                    which = msg.WhichOneof("payload")
                except Exception:  # noqa: BLE001
                    which = None
                yield which, msg, body
        except flight.FlightServerError as exc:
            if unknown_action_msg is not None and "Unknown action" in str(exc):
                raise RuntimeError(unknown_action_msg) from exc
            raise

    def resolve(
        self,
        source_id: str,
        *,
        on_progress: Optional[Callable[["ResolveProgress"], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Backs TensorFlightClient.resolve; see that method for the full
        documentation."""
        # One dedicated, streaming ``resolve`` action: it is the SINGLE server
        # entry point that performs the (possibly minutes-long) recall, and its
        # terminal message carries the source's now-concrete catalog row -- no
        # GetFlightInfo + list_sources two-step, so no truncation hole for
        # multi-field sources beyond the list cap. The action streams
        # ``ResolveStreamMessage`` heartbeats (a ``progress`` arm) to keep the
        # connection warm under proxy idle timeouts. ``should_cancel`` /
        # ``on_progress`` are polled once per received message, i.e. roughly
        # once per server heartbeat.
        action = flight.Action("resolve", source_id.encode("utf-8"))
        row: Optional[Mapping[str, Any]] = None
        for which, msg, _ in self._iter_action_messages(action, ResolveStreamMessage):
            if should_cancel is not None and should_cancel():
                raise ResolveCancelled(f"resolve('{source_id}') cancelled by caller")
            if which == "progress":
                if on_progress is not None:
                    on_progress(msg.progress)
            elif which == "source_row":
                # The same row list_sources reads, through the same decoder --
                # one representation of a source, so a resolve and a subsequent
                # browse cannot disagree about it.
                rows = pa.ipc.open_stream(msg.source_row).read_all().to_pylist()
                if rows:
                    row = rows[0]
        if row is None:
            raise RuntimeError(
                f"resolve('{source_id}') returned no catalog row "
                "(server closed the stream without a result)"
            )
        return dict(row)

    def warm(
        self,
        source_id: str,
        *,
        on_progress: Optional[Callable[["WarmProgress"], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> "WarmProgress":
        """Backs TensorFlightClient.warm; see that method for the full
        documentation."""
        action = flight.Action("warm", source_id.encode("utf-8"))
        done: Optional[WarmProgress] = None
        unknown = (
            "Hydrate-ahead is unavailable: the tensor server is too old "
            "to support the 'warm' action. Upgrade the server, or just "
            "read the data on demand (it will recall lazily)."
        )
        for which, msg, _ in self._iter_action_messages(
            action, WarmStreamMessage, unknown_action_msg=unknown
        ):
            if should_cancel is not None and should_cancel():
                raise ResolveCancelled(f"warm('{source_id}') cancelled by caller")
            if which == "progress":
                if on_progress is not None:
                    on_progress(msg.progress)
            elif which == "done":
                done = WarmProgress()
                done.CopyFrom(msg.done)
        if done is None:
            raise RuntimeError(
                f"warm('{source_id}') returned no terminal status "
                "(server closed the stream without a 'done')"
            )
        return done

    def get_upload_status(self, source_id: str) -> Dict[str, Any]:
        """Backs TensorFlightClient.get_upload_status; see that method for the full
        documentation.

        A describe-only ``GetFlightInfo``, not an action: ``do_action`` takes
        the server-wide token, while the caller waiting on a fast-return result
        holds a per-source read capability and nothing else (biopb/biopb#1048).
        Describe-only also means the server skips the O(chunks) endpoint
        enumeration, so a poll is one small round trip.

        Lives here rather than on ``UploadSession`` because it stopped being an
        upload operation: it is a read of one field of a descriptor, which is
        this class's primitive.

        Nothing is stored: a poll that answered from a cache would answer with
        the state it came to replace.
        """
        try:
            desc = self._fetch_tensor_descriptor(source_id, with_upload_status=True)
        except (flight.FlightError, _AddressingError):
            # UNKNOWN means "no upload record here", and an id the server does
            # not serve at all is the strongest form of that. `describe` raises;
            # this caller asked a narrower question.
            return _unknown_upload_status(source_id)
        if not desc.HasField("upload_status"):
            # Registered, but not an upload -- an ordinary catalog source.
            # Distinct from PENDING, and no amount of polling moves it
            # (biopb/biopb#109).
            return _unknown_upload_status(source_id)
        return _upload_status_dict(source_id, desc.upload_status)

    def add_source(
        self,
        url: str,
        *,
        source_type: str = "",
        on_progress: Optional[Callable[["AddSourceProgress"], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> "AddSourceResult":
        """Backs TensorFlightClient.add_source; see that method for the full
        documentation."""
        req = AddSourceRequest(
            url=url,
            source_type=source_type,
        )
        action = flight.Action("add_source", req.SerializeToString())
        unknown = (
            "Runtime source registration is unavailable: the tensor "
            "server is too old to support the 'add_source' action. "
            "Upgrade the server, or add the source via its config file."
        )
        result: Optional[AddSourceResult] = None
        for which, msg, _ in self._iter_action_messages(
            action, AddSourceStreamMessage, unknown_action_msg=unknown
        ):
            if which == "progress":
                if on_progress is not None:
                    on_progress(msg.progress)
            elif which == "result":
                result = AddSourceResult()
                result.CopyFrom(msg.result)
            # Poll AFTER consuming this message, not before: a cancel landing
            # exactly on the terminal ``result`` must not discard a completed
            # tally already captured above (issue #4). Closing the stream keeps
            # everything already registered server-side.
            if should_cancel is not None and should_cancel():
                break
        if result is None:
            # A caller-driven cancel breaks before the terminal result; report an
            # empty tally rather than an error (the cancel was intentional).
            if should_cancel is not None and should_cancel():
                return AddSourceResult()
            raise RuntimeError(
                f"add_source('{url}') returned no terminal result "
                "(server closed the stream without a result)"
            )
        return result

    def _do_action_one_result(
        self, action: flight.Action, *, unavailable_hint: str
    ) -> bytes:
        return do_action_one_result(
            self._state, action, unavailable_hint=unavailable_hint
        )

    def remove_source(self, root_url: str) -> "RemoveSourceResult":
        """Backs TensorFlightClient.remove_source; see that method for the full
        documentation."""
        req = RemoveSourceRequest(root_url=root_url)
        action = flight.Action("remove_source", req.SerializeToString())
        result_bytes = self._do_action_one_result(
            action, unavailable_hint="Source removal is unavailable"
        )
        return RemoveSourceResult.FromString(result_bytes)

    # ---- label sets (biopb-tensor-server/docs/label-tensors.md) ----

    def label_sets(self, image_array_id: str) -> List[str]:
        """Backs TensorFlightClient.label_sets; see that method."""
        prefix = sql_literal(f"{image_array_id}/{LABELS_SEGMENT}/")
        table = self._query_table(
            "SELECT t.array_id FROM sources, UNNEST(tensors) AS u(t) "
            f"WHERE starts_with(t.array_id, {prefix}) ORDER BY t.array_id"
        )
        return table.column(0).to_pylist()

    # ---- ROI annotations (biopb-tensor-server/docs/roi-annotations.md) ----

    def list_rois(self, array_id: str, set_name: str = "") -> "RoiListResult":
        """Backs TensorFlightClient.list_rois; see that method."""
        ticket = TensorTicket(roi_read=RoiRead(array_id=array_id, set_name=set_name))
        reader = self._state.client.do_get(
            flight.Ticket(ticket.SerializeToString()), options=self._state.call_options
        )
        table = reader.read_all()
        metadata = table.schema.metadata or {}
        sets = [
            RoiSetInfo(
                set_name=entry["set_name"],
                count=int(entry["count"]),
                reserved=bool(entry.get("reserved")),
            )
            for entry in json.loads(metadata.get(b"sets", b"[]"))
        ]
        return RoiListResult(
            rois=table_to_rois(table),
            truncated=metadata.get(b"truncated", b"").decode() == "True",
            sets=sets,
        )

    def _roi_put_stream(self, cmd: PutCommand, table: pa.Table) -> bytes:
        """One DoPut on the ``roi`` flight: the command in the descriptor, the
        rows in the stream, the structured reply in the put's app_metadata."""
        descriptor = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        writer, reader = self._state.client.do_put(
            descriptor, table.schema, options=self._state.call_options
        )
        with writer:
            if table.num_rows:
                writer.write_table(table)
            writer.done_writing()
            reply = reader.read()
        if reply is None:
            raise RuntimeError("the server acknowledged the ROI put with no result")
        return reply.to_pybytes()

    def put_rois(
        self,
        array_id: str,
        rois: Sequence["RoiAnnotation"],
        *,
        check_rev: bool = False,
    ) -> "RoiPutResult":
        """Backs TensorFlightClient.put_rois; see that method."""
        cmd = PutCommand(roi_put=RoiPut(array_id=array_id, check_rev=check_rev))
        table = rois_to_table(rois) if rois else ROI_ROW_SCHEMA.empty_table()
        return RoiPutResult.FromString(self._roi_put_stream(cmd, table))

    def delete_rois(
        self,
        array_id: str,
        roi_ids: Sequence[str] = (),
        set_name: str = "",
    ) -> "RoiDeleteResult":
        """Backs TensorFlightClient.delete_rois; see that method."""
        cmd = PutCommand(roi_delete=RoiDelete(array_id=array_id, set_name=set_name))
        table = roi_ids_to_table(roi_ids) if roi_ids else ROI_ID_SCHEMA.empty_table()
        return RoiDeleteResult.FromString(self._roi_put_stream(cmd, table))

    def prune_rois(self, unseen_days: int, *, apply: bool = False) -> "RoiPruneResult":
        """Backs TensorFlightClient.prune_rois; see that method."""
        req = RoiPruneRequest(unseen_days=unseen_days, apply=apply)
        action = flight.Action("roi_prune", req.SerializeToString())
        result_bytes = self._do_action_one_result(
            action, unavailable_hint="ROI pruning is unavailable"
        )
        return RoiPruneResult.FromString(result_bytes)


class ChunkFetcher:
    """Tensor reads: GetFlightInfo planning + lazy dask chunk fetching.

    Plans a read against the server (resolving the tensor through the catalog's
    caches) and builds the picklable, lazy dask array whose leaf tasks fetch
    chunks via the worker-side pool in :mod:`biopb.tensor._pool`.
    """

    def __init__(self, state: "_ClientState", catalog: "CatalogClient"):
        self._state = state
        self._catalog = catalog

    def _plan_read(
        self,
        array_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> "flight.FlightInfo":
        """Plan one read: GetFlightInfo the tensor's endpoints.

        The shared body of :meth:`get_tensor` and :meth:`get_tensor_pb`: one
        builds the array from the returned FlightInfo, the other hands the
        FlightInfo on.

        **One RPC for a qualified id with bounded slice bounds** -- the shape
        the tile route issues: the same ``GetFlightInfo`` both plans the read and
        supplies the addressing facts.

        Two shapes cost a catalog query on top, each charged to the request that
        needs it: an open-ended slice ``stop``, which nothing but the tensor can
        fill, and a *bare* id, whose ambiguity only the catalog can judge. The
        second is not rare -- scene-based adapters qualify their array_ids even
        for a single-scene file -- so a caller reading one tensor repeatedly
        should pass the qualified array_id.

        Args:
            array_id: Globally-unique tensor id (identity policy) -- e.g.
                ``"zarr_a3f2"`` or ``"aics_7f3/Image:0"``.
            slice_hint: Optional slice tuple to filter chunks. An open-ended
                ``stop`` is filled from the tensor's shape, which costs a resolve.
            scale_hint: Optional per-dimension downsampling factors
            reduction_method: Optional dynamic reduction method
        """
        logger.debug(f"_plan_read: array_id={array_id}")

        slice_hint_proto = None
        refusals_settled = False
        if slice_hint is not None:
            starts = [s.start if s.start is not None else 0 for s in slice_hint]
            stops = [s.stop for s in slice_hint]
            if any(stop is None for stop in stops):
                # Where the tensor ends is the one thing a request cannot state
                # about itself. The resolve reads the catalog row, so it also
                # makes the addressing refusals the check below would.
                shape = self._catalog._resolve_descriptor(array_id).shape
                refusals_settled = True
                stops = [
                    shape[axis] if stop is None else stop
                    for axis, stop in enumerate(stops)
                ]
            slice_hint_proto = SliceHint(start=starts, stop=stops)

        # Build TensorReadOption with flattened fields. `endpoints` is explicit:
        # this is the read path, and the plan is what it came for.
        read_opt = _read_option(endpoints=True)
        if slice_hint_proto is not None:
            read_opt.slice_hint.CopyFrom(slice_hint_proto)
        if scale_hint is not None:
            read_opt.scale_hint[:] = list(scale_hint)
        if reduction_method is not None:
            read_opt.reduction_method = reduction_method

        # Route on the caller's id: only the caller's prefix is guaranteed to
        # name a registered source. A bare id leaves tensor_id empty and takes
        # the server's default-tensor path (#44).
        cmd = _tensor_read_cmd(array_id, read_opt)

        flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        try:
            info = self._state.client.get_flight_info(
                flight_desc, options=self._state.call_options
            )
        except flight.FlightError as exc:
            _raise_read_refusal(exc, array_id)
            raise

        if not refusals_settled:
            # The server echoes back the tensor it bound, so an array_id that
            # came back changed means it substituted a default (#44). Checked
            # after the response so a qualified id costs nothing.
            response_desc = TensorDescriptor.FromString(info.descriptor.command)
            if response_desc.array_id != array_id:
                self._catalog._refuse_ambiguous_default(array_id)
        return info

    def get_tensor(
        self,
        array_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> da.Array:
        """Backs TensorFlightClient.get_tensor; see that method for the full
        documentation."""
        info = self._plan_read(array_id, slice_hint, scale_hint, reduction_method)
        return _dask_from_flight_info(
            info,
            self._state.location,
            self._state.token,
            self._state.cache_bytes,
            self._state.tls_trust,
        )

    def get_tensor_pb(
        self,
        array_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> SerializedTensor:
        """Backs TensorFlightClient.get_tensor_pb; see that method for the full
        documentation."""
        info = self._plan_read(array_id, slice_hint, scale_hint, reduction_method)
        return SerializedTensor(
            location=self._state.location,
            auth_token=self._state.token or "",
            flight_info=info.serialize(),
        )
