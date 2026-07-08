"""Per-connection catalog + read core for TensorFlightClient.

Extracted from :mod:`biopb.tensor.client` (issue #278 item C). The two
collaborators share the connection and the catalog caches via
:class:`_ClientState`:

- :class:`CatalogClient` -- discovery / metadata / resolve / warm / source
  registration (``list_sources`` / ``query_sources`` / ``resolve`` / ... RPCs).
- :class:`ChunkFetcher` -- tensor reads: plan a read with GetFlightInfo (through
  the catalog's caches) and build the lazy dask chunk-fetching array.

``TensorFlightClient`` holds one of each and delegates its public API to them;
``client.py`` re-exports the module helpers that external callers still import
from ``biopb.tensor.client``.
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

from biopb.tensor._pool import _build_dask_array_from_chunk_map
from biopb.tensor.descriptor_pb2 import (
    AddSourceProgress,
    AddSourceRequest,
    AddSourceResult,
    AddSourceStreamMessage,
    DataSourceDescriptor,
    FlightCmd,
    MetadataQueryOption,
    RemoveSourceRequest,
    RemoveSourceResult,
    ResolveProgress,
    ResolveStreamMessage,
    SliceHint,
    TensorDescriptor,
    TensorReadOption,
    WarmProgress,
    WarmStreamMessage,
)
from biopb.tensor.serialized_pb2 import SerializedEndpoint, SerializedTensor
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket

logger = logging.getLogger(__name__)


@dataclass
class _ClientState:
    """Per-connection state shared by CatalogClient / ChunkFetcher / the facade.

    Holds the Flight connection handles plus the two catalog caches. The caches
    are mutable and shared by reference: every collaborator reads
    ``state.sources`` / ``state.descriptors`` live, and TensorFlightClient
    exposes them as its ``_sources`` / ``_descriptors`` (property + setter) so
    the historical ``client._sources = {...}`` reset semantics still hold.

    ``descriptors`` is keyed by ``(source_id, bare array_id)`` -- array_id alone
    is not unique across sources (issue #45); see :func:`_descriptor_key`.
    """

    client: flight.FlightClient
    call_options: flight.FlightCallOptions
    location: str
    token: Optional[str]
    cache_bytes: int
    sources: Dict[str, DataSourceDescriptor] = field(default_factory=dict)
    descriptors: Dict[Tuple[str, str], TensorDescriptor] = field(default_factory=dict)


class ResolveCancelled(Exception):
    """Raised by :meth:`TensorFlightClient.resolve` when its ``should_cancel``
    callback asks it to stop.

    The client stops consuming the resolve stream and unwinds; the server's
    recall daemon thread runs to completion and caches its result, so a later
    :meth:`resolve` coalesces onto the finished work rather than re-downloading.
    """


@dataclass
class _TensorContext:
    """Internal context returned by _get_tensor_context().

    Contains all parsed flight info needed to build either a dask array
    or a SerializedTensor protobuf.
    """

    descriptor: TensorDescriptor
    endpoints: List[Tuple[bytes, ChunkBounds]]  # (chunk_id, bounds) pairs
    read_opt: TensorReadOption
    original_slice_hint: Optional[SliceHint]
    schema_metadata: Optional[Dict[str, str]] = (
        None  # For SHM transfer feature detection
    )


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


def _fetch_endpoints_via_get_flight_info(
    pb: SerializedTensor,
) -> Tuple[List[bytes], List[ChunkBounds]]:
    """Fetch endpoints from server via GetFlightInfo when not provided in SerializedTensor.

    This is used when the endpoints field in SerializedTensor is empty.
    The client connects to the server and calls GetFlightInfo to get
    the endpoint list for the tensor.

    Args:
        pb: SerializedTensor protobuf (endpoints field empty)

    Returns:
        Tuple of (chunk_ids, chunk_bounds) extracted from FlightInfo
    """
    descriptor = pb.tensor_descriptor

    # Build TensorReadOption from descriptor's fields
    read_opt = TensorReadOption(
        tensor_id=descriptor.array_id,
        with_metadata=False,
    )
    if descriptor.HasField("slice_hint"):
        read_opt.slice_hint.CopyFrom(descriptor.slice_hint)
    if descriptor.scale_hint:
        read_opt.scale_hint[:] = list(descriptor.scale_hint)
    if descriptor.reduction_method:
        read_opt.reduction_method = descriptor.reduction_method

    # FlightCmd.source_id is the slash-free prefix of the array_id (identity
    # policy: array_id is source_id or source_id/field). tensor_id (above)
    # carries the full array_id, which the server reduces to the within-source
    # field -- so this works for multi-tensor SerializedTensors too, not only
    # the single-tensor case where array_id == source_id.
    cmd = FlightCmd(
        source_id=descriptor.array_id.split("/", 1)[0],
        tensor_read=read_opt,
    )

    # Create temporary client for this GetFlightInfo call
    client = flight.FlightClient(pb.location)
    call_options = (
        flight.FlightCallOptions(
            headers=[(b"authorization", f"Bearer {pb.auth_token}".encode())]
        )
        if pb.auth_token
        else flight.FlightCallOptions()
    )

    flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
    info = client.get_flight_info(flight_desc, options=call_options)

    # Check schema version compatibility
    _check_wire_protocol(info.schema)

    # Parse endpoints into chunk info
    chunks = []
    chunk_bounds_list = []
    for endpoint in info.endpoints:
        ticket = TensorTicket.FromString(endpoint.ticket.ticket)
        bounds = ChunkBounds.FromString(endpoint.app_metadata)
        chunks.append(ticket.chunk_id)
        chunk_bounds_list.append(bounds)

    client.close()
    logger.debug(f"_fetch_endpoints_via_get_flight_info: got {len(chunks)} endpoints")

    return chunks, chunk_bounds_list


def _extract_schema_metadata(schema: pa.Schema) -> Optional[Dict[str, str]]:
    """Extract schema metadata as Python dict for feature detection.

    Args:
        schema: PyArrow Schema from FlightInfo

    Returns:
        Dict with metadata key-value pairs, or None if no metadata
    """
    if schema.metadata is None:
        return None

    return {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in schema.metadata.items()
    }


def _parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string to (major, minor, patch) tuple."""
    # Handle dev versions like "0.3.1.dev43+g..."
    base = version_str.split(".dev")[0].split("+")[0]
    parts = base.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (major, minor, patch)


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


def _descriptor_key(source_id: str, array_id: str) -> Tuple[str, str]:
    """Composite, source-unique key for the descriptor cache (issue #45).

    ``array_id`` arrives in two forms depending on the RPC: bare
    (``"Image:0"`` from ``list_sources`` / ``get_descriptor``) or
    source-qualified (``"src/Image:0"`` from an older data endpoint). Strip a
    leading ``"{source_id}/"`` so both forms map to one key and never
    collide across sources or split the cache for the same tensor.
    """
    prefix = f"{source_id}/"
    if array_id.startswith(prefix):
        array_id = array_id[len(prefix) :]
    return (source_id, array_id)


def _resolve_array_id(
    array_id: Optional[str],
    tensor_id: Optional[str],
    _method: str,
    source_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Resolve the ``(source_id, request tensor_id)`` pair for an array_id-first
    read call, per the tensor identity policy.

    A tensor is identified by its globally-unique ``array_id`` ALONE (see the
    policy at the top of ``proto/biopb/tensor/descriptor.proto``). The public
    read methods take that single ``array_id``; the legacy two-argument
    addressing -- a positional ``tensor_id`` or the ``source_id=`` keyword --
    is still accepted but DEPRECATED.

    - legacy form (``tensor_id`` given, or the ``source_id=`` keyword used) ->
      warns; the routing source_id and request tensor_id are used verbatim.
    - else, ``array_id`` contains '/' -> source-qualified id: the routing
      ``source_id`` is the slash-free prefix and the full ``array_id`` is the
      request tensor_id.
    - else -> a bare ``source_id``: return ``tensor_id=None``. The downstream
      resolution is then CALLER-dependent. A single-tensor source always
      resolves to its sole tensor. For a multi-tensor source it differs:
      ``get_tensor``/``get_tensor_pb`` go through ``_get_tensor_context`` and
      raise "tensor_id must be specified" rather than guess (issue #75),
      whereas ``get_physical_scale`` rides ``_fetch_tensor_descriptor``'s
      empty-tensor_id path and anchors on the source's default (first) tensor
      (server-side, #44). So a bare multi-tensor id is not a uniform error.

    Passing BOTH the positional ``array_id`` and the legacy ``source_id=``
    keyword is contradictory addressing and raises ``ValueError``.
    """
    # Reject contradictory addressing: the new positional ``array_id`` and the
    # legacy ``source_id=`` keyword name the routing source two different ways.
    # Check before the back-compat mapping below, which makes them equal.
    if array_id is not None and source_id is not None:
        raise ValueError(
            f"{_method}() received both the array_id (first argument) and the "
            "deprecated 'source_id=' keyword. A tensor is identified by its "
            "array_id alone -- pass only the array_id as the first argument."
        )
    # Back-compat for the legacy ``source_id=`` keyword: the first positional
    # parameter is now ``array_id``, so map a keyword-only source_id onto it.
    if source_id is not None and array_id is None:
        array_id = source_id
    if array_id is None:
        raise TypeError(f"{_method}() missing required argument: 'array_id'")
    if tensor_id is not None or source_id is not None:
        warnings.warn(
            f"The (source_id, tensor_id) addressing of {_method}() is "
            "deprecated and will be removed in a future release: a tensor is "
            "identified by its globally-unique array_id alone (see the identity "
            "policy in proto/biopb/tensor/descriptor.proto). Pass the array_id "
            f"as the single first argument instead -- {_method}('source_id/field'), "
            f"or {_method}('source_id') for a single-tensor source.",
            DeprecationWarning,
            stacklevel=4,
        )
        return array_id, tensor_id
    if "/" in array_id:
        return array_id.split("/", 1)[0], array_id
    return array_id, None


class CatalogClient:
    """Catalog, metadata, and source-lifecycle RPCs over one Flight connection.

    Owns discovery (``list_sources`` / ``query_sources``), per-tensor metadata
    probes, the experimental cloud ``resolve`` / ``warm`` streams, and runtime
    source registration. Reads and writes the shared ``_ClientState`` caches.
    """

    def __init__(self, state: "_ClientState"):
        self._state = state

    def list_sources(self) -> Dict[str, DataSourceDescriptor]:
        """List available data sources.

        Returns:
            Dictionary mapping source_id to DataSourceDescriptor.
            Each DataSourceDescriptor.tensors contains TensorDescriptor info
            with shape/dtype for all tensors in that source.

        Note:
            Results may be truncated if server has max_list_flights_results configured.
            Check schema metadata for truncation info (truncated=True indicates
            more sources exist on server than were returned).
        """
        source_descriptors = {}
        truncated = False
        total_sources = None

        for info in self._state.client.list_flights(options=self._state.call_options):
            source_desc = DataSourceDescriptor.FromString(info.descriptor.command)
            source_descriptors[source_desc.source_id] = source_desc
            # Cache tensor descriptors
            for tensor_desc in source_desc.tensors:
                self._state.descriptors[
                    _descriptor_key(source_desc.source_id, tensor_desc.array_id)
                ] = tensor_desc

            # Check schema metadata for truncation info
            if info.schema.metadata:
                truncated_bytes = info.schema.metadata.get(b"truncated")
                if truncated_bytes:
                    truncated = truncated_bytes.decode() == "True"
                total_sources_bytes = info.schema.metadata.get(b"total_sources")
                if total_sources_bytes:
                    total_sources = int(total_sources_bytes.decode())

        self._state.sources = source_descriptors

        if truncated and total_sources:
            logger.warning(
                f"list_sources: returned {len(source_descriptors)} of {total_sources} sources (truncated)"
            )
        else:
            logger.info(f"list_sources: returned {len(source_descriptors)} sources")

        return source_descriptors

    def query_sources(self, sql: str, *, format: str = "arrow") -> Any:
        """Execute SQL query against server's source metadata database.

        The server-side metadata database is mandatory (biopb/biopb#225), so any
        standard tensor-server supports this. Only an embedded server explicitly
        constructed without a metadata database rejects the query.

        Args:
            sql: SQL query (e.g., "SELECT source_id, source_type FROM sources WHERE dtype='uint16'")
            format: Shape of the returned result:

                - ``"arrow"`` (default) — a ``pyarrow.Table``. This is the
                  historical return type; the default is unchanged for backward
                  compatibility. Zero-copy, and the only format that preserves
                  the schema metadata described under *Note*.
                - ``"pandas"`` — a ``pandas.DataFrame`` (requires pandas).
                - ``"records"`` — a ``list[dict]``, one dict per row.

        Returns:
            The query result in the requested ``format``; an empty query
            returns an empty object of that same type. For ``"pandas"`` and
            ``"records"`` the usual Arrow->Python coercion applies (list
            columns such as ``shape_summary`` become Python lists / object
            dtype, and nullable integer columns may widen to float). For
            ``"pandas"``, NULLs in string columns (e.g. ``metadata_json``) are
            normalized to ``None`` rather than the truthy float ``NaN`` Arrow
            would otherwise produce, so ``if row.metadata_json:`` behaves as
            expected.

        Note:
            The server reports truncation via schema metadata
            (``total_sources`` / ``returned_sources``). Those keys survive only
            on the ``"arrow"`` result; for every format truncation is also
            surfaced via a logged INFO line.

        Raises:
            ValueError: If *format* is not one of the supported values. (SQL
                validation -- forbidden keywords / disallowed tables -- happens
                server-side and surfaces as a Flight error, below, not a
                client-side ValueError.)
            ImportError: If ``format="pandas"`` but pandas is not installed.
            FlightServerError: If the server has no metadata database enabled,
                or rejects the query (e.g. forbidden keywords / disallowed
                tables).

        Example:
            >>> client = TensorFlightClient('grpc://localhost:8815')
            >>> table = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
            >>> table.to_pandas()  # or pass format="pandas" to get a DataFrame
        """
        if format not in ("pandas", "arrow", "records"):
            raise ValueError(
                f"query_sources: unknown format {format!r}; "
                "expected 'pandas', 'arrow', or 'records'"
            )

        cmd = FlightCmd(
            source_id="__metadata_query__",
            metadata_query=MetadataQueryOption(sql=sql),
        )
        descriptor = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = self._state.client.get_flight_info(
            descriptor, options=self._state.call_options
        )

        # Check schema metadata for truncation info
        if info.schema.metadata:
            total_sources = info.schema.metadata.get(b"total_sources")
            if total_sources:
                total = int(total_sources.decode())
                returned = info.schema.metadata.get(b"returned_sources")
                if returned:
                    returned_count = int(returned.decode())
                    if returned_count < total:
                        logger.info(
                            f"query_sources: returned {returned_count} of {total} sources (truncated)"
                        )
                    else:
                        logger.info(f"query_sources: returned {returned_count} sources")

        # Fetch results via DoGet
        if info.endpoints:
            reader = self._state.client.do_get(
                info.endpoints[0].ticket, options=self._state.call_options
            )
            table = reader.read_all()
        else:
            # Empty result
            table = info.schema.empty_table()

        return self._format_query_result(table, format)

    @staticmethod
    def _format_query_result(table: pa.Table, format: str):
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
        """Get source-level OME/vendor metadata as a dict.

        Args:
            source_id: Source identifier

        Returns:
            The source's metadata dict (the format-specific OME/vendor metadata),
            or an empty dict if the source carries none.

        Raises:
            ValueError: If the source is unknown, or unresolved (cloud /
                synced-folder) -- call :meth:`resolve` first.
        """

        if source_id not in self._state.sources:
            self.list_sources()

        source_desc = self._state.sources.get(source_id)
        if source_desc is None:
            raise ValueError(f"Source not found: {source_id}")

        if not source_desc.tensors:
            # Unresolved (cloud / synced-folder) source: tensors are unknown
            # until resolve. Don't silently return {} -- that conflates
            # "unresolved" with "resolved, no metadata" (the line below). Steer
            # the caller to the explicit, consented resolve() instead, matching
            # get_physical_scale / get_tensor (#108). Crucially this stays a
            # cheap read: it must NOT silently recall the whole file the way a
            # resolve-on-serve probe (get_descriptor) would.
            raise _unresolved_source_error(source_id)

        # metadata_json is populated on the descriptor GetFlightInfo returns, so
        # we fetch it via the source's first tensor.
        first_tensor = source_desc.tensors[0]
        cmd = FlightCmd(
            source_id=source_id,
            tensor_read=TensorReadOption(
                tensor_id=first_tensor.array_id,
                with_metadata=True,
            ),
        )
        flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = self._state.client.get_flight_info(
            flight_desc, options=self._state.call_options
        )
        response_desc = TensorDescriptor.FromString(info.descriptor.command)

        if response_desc.metadata_json:
            # The server wraps it as {"type": ..., "dim_label": [...],
            # "metadata": {...}}; return just the inner metadata dict.
            wrapped = json.loads(response_desc.metadata_json)
            return wrapped.get("metadata", {})
        return {}

    def get_physical_scale(
        self,
        array_id: Optional[str] = None,
        tensor_id: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
    ) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dimension physical pixel size + unit for a tensor.

        Returns ``(scale, unit)``: two lists aligned with the tensor's
        ``dim_labels`` (source axis order), or ``None`` when no physical sizes
        are known (an older server, or a format that carries none).

        ``physical_scale``/``physical_unit`` are ``TensorDescriptor`` fields the
        server fills on every ``GetFlightInfo`` (issue #31), so this reads the
        descriptor a prior :meth:`get_tensor` already cached -- no extra RPC when
        it is cached, and it never requests the opt-in ``metadata_json`` field on
        that same descriptor. (Contrast :meth:`get_source_metadata`, which forces
        ``with_metadata`` to ship the whole OME tree; do not dig physical sizes
        out of that -- this is the compact projection meant for display scale.)

        Args:
            array_id: Globally-unique tensor id (identity policy) -- e.g.
                ``"zarr_a3f2"`` or ``"aics_7f3/Image:0"``. A bare single-tensor
                source id resolves to its sole tensor. A bare *multi*-tensor
                source id anchors on the source's default (first) tensor --
                unlike ``get_tensor``, which requires the field be named; pass the
                qualified ``source_id/field`` to target a specific scene.
            tensor_id: DEPRECATED. The legacy ``(source_id, tensor_id)`` form;
                pass the array_id as the single first argument instead.

        Returns:
            ``(scale, unit)`` lists, or ``None`` if no physical scale is known.
        """
        source_id, tensor_id = _resolve_array_id(
            array_id, tensor_id, "get_physical_scale", source_id=source_id
        )
        desc = (
            self._state.descriptors.get(_descriptor_key(source_id, tensor_id))
            if tensor_id
            else None
        )
        if desc is None:
            # Don't silently recall (download) a whole cloud file just to read its
            # pixel size: if the source is known-unresolved, steer the caller to
            # resolve() explicitly -- consistent with get_tensor, and faithful to
            # resolution being a consented act, not a side effect of a metadata
            # probe. (Only catches sources already in the catalog cache; a
            # never-listed id still falls through to the fetch below, same as
            # every other entry point.)
            cached = self._state.sources.get(source_id)
            if cached is not None and not cached.tensors:
                raise _unresolved_source_error(source_id)
            # tensor_id None -> the source's default (first) tensor. A real fetch
            # error (server unreachable, source not found) propagates to the
            # caller -- it must stay distinguishable from "no physical scale
            # recorded", which is the only case that yields None. This matches
            # the pre-#75 contract, where the get_source() fetch was unguarded.
            desc = self._fetch_tensor_descriptor(source_id, tensor_id)
        if not desc.physical_scale:
            return None
        return list(desc.physical_scale), list(desc.physical_unit)

    def _fetch_tensor_descriptor(
        self,
        source_id: str,
        tensor_id: Optional[str] = None,
    ) -> "TensorDescriptor":
        """Fetch one tensor's descriptor directly from the server (internal).

        Backs the public ``get_descriptor`` (the array_id-keyed primitive) and
        the deprecated ``get_source``. Uses the per-tensor ``GetFlightInfo`` RPC,
        which works even when the source is beyond the (truncatable)
        ``list_sources()`` cap. ``tensor_id`` unset/empty -> the source's default
        (first) tensor (#44). This is a CHEAP probe: it does NOT resolve. An
        unresolved (cloud / synced-folder) source raises the directive
        ``_unresolved_source_error`` steering the caller to :meth:`resolve`,
        rather than triggering a download.

        The descriptor is cached in ``self._state.descriptors`` (keyed by the
        echoed-back array_id). ``self._state.sources`` is intentionally NOT touched, so
        a single-tensor probe never clobbers a full enumeration cached by
        ``list_sources()`` (issue #75).
        """
        read_opt = TensorReadOption(with_metadata=True)
        # Anchor on the source's default tensor via the EMPTY-tensor_id path
        # (server resolves it to the first descriptor's qualified array_id, #44)
        # for both the unset case and a bare source_id. Sending the source_id as
        # the tensor_id instead reduces to field=None, which a multi-tensor
        # adapter need not resolve to a default; the empty path is robust and
        # back-compatible. A within-source field is always sent verbatim.
        if tensor_id and tensor_id != source_id:
            read_opt.tensor_id = tensor_id
        cmd = FlightCmd(source_id=source_id, tensor_read=read_opt)
        fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        try:
            info = self._state.client.get_flight_info(
                fd, options=self._state.call_options
            )
        except flight.FlightUnavailableError as exc:
            # GetFlightInfo no longer resolves on serve: an unresolved (cloud /
            # synced-folder) source now refuses with FlightUnavailableError
            # ("Source unresolved ...") instead of silently downloading. Make this
            # a cheap steering probe -- restate it as the shared directive so
            # get_descriptor / get_source point the caller at the explicit,
            # consented resolve(), consistent with get_tensor / get_physical_scale.
            if "unresolved" in str(exc).lower():
                raise _unresolved_source_error(source_id) from exc
            raise
        tensor_desc = TensorDescriptor.FromString(info.descriptor.command)
        self._state.descriptors[_descriptor_key(source_id, tensor_desc.array_id)] = (
            tensor_desc
        )
        return tensor_desc

    def get_descriptor(self, array_id: str) -> "TensorDescriptor":
        """Fetch one tensor's ``TensorDescriptor`` by its globally-unique array_id.

        A tensor is identified by its ``array_id`` alone (see the tensor identity
        policy at the top of ``proto/biopb/tensor/descriptor.proto``), so this
        takes that one identifier rather than a ``(source_id, tensor_id)`` pair.
        Works even when the source is beyond the (truncatable) ``list_sources()``
        cap, and the result is cached. Passing a bare ``source_id`` (single-tensor
        source, or to anchor on a multi-tensor source's default/first tensor) is
        accepted. To enumerate ALL tensors/scenes of a source, use
        ``list_sources()[source_id].tensors`` -- NOT this method.

        This is a cheap probe -- it does NOT resolve. On an unresolved (cloud /
        synced-folder) source it raises an error pointing at :meth:`resolve`,
        never triggering a download. Call :meth:`resolve` first to read such a
        source.

        Args:
            array_id: Globally-unique tensor id, e.g. ``"zarr_a3f2"`` (single-
                tensor source) or ``"aics_7f3/Image:0"`` (multi-tensor source).

        Returns:
            The ``TensorDescriptor`` for that tensor.
        """
        # source_id is the slash-free prefix; the full array_id is the tensor_id.
        source_id = array_id.split("/", 1)[0]
        return self._fetch_tensor_descriptor(source_id, array_id)

    def _iter_action_messages(self, action, msg_cls, *, unknown_action_msg=None):
        """Iterate a streaming ``do_action``, yielding ``(which, msg, body)`` per
        non-empty message.

        The loop shared by :meth:`resolve` / :meth:`warm` / :meth:`add_source`:
        the ``do_action`` call, the empty-body heartbeat skip, the envelope parse
        into ``msg_cls`` (a bad parse yields ``which=None`` so a legacy bare-body
        caller can fall back on the raw ``body``), and the old-server
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
    ) -> "DataSourceDescriptor":
        """Resolve an unresolved source and return its full ``DataSourceDescriptor``.

        .. note:: Experimental. Cloud / remote source support (unresolved sources,
           resolve, and :meth:`warm`) is experimental and its behavior may change.

        An *unresolved* source is catalogued by URL only -- its shape/dtype/field
        list are unknown until first access (it lists with ``data_resident`` False
        and an empty ``list_sources()[source_id].tensors``). The canonical case is
        a cloud / synced-folder ("Files-On-Demand") source.

        Resolving asks the server to hydrate it. For a dehydrated placeholder this
        **downloads the whole file** -- a recall that can take minutes, consume
        local disk, and fail when offline -- then reads its real shape, dtype, and
        field list. This is the heavyweight, *consenting* operation that catalog
        browsing (:meth:`list_sources` / :meth:`query_sources`) deliberately
        avoids; call it only when you intend to read the data. After it returns,
        :meth:`get_tensor` and friends work normally.

        Idempotent: resolving an already-resolved source just re-fetches it.

        Args:
            source_id: The source to resolve (e.g. ``"onedrive_a3f2"``).
            on_progress: Optional callback invoked with a ``ResolveProgress``
                (elapsed seconds, target name, target size in bytes) on each
                server heartbeat, so a caller can display progress. Called on the
                calling thread; keep it cheap and non-blocking.
            should_cancel: Optional predicate polled on each heartbeat; when it
                returns True the client stops consuming the stream and raises
                :class:`ResolveCancelled`. The server-side recall continues to
                completion and is cached, so a later ``resolve`` reuses it.

        Returns:
            The full ``DataSourceDescriptor`` with every tensor/field enumerated
            -- the complete field set in one call, regardless of catalog size.

        Raises:
            ResolveCancelled: if ``should_cancel`` asked to stop mid-resolve.
        """
        # One dedicated, streaming ``resolve`` action: it is the SINGLE server
        # entry point that performs the (possibly minutes-long) recall, and it
        # returns the full DataSourceDescriptor directly -- no GetFlightInfo +
        # list_sources two-step, so no truncation hole for multi-field sources
        # beyond the list cap. The action streams ``ResolveStreamMessage``
        # heartbeats (a ``progress`` arm) to keep the connection warm under proxy
        # idle timeouts; the single terminal message carries the descriptor in
        # its ``result`` arm. ``should_cancel`` / ``on_progress`` are polled once
        # per received message, i.e. roughly once per server heartbeat.
        action = flight.Action("resolve", source_id.encode("utf-8"))
        desc: Optional[DataSourceDescriptor] = None
        for which, msg, body in self._iter_action_messages(
            action, ResolveStreamMessage
        ):
            if should_cancel is not None and should_cancel():
                raise ResolveCancelled(f"resolve('{source_id}') cancelled by caller")
            if which == "progress":
                if on_progress is not None:
                    on_progress(msg.progress)
            elif which == "result":
                desc = DataSourceDescriptor()
                desc.CopyFrom(msg.result)
            else:
                # Legacy server: a non-empty body IS a bare serialized
                # DataSourceDescriptor (pre-envelope protocol).
                desc = DataSourceDescriptor.FromString(body)
        if desc is None:
            raise RuntimeError(
                f"resolve('{source_id}') returned no descriptor "
                "(server closed the stream without a result)"
            )
        self._state.sources[source_id] = desc
        return desc

    def warm(
        self,
        source_id: str,
        *,
        on_progress: Optional[Callable[["WarmProgress"], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> "WarmProgress":
        """Hydrate-ahead: recall a resolved source's member files on the server.

        .. note:: Experimental. Cloud / remote source support (:meth:`resolve` and
           this hydrate-ahead path) is experimental and its behavior may change.

        :meth:`resolve` populates a source's *metadata* but, for a multi-file
        cloud source (zarr / ome-zarr / ndtiff / tiff-sequence / micromanager),
        leaves the bulk pixel data dehydrated -- each member file then recalls
        one-at-a-time, slowly, the first time a read touches it (the viewer
        scrubbing planes is the worst case). ``warm`` opts into pulling them all
        resident up front so later reads never stall.

        The recall happens **entirely server-side** (the server walks the source
        directory and reads each file to force the sync engine's recall); no
        pixels cross the wire, only progress. It is idempotent -- already-resident
        files are cheap local reads -- so a ``warm`` re-run after a cancel simply
        finishes the remainder. Only meaningful for multi-file sources; a
        single-file source returns immediately (resolve already recalled it).

        Args:
            source_id: The (already-resolved) source to warm.
            on_progress: Optional callback invoked with a ``WarmProgress``
                (files/bytes done vs total, current file name, elapsed) on each
                progress message. Called on the calling thread; keep it cheap.
            should_cancel: Optional predicate polled per message; when it returns
                True the client closes the stream -- which the server observes and
                stops the recall promptly -- and this raises
                :class:`ResolveCancelled`. Files already recalled stay resident.

        Returns:
            The terminal ``WarmProgress`` snapshot (``files_done`` /
            ``bytes_done`` reflect what was made resident; on a no-op source
            ``files_total == 0``).

        Raises:
            ResolveCancelled: if ``should_cancel`` asked to stop mid-warm.
            RuntimeError: if the server predates the ``warm`` action (too old for
                hydrate-ahead), or closes the stream without a terminal status.
        """
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

    def add_source(
        self,
        url: str,
        *,
        source_type: str = "",
        dim_labels: Optional[List[str]] = None,
        on_progress: Optional[Callable[["AddSourceProgress"], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> "AddSourceResult":
        """Register a local path on the SERVER as a served source at runtime.

        This is the wire entrypoint behind the tensor-browser's drag-drop: it
        hands the server a filesystem path (or directory) that it interprets on
        *its own* filesystem, and the server routes it through the same claim ->
        adapter -> catalog pipeline the directory watcher uses. A dropped
        directory that is not itself a dataset is walked recursively and may
        register several sources, so the action streams progress and a final
        tally rather than returning a single descriptor.

        The path must exist on the server. Because a dropped directory's walk has
        no known size up front, there is no percentage -- progress is a running
        count of sources registered so far.

        Args:
            url: Absolute path (or directory) on the server's filesystem.
            source_type: Explicit adapter type (e.g. ``"zarr"``, ``"ome-zarr"``);
                empty means auto-detect via the adapters' claim protocol.
            dim_labels: Optional dimension labels for the registered tensor(s).
            on_progress: Optional callback invoked with an ``AddSourceProgress``
                (count + current path + last descriptor) per source as it
                registers. Called on the calling thread; keep it cheap.
            should_cancel: Optional predicate polled per message; when it returns
                True the client closes the stream, which the server observes and
                stops discovery -- sources already registered stay registered.

        Returns:
            The terminal ``AddSourceResult`` (``added`` descriptors,
            ``already_present`` source_ids, ``failed`` ``(path, reason)`` pairs).
            A directory dropped above the large-scan threshold comes back as a
            ``failed`` entry, not a special flag.

        Raises:
            flight.FlightServerError: whole-request failure (path not found /
                unreadable on the server, or the server declines the request).
            RuntimeError: the server predates the ``add_source`` action, or
                closed the stream without a terminal result.
        """
        req = AddSourceRequest(
            url=url,
            source_type=source_type,
            dim_labels=dim_labels or [],
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

    def remove_source(self, root_url: str) -> "RemoveSourceResult":
        """Deregister a drag-dropped source branch on the SERVER at runtime.

        The narrow counterpart to :meth:`add_source`: it removes ONLY
        drag-dropped sources, which the server identifies by the ``dnd://``
        origin scheme on their catalog ``source_url``. ``root_url`` is such a
        branch root (a ``dnd://...`` value); every source at or under it is
        removed as a unit. A non-``dnd://`` ``root_url`` is refused by the server.

        Args:
            root_url: The ``dnd://`` branch root to remove (from the browser's
                dropped-root node).

        Returns:
            A ``RemoveSourceResult`` with ``removed`` (source_ids) and ``failed``
            (``AddSourceFailure`` whose ``path`` carries the source_id).

        Raises:
            flight.FlightServerError: the server refused the request (e.g. a
                non-``dnd://`` root, or removal not enabled).
            RuntimeError: the server predates the ``remove_source`` action, or
                returned no result.
        """
        req = RemoveSourceRequest(root_url=root_url)
        action = flight.Action("remove_source", req.SerializeToString())
        try:
            results = self._state.client.do_action(
                action, options=self._state.call_options
            )
            result_bytes = next(results)
        except flight.FlightError as exc:
            if "Unknown action" in str(exc):
                raise RuntimeError(
                    "Source removal is unavailable: the tensor server is too old "
                    "to support the 'remove_source' action. Upgrade the server."
                ) from exc
            raise
        except StopIteration as exc:
            raise RuntimeError(
                f"remove_source('{root_url}') returned no result"
            ) from exc
        return RemoveSourceResult.FromString(result_bytes.body.to_pybytes())

    def get_source(
        self,
        source_id: str,
        tensor_id: Optional[str] = None,
    ) -> "DataSourceDescriptor":
        """DEPRECATED -- use :meth:`get_descriptor` (one tensor) or
        :meth:`list_sources` (all tensors of a source) instead.

        This method is inconsistent with the tensor identity policy
        (``proto/biopb/tensor/descriptor.proto``): it splits a tensor's identity
        into a ``(source_id, tensor_id)`` pair, whereas a tensor is identified by
        its globally-unique ``array_id`` alone. It is also misnamed -- despite
        returning a ``DataSourceDescriptor``, it is a single-tensor probe: the
        returned ``.tensors`` holds ONLY the resolved tensor (or the source's
        default/first tensor when ``tensor_id`` is None), never the full scene
        list. For a multi-scene source, ``get_source(id).tensors`` is length-1
        and scenes 2..N are NOT enumerated here. Use
        ``list_sources()[id].tensors`` for the complete enumeration.

        Args:
            source_id: Data source identifier.
            tensor_id: Optional tensor to anchor the lookup; None -> the source's
                default (first) tensor.

        Returns:
            DataSourceDescriptor wrapping the single resolved TensorDescriptor.
        """
        warnings.warn(
            "TensorFlightClient.get_source() is deprecated and will be removed in "
            "a future release: it is inconsistent with the array_id identity "
            "policy and returns only a single tensor, not a source's full scene "
            "list. Use get_descriptor(array_id) for one tensor, or list_sources() "
            "to enumerate all tensors of a source.",
            DeprecationWarning,
            stacklevel=3,
        )
        tensor_desc = self._fetch_tensor_descriptor(source_id, tensor_id)
        source_desc = DataSourceDescriptor(source_id=source_id, tensors=[tensor_desc])
        # Do NOT clobber a full enumeration previously cached by list_sources();
        # only seed _sources when this source is otherwise unknown (issue #75).
        if source_id not in self._state.sources:
            self._state.sources[source_id] = source_desc
        return source_desc


class ChunkFetcher:
    """Tensor reads: GetFlightInfo planning + lazy dask chunk fetching.

    Plans a read against the server (resolving the tensor through the catalog's
    caches) and builds the picklable, lazy dask array whose leaf tasks fetch
    chunks via the worker-side pool in :mod:`biopb.tensor._pool`.
    """

    def __init__(self, state: "_ClientState", catalog: "CatalogClient"):
        self._state = state
        self._catalog = catalog

    def _get_tensor_context(
        self,
        source_id: str,
        tensor_id: Optional[str] = None,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> _TensorContext:
        """Get flight info context for a tensor (internal helper).

        This is a shared helper used by both get_tensor() and get_tensor_pb()
        to avoid code duplication. It handles all the common logic for:
        - Source validation and tensor resolution
        - Slice hint conversion
        - TensorReadOption building
        - FlightCmd construction
        - GetFlightInfo call and endpoint parsing

        Args:
            source_id: Data source identifier
            tensor_id: Tensor identifier (optional if source has single tensor)
            slice_hint: Optional slice tuple to filter chunks
            scale_hint: Optional per-dimension downsampling factors
            reduction_method: Optional dynamic reduction method

        Returns:
            _TensorContext with descriptor, endpoints, read_opt, and original_slice_hint
        """
        logger.debug(
            f"_get_tensor_context: source_id={source_id}, tensor_id={tensor_id}"
        )

        # Ensure sources are loaded; fall back to direct server fetch if list_sources
        # didn't return this source (e.g. truncated result set).
        if source_id not in self._state.sources:
            self._catalog.list_sources()
        if source_id not in self._state.sources:
            logger.debug(
                f"Source '{source_id}' not in list_sources() result, fetching directly"
            )
            try:
                td = self._catalog._fetch_tensor_descriptor(source_id, tensor_id)
                self._state.sources[source_id] = DataSourceDescriptor(
                    source_id=source_id, tensors=[td]
                )
            except Exception:
                pass  # let the ValueError below surface the clean message

        source_desc = self._state.sources.get(source_id)
        if source_desc is None:
            raise ValueError(f"Source not found: {source_id}")

        # Resolve tensor_id if not provided
        if tensor_id is None:
            if len(source_desc.tensors) == 1:
                tensor_id = source_desc.tensors[0].array_id
            elif len(source_desc.tensors) == 0:
                raise _unresolved_source_error(source_id)
            else:
                raise ValueError(
                    f"Source '{source_id}' has multiple tensors ({len(source_desc.tensors)}), "
                    f"tensor_id must be specified"
                )

        # Find tensor descriptor to get shape for slice validation; fall back to a
        # direct server fetch when the cached source descriptor is stale or partial.
        tensor_desc = None
        for desc in source_desc.tensors:
            if desc.array_id == tensor_id:
                tensor_desc = desc
                break

        if tensor_desc is None:
            logger.debug(
                f"Tensor '{tensor_id}' not in local catalog for source '{source_id}', "
                f"fetching descriptor from server"
            )
            try:
                self._catalog._fetch_tensor_descriptor(source_id, tensor_id)
            except Exception:
                pass  # let the ValueError below surface the clean message
            tensor_desc = self._state.descriptors.get(
                _descriptor_key(source_id, tensor_id)
            )

        if tensor_desc is None:
            raise ValueError(f"Tensor '{tensor_id}' not found in source '{source_id}'")

        # Convert slice_hint to SliceHint proto
        slice_hint_proto = None
        if slice_hint is not None:
            starts = []
            stops = []
            for s in slice_hint:
                starts.append(s.start if s.start is not None else 0)
                stops.append(
                    s.stop if s.stop is not None else tensor_desc.shape[len(starts) - 1]
                )
            slice_hint_proto = SliceHint(start=starts, stop=stops)

        # Build TensorReadOption with flattened fields
        read_opt = TensorReadOption(
            tensor_id=tensor_id,
            with_metadata=False,
        )
        if slice_hint_proto is not None:
            read_opt.slice_hint.CopyFrom(slice_hint_proto)
        if scale_hint is not None:
            read_opt.scale_hint[:] = list(scale_hint)
        if reduction_method is not None:
            read_opt.reduction_method = reduction_method

        # Build FlightCmd for the request
        cmd = FlightCmd(
            source_id=source_id,
            tensor_read=read_opt,
        )

        # Get flight info
        flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        info = self._state.client.get_flight_info(
            flight_desc, options=self._state.call_options
        )
        response_desc = TensorDescriptor.FromString(info.descriptor.command)

        # Check schema version compatibility
        _check_wire_protocol(info.schema)

        # Extract schema metadata for SHM transfer feature detection
        schema_metadata = _extract_schema_metadata(info.schema)

        # Cache the response descriptor
        self._state.descriptors[_descriptor_key(source_id, response_desc.array_id)] = (
            response_desc
        )

        # Parse endpoints into (chunk_id, bounds) pairs
        endpoints = []
        for endpoint in info.endpoints:
            ticket = TensorTicket.FromString(endpoint.ticket.ticket)
            bounds = ChunkBounds.FromString(endpoint.app_metadata)
            endpoints.append((ticket.chunk_id, bounds))

        return _TensorContext(
            descriptor=response_desc,
            endpoints=endpoints,
            read_opt=read_opt,
            original_slice_hint=slice_hint_proto,
            schema_metadata=schema_metadata,
        )

    def get_tensor(
        self,
        array_id: Optional[str] = None,
        tensor_id: Optional[str] = None,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
    ) -> da.Array:
        """Get a lazy dask array for a tensor, addressed by its array_id.

        Args:
            array_id: Globally-unique tensor id (identity policy) -- e.g.
                ``"zarr_a3f2"`` for a single-tensor source or
                ``"aics_7f3/Image:0"`` for a multi-tensor source.
            tensor_id: DEPRECATED. The legacy ``(source_id, tensor_id)`` form;
                pass the array_id as the single first argument instead.
            slice_hint: Optional slice tuple to filter chunks
            scale_hint: Optional per-dimension integer downsampling factors
            reduction_method: Optional dynamic reduction method for scaled reads

        Returns:
            dask.array with lazy chunk loading

        Raises:
            ValueError: If source not found, tensor not found, or a bare
                multi-tensor source id is given without a within-source field
        """
        source_id, tensor_id = _resolve_array_id(
            array_id, tensor_id, "get_tensor", source_id=source_id
        )
        logger.debug(f"get_tensor: source_id={source_id}, tensor_id={tensor_id}")

        # Get flight info context
        ctx = self._get_tensor_context(
            source_id=source_id,
            tensor_id=tensor_id,
            slice_hint=slice_hint,
            scale_hint=scale_hint,
            reduction_method=reduction_method,
        )

        # Build dask array from context
        chunks = [ep[0] for ep in ctx.endpoints]
        chunk_bounds_list = [ep[1] for ep in ctx.endpoints]
        dask_arr = self._build_dask_array(
            desc=ctx.descriptor,
            chunks=chunks,
            chunk_bounds=chunk_bounds_list,
            schema_metadata=ctx.schema_metadata,
        )

        # Crop to the originally requested region.
        # The server snaps slice_hint outward to lcm-aligned chunk boundaries, so
        # the returned descriptor.shape may be larger than what was requested.
        # We crop the dask array back to the exact requested region here.
        if ctx.original_slice_hint is not None and ctx.descriptor.HasField(
            "slice_hint"
        ):
            dask_arr = dask_arr[
                _request_crop_slices(
                    len(ctx.descriptor.shape),
                    ctx.original_slice_hint,
                    ctx.descriptor.slice_hint,
                    list(ctx.read_opt.scale_hint) if ctx.read_opt.scale_hint else None,
                )
            ]

        return dask_arr

    def get_tensor_pb(
        self,
        array_id: Optional[str] = None,
        tensor_id: Optional[str] = None,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
        reduction_method: Optional[str] = None,
        *,
        source_id: Optional[str] = None,
    ) -> SerializedTensor:
        """Get a SerializedTensor protobuf for cross-process transfer.

        Returns a protobuf containing connection info and chunk tickets
        for lazy reconstruction. The protobuf can be serialized to bytes
        and broadcast to worker processes, where each worker can call
        tensor_from_pb() to reconstruct a lazy dask array.

        Args:
            array_id: Globally-unique tensor id (identity policy) -- e.g.
                ``"zarr_a3f2"`` or ``"aics_7f3/Image:0"``.
            tensor_id: DEPRECATED. The legacy ``(source_id, tensor_id)`` form;
                pass the array_id as the single first argument instead.
            slice_hint: Optional slice tuple to filter chunks
            scale_hint: Optional per-dimension integer downsampling factors
            reduction_method: Optional dynamic reduction method for scaled reads

        Returns:
            SerializedTensor protobuf object
        """
        source_id, tensor_id = _resolve_array_id(
            array_id, tensor_id, "get_tensor_pb", source_id=source_id
        )
        logger.debug(f"get_tensor_pb: source_id={source_id}, tensor_id={tensor_id}")

        # Get flight info context
        ctx = self._get_tensor_context(
            source_id=source_id,
            tensor_id=tensor_id,
            slice_hint=slice_hint,
            scale_hint=scale_hint,
            reduction_method=reduction_method,
        )

        # Serialize endpoints
        serialized_endpoints = []
        for chunk_id, bounds in ctx.endpoints:
            ticket = TensorTicket(chunk_id=chunk_id)
            serialized_ep = SerializedEndpoint(
                ticket=ticket,
                chunk_bounds=bounds,
            )
            serialized_endpoints.append(serialized_ep)

        # Build SerializedTensor
        serialized_tensor = SerializedTensor(
            tensor_descriptor=ctx.descriptor,
            location=self._state.location,
            auth_token=self._state.token or "",
            endpoints=serialized_endpoints,
        )
        if ctx.original_slice_hint is not None:
            serialized_tensor.original_slice_hint.CopyFrom(ctx.original_slice_hint)

        # Add schema metadata for SHM transfer feature detection
        if ctx.schema_metadata is not None:
            serialized_tensor.schema_metadata.update(ctx.schema_metadata)

        return serialized_tensor

    def _build_dask_array(
        self,
        desc: TensorDescriptor,
        chunks: List[bytes],
        chunk_bounds: List[ChunkBounds],
        schema_metadata: Optional[Dict[str, str]] = None,
    ) -> da.Array:
        """Build a dask array from chunk info.

        Args:
            desc: Tensor descriptor
            chunks: List of chunk IDs
            chunk_bounds: List of chunk bounds
            schema_metadata: Optional schema metadata for SHM transfer feature detection

        Returns:
            dask.array with lazy chunk loading
        """
        shape = tuple(desc.shape)
        dtype = np.dtype(desc.dtype)

        # Create a mapping from chunk index to chunk_id and bounds
        chunk_map = {}
        axis_starts = [
            sorted({int(bounds.start[axis]) for bounds in chunk_bounds})
            for axis in range(len(shape))
        ]
        axis_index_maps = [
            {start: index for index, start in enumerate(starts)}
            for starts in axis_starts
        ]
        for chunk_id, bounds in zip(chunks, chunk_bounds):
            chunk_idx = tuple(
                axis_index_maps[d][int(bounds.start[d])] for d in range(len(shape))
            )
            chunk_map[chunk_idx] = (chunk_id, bounds)

        # The actual fetch is done by _fetch_chunk_distributed which uses
        # module-level pools; _build_dask_array_from_chunk_map emits a single
        # Blockwise (map_blocks) layer for a regular grid, falling back to
        # da.block-of-from_delayed for ragged/sparse grids.
        ndim = len(shape)
        grid_shape = tuple(max(idx[d] + 1 for idx in chunk_map) for d in range(ndim))

        return _build_dask_array_from_chunk_map(
            chunk_map,
            grid_shape,
            shape,
            dtype,
            self._state.location,
            self._state.token,
            self._state.cache_bytes,
            schema_metadata,
        )
