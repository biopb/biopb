"""Remote tensor-server adapter -- a caching passthrough proxy (biopb/biopb#178).

A ``RemoteTensorAdapter`` fronts a *single tensor source on an upstream biopb
tensor server* and re-serves it from the local server. One instance is bound to
``(upstream_location, upstream_source_id, local_source_id)``; the local server
registers it under ``local_source_id`` like any other adapter. Because the
local-server data path already wraps every adapter's chunk read in the segment
cache (``TensorAdapter.resolve_chunk_data`` -> ``CacheManager``), a passthrough
adapter inherits the persistent ``ArrowFileBackend`` cache, eviction, and the
localhost ``chunk_locate`` mmap fast path *for free* -- the proxy adds no caching
code of its own.

The adapter is, by design, **format-agnostic and chunking-agnostic**. It decodes
no pixels and re-derives no chunk grid, and it treats the upstream's chunk_id as
OPAQUE: a served chunk_id is a **proxy envelope** (``chunk.encode_proxy_envelope``)
wrapping the upstream chunk_id byte-for-byte, plus a local route and the upstream's
content_version. A later ``do_get`` peels the envelope and forwards the inner
VERBATIM -- no decode, no rewrite of the upstream id (biopb/biopb#178 W1). The
upstream array_id is read once at flight-info time only to build the local route.

Scope of this slice (§2 of ``docs/remote-tensor-cache.md``): the adapter + its
data path, constructible directly (and via ``create_from_config`` for the
single-source ``grpc://host:port/<upstream_source_id>`` url form). Catalog
expansion of a bare ``grpc://host:port`` into one source per upstream tensor,
alias namespacing of the registered ``source_id``, the collision check, and the
monitor->re-list refresh are the next slice (§3).
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import (
    FlightCmd,
    TensorDescriptor,
    TensorReadOption,
)
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket

from biopb_tensor_server.core.base import (
    TensorAdapter,
    TensorReadPlan,
    unpack_chunk_array,
)
from biopb_tensor_server.core.chunk import (
    ChunkEndpoint,
    cache_key_for_chunk_id,
    decode_chunk_id,
    encode_chunk_id,
    encode_proxy_envelope,
    is_proxy_envelope,
    is_scaled_chunk,
    peel_proxy_envelope,
)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig

logger = logging.getLogger(__name__)

# Env-var convenience fallback for the single-upstream case (a per-upstream
# credentials profile -- storage_type="biopb-tensor" -- is the multi-upstream
# path, wired in the §3 expansion slice).
_UPSTREAM_TOKEN_ENV = "BIOPB_UPSTREAM_TENSOR_TOKEN"


# Process-wide pool of upstream clients, keyed by (endpoint, token) so N mirrored
# sources of one upstream share a single connection instead of each opening its
# own (biopb/biopb#266 B1, the former #249). Distinct tokens stay isolated (they
# authenticate as different principals). Entries live until the process exits or
# an unreachable failure evicts one: there is no per-source teardown (nothing
# closes an upstream client today -- an adapter is dropped by GC), so a handful
# of endpoints means a handful of clients. A dead entry is evicted on failure and
# the next access rebuilds it.
_CLIENT_POOL: Dict[Tuple[str, Optional[str]], Any] = {}
_CLIENT_POOL_LOCK = threading.Lock()


def _pooled_upstream_client(location: str, token: Optional[str]):
    """Return the shared ``TensorFlightClient`` for ``(location, token)``.

    Built lazily on first use and cached process-wide so mirrored sources of the
    same upstream reuse one connection (biopb/biopb#266 B1). Construction happens
    under the pool lock; ``TensorFlightClient`` opens its socket lazily, so the
    critical section is short.
    """
    key = (location, token)
    with _CLIENT_POOL_LOCK:
        client = _CLIENT_POOL.get(key)
        if client is None:
            from biopb.tensor import TensorFlightClient

            client = TensorFlightClient(location, cache_bytes=0, token=token)
            _CLIENT_POOL[key] = client
        return client


def _evict_pooled_upstream_client(location: str, token: Optional[str]) -> None:
    """Drop the pooled client for ``(location, token)`` so the next access rebuilds.

    Called when an upstream call fails: the shared connection may be dead, so
    remove it and best-effort close it. Other adapters still holding a reference
    to the old client rediscover the failure on their own next call and re-evict
    (idempotent) -- the same per-adapter reconnect behavior as before pooling.
    """
    key = (location, token)
    with _CLIENT_POOL_LOCK:
        client = _CLIENT_POOL.pop(key, None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def _clear_client_pool() -> None:
    """Close and drop all pooled clients (test isolation / shutdown helper)."""
    with _CLIENT_POOL_LOCK:
        clients = list(_CLIENT_POOL.values())
        _CLIENT_POOL.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


def _split_grpc_url(url: str) -> tuple[str, Optional[str]]:
    """Split ``grpc://host:port[/upstream_source_id]`` into (endpoint, source_id).

    The path component, when present, is the upstream ``source_id`` (slash-free by
    the array_id spec, so the first '/' after the authority cleanly separates the
    endpoint from the source). Returns ``(endpoint, None)`` for the bare-host form
    (mirror every source -- handled by the §3 expansion, not this adapter).
    """
    parts = urlsplit(url)
    endpoint = f"{parts.scheme}://{parts.netloc}"
    source_id = parts.path.lstrip("/") or None
    return endpoint, source_id


def list_upstream_source_ids(client, location: str) -> tuple[List[str], bool]:
    """Every source_id on an upstream tensor server. Returns ``(ids, complete)``.

    ``location`` is the upstream endpoint, named in the fallback warning. It is a
    parameter rather than something read off the client because the callers
    already computed it (``_split_grpc_url``) to build the client, and the SDK
    exposes no public accessor -- reaching for ``client._location`` was borrowing
    another package's private state to recover a value that was in scope
    (biopb/biopb#529).

    Enumerating a catalog with ``list_sources()`` is **unsafe**: it is capped at
    the server's ``max_list_flights_results``, so a large upstream is silently
    truncated -- mirroring it would drop sources, and reconciling against a
    truncated list would spuriously *remove* the ones past the cap. Use the
    server-side DuckDB catalog instead (``query_sources`` -- complete, not
    truncated; the canonical browse surface, biopb/biopb#225). Fall back to the
    capped ``list_sources()`` only when the upstream has no metadata DB, and flag
    the result ``complete=False`` so a caller (e.g. the monitor re-list) can avoid
    destructive reconciliation on a partial list.
    """
    try:
        rows = client.query_sources("SELECT source_id FROM sources", format="records")
        return [row["source_id"] for row in rows], True
    except Exception as exc:
        ids = list(client.list_sources().keys())
        logger.warning(
            "upstream %s has no SQL catalog (query_sources failed: %s); falling "
            "back to the capped list_sources() -- the mirror may be incomplete "
            "(%d sources seen). Enable the upstream's metadata DB for a complete "
            "mirror.",
            location,
            exc,
            len(ids),
        )
        return ids, False


def fetch_upstream_catalog(client, location: str) -> tuple[Optional[List[dict]], bool]:
    """Bulk-fetch an upstream's full catalog rows in ONE ``query_sources``.

    Returns ``(rows, complete)``. Each row is a dict with ``source_id``,
    ``source_url``, ``source_type``, ``metadata_json``, ``data_resident``, the
    per-tensor ``tensors`` STRUCT[] (biopb/biopb#224), and ``indexed_at`` (the
    upstream's per-source register timestamp) -- everything needed to seed a
    mirrored source's catalog entry without a per-source upstream RPC
    (biopb/biopb#266). ``source_url`` carries the upstream's real path so the
    mirror can be treed by filepath in the browser (biopb/biopb#297).
    ``indexed_at`` becomes the mirror's content_version (biopb/biopb#178): it
    changes when the upstream re-registers the source, so the proxy's chunk cache
    re-namespaces instead of serving stale chunks.
    ``data_resident`` is carried so an unresolved upstream
    source (``data_resident=false``, empty ``tensors``) mirrors as non-resident
    rather than being advertised resident. ``complete`` is True because the
    server-side DuckDB catalog is not truncated like ``list_sources()``.

    ``rows`` is ``None`` when the upstream has no SQL catalog (``query_sources``
    errors) -- the caller then falls back to id-only enumeration
    (``list_upstream_source_ids``) and the per-source live sync path.

    ``location`` names the upstream in the fallback warning; see
    :func:`list_upstream_source_ids` for why it is a parameter.
    """
    try:
        rows = client.query_sources(
            "SELECT source_id, source_url, source_type, metadata_json, "
            "data_resident, tensors, indexed_at FROM sources",
            format="records",
        )
        return rows, True
    except Exception as exc:
        logger.warning(
            "upstream %s bulk catalog fetch failed (%s); falling back to id-only "
            "enumeration + per-source sync",
            location,
            exc,
        )
        return None, False


class RemoteTensorAdapter(TensorAdapter):
    """Caching passthrough proxy for one source on an upstream tensor server."""

    _single_tensor_source = False  # an upstream source may carry several tensors

    def __init__(
        self,
        source_id: str,
        upstream_location: str,
        upstream_source_id: str,
        *,
        token: Optional[str] = None,
        tensor_name: Optional[str] = None,
        alias: Optional[str] = None,
    ):
        """Bind a proxy to one upstream source.

        Args:
            source_id: The LOCAL source_id this adapter is registered under
                (slash-free; alias-namespaced by the §3 expansion when fronting
                multiple upstreams). For a lone upstream it may equal the
                upstream source_id.
            upstream_location: ``grpc://host:port`` of the upstream server.
            upstream_source_id: The source_id on the upstream server.
            token: Bearer token for the upstream (auth). ``None`` disables auth.
            tensor_name: Within-source field for a tensor-layer view (set by
                ``get_tensor_adapter``); ``None`` is the source / default tensor.
            alias: The configured namespace alias for this upstream, used only to
                build a display-friendly catalog ``source_url`` (``<alias>:<id>``);
                ``None`` falls back to the upstream host.
        """
        self.source_id = source_id
        self._source_type = "tensor-server"
        self._tensor_name = tensor_name
        self._alias = alias
        # Display authority for the catalog source_url: the alias, or the
        # host:port when there is none. (self._upstream_location keeps the real
        # endpoint for dialing.)
        self._authority = alias or (
            urlsplit(upstream_location).netloc or upstream_location
        )
        # Display-friendly catalog source_url. Until the upstream's real path is
        # seeded (seed_catalog, biopb/biopb#297), fall back to the endpoint + the
        # upstream source_id -- grpc://lab:experiment1 (aliased) or
        # grpc://lab-store:8815:experiment1 (no alias) -- which is at least more
        # legible than the bare endpoint shared by every source of an upstream.
        self._source_url = f"grpc://{self._authority}:{upstream_source_id}"

        self._upstream_location = upstream_location
        self._upstream_source_id = upstream_source_id
        self._token = token
        # Per-source capability token for the LOCAL server's auth (server reads
        # adapter.capability_token in _authorize_source). Proxied sources inherit
        # server-wide auth, so leave it unset.
        self._capability_token: Optional[str] = None

        self._client = None  # lazy TensorFlightClient to the upstream
        # Best-effort reachability, updated by every catalog-surface call to the
        # upstream. Optimistic until proven otherwise so a never-yet-listed source
        # is not pre-emptively reported unresolved.
        self._reachable = True

        # Bulk-seeded catalog surface (biopb/biopb#266). When the reconcile fetches
        # the whole upstream catalog in one query_sources, it seeds these so
        # registration (sync_source_added -> get_source_descriptor/get_metadata)
        # needs no per-source upstream RPC. None = not seeded (fall back to a live
        # per-source fetch). See seed_catalog().
        self._descriptors_cache: Optional[List[TensorDescriptor]] = None
        self._metadata_cache: Optional[dict] = None
        # Whether the upstream *source* is resident (carried from the bulk row).
        # None until seeded; is_resident() = reachable AND this (when seeded), so
        # an unresolved upstream source mirrors as non-resident (biopb/biopb#266).
        self._upstream_resident: Optional[bool] = None

    # ------------------------------------------------------------------ upstream

    @property
    def client(self):
        """The upstream ``TensorFlightClient``, shared per endpoint (cache_bytes=0).

        Resolved from a process-wide pool keyed by ``(endpoint, token)`` so every
        mirrored source of one upstream reuses a single connection
        (biopb/biopb#266 B1). Still cached on ``self`` (the first access binds the
        pooled client) so ``self._client is None`` keeps meaning "this adapter has
        not dialed yet". Lazy: constructing/registering the adapter opens no
        socket; the connection opens on the first metadata/chunk call.
        """
        if self._client is None:
            self._client = _pooled_upstream_client(self._upstream_location, self._token)
        return self._client

    def _to_upstream_array_id(self, local_array_id: str) -> str:
        """Map a local array_id to the upstream's (swap the source_id prefix)."""
        field = local_array_id[len(self.source_id) :]  # "" or "/<field...>"
        return self._upstream_source_id + field

    def _to_local_array_id(self, upstream_array_id: str) -> str:
        """Map an upstream array_id back to the local one (inverse of above)."""
        field = upstream_array_id[len(self._upstream_source_id) :]
        return self.source_id + field

    def _localize_descriptor(self, desc: TensorDescriptor) -> TensorDescriptor:
        """Copy an upstream TensorDescriptor with its array_id rewritten local-ward.

        ``metadata_json`` and ``pyramid`` are cleared so the mirrored descriptor
        stays lean, exactly like a native adapter's: the LOCAL server fills both
        itself on a ``GetFlightInfo`` (metadata from ``get_metadata()``; the
        advertised pyramid from its own config). The upstream's ``get_descriptor``
        result carries them, so without clearing they would leak onto the proxy's
        catalog surface and the metadata would get double-wrapped on re-serialize.
        """
        out = TensorDescriptor()
        out.CopyFrom(desc)
        out.array_id = self._to_local_array_id(desc.array_id)
        out.metadata_json = ""
        out.ClearField("pyramid")
        return out

    def _localize_forwarded_descriptor(
        self, desc: TensorDescriptor
    ) -> TensorDescriptor:
        """Localize an upstream *GetFlightInfo* descriptor, keeping pyramid + scale.

        The serve-path counterpart to ``_localize_descriptor`` (which strips the
        pyramid so the LOCAL server advertises its own). A forwarded
        ``GetFlightInfo`` descriptor is **authoritative**: the upstream already
        computed the server-advertised pyramid (its native precompute levels for
        a pyramidal OME-Zarr) and the physical scale for this exact tensor, and
        the proxy mirrors it 1:1, so those are kept verbatim rather than
        recomputed against a locally-guessed grid. Only ``metadata_json`` is
        cleared -- the local server fills it from the mirror catalog row
        (biopb/biopb#253), not from this forwarded call.
        """
        out = TensorDescriptor()
        out.CopyFrom(desc)
        out.array_id = self._to_local_array_id(desc.array_id)
        out.metadata_json = ""
        return out

    def _mark_unreachable(self, exc: Exception) -> None:
        """Record an upstream connectivity failure (catalog-surface degradation)."""
        self._reachable = False
        # Drop this adapter's reference and evict the shared pooled client so the
        # next call (from any mirrored source of this endpoint) reconnects.
        self._client = None
        _evict_pooled_upstream_client(self._upstream_location, self._token)
        logger.warning(
            "upstream tensor server %s unreachable: %s", self._upstream_location, exc
        )

    # -------------------------------------------------------------- source layer

    @classmethod
    def create_from_config(
        cls,
        source: SourceConfig,
        credentials_config: Optional[Any] = None,
    ) -> RemoteTensorAdapter:
        """Build a proxy from a ``grpc://host:port/<upstream_source_id>`` source.

        The bare-host form (mirror every upstream source) is expanded into
        per-source entries upstream of this call (§3); here a single upstream
        source_id is required.
        """
        endpoint, upstream_source_id = _split_grpc_url(source.url)
        if upstream_source_id is None:
            raise ValueError(
                "tensor-server source url must name an upstream source as "
                f"grpc://host:port/<source_id>; got {source.url!r} (the bare-host "
                "'mirror everything' form is expanded during discovery)."
            )

        token = _resolve_upstream_token(source, credentials_config)
        return cls(
            source_id=source.source_id,
            upstream_location=endpoint,
            upstream_source_id=upstream_source_id,
            token=token,
            alias=source.alias,
        )

    def _display_source_url(self, upstream_source_url: Optional[str]) -> str:
        """Build the catalog ``source_url`` so a browser can tree a mirror by path.

        Embeds the upstream source's REAL location under the (aliased) endpoint --
        ``grpc://<authority>/<remote-path>`` -- so a client nests mirrored sources
        by their upstream filepath beneath an endpoint root, instead of collapsing
        every source of an upstream into a flat ``grpc:`` node (biopb/biopb#297).
        The upstream url is a normalized catalog url (e.g.
        ``file:///labs/x/img.tif`` or ``s3://bucket/key``); keep its authority +
        path, drop the scheme. Falls back to the endpoint + upstream source_id when
        no usable path is available (empty/opaque upstream url).
        """
        if upstream_source_url:
            parts = urlsplit(upstream_source_url)
            remote = (parts.netloc + parts.path).strip("/")
            if remote:
                return f"grpc://{self._authority}/{remote}"
        return f"grpc://{self._authority}:{self._upstream_source_id}"

    def seed_catalog(
        self,
        upstream_tensors: List[dict],
        metadata: Optional[dict],
        data_resident: bool = True,
        source_url: Optional[str] = None,
        indexed_at: object = None,
    ) -> bool:
        """(Re)populate the catalog surface from a bulk upstream ``query_sources``.

        Called by the reconcile (biopb/biopb#266) with this source's row from a
        single upstream catalog fetch, so ``sync_source_added``
        (``get_source_descriptor`` + ``get_metadata``) needs no per-source upstream
        RPC. ``upstream_tensors`` is the row's ``tensors`` STRUCT[] (upstream
        array_ids) as list-of-dicts; each is localized (source_id prefix swapped)
        exactly as the live path's ``_localize_descriptor`` would. Unlike the live
        ``list_tensor_descriptors`` (default field only), this seeds **all** of the
        source's tensors, so a multi-field upstream mirrors completely.

        ``data_resident`` is the upstream *source*'s residency (from its row): an
        unresolved upstream source (``data_resident=false``, empty tensors) must
        mirror as non-resident, not be advertised resident. Idempotent and
        re-appliable: the reconcile re-seeds every mirrored source each re-list,
        so an in-place upstream resolution (empty -> populated tensors,
        false -> true) refreshes here rather than going stale.

        ``source_url`` is the upstream source's own catalog url; it is folded into
        the mirror's display url so the browser can tree it by the remote path
        (biopb/biopb#297).

        ``indexed_at`` is the upstream source's register timestamp; it becomes this
        mirror's ``content_version`` (``b"iat:<ts>"``, biopb/biopb#178), folded into
        every minted proxy envelope so the chunk cache re-namespaces when the
        upstream re-registers the source. It is set unconditionally (every reconcile
        refreshes it) and deliberately NOT part of the ``changed`` result: a re-sync
        re-stamps the LOCAL ``indexed_at``, so gating re-sync on it would churn; the
        content_version only needs to ride the adapter for minting, not the catalog.

        We just queried the upstream, so mark it reachable. Returns whether the
        seeded catalog surface actually changed, so the caller can skip a
        redundant metadata-DB re-sync (and its ``indexed_at`` churn).
        """
        # The upstream register timestamp is this mirror's content_version. An
        # unversioned upstream (no indexed_at) leaves the proxy unversioned -> the
        # envelope carries an empty cv, exactly as before this plumbing.
        self._content_version = (
            b"iat:" + str(indexed_at).encode() if indexed_at is not None else None
        )
        descs: List[TensorDescriptor] = []
        for t in upstream_tensors or []:
            descs.append(
                TensorDescriptor(
                    array_id=self._to_local_array_id(t["array_id"]),
                    dim_labels=t.get("dim_labels") or [],
                    shape=t.get("shape") or [],
                    chunk_shape=t.get("chunk_shape") or [],
                    dtype=t.get("dtype") or "",
                )
            )
        new_metadata = metadata or {}
        new_resident = bool(data_resident)
        # Mirror the upstream's real path into the display url (biopb/biopb#297).
        new_url = self._display_source_url(source_url)
        changed = (
            descs != self._descriptors_cache
            or new_metadata != self._metadata_cache
            or new_resident != self._upstream_resident
            or new_url != self._source_url
        )
        self._descriptors_cache = descs
        self._metadata_cache = new_metadata
        self._reachable = True
        self._upstream_resident = new_resident
        self._source_url = new_url
        return changed

    def get_metadata(self) -> dict:
        """Mirror the upstream source's metadata dict (OME etc.), best-effort.

        ``list_flights`` / ``list_sources`` is deliberately lean and leaves
        ``metadata_json`` empty, and the only *live* RPC that fills it
        (``GetFlightInfo(with_metadata=True)``) returns it *wrapped* in a
        ``{"type","dim_label","metadata"}`` envelope. Instead read it from the
        upstream's metadata catalog with a server-side SQL query: the DuckDB
        ``sources.metadata_json`` column stores ``json.dumps(get_metadata())``
        verbatim -- the **raw** dict, no envelope -- which is exactly this
        method's contract (the LOCAL server adds the envelope when it serializes
        the response on a ``GetFlightInfo(with_metadata=True)``). Best-effort: an
        unreachable upstream, a metadata-DB-disabled upstream, or a not-yet-synced
        source all degrade to ``{}`` (metadata is non-critical for serving).
        """
        import json

        # Bulk-seeded at registration -> no upstream RPC (biopb/biopb#266).
        if self._metadata_cache is not None:
            return self._metadata_cache

        escaped = self._upstream_source_id.replace("'", "''")
        sql = f"SELECT metadata_json FROM sources WHERE source_id = '{escaped}'"
        try:
            rows = self.client.query_sources(sql, format="records")
        except Exception as exc:
            logger.debug(
                "upstream metadata query failed for %s: %s", self.source_id, exc
            )
            return {}
        if not rows:
            return {}
        raw = rows[0].get("metadata_json")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Mirror this one upstream source's tensor descriptor(s).

        Fetched per-source via ``get_descriptor`` (a targeted GetFlightInfo), NOT
        by scanning the upstream's whole ``list_sources()`` catalog: that call is
        *capped* (``max_list_flights_results``), so for a large upstream this
        source could be truncated out of it -- and it would re-fetch the entire
        catalog on every ListFlights, an O(N^2) cost. A single source's
        descriptor has no such cap.

        Catalog surface: an **unreachable** (or upstream-unresolved) source
        degrades to ``[]`` (an empty placeholder row) rather than raising, so the
        source stays catalogued while the upstream is down and reappears
        transparently once it is back. The serve surface still raises.

        Limitation: this returns the upstream source's *default* tensor. A
        multi-tensor (multi-field) upstream source is advertised by its default
        field only -- reading a specific other field still works (the chunk_id
        carries the full array_id), but the upstream exposes no cheap,
        non-truncatable way to enumerate a single source's full field list.
        (The bulk-seed path in ``seed_catalog`` does not have this limitation --
        it mirrors every field from the upstream catalog row, biopb/biopb#266.)
        """
        # Bulk-seeded at registration -> no upstream RPC (biopb/biopb#266).
        if self._descriptors_cache is not None:
            return self._descriptors_cache

        try:
            desc = self.client.get_descriptor(self._to_upstream_array_id(self.array_id))
        except Exception as exc:
            self._mark_unreachable(exc)
            return []  # unreachable / unresolved upstream -> placeholder row
        self._reachable = True
        return [self._localize_descriptor(desc)]

    def is_resident(self) -> bool:
        """Best-effort residency of the mirrored source.

        The base implementation would call a ``grpc://`` source non-resident (a
        remote scheme), wrongly tripping unresolved-source handling. Instead track
        reachability from the catalog-surface upstream calls: an unreachable
        upstream reports ``data_resident=False`` (paired with the empty
        placeholder tensor list above) until it recovers.

        When bulk-seeded (biopb/biopb#266), also require the upstream *source* to
        be resident -- so a mirror of an unresolved upstream source
        (``data_resident=false`` on the upstream) reports non-resident rather
        than being advertised resident just because the endpoint is reachable.
        """
        if self._upstream_resident is not None:
            return self._reachable and self._upstream_resident
        return self._reachable

    def get_tensor_adapter(self, tensor_id: Optional[str]):
        """Return a tensor-layer view bound to the requested within-source field."""
        field = self._within_source_field(tensor_id)
        if field == self._tensor_name:
            return self
        view = copy.copy(self)
        view._tensor_name = field
        return view

    def plan_flight_info(self, read_opt, pyramid_config):
        """Forward the upstream's authoritative GetFlightInfo, else plan locally.

        A caching proxy mirrors its upstream 1:1 and re-derives no chunk grid,
        pyramid, or physical scale of its own, so consult the upstream once
        (``forward_flight_info``) and use its localized plan verbatim -- the
        forwarded descriptor already carries the upstream's native grid, its
        server-advertised pyramid, and its physical scale (kept by
        ``_localize_forwarded_descriptor``; only ``metadata_json`` is stripped and
        refilled locally from the mirror catalog). On an upstream failure the
        forward returns ``None`` and we fall back to the base local planner --
        never worse than a non-proxy adapter (biopb/biopb#295). On that fallback
        the proxy advertises no physical scale of its own (inherited
        ``_physical_scale`` default ``None``, exactly as before; tracked by
        #266/#274).
        """
        plan = self.forward_flight_info(read_opt)
        if plan is not None:
            return plan
        return super().plan_flight_info(read_opt, pyramid_config)

    # -------------------------------------------------------------- tensor layer

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Mirror the upstream tensor descriptor under the local array_id.

        Served from the bulk-seeded cache when available (biopb/biopb#266 B2):
        ``seed_catalog`` already localized every tensor's structural fields
        (shape/chunk/dtype/dim_labels) from a single upstream ``query_sources``,
        so the serve-path ``GetFlightInfo`` reads the descriptor locally instead
        of a per-open ``get_descriptor`` RPC. This removes the last structural
        upstream call on the serve path; together with metadata served from the
        local catalog (#253 core) it leaves ``get_physical_scale`` as the only
        residual ``GetFlightInfo`` RPC (dropped separately, #266/#274) before
        ``do_get`` is the sole upstream contact. Falls back to a live fetch when
        this tensor was not seeded (a single-source static remote, or a field
        absent from the seed).
        """
        if self._descriptors_cache is not None:
            for desc in self._descriptors_cache:
                if desc.array_id == self.array_id:
                    out = TensorDescriptor()
                    out.CopyFrom(desc)
                    return out

        upstream_array_id = self._to_upstream_array_id(self.array_id)
        desc = self.client.get_descriptor(upstream_array_id)
        return self._localize_descriptor(desc)

    def forward_flight_info(
        self, read_opt: TensorReadOption
    ) -> Optional[TensorReadPlan]:
        """Forward a whole ``GetFlightInfo`` to the upstream and localize it.

        The server's ``get_flight_info`` calls this for a proxy tensor *instead*
        of running its local read planner (biopb/biopb#295). A caching proxy
        re-derives **no** chunk grid, pyramid, or physical scale of its own (see
        the module docstring); the default planner would guess a grid from the
        seed's ``chunk_shape`` (advisory, and often empty for the aicsimageio/
        OME-TIFF family), fall through to the 64 MB default grid, and
        over-amplify a single-plane read ~125x. Instead, consult the upstream
        once for its **authoritative** ``GetFlightInfo`` -- carrying the native
        grid, the server-advertised pyramid (a pyramidal OME-Zarr upstream's
        precompute levels included), the physical scale, and scaled ``chunk_id``s
        for a downsampled read -- and return it localized, with each
        ``chunk_id``'s ``array_id`` rewritten upstream->local. Nothing is
        re-implemented, and the rewritten ``chunk_id``s round-trip: a later
        ``do_get`` on one forwards straight back upstream via
        ``resolve_chunk_data`` (the same array_id swap).

        Returns ``None`` in either failure mode, so the caller falls back to the
        local planner -- never worse than a non-proxy adapter. The two modes are
        caught separately: a **transport** failure of the upstream RPC (unreachable
        / UNAVAILABLE / timeout / auth / upstream-side error) is an expected
        operational condition, logged at DEBUG; a **logic** failure localizing a
        response we *did* receive (a too-old / unexpected upstream, a corrupt
        payload, or a proxy bug) is unexpected and logged at WARNING, so the
        fallback never silently masks it.
        """
        # Transport step -- the upstream GetFlightInfo RPC. flight.FlightError
        # covers every upstream/gRPC failure (unavailable, timeout, auth, and an
        # upstream-side internal error); OSError covers a socket-level fault.
        try:
            info = self._upstream_flight_info(read_opt)
        except (flight.FlightError, OSError) as exc:
            logger.debug(
                "upstream flight-info RPC failed for %s (%r); falling back to the "
                "local read planner",
                self.array_id,
                exc,
            )
            return None

        # Logic step -- localize the response. A failure here is not a transport
        # problem, so surface it (WARNING) rather than letting the fallback hide a
        # protocol mismatch or a proxy bug.
        try:
            up_desc = TensorDescriptor.FromString(info.descriptor.command)
            endpoints = []
            for ep in info.endpoints:
                ticket = TensorTicket.FromString(ep.ticket.ticket)
                bounds = ChunkBounds.FromString(ep.app_metadata)
                # Wrap the upstream chunk_id in a proxy envelope: it is carried
                # VERBATIM (never rewritten) and forwarded byte-for-byte on do_get,
                # so bounds/scale/version bytes are untouched and the proxy stays
                # blind to the upstream codec. We read the chunk's OWN upstream
                # array_id (not self.array_id -- a sibling-field chunk keeps its own)
                # only to build the LOCAL route, so the server dispatches a later
                # do_get back to the right local tensor view. The upstream's
                # content_version rides the envelope so the proxy cache namespaces
                # by upstream content.
                upstream_aid, _ = decode_chunk_id(ticket.chunk_id)
                local_chunk_id = encode_proxy_envelope(
                    ticket.chunk_id,
                    self._to_local_array_id(upstream_aid),
                    self.content_version,
                )
                endpoints.append(ChunkEndpoint(chunk_id=local_chunk_id, bounds=bounds))
            return TensorReadPlan(
                descriptor=self._localize_forwarded_descriptor(up_desc),
                chunk_endpoints=endpoints,
            )
        except Exception as exc:
            logger.warning(
                "upstream flight-info response for %s could not be localized (%r); "
                "falling back to the local read planner",
                self.array_id,
                exc,
                exc_info=True,
            )
            return None

    def _upstream_flight_info(self, read_opt: TensorReadOption):
        """One ``GetFlightInfo`` to the upstream for this tensor, hints forwarded.

        Re-targets the incoming ``read_opt`` at the upstream array_id and forwards
        its slice/scale/reduction hints verbatim. ``with_metadata`` is forced
        False: the local server fills the response ``metadata_json`` itself from
        the mirror catalog (biopb/biopb#253), and ``pyramid``/``physical_scale``
        ride the upstream descriptor unconditionally (the upstream fills them at
        open time regardless of ``with_metadata``), so the forwarded call needs
        only the descriptor + endpoints.
        """
        upstream_array_id = self._to_upstream_array_id(self.array_id)
        up_read_opt = TensorReadOption(
            tensor_id=upstream_array_id,
            with_metadata=False,
        )
        if read_opt.HasField("slice_hint"):
            up_read_opt.slice_hint.CopyFrom(read_opt.slice_hint)
        if read_opt.scale_hint:
            up_read_opt.scale_hint[:] = list(read_opt.scale_hint)
        if read_opt.reduction_method:
            up_read_opt.reduction_method = read_opt.reduction_method
        # FlightCmd.source_id is the slash-free array_id prefix (identity policy);
        # tensor_id carries the full array_id, which the upstream reduces to the
        # within-source field -- so this works for a multi-tensor source too.
        cmd = FlightCmd(
            source_id=upstream_array_id.split("/", 1)[0],
            tensor_read=up_read_opt,
        )
        flight_desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        return self.client._client.get_flight_info(
            flight_desc, options=self.client._call_options
        )

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Fetch one region from the upstream (fallback / abstract-method satisfier).

        The hot path is ``resolve_chunk_data`` (it forwards the exact chunk_id so
        the upstream does any downsampling); this builds an upstream chunk_id for
        ``bounds`` and reads it back as a numpy array.
        """
        super().get_data(bounds)  # validate bounds against the mirrored shape
        upstream_chunk_id = encode_chunk_id(
            self._to_upstream_array_id(self.array_id), bounds
        )
        batch = self._upstream_record_batch(upstream_chunk_id)
        return unpack_chunk_array(batch)

    def resolve_chunk_data(self, chunk_id: bytes, cache_manager=None) -> pa.RecordBatch:
        """Serve a chunk by forwarding the envelope's inner chunk_id to the upstream.

        The served chunk_id is a proxy envelope; peel it and forward the opaque
        inner (the upstream chunk_id) VERBATIM -- no decode, no rewrite -- so the
        **upstream** does any downsampling for a scaled chunk_id and only the small
        result crosses the network. The result is cached under the envelope itself
        (its canonical key), so the segment cache and the localhost mmap fast path
        are inherited unchanged, namespaced by the upstream's content_version.
        """
        from biopb_tensor_server.cache import ArrowFileBackend

        if not is_proxy_envelope(chunk_id):
            # The proxy only mints envelope chunk_ids (biopb/biopb#178 W1); a
            # non-envelope id is a stale pre-upgrade ticket. Fail clearly so the
            # client re-opens the tensor to refresh its endpoints.
            raise flight.FlightServerError(
                f"proxy source {self.source_id} received a non-envelope chunk_id "
                f"(stale pre-upgrade ticket); re-open the tensor to refresh."
            )

        route, _cv, inner = peel_proxy_envelope(chunk_id)
        should_cache = cache_manager is not None and (
            is_scaled_chunk(inner)
            or isinstance(cache_manager.backend, ArrowFileBackend)
        )

        def compute_fn():
            # Forward the upstream chunk_id VERBATIM (the opaque inner); the upstream
            # does any downsampling and only the result crosses the network.
            batch = self._upstream_record_batch(inner)
            return batch, batch.nbytes

        if should_cache:
            # The envelope is itself the canonical cache key (route + content_version
            # + opaque inner); the inner is never parsed here.
            cache_key = cache_key_for_chunk_id(chunk_id)
            entry = cache_manager.get_or_acquire(cache_key, compute_fn)
            data = entry.data
            cache_manager.release(cache_key)
            return data
        data, _ = compute_fn()
        return data

    def _upstream_record_batch(self, upstream_chunk_id: bytes) -> pa.RecordBatch:
        """do_get one chunk from the upstream, as a single unified RecordBatch."""
        ticket = TensorTicket(chunk_id=upstream_chunk_id)
        reader = self.client._client.do_get(
            flight.Ticket(ticket.SerializeToString()),
            options=self.client._call_options,
        )
        table = reader.read_all()
        # The unified chunk schema (data/shape/dtype) is a single row; combine any
        # chunking so a single RecordBatch is returned (what do_get expects).
        return table.combine_chunks().to_batches()[0]


def _resolve_upstream_token(
    source: SourceConfig, credentials_config: Optional[Any]
) -> Optional[str]:
    """Resolve the upstream bearer token: credentials profile, then env var.

    The full per-upstream ``storage_type="biopb-tensor"`` credentials-profile wiring
    lands with the §3 expansion; for now honor a named profile's ``token`` when one
    is configured, else the single-upstream ``BIOPB_UPSTREAM_TENSOR_TOKEN`` env var.
    """
    profile_name = getattr(source, "credentials_profile", None)
    if credentials_config is not None and profile_name:
        try:
            profile = credentials_config.get_profile(profile_name)
        except Exception:
            profile = None
        if profile is not None and getattr(profile, "token", None):
            return profile.token
    return os.environ.get(_UPSTREAM_TOKEN_ENV) or None
