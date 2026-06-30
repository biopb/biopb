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
no pixels and re-derives no chunk grid; the only thing it understands beyond
pass-through is the **array_id rewrite** that maps its local (possibly
alias-namespaced) identifiers to the upstream's and back -- a pure byte splice on
the chunk_id's length-prefixed array_id field (``chunk.rewrite_chunk_id_array_id``),
so bounds/scale bytes are never touched.

Scope of this slice (§2 of ``docs/remote-tensor-cache.md``): the adapter + its
data path, constructible directly (and via ``create_from_config`` for the
single-source ``grpc://host:port/<upstream_source_id>`` url form). Catalog
expansion of a bare ``grpc://host:port`` into one source per upstream tensor,
alias namespacing of the registered ``source_id``, the collision check, and the
monitor->re-list refresh are the next slice (§3).
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any, List, Optional
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.chunk import (
    decode_chunk_id,
    encode_chunk_id,
    is_scaled_chunk,
    rewrite_chunk_id_array_id,
)

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig

# Env-var convenience fallback for the single-upstream case (a per-upstream
# credentials profile -- storage_type="biopb-tensor" -- is the multi-upstream
# path, wired in the §3 expansion slice).
_UPSTREAM_TOKEN_ENV = "BIOPB_UPSTREAM_TENSOR_TOKEN"


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


class RemoteTensorAdapter(SourceAdapter, TensorAdapter):
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
        """
        self.source_id = source_id
        self._source_url = upstream_location
        self._source_type = "tensor-server"
        self._tensor_name = tensor_name

        self._upstream_location = upstream_location
        self._upstream_source_id = upstream_source_id
        self._token = token
        # Per-source capability token for the LOCAL server's auth (server reads
        # adapter.token in _authorize_source). Proxied sources inherit server-wide
        # auth, so leave it unset.
        self.token: Optional[str] = None

        self._client = None  # lazy TensorFlightClient to the upstream

    # ------------------------------------------------------------------ upstream

    @property
    def client(self):
        """Lazily-built ``TensorFlightClient`` to the upstream (cache_bytes=0).

        Built lazily so constructing the adapter (e.g. at registration) does not
        open a socket; the connection opens on the first metadata/chunk call.
        """
        if self._client is None:
            from biopb.tensor import TensorFlightClient

            self._client = TensorFlightClient(
                self._upstream_location, cache_bytes=0, token=self._token
            )
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
        """Copy an upstream TensorDescriptor with its array_id rewritten local-ward."""
        out = TensorDescriptor()
        out.CopyFrom(desc)
        out.array_id = self._to_local_array_id(desc.array_id)
        return out

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
        )

    def get_metadata(self) -> dict:
        """Mirror the upstream source's metadata dict (OME etc.), best-effort."""
        import json

        try:
            descriptors = self.client.list_sources()
        except Exception:
            return {}
        src = descriptors.get(self._upstream_source_id)
        if src is None or not src.metadata_json:
            return {}
        try:
            return json.loads(src.metadata_json)
        except (json.JSONDecodeError, ValueError):
            return {}

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Mirror the upstream source's tensor descriptors (ids rewritten local)."""
        descriptors = self.client.list_sources()
        src = descriptors.get(self._upstream_source_id)
        if src is None:
            # Fall back to the default tensor via the per-tensor probe.
            return [self.get_tensor_descriptor()]
        return [self._localize_descriptor(t) for t in src.tensors]

    def is_resident(self) -> bool:
        """A reachable upstream source is treated as resident.

        The base implementation would call it non-resident (a ``grpc://`` url is a
        remote scheme), which would wrongly steer the server's serve path toward
        unresolved-source handling. An *unreachable* upstream is a separate case
        handled by the UnresolvedSourceAdapter deferral in the §3 slice.
        """
        return True

    def get_tensor_adapter(self, tensor_id: Optional[str]):
        """Return a tensor-layer view bound to the requested within-source field."""
        field = self._within_source_field(tensor_id)
        if field == self._tensor_name:
            return self
        view = copy.copy(self)
        view._tensor_name = field
        return view

    # -------------------------------------------------------------- tensor layer

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Mirror the upstream tensor descriptor under the local array_id."""
        upstream_array_id = self._to_upstream_array_id(self.array_id)
        desc = self.client.get_descriptor(upstream_array_id)
        return self._localize_descriptor(desc)

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
        arr = batch.column("data").to_numpy(zero_copy_only=False)[0]
        shape = tuple(batch.column("shape").to_pylist()[0])
        return arr.reshape(shape)

    def resolve_chunk_data(self, chunk_id: bytes, cache_manager=None) -> pa.RecordBatch:
        """Serve a chunk by forwarding the (rewritten) chunk_id to the upstream.

        Forwarding the chunk_id verbatim (only the array_id field swapped) means
        the **upstream** does any downsampling for a scaled chunk_id and only the
        small result crosses the network. The returned Arrow RecordBatch is cached
        under the LOCAL chunk_id, so the segment cache and the localhost mmap fast
        path are inherited unchanged.
        """
        from biopb_tensor_server.cache import ArrowFileBackend

        local_array_id, _ = decode_chunk_id(chunk_id)
        should_cache = cache_manager is not None and (
            is_scaled_chunk(chunk_id)
            or isinstance(cache_manager.backend, ArrowFileBackend)
        )

        def compute_fn():
            upstream_chunk_id = rewrite_chunk_id_array_id(
                chunk_id, self._to_upstream_array_id(local_array_id)
            )
            batch = self._upstream_record_batch(upstream_chunk_id)
            return batch, batch.nbytes

        if should_cache:
            entry = cache_manager.get_or_acquire(
                chunk_id, compute_fn, metadata={"array_id": local_array_id}
            )
            data = entry.data
            cache_manager.release(chunk_id)
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
