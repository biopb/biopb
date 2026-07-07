"""Writable-server upload path: source creation, chunk writing, progress state.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). Owns everything
behind ``DoPut`` when the server is writable:

- **Source creation** (``create_source``) -- the ``cache:`` / ``ome_zarr:``
  array_id prefixes, adapter construction, catalog sync, and OME-Zarr metadata
  scaffolding.
- **Chunk writing** (``write_chunk``) -- polymorphic ``put_chunk`` dispatch to
  the source adapter, translating adapter write errors into Flight errors at the
  server boundary.
- **Upload-progress state machine** -- a per-source ``_UploadState`` (typed,
  replacing the former stringly-keyed dict) counting uploaded chunks and
  reporting PENDING/READY, surfaced by the ``upload_status`` Flight action.

The manager registers created sources through the shared ``SourceRegistry`` and
never holds a back-reference to the server, so the collaborators stay acyclic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds, ChunkUpload

from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.chunk import encode_chunk_id
from biopb_tensor_server.errors import WriteNotSupportedError
from biopb_tensor_server.metadata_db import MetadataDatabase
from biopb_tensor_server.source_registry import SourceRegistry

logger = logging.getLogger(__name__)


class UploadStatus(str, Enum):
    """Wire-facing upload state (the string values are the ``upload_status`` API
    contract, parsed by the client SDK)."""

    PENDING = "PENDING"
    READY = "READY"
    UNKNOWN = "UNKNOWN"


def _expected_chunk_count(shape: List[int], chunk_shape: List[int]) -> int:
    count = 1
    for dim, chunk in zip(shape, chunk_shape):
        count *= ceil(dim / chunk)
    return count


@dataclass
class _UploadState:
    """Per-source upload progress. ``READY`` once every expected chunk arrives."""

    source_id: str
    expected_chunks: int
    status: UploadStatus = UploadStatus.PENDING
    uploaded_chunk_ids: Set[bytes] = field(default_factory=set)

    @property
    def uploaded_chunks(self) -> int:
        return len(self.uploaded_chunk_ids)

    def as_status_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "state": self.status.value,
            "expected_chunks": self.expected_chunks,
            "uploaded_chunks": self.uploaded_chunks,
        }


class UploadManager:
    """Owns source creation, chunk writes, and per-source upload progress."""

    def __init__(
        self,
        registry: SourceRegistry,
        write_dir: Optional[Path],
        metadata_db: Optional[MetadataDatabase],
    ) -> None:
        self._registry = registry
        self._write_dir = write_dir
        self._metadata_db = metadata_db
        self._lock = threading.RLock()
        self._states: Dict[str, _UploadState] = {}

    # -- progress state machine ------------------------------------------------

    def initialize(
        self,
        source_id: str,
        shape: List[int] | Tuple[int, ...],
        chunk_shape: List[int] | Tuple[int, ...],
    ) -> None:
        """Begin tracking upload progress for a source.

        Called by ``create_source`` on the DoPut wire path, and directly by
        in-process cache producers (e.g. the image-runtime EmbeddedTensorCache)
        that register a cache source and write its chunks without going over the
        wire. Public so those in-process callers reach it through
        ``server.uploads`` rather than a removed server method.
        """
        expected = _expected_chunk_count(list(shape), list(chunk_shape))
        with self._lock:
            self._states[source_id] = _UploadState(
                source_id=source_id, expected_chunks=expected
            )

    def mark_chunk(self, source_id: str, bounds: ChunkBounds) -> None:
        """Record that the chunk at *bounds* arrived (flips to READY when full).

        Public for the same in-process producers as :meth:`initialize`; the
        wire path reaches it via :meth:`write_chunk`.
        """
        chunk_id = encode_chunk_id(source_id, bounds)
        with self._lock:
            state = self._states.get(source_id)
            if state is None:
                return
            state.uploaded_chunk_ids.add(chunk_id)
            state.status = (
                UploadStatus.READY
                if state.uploaded_chunks >= state.expected_chunks
                else UploadStatus.PENDING
            )

    def status(self, source_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._states.get(source_id)
            if state is None:
                return {
                    "source_id": source_id,
                    "state": UploadStatus.UNKNOWN.value,
                    "expected_chunks": 0,
                    "uploaded_chunks": 0,
                }
            return state.as_status_dict()

    def forget(self, source_id: str) -> None:
        """Drop a source's upload state (called when the source is unregistered)."""
        with self._lock:
            self._states.pop(source_id, None)

    # -- write path ------------------------------------------------------------

    def create_source(self, req_desc: TensorDescriptor) -> TensorDescriptor:
        """Create a source from a TensorDescriptor, return its resolved descriptor.

        array_id format in request:
        - "cache:name" → cache-backed with given name
        - "cache:" → cache-backed with server-generated name
        - "ome_zarr:name" → zarr-backed with given name
        - "ome_zarr:" → zarr-backed with server-generated name
        """
        array_id = req_desc.array_id

        if array_id.startswith("cache:"):
            # Cache-backed source
            provided_name = array_id[6:]  # After 'cache:'
            if provided_name:
                source_id = (
                    f"cache_{hashlib.sha256(provided_name.encode()).hexdigest()[:12]}"
                )
            else:
                source_id = f"cache_{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"

            ome_metadata = (
                json.loads(req_desc.metadata_json) if req_desc.metadata_json else {}
            )

            adapter = CachedSourceAdapter(
                source_id=source_id,
                shape=list(req_desc.shape),
                dtype=req_desc.dtype,
                chunk_shape=list(req_desc.chunk_shape),
                dim_labels=list(req_desc.dim_labels) if req_desc.dim_labels else None,
                ome_metadata=ome_metadata,
            )
            self._registry.register(source_id, adapter)
            self.initialize(source_id, req_desc.shape, req_desc.chunk_shape)

            logger.info(f"Created cache-backed source: {source_id}")

        elif array_id.startswith("ome_zarr:"):
            # Zarr-backed source
            import zarr

            provided_name = array_id[9:]  # After 'ome_zarr:'
            zarr_name = (
                provided_name
                or f"upload_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}"
            )

            if self._write_dir is None:
                raise flight.FlightServerError(
                    "write_dir not configured for zarr-backed sources"
                )

            zarr_path = self._write_dir / f"{zarr_name}.zarr"
            zarr_path.mkdir(parents=True, exist_ok=True)

            store = zarr.DirectoryStore(str(zarr_path))
            arr = zarr.create(
                store=store,
                shape=req_desc.shape,
                dtype=req_desc.dtype,
                chunks=req_desc.chunk_shape,
            )

            # Write OME metadata
            if req_desc.metadata_json:
                zattrs = json.loads(req_desc.metadata_json)
            else:
                zattrs = self._build_minimal_ome_metadata(req_desc)

            with open(zarr_path / ".zattrs", "w") as f:
                json.dump(zattrs, f)

            source_id = f"ome_zarr_{hashlib.sha256(str(zarr_path.resolve()).encode()).hexdigest()[:12]}"

            adapter = OmeZarrAdapter(
                arr,
                source_id,
                list(req_desc.dim_labels) if req_desc.dim_labels else None,
            )

            self._registry.register(source_id, adapter)
            # File-backed uploads are durable (a real .zarr on disk), so add them
            # to the catalog to be discoverable via list_sources/query_sources.
            # Cache-backed uploads (the `cache:` branch above) are intentionally
            # NOT synced: they are volatile and have no removal hook, so a row
            # would dangle after eviction -- they stay readable by their returned
            # id but are not enumerable (biopb/biopb#265). Best-effort: a catalog
            # write must not fail the upload (the source is already usable by id).
            if self._metadata_db is not None:
                try:
                    self._metadata_db.sync_source_added(source_id, adapter)
                except Exception as e:
                    logger.warning(
                        f"Failed to sync uploaded source {source_id} to catalog "
                        f"(readable by id, not listed): {e}"
                    )
            self.initialize(source_id, req_desc.shape, req_desc.chunk_shape)

            logger.info(f"Created zarr-backed source: {source_id} at {zarr_path}")

        else:
            raise flight.FlightServerError(
                f"Invalid array_id format: {array_id}. Use 'cache:' or 'ome_zarr:' prefix"
            )

        return TensorDescriptor(
            array_id=source_id,
            dim_labels=req_desc.dim_labels,
            shape=req_desc.shape,
            chunk_shape=req_desc.chunk_shape,
            dtype=req_desc.dtype,
        )

    def write_chunk(
        self, upload: ChunkUpload, reader: flight.MetadataRecordBatchReader
    ) -> None:
        """Write chunk data by delegating to the source adapter's ``put_chunk``.

        Each source format owns its write contract: OmeZarr/Zarr enforce
        chunk-grid alignment; cache-backed sources accept arbitrary bounds;
        read-only formats reject the write. The handler no longer sniffs adapter
        attributes to pick a path.
        """
        table = reader.read_all()
        data_column = table.column(0)

        adapter = self._registry.get(upload.source_id)
        if adapter is None:
            raise flight.FlightServerError(f"Source not found: {upload.source_id}")

        bounds = upload.bounds
        expected_shape = tuple(
            stop - start for start, stop in zip(bounds.start, bounds.stop)
        )

        # Dispatch the write polymorphically: each source format owns its write
        # contract (Zarr/OmeZarr enforce chunk-grid alignment; a cache source
        # takes arbitrary bounds; read-only formats raise WriteNotSupportedError).
        # Adapters stay transport-agnostic, so translate their write errors into a
        # Flight error at this server boundary (alignment/size -> ValueError).
        dtype = table.schema.field(0).type.to_pandas_dtype()
        try:
            adapter.put_chunk(bounds, data_column, expected_shape, dtype)
        except (ValueError, WriteNotSupportedError) as e:
            raise flight.FlightServerError(str(e))

        self.mark_chunk(upload.source_id, bounds)

        logger.debug(
            f"Uploaded chunk to {upload.source_id}: bounds={list(bounds.start)}-{list(bounds.stop)}"
        )

    @staticmethod
    def _build_minimal_ome_metadata(desc: TensorDescriptor) -> dict:
        """Build minimal OME-Zarr metadata from a TensorDescriptor."""
        dim_labels = (
            list(desc.dim_labels)
            if desc.dim_labels
            else [f"dim{i}" for i in range(len(desc.shape))]
        )

        axes = []
        for label in dim_labels:
            if label.lower() in ("x", "y", "z"):
                axes.append({"name": label, "type": "space"})
            elif label.lower() in ("c", "channel"):
                axes.append({"name": label, "type": "channel"})
            elif label.lower() in ("t", "time"):
                axes.append({"name": label, "type": "time"})
            else:
                axes.append({"name": label})

        return {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0] * len(desc.shape)}
                            ],
                        }
                    ],
                }
            ]
        }
