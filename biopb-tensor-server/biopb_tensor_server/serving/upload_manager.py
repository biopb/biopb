"""Writable-server upload path: the DoPut boundary.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). What is left
here is only what a boundary does:

- **Kind selection** -- the ``cache:`` / ``ome_zarr:`` ``array_id`` prefix names
  the adapter class (``UPLOAD_KINDS``); the class builds its own upload
  (``create_upload``) and the manager registers it, syncing the catalog when
  the kind is durable.
- **Error translation** -- adapters stay transport-agnostic and raise typed
  errors; this is where they become Flight errors.
- **Lookup** -- ``status`` / ``discard`` / ``write_chunk`` find the adapter
  and hand over.

Progress, completion and disposal are the adapter's own
(:class:`~biopb_tensor_server.adapters._writable.WritableSource`), so there is no
second registry to keep in step with ``SourceRegistry``: an upload's state is
created with its adapter, lives as long as it is registered, and a discarded
one stays registered as a tombstone until reclaimed.

The manager registers created sources through the shared ``SourceRegistry`` and
never holds a back-reference to the server, so the collaborators stay acyclic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkUpload

from biopb_tensor_server.adapters._writable import (
    UploadStatus,
    WritableSource,
    unknown_upload_status,
    upload_of,
)
from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.core.axes import noncanonical_order
from biopb_tensor_server.core.errors import UploadDiscardedError, WriteNotSupportedError
from biopb_tensor_server.core.source_registry import SourceRegistry, close_adapter
from biopb_tensor_server.serving.metadata_db import MetadataDatabase

__all__ = ["UPLOAD_KINDS", "UploadManager", "UploadStatus"]

logger = logging.getLogger(__name__)

#: ``array_id`` prefix -> the adapter class that builds an upload of that kind.
#: The prefixes are a wire contract, so the table is closed; what each kind
#: does with a request is the class's own (``create_upload``).
UPLOAD_KINDS: Dict[str, Type[WritableSource]] = {
    "cache": CachedSourceAdapter,
    "ome_zarr": OmeZarrAdapter,
}


class UploadManager:
    """The DoPut boundary: picks the kind, registers, translates errors."""

    def __init__(
        self,
        registry: SourceRegistry,
        write_dir: Optional[Path],
        metadata_db: Optional[MetadataDatabase],
    ) -> None:
        self._registry = registry
        self._write_dir = write_dir
        self._metadata_db = metadata_db

    # -- lookup ----------------------------------------------------------------

    def status(self, source_id: str) -> Dict[str, Any]:
        """The ``upload_status`` answer: UNKNOWN for anything not tracking an upload."""
        upload = upload_of(self._registry.get(source_id))
        if upload is None:
            return unknown_upload_status(source_id)
        return upload.as_status_dict(source_id)

    def discard(self, source_id: str, reason: str = "") -> Dict[str, Any]:
        """Give up on an upload; see ``WritableSource.discard``.

        Total: a source that is not tracking an upload -- never was, or has
        since been reclaimed -- reads UNKNOWN rather than raising, so a retry
        after the tombstone is gone is not an error. A kind that refuses raises
        ``ValueError``; callers today are in-process, so it is not translated.
        """
        adapter = self._registry.get(source_id)
        if upload_of(adapter) is None:
            return unknown_upload_status(source_id)
        return adapter.discard(reason)

    # -- write path ------------------------------------------------------------

    @staticmethod
    def _require_canonical_axes(req_desc: TensorDescriptor) -> None:
        """Reject an upload whose declared axis order is not canonical (#596).

        A writable source is the one place the axis order is *declared* rather
        than read out of a file, and both ends of it belong to the same client:
        ``physical_scale`` and ``chunk_shape`` come in aligned to these labels,
        and ``put_chunk`` writes in this order. So the server refuses the order
        up front instead of permuting reads behind the uploader's back, which
        would desynchronize what it reads back from what it wrote. Transposing
        before upload is the client-side fix, and it is a cheap one.

        This keeps the canonical-order guarantee unconditional -- it holds for
        uploaded sources too -- at zero cost on the write data path. The remote
        proxy refuses for the same reason, in the same words
        (``core.axes.noncanonical_order``): there too the order belongs to
        someone who has aligned the rest of their state to it.
        """
        why = noncanonical_order(req_desc.dim_labels, req_desc.shape)
        if why is None:
            return
        raise flight.FlightServerError(
            f"create_source: {why}. The data plane advertises canonical order on "
            f"every source (biopb/biopb#596); transpose the array before "
            f"uploading."
        )

    def create_source(self, req_desc: TensorDescriptor) -> TensorDescriptor:
        """Create a source from a TensorDescriptor, return its resolved descriptor.

        array_id format in request:
        - "cache:name" → cache-backed with given name
        - "cache:" → cache-backed with server-generated name
        - "ome_zarr:name" → zarr-backed with given name
        - "ome_zarr:" → zarr-backed with server-generated name
        """
        self._require_canonical_axes(req_desc)

        prefix, sep, name = req_desc.array_id.partition(":")
        kind = UPLOAD_KINDS.get(prefix) if sep else None
        if kind is None:
            raise flight.FlightServerError(
                f"Invalid array_id format: {req_desc.array_id}. Use 'cache:' or "
                f"'ome_zarr:' prefix"
            )

        # Parsed at the boundary: a malformed payload is the request's fault and
        # must fail before the kind touches anything (biopb/biopb#354).
        metadata = (
            self._parse_metadata_json(req_desc.metadata_json)
            if req_desc.metadata_json
            else None
        )
        try:
            adapter = kind.create_upload(
                name, req_desc, metadata=metadata, write_dir=self._write_dir
            )
        except ValueError as e:
            raise flight.FlightServerError(str(e)) from e

        # A deterministic id (a named cache: upload) lands on whatever holds the
        # name now -- a prior upload, or its tombstone. Replace rather than
        # overwrite so the displaced adapter is released, not leaked.
        source_id = adapter.source_id
        registered, displaced = self._registry.swap(source_id, adapter)
        close_adapter(displaced)

        # Only a durable upload belongs in the catalog; a volatile one is
        # readable by its returned id but not enumerable (biopb/biopb#265).
        # Best-effort: a catalog write must not fail the upload.
        if kind.durable and self._metadata_db is not None:
            try:
                self._metadata_db.sync_source_added(source_id, registered)
            except Exception as e:
                logger.warning(
                    f"Failed to sync uploaded source {source_id} to catalog "
                    f"(readable by id, not listed): {e}"
                )

        logger.info(f"Created {prefix} upload: {source_id}")
        return adapter.upload_response(req_desc)

    def write_chunk(
        self, upload: ChunkUpload, reader: flight.MetadataRecordBatchReader
    ) -> None:
        """Hand one uploaded chunk to its source's ``put_chunk``.

        Each source format owns its write contract: OmeZarr/Zarr enforce
        chunk-grid alignment; cache-backed sources accept arbitrary bounds;
        read-only formats reject the write; a discarded upload refuses with its
        reason. Adapters stay transport-agnostic, so their errors become Flight
        errors here -- a discard as ``FlightCancelledError``, so a client
        discriminates on the type rather than the message (biopb/biopb#1).
        """
        table = reader.read_all()
        data_column = table.column(0)

        adapter = self._registry.get(upload.source_id)
        if adapter is None:
            raise flight.FlightServerError(f"Source not found: {upload.source_id}")

        bounds = upload.bounds
        expected_shape = tuple(
            stop - start for start, stop in zip(bounds.start, bounds.stop, strict=True)
        )
        dtype = table.schema.field(0).type.to_pandas_dtype()
        try:
            adapter.put_chunk(bounds, data_column, expected_shape, dtype)
        except UploadDiscardedError as e:
            raise flight.FlightCancelledError(str(e)) from e
        except (ValueError, WriteNotSupportedError) as e:
            raise flight.FlightServerError(str(e)) from e

        logger.debug(
            f"Uploaded chunk to {upload.source_id}: bounds={list(bounds.start)}-{list(bounds.stop)}"
        )

    @staticmethod
    def _parse_metadata_json(metadata_json: str) -> dict:
        """Parse the request's ``metadata_json`` into an OME-metadata dict,
        translating a malformed payload into a legible Flight error at the create
        boundary.

        A bare ``json.loads`` would raise ``JSONDecodeError``: on the DoPut path
        it is swallowed by the command-discrimination try (mis-surfaced as
        "Invalid upload command"), and on the ``create_source`` Flight action it
        escapes as a generic internal error. Either way the client gets no
        actionable signal, so map it to ``FlightServerError`` here (biopb/biopb#354).

        Well-formed JSON that isn't an object (e.g. ``"123"`` or ``"[...]"``) is
        rejected too: callers spread the result into adapter metadata / a
        ``.zattrs``, both of which require a mapping, so a non-dict must fail here
        rather than surface as a confusing error downstream.
        """
        try:
            parsed = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            raise flight.FlightServerError(f"invalid metadata_json: {e}") from e
        if not isinstance(parsed, dict):
            raise flight.FlightServerError(
                f"invalid metadata_json: expected a JSON object, got {type(parsed).__name__}"
            )
        return parsed
