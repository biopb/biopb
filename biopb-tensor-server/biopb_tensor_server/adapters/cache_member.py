"""The ``cache://`` store format: a member whose chunks are kept as uploaded.

``docs/upload-model.md`` step 7. A ``zarr://`` member decodes on every read; this
one does not, because what was uploaded is what is served. Its directory is

    <write_dir>/sources/<name>.zarr/<field>/
        descriptor.json     shape, dtype, grid, axes, the token, the marker
        seg_NNNN.arrow      one Arrow batch per chunk, exactly as uploaded
        seg_NNNN.idx        that segment's index, written when READY seals it

The bytes are the tensor and not a cache of it: the segments sit under the
member, outside ``max_total_bytes``, outside the eviction sweep and outside the
retention classes. What is shared with the chunk cache is the format and the
boot index (:mod:`~biopb_tensor_server.cache.segment_store`), which is why a
``cache://`` member now survives the process that uploaded it -- before this it
was the old volatile ``cache:`` source addressed as a tensor, and its chunks
died with the cache entry they were in.

**The index keys by bounds, never by chunk id.** A chunk id carries the
server's serving-semantics epoch, which moves on an upgrade that changes what
the bytes mean; the bytes on disk do not. The chunk ids this build mints are
therefore re-derived at registration, into the record
(``CachedSourceAdapter._written_chunks``) the read path already consults.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.adapters.members import (
    MEMBER_DESCRIPTOR,
    member_attrs,
    read_member_version,
)
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, UPLOAD_READY
from biopb_tensor_server.cache.segment_store import SegmentStore
from biopb_tensor_server.cache.types import ChunkLocation
from biopb_tensor_server.core.chunk import decode_chunk_id, is_scaled_chunk

__all__ = ["CacheMember", "create_cache_member", "open_cache_member"]

logger = logging.getLogger(__name__)


def bounds_key(bounds: ChunkBounds) -> bytes:
    """The store key of the chunk at *bounds*: its extent, as text.

    Bounds and not a chunk id (see the module docstring), and the whole extent
    rather than its start alone -- this format stores the grid it was uploaded
    on, and a key that named only a corner could not tell two grids apart if a
    later life ever served both.
    """
    start = ",".join(str(int(v)) for v in bounds.start)
    stop = ",".join(str(int(v)) for v in bounds.stop)
    return f"{start}-{stop}".encode()


def _parse_bounds_key(key: bytes) -> Optional[ChunkBounds]:
    """The bounds *key* names, or None if it is not one this module wrote."""
    try:
        start, stop = key.decode().split("-")
        return ChunkBounds(
            start=[int(v) for v in start.split(",")],
            stop=[int(v) for v in stop.split(",")],
        )
    except (UnicodeDecodeError, ValueError):
        return None


class CacheMember(CachedSourceAdapter):
    """An uploaded tensor served straight out of its own segments.

    :class:`~biopb_tensor_server.adapters.cached_source.CachedSourceAdapter`
    with the bytes somewhere durable: the record, the gap-versus-loss rule and
    the region assembly are all inherited, and what this overrides is only
    where a chunk is put and where it is read back from. So a ``cache://``
    member and the embedded result cache behave identically on every read; they
    differ in whether the bytes outlive the process.
    """

    #: A store on disk under ``write_dir``, catalogued through its source: the
    #: boundary keeps the row in step and discard removes the directory.
    durable = True

    def __init__(
        self,
        source_id: str,
        field: str,
        store: Path,
        chunks: SegmentStore,
        *,
        shape: List[int],
        dtype: str,
        chunk_shape: List[int],
        dim_labels: Optional[List[str]] = None,
        physical_scale: Optional[List[float]] = None,
        physical_unit: Optional[List[str]] = None,
        content_version: bytes,
        track_upload: bool = True,
    ) -> None:
        super().__init__(
            source_id=source_id,
            shape=shape,
            dtype=dtype,
            chunk_shape=chunk_shape,
            dim_labels=dim_labels,
            physical_scale=physical_scale,
            physical_unit=physical_unit,
            content_version=content_version,
            track_upload=track_upload,
        )
        self._tensor_name = field
        self.store = Path(store)
        self._chunks = chunks
        self._source_url = f"cache://{self.array_id}"

    # -- where the bytes are ---------------------------------------------------

    def _store_chunk_batch(
        self,
        chunk_id: bytes,
        bounds: ChunkBounds,
        batch: pa.RecordBatch,
        size_bytes: int,
    ) -> None:
        """Append the chunk to this member's segments.

        Put-once: a bounds already stored is refused rather than appended
        again, because a second copy would leave the first as bytes no index
        names and a reader may already hold a range into it.
        """
        try:
            self._chunks.append(bounds_key(bounds), batch)
        except KeyError:
            raise ValueError(
                f"{self.array_id} already holds the chunk at "
                f"{list(bounds.start)}..{list(bounds.stop)}. An uploaded chunk "
                f"is written once; discard the tensor to replace it."
            ) from None
        except RuntimeError:
            # The store closed under us: a discard landed between this write
            # passing the upload's gate and reaching the disk.
            raise ValueError(
                f"{self.array_id} was discarded while this chunk was in flight."
            ) from None

    def _read_chunk_batch(self, chunk_id: bytes, bounds: ChunkBounds) -> pa.RecordBatch:
        """The chunk's stored batch, served as it was uploaded -- no decode."""
        batch = self._chunks.read(bounds_key(bounds))
        if batch is None:
            raise _lost_chunk(self.array_id, bounds)
        return batch

    def locate_chunk(self, chunk_id: bytes) -> Optional[ChunkLocation]:
        """This chunk's byte range in its sealed segment (issue #9).

        The localhost fast path without the copy the default takes: the default
        resolves a chunk into the chunk cache and answers *its* byte range,
        while here the chunk is already stored as the batch the client wants.
        Only an **unscaled** ticket at a stored bounds is claimed -- a scaled
        one has no stored batch and falls through to be resolved and cached
        like any other read.
        """
        if is_scaled_chunk(chunk_id) or chunk_id not in self._written_chunks:
            return None
        _, bounds = decode_chunk_id(chunk_id)
        return self._chunks.locate(bounds_key(bounds))

    # -- the store's lifecycle -------------------------------------------------

    def _publish_store(self) -> None:
        """Seal the segments, then mark the store ready; what READY does.

        Sealing first is what makes the member servable by byte range -- the
        open segment gets its ``.idx`` sidecar here -- and the marker is what a
        restart reads, so it must not be written over a store that is not yet
        sealed. Raises ``OSError`` if the marker cannot be written, leaving the
        upload PENDING for the transition to retry.
        """
        self._chunks.seal()
        self._write_descriptor(UPLOAD_READY)

    def _dispose_store(self) -> None:
        """Release the handles and remove the directory, on discard.

        The descriptor goes first, and separately: a removal that cannot finish
        -- an open mapping blocks unlink on Windows -- must not leave behind a
        directory the next boot reads as a published member. Without its
        descriptor it is a directory no format claims, which is skipped.
        """
        self._chunks.close()
        try:
            (self.store / MEMBER_DESCRIPTOR).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"{self.array_id}: could not drop the descriptor: {e}")
        shutil.rmtree(self.store, ignore_errors=True)
        logger.info(f"Removed the store of discarded upload {self.array_id}")

    def delete_store(self) -> None:
        """Remove the store of a member adopted from an earlier life.

        The counterpart of ``registered.ZarrMember.delete_store``: an adopted
        member holds no upload record to seal, so discarding it is store
        removal alone.
        """
        self._dispose_store()

    def close(self) -> None:
        """Drop the segment handles; an open one blocks unlink on Windows."""
        self._chunks.close()
        super().close()

    def _write_descriptor(self, state: str) -> None:
        """Rewrite ``descriptor.json`` with the upload marker set to *state*.

        Atomic (write-then-replace), so a crash mid-write cannot leave a member
        with no descriptor at all -- which the next boot would read as a
        directory it did not mint, and sweep.
        """
        path = self.store / MEMBER_DESCRIPTOR
        tmp = path.with_name(MEMBER_DESCRIPTOR + ".tmp")
        tmp.write_text(json.dumps(self._descriptor_json(state)))
        os.replace(tmp, path)

    def _descriptor_json(self, state: str) -> Dict[str, Any]:
        """What ``descriptor.json`` holds: everything needed to reopen this.

        The ``.zattrs`` of the other format, for a directory that is not a zarr
        group -- so the boot sweep and the adoption pass read one marker shape
        whichever format they are looking at.
        """
        return {
            "shape": list(self._shape),
            "dtype": self._dtype,
            "chunk_shape": list(self._chunk_shape),
            "dim_labels": list(self._dim_labels),
            "physical_scale": list(self._physical_scale_vec),
            "physical_unit": list(self._physical_unit_vec),
            **member_attrs(self.content_version, state),
        }


def _lost_chunk(array_id: str, bounds: ChunkBounds) -> Exception:
    """The read of a chunk the index holds and the segments cannot produce.

    A torn segment tail is the one way in: the entry was indexed by a walk that
    found its head and not its body. Never a gap -- a gap is a bounds the index
    does not hold at all, and that reads as zeros.
    """
    return flight.FlightServerError(
        f"{array_id} has lost the chunk at {list(bounds.start)}.."
        f"{list(bounds.stop)}: it is indexed, and its segment cannot produce "
        f"it. The segment was probably truncated by a crash mid-write."
    )


def create_cache_member(
    store: Path, source_id: str, field: str, desc: TensorDescriptor
) -> CacheMember:
    """Mint a ``cache://`` member at *store*, ready to take chunks.

    Born carrying the ``pending`` marker, so a crash before it is published
    leaves a directory the boot sweep recognizes and removes rather than a
    partial tensor the next life would adopt. Raises ``ValueError`` if the
    directory is already there, which is that crash's leftovers.
    """
    store = Path(store)
    try:
        store.mkdir(parents=True)
    except FileExistsError:
        raise ValueError(
            f"{store} already exists. Restart the server to clear a crashed "
            f"upload, or add the tensor under another name."
        ) from None

    member = CacheMember(
        source_id,
        field,
        store,
        SegmentStore(store),
        shape=list(desc.shape),
        dtype=desc.dtype,
        chunk_shape=list(desc.chunk_shape),
        dim_labels=list(desc.dim_labels) if desc.dim_labels else None,
        physical_scale=list(desc.physical_scale) if desc.physical_scale else None,
        physical_unit=list(desc.physical_unit) if desc.physical_unit else None,
        content_version=os.urandom(8),
    )
    # The grid is the uploaded one exactly, not a coalesced one: this format
    # stores the chunks as they arrive, and the index knows those bounds and no
    # others (``get_transfer_chunk_size``).
    member.begin_upload(desc.shape, member.get_transfer_chunk_size())
    member._write_descriptor(UPLOAD_PENDING)
    return member


def open_cache_member(
    store: Path, *, source_id: str, field: str
) -> Optional[CacheMember]:
    """The :class:`CacheMember` at *store*, or None if it is not one.

    None, with a warning, for anything this server did not mint as a cache
    member: no readable descriptor, or none recording a token. Skipped rather
    than served, because a member with no token has no chunk-id namespace of
    its own and would collide with whatever last held the name.

    The chunk ids are **this build's**, minted here over the bounds the index
    holds: what is on disk is keyed by bounds, and only the record the read
    path consults is keyed by id.
    """
    store = Path(store)
    try:
        descriptor = json.loads((store / MEMBER_DESCRIPTOR).read_text())
    except (OSError, ValueError) as e:
        logger.warning(f"member: {store} has no readable descriptor ({e}); skipped")
        return None
    content_version = read_member_version(descriptor)
    if content_version is None:
        logger.warning(f"member: {store} records no content_version; skipped")
        return None
    try:
        member = CacheMember(
            source_id,
            field,
            store,
            SegmentStore.open(store),
            shape=descriptor["shape"],
            dtype=descriptor["dtype"],
            chunk_shape=descriptor["chunk_shape"],
            dim_labels=descriptor.get("dim_labels") or None,
            physical_scale=descriptor.get("physical_scale") or None,
            physical_unit=descriptor.get("physical_unit") or None,
            content_version=content_version,
            # Adopted, so it tracks no upload: the record died with the process
            # that filled it, and only published members are adopted.
            track_upload=False,
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"member: {store} has an unusable descriptor ({e}); skipped")
        return None
    member.adopt_uploaded(
        _parse_bounds_key(key) for key in member._chunks.stored_keys()
    )
    return member
