"""The two formats an uploaded tensor's store is minted in.

A member is one uploaded tensor's bytes, under ``<write_dir>/fields/``. There
are two layouts -- an OME-Zarr image group (:class:`ZarrMember`) and a segment
store (``adapters.cache_member.CacheMember``) -- and this is where one is
chosen (:func:`create_member_at`) and where a directory is read back as
whichever minted it (:func:`open_any_member`), for a caller that has a path and
no idea which format it holds.

The vocabulary the two share -- the bookkeeping block, the upload marker, the
deadline -- is ``adapters.members``, which sits below both so either can be
imported without the other. The *layout* they sit in, and the name rules over
it, are ``adapters.fields``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters._writable import upload_grid
from biopb_tensor_server.adapters.cache_member import (
    create_cache_member,
    open_cache_member,
)
from biopb_tensor_server.adapters.members import (
    MEMBER_ATTR,
    MEMBER_DESCRIPTOR,
    member_attrs,
    read_member_version,
    upload_expires_at,
)
from biopb_tensor_server.adapters.ome_zarr import (
    OmeZarrAdapter,
    _first_dataset_path,
    minimal_ome_metadata,
)
from biopb_tensor_server.adapters.zarr import (
    UPLOAD_PENDING,
    read_zattrs,
)
from biopb_tensor_server.core.adapter_base import TensorAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "MEMBER_ATTR",
    "STORE_FORMATS",
    "ZarrMember",
    "create_member_at",
    "member_attrs",
    "open_any_member",
    "read_member_version",
]

#: The schemes ``add_tensor`` takes, each naming a store format and nothing
#: else. The set is a wire contract, so the table is closed; what each does with
#: a request is :func:`create_member_at`'s.
STORE_FORMATS = ("zarr", "cache")


class ZarrMember(OmeZarrAdapter):
    """An uploaded OME-Zarr image, bound as tensor *field* of its source.

    :class:`~biopb_tensor_server.adapters.labels.LabelSetAdapter` without the
    label half: an ``OmeZarrAdapter`` opened on one group and given its source's
    id, so its chunk ids, catalog entry and ``get_flight_info`` answer are the
    source's and its ``array_id`` is ``<source_id>/@fields/<name>``.

    Its ``content_version`` is the token in its own ``.zattrs``, not the store
    directory's stat signature: an uploaded tensor's bytes never change under
    it, and a sampled signature would move on any touch and orphan every chunk
    a client had already cached (biopb/biopb#178).
    """

    def __init__(
        self,
        zarr_array: Any,
        source_id: str,
        field: str,
        *,
        zattrs: dict,
        root_path: str,
        content_version: bytes,
    ) -> None:
        # The root is threaded so the constructor never walks up from the
        # array: the collection's own ``.zattrs`` sits one level above and
        # would otherwise win the walk, presenting this member as its source.
        super().__init__(
            zarr_array,
            source_id,
            None,
            _threaded_zattrs=zattrs,
            _threaded_root=root_path,
        )
        self._tensor_name = field
        self._content_version = content_version

    def get_tensor_metadata(self) -> Optional[dict]:
        """The member's own NGFF, minus the server's bookkeeping block.

        Merged over the source's catalog row by ``get_flight_info``, the way a
        label set's is -- so the ``multiscales`` a reader sees describes this
        member's array and the physical scale it inherits is the source's.
        """
        return {k: v for k, v in self.ome_metadata.items() if k != "biopb"}

    def delete_store(self) -> None:
        """Remove this member's store, for one with no upload record left.

        The counterpart of ``LabelSetAdapter.delete_store``: a member adopted
        from an earlier life holds no record to seal, so discarding it is store
        removal alone. A member *this* life uploaded goes through its upload's
        discard instead, which seals the record as well.
        """
        self._dispose_store()


def open_member(
    group: Path, *, source_id: str, field: str, zattrs: Optional[dict] = None
) -> Optional[ZarrMember]:
    """A :class:`ZarrMember` on the OME-Zarr group at *group*, or None.

    None, with a warning, for anything this server did not mint as a member: no
    readable ``.zattrs``, no recorded token, or no level-0 array. Skipped rather
    than served, because a member with no token has no chunk-id namespace of
    its own and would collide with whatever last held the name.

    *zattrs*, when the caller already has it (``fields.scan_source_fields``
    reads it as the boot marker), is used as-is rather than read again.
    """
    import zarr

    if zattrs is None:
        zattrs = read_zattrs(group)
    if zattrs is None:
        logger.warning(f"member: {group} has no readable .zattrs; skipped")
        return None
    content_version = read_member_version(zattrs)
    if content_version is None:
        logger.warning(f"member: {group} records no content_version; skipped")
        return None
    level0 = _first_dataset_path(zattrs.get("multiscales", [])) or "0"
    try:
        arr = zarr.open_array(os.path.join(str(group), level0), mode="r")
    except Exception as e:
        logger.warning(f"member: cannot open {group}/{level0}: {e}; skipped")
        return None
    member = ZarrMember(
        arr,
        source_id,
        field,
        zattrs=zattrs,
        root_path=str(group),
        content_version=content_version,
    )
    # Minted by an earlier life of this server, under its own ``write_dir``, so
    # it is still the server's to delete -- the same note ``sidecar_label_sets``
    # puts on a re-opened set. Without it ``delete_store`` has no path and
    # silently leaves the bytes behind.
    member._upload_store_path = group
    # The deadline outlives the record, so an adopted member's lifetime is still
    # enforced by the sweep (``WritableSource.expired``).
    member._expires_at = upload_expires_at(zattrs)
    return member


def open_any_member(
    group: Path, *, source_id: str, field: str, marker: Optional[dict] = None
) -> Optional[TensorAdapter]:
    """The member at *group*, in whichever format minted it, or None.

    The one dispatch on layout: the adoption pass has a directory and no idea
    which format it is, and every other caller is in the same position.

    *marker*, when the caller already read it (``fields.scan_source_fields``'s
    own ``member_marker`` call), is handed to whichever format this dispatches
    to instead of being read from disk a second time.
    """
    if (group / MEMBER_DESCRIPTOR).exists():
        return open_cache_member(
            group, source_id=source_id, field=field, descriptor=marker
        )
    return open_member(group, source_id=source_id, field=field, zattrs=marker)


def create_member_at(
    store: Path,
    source_id: str,
    field: str,
    scheme: str,
    desc: TensorDescriptor,
    expires_at: Optional[float] = None,
) -> TensorAdapter:
    """Mint a member directory at *store* in the format *scheme* names.

    The one place a store format is chosen. The name rules and the directory
    belong to whoever owns the layout (``adapters.fields.create_field_upload``),
    so a field gets both formats wherever its source came from.
    """
    if scheme == "cache":
        return create_cache_member(store, source_id, field, desc, expires_at)
    if scheme == "zarr":
        return _create_zarr_member(store, source_id, field, desc, expires_at)
    raise ValueError(
        f"{scheme!r} is not a store format: use "
        f"{' or '.join(repr(s) for s in STORE_FORMATS)}."
    )


def _create_zarr_member(
    store: Path,
    source_id: str,
    field: str,
    desc: TensorDescriptor,
    expires_at: Optional[float] = None,
) -> ZarrMember:
    """A ``zarr://`` member: an OME-Zarr image group at *store*.

    Born carrying the ``pending`` marker, so a crash before it is published
    leaves a directory the boot sweep recognizes and removes rather than a
    partial tensor the next life would adopt.
    """
    import zarr

    content_version = os.urandom(8)
    zattrs = {
        **minimal_ome_metadata(desc),
        **member_attrs(content_version, UPLOAD_PENDING, expires_at),
    }
    # Exclusive: the directory must be this create's own, because a discard
    # removes it whole. One already on disk under a field the source does not
    # serve is a crashed upload for the boot sweep, not something to adopt.
    try:
        store.mkdir(parents=True)
    except FileExistsError:
        raise ValueError(
            f"{store} already exists. Restart the server to clear a crashed "
            f"upload, or add the tensor under another name."
        ) from None
    grid = upload_grid(desc)
    group = zarr.open_group(str(store), mode="w")
    arr = group.create_dataset(
        "0", shape=list(desc.shape), chunks=grid, dtype=desc.dtype
    )
    (store / ".zattrs").write_text(json.dumps(zattrs))

    adapter = ZarrMember(
        arr,
        source_id,
        field,
        zattrs=zattrs,
        root_path=str(store),
        content_version=content_version,
    )
    adapter._upload_store_path = store
    adapter.begin_upload(desc.shape, grid, expires_at)
    return adapter
