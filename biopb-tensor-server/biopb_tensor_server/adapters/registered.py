"""Sources the server mints on request, and re-adopts at boot.

A *registered* source is a container the client asks for by name
(``register_source``) and then adds tensors to. It holds no bytes of its own:
its members are separate stores under its directory, one tensor adapter each,
the way a plate owns its fields.

**Nothing under ``write_dir`` is discovered.** The reconciler owns the user's
directories; this subsystem owns ``write_dir``, all of it -- the collections
here, the label sidecars beside them, and (later) the fields on a file source.
One registrar per subtree is what keeps the same bytes from reaching the
catalog twice, once under the id its parent gives it and once under a
path-hash id of its own. Registration therefore happens here, at two moments:
when the client asks, and at boot, when :func:`scan_registered_sources` walks
what the last life left behind.

That boot half is the whole reason this module exists. A store minted by the
upload path used to be registered only in the life that created it, so its
catalog row outlived the adapter behind it and a finished upload stopped being
readable after a restart. Nothing was missing from discovery; what was missing
was the adoption pass (biopb/biopb#1048).

**The id is recorded, not derived.** It is minted here, written into the
collection's ``.zattrs``, and read back at boot -- so it survives moving
``write_dir``, which a hash of the path would not, and it is not predictable
from the name the client chose. A store whose ``.zattrs`` cannot be read is
therefore not a registered source at all: it has no id, so it is swept rather
than served, the same way a label sidecar that lost its token is skipped
rather than served unversioned (``labels.sidecar_label_sets``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters._writable import (
    folded_match,
    taken_store_name,
    unsafe_field_name,
    unsafe_store_name,
    upload_grid,
)
from biopb_tensor_server.adapters.cache_member import (
    create_cache_member,
    open_cache_member,
)
from biopb_tensor_server.adapters.members import (
    MEMBER_ATTR,
    MEMBER_DESCRIPTOR,
    member_attrs,
    member_marker,
    read_member_version,
)
from biopb_tensor_server.adapters.ome_zarr import (
    OmeZarrAdapter,
    _first_dataset_path,
    minimal_ome_metadata,
)
from biopb_tensor_server.adapters.zarr import (
    UPLOAD_PENDING,
    read_zattrs,
    upload_state,
)
from biopb_tensor_server.core.adapter_base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.errors import TensorNotFound
from biopb_tensor_server.core.source_registry import close_adapter

logger = logging.getLogger(__name__)

__all__ = [
    "MEMBER_ATTR",
    "SOURCE_ATTR",
    "STORE_FORMATS",
    "RegisterAdapter",
    "ZarrMember",
    "create_member",
    "create_registered_source",
    "member_attrs",
    "open_any_member",
    "read_member_version",
    "scan_members",
    "scan_registered_sources",
    "sources_root",
]

#: The ``biopb`` sub-block a collection's root ``.zattrs`` carries:
#: ``{"source": {"source_id": ..., "content_version": "<hex>"}}``. Named
#: separately from the sidecar's ``labels`` block so one directory can never be
#: read as the other kind.
SOURCE_ATTR = "source"

#: The schemes ``add_tensor`` takes, each naming a store format and nothing
#: else. The set is a wire contract, so the table is closed; what each does with
#: a request is :func:`create_member`'s.
STORE_FORMATS = ("zarr", "cache")

#: What a minted ``source_id`` looks like: the shape ``generate_source_id``
#: gives a discovered one, with a type of our own so the two namespaces cannot
#: collide.
_ID_PREFIX = "registered"

#: Directory suffix. A collection is a zarr group, so it reads as one to
#: anything that opens it by hand.
_SUFFIX = ".zarr"


def sources_root(write_dir: Path) -> Path:
    """Where every registered source lives: ``<write_dir>/sources/``.

    The one definition of the layout -- ``register_source`` writes under it,
    the boot adoption walks it, and discard removes from it.
    """
    return Path(write_dir) / "sources"


def _mint_source_id() -> str:
    return f"{_ID_PREFIX}_{os.urandom(6).hex()}"


def source_attrs(source_id: str, content_version: bytes) -> dict:
    """The ``biopb`` block a collection's root ``.zattrs`` carries.

    Merged with the source's OME metadata by whoever writes the store. No
    upload marker: a collection is not itself an upload, and an empty one is a
    valid source with an empty tensor list. Its *members* carry the
    pending/ready marker, because the lifecycle is per tensor.
    """
    return {
        "biopb": {
            SOURCE_ATTR: {
                "source_id": source_id,
                "content_version": content_version.hex(),
            }
        }
    }


def read_source_block(store: Path) -> Optional[Dict[str, Any]]:
    """What a collection records: its id, its token, and its own metadata.

    None for anything that is not a registered source: no ``.zattrs``, no
    ``biopb`` block, or a token that will not parse. The caller sweeps those
    rather than adopting them -- an id that cannot be read is an id the catalog
    row cannot be matched to.

    The metadata rides along so adoption parses each ``.zattrs`` once, the way
    ``labels.sidecar_label_sets`` hands its own read down to ``open_label_set``.
    """
    zattrs = read_zattrs(store)
    if zattrs is None:
        return None
    block = (zattrs.get("biopb") or {}).get(SOURCE_ATTR) or {}
    source_id = block.get("source_id")
    try:
        content_version = bytes.fromhex(block["content_version"])
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(source_id, str) or not source_id:
        return None
    return {
        "source_id": source_id,
        "content_version": content_version,
        # The source's own metadata is everything that is not the server's.
        "metadata": {k: v for k, v in zattrs.items() if k != "biopb"},
    }


class RegisterAdapter(SourceAdapter):
    """A source the server minted, owning one tensor adapter per member.

    Holds no bytes: every read goes to a member, and a collection with no
    member is a resolved source with an empty tensor list -- which the catalog
    already models, since an unresolved source carries one too.

    Its ``content_version`` is minted at ``register_source`` and persisted, not
    sampled from a stat signature. It is the collection's identity, not a
    change signal: what it separates is one registered source from the next to
    reuse a name, so that the re-created source's chunk ids cannot collide with
    a tombstone's still sitting in the cache. Members carry their own for the
    same reason, so publishing one never invalidates its siblings.
    """

    _source_type = _ID_PREFIX

    def __init__(
        self,
        source_id: str,
        store: Path,
        content_version: bytes,
        metadata: Optional[dict] = None,
    ) -> None:
        self.source_id = source_id
        self.store = Path(store)
        self._source_url = str(self.store)
        self._content_version = content_version
        self._metadata = metadata or {}
        self._members: Dict[str, TensorAdapter] = {}
        #: When this source last received a tensor -- monotonic, like an
        #: upload's ``updated_at``, and stamped at registration so a create
        #: nobody ever added to is on the same clock as one whose tensors have
        #: all been discarded (``UploadManager._reclaim_empty_source``).
        self.touched_at = time.monotonic()

    # ---- the source surface ------------------------------------------------

    @classmethod
    def create_from_config(cls, config: Any) -> RegisterAdapter:
        raise NotImplementedError(
            "a registered source is minted by register_source, never configured"
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """The members a reader may see: the published ones.

        A member is attached the moment ``add_tensor`` mints it, because that
        is what routes its own writes, and it is enumerated only once READY.
        One gate rather than two dicts (``label_uploads`` beside
        ``label_sets``): the member's own upload record already answers the
        question, so a second map would be the same fact stored twice.
        """
        return [
            member.get_tensor_descriptor()
            for member in self._members.values()
            if _is_published(member)
        ]

    def get_metadata(self) -> dict:
        return dict(self._metadata)

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> TensorAdapter:
        field = self._within_source_field(tensor_id)
        if field is None:
            # No field, and the source's default is its first member -- of
            # which an empty collection has none. Named as a resolution miss
            # rather than a server error: asking a source with no tensors for
            # its tensor is a caller's mistake about what it holds.
            member = next(iter(self._members.values()), None)
            if member is None:
                raise TensorNotFound(
                    f"{self.source_id} has no tensors yet: add one with "
                    f"add_tensor before reading it.",
                    reason="empty_source",
                )
            return member
        member = self._members.get(field)
        if member is None:
            raise TensorNotFound(
                f"{self.source_id} has no tensor {field!r}.",
                reason="unknown_field",
            )
        return member

    def close(self) -> None:
        for member in self._members.values():
            close_adapter(member)
        self._members.clear()

    # ---- members -----------------------------------------------------------

    def member_store(self, field: str) -> Path:
        """Where member *field* keeps its bytes: ``<collection>/<field>/``.

        The one definition of the layout, as :func:`sources_root` is for the
        collection: ``add_tensor`` mints under it, adoption walks it, and a
        discard removes from it.
        """
        return self.store / field

    @property
    def members(self) -> Dict[str, TensorAdapter]:
        return dict(self._members)

    def attach_member(self, field: str, adapter: TensorAdapter) -> None:
        """Take *field* into the source. The member owns its own bytes."""
        self._members[field] = adapter
        self.touched_at = time.monotonic()

    def detach_member(self, field: str) -> Optional[TensorAdapter]:
        return self._members.pop(field, None)

    def taken_field(self, field: str) -> Optional[str]:
        """The member *field* would collide with, folded, or None.

        Folded because NTFS, APFS and HFS+ are case-insensitive and HFS+ stores
        NFD: ``Nuclei`` and ``nuclei`` are two members here and one directory
        there, so an unfolded check mints a second store the next boot on such
        a host cannot tell from the first.
        """
        return folded_match(field, self._members)

    # ---- the store ---------------------------------------------------------

    def dispose_store(self) -> None:
        """Remove the collection whole; what a discard of the source does."""
        shutil.rmtree(self.store, ignore_errors=True)


# -- members ------------------------------------------------------------------


def _is_published(member: TensorAdapter) -> bool:
    """Whether *member* may be enumerated: it is READY, or older than this life.

    A member adopted at boot tracks no upload -- the record died with the
    process that filled it -- and only READY members are adopted, so "no
    record" reads as published. A member of *this* life answers from its own.
    """
    progress = getattr(member, "upload", None)
    return progress is None or progress.is_readable


class ZarrMember(OmeZarrAdapter):
    """An uploaded OME-Zarr image, bound as tensor *field* of its source.

    :class:`~biopb_tensor_server.adapters.labels.LabelSetAdapter` without the
    label half: an ``OmeZarrAdapter`` opened on one group and given its source's
    id, so its chunk ids, catalog entry and ``get_flight_info`` answer are the
    source's and its ``array_id`` is ``<source_id>/<field>``.

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

    *zattrs*, when the caller already has it (``scan_members`` reads it as the
    boot marker), is used as-is rather than read again.
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
    return ZarrMember(
        arr,
        source_id,
        field,
        zattrs=zattrs,
        root_path=str(group),
        content_version=content_version,
    )


def scan_members(adapter: RegisterAdapter) -> Dict[str, TensorAdapter]:
    """The published members of *adapter*'s collection, keyed by field.

    Derived from the directory rather than from a dict something has to
    remember to fill, the shape every other multi-tensor source here uses
    (``OmeZarrAdapter._enumerate_hcs_fields``, ``labels.sidecar_label_sets``).
    A member still PENDING is one a crash left behind; the boot sweep removes
    it, and this pass skips whatever the sweep has not reached yet rather than
    adopting a half-written tensor.

    The format is read off the directory and not off a name -- a zarr member
    carries ``.zattrs``, a cache member ``descriptor.json`` -- which is what
    lets the scheme stay out of the stored ``array_id``.
    """
    members: Dict[str, TensorAdapter] = {}
    for group in sorted(adapter.store.iterdir()):
        if not group.is_dir() or group.name.startswith("."):
            continue
        marker = member_marker(group)
        if upload_state(marker) == UPLOAD_PENDING:
            logger.info(f"member: {group} was left pending; not adopted")
            continue
        member = open_any_member(
            group, source_id=adapter.source_id, field=group.name, marker=marker
        )
        if member is not None:
            members[group.name] = member
    return members


def open_any_member(
    group: Path, *, source_id: str, field: str, marker: Optional[dict] = None
) -> Optional[TensorAdapter]:
    """The member at *group*, in whichever format minted it, or None.

    The one dispatch on layout: the adoption pass has a directory and no idea
    which format it is, and every other caller is in the same position.

    *marker*, when the caller already read it (``scan_members``'s own
    ``member_marker`` call), is handed to whichever format this dispatches to
    instead of being read from disk a second time.
    """
    if (group / MEMBER_DESCRIPTOR).exists():
        return open_cache_member(
            group, source_id=source_id, field=field, descriptor=marker
        )
    return open_member(group, source_id=source_id, field=field, zattrs=marker)


def create_member(
    parent: RegisterAdapter, field: str, scheme: str, desc: TensorDescriptor
) -> TensorAdapter:
    """Mint member *field* of *parent* in the format *scheme* names.

    Takes no metadata, because a member has none of its own: it is
    source-scoped, rides on ``register_source``, and a member declares shape,
    dtype, grid and axes. A label set is the one per-tensor exception and does
    not come through here (``labels.create_label_upload``).

    Raises ``ValueError`` for a request no format can serve -- an unusable or
    reserved field name, a field already taken, an unknown scheme -- and
    nothing touches disk until every one of them has passed, so a refused
    request leaves no store behind.
    """
    why = unsafe_field_name(field)
    if why is not None:
        raise ValueError(f"{field!r} cannot name a tensor: the name {why}.")
    taken = parent.taken_field(field)
    if taken is not None:
        raise ValueError(
            f"{parent.source_id} already has a tensor {taken!r}. A field is "
            f"taken for as long as it is served, and two names differing only "
            f"by case or accent form are one directory on Windows and macOS; "
            f"discard that tensor, or add this one under another name."
        )
    if scheme == "cache":
        return create_cache_member(
            parent.member_store(field), parent.source_id, field, desc
        )
    if scheme == "zarr":
        return _create_zarr_member(parent, field, desc)
    raise ValueError(
        f"{scheme!r} is not a store format: use "
        f"{' or '.join(repr(s) for s in STORE_FORMATS)}."
    )


def _create_zarr_member(
    parent: RegisterAdapter, field: str, desc: TensorDescriptor
) -> ZarrMember:
    """A ``zarr://`` member: an OME-Zarr image group under the collection.

    Born carrying the ``pending`` marker, so a crash before it is published
    leaves a directory the boot sweep recognizes and removes rather than a
    partial tensor the next life would adopt.
    """
    import zarr

    content_version = os.urandom(8)
    zattrs = {
        **minimal_ome_metadata(desc),
        **member_attrs(content_version, UPLOAD_PENDING),
    }
    store = parent.member_store(field)
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
        parent.source_id,
        field,
        zattrs=zattrs,
        root_path=str(store),
        content_version=content_version,
    )
    adapter._upload_store_path = store
    adapter.begin_upload(desc.shape, grid)
    return adapter


def create_registered_source(
    name: str, metadata: Optional[dict], sources_dir: Path
) -> RegisterAdapter:
    """Mint a registered source under *sources_dir*.

    Raises ``ValueError`` when the name cannot be a directory or is already
    taken -- the caller turns that into the wire refusal.
    """
    zarr_name = name or f"source_{os.urandom(8).hex()}"
    why = unsafe_store_name(zarr_name, suffix=_SUFFIX)
    if why is not None:
        raise ValueError(f"register_source: the name {zarr_name!r} {why}.")

    sources_dir = Path(sources_dir)
    sources_dir.mkdir(parents=True, exist_ok=True)
    taken = taken_store_name(sources_dir, zarr_name, _SUFFIX)
    if taken is not None:
        raise ValueError(
            f"register_source: {taken} already exists, and two names differing "
            f"only by case or accent form are one directory on Windows and "
            f"macOS. Discard that source, or use another name."
        )

    store = sources_dir / f"{zarr_name}{_SUFFIX}"
    try:
        store.mkdir()
    except FileExistsError:
        raise ValueError(
            f"register_source: {store} already exists. A name is taken while "
            f"its directory is on disk; discard that source, or use another "
            f"name."
        ) from None

    source_id = _mint_source_id()
    content_version = os.urandom(8)
    zattrs = dict(metadata or {})
    zattrs.update(source_attrs(source_id, content_version))
    # A zarr group, so the directory reads as one to anything that opens it.
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (store / ".zattrs").write_text(json.dumps(zattrs))
    logger.info(f"Registered source {source_id} at {store}")
    return RegisterAdapter(source_id, store, content_version, metadata or {})


def scan_registered_sources(sources_dir: Path) -> Dict[str, RegisterAdapter]:
    """Every registered source under *sources_dir*, keyed by ``source_id``.

    The boot half of registration, and the half whose absence made a finished
    upload stop being readable after a restart. A directory that records no
    readable id is **removed**: it has no identity to adopt, so leaving it
    would accumulate bytes nothing can reach or name.
    """
    sources_dir = Path(sources_dir)
    if not sources_dir.is_dir():
        return {}
    adopted: Dict[str, RegisterAdapter] = {}
    for store in sorted(sources_dir.glob(f"*{_SUFFIX}")):
        block = read_source_block(store)
        if block is None:
            logger.warning(f"Removing {store}: it records no usable source_id")
            shutil.rmtree(store, ignore_errors=True)
            continue
        source_id = block["source_id"]
        if source_id in adopted:
            # Two directories recording one id: the second cannot be served
            # under it, and guessing which is current would be a coin flip.
            logger.warning(
                f"Skipping {store}: source_id {source_id} is already adopted "
                f"from {adopted[source_id].store}"
            )
            continue
        adapter = RegisterAdapter(
            source_id, store, block["content_version"], block["metadata"]
        )
        for field, member in scan_members(adapter).items():
            adapter.attach_member(field, member)
        adopted[source_id] = adapter
    return adopted
