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
when the client asks, and at boot, when :func:`adopt_registered_sources` walks
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
from pathlib import Path
from typing import Any, Dict, List, Optional

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters._writable import fold_name, unsafe_store_name
from biopb_tensor_server.adapters.zarr import read_zattrs
from biopb_tensor_server.core.adapter_base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.errors import TensorNotFound

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCE_ATTR",
    "RegisterAdapter",
    "adopt_registered_sources",
    "create_registered_source",
    "sources_root",
]

#: The ``biopb`` sub-block a collection's root ``.zattrs`` carries:
#: ``{"source": {"source_id": ..., "content_version": "<hex>"}}``. Named
#: separately from the sidecar's ``labels`` block so one directory can never be
#: read as the other kind.
SOURCE_ATTR = "source"

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
    """The ``(source_id, content_version)`` a collection records, or None.

    None for anything that is not a registered source: no ``.zattrs``, no
    ``biopb`` block, or a token that will not parse. The caller sweeps those
    rather than adopting them -- an id that cannot be read is an id the catalog
    row cannot be matched to.
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
    return {"source_id": source_id, "content_version": content_version}


def _ome_metadata(store: Path) -> dict:
    """The source's own metadata: its ``.zattrs`` minus the server's block."""
    zattrs = read_zattrs(store) or {}
    return {k: v for k, v in zattrs.items() if k != "biopb"}


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
        self._metadata = metadata if metadata is not None else _ome_metadata(self.store)
        self._members: Dict[str, TensorAdapter] = {}

    # ---- the source surface ------------------------------------------------

    @classmethod
    def create_from_config(cls, config: Any) -> RegisterAdapter:
        raise NotImplementedError(
            "a registered source is minted by register_source, never configured"
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [member.get_tensor_descriptor() for member in self._members.values()]

    def get_metadata(self) -> dict:
        return dict(self._metadata)

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> TensorAdapter:
        field = self._field_of(tensor_id)
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

    def _field_of(self, tensor_id: Optional[str]) -> Optional[str]:
        if not tensor_id or tensor_id == self.source_id:
            return None
        prefix = f"{self.source_id}/"
        return tensor_id[len(prefix) :] if tensor_id.startswith(prefix) else tensor_id

    def close(self) -> None:
        for member in self._members.values():
            close = getattr(member, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # pragma: no cover - teardown is best effort
                    logger.debug(f"closing member of {self.source_id}", exc_info=True)
        self._members.clear()

    # ---- members -----------------------------------------------------------

    @property
    def members(self) -> Dict[str, TensorAdapter]:
        return dict(self._members)

    def attach_member(self, field: str, adapter: TensorAdapter) -> None:
        """Take *field* into the listing. The member owns its own bytes."""
        self._members[field] = adapter

    def detach_member(self, field: str) -> Optional[TensorAdapter]:
        return self._members.pop(field, None)

    def taken_field(self, field: str) -> Optional[str]:
        """The member *field* would collide with, folded, or None.

        Folded because NTFS, APFS and HFS+ are case-insensitive and HFS+ stores
        NFD: two fields differing only that way are one directory there and two
        here (``fold_name``).
        """
        folded = fold_name(field)
        return next((f for f in self._members if fold_name(f) == folded), None)

    # ---- the store ---------------------------------------------------------

    def dispose_store(self) -> None:
        """Remove the collection whole; what a discard of the source does."""
        shutil.rmtree(self.store, ignore_errors=True)


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
    # Folded before the mkdir: on a case-insensitive filesystem the exclusive
    # create below would catch this by itself, on ext4 it would not, and the
    # difference is a source that splits in two on the next host to serve this
    # write_dir.
    folded = fold_name(zarr_name)
    for existing in sources_dir.glob(f"*{_SUFFIX}"):
        if fold_name(existing.name[: -len(_SUFFIX)]) == folded:
            raise ValueError(
                f"register_source: {existing} already exists, and two names "
                f"differing only by case or accent form are one directory on "
                f"Windows and macOS. Discard that source, or use another name."
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


def adopt_registered_sources(sources_dir: Path) -> Dict[str, RegisterAdapter]:
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
        adopted[source_id] = RegisterAdapter(source_id, store, block["content_version"])
    return adopted
