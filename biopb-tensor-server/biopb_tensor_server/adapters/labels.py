"""A label set as a tensor of its image (biopb/biopb#1059).

An NGFF label image is an OME-Zarr multiscales group with an ``image-label``
block, wherever it sits: under its image's ``labels/`` group in the file, or in
a sidecar the server minted for an upload. :class:`LabelSetAdapter` is
:class:`~biopb_tensor_server.adapters.ome_zarr.OmeZarrAdapter` opened on that
group and bound as a tensor of the *parent* source -- same ``source_id``, field
``[<image field>/]labels/<name>`` -- so its chunk ids, catalog entry and
``get_flight_info`` answer are the parent's, and its native levels ride under
its own field (``OmeZarrAdapter.get_level_adapter`` composes the level's name
from the set's).

What a set adds over a plain OME-Zarr image:

- **nearest on every computed level**: averaging label ids produces ids that
  exist nowhere, so the advertised ladder never says ``area`` for a set;
- **a ``content_version`` that is not its own directory's**: the parent's for
  a set the file carries (it goes stale with the file), the token minted at
  upload for a sidecar (persisted in its attrs);
- **``image-label`` in its per-tensor metadata**, ``source.image`` naming the
  parent's ``array_id`` -- NGFF's relative path would be meaningless for a
  sidecar that is not adjacent to its image;
- **write-once**: the adapter :func:`create_label_upload` mints takes the
  upload's chunks and nothing else. Every other set -- one the file carries,
  one read back off a sidecar at startup -- tracks no upload and refuses a
  write outright; replacement is a new name, or a delete first.

Three ways a set reaches a parent: :func:`native_label_sets` for an image
group's ``labels/`` (called from ``OmeZarrAdapter.get_embedded_labels``),
:func:`sidecar_label_sets` for the finished stores under
``<write_dir>/labels/<source_id>/``, which :func:`sidecar_attacher` runs at
registration, and :func:`create_label_upload` for a set arriving over the wire.
The readers skip only what they cannot *open* -- a float dtype, an unreadable
``.zattrs`` -- with a warning; whether a set spans its image is checked once
for every origin where the sets meet (``SourceAdapter.label_binding_error``,
from the upload's create and from ``label_sets``).

Design: ``biopb-tensor-server/docs/label-tensors.md``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import PyramidLevel, TensorDescriptor

from biopb_tensor_server.adapters._writable import unsafe_store_name
from biopb_tensor_server.adapters.ome_zarr import (
    OmeZarrAdapter,
    _first_dataset_path,
    minimal_ome_metadata,
)
from biopb_tensor_server.adapters.zarr import (
    UPLOAD_PENDING,
    UPLOAD_READY,
    read_zattrs,
    upload_state,
    with_upload_state,
)
from biopb_tensor_server.core.config import PyramidConfig
from biopb_tensor_server.core.errors import WriteNotSupportedError
from biopb_tensor_server.core.labels import (
    RESERVED_PREFIX,
    join_fields,
    label_extent,
    label_field,
    split_label_field,
)

__all__ = [
    "LabelSetAdapter",
    "create_label_upload",
    "labels_root",
    "native_label_sets",
    "open_label_set",
    "sidecar_attacher",
    "sidecar_attrs",
    "sidecar_dir",
    "sidecar_label_sets",
]

logger = logging.getLogger(__name__)

#: Inside a sidecar's root ``.zattrs`` ``biopb`` block, beside the upload
#: marker: ``{"labels": {"image_field": "<field or ''>", "content_version":
#: "<hex>"}}``. The image field is what binds the set to one tensor of a
#: multi-tensor parent; the token is what namespaces its chunk ids.
SIDECAR_ATTR = "labels"


class LabelSetAdapter(OmeZarrAdapter):
    """An NGFF label image, bound as tensor *field* of source *source_id*."""

    def __init__(
        self,
        zarr_array: Any,
        source_id: str,
        field: str,
        *,
        zattrs: dict,
        root_path: str,
        content_version: Optional[bytes],
        parent_array_id: str,
    ) -> None:
        # The root is threaded so the constructor never walks up from the
        # array: an ancestor plate or image ``.zattrs`` would otherwise win the
        # walk and this set would present as its parent.
        super().__init__(
            zarr_array,
            source_id,
            None,
            _threaded_zattrs=zattrs,
            _threaded_root=root_path,
        )
        self._tensor_name = field
        self._content_version = content_version
        self._parent_array_id = parent_array_id

    def get_embedded_labels(self) -> Dict[str, Any]:
        return {}  # a set has no sets, and never looks for a labels/ of its own

    def _advertised_pyramid(
        self, base_desc: TensorDescriptor, pyramid_config: PyramidConfig
    ) -> List[PyramidLevel]:
        """Native levels as they are; computed levels always ``nearest``."""
        return super()._advertised_pyramid(
            base_desc, dataclasses.replace(pyramid_config, reduction_method="nearest")
        )

    def get_tensor_metadata(self) -> Optional[dict]:
        """The set's own NGFF metadata, with ``image-label`` naming its image.

        Merged over the parent's catalog row by ``get_flight_info``, so the
        row's ``multiscales`` (the image's) is replaced by the set's. The
        ``biopb`` block a sidecar carries is bookkeeping and stays out.
        """
        meta = {k: v for k, v in self.ome_metadata.items() if k != "biopb"}
        label = dict(meta.get("image-label") or {})
        label.setdefault("version", "0.4")
        label["source"] = {"image": self._parent_array_id}
        meta["image-label"] = label
        return meta

    def put_chunk(self, bounds, data, expected_shape, dtype) -> None:
        """Write one chunk -- only while this adapter is the upload that minted it.

        A set the file carries, or one read back off a sidecar at startup,
        tracks no upload and is read-only: replacement is a new name or a
        delete, never a chunk landing under a finished set (design,
        "No per-instance edits").
        """
        if self.upload is None:
            raise WriteNotSupportedError(
                f"label set {self.array_id!r} is read-only; upload a new set to "
                "replace it"
            )
        super().put_chunk(bounds, data, expected_shape, dtype)

    def delete_store(self) -> None:
        """Remove this set's sidecar; the adapter's half of ``delete_labels``.

        A store the server minted under its ``write_dir`` is the server's to
        throw away (design, "Lifecycle"), and only such a set reaches here --
        a group inside a user's file carries no store path and this is a
        no-op on it. A set *this* server uploaded still holds its (READY)
        progress record, so it goes through ``discard``, which seals that
        record as well as releasing the store; one adopted from an earlier
        life holds no record and only the store goes.
        """
        if self.upload is not None:
            self.discard("deleted")
        else:
            self._dispose_store()

    def upload_response(self, desc: TensorDescriptor) -> TensorDescriptor:
        """The create echo, under the set's own ``array_id``.

        The two prefixed kinds mint a ``source_id`` and answer with it; a set
        is a tensor of a source that already exists, so the id the request
        carried is the id it keeps -- and is what every later write, poll and
        finish names.
        """
        response = super().upload_response(desc)
        response.array_id = self.array_id
        return response


# -- readers ------------------------------------------------------------------


def open_label_set(
    group: Path,
    *,
    source_id: str,
    image_field: str,
    name: str,
    content_version: Optional[bytes],
    zattrs: Optional[dict] = None,
) -> Optional[LabelSetAdapter]:
    """A :class:`LabelSetAdapter` on the NGFF label group at *group*, or None.

    None, with a warning, for anything that cannot be opened as labels: no
    readable ``.zattrs``, no level-0 array, or a dtype that is not an integer
    (a label is an id; there is nothing a float set could mean here). Pass an
    already-parsed *zattrs* when the caller has one, so *group*'s root file
    is not read twice.
    """
    import zarr

    if zattrs is None:
        zattrs = read_zattrs(group)
    if zattrs is None:
        logger.warning(f"labels: {group} has no readable .zattrs; skipped")
        return None
    level0 = _first_dataset_path(zattrs.get("multiscales", [])) or "0"
    try:
        arr = zarr.open_array(os.path.join(str(group), level0), mode="r")
    except Exception as e:
        logger.warning(f"labels: cannot open {group}/{level0}: {e}; skipped")
        return None
    if arr.dtype.kind not in "ui":
        logger.warning(f"labels: {group} is {arr.dtype}, not an integer dtype; skipped")
        return None
    return LabelSetAdapter(
        arr,
        source_id,
        label_field(image_field, name),
        zattrs=zattrs,
        root_path=str(group),
        content_version=content_version,
        parent_array_id=join_fields(source_id, image_field),
    )


def native_label_sets(
    parent: Any, image_group: Path, image_field: str
) -> Dict[str, LabelSetAdapter]:
    """The sets in *image_group*'s NGFF ``labels/`` group, keyed by field.

    ``labels/.zattrs`` lists the names (``{"labels": [...]}``); a group that
    omits the list is scanned for subgroups with a ``.zattrs`` instead. A
    name with a slash cannot be addressed and is skipped. Each set carries the
    parent's ``content_version``: it is part of the file and goes stale with it.
    """
    labels_dir = image_group / "labels"
    if not labels_dir.is_dir():
        return {}
    names = (read_zattrs(labels_dir) or {}).get("labels")
    if not isinstance(names, list):
        names = sorted(d.name for d in labels_dir.iterdir() if (d / ".zattrs").exists())
    sets: Dict[str, LabelSetAdapter] = {}
    for name in names:
        if not isinstance(name, str) or not name or "/" in name:
            logger.warning(
                f"labels: {labels_dir} lists unusable name {name!r}; skipped"
            )
            continue
        label_set = open_label_set(
            labels_dir / name,
            source_id=parent.source_id,
            image_field=image_field,
            name=name,
            content_version=parent.content_version,
        )
        if label_set is not None:
            sets[label_field(image_field, name)] = label_set
    return sets


def labels_root(write_dir: Path) -> Path:
    """Where every source's uploaded sets live: ``<write_dir>/labels/``.

    The one definition of the sidecar layout: the attacher reads it, the
    upload kind writes under it, and the boot sweep globs it.
    """
    return write_dir / "labels"


def sidecar_dir(labels_dir: Path, source_id: str) -> Path:
    """Where one source's uploaded sets live: ``<write_dir>/labels/<source_id>/``."""
    return labels_dir / source_id


def sidecar_attrs(
    image_field: str, content_version: bytes, *, state: str = UPLOAD_READY
) -> dict:
    """The ``biopb`` block a sidecar's root ``.zattrs`` carries.

    Merged with the NGFF metadata by whoever writes the store. The upload
    marker rides in the same block: ``pending`` from create, flipped to
    ``ready`` by ``finish`` (``ZarrAdapter._mark_store_finished``), and
    :func:`sidecar_label_sets` attaches nothing that is not ``ready``.
    """
    attrs = with_upload_state({}, state)
    attrs["biopb"][SIDECAR_ATTR] = {
        "image_field": image_field,
        "content_version": content_version.hex(),
    }
    return attrs


def sidecar_label_sets(source_id: str, labels_dir: Path) -> Dict[str, LabelSetAdapter]:
    """The finished sidecar sets of *source_id* under *labels_dir*, keyed by field.

    ``<name>.zarr`` groups whose upload marker reads ``ready``; a pending one
    is an upload still filling (or one that died, which the boot sweep takes),
    and is not a tensor yet. The token is server-minted, so a store without a
    readable one is corrupt and skipped rather than served unversioned.
    """
    root = sidecar_dir(labels_dir, source_id)
    if not root.is_dir():
        return {}
    sets: Dict[str, LabelSetAdapter] = {}
    for store in sorted(root.glob("*.zarr")):
        zattrs = read_zattrs(store)
        if zattrs is None or upload_state(zattrs) != UPLOAD_READY:
            continue
        block = (zattrs.get("biopb") or {}).get(SIDECAR_ATTR) or {}
        try:
            content_version = bytes.fromhex(block["content_version"])
        except (KeyError, TypeError, ValueError):
            logger.warning(
                f"labels: {store} carries no usable content_version; skipped"
            )
            continue
        image_field = block.get("image_field") or ""
        name = store.name[: -len(".zarr")]
        label_set = open_label_set(
            store,
            source_id=source_id,
            image_field=image_field,
            name=name,
            content_version=content_version,
            zattrs=zattrs,
        )
        if label_set is not None:
            # Minted by an earlier life of this server, under its own
            # ``write_dir``, so it is still the server's to delete
            # (``delete_store``) -- unlike a label group inside a user's file.
            label_set._upload_store_path = store
            sets[label_field(image_field, name)] = label_set
    return sets


# -- the upload kind ----------------------------------------------------------


def create_label_upload(
    parent: Any,
    field: str,
    desc: TensorDescriptor,
    *,
    labels_dir: Path,
    metadata: Optional[dict] = None,
) -> LabelSetAdapter:
    """Mint the sidecar for a new uploaded set on *parent* and track its upload.

    The third upload kind (design, "Upload"): unlike ``cache:`` / ``ome_zarr:``
    it is not selected by a prefix and mints no ``source_id`` -- the request's
    ``array_id`` is a tensor of a source that already exists, and is the id the
    set keeps. *field* is that id's within-source half, already split by the
    boundary; *desc* is the request in canonical order (the boundary refuses
    any other), and is filled in with the image's axes when it named none --
    in place, because it is also the descriptor the client is answered with.

    Raises ``ValueError`` for a request the kind cannot serve -- an
    unresolved parent, a reserved name, a dtype that is not an unsigned
    integer, a name already taken on this parent, or a shape that does not
    span the image (``SourceAdapter.label_binding_error``, the same rule the
    listing re-checks). Nothing touches disk until every one of them has
    passed, so a refused request leaves no store behind.
    """
    import zarr

    array_id = join_fields(parent.source_id, field)
    parsed = split_label_field(field)
    if parsed is None or parsed.level is not None:
        raise ValueError(
            f"{array_id!r} does not name a label set: an uploaded set is "
            f"'<image array_id>/labels/<name>' with a slash-free name."
        )
    if parsed.name.startswith(RESERVED_PREFIX):
        raise ValueError(
            f"{array_id!r}: names under {RESERVED_PREFIX!r} are the server's "
            f"own; upload under another name."
        )
    # The name is slash-free by construction -- the field was split on "/" --
    # but it still becomes a directory the server creates and later removes
    # whole, so the rest of the rule applies: ".." and a Windows separator or
    # drive letter would both escape the sidecar directory.
    why = unsafe_store_name(parsed.name)
    if why is not None:
        raise ValueError(f"{array_id!r}: the set's name {why}.")
    if not parent.is_resolved():
        raise ValueError(
            f"{array_id!r}: source {parent.source_id!r} is not resolved, so "
            f"there is no image extent to check a set against; resolve it first."
        )
    if np.dtype(desc.dtype).kind != "u":
        raise ValueError(
            f"{array_id!r}: a label set is unsigned integer ids with 0 for "
            f"background, not {np.dtype(desc.dtype)}."
        )
    if field in parent.label_sets or field in parent.label_uploads:
        raise ValueError(
            f"{array_id!r} already exists. A set's name is taken for as long "
            f"as it is served; delete it first, or upload under another name."
        )
    images = parent._normalized_tensors()
    if not desc.dim_labels:
        # The extent rule leaves exactly one legal set of axes for this image,
        # so a request that named none is filled in rather than refused. In
        # place, so the descriptor ``create_tensor`` echoes back carries them:
        # everything downstream (the sidecar's NGFF, the chunk grid, the
        # client's own later calls) is built from that descriptor.
        image = parent.label_image_descriptor(field, images=images)
        if image is not None:
            desc.dim_labels.extend(label_extent(image.dim_labels, image.shape)[0])
    why = parent.label_binding_error(field, desc, images=images)
    if why is not None:
        raise ValueError(f"{array_id!r} {why}")

    content_version = os.urandom(8)
    # The store's own NGFF, composed here rather than taken from the request:
    # the multiscales must describe the array this call is about to create.
    # What the client may contribute is the ``image-label`` block -- the
    # colours and properties of its ids, which nothing else knows; its
    # ``source`` is the parent's and is stamped on read.
    image_label = dict((metadata or {}).get("image-label") or {})
    image_label.pop("source", None)
    image_label.setdefault("version", "0.4")
    zattrs = {
        **minimal_ome_metadata(desc),
        "image-label": image_label,
        **sidecar_attrs(parsed.image_field, content_version, state=UPLOAD_PENDING),
    }
    store = sidecar_dir(labels_dir, parent.source_id) / f"{parsed.name}.zarr"
    # Exclusive, like ``OmeZarrAdapter.create_upload``: the directory must be
    # this create's own, because discard removes it whole. A store on disk under
    # a name the parent does not serve is a crashed upload the boot sweep will
    # take, not something to adopt.
    try:
        store.mkdir(parents=True)
    except FileExistsError:
        raise ValueError(
            f"{array_id!r}: {store} already exists. Restart the server to clear "
            f"a crashed upload, or upload under another name."
        ) from None
    group = zarr.open_group(str(store), mode="w")
    arr = group.create_dataset(
        "0",
        shape=list(desc.shape),
        chunks=list(desc.chunk_shape),
        dtype=desc.dtype,
    )
    (store / ".zattrs").write_text(json.dumps(zattrs))

    adapter = LabelSetAdapter(
        arr,
        parent.source_id,
        field,
        zattrs=zattrs,
        root_path=str(store),
        content_version=content_version,
        parent_array_id=join_fields(parent.source_id, parsed.image_field),
    )
    adapter._upload_store_path = store
    adapter.begin_upload(desc.shape, desc.chunk_shape)
    return adapter


def sidecar_attacher(labels_dir: Path) -> Callable[[str, Any], None]:
    """The registry's ``on_register`` hook: attach a source's finished sidecars.

    Runs at the one registration chokepoint, because a sidecar is keyed by
    ``source_id`` and no format knows about it. The registry stays ignorant of
    the layout; this module owns it. A sidecar that will not open costs the
    set, never the source.
    """

    def attach(source_id: str, adapter: Any) -> None:
        for field, label_set in sidecar_label_sets(source_id, labels_dir).items():
            adapter.attach_label_set(field, label_set)

    return attach
