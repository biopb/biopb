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
- **read-only**: a native set is the file's, a sidecar is written by the
  upload path and sealed before it is ever attached.

Two readers build sets: :func:`native_label_sets` for an image group's
``labels/`` (called from ``OmeZarrAdapter.get_embedded_labels``) and
:func:`sidecar_label_sets` for ``<write_dir>/labels/<source_id>/``, which
:func:`sidecar_attacher` runs at registration. Both skip only what they cannot
*open* -- a float dtype, an unreadable ``.zattrs`` -- with a warning; whether
a set spans its image is checked once for every origin where the sets meet
(``SourceAdapter.label_sets``).

Design: ``biopb-tensor-server/docs/label-tensors.md``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from biopb.tensor.descriptor_pb2 import PyramidLevel, TensorDescriptor

from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter, _first_dataset_path
from biopb_tensor_server.adapters.zarr import (
    UPLOAD_READY,
    read_zattrs,
    upload_state,
    with_upload_state,
)
from biopb_tensor_server.core.config import PyramidConfig
from biopb_tensor_server.core.errors import WriteNotSupportedError
from biopb_tensor_server.core.labels import join_fields, label_field

__all__ = [
    "LabelSetAdapter",
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
        raise WriteNotSupportedError(
            f"label set {self.array_id!r} is read-only; upload a new set to replace it"
        )


# -- readers ------------------------------------------------------------------


def open_label_set(
    group: Path,
    *,
    source_id: str,
    image_field: str,
    name: str,
    content_version: Optional[bytes],
) -> Optional[LabelSetAdapter]:
    """A :class:`LabelSetAdapter` on the NGFF label group at *group*, or None.

    None, with a warning, for anything that cannot be opened as labels: no
    readable ``.zattrs``, no level-0 array, or a dtype that is not an integer
    (a label is an id; there is nothing a float set could mean here).
    """
    import zarr

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


def sidecar_dir(labels_dir: Path, source_id: str) -> Path:
    """Where a source's uploaded sets live: ``<write_dir>/labels/<source_id>/``."""
    return labels_dir / source_id


def sidecar_attrs(image_field: str, content_version: bytes) -> dict:
    """The ``biopb`` block a sidecar's root ``.zattrs`` carries once finished.

    Merged with the NGFF metadata by whoever writes the store (the label
    upload kind; tests). The upload marker rides in the same block, sealed to
    ``ready`` by ``finish`` -- :func:`sidecar_label_sets` attaches nothing
    else.
    """
    attrs = with_upload_state({}, UPLOAD_READY)
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
        )
        if label_set is not None:
            sets[label_field(image_field, name)] = label_set
    return sets


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
