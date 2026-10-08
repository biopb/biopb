"""What a source hands the catalog when it registers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


@dataclass(frozen=True)
class RegistrationRecord:
    """What a source contributes to its catalog entry at registration.

    ``metadata`` is the ``sources.metadata_json`` payload, already stripped of
    whatever the format moved elsewhere (ROIs it imported, mask bitmaps).
    ``rois`` are the annotations its file carries, by ``array_id``, and ``report``
    is an opaque summary for the log (``summary()``), or ``None``.
    """

    metadata: Mapping[str, Any]
    rois: Mapping[str, List[Any]] = field(default_factory=dict)
    report: Any = None


def metadata_record(metadata: Optional[Mapping[str, Any]]) -> RegistrationRecord:
    """A :class:`RegistrationRecord` of just *metadata*, minus any mask bitmaps.

    For a format whose catalog entry is its metadata dict and nothing else; a
    mask's bitmap is arbitrary binary and never reaches the SQL-queryable column.
    """
    return RegistrationRecord(strip_mask_bindata(metadata or {}))


def strip_mask_bindata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """*metadata* with every ``rois[].union.masks[].bin_data.value`` dropped.

    A mask's bitmap is arbitrary binary -- base64 text on the fast metadata
    path (:func:`~biopb_tensor_server.adapters.ome_tiff._b64_encode_mask_bindata`),
    raw bytes elsewhere -- and can be large; it belongs in the rasterized
    ``@ome`` tensor this module builds, never in the SQL-queryable
    ``sources.metadata_json`` column. Unlike ``rois`` as a whole, this runs
    whether or not ROI *annotations* import ran: a mask is not an
    annotation, so it is not covered by that stripping (``metadata_db.py``),
    and its bitmap must not leak into metadata_json regardless.

    Returns *metadata* unchanged (same object) when there is nothing to
    strip, so a caller can skip the JSON re-dump in the common case of no
    masks at all.
    """
    rois = metadata.get("rois")
    if not isinstance(rois, list) or not rois:
        return metadata
    changed = False
    new_rois = []
    for roi in rois:
        union = roi.get("union") if isinstance(roi, Mapping) else None
        masks = union.get("masks") if isinstance(union, Mapping) else None
        if not masks:
            new_rois.append(roi)
            continue
        new_masks = []
        for mask in masks:
            bin_data = mask.get("bin_data") if isinstance(mask, Mapping) else None
            if isinstance(bin_data, Mapping) and "value" in bin_data:
                mask = {
                    **mask,
                    "bin_data": {k: v for k, v in bin_data.items() if k != "value"},
                }
                changed = True
            new_masks.append(mask)
        new_rois.append({**roi, "union": {**union, "masks": new_masks}})
    if not changed:
        return metadata
    return {**metadata, "rois": new_rois}
