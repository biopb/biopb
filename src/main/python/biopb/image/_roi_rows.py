"""The ROI row schema: how annotations travel on the ``roi`` flight.

One Arrow row per :class:`RoiAnnotation`, the same columns in both
directions -- what the server streams on a DoGet is what it accepts on a
DoPut -- so a client that can read a set can write one back. The geometry is
the ``biopb.image.ROI`` proto as canonical proto3 JSON text, which is also
how the server's catalog stores it, so a browser can hand it on verbatim.

Shared by the SDK and the tensor server (which imports ``biopb``).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pyarrow as pa
from google.protobuf import json_format

from biopb.image.annotation_pb2 import RoiAnnotation

ROI_ROW_SCHEMA = pa.schema(
    [
        ("roi_id", pa.string()),
        ("array_id", pa.string()),
        ("set_name", pa.string()),
        ("label", pa.string()),
        # biopb.image.ROI as proto3 JSON.
        ("geometry", pa.string()),
        # Sparse plane pin, wire axis index -> index on that axis.
        ("plane", pa.map_(pa.uint32(), pa.uint32())),
        ("props_json", pa.string()),
        # Null when the annotation was not drawn against a known version.
        ("drawn_against_version", pa.binary()),
        ("rev", pa.int64()),
        ("created_at_unix_ms", pa.int64()),
        ("updated_at_unix_ms", pa.int64()),
    ]
)

#: The stream of a ``RoiDelete`` put: which ids to remove.
ROI_ID_SCHEMA = pa.schema([("roi_id", pa.string())])


def rois_to_table(
    rois: Iterable[RoiAnnotation], metadata: Optional[dict] = None
) -> pa.Table:
    """Annotations as rows of :data:`ROI_ROW_SCHEMA`.

    ``metadata`` (bytes -> bytes) rides on the table's schema, which Flight
    carries with the stream; the server uses it for ``truncated`` and ``sets``.
    """
    rows = [
        {
            "roi_id": r.roi_id,
            "array_id": r.array_id,
            "set_name": r.set_name,
            "label": r.label,
            "geometry": json_format.MessageToJson(r.roi, indent=0).replace("\n", ""),
            "plane": list(r.plane.items()),
            "props_json": r.props_json,
            "drawn_against_version": (
                r.drawn_against_version if r.HasField("drawn_against_version") else None
            ),
            "rev": r.rev,
            "created_at_unix_ms": r.created_at_unix_ms,
            "updated_at_unix_ms": r.updated_at_unix_ms,
        }
        for r in rois
    ]
    schema = ROI_ROW_SCHEMA.with_metadata(metadata) if metadata else ROI_ROW_SCHEMA
    return pa.Table.from_pylist(rows, schema=schema)


def table_to_rois(table: pa.Table) -> List[RoiAnnotation]:
    """The inverse of :func:`rois_to_table`.

    Raises ``ValueError`` for a stream that is not in the row schema or whose
    geometry is not a ``biopb.image.ROI``, so a caller can report the request
    rather than an internal error.
    """
    missing = set(ROI_ROW_SCHEMA.names) - set(table.schema.names)
    if missing:
        raise ValueError(
            f"ROI rows are missing column(s) {sorted(missing)}; expected the "
            f"ROI row schema {ROI_ROW_SCHEMA.names}"
        )
    out: List[RoiAnnotation] = []
    for row in table.select(ROI_ROW_SCHEMA.names).to_pylist():
        roi = RoiAnnotation(
            roi_id=row["roi_id"] or "",
            array_id=row["array_id"] or "",
            set_name=row["set_name"] or "",
            label=row["label"] or "",
            props_json=row["props_json"] or "",
            rev=row["rev"] or 0,
            created_at_unix_ms=row["created_at_unix_ms"] or 0,
            updated_at_unix_ms=row["updated_at_unix_ms"] or 0,
        )
        try:
            json_format.Parse(row["geometry"] or "{}", roi.roi)
        except json_format.ParseError as exc:
            raise ValueError(
                f"ROI {roi.roi_id or '<new>'}: geometry is not a biopb.image.ROI: {exc}"
            ) from exc
        for axis, index in row["plane"] or []:
            roi.plane[axis] = index
        if row["drawn_against_version"] is not None:
            roi.drawn_against_version = row["drawn_against_version"]
        out.append(roi)
    return out


def roi_ids_to_table(roi_ids: Iterable[str]) -> pa.Table:
    return pa.table(
        {"roi_id": pa.array(list(roi_ids), pa.string())}, schema=ROI_ID_SCHEMA
    )


def table_to_roi_ids(table: pa.Table) -> List[str]:
    if table.num_rows == 0:
        return []
    if "roi_id" not in table.schema.names:
        raise ValueError("a ROI delete stream carries one 'roi_id' column")
    return [v for v in table.column("roi_id").to_pylist() if v]
