"""The ``sources`` catalog row as a ``DataSourceDescriptor``.

The catalog is the browse surface (the ``catalog`` flight, SQL over DoGet); a
``DataSourceDescriptor`` is the SDK's structural view of one row. Only the
cheap, structural fields are carried: per-tensor ``array_id`` / ``dim_labels``
/ ``shape`` / ``dtype`` from the ``tensors`` STRUCT[]. ``chunk_shape`` (the
transfer grid), ``pyramid`` and ``metadata_json`` belong to the tensor-bound
adapter and are answered by GetFlightInfo (biopb/biopb#812).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

#: The columns :func:`descriptor_from_row` reads, as a SELECT list.
SOURCE_ROW_COLUMNS = "source_id, source_url, source_type, data_resident, tensors"


def descriptor_from_row(row: Mapping[str, Any]) -> DataSourceDescriptor:
    """One ``sources`` row (as ``pa.Table.to_pylist()`` yields it) -> descriptor."""
    tensors = [
        TensorDescriptor(
            array_id=t["array_id"],
            dim_labels=t.get("dim_labels") or [],
            shape=t.get("shape") or [],
            dtype=t.get("dtype") or "",
        )
        for t in (row.get("tensors") or [])
    ]
    desc = DataSourceDescriptor(
        source_id=row["source_id"],
        source_url=row.get("source_url") or "",
        source_type=row.get("source_type") or "",
        tensors=tensors,
        metadata_json="",
    )
    resident = row.get("data_resident")
    if resident is not None:
        desc.data_resident = bool(resident)
    return desc


def descriptors_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> List[DataSourceDescriptor]:
    return [descriptor_from_row(r) for r in rows]


def sql_literal(value: str) -> str:
    """Quote a string for the catalog's SQL surface, which takes no parameters."""
    return "'" + value.replace("'", "''") + "'"
