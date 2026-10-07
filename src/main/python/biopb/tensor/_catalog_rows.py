"""The ``sources`` catalog row.

A row is the only representation of a source that crosses the wire -- the
``catalog`` flight streams them (SQL over DoGet) and ``resolve`` returns the one
it just wrote. **The SDK does not impose a structure on it.** ``query``
hands rows back in whichever form you ask for (``records`` / ``arrow`` /
``pandas``), ``resolve`` returns the single row it just wrote in the same
``records`` shape, and what you decode them into is yours: a dict, a DataFrame,
your own class.

Only the cheap, structural fields are in a row's ``tensors`` STRUCT[]:
``array_id`` / ``dim_labels`` / ``shape`` / ``dtype``. ``chunk_shape`` (the
transfer grid), ``pyramid``, ``physical_scale`` and ``metadata_json`` belong to
the tensor-bound adapter and are answered by GetFlightInfo (biopb/biopb#812).

``SOURCE_ROW_COLUMNS`` is the shared column contract, so the server selects it
too (``MetadataDatabase.source_row_ipc``) and every reader sees one shape.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from biopb.tensor.descriptor_pb2 import TensorDescriptor

#: The columns a ``sources`` row carries, as a SELECT list.
SOURCE_ROW_COLUMNS = "source_id, source_url, source_type, is_resolved, tensors"


def tensor_descriptors_from_row(row: Mapping[str, Any]) -> List[TensorDescriptor]:
    """A row's ``tensors`` STRUCT[] as ``TensorDescriptor``s.

    The structural fields listed in the module docstring are all a row carries;
    the serving fields stay unset. Not deprecated, unlike the decoders below:
    this targets the message the read path addresses tensors with, not the
    source-shaped struct.
    """
    return [
        TensorDescriptor(
            array_id=t["array_id"],
            dim_labels=t.get("dim_labels") or [],
            shape=t.get("shape") or [],
            dtype=t.get("dtype") or "",
        )
        for t in (row.get("tensors") or [])
    ]


def unresolved_reasons(
    query: Callable[[str], Iterable[Mapping[str, Any]]],
    where: str = "",
    *,
    errors: Any = Exception,
) -> Dict[str, Optional[str]]:
    """``{source_id: unresolved_reason}`` for the catalog rows that are not resolved.

    A query of its own, never a column of a row projection: a server older than
    the column refuses it, and what the caller is doing (a listing, a read) must
    still work. A refusal (*errors*) reads as no reasons, so the caller treats the
    row as it always did, a cloud placeholder. *query* takes SQL and returns rows
    as mappings; *where* narrows it (``"AND source_id = ..."``).
    """
    try:
        rows = query(
            f"SELECT source_id, unresolved_reason FROM sources WHERE NOT is_resolved {where}"
        )
        return {
            row["source_id"]: row.get("unresolved_reason")
            for row in rows
            if "source_id" in row
        }
    except errors:
        return {}


def with_reason(columns: str, have: Iterable[str]) -> str:
    """*columns* with ``unresolved_reason`` appended when the server's ``sources``
    schema (*have*) lists it, so one query carries the reason."""
    return columns + ", unresolved_reason" if "unresolved_reason" in have else columns


def reasons_for(
    rows: Iterable[Mapping[str, Any]],
    query: Callable[[str], Iterable[Mapping[str, Any]]],
    where: str = "",
    *,
    errors: Any = Exception,
) -> Dict[str, Optional[str]]:
    """``{source_id: unresolved_reason}`` for the unresolved *rows*.

    Read off the rows when they were projected with the column
    (:func:`with_reason`); otherwise one :func:`unresolved_reasons` query, and only
    when a row is unresolved. *query*, *where* and *errors* are its own.
    """
    rows = list(rows)
    if any("unresolved_reason" in row for row in rows):
        return {r["source_id"]: r["unresolved_reason"] for r in rows}
    if any(not row.get("is_resolved", True) for row in rows):
        return unresolved_reasons(query, where, errors=errors)
    return {}


def sql_literal(value: str) -> str:
    """Quote a string for the catalog's SQL surface, which takes no parameters."""
    return "'" + value.replace("'", "''") + "'"
