"""The ``sources`` catalog row as the SDK's own struct.

The row is the only representation of a source that crosses the wire -- the
``catalog`` flight streams them (SQL over DoGet) and ``resolve`` returns the one
it just wrote. :class:`CatalogSource` is the SDK's structural view of one row,
built here.

Plain dataclasses, not the generated ``DataSourceDescriptor`` (biopb/biopb#1032).
That message never crossed the wire and each SDK built it for itself, yet every
catalog column a client wanted cost a ``.proto`` edit and a ``buf generate``
across all bindings. The "one schema, every language" reason for paying that was
already gone: TypeScript hand-rolls the same shape in ``types.ts``, and the HTTP
sidecar decodes rows to plain dicts. ``is_resolved`` is what made the price
concrete -- one boolean, a whole codegen cycle.

Only the cheap, structural fields are carried: per-tensor ``array_id`` /
``dim_labels`` / ``shape`` / ``dtype`` from the ``tensors`` STRUCT[].
``chunk_shape`` (the transfer grid), ``pyramid``, ``physical_scale`` and
``metadata_json`` belong to the tensor-bound adapter and are answered by
GetFlightInfo (biopb/biopb#812) -- so they are *absent* here rather than present
and permanently empty, which is the one thing a hand-written struct can say that
a proto message cannot.

``SOURCE_ROW_COLUMNS`` is the shared column contract, so the server selects it
too (``MetadataDatabase.source_row_ipc``) and a client has one decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

#: The columns :func:`source_from_row` reads, as a SELECT list.
SOURCE_ROW_COLUMNS = (
    "source_id, source_url, source_type, data_resident, is_resolved, tensors"
)


@dataclass(frozen=True, slots=True)
class CatalogTensor:
    """One entry of a row's ``tensors`` STRUCT[]: a tensor's identity and shape.

    Enough to enumerate a source's tensors and address one; not enough to plan a
    read. Describe the tensor (GetFlightInfo) for the transfer grid and pyramid.
    """

    array_id: str
    dim_labels: Tuple[str, ...] = ()
    shape: Tuple[int, ...] = ()
    dtype: str = ""


@dataclass(frozen=True, slots=True)
class CatalogSource:
    """One ``sources`` catalog row."""

    source_id: str
    source_url: str = ""
    source_type: str = ""
    tensors: Tuple[CatalogTensor, ...] = ()
    #: Whether the server has hydrated this source enough to know its tensors.
    #: Monotonic -- false to true once, never back -- which is what makes it
    #: safe to read off a stored row. Not to be confused with
    #: :attr:`data_resident`.
    #:
    #: An unresolved source lists with an empty :attr:`tensors`, and that
    #: emptiness used to be the only signal a client had; it conflates "never
    #: resolved" with "resolved, and there was nothing readable in it"
    #: (biopb/biopb#1032).
    is_resolved: bool = True
    #: Advisory, point-in-time: the content is local and cheap to read *right
    #: now*. VOLATILE -- a synced-folder source re-dehydrates under storage
    #: pressure -- so a read path wanting certainty asks the server, not this.
    #: ``None`` when the server did not report it.
    data_resident: Optional[bool] = None


def source_from_row(row: Mapping[str, Any]) -> CatalogSource:
    """One ``sources`` row (as ``pa.Table.to_pylist()`` yields it) -> source."""
    tensors = tuple(
        CatalogTensor(
            array_id=t["array_id"],
            dim_labels=tuple(t.get("dim_labels") or ()),
            shape=tuple(t.get("shape") or ()),
            dtype=t.get("dtype") or "",
        )
        for t in (row.get("tensors") or ())
    )
    resident = row.get("data_resident")
    return CatalogSource(
        source_id=row["source_id"],
        source_url=row.get("source_url") or "",
        source_type=row.get("source_type") or "",
        tensors=tensors,
        # True for an absent column, the harmless direction for a monotonic
        # flag: it costs a resolve the UI does not offer, never a browse that
        # silently treats a real source as a placeholder. A server whose table
        # predates the column fails the SELECT outright, so this only covers a
        # caller's own narrower projection.
        is_resolved=bool(row.get("is_resolved", True)),
        data_resident=None if resident is None else bool(resident),
    )


def sources_from_rows(rows: Iterable[Mapping[str, Any]]) -> List[CatalogSource]:
    return [source_from_row(r) for r in rows]


def sql_literal(value: str) -> str:
    """Quote a string for the catalog's SQL surface, which takes no parameters."""
    return "'" + value.replace("'", "''") + "'"
