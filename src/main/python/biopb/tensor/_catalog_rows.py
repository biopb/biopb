"""The ``sources`` catalog row, decoded two ways.

The row is the only representation of a source that crosses the wire -- the
``catalog`` flight streams them (SQL over DoGet) and ``resolve`` returns the one
it just wrote. This module turns one into a structural view of a source.

:class:`CatalogSource` is that view, and a plain dataclass rather than the
generated ``DataSourceDescriptor`` (biopb/biopb#1032). That message never
crossed the wire and each SDK built it for itself, yet every catalog column a
client wanted cost a ``.proto`` edit and a ``buf generate`` across all bindings.
The "one schema, every language" reason for paying that was already gone:
TypeScript hand-rolls the same shape in ``types.ts``, and the HTTP sidecar
decodes rows to plain dicts. ``is_resolved`` is what made the price concrete --
one boolean, a whole codegen cycle.

:func:`descriptor_from_row` / :func:`descriptors_from_rows` still build the
proto, byte for byte as before, and still answer to their own names: they are
public API. They are deprecated rather than changed, because what they cannot do
is grow -- ``is_resolved`` has no field to land in, and adding one is the cost
this change exists to stop paying.

Only the cheap, structural fields are carried either way: per-tensor
``array_id`` / ``dim_labels`` / ``shape`` / ``dtype`` from the ``tensors``
STRUCT[]. ``chunk_shape`` (the transfer grid), ``pyramid``, ``physical_scale``
and ``metadata_json`` belong to the tensor-bound adapter and are answered by
GetFlightInfo (biopb/biopb#812). The proto carries them *empty* -- present,
testable, permanently meaningless; :class:`CatalogSource` does not carry them at
all, which is the one thing a hand-written struct can say that a generated one
cannot.

``SOURCE_ROW_COLUMNS`` is the shared column contract, so the server selects it
too (``MetadataDatabase.source_row_ipc``) and a client has one decoder.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

#: The columns the row decoders read, as a SELECT list.
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


_DEPRECATION = (
    "biopb.tensor.{name}() is deprecated: DataSourceDescriptor is a generated "
    "message, so a catalog column it has no field for (`is_resolved`) cannot "
    "reach you without a proto change regenerated in every language. Use "
    "biopb.tensor.{replacement}(), which returns a plain CatalogSource "
    "(biopb/biopb#1032)."
)


def _descriptor_from_row(row: Mapping[str, Any]) -> DataSourceDescriptor:
    """The legacy decode, unwarned -- for this package's own deprecated paths."""
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
    # `is_resolved` is dropped here, and that is the point: there is no field to
    # put it in. Callers that need it want `source_from_row`.
    return desc


def descriptor_from_row(row: Mapping[str, Any]) -> DataSourceDescriptor:
    """One ``sources`` row -> ``DataSourceDescriptor``.

    .. deprecated::
        Use :func:`source_from_row`.
    """
    warnings.warn(
        _DEPRECATION.format(name="descriptor_from_row", replacement="source_from_row"),
        DeprecationWarning,
        stacklevel=2,
    )
    return _descriptor_from_row(row)


def descriptors_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> List[DataSourceDescriptor]:
    """``sources`` rows -> ``DataSourceDescriptor``s.

    .. deprecated::
        Use :func:`sources_from_rows`.
    """
    warnings.warn(
        _DEPRECATION.format(
            name="descriptors_from_rows", replacement="sources_from_rows"
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return [_descriptor_from_row(r) for r in rows]


def sql_literal(value: str) -> str:
    """Quote a string for the catalog's SQL surface, which takes no parameters."""
    return "'" + value.replace("'", "''") + "'"
