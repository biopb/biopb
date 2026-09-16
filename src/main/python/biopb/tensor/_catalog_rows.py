"""The ``sources`` catalog row.

A row is the only representation of a source that crosses the wire -- the
``catalog`` flight streams them (SQL over DoGet) and ``resolve`` returns the one
it just wrote. **The SDK does not impose a structure on it.** ``query_sources``
hands rows back in whichever form you ask for (``records`` / ``arrow`` /
``pandas``), ``resolve`` returns the single row it just wrote in the same
``records`` shape, and what you decode them into is yours: a dict, a DataFrame,
your own class.

That is the point of biopb/biopb#1032. The SDK used to build a
``DataSourceDescriptor`` for you, which meant every catalog column a client
wanted cost a ``.proto`` edit and a ``buf generate`` across every binding -- for
a struct no message carries. ``is_resolved`` made the price concrete: one
boolean, a whole codegen cycle. Replacing it with an SDK-chosen dataclass would
have kept the shape of the mistake, just cheaper; the fix is to stop choosing.

:func:`descriptor_from_row` / :func:`descriptors_from_rows` remain, deprecated,
building the proto byte for byte as before. They are public API, so they keep
working and keep their signatures. What they cannot do is grow: ``is_resolved``
has no field to land in, and there is no replacement decoder to point at because
the row *is* the data structure.

Only the cheap, structural fields are in a row's ``tensors`` STRUCT[]:
``array_id`` / ``dim_labels`` / ``shape`` / ``dtype``. ``chunk_shape`` (the
transfer grid), ``pyramid``, ``physical_scale`` and ``metadata_json`` belong to
the tensor-bound adapter and are answered by GetFlightInfo (biopb/biopb#812).

``SOURCE_ROW_COLUMNS`` is the shared column contract, so the server selects it
too (``MetadataDatabase.source_row_ipc``) and every reader sees one shape.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, List, Mapping

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

#: The columns a ``sources`` row carries, as a SELECT list.
SOURCE_ROW_COLUMNS = "source_id, source_url, source_type, is_resolved, tensors"

_DEPRECATION = (
    "biopb.tensor.{name}() is deprecated. DataSourceDescriptor is a generated "
    "message, so a catalog column it has no field for (`is_resolved`) cannot "
    "reach you without a proto change regenerated in every language. There is "
    "no replacement decoder: read the row directly, or ask query_sources() for "
    "the format you want (records / arrow / pandas) -- the structure is yours "
    "to choose (biopb/biopb#1032)."
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
    # No current server sends `data_resident` -- residency is the `is_resident()`
    # action now (biopb/biopb#1035) -- but an older one does, and this decode
    # still answers it identically against that server.
    resident = row.get("data_resident")
    if resident is not None:
        desc.data_resident = bool(resident)
    # `is_resolved` is dropped here, and that is the point: there is no field to
    # put it in. Read it off the row.
    return desc


def descriptor_from_row(row: Mapping[str, Any]) -> DataSourceDescriptor:
    """One ``sources`` row -> ``DataSourceDescriptor``.

    .. deprecated::
        Use the row. See the module docstring.
    """
    warnings.warn(
        _DEPRECATION.format(name="descriptor_from_row"),
        DeprecationWarning,
        stacklevel=2,
    )
    return _descriptor_from_row(row)


def descriptors_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> List[DataSourceDescriptor]:
    """``sources`` rows -> ``DataSourceDescriptor``s.

    .. deprecated::
        Use the rows. See the module docstring.
    """
    warnings.warn(
        _DEPRECATION.format(name="descriptors_from_rows"),
        DeprecationWarning,
        stacklevel=2,
    )
    return [_descriptor_from_row(r) for r in rows]


def sql_literal(value: str) -> str:
    """Quote a string for the catalog's SQL surface, which takes no parameters."""
    return "'" + value.replace("'", "''") + "'"
