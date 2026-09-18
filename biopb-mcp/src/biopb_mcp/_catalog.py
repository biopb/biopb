"""This package's view of a ``sources`` catalog row.

The SDK hands back rows and deliberately picks no structure for them
(biopb/biopb#1032): ``query_sources`` answers in whichever format you ask for,
``resolve`` returns the one row it just wrote, and choosing what to decode them
into is the caller's job. This module is that choice, made once for biopb-mcp.

Attributes rather than dict keys because the consumer is a Qt widget: the tree
builder, the residency badge and the resolve/warm branches read these fields
across a few hundred lines, and ``src.source_url`` survives a typo where
``src["source_url"]`` does not. Frozen because :attr:`TensorConnection.sources`
is rebound wholesale from a watcher thread and read on the Qt main thread -- the
snapshot is shared, so nothing may mutate an entry in place.

Only the structural fields are carried, because that is all a row has: the
transfer grid, pyramid, physical scale and source metadata are answered by
GetFlightInfo against a bound tensor (biopb/biopb#812), never by a listing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class CatalogTensor:
    """One entry of a row's ``tensors`` STRUCT[]: a tensor's identity and shape.

    Enough to enumerate a source's tensors and address one; not enough to plan a
    read. Describe the tensor for the transfer grid and the pyramid.
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
    #: Monotonic -- false to true once, never back -- which is what makes it safe
    #: to read off a stored row.
    #:
    #: An unresolved source lists with an empty :attr:`tensors`, and that
    #: emptiness used to be the only signal a client had; it conflates "never
    #: resolved" with "resolved, and there was nothing readable in it"
    #: (biopb/biopb#1032).
    is_resolved: bool = True


def source_from_row(row: Mapping[str, Any]) -> CatalogSource:
    """One ``sources`` row (as ``query_sources(format="records")`` yields it)."""
    tensors = tuple(
        CatalogTensor(
            array_id=t["array_id"],
            dim_labels=tuple(t.get("dim_labels") or ()),
            shape=tuple(t.get("shape") or ()),
            dtype=t.get("dtype") or "",
        )
        for t in (row.get("tensors") or ())
    )
    return CatalogSource(
        source_id=row["source_id"],
        source_url=row.get("source_url") or "",
        source_type=row.get("source_type") or "",
        tensors=tensors,
        # True for an absent column, the harmless direction for a monotonic flag:
        # it costs a resolve the UI does not offer, never a browse that silently
        # treats a real source as a placeholder.
        is_resolved=bool(row.get("is_resolved", True)),
    )


def sources_from_rows(rows: Iterable[Mapping[str, Any]]) -> List[CatalogSource]:
    """Decode a listing's rows.

    Residency is deliberately not among them. It is a live filesystem check,
    and asking it per row made a browse cost a stat walk per source
    (biopb/biopb#1048); it is a per-source read on GetFlightInfo now, for a
    caller opening one source rather than listing many.
    """
    return [source_from_row(r) for r in rows]
