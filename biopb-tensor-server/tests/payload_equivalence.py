"""The one check every adapter's payload must pass.

An adapter rebuilt from what its registration stored (``catalog_payload`` and the
row's metadata) must serve what the parsed adapter serves: the same listing,
descriptors, transfer grid, pyramid, physical scale, metadata, content version and
pixels, and it must get there without opening the file.

``assert_hydrates_equivalently`` is that check. What it stores goes through the same
``MetadataDatabase.sync_source_added`` / ``read_hydration`` path a restore reads, so
the metadata is the row's, not the adapter's.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.core.discovery import SourceClaim
from biopb_tensor_server.serving.metadata_db import (
    CatalogRecord,
    MetadataDatabase,
    NumpyEncoder,
)
from google.protobuf.json_format import MessageToDict


def stored(adapter, source_id: str = "src") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(payload, metadata)`` as a registration stores them and a restore reads them."""
    db = MetadataDatabase()
    try:
        db.sync_roots([("r", "file:///")])
        path = str(adapter.source_url)
        claim = SourceClaim(adapter.source_type, path, source_id, member_paths=[path])
        db.sync_source_added(
            source_id, adapter, CatalogRecord(claim, {path: (1, 2, 3, 4)}, "r", "x")
        )
        hydration = db.read_hydration(source_id)
    finally:
        db.close()
    assert hydration is not None, "the adapter stored no payload"
    return hydration


def hydrate(adapter, source, source_id: str = "src"):
    """The adapter a restart would build for *adapter*'s source, from its row."""
    payload, metadata = stored(adapter, source_id)
    rebuilt = type(adapter).create_from_payload(source, payload, metadata, None)
    assert rebuilt is not None, f"{type(adapter).__name__} has no create_from_payload"
    return rebuilt


def stored_form(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Metadata as the row holds it: through the row's JSON encoder, without ``rois``
    (``sync_source_added`` files those in the annotation store, not the row)."""
    form = json.loads(json.dumps(metadata or {}, cls=NumpyEncoder))
    form.pop("rois", None)
    return form


def _read_all(tensor) -> np.ndarray:
    shape = list(tensor.get_tensor_descriptor().shape)
    return np.asarray(tensor.get_data(ChunkBounds(start=[0] * len(shape), stop=shape)))


def snapshot(adapter, *, read: bool = True) -> Dict[str, Any]:
    """Everything a client sees of a source before it asks for pixels, and the pixels."""
    out: Dict[str, Any] = {
        "metadata": adapter.get_metadata(),
        "content_version": adapter.content_version,
        "is_resolved": adapter.is_resolved(),
        "listing": [MessageToDict(d) for d in adapter.list_tensor_descriptors()],
    }
    for entry in adapter.list_tensor_descriptors():
        tensor = adapter.get_tensor_adapter(entry.array_id)
        facts: Dict[str, Any] = {
            "descriptor": MessageToDict(tensor.get_tensor_descriptor()),
            "transfer_chunk": list(tensor.get_transfer_chunk_size()),
            "scale": tensor._physical_scale(),
            "pyramid": [
                MessageToDict(level)
                for level in (tensor.get_native_pyramid_levels() or [])
            ],
        }
        if read:
            facts["pixels"] = _read_all(tensor)
        out[entry.array_id] = facts
    return out


def _same(a: Any, b: Any, where: str) -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(a, b), f"pixels differ at {where}"
    elif isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"keys differ at {where}: {set(a) ^ set(b)}"
        for key in a:
            _same(a[key], b[key], f"{where}/{key}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"length differs at {where}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def forbid_opens(monkeypatch, *targets: Tuple[Any, str]) -> None:
    """Make opening the file raise, for the window a hydration must not touch it.

    Each target is ``(module_or_class, attribute)``: the reader the adapter would
    open (``(tifffile, "TiffFile")``, ``(nd2, "ND2File")``, ...). A hydration that
    reaches one fails the test.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("hydration opened the file")

    for owner, name in targets:
        monkeypatch.setattr(owner, name, refuse)


def assert_hydrates_equivalently(
    parsed,
    source,
    *,
    monkeypatch=None,
    opens: Iterable[Tuple[Any, str]] = (),
    read: bool = True,
    source_id: str = "src",
    close: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Rebuild *parsed* from its stored row and require the same service.

    *opens* are the file readers that must not be called while the adapter is
    rebuilt (needs *monkeypatch*); they are restored before the snapshots are
    compared, which read the file as any request would. Returns the rebuilt adapter.
    """
    payload, metadata = stored(parsed, source_id)
    cls = type(parsed)
    with _patched(monkeypatch, opens):
        rebuilt = cls.create_from_payload(source, payload, metadata, None)
    assert rebuilt is not None, f"{cls.__name__} has no create_from_payload"
    # What a rebuilt adapter reports as metadata is the row's, so that is what the
    # parsed one is held to; a parse's own second answer can differ (BioIO ids come
    # from a counter) and is not the contract.
    expected = snapshot(parsed, read=read)
    actual = snapshot(rebuilt, read=read)
    expected["metadata"] = stored_form(metadata)
    actual["metadata"] = stored_form(actual["metadata"])
    _same(actual, expected, cls.__name__)
    if close is not None:
        close(rebuilt)
    return rebuilt


class _patched:
    """``forbid_opens`` for the length of a ``with`` block."""

    def __init__(self, monkeypatch, opens):
        self._monkeypatch = monkeypatch
        self._opens = list(opens)

    def __enter__(self):
        if self._opens:
            assert self._monkeypatch is not None, "opens needs the monkeypatch fixture"
            self._context = self._monkeypatch.context()
            self._ctx = self._context.__enter__()
            forbid_opens(self._ctx, *self._opens)

    def __exit__(self, *exc):
        if self._opens:
            self._context.__exit__(*exc)
