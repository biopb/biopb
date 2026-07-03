"""Drag-and-drop add logic for the tensor-browser widget.

Exercises the pieces that carry real logic without a full widget/napari
construction: the off-thread ``_AddSourceWorker`` tally aggregation across
several dropped paths, the ambient drop-gate predicate, and mime-URL filtering.
A live server round-trip through the ``add_source`` action is covered separately
in biopb-tensor-server's ``add_source_test.py``.
"""

from unittest.mock import MagicMock

import pytest

# A real Qt binding is required (QThread signals, QMimeData). Skip cleanly if the
# GUI stack is not installed in this environment.
pytest.importorskip("qtpy")
try:
    from qtpy.QtCore import QMimeData, QUrl
    from qtpy.QtWidgets import QApplication
except Exception:  # pragma: no cover - no Qt platform
    pytest.skip("Qt binding unavailable", allow_module_level=True)

from biopb_mcp.tensor_browser._widget import TensorBrowserWidget, _AddSourceWorker


@pytest.fixture(scope="module")
def _qapp():
    yield QApplication.instance() or QApplication([])


def _result(added=(), already=(), failed=(), needs=False):
    r = MagicMock()
    r.added = list(added)
    r.already_present = list(already)
    r.failed = [MagicMock(path=p, reason=why) for p, why in failed]
    r.needs_confirm_large = needs
    return r


def test_worker_aggregates_tallies_across_paths(_qapp):
    conn = MagicMock()
    conn.add_source.side_effect = [
        _result(added=[MagicMock(source_id="a")]),
        _result(needs=True),  # path B flagged large -> collected for confirm
        _result(already=["c"], failed=[("/p/bad", "not a recognized image format")]),
    ]
    worker = _AddSourceWorker(conn, ["/A", "/B", "/C"])

    captured = {}
    worker.done.connect(
        lambda payload: captured.update(
            zip(("added", "already", "failed", "needs"), payload)
        )
    )
    worker.run()  # synchronous; direct-connected slot fires inline

    assert [d.source_id for d in captured["added"]] == ["a"]
    assert captured["already"] == ["c"]
    assert captured["failed"] == [("/p/bad", "not a recognized image format")]
    assert captured["needs"] == ["/B"]


def test_worker_surfaces_request_failure(_qapp):
    conn = MagicMock()
    conn.add_source.side_effect = RuntimeError("Path not found on server: /nope")
    worker = _AddSourceWorker(conn, ["/nope"])
    errors = []
    worker.failed.connect(errors.append)

    worker.run()

    assert errors and "Path not found" in errors[0]


class _GateStub:
    """Minimal stand-in exposing just what _can_accept_drop reads."""

    _connecting = False

    def __init__(self, connected, local, adding=False):
        self._conn = MagicMock()
        self._conn.is_connected = connected
        self._conn.is_localhost.return_value = local
        self._add_worker = MagicMock() if adding else None


@pytest.mark.parametrize(
    "connected,local,adding,ok",
    [
        (False, True, False, False),  # not connected
        (True, False, False, False),  # remote server
        (True, True, True, False),  # add already in flight
        (True, True, False, True),  # ready
    ],
)
def test_can_accept_drop_gate(connected, local, adding, ok):
    accept, reason = TensorBrowserWidget._can_accept_drop(
        _GateStub(connected, local, adding)
    )
    assert accept is ok
    assert reason  # a human-readable reason is always present


def test_local_paths_from_mime_rejects_mixed_or_remote(_qapp):
    from_mime = TensorBrowserWidget._local_paths_from_mime

    local = QMimeData()
    local.setUrls([QUrl.fromLocalFile("/data/a.zarr"), QUrl.fromLocalFile("/data/b")])
    assert from_mime(local) == ["/data/a.zarr", "/data/b"]

    # A non-file URL anywhere rejects the whole drop (returns []).
    mixed = QMimeData()
    mixed.setUrls([QUrl.fromLocalFile("/data/a.zarr"), QUrl("https://example.com")])
    assert from_mime(mixed) == []

    assert from_mime(QMimeData()) == []  # no urls at all
