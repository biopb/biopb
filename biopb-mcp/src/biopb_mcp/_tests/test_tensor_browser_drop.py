"""Drag-and-drop add logic for the tensor-browser widget.

Exercises the pieces that carry real logic without a full widget/napari
construction: the off-thread single-path ``_AddSourceWorker`` result mapping,
the ambient drop-gate predicate, and mime-URL filtering (which refuses
multi-item drops). A live server round-trip through the ``add_source`` action is
covered separately in biopb-tensor-server's ``add_source_test.py``.
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

from biopb_mcp.tensor_browser import _widget
from biopb_mcp.tensor_browser._widget import (
    TensorBrowserWidget,
    _AddSourceWorker,
    _cloud_drop_warning,
    _dir_exceeds_entry_threshold,
)


@pytest.fixture(scope="module")
def _qapp():
    yield QApplication.instance() or QApplication([])


def _result(added=(), already=(), failed=()):
    r = MagicMock()
    r.added = list(added)
    r.already_present = list(already)
    r.failed = [MagicMock(path=p, reason=why) for p, why in failed]
    return r


def test_worker_maps_single_result(_qapp):
    conn = MagicMock()
    conn.add_source.return_value = _result(
        added=[MagicMock(source_id="a")],
        already=["c"],
        failed=[("/p/bad", "not a recognized image format")],
    )
    worker = _AddSourceWorker(conn, "/A")

    captured = {}
    worker.done.connect(
        lambda payload: captured.update(zip(("added", "already", "failed"), payload))
    )
    worker.run()  # synchronous; direct-connected slot fires inline

    conn.add_source.assert_called_once()
    assert conn.add_source.call_args.args[0] == "/A"
    assert [d.source_id for d in captured["added"]] == ["a"]
    assert captured["already"] == ["c"]
    assert captured["failed"] == [("/p/bad", "not a recognized image format")]


def test_worker_surfaces_request_failure(_qapp):
    conn = MagicMock()
    conn.add_source.side_effect = RuntimeError("Path not found on server: /nope")
    worker = _AddSourceWorker(conn, "/nope")
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


def test_local_paths_from_mime_accepts_single_local_only(_qapp):
    from_mime = TensorBrowserWidget._local_paths_from_mime

    single = QMimeData()
    single.setUrls([QUrl.fromLocalFile("/data/a.zarr")])
    assert from_mime(single) == ["/data/a.zarr"]

    # A multi-item drop is refused at the source (returns []): the add pipeline
    # is one path per drop, so a multi-select drag shows "no drop".
    multi = QMimeData()
    multi.setUrls([QUrl.fromLocalFile("/data/a.zarr"), QUrl.fromLocalFile("/data/b")])
    assert from_mime(multi) == []

    # A single non-file URL (e.g. a web link) is rejected too.
    remote = QMimeData()
    remote.setUrls([QUrl("https://example.com")])
    assert from_mime(remote) == []

    assert from_mime(QMimeData()) == []  # no urls at all


def test_add_progress_label_distinguishes_path_from_status():
    stub = MagicMock()

    def render(count, current_path):
        TensorBrowserWidget._on_add_progress(
            stub, MagicMock(added_count=count, current_path=current_path)
        )
        return stub._drop_hint_label.setText.call_args.args[0]

    # A real absolute path is shown as "scanning {basename}".
    assert (
        render(2, "/data/exp.zarr")
        == "Adding… 2 sources registered (scanning exp.zarr)"
    )
    # The catalog-lock wait heartbeat is a status sentence, not a path: show it
    # verbatim, never run through the "scanning {name}" label (cosmetic fix).
    status = render(0, "waiting for catalog scan to finish")
    assert "scanning" not in status
    assert "(waiting for catalog scan to finish)" in status


def test_dir_exceeds_entry_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(_widget, "_LARGE_DROP_ENTRY_THRESHOLD", 3)

    # A single file is never "large".
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert _dir_exceeds_entry_threshold(str(f)) is False

    # A small folder (under threshold) does not trip it.
    small = tmp_path / "small"
    small.mkdir()
    (small / "one").write_text("x")
    (small / "two").write_text("x")
    assert _dir_exceeds_entry_threshold(str(small)) is False

    # A folder over the threshold does.
    big = tmp_path / "big"
    big.mkdir()
    for i in range(5):
        (big / f"f{i}").write_text("x")
    assert _dir_exceeds_entry_threshold(str(big)) is True

    # A nonexistent path is not a directory -> not large (no raise).
    assert _dir_exceeds_entry_threshold(str(tmp_path / "nope")) is False


def test_confirm_large_drop_prompts_only_when_large(monkeypatch):
    # Small drop: no prompt, proceed silently.
    monkeypatch.setattr(_widget, "_dir_exceeds_entry_threshold", lambda _p: False)
    asked = []
    monkeypatch.setattr(
        _widget.QMessageBox,
        "question",
        lambda *a, **k: asked.append(a) or _widget.QMessageBox.Yes,
    )
    assert TensorBrowserWidget._confirm_large_drop(MagicMock(), "/small") is True
    assert asked == []  # never prompted

    # Large drop: prompt; the user's choice is returned verbatim.
    monkeypatch.setattr(_widget, "_dir_exceeds_entry_threshold", lambda _p: True)
    monkeypatch.setattr(
        _widget.QMessageBox, "question", lambda *a, **k: _widget.QMessageBox.No
    )
    assert TensorBrowserWidget._confirm_large_drop(MagicMock(), "/huge") is False

    monkeypatch.setattr(
        _widget.QMessageBox, "question", lambda *a, **k: _widget.QMessageBox.Yes
    )
    assert TensorBrowserWidget._confirm_large_drop(MagicMock(), "/huge") is True


@pytest.mark.parametrize(
    "path,expect_warn",
    [
        (r"C:\Users\me\OneDrive\Desktop\samples", True),  # under OneDrive
        (r"C:\Users\me\OneDrive - Acme\data\a.zarr", True),  # OneDrive - <Org>
        ("/mnt/c/Users/me/OneDrive/data", True),  # forward-slash variant
        (r"C:\Users\me\Desktop\samples", False),  # not under OneDrive
        ("/data/microscopy/exp.zarr", False),  # plain local
        (r"C:\Onedriver\notcloud", False),  # substring, not a OneDrive component
    ],
)
def test_cloud_drop_warning_detects_onedrive(path, expect_warn):
    warning = _cloud_drop_warning(path)
    if expect_warn:
        assert warning is not None and "OneDrive" in warning
    else:
        assert warning is None


def test_confirm_cloud_drop_prompts_only_under_cloud(monkeypatch):
    # Non-cloud path: no prompt, proceed silently.
    monkeypatch.setattr(_widget, "_cloud_drop_warning", lambda _p: None)
    asked = []
    monkeypatch.setattr(
        _widget.QMessageBox,
        "question",
        lambda *a, **k: asked.append(a) or _widget.QMessageBox.Yes,
    )
    assert TensorBrowserWidget._confirm_cloud_drop(MagicMock(), "/local") is True
    assert asked == []  # never prompted

    # Cloud path: prompt; the user's choice is returned verbatim.
    monkeypatch.setattr(_widget, "_cloud_drop_warning", lambda _p: "heads up")
    monkeypatch.setattr(
        _widget.QMessageBox, "question", lambda *a, **k: _widget.QMessageBox.No
    )
    assert TensorBrowserWidget._confirm_cloud_drop(MagicMock(), "/od") is False

    monkeypatch.setattr(
        _widget.QMessageBox, "question", lambda *a, **k: _widget.QMessageBox.Yes
    )
    assert TensorBrowserWidget._confirm_cloud_drop(MagicMock(), "/od") is True
