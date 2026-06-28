"""Tests for the TensorBrowserWidget connect flow.

The widget delegates connecting to ``TensorConnection.auto_connect`` — the
single, GUI-independent connect policy shared with the headless kernel — run on
a worker thread, then renders the outcome on the Qt main thread via the
``_connect_done`` signal (no modal prompt; the old blocking autostart dialog is
gone). These tests drive that flow deterministically: the worker thread is
*captured* rather than really spawned, so the test runs it explicitly and can
assert both the in-flight ("Connecting…") and completed states. The connection
is a mock whose ``auto_connect`` sets the post-connect state the render reads.

A real ``napari`` viewer (and thus a Qt/OpenGL context) is required, so the
suite is skipped on macOS CI like the other viewer tests.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("CI") == "true",
    reason="OpenGL context unavailable on macOS CI headless environment",
)


@pytest.fixture
def widget(make_napari_viewer, monkeypatch):
    from qtpy.QtCore import QTimer

    from biopb_mcp.tensor_browser import _widget as widget_mod
    from biopb_mcp.tensor_browser._widget import TensorBrowserWidget

    viewer = make_napari_viewer(show=False)
    conn = MagicMock()
    conn.url = "grpc://localhost:8815"
    conn.token = None
    conn.use_server_query = False
    # Default outcome: a connect that resolved to "not connected" (down). Tests
    # that exercise a successful connect give auto_connect a side effect that
    # flips these to the connected state the render reads.
    conn.is_connected = False
    conn.sources = {}
    conn.last_message = ""

    # Capture connect workers instead of spawning real threads so the tests run
    # them explicitly (and can assert the in-flight state before completion).
    workers = []

    class _FakeThread:
        def __init__(self, target=None, name=None, daemon=None):
            self._target = target

        def start(self):
            workers.append(self._target)

    monkeypatch.setattr(widget_mod.threading, "Thread", _FakeThread)
    # Neutralize the auto-connect-on-construction tick — the tests drive connect
    # explicitly for determinism.
    monkeypatch.setattr(QTimer, "singleShot", lambda *a, **k: None)

    w = TensorBrowserWidget(viewer, connection=conn)
    # Isolate the render from tree building (which needs real descriptors).
    w._build_and_display_tree = MagicMock()
    return w, conn, workers


def _connected_with(conn, sources, *, use_server_query=False):
    """Make ``conn.auto_connect`` resolve to a connected state."""

    def _side_effect():
        conn.is_connected = True
        conn.sources = sources
        conn.use_server_query = use_server_query

    conn.auto_connect.side_effect = _side_effect


class TestConnect:
    def test_shows_connecting_then_builds_tree(self, widget):
        w, conn, workers = widget
        _connected_with(conn, {"a": object()})

        w._auto_connect()

        # In flight: status shown, button disabled, worker captured not yet run.
        assert w._connecting is True
        assert not w._connect_button.isEnabled()
        assert "Connecting" in w._status_label.text()
        assert len(workers) == 1
        w._build_and_display_tree.assert_not_called()

        workers.pop(0)()  # run the worker -> auto_connect + render

        conn.auto_connect.assert_called_once()
        w._build_and_display_tree.assert_called_once()
        assert w._refresh_button.isEnabled()
        assert w._status_label.isHidden()
        assert w._connect_button.isEnabled()
        assert w._connecting is False

    def test_down_shows_error_no_prompt(self, widget):
        w, conn, workers = widget
        # auto_connect tried, failed, recorded the friendly reason. There is no
        # dialog anymore — the failure just renders inline.
        conn.last_message = (
            "Cannot reach tensor server at grpc://localhost:8815 — is it running?"
        )

        w._auto_connect()
        workers.pop(0)()

        conn.auto_connect.assert_called_once()
        assert not w._error_label.isHidden()
        assert "Cannot reach" in w._error_label.text()
        assert not w._refresh_button.isEnabled()
        assert w._connecting is False
        assert w._connect_button.isEnabled()

    def test_down_uses_generic_message_without_last_message(self, widget):
        w, conn, workers = widget
        conn.last_message = ""  # nothing recorded -> generic fallback

        w._auto_connect()
        workers.pop(0)()

        assert "Cannot reach tensor server" in w._error_label.text()

    def test_empty_catalog_shows_error(self, widget):
        w, conn, workers = widget
        _connected_with(conn, {})  # connected, but no sources

        w._auto_connect()
        workers.pop(0)()

        assert "No sources found" in w._error_label.text()
        assert not w._refresh_button.isEnabled()
        w._build_and_display_tree.assert_not_called()

    def test_large_catalog_enables_sql_filter(self, widget):
        w, conn, workers = widget
        _connected_with(conn, {"a": object()}, use_server_query=True)

        w._auto_connect()
        workers.pop(0)()

        w._build_and_display_tree.assert_called_once()
        assert w._refresh_button.isEnabled()
        assert "SQL filter" in w._filter_input.placeholderText()

    def test_connect_button_retargets_to_typed_url(self, widget):
        w, conn, workers = widget
        w._server_input.setText("grpc://other:9")
        w._token_input.setText("secret")

        w._on_connect_clicked()

        # The typed URL/token are pushed onto the connection before connecting.
        assert conn.url == "grpc://other:9"
        assert conn.token == "secret"
        assert len(workers) == 1  # a connect worker was started

    def test_stale_generation_is_dropped(self, widget):
        w, conn, workers = widget
        _connected_with(conn, {"a": object()})

        w._auto_connect()  # gen 1, worker captured
        stale_worker = workers.pop(0)
        w._auto_connect()  # gen 2 supersedes; new worker captured
        workers.pop(0)()  # gen 2 completes and renders
        w._build_and_display_tree.assert_called_once()

        # The superseded (gen 1) worker finishing now must NOT re-render.
        w._build_and_display_tree.reset_mock()
        stale_worker()
        w._build_and_display_tree.assert_not_called()

    def test_worker_signals_completion_even_if_auto_connect_raises(self, widget):
        w, conn, workers = widget
        # auto_connect is documented best-effort, but the worker must still
        # signal completion (and not die) if it ever leaks an exception.
        conn.auto_connect.side_effect = RuntimeError("boom")
        conn.is_connected = False

        w._auto_connect()
        workers.pop(0)()  # must not raise

        assert w._connecting is False
        assert w._connect_button.isEnabled()
        assert not w._error_label.isHidden()


class TestSourcesChangedGuard:
    """The background source watcher's re-render is suppressed mid-connect."""

    def test_skipped_while_connecting(self, widget):
        w, conn, workers = widget
        conn.is_connected = True
        w._connecting = True
        w._apply_filter = MagicMock()

        w._on_sources_changed({"a": object()})

        # A connect in flight owns the repaint; don't fight it.
        w._apply_filter.assert_not_called()

    def test_renders_when_idle(self, widget):
        w, conn, workers = widget
        conn.is_connected = True
        w._connecting = False
        w._apply_filter = MagicMock()

        w._on_sources_changed({"a": object()})

        w._apply_filter.assert_called_once()

    def test_skipped_when_disconnected(self, widget):
        w, conn, workers = widget
        conn.is_connected = False
        w._connecting = False
        w._apply_filter = MagicMock()

        w._on_sources_changed({"a": object()})

        w._apply_filter.assert_not_called()


class TestResidencyIndicator:
    """`_add_tree_node` decorates a source row from its `data_resident` state."""

    def _node(self, data_resident):
        from biopb.tensor.descriptor_pb2 import (
            DataSourceDescriptor,
            TensorDescriptor,
        )

        from biopb_mcp.tensor_browser._widget import _TreeNode

        src = DataSourceDescriptor(
            source_id="s",
            source_url="/data/s.zarr",
            tensors=[TensorDescriptor(array_id="s", shape=[10, 10], dtype="uint8")],
        )
        if data_resident is not None:
            src.data_resident = data_resident
        return _TreeNode(
            node_id="s", name="s.zarr", node_type="source", depth=0, source=src
        )

    def test_remote_source_marked_and_greyed(self, widget):
        from biopb_mcp.tensor_browser._widget import _RESIDENCY_GLYPH

        w, _, _ = widget
        w._add_tree_node(w._tree_widget, self._node(False))
        item = w._tree_widget.topLevelItem(0)
        assert item.text(0).startswith(_RESIDENCY_GLYPH)
        assert "Not resident" in item.toolTip(0)
        assert item.foreground(0).color().name() == "#888888"

    def test_resident_source_unmarked_with_tooltip(self, widget):
        from biopb_mcp.tensor_browser._widget import _RESIDENCY_GLYPH

        w, _, _ = widget
        w._add_tree_node(w._tree_widget, self._node(True))
        item = w._tree_widget.topLevelItem(0)
        assert _RESIDENCY_GLYPH not in item.text(0)
        assert "Resident" in item.toolTip(0)

    def test_unknown_residency_has_no_indicator(self, widget):
        from biopb_mcp.tensor_browser._widget import _RESIDENCY_GLYPH

        w, _, _ = widget
        w._add_tree_node(w._tree_widget, self._node(None))
        item = w._tree_widget.topLevelItem(0)
        assert _RESIDENCY_GLYPH not in item.text(0)
        assert item.toolTip(0) == ""


class TestRestoreSelection:
    """`_restore_selection` re-highlights the tracked row in a rebuilt tree (#191)."""

    def _source_node(self, source_id, tensors):
        from biopb.tensor.descriptor_pb2 import (
            DataSourceDescriptor,
            TensorDescriptor,
        )

        from biopb_mcp.tensor_browser._widget import _TreeNode

        src = DataSourceDescriptor(
            source_id=source_id,
            source_url=f"/data/{source_id}.zarr",
            tensors=[
                TensorDescriptor(array_id=tid, shape=[8, 8], dtype="uint8")
                for tid in tensors
            ],
        )
        return _TreeNode(
            node_id=source_id,
            name=f"{source_id}.zarr",
            node_type="source",
            depth=0,
            source=src,
        )

    def _build(self, w, *source_nodes):
        w._tree_widget.clear()
        for node in source_nodes:
            w._add_tree_node(w._tree_widget, node)

    def test_reselects_source_node_after_rebuild(self, widget):
        w, _, _ = widget
        self._build(
            w,
            self._source_node("a", ["a"]),
            self._source_node("b", ["b"]),
        )
        # No current item right after a rebuild.
        assert w._tree_widget.currentItem() is None

        w._selected_source_id = "b"
        w._selected_tensor_id = None
        w._restore_selection()

        current = w._tree_widget.currentItem()
        assert current is not None
        assert current.data(0, _user_role()) == "b"

    def test_reselects_tensor_child_when_field_selected(self, widget):
        from qtpy.QtCore import Qt

        w, _, _ = widget
        self._build(w, self._source_node("multi", ["multi/f0", "multi/f1"]))

        w._selected_source_id = "multi"
        w._selected_tensor_id = "multi/f1"
        w._restore_selection()

        current = w._tree_widget.currentItem()
        assert current is not None
        assert current.data(0, Qt.ItemDataRole.UserRole + 1) == "tensor"
        assert current.data(0, _user_role()) == "multi/f1"

    def test_no_selection_leaves_current_unset(self, widget):
        w, _, _ = widget
        self._build(w, self._source_node("a", ["a"]))
        w._selected_source_id = None
        w._restore_selection()
        assert w._tree_widget.currentItem() is None


def _user_role():
    from qtpy.QtCore import Qt

    return Qt.ItemDataRole.UserRole


def _descriptor(source_id, *, tensors, source_type=""):
    from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

    return DataSourceDescriptor(
        source_id=source_id,
        source_url=f"/cloud/{source_id}.zarr",
        source_type=source_type,
        tensors=[
            TensorDescriptor(array_id=tid, shape=[8, 8], dtype="uint8")
            for tid in tensors
        ],
    )


class _Sig:
    """Minimal stand-in for a Qt signal: connect()/emit() on the calling thread."""

    def __init__(self):
        self._cbs = []

    def connect(self, cb):
        self._cbs.append(cb)

    def emit(self, *args):
        for cb in self._cbs:
            cb(*args)


class _FakeProgress:
    """Non-blocking QProgressDialog stand-in (exec_ returns immediately)."""

    def __init__(self, *a, **k):
        self.closed = False
        self.label = ""
        self.canceled = _Sig()  # user Cancel button -> request_cancel

    def setWindowTitle(self, *a):
        pass

    def setLabelText(self, text):
        self.label = text

    setWindowModality = setMinimumDuration = setCancelButton = setValue = (
        setAutoClose
    ) = setAutoReset = lambda self, *a: None

    def close(self):
        self.closed = True

    def exec_(self):
        pass

    def show(self):  # non-modal path (warm/hydrate)
        pass


class TestUnresolvedHelper:
    def test_empty_tensors_is_unresolved(self):
        from biopb_mcp.tensor_browser._widget import _is_unresolved

        assert _is_unresolved(_descriptor("c", tensors=[]))
        assert not _is_unresolved(_descriptor("c", tensors=["c"]))
        assert not _is_unresolved(_descriptor("c", tensors=["c/a", "c/b"]))


class TestResolveAction:
    """Double-click / context-menu on an unresolved source resolves it off-thread."""

    def _arm(self, widget, monkeypatch, *, accept, outcome):
        """Wire a widget so _resolve_source runs without Qt threads/modals.

        ``accept`` chooses the warning-dialog answer; ``outcome`` is the fake
        worker's result: ("resolved", desc) or ("failed", message).
        """
        from biopb_mcp.tensor_browser import _widget as widget_mod

        w, conn, _ = widget
        conn.is_connected = True
        conn.sources = {"cloud_x": _descriptor("cloud_x", tensors=[])}

        answer = widget_mod.QMessageBox.Ok if accept else widget_mod.QMessageBox.Cancel
        monkeypatch.setattr(
            widget_mod.QMessageBox, "warning", staticmethod(lambda *a, **k: answer)
        )
        monkeypatch.setattr(widget_mod, "QProgressDialog", _FakeProgress)

        started = {"n": 0}

        class _FakeWorker:
            def __init__(self, conn_, source_id):
                self.resolved = _Sig()
                self.failed = _Sig()
                self.finished = _Sig()
                self.cancelled = _Sig()
                self.progress = _Sig()

            def request_cancel(self):
                pass

            def start(self):
                started["n"] += 1
                kind, payload = outcome
                sig = getattr(self, kind)
                sig.emit(payload) if payload is not None else sig.emit()

            def deleteLater(self):
                pass

        monkeypatch.setattr(widget_mod, "_ResolveWorker", _FakeWorker)
        w._apply_filter = MagicMock()
        w._show_error = MagicMock()
        return w, started

    def test_declined_warning_does_nothing(self, widget, monkeypatch):
        w, started = self._arm(
            widget, monkeypatch, accept=False, outcome=("resolved", object())
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 0  # no worker spawned
        w._apply_filter.assert_not_called()

    def test_accepted_resolves_then_repopulates(self, widget, monkeypatch):
        # A plain (non-multifile) resolved descriptor: repopulate, no hydrate offer.
        w, started = self._arm(
            widget,
            monkeypatch,
            accept=True,
            outcome=("resolved", _descriptor("cloud_x", tensors=["cloud_x"])),
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 1  # resolve ran off-thread
        w._apply_filter.assert_called_once()  # tree repopulated from fresh catalog
        w._show_error.assert_not_called()
        # The resolved source is pinned as the selection so the rebuild re-selects
        # it and the user doesn't lose track of it (issue #191).
        assert w._selected_source_id == "cloud_x"
        assert w._selected_tensor_id is None

    def test_failure_surfaces_error(self, widget, monkeypatch):
        w, _ = self._arm(
            widget, monkeypatch, accept=True, outcome=("failed", "offline")
        )
        w._resolve_source("cloud_x")
        w._show_error.assert_called_once()
        assert "offline" in w._show_error.call_args[0][0]

    def test_cancelled_closes_quietly(self, widget, monkeypatch):
        # A user-cancelled resolve is not an error: no banner, no repopulate (the
        # server recall finishes + caches, so a later resolve coalesces).
        w, started = self._arm(
            widget, monkeypatch, accept=True, outcome=("cancelled", None)
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 1
        w._show_error.assert_not_called()
        w._apply_filter.assert_not_called()

    def test_overlapping_workers_are_each_retained(self, widget, monkeypatch):
        # Two in-flight workers must not clobber each other's only ref (which
        # would let a still-running QThread be GC'd / destroyed mid-run). We hold
        # them by thread lifetime in a set, discarded on `finished`.
        from biopb_mcp.tensor_browser import _widget as widget_mod

        w, conn, _ = widget
        conn.is_connected = True
        conn.sources = {"cloud_x": _descriptor("cloud_x", tensors=[])}
        monkeypatch.setattr(
            widget_mod.QMessageBox,
            "warning",
            staticmethod(lambda *a, **k: widget_mod.QMessageBox.Ok),
        )
        monkeypatch.setattr(widget_mod, "QProgressDialog", _FakeProgress)

        made = []

        class _PendingWorker:
            """Starts but never emits resolved/failed/finished (stays in flight)."""

            def __init__(self, conn_, source_id):
                self.resolved = _Sig()
                self.failed = _Sig()
                self.finished = _Sig()
                self.cancelled = _Sig()
                self.progress = _Sig()
                made.append(self)

            def request_cancel(self):
                pass

            def start(self):
                pass

            def deleteLater(self):
                pass

        monkeypatch.setattr(widget_mod, "_ResolveWorker", _PendingWorker)

        w._resolve_source("cloud_x")
        w._resolve_source("cloud_x")

        # Both workers are alive (neither dropped); discard happens on `finished`.
        assert len(made) == 2
        assert set(made) == w._resolve_workers
        for worker in made:  # finishing one removes only itself
            worker.finished.emit()
        assert w._resolve_workers == set()

    def test_multifile_resolve_offers_hydrate(self, widget, monkeypatch):
        # Resolving a multi-file (dir-backed) source pops the hydrate-ahead offer;
        # a plain single-file one (test above) does not.
        w, started = self._arm(
            widget,
            monkeypatch,
            accept=True,
            outcome=(
                "resolved",
                _descriptor("cloud_x", tensors=["cloud_x"], source_type="zarr"),
            ),
        )
        w._offer_hydrate = MagicMock()
        w._resolve_source("cloud_x")
        w._offer_hydrate.assert_called_once()

    def test_double_click_routes_unresolved_to_resolve(self, widget, monkeypatch):
        from biopb_mcp.tensor_browser._widget import _TreeNode

        w, conn, _ = widget
        conn.sources = {"cloud_x": _descriptor("cloud_x", tensors=[])}
        w._add_tree_node(
            w._tree_widget,
            _TreeNode(
                node_id="cloud_x",
                name="cloud_x.zarr",
                node_type="source",
                depth=0,
                source=conn.sources["cloud_x"],
            ),
        )
        item = w._tree_widget.topLevelItem(0)
        w._resolve_source = MagicMock()
        w._add_to_viewer = MagicMock()

        w._on_tree_item_double_clicked(item, 0)

        w._resolve_source.assert_called_once_with("cloud_x")
        w._add_to_viewer.assert_not_called()  # unresolved never hits the add path


def _warm_progress(
    files_done=1, files_total=3, bytes_done=10, bytes_total=30, name="c"
):
    from biopb.tensor.descriptor_pb2 import WarmProgress

    return WarmProgress(
        files_total=files_total,
        files_done=files_done,
        bytes_total=bytes_total,
        bytes_done=bytes_done,
        current_name=name,
    )


class TestMultifileHelper:
    def test_multifile_types_detected(self):
        from biopb_mcp.tensor_browser._widget import _is_multifile_source

        assert _is_multifile_source(_descriptor("z", tensors=["z"], source_type="zarr"))
        assert _is_multifile_source(
            _descriptor("o", tensors=["o"], source_type="ome-zarr")
        )
        # single-file / unknown type -> no warm offer
        assert not _is_multifile_source(
            _descriptor("t", tensors=["t"], source_type="ome-tiff")
        )
        assert not _is_multifile_source(_descriptor("p", tensors=["p"]))
        # unresolved (no tensors) is never "multifile" regardless of type
        assert not _is_multifile_source(
            _descriptor("u", tensors=[], source_type="zarr")
        )


class TestHydrateAction:
    """`_warm_source` runs hydrate-ahead off-thread behind a NON-modal dialog."""

    def _arm(self, widget, monkeypatch, *, events):
        from biopb_mcp.tensor_browser import _widget as widget_mod

        w, conn, _ = widget
        conn.is_connected = True
        conn.sources = {"m": _descriptor("m", tensors=["m"], source_type="zarr")}

        made_progress = []

        class _RecordingProgress(_FakeProgress):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                made_progress.append(self)

        monkeypatch.setattr(widget_mod, "QProgressDialog", _RecordingProgress)

        made_workers = []

        class _FakeWarmWorker:
            def __init__(self, conn_, source_id):
                self.warmed = _Sig()
                self.failed = _Sig()
                self.cancelled = _Sig()
                self.progress = _Sig()
                self.finished = _Sig()
                self.cancel_requested = False
                made_workers.append(self)

            def request_cancel(self):
                self.cancel_requested = True

            def start(self):
                for kind, payload in events:
                    sig = getattr(self, kind)
                    sig.emit(payload) if payload is not None else sig.emit()

            def deleteLater(self):
                pass

        monkeypatch.setattr(widget_mod, "_WarmWorker", _FakeWarmWorker)
        w._show_error = MagicMock()
        return w, made_progress, made_workers

    def test_progress_updates_label_then_warmed_closes(self, widget, monkeypatch):
        w, progs, _ = self._arm(
            widget,
            monkeypatch,
            events=[("progress", _warm_progress()), ("warmed", object())],
        )
        w._warm_source("m")
        assert progs and progs[0].closed  # dialog closed on completion
        assert "1/3 files" in progs[0].label  # files progress rendered
        w._show_error.assert_not_called()

    def test_failure_surfaces_error(self, widget, monkeypatch):
        w, progs, _ = self._arm(widget, monkeypatch, events=[("failed", "disk full")])
        w._warm_source("m")
        assert progs[0].closed
        w._show_error.assert_called_once()
        assert "disk full" in w._show_error.call_args[0][0]

    def test_cancelled_closes_quietly(self, widget, monkeypatch):
        w, progs, _ = self._arm(widget, monkeypatch, events=[("cancelled", None)])
        w._warm_source("m")
        assert progs[0].closed
        w._show_error.assert_not_called()

    def test_cancel_button_requests_worker_cancel(self, widget, monkeypatch):
        # No events -> worker stays "in flight"; firing the dialog's Cancel asks
        # the worker to stop and shows a Cancelling… state.
        w, progs, workers = self._arm(widget, monkeypatch, events=[])
        w._warm_source("m")
        assert workers and not workers[0].cancel_requested
        progs[0].canceled.emit()
        assert workers[0].cancel_requested
        assert progs[0].label == "Cancelling…"

    def test_offer_accepted_starts_warm(self, widget, monkeypatch):
        from biopb_mcp.tensor_browser import _widget as widget_mod

        w, _conn, _ = widget
        monkeypatch.setattr(
            widget_mod.QMessageBox,
            "question",
            staticmethod(lambda *a, **k: widget_mod.QMessageBox.Yes),
        )
        w._warm_source = MagicMock()
        w._offer_hydrate("m", "m.zarr")
        w._warm_source.assert_called_once_with("m")

    def test_offer_declined_does_nothing(self, widget, monkeypatch):
        from biopb_mcp.tensor_browser import _widget as widget_mod

        w, _conn, _ = widget
        monkeypatch.setattr(
            widget_mod.QMessageBox,
            "question",
            staticmethod(lambda *a, **k: widget_mod.QMessageBox.No),
        )
        w._warm_source = MagicMock()
        w._offer_hydrate("m", "m.zarr")
        w._warm_source.assert_not_called()
