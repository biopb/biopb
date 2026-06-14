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
            "Cannot reach tensor server at grpc://localhost:8815 — "
            "is it running?"
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

    def test_worker_signals_completion_even_if_auto_connect_raises(
        self, widget
    ):
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
