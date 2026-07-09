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


class TestGetPathParts:
    """source_url -> tree path parts, incl. remote authority roots (biopb/biopb#297)."""

    @staticmethod
    def _parts(url):
        from biopb_mcp.tensor_browser._widget import _get_path_parts

        return _get_path_parts(url)

    def test_local_file_url_is_just_its_path(self):
        assert self._parts("file:///home/jiyu/data/img.tif") == [
            "home",
            "jiyu",
            "data",
            "img.tif",
        ]

    def test_bare_posix_path(self):
        assert self._parts("/data/cells/img.tif") == ["data", "cells", "img.tif"]

    def test_remote_grpc_endpoint_is_the_root(self):
        assert self._parts("grpc://mantis-060:8815/labs/Yu/exp1/img.ome.tif") == [
            "grpc://mantis-060:8815",
            "labs",
            "Yu",
            "exp1",
            "img.ome.tif",
        ]

    def test_remote_alias_endpoint_root(self):
        assert self._parts("grpc://lab/data/x.tif") == ["grpc://lab", "data", "x.tif"]

    def test_s3_bucket_nests_under_endpoint(self):
        assert self._parts("s3://bucket/key/img.zarr") == [
            "s3://bucket",
            "key",
            "img.zarr",
        ]

    def test_windows_drive_letter_dropped(self):
        assert self._parts("file:///C:/Users/me/img.tif") == ["Users", "me", "img.tif"]

    def test_empty_url(self):
        assert self._parts("") == []

    def test_dnd_single_source_strips_scheme_to_own_root(self):
        # A drag-dropped source's "dnd://" origin scheme is stripped for display,
        # so it renders under a clean top-level root (same as a scheme-less
        # re-root), not a literal "dnd://exp.zarr" node.
        assert self._parts("dnd://exp.zarr") == ["exp.zarr"]

    def test_dnd_folder_children_nest_under_stripped_root(self):
        assert self._parts("dnd://my_experiment/sub/b.tif") == [
            "my_experiment",
            "sub",
            "b.tif",
        ]

    def test_dnd_basename_with_colon_not_misparsed_as_port(self):
        # String-strip (not urlparse) so a basename with a colon can't misparse
        # as a netloc/port.
        assert self._parts("dnd://exp:2.zarr") == ["exp:2.zarr"]


class TestBuildTreeDroppedTagging:
    """_build_tree tags a drag-dropped branch's root node so the UI shows [x]."""

    @staticmethod
    def _src(source_id, source_url):
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

        return DataSourceDescriptor(
            source_id=source_id,
            source_url=source_url,
            tensors=[TensorDescriptor(array_id=source_id, shape=[8, 8], dtype="uint8")],
        )

    @staticmethod
    def _tree(sources):
        from biopb_mcp.tensor_browser._widget import _build_tree

        return _build_tree({s.source_id: s for s in sources})

    def test_single_file_drop_leaf_is_tagged(self):
        root = self._tree([self._src("s1", "dnd://exp.zarr")])
        (leaf,) = root.children
        assert leaf.node_type == "source"
        assert leaf.dropped is True
        assert leaf.remove_root == "dnd://exp.zarr"

    def test_folder_drop_top_folder_is_tagged(self):
        root = self._tree(
            [
                self._src("a", "dnd://my_experiment/a.zarr"),
                self._src("b", "dnd://my_experiment/sub/b.zarr"),
            ]
        )
        (folder,) = root.children
        assert folder.node_type == "folder"
        assert folder.dropped is True
        assert folder.remove_root == "dnd://my_experiment"
        # Children (the actual sources) are NOT individually tagged.
        assert all(not _leaf_dropped(c) for c in folder.children)

    def test_dropped_root_survives_path_flattening(self):
        # A drop whose only content is nested gets its single-child folders
        # flattened; the dropped tag + remove_root must ride along on the merged
        # node (flatten mutates the node in place, preserving the attributes).
        root = self._tree([self._src("x", "dnd://exp/sub/deep/img.zarr")])
        (node,) = root.children
        assert node.dropped is True
        assert node.remove_root == "dnd://exp"
        assert node.name.startswith("exp")  # flattened, e.g. "exp/sub/deep"

    def test_configured_source_is_not_tagged(self):
        root = self._tree([self._src("c", "file:///data/cells/img.zarr")])
        assert all(not n.dropped for n in _walk(root))


def _leaf_dropped(node):
    return getattr(node, "dropped", False)


def _walk(node):
    for child in node.children:
        yield child
        yield from _walk(child)


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
    # Default: server is not mid-scan, so an empty catalog renders as a genuine
    # "no sources" error (progressive-discovery indexing case is opt-in per test).
    conn.scan_in_progress.return_value = False
    conn.scan_source_count.return_value = 0

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


class TestTreeLayoutStability:
    """The tree pins its horizontal scrollbar off so a content-width change on
    an otherwise-unchanged refresh can't toggle the bar and shift rows
    vertically on non-overlay-scrollbar platforms (biopb/biopb#367)."""

    def test_horizontal_scrollbar_pinned_off(self, widget):
        from qtpy.QtCore import Qt

        w, _, _ = widget
        assert w._tree_widget.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


class TestConnect:
    def test_shows_connecting_then_builds_tree(self, widget):
        w, conn, workers = widget
        _connected_with(conn, {"a": object()})

        w._auto_connect()

        # In flight: status shown, button disabled, worker captured not yet run.
        assert w._connecting is True
        assert not w._connect_button.isEnabled()
        assert w._message_level == "busy"  # "Connecting…" is an ongoing state
        assert "Connecting" in w._message_label.text()
        assert len(workers) == 1
        w._build_and_display_tree.assert_not_called()

        workers.pop(0)()  # run the worker -> auto_connect + render

        conn.auto_connect.assert_called_once()
        w._build_and_display_tree.assert_called_once()
        assert w._refresh_button.isEnabled()
        assert w._message_label.isHidden()  # status cleared once connected
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
        assert w._message_level == "error"
        assert not w._message_label.isHidden()
        assert "Cannot reach" in w._message_label.text()
        assert not w._refresh_button.isEnabled()
        assert w._connecting is False
        assert w._connect_button.isEnabled()

    def test_down_uses_generic_message_without_last_message(self, widget):
        w, conn, workers = widget
        conn.last_message = ""  # nothing recorded -> generic fallback

        w._auto_connect()
        workers.pop(0)()

        assert w._message_level == "error"
        assert "Cannot reach tensor server" in w._message_label.text()

    def test_empty_catalog_shows_error(self, widget):
        w, conn, workers = widget
        _connected_with(conn, {})  # connected, but no sources

        w._auto_connect()
        workers.pop(0)()

        assert w._message_level == "error"
        assert "No sources found" in w._message_label.text()
        assert not w._refresh_button.isEnabled()
        w._build_and_display_tree.assert_not_called()

    def test_empty_catalog_while_indexing_shows_status(self, widget):
        # Progressive discovery: SERVING but the scan is still running. An empty
        # catalog is "not done indexing yet", not an error -- show grey status,
        # keep Refresh enabled (more sources are coming), no error label.
        w, conn, workers = widget
        _connected_with(conn, {})
        conn.scan_in_progress.return_value = True
        conn.scan_source_count.return_value = 7

        w._auto_connect()
        workers.pop(0)()

        # One pane, showing a sticky "busy" status -- not an error.
        assert w._message_level == "busy"
        assert not w._message_label.isHidden()
        assert "Indexing" in w._message_label.text()
        assert "7 sources so far" in w._message_label.text()
        assert w._refresh_button.isEnabled()
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


class TestConnectionSummary:
    def test_advanced_panel_collapsed_by_default(self, widget):
        w, _conn, _workers = widget
        # The full connection controls start hidden behind the summary line.
        # (isHidden reflects the explicit visibility flag even when the top-level
        # widget is never shown, as in these headless tests.)
        assert w._advanced_panel.isHidden()
        assert not w._advanced_expanded
        # The collapsed caret is part of the (clickable) summary line.
        assert "▸" in w._status_summary.text()

    def test_clicking_summary_reveals_and_hides_panel(self, widget):
        w, _conn, _workers = widget
        w._toggle_advanced()  # what the summary line's mousePressEvent calls
        assert not w._advanced_panel.isHidden()
        assert w._advanced_expanded
        assert "▾" in w._status_summary.text()
        w._toggle_advanced()
        assert w._advanced_panel.isHidden()
        assert not w._advanced_expanded
        assert "▸" in w._status_summary.text()

    def test_summary_shows_url_and_state_across_lifecycle(self, widget):
        w, conn, workers = widget
        _connected_with(conn, {"a": object()})

        # Before connecting: disconnected.
        assert conn.url in w._status_summary.text()
        assert "disconnected" in w._status_summary.text()

        w._auto_connect()
        assert "connecting" in w._status_summary.text()  # in flight

        workers.pop(0)()  # run worker -> connected
        assert "connected" in w._status_summary.text()
        assert "disconnected" not in w._status_summary.text()

    def test_summary_escapes_url_for_rich_text(self, widget):
        w, conn, _workers = widget
        # The summary is a rich-text QLabel; a URL with markup-significant chars
        # must be escaped, not injected raw.
        conn.url = "grpc://h?a=1&b=<x>"
        w._update_status_summary()
        text = w._status_summary.text()
        assert "&amp;" in text and "&lt;x&gt;" in text
        assert "<x>" not in text

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
        assert w._message_level == "error"
        assert not w._message_label.isHidden()


class TestRefreshFailure:
    def test_failed_refresh_marks_disconnected_and_updates_indicator(self, widget):
        w, conn, _workers = widget
        # Start from a connected state, then make the re-list blow up (server
        # gone). mark_disconnected is what flips is_connected on the real
        # connection; emulate that side effect on the mock so the status line
        # re-render reads the disconnected state.
        conn.is_connected = True

        def _drop(*_a, **_k):
            conn.is_connected = False

        conn.mark_disconnected.side_effect = _drop
        conn.refresh.side_effect = RuntimeError("unreachable")

        w._refresh()

        conn.mark_disconnected.assert_called_once()
        assert not w._refresh_button.isEnabled()
        assert w._message_level == "error"
        assert "lost connection" in w._message_label.text().lower()
        assert "disconnected" in w._status_summary.text()

    def test_render_error_does_not_mark_disconnected(self, widget):
        w, conn, _workers = widget
        # The server answered fine (refresh returned sources), but building the
        # tree blows up -- a client-side bug, not a lost server. The connection
        # must stay up and the indicator must not flip to disconnected; the
        # error is reported without dropping the client (mark_disconnected is
        # scoped to the re-list call, not the render).
        conn.is_connected = True
        conn.refresh.return_value = {"a": object()}
        w._build_and_display_tree.side_effect = RuntimeError("render boom")

        w._refresh()

        conn.mark_disconnected.assert_not_called()
        assert conn.is_connected
        assert w._message_level == "error"
        assert "lost connection" not in w._message_label.text().lower()


class TestMessagePane:
    """The unified bottom pane's level + auto-clear lifecycle (biopb/biopb#312)."""

    def test_info_self_clears_while_busy_and_error_are_sticky(self, widget):
        w, _conn, _workers = widget

        # A one-shot outcome ("added N") is info and arms the auto-clear timer.
        w._show_status("added 3 sources")
        assert w._message_level == "info"
        assert not w._message_label.isHidden()
        assert w._message_timer.isActive()

        # An ongoing state is a sticky "busy" status -- no timer.
        w._show_status("Indexing…", sticky=True)
        assert w._message_level == "busy"
        assert not w._message_timer.isActive()

        # An error is sticky (no timer) and replaces the status.
        w._show_error("boom")
        assert w._message_level == "error"
        assert "boom" in w._message_label.text()
        assert not w._message_timer.isActive()

    def test_clears_are_scoped_to_their_own_level(self, widget):
        w, _conn, _workers = widget

        # _clear_status must not wipe a live error...
        w._show_error("boom")
        w._clear_status()
        assert w._message_level == "error"
        assert not w._message_label.isHidden()

        # ...and _clear_error must not wipe a live status.
        w._show_status("Indexing…", sticky=True)
        w._clear_error()
        assert w._message_level == "busy"
        assert not w._message_label.isHidden()

        # Each clear still works on its own level.
        w._clear_status()
        assert w._message_level is None
        assert w._message_label.isHidden()

    def test_auto_clear_hides_and_resets(self, widget):
        w, _conn, _workers = widget
        w._show_status("added 3 sources")  # info -> timer armed
        assert w._message_timer.isActive()

        w._clear_message()  # what the timer's timeout invokes
        assert w._message_level is None
        assert w._message_label.isHidden()
        assert w._message_label.text() == ""
        assert not w._message_timer.isActive()


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
        # The full label rides in the tooltip too, so an elided long name is
        # still readable on hover (biopb/biopb#367).
        assert item.text(0) in item.toolTip(0)
        assert item.foreground(0).color().name() == "#888888"

    def test_resident_source_unmarked_with_tooltip(self, widget):
        from biopb_mcp.tensor_browser._widget import _RESIDENCY_GLYPH

        w, _, _ = widget
        w._add_tree_node(w._tree_widget, self._node(True))
        item = w._tree_widget.topLevelItem(0)
        assert _RESIDENCY_GLYPH not in item.text(0)
        assert "Resident" in item.toolTip(0)
        assert item.text(0) in item.toolTip(0)

    def test_unknown_residency_has_no_indicator(self, widget):
        from biopb_mcp.tensor_browser._widget import _RESIDENCY_GLYPH

        w, _, _ = widget
        w._add_tree_node(w._tree_widget, self._node(None))
        item = w._tree_widget.topLevelItem(0)
        assert _RESIDENCY_GLYPH not in item.text(0)
        # The row still carries a full-label tooltip (biopb/biopb#367), but with
        # no residency note appended for an unknown state.
        assert item.toolTip(0) == item.text(0)
        assert "resident" not in item.toolTip(0).lower()


class TestRemoveButton:
    """`_add_tree_node` puts a remove [x] in column 1 for dropped roots only."""

    def _node(self, *, dropped, source_url="dnd://exp.zarr", name="exp.zarr"):
        from biopb.tensor.descriptor_pb2 import (
            DataSourceDescriptor,
            TensorDescriptor,
        )

        from biopb_mcp.tensor_browser._widget import _TreeNode

        src = DataSourceDescriptor(
            source_id="s",
            source_url=source_url,
            tensors=[TensorDescriptor(array_id="s", shape=[10, 10], dtype="uint8")],
        )
        node = _TreeNode(
            node_id="s", name=name, node_type="source", depth=0, source=src
        )
        if dropped:
            node.dropped = True
            node.remove_root = source_url
        return node

    def test_dropped_root_gets_remove_button(self, widget):
        from qtpy.QtWidgets import QPushButton

        w, _, _ = widget
        w._add_tree_node(w._tree_widget, self._node(dropped=True))
        item = w._tree_widget.topLevelItem(0)
        assert isinstance(w._tree_widget.itemWidget(item, 1), QPushButton)

    def test_non_dropped_row_has_no_button(self, widget):
        w, _, _ = widget
        w._add_tree_node(
            w._tree_widget, self._node(dropped=False, source_url="/data/c.zarr")
        )
        item = w._tree_widget.topLevelItem(0)
        assert w._tree_widget.itemWidget(item, 1) is None

    def test_button_click_routes_to_remove_with_branch_root(self, widget, monkeypatch):
        w, _, _ = widget
        called = {}
        monkeypatch.setattr(
            w, "_on_remove_dropped", lambda r, n: called.update(root=r, name=n)
        )
        w._add_tree_node(w._tree_widget, self._node(dropped=True))
        item = w._tree_widget.topLevelItem(0)
        w._tree_widget.itemWidget(item, 1).click()
        assert called == {"root": "dnd://exp.zarr", "name": "exp.zarr"}

    def test_confirm_yes_starts_remove(self, widget, monkeypatch):
        from biopb_mcp.tensor_browser import _widget as m

        w, _, _ = widget
        monkeypatch.setattr(
            m.QMessageBox, "question", lambda *a, **k: m.QMessageBox.Yes
        )
        started = {}
        monkeypatch.setattr(
            w, "_start_remove", lambda r, n: started.update(root=r, name=n)
        )
        w._on_remove_dropped("dnd://exp", "exp")
        assert started == {"root": "dnd://exp", "name": "exp"}

    def test_confirm_no_does_not_remove(self, widget, monkeypatch):
        from biopb_mcp.tensor_browser import _widget as m

        w, _, _ = widget
        monkeypatch.setattr(m.QMessageBox, "question", lambda *a, **k: m.QMessageBox.No)
        w._start_remove = MagicMock()
        w._on_remove_dropped("dnd://exp", "exp")
        w._start_remove.assert_not_called()


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
        self.shown = False
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
        self.shown = True


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
        w._report_failure = MagicMock()
        return w, started

    def test_declined_warning_does_nothing(self, widget, monkeypatch):
        w, started = self._arm(
            widget, monkeypatch, accept=False, outcome=("resolved", object())
        )
        w._resolve_source("cloud_x")
        assert started["n"] == 0  # no worker spawned
        w._apply_filter.assert_not_called()

    def test_accepted_resolves_then_repopulates(self, widget, monkeypatch):
        # A plain (non-multifile) resolved descriptor: repopulate, no warm started.
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
        # A failed user-initiated resolve reports via a modal box, not the
        # easily-missed inline pane (issue #206).
        w, _ = self._arm(
            widget, monkeypatch, accept=True, outcome=("failed", "offline")
        )
        w._resolve_source("cloud_x")
        w._report_failure.assert_called_once()
        assert "offline" in w._report_failure.call_args[0][1]
        w._show_error.assert_not_called()

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

    def test_multifile_resolve_starts_warm(self, widget, monkeypatch):
        # Resolving a multi-file (dir-backed) source starts hydrate-ahead directly
        # -- no confirmation dialog (biopb/biopb#202); a plain single-file one
        # (test above) does not.
        w, started = self._arm(
            widget,
            monkeypatch,
            accept=True,
            outcome=(
                "resolved",
                _descriptor("cloud_x", tensors=["cloud_x"], source_type="zarr"),
            ),
        )
        w._warm_source = MagicMock()
        w._resolve_source("cloud_x")
        w._warm_source.assert_called_once_with("cloud_x")

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
    """`_warm_source` hydrates off-thread and paints progress as an inline bar on
    the source's tree row (biopb/biopb#202); cancel is a context-menu action."""

    def _add_row(self, w, src):
        from biopb_mcp.tensor_browser._widget import _TreeNode

        w._tree_widget.clear()
        w._add_tree_node(
            w._tree_widget,
            _TreeNode(
                node_id="m",
                name="m.zarr",
                node_type="source",
                depth=0,
                source=src,
            ),
        )

    def _arm(self, widget, monkeypatch, *, events):
        from biopb_mcp.tensor_browser import _widget as widget_mod

        w, conn, _ = widget
        conn.is_connected = True
        src = _descriptor("m", tensors=["m"], source_type="zarr")
        conn.sources = {"m": src}
        # A real tree row so the inline indicator can be read back off the item.
        self._add_row(w, src)

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
        w._report_failure = MagicMock()
        return w, made_workers

    def _row_fraction(self, w):
        from biopb_mcp.tensor_browser._widget import _WARM_ROLE

        item = w._find_source_item("m")
        return item.data(0, _WARM_ROLE)

    def test_warm_paints_indeterminate_bar_until_progress(self, widget, monkeypatch):
        from biopb_mcp.tensor_browser._widget import _WARM_INDETERMINATE

        w, _ = self._arm(widget, monkeypatch, events=[])  # stays in flight
        w._warm_source("m")
        assert "m" in w._warms
        assert w._warms["m"].fraction == _WARM_INDETERMINATE
        assert self._row_fraction(w) == _WARM_INDETERMINATE  # bar on the row
        w._show_error.assert_not_called()

    def test_progress_fills_bar_by_bytes(self, widget, monkeypatch):
        # bytes_done/bytes_total drives the fraction when byte counts are known.
        w, _ = self._arm(
            widget,
            monkeypatch,
            events=[("progress", _warm_progress(bytes_done=10, bytes_total=30))],
        )
        w._warm_source("m")
        assert w._warms["m"].fraction == 10 / 30
        assert self._row_fraction(w) == 10 / 30

    def test_progress_falls_back_to_files_without_bytes(self, widget, monkeypatch):
        w, _ = self._arm(
            widget,
            monkeypatch,
            events=[
                ("progress", _warm_progress(files_done=1, files_total=4, bytes_total=0))
            ],
        )
        w._warm_source("m")
        assert w._warms["m"].fraction == 1 / 4

    def test_warmed_clears_bar_and_state(self, widget, monkeypatch):
        w, _ = self._arm(widget, monkeypatch, events=[("warmed", object())])
        w._warm_source("m")
        assert "m" not in w._warms
        assert self._row_fraction(w) is None  # bar removed
        w._show_error.assert_not_called()

    def test_cancelled_clears_bar_and_state(self, widget, monkeypatch):
        w, _ = self._arm(widget, monkeypatch, events=[("cancelled", None)])
        w._warm_source("m")
        assert "m" not in w._warms
        assert self._row_fraction(w) is None
        w._show_error.assert_not_called()

    def test_failure_surfaces_inline_and_clears(self, widget, monkeypatch):
        # Hydrate is a background op (inline bar, no modal), so its failure lands
        # on the inline error pane -- not the modal _report_failure (#202/#206).
        w, _ = self._arm(widget, monkeypatch, events=[("failed", "disk full")])
        w._warm_source("m")
        assert "m" not in w._warms
        assert self._row_fraction(w) is None
        w._show_error.assert_called_once()
        assert "disk full" in w._show_error.call_args[0][0]
        w._report_failure.assert_not_called()

    def test_second_start_while_in_flight_is_noop(self, widget, monkeypatch):
        w, workers = self._arm(widget, monkeypatch, events=[])  # stays in flight
        w._warm_source("m")
        w._warm_source("m")  # dedup -- no second worker
        assert len(workers) == 1

    def test_cancel_warm_requests_worker_cancel(self, widget, monkeypatch):
        w, workers = self._arm(widget, monkeypatch, events=[])
        w._warm_source("m")
        assert workers and not workers[0].cancel_requested
        w._cancel_warm("m")
        assert workers[0].cancel_requested

    def test_worker_gc_retained_until_finished(self, widget, monkeypatch):
        # _warms drops the worker on `warmed` (so the menu flips back to
        # "Hydrate"), but the QThread must stay GC-anchored in _warm_retain until
        # `finished` fires, else a backend that doesn't keep the wrapper alive
        # could destroy a still-running thread (#202 review).
        w, workers = self._arm(widget, monkeypatch, events=[("warmed", object())])
        w._warm_source("m")
        assert "m" not in w._warms  # UI state already dropped
        assert workers[0] in w._warm_retain  # ...but GC ref still held
        workers[0].finished.emit()  # released only on finished
        assert workers[0] not in w._warm_retain

    def test_indicator_survives_tree_rebuild(self, widget, monkeypatch):
        # Every refresh/filter clear()s and rebuilds the tree, dropping the
        # per-item role; _reapply_warm_indicators paints the bar back (#202).
        w, _ = self._arm(
            widget,
            monkeypatch,
            events=[("progress", _warm_progress(bytes_done=10, bytes_total=40))],
        )
        w._warm_source("m")
        assert self._row_fraction(w) == 10 / 40
        self._add_row(w, w._sources["m"])  # simulate the clear/rebuild
        assert self._row_fraction(w) is None  # role gone after rebuild
        w._reapply_warm_indicators()
        assert self._row_fraction(w) == 10 / 40  # bar repainted

    def _menu_labels(self, w, monkeypatch):
        """Drive `_show_context_menu` over the 'm' row with a recording menu;
        return ``{label: action}`` (the action's ``trigger()`` fires its slot)."""
        from qtpy.QtCore import QPoint

        from biopb_mcp.tensor_browser import _widget as widget_mod

        recorded = {}

        class _RecAction:
            def __init__(self, text):
                self.text = text
                self.triggered = self
                self._slot = None

            def connect(self, cb):
                self._slot = cb

            def trigger(self):
                self._slot()

            def setEnabled(self, *_):
                pass

        class _RecMenu:
            def __init__(self, *a):
                pass

            def addAction(self, text):
                act = _RecAction(text)
                recorded[text] = act
                return act

            def addSeparator(self):
                pass

            def exec_(self, *a):
                pass

        monkeypatch.setattr(widget_mod, "QMenu", _RecMenu)
        item = w._find_source_item("m")
        w._tree_widget.itemAt = MagicMock(return_value=item)
        w._tree_widget.mapToGlobal = MagicMock(return_value=QPoint(0, 0))
        w._show_context_menu(QPoint(0, 0))
        return recorded

    def test_context_menu_offers_hydrate_when_idle(self, widget, monkeypatch):
        w, _ = self._arm(widget, monkeypatch, events=[("warmed", object())])
        w._warm_source("m")  # completes -> not warming
        labels = self._menu_labels(w, monkeypatch)
        assert "Hydrate all files…" in labels
        assert "Cancel hydration" not in labels

    def test_context_menu_offers_cancel_while_warming(self, widget, monkeypatch):
        w, workers = self._arm(widget, monkeypatch, events=[])  # in flight
        w._warm_source("m")
        labels = self._menu_labels(w, monkeypatch)
        assert "Cancel hydration" in labels
        assert "Hydrate all files…" not in labels
        labels["Cancel hydration"].trigger()  # the action cancels the warm
        assert workers[0].cancel_requested


class TestAddToViewer:
    """`_add_to_viewer` loads the selected tensor behind a busy cursor."""

    def _arm(self, widget):
        # `_client`/`_sources` are read-only views onto the connection.
        w, conn, _ = widget
        conn.client = MagicMock()
        conn.sources = {"m": _descriptor("m", tensors=["m"], source_type="zarr")}
        w._selected_source_id = "m"
        w._selected_tensor_id = "m"
        w._show_error = MagicMock()
        w._report_failure = MagicMock()
        return w

    def test_load_failure_reports_modally(self, widget, monkeypatch):
        # A failed view/load is user-initiated too, so it reports modally rather
        # than on the easily-missed inline pane (#206 consistency).
        from biopb_mcp.tensor_browser import _widget as widget_mod

        w = self._arm(widget)
        monkeypatch.setattr(
            widget_mod,
            "add_tensor_layer",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        w._add_to_viewer()
        w._report_failure.assert_called_once()
        assert "Failed to load tensor" in w._report_failure.call_args[0][1]
        w._show_error.assert_not_called()
