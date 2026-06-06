"""Tests for the TensorBrowserWidget non-blocking connect poller (issue #12).

These exercise the connect/auto-connect state machine — readiness gating
(STARTING), fail-fast on down, autostart hand-off, capped-backoff re-arm, and
generation-based supersession — with a mocked ``TensorConnection`` and the tree
renderer patched out, so they test the poller rather than tree rendering.

A real ``napari`` viewer (and thus a Qt/OpenGL context) is required, so the
suite is skipped on macOS CI like the other viewer tests.
"""

import os
import sys
import time
from unittest.mock import MagicMock

import pytest

from biopb_mcp._connection import ServerStarting

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("CI") == "true",
    reason="OpenGL context unavailable on macOS CI headless environment",
)


@pytest.fixture
def widget(make_napari_viewer, monkeypatch):
    from qtpy.QtCore import QTimer

    from biopb_mcp.tensor_browser._widget import TensorBrowserWidget

    viewer = make_napari_viewer(show=False)
    conn = MagicMock()
    conn.url = "grpc://localhost:8815"
    conn.token = None
    conn.use_server_query = False
    conn.last_message = "Tensor server is starting — scanning its data folder."
    # Neutralize QTimer.singleShot so neither the auto-connect-on-construction
    # tick nor the poller's backoff re-arm fire on their own — the tests drive
    # the poll chain explicitly for determinism.
    monkeypatch.setattr(QTimer, "singleShot", lambda *a, **k: None)
    w = TensorBrowserWidget(viewer, connection=conn)
    # Isolate the poller from tree rendering (which needs real descriptors).
    w._build_and_display_tree = MagicMock()
    return w, conn


class TestConnectPoller:
    def test_success_builds_tree(self, widget):
        w, conn = widget
        conn.connect.return_value = {"a": object()}

        w._connect()

        conn.connect.assert_called_once_with("grpc://localhost:8815", None)
        w._build_and_display_tree.assert_called_once()
        assert w._refresh_button.isEnabled()
        assert w._status_label.isHidden()

    def test_down_fails_fast_no_autostart_offer(self, widget):
        w, conn = widget
        conn.connect.side_effect = RuntimeError("connection refused")
        w._maybe_offer_start_server = MagicMock()

        w._on_connect_clicked()  # manual connect -> offer flag False

        assert not w._error_label.isHidden()
        assert "Cannot reach" in w._error_label.text()
        assert not w._refresh_button.isEnabled()
        w._maybe_offer_start_server.assert_not_called()

    def test_down_offers_autostart_on_auto_connect(self, widget):
        w, conn = widget
        conn.connect.side_effect = RuntimeError("connection refused")
        w._maybe_offer_start_server = MagicMock()

        w._auto_connect()  # sets offer flag True

        w._maybe_offer_start_server.assert_called_once()

    def test_empty_catalog_shows_error(self, widget):
        w, conn = widget
        conn.connect.return_value = {}

        w._connect()

        assert "No sources found" in w._error_label.text()
        assert not w._refresh_button.isEnabled()
        w._build_and_display_tree.assert_not_called()

    def test_large_catalog_enables_sql_filter(self, widget):
        w, conn = widget
        conn.use_server_query = True
        conn.connect.return_value = {"a": object()}

        w._connect()

        w._build_and_display_tree.assert_called_once()
        assert w._refresh_button.isEnabled()
        assert "SQL filter" in w._filter_input.placeholderText()

    def test_autostart_launch_failure_shows_error(self, widget, monkeypatch):
        from qtpy.QtWidgets import QApplication, QMessageBox

        w, conn = widget
        monkeypatch.setattr(QApplication, "platformName", lambda *a: "xcb")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.Yes
        )
        conn.can_autostart_server.return_value = True
        conn.server_start_timeout.return_value = 30.0
        conn.launch_local_server.side_effect = RuntimeError("port in use")

        # Launch failure is unexpected -> inline message + propagate to napari.
        with pytest.raises(RuntimeError, match="port in use"):
            w._maybe_offer_start_server()

        assert "Failed to start local biopb server" in w._error_label.text()
        # No poll started when the launch itself failed.
        assert w._connect_boot_deadline is None

    def test_starting_then_ready(self, widget):
        w, conn = widget
        conn.connect.side_effect = [
            ServerStarting("STARTING"),
            {"a": object()},
        ]

        w._connect()  # first tick: STARTING -> show status, re-arm
        assert not w._status_label.isHidden()
        assert "scanning" in w._status_label.text()
        w._build_and_display_tree.assert_not_called()

        # Simulate the backoff timer firing the next tick.
        w._connect_tick(w._connect_gen)
        w._build_and_display_tree.assert_called_once()
        assert w._status_label.isHidden()

    def test_stale_tick_is_superseded(self, widget):
        w, conn = widget
        w._connect_gen = 5

        w._connect_tick(3)  # older generation

        conn.connect.assert_not_called()

    def test_retarget_supersedes_inflight_wait(self, widget):
        w, conn = widget
        conn.connect.side_effect = ServerStarting("STARTING")

        w._connect()  # enters the STARTING wait
        gen_first = w._connect_gen

        # User points at a different server and reconnects.
        conn.connect.side_effect = None
        conn.connect.return_value = {"a": object()}
        w._server_input.setText("grpc://other:9")
        w._on_connect_clicked()
        assert w._connect_gen != gen_first

        # A late tick from the original chain must no-op (local server is left
        # running; we just stop polling it).
        conn.connect.reset_mock()
        w._connect_tick(gen_first)
        conn.connect.assert_not_called()

    def test_boot_tolerance_keeps_waiting(self, widget):
        w, conn = widget
        conn.connect.side_effect = RuntimeError("connection refused")
        w._connect_boot_deadline = time.monotonic() + 100

        w._connect_tick(w._connect_gen)

        assert "Starting local biopb server" in w._status_label.text()
        assert w._error_label.isHidden()

    def test_boot_timeout_gives_up(self, widget):
        w, conn = widget
        conn.connect.side_effect = RuntimeError("connection refused")
        w._connect_boot_deadline = time.monotonic() - 1  # already elapsed

        # A server we launched ourselves never coming up is unexpected -> the
        # cause is surfaced inline and re-raised for napari's notifications.
        with pytest.raises(RuntimeError, match="connection refused"):
            w._connect_tick(w._connect_gen)

        assert "did not become ready" in w._error_label.text()
        assert "connection refused" in w._error_label.text()
        assert w._connect_boot_deadline is None
        assert not w._refresh_button.isEnabled()

    def test_autostart_launches_and_polls_without_blocking(
        self, widget, monkeypatch
    ):
        from qtpy.QtWidgets import QApplication, QMessageBox

        w, conn = widget
        # Pretend we are not headless so the offer proceeds, and auto-accept.
        monkeypatch.setattr(QApplication, "platformName", lambda *a: "xcb")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.Yes
        )
        conn.can_autostart_server.return_value = True
        conn.server_start_timeout.return_value = 30.0
        # The just-launched server is not reachable yet -> boot mode tolerates.
        conn.connect.side_effect = RuntimeError("connection refused")

        w._maybe_offer_start_server()

        conn.launch_local_server.assert_called_once()
        assert w._connect_boot_deadline is not None
        assert w._offer_autostart_on_fail is False
        assert "Starting local biopb server" in w._status_label.text()
