"""Unit tests for mcp/_bootstrap.py's Windows AppUserModelID helper.

biopb-napari-widget#1143: the taskbar icon needs the AUMID set before
``ip.enable_gui("qt")`` creates the QApplication (before napari.Viewer() gets
a chance to). These tests run without ``os.name == "nt"`` or a real
``ctypes.windll``, so they patch both.
"""

import ctypes
from unittest.mock import MagicMock, patch

from biopb_mcp.mcp import _bootstrap


class TestSetWindowsAppId:
    def test_noop_off_windows(self):
        with patch.object(_bootstrap.os, "name", "posix"):
            _bootstrap._set_windows_app_id()  # must not raise, must not touch ctypes

    def test_sets_the_app_id_on_windows(self):
        fake_shell32 = MagicMock()
        fake_windll = MagicMock(shell32=fake_shell32)
        with (
            patch.object(_bootstrap.os, "name", "nt"),
            patch.object(ctypes, "windll", fake_windll, create=True),
        ):
            _bootstrap._set_windows_app_id()
        fake_shell32.SetCurrentProcessExplicitAppUserModelID.assert_called_once()

    def test_fails_open_on_a_ctypes_error(self):
        fake_shell32 = MagicMock()
        fake_shell32.SetCurrentProcessExplicitAppUserModelID.side_effect = OSError(
            "boom"
        )
        fake_windll = MagicMock(shell32=fake_shell32)
        with (
            patch.object(_bootstrap.os, "name", "nt"),
            patch.object(ctypes, "windll", fake_windll, create=True),
        ):
            _bootstrap._set_windows_app_id()  # must not raise
