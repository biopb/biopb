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


class TestSetWindowsClassIcon:
    """The busy-thread fallback: the taskbar reads the class icon (#1143)."""

    @staticmethod
    def _user32(icons):
        user32 = MagicMock()
        user32.SendMessageW.side_effect = lambda hwnd, msg, which, _: icons.get(which)
        user32.CopyIcon.side_effect = lambda icon: icon + 1000
        return user32

    @staticmethod
    def _window(hwnd=42):
        window = MagicMock()
        window.winId.return_value = hwnd
        return window

    def _run(self, user32, window):
        with (
            patch.object(_bootstrap.os, "name", "nt"),
            patch.object(ctypes, "windll", MagicMock(user32=user32), create=True),
        ):
            _bootstrap._set_windows_class_icon(window)

    def test_noop_off_windows(self):
        window = self._window()
        with patch.object(_bootstrap.os, "name", "posix"):
            _bootstrap._set_windows_class_icon(window)
        window.winId.assert_not_called()

    def test_copies_both_window_icons_onto_the_class(self):
        user32 = self._user32({_bootstrap._ICON_BIG: 7, _bootstrap._ICON_SMALL: 9})
        self._run(user32, self._window(42))
        user32.SetClassLongPtrW.assert_any_call(42, _bootstrap._GCLP_HICON, 1007)
        user32.SetClassLongPtrW.assert_any_call(42, _bootstrap._GCLP_HICONSM, 1009)

    def test_a_missing_icon_leaves_that_class_slot_alone(self):
        user32 = self._user32({_bootstrap._ICON_BIG: 7})
        self._run(user32, self._window(42))
        user32.SetClassLongPtrW.assert_called_once_with(
            42, _bootstrap._GCLP_HICON, 1007
        )

    def test_fails_open_on_a_ctypes_error(self):
        user32 = self._user32({})
        user32.SendMessageW.side_effect = OSError("boom")
        self._run(user32, self._window())  # must not raise
