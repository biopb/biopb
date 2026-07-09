"""Unit tests for the startup splash (mcp/_splash.py), display-free.

The splash is best-effort cosmetic feedback for the slow kernel bring-up, so
what matters is that it never raises and always yields a usable handle. These
tests patch Qt so they run without a real display or QApplication.
"""

from unittest.mock import MagicMock, patch

import qtpy.QtWidgets as qtw

from biopb_mcp.mcp import _splash
from biopb_mcp.mcp._splash import _NullSplash, _QtSplash, show_splash


class TestShowSplash:
    def test_no_qapplication_returns_null(self):
        """Without a live QApplication, show_splash fails open to _NullSplash."""
        with patch.object(qtw.QApplication, "instance", return_value=None):
            handle = show_splash()
        assert isinstance(handle, _NullSplash)

    def test_qt_error_returns_null(self):
        """Any Qt exception during setup fails open to _NullSplash."""
        with (
            patch.object(qtw.QApplication, "instance", return_value=MagicMock()),
            patch.object(_splash, "_render_pixmap", side_effect=RuntimeError("boom")),
        ):
            handle = show_splash()
        assert isinstance(handle, _NullSplash)

    def test_live_app_returns_qt_splash(self):
        """With an app present, a real _QtSplash is built and shown."""
        app = MagicMock()
        with (
            patch.object(qtw.QApplication, "instance", return_value=app),
            patch.object(_splash, "_render_pixmap", return_value=MagicMock()),
            patch("qtpy.QtWidgets.QSplashScreen") as mock_splash_cls,
        ):
            handle = show_splash()
            splash_obj = mock_splash_cls.return_value
        assert isinstance(handle, _QtSplash)
        splash_obj.show.assert_called_once()
        app.processEvents.assert_called()


class TestNullSplash:
    def test_methods_are_noops(self):
        null = _NullSplash()
        # None of these raise.
        null.message("anything")
        null.finish(object())
        null.close()


class TestQtSplashFinish:
    def test_finish_uses_viewer_qt_window(self):
        """finish() hands off to the napari Qt main window when reachable."""
        splash_obj, app = MagicMock(), MagicMock()
        handle = _QtSplash(splash_obj, app)

        qt_window = MagicMock()
        viewer = MagicMock()
        viewer.window._qt_window = qt_window

        handle.finish(viewer)
        splash_obj.finish.assert_called_once_with(qt_window)
        splash_obj.close.assert_not_called()

    def test_finish_falls_back_to_close_without_window(self):
        splash_obj, app = MagicMock(), MagicMock()
        handle = _QtSplash(splash_obj, app)

        viewer = MagicMock()
        viewer.window = None  # no window handle to hand off to

        handle.finish(viewer)
        splash_obj.close.assert_called_once()
        splash_obj.finish.assert_not_called()

    def test_message_never_raises_on_qt_failure(self):
        splash_obj, app = MagicMock(), MagicMock()
        splash_obj.showMessage.side_effect = RuntimeError("qt gone")
        handle = _QtSplash(splash_obj, app)
        handle.message("Loading…")  # swallowed, no raise
