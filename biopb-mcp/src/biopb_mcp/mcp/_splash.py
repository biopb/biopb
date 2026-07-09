"""A lightweight Qt splash shown while the napari viewer boots.

Kernel bring-up is slow — the dominant cost is ``import napari`` plus
``napari.Viewer()`` construction (seconds; worse on Windows and on the first
launch of a session). The bootstrap enables the Qt event loop
(``ip.enable_gui("qt")``) *before* that heavy import, so Qt is already live and
a ``QSplashScreen`` can give the watching scientist an immediate
"starting, please wait" cue instead of a blank screen while the window is
built. It is purely cosmetic feedback — a workaround for the startup latency,
not a fix — so it is **fully best-effort**: every entry point fails open to a
no-op handle and can never break or delay the bootstrap.

Only the GUI (non-headless) branch uses it; a headless kernel has no Qt loop
and nothing to look at.
"""

import logging

logger = logging.getLogger(__name__)

_TITLE = "biopb-mcp"
# The splash paints these below the title; showMessage() overwrites the bottom
# strip with live progress ("Loading napari…", etc.).
_SUBTITLE = "Starting the napari session"


class _NullSplash:
    """No-op splash handle returned whenever a real one can't be shown.

    Lets the bootstrap call ``.message()`` / ``.finish()`` unconditionally
    without guarding for Qt availability.
    """

    def message(self, text: str) -> None:
        pass

    def finish(self, viewer) -> None:
        pass

    def close(self) -> None:
        pass


class _QtSplash:
    """Wraps a live ``QSplashScreen`` so callers need no Qt imports."""

    def __init__(self, splash, app):
        self._splash = splash
        self._app = app

    def message(self, text: str) -> None:
        """Update the progress line and repaint (best-effort).

        The napari import blocks the Qt main thread, so we can't animate; we
        just repaint once per step via ``processEvents`` so the latest message
        is on screen before the next blocking call.
        """
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtGui import QColor

            self._splash.showMessage(
                text,
                int(Qt.AlignBottom | Qt.AlignHCenter),
                QColor("#c8cee0"),
            )
            self._app.processEvents()
        except Exception:
            logger.debug("splash message failed (fail-open)", exc_info=True)

    def finish(self, viewer) -> None:
        """Dismiss the splash once the viewer's window is up.

        ``QSplashScreen.finish(widget)`` keeps the splash visible until
        ``widget`` is shown, so passing napari's Qt main window hands the
        screen off to the viewer with no flicker/gap. Falls back to a plain
        close if the window handle can't be reached.
        """
        try:
            qt_window = None
            window = getattr(viewer, "window", None)
            if window is not None:
                qt_window = getattr(window, "_qt_window", None)
            if qt_window is not None:
                self._splash.finish(qt_window)
            else:
                self._splash.close()
            self._app.processEvents()
        except Exception:
            logger.debug("splash finish failed (fail-open)", exc_info=True)
            self.close()

    def close(self) -> None:
        try:
            self._splash.close()
        except Exception:
            logger.debug("splash close failed (fail-open)", exc_info=True)


def _render_pixmap():
    """Paint the static splash background (no image asset — self-contained)."""
    from qtpy.QtCore import QRect, Qt
    from qtpy.QtGui import QColor, QFont, QPainter, QPixmap

    width, height = 440, 220
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#1b1e2b"))

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)

        painter.setPen(QColor("#eef1f8"))
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(QRect(0, 62, width, 44), int(Qt.AlignHCenter), _TITLE)

        painter.setPen(QColor("#9aa0b4"))
        sub_font = QFont()
        sub_font.setPointSize(11)
        painter.setFont(sub_font)
        painter.drawText(QRect(0, 112, width, 26), int(Qt.AlignHCenter), _SUBTITLE)
    finally:
        painter.end()
    return pixmap


def show_splash():
    """Show the startup splash and return a handle, or a no-op on any failure.

    Requires a live ``QApplication`` (the bootstrap's ``enable_gui("qt")``
    creates one). Returns ``_NullSplash`` when Qt is unavailable or no app
    exists, so the caller never has to branch.
    """
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication, QSplashScreen

        app = QApplication.instance()
        if app is None:
            return _NullSplash()

        splash = QSplashScreen(_render_pixmap())
        splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        splash.show()
        app.processEvents()
        return _QtSplash(splash, app)
    except Exception:
        logger.debug("splash unavailable (fail-open)", exc_info=True)
        return _NullSplash()
