"""Thread bridge between MCP server (background thread) and Qt main thread.

Dispatches typed callables to the Qt event loop via a queue, drained by a
QTimer every 50 ms. MCP tool handlers call ``bridge.run_on_gui_thread(fn)``
and block until the result (or timeout) is ready.
"""

import logging
import queue
from typing import Any, Callable, Dict, Optional

from qtpy.QtCore import QTimer

logger = logging.getLogger(__name__)


class ThreadBridge:
    def __init__(self, viewer):
        self.viewer = viewer
        self._cmd_queue: queue.Queue = queue.Queue()
        self._timer: Optional[QTimer] = None

    @property
    def tensor_client(self):
        """Get TensorFlightClient from active TensorBrowserWidget."""
        from ..tensor_browser import TensorBrowserWidget

        for dock in self.viewer.window._dock_widgets.values():
            inner = dock.widget()
            if (
                isinstance(inner, TensorBrowserWidget)
                and inner._client is not None
            ):
                return inner._client
        return None

    @property
    def tensor_sources(self) -> Dict:
        """Get cached sources dict from active TensorBrowserWidget."""
        from ..tensor_browser import TensorBrowserWidget

        for dock in self.viewer.window._dock_widgets.values():
            inner = dock.widget()
            if isinstance(inner, TensorBrowserWidget) and inner._sources:
                return inner._sources
        return {}

    def run_on_gui_thread(
        self, fn: Callable, *args, timeout: float = 30.0
    ) -> Any:
        """Submit *fn(*args)* for execution on the Qt main thread.

        Blocks the calling (MCP server) thread until the result is ready or
        *timeout* seconds elapse.  Exceptions raised inside *fn* are
        re-raised in the caller.
        """
        result_queue: queue.Queue = queue.Queue()
        self._cmd_queue.put((fn, args, result_queue))
        try:
            ok, value = result_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"GUI-thread call timed out after {timeout}s")
        if not ok:
            raise value
        return value

    def process_pending(self):
        """Drain the command queue — called by QTimer on the Qt thread."""
        while not self._cmd_queue.empty():
            try:
                fn, args, result_queue = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            try:
                result = fn(*args)
                result_queue.put((True, result))
            except Exception as exc:
                result_queue.put((False, exc))

    def start_timer(self):
        if self._timer is not None:
            return
        self._timer = QTimer()
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.process_pending)
        self._timer.start()
        logger.debug("Bridge timer started (50 ms)")

    def stop_timer(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
            logger.debug("Bridge timer stopped")
