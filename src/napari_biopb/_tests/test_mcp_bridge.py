"""Tests for the MCP thread bridge."""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from napari_biopb.mcp._bridge import ThreadBridge


@pytest.fixture
def mock_viewer():
    viewer = MagicMock()
    viewer.window._dock_widgets = {}
    return viewer


@pytest.fixture
def bridge(mock_viewer):
    return ThreadBridge(mock_viewer)


class TestRunOnGuiThread:
    """Tests for ThreadBridge.run_on_gui_thread."""

    def test_executes_callable_and_returns_result(self, bridge):
        """Submitted callable is executed and its return value propagated."""

        # Simulate the Qt timer by draining manually from another thread
        def drain():
            time.sleep(0.05)
            bridge.process_pending()

        t = threading.Thread(target=drain)
        t.start()

        result = bridge.run_on_gui_thread(lambda: 42)
        t.join()
        assert result == 42

    def test_passes_args_to_callable(self, bridge):
        def drain():
            time.sleep(0.05)
            bridge.process_pending()

        t = threading.Thread(target=drain)
        t.start()

        result = bridge.run_on_gui_thread(lambda a, b: a + b, 3, 7)
        t.join()
        assert result == 10

    def test_propagates_exception(self, bridge):
        """Exceptions raised in the callable are re-raised in the caller."""

        def drain():
            time.sleep(0.05)
            bridge.process_pending()

        t = threading.Thread(target=drain)
        t.start()

        with pytest.raises(ValueError, match="boom"):
            bridge.run_on_gui_thread(_raise_value_error)
        t.join()

    def test_timeout_raises(self, bridge):
        """TimeoutError is raised when the queue is never drained."""
        with pytest.raises(TimeoutError):
            bridge.run_on_gui_thread(lambda: None, timeout=0.1)


class TestProcessPending:
    """Tests for ThreadBridge.process_pending."""

    def test_drains_multiple_items(self, bridge):
        """All queued items are processed in one call."""
        results = []
        for i in range(3):
            rq = queue.Queue()
            bridge._cmd_queue.put((lambda x=i: x * 2, (), rq))
            results.append(rq)

        bridge.process_pending()

        for i, rq in enumerate(results):
            ok, val = rq.get_nowait()
            assert ok is True
            assert val == i * 2

    def test_noop_when_queue_empty(self, bridge):
        """No error when the queue is empty."""
        bridge.process_pending()

    def test_error_in_one_does_not_block_others(self, bridge):
        rq1 = queue.Queue()
        rq2 = queue.Queue()
        bridge._cmd_queue.put((_raise_value_error, (), rq1))
        bridge._cmd_queue.put((lambda: "ok", (), rq2))

        bridge.process_pending()

        ok1, val1 = rq1.get_nowait()
        assert ok1 is False
        assert isinstance(val1, ValueError)

        ok2, val2 = rq2.get_nowait()
        assert ok2 is True
        assert val2 == "ok"


class TestTensorClientProperty:
    """Tests for ThreadBridge.tensor_client property."""

    def test_returns_none_when_no_widgets(self, bridge):
        assert bridge.tensor_client is None

    def test_returns_none_when_widget_not_connected(self, bridge):
        from napari_biopb.tensor_browser import TensorBrowserWidget

        mock_dock = MagicMock()
        mock_widget = MagicMock(spec=TensorBrowserWidget)
        mock_widget._client = None
        mock_dock.widget.return_value = mock_widget
        bridge.viewer.window._dock_widgets = {"tb": mock_dock}

        assert bridge.tensor_client is None

    def test_returns_client_from_connected_widget(self, bridge):
        from napari_biopb.tensor_browser import TensorBrowserWidget

        mock_dock = MagicMock()
        mock_widget = MagicMock(spec=TensorBrowserWidget)
        mock_client = MagicMock()
        mock_widget._client = mock_client
        mock_dock.widget.return_value = mock_widget
        bridge.viewer.window._dock_widgets = {"tb": mock_dock}

        assert bridge.tensor_client is mock_client


class TestTensorSourcesProperty:
    """Tests for ThreadBridge.tensor_sources property."""

    def test_returns_empty_dict_when_no_widgets(self, bridge):
        assert bridge.tensor_sources == {}

    def test_returns_sources_from_widget(self, bridge):
        from napari_biopb.tensor_browser import TensorBrowserWidget

        mock_dock = MagicMock()
        mock_widget = MagicMock(spec=TensorBrowserWidget)
        mock_widget._sources = {"src1": MagicMock()}
        mock_dock.widget.return_value = mock_widget
        bridge.viewer.window._dock_widgets = {"tb": mock_dock}

        assert "src1" in bridge.tensor_sources


def _raise_value_error():
    raise ValueError("boom")
