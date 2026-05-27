"""Tests for the MCP server tools and resources."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from napari_biopb.mcp import _server


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset module-level state between tests."""
    old_bridge = _server._bridge
    yield
    _server._bridge = old_bridge


@pytest.fixture
def mock_bridge():
    bridge = MagicMock()
    bridge.viewer = MagicMock()
    bridge.viewer.layers = []
    bridge.tensor_client = None
    bridge.tensor_sources = {}
    return bridge


@pytest.fixture
def server_with_bridge(mock_bridge):
    """Set the module-level bridge so tools can run.

    The bridge's run_on_gui_thread immediately calls the function
    (bypassing the real queue) so tests stay single-threaded.
    """
    mock_bridge.run_on_gui_thread.side_effect = lambda fn, *a, **kw: fn()
    _server._bridge = mock_bridge
    return mock_bridge


# -----------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------


class TestResources:
    def test_guide_resource_returns_string(self):
        content = _server.get_guide()
        assert "napari-biopb" in content
        assert "execute_code" in content

    def test_viewer_resource_mentions_layers(self):
        content = _server.get_viewer_guide()
        assert "viewer.layers" in content

    def test_tensor_resource_mentions_client(self):
        content = _server.get_tensor_guide()
        assert "client" in content

    def test_annotations_resource_mentions_points(self):
        content = _server.get_annotations_guide()
        assert "add_points" in content


# -----------------------------------------------------------------------
# take_screenshot
# -----------------------------------------------------------------------


class TestTakeScreenshot:
    def test_returns_error_when_no_bridge(self):
        _server._bridge = None
        result = _server.take_screenshot()
        assert len(result) == 1
        assert result[0].type == "text"
        assert "not initialized" in result[0].text

    def test_returns_png_image(self, server_with_bridge):
        fake_arr = np.zeros((100, 100, 4), dtype=np.uint8)
        server_with_bridge.viewer.screenshot.return_value = fake_arr

        result = _server.take_screenshot(canvas_only=True)

        assert len(result) == 1
        assert result[0].type == "image"
        assert result[0].mimeType == "image/png"
        assert len(result[0].data) > 0

    def test_passes_canvas_only_flag(self, server_with_bridge):
        fake_arr = np.zeros((50, 50, 4), dtype=np.uint8)
        server_with_bridge.viewer.screenshot.return_value = fake_arr

        _server.take_screenshot(canvas_only=False)

        # The inner closure captures canvas_only=False
        server_with_bridge.viewer.screenshot.assert_called_once_with(
            canvas_only=False
        )


# -----------------------------------------------------------------------
# execute_code
# -----------------------------------------------------------------------


class TestExecuteCode:
    def test_returns_error_when_no_bridge(self):
        _server._bridge = None
        result = _server.execute_code("print('hi')")
        assert "not initialized" in result

    def test_executes_print_statement(self, server_with_bridge):
        result = _server.execute_code("print('hello world')")
        assert "hello world" in result

    def test_returns_expression_repr(self, server_with_bridge):
        result = _server.execute_code("1 + 2")
        assert "3" in result

    def test_multi_line_code(self, server_with_bridge):
        code = "x = 5\nprint(x * 2)"
        result = _server.execute_code(code)
        assert "10" in result

    def test_reports_errors(self, server_with_bridge):
        result = _server.execute_code("1 / 0")
        assert "Error" in result
        assert "division by zero" in result

    def test_namespace_has_numpy(self, server_with_bridge):
        result = _server.execute_code("print(type(np).__name__)")
        assert "module" in result

    def test_namespace_has_viewer(self, server_with_bridge):
        result = _server.execute_code("print(type(viewer))")
        assert "Mock" in result or "Viewer" in result

    def test_import_blocked(self, server_with_bridge):
        result = _server.execute_code("import os")
        assert "Error" in result

    def test_open_blocked(self, server_with_bridge):
        result = _server.execute_code("open('/etc/passwd')")
        assert "Error" in result

    def test_no_output_message(self, server_with_bridge):
        result = _server.execute_code("x = 42")
        assert result == "(no output)"


# -----------------------------------------------------------------------
# inspect_object
# -----------------------------------------------------------------------


class TestInspectObject:
    def test_returns_error_when_no_bridge(self):
        _server._bridge = None
        result = _server.inspect_object("viewer")
        assert "not initialized" in result

    def test_inspects_viewer(self, server_with_bridge):
        result = _server.inspect_object("viewer")
        assert "Type:" in result

    def test_inspects_numpy(self, server_with_bridge):
        result = _server.inspect_object("np")
        assert "module" in result.lower() or "Type:" in result

    def test_invalid_path_returns_error(self, server_with_bridge):
        result = _server.inspect_object("nonexistent_object_xyz")
        assert "Error" in result
        assert "nonexistent_object_xyz" in result


# -----------------------------------------------------------------------
# Safe builtins
# -----------------------------------------------------------------------


class TestSafeBuiltins:
    def test_safe_builtins_allow_basic_operations(self, server_with_bridge):
        result = _server.execute_code(
            "print(len([1,2,3]), max(1,2), min(1,2))"
        )
        assert "3" in result
        assert "2" in result
        assert "1" in result

    def test_safe_builtins_allow_type_conversions(self, server_with_bridge):
        result = _server.execute_code("print(int('42'), float('3.14'))")
        assert "42" in result
        assert "3.14" in result

    def test_eval_blocked(self, server_with_bridge):
        result = _server.execute_code("eval('1+1')")
        assert "Error" in result

    def test_exec_blocked(self, server_with_bridge):
        result = _server.execute_code("exec('x=1')")
        assert "Error" in result


# -----------------------------------------------------------------------
# launch / shutdown
# -----------------------------------------------------------------------


class TestServerLifecycle:
    def test_shutdown_clears_bridge(self):
        _server._bridge = MagicMock()
        _server.shutdown_server()
        assert _server._bridge is None

    def test_shutdown_noop_when_no_bridge(self):
        _server._bridge = None
        _server.shutdown_server()
        assert _server._bridge is None
