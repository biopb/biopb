"""Tests for the MCP server tools and resources.

The tools dispatch into a child kernel; here that kernel is replaced by a
``mock_kernel_host`` returning canned ``execute`` result dicts, so the tests
exercise the server-side formatting/extraction without a real kernel.
"""

import base64

from unittest.mock import MagicMock

import pytest

from napari_biopb.mcp import _server


def _result(stdout="", result_text="", error_text="", status="ok"):
    return {
        "stdout": stdout,
        "result_text": result_text,
        "error_text": error_text,
        "status": status,
    }


@pytest.fixture(autouse=True)
def reset_server_state():
    old_host = _server._kernel_host
    yield
    _server._kernel_host = old_host


@pytest.fixture
def mock_kernel_host():
    host = MagicMock()
    host.is_alive.return_value = True
    host.is_busy.return_value = False
    host.execute.return_value = _result()
    return host


@pytest.fixture
def server_with_host(mock_kernel_host):
    _server.set_kernel_host(mock_kernel_host)
    return mock_kernel_host


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
    def test_returns_error_when_no_host(self):
        _server._kernel_host = None
        result = _server.take_screenshot()
        assert len(result) == 1
        assert result[0].type == "text"
        assert "not initialized" in result[0].text

    def test_returns_png_image_from_delimited_stdout(self, server_with_host):
        data = base64.b64encode(b"fake-png-bytes").decode()
        server_with_host.execute.return_value = _result(
            stdout=f"<<PNG_B64>>{data}\n"
        )

        result = _server.take_screenshot(canvas_only=True)

        assert len(result) == 1
        assert result[0].type == "image"
        assert result[0].mimeType == "image/png"
        assert result[0].data == data

    def test_returns_text_when_no_delimiter(self, server_with_host):
        server_with_host.execute.return_value = _result(
            error_text="boom", status="error"
        )
        result = _server.take_screenshot()
        assert result[0].type == "text"
        assert "Screenshot failed" in result[0].text

    def test_passes_canvas_only_flag(self, server_with_host):
        data = base64.b64encode(b"x").decode()
        server_with_host.execute.return_value = _result(
            stdout=f"<<PNG_B64>>{data}"
        )
        _server.take_screenshot(canvas_only=False)
        snippet = server_with_host.execute.call_args[0][0]
        assert "canvas_only=False" in snippet


# -----------------------------------------------------------------------
# execute_code
# -----------------------------------------------------------------------


class TestExecuteCode:
    def test_returns_error_when_no_host(self):
        _server._kernel_host = None
        result = _server.execute_code("print('hi')")
        assert "not initialized" in result

    def test_prepends_refresh_prefix(self, server_with_host):
        _server.execute_code("print('hi')")
        code = server_with_host.execute.call_args[0][0]
        assert code.startswith("client = _tbw._client")
        assert "print('hi')" in code

    def test_formats_stdout_and_result(self, server_with_host):
        server_with_host.execute.return_value = _result(
            stdout="hello\n", result_text="3"
        )
        result = _server.execute_code("print('hello'); 1 + 2")
        assert "hello" in result
        assert "3" in result

    def test_no_output_message(self, server_with_host):
        server_with_host.execute.return_value = _result()
        result = _server.execute_code("x = 42")
        assert result == "(no output)"

    def test_error_path_includes_traceback(self, server_with_host):
        server_with_host.execute.return_value = _result(
            error_text="Traceback...\nZeroDivisionError: division by zero",
            status="error",
        )
        result = _server.execute_code("1 / 0")
        assert "division by zero" in result

    def test_timeout_status_returned(self, server_with_host):
        server_with_host.execute.return_value = _result(
            error_text="Execution exceeded 0.5s and was interrupted.",
            status="timeout",
        )
        result = _server.execute_code("while True: pass")
        assert "interrupted" in result


# -----------------------------------------------------------------------
# inspect_object
# -----------------------------------------------------------------------


class TestInspectObject:
    def test_returns_error_when_no_host(self):
        _server._kernel_host = None
        result = _server.inspect_object("viewer")
        assert "not initialized" in result

    def test_injects_repr_of_path(self, server_with_host):
        server_with_host.execute.return_value = _result(stdout="Type: Mock")
        _server.inspect_object("viewer.layers")
        snippet = server_with_host.execute.call_args[0][0]
        assert "'viewer.layers'" in snippet

    def test_returns_stdout_on_success(self, server_with_host):
        server_with_host.execute.return_value = _result(
            stdout="Type: list\nAttributes:\n"
        )
        result = _server.inspect_object("my_obj")
        assert "Type: list" in result

    def test_returns_error_text_on_failure(self, server_with_host):
        server_with_host.execute.return_value = _result(
            error_text="NameError: name 'nope' is not defined",
            status="error",
        )
        result = _server.inspect_object("nope")
        assert "NameError" in result


# -----------------------------------------------------------------------
# interrupt / restart
# -----------------------------------------------------------------------


class TestInterruptRestart:
    def test_interrupt_delegates_to_host(self, server_with_host):
        result = _server.interrupt_kernel()
        server_with_host.interrupt.assert_called_once()
        assert "SIGINT" in result

    def test_interrupt_no_host(self):
        _server._kernel_host = None
        assert "not initialized" in _server.interrupt_kernel()

    def test_restart_delegates_to_host(self, server_with_host):
        result = _server.restart_kernel()
        server_with_host.restart.assert_called_once()
        assert "restarted" in result.lower()

    def test_restart_reports_failure(self, server_with_host):
        server_with_host.restart.side_effect = RuntimeError("nope")
        result = _server.restart_kernel()
        assert "failed" in result.lower()

    def test_restart_no_host(self):
        _server._kernel_host = None
        assert "not initialized" in _server.restart_kernel()


# -----------------------------------------------------------------------
# server_status
# -----------------------------------------------------------------------


class TestServerStatus:
    def test_reports_not_initialized(self):
        _server._kernel_host = None
        result = _server.server_status()
        assert "System" in result
        assert "not initialized" in result

    def test_reports_system_info(self, server_with_host):
        result = _server.server_status()
        assert "cpu_usage" in result
        assert "memory_total" in result
        assert "process_rss" in result

    def test_reports_kernel_state(self, server_with_host):
        result = _server.server_status()
        assert "## Kernel" in result
        assert "alive: True" in result
        assert "busy: False" in result

    def test_appends_kernel_query_output(self, server_with_host):
        server_with_host.execute.return_value = _result(
            stdout="## Dask\n  scheduler: threads\n## Viewer\n  layers: 0"
        )
        result = _server.server_status()
        assert "scheduler: threads" in result
        assert "layers: 0" in result

    def test_handles_busy_kernel(self, server_with_host):
        server_with_host.execute.return_value = _result(status="busy")
        result = _server.server_status()
        assert "busy" in result.lower()

    def test_no_sessions_or_bridge_sections(self, server_with_host):
        result = _server.server_status()
        assert "Sessions" not in result
        assert "Bridge" not in result
