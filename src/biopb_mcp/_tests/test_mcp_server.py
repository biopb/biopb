"""Tests for the MCP server tools and resources.

The tools dispatch into a child kernel; here that kernel is replaced by a
``mock_kernel_host`` returning canned ``execute`` result dicts, so the tests
exercise the server-side formatting/extraction without a real kernel.
"""

import base64
import json
from unittest.mock import MagicMock

import pytest

from biopb_mcp.mcp import _server


def _result(stdout="", result_text="", error_text="", status="ok"):
    return {
        "stdout": stdout,
        "result_text": result_text,
        "error_text": error_text,
        "status": status,
    }


def _job_reply(**payload):
    """A kernel ``execute`` result whose stdout carries the job runner's
    single-line ``<<JOB_JSON>>`` payload (what submit/poll/cancel/list print).
    """
    return _result(stdout=_server._JOB_DELIM + json.dumps(payload) + "\n")


def _snapshot(
    job_id="job-1",
    status="ok",
    stdout="",
    result_text="",
    error_text="",
    elapsed=0.1,
):
    return {
        "job_id": job_id,
        "status": status,
        "stdout": stdout,
        "result_text": result_text,
        "error_text": error_text,
        "elapsed": elapsed,
    }


@pytest.fixture(autouse=True)
def reset_server_state():
    old_host = _server._kernel_host
    old_promote = _server._promote_after
    old_headless = _server._headless
    old_instructions = _server.mcp._mcp_server.instructions
    yield
    _server._kernel_host = old_host
    _server._promote_after = old_promote
    _server._headless = old_headless
    _server.mcp._mcp_server.instructions = old_instructions


@pytest.fixture
def mock_kernel_host():
    host = MagicMock()
    host.is_alive.return_value = True
    host.is_busy.return_value = False
    host.health.return_value = {
        "alive": True,
        "ready": True,
        "start_error": None,
        "busy": False,
        "dead": False,
        "recent_respawns": 0,
        "watchdog_running": True,
    }
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
        assert "biopb-mcp" in content
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

    def test_headless_returns_message_without_touching_kernel(
        self, server_with_host
    ):
        _server.set_headless(True)
        result = _server.take_screenshot()
        assert result[0].type == "text"
        assert "headless" in result[0].text.lower()
        # Must short-circuit before dispatching into the kernel.
        server_with_host.execute.assert_not_called()

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
# headless mode
# -----------------------------------------------------------------------


class TestSetHeadless:
    def test_base_instructions_carry_guardrails(self):
        # The operation guardrails must be delivered up front via the handshake
        # instructions (not left to a pull-on-demand resource).
        base = _server._BASE_INSTRUCTIONS
        assert "query_sources" in base
        assert "run_on_main" in base
        assert "filesystem" in base.lower()
        # And they are advertised when visible (no headless directive).
        _server.set_headless(False)
        assert _server.mcp._mcp_server.instructions == base

    def test_headless_appends_directive_to_base_instructions(self):
        _server.set_headless(True)
        instr = _server.mcp._mcp_server.instructions
        assert instr is not None
        # Always-on base guidance plus the headless directive.
        assert instr.startswith(_server._BASE_INSTRUCTIONS)
        assert "HEADLESS" in instr
        # The directive is conditioned on the user reaching for the viewer.
        assert "viewer" in instr.lower()

    def test_visible_keeps_base_drops_headless_directive(self):
        # A flip headless -> visible must not leave the HEADLESS directive in
        # the handshake, but must retain the always-on base guidance
        # (set_headless owns the field in both directions).
        _server.set_headless(True)
        assert "HEADLESS" in _server.mcp._mcp_server.instructions
        _server.set_headless(False)
        assert _server._headless is False
        assert (
            _server.mcp._mcp_server.instructions == _server._BASE_INSTRUCTIONS
        )
        assert "HEADLESS" not in _server.mcp._mcp_server.instructions

    def test_server_status_reports_display_mode(self, server_with_host):
        _server.set_headless(True)
        out = _server.server_status()
        assert "headless (no viewer)" in out


# -----------------------------------------------------------------------
# execute_code
# -----------------------------------------------------------------------


class TestExecuteCode:
    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch):
        # Skip the inter-poll sleep so tests don't wait real seconds.
        monkeypatch.setattr(_server.time, "sleep", lambda *a, **k: None)

    def test_returns_error_when_no_host(self):
        _server._kernel_host = None
        result = _server.execute_code("print('hi')")
        assert "not initialized" in result

    def test_submits_code_via_job_runner(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id="job-1", status="running"
        )
        _server.set_promote_after(0.0)  # return a handle immediately
        result = _server.execute_code("print('hi')")
        snippet = server_with_host.execute.call_args_list[0][0][0]
        assert "_jobs.submit(" in snippet
        assert "print('hi')" in snippet  # code embedded via repr
        assert "job-1" in result  # job handle returned

    def test_inline_result_when_job_finishes_fast(self, server_with_host):
        # submit -> running, first poll -> terminal ok with output.
        server_with_host.execute.side_effect = [
            _job_reply(job_id="job-1", status="running"),
            _job_reply(**_snapshot(stdout="hello\n", result_text="3")),
        ]
        result = _server.execute_code("print('hello'); 1 + 2")
        assert "hello" in result
        assert "3" in result

    def test_no_output_message(self, server_with_host):
        server_with_host.execute.side_effect = [
            _job_reply(job_id="job-1", status="running"),
            _job_reply(**_snapshot(stdout="", result_text="")),
        ]
        result = _server.execute_code("x = 42")
        assert result == "(no output)"

    def test_error_path_includes_traceback(self, server_with_host):
        server_with_host.execute.side_effect = [
            _job_reply(job_id="job-1", status="running"),
            _job_reply(
                **_snapshot(
                    status="error",
                    error_text="Traceback...\nZeroDivisionError: division "
                    "by zero",
                )
            ),
        ]
        result = _server.execute_code("1 / 0")
        assert "division by zero" in result

    def test_promotes_to_job_handle_when_slow(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id="job-7", status="running"
        )
        _server.set_promote_after(0.0)
        result = _server.execute_code("while True: pass")
        assert "job-7" in result
        assert "still running" in result
        assert "poll_job" in result

    def test_busy_rejects_second_job(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            error="busy", running_job_id="job-3"
        )
        result = _server.execute_code("x = 1")
        assert "already running" in result
        assert "job-3" in result

    def test_submit_timeout_surfaces_error(self, server_with_host):
        # The quick submit snippet itself timed out (kernel main thread wedged).
        server_with_host.execute.return_value = _result(
            error_text="Execution exceeded 0.5s and was interrupted.",
            status="timeout",
        )
        result = _server.execute_code("x = 1")
        assert "interrupted" in result


class TestJobTools:
    def test_poll_job_formats_status(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            **_snapshot(status="running", stdout="step 1\n", elapsed=2.5)
        )
        result = _server.poll_job("job-1")
        assert "job-1: running" in result
        assert "step 1" in result

    def test_poll_job_unknown(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id="job-9", status="unknown", error_text=""
        )
        assert "No such job" in _server.poll_job("job-9")

    def test_cancel_job_requests_cancellation(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id="job-1", cancelled=True, status="running"
        )
        result = _server.cancel_job("job-1")
        assert "Cancellation requested" in result
        assert "restart_kernel" in result

    def test_cancel_job_nothing_to_cancel(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id="job-1", cancelled=False, status="ok"
        )
        assert "nothing to cancel" in _server.cancel_job("job-1")

    def test_job_tools_no_host(self):
        _server._kernel_host = None
        assert "not initialized" in _server.poll_job("job-1")
        assert "not initialized" in _server.cancel_job("job-1")


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

    def test_starting_kernel_skips_query(self, server_with_host):
        # Kernel still booting (launcher serves the handshake first): report the
        # state and do NOT query the kernel — execute() would block on readiness.
        server_with_host.health.return_value = {
            "alive": True,
            "ready": False,
            "start_error": None,
            "busy": False,
            "dead": False,
            "recent_respawns": 0,
            "watchdog_running": True,
        }
        result = _server.server_status()
        assert "ready: False" in result
        assert "starting" in result.lower()
        server_with_host.execute.assert_not_called()

    def test_dead_kernel_reports_only_dead_not_starting(
        self, server_with_host
    ):
        # When the watchdog marks the host dead, ready is also false. Report a
        # single DEAD state and return — not DEAD *and* a contradictory
        # "starting"/"failed" line.
        server_with_host.health.return_value = {
            "alive": False,
            "ready": False,
            "start_error": "respawn after unexpected death failed",
            "busy": False,
            "dead": True,
            "recent_respawns": 3,
            "watchdog_running": False,
        }
        result = _server.server_status()
        assert "DEAD" in result
        assert "starting" not in result.lower()
        assert "state: failed" not in result
        # The recorded reason rides along under DEAD (not as a second state).
        assert "respawn after unexpected death failed" in result
        server_with_host.execute.assert_not_called()

    def test_failed_startup_reports_error_not_starting(self, server_with_host):
        # A terminal bootstrap failure (start_error recorded) must read as
        # "failed" with the reason — not the generic "starting" that a slow but
        # progressing bring-up shows — so the two are distinguishable.
        server_with_host.health.return_value = {
            "alive": False,
            "ready": False,
            "start_error": "viewer absent: ImportError: no Qt platform",
            "busy": False,
            "dead": False,
            "recent_respawns": 0,
            "watchdog_running": False,
        }
        result = _server.server_status()
        assert "failed" in result.lower()
        assert "no Qt platform" in result
        assert "restart_kernel" in result
        assert "starting" not in result.lower()
        server_with_host.execute.assert_not_called()


# -----------------------------------------------------------------------
# Transport security (DNS-rebinding / Origin allowlist — review finding A2)
# -----------------------------------------------------------------------


class TestTransportSecurity:
    def test_protection_enabled_with_loopback_allowlist(self):
        ts = _server.mcp.settings.transport_security
        assert ts is not None
        assert ts.enable_dns_rebinding_protection is True
        assert "127.0.0.1:*" in ts.allowed_hosts
        assert "http://127.0.0.1:*" in ts.allowed_origins

    def test_middleware_rejects_forged_headers(self):
        from mcp.server.transport_security import (
            TransportSecurityMiddleware,
        )

        mw = TransportSecurityMiddleware(
            _server.mcp.settings.transport_security
        )
        assert mw._validate_origin("http://evil.com") is False
        assert mw._validate_origin("http://127.0.0.1:8765") is True
        assert mw._validate_host("evil.com") is False
        assert mw._validate_host("127.0.0.1:8765") is True

    def test_build_merges_extra_allowlist(self):
        ts = _server.build_transport_security(
            extra_origins=["https://proxy.example"],
            extra_hosts=["proxy.example"],
        )
        # extras present...
        assert "https://proxy.example" in ts.allowed_origins
        assert "proxy.example" in ts.allowed_hosts
        # ...without dropping the loopback defaults.
        assert "http://127.0.0.1:*" in ts.allowed_origins
        assert "127.0.0.1:*" in ts.allowed_hosts


# -----------------------------------------------------------------------
# Transport dispatch
# -----------------------------------------------------------------------


class TestRunStdio:
    def test_run_stdio_uses_stdio_transport(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(_server.mcp, "run", lambda **kw: calls.update(kw))
        _server.run_stdio()
        assert calls == {"transport": "stdio"}

    def test_run_http_uses_streamable_http(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(_server.mcp, "run", lambda **kw: calls.update(kw))
        _server.run(port=9999)
        assert calls == {"transport": "streamable-http"}
        # http binds loopback on the requested port.
        assert _server.mcp.settings.host == "127.0.0.1"
        assert _server.mcp.settings.port == 9999
