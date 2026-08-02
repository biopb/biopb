"""Tests for the MCP server tools and resources.

The tools dispatch into a child kernel; here that kernel is replaced by a
``mock_kernel_host`` returning canned ``execute`` result dicts, so the tests
exercise the server-side formatting/extraction without a real kernel.
"""

import base64
import json
import sys
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


def _job_reply(window_alive=True, **payload):
    """A kernel ``execute`` result whose stdout carries the job runner's
    single-line ``<<JOB_JSON>>`` payload.

    The snippet wraps the call result as ``{"r": <result>, "w": <window
    alive?>}``; ``payload`` becomes ``r`` and ``window_alive`` becomes ``w``.
    """
    envelope = {"r": payload, "w": window_alive}
    return _result(stdout=_server._JOB_DELIM + json.dumps(envelope) + "\n")


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
    old_skills = _server._skills_enabled
    old_instructions = _server.mcp._mcp_server.instructions
    yield
    _server._kernel_host = old_host
    _server._promote_after = old_promote
    _server._skills_enabled = old_skills
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
        "teardown_reason": None,
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
        content = _server.get_kernel_guide()
        assert "biopb-mcp" in content
        assert "execute_code" in content

    def test_guide_routes_every_requires_token_to_a_status_section(self):
        # The guide is where a skill's `checklist:` is resolved, so every token
        # kind must name the section that answers it -- a token with no route is
        # one the agent will guess at.
        guide = _server.get_kernel_guide()
        section = guide[guide.index("## Skill requirements") :]
        for token, where in [
            ("`viewer`", "## Viewer"),
            ("`tensor`", "## Tensor Server"),
            ("`dask`", "## Dask"),
            ("`ops:<kind>`", "## Ops"),
            ("`plugin:<name>`", "## Kernel plugins"),
            ("`pkg:biopb-mcp`", "## Versions"),
        ]:
            assert token in section and where in section

    def test_guide_gives_a_missing_package_three_options(self):
        # The choice is the user's, so all three have to be on the table: the
        # agent installing is one option among them, not the default, and the
        # degraded path is the one that survives a managed-env upgrade.
        section = _server.get_kernel_guide()
        section = section[section.index("### When something is missing") :]
        assert "They install it" in section
        assert "You install it for them" in section
        assert "only after they say yes" in section
        assert "degraded path" in section
        # A newly installed package is invisible to an interpreter that already
        # looked -- guidance that skips this reads as "the install didn't work".
        assert "invalidate_caches" in section

    def test_extras_file_is_advice_about_installing_not_about_not_installing(self):
        # The durability note belongs to the two options that install something.
        # Indented under option 3 -- the one where nothing is installed -- it reads
        # as a non-sequitur, so pin it as its own unindented paragraph.
        section = _server.get_kernel_guide()
        section = section[section.index("### When something is missing") :]
        (line,) = [ln for ln in section.splitlines() if "extra-packages.txt" in ln]
        assert not line.startswith(" "), line

    def test_guide_separates_the_three_missing_plugin_causes(self):
        # Seeding cannot fix an install that predates the plugin, and a file that
        # failed to load is not a file that is absent -- different fixes, so the
        # guide must not collapse them into "run the seeder".
        section = _server.get_kernel_guide()
        section = section[section.index("### When something is missing") :]
        assert "predates the plugin" in section
        assert "failed to load" in section
        assert "biopb-mcp-seed-plugins" in section

    def test_guide_skill_section_gated_on_the_catalog_switch(self):
        # With the catalog off there is no find_skills to hand back a
        # `checklist:`, so the section documents a tool the agent cannot
        # call -- the gate the handshake instructions already use.
        _server.set_skills_enabled(False)
        off = _server.get_kernel_guide()
        assert "## Skill requirements" not in off
        _server.set_skills_enabled(True)
        on = _server.get_kernel_guide()
        assert "## Skill requirements" in on
        # Everything else is the same guide, in both directions (no stale copy).
        assert on.startswith(off)
        _server.set_skills_enabled(False)
        assert _server.get_kernel_guide() == off

    def test_guide_points_at_server_status_for_which_plugins_loaded(self):
        # The loader is fail-open, so "file on disk" != "plugin loaded"; the
        # report is the only place that distinction is readable.
        content = _server.get_kernel_guide()
        assert "## Kernel plugins" in content
        assert "services.namespace_enabled" in content
        # ...and introspection remains the answer to the other question.
        assert "inspect_object" in content

    def test_viewer_resource_mentions_layers(self):
        content = _server.get_viewer_guide()
        assert "viewer.layers" in content

    def test_client_resource_mentions_client(self):
        content = _server.get_client_guide()
        assert "client" in content

    def test_viewer_resource_absorbed_the_annotation_guide(self):
        # guide://annotations was folded in here: one handle, one guide.
        content = _server.get_viewer_guide()
        assert "add_labels" in content
        assert "add_points" in content
        assert not hasattr(_server, "get_annotations_guide")


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
        server_with_host.execute.return_value = _result(stdout=f"<<PNG_B64>>{data}\n")

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
        server_with_host.execute.return_value = _result(stdout=f"<<PNG_B64>>{data}")
        _server.take_screenshot(canvas_only=False)
        snippet = server_with_host.execute.call_args[0][0]
        assert "canvas_only=False" in snippet

    def test_window_closed_returns_clear_message(self, server_with_host):
        server_with_host.execute.return_value = _result(
            stdout=_server._WINDOW_CLOSED_DELIM + "\n"
        )
        result = _server.take_screenshot()
        assert result[0].type == "text"
        assert "window was closed" in result[0].text
        assert "restart_kernel" in result[0].text


# -----------------------------------------------------------------------
# handshake instructions
# -----------------------------------------------------------------------


class TestInstructions:
    def test_base_instructions_carry_guardrails(self):
        # The operation guardrails must be delivered up front via the handshake
        # instructions (not left to a pull-on-demand resource).
        base = _server._BASE_INSTRUCTIONS
        assert "guardrails" in base.lower()
        assert "query_sources" in base
        assert "filesystem" in base.lower()
        # The catalog contract agents most often get wrong must be pushed up
        # front (return type + the real column name), not left to a pull-only
        # resource -- see also execute_code's docstring.
        assert 'format="pandas"' in base
        assert "source_url" in base
        # Skills stay a separate fragment: the base guidance must not point the
        # agent at find_skills, which returns nothing once the catalog is off.
        assert "find_skills" not in base
        # And the base alone is the handshake when skills are off.
        _server.set_skills_enabled(False)
        assert _server.mcp._mcp_server.instructions == base

    def test_module_default_mirrors_config_default(self):
        # The launcher always sets this from config, but the module literal is a
        # restated default -- pin it, since that is how it diverged once before.
        from biopb_mcp._config import DEFAULT_CONFIG

        assert _server._skills_enabled is DEFAULT_CONFIG["services"]["skills_enabled"]

    def test_skills_directive_gated_on_enable(self):
        # Off: no find_skills mention in the handshake.
        _server.set_skills_enabled(False)
        assert "find_skills" not in _server.mcp._mcp_server.instructions
        # On: the skills fragment is appended to the base guidance.
        _server.set_skills_enabled(True)
        instr = _server.mcp._mcp_server.instructions
        assert instr.startswith(_server._BASE_INSTRUCTIONS)
        assert "find_skills" in instr
        assert "skill://" in instr
        # Back off: no stale fragment left behind.
        _server.set_skills_enabled(False)
        assert _server.mcp._mcp_server.instructions == _server._BASE_INSTRUCTIONS


# -----------------------------------------------------------------------
# execute_code
# -----------------------------------------------------------------------


class TestExecuteCode:
    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch):
        # Skip the inter-poll sleep so tests don't wait real seconds.
        monkeypatch.setattr(_server.time, "sleep", lambda *a, **k: None)

    def test_docstring_carries_catalog_contract(self):
        # The tool description is always in the model's context, unlike the
        # pull-only guide:// resources; the high-failure catalog facts must
        # live here so the agent sees them at the point of action.
        doc = _server.execute_code.__doc__ or _server.execute_code.fn.__doc__
        assert "source_url" in doc
        assert 'format="pandas"' in doc
        assert "add_tensor" in doc

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
                    error_text="Traceback...\nZeroDivisionError: division by zero",
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

    def test_inline_result_appends_window_closed_note(self, server_with_host):
        server_with_host.execute.side_effect = [
            _job_reply(job_id="job-1", status="running", window_alive=False),
            _job_reply(window_alive=False, **_snapshot(stdout="done\n")),
        ]
        result = _server.execute_code("viewer.add_image(arr)")
        assert "done" in result
        assert "viewer window is closed" in result
        assert "restart_kernel" in result

    def test_job_handle_appends_window_closed_note(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id="job-7", status="running", window_alive=False
        )
        _server.set_promote_after(0.0)
        result = _server.execute_code("while True: pass")
        assert "job-7" in result
        assert "viewer window is closed" in result


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

    def test_poll_job_terminal_appends_window_closed_note(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            window_alive=False, **_snapshot(status="ok", stdout="done\n")
        )
        result = _server.poll_job("job-1")
        assert "viewer window is closed" in result

    def test_poll_job_running_omits_window_note(self, server_with_host):
        # A still-running job: no terminal result yet, so no closed-window note.
        server_with_host.execute.return_value = _job_reply(
            window_alive=False, **_snapshot(status="running", stdout="step\n")
        )
        result = _server.poll_job("job-1")
        assert "viewer window is closed" not in result

    def test_job_tools_no_host(self):
        _server._kernel_host = None
        assert "not initialized" in _server.poll_job("job-1")


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
    def test_interrupt_forces_running_job(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id="job-3", interrupted=True
        )
        result = _server.interrupt_kernel()
        snippet = server_with_host.execute.call_args[0][0]
        assert "interrupt_current(" in snippet
        assert "job-3" in result

    def test_interrupt_no_running_job(self, server_with_host):
        server_with_host.execute.return_value = _job_reply(
            job_id=None, interrupted=False
        )
        assert "No running job" in _server.interrupt_kernel()

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


class TestStartKernel:
    def test_ready_state_message(self, server_with_host):
        # ensure_started is synchronous: a ready result means the kernel is up.
        server_with_host.ensure_started.return_value = {"state": "ready"}
        result = _server.start_kernel()
        server_with_host.ensure_started.assert_called_once()
        assert "ready" in result.lower()
        assert "execute_code" in result

    def test_error_state_message(self, server_with_host):
        server_with_host.ensure_started.return_value = {
            "state": "error",
            "error": "no Qt platform",
        }
        result = _server.start_kernel()
        assert "failed to start" in result.lower()
        assert "no Qt platform" in result
        assert "start_kernel" in result  # retry guidance

    def test_no_host(self):
        _server._kernel_host = None
        assert "not initialized" in _server.start_kernel()

    def test_execute_code_when_not_started_points_to_start_kernel(
        self, server_with_host
    ):
        # A kernel-dependent tool funnels through host.execute(); a not_started
        # status must surface the "call start_kernel" guidance verbatim.
        server_with_host.execute.return_value = _result(
            status="not_started",
            error_text=(
                "Kernel not started. Call start_kernel first, then poll "
                "server_status until it reports ready."
            ),
        )
        result = _server.execute_code("1 + 1")
        assert "start_kernel" in result


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

    def test_reports_observe_disabled_by_default(self, server_with_host, monkeypatch):
        from biopb_mcp.mcp import _observe

        monkeypatch.setattr(_observe, "_mounted_http", False)
        result = _server.server_status()
        assert "## Observe" in result
        assert "not running" in result

    def test_reports_observe_url_when_running(self, server_with_host, monkeypatch):
        from biopb_mcp.mcp import _observe

        monkeypatch.setattr(_observe, "_mounted_http", True)
        result = _server.server_status()
        assert "## Observe" in result
        # The observe page is served by the control front; this child hosts only
        # the /api/* it calls, so server_status points at the API mount.
        assert "/api" in result
        assert "http://127.0.0.1:" in result

    def test_reports_observe_even_when_kernel_not_initialized(self, monkeypatch):
        from biopb_mcp.mcp import _observe

        # Observe is server-process state -> reported despite no kernel.
        _server._kernel_host = None
        monkeypatch.setattr(_observe, "_mounted_http", True)
        result = _server.server_status()
        assert "## Observe" in result
        assert "/api" in result

    def test_starting_kernel_skips_query(self, server_with_host):
        # Kernel still booting (launcher serves the handshake first): report the
        # state and do NOT query the kernel — execute() would block on readiness.
        server_with_host.health.return_value = {
            "alive": True,
            "ready": False,
            "start_error": None,
            "teardown_reason": None,
            "busy": False,
            "dead": False,
            "recent_respawns": 0,
            "watchdog_running": True,
        }
        result = _server.server_status()
        assert "ready: False" in result
        # alive but not ready -> booting (e.g. a watchdog respawn).
        assert "starting" in result.lower()
        server_with_host.execute.assert_not_called()

    def test_kernel_snippet_reports_ops_and_plugins(self):
        # The snippet runs *in* the kernel, so run it against a stand-in namespace:
        # it is what an agent resolves a skill's `ops:` / `plugin:` against, and
        # every section has to survive the same exec.
        import contextlib
        import io

        from biopb_mcp.mcp import _requires

        _requires.record_loaded_plugins(["rolling_ball"], ["labshop_tools"])
        ns = {
            "_dask_client": None,
            "_dask_attach_done": True,
            "_conn": MagicMock(client=None, last_status="", last_message=""),
            "viewer": MagicMock(layers=[]),
            "_viewer_window_alive": lambda: True,
            "ops": {"segmentation": object(), "restoration": object()},
            "_jobs": MagicMock(jobs_summary=list),
        }
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(_server._STATUS_SNIPPET, ns)  # noqa: S102 - the canned snippet
        report = out.getvalue()

        assert "## Ops\n  restoration, segmentation" in report
        assert "## Kernel plugins" in report
        assert "files: rolling_ball" in report
        assert "packages: labshop_tools" in report
        # `pkg:biopb-mcp>=X` (a skill needing a release-carried plugin) is
        # answered here, from the kernel's own interpreter, not by an import.
        import biopb_mcp

        assert "biopb-mcp: " + biopb_mcp.__version__ in report
        # The interpreter, not just its version: a bare `pip install` targets the
        # user's active env, which need not be this one. Which command is right is
        # decided in _requires (tested there for both env shapes); here, only that
        # the section reaches the report at all.
        assert sys.executable in report
        assert "add a package" in report

    def test_kernel_snippet_names_the_config_key_when_no_ops(self):
        import contextlib
        import io

        ns = {
            "_dask_client": None,
            "_dask_attach_done": True,
            "_conn": MagicMock(client=None, last_status="", last_message=""),
            "viewer": MagicMock(layers=[]),
            "_viewer_window_alive": lambda: True,
            "ops": {},
            "_jobs": MagicMock(jobs_summary=list),
        }
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(_server._STATUS_SNIPPET, ns)  # noqa: S102 - the canned snippet
        assert "services.process_image_servers" in out.getvalue()

    def test_idle_kernel_reports_not_started(self, server_with_host):
        # Not alive and not ready (never started / torn down): point the agent
        # at start_kernel, don't query the kernel.
        server_with_host.health.return_value = {
            "alive": False,
            "ready": False,
            "start_error": None,
            "teardown_reason": "the user closed the napari viewer window",
            "busy": False,
            "dead": False,
            "recent_respawns": 0,
            "watchdog_running": False,
        }
        result = _server.server_status()
        assert "not started" in result.lower()
        assert "start_kernel" in result
        assert "napari viewer window" in result  # teardown attribution
        server_with_host.execute.assert_not_called()

    def test_dead_kernel_reports_only_dead_not_starting(self, server_with_host):
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
        assert "start_kernel" in result
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

        mw = TransportSecurityMiddleware(_server.mcp.settings.transport_security)
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


class TestRun:
    def test_no_stdio_serving_in_this_process(self):
        # Direction 1: this process serves http only; stdio is the launcher's
        # bridge (`_shim`), not a second serving path here.
        assert not hasattr(_server, "run_stdio")

    def test_run_http_uses_streamable_http(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(_server.mcp, "run", lambda **kw: calls.update(kw))
        _server.run(port=9999)
        assert calls == {"transport": "streamable-http"}
        # http binds loopback on the requested port.
        assert _server.mcp.settings.host == "127.0.0.1"
        assert _server.mcp.settings.port == 9999


# -----------------------------------------------------------------------
# guide://data
# -----------------------------------------------------------------------


class TestDataGuide:
    """The data-representation guide, and the places that must point at it.

    Layer data here is a pyramid of proxies in display axis order, none of which
    a napari-shaped habit expects -- so the guide has to be discoverable from the
    handshake and from every guide whose examples touch pixels.
    """

    def test_registered_and_advertised_in_the_handshake(self):
        import asyncio

        uris = {str(r.uri) for r in asyncio.run(_server.mcp.list_resources())}
        assert "guide://data" in uris
        # Pull-only resources are read on demand, so the instructions are the
        # only place the agent learns this one exists.
        assert "guide://data" in _server._BASE_INSTRUCTIONS

    def test_names_all_three_sources_of_array_data(self):
        guide = _server._resources.DATA
        assert "client.get_tensor" in guide  # the server
        assert "layer.data" in guide  # the viewer
        assert "multiscale" in guide  # ...which may be a list of levels

    def test_pairs_each_scale_with_the_array_it_belongs_to(self):
        # The two scale vectors sit on the same axes now, so crossing them no
        # longer transposes anything -- but for interleaved colour layer.scale
        # is one shorter, so the guide must still name both.
        guide = _server._resources.DATA
        assert "get_physical_scale" in guide
        assert "layer.scale" in guide

    def test_viewer_guide_reads_layer_data_the_safe_way(self):
        # The layer-listing example is the snippet most likely to be copied;
        # a bare `layer.data.shape` breaks on every multiscale layer.
        viewer_guide = _server._resources.VIEWER
        assert "layer.data.shape" not in viewer_guide
        assert "layer.multiscale" in viewer_guide
