"""Tests for the MCP server tools and resources.

The tools dispatch into a child kernel; here that kernel is replaced by a
``mock_kernel_host`` returning canned ``execute`` result dicts, so the tests
exercise the server-side formatting/extraction without a real kernel.
"""

import asyncio
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


def _job_envelope(r, window_alive=True):
    """A kernel ``execute`` result whose stdout carries the job runner's
    single-line ``<<JOB_JSON>>`` payload, with *r* as the call's return value.

    The snippet wraps the call result as ``{"r": <result>, "w": <window
    alive?>}``.
    """
    return _result(
        stdout=_server._JOB_DELIM + json.dumps({"r": r, "w": window_alive}) + "\n"
    )


def _job_reply(window_alive=True, **payload):
    """:func:`_job_envelope` for the common case where the call returns a dict
    (a job snapshot, a submit result): ``payload`` becomes ``r``."""
    return _job_envelope(payload, window_alive=window_alive)


def _install_replies(host, *, returns=None, queue=None, digest=()):
    """Install kernel replies that dispatch on the *snippet*, not on call order.

    Agent-facing tools carry a user-activity digest round-trip
    (``_server._foreign_activity_note``) alongside the call each test is actually
    about. Answering that by content — rather than letting it consume a slot in
    an ordered ``side_effect`` list — keeps every test's queue one-to-one with
    the calls it asserts on, so an auxiliary round-trip can be added or removed
    without renumbering unrelated tests.

    ``queue`` is consumed in order; ``returns`` answers anything after it (or
    everything, if no queue). ``digest`` is the user-job list the digest call
    returns — empty by default, i.e. "the user ran nothing".
    """
    pending = list(queue or [])

    def execute(code, *_args, **_kwargs):
        if "_jobs.foreign_digest(" in code:
            return _job_envelope(list(digest))
        if "_jobs.ack_foreign_digest(" in code:
            return _job_envelope(0)
        if pending:
            return pending.pop(0)
        return returns if returns is not None else _result()

    host.execute.side_effect = execute
    return host


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
    # The mirrored one-agent claim is process state like the rest: a test that
    # claims the kernel must not decide whether the next one is refused.
    _server.clear_claim()


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

    def test_guide_tells_the_agent_it_shares_the_namespace(self):
        # The runtime note ("the user ran job-N") says a change happened; this
        # section is what makes that legible -- without it the agent has no model
        # of a second writer, and reads the note as noise.
        guide = _server.get_kernel_guide()
        section = guide[guide.index("## You are not the only writer") :]
        assert "observe" in section
        assert "poll_job" in section
        # The three rules that keep the two writers off each other: it is told
        # after the fact, it waits when busy, and it does not stop their cell --
        # including the workaround it would otherwise reach for.
        assert "rejected as busy" in section
        assert "refuses a user job" in section
        assert "restart_kernel" in section

    def test_guide_skill_section_gated_on_the_catalog_switch(self):
        # With the catalog off there is no list_skills to hand back a
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
        # agent at list_skills, which returns nothing once the catalog is off.
        assert "list_skills" not in base
        # And the base alone is the handshake when skills are off.
        _server.set_skills_enabled(False)
        assert _server.mcp._mcp_server.instructions == base

    def test_module_default_mirrors_config_default(self):
        # The launcher always sets this from config, but the module literal is a
        # restated default -- pin it, since that is how it diverged once before.
        from biopb_mcp._config import DEFAULT_CONFIG

        assert _server._skills_enabled is DEFAULT_CONFIG["services"]["skills_enabled"]

    def test_skills_directive_gated_on_enable(self):
        # Off: no list_skills mention in the handshake.
        _server.set_skills_enabled(False)
        assert "list_skills" not in _server.mcp._mcp_server.instructions
        # On: the skills fragment is appended to the base guidance.
        _server.set_skills_enabled(True)
        instr = _server.mcp._mcp_server.instructions
        assert instr.startswith(_server._BASE_INSTRUCTIONS)
        assert "list_skills" in instr
        assert "skill://" in instr
        # Back off: no stale fragment left behind.
        _server.set_skills_enabled(False)
        assert _server.mcp._mcp_server.instructions == _server._BASE_INSTRUCTIONS


# -----------------------------------------------------------------------
# verify_workflow
# -----------------------------------------------------------------------


def _verify_snapshot(status="ok", cells=None, added_layers=(), **kw):
    """A job snapshot carrying a verification record, as the kernel returns it."""
    cells = cells or [
        {
            "code": "a = 2",
            "status": "ok",
            "stdout": "",
            "result_text": "",
            "error_text": "",
            "elapsed": 0.1,
        }
    ]
    snap = _snapshot(status="ok" if status == "ok" else "error", **kw)
    snap["verify"] = {
        "title": "Count foci per cell",
        "created": 1_700_000_000.0,
        "status": status,
        "added_layers": list(added_layers),
        "cells": cells,
    }
    return snap


class TestVerifyWorkflow:
    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch):
        monkeypatch.setattr(_server.time, "sleep", lambda *a, **k: None)

    def test_returns_error_when_no_host(self):
        _server._kernel_host = None
        assert "not initialized" in _server.verify_workflow(["1"])

    def test_no_cells_is_refused_before_the_kernel_is_touched(self, server_with_host):
        assert "at least one cell" in _server.verify_workflow([])
        assert not server_with_host.execute.called

    def test_cells_and_title_ride_the_submit_snippet(self, server_with_host):
        # The runner is in the kernel, so the record only exists if both are
        # marshaled into the snippet -- repr'd, like the code execute_code sends.
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-1", status="running")
        )
        _server.set_promote_after(0.0)
        _server.verify_workflow(["a = 2", "print(a)"], title="Count foci")
        (snippet,) = [
            c[0][0]
            for c in server_with_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        assert "verify_cells=['a = 2', 'print(a)']" in snippet
        assert "verify_title='Count foci'" in snippet

    def test_a_clean_run_reports_the_verdict_and_where_to_save(self, server_with_host):
        _install_replies(
            server_with_host,
            queue=[_job_reply(job_id="job-1", status="running")],
            returns=_job_reply(**_verify_snapshot()),
        )
        _server.set_promote_after(1.0)
        result = _server.verify_workflow(["a = 2"], title="Count foci")
        assert "Verified" in result
        assert "scratch namespace" in result
        # The agent must hand the save to the user, not write a file itself.
        assert "Save workflow" in result

    def test_a_failure_names_the_cell_and_quotes_its_traceback(self, server_with_host):
        cells = [
            {
                "code": "a = 2",
                "status": "ok",
                "stdout": "",
                "result_text": "",
                "error_text": "",
                "elapsed": 0.1,
            },
            {
                "code": "print(leftover)",
                "status": "error",
                "stdout": "",
                "result_text": "",
                "error_text": "NameError: name 'leftover' is not defined",
                "elapsed": 0.0,
            },
            {
                "code": "print('never')",
                "status": "skipped",
                "stdout": "",
                "result_text": "",
                "error_text": "",
                "elapsed": 0.0,
            },
        ]
        _install_replies(
            server_with_host,
            queue=[_job_reply(job_id="job-1", status="running")],
            returns=_job_reply(**_verify_snapshot(status="error", cells=cells)),
        )
        _server.set_promote_after(1.0)
        result = _server.verify_workflow([c["code"] for c in cells])
        assert "NOT verified" in result
        assert "cell 2" in result
        assert "leftover" in result
        # The cascade is named as skipped rather than silently absent.
        assert "skipped" in result

    def test_added_layers_are_reported_back(self, server_with_host):
        _install_replies(
            server_with_host,
            queue=[_job_reply(job_id="job-1", status="running")],
            returns=_job_reply(**_verify_snapshot(added_layers=["foci"])),
        )
        _server.set_promote_after(1.0)
        result = _server.verify_workflow(["a = 2"])
        assert "foci" in result
        assert "isolates variables, not the viewer" in result

    def test_a_kernel_without_the_record_falls_back_to_the_job_result(
        self, server_with_host
    ):
        # An older kernel, or a submit that never built one: report the job the
        # ordinary way rather than invent a verdict.
        _install_replies(
            server_with_host,
            queue=[_job_reply(job_id="job-1", status="running")],
            returns=_job_reply(**_snapshot(status="ok", stdout="hi\n")),
        )
        _server.set_promote_after(1.0)
        assert "hi" in _server.verify_workflow(["a = 2"])

    def test_a_long_verification_hands_back_a_job_handle(self, server_with_host):
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-1", status="running")
        )
        _server.set_promote_after(0.0)
        result = _server.verify_workflow(["a = 2"])
        assert "still running" in result and "poll_job('job-1')" in result


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
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-1", status="running")
        )
        _server.set_promote_after(0.0)  # return a handle immediately
        result = _server.execute_code("print('hi')")
        # By content, not by position: the tool also carries the user-activity
        # digest round-trip, so "the first call" is not the submit.
        (snippet,) = [
            c[0][0]
            for c in server_with_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        assert "print('hi')" in snippet  # code embedded via repr
        assert "job-1" in result  # job handle returned

    def test_intent_rides_the_submit_snippet(self, server_with_host):
        # The job runner lives in the kernel, so the field only reaches the
        # record if it is marshaled into the submit snippet -- and it must be
        # repr'd like the code, since it is arbitrary user-supplied text.
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-1", status="running")
        )
        _server.set_promote_after(0.0)
        _server.execute_code("x = 1", intent="isolate the nuclei channel")
        (snippet,) = [
            c[0][0]
            for c in server_with_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        assert "intent='isolate the nuclei channel'" in snippet

    def test_refusal_when_another_client_holds_the_kernel(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(error="not_owner", owner="claude-code"),
        )
        result = _server.execute_code("x = 1")
        assert "already in use by another client (claude-code)" in result
        # It must be told what still works, or it reads the refusal as a broken
        # kernel...
        assert "poll_job" in result
        # ...and it must not be pointed at restart_kernel, which is refused for
        # the same reason: naming it here would send the agent to try anyway.
        assert "restart_kernel" not in result
        assert "the user's to do" in result

    def test_writer_identity_rides_the_submit_snippet(self, server_with_host):
        # Outside a request there is no client, so nothing is claimed -- the
        # kernel reads writer=None as "nothing to tell two callers apart with".
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-1", status="running")
        )
        _server.set_promote_after(0.0)
        _server.execute_code("x = 1")
        (snippet,) = [
            c[0][0]
            for c in server_with_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        assert "writer=None" in snippet

    def test_intent_is_optional(self, server_with_host):
        # Every existing MCP client calls execute_code with one argument.
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-1", status="running")
        )
        _server.set_promote_after(0.0)
        _server.execute_code("x = 1")
        (snippet,) = [
            c[0][0]
            for c in server_with_host.execute.call_args_list
            if "_jobs.submit(" in c[0][0]
        ]
        assert "intent=''" in snippet

    def test_inline_result_when_job_finishes_fast(self, server_with_host):
        # submit -> running, first poll -> terminal ok with output.
        _install_replies(
            server_with_host,
            queue=[
                _job_reply(job_id="job-1", status="running"),
                _job_reply(**_snapshot(stdout="hello\n", result_text="3")),
            ],
        )
        result = _server.execute_code("print('hello'); 1 + 2")
        assert "hello" in result
        assert "3" in result

    def test_no_output_message(self, server_with_host):
        _install_replies(
            server_with_host,
            queue=[
                _job_reply(job_id="job-1", status="running"),
                _job_reply(**_snapshot(stdout="", result_text="")),
            ],
        )
        result = _server.execute_code("x = 42")
        assert result == "(no output)"

    def test_error_path_includes_traceback(self, server_with_host):
        _install_replies(
            server_with_host,
            queue=[
                _job_reply(job_id="job-1", status="running"),
                _job_reply(
                    **_snapshot(
                        status="error",
                        error_text="Traceback...\nZeroDivisionError: division by zero",
                    )
                ),
            ],
        )
        result = _server.execute_code("1 / 0")
        assert "division by zero" in result

    def test_promotes_to_job_handle_when_slow(self, server_with_host):
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-7", status="running")
        )
        _server.set_promote_after(0.0)
        result = _server.execute_code("while True: pass")
        assert "job-7" in result
        assert "still running" in result
        assert "poll_job" in result

    def test_busy_rejects_second_job(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(error="busy", running_job_id="job-3"),
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
        _install_replies(
            server_with_host,
            queue=[
                _job_reply(job_id="job-1", status="running", window_alive=False),
                _job_reply(window_alive=False, **_snapshot(stdout="done\n")),
            ],
        )
        result = _server.execute_code("viewer.add_image(arr)")
        assert "done" in result
        assert "viewer window is closed" in result
        assert "restart_kernel" in result

    def test_job_handle_appends_window_closed_note(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(job_id="job-7", status="running", window_alive=False),
        )
        _server.set_promote_after(0.0)
        result = _server.execute_code("while True: pass")
        assert "job-7" in result
        assert "viewer window is closed" in result


class TestJobTools:
    def test_poll_job_formats_status(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(
                **_snapshot(status="running", stdout="step 1\n", elapsed=2.5)
            ),
        )
        result = _server.poll_job("job-1")
        assert "job-1: running" in result
        assert "step 1" in result

    def test_poll_job_unknown(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(job_id="job-9", status="unknown", error_text=""),
        )
        assert "No such job" in _server.poll_job("job-9")

    def test_poll_job_terminal_appends_window_closed_note(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(
                window_alive=False, **_snapshot(status="ok", stdout="done\n")
            ),
        )
        result = _server.poll_job("job-1")
        assert "viewer window is closed" in result

    def test_poll_job_running_omits_window_note(self, server_with_host):
        # A still-running job: no terminal result yet, so no closed-window note.
        _install_replies(
            server_with_host,
            returns=_job_reply(
                window_alive=False, **_snapshot(status="running", stdout="step\n")
            ),
        )
        result = _server.poll_job("job-1")
        assert "viewer window is closed" not in result

    def test_job_tools_no_host(self):
        _server._kernel_host = None
        assert "not initialized" in _server.poll_job("job-1")


# -----------------------------------------------------------------------
# inspect_object
# -----------------------------------------------------------------------


class TestUserActivityNote:
    """The agent's notice that a human wrote to its namespace.

    See ``docs/user-console.md``: the user runs cells through the same job
    runner, so the agent's picture of the namespace can go stale between calls
    with nothing in its own results to say so.
    """

    _DIGEST = [
        {"job_id": "job-7", "status": "ok", "elapsed": 1.0},
        {"job_id": "job-8", "status": "error", "elapsed": 2.0},
    ]

    def test_no_note_when_the_user_ran_nothing(self, server_with_host):
        _install_replies(server_with_host, returns=_job_reply(**_snapshot()))
        assert _server._foreign_activity_note(server_with_host) == ""

    def test_note_lists_the_jobs_and_points_at_poll_job(self, server_with_host):
        _install_replies(server_with_host, digest=self._DIGEST)
        note = _server._foreign_activity_note(server_with_host)
        assert "job-7 (ok)" in note and "job-8 (error)" in note
        # No job id in the instruction: pointing at one of several invites the
        # agent to read that one, call the notice discharged, and never see the
        # rest -- which it is not offered again.
        assert "poll_job" in note
        assert "poll_job('" not in note
        # Says *that* something changed and where to look -- not what changed,
        # which would be a second thing to keep true.
        assert "re-check" in note

    def test_ack_is_a_second_call_naming_only_terminal_jobs(self, server_with_host):
        # The read must not ack: execute_interactive sends before it starts its
        # timeout clock, so a probe that times out still runs at the kernel
        # later -- acking inside it would retire a notice nobody received.
        running = {"job_id": "job-9", "status": "running", "elapsed": 1.0}
        _install_replies(server_with_host, digest=[*self._DIGEST, running])
        _server._foreign_activity_note(server_with_host)
        calls = [c[0][0] for c in server_with_host.execute.call_args_list]
        # Named point of view: the digest is read as whoever is asking, so an
        # MCP client is not handed the chat loop's cells (or its own).
        assert any("_jobs.foreign_digest('mcp')" in c for c in calls)
        (ack,) = [c for c in calls if "ack_foreign_digest(" in c]
        # Terminal ones only: a job reported `running` was not given its final
        # status, so it must stay pending.
        assert "'job-7'" in ack and "'job-8'" in ack
        assert "job-9" not in ack

    def test_no_ack_when_there_is_nothing_to_report(self, server_with_host):
        _install_replies(server_with_host, digest=[])
        assert _server._foreign_activity_note(server_with_host) == ""
        calls = [c[0][0] for c in server_with_host.execute.call_args_list]
        assert not [c for c in calls if "ack_foreign_digest(" in c]

    def test_note_says_a_repeat_is_not_a_new_cell(self, server_with_host):
        # foreign_digest re-reports a still-running cell every round trip, so the
        # wording must not read as "another cell ran since last time" -- an
        # agent polling a 5-minute user cell would re-verify on every poll.
        _install_replies(
            server_with_host,
            digest=[{"job_id": "job-9", "status": "running", "elapsed": 2.0}],
        )
        note = _server._foreign_activity_note(server_with_host)
        assert "since your last call" not in note
        assert "repeats until it ends" in note

    def test_malformed_digest_yields_no_note(self, server_with_host):
        # Auxiliary, like the window-liveness probe: it must never break the
        # result the agent actually asked for.
        for bad in ("not-a-list", [{"no_job_id": 1}], [None]):
            server_with_host.execute.side_effect = None
            server_with_host.execute.return_value = _job_envelope(bad)
            assert _server._foreign_activity_note(server_with_host) == ""

    def test_unreachable_kernel_yields_no_note(self, server_with_host):
        # Nothing is acked on this path either, so the notice is deferred to the
        # next call rather than dropped.
        server_with_host.execute.side_effect = None
        server_with_host.execute.return_value = _result(status="busy")
        assert _server._foreign_activity_note(server_with_host) == ""

    def test_execute_code_carries_the_note(self, server_with_host):
        _install_replies(
            server_with_host,
            queue=[
                _job_reply(job_id="job-1", status="running"),
                _job_reply(**_snapshot(stdout="done\n")),
            ],
            digest=self._DIGEST,
        )
        result = _server.execute_code("x = 1")
        assert "done" in result  # the agent's own result still leads
        assert "job-7 (ok)" in result

    def test_a_non_owners_read_does_not_discharge_the_notice(self, server_with_host):
        # poll_job is open to a watching client, but the ack must carry that
        # client's id so the kernel can refuse it: retiring a notice the holder
        # never received is the one failure the read/ack split exists to prevent.
        _install_replies(
            server_with_host,
            returns=_job_reply(**_snapshot(status="ok")),
            digest=self._DIGEST,
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(_server, "_client_identity", lambda: ("sess-B", "other"))
            _server.poll_job("job-1")
        (snippet,) = [
            c[0][0]
            for c in server_with_host.execute.call_args_list
            if "ack_foreign_digest(" in c[0][0]
        ]
        assert "writer='sess-B'" in snippet

    def test_poll_job_carries_the_note(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(**_snapshot(status="ok", stdout="out\n")),
            digest=self._DIGEST,
        )
        assert "job-7 (ok)" in _server.poll_job("job-1")

    def test_busy_on_a_user_cell_tells_the_agent_to_wait(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(
                error="busy", running_job_id="job-9", running_job_origin="user"
            ),
        )
        result = _server.execute_code("x = 1")
        assert "The user is running a cell" in result
        assert "job-9" in result
        # The agent must not be pointed at interrupt_kernel here: it would be
        # refused, and the suggestion alone invites it to try.
        assert "interrupt_kernel" not in result
        assert "restart_kernel" not in result

    def test_busy_on_a_chat_cell_does_not_call_it_the_user(self, server_with_host):
        # Same refusal, different writer: the advice must not attribute a chat
        # agent's cell to the person sitting there.
        _install_replies(
            server_with_host,
            returns=_job_reply(
                error="busy", running_job_id="job-9", running_job_origin="chat"
            ),
        )
        result = _server.execute_code("x = 1")
        assert "Another writer is running a cell" in result
        assert "The user" not in result
        assert "interrupt_kernel" not in result

    def test_note_names_the_writer_when_it_is_not_the_user(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(**_snapshot(status="ok", stdout="out\n")),
            digest=[{"job_id": "job-7", "status": "ok", "origin": "chat"}],
        )
        result = _server.poll_job("job-1")
        assert "Another writer ran code" in result
        assert "job-7 (ok, chat)" in result

    def test_busy_on_its_own_job_keeps_the_stop_advice(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(
                error="busy", running_job_id="job-3", running_job_origin="mcp"
            ),
        )
        result = _server.execute_code("x = 1")
        assert "already running" in result
        assert "interrupt_kernel" in result


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

    def test_interrupt_asks_as_the_agent(self, server_with_host):
        # The requester is what lets the runner refuse a user's cell; without it
        # the refusal below can never trigger.
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-3", interrupted=True)
        )
        _server.interrupt_kernel()
        (snippet,) = [
            c[0][0]
            for c in server_with_host.execute.call_args_list
            if "interrupt_current(" in c[0][0]
        ]
        assert "requester='mcp'" in snippet

    def test_interrupt_refused_when_another_client_holds_the_kernel(
        self, server_with_host
    ):
        _install_replies(
            server_with_host,
            returns=_job_reply(
                job_id="job-3",
                interrupted=False,
                status="running",
                refused="not_owner",
            ),
        )
        result = _server.interrupt_kernel()
        assert "already in use by another client" in result
        # The recovery named must be the person, not restart_kernel -- which is
        # refused for the same reason and would read as the way around this.
        assert "the user's to do" in result

    def test_restart_refused_when_another_client_holds_the_kernel(
        self, server_with_host
    ):
        # The holder is learned from the kernel *accepting* its code, then
        # mirrored here -- see _claimed_by.
        _install_replies(
            server_with_host, returns=_job_reply(job_id="job-1", status="running")
        )
        _server.set_promote_after(0.0)
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(_server, "_client_identity", lambda: ("sess-A", "claude-code"))
            _server.execute_code("x = 1")
        assert _server._claimed_by == "sess-A"

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(_server, "_client_identity", lambda: ("sess-B", "other"))
            result = _server.restart_kernel()
        assert "already in use by another client" in result
        server_with_host.restart.assert_not_called()

        # The holder itself is not blocked, and a fresh kernel is unclaimed.
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(_server, "_client_identity", lambda: ("sess-A", "claude-code"))
            assert "Kernel restarted" in _server.restart_kernel()
        server_with_host.restart.assert_called_once()
        assert _server._claimed_by is None

    def test_a_lost_submit_reply_still_leaves_the_kernel_claimed(
        self, server_with_host
    ):
        # execute_interactive hands the request over before it starts its clock,
        # so a timed-out submit still runs -- the kernel claims and starts the
        # job while this process sees nothing come back. Recording the claim only
        # on the way back would leave the mirror empty and let a stranger restart
        # the session that just began.
        _install_replies(server_with_host, returns=_result(status="timeout"))
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(_server, "_client_identity", lambda: ("sess-A", "claude-code"))
            _server.execute_code("x = 1")
        assert _server._claimed_by == "sess-A"

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(_server, "_client_identity", lambda: ("sess-B", "other"))
            assert "already in use" in _server.restart_kernel()
        server_with_host.restart.assert_not_called()

    def test_a_refusal_corrects_a_mirror_that_guessed_wrong(self, server_with_host):
        # The presumed claim is only a guess when this process has seen none. The
        # kernel's refusal names the real holder, and that must win -- otherwise
        # a stranger's first call would leave itself recorded as the owner.
        _install_replies(
            server_with_host,
            returns=_job_reply(
                error="not_owner", owner="claude-code", owner_id="sess-A"
            ),
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(_server, "_client_identity", lambda: ("sess-B", "other"))
            assert "already in use" in _server.execute_code("x = 1")
        assert _server._claimed_by == "sess-A"

    def test_a_known_holder_is_not_overwritten_by_a_stranger(self, server_with_host):
        # The presumption is guarded on "no claim seen yet"; a stranger arriving
        # after the holder is known must not take the mirror even for the length
        # of one call, since a lost reply would freeze it that way.
        _server._claimed_by = "sess-A"
        try:
            _install_replies(server_with_host, returns=_result(status="timeout"))
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr(_server, "_client_identity", lambda: ("sess-B", "other"))
                _server.execute_code("x = 1")
            assert _server._claimed_by == "sess-A"
        finally:
            _server.clear_claim()

    def test_restart_is_not_gated_on_a_kernel_round_trip(self, server_with_host):
        # A busy kernel must never read as an unclaimed one: asking it who owns
        # it would fail *open* exactly when the holder has a job running, which
        # is when a stray restart costs the most.
        _install_replies(server_with_host, returns=_result(status="busy"))
        _server._claimed_by = "sess-A"
        try:
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr(_server, "_client_identity", lambda: ("sess-B", "other"))
                result = _server.restart_kernel()
        finally:
            _server.clear_claim()
        assert "already in use by another client" in result
        server_with_host.restart.assert_not_called()

    def test_an_unidentified_caller_can_still_restart(self, server_with_host):
        # In-process callers have no identity, so they are not measured against
        # the claim -- the same rule submit() uses.
        _server._claimed_by = "sess-A"
        try:
            assert "Kernel restarted" in _server.restart_kernel()
        finally:
            _server.clear_claim()

    def test_interrupt_refused_on_a_user_job(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(
                job_id="job-3",
                interrupted=False,
                status="running",
                refused="foreign_job",
            ),
        )
        result = _server.interrupt_kernel()
        # Must not read as "nothing was running" -- the agent would retry or move
        # on, when what it should do is wait for a person.
        assert "No running job" not in result
        assert "started by the user" in result
        assert "poll_job('job-3')" in result

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


class TestToolReturnShape:
    """What an *in-process* caller gets back from a tool call, per tool.

    A caller inside the session child (a built-in chat loop, say) reaches the
    tools below the layer that builds a ``CallToolResult``, and FastMCP hands it
    one of two shapes there: a bare ``list[ContentBlock]``, or a
    ``(blocks, structured)`` tuple. Both are declared —
    ``lowlevel/server.py`` names them ``UnstructuredContent`` and
    ``CombinationContent`` — and the low-level server collapses them on the way
    to the wire.

    Which shape a tool yields is decided by its **return annotation**: an
    annotation FastMCP can build an output schema from gets the tuple, and one
    it cannot gets the bare list. That makes the split easy to change by
    accident — retyping ``list_skills`` from ``list`` to ``list[dict]`` would
    silently move it, and reshape what an in-process caller receives without
    touching a line of that caller. It is also the wire contract: the same
    annotation decides whether an ``outputSchema`` is advertised to real MCP
    clients on ``tools/list``.

    So this pins both halves together. It is a change-detector on purpose: if it
    fails, the question to ask is whether the wire contract change was intended,
    not how to make the assertion pass.
    """

    #: (tool, minimal kwargs, declares an outputSchema / returns the tuple)
    SHAPES = [
        ("list_skills", {}, False),
        ("take_screenshot", {}, False),
        ("execute_code", {"python_code": "1"}, True),
        ("verify_workflow", {"cells": ["1"]}, True),
        ("poll_job", {"job_id": "job-1"}, True),
        ("inspect_object", {"object_path": "np"}, True),
        ("interrupt_kernel", {}, True),
        ("start_kernel", {}, True),
        ("restart_kernel", {}, True),
        ("server_status", {}, True),
    ]

    @pytest.fixture(autouse=True)
    def _no_kernel(self, monkeypatch):
        # Shape follows the annotation, not the runtime: with no host every tool
        # returns its not-ready answer, in the shape it always uses. Forced to
        # None so a stray MagicMock host from another test cannot reach a code
        # path that formats one.
        monkeypatch.setattr(_server, "_kernel_host", None)

    def test_every_tool_is_covered(self):
        """A new tool must land in the table above, with its shape chosen."""
        listed = {t.name for t in asyncio.run(_server.mcp.list_tools())}
        assert listed == {name for name, _, _ in self.SHAPES}

    @pytest.mark.parametrize("name,kwargs,structured", SHAPES)
    def test_output_schema_matches_the_table(self, name, kwargs, structured):
        tool = {t.name: t for t in asyncio.run(_server.mcp.list_tools())}[name]
        assert (tool.outputSchema is not None) is structured

    @pytest.mark.parametrize("name,kwargs,structured", SHAPES)
    def test_dispatch_shape_follows_the_schema(self, name, kwargs, structured):
        result = asyncio.run(
            _server.mcp._tool_manager.call_tool(name, kwargs, convert_result=True)
        )
        assert isinstance(result, tuple) is structured
        blocks = result[0] if isinstance(result, tuple) else result
        # Either way the content blocks are the first thing a caller wants, and
        # there is always at least one.
        assert len(blocks) >= 1


class TestClientIdentity:
    """What the kernel's one-agent claim is keyed on.

    Read through ``mcp.get_context()`` rather than a ``Context`` tool parameter,
    so the tools stay callable in-process; these fake the context the SDK would
    install around a real request.
    """

    @staticmethod
    def _ctx(headers, client_name="claude-code"):
        info = type("Info", (), {"name": client_name})()
        params = type("Params", (), {"clientInfo": info})()
        session = type("Session", (), {"client_params": params})()
        request = type("Request", (), {"headers": headers})()
        rc = type("RC", (), {"request": request, "session": session})()
        return type("Ctx", (), {"request_context": rc})()

    def test_reads_the_transport_session_id(self, monkeypatch):
        # Two clients on one session child differ here and nowhere else: the
        # tool surface is stateless, so this header is the only thing that
        # tells them apart.
        monkeypatch.setattr(
            _server.mcp, "get_context", lambda: self._ctx({"mcp-session-id": "abc123"})
        )
        assert _server._client_identity() == ("abc123", "claude-code")

    def test_falls_back_to_the_connection_when_the_header_is_absent(self, monkeypatch):
        # A client that negotiated no transport session still gets one identity
        # per connection, which is all the claim needs.
        monkeypatch.setattr(_server.mcp, "get_context", lambda: self._ctx({}))
        ident, label = _server._client_identity()
        assert ident.startswith("conn-") and label == "claude-code"

    def test_no_request_yields_no_identity(self):
        # An in-process call (these tests; a chat loop later) has no client, and
        # submit() reads that as "nothing to claim with" rather than refusing.
        assert _server._client_identity() == (None, "")


class TestPollJobRendersAVerification:
    """poll_job is where a long verification is collected, so it must render the
    ledger rather than the flattened output of the job that produced it."""

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch):
        monkeypatch.setattr(_server.time, "sleep", lambda *a, **k: None)

    def test_a_terminal_verification_polls_as_its_report(self, server_with_host):
        _install_replies(server_with_host, returns=_job_reply(**_verify_snapshot()))
        result = _server.poll_job("job-1")
        assert "job-1: ok" in result
        assert "Verified" in result and "Save workflow" in result

    def test_a_running_verification_still_shows_partial_output(self, server_with_host):
        snap = _verify_snapshot()
        snap["status"] = "running"
        snap["stdout"] = "one\n"
        _install_replies(server_with_host, returns=_job_reply(**snap))
        result = _server.poll_job("job-1")
        assert "Partial output" in result and "one" in result

    def test_an_ordinary_job_is_unaffected(self, server_with_host):
        _install_replies(
            server_with_host,
            returns=_job_reply(**_snapshot(status="ok", stdout="hi\n")),
        )
        assert "hi" in _server.poll_job("job-1")
