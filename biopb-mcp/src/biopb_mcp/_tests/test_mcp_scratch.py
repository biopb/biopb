"""The scratch kernel a verification runs in, and the slot it takes.

The session child half of ``verify_workflow``: spawning a second kernel,
running the cells there, collecting the record, and discarding the process.
The kernel half — how the cells themselves run — is ``test_mcp_jobs.py``.

No real kernel here. ``_scratch`` reaches its kernel only through
``_kernel_rpc``, which is ``host.execute(snippet)`` and a JSON envelope back, so
a host that answers snippets by content exercises the whole path.
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from biopb_mcp._tests.conftest import call_tool as _tool
from biopb_mcp.mcp import _app, _kernel_rpc, _scratch, _server, _writers


def _envelope(result):
    return {
        "stdout": _kernel_rpc._JOB_DELIM + json.dumps({"r": result, "w": True}) + "\n",
        "result_text": "",
        "error_text": "",
        "status": "ok",
    }


def _cells_record(status="ok", title="wf"):
    return {
        "title": title,
        "created": 1_700_000_000.0,
        "status": status,
        "cells": [
            {
                "code": "a = 2",
                "status": status,
                "stdout": "full output",
                "stdout_head": "full",
                "stdout_len": 11,
                "error_text": "",
                "elapsed": 0.1,
            }
        ],
    }


def _scratch_host(
    job_status="ok",
    record=None,
    on_start=None,
    title="wf",
    hold=None,
    interrupt_lands=True,
):
    """A stand-in scratch kernel that answers the four snippets ``_scratch`` sends.

    *hold* is an ``Event``: while it is unset the kernel's job polls as still
    running, so a test can act on a verification that is genuinely in flight.
    *interrupt_lands* False models the case the escalation exists for -- cells
    wedged in a C call, where the KeyboardInterrupt is accepted and changes
    nothing.
    """
    host = MagicMock()
    host.start.side_effect = on_start or (lambda: None)
    record = _cells_record(job_status, title) if record is None else record

    def execute(code, *_a, **_k):
        if "_jobs.submit(" in code:
            return _envelope({"job_id": "job-1"})
        if "_jobs.poll(" in code:
            snap = {
                "job_id": "job-1",
                "status": "running"
                if (hold is not None and not hold.is_set())
                else job_status,
                "stdout": "",
                "error_text": "",
                "verify": {**record, "cells": [{**record["cells"][0]}]},
            }
            # The polled ledger carries a head, never the full output.
            snap["verify"]["cells"][0].pop("stdout", None)
            return _envelope(snap)
        if "_jobs.verify_record(" in code:
            return _envelope(record)
        if "_jobs.interrupt_current(" in code:
            if interrupt_lands:
                hold.set() if hold is not None else None
            return _envelope({"interrupted": True, "job_id": "job-1"})
        return _envelope(None)

    host.execute.side_effect = execute
    return host


def _session_host(running=None):
    """The session kernel, which ``_scratch`` asks only whether it is busy."""
    host = MagicMock()
    host.execute.side_effect = lambda code, *a, **k: _envelope(
        running if "_jobs.running_job(" in code else None
    )
    return host


def _settle(job_id, timeout=5.0):
    """Wait for a run to leave ``running`` and return its final snapshot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = _scratch.poll(job_id)
        if snap is None or snap["status"] != "running":
            return snap
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never finished")


@pytest.fixture(autouse=True)
def clean_scratch():
    _scratch.reset()
    _scratch.set_host_factory(None)
    yield
    _scratch.reset()
    _scratch.set_host_factory(None)


class TestRunningAVerification:
    def test_a_clean_run_is_kept_as_the_verified_workflow(self):
        host = _scratch_host()
        _scratch.set_host_factory(lambda: host)
        started = _scratch.start(["a = 2"], "wf", _session_host())
        snap = _settle(started["job_id"])

        assert snap["status"] == "ok"
        assert _scratch.verified()["title"] == "wf"
        assert _scratch.verified_summary() == {
            "title": "wf",
            "cells": 1,
            "created": 1_700_000_000.0,
        }

    def test_the_kept_record_is_the_full_one_not_the_polled_ledger(self):
        # The poll ships a head every 0.4s; the document needs the output. It is
        # read once, before the kernel holding it is discarded -- after which
        # there is nobody left to ask.
        host = _scratch_host()
        _scratch.set_host_factory(lambda: host)
        _settle(_scratch.start(["a = 2"], "wf", _session_host())["job_id"])
        assert _scratch.verified()["cells"][0]["stdout"] == "full output"

    def test_the_scratch_kernel_is_discarded_either_way(self):
        for status in ("ok", "error"):
            _scratch.reset()
            host = _scratch_host(job_status=status)
            _scratch.set_host_factory(lambda h=host: h)
            _settle(_scratch.start(["a = 2"], "wf", _session_host())["job_id"])
            assert host.shutdown.called, status

    def test_a_failed_run_is_not_kept(self):
        _scratch.set_host_factory(lambda: _scratch_host(job_status="error"))
        snap = _settle(_scratch.start(["1/0"], "bad", _session_host())["job_id"])
        assert snap["status"] == "error"
        assert _scratch.verified() is None
        assert _scratch.verified_summary() is None

    def test_a_later_failure_does_not_unverify_what_passed(self):
        _scratch.set_host_factory(lambda: _scratch_host(title="good"))
        _settle(_scratch.start(["1"], "good", _session_host())["job_id"])
        _scratch.set_host_factory(
            lambda: _scratch_host(job_status="error", title="bad")
        )
        _settle(_scratch.start(["1/0"], "bad", _session_host())["job_id"])
        assert _scratch.verified()["title"] == "good"

    def test_a_kernel_that_never_starts_is_the_verdict_not_a_crash(self):
        # Its death IS the answer: an OOM means the workflow does not fit. The
        # watchdog is off for exactly this reason -- a respawn would re-run a
        # workflow that just killed a process.
        def boom():
            raise MemoryError("Cannot allocate memory")

        _scratch.set_host_factory(lambda: _scratch_host(on_start=boom))
        snap = _settle(_scratch.start(["a = 2"], "wf", _session_host())["job_id"])
        assert snap["status"] == "error"
        assert "Cannot allocate memory" in snap["error_text"]
        assert snap["verify"] is None
        assert _scratch.verified() is None

    def test_without_a_factory_it_says_so_rather_than_failing_obscurely(self):
        assert "unavailable" in _scratch.start(["1"], "", _session_host())["error"]


class TestTheSlot:
    """One job at a time is a rule about the session, not about one kernel.

    The dask cluster is shared and finite and the agent's whole model is one
    cell at a time, so a verification takes the slot ordinary work does.
    """

    def test_a_verification_is_refused_while_a_session_job_runs(self):
        _scratch.set_host_factory(lambda: _scratch_host())
        session = _session_host(running={"job_id": "job-7", "origin": "user"})
        started = _scratch.start(["a = 2"], "wf", session)
        assert started == {
            "error": "busy",
            "running_job_id": "job-7",
            "running_job_origin": "user",
        }
        assert _scratch.running() is None

    def test_a_second_verification_is_refused_while_one_runs(self):
        release = {"go": False}

        def slow_start():
            while not release["go"]:
                time.sleep(0.01)

        _scratch.set_host_factory(lambda: _scratch_host(on_start=slow_start))
        first = _scratch.start(["a = 2"], "one", _session_host())
        try:
            deadline = time.monotonic() + 5.0
            while _scratch.running() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            second = _scratch.start(["a = 2"], "two", _session_host())
            assert second["error"] == "busy"
            assert second["running_job_id"] == first["job_id"]
        finally:
            release["go"] = True
        _settle(first["job_id"])

    def test_execute_code_is_refused_while_a_verification_runs(
        self, monkeypatch, session_host
    ):
        monkeypatch.setattr(
            _scratch, "running", lambda: {"job_id": "verify-1", "elapsed": 1.0}
        )
        result = _tool(_server.execute_code, "x = 1")
        assert "verify-1" in result and "already running" in result
        assert not any(
            "_jobs.submit(" in c[0][0] for c in session_host.execute.call_args_list
        )


class TestInterrupting:
    """Stopping a verification, and who is allowed to.

    The claim is the scratch kernel's own: ``start`` submits with the verifying
    client's writer, so that kernel's ``_jobs.submit`` claims it and its
    ``interrupt_current`` refuses everyone else -- the same rule as any other
    job, enforced by the same code.
    """

    def _running(self, writer="agent-A", on_start=None, hold=None, lands=True):
        """Start a verification, optionally held mid-flight by *hold*."""
        host = _scratch_host(on_start=on_start, hold=hold, interrupt_lands=lands)
        _scratch.set_host_factory(lambda: host)
        started = _scratch.start(
            ["a = 2"], "wf", _session_host(), writer=writer, writer_label="A"
        )
        return started["job_id"], host

    def test_the_run_claims_its_kernel_for_the_client_that_asked(self):
        job_id, host = self._running()
        _settle(job_id)
        (submit,) = [
            c[0][0] for c in host.execute.call_args_list if "_jobs.submit(" in c[0][0]
        ]
        assert "writer='agent-A'" in submit and "writer_label='A'" in submit

    def test_the_owning_agent_stops_its_own_verification(self):
        hold = threading.Event()
        job_id, host = self._running(hold=hold)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _scratch.poll(job_id)["stdout"].startswith("Starting"):
                    break
                time.sleep(0.01)
            data = _scratch.interrupt(None, "mcp", "agent-A")
        finally:
            hold.set()
        _settle(job_id)
        # Handed to the kernel, which owns the decision; it answered yes.
        assert data["interrupted"] is True
        assert data["job_id"] == job_id
        (call,) = [
            c[0][0]
            for c in host.execute.call_args_list
            if "_jobs.interrupt_current(" in c[0][0]
        ]
        assert "requester='mcp'" in call and "writer='agent-A'" in call

    def test_a_stranger_cannot_stop_it_during_the_bring_up(self):
        # The one window the kernel cannot answer for itself: it does not exist
        # yet. Same rule, applied here, so a second client cannot discard an
        # attempt in its first few seconds either.
        release = {"go": False}

        def slow_start():
            while not release["go"]:
                time.sleep(0.01)

        job_id, _host = self._running(on_start=slow_start)
        try:
            deadline = time.monotonic() + 5.0
            while _scratch.running() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert _scratch.interrupt(None, "mcp", "agent-B") == {
                "refused": "not_owner",
                "job_id": job_id,
            }
            assert _scratch.running() is not None, "the run was discarded anyway"
            # The person at the machine is exempt, as everywhere else.
            assert _scratch.interrupt("stop", "user")["interrupted"] is True
        finally:
            release["go"] = True

    def test_cells_that_will_not_stop_are_killed_with_their_kernel(self, monkeypatch):
        """The escalation, and the reason restart_kernel stays out of this.

        A verification wedged in a blocking C call does not notice a
        KeyboardInterrupt. On the session kernel that is where best-effort
        stops and the user decides, because the guaranteed stop costs them
        their session. Here it costs nothing -- so the process goes, and the
        agent never needs the tool that would take the user's work with it.
        """
        monkeypatch.setattr(_scratch, "_INTERRUPT_GRACE", 0.3)
        hold = threading.Event()
        job_id, host = self._running(hold=hold, lands=False)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _scratch.poll(job_id)["stdout"].startswith("Starting"):
                    break
                time.sleep(0.01)
            data = _scratch.interrupt(None, "mcp", "agent-A")
        finally:
            hold.set()

        assert data == {"interrupted": True, "job_id": job_id, "killed": True}
        assert _scratch.poll(job_id)["status"] == "interrupted"
        assert host.shutdown.called
        # The slot is free without anyone having touched the session.
        assert _scratch.running() is None

    def test_a_clean_stop_is_not_reported_as_a_kill(self):
        hold = threading.Event()
        job_id, _host = self._running(hold=hold, lands=True)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _scratch.poll(job_id)["stdout"].startswith("Starting"):
                    break
                time.sleep(0.01)
            data = _scratch.interrupt(None, "mcp", "agent-A")
        finally:
            hold.set()
        assert data.get("killed") is None
        assert data["interrupted"] is True

    def test_the_tool_says_a_stuck_verification_needs_no_restart(
        self, monkeypatch, session_host
    ):
        # The advice matters more than the wording: an agent told to use
        # restart_kernel here would destroy the user's session to end a process
        # built to be thrown away.
        monkeypatch.setattr(
            _scratch,
            "interrupt",
            lambda *a, **k: {"interrupted": True, "job_id": "verify-1", "killed": True},
        )
        result = _tool(_server.interrupt_kernel)
        assert "kernel was killed" in result
        assert "session is untouched" in result
        assert "nothing here needs restart_kernel" in result

    def test_interrupt_with_nothing_verifying_says_so_by_returning_none(self):
        # Not "nothing is running": the caller has to fall through and ask the
        # session kernel, which may well have started something.
        assert _scratch.interrupt(None, "mcp", "agent-A") is None

    def test_the_tool_falls_through_to_the_session_kernel(
        self, monkeypatch, session_host
    ):
        monkeypatch.setattr(_scratch, "interrupt", lambda *a, **k: None)
        _tool(_server.interrupt_kernel)
        assert any(
            "_jobs.interrupt_current(" in c[0][0]
            for c in session_host.execute.call_args_list
        )

    def test_the_tool_reports_a_refusal_as_a_refusal(self, monkeypatch, session_host):
        # Not as "no running job to interrupt" -- an agent told that would reach
        # for restart_kernel, which is the one thing it must not do here.
        monkeypatch.setattr(
            _scratch,
            "interrupt",
            lambda *a, **k: {"refused": "not_owner", "job_id": "verify-1"},
        )
        result = _tool(_server.interrupt_kernel)
        assert "already in use" in result
        assert not any(
            "_jobs.interrupt_current(" in c[0][0]
            for c in session_host.execute.call_args_list
        )


class TestDiscarding:
    def test_discard_stops_the_run_and_takes_the_kernel_with_it(self):
        release = {"go": False}
        host = _scratch_host(
            on_start=lambda: (
                [time.sleep(0.01) for _ in iter(lambda: release["go"], True)] and None
            )
        )
        _scratch.set_host_factory(lambda: host)
        started = _scratch.start(["a = 2"], "wf", _session_host())
        deadline = time.monotonic() + 5.0
        while _scratch.running() is None and time.monotonic() < deadline:
            time.sleep(0.01)

        assert _scratch.discard("restart") == started["job_id"]
        release["go"] = True
        snap = _scratch.poll(started["job_id"])
        assert snap["status"] == "interrupted"
        assert "restart" in snap["error_text"]
        # The slot is free again the moment the status changes, not when the
        # process finishes dying.
        assert _scratch.running() is None

    def test_discarding_nothing_is_not_an_error(self):
        assert _scratch.discard() is None

    def test_restart_kernel_discards_an_in_flight_verification(
        self, monkeypatch, session_host
    ):
        discarded = []
        monkeypatch.setattr(
            _scratch,
            "discard",
            lambda reason=None: discarded.append(reason) or "verify-1",
        )
        result = _tool(_server.restart_kernel)
        # Not because the user asked to kill it: it holds the slot, so leaving
        # it would hand back a fresh kernel that can accept nothing.
        assert discarded == ["restart_kernel"]
        assert "verify-1 was discarded" in result
        assert session_host.restart.called

    def test_a_refused_restart_does_not_discard_the_holders_verification(
        self, monkeypatch, session_host
    ):
        """The gate has to come first, or it is not a gate.

        A client that does not hold the kernel is refused a restart -- but if
        the discard ran before that decision, the refusal arrives *after* the
        stranger's call has already destroyed the holder's verification. That
        hands a client which cannot take the session the power to wreck it,
        which is the one thing the gate exists to prevent.
        """
        discarded = []
        monkeypatch.setattr(
            _scratch,
            "discard",
            lambda reason=None: discarded.append(reason) or "verify-1",
        )
        _writers._note_claim("agent-A")
        _writers._local_identity.set(("agent-B", "B"))
        try:
            result = _tool(_server.restart_kernel)
        finally:
            _writers._local_identity.set(None)

        assert "already in use" in result
        assert discarded == [], "a refused restart destroyed the verification"
        assert not session_host.restart.called

    def test_the_user_restart_discards_before_it_replaces_the_kernel(
        self, monkeypatch, observe_client
    ):
        # Ungated -- the person at the machine can always replace the kernel --
        # so there is no gate to order against, but the discard still has to
        # come first: until the slot is released the new kernel accepts nothing.
        order = []
        monkeypatch.setattr(
            _scratch, "discard", lambda reason=None: order.append("discard")
        )
        client, host = observe_client
        host.restart.side_effect = lambda: order.append("restart")
        assert client.post("/api/kernel/restart").status_code == 200
        assert order == ["discard", "restart"]


class TestTheScratchKernelIsMarkedForTheBootstrap:
    """The env var is spelled in two modules and imported in neither.

    ``_kernel`` belongs to the session child and ``_bootstrap`` runs inside the
    kernel, so the literal is written twice and kept in sync by a comment -- the
    same split ``ENV_WINDOW_CLOSE_FD`` uses. A drift here is silent and
    expensive: the scratch kernel would build a *visible* viewer on the user's
    display, mid-session.
    """

    def test_the_launcher_and_the_kernel_agree_on_the_name(self, monkeypatch):
        from biopb_mcp.mcp import _bootstrap, _kernel

        monkeypatch.delenv(_kernel.ENV_SCRATCH, raising=False)
        assert not _bootstrap.is_scratch_kernel()
        monkeypatch.setenv(_kernel.ENV_SCRATCH, "1")
        assert _bootstrap.is_scratch_kernel()

    def test_a_watchdog_interval_of_zero_starts_no_watchdog(self):
        # How the scratch host turns respawns off: its death is the verdict, and
        # a respawn would re-run a workflow that just killed a process.
        from biopb_mcp.mcp._kernel import KernelHost

        host = KernelHost(watchdog_interval=0)
        host._start_watchdog()
        assert host._watchdog_thread is None
        assert host.health()["watchdog_running"] is False


class TestRouting:
    def test_the_id_says_which_kernel_owns_the_job(self):
        assert _scratch.owns("verify-1")
        assert not _scratch.owns("job-1")
        assert not _scratch.owns(None)

    def test_polling_an_unknown_verification_returns_nothing(self):
        assert _scratch.poll("verify-99") is None
        assert _scratch.detail("verify-99") is None

    def test_the_detail_view_shows_the_program_and_the_ledger(self):
        _scratch.set_host_factory(lambda: _scratch_host())
        started = _scratch.start(["a = 2", "print(a * 3)"], "wf", _session_host())
        _settle(started["job_id"])
        detail = _scratch.detail(started["job_id"])
        # The program the run was given...
        assert "a = 2" in detail["code"] and "print(a * 3)" in detail["code"]
        # ...and a line per cell, not the per-cell output (that is the
        # notebook's).
        assert "1. ok · 0.1s · full" in detail["stdout"]
        # The scratch kernel's hidden viewer is not the session's window.
        assert detail["window_alive"] is None

    def test_the_observe_detail_route_answers_a_verification_id(
        self, monkeypatch, observe_client
    ):
        """A row the page lists must be a row the page can open.

        The session kernel has never heard of a verify-N id, so the detail
        route asking it would 404 the very row _api_jobs added -- no progress,
        no output, for the one job that takes the longest.
        """
        client, host = observe_client
        monkeypatch.setattr(
            _scratch,
            "detail",
            lambda job_id: {
                "job_id": job_id,
                "status": "running",
                "elapsed": 3.2,
                "stdout": "Scratch kernel ready; running the workflow…\n",
                "code": "a = 2",
                "intent": "verify workflow: wf",
                "error_text": "",
                "result_text": "",
                "window_alive": None,
            },
        )
        r = client.get("/api/jobs/verify-1")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == "verify-1" and body["status"] == "running"
        assert "running the workflow" in body["stdout"]
        assert body["code"] == "a = 2"
        assert body["truncated"] is False
        # Answered from this process; the session kernel was never asked.
        assert not any("_jobs.poll(" in c[0][0] for c in host.execute.call_args_list)

    def test_an_unknown_verification_id_still_404s(self, observe_client):
        client, _host = observe_client
        assert client.get("/api/jobs/verify-99").status_code == 404


@pytest.fixture
def session_host():
    host = _session_host()
    _app.set_kernel_host(host)
    old = _app._kernel_host
    yield host
    _app._kernel_host = old
    _writers.clear_claim()


@pytest.fixture
def observe_client():
    """The observe API over a mock kernel host."""
    from starlette.testclient import TestClient

    from biopb_mcp.mcp import _http, _observe

    host = _session_host()
    old_host, old_console = _app._kernel_host, _observe._console_enabled
    _app.set_kernel_host(host)
    _observe.configure(console_enabled=True)
    try:
        yield (
            TestClient(
                _observe._build_standalone_app(), base_url="http://127.0.0.1:8766"
            ),
            host,
        )
    finally:
        _app._kernel_host = old_host
        _observe._console_enabled = old_console
        _http._mw = None
        _writers.clear_claim()
