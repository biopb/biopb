"""The scratch kernel a verification runs in, and the slot it takes.

The session child half of ``verify_workflow``: spawning a second kernel,
running the cells there, collecting the record, and discarding the process.
The kernel half — how the cells themselves run — is ``test_mcp_jobs.py``.

No real kernel here. ``_scratch`` sends each cell with ``host.run_cell`` and
reads how it went from ``host.jobs``, so a host whose ``run_cell`` plays the
protocol into real records exercises the whole path. Real kernels are
``test_mcp_jobs.py`` (``TestJobConcurrency``).
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from biopb_mcp import _config
from biopb_mcp._tests.conftest import call_tool as _tool, rpc_reply
from biopb_mcp.mcp import _app, _scratch, _server, _writers
from biopb_mcp.mcp._job_log import JobLog


def _msg(msg_type, request, session=None, **content):
    return {
        "header": {"msg_type": msg_type},
        "parent_header": {"msg_id": request, "session": session},
        "content": content,
    }


def _scratch_host(job_status="ok", on_start=None, hold=None, interrupt_lands=True):
    """A stand-in scratch kernel host over real job records (``JobLog``): its
    ``run_cell`` plays the protocol a cell produces -- the echo, a line of
    output, an error for a failing cell, the idle.

    *job_status* is how every cell ends (``"ok"`` or ``"error"``). *hold* is an
    ``Event``: while it is unset a cell stays running, so a test can act on a
    verification that is genuinely in flight. *interrupt_lands* False models
    the case the escalation exists for -- cells wedged in a C call, where the
    interrupt is accepted and changes nothing.
    """
    host = MagicMock()
    host.start.side_effect = on_start or (lambda: None)
    host.is_alive.return_value = True
    log = JobLog(host_session="host")
    host.jobs = log
    stopped = threading.Event()

    def finish(request, status):
        log.on_iopub(_msg("stream", request, name="stdout", text="full output"))
        if status != "ok":
            ename = "KeyboardInterrupt" if status == "interrupted" else "ValueError"
            log.on_iopub(_msg("error", request, ename=ename, traceback=[ename]))
        log.on_iopub(_msg("status", request, execution_state="idle"))

    def run_cell(code, job_id, origin, intent="", bare=False):
        request = f"req-{job_id}"
        log.start_cell(job_id, request, code, origin, intent)
        log.on_iopub(_msg("execute_input", request, session="host", code=code))
        if hold is None:
            finish(request, job_status)
        else:

            def later():
                hold.wait()
                finish(request, "interrupted" if stopped.is_set() else job_status)

            threading.Thread(target=later, daemon=True).start()
        return request

    def interrupt_job(job_id, reason=None):
        log.note_cancel(job_id, reason)
        if interrupt_lands and hold is not None:
            stopped.set()
            hold.set()
        return {"interrupted": True, "job_id": job_id}

    host.run_cell.side_effect = run_cell
    host.interrupt_job.side_effect = interrupt_job
    return host


def _blocks(cells, prose="What this workflow does."):
    """A parsed document: one markdown block, then *cells* as code blocks.

    `start` takes the whole document now, not a list of cells: the prose is
    what the saved notebook is mostly made of, and it never goes to the kernel.
    """
    return [{"kind": "markdown", "text": prose}] + [
        {"kind": "code", "text": c} for c in cells
    ]


def _session_host(running=None):
    """The session host, whose records ``_scratch`` reads only for whether a
    job is running."""
    host = MagicMock()
    host.jobs.running.return_value = running
    host.jobs.foreign_digest.return_value = []
    host.execute.side_effect = lambda *a, **k: rpc_reply(None)
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
def spool(tmp_path, monkeypatch):
    """Redirect the workflow spool. Autouse: a verification that passes writes
    one, and a test suite must not write into the user's own state tree."""
    d = tmp_path / "workflows"
    d.mkdir()
    monkeypatch.setattr(_config, "get_workflow_dir", lambda: d)
    return d


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
        started = _scratch.start(_blocks(["a = 2"]), "wf", _session_host())
        snap = _settle(started["job_id"])

        assert snap["status"] == "ok"
        assert _scratch.verified()["title"] == "wf"
        summary = _scratch.verified_summary()
        assert summary["job_id"] == "verify-1" and summary["cells"] == 1
        assert summary["created"] == _scratch.verified()["created"]

    def test_the_kept_record_is_the_full_one_not_the_polled_ledger(self):
        # The poll carries a head; the document needs the output.
        host = _scratch_host()
        _scratch.set_host_factory(lambda: host)
        job_id = _scratch.start(_blocks(["a = 2"]), "wf", _session_host())["job_id"]
        _settle(job_id)
        assert _scratch.verified()["cells"][0]["stdout"] == "full output"

    def test_each_cell_is_its_own_request_sent_bare_as_its_writer(self):
        # Bare: a document that never builds its own `client` must fail here,
        # as it would for its reader. The origin is the asker's, which the
        # stop rules read.
        host = _scratch_host()
        _scratch.set_host_factory(lambda: host)
        _settle(
            _scratch.start(
                _blocks(["a = 2", "print(a)"]), "wf", _session_host(), origin="chat"
            )["job_id"]
        )
        calls = host.run_cell.call_args_list
        assert [c.args[0] for c in calls] == ["a = 2", "print(a)"]
        assert all(c.args[2] == "chat" and c.kwargs["bare"] for c in calls)

    def test_the_cells_after_a_failure_are_skipped_not_sent(self):
        host = _scratch_host(job_status="error")
        _scratch.set_host_factory(lambda: host)
        snap = _settle(
            _scratch.start(_blocks(["1/0", "a = 1", "b = 2"]), "bad", _session_host())[
                "job_id"
            ]
        )
        assert snap["status"] == "error"
        assert host.run_cell.call_count == 1
        cells = snap["verify"]["cells"]
        assert [c["status"] for c in cells] == ["error", "skipped", "skipped"]
        assert cells[0]["error_text"] == "ValueError"

    def test_a_kernel_death_mid_cell_is_the_verdict(self):
        # The OOM a verification exists to catch: the cell it died in failed,
        # and the rest never ran.
        hold = threading.Event()
        host = _scratch_host(hold=hold)
        host.is_alive.return_value = False
        _scratch.set_host_factory(lambda: host)
        try:
            snap = _settle(
                _scratch.start(_blocks(["a = 2", "b = 3"]), "wf", _session_host())[
                    "job_id"
                ]
            )
        finally:
            hold.set()
        assert snap["status"] == "error"
        assert "died" in snap["error_text"]
        cells = snap["verify"]["cells"]
        assert [c["status"] for c in cells] == ["error", "skipped"]
        assert "died" in cells[0]["error_text"]

    def test_the_scratch_kernel_is_discarded_either_way(self):
        for status in ("ok", "error"):
            _scratch.reset()
            host = _scratch_host(job_status=status)
            _scratch.set_host_factory(lambda h=host: h)
            _settle(_scratch.start(_blocks(["a = 2"]), "wf", _session_host())["job_id"])
            assert host.shutdown.called, status

    def test_a_failed_run_is_not_kept(self):
        _scratch.set_host_factory(lambda: _scratch_host(job_status="error"))
        snap = _settle(
            _scratch.start(_blocks(["1/0"]), "bad", _session_host())["job_id"]
        )
        assert snap["status"] == "error"
        assert _scratch.verified() is None
        assert _scratch.verified_summary() is None

    def test_a_kernel_that_never_starts_is_the_verdict_not_a_crash(self):
        # Its death IS the answer: an OOM means the workflow does not fit. The
        # watchdog is off for exactly this reason -- a respawn would re-run a
        # workflow that just killed a process.
        def boom():
            raise MemoryError("Cannot allocate memory")

        _scratch.set_host_factory(lambda: _scratch_host(on_start=boom))
        snap = _settle(
            _scratch.start(_blocks(["a = 2"]), "wf", _session_host())["job_id"]
        )
        assert snap["status"] == "error"
        assert "Cannot allocate memory" in snap["error_text"]
        assert snap["verify"] is None
        assert _scratch.verified() is None

    def test_without_a_factory_it_says_so_rather_than_failing_obscurely(self):
        assert (
            "unavailable"
            in _scratch.start(_blocks(["1"]), "", _session_host())["error"]
        )


class TestTheRunList:
    """What the observe page's verification pane reads.

    One run: the last, or the one in flight. Older runs are not kept -- their
    kernels are gone and the page offers no way to reach them -- so the download
    follows what the page is showing.
    """

    def test_the_pane_shows_the_last_run(self):
        _scratch.set_host_factory(lambda: _scratch_host())
        first = _scratch.start(_blocks(["a = 2"]), "first", _session_host())["job_id"]
        _settle(first)
        _scratch.set_host_factory(lambda: _scratch_host())
        second = _scratch.start(_blocks(["b = 3"]), "second", _session_host())["job_id"]
        _settle(second)

        rows = _scratch.runs_view()
        assert [r["job_id"] for r in rows] == [second]
        assert rows[0]["verify"] and rows[0]["origin"] == "mcp"
        # The title is the row's "why"; a verification has no other one.
        assert rows[0]["intent_preview"] == "second"
        assert rows[0]["code_preview"] == "1 cell"

    def test_the_row_names_the_client_that_asked_for_the_run(self):
        # The pane said "mcp" whoever asked, so a verification the chat loop
        # started was listed as a remote client's (biopb/biopb#880).
        _scratch.set_host_factory(lambda: _scratch_host())
        _settle(
            _scratch.start(_blocks(["a = 2"]), "wf", _session_host(), origin="chat")[
                "job_id"
            ]
        )
        assert _scratch.runs_view()[0]["origin"] == "chat"

    def test_there_is_nothing_to_save_until_a_run_passes(self):
        assert _scratch.runs_view() == []
        assert _scratch.verified_summary() is None

        _scratch.set_host_factory(lambda: _scratch_host(job_status="error"))
        _settle(_scratch.start(_blocks(["boom"]), "bad", _session_host())["job_id"])
        # A failed run is listed -- that is the report -- but there is no
        # document behind it, so the page's download must stay closed.
        assert len(_scratch.runs_view()) == 1
        assert _scratch.verified() is None
        assert _scratch.verified_summary() is None

    def test_a_later_failure_replaces_an_earlier_pass(self):
        _scratch.set_host_factory(lambda: _scratch_host())
        good = _scratch.start(_blocks(["a = 2"]), "good", _session_host())["job_id"]
        _settle(good)
        assert _scratch.verified_summary()["job_id"] == good

        _scratch.set_host_factory(lambda: _scratch_host(job_status="error"))
        _settle(_scratch.start(_blocks(["boom"]), "bad", _session_host())["job_id"])
        # Deliberate: the page shows one run, so offering a download of a
        # document it is not showing is the confusing half. Re-run to get it
        # back.
        assert _scratch.verified() is None

    def test_a_run_that_passes_is_written_to_the_spool(self, spool):
        _scratch.set_host_factory(lambda: _scratch_host())
        _settle(
            _scratch.start(_blocks(["a = 2"]), "segment nuclei", _session_host())[
                "job_id"
            ]
        )

        (saved,) = list(spool.glob("*.ipynb"))
        assert saved.name.startswith("biopb-segment-nuclei-")
        # A real notebook, not a fragment: this is the file someone opens.
        nb = json.loads(saved.read_text())
        assert nb["nbformat"] == 4 and nb["cells"]
        assert _scratch.verified_summary()["saved_path"] == str(saved)
        # A pass *promotes*: the draft it was written from is gone, so a draft
        # on disk means "this one has not passed yet".
        assert list((spool / "drafts").glob("*.md")) == []

    def test_the_document_exists_before_the_run_says_it_passed(self, spool):
        # Everything waits on the status, so writing after it would hand a
        # caller a passed run whose file is not there yet.
        _scratch.set_host_factory(lambda: _scratch_host())
        job = _scratch.start(_blocks(["a = 2"]), "wf", _session_host())["job_id"]
        _settle(job)
        assert list(spool.glob("*.ipynb"))

    def test_a_run_that_fails_writes_no_record_but_keeps_the_draft(self, spool):
        # The failed attempt is the one a person most wants to open and fix, so
        # it is on disk in the format the tool takes -- but it is not a record:
        # nothing here ran to the end.
        _scratch.set_host_factory(lambda: _scratch_host(job_status="error"))
        snap = _settle(
            _scratch.start(_blocks(["boom"]), "bad", _session_host())["job_id"]
        )
        assert list(spool.glob("*.ipynb")) == []
        (draft,) = list((spool / "drafts").glob("*.md"))
        assert draft.name == "bad.md"
        # In the document's own spelling, so what is read back can be sent back.
        assert "```python\nboom\n```" in draft.read_text()
        assert snap["draft_path"] == str(draft)

    def test_the_draft_is_one_file_per_workflow_not_a_pile_of_attempts(self, spool):
        # Attempts at one workflow are drafts of one document.
        for cell in ("boom", "boom  # try again"):
            _scratch.reset()
            _scratch.set_host_factory(lambda: _scratch_host(job_status="error"))
            _settle(_scratch.start(_blocks([cell]), "bad", _session_host())["job_id"])
        (draft,) = list((spool / "drafts").glob("*.md"))
        assert "try again" in draft.read_text()

    def test_a_spool_that_cannot_be_written_is_not_a_failed_verification(
        self, monkeypatch
    ):
        # A disk error is not a fact about the user's code. Reporting it as a
        # failed workflow would be a lie about what their cells did.
        def boom():
            raise OSError("read-only file system")

        monkeypatch.setattr(_config, "get_workflow_dir", boom)
        _scratch.set_host_factory(lambda: _scratch_host())
        snap = _settle(
            _scratch.start(_blocks(["a = 2"]), "wf", _session_host())["job_id"]
        )
        assert snap["status"] == "ok"
        assert _scratch.verified()["saved_path"] is None

    def test_the_spool_keeps_only_the_newest(self, spool, monkeypatch):
        monkeypatch.setattr(_scratch, "_SPOOL_KEEP", 3)
        for i in range(6):
            # Distinct names: the stamp has one-second resolution, so a loop
            # this fast would otherwise overwrite one file six times.
            _scratch.set_host_factory(lambda: _scratch_host())
            _settle(
                _scratch.start(_blocks(["a = 2"]), f"wf{i}", _session_host())["job_id"]
            )
        assert len(list(spool.glob("*.ipynb"))) == 3

    def test_reset_forgets_the_run(self):
        _scratch.set_host_factory(lambda: _scratch_host())
        _settle(_scratch.start(_blocks(["a = 2"]), "wf", _session_host())["job_id"])
        _scratch.reset()
        assert _scratch.runs_view() == []


class TestTheSlot:
    """One job at a time is a rule about the session, not about one kernel.

    The dask cluster is shared and finite and the agent's whole model is one
    cell at a time, so a verification takes the slot ordinary work does.
    """

    def test_a_verification_is_refused_while_a_session_job_runs(self):
        _scratch.set_host_factory(lambda: _scratch_host())
        session = _session_host(running={"job_id": "job-7", "origin": "user"})
        started = _scratch.start(_blocks(["a = 2"]), "wf", session)
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
        first = _scratch.start(_blocks(["a = 2"]), "one", _session_host())
        try:
            deadline = time.monotonic() + 5.0
            while _scratch.running() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            second = _scratch.start(_blocks(["a = 2"]), "two", _session_host())
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
        session_host.run_cell.assert_not_called()


class TestInterrupting:
    """Stopping a verification, and who is allowed to: the rule the session
    kernel's stop follows, checked here (``_scratch.interrupt``)."""

    def _running(self, writer="agent-A", on_start=None, hold=None, lands=True):
        """Start a verification, optionally held mid-flight by *hold*."""
        host = _scratch_host(on_start=on_start, hold=hold, interrupt_lands=lands)
        _scratch.set_host_factory(lambda: host)
        started = _scratch.start(
            _blocks(["a = 2"]), "wf", _session_host(), writer=writer
        )
        return started["job_id"], host

    def _in_flight(self, job_id):
        deadline = time.monotonic() + 5.0
        while not _scratch.poll(job_id)["stdout"].startswith("Running cell"):
            assert time.monotonic() < deadline
            time.sleep(0.01)

    def test_the_owning_agent_stops_its_own_verification(self):
        hold = threading.Event()
        job_id, host = self._running(hold=hold)
        try:
            self._in_flight(job_id)
            data = _scratch.interrupt(None, "mcp", "agent-A")
        finally:
            hold.set()
        assert data == {"interrupted": True, "job_id": job_id}
        # Naming the running cell, which its kernel checks is still running.
        (call,) = host.interrupt_job.call_args_list
        assert call.args == ("job-1",)
        snap = _scratch.poll(job_id)
        assert snap["status"] == "interrupted"
        assert snap["verify"]["cells"][0]["status"] == "error"

    def test_a_stranger_cannot_stop_it(self):
        hold = threading.Event()
        job_id, host = self._running(hold=hold)
        try:
            self._in_flight(job_id)
            assert _scratch.interrupt(None, "mcp", "agent-B") == {
                "refused": "not_owner",
                "job_id": job_id,
            }
            # Nor can a writer of another origin: the chat loop did not ask
            # for this one.
            assert _scratch.interrupt(None, "chat", None)["refused"] == "foreign_job"
            host.interrupt_job.assert_not_called()
        finally:
            hold.set()
        assert _settle(job_id)["status"] == "ok"

    def test_a_stop_between_cells_sends_no_further_cell(self):
        # Nothing running to interrupt, but the run still stops: the next
        # cell is checked for the stop under the lock it is sent under.
        started = threading.Event()

        def on_start():
            started.set()
            time.sleep(0.2)

        host = _scratch_host(on_start=on_start)
        _scratch.set_host_factory(lambda: host)
        job_id = _scratch.start(
            _blocks(["a = 2", "b = 3"]), "wf", _session_host(), writer="agent-A"
        )["job_id"]
        started.wait(5.0)
        with _scratch._lock:
            _scratch._run["stop"] = "stopped between cells"
        snap = _settle(job_id)
        assert snap["status"] == "interrupted"
        assert "between cells" in snap["error_text"]
        host.run_cell.assert_not_called()
        assert snap["verify"] is None

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
            self._in_flight(job_id)
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
            self._in_flight(job_id)
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
        session_host.jobs.running.return_value = {"job_id": "job-5", "origin": "mcp"}
        session_host.interrupt_job.return_value = {
            "job_id": "job-5",
            "interrupted": True,
        }
        _tool(_server.interrupt_kernel)
        assert session_host.interrupt_job.call_args.args == ("job-5",)

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
        session_host.interrupt_job.assert_not_called()


class TestDiscarding:
    def test_discard_stops_the_run_and_takes_the_kernel_with_it(self):
        release = {"go": False}
        host = _scratch_host(
            on_start=lambda: (
                [time.sleep(0.01) for _ in iter(lambda: release["go"], True)] and None
            )
        )
        _scratch.set_host_factory(lambda: host)
        started = _scratch.start(_blocks(["a = 2"]), "wf", _session_host())
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
        started = _scratch.start(
            _blocks(["a = 2", "print(a * 3)"]), "wf", _session_host()
        )
        _settle(started["job_id"])
        detail = _scratch.detail(started["job_id"])
        # The program the run was given...
        assert "a = 2" in detail["code"] and "print(a * 3)" in detail["code"]
        # ...and a line per cell, not the per-cell output (that is the
        # notebook's).
        assert "1. ok · " in detail["stdout"] and " · full" in detail["stdout"]
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
        # Answered from this process; the session's records were never asked.
        host.jobs.poll.assert_not_called()

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
    old_host = _app._kernel_host
    _app.set_kernel_host(host)
    try:
        yield (
            TestClient(
                _observe._build_standalone_app(), base_url="http://127.0.0.1:8766"
            ),
            host,
        )
    finally:
        _app._kernel_host = old_host
        _http._mw = None
        _writers.clear_claim()
