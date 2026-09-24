"""The host's job records (``_job_log``), fed synthetic iopub messages.

The kernel side of the announcements is ``test_mcp_jobs``; a real kernel
publishing them is ``test_mcp_kernel`` (``TestHostRecords``).
"""

import pytest

from biopb_mcp._tests.conftest import iopub_event
from biopb_mcp.mcp import _job_log
from biopb_mcp.mcp._job_log import JobLog


def _event(**content):
    return iopub_event(content)


def _msg(msg_type, request, **content):
    return {
        "header": {"msg_type": msg_type},
        "parent_header": {"msg_id": request},
        "content": content,
    }


def _start(log, job_id="job-1", request="req-1", origin="mcp", code="x = 1", **kw):
    log.on_iopub(
        _event(
            event="start",
            job_id=job_id,
            request=request,
            origin=origin,
            code=code,
            **kw,
        )
    )


def _end(log, job_id="job-1", status="ok", **kw):
    log.on_iopub(_event(event="end", job_id=job_id, status=status, **kw))


def _print(log, text, request="req-1", name="stdout"):
    log.on_iopub(_msg("stream", request, name=name, text=text))


class TestRecord:
    def test_output_under_the_request_is_the_jobs(self):
        log = JobLog()
        _start(log)
        _print(log, "hello\n")
        _print(log, "not ours\n", request="req-other")
        assert log.poll("job-1")["stdout"] == "hello\n"
        assert log.poll("job-1")["status"] == "running"
        _end(log, result_text="3", elapsed=0.5)
        snap = log.poll("job-1")
        assert snap["status"] == "ok"
        assert snap["result_text"] == "3"
        assert snap["elapsed"] == 0.5
        assert snap["stdout"] == "hello\n"

    def test_both_streams_are_kept(self):
        log = JobLog()
        _start(log)
        _print(log, "out\n")
        _print(log, "err\n", name="stderr")
        assert log.poll("job-1")["stdout"] == "out\nerr\n"

    def test_output_after_the_end_is_not_the_jobs(self):
        log = JobLog()
        _start(log)
        _end(log)
        _print(log, "later\n")
        assert log.poll("job-1")["stdout"] == ""

    def test_a_task_keeps_its_own_error_text(self):
        # A task's failure is its announced end: the iopub error under the same
        # request is the cell's (TestCells).
        log = JobLog()
        _start(log)
        _end(log, status="error", error_text="Traceback ... ZeroDivisionError")
        assert log.poll("job-1")["error_text"] == "Traceback ... ZeroDivisionError"

    def test_unknown_job(self):
        assert JobLog().poll("job-9")["status"] == "unknown"

    def test_the_snapshot_keeps_the_kernel_shape(self):
        log = JobLog()
        _start(log, intent="why", created=123.0)
        assert set(log.poll("job-1")) == {
            "job_id",
            "code",
            "status",
            "stdout",
            "stdout_dropped",
            "stdout_total",
            "result_text",
            "error_text",
            "cancel_reason",
            "origin",
            "intent",
            "elapsed",
            "created",
        }
        assert log.poll("job-1")["created"] == 123.0


def _input(log, code, request, session="client"):
    log.on_iopub(
        {
            "header": {"msg_type": "execute_input"},
            "parent_header": {"msg_id": request, "session": session},
            "content": {"code": code},
        }
    )


def _idle(log, request):
    log.on_iopub(_msg("status", request, execution_state="idle"))


class TestCells:
    """A foreign client's cell, read from the protocol: its execute_input
    starts it, an error fails it, its request's idle ends it."""

    def _log(self):
        return JobLog(host_session="host")

    def _only(self, log):
        (rec,) = log.export()
        return rec

    def test_a_cell_runs_and_ends_on_its_idle(self):
        log = self._log()
        _input(log, "print('hi')", "c1")
        _print(log, "hi\n", request="c1")
        rec = self._only(log)
        assert rec["status"] == "running"
        assert rec["origin"] == "user" and rec["code"] == "print('hi')"
        _idle(log, "c1")
        rec = self._only(log)
        assert rec["status"] == "ok" and rec["stdout"] == "hi\n"

    def test_the_hosts_own_requests_are_not_cells(self):
        log = self._log()
        _input(log, "_viewer_window_alive()", "h1", session="host")
        _input(log, "   ", "c1")  # a client asking for its prompt number
        assert log.export() == []

    def test_an_error_fails_it_with_its_traceback(self):
        log = self._log()
        _input(log, "1/0", "c1")
        log.on_iopub(
            _msg("error", "c1", ename="ZeroDivisionError", traceback=["Zero", "at 1"])
        )
        _idle(log, "c1")
        rec = self._only(log)
        assert rec["status"] == "error" and rec["error_text"] == "Zero\nat 1"

    def test_a_stop_is_interrupted_and_says_who(self):
        log = self._log()
        _input(log, "loop()", "c1")
        (rec,) = log.export()
        log.note_cancel(rec["job_id"], "stopped by the user")
        log.on_iopub(_msg("error", "c1", ename="KeyboardInterrupt", traceback=["KI"]))
        _idle(log, "c1")
        rec = self._only(log)
        assert rec["status"] == "interrupted"
        assert rec["error_text"] == "stopped by the user\nKI"

    def test_a_cell_that_ran_ends_a_cell_whose_end_was_lost(self):
        # One cell at a time on the main thread.
        log = self._log()
        _input(log, "a = 1", "c1")
        _input(log, "b = 2", "c2")
        _idle(log, "c2")
        (first, second) = log.export()
        assert first["status"] == "error" and "not recorded" in first["error_text"]
        assert second["status"] == "ok"

    def test_a_host_cell_still_queued_is_not_ended_by_another(self):
        # Sent, but waiting in the kernel's queue behind a user's cell: it has
        # not begun, so that cell's end proves nothing about it.
        log = self._log()
        log.start_cell(log.new_id(), "h1", "x = 1", origin="mcp")
        _input(log, "y = 2", "c1")
        _idle(log, "c1")
        assert log.poll("job-1")["status"] == "running"
        # Once it has begun, it is one cell of the one-at-a-time sequence.
        _input(log, "x = 1", "h1", session="host")
        _input(log, "z = 3", "c2")
        _idle(log, "c2")
        assert log.poll("job-1")["status"] == "error"

    def test_a_cell_does_not_end_a_task(self):
        # A task runs beside the cells (run_async).
        log = self._log()
        _start(log, job_id="task-1", request="c0")
        _input(log, "x = 1", "c1")
        _idle(log, "c1")
        assert log.poll("task-1")["status"] == "running"

    def test_a_task_takes_its_cells_output_and_writer(self):
        log = self._log()
        log.start_cell(log.new_id(), "c1", "run_async(f)", origin="chat")
        _print(log, "before\n", request="c1")
        _start(log, job_id="task-1", request="c1", origin="mcp")
        _print(log, "from the task\n", request="c1")
        _idle(log, "c1")
        assert log.poll("job-1")["status"] == "ok"
        assert log.poll("job-1")["stdout"] == "before\n"
        task = log.poll("task-1")
        assert task["stdout"] == "from the task\n"
        assert task["origin"] == "chat"
        assert task["status"] == "running"

    def test_a_host_cell_ends_on_its_reply_if_its_idle_is_lost(self):
        log = self._log()
        log.start_cell(log.new_id(), "c1", "1/0", origin="mcp")
        log.note_reply("job-1", {"status": "error", "ename": "ZeroDivisionError"})
        log.cell_replied("job-1")
        assert log.poll("job-1")["status"] == "error"

    def test_the_reply_fails_a_cell_whose_iopub_error_was_lost(self):
        # Its idle ended it "ok"; the reply, which cannot be lost, says not.
        log = self._log()
        log.start_cell(log.new_id(), "c1", "1/0", origin="mcp")
        _input(log, "1/0", "c1", session="host")
        _idle(log, "c1")
        assert log.poll("job-1")["status"] == "ok"
        log.note_reply(
            "job-1",
            {"status": "error", "ename": "ZeroDivisionError", "traceback": ["Zero"]},
        )
        snap = log.poll("job-1")
        assert snap["status"] == "error" and snap["error_text"] == "Zero"

    def test_the_outcome_is_light_unless_asked_for_in_full(self):
        log = self._log()
        log.start_cell(log.new_id(), "c1", "print('one')\n1", origin="mcp")
        _print(log, "one\n", request="c1")
        log.on_iopub(_msg("execute_result", "c1", data={"text/plain": "1"}))
        _idle(log, "c1")
        light = log.outcome("job-1")
        assert light["status"] == "ok" and "stdout" not in light
        assert light["stdout_head"] == "one" and light["stdout_len"] == 4
        full = log.outcome("job-1", full=True)
        assert full["stdout"] == "one\n" and full["result_text"] == "1"
        assert log.outcome("job-9") is None

    def test_the_viewer_window_comes_back_on_the_reply(self):
        log = self._log()
        log.start_cell(log.new_id(), "c1", "x", origin="mcp")
        assert log.window_alive("job-1") is None
        log.note_reply(
            "job-1",
            {
                "status": "ok",
                "user_expressions": {
                    "w": {"status": "ok", "data": {"text/plain": "False"}}
                },
            },
        )
        assert log.window_alive("job-1") is False

    def test_a_submits_idle_does_not_end_its_task(self):
        # The task outlives the request that started it.
        log = self._log()
        _start(log, job_id=log.new_id(), request="submit-1")
        _idle(log, "submit-1")
        assert log.poll("job-1")["status"] == "running"

    def test_the_kernel_names_a_cell_by_request_and_a_task_by_id(self):
        log = self._log()
        _input(log, "loop()", "c1")
        (rec,) = log.export()
        assert log.stop_key(rec["job_id"]) == "c1"
        _start(log, job_id="task-1", request="c1")
        assert log.stop_key("task-1") == "task-1"
        _idle(log, "c1")
        assert log.stop_key(rec["job_id"]) is None


class TestLostAndGone:
    def test_a_new_start_ends_a_record_whose_end_was_lost(self):
        # Only one job runs at a time, so a start proves the last one is over.
        log = JobLog()
        _start(log)
        _start(log, job_id="job-2", request="req-2")
        assert log.poll("job-1")["status"] == "error"
        assert "not recorded" in log.poll("job-1")["error_text"]
        assert log.poll("job-2")["status"] == "running"

    def test_the_kernel_going_ends_what_it_was_running(self):
        log = JobLog()
        _start(log)
        _print(log, "partial\n")
        log.kernel_gone("the user closed the window")
        snap = log.poll("job-1")
        assert snap["status"] == "interrupted"
        assert "the user closed the window" in snap["error_text"]
        # Kept: the record outlives its kernel.
        assert snap["stdout"] == "partial\n"

    def test_a_late_end_does_not_reopen_a_record(self):
        log = JobLog()
        _start(log)
        log.kernel_gone()
        _end(log, status="ok")
        assert log.poll("job-1")["status"] == "interrupted"

    def test_ids_are_the_hosts_and_never_repeat(self):
        log = JobLog(host_session="host")
        assert log.new_id() == "job-1"
        _input(log, "x = 1", request="c1")
        assert log.new_id() == "job-3"


class TestDigest:
    def test_reports_only_unseen_foreign_jobs(self):
        log = JobLog()
        _start(log, job_id="job-1", request="r1")
        _end(log, job_id="job-1")
        _start(log, job_id="job-2", request="r2", origin="user")
        _end(log, job_id="job-2")
        digest = log.foreign_digest("mcp")
        assert [(d["job_id"], d["origin"], d["status"]) for d in digest] == [
            ("job-2", "user", "ok")
        ]
        # Reading never consumes; the ack does.
        assert len(log.foreign_digest("mcp")) == 1
        assert log.ack_foreign_digest(["job-2"]) == 1
        assert log.foreign_digest("mcp") == []

    def test_the_chat_loop_is_not_told_about_its_own_cells(self):
        log = JobLog()
        _start(log, job_id="job-1", request="r1", origin="chat")
        _end(log, job_id="job-1")
        _start(log, job_id="job-2", request="r2", origin="user")
        _end(log, job_id="job-2")
        assert [d["job_id"] for d in log.foreign_digest("chat")] == ["job-2"]
        assert [d["job_id"] for d in log.foreign_digest("mcp")] == ["job-1", "job-2"]


class TestPointOfView:
    """The digest and the eviction hold read "foreign" from the agent's own
    origin: the last non-user writer to run a cell."""

    def _cell(self, log, origin, code="a = 1"):
        """An ended cell of *origin*: the host's own, or a user's."""
        request = f"r{len(log._records) + 1}"
        if origin == "user":
            _input(log, code, request)
        else:
            log.start_cell(log.new_id(), request, code, origin=origin)
            _input(log, code, request, session="host")
        _idle(log, request)
        return log.export()[-1]["job_id"]

    def _fill(self, log, origin):
        for _ in range(_job_log._MAX_RETAINED_JOBS + 5):
            self._cell(log, origin)

    def _log(self):
        return JobLog(host_session="host")

    def test_a_running_user_cell_stays_in_the_digest_until_it_ends(self):
        # Reading never consumes, so a running cell stays reported: otherwise
        # the agent would hear that a cell started and never learn how it
        # ended. (Excluding it from the ack is the caller's job.)
        log = self._log()
        _input(log, "loop()", "c1")
        (entry,) = log.foreign_digest("mcp")
        assert entry["status"] == "running"
        _idle(log, "c1")
        assert [d["status"] for d in log.foreign_digest("mcp")] == ["ok"]
        log.ack_foreign_digest([entry["job_id"]])
        assert log.foreign_digest("mcp") == []

    def test_prune_never_evicts_an_unreported_user_cell(self):
        # The digest entry is the agent's only notice that its namespace
        # changed under it; evicting the record would drop the notice.
        log = self._log()
        user = self._cell(log, "user")
        self._fill(log, "mcp")
        assert user in log._records
        log.ack_foreign_digest([user])
        # Once reported it is an ordinary record again, and prunes normally.
        self._fill(log, "mcp")
        assert user not in log._records

    def test_a_chat_cell_is_foreign_to_the_mcp_agent_as_a_users_is(self):
        log = self._log()
        chat = self._cell(log, "chat")
        self._cell(log, "mcp")
        assert [(d["job_id"], d["origin"]) for d in log.foreign_digest("mcp")] == [
            (chat, "chat")
        ]
        self._fill(log, "mcp")
        assert chat in log._records

    def test_a_chat_sessions_own_cells_are_evicted(self):
        # From a fixed "mcp" every chat cell was foreign and could never be
        # acked, so the cap bounded nothing (biopb/biopb#879).
        log = self._log()
        self._fill(log, "chat")
        assert len(log._records) == _job_log._MAX_RETAINED_JOBS

    def test_a_chat_session_still_holds_the_users_unreported_cell(self):
        log = self._log()
        user = self._cell(log, "user")
        self._fill(log, "chat")
        assert user in log._records
        assert [d["job_id"] for d in log.foreign_digest("chat")] == [user]
        assert log.ack_foreign_digest([user]) == 1
        self._fill(log, "chat")
        assert user not in log._records


class TestOutputCap:
    """The bound ``_MAX_RETAINED_JOBS`` is not: that caps how many records are
    kept, not how large one gets."""

    @pytest.fixture(autouse=True)
    def small_cap(self, monkeypatch):
        """A tiny cap, so a test does not have to print 200k characters."""
        monkeypatch.setattr(_job_log, "_MAX_JOB_OUTPUT_CHARS", 100)

    def _loud(self):
        log = JobLog()
        _start(log)
        for _ in range(100):
            _print(log, "xxx\n")
        _end(log)
        return log

    def test_output_under_the_cap_is_untouched(self):
        log = JobLog()
        _start(log)
        _print(log, "hi\n")
        snap = log.poll("job-1")
        assert snap["stdout"] == "hi\n"
        assert snap["stdout_dropped"] == 0
        assert snap["stdout_total"] == 3

    def test_a_runaway_cell_keeps_only_its_tail(self):
        snap = self._loud().poll("job-1")
        assert snap["stdout_dropped"] > 0
        assert len(snap["stdout"]) < 400
        # The newest output survives: while a cell runs that is the part worth
        # having, which is why the detail view keeps the tail too.
        assert snap["stdout"].endswith("xxx\n")

    def test_the_record_says_it_is_partial(self):
        # Marked on read rather than stored -- a marker written into the buffer
        # would itself be compacted away by the next rewrite.
        assert "earlier chars dropped" in self._loud().poll("job-1")["stdout"]

    def test_the_total_stays_monotonic_across_compaction(self):
        # What a reader streaming output as it grows has to diff against.
        snap = self._loud().poll("job-1")
        assert snap["stdout_total"] == 400
        assert snap["stdout_total"] > len(snap["stdout"])

    def test_the_row_reports_what_was_printed_not_what_was_kept(self):
        (row,) = self._loud().summary()
        assert row["stdout_len"] == 400


def test_one_line_helper():
    assert _job_log._one_line("") == ""
    assert _job_log._one_line("\n\n  hello  \nworld") == "hello"
    capped = _job_log._one_line("x" * 100)
    assert len(capped) == 80 and capped.endswith("…")
