"""The host's job records (``_job_log``), fed synthetic iopub messages.

The kernel side of the announcements is ``test_mcp_jobs``; a real kernel
publishing them is ``test_mcp_kernel`` (``TestHostRecords``).
"""

import pytest

from biopb_mcp._tests.conftest import iopub_event, kernel_snapshot
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

    def test_a_foreign_cells_traceback_is_its_error(self):
        # The end announcement carries "ename: evalue"; the iopub error the
        # cell's own client got carries the traceback, which says where.
        log = JobLog()
        _start(log, origin="user")
        log.on_iopub(
            _msg(
                "error", "req-1", traceback=["\x1b[31mZeroDivisionError\x1b[0m", "at 1"]
            )
        )
        _end(log, status="error", error_text="ZeroDivisionError: division by zero")
        assert log.poll("job-1")["error_text"] == "ZeroDivisionError\nat 1"

    def test_a_foreign_cells_result_is_its_execute_result(self):
        log = JobLog()
        _start(log, origin="user")
        log.on_iopub(_msg("execute_result", "req-1", data={"text/plain": "42"}))
        _end(log)
        assert log.poll("job-1")["result_text"] == "42"

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
            "verify",
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
        _input(log, "_jobs.submit('x')", "h1", session="host")
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

    def test_a_refused_cell_is_not_recorded_and_proves_nothing(self):
        # The gate refuses a cell while a task runs, after ipykernel echoed it.
        log = self._log()
        _start(log, job_id=log.new_id(), request="submit-1")
        _input(log, "y = 1", "c1")
        log.on_iopub(_msg("error", "c1", ename="KernelBusy", traceback=["busy"]))
        _idle(log, "c1")
        assert [r["job_id"] for r in log.export()] == ["job-1"]
        assert log.poll("job-1")["status"] == "running"

    def test_a_cell_that_ran_ends_a_task_whose_end_was_lost(self):
        log = self._log()
        _start(log, job_id=log.new_id(), request="submit-1")
        _input(log, "x = 1", "c1")
        _idle(log, "c1")
        assert log.poll("job-1")["status"] == "error"
        assert "not recorded" in log.poll("job-1")["error_text"]

    def test_a_submits_idle_does_not_end_its_task(self):
        # The task outlives the request that started it.
        log = self._log()
        _start(log, job_id=log.new_id(), request="submit-1")
        _idle(log, "submit-1")
        assert log.poll("job-1")["status"] == "running"

    def test_a_stop_is_named_by_the_request(self):
        log = self._log()
        _input(log, "loop()", "c1")
        (rec,) = log.export()
        assert log.request_of(rec["job_id"]) == "c1"
        assert log.job_of("c1") == rec["job_id"]
        _idle(log, "c1")
        assert log.request_of(rec["job_id"]) is None


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


class TestSettle:
    """The kernel's own account, fetched on a shell reply, settles what iopub
    lost."""

    def test_a_lost_end_is_settled(self):
        log = JobLog()
        _start(log)
        _print(log, "partial\n")
        log.settle(kernel_snapshot(status="error", error_text="boom"))
        snap = log.poll("job-1")
        assert snap["status"] == "error"
        assert snap["error_text"] == "boom"
        assert snap["stdout"].startswith("partial\n")
        assert "lost on iopub" in snap["stdout"]

    def test_a_lost_start_is_replayed_and_a_running_job_stays_running(self):
        log = JobLog()
        log.settle(kernel_snapshot(status="running"))
        assert log.poll("job-1")["status"] == "running"
        # Its output, now that the request is known, is filed under it.
        _print(log, "later\n")
        assert log.poll("job-1")["stdout"] == "later\n"

    def test_a_verification_gets_its_cells_from_the_kernel(self):
        log = JobLog()
        log.settle(
            kernel_snapshot(status="error", cells=[("a = 1", "ok"), ("1/0", "error")])
        )
        cells = log.verify_record("job-1")["cells"]
        assert [c["status"] for c in cells] == ["ok", "error"]
        assert log.poll("job-1")["status"] == "error"

    def test_a_settled_record_is_not_touched_again(self):
        log = JobLog()
        _start(log)
        _end(log, status="ok")
        log.settle(kernel_snapshot(status="error"))
        assert log.poll("job-1")["status"] == "ok"
        assert "lost" not in log.poll("job-1")["stdout"]

    def test_the_late_original_start_does_not_reopen_it(self):
        log = JobLog()
        log.settle(kernel_snapshot(status="ok"))
        _start(log)
        assert log.poll("job-1")["status"] == "ok"


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


class TestVerification:
    """A scratch kernel's run: one stream, split at the cell boundaries it
    announces."""

    def _verify(self, log, cells=("a = 1", "print(a)")):
        _start(
            log,
            code="\n\n".join(cells),
            verify={"title": "wf", "cells": list(cells), "created": 1.0},
        )

    def _cell(self, log, i, status="ok", **kw):
        log.on_iopub(_event(event="cell_start", job_id="job-1", index=i))
        if "out" in kw:
            _print(log, kw.pop("out"))
        log.on_iopub(
            _event(
                event="cell_end",
                job_id="job-1",
                index=i,
                status=status,
                elapsed=0.1,
                **kw,
            )
        )

    def test_output_is_split_per_cell_and_kept_whole(self):
        log = JobLog()
        self._verify(log)
        self._cell(log, 0, out="one\n")
        self._cell(log, 1, out="two\n", result_text="1")
        _end(log)
        full = log.verify_record("job-1")
        assert [c["stdout"] for c in full["cells"]] == ["one\n", "two\n"]
        assert full["cells"][1]["result_text"] == "1"
        assert full["status"] == "ok"
        assert log.poll("job-1")["stdout"] == "one\ntwo\n"

    def test_the_polled_record_carries_heads(self):
        log = JobLog()
        self._verify(log)
        self._cell(log, 0, out="one\n")
        (cell, _second) = log.poll("job-1")["verify"]["cells"]
        assert "stdout" not in cell
        assert cell["stdout_head"] == "one" and cell["stdout_len"] == 4

    def test_the_cells_a_failure_never_reached_are_skipped(self):
        log = JobLog()
        self._verify(log)
        self._cell(log, 0, status="error", error_text="Traceback ...")
        _end(log, status="error")
        cells = log.poll("job-1")["verify"]["cells"]
        assert [c["status"] for c in cells] == ["error", "skipped"]
        assert cells[0]["error_text"] == "Traceback ..."

    def test_a_kernel_death_fails_the_cell_it_died_in(self):
        # The OOM a verification exists to catch: the cell it died in is the
        # failure, and the rest never ran.
        log = JobLog()
        self._verify(log)
        log.on_iopub(_event(event="cell_start", job_id="job-1", index=0))
        log.kernel_gone("the scratch kernel died")
        cells = log.verify_record("job-1")["cells"]
        assert [c["status"] for c in cells] == ["error", "skipped"]
        assert "died" in cells[0]["error_text"]

    def test_an_ordinary_job_has_no_verification(self):
        log = JobLog()
        _start(log)
        assert log.poll("job-1")["verify"] is None
        assert log.verify_record("job-1") is None


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
