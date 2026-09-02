"""Tests for the async submit->poll execution model (review finding B2).

Three layers:

* ``TestJobRunnerUnit`` / ``TestJobOrigin`` — the in-kernel job runner driven
  directly with a fake InteractiveShell (no kernel, fast): submit/poll/interrupt,
  output capture, distributed future-cancel, and the agent/user ``origin`` split
  (``docs/user-console.md``).
* ``TestJobConcurrency`` — a real *bare* kernel (no napari/display): proves the
  kernel main thread stays free while a background job runs (the agent is no
  longer blind).
* ``TestNapariJobs`` — display-gated end-to-end: viewer mutation from a worker
  thread (main-thread marshaling), screenshot/status mid-job, restart clears
  jobs.
"""

import os
import re
import sys
import time
import types

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from biopb_mcp._tests.conftest import call_tool as _tool
from biopb_mcp.mcp import _app, _jobs, _kernel_rpc, _server, _writers  # noqa: E402
from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


def _job_result(stdout):
    """Unwrap the ``{"r": result, "w": window_alive}`` job-snippet envelope."""
    payload = _kernel_rpc._extract_json(stdout)
    return payload["r"] if payload else None


@pytest.fixture
def runner():
    """The in-kernel job runner wired to a fake InteractiveShell (no kernel).

    Defined at module level so the runner classes below share one definition;
    each test still gets a fresh namespace and a cleared job table.
    """
    ns = {
        "_dask_client": None,
        "_conn": types.SimpleNamespace(client=None),
    }
    _jobs.install(types.SimpleNamespace(user_ns=ns))
    yield ns
    _jobs.reset()


def _wait_job(job_id, timeout=5.0):
    """Block until *job_id* leaves ``running``, and return its snapshot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = _jobs.poll(job_id)
        if snap["status"] != "running":
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


# ---------------------------------------------------------------------------
# Unit: in-kernel job runner with a fake shell (no real kernel)
# ---------------------------------------------------------------------------


class TestJobRunnerUnit:
    _wait = staticmethod(_wait_job)

    def test_quick_job_captures_stdout_and_result(self, runner):
        jid = _jobs.submit("print('hello'); 1 + 2")["job_id"]
        snap = self._wait(jid)
        assert snap["status"] == "ok"
        assert snap["stdout"] == "hello\n"
        assert snap["result_text"] == "3"
        # The refresh prefix ran: client mirrors _conn.client.
        assert runner["client"] is None

    def test_statement_only_has_no_result_text(self, runner):
        jid = _jobs.submit("x = 41 + 1")["job_id"]
        snap = self._wait(jid)
        assert snap["status"] == "ok"
        assert snap["result_text"] == ""

    def test_error_is_captured(self, runner):
        jid = _jobs.submit("raise ValueError('boom')")["job_id"]
        snap = self._wait(jid)
        assert snap["status"] == "error"
        assert "ValueError" in snap["error_text"]

    def test_one_job_at_a_time(self, runner):
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        try:
            busy = _jobs.submit("1 + 1")
            assert busy.get("error") == "busy"
            assert busy["running_job_id"] == jid
        finally:
            _jobs.interrupt_current()
        self._wait(jid)

    def test_distributed_cancel_rebuilds_futures(self, runner):
        # _cancel() must rebuild real Future objects from dc.futures' string
        # keys: Client.cancel() filters its arg through futures_of(), which
        # silently drops bare strings -- so passing list(dc.futures) cancels
        # nothing.  Assert real Futures (resolvable by futures_of) + force=True.
        from distributed import Future
        from distributed.client import futures_of

        calls = {}

        class _Loop:
            def add_callback(self, fn, *a, **k):  # swallow Future.release()
                pass

        class _StubClient:
            futures = {"('grad', 0, 0)": object(), "('grad', 1, 0)": object()}
            generation = 0
            loop = _Loop()

            def _inc_ref(self, key):
                pass

            def _dec_ref(self, key):
                pass

            def cancel(self, futures, force=False):
                calls["futures"] = list(futures)
                calls["force"] = force

        runner["_dask_client"] = _StubClient()
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        time.sleep(0.05)
        _jobs._cancel_dask_futures(_jobs._jobs[jid])
        passed = calls["futures"]
        assert passed and all(isinstance(f, Future) for f in passed)
        assert {f.key for f in futures_of(passed)} == set(_StubClient.futures)
        assert calls["force"] is True
        _jobs.interrupt_current()  # actually stop the uncooperative loop
        self._wait(jid)

    def test_poll_unknown_job(self, runner):
        assert _jobs.poll("job-999")["status"] == "unknown"

    def test_reset_clears_registry(self, runner):
        jid = _jobs.submit("1")["job_id"]
        self._wait(jid)
        assert _jobs.jobs_summary()
        _jobs.reset()
        assert _jobs.jobs_summary() == []

    # -- user-action attribution --------------------------------------------

    def test_interrupt_current_stops_uncooperative_job(self, runner):
        # A pure-Python loop is stoppable only by a KeyboardInterrupt raised into
        # the worker thread (short of restart); interrupt_current also threads a
        # user-supplied reason into the finalized record.
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        while _jobs.poll(jid)["status"] != "running":
            time.sleep(0.02)
        out = _jobs.interrupt_current(reason="forced by Bob")
        assert out["job_id"] == jid and out["interrupted"] is True
        snap = self._wait(jid)
        assert snap["status"] == "interrupted"
        assert snap["cancel_reason"] == "forced by Bob"
        assert "forced by Bob" in snap["error_text"]
        assert "KeyboardInterrupt" in snap["error_text"]

    def test_interrupt_current_when_idle(self):
        _jobs.reset()
        assert _jobs.interrupt_current("x") == {
            "job_id": None,
            "interrupted": False,
            "status": "idle",
        }

    def test_raise_in_thread_no_ident(self):
        assert _jobs._raise_in_thread(None, KeyboardInterrupt) == 0

    def test_external_interrupt_is_labeled_not_reported_as_an_error(self, runner):
        # A KeyboardInterrupt this runner did not raise -- an external SIGINT
        # relayed through a run_on_main slot. Unlabeled it renders as the
        # submitted code failing, which for a user's cell reads as their own
        # code breaking rather than as a stop someone else caused.
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        while _jobs.poll(jid)["status"] != "running":
            time.sleep(0.02)
        job = _jobs._jobs[jid]
        assert _jobs._raise_in_thread(job.thread.ident, KeyboardInterrupt) == 1
        snap = self._wait(jid)

        assert snap["status"] == "interrupted"  # not "error"
        assert snap["cancel_reason"] == _jobs._EXTERNAL_INTERRUPT_MSG
        # The reason is prefixed onto the traceback, so poll_job / execute_code
        # render the attribution rather than a bare KeyboardInterrupt.
        assert snap["error_text"].startswith(_jobs._EXTERNAL_INTERRUPT_MSG)
        assert "KeyboardInterrupt" in snap["error_text"]

    def test_an_owned_interrupt_keeps_its_own_reason(self, runner):
        # interrupt_current still owns the attribution when it is the cause --
        # the external path must not overwrite a reason someone else set.
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        while _jobs.poll(jid)["status"] != "running":
            time.sleep(0.02)
        _jobs.interrupt_current(reason="forced by Bob")
        snap = self._wait(jid)
        assert snap["status"] == "interrupted"
        assert snap["cancel_reason"] == "forced by Bob"
        assert _jobs._EXTERNAL_INTERRUPT_MSG not in snap["error_text"]

    # -- submitted code is recorded (observe UI) ----------------------------

    def test_job_stores_submitted_code(self, runner):
        src = "x = 1 + 1\nprint('hi')"
        jid = _jobs.submit(src)["job_id"]
        snap = self._wait(jid)
        assert snap["code"] == src

    def test_jobs_summary_has_code_preview(self, runner):
        jid = _jobs.submit("\n\n  print('first real line')  \nmore = 2")["job_id"]
        self._wait(jid)
        summ = {j["job_id"]: j for j in _jobs.jobs_summary()}[jid]
        assert summ["code_preview"] == "print('first real line')"

    def test_jobs_summary_has_intent_preview(self, runner):
        # What the row actually shows. Empty when nobody said why, which is the
        # case the UI falls back to the code line for.
        jid = _jobs.submit("x = 1", intent="isolate the nuclei channel")["job_id"]
        bare = _jobs.submit("y = 2")["job_id"]
        self._wait(jid)
        self._wait(bare)
        summ = {j["job_id"]: j for j in _jobs.jobs_summary()}
        assert summ[jid]["intent_preview"] == "isolate the nuclei channel"
        assert summ[bare]["intent_preview"] == ""

    def test_one_line_helper(self):
        assert _jobs._one_line("") == ""
        assert _jobs._one_line("\n\n  hello  \nworld") == "hello"
        capped = _jobs._one_line("x" * 100)
        assert len(capped) == 80 and capped.endswith("…")


# ---------------------------------------------------------------------------
# Two writers: the agent and a human sharing one kernel (docs/user-console.md)
# ---------------------------------------------------------------------------


class TestJobOrigin:
    """The `origin` field and everything that reads it."""

    _wait = staticmethod(_wait_job)

    def test_origin_defaults_to_agent_and_rides_the_snapshot(self, runner):
        jid = _jobs.submit("x = 1")["job_id"]
        snap = self._wait(jid)
        assert snap["origin"] == "mcp"
        summ = {j["job_id"]: j for j in _jobs.jobs_summary()}[jid]
        assert summ["origin"] == "mcp"
        # export() feeds the notebook writer -- provenance must survive there too.
        assert {e["job_id"]: e for e in _jobs.export()}[jid]["origin"] == "mcp"

    def test_busy_reports_who_is_running(self, runner):
        jid = _jobs.submit(
            "import time\nwhile True:\n    time.sleep(0.02)", origin="user"
        )["job_id"]
        try:
            busy = _jobs.submit("1 + 1")
            assert busy["error"] == "busy"
            assert busy["running_job_id"] == jid
            # Without this the agent cannot tell whose job it collided with, and
            # the advice it gets ("stop it") would be wrong.
            assert busy["running_job_origin"] == "user"
        finally:
            _jobs.interrupt_current()
        self._wait(jid)

    def test_agent_is_refused_a_user_job(self, runner):
        jid = _jobs.submit(
            "import time\nwhile True:\n    time.sleep(0.02)", origin="user"
        )["job_id"]
        try:
            res = _jobs.interrupt_current(requester="mcp")
            assert res == {
                "job_id": jid,
                "interrupted": False,
                "status": "running",
                "refused": "foreign_job",
                "origin": "user",
            }
            # Refused means untouched, not merely unreported.
            assert _jobs.poll(jid)["status"] == "running"
        finally:
            _jobs.interrupt_current()
        self._wait(jid)

    def test_user_may_stop_an_agent_job(self, runner):
        # The converse of the rule above: a person can stop anything in their
        # own session, which is what the observe UI's Interrupt does.
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        res = _jobs.interrupt_current(reason="stopped by the user")
        assert res["interrupted"] is True
        snap = self._wait(jid)
        assert snap["status"] == "interrupted"
        assert "stopped by the user" in snap["error_text"]

    def test_agent_may_stop_its_own_job(self, runner):
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        assert _jobs.interrupt_current(requester="mcp")["interrupted"] is True
        assert self._wait(jid)["status"] == "interrupted"

    def test_digest_reports_only_unseen_user_jobs(self, runner):
        self._wait(_jobs.submit("a = 1", origin="mcp")["job_id"])
        user_jid = self._wait(_jobs.submit("b = 2", origin="user")["job_id"])["job_id"]

        digest = _jobs.foreign_digest()
        assert [d["job_id"] for d in digest] == [user_jid]
        assert digest[0]["status"] == "ok"
        # The entry names its writer: the caller words the notice differently
        # for a person than for another agent.
        assert digest[0]["origin"] == "user"

        # Reading never consumes; only an explicit ack does, so a finished
        # user job is reported exactly once.
        assert [d["job_id"] for d in _jobs.foreign_digest()] == [user_jid]
        assert _jobs.ack_foreign_digest([user_jid]) == 1
        assert _jobs.foreign_digest() == []

    def test_running_user_job_stays_in_the_digest_until_it_ends(self, runner):
        jid = _jobs.submit(
            "import time\nwhile True:\n    time.sleep(0.02)", origin="user"
        )["job_id"]
        try:
            # Reading never consumes, so a running cell stays reported:
            # otherwise the agent would hear that a cell started and never
            # learn how it ended. (Excluding it from the ack is the *caller's*
            # job -- _writers._ack_foreign_digest filters on the reported status,
            # because re-reading it here is the race this split closes.)
            assert _jobs.foreign_digest()[0]["status"] == "running"
            assert _jobs.foreign_digest()[0]["status"] == "running"
        finally:
            _jobs.interrupt_current()
        self._wait(jid)
        final = _jobs.foreign_digest()
        assert [d["status"] for d in final] == ["interrupted"]
        _jobs.ack_foreign_digest([jid])
        assert _jobs.foreign_digest() == []

    def test_prune_never_evicts_an_unreported_user_job(self, runner):
        # The digest entry is the agent's only notice that its namespace changed
        # under it; evicting the record would silently drop the notice.
        user_jid = self._wait(_jobs.submit("b = 2", origin="user")["job_id"])["job_id"]
        for _ in range(_jobs._MAX_RETAINED_JOBS + 5):
            self._wait(_jobs.submit("a = 1")["job_id"])

        assert user_jid in _jobs._jobs
        assert [d["job_id"] for d in _jobs.foreign_digest()] == [user_jid]
        _jobs.ack_foreign_digest([user_jid])

        # Once reported it is an ordinary record again, and prunes normally.
        for _ in range(_jobs._MAX_RETAINED_JOBS + 5):
            self._wait(_jobs.submit("a = 1")["job_id"])
        assert user_jid not in _jobs._jobs

    def test_the_chat_loop_is_not_told_about_its_own_cells(self, runner):
        # "Someone else's cell" is a relation, not a property of the cell. Read
        # from the MCP client's fixed point of view -- the only one there used
        # to be -- the loop was handed its own cells as another writer's, and
        # acking that discharged the user's notices along with them.
        chat_jid = self._wait(_jobs.submit("a = 1", origin="chat")["job_id"])["job_id"]
        user_jid = self._wait(_jobs.submit("b = 2", origin="user")["job_id"])["job_id"]

        assert [d["job_id"] for d in _jobs.foreign_digest("chat")] == [user_jid]
        # The MCP client's view is unchanged: both of those are someone else's.
        assert [d["job_id"] for d in _jobs.foreign_digest()] == [chat_jid, user_jid]

    def test_a_chat_job_is_foreign_to_the_agent_just_as_a_user_cell_is(self, runner):
        # Every rule that reads origin means "not the agent", not "the user" --
        # they were the same set until a third writer existed. A chat job the
        # agent has not been told about must therefore be digested and held
        # against eviction exactly like a human's cell.
        chat_jid = self._wait(_jobs.submit("b = 2", origin="chat")["job_id"])["job_id"]
        assert [(d["job_id"], d["origin"]) for d in _jobs.foreign_digest()] == [
            (chat_jid, "chat")
        ]
        for _ in range(_jobs._MAX_RETAINED_JOBS + 5):
            self._wait(_jobs.submit("a = 1")["job_id"])
        assert chat_jid in _jobs._jobs

        assert _jobs.ack_foreign_digest([chat_jid]) == 1
        assert _jobs.foreign_digest() == []

    def test_agent_is_refused_a_chat_job(self, runner):
        jid = _jobs.submit(
            "import time\nwhile True:\n    time.sleep(0.02)", origin="chat"
        )["job_id"]
        try:
            assert _jobs.interrupt_current(requester="mcp") == {
                "job_id": jid,
                "interrupted": False,
                "status": "running",
                "refused": "foreign_job",
                "origin": "chat",
            }
        finally:
            _jobs.interrupt_current()
        assert self._wait(jid)["status"] == "interrupted"


class TestKernelOwner:
    """One agent per kernel: the first non-user submitter claims it."""

    _wait = staticmethod(_wait_job)

    def test_first_agent_claims_and_a_second_is_refused(self, runner):
        jid = _jobs.submit("a = 1", writer="sess-A", writer_label="claude-code")[
            "job_id"
        ]
        self._wait(jid)
        assert _jobs.owner() == {"owner": "sess-A", "label": "claude-code"}

        refused = _jobs.submit("b = 2", writer="sess-B")
        # The id as well as the label: the server mirrors the claim, and a
        # refusal is its chance to correct a mirror that guessed wrong.
        assert refused == {
            "error": "not_owner",
            "owner": "claude-code",
            "owner_id": "sess-A",
        }
        # Refused at the door: no record, so nothing to poll or export either.
        assert [j["code"] for j in _jobs.export()] == ["a = 1"]

        # The owner keeps working.
        assert (
            self._wait(_jobs.submit("c = 3", writer="sess-A")["job_id"])["status"]
            == "ok"
        )

    def test_a_human_cell_is_never_gated(self, runner):
        # The person at the machine has standing no client does -- and the
        # observe console has no identity to gate on in the first place.
        self._wait(_jobs.submit("a = 1", writer="sess-A")["job_id"])
        jid = self._wait(_jobs.submit("b = 2", origin="user")["job_id"])["job_id"]
        assert _jobs._jobs[jid].status == "ok"
        # Running one does not steal the claim from the agent that holds it.
        assert _jobs.owner()["owner"] == "sess-A"

    def test_a_caller_with_no_identity_neither_claims_nor_is_checked(self, runner):
        # Direct in-process calls (these tests, an in-process chat loop) have no
        # request and no client id; there is nothing to tell two of them apart
        # with, so the rule does not apply rather than misfiring.
        self._wait(_jobs.submit("a = 1")["job_id"])
        assert _jobs.owner()["owner"] is None
        self._wait(_jobs.submit("b = 2")["job_id"])

        # ...and an identified client can still claim afterwards.
        self._wait(_jobs.submit("c = 3", writer="sess-A")["job_id"])
        assert _jobs.owner()["owner"] == "sess-A"

    def test_a_non_owner_cannot_stop_the_owners_job(self, runner):
        # Stopping a job changes kernel state, so it is gated like running one.
        jid = _jobs.submit(
            "import time\nwhile True:\n    time.sleep(0.02)", writer="sess-A"
        )["job_id"]
        try:
            assert _jobs.interrupt_current(requester="mcp", writer="sess-B") == {
                "job_id": jid,
                "interrupted": False,
                "status": "running",
                "refused": "not_owner",
            }
            # The owner still can, and so can the human (the default requester).
            assert (
                _jobs.interrupt_current(requester="mcp", writer="sess-A")["interrupted"]
                is True
            )
        finally:
            _jobs.interrupt_current()
        assert self._wait(jid)["status"] == "interrupted"

    def test_a_non_owner_may_read_the_digest_but_not_discharge_it(self, runner):
        # A watching client's poll_job carries the same digest round trip. It may
        # see what ran -- but acking would retire a notice the holder never
        # received, and the holder is promised it exactly once.
        self._wait(_jobs.submit("a = 1", writer="sess-A")["job_id"])
        user_jid = self._wait(_jobs.submit("b = 2", origin="user")["job_id"])["job_id"]

        assert [d["job_id"] for d in _jobs.foreign_digest()] == [user_jid]
        assert _jobs.ack_foreign_digest([user_jid], writer="sess-B") == 0
        assert [d["job_id"] for d in _jobs.foreign_digest()] == [user_jid]

        assert _jobs.ack_foreign_digest([user_jid], writer="sess-A") == 1
        assert _jobs.foreign_digest() == []

    def test_the_foreign_refusal_names_the_writer(self, runner):
        # "Foreign" stopped being a synonym for "the user's" when a third writer
        # appeared; a caller that assumes otherwise tells the agent to wait on a
        # person who is not there.
        jid = _jobs.submit(
            "import time\nwhile True:\n    time.sleep(0.02)", origin="chat"
        )["job_id"]
        try:
            refused = _jobs.interrupt_current(requester="mcp")
            assert refused["refused"] == "foreign_job"
            assert refused["origin"] == "chat"
        finally:
            _jobs.interrupt_current()
        self._wait(jid)

    def test_the_human_can_always_stop_a_held_kernel(self, runner):
        # The recovery belongs to the person at the machine: requester="user" is
        # the observe UI, and it is never gated on the claim.
        jid = _jobs.submit(
            "import time\nwhile True:\n    time.sleep(0.02)", writer="sess-A"
        )["job_id"]
        assert _jobs.interrupt_current(reason="stopped by Bob")["interrupted"] is True
        assert self._wait(jid)["status"] == "interrupted"

    def test_reset_releases_the_claim(self, runner):
        # install() calls reset() on every bootstrap, so the claim lasts exactly
        # one kernel lifetime -- restart_kernel is the documented takeover.
        self._wait(_jobs.submit("a = 1", writer="sess-A")["job_id"])
        _jobs.reset()
        assert _jobs.owner()["owner"] is None
        self._wait(_jobs.submit("b = 2", writer="sess-B")["job_id"])
        assert _jobs.owner()["owner"] == "sess-B"


class TestJobIntent:
    """The `intent` field: recorded with the job, never acted on."""

    _wait = staticmethod(_wait_job)

    def test_intent_defaults_empty_and_rides_the_snapshot(self, runner):
        assert self._wait(_jobs.submit("x = 1")["job_id"])["intent"] == ""

        jid = _jobs.submit("y = 2", intent="check the drift estimate")["job_id"]
        assert self._wait(jid)["intent"] == "check the drift estimate"
        # export() feeds the notebook writer, which is the whole point of the
        # field -- it has to survive the trip.
        by_id = {e["job_id"]: e for e in _jobs.export()}
        assert by_id[jid]["intent"] == "check the drift estimate"

    def test_intent_is_free_text_and_never_reaches_the_kernel(self, runner):
        # Provenance, not a control input: whatever is in it is stored verbatim
        # and the job runs exactly the code it was given.
        weird = "print('boom')  -- why: fix the mask; rm -rf /"
        snap = self._wait(_jobs.submit("x = 1", intent=weird)["job_id"])
        assert snap["intent"] == weird
        assert snap["status"] == "ok"
        assert snap["stdout"] == ""


# ---------------------------------------------------------------------------
# Real bare kernel: the main thread stays free while a job runs
# ---------------------------------------------------------------------------

_SETUP = """
import biopb_mcp.mcp._jobs as _jobs
from types import SimpleNamespace
_ip = get_ipython()
_ip.user_ns['_conn'] = SimpleNamespace(client=None)
_ip.user_ns['_dask_client'] = None
_jobs.install(_ip)
print('JOBS_READY')
"""


class TestJobOutputCap:
    """The bound ``_MAX_RETAINED_JOBS`` is not.

    That caps how many records are kept, not how large one gets, so a single
    cell printing in a loop grew one record without limit and 32 of them
    bounded nothing.
    """

    _wait = staticmethod(_wait_job)

    @pytest.fixture
    def small_cap(self, monkeypatch):
        """A tiny cap, so a test does not have to print 200k characters."""
        monkeypatch.setattr(_jobs, "_MAX_JOB_OUTPUT_CHARS", 100)
        return 100

    #: 100 iterations x ("xxx" + "\n") -- print writes the text and the end
    #: separately, so this is 400 characters through a cap of 100.
    LOUD = "for i in range(100): print('x' * 3)"

    def test_output_under_the_cap_is_untouched(self, runner, small_cap):
        jid = _jobs.submit("print('hi')")["job_id"]
        snap = self._wait(jid)
        assert snap["stdout"] == "hi\n"
        assert snap["stdout_dropped"] == 0
        assert snap["stdout_total"] == 3

    def test_a_runaway_cell_keeps_only_its_tail(self, runner, small_cap):
        snap = self._wait(_jobs.submit(self.LOUD)["job_id"])
        assert snap["stdout_dropped"] > 0
        assert len(snap["stdout"]) < 400
        # The newest output survives: while a cell is running that is the part
        # worth having, which is why the detail view keeps the tail too.
        assert snap["stdout"].endswith("xxx\n")

    def test_the_record_says_it_is_partial(self, runner, small_cap):
        snap = self._wait(_jobs.submit(self.LOUD)["job_id"])
        # Marked on read rather than stored -- a marker written into the buffer
        # would itself be compacted away by the next rewrite.
        assert "earlier chars dropped" in snap["stdout"]

    def test_the_total_stays_monotonic_across_compaction(self, runner, small_cap):
        # What a reader streaming output as it grows has to diff against:
        # len(stdout) goes *down* when the window moves, and diffing against
        # that is what would leave the chat pane silent for the rest of a cell.
        snap = self._wait(_jobs.submit(self.LOUD)["job_id"])
        assert snap["stdout_total"] == 400
        assert snap["stdout_total"] > len(snap["stdout"])

    def test_the_row_reports_what_was_printed_not_what_was_kept(
        self, runner, small_cap
    ):
        jid = _jobs.submit(self.LOUD)["job_id"]
        self._wait(jid)
        row = next(j for j in _jobs.jobs_summary() if j["job_id"] == jid)
        assert row["stdout_len"] == 400


class TestJobConcurrency:
    @pytest.fixture
    def kernel(self):
        host = KernelHost(health_probe_code=None, startup_timeout=60.0)
        host.start()
        res = host.execute(_SETUP, timeout=30.0)
        assert "JOBS_READY" in res["stdout"], res
        yield host
        host.shutdown()

    def _submit(self, kernel, code):
        res = kernel.execute(
            _kernel_rpc._job_snippet("submit(" + repr(code) + ")"), timeout=15.0
        )
        return _job_result(res["stdout"])

    def _poll(self, kernel, job_id):
        res = kernel.execute(
            _kernel_rpc._job_snippet("poll(" + repr(job_id) + ")"), timeout=15.0
        )
        return _job_result(res["stdout"])

    def test_main_thread_free_while_job_runs(self, kernel):
        # A GIL-releasing background job (time.sleep) must not block the kernel
        # main thread — the whole point of B2.
        sub = self._submit(kernel, "import time; time.sleep(2.0); print('job-done')")
        assert sub["status"] == "running"
        job_id = sub["job_id"]

        # Mid-job: a quick execute returns OK (not 'busy'), proving the main
        # thread is free to service screenshot/status/poll.
        quick = kernel.execute("print('responsive')", timeout=5.0)
        assert quick["status"] == "ok"
        assert "responsive" in quick["stdout"]

        assert self._poll(kernel, job_id)["status"] == "running"

        # Eventually the job finishes; its stdout was captured to the job
        # buffer (not leaked into the quick execute above).
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snap = self._poll(kernel, job_id)
            if snap["status"] != "running":
                break
            time.sleep(0.1)
        assert snap["status"] == "ok"
        assert "job-done" in snap["stdout"]
        assert "job-done" not in quick["stdout"]

    def test_a_real_scratch_kernel_verifies_a_workflow_and_is_discarded(self):
        """End to end with two real kernels — and no display, which is the point.

        The scratch kernel is headless by policy (no Qt, no GL, no napari), so
        the one test that exercises the whole path now runs in CI instead of
        being gated on a desktop. That is the practical dividend of dropping the
        viewer, over and above the ~330 MiB.
        """
        from biopb_mcp.mcp import _scratch
        from biopb_mcp.mcp._kernel import ENV_SCRATCH

        env = dict(os.environ)
        env[ENV_SCRATCH] = "1"
        _scratch.set_host_factory(
            lambda: KernelHost(
                extra_arguments=[
                    "--IPKernelApp.exec_lines="
                    "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
                ],
                startup_timeout=120.0,
                env=env,
                watchdog_interval=0,
                window_close_pipe=False,
                health_probe_code="print('_jobs' in dir())",
            )
        )
        session = KernelHost(health_probe_code=None, startup_timeout=60.0)
        session.start()
        assert "JOBS_READY" in session.execute(_SETUP, timeout=30.0)["stdout"]
        try:
            started = _scratch.start(
                [
                    {"kind": "markdown", "text": "# arithmetic\n\nTriple it."},
                    {"kind": "code", "text": "a = 2"},
                    {"kind": "code", "text": "print(a * 3)"},
                ],
                "arithmetic",
                session,
            )
            job_id = started["job_id"]
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                snap = _scratch.poll(job_id)
                if snap["status"] != "running":
                    break
                time.sleep(0.5)
            assert snap["status"] == "ok", snap
            assert _scratch.verified()["title"] == "arithmetic"
            assert _scratch.verified()["cells"][1]["stdout"] == "6\n"
            assert _scratch.running() is None  # the process is gone; slot free
        finally:
            _scratch.reset()
            _scratch.set_host_factory(None)
            session.shutdown()

    def test_a_scratch_kernel_binds_nothing_a_document_must_build(self):
        """The policy, enforced where it is decided.

        The reader of a saved workflow gets a bare kernel, so the verification
        runs on one too: no `viewer` (nobody is watching a verification), and no
        `np`, `client` or `ops` either — the document builds those itself with
        `biopb_mcp.workflow_env`, and this is what makes running it here proof
        that it runs there. Anything handed to the run for free is something the
        document can lean on and fail on later.
        """
        from biopb_mcp.mcp import _scratch
        from biopb_mcp.mcp._kernel import ENV_SCRATCH

        env = dict(os.environ)
        env[ENV_SCRATCH] = "1"
        _scratch.set_host_factory(
            lambda: KernelHost(
                extra_arguments=[
                    "--IPKernelApp.exec_lines="
                    "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
                ],
                startup_timeout=120.0,
                env=env,
                watchdog_interval=0,
                window_close_pipe=False,
                health_probe_code="print('_jobs' in dir())",
            )
        )
        session = KernelHost(health_probe_code=None, startup_timeout=60.0)
        session.start()
        assert "JOBS_READY" in session.execute(_SETUP, timeout=30.0)["stdout"]
        try:
            started = _scratch.start(
                [
                    {"kind": "code", "text": "import numpy as np"},
                    {"kind": "code", "text": "print('np ok', np.arange(3).sum())"},
                    {"kind": "code", "text": "viewer.add_image(np.zeros((4, 4)))"},
                ],
                "leans on the viewer",
                session,
            )
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                snap = _scratch.poll(started["job_id"])
                if snap["status"] != "running":
                    break
                time.sleep(0.5)
            cells = snap["verify"]["cells"]
            # What the document imports, it has...
            assert cells[0]["status"] == "ok", cells[0]
            assert cells[1]["status"] == "ok", cells[1]
            assert "np ok 3" in cells[1]["stdout"]
            # ...and what it only assumed is simply absent.
            assert cells[2]["status"] == "error"
            assert "NameError" in cells[2]["error_text"]
            assert "viewer" in cells[2]["error_text"]
            assert _scratch.verified() is None
        finally:
            _scratch.reset()
            _scratch.set_host_factory(None)
            session.shutdown()

    def test_poll_job_waits_out_a_real_job_and_answers_when_it_ends(self, kernel):
        """The wait against a real kernel and a real job: it ends when the job
        ends, not when the budget does, and the caller never asks twice."""
        old_host, old_promote = _app._kernel_host, _app._promote_after
        _app.set_kernel_host(kernel)
        _app.set_promote_after(0.5)  # hand back a handle rather than inline it
        try:
            handle = _tool(
                _server.execute_code, "import time; time.sleep(3.0); print('finished')"
            )
            assert "still running" in handle, handle
            job_id = re.search(r"job-\d+", handle).group(0)

            started = time.monotonic()
            result = _tool(_server.poll_job, job_id, wait=30)
            elapsed = time.monotonic() - started
        finally:
            _app.set_kernel_host(old_host)
            _app.set_promote_after(old_promote)
            _writers.clear_claim()

        assert f"{job_id}: ok" in result, result
        assert "finished" in result
        # ~2.5s of the job was left; the 30s budget is nowhere near it.
        assert elapsed < 15.0, f"waited past the job's end ({elapsed:.1f}s)"


# ---------------------------------------------------------------------------
# Full napari bootstrap — only in a real desktop session.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("QT_QPA_PLATFORM") == "offscreen"
    or not os.getenv("DISPLAY")
    or (sys.platform == "darwin" and os.getenv("CI") == "true"),
    reason="napari bootstrap needs a real display",
)
class TestNapariJobs:
    @pytest.fixture
    def napari_kernel(self):
        line = "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
        host = KernelHost(
            extra_arguments=[f"--IPKernelApp.exec_lines={line}"],
            startup_timeout=120.0,
        )
        host.start()
        _app.set_kernel_host(host)
        old_promote = _app._promote_after
        yield host
        _app._promote_after = old_promote
        host.shutdown()

    def _poll_until_done(self, host, job_id, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            res = host.execute(
                _kernel_rpc._job_snippet("poll(" + repr(job_id) + ")"),
                timeout=15.0,
            )
            snap = _job_result(res["stdout"])
            if snap and snap["status"] != "running":
                return snap
            time.sleep(0.2)
        raise AssertionError("job did not finish")

    def test_viewer_mutation_from_worker_thread(self, napari_kernel):
        # add_image from the background job thread must be marshaled to the Qt
        # main thread (no crash) and the layer must appear.
        before = napari_kernel.execute("print(len(viewer.layers))")["stdout"]
        sub = napari_kernel.execute(
            _kernel_rpc._job_snippet(
                "submit("
                + repr("viewer.add_image(np.zeros((8, 8)), name='t'); 'ok'")
                + ")"
            )
        )
        job_id = _job_result(sub["stdout"])["job_id"]
        snap = self._poll_until_done(napari_kernel, job_id)
        assert snap["status"] == "ok", snap
        after = napari_kernel.execute("print(len(viewer.layers))")["stdout"]
        assert int(after.strip()) == int(before.strip()) + 1

    def test_screenshot_and_status_during_job(self, napari_kernel):
        _app.set_promote_after(0.5)
        handle = _tool(
            _server.execute_code, "import time; time.sleep(4.0); print('done')"
        )
        assert "still running" in handle  # promoted to a job

        # The agent is NOT blind: screenshot + status work mid-job.
        shot = _tool(_server.take_screenshot)
        assert shot[0].type == "image"
        status = _tool(_server.server_status)
        assert "## Jobs" in status
        assert "running" in status

    def test_restart_clears_jobs(self, napari_kernel):
        sub = napari_kernel.execute(
            _kernel_rpc._job_snippet(
                "submit(" + repr("import time; time.sleep(30)") + ")"
            )
        )
        job_id = _job_result(sub["stdout"])["job_id"]
        napari_kernel.restart()  # respawns + re-bootstraps (resets jobs)
        res = napari_kernel.execute(
            _kernel_rpc._job_snippet("poll(" + repr(job_id) + ")"), timeout=15.0
        )
        snap = _job_result(res["stdout"])
        assert snap["status"] == "unknown"


class TestVerification:
    """A candidate workflow run in a kernel of its own (``submit(verify_cells=)``).

    The mechanism behind ``verify_workflow``: an agent rewrites a session into a
    clean program and proves it runs without leaning on the session's leftovers
    — which a *filter* over the transcript cannot do, since the correct program
    is a rewrite of it and not a subsequence.

    These are the *kernel* half. What makes the namespace clean is that the
    session child spawns a kernel per verification and discards it
    (``_scratch``, and ``test_mcp_scratch.py``); in here the cells simply run in
    the kernel's own namespace.
    """

    def test_cells_run_in_the_kernels_own_namespace(self, runner):
        # In a scratch kernel that namespace holds the bootstrap and nothing
        # else, so writing to it is free -- the process is discarded. The
        # isolation this used to fake with a filtered dict is the process now.
        _wait_job(_jobs.submit("", verify_cells=["scratch_only = 1"])["job_id"])
        assert runner["scratch_only"] == 1

    def test_the_bootstrap_handles_are_still_there(self, runner):
        # A workflow that cannot reach `client` would verify nothing.
        snap = _wait_job(
            _jobs.submit("", verify_cells=["print(_conn is not None)"])["job_id"]
        )
        assert snap["status"] == "ok"
        assert snap["verify"]["cells"][0]["stdout_head"] == "True"

    def test_cells_run_in_order_and_share_one_namespace(self, runner):
        snap = _wait_job(
            _jobs.submit("", verify_cells=["a = 2", "print(a * 3)\na * 3"])["job_id"]
        )
        cells = _jobs.verify_record(snap["job_id"])["cells"]
        assert snap["verify"]["status"] == "ok"
        assert cells[1]["stdout"] == "6\n"
        assert cells[1]["result_text"] == "6"

    def test_cells_after_a_failure_are_skipped_not_dropped(self, runner):
        # Dropping them would report a workflow that mysteriously got shorter;
        # running them would report the cascade as separate defects.
        snap = _wait_job(
            _jobs.submit("", verify_cells=["1 / 0", "print('a')", "print('b')"])[
                "job_id"
            ]
        )
        assert [c["status"] for c in snap["verify"]["cells"]] == [
            "error",
            "skipped",
            "skipped",
        ]

    def test_output_is_split_per_cell_and_teed_to_the_job(self, runner):
        # The notebook needs the split; poll_job on a long verification needs
        # the whole run accumulating where it always does.
        snap = _wait_job(
            _jobs.submit("", verify_cells=["print('one')", "print('two')"])["job_id"]
        )
        record = _jobs.verify_record(snap["job_id"])
        assert [c["stdout"] for c in record["cells"]] == ["one\n", "two\n"]
        assert snap["stdout"] == "one\ntwo\n"

    def test_the_polled_record_carries_a_head_not_the_output(self, runner):
        # The polled snapshot crosses a JSON round trip every 0.4s while a
        # verification runs; carrying every cell's output there would ship the
        # bytes `stdout` already holds, once more per cell, growing with the
        # workflow. The full text is read once, by verify_record(), for the
        # notebook -- before the kernel holding it is discarded.
        big = "print('x' * 40_000)"
        snap = _wait_job(_jobs.submit("", verify_cells=[big] * 5)["job_id"])
        polled = snap["verify"]["cells"]
        assert all("stdout" not in c for c in polled)
        assert all(c["stdout_len"] == 40_001 for c in polled)
        assert all(c["stdout_head"] == "x" * 79 + "…" for c in polled)
        # ...and the notebook still gets all of it.
        full = _jobs.verify_record(snap["job_id"])["cells"]
        assert all(len(c["stdout"]) == 40_001 for c in full)

    def test_the_polled_record_does_not_grow_with_what_the_cells_printed(self, runner):
        # The property, stated as a shape rather than a number: the polled
        # record scales with the *workflow* -- a line per cell, which is the
        # point of a ledger -- and not with its output.
        import json

        def polled_size(chars):
            jid = _jobs.submit("", verify_cells=[f"print('y' * {chars})"] * 5)["job_id"]
            return len(json.dumps(_wait_job(jid)["verify"]))

        # ~495,000 more characters printed across the five cells; the record
        # moves by the digits of a length and the source that names them.
        assert polled_size(100_000) - polled_size(1_000) < 100

    def test_the_job_code_is_derived_from_the_cells(self, runner):
        # The audit view of this job must not disagree with the workflow view.
        snap = _wait_job(_jobs.submit("", verify_cells=["a = 1", "a"])["job_id"])
        assert "a = 1" in snap["code"] and snap["code"].endswith("a")

    def test_an_ordinary_job_carries_no_verification(self, runner):
        assert _jobs.poll(_jobs.submit("1 + 1")["job_id"])["verify"] is None
        assert _jobs.verify_record(_jobs.submit("1 + 1")["job_id"]) is None

    def test_jobs_view_carries_the_job_list(self, runner):
        # The workflow is no longer this kernel's to report: the run happened in
        # a scratch kernel, so the session child merges it in (_observe).
        jid = _jobs.submit("", verify_cells=["1"], verify_title="good")["job_id"]
        _wait_job(jid)
        assert [j["job_id"] for j in _jobs.jobs_view()["jobs"]] == [jid]
        assert "workflow" not in _jobs.jobs_view()
