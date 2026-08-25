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
import sys
import time
import types

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from biopb_mcp.mcp import _jobs, _server  # noqa: E402
from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


def _job_result(stdout):
    """Unwrap the ``{"r": result, "w": window_alive}`` job-snippet envelope."""
    payload = _server._extract_json(stdout)
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
        _jobs._cancel(jid)
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

    def test_code_preview_helper(self):
        assert _jobs._code_preview("") == ""
        assert _jobs._code_preview("\n\n  hello  \nworld") == "hello"
        capped = _jobs._code_preview("x" * 100)
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
        assert snap["origin"] == "agent"
        summ = {j["job_id"]: j for j in _jobs.jobs_summary()}[jid]
        assert summ["origin"] == "agent"
        # export() feeds the notebook writer -- provenance must survive there too.
        assert {e["job_id"]: e for e in _jobs.export()}[jid]["origin"] == "agent"

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
            res = _jobs.interrupt_current(requester="agent")
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
        assert _jobs.interrupt_current(requester="agent")["interrupted"] is True
        assert self._wait(jid)["status"] == "interrupted"

    def test_digest_reports_only_unseen_user_jobs(self, runner):
        self._wait(_jobs.submit("a = 1", origin="agent")["job_id"])
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
            # job -- _server._ack_foreign_digest filters on the reported status,
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
            assert _jobs.interrupt_current(requester="agent") == {
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
            assert _jobs.interrupt_current(requester="agent", writer="sess-B") == {
                "job_id": jid,
                "interrupted": False,
                "status": "running",
                "refused": "not_owner",
            }
            # The owner still can, and so can the human (the default requester).
            assert (
                _jobs.interrupt_current(requester="agent", writer="sess-A")[
                    "interrupted"
                ]
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
            refused = _jobs.interrupt_current(requester="agent")
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
            _server._job_snippet("submit(" + repr(code) + ")"), timeout=15.0
        )
        return _job_result(res["stdout"])

    def _poll(self, kernel, job_id):
        res = kernel.execute(
            _server._job_snippet("poll(" + repr(job_id) + ")"), timeout=15.0
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
        _server.set_kernel_host(host)
        old_promote = _server._promote_after
        yield host
        _server._promote_after = old_promote
        host.shutdown()

    def _poll_until_done(self, host, job_id, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            res = host.execute(
                _server._job_snippet("poll(" + repr(job_id) + ")"),
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
            _server._job_snippet(
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
        _server.set_promote_after(0.5)
        handle = _server.execute_code("import time; time.sleep(4.0); print('done')")
        assert "still running" in handle  # promoted to a job

        # The agent is NOT blind: screenshot + status work mid-job.
        shot = _server.take_screenshot()
        assert shot[0].type == "image"
        status = _server.server_status()
        assert "## Jobs" in status
        assert "running" in status

    def test_restart_clears_jobs(self, napari_kernel):
        sub = napari_kernel.execute(
            _server._job_snippet("submit(" + repr("import time; time.sleep(30)") + ")")
        )
        job_id = _job_result(sub["stdout"])["job_id"]
        napari_kernel.restart()  # respawns + re-bootstraps (resets jobs)
        res = napari_kernel.execute(
            _server._job_snippet("poll(" + repr(job_id) + ")"), timeout=15.0
        )
        snap = _job_result(res["stdout"])
        assert snap["status"] == "unknown"
