"""The kernel's side of the jobs (``_jobs``): tasks, held cells, stopping.

Three layers:

* ``TestTasks`` / ``TestHoldCell`` / ``TestRunAsync`` -- the module driven
  directly with a fake InteractiveShell (no kernel, fast). The ``log`` fixture
  feeds what it announces into a host ``JobLog``.
* ``TestJobConcurrency`` -- a real *bare* kernel (no napari/display): a task
  leaves the main thread free, and a real scratch kernel verifies a workflow.
* ``TestNapariJobs`` -- display-gated end-to-end: viewer mutation from a task
  (main-thread marshaling), screenshot/status beside a task, restart.
"""

import itertools
import os
import re
import sys
import threading
import time
import types

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from biopb_mcp._tests.conftest import call_tool as _tool, iopub_event
from biopb_mcp.mcp import (  # noqa: E402
    _app,
    _job_log,
    _jobs,
    _server,
    _writers,
)
from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


@pytest.fixture
def runner(monkeypatch):
    """The module wired to a fake InteractiveShell (no kernel).

    Each task gets a request id of its own, as the cell that starts it would
    from a kernel.
    """
    requests = itertools.count(1)
    monkeypatch.setattr(_jobs, "_request_id", lambda: f"req-{next(requests)}")
    ns = {
        "_conn": types.SimpleNamespace(client=None),
    }
    _jobs.install(types.SimpleNamespace(user_ns=ns))
    yield ns
    _jobs.reset()


@pytest.fixture
def log(runner, monkeypatch):
    """A host ``JobLog`` fed the module's announcements, as iopub would.

    Only the events: with no kernel there is no iopub for a task's prints to
    reach, so output is the business of ``test_mcp_job_log`` and the
    real-kernel tests below.
    """
    log = _job_log.JobLog()

    monkeypatch.setattr(_jobs, "_publish", lambda c: log.on_iopub(iopub_event(c)))
    return log


def _spin():
    """A task that never ends on its own: a pure-Python loop, stoppable only
    by a KeyboardInterrupt raised into its thread."""
    while True:
        time.sleep(0.02)


def _task(fn, *args, **kwargs):
    """Start *fn* as a task; its ``_Job``."""
    return _jobs._jobs[_jobs.run_async(fn, *args, **kwargs)]


def _wait(job, timeout=5.0):
    """Block until *job* ends; return it."""
    deadline = time.monotonic() + timeout
    while job.status == "running":
        assert time.monotonic() < deadline, f"{job.job_id} did not finish"
        time.sleep(0.01)
    return job


# ---------------------------------------------------------------------------
# Unit: the module with a fake shell (no real kernel)
# ---------------------------------------------------------------------------


class TestTasks:
    """How a task ends, and how it is stopped."""

    def test_start_and_end_are_announced(self, runner, log):
        job = _wait(_task(lambda: 1 + 2))
        snap = log.poll(job.job_id)
        assert snap["status"] == "ok"
        assert snap["result_text"] == "3"
        assert snap["code"].startswith("run_async(")

    def test_error_is_captured(self, runner):
        job = _wait(_task(lambda: 1 / 0))
        assert job.status == "error"
        assert "ZeroDivisionError" in job.error_text

    def test_reset_clears_registry(self, runner):
        _task(lambda: None)
        assert _jobs._jobs
        _jobs.reset()
        assert _jobs._jobs == {}

    def test_an_ended_job_is_dropped_by_the_next(self, runner):
        # The records are the host's; the kernel keeps only what it can stop.
        first = _wait(_task(lambda: None))
        _task(lambda: None)
        assert first.job_id not in _jobs._jobs

    def test_interrupt_stops_an_uncooperative_task(self, runner, log):
        job = _task(_spin)
        out = _jobs.interrupt(job.job_id, reason="forced by Bob")
        assert out["interrupted"] is True
        _wait(job)
        snap = log.poll(job.job_id)
        assert snap["status"] == "interrupted"
        assert snap["cancel_reason"] == "forced by Bob"
        assert "forced by Bob" in snap["error_text"]
        assert "KeyboardInterrupt" in snap["error_text"]

    def test_interrupt_when_idle_stops_nothing(self):
        _jobs.reset()
        assert _jobs.interrupt("req-1") == {
            "interrupted": False,
            "refused": "not_running",
        }

    def test_a_stop_aimed_at_an_ended_job_does_not_land_on_the_next(self, runner):
        # The caller's view comes from iopub and may be stale; the kernel's is
        # not. The job the stop was aimed at is gone, so nothing is stopped.
        done = _wait(_task(lambda: None))
        job = _task(_spin)
        try:
            assert _jobs.interrupt(done.job_id)["refused"] == "not_running"
            time.sleep(0.1)
            assert job.status == "running"
        finally:
            _jobs.interrupt(job.job_id)
            _wait(job)

    def test_raise_in_thread_no_ident(self):
        assert _jobs._raise_in_thread(None, KeyboardInterrupt) == 0

    def test_external_interrupt_is_labeled_not_reported_as_an_error(self, runner):
        # A KeyboardInterrupt this module did not raise -- an external SIGINT
        # relayed through a run_on_main slot. Unlabeled it renders as the code
        # failing rather than as a stop someone else caused.
        job = _task(_spin)
        assert _jobs._raise_in_thread(job.thread.ident, KeyboardInterrupt) == 1
        _wait(job)
        assert job.status == "interrupted"  # not "error"
        assert job.cancel_reason == _jobs._EXTERNAL_INTERRUPT_MSG
        assert job.error_text.startswith(_jobs._EXTERNAL_INTERRUPT_MSG)
        assert "KeyboardInterrupt" in job.error_text

    def test_an_owned_interrupt_keeps_its_own_reason(self, runner):
        job = _task(_spin)
        _jobs.interrupt(job.job_id, reason="forced by Bob")
        _wait(job)
        assert job.status == "interrupted"
        assert job.cancel_reason == "forced by Bob"
        assert _jobs._EXTERNAL_INTERRUPT_MSG not in job.error_text


class TestHoldCell:
    """A cell, held as a running job while it runs on the main thread
    (``_kernel_gate``). The host records it from the protocol, so nothing is
    announced (``test_mcp_job_log``, and the gate's real-kernel tests)."""

    def test_nothing_is_announced(self, runner, log):
        with _jobs.hold_cell("req-1"):
            pass
        assert log.export() == []

    def test_output_is_left_to_the_real_stream(self, runner, capsys):
        with _jobs.hold_cell("req-1"):
            print("hi")
        assert capsys.readouterr().out == "hi\n"

    def test_running_for_its_duration(self, runner):
        with _jobs.hold_cell("req-1"):
            assert _jobs._running("req-1") is not None
        assert _jobs._running("req-1") is None

    def test_ended_even_when_the_block_raises(self, runner):
        with pytest.raises(RuntimeError):
            with _jobs.hold_cell("req-1"):
                raise RuntimeError("do_execute itself failed")
        assert _jobs._running("req-1") is None

    def test_a_stop_names_it_by_its_request(self, runner, monkeypatch):
        # The host's id for a cell is its own; the kernel knows the request.
        signals = []
        monkeypatch.setattr(_jobs, "_interrupt_main", lambda: signals.append(1))
        with _jobs.hold_cell("req-7"):
            assert _jobs.interrupt("req-8")["refused"] == "not_running"
            assert signals == []
            assert _jobs.interrupt("req-7") == {"interrupted": True}
            assert signals == [1]


class TestRunAsync:
    """A long compute off the main thread, polled by its task id."""

    def test_returns_its_task_id_at_once(self, runner, log):
        gate = threading.Event()
        task = _jobs.run_async(gate.wait)
        try:
            assert task.startswith("task-")
            assert log.poll(task)["status"] == "running"
        finally:
            gate.set()
        _wait(_jobs._jobs[task])
        assert log.poll(task)["status"] == "ok"

    def test_what_it_returns_is_its_result(self, runner):
        assert _wait(_task(lambda a, b: a + b, 40, b=2)).result_text == "42"

    def test_one_task_at_a_time(self, runner):
        gate = threading.Event()
        job = _task(gate.wait)
        try:
            with pytest.raises(RuntimeError, match=job.job_id):
                _jobs.run_async(print)
        finally:
            gate.set()
            _wait(job)

    def test_takes_a_function(self, runner):
        with pytest.raises(TypeError):
            _jobs.run_async("x = 1")


# ---------------------------------------------------------------------------
# Real bare kernel: the main thread stays free while a job runs
# ---------------------------------------------------------------------------

_SETUP = """
import biopb_mcp.mcp._jobs as _jobs
from types import SimpleNamespace
_ip = get_ipython()
_ip.user_ns['_conn'] = SimpleNamespace(client=None)
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

    def test_main_thread_free_while_a_task_runs(self, kernel):
        # A GIL-releasing task (time.sleep) must not block the kernel's main
        # thread: that is what run_async is for.
        cell = kernel.jobs.new_id()
        kernel.run_cell(
            "import time\n"
            "def work():\n"
            "    time.sleep(2.0)\n"
            "    print('task-done')\n"
            "_jobs.run_async(work)",
            cell,
            "mcp",
        )
        deadline = time.monotonic() + 10.0
        while kernel.jobs.poll(cell)["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        task = kernel.jobs.running()
        assert task is not None and task["job_id"].startswith("task-"), task

        # Mid-task: a quick execute returns, proving the main thread is free
        # to service screenshot/status.
        quick = kernel.execute("print('responsive')", timeout=5.0)
        assert quick["status"] == "ok"
        assert "responsive" in quick["stdout"]

        # The task's output reaches its record, not the quick execute's.
        while kernel.jobs.poll(task["job_id"])["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.1)
        snap = kernel.jobs.poll(task["job_id"])
        assert snap["status"] == "ok"
        assert "task-done" in snap["stdout"]
        assert "task-done" not in quick["stdout"]

    @pytest.fixture
    def scratch(self):
        """A session kernel and a factory for real scratch kernels: headless by
        policy (no Qt, no GL, no napari), so the whole verification path runs
        in CI."""
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
            yield _scratch, session
        finally:
            _scratch.reset()
            _scratch.set_host_factory(None)
            session.shutdown()

    @staticmethod
    def _verify(scratch, session, cells, title="wf"):
        blocks = [{"kind": "code", "text": c} for c in cells]
        return scratch.start(blocks, title, session)["job_id"]

    @staticmethod
    def _until(scratch, job_id, done, timeout=180.0):
        deadline = time.monotonic() + timeout
        while True:
            snap = scratch.poll(job_id)
            if done(snap):
                return snap
            assert time.monotonic() < deadline, snap
            time.sleep(0.2)

    def test_a_real_scratch_kernel_verifies_a_workflow_and_is_discarded(self, scratch):
        # One request per cell, sharing the kernel's namespace; the output of
        # each is its own, and the last expression is its result.
        _scratch, session = scratch
        started = _scratch.start(
            [
                {"kind": "markdown", "text": "# arithmetic\n\nTriple it."},
                {"kind": "code", "text": "a = 2"},
                {"kind": "code", "text": "print(a * 3)\na * 3"},
            ],
            "arithmetic",
            session,
        )
        snap = self._until(
            _scratch, started["job_id"], lambda s: s["status"] != "running"
        )
        assert snap["status"] == "ok", snap
        verified = _scratch.verified()
        assert verified["title"] == "arithmetic"
        assert verified["cells"][1]["stdout"] == "6\n"
        assert verified["cells"][1]["result_text"] == "6"
        assert _scratch.running() is None  # the process is gone; slot free

    def test_cells_after_a_failure_are_skipped(self, scratch):
        _scratch, session = scratch
        job_id = self._verify(_scratch, session, ["1 / 0", "print('a')", "print('b')"])
        snap = self._until(_scratch, job_id, lambda s: s["status"] != "running")
        assert snap["status"] == "error"
        cells = snap["verify"]["cells"]
        assert [c["status"] for c in cells] == ["error", "skipped", "skipped"]
        assert "ZeroDivisionError" in cells[0]["error_text"]

    def test_a_stop_interrupts_the_running_cell_and_keeps_the_kernel(self, scratch):
        # The SIGINT on the control channel, not the kill: the run ends
        # "interrupted" with the cell's own record, inside the grace.
        _scratch, session = scratch
        job_id = self._verify(
            _scratch, session, ["import time\ntime.sleep(60)", "print('never')"]
        )
        self._until(
            _scratch,
            job_id,
            lambda s: (
                (s["verify"] or {}).get("cells", [{}])[0].get("status") == "running"
            ),
        )
        time.sleep(0.5)  # into the sleep, past the cell's start
        out = _scratch.interrupt("stopped by the test")
        assert out["interrupted"] is True and not out.get("killed"), out
        snap = _scratch.poll(job_id)
        assert snap["status"] == "interrupted"
        cells = snap["verify"]["cells"]
        assert [c["status"] for c in cells] == ["error", "skipped"]
        assert "KeyboardInterrupt" in cells[0]["error_text"]
        assert "stopped by the test" in cells[0]["error_text"]

    def test_a_scratch_kernel_binds_nothing_a_document_must_build(self, scratch):
        """The policy, enforced where it is decided.

        The reader of a saved workflow gets a bare kernel, so the verification
        runs on one too: no `viewer` (nobody is watching a verification), and no
        `np`, `client` or `ops` either — the document builds those itself with
        `biopb_mcp.workflow_env`, and this is what makes running it here proof
        that it runs there. Anything handed to the run for free is something the
        document can lean on and fail on later.
        """
        _scratch, session = scratch
        job_id = self._verify(
            _scratch,
            session,
            [
                "import numpy as np",
                "print('np ok', np.arange(3).sum())",
                "viewer.add_image(np.zeros((4, 4)))",
            ],
            "leans on the viewer",
        )
        snap = self._until(_scratch, job_id, lambda s: s["status"] != "running")
        cells = _scratch._run["record"]["cells"]
        # What the document imports, it has...
        assert cells[0]["status"] == "ok", cells[0]
        assert cells[1]["status"] == "ok", cells[1]
        assert "np ok 3" in cells[1]["stdout"]
        # ...and what it only assumed is simply absent: not even `client`,
        # which the session's cells are handed.
        assert cells[2]["status"] == "error"
        assert "NameError" in cells[2]["error_text"]
        assert "viewer" in cells[2]["error_text"]
        assert snap["status"] == "error"
        assert _scratch.verified() is None

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

    def _until_done(self, host, job_id, timeout=20.0):
        deadline = time.monotonic() + timeout
        while host.jobs.poll(job_id)["status"] == "running":
            assert time.monotonic() < deadline, f"{job_id} did not finish"
            time.sleep(0.2)
        return host.jobs.poll(job_id)

    def test_viewer_mutation_from_a_task(self, napari_kernel):
        # add_image from the task's thread must be marshaled to the Qt main
        # thread (no crash) and the layer must appear.
        before = napari_kernel.execute("print(len(viewer.layers))")["stdout"]
        cell = napari_kernel.jobs.new_id()
        napari_kernel.run_cell(
            "task = run_async(viewer.add_image, np.zeros((8, 8)), name='t')",
            cell,
            "mcp",
        )
        assert self._until_done(napari_kernel, cell)["status"] == "ok"
        task = napari_kernel.execute("print(task)")["stdout"].strip()
        snap = self._until_done(napari_kernel, task)
        assert snap["status"] == "ok", snap
        after = napari_kernel.execute("print(len(viewer.layers))")["stdout"]
        assert int(after.strip()) == int(before.strip()) + 1

    def test_screenshot_and_status_beside_a_task(self, napari_kernel):
        handle = _tool(
            _server.execute_code,
            "import time\nrun_async(time.sleep, 4.0)",
        )
        assert "task-" in handle, handle

        # The agent is NOT blind: screenshot + status work while it runs.
        shot = _tool(_server.take_screenshot)
        assert shot[0].type == "image"
        status = _tool(_server.server_status)
        assert "## Jobs" in status
        assert "running" in status

    def test_restart_keeps_the_record_and_ends_it(self, napari_kernel):
        job_id = napari_kernel.jobs.new_id()
        napari_kernel.run_cell("import time; time.sleep(30)", job_id, "mcp")
        deadline = time.monotonic() + 10.0
        while napari_kernel.jobs.stop_key(job_id) is None:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        napari_kernel.restart()  # respawns + re-bootstraps
        # The host's record outlives the kernel.
        assert napari_kernel.jobs.poll(job_id)["status"] == "interrupted"
