"""Tests for the async submit->poll execution model (review finding B2).

Three layers:

* ``TestJobRunnerUnit`` — the in-kernel job runner driven directly with a fake
  InteractiveShell (no kernel, fast): submit/poll/cancel, output capture,
  cooperative + distributed cancel.
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


# ---------------------------------------------------------------------------
# Unit: in-kernel job runner with a fake shell (no real kernel)
# ---------------------------------------------------------------------------


class TestJobRunnerUnit:
    @pytest.fixture
    def runner(self):
        ns = {
            "_dask_client": None,
            "_conn": types.SimpleNamespace(client=None),
        }
        fake_ip = types.SimpleNamespace(user_ns=ns)
        _jobs.install(fake_ip)
        ns["cancelled"] = _jobs.cancelled
        yield ns
        _jobs.reset()

    def _wait(self, job_id, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = _jobs.poll(job_id)
            if snap["status"] != "running":
                return snap
            time.sleep(0.02)
        raise AssertionError(f"job {job_id} did not finish")

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
        jid = _jobs.submit("import time\nwhile not cancelled():\n    time.sleep(0.02)")[
            "job_id"
        ]
        try:
            busy = _jobs.submit("1 + 1")
            assert busy.get("error") == "busy"
            assert busy["running_job_id"] == jid
        finally:
            _jobs.cancel(jid)
        self._wait(jid)

    def test_cooperative_cancel(self, runner):
        jid = _jobs.submit(
            "import time\n"
            "while not cancelled():\n    time.sleep(0.02)\n"
            "print('stopped')"
        )["job_id"]
        time.sleep(0.1)
        assert _jobs.poll(jid)["status"] == "running"
        res = _jobs.cancel(jid)
        assert res["cancelled"] is True
        snap = self._wait(jid)
        assert snap["status"] == "cancelled"
        assert "stopped" in snap["stdout"]

    def test_distributed_cancel_rebuilds_futures(self, runner):
        # cancel() must rebuild real Future objects from dc.futures' string
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
        jid = _jobs.submit("import time\nwhile not cancelled():\n    time.sleep(0.02)")[
            "job_id"
        ]
        time.sleep(0.05)
        _jobs.cancel(jid)
        passed = calls["futures"]
        assert passed and all(isinstance(f, Future) for f in passed)
        assert {f.key for f in futures_of(passed)} == set(_StubClient.futures)
        assert calls["force"] is True
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

    def test_cancel_attributes_reason(self, runner):
        jid = _jobs.submit("import time\nwhile not cancelled():\n    time.sleep(0.02)")[
            "job_id"
        ]
        while _jobs.poll(jid)["status"] != "running":
            time.sleep(0.02)
        out = _jobs.cancel(jid, reason="stopped by Alice")
        assert out["cancelled"] is True
        assert out["job_id"] == jid
        snap = self._wait(jid)
        assert snap["status"] == "cancelled"
        assert snap["cancel_reason"] == "stopped by Alice"
        # The reason is surfaced in error_text so poll_job/execute_code render it.
        assert "stopped by Alice" in snap["error_text"]

    def test_interrupt_current_stops_uncooperative_job(self, runner):
        # A loop that NEVER checks cancelled() -> only a KeyboardInterrupt raised
        # into the worker thread can stop it (short of restart).
        jid = _jobs.submit("import time\nwhile True:\n    time.sleep(0.02)")["job_id"]
        while _jobs.poll(jid)["status"] != "running":
            time.sleep(0.02)
        out = _jobs.interrupt_current(reason="forced by Bob")
        assert out["job_id"] == jid and out["interrupted"] is True
        snap = self._wait(jid)
        # Distinct terminal status from a plain cooperative cancel.
        assert snap["status"] == "interrupted"
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
# Real bare kernel: the main thread stays free while a job runs
# ---------------------------------------------------------------------------

_SETUP = """
import biopb_mcp.mcp._jobs as _jobs
from types import SimpleNamespace
_ip = get_ipython()
_ip.user_ns['_conn'] = SimpleNamespace(client=None)
_ip.user_ns['_dask_client'] = None
_jobs.install(_ip)
_ip.user_ns['cancelled'] = _jobs.cancelled
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
