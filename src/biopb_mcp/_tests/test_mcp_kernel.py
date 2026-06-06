"""Tests for KernelHost (the child Jupyter kernel manager).

The unit tests start a *plain* python kernel (no napari bootstrap, no display)
and exercise execute/interrupt/restart/shutdown.  A separate, display-gated
test runs the real napari bootstrap end-to-end.
"""

import os
import signal
import sys
import time

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


class TestConfigureDask:
    """Unit tests for _configure_dask (no kernel / no display needed)."""

    def test_in_process_scheduler_returns_no_client(self):
        """threads/synchronous schedulers yield no client and no cluster."""
        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask({"dask_scheduler": "threads"})
        assert client is None
        assert cluster is None

    def test_external_address_connects_without_cluster(self, monkeypatch):
        """distributed + an explicit address attaches a Client, no cluster."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        created = {}

        class _FakeClient:
            def __init__(self, address):
                created["address"] = address

        monkeypatch.setattr(dd, "Client", _FakeClient)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {
                "dask_scheduler": "distributed",
                "dask_distributed_address": "tcp://1.2.3.4:8786",
            }
        )
        assert isinstance(client, _FakeClient)
        assert created["address"] == "tcp://1.2.3.4:8786"
        assert cluster is None

    def test_local_cluster_failure_falls_back_to_threads(self, monkeypatch):
        """A LocalCluster spawn failure degrades to in-process, not a crash."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        def _boom(*args, **kwargs):
            raise RuntimeError("no cluster for you")

        monkeypatch.setattr(dd, "LocalCluster", _boom)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {"dask_scheduler": "distributed", "dask_distributed_address": ""}
        )
        assert client is None
        assert cluster is None


@pytest.fixture
def kernel():
    """A bare kernel with no bootstrap and no health probe."""
    host = KernelHost(health_probe_code=None, startup_timeout=60.0)
    host.start()
    yield host
    host.shutdown()


class TestKernelExecute:
    def test_stdout_captured(self, kernel):
        res = kernel.execute("print('hi')")
        assert res["status"] == "ok"
        assert "hi" in res["stdout"]

    def test_expression_result(self, kernel):
        res = kernel.execute("1 + 2")
        assert res["status"] == "ok"
        assert "3" in res["result_text"]

    def test_error_status_and_traceback(self, kernel):
        res = kernel.execute("1 / 0")
        assert res["status"] == "error"
        assert "ZeroDivisionError" in res["error_text"]
        # ANSI escape codes are stripped.
        assert "\x1b[" not in res["error_text"]

    def test_variables_persist(self, kernel):
        kernel.execute("my_var = 99")
        res = kernel.execute("print(my_var)")
        assert "99" in res["stdout"]

    def test_timeout_interrupts(self, kernel):
        res = kernel.execute("import time; time.sleep(10)", timeout=0.5)
        assert res["status"] == "timeout"
        # Kernel survives and accepts new work afterwards.
        res2 = kernel.execute("print('alive')", timeout=10.0)
        assert "alive" in res2["stdout"]


class TestKernelControl:
    def test_interrupt_frees_busy_loop(self, kernel):
        import threading

        results = {}

        def run():
            results["res"] = kernel.execute("while True: pass", timeout=30.0)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(1.0)
        kernel.interrupt()
        t.join(timeout=15.0)
        assert not t.is_alive()
        assert results["res"]["status"] in ("error", "ok")

    def test_restart_clears_namespace(self, kernel):
        kernel.execute("survivor = 1")
        assert "1" in kernel.execute("print(survivor)")["stdout"]
        kernel.restart()
        res = kernel.execute("print('survivor' in dir())")
        assert "False" in res["stdout"]

    def test_busy_returns_busy_status(self, kernel):
        import threading

        kernel._busy_lock_timeout = 0.2

        def run():
            kernel.execute("import time; time.sleep(3)", timeout=10.0)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.5)
        res = kernel.execute("print('x')")
        assert res["status"] == "busy"
        t.join(timeout=15.0)


class TestKernelLifecycle:
    def test_is_alive(self, kernel):
        assert kernel.is_alive()

    def test_shutdown_removes_connection_file(self):
        host = KernelHost(health_probe_code=None, startup_timeout=60.0)
        host.start()
        conn_file = host._km.connection_file
        assert os.path.exists(conn_file)
        host.shutdown()
        assert not host.is_alive()
        assert not os.path.exists(conn_file)

    def test_kernel_native_stdout_redirected_to_file(self, tmp_path):
        """stdio mode passes the kernel a log file for its native fds so
        Qt/GL/dask/gRPC output never lands on fd 1 (the JSON-RPC channel).

        A raw ``os.write(1, ...)`` bypasses IPython's ZMQ stdout capture, so
        it must NOT appear in the execute result but MUST land in the file.
        """
        log = tmp_path / "kernel.log"
        f = open(log, "ab", buffering=0)
        try:
            host = KernelHost(
                health_probe_code=None,
                startup_timeout=60.0,
                kernel_stdout=f,
                kernel_stderr=f,
            )
            host.start()
            res = host.execute("import os; os.write(1, b'NATIVE_FD1_MARKER')")
            assert "NATIVE_FD1_MARKER" not in res["stdout"]
            host.shutdown()
        finally:
            f.close()
        assert b"NATIVE_FD1_MARKER" in log.read_bytes()

    def test_shutdown_attempts_graceful_close_before_kill(self, monkeypatch):
        # On shutdown the kernel should be asked to close the tensor client /
        # dask *before* _shutdown_current() group-kills it, so the tensor
        # server sees a clean Flight GOAWAY rather than an abrupt socket drop
        # (which can hang a subsequent `biopb server stop`).
        from biopb_mcp.mcp import _kernel

        host = KernelHost(health_probe_code=None, startup_timeout=60.0)
        host.start()

        calls = []
        real_execute_locked = host._execute_locked
        real_shutdown_current = host._shutdown_current

        def _spy_execute(code, timeout):
            calls.append(("execute", code, timeout))
            return real_execute_locked(code, timeout)

        def _spy_shutdown_current():
            calls.append(("shutdown_current",))
            return real_shutdown_current()

        monkeypatch.setattr(host, "_execute_locked", _spy_execute)
        monkeypatch.setattr(host, "_shutdown_current", _spy_shutdown_current)

        host.shutdown()

        # The graceful-close snippet ran, bounded, before the group-kill.
        assert calls[0][0] == "execute"
        assert calls[0][1] is _kernel._GRACEFUL_CLOSE_SNIPPET
        assert calls[0][2] == 2.0
        assert ("shutdown_current",) in calls
        assert calls.index(("shutdown_current",)) > 0
        assert not host.is_alive()

    def test_health_probe_failure_raises(self):
        # Probe expects a name that does not exist in a bare kernel.
        host = KernelHost(
            health_probe_code="print('viewer' in dir())",
            health_probe_expect="True",
            startup_timeout=60.0,
        )
        with pytest.raises(RuntimeError, match="health probe failed"):
            host.start()
        host.shutdown()


# ---------------------------------------------------------------------------
# Orphan hardening (issue #13)
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout=15.0, interval=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# pgid / killpg / SIGKILL are POSIX-only; the hardening they test degrades to a
# no-op on Windows (guarded by os.name == "posix" / hasattr(os, "killpg")).
posix_only = pytest.mark.skipif(
    os.name != "posix", reason="POSIX-only (pgid / killpg / SIGKILL)"
)


@posix_only
class TestPgidCapture:
    """Fix 3: pgid captured at launch, reaped via stored pgid."""

    def test_pgid_captured_equals_kernel_group(self):
        host = KernelHost(
            health_probe_code=None,
            watchdog_interval=0,
            parent_death_pipe=False,
        )
        host.start()
        try:
            pid = host._kernel_pid()
            assert host._pgid == os.getpgid(pid)
        finally:
            host.shutdown()
        # pgid is cleared after shutdown so a recycled pid can't be re-killed.
        assert host._pgid is None

    def test_shutdown_never_kills_own_group(self, monkeypatch):
        host = KernelHost(
            health_probe_code=None,
            watchdog_interval=0,
            parent_death_pipe=False,
        )
        host.start()
        # Simulate the captured pgid resolving to the launcher's own group:
        # our group-kill must be skipped (never SIGKILL ourselves). The spy
        # passes through so jupyter_client still reaps the real kernel group
        # (and its own internal kill is recorded too, hence not asserting []).
        host._pgid = os.getpgrp()
        real_killpg = os.killpg
        killed = []

        def _spy(pg, sig):
            killed.append(pg)
            return real_killpg(pg, sig)

        monkeypatch.setattr(os, "killpg", _spy)
        host._shutdown_current()
        assert os.getpgrp() not in killed


@posix_only
class TestWatchdog:
    """Fix 2: liveness watchdog reaps + respawns, bounded."""

    def test_respawns_unexpectedly_dead_kernel(self):
        host = KernelHost(
            health_probe_code=None,
            watchdog_interval=0.3,
            parent_death_pipe=False,
        )
        host.start()
        try:
            pid1 = host._kernel_pid()
            os.kill(pid1, signal.SIGKILL)
            assert _wait_until(
                lambda: host.is_alive() and host._kernel_pid() != pid1
            )
            assert host.is_alive()
            assert host._kernel_pid() != pid1
            assert not host._dead
        finally:
            host.shutdown()

    def test_respawn_bound_marks_host_dead(self):
        # max_respawns=0 -> the first unexpected death exhausts the budget
        # immediately, so the host is marked dead instead of respawning.
        host = KernelHost(
            health_probe_code=None,
            watchdog_interval=0.3,
            watchdog_max_respawns=0,
            parent_death_pipe=False,
        )
        host.start()
        try:
            os.kill(host._kernel_pid(), signal.SIGKILL)
            assert _wait_until(lambda: host._dead)
            assert not host.is_alive()
            assert host.health()["dead"] is True
        finally:
            host.shutdown()

    def test_restart_does_not_trip_watchdog(self):
        host = KernelHost(
            health_probe_code=None,
            watchdog_interval=0.3,
            parent_death_pipe=False,
        )
        host.start()
        try:
            host.execute("survivor = 1")
            host.restart()
            # The intentional restart must not be treated as a death: the
            # kernel is alive, not marked dead, and the namespace is cleared.
            assert host.is_alive()
            assert not host._dead
            assert (
                "False" in host.execute("print('survivor' in dir())")["stdout"]
            )
        finally:
            host.shutdown()


class TestHealth:
    def test_health_fields(self):
        host = KernelHost(health_probe_code=None, parent_death_pipe=False)
        host.start()
        try:
            h = host.health()
            assert set(h) == {
                "alive",
                "busy",
                "dead",
                "recent_respawns",
                "watchdog_running",
            }
            assert h["alive"] is True
            assert h["dead"] is False
            assert h["watchdog_running"] is True
        finally:
            host.shutdown()
        assert host.health()["watchdog_running"] is False


class TestParentDeathPipe:
    """Fix 1: kernel self-terminates when the launcher process dies."""

    def test_deathwatch_install_noop_without_fd(self, monkeypatch):
        from biopb_mcp.mcp import _deathwatch

        monkeypatch.delenv(_deathwatch.ENV_FD, raising=False)
        assert _deathwatch.install() is False

    @posix_only
    def test_deathwatch_self_terminates_on_pipe_eof(self, monkeypatch):
        # In-process exercise of the watcher: install() on a real pipe, then
        # close the write end (the launcher "dying"); the watcher thread should
        # hit EOF and call the group-kill. killpg is stubbed so we record the
        # call instead of killing the test process.
        from biopb_mcp.mcp import _deathwatch

        r, w = os.pipe()
        monkeypatch.setenv(_deathwatch.ENV_FD, str(r))
        killed = []
        monkeypatch.setattr(
            os, "killpg", lambda pg, sig: killed.append((pg, sig))
        )

        assert _deathwatch.install() is True
        os.close(w)  # launcher "dies" -> read end sees EOF
        # The watcher closes the read end itself (in its finally), so we don't.
        assert _wait_until(lambda: bool(killed), timeout=5.0, interval=0.05)
        assert killed[0][1] == signal.SIGKILL

    @posix_only
    def test_kernel_dies_when_launcher_dies(self):
        import subprocess
        import textwrap

        # A throwaway "launcher" that starts a kernel and then dies abruptly
        # (os._exit, no shutdown). Its death closes the pipe write end, which
        # the kernel's watcher sees as EOF and self-group-kills on.
        script = textwrap.dedent(
            """
            import os, sys
            from biopb_mcp.mcp._kernel import KernelHost
            h = KernelHost(health_probe_code=None, watchdog_interval=0,
                           parent_death_pipe=True)
            h.start()
            sys.stdout.write(str(h._kernel_pid()) + "\\n")
            sys.stdout.flush()
            os._exit(0)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ),
        )
        assert proc.stdout.strip(), proc.stderr
        pid = int(proc.stdout.strip().splitlines()[-1])

        def _gone():
            try:
                os.kill(pid, 0)
                return False
            except ProcessLookupError:
                return True
            except PermissionError:
                return False

        assert _wait_until(_gone), f"kernel {pid} survived launcher death"


# ---------------------------------------------------------------------------
# Full napari bootstrap — only in a real desktop session.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("QT_QPA_PLATFORM") == "offscreen"
    or not os.getenv("DISPLAY")
    or (sys.platform == "darwin" and os.getenv("CI") == "true"),
    reason="napari bootstrap needs a real display",
)
class TestNapariBootstrap:
    @pytest.fixture
    def napari_kernel(self):
        line = "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
        host = KernelHost(
            extra_arguments=[f"--IPKernelApp.exec_lines={line}"],
            startup_timeout=120.0,
        )
        host.start()
        yield host
        host.shutdown()

    def test_viewer_in_namespace(self, napari_kernel):
        res = napari_kernel.execute("print('viewer' in dir())")
        assert "True" in res["stdout"]

    def test_screenshot_round_trips(self, napari_kernel):
        snippet = (
            "import base64 as _b64, cv2 as _cv2\n"
            "_arr = viewer.screenshot(canvas_only=True)\n"
            "_bgra = _cv2.cvtColor(_arr, _cv2.COLOR_RGBA2BGRA)\n"
            "_ok, _buf = _cv2.imencode('.png', _bgra)\n"
            "print('<<PNG_B64>>' + _b64.b64encode(_buf.tobytes()).decode())\n"
        )
        res = napari_kernel.execute(snippet)
        assert "<<PNG_B64>>" in res["stdout"]
