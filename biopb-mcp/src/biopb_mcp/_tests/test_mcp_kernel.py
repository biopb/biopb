"""Tests for KernelHost (the child Jupyter kernel manager).

The unit tests start a *plain* python kernel (no napari bootstrap, no display)
and exercise execute/interrupt/restart/shutdown.  A separate, display-gated
test runs the real napari bootstrap end-to-end.
"""

import os
import signal
import sys
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from biopb_mcp.mcp import _kernel  # noqa: E402
from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


class TestConfigureDask:
    """Unit tests for _configure_dask (no kernel / no display needed)."""

    def test_in_process_scheduler_returns_no_client(self):
        """threads/synchronous schedulers yield no client and no cluster."""
        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask({"dask": {"scheduler": "threads"}})
        assert client is None
        assert cluster is None

    def test_external_address_connects_without_cluster(self, monkeypatch):
        """distributed + an explicit address attaches a Client, no cluster."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        monkeypatch.delenv("BIOPB_DASK_ADDRESS", raising=False)
        created = {}

        class _FakeClient:
            def __init__(self, address):
                created["address"] = address

        monkeypatch.setattr(dd, "Client", _FakeClient)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {
                "dask": {
                    "scheduler": "distributed",
                    "address": "tcp://1.2.3.4:8786",
                }
            }
        )
        assert isinstance(client, _FakeClient)
        assert created["address"] == "tcp://1.2.3.4:8786"
        assert cluster is None

    def test_injected_address_takes_precedence(self, monkeypatch):
        """BIOPB_DASK_ADDRESS (daemon-injected) wins over the config address."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        monkeypatch.setenv("BIOPB_DASK_ADDRESS", "tcp://daemon:8786")
        created = {}

        class _FakeClient:
            def __init__(self, address):
                created["address"] = address

        monkeypatch.setattr(dd, "Client", _FakeClient)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {"dask": {"scheduler": "distributed", "address": "tcp://cfg:1"}}
        )
        assert created["address"] == "tcp://daemon:8786"
        assert cluster is None

    def test_daemon_owner_no_address_falls_back_to_threads(self, monkeypatch):
        """owner='daemon' with no injected address -> threads, not a competing
        kernel-local cluster (LocalCluster must never be constructed)."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        monkeypatch.delenv("BIOPB_DASK_ADDRESS", raising=False)

        def _must_not_spin(*args, **kwargs):
            raise AssertionError("owner=daemon must not spin a kernel-local cluster")

        monkeypatch.setattr(dd, "LocalCluster", _must_not_spin)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {
                "dask": {
                    "scheduler": "distributed",
                    "address": "",
                    "owner": "daemon",
                }
            }
        )
        assert client is None
        assert cluster is None

    def test_kernel_owner_spins_local_cluster(self, monkeypatch):
        """owner='kernel' (escape hatch) spins a kernel-local LocalCluster."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        monkeypatch.delenv("BIOPB_DASK_ADDRESS", raising=False)
        spun = {}

        class _FakeCluster:
            def __init__(self, **kwargs):
                spun["kwargs"] = kwargs
                self.scheduler_address = "tcp://local:1"
                self.workers = {"w0": object()}

        class _FakeClient:
            def __init__(self, target):
                spun["client_target"] = target

        monkeypatch.setattr(dd, "LocalCluster", _FakeCluster)
        monkeypatch.setattr(dd, "Client", _FakeClient)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {
                "dask": {
                    "scheduler": "distributed",
                    "address": "",
                    "owner": "kernel",
                }
            }
        )
        assert isinstance(cluster, _FakeCluster)
        assert isinstance(client, _FakeClient)
        assert spun["client_target"] is cluster

    def test_local_cluster_failure_falls_back_to_threads(self, monkeypatch):
        """A LocalCluster spawn failure degrades to in-process, not a crash.

        Uses owner='kernel' so the LocalCluster branch is actually reached.
        """
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        monkeypatch.delenv("BIOPB_DASK_ADDRESS", raising=False)

        def _boom(*args, **kwargs):
            raise RuntimeError("no cluster for you")

        monkeypatch.setattr(dd, "LocalCluster", _boom)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {
                "dask": {
                    "scheduler": "distributed",
                    "address": "",
                    "owner": "kernel",
                }
            }
        )
        assert client is None
        assert cluster is None


class TestClusterAddressInjection:
    """_launch injects BIOPB_DASK_ADDRESS from cluster_host.ensure().

    Uses a real bare kernel and reads its inherited env back out, so it covers
    the full injection path.
    """

    class _FakeClusterHost:
        def __init__(self, address):
            self._address = address
            self.calls = 0

        def ensure(self):
            self.calls += 1
            return self._address

    def test_injects_address_when_ensure_returns_one(self):
        fake = self._FakeClusterHost("tcp://127.0.0.1:12345")
        host = KernelHost(
            health_probe_code=None, startup_timeout=60.0, cluster_host=fake
        )
        host.start()
        try:
            res = host.execute("import os; print(os.environ.get('BIOPB_DASK_ADDRESS'))")
            assert "tcp://127.0.0.1:12345" in res["stdout"]
            assert fake.calls >= 1
        finally:
            host.shutdown()

    def test_omits_address_when_ensure_returns_none(self, monkeypatch):
        monkeypatch.delenv("BIOPB_DASK_ADDRESS", raising=False)
        fake = self._FakeClusterHost(None)
        host = KernelHost(
            health_probe_code=None, startup_timeout=60.0, cluster_host=fake
        )
        host.start()
        try:
            res = host.execute(
                "import os; print(repr(os.environ.get('BIOPB_DASK_ADDRESS')))"
            )
            assert "None" in res["stdout"]
        finally:
            host.shutdown()


class TestTokenReportParsing:
    """Pure-unit tests for the token-report cache (issue #86), no kernel needed."""

    def test_record_updates_remembered_token(self):
        host = KernelHost(health_probe_code=None)
        host._record_token_line(b"grpc://host:9\ttok-123")
        assert host._tensor_url == "grpc://host:9"
        assert host._tensor_token == "tok-123"

    def test_empty_token_field_clears(self):
        host = KernelHost(health_probe_code=None)
        host._record_token_line(b"grpc://host:9\ttok-123")
        # User switched to a no-auth server: the report carries an empty token.
        host._record_token_line(b"grpc://other:1\t")
        assert host._tensor_url == "grpc://other:1"
        assert host._tensor_token is None

    def test_malformed_line_ignored(self):
        host = KernelHost(health_probe_code=None)
        host._record_token_line(b"grpc://host:9\ttok")
        host._record_token_line(b"no-tab-here")  # not a token-report line
        assert host._tensor_token == "tok"

    def test_watch_loop_caches_from_pipe(self):
        """The reader thread caches lines written to the pipe and exits on EOF."""
        host = KernelHost(health_probe_code=None)
        r, w = os.pipe()
        host._token_r = r
        host._start_token_watch()
        try:
            os.write(w, b"grpc://h:9\ttok-A\n")
            # Two messages in one write are both processed (line-framed).
            os.write(w, b"grpc://h:9\ttok-B\ngrpc://h:9\ttok-")
            deadline = time.time() + 5.0
            while host._tensor_token != "tok-B" and time.time() < deadline:
                time.sleep(0.02)
            assert host._tensor_token == "tok-B"  # partial trailing line buffered
        finally:
            os.close(w)  # EOF -> reader thread returns
        host._token_thread.join(timeout=5.0)
        assert not host._token_thread.is_alive()
        os.close(r)


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
        with open(log, "ab", buffering=0) as f:
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

    def test_restart_attempts_graceful_close_before_kill(self, monkeypatch):
        # restart() drops the tensor connection just as abruptly as shutdown(),
        # so it must send the same graceful-close snippet (not just the dask
        # release) before _shutdown_current() group-kills the old kernel --
        # only the timeout budget differs (restart is not on the Ctrl-C path).
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

        try:
            host.restart()

            assert calls[0][0] == "execute"
            assert calls[0][1] is _kernel._GRACEFUL_CLOSE_SNIPPET
            assert calls[0][2] == 5.0
            assert ("shutdown_current",) in calls
            # Unlike shutdown, a restart respawns: the host comes back alive.
            assert host.is_alive()
        finally:
            host.shutdown()

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

    def test_failed_start_surfaces_terminal_error_not_starting(self):
        # A bootstrap failure (probe never passes) must be distinguishable from
        # a slow-but-progressing startup: the launcher runs start() on a
        # background thread that only logs the raise, so execute()/health() are
        # the only window the client has into *why* the kernel never came up.
        host = KernelHost(
            health_probe_code="print('viewer' in dir())",
            health_probe_expect="True",
            startup_timeout=60.0,
            parent_death_pipe=False,
        )
        # Mimic the launcher's background start: swallow the raise, leave the
        # host not-ready with its recorded start_error.
        with pytest.raises(RuntimeError):
            host.start()
        try:
            assert host.health()["ready"] is False
            assert host.health()["start_error"]  # the recorded reason
            # execute() reports a terminal error immediately (no startup wait)
            # rather than the generic "starting".
            res = host.execute("print('hi')")
            assert res["status"] == "error"
            assert "startup failed" in res["error_text"].lower()
            assert "start_kernel" in res["error_text"]
        finally:
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
            assert _wait_until(lambda: host.is_alive() and host._kernel_pid() != pid1)
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
            assert "False" in host.execute("print('survivor' in dir())")["stdout"]
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
                "ready",
                "start_error",
                "teardown_reason",
                "busy",
                "dead",
                "recent_respawns",
                "watchdog_running",
            }
            assert h["alive"] is True
            assert h["ready"] is True
            assert h["start_error"] is None
            assert h["teardown_reason"] is None
            assert h["dead"] is False
            assert h["watchdog_running"] is True
        finally:
            host.shutdown()
        assert host.health()["watchdog_running"] is False


class TestReadiness:
    """The kernel boots off-thread (launcher serves the handshake first), so
    execute() returns a not-ready status immediately rather than blocking on
    bring-up; the agent polls server_status to know when to retry."""

    def test_execute_before_started_returns_not_started_immediately(self):
        # Never started (on-demand model): not ready, not starting, no
        # _start_error. execute() must return a "not_started" status pointing at
        # start_kernel *without blocking* on readiness — a blocking wait could
        # hang for the whole startup budget and trip the client's per-call
        # timeout. The agent calls start_kernel / polls server_status instead.
        host = KernelHost(startup_timeout=60.0, parent_death_pipe=False)
        assert host.health()["ready"] is False
        t0 = time.monotonic()
        res = host.execute("print('hi')")
        elapsed = time.monotonic() - t0
        assert res["status"] == "not_started"
        assert "start_kernel" in res["error_text"]
        assert "server_status" in res["error_text"]
        assert elapsed < 1.0  # returned immediately, did not wait on readiness

    def test_ready_after_start_then_cleared_on_shutdown(self):
        host = KernelHost(health_probe_code=None, parent_death_pipe=False)
        host.start()
        try:
            assert host.health()["ready"] is True
        finally:
            host.shutdown()
        assert host.health()["ready"] is False


class TestStartRestartSerialization:
    """The launcher runs start() on a background thread, so a restart_kernel
    can land while the initial start() is still in _launch(). start() and
    restart() must serialize on the lifecycle lock — otherwise both mutate the
    shared _km/_kc/_pgid state at once (wrong kernel / orphaned process)."""

    def test_restart_during_startup_is_serialized(self):
        import threading

        host = KernelHost(
            health_probe_code=None,
            parent_death_pipe=False,
            watchdog_interval=0,
        )
        real_launch = host._launch
        counter_lock = threading.Lock()  # guards the counters, not the host
        active = {"n": 0, "max": 0}

        def slow_launch():
            # Widen the bring-up window and record overlap: if start() and
            # restart() are NOT serialized, both reach here at once and max > 1.
            with counter_lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            try:
                time.sleep(0.3)
                real_launch()
            finally:
                with counter_lock:
                    active["n"] -= 1

        host._launch = slow_launch

        start_thread = threading.Thread(target=host.start, name="kernel-start")
        start_thread.start()
        try:
            # Fire a restart while start() is still inside its (slowed) _launch.
            time.sleep(0.1)
            host.restart()
            start_thread.join(timeout=60.0)
            assert not start_thread.is_alive()
            # The two lifecycle ops never overlapped, and exactly one live
            # kernel remains and accepts work.
            assert active["max"] == 1
            assert host.is_alive()
            assert host.execute("x = 1")["status"] == "ok"
        finally:
            host.shutdown()


class TestParentDeathPipe:
    """Fix 1: kernel self-terminates when the launcher process dies."""

    def test_deathwatch_install_noop_without_fd(self, monkeypatch):
        from biopb._lifecycle import deathwatch as _deathwatch

        monkeypatch.delenv(_deathwatch.ENV_FD, raising=False)
        assert _deathwatch.install() is False

    @posix_only
    def test_deathwatch_self_terminates_on_pipe_eof(self, monkeypatch):
        # In-process exercise of the watcher: install() on a real pipe, then
        # close the write end (the launcher "dying"); the watcher thread should
        # hit EOF and call the group-kill. killpg is stubbed so we record the
        # call instead of killing the test process.
        from biopb._lifecycle import deathwatch as _deathwatch

        r, w = os.pipe()
        monkeypatch.setenv(_deathwatch.ENV_FD, str(r))
        killed = []
        monkeypatch.setattr(os, "killpg", lambda pg, sig: killed.append((pg, sig)))

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


class TestOnDemandStart:
    """ensure_started(): the kernel is launched on demand, not at construction."""

    def test_idle_host_reports_not_started(self):
        # Constructed but never started: execute() must return a structured
        # not_started status pointing at start_kernel, not block or crash.
        host = KernelHost(health_probe_code=None, watchdog_interval=0)
        try:
            res = host.execute("1 + 1")
            assert res["status"] == "not_started"
            assert "start_kernel" in res["error_text"]
            assert host.health()["ready"] is False
            assert host.is_alive() is False
        finally:
            host.shutdown()

    def test_ensure_started_is_synchronous_and_idempotent(self):
        host = KernelHost(health_probe_code=None, watchdog_interval=0)
        try:
            # Synchronous: ensure_started blocks until ready and reports ready
            # (no "starting"/poll dance).
            assert host.ensure_started() == {"state": "ready"}
            assert host.is_alive()
            assert host.execute("2 + 2")["status"] == "ok"
            # A ready kernel is a no-op.
            assert host.ensure_started() == {"state": "ready"}
        finally:
            host.shutdown()

    def test_ensure_started_clears_prior_start_error(self):
        # A recorded terminal failure must not wedge ensure_started: an explicit
        # (re)start clears it and re-attempts (start_kernel is the retry path).
        host = KernelHost(health_probe_code=None, watchdog_interval=0)
        try:
            host._start_error = "stale failure"
            assert host.ensure_started() == {"state": "ready"}
            assert host._start_error is None
            assert host.is_alive()
        finally:
            host.shutdown()

    def test_ensure_started_reports_error_on_failure(self):
        # A bring-up failure is returned as a state dict (not raised) so the
        # start_kernel tool can turn it into a message.
        host = KernelHost(
            health_probe_code="print('viewer' in dir())",
            health_probe_expect="True",
            startup_timeout=60.0,
            watchdog_interval=0,
        )
        try:
            result = host.ensure_started()
            assert result["state"] == "error"
            assert result["error"]
            assert host.health()["ready"] is False
        finally:
            host.shutdown()


@posix_only
class TestWindowClosePipe:
    """The reverse kernel->server pipe reaps the kernel when the window closes."""

    def test_window_close_byte_tears_down_to_idle(self):
        host = KernelHost(
            health_probe_code=None, window_close_pipe=True, watchdog_interval=0
        )
        try:
            host.start()
            assert host._window_r is not None
            # Simulate the in-kernel close hook: the kernel writes a byte to its
            # inherited write end of the window-close pipe.
            host.execute(
                "import os; os.write(int(os.environ['BIOPB_WINDOW_CLOSE_FD']), b'x')"
            )
            deadline = time.time() + 10
            while host.is_alive() and time.time() < deadline:
                time.sleep(0.05)
            assert not host.is_alive()
            assert host._teardown_reason and "window" in host._teardown_reason
            # The teardown is attributed on the next tool call...
            res = host.execute("1 + 1")
            assert res["status"] == "not_started"
            assert "window" in res["error_text"]
            # ...and cleared by an explicit restart (synchronous).
            assert host.ensure_started() == {"state": "ready"}
            assert host._teardown_reason is None
        finally:
            host.shutdown()

    def test_normal_shutdown_is_not_attributed_to_window_close(self):
        # The reader thread's EOF path (kernel died via another teardown) must
        # not misfire as a window close.
        host = KernelHost(
            health_probe_code=None, window_close_pipe=True, watchdog_interval=0
        )
        host.start()
        host.shutdown()
        assert host._teardown_reason is None

    def test_disabled_pipe_has_no_read_fd(self):
        host = KernelHost(
            health_probe_code=None,
            window_close_pipe=False,
            watchdog_interval=0,
        )
        try:
            host.start()
            assert host._window_r is None
        finally:
            host.shutdown()


class TestWindowClosePoll:
    """Windows fallback for the POSIX pipe: poll the in-kernel window-alive
    probe and reap the kernel when the user closes the napari window. The tick
    logic is unit-tested (no kernel/display) by forcing the poll path on and
    stubbing the probe; one integration test drives the real poll loop against a
    plain kernel with the probe symbol injected."""

    def _host(self):
        host = KernelHost(
            health_probe_code=None, window_close_pipe=True, watchdog_interval=0
        )
        # Force the Windows poll path regardless of the test platform.
        host._window_close_poll = True
        return host

    def test_tick_tears_down_when_window_gone(self, monkeypatch):
        host = self._host()
        host._ready.set()
        calls = {}
        monkeypatch.setattr(host, "is_busy", lambda: False)
        monkeypatch.setattr(
            host,
            "_execute_internal",
            lambda *a, **k: {"status": "ok", "stdout": "False\n"},
        )
        monkeypatch.setattr(host, "shutdown", lambda: calls.setdefault("down", True))
        assert host._window_close_tick() is True
        assert calls.get("down")
        assert host._teardown_reason and "window" in host._teardown_reason

    def test_tick_noop_when_window_alive(self, monkeypatch):
        host = self._host()
        host._ready.set()
        monkeypatch.setattr(host, "is_busy", lambda: False)
        monkeypatch.setattr(
            host,
            "_execute_internal",
            lambda *a, **k: {"status": "ok", "stdout": "True\n"},
        )
        monkeypatch.setattr(
            host, "shutdown", lambda: pytest.fail("tore down a live window")
        )
        assert host._window_close_tick() is False
        assert host._teardown_reason is None

    def test_tick_skips_busy_kernel(self, monkeypatch):
        # A running job holds the lock: never probe or tear down mid-job.
        host = self._host()
        host._ready.set()
        monkeypatch.setattr(host, "is_busy", lambda: True)
        monkeypatch.setattr(
            host,
            "_execute_internal",
            lambda *a, **k: pytest.fail("probed a busy kernel"),
        )
        monkeypatch.setattr(
            host, "shutdown", lambda: pytest.fail("tore down a busy kernel")
        )
        assert host._window_close_tick() is False

    def test_tick_inconclusive_probe_is_noop(self, monkeypatch):
        # A busy/timeout/error probe must not be read as "window gone".
        host = self._host()
        host._ready.set()
        monkeypatch.setattr(host, "is_busy", lambda: False)
        monkeypatch.setattr(
            host,
            "_execute_internal",
            lambda *a, **k: {"status": "busy", "stdout": ""},
        )
        monkeypatch.setattr(
            host, "shutdown", lambda: pytest.fail("tore down on inconclusive probe")
        )
        assert host._window_close_tick() is False

    def test_tick_skips_until_ready(self, monkeypatch):
        # _ready unset (mid (re)spawn): don't probe a half-built kernel.
        host = self._host()
        monkeypatch.setattr(host, "is_busy", lambda: False)
        monkeypatch.setattr(
            host,
            "_execute_internal",
            lambda *a, **k: pytest.fail("probed before ready"),
        )
        assert host._window_close_tick() is False

    def test_poll_loop_reaps_on_real_probe(self):
        # End-to-end against a plain kernel: the bootstrap normally injects
        # _viewer_window_alive; here we inject it returning False and let the
        # real poll thread (started by start()) detect the close and reap.
        host = self._host()
        host._window_poll_interval = 0.05
        try:
            host.start()
            assert host._window_thread is not None and host._window_thread.is_alive()
            host.execute("_viewer_window_alive = lambda: False")
            deadline = time.time() + 10
            while host.is_alive() and time.time() < deadline:
                time.sleep(0.05)
            assert not host.is_alive()
            assert host._teardown_reason and "window" in host._teardown_reason
            res = host.execute("1 + 1")
            assert res["status"] == "not_started"
            assert "window" in res["error_text"]
        finally:
            host.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="token-report pipe is POSIX-only")
class TestTokenReportPipe:
    """End-to-end token persistence across a kernel restart (issue #86)."""

    def test_kernel_reports_token_to_launcher(self):
        """A token the kernel writes to its report fd is cached in the host."""
        host = KernelHost(health_probe_code=None, watchdog_interval=0)
        try:
            host.start()
            assert host._token_r is not None
            # Simulate the in-kernel on_connect hook reporting (url, token).
            host.execute(
                "import os; os.write(int(os.environ['BIOPB_TOKEN_REPORT_FD']),"
                " b'grpc://srv:8815\\tsecret-tok\\n')"
            )
            deadline = time.time() + 10
            while host._tensor_token != "secret-tok" and time.time() < deadline:
                time.sleep(0.05)
            assert host._tensor_url == "grpc://srv:8815"
            assert host._tensor_token == "secret-tok"
        finally:
            host.shutdown()

    def test_remembered_token_is_reinjected_on_launch(self):
        """A token remembered in the host reaches the (re)launched kernel's env —
        the mechanism that lets a GUI-entered token survive restart_kernel."""
        host = KernelHost(health_probe_code=None, watchdog_interval=0)
        # Pre-seed as if a prior kernel had reported it before dying.
        host._tensor_url = "grpc://srv:8815"
        host._tensor_token = "remembered-tok"
        try:
            host.start()
            res = host.execute(
                "import os; print(os.environ.get('BIOPB_TENSOR_TOKEN'),"
                " os.environ.get('BIOPB_TENSOR_URL'))"
            )
            assert "remembered-tok" in res["stdout"]
            assert "grpc://srv:8815" in res["stdout"]
        finally:
            host.shutdown()

    def test_token_survives_across_restart(self):
        """The full loop: kernel reports a token, restart_kernel rebuilds the
        kernel, and the new kernel's env carries the remembered token."""
        host = KernelHost(health_probe_code=None, watchdog_interval=0)
        try:
            host.start()
            host.execute(
                "import os; os.write(int(os.environ['BIOPB_TOKEN_REPORT_FD']),"
                " b'grpc://srv:8815\\tround-trip-tok\\n')"
            )
            deadline = time.time() + 10
            while host._tensor_token != "round-trip-tok" and time.time() < deadline:
                time.sleep(0.05)
            assert host._tensor_token == "round-trip-tok"

            host.restart()  # the #86 repro: kernel process is replaced

            res = host.execute("import os; print(os.environ.get('BIOPB_TENSOR_TOKEN'))")
            assert "round-trip-tok" in res["stdout"]
        finally:
            host.shutdown()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object wiring (#403)")
class TestWindowsJobObjectWiring:
    """KernelHost's kill-on-close Job Object wiring (biopb/biopb#403).

    Windows-only: the wired branches gate on os.name == 'nt', and faking that
    globally is unsafe (pathlib picks WindowsPath, which cannot be instantiated
    off Windows). So these run for real on the Windows CI runner and only fake
    _winjob -- pinning the KernelHost orchestration (create the job once, assign
    each launched kernel to it, tree-kill on teardown, and drop the handle only
    on the terminal shutdown() path) without invoking the actual Win32 calls
    (those are covered end-to-end by TestWinJobReal).
    """

    def _host(self, monkeypatch, fake):
        host = KernelHost(health_probe_code=None)
        monkeypatch.setattr(_kernel, "_winjob", fake)
        return host

    def test_launch_creates_job_once_and_assigns_each_kernel(self, monkeypatch):
        fake = MagicMock()
        fake.create_kill_on_close_job.return_value = "JOB"
        host = self._host(monkeypatch, fake)
        monkeypatch.setattr(host, "_kernel_pid", lambda: 4321)

        host._assign_kernel_to_job()
        host._assign_kernel_to_job()  # a restart reuses the job, not recreates it

        fake.create_kill_on_close_job.assert_called_once()
        assert host._winjob == "JOB"
        assert fake.assign_process.call_count == 2
        fake.assign_process.assert_called_with("JOB", 4321)

    def test_shutdown_current_tree_kills_and_keeps_handle(self, monkeypatch):
        fake = MagicMock()
        host = self._host(monkeypatch, fake)
        host._winjob = "JOB"

        host._shutdown_current()  # the restart / respawn path

        fake.terminate_job.assert_called_once_with("JOB")
        fake.close_job.assert_not_called()  # kept so the next kernel reuses it
        assert host._winjob == "JOB"

    def test_shutdown_closes_handle(self, monkeypatch):
        fake = MagicMock()
        host = self._host(monkeypatch, fake)
        host._winjob = "JOB"

        host.shutdown()  # the terminal daemon-exit path

        fake.terminate_job.assert_called_once_with("JOB")  # via _shutdown_current
        fake.close_job.assert_called_once_with("JOB")
        assert host._winjob is None


@pytest.mark.skipif(os.name != "nt", reason="real Win32 Job Object (Windows CI)")
class TestWinJobReal:
    """Exercise the real _winjob ctypes path on Windows CI: a member process
    must die when the job is terminated or its last handle closes (#403)."""

    def _sleeper(self):
        import subprocess

        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])

    def test_close_job_kills_member(self):
        from biopb._lifecycle import winjob as _winjob

        job = _winjob.create_kill_on_close_job()
        assert job is not None
        proc = self._sleeper()
        try:
            assert _winjob.assign_process(job, proc.pid) is True
            # Closing the last handle fires KILL_ON_JOB_CLOSE -- the daemon-death
            # guarantee, needing no in-process teardown code.
            _winjob.close_job(job)
            job = None
            proc.wait(timeout=10)  # raises if the OS did not reap it
        finally:
            if job is not None:
                _winjob.close_job(job)
            if proc.poll() is None:
                proc.kill()

    def test_terminate_job_kills_member_and_keeps_job_usable(self):
        from biopb._lifecycle import winjob as _winjob

        job = _winjob.create_kill_on_close_job()
        assert job is not None
        proc = self._sleeper()
        try:
            assert _winjob.assign_process(job, proc.pid) is True
            _winjob.terminate_job(job)
            proc.wait(timeout=10)  # tree-killed from outside
            # Still usable after terminate: a restart reassigns a fresh kernel.
            proc2 = self._sleeper()
            try:
                assert _winjob.assign_process(job, proc2.pid) is True
            finally:
                proc2.kill()
        finally:
            _winjob.close_job(job)
            if proc.poll() is None:
                proc.kill()

    def test_wait_for_process_observes_exit(self):
        # The client-death watchdog primitive: a handle-based wait must unblock
        # with True exactly when the watched process exits (immune to pid reuse).
        import threading

        from biopb._lifecycle import winjob as _winjob

        proc = self._sleeper()
        handle = _winjob.open_for_wait(proc.pid)
        assert handle is not None
        result = {}
        t = threading.Thread(
            target=lambda: result.update(done=_winjob.wait_for_process(handle))
        )
        t.start()
        try:
            assert t.is_alive()  # still blocked while the process lives
            proc.kill()
            t.join(timeout=10)
            assert not t.is_alive()
            assert result.get("done") is True
        finally:
            if proc.poll() is None:
                proc.kill()
            t.join(timeout=5)
