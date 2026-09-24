"""Tests for KernelHost (the child Jupyter kernel manager).

The unit tests start a *plain* python kernel (no napari bootstrap, no display)
and exercise execute/interrupt/restart/shutdown.  A separate, display-gated
test runs the real napari bootstrap end-to-end.
"""

import os
import signal
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from biopb_mcp.mcp import _kernel  # noqa: E402
from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


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

    def test_a_timeout_does_not_interrupt(self, kernel):
        # Whatever holds the main thread past a timeout -- an attached client's
        # cell, a job's viewer call -- is someone else's, so it runs on.
        res = kernel.execute("import time; time.sleep(2); done = True", timeout=0.5)
        assert res["status"] == "timeout"
        assert "Nothing was interrupted" in res["error_text"]
        # The next call queues behind it and sees it finished, not stopped; the
        # timed-out request's late reply is not mistaken for this one's.
        res2 = kernel.execute("print(done)", timeout=10.0)
        assert res2["status"] == "ok"
        assert res2["stdout"].strip() == "True"


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

    def test_calls_overlap_and_each_gets_its_own_output(self, kernel):
        # No host-side lock: a second call is sent while the first runs, the
        # kernel queues it, and neither call sees the other's output.
        results = {}

        def run():
            results["slow"] = kernel.execute(
                "import time; time.sleep(1.5); print('slow')", timeout=10.0
            )

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.3)
        assert kernel.is_busy()  # the kernel's own status, not a lock
        fast = kernel.execute("print('fast')", timeout=10.0)
        t.join(timeout=15.0)
        assert fast["status"] == "ok" and fast["stdout"] == "fast\n"
        assert results["slow"]["stdout"] == "slow\n"
        assert not kernel.is_busy()

    def test_a_restart_turns_new_calls_away_until_it_is_back(self, kernel):
        # A call made mid-restart must not reach the kernel being replaced (a
        # submit there would start a job that is killed, after its tensor
        # client closed), and is told the kernel is starting -- not "call
        # start_kernel", which would be wrong advice between kill and relaunch.
        restarter = threading.Thread(target=kernel.restart, daemon=True)
        restarter.start()
        # From the moment it turns calls away: one sent just before reached the
        # old kernel and fails fast with it (test_a_call_in_flight_fails_fast).
        assert _wait_until(
            lambda: not kernel._ready.is_set(), timeout=5.0, interval=0.01
        )
        seen = set()
        while restarter.is_alive():
            started = time.monotonic()
            res = kernel.execute("x = 1", timeout=30.0)
            if res["status"] != "ok":
                seen.add(res["status"])
                assert time.monotonic() - started < 1.0
            time.sleep(0.05)
        restarter.join(timeout=60.0)
        assert seen == {"starting"}
        assert kernel.execute("print('back')")["stdout"] == "back\n"

    def test_a_call_in_flight_fails_fast_on_shutdown(self, kernel):
        results = {}

        def run():
            results["res"] = kernel.execute("import time; time.sleep(30)", timeout=60.0)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.5)
        kernel._shutdown_current()
        t.join(timeout=10.0)
        assert not t.is_alive(), "the call waited out its timeout"
        assert results["res"]["status"] == "error"


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

    @staticmethod
    def _closable(host, marker):
        """Bind a ``_conn`` whose client's close writes *marker*, the way the
        kernel's graceful close (``_kernel_gate._close_session``) finds it."""
        host.execute(
            "import types\n"
            "class _Client:\n"
            f"    def close(self): open({str(marker)!r}, 'w').write('closed')\n"
            "_conn = types.SimpleNamespace(client=_Client())"
        )

    def _spy_kill(self, host, monkeypatch, marker):
        """Record, at the group-kill, whether the close had run."""
        seen = []
        real = host._shutdown_current

        def spy():
            seen.append(marker.exists())
            return real()

        monkeypatch.setattr(host, "_shutdown_current", spy)
        return seen

    def test_shutdown_closes_the_session_before_the_kill(self, monkeypatch, tmp_path):
        # So the tensor server sees a clean Flight GOAWAY rather than an abrupt
        # socket drop (which can hang a subsequent `biopb server stop`).
        host = KernelHost(health_probe_code=None, startup_timeout=60.0)
        host.start()
        marker = tmp_path / "closed"
        self._closable(host, marker)
        seen = self._spy_kill(host, monkeypatch, marker)
        host.shutdown()
        assert seen == [True]
        assert not host.is_alive()

    def test_restart_closes_the_session_before_the_kill(self, monkeypatch, tmp_path):
        # A restart drops the tensor connection just as abruptly as a shutdown.
        host = KernelHost(health_probe_code=None, startup_timeout=60.0)
        host.start()
        try:
            marker = tmp_path / "closed"
            self._closable(host, marker)
            seen = self._spy_kill(host, monkeypatch, marker)
            host.restart()
            assert seen == [True]
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

    def test_respawns_while_a_client_polls_the_dead_kernel(self):
        """A polling client must not be able to starve the respawn.

        The regression: a dead kernel leaves its zmq channels open, so every
        execute() blocked for the full execute_timeout holding the lifecycle
        lock. The observe page polls /api/jobs every few seconds, so those waits
        overlapped end to end, the lock was never free, and the watchdog -- which
        needs the same lock -- never got it. Observed in the wild as a kernel
        that stayed dead for 11 minutes, 502ing every poll, silently.

        Recovery is bounded by execute_timeout, not by how long clients keep
        polling: the one call already in flight when the kernel died still waits
        out its own timeout, but nothing queues behind it.
        """
        execute_timeout = 3.0
        host = KernelHost(
            health_probe_code=None,
            watchdog_interval=0.3,
            execute_timeout=execute_timeout,
            parent_death_pipe=False,
        )
        host.start()
        stop = threading.Event()
        pollers = []

        def _poll():
            while not stop.is_set():
                host.execute("pass")
                stop.wait(0.2)

        try:
            pid1 = host._kernel_pid()
            for _ in range(3):
                t = threading.Thread(target=_poll, daemon=True)
                t.start()
                pollers.append(t)
            os.kill(pid1, signal.SIGKILL)
            assert _wait_until(
                lambda: host.is_alive() and host._kernel_pid() != pid1,
                timeout=execute_timeout + 25.0,
            ), "watchdog never respawned while clients kept polling"
            assert not host._dead
        finally:
            stop.set()
            for t in pollers:
                t.join(timeout=15.0)
            host.shutdown()

    def test_execute_on_a_dead_kernel_fails_fast(self):
        """execute() must not wait out execute_timeout on a gone process."""
        host = KernelHost(
            health_probe_code=None,
            # The watchdog would respawn and mask the block; this test is about
            # the lock hold, so run without it.
            watchdog_interval=0,
            execute_timeout=60.0,
            parent_death_pipe=False,
        )
        host.start()
        try:
            pid = host._kernel_pid()
            os.kill(pid, signal.SIGKILL)
            assert _wait_until(lambda: not host.is_alive())
            started = time.monotonic()
            res = host.execute("pass")
            elapsed = time.monotonic() - started
            assert res["status"] == "error"
            assert elapsed < 5.0, f"blocked {elapsed:.1f}s on a dead kernel"
        finally:
            host.shutdown()

    def test_cold_start_is_not_mistaken_for_a_death(self):
        """A slow bring-up holds the lock while not alive; that is not a death.

        The watchdog's death path withdraws _ready, so it must arm only on a
        host that reached ready -- otherwise every cold start that outlasts a
        tick would trip it.
        """
        host = KernelHost(
            health_probe_code=None,
            watchdog_interval=0.05,
            parent_death_pipe=False,
        )
        host._start_watchdog()
        try:
            # Never started: not alive, never ready. Several ticks must pass
            # with no respawn attempt and no dead marking.
            time.sleep(0.5)
            assert not host._dead
            assert host._respawn_times == []
            assert host._km is None
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
                "connection_file",
                "attach_command",
            }
            assert os.path.isfile(h["connection_file"])
            from jupyter_core.paths import jupyter_runtime_dir

            # Where Jupyter tools look, not a tempfile.
            assert os.path.dirname(h["connection_file"]) == jupyter_runtime_dir()
            conn = h["connection_file"]
            assert h["attach_command"].endswith(f"-m qtconsole --existing {conn}")
            assert h["alive"] is True
            assert h["ready"] is True
            assert h["start_error"] is None
            assert h["teardown_reason"] is None
            assert h["dead"] is False
            assert h["watchdog_running"] is True
        finally:
            host.shutdown()
        assert host.health()["watchdog_running"] is False
        assert host.health()["connection_file"] is None
        assert host.health()["attach_command"] is None
        assert not os.path.exists(conn)  # jupyter_client removes it on shutdown

    @pytest.mark.parametrize(
        "windows, python, path, expected",
        [
            (
                False,
                "/home/a b/.local/share/uv/tools/biopb/bin/python",
                "/home/a b/.local/share/jupyter/runtime/kernel-1.json",
                "'/home/a b/.local/share/uv/tools/biopb/bin/python' -m qtconsole "
                "--existing '/home/a b/.local/share/jupyter/runtime/kernel-1.json'",
            ),
            (
                True,
                r"C:\Users\First Last\biopb\Scripts\python.exe",
                r"C:\Users\First Last\AppData\Roaming\jupyter\runtime\kernel-1.json",
                r'"C:\Users\First Last\biopb\Scripts\python.exe" -m qtconsole '
                r'--existing "C:\Users\First Last\AppData\Roaming\jupyter'
                r'\runtime\kernel-1.json"',
            ),
        ],
    )
    def test_attach_command_runs_our_interpreter_quoted_for_the_shell(
        self, windows, python, path, expected
    ):
        # Not a bare `jupyter`: biopb puts none on PATH. The platform is passed
        # in, never patched: `os.name` is global, and faking it mid-session
        # makes pytest's own path handling fail on Python < 3.12.
        got = _kernel.attach_command(path, python=python, windows=windows)
        assert got == expected

    def test_no_attach_command_from_a_frozen_build(self, monkeypatch):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert _kernel.attach_command("/tmp/kernel-1.json") is None


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
    shared _km/_io/_pgid state at once (wrong kernel / orphaned process)."""

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
    def test_kernel_dies_when_launcher_dies(self, tmp_path):
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
            # The launcher dies before it can clean up, so its connection file
            # stays behind: keep it out of the user's Jupyter runtime dir.
            env=dict(os.environ, JUPYTER_RUNTIME_DIR=str(tmp_path)),
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

    @pytest.fixture
    def default_config_kernel(self, tmp_path):
        """A bootstrapped kernel that reads *no* user config.

        The machine's own ``mcp-config.json`` decides whether a kernel attaches
        at startup, and an install that predates #970 has ``dask.scheduler`` set
        to ``"distributed"`` on disk -- so a test of the *default* has to isolate
        the config tree (``$BIOPB_CONFIG_HOME``, biopb/biopb#790) or it measures
        this machine instead.
        """
        line = "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
        host = KernelHost(
            extra_arguments=[f"--IPKernelApp.exec_lines={line}"],
            startup_timeout=120.0,
            env=dict(os.environ, BIOPB_CONFIG_HOME=str(tmp_path)),
        )
        host.start()
        yield host
        host.shutdown()

    def test_kernel_computes_in_process_by_default(self, default_config_kernel):
        # The #970 default, end to end: a real bootstrap spins no cluster and
        # leaves dask on the in-process scheduler the viewer reads through.
        napari_kernel = default_config_kernel
        snippet = (
            "import time as _t\n"
            "for _ in range(100):\n"
            "    if _dask_attach_done:\n"
            "        break\n"
            "    _t.sleep(0.05)\n"
            "print(_dask_client, _dask_ctl.status()['mode'], "
            "_dask_ctl.status()['scheduler'])\n"
        )
        res = napari_kernel.execute(snippet, 30.0)
        assert "None in-process threads" in res["stdout"]

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
        # A busy main thread would only queue the probe behind it, one per tick.
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
        # A timeout/error probe must not be read as "window gone".
        host = self._host()
        host._ready.set()
        monkeypatch.setattr(host, "is_busy", lambda: False)
        monkeypatch.setattr(
            host,
            "_execute_internal",
            lambda *a, **k: {"status": "timeout", "stdout": ""},
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


# ---------------------------------------------------------------------------
# A second Jupyter client on the session kernel (docs/jupyter-clients.md)
# ---------------------------------------------------------------------------

_GATED_ARGS = [
    # `_conn` because every job starts by reading `client` off it.
    "--IPKernelApp.exec_lines=import biopb_mcp.mcp._jobs as _jobs, types; "
    "_conn = types.SimpleNamespace(client=None); _jobs.install(get_ipython())",
]

# A job that keeps the worker thread busy and stops at the next bytecode when
# interrupted (a bare sleep would hold the interrupt until it returned).
_LONG_JOB = "_jobs.submit('import time\\nfor _ in range(400): time.sleep(0.05)')"


class TestJupyterClientGate:
    @pytest.fixture
    def gated(self):
        host = KernelHost(
            extra_arguments=_GATED_ARGS,
            health_probe_code="print('_jobs' in dir())",
            parent_death_pipe=False,
            window_close_pipe=False,
            watchdog_interval=0,
        )
        host.start()
        yield host
        host.shutdown()

    @pytest.fixture
    def foreign(self, gated):
        from jupyter_client import BlockingKernelClient

        kc = BlockingKernelClient(connection_file=gated.connection_file)
        kc.load_connection_file()
        kc.start_channels()
        kc.wait_for_ready(timeout=30)
        yield kc
        kc.stop_channels()

    @staticmethod
    def _run(kc, code, **kwargs):
        """Execute on *kc*; return the reply content and its iopub messages."""
        msgs = []
        reply = kc.execute_interactive(
            code, timeout=30, output_hook=msgs.append, **kwargs
        )
        return reply["content"], msgs

    @staticmethod
    def _jobs(host):
        """The host's records, once every foreign cell's end has arrived: it
        travels on iopub, which can trail the reply the client already has."""
        _wait_until(
            lambda: all(
                j["status"] != "running"
                for j in host.jobs.export()
                if j["origin"] == "user"
            ),
            timeout=5.0,
        )
        return host.jobs.export()

    @staticmethod
    def _stop_job(host):
        running = host.jobs.running()
        if running is not None:
            host.interrupt_job(running["job_id"])
        _wait_until(
            lambda: "None" in host.execute("print(_jobs.running_job())")["stdout"]
        )

    @staticmethod
    def _hold_main(host, kc, seconds=30, blocking=False):
        """Start a foreign cell that holds the main thread, in one blocking
        sleep or in short ones; return its job id once the host's records have
        it running."""
        code = (
            f"import time\ntime.sleep({seconds})"
            if blocking
            else f"import time\nfor _ in range({seconds * 20}): time.sleep(0.05)"
        )
        kc.execute(code)
        _wait_until(lambda: host.jobs.running() is not None, timeout=10.0)
        running = host.jobs.running()
        assert running is not None and running["origin"] == "user"
        return running["job_id"]

    def test_stop_reaches_a_foreign_cell_on_the_main_thread(self, gated, foreign):
        # The observe page's Stop, on a user's cell. On the control channel,
        # so it is not queued behind the cell it stops, and a real SIGINT, so
        # a blocking sleep wakes up to take it.
        job_id = self._hold_main(gated, foreign, blocking=True)
        t0 = time.monotonic()
        out = gated.interrupt_job(job_id, reason="stopped by the user")
        assert out["interrupted"] is True
        reply = foreign.get_shell_msg(timeout=10)["content"]
        assert reply["ename"] == "KeyboardInterrupt"
        assert time.monotonic() - t0 < 10
        (job,) = [j for j in self._jobs(gated) if j["job_id"] == job_id]
        assert job["status"] == "interrupted"

    def test_the_agent_is_refused_a_foreign_cell_without_waiting_on_it(
        self, gated, foreign
    ):
        job_id = self._hold_main(gated, foreign)
        try:
            t0 = time.monotonic()
            out = gated.interrupt_job(job_id, origin="mcp")
            assert out["refused"] == "foreign_job"
            assert time.monotonic() - t0 < 5
            assert gated.jobs.running()["job_id"] == job_id
        finally:
            gated.interrupt_job(job_id)
            foreign.get_shell_msg(timeout=10)

    def test_a_stop_for_a_job_that_ended_stops_nothing(self, gated, foreign):
        job_id = self._hold_main(gated, foreign)
        try:
            out = gated.interrupt_job("job-999")
            assert out["refused"] == "not_running"
            assert out["running_job_id"] == job_id
            time.sleep(0.3)
            assert gated.jobs.running()["job_id"] == job_id
        finally:
            gated.interrupt_job(job_id)
            foreign.get_shell_msg(timeout=10)

    def test_a_control_request_does_not_read_as_idle(self, gated, foreign):
        # The kernel publishes busy/idle around a control request too; the idle
        # after it says nothing about the main thread, still in the cell.
        job_id = self._hold_main(gated, foreign)
        try:
            # Any control request: the kernel knows a foreign cell by its
            # request, not the host's id, so this one answers "unknown".
            gated.control("status", job_id=job_id)
            time.sleep(0.3)
            assert gated.is_busy()
        finally:
            gated.interrupt_job(job_id)
            foreign.get_shell_msg(timeout=10)

    def test_restart_closes_the_session_while_a_cell_holds_the_main_thread(
        self, gated, foreign, tmp_path
    ):
        marker = tmp_path / "closed"
        TestKernelLifecycle._closable(gated, marker)
        self._hold_main(gated, foreign, seconds=60)
        gated.restart()
        assert marker.exists()

    def test_the_kernel_knows_its_host_from_launch(self, gated):
        res = gated.execute(
            "import biopb_mcp.mcp._kernel_gate as g; print(g._host_session)"
        )
        assert res["stdout"].strip() == gated._km.session.session

    def test_the_gate_is_armed_before_the_host_sends_anything(self):
        # No health probe, so the host never executes a thing: a gate that
        # learned its host from a request would pass this cell unrecorded.
        from jupyter_client import BlockingKernelClient

        host = KernelHost(
            extra_arguments=_GATED_ARGS,
            health_probe_code=None,
            parent_death_pipe=False,
            window_close_pipe=False,
            watchdog_interval=0,
        )
        host.start()
        kc = BlockingKernelClient(connection_file=host.connection_file)
        kc.load_connection_file()
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=30)
            reply, _ = self._run(kc, "x = 1")
            assert reply["status"] == "ok"
            assert [j["origin"] for j in self._jobs(host)] == ["user"]
        finally:
            kc.stop_channels()
            host.shutdown()

    def test_an_idle_foreign_cell_runs_and_is_recorded(self, gated, foreign):
        reply, msgs = self._run(foreign, "x = 41 + 1\nprint('hello')")
        assert reply["status"] == "ok"
        # The client still sees its own output.
        assert any(
            m["msg_type"] == "stream" and "hello" in m["content"]["text"] for m in msgs
        )
        assert gated.execute("print(x)")["stdout"].strip() == "42"
        (job,) = [j for j in self._jobs(gated) if j["origin"] == "user"]
        assert job["code"] == "x = 41 + 1\nprint('hello')"
        assert job["status"] == "ok"
        assert job["stdout"] == "hello\n"

    def test_a_failing_foreign_cell_is_recorded_as_an_error(self, gated, foreign):
        reply, _ = self._run(foreign, "1 / 0")
        assert reply["status"] == "error"
        (job,) = [j for j in self._jobs(gated) if j["origin"] == "user"]
        assert job["status"] == "error"
        assert "ZeroDivisionError" in job["error_text"]

    def test_a_client_interrupt_records_the_cell_as_interrupted(self, gated, foreign):
        msg_id = foreign.execute("import time\nfor _ in range(200): time.sleep(0.05)")
        _wait_until(
            lambda: any(
                m["msg_type"] == "execute_input"
                and m["parent_header"].get("msg_id") == msg_id
                for m in [foreign.get_iopub_msg(timeout=5)]
            )
        )
        time.sleep(0.3)
        gated.interrupt()
        foreign.get_shell_msg(timeout=30)
        (job,) = [j for j in self._jobs(gated) if j["origin"] == "user"]
        assert job["status"] == "interrupted"

    def test_a_foreign_cell_waits_for_the_agents_cell(self, gated, foreign):
        # Both run on the main thread, one at a time: the user's cell queues
        # behind the agent's instead of being refused.
        job = gated.jobs.new_id()
        gated.run_cell("import time\nfor _ in range(20): time.sleep(0.05)", job, "mcp")
        assert _wait_until(gated.is_busy, timeout=5, interval=0.01)
        reply, _ = self._run(foreign, "y = 1")
        assert reply["status"] == "ok"
        assert gated.jobs.poll(job)["status"] == "ok"
        assert gated.execute("print(y)")["stdout"].strip() == "1"
        assert [j["origin"] for j in self._jobs(gated)] == ["mcp", "user"]

    def test_a_foreign_cell_runs_beside_a_task(self, gated, foreign):
        assert gated.execute(_LONG_JOB)["status"] == "ok"
        try:
            reply, _ = self._run(foreign, "y = 1")
            assert reply["status"] == "ok"
            assert gated.jobs.running(prefer="mcp")["status"] == "running"
        finally:
            self._stop_job(gated)

    def test_silent_code_is_recorded_like_any_cell(self, gated, foreign):
        # `silent` only stops output being broadcast; the code still runs with
        # full effect, so the agent is told of it all the same.
        reply, _ = self._run(foreign, "z = 3", silent=True)
        assert reply["status"] == "ok"
        (job,) = self._jobs(gated)
        assert job["origin"] == "user" and job["code"] == "z = 3"

    def test_an_empty_request_passes_while_a_job_runs(self, gated, foreign):
        # What qtconsole sends silently: a prompt-number request, and
        # user_expressions evaluated for its UI.
        assert gated.execute(_LONG_JOB)["status"] == "ok"
        try:
            reply, _ = self._run(
                foreign, "", silent=True, user_expressions={"k": "1 + 1"}
            )
            assert reply["status"] == "ok"
            assert reply["user_expressions"]["k"]["data"]["text/plain"] == "2"
            assert [j["origin"] for j in self._jobs(gated)] == ["mcp"]
        finally:
            self._stop_job(gated)

    def test_a_host_call_aborted_by_a_failing_cell_is_retried(self, gated, foreign):
        # A client's stop_on_error makes ipykernel abort what is queued behind
        # its failing cell; the host retries an aborted snippet once. Queue both
        # behind a sleep so the poll is waiting when the failure lands.
        statuses = []
        run_once = gated._run_once

        def spy(*args):
            res = run_once(*args)
            statuses.append(res["status"])
            return res

        gated._run_once = spy
        try:
            foreign.execute(
                "",
                silent=True,
                user_expressions={"s": "__import__('time').sleep(1.5)"},
            )
            foreign.execute("1 / 0")
            time.sleep(0.3)
            res = gated.execute("print('poll')")
            assert res["status"] == "ok"
            assert res["stdout"].strip() == "poll"
            assert statuses == ["aborted", "ok"]
        finally:
            gated._run_once = run_once

    def test_a_host_timeout_leaves_a_long_foreign_cell_running(self, gated, foreign):
        # A host call queued behind a client's long cell used to SIGINT it.
        msg_id = foreign.execute("import time\nfor _ in range(40): time.sleep(0.05)")
        time.sleep(0.5)
        res = gated.execute("print(1)", timeout=0.5)
        assert res["status"] == "timeout"
        reply = foreign.get_shell_msg(timeout=30)
        assert reply["parent_header"]["msg_id"] == msg_id
        assert reply["content"]["status"] == "ok"


class TestHostRecords:
    """The job records the host builds from iopub (``_job_log``)."""

    @pytest.fixture
    def host(self):
        host = KernelHost(
            extra_arguments=_GATED_ARGS,
            health_probe_code="print('_jobs' in dir())",
            parent_death_pipe=False,
            window_close_pipe=False,
            watchdog_interval=0,
        )
        host.start()
        yield host
        host.shutdown()

    @staticmethod
    def _submit(host, code):
        from biopb_mcp.mcp import _kernel_rpc

        sub, res, _w = _kernel_rpc._run_job_call(
            host, "submit", code, job_id=host.jobs.new_id(), timeout=15.0
        )
        assert sub is not None, res
        return sub["job_id"]

    def test_a_jobs_output_streams_in_while_it_runs(self, host):
        jid = self._submit(
            host,
            "import time\nprint('first', flush=True)\ntime.sleep(1.5)\nprint('second')\n6 * 7",
        )
        assert _wait_until(lambda: "first" in host.jobs.poll(jid)["stdout"], timeout=5)
        assert host.jobs.poll(jid)["status"] == "running"
        assert _wait_until(lambda: host.jobs.poll(jid)["status"] == "ok", timeout=10)
        snap = host.jobs.poll(jid)
        assert snap["stdout"] == "first\nsecond\n"
        assert snap["result_text"] == "42"

    def test_the_submit_reply_is_not_the_jobs_output(self, host):
        # The payload rides a user_expression, so a job printing at once cannot
        # split it, and nothing of the call lands in the job's record.
        jid = self._submit(host, "print('x', end='')")
        assert _wait_until(lambda: host.jobs.poll(jid)["status"] == "ok", timeout=5)
        assert host.jobs.poll(jid)["stdout"] == "x"

    def test_a_failing_job(self, host):
        jid = self._submit(host, "1 / 0")
        assert _wait_until(lambda: host.jobs.poll(jid)["status"] == "error", timeout=5)
        assert "ZeroDivisionError" in host.jobs.poll(jid)["error_text"]

    def test_records_survive_a_restart_and_ids_continue(self, host):
        done = self._submit(host, "print('kept')")
        assert _wait_until(lambda: host.jobs.poll(done)["status"] == "ok", timeout=5)
        running = self._submit(host, "import time\ntime.sleep(30)")
        host.restart()
        assert host.jobs.poll(done)["stdout"] == "kept\n"
        snap = host.jobs.poll(running)
        assert snap["status"] == "interrupted"
        assert "kernel stopped" in snap["error_text"]
        after = self._submit(host, "1")
        assert int(after.split("-")[1]) > int(running.split("-")[1])
