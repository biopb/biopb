"""Tests for biopb._lifecycle.owned_child (the Pattern O primitive).

Spawn a real Python child, assert liveness/identity, and the graceful-then-hard
stop escalation. Runs on every OS: the stop escalation differs by platform
(SIGTERM->SIGKILL on POSIX; TerminateJobObject on Windows), and spawn binds a
Job Object on Windows (a no-op off it), so the identity check gates on os.name.
"""

import os
import subprocess
import sys
import time

from biopb._lifecycle.owned_child import OwnedChild, open_child_log


class TestOpenChildLog:
    def test_writes_to_path_creating_parent(self, tmp_path):
        path = tmp_path / "sub" / "child.log"  # parent does not exist yet
        fh, to_file = open_child_log(str(path))
        assert to_file is True
        try:
            fh.write(b"hello\n")
        finally:
            fh.close()
        assert path.read_bytes() == b"hello\n"

    def test_falls_back_to_stderr_when_unopenable(self, tmp_path):
        # Parent path is a FILE, so mkdir(parents=True) fails regardless of
        # privilege (works even when tests run as root).
        blocker = tmp_path / "afile"
        blocker.write_text("x")
        fh, to_file = open_child_log(str(blocker / "sub" / "x.log"))
        assert to_file is False
        assert fh is getattr(sys.stderr, "buffer", sys.stderr)


def _sleeper():
    return OwnedChild([sys.executable, "-c", "import time; time.sleep(30)"])


class TestSpawn:
    def test_spawn_reports_live_identity(self):
        child = _sleeper().spawn()
        try:
            assert child.pid is not None and child.pid > 0
            assert child.alive() is True
            assert child.poll() is None
            assert child.returncode is None
            if os.name == "nt":
                assert child.job is not None  # Windows: a kill-on-close Job Object
            else:
                assert child.job is None  # POSIX: reaped via the group, not a job
        finally:
            child.stop()

    def test_before_spawn_is_inert(self):
        child = _sleeper()
        assert child.pid is None
        assert child.alive() is False
        assert child.poll() is None
        child.stop()  # no-op, must not raise


class TestStop:
    def test_stop_reaps_running_child(self):
        child = _sleeper().spawn()
        assert child.alive()
        child.stop()
        assert child.poll() is not None
        assert child.alive() is False

    def test_stop_is_idempotent_on_dead_child(self):
        child = OwnedChild([sys.executable, "-c", "pass"]).spawn()
        child.wait(timeout=10)
        child.stop()
        child.stop()  # second call must be a no-op
        assert child.poll() is not None

    def test_stop_escalates_to_sigkill_on_ignored_sigterm(self):
        # A child that traps SIGTERM and keeps running forces the SIGKILL leg.
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(60)\n"
        )
        child = OwnedChild([sys.executable, "-c", code], log=subprocess.DEVNULL).spawn()
        try:
            time.sleep(0.5)  # let the child install its SIGTERM-ignore handler
            child.stop(timeout=1.0)  # SIGTERM ignored -> escalate to SIGKILL
            assert child.poll() is not None
        finally:
            child.stop()


class TestAdopt:
    def test_adopt_then_stop(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        child = OwnedChild.adopt(proc)
        assert child.proc is proc
        assert child.alive()
        child.stop()
        assert proc.poll() is not None

    def test_adopt_already_dead_is_noop(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        child = OwnedChild.adopt(proc)
        child.stop()  # must not raise
        assert child.poll() is not None
