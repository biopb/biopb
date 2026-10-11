"""Cross-process test for biopb._lifecycle.deathwatch (the parent-death pipe).

Proves the Pattern-O invariant the control relies on: a child that installs the
watcher self-terminates when its parent process dies *uncatchably* (SIGKILL), so
an orphan can never outlive the parent that owned it. POSIX-only — the pipe is
the POSIX half of the bind (Windows uses a Job Object, covered elsewhere).

The test can't kill pytest itself, so it spawns a throwaway *parent* subprocess
that arms the pipe and launches the *child*; killing that parent is what the
child must notice.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="parent-death pipe is the POSIX bind"
)

# Child: install the watcher off the inherited fd, publish our pid, then idle
# forever. When the parent dies the pipe EOFs and the watcher group-kills us.
_CHILD = (
    "import os, sys, time\n"
    "from biopb._lifecycle import deathwatch\n"
    "assert deathwatch.install() is True\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "time.sleep(300)\n"
)

# Parent: arm the pipe exactly as the supervisor does (pass_fds + env +
# start_new_session so the child's killpg stays contained), keep the write end,
# then idle so the test can kill it.
_PARENT = (
    "import os, sys, subprocess, time\n"
    "r, w = os.pipe()\n"
    "env = dict(os.environ, BIOPB_PARENT_DEATH_FD=str(r))\n"
    "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]],\n"
    "                 pass_fds=(r,), env=env, start_new_session=True)\n"
    "os.close(r)\n"
    "time.sleep(300)\n"
)


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _await_pid(path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = path.read_text().strip()
        except OSError:
            text = ""
        if text:
            return int(text)
        time.sleep(0.05)
    raise TimeoutError("child never reported its pid")


def test_child_dies_when_parent_killed(tmp_path):
    pidfile = tmp_path / "child.pid"
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, str(pidfile), _CHILD])
    child_pid = None
    try:
        child_pid = _await_pid(pidfile, timeout=20)
        assert _alive(child_pid)  # child up and watching

        parent.kill()  # uncatchable SIGKILL — no parent code runs
        parent.wait(timeout=10)

        # The child's watcher sees the pipe EOF and self-terminates.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _alive(child_pid):
            time.sleep(0.1)
        assert not _alive(child_pid), "child outlived its killed parent"
    finally:
        try:
            parent.kill()
        except OSError:
            pass
        if child_pid is not None and _alive(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except OSError:
                pass
