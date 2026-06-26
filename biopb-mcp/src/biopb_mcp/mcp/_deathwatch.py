"""In-kernel parent-death watcher (installed via ``IPKernelApp.exec_lines``).

The MCP launcher holds the *write* end of a pipe whose *read* end is inherited
by the kernel (the fd number is passed in ``BIOPB_PARENT_DEATH_FD``). When the
launcher **process** dies for any reason — ``SIGKILL``, OOM, segfault, crash —
the OS closes its write end, this watcher's blocking ``os.read`` returns EOF,
and the kernel group-kills itself. That reaps the kernel *and* the dask child
processes it spawned (same process group) instead of orphaning the whole tree
(issue #13, failure mode 1).

Why a pipe and not ``PR_SET_PDEATHSIG``: the parent-death *signal* is tied to
the **thread** that forked the kernel, so it fires early when a transient
restart/respawn thread exits. The pipe is tied to the launcher **process** and
is cross-platform (POSIX), so it stays correct across respawn and restart.
"""

import os
import signal
import threading

ENV_FD = "BIOPB_PARENT_DEATH_FD"


def install() -> bool:
    """Start the parent-death watcher thread if a death fd was inherited.

    Returns True if a watcher was started, False if no death fd is configured
    (so a bare kernel launched without the pipe is unaffected).
    """
    fd_str = os.environ.get(ENV_FD)
    if not fd_str:
        return False
    try:
        fd = int(fd_str)
    except ValueError:
        return False

    def _watch():
        try:
            try:
                # Blocks until the launcher closes its write end (process
                # death), at which point os.read returns b"" (EOF). The
                # launcher never writes, so any byte returned is spurious —
                # keep waiting.
                while os.read(fd, 1):
                    pass
            except OSError:
                # The fd is broken / was never inherited: the watcher can't
                # function. Do NOT treat this as launcher death — self-killing
                # here would take the kernel down at startup. Just stop.
                return
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        # Clean EOF: the launcher process has died — self-terminate.
        _self_terminate()

    threading.Thread(target=_watch, name="parent-death-watch", daemon=True).start()
    return True


def _self_terminate():
    """Hard group-kill: reap this kernel and the dask children it spawned."""
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception:
        os._exit(1)
