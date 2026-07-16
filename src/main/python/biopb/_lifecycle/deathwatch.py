"""In-child parent-death watcher for the "owned child" lifecycle pattern.

An owned child inherits the *read* end of a pipe whose *write* end its parent
holds (the fd number is passed in ``BIOPB_PARENT_DEATH_FD``). When the parent
**process** dies for any reason -- ``SIGKILL``, OOM, segfault, crash -- the OS
closes its write end, this watcher's blocking ``os.read`` returns EOF, and the
child group-kills itself, so it does not outlive its parent (issue #13, failure
mode 1). Shared by both owned-child owners: the biopb-mcp kernel (installed via
``IPKernelApp.exec_lines``, watching its session-child parent) and -- once the
control supervisor passes the fd -- the tensor server (watching the control).

For the kernel specifically: the dask cluster is *not* in the kernel's group --
it lives in the kernel's parent, the shim-owned session child (the MCP server
process). On that parent's death its workers self-terminate on scheduler loss --
the sole reaper on an *uncatchable* parent death (the parent's own ``_shutdown``
/ ``atexit`` close covers the graceful exits), which is why the ``mcp`` extra
floors ``distributed>=2023.9`` (post-``reconnect``, when a worker that loses its
scheduler shuts down instead of retrying forever). Only under the
``dask.owner="kernel"`` escape hatch does the kernel's group also contain dask
children this reap takes down.

Why a pipe and not ``PR_SET_PDEATHSIG``: the parent-death *signal* is tied to
the **thread** that forked the child, so it fires early when a transient
restart/respawn thread exits. The pipe is tied to the parent **process** and
is cross-platform (POSIX), so it stays correct across respawn and restart.
"""

import os
import signal
import threading

ENV_FD = "BIOPB_PARENT_DEATH_FD"


def install() -> bool:
    """Start the parent-death watcher thread if a death fd was inherited.

    Returns True if a watcher was started, False if no death fd is configured
    (so a bare child launched without the pipe is unaffected).
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
                # Blocks until the parent closes its write end (process death),
                # at which point os.read returns b"" (EOF). The parent never
                # writes, so any byte returned is spurious -- keep waiting.
                while os.read(fd, 1):
                    pass
            except OSError:
                # The fd is broken / was never inherited: the watcher can't
                # function. Do NOT treat this as parent death -- self-killing
                # here would take the child down at startup. Just stop.
                return
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        # Clean EOF: the parent process has died -- self-terminate.
        _self_terminate()

    threading.Thread(target=_watch, name="parent-death-watch", daemon=True).start()
    return True


def _self_terminate():
    """Hard group-kill: reap this child and any subprocess it spawned (agent
    code; and, under owner="kernel", a kernel-local dask cluster)."""
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception:
        os._exit(1)
