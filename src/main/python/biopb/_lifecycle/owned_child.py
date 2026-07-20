"""``OwnedChild``: a subprocess a live parent spawns, holds, and reaps.

The shared core of the "Pattern O" (owned-child) lifecycle, used by both owners
-- the biopb-mcp stdio shim (its session child) and the control supervisor (the
tensor-server child). It captures exactly the mechanics they have in common and
nothing policy-specific:

* **identity is the OS handle** -- the live ``Popen`` object, never a pid file.
  A crashed child clears to ``None``; there is no on-disk record to go stale or
  to name a pid-reused stranger.
* **bind** -- the child dies with its parent. POSIX: no ``start_new_session``, so
  the child shares the parent's process group and the parent's group-teardown
  reaches it (a parent-death pipe, :mod:`biopb._lifecycle.deathwatch`, covers the
  *uncatchable* parent death; the owner wires that in when its child installs the
  watcher). Windows: a kill-on-close Job Object the OS empties when the parent's
  last handle closes (:mod:`biopb._lifecycle.winjob`).
* **stop** -- graceful-then-hard reap: SIGTERM -> wait -> SIGKILL on POSIX;
  ``TerminateJobObject`` (tree-kill) + a ``TerminateProcess`` backstop on Windows,
  then the job handle is released.

*Keepalive* (restart-on-crash) is **not** here -- it is the supervisor's loop,
layered on top of an ``OwnedChild``, and only the tensor server wants it. Policy
about *who* triggers a stop (a signal handler, a client-death watchdog, a
supervision tick) also stays with the owner; this class only performs it.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

from . import winjob

logger = logging.getLogger(__name__)

# How long a reap waits for the child to exit at each escalation step.
DEFAULT_REAP_TIMEOUT = 10.0


def open_child_log(path):
    """Open ``path`` for an owned child's stdout/stderr; return ``(fh, to_file)``.

    Binary + unbuffered + append: native Qt/GL/dask/gRPC writers emit arbitrary
    bytes, and the fd is inherited by the child (and its grandchildren), so it
    must not be a text wrapper. The parent directory is created. On any open
    failure this falls back to the parent's own stderr buffer and returns
    ``to_file=False`` -- the child still starts, its output just interleaves with
    the parent's logging (harmless; stderr is not a protocol channel). The caller
    owns the returned handle (closes its copy after spawn; the child holds a dup).
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return open(path, "ab", buffering=0), True
    except OSError:
        logger.warning("Could not open child log %s; routing output to stderr", path)
        return getattr(sys.stderr, "buffer", sys.stderr), False


class OwnedChild:
    """A parent-owned subprocess: spawn, bind, poll, stop. No pid file."""

    def __init__(self, argv, *, log=None, env=None, cwd=None):
        self._argv = list(argv)
        self._log = log
        self._env = env
        self._cwd = cwd
        self._proc = None
        self._job = None

    @classmethod
    def adopt(cls, proc, job=None):
        """Wrap an already-spawned ``Popen`` (+ optional Job Object) as an owned
        child, for a caller that ran the ``Popen`` itself. ``stop`` then reaps it
        with the same escalation a spawned child gets."""
        self = cls([])
        self._proc = proc
        self._job = job
        return self

    def spawn(self):
        """Start the child with the owned-child conventions and bind it to this
        parent; return ``self``.

        ``stdin`` is ``DEVNULL`` (an owned child never reads the parent's stdin),
        stdout/stderr go to ``log``, and ``close_fds`` keeps unrelated parent fds
        out of the child. POSIX: intentionally **no** ``start_new_session`` -- the
        child shares this parent's process group so a group-teardown reaches it.
        Windows: ``CREATE_NO_WINDOW`` (no console window pops) and a kill-on-close
        Job Object the child is assigned to -- not a new process group, since the
        job, not group signals, ties the child here.
        """
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=self._log,
            close_fds=True,
            env=self._env,
            cwd=self._cwd,
            **kwargs,
        )
        if os.name == "nt":
            self._job = winjob.create_kill_on_close_job()
            winjob.assign_process(self._job, self._proc.pid)
        return self

    @property
    def proc(self):
        return self._proc

    @property
    def job(self):
        """The bound Job Object handle (Windows) or ``None``. Exposed mainly for
        assertions/diagnostics; ``stop`` manages it internally."""
        return self._job

    @property
    def pid(self):
        return self._proc.pid if self._proc is not None else None

    @property
    def returncode(self):
        return self._proc.returncode if self._proc is not None else None

    def poll(self):
        """The child's exit code if it has exited, else ``None`` (also ``None``
        when nothing was spawned). ``poll`` reaps the OS zombie."""
        return None if self._proc is None else self._proc.poll()

    def alive(self):
        return self._proc is not None and self._proc.poll() is None

    def wait(self, timeout=None):
        return self._proc.wait(timeout=timeout)

    def stop(self, timeout=DEFAULT_REAP_TIMEOUT):
        """Graceful-then-hard reap; idempotent and best-effort.

        Safe to call more than once and on an already-dead child. POSIX: SIGTERM
        (the child's handler gets to clean up) -> wait -> SIGKILL. Windows:
        ``TerminateJobObject`` force-reaps the whole tree, with ``TerminateProcess``
        as the backstop for when the job is unavailable, then the job handle is
        released.
        """
        proc, job = self._proc, self._job
        if proc is None:
            return
        if proc.poll() is not None:
            if job is not None:
                winjob.close_job(job)
                self._job = None
            return

        if os.name == "nt":
            winjob.terminate_job(job)
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            winjob.close_job(job)
            self._job = None
            return

        try:
            proc.terminate()  # SIGTERM -> the child's handler runs
        except OSError:
            pass
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.kill()  # SIGKILL
        except OSError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
