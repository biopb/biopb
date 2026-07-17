"""Detached-daemon lifecycle: pidfile identity, graceful stop, console detach.

The *other* process-lifecycle pattern in this package. Where :mod:`.owned_child`
covers a child a live parent holds by its OS handle and reaps, these helpers
cover a **detached daemon**: a background process that outlives the command that
spawned it, is found again by a pidfile rather than a handle, and is stopped by
signalling that pid. The one such daemon in biopb is the **control plane**
(``biopb control start``); the former standalone tensor-server / biopb-mcp
daemons are gone, so there is a single owner today.

The delicate part is *identity across a reused pid*: a pidfile records the pid
plus a process create-time token (see :mod:`biopb._lifecycle.proc`), so
``stop``/``status`` never signal or trust an unrelated process that later
inherited the pid. The
pidfile is written atomically so a racing reader — or a racing writer, now that
the shim can start the control on demand — never sees a torn record.

Kept **stdlib-only** (like the rest of :mod:`biopb._lifecycle`): the one place
that wants to surface a message to a user, :func:`stop_daemon`, takes a ``notify``
callback rather than importing a console, so this module never drags in ``rich``
or ``typer``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from .proc import (
    is_process_running as _is_process_running,
    process_create_time as _process_create_time,
)


def read_pid_record(pid_file: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read (pid, identity_token) from `pid_file`.

    The file holds one or two whitespace-separated integers: the PID and, since
    the PID-identity fix, a process create-time token (see _process_create_time)
    that distinguishes our daemon from an unrelated process that later inherited
    a reused PID. Tolerates the legacy bare-PID format (token None -> identity
    unverifiable, callers fall back to a liveness-only check) so a pre-upgrade
    file still reads. Returns (None, None) if missing or unparseable.
    """
    if not pid_file.exists():
        return None, None
    try:
        parts = pid_file.read_text().split()
        pid = int(parts[0])
    except (OSError, ValueError, IndexError):
        return None, None
    token: Optional[int] = None
    if len(parts) > 1:
        try:
            token = int(parts[1])
        except ValueError:
            token = None
    return pid, token


def write_pid_file(pid_file: Path, pid: int, token: Optional[int] = None):
    """Write `pid` (and, when known, its create-time `token`) to `pid_file`.

    Written atomically (sibling temp file + os.replace on the same filesystem) so a
    concurrent reader -- or a racing writer, now that several processes can start
    the control plane on demand at once -- never observes a half-written or
    interleaved file. A torn read parses as "no daemon" (read_pid_record) and
    would trigger a spurious extra spawn. The two-line `pid\\ntoken` form is read
    back by read_pid_record; a None token falls back to the legacy bare-PID form
    (callers then verify by liveness only).
    """
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    content = f"{pid}\n{token}" if token is not None else str(pid)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{pid_file.name}-", suffix=".tmp", dir=str(pid_file.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, pid_file)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remove_pid_file(pid_file: Path):
    """Remove `pid_file` if present."""
    if pid_file.exists():
        pid_file.unlink()


def is_our_daemon(pid: Optional[int], token: Optional[int]) -> bool:
    """Whether `pid` is alive AND is the daemon we recorded -- not a reused PID.

    Returns False only when the PID can be PROVEN to be someone else (alive but a
    different creation time), so `stop`/`restart` never force-kill, and `status`
    never trusts, an unrelated process. When identity can't be established -- a
    legacy bare-PID file, or a platform/moment with no create-time -- it falls
    back to liveness, matching the pre-fix behavior rather than risk a false
    "stopped" (which would strand a running daemon).
    """
    if not pid or not _is_process_running(pid):
        return False
    if token is None:
        return True
    current = _process_create_time(pid)
    if current is None:
        return True
    return current == token


# Diagnostic from the most recent win_request_shutdown() failure, surfaced by
# stop_daemon() so a Windows user can see why graceful stop fell back to force-kill.
_LAST_WIN_SHUTDOWN_DIAG: Optional[str] = None


def win_request_shutdown(sentinel: Path) -> bool:
    """Windows: ask a daemon to shut down gracefully by dropping its stop
    sentinel. Returns True if the request was delivered (not whether the
    process has exited yet).

    The daemon is a windowless background process in its own process group, so
    it has no console to receive a CTRL_BREAK, and Win32 named events are brittle
    across sessions/elevation. We instead drop a sentinel *file* the daemon polls
    for (the control watches via biopb_control._run; its supervised tensor server
    via _install_windows_shutdown_listener in
    biopb_tensor_server.serving.http_server); a file under the user's biopb dir is
    unambiguous regardless of how the process was launched.
    """
    global _LAST_WIN_SHUTDOWN_DIAG
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("stop")
        return True
    except OSError as e:
        _LAST_WIN_SHUTDOWN_DIAG = f"could not write shutdown sentinel: {e}"
        return False


def win_remove_sentinel(sentinel: Path) -> None:
    """Remove a shutdown sentinel (best effort), so it doesn't linger after a
    force-kill where the daemon never consumed it."""
    try:
        sentinel.unlink()
    except OSError:
        pass


def request_graceful_stop(pid: int, sentinel: Path) -> bool:
    """Ask a daemon to shut down gracefully. Returns whether the request was
    delivered (not whether the process has exited yet). `sentinel` is that
    daemon's Windows stop-sentinel path (unused on POSIX, which signals)."""
    if sys.platform == "win32":
        return win_request_shutdown(sentinel)
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def stop_daemon(
    pid: int,
    timeout: int,
    token: Optional[int] = None,
    *,
    sentinel: Path,
    remove_pid: Callable[[], None],
    notify: Optional[Callable[[str], None]] = None,
) -> bool:
    """Stop a running daemon: request graceful shutdown, wait up to `timeout`
    seconds, then force-kill. Clears the PID file via `remove_pid`. Returns True
    if it exited gracefully, False if it had to be force-killed. Assumes `pid` is
    running.

    The stop path for the control daemon -- the only SDK daemon today. `sentinel`
    (its Windows stop-sentinel path) and `remove_pid` (which PID file to clear)
    stay parameters so a future owner reuses the same path.

    On POSIX, graceful shutdown is a SIGTERM the daemon's handler catches. On
    Windows os.kill is TerminateProcess -- immediate and uncatchable, so a
    handler never runs; the control-managed daemon therefore watches for a
    sentinel *file* instead (the control via biopb_control._run; its supervised
    tensor server via http_server._install_windows_shutdown_listener). See
    request_graceful_stop.

    `token` is the recorded create-time identity (see _process_create_time):
    delivery, the wait loop, and the force-kill are all gated on it so that if
    the daemon exits and its PID is reused mid-stop, we neither signal, keep
    waiting on, nor TerminateProcess the innocent new owner.

    `notify`, if given, is called with the Windows graceful-stop diagnostic when
    delivery falls back to a force-kill -- the caller renders it (this module
    stays free of a console).
    """
    if is_our_daemon(pid, token):
        delivered = request_graceful_stop(pid, sentinel)
        if not delivered and sys.platform == "win32" and _LAST_WIN_SHUTDOWN_DIAG:
            if notify is not None:
                notify(_LAST_WIN_SHUTDOWN_DIAG)

    graceful = False
    for _ in range(timeout):
        if not is_our_daemon(pid, token):
            graceful = True
            break
        time.sleep(1)

    if not graceful and is_our_daemon(pid, token):
        # Force kill. signal.SIGKILL is POSIX-only; on Windows fall back to
        # SIGTERM, which os.kill maps to an unconditional TerminateProcess.
        # Re-verify identity first: a reused PID must never take this
        # unconditional kill (and short-circuited out above when graceful).
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass
        time.sleep(0.5)

    remove_pid()
    if sys.platform == "win32":
        # tidy up if the daemon never consumed it
        win_remove_sentinel(sentinel)
    return graceful


def detach_kwargs() -> dict:
    """Popen kwargs that detach a spawned daemon from the launching console and
    process group, so it survives this command returning and isn't killed by
    the terminal's Ctrl+C or close (SIGHUP).

    POSIX: start_new_session (setsid) gives the daemon its own session/process
    group. Windows: CREATE_NO_WINDOW runs it without a console *window* (so none
    pops); CREATE_NEW_PROCESS_GROUP makes it a group leader the terminal's
    Ctrl+C does not reach (start_new_session is a silent no-op on Windows).

    Note this detaches the daemon *within* the login session; it does NOT make
    it persistent across logout, and is not meant to. Windows hard-kills session
    processes on logout, and modern systemd-logind kills the user scope on
    logout regardless of setsid (unless `loginctl enable-linger` is set). A
    daemon that must outlive the session — e.g. a shared/remote control plane —
    must be made persistent at the container/service/wrapper level (a systemd
    unit, `enable-linger`, or a long-lived container), not here.
    """
    if sys.platform == "win32":
        return {
            "creationflags": (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        }
    return {"start_new_session": True}
