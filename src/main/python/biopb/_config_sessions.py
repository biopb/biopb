"""Filesystem registry of live MCP sessions — shared, stdlib-only.

Two independent processes need to agree on where the ephemeral MCP sessions are:

- ``biopb-mcp``'s stdio **shim**, which spawns a private http session child on a
  dynamic port and must *publish* (session id → port + pid) so the session is
  discoverable, and
- the **control plane** (``biopb-control``), which reads this registry to list
  live sessions (``/api/sessions``) and reverse-proxy ``/session/<id>/*`` to the
  right port.

Neither can import the other (``biopb-mcp`` cannot import ``biopb-control`` any
more than ``biopb-tensor-server`` — see the "shared config lives in core biopb
SDK" rationale), so the one thing they must share — the on-disk layout — lives
here in the dependency-light core ``biopb`` SDK, beside ``_config_control`` /
``_config_location``. Kept **stdlib-only** so importing it never drags in the
heavy server/mcp stacks and stays cheap on the featherweight shim.

Why a filesystem registry (see ``mcp-dedaemonization-migration.md`` §6.1): it
matches the existing sentinel/PID-file idioms, needs no always-listening
endpoint, and self-heals — a reader prunes any record whose owning process is
gone, so a shim killed hard enough to skip its reap (§6.1 gotcha 3) leaves no
routing ghost. (HTTP self-registration would be the alternative only if the
front and the sessions could later cross a machine boundary.)

Concurrency: writes are atomic (temp file + ``os.replace`` in the same dir), so a
reader never observes a half-written record; a record unlinked between listing
and reading is simply skipped. Records are owned by whoever wrote them; anyone
may prune a dead one.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Record filename suffix; the stem is the session id.
_SUFFIX = ".json"


def _default_sessions_dir() -> Path:
    """Live-session records live under the shared biopb *data* tree (persistent
    runtime state, not user-editable config), mirroring the tensor server's
    ``~/.local/share/biopb/log``. Deliberately the ``biopb`` tree, not
    ``biopb-mcp`` — it is cross-component state the control (not biopb-mcp) also
    reads. Resolved at call time (not a module constant) so a test that repoints
    ``Path.home()`` gets an isolated dir for free."""
    return Path.home() / ".local" / "share" / "biopb" / "sessions"


def sessions_dir() -> Path:
    """The registry directory (``BIOPB_SESSIONS_DIR`` override, else the default).

    Created on access so both the writing shim and the reading control can call
    this without a separate mkdir step.
    """
    raw = os.environ.get("BIOPB_SESSIONS_DIR")
    d = Path(raw) if raw else _default_sessions_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_path(session_id: str) -> Path:
    return sessions_dir() / f"{session_id}{_SUFFIX}"


def register(
    session_id: str,
    *,
    port: int,
    pid: int,
    mcp_url: Optional[str] = None,
    host: str = "127.0.0.1",
    **extra,
) -> Path:
    """Publish a live session's routing record and return its path.

    ``pid`` is the *child* (port-owning) process — the liveness signal
    :func:`list_sessions` prunes on. ``port`` is the session child's http port on
    ``host`` (loopback); the control proxies ``/session/<id>/*`` there. Written
    atomically so a concurrent reader never sees a partial record. Any ``extra``
    keys are stored verbatim (forward room for e.g. an observe base path).
    """
    record = {
        "session_id": session_id,
        "host": host,
        "port": int(port),
        "pid": int(pid),
        "mcp_url": mcp_url,
        "started_at": time.time(),
        **extra,
    }
    dest = _record_path(session_id)
    # Atomic publish: write a sibling temp file, then os.replace (atomic on the
    # same filesystem) so a reader observes either the old record or the whole
    # new one, never a truncated read.
    fd, tmp = tempfile.mkstemp(
        prefix=f".{session_id}-", suffix=_SUFFIX, dir=str(dest.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.debug("Registered session %s -> %s:%s (pid %s)", session_id, host, port, pid)
    return dest


def unregister(session_id: str) -> None:
    """Remove a session's record. Best-effort and idempotent — a missing record
    (already reaped, never registered) is not an error."""
    try:
        _record_path(session_id).unlink()
    except FileNotFoundError:
        return
    except OSError as e:
        logger.debug("Could not unregister session %s: %s", session_id, e)


def read_session(session_id: str) -> Optional[dict]:
    """The record for ``session_id``, or ``None`` if absent/unreadable."""
    try:
        with open(_record_path(session_id)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_sessions(prune: bool = True) -> list[dict]:
    """All live session records, newest first (session id sorts by timestamp).

    With ``prune`` (the default) a record whose owning ``pid`` is no longer alive
    is dropped *and its file unlinked* — this is the registry's self-heal: a shim
    that died without running its reap (§6.1 gotcha 3) leaves a ghost record that
    the first reader cleans up, so the control never proxies to a dead port. A
    record missing a pid, or one we cannot decide on, is kept (fail-open, so a
    transient probe error never hides a live session).
    """
    dir_ = sessions_dir()
    try:
        paths = sorted(dir_.glob(f"*{_SUFFIX}"), reverse=True)
    except OSError:
        return []
    out: list[dict] = []
    for p in paths:
        try:
            with open(p) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            # Vanished mid-scan or corrupt; skip. A corrupt record is left in
            # place — unlinking on parse failure could race a concurrent write.
            continue
        if prune and not _pid_alive(rec.get("pid")):
            try:
                p.unlink()
            except OSError:
                pass
            logger.debug(
                "Pruned stale session record %s (pid %s)", p.name, rec.get("pid")
            )
            continue
        out.append(rec)
    return out


def _pid_alive(pid) -> bool:
    """Whether ``pid`` names a live process on this host.

    Fail-open: an unknown/malformed pid or an unexpected probe error returns
    ``True`` so :func:`list_sessions` never drops a session it cannot disprove.
    """
    if not isinstance(pid, int) or pid <= 0:
        return True  # no usable pid -> can't disprove liveness, keep the record
    if os.name == "nt":
        return _pid_alive_windows(pid)
    # NB: never signal 0 on Windows (CPython maps it to TerminateProcess — it
    # would *kill* the process); the POSIX-only branch is guarded above.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return True  # unexpected; keep the record
    return True


def _pid_alive_windows(pid: int) -> bool:
    """Windows liveness via OpenProcess + GetExitCodeProcess (stdlib ctypes).

    Avoids ``os.kill(pid, 0)``, which on Windows terminates the target. Any
    ctypes/OS hiccup falls open (returns ``True``) so a probe error never drops a
    live session.
    """
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False  # no such process (same-user localhost: not access-denied)
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return True
