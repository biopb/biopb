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

Liveness is an **identity** check, not a bare PID probe: a record stores the
child's per-process create-time token (``biopb._proc.process_create_time``), and a
reader treats the session as gone if the PID is dead *or* the PID is alive but now
names a different process (create-time mismatch → the OS recycled the PID). This
is the same PID-reuse guard as the daemon PID files (biopb#138); without it a
recycled PID would keep a ghost "alive" and, worse, could route ``/session/<id>``
traffic to an unrelated process. Delegated to ``biopb._proc`` (itself
dependency-free) so the liveness/identity computation matches the CLI's exactly.

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

from ._proc import is_process_running, process_create_time

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

    ``pid`` is the *child* (port-owning) process — the liveness/identity signal
    readers prune on. A ``create_time`` identity token for that pid is recorded
    alongside it (``None`` where the platform can't produce one, e.g. macOS) so a
    reader can tell the original child from an unrelated process that later
    inherited a recycled PID (biopb#138). ``port`` is the session child's http
    port on ``host`` (loopback); the control proxies ``/session/<id>/*`` there.
    Written atomically so a concurrent reader never sees a partial record. Any
    ``extra`` keys are stored verbatim (forward room for e.g. an observe base path).
    """
    record = {
        "session_id": session_id,
        "host": host,
        "port": int(port),
        "pid": int(pid),
        "create_time": process_create_time(int(pid)),
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


def resolve(session_id: str) -> Optional[dict]:
    """The record for a *live* ``session_id``, or ``None`` — for routers.

    Unlike :func:`read_session` this applies the liveness policy: a record whose
    owning pid is gone is treated as absent *and pruned* (so the front returns a
    clean "session ended" and the ghost is cleaned up), while a record we cannot
    disprove is returned (fail-open). This is the single entry point the control
    uses to turn a ``/session/<id>/`` request into a target.
    """
    rec = read_session(session_id)
    if rec is None:
        return None
    if not _record_is_live(rec):
        unregister(session_id)
        return None
    return rec


def list_sessions(prune: bool = True) -> list[dict]:
    """All live session records, newest first (session id sorts by timestamp).

    With ``prune`` (the default) a record whose owning process is gone — the pid
    is dead, or alive but a different process on a recycled pid — is dropped *and
    its file unlinked*. This is the registry's self-heal: a shim that died without
    running its reap (§6.1 gotcha 3) leaves a ghost record that the first reader
    cleans up, so the control never proxies to a dead (or reused) port. A record
    with no usable pid, or one we cannot decide on, is kept (fail-open, so a
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
        if prune and not _record_is_live(rec):
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


def _record_is_live(rec: dict) -> bool:
    """Whether ``rec``'s owning process is still the session child we registered.

    Liveness *and* identity (biopb#138), delegated to :mod:`biopb._proc` so the
    computation matches the daemon PID files exactly:

    - No usable pid -> ``True`` (fail-open: can't disprove, so never drop it).
    - Dead pid -> ``False``.
    - Alive pid but the recorded ``create_time`` token no longer matches the live
      one -> ``False`` (the OS recycled the pid onto an unrelated process).
    - Alive pid and either the token matches or one side is unavailable (e.g.
      macOS has no cheap source) -> ``True`` (degrade to liveness-only, never a
      false "dead").
    """
    pid = rec.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return True
    if not is_process_running(pid):
        return False
    recorded = rec.get("create_time")
    if recorded is not None:
        live = process_create_time(pid)
        if live is not None and live != recorded:
            return False  # pid reused -> different process now
    return True
