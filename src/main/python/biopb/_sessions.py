"""Filesystem registry of live MCP sessions (shared, stdlib-only).

Sessions publish ``session id -> port + pid`` records here; the control plane
reads them to list sessions (``/api/sessions``) and proxy ``/session/<id>/*``.
It lives in the core SDK because neither side can import the other.

Readers prune records whose process is gone, so a hard-killed session leaves no
routing ghost. Liveness is an identity check: the record stores the process
create-time token (``biopb._lifecycle.proc.process_create_time``), so a recycled
PID is not mistaken for the session.

Writes are atomic (temp file + ``os.replace``); a record unlinked between listing
and reading is skipped.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from . import _locations
from ._lifecycle.proc import is_process_running, process_create_time

logger = logging.getLogger(__name__)

# Record filename suffix; the stem is the session id.
_SUFFIX = ".json"


def sessions_dir() -> Path:
    """The registry directory (``BIOPB_SESSIONS_DIR`` override, else the biopb
    state tree), created on access by :func:`biopb._locations.sessions_dir`."""
    return _locations.sessions_dir()


# Characters that would let a session id escape the registry dir. Both platforms'
# separators are rejected regardless of host OS, plus ``:`` (Windows drive/ADS)
# and NUL.
_UNSAFE_ID_CHARS = frozenset({"/", "\\", ":", "\x00"})


def new_session_id() -> str:
    """Mint a session id, ``<timestamp>-<pid>``, of the minting process.

    The pid a reader prunes on is the one passed to :func:`register`, not this.
    """
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def _is_safe_session_id(session_id: str) -> bool:
    """Whether ``session_id`` is a single, non-traversing path component.

    Ids reach here straight from ``/session/<id>/...`` URLs, so this module
    sanitizes them itself (Starlette's convertor does not block ``\\``).
    """
    return (
        bool(session_id)
        and session_id not in (".", "..")
        and not _UNSAFE_ID_CHARS.intersection(session_id)
        and os.sep not in session_id
        and (os.altsep is None or os.altsep not in session_id)
    )


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

    ``pid`` is the port-owning child readers prune on; its ``create_time`` token
    is recorded too (``None`` where unavailable, e.g. macOS). ``port`` is the
    child's http port on ``host``. ``extra`` keys are stored verbatim.

    Raises ``ValueError`` for an unsafe ``session_id``.
    """
    if not _is_safe_session_id(session_id):
        raise ValueError(f"unsafe session id: {session_id!r}")
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
    fd, tmp = tempfile.mkstemp(
        prefix=f".{session_id}-", suffix=_SUFFIX, dir=str(dest.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
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
    """Remove a session's record; best-effort, idempotent, unsafe ids ignored."""
    if not _is_safe_session_id(session_id):
        return
    try:
        _record_path(session_id).unlink()
    except FileNotFoundError:
        return
    except OSError as e:
        logger.debug("Could not unregister session %s: %s", session_id, e)


def read_session(session_id: str) -> Optional[dict]:
    """The record for ``session_id``, or ``None`` if absent/unreadable/unsafe."""
    if not _is_safe_session_id(session_id):
        return None
    try:
        with open(_record_path(session_id), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def resolve(session_id: str) -> Optional[dict]:
    """The record for a *live* ``session_id``, or ``None`` (for routers).

    Unlike :func:`read_session`, a record whose process is gone is pruned and
    treated as absent; one we cannot disprove is returned (fail-open).
    """
    rec = read_session(session_id)
    if rec is None:
        return None
    if not _record_is_live(rec):
        unregister(session_id)
        return None
    return rec


def list_sessions(prune: bool = True) -> list[dict]:
    """All live session records, newest first (by ``started_at``).

    With ``prune`` a record whose process is gone (dead, or pid recycled) is
    dropped and its file unlinked. Records we cannot decide on are kept.
    """
    dir_ = sessions_dir()
    try:
        paths = list(dir_.glob(f"*{_SUFFIX}"))
    except OSError:
        return []
    out: list[dict] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
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
    # Not by filename: un-padded pids mis-order sessions within one second.
    # A record missing ``started_at`` sorts last.
    out.sort(key=lambda rec: rec.get("started_at") or 0.0, reverse=True)
    return out


def _record_is_live(rec: dict) -> bool:
    """Whether ``rec``'s process is still the session child we registered.

    No usable pid -> True (fail-open); dead -> False; alive with a mismatched
    ``create_time`` (recycled pid) -> False; alive with a match or a missing
    token -> True.
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
