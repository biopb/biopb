"""Per-launch log files for the processes the control starts for a person.

**One file per launch**, not one shared file appended to. A shared log
interleaves concurrent launches, and lines that cannot be attributed to a
process are no use for diagnosing one that is still running. Exclusive-create
rather than a bare timestamp: two launches in the same second would otherwise
land on one name and reintroduce the interleaving in miniature.
"""

from __future__ import annotations

import logging
import os
import re
import time

from biopb._config import locations as _locations

logger = logging.getLogger(__name__)

#: How many past launches' logs to keep per prefix: enough to look back over a
#: couple of failed attempts, not an unbounded pile of Qt chatter.
KEEP = 5

# Cap on same-second name collisions before giving up on a log for this launch.
# Two launches inside one second is already a double-click; a hundred is a bug,
# and a process that starts without a log beats one that does not.
_ATTEMPTS = 100

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")


def _prune(log_dir, prefix: str) -> None:
    """Keep only the newest ``KEEP - 1`` logs of *prefix*, leaving room for the
    one about to be created. Best-effort."""
    try:
        logs = sorted(
            log_dir.glob(f"{prefix}-*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for old in logs[max(KEEP - 1, 0) :]:
        try:
            old.unlink()
        except OSError:
            pass


def open_log(prefix: str):
    """Create this launch's own logfile; return ``(handle, path)``.

    On any failure the caller still spawns, with the child's output discarded:
    ``(None, None)``.

    Binary + unbuffered, like every other owned-child log here: the fd is
    inherited by the child, which emits arbitrary bytes from native writers, so
    it must not be a text wrapper.
    """
    try:
        log_dir = _locations.mcp_viewer_log_dir()
    except OSError:
        logger.warning("No log dir; discarding %s output", prefix, exc_info=True)
        return None, None
    _prune(log_dir, prefix)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(1, _ATTEMPTS + 1):
        suffix = "" if n == 1 else f"-{n}"
        path = log_dir / f"{prefix}-{stamp}{suffix}.log"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        except OSError:
            break
        return os.fdopen(fd, "wb", buffering=0), path
    logger.warning(
        "Could not create a %s log in %s; discarding output", prefix, log_dir
    )
    return None, None


def tail(path, limit: int) -> str:
    """The last *limit* characters of this launch's own output, ANSI escapes and
    logo lines removed. The whole file *is* this launch, so no anchor is needed.
    Reads only the final bytes."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - 4 * limit, 0))
            data = f.read()
    except OSError:
        return ""
    return clean(data.decode("utf-8", "replace"))[-limit:]


def clean(text: str) -> str:
    """Output for display: no ANSI escapes, no logo lines (a line with nothing
    alphanumeric on it is box-drawing)."""
    lines = _ANSI.sub("", text).replace("\r", "").split("\n")
    kept = [ln for ln in lines if not ln.strip() or any(c.isalnum() for c in ln)]
    return "\n".join(kept).strip()
