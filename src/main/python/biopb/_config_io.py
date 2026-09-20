"""Atomic JSON config writes, shared by every biopb package.

Every config surface writes the same way -- serialize, write a sibling temp file,
``os.replace`` over the target -- so a reader never sees a half-written config and
a failed write leaves the previous file intact. The two packages had a copy each;
they differed only in what they did with an error, which is now a parameter.

Stdlib-only, like its ``_config_*`` siblings.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _atomic_write(
    path: Path, write_into: Any, *, raise_on_error: bool, what: str, wrote: str
) -> None:
    """Write to *path* via a sibling temp file plus ``os.replace``.

    *write_into* does the actual write, given the temp path to write into.
    *raise_on_error* is the one real difference between the callers: an admin
    endpoint must surface a permission/disk error to the user who clicked save,
    while a best-effort settings write from a running session logs and carries on
    rather than taking the session down with it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process *and* thread, so two concurrent writers never collide on
    # the temp file (the MCP kernel writes settings from background threads).
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        write_into(tmp)
        os.replace(tmp, path)
        logger.debug("%s %s", wrote, path)
    except Exception as e:  # noqa: BLE001 - re-raised or logged per the caller's policy
        try:
            tmp.unlink()
        except OSError:
            pass
        if raise_on_error:
            raise
        logger.warning("Failed to %s %s: %s", what, path, e)


def atomic_write_json(
    path: Path, data: Dict[str, Any], *, raise_on_error: bool
) -> None:
    """Write *data* to *path* as pretty JSON, atomically. See :func:`_atomic_write`."""

    def write_into(tmp: Path) -> None:
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    _atomic_write(
        path,
        write_into,
        raise_on_error=raise_on_error,
        what="save config to",
        wrote="Wrote config to",
    )


def atomic_write_text(path: Path, text: str, *, raise_on_error: bool) -> None:
    """Write *text* to *path* atomically. See :func:`_atomic_write`."""
    _atomic_write(
        path,
        lambda tmp: tmp.write_text(text, encoding="utf-8"),
        raise_on_error=raise_on_error,
        what="write",
        wrote="Wrote",
    )
