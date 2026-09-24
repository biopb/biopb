"""Start the biopb control for the stdio shim.

Asking a running control anything is :mod:`biopb.control`'s job. This is the
one thing that is not: launching one, which only the shim does, because it is
the one entry point with no human to ask.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _biopb_executable() -> str | None:
    """Locate the core ``biopb`` CLI executable, or ``None`` if not found.

    Prefer the console script installed alongside this interpreter (the venv /
    uv-tool ``Scripts``/``bin`` dir, where ``biopb = biopb.cli:app`` lands), so we
    hit the same environment that installed biopb-mcp even when PATH is not
    inherited (GUI agents launch us without a shell PATH). Fall back to PATH.
    ``None`` when neither resolves -- the caller then skips the best-effort control
    start and the session child surfaces the error on first data-plane use.
    """
    import shutil

    name = "biopb.exe" if os.name == "nt" else "biopb"
    # Do NOT resolve() sys.executable: a venv's `python` is a symlink to the base
    # interpreter, so resolving would follow it OUT of the venv bin/ (where the
    # console script actually lives) to the base dir, and the sibling lookup would
    # miss -- exactly the symlinked-venv + no-PATH case this is meant to cover.
    sibling = Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    return shutil.which("biopb")


def start_control_detached() -> bool:
    """Best-effort, non-blocking launch of ``biopb control start --no-data-plane``.

    Returns whether the launch was *issued* -- never whether the control is up.

    Why fire-and-forget rather than a blocking ensure: the stdio shim must get its
    bridge ready within the MCP client's initialize timeout, so it cannot pause to
    verify the control -- lean as it is -- is fully listening. We fire the start and
    return immediately; the control boots in the background, in parallel with the
    session child's own (import-dominated) startup, which normally more than covers
    the control's boot. If the control still isn't reachable when the child first
    needs the data plane, :func:`biopb.control.ensure_data_plane` returns ``None``
    and the connection surfaces the actionable "Run ``biopb control start``" status --
    the mcp server, not the shim, is where a control-interaction failure belongs.

    The launched process is detached from the caller's process group / console, so
    it (and the durable control it spawns) survives a client that disconnects during
    the first seconds. Idempotent: ``biopb control start`` no-ops when a control is
    already running and serializes concurrent starts (``biopb._lifecycle.file_lock``), so racing
    shims are safe. ``--no-data-plane`` keeps the footprint minimal -- the data plane
    comes up on demand when a session actually asks for it.
    """
    exe = _biopb_executable()
    if exe is None:
        logger.info("biopb CLI not found; not auto-starting the control plane")
        return False
    argv = [exe, "control", "start", "--no-data-plane"]
    kwargs: dict = {}
    if os.name == "nt":
        # No console window, own process group -> not reaped with the shim.
        kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        kwargs["start_new_session"] = True  # detach from the shim's process group
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **kwargs,
        )
    except OSError as exc:
        logger.info("could not launch `biopb control start`: %s", exc)
        return False
    # Reap the short-lived launcher (it exits within ~15s, after spawning the
    # durable detached control) off a daemon thread, so we neither block here nor
    # leave a zombie for the shim's possibly hours-long lifetime. Windows has no
    # POSIX zombies; the durable control is unaffected either way.
    if os.name != "nt":
        threading.Thread(target=proc.wait, daemon=True).start()
    logger.info("launched `biopb control start --no-data-plane` (detached)")
    return True
