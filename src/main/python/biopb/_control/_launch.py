"""Start the biopb control for the shim.

Asking a running control anything is the core ``biopb`` SDK's job (its
top-level ``ensure_data_plane``/``algorithms``/etc., backed by the private
``biopb._control``). This is the one thing that is not: launching one, which
only the shim does, because it is the one entry point with no human to ask.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading

from . import _agents
from ._client import base_url
from ._data_plane import resolve_data_plane_token

logger = logging.getLogger(__name__)


def _biopb_executable() -> str | None:
    """The core ``biopb`` CLI executable, or ``None`` if not found.

    ``None`` makes the caller skip the best-effort control start; the session
    then surfaces the error on first data-plane use.
    """
    return _agents.console_script("biopb")


def start_control_detached() -> bool:
    """Best-effort, non-blocking launch of ``biopb control start --no-data-plane``.

    Returns whether the launch was *issued* -- never whether the control is up.

    Fire-and-forget because the shim must answer the MCP initialize within its
    timeout. If the control is still down when the child first needs the data
    plane, :func:`biopb._control.ensure_data_plane` returns ``None`` and the connection
    reports it.

    The process is detached, so it survives a client that disconnects early.
    Idempotent: ``biopb control start`` no-ops when a control is running and
    serializes concurrent starts. The data plane comes up on demand.
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
    # Reap the short-lived launcher off a daemon thread (no zombie; Windows has none).
    if os.name != "nt":
        threading.Thread(target=proc.wait, daemon=True).start()
    logger.info("launched `biopb control start --no-data-plane` (detached)")
    return True


def _control_url() -> str:
    return base_url()


def control_up(timeout: float = 1.0) -> bool:
    """Whether a control answers ``/health`` on the address clients use."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_control_url()}/health", timeout=timeout):
            return True
    except OSError:
        return False


def ensure_control(wait: float) -> bool:
    """A control that answers, started here if need be; False if none does
    within *wait* seconds.
    """
    import time

    if control_up():
        return True
    if not start_control_detached():
        return False
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if control_up():
            return True
    return False


def launch_session(
    *, display: dict, timeout: float, start_kernel: bool = False
) -> dict:
    """Ask the control to launch a session; its answer (``{"state", ...}``).

    *display* is the client's own display environment, which the control uses
    in place of its own (see ``biopb_control``'s ``_launch_env``). The kernel is
    left for the agent's ``start_kernel``. ``OSError`` (an ``HTTPError``
    included) when no control answers or it refuses.
    """
    import json
    import urllib.request
    from urllib.parse import urlencode

    params = {
        "start_kernel": int(start_kernel),
        "display": json.dumps(display),
        # Keeps the control's own wait under ours, so a slow start returns a verdict.
        "client_timeout": timeout,
    }
    token = resolve_data_plane_token()
    req = urllib.request.Request(
        f"{_control_url()}/api/sessions/new?{urlencode(params)}",
        data=b"",
        method="POST",
        headers={"X-Biopb-Token": token} if token else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())
