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

    Why fire-and-forget rather than a blocking ensure: the stdio shim must get its
    bridge ready within the MCP client's initialize timeout, so it cannot pause to
    verify the control -- lean as it is -- is fully listening. We fire the start and
    return immediately; the control boots in the background, in parallel with the
    session child's own (import-dominated) startup, which normally more than covers
    the control's boot. If the control still isn't reachable when the child first
    needs the data plane, :func:`biopb.ensure_data_plane` returns ``None``
    and the connection surfaces the actionable "Run ``biopb control start``" status --
    the mcp server, not the shim, is where a control-interaction failure belongs.

    The launched process is detached from the caller's process group / console, so
    it (and the durable control it spawns) survives a client that disconnects during
    the first seconds. Idempotent: ``biopb control start`` no-ops when a control is
    already running and serializes concurrent starts (``biopb.lifecycle.file_lock``), so racing
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


def _control_url() -> str:
    import biopb

    return biopb.base_url()


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

    The one place the shim blocks on the control: asking it for a session is
    only possible once it is up. Callers fall back to a session of their own
    rather than fail, so a missing control costs a wait and no more.
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

    import biopb

    params = {
        "start_kernel": int(start_kernel),
        "display": json.dumps(display),
        # Bounds the control's own wait under ours, so a slow start comes back
        # as a verdict and not as a timeout that looks like no control.
        "client_timeout": timeout,
    }
    token = biopb.resolve_data_plane_token()
    req = urllib.request.Request(
        f"{_control_url()}/api/sessions/new?{urlencode(params)}",
        data=b"",
        method="POST",
        headers={"X-Biopb-Token": token} if token else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())
