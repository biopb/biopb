"""Client for the biopb control (control plane) control API — stdlib only.

Since the de-daemonization (ARCHITECTURE.md, Lifecycle), ``_connection`` is a
*pure client*: it
never shells out ``biopb server start`` to bring the data plane up. When the
plane is down it asks the control to ensure it, via this thin urllib client. The
control is the durable root that owns the data plane; ``_connection`` only uses it.

The endpoint is resolved from ``biopb._config_control`` (the shared, stdlib-only
core-SDK module), the same location the control itself binds — biopb-mcp cannot
import ``biopb-control`` any more than it can import ``biopb-tensor-server``, so the
one shared fact (where the control listens) lives in core ``biopb``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def _base_url() -> str:
    from biopb._config_control import control_base_url

    return control_base_url()


def control_reachable(timeout: float = 1.0) -> bool:
    """Whether the control's control API answers ``GET /health``."""
    try:
        with urllib.request.urlopen(f"{_base_url()}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def data_plane_url(timeout: float = 1.0) -> str | None:
    """The data-plane gRPC URL the control owns, or ``None`` if no control answers.

    GETs the control's bare, unauthenticated ``/health`` and reads
    ``data_plane.grpc_url`` from the supervisor snapshot. The control resolves
    that endpoint from the tensor-server ``[server]`` config, so it is the single
    source of truth for *where the data plane lives* (#413) -- the model in which
    the admin owns the data plane. ``None`` when the control is unreachable (not
    running) so the caller falls back to its own local config; this is a plain
    read (unlike ``ensure_data_plane`` it never spawns the plane).
    """
    try:
        with urllib.request.urlopen(f"{_base_url()}/health", timeout=timeout) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - best-effort; caller falls back to config
        logger.debug("control data_plane_url probe failed: %s", exc)
        return None
    data_plane = payload.get("data_plane")
    if isinstance(data_plane, dict):
        url = data_plane.get("grpc_url")
        if isinstance(url, str) and url:
            return url
    return None


def ensure_data_plane(timeout: float = 60.0) -> dict | None:
    """Ask the control to ensure the data plane is up; return its snapshot.

    POSTs ``/api/data_plane/ensure`` — idempotent on the control side (spawn the
    plane it owns, then wait until listening). Returns the ``data_plane`` snapshot dict
    on success, or ``None`` if the control is unreachable (not running) or errored,
    so the caller can fall back to surfacing "start the control" rather than
    raising.

    ``timeout`` is BOTH our HTTP timeout and the hint we pass the server as
    ``?client_timeout``: the server caps its own ensure wait below this so it
    always returns a verdict before our ``urlopen`` times out — otherwise a
    slow-but-working control plane would look unreachable and we'd wrongly report
    "no control plane".
    """
    from urllib.parse import urlencode

    url = (
        f"{_base_url()}/api/data_plane/ensure?{urlencode({'client_timeout': timeout})}"
    )
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("data_plane")
    except Exception as exc:  # noqa: BLE001 - best-effort; caller handles None
        logger.info("control ensure_data_plane failed: %s", exc)
        return None


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
    needs the data plane, :func:`ensure_data_plane` returns ``None`` and
    ``_connection`` surfaces the actionable "Run ``biopb control start``" status --
    the mcp server, not the shim, is where a control-interaction failure belongs.

    The launched process is detached from the caller's process group / console, so
    it (and the durable control it spawns) survives a client that disconnects during
    the first seconds. Idempotent: ``biopb control start`` no-ops when a control is
    already running and serializes concurrent starts (``biopb._filelock``), so racing
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
