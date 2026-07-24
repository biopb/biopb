"""``run_control`` — the blocking control process: supervise + serve control API.

Wires a :class:`DataPlaneSupervisor` to a supervision-loop thread and the stdlib
control API, then blocks until asked to stop. Stop is delivered the same way the
other biopb daemons receive it (so ``biopb control stop`` can reach it): SIGTERM /
SIGINT on POSIX, a stop-sentinel *file* on Windows (os.kill there is an
uncatchable TerminateProcess, so a graceful stop must be file-driven).

On stop it tears down the control API and, if it owns the data plane, shuts the
plane down too. A plane it *adopted* is left running -- the control did not start
it and does not assume the right to stop it.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from biopb import _credentials

from ._control import serve_control_api
from ._supervisor import DataPlaneSpec, DataPlaneSupervisor

logger = logging.getLogger(__name__)

_SUPERVISION_INTERVAL = 1.0  # seconds between liveness/restart checks


def _supervision_loop(
    supervisor: DataPlaneSupervisor, stop: threading.Event, interval: float
) -> None:
    while not stop.is_set():
        try:
            supervisor.tick()
        except Exception:  # noqa: BLE001 - a supervision tick must never die
            logger.exception("supervision tick failed")
        stop.wait(interval)


def _watch_stop_sentinel(sentinel: Path, stop: threading.Event) -> None:
    """Windows graceful-stop: set ``stop`` when the sentinel file appears.

    Mirrors the tensor server / MCP daemon shutdown-sentinel watchers. Polled on
    a daemon thread; a pre-existing sentinel (stale) is consumed on first tick so
    a leftover file from a prior run can't instantly stop a fresh control.
    """
    try:
        if sentinel.exists():
            sentinel.unlink()
    except OSError:
        pass
    while not stop.is_set():
        try:
            if sentinel.exists():
                sentinel.unlink()
                logger.info("stop sentinel observed; shutting down")
                stop.set()
                return
        except OSError:
            pass
        stop.wait(0.5)


def run_control(
    spec: DataPlaneSpec,
    *,
    control_host: str,
    control_port: int,
    data_plane: bool = True,
    ensure_timeout: float = 60.0,
    win_sentinel: Optional[Path] = None,
    log_level: str = "INFO",
) -> int:
    """Run the control plane in the foreground until stopped.

    Returns a process exit code (0 on a clean stop). ``data_plane=False`` runs an
    adopt-only control: it supervises/restarts a tensor server that is already
    running but does not spawn one.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    supervisor = DataPlaneSupervisor(spec)
    stop = threading.Event()

    try:
        server, _api_thread = serve_control_api(
            control_host, control_port, supervisor, ensure_timeout
        )
    except OSError as exc:
        logger.error(
            "Could not bind control API on %s:%d (%s). "
            "Is another control already running?",
            control_host,
            control_port,
            exc,
        )
        return 1

    # Hand the data-plane token off to local clients on the filesystem, keyed to
    # this control's lifetime (biopb/biopb#470). Written *after* the bind succeeds
    # (so a failed start that hits the `return 1` above never clobbers a running
    # control's credential) and *before* the plane comes up, since a client must
    # be able to authenticate the very `/api/data_plane/ensure` POST that starts
    # it. Only when a token is actually enforced: a tokenless local plane needs no
    # credential and the gate falls back to a loopback-`Host` check. Owner-only
    # perms are applied by the writer (0600 / an equivalent Windows DACL).
    if spec.token:
        try:
            _credentials.write_credential(spec.token)
        except OSError as exc:
            # Best-effort: a client that then can't find the credential degrades to
            # an actionable auth error, which is better than failing control startup.
            logger.warning("could not write the local credential file: %s", exc)

    if data_plane:
        logger.info("Bringing up the data plane")
        supervisor.ensure()

    loop = threading.Thread(
        target=_supervision_loop,
        args=(supervisor, stop, _SUPERVISION_INTERVAL),
        name="control-supervision",
        daemon=True,
    )
    loop.start()

    # Stop delivery. POSIX: catch SIGTERM/SIGINT. Windows: watch a sentinel file
    # (uncatchable TerminateProcess otherwise) AND still honor Ctrl-C via SIGINT.
    def _on_signal(signum, _frame):
        logger.info("signal %s received; shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)
    if sys.platform == "win32" and win_sentinel is not None:
        threading.Thread(
            target=_watch_stop_sentinel,
            args=(win_sentinel, stop),
            name="control-stop-sentinel",
            daemon=True,
        ).start()

    logger.info(
        "biopb control ready (control API on %s:%d)", control_host, control_port
    )
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        stop.set()

    logger.info("shutting down")
    server.shutdown()
    supervisor.stop()
    # Retract the credential we published so a stopped control leaves no readable
    # token behind (a crash can't run this; the next control start overwrites it).
    if spec.token:
        _credentials.remove_credential()
    return 0
