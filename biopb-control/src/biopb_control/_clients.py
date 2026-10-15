"""Launching an MCP client's web or remote session from the dashboard.

The control holds each client it starts as a Pattern-O ``OwnedChild`` (it dies
with the control) and never holds a biopb session: the client's ``biopb-shim``
takes a session's lease when its agent attaches. A client whose launch is a
command that returns (``oneshot``: a machine-global service such as the Codex
daemon or opencode 2's) is run to completion and remembered, since it is not
ours to kill.

One launch per client at a time. Output goes to a per-launch log; the tail is
returned on success as well as failure because it carries what the user needs
(opencode's URL, a pairing link).
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import subprocess
import threading
import time

from biopb._config import locations as _locations
from biopb._control import _agents
from biopb._lifecycle import OwnedChild
from biopb._security import credentials as _credentials

logger = logging.getLogger(__name__)

#: Newest client logs kept per client.
LOG_KEEP = 5
#: Bytes of output returned to the dashboard.
LOG_TAIL = 4000
#: Seconds a ``child`` launch waits for its readiness line, before reporting
#: ``starting`` rather than failing a client that is only slow.
READY_TIMEOUT = 15.0
#: Seconds a ``child`` with no readiness line must stay alive to count as started.
GRACE = 3.0
#: Seconds a ``oneshot`` command may run.
ONESHOT_TIMEOUT = 60.0
_POLL = 0.2

#: What a ``child`` prints once it is usable. opencode names its URL; the relay
#: clients print nothing that is a reliable signal.
_READY = {"opencode": re.compile(r"https?://\S+")}

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")


def clean(text: str) -> str:
    """Output for display: no ANSI escapes, no logo lines (a line with nothing
    alphanumeric on it is box-drawing)."""
    lines = _ANSI.sub("", text).replace("\r", "").split("\n")
    kept = [ln for ln in lines if not ln.strip() or any(c.isalnum() for c in ln)]
    return "\n".join(kept).strip()


def _client_env() -> dict:
    """The control's environment minus the control's own secrets: the client is
    an agent with a shell, and must not inherit the credential that gates it."""
    env = dict(os.environ)
    _credentials.scrub_control_secrets(env)
    return env


def _open_log(client_id: str):
    """``(handle, path)`` for this launch's log, or ``(None, None)``: a client
    that starts without a log beats one that does not start."""
    try:
        log_dir = _locations.mcp_viewer_log_dir()
        stem = f"client-{client_id}-"
        old = sorted(
            log_dir.glob(stem + "*.log"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for path in old[max(LOG_KEEP - 1, 0) :]:
            path.unlink(missing_ok=True)
        path = (
            log_dir
            / f"{stem}{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}.log"
        )
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        return os.fdopen(fd, "wb", buffering=0), path
    except OSError:
        logger.warning("No log for the %s launch; discarding output", client_id)
        return None, None


def _tail(path) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return clean(data.decode("utf-8", "replace"))[-LOG_TAIL:]


class _Launch:
    """What the control holds of one launched client."""

    def __init__(self, kind, child=None, log_path=None):
        self.kind = kind  # "child" | "oneshot"
        self.child = child
        self.log_path = log_path


class ClientLauncher:
    """Launch and stop clients; thread-safe, one launch per client id."""

    def __init__(self):
        self._lock = threading.Lock()
        self._launches: dict[str, _Launch] = {}

    # -- state ------------------------------------------------------------- #

    def running(self, client_id: str) -> bool:
        """Whether a client this control launched is still up (a ``oneshot`` is
        up until stopped here)."""
        with self._lock:
            return self._running_locked(client_id)

    def _running_locked(self, client_id: str) -> bool:
        launch = self._launches.get(client_id)
        if launch is None:
            return False
        if launch.kind == "child" and not launch.child.alive():
            del self._launches[client_id]
            return False
        return True

    def stop_all(self) -> None:
        """Stop every owned child, for control shutdown. A ``oneshot`` service is
        left as it is."""
        with self._lock:
            launches = list(self._launches.values())
            self._launches.clear()
        for launch in launches:
            if launch.child is not None:
                launch.child.stop()

    # -- launch ------------------------------------------------------------ #

    def launch(self, client_id: str) -> dict:
        """Start *client_id*; ``{"state": started|starting|failed, ...}``.

        Only ``failed`` carries ``error``; every state carries the output so far
        as ``log`` and its file as ``log_path``.
        """
        backend = _agents.backend(client_id)
        argv = backend.launch_argv()
        if not argv or not backend.can_launch():
            return _failed("this client cannot be launched here")
        with self._lock:
            if self._running_locked(client_id):
                return {
                    "state": "started",
                    "already": True,
                    "log": "",
                    "log_path": None,
                }
            if backend.launch_kind == "oneshot":
                return self._run_oneshot(client_id, argv)
            return self._spawn_child(client_id, argv)

    def _spawn_child(self, client_id: str, argv: list) -> dict:
        log_fh, log_path = _open_log(client_id)
        logger.info("Launching %s: %s (log: %s)", client_id, " ".join(argv), log_path)
        child = OwnedChild(
            argv,
            log=log_fh if log_fh is not None else subprocess.DEVNULL,
            env=_client_env(),
        )
        try:
            child.spawn()
        except OSError as exc:
            logger.exception("Could not launch %s", client_id)
            return _failed(str(exc), log_path=log_path)
        finally:
            if log_fh is not None:
                log_fh.close()  # the child holds its own dup
        ready = _READY.get(client_id)
        start = time.monotonic()
        while True:
            code = child.poll()
            output = _tail(log_path) if log_path else ""
            if code is not None:
                return _failed(
                    f"{client_id} exited with code {code} before it was ready",
                    log=output,
                    log_path=log_path,
                )
            elapsed = time.monotonic() - start
            if (ready and ready.search(output)) or (not ready and elapsed >= GRACE):
                self._launches[client_id] = _Launch("child", child, log_path)
                return _ok("started", output, log_path)
            if elapsed >= READY_TIMEOUT:
                # Alive, just quiet: keep it and say so, as a slow session is.
                self._launches[client_id] = _Launch("child", child, log_path)
                return _ok("starting", output, log_path)
            time.sleep(_POLL)

    def _run_oneshot(self, client_id: str, argv: list) -> dict:
        logger.info("Running %s: %s", client_id, " ".join(argv))
        try:
            done = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=ONESHOT_TIMEOUT,
                env=_client_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _failed(str(exc))
        output = clean(done.stdout + "\n" + done.stderr)[-LOG_TAIL:]
        if done.returncode != 0:
            return _failed(f"{argv[0]} exited with code {done.returncode}", log=output)
        self._launches[client_id] = _Launch("oneshot")
        return _ok("started", output, None)

    # -- stop -------------------------------------------------------------- #

    def stop(self, client_id: str) -> dict:
        """Stop a client this control launched; ``{"stopped": bool, ...}``.

        Refuses one it did not launch: a handle, not a pid, is what makes
        stopping safe, and a user's own client is theirs.
        """
        backend = _agents.backend(client_id)
        with self._lock:
            launch = self._launches.get(client_id)
            if launch is None:
                return {"stopped": False, "error": "not started from this control"}
            del self._launches[client_id]
        if launch.kind == "child":
            launch.child.stop()
            return {"stopped": True}
        argv = backend.stop_argv()
        if not argv:
            return {"stopped": True, "note": "nothing to stop; its service stays up"}
        try:
            done = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=ONESHOT_TIMEOUT,
                env=_client_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"stopped": False, "error": str(exc)}
        output = clean(done.stdout + "\n" + done.stderr)[-LOG_TAIL:]
        if done.returncode != 0:
            return {"stopped": False, "error": output or f"exit {done.returncode}"}
        return {"stopped": True, "log": output}


def _ok(state: str, log: str, log_path) -> dict:
    return {"state": state, "log": log, "log_path": str(log_path) if log_path else None}


def _failed(error: str, *, log: str = "", log_path=None) -> dict:
    return {
        "state": "failed",
        "error": error,
        "log": log,
        "log_path": str(log_path) if log_path else None,
    }
