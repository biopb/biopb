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
import subprocess
import threading
from typing import Optional

from biopb._control import _agents
from biopb._lifecycle import OwnedChild
from biopb._security import credentials as _credentials

from . import _logs

logger = logging.getLogger(__name__)

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


class LaunchRefused(Exception):
    """The launch is not allowed here; carries the reason."""


def _client_env() -> dict:
    """The control's environment minus the control's own secrets: the client is
    an agent with a shell, and must not inherit the credential that gates it."""
    return _credentials.scrub_control_secrets(os.environ.copy())


def _run(argv: list) -> tuple[int, str]:
    """Run a command to completion; ``(exit code, cleaned output)``. ``-1`` when
    it could not run, with the reason as the output."""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        done = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=ONESHOT_TIMEOUT,
            env=_client_env(),
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, str(exc)
    return done.returncode, _logs.clean(done.stdout + "\n" + done.stderr)[-LOG_TAIL:]


class ClientLauncher:
    """Launch and stop clients; thread-safe, one launch per client id.

    *chat_served* is whether this control serves the ``chat`` execute root, and
    *relay_clients* whether the operator allowed clients reachable from other
    devices: a launch starts an agent with a shell, so it needs the first, and a
    relay client needs the second as well.
    """

    def __init__(self, *, chat_served: bool, relay_clients: bool):
        self._chat_served = chat_served
        self._relay_clients = relay_clients
        self._lock = threading.Lock()
        # client id -> the owned child, or None for a ``oneshot`` we ran
        self._launches: dict[str, Optional[OwnedChild]] = {}
        self._pending: set[str] = set()

    def blocked(self, agent: dict) -> Optional[str]:
        """Why *agent* (an ``_agents.status`` dict) may not be launched, or
        ``None``. The verb enforces the same answer the dashboard shows."""
        if not agent.get("can_launch"):
            return agent.get("launch_hint") or "not launchable here"
        if not self._chat_served:
            return "this control does not serve the execute roots"
        if agent.get("relay") and not self._relay_clients:
            return (
                "reachable from other devices; "
                "start the control with --allow-relay-clients"
            )
        return None

    def running_ids(self) -> set:
        """Clients this control launched that are still up (a ``oneshot`` is up
        until stopped here)."""
        with self._lock:
            self._prune_locked()
            return set(self._launches)

    def _prune_locked(self) -> None:
        for client_id, child in list(self._launches.items()):
            if child is not None and not child.alive():
                del self._launches[client_id]

    def stop_all(self) -> None:
        """Stop every owned child, for control shutdown. A ``oneshot`` service is
        left as it is."""
        with self._lock:
            children = [c for c in self._launches.values() if c is not None]
            self._launches.clear()
        for child in children:
            child.stop()

    # -- launch ------------------------------------------------------------ #

    def launch(self, client_id: str) -> dict:
        """Start *client_id*; ``{"state": started|starting|failed, ...}``.

        Only ``failed`` carries ``error``; every state carries the output so far
        as ``log`` and its file as ``log_path``. Raises :class:`LaunchRefused`.
        """
        backend = _agents.backend(client_id)
        reason = self.blocked(_agents.status(client_id))
        if reason is not None:
            raise LaunchRefused(reason)
        argv = backend.launch_argv()
        with self._lock:
            if client_id in self._pending:
                return _failed("a launch is already in progress")
            self._prune_locked()
            if client_id in self._launches:
                return {**_ok("started", "", None), "already": True}
            self._pending.add(client_id)
        try:
            if backend.launch_kind == "oneshot":
                result, child = self._run_oneshot(argv), None
            else:
                result, child = self._spawn_child(backend, client_id, argv)
            if result["state"] != "failed":
                with self._lock:
                    self._launches[client_id] = child
            return result
        finally:
            with self._lock:
                self._pending.discard(client_id)

    def _spawn_child(self, backend, client_id: str, argv: list):
        log_fh, log_path = _logs.open_log(f"client-{client_id}")
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
            return _failed(str(exc), log_path=log_path), None
        finally:
            if log_fh is not None:
                log_fh.close()  # the child holds its own dup

        def output() -> str:
            return _logs.tail(log_path, LOG_TAIL) if log_path else ""

        # A client that names itself ready is waited for; one that does not is
        # started once it has stayed up for a moment.
        ready = backend.ready_pattern
        deadline = READY_TIMEOUT if ready else GRACE
        waited = 0.0
        while waited < deadline:
            try:
                code = child.wait(_POLL)
            except subprocess.TimeoutExpired:
                code = None
            if code is not None:
                return (
                    _failed(
                        f"{client_id} exited with code {code} before it was ready",
                        log=output(),
                        log_path=log_path,
                    ),
                    None,
                )
            waited += _POLL
            if ready and ready.search(output()):
                return _ok("started", output(), log_path), child
        # Alive; a client with a readiness line that never came is only slow.
        return _ok("starting" if ready else "started", output(), log_path), child

    def _run_oneshot(self, argv: list) -> dict:
        logger.info("Running %s", " ".join(argv))
        code, output = _run(argv)
        if code != 0:
            return _failed(f"{argv[0]} exited with code {code}", log=output)
        return _ok("started", output, None)

    # -- stop -------------------------------------------------------------- #

    def stop(self, client_id: str) -> dict:
        """Stop a client this control launched; ``{"stopped": bool, ...}``.

        Refuses one it did not launch: a handle, not a pid, is what makes
        stopping safe, and a user's own client is theirs.
        """
        backend = _agents.backend(client_id)
        with self._lock:
            if client_id not in self._launches:
                return {"stopped": False, "error": "not started from this control"}
            child = self._launches.pop(client_id)
        if child is not None:
            child.stop()
            return {"stopped": True}
        argv = backend.stop_argv()
        if not argv:
            return {"stopped": True, "note": "nothing to stop; its service stays up"}
        code, output = _run(argv)
        if code != 0:
            return {"stopped": False, "error": output or f"exit {code}"}
        return {"stopped": True, "log": output}


def _ok(state: str, log: str, log_path) -> dict:
    return {"state": state, "log": log, "log_path": str(log_path) if log_path else None}


def _failed(error: str, *, log: str = "", log_path=None) -> dict:
    return {**_ok("failed", log, log_path), "error": error}
