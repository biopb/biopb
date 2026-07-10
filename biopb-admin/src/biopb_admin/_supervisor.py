"""Supervise the data (tensor) plane as a subprocess — never import it.

:class:`DataPlaneSupervisor` owns the tensor-server process the way the existing
``biopb server start`` command does (same ``python -m biopb_tensor_server.cli
launch`` argv, same detach/log idiom), but *persistently*: it starts the plane,
polls its liveness, and restarts it on crash with capped backoff. It **adopts**
an already-running server instead of double-binding (the transition case in the
migration doc), and it never imports ``biopb_tensor_server`` — liveness is a
cheap stdlib TCP connect to the gRPC port, so no pyarrow/grpc is pulled into the
lean admin (invariant I2).

Readiness beyond "port bound" (the progressive-discovery ``SERVING`` scan) is
left to the *client*: ``biopb-mcp``'s ``_connection`` connects and waits the
server through its data-folder scan itself. The supervisor's job ends at "the
process is up and listening".
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Restart backoff (seconds), indexed by consecutive-failure count and clamped to
# the last entry. First (re)start is immediate; a persistently crash-looping
# plane settles at one attempt / 30s rather than hammering.
_BACKOFF_SCHEDULE = (0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0)

# A child that stays up this long is deemed recovered: the failure counter (and
# thus the backoff) resets, so an occasional crash hours apart never inherits a
# stale long backoff from a burst earlier in the day.
_HEALTHY_RESET_SECONDS = 60.0


@dataclass
class DataPlaneSpec:
    """Everything needed to launch + probe the tensor server, resolved by the
    caller (the ``biopb admin`` CLI) so the supervisor imports no server config.

    ``grpc_host`` / ``grpc_port`` are the loopback-reachable endpoint the
    liveness probe connects to (a server bound to 0.0.0.0/:: is reached over
    127.0.0.1). ``token`` / ``local_bypass`` carry the same token policy
    ``biopb server start`` applies, precomputed by the caller.
    """

    config: Path
    grpc_host: str = "127.0.0.1"
    grpc_port: int = 8815
    web_host: str = "127.0.0.1"
    web_port: int = 8814
    static_dir: Optional[Path] = None
    log_level: str = "INFO"
    server_log: Optional[Path] = None
    token: Optional[str] = None
    local_bypass: bool = False


@dataclass
class _State:
    want: bool = False  # should the plane be running (set by ensure/stop)
    owned: bool = False  # did we spawn it (vs adopt an already-running one)
    failures: int = 0  # consecutive failed/crashed attempts (drives backoff)
    restarts: int = 0  # total restarts since admin start (for status)
    next_attempt_at: float = 0.0  # monotonic; earliest time to (re)spawn
    up_since: Optional[float] = None  # monotonic; when we last observed it up
    last_error: Optional[str] = None
    last_exit_code: Optional[int] = None  # exit code of the most recent crash


class DataPlaneSupervisor:
    """Persistent supervisor for the tensor (data) plane subprocess."""

    def __init__(self, spec: DataPlaneSpec):
        self._spec = spec
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self._state = _State()
        self._lock = threading.RLock()

    # --- liveness / argv / env ------------------------------------------- #

    def _port_up(self, timeout: float = 0.5) -> bool:
        try:
            with socket.create_connection(
                (self._spec.grpc_host, self._spec.grpc_port), timeout=timeout
            ):
                return True
        except OSError:
            return False

    def _build_argv(self) -> list[str]:
        s = self._spec
        argv = [
            sys.executable,
            "-m",
            "biopb_tensor_server.cli",
            "launch",
            "--config",
            str(s.config),
            "--web-port",
            str(s.web_port),
            "--web-host",
            str(s.web_host),
            "--log-level",
            str(s.log_level),
        ]
        if s.static_dir and Path(s.static_dir).exists():
            argv += ["--static-dir", str(s.static_dir)]
        return argv

    def _child_env(self) -> dict:
        env = os.environ.copy()
        if self._spec.token:
            env["BIOPB_TENSOR_TOKEN"] = self._spec.token
        elif self._spec.local_bypass:
            # No token, all-localhost: tell the server to skip token enforcement
            # without prompting -- matches `biopb server start`'s local-only path.
            env["BIOPB_WEB_DEV_BYPASS"] = "1"
        return env

    def _open_log(self):
        """Open (once) the append-binary file the child's stdout/stderr go to.

        Binary + unbuffered for the same reason the CLI's server log is: the
        child and its native libraries (gRPC, Arrow) emit arbitrary bytes. Falls
        back to the admin's own stderr if the file can't be opened, so a bad log
        path never blocks the plane from starting.
        """
        if self._log_fh is not None:
            return self._log_fh
        path = self._spec.server_log
        if path is None:
            self._log_fh = getattr(sys.stderr, "buffer", sys.stderr)
            return self._log_fh
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(path, "ab", buffering=0)
        except OSError:
            logger.warning("Cannot open data-plane log %s; using stderr", path)
            self._log_fh = getattr(sys.stderr, "buffer", sys.stderr)
        return self._log_fh

    # --- lifecycle ------------------------------------------------------- #

    def _spawn_locked(self) -> None:
        argv = self._build_argv()
        log = self._open_log()
        try:
            log.write(
                f"\n--- admin: starting data plane at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode()
            )
        except (OSError, ValueError):
            pass
        logger.info("Spawning data plane: %s", " ".join(argv))
        # No detachment: the plane is a *tracked* child of the admin -- if the
        # admin dies the plane stays serving (already-spawned planes survive an
        # admin blip, migration doc S3), but while the admin lives it owns and
        # can reap this child directly.
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=self._child_env(),
            close_fds=True,
        )
        self._state.owned = True
        self._state.up_since = None

    def _child_alive(self) -> bool:
        """Whether we hold a spawned child that is still running.

        poll() reaps a just-exited child (no zombie) and caches its returncode,
        so a non-None poll means gone.
        """
        return self._proc is not None and self._proc.poll() is None

    def _reap_dead_child(self) -> Optional[int]:
        """Drop the handle to our spawned child if it has exited; return its exit
        code (else ``None``).

        Clearing ``self._proc`` to ``None`` the moment the child dies is what
        keeps it an honest signal: every ``self._proc is None`` test then means
        "no live child of ours", so the adopt-vs-restart decisions that key on it
        can't be fooled by a crashed child's ``Popen`` left lying around (the
        fragility this guards against). poll() has already reaped the OS zombie.
        """
        if self._proc is None or self._proc.poll() is None:
            return None
        rc = self._proc.returncode
        self._proc = None
        return rc

    def _note_healthy(self) -> None:
        """Record that the plane is up; after a sustained healthy run, clear the
        failure count so an earlier crash burst stops inflating the backoff."""
        st = self._state
        now = time.monotonic()
        if st.up_since is None:
            st.up_since = now
        elif now - st.up_since >= _HEALTHY_RESET_SECONDS:
            st.failures = 0

    def ensure(self) -> None:
        """Idempotently make the data plane running. Spawn it, or adopt one that
        is already listening. Safe to call repeatedly / concurrently."""
        with self._lock:
            st = self._state
            st.want = True
            st.last_error = None
            self._reap_dead_child()  # normalize a stale handle before deciding
            if self._proc is not None:
                return  # our child is alive (starting or serving)
            if self._port_up():
                # Something is already serving on the port we'd bind. Adopt it
                # rather than double-bind; we monitor but do not own its death.
                st.owned = False
                return
            self._spawn_locked()
            st.next_attempt_at = time.monotonic() + self._backoff()

    def _backoff(self) -> float:
        idx = min(self._state.failures, len(_BACKOFF_SCHEDULE) - 1)
        return _BACKOFF_SCHEDULE[idx]

    def tick(self) -> None:
        """One supervision step: restart a crashed owned child with backoff.

        Called on a fixed cadence by the supervision loop. Adopted (unowned)
        planes are only monitored, never restarted -- we did not spawn them and
        another manager may.
        """
        with self._lock:
            st = self._state
            if not st.want:
                return

            # Reap a crashed child up front so self._proc never lingers as a dead
            # handle, and count the crash exactly once -- at the tick we observe
            # it, independent of whether the backoff lets us respawn this tick.
            rc = self._reap_dead_child()
            if rc is not None:
                st.restarts += 1
                st.failures += 1
                st.last_error = f"data plane exited (code {rc}); restarting"
                logger.warning(st.last_error)

            if self._proc is not None:
                # A live child of ours (still binding its port, or serving).
                if self._port_up():
                    self._note_healthy()
                return

            if self._port_up():
                # The port is served, but not by a child of ours: an adopted or
                # external server. Monitor it; never restart or kill what we did
                # not spawn.
                st.owned = False
                self._note_healthy()
                return

            # Port down and no live child of ours: (re)start when backoff allows.
            st.up_since = None
            now = time.monotonic()
            if now < st.next_attempt_at:
                return
            if not st.owned:
                # A previously-adopted server vanished; the admin is its
                # supervisor now (a fresh spawn always sets owned via ensure).
                logger.info("data plane is down and unowned; taking it over")
            self._spawn_locked()
            st.next_attempt_at = now + self._backoff()

    def wait_until_up(self, timeout: float) -> bool:
        """Block until the plane's port is listening, or ``timeout`` elapses.

        Advances the supervision loop's restart logic itself (via :meth:`tick`)
        so a child that crashes on boot is retried within the wait, not just
        polled. Returns whether the port came up.
        """
        deadline = time.monotonic() + timeout
        while True:
            self.tick()
            if self._port_up():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

    def stop(self) -> None:
        """Stop supervising and, if we own the plane, shut it down."""
        with self._lock:
            self._state.want = False
            proc, owned = self._proc, self._state.owned
            self._proc = None
        if proc is not None and owned:
            self._terminate(proc)
        self._close_log()

    def _terminate(self, proc: subprocess.Popen, timeout: float = 10.0) -> None:
        if proc.poll() is not None:
            return
        sentinel = self._win_stop_sentinel()
        if sys.platform == "win32":
            # os.kill/terminate is an uncatchable TerminateProcess on Windows, so
            # the server never runs its shutdown handler. It instead watches for a
            # stop-sentinel file (http_server._install_windows_shutdown_listener);
            # drop it for a graceful stop, then hard-kill as a backstop.
            try:
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text("stop")
            except OSError:
                pass
        else:
            proc.terminate()  # SIGTERM; the server's handler catches it
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        if sys.platform == "win32":
            try:
                sentinel.unlink()
            except OSError:
                pass

    @staticmethod
    def _win_stop_sentinel() -> Path:
        # Must match _win_shutdown_sentinel() in biopb.cli /
        # http_server.shutdown_sentinel_path(): a single fixed name under the
        # biopb data dir, not keyed by PID.
        return Path.home() / ".local" / "share" / "biopb" / "tensor-server.stop"

    def _close_log(self) -> None:
        fh = self._log_fh
        self._log_fh = None
        if fh is not None and fh is not getattr(sys.stderr, "buffer", sys.stderr):
            try:
                fh.close()
            except OSError:
                pass

    # --- status ---------------------------------------------------------- #

    def snapshot(self) -> dict:
        """A JSON-able status dict for the control API / ``biopb admin status``."""
        with self._lock:
            st = self._state
            up = self._port_up()
            child_alive = self._child_alive()
            # State reflects reality first: a listening port is "serving" whether
            # we spawned, adopted, or have not yet been asked to manage it. Only
            # when nothing is listening does intent matter -- "down" if we want it
            # up (and are backing off / restarting), else "stopped".
            if up:
                state = "serving"
            elif child_alive:
                state = "starting"
            elif st.want:
                state = "down"
            else:
                state = "stopped"
            return {
                "state": state,
                "grpc_url": f"grpc://{self._spec.grpc_host}:{self._spec.grpc_port}",
                "web_url": f"http://{self._spec.web_host}:{self._spec.web_port}/",
                "pid": self._proc.pid if child_alive else None,
                "owned": st.owned,
                "restarts": st.restarts,
                "last_error": st.last_error,
            }
