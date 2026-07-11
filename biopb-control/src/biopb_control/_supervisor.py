"""Supervise the data (tensor) plane as a subprocess — never import it.

:class:`DataPlaneSupervisor` owns the tensor-server process the way the existing
``biopb server start`` command does (same ``python -m biopb_tensor_server.cli
launch`` argv, same detach/log idiom), but *persistently*: it starts the plane,
polls its liveness, and restarts it on crash with capped backoff. The control is
the **sole owner** of the data plane — it always spawns and manages its own
child and never adopts a server it did not start; a gRPC port already held by
another process is a *conflict* it refuses (surface, don't manage), not
something to attach to. This single-owner rule is what makes the state simple
(``self._proc`` alone tracks the child) and makes ``biopb control stop`` a complete
teardown. It never imports ``biopb_tensor_server`` — liveness is a cheap stdlib
TCP connect to the gRPC port, so no pyarrow/grpc is pulled into the lean control
(invariant I2).

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
    caller (the ``biopb control`` CLI) so the supervisor imports no server config.

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
    failures: int = 0  # consecutive failed/crashed attempts (drives backoff)
    restarts: int = 0  # total restarts since control start (for status)
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
        back to the control's own stderr if the file can't be opened, so a bad log
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

    def _spawn_locked(self) -> bool:
        """(Re)spawn the data plane. Returns True on success, False on a spawn
        failure that has been counted toward the backoff.

        A ``Popen`` that raises (OSError: a bad executable, ENOMEM, EAGAIN/too
        many processes) is treated like a failed attempt -- ``failures`` is
        bumped, the backoff window is armed, and ``last_error`` records it --
        rather than propagating. That keeps a failing spawn from escaping
        ``ensure`` / ``tick`` (and the ``/data_plane/ensure`` handler) uncounted
        and hammering with no backoff.
        """
        argv = self._build_argv()
        log = self._open_log()
        try:
            log.write(
                f"\n--- control: starting data plane at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode()
            )
        except (OSError, ValueError):
            pass
        logger.info("Spawning data plane: %s", " ".join(argv))
        # No detachment: the plane is a *tracked* child of the control -- if the
        # control dies the plane stays serving (already-spawned planes survive a
        # control blip, migration doc S3), but while the control lives it owns and
        # can reap this child directly.
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=self._child_env(),
                close_fds=True,
            )
        except OSError as exc:
            st = self._state
            st.failures += 1
            st.last_error = f"failed to spawn data plane: {exc}"
            st.next_attempt_at = time.monotonic() + self._backoff()
            logger.error(st.last_error)
            return False
        self._state.up_since = None
        return True

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

    def _conflict_message(self) -> str:
        return (
            f"data-plane port {self._spec.grpc_host}:{self._spec.grpc_port} is "
            "held by a process the control did not start; refusing to manage it. "
            "Stop that server, then the control will bring up one it owns."
        )

    def ensure(self) -> None:
        """Idempotently bring the data plane up under control ownership.

        The control is the *sole* owner of the data plane: it always spawns and
        manages its own tensor-server child, and never adopts or coexists with a
        server it did not start. If the gRPC port is already held by another
        process, that is a conflict the control refuses (recorded in ``last_error``
        / the ``conflict`` status) rather than silently attaching to -- stop the
        stray server and the control will own a freshly-spawned one. Safe to call
        repeatedly / concurrently.
        """
        with self._lock:
            st = self._state
            st.want = True
            self._reap_dead_child()  # normalize a stale handle before deciding
            if self._proc is not None:
                st.last_error = None
                return  # our child is alive (starting or serving)
            if self._port_up():
                st.last_error = self._conflict_message()
                logger.error(st.last_error)
                return
            st.last_error = None
            # On success arm the next-restart window; on a spawn failure
            # _spawn_locked already counted it and armed the retry backoff.
            if self._spawn_locked():
                st.next_attempt_at = time.monotonic() + self._backoff()

    def _backoff(self) -> float:
        idx = min(self._state.failures, len(_BACKOFF_SCHEDULE) - 1)
        return _BACKOFF_SCHEDULE[idx]

    def tick(self) -> None:
        """One supervision step: restart a crashed child with backoff.

        Called on a fixed cadence by the supervision loop. The control owns the
        plane exclusively, so there is no adopted/unowned case to juggle: a live
        child is left alone, a crashed one is reaped and restarted, and a port
        held by a process we did not spawn is a conflict we surface, never manage.
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
                # The port is held, but not by a child of ours (a stray server,
                # or our child died and something else grabbed it). As the sole
                # owner we do not manage it -- surface the conflict (log once on
                # entry) and wait for it to clear.
                if st.last_error != self._conflict_message():
                    logger.error(self._conflict_message())
                st.last_error = self._conflict_message()
                st.up_since = None
                return

            # Port down and no live child of ours: (re)start when backoff allows.
            st.up_since = None
            now = time.monotonic()
            if now < st.next_attempt_at:
                return
            if self._spawn_locked():
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
        """Stop supervising and shut the data plane down.

        The control owns the plane exclusively, so stopping the control always stops
        the plane -- there is no 'adopted, leave it running' case. This is what
        makes ``biopb control stop`` a complete data-plane teardown.
        """
        with self._lock:
            self._state.want = False
            proc = self._proc
            self._proc = None
        if proc is not None:
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
        """A JSON-able status dict for the control API / ``biopb control status``."""
        with self._lock:
            st = self._state
            up = self._port_up()
            child_alive = self._child_alive()
            # The control owns the plane exclusively, so the port being up means one
            # of two things: our child serves it ("serving"), or a process we did
            # not start holds it ("conflict"). With nothing listening, intent
            # decides: "starting" while our child binds, "down" if we want it up
            # (backing off / restarting), else "stopped".
            if child_alive:
                state = "serving" if up else "starting"
            elif up:
                state = "conflict"
            elif st.want:
                state = "down"
            else:
                state = "stopped"
            return {
                "state": state,
                "grpc_url": f"grpc://{self._spec.grpc_host}:{self._spec.grpc_port}",
                "web_url": f"http://{self._spec.web_host}:{self._spec.web_port}/",
                "pid": self._proc.pid if child_alive else None,
                "restarts": st.restarts,
                "last_error": st.last_error,
            }
