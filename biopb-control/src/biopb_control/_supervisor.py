"""Supervise the data (tensor) plane as a subprocess — never import it.

:class:`DataPlaneSupervisor` owns the tensor-server process — it spawns ``python
-m biopb_tensor_server.cli launch`` with the shared log idiom, but *persistently*:
it starts the plane, polls its liveness, and restarts it on crash with capped
backoff. The control is
the **sole owner** of the data plane — it always spawns and manages its own
child and never adopts a server it did not start; a gRPC port already held by
another process is a *conflict* it refuses (surface, don't manage), not
something to attach to. This single-owner rule is what makes the state simple
(``self._proc`` alone tracks the child) and makes ``biopb control stop`` a complete
teardown. It never imports ``biopb_tensor_server`` — liveness is a cheap stdlib
TCP connect to the gRPC port, so no pyarrow/grpc is pulled into the lean control
(invariant I2).

**The plane is bound to the control's lifetime (Pattern O).** A crashed, killed,
or logged-out control must not orphan the tensor server: an orphan keeps holding
the gRPC port, which the next control start then reads as a *conflict* it refuses
— so the installer's stop→start (and every restart) would wedge behind a plane
nobody owns. The bind closes that: on POSIX the child inherits a parent-death
pipe (:mod:`biopb._lifecycle.deathwatch`) and runs in its own session, so an
*uncatchable* control death (SIGKILL/OOM/crash) EOFs the pipe and the plane
group-kills itself; on Windows it is assigned to a kill-on-close Job Object
(:mod:`biopb._lifecycle.winjob`) the control holds, so the OS reaps it when the
control's last handle closes. This is orthogonal to the *graceful* stop path
below (SIGTERM / the Windows sentinel), which still runs the plane's orderly
shutdown when the control is alive to ask for it; the bind is only the backstop
for when it is not.

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

from biopb import _locations
from biopb._lifecycle import deathwatch as _deathwatch, winjob as _winjob

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
    127.0.0.1). ``token`` is the data-plane access token the caller resolved:
    always set in remote mode, and ``None`` in local mode *unless* a token was
    supplied there too (local mode allows an optional token — enforcement is
    independent of the loopback/public bind).
    """

    config: Path
    grpc_host: str = "127.0.0.1"
    grpc_port: int = 8815
    web_host: str = "127.0.0.1"
    web_port: int = 8814
    # The built web/ SPA bundle. Consumed by the *control* (it is the single web
    # origin and serves the bundle itself, see _control.build_app), NOT forwarded
    # to the tensor subprocess — the sidecar no longer serves static assets.
    static_dir: Optional[Path] = None
    log_level: str = "INFO"
    server_log: Optional[Path] = None
    token: Optional[str] = None


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
        # Death-binding handles (see the module docstring). POSIX: the write end
        # of the child's parent-death pipe, re-armed per spawn and closed once the
        # child is gone. Windows: a kill-on-close Job Object created once and
        # reused across restarts, closed only on final stop.
        self._death_w: Optional[int] = None
        self._winjob = None
        self._state = _State()
        self._lock = threading.RLock()

    @property
    def log_path(self) -> Optional[Path]:
        """The file the data-plane subprocess's stdout/stderr is appended to, or
        ``None`` when no ``server_log`` was configured (output then goes to the
        control's own stderr). A public accessor so the control's log endpoint can
        tail it without reaching into ``_spec``."""
        p = self._spec.server_log
        return Path(p) if p is not None else None

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
        return argv

    def _child_env(self) -> dict:
        env = os.environ.copy()
        if self._spec.token:
            env["BIOPB_TENSOR_TOKEN"] = self._spec.token
        # No token resolved (tokenless local mode): the tensor `launch` runs
        # tokenless on its own because the config binds the flight server to
        # loopback — no bypass signal is needed or read anymore. (A local
        # deployment *may* still carry a token; then the branch above sets it.)
        #
        # Mark the plane as control-owned so its HTTP sidecar reports
        # `supervised` on /api/admin/status; the admin UI then routes restarts
        # through the control (POST /api/data_plane/restart) rather than telling
        # the user the plane is self-managed (biopb/biopb#418).
        env["BIOPB_DATA_PLANE_SUPERVISED"] = "1"
        return env

    def _open_log(self):
        """Open (once) the append-binary file the child's stdout/stderr go to.

        Binary + unbuffered for the same reason the CLI's server log is: the
        child and its native libraries (gRPC, Arrow) emit arbitrary bytes. Falls
        back to the control's own stderr if the file can't be opened, so a bad log
        path never blocks the plane from starting.

        Rotated once here, before the first open of this supervisor's lifetime, so
        the plane's stdout log (which has no in-process RotatingFileHandler) does
        not grow unbounded across control restarts — mirroring what ``control
        start`` does for ``control.log``. Rotating here (not per child respawn)
        keeps the appended fd stable while the control lives.
        """
        if self._log_fh is not None:
            return self._log_fh
        path = self._spec.server_log
        if path is None:
            self._log_fh = getattr(sys.stderr, "buffer", sys.stderr)
            return self._log_fh
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            _locations.rotate_log(Path(path))
            self._log_fh = open(path, "ab", buffering=0)
        except OSError:
            logger.warning("Cannot open data-plane log %s; using stderr", path)
            self._log_fh = getattr(sys.stderr, "buffer", sys.stderr)
        return self._log_fh

    # --- lifecycle ------------------------------------------------------- #

    def _spawn_locked(self) -> bool:
        """(Re)spawn the data plane. Returns True on success, False on a spawn
        failure that has been counted toward the backoff.

        Any ``OSError`` from bring-up -- the parent-death pipe's ``os.pipe`` under
        fd exhaustion (EMFILE/ENFILE), or ``Popen`` (a bad executable, ENOMEM,
        EAGAIN/too many processes) -- is treated like a failed attempt: ``failures``
        is bumped, the backoff window is armed, and ``last_error`` records it,
        rather than propagating. That keeps a failing spawn from escaping ``ensure``
        / ``tick`` (and the ``/data_plane/ensure`` handler) uncounted and hammering
        with no backoff. So the pipe arm runs *inside* the try, alongside ``Popen``.
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
        # The plane is a tracked child bound to the control's lifetime (module
        # docstring): while the control lives it owns and reaps this child
        # directly; if the control dies uncatchably the child reaps *itself* off
        # the parent-death pipe (POSIX) or the OS reaps it off the closed Job
        # Object (Windows), so a dead control never orphans the plane into a
        # port-holding conflict.
        env = self._child_env()
        popen_kwargs: dict = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        # Arm the pipe inside the try so an os.pipe() failure (fd exhaustion) is
        # counted toward the backoff exactly like a Popen failure, not raised out
        # of ensure/tick uncounted. death_r is pre-set so `finally` can reference
        # it whether or not the arm completed.
        death_r = None
        try:
            death_r = self._arm_parent_death(env, popen_kwargs)
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                close_fds=True,
                **popen_kwargs,
            )
        except OSError as exc:
            self._close_death_pipe()  # drop the half-armed write end
            st = self._state
            st.failures += 1
            st.last_error = f"failed to spawn data plane: {exc}"
            st.next_attempt_at = time.monotonic() + self._backoff()
            logger.error(st.last_error)
            return False
        finally:
            # The child inherited its own copy of the read end; close the
            # control's copy so only the child's death (or ours) shuts the pipe.
            if death_r is not None:
                try:
                    os.close(death_r)
                except OSError:
                    pass
        self._assign_to_job()
        self._state.up_since = None
        return True

    def _arm_parent_death(self, env: dict, popen_kwargs: dict) -> Optional[int]:
        """POSIX: arm the child's parent-death pipe; return the read fd to close
        after ``Popen`` (``None`` off POSIX, where the Job Object binds instead).

        Creates a pipe, passes the read end to the child (fd inherited via
        ``pass_fds``, its number in ``BIOPB_PARENT_DEATH_FD``), and keeps the
        write end on ``self._death_w``. The child's :func:`biopb._lifecycle.
        deathwatch.install` blocks on the read end and self-terminates on EOF, so
        an uncatchable control death takes the plane down. The child is put in its
        **own session** so the deathwatch's group-kill reaps only the plane and
        its descendants, never anything else in the control's group.
        """
        if os.name == "nt":
            return None
        self._close_death_pipe()  # drop any stale write end before re-arming
        death_r, self._death_w = os.pipe()
        env[_deathwatch.ENV_FD] = str(death_r)
        popen_kwargs["pass_fds"] = (death_r,)
        popen_kwargs["start_new_session"] = True
        return death_r

    def _assign_to_job(self) -> None:
        """Windows: put the freshly-spawned plane in a kill-on-close Job Object so
        it (and everything it spawns) dies with the control even on an uncatchable
        ``TerminateProcess`` (biopb/biopb#403). No-op on POSIX (the parent-death
        pipe covers it) and best-effort — a failure just leaves the child
        unbound. The job is created once and reused across restarts."""
        if os.name != "nt":
            return
        if self._winjob is None:
            self._winjob = _winjob.create_kill_on_close_job()
        if self._winjob is not None and self._proc is not None:
            _winjob.assign_process(self._winjob, self._proc.pid)

    def _close_death_pipe(self) -> None:
        """Close the control's write end of the parent-death pipe, if held.

        Only ever called once the child is gone (reaped / stopped), so it never
        races the child into a spurious self-kill; re-arming a fresh pipe on the
        next spawn drops any leftover here first."""
        w = self._death_w
        self._death_w = None
        if w is not None:
            try:
                os.close(w)
            except OSError:
                pass

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
        # The crashed child held the read end; free our write end so the next
        # spawn re-arms a fresh pipe (the reused Windows job is left intact).
        self._close_death_pipe()
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
        self._close_child_bindings()
        self._close_log()

    def _close_child_bindings(self) -> None:
        """Release the death-binding handles after the child is stopped.

        Closes the POSIX parent-death pipe and, on Windows, force-reaps any
        surviving job member and closes the Job Object (the graceful ``_terminate``
        has usually already emptied it). Called only on a full stop — a crash
        respawn keeps the reused job and re-arms the pipe."""
        self._close_death_pipe()
        if self._winjob is not None:
            _winjob.terminate_job(self._winjob)
            _winjob.close_job(self._winjob)
            self._winjob = None

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
        # The one definition the tensor server's shutdown listener also binds to
        # (biopb._locations.tensor_stop_sentinel), so writer and watcher
        # cannot disagree — a single fixed name under the biopb state dir, not
        # keyed by PID.
        return _locations.tensor_stop_sentinel()

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
