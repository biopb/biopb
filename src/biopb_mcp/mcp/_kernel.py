"""KernelHost: owns a single child Jupyter kernel for MCP code execution.

The kernel is a separate process (started via ``jupyter_client``) that hosts
the napari viewer, dask, and the TensorFlightClient.  Running agent code there
— instead of on napari's Qt event-loop thread — means a runaway execution can
be interrupted (``SIGINT``) or hard-restarted (group ``SIGKILL`` + respawn)
without taking down the MCP server process.

A single ``threading.RLock`` serializes access to the one shared kernel.
"""

import logging
import os
import queue
import re
import signal
import threading
import time
from typing import List, Optional

from . import _deathwatch

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Prepended exec-line that installs the in-kernel parent-death watcher before
# the (possibly slow) napari bootstrap runs. Paired with the inherited read-end
# fd passed in BIOPB_PARENT_DEATH_FD; see _deathwatch and KernelHost._launch.
# Repeated --IPKernelApp.exec_lines args append, so this composes with the
# bootstrap line the launcher already passes.
_DEATHWATCH_ARG = (
    "--IPKernelApp.exec_lines="
    "import biopb_mcp.mcp._deathwatch as _dw; _dw.install()"
)

# Best-effort snippet run before a restart so dask releases child processes /
# cluster keys cleanly.  ``_dask_client`` / ``_dask_cluster`` are set by the
# bootstrap (both None for the in-process scheduler; for an auto-spun
# LocalCluster the cluster is closed after the client so workers don't orphan).
_DASK_RELEASE_SNIPPET = (
    "try:\n"
    "    _dc = globals().get('_dask_client')\n"
    "    if _dc is not None:\n"
    "        _dc.close()\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    _dk = globals().get('_dask_cluster')\n"
    "    if _dk is not None:\n"
    "        _dk.close()\n"
    "except Exception:\n"
    "    pass\n"
)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class KernelHost:
    """Manage the lifecycle of a single child Jupyter kernel."""

    def __init__(
        self,
        extra_arguments: Optional[List[str]] = None,
        kernel_name: str = "python3",
        startup_timeout: float = 60.0,
        execute_timeout: float = 120.0,
        busy_lock_timeout: float = 5.0,
        health_probe_code: Optional[str] = "print('viewer' in dir())",
        health_probe_expect: str = "True",
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        kernel_stdout=None,
        kernel_stderr=None,
        watchdog_interval: float = 5.0,
        watchdog_max_respawns: int = 3,
        watchdog_respawn_window: float = 60.0,
        parent_death_pipe: bool = True,
    ):
        self._extra_arguments = list(extra_arguments or [])
        self._kernel_name = kernel_name
        self._startup_timeout = startup_timeout
        self._execute_timeout = execute_timeout
        self._busy_lock_timeout = busy_lock_timeout
        self._health_probe_code = health_probe_code
        self._health_probe_expect = health_probe_expect
        self._cwd = cwd
        self._env = env
        # Where the kernel subprocess' native stdout/stderr fds go. None ->
        # inherit the launcher's fds (http mode). In stdio mode the launcher
        # passes a log file so native kernel output (Qt/GL/dask/gRPC) never
        # lands on fd 1, which there *is* the JSON-RPC protocol channel.
        self._kernel_stdout = kernel_stdout
        self._kernel_stderr = kernel_stderr
        self._km = None
        self._kc = None
        self._lock = threading.RLock()

        # -- orphan hardening (issue #13) -------------------------------
        # pgid captured at launch so the group-kill never re-derives it from a
        # possibly-dead / pid-recycled kernel pid.
        self._pgid = None
        # Launcher-held write end of the parent-death pipe; its closure (on
        # launcher death) makes the kernel self-terminate (failure mode 1).
        self._parent_death_pipe = parent_death_pipe and os.name == "posix"
        self._death_w = None
        # Liveness watchdog (failure mode 2): respawn an unexpectedly-dead
        # kernel after reaping its orphaned dask group, bounded to avoid a
        # crash-respawn thrash loop.
        self._watchdog_interval = watchdog_interval
        self._watchdog_max_respawns = watchdog_max_respawns
        self._watchdog_respawn_window = watchdog_respawn_window
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()
        self._respawn_times = []  # monotonic timestamps of recent respawns
        self._dead = False  # respawn budget exhausted -> manual restart needed
        self._stopping = False  # an intentional restart/shutdown is in flight

    # -- lifecycle ------------------------------------------------------

    def start(self):
        """Launch the kernel, wait until ready, then run the health probe."""
        self._launch()
        self._run_health_probe()
        self._start_watchdog()

    def _launch(self):
        from jupyter_client import KernelManager

        env = self._env if self._env is not None else os.environ.copy()
        extra_args = list(self._extra_arguments)
        popen_kwargs = {}

        # Redirect the kernel subprocess' native stdout/stderr fds. None ->
        # inherit the launcher's fds (http mode). In stdio mode the launcher
        # passes a log file so native kernel output (Qt/GL/dask/gRPC) never
        # lands on fd 1, which there *is* the JSON-RPC protocol channel.
        if self._kernel_stdout is not None:
            popen_kwargs["stdout"] = self._kernel_stdout
        if self._kernel_stderr is not None:
            popen_kwargs["stderr"] = self._kernel_stderr

        # Parent-death pipe: the kernel inherits the read end and self-kills its
        # process group when the launcher *process* dies (issue #13, mode 1).
        death_r = None
        if self._parent_death_pipe:
            death_r, self._death_w = os.pipe()
            env = dict(env)
            env[_deathwatch.ENV_FD] = str(death_r)
            popen_kwargs["pass_fds"] = (death_r,)
            extra_args = [_DEATHWATCH_ARG] + extra_args

        self._km = KernelManager(kernel_name=self._kernel_name)
        try:
            try:
                self._km.start_kernel(
                    extra_arguments=extra_args,
                    env=env,
                    cwd=self._cwd,
                    # Own session/process group so a hard restart — or the
                    # kernel's own parent-death watcher — can group-kill the
                    # dask children it spawned.
                    start_new_session=True,
                    **popen_kwargs,
                )
            finally:
                # The child has its own copy of the read end; the launcher keeps
                # only the write end so its closure reaches the kernel.
                if death_r is not None:
                    try:
                        os.close(death_r)
                    except OSError:
                        pass
            self._pgid = self._capture_pgid()
            self._kc = self._km.client()
            self._kc.start_channels()
            self._kc.wait_for_ready(timeout=self._startup_timeout)
        except Exception:
            self._shutdown_current()
            raise

    def _capture_pgid(self):
        """The kernel's process-group id, read once at launch time."""
        try:
            pgid = self._km.provisioner.pgid
            if pgid:
                return pgid
        except Exception:
            pass
        pid = self._kernel_pid()
        if pid is not None and hasattr(os, "getpgid"):
            try:
                return os.getpgid(pid)
            except OSError:
                pass
        return None

    def _run_health_probe(self):
        if not self._health_probe_code:
            return
        res = self.execute(
            self._health_probe_code, timeout=self._startup_timeout
        )
        haystack = res.get("stdout", "") + res.get("result_text", "")
        ok = (
            res.get("status") == "ok" and self._health_probe_expect in haystack
        )
        if not ok:
            raise RuntimeError(
                "Kernel bootstrap health probe failed "
                f"(status={res.get('status')!r}, stdout={res.get('stdout')!r}, "
                f"error={res.get('error_text')!r})"
                + self._bootstrap_error_detail()
            )

    def _bootstrap_error_detail(self) -> str:
        """Best-effort fetch of the traceback ``_bootstrap.bootstrap()`` stashes
        in the kernel namespace, so a probe failure says *why* the viewer is
        absent (a missing dep, a Qt/GL init error) instead of just ``False``.
        """
        try:
            res = self.execute(
                "print(globals().get('_BOOTSTRAP_ERROR', ''), end='')",
                timeout=self._startup_timeout,
            )
        except Exception:  # best-effort; never mask the original failure
            return ""
        tb = res.get("stdout", "").strip()
        return f"\n--- kernel bootstrap traceback ---\n{tb}" if tb else ""

    # -- execution ------------------------------------------------------

    def execute(self, code: str, timeout: Optional[float] = None) -> dict:
        """Run *code* in the kernel and return a result dict.

        Returns ``{stdout, result_text, error_text, status}`` where ``status``
        is one of ``ok``/``error`` (from the kernel reply), ``timeout`` (the
        execution exceeded *timeout* and was interrupted), or ``busy`` (the
        kernel lock could not be acquired within ``busy_lock_timeout``).
        """
        if timeout is None:
            timeout = self._execute_timeout

        acquired = self._lock.acquire(timeout=self._busy_lock_timeout)
        if not acquired:
            return {
                "stdout": "",
                "result_text": "",
                "error_text": (
                    "Kernel is busy with another execution. Wait for it to "
                    "finish, or call restart_kernel to force-stop it."
                ),
                "status": "busy",
            }
        try:
            return self._execute_locked(code, timeout)
        finally:
            self._lock.release()

    def _execute_locked(self, code: str, timeout: float) -> dict:
        if self._kc is None:
            return {
                "stdout": "",
                "result_text": "",
                "error_text": "Kernel is not running.",
                "status": "error",
            }

        res = self._run_once(code, timeout)
        # A preceding interrupt/error aborts requests already queued at the
        # kernel; "aborted" means our code never ran, so retry once.
        if res["status"] == "aborted":
            import time

            time.sleep(0.2)
            res = self._run_once(code, timeout)
        return res

    def _run_once(self, code: str, timeout: float) -> dict:
        stdout_parts: List[str] = []
        result_parts: List[str] = []
        error_parts: List[str] = []

        def output_hook(msg):
            msg_type = msg["header"]["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                stdout_parts.append(content.get("text", ""))
            elif msg_type in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    result_parts.append(text)
            elif msg_type == "error":
                tb = "\n".join(content.get("traceback", []))
                error_parts.append(_strip_ansi(tb))

        try:
            reply = self._kc.execute_interactive(
                code,
                store_history=False,
                allow_stdin=False,
                timeout=timeout,
                output_hook=output_hook,
            )
        except (queue.Empty, TimeoutError):
            self.interrupt()
            return {
                "stdout": "".join(stdout_parts),
                "result_text": "".join(result_parts),
                "error_text": (
                    f"Execution exceeded {timeout}s and was interrupted. "
                    "Wait for the kernel to settle, or call restart_kernel if "
                    "it stays unresponsive (a blocking C call ignores SIGINT)."
                ),
                "status": "timeout",
            }

        return {
            "stdout": "".join(stdout_parts),
            "result_text": "".join(result_parts),
            "error_text": "".join(error_parts),
            "status": reply["content"].get("status", "unknown"),
        }

    def interrupt(self):
        """Send SIGINT to the kernel.  Does NOT take the lock so it can fire
        while ``execute`` is blocked on a busy kernel."""
        if self._km is not None:
            try:
                self._km.interrupt_kernel()
            except Exception:
                logger.debug("interrupt_kernel failed", exc_info=True)

    def restart(self):
        """Hard-restart: release dask, group-kill the kernel, respawn."""
        with self._lock:
            # Tell the watchdog this alive->dead transition is intentional.
            self._stopping = True
            try:
                try:
                    self._execute_locked(_DASK_RELEASE_SNIPPET, timeout=5.0)
                except Exception:
                    logger.debug(
                        "dask release failed on restart", exc_info=True
                    )

                self._shutdown_current()
                self._launch()
                self._run_health_probe()
                # A manual restart clears the dead state and respawn budget.
                self._dead = False
                self._respawn_times.clear()
            finally:
                self._stopping = False
        # Re-arm the watchdog if it had stopped (e.g. respawn budget exhausted).
        self._start_watchdog()

    def shutdown(self):
        """Stop the watchdog, then group-kill the kernel and clean up."""
        # Stop the watchdog *before* taking the lock: it may itself be holding
        # the lock mid-respawn, and joining it while we hold the lock would
        # deadlock. _stopping also suppresses any respawn it is about to start.
        self._stopping = True
        self._stop_watchdog()
        with self._lock:
            self._shutdown_current()

    def _shutdown_current(self):
        try:
            if self._kc is not None:
                self._kc.stop_channels()
        except Exception:
            logger.debug("stop_channels failed", exc_info=True)

        # Group-kill via the pgid captured at launch — not os.getpgid(pid) now:
        # the kernel may already be dead (raising), and a recycled pid could
        # resolve to an unrelated group. Never signal the launcher's own group.
        pgid = self._pgid
        if hasattr(os, "killpg") and pgid and pgid != os.getpgrp():
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                logger.debug("killpg failed", exc_info=True)

        try:
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
        except Exception:
            logger.debug("shutdown_kernel failed", exc_info=True)
        try:
            if self._km is not None:
                self._km.cleanup_resources()
        except Exception:
            logger.debug("cleanup_resources failed", exc_info=True)

        self._kc = None
        self._pgid = None
        self._close_death_pipe()

    def _close_death_pipe(self):
        if self._death_w is not None:
            try:
                os.close(self._death_w)
            except OSError:
                pass
            self._death_w = None

    # -- liveness watchdog (issue #13, failure mode 2) ------------------

    def _start_watchdog(self):
        """Start the liveness watchdog thread if enabled and not running."""
        if self._watchdog_interval <= 0:
            return
        if (
            self._watchdog_thread is not None
            and self._watchdog_thread.is_alive()
        ):
            return
        self._stopping = False
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_run, name="kernel-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self):
        """Signal the watchdog to exit and (unless we *are* it) join it."""
        self._watchdog_stop.set()
        t = self._watchdog_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=self._watchdog_interval + 5.0)
            self._watchdog_thread = None

    def _watchdog_run(self):
        # wait() returns True only when stop is set; False on each interval.
        while not self._watchdog_stop.wait(self._watchdog_interval):
            if self._stopping or self.is_alive():
                continue
            # Possible unexpected death: confirm and act under the lock so we
            # never race an in-flight restart()/shutdown().
            if not self._lock.acquire(timeout=self._watchdog_interval):
                continue  # busy (likely a restart); re-check next tick
            try:
                if self._watchdog_stop.is_set() or self._stopping:
                    continue
                if self.is_alive():
                    continue  # a restart finished while we waited — fine
                self._handle_unexpected_death()
            finally:
                self._lock.release()

    def _handle_unexpected_death(self):
        """Reap the orphaned process group and respawn, bounded. Held lock."""
        logger.warning(
            "Kernel died unexpectedly; reaping orphans and respawning."
        )
        self._shutdown_current()  # killpg the captured pgid -> reap dask group

        now = time.monotonic()
        window = self._watchdog_respawn_window
        self._respawn_times = [
            t for t in self._respawn_times if now - t < window
        ]
        if len(self._respawn_times) >= self._watchdog_max_respawns:
            logger.error(
                "Kernel respawn limit reached (%d within %.0fs); marking the "
                "host dead. Call restart_kernel to recover.",
                self._watchdog_max_respawns,
                window,
            )
            self._dead = True
            self._watchdog_stop.set()
            return
        self._respawn_times.append(now)
        try:
            self._launch()
            self._run_health_probe()
            logger.info("Kernel respawned after unexpected death.")
        except Exception:
            logger.exception("Respawn after unexpected death failed.")
            self._dead = True
            self._watchdog_stop.set()

    # -- status ---------------------------------------------------------

    def health(self) -> dict:
        """Liveness summary for server_status (cheap; takes no lock)."""
        return {
            "alive": self.is_alive(),
            "busy": self.is_busy(),
            "dead": self._dead,
            "recent_respawns": len(self._respawn_times),
            "watchdog_running": (
                self._watchdog_thread is not None
                and self._watchdog_thread.is_alive()
            ),
        }

    def is_alive(self) -> bool:
        try:
            return self._km is not None and self._km.is_alive()
        except Exception:
            return False

    def is_busy(self) -> bool:
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._lock.release()
            return False
        return True

    def _kernel_pid(self):
        try:
            return self._km.provisioner.pid
        except Exception:
            pass
        try:
            return self._km.kernel.pid
        except Exception:
            return None
