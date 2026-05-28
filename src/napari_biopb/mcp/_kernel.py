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
from typing import List, Optional

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Best-effort snippet run before a restart so dask releases child processes /
# cluster keys cleanly.  ``_dask_client`` is set by the bootstrap (None for the
# in-process scheduler, a distributed Client otherwise).
_DASK_RELEASE_SNIPPET = (
    "try:\n"
    "    _dc = globals().get('_dask_client')\n"
    "    if _dc is not None:\n"
    "        _dc.close()\n"
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
    ):
        import threading

        self._extra_arguments = list(extra_arguments or [])
        self._kernel_name = kernel_name
        self._startup_timeout = startup_timeout
        self._execute_timeout = execute_timeout
        self._busy_lock_timeout = busy_lock_timeout
        self._health_probe_code = health_probe_code
        self._health_probe_expect = health_probe_expect
        self._cwd = cwd
        self._env = env
        self._km = None
        self._kc = None
        self._lock = threading.RLock()

    # -- lifecycle ------------------------------------------------------

    def start(self):
        """Launch the kernel, wait until ready, then run the health probe."""
        self._launch()
        self._run_health_probe()

    def _launch(self):
        from jupyter_client import KernelManager

        self._km = KernelManager(kernel_name=self._kernel_name)
        self._km.start_kernel(
            extra_arguments=list(self._extra_arguments),
            env=self._env if self._env is not None else os.environ.copy(),
            cwd=self._cwd,
            # Own process group so a hard restart can group-kill any dask
            # child processes the kernel spawned.
            start_new_session=True,
        )
        self._kc = self._km.client()
        self._kc.start_channels()
        try:
            self._kc.wait_for_ready(timeout=self._startup_timeout)
        except RuntimeError:
            self._shutdown_current(self._kernel_pid())
            raise

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
            )

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
            try:
                self._execute_locked(_DASK_RELEASE_SNIPPET, timeout=5.0)
            except Exception:
                logger.debug("dask release failed on restart", exc_info=True)

            pid = self._kernel_pid()
            self._shutdown_current(pid)
            self._launch()
            self._run_health_probe()

    def shutdown(self):
        """Stop channels, shut down the kernel, remove the connection file."""
        with self._lock:
            self._shutdown_current(self._kernel_pid())

    def _shutdown_current(self, pid):
        try:
            if self._kc is not None:
                self._kc.stop_channels()
        except Exception:
            logger.debug("stop_channels failed", exc_info=True)

        # Group-kill to reap the kernel and any dask child processes it
        # spawned (an external cluster lives in another session and is spared).
        if hasattr(os, "killpg") and pid is not None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
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

    # -- status ---------------------------------------------------------

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
