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

from biopb._lifecycle import deathwatch as _deathwatch, winjob as _winjob

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Env var carrying the inherited *write* end of the window-close pipe. The
# in-kernel bootstrap (non-headless) writes a byte to this fd when the user
# closes the napari window; the launcher's reader thread reaps the kernel back
# to idle on the signal. The literal is mirrored in _bootstrap._install_window_
# close_hook (kept in sync by this comment, like _deathwatch.ENV_FD).
ENV_WINDOW_CLOSE_FD = "BIOPB_WINDOW_CLOSE_FD"

# Windows window-close fallback (no inherited fd there): the launcher polls this
# probe -- the zero-arg _viewer_window_alive() the bootstrap injects into the
# kernel namespace (see _bootstrap, mirrored by this comment) -- and tears the
# kernel down when it prints False. The timeout bounds a wedged kernel main
# thread; the probe itself is instant.
_WINDOW_ALIVE_PROBE = "print(_viewer_window_alive())"
_WINDOW_PROBE_TIMEOUT = 10.0

# Env var carrying the inherited *write* end of the token-report pipe (issue
# #86). On every successful tensor-server connect the in-kernel bootstrap writes
# one "url\ttoken\n" line to this fd; the launcher's reader thread caches the
# latest (url, token) in the MCP-server process. Because that process outlives a
# kernel restart, the cached token is re-injected into the next kernel's env as
# BIOPB_TENSOR_TOKEN (see _launch) -- so a token the user typed into the Tensor
# Browser survives restart_kernel / close-window->start_kernel without ever
# touching disk. The literal is mirrored in _bootstrap._make_token_report_hook.
ENV_TOKEN_REPORT_FD = "BIOPB_TOKEN_REPORT_FD"

# Env vars the kernel's TensorConnection.resolve_from_config reads to seed its
# (url, token) before auto_connect. Re-injecting the remembered token through the
# same env path means the kernel needs no special "restored token" code -- it is
# indistinguishable from a token set in the launch environment.
_TENSOR_TOKEN_ENV = "BIOPB_TENSOR_TOKEN"
_TENSOR_URL_ENV = "BIOPB_TENSOR_URL"

# Prepended exec-line that installs the in-kernel parent-death watcher before
# the (possibly slow) napari bootstrap runs. Paired with the inherited read-end
# fd passed in BIOPB_PARENT_DEATH_FD; see _deathwatch and KernelHost._launch.
# Repeated --IPKernelApp.exec_lines args append, so this composes with the
# bootstrap line the launcher already passes.
_DEATHWATCH_ARG = (
    "--IPKernelApp.exec_lines=import biopb._lifecycle.deathwatch as _dw; _dw.install()"
)

# Best-effort dask release, the tail of _GRACEFUL_CLOSE_SNIPPET below (no
# standalone caller).  ``_dask_client`` is set by the bootstrap (None for the
# in-process scheduler; a real Client when attached to the session child's
# distributed cluster). The kernel never owns the cluster, so closing the
# client is all there is to release here.
_DASK_RELEASE_SNIPPET = (
    "try:\n"
    "    _dc = globals().get('_dask_client')\n"
    "    if _dc is not None:\n"
    "        _dc.close()\n"
    "except Exception:\n"
    "    pass\n"
)

# Best-effort snippet run *before* the group-kill on shutdown AND restart so the
# tensor server sees a clean Flight GOAWAY (and cancels any in-flight do_get)
# instead of discovering the dropped connection only via async socket teardown
# after the kill. That teardown lag is what lets a `biopb server stop` issued
# right after Ctrl-C block on its graceful drain (pyarrow FlightServerBase.
# shutdown waits for in-flight requests to finish) — and a restart drops the
# connection just as abruptly, so both teardown paths share this snippet.
# Closes the tensor client, then releases dask. Bounded + best-effort: a
# busy/wedged kernel just falls through to the kill, so this never holds up
# Ctrl-C beyond the short timeout in shutdown().
_GRACEFUL_CLOSE_SNIPPET = (
    "try:\n"
    "    _c = globals().get('_conn')\n"
    "    if _c is not None and getattr(_c, 'client', None) is not None:\n"
    "        _c.client.close()\n"
    "except Exception:\n"
    "    pass\n"
) + _DASK_RELEASE_SNIPPET


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
        window_close_pipe: bool = True,
        window_poll_interval: float = 2.0,
        cluster_host=None,
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
        # Set once the kernel has launched AND its bootstrap health probe has
        # passed. The kernel is started on demand (start_kernel -> ensure_started)
        # and tool calls gate on this so they never dispatch into a half-built or
        # not-yet-started kernel (they get a structured not-ready status instead).
        # Cleared on teardown.
        self._ready = threading.Event()
        # The reason the last bring-up failed terminally (str), or None. A failed
        # bootstrap (missing Qt/OpenGL, a health probe that never passes) and a
        # never-started/booting kernel both leave _ready unset, so this records
        # *why* so a tool call can report a terminal error rather than a generic
        # "starting". Set under the lock by start()/restart()/respawn on
        # failure and cleared on a successful bring-up; execute() and health()
        # read it to surface a terminal error instead of an endless "starting".
        self._start_error = None

        # On-demand start: the kernel is NOT launched at construction. The
        # launcher constructs the host idle and the first start_kernel tool call
        # drives ensure_started() (synchronous, like restart()). A never-started
        # host (idle: not alive) is distinguished from one mid-boot (alive but
        # not ready, e.g. a watchdog respawn) via is_alive(), so no extra flag is
        # needed.
        # Why the kernel was last torn down, when the cause is a user action
        # rather than a crash (e.g. the user closed the napari window). Set just
        # before the teardown; surfaced by execute()/health() so the agent is
        # told *why* a running job vanished instead of a bare "not started".
        # Cleared by ensure_started() on the next explicit start.
        self._teardown_reason = None

        # -- orphan hardening (issue #13) -------------------------------
        # pgid captured at launch so the group-kill never re-derives it from a
        # possibly-dead / pid-recycled kernel pid.
        self._pgid = None
        # Windows counterpart to the pgid + parent-death pair below: a Job Object
        # (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) the daemon holds and the kernel is
        # assigned to at launch. os.killpg does not exist on Windows, so both
        # POSIX reapers are inert there; the job ties the kernel's whole process
        # tree to the daemon's handle, so even a force-killed or crashed daemon
        # reaps the kernel -- GIL state and all (biopb/biopb#403). None off
        # Windows, or if job creation is unavailable (degrades to the old path).
        self._winjob = None
        # Launcher-held write end of the parent-death pipe; its closure (on
        # launcher death) makes the kernel self-terminate (failure mode 1).
        self._parent_death_pipe = parent_death_pipe and os.name == "posix"
        self._death_w = None
        # Window-close pipe (reverse of the death pipe): the *kernel* holds the
        # write end and writes a byte when the user closes the napari window; the
        # launcher holds this read end and a reader thread reaps the kernel back
        # to idle on the signal. Only meaningful with a viewer (the launcher sets
        # window_close_pipe=False for a headless kernel).
        self._window_close_pipe = window_close_pipe and os.name == "posix"
        self._window_r = None
        self._window_thread = None
        # Windows can't inherit the pipe fd (subprocess has no pass_fds there),
        # so the window-close -> shutdown feature falls back to polling the
        # in-kernel _viewer_window_alive() probe on a thread and tearing the
        # kernel down to idle once the user closes the napari window. Same
        # feature flag, same teardown path -- only the transport differs by OS.
        self._window_close_poll = window_close_pipe and os.name == "nt"
        self._window_poll_interval = window_poll_interval
        self._window_poll_stop = threading.Event()
        # Token-report pipe (issue #86): the *kernel* holds the write end and
        # writes "url\ttoken" on each successful connect; the launcher holds this
        # read end and a reader thread caches the latest (url, token) so the next
        # _launch re-injects it. POSIX-only, like the death/window pipes (it uses
        # the same inherited-fd plumbing); on Windows a GUI-entered token is not
        # persisted across a kernel restart.
        self._token_report_pipe = os.name == "posix"
        self._token_r = None
        self._token_thread = None
        # Latest tensor (url, token) reported by a connected kernel, remembered
        # in this (server) process across kernel restarts and re-injected into the
        # next kernel's env (issue #86). Deliberately kept across teardown -- that
        # is the whole point. Never written to disk.
        self._tensor_url = None
        self._tensor_token = None
        # Liveness watchdog (failure mode 2): respawn an unexpectedly-dead
        # kernel after reaping its orphaned process group, bounded to avoid a
        # crash-respawn thrash loop.
        self._watchdog_interval = watchdog_interval
        self._watchdog_max_respawns = watchdog_max_respawns
        self._watchdog_respawn_window = watchdog_respawn_window
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()
        self._respawn_times = []  # monotonic timestamps of recent respawns
        self._dead = False  # respawn budget exhausted -> manual restart needed
        self._stopping = False  # an intentional restart/shutdown is in flight

        # Daemon-owned dask cluster (or None). _launch calls ensure() and injects
        # the scheduler address so the kernel attaches to it instead of spinning
        # its own; the daemon owns its lifetime, so a kernel restart/reap here
        # leaves the cluster (and its warm workers) untouched. See _cluster.py.
        self._cluster_host = cluster_host

    # -- lifecycle ------------------------------------------------------

    def start(self):
        """Launch the kernel, wait until ready, then run the health probe.

        Holds the lifecycle lock for the whole bring-up. start() is the
        synchronous primitive: ensure_started() (start_kernel) and the tests call
        it. Taking the lock serializes it against a concurrent restart()/
        shutdown() (which take the same lock); without it both paths mutate the
        shared _km/_kc/_pgid state concurrently and can leave the host attached to
        the wrong kernel or leak an orphaned kernel process. The lock is
        reentrant, so the health probe's internal execute() — and ensure_started()
        calling start() while already holding the lock — re-enter without
        deadlocking.
        """
        with self._lock:
            try:
                self._launch()
                self._run_health_probe()
            except Exception as exc:
                # Record *why* so a tool call reports a terminal startup error
                # (via _not_ready_result) instead of an opaque failure.
                # _run_health_probe folds the in-kernel bootstrap traceback into
                # its message, so the reason flows through.
                self._start_error = str(exc) or repr(exc)
                raise
            self._start_error = None
            self._start_watchdog()

    def ensure_started(self) -> dict:
        """Idempotent, synchronous on-demand start. Returns the host state.

        The launcher constructs the host idle (no eager bring-up); the
        ``start_kernel`` tool calls this on first demand. A ready host no-ops;
        otherwise this brings the kernel up and **blocks until it is ready or the
        bring-up fails** (bounded by ``startup_timeout``) — the same blocking
        contract as :meth:`restart`. It is also the recovery path: an explicit
        start clears a prior terminal failure / dead state / teardown reason (and
        tears down a half-up kernel left by a failed probe) and re-attempts, so a
        failed or window-closed kernel comes back without a separate
        restart_kernel call.

        Returns ``{"state": "ready"}`` or ``{"state": "error", "error": <why>}``.
        """
        with self._lock:
            if self._ready.is_set():
                return {"state": "ready"}
            # Explicit (re)start: drop any stale terminal/teardown state, and
            # tear down a half-up kernel from a prior failed probe so _launch
            # doesn't orphan it.
            self._teardown_reason = None
            self._dead = False
            self._respawn_times.clear()
            if self._km is not None:
                self._shutdown_current()
            try:
                self.start()
            except Exception as exc:
                return {"state": "error", "error": str(exc) or repr(exc)}
            return {"state": "ready"}

    def _launch(self):
        from jupyter_client import KernelManager

        env = self._env if self._env is not None else os.environ.copy()
        extra_args = list(self._extra_arguments)
        popen_kwargs = {}

        # Re-inject a remembered tensor token (issue #86) so a kernel rebuilt by
        # restart_kernel / start_kernel reconnects with the token the user entered
        # before -- which lived only in the now-dead prior kernel. The kernel's
        # resolve_from_config already reads BIOPB_TENSOR_TOKEN, so this is the same
        # env path, just sourced from this process's memory instead of the shell.
        if self._tensor_token:
            env = dict(env)
            env[_TENSOR_TOKEN_ENV] = self._tensor_token
            if self._tensor_url:
                env[_TENSOR_URL_ENV] = self._tensor_url

        # Attach this kernel to the daemon-owned dask cluster. ensure() spins it
        # on the first launch (returning as soon as the scheduler is bound, so
        # workers register while the kernel imports napari) and returns the cached
        # address on later launches. None -> the daemon owns no cluster (owner
        # "kernel", a non-distributed scheduler, an external address, or a spin
        # failure); the kernel then resolves dask from its own config.
        if self._cluster_host is not None:
            from ._cluster import DASK_ADDRESS_ENV

            address = self._cluster_host.ensure()
            if address:
                env = dict(env)
                env[DASK_ADDRESS_ENV] = address

        # Redirect the kernel subprocess' native stdout/stderr fds. None ->
        # inherit the launcher's fds (http mode). In stdio mode the launcher
        # passes a log file so native kernel output (Qt/GL/dask/gRPC) never
        # lands on fd 1, which there *is* the JSON-RPC protocol channel.
        if self._kernel_stdout is not None:
            popen_kwargs["stdout"] = self._kernel_stdout
        if self._kernel_stderr is not None:
            popen_kwargs["stderr"] = self._kernel_stderr

        pass_fds = []

        # Parent-death pipe: the kernel inherits the read end and self-kills its
        # process group when the launcher *process* dies (issue #13, mode 1).
        death_r = None
        if self._parent_death_pipe:
            death_r, self._death_w = os.pipe()
            env = dict(env)
            env[_deathwatch.ENV_FD] = str(death_r)
            pass_fds.append(death_r)
            extra_args = [_DEATHWATCH_ARG] + extra_args

        # Window-close pipe (reverse direction): the kernel inherits the *write*
        # end and writes a byte when the user closes the napari window; the
        # launcher keeps the read end and a reader thread reaps the kernel back
        # to idle on the signal. No exec-line — the bootstrap installs the hook.
        win_w = None
        if self._window_close_pipe:
            self._window_r, win_w = os.pipe()
            env = dict(env)
            env[ENV_WINDOW_CLOSE_FD] = str(win_w)
            pass_fds.append(win_w)

        # Token-report pipe (issue #86): same direction as the window pipe — the
        # kernel inherits the *write* end and reports its (url, token) on connect;
        # the launcher keeps the read end and caches the token for re-injection.
        token_w = None
        if self._token_report_pipe:
            self._token_r, token_w = os.pipe()
            env = dict(env)
            env[ENV_TOKEN_REPORT_FD] = str(token_w)
            pass_fds.append(token_w)

        if pass_fds:
            popen_kwargs["pass_fds"] = tuple(pass_fds)

        self._km = KernelManager(kernel_name=self._kernel_name)
        try:
            try:
                self._km.start_kernel(
                    extra_arguments=extra_args,
                    env=env,
                    cwd=self._cwd,
                    # Own session/process group so a hard restart — or the
                    # kernel's own parent-death watcher — can group-kill the
                    # kernel and any subprocess it spawned (arbitrary agent
                    # code). The session child owns the dask cluster in *its*
                    # group, so this group-kill never touches it.
                    start_new_session=True,
                    **popen_kwargs,
                )
            finally:
                # The child has its own copy of each inherited end; the launcher
                # keeps the opposite end so a closure reaches across the pipe.
                # Death pipe: launcher keeps the write end. Window pipe: launcher
                # keeps the read end, so close its copy of the write end here.
                # (The token pipe is the same direction as the window pipe.)
                for _fd in (death_r, win_w, token_w):
                    if _fd is not None:
                        try:
                            os.close(_fd)
                        except OSError:
                            pass
            self._pgid = self._capture_pgid()
            self._assign_kernel_to_job()
            self._kc = self._km.client()
            self._kc.start_channels()
            self._kc.wait_for_ready(timeout=self._startup_timeout)
            self._start_window_watch()
            self._start_token_watch()
        except Exception:
            self._shutdown_current()
            raise

    def _assign_kernel_to_job(self):
        """Windows: put the freshly-launched kernel in the daemon's kill-on-close
        Job Object, so it (and everything it spawns) dies with the daemon even on
        an uncatchable force-kill (biopb/biopb#403). No-op on POSIX (the pgid +
        parent-death pipe cover it) and best-effort -- a failure just leaves the
        pre-#403 behavior. The job is created once and reused across restarts."""
        if os.name != "nt":
            return
        if self._winjob is None:
            self._winjob = _winjob.create_kill_on_close_job()
        pid = self._kernel_pid()
        if self._winjob is not None and pid is not None:
            _winjob.assign_process(self._winjob, pid)

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
            self._ready.set()
            return
        # Use the internal executor: the public execute() waits on _ready, which
        # this probe is what *sets* — waiting on ourselves would deadlock.
        res = self._execute_internal(
            self._health_probe_code, timeout=self._startup_timeout
        )
        haystack = res.get("stdout", "") + res.get("result_text", "")
        ok = res.get("status") == "ok" and self._health_probe_expect in haystack
        if not ok:
            raise RuntimeError(
                "Kernel bootstrap health probe failed "
                f"(status={res.get('status')!r}, stdout={res.get('stdout')!r}, "
                f"error={res.get('error_text')!r})" + self._bootstrap_error_detail()
            )
        # Probe passed: the kernel is fully booted and tools may dispatch to it.
        self._ready.set()

    def _bootstrap_error_detail(self) -> str:
        """Best-effort fetch of the traceback ``_bootstrap.bootstrap()`` stashes
        in the kernel namespace, so a probe failure says *why* the viewer is
        absent (a missing dep, a Qt/GL init error) instead of just ``False``.
        """
        try:
            res = self._execute_internal(
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
        execution exceeded *timeout* and was interrupted), ``busy`` (the kernel
        lock could not be acquired within ``busy_lock_timeout``), or
        ``starting`` (the kernel is not ready yet — see below).

        The kernel is started on demand (start_kernel -> ensure_started), so a
        tool call may land while it is not running — idle/never-started, a failed
        start, or mid watchdog-respawn. We do NOT block on bring-up here: instead
        return immediately with a structured not-ready status the agent can act on
        (see :meth:`_not_ready_result`) — ``not_started`` / ``error`` (call
        start_kernel) or ``starting`` (a respawn in flight; poll ``server_status``
        / retry). ``server_status`` is a cheap, non-blocking readiness probe meant
        for exactly this.
        """
        if not self._ready.is_set():
            return self._not_ready_result()
        return self._execute_internal(code, timeout)

    def _not_ready_result(self) -> dict:
        """Structured status for a tool call that landed while the kernel is not
        ready, differentiated so the agent knows what to do:

        * ``error`` — a terminal startup failure (``_start_error``) or a dead
          kernel (respawn budget exhausted): call ``start_kernel`` to retry.
        * ``starting`` — a kernel exists but isn't ready yet (a watchdog respawn
          in progress): poll ``server_status`` / retry. (start_kernel itself is
          synchronous, so its caller blocks rather than seeing this.)
        * ``not_started`` — idle / never started: call ``start_kernel`` first.

        A user-attributed ``_teardown_reason`` (e.g. the user closed the window)
        is appended so an abandoned job is explained, not bare.
        """
        reason = self._teardown_reason
        suffix = f" ({reason})" if reason else ""
        if self._start_error is not None:
            return {
                "stdout": "",
                "result_text": "",
                "error_text": (
                    "Kernel startup failed: "
                    + self._start_error
                    + " The kernel is not running; call start_kernel to retry."
                    + suffix
                ),
                "status": "error",
            }
        if self._dead:
            return {
                "stdout": "",
                "result_text": "",
                "error_text": (
                    "Kernel is dead (respawn budget exhausted). Call "
                    "start_kernel to launch a fresh kernel." + suffix
                ),
                "status": "error",
            }
        if self.is_alive():
            # A kernel exists but its bootstrap/health probe hasn't passed yet
            # (e.g. a watchdog respawn in flight) — booting, not idle.
            return {
                "stdout": "",
                "result_text": "",
                "error_text": (
                    "Kernel is still starting (napari viewer / dask bring-up). "
                    "Poll server_status or retry in a few seconds."
                ),
                "status": "starting",
            }
        return {
            "stdout": "",
            "result_text": "",
            "error_text": (
                "Kernel not started. Call start_kernel first, then poll "
                "server_status until it reports ready." + suffix
            ),
            "status": "not_started",
        }

    def _execute_internal(self, code: str, timeout: Optional[float] = None) -> dict:
        """Lock-guarded execution, bypassing the readiness wait.

        Used by the startup health probe and bootstrap-error fetch, which run
        *before* the kernel is marked ready (so they must not wait on it).
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
        """Hard-restart: graceful-close tensor/dask, group-kill, respawn.

        Deliberately NOT ``shutdown()`` + ``start()``: shutdown() must join
        the watchdog *outside* the lock (the watchdog may hold the lock
        mid-respawn, so joining under the lock deadlocks), while restart
        keeps the watchdog alive and needs the whole dead->alive transition
        under one lock hold so concurrent tools/the watchdog never observe —
        or race into — the gap between kill and relaunch.
        """
        with self._lock:
            # Tell the watchdog this alive->dead transition is intentional.
            self._stopping = True
            # A restart is a recovery attempt: clear any stale failure / teardown
            # reason up front so a concurrent server_status/execute (which read
            # them without the lock) see "starting" (recovering) rather than the
            # old error while we rebuild. A fresh failure below records a new one.
            self._start_error = None
            self._teardown_reason = None
            try:
                try:
                    self._execute_locked(_GRACEFUL_CLOSE_SNIPPET, timeout=5.0)
                except Exception:
                    logger.debug("graceful close failed on restart", exc_info=True)

                self._shutdown_current()
                try:
                    self._launch()
                    self._run_health_probe()
                except Exception as exc:
                    self._start_error = str(exc) or repr(exc)
                    raise
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
            # Best-effort, bounded graceful close so the tensor server isn't left
            # to discover SIGKILL'd connections via async socket teardown — which
            # can make a `biopb server stop` right after Ctrl-C hang on its drain
            # (see _GRACEFUL_CLOSE_SNIPPET). A busy/wedged kernel falls through to
            # the SIGKILL below; the short timeout keeps this off the Ctrl-C path.
            if self.is_alive():
                try:
                    self._execute_locked(_GRACEFUL_CLOSE_SNIPPET, timeout=2.0)
                except Exception:
                    logger.debug("graceful close on shutdown failed", exc_info=True)
            self._shutdown_current()
            # Terminal path only (restart() drives _shutdown_current directly and
            # must keep the job): drop the job handle so it doesn't leak and its
            # closure fires kill-on-close as a final backstop (biopb/biopb#403).
            if self._winjob is not None:
                _winjob.close_job(self._winjob)
                self._winjob = None

    def _shutdown_current(self):
        # The kernel is going away: tools must wait for the next successful
        # health probe (restart/respawn) before dispatching again.
        self._ready.clear()
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

        # Windows has no killpg; TerminateJobObject is the from-outside tree-kill
        # equivalent -- it reaps the kernel and everything it spawned, and works
        # even if km.shutdown_kernel below is wedged or raises (biopb/biopb#403).
        # The handle is kept open (closed only in shutdown()) so a restart reuses
        # it and it stays the kill-on-close backstop for daemon death.
        if os.name == "nt" and self._winjob is not None:
            _winjob.terminate_job(self._winjob)

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
        self._close_window_pipe()
        self._close_token_pipe()

    def _close_death_pipe(self):
        if self._death_w is not None:
            try:
                os.close(self._death_w)
            except OSError:
                pass
            self._death_w = None

    def _close_window_pipe(self):
        # Signal the Windows poll thread to exit on the next tick (it never
        # joins, so this is just a flag; no-op on POSIX where the poll is
        # unused). Cleared again by _start_window_watch on the next launch.
        self._window_poll_stop.set()
        # Sole closer of the read end (the reader thread never closes it), so a
        # teardown by any path closes it exactly once. The reader thread is a
        # daemon that exits on EOF once the kernel's write end is gone.  Safe to
        # call while the reader thread is blocking on os.read(fd, 1): os.close
        # unblocks the read with OSError (caught by the thread's try/except), so
        # no deadlock — the daemon thread simply returns.
        if self._window_r is not None:
            try:
                os.close(self._window_r)
            except OSError:
                pass
            self._window_r = None
        self._window_thread = None

    def _close_token_pipe(self):
        # Sole closer of the token-report read end (the reader thread never closes
        # it). Closing unblocks a thread parked in os.read with OSError, so the
        # daemon thread just returns -- no deadlock. We deliberately KEEP
        # self._tensor_url/_token across teardown: re-injecting them on the next
        # launch is the whole point (issue #86).
        if self._token_r is not None:
            try:
                os.close(self._token_r)
            except OSError:
                pass
            self._token_r = None
        self._token_thread = None

    # -- window-close watcher -------------------------------------------

    def _start_window_watch(self):
        """Start the window-close watcher: the POSIX pipe reader, or the Windows
        poll thread, whichever transport is configured. No-op for neither."""
        if self._window_close_poll:
            self._window_poll_stop.clear()
            self._window_thread = threading.Thread(
                target=self._poll_window_close,
                name="window-close-poll",
                daemon=True,
            )
            self._window_thread.start()
            return
        fd = self._window_r
        if fd is None:
            return
        self._window_thread = threading.Thread(
            target=self._watch_window_close,
            args=(fd,),
            name="window-close-watch",
            daemon=True,
        )
        self._window_thread.start()

    def _watch_window_close(self, fd):
        """Block until the kernel signals a window close (byte) or dies (EOF).

        A byte = the user closed the napari window -> tear the kernel down to
        idle so the agent rebuilds it with start_kernel; record a teardown reason
        first so execute()/server_status tell the agent *why* (a running job is
        abandoned). EOF = the kernel already went away via another teardown path
        -> just exit. The fd is owned/closed by _close_window_pipe, so we never
        close it here (avoids racing a concurrent teardown that closes it too).
        """
        try:
            data = os.read(fd, 1)
        except OSError:
            return
        if not data:
            return  # EOF: kernel died via another path; nothing to do
        if self._stopping:
            return  # a restart/shutdown is already tearing the kernel down
        self._teardown_reason = (
            "the user closed the napari viewer window; the kernel was shut "
            "down and any running job was stopped"
        )
        try:
            self.shutdown()
        except Exception:
            logger.exception("teardown after window close failed")

    def _poll_window_close(self):
        """Windows fallback for the POSIX window-close pipe.

        Windows can't inherit the kernel's write-end fd, so instead of a
        byte-on-close we poll the in-kernel ``_viewer_window_alive()`` probe
        (injected by the bootstrap) and tear the kernel down to idle once the
        user closes the napari window. The loop exits on the first teardown or
        when signalled to stop (a restart/shutdown via _close_window_pipe).
        """
        while not self._window_poll_stop.wait(self._window_poll_interval):
            if self._window_close_tick():
                return

    def _window_close_tick(self) -> bool:
        """One window-close poll iteration. Returns True if the window was found
        closed and the kernel was torn down (the poll loop then exits).

        Acts only on a *positive* "window gone" reading from a healthy, idle
        kernel. Skipped (return False, retry next tick) when: an intentional
        stop is in flight; the kernel isn't ready (mid (re)spawn); or the kernel
        is busy -- a job holds the lock, the window is in use, and we must
        neither contend with it nor abort it. A busy/timeout/error probe is
        inconclusive and likewise retried; only a clean ``False`` reading (the
        Qt window's C++ object is gone) triggers teardown. Unlike the POSIX byte
        signal, this cannot fire mid-job (the probe can't run while the lock is
        held) -- the close is detected on the next idle tick instead.
        """
        if self._stopping or not self._ready.is_set() or self.is_busy():
            return False
        res = self._execute_internal(_WINDOW_ALIVE_PROBE, timeout=_WINDOW_PROBE_TIMEOUT)
        if res.get("status") != "ok" or res.get("stdout", "").strip() != "False":
            return False  # alive, or an inconclusive (busy/timeout/error) probe
        if self._stopping or self._window_poll_stop.is_set():
            return False  # a concurrent restart/shutdown started -- don't race it
        self._teardown_reason = (
            "the user closed the napari viewer window; the kernel was shut "
            "down and any running job was stopped"
        )
        try:
            self.shutdown()
        except Exception:
            logger.exception("teardown after window close failed")
        return True

    # -- token-report watcher (issue #86) -------------------------------

    def _start_token_watch(self):
        """Start the token-report reader thread if the pipe is configured."""
        fd = self._token_r
        if fd is None:
            return
        self._token_thread = threading.Thread(
            target=self._watch_token_report,
            args=(fd,),
            name="token-report-watch",
            daemon=True,
        )
        self._token_thread.start()

    def _watch_token_report(self, fd):
        """Cache the (url, token) the kernel reports on each successful connect.

        The kernel writes one ``url\\ttoken\\n`` line per connect (an empty token
        field means "no token / cleared"). We keep only the latest in
        ``_tensor_url`` / ``_tensor_token`` so the next :meth:`_launch` re-injects
        it (issue #86). EOF (the kernel went away via any teardown path) ends the
        thread. The fd is owned/closed by :meth:`_close_token_pipe`, so we never
        close it here — avoiding a race with a concurrent teardown.
        """
        buf = b""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                return
            if not chunk:
                return  # EOF: kernel gone
            buf += chunk
            # Process complete lines; keep any trailing partial in the buffer.
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._record_token_line(line)

    def _record_token_line(self, line: bytes) -> None:
        """Update the remembered (url, token) from one reported ``url\\ttoken``
        line. A malformed/undecodable line is ignored; an empty token field
        clears the remembered token (e.g. the user switched to a no-auth
        server)."""
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            return
        url, sep, token = text.partition("\t")
        if not sep:
            return  # not a token-report line
        self._tensor_url = url or None
        self._tensor_token = token or None

    # -- liveness watchdog (issue #13, failure mode 2) ------------------

    def _start_watchdog(self):
        """Start the liveness watchdog thread if enabled and not running."""
        if self._watchdog_interval <= 0:
            return
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
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
        logger.warning("Kernel died unexpectedly; reaping orphans and respawning.")
        self._shutdown_current()  # killpg the captured pgid -> reap kernel group

        now = time.monotonic()
        window = self._watchdog_respawn_window
        self._respawn_times = [t for t in self._respawn_times if now - t < window]
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
            self._start_error = None
            logger.info("Kernel respawned after unexpected death.")
        except Exception as exc:
            logger.exception("Respawn after unexpected death failed.")
            self._start_error = str(exc) or repr(exc)
            self._dead = True
            self._watchdog_stop.set()

    # -- status ---------------------------------------------------------

    def health(self) -> dict:
        """Liveness summary for server_status (cheap; takes no lock)."""
        return {
            "alive": self.is_alive(),
            "ready": self._ready.is_set(),
            "start_error": self._start_error,
            "teardown_reason": self._teardown_reason,
            "busy": self.is_busy(),
            "dead": self._dead,
            "recent_respawns": len(self._respawn_times),
            "watchdog_running": (
                self._watchdog_thread is not None and self._watchdog_thread.is_alive()
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
