"""KernelHost: owns a single child Jupyter kernel for MCP code execution.

The kernel is a separate process (started via ``jupyter_client``) that hosts
the napari viewer, dask, and the TensorFlightClient.  Running agent code there
— instead of on napari's Qt event-loop thread — means a runaway execution can
be interrupted (``SIGINT``) or hard-restarted (group ``SIGKILL`` + respawn)
without taking down the MCP server process.

Round trips go through :class:`_kernel_io.KernelChannels` and may overlap; the
kernel runs them one at a time. ``threading.RLock`` ``_lock`` serializes the
lifecycle only (start, restart, shutdown, respawn).
"""

import logging
import os
import re
import signal
import threading
import time
from typing import List, Optional

from biopb._lifecycle import deathwatch as _deathwatch, winjob as _winjob

from ._job_log import JobLog
from ._kernel_io import _IDLE_GRACE, KernelChannels, KernelGone

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Env var carrying the inherited *write* end of the window-close pipe. The
# in-kernel bootstrap writes a byte to this fd when the user
# closes the napari window; the launcher's reader thread reaps the kernel back
# to idle on the signal. The literal is mirrored in _bootstrap._install_window_
# close_hook (kept in sync by this comment, like _deathwatch.ENV_FD).
ENV_WINDOW_CLOSE_FD = "BIOPB_WINDOW_CLOSE_FD"

# Env var marking a kernel as the scratch one a verification runs in
# (``_scratch``). The in-kernel bootstrap reads it and leaves out everything
# user-facing -- above all it builds ``napari.Viewer(show=False)``, so the
# scratch kernel can take the session's own display, and its real GPU, without a
# window appearing in front of the user. The literal is mirrored in
# _bootstrap.is_scratch_kernel (kept in sync by this comment, like
# ENV_WINDOW_CLOSE_FD above).
ENV_SCRATCH = "BIOPB_SCRATCH_KERNEL"

# Env var the launcher sets when the session has no viewer, carrying why (config
# off, napari not installed, no display). Its presence makes the bootstrap skip
# Qt and napari; its value is what the tools tell the agent. The literal is
# mirrored in _bootstrap.no_viewer_reason (kept in sync by this comment).
ENV_NO_VIEWER = "BIOPB_NO_VIEWER"

# Env var handing the kernel the host client's session id, so its gate
# (_kernel_gate) can tell this process's requests from another Jupyter
# client's. The literal is mirrored in _kernel_gate.ENV_HOST_SESSION (kept in
# sync by this comment).
ENV_HOST_SESSION = "BIOPB_HOST_SESSION"

# Windows window-close fallback (no inherited fd there): the launcher polls this
# probe -- the zero-arg _viewer_window_alive() the bootstrap injects into the
# kernel namespace (see _bootstrap, mirrored by this comment) -- and tears the
# kernel down when it prints False. The timeout bounds a wedged kernel main
# thread; the probe itself is instant.
_WINDOW_ALIVE_PROBE = "print(_viewer_window_alive())"
_WINDOW_PROBE_TIMEOUT = 10.0


def _runtime_connection_file() -> str:
    """A fresh connection-file path in Jupyter's runtime dir.

    Left unset, jupyter_client writes a tempfile, which no Jupyter tool looks
    for. The runtime dir is per-platform (``jupyter --runtime-dir``), private
    to the user, and where ``jupyter qtconsole --existing`` resolves a bare
    file name.
    """
    import uuid

    from jupyter_core.paths import jupyter_runtime_dir
    from jupyter_core.utils import ensure_dir_exists

    runtime = jupyter_runtime_dir()
    ensure_dir_exists(runtime, mode=0o700)
    return os.path.join(runtime, f"kernel-biopb-{uuid.uuid4()}.json")


def attach_command(
    connection_file: Optional[str],
    *,
    python: Optional[str] = None,
    windows: Optional[bool] = None,
) -> Optional[str]:
    """The shell command that attaches qtconsole to *connection_file*.

    Run by this process's own interpreter (*python*, default
    ``sys.executable``): biopb installs qtconsole (through napari) but puts no
    ``jupyter`` on PATH, so a bare ``jupyter qtconsole`` finds nothing, or
    another install's. Quoted for the platform's shell (*windows*, default
    this one's), since a Windows profile path may contain spaces.
    """
    import sys

    if not connection_file:
        return None
    argv = [python or sys.executable, "-m", "qtconsole", "--existing", connection_file]
    if windows is None:
        windows = os.name == "nt"
    if windows:
        import subprocess

        return subprocess.list2cmdline(argv)
    import shlex

    return shlex.join(argv)


def _status_result(status: str, error_text: str) -> dict:
    """An execute-shaped result carrying only a status and why.

    Every "the kernel could not run this" answer has the same two empty output
    fields, present so `_format_execute_result` finds the keys it reads on the
    success path. Naming them once leaves each caller showing the one thing
    that distinguishes it -- which of the states it is.
    """
    return {"stdout": "", "result_text": "", "error_text": error_text, "status": status}


# A token-report pipe used to carry the connected (url, token) back to this
# process so a kernel restart could re-inject it (issue #86). It is gone with
# biopb/biopb#628: every token a kernel can now use is re-derivable at connect
# time from a source the next kernel reads too -- the control's credential file on
# disk, or $BIOPB_TENSOR_TOKEN inherited through the launch environment. Nothing
# is typed into the widget any more, so there is nothing left to remember.

# Prepended exec-line that installs the in-kernel parent-death watcher before
# the (possibly slow) napari bootstrap runs. Paired with the inherited read-end
# fd passed in BIOPB_PARENT_DEATH_FD; see _deathwatch and KernelHost._launch.
# Repeated --IPKernelApp.exec_lines args append, so this composes with the
# bootstrap line the launcher already passes.
_DEATHWATCH_ARG = (
    "--IPKernelApp.exec_lines=import biopb._lifecycle.deathwatch as _dw; _dw.install()"
)

# Prepended to an agent's cell so ``client`` tracks the tensor connection, which
# connects asynchronously.
_CELL_PREFIX = "client = _conn.client\n"

# Whether the viewer window is still open, evaluated after an agent's cell: a
# user-closed window turns viewer mutations into silent no-ops. None where the
# bootstrap bound no probe.
_WINDOW_ALIVE_EXPR = "globals().get('_viewer_window_alive', lambda: None)()"

# The kernel class every host kernel runs: the host's control requests
# (restart's graceful close, stopping a job) are its ops.
_KERNEL_CLASS_ARG = "--IPKernelApp.kernel_class=biopb_mcp.mcp._kernel_gate.GatedKernel"

# How long a control op may take in the kernel, unless its caller says.
_CONTROL_TIMEOUT = 5.0


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
        health_probe_code: Optional[str] = "print('_jobs' in dir())",
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
    ):
        self._extra_arguments = list(extra_arguments or [])
        self._kernel_name = kernel_name
        self._startup_timeout = startup_timeout
        self._execute_timeout = execute_timeout
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
        self._io = None  # KernelChannels, one per launched kernel
        # The job records, built from what each kernel publishes on iopub. One
        # per host rather than per kernel, so they outlive a restart.
        self.jobs = JobLog()
        # Set once per _launch(), alongside self._km: the connection file (and
        # so the attach command) is fixed for the kernel's lifetime, and
        # health() is polled every few seconds, so it's cached rather than
        # rebuilt on each call.
        self._attach_command = None
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
        # to idle on the signal.
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
        self._restarting = False  # restart() in flight: calls read "starting"
        # Which kernel this is, counted per launch: a claim on the kernel lasts
        # one generation (_writers).
        self.generation = 0

    # -- lifecycle ------------------------------------------------------

    def start(self):
        """Launch the kernel, wait until ready, then run the health probe.

        Holds the lifecycle lock for the whole bring-up. start() is the
        synchronous primitive: ensure_started() (start_kernel) and the tests call
        it. Taking the lock serializes it against a concurrent restart()/
        shutdown() (which take the same lock); without it both paths mutate the
        shared _km/_io/_pgid state concurrently and can leave the host attached to
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
        extra_args = [_KERNEL_CLASS_ARG, *self._extra_arguments]
        popen_kwargs = {}

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

        if pass_fds:
            popen_kwargs["pass_fds"] = tuple(pass_fds)

        self._km = KernelManager(
            kernel_name=self._kernel_name,
            connection_file=_runtime_connection_file(),
        )
        self._attach_command = attach_command(self._km.connection_file)
        # The client made below shares this session id, so the kernel knows
        # its host before anything can connect.
        env = dict(env)
        env[ENV_HOST_SESSION] = self._km.session.session
        self.jobs.host_session = self._km.session.session
        self.generation += 1
        try:
            try:
                self._km.start_kernel(
                    extra_arguments=extra_args,
                    env=env,
                    cwd=self._cwd,
                    # Own session/process group so a hard restart — or the
                    # kernel's own parent-death watcher — can group-kill the
                    # kernel and any subprocess it spawned (arbitrary agent
                    # code, a dask cluster a cell spun).
                    start_new_session=True,
                    **popen_kwargs,
                )
            finally:
                # The child has its own copy of each inherited end; the launcher
                # keeps the opposite end so a closure reaches across the pipe.
                # Death pipe: launcher keeps the write end. Window pipe: launcher
                # keeps the read end, so close its copy of the write end here.
                for _fd in (death_r, win_w):
                    if _fd is not None:
                        try:
                            os.close(_fd)
                        except OSError:
                            pass
            self._pgid = self._capture_pgid()
            self._assign_kernel_to_job()
            self._io = KernelChannels(self._km, on_iopub=self.jobs.on_iopub)
            self._io.start(self._startup_timeout, self._km.is_alive)
            self._start_window_watch()
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
        in the kernel namespace, so a probe failure says *why* the bootstrap
        failed (a missing dep, a Qt/GL init error) instead of just ``False``.
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

    def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        user_expressions: Optional[dict] = None,
    ) -> dict:
        """Run *code* in the kernel and return a result dict.

        Returns ``{stdout, result_text, error_text, status}`` where ``status``
        is one of ``ok``/``error`` (from the kernel reply), ``timeout`` (no
        reply within *timeout*; nothing is interrupted, see :meth:`_run_once`),
        or ``starting`` (the kernel is not ready yet — see below). Calls from
        several threads overlap; the kernel runs them in arrival order.

        The kernel is started on demand (start_kernel -> ensure_started), so a
        tool call may land while it is not running — idle/never-started, a failed
        start, or mid watchdog-respawn. We do NOT block on bring-up here: instead
        return immediately with a structured not-ready status the agent can act on
        (see :meth:`_not_ready_result`) — ``not_started`` / ``error`` (call
        start_kernel) or ``starting`` (a respawn in flight; poll ``server_status``
        / retry). ``server_status`` is a cheap, non-blocking readiness probe meant
        for exactly this.
        """
        # The channels are read *before* readiness: a restart clears _ready
        # before it replaces them, so a call that sees _ready set holds either
        # the ready kernel's or the outgoing one's (then failed by its close) --
        # never the next kernel's before its bootstrap has run.
        io = self._io
        if not self._ready.is_set():
            return self._not_ready_result()
        return self._execute_internal(code, timeout, user_expressions, io=io)

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
            return _status_result(
                "error",
                "Kernel startup failed: "
                + self._start_error
                + " The kernel is not running; call start_kernel to retry."
                + suffix,
            )
        if self._dead:
            return _status_result(
                "error",
                "Kernel is dead (respawn budget exhausted). Call "
                "start_kernel to launch a fresh kernel." + suffix,
            )
        if self.is_alive() or self._restarting:
            # A kernel exists but its bootstrap/health probe hasn't passed yet
            # (e.g. a watchdog respawn in flight), or a restart is between the
            # kill and the relaunch — booting, not idle.
            return _status_result(
                "starting",
                "Kernel is still starting. "
                "Poll server_status or retry in a few seconds.",
            )
        return _status_result(
            "not_started",
            "Kernel not started. Call start_kernel first, then poll "
            "server_status until it reports ready." + suffix,
        )

    def _execute_internal(
        self,
        code: str,
        timeout: Optional[float] = None,
        user_expressions: Optional[dict] = None,
        io: Optional[KernelChannels] = None,
    ) -> dict:
        """Execution bypassing the readiness wait.

        Used by the startup health probe and bootstrap-error fetch, which run
        *before* the kernel is marked ready (so they must not wait on it).
        With *user_expressions*, the result also carries their evaluated values
        under ``user_expressions``.
        """
        if timeout is None:
            timeout = self._execute_timeout
        if io is None:
            io = self._io
        if io is None:
            return _status_result("error", "Kernel is not running.")
        # Never wait on a kernel whose process is gone: its zmq channels stay
        # open (they reconnect to nothing), so the call would wait out its whole
        # timeout for a reply that cannot come.
        if not self.is_alive():
            return _status_result("error", "Kernel is not running.")

        res = self._run_once(io, code, timeout, user_expressions)
        # A preceding interrupt/error aborts requests already queued at the
        # kernel; "aborted" means our code never ran, so retry once.
        if res["status"] == "aborted":
            time.sleep(0.2)
            res = self._run_once(io, code, timeout, user_expressions)
        return res

    def _run_once(
        self,
        io: KernelChannels,
        code: str,
        timeout: float,
        user_expressions: Optional[dict] = None,
    ) -> dict:
        try:
            call = io.execute(code, timeout, user_expressions)
        except KernelGone:
            return _status_result(
                "error", "The kernel was shut down or restarted during this call."
            )
        except TimeoutError:
            # No interrupt. Everything sent here is a short snippet, so
            # overrunning means the main thread is busy with something else --
            # a cell, the agent's or an attached client's, or a task's
            # marshaled viewer call -- and a SIGINT lands in *that*. The
            # request stays queued and runs once the thread frees; its late
            # reply is skipped by message id.
            return _status_result(
                "timeout",
                (
                    f"No reply within {timeout}s: the kernel's main thread is "
                    "busy, most likely with a cell from a Jupyter client attached "
                    "to this kernel, or with a viewer call. Nothing was "
                    "interrupted, and this call will still run once it frees. "
                    "Retry later. restart_kernel only if it never frees -- it "
                    "destroys whatever the user is running, and their variables "
                    "and layers."
                ),
            )

        res = {
            "stdout": "".join(call.stdout),
            "result_text": "".join(call.results),
            "error_text": "".join(_strip_ansi("\n".join(tb)) for tb in call.errors),
            "status": call.reply.get("status", "unknown"),
        }
        if user_expressions:
            res["user_expressions"] = call.reply.get("user_expressions") or {}
        return res

    def control(self, op, timeout=_CONTROL_TIMEOUT, **args):
        """Run the kernel's control *op* (``_kernel_gate``) and return its result.

        On the control channel, so it does not wait behind a cell on the main
        thread. Needs no readiness: a restart's graceful close runs after
        readiness is cleared. Raises ``RuntimeError`` when the kernel is down or
        the op failed, ``TimeoutError`` when it took longer than *timeout*.
        """
        io = self._io
        if io is None:
            raise RuntimeError("kernel not running")
        try:
            reply = io.control(op, args, timeout)
        except KernelGone as exc:
            raise RuntimeError("the kernel went away") from exc
        if reply.get("status") != "ok":
            raise RuntimeError(reply.get("evalue") or "control op failed")
        return reply.get("r")

    def run_cell(self, code, job_id, origin, intent="", bare=False):
        """Send *code* as a cell, recorded as *job_id*; return at once.

        The cell runs on the kernel's main thread like any client's, queued
        behind whatever runs there. Its record (``self.jobs``) fills from iopub
        and ends on its request's idle, or -- if that is lost -- on its shell
        reply, which cannot be. Raises ``RuntimeError`` carrying what to do
        when the kernel is not ready (:meth:`_not_ready_result`).

        *bare* sends the code alone, without the ``client`` refresh: a
        verification runs the document and nothing else, so a workflow that
        never builds its own ``client`` fails there, as it would for its reader.
        """
        io = self._io
        if not self._ready.is_set() or io is None:
            raise RuntimeError(self._not_ready_result()["error_text"])

        def before_send(request):
            self.jobs.start_cell(job_id, request, code, origin, intent)

        def on_reply(reply):
            if reply is None:
                return  # the kernel went away; its records say so
            self.jobs.note_reply(job_id, reply)
            # The idle marks the output complete and normally ends the record
            # first; the reply ends it only if that idle never comes.
            timer = threading.Timer(_IDLE_GRACE, self.jobs.cell_replied, (job_id,))
            timer.daemon = True
            timer.start()

        return io.send_execute(
            code if bare else _CELL_PREFIX + code,
            on_reply,
            before_send,
            user_expressions={"w": _WINDOW_ALIVE_EXPR},
        )

    def interrupt_job(self, job_id, reason=None):
        """Stop *job_id* if it still runs (``_jobs.interrupt``), on the control
        channel. Whether the caller may is decided before this; *reason* is a
        person's, prefixed to the job's error.

        The kernel knows a cell by its request -- a cell's id is this host's --
        and a task by its own id, which is also what it is sent when the
        records do not hold it as running (its start may be in flight). When
        *job_id* no longer runs, ``running_job_id`` names what does.
        """
        key = self.jobs.stop_key(job_id) or job_id
        # Before the stop, which a cell's end can follow at once; taken back
        # if the stop does not happen.
        self.jobs.note_cancel(job_id, reason)
        reply = {}
        try:
            reply = self.control("interrupt", key=key, reason=reason)
        finally:
            if not reply.get("interrupted"):
                self.jobs.note_cancel(job_id, None)
        reply["job_id"] = job_id
        if reply.get("refused") == "not_running":
            running = self.jobs.running()
            reply["running_job_id"] = running["job_id"] if running else None
        return reply

    def _close_session(self, timeout):
        """Close the kernel's tensor client before a kill, bounded and
        best-effort (``_kernel_gate._close_session``): a wedged kernel just
        falls through to the kill."""
        try:
            self.control("close", timeout=timeout)
        except Exception:  # noqa: BLE001
            logger.debug("graceful close failed", exc_info=True)

    def interrupt(self):
        """Send SIGINT to the kernel. Takes no lock, so it can fire during a
        restart's graceful close or any call waiting on a busy kernel."""
        if self._km is not None:
            try:
                self._km.interrupt_kernel()
            except Exception:
                logger.debug("interrupt_kernel failed", exc_info=True)

    def restart(self):
        """Hard-restart: graceful-close the tensor client, group-kill, respawn.

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
            self._restarting = True
            # A restart is a recovery attempt: clear any stale failure / teardown
            # reason up front so a concurrent server_status/execute (which read
            # them without the lock) see "starting" (recovering) rather than the
            # old error while we rebuild. A fresh failure below records a new one.
            self._start_error = None
            self._teardown_reason = None
            # Before the graceful close, not at the kill: calls do not take this
            # lock, so only _ready keeps a submit from landing in the kernel
            # being replaced, after the tensor client it would use is gone.
            self._ready.clear()
            try:
                self._close_session(timeout=5.0)

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
                self._restarting = False
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
            # The short timeout keeps the graceful close off the Ctrl-C path.
            # Readiness goes first, as in restart().
            self._ready.clear()
            if self.is_alive():
                self._close_session(timeout=2.0)
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
            if self._io is not None:
                # Fails any call still waiting, rather than leaving it to time
                # out on a kernel that is going away.
                self._io.close()
        except Exception:
            logger.debug("stop_channels failed", exc_info=True)
        # A job still running dies with its kernel, which will never announce
        # its end.
        self.jobs.kernel_gone(self._teardown_reason or "")

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

        self._io = None
        self._pgid = None
        self._close_death_pipe()
        self._close_window_pipe()

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
        self._teardown_after_window_close()

    def _teardown_after_window_close(self):
        """Take the kernel down to idle because the viewer window went away.

        Shared by the POSIX pipe signal and the Windows poll: the reason is the
        sentence the agent reads back in `server_status` and in any abandoned
        job's error, so the two paths must not be able to word it differently.
        """
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
        stop is in flight; the kernel isn't ready (mid (re)spawn); or its main
        thread is busy -- a probe would only queue behind whatever holds it, one
        more per tick. A timeout/error probe is inconclusive and likewise
        retried; only a clean ``False`` reading (the Qt window's C++ object is
        gone) triggers teardown. A ``run_async`` task does not hold the main
        thread, so, like the POSIX byte signal, this can fire mid-task and stop
        it.
        """
        if self._stopping or not self._ready.is_set() or self.is_busy():
            return False
        res = self._execute_internal(_WINDOW_ALIVE_PROBE, timeout=_WINDOW_PROBE_TIMEOUT)
        if res.get("status") != "ok" or res.get("stdout", "").strip() != "False":
            return False  # alive, or an inconclusive (busy/timeout/error) probe
        if self._stopping or self._window_poll_stop.is_set():
            return False  # a concurrent restart/shutdown started -- don't race it
        self._teardown_after_window_close()
        return True

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
        # Armed once a ready host has been seen dead, and disarmed only by the
        # respawn or by the kernel coming back. It has to survive across ticks
        # because the death path withdraws _ready, which is the very flag the
        # first tick tested to get here.
        death_seen = False
        starved_since = None
        # wait() returns True only when stop is set; False on each interval.
        while not self._watchdog_stop.wait(self._watchdog_interval):
            if self._stopping or self.is_alive():
                death_seen = False
                starved_since = None
                continue
            if not death_seen:
                # Not ready means starting, not dying: start()/ensure_started()
                # holds the lock across a cold bring-up (napari, Qt, dask) that
                # legitimately outlasts several ticks, and a host that has never
                # been up has nothing to respawn -- its failure is already
                # reported through _start_error.
                if not self._ready.is_set():
                    continue
                death_seen = True
                # Withdraw readiness *before* contending for the lock. execute()
                # gates on _ready, so clearing it here turns new tool calls
                # into structured not-ready results instead of waits on a
                # kernel that cannot answer.
                self._ready.clear()
            # Confirm and act under the lock so we never race an in-flight
            # restart()/shutdown().
            if not self._lock.acquire(timeout=self._watchdog_interval):
                # Once per episode. The failure this replaces was silent: a dead
                # kernel, a 502 on every /api/jobs, and not one line in the log.
                if starved_since is None:
                    starved_since = time.monotonic()
                    logger.warning(
                        "Kernel died but the lifecycle lock is held; the "
                        "respawn is waiting for it to free."
                    )
                continue  # busy (likely a restart); re-check next tick
            try:
                if self._watchdog_stop.is_set() or self._stopping:
                    continue
                if self.is_alive():
                    continue  # a restart finished while we waited — fine
                if starved_since is not None:
                    logger.warning(
                        "Lifecycle lock freed after %.1fs; respawning now.",
                        time.monotonic() - starved_since,
                    )
                    starved_since = None
                self._handle_unexpected_death()
                death_seen = False
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

    @property
    def virtual_display(self):
        """The Xvfb display the kernel renders on, or None when it has the
        user's real one. The launcher (#90) marks the kernel env it hands us;
        the session child's own environ never carries it."""
        env = self._env or {}
        return env.get("DISPLAY") if env.get("BIOPB_VIRTUAL_DISPLAY") else None

    @property
    def no_viewer_reason(self):
        """Why this session has no napari viewer, or None when it has one. Set by
        the launcher in the kernel env it hands us (ENV_NO_VIEWER)."""
        return (self._env or {}).get(ENV_NO_VIEWER) or None

    def health(self) -> dict:
        """Liveness summary for server_status (cheap; no kernel round trip)."""
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
            "connection_file": self.connection_file,
            "attach_command": self._attach_command if self.is_alive() else None,
        }

    @property
    def connection_file(self):
        """Where a Jupyter client attaches to this kernel, or None when it is
        not running."""
        if not self.is_alive():
            return None
        return self._km.connection_file or None

    def is_alive(self) -> bool:
        try:
            return self._km is not None and self._km.is_alive()
        except Exception:
            return False

    def is_busy(self) -> bool:
        """Whether the kernel's main thread is running something, by its own
        last published status -- any client's request, not only ours. A
        ``run_async`` task does not count: the cell that started it has
        returned."""
        io = self._io
        return io is not None and io.execution_state == "busy"

    def _kernel_pid(self):
        try:
            return self._km.provisioner.pid
        except Exception:
            pass
        try:
            return self._km.kernel.pid
        except Exception:
            return None
