"""In-kernel async job runner for the MCP execute_code path.

Runs *inside* the child Jupyter kernel.  ``execute_code`` submits agent code
here; it executes in a **background daemon thread** so the kernel's main thread
(and its integrated ``%gui qt`` Qt event loop) stays free to service quick tool
calls — ``take_screenshot`` / ``server_status`` / ``poll_job`` — while a
multi-minute job runs.  Long C calls will block context switching, although dask,
gRPC and numpy all drop GIL, so the job and the viewer/tools are expected to run
smoothly.

Design notes
------------
* **One job at a time.** A second :func:`submit` while a job is running is
  rejected with the running job id (the single shared viewer / namespace makes
  concurrent mutation unsafe).
* **Main-thread affinity.** The viewer is a Qt/vispy object bound to the kernel
  main thread.  GUI mutations from the worker thread are marshaled via
  :func:`run_on_main`; ``_bootstrap`` wraps ``add_tensor`` + the ``add_*``
  family so the common paths are automatic.
* **Output capture.** A thread-aware stdout/stderr dispatcher (installed once by
  :func:`install`) routes a job thread's prints into that job's buffer instead
  of the kernel's iopub stream — keeping worker output out of iopub and away
  from the main-thread ``<<JOB_JSON>>`` reply line.
* **Stopping a job.** :func:`interrupt_current` force-stops the running job: it
  raises ``KeyboardInterrupt`` into the worker thread and, when a distributed dask
  client is active (the kernel's ``Client`` attached to the session child's
  ``LocalCluster``), :func:`_cancel` *also* cancels the client's in-flight futures
  — the only mid-``compute()`` stop short of ``restart_kernel``.  The in-process
  ``threads`` / ``synchronous`` schedulers have no futures to cancel, so a running
  ``compute()`` under them is stopped by the raised ``KeyboardInterrupt`` once it
  returns to Python bytecode, or by ``restart_kernel``.
"""

import ast
import ctypes
import io
import logging
import sys
import threading
import time
import traceback
from concurrent.futures import Future

logger = logging.getLogger(__name__)

# Prepended to every job so the namespace tracks the asynchronously-connecting
# tensor connection service (mirrors the old _server._REFRESH_PREFIX).
_REFRESH_PREFIX = "client = _conn.client\n"

# Keep at most this many terminal job records before evicting the oldest.
_MAX_RETAINED_JOBS = 32

# How long run_on_main waits for the main thread to service a marshaled call
# before giving up (seconds).  Generous: GUI ops are normally fast, but a first
# multiscale texture upload can take a while.
_RUN_ON_MAIN_TIMEOUT = 300.0

# Module state, wired by install().
_ip = None
_jobs = {}  # job_id -> _Job
_jobs_by_thread = {}  # thread ident -> _Job (active worker threads only)
_job_seq = 0
_lock = threading.RLock()


class _Job:
    __slots__ = (
        "job_id",
        "code",
        "status",
        "stdout",
        "result_text",
        "error_text",
        "cancel_reason",
        "interrupted",
        "thread",
        "started",
        "started_wall",
        "finished",
    )

    def __init__(self, job_id, code=""):
        self.job_id = job_id
        # The submitted source (as passed to submit(), before the internal
        # _REFRESH_PREFIX), so the observe UI can show what each job ran.
        self.code = code
        # running | ok | error | interrupted
        self.status = "running"
        self.stdout = io.StringIO()
        self.result_text = ""
        self.error_text = ""
        # Set by interrupt_current(): the job was force-stopped with a
        # KeyboardInterrupt raised into its thread, so its finalizer labels the
        # stop "interrupted" rather than a generic "error".
        self.interrupted = False
        # Human-readable reason a *user* acted on this job (cancel/interrupt via
        # the observe web UI). Threaded into the finalized error_text so the
        # agent sees the attribution through its normal poll_job / execute_code
        # result, instead of an unexplained cancellation. None for agent-driven
        # or untagged stops.
        self.cancel_reason = None
        self.thread = None
        self.started = time.monotonic()
        # Wall-clock epoch at submit, for human-readable audit timestamps in the
        # notebook export (`started` is monotonic and not displayable).
        self.started_wall = time.time()
        self.finished = None

    def elapsed(self):
        end = self.finished if self.finished is not None else time.monotonic()
        return round(end - self.started, 3)

    def snapshot(self):
        return {
            "job_id": self.job_id,
            "code": self.code,
            "status": self.status,
            "stdout": self.stdout.getvalue(),
            "result_text": self.result_text,
            "error_text": self.error_text,
            "cancel_reason": self.cancel_reason,
            "elapsed": self.elapsed(),
            "created": self.started_wall,
        }


# -- output capture ---------------------------------------------------------


class _JobStream:
    """stdout/stderr proxy: route a job thread's writes to its job buffer,
    otherwise delegate to the real (ipykernel) stream."""

    def __init__(self, real):
        self._real = real

    def write(self, s):
        job = _jobs_by_thread.get(threading.get_ident())
        if job is not None:
            return job.stdout.write(s)
        return self._real.write(s)

    def flush(self):
        try:
            return self._real.flush()
        except Exception:  # noqa: BLE001 - flush is best-effort
            pass

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_streams():
    if not isinstance(sys.stdout, _JobStream):
        sys.stdout = _JobStream(sys.stdout)
    if not isinstance(sys.stderr, _JobStream):
        sys.stderr = _JobStream(sys.stderr)


# -- main-thread marshaling -------------------------------------------------

_caller_cls = None


def _get_caller_cls():
    """Build (once) a QObject whose slot runs a callable and resolves a Future,
    propagating both result and exception across the thread boundary."""
    global _caller_cls
    if _caller_cls is not None:
        return _caller_cls

    from qtpy.QtCore import QObject, Slot

    class _MainThreadCaller(QObject):
        def __init__(self, fn, future):
            super().__init__()
            self._fn = fn
            self._future = future

        @Slot()
        def run(self):
            try:
                self._future.set_result(self._fn())
            except BaseException as exc:  # noqa: BLE001 - relay to caller
                self._future.set_exception(exc)

    _caller_cls = _MainThreadCaller
    return _caller_cls


def run_on_main(fn, *args, **kwargs):
    """Call ``fn(*args, **kwargs)`` on the Qt main thread and return its result.

    A no-op dispatch when already on the main thread.  Used to make viewer
    mutations from a background job thread safe; exceptions raised on the main
    thread are re-raised to the caller.
    """
    if threading.current_thread() is threading.main_thread():
        return fn(*args, **kwargs)

    from qtpy.QtCore import QCoreApplication, QMetaObject, Qt

    app = QCoreApplication.instance()
    if app is None:
        # No Qt loop running; best-effort inline (e.g. headless unit tests).
        return fn(*args, **kwargs)

    future = Future()
    caller = _get_caller_cls()(lambda: fn(*args, **kwargs), future)
    caller.moveToThread(app.thread())
    QMetaObject.invokeMethod(caller, "run", Qt.ConnectionType.QueuedConnection)
    try:
        return future.result(timeout=_RUN_ON_MAIN_TIMEOUT)
    finally:
        caller.deleteLater()


# -- execution --------------------------------------------------------------


def _exec_capture(code, ns, job):
    """Exec *code* in *ns*; if it ends in an expression, store its repr."""
    tree = ast.parse(code)
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body.pop()
    if tree.body:
        exec(compile(tree, "<job>", "exec"), ns)
    if last_expr is not None:
        value = eval(compile(ast.Expression(last_expr.value), "<job>", "eval"), ns)
        if value is not None:
            job.result_text = repr(value)


def _run(job, code):
    _jobs_by_thread[threading.get_ident()] = job
    exc = None
    try:
        _exec_capture(_REFRESH_PREFIX + code, _ip.user_ns, job)
    except BaseException:  # noqa: BLE001 - capture everything for the agent
        exc = True
        job.error_text = traceback.format_exc()
    finally:
        _jobs_by_thread.pop(threading.get_ident(), None)
        job.finished = time.monotonic()
        # A user-triggered interrupt raises KeyboardInterrupt into the thread,
        # surfacing here as exc; interrupt_current flags it so the stop is
        # labeled "interrupted" rather than a generic "error".
        if job.interrupted:
            job.status = "interrupted"
        else:
            job.status = "error" if exc else "ok"
        # Surface a user-attributed stop (interrupt via the observe web UI) to
        # the agent: prefix error_text with the reason so poll_job /
        # execute_code render it. The interrupt's KeyboardInterrupt traceback is
        # annotated with who triggered it.
        if job.cancel_reason and job.status in ("error", "interrupted"):
            job.error_text = (
                job.cancel_reason
                if not job.error_text
                else job.cancel_reason + "\n" + job.error_text
            )


def _has_running_job():
    return any(j.status == "running" for j in _jobs.values())


def _prune():
    terminal = [jid for jid, j in _jobs.items() if j.status != "running"]
    while len(_jobs) > _MAX_RETAINED_JOBS and terminal:
        del _jobs[terminal.pop(0)]


def submit(code):
    """Start *code* in a background thread; return ``{"job_id": ...}`` or,
    if a job is already running, ``{"error": "busy", "running_job_id": ...}``.
    """
    global _job_seq
    with _lock:
        # Re-assert the thread-aware stream wrap (idempotent) so a job thread's
        # output is captured even if something replaced sys.stdout since
        # install() — and so it works under pytest's per-phase capture.
        _install_streams()
        for jid, j in _jobs.items():
            if j.status == "running":
                return {"error": "busy", "running_job_id": jid}
        _job_seq += 1
        job_id = f"job-{_job_seq}"
        job = _Job(job_id, code)
        _jobs[job_id] = job
        _prune()
        thread = threading.Thread(
            target=_run, args=(job, code), name=job_id, daemon=True
        )
        job.thread = thread
        thread.start()
        return {"job_id": job_id, "status": "running"}


def poll(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "unknown", "error_text": ""}
    return job.snapshot()


def _cancel(job_id, reason=None):
    job = _jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "cancelled": False, "status": "unknown"}
    if job.status != "running":
        return {"job_id": job_id, "cancelled": False, "status": job.status}
    # Set the reason before cancelling futures: the job only unwinds after the
    # future-cancel makes its gather raise, so its finalizer is guaranteed to
    # see the reason.
    if reason:
        job.cancel_reason = reason
    # Distributed dask: cancel in-flight futures.  This is what actually stops a
    # blocking ``.compute()`` -- its tasks ARE registered in ``dc.futures`` for
    # the duration of the internal ``gather``, so cancelling them makes that
    # gather raise and unwinds the job thread.  ``dc.futures`` is keyed by task
    # key *string*, so we must rebuild ``Future`` objects from those keys:
    # ``Client.cancel`` filters its argument through ``futures_of()``, which
    # silently drops bare strings -- ``cancel(list(dc.futures))`` cancels nothing.
    # One job at a time, so every tracked future belongs to this job.
    dc = _ip.user_ns.get("_dask_client") if _ip is not None else None
    if dc is not None:
        try:
            from distributed import Future

            keys = list(dc.futures)
            if keys:
                dc.cancel([Future(k, dc) for k in keys], force=True)
        except Exception:  # noqa: BLE001 - cancel is best-effort
            logger.debug("distributed cancel failed", exc_info=True)
    return {"job_id": job_id, "cancelled": True, "status": job.status}


def _running_job():
    """The single running job, or None. One job at a time (see submit())."""
    for j in _jobs.values():
        if j.status == "running":
            return j
    return None


def _raise_in_thread(ident, exctype):
    """Asynchronously raise *exctype* in the thread with *ident*.

    CPython's ``PyThreadState_SetAsyncExc`` schedules the exception for the next
    bytecode executed by that thread — so it does *not* break a blocking C call
    (``time.sleep``, gRPC) until it returns to Python. Returns the number of
    threads affected (1 on success, 0 if the thread already finished).
    """
    if not ident:
        return 0
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(ident), ctypes.py_object(exctype)
    )
    if res > 1:  # never expected to hit >1; undo to avoid corrupting a bystander
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), None)
        return 0
    return res


def interrupt_current(reason=None):
    """Force-stop the running job: cooperative cancel *plus* a ``KeyboardInterrupt``
    raised directly into the job's worker thread.

    ``SIGINT`` can't do this — Python delivers signals only to the kernel main
    thread, while the job runs in a background worker — so a pure-Python loop
    would otherwise be stoppable only by ``restart_kernel``. This first runs
    :func:`_cancel` (attribution reason + in-flight dask-future cancel), then
    forces the worker thread via :func:`_raise_in_thread`. The exception lands at
    the next bytecode, so a blocking C call ends when it returns. ``{"interrupted":
    False, "status": "idle"}`` when the kernel is idle.
    """
    job = _running_job()
    if job is None:
        return {"job_id": None, "interrupted": False, "status": "idle"}
    job.interrupted = True  # finalize as "interrupted"
    _cancel(job.job_id, reason=reason)
    ident = job.thread.ident if job.thread is not None else None
    raised = _raise_in_thread(ident, KeyboardInterrupt)
    return {"job_id": job.job_id, "interrupted": bool(raised)}


def _code_preview(code, limit=80):
    """First non-blank line of *code*, trimmed and length-capped.

    Keeps jobs_summary light (the full source is in the per-job snapshot) while
    giving each list row an identifying one-liner.
    """
    for line in code.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def jobs_summary():
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "elapsed": j.elapsed(),
            "stdout_len": len(j.stdout.getvalue()),
            "code_preview": _code_preview(j.code),
        }
        for j in _jobs.values()
    ]


def export():
    """Full snapshots of all retained jobs, oldest-first, for notebook export.

    A read like :func:`jobs_summary` (round-tripped on the kernel main thread, no
    background job thread), but carrying each job's *full* source and captured
    output so the observe UI can serialize the session to a Jupyter notebook.
    """
    return [j.snapshot() for j in _jobs.values()]


def reset():
    """Drop all job records (used on kernel restart / re-bootstrap)."""
    with _lock:
        _jobs.clear()
        _jobs_by_thread.clear()


# -- viewer wrapping --------------------------------------------------------
#
# The agent-facing ``viewer`` is wrapped by a full main-thread marshaling proxy
# (``_viewer_proxy.make_viewer_proxy``) rather than the old method-by-method
# wrap, which leaked any returned handle (``viewer.layers``, ``viewer.dims``,
# ``viewer.layers[0]``) and let off-main mutations on it segfault Qt
# (biopb/biopb#100). ``run_on_main`` above remains the marshaling primitive the
# proxy uses, and is still exposed for power users.


def install(ip):
    """Wire the job runner into the kernel: store the InteractiveShell, install
    the thread-aware streams, and clear any prior job state."""
    global _ip
    _ip = ip
    _install_streams()
    reset()
