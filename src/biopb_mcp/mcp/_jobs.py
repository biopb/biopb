"""In-kernel async job runner for the MCP execute_code path.

Runs *inside* the child Jupyter kernel.  ``execute_code`` submits agent code
here; it executes in a **background daemon thread** so the kernel's main thread
(and its integrated ``%gui qt`` Qt event loop) stays free to service quick tool
calls — ``take_screenshot`` / ``server_status`` / ``poll_job`` — while a
multi-minute job runs.  Because dask / gRPC / numpy release the GIL, the viewer
keeps updating live and the agent can watch and screenshot mid-job.

Design notes
------------
* **One job at a time.** A second :func:`submit` while a job is running is
  rejected with the running job id (the single shared viewer / namespace makes
  concurrent mutation unsafe).
* **Main-thread affinity.** The viewer is a Qt/vispy object bound to the kernel
  main thread.  GUI mutations from the worker thread are marshaled via
  :func:`run_on_main`; ``_bootstrap`` wraps ``load_tensor`` + the ``add_*``
  family so the common paths are automatic.
* **Output capture.** A thread-aware stdout/stderr dispatcher (installed once by
  :func:`install`) routes a job thread's prints into that job's buffer instead
  of the kernel's iopub stream — keeping worker output out of iopub and away
  from the main-thread ``<<JOB_JSON>>`` reply line.
* **Cancellation is cooperative.** :func:`cancel` sets a per-job event that job
  code may poll via ``cancelled()``; when a distributed dask client is active
  (the default kernel-local ``LocalCluster``) it *also* cancels the client's
  in-flight futures — the only mid-``compute()`` stop short of
  ``restart_kernel``.  The in-process ``threads`` / ``synchronous`` schedulers
  have no futures to cancel, so a running ``compute()`` under them can only be
  stopped by ``restart_kernel``.
"""

import ast
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
        "status",
        "stdout",
        "result_text",
        "error_text",
        "cancel_event",
        "thread",
        "started",
        "finished",
    )

    def __init__(self, job_id):
        self.job_id = job_id
        self.status = "running"  # running | ok | error | cancelled
        self.stdout = io.StringIO()
        self.result_text = ""
        self.error_text = ""
        self.cancel_event = threading.Event()
        self.thread = None
        self.started = time.monotonic()
        self.finished = None

    def elapsed(self):
        end = self.finished if self.finished is not None else time.monotonic()
        return round(end - self.started, 3)

    def snapshot(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stdout": self.stdout.getvalue(),
            "result_text": self.result_text,
            "error_text": self.error_text,
            "elapsed": self.elapsed(),
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


def cancelled():
    """True if the current job thread has been asked to cancel.

    Bound into the kernel namespace so job code can cooperatively check it
    (e.g. ``if cancelled(): break``).  Returns False off a job thread.
    """
    job = _jobs_by_thread.get(threading.get_ident())
    return bool(job is not None and job.cancel_event.is_set())


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
        value = eval(
            compile(ast.Expression(last_expr.value), "<job>", "eval"), ns
        )
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
        if job.cancel_event.is_set():
            job.status = "cancelled"
        else:
            job.status = "error" if exc else "ok"


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
        job = _Job(job_id)
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


def cancel(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "cancelled": False, "status": "unknown"}
    if job.status != "running":
        return {"job_id": job_id, "cancelled": False, "status": job.status}
    job.cancel_event.set()
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


def jobs_summary():
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "elapsed": j.elapsed(),
            "stdout_len": len(j.stdout.getvalue()),
        }
        for j in _jobs.values()
    ]


def reset():
    """Drop all job records (used on kernel restart / re-bootstrap)."""
    with _lock:
        _jobs.clear()
        _jobs_by_thread.clear()


# -- viewer wrapping --------------------------------------------------------

_VIEWER_GUI_METHODS = (
    "load_tensor",
    "add_image",
    "add_labels",
    "add_points",
    "add_shapes",
    "add_vectors",
    "add_surface",
    "add_tracks",
)


def wrap_viewer_for_threads(viewer):
    """Wrap the common viewer-mutating methods so a call from a background job
    thread is marshaled to the Qt main thread (a no-op on the main thread).

    Open-ended mutations (``layer.data = ...``, contrast limits, camera, dims)
    are not wrapped — job code must use :func:`run_on_main` for those.
    """
    import functools

    for name in _VIEWER_GUI_METHODS:
        method = getattr(viewer, name, None)
        if method is None:
            continue

        def make(bound):
            @functools.wraps(bound)
            def wrapper(*args, **kwargs):
                if threading.current_thread() is threading.main_thread():
                    return bound(*args, **kwargs)
                return run_on_main(bound, *args, **kwargs)

            return wrapper

        # napari.Viewer is a pydantic evented model with validate_assignment;
        # write through to the instance dict (same bypass _helpers.py uses).
        object.__setattr__(viewer, name, make(method))


def install(ip):
    """Wire the job runner into the kernel: store the InteractiveShell, install
    the thread-aware streams, and clear any prior job state."""
    global _ip
    _ip = ip
    _install_streams()
    reset()
