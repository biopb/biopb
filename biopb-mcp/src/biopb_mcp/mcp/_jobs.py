"""The kernel's side of the jobs: holding cells for Stop, and ``run_async``.

Runs *inside* the child Jupyter kernel. Every cell -- the agent's, a
verification's, a user's from an attached client -- is a plain execute request
on the main thread, and the host records it from the protocol (``_job_log``).
What this module keeps is what the host cannot do from outside:

* **Holding the running cell** (:func:`hold_cell`, from ``_kernel_gate``), so
  a Stop can name it and be checked against it.
* **Tasks.** :func:`run_async` runs a long compute on a worker thread, leaving
  the main thread -- and so Qt and the screenshots -- free. A task outlives the
  cell that started it, so its start and end are announced on iopub
  (:func:`_publish`), and the host files the thread's prints (which ipykernel
  attributes to that cell's request) under the task. One task at a time.
* **Stopping** (:func:`interrupt`, on the control thread): a ``SIGINT`` for
  the cell on the main thread, a ``KeyboardInterrupt`` raised into a task's
  thread. Who may stop what is the host's decision; this checks only that the
  job named still runs.
* **Main-thread affinity.** The viewer is a Qt/vispy object bound to the main
  thread. A task's viewer calls are marshaled through :func:`run_on_main`,
  which ``_viewer_proxy`` does for the whole ``viewer``.
"""

import contextlib
import ctypes
import logging
import os
import secrets
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import Future

from ._job_log import MSG_TYPE

logger = logging.getLogger(__name__)

# Attribution for a KeyboardInterrupt this runner did not raise (see _run). The
# kernel ignores SIGINT except while servicing a message (ipykernel installs
# default_int_handler only between its pre/post handler hooks), and the host no
# longer sends one on a timeout, so the realistic source is a Jupyter client
# attached to this kernel interrupting it while the job held the main thread.
_EXTERNAL_INTERRUPT_MSG = (
    "Stopped by an interrupt sent to the whole kernel, not by an error in this "
    "code. Most likely a Jupyter client attached to this kernel sent it while "
    "this job was running on the main thread (a viewer call)."
)

# How long run_on_main waits for the main thread to service a marshaled call
# before giving up (seconds).  Generous: GUI ops are normally fast, but a first
# multiscale texture upload can take a while.
_RUN_ON_MAIN_TIMEOUT = 300.0

# Module state, wired by install().
_ip = None
_jobs = {}  # a cell's request id, or a task's id -> _Job
_lock = threading.RLock()

# What the bootstrap binds into the kernel namespace. `_bootstrap` refuses to
# let a user plugin shadow any of it (#92), and names it from here rather than
# writing the list out twice. `_bootstrap` is the one that binds them, but
# `_jobs` is the module it already imports, and a set defined in the importer
# would make the dependency point the wrong way.
KERNEL_HANDLE_NAMES = frozenset(
    {
        "viewer",
        "client",
        "np",
        "da",
        "ops",
        "run_async",
        "_conn",
        "_jobs",
        "_viewer_window_alive",
        "_resync_view",
    }
)


class _Cell:
    """A cell held for Stop (:func:`hold_cell`): all the kernel needs of it is
    whether it still runs. How it ended is the host's, read from the protocol."""

    __slots__ = ("status",)
    thread = None  # on the main thread

    def __init__(self):
        self.status = "running"


class _Job:
    """A task (:func:`run_async`)."""

    __slots__ = (
        "job_id",
        "code",
        "status",
        "error_text",
        "cancel_reason",
        "interrupted",
        "thread",
        "started",
        "finished",
        "request",
        "result_text",
    )

    def __init__(self, job_id, code="", request=None):
        self.job_id = job_id
        # The execute request the task's output is published under: the one
        # of the cell that started it.
        self.request = request
        self.code = code
        # running | ok | error | interrupted
        self.status = "running"
        self.error_text = ""
        # A task's return value's repr, announced with the end.
        self.result_text = ""
        # Set by interrupt(), so the finalizer labels the stop "interrupted"
        # rather than a generic "error".
        self.interrupted = False
        # Why a person stopped it (the observe page's Stop), prefixed to the
        # error so the agent sees the stop was not its code failing.
        self.cancel_reason = None
        self.thread = None
        self.started = time.monotonic()
        self.finished = None

    def elapsed(self):
        end = self.finished if self.finished is not None else time.monotonic()
        return round(end - self.started, 3)


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
    mutations from a task's thread safe; exceptions raised on the main
    thread are re-raised to the caller.
    """
    if threading.current_thread() is threading.main_thread():
        return fn(*args, **kwargs)

    from qtpy.QtCore import QCoreApplication, QMetaObject, Qt

    app = QCoreApplication.instance()
    if app is None:
        # No Qt loop running; best-effort inline (e.g. unit tests with no Qt app).
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


def _request_id():
    """The msg_id of the execute request this thread is serving, or None
    outside a kernel (unit tests drive this module directly)."""
    kernel = getattr(_ip, "kernel", None)
    try:
        return kernel.get_parent("shell")["header"]["msg_id"]
    except Exception:  # noqa: BLE001 - no kernel, or no request in flight
        return None


def _publish(content):
    """Announce a job event to the host on iopub (``_job_log`` reads it).

    No parent header: a Jupyter client drops iopub from other sessions, and an
    unknown message type under its own request would be one more thing for it
    to ignore. The request the job's output goes under rides in the content.

    The send runs on ipykernel's IOPub thread, as ``OutStream`` does: the
    Session (its msg_id counter) is not thread-safe, and a task's event is
    sent from the task's thread. Streams are flushed first -- ``flush`` waits until the
    IOPub thread has taken the output -- so the job's last output is on iopub
    ahead of its end.

    Sent from that thread straight to its socket (:class:`_OnIOPubThread`):
    ``send_multipart`` would queue it a second time, behind output a flush
    timer queued meanwhile, and a cell's start would then trail the cell's
    first output.
    """
    kernel = getattr(_ip, "kernel", None)
    iopub = getattr(kernel, "iopub_thread", None)
    if iopub is None:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - flush is best-effort
            pass

    def send():
        try:
            kernel.session.send(
                _OnIOPubThread(iopub), MSG_TYPE, content, ident=MSG_TYPE.encode()
            )
        except Exception:  # noqa: BLE001 - a lost event must not fail the job
            logger.debug("job event not published", exc_info=True)

    iopub.schedule(send)


class _OnIOPubThread:
    """The IOPub thread's socket, for a send already running on that thread.

    ``_really_send`` is what the thread's own queue ends in; it is ipykernel's
    (6.x) and not public, so without it the send is queued as usual.
    """

    def __init__(self, iopub):
        self.send_multipart = getattr(iopub, "_really_send", iopub.send_multipart)


def _publish_start(job):
    _publish(
        {
            "event": "start",
            "job_id": job.job_id,
            "request": job.request,
            "code": job.code,
        }
    )


def _publish_end(job):
    _publish(
        {
            "event": "end",
            "job_id": job.job_id,
            "status": job.status,
            "error_text": job.error_text,
            "result_text": job.result_text,
            "cancel_reason": job.cancel_reason,
            "elapsed": job.elapsed(),
        }
    )


def _run(job, body):
    """Run *body* (no arguments) on this worker thread as *job*, and settle
    and announce how it ended."""
    exc = None
    try:
        body()
    except KeyboardInterrupt:
        exc = True
        job.error_text = traceback.format_exc()
        # A KeyboardInterrupt this runner did not cause was delivered from
        # outside: a SIGINT to the kernel, relayed here because the job was
        # inside a run_on_main slot on the main thread when it landed. It is a
        # *stop*, not a defect in the submitted code, so label and attribute it
        # rather than hand back a bare traceback -- the same reasoning that gave
        # interrupt its flag, applied to the door it does not own.
        # Unlabeled, it reads as the code itself breaking.
        if not job.interrupted:
            job.interrupted = True
            job.cancel_reason = job.cancel_reason or _EXTERNAL_INTERRUPT_MSG
    except BaseException:  # noqa: BLE001 - capture everything for the agent
        exc = True
        job.error_text = traceback.format_exc()
    finally:
        job.finished = time.monotonic()
        # A user-triggered interrupt raises KeyboardInterrupt into the thread,
        # surfacing here as exc; interrupt flags it so the stop is
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
        _publish_end(job)


def _prune():
    # Only a running job is needed here, to stop it: the records are the
    # host's (_job_log).
    for key in [k for k, j in _jobs.items() if j.status != "running"]:
        del _jobs[key]


@contextlib.contextmanager
def hold_cell(request):
    """Hold a cell running on the main thread, for its duration.

    For ``_kernel_gate``, around every non-empty execute request, whoever sent
    it: nothing is announced -- the host records cells from the protocol --
    this is for Stop, which names a cell by its *request* (:func:`interrupt`).

    Ended under :data:`_lock`, which :func:`interrupt` checks and signals
    under: a ``SIGINT`` aimed at this cell lands before the cell is marked
    ended -- in the cell, or at the latest in the lock wait here, which a
    signal interrupts -- so it can never reach the next one. Landing here it
    is dropped: the cell it was aimed at is already over.
    """
    with _lock:
        cell = _jobs[request] = _Cell()
        _prune()
    try:
        yield
    finally:
        while True:
            try:
                with _lock:
                    cell.status = "ended"
                break
            except KeyboardInterrupt:
                continue


def run_async(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` on a worker thread; return its task id now.

    For a long compute the agent wants to watch: the cell that calls this ends
    at once, leaving the main thread -- and so the viewer and the screenshots
    -- free while the task runs. Poll the task by its id (``poll_job``); its
    prints and the repr of what ``fn`` returns are its record's.

    One task at a time: a second call while one runs raises. The task may use
    ``viewer``, which marshals to the main thread as it would from any thread;
    a user's cell can run meanwhile, and the two can race over the namespace
    and the viewer as in any asynchronous notebook. Stop a task with
    ``interrupt_kernel``: a ``KeyboardInterrupt`` raised into its thread, which
    lands at the next bytecode.

    Everything the cell prints after this call is filed with the task, since
    the two share the cell's request; make it the cell's last statement.
    """
    if not callable(fn):
        raise TypeError("run_async takes a function: run_async(fn, *args)")
    name = getattr(fn, "__qualname__", None) or repr(fn)
    with _lock:
        running = _running_task()
        if running is not None:
            raise RuntimeError(
                f"{running.job_id} is still running, and one task runs at a "
                f"time. Poll it with poll_job('{running.job_id}'), or stop it "
                "with interrupt_kernel."
            )
        job = _Job(
            f"task-{secrets.token_hex(3)}",
            f"run_async({name})",
            request=_request_id(),
        )
        _jobs[job.job_id] = job
        _prune()
        # Before the thread starts, so the host has the record before any of
        # the task's output reaches it.
        _publish_start(job)

        def body():
            value = fn(*args, **kwargs)
            if value is not None:
                job.result_text = repr(value)

        thread = threading.Thread(
            target=_run, args=(job, body), name=job.job_id, daemon=True
        )
        job.thread = thread
        thread.start()
    return job.job_id


def _running_task():
    """The running task, or None: one at a time (see run_async())."""
    for j in _jobs.values():
        if j.status == "running" and j.thread is not None:
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


def interrupt(key, reason=None):
    """Force-stop the job *key* names if it still runs: a ``KeyboardInterrupt``
    where it runs.

    Called on the kernel's control thread (``_kernel_gate``), so it is answered
    while the main thread is busy -- including with the very cell it stops.

    **The caller names the job, and this checks it is still running** under
    :data:`_lock`: a stop aimed at a job that has just ended is ``{"refused":
    "not_running"}`` and touches nothing, rather than landing on whatever runs
    now. The caller's view comes from iopub and may be stale; the kernel's is
    not. *key* is a cell's request -- its id is the host's, which this kernel
    never learns -- or a task's id. Whether the caller may stop it is the
    host's decision, made before it gets here.

    Where the interrupt goes depends on where the job runs. A task runs on a
    worker thread, which ``SIGINT`` cannot reach (Python delivers signals only
    to the main thread), so :func:`_raise_in_thread` raises into it; it lands at
    the next bytecode, so a blocking C call ends when it returns. A cell runs on
    the main thread (:func:`hold_cell`), so it gets a real ``SIGINT``, which
    also breaks a blocking sleep or wait. Checked and sent under the lock the
    cell is ended under, so the signal cannot reach the next cell (see
    :func:`hold_cell`). *reason* (a person's Stop) is prefixed to a task's
    error.

    **No dask cancel.** A blocking dask ``Client`` call waits in
    ``distributed.utils.sync``, whose own ``KeyboardInterrupt`` handler cancels
    that call's futures and nothing else's. A cell's ``SIGINT`` breaks the wait
    at once; a task's exception lands when the wait next wakes, within 10 s
    (hardcoded there). Cancelling every future on the client was faster for a
    task, but took a user's compute and any persisted data with it.
    """
    with _lock:
        job = _running(key)
        if job is None:
            return {"interrupted": False, "refused": "not_running"}
        if job.thread is not None:
            job.interrupted = True  # finalize as "interrupted"
            # Before the interrupt, so the task's finalizer sees it.
            job.cancel_reason = reason or job.cancel_reason
            raised = bool(_raise_in_thread(job.thread.ident, KeyboardInterrupt))
        else:
            # A cell's reason is the host's to attach (note_cancel).
            _interrupt_main()
            raised = True
    return {"interrupted": raised}


def _running(key):
    """The running job *key* names: a cell by its request, a task by its id."""
    job = _jobs.get(key)
    if job is not None and job.status == "running":
        return job
    return None


def _interrupt_main():
    """``SIGINT`` to the main thread, so a cell blocked in a sleep or a wait
    wakes up to take it.

    Only this process, where ``km.interrupt_kernel`` signals the whole group
    (dask workers included). On POSIX, to the main thread itself: a
    process-directed signal may be taken by another thread, leaving the main
    thread's wait unbroken. On Windows ``raise_signal`` runs Python's C handler,
    which sets the event a main-thread sleep waits on; ``interrupt_main`` only
    sets the flag, which waits for the sleep to end.

    Safe only while the main thread runs a request: ipykernel installs the
    ``KeyboardInterrupt`` handler for exactly that span, and outside it a
    raised SIGINT would take the default action. :func:`interrupt` holds the
    lock the cell is ended under, so the cell is still in its request.
    """
    if os.name == "posix":
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
    else:
        signal.raise_signal(signal.SIGINT)


def reset():
    """Drop every job (on every bootstrap, :func:`install`)."""
    with _lock:
        _jobs.clear()


# -- viewer wrapping --------------------------------------------------------
#
# The agent-facing ``viewer`` is wrapped by a full main-thread marshaling proxy
# (``_viewer_proxy.make_viewer_proxy``) rather than the old method-by-method
# wrap, which leaked any returned handle (``viewer.layers``, ``viewer.dims``,
# ``viewer.layers[0]``) and let off-main mutations on it segfault Qt
# (biopb/biopb#100). ``run_on_main`` above remains the marshaling primitive the
# proxy uses, and is still exposed for power users.


def install(ip):
    """Wire the job runner into the kernel: store the InteractiveShell and
    clear any prior job state."""
    global _ip
    _ip = ip
    reset()
