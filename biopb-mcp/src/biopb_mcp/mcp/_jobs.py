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
* **Several writers, serialized.** Jobs carry an ``origin`` — see
  :class:`_Job`. They share this one runner, so the rejection above is also what
  keeps the writers off each other's toes: no preemption, no queue, one ordering
  of writes to the namespace.
* **The records live in the host.** Each job's start and end are announced on
  iopub (:func:`_publish`) under the request its output is published under, and
  the host keeps the record (``_job_log``): poll, the job list, export and the
  foreign-activity digest are reads of the host's memory. This module keeps only
  what it takes to run and stop jobs.
* **One agent per kernel.** Serializing two *agents* would order their writes
  without making them mean anything — neither can see the other's model of the
  namespace. So the first non-user submitter claims the kernel and a second is
  refused (:func:`submit`); a human's cell is never gated. Everything that
  changes kernel state is gated the same way — running a job, stopping one
  (:func:`interrupt_current`), restarting the kernel (server-side) — while the
  read-only tools stay open to anyone, since they mutate nothing.
* **Main-thread affinity.** The viewer is a Qt/vispy object bound to the kernel
  main thread.  GUI mutations from the worker thread are marshaled via
  :func:`run_on_main`; ``_bootstrap`` wraps ``add_tensor`` + the ``add_*``
  family so the common paths are automatic.
* **Output.** A job thread's prints go to iopub like any cell's: ipykernel
  attributes a thread to the request that started it, which is the submit, so
  the host files them under the job. A verification announces each cell's
  start and end as well, so the host can split that one stream per cell.
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
import contextlib
import ctypes
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import Future

from ._job_log import _MAX_RETAINED_JOBS, MSG_TYPE, _one_line

logger = logging.getLogger(__name__)

# Prepended to every job so the namespace tracks the asynchronously-connecting
# tensor connection service (mirrors the old _server._REFRESH_PREFIX).
_REFRESH_PREFIX = "client = _conn.client\n"

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

# Env var carrying the number this kernel's job ids continue from: the host
# keeps the records across kernel restarts, so ids must not start over. The
# literal is mirrored in _kernel.ENV_JOB_SEQ (kept in sync by this comment).
ENV_JOB_SEQ = "BIOPB_JOB_SEQ"

# Module state, wired by install().
_ip = None
_jobs = {}  # job_id -> _Job
_job_seq = int(os.environ.get(ENV_JOB_SEQ) or 0)
_lock = threading.RLock()

# The one agent allowed to run code in this kernel, claimed by whoever submits
# first and held until the kernel restarts (see :func:`submit`). An opaque id
# supplied by the caller plus a label for the refusal message; ``None`` means
# unclaimed.
_owner = None
_owner_label = ""

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
        "run_on_main",
        "_conn",
        "_jobs",
        "_dask_client",
        "_dask_attach_done",
        "_dask_ctl",
        "_viewer_window_alive",
        "_resync_view",
    }
)


class _Job:
    __slots__ = (
        "job_id",
        "code",
        "status",
        "error_text",
        "cancel_reason",
        "interrupted",
        "origin",
        "intent",
        "thread",
        "started",
        "started_wall",
        "finished",
        "verify",
        "code_preview",
        "intent_preview",
        "request",
        "result_text",
    )

    def __init__(self, job_id, code="", origin="mcp", intent="", request=None):
        self.job_id = job_id
        # The execute request this job's output is published under: the submit
        # for an agent's job, the cell's own for a foreign client's.
        self.request = request
        # The submitted source (as passed to submit(), before the internal
        # _REFRESH_PREFIX), so the observe UI can show what each job ran.
        self.code = code
        # running | ok | error | interrupted
        self.status = "running"
        self.error_text = ""
        # The last expression's repr (_exec_capture), announced with the end.
        self.result_text = ""
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
        # Who started this job. Each value names a *surface*, not a kind of
        # actor: "agent" was two of these at once once the chat loop arrived,
        # and code asking "is this the agent's?" quietly meant "the MCP one's".
        #   "mcp"   — the execute_code tool, driven by an external MCP client
        #   "user"  — a cell run by a human from a Jupyter client on the kernel
        #   "chat"  — the in-process chat loop (docs/chat-engines.md)
        # Set at submit and never inferred later — a job outlives the request
        # that started it, and poll/export read this long after that request is
        # gone. "chat" has no writer yet and is declared ahead of one on
        # purpose: origin is the provenance an export is read by, and a value
        # introduced after the fact cannot relabel the records made without it.
        self.origin = origin
        # Why this job was run, in the words of whoever asked for it — the
        # client's own statement of purpose under "mcp", the user's turn once
        # a chat loop fills it. Free text, optional and unvalidated: it is
        # best-effort provenance for the notebook export, never a control input.
        self.intent = intent
        self.thread = None
        self.started = time.monotonic()
        # Wall-clock epoch at submit, for human-readable audit timestamps in the
        # notebook export (`started` is monotonic and not displayable).
        self.started_wall = time.time()
        self.finished = None
        # The candidate workflow this job is verifying, or None for an ordinary
        # cell. Set at submit and never after: it decides which namespace the
        # job runs in, so a job cannot become a verification once started.
        self.verify = None
        # The one-liners the gate's refusal names the running job by
        # (running_job), cut once here.
        self.code_preview = _one_line(code)
        self.intent_preview = _one_line(intent)

    def elapsed(self):
        end = self.finished if self.finished is not None else time.monotonic()
        return round(end - self.started, 3)

    def snapshot(self):
        """How the job stands, without its output, which only the host has."""
        return {
            "job_id": self.job_id,
            "request": self.request,
            "code": self.code,
            "status": self.status,
            "result_text": self.result_text,
            "error_text": self.error_text,
            "cancel_reason": self.cancel_reason,
            "origin": self.origin,
            "intent": self.intent,
            "elapsed": self.elapsed(),
            "created": self.started_wall,
            "verify": self.verify.snapshot() if self.verify is not None else None,
        }


class _Cell:
    """One cell of a verification run: its source and how it ended. Its output
    is the host's (``_job_log._CellRecord``), split from the job's stream at the
    boundaries :func:`_exec_cells` announces."""

    __slots__ = ("code", "status", "error_text", "result_text", "started", "finished")

    def __init__(self, code):
        self.code = code
        # pending | ok | error | skipped
        self.status = "pending"
        self.error_text = ""
        self.result_text = ""
        self.started = None
        self.finished = None

    def elapsed(self):
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else time.monotonic()
        return round(end - self.started, 3)


class _Verification:
    """A candidate workflow's cells, run in order in a scratch kernel. The
    record a notebook is built from is the host's (``_job_log._VerifyRecord``)."""

    __slots__ = ("title", "cells", "created")

    def __init__(self, title, cells):
        self.title = title
        self.cells = [_Cell(code) for code in cells]
        self.created = time.time()

    def snapshot(self):
        return {
            "title": self.title,
            "created": self.created,
            "cells": [
                {
                    "code": c.code,
                    "status": c.status,
                    "error_text": c.error_text,
                    "result_text": c.result_text,
                }
                for c in self.cells
            ],
        }


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


def _exec_cells(job, verification):
    """Run *verification*'s cells in order in one scratch namespace.

    **Stops at the first failure.** The cells after it were written against the
    state the failed one was supposed to produce, so running them anyway reports
    a cascade of consequences as if they were separate defects. The remainder is
    marked ``skipped`` rather than dropped, so the report says how far the
    workflow got.

    Each cell's start and end are announced (:func:`_publish` flushes the
    streams first), so the host files the output between them under that cell,
    as well as under the job. The failure is re-raised so the job's own
    finalizer sets the status and does the interrupt attribution; there is one
    place that decides how a job ended, and this is not it.
    """
    # The kernel's own namespace, because this only ever runs in a scratch
    # kernel: a process spawned for this verification and discarded after it
    # (`_scratch`). The isolation that used to be a filtered dict is the process
    # boundary now, which is what extends it past bindings to the viewer,
    # `sys.modules`, and anything a cell mutates in place.
    ns = _ip.user_ns if _ip is not None else {}
    try:
        for index, cell in enumerate(verification.cells):
            _publish({"event": "cell_start", "job_id": job.job_id, "index": index})
            cell.started = time.monotonic()
            try:
                # No refresh prefix, unlike an ordinary cell: a verification
                # runs the document and nothing else. `client = _conn.client`
                # prepended here would bind a handle the document never asks
                # for, so a workflow that forgot to build its own would pass
                # and then fail for its reader -- the one defect this exists to
                # catch. The document's own first cell calls `workflow_env()`.
                _exec_capture(cell.code, ns, cell)
                cell.status = "ok"
            except BaseException:
                cell.status = "error"
                cell.error_text = traceback.format_exc()
                raise
            finally:
                cell.finished = time.monotonic()
                _publish(
                    {
                        "event": "cell_end",
                        "job_id": job.job_id,
                        "index": index,
                        "status": cell.status,
                        "error_text": cell.error_text,
                        "result_text": cell.result_text,
                        "elapsed": cell.elapsed(),
                    }
                )
    finally:
        # Whatever ended the run -- a failing cell, an interrupt -- the cells it
        # never reached are still `pending`. Relabel them here rather than in the
        # loop, which the raise leaves for good the moment there is anything to
        # relabel.
        for cell in verification.cells:
            if cell.status == "pending":
                cell.status = "skipped"


def _dask_backstop():
    """Why nothing this job computes could finish, or ``None``.

    biopb/biopb#970's backstop: an attached scheduler whose workers have all gone
    (a host suspend outliving their TTL) still *accepts* work and never runs it,
    turning a 0.2 s read into a cell that hangs until someone interrupts it. A
    worker count off the client makes that an error naming the fix instead. Only
    fires while attached, which is opt-in (``_dask_ctl.attach()``).
    """
    ctl = _ip.user_ns.get("_dask_ctl") if _ip is not None else None
    if ctl is None:
        return None
    try:
        return ctl.dead_message()
    except Exception:  # noqa: BLE001 - a backstop must not become the failure
        logger.debug("dask liveness probe failed", exc_info=True)
        return None


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
    Session (its msg_id counter) is not thread-safe, and a job event is sent
    from a job thread. Streams are flushed first -- ``flush`` waits until the
    IOPub thread has taken the output -- so the job's last output is on iopub
    ahead of its end.
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
            kernel.session.send(iopub, MSG_TYPE, content, ident=MSG_TYPE.encode())
        except Exception:  # noqa: BLE001 - a lost event must not fail the job
            logger.debug("job event not published", exc_info=True)

    iopub.schedule(send)


def _publish_start(job):
    _publish(
        {
            "event": "start",
            "job_id": job.job_id,
            "request": job.request,
            "origin": job.origin,
            "intent": job.intent,
            "code": job.code,
            "created": job.started_wall,
            # A verification's cells, so the host can hold a record per cell.
            "verify": (
                {
                    "title": job.verify.title,
                    "cells": [c.code for c in job.verify.cells],
                    "created": job.verify.created,
                }
                if job.verify is not None
                else None
            ),
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


def _run(job, code):
    exc = None
    try:
        _dead = _dask_backstop()
        if _dead:
            raise RuntimeError(_dead)
        if job.verify is not None:
            _exec_cells(job, job.verify)
        else:
            _exec_capture(_REFRESH_PREFIX + code, _ip.user_ns, job)
    except KeyboardInterrupt:
        exc = True
        job.error_text = traceback.format_exc()
        # A KeyboardInterrupt this runner did not cause was delivered from
        # outside: a SIGINT to the kernel, relayed here because the job was
        # inside a run_on_main slot on the main thread when it landed. It is a
        # *stop*, not a defect in the submitted code, so label and attribute it
        # rather than hand back a bare traceback -- the same reasoning that gave
        # interrupt_current its flag, applied to the door it does not own.
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
        _publish_end(job)


def _foreign(job, for_origin):
    """Whether *job* was written by someone other than *for_origin*'s client.

    "Someone else's cell" is a relation, not a property: the chat loop submits
    as ``chat``, and a fixed "is it the MCP client's?" refused the loop its own
    cell (biopb/biopb#880).
    """
    return job.origin != for_origin


def _prune():
    # Oldest terminal first. Only the host's copy is the record anyone reads
    # (_job_log, which also holds unseen foreign jobs against eviction); this
    # one serves the refusal and the stop, and a verification's poll.
    terminal = [jid for jid, j in _jobs.items() if j.status != "running"]
    while len(_jobs) > _MAX_RETAINED_JOBS and terminal:
        del _jobs[terminal.pop(0)]


def _new_job(code, origin, intent="", request=None):
    """Register a job record under the next id. Call with `_lock` held."""
    global _job_seq
    _job_seq += 1
    job = _Job(f"job-{_job_seq}", code, origin=origin, intent=intent, request=request)
    _jobs[job.job_id] = job
    return job


def submit(
    code,
    origin="mcp",
    intent="",
    writer=None,
    writer_label="",
    verify_cells=None,
    verify_title="",
):
    """Start *code* in a background thread; return ``{"job_id": ...}`` or, if a
    job is already running, ``{"error": "busy", "running_job_id": ...,
    "running_job_origin": ...}``.

    **Verification runs come through this same door**, but only ever in a
    *scratch kernel*. With *verify_cells* — a list of cell sources — the job runs
    them in order in this kernel's own namespace instead of running *code*, and
    carries a :class:`_Verification` record. The session child spawns a kernel
    per verification and discards it (``_scratch``), so "this kernel's own
    namespace" is a fresh one and the isolation covers the viewer and
    ``sys.modules`` too. Submitting *verify_cells* to a session kernel would run
    the cells in the user's namespace; nothing does.

    *origin* and *intent* are recorded on the job and never acted on beyond the
    rules in :class:`_Job`; see there for the origin vocabulary. The busy return
    carries the running job's origin because the caller's advice depends on it:
    the agent may stop its *own* job, but another writer's is not its to stop.

    **One agent per kernel.** *writer* is an opaque id for the client asking.
    The first non-user submitter claims the kernel; a later submit under a
    *different* id is refused with ``{"error": "not_owner", "owner": <label>}``.
    Two agents sharing one namespace is not a race the runner can serialize away
    — the writes land in a defined order and still mean nothing, because neither
    agent can see the other's model of what the variables and layers are. So it
    fails loudly at the door instead. The claim is dropped by :func:`reset`, i.e.
    it lasts for the life of the kernel.

    Two deliberate holes. A **human** cell (``origin="user"``) is never gated:
    the person at the machine has standing here that no client does, and the
    Jupyter client they typed it in has no identity to gate on anyway. And a
    caller with ``writer=None`` — a direct in-process call, or a transport that
    yields no client id — neither claims nor is checked, since there is nothing
    to tell two of them apart with.

    **The recovery belongs to the human, not to a second agent.** Every tool
    that changes kernel state is gated the same way — ``interrupt_current`` here,
    ``restart_kernel`` server-side — so a client that does not hold the kernel
    cannot take it by force; it keeps the read-only tools and nothing else. What
    frees a claim is the kernel going away: the person at the machine restarting
    from the observe page (never gated), or the session ending. That is the same
    principle as the ``origin="user"`` exemption, applied to recovery.
    """
    global _owner, _owner_label
    with _lock:
        if origin != "user" and writer is not None:
            if _owner is None:
                _owner, _owner_label = writer, writer_label
            elif writer != _owner:
                # The id as well as the label: the caller mirrors the claim, and
                # a refusal is its chance to correct a mirror that guessed wrong.
                return {
                    "error": "not_owner",
                    "owner": _owner_label,
                    "owner_id": _owner,
                }
        for jid, j in _jobs.items():
            if j.status == "running":
                return {
                    "error": "busy",
                    "running_job_id": jid,
                    "running_job_origin": j.origin,
                }
        if verify_cells is not None:
            # The record's cells are the source of truth; `code` is derived from
            # them so the audit view of this job cannot disagree with the
            # workflow view of it.
            code = "\n\n# ---\n\n".join(verify_cells)
        job = _new_job(code, origin, intent, request=_request_id())
        if verify_cells is not None:
            job.verify = _Verification(verify_title, verify_cells)
        _prune()
        # Before the thread starts, so the host has the record before any of
        # the job's output reaches it.
        _publish_start(job)
        thread = threading.Thread(
            target=_run, args=(job, code), name=job.job_id, daemon=True
        )
        job.thread = thread
        thread.start()
        return {"job_id": job.job_id, "status": "running"}


@contextlib.contextmanager
def record_inline(code, request=None, origin="user"):
    """Record a cell running inline on this thread as a job, for its duration.

    For a foreign client's cell (``_kernel_gate``): it runs on the main thread
    as its client expects, and the record is what tells the agent about it, the
    same as a job from :func:`submit`. Its output goes to its own client as
    usual, under *request*, which is how the host files it too. The caller
    settles ``status`` (and ``error_text``) from the reply before leaving the
    block; a block that raises instead is an ``error``.

    Not gated here: the caller refuses while a job runs, and nothing can start
    one meanwhile, since a submit is an execute request queued behind this one.
    """
    with _lock:
        job = _new_job(code, origin, request=request)
        _prune()
    _publish_start(job)
    try:
        yield job
    finally:
        job.finished = time.monotonic()
        if job.status == "running":
            job.status = "error"
        _publish_end(job)


def poll(job_id):
    """*job_id*'s snapshot from this kernel's own copy: how it stands, without
    output. The host's records (``_job_log``) come from iopub, which can drop a
    message; this comes back on the shell reply, which cannot, so the host
    settles a disagreement from it (``JobLog.settle``)."""
    job = _jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "unknown", "error_text": ""}
    return job.snapshot()


def status(job_id):
    """*job_id*'s status alone: what the host checks its records against, often
    enough that the whole snapshot would be waste."""
    job = _jobs.get(job_id)
    return "unknown" if job is None else job.status


def _cancel_dask_futures(job, reason=None):
    """Stop *job*'s in-flight dask work, tagging why.

    Takes the job rather than its id: the one caller
    (:func:`interrupt_current`) has already resolved it and established that it
    is running, and re-deriving both here only created return values -- an
    "unknown" job, a non-running one -- that no caller could observe.
    """
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
    # Whatever client is live, not the `_dask_client` binding: a cell that made
    # its own `Client(...)` is attached just as much as `_dask_ctl.attach()` is,
    # and its futures are just as stuck. `current` finds the global default,
    # which is what dask itself computes on (raises ValueError when there is
    # none, i.e. the in-process default).
    try:
        from distributed import Client, Future

        dc = Client.current(allow_global=True)
        keys = list(dc.futures)
        if keys:
            dc.cancel([Future(k, dc) for k in keys], force=True)
    except Exception:  # noqa: BLE001 - cancel is best-effort
        logger.debug("distributed cancel failed", exc_info=True)


def _running_job():
    """The single running job, or None. One job at a time (see submit())."""
    for j in _jobs.values():
        if j.status == "running":
            return j
    return None


def running_job():
    """The running job's id, origin, one-line intent and code, and elapsed
    seconds, or ``None``.

    Two readers. The session child's cross-kernel admission check: a
    verification runs in a *second* kernel, which this one cannot see, so the
    rule that only one job runs at a time has to be decided a level up
    (``_scratch``). And ``_kernel_gate``'s refusal, which names the job.
    """
    job = _running_job()
    if job is None:
        return None
    return {
        "job_id": job.job_id,
        "origin": job.origin,
        "intent": job.intent_preview,
        "code": job.code_preview,
        "elapsed": job.elapsed(),
    }


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


def interrupt_current(reason=None, origin="user", writer=None):
    """Force-stop the running job: cooperative cancel *plus* a ``KeyboardInterrupt``
    raised directly into the job's worker thread.

    ``SIGINT`` can't do this — Python delivers signals only to the kernel main
    thread, while the job runs in a background worker — so a pure-Python loop
    would otherwise be stoppable only by ``restart_kernel``. This first runs
    :func:`_cancel` (attribution reason + in-flight dask-future cancel), then
    forces the worker thread via :func:`_raise_in_thread`. The exception lands at
    the next bytecode, so a blocking C call ends when it returns. ``{"interrupted":
    False, "status": "idle"}`` when the kernel is idle.

    *origin* is the asking client's own job origin, in the vocabulary the jobs
    themselves are recorded in: ``"user"`` (the observe UI, the default: a
    person may stop anything running in their own session), ``"mcp"`` for a
    remote client, ``"chat"`` for the in-process loop. A **client is refused a
    job it did not start** (``{"refused": "foreign_job"}``): the stop would be
    silent, since attribution runs one way only — a user stop reaches it through
    ``cancel_reason``, but the other writer would see nothing beyond an
    unexplained ``interrupted`` badge. The human has the observe UI and can stop
    their own work; a program has no consent to.

    "Did not start" is therefore relative to the asker, which is why this takes
    an origin rather than a fixed "is it the MCP client?" flag: the flag refused
    the chat loop the cell it had just run itself (biopb/biopb#880), on the one
    kernel where the interrupt is not guaranteed.

    *writer* is the asking client's id, checked against the kernel's one-agent
    claim (:func:`submit`): a client that does not hold this kernel cannot stop
    what runs in it (``{"refused": "not_owner"}``). Stopping a job is a change to
    kernel state, so it is gated like running one; only the read-only tools stay
    open to a second client. As in :func:`submit`, a caller with ``writer=None``
    is not checked — there is nothing to compare.
    """
    job = _running_job()
    if job is None:
        return {"job_id": None, "interrupted": False, "status": "idle"}
    if origin != "user" and writer is not None and _owner not in (None, writer):
        return {
            "job_id": job.job_id,
            "interrupted": False,
            "status": "running",
            "refused": "not_owner",
        }
    if origin != "user" and _foreign(job, origin):
        return {
            "job_id": job.job_id,
            "interrupted": False,
            "status": "running",
            "refused": "foreign_job",
            # Whose job it is, so the caller can name the writer. "Foreign" is
            # no longer a synonym for "the user's" -- see _foreign().
            "origin": job.origin,
        }
    job.interrupted = True  # finalize as "interrupted"
    _cancel_dask_futures(job, reason=reason)
    ident = job.thread.ident if job.thread is not None else None
    raised = _raise_in_thread(ident, KeyboardInterrupt)
    return {"job_id": job.job_id, "interrupted": bool(raised)}


def owner():
    """``{"owner": <id or None>, "label": <str>}`` — who holds this kernel."""
    with _lock:
        return {"owner": _owner, "label": _owner_label}


def reset():
    """Drop all job records and the kernel's agent claim (used on kernel restart
    / re-bootstrap).

    Releasing here is what makes the claim last exactly one kernel lifetime:
    :func:`install` calls this on every bootstrap, and a hard restart replaces
    the process and its module state outright.
    """
    global _owner, _owner_label
    with _lock:
        _jobs.clear()
        _owner, _owner_label = None, ""


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
