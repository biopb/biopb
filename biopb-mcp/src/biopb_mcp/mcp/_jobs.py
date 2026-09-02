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
  of writes to the namespace. :func:`foreign_digest` is how the ``execute_code``
  agent finds out its namespace changed under it; see ``docs/user-console.md``.
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

# Keep at most this many terminal job records before evicting the oldest. The
# ceiling is what a workflow can be *reconstructed* from: rewriting a session
# into a clean program (:func:`submit` with ``verify_cells``) reads the
# transcript, so eviction takes away the source material for the one step
# nothing can automate. Raised from 32 once _MAX_JOB_OUTPUT_CHARS bounded a
# single record -- until then one runaway cell grew without limit and a record
# count bounded nothing.
_MAX_RETAINED_JOBS = 200

# Keep at most this many characters of one job's captured output. This is the
# bound _MAX_RETAINED_JOBS is not: that caps how many records are kept, while a
# single cell printing in a loop grew its buffer without limit, so 32 records
# bounded nothing. Well above observe's 20k display cap, so a truncated *view*
# still means "there is more in the record" rather than "the record ends here".
_MAX_JOB_OUTPUT_CHARS = 200_000

# How much of the front of a stream `write_output` copies aside so that
# `output_head` can find the first line without rebuilding the buffer. Two
# orders of magnitude above the 80-char line it has to find, so the only text
# it can miss is a first line already too long to survive the cap anyway.
_HEAD_SCAN_CHARS = 4096

# Attribution for a KeyboardInterrupt this runner did not raise (see _run). The
# kernel ignores SIGINT except while servicing a message (ipykernel installs
# default_int_handler only between its pre/post handler hooks), so the realistic
# source is the one place that sends one: KernelHost._run_once interrupting the
# kernel when a *quick* snippet overruns its timeout.
_EXTERNAL_INTERRUPT_MSG = (
    "Stopped by an interrupt sent to the whole kernel, not by an error in this "
    "code. Most likely a short tool call (server_status / poll_job / a "
    "screenshot) overran its timeout and interrupted the kernel to unwedge it."
)

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
        "_viewer_window_alive",
        "_resync_view",
    }
)


def _dropped_marker(n):
    """The line `output` prepends once the cap has discarded a head.

    One spelling, because `output_head` has to recognise the same sentence it
    would otherwise have to re-derive from the rebuilt text.
    """
    return f"...({n} earlier chars dropped)..."


class _OutputBuffer:
    """Capped stdout capture, plus the last expression's repr.

    Shared by a job and by one cell of a verification run, because the two are
    written to through the same two doors: :class:`_JobStream` routes a thread's
    prints to whichever is bound to that thread, and :func:`_exec_capture`
    stores the last expression's repr on whatever it is handed. One cap, one
    dropped-head marker, one monotonic total -- so a cell reports its output the
    way a job does without either having to remember to.
    """

    __slots__ = ("stdout", "stdout_dropped", "result_text", "head_prefix")

    def __init__(self):
        self.stdout = io.StringIO()
        # Characters the cap has discarded from the front of `stdout`. Kept so
        # the record can say it is partial and so a reader tracking growth has
        # a number that only ever increases (see `output_total`).
        self.stdout_dropped = 0
        self.result_text = ""
        # A bounded copy of the start of the stream, so `output_head` never has
        # to rebuild the whole buffer to answer with one line. See there.
        self.head_prefix = ""

    def write_output(self, s):
        """Append captured output, keeping at most the newest cap-worth.

        The tail survives, for the reason the detail view keeps the tail: while
        a cell is still running the newest output is the informative part.

        Compacted at twice the cap rather than on every write, so the rewrite
        happens once per cap-worth of output instead of once per print. Only the
        job's own worker thread writes here (`_jobs_by_thread` is keyed by
        thread), so no lock: a reader racing the swap gets the pre-compaction
        buffer, which is longer but never torn.
        """
        if len(self.head_prefix) < _HEAD_SCAN_CHARS:
            self.head_prefix += s[: _HEAD_SCAN_CHARS - len(self.head_prefix)]
        n = self.stdout.write(s)
        if self.stdout.tell() > 2 * _MAX_JOB_OUTPUT_CHARS:
            text = self.stdout.getvalue()
            keep = text[-_MAX_JOB_OUTPUT_CHARS:]
            self.stdout_dropped += len(text) - len(keep)
            buf = io.StringIO()
            buf.write(keep)
            self.stdout = buf
        return n

    def output(self):
        """The captured output, marked when the cap dropped its head.

        The marker is added on read rather than stored, so it cannot itself be
        compacted away later, and so every consumer -- the agent's poll, the
        observe detail, the notebook cell -- says the same thing without each
        having to remember to.
        """
        text = self.stdout.getvalue()
        if not self.stdout_dropped:
            return text
        return _dropped_marker(self.stdout_dropped) + "\n" + text

    def output_total(self):
        """Everything this buffer has ever taken, including what was dropped.

        Monotonic, which `len(stdout)` is not once the cap compacts. A reader
        streaming the output as it grows has to diff against this.

        `tell()` rather than `len(getvalue())`: the buffer is append-only, so
        the two agree, but `getvalue()` copies it -- and `jobs_summary` asks
        every retained job for this on each ~1s observe poll.
        """
        return self.stdout_dropped + self.stdout.tell()

    def output_head(self, limit=80):
        """First non-blank line of :meth:`output`, without rebuilding it.

        `_one_line(self.output())` would copy the whole capped buffer (up to
        `_MAX_JOB_OUTPUT_CHARS`) to keep 80 characters, once per cell per poll
        while a verification runs. `write_output` keeps the first
        `_HEAD_SCAN_CHARS` instead, which is where a first line short enough to
        survive `limit` must be. An opening run of whitespace longer than that
        scan reports no head rather than a later line -- a preview field, and
        nothing prints 4 KB of blanks before its first word.
        """
        if self.stdout_dropped:
            # The real head is gone; say so, the way `output` does.
            return _one_line(_dropped_marker(self.stdout_dropped), limit)
        return _one_line(self.head_prefix, limit)


class _Job(_OutputBuffer):
    __slots__ = (
        "job_id",
        "code",
        "status",
        "error_text",
        "cancel_reason",
        "interrupted",
        "origin",
        "intent",
        "seen_by_agent",
        "thread",
        "started",
        "started_wall",
        "finished",
        "verify",
        "code_preview",
        "intent_preview",
    )

    def __init__(self, job_id, code="", origin="mcp", intent=""):
        super().__init__()
        self.job_id = job_id
        # The submitted source (as passed to submit(), before the internal
        # _REFRESH_PREFIX), so the observe UI can show what each job ran.
        self.code = code
        # running | ok | error | interrupted
        self.status = "running"
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
        # Who started this job. Each value names a *surface*, not a kind of
        # actor: "agent" was two of these at once once the chat loop arrived,
        # and code asking "is this the agent's?" quietly meant "the MCP one's".
        #   "mcp"   — the execute_code tool, driven by an external MCP client
        #   "user"  — a cell run by a human from the observe page
        #   "chat"  — the in-process chat loop (docs/chat-client-evaluation.md)
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
        # Whether the execute_code agent has been told about this *foreign* job
        # (via foreign_digest). Unset on the agent's own jobs, which need no
        # notice. One flag, because there is one reader; a second in-process
        # reader would need this per-reader, not a bool.
        self.seen_by_agent = False
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
        # The one-liners `jobs_summary` shows, cut once here rather than on
        # every observe poll: `code` and `intent` are fixed at submit, and the
        # summary re-split every retained job's full source at ~1 Hz.
        self.code_preview = _one_line(code)
        self.intent_preview = _one_line(intent)

    def elapsed(self):
        end = self.finished if self.finished is not None else time.monotonic()
        return round(end - self.started, 3)

    def snapshot(self):
        return {
            "job_id": self.job_id,
            "code": self.code,
            "status": self.status,
            "stdout": self.output(),
            "stdout_dropped": self.stdout_dropped,
            "stdout_total": self.output_total(),
            "result_text": self.result_text,
            "error_text": self.error_text,
            "cancel_reason": self.cancel_reason,
            "origin": self.origin,
            "intent": self.intent,
            "elapsed": self.elapsed(),
            "created": self.started_wall,
            # Present only on a verification run, so an ordinary poll is
            # unchanged and a client that predates this ignores the key. Light:
            # a job snapshot is the *polled* shape, and the cells' output is
            # already here once, in `stdout` (see _Cell.snapshot).
            "verify": self.verify.snapshot() if self.verify is not None else None,
        }


class _Cell(_OutputBuffer):
    """One cell of a verification run: its source, its outcome, its output.

    Prints are teed to the owning job as well as kept here, because the two
    readers want different cuts of the same stream: the notebook needs the
    output split per cell, and ``poll_job`` on a long verification needs the
    whole run's output accumulating in one place, the way it does for any other
    job.
    """

    __slots__ = ("code", "status", "error_text", "job", "started", "finished")

    def __init__(self, code, job):
        super().__init__()
        self.code = code
        self.job = job
        # pending | ok | error | skipped
        self.status = "pending"
        self.error_text = ""
        self.started = None
        self.finished = None

    def write_output(self, s):
        self.job.write_output(s)
        return super().write_output(s)

    def elapsed(self):
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else time.monotonic()
        return round(end - self.started, 3)

    def snapshot(self, full=False):
        """This cell's outcome; *full* adds the captured output.

        The output is the expensive half, and not because building the dict
        costs anything -- it is that this crosses a JSON round trip out of the
        kernel every 0.4s while a verification runs. Shipping every cell's
        output there sends the same bytes the job's own teed buffer already
        carries, once more per cell: a 20-cell run polled 1.2 MB where an
        ordinary job polls 200 KB, growing linearly with the workflow.

        So the polled snapshot carries a one-line head and a length, the way
        :func:`jobs_summary` does for a job, and the text is read once with
        ``full=True`` by :func:`verified` when the notebook is built. The
        ledger a report prints needs no more than the head; the notebook needs
        all of it, and asks for it exactly once.
        """
        snap = {
            "code": self.code,
            "status": self.status,
            "error_text": self.error_text,
            "elapsed": self.elapsed(),
            "stdout_len": self.output_total(),
            "stdout_head": self.output_head(),
        }
        if full:
            snap["stdout"] = self.output()
            snap["result_text"] = self.result_text
        return snap


class _Verification:
    """A candidate workflow, its cells, and what running them in a scratch
    namespace did.

    The record a workflow notebook is built from. It is deliberately *not* a
    list of job ids: the program that works is a rewrite of the transcript, not
    a selection from it — a cell that created a variable and a later cell that
    corrected its value merge into one, and neither keeping nor dropping either
    original gives a runnable document. So the cells here are the agent's own
    text, and what makes them trustworthy is that they ran.
    """

    __slots__ = ("title", "cells", "created")

    def __init__(self, title, cells, job):
        self.title = title
        self.cells = [_Cell(code, job) for code in cells]
        self.created = time.time()

    def status(self):
        """``ok`` once every cell ran, ``error`` at the first failure."""
        if any(c.status == "error" for c in self.cells):
            return "error"
        if self.cells and all(c.status == "ok" for c in self.cells):
            return "ok"
        return "running"

    def snapshot(self, full=False):
        """The record; *full* carries each cell's captured output (see
        :meth:`_Cell.snapshot`)."""
        return {
            "title": self.title,
            "created": self.created,
            "status": self.status(),
            "cells": [c.snapshot(full=full) for c in self.cells],
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
            return job.write_output(s)
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

    Prints route to the current cell — teed to the job — by rebinding this
    thread's ``_jobs_by_thread`` entry around each one. The failure is re-raised
    so the job's own finalizer sets the status and does the interrupt
    attribution; there is one place that decides how a job ended, and this is
    not it.
    """
    ident = threading.get_ident()
    # The kernel's own namespace, because this only ever runs in a scratch
    # kernel: a process spawned for this verification and discarded after it
    # (`_scratch`). The isolation that used to be a filtered dict is the process
    # boundary now, which is what extends it past bindings to the viewer,
    # `sys.modules`, and anything a cell mutates in place.
    ns = _ip.user_ns if _ip is not None else {}
    try:
        for cell in verification.cells:
            _jobs_by_thread[ident] = cell
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
                _jobs_by_thread[ident] = job
    finally:
        # Whatever ended the run -- a failing cell, an interrupt -- the cells it
        # never reached are still `pending`. Relabel them here rather than in the
        # loop, which the raise leaves for good the moment there is anything to
        # relabel.
        for cell in verification.cells:
            if cell.status == "pending":
                cell.status = "skipped"


def _run(job, code):
    _jobs_by_thread[threading.get_ident()] = job
    exc = None
    try:
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
        # Sharpest for a user cell: the agent is refused interrupt_current on
        # one, yet an overrunning tool probe can still end it this way, and
        # unlabeled it reads to the human as their own code breaking.
        if not job.interrupted:
            job.interrupted = True
            job.cancel_reason = job.cancel_reason or _EXTERNAL_INTERRUPT_MSG
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


def _foreign(job, for_origin="mcp"):
    """Whether *job* was written by someone other than *for_origin*'s client.

    The rules this serves — the digest and the eviction hold — are about *whose*
    job it is from that client's point of view, and both were written when "not
    the agent" and "the user" were the same set. They are not once a chat loop
    submits, so the test is spelled out here rather than inlined as
    ``origin == "user"``.

    Deliberately not the interrupt's question, which is "is this the *asker's*
    job?" — see :func:`interrupt_current`. Answering it with this one refused
    the chat loop its own cell.

    *for_origin* is the asking client's own origin, because "someone else's
    cell" is a relation, not a property: the chat loop submits as ``chat``, so
    reading the digest from the MCP client's fixed point of view reported the
    loop its own cells back to it as another writer's.
    """
    return job.origin != for_origin


def _prune():
    # Evict oldest-first, but never a foreign job the agent has not been told
    # about yet: that digest entry is the agent's only notice that its namespace
    # changed under it, and evicting the record silently drops the notice. So
    # the cap can be exceeded — bounded by how many cells another writer runs
    # between two agent calls, which is small.
    terminal = [
        jid
        for jid, j in _jobs.items()
        if j.status != "running" and not (_foreign(j) and not j.seen_by_agent)
    ]
    while len(_jobs) > _MAX_RETAINED_JOBS and terminal:
        del _jobs[terminal.pop(0)]


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
    observe console has no identity to gate on anyway. And a caller with
    ``writer=None`` — a direct in-process call, or a transport that yields no
    client id — neither claims nor is checked, since there is nothing to tell
    two of them apart with.

    **The recovery belongs to the human, not to a second agent.** Every tool
    that changes kernel state is gated the same way — ``interrupt_current`` here,
    ``restart_kernel`` server-side — so a client that does not hold the kernel
    cannot take it by force; it keeps the read-only tools and nothing else. What
    frees a claim is the kernel going away: the person at the machine restarting
    from the observe page (never gated), or the session ending. That is the same
    principle as the ``origin="user"`` exemption, applied to recovery.
    """
    global _job_seq, _owner, _owner_label
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
        # Re-assert the thread-aware stream wrap (idempotent) so a job thread's
        # output is captured even if something replaced sys.stdout since
        # install() — and so it works under pytest's per-phase capture.
        _install_streams()
        for jid, j in _jobs.items():
            if j.status == "running":
                return {
                    "error": "busy",
                    "running_job_id": jid,
                    "running_job_origin": j.origin,
                }
        _job_seq += 1
        job_id = f"job-{_job_seq}"
        if verify_cells is not None:
            # The record's cells are the source of truth; `code` is derived from
            # them so the audit view of this job cannot disagree with the
            # workflow view of it.
            code = "\n\n# ---\n\n".join(verify_cells)
        job = _Job(job_id, code, origin=origin, intent=intent)
        if verify_cells is not None:
            job.verify = _Verification(verify_title, verify_cells, job)
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
    dc = _ip.user_ns.get("_dask_client") if _ip is not None else None
    if dc is not None:
        try:
            from distributed import Future

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
    """``{"job_id": ..., "origin": ...}`` for the running job, or ``None``.

    The session child's cross-kernel admission check reads this: a verification
    runs in a *second* kernel, which this one cannot see, so the rule that only
    one job runs at a time has to be decided a level up (``_scratch``).
    """
    job = _running_job()
    if job is None:
        return None
    return {"job_id": job.job_id, "origin": job.origin}


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


def interrupt_current(reason=None, requester="user", writer=None):
    """Force-stop the running job: cooperative cancel *plus* a ``KeyboardInterrupt``
    raised directly into the job's worker thread.

    ``SIGINT`` can't do this — Python delivers signals only to the kernel main
    thread, while the job runs in a background worker — so a pure-Python loop
    would otherwise be stoppable only by ``restart_kernel``. This first runs
    :func:`_cancel` (attribution reason + in-flight dask-future cancel), then
    forces the worker thread via :func:`_raise_in_thread`. The exception lands at
    the next bytecode, so a blocking C call ends when it returns. ``{"interrupted":
    False, "status": "idle"}`` when the kernel is idle.

    *requester* is who is asking — ``"user"`` (the observe UI, the default: a
    person may stop anything running in their own session) or ``"mcp"``. An
    **MCP client is refused a job it did not start** (``{"refused":
    "foreign_job"}``): the stop would be silent, since attribution runs one way
    only — a user stop reaches it through ``cancel_reason``, but the other
    writer would see nothing beyond an unexplained ``interrupted`` badge. The
    human has the observe UI and can stop their own work; a program has no
    consent to.

    Note the test is against *the MCP client*, not against each writer and its
    own work: a second writer asking as ``"mcp"`` would be refused its own cell.
    Nothing does that today — the chat loop's cancel stops its turn and leaves
    the cell to the human, exactly as an MCP client's does. Worth knowing before
    adding a programmatic interrupt for a writer that is not this one.

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
    if requester == "mcp" and writer is not None and _owner not in (None, writer):
        return {
            "job_id": job.job_id,
            "interrupted": False,
            "status": "running",
            "refused": "not_owner",
        }
    if requester == "mcp" and _foreign(job):
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


def _one_line(text, limit=80):
    """First non-blank line of *text*, trimmed and length-capped.

    Keeps jobs_summary light (the full source and the full intent are both in
    the per-job snapshot) while giving each list row an identifying one-liner.
    """
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def jobs_summary():
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "origin": j.origin,
            "elapsed": j.elapsed(),
            "stdout_len": j.output_total(),
            "code_preview": j.code_preview,
            # Why the cell was run, when whoever ran it said. The observe list
            # prefers it over the code line: "isolate the nuclei channel" tells
            # the person watching what is happening to their data, and
            # `arr = arr[..., 1]` makes them reconstruct it.
            "intent_preview": j.intent_preview,
        }
        for j in _jobs.values()
    ]


def foreign_digest(for_origin="mcp"):
    """Jobs the asking agent did not start and has not been told about yet,
    oldest-first.

    Returns ``[{"job_id", "status", "elapsed", "origin"}, ...]``; *origin* is
    carried because the caller words the notice differently for a person than
    for another agent. This is the agent's only notice that a second writer
    touched its namespace — a redefined variable, a deleted layer — so it is
    read on every agent-facing round trip and rendered into that call's result
    (``_server._foreign_activity_note``). Pull, not push:
    an MCP server->client notification is not reliably surfaced mid-turn, and
    when the agent is idle there is no turn to interrupt.

    *for_origin* is the asking client's own job origin -- ``"mcp"`` for a remote
    client, ``"chat"`` for the in-process loop -- so each is told about the
    *other* writers rather than about itself. ``seen_by_agent`` stays a single
    flag because the kernel's one-agent claim makes the two mutually exclusive:
    only one of them is ever the agent being promised a notice exactly once.

    A pure read: marking entries reported is :func:`ack_foreign_digest`, a
    **separate** call the caller makes only once the notice has actually reached
    it. Acking here instead would consume the notice on a round trip whose reply
    never arrived — ``execute_interactive`` sends before it starts its clock, so
    a probe that times out is still queued at the kernel and runs when the main
    thread frees up, setting the flag for a note nobody received.
    """
    with _lock:
        return [
            {
                "job_id": j.job_id,
                "status": j.status,
                "elapsed": j.elapsed(),
                "origin": j.origin,
            }
            for j in _jobs.values()
            if _foreign(j, for_origin) and not j.seen_by_agent
        ]


def ack_foreign_digest(job_ids, writer=None):
    """Mark the jobs in *job_ids* as reported; return how many were marked.

    **Only the kernel's owner can discharge a notice.** Reading the digest is
    open to anyone — a second client watching the session is welcome to see that
    a cell ran — but ``seen_by_agent`` records that *the agent working here* has
    been told, and it is promised the notice exactly once. A bystander's
    ``poll_job`` acking it would retire a notice the owner never received, which
    is the one failure this whole split exists to prevent. A caller with
    ``writer=None`` is the in-process case and acks as before; an unclaimed
    kernel has no owner to defer to.

    *job_ids* is what the caller actually told the agent **and reported as
    terminal** — never the whole digest. The status is deliberately **not**
    consulted here: a job reported ``running`` that finished a moment later must
    stay pending, because "job-7 ran (running)" is not the final status
    the agent is promised exactly once. Re-reading the status instead would ack
    precisely that job and retire it unheard — the race this split exists to
    close. A status is monotone into terminal, so an id reported terminal is
    still terminal now; no re-check can add information.
    """
    wanted = set(job_ids)
    with _lock:
        if writer is not None and _owner not in (None, writer):
            return 0
        acked = 0
        for job in _jobs.values():
            if _foreign(job) and not job.seen_by_agent and job.job_id in wanted:
                job.seen_by_agent = True
                acked += 1
        return acked


def export():
    """Full snapshots of all retained jobs, oldest-first, for notebook export.

    A read like :func:`jobs_summary` (round-tripped on the kernel main thread, no
    background job thread), but carrying each job's *full* source and captured
    output so the observe UI can serialize the session to a Jupyter notebook.
    """
    return [j.snapshot() for j in _jobs.values()]


def verify_record(job_id):
    """*job_id*'s verification record with every cell's full output, or ``None``.

    The other half of the polled/full split (:meth:`_Cell.snapshot`): a poll
    ships a head and a length once every 0.4 s, and this is read **once**, when
    the run ends, for the document the record exists to become. The session child
    calls it before discarding the scratch kernel -- after which there is nobody
    left to ask.
    """
    job = _jobs.get(job_id)
    if job is None or job.verify is None:
        return None
    return job.verify.snapshot(full=True)


def jobs_view():
    """``{"jobs": [...]}`` for the observe poll.

    The page also redraws from whether a verified workflow is available to
    download, but that is no longer a fact about this kernel: verification runs
    in a scratch kernel the session child spawns and discards, so the child
    holds the record and merges it into this reply (``_observe._api_jobs``).
    """
    return {"jobs": jobs_summary()}


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
        _jobs_by_thread.clear()
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
    """Wire the job runner into the kernel: store the InteractiveShell, install
    the thread-aware streams, and clear any prior job state."""
    global _ip
    _ip = ip
    _install_streams()
    reset()
