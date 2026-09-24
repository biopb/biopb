"""The job records, kept in the host from what the kernel publishes.

Runs **in the MCP server process**, owned by ``KernelHost`` for its lifetime,
so the records outlive any one kernel, and it names every job (``job-N``).
Built from iopub alone (``_kernel_io.KernelChannels`` hands every message
here); every ``stream`` / ``execute_result`` / ``error`` under a job's request
is the job's. Reading a record is a read of this process's memory, never a
kernel round trip.

Two kinds of job, told apart by how they start and end:

* **A cell** -- an execute request, the agent's or a foreign Jupyter
  client's -- is read from the protocol itself: an ``error`` fails it, and the
  ``status: idle`` for its request ends it. A foreign cell starts at its
  ``execute_input``; the host records its own as it sends them
  (:meth:`JobLog.start_cell`), and its shell reply ends one whose idle is lost.
  A verification's cells are these too, one request each (``_scratch``).
* **A task** (``run_async``) runs on a worker thread, outliving the request
  that started it, so the kernel announces its start and end as ``biopb_job``
  messages (``_jobs._publish``); ipykernel files its prints under the cell
  that started it.
"""

import io
import re
import threading
import time

# Keep at most this many terminal job records before evicting the oldest. The
# ceiling is what a workflow can be *reconstructed* from: rewriting a session
# into a clean program (``verify_workflow``) reads the transcript, so eviction
# takes away the source material for the one step nothing can automate.
_MAX_RETAINED_JOBS = 200

# Keep at most this many characters of one job's captured output. This is the
# bound _MAX_RETAINED_JOBS is not: that caps how many records are kept, while a
# single cell printing in a loop grew its buffer without limit. Well above
# observe's 20k display cap, so a truncated *view* still means "there is more in
# the record" rather than "the record ends here".
_MAX_JOB_OUTPUT_CHARS = 200_000

# How much of the front of a stream `write_output` copies aside so that
# `output_head` can find the first line without rebuilding the buffer. Two
# orders of magnitude above the 80-char line it has to find, so the only text
# it can miss is a first line already too long to survive the cap anyway.
_HEAD_SCAN_CHARS = 4096

# The iopub message type of the kernel's job announcements (_jobs._publish).
MSG_TYPE = "biopb_job"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# error_text of a record whose kernel went away before announcing its end.
_KERNEL_GONE = "The kernel stopped before this job finished"

# error_text of a record whose end announcement never arrived: iopub is a PUB
# socket and drops under pressure. Only one job runs at a time, so the next
# start proves this one is over.
_END_LOST = (
    "This job has ended, but how was not recorded (its end announcement was "
    "lost). Its output above is complete as far as it goes."
)


def _dropped_marker(n):
    """The line `output` prepends once the cap has discarded a head.

    One spelling, because `output_head` has to recognise the same sentence it
    would otherwise have to re-derive from the rebuilt text.
    """
    return f"...({n} earlier chars dropped)..."


def _one_line(text, limit=80):
    """First non-blank line of *text*, trimmed and length-capped.

    Keeps the job list light (the full source and the full intent are both in
    the per-job snapshot) while giving each row an identifying one-liner.
    """
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


class _OutputBuffer:
    """Capped stdout capture, plus the last expression's repr."""

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
        happens once per cap-worth of output instead of once per print. One
        writer thread per buffer, so no lock: a reader racing the swap gets the
        pre-compaction buffer, which is longer but never torn.
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
        streaming the output as it grows has to diff against this. `tell()`
        rather than `len(getvalue())`, which copies the buffer.
        """
        return self.stdout_dropped + self.stdout.tell()

    def output_head(self, limit=80):
        """First non-blank line of :meth:`output`, without rebuilding it.

        `write_output` keeps the first `_HEAD_SCAN_CHARS`, which is where a
        first line short enough to survive `limit` must be. An opening run of
        whitespace longer than that scan reports no head rather than a later
        line -- a preview field, and nothing prints 4 KB of blanks before its
        first word.
        """
        if self.stdout_dropped:
            # The real head is gone; say so, the way `output` does.
            return _one_line(_dropped_marker(self.stdout_dropped), limit)
        return _one_line(self.head_prefix, limit)


class _Record(_OutputBuffer):
    """One job, as the host has seen it."""

    __slots__ = (
        "job_id",
        "request",
        "code",
        "origin",
        "intent",
        "status",
        "error_text",
        "traceback",
        "cancel_reason",
        "seen_by_agent",
        "started",
        "started_wall",
        "elapsed_final",
        "code_preview",
        "intent_preview",
        "kind",
        "ename",
        "reply",
        "began",
    )

    def __init__(self, event, kind="task"):
        super().__init__()
        self.job_id = event["job_id"]
        # "cell" (read from the protocol) or "task" (announced); see module.
        self.kind = kind
        # The iopub error's exception name, which decides how a cell ended.
        self.ename = None
        # A host cell's shell reply, once it arrives (note_reply).
        self.reply = None
        # Whether it has started running: a host cell is recorded when sent,
        # and may wait in the kernel's queue behind someone else's.
        self.began = kind != "cell" or event.get("origin") == "user"
        # The execute request this job's output is published under.
        self.request = event.get("request")
        self.code = event.get("code", "")
        # Who ran it: "mcp" (the execute_code tool), "user" (a cell from an
        # attached Jupyter client) or "chat" (the in-process chat loop).
        self.origin = event.get("origin", "mcp")
        # Why, in the words of whoever asked; free text, provenance only.
        self.intent = event.get("intent", "")
        # running | ok | error | interrupted
        self.status = "running"
        self.error_text = ""
        # An iopub error's traceback, for a foreign cell: fuller than the
        # "ename: evalue" its end announcement carries.
        self.traceback = ""
        self.cancel_reason = None
        # Whether the agent has been told of this *foreign* job
        # (foreign_digest / ack_foreign_digest).
        self.seen_by_agent = False
        self.started = time.monotonic()
        self.started_wall = event.get("created") or time.time()
        # The kernel's own measure, once it announces the end.
        self.elapsed_final = None
        # Cut once, rather than on every observe poll.
        self.code_preview = _one_line(self.code)
        self.intent_preview = _one_line(self.intent)

    def elapsed(self):
        if self.elapsed_final is not None:
            return self.elapsed_final
        return round(time.monotonic() - self.started, 3)

    def end(self, status, error_text="", elapsed=None):
        self.status = status
        self.error_text = error_text
        self.elapsed_final = (
            elapsed
            if elapsed is not None
            else round(time.monotonic() - self.started, 3)
        )

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
        }

    def summary(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "origin": self.origin,
            "elapsed": self.elapsed(),
            "stdout_len": self.output_total(),
            "code_preview": self.code_preview,
            # Why the cell was run, when whoever ran it said. The observe list
            # prefers it over the code line.
            "intent_preview": self.intent_preview,
        }


def _verdict(rec):
    """``(status, error_text)`` for a cell that has ended."""
    if rec.ename == "KeyboardInterrupt":
        status = "interrupted"
    else:
        status = "error" if rec.ename else "ok"
    error_text = rec.traceback
    if rec.cancel_reason and status != "ok":
        error_text = rec.cancel_reason + ("\n" + error_text if error_text else "")
    return status, error_text


class JobLog:
    """Every job the kernels of one host have announced, oldest first."""

    def __init__(self, host_session=None):
        self._records = {}  # job_id -> _Record, in start order
        # request msg_id -> the running record its stream output goes to
        self._by_request = {}
        # request msg_id -> the running cell under it, which its idle ends
        self._cells = {}
        self._lock = threading.Lock()
        # The highest job number seen, handed to the next kernel so its ids
        # continue rather than restart (_jobs reads it at import).
        self._seq = 0
        # The host's client session: its requests are the host's own snippets,
        # never a cell to record. Set per kernel (KernelHost._launch).
        self.host_session = host_session
        # The agent's own job origin, the point of view the eviction hold is
        # read from: the last non-user origin to start a job. "Foreign" is a
        # relation, and the hold runs under no asker to take one from.
        self._agent_origin = "mcp"

    # -- fed from iopub, on the IO loop thread -------------------------------

    def on_iopub(self, msg):
        msg_type = msg["header"]["msg_type"]
        content = msg["content"]
        if msg_type == MSG_TYPE:
            self._on_event(content)
            return
        if msg_type == "execute_input":
            self._on_cell_start(msg)
            return
        request = (msg.get("parent_header") or {}).get("msg_id")
        if request is None:
            return
        with self._lock:
            cell = self._cells.get(request)
            # A cell's prints are its own until it starts a task, whose thread
            # prints under the same request (run_async).
            route = self._by_request.get(request)
        if msg_type == "stream":
            if route is not None:
                route.write_output(content.get("text", ""))
        elif cell is None:
            return
        elif msg_type == "execute_result":
            text = content.get("data", {}).get("text/plain", "")
            if text:
                cell.result_text = text
        elif msg_type == "error":
            cell.ename = content.get("ename")
            cell.traceback = _ANSI_RE.sub("", "\n".join(content.get("traceback", [])))
        elif msg_type == "status" and content.get("execution_state") == "idle":
            self._on_cell_end(cell)

    def start_cell(self, job_id, request, code, origin, intent=""):
        """Record a cell the host is about to send under *request*. Before the
        send, so none of its output arrives unrouted."""
        with self._lock:
            rec = _Record(
                {
                    "job_id": job_id,
                    "request": request,
                    "code": code,
                    "origin": origin,
                    "intent": intent,
                },
                kind="cell",
            )
            self._add_cell(rec)
            if origin != "user":
                self._agent_origin = origin

    def note_reply(self, job_id, reply):
        """Keep a host cell's shell reply: how it ended, and the viewer
        window's state after it (:meth:`window_alive`)."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.reply = reply
            # The reply carries the error too, and cannot be lost as the
            # iopub error can.
            if reply.get("status") == "error":
                rec.ename = rec.ename or reply.get("ename")
                if not rec.traceback:
                    rec.traceback = _ANSI_RE.sub(
                        "", "\n".join(reply.get("traceback") or [])
                    )
                if rec.kind == "cell" and rec.status == "ok":
                    # Ended on its idle, its iopub error lost.
                    rec.end(*_verdict(rec), rec.elapsed_final)

    def cell_replied(self, job_id):
        """End a host cell from its shell reply, which cannot be lost -- for
        when its idle, which marks its output complete, never arrives."""
        with self._lock:
            rec = self._records.get(job_id)
        if rec is not None and rec.kind == "cell":
            self._on_cell_end(rec)

    def window_alive(self, job_id):
        """Whether the viewer window was open after *job_id*'s cell: True,
        False, or None when unknown."""
        with self._lock:
            rec = self._records.get(job_id)
            reply = rec.reply if rec is not None else None
        value = ((reply or {}).get("user_expressions") or {}).get("w") or {}
        text = (value.get("data") or {}).get("text/plain")
        return {"True": True, "False": False}.get(text)

    def _on_cell_start(self, msg):
        parent = msg.get("parent_header") or {}
        code = msg["content"].get("code", "")
        # The host's own cells are recorded as it sends them (start_cell), and
        # start here; its snippets are not cells. An empty cell is a client
        # asking for its prompt number.
        if parent.get("session") == self.host_session:
            with self._lock:
                rec = self._cells.get(parent.get("msg_id"))
                if rec is not None:
                    rec.began = True
                    rec.started = time.monotonic()
            return
        if not code.strip():
            return
        with self._lock:
            self._add_cell(
                _Record(
                    {
                        "job_id": self._next_id(),
                        "request": parent.get("msg_id"),
                        "code": code,
                        "origin": "user",
                    },
                    kind="cell",
                )
            )

    def _add_cell(self, rec):
        """Call with `_lock` held."""
        self._records[rec.job_id] = rec
        self._cells[rec.request] = rec
        self._by_request[rec.request] = rec
        self._prune()

    def _on_cell_end(self, rec):
        with self._lock:
            if self._cells.get(rec.request) is not rec or rec.status != "running":
                return
            del self._cells[rec.request]
            if self._by_request.get(rec.request) is rec:
                del self._by_request[rec.request]
            # Cells run one at a time on the main thread, so another cell that
            # had begun had ended, its end lost; one still queued has not. A
            # task runs beside them.
            for other in [c for c in self._cells.values() if c.began]:
                other.end("error", _END_LOST)
                del self._cells[other.request]
                if self._by_request.get(other.request) is other:
                    del self._by_request[other.request]
            rec.end(*_verdict(rec))

    def _on_event(self, event):
        kind = event.get("event")
        job_id = event.get("job_id")
        if not job_id:
            return
        with self._lock:
            if kind == "start":
                if job_id in self._records:
                    return
                # One task at a time, so a task still running here had ended,
                # its end lost. Cells run beside a task.
                for other in self._records.values():
                    if other.kind == "task" and other.status == "running":
                        other.end("error", _END_LOST)
                        if self._by_request.get(other.request) is other:
                            del self._by_request[other.request]
                rec = _Record(event)
                # Started from a cell (run_async): the cell's writer is its.
                cell = self._cells.get(rec.request)
                if cell is not None:
                    rec.origin = cell.origin
                self._records[job_id] = rec
                if rec.request:
                    self._by_request[rec.request] = rec
                if rec.origin != "user":
                    self._agent_origin = rec.origin
                self._prune()
            elif kind == "end":
                rec = self._records.get(job_id)
                if rec is None or rec.status != "running":
                    return
                if event.get("result_text"):
                    rec.result_text = event["result_text"]
                rec.cancel_reason = event.get("cancel_reason")
                error_text = event.get("error_text") or ""
                if rec.traceback and event.get("status") == "error":
                    error_text = rec.traceback
                rec.end(event.get("status", "error"), error_text, event.get("elapsed"))
                if self._by_request.get(rec.request) is rec:
                    del self._by_request[rec.request]

    def kernel_gone(self, why=""):
        """End every running record: its kernel is going away."""
        text = _KERNEL_GONE + (f" ({why})." if why else ".")
        with self._lock:
            for rec in self._records.values():
                if rec.status != "running":
                    continue
                rec.end("interrupted", text)
            self._by_request.clear()
            self._cells.clear()

    def _next_id(self):
        """The next job id. Call with `_lock` held."""
        self._seq += 1
        return f"job-{self._seq}"

    def new_id(self):
        """An id for a job the host is about to start (``run_cell``)."""
        with self._lock:
            return self._next_id()

    def stop_key(self, job_id):
        """What the kernel names *job_id* by while it runs (``_jobs.interrupt``):
        a cell's request, a task's own id; None when it does not run here."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None or rec.status != "running":
                return None
            return rec.request if rec.kind == "cell" else rec.job_id

    def note_cancel(self, job_id, reason):
        """Attribute a stop the host made: a cell's end carries no reason."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is not None and rec.status == "running":
                rec.cancel_reason = reason

    def _prune(self):
        # Oldest-first, never a running job and never a foreign job the agent
        # has not been told about: that notice is its only word that the
        # namespace changed under it. So the cap can be exceeded, by as many
        # cells as another writer runs between two agent calls.
        terminal = [
            jid
            for jid, r in self._records.items()
            if r.status != "running"
            and not (r.origin != self._agent_origin and not r.seen_by_agent)
        ]
        while len(self._records) > _MAX_RETAINED_JOBS and terminal:
            del self._records[terminal.pop(0)]

    # -- reads ------------------------------------------------------------------

    def poll(self, job_id):
        """*job_id*'s snapshot, or ``{"status": "unknown"}``."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return {"job_id": job_id, "status": "unknown", "error_text": ""}
            return rec.snapshot()

    def outcome(self, job_id, full=False):
        """How *job_id* stands, without its output unless *full*: a
        verification's per-cell ledger (``_scratch``), read on every poll. None
        for an unknown id."""
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            out = {
                "status": rec.status,
                "error_text": rec.error_text,
                "elapsed": rec.elapsed(),
                "stdout_len": rec.output_total(),
                "stdout_head": rec.output_head(),
            }
            if full:
                out["stdout"] = rec.output()
                out["result_text"] = rec.result_text
            return out

    def summary(self):
        """One light row per retained job, for the observe list."""
        with self._lock:
            return [r.summary() for r in self._records.values()]

    def export(self):
        """Full snapshots of every retained job, for the notebook export."""
        with self._lock:
            return [r.snapshot() for r in self._records.values()]

    def running(self, prefer=None):
        """A running job's snapshot, or ``None``: one of origin *prefer* if
        any, since a task can run beside another writer's cell."""
        with self._lock:
            live = [r for r in self._records.values() if r.status == "running"]
            own = [r for r in live if r.origin == prefer]
            pick = (own or live or [None])[0]
            return pick.snapshot() if pick is not None else None

    def running_cell(self):
        """The running cell's snapshot -- whatever holds the main thread -- or
        ``None``."""
        with self._lock:
            for rec in self._cells.values():
                if rec.status == "running":
                    return rec.snapshot()
        return None

    def foreign_digest(self, for_origin):
        """Jobs *for_origin*'s client did not start and has not been told
        about yet, oldest first: ``[{"job_id", "status", "elapsed",
        "origin"}]``.

        A pure read; retiring is :meth:`ack_foreign_digest`, called once the
        notice has actually reached the agent.
        """
        with self._lock:
            return [
                {
                    "job_id": r.job_id,
                    "status": r.status,
                    "elapsed": r.elapsed(),
                    "origin": r.origin,
                }
                for r in self._records.values()
                if r.origin != for_origin and not r.seen_by_agent
            ]

    def ack_foreign_digest(self, job_ids):
        """Mark the jobs in *job_ids* as reported; return how many were.

        *job_ids* is what the caller told the agent **and reported as
        terminal**; the status is deliberately not re-read, since a job
        reported running that finished since must stay pending.
        """
        wanted = set(job_ids)
        acked = 0
        with self._lock:
            for rec in self._records.values():
                if (
                    rec.origin != self._agent_origin
                    and not rec.seen_by_agent
                    and rec.job_id in wanted
                ):
                    rec.seen_by_agent = True
                    acked += 1
        return acked
