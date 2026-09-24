"""The job records, kept in the host from what the kernel publishes.

Runs **in the MCP server process**, owned by ``KernelHost`` for its lifetime,
so the records outlive any one kernel, and it names every job (``job-N``).
Built from iopub alone (``_kernel_io.KernelChannels`` hands every message
here); every ``stream`` / ``execute_result`` / ``error`` under a job's request
is the job's. Reading a record is a read of this process's memory, never a
kernel round trip.

Two kinds of job, told apart by how they start and end:

* **A cell** -- a foreign Jupyter client's execute request -- is read from the
  protocol itself: its ``execute_input`` starts it, an ``error`` fails it, and
  the ``status: idle`` for its request ends it.
* **A task** runs on a worker thread, outliving the request that started it,
  so the kernel announces its start and end as ``biopb_job`` messages
  (``_jobs._publish``): an agent's job, its prints attributed by ipykernel to
  the submit request, or a verification in a scratch kernel, whose cells are
  announced one by one so its output can be split per cell here.
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

# The error a gate's refusal carries (_kernel_gate): the cell never ran, though
# ipykernel published its execute_input before the gate saw it.
_REFUSED = "KernelBusy"

# Appended to a record settled from the kernel's own account (JobLog.settle).
_OUTPUT_LOST = (
    "\n[biopb: this job's end announcement was lost on iopub; its outcome is "
    "the kernel's, and output may be missing.]\n"
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
    """Capped stdout capture, plus the last expression's repr.

    Shared by a host job record and by one cell of a kernel verification run.
    One cap, one dropped-head marker, one monotonic total -- so a cell reports
    its output the way a job does without either having to remember to.
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


class _CellRecord(_OutputBuffer):
    """One cell of a verification run: its source, its outcome, its output.

    The job keeps the whole run's output as well, because the two readers want
    different cuts of one stream: the notebook needs it split per cell, and
    ``poll_job`` on a long verification needs it accumulating in one place, the
    way it does for any other job.
    """

    __slots__ = ("code", "status", "error_text", "started", "elapsed_final")

    def __init__(self, code):
        super().__init__()
        self.code = code
        # pending | ok | error | skipped
        self.status = "pending"
        self.error_text = ""
        self.started = None
        self.elapsed_final = None

    def elapsed(self):
        if self.elapsed_final is not None:
            return self.elapsed_final
        if self.started is None:
            return 0.0
        return round(time.monotonic() - self.started, 3)

    def snapshot(self, full=False):
        """This cell's outcome; *full* adds the captured output.

        The polled snapshot carries a one-line head and a length, the way the
        job list does for a job: a report's ledger needs no more, and shipping
        every cell's output on each poll would send the job's own buffer again,
        once per cell. The notebook reads it once, with ``full=True``
        (:meth:`JobLog.verify_record`).
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


class _VerifyRecord:
    """A candidate workflow, its cells, and what running them did.

    The record a workflow notebook is built from. Deliberately *not* a list of
    job ids: the program that works is a rewrite of the transcript, not a
    selection from it. So the cells are the agent's own text, and what makes
    them trustworthy is that they ran.
    """

    __slots__ = ("title", "created", "cells", "current")

    def __init__(self, spec):
        self.title = spec.get("title", "")
        self.created = spec.get("created") or time.time()
        self.cells = [_CellRecord(code) for code in spec.get("cells", [])]
        # The cell whose output is arriving, between its start and end.
        self.current = None

    def status(self):
        """``ok`` once every cell ran, ``error`` at the first failure."""
        if any(c.status == "error" for c in self.cells):
            return "error"
        if self.cells and all(c.status == "ok" for c in self.cells):
            return "ok"
        return "running"

    def snapshot(self, full=False):
        return {
            "title": self.title,
            "created": self.created,
            "status": self.status(),
            "cells": [c.snapshot(full=full) for c in self.cells],
        }


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
        "verify",
        "kind",
        "ename",
    )

    def __init__(self, event, kind="task"):
        super().__init__()
        self.job_id = event["job_id"]
        # "cell" (read from the protocol) or "task" (announced); see module.
        self.kind = kind
        # The iopub error's exception name, which decides how a cell ended.
        self.ename = None
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
        # A verification's per-cell record, or None for an ordinary job.
        spec = event.get("verify")
        self.verify = _VerifyRecord(spec) if spec else None

    def write_output(self, s):
        n = super().write_output(s)
        v = self.verify
        if v is not None and v.current is not None:
            v.cells[v.current].write_output(s)
        return n

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
            "verify": self.verify.snapshot() if self.verify is not None else None,
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


class JobLog:
    """Every job the kernels of one host have announced, oldest first."""

    def __init__(self, host_session=None):
        self._records = {}  # job_id -> _Record, in start order
        self._by_request = {}  # request msg_id -> _Record, running ones only
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
            rec = self._by_request.get(request)
        if rec is None:
            return
        if msg_type == "stream":
            rec.write_output(content.get("text", ""))
        elif msg_type == "execute_result":
            text = content.get("data", {}).get("text/plain", "")
            if text:
                rec.result_text = text
        elif msg_type == "error":
            rec.ename = content.get("ename")
            rec.traceback = _ANSI_RE.sub("", "\n".join(content.get("traceback", [])))
        elif (
            msg_type == "status"
            and content.get("execution_state") == "idle"
            and rec.kind == "cell"
        ):
            self._on_cell_end(rec)

    def _on_cell_start(self, msg):
        parent = msg.get("parent_header") or {}
        code = msg["content"].get("code", "")
        # The host's own snippets are not cells; an empty cell is a client
        # asking for its prompt number, which the gate lets through unrecorded.
        if parent.get("session") == self.host_session or not code.strip():
            return
        with self._lock:
            rec = _Record(
                {
                    "job_id": self._next_id(),
                    "request": parent.get("msg_id"),
                    "code": code,
                    "origin": "user",
                },
                kind="cell",
            )
            self._records[rec.job_id] = rec
            self._by_request[rec.request] = rec
            self._prune()

    def _on_cell_end(self, rec):
        with self._lock:
            self._by_request.pop(rec.request, None)
            if rec.ename == _REFUSED:
                # Refused by the gate while a task ran: it never ran, and its
                # start proved nothing about the task.
                self._records.pop(rec.job_id, None)
                return
            # It ran, so the main thread was free: a task or cell still
            # running here had ended, its end lost.
            for other in self._records.values():
                if other is not rec and other.status == "running":
                    other.end("error", _END_LOST)
                    self._by_request.pop(other.request, None)
            if rec.ename == "KeyboardInterrupt":
                status = "interrupted"
            else:
                status = "error" if rec.ename else "ok"
            error_text = rec.traceback
            if rec.cancel_reason and status != "ok":
                error_text = rec.cancel_reason + (
                    "\n" + error_text if error_text else ""
                )
            rec.end(status, error_text)

    def _on_event(self, event):
        kind = event.get("event")
        job_id = event.get("job_id")
        if not job_id:
            return
        with self._lock:
            if kind == "start":
                if job_id in self._records:
                    return  # already settled from the kernel (settle)
                for other in self._records.values():
                    if other.status == "running":
                        other.end("error", _END_LOST)
                        self._by_request.pop(other.request, None)
                rec = _Record(event)
                self._records[job_id] = rec
                if rec.request:
                    self._by_request[rec.request] = rec
                if rec.origin != "user":
                    self._agent_origin = rec.origin
                self._prune()
            elif kind in ("cell_start", "cell_end"):
                rec = self._records.get(job_id)
                if rec is None or rec.verify is None:
                    return
                i = event.get("index", -1)
                if not 0 <= i < len(rec.verify.cells):
                    return
                cell = rec.verify.cells[i]
                if kind == "cell_start":
                    cell.started = time.monotonic()
                    rec.verify.current = i
                else:
                    cell.status = event.get("status", "error")
                    cell.error_text = event.get("error_text") or ""
                    cell.result_text = event.get("result_text") or ""
                    cell.elapsed_final = event.get("elapsed")
                    rec.verify.current = None
            elif kind == "end":
                rec = self._records.get(job_id)
                if rec is None or rec.status != "running":
                    return
                if rec.verify is not None:
                    # The cells a failure never reached: marked, not dropped,
                    # so the report says how far the workflow got.
                    rec.verify.current = None
                    for cell in rec.verify.cells:
                        if cell.status == "pending":
                            cell.status = "skipped"
                if event.get("result_text"):
                    rec.result_text = event["result_text"]
                rec.cancel_reason = event.get("cancel_reason")
                error_text = event.get("error_text") or ""
                if rec.traceback and event.get("status") == "error":
                    error_text = rec.traceback
                rec.end(event.get("status", "error"), error_text, event.get("elapsed"))
                self._by_request.pop(rec.request, None)

    def kernel_gone(self, why=""):
        """End every running record: its kernel is going away."""
        text = _KERNEL_GONE + (f" ({why})." if why else ".")
        with self._lock:
            for rec in self._records.values():
                if rec.status != "running":
                    continue
                rec.end("interrupted", text)
                v = rec.verify
                if v is not None:
                    # The cell it died in failed; the rest never ran.
                    for i, cell in enumerate(v.cells):
                        if i == v.current:
                            cell.status, cell.error_text = "error", text
                        elif cell.status == "pending":
                            cell.status = "skipped"
                    v.current = None
            self._by_request.clear()

    def settle(self, snap):
        """Bring a record in line with the kernel's own account of the job.

        *snap* is ``_jobs.poll``'s, which came back on a shell reply: iopub
        can drop, that cannot. Replays what was missed as the events would
        have (start, cell ends, end), so a lost announcement cannot leave a
        record unknown or running for good. The output lost with it stays lost,
        and the record says so.
        """
        job_id = snap.get("job_id")
        status = snap.get("status")
        if not job_id or status in (None, "unknown"):
            return
        verify = snap.get("verify")
        with self._lock:
            rec = self._records.get(job_id)
        if rec is None:
            spec = None
            if verify is not None:
                spec = dict(verify, cells=[c["code"] for c in verify["cells"]])
            # The snapshot carries every field the start does.
            self._on_event(dict(snap, event="start", verify=spec))
            rec = self._records[job_id]
        if status == "running" or rec.status != "running":
            return
        if verify is not None and rec.verify is not None:
            for i, cell in enumerate(verify["cells"]):
                if cell["status"] in ("ok", "error") and (
                    rec.verify.cells[i].status == "pending"
                ):
                    self._on_event(dict(cell, event="cell_end", job_id=job_id, index=i))
        rec.write_output(_OUTPUT_LOST)
        self._on_event(dict(snap, event="end"))

    def _next_id(self):
        """The next job id. Call with `_lock` held."""
        self._seq += 1
        return f"job-{self._seq}"

    def new_id(self):
        """An id for a job the host is about to submit (``_jobs.submit``)."""
        with self._lock:
            return self._next_id()

    def request_of(self, job_id):
        """The request *job_id* runs under, while it runs; else None. What the
        kernel names a job by when stopping it (``_jobs.interrupt``)."""
        with self._lock:
            rec = self._records.get(job_id)
            return rec.request if rec is not None and rec.status == "running" else None

    def job_of(self, request):
        """The running job under *request*, or None."""
        with self._lock:
            rec = self._by_request.get(request)
            return rec.job_id if rec is not None else None

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

    def summary(self):
        """One light row per retained job, for the observe list."""
        with self._lock:
            return [r.summary() for r in self._records.values()]

    def export(self):
        """Full snapshots of every retained job, for the notebook export."""
        with self._lock:
            return [r.snapshot() for r in self._records.values()]

    def verify_record(self, job_id):
        """*job_id*'s verification record with every cell's output, or None.

        The other half of the polled/full split (:meth:`_CellRecord.snapshot`):
        read once, when the run ends, for the document the record becomes.
        """
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None or rec.verify is None:
                return None
            return rec.verify.snapshot(full=True)

    def running(self):
        """The running job's snapshot, or ``None``."""
        with self._lock:
            for rec in self._records.values():
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
