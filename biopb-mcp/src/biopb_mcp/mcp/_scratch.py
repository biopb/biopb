"""Verifying a workflow in a scratch kernel: the process, and the slot it takes.

Runs **in the MCP server process**. ``verify_workflow`` asks one question — will
this program run on its own? — and the only honest way to answer it is to run it
somewhere that has none of the session's state. This module owns that somewhere:
a second kernel, spawned per verification, discarded after it.

Why a process and not a namespace, why the session's own display, and what
"discarded" does *not* cover: ``docs/verify-workflow.md``.

Three things live here, and they are one module because they are one decision.

* **The kernel.** A full bootstrap with the user-facing parts left out
  (``_bootstrap.ENV_SCRATCH``), no watchdog, and no window-close pipe. The
  watchdog is off deliberately: for the session kernel a respawn is recovery,
  but for this one **death is the verdict**. An OOM means "this workflow does
  not fit", and respawning would re-run a workflow that just killed a process,
  three more times, each allocating gigabytes on a machine already under
  pressure.
* **The run.** Each cell is its own execute request, sent once the one before
  it ended ``ok``, and recorded the way any cell is: by that kernel's host,
  from the protocol (``_job_log``). So the per-cell output, the stop at the
  first failure and the completion all come from the protocol, and a stop is
  the same interrupt any cell gets.
* **The slot.** Two kernels must not become two schedulers. The dask cluster is
  shared and finite, and the agent's whole model is "one cell at a time", so a
  verification takes the same slot ordinary work does — see :func:`start`.

**One run at a time, and only the last one.** The record lives here, in the
child, because the kernel that produced it is gone -- but only the newest, and
the workflow download follows it: the answer is "this run passed, just now", not
"some run passed at some point in a kernel that has since been restarted".
"""

import json
import logging
import threading
import time
from pathlib import Path

from . import _notebook, _workflow_doc

logger = logging.getLogger(__name__)

# Builds an unstarted scratch KernelHost. Set by the launcher, which is the only
# thing that knows the config the session kernel was built from -- this module
# must spawn a kernel that matches it, and deriving that twice is how the two
# drift. None outside a configured session (unit tests), which reads as "no
# scratch kernel available" rather than a crash.
_host_factory = None


def set_host_factory(factory):
    """Register the callable that builds an unstarted scratch ``KernelHost``."""
    global _host_factory
    _host_factory = factory


# The single cross-kernel slot, and the run holding it. `_lock` guards both, and
# is held only to take/read/release -- never across the run, because a scratch
# kernel OOM-killed mid-verification is the exact scenario this design is for
# and it must not die holding a lock.
_lock = threading.RLock()
_run = None
_seq = 0

# There is no slot for "the last run that passed". The page shows the last run
# and the download follows it, so a second attempt that fails leaves nothing to
# offer until it is re-run. That is the honest reading of one visible run: an
# offer to download a document the reader cannot see, described by a row that
# says `error`, is the confusing half of keeping the older one.
#
# The document itself is not lost, though -- see :func:`_spool`. What a later
# failure takes away is the *offer*, not the file.

#: How many spooled workflow notebooks to keep. Every run that passes is
#: written, because this process cannot know which passing run has the right
#: numbers -- only that every cell ran -- so the choice is deferred to whoever
#: reads them. They are a few kilobytes each and verifications are rare, so the
#: cap is generous; it exists so a long-lived session cannot fill a disk.
_SPOOL_KEEP = 50

#: Where drafts live, under the workflow dir. A subdirectory so the spool's
#: glob (and its keep-newest rule) sees records only: a draft is a working file,
#: not a run that happened.
_DRAFT_DIR = "drafts"

#: How long an interrupt waits for the cells to stop before taking the process.
#:
#: An interrupt on the *session* kernel is best-effort by necessity: a
#: blocking C call (a gRPC fetch, a native dask compute) does not notice it
#: until it returns to Python -- and the guaranteed stop, a group-kill, costs
#: the user their whole session, so it is theirs to ask for.
#:
#: None of that holds here. A scratch kernel has no variables anyone wants, no
#: layers, no session -- killing it costs nothing, and it was going to be
#: discarded seconds later anyway. So the interrupt is only *briefly*
#: best-effort: long enough for a clean stop to give the better record, then
#: the process goes.
#:
#: Which is what keeps `restart_kernel` out of this. Without the escalation, a
#: verification wedged in a C call would leave the agent nothing but the tool
#: that destroys the user's session, to kill a process built to be thrown away.
_INTERRUPT_GRACE = 5.0

#: Job ids issued here, in their own namespace. The session kernel issues
#: ``job-N``; a verification never runs there, so a distinct prefix is what lets
#: ``poll_job`` route an id to the kernel that owns it without asking either.
_ID_PREFIX = "verify-"

# error_text of a cell the run ended in, when its own record never did (the
# kernel was taken down around it).
_UNFINISHED = "The verification ended before this cell finished."


def _elapsed(run):
    return round((run["finished"] or time.monotonic()) - run["started"], 1)


def _record(run, full=False, final=False):
    """The run's per-cell record, read from the scratch host's cell records,
    or None before a cell was sent.

    *full* adds each cell's output, for the notebook; the polled record carries
    a one-line head and a length, as the job list does for a job. *final* is
    for the record the run ends with: a cell never sent is ``skipped``, not
    ``pending``, and one still running when the run ended failed there.
    """
    log = run["log"]
    if log is None or _current_cell(run) is None:
        return None
    cells = []
    for code, job_id in zip(run["cells"], run["cell_jobs"], strict=True):
        out = log.outcome(job_id, full) if job_id is not None else None
        if out is None:
            out = {"error_text": "", "elapsed": 0.0, "stdout_len": 0}
            out["stdout_head"] = ""
            if full:
                out.update(stdout="", result_text="")
            status = "skipped" if final else "pending"
        elif out["status"] == "running":
            status = "error" if final else "running"
            if final:
                out["error_text"] = out["error_text"] or _UNFINISHED
        else:
            # A stopped cell failed as far as the workflow is concerned; the
            # run's own status says it was a stop.
            status = "ok" if out["status"] == "ok" else "error"
        cells.append({**out, "code": code, "status": status})
    if any(c["status"] == "error" for c in cells):
        status = "error"
    elif all(c["status"] == "ok" for c in cells):
        status = "ok"
    else:
        status = "running"
    return {
        "title": run["title"],
        "created": run["created"],
        "status": status,
        "cells": cells,
    }


def _current_cell(run):
    """The host job id of the last cell sent, or None. Call with `_lock` held."""
    return next((j for j in reversed(run["cell_jobs"]) if j is not None), None)


def _snapshot(run, output=False):
    """A verification rendered in the shape ``poll_job`` renders a job in.

    Deliberately the same keys a host record's snapshot carries (``job_id``,
    ``status``, ``elapsed``, ``stdout``) plus ``verify``, so the tool surface
    has one renderer rather than two: ``_server._format_job_status`` already
    knows what to do with a ``verify`` record. *output* puts the running
    cell's output under the progress note, for a poll that shows it.
    """
    record = run["record"] if run["record"] is not None else _record(run)
    stdout = run["note"]
    if output and run["status"] == "running" and run["log"] is not None:
        current = _current_cell(run)
        out = run["log"].outcome(current, full=True) if current else None
        if out is not None and out["status"] == "running":
            stdout += out["stdout"]
    return {
        "job_id": run["job_id"],
        "status": run["status"],
        "elapsed": _elapsed(run),
        "stdout": stdout,
        "error_text": run["error"] or "",
        "result_text": "",
        "verify": record,
        "origin": run["origin"],
        "intent": run["intent"],
        "title": run["title"],
        "cell_count": len(run["cells"]),
        "saved_path": run["saved_path"],
        "draft_path": run["draft_path"],
    }


def detail(job_id):
    """A verification in the shape the observe detail view reads, or ``None``.

    The page lists a running verification as one row (``_observe._api_jobs``),
    and a row that cannot be opened is worse than no row: the session kernel has
    never heard of a ``verify-N`` id, so asking it returns 404 and the run shows
    neither progress nor output.

    What it shows instead of a job's stdout is the run's own progress -- which
    stage the bring-up reached, then a line per cell as each finishes. The full
    per-cell output belongs to the notebook, not to a ledger (see
    :func:`_record`).
    """
    snap = poll(job_id)
    if snap is None:
        return None
    with _lock:
        blocks = list(_run["blocks"]) if _run and _run["job_id"] == job_id else []
    record = snap.get("verify") or {}
    lines = [snap["stdout"].rstrip()] if snap.get("stdout") else []
    for i, cell in enumerate(record.get("cells") or [], 1):
        head = (cell.get("stdout_head") or "").strip()
        lines.append(
            f"  {i}. {cell.get('status')} · {cell.get('elapsed')}s"
            + (f" · {head}" if head else "")
        )
        if cell.get("error_text"):
            lines.append(cell["error_text"].rstrip())
    return {
        **snap,
        # The document, prose and all -- not the cells joined back together.
        # What is being verified is a document, and the page has no better
        # rendering of one than its own text.
        "code": _workflow_doc.to_markdown(blocks) if blocks else "",
        "stdout": "\n".join(lines) + ("\n" if lines else ""),
        # The scratch kernel's viewer is hidden and is not the session's, so its
        # liveness is not a thing to warn the user about.
        "window_alive": None,
    }


def running():
    """The verification holding the slot, as a snapshot, or ``None``."""
    with _lock:
        if _run is None or _run["status"] != "running":
            return None
        return _snapshot(_run)


def owns(job_id):
    """Whether *job_id* names a verification rather than a session-kernel job."""
    return isinstance(job_id, str) and job_id.startswith(_ID_PREFIX)


def poll(job_id):
    """The snapshot for *job_id*, or ``None`` if this is not a run we know."""
    with _lock:
        if _run is not None and _run["job_id"] == job_id:
            return _snapshot(_run, output=True)
    return None


def runs_view():
    """The last verification, as a one-row list, or an empty one.

    A list rather than a record so the observe page's verification pane is the
    same list component as its session pane rather than a second one. One row,
    because the runs before it are not reachable: their kernels are gone, and
    the only thing anyone does with an older run -- download the workflow it
    proved -- is a lookup this page does not yet offer.

    Where a session job's "why" is the intent its writer gave that cell, a
    verification has neither: the cells arrive as bare code and the run's intent
    is synthesized from the title (``_server.verify_workflow``). So the title
    *is* the why, and the row says it once -- the count is the what. A workflow
    cell needs no intent of its own the way a transcript cell does: it was
    written to be read.
    """
    with _lock:
        if _run is None:
            return []
        cells = len(_run["cells"])
        return [
            {
                "job_id": _run["job_id"],
                "status": _run["status"],
                "origin": _run["origin"],
                "elapsed": _elapsed(_run),
                "code_preview": f"{cells} cell" + ("s" if cells != 1 else ""),
                "intent_preview": _run["title"],
                "verify": True,
            }
        ]


def verified():
    """The last verification's record if every cell ran, else ``None``.

    The download follows the run the page is showing, so this is empty while one
    is running and empty after one fails -- see the note where ``_run`` is
    declared for why no earlier pass is kept.
    """
    with _lock:
        if _run is None or _run["status"] != "ok" or _run["record"] is None:
            return None
        return {
            **_run["record"],
            # The prose never went to the kernel -- it is not code and the
            # kernel has no use for it -- so it is merged back here, where both
            # readers of a record (the spool, and the observe download) get it
            # from one place.
            "blocks": list(_run["blocks"]),
            "job_id": _run["job_id"],
            "saved_path": _run["saved_path"],
        }


def verified_summary():
    """A one-line description of :func:`verified`, or ``None``.

    Carried on the observe poll so the page can offer the workflow download
    without a second round trip per second for a value that changes rarely.
    """
    record = verified()
    if record is None:
        return None
    return {
        "job_id": record["job_id"],
        "title": record.get("title", ""),
        "cells": len(record.get("cells") or []),
        "created": record.get("created"),
        "saved_path": record.get("saved_path"),
    }


def start(
    blocks, title, session_host, intent="", writer=None, writer_label="", origin="mcp"
):
    """Take the slot and begin verifying a document in a fresh scratch kernel.

    *blocks* is the parsed document (:mod:`_workflow_doc`); its code blocks are
    what runs, and the whole of it is what the notebook is built from. The draft
    is written before the run starts, so an attempt that fails -- the one a
    person most wants to open and fix -- is on disk like any other.

    Returns ``{"job_id": ...}``, or ``{"error": "busy", ...}`` naming what holds
    the slot, or ``{"error": <reason>}`` when no scratch kernel can be built.

    *writer* is the client this run belongs to, and *origin* is its point of
    view (``"chat"`` when the in-process loop asked for the verification):
    a second client is refused a stop on it, as on the session kernel (see
    :func:`interrupt`).

    **The slot is why the cluster can be shared.** Two clients on one
    ``LocalCluster`` sounds like contention, but under one slot there is no
    second computation to contend with: while a verification runs the session
    kernel is by construction not running a job, and the viewer's own slice
    reads are deliberately kept off the cluster
    (``ViewerConfig.compute_scheduler`` defaults to ``"threads"``).

    The two directions of that rule are not enforced the same way, and the
    difference is worth knowing. A session job is refused while a verification
    holds the slot **exactly**, because the slot is in this process and the
    refusal reads it. A verification is refused while a session job runs by
    reading the session host's records, which learn of a job from iopub a moment
    after it starts -- a check-then-act, so two jobs can still start within a
    millisecond of each other. The consequence is bounded (two jobs briefly
    sharing a warm cluster, not a corrupted anything).
    """
    global _run, _seq
    if _host_factory is None:
        return {"error": "Verification is unavailable: no scratch kernel configured."}

    with _lock:
        if _run is not None and _run["status"] == "running":
            return {
                "error": "busy",
                "running_job_id": _run["job_id"],
                "running_job_origin": _run["origin"],
            }
        # Read the session's records before claiming, so a verification does
        # not start on top of the user's or the agent's running cell.
        busy = session_host.jobs.running()
        if busy:
            return {
                "error": "busy",
                "running_job_id": busy.get("job_id"),
                "running_job_origin": busy.get("origin"),
            }
        _seq += 1
        _run = {
            "job_id": f"{_ID_PREFIX}{_seq}",
            "title": title,
            "blocks": list(blocks),
            "cells": _workflow_doc.code_cells(blocks),
            "intent": intent,
            "started": time.monotonic(),
            "finished": None,
            "status": "running",
            "created": time.time(),
            "note": "Starting a scratch kernel (a few seconds)…\n",
            # The record the run ended with (_record, final); live until then.
            "record": None,
            "error": None,
            "host": None,
            # The scratch host's records, which outlive its kernel.
            "log": None,
            # Each cell's id in those records once sent; the last is what a
            # stop names.
            "cell_jobs": [None] * len(_workflow_doc.code_cells(blocks)),
            # Set by interrupt: no further cell is sent.
            "stopping": False,
            "stop_reason": None,
            "writer": writer,
            "writer_label": writer_label,
            "origin": origin,
            "discarded": False,
            "saved_path": None,
            "draft_path": None,
        }
        run = _run

    _write_draft(run)

    thread = threading.Thread(
        target=_execute, args=(run,), name="scratch-verify", daemon=True
    )
    thread.start()
    return {"job_id": run["job_id"]}


def _finish(run, status, error=None):
    with _lock:
        if run["status"] != "running":
            return  # already discarded; the first verdict stands
        run["record"] = _record(run, full=True, final=True)
    # Written *before* the verdict is published, and outside the lock (which is
    # held only to take, read and release -- see where it is declared). The
    # order matters: the status is what every reader waits on, so spooling
    # after it would let a caller see a passed run whose document does not
    # exist yet, and report `saved_path: None` for a run that has one.
    if status == "ok" and run["record"] is not None:
        _spool(run)
    with _lock:
        if run["status"] != "running":
            return  # discarded while the document was being written
        run["status"] = status
        run["finished"] = time.monotonic()
        if error:
            run["error"] = error
        # `verified()` reads the status rather than a slot set here: what the
        # download is offered for is a whole run, and a document is not a
        # partial one -- half a workflow that stops at a NameError is a report,
        # which the record already is.


def _write_draft(run):
    """Write this attempt's document to its draft file. Never raises.

    **The draft is the working copy, and it is written whether the run passes or
    not.** A failed attempt is the one a person most wants to open, and the
    agent's next attempt should edit a file rather than retype the document from
    memory -- which is also how a human edit survives: the file is what gets
    sent back.

    Kept in the document's own spelling, not as ``.ipynb``: it is text to edit,
    it diffs, and it is exactly the format the tool takes, so a draft read back
    is a draft that can be sent back unchanged.

    One file per title, overwritten. Attempts at one workflow are drafts of one
    document, not a history worth keeping -- the history that is worth keeping
    is the record a pass promotes it to.
    """
    from .._config import get_workflow_dir

    try:
        drafts = get_workflow_dir() / _DRAFT_DIR
        drafts.mkdir(parents=True, exist_ok=True)
        path = drafts / _notebook.draft_filename(run["title"])
        path.write_text(_workflow_doc.to_markdown(run["blocks"]), encoding="utf-8")
    except OSError:
        logger.exception("Could not write the workflow draft")
        return
    with _lock:
        run["draft_path"] = str(path)


def _spool(run):
    """Write the workflow this run proved into the spool. Never raises.

    The document outlives the page's offer of it. A later attempt that fails
    takes the download away -- deliberately, because the page shows one run --
    but the file it wrote is still there, which is what makes that a UI decision
    rather than data loss.

    A pass **promotes**: the record is written and the draft that became it is
    removed, so a draft on disk means "this one has not passed yet". The draft
    is only retired once the record exists, so there is no moment with neither.

    Best-effort by construction: a verification that passed and could not be
    written is still a verification that passed, and reporting a disk error as a
    failed workflow would be a lie about the user's code.
    """
    from .._config import get_workflow_dir

    try:
        path = get_workflow_dir() / _notebook.suggested_workflow_filename(run["title"])
        nb = _notebook.build_workflow_notebook(
            {**run["record"], "blocks": run["blocks"]}
        )
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the verification stands either way
        logger.exception("Could not spool the verified workflow")
        return
    with _lock:
        run["saved_path"] = str(path)
        draft = run["draft_path"]
        run["draft_path"] = None
    if draft:
        try:
            Path(draft).unlink()
        except OSError:
            logger.debug("Could not remove the promoted draft %s", draft)
    _prune_spool()


def _prune_spool():
    """Keep only the newest :data:`_SPOOL_KEEP` spooled notebooks; best-effort.

    Run after the current one is written, so the newest always survives -- the
    same shape as ``_shim._prune_session_logs``, for the same reason.
    """
    from .._config import get_workflow_dir

    try:
        spooled = sorted(
            get_workflow_dir().glob("*.ipynb"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for old in spooled[_SPOOL_KEEP:]:
        try:
            old.unlink()
        except OSError:
            pass


def _execute(run):
    """Bring the scratch kernel up, run the cells, take the kernel back down."""
    host = None
    try:
        host = _host_factory()
        with _lock:
            if run["discarded"]:
                return
        host.start()
        with _lock:
            # Published only once the kernel is up, so "no host yet" means
            # exactly "still bringing up" -- which is the window `interrupt` has
            # to answer for itself. Teardown does not depend on this: the
            # `finally` below holds the host either way.
            run["host"] = host
            run["log"] = host.jobs
            run["note"] = "Scratch kernel ready; running the workflow…\n"
        _run_cells(run, host)
    except Exception as exc:  # noqa: BLE001 - the verdict, not a crash of ours
        # A scratch kernel that dies IS the answer -- an OOM means the workflow
        # does not fit -- so report it as a failed verification rather than as a
        # broken tool.
        _finish(run, "error", f"The scratch kernel failed: {exc}")
    finally:
        _discard_host(host)
        with _lock:
            run["host"] = None
        if run["status"] == "running":  # no branch above reached a verdict
            _finish(run, "error", "The verification ended without a result.")


def _run_cells(run, host):
    """Send the cells one request each, each once the one before ended ok.

    **Stops at the first failure.** The cells after it were written against the
    state the failed one was supposed to produce, so running them anyway would
    report a cascade of consequences as if they were separate defects. The rest
    are marked ``skipped`` (:func:`_record`), so the report says how far the
    workflow got.

    Bare cells (``run_cell(bare=True)``): a verification runs the document and
    nothing else, so the document's own first cell builds its ``client``.
    """
    n = len(run["cells"])
    for i, code in enumerate(run["cells"]):
        with _lock:
            # Checked and sent under the lock a stop takes: a stop either finds
            # this cell to interrupt, or keeps it from being sent.
            if run["discarded"]:
                return
            if run["stopping"]:
                _finish(
                    run,
                    "interrupted",
                    run["stop_reason"] or "The verification was stopped.",
                )
                return
            job_id = host.jobs.new_id()
            run["cell_jobs"][i] = job_id
            run["note"] = f"Running cell {i + 1} of {n}…\n"
            host.run_cell(code, job_id, run["origin"], bare=True)
        out = _await_cell(run, host, job_id)
        if out is None:
            return  # discarded
        if out.get("died"):
            _finish(
                run,
                "error",
                "The scratch kernel died while running the workflow -- most "
                "often out of memory.",
            )
            return
        if out["status"] != "ok":
            _finish(run, out["status"], out["error_text"] or None)
            return
    _finish(run, "ok")


def _await_cell(run, host, job_id):
    """Wait for the cell *job_id* to end; its outcome, or None if the run was
    discarded meanwhile.

    Its record ends on the request's idle, or on its shell reply, which cannot
    be lost (``KernelHost.run_cell``). A death ends it too: this host runs no
    watchdog, and a death is the verdict (an OOM means the workflow does not
    fit), so it is checked for here.
    """
    while True:
        with _lock:
            if run["discarded"]:
                return None
        out = host.jobs.outcome(job_id)
        if out["status"] != "running":
            return out
        if not host.is_alive():
            host.jobs.kernel_gone("the scratch kernel died")
            return {**host.jobs.outcome(job_id), "died": True}
        time.sleep(0.1)


def _discard_host(host):
    """Take the scratch kernel down, whatever state it is in. Never raises."""
    if host is None:
        return
    try:
        host.shutdown()
    except Exception:  # noqa: BLE001 - a kernel we are throwing away anyway
        logger.exception("Failed to shut down the scratch kernel")


def interrupt(reason=None, origin="user", writer=None):
    """Stop the running verification, leaving its kernel up if it stops.

    Returns ``None`` when there is no verification to stop -- which the caller
    must treat as "ask the session kernel instead", not as "nothing is running":
    a run that ended between the check and this call leaves the session kernel
    free to have started something.

    Mirrors the session kernel's vocabulary (``{"refused": "not_owner"}``,
    ``{"interrupted": True}``) so the tool surface routes to whichever kernel
    holds the running job without a second one, plus ``"killed"`` when the
    cells had to be taken with the process.

    **This is the guaranteed stop for a verification**, which the session
    kernel's interrupt deliberately is not: no further cell is sent, the one
    running is interrupted, and if the run has not ended within
    :data:`_INTERRUPT_GRACE` seconds the process goes.

    **Who may stop it** is the rule the session kernel's stop follows: a client
    other than the one that asked for it gets ``not_owner``, a writer of
    another origin ``foreign_job``, and ``origin="user"`` is exempt, so the
    person at the machine can stop a verification they did not start.
    """
    with _lock:
        if _run is None or _run["status"] != "running":
            return None
        run, host = _run, _run["host"]
        if origin != "user":
            if writer is not None and run["writer"] not in (None, writer):
                return {"refused": "not_owner", "job_id": run["job_id"]}
            if run["origin"] != origin:
                return {
                    "refused": "foreign_job",
                    "origin": run["origin"],
                    "job_id": run["job_id"],
                }
        run["stopping"] = True
        run["stop_reason"] = reason
        current = _current_cell(run)
    if host is None:
        # Still bringing the kernel up: stopping means discarding the attempt.
        discard(reason=reason)
        return {"interrupted": True, "job_id": run["job_id"]}
    if current is not None:
        try:
            # A cell that has just ended is refused, and the run stops before
            # the next one (_run_cells).
            host.interrupt_job(current, reason=reason)
        except Exception:  # noqa: BLE001 - a kernel that cannot answer is killed below
            logger.debug("scratch interrupt failed", exc_info=True)

    deadline = time.monotonic() + _INTERRUPT_GRACE
    while time.monotonic() < deadline:
        with _lock:
            if run["status"] != "running":
                return {"interrupted": True, "job_id": run["job_id"]}
        time.sleep(0.1)
    # It did not land. Take the process: see _INTERRUPT_GRACE for why that is
    # this kernel's guaranteed stop and not the user's session's.
    discard(reason=reason)
    return {"interrupted": True, "job_id": run["job_id"], "killed": True}


def discard(reason=None):
    """Throw away an in-flight verification and its kernel. Returns its id or None.

    ``restart_kernel`` calls this, and not because the user asked to kill the
    verification: it holds the slot, so leaving it running would leave the
    freshly restarted kernel able to accept nothing, and a wedged verification
    with no escape hatch — which is precisely what ``restart_kernel`` is
    documented to be. Discarding is cheap; a verification is seconds to start
    again.
    """
    with _lock:
        if _run is None or _run["status"] != "running":
            return None
        run = _run
        run["discarded"] = True
        host = run["host"]
    _finish(run, "interrupted", reason or "The verification was discarded.")
    # Outside the lock: shutdown is a process teardown, and the slot is already
    # released by the status change above.
    _discard_host(host)
    return run["job_id"]


def reset():
    """Forget everything (tests, and a launcher reconfiguring the session)."""
    global _run, _seq
    discard()
    with _lock:
        _run, _seq = None, 0
