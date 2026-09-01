"""Verifying a workflow in a scratch kernel: the process, and the slot it takes.

Runs **in the MCP server process**. ``verify_workflow`` asks one question — will
this program run on its own? — and the only honest way to answer it is to run it
somewhere that has none of the session's state. This module owns that somewhere:
a second kernel, spawned per verification, discarded after it.

Why a process and not a namespace, why the session's own display, and what
"discarded" does *not* cover: ``docs/verification-scratch-kernel.md``.

Three things live here, and they are one module because they are one decision.

* **The kernel.** A full bootstrap with the user-facing parts left out
  (``_bootstrap.ENV_SCRATCH``), no watchdog, and no window-close pipe. The
  watchdog is off deliberately: for the session kernel a respawn is recovery,
  but for this one **death is the verdict**. An OOM means "this workflow does
  not fit", and respawning would re-run a workflow that just killed a process,
  three more times, each allocating gigabytes on a machine already under
  pressure.
* **The run.** Submitted to that kernel through the same ``_jobs.submit``
  door as any other job, so the per-cell record, the output capture and the
  interrupt path are the ones that already exist.
* **The slot.** Two kernels must not become two schedulers. The dask cluster is
  shared and finite, and the agent's whole model is "one cell at a time", so a
  verification takes the same slot ordinary work does — see :func:`start`.

**No ``_verified`` slot.** The record lives here, in the child, for the length of
the session: the answer is "this run passed, just now", not "some run passed at
some point in a kernel that has since been restarted".
"""

import logging
import threading
import time

from . import _kernel_rpc

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

#: Verifications that have finished, oldest first. Kept because the observe page
#: shows this kernel's runs as a list beside the session kernel's, and a list
#: that forgets everything but the newest cannot say what the agent has already
#: tried. Bounded: a run holds its per-cell record, and nobody scrolls past a
#: few.
_history = []
_HISTORY_MAX = 20

# The last verification whose every cell ran, as its record dict. Kept across
# later attempts: a run that fails afterwards does not un-verify the one that
# passed, and the user may well be mid-way through a second attempt when they
# decide to save the first.
_verified = None

#: How long an interrupt waits for the cells to stop before taking the process.
#:
#: An interrupt on the *session* kernel is best-effort by necessity: it raises
#: KeyboardInterrupt into the job thread, which a blocking C call (a gRPC fetch,
#: a native dask compute) does not notice until it returns to Python -- and the
#: guaranteed stop, a group-kill, costs the user their whole session, so it is
#: theirs to ask for.
#:
#: None of that holds here. A scratch kernel has no variables anyone wants, no
#: layers, no session -- killing it costs nothing, and it was going to be
#: discarded seconds later anyway. So the interrupt is only *briefly*
#: best-effort: long enough for a clean stop to give the better record (the full
#: per-cell one, rather than the last ledger polled), then the process goes.
#:
#: Which is what keeps `restart_kernel` out of this. Without the escalation, a
#: verification wedged in a C call would leave the agent nothing but the tool
#: that destroys the user's session, to kill a process built to be thrown away.
_INTERRUPT_GRACE = 5.0

#: Job ids issued here, in their own namespace. The session kernel issues
#: ``job-N``; a verification never runs there, so a distinct prefix is what lets
#: ``poll_job`` route an id to the kernel that owns it without asking either.
_ID_PREFIX = "verify-"


def _elapsed(run):
    return round((run["finished"] or time.monotonic()) - run["started"], 1)


def _snapshot(run):
    """A verification rendered in the shape ``poll_job`` renders a job in.

    Deliberately the same keys a ``_jobs`` snapshot carries (``job_id``,
    ``status``, ``elapsed``, ``stdout``, ``verify``), so the tool surface has one
    renderer rather than two: ``_server._format_job_status`` already knows what
    to do with a ``verify`` record, and this is the same record.
    """
    return {
        "job_id": run["job_id"],
        "status": run["status"],
        "elapsed": _elapsed(run),
        "stdout": run["note"],
        "error_text": run["error"] or "",
        "result_text": "",
        "verify": run["record"],
        "origin": "mcp",
        "intent": run["intent"],
        "title": run["title"],
        "cell_count": len(run["cells"]),
    }


#: How ``_jobs.submit`` joins verification cells into the job's ``code``. Spelled
#: again here so the observe detail view shows the same program the kernel ran.
_CELL_SEP = "\n\n# ---\n\n"


def detail(job_id):
    """A verification in the shape the observe detail view reads, or ``None``.

    The page lists a running verification as one row (``_observe._api_jobs``),
    and a row that cannot be opened is worse than no row: the session kernel has
    never heard of a ``verify-N`` id, so asking it returns 404 and the run shows
    neither progress nor output.

    What it shows instead of a job's stdout is the run's own progress -- which
    stage the bring-up reached, then a line per cell as each finishes. The full
    per-cell output belongs to the notebook, not to a ledger (see
    ``_jobs._Cell.snapshot``).
    """
    snap = poll(job_id)
    if snap is None:
        return None
    run = _find(job_id)
    cells = list(run["cells"]) if run is not None else []
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
        "code": _CELL_SEP.join(cells),
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


def _find(job_id):
    """The run dict for *job_id* -- current or finished -- or ``None``."""
    with _lock:
        if _run is not None and _run["job_id"] == job_id:
            return _run
        for run in _history:
            if run["job_id"] == job_id:
                return run
    return None


def poll(job_id):
    """The snapshot for *job_id*, or ``None`` if this is not a run we know."""
    run = _find(job_id)
    return None if run is None else _snapshot(run)


def runs_view():
    """Every verification this session has run, oldest first.

    The row shape the observe page renders a job in, so its verification pane is
    the same list component as its session pane rather than a second one.

    Where a session job's "why" is the intent its writer gave that cell, a
    verification has neither: the cells arrive as bare code and the run's intent
    is synthesized from the title (``_server.verify_workflow``). So the title
    *is* the why, and the row says it once -- the count is the what. A workflow
    cell needs no intent of its own the way a transcript cell does: it was
    written to be read.
    """
    with _lock:
        runs = [*_history, _run] if _run is not None else list(_history)
        return [
            {
                "job_id": run["job_id"],
                "status": run["status"],
                "origin": "mcp",
                "elapsed": _elapsed(run),
                "code_preview": (
                    f"{len(run['cells'])} cell"
                    + ("s" if len(run["cells"]) != 1 else "")
                ),
                "intent_preview": run["title"],
                "verify": True,
            }
            for run in runs
        ]


def verified():
    """The last fully-successful verification's record, or ``None``."""
    with _lock:
        return _verified


def verified_summary():
    """A one-line description of :func:`verified`, or ``None``.

    Carried on the observe poll so the page can offer the workflow download
    without a second round trip per second for a value that changes rarely.
    """
    with _lock:
        if _verified is None:
            return None
        return {
            "title": _verified.get("title", ""),
            "cells": len(_verified.get("cells") or []),
            "created": _verified.get("created"),
        }


def start(cells, title, session_host, intent="", writer=None, writer_label=""):
    """Take the slot and begin verifying *cells* in a fresh scratch kernel.

    Returns ``{"job_id": ...}``, or ``{"error": "busy", ...}`` naming what holds
    the slot, or ``{"error": <reason>}`` when no scratch kernel can be built.

    *writer* is the client this run belongs to. It is passed straight through to
    the scratch kernel's ``_jobs.submit``, which claims that kernel for it -- so
    the one-agent rule on a verification is the *same* rule, enforced by the same
    code, as on any other job. A second client is then refused an interrupt
    there exactly as it is on the session kernel, with no second gate to keep in
    step. It is kept here as well, but only for the seconds before the kernel
    exists to answer for itself (see :func:`interrupt`).

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
    *asking* the session kernel, which is a check-then-act with a round trip in
    it — two jobs can still start within a millisecond of each other. The
    consequence is bounded (two jobs briefly sharing a warm cluster, not a
    corrupted anything), and closing it properly means the session child issuing
    every job id, which is a larger change than this one.
    """
    global _run, _seq
    if _host_factory is None:
        return {"error": "Verification is unavailable: no scratch kernel configured."}

    with _lock:
        if _run is not None and _run["status"] == "running":
            return {
                "error": "busy",
                "running_job_id": _run["job_id"],
                "running_job_origin": "mcp",
            }
        # Ask the session kernel before claiming, so a verification does not
        # start on top of the user's or the agent's running cell.
        busy, _res, _w = _kernel_rpc._run_job_call(session_host, "running_job")
        if busy:
            return {
                "error": "busy",
                "running_job_id": busy.get("job_id"),
                "running_job_origin": busy.get("origin"),
            }
        if _run is not None:
            _history.append(_run)
            del _history[:-_HISTORY_MAX]
        _seq += 1
        _run = {
            "job_id": f"{_ID_PREFIX}{_seq}",
            "title": title,
            "cells": list(cells),
            "intent": intent,
            "started": time.monotonic(),
            "finished": None,
            "status": "running",
            "note": "Starting a scratch kernel (a few seconds)…\n",
            "record": None,
            "error": None,
            "host": None,
            "writer": writer,
            "writer_label": writer_label,
            "discarded": False,
        }
        run = _run

    thread = threading.Thread(
        target=_execute, args=(run,), name="scratch-verify", daemon=True
    )
    thread.start()
    return {"job_id": run["job_id"]}


def _finish(run, status, error=None):
    global _verified
    with _lock:
        if run["status"] != "running":
            return  # already discarded; the first verdict stands
        run["status"] = status
        run["finished"] = time.monotonic()
        if error:
            run["error"] = error
        record = run["record"]
        if status == "ok" and record is not None:
            # Kept only when every cell ran. What it is kept *for* is the
            # workflow document, and a document is not a partial run: half a
            # workflow that stops at a NameError is a report, which the record
            # already is.
            _verified = record


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
            run["note"] = "Scratch kernel ready; running the workflow…\n"

        submitted, res, _w = _kernel_rpc._run_job_call(
            host,
            "submit",
            "",
            intent=run["intent"] or "verify workflow",
            writer=run["writer"],
            writer_label=run["writer_label"],
            verify_cells=run["cells"],
            verify_title=run["title"],
        )
        if submitted is None or "job_id" not in submitted:
            _finish(run, "error", _kernel_rpc._format_execute_result(res))
            return
        _poll_to_completion(run, host, submitted["job_id"])
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


def _poll_to_completion(run, host, kernel_job_id):
    """Watch the scratch kernel's verification job until it is terminal."""
    while True:
        with _lock:
            if run["discarded"]:
                return
        snap, res, _w = _kernel_rpc._run_job_call(host, "poll", kernel_job_id)
        if snap is None:
            _finish(run, "error", _kernel_rpc._format_execute_result(res))
            return
        record = snap.get("verify")
        with _lock:
            if record is not None:
                run["record"] = record
            run["note"] = snap.get("stdout") or run["note"]
        if snap.get("status") != "running":
            # Terminal: swap the polled ledger for the full record, once, before
            # the kernel holding it is discarded. Best-effort -- a kernel too far
            # gone to answer still has a verdict, and the polled heads are a
            # worse report but not a wrong one.
            full, _res, _w = _kernel_rpc._run_job_call(
                host, "verify_record", kernel_job_id
            )
            with _lock:
                if full is not None:
                    run["record"] = full
            _finish(run, snap.get("status"), snap.get("error_text") or None)
            return
        time.sleep(0.4)


def _discard_host(host):
    """Take the scratch kernel down, whatever state it is in. Never raises."""
    if host is None:
        return
    try:
        host.shutdown()
    except Exception:  # noqa: BLE001 - a kernel we are throwing away anyway
        logger.exception("Failed to shut down the scratch kernel")


def interrupt(reason=None, requester="user", writer=None):
    """Stop the running verification's cells, leaving its kernel up.

    Returns ``None`` when there is no verification to stop -- which the caller
    must treat as "ask the session kernel instead", not as "nothing is running":
    a run that ended between the check and this call leaves the session kernel
    free to have started something.

    Mirrors ``_jobs.interrupt_current``'s vocabulary (``{"refused":
    "not_owner"}``, ``{"interrupted": True}``) so the tool surface routes to
    whichever kernel holds the running job without a second one, plus
    ``"killed"`` when the cells had to be taken with the process.

    **This is the guaranteed stop for a verification**, which the session
    kernel's interrupt deliberately is not: a cooperative interrupt gets
    :data:`_INTERRUPT_GRACE` seconds to land, and then the process goes.

    **Who may stop it is the scratch kernel's own answer**, because ``start``
    claimed that kernel for the client that asked for the verification: a second
    client gets ``not_owner`` from the same check that guards any other job, and
    ``requester="user"`` is exempt there as it is here, so the person at the
    machine can still stop a verification they did not start.

    The one window that check cannot cover is the five to eight seconds before
    the kernel exists. The rule is the same, applied locally, so a stranger
    cannot discard an attempt during its bring-up either.
    """
    with _lock:
        if _run is None or _run["status"] != "running":
            return None
        run, host = _run, _run["host"]
        if (
            host is None  # no kernel to answer for itself yet
            and requester == "mcp"
            and writer is not None
            and run["writer"] not in (None, writer)
        ):
            return {"refused": "not_owner", "job_id": run["job_id"]}
    if host is None:
        # Still bringing the kernel up: there is nothing to interrupt yet, so
        # stopping means discarding the whole attempt.
        discard(reason=reason)
        return {"interrupted": True, "job_id": run["job_id"]}
    data, _res, _w = _kernel_rpc._run_job_call(
        host, "interrupt_current", reason, requester=requester, writer=writer
    )
    if data and data.get("refused"):
        return data
    if not data or not data.get("interrupted"):
        # Nothing was running in there after all -- the run is between cells, or
        # finishing. Let the poll loop reach its own verdict.
        return {"interrupted": False, "job_id": run["job_id"]}

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
    global _run, _verified, _seq
    discard()
    with _lock:
        _run, _verified, _seq = None, None, 0
        _history.clear()
