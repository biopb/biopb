"""Serialize a kernel session to a Jupyter notebook — two documents, one writer.

:func:`build_notebook` is the **audit export**: every retained job in order,
faithfully, dead ends included. It is built here, from job snapshots.

:func:`build_workflow_notebook` is the **workflow export**, and it is not built
here in the same sense: the document is the agent's, written as markdown with
fenced cells and parsed by :mod:`_workflow_doc`. This module adds the two things
the *run* owns — the provenance cell and the captured outputs — and serializes
the result. Prose that is not the run's is not ours to write.

The two answer different questions — "what happened here?" and "what should I
run again?" — and neither substitutes for the other, so both ship.

Runs in the *MCP server process* (no kernel/Qt imports): the observe UI rounds a
:func:`biopb_mcp.mcp._jobs.export` read off the kernel main thread, then hands
the list of job snapshots here to build an nbformat-v4 document.

The notebook is an **audit record first, a runnable script second.** The cells
faithfully reproduce, in order, every job's source and captured output. Re-running
top-to-bottom works only for self-contained, in-namespace computation: external
state is *not* captured — tensor-server source ids and napari viewer layers from
the live session do not exist on a fresh kernel, so source-chaining / viewer
cells need the same live server (or hand edits). The bootstrap cell rebuilds
``np``/``da``/``client``/``ops``, an empty viewer (via the ``%gui qt`` magic),
and the user's kernel plugins, on a best-effort basis — the plugins from *this*
machine's `~/.config/biopb/kernel`, which need not be the ones the session had.
``nbformat`` is intentionally not a dependency — the v4 schema is small and
hand-built here.
"""

import datetime

# Best-effort namespace reconstruction, mirroring _bootstrap.py (steps 2-5) but
# synchronous and guarded. Runs in the notebook's own kernel at re-run time, so
# it reads the *current* config — the audit notebook is not pinned to the config
# captured at export. The live napari layers, the distributed dask cluster, and
# any interactive state are not reproducible (see module docstring).
_BOOTSTRAP_HEAD = """\
# === biopb-mcp session bootstrap (best-effort audit reconstruction) ===
# Rebuilds np / da, the data-plane `client`, the compute-plane `ops`, and an
# (empty) napari `viewer` so the recorded cells below can, in principle, re-run.
# NOT a faithful replica: tensor-server source ids and viewer layers from the
# original session are gone, and the dask cluster is not reproduced.
import numpy as np
import dask.array as da

from biopb_mcp._config import load_config
from biopb_mcp._connection import TensorConnection
from biopb_mcp.mcp._process_ops import build_ops_from_config

config = load_config()
_conn = TensorConnection()
_conn.auto_connect()          # synchronous best-effort connect (no async service here)
client = _conn.client

ops = build_ops_from_config(config, lambda: _conn.client)
"""

#: The audit export's extra stage. A session transcript is full of viewer cells,
#: so the reconstruction needs somewhere for them to land -- best-effort, and
#: None when there is no display, in which case those cells fail where they are.
#: The *workflow* export has no bootstrap cell at all: its document builds its
#: own environment (``biopb_mcp.workflow_env``) and the verification runs that
#: cell, so there is nothing left here to rebuild on its behalf.
_BOOTSTRAP_VIEWER = """
# Best-effort empty viewer via the Qt magic; degrades to None when headless
# (e.g. `nbconvert --execute` with no display), in which case viewer cells fail.
try:
    get_ipython().run_line_magic("gui", "qt")
    import napari

    viewer = napari.Viewer()
except Exception as _exc:  # noqa: BLE001 - audit notebook tolerates no display
    viewer = None
    print("napari viewer unavailable (audit notebook):", _exc)
"""

_BOOTSTRAP_PLUGINS = """
# User kernel plugins (~/.config/biopb/kernel/*.py and biopb_mcp.namespace entry
# points), loaded by the kernel's own loader so a cell calling one of them --
# `rolling_ball.subtract_background(...)` -- resolves the same name it did in the
# session. Last, like the kernel's step 7b, so a plugin can reference the handles
# above. Fail-open per plugin, as in the kernel; a plugin this machine does not
# have simply does not bind, and the cell using it fails where it is used.
try:
    from biopb_mcp.mcp import _requires
    from biopb_mcp.mcp._bootstrap import _load_namespace_plugins

    _load_namespace_plugins(get_ipython(), config)
    _bound = _requires._LOADED_FILES + _requires._LOADED_ENTRY_POINTS
    print("kernel plugins:", ", ".join(_bound) if _bound else "(none)")
except Exception as _exc:  # noqa: BLE001 - a plugin gap must not stop the rebuild
    print("kernel plugins not loaded:", _exc)
"""

#: The audit export: everything the session had, viewer included.
BOOTSTRAP_SRC = _BOOTSTRAP_HEAD + _BOOTSTRAP_VIEWER + _BOOTSTRAP_PLUGINS


def _lines(text):
    """Split *text* into the line list nbformat uses (newlines kept, no trailing)."""
    if not text:
        return []
    return text.splitlines(keepends=True)


def _code_cell(source, *, outputs=None, metadata=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata or {},
        "outputs": outputs or [],
        "source": _lines(source),
    }


def _markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }


def _fmt_ts(epoch):
    """Local wall-clock ``YYYY-MM-DD HH:MM:SS`` for an epoch, or ``"?"``."""
    if not epoch:
        return "?"
    try:
        return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return "?"


def _notebook(cells):
    """Wrap *cells* in the nbformat-v4 envelope both builders emit.

    The session export and the workflow export differ in their cells and in
    nothing else; written twice, a metadata change would have to be noticed in
    the other one.
    """
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _job_outputs(snap):
    """Build the nbformat output list for one job snapshot.

    ``stdout`` -> a stream output; ``result_text`` (the repr of the job's last
    expression) -> an execute_result; ``error_text`` (tracebacks and/or the
    user-attributed cancel/interrupt reason) -> a stderr stream so a stopped job
    visibly shows why it stopped.
    """
    outputs = []
    stdout = snap.get("stdout") or ""
    if stdout:
        outputs.append(
            {"output_type": "stream", "name": "stdout", "text": _lines(stdout)}
        )
    result_text = snap.get("result_text") or ""
    if result_text:
        outputs.append(
            {
                "output_type": "execute_result",
                "execution_count": None,
                "data": {"text/plain": _lines(result_text)},
                "metadata": {},
            }
        )
    error_text = snap.get("error_text") or ""
    if error_text:
        outputs.append(
            {"output_type": "stream", "name": "stderr", "text": _lines(error_text)}
        )
    return outputs


def _intent_cell(snap):
    """Markdown cell carrying the *why* recorded with a job, or ``None``.

    The code is the one thing an execute_code session records natively; the goal
    behind it exists only in the caller's head, and nowhere in the export unless
    it was passed in. Rendered as its own cell above the code rather than folded
    into the header comment so it survives being read as prose — and so a chat
    loop, which will fill this field with the user's own turn, has a cell shape
    already waiting for it.

    Optional and free text: a job with no intent gets no cell, which is why this
    returns ``None`` rather than an empty one.
    """
    intent = (snap.get("intent") or "").strip()
    if not intent:
        return None
    job_id = snap.get("job_id", "?")
    origin = snap.get("origin") or "mcp"
    return _markdown_cell(f"**{job_id}** · {origin} — {intent}")


def _job_cell(snap):
    job_id = snap.get("job_id", "?")
    status = snap.get("status", "?")
    elapsed = snap.get("elapsed", "?")
    # Who ran it. This is what makes the export an audit rather than a
    # transcript: agent and user cells interleave in one kernel, and read
    # without provenance a human's `mask = mask > 0.7` is indistinguishable from
    # the agent's own work. Older records carry no origin, so default rather
    # than assert -- an export must never fail on a field added later.
    origin = snap.get("origin") or "mcp"
    header = (
        f"# [{job_id} · {origin} · {status} · {elapsed}s · "
        f"{_fmt_ts(snap.get('created'))}]\n"
    )
    source = header + (snap.get("code") or "")
    return _code_cell(
        source,
        outputs=_job_outputs(snap),
        metadata={
            "biopb": {
                "job_id": job_id,
                "origin": origin,
                # Stripped, like _intent_cell: a whitespace-only intent must
                # not survive in the metadata as one that exists while the
                # rendered note says it does not.
                "intent": (snap.get("intent") or "").strip(),
                "status": status,
                "elapsed": elapsed,
                "created": snap.get("created"),
            }
        },
    )


#: The provenance cell, prepended to every saved workflow.
#:
#: **Server-written, not the agent's.** The body of the document is the agent's
#: -- prose, headings, whatever it wants to say -- but the claims about what the
#: verification proved are not: an author that writes both can write "verified
#: correct" over a run that only proved the cells execute. So the document is
#: the agent's and the verdict is ours, and a reader can tell which is which by
#: where it sits.
_WORKFLOW_INTRO = (
    "Verified {ts}{ncells}.\n\n"
    "Every cell below ran, in this order, in a **scratch kernel** — a second "
    "process spawned for the verification and discarded after it. It was given "
    "nothing: no `client`, no `ops`, no `np`, no plugins, and no napari viewer. "
    "Whatever this notebook needs, the cells below build, which is why running "
    "them here proves they run for you.\n\n"
    "**What the run proves.** Every cell executed without raising, on a bare "
    "kernel with biopb-mcp installed — no leftover variable, no layer the "
    "session happened to have, no module some earlier cell imported. That is "
    "the whole class of defect that makes a session transcript unrunnable. It "
    "is not a claim that the numbers are right; that is yours to check, and the "
    "outputs are kept below so there is something to check against.\n\n"
    "**The prose is the author's, and unverified.** Running a cell checks the "
    "code; nothing checks the sentence above it.\n\n"
    "**Plugins are yours, not the session's.** `workflow_env()` loads them from "
    "*your* `~/.config/biopb/kernel`, so a workflow calling a plugin this "
    "machine does not have binds nothing and the cell using it raises "
    "`NameError`.\n\n"
    "**What it does not prove.** A scratch *process* is not a scratch *world*. "
    "The run used the same tensor server and the same filesystem as the "
    "session, so anything the workflow uploaded or wrote is really there. And "
    "external state is unchanged by any of it — tensor-server source ids still "
    "have to exist, and `workflow_env()` still needs a running control "
    "(`biopb control start`) or `$BIOPB_TENSOR_URL`."
)


_TITLE = "# biopb-mcp session — audit export\n"

_INTRO = (
    "Exported {ts} · {n} job(s).\n\n"
    "This notebook is an **audit record** of an `execute_code` session. The "
    "first code cell rebuilds the namespace (`np`, `da`, `client`, `ops`, an "
    "empty `viewer`, and this machine's kernel plugins) on a best-effort basis; "
    "each cell below is one job, with its "
    "recorded output. A job that was submitted with a stated intent carries it "
    "as the markdown note above the code — the only record of *why* a cell was "
    "run, and present only where whoever ran it supplied one.\n\n"
    "**Runnability caveats.** External state is not captured: tensor-server "
    "source ids and napari viewer layers from the live session do not exist on a "
    "fresh kernel, so any cell that chains `ops` source ids or reads `viewer` "
    "layers needs the same live server (or edits). In-namespace Python variables "
    "*do* carry across cells. Cells whose header reads `interrupted` / `error` "
    "are kept verbatim — re-running one may re-trigger the same hang or "
    "failure, so skip or edit it. Only the most recent jobs are retained, so a "
    "long session may be missing its start. `auto_connect()` asks the control "
    "plane where the data plane is, so a re-run needs a running control "
    "(`biopb control start`) or `$BIOPB_TENSOR_URL`; under a headless "
    "`nbconvert --execute` the `viewer` becomes `None` and viewer cells fail."
)


def build_notebook(jobs):
    """Build an nbformat-v4 notebook dict from a list of job snapshots.

    *jobs* is the oldest-first list returned by ``_jobs.export()`` (each a
    ``_Job.snapshot()`` dict). The result is a plain dict ready to
    ``json.dumps`` into a ``.ipynb`` file.
    """
    jobs = jobs or []
    intro = _INTRO.format(ts=_fmt_ts(_now_epoch()), n=len(jobs))

    cells = [_markdown_cell(_TITLE + "\n" + intro), _code_cell(BOOTSTRAP_SRC)]
    if jobs:
        for snap in jobs:
            note = _intent_cell(snap)
            if note is not None:
                cells.append(note)
            cells.append(_job_cell(snap))
    else:
        cells.append(_markdown_cell("_No jobs were recorded in this session._"))

    return _notebook(cells)


def _workflow_cells(blocks, results):
    """The document's blocks as notebook cells, with the run's outputs in them.

    *results* is the verification's per-cell record, in the same order as the
    document's code blocks -- they are the same list, so they are zipped by
    position rather than matched. A missing result renders as a code cell with
    no output, which is what a document read back and not yet re-run looks like.

    No provenance header comment, unlike :func:`_job_cell`. An audit cell needs
    one because a reader has to know who ran it; a workflow has one author and
    one purpose, and a banner on every cell is noise in a document someone is
    meant to read and edit.
    """
    out = []
    i = 0
    for block in blocks:
        if block["kind"] != "code":
            out.append(_markdown_cell(block["text"]))
            continue
        result = results[i] if i < len(results) else {}
        i += 1
        out.append(_code_cell(block["text"], outputs=_job_outputs(result)))
    return out


def build_workflow_notebook(record):
    """Build an nbformat-v4 notebook from a verification record.

    *record* is ``_scratch.verified()``: the run's per-cell results plus the
    document they came from. A partial run is a report rather than a document
    (``_jobs._run``), so only a fully-successful one reaches here.

    The notebook is the agent's document with two things added, both of them the
    run's: the provenance cell at the top, and the outputs in the code cells.
    Nothing else is inserted -- there is no bootstrap cell any more, because the
    document builds its own environment and the run proved that it does.
    """
    record = record or {}
    blocks = record.get("blocks") or []
    results = record.get("cells") or []
    title = (record.get("title") or "").strip() or "Verified workflow"
    intro = _WORKFLOW_INTRO.format(
        ts=_fmt_ts(record.get("created")),
        ncells=f" · {len(results)} cell(s)" if results else "",
    )
    cells = [_markdown_cell(f"# {title}\n\n" + intro)]
    cells.extend(_workflow_cells(blocks, results))
    return _notebook(cells)


def _now_epoch():
    """Current epoch seconds (own helper so tests can monkeypatch the stamp)."""
    return datetime.datetime.now().timestamp()


def _stamp():
    """``YYYYMMDD-HHMMSS`` for a suggested download filename."""
    return datetime.datetime.fromtimestamp(_now_epoch()).strftime("%Y%m%d-%H%M%S")


def suggested_filename():
    """``biopb-mcp-session-YYYYMMDD-HHMMSS.ipynb`` for the download."""
    stamp = _stamp()
    return f"biopb-mcp-session-{stamp}.ipynb"


def _slug(title, limit=48):
    """A filename-safe stem from a workflow title, or ``""`` when there is none.

    Conservative: lowercase, and anything that is not a letter, digit, or dash
    becomes a dash. A title is free text typed by whoever ran the verification
    and this becomes a filename on their disk.
    """
    out = []
    for ch in (title or "").strip().lower():
        out.append(ch if ch.isalnum() and ch.isascii() else "-")
    slug = "-".join(part for part in "".join(out).split("-") if part)
    return slug[:limit].strip("-")


def draft_filename(title=""):
    """``<title>.md`` for the draft, or ``workflow.md``.

    No stamp, unlike the record: attempts at one workflow are drafts of one
    document, and a timestamp per attempt would make a directory of dead ends
    where what is wanted is the current text.
    """
    slug = _slug(title)
    return f"{slug}.md" if slug else "workflow.md"


def suggested_workflow_filename(title=""):
    """``biopb-<title>-YYYYMMDD-HHMMSS.ipynb`` for the workflow download.

    The stamp stays even with a title: a workflow is verified more than once
    while it is being worked on, and two downloads of the same title must not
    silently be the same file.
    """
    stamp = _stamp()
    slug = _slug(title)
    return f"biopb-{slug}-{stamp}.ipynb" if slug else f"biopb-workflow-{stamp}.ipynb"
