"""Serialize a kernel session to a Jupyter notebook — two documents, one writer.

:func:`build_notebook` is the **audit export**: every retained job in order,
faithfully, dead ends included. :func:`build_workflow_notebook` is the
**workflow export**: the cells of a verified workflow (``_jobs`` verification
run), which are a *rewrite* of that transcript rather than a selection from it,
and which are known to run because they just did.

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
BOOTSTRAP_SRC = """\
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
_conn.auto_connect()          # synchronous best-effort connect (audit; no async service)
client = _conn.client

ops = build_ops_from_config(config, lambda: _conn.client)

# Best-effort empty viewer via the Qt magic; degrades to None when headless
# (e.g. `nbconvert --execute` with no display), in which case viewer cells fail.
try:
    get_ipython().run_line_magic("gui", "qt")
    import napari

    viewer = napari.Viewer()
except Exception as _exc:  # noqa: BLE001 - audit notebook tolerates no display
    viewer = None
    print("napari viewer unavailable (audit notebook):", _exc)

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


_WORKFLOW_INTRO = (
    "Verified {ts}{ncells}.\n\n"
    "Each cell below ran, in this order, in a **scratch namespace** — one seeded "
    "with the kernel's own handles (`np`, `da`, `client`, `ops`, `viewer`) and "
    "the loaded kernel plugins, and nothing the session had bound since it "
    "started. That is what the first code cell rebuilds, so this notebook asks of "
    "a fresh kernel only what the verification run was given — with one gap worth "
    "naming: plugins are loaded from *the reader's* "
    "`~/.config/biopb/kernel`, so a workflow calling a plugin this machine does "
    "not have binds nothing, and the cell using it raises `NameError`. The "
    "bootstrap cell prints what bound.\n\n"
    "**What the run proves.** Every cell executed without raising, and no cell "
    "leaned on a variable it did not itself create — the defect that makes a "
    "session transcript unrunnable. It is not a claim that the numbers are "
    "right; that is the reader's to check, and the outputs are kept below so "
    "there is something to check against.\n\n"
    "**What it does not prove.** A scratch namespace isolates *bindings*, not "
    "the world: the napari viewer, `sys.modules`, and anything mutated in place "
    "were shared with the live session. A cell that reads an existing layer by "
    "name found one there, and will not on a fresh kernel. External state is "
    "unchanged by any of this — tensor-server source ids still have to exist, "
    "and `auto_connect()` still needs a running control (`biopb control start`) "
    "or `$BIOPB_TENSOR_URL`.{layers}"
)

_LAYERS_NOTE = (
    "\n\n**Layers this run added to the live viewer:** {names}. Shared viewer, "
    "so these are real additions, not a sandbox's."
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


def _workflow_cell(cell):
    """One verified cell: its source, and the output it produced when verified.

    No provenance header comment, unlike :func:`_job_cell`. An audit cell needs
    one because a reader has to know who ran it; a workflow cell has one author
    and one purpose, and a banner on every cell is noise in a document someone
    is meant to read and edit.

    A verification cell carries the same output keys a job snapshot does, so it
    renders through :func:`_job_outputs`. Its ``error_text`` is empty by
    construction -- ``verified()`` only yields fully-successful runs -- and if
    one ever were not, showing it beats dropping it.
    """
    return _code_cell(cell.get("code") or "", outputs=_job_outputs(cell))


def build_workflow_notebook(record):
    """Build an nbformat-v4 notebook from a verification record.

    *record* is ``_jobs.verified()`` — a fully-successful run, since a partial
    one is a report rather than a document (``_jobs._run``). Returns a plain
    dict ready to ``json.dumps``.
    """
    record = record or {}
    cells_in = record.get("cells") or []
    title = (record.get("title") or "").strip() or "Verified workflow"
    added = record.get("added_layers") or []
    intro = _WORKFLOW_INTRO.format(
        ts=_fmt_ts(record.get("created")),
        ncells=f" · {len(cells_in)} cell(s)" if cells_in else "",
        layers=_LAYERS_NOTE.format(names=", ".join(f"`{n}`" for n in added))
        if added
        else "",
    )

    cells = [_markdown_cell(f"# {title}\n\n" + intro), _code_cell(BOOTSTRAP_SRC)]
    cells.extend(_workflow_cell(c) for c in cells_in)
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


def suggested_workflow_filename(title=""):
    """``biopb-<title>-YYYYMMDD-HHMMSS.ipynb`` for the workflow download.

    The stamp stays even with a title: a workflow is verified more than once
    while it is being worked on, and two downloads of the same title must not
    silently be the same file.
    """
    stamp = _stamp()
    slug = _slug(title)
    return f"biopb-{slug}-{stamp}.ipynb" if slug else f"biopb-workflow-{stamp}.ipynb"
