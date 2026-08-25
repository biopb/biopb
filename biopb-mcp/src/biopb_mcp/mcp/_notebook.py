"""Serialize a recorded ``execute_code`` session to a Jupyter notebook.

Runs in the *MCP server process* (no kernel/Qt imports): the observe UI rounds a
:func:`biopb_mcp.mcp._jobs.export` read off the kernel main thread, then hands
the list of job snapshots here to build an nbformat-v4 document.

The notebook is an **audit record first, a runnable script second.** The cells
faithfully reproduce, in order, every job's source and captured output. Re-running
top-to-bottom works only for self-contained, in-namespace computation: external
state is *not* captured — tensor-server source ids and napari viewer layers from
the live session do not exist on a fresh kernel, so source-chaining / viewer
cells need the same live server (or hand edits). The bootstrap cell rebuilds
``np``/``da``/``client``/``ops`` and an empty viewer (via the ``%gui qt`` magic)
on a best-effort basis. ``nbformat`` is intentionally not a dependency — the v4
schema is small and hand-built here.
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

from biopb import _algorithms
from biopb_mcp._config import load_config, get_setting
from biopb_mcp._connection import TensorConnection
from biopb_mcp.mcp._process_ops import build_ops

config = load_config()
_conn = TensorConnection()
_conn.auto_connect()          # synchronous best-effort connect (audit; no async service)
client = _conn.client

_mb = get_setting(config, "grpc.max_message_size_mb") * 1024 * 1024
ops = build_ops(
    client_getter=lambda: _conn.client,
    server_urls=_algorithms.servers_from_config(config),
    op_names_timeout=get_setting(config, "timeout.get_op_names"),
    run_timeout=get_setting(config, "timeout.process_image"),
    channel_options=[
        ("grpc.max_receive_message_length", _mb),
        ("grpc.max_send_message_length", _mb),
    ],
)

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
    origin = snap.get("origin") or "agent"
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
    origin = snap.get("origin") or "agent"
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


_TITLE = "# biopb-mcp session — audit export\n"

_INTRO = (
    "Exported {ts} · {n} job(s).\n\n"
    "This notebook is an **audit record** of an `execute_code` session. The "
    "first code cell rebuilds the namespace (`np`, `da`, `client`, `ops`, and an "
    "empty `viewer`) on a best-effort basis; each cell below is one job, with its "
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


def _now_epoch():
    """Current epoch seconds (own helper so tests can monkeypatch the stamp)."""
    return datetime.datetime.now().timestamp()


def suggested_filename():
    """``biopb-mcp-session-YYYYMMDD-HHMMSS.ipynb`` for the download."""
    stamp = datetime.datetime.fromtimestamp(_now_epoch()).strftime("%Y%m%d-%H%M%S")
    return f"biopb-mcp-session-{stamp}.ipynb"
