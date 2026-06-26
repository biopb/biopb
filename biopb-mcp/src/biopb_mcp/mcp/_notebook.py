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

from biopb_mcp._config import load_config, get_setting
from biopb_mcp._connection import TensorConnection
from biopb_mcp.mcp._process_ops import build_ops

config = load_config()
_conn = TensorConnection(config)
_conn.auto_connect()          # synchronous best-effort connect (audit; no async service)
client = _conn.client

_mb = get_setting(config, "grpc.max_message_size_mb") * 1024 * 1024
ops = build_ops(
    client_getter=lambda: _conn.client,
    server_urls=get_setting(config, "mcp.services.process_image_servers"),
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


def _job_cell(snap):
    job_id = snap.get("job_id", "?")
    status = snap.get("status", "?")
    elapsed = snap.get("elapsed", "?")
    header = f"# [{job_id} · {status} · {elapsed}s · {_fmt_ts(snap.get('created'))}]\n"
    source = header + (snap.get("code") or "")
    return _code_cell(
        source,
        outputs=_job_outputs(snap),
        metadata={
            "biopb": {
                "job_id": job_id,
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
    "recorded output.\n\n"
    "**Runnability caveats.** External state is not captured: tensor-server "
    "source ids and napari viewer layers from the live session do not exist on a "
    "fresh kernel, so any cell that chains `ops` source ids or reads `viewer` "
    "layers needs the same live server (or edits). In-namespace Python variables "
    "*do* carry across cells. Cells whose header reads `cancelled` / `interrupted` "
    "/ `error` are kept verbatim — re-running one may re-trigger the same hang or "
    "failure, so skip or edit it. Only the most recent jobs are retained, so a "
    "long session may be missing its start. `auto_connect()` persists the server "
    "URL to your config; under a headless `nbconvert --execute` the `viewer` "
    "becomes `None` and viewer cells fail."
)


def build_notebook(jobs, *, headless=False):
    """Build an nbformat-v4 notebook dict from a list of job snapshots.

    *jobs* is the oldest-first list returned by ``_jobs.export()`` (each a
    ``_Job.snapshot()`` dict). *headless* tweaks the intro wording. The result is
    a plain dict ready to ``json.dumps`` into a ``.ipynb`` file.
    """
    jobs = jobs or []
    intro = _INTRO.format(ts=_fmt_ts(_now_epoch()), n=len(jobs))
    if headless:
        intro += "\n\n_Exported from a headless (no-display) session._"

    cells = [_markdown_cell(_TITLE + "\n" + intro), _code_cell(BOOTSTRAP_SRC)]
    if jobs:
        cells.extend(_job_cell(s) for s in jobs)
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
