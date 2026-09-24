"""FastMCP tool and resource surface, over a child Jupyter kernel.

The server runs in the foreground (uvicorn, streamable-http on
127.0.0.1:<port>/mcp) and drives the :class:`KernelHost` that ``_app`` owns.
Every tool call is a round-trip into that kernel, where the napari viewer,
dask, and the TensorFlightClient live.  The kernel can be interrupted or
hard-restarted independently of this process.

What is *not* here, and why: the app object and the launcher-set state are in
``_app``, the kernel round trip in ``_kernel_rpc``, and the one-agent claim and
foreign-activity digest in ``_writers``.  Each had two or three consumers
outside this module -- the observe page, the chat loop, the shared HTTP guard --
which had to reach in through a dozen private names to get at them.  What is
left is the agent-facing surface: the snippets its tools run, the job
submit/await client they share, and the tools and resources themselves.

**Every tool is ``async def``, and no tool blocks.** The SDK invokes a
synchronous tool function directly on the event loop, and this process serves
``/mcp``, the observe page and the chat turn on that one loop -- so a tool that
waits on the kernel there does not make its caller wait, it makes every caller
wait, for as long as the round trip takes (``execute_code`` used to hold it for
the whole ``promote_after`` window). Kernel round trips therefore go to a thread
(``_kernel_rpc._execute``), and the promote window is a loop
sleep. Adding a tool means adding an async one.
"""

import asyncio
import functools
import logging
import os
import threading
import time
from typing import Annotated

from mcp.types import ImageContent, TextContent
from pydantic import AnyUrl, Field

from . import (
    _app,
    _docs,
    _kernel_rpc,
    _scratch,
    _workflow_doc,
    _writers,
)
from ._app import mcp

logger = logging.getLogger(__name__)

_SCREENSHOT_SNIPPET = (
    "import base64 as _b64, cv2 as _cv2\n"
    "if not _viewer_window_alive():\n"
    "    print('" + _kernel_rpc._WINDOW_CLOSED_DELIM + "')\n"
    "else:\n"
    # Under async slicing, force-sync the current view so the capture reflects
    # the state the agent just set, not a pre-load frame. No-op when async is
    # off or the bootstrap predates the helper (defensive globals().get).
    "    globals().get('_resync_view', lambda: None)()\n"
    "    _arr = viewer.screenshot(canvas_only={canvas_only})\n"
    "    _bgra = _cv2.cvtColor(_arr, _cv2.COLOR_RGBA2BGRA)\n"
    "    _ok, _buf = _cv2.imencode('.png', _bgra)\n"
    "    print('"
    + _kernel_rpc._PNG_DELIM
    + "' + _b64.b64encode(_buf.tobytes()).decode())\n"
)

# Self-contained inspection snippet.  Built by string concatenation (no
# f-strings/format) so the object path is the only injected value.
_INSPECT_TEMPLATE = """
import inspect as _inspect
__path = __PATH__
try:
    __obj = eval(__path)
except Exception as __exc:
    print("Error resolving " + repr(__path) + ": " + str(__exc))
else:
    __lines = [
        "Type: " + type(__obj).__name__,
        "Docstring: " + (_inspect.getdoc(__obj) or "No documentation."),
        "",
        "Attributes:",
    ]
    for __name in sorted(dir(__obj)):
        if __name.startswith("_"):
            continue
        try:
            __attr = getattr(__obj, __name)
        except Exception:
            continue
        if _inspect.ismethod(__attr) or _inspect.isfunction(__attr):
            try:
                __sig = str(_inspect.signature(__attr))
                __short = (_inspect.getdoc(__attr) or "").split(chr(10))[0]
                __lines.append("  ." + __name + __sig + "  -- " + __short)
            except (ValueError, TypeError):
                __lines.append("  ." + __name + "(...)")
        else:
            __lines.append("  ." + __name + " [" + type(__attr).__name__ + "]")
    print(chr(10).join(__lines))
"""

_STATUS_SNIPPET = """
# This kernel's interpreter -- the one a skill's `pkg:` requirement is about, and
# not necessarily the server process's env (the kernelspec need not be it). The
# common such token is `pkg:biopb-mcp>=X`, how a skill says it needs a release
# that carries some plugin, so report that one instead of making the agent import.
# The interpreter, and how to install into it, come from _requires (which decides
# the command from the env's shape) rather than being composed here.
print("## Versions")
try:
    from biopb_mcp.mcp import _requires as _req
    for _line in _req.versions_status_lines():
        print(_line)
except Exception as _e:
    print("  error: " + str(_e))

print("")
print("## Dask")
# Where a .compute() runs: dask's own default, in-process unless a cell built a
# dask Client. Read without importing distributed, which only a cell that
# built one has loaded.
try:
    import sys as _sys
    _dc = None
    if "distributed" in _sys.modules:
        from distributed import Client as _Client
        try:
            _dc = _Client.current(allow_global=True)
        except ValueError:
            _dc = None
    if _dc is None:
        print("  mode: in-process (dask's default), shared with the viewer")
    else:
        try:
            _info = _dc.scheduler_info(n_workers=-1)  # the default caps at 5
        except TypeError:  # distributed without the argument
            _info = _dc.scheduler_info()
        _nw = len(_info.get("workers", {}))
        print("  mode: dask Client at " + str(_dc.scheduler.address))
        print("  workers: " + str(_nw))
        print("  dashboard: " + str(_dc.dashboard_link))
        if _nw == 0:
            print("  WARNING: no workers left -- a .compute() would block forever; "
                  "close this dask Client or build a new one")
except Exception as _e:
    print("  error: " + str(_e))

print("")
print("## Tensor Server")
_tc = _conn.client
if _tc is not None:
    try:
        print("  connected: true")
        print("  health: " + str(_tc.health_check()))
        print("  sources_cached: " + str(len(_conn.sources or {})))
    except Exception as _e:
        print("  connected: true")
        print("  health_error: " + str(_e))
elif getattr(_conn, "last_status", "") == "starting":
    print("  connected: false")
    print("  state: starting — " + str(getattr(_conn, "last_message", "")))
else:
    print("  connected: false")
    _lm = str(getattr(_conn, "last_message", ""))
    if _lm:
        # issue #86: surface the reason (auth required / unreachable) instead of
        # a bare "connected: false" the agent can't act on.
        print("  error: " + _lm)

print("")
print("## Viewer")
import os as _os
import sys as _sys
if _sys.platform == "darwin" or _os.name == "nt":
    # Mirrors _has_display(): the native window server is ambient, so $DISPLAY
    # (XQuartz, VcXsrv) says nothing about where Qt actually renders.
    print("  display: (host window server)")
elif _os.environ.get("BIOPB_VIRTUAL_DISPLAY"):
    # Launcher-owned Xvfb (#90). A silent degradation: every tool below still
    # works, so the agent relaying it is the only thing that reaches the user
    # (#892). Kept as loud as start_kernel's — a session can reach here without
    # having seen that message (context cleared, kernel already up).
    print("  display: VIRTUAL (Xvfb " + str(_os.environ.get("DISPLAY", "?")) + ")")
    print("    The user sees NO napari window, and software GL renders 3-D")
    print("    volumes ~13x slower than a real GPU. Show results through the")
    print("    web viewer instead (## Web viewer above) — it needs no display")
    print("    here and is what the user can actually look at.")
    print("    Say so once, and ask: usually the host does have a display and")
    print("    the MCP client dropped $DISPLAY on the way in (Codex CLI does),")
    print("    in which case a restart with it set gives them a real window.")
else:
    print("  display: " + str(
        _os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY") or "?"
    ))
if not _viewer_window_alive():
    print("  window: CLOSED — the napari window was closed; layer mutations")
    print("    won't display. Data/compute still work; restart_kernel to restore,")
    print("    or show results through the web viewer, which needs no window.")
    print("  layers: " + str(len(viewer.layers)) + " (model only, not shown)")
else:
    print("  window: open")
    print("  layers: " + str(len(viewer.layers)))
    for _layer in list(viewer.layers)[:10]:
        _shape = getattr(_layer.data, "shape", "?")
        print("    - " + str(_layer.name) + " (" + str(_shape) + ")")

print("")
print("## Ops")
_ops = globals().get("ops")
if _ops:
    print("  " + ", ".join(sorted(_ops)))
else:
    print("  (none configured -- services.process_image_servers -- or unreachable)")

print("")
# What the plugin loader actually loaded, which neither the kernel dir (fail-open:
# a file that raised is on disk and not loaded) nor dir() (a file contributes its
# function names, not its own name) can tell the agent. It reads this to resolve a
# skill's `plugin:<name>` requirement.
print("## Kernel plugins")
try:
    from biopb_mcp.mcp import _requires as _req

    for _line in _req.plugin_status_lines():
        print(_line)
except Exception as _e:
    print("  error: " + str(_e))
"""


# Whether psutil's CPU counter has a previous reading to measure against.
_cpu_primed = False


def _cpu_percent(psutil):
    """System CPU usage, without a 100 ms sleep in every ``server_status``.

    ``interval=0.1`` blocks for a tenth of a second by definition, and this tool
    is called at session start and repeatedly through an agent run. The
    non-blocking form reports usage since the *previous* call, which for a tool
    called repeatedly is the more meaningful number anyway -- it just needs one
    reading to subtract from, so the first call still pays the sample.
    """
    global _cpu_primed
    if _cpu_primed:
        return psutil.cpu_percent(interval=None)
    _cpu_primed = True
    return psutil.cpu_percent(interval=0.1)


async def _start_job(host, code, **kwargs):
    """Submit a job and resolve everything that can happen before it runs.

    *code* and *kwargs* are :func:`_submit_job`'s. The client identity is
    added here rather than passed in, because
    claiming the kernel is this function's business:
    every submitting tool answers "am I the holder?" and "is something already
    running?" the same way, and a second answer to either is a second policy.

    Returns ``(job_id, foreign_note, window_alive, message)``. Exactly one of
    *job_id* and *message* is not None — *message* is the finished tool reply
    for every outcome that never started a job, and *foreign_note* is what the
    caller appends to whatever it returns instead. *window_alive* is always
    None (see :func:`_submit_job`).
    """
    writer, _label = _writers._client_identity()

    # Read once at entry, append to whichever path returns below. The submit
    # goes to a thread for the reason on _kernel_rpc._execute: blocking here
    # blocks the whole process, not this one call.
    digest = _writers._foreign_digest(host)
    foreign_note = _writers._render_foreign_note(digest)
    if foreign_note:
        _writers._ack_foreign_digest(host, digest, writer)

    job_id, message, drop_note, window_alive = await asyncio.to_thread(
        _submit_job, host, code, digest, _tool_busy_message, **kwargs
    )
    if drop_note:
        foreign_note = ""
    if message is not None:
        return None, foreign_note, window_alive, message + foreign_note
    return job_id, foreign_note, window_alive, None


def _tool_busy_message(running, running_origin) -> str:
    """An MCP tool's reply when the kernel already has a job running.

    Whose job is running decides the advice. Telling the agent to "stop it with
    interrupt_kernel" while *someone else* is running a cell would have it kill
    their work; interrupt_kernel refuses that anyway, so the wording must not
    send it there.
    """
    if running_origin and running_origin != "mcp":
        who = "The user" if running_origin == "user" else "Another writer"
        return (
            f"{who} is running a cell ({running}) in this kernel. Only one "
            f"job runs at a time — wait for it and poll_job('{running}'); do "
            "not interrupt it."
        )
    return (
        f"A job ({running}) is already running. Poll it with "
        f"poll_job('{running}'), or stop it with interrupt_kernel / "
        "restart_kernel before starting another."
    )


# Serializes _submit_job's check and send: two submits must not both find the
# kernel free and queue a cell each.
_submit_lock = threading.Lock()


def _submit_job(host, code, digest, busy_message, intent=""):
    """Claim the kernel, send *code* as a cell, and classify the outcome.

    The claim protocol is the same for every writer of this namespace, and it
    is a security-relevant one, so it is written once here. Both the MCP tools
    (via :func:`_start_job`) and the in-process chat loop go through it.

    The origin is added here for the same reason the identity is: it is the
    caller's point of view, every seam that reads a job back compares against
    it, and a surface that let it default recorded its cells as the MCP
    client's (biopb/biopb#880). One contextvar, read at one seam.

    **One job at a time.** A cell is refused while anything runs -- a cell
    (anyone's) or a task -- rather than queued behind it: an agent that queued
    would lose track of what ran when. What legitimately differs between the
    surfaces is only how that refusal reads: a tool caller gets a job handle it
    can poll, the chat loop never does, so it must not be told to poll one.
    Hence *busy_message*, a ``(running_job_id, running_origin) -> str``.

    Returns ``(job_id, message, drop_note, window_alive)``; exactly one of
    *job_id* and *message* is not None. *drop_note* asks the caller to suppress
    its foreign-activity note: a running foreign job stays in the digest by
    design, so the note would report the very job the refusal already reports.
    *window_alive* is always None here: it is known only once the cell has run
    (``host.jobs.window_alive``).
    """
    writer, writer_label = _writers._client_identity()
    holder = _writers.take_claim(host, writer, writer_label)
    if holder is not None:
        held_by = f" ({holder})" if holder else ""
        return None, _writers._NOT_OWNER_MSG.format(held_by=held_by), False, None
    with _submit_lock:
        running = host.jobs.running()
        if running is not None:
            drop_note = [d.get("job_id") for d in digest] == [running["job_id"]]
            message = busy_message(running["job_id"], running.get("origin"))
            return None, message, drop_note, None
        job_id = host.jobs.new_id()
        try:
            host.run_cell(
                code, job_id, origin=_writers._local_origin.get(), intent=intent
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return None, str(exc), False, None
    return job_id, None, False, None


def _poll_submitted(host, job_id):
    """*job_id*'s snapshot, for a job this caller has just submitted.

    Its start travels on iopub and the submit's reply on the shell socket, so
    the reply can arrive first: until the record appears, the job is running as
    far as the caller knows, not unknown.
    """
    snap = host.jobs.poll(job_id)
    if snap.get("status") == "unknown":
        return {"job_id": job_id, "status": "running", "stdout": ""}
    return snap


async def _await_job(host, job_id, budget=None, submitted=False):
    """*job_id*'s snapshot once it is terminal, or when *budget* seconds run out.

    *budget* defaults to the promote window, which is what a submitting tool
    waits; ``poll_job`` passes its own. A ``status`` of ``running`` means the
    budget expired and the caller hands back a job handle rather than a result.
    *submitted* is the submitting tool's: see :func:`_poll_submitted`.

    Each look is a read of the host's records (``host.jobs``), and the wait a
    loop sleep, so waiting costs the kernel nothing and leaves this process's
    loop free for every other caller.
    """
    look = _poll_submitted if submitted else (lambda h, j: h.jobs.poll(j))
    deadline = time.monotonic() + (_app._promote_after if budget is None else budget)
    snap = look(host, job_id)
    while snap.get("status") == "running" and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
        snap = look(host, job_id)
    return snap


def _format_verification(record: dict, job_id: str, saved_path=None) -> str:
    """The verification report: the verdict, then the per-cell ledger.

    Written for an agent about to decide between "save it" and "fix cell 4",
    so the failing cell's own traceback is quoted in full and the cells after it
    are named as skipped rather than silently absent — otherwise a cascade reads
    as a workflow that mysteriously got shorter.
    """
    cells = record.get("cells") or []
    status = record.get("status")
    lines = []

    if status == "ok":
        lines.append(
            f"Verified: all {len(cells)} cell(s) ran in a scratch kernel — a "
            "fresh, headless process with none of this session's state — which "
            "has since been discarded."
        )
    else:
        failed = next(
            (i for i, c in enumerate(cells) if c.get("status") == "error"), None
        )
        where = f"cell {failed + 1}" if failed is not None else "the run"
        lines.append(
            f"NOT verified — {where} failed, so the cells after it were "
            "skipped rather than run against state it never produced."
        )

    for i, cell in enumerate(cells, 1):
        # The head, not the output: a verification's record is polled, and the
        # full text of every cell belongs to the notebook, not to a ledger line
        # (see _scratch._record).
        head = (cell.get("stdout_head") or "").strip()
        lines.append(
            f"  {i}. {cell.get('status')} · {cell.get('elapsed')}s"
            + (f" · {head}" if head else "")
        )

    for i, cell in enumerate(cells, 1):
        if cell.get("status") == "error":
            lines.append(f"\nCell {i}:\n{cell.get('code', '')}")
            lines.append(f"\n{cell.get('error_text', '')}")
            break

    if status == "ok":
        # Already on disk: every run that passes is spooled, because this
        # process cannot know which passing run has the right *numbers* -- only
        # that every cell ran. So the agent's job is to say where it went, not
        # to write it.
        lines.append(
            f"\nSaved as a notebook: {saved_path}"
            if saved_path
            else "\nThe notebook could not be written to disk (see the server "
            "log); the user can still download it from the observe page."
        )
        lines.append(
            "The user can download a copy where they want it from the observe "
            "page (Verification -> Download). Tell them where it is; do not "
            "write the file yourself."
        )
    else:
        lines.append(
            f"\nFix the cell and call verify_workflow again. Full record: "
            f"poll_job('{job_id}')."
        )
    return "\n".join(lines)


def _viewer_base_url() -> str:
    """The control's origin, which is where the web viewer is served.

    Resolved per call because the port is configurable and a control that
    restarts elsewhere republishes it. A control published below the root
    (``--url-prefix``) still answers here; what carries the prefix is the URL
    the *user's* browser reaches it by, which nothing in this process can know.
    """
    try:
        from biopb._endpoints import control_base_url

        return control_base_url()
    except Exception:  # pragma: no cover - core SDK always present in practice
        logger.debug("status: control base url unresolvable", exc_info=True)
        return "http://127.0.0.1:8813"


def _format_job_status(snap: dict) -> str:
    """Render a job snapshot (poll_job output).

    A verification job renders as its report once it is terminal, because that
    is what its caller was told to poll for: ``verify_workflow`` hands back this
    job id when the run outlives the promote window, and a per-cell ledger
    flattened into one blob of output would not answer the question it was
    handed back for.
    """
    job_id = snap.get("job_id", "?")
    status = snap.get("status")
    header = f"{job_id}: {status} ({snap.get('elapsed', '?')}s)"
    record = snap.get("verify")
    if record and status != "running":
        return (
            header + "\n" + _format_verification(record, job_id, snap.get("saved_path"))
        )
    body = _kernel_rpc._format_execute_result(snap)
    if status == "running":
        return header + "\nPartial output:\n" + (body or "(none yet)")
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("docs://index")
def get_index() -> str:
    """The doc index, for hosts that subscribe to resources.

    The same text the handshake carries and ``read_doc("index")`` returns. It is
    a resource as well so a host that honours ``resources/updated`` re-reads it
    after a write -- the nearest thing MCP has to a harness injecting recall.
    Nothing depends on it; the tool is the contract.
    """
    return _docs.render_index()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def read_doc(id: str) -> str:  # noqa: A002 - the parameter name is the wire contract
    """Read one doc from the biopb knowledge store.

    `id` is what the index lists. `read_doc("index")` returns the index itself
    -- the catalog of everything readable, with the agent's own hooks -- and is
    worth re-reading when the session's handshake copy has gone stale.

    A doc links another as `[[other-id]]`; follow one with another `read_doc`.
    """
    return await asyncio.to_thread(_docs.read_doc, id)


@mcp.tool()
async def write_doc(
    id: str,  # noqa: A002 - the parameter name is the wire contract
    body: str | None = None,
    old: str | None = None,
    new: str | None = None,
) -> str:
    """Write a doc into the biopb knowledge store. Returns a diff of the change.

    Two forms. `body` writes the whole doc. `old`/`new` replaces one exact
    occurrence of `old` -- the cheaper form, and the only sane one for the
    index, which is long enough that rewriting it loses lines. The call is
    refused if `old` is absent or matches more than once, so include enough
    surrounding text to name one place. Several edits are several calls.

    Writing a shipped doc creates your own copy shadowing it; the shipped file
    is never touched, so an upgrade can still replace it. There is no delete:
    retire a doc of your own by removing its index entry, and a shipped one by
    adding its id to the index's `ignored:` line.

    A new doc is filed in the index automatically, under `## Unfiled` -- move
    the line where it belongs next time you edit the index.

    Five rules, and they are what keep this store worth reading:

    - Write only a **validated, multi-step** procedure -- one the user has just
      confirmed on real data. A doc is a claim that the procedure works.
    - Never a **dataset-specific** one. A source_id, an array_id or a pathname
      makes it unusable by the next session; that run belongs in a notebook.
    - Phrase the index hook **as the user's request**, not as an implementation
      summary. It is what a later session matches against.
    - **Update an existing doc rather than write a near-duplicate.** Read the
      index first, and prefer an `old`/`new` edit to a new file.
    - **Verify that a name, flag or call the doc quotes still exists** before
      relying on it. Nothing else checks a doc of yours.

    `read_doc("authoring")` has what a procedure doc must contain.
    """
    result = await asyncio.to_thread(_docs.write_doc, id, body, old, new)
    await _notify_doc_written(id)
    return result


async def _notify_doc_written(doc_id: str) -> None:
    """Tell a subscribing host the index resource moved. Best-effort.

    Only the index is exposed as a resource, and only some hosts subscribe, so
    a failure here is not a failed write.
    """
    if doc_id != _docs.INDEX_ID:
        return
    try:
        session = mcp.get_context().session
        await session.send_resource_updated(AnyUrl("docs://index"))
    except Exception:
        logger.debug("docs: could not notify the index resource update", exc_info=True)


def _own_cell_holds_main(host, what):
    """The refusal for a tool that needs the kernel's main thread while the
    caller's own cell holds it, or None.

    Answered at once rather than queued: the tool would only wait for the cell
    to end, and asking while it runs is a mistake in the plan -- the agent
    started that cell -- not a busy kernel. A user's cell is not the agent's to
    foresee, so behind one the tool still waits.
    """
    cell = host.jobs.running_cell()
    if cell is None or cell.get("origin") != _writers._local_origin.get():
        return None
    job_id = cell["job_id"]
    return (
        f"Refused: your cell {job_id} is running on the kernel's main thread, "
        f"and {what} needs that thread, so it could only wait for the cell to "
        "end. Asking now is a mistake in the plan, not a busy kernel: run a "
        "compute you want to watch with run_async(fn), which leaves the main "
        f"thread free. Poll the cell with poll_job('{job_id}')."
    )


@mcp.tool()
async def take_screenshot(canvas_only: bool = True) -> list:
    """Capture the napari viewer as a PNG image.

    Args:
        canvas_only: If True, capture only the canvas area. If False,
            capture the entire viewer window.

    Returns a PNG screenshot as an image content block.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return [TextContent(type="text", text=err)]
    refusal = _own_cell_holds_main(host, "a screenshot")
    if refusal is not None:
        return [TextContent(type="text", text=refusal)]

    snippet = _SCREENSHOT_SNIPPET.format(canvas_only=bool(canvas_only))
    res = await _kernel_rpc._execute(host, snippet)
    if (
        _kernel_rpc._extract_delimited(
            res.get("stdout", ""), _kernel_rpc._WINDOW_CLOSED_DELIM
        )
        is not None
    ):
        return [
            TextContent(
                type="text",
                text=(
                    "No screenshot: the napari viewer window was closed. Data "
                    "access and compute via execute_code still work; call "
                    "restart_kernel to restore a viewer window."
                ),
            )
        ]
    data = _kernel_rpc._extract_delimited(res.get("stdout", ""), _kernel_rpc._PNG_DELIM)
    if data is None:
        detail = res.get("error_text") or res.get("stdout") or res.get("status")
        return [TextContent(type="text", text=f"Screenshot failed: {detail}")]
    return [ImageContent(type="image", mimeType="image/png", data=data)]


#: ``intent``'s guidance, on the parameter rather than only in the prose above
#: it: a function-calling model reads the schema per argument, and an
#: undocumented optional string is one nothing asks it to fill in.
_INTENT_DESC = (
    "One short sentence on *why* you are running this cell -- the goal you are "
    "pursuing for the user, not a restatement of what the code does. Recorded "
    "with the job and written into the session's notebook export, which is "
    "otherwise a log of code with no record of what anyone was trying to "
    "achieve. Leave it empty rather than padding it."
)

#: The paragraph of :func:`execute_code`'s description that is true only over
#: the wire. The in-process chat loop waits for the cell instead of promoting it,
#: so it substitutes its own (``_chat._CHAT_RUN_PARAGRAPH``); named here, and
#: pinned by a test, so a reworded docstring fails loudly rather than quietly
#: leaving the loop's model told to poll for a handle it will never be given.
PROMOTE_PARAGRAPH = """Code runs as a cell on the kernel's main thread, like a notebook cell.
    If it finishes quickly the result is returned inline; otherwise this returns
    a job handle (job-N) and the cell keeps running. Poll it with poll_job, and
    stop it with interrupt_kernel or restart_kernel (guaranteed). While it runs
    it holds the main thread: the viewer does not repaint and take_screenshot /
    inspect_object refuse. For a long compute you want to watch, end the cell
    with run_async(fn) instead -- fn runs on a worker thread, the cell returns
    at once with a task id (task-...) to poll, and the viewer stays live. Only
    one job runs at a time."""


@mcp.tool()
async def execute_code(
    python_code: str,
    intent: Annotated[str, Field(description=_INTENT_DESC)] = "",
) -> str:
    """Execute Python code in the napari kernel.

    The kernel is a full Jupyter/IPython kernel (imports allowed) with the
    namespace: viewer (with add_tensor/tensor methods), client(image data access), and ops (a
    dict of image processing operations). np and da are also imported. Variables persist
    across calls until the kernel is restarted.

    Code runs as a cell on the kernel's main thread, like a notebook cell.
    If it finishes quickly the result is returned inline; otherwise this returns
    a job handle (job-N) and the cell keeps running. Poll it with poll_job, and
    stop it with interrupt_kernel or restart_kernel (guaranteed). While it runs
    it holds the main thread: the viewer does not repaint and take_screenshot /
    inspect_object refuse. For a long compute you want to watch, end the cell
    with run_async(fn) instead -- fn runs on a worker thread, the cell returns
    at once with a task id (task-...) to poll, and the viewer stays live. Only
    one job runs at a time.

    Only one *agent* runs code in a kernel, too: whoever calls this first holds
    it until the kernel restarts. A second client is refused here and by every
    other tool that changes kernel state (interrupt_kernel, restart_kernel), and
    keeps only the read-only ones. The person at the machine is exempt — they
    can run cells from a Jupyter notebook attached to this kernel while you
    work, which is what the user-activity notice on these results is telling
    you about. A cell of theirs sent while yours runs waits for it; one sent
    while a run_async task runs runs beside it.

    Results include print() output and the last expression's repr. Rich IPython
    display() output is not captured; use print().

    * viewer mutations (read_doc("napari-viewer") has more):
    Mutate the viewer directly. From a run_async task too: its mutations are
    marshaled to the main thread. run_on_main(fn) batches many mutations into
    one hop, or touches raw Qt (viewer.window), from a task.

    * data access (read_doc("tensor-server-client") has more):
    - client.query_sources(sql, format="pandas") runs server-side DuckDB and
      returns a DataFrame. The `sources` table columns are: source_id,
      source_url, source_type, indexed_at, metadata_json, is_resolved, and
      `tensors`, a LIST of STRUCT(array_id, dim_labels, shape, dtype) with one
      entry per tensor (note source_url, not "url"). This is the browse
      surface; there is no other. Structure is a per-tensor question, so ask it
      of `tensors`: `WHERE len(list_filter(tensors, t -> t.dtype='uint16')) > 0`,
      or `tensors[1].dtype` for the source's first tensor (DuckDB lists are
      1-indexed, unlike `tensors[0]` in Python/TS code). An unresolved (cloud)
      source has an empty `tensors`, so any such predicate hides it; use
      `is_resolved` to filter on them on purpose (e.g. `WHERE NOT is_resolved`
      to list what hasn't been resolved yet).
    - resolved is not the same as local. Assume a cloud or synced-folder
      source's bytes may not be on the serving machine, so its first read can
      be slow or fail offline -- say so before starting one, not after.
    - viewer.add_tensor(array_id) loads a tensor as a layer (auto-handles the
      multiscale pyramid); client.get_tensor(array_id) returns a lazy dask
      array without adding a layer. Both take the same id: "source_id/t1"
      within a multi-tensor source, a bare "source_id" for a single-tensor one.
    - reading pixels back off a layer is not plain napari: layer.data is
      napari's MultiScaleData sequence of pyramid levels when layer.multiscale,
      in display axis order ([..., Z, Y, X], at the source's own rank), and
      lazy -- np.asarray() of it silently gives the *lowest* level. Use
      viewer.tensor(layer), which returns a plain full-resolution dask array
      from any layer, and read_doc("napari-viewer") before measuring or computing from
      a layer.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err

    verifying = _scratch.running()
    if verifying is not None:
        # One job at a time is a rule about the *session*, not about one kernel:
        # a verification is running in a second one, on the same dask cluster.
        return _tool_busy_message(verifying["job_id"], "mcp")

    job_id, foreign_note, _w, msg = await _start_job(host, python_code, intent=intent)
    if msg is not None:
        return msg

    snap = await _await_job(host, job_id, submitted=True)
    if snap.get("status") != "running":
        return (
            _kernel_rpc._format_execute_result(snap)
            + _kernel_rpc._window_note(host.jobs.window_alive(job_id))
            + foreign_note
        )

    partial = snap.get("stdout", "") if snap else ""
    return (
        f"Job {job_id} is still running after {_app._promote_after:.0f}s, "
        "holding the main thread: take_screenshot and inspect_object refuse "
        f"until it ends. Poll it with poll_job('{job_id}'); stop it with "
        "interrupt_kernel or restart_kernel. Next time, run a compute this long "
        "with run_async(fn) to keep the viewer live.\n"
        "Partial output:\n" + (partial or "(none yet)") + foreign_note
    )


@mcp.tool()
async def verify_workflow(document: str, title: str = "") -> str:
    """Check that a workflow notebook runs on its own, in a scratch kernel.

    Use this when the user wants a workflow they have just proven kept as a
    document. **You write the whole document** — prose and code — and this runs
    it; on success the notebook is saved for them and this tool reports where.
    Tell them the path; do not write the file yourself.

    **The format** is markdown with fenced ``python`` cells. Each fence is one
    notebook cell, run in order; everything between them is markdown, rendered
    as written. A fence in another language (```bash) is prose about a command,
    not a cell. The first ``# `` heading becomes the title. A saved ``.ipynb``
    is also accepted verbatim, which is what to send back when the user has
    edited one.

    Write it as a document, not as a list of cells with comments: a heading, a
    sentence on what each step does and why in the terms the user would use
    ("the threshold is 0.4 because the background peak sits at 0.3"), and the
    code between them. That prose is most of what makes the notebook usable
    months later, and you are the only one who knows it.

    **It starts with its own setup cell.** The scratch kernel is given nothing —
    no ``client``, no ``ops``, no ``np``, no plugins, no ``viewer`` — because
    the reader's kernel will have nothing either. So the first cell is the
    document's own environment:

        import numpy as np
        from biopb_mcp.workflow_env import workflow_env

        conn, ops = workflow_env()
        client = conn.client

    A workflow that skips it fails with ``NameError``, which is the verdict: it
    would have failed the same way for whoever opened the notebook.

    **Rewrite the session, do not select from it.** The program that works is
    almost never a subsequence of what was run: a cell that created a variable
    and a later cell that corrected its value have to merge into one, and dead
    ends, retries and debugging prints drop out. Read the session with poll_job
    and write the document you *mean*.

    **There is no `viewer`, and that is deliberate.** The viewer exists so you
    can show something to the person you are working with, and nobody is
    watching a verification. Write the workflow to *compute* and to ``print``
    what matters; leave displaying the result to the live session.

    **Failing is normal; the draft is where you fix it.** Every attempt, pass or
    fail, is written to a draft file in the document's own format, and this tool
    reports its path. On a failure, **read that file, edit it, and send the
    whole thing back** — do not retype the document from memory, because the
    user may have edited it themselves in the meantime, and their edit is the
    one that should survive. A pass promotes the draft to the saved notebook and
    removes it.

    **The live session is untouched** — its variables, its layers, its viewer.
    Bringing the scratch kernel up takes a few seconds, and while a verification
    runs the session kernel accepts no cells (one job at a time, across both).

    **What it proves:** every cell runs, in order, on a bare kernel with
    biopb-mcp installed. That covers the whole class of defect that makes a
    transcript unrunnable — a cell reading a variable an earlier discarded cell
    created, a handle only this session had.

    **What it does not.** The numbers are right: check them. And a scratch
    *process* is not a scratch *world* — it talks to the same tensor server and
    the same filesystem, so `client.upload_array` / `add_source`,
    and any cell that writes a file, write through for real. Verify a workflow
    three times and you have three uploaded arrays. **Say so before running one
    that writes.**

    Args:
        document: the workflow, as markdown with fenced ``python`` cells (or a
            complete ``.ipynb`` document).
        title: overrides the document's own ``# `` heading. Names the saved
            file; leave it out unless the heading is wrong.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    try:
        blocks = _workflow_doc.parse(document)
    except _workflow_doc.DocumentError as exc:
        # The agent's mistake to correct, and it has not cost anything yet: no
        # kernel was spawned and nothing was written.
        return f"{exc} Send markdown with ```python cells, or a saved .ipynb."
    title = title.strip() or _workflow_doc.title_of(blocks)

    # The verification is this client's: a stranger is refused an interrupt on
    # it (_scratch.interrupt), as on the session kernel.
    writer, _label = _writers._client_identity()
    foreign_note = _writers._foreign_activity_note(host)
    started = await asyncio.to_thread(
        _scratch.start,
        blocks,
        title,
        host,
        writer,
        _writers._local_origin.get(),
    )
    if started.get("error") == "busy":
        return (
            _tool_busy_message(
                started.get("running_job_id"), started.get("running_job_origin")
            )
            + foreign_note
        )
    if "job_id" not in started:
        return str(started.get("error")) + foreign_note

    job_id = started["job_id"]
    snap = await _await_verification(job_id)
    if snap.get("status") == "running":
        return (
            f"Verification {job_id} is still running after "
            f"{_app._promote_after:.0f}s (a scratch kernel takes a few seconds "
            f"to build). Poll it with poll_job('{job_id}') — the per-cell record "
            "is in the result." + foreign_note
        )
    record = snap.get("verify")
    if not record:
        # Never reached a cell: the scratch kernel failed to build, or died
        # before the first one. Its death IS the verdict, so report why.
        return (
            f"Verification {job_id} did not run: "
            + (snap.get("error_text") or "the scratch kernel produced no record.")
            + _draft_note(snap)
            + foreign_note
        )
    return (
        _format_verification(record, job_id, snap.get("saved_path"))
        + _draft_note(snap)
        + foreign_note
    )


def _draft_note(snap):
    """Where the document that just failed is, so the next attempt edits it.

    Only on a failure: a pass promotes the draft and the report already names
    the notebook it became, so pointing at a file that is gone would be worse
    than saying nothing.
    """
    draft = snap.get("draft_path")
    if not draft:
        return ""
    return (
        f"\n\nThe document is at {draft} — read it, edit it, and send the whole "
        "file back rather than retyping it: the user may have edited it too."
    )


async def _await_verification(job_id, budget=None):
    """Poll a verification until it is terminal or *budget* seconds run out.

    ``_await_job``'s counterpart for the scratch kernel. Separate because there
    is no kernel round trip to make: the run is a thread in this process, so a
    poll is a dict read and the wait is pure loop sleep.
    """
    deadline = time.monotonic() + (_app._promote_after if budget is None else budget)
    snap = _scratch.poll(job_id)
    while snap is not None and snap.get("status") == "running":
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.4)
        snap = _scratch.poll(job_id)
    return snap or {"status": "unknown"}


#: How long ``poll_job`` waits for a running job when the caller says nothing,
#: and the most it will wait when the caller asks for more.
#:
#: Defaulted rather than opt-in because the caller this exists for will not pass
#: it: an agent loop has no innate sense of time, and one that would think to
#: raise `wait` is not the one spinning. A ceiling because the wait holds an MCP
#: request open, and a client that gives up on it turns a saving into a retry.
_POLL_WAIT_DEFAULT = 10.0
_POLL_WAIT_MAX = 30.0

_WAIT_DESC = (
    "Seconds to wait for a *running* job before answering. The call returns "
    "the moment the job finishes, so this is a ceiling and not a delay -- a job "
    "that ends in 2s answers in 2s. Leave it alone unless you have a reason: "
    f"the default ({_POLL_WAIT_DEFAULT:.0f}s) is there so watching a job costs "
    f"a handful of calls instead of hundreds. Raise it (up to "
    f"{_POLL_WAIT_MAX:.0f}s) when you have nothing to do until the job ends; "
    "set it to 0 only when you want a snapshot right now and will not "
    "immediately ask again."
)


@mcp.tool()
async def poll_job(
    job_id: str,
    wait: Annotated[float, Field(description=_WAIT_DESC, ge=0)] = _POLL_WAIT_DEFAULT,
) -> str:
    """Get the status and output of a job started by execute_code.

    Returns the job's status (running/ok/error/interrupted), elapsed time, and
    output so far (full output once terminal). Job records persist until the
    kernel is restarted (older terminal jobs are eventually evicted).

    **This call already waits, so do not poll it in a loop.** A running job is
    watched here for up to `wait` seconds and answered the instant it ends;
    calling again immediately just asks the same question sooner and buys
    nothing. If you have nothing else to do until the job finishes, raise
    `wait` rather than calling more often. A terminal job, an unknown job id
    and a dead kernel all answer immediately whatever `wait` says.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err

    foreign_note = _writers._foreign_activity_note(host)
    if _scratch.owns(job_id):
        # A verification runs in a kernel this one cannot see. The id says which,
        # because the session child issued it (_scratch._ID_PREFIX).
        snap = _scratch.poll(job_id)
        if snap is None:
            return f"No such job '{job_id}'." + foreign_note
        if snap.get("status") == "running" and wait > 0:
            snap = await _await_verification(job_id, min(wait, _POLL_WAIT_MAX))
        return _format_job_status(snap) + foreign_note
    # A job that is already terminal -- the common case for a caller that has
    # been polling -- answers with no delay; only a running one costs the wait.
    snap = await _await_job(host, job_id, budget=min(max(wait, 0), _POLL_WAIT_MAX))
    if snap.get("status") == "unknown":
        return f"No such job '{job_id}'." + foreign_note
    return _format_job_status(snap) + foreign_note


@mcp.tool()
async def inspect_object(object_path: str) -> str:
    """Inspect a live object in the napari kernel namespace.

    Returns the type, docstring, and public methods/attributes.
    Example: inspect_object("viewer.layers") or inspect_object("viewer.camera")
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    refusal = _own_cell_holds_main(host, "inspecting an object")
    if refusal is not None:
        return refusal

    snippet = _INSPECT_TEMPLATE.replace("__PATH__", repr(object_path))
    res = await _kernel_rpc._execute(host, snippet)
    if res.get("status") == "ok":
        return res.get("stdout", "").rstrip() or "(no output)"
    return res.get("error_text") or f"(status: {res.get('status')})"


@mcp.tool()
async def interrupt_kernel() -> str:
    """Force-stop your running job: your cell, or your run_async task.

    A cell runs on the kernel's main thread and gets a SIGINT, which also wakes
    a blocking sleep or wait; a task gets a KeyboardInterrupt raised into its
    thread. Either lands at the next bytecode, so a blocking C-level call (gRPC
    tensor fetch, native compute) stops only when it returns to Python. Also
    cancels the job's in-flight dask futures, which is what stops a blocking
    `.compute()` -- but only on a dask `Client` a cell built. If YOUR job
    stays stuck, use restart_kernel -- the guaranteed stop.

    Stops YOUR job only. A cell the user runs from an attached Jupyter notebook
    shares this kernel but is not yours to stop: this refuses it, so wait for it
    instead. A refusal is not a stuck
    kernel and restart_kernel is not the way around it — restarting would destroy
    the user's running cell, variables and layers along with yours. Wait, or ask
    them.

    Takes no argument for which job because you have at most one running: the
    slot is the session's, not a kernel's. So this also stops a verification of yours
    running in its scratch kernel, and refuses one that is not yours, by the
    same rule and for the same reason.

    **On a verification it is the guaranteed stop, not a best-effort one.** If
    the cells do not stop promptly the scratch kernel is killed outright and
    discarded — which is safe precisely because nothing in it is anyone's work.
    So a stuck verification never needs restart_kernel: reaching for it there
    would destroy the user's session to end a process that exists to be thrown
    away.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    writer, _label = _writers._client_identity()
    # Which kernel to signal is this process's business, not the tool
    # signature's: the slot is global, so "the running job" is unambiguous.
    # `None` back means there was no verification to stop -- which is not the
    # same as nothing running, since one that ended a moment ago leaves the
    # session kernel free to have started something. So fall through; never
    # answer "nothing is running" from here.
    data = await asyncio.to_thread(
        _scratch.interrupt, None, _writers._local_origin.get(), writer
    )
    if data is not None:
        job_id = data.get("job_id")
        if data.get("interrupted"):
            how = (
                "its cells would not stop, so the kernel was killed"
                if data.get("killed")
                else "its cells stopped"
            )
            return (
                f"Stopped verification {job_id} — {how}, and the scratch kernel "
                "is discarded. Your session is untouched; nothing here needs "
                "restart_kernel."
            )
        refused = _stop_refused_message(data, job_id)
        if refused is not None:
            return refused
        return (
            f"Nothing to interrupt in verification {job_id}; it may have "
            "finished already. Poll it to see."
        )
    origin = _writers._local_origin.get()
    running = host.jobs.running(prefer=origin)
    if running is None:
        return "No running job to interrupt."
    job_id = running["job_id"]
    # Decided here, where the records and the claim are: the kernel stops
    # what it is told to.
    data = _writers.stop_refusal(
        _writers.claim_holder(), running.get("origin"), writer, origin
    )
    if data is None:
        try:
            data = await asyncio.to_thread(host.interrupt_job, job_id)
        except Exception as exc:  # noqa: BLE001 - the agent reads why
            return f"Could not reach the kernel to interrupt {job_id}: {exc}"
    if data.get("refused") == "not_running":
        now = data.get("running_job_id")
        return f"{job_id} is no longer running" + (
            f"; {now} is. Poll it before deciding to stop it." if now else "."
        )
    refused = _stop_refused_message(data, job_id)
    if refused is not None:
        return refused
    if data.get("interrupted"):
        return (
            f"Interrupted job {job_id} (a KeyboardInterrupt, at its next "
            "bytecode). If it does not stop, use restart_kernel."
        )
    return "No running job to interrupt."


def _stop_refused_message(data, job_id):
    """The tool's reply to a stop refused by ``_writers.stop_refusal``, or
    None when *data* is no such refusal."""
    if data.get("refused") == "not_owner":
        return _writers._NOT_OWNER_MSG.format(held_by="")
    if data.get("refused") == "foreign_job":
        # "Foreign" is not a synonym for "the user's": it is anything this agent
        # did not start. Naming the wrong writer would tell the agent to wait on
        # a person who is not there.
        by = (
            "the user" if (data.get("origin") or "user") == "user" else "another writer"
        )
        who = "The user" if by == "the user" else "Whoever started it"
        return (
            f"Refused: {job_id} was started by {by}, not by you — it is not "
            f"yours to stop. Wait for it and poll_job('{job_id}'). ({who} can "
            "stop it.)"
        )
    return None


@mcp.tool()
async def start_kernel() -> str:
    """Start biopb: bring up the napari viewer, dask, and the tensor client.

    Call this as the first action of every session, and whenever the user asks
    to start, open, or launch biopb, napari, or the viewer. Nothing auto-starts
    and every other kernel tool fails until this returns, so when in doubt call
    it -- a ready kernel is a no-op.

    It BLOCKS until the kernel is ready (or the bring-up fails), so on return
    you can use execute_code / take_screenshot / inspect_object directly, with
    no polling.

    It is also the recovery path: after a failed start, a dead kernel, or the
    user closing the viewer window (which tears the kernel down to idle), call
    it again to rebuild. A kernel that is already up is left untouched, so this
    is not the way to clear a wedged or misbehaving one -- that is
    interrupt_kernel (stop the running job) or restart_kernel (hard-restart,
    discarding the user's work, which this never does).
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    result = await asyncio.to_thread(host.ensure_started)
    if result.get("state") == "ready":
        ready = (
            "Kernel ready. The napari viewer, dask, and tensor client are up; "
            "use execute_code / take_screenshot now."
        )
        # A virtual display is a silent degradation: screenshots still work, so
        # nothing downstream notices, but the user is watching a window that
        # does not exist and paying software GL for it. Only they can fix it, so
        # the agent has to be told to say so (#892).
        display = host.virtual_display
        if display:
            ready += (
                "\n\nWARNING: no display was detected, so the napari window is "
                f"on a virtual one (Xvfb {display}). Screenshots work, but the "
                "window is invisible to the user and software GL renders 3-D "
                "volumes ~13x slower than a real GPU.\n"
                "TELL THE USER THIS NOW, before doing any work: no napari "
                "window will appear for them. Show results through the web "
                "viewer instead — server_status has its URL, and it needs no "
                "display on this machine. Usually the host does have one and "
                "their MCP client dropped $DISPLAY on the way in (Codex CLI "
                "does this) — so ask whether they are at a desktop here before "
                "treating the host as headless."
            )
        return ready
    return (
        "Kernel failed to start: "
        + str(result.get("error", "unknown error"))
        + " Check server_status; call start_kernel to retry."
    )


@mcp.tool()
async def restart_kernel() -> str:
    """Hard-restart the kernel: the guaranteed stop for runaway execution.

    Kills the kernel process group (reaping any dask child processes) and
    respawns a fresh kernel, rebuilding the tensor client and the napari
    viewer. All variables defined in previous execute_code calls are lost; a
    new viewer window replaces the old one.

    This destroys the USER's work too, not only yours — their running cell,
    their variables, their layers — and it is not undoable or announced to them
    beforehand. So it is not the way past a refused interrupt_kernel or a kernel
    busy with a user cell: neither is a runaway. Use it when the kernel is truly
    wedged, and prefer asking first when someone is working in it.

    It is also not the way past a kernel held by another client: if you do not
    hold this one, this is refused too, and restarting it is the user's to do.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    # Gated like every other state change, and this is the sharpest of them: a
    # restart discards the holder's whole session. Gate, restart and clear are
    # one call because they have to be one critical section -- see
    # _writers.restart. It blocks for the length of the bring-up, so it goes to
    # a thread like every other kernel round trip here.
    writer, _label = _writers._client_identity()
    # A verification holds the slot in this process, not in the kernel being
    # replaced, so a restart that left it running would hand back a fresh kernel
    # that can accept nothing -- and a wedged verification would have no escape
    # hatch, which is exactly what this tool is documented to be. Handed to
    # _writers.restart rather than done here, so it happens only once the gate
    # above has passed: a refused caller must not have destroyed anything.
    try:
        refusal, discarded = await asyncio.to_thread(
            _writers.restart,
            host,
            writer,
            functools.partial(_scratch.discard, "restart_kernel"),
        )
    except Exception as exc:
        return f"Kernel restart failed: {exc}"
    if refusal is not None:
        return refusal
    note = f" Verification {discarded} was discarded with it." if discarded else ""
    return "Kernel restarted. Viewer rebuilt; previous variables are gone." + note


@mcp.tool()
async def server_status() -> str:
    """Report server health, system load, and resource usage.

    Returns CPU/memory usage (this MCP process / host), kernel liveness, and —
    queried from the kernel — its biopb-mcp/python versions, dask scheduler info,
    tensor server connectivity, viewer layer count, the available `ops`, and which
    kernel plugins loaded. Use before heavy computation, and to resolve a skill's
    `checklist:` list.
    """
    import psutil

    host = _app._kernel_host

    # Off the loop for the one call that pays the 0.1s priming sample.
    cpu_percent = await asyncio.to_thread(_cpu_percent, psutil)
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    proc_mem = process.memory_info()

    lines = [
        "## System",
        f"  cpu_usage: {cpu_percent}%",
        f"  cpu_count: {os.cpu_count()}",
        f"  memory_total: {mem.total / (1024**3):.1f} GB",
        f"  memory_available: {mem.available / (1024**3):.1f} GB",
        f"  memory_used_percent: {mem.percent}%",
        f"  process_rss: {proc_mem.rss / (1024**2):.0f} MB",
        f"  log_file: {_app._session_log_path or 'stdout (not file-logged)'}",
        "",
    ]

    # Observe web UI: server-process state, independent of the kernel, so report
    # it before (and regardless of) kernel health. No kernel round-trip.
    from . import _observe

    obs = _observe.describe(getattr(mcp.settings, "port", None))
    lines.append("## Observe")
    if obs["running"]:
        lines.append(f"  url: {obs['url']}")
        lines.append(f"  mode: {obs['mode']}")
    else:
        lines.append("  status: not running (observe.enabled off or failed to start)")
    lines.append("")

    # The display surface that does not need this session to have one. Reported
    # rather than probed: the control serves the page *and* this session's data
    # plane, so "## Tensor Server: connected" already answers whether it is up,
    # and a probe here would put a network round trip on every status call.
    lines.append("## Web viewer")
    lines.append(f"  url: {_viewer_base_url()}/viewer?id=<array_id>")
    lines.append("    Served by the control. Works with no napari window; shows")
    lines.append("    what is in the catalog, so a result has to be uploaded")
    lines.append('    first. read_doc("web-viewer") has the parameters.')
    lines.append("")

    # Where a doc the agent writes lands. Server-process state (the store is
    # read here, not in the kernel) and the path is configurable, so a doc body
    # quoting ~/.config/biopb/docs can be wrong.
    lines.append("## Docs")
    lines.append(_docs.local_dir_status())
    lines.append("")

    lines.append("## Kernel")

    if host is None:
        lines.append("  state: not initialized")
        return "\n".join(lines)

    health = host.health()
    lines.append(f"  alive: {health['alive']}")
    lines.append(f"  ready: {health['ready']}")
    lines.append(f"  busy: {health['busy']}")
    lines.append(f"  watchdog_running: {health['watchdog_running']}")
    if health["recent_respawns"]:
        lines.append(f"  recent_respawns: {health['recent_respawns']}")
    if health.get("connection_file"):
        # For the user: a notebook on this kernel shares its namespace. Its
        # cells are refused while a job runs and reported to the agent otherwise.
        lines.append(f"  connection_file: {health['connection_file']}")
        if health.get("attach_command"):
            lines.append(f"    attach a notebook: {health['attach_command']}")

    # Kernel-state summary: dead / failed / starting / not-started are mutually
    # exclusive (each implies ready is false), so report exactly one and return —
    # don't fall through and print a second, contradictory state. Each also skips
    # the kernel query below, which would block on readiness for the whole
    # startup budget. A user-attributed teardown reason (window close) is shown.
    teardown = health.get("teardown_reason")
    if health["dead"]:
        lines.append("  state: DEAD — respawn budget exhausted; call start_kernel")
        if health.get("start_error"):
            lines.append(f"    last error: {health['start_error']}")
        return "\n".join(lines)
    if not health["ready"]:
        # A recorded start_error means the bring-up failed terminally (vs. still
        # in progress); report it as failed, not "starting", so a broken
        # bootstrap is distinguishable from a slow boot.
        if health.get("start_error"):
            lines.append("  state: failed — kernel startup error:")
            lines.append(f"    {health['start_error']}")
            lines.append("  (call start_kernel to retry)")
        elif health.get("alive"):
            # A kernel process exists but isn't ready yet (e.g. a watchdog
            # respawn in flight). start_kernel itself blocks, so its caller won't
            # see this — but a concurrent observer / respawn can.
            lines.append(
                "  state: starting — kernel/viewer still booting; retry shortly"
            )
        else:
            line = "  state: not started — call start_kernel to launch the kernel"
            if teardown:
                line += f" (torn down: {teardown})"
            lines.append(line)
        return "\n".join(lines)

    res = await _kernel_rpc._execute(host, _STATUS_SNIPPET, 15.0)
    if res.get("status") == "ok":
        lines.append("")
        lines.append(res.get("stdout", "").rstrip())
    elif res.get("status") == "timeout":
        lines.append("  (kernel busy — dask/tensor/viewer status unavailable)")
    else:
        lines.append("")
        lines.append(
            "  kernel query error: " + (res.get("error_text") or str(res.get("status")))
        )

    # From the host's records, so the list survives a kernel that is busy.
    lines.append("")
    lines.append("## Jobs")
    jobs = host.jobs.summary()
    for j in jobs:
        lines.append(
            f"  - {j['job_id']}: {j['status']} ({j['elapsed']}s, "
            f"stdout {j['stdout_len']}b)"
        )
    if not jobs:
        lines.append("  (none)")

    # Only on this path: the early returns above are all "kernel not usable".
    return "\n".join(lines) + _writers._foreign_activity_note(host)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def run(port: int = 8765, allowed_origins=(), allowed_hosts=(), *, sock=None):
    """Run the MCP server in the foreground (streamable-http).

    ``allowed_origins`` / ``allowed_hosts`` extend the loopback Host/Origin
    allowlist (see :func:`build_transport_security`).  They are applied before
    serving, when the streamable-http app reads ``transport_security``.

    ``sock`` is an already-bound listening socket. When given we serve over it
    with an explicit ``uvicorn.Server`` instead of letting FastMCP bind ``port``
    itself: the de-daemonized shim-owned child (ARCHITECTURE.md, Lifecycle)
    binds port 0 up front so it can report the OS-assigned port back to
    its shim *before* serving, then hands the socket here. The Starlette app
    FastMCP builds carries the ``session_manager.run()`` lifespan on its own
    (``streamable_http_app``), so a plain uvicorn run drives it — identical to
    the ``mcp.run`` path, only with the socket pre-bound.
    """
    mcp.settings.transport_security = _app.build_transport_security(
        allowed_origins, allowed_hosts
    )
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = port
    logger.info("MCP server listening on http://127.0.0.1:%d/mcp", port)
    if sock is None:
        mcp.run(transport="streamable-http")
        return

    import asyncio

    import uvicorn

    config = uvicorn.Config(
        mcp.streamable_http_app(),
        host="127.0.0.1",
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


# run_stdio() is gone: this process serves http only (the shim-owned session
# model). stdio clients are served by the launcher's bridge mode
# instead — see `_shim`, which fronts this server's /mcp endpoint.
