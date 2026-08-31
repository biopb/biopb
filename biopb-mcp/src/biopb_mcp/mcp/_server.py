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
"""

import logging
import os
import time
from typing import Annotated

from mcp.types import ImageContent, TextContent
from pydantic import Field

from . import _app, _kernel_rpc, _resources, _skills, _writers
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
try:
    import dask as _dask
    print("  scheduler: " + str(_dask.config.get("scheduler", default="unknown")))
except Exception as _e:
    print("  error: " + str(_e))
try:
    if _dask_client is not None:
        _info = _dask_client.scheduler_info()
        print("  distributed_workers: " + str(len(_info.get("workers", {}))))
        print("  dashboard: " + str(_dask_client.dashboard_link))
    elif not globals().get("_dask_attach_done", True):
        print("  distributed: starting (attaching to cluster)")
    else:
        print("  distributed: not active")
except Exception:
    print("  distributed: not active")

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
    print("    volumes ~13x slower than a real GPU. TELL THE USER, if you have")
    print("    not already. Usually the host does have a display and the MCP")
    print("    client dropped $DISPLAY on the way in (Codex CLI does) — ask")
    print("    whether they sit at a desktop on this machine before treating")
    print("    the host as headless.")
else:
    print("  display: " + str(
        _os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY") or "?"
    ))
if not _viewer_window_alive():
    print("  window: CLOSED — the napari window was closed; layer mutations")
    print("    won't display. Data/compute still work; restart_kernel to restore.")
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

print("")
print("## Jobs")
try:
    _js = _jobs.jobs_summary()
    if _js:
        for _j in _js:
            print(
                "  - " + _j["job_id"] + ": " + _j["status"]
                + " (" + str(_j["elapsed"]) + "s, stdout "
                + str(_j["stdout_len"]) + "b)"
            )
    else:
        print("  (none)")
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


def _start_job(host, code, **kwargs):
    """Submit a job and resolve everything that can happen before it runs.

    *code* and *kwargs* are :func:`_jobs.submit`'s own arguments, passed as
    values. The client identity is added here rather than passed in, because
    claiming the kernel is this function's business:
    every submitting tool answers "am I the holder?" and "is something already
    running?" the same way, and a second answer to either is a second policy.

    Returns ``(job_id, foreign_note, window_alive, message)``. Exactly one of
    *job_id* and *message* is not None — *message* is the finished tool reply
    for every outcome that never started a job, and *foreign_note* is what the
    caller appends to whatever it returns instead. *window_alive* rides along
    because the submit round trip carries it too, and with a promote window of
    zero it is the only one there will be.
    """
    writer, _label = _writers._client_identity()

    # Read once at entry, append to whichever path returns below.
    digest = _writers._foreign_digest(host)
    foreign_note = _writers._render_foreign_note(digest)
    if foreign_note:
        _writers._ack_foreign_digest(host, digest, writer)

    job_id, message, drop_note, window_alive = _submit_job(
        host, code, digest, _tool_busy_message, **kwargs
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
    their work; interrupt_kernel refuses that anyway (_jobs.interrupt_current),
    so the wording must not send it there.
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


def _submit_job(host, code, digest, busy_message, **kwargs):
    """Claim the kernel, submit *code*, and classify whatever comes back.

    The claim protocol -- presume before the call, then believe the kernel's
    answer -- is the same for every writer of this namespace, and it is a
    security-relevant one, so it is written once here. Both the MCP tools (via
    :func:`_start_job`) and the in-process chat loop go through it.

    What legitimately differs between those surfaces is only how a busy kernel
    is described: a tool caller gets a job handle it can poll, the chat loop
    never does (its path has no promote window), so it must not be told to poll
    one. Hence *busy_message*, a ``(running_job_id, running_origin) -> str``.

    Returns ``(job_id, message, drop_note, window_alive)``; exactly one of
    *job_id* and *message* is not None. *drop_note* asks the caller to suppress
    its foreign-activity note: a running foreign job stays in the digest by
    design, so the note would report the very job the refusal already reports.
    Keep it when other cells also finished -- those were acked and will not be
    offered again.
    """
    writer, writer_label = _writers._client_identity()
    # Before the call, not after: a lost reply must not leave the kernel claimed
    # while this process still reads as unclaimed. See _claimed_by.
    _writers._presume_claim(writer)
    submitted, res, window_alive = _kernel_rpc._run_job_call(
        host, "submit", code, writer=writer, writer_label=writer_label, **kwargs
    )
    if submitted is None:
        return None, _kernel_rpc._format_execute_result(res), False, window_alive
    if submitted.get("error") == "not_owner":
        # The authority speaking: whatever this process presumed above, the
        # kernel just named the real holder.
        _writers._note_claim(submitted.get("owner_id"))
        held_by = submitted.get("owner") or ""
        held_by = f" ({held_by})" if held_by else ""
        return (
            None,
            _writers._NOT_OWNER_MSG.format(held_by=held_by),
            False,
            window_alive,
        )
    # Anything the kernel did not refuse came from the holder, "busy" included:
    # submit() decides the claim before it looks at what is running.
    _writers._note_claim(writer)
    if submitted.get("error") == "busy":
        running = submitted.get("running_job_id")
        drop_note = [d.get("job_id") for d in digest] == [running]
        return (
            None,
            busy_message(running, submitted.get("running_job_origin")),
            drop_note,
            window_alive,
        )
    return submitted["job_id"], None, False, window_alive


def _await_job(host, job_id, window_alive=None):
    """Poll *job_id* until it is terminal or the promote window runs out.

    Returns ``(snap, res, window_alive)``. *snap* is None when a poll round trip
    failed, and *res* is the raw kernel reply to report instead. Otherwise a
    ``status`` of ``running`` means the window expired and the caller hands back
    a job handle rather than a result.

    *window_alive* seeds the flag with the submit round trip's, so a promote
    window short enough to poll zero times still reports a closed viewer.
    """
    deadline = time.monotonic() + _app._promote_after
    snap, res = {"status": "running"}, None
    while time.monotonic() < deadline:
        time.sleep(0.4)
        snap, res, window_alive = _kernel_rpc._run_job_call(host, "poll", job_id)
        if snap is None:
            return None, res, None
        if snap.get("status") != "running":
            break
    return snap, res, window_alive


def _format_verification(record: dict, job_id: str) -> str:
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
            f"Verified: all {len(cells)} cell(s) ran in a scratch namespace "
            "(the kernel's own handles, none of this session's variables)."
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
        # (see _jobs._Cell.snapshot).
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

    added = record.get("added_layers") or []
    if added:
        lines.append(
            "\nLayers this run added to the live viewer: "
            + ", ".join(added)
            + ". A scratch namespace isolates variables, not the viewer — remove "
            "them if they are duplicates of what was already there."
        )

    if status == "ok":
        lines.append(
            "\nThe user can now save this workflow as a notebook from the "
            "observe page ('Save workflow'). Tell them it is there; do not write "
            "the file yourself."
        )
    else:
        lines.append(
            f"\nFix the cell and call verify_workflow again. Full record: "
            f"poll_job('{job_id}')."
        )
    return "\n".join(lines)


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
        return header + "\n" + _format_verification(record, job_id)
    body = _kernel_rpc._format_execute_result(snap)
    if status == "running":
        return header + "\nPartial output:\n" + (body or "(none yet)")
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("guide://kernel")
def get_kernel_guide() -> str:
    """Overview: available namespaces, helper functions, resource URIs.

    The skill-requirements section is appended only when the catalog is enabled
    (``services.skills_enabled``): with it off there is no ``list_skills`` to
    return a ``checklist:``, so the section would document an unreachable
    tool -- the same gate the handshake instructions use.
    """
    if _app._skills_enabled:
        return _resources.GUIDE + _resources.SKILL_REQUIREMENTS
    return _resources.GUIDE


@mcp.resource("guide://data")
def get_data_guide() -> str:
    """How array data is represented here: the three sources, and their traps."""
    return _resources.DATA


@mcp.resource("guide://viewer")
def get_viewer_guide() -> str:
    """Viewer operations: layers, camera, dims, display."""
    return _resources.VIEWER


@mcp.resource("guide://client")
def get_client_guide() -> str:
    """The `client` handle: listing sources, loading, uploading."""
    return _resources.CLIENT


@mcp.resource("guide://ops")
def get_ops_guide() -> str:
    """Image processing operations: segmentation, feature extraction, super-resolution."""
    return _resources.OPS


@mcp.resource("skill://{skill_id}")
def get_skill(skill_id: str) -> str:
    """Full workflow body for a curated skill; discover ids with `list_skills`.

    The catalog (metadata) is served separately via the `list_skills` tool; this
    resource reads one skill's markdown body from the file the catalog named.
    Fail-open: returns a short explanatory string rather than erroring when a
    skill is unknown or its file is unreadable.
    """
    return _skills.get_skill_body(skill_id)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_skills(keywords: list[str] | None = None) -> list:
    """Discover curated biopb workflows ("skills"). Call at the start of a task.

    Skills are vetted, reusable recipes (e.g. "segment nuclei", "measure
    labels").

    **`keywords` is a keyword filter, not a search engine.** Each keyword must
    appear in a skill's id/title/description/tags, so every one you add can only
    remove results. Pass **one or two** domain terms and widen from there:
    `["drift"]`, `["fret"]`, `["illumination"]`, `["stitch", "tiles"]`. Omit it
    to list the whole catalog — worth doing once, since it is small.

    **An empty result usually means too many keywords, not no such skill.**
    `["count", "foci", "per", "nucleus"]` returns nothing while `["foci"]`
    returns the skill that counts them. If you get nothing back, drop keywords
    and call again, or call with none and read the list.

    **Results are of two `kind`s, and they are used differently.**

    - `kind="skill"` — a curated workflow. It carries a `uri` (`skill://<id>`);
      read that resource for the full step-by-step body. Prefer an existing
      skill over improvising.
    - `kind="plugin"` — a Python module already loaded into the kernel
      namespace, listed with its docstring summary. There is no body to read.
      It carries a `handle`, the name it is bound under: call
      `inspect_object(handle)` for its callables and signatures, then use it in
      `execute_code` as `handle.some_function(...)`. **Prefer it over writing
      your own** — these exist because the from-scratch version is slow, subtly
      wrong, or both, and the docstring says which.

    Skills are listed before plugins.

    A result's `checklist` lists what the skill touches (`viewer`, `tensor`, `dask`,
    `ops:<name>`, `plugin:<name>`, `pkg:<name>`). Resolve it before starting the
    skill — it informs rather than blocks, so a gap is something to name and
    work around, not a reason to abandon the skill: `server_status` answers every token except a third-party `pkg:` — it
    does carry biopb-mcp's own version — and for those, `execute_code` an
    `import <name>` and read the version with
    `importlib.metadata.version("<name>")`, not the module's `__version__`
    (packages forget to bump it). A gap is the user's call —
    installing, seeding a plugin, restarting the kernel all need their consent —
    but naming it up front beats failing halfway through.

    Fail-open: returns an empty list (never errors) rather than reporting a
    catalog that could not be read.
    """
    return _skills.list_skills(keywords or ())


@mcp.tool()
def take_screenshot(canvas_only: bool = True) -> list:
    """Capture the napari viewer as a PNG image.

    Args:
        canvas_only: If True, capture only the canvas area. If False,
            capture the entire viewer window.

    Returns a PNG screenshot as an image content block.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return [TextContent(type="text", text=err)]

    snippet = _SCREENSHOT_SNIPPET.format(canvas_only=bool(canvas_only))
    res = host.execute(snippet)
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
PROMOTE_PARAGRAPH = """Code runs in a background thread so it does not block the main thread.
    If it finishes quickly the result is returned inline; otherwise this returns
    a job handle (job-N) and the code keeps running. Poll it with poll_job,
    watch it with take_screenshot / server_status, and stop it with
    interrupt_kernel (best-effort) or restart_kernel (guaranteed). Only one job
    runs at a time."""


@mcp.tool()
def execute_code(
    python_code: str,
    intent: Annotated[str, Field(description=_INTENT_DESC)] = "",
) -> str:
    """Execute Python code in the napari kernel.

    The kernel is a full Jupyter/IPython kernel (imports allowed) with the
    namespace: viewer (with an add_tensor method), client(image data access), and ops (a
    dict of image processing operations). np and da are also imported. Variables persist
    across calls until the kernel is restarted.

    Code runs in a background thread so it does not block the main thread.
    If it finishes quickly the result is returned inline; otherwise this returns
    a job handle (job-N) and the code keeps running. Poll it with poll_job,
    watch it with take_screenshot / server_status, and stop it with
    interrupt_kernel (best-effort) or restart_kernel (guaranteed). Only one job
    runs at a time.

    Only one *agent* runs code in a kernel, too: whoever calls this first holds
    it until the kernel restarts. A second client is refused here and by every
    other tool that changes kernel state (interrupt_kernel, restart_kernel), and
    keeps only the read-only ones. The person at the machine is exempt — they
    can run cells from the observe page while you work, which is what the
    user-activity notice on these results is telling you about.

    Results include print() output and the last expression's repr. Rich IPython
    display() output is not captured; use print().

    * viewer mutations (see guide://viewer for more details):
    The viewer is thread-safe: mutations are auto-marshaled to the Qt main
    thread, so mutate it directly from job code. run_on_main(fn) is optional --
    use it to batch many mutations into one main-thread hop, or to touch raw Qt
    (viewer.window), which still requires the main thread.

    * data access (see guide://client for more details):
    - client.query_sources(sql, format="pandas") runs server-side DuckDB and
      returns a DataFrame. The `sources` table columns are: source_id,
      source_url, source_type, dtype, indexed_at, metadata_json, shape_summary,
      data_resident (note source_url, not "url"). Prefer this over
      client.list_sources() (server-capped for large catalogs). Unresolved
      (cloud) sources have NULL dtype/shape_summary, so a `WHERE dtype=...`
      predicate hides them; use `data_resident` to filter on residency on
      purpose (e.g. `WHERE NOT data_resident` to list what isn't resolved yet).
    - viewer.add_tensor(array_id) loads a tensor as a layer (auto-handles the
      multiscale pyramid); client.get_tensor(array_id) returns a lazy dask
      array without adding a layer. Both take the same id: "source_id/t1"
      within a multi-tensor source, a bare "source_id" for a single-tensor one.
    - reading pixels back off a layer is not plain napari: layer.data is a
      *list* of pyramid levels when layer.multiscale, in display axis order
      ([..., Z, Y, X], at the source's own rank), and lazy. Use
      `layer.data[0] if layer.multiscale else layer.data`, and read
      guide://data before measuring or computing from a layer.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err

    job_id, foreign_note, window_alive, msg = _start_job(
        host, python_code, intent=intent
    )
    if msg is not None:
        return msg

    snap, res, window_alive = _await_job(host, job_id, window_alive)
    if snap is None:
        return _kernel_rpc._format_execute_result(res) + foreign_note
    if snap.get("status") != "running":
        return (
            _kernel_rpc._format_execute_result(snap)
            + _kernel_rpc._window_note(window_alive)
            + foreign_note
        )

    partial = snap.get("stdout", "") if snap else ""
    return (
        f"Job {job_id} is still running after {_app._promote_after:.0f}s. "
        f"Poll it with poll_job('{job_id}'); watch with take_screenshot / "
        f"server_status; stop with interrupt_kernel or restart_kernel.\n"
        "Partial output:\n"
        + (partial or "(none yet)")
        + _kernel_rpc._window_note(window_alive)
        + foreign_note
    )


@mcp.tool()
def verify_workflow(
    cells: list[str],
    title: str = "",
) -> str:
    """Check that a candidate workflow runs on its own, in a scratch namespace.

    Use this when the user wants a workflow they have just proven kept as a
    document. Rewrite the session into a clean program — one entry in *cells*
    per notebook cell — and verify it here; on success the user can save it as a
    notebook from the observe page.

    **Rewrite it, do not select from it.** The program that works is almost
    never a subsequence of what was run: a cell that created a variable and a
    later cell that corrected its value have to merge into one, and dead ends,
    retries, and debugging prints drop out. Read the session with poll_job and
    write the cells you *mean*, in the order a reader would want them.

    Each cell runs in order in a namespace seeded with the kernel's own handles
    (np, da, client, ops, viewer) and nothing this session has bound since it
    started. The run stops at the first failure; the cells after it are reported
    as skipped.

    **This costs no restart.** The live session — its variables, its layers, its
    dask cluster — is untouched, so there is nothing to ask the user about
    before calling this.

    **What it proves:** every cell ran, and no cell leaned on a variable it did
    not itself create. That is the defect that makes a transcript unrunnable.
    **What it does not:** the numbers are right (check them), and the viewer and
    imported modules are shared — a cell that reads an existing layer by name
    will find one here and not on a fresh kernel. Layers the run adds are added
    to the real viewer, and are reported back so you can say so.

    Args:
        cells: the workflow's cells, in order, each a complete piece of Python.
        title: what the workflow does, in a few words. Names the saved file and
            titles the notebook.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    if not cells:
        return "verify_workflow needs at least one cell."

    job_id, foreign_note, window_alive, msg = _start_job(
        host,
        "",
        intent=f"verify workflow: {title}" if title else "verify workflow",
        verify_cells=list(cells),
        verify_title=title,
    )
    if msg is not None:
        return msg

    snap, res, window_alive = _await_job(host, job_id, window_alive)
    if snap is None:
        return _kernel_rpc._format_execute_result(res) + foreign_note
    if snap.get("status") == "running":
        return (
            f"Verification {job_id} is still running after "
            f"{_app._promote_after:.0f}s. Poll it with poll_job('{job_id}') — the "
            "per-cell record is in the result." + foreign_note
        )
    record = snap.get("verify")
    if not record:
        # A kernel that predates verify_cells, or a submit that never built the
        # record: report the job the ordinary way rather than invent a verdict.
        return (
            _kernel_rpc._format_execute_result(snap)
            + _kernel_rpc._window_note(window_alive)
            + foreign_note
        )
    return (
        _format_verification(record, job_id)
        + _kernel_rpc._window_note(window_alive)
        + foreign_note
    )


@mcp.tool()
def poll_job(job_id: str) -> str:
    """Get the status and output of a job started by execute_code.

    Returns the job's status (running/ok/error/interrupted), elapsed time, and
    output so far (full output once terminal). Job records persist until the
    kernel is restarted (older terminal jobs are eventually evicted).
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err

    foreign_note = _writers._foreign_activity_note(host)
    snap, res, window_alive = _kernel_rpc._run_job_call(host, "poll", job_id)
    if snap is None:
        return _kernel_rpc._format_execute_result(res) + foreign_note
    if snap.get("status") == "unknown":
        return f"No such job '{job_id}'." + foreign_note
    note = (
        _kernel_rpc._window_note(window_alive)
        if snap.get("status") != "running"
        else ""
    )
    return _format_job_status(snap) + note + foreign_note


@mcp.tool()
def inspect_object(object_path: str) -> str:
    """Inspect a live object in the napari kernel namespace.

    Returns the type, docstring, and public methods/attributes.
    Example: inspect_object("viewer.layers") or inspect_object("viewer.camera")
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err

    snippet = _INSPECT_TEMPLATE.replace("__PATH__", repr(object_path))
    res = host.execute(snippet)
    if res.get("status") == "ok":
        return res.get("stdout", "").rstrip() or "(no output)"
    return res.get("error_text") or f"(status: {res.get('status')})"


@mcp.tool()
def interrupt_kernel() -> str:
    """Force-stop the current job by raising KeyboardInterrupt in its thread.

    Also cancels the job's in-flight dask futures. The job runs in a background
    worker thread, so a SIGINT (which Python delivers only to the kernel main
    thread) can't reach it — this raises the exception directly into the worker.
    Best-effort: it lands at the next bytecode, so a
    blocking C-level call (gRPC tensor fetch, native dask compute) stops only when
    it returns to Python; if YOUR job stays stuck, use restart_kernel — the
    guaranteed stop.

    Stops YOUR job only. A cell the user ran from the observe page shares this
    kernel and this one-job-at-a-time runner, but is not yours to stop: this
    refuses it, and you should wait for it instead. A refusal is not a stuck
    kernel and restart_kernel is not the way around it — restarting would destroy
    the user's running cell, variables and layers along with yours. Wait, or ask
    them.
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    writer, _label = _writers._client_identity()
    data, res, _w = _kernel_rpc._run_job_call(
        host, "interrupt_current", requester="mcp", writer=writer
    )
    if data is None:
        return _kernel_rpc._format_execute_result(res)
    if data.get("refused") == "not_owner":
        return _writers._NOT_OWNER_MSG.format(held_by="")
    if data.get("refused") == "foreign_job":
        running = data.get("job_id")
        # "Foreign" is not a synonym for "the user's": it is anything this agent
        # did not start. Naming the wrong writer would tell the agent to wait on
        # a person who is not there.
        by = (
            "the user" if (data.get("origin") or "user") == "user" else "another writer"
        )
        who = "The user" if by == "the user" else "Whoever started it"
        return (
            f"Refused: {running} was started by {by}, not by you — it is not "
            f"yours to stop. Wait for it and poll_job('{running}'). ({who} can "
            "stop it.)"
        )
    if data.get("interrupted"):
        return (
            f"Interrupted job {data.get('job_id')} (KeyboardInterrupt raised in "
            "its thread). If it does not stop, use restart_kernel."
        )
    return "No running job to interrupt."


@mcp.tool()
def start_kernel() -> str:
    """Start the napari kernel on demand (it does not auto-start).

    The MCP server stays cheap and idle until you call this; it then brings up
    the child IPython kernel, dask, the tensor client, and the napari viewer
    window. This BLOCKS until the kernel is ready (or the bring-up fails), so
    on return you can use execute_code / take_screenshot / inspect_object
    directly (no polling needed). A ready kernel is a no-op.

    Call this once at the start of a session. It is also the recovery path:
    after a failed start, a dead kernel, or the user closing the viewer window
    (which tears the kernel down to idle), call start_kernel again to rebuild.
    (restart_kernel is for hard-restarting an already-running kernel.)
    """
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    result = host.ensure_started()
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
                "\n\nWARNING: no display was detected, so the viewer is on a "
                f"virtual one (Xvfb {display}). Screenshots work, but the "
                "window is invisible to the user and software GL renders 3-D "
                "volumes ~13x slower than a real GPU.\n"
                "TELL THE USER THIS NOW, before doing any work: no napari "
                "window will appear for them. Usually the host does have a "
                "display and their MCP client dropped $DISPLAY on the way in "
                "(Codex CLI does this) — so ask whether they are at a desktop "
                "on this machine before treating the host as headless."
            )
        return ready
    return (
        "Kernel failed to start: "
        + str(result.get("error", "unknown error"))
        + " Check server_status; call start_kernel to retry."
    )


@mcp.tool()
def restart_kernel() -> str:
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
    # restart discards the holder's whole session. Decided against the local
    # mirror (_claimed_by) rather than a round trip to the kernel, so a kernel
    # too busy to answer cannot be mistaken for an unclaimed one.
    writer, _label = _writers._client_identity()
    held = _writers.claim_holder()
    if held is not None and writer is not None and writer != held:
        return _writers._NOT_OWNER_MSG.format(held_by="")
    try:
        host.restart()
    except Exception as exc:
        return f"Kernel restart failed: {exc}"
    _writers.clear_claim()  # a fresh kernel is unclaimed until someone runs code in it
    return "Kernel restarted. Viewer rebuilt; previous variables are gone."


@mcp.tool()
def server_status() -> str:
    """Report server health, system load, and resource usage.

    Returns CPU/memory usage (this MCP process / host), kernel liveness, and —
    queried from the kernel — its biopb-mcp/python versions, dask scheduler info,
    tensor server connectivity, viewer layer count, the available `ops`, and which
    kernel plugins loaded. Use before heavy computation, and to resolve a skill's
    `checklist:` list.
    """
    import psutil

    host = _app._kernel_host

    cpu_percent = _cpu_percent(psutil)
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

    # Where a skill the agent writes has to land. Server-process state (the
    # catalog is scanned here, not in the kernel), and the path is configurable,
    # so a hard-coded ~/.config/biopb/skills in a skill body can be wrong.
    if _app._skills_enabled:
        lines.append("## Skills")
        lines.append(_skills.local_dir_status())
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

    res = host.execute(_STATUS_SNIPPET, timeout=15.0)
    if res.get("status") == "ok":
        lines.append("")
        lines.append(res.get("stdout", "").rstrip())
    elif res.get("status") == "busy":
        lines.append("  (kernel busy — dask/tensor/viewer status unavailable)")
    else:
        lines.append("")
        lines.append(
            "  kernel query error: " + (res.get("error_text") or str(res.get("status")))
        )

    # Only on this path: the early returns above are all "kernel not usable",
    # where the digest round-trip cannot land anyway.
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
