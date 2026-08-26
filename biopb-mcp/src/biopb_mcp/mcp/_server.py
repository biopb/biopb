"""FastMCP server exposing the napari viewer through a child Jupyter kernel.

The server runs in the foreground (uvicorn, streamable-http on
127.0.0.1:<port>/mcp) and owns a :class:`~biopb_mcp.mcp._kernel.KernelHost`.
Every tool call is a round-trip into that kernel, where the napari viewer,
dask, and the TensorFlightClient live.  The kernel can be interrupted or
hard-restarted independently of this process.
"""

import contextvars
import json
import logging
import os
import time

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent, TextContent

from . import _resources, _skills
from ._kernel import KernelHost

logger = logging.getLogger(__name__)

_kernel_host: KernelHost | None = None

# Seconds execute_code waits for a job to finish before returning a job handle
# instead of an inline result (set from config by the launcher).
_promote_after: float = 10.0

# Whether the curated-skills catalog is advertised to the agent (mirrors
# `services.skills_enabled`, on by default). Set by the launcher
# (set_skills_enabled); gates the _SKILLS_INSTRUCTIONS fragment in the handshake.
# test_mcp_server pins this literal to the config default so the two can't drift.
_skills_enabled: bool = True

# This process's logfile path (set by the launcher), surfaced by server_status so
# an agent can find its own log. None when output goes to a terminal (foreground
# `--transport http` / `biopb mcp view`) rather than a file.
_session_log_path: str | None = None

# Handed to the client in the initialize handshake (the only handshake-time
# carrier MCP defines). Clients that honor it inject it into the model's
# context from the first turn (compliance is up to the client/agent), so this
# field carries the guidance that must hold on *every* turn — the operation
# guardrails.
_BASE_INSTRUCTIONS = (
    "This biopb-mcp session drives a live napari viewer through a child IPython "
    "kernel; `execute_code` runs arbitrary Python in that kernel. Read these resources "
    "for detail before non-trivial work: guide://kernel (namespace, skill "
    "requirements, long-running jobs & cancellation), guide://data (how arrays are "
    "represented here -- pyramids, laziness, axis order and rank -- and the traps), "
    "guide://client (the `client` handle: catalog, load, upload), "
    "guide://viewer (layers/camera/dims, annotation layers), "
    "guide://ops (server-side image-processing ops).\n"
    "\n"
    "The napari kernel does NOT auto-start. Call `start_kernel` once at the "
    "start of the session (and again to recover after a failure or after the "
    "user closes the viewer window); it blocks until the kernel is ready.\n"
    "\n"
    "Operation guardrails (apply on every turn):\n"
    "- Use data from `client` or `viewer`; avoid the filesystem unless the user "
    "explicitly asks.\n"
    '- Browse the catalog with `client.query_sources(sql, format="pandas")` '
    "(server-side DuckDB, complete), not `client.list_sources()` "
    "(server-capped for large catalogs); the `sources` columns are source_id, "
    "source_url, source_type, dtype, indexed_at, metadata_json, "
    "shape_summary, data_resident, and `tensors` (a LIST of "
    "STRUCT(array_id, dim_labels, shape, dtype), one per tensor -- "
    "query per-tensor with UNNEST(tensors) or list_filter; the scalar "
    "dtype/shape_summary only describe tensors[0]). Unresolved (cloud / "
    "synced-folder) sources "
    "have NULL dtype/shape_summary, so a predicate like `WHERE dtype='uint8'` "
    "silently drops them; filter on `data_resident` to opt them in/out on "
    "purpose (`WHERE NOT data_resident` finds what hasn't been resolved yet).\n"
    "- Prefer lazy dask operations; only `.compute()` the final result.\n"
    "- Put intermediate results back on `viewer` for the user to validate at "
    "each step.\n"
    "- Do not assume — ask the user to clarify uncertainties; they know the "
    "data better than you do."
)

# Appended to _BASE_INSTRUCTIONS only when the skills catalog is enabled
# (`services.skills_enabled`, on by default). Kept out of the base so an install
# that switches skills off neither points the agent at `find_skills` (which would
# return nothing) nor prompts it to author skills — set_skills_enabled owns the
# field.
_SKILLS_INSTRUCTIONS = (
    "At the start of a task, call `find_skills` to check for a curated workflow "
    "before improvising; read the matching `skill://<id>` resource for the "
    "steps. Results marked `origin: local` are the user's own unreviewed skills "
    "from ~/.config/biopb/skills; prefer a curated one when both fit. After "
    "accomplishing a task, ask the user whether a new skill should be generated "
    "and added to the agent's toolbox for future use.\n"
    "\n"
    "Skills name three checkpoint types in their steps; honor them:\n"
    "- confirm-input: ask before computing, but only for facts the data cannot "
    "give you (voxel spacing, which channel is which, expected object size).\n"
    "- visual checklist: put the intermediate on the viewer and report two or three "
    "numbers with it -- never a screenshot alone, and report the numbers alone "
    "when the data is too large to show usefully.\n"
    "- validate-and-gate: stop and get the user's agreement before anything "
    "expensive or hard to walk back.\n"
    "Destructive steps always ask first, whatever a skill says: restarting the "
    "kernel, interrupting a running job, overwriting a layer, or writing files."
)

# DNS-rebinding / cross-origin protection (review finding A2).  execute_code is
# a full kernel (RCE by design), so the only thing standing between a malicious
# page in the user's own browser and the loopback port is Host/Origin
# validation.  The MCP SDK enforces these lists; we set them explicitly rather
# than relying on its implicit loopback auto-enable so the control can't
# silently regress.  Wildcard ports mean the configured port never matters.
_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOOPBACK_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]


def build_transport_security(
    extra_origins=(), extra_hosts=()
) -> TransportSecuritySettings:
    """Build DNS-rebinding protection settings for the loopback server.

    The loopback allowlists are always enforced; ``extra_origins`` /
    ``extra_hosts`` (from ``transport.allowed_origins`` /
    ``transport.allowed_hosts``) are appended so an admin fronting the server
    with a reverse proxy can permit the proxy's Host/Origin.
    """
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_LOOPBACK_HOSTS + list(extra_hosts),
        allowed_origins=_LOOPBACK_ORIGINS + list(extra_origins),
    )


mcp = FastMCP("biopb-mcp", transport_security=build_transport_security())

# FastMCP built the low-level server with instructions=None at import; seed the
# always-on base guidance now so it is present even if set_skills_enabled is
# never called (e.g. tests, or a standalone import), which recomposes from this
# base.
mcp._mcp_server.instructions = _BASE_INSTRUCTIONS

_PNG_DELIM = "<<PNG_B64>>"

# Delimiter for the single-line JSON payload the in-kernel job runner prints in
# reply to a submit/poll/cancel/list snippet (mirrors the _PNG_DELIM pattern).
_JOB_DELIM = "<<JOB_JSON>>"

# Sentinel printed by the screenshot snippet when the napari window has been
# closed (the viewer survives in the namespace, but its canvas is destroyed).
_WINDOW_CLOSED_DELIM = "<<WINDOW_CLOSED>>"

# Appended to a result when the agent's code ran but the viewer window is closed,
# so the silent no-op of viewer mutations is surfaced rather than read as success.
_WINDOW_CLOSED_NOTE = (
    "\n\n⚠ The napari viewer window is closed — viewer layers won't be "
    "displayed (data/compute results are still valid). Call restart_kernel to "
    "restore the viewer."
)


def _job_snippet(call: str) -> str:
    """Build a snippet that prints ``_jobs.<call>``'s result as delimited JSON.

    ``call`` is a fully-formed call expression with arguments already embedded
    via ``repr`` by the caller (agent code is RCE by design, but embedding via
    ``repr`` keeps the payload a valid literal regardless of its contents).

    The payload is ``{"r": <call result>, "w": <viewer window alive?>}`` so the
    same round-trip also reports whether the viewer window is still open (a
    user-closed window turns viewer mutations into silent no-ops). The liveness
    probe is auxiliary, so a kernel that never bound ``_viewer_window_alive``
    (e.g. a partial/test bootstrap) reports ``w: null`` rather than breaking the
    job round-trip.
    """
    return (
        "import json as _json\n"
        "print('" + _JOB_DELIM + "' + _json.dumps("
        "{'r': _jobs." + call + ", "
        "'w': globals().get('_viewer_window_alive', lambda: None)()}))\n"
    )


_SCREENSHOT_SNIPPET = (
    "import base64 as _b64, cv2 as _cv2\n"
    "if not _viewer_window_alive():\n"
    "    print('" + _WINDOW_CLOSED_DELIM + "')\n"
    "else:\n"
    # Under async slicing, force-sync the current view so the capture reflects
    # the state the agent just set, not a pre-load frame. No-op when async is
    # off or the bootstrap predates the helper (defensive globals().get).
    "    globals().get('_resync_view', lambda: None)()\n"
    "    _arr = viewer.screenshot(canvas_only={canvas_only})\n"
    "    _bgra = _cv2.cvtColor(_arr, _cv2.COLOR_RGBA2BGRA)\n"
    "    _ok, _buf = _cv2.imencode('.png', _bgra)\n"
    "    print('" + _PNG_DELIM + "' + _b64.b64encode(_buf.tobytes()).decode())\n"
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
if _os.environ.get("BIOPB_VIRTUAL_DISPLAY"):
    # Launcher-owned Xvfb (#90): screenshots work, but no human sees the window.
    print("  display: virtual (Xvfb) — the viewer window is not visible to the user")
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


def set_kernel_host(host: KernelHost):
    """Register the kernel host the tools dispatch to.

    A different host is a different kernel, so the mirrored one-agent claim goes
    with the old one rather than being inherited by the new.
    """
    global _kernel_host
    _kernel_host = host
    clear_claim()


def set_promote_after(seconds: float):
    """Set how long execute_code waits inline before returning a job handle."""
    global _promote_after
    _promote_after = float(seconds)


def set_session_log_path(path: str | None):
    """Record this process's logfile path for server_status to report."""
    global _session_log_path
    _session_log_path = path


def _recompose_instructions():
    """Rebuild the handshake ``instructions`` from ``_BASE_INSTRUCTIONS`` plus
    whichever optional fragments the current mode enables (skills).

    Recomposing from the base in both directions is idempotent, so flipping any
    dimension back off can't leave a stale fragment in the handshake while
    preserving the always-on base guidance. The low-level Server holds the
    `instructions` returned in the handshake.
    """
    parts = [_BASE_INSTRUCTIONS]
    if _skills_enabled:
        parts.append(_SKILLS_INSTRUCTIONS)
    mcp._mcp_server.instructions = "\n\n".join(parts)


def set_skills_enabled(enabled: bool):
    """Advertise (or hide) the curated-skills catalog in the agent's initialize
    ``instructions``. On by default; switching skills off also drops the
    directive, so the agent is never pointed at ``find_skills`` when it would
    return nothing."""
    global _skills_enabled
    _skills_enabled = bool(enabled)
    _recompose_instructions()


def _format_execute_result(res: dict) -> str:
    status = res.get("status")
    stdout = res.get("stdout", "")
    result_text = res.get("result_text", "")
    error_text = res.get("error_text", "")

    if status == "ok":
        out = stdout
        if result_text:
            out += result_text
        return out or "(no output)"

    parts = []
    if stdout:
        parts.append(stdout)
    if error_text:
        parts.append(error_text)
    return "\n".join(parts) if parts else f"(status: {status})"


def _extract_delimited(text: str, delimiter: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(delimiter):
            return line[len(delimiter) :]
    return None


def _extract_json(text: str):
    """Parse the single-line ``<<JOB_JSON>>`` payload from a job snippet."""
    payload = _extract_delimited(text, _JOB_DELIM)
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


def _run_job_call(host, call: str):
    """Run a ``_jobs.<call>`` snippet.

    Returns ``(result, raw_result, window_alive)`` where ``result`` is the
    parsed ``_jobs.<call>`` value (None if the snippet failed) and
    ``window_alive`` is the viewer-window liveness flag carried in the same
    payload (None when unknown, e.g. the snippet did not run cleanly).
    """
    res = host.execute(_job_snippet(call))
    if res.get("status") != "ok":
        return None, res, None
    payload = _extract_json(res.get("stdout", ""))
    if payload is None:
        return None, res, None
    return payload.get("r"), res, payload.get("w")


def _window_note(window_alive) -> str:
    """Closed-window warning to append when a result returns with no viewer.

    ``window_alive`` is None when liveness is unknown -> no note.
    """
    if window_alive is False:
        return _WINDOW_CLOSED_NOTE
    return ""


# This process's mirror of the kernel's one-agent claim: the last client whose
# code the kernel actually accepted, or None while unclaimed.
#
# The kernel owns the claim -- ``_jobs.submit`` is the choke point and the only
# thing that can enforce it atomically. But ``restart_kernel`` cannot be gated
# from inside the kernel it destroys, and asking the kernel who owns it first is
# both a check-then-act race and *fail-open on a busy kernel*: a round trip that
# comes back "busy" would read as "no owner", and a kernel busy running the
# holder's job is exactly when a stray restart costs the most. Every claim
# passes through this process, so mirroring it here answers the question with no
# round trip and no window.
#
# Set from the kernel's own decision wherever a reply arrives: any submit the
# kernel did not refuse came from the holder, and a refusal names the holder
# outright, so assigning on both keeps the mirror true through a restart that
# happened somewhere else (the observe page's, which clears it explicitly).
#
# **Recorded before the submit is sent, not after.** A reply can be lost while
# the kernel goes on to claim and run the code anyway -- ``execute_interactive``
# hands the request over before it starts its clock, so a timed-out call is still
# queued and executes when the main thread frees up. Setting the mirror only on
# the way back would leave it empty while the kernel is genuinely held, and an
# empty mirror lets a stranger restart the session that just started. The window
# is claimed first and corrected from whatever the kernel says, so the failure
# direction is "held by the client that asked" rather than "held by nobody".
_claimed_by: str | None = None


def _note_claim(writer):
    """Record that the kernel is held by *writer* (ignores ``None``)."""
    global _claimed_by
    if writer is not None:
        _claimed_by = writer


def _presume_claim(writer):
    """Take the claim for *writer* only if this process has not seen one.

    Guarded on "not seen": a client the kernel is about to refuse must never
    overwrite a holder already known here, and it will be corrected by the
    refusal in any case.
    """
    if _claimed_by is None:
        _note_claim(writer)


def clear_claim():
    """Forget the mirrored claim, for a caller that just replaced the kernel."""
    global _claimed_by
    _claimed_by = None


# Refusal for a client that does not hold this kernel's one-agent claim
# (_jobs.submit). Shared by every state-changing tool so the agent gets one
# explanation rather than three, and so the recovery named is the same in all of
# them: the person at the machine, never a second agent.
_NOT_OWNER_MSG = (
    "This kernel is already in use by another client{held_by}, and only one "
    "agent runs code in a session. Two of you writing to the same namespace and "
    "viewer would order the writes without either being able to see what the "
    "other believes is there. Reading tools (poll_job, server_status, "
    "take_screenshot, inspect_object) still work, so you can watch. You cannot "
    "take the session over — restarting it is the user's to do, from the observe "
    "page. Tell them what you wanted and let them decide."
)


# Identity for a caller that reaches the tools without an MCP request at all --
# the in-process chat loop, which is a client of this server in every sense that
# matters but arrives as a plain function call. Set for the length of one
# dispatch (``_chat``), so every tool gates it the way it gates a remote client
# instead of letting it through as "no identity".
_local_identity: contextvars.ContextVar = contextvars.ContextVar(
    "biopb_local_identity", default=None
)


def _client_identity():
    """``(id, label)`` for the client behind this call, or ``(None, "")``.

    The streamable-http transport mints a per-connection ``mcp-session-id``, so
    two clients reaching one session child are distinguishable even though the
    tool surface itself is stateless — this is the id the kernel's one-agent
    claim is keyed on (``_jobs.submit``). ``clientInfo.name`` from the initialize
    handshake rides along as a label, purely so a refusal can name who holds the
    kernel.

    Read through ``mcp.get_context()`` rather than a ``Context`` tool parameter:
    the parameter form is excluded from the advertised input schema, but it also
    makes the function uncallable without one, and every in-process caller (the
    tests today, an in-process chat loop later) has no request at all. Outside a
    request this yields no identity, which ``submit`` reads as "nothing to claim
    with" and lets through -- unless an in-process caller has announced itself
    through :data:`_local_identity`, which takes precedence over the request
    because it *is* the caller; a loop dispatching tools has no request of its
    own to be found.
    """
    local = _local_identity.get()
    if local is not None:
        return local
    try:
        rc = mcp.get_context().request_context
    except Exception:  # noqa: BLE001 - no request, or an SDK shape we don't know
        return None, ""
    request = getattr(rc, "request", None)
    ident = request.headers.get("mcp-session-id") if request is not None else None
    session = getattr(rc, "session", None)
    if not ident and session is not None:
        # A client that negotiated no transport session still gets one identity
        # per connection: the ServerSession object is per-connection and lives
        # as long as it does, which is all the claim needs.
        ident = f"conn-{id(session):x}"
    params = getattr(session, "client_params", None)
    label = getattr(getattr(params, "clientInfo", None), "name", "") or ""
    return ident, label


def _foreign_digest(host) -> list:
    """The cells run by another writer that the agent has not been told about,
    or ``[]``.

    A pure read — see :func:`_ack_foreign_digest` for why the ack is a second call.
    Auxiliary, like the window-liveness probe: a kernel that answers with
    anything but the expected list yields no digest rather than breaking the
    result the agent actually asked for.
    """
    digest, _res, _w = _run_job_call(host, "foreign_digest()")
    if not digest or not isinstance(digest, list):
        return []
    if not all(isinstance(d, dict) and "job_id" in d for d in digest):
        return []
    return digest


def _ack_foreign_digest(host, digest, writer=None) -> None:
    """Retire the *terminal* entries of *digest*, once the note carrying them is
    on its way back to the agent.

    Split from the read because acking inside it consumed notices that were
    never delivered: ``execute_interactive`` sends the request before it starts
    its timeout clock, so a probe that times out is still queued at the kernel
    and runs when the main thread frees up — setting the flag for a note nobody
    received. Acking only after this process has parsed a reply keeps the
    guarantee that a notice is deferred, never dropped.

    Running entries are excluded here rather than in the kernel: they were
    reported as ``running``, which is not the final status the agent is promised,
    so they must stay pending even if they have finished since.

    *writer* is the asking client, passed through so the kernel can refuse an ack
    from a client that does not hold it: a second client's ``poll_job`` may
    *read* the digest, but discharging a notice the holder has not received
    would defeat the exactly-once promise this split exists to keep.
    """
    ids = [d["job_id"] for d in digest if d.get("status") != "running"]
    if ids:
        _run_job_call(
            host,
            "ack_foreign_digest(" + repr(ids) + ", writer=" + repr(writer) + ")",
        )


def _render_foreign_note(digest) -> str:
    """The digest as a line appended to an agent-facing result, or ``""``.

    The agent is not the only writer of this namespace: a person can run code
    from the observe page, through the same job runner (``docs/user-console.md``).
    That leaves the agent's picture of the namespace stale with nothing in its
    own results to say so — hence this note, appended at the same seam as
    ``_window_note``, which is how every other user-attributed fact already
    reaches the agent (``cancel_reason``, ``teardown_reason``).

    Deliberately says *that* something changed, not *what*: the agent is told to
    re-verify, which is cheap and cannot go stale itself. It names **no** job id
    in the instruction either — pointing at one of several invites an agent to
    read that one, call the notice discharged, and never see the rest, which it
    will not be offered again.
    """
    if not digest:
        return ""
    # Older kernels' digest entries carry no origin; they could only ever have
    # been user cells, so read a missing one as "user" rather than dropping the
    # attribution.
    origins = {(d.get("origin") or "user") for d in digest}
    if origins == {"user"}:
        who = "The user"
        listed = ", ".join(f"{d['job_id']} ({d.get('status')})" for d in digest)
    else:
        who = "Another writer"
        listed = ", ".join(
            f"{d['job_id']} ({d.get('status')}, {d.get('origin') or 'user'})"
            for d in digest
        )
    return (
        f"\n\nⓘ {who} ran code in this kernel: "
        f"{listed}. A finished cell is reported once; a running one repeats "
        "until it ends, so a repeat is not a new cell. Read them with poll_job. "
        "Variables and layers may have changed — re-check with dir() / "
        "viewer.layers rather than trusting what you last saw."
    )


def _foreign_activity_note(host) -> str:
    """Read, render, and retire the activity notice, in that order.

    Retiring is the holder's alone (see :func:`_ack_foreign_digest`): a second
    client reaching a read-only tool still gets shown what ran, but does not
    consume the notice out from under the agent actually working here.
    """
    digest = _foreign_digest(host)
    note = _render_foreign_note(digest)
    if note:
        _ack_foreign_digest(host, digest, _client_identity()[0])
    return note


def _format_job_status(snap: dict) -> str:
    """Render a job snapshot (poll_job output)."""
    job_id = snap.get("job_id", "?")
    status = snap.get("status")
    header = f"{job_id}: {status} ({snap.get('elapsed', '?')}s)"
    body = _format_execute_result(snap)
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
    (``services.skills_enabled``): with it off there is no ``find_skills`` to
    return a ``checklist:``, so the section would document an unreachable
    tool -- the same gate the handshake instructions use.
    """
    if _skills_enabled:
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
    """Full workflow body for a curated skill; discover ids with `find_skills`.

    The catalog (metadata) is served separately via the `find_skills` tool; this
    resource lazily fetches one skill's markdown body, verifies it against the
    catalog checksum, and caches it. Fail-open: returns a short explanatory
    string rather than erroring when a skill is unknown or unreachable.
    """
    return _skills.get_skill_body(skill_id)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def find_skills(keywords: list[str] | None = None) -> list:
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

    Fail-open: returns an empty list (never errors) when the catalog is
    unreachable and nothing is cached or bundled.
    """
    return _skills.find_skills(keywords or ())


@mcp.tool()
def take_screenshot(canvas_only: bool = True) -> list:
    """Capture the napari viewer as a PNG image.

    Args:
        canvas_only: If True, capture only the canvas area. If False,
            capture the entire viewer window.

    Returns a PNG screenshot as an image content block.
    """
    host = _kernel_host
    if host is None:
        return [TextContent(type="text", text="Kernel host not initialized")]

    snippet = _SCREENSHOT_SNIPPET.format(canvas_only=bool(canvas_only))
    res = host.execute(snippet)
    if _extract_delimited(res.get("stdout", ""), _WINDOW_CLOSED_DELIM) is not None:
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
    data = _extract_delimited(res.get("stdout", ""), _PNG_DELIM)
    if data is None:
        detail = res.get("error_text") or res.get("stdout") or res.get("status")
        return [TextContent(type="text", text=f"Screenshot failed: {detail}")]
    return [ImageContent(type="image", mimeType="image/png", data=data)]


@mcp.tool()
def execute_code(python_code: str, intent: str = "") -> str:
    """Execute Python code in the napari kernel.

    intent: one short sentence on *why* you are running this cell — the goal you
    are pursuing for the user, not a restatement of what the code does. It is
    recorded with the job and written into the session's notebook export, which
    is otherwise a log of code with no record of what anyone was trying to
    achieve. Leave it empty rather than padding it.

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
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    writer, writer_label = _client_identity()

    # Read once at entry, append to whichever path returns below.
    digest = _foreign_digest(host)
    foreign_note = _render_foreign_note(digest)
    if foreign_note:
        _ack_foreign_digest(host, digest, writer)

    # Before the call, not after: a lost reply must not leave the kernel claimed
    # while this process still reads as unclaimed. See _claimed_by.
    _presume_claim(writer)
    submitted, res, window_alive = _run_job_call(
        host,
        "submit("
        + repr(python_code)
        + ", intent="
        + repr(intent)
        + ", writer="
        + repr(writer)
        + ", writer_label="
        + repr(writer_label)
        + ")",
    )
    if submitted is None:
        return _format_execute_result(res) + foreign_note
    if submitted.get("error") == "not_owner":
        # The authority speaking: whatever this process presumed above, the
        # kernel just named the real holder.
        _note_claim(submitted.get("owner_id"))
        held_by = submitted.get("owner") or ""
        held_by = f" ({held_by})" if held_by else ""
        return _NOT_OWNER_MSG.format(held_by=held_by) + foreign_note
    # Anything the kernel did not refuse came from the holder, "busy" included:
    # submit() decides the claim before it looks at what is running.
    _note_claim(writer)
    if submitted.get("error") == "busy":
        running = submitted.get("running_job_id")
        # Whose job is running decides the advice. Telling the agent to
        # "stop it with interrupt_kernel" while *someone else* is running a cell
        # would have it kill their work; interrupt_kernel refuses that anyway
        # (_jobs.interrupt_current), so the wording must not send it there.
        running_origin = submitted.get("running_job_origin")
        if running_origin and running_origin != "agent":
            # A running foreign job stays in the digest by design, so the note is
            # about to report the very job this branch is reporting. Drop it
            # when that is *all* it says; keep it when other cells also finished,
            # since those were acked above and will not be offered again.
            if [d.get("job_id") for d in digest] == [running]:
                foreign_note = ""
            who = "The user" if running_origin == "user" else "Another writer"
            return (
                f"{who} is running a cell ({running}) in this kernel. Only one "
                f"job runs at a time — wait for it and poll_job('{running}'); do "
                "not interrupt it." + foreign_note
            )
        return (
            f"A job ({running}) is already running. Poll it with "
            f"poll_job('{running}'), or stop it with interrupt_kernel / "
            "restart_kernel before starting another." + foreign_note
        )

    job_id = submitted["job_id"]
    deadline = time.monotonic() + _promote_after
    snap = submitted
    while time.monotonic() < deadline:
        time.sleep(0.4)
        snap, res, window_alive = _run_job_call(host, "poll(" + repr(job_id) + ")")
        if snap is None:
            return _format_execute_result(res) + foreign_note
        if snap.get("status") != "running":
            # terminal: inline result
            return (
                _format_execute_result(snap) + _window_note(window_alive) + foreign_note
            )

    # Still running after promote_after: hand back a job handle.
    partial = snap.get("stdout", "") if snap else ""
    return (
        f"Job {job_id} is still running after {_promote_after:.0f}s. "
        f"Poll it with poll_job('{job_id}'); watch with take_screenshot / "
        f"server_status; stop with interrupt_kernel or restart_kernel.\n"
        "Partial output:\n"
        + (partial or "(none yet)")
        + _window_note(window_alive)
        + foreign_note
    )


@mcp.tool()
def poll_job(job_id: str) -> str:
    """Get the status and output of a job started by execute_code.

    Returns the job's status (running/ok/error/interrupted), elapsed time, and
    output so far (full output once terminal). Job records persist until the
    kernel is restarted (older terminal jobs are eventually evicted).
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    foreign_note = _foreign_activity_note(host)
    snap, res, window_alive = _run_job_call(host, "poll(" + repr(job_id) + ")")
    if snap is None:
        return _format_execute_result(res) + foreign_note
    if snap.get("status") == "unknown":
        return f"No such job '{job_id}'." + foreign_note
    note = _window_note(window_alive) if snap.get("status") != "running" else ""
    return _format_job_status(snap) + note + foreign_note


@mcp.tool()
def inspect_object(object_path: str) -> str:
    """Inspect a live object in the napari kernel namespace.

    Returns the type, docstring, and public methods/attributes.
    Example: inspect_object("viewer.layers") or inspect_object("viewer.camera")
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

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
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    writer, _label = _client_identity()
    data, res, _w = _run_job_call(
        host, "interrupt_current(requester='agent', writer=" + repr(writer) + ")"
    )
    if data is None:
        return _format_execute_result(res)
    if data.get("refused") == "not_owner":
        return _NOT_OWNER_MSG.format(held_by="")
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
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    result = host.ensure_started()
    if result.get("state") == "ready":
        return (
            "Kernel ready. The napari viewer, dask, and tensor client are up; "
            "use execute_code / take_screenshot now."
        )
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
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    # Gated like every other state change, and this is the sharpest of them: a
    # restart discards the holder's whole session. Decided against the local
    # mirror (_claimed_by) rather than a round trip to the kernel, so a kernel
    # too busy to answer cannot be mistaken for an unclaimed one.
    writer, _label = _client_identity()
    if _claimed_by is not None and writer is not None and writer != _claimed_by:
        return _NOT_OWNER_MSG.format(held_by="")
    try:
        host.restart()
    except Exception as exc:
        return f"Kernel restart failed: {exc}"
    clear_claim()  # a fresh kernel is unclaimed until someone runs code in it
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

    host = _kernel_host

    cpu_percent = psutil.cpu_percent(interval=0.1)
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
        f"  log_file: {_session_log_path or 'stdout (not file-logged)'}",
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
    return "\n".join(lines) + _foreign_activity_note(host)


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
    mcp.settings.transport_security = build_transport_security(
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
