"""The server application object, and the process-wide state the launcher sets.

Runs **in the MCP server process**. This module owns the things every surface
needs and none of them should own: the ``FastMCP`` instance the tools decorate,
the transport-security allowlists that guard every route on its port, the one
:class:`~biopb_mcp.mcp._kernel.KernelHost` the tools dispatch to, and the
handshake ``instructions``.

It is deliberately the bottom of the package's import graph. ``_server`` (the
tool surface), ``_observe`` and ``_chat_api`` (the two HTTP surfaces) and
``_http`` (their shared guard) all need ``mcp`` and the kernel host, and before
this module existed they got them by importing ``_server`` -- which imports
``_observe`` in turn, so the two were a cycle. Nothing here imports any of them.

``__main__`` is the only caller of the setters below; it configures this module
once, before serving.
"""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

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
# that switches skills off neither points the agent at `list_skills` (which would
# return nothing) nor prompts it to author skills — set_skills_enabled owns the
# field.
_SKILLS_INSTRUCTIONS = (
    "At the start of a task, call `list_skills` to check for a curated workflow "
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


def set_kernel_host(host: KernelHost):
    """Register the kernel host the tools dispatch to.

    A different host is a different kernel, so the mirrored one-agent claim goes
    with the old one rather than being inherited by the new.
    """
    # Imported at call time, not at module scope: `_writers` reads `mcp` from
    # here, and this module is meant to stay the bottom of the import graph.
    from ._writers import clear_claim

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
    directive, so the agent is never pointed at ``list_skills`` when it would
    return nothing."""
    global _skills_enabled
    _skills_enabled = bool(enabled)
    _recompose_instructions()


def _require_kernel_host():
    """The kernel host, or the agent-facing refusal to return instead.

    Returns ``(host, None)`` or ``(None, message)`` -- the shape
    ``_http.require_host`` already uses for the HTTP side, so the one
    precondition every kernel-touching tool has is spelled once rather than at
    each entry point (where three wordings had already appeared).
    """
    if _kernel_host is None:
        return None, "Error: kernel host not initialized"
    return _kernel_host, None
