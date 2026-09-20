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

# Whether the knowledge store's procedure docs are served (mirrors
# `services.docs_enabled`, on by default). Set by the launcher
# (set_docs_enabled); the store reads the same setting, and the handshake drops
# the authoring directive with it. test_mcp_server pins this literal to the
# config default so the two can't drift.
_docs_enabled: bool = True

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
    "First action of every session: call `start_kernel`. It brings up the kernel, "
    "dask, the tensor client and -- where the session has a display -- a napari "
    "window, and blocks until they are ready; nothing "
    "auto-starts, and every other kernel tool fails until it returns. It also "
    "rebuilds a kernel that never started, died, or was torn down by the user "
    "closing the viewer window -- but a kernel that is already up it leaves "
    "alone, so it cannot clear a wedged one: that is `interrupt_kernel` (stop "
    "the job) or `restart_kernel` (hard-restart). A user asking to start, open, "
    "or launch biopb or napari is asking for this tool.\n"
    "\n"
    "This biopb-mcp session drives a child IPython kernel over bioimage data; "
    "`execute_code` runs arbitrary Python in it. There are two ways to show the "
    "user an image and a session need not have both -- a napari window, and the "
    "browser page the control serves -- so `server_status` is what says which, "
    "and nothing should assume a window exists. The index below "
    "lists the docs; read one with `read_doc(id)`, and read the reference docs "
    "it lists before non-trivial work.\n"
    "\n"
    "Operation guardrails (apply on every turn):\n"
    "- Use data from `client`, or from `viewer`'s layers where the session has a "
    "window; avoid the filesystem unless the user explicitly asks.\n"
    '- Browse the catalog with `client.query_sources(sql, format="pandas")` '
    "(server-side DuckDB, the only browse surface); the `sources` columns are source_id, "
    "source_url, source_type, dtype, indexed_at, metadata_json, "
    "shape_summary, is_resolved, and `tensors` (a LIST of "
    "STRUCT(array_id, dim_labels, shape, dtype), one per tensor -- "
    "query per-tensor with UNNEST(tensors) or list_filter; the scalar "
    "dtype/shape_summary only describe tensors[0]). Unresolved (cloud / "
    "synced-folder) sources "
    "have NULL dtype/shape_summary, so a predicate like `WHERE dtype='uint8'` "
    "silently drops them; filter on `is_resolved` to opt them in/out on "
    "purpose (`WHERE NOT is_resolved` finds what hasn't been resolved yet). "
    "Resolved is not the same as local: assume a cloud or synced-folder source "
    "may not have its bytes on the serving machine, so the first read can be "
    "slow or fail offline. Plan for that -- warn the user before a long read "
    "rather than after it.\n"
    "- Prefer lazy dask operations; only `.compute()` the final result.\n"
    "- Show intermediate results for the user to validate at each step: a layer "
    "on `viewer` where the session has a window, otherwise upload the result and "
    'give them a web-viewer link (`read_doc("web-viewer")`).\n'
    "- Do not assume — ask the user to clarify uncertainties; they know the "
    "data better than you do."
)

# Appended to _BASE_INSTRUCTIONS only when procedure docs are served
# (`services.docs_enabled`, on by default), so an install that switches them off
# is told neither to follow one nor to write one.
_AUTHORING_INSTRUCTIONS = (
    "A procedure doc opens with a Requirements line; resolve it against "
    "`server_status` before starting, and treat a gap as something to name and "
    'work around rather than a reason to stop (`read_doc("requirements")`). '
    "Never substitute silently -- the user cannot judge a result whose method "
    "they were not told changed.\n"
    "\n"
    "After accomplishing a task worth repeating, ask the user whether it should "
    "become a doc, and write it with `write_doc`.\n"
    "\n"
    "Destructive steps always ask first, whatever a doc says: restarting the "
    "kernel, interrupting a running job, overwriting a layer, or writing files."
)

# The header the rendered index is carried under. The index is inlined here
# rather than fetched on request because every prompted hop loses agents
# (biopb/biopb#894 is the record of agents missing the *first* hop).
_INDEX_HEADER = (
    "The doc index follows. It is itself doc `index`: re-read it with "
    '`read_doc("index")` and edit it with `write_doc`.'
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
# always-on base guidance now so it is present even if set_docs_enabled is never
# called (e.g. tests, or a standalone import). No index here: composing one at
# import would read the config tree before the launcher has configured it.
mcp._mcp_server.instructions = _BASE_INSTRUCTIONS


def _recompose_per_session(build=mcp._mcp_server.create_initialization_options):
    """Recompose ``instructions`` each time a session is initialized.

    The index is a file the agent edits, and a long-lived HTTP server outlives
    many sessions, so composing once at launch would hand later sessions an
    index that has moved. ``create_initialization_options`` is the one call the
    SDK makes per session that reads ``instructions``.
    """

    def create_initialization_options(*args, **kwargs):
        _recompose_instructions()
        return build(*args, **kwargs)

    return create_initialization_options


mcp._mcp_server.create_initialization_options = _recompose_per_session()


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


def _compose_instructions() -> str:
    """The handshake text: the base guidance, the rendered index, the rest.

    Composed from :data:`_BASE_INSTRUCTIONS` every time rather than appended to,
    so switching a dimension back off cannot leave a stale fragment behind.
    """
    parts = [_BASE_INSTRUCTIONS]
    try:
        from ._docs import render_index

        parts.append(f"{_INDEX_HEADER}\n\n{render_index()}")
    except Exception:  # pragma: no cover - the store is fail-open everywhere else
        logger.debug(
            "docs: could not render the index for the handshake", exc_info=True
        )
    if _docs_enabled:
        parts.append(_AUTHORING_INSTRUCTIONS)
    return "\n\n".join(parts)


def _recompose_instructions():
    """Refresh the ``instructions`` the low-level Server hands out."""
    mcp._mcp_server.instructions = _compose_instructions()


def set_docs_enabled(enabled: bool):
    """Serve (or withhold) the knowledge store's procedure docs.

    Mirrors ``services.docs_enabled`` into this process so the handshake and
    ``server_status`` agree with what ``read_doc`` will actually return.
    """
    global _docs_enabled
    _docs_enabled = bool(enabled)
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
