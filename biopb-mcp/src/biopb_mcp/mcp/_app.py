"""The server application object, and the process-wide state the launcher sets.

Runs **in the MCP server process**. This module owns the things every surface
needs and none of them should own: the ``FastMCP`` instance the tools decorate,
the transport-security allowlists that guard every route on its port, and the
one :class:`~biopb_mcp.mcp._kernel.KernelHost` the tools dispatch to. The
handshake ``instructions`` are composed in ``_instructions``, which the stdio
shim also reads.

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

from ._instructions import BASE_INSTRUCTIONS, compose as compose_instructions
from ._kernel import KernelHost

logger = logging.getLogger(__name__)

_kernel_host: KernelHost | None = None

# Seconds execute_code waits for a job to finish before returning a job handle
# instead of an inline result (set from config by the launcher).
_promote_after: float = 10.0

# This process's logfile path (set by the launcher), surfaced by server_status so
# an agent can find its own log. None when output goes to a terminal (foreground
# `--transport http` / `biopb mcp view`) rather than a file.
_session_log_path: str | None = None

# This session's own /mcp url (set by the launcher when it binds a dynamic
# port), for a client it starts itself -- the chat pane's ACP harness.
_mcp_url: str | None = None

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
# always-on base guidance now so it is present even if no session is ever
# initialized (e.g. tests, or a standalone import). No index here: composing one
# at import would read the config tree before the launcher has configured it.
mcp._mcp_server.instructions = BASE_INSTRUCTIONS


# The index is a file the agent edits, and a long-lived HTTP server outlives
# many sessions, so composing once at launch would hand later sessions an index
# that has moved. `create_initialization_options` is the one call the SDK makes
# per session that reads `instructions`.
_create_initialization_options = mcp._mcp_server.create_initialization_options


def _recompose_per_session(*args, **kwargs):
    _recompose_instructions()
    return _create_initialization_options(*args, **kwargs)


mcp._mcp_server.create_initialization_options = _recompose_per_session


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


def set_mcp_url(url: str | None):
    """Record where this session serves ``/mcp``."""
    global _mcp_url
    _mcp_url = url


def set_session_log_path(path: str | None):
    """Record this process's logfile path for server_status to report."""
    global _session_log_path
    _session_log_path = path


def _recompose_instructions():
    """Refresh the ``instructions`` the low-level Server hands out."""
    mcp._mcp_server.instructions = compose_instructions()


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
