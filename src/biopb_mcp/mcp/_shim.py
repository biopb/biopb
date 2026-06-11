"""stdio bridge ("shim") to the biopb-mcp http daemon.

``biopb-mcp --transport stdio`` no longer serves MCP over fd 0/1 from the
heavy launcher process. Instead the launcher runs this module, which

1. ensures the http daemon is listening on the configured loopback port,
   spawning it detached if nothing is (``ensure_daemon``), then
2. bridges stdio JSON-RPC <-> the daemon's streamable-http endpoint
   (``run_bridge``) until the client closes stdin.

This is the daemon-migration Direction 1 shape (docs/daemon-migration.md):
the process that owns fd 1 as a protocol channel imports nothing that could
write to stdout (no Qt, dask, uvicorn, or kernel — only the mcp SDK), so the
fd-1 corruption class is structurally impossible here. The heavy process
serves http only, outlives any one client, and is shared by all of them.

The bridge is vendored rather than delegated to ``mcp-proxy`` per the vetting
report (docs/mcp-proxy-vet.md): mcp-proxy drops the initialize
``instructions`` field that carries biopb-mcp's operation guardrails, has no
lifetime guard when the daemon dies, and floats its dependencies. Here the
remote's initialize result — capabilities, serverInfo, and ``instructions`` —
is replayed to the stdio client verbatim, and any bridge failure exits the
shim so the client sees EOF instead of a hung proxy.
"""

import logging
import os
import socket
import subprocess
import sys
import time

import anyio
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel.server import Server, request_ctx
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)

# How long ensure_daemon waits for a spawned daemon to start listening.
# Dominated by the http stack's import time (FastMCP + uvicorn + the tensor
# client), not the kernel — the kernel starts later, on the first
# `start_kernel` tool call.
DAEMON_START_TIMEOUT = 60.0
_PROBE_INTERVAL = 0.25


def _port_listening(port, timeout=0.5):
    """Whether something accepts TCP connections on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _daemon_command(port):
    """The argv that launches the http daemon this shim fronts.

    A frozen (PyInstaller) build has no importable module tree behind
    ``sys.executable``, but its entry binary *is* the launcher, so plain args
    suffice; a normal install re-enters via ``-m``.
    """
    if getattr(sys, "frozen", False):
        cmd = [sys.executable]
    else:
        cmd = [sys.executable, "-m", "biopb_mcp.mcp"]
    return [*cmd, "--transport", "http", "--port", str(port)]


def _open_daemon_log(config):
    """Open the file the detached daemon's stdout/stderr is sent to.

    Reuses ``mcp.transport.kernel_log`` (empty -> <log dir>/kernel.log): the
    daemon's fds are inherited by its child kernel, so this file carries the
    same native Qt/GL/dask/gRPC output the key always named — plus the
    daemon's own logs. Binary, append, unbuffered, for the same reason the
    kernel redirection always was: native writers emit arbitrary bytes. On
    failure, falls back to the shim's stderr buffer so the daemon still
    starts (its output then interleaves with the shim's logging — harmless,
    stderr is not a protocol channel).
    """
    from .._config import get_log_dir, get_setting

    path = get_setting(config, "mcp.transport.kernel_log") or str(
        get_log_dir() / "kernel.log"
    )
    try:
        return open(path, "ab", buffering=0)
    except OSError:
        logger.warning(
            "Could not open daemon log %s; routing daemon output to stderr",
            path,
        )
        return getattr(sys.stderr, "buffer", sys.stderr)


def ensure_daemon(config, port, timeout=DAEMON_START_TIMEOUT):
    """Make sure the http daemon is listening on ``port``; spawn it if not.

    Returns True if this call spawned the daemon, False if one was already
    listening. Concurrent shims race benignly: each spawns a daemon, the
    kernel's port bind picks one winner, every loser process exits on
    EADDRINUSE, and all shims converge on whoever is listening — so this only
    ever polls the port, never tracks the child it spawned.

    Raises TimeoutError if nothing is listening after ``timeout`` seconds
    (e.g. the daemon crashed on boot; its log has the trace).
    """
    if _port_listening(port):
        return False

    log = _open_daemon_log(config)
    daemon_log_name = getattr(log, "name", "stderr")
    cmd = _daemon_command(port)
    logger.info("No daemon on 127.0.0.1:%d; spawning: %s", port, cmd)
    # Detach fully: new session/process group, no inherited stdio. The daemon
    # must outlive this shim (and its client), which is the point of the
    # daemon model.
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        close_fds=True,
        **popen_kwargs,
    )
    if log is not getattr(sys.stderr, "buffer", sys.stderr):
        log.close()  # the child holds its own duplicate of the fd

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_listening(port):
            return True
        time.sleep(_PROBE_INTERVAL)
    raise TimeoutError(
        f"biopb-mcp daemon did not start listening on 127.0.0.1:{port} "
        f"within {timeout:.0f}s; check the daemon log ({daemon_log_name})"
    )


def build_proxy(remote):
    """Build the stdio-facing MCP server that forwards every request to
    ``remote`` (a ClientSession connected to the daemon).

    Handlers are registered unconditionally and the *remote's* capabilities
    are advertised verbatim (see ``run_bridge``), so the bridge never narrows
    what the daemon offers; a client simply won't call what the capabilities
    don't advertise. Server->client traffic other than tool-call progress
    (sampling, elicitation, list_changed) is not forwarded — biopb-mcp emits
    none of it (the on-demand kernel design avoids dynamic tool lists on
    purpose), and a future feature that needs it must extend this bridge.
    """
    app = Server(name="biopb-mcp")

    async def _list_tools(_):
        return types.ServerResult(await remote.list_tools())

    app.request_handlers[types.ListToolsRequest] = _list_tools

    async def _call_tool(req):
        meta = dict(req.params.meta) if req.params.meta else None
        progress_token = meta.get("progressToken") if meta else None
        progress_callback = None
        if progress_token is not None:
            ctx = request_ctx.get()

            async def _forward_progress(progress, total, message):
                await ctx.session.send_progress_notification(
                    progress_token=progress_token,
                    progress=progress,
                    total=total,
                    message=message,
                    related_request_id=str(ctx.request_id),
                )

            progress_callback = _forward_progress
        try:
            result = await remote.call_tool(
                req.params.name,
                req.params.arguments or {},
                progress_callback=progress_callback,
                meta=meta,
            )
            return types.ServerResult(result)
        except Exception as e:  # surface as a tool error, not a dead bridge
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(e))],
                    isError=True,
                )
            )

    app.request_handlers[types.CallToolRequest] = _call_tool

    async def _list_resources(_):
        return types.ServerResult(await remote.list_resources())

    app.request_handlers[types.ListResourcesRequest] = _list_resources

    async def _list_resource_templates(_):
        return types.ServerResult(await remote.list_resource_templates())

    app.request_handlers[types.ListResourceTemplatesRequest] = (
        _list_resource_templates
    )

    async def _read_resource(req):
        return types.ServerResult(await remote.read_resource(req.params.uri))

    app.request_handlers[types.ReadResourceRequest] = _read_resource

    async def _list_prompts(_):
        return types.ServerResult(await remote.list_prompts())

    app.request_handlers[types.ListPromptsRequest] = _list_prompts

    async def _get_prompt(req):
        return types.ServerResult(
            await remote.get_prompt(req.params.name, req.params.arguments)
        )

    app.request_handlers[types.GetPromptRequest] = _get_prompt

    async def _complete(req):
        return types.ServerResult(
            await remote.complete(
                req.params.ref, req.params.argument.model_dump()
            )
        )

    app.request_handlers[types.CompleteRequest] = _complete

    async def _set_logging_level(req):
        await remote.set_logging_level(req.params.level)
        return types.ServerResult(types.EmptyResult())

    app.request_handlers[types.SetLevelRequest] = _set_logging_level

    async def _forward_progress_notification(req):
        await remote.send_progress_notification(
            req.params.progressToken, req.params.progress, req.params.total
        )

    app.notification_handlers[types.ProgressNotification] = (
        _forward_progress_notification
    )

    return app


def replay_init_options(init):
    """Map the daemon's InitializeResult onto the options the bridge serves.

    Verbatim replay — most importantly ``instructions``, the handshake-time
    carrier for the operation guardrails and the headless notice (the field
    mcp-proxy drops, per docs/mcp-proxy-vet.md), and the remote's capability
    set rather than one recomputed from the bridge's own handlers.
    """
    return InitializationOptions(
        server_name=init.serverInfo.name,
        server_version=init.serverInfo.version,
        capabilities=init.capabilities,
        instructions=init.instructions,
        website_url=init.serverInfo.websiteUrl,
        icons=init.serverInfo.icons,
    )


async def _bridge(url):
    async with (
        streamablehttp_client(url=url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        app = build_proxy(session)
        options = replay_init_options(init)
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, options)


def run_bridge(url):
    """Bridge stdio <-> the daemon at ``url`` until the client closes stdin.

    Any failure (daemon death mid-session included) propagates out, so the
    shim process exits and the client sees EOF — never a hung bridge.
    """
    anyio.run(_bridge, url)


def serve(config, port):
    """Launcher entry point for ``--transport stdio``: ensure + bridge."""
    logger.warning(
        "stdio is served via a bridge to the local biopb-mcp http daemon "
        "(spawned on demand, shared across clients, survives this client). "
        "Native http is recommended where the client supports it: "
        "`claude mcp add --transport http biopb http://127.0.0.1:%d/mcp`.",
        port,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    spawned = ensure_daemon(config, port)
    logger.info(
        "Bridging stdio to %s (%s)",
        url,
        "daemon spawned" if spawned else "daemon already running",
    )
    run_bridge(url)
