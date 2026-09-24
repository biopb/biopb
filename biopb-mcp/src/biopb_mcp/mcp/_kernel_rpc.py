"""The tools' kernel round trips: running a snippet, and reading its reply.

Runs **in the MCP server process**. A leaf module: the shape of the hop and of
its reply text (the screenshot and closed-window sentinels, the formatting of
a result), and nothing about who is making it (no claim, no digest, no tool
surface).
"""

import asyncio

_PNG_DELIM = "<<PNG_B64>>"

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


async def _execute(host, code: str, timeout=None):
    """``host.execute`` off the event loop, for the tools' snippets
    (screenshot, inspect, status).

    The round trip blocks until the kernel replies, which is as long as its
    main thread is busy with something else. Every surface in this process
    shares one event loop -- ``/mcp``, the observe page, the chat turn -- so a
    round trip made on it is not one caller waiting but all of them. The
    kernel host takes calls from any thread and lets them overlap, so the
    thread is free of anything but the wait.
    """
    return await asyncio.to_thread(host.execute, code, timeout)


def _window_note(window_alive) -> str:
    """Closed-window warning to append when a result returns with no viewer.

    ``window_alive`` is None when liveness is unknown -> no note.
    """
    if window_alive is False:
        return _WINDOW_CLOSED_NOTE
    return ""
