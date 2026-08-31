"""The kernel round trip: calling into the in-kernel job runner, and reading back.

Runs **in the MCP server process**. Every tool call and every observe/chat poll
crosses this seam, and it is the same crossing each time: build a call
expression, wrap it in a snippet that prints a delimited JSON payload, run it
through the kernel host, and parse what comes back.

There is no marshaller on this hop -- the kernel execs source text -- so
:func:`_call_expr` is what makes passing values safe: ``repr`` on every argument
is a property of one function here rather than a convention each call site has
to remember.

A leaf module: it knows the shape of the hop and nothing about who is making it
(no claim, no digest, no tool surface).
"""

import json
import logging

logger = logging.getLogger(__name__)

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


def _call_expr(name: str, *args, **kwargs) -> str:
    """``name(*args, **kwargs)`` as Python source, every value embedded by ``repr``.

    The kernel hop has no marshaller -- a call is source text the kernel execs --
    so somebody has to turn values into literals. Doing it here rather than at
    each call site is what makes that safe: ``repr`` on every argument is a
    property of this one function instead of a convention twelve places have to
    remember, and adding an argument to a ``_jobs`` entry point stops being a
    string-concatenation edit.
    """
    parts = [repr(a) for a in args]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def _job_snippet(call: str) -> str:
    """Build a snippet that prints ``_jobs.<call>``'s result as delimited JSON.

    ``call`` is a fully-formed call expression, normally from :func:`_call_expr`
    (agent code is RCE by design, but embedding via ``repr`` keeps the payload a
    valid literal regardless of its contents).

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


def _run_job_call(host, name: str, *args, **kwargs):
    """Call ``_jobs.<name>(*args, **kwargs)`` in the kernel.

    Arguments are passed as values, not as pre-built source: :func:`_call_expr`
    reprs them. Returns ``(result, raw_result, window_alive)`` where ``result``
    is the parsed return value (None if the snippet failed) and ``window_alive``
    is the viewer-window liveness flag carried in the same payload (None when
    unknown, e.g. the snippet did not run cleanly).
    """
    res = host.execute(_job_snippet(_call_expr(name, *args, **kwargs)))
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
