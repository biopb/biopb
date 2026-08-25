"""In-process chat loop: a built-in agent for users without an MCP harness.

Runs in the **session child** — the process that owns the ``KernelHost`` and
serves ``/mcp`` — for the same reason the dask cluster does: it must outlive
kernel restarts, and a widget inside the kernel could be destroyed by its own
agent. See ``docs/chat-client-evaluation.md`` for the argument.

Design notes
------------
* **One conversation per session, shared by every view.** The thread is server
  state, so two browser windows are two views of one writer rather than two
  writers, and a page reload costs nothing. It follows the shape the observe
  page already renders for job history: a list nobody's client owns.
* **Tools are Python calls, not wire calls.** Schemas come from the same
  ``FastMCP`` registry the MCP clients read, so there is one definition and no
  drift; ``_tool_manager.call_tool`` puts us one layer below the low-level
  server, which is why :func:`_dispatch` collapses its two return shapes the way
  ``lowlevel/server.py`` does.
* **``execute_code`` is the one tool not taken as-is.** Its ``promote_after``
  window is a wire-latency optimization — a poll is a round trip over HTTP and a
  function call here — and for a chat surface it is a cost: the model blocks and
  the stream says nothing. :func:`_run_code` submits with no window and reports
  partial output as it arrives.
* **The loop is a client like any other.** It announces itself through
  ``_server._local_identity`` for the length of a dispatch, so the kernel's
  one-agent claim covers it: a chat session and an attached MCP harness are
  mutually exclusive, and neither can quietly write into the other's namespace.
* **The model is injected.** Nothing here knows a provider. The caller passes an
  async ``model(messages, tools) -> assistant message``, which keeps the loop
  testable with no key and no network, and keeps provider choice out of it.
"""

import json
import time

from mcp.types import ImageContent, TextContent

from . import _server

#: This loop's client id, for the kernel's one-agent claim (``_jobs.submit``).
#: A fixed string rather than a per-view id, because every view drives the one
#: conversation: the loop is a single writer no matter how many windows are open.
WRITER_ID = "biopb-chat"
WRITER_LABEL = "chat"

#: Cap on tool-call rounds within one turn. A model that keeps calling tools
#: without answering is not converging, and the user is sitting there watching.
_MAX_TOOL_ROUNDS = 12

#: How often a running job's output is re-read while the turn waits on it.
#: In-process this is a function call, so it is set for a responsive stream
#: rather than to spare a round trip.
_POLL_INTERVAL = 0.3

# The shared conversation. Module state like ``_jobs``: it belongs to the
# session, survives kernel restarts, and dies with this process.
_messages: list = []
_seq = 0


def reset():
    """Drop the conversation (used by tests and on an explicit new session)."""
    global _seq
    _messages.clear()
    _seq = 0


def _append(role, content, **extra):
    """Record one message and return it.

    Every message gets an id because the views render server state rather than
    their own: an id is what lets a window tell a message it has already drawn
    from one another window just sent.
    """
    global _seq
    _seq += 1
    msg = {"id": f"m-{_seq}", "role": role, "content": content, "ts": time.time()}
    msg.update(extra)
    _messages.append(msg)
    return msg


def history():
    """The conversation as the views render it, oldest first."""
    return list(_messages)


def _last_user_text():
    for msg in reversed(_messages):
        if msg["role"] == "user" and not msg.get("image"):
            return msg["content"]
    return ""


# ---------------------------------------------------------------------------
# The tool surface
# ---------------------------------------------------------------------------


def _clean_schema(schema):
    """The tool's input schema, minus the keys some providers reject.

    ``$schema`` and ``title`` are pydantic's, meaningful to a validator and noise
    to a function-calling API — several reject the payload outright rather than
    ignoring them.
    """
    return {k: v for k, v in (schema or {}).items() if k not in ("$schema", "title")}


async def tool_payload():
    """The function-calling tool list, generated from the live MCP registry.

    Read from ``list_tools()`` rather than declared here: a hand-written copy is
    the one thing that can silently stop matching the tools that actually run.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": _clean_schema(tool.inputSchema),
            },
        }
        for tool in await _server.mcp.list_tools()
    ]


async def _dispatch(name, arguments, on_progress):
    """Run one tool call in-process; return ``(text, [image blocks])``.

    ``call_tool`` returns a bare block list for a tool with no output schema and
    a ``(blocks, structured)`` tuple for one that has it — the layer's declared
    contract, collapsed here the way the low-level server collapses it on its way
    to a ``CallToolResult``. A test pins the shape per tool, so a FastMCP bump
    fails loudly rather than quietly reshaping what the loop receives.
    """
    if name == "execute_code":
        return _run_code(arguments, on_progress), []
    result = await _server.mcp._tool_manager.call_tool(
        name, arguments, convert_result=True
    )
    blocks = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    text = "\n".join(b.text for b in blocks if isinstance(b, TextContent))
    return text, [b for b in blocks if isinstance(b, ImageContent)]


def _run_code(arguments, on_progress):
    """``execute_code`` without the promote window, streaming partial output.

    The tool waits ``promote_after`` seconds before handing back a job handle,
    which saves a round trip for an MCP client and buys a chat surface nothing —
    the model is blocked, so the stream falls silent for exactly as long. Here
    the job is submitted, its output reported as it accumulates, and the turn
    resumes when it ends.

    *intent* prefers what the model supplied — it is closer to the cell than the
    turn is — and falls back to the user's own words, so the notebook export has
    a stated purpose either way.
    """
    host = _server._kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    code = arguments.get("python_code") or ""
    intent = arguments.get("intent") or _last_user_text()

    # Claimed before the call, and mirrored, for the same reason execute_code
    # does it: a lost reply must not leave the kernel held while this process
    # reads as unclaimed.
    _server._presume_claim(WRITER_ID)
    submitted, res, _w = _server._run_job_call(
        host,
        "submit("
        + repr(code)
        + ", origin='chat', intent="
        + repr(intent)
        + ", writer="
        + repr(WRITER_ID)
        + ", writer_label="
        + repr(WRITER_LABEL)
        + ")",
    )
    if submitted is None:
        return _server._format_execute_result(res)
    if submitted.get("error") == "not_owner":
        held = submitted.get("owner") or ""
        _server._note_claim(submitted.get("owner_id"))
        return _server._NOT_OWNER_MSG.format(held_by=f" ({held})" if held else "")
    _server._note_claim(WRITER_ID)
    if submitted.get("error") == "busy":
        return (
            f"A job ({submitted.get('running_job_id')}) is already running in this "
            "kernel; wait for it to finish."
        )

    job_id = submitted["job_id"]
    seen = 0
    while True:
        time.sleep(_POLL_INTERVAL)
        snap, res, _w = _server._run_job_call(host, "poll(" + repr(job_id) + ")")
        if snap is None:
            return _server._format_execute_result(res)
        out = snap.get("stdout") or ""
        if on_progress is not None and len(out) > seen:
            on_progress(out[seen:])
            seen = len(out)
        if snap.get("status") != "running":
            return _server._format_execute_result(snap)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _llm_messages():
    """Project the stored conversation into the provider's message shape.

    Kept separate from :func:`history` because the two audiences want different
    things: the views want ids and timestamps, the model wants neither and does
    want a system message the views should not render.
    """
    out = [{"role": "system", "content": _server.mcp._mcp_server.instructions or ""}]
    for msg in _messages:
        if msg.get("image"):
            # Chat-completions tool messages are plain strings, so an image
            # cannot ride back in a tool result -- it travels as its own user
            # message and the tool result carries a placeholder.
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{msg['mime']};base64,{msg['image']}"
                            },
                        },
                    ],
                }
            )
        elif msg["role"] == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
            )
        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": msg["content"] or None,
                    "tool_calls": msg["tool_calls"],
                }
            )
        else:
            out.append({"role": msg["role"], "content": msg["content"]})
    return out


async def run_turn(user_text, model, on_progress=None):
    """Append the user's turn, run it to an answer, return the new messages.

    *model* is an async ``(messages, tools) -> assistant message`` in the
    chat-completions shape. Injected rather than imported so the loop can be
    driven with a scripted stub — no key, no network — and so choosing a provider
    stays a decision for the caller.

    *on_progress* receives partial output from a running cell as it arrives. It
    is what makes a long job legible in a chat window instead of a stalled
    cursor; a transport will hand it a stream, tests hand it a list.
    """
    # Before the append: the user's own turn is one of the new messages a view
    # has to render, not context it already had.
    start = len(_messages)
    _append("user", user_text)
    tools = await tool_payload()

    token = _server._local_identity.set((WRITER_ID, WRITER_LABEL))
    try:
        for _round in range(_MAX_TOOL_ROUNDS):
            reply = await model(_llm_messages(), tools)
            calls = reply.get("tool_calls") or []
            _append("assistant", reply.get("content") or "", tool_calls=calls)
            if not calls:
                break
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                text, images = await _dispatch(name, args, on_progress)
                _append("tool", text, tool_call_id=call.get("id"), name=name)
                for img in images:
                    _append(
                        "user",
                        f"(image returned by {name})",
                        image=img.data,
                        mime=img.mimeType,
                    )
        else:
            # Not an error the model can be told about mid-turn: it is the turn
            # ending without an answer, and the user is owed that plainly.
            _append(
                "assistant",
                f"Stopped after {_MAX_TOOL_ROUNDS} tool calls without reaching an "
                "answer. Ask me to continue, or narrow the question.",
            )
    finally:
        _server._local_identity.reset(token)

    return _messages[start:]
