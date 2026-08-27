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
* **A cancel stops the turn, not the cell.** ``execute_code`` over MCP hands
  back a job handle after ``promote_after`` and the cell outlives the call; the
  human stops it, or does not, from the observe page. A chat turn holds for the
  life of the cell instead (see ``_run_code``), but ending the turn should not
  quietly change what happens to the user's running code. So cancelling stops
  the polling and says which cell was left behind — the same two decisions an
  MCP user gets, made in the same order.
* **The model is injected.** Nothing here knows a provider. The caller passes an
  async ``model(messages, tools) -> assistant message``, which keeps the loop
  testable with no key and no network, and keeps provider choice out of it.
"""

import asyncio
import json
import logging
import time

from mcp.types import ImageContent, TextContent

from . import _server

logger = logging.getLogger(__name__)

#: This loop's client id, for the kernel's one-agent claim (``_jobs.submit``).
#: A fixed string rather than a per-view id, because every view drives the one
#: conversation: the loop is a single writer no matter how many windows are open.
WRITER_ID = "biopb-chat"
WRITER_LABEL = "chat"

#: The job origin this loop submits under. Also the point of view its
#: foreign-activity digest is read from (``_server._local_origin``): a cell is
#: someone *else's* only relative to whoever is asking.
ORIGIN = "chat"

#: Name of the synthesized tool that reads ``guide://`` and ``skill://``.
#: Resources are an MCP concept with no function-calling equivalent, so a model
#: driven by this loop cannot reach them unless one is invented -- and the
#: session instructions and ``find_skills`` both send it there, so without this
#: the agent is told to read documents it has no way to open.
RESOURCE_TOOL = "read_resource"

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

# The cell a turn is currently waiting on, or None. Only so a cancelled turn can
# name what it walked away from -- the job outlives the turn by design.
_running_job_id = None

# One turn at a time, for the same reason the kernel runs one job at a time.
# Two turns would interleave into the one thread and each compose against a
# history the other is still writing -- and both would reach for the same
# kernel. Held for the whole turn, not per tool call.
_turn_lock = asyncio.Lock()


class TurnInProgress(RuntimeError):
    """Raised when a turn is asked for while one is already running.

    Refused rather than queued, matching ``_jobs.submit``: a queued turn would
    be composed against a conversation its sender has not seen the end of, which
    is an ordering nobody can inspect. The transport reports it the way the user
    console reports a busy kernel -- as state, with a 409.
    """


def busy():
    """Whether a turn is running. A read, for the transport's status/409."""
    return _turn_lock.locked()


def running_job_id():
    """The cell the turn is waiting on right now, or None.

    Public because a transport buffering that cell's partial output has to know
    which cell the chunks belong to, and when it stopped being the current one.
    """
    return _running_job_id


def note_error(text):
    """Record a failed turn in the thread itself.

    A turn that dies inside a background task would otherwise vanish: the view
    is polling a conversation that simply stops growing, which reads as a hung
    session rather than a failure. Flagged so a view can render it as an error
    instead of as something the agent said.
    """
    return _append("assistant", text, error=True)


def reset():
    """Drop the conversation (used by tests and on an explicit new session)."""
    global _seq, _running_job_id
    _messages.clear()
    _seq = 0
    _running_job_id = None


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


async def _job_call(host, snippet):
    """``_server._run_job_call`` off the event loop.

    The round trip blocks: it waits on the kernel's lock (up to
    ``kernel.busy_lock_timeout``) and then on the reply. One of those in an
    ``/api/*`` handler is a blip; a chat turn makes one per poll for as long as
    a job runs, and this process is also serving ``/mcp`` to any attached client
    and the observe page to the user. So it goes to a thread, which the kernel
    host already expects -- its lock exists because the tools and the observe
    API already reach it concurrently.
    """
    return await asyncio.to_thread(_server._run_job_call, host, snippet)


def _clean_schema(schema):
    """The tool's input schema, minus the keys some providers reject.

    ``$schema`` and ``title`` are pydantic's, meaningful to a validator and noise
    to a function-calling API — several reject the payload outright rather than
    ignoring them.
    """
    return {k: v for k, v in (schema or {}).items() if k not in ("$schema", "title")}


async def _resource_tool():
    """A function-calling tool for the resource surface, built from the registry.

    An MCP client reads ``guide://kernel`` through ``resources/read``; a model
    speaking function-calling has no such verb, so the loop hands it one. This
    is not a convenience — ``_BASE_INSTRUCTIONS`` tells the agent to read the
    guides before non-trivial work and ``find_skills`` answers with
    ``skill://<id>``, so an agent without it is instructed to open documents it
    cannot reach, and will answer from guesswork instead.

    The catalogue in the description is generated, not written down, for the
    same reason the tool list is: a hand-kept copy is what silently stops
    matching what is registered.
    """
    listed = await _server.mcp.list_resources()
    templates = await _server.mcp.list_resource_templates()
    lines = [f"- {r.uri} — {r.description or ''}".rstrip() for r in listed]
    lines += [f"- {t.uriTemplate} — {t.description or ''}".rstrip() for t in templates]
    return {
        "type": "function",
        "function": {
            "name": RESOURCE_TOOL,
            "description": (
                "Read one of this session's reference documents. The guides are "
                "what the session instructions mean by 'read guide://...' — read "
                "the relevant one before non-trivial work rather than guessing "
                "at the API. A curated workflow's steps come from "
                "skill://<skill_id>, and find_skills is what gives you the id.\n"
                + "\n".join(lines)
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "e.g. guide://data or skill://drift-correction",
                    }
                },
                "required": ["uri"],
            },
        },
    }


async def _read_resource(uri):
    """Resolve *uri*, or say why it did not resolve.

    A bad URI is the model's mistake to correct on the next round, not the
    turn's end -- so it comes back as a tool result like any other.
    """
    try:
        parts = list(await _server.mcp.read_resource(uri))
    except Exception as exc:  # noqa: BLE001 - unknown uri, or the reader raised
        return f"Could not read {uri!r}: {exc}"
    out = []
    for part in parts:
        content = part.content
        out.append(
            content.decode("utf-8", "replace")
            if isinstance(content, bytes)
            else str(content)
        )
    return "\n".join(out)


async def tool_payload():
    """The function-calling tool list, generated from the live MCP registry.

    Read from ``list_tools()`` rather than declared here: a hand-written copy is
    the one thing that can silently stop matching the tools that actually run.
    The resource reader is appended because the registry has no such tool to
    generate from -- see :func:`_resource_tool`.
    """
    return [await _resource_tool()] + [
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
    if name == RESOURCE_TOOL:
        return await _read_resource(arguments.get("uri") or ""), []
    if name == "execute_code":
        return await _run_code(arguments, on_progress), []
    result = await _server.mcp._tool_manager.call_tool(
        name, arguments, convert_result=True
    )
    blocks = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    text = "\n".join(b.text for b in blocks if isinstance(b, TextContent))
    return text, [b for b in blocks if isinstance(b, ImageContent)]


async def _run_code(arguments, on_progress):
    """``execute_code`` without the promote window, streaming partial output.

    The tool waits ``promote_after`` seconds before handing back a job handle,
    which saves a round trip for an MCP client and buys a chat surface nothing —
    the model is blocked, so the stream falls silent for exactly as long. Here
    the job is submitted, its output reported as it accumulates, and the turn
    resumes when it ends.

    *intent* prefers what the model supplied — it is closer to the cell than the
    turn is — and falls back to the user's own words, so the notebook export has
    a stated purpose either way.

    The foreign-activity note is read and appended here for the same reason
    ``execute_code`` does it, and this path is the one that matters: it is the
    tool the model reaches for most, so a session where it only ever runs code
    never learned that the person at the machine had run any. Read once at
    entry, appended to whichever branch returns.
    """
    host = _server._kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    code = arguments.get("python_code") or ""
    intent = arguments.get("intent") or _last_user_text()

    # Off the loop, like every other kernel round trip here. The context copy
    # carries `_local_origin`, which is what keeps this loop's own cells out of
    # its own digest.
    digest = await asyncio.to_thread(_server._foreign_digest, host)
    foreign_note = _server._render_foreign_note(digest)
    if foreign_note:
        await asyncio.to_thread(_server._ack_foreign_digest, host, digest, WRITER_ID)

    # Claimed before the call, and mirrored, for the same reason execute_code
    # does it: a lost reply must not leave the kernel held while this process
    # reads as unclaimed.
    _server._presume_claim(WRITER_ID)
    submitted, res, _w = await _job_call(
        host,
        "submit("
        + repr(code)
        + ", origin="
        + repr(ORIGIN)
        + ", intent="
        + repr(intent)
        + ", writer="
        + repr(WRITER_ID)
        + ", writer_label="
        + repr(WRITER_LABEL)
        + ")",
    )
    if submitted is None:
        return _server._format_execute_result(res) + foreign_note
    if submitted.get("error") == "not_owner":
        held = submitted.get("owner") or ""
        _server._note_claim(submitted.get("owner_id"))
        return (
            _server._NOT_OWNER_MSG.format(held_by=f" ({held})" if held else "")
            + foreign_note
        )
    _server._note_claim(WRITER_ID)
    if submitted.get("error") == "busy":
        running = submitted.get("running_job_id")
        # A running foreign job stays in the digest by design, so the note is
        # about to report the very cell this branch reports. Drop it when that
        # is all it says, and keep it when other cells also finished -- those
        # were acked above and will not be offered again.
        if [d.get("job_id") for d in digest] == [running]:
            foreign_note = ""
        return (
            f"A job ({running}) is already running in this "
            "kernel; wait for it to finish." + foreign_note
        )

    global _running_job_id
    job_id = submitted["job_id"]
    # Recorded so a cancelled turn can name the cell it walked away from. It
    # means "the cell being polled right now", and every way of ceasing to poll
    # clears it except the one where the statement stays true.
    _running_job_id = job_id
    seen = 0
    try:
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            snap, res, _w = await _job_call(host, "poll(" + repr(job_id) + ")")
            if snap is None:
                _running_job_id = None
                return _server._format_execute_result(res) + foreign_note
            out = snap.get("stdout") or ""
            if on_progress is not None and len(out) > seen:
                on_progress(out[seen:])
                seen = len(out)
            if snap.get("status") != "running":
                _running_job_id = None
                return _server._format_execute_result(snap) + foreign_note
    except asyncio.CancelledError:
        # The one case that keeps it: the cell really is still running, and the
        # turn's handler is about to name it. That handler clears it.
        raise
    except BaseException:
        # Stopping for any other reason says nothing about the cell, and the id
        # must not linger to be reported as "still running" by a later cancel.
        _running_job_id = None
        raise


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
    cursor; the HTTP surface hands it a buffer it publishes on the history read,
    tests hand it a list.

    Raises :class:`TurnInProgress` if a turn is already running. Checking the
    lock before taking it is safe on the one event loop this process runs:
    acquiring a free lock does not yield, so nothing can slip between.
    """
    if _turn_lock.locked():
        raise TurnInProgress("a turn is already running in this session")
    async with _turn_lock:
        return await _run_turn(user_text, model, on_progress)


def _close_open_calls(reason):
    """Answer every tool call the last assistant turn left open.

    A turn that stops mid-round would otherwise store an assistant message whose
    calls were never all answered, and ``_llm_messages`` re-projects the whole
    thread on every later turn -- so an interrupted turn would fail the
    conversation at the provider from then on. The thread has to be left
    well-formed whatever ends the turn, not only when it ends by finishing.
    """
    for i in range(len(_messages) - 1, -1, -1):
        if _messages[i].get("tool_calls"):
            break
    else:
        return
    answered = {
        m.get("tool_call_id") for m in _messages[i + 1 :] if m["role"] == "tool"
    }
    for call in _messages[i]["tool_calls"]:
        if call.get("id") in answered:
            continue
        _append(
            "tool",
            reason,
            tool_call_id=call.get("id"),
            name=(call.get("function") or {}).get("name") or "",
            error=True,
        )


async def _run_turn(user_text, model, on_progress):
    # Everything that touches the thread is inside the try, so the turn is
    # either wholly absent or recorded with an outcome. Appending the user's
    # message and *then* awaiting outside the handler left the one state a view
    # cannot read: its own message, followed by nothing, forever.
    start = len(_messages)
    token = _server._local_identity.set((WRITER_ID, WRITER_LABEL))
    origin_token = _server._local_origin.set(ORIGIN)
    try:
        # Before the first await: the user's own turn is one of the new messages
        # a view has to render, not context it already had.
        _append("user", user_text)
        tools = await tool_payload()
        for _round in range(_MAX_TOOL_ROUNDS):
            reply = await model(_llm_messages(), tools)
            calls = reply.get("tool_calls") or []
            _append("assistant", reply.get("content") or "", tool_calls=calls)
            if not calls:
                break
            # Images are held back until every call in the round has answered.
            # A tool message that does not directly follow its assistant turn is
            # rejected outright, so an image landing between two tool results
            # would strand the second call's id -- and, being stored, would fail
            # the same way on every later turn rather than just this one.
            pending = []
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                try:
                    text, images = await _dispatch(name, args, on_progress)
                    failed = False
                except Exception as exc:  # noqa: BLE001 - a raising tool is content
                    # The hand-written paths already answer their own failures
                    # (_read_resource, _run_code); this gives the generic one the
                    # same manners. A tool that raises is usually the model's
                    # mistake to correct -- a hallucinated name, an argument the
                    # schema let through -- so it gets another round to do that,
                    # bounded by _MAX_TOOL_ROUNDS. Letting it escape would also
                    # store an assistant turn whose calls were never answered,
                    # which fails at the provider on every turn after it.
                    logger.warning("chat tool %s failed: %s", name, exc)
                    text, images, failed = f"Error: {exc}", [], True
                _append(
                    "tool",
                    text,
                    tool_call_id=call.get("id"),
                    name=name,
                    **({"error": True} if failed else {}),
                )
                pending.extend((name, img) for img in images)
            for name, img in pending:
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
    except asyncio.CancelledError:
        # Recorded, then re-raised. Recorded because a view polling the history
        # would otherwise see the conversation simply stop growing, which reads
        # as a hang -- the same reason a failed turn lands in the thread.
        # Re-raised because swallowing a cancel lies to whoever asked for it,
        # and the transport is the layer that knows this one was deliberate.
        global _running_job_id
        _close_open_calls("Cancelled before this call finished.")
        # The cell is deliberately left alone -- see the module docstring. Named,
        # because a turn that vanishes while the kernel is still busy is the one
        # thing a user cannot work out from the thread.
        _append(
            "assistant",
            "Turn cancelled."
            + (
                f" Cell {_running_job_id} is still running in the kernel;"
                " interrupt it from the job list if you want it stopped."
                if _running_job_id
                else ""
            ),
            cancelled=True,
        )
        # Cleared only now, once it has been reported. Held past the poll loop
        # so this message could name the cell; held any longer and the *next*
        # cancel would name it too, by which time it has almost certainly
        # finished -- a stale id here does not read as stale, it reads as a
        # second cell nobody started.
        _running_job_id = None
        raise
    finally:
        _server._local_identity.reset(token)
        _server._local_origin.reset(origin_token)

    return _messages[start:]
