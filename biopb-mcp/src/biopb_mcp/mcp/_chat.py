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
  ``_writers._local_identity`` for the length of a dispatch, so the kernel's
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

from .. import _endpoint
from . import _app, _kernel_rpc, _server, _writers

# The kernel round trip, off the loop. Bound here under the name this module
# uses so a test can still swap it (test_mcp_chat_api).
from ._kernel_rpc import _job_call

logger = logging.getLogger(__name__)

#: This loop's client id, for the kernel's one-agent claim (``_jobs.submit``).
#: A fixed string rather than a per-view id, because every view drives the one
#: conversation: the loop is a single writer no matter how many windows are open.
WRITER_ID = "biopb-chat"
WRITER_LABEL = "chat"

#: The job origin this loop submits under. Also the point of view its
#: foreign-activity digest is read from (``_writers._local_origin``): a cell is
#: someone *else's* only relative to whoever is asking.
ORIGIN = "chat"

#: Name of the synthesized tool that reads ``guide://`` and ``skill://``.
#: Resources are an MCP concept with no function-calling equivalent, so a model
#: driven by this loop cannot reach them unless one is invented -- and the
#: session instructions and ``list_skills`` both send it there, so without this
#: the agent is told to read documents it has no way to open.
RESOURCE_TOOL = "read_resource"

#: Cap on tool-call rounds within one turn. A model that keeps calling tools
#: without answering is not converging, and the user is sitting there watching.
_MAX_TOOL_ROUNDS = 12

#: Tools whose only useful result is a picture. Withdrawn when images cannot
#: reach the model: it would spend a round on a screenshot and be handed a
#: parenthetical.
_IMAGE_TOOLS = frozenset({"take_screenshot"})

#: What stands in for an image the provider will not be sent, in the projection.
_NO_VISION_NOTE = " -- not sent: this model does not accept images."

#: How often a running job's output is re-read while the turn waits on it.
#: In-process this is a function call, so it is set for a responsive stream
#: rather than to spare a round trip.
_POLL_INTERVAL = 0.3

# The shared conversation. Module state like ``_jobs``: it belongs to the
# session, survives kernel restarts, and dies with this process.
_messages: list = []
_seq = 0

# This conversation's own id, for anything that has to name it to something
# outside this process -- today a gateway that attributes traffic per
# conversation (`_endpoint`), tomorrow a log or a trace. Minted here and
# re-minted by `reset`, so it tracks the thread rather than the process: two
# conversations in one session are two, and a kernel restart does not end one.
_session_id = _endpoint.new_session_id()

# A foreign-activity digest that has been written into a tool result but not yet
# discharged at the kernel. Held here for the step between the two, which is the
# only point at which the notice is provably delivered -- see
# :func:`_discharge_notice`.
_pending_notice = None
# The compacted prefix: a summary of the first `_compacted` messages, standing
# in for them when the thread is projected to the provider. Projection only --
# `_messages` keeps every word, because the human's record of the conversation
# is not the model's context budget and there is no reason to spend one to buy
# the other.
_summary = None
_compacted = 0

#: User turns left verbatim by :func:`compact`. Recent exchanges are what the
#: next answer is actually built on, and a summary of "what we just did" is the
#: part a summary is worst at.
_KEEP_TURNS = 2

# The cell a turn is currently waiting on, or None. Only so a cancelled turn can
# name what it walked away from -- the job outlives the turn by design.
_running_job_id = None

# The image policy, and what the provider has said about it. Session state
# rather than a config read at the point of use, because the model can be
# changed at runtime (`POST /chat/model`) and what one model refused says
# nothing about the next -- see :func:`set_vision`.
_vision_mode = "auto"
_vision_refused = False

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


class VisionUnsupported(RuntimeError):
    """The provider refused the payload because it carried an image.

    Part of the *injected model's* contract rather than the provider adapter's,
    for the same reason the adapter is injected: the loop is what has to act on
    it -- stop projecting images, withdraw the screenshot tool, try again -- and
    a second adapter has to be able to say the same thing.
    """


def set_vision(mode):
    """Set the image policy, forgetting what a previous model refused.

    Called with the configured value when chat is configured and again whenever
    the model is switched: a refusal is a fact about one model, and carrying it
    across a switch would leave a vision model blind for the rest of the
    session. An unknown value reads as "auto", which is the mode that recovers
    from being wrong.
    """
    global _vision_mode, _vision_refused
    _vision_mode = mode if mode in ("auto", "on", "off") else "auto"
    _vision_refused = False


def images_allowed():
    """Whether an image may be projected to the provider.

    Only the projection, never the thread: a screenshot is recorded and shown in
    the pane whatever this says, because the human can see it even when the
    model cannot.
    """
    if _vision_mode == "off":
        return False
    if _vision_mode == "on":
        return True
    return not _vision_refused


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


def session_id() -> str:
    """This conversation's id. See :data:`_session_id`."""
    return _session_id


def reset():
    """Drop the conversation.

    The thread only grows -- ``_llm_messages`` re-projects all of it every turn
    -- so starting a new one is the only bound it has until the projection
    itself gets a budget. Clears the running-cell id with it: the next thread
    must not open by naming a cell it never started, and the compacted prefix,
    which summarised messages that are now gone.

    The *message* id sequence is **not** restarted. Ids are how a view tells a message it
    has already drawn from one it has not, and ``/api/chat/history?after=`` is
    documented to return everything for an id it does not recognise -- which is
    how a window open across a reset notices. Restarting the count reissues the
    old thread's ids to the new one, so that stale cursor matches a message the
    view has never seen and it skips the new conversation's opening instead.
    """
    global _running_job_id, _summary, _compacted, _session_id
    _messages.clear()
    _running_job_id = None
    _summary, _compacted = None, 0
    # A new thread is a new conversation to anyone outside this process, so it
    # gets a new id. Unlike the message ids below, nothing holds a cursor on
    # this one -- it names the thread, it does not order anything within it.
    _session_id = _endpoint.new_session_id()


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


async def _resource_tool():
    """A function-calling tool for the resource surface, built from the registry.

    An MCP client reads ``guide://kernel`` through ``resources/read``; a model
    speaking function-calling has no such verb, so the loop hands it one. This
    is not a convenience — ``_BASE_INSTRUCTIONS`` tells the agent to read the
    guides before non-trivial work and ``list_skills`` answers with
    ``skill://<id>``, so an agent without it is instructed to open documents it
    cannot reach, and will answer from guesswork instead.

    The catalogue in the description is generated, not written down, for the
    same reason the tool list is: a hand-kept copy is what silently stops
    matching what is registered.
    """
    listed = await _app.mcp.list_resources()
    templates = await _app.mcp.list_resource_templates()
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
                "skill://<skill_id>, and list_skills is what gives you the id.\n"
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
        parts = list(await _app.mcp.read_resource(uri))
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


#: What :func:`_run_code` actually does, in place of the promote-and-poll
#: paragraph the wire tools describe. The behaviour is already overridden in
#: :func:`_dispatch`; the description has to be overridden at the same seam, or
#: the model is told to poll for a handle it will never be given -- and offered
#: ``poll_job`` to do it with.
_CHAT_RUN_PARAGRAPH = """Code runs in a background thread so it does not block the main thread.
    This call waits for the cell to finish and returns its output -- there is no
    job handle and nothing to poll for. Only one job runs at a time; stop a cell
    with interrupt_kernel (best-effort) or restart_kernel (guaranteed).

    poll_job still reads cells *the user* ran from the observe page, which is
    what the activity notice on these results points you at."""


def _describe(tool):
    """*tool*'s description as this loop's model should read it.

    Only ``execute_code`` differs, and only in the paragraph that describes a
    wire behaviour :func:`_run_code` replaces. Substituted rather than rewritten
    whole, so the rest stays the registry's own words -- a hand-written copy is
    the one thing that can silently stop matching the tool that actually runs.
    A test pins that this still finds its paragraph.
    """
    description = tool.description or ""
    if tool.name != "execute_code":
        return description
    return description.replace(_server.PROMOTE_PARAGRAPH, _CHAT_RUN_PARAGRAPH)


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
                "description": _describe(tool),
                "parameters": _clean_schema(tool.inputSchema),
            },
        }
        for tool in await _app.mcp.list_tools()
        if images_allowed() or tool.name not in _IMAGE_TOOLS
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
    result = await _app.mcp._tool_manager.call_tool(
        name, arguments, convert_result=True
    )
    blocks = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    text = "\n".join(b.text for b in blocks if isinstance(b, TextContent))
    return text, [b for b in blocks if isinstance(b, ImageContent)]


def _busy_message(running, running_origin) -> str:
    """This loop's reply when the kernel already has a job running.

    Deliberately not `_server._tool_busy_message`: that one hands the agent a
    `poll_job('<id>')` to wait on, and this path never returns a job handle for
    the model to poll (there is no promote window here -- `_run_code` streams
    the cell to its end). Naming who is running it still helps, and costs
    nothing the model can act wrongly on.
    """
    who = {"user": "The user", "chat": "This session"}.get(
        running_origin, "Another writer"
    )
    return (
        f"{who} is running a cell ({running}) in this kernel; only one job runs "
        "at a time. Wait for it to finish."
    )


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
    host, err = _app._require_kernel_host()
    if err is not None:
        return err
    code = arguments.get("python_code") or ""
    intent = arguments.get("intent") or _last_user_text()

    # Off the loop, like every other kernel round trip here. The context copy
    # carries `_local_origin`, which is what keeps this loop's own cells out of
    # its own digest.
    digest = await asyncio.to_thread(_writers._foreign_digest, host)
    foreign_note = _writers._render_foreign_note(digest)

    def deliver(text):
        """Attach the notice, and hold the digest for discharge.

        Not discharged here. The ack is a promise that the agent *has been
        told*, and it is told when the result carrying the note is appended to
        the thread -- which is after this returns, and after everything below
        that can be cancelled. Acking at entry made the loss window the whole
        cell: a turn cancelled three minutes into a job had retired a notice
        that never reached anyone, and the digest does not offer it twice.

        Nothing has to unset this on the way out. Every path that reaches it
        returns, and from here to the ``_append`` that records the result there
        is no await for a cancellation to be delivered at -- so the slot is
        never left armed by a turn that died. Put an await in between and that
        stops being true.
        """
        global _pending_notice
        if not foreign_note:
            return text
        _pending_notice = digest
        return text + foreign_note

    # The claim protocol is `_server._submit_job`'s, not a copy of it: presume
    # the claim, submit, believe whatever the kernel answers. `_local_identity`
    # and `_local_origin` are both set for this whole dispatch, so the writer
    # and the origin `_submit_job` records are already this loop's. Off the
    # loop, like every other kernel round trip here.
    job_id, message, drop_note, _w = await asyncio.to_thread(
        _server._submit_job,
        host,
        code,
        digest,
        _busy_message,
        intent=intent,
    )
    if drop_note:
        foreign_note = ""
    if message is not None:
        return deliver(message)

    global _running_job_id
    # Recorded so a cancelled turn can name the cell it walked away from. It
    # means "the cell being polled right now", and every way of ceasing to poll
    # clears it except the one where the statement stays true.
    _running_job_id = job_id
    seen = 0
    try:
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            snap, res, _w = await _job_call(host, "poll", job_id)
            if snap is None:
                _running_job_id = None
                return deliver(_kernel_rpc._format_execute_result(res))
            out = snap.get("stdout") or ""
            # Diffed against the job's monotonic total, not against `len(out)`:
            # the output cap compacts the buffer from the front mid-cell, so a
            # plain offset into it would stop matching and the pane would fall
            # silent for the rest of the run. `total - seen` is how much is
            # genuinely new; it comes off the end of the window we still hold,
            # bounded by that window when the new text was itself partly capped.
            total = snap.get("stdout_total")
            if total is None:
                total = seen + max(0, len(out) - seen)
            if on_progress is not None and total > seen:
                fresh = min(total - seen, len(out))
                on_progress(out[len(out) - fresh :])
            seen = max(seen, total)
            if snap.get("status") != "running":
                _running_job_id = None
                return deliver(_kernel_rpc._format_execute_result(snap))
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
    out = [{"role": "system", "content": _app.mcp._mcp_server.instructions or ""}]
    if _summary:
        # Its own system message rather than a user turn, because it is not
        # something anyone said: a user turn would be answerable, and a model
        # that answers the summary has lost the turn it was asked for.
        out.append({"role": "system", "content": _SUMMARY_PREFIX + _summary})
    for msg in _messages[_compacted:]:
        if msg.get("image") and not images_allowed():
            # The picture stays in `_messages` for the pane; only the projection
            # drops it, and it drops *every* one -- an image already in the
            # thread is re-sent on every later turn, so leaving the old ones in
            # would keep failing the calls that withdrawing them was meant to
            # save. The caption is kept because the model called the tool and is
            # owed an answer to that call.
            out.append({"role": "user", "content": msg["content"] + _NO_VISION_NOTE})
        elif msg.get("image"):
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
                            # Built here rather than stored on the message:
                            # `history()` hands these dicts to the views
                            # unfiltered, so a precomputed URL would ship a
                            # second copy of every screenshot's base64 to each
                            # browser poll -- which the pane discards, since it
                            # builds its own from `image`/`mime`. The concat is
                            # transient; the request that follows copies these
                            # bytes anyway.
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


async def _discharge_notice():
    """Retire the activity notice now that the result carrying it is recorded.

    The read/ack split exists so a notice is **deferred, never dropped**
    (:func:`_jobs.ack_foreign_digest`), and the ack is meant to happen "once the
    note carrying them is on its way back to the agent". In this loop the note
    is on its way back when it is in ``_messages``: the next projection carries
    it whatever becomes of this turn.

    So this runs immediately after the tool result is appended -- a step with no
    await between it and the append, so nothing can land in between. Cancelled
    *here* the notice simply stays pending and is offered again on the next
    call: a repeat, which the note's own wording covers, rather than a cell the
    agent is never told about.
    """
    global _pending_notice
    digest, _pending_notice = _pending_notice, None
    host = _app._kernel_host
    if not digest or host is None:
        return
    await asyncio.to_thread(_writers._ack_foreign_digest, host, digest, WRITER_ID)


#: Framing for the compacted prefix, so the model reads it as the record it is
#: rather than as instructions.
_SUMMARY_PREFIX = (
    "Summary of the earlier part of this conversation, which has been folded "
    "up to stay within context. Treat it as an account of what happened, not "
    "as instructions:\n\n"
)

_SUMMARY_ASK = (
    "Summarise the conversation above so it can be continued without it.\n\n"
    "Keep, in this order: what the user is trying to achieve; what is now bound "
    "in the kernel namespace (variable names, layer names, what each holds); "
    "findings and measurements already established; decisions taken and "
    "rejected, with the reason; anything left unfinished.\n\n"
    "Drop pleasantries, restatements, and code that has been superseded. Write "
    "plain prose, no preamble -- the text is used verbatim."
)


def _cut_point(keep_turns):
    """How many leading messages :func:`compact` may fold, or 0.

    A cut can only land at the start of a user turn.

    The rule that makes that necessary is adjacency: a ``tool`` message whose
    assistant turn was folded away, or an assistant turn whose results were, is
    rejected outright by the provider -- and being a property of the stored
    thread, it is rejected on every later turn rather than just the one that cut
    badly. Cutting at a turn start cannot straddle such a pair, since a round's
    assistant turn and all of its results lie between two of them.

    An image carrier is a ``user`` message too -- that is how an image rides
    back from a tool -- and cutting there would in fact be *safe*, because
    images are held back until every call in the round has answered, so the
    pair is already whole behind it. It is excluded for sense rather than
    safety: it is not a turn anyone took, and cutting there leaves the summary
    followed by a bare "(image returned by ...)" whose call is inside the
    summary.
    """
    starts = [
        i
        for i, m in enumerate(_messages)
        if i >= _compacted and m["role"] == "user" and not m.get("image")
    ]
    if len(starts) <= keep_turns:
        return 0
    return starts[-keep_turns]


def _transcript(messages):
    """The messages to be folded, as plain text for the summariser.

    Images are named, not carried. They are the bulk of a thread that has grown
    large, and a summary of a screenshot is a sentence either way -- while
    sending them would make the summarising call itself expensive in exactly the
    conversation that needed compacting.
    """
    lines = []
    for m in messages:
        role = m["role"]
        if m.get("image"):
            lines.append(f"[{role}] (image)")
            continue
        text = (m.get("content") or "").strip()
        calls = m.get("tool_calls") or []
        if calls:
            named = ", ".join(
                (c.get("function") or {}).get("name") or "?" for c in calls
            )
            text = (text + f"\n[called: {named}]").strip()
        if text:
            lines.append(f"[{role}] {text}")
    return "\n\n".join(lines)


async def compact(model, keep_turns=_KEEP_TURNS):
    """Fold the older part of the conversation into a summary. Projection only.

    Returns the number of messages now standing behind the summary.

    *model* is the same injected ``(messages, tools) -> assistant message`` a
    turn takes, called with no tools and with the transcript in one user
    message rather than as a conversation -- summarising is not a turn, and
    replaying the thread as itself is what we are trying to stop doing.

    **Incremental.** A previous summary is folded into the new one instead of
    the messages behind it being read again, so repeated compaction stays cheap
    and the call does not grow with the session.

    Raises :class:`TurnInProgress` if a turn is running: the projection is read
    once per tool round, so changing it mid-turn would move the ground under a
    round already in flight.
    """
    global _summary, _compacted
    if _turn_lock.locked():
        raise TurnInProgress("a turn is already running in this session")
    cut = _cut_point(keep_turns)
    if not cut:
        return _compacted
    async with _turn_lock:
        body = _transcript(_messages[_compacted:cut])
        prior = f"{_SUMMARY_PREFIX}{_summary}\n\n" if _summary else ""
        reply = await model(
            [{"role": "user", "content": f"{prior}{body}\n\n{_SUMMARY_ASK}"}], []
        )
        text = (reply or {}).get("content") or ""
        if not text.strip():
            raise RuntimeError("the summariser returned nothing")
        _summary, _compacted = text.strip(), cut
    return _compacted


def summary_state():
    """``(summary, compacted count)`` -- a read, for the transport's status."""
    return _summary, _compacted


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


async def _ask(model, tools):
    """One provider call; return ``(assistant message, tool list)``.

    The tool list comes back because it can change here. A model with no vision
    does not fail only the call that carries the screenshot -- the image is
    stored, so it is re-sent on every later turn and the conversation is dead
    until someone resets it. So the refusal is taken as the answer to a question
    nobody asked explicitly: images stop being projected, ``take_screenshot`` is
    withdrawn, and the same call is made again.

    Only in ``auto``, and only once. ``on`` is a user who has said to send them
    anyway, and repeating a call the provider has already refused for the same
    reason would just spend their money twice.
    """
    global _vision_refused
    try:
        return await model(_llm_messages(), tools), tools
    except VisionUnsupported as exc:
        if _vision_mode != "auto" or _vision_refused:
            raise
        logger.warning("chat model refused an image; images withdrawn: %s", exc)
        _vision_refused = True
        # In the thread, because the pane goes on showing screenshots the model
        # is no longer being sent and nothing else would say why its answers
        # stopped mentioning them.
        _append(
            "assistant",
            "This model does not accept images. Screenshots still appear here, "
            "but are no longer sent to it -- point chat.model at a model with "
            "vision, or set chat.vision to 'off' to stop taking them.",
            error=True,
        )
        tools = await tool_payload()
        return await model(_llm_messages(), tools), tools


async def _run_turn(user_text, model, on_progress):
    # Everything that touches the thread is inside the try, so the turn is
    # either wholly absent or recorded with an outcome. Appending the user's
    # message and *then* awaiting outside the handler left the one state a view
    # cannot read: its own message, followed by nothing, forever.
    start = len(_messages)
    token = _writers._local_identity.set((WRITER_ID, WRITER_LABEL))
    origin_token = _writers._local_origin.set(ORIGIN)
    try:
        # Before the first await: the user's own turn is one of the new messages
        # a view has to render, not context it already had.
        _append("user", user_text)
        tools = await tool_payload()
        for _round in range(_MAX_TOOL_ROUNDS):
            reply, tools = await _ask(model, tools)
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
                # Recorded, so the notice that rode in on it has been delivered.
                # Immediately after the append and before any await, so a cancel
                # cannot land between the two.
                await _discharge_notice()
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
        _writers._local_identity.reset(token)
        _writers._local_origin.reset(origin_token)

    return _messages[start:]
