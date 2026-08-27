"""HTTP surface for the built-in chat loop, on the session child.

Mounted beside the observe API on the MCP app (``mcp.custom_route``), so it
shares the port and event loop with ``/mcp`` and reuses that module's
Host/Origin guard — a custom route is not covered by FastMCP's transport
security, and this one reaches a kernel.

**Two roots, on purpose.** The reads live under ``/api/*``, which the control
proxies always; the one write lives under ``/chat/*``, which the control proxies
only when it is loopback-bound. That is the user console's split
(``docs/user-console.md``) applied to the same problem: ``_SESSION_ALLOWED_ROOTS``
exists to keep an RCE off the browser origin, and a chat turn runs arbitrary code
just as surely as a console cell does. Folding the turn into ``api`` would leave
the allowlist in place, still enforced, and no longer true.

The split is also what the control's own gate *assumes*: it enforces the local
root as POST-only, because its CSRF check skips safe methods and a cross-site GET
to a proxied root is forwarded unchecked. So a readable history could not have
lived there anyway — and it does not need to, being a read like the job list.

**A turn is accepted, not awaited.** ``POST /chat/turn`` starts the turn and
returns; the view polls ``/api/chat/history``. That is the shape ``execute_code``
already uses when a job outlives its call, and here it is not optional: a turn
that runs a long cell would sit well past the control proxy's 300s per-read
bound, and a buffered reply has no bytes in between to hold it open. Polling also
happens to be what the shared conversation wants — every view reads the same
thread, and the one that sent a message has no special claim on the answer.
"""

import asyncio
import logging

from starlette.responses import JSONResponse

from .._config import get_setting
from . import _chat, _model, _observe

logger = logging.getLogger(__name__)

# Set by configure(); the routes are not mounted at all when chat is off, so
# these are only read on a surface that exists.
_config = None
_enabled = False

# The in-flight turn. Held so the task is not garbage collected mid-run (asyncio
# keeps only a weak reference), and so a second POST can be refused without
# waiting on the lock.
_turn_task = None

# Partial output of the cell a turn is waiting on, and the cell it belongs to.
#
# Held here rather than appended to the conversation because ``_llm_messages``
# re-projects every stored message on every later turn: streamed stdout would go
# back to the provider again and again, growing the prompt with text the
# finished tool result already carries in full. This is a view's read, so it
# lives on the transport and never enters the thread.
_live_job = None
_live_text = ""
_live_len = 0


def configure(config, *, agentless):
    """Take the resolved config; return whether chat should be served.

    The switch is ``observe.chat_enabled``, beside the console's, because what it
    turns on is a pane on the observe page — and it is read together with
    ``observe.enabled`` because that page is how anyone reaches these routes.

    *agentless* is the third term, and the one that is not configuration: this
    loop exists for users **without** an MCP harness, so a session an agent is
    already driving does not get one. Offering it there would be technically
    sound and practically confusing -- two agents on one kernel, of which only
    one can hold the claim, so the pane would answer questions and then refuse
    to run anything. The console has the same shape and is offered anyway,
    because a human typing a cell is not a second agent; a chat loop is.

    Required rather than defaulted, because both answers are wrong to assume: a
    default of True serves chat to every harness-driven session, and a default
    of False silently withholds it from the viewer it was built for. The two
    call sites both know.
    """
    global _config, _enabled
    _config = config
    _enabled = bool(
        get_setting(config, "observe.enabled")
        and get_setting(config, "observe.chat_enabled")
        and agentless
    )
    return _enabled


def _readiness():
    """``(ready, reason)`` — why chat cannot run, if it cannot.

    Reported rather than raised so the view can render the reason once, at the
    top of an empty thread, instead of the user discovering it by typing a
    message and having it rejected.
    """
    try:
        _model.check_ready(_config)
    except _model.ChatNotConfigured as exc:
        return False, str(exc)
    return True, None


async def _api_chat_status(request):
    ready, reason = _readiness()
    return JSONResponse(
        {
            "enabled": _enabled,
            "ready": ready,
            "reason": reason,
            "busy": _chat.busy(),
            "model": get_setting(_config, "chat.model") if ready else "",
        }
    )


async def _api_chat_history(request):
    """The conversation, or the part of it the caller has not seen.

    ``?after=<message id>`` is what keeps polling cheap: a screenshot rides in
    the thread as base64, so re-sending the whole conversation every few seconds
    would be wasteful in exactly the sessions that matter. An unknown id returns
    everything, which is the right answer for a view that has just loaded or has
    fallen behind a reset.

    ``partial`` carries the running cell's output so far. It rides this read
    rather than a route of its own because a view wants the two together: the
    thread says which tool is running, and this says what it has printed since.
    """
    messages = _chat.history()
    after = request.query_params.get("after")
    if after:
        for i, msg in enumerate(messages):
            if msg["id"] == after:
                messages = messages[i + 1 :]
                break
    return JSONResponse(
        {"messages": messages, "busy": _chat.busy(), "partial": _partial()}
    )


def _note_progress(chunk):
    """Accumulate a running cell's output, keyed to the cell it came from.

    Chunks carry no job id, but they only ever arrive from inside that cell's
    poll loop, so the loop's own ``running_job_id()`` is the answer -- and a
    change in it is how a new cell resets the buffer.
    """
    global _live_job, _live_text, _live_len
    job_id = _chat.running_job_id()
    if job_id != _live_job:
        _live_job, _live_text, _live_len = job_id, "", 0
    _live_text += chunk
    _live_len += len(chunk)
    # Bounded here, not only in the response: a cell printing in a loop would
    # otherwise grow this for as long as it runs. The tail is what is kept, for
    # the reason the job detail keeps it -- on a running job the newest output
    # is the informative part.
    cap = _observe._max_output_chars
    if len(_live_text) > cap:
        _live_text = _live_text[-cap:]


def _partial():
    """The running cell's output so far, or None when no cell is running.

    Gated on the job still being the current one. Once it finishes, the same
    text is in the thread as the tool's result, and reporting both would show a
    reader the cell's output twice.
    """
    job_id = _chat.running_job_id()
    if job_id is None or job_id != _live_job:
        return None
    return {
        "job_id": job_id,
        "stdout": _live_text,
        "truncated": _live_len > len(_live_text),
        "stdout_len": _live_len,
    }


async def _run_turn(text):
    """Run one turn, recording a failure in the thread rather than losing it."""
    try:
        await _chat.run_turn(
            text, _model.make_model(_config), on_progress=_note_progress
        )
    except asyncio.CancelledError:
        # Deliberate, and already in the thread. Absorbed here rather than in
        # _chat because this is the layer that asked for it; anywhere else, a
        # swallowed cancel is a bug.
        logger.info("chat turn cancelled")
    except Exception as exc:  # noqa: BLE001 - a provider/tool failure is content
        logger.warning("chat turn failed: %s", exc)
        _chat.note_error(f"The turn failed: {exc}")


def _in_flight():
    """Whether a turn is accepted-or-running.

    Two signals, because they cover different windows. The task exists from the
    moment a turn is *accepted*; the lock is held from the moment it *starts*,
    one event-loop step later — ``create_task`` only schedules, and the 202 goes
    out before the coroutine has run a line. Either alone leaves that step
    unguarded: reading only the lock, a second POST lands while the first turn
    is still queued and orphans it; reading only the task, a cancel reports
    success on a turn that had not begun and so recorded nothing.

    Neither is redundant in the other direction either — the task is the handle
    a cancel needs, and the lock is what a turn started by any other caller
    holds.
    """
    task = _turn_task
    return _chat.busy() or (task is not None and not task.done())


async def _chat_turn(request):
    """Accept a message and start the turn; the view polls for the answer."""
    ready, reason = _readiness()
    if not ready:
        return JSONResponse({"error": reason}, status_code=503)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is the client's error
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        return JSONResponse({"error": "missing 'text'"}, status_code=400)

    if _in_flight():
        # State, not a failed action -- the same shape the console reports a busy
        # kernel with, so the view renders it as "wait" rather than "retry".
        return JSONResponse(
            {"error": "a turn is already running", "busy": True}, status_code=409
        )

    global _turn_task
    _turn_task = asyncio.create_task(_run_turn(text))
    return JSONResponse({"accepted": True}, status_code=202)


async def _chat_cancel(request):
    """Stop the running turn, if there is one.

    POST because it changes state and because the control narrows this root to
    POST -- but also because it *should* be unforgeable: a cross-site request
    that kills someone's turn is a smaller harm than one that runs code, not a
    different kind of one. Hence ``_json_route`` too, though this carries no
    body and that guard's rationale exempts body-less POSTs: a JSON
    content-type is one an HTML form cannot set, and it costs a caller one
    header.

    Not an error when nothing is running. A person clicking cancel on a turn
    that has just finished has made no mistake, and an error toast would say
    they had.
    """
    if not _in_flight():
        return JSONResponse({"cancelled": False, "reason": "no turn is running"})
    # Reported, not just acted on: a turn cancelled before it started recorded
    # nothing, because nothing of it ran. That is the right thread -- the turn
    # is wholly absent rather than half-present -- but a view waiting for a
    # cancellation message would wait forever, so it is told which happened.
    started = _chat.busy()
    task = _turn_task
    # Only the turn. A cell it started keeps running, exactly as one started
    # through `execute_code` does when its MCP client goes away: the call stops
    # waiting, the kernel does not stop working, and whether to interrupt is the
    # human's call from the job list. Reaching into the kernel from here would
    # make the chat pane the one place where walking away also destroys work.
    if task is not None:
        task.cancel()
    return JSONResponse({"cancelled": True, "started": started})


async def _chat_reset(request):
    """Drop the conversation and start a new one.

    The escape hatch the thread had no other exit from. ``_llm_messages``
    re-projects every stored message on every turn, so a conversation only
    grows -- and once it outgrows the provider's context the turn fails, records
    the failure *in the thread*, and fails the same way forever. Without this
    the only way out was restarting the session child, which takes the kernel
    and the viewer window with it: losing a namespace to clear a chat.

    Refused while a turn is in flight rather than cancelling it. A reset lands
    mid-flight as a cleared thread that ``_run_turn`` then appends the rest of
    its round into -- an assistant turn whose calls have no history behind them,
    which is the shape that fails at the provider on every later turn. Cancel is
    right there, and says what it is doing.

    The cells the conversation ran are *not* touched. They stay in the job list
    and the notebook export, because clearing a conversation is not undoing the
    work it did -- and the kernel namespace it built is still live.
    """
    if _in_flight():
        return JSONResponse(
            {"error": "a turn is running; cancel it first", "busy": True},
            status_code=409,
        )
    global _turn_task, _live_job, _live_text, _live_len
    _chat.reset()
    _turn_task = None
    # The partial belongs to the thread that just went away. Left behind, it
    # would be published beside the first message of the new one.
    _live_job, _live_text, _live_len = None, "", 0
    return JSONResponse({"reset": True})


# Reads under `api` (always proxied), the writes under `chat` (proxied only by a
# loopback-bound control, and POST-only, which that gate enforces).
_ROUTES = [
    ("/api/chat/status", ["GET"], _observe._route(_api_chat_status)),
    ("/api/chat/history", ["GET"], _observe._route(_api_chat_history)),
    ("/chat/turn", ["POST"], _observe._json_route(_chat_turn)),
    ("/chat/cancel", ["POST"], _observe._json_route(_chat_cancel)),
    ("/chat/reset", ["POST"], _observe._json_route(_chat_reset)),
]


def register_http_routes():
    """Mount the chat routes on the existing FastMCP app.

    Must run before ``_server.run()``; custom routes are read when the
    streamable-http app is built.
    """
    for path, methods, handler in _ROUTES:
        _server_custom_route(path, methods)(handler)
    logger.info("chat API mounted at /api/chat/* and /chat/{turn,cancel,reset}")


def _server_custom_route(path, methods):
    from . import _server

    return _server.mcp.custom_route(path, methods=methods)
