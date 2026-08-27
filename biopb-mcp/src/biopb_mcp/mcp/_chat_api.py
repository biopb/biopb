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
from . import _chat, _chat_acp, _model, _observe, _server

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
    global _config, _enabled, _engine
    _config = config
    _engine = get_setting(config, "chat.engine") or "builtin"
    _enabled = bool(
        get_setting(config, "observe.enabled")
        and get_setting(config, "observe.chat_enabled")
        and agentless
    )
    return _enabled


# Which engine is driving the pane. Held here rather than read from the config
# on every request because it can be switched at runtime (`POST /chat/engine`)
# and the config file is not where a runtime switch belongs -- the file says
# what this session *starts* as.
_engine = "builtin"


def _acp():
    return _engine == "acp"


def _readiness(engine=None):
    """``(ready, reason)`` — why the engine cannot run, if it cannot.

    Reported rather than raised so the view can render the reason once, at the
    top of an empty thread, instead of the user discovering it by typing a
    message and having it rejected.

    The two engines fail for unrelated reasons -- the built-in loop for want of
    a model and a key, the ACP one for want of an installed harness -- so each
    answers for itself and the caller never has to know which is which.
    """
    engine = engine or _engine
    try:
        if engine == "acp":
            _chat_acp.check_ready(_config)
        else:
            _model.check_ready(_config)
    except (_model.ChatNotConfigured, _chat_acp.AcpNotConfigured) as exc:
        return False, str(exc)
    return True, None


def _engine_rows():
    """Every engine, whether it could run, and why not — for the switcher.

    A view offering a choice has to be able to grey out the half that cannot
    work and say why, rather than letting someone pick an engine that then
    refuses their first message.
    """
    rows = []
    for name in ("builtin", "acp"):
        ready, reason = _readiness(name)
        rows.append({"engine": name, "ready": ready, "reason": reason})
    return rows


async def _api_chat_status(request):
    ready, reason = _readiness()
    return JSONResponse(
        {
            "enabled": _enabled,
            "ready": ready,
            "reason": reason,
            "busy": _busy(),
            "engine": _engine,
            "engines": _engine_rows(),
            # What is answering, in the words that fit the engine: a model id
            # for the built-in loop, a harness name for ACP. One field, because
            # to a reader they are the same fact -- who am I talking to.
            "model": _who(ready),
            # How much of the thread the model no longer sees in full. The pane
            # renders every message either way, so without this the compaction
            # would be invisible to the person who asked for it. Always zero
            # under ACP, which folds its own context and never tells us.
            "compacted": 0 if _acp() else _chat.summary_state()[1],
        }
    )


def _busy():
    return _chat_acp.busy() if _acp() else _chat.busy()


def _who(ready):
    if not ready:
        return ""
    if _acp():
        agent = _chat_acp.agent_name() or get_setting(_config, "chat.acp_agent")
        # Both, when the model was named: the harness is who is answering and
        # the model is what it costs, and a reader wants the second as much as
        # the first. Agent alone when it was not, because naming the harness's
        # default here would be a guess rendered as fact.
        #
        # The live session first: it is the one answering, and it can be moved
        # off the configured model -- by `/model`, or by the harness itself.
        model = _chat_acp.current_model() or get_setting(_config, "chat.acp_model")
        return f"{agent} · {model}" if model else agent
    return get_setting(_config, "chat.model")


async def _api_chat_engine(request):
    """Which engine is driving the pane, right now.

    A read of its own rather than a field on the status probe, because the
    engine is session state and any window can change it: two observe pages on
    one session, and the one that did not click has to find out. Status is
    probed once per page, so it cannot be where that lands.

    The pane reads this ahead of every history read. Everything about how it
    renders is keyed to the engine -- which adapter parses the page, how the
    cursor is spelled, which slash commands exist -- so it is asked before the
    thread rather than inferred from it.
    """
    ready, reason = _readiness()
    # `model` rides along because it is the same fact from the reader's side:
    # an engine switched under them that still names the outgoing engine's model
    # is a header contradicting the switcher beside it.
    return JSONResponse({"engine": _engine, "model": _who(ready)})


async def _api_chat_models(request):
    """What this engine can be pointed at, and what it is pointed at now.

    Read when ``/model`` is typed rather than polled: the list changes only when
    the session does, and it is long enough that carrying it on every poll would
    be paying for it a hundred times to read it once.

    ``choices`` is empty under the built-in loop, and that is not a degraded
    answer -- an OpenAI-compatible endpoint has no model list we could trust, so
    the honest report is the model in force and an invitation to name another.
    The pane says which of the two it is by whether the list is empty.
    """
    if _acp():
        return JSONResponse(
            {
                "engine": "acp",
                "model": _chat_acp.current_model()
                or get_setting(_config, "chat.acp_model"),
                "choices": _chat_acp.model_choices(),
            }
        )
    return JSONResponse(
        {
            "engine": "builtin",
            "model": get_setting(_config, "chat.model"),
            "choices": [],
        }
    )


async def _chat_model(request):
    """Point the engine in force at another model.

    Refused while a turn is running, and for a sharper reason than a reset is: a
    turn is several provider calls, and the built-in loop reads the model on
    each one. Switching mid-turn would answer half a round in one model's voice
    and half in another's.

    Not persisted, for the reason the engine is not: the config file says what a
    session *starts* as, and a change made in one window should not re-aim every
    future viewer. It does reach every window of *this* session -- the pane
    reads the model beside the engine on every poll.
    """
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is the client's error
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    wanted = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(wanted, str) or not wanted.strip():
        return JSONResponse(
            {"error": "model must be a non-empty string"}, status_code=400
        )
    wanted = wanted.strip()
    if _in_flight():
        return JSONResponse(
            {"error": "a turn is running; cancel it first", "busy": True},
            status_code=409,
        )
    key = "acp_model" if _acp() else "model"
    if _acp() and _chat_acp.session_started():
        # A live session moves without a respawn, which is the point: changing
        # model should not cost the conversation.
        try:
            await _chat_acp.set_model(wanted)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001 - the harness's failure to report
            logger.warning("could not set the ACP model: %s", exc)
            return JSONResponse(
                {"error": f"the agent refused the model: {exc}"}, status_code=502
            )
    # Written back either way, so the answer survives what ends the session: an
    # agent not yet spawned starts on it, and one restarted by `/new` comes back
    # on it rather than on the model the config file named an hour ago.
    _config.setdefault("chat", {})[key] = wanted
    return JSONResponse({"model": wanted, "engine": _engine})


async def _api_chat_history(request):
    """The conversation, or the part of it the caller has not seen.

    ``?after=<message id>`` is what keeps polling cheap: a screenshot rides in
    the thread as base64, so re-sending the whole conversation every few seconds
    would be wasteful in exactly the sessions that matter. An unknown id returns
    everything, which is the right answer for a view that has just loaded or has
    fallen behind a reset -- and ``full`` says which of those two a page is, so
    a view that has fallen behind a reset replaces what it holds instead of
    appending to it.

    ``partial`` carries the running cell's output so far. It rides this read
    rather than a route of its own because a view wants the two together: the
    thread says which tool is running, and this says what it has printed since.

    Under the ACP engine the payload is ``items`` rather than ``messages``, and
    the cursor is a revision watermark rather than a last-seen id -- see
    :func:`_acp_history`. A view picks its adapter by which key it is given.
    """
    if _acp():
        return _acp_history(request)
    messages = _chat.history()
    after = request.query_params.get("after")
    full = True
    if after:
        for i, msg in enumerate(messages):
            if msg["id"] == after:
                messages, full = messages[i + 1 :], False
                break
    return JSONResponse(
        {
            "messages": messages,
            # Whether this is the whole thread or a delta. A view cannot tell
            # them apart from the messages alone, and after a reset it must:
            # every other window is still holding the old conversation and a
            # cursor into it, and appending the new thread to the old one leaves
            # the cleared conversation on screen. Ids are monotone across a
            # reset (see _chat.reset), so an unrecognised cursor is exactly this
            # case -- and a *full* page can be empty, which is what a reset with
            # nothing said since looks like.
            "full": full,
            "busy": _chat.busy(),
            "partial": _partial(),
        }
    )


def _acp_history(request):
    """The ACP thread, or the part of it revised since ``?since=<rev>``.

    A revision watermark rather than an id cursor because ACP updates items in
    place: a tool call the view already has moves from ``in_progress`` to
    ``completed`` without any new item appearing, and an "everything after id X"
    read cannot express that.

    ``commands`` rides this read rather than the status one because the harness
    advertises them by notification and may change them mid-session, while the
    pane probes status once per session.
    """
    raw = request.query_params.get("since")
    try:
        since = int(raw) if raw is not None else None
    except ValueError:
        since = None
    items, full = _chat_acp.history(since)
    return JSONResponse(
        {
            "items": items,
            "rev": _chat_acp.revision(),
            "full": full,
            "busy": _chat_acp.busy(),
            "commands": _chat_acp.commands(),
            "usage": _chat_acp.usage(),
            # No partial: the harness runs its own cells through `execute_code`
            # over MCP, so a running job's output belongs to the job list the
            # observe page already shows, not to a buffer on this transport.
            "partial": None,
        }
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
    engine = _chat_acp if _acp() else _chat
    try:
        if _acp():
            await _chat_acp.run_turn(text, _config)
        else:
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
        engine.note_error(f"The turn failed: {exc}")


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
    return _busy() or (task is not None and not task.done())


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
    if _acp():
        # The engine needs the handle too: its cancel stops the task *and* tells
        # the harness to stop, and only one of those is this module's to do.
        _chat_acp.set_turn_task(_turn_task)
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
    started = _busy()
    task = _turn_task
    if _acp():
        # The harness has to be told, or it keeps working (and keeps billing)
        # after the turn that asked for it is gone. Its own cancel path settles
        # any permission question left hanging, which a bare task.cancel() here
        # would leave the harness waiting on forever.
        await _chat_acp.cancel()
        return JSONResponse({"cancelled": True, "started": started})
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
    if _acp():
        # Also ends the harness's session, not just our transcript. Half a reset
        # is worse than none: the agent would go on composing against a history
        # nobody can see.
        try:
            await _chat_acp.reset()
        except _chat_acp.TurnInProgress:
            return JSONResponse(
                {"error": "a turn is running; cancel it first", "busy": True},
                status_code=409,
            )
    else:
        _chat.reset()
    _turn_task = None
    # The partial belongs to the thread that just went away. Left behind, it
    # would be published beside the first message of the new one.
    _live_job, _live_text, _live_len = None, "", 0
    return JSONResponse({"reset": True})


async def _chat_summary(request):
    """Fold the older part of the conversation into a summary.

    The other half of the thread's context problem, and the gentler one: a
    reset gives up what was said, this keeps it. The summary stands in for the
    folded messages **only when the thread is projected to the provider** --
    ``_chat.history()`` is untouched, so the pane still shows every word. The
    human's record and the model's budget are different things and there is no
    reason to spend one to buy the other.

    A write, so it lives beside turn and cancel: it costs a provider call and
    changes what every later turn is composed against.

    User-triggered rather than automatic, which is also the kind answer for
    prompt caching. Every provider caches on an exact prefix, so folding the
    front of the thread invalidates it -- once, here, when a person asked for
    it. A rule that trimmed a little each turn would move the prefix on every
    call instead, including the twelve a single turn can make.

    Cannot rescue a thread that has *already* overflowed: summarising reads the
    part being folded, so a conversation too large to send is too large to
    summarise. That case is what /chat/reset is for.
    """
    if _acp():
        # Not ours to do. The harness manages its own context, with its own
        # summariser and its own budget; folding a transcript it does not read
        # would change nothing but what the pane shows.
        return JSONResponse(
            {
                "error": f"{_chat_acp.agent_name() or 'this agent'} manages its own "
                "context; compacting is not something biopb does for it"
            },
            status_code=400,
        )
    ready, reason = _readiness()
    if not ready:
        return JSONResponse({"error": reason}, status_code=503)
    if _in_flight():
        return JSONResponse(
            {"error": "a turn is running; wait for it or cancel it", "busy": True},
            status_code=409,
        )
    before = _chat.summary_state()[1]
    try:
        compacted = await _chat.compact(_model.make_model(_config))
    except _chat.TurnInProgress:
        return JSONResponse(
            {"error": "a turn is already running", "busy": True}, status_code=409
        )
    except Exception as exc:  # noqa: BLE001 - a provider failure is the answer
        logger.warning("chat compaction failed: %s", exc)
        return JSONResponse({"error": f"could not summarise: {exc}"}, status_code=502)
    return JSONResponse({"compacted": compacted, "folded": compacted - before})


async def _chat_permission(request):
    """Answer a permission question the harness is waiting on.

    A write under `chat` rather than a read under `api` for the obvious reason:
    saying yes here is what lets the agent run the thing it asked about, so this
    is the most execute-adjacent route in the file and belongs behind the same
    loopback-only, POST-only, JSON-content-type gate as the turn.

    Answering an unknown or already-settled question is a 409 rather than an
    error: two windows watch one conversation, and the second person to click is
    not making a mistake -- they are just second.
    """
    if not _acp():
        return JSONResponse(
            {"error": "the built-in loop asks no permission questions"},
            status_code=400,
        )
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is the client's error
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return JSONResponse({"error": "missing 'request_id'"}, status_code=400)
    option_id = payload.get("option_id")
    if option_id is not None and not isinstance(option_id, str):
        return JSONResponse({"error": "'option_id' must be a string"}, status_code=400)
    # A null option is a deliberate refusal, not a missing field: it is how the
    # pane says the person dismissed the question rather than choosing from it.
    if not _chat_acp.answer_permission(request_id, option_id):
        return JSONResponse(
            {"error": "that question is no longer open", "stale": True},
            status_code=409,
        )
    return JSONResponse({"answered": True})


async def _chat_engine(request):
    """Switch which agent drives the pane.

    Refused while a turn is running, for the reason a reset is: the turn in
    flight belongs to the engine that started it.

    The harder refusal is the kernel's. Both engines run code as an MCP client
    and the kernel admits one (``_jobs.submit``), so a switch made after the
    outgoing engine has claimed it leaves the incoming one refused on its first
    cell -- with a ``not_owner`` deep inside a tool result, which is the worst
    possible place to learn it. The claim is released only by a kernel restart,
    so this says so up front and points at the control that does it, rather than
    switching into a session that cannot work.

    Not persisted. The config file says what a session *starts* as; this is a
    choice about the session in front of you, and writing it back would silently
    re-aim every future viewer from a click in one of them.
    """
    global _engine, _turn_task
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is the client's error
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    wanted = payload.get("engine") if isinstance(payload, dict) else None
    if wanted not in ("builtin", "acp"):
        return JSONResponse(
            {"error": "engine must be 'builtin' or 'acp'"}, status_code=400
        )
    if wanted == _engine:
        return JSONResponse({"engine": _engine, "changed": False})
    if _in_flight():
        return JSONResponse(
            {"error": "a turn is running; cancel it first", "busy": True},
            status_code=409,
        )
    ready, reason = _readiness(wanted)
    if not ready:
        return JSONResponse({"error": reason}, status_code=503)
    holder = _server.claim_holder()
    if holder is not None:
        return JSONResponse(
            {
                "error": (
                    f"the kernel is already claimed by {holder}. Restart the "
                    "kernel from the job list to hand it over, then switch."
                ),
                "claimed_by": holder,
            },
            status_code=409,
        )
    if _engine == "acp":
        # Leaving ACP: the harness is a running process and a live subscription
        # to this session's /mcp. Nothing else reaps it, and left behind it goes
        # on holding an MCP session against a kernel the other engine now wants.
        await _chat_acp.shutdown()
    _engine = wanted
    _turn_task = None
    return JSONResponse({"engine": _engine, "changed": True})


# Reads under `api` (always proxied), the writes under `chat` (proxied only by a
# loopback-bound control, and POST-only, which that gate enforces).
_ROUTES = [
    ("/api/chat/status", ["GET"], _observe._route(_api_chat_status)),
    ("/api/chat/history", ["GET"], _observe._route(_api_chat_history)),
    ("/api/chat/engine", ["GET"], _observe._route(_api_chat_engine)),
    ("/api/chat/models", ["GET"], _observe._route(_api_chat_models)),
    ("/chat/turn", ["POST"], _observe._json_route(_chat_turn)),
    ("/chat/cancel", ["POST"], _observe._json_route(_chat_cancel)),
    ("/chat/reset", ["POST"], _observe._json_route(_chat_reset)),
    ("/chat/summary", ["POST"], _observe._json_route(_chat_summary)),
    ("/chat/permission", ["POST"], _observe._json_route(_chat_permission)),
    ("/chat/engine", ["POST"], _observe._json_route(_chat_engine)),
    ("/chat/model", ["POST"], _observe._json_route(_chat_model)),
]


def register_http_routes():
    """Mount the chat routes on the existing FastMCP app.

    Must run before ``_server.run()``; custom routes are read when the
    streamable-http app is built.
    """
    for path, methods, handler in _ROUTES:
        _server_custom_route(path, methods)(handler)
    logger.info(
        "chat API mounted at /api/chat/* and "
        "/chat/{turn,cancel,reset,summary,permission,engine,model} (engine: %s)",
        _engine,
    )


def _server_custom_route(path, methods):
    from . import _server

    return _server.mcp.custom_route(path, methods=methods)
