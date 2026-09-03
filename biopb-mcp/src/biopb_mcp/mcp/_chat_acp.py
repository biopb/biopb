"""The ACP chat engine: hand the pane to a harness the user already runs.

The built-in loop (``_chat``) exists for someone with no agent at all, and is
deliberately thin. This is the other half: a user who already runs a coding
harness gets *that* harness in the observe page's chat pane, over the Agent
Client Protocol, with biopb's own tools wired into it.

Design notes
------------
* **biopb is handed over in the handshake, not registered.** ``session/new``
  carries an ``mcpServers`` array, and what goes in it is *this* session's
  ``/mcp`` over http. The entry the installer writes into the user's client
  config would be wrong here: it runs ``biopb-mcp --transport stdio``, which the
  shim turns into a new session child and a second napari window. The point is
  to drive the viewer the user is looking at.
* **http, and no bridge.** ``/mcp`` is this server's only transport already
  (``_server.run``), so the agent's MCP client just opens a socket. The
  alternative — a stdio entry pointing at a bridge process — would reintroduce
  the one process shape the http-only restructure exists to avoid: a process
  whose fd 1 is a protocol channel, sharing a tree with Qt/GL/dask native code
  that writes to fd 1 past Python.
* **The agent is a real MCP client, so it takes the kernel the ordinary way.**
  Nothing here calls the kernel. Cells arrive through ``_server``'s tool
  handlers like any other client's, which is also how the one-agent claim comes
  to cover it (``_jobs.submit``) and why its cells show up in the job list under
  ``origin="mcp"`` — it *is* an MCP client, just one we launched.
* **Items, not messages.** ``_chat`` appends to a chat-completions transcript;
  ACP sends normalized items and then updates them in place by id. So the
  transcript here is a list of items with a monotone ``rev``, and a view polls
  with a revision watermark rather than a last-seen id. An id cursor cannot
  express "this item you already have has changed".
* **The subprocess is owned, and its stderr is its own.** Reaped through the
  same ``OwnedChild`` path as the session child (Windows job object included),
  and its stderr goes to a file of its own — inherited, the harness's logs would
  land in the middle of the session log.
* **Threads move the pipes, not the event loop.** ``acp`` will speak over any
  ``Transport``, so this hands it one backed by a plain ``Popen`` and a reader
  thread. asyncio's own subprocess and pipe support is unavailable here on
  Windows: the http server runs on the Selector loop (see ``_serve_http``),
  which does not implement either.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: The harnesses this engine knows how to launch, by ``chat.acp_agent``.
#:
#: One entry, and the shortness is the finding rather than a starting point.
#: Claude Desktop has no headless agent at all; Claude Code needs a separately
#: fetched npm adapter; Cursor's ACP mode is reported to drop the ``mcpServers``
#: it is handed, which is the whole mechanism this depends on. opencode ships
#: ``opencode acp`` natively and honours the handshake.
_ACP_AGENTS = {
    "opencode": {
        "name": "opencode",
        # opencode ACP also starts its headless HTTP server. Keep that incidental
        # listener loopback-only and let the OS choose a collision-free port; ACP
        # itself uses the stdio pipe below and does not need this port.
        "argv": ("acp", "--hostname", "127.0.0.1", "--port", "0"),
        # Its own installer puts it here, which is not always on a GUI-launched
        # process's PATH -- the same reason _agents resolves biopb-mcp absolutely.
        "extra_paths": ("~/.opencode/bin/opencode",),
        # Where biopb's pinned settings go: an env var carrying an inline
        # config, merged above the user's global config, above a custom config
        # file, and above any project config -- so nothing the agent can write
        # displaces it. Only a machine-managed config outranks it.
        #
        # In the **environment**, not a file in the working directory, and that
        # is a security property rather than a convenience: the agent can write
        # anywhere under its cwd, so a permission file living there is one
        # approved edit away from the agent turning its own prompts off. This is
        # read once at process start and fixed for the run.
        "config_env": "OPENCODE_CONFIG_CONTENT",
        # ``chat.acp_permission="ask"`` expressed in this harness's spelling.
        # Its own defaults are permissive -- edit, bash, read and webfetch all
        # default to "allow" -- so without this the setting would only decide
        # how to answer questions the harness never asks, and a file write lands
        # with nothing shown to anyone.
        #
        # Enumerated rather than `{"*": "ask"}`, and the difference is the whole
        # point: the wildcard would put a prompt in front of biopb's own tools
        # too, so every cell the agent ran would stop for a click. These four
        # are the harness's *own* mutating and outbound actions. Reads stay
        # unprompted (noisy, low harm) and `external_directory` is left alone
        # because opencode already defaults it to "ask".
        # The harness's *own* registration of biopb, switched off. The installer
        # writes one into the user's client config under exactly this key
        # (``biopb._agents``), and opencode merges config MCP servers into an
        # ACP session alongside the ones handed to it -- so without this the
        # agent gets biopb twice: ours over http, driving the viewer the user is
        # looking at, and theirs over stdio, which the shim turns into a second
        # session child and a second napari window.
        #
        # Today the two happen to collide on the name "biopb" and ours wins, so
        # the duplicate does not appear. That is an accident of naming, not a
        # guarantee, and it stops holding the moment anyone registers biopb
        # under a different key. Measured: with our entry renamed, the agent
        # listed eighteen biopb tools.
        #
        # Only biopb's own entry. Any other MCP server the user configured is
        # theirs and stays: this is about not being present twice, not about
        # taking their tools away.
        "suppress_mcp": ("biopb",),
        "strict_permission": {
            "edit": "ask",
            "bash": "ask",
            "webfetch": "ask",
            "websearch": "ask",
        },
    },
}

#: Cap on how long a turn may wait for the harness to answer. Not a model
#: timeout: the harness owns its own retries and a long tool call is legitimate,
#: so this only stops a wedged child from holding the turn lock forever.
_TURN_TIMEOUT = 3600.0

#: How long to wait for the harness to come up and answer ``initialize``.
#: Dominated by process start, not by the network.
_START_TIMEOUT = 60.0


class AcpNotConfigured(RuntimeError):
    """The ACP engine cannot run: unknown agent, or its binary is not installed.

    One exception for both because they are one situation to the user -- "this
    is not set up yet" -- with a message that says which part is missing.
    """


# --------------------------------------------------------------------------- #
# The transcript
# --------------------------------------------------------------------------- #

# Module state, like ``_chat`` and ``_jobs``: it belongs to the session, outlives
# kernel restarts, and dies with this process.
_items: list = []
_seq = 0
_rev = 0
# The revision at the last reset. A view whose watermark predates it is holding
# a conversation that no longer exists, and must be given the whole thread
# rather than a delta to append to it.
_reset_rev = 0


def _bump():
    global _rev
    _rev += 1
    return _rev


def _new_item(kind, **fields):
    global _seq
    _seq += 1
    item = {"id": f"i-{_seq}", "kind": kind, "rev": _bump(), "ts": time.time()}
    item.update(fields)
    _items.append(item)
    return item


def _touch(item, **fields):
    """Apply an in-place update and re-stamp it so a polling view sees it.

    ACP sends only the fields that changed -- a completed tool call carries a
    status and nothing else -- so this merges rather than replaces, and drops
    ``None`` instead of writing it over a title the view already has.
    """
    for key, value in fields.items():
        if value is not None:
            item[key] = value
    item["rev"] = _bump()
    return item


def history(since=None):
    """``(items, full)`` -- the thread, or the part after revision *since*.

    *full* says which of the two a view is being given. It is not a nicety: a
    window open across a reset holds items that are gone, and appending a delta
    to them would leave the cleared conversation on screen.
    """
    if since is None or since < _reset_rev:
        return list(_items), True
    return [i for i in _items if i["rev"] > since], False


def revision():
    """The current watermark, for a view that wants to poll from now on."""
    return _rev


def note_error(text):
    """Record a failed turn in the thread itself.

    A turn that dies in a background task would otherwise just stop growing the
    conversation, which reads as a hung session rather than a failure.
    """
    return _new_item(
        "message", role="assistant", blocks=[{"type": "text", "text": text}], error=True
    )


# --------------------------------------------------------------------------- #
# Content translation
# --------------------------------------------------------------------------- #


def _block(content):
    """One ACP content block as the pane renders it, or None to drop it.

    The pane shows text and pictures. ACP can also carry a diff or a terminal
    handle inside a tool call; those are named rather than rendered, because a
    chat column is not a diff viewer and silently dropping them would make a
    tool look like it did nothing.
    """
    if content is None:
        return None
    data = content if isinstance(content, dict) else _dump(content)
    kind = data.get("type")
    if kind == "content":  # ContentToolCallContent wraps the real block
        return _block(data.get("content"))
    if kind == "text":
        return {"type": "text", "text": data.get("text") or ""}
    if kind == "image":
        return {
            "type": "image",
            "data": data.get("data") or "",
            "mime": data.get("mimeType") or "image/png",
        }
    if kind == "diff":
        return {"type": "text", "text": f"(edited {data.get('path') or 'a file'})"}
    if kind == "terminal":
        return {"type": "text", "text": "(ran a terminal command)"}
    return None


def _blocks(contents):
    out = []
    for c in contents or ():
        b = _block(c)
        if b is not None:
            out.append(b)
    return out


def _dump(model):
    """A pydantic ACP model as a plain dict in wire spelling."""
    return model.model_dump(by_alias=True, exclude_none=True)


# --------------------------------------------------------------------------- #
# The pipe transport
# --------------------------------------------------------------------------- #


class _PipeTransport:
    """``acp.Transport`` over a plain ``Popen``'s stdin/stdout.

    A reader thread does the blocking read and hands decoded messages to the
    loop; ``send`` writes under a lock from a worker thread. Both stay off the
    event loop, and neither needs asyncio's pipe support -- which the Windows
    Selector loop this server runs on does not have.
    """

    def __init__(self, proc, loop):
        self._proc = proc
        self._loop = loop
        self._queue = asyncio.Queue()
        self._write_lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_loop, name="acp-agent-reader", daemon=True
        )
        self._reader.start()

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    # The harness owns its stdout discipline; a stray line is
                    # its bug, and dropping it beats killing the session.
                    logger.warning("non-JSON line from ACP agent: %.200s", line)
                    continue
                self._offer(msg)
        except Exception:  # noqa: BLE001 - a dead pipe is EOF, not a crash
            logger.debug("ACP agent reader ended", exc_info=True)
        finally:
            self._offer(None)

    def _offer(self, msg):
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)
        except RuntimeError:
            pass  # loop already closed; nothing left to deliver to

    async def send(self, message):
        await asyncio.get_running_loop().run_in_executor(None, self._send, message)

    def _send(self, message):
        if self._closed:
            raise RuntimeError("ACP agent is gone")
        with self._write_lock:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()

    async def receive(self):
        return await self._queue.get()

    async def close(self):
        self._closed = True
        try:
            self._proc.stdin.close()
        except Exception:  # noqa: BLE001 - closing a dead pipe is success
            pass


# --------------------------------------------------------------------------- #
# The client half of the protocol
# --------------------------------------------------------------------------- #


class _PaneClient:
    """What the harness can ask *us* for.

    Deliberately close to nothing. We are a chat pane, not an editor: the
    filesystem and terminal capabilities are declined at ``initialize``, so a
    harness that touches files does it with its own tools, under its own
    permission model, and we neither pretend to mediate that nor lie about it.
    What is implemented is the part that has a place in a chat column --
    the transcript, and a permission question.
    """

    def __init__(self, permission_mode):
        self._permission_mode = permission_mode

    async def session_update(self, session_id, update, **kwargs):
        data = update if isinstance(update, dict) else _dump(update)
        kind = data.get("sessionUpdate")
        if kind in ("agent_message_chunk", "user_message_chunk"):
            role = "assistant" if kind.startswith("agent") else "user"
            _append_chunk(role, data.get("messageId"), _block(data.get("content")))
        elif kind == "agent_thought_chunk":
            # Dropped on purpose. Thinking is the model talking to itself, and a
            # narrow column reading someone else's notes is how the answer gets
            # lost. The harness's own UI is where that belongs.
            pass
        elif kind == "tool_call":
            _new_item(
                "tool_call",
                tool_call_id=data.get("toolCallId"),
                title=data.get("title") or "",
                status=data.get("status") or "pending",
                blocks=_blocks(data.get("content")),
            )
        elif kind == "tool_call_update":
            item = _find_tool_call(data.get("toolCallId"))
            if item is not None:
                blocks = _blocks(data.get("content"))
                _touch(
                    item,
                    title=data.get("title"),
                    status=data.get("status"),
                    blocks=blocks or None,
                )
        elif kind == "available_commands_update":
            _set_commands(data.get("availableCommands") or [])
        elif kind == "usage_update":
            _set_usage(data)
        elif kind == "config_option_update":
            # The harness changing its own mind -- a model picked in its TUI, or
            # one swapped out from under a session. Taken rather than ignored so
            # the pane names the model that is actually answering.
            _note_config_options(data.get("configOptions"))

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

        opts = [o if isinstance(o, dict) else _dump(o) for o in options or ()]
        if self._permission_mode == "allow":
            pick = _first_allow(opts)
            if pick is not None:
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=pick)
                )
        call = tool_call if isinstance(tool_call, dict) else _dump(tool_call)
        chosen = await _ask_permission(call, opts)
        if chosen is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=chosen)
        )

    # The capabilities we declined. Implemented because the protocol object
    # requires them, and refusing loudly beats a hang if a harness asks anyway.
    async def write_text_file(self, *a, **k):
        raise _unsupported("write_text_file")

    async def read_text_file(self, *a, **k):
        raise _unsupported("read_text_file")

    async def create_terminal(self, *a, **k):
        raise _unsupported("create_terminal")

    async def terminal_output(self, *a, **k):
        raise _unsupported("terminal_output")

    async def release_terminal(self, *a, **k):
        raise _unsupported("release_terminal")

    async def wait_for_terminal_exit(self, *a, **k):
        raise _unsupported("wait_for_terminal_exit")

    async def kill_terminal(self, *a, **k):
        raise _unsupported("kill_terminal")


def _unsupported(method):
    import acp

    return acp.RequestError.method_not_found(
        f"biopb's chat pane does not provide {method}: it is a chat surface, "
        "not an editor."
    )


def _first_allow(options):
    for opt in options:
        if str(opt.get("kind") or "").startswith("allow"):
            return opt.get("optionId")
    return options[0].get("optionId") if options else None


# --------------------------------------------------------------------------- #
# Streaming assembly
# --------------------------------------------------------------------------- #

# The message an assistant chunk is currently landing in, keyed by ACP's
# messageId. Chunks arrive a few characters at a time and many share an id;
# appending each as its own item would render one sentence as thirty.
_open_message = {}


def _append_chunk(role, message_id, block):
    if block is None:
        return
    key = (role, message_id)
    item = _open_message.get(key)
    if item is None or item not in _items:
        item = _new_item("message", role=role, blocks=[block])
        _open_message.clear()
        _open_message[key] = item
        return
    blocks = item["blocks"]
    if blocks and blocks[-1]["type"] == "text" and block["type"] == "text":
        blocks[-1] = {"type": "text", "text": blocks[-1]["text"] + block["text"]}
    else:
        blocks.append(block)
    _touch(item)


def _find_tool_call(tool_call_id):
    for item in reversed(_items):
        if item["kind"] == "tool_call" and item.get("tool_call_id") == tool_call_id:
            return item
    return None


# --------------------------------------------------------------------------- #
# Session-wide agent facts a view wants to show
# --------------------------------------------------------------------------- #

# What the harness says it can do, which is not ours to invent. Commands are
# advertised by notification and can change mid-session, so they ride the
# history read rather than the once-probed status.
_commands: list = []
_usage: dict = {}


def _set_commands(commands):
    global _commands
    _commands = [
        {
            "name": c.get("name") or "",
            "description": c.get("description") or "",
            "hint": (c.get("input") or {}).get("hint") or "",
        }
        for c in commands
        if c.get("name")
    ]


def _set_usage(data):
    global _usage
    _usage = {
        "used": data.get("used"),
        "size": data.get("size"),
        "cost": (data.get("cost") or {}).get("amount"),
    }


def commands():
    return list(_commands)


def usage():
    return dict(_usage)


# The models the harness offers and the one this session is on. Recorded when
# the session opens and refreshed from every answer that carries the options,
# so `/model` can list them without a round trip and a model changed inside the
# harness is not reported back as the one we last set.
_model_choices: list = []
_model_current = None


def _flatten_select(option):
    """The values a select option offers, groups flattened away.

    A harness may send a flat list or one grouped by provider (opencode groups),
    and the difference is presentation, not meaning: a `/model` list is a list.
    Getting this wrong is not cosmetic -- reading a group as an option yields a
    value of None, which is what a validity check would then compare against.
    """
    out = []
    for entry in option.get("options") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("group") is not None or (
            "value" not in entry and entry.get("options")
        ):
            out.extend(e for e in entry.get("options") or [] if isinstance(e, dict))
        else:
            out.append(entry)
    return [
        {"value": e["value"], "name": e.get("name") or e["value"]}
        for e in out
        if e.get("value")
    ]


def _note_config_options(options):
    """Take the session's config options as the harness last stated them."""
    global _model_choices, _model_current
    for raw in options or ():
        opt = raw if isinstance(raw, dict) else _dump(raw)
        if opt.get("id") != "model":
            continue
        _model_choices = _flatten_select(opt)
        if opt.get("currentValue") is not None:
            _model_current = opt.get("currentValue")
        return


def model_choices():
    return list(_model_choices)


def current_model():
    return _model_current


async def set_model(value):
    """Point the running session at *value*.

    Runtime rather than a respawn, which is the whole reason the model is not in
    the pinned environment (:func:`_pinned_config`): changing model should not
    cost the conversation.

    Raises ``ValueError`` when the harness does not offer the model. Checked
    here against what it advertised, so a typo says so in the pane instead of
    failing at the provider on the next turn -- and unchecked when it advertised
    nothing, because then we have nothing to check against and refusing would
    withhold a model that works.
    """
    global _model_current
    if _conn is None or _session_id is None:
        raise ValueError("the agent is not running yet")
    known = [c["value"] for c in _model_choices]
    if known and value not in known:
        raise ValueError(
            f"{_agent_name or 'the agent'} does not offer {value!r}. "
            f"Offered: {', '.join(known[:8])}" + ("..." if len(known) > 8 else "")
        )
    reply = await _conn.set_config_option(
        config_id="model", session_id=_session_id, value=value
    )
    _model_current = value
    # The answer carries the options as they now stand; taking them keeps the
    # list honest when setting one model changes what the others are.
    _note_config_options(getattr(reply, "config_options", None))
    return value


async def choose_model(value, config):
    """Set the model a person just typed, starting the agent if it is not up.

    The agent is otherwise started by the first turn, and before that there is
    no session -- so no advertised list, so nothing to check a name against.
    ACP has no session-less way to ask: ``config_options`` rides
    ``session/new``, ``session/load``, ``session/fork`` and the set call itself,
    and nothing else. A name accepted unchecked is then applied at spawn, found
    wanting, and silently replaced by the harness's default, which is a pane
    that reported a model it is not using.

    So this starts it. Cheap next to the alternative, and it does not take the
    kernel with it: the one-agent claim is made when a client *runs code*
    (``_server._presume_claim``, on submit), not when one connects, so a switch
    back to the built-in loop is still available afterwards.
    """
    async with _lock():
        await ensure_agent(config)
        return await set_model(value)


# --------------------------------------------------------------------------- #
# Permission questions in flight
# --------------------------------------------------------------------------- #

# request id -> (future, item). A question is a thread item so it renders where
# it was asked, and a future so the protocol call can wait on the answer.
_pending: dict = {}
_permission_seq = 0


async def _ask_permission(tool_call, options):
    """Put the question in the thread and wait. None means cancelled."""
    global _permission_seq
    _permission_seq += 1
    request_id = f"p-{_permission_seq}"
    item = _new_item(
        "permission",
        request_id=request_id,
        title=tool_call.get("title") or tool_call.get("toolCallId") or "run something",
        # What kind of action it is, which the title alone does not say: an
        # edit's title is a bare file path, and "src/x.py" reads very
        # differently once you know it is about to be written.
        tool_kind=tool_call.get("kind") or "",
        options=[
            {
                "id": o.get("optionId"),
                "name": o.get("name") or o.get("optionId"),
                "kind": o.get("kind") or "",
            }
            for o in options
        ],
        outcome=None,
    )
    future = asyncio.get_running_loop().create_future()
    _pending[request_id] = (future, item)
    try:
        return await future
    except asyncio.CancelledError:
        _touch(item, outcome="cancelled")
        raise
    finally:
        _pending.pop(request_id, None)


def answer_permission(request_id, option_id):
    """Answer a pending question. False if it is not (or no longer) open."""
    entry = _pending.get(request_id)
    if entry is None:
        return False
    future, item = entry
    if future.done():
        return False
    valid = {o["id"] for o in item["options"]}
    if option_id is not None and option_id not in valid:
        return False
    _touch(item, outcome=option_id or "cancelled")
    future.set_result(option_id)
    return True


def _cancel_pending():
    """Answer every open question with 'cancelled'.

    A cancelled turn leaves the harness waiting on a reply it will never be
    asked for again; the protocol says to settle them rather than drop them.
    """
    for request_id in list(_pending):
        entry = _pending.get(request_id)
        if entry is None:
            continue
        future, item = entry
        if not future.done():
            _touch(item, outcome="cancelled")
            future.set_result(None)


# --------------------------------------------------------------------------- #
# Resolving and launching the harness
# --------------------------------------------------------------------------- #


def resolve_command(config):
    """``(argv, name)`` for the configured harness.

    Raises :class:`AcpNotConfigured` when the agent is unknown or its binary is
    not installed. Note this asks a different question from
    ``biopb._agents.status``: that one reads a *client's config file* to see
    whether biopb is registered with it, this one asks whether a CLI exists to
    run. A user can have Cursor registered and no ACP harness, or the reverse.
    """
    from .._config import get_setting

    agent_id = get_setting(config, "chat.acp_agent")
    spec = _ACP_AGENTS.get(agent_id)
    if spec is None:
        raise AcpNotConfigured(
            f"Unknown ACP agent {agent_id!r}. Supported: "
            + ", ".join(sorted(_ACP_AGENTS))
        )
    override = (get_setting(config, "chat.acp_command") or "").strip()
    if override:
        if not Path(override).exists():
            raise AcpNotConfigured(f"chat.acp_command points at nothing: {override}")
        return (override, *spec["argv"]), spec["name"]
    found = shutil.which(spec["name"])
    if not found:
        for candidate in spec["extra_paths"]:
            path = Path(candidate).expanduser()
            if path.exists():
                found = str(path)
                break
    if not found:
        raise AcpNotConfigured(
            f"{spec['name']} is not installed, or is not on this process's PATH. "
            f"Install it, or set chat.acp_command to its full path."
        )
    return (found, *spec["argv"]), spec["name"]


def check_ready(config):
    """Raise :class:`AcpNotConfigured` unless a turn could be started.

    Asked before a turn is accepted rather than at spawn time, so a misconfigured
    install says so instead of taking the user's message and failing later.
    """
    resolve_command(config)


# The running harness, or None. Module state for the same reason the transcript
# is: one session, one agent, however many browser windows are watching.
_child = None
_conn = None
_session_id = None
_agent_name = ""
_cwd = None


def _agent_log_path():
    """Where the harness's own stderr goes.

    Beside the per-session shim logs and named for this process, so a session's
    agent log sits next to the session log a reader is already looking at.
    """
    from .._config import get_session_log_dir

    return get_session_log_dir() / f"acp-{os.getpid()}.log"


def _spawn(config):
    """Start the harness and return an owned child speaking JSON-RPC on pipes."""
    from biopb._lifecycle import winjob
    from biopb._lifecycle.owned_child import OwnedChild, open_child_log

    argv, name = resolve_command(config)
    env = {**os.environ, **_agent_env(config)}
    log, _ = open_child_log(_agent_log_path())
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(  # noqa: S603 - argv is resolved, never a shell
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            cwd=_cwd,
            env=env,
            text=True,
            bufsize=1,
            close_fds=True,
            **kwargs,
        )
    finally:
        log.close()
    job = None
    if os.name == "nt":
        # OwnedChild.adopt does not make one, and without it a force-killed
        # session leaves the harness (and its own children) running.
        try:
            job = winjob.create_kill_on_close_job()
            winjob.assign_process(job, proc)
        except Exception:  # noqa: BLE001 - best effort, as in _shim
            logger.debug("could not put the ACP agent in a job object", exc_info=True)
    return OwnedChild.adopt(proc, job), name


def _pinned_config(config):
    """The settings biopb fixes on the harness for the life of the process.

    Only what must not move once the agent is running. Permissions are here
    because they are a promise to the person watching: a policy the agent could
    change mid-session -- by editing a config in the directory it writes to --
    is not a policy.

    The model is deliberately **not** here. It is a choice, not a guarantee, and
    a choice the user should be able to change without losing the conversation;
    ACP's ``session/set_config_option`` does that at runtime (see
    :func:`_apply_model`), and anything an eventual ``/model`` command drives
    belongs on the same seam. Pinning it in the environment would mean
    respawning the agent to change it.
    """
    from .._config import get_setting

    spec = _ACP_AGENTS.get(get_setting(config, "chat.acp_agent")) or {}
    body = {}
    # Unconditional: two biopb servers means two viewers, which is a wrong
    # session rather than a preference about one.
    suppress = spec.get("suppress_mcp") or ()
    if suppress:
        body["mcp"] = {name: {"enabled": False} for name in suppress}
    if get_setting(config, "chat.acp_permission") == "ask":
        # Absent under "allow": that answer means do not interfere, including
        # not overriding a stricter choice the user made in their own config.
        permission = spec.get("strict_permission")
        if permission:
            body["permission"] = dict(permission)
    return body


def _agent_env(config):
    """Environment for the harness: this process's, plus what biopb pins.

    Empty when there is nothing to pin, so a harness we know no settings for is
    launched exactly as the user's shell would launch it.
    """
    from .._config import get_setting

    spec = _ACP_AGENTS.get(get_setting(config, "chat.acp_agent")) or {}
    var = spec.get("config_env")
    body = _pinned_config(config)
    if not var or not body:
        return {}
    return {var: json.dumps(body)}


async def ensure_agent(config):
    """Start the harness and open a session, once.

    Lazy, like the dask cluster: a viewer whose owner never opens the chat pane
    should not have a second agent process sitting in their session.
    """
    global _child, _conn, _session_id, _agent_name, _cwd
    if _conn is not None and _child is not None and _child.alive():
        return
    if _child is not None:
        # It died. Drop the wreckage before building on the same names.
        await shutdown()

    import tempfile

    import acp
    from acp.schema import (
        ClientCapabilities,
        FileSystemCapabilities,
        HttpMcpServer,
        Implementation,
    )

    from .._config import get_setting

    if _cwd is None:
        # The harness insists on a working directory and will read files under
        # it. It gets an empty one of its own rather than wherever the viewer
        # happened to be launched from -- which is frequently the user's home.
        _cwd = tempfile.mkdtemp(prefix="biopb-mcp-acp-")

    child = conn = None
    globals_owned = False
    try:
        child, name = _spawn(config)
        transport = _PipeTransport(child.proc, asyncio.get_running_loop())
        client = _PaneClient(get_setting(config, "chat.acp_permission"))
        conn = acp.connect_to_agent(client, transport)

        init = await asyncio.wait_for(
            conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapabilities(readTextFile=False, writeTextFile=False),
                    terminal=False,
                ),
                client_info=Implementation(name="biopb-chat", version="1"),
            ),
            _START_TIMEOUT,
        )

        mcp_url = _own_mcp_url()
        servers = []
        if mcp_url:
            # headers is required, not optional: both this library and opencode
            # reject the entry without it, whatever the spec's example suggests.
            servers.append(
                HttpMcpServer(type="http", name="biopb", url=mcp_url, headers=[])
            )
        else:
            logger.warning(
                "no /mcp url for this session; the ACP agent will start without "
                "biopb's tools"
            )

        try:
            session = await conn.new_session(cwd=_cwd, mcp_servers=servers)
        except Exception as exc:  # noqa: BLE001 - auth is a normal first-run answer
            method_id = _auth_method(init)
            if method_id is None:
                raise
            logger.info("ACP agent wants authentication (%s); retrying", method_id)
            await conn.authenticate(method_id=method_id)
            session = await conn.new_session(cwd=_cwd, mcp_servers=servers)
            del exc

        # _apply_model uses set_model(), which reads the module globals. Publish
        # provisional ownership before it runs, but retain rollback responsibility
        # until model setup has also completed.
        _child, _conn, _agent_name = child, conn, name
        _session_id = session.session_id
        globals_owned = True
        await _apply_model(session, get_setting(config, "chat.acp_model"))
        child = conn = None  # ownership is now fully transferred
        logger.info("ACP agent %s ready (session %s)", name, _session_id)
    except BaseException:
        if globals_owned:
            await shutdown()
        else:
            if conn is not None:
                try:
                    await conn.close()
                except Exception:  # noqa: BLE001 - preserve the startup error
                    logger.debug("closing the ACP connection failed", exc_info=True)
            _stop_child(child)
        raise


async def _apply_model(session, wanted):
    """Point the new session at the configured model, if one was named.

    A fresh session takes the harness's default, and the default is neither
    ours nor the user's: opencode's is a hosted model whose endpoint can simply
    be down, which surfaces as a turn that fails at the provider with nothing in
    the pane to act on. This is the same position ``chat.model`` takes for the
    built-in loop -- name the model, do not inherit one.

    Reported and swallowed rather than raised. A model that will not set is a
    session that still works, on the default; refusing to open the pane over it
    would trade a degraded chat for no chat.
    """
    global _model_choices, _model_current
    _model_choices, _model_current = [], None
    _note_config_options(getattr(session, "config_options", None))
    if not wanted:
        return
    if not _model_choices and _model_current is None:
        logger.warning(
            "chat.acp_model is set but %s exposes no model setting; using its default",
            _agent_name,
        )
        return
    try:
        await set_model(wanted)
        logger.info("ACP session model set to %s", wanted)
    except ValueError as exc:
        # A typo in the config file is not worth refusing to open the pane over:
        # the session still works, on the harness's default. But it is worth
        # *saying* -- the only other sign is the model name in the header
        # quietly becoming one the user did not choose, which reads as the pane
        # being wrong rather than as a setting being.
        logger.warning("%s; using its default", exc)
        running = _model_current or "its default"
        note_error(
            f"chat.acp_model is {wanted!r}, which {_agent_name or 'the agent'} "
            f"does not offer. Answering as {running} instead — "
            "/model lists what it does offer."
        )
    except Exception as exc:  # noqa: BLE001 - a default model still works
        logger.warning("could not set the ACP model to %r: %s", wanted, exc)


def _auth_method(init):
    """The first advertised auth method id, or None if the agent wants none."""
    methods = getattr(init, "auth_methods", None) or []
    for method in methods:
        ident = getattr(method, "id", None) or getattr(method, "method_id", None)
        if ident:
            return ident
    return None


def _own_mcp_url():
    """This session's ``/mcp``, as the agent should dial it.

    Read from the registry record this session published rather than
    reconstructed, so there is one answer to "where is this session" and the
    agent gets the same one every other client does.
    """
    from biopb import _sessions

    try:
        for record in _sessions.list_sessions(prune=False):
            if record.get("pid") == os.getpid():
                return record.get("mcp_url")
    except Exception:  # noqa: BLE001 - a missing registry is not fatal here
        logger.debug("could not read the session registry", exc_info=True)
    return None


def _drop_session():
    """Forget the live session, returning its ``(conn, child)`` to be reaped.

    Clearing the globals *before* the teardown, and in one place, is what makes
    both callers idempotent: a second entrant finds nothing to stop. The model
    goes with them -- left behind it would name a model nothing is running on.
    """
    global _child, _conn, _session_id, _model_choices, _model_current
    conn, child = _conn, _child
    _conn, _child, _session_id = None, None, None
    _model_choices, _model_current = [], None
    return conn, child


def _stop_child(child):
    """Reap the harness process, best-effort -- we are tearing down either way."""
    if child is not None:
        try:
            child.stop()
        except Exception:  # noqa: BLE001
            logger.debug("stopping the ACP agent failed", exc_info=True)


async def shutdown():
    """Stop the harness. Idempotent, and safe to call on a dead one."""
    _cancel_pending()
    conn, child = _drop_session()
    if conn is not None:
        try:
            await conn.close()
        except Exception:  # noqa: BLE001 - we are tearing down either way
            logger.debug("closing the ACP connection failed", exc_info=True)
    _stop_child(child)


def stop_sync():
    """Reap the harness from a non-async caller (``atexit``, ``_shutdown``).

    No ``conn.close()``: that is a coroutine and there is no loop to run it on.
    The child's death closes the pipe underneath it anyway.
    """
    _stop_child(_drop_session()[1])


def cleanup_cwd():
    global _cwd
    if _cwd:
        shutil.rmtree(_cwd, ignore_errors=True)
        _cwd = None


# --------------------------------------------------------------------------- #
# Turns
# --------------------------------------------------------------------------- #


class TurnInProgress(RuntimeError):
    """A turn was asked for while one was running. Refused, not queued.

    Same call as ``_chat``: a queued turn would be composed against a
    conversation its sender has not seen the end of.
    """


_turn_lock = None
_turn_task = None


def _lock():
    global _turn_lock
    if _turn_lock is None:
        _turn_lock = asyncio.Lock()
    return _turn_lock


def busy():
    return _turn_lock is not None and _turn_lock.locked()


def agent_name():
    return _agent_name


def session_started():
    return _session_id is not None


async def run_turn(text, config):
    """Send one prompt and stream the answer into the thread.

    The harness runs its own loop -- tools, retries, subagents -- so this is
    nothing like ``_chat.run_turn``: it hands over a prompt, and everything the
    view renders arrives as notifications while this waits.
    """
    if busy():
        raise TurnInProgress("a turn is already running in this session")
    async with _lock():
        await ensure_agent(config)
        _new_item("message", role="user", blocks=[{"type": "text", "text": text}])
        from acp.schema import TextContentBlock

        try:
            result = await asyncio.wait_for(
                _conn.prompt(
                    session_id=_session_id,
                    prompt=[TextContentBlock(type="text", text=text)],
                ),
                _TURN_TIMEOUT,
            )
        except asyncio.CancelledError:
            await _cancel_turn()
            _new_item(
                "message",
                role="assistant",
                blocks=[{"type": "text", "text": "Stopped."}],
                cancelled=True,
            )
            raise
        except asyncio.TimeoutError:
            await _cancel_turn()
            note_error(
                "The agent did not finish within an hour and was stopped. Its "
                "log is beside this session's."
            )
            return
        _note_stop(getattr(result, "stop_reason", None))


def _note_stop(stop_reason):
    """Say why a turn ended when the reason is not "it answered".

    ``end_turn`` needs no line -- the answer is the answer. The others are
    states a reader would otherwise have to infer from a conversation that
    simply stopped.
    """
    said = {
        "max_tokens": "The agent hit its token limit for this turn.",
        "max_turn_requests": "The agent hit its own limit on tool rounds.",
        "refusal": "The agent declined to continue.",
    }.get(str(stop_reason or ""))
    if said:
        note_error(said)


async def _cancel_turn():
    _cancel_pending()
    if _conn is not None and _session_id is not None:
        try:
            await _conn.cancel(session_id=_session_id)
        except Exception:  # noqa: BLE001 - the turn is over either way
            logger.debug("cancelling the ACP turn failed", exc_info=True)


def set_turn_task(task):
    """Hold the running turn's task so a cancel has something to cancel."""
    global _turn_task
    _turn_task = task


def turn_task():
    return _turn_task


async def cancel():
    """Stop the running turn.

    Cancelling nothing is a success: what actually happened shows up in the
    thread on the next poll, the way the console reports an interrupt.
    """
    task = _turn_task
    if task is not None and not task.done():
        task.cancel()
        return
    await _cancel_turn()


async def reset():
    """Start a new conversation, and a new session at the harness.

    Clearing our transcript alone would leave the agent composing against a
    history nobody can see any more -- the opposite of what "new conversation"
    means to the person who asked for it.
    """
    global _reset_rev
    if busy():
        raise TurnInProgress("a turn is running in this session")
    _cancel_pending()
    _items.clear()
    _open_message.clear()
    _usage.clear()
    # Ids are *not* restarted, for the reason `_chat.reset` gives: a view's
    # stale cursor must not match an item it has never seen.
    _reset_rev = _bump()
    await shutdown()
