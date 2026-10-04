# Attaching an agent to an existing session

Phases 1 and 2 are implemented; phase 3 is not.

An agent can attach to a session that is already running instead of getting its
own. The control launches the session and the user ends it; the stdio shim
becomes a stateless attach client. Local attach comes first. Remote attach
follows once the security model for `/mcp` is settled.

## Where the shim lives

The shim is part of the SDK: `biopb._shim`, run as `biopb-shim` (extra
`biopb[shim]`, which needs only the `mcp` package). It imports no session code, so
an agent host needs the SDK and not the kernel stack. `biopb-mcp --transport stdio`
runs the same code. The registration helper (`biopb._agents`) registers
`biopb-shim`. Shim and session meet only over HTTP and the session registry:
`/api/lease` (acquire, renew, release), `/api/status`, `/mcp`, the control's
`/api/sessions` and `/api/sessions/new`, and the records in `biopb._sessions`. A
change to any of them is a change to both packages.

## Rules

- **One agent per session.** A session accepts one attach at a time.
- **Chat and agent are exclusive.** One lease per session, held by an agent or by
  chat, never both.
- **Explicit session.** The agent names a session id. With no id it gets the list
  of sessions and their busy status. `new` asks for a new session.
- **One token for all sessions.** Remote access uses the existing server token;
  there are no per-session grants yet.

## Ownership

The control launches sessions and does not own them. Sessions stay detached and
self-registering, so a control restart never ends one, and the registry
(`biopb._sessions`) stays the source of truth, pruned by pid identity. A session
runs until a person stops it from the dashboard; nothing ends one on a timer.

The shim owns no session: it never spawns one and never ends one. Its exit is a
detach. What it does own is itself: a shim that outlived its client would keep
renewing its lease and lock the session, so it releases and exits when the client
goes (stdin EOF, SIGTERM/SIGHUP, and on Windows a watchdog on the client).

## The lease

One lease per session, held by one **holder** of kind `agent` or `chat`. The
session holds it, not the control: the shim reaches a session directly on its
loopback port, and the control is not on that path. Two holders never coexist, so
chat needs no separate gate and no knowledge of whether an agent is attached.

- **Agent.** Acquired by `attach`, renewed while the shim is attached, expired by
  the session after a short silence. Streamable-http is request-based, so a
  dropped connection is not a reliable release signal; a crashed or `kill -9`'d
  shim frees the session when its lease lapses.
- **Chat.** Acquired lazily by the first turn, not by opening the pane, so a
  session with a chat pane open stays attachable. Renewed by turns only, never by
  the pane's history poll, or an idle open pane would hold the session
  indefinitely. Released on idle expiry (minutes after the last turn ends), on
  clear, and by an explicit release in the pane.
- **Refusals.** `attach` on a session with a live lease answers with the holder's
  kind and age and a `force` hint. A chat turn on a session held by an agent
  answers `409` with the same facts. Reading chat history needs no lease.
- **Force.** An explicit `force` takes the lease. Taking it from a chat that is
  mid-turn cancels the turn; nothing else ever aborts a turn.
- **Busy status** in the session list is the lease: `free`, `agent`, or `chat`.
- **Reattach.** The holder is identified by a token it keeps, not by the
  connection, so a shim that restarts reclaims its own lease.

## Chat

The chat routes are always mounted; the lease decides whether a turn runs. The
status endpoint reports the holder, so the pane can show "agent attached" and
stay read-only. The turn lock stays, serializing turns within a chat holder.
Releasing or losing the chat lease clears `_pending_user` and `_noted_jobs`, so
queued messages and activity notes do not replay against a kernel the agent
changed.

A human on the napari console or a Jupyter client may still share the kernel with
the holder. That is existing behavior.

## Selecting a session

The shim starts unbound. It has one local tool, `attach(session)`, with a
session id or `new`:

- No id: the tool error lists live sessions as `id, busy|free, viewer|no viewer`.
- Id: take the lease, then proxy every other request to it.
- `new`: asks the control to launch a session for this client, leased from
  birth. It keeps running after the agent disconnects, so it can be attached to
  again; the user stops it from the dashboard. If no control answers, that is an
  error naming the control's log: a session started without one has no data
  plane, so attaching to it would succeed and then fail on first use.
- `--session <id|new|auto>` (or `$BIOPB_SESSION`) binds before the handshake;
  see below.

While unbound the shim lists `attach` alone, with empty resources and prompts,
and tool calls other than `attach` return the session list. It imports no session
code and holds no copy of any session's surface. Once bound, the tool, resource
and prompt lists come from the session, and so do the `instructions`, which are
composed per session (viewer or not, config). `instructions` cannot be resent
after the handshake, so `attach` returns them as its result, and attaching again
to the same session returns them again, for an agent whose context no longer
holds them. The handshake declares `listChanged`, and the shim sends it on attach
and when an attachment ends.

**Clients that do not follow `list_changed`.** Claude Code and opencode refresh
their tool list after `attach`; Codex does not, within a turn. For such a client
the registration passes `--session` (`biopb agents register` writes `--session auto` into Codex's entry and nothing
extra into the others; an entry registered before that, or for `biopb-mcp`, reads
as drifted, so the dashboard offers a Re-register): the shim binds *before* it answers
`initialize`, so the answer carries the session's own instructions -- in the slot
a client puts in front of the model from the first turn -- and its tools, and
nothing needs refreshing. `auto` takes the newest free session, else has the
control launch one. The client cannot choose a session, which is the price. The
bind happens inside the client's startup window, so a cold control start plus a
launch can approach its timeout; `auto` prefers an existing session because that
is nearly instant. A bind that fails is retried on the first request
that needs a session, and the handshake says why it failed.

## Phases

1. **Local attach.** The lease in the session (agent and chat holders), the
   `attach` tool, direct loopback connection. No ownership change.
2. **Control-launched sessions.** A session has a named mode (`durable`, `direct`;
   `mcp/_session_mode.py`) in place of the `agentless`/`shim_owned` inference.
   `attach new` goes through `POST /api/sessions/new`, carrying the client's
   allowlisted display variables in place of the control's own, and the shim no
   longer spawns, reaps or owns any session. There is no idle reaper: a session's kernel can be worth keeping after its agent has
   gone, so a session runs until a person stops it, and every session serves the
   stop route so every dashboard row can be stopped. A shim whose session is
   stopped sees it stop answering and unbinds. Control invariant I1 is rewritten
   to match.
3. **Remote.** `/mcp` proxied by the control under the token, with the Host and
   Origin guards. `/mcp` runs arbitrary code in a kernel, so it is proxied only
   when a token is enforced. Sessions stay loopback-bound; the control is the only
   public listener. The viewer, screenshots, filesystem paths and the data-plane
   endpoint all belong to the session's host, not the client's.

## Open

- The chat idle expiry: long enough not to drop a user mid-thought, short enough
  that an agent is not locked out after the user walks away.
- Whether the lease needs a heartbeat tool call or can ride on ordinary requests.
- Per-session grants, if one token for all sessions stops being enough.
