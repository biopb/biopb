# Attaching an agent to an existing session

Phases 1 and 2 are implemented; phase 3 is not.

An agent can attach to a session that is already running instead of getting its
own. The control becomes the authority for session lifecycle; the stdio shim
becomes a stateless attach client. Local attach comes first. Remote attach
follows once the security model for `/mcp` is settled.

## Rules

- **One agent per session.** A session accepts one attach at a time.
- **Chat and agent are exclusive.** One lease per session, held by an agent or by
  chat, never both.
- **Explicit session.** The agent names a session id. With no id it gets the list
  of sessions and their busy status. `new` asks for a new session.
- **One token for all sessions.** Remote access uses the existing server token;
  there are no per-session grants yet.

## Ownership

The control owns lifecycle *policy and API*: launch, stop, idle expiry. It does
not own process parentage. Sessions stay detached and self-registering, so a
control restart never ends one, and the registry (`biopb._sessions`) stays the
source of truth, pruned by pid identity.

The shim owns nothing. Its exit is a detach, never a reap. This replaces the
process-group, Job Object and client-death-watchdog teardown for attached
sessions.

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
- `new`: asks the control to launch an ephemeral session for this client,
  leased from birth. If no control answers, that is an error naming the control's
  log: a session started without one has no data plane, so attaching to it would
  succeed and then fail on first use. The one exception is a client that pins its
  data plane in its environment (`BIOPB_TENSOR_*`) and so runs without the
  control's: a control-launched session would not see the pin, so the shim spawns
  a session of its own and reaps it, as it always did.
- `--session <id>` (or env) pre-binds for people and scripts.

While unbound the shim answers the handshake and list requests from the imported
server, as now, and tool calls other than `attach` return the session list. Once
bound, the list and `instructions` come from the session, because they are
composed per session (viewer or not, config) and the session may be a different
version from the shim. That needs a list-changed notification, which the bridge
does not forward today.

## Phases

1. **Local attach.** The lease in the session (agent and chat holders), the
   `attach` tool, direct loopback connection. No ownership change.
2. **Control-owned lifecycle.** A session has a named mode (`shim`, `durable`,
   `ephemeral`, `direct`; `mcp/_session_mode.py`) that decides its stop route,
   chat and registration, in place of the `agentless`/`shim_owned` inference.
   `attach new` goes through `POST /api/sessions/new?ephemeral=1`, carrying the
   client's allowlisted display variables in place of the control's own. The
   control ends an ephemeral session once its lease has been free for the grace
   period, by asking it to end itself; `POST /api/sessions/<id>/stop` does the same
   on request, refusing a held session unless forced. Control invariant I1 is
   rewritten to match.
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
