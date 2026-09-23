# The user console — a second writer in the kernel

A code cell on the observe page (`/session/<id>/observe`) lets a **human**
run code in the session kernel, in the same namespace, through the same job
runner (`_jobs.submit()`) the agent's `execute_code` uses.

**Guarantee:** the human and the agent are serialized against each other —
one job at a time, no preemption, no queue — and neither can act on the
other's running job without the other finding out.

## Gating: the console is its own path root

An execute route adds no authority at the session child itself: it already
serves `/mcp` (i.e. `execute_code`) on the same loopback port behind the
same Host/Origin guard, so anything that can reach the child's port already
has RCE. What it changes is the **control's** proxy allowlist
(`_SESSION_ALLOWED_ROOTS`), whose whole purpose is to keep `/mcp` off the
public origin. So the console is a separate root with its own gate:

| Root | Proxied | Gate |
|---|---|---|
| `/session/<id>/api/*` | always | `_ControlAuthMiddleware` (token or loopback Host; CSRF on unsafe verbs) |
| `/session/<id>/console/*` | **local mode only** | the above, plus a public-bind refusal and a JSON content-type requirement |
| `/session/<id>/mcp` | never | agents reach the child directly |

The gate is control-side because the proxy hop strips Host/Origin toward the
child, so only the control knows whether it is loopback-bound or
`--remote` (the child's own `_check_origin` passes trivially on every
proxied request). `serve_control_api` passes its bind `host` through
`_web_auth.host_is_public_bind` into one `console_enabled` boolean, which
both the proxy's root set and the auth middleware read — so the switch that
makes the console reachable is the same switch that guards it.

Remote mode does not reuse the data-plane token for this: that token also
lives in a same-uid-readable credential file and rides the render WebSocket
as a query param — fine for viewing pixels, wrong for running code — so
local-mode-only is the boundary instead, matching the actual story: the
human is at the machine with the napari window. A remote console would need
a distinct credential.

`_check_origin`'s content-type exemption ("no JSON body") doesn't hold for
the console's POST, so it goes through `_json_route`
(`_route` plus `TransportSecurityMiddleware._validate_content_type`)
instead: `Content-Type: application/json` isn't CORS-simple, so a
cross-site form POST can't forge it — a real CSRF defense.

**Known limitation:** the gate reads the listener's *bind*, so a loopback
control published by a reverse proxy for untrusted networks reads as local
and gets the console; that operator already owns the token in front of the
data plane and what their proxy exposes.

`observe.console_enabled` (default `true`) drops the route entirely when
off, rather than serving a refusing one, and can only narrow — never
override the public-bind refusal. Availability is the conjunction of both
gates: the control's `/health` reports `console_enabled` and the child's
`/api/status` reports its own knob; `ObservePage` renders the editor only
when both are true.

## Serialization: reject, never queue, never preempt

User code goes through `_jobs.submit()` — one-job-at-a-time exclusion,
output capture, observe rendering and notebook export all come for free,
and write ordering to the namespace stays single. `submit()` returns
`{"error": "busy", "running_job_id": ...}` and nothing about `_jobs` locking
changes for it: queueing the user's cell behind the agent's would create a
hidden second writer with invisible ordering, and preemption would let two
live jobs cross-cancel each other's dask futures (`_cancel()` cancels every
future the client tracks). The human's escape from a busy kernel is the
interrupt they already have.

The UI renders busy as **state**, not an error after the click — Run
disabled, labelled `kernel busy · job-7 (agent)`, Interrupt beside it. The
route's `409` (carrying `running_job_id` + `running_job_origin`) is the race
backstop for a collision the disabled button didn't prevent, not the
primary signal.

Two "busy"es on this route are not the same thing: a **409** is the job
runner's one-at-a-time rule (something is running — wait or interrupt),
while a **503** is the kernel lock held momentarily by another quick
snippet (nothing is running — retry works).

## Attribution: `origin` on the job

`_Job` carries `origin` (`"mcp"` | `"user"` | `"chat"`, see
[chat-engines.md](chat-engines.md)), set at `submit()` and carried through
`snapshot()`, `jobs_summary()` and `export()`. `requester` shares the same
vocabulary, so "may this writer stop this job?" is one comparison.

The writer count is two, by construction: the first non-user submitter
claims the kernel and a second agent's `execute_code` is refused (keyed on
the client id) — serializing two agents would order their writes without
either seeing the other's model of the namespace. The human is never
gated, since the console carries no client identity to gate on, and the
observe page's restart is likewise never gated, so a stale claim is cleared
by the person at the machine rather than seized by a second agent.

Every tool that changes kernel state (`execute_code`, `interrupt_kernel`,
`restart_kernel`) is gated the same way, so a client that doesn't hold the
kernel keeps the read-only tools and nothing else. `seen_by_agent` stays a
single flag because there is provably one agent reader, and the observe
page's "you" means `"user"` alone.

**The busy message branches by owner.** Agent-owned: *"A job (job-7) is
already running… stop it with interrupt_kernel / restart_kernel."*
User-owned: *"the user is running job-7; poll and wait — do not interrupt
it."* **`interrupt_kernel` refuses a user-origin job outright** — *"job-7
was started by the user; it is not yours to stop"* — rather than silently
killing a human's cell; `restart_kernel` stays permitted as the documented
guaranteed-stop escape hatch.

## Informing the agent: pull, not push

The agent's world model goes stale the moment a human writes to the
namespace, and it cannot be pushed a notice mid-turn — MCP server→client
notifications aren't reliably surfaced, and an idle agent has no turn to
interrupt. So every agent-facing round trip (`execute_code`, `poll_job`,
`server_status`) appends a note at return time, the same seam
`_teardown_reason`/`_WINDOW_CLOSED_NOTE` already use:

```
[2 cells were run by the user since your last call: job-7 (ok), job-8 (error).
Read them with poll_job('job-7'). Variables and layers may have changed.]
```

Each job carries a `seen_by_agent` flag (not a global watermark), read by
`foreign_digest()` and retired only by a later `ack_foreign_digest(ids)`
call the server makes after it has rendered the note — reading never
consumes, since a probe that times out can still be queued at the kernel
and run later. Only ids reported **terminal** are acked, without
re-reading status: a cell that finished between read and ack was reported
`running` and must stay pending, or it would be retired unheard; while it's
still running, the repeated note is what explains the busy kernel, not new
activity.

The agent is told only that something changed and where to look, never
what changed. `_MAX_RETAINED_JOBS` (32) never evicts an unseen user job, so
the cap can be exceeded — bounded by how many cells a human types between
two agent calls — and prunes normally once reported. The `kernel` doc tells
the agent it is not the namespace's only writer, and to re-verify with
`dir()` / `viewer.layers` on seeing the note rather than trust cached
state.

## Gotchas

- **The child's `_check_origin` is not the boundary for proxied requests**
  — the hop strips the headers it validates. Hardening the child does not
  gate the console.
- **A local process can already reach the child's `/mcp`.** The console
  changes the *browser*-reachable surface, not the same-uid one.
- **`origin` is set at `submit()`, not inferred from the calling route** —
  a job outlives the request that started it, and `poll`/`export` read it
  long after.
- **Busy is not an error.** A rejected user cell while the agent computes
  is the design working, not a failure to surface as red.
