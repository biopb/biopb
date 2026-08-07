# The user console — a second writer in the kernel

Today only the agent runs code in the session kernel (`execute_code` → `_jobs.submit`).
This adds a **human** writer: a code cell on the observe page (`/session/<id>/observe`),
running in the *same* kernel, the *same* namespace, through the *same* job runner.

Guarantee: the human and the agent are **serialized against each other** (one job at a
time, no preemption, no queue), and neither can act on the other's running job without
the other finding out.

## Why

The observe page already shows the agent's work — job history, output, interrupt,
restart, notebook export. It is the natural place for the human to *join* the work
rather than only watch it: fix a threshold the agent guessed wrong, inspect a variable
the agent didn't print, try one line before asking for twenty.

The two hard parts are not the textarea. They are: (1) an execute endpoint on a web
origin that may be public, and (2) an agent whose world model silently goes stale the
moment someone else writes to its namespace.

## Security — the gate belongs to the control, not the child

**At the session child, an execute route adds no authority.** The child already serves
`/mcp` on the same loopback port behind the same Host/Origin guard, and `/mcp` *is*
`execute_code`. Anything that can reach the child's port already has RCE; a second door
into an open room changes nothing.

**At the control front it changes exactly one thing, and it is the thing the code was
built to prevent.** `_SESSION_ALLOWED_ROOTS = {"api"}` (`_control.py:123`) is an
allowlist whose whole purpose is to keep `/mcp` — "an RCE hole on the public origin",
per `session_proxy`'s own comment (`_control.py:817-823`) — off the proxied origin.
Folding an execute route into `/session/<id>/api/*` puts arbitrary code back through
that hole. The invariant would not break loudly; it would just stop being true.

So the console gets its **own path root**, and its own gate:

| Root | Proxied | Gate |
|---|---|---|
| `/session/<id>/api/*` | always | `_ControlAuthMiddleware` (token or loopback Host; CSRF on unsafe verbs) |
| `/session/<id>/console/*` | **local mode only** | the above, **plus** a public-bind refusal and a JSON content-type requirement |
| `/session/<id>/mcp` | never | — (agents reach the child directly) |

The amended allowlist statement, which stays true: *`api` always; `console` only when the
control is loopback-bound; `/mcp` never.*

### Why the child cannot decide this

The proxy hop deliberately strips Host and Origin toward the child (`_control.py:836-842`)
so the child cannot judge the browser's origin — which also means the child's own
`_check_origin` (`_observe.py:116`) passes trivially for *every* proxied request. The
control-side check is already the child's only auth (CLAUDE.md, Security model). Only the
control knows whether it is loopback-bound or `--remote`. Therefore the local-mode gate is
control-side, full stop; the child's route may exist unconditionally.

`serve_control_api` already receives the control's own bind `host`, so the gate is
`_web_auth.host_is_public_bind(host)` computed there and passed into `build_app` as one
boolean (`console_enabled`). No new plumbing from `__main__`. `_session_proxy_roots()`
turns that boolean into the root set, and **both** the proxy's own gate and the auth
middleware read that one set — so the switch that makes the console reachable is the same
switch that makes it guarded; an unauthenticated execute path is unrepresentable.

**Known limitation, deliberately accepted.** The gate reads this listener's *bind*, so a
loopback control deliberately published by a reverse proxy — the topology biopb-mcp's
CLAUDE.md points at for untrusted networks — reads as local and gets the console. That
operator is already responsible for the token in front of the data plane and for what
their proxy exposes. A control-side `--no-session-console` flag is the follow-up if that
topology stops being the exception; `console_enabled` is already the parameter it would
set.

### Why not "just gate it with the token"

In remote mode `_ControlAuthMiddleware` gates `/session/<id>/api/*` with the **data-plane
token**. That token today authorizes reading pixels and restarting the plane. It also:

- lives in the local credential file (`biopb._credentials`), readable by any same-uid
  process — deliberately, that is the #470 handoff;
- rides the render WebSocket as `?token=` (`token_valid_with_query`), i.e. into logs.

Those properties are fine for "view my data" and wrong for "run code as me". Reusing it
for the console silently upgrades every existing holder of a viewing credential to a
shell. Local-mode-only sidesteps this entirely and matches the actual collaboration
story: the human is at the machine where the napari window is. If remote console is ever
wanted, it must be a **distinct** credential — not this one.

### The content-type detail

`_observe.py:119-121` skips the SDK's content-type validation because "our control POSTs
carry no JSON body". The console POST does carry one, and `Content-Type: application/json`
is **unforgeable by a cross-site form POST** — it is a live CSRF defense, not ceremony.
The console route must therefore *restore* the check the sibling routes skip, rather than
inherit `_route`'s guard unchanged. This is easy to miss precisely because the guard is
shared.

### Config

`observe.console_enabled` (default `true`) is the opt-out for a site that wants observe
read-only. It cannot opt *in* past the public-bind refusal — a knob may narrow the
surface, never widen it.

## Serialization — reject, never queue, never preempt

User code goes through **`_jobs.submit()`**, not a side channel. That inherits
one-job-at-a-time exclusion, output capture, the observe rendering, and notebook export
for free, and it keeps a single ordering of writes to the namespace.

`submit()` already returns `{"error": "busy", "running_job_id": ...}`. That is the whole
concurrency story — **no changes to `_jobs` locking**. Two readings were rejected:

- **Queueing** the user's cell behind the agent's would create a hidden second writer
  whose ordering nobody can see. Rejecting keeps every ordering decision explicit and
  human-made.
- **Preemption** is not a small change: `_running_job()` returns "the single running job"
  and `_cancel()` cancels *all* futures tracked on the dask client (`_jobs.py:327-336`),
  so two live jobs would cross-cancel each other's compute.

The human's escape from a busy kernel is the interrupt they already have. The UI must
therefore render busy as **state, not as an error after the click**: the Run button
disabled, labelled `kernel busy · job-7 (agent) running`, with the existing Interrupt
button beside it.

## Attribution — `origin` on the job

`_Job` gains `origin` (`"agent"` | `"user"`), set at `submit()`, carried through
`__slots__`, `snapshot()`, `jobs_summary()` and `export()`. Everything below is a
consequence of that one field.

**The notebook export becomes a real audit.** Cell provenance is recorded rather than
implied, and the interleaving is already correct — `export()` is job-ordered.

**The busy message must branch.** `execute_code` today tells the agent: *"A job (job-7)
is already running… stop it with interrupt_kernel / restart_kernel"* (`_server.py:647`).
An agent reading that while a **human's** cell is running will cheerfully kill it. Split
it: agent-owned → current wording; user-owned → *"the user is running job-7; poll and
wait — do not interrupt it."* This is the one place the no-preemption rule has to be
spelled out agent-facing, and it is easy to miss because the code path is shared.

**`interrupt_kernel` must refuse a user-origin job.** Attribution is one-way today:
`_USER_INTERRUPT_MSG` (`_observe.py:62`) explains a user→agent stop, but the agent has no
equivalent — `interrupt_kernel` hits whatever is running, so an agent interrupting during
a human's cell kills it silently, leaving the human an unexplained `interrupted` badge.
Rather than mint a reason string for that, refuse it: *"job-7 was started by the user; it
is not yours to stop."* The human has the observe UI and can stop their own work; the
agent has no consent to. `restart_kernel` stays permitted — it is the documented
guaranteed-stop escape hatch and is destructive by advertisement.

## Informing the agent — pull, not push

The agent's world model goes stale the moment the human writes to the namespace: a
redefined variable, a deleted layer, a fresh import. The agent cannot be *told* at the
moment it happens — MCP server→client notifications are not reliably surfaced mid-turn,
and when the agent is idle there is no turn to interrupt.

So annotate the agent's own return path, which is exactly how every other user-attributed
fact already reaches it (`cancel_reason` on a job, `_teardown_reason` on a not-ready
result, `_WINDOW_CLOSED_NOTE` on a result with no viewer).

Each job carries a `seen_by_agent` flag rather than a global seq watermark, so the rule
can be stated per job: `user_digest(ack=True)` reports every unseen user job and marks
the **terminal** ones seen. A still-running user cell therefore stays in the digest — the
agent hears its final status exactly once instead of only ever hearing that it started,
and while it runs the repeat is what explains a busy kernel.

Every agent-facing round trip (`execute_code`, `poll_job`, `server_status`) then appends,
at the same seam `_window_note` uses in `_server.py`:

```
[2 cells were run by the user since your last call: job-7 (ok), job-8 (error).
Read them with poll_job('job-7'). Variables and layers may have changed.]
```

The agent is not told *what* changed — only that something did, and where to look. That
is cheap, honest, and enough.

Two supporting pieces:

- **`guide://kernel` gains a paragraph**: you are not the only writer of this namespace; a
  human may run cells from the observe page; on seeing that note, re-verify with `dir()` /
  `viewer.layers` rather than trusting cached state.
- **Retention**: `_MAX_RETAINED_JOBS = 32` (`_jobs.py:51`) could evict a user job before
  the agent ever read it, silently dropping the only notice it gets. `_prune()` therefore
  never evicts an unseen user job; the cap can be exceeded, bounded by how many cells a
  human types between two agent calls. Once reported, the record prunes normally.

## Gotchas

- **The child's `_check_origin` is not the boundary for proxied requests** and must not be
  mistaken for one — the hop strips the headers it validates. Do not "harden" the child
  and consider the console gated.
- **A local process can already reach the child's `/mcp`.** The console changes the
  *browser*-reachable surface, not the same-uid one. Nothing here defends against a
  same-uid attacker, and nothing here needs to — that boundary was already gone.
- **`origin` must be set at `submit()`, not inferred from the calling route.** A job
  outlives the request that started it, and `poll`/`export` read it long after.
- **Busy is not an error.** A user cell rejected while the agent computes is the design
  working. Surfacing it as a red failure would train users to interrupt the agent
  reflexively.

## Shape of the work

Naturally three stacked PRs against `dev`, the control gate reviewable on its own:

1. **`_jobs.py` + `_server.py`** — `origin`, `user_digest`, the busy-message branch, the
   `interrupt_kernel` refusal, the `guide://kernel` paragraph. **Done.**
2. **`_control.py`** — the `console` root, `host_is_public_bind` gate via
   `serve_control_api`, proxy tests for the public-bind refusal and the traversal cases
   already covered for `api`. **Done.**
3. **`_observe.py` + `ObservePage.tsx`** — the child route (with the restored content-type
   check), `observe.console_enabled`, the editor and its busy state.
