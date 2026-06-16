# biopb crash/error reporting (draft)

> **Status: draft.** A design sketch, not yet a build plan. Captures the
> approach and the decisions it rests on so they can be reviewed before any code.

How a failure anywhere in biopb becomes a report you can act on. The guiding
insight: a single user-visible failure is rarely born where it surfaces, and the
pieces of it live on **different machines owned by different parties**. So the
system is mostly about **correlating fragments across trust boundaries**, not
about prettier per-process logs.

## 1. The problem that's specific to biopb

Because of the lazy-input framework, one logical operation fans across
processes, languages, and hosts:

```
agent execute_code  →  ops[...].run(source ref)  →  algorithm servicer (gRPC, REMOTE)
                                                       └→ pulls input from a tensor server (Flight)
                                                            └→ adapter decodes a file  ← real fault here
```

The agent sees an opaque `grpc INTERNAL`; the root cause is two hops and a
language boundary away. The highest-value capability is **causal correlation**,
not local capture — each component already logs its own errors.

### Deployment reality (what shapes everything)

- **Compute servicers are always remote**, operator-run, and **multi-tenant**:
  we host a few servicers on `biopb.org` for all users, who never set one up.
  The real diagnostic traceback is therefore born on **our** infra, not the
  user's machine.
- **The data (tensor) server is primarily local** to the user, with occasional
  exceptions (a lab box, or one of ours).
- **The client (MCP + kernel + napari)** runs on the user's machine.

Two populations with opposite roles:

| | Where it runs | Role |
|---|---|---|
| Bench scientist | MCP + kernel + napari + (usually) local data server | *source* of client-side reports; should never run infra |
| Operator (us) | the few servicers + occasional data server on `biopb.org` | *owner* of the reports that actually diagnose servicer bugs |

## 2. The core pattern: correlation ID as a claim check

A `trace_id` is minted at the entry point (the `execute_code` call) and
propagated on the wire so every fragment of one failure carries it. Crucially,
the **server-side detail is never returned to the client** — the servicer is
multi-tenant, so leaking its traceback leaks other users' paths and server
internals. The client receives only:

- the `trace_id`, and
- a generic, redacted message.

The user quotes that ID (or their MCP opt-in-pushes it); we look it up on our
side and join it to the full report. **The ID is a claim check, not the report.**
This is what makes the cross-ownership trace propagation load-bearing rather than
nice-to-have: it is the only thing linking the user's "it broke" to our server's
"here's why".

### Propagation rules

- Use the **W3C `traceparent`** format (16-byte IDs) even though we hand-roll the
  propagation. It's a trivial header to copy, and it keeps the door open to
  dropping in OpenTelemetry later (for latency/perf observability) without
  re-plumbing the wire.
- gRPC: carry `traceparent` in call **metadata**, alongside the existing auth
  interceptor.
- Flight: carry it in call **headers**, alongside the existing bearer middleware.
- **Across the hop, inside a node:** when a servicer does a lazy pull, it must
  **forward the same `traceparent`** into its outbound Flight request. Otherwise
  the data-plane fault is orphaned from the chain.
- **Join only on `trace_id`, never timestamps** — fragments live on machines with
  unsynchronized clocks.

## 3. Where reports live, and how they're collected

### Server side (compute plane, `biopb.org`) — journald *is* the backend

Server logging is already **systemd/journald**, and journald is not just a log
file — it's an **indexed structured-field store you can query**. The image
runtime already logs the traceback + an 8-hex error id at its one error
chokepoint (`BiopbServicerBase._server_context`). The change is to emit
**custom journal fields** alongside it:

```
BIOPB_TRACE_ID=<from gRPC metadata>   BIOPB_ERROR_ID=<the existing 8-hex>
BIOPB_OP=cellpose.run                 BIOPB_COMMIT=<server build>
BIOPB_SEVERITY=error
```

Then the entire triage path is one command — no database, no Sentry, no
collector:

```
journalctl BIOPB_TRACE_ID=<id> --output=json
```

This spans all servicers on the host at once. **Defer any platform** (self-hosted
Sentry, OTel collector) until there's more than one host or we want
dashboards/alerting; journald earns its keep until then.

**Docker wrinkle.** The servicers are containerized, so app-level structured
fields don't reach the host journal by default. Two options:
- *(preferred)* mount `/run/systemd/journal/socket` into the container and use
  `systemd.journal.send` → true native fields, `journalctl BIOPB_*=` works.
- *(fallback)* emit one JSON line per event on stdout + run with
  `--log-driver=journald`; query with `journalctl ... | jq` (no native field
  index).

Bump retention for these records (`SystemMaxUse` or a dedicated journal
namespace) so crash records outlive verbose request logs.

### Client side (user machine) — the only place intent lives

The servicer is **stateless** (§2.3 of the architecture overview): it sees one
RPC, runs a model, returns. So the server-side record is by nature an
**anonymous mechanical crash** — it can tell us *what* broke, never *why the user
hit it*. The client half is the only place intent exists, which makes the opt-in
client push the thing that renders the anonymous server record *interpretable*.

Lean into the statelessness — keep the client report **thin**:

```
trace_id, client_commit, the (redacted) execute_code cell, + a one-shot snapshot
(layer names / loaded source_ids at crash time)
```

A *snapshot*, not a history. Don't build a session model the architecture
doesn't produce. The single most valuable field is the **`execute_code` cell that
was running** — because the agent writes arbitrary Python, that cell *is* the
intent in concrete executable form, already captured by the kernel's job
tracking.

The client also captures the three MCP-specific crash classes already detected
but not reported today: agent-code exception (`execute_code`), bootstrap failure
(`_BOOTSTRAP_ERROR`), and watchdog-detected kernel kill (emitted by the
surviving MCP process).

A **`session_id`** (per kernel lifetime) groups a user's sitting across several
`trace_id`s. Keep it **client-only** — propagating it to the servicers would buy
server-side grouping but re-introduce user-linkable data into the multi-tenant
store, defeating the statelessness. Group client reports by `session_id`; join to
the server by `trace_id` on demand.

### Data server — reports are *pulled*, not pushed

A data server may sit on the user's machine, on a lab box "near but not on" the
MCP machine, or on `biopb.org`. Nobody runs a collector on the lab box, and it's
often unreachable from `biopb.org` (behind the lab's NAT). So don't push —
**expose**: add a **`diagnostics` `do_action`, keyed by `trace_id`**, to the
tensor server's existing Flight control verbs (`health`, `cache_stats`,
`chunk_locate`). Whoever holds a Flight handle + the `trace_id` pulls the
data-plane fragment on demand.

The **MCP holds both** (it has the connection and minted the id), so it is the
fan-out aggregator: given a `trace_id`, hit every data endpoint it has a handle
to, collect fragments, join. This subsumes local/lab/`biopb.org` data servers
under one mechanism.

### Edge case: data server near-but-not-on the MCP machine

"Near but not on" usually means on the lab intranet, **behind NAT** — reachable
from the MCP over the LAN, not from `biopb.org`. This forks the failure shape:

- **Data server routable from the servicer** (VPN/public): lazy works → a genuine
  **three-way chain** (user → servicer → data server). Requires the servicer to
  forward `traceparent` into its lazy pull (§2). The lab fragment is retrieved via
  its `diagnostics` action.
- **Data server behind NAT** (common): the servicer can't pull from it, so lazy
  **degrades to eager** (the eager/lazy duality, §2.3 of the overview). The data
  plane is then touched **only by the client**, so there is no `biopb.org`-side
  data fragment — both fragments are LAN-local and the MCP joins them directly.
  *Simpler*, not harder.

## 4. Build vs. buy (per layer)

Not one decision — the layers have different economics. **Do not adopt a
monolith** (Sentry): self-hosted is operationally huge for a small-lab tool, and
SaaS contradicts the local-first / opt-in posture.

| Layer | Decision | Why |
|---|---|---|
| Schema + chokepoint capture | **in-house** | biopb-specific vocabulary (array_id, op, kernel crash classes); tiny; fits the buf pipeline; backend-agnostic |
| Trace correlation | **hand-roll, OTel-shaped IDs** | the hard part is the propagation *format*, not the id; W3C `traceparent` keeps OTel a drop-in later |
| Server sink | **journald structured fields** | already running; queryable; zero new infra |
| Client sink | **stdlib logging + existing redactor** | already present; no new dep |
| Data-server fragment retrieval | **`diagnostics` `do_action`** | uniform across locations; no collector on third-party boxes |
| Submit/egress | **opt-in push to operator endpoint** + `gh` issue option | best UX for non-technical users; no server to run |
| Whole platform | **no (defer)** | only justified at multi-host scale we don't have |

Any external dependency stays an **optional extra** — the core path must work with
stdlib + `systemd` bindings alone, to respect the no-AVX eager-only build and the
"services stay simple" stance.

## 5. Redaction & privacy

- **Server side:** redact **before** `journal.send` — journald persists it, and
  it's *user* data crossing into our host (paths, array_ids, microscopy
  metadata). Reuse/centralize the existing path+token redactor
  (`http_server.py`).
- **Egress is the redaction boundary, not capture.** A lab data server's reports
  hold the *lab's own* data, pulled by the *lab's own* user — capture can be
  fuller there. But the moment the MCP folds a fragment into a bundle headed to
  us, it must be redacted.
- **Consent line:** using the hosted servicers means failure context may be
  logged operator-side; state it.

## 6. Shared vocabulary

One source of truth for field names — a `proto/biopb/diagnostics/report.proto`
(generated for Py/Java/TS via buf, like everything else). It defines the report
message **and** the journal field-name registry, so the client report fields and
the server journal fields use **identical names** and join trivially. Stamp
**both** `client_commit` and `server_commit` — version skew between a user's
client and our deployed servicer is expected to be a large share of "bugs", and
showing the two side by side will explain many reports without reading a
traceback. (Add `BIOPB_BUILD_COMMIT` at container build; expose via the existing
`health` actions.)

## 7. Suggested sequencing

1. **Schema + commit stamping + capture at the existing chokepoints** — no wire
   changes. Server: journal fields at `_server_context`. Client: thin report +
   the three kernel crash classes. Immediate value: structured, redacted,
   queryable records per component.
2. **trace_id propagation** (`traceparent` over gRPC metadata + Flight headers +
   the servicer's lazy-pull forwarding) → cross-plane / cross-host correlation.
3. **`diagnostics` `do_action`** + the MCP fan-out aggregator + opt-in client
   push / `gh` bundle.

Step 1 is useful on its own; the wire changes are deferred until local capture
has proven its shape.

## Open items / decisions to confirm

- Docker journal: mount the journal socket (native fields) vs. JSON-on-stdout
  (fallback)? Default to the socket for servicers we control.
- Where does the opt-in client push land — a small endpoint on `biopb.org`, or
  GitHub issues only? (Endpoint is far better UX for bench scientists.)
- Retention policy for crash records vs. verbose request logs (dedicated journal
  namespace?).
- Server-side volume controls for a crashing multi-tenant servicer: carry over
  the existing ring-buffer + per-session rate-limit patterns; add
  dedup-by-fingerprint + sampling.
- Field registry: confirm `report.proto` field names match what `journalctl`
  allows (uppercase, alnum + underscore) so the proto and journal stay 1:1.
