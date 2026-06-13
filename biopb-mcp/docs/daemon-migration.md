# Design note: http-only transport + on-demand kernel (daemon model)

**Status:** Direction 2 (on-demand kernel) is **implemented** — though as
*explicit* start (a `start_kernel` tool the agent calls), **not** the implicit
lazy-on-first-`execute()` trigger this note sketches; see the "Direction 2"
section below for the as-built notes. Direction 1 (http-only) is
**implemented** (June 2026), with one deliberate departure from the sketch
below: instead of documenting an external `mcp-proxy` recipe for stdio-only
clients, the bridge is **vendored as a shim** (`mcp/_shim.py`) and
`--transport stdio` *is* the shim — it ensures the http daemon is running on
the configured port (spawning it detached on first use) and pumps stdio
JSON-RPC to `/mcp`. So `_server.run_stdio()` and the launcher's stdio serving
branch are gone, the kernel-log fd-redirection machinery is retired (the
`kernel_log` key now names the detached daemon's output file), and
installer-seeded `biopb-mcp --transport stdio` client configs migrate with
zero config change. mcp-proxy was vetted and rejected for this role — it
drops the initialize `instructions` field, hangs on daemon death, and floats
broken deps; see docs/mcp-proxy-vet.md. Still open: idle teardown / the
daemon stop story, and flipping the default `transport.kind` to http once
clients are predominantly native-http.

**Why this exists.** Adding the web "observe" UI (`mcp/_observe.py`)
exposed a structural weakness: a feature that worked under the **http** transport
silently broke under **stdio**, because the two transports are separate serving
paths. The stdio breakage was real and twofold — see *Background* — and it points
at two architectural moves worth making deliberately rather than patching around.

---

## Background: what went wrong with observe under stdio

The observe UI needs an HTTP surface. Under http it mounts cleanly on the existing
FastMCP app (`_observe.register_http_routes` via `mcp.custom_route`). Under stdio
there is no HTTP server, so the first cut stood up a **separate uvicorn server in
the launcher process**. That had two defects, both specific to stdio:

1. **fd-1 corruption.** In stdio mode fd 1 *is* the JSON-RPC channel. uvicorn
   attaches a `sys.stdout` log handler; any WARNING+ record / stray write on that
   thread corrupts the protocol stream and crashes the client.
2. **Kernel wedge.** The standalone server drove the one `KernelHost` from its own
   event-loop thread, concurrently with the MCP thread. A poll's `execute()` in
   flight during a kernel restart got stuck holding the `RLock` → the host wedged
   (`alive: False / busy: True`).

**Current resolution:** observe is **http-only** (`__main__._setup_observe` skips
it under stdio with a logged hint). That is safe and shipped, and it is also a
*preview* of the direction below — http is the canonical surface.

Key fact that makes the rest cheap: **both transports already share the MCP core.**
`FastMCP.run_stdio_async` and `run_streamable_http_async` both wrap the same
low-level `self._mcp_server`. The "two transports" are thin I/O adapters over one
message handler; the only genuinely http-only thing is *auxiliary HTTP routes*.

---

## Direction 1 — make biopb-mcp http-only; delete the stdio code

Don't maintain a stdio↔http bridge — http-over-stdio is a common pattern with
maintained, generic tooling. So:

- **biopb-mcp serves http only.** Drop `_server.run_stdio()` and the stdio branch
  in `__main__.main()`.
- **Clients that speak http connect by URL.** Claude Code does natively:
  `claude mcp add --transport http biopb http://127.0.0.1:8765/mcp`.
- **stdio-only clients front it with a generic bridge** —
  [`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy) (PyPI; `uvx mcp-proxy`)
  bridges streamable-http→stdio in exactly this direction. Not vendored; just
  documented as the recipe.

**Why this is strictly better, not just tidier:** in http-only, biopb-mcp **never
owns stdin/stdout as a protocol channel.** The fd-1 corruption class becomes
*structurally impossible* — the heavy process (uvicorn + Qt + dask + kernel) and
the protocol I/O (the bridge, in a separate process) are no longer the same
process. Observe and every future HTTP surface are always available. One transport
to maintain.

**What can be retired:** the `kernel_log` redirection and the "keep launcher stdout
pristine" rules (`mcp.transport.kernel_log`, the `_open_kernel_log` machinery,
`KernelHost._kernel_stdout/_stderr` redirection) exist *only* because stdio runs
the kernel in the protocol process. Under http, fd 1 is not the protocol channel,
so native kernel output to the launcher's fds is harmless — that whole class of
care largely goes away.

**Honest costs:**

- **Lifecycle/onboarding change.** Today a client spawns `biopb-mcp --transport
  stdio` and the subprocess dies with it (parent-death-pipe reaping, issue #13).
  In http-only the server is a **separately-managed daemon**; `mcp-proxy` connects
  to it but does not own its lifetime. Needs a start/stop story, and the biopb
  installer's seeded `transport: stdio` config must migrate.
- **A loopback port is always bound** — minor; the Host/Origin allowlist +
  loopback bind already cover the browser-attacker threat.

---

## Direction 2 — decouple kernel lifecycle from server lifecycle (load-bearing)

This is the change that *makes the daemon model viable*, and it is independent of
Direction 1 (could be done today) — but it is essential once the server is a
long-running daemon.

> **As built (PR #41).** Shipped, but with three deliberate departures from the
> sketch below, decided in review:
> 1. **Explicit, not lazy.** Start is driven by a `start_kernel` tool the agent
>    calls — not an implicit trigger inside `execute()`. Kernel-dependent tools
>    instead *guard* (return "call `start_kernel`") when the kernel is down, which
>    is correct on any client and needs no dynamic tool list / `list_changed`.
> 2. **Synchronous.** `ensure_started()` blocks until ready (like `restart()`),
>    rather than returning "starting" for the agent to poll — FastMCP runs sync
>    tool handlers on the event loop, so `restart_kernel` already blocks and the
>    two now match. The transient "starting" status survives only for a watchdog
>    respawn, derived from `is_alive() && !ready` (no `_starting` flag).
> 3. **Display resolved at boot, not spawn.** Re-reading `$DISPLAY` at spawn is a
>    no-op (a process's env is frozen at exec; `_has_display()` checks presence,
>    not liveness), so the "resolve at spawn time" refactor below was skipped —
>    `set_headless` / the `visible` fail-fast stay at boot. Revisit with Direction
>    1 if an env-refresh mechanism appears.
>
> Also added, beyond the sketch: **closing the napari window tears the kernel
> down to idle** (a reverse window-close pipe), with the reason attributed to the
> agent. Idle teardown and the multi-client policy remain deferred, as below.

**Principle:** server up = cheap and always; **kernel up = lazy**, because the
kernel is the heavy/GUI part (child Jupyter kernel → napari viewer → dask cluster).
A daemon may start at login/boot, before a display exists, before any client
connects. Eagerly spawning the kernel then pops a viewer window (or Qt-aborts)
with nobody there.

### Current state (fused)

`__main__.main()` eagerly calls `host.start()` on a background thread at boot,
which runs `KernelHost._launch()` + the health probe → viewer + dask come up
immediately. The display decision (`_has_display()` / `_resolve_headless()`) is
made at launcher boot and baked into how `KernelHost` is constructed (env vars,
the bootstrap exec-line).

### What's nearly free — the async readiness machinery already exists

`KernelHost.execute()` deliberately does **not** block on bring-up: when `_ready`
is unset it returns `status: "starting"` ("retry in a few seconds"), or
`status: "error"` when `_start_error` is set. `server_status` is a cheap,
non-blocking probe via `host.health()`. This was built so the kernel could boot
off-thread while the handshake is served — and it is exactly the shape on-demand
needs.

### Target design

- **Remove the eager `host.start()`** in `main()`. The `KernelHost` is constructed
  idle.
- **Add `KernelHost.ensure_started()`** — idempotent, lock-guarded. First caller
  kicks off `_launch()` + health probe on a background thread; concurrent callers
  no-op (the `RLock` + `_ready` + a new `_starting` flag handle the multi-client
  race). Conceptually this is what the launcher's background-start thread does
  today, but triggered on first demand instead of at boot.
- **Trigger from the single chokepoint, `execute()`** (when not ready, not
  errored, not dead). Every kernel-dependent tool funnels through `execute()`, so
  the first `execute_code` / `take_screenshot` / `inspect_object` spawns the kernel
  and returns the existing "starting, retry"; the next call lands ready.
  `server_status` keeps using `health()`, so it stays **passive** — report
  `kernel: not started (spawns on first use)` rather than forcing a boot.

### The actual refactor — resolve the display at spawn time

The non-trivial part. Today headless-vs-visible is resolved at launcher boot and
baked into the `KernelHost` construction (env + bootstrap exec-line). For a daemon
that predates the display, a boot-time decision is meaningless. So the env /
bootstrap spec must be built **lazily inside `ensure_started()` / `_launch()`** —
evaluate `_has_display()` / `_resolve_headless()` *when a client actually triggers
the kernel*, not when the daemon started. Practically: `main()` passes the config
(not a pre-resolved env) and `KernelHost` builds the launch spec at spawn time.

### Decisions to settle in the implementation plan (deferred, not now)

- **Idle teardown** — on-demand spawn's natural twin. A long-lived daemon probably
  should not hold a Qt+dask kernel forever after the last client leaves, but
  tearing down a viewer mid-session is disruptive. Policy: teardown only after all
  sessions disconnect + an idle timeout. Separable from the first step.
- **One shared viewer across concurrent clients** — the daemon can serve multiple
  sessions, but there is one kernel / one viewer (the project thesis). On-demand
  makes "first user spawns it, others attach" explicit — now literally shared
  across separate client processes, unlike today's one-kernel-per-stdio-launch.
- **Watchdog while idle** — issue-#13 respawn logic should probably not resurrect
  a kernel that died while no one is using it; respawn matters only during active
  use.

---

## Sequencing

1. **Direction 2 first (lazy kernel).** It is the load-bearing change and is
   independently valuable. Order: remove eager start → `ensure_started()` →
   `execute()` chokepoint → spawn-time display resolution. Defer idle-teardown and
   the multi-client policy.
2. **Then Direction 1 (http-only).** Drop `run_stdio()` + stdio dispatch; migrate
   the installer's `transport` default; document the `mcp-proxy` recipe for
   stdio-only clients; retire the `kernel_log` / fd-1 machinery.

Each is shippable on its own; together they yield a long-running http daemon that
binds cheaply, never risks the protocol channel, exposes HTTP features uniformly,
and only spins up the GUI kernel when a client actually uses it.

---

## Code anchors

- `mcp/__main__.py` — `main()` eager `host.start()`; `_has_display`,
  `_resolve_headless`, `_open_kernel_log`, `_setup_observe`; the transport dispatch.
- `mcp/_kernel.py` — `KernelHost.start` / `_launch` / `execute` (the "starting"
  short-circuit) / `health`; `_ready` / `_start_error` / `_dead` state.
- `mcp/_server.py` — `run` (http; the only serving path — `run_stdio` is
  gone); `build_transport_security`; `server_status`.
- `mcp/_shim.py` — the as-built Direction 1 bridge: `ensure_daemon` (probe /
  detached spawn / readiness wait), `build_proxy` + `replay_init_options`
  (vendored stdio↔http pump), `serve` (launcher entry).
- `mcp/_observe.py` — http-only mount (`register_http_routes`), the precedent.

## References

- [sparfenyuk/mcp-proxy](https://github.com/sparfenyuk/mcp-proxy) — generic
  streamable-http ↔ stdio bridge (PyPI). **Vetted June 2026 — see
  [mcp-proxy-vet.md](mcp-proxy-vet.md):** viable only with pinned deps, an
  upstream fix for the dropped initialize `instructions`, and a shim-level
  lifetime guard; vendoring the ~150-line bridge is the recommended
  alternative.
- [Claude Code MCP docs](https://code.claude.com/docs/en/mcp) — native http
  transport (`claude mcp add --transport http`).
