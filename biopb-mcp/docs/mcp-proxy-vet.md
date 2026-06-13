# Vetting report: mcp-proxy as the stdio→http bridge

**Context:** docs/daemon-migration.md (Direction 1) plans to recommend
[sparfenyuk/mcp-proxy](https://github.com/sparfenyuk/mcp-proxy) as the generic
streamable-http↔stdio bridge for stdio-only clients once biopb-mcp goes
http-only, likely wrapped in a connect-or-spawn shim. This is the pre-adoption
vet (June 2026).

**Method:** source review of v0.12.0 (`153a96a`, the released tip), project
health via the GitHub API, and a live end-to-end test: biopb-mcp http daemon on
a scratch port, `uvx mcp-proxy --transport streamablehttp <url>` in front,
raw JSON-RPC over the proxy's stdio (the exact shape an MCP client config
produces), diffed against a direct streamable-http connection.

**Verdict: viable, with three conditions.** It is small (~900 lines of source,
MIT, typed, tested, active), stdout-hygienic, and the happy path works against
our server. But it cannot be adopted as the README's bare `uvx mcp-proxy`
one-liner: deps must be pinned, the `instructions` drop must be fixed upstream
(or the bridge vendored), and the shim must own daemon-death cleanup.

---

## Findings

### 1. The direction we need works (verified live)

Client mode `--transport streamablehttp <url>` bridges a stdio client to a
streamable-http backend. Against the biopb-mcp daemon: `initialize` →
`tools/list` (all 9 tools) → `tools/call server_status` all round-tripped
correctly, and the proxy exits 0 when the client closes stdin (daemon keeps
running) — the right lifecycle for a client-spawned bridge.

### 2. BLOCKER: the initialize `instructions` field is dropped

`proxy_server.py` rebuilds the server as
`server.Server(name=response.serverInfo.name)` — it never passes
`response.instructions` through (the SDK's `Server(...)` accepts an
`instructions=` kwarg, so this is a one-line upstream fix). Verified live:
direct connection returns our full `_BASE_INSTRUCTIONS`; through the proxy,
`instructions: None`.

biopb-mcp leans on `instructions` as the handshake-time carrier for the
operation guardrails and the headless notice — losing it silently degrades
every bridged agent session. **This must be fixed upstream (PR is trivial) or
the bridge vendored before the stdio cutover.** Related symptom, same root
cause (the initialize result is reconstructed, not forwarded): proxied
`serverInfo.version` reports the proxy's SDK version, not ours (upstream
issue #214).

### 3. It is a capability-gated re-implementation, not a transparent pump

The proxy registers explicit per-request handlers (tools, prompts, resources,
logging, completion) and forwards progress notifications. **Not** forwarded:
server→client requests (sampling/`createMessage`, elicitation) and
server→client notifications other than progress (`tools/list_changed`,
`resources/list_changed`, logging messages). biopb-mcp uses none of these
today — the on-demand-kernel design deliberately avoided `list_changed`
(PR #41) — but any future feature relying on them silently won't cross the
bridge. Tool results pass through as SDK objects, so `ImageContent`
(screenshots) is fine structurally (not exercised live; needs a kernel).

### 4. No reconnect; daemon death can strand a hung proxy

There is no retry/reconnect logic anywhere in the source. Daemon death
mid-session kills the bridged session — acceptable, the client re-spawns the
shim. Worse, observed live: after the daemon died, the proxy process did
**not** exit even once its client was gone (stdin EOF); it lingered until
killed. A connect-or-spawn shim should therefore wrap the proxy with its own
lifetime guard (e.g. die-with-child / timeout), or this needs an upstream fix,
or vendoring.

### 5. Supply chain: floating deps are broken *today*

`uvx mcp-proxy` resolves dependencies at run time with no lockfile. As of this
vet that recipe is **unusable**: anyio 4.13.0 (current latest, pulled in
transitively via the `mcp` SDK) ships a literal typo in `fail_after`
(`as cancel_Scope:` / `yield cancel_scope` → `NameError`), which the SDK hits
on the first `initialize`. Pinning `uvx --with 'anyio<4.13'
mcp-proxy==0.12.0` fixes it. Two consequences:

- Any documented recipe must pin the proxy version **and** be resilient to
  transitive breakage — i.e. prefer a shim that controls its own resolved,
  locked environment over a bare `uvx` line in user-facing docs.
- **biopb-mcp's own `uv.lock` also pins anyio 4.13.0.** Our tested paths don't
  hit `fail_after`, but anything that does will crash. Add an
  `anyio!=4.13.0` constraint and re-lock (deferred during this vet — the dev
  `.venv` was serving a live session).

### 6. Project health

2.6k stars / 242 forks, MIT, active (pushed within days of this vet),
releases roughly quarterly (v0.12.0 May 2026), CI + tests + mypy, small dep
surface (`mcp`, `uvicorn`, `httpx-auth`). Logging goes through
`logging.basicConfig` (stderr by default) — no stdout pollution, so fd-1
hygiene holds. Main risk is **bus factor 1**: effectively a single maintainer
(next human contributor: 7 commits). Open issues are mostly dependabot noise;
real ones (e.g. #214, #220 — unknown server notifications crash the TaskGroup
in the reverse direction) sit unfixed for weeks-to-months.

---

## Recommendation

Either path is acceptable; both require the same shim wrapper anyway:

1. **Adopt mcp-proxy, pinned** — submit the one-line `instructions` PR
   upstream, pin `mcp-proxy==<vetted>` (+ transitive guards) in the shim's
   environment rather than documenting bare `uvx mcp-proxy`, and give the shim
   a daemon-death lifetime guard (finding 4).
2. **Vendor the bridge** (stronger, recommended): findings 2–5 are all
   consequences of not controlling the bridge. The `mcp` SDK we already depend
   on contains both halves (`mcp.server.stdio.stdio_server` +
   `mcp.client.streamable_http.streamablehttp_client`); mcp-proxy's own core
   is ~45 + 160 lines on top of them. A vendored bridge lives in our lockfile,
   forwards `instructions` correctly, and dies when we say so — at the cost of
   maintaining ~150 lines and tracking SDK changes ourselves.

Re-vet before the cutover if adoption (path 1) is chosen and months have
passed: single-maintainer projects drift.
