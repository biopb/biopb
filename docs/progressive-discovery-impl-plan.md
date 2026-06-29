# Progressive Discovery — Implementation Plan

**Status:** Implementation plan — **all phases complete** (audit, freshness fields, client UI, background scan, Option B streaming, docs)
**Component:** `biopb-tensor-server` (+ a small client-gating change in `biopb-mcp`)
**Design:** [progressive-discovery.md](progressive-discovery.md) — read first; this doc
turns that design's *Changes by file* and *Suggested sequence* into PR-sized,
independently-mergeable phases with verified code anchors, acceptance criteria,
and test mapping.

---

## Locked decisions (from the design)

These are settled in the design doc and are not re-opened here:

- **First-scan population = Option B** (stream each claim as it is discovered;
  defer *removals* to the end-of-first-scan reconcile). Option 0 (background only)
  is the intermediate baseline; **Option A** (per-root reconcile) is deferred.
- **`mark_ready()` / `SERVING` is redefined** to "up and serving the
  possibly-still-populating catalog." Freshness is carried by two new `health`
  fields, not by `SERVING`.
- **Freshness reuses `SourceManager._last_full_rescan_at`** — boot and steady
  state share one mechanism.
- **Scope = monitored directories.** Static explicit sources stay seeded
  synchronously; static directory *expansion* (`resolve_all_sources`) stays
  synchronous this pass (Follow-ups in the design).

## One safety reordering vs. the design's numbering

The design lists "audit health consumers" as step 1 and the `cli.py` reorder
(which flips `mark_ready()` early) as step 4. **The consumer gating must land
*before* the `mark_ready()`-early flip**, not merely be audited first: once
`SERVING` can be reported over a partial catalog, any consumer that reads
`SERVING` as "catalog complete" regresses. So this plan sequences the additive
wire fields (Phase 1) → consumer gating on those fields (Phase 2) → the
background flip (Phase 3). The fields are inert until Phase 3, so Phase 2 can
land safely against the still-synchronous scan (it just reads
`full_scan_in_progress=false`).

## Verified anchors (corrections to the design doc)

Confirmed against the current tree; three design-doc line cites were stale:

- `do_get` reads the snapshot under `_sources_lock` at **`server.py:1191`** (design said `:921`).
- The HTTP sidecar actually starts at **`cli.py:971`** (`run_http_server`) — design's `:904/:920` are the surrounding setup.
- Streamed/normal adds reach `register_source` via **`_register_source_claim` (~`source_manager.py:1260`)**, not a bare call at `:881-882`.

All other anchors in the design's *Changes by file* are accurate.

---

## Phase 0 — Audit health consumers ✅ (complete)

**Goal:** enumerate every reader of the `health`/`SERVING` signal and decide,
per consumer, whether it breaks (or misleads) when `SERVING` is reported over a
still-populating catalog. This sizes Phase 2.

### Field reachability (no plumbing needed)

The two new fields reach every consumer for free:
- Python SDK `health_check()` (`src/main/python/biopb/tensor/client.py:2435`)
  returns the raw `json.loads(...)` dict — new keys pass through; only a docstring
  touch-up is warranted.
- The HTTP sidecar `/readyz` already forwards the **full** `backend_health` dict
  (`http_server.py:450`), so the webapp can read `backend_health.full_scan_in_progress`
  with no new endpoint.

### Findings — every consumer self-heals; two surfaces *mislead*

| Consumer | Location | Reads | Behavior under progressive `SERVING` | Verdict |
|---|---|---|---|---|
| `connect()` health gate | `_connection.py:231,239` | `status` | SERVING comes early → connects immediately with a partial/empty catalog (intended); still waits out an *old* server's long `STARTING` and the brief new-server bind window | **Tolerant** — keep gate as-is |
| `_source_watch_loop` | `_connection.py:427,432,443` | `source_count` | re-lists whenever count changes → **this is the client-side progressive-fill mechanism**; Option B's streamed adds grow the count and trigger re-list | **Tolerant — load-bearing** |
| `connect_when_booted` / `auto_connect` | `_connection.py:494,533` | via `connect()` | return early on SERVING instead of blocking through the whole scan; watcher fills in | **Tolerant** (desirable: serve ASAP) |
| `_starting_message` | `_connection.py:70` | `source_count`,`uptime` | only shown during `STARTING` (now brief) | Tolerant (minor) |
| `server_status` tool / bootstrap status | `mcp/_server.py:239,244` | full health dict | prints the whole dict → new fields appear automatically | Tolerant |
| napari widget empty-state | `tensor_browser/_widget.py:829-830` | `sources` (post-connect) | shows **"No sources found on server"** on an early/empty catalog — but `_on_sources_changed` (`:914`) re-renders and `_clear_error()`s when the watcher re-lists | **Self-heals, but misleads transiently** ⚠ |
| sidecar `/readyz` | `http_server.py:438` | `status` + forwards `backend_health` | `ready = SERVING or connected` flips true early — correct "service is up" semantics for a readiness probe | **Tolerant** (semantics correct) |
| webapp `ClientBootstrap.waitForServer` | `web/src/ClientBootstrap.tsx:15` | `/readyz` | proceeds when `ready`, then `listSources()` | Tolerant |
| webapp `SourceTree` empty render | `web/src/components/SourceTree.tsx:427` | `filteredSources.length === 0` | shows **"No sources"** on a partial/empty catalog | **Genuine "no-data" UI** ⚠ |
| webapp `startCatalogPolling` | `web/src/store.ts:216` | `listSources()` every 60s while connected | fills catalog progressively (slow cadence) | Tolerant (slow) |
| server read handlers (`do_get`/`list_flights`/`get_flight_info`) | `server.py:1191,886,414` | catalog snapshot under `_sources_lock` | already tolerate a partial/growing catalog (design §"already progressive-safe") | Tolerant (server-side; covered by Phase 3/4 tests) |

> **gRPC compute-plane health** (`image_processing/_grpc.py:356`, `health_pb2`) is a
> *different* signal — the algorithm server's standard gRPC health — and is **out
> of scope**.

### Bottom line

**No consumer hard-breaks.** The catalog self-heals everywhere via an existing
re-list mechanism (`_source_watch_loop` in biopb-mcp; `startCatalogPolling` in the
webapp), and the only hard gate — `connect()`'s `status != "SERVING"` — is exactly
the one the design intends to relax (and it still protects against old synchronous
servers). This **shrinks Phase 2** to two empty-state branches plus optional
polish:

1. **napari widget** `_widget.py:830` — branch on `full_scan_in_progress` → show
   "Indexing… (N so far)" instead of the error.
2. **webapp** `SourceTree.tsx:427` — same branch, reading
   `backend_health.full_scan_in_progress` from `/readyz`.

Optional (low-value) polish: a "catalog still indexing" legibility hint in
`server_status`; re-list on `full_scan_in_progress` true→false in
`_source_watch_loop` (catches the issue-#44 scene-growth case and the Option-0
final batch); faster webapp polling while scanning.

---

## Phase 1 — Freshness fields on `health` ✅ (complete)

**Goal:** put `full_scan_in_progress` and `last_full_scan_finished_at` on the
wire with safe defaults, before anything changes startup ordering.

**Landed:** `_scan_status_lock` + `_full_scan_in_progress` + `_last_full_scan_at`
in `TensorFlightServer.__init__`; `set_full_scan_in_progress()` /
`set_last_full_scan()` setters; the two fields added to the `health` dict (read
under the lock). SDK `health_check()` docstring documents both (and the relaxed
`SERVING` meaning). Tests: `health_status_test.py` covers defaults
(`False`/`null`), setter round-trip, and the unchanged-shape guard. Full suite
green (server health + sidecar + mcp connection/config: 160 passed). No
startup-ordering change — fields are inert until Phase 3 wires the setters.

**Changes — `server.py`:**
- In `__init__` (next to `_activity_lock`, **`:220`**) add, guarded by a new lock:
  `self._scan_status_lock = threading.Lock()`,
  `self._full_scan_in_progress = False`,
  `self._last_full_scan_at: Optional[float] = None`.
- Add `set_full_scan_in_progress(bool)` and `set_last_full_scan(float)`, each
  taking `_scan_status_lock`. (Called from the SourceManager thread in Phase 3.)
- Extend the health dict (`do_action`, **`:574-581`**) with
  `full_scan_in_progress` and `last_full_scan_finished_at`, read under the lock.

**Tests — extend `tests/health_status_test.py`:**
- New fields present; defaults are `False` / `null` on a fresh server.
- `set_full_scan_in_progress(True)` / `set_last_full_scan(t)` are reflected in the
  next `health` payload.
- `test_health_payload_shape_unchanged` still passes (add the two keys to its set).

**Acceptance:** health payload carries both fields; no startup-ordering change;
all existing tests green. **Risk:** ~none (additive read-only fields).

---

## Phase 2 — Distinguish "indexing" from "empty" in two UIs ✅ (complete)

**Landed:**
- **biopb-mcp** — `TensorConnection` caches the last-observed health dict
  (`last_health`, populated by the connect probe *and* the source-watch poll) and
  exposes `scan_in_progress()` / `scan_source_count()` (cached, no round-trip;
  absence of the field ⇒ "not scanning", preserving old behavior). The napari
  widget's two empty-catalog sites (`_on_connect_done`, `_refresh`) now call a
  shared `_show_empty_state()` that shows a grey "Indexing… (N so far)" status
  (and keeps Refresh enabled) while a scan runs, falling back to the
  "No sources found" error otherwise.
- **webapp** — `ReadyzSnapshot` gains a typed `backend_health` (incl.
  `full_scan_in_progress`); the store carries a `scanning` flag seeded from
  `/readyz` at bootstrap and refreshed in the catalog poller; `SourceTree`
  renders "Indexing data folder…" instead of "No sources" when the catalog is
  empty *and* scanning (guarded by `sources.length === 0` so a filtered-to-empty
  search still says "No sources").
- **Tests** — connection freshness helpers + `last_health` caching (5),
  widget indexing-state render (1), `readyz()` freshness round-trip (1). Affected
  suites green (175 Python; 50 tensor-flight-client; web `tsc --noEmit` clean).

Sidecar `/readyz` needed no change — it already forwards the full
`backend_health`, and its `ready=true`-early semantics ("service is up") are
correct. Optional polish (server_status hint, watcher re-list on scan-done,
faster webapp polling while scanning) deferred.

---

### Original plan (for reference)

**Goal:** before Phase 3 makes a `SERVING`-but-still-scanning catalog real, fix
the only two surfaces the Phase 0 audit found that *mislead* in that state — both
currently say "no sources" when the right message is "still indexing." Everything
else is already tolerant (the audit confirms catalogs self-heal via the existing
re-list loops), so this phase is deliberately small.

**Change 1 — napari widget (`tensor_browser/_widget.py:829-830`):** when
`sources` is empty, read `self._conn.health()` (or the value carried on the last
health probe) and, if `full_scan_in_progress` is true, show a non-error status
like `"Indexing… ({source_count} sources so far)"` instead of
`_show_error("No sources found on server")`. The existing `_on_sources_changed`
(`:914`) already repaints + `_clear_error()`s as the watcher re-lists, so this
only needs to fix the transient message, not add a fill mechanism.

**Change 2 — webapp `SourceTree` (`web/src/components/SourceTree.tsx:427`):** when
`filteredSources.length === 0`, branch on `backend_health.full_scan_in_progress`
(already returned by `/readyz`, surfaced through the store) to render
"Indexing…" instead of "No sources." `startCatalogPolling` (`store.ts:216`)
already fills the list; optionally shorten its 60s cadence while scanning.

**Optional polish (defer unless cheap):** a "catalog still indexing" hint in the
`server_status` tool (`mcp/_server.py`); re-list on `full_scan_in_progress`
true→false in `_source_watch_loop`; SDK `health_check()` docstring mentions the
two new fields.

**Tests:**
- biopb-mcp: a unit test that the widget's empty-state decision shows
  "Indexing…" when a stub `health()` returns `full_scan_in_progress=true`, and
  the error when it is false. (No Qt needed if the branch is factored into a small
  helper on the connection/service; otherwise a thin widget test.)
- webapp: a vitest on the `SourceTree` empty render given a store with
  `scanning=true` vs `false`.

**Acceptance:** with the server still scanning synchronously (fields inert,
`full_scan_in_progress=false`), behavior is unchanged; the indexing branch is
exercised by injecting `full_scan_in_progress=true`. **Risk:** low; purely
defensive copy/branch changes, no new fill mechanism.

---

## Phase 3 — Background the scan (Option 0 baseline) ✅ (complete)

**Landed:**
- **`source_manager.py`** — replaced `_runtime_phase` with `_initial_scan_done`
  (the correct startup/runtime boundary now that the first scan runs *after*
  `start()`); the precache enqueue gate keys on it. Added `_on_initial_scan_complete`.
  `_handle_rescan` now (on a force-full pass) pushes `set_full_scan_in_progress(True)`
  on entry and `False` in a guaranteed outer `finally`, advances
  `set_last_full_scan(...)` on success, and on the *first* success flips
  `_initial_scan_done` and fires the completion callback (best-effort, via
  `_fire_initial_scan_complete`). `create_source_manager` no longer runs the
  synchronous bootstrap scan.
- **`watcher.py`** — `PeriodicRescanWatcher(initial_immediate=True)` fires the
  first rescan at once on `start()` (cadence tests opt out with `False`).
- **`cli.py`** — wires the first-scan-complete callback (seeds the precache
  backlog) and pre-sets `full_scan_in_progress=True` before `start()`; calls
  `mark_ready()` **early**; and adds a `background_scan_running` fallback: if no
  event loop will run the scan (watcher failed to start, *or* a static-only
  config), it scans synchronously (monitored) or drives the completion path
  directly (static-only) — preserving registration for watcher-less setups.
- **Tests** — watcher immediate-tick (1); precache gate retargeted to
  `_initial_scan_done` (3 updated); `_handle_rescan` freshness push + one-shot
  callback + failure-reset/retry + incremental-no-toggle (3 new); static-only
  `_setup_flight_server` integration (SERVING + freshness stamped, 1 new);
  `_FakeServer`s gained the setters. Full tensor-server suite green (793 passed,
  1 skipped).

**Note (correctness catch):** removing the synchronous scan made monitored-source
registration depend on the event loop. The `background_scan_running` fallback in
`cli.py` restores the pre-existing guarantee that a watcher-less monitored config
still registers its sources (synchronously).

Catalog still appears in one batch at end-of-walk — within-walk streaming is
Phase 4 (Option B).

---

### Original plan (for reference)

**Goal:** reach `SERVING` immediately; run the monitored bootstrap scan in the
SourceManager event loop; push freshness status. Catalog still appears in one
batch at end-of-walk (Option 0) — streaming is Phase 4. This is the phase that
flips the `mark_ready()` contract, which is why Phase 2 precedes it.

**Changes — `source_manager.py`:**
- Add `self._initial_scan_done = False` and
  `self._on_initial_scan_complete: Optional[Callable[[], None]] = None`
  in `__init__` (near **`:105`**).
- **Swap the precache enqueue gate** at **`:1015`** from `self._runtime_phase` to
  `not self._initial_scan_done` (keep the `_on_source_committed is not None`
  half). *Why:* today the startup scan runs *before* `start()` sets
  `_runtime_phase=True` (**`:148`**), so startup sources skip the prompt enqueue
  and are seeded into the backlog instead. Backgrounding runs the scan *after*
  `start()`, so `_runtime_phase` would be `True` and every startup source would
  wrongly prompt-enqueue. `_initial_scan_done` restores the intent: startup set →
  backlog, live additions → prompt enqueue.
- In `_handle_rescan` (**`:227-289`**), inside the `if force_full_rescan` paths:
  push `self._server.set_full_scan_in_progress(True)` at entry and reset to
  `False` in a `finally`; on success (**`:288-289`**, alongside
  `_last_full_rescan_at = time.time()`) push `set_last_full_scan(time.time())`;
  on the **first** successful full rescan set `_initial_scan_done = True` and fire
  `_on_initial_scan_complete` (best-effort try/except, like the existing hooks).
- **Remove** the synchronous `manager._handle_rescan()` from
  `create_source_manager` (**`:1455`**); the first scan now runs in the event loop
  after `start()`.

**Changes — `watcher.py`:**
- `PeriodicRescanWatcher.start()` (**`:80-88`**): make the first tick immediate —
  set `self._next_rescan_at = time.monotonic()` instead of `+ self._rescan_interval`
  (**`:83`**), behind a default-`True` `initial_immediate` param so existing
  timing tests can opt out.

**Changes — `cli.py` (`_setup_flight_server`):**
- Keep static seeding synchronous inside `create_source_manager` (**`:1439-1451`**).
- Call `server.mark_ready()` **early** — right after the watcher/source_manager
  are wired and `start()`ed — instead of at **`:463`** (after the full scan).
- Move `precache_worker.seed_backlog(source_manager.iter_local_source_mtimes())`
  (**`:470`**) into the `_on_initial_scan_complete` callback so the startup set is
  seeded when the first scan finishes. Wire the callback before `start()`.
- **Static-only / no-watcher path:** `SourceManager.start()` returns early with no
  watcher (**`:138`**), so the event loop and the callback never run. After
  seeding, if there are no monitored dirs, drive the completion path directly:
  `server.set_last_full_scan(time.time())` and (if precache) `seed_backlog(...)`,
  so a purely static config still reports a timestamp and warms its backlog.
- `metadata_db.initial_sync(...)` (**`:430`**) is now redundant under per-source
  `sync_source_added` during the scan; keep it only for the synchronously-seeded
  static set (idempotent upsert either way).

**Tests — `tests/` (extend `source_manager_test.py` fixtures; `_FakeServer`
needs `set_full_scan_in_progress`/`set_last_full_scan` stubs):**
- `health` reports `SERVING` immediately while `full_scan_in_progress=true`, then
  `false` with a non-null `last_full_scan_finished_at` after the scan.
- First-scan completion seeds the **backlog**, not the prompt enqueue
  (assert via the `_on_source_committed` hook *not* firing for startup sources).
- Static-only config reports a timestamp and seeds the backlog (no-watcher path).
- A **failed** first scan leaves `in_progress=false`, timestamp `null`, and
  retries on the next tick (drive via the `finally`).
- Existing startup tests that assert "all sources present immediately after setup"
  → wait on `full_scan_in_progress==false` via `health` first.

**Acceptance:** server binds and reports `SERVING` without blocking on the scan;
catalog appears (all at once) when the background scan completes; freshness
fields track it; static-only configs still report a timestamp. **Risk:** the
`SERVING` contract change — mitigated by Phase 2. Startup `Ctrl+C` becomes clean;
the launch-path sidecar (**`cli.py:971`**) no longer waits on the scan.

---

## Phase 4 — Stream first-scan additions (Option B) ✅ (complete)

**Landed (`source_manager.py`):** `_handle_rescan` now, on the *first* full scan
(`force_full and not _initial_scan_done`), wires the discovery state's
`on_source_added` to a new `_stream_first_scan_add`, so each source is committed
(via `_commit_add_claim` → register + metadata sync + signature + precache gate)
the moment the walk claims it — the catalog grows within the walk instead of in
one end-of-walk batch. The implementation is small because the claim phase
already applies the stability gate (`path_filter`), so deferred/unstable entries
are never claimed and therefore never streamed; and the existing end-of-walk
`_reconcile_discovered_state` runs unchanged, now idempotent (discovered ==
current ⇒ no adds/removes). Steady-state rescans are untouched (no callback
wired). Streamed adds route to the precache *backlog*, not the prompt enqueue,
because `_initial_scan_done` is still False during the first scan.

**Sharp edge handled:** `_commit_add_claim` *unregisters* on a duplicate add, so
a retried first scan (after a partial failure) would delete already-streamed
sources. `_stream_first_scan_add` guards with a presence check, so re-streaming
is a no-op and the duplicate-rollback is never hit.

**Tests (`source_manager_test.py::TestProgressiveStreaming`, 4 new):** sources
are registered *before* reconcile runs (the streaming assertion) with no
double-register; a steady-state full rescan does *not* stream (adds go through
batch reconcile); a retried first scan neither re-registers nor unregisters the
streamed set; an unstable first-scan entry is deferred (not streamed) and picked
up by a later rescan. Full tensor-server suite green (797 passed, 1 skipped).

> Concurrency note: the design's "read handlers never raise during an in-progress
> first scan" is covered structurally — catalog mutation/reads already serialize
> on `_sources_lock` (Phase 3 / the design's "already progressive-safe" finding),
> and `register_source` goes through that lock. Not re-tested with a threaded
> server here.

---

### Original plan (for reference)

**Goal:** true within-walk progressive population at startup — sources become
visible as they are claimed, not only at end-of-walk.

**Changes — `source_manager.py`:**
- Add a first-scan path that, while `not self._initial_scan_done`, emits each
  claim straight into `_commit_add_claim` (**`:996`**) from the claim phase
  instead of collecting into a `DiscoveryState` for a single
  `_reconcile_discovered_state` (**`:839`**). Apply the **same stability gate**
  the reconcile path uses — deferred/unstable claims are *not* streamed; they are
  picked up by the next steady-state rescan.
- The **end-of-first-scan reconcile still runs** to (a) set `_last_full_scan_at`
  / clear `full_scan_in_progress`, (b) flip `_initial_scan_done`, (c) establish
  the confirmed snapshot steady-state diffs against. It must be **idempotent** for
  already-streamed adds (re-adding an existing claim is a no-op; signatures
  already set).
- Steady-state reconcile is **untouched** — only the add-only first scan streams
  (the empty/force-full first scan has no removals, so there is no diff to
  compute; this is what makes streaming low-risk).

**Tests:**
- Catalog **grows during** the first scan — sources visible before the walk of a
  large root finishes (the core Option B assertion; simulate a slow walk).
- A deferred/unstable claim is **not** streamed and appears on the next rescan.
- `list_flights` / `get_flight_info` during an in-progress first scan **never
  raise** (more interleaving than Option 0; catalog mutation/read already locked).
- End-of-first-scan reconcile is a no-op for already-streamed adds (no double
  register / no precache double-enqueue).

**Acceptance:** sources appear incrementally during the first scan; steady-state
behavior unchanged; no raises from read handlers mid-scan. **Risk:** the new
first-scan code path (a second path alongside steady-state reconcile) — contained
by special-casing only the add-only first scan and keeping reconcile idempotent.

---

## Phase 5 — Docs ✅ (complete)

**Landed:**
- `biopb-tensor-server/ARCHITECTURE.md` (the `CLAUDE.md` symlink target): the
  Registration section now defines `mark_ready()`/`SERVING` as "serving the
  possibly-still-populating catalog" and documents the two `health` freshness
  fields; the Directory Monitoring section describes progressive/streamed startup.
- Root `development.md` (the root `CLAUDE.md` symlink target): the `health`
  `do_action` bullet notes progressive startup + the freshness fields.
- `docs/progressive-discovery.md` Status flipped to "Implemented (Option B)" with
  a pointer to this plan.

---

## Sequencing & dependencies

```
Phase 0 (audit) ──┐
                  ▼
Phase 1 (server fields, inert) ──► Phase 2 (consumer gating)
                                          │
                                          ▼
                                   Phase 3 (background, Option 0)  ◄── flips mark_ready() early
                                          │
                                          ▼
                                   Phase 4 (Option B streaming)
                                          │
                                          ▼
                                   Phase 5 (docs)
```

- Phases 1 and 2 are safe to land while the scan is still synchronous (fields
  inert). **Phase 3 must not land before Phase 2** (the contract flip).
- Phase 4 is an isolated diff on top of a working background scan, so it can be
  reviewed against a known-good Option 0 baseline (the design's rationale for the
  3→5 split).
- Each phase is one PR against **`dev`** (per repo convention). Lint/format runs
  via the pre-commit hooks (ruff) — no manual ruff step. Run tests via the root
  `.venv`.

## Out of scope (design Follow-ups)

- **Option A** (per-source-dir reconcile) — only if steady-state multi-root
  staleness bounds are wanted.
- **Static directory expansion** backgrounding (`resolve_all_sources`,
  `cli.py:174`) — persistent DuckDB will largely moot its restart cost.
- **Persistent DuckDB catalog** — turns the background scan into a revalidation;
  the freshness timestamp is then exactly the staleness signal a client needs.
