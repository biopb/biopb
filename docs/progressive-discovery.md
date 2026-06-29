# Progressive Discovery & Catalog Freshness Signals

**Status:** Design / not yet implemented
**Component:** `biopb-tensor-server`
**Related:** startup discovery scan, `mark_ready()`, the Flight `health` action, the
background precache worker.

---

## Problem

The tensor server runs a discovery scan at startup that, for a large data
directory, can take a long time. Today that scan runs **synchronously** before
the server reports `SERVING`, so clients wait through the whole scan even though
the server could serve partial results almost immediately. The goal is to
**convert startup to progressive discovery**: reach `SERVING` right away, let the
catalog populate in the background, and expose a separate signal for how *fresh*
the catalog is.

## Key finding — the data plane is already progressive-safe

The conversion is contained because most of the machinery already exists:

- The gRPC server **binds and starts serving in `FlightServerBase.__init__`**,
  before any scan runs (`server.py:207-214`). `serve()` only parks the calling
  thread; handler threads are already live.
- Catalog mutation and reads are already concurrency-safe: `register_source` /
  `unregister_source` mutate under `_sources_lock` (`server.py:292`, `:302`), and
  the read handlers (`list_flights`, `get_flight_info`, `do_get`) read through
  `_get_sources_snapshot()` under the same lock (`server.py:886`, `:414`).
  `list_flights` already tolerates a partial/growing catalog — it skips any source
  whose descriptor build fails and only yields a fully-built `FlightInfo`
  (`server.py:921`).
- Per-source metadata-DB sync already happens incrementally as each source
  registers, via `sync_source_added` (an upsert, `source_manager.py:1267`).

So the wire layer does **not** assume a complete catalog. The only real gate is
that `health` reports `STARTING` until `mark_ready()` runs *after* the scan
(`server.py:251`, `:576`; `cli.py:463`).

### Granularity caveat — the catalog updates once per *rescan*, at the end of the walk

A correction to a tempting misreading: although `register_source` is called
one-source-at-a-time, those calls all happen at the **end** of a rescan, not as
sources are encountered. `_handle_rescan` (`source_manager.py:227`) runs three
serial phases over **all monitored dirs at once**:

1. `_refresh_entry_state` — walks every monitored root into one combined
   `next_state` (`:374` loops over `sorted(self._monitored_dirs)`).
2. `discover_sources_from_entries` — claim phase over that whole combined
   snapshot (`:264`).
3. `_reconcile_discovered_state` — one diff + apply over the whole snapshot
   (`:279`); the per-source `register_source` calls happen here (`:881-882`).

So the catalog changes **once per rescan, covering all configured dirs together**,
only after the entire (slow) walk + claim completes. It is not per-source-dir
(that coincides with per-rescan only when a single root is monitored), and
nothing is registered mid-scan of a single dir.

The structural reason is the **removal diff**: `_reconcile_discovered_state`
computes `removed_ids = current_ids - discovered_ids` (`:866`), so it cannot know
a source was deleted until the full walk confirms its absence — reconcile needs
the complete snapshot. Backgrounding the scan (below) removes the blocking, but on
its own it does **not** make population progressive *within* the walk. See
[Population granularity options](#population-granularity-options).

## Why a freshness *timestamp*, not a "scan complete" boolean

A one-shot "initial scan complete" milestone stops being meaningful once the
metadata catalog is persisted to DuckDB (planned): on restart the catalog is
already populated from disk, so the interesting question is not *whether* an
initial scan finished but *how fresh* the catalog is. A
`last_full_scan_finished_at` timestamp answers that and **unifies boot with
steady state** — a full rescan already runs periodically
(`full_rescan_interval`, default 1h), advancing the same value, so there is no
special "startup" concept to expose.

This reuses `SourceManager._last_full_rescan_at` (`source_manager.py:127`, set at
`:289`), which already means "last time the whole tree was fully reconciled" and
is updated only on a successful **force-full** rescan (incremental rescans, which
skip stable/cloud subtrees, deliberately leave it untouched).

A cheap `full_scan_in_progress` boolean is added alongside it so a client can
also tell "a full scan is running right now" from "idle since T".

## New `health` action fields

In addition to the existing `status` / `source_count` / `uptime_seconds`
(`server.py:575-581`):

- `full_scan_in_progress: bool`
- `last_full_scan_finished_at: float | null` — epoch seconds; `null` until the
  first full scan succeeds.

`mark_ready()` / `SERVING` is **redefined** to mean "the server is up and serving
the (possibly still-populating, possibly persisted) catalog." Freshness is
carried by the two fields above, not by `SERVING`.

## Design decisions

1. **Background the monitored bootstrap scan** into the SourceManager's existing
   event-loop thread instead of running it synchronously before `mark_ready()`.
   This also unblocks the launch-path HTTP sidecar (today it cannot start until
   the scan returns, `cli.py:904→920`) and makes a startup `Ctrl+C` clean
   (`serve()`'s `try/except` is reached immediately).
2. **`mark_ready()` moves early.** `SERVING` no longer implies a complete
   catalog. This is the contract change that survives the move to persistent
   DuckDB.
3. **Stream the first scan's additions** so the catalog populates *within* the
   walk, not only at the end — see [Population granularity
   options](#population-granularity-options) (Option B is recommended).
4. **Scope: monitored directories only.** Static *explicit* sources stay seeded
   synchronously (cheap). Static *directory expansion*
   (`resolve_all_sources` in `_resolve_serve_sources`, `cli.py:174`) remains
   synchronous in this pass — see Follow-ups.
5. **Freshness reuses `_last_full_rescan_at`**, so boot and steady state share
   one mechanism.

## Population granularity options

Backgrounding the scan (decision 1) stops it blocking startup, but by itself the
catalog still appears in one batch at the end of the rescan (see the granularity
caveat above). To make population *progressive within the walk*, choose one of:

### Option 0 — background only (no streaming)

Just background the scan; keep the single end-of-walk reconcile. Simplest.
Catalog appears all at once when the full walk+claim+reconcile completes; a large
single root shows nothing until it finishes.

- **Pros:** smallest change; reconcile logic untouched; removal diff unaffected.
- **Cons:** not actually progressive *within* a dir — defeats the main goal for
  the common "one big monitored root" deployment.
- **Risk:** low.

### Option A — per-source-dir reconcile

Loop walk + claim + reconcile **per monitored root**, scoping the removal diff to
that root's subtree. Each configured dir's sources appear as that dir finishes.

- **Pros:** general (works for steady-state rescans too); bounds staleness to one
  root rather than the whole config; reuses the existing snapshot-diff model
  per-root.
- **Cons:** still nothing mid-scan of a *single* large root; moderate refactor.
- **Risk / care points:**
  - The removal diff must be scoped per root: `current_ids` / `discovered_ids`
    and `_is_monitored_claim` (`source_manager.py:884`) currently span all roots;
    scoping them wrong would unregister sources from *other* roots.
  - The cross-root `visited_identities` dedup (`:373`) is shared to avoid
    double-walking overlapping/nested roots; per-root passes must still share one
    identity set or overlapping roots get walked twice (or a shared subtree gets
    double-registered).
  - `_last_full_rescan_at` / `full_scan_in_progress` semantics: a "full scan" is
    now N per-root passes — decide whether the timestamp advances per root or only
    when all roots have completed (recommend: only when all complete, to preserve
    "whole tree reconciled").

### Option B — stream first-scan additions (recommended)

Register each source the moment it is claimed during the **first** scan, deferring
*removals* to the existing end-of-walk reconcile. Steady-state rescans keep the
unchanged snapshot-diff model.

This is clean precisely for the first scan: the catalog starts **empty and
force-full**, so there are no removals possible — every claim is a pure add and
can register immediately. There is no diff to compute, so the structural reason
reconcile needs the whole snapshot does not apply yet.

- **Pros:** true within-dir progressive population at startup (the actual goal);
  lowest-risk way to get it because the add-only first scan sidesteps the
  removal-diff dependency; steady-state reconcile logic is untouched.
- **Cons:** introduces a distinct first-scan code path alongside the steady-state
  one (two paths to maintain); only the first scan streams — subsequent full
  rescans still batch (acceptable, since after boot the catalog is already
  populated and only deltas matter).
- **Risk / care points:**
  - **Wire the stream through `_commit_add_claim`** (`source_manager.py:~995`) so
    streamed adds still go through `register_source` + `sync_source_added` +
    signature bookkeeping + the precache gate — i.e. emit per-claim from the claim
    phase instead of collecting into a `DiscoveryState` for a single reconcile.
  - **Stability window:** the normal path defers unstable/recent-mtime entries
    (the documented "0 sources on fresh data" artifact) and re-tries them on later
    rescans. The first-scan stream must apply the same stability gate per claim;
    anything deferred is simply picked up by the next (steady-state) rescan — do
    **not** stream an unstable claim just because it was seen.
  - **End-of-first-scan reconcile still runs** to (a) set
    `_last_full_scan_at` / clear `full_scan_in_progress`, (b) flip
    `_initial_scan_done`, and (c) establish the confirmed snapshot that
    steady-state diffs against. It should be a no-op for already-streamed adds
    (idempotent: re-adding an existing claim is a no-op; signatures already set).
  - **Precache boundary:** streamed first-scan adds must route to the *backlog*
    (slow, idle-time warm), not the prompt enqueue — same requirement as the
    `_initial_scan_done` gate elsewhere in this doc. Keep the enqueue gated until
    `_initial_scan_done` is set at end of first scan; seed the backlog from the
    streamed set in `_on_initial_scan_complete`.
  - **Concurrency:** streaming means handler threads observe a catalog that grows
    during the walk — already safe (`_sources_lock`), but more interleaving than
    Option 0, so worth an explicit test that `list_flights`/`get_flight_info`
    during an in-progress first scan never raise.

**Recommendation:** Option B for the first scan (delivers the goal at low risk),
optionally layering Option A later if per-root staleness bounds are wanted for
steady-state rescans of multi-root configs. Option 0 only if streaming is
deferred.

## Changes by file

### `server.py` — readiness/freshness state
- In `__init__` (near `_activity_lock`, `:220`): add
  `self._scan_status_lock = threading.Lock()`,
  `self._full_scan_in_progress = False`,
  `self._last_full_scan_at: Optional[float] = None`.
- Add setters called from the SourceManager thread:
  `set_full_scan_in_progress(bool)` and `set_last_full_scan(ts)`, each guarded by
  `_scan_status_lock`.
- Extend the health dict (`do_action`, `:575-581`) with `full_scan_in_progress`
  and `last_full_scan_finished_at`, read under the lock.

### `watcher.py` — fire the first rescan immediately
- `PeriodicRescanWatcher.start()` (`:80-88`): set
  `self._next_rescan_at = time.monotonic()` (immediate) instead of
  `+ rescan_interval`, so the background event loop performs the first full scan
  right away instead of after `rescan_interval` (default 30s). Keep it behind a
  default-`True` `initial_immediate` param to preserve existing test
  expectations where needed.

### `source_manager.py` — background scan, stream first-scan adds, push status, fix the precache boundary
- **Remove** the synchronous `manager._handle_rescan()` from
  `create_source_manager` (`:1453-1455`); the first scan now happens in the event
  loop after `start()`.
- **Stream the first scan (Option B).** Add a first-scan path that registers each
  claim as it is discovered rather than collecting into a single end-of-walk
  `_reconcile_discovered_state`. Concretely: have the claim phase emit per-claim
  into `_commit_add_claim` (`:~995`) when `not self._initial_scan_done`, applying
  the same stability gate the reconcile path uses (deferred/unstable claims are
  left for the next steady-state rescan, not streamed). The end-of-first-scan
  reconcile still runs to establish the confirmed snapshot and is idempotent for
  already-streamed adds. See [Population granularity
  options](#population-granularity-options) for the care points.
- **Swap the precache enqueue gate.** Today the prompt precache enqueue is gated
  by `_runtime_phase` (`:1015`), and the startup set is meant to go to the slow
  *backlog*, not the prompt *enqueue*. Backgrounding/streaming the scan would run
  it under `_runtime_phase=True` and wrongly route every startup source to the
  prompt enqueue. Fix: add `self._initial_scan_done = False` and gate the enqueue
  on it (`:1015`) instead of `_runtime_phase`. Streamed first-scan adds therefore
  do **not** prompt-enqueue; they are seeded into the backlog at first-scan
  completion (below).
- In `_handle_rescan` (`:227-289`): when `force_full_rescan` is true, push
  `self._server.set_full_scan_in_progress(True)` at entry and reset it to `False`
  in a `finally`. On success (`:288`) also push
  `set_last_full_scan(time.time())`. On the **first** successful full rescan, set
  `_initial_scan_done = True` and fire a new `self._on_initial_scan_complete`
  callback (best-effort, like the existing hooks).
- Add `self._on_initial_scan_complete: Optional[Callable[[], None]] = None`
  (`:105` area).

### `cli.py` — reorder and rewire (`_setup_flight_server`)
- Keep static seeding synchronous (inside `create_source_manager`,
  `:1439-1451`).
- Call `server.mark_ready()` **early** — right after the watcher/source_manager
  are wired and `start()`ed — instead of at `:463` after the full scan.
- Move the `seed_backlog(...)` call (`:470`) into the new
  `_on_initial_scan_complete` callback so the startup set is seeded into the
  backlog *when the first scan actually finishes*. Wire it before `start()`.
- **Static-only / no-watcher path:** `SourceManager.start()` returns early when
  there is no watcher (`:138`), so the event loop and the completion callback
  never run. After seeding, if there are no monitored dirs, call the completion
  path directly in cli — `server.set_last_full_scan(time.time())` and (if
  precache) `seed_backlog(...)` — so a purely static config still reports a
  freshness timestamp and warms its backlog.
- `metadata_db.initial_sync(...)` (`:430`) becomes redundant under per-source
  `sync_source_added` during the scan; keep it only for the synchronously-seeded
  static set, or drop it (idempotent upsert either way).

### Tests (`tests/`)
- Update startup tests that assert "all sources present immediately after setup"
  to wait on `full_scan_in_progress == False` / `last_full_scan_finished_at !=
  null` via the health action.
- Add: health exposes the two new fields; **catalog grows progressively during
  the first scan** (sources visible before the walk of a large root finishes —
  the Option B behavior); a deferred/unstable claim is *not* streamed and appears
  on the next rescan; first-scan-complete flips the flag and seeds the backlog
  (not the prompt enqueue); `list_flights`/`get_flight_info` during an in-progress
  first scan never raise; static-only config reports a timestamp; a failed first
  scan leaves `in_progress=False` and timestamp `null` and retries on the next
  tick.

### Docs
- `biopb-tensor-server/CLAUDE.md`: update the Discovery/Startup and health-action
  sections (progressive startup, redefined `mark_ready()`/`SERVING`, two new
  health fields). Root `CLAUDE.md` §2.2/§3 mention the startup scan in passing —
  light touch.

## Edge cases

- **First scan fails:** `finally` resets `full_scan_in_progress=False`;
  `_initial_scan_done` stays false; timestamp stays `null`; the next periodic
  tick retries; backlog not seeded until a full scan succeeds.
- **Concurrency:** the new server fields are written by the one event-loop thread
  and read by gRPC handler threads, guarded by `_scan_status_lock`. Catalog
  mutation/read is already locked.
- **Incremental rescans** do not toggle the flag or timestamp (both live inside
  `if force_full_rescan`), so the timestamp keeps meaning "last *full*
  reconcile."

## Risk

- **Main risk — the `SERVING` contract change.** A client that treats `SERVING`
  as "catalog complete" could briefly see an empty/partial catalog. **Pre-req:**
  audit health consumers — the biopb-mcp bootstrap health probe, the napari
  plugin, and the HTTP sidecar `/readyz` — and have them gate any "no data" UI on
  `full_scan_in_progress`.
- **Streaming (Option B) risk** is the new first-scan code path: a second path
  alongside the steady-state reconcile, plus correct handling of the stability
  gate and the idempotent end-of-scan reconcile. See the Option B care points.
  Mitigated by keeping steady-state reconcile untouched and special-casing only
  the add-only first scan.
- Everything else is low risk: catalog mutation/reads are already
  concurrency-safe and the freshness value is already computed internally.

## Suggested sequence

1. Audit the `health` consumers (biopb-mcp / napari / sidecar) — sizes the
   client-side blast radius.
2. `server.py` fields + health, `watcher.py` immediate tick (independent,
   testable alone).
3. `source_manager.py` backgrounding + status push + precache-gate swap
   (Option 0 baseline: background only, catalog still appears at end of scan).
4. `cli.py` reorder + callback wiring + static-only path.
5. Layer in Option B (first-scan streaming) on top of the working baseline, so
   streaming can be validated as an isolated diff against a known-good background
   scan.
6. Tests, then docs.

> Implementation note: steps 3–4 give a working *background* scan (Option 0); step
> 5 turns it into true within-dir progressive population (Option B). Splitting them
> keeps the streaming change reviewable in isolation. Option A (per-root reconcile)
> is deferred unless steady-state multi-root staleness bounds are wanted.

## Follow-ups (out of scope here)

- **Option A — per-source-dir reconcile** (see [Population granularity
  options](#population-granularity-options)): adopt only if steady-state full
  rescans of multi-root configs need per-root staleness bounds. Heavier than
  Option B and not required to meet the "serve ASAP" goal.
- **Static directory expansion** (`resolve_all_sources`, `cli.py:174`) remains
  synchronous; persistent DuckDB will largely moot its restart cost. Background
  it in a later pass if it proves to be a startup bottleneck for static configs.
- **Persistent DuckDB catalog**: once landed, startup serves a real (persisted)
  catalog immediately and the background scan becomes a revalidation; the
  freshness timestamp is exactly the signal a client needs to judge staleness.
