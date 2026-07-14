# Progressive discovery & catalog freshness

**Status:** implemented (Option B streaming). The server reaches `SERVING`
immediately and streams its bootstrap scan in the background; the `health` action
carries `full_scan_in_progress` / `last_full_scan_finished_at` as the freshness
signal. Touches `biopb-tensor-server` (+ a client indexing-state hint in
`biopb-mcp` and the webapp).

## Why

The startup discovery scan can take a long time on a large data directory. It used
to run **synchronously before the server reported `SERVING`**, so clients waited
through the whole scan even though partial results were servable almost
immediately. Progressive discovery makes startup: reach `SERVING` right away,
populate the catalog in the background, and expose a **separate freshness signal**
for how up-to-date the catalog is.

**A timestamp, not a "scan complete" boolean.** Once the catalog persists to DuckDB
(planned), a one-shot "initial scan done" milestone stops meaning anything — on
restart the catalog is already populated from disk. The useful question is *how
fresh* it is, which `last_full_scan_finished_at` answers, and it **unifies boot
with steady state**: the periodic full rescan (`full_rescan_interval`, default 1h)
advances the same value, so there is no special "startup" concept.

## What `SERVING` now means

`mark_ready()` / `SERVING` is **redefined** to "the server is up and serving the
(possibly still-populating, possibly persisted) catalog" — *not* "the catalog is
complete." `mark_ready()` is called early, right after the watcher / SourceManager
are wired and started, instead of after the scan. Freshness is carried by two
`health` fields, never by `SERVING`:

- `full_scan_in_progress: bool` — a full scan is running right now.
- `last_full_scan_finished_at: float | null` — epoch seconds; `null` until the
  first full scan succeeds. Reuses `SourceManager._last_full_rescan_at` ("last time
  the whole tree was fully reconciled"), advanced only by a **force-full** rescan
  (incremental rescans, which skip stable/cloud subtrees, leave it untouched).

## How it works

**The data plane was already progressive-safe** — the conversion was mostly
plumbing:

- The gRPC server binds and serves in `FlightServerBase.__init__`, before any scan
  runs; `serve()` only parks the calling thread.
- Catalog mutation (`register_source` / `unregister_source`) and reads
  (`list_flights` / `get_flight_info` / `do_get`) already serialize on
  `_sources_lock`, and `list_flights` already skips any source whose descriptor
  isn't fully built — so the wire layer never assumed a complete catalog.

**Background the scan.** The monitored bootstrap scan runs in the SourceManager's
event-loop thread instead of synchronously before `mark_ready()`. The watcher fires
its first rescan immediately (`initial_immediate=True`) rather than after the
rescan interval. This also unblocks the launch-path HTTP sidecar and makes a
startup `Ctrl+C` clean.

**Option B — stream the first scan.** Backgrounding alone still makes the catalog
appear in one batch at end-of-walk, because the reconcile computes a removal diff
(`removed = current − discovered`) that needs the *whole* snapshot. Option B makes
population progressive *within* the walk: on the **first** scan, each source is
registered the moment the walk claims it (`_stream_first_scan_add` →
`_commit_add_claim`), deferring only *removals* to the end-of-walk reconcile.

This is low-risk precisely because the first scan is **add-only**: the catalog
starts empty and force-full, so there are no removals and no diff to compute —
every claim is a pure add that can register immediately. Steady-state rescans keep
the unchanged snapshot-diff model (no streaming). Scope is monitored directories;
static explicit sources stay seeded synchronously.

**The client self-heals.** Every `health`/catalog consumer tolerates a
partial/growing catalog via an existing re-list mechanism — `biopb-mcp`'s
`_source_watch_loop` re-lists when `source_count` changes; the webapp polls
`listSources()`. The only two surfaces that *misled* (showing "No sources" on an
early/empty catalog) now branch on `full_scan_in_progress` to show "Indexing… (N so
far)": the napari tensor-browser widget and the webapp `SourceTree`.

## Gotchas

- **Precache boundary.** The startup set must warm the precache **backlog** (slow,
  idle-time), not the prompt **enqueue**. The gate is `_initial_scan_done` (set at
  end of first scan), *not* `_runtime_phase` — backgrounding runs the scan after
  `start()`, so `_runtime_phase` would already be true and wrongly prompt-enqueue
  every startup source. Streamed first-scan adds route to the backlog; live
  additions after boot prompt-enqueue.
- **The stability gate holds for streamed adds.** Unstable / recent-mtime entries
  (the "0 sources on fresh data" artifact) are deferred by the claim phase, so they
  are never claimed and never streamed; the next steady-state rescan picks them up.
- **Duplicate-add sharp edge.** `_commit_add_claim` *unregisters* on a duplicate
  add, so a retried first scan (after a partial failure) would delete
  already-streamed sources. `_stream_first_scan_add` guards with a presence check,
  making re-streaming a no-op.
- **End-of-first-scan reconcile still runs** — idempotently for already-streamed
  adds — to stamp the freshness timestamp, clear `full_scan_in_progress`, flip
  `_initial_scan_done`, and establish the confirmed snapshot steady-state diffs
  against.
- **Static-only / no-watcher configs.** `SourceManager.start()` returns early with
  no watcher, so the event loop never runs. `cli.py` drives the completion path
  directly (stamp the timestamp, seed the backlog) so a purely static config still
  reports freshness — and a `background_scan_running` fallback scans synchronously
  if the watcher failed to start, preserving registration.
- **Incremental rescans** don't toggle the flag or timestamp (both live inside `if
  force_full_rescan`), so the timestamp keeps meaning "last *full* reconcile."
- **The `SERVING` contract change is the main risk.** A consumer that reads
  `SERVING` as "catalog complete" would flash an empty catalog; every "no data" UI
  must gate on `full_scan_in_progress` (done for the two surfaces above).

## Not done / future

- **Option A (per-root reconcile)** — scope the removal diff per monitored root so
  each root's sources appear as it finishes (bounds steady-state staleness for
  multi-root configs). Deferred; heavier than Option B and not needed for "serve
  ASAP."
- **Static directory expansion** (`resolve_all_sources`) stays synchronous;
  persistent DuckDB will largely moot its restart cost.
- **Persistent DuckDB catalog** — once landed, startup serves the persisted catalog
  immediately and the background scan becomes a revalidation; the freshness
  timestamp is exactly the staleness signal a client needs.
