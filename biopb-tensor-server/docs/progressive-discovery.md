# Progressive discovery & catalog freshness

Scope: `biopb-tensor-server`, with a client indexing-state hint in `biopb-mcp`
and the webapp.

## SERVING vs. freshness

`SERVING` means the server is up and serving the catalog, not that the catalog
is complete. `mark_ready()` flips `health`'s status from `STARTING` to
`SERVING` as soon as the watcher / `SourceManager` are wired and started --
before the bootstrap scan runs. A large data directory would otherwise force
every client to wait through the whole scan even though partial results are
servable almost immediately.

Freshness is a separate, continuous signal, not a boot milestone: `health`
carries `full_scan_in_progress: bool` and `last_full_scan_finished_at: float |
null` (epoch seconds, `null` until the first full scan succeeds). The same
value backs both boot and steady state -- the periodic full rescan
(`full_rescan_interval`, default 3600s) advances it exactly like the initial
scan, so there is no separate "startup" concept for a client to special-case.
Incremental rescans, which skip stable/cloud subtrees, touch neither field;
only a force-full rescan does.

## How the catalog populates

The data plane is progressive-safe independent of discovery: the gRPC server
binds and serves in `FlightServerBase.__init__` before any scan runs, and
catalog mutation (`register_source` / `unregister_source`) and reads
(`list_flights` / `get_flight_info` / `do_get`) already serialize on
`_sources_lock` -- `list_flights` skips any source whose descriptor isn't
fully built. The bootstrap scan itself runs in the `SourceManager`'s
event-loop thread rather than blocking `mark_ready()`, and its first rescan
fires immediately instead of waiting out the rescan interval.

Within the first scan, each source registers the moment the walk claims it
(`_stream_first_scan_add` -> `_commit_add_claim`), rather than appearing in
one batch at end-of-walk. This is safe because the first scan is add-only --
the catalog starts empty and force-full, so there is no removal diff to
compute and every claim is a pure add. `_initial_scan_done` marks the
boundary: it flips once, at the end of the first successful full scan, and
gates the behavior below. Steady-state rescans keep the ordinary
snapshot-diff model (`removed = current − discovered`), since only they need
to detect removals; they do not stream. Static, explicitly-configured sources
are seeded synchronously and never go through this path.

Every `health`/catalog consumer tolerates a partial, growing catalog:
`biopb-mcp`'s `_source_watch_loop` re-lists when `source_count` changes, and
the webapp polls `listSources()`. The napari tensor-browser widget and the
webapp `SourceTree` -- the two surfaces that would otherwise show "No
sources" on an early/empty catalog -- branch on `full_scan_in_progress` to
show "Indexing... (N so far)" instead.

## Gotchas

- **Precache boundary.** The startup set warms the precache backlog (slow,
  idle-time), not the prompt enqueue. The gate is `_initial_scan_done`, not
  the general runtime-phase flag -- the scan runs after `start()`, so that
  flag is already true and would otherwise prompt-enqueue every startup
  source. Streamed first-scan adds route to the backlog; live additions after
  boot prompt-enqueue.
- **The stability gate holds for streamed adds.** Unstable / recent-mtime
  entries are deferred by the claim phase, so they are never claimed or
  streamed; the next steady-state rescan picks them up.
- **Duplicate-add sharp edge.** `_commit_add_claim` unregisters on a
  duplicate add, so a retried first scan (after a partial failure) would
  otherwise delete already-streamed sources. `_stream_first_scan_add` guards
  with a presence check, making re-streaming a no-op.
- **End-of-first-scan reconcile still runs**, idempotently for
  already-streamed adds, to stamp the freshness timestamp, clear
  `full_scan_in_progress`, flip `_initial_scan_done`, and establish the
  snapshot steady-state diffs against from then on.
- **Static-only / no-watcher configs.** `SourceManager.start()` returns early
  with no watcher, so the event loop never runs; `cli.py` drives the
  completion path directly (stamp the timestamp, seed the backlog) so a
  purely static config still reports freshness, and falls back to a
  synchronous scan if the watcher failed to start.
- **A consumer that reads `SERVING` as "catalog complete" is wrong** and will
  flash an empty catalog; gate "no data" UI on `full_scan_in_progress`
  instead.

## Not done / future

- **Per-root reconcile.** The removal diff is computed over the whole
  monitored tree, not scoped per root, so a multi-root config's steady-state
  staleness is bounded by the slowest root rather than surfacing sources
  root-by-root as each finishes.
- **`resolve_all_sources` (static directory expansion) is still synchronous**
  at startup.
- **The catalog itself is not persisted.** `MetadataDatabase`'s `sources`
  table is truncated on open (only `rois` and `decode_rates` survive a
  restart), so every boot re-discovers from disk; a persisted catalog would
  let startup serve immediately and turn the background scan into pure
  revalidation, with `last_full_scan_finished_at` as the staleness signal.
