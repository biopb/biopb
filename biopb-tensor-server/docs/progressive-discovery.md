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
boundary: it flips once, at the end of the first tick -- after the one-shot
directories, the monitored walk and the upstream mirror have each had their
first pass -- and gates the behavior below. Steady-state rescans keep the ordinary
snapshot-diff model (`removed = current − discovered`), since only they need
to detect removals; they do not stream. Static sources -- nothing to discover:
a typed entry, a file, one remote source -- are seeded synchronously and never go
through this path. A `monitor = false` directory is scanned once by the first
tick, ahead of the monitored walk (so it is startup set too), and never again.

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
- **End-of-walk reconcile still runs**, idempotently for already-streamed
  adds. On the first tick it does not stamp freshness or clear
  `full_scan_in_progress`; the tick does that once, at its end.
- **The loop always runs.** `SourceManager.start()` starts it for every config.
  The first tick runs the one-shot directories, the monitored walk and the
  upstream pass, then completes the startup protocol itself: clears
  `full_scan_in_progress`, stamps the freshness timestamp, flips the precache gate
  and fires the completion hook -- also for a config of static sources only. The
  upstream mirror is therefore startup set and routes to the precache backlog;
  an unreachable upstream delays the flip by one failed attempt. A first tick
  that raises clears `full_scan_in_progress` and the next tick retries. There is
  no launcher-driven fallback path.
- **A consumer that reads `SERVING` as "catalog complete" is wrong** and will
  flash an empty catalog; gate "no data" UI on `full_scan_in_progress`
  instead.

## Not done / future

- **Per-root reconcile.** The removal diff is computed over the whole
  monitored tree, not scoped per root, so a multi-root config's steady-state
  staleness is bounded by the slowest root rather than surfacing sources
  root-by-root as each finishes.
- **The catalog itself is not persisted.** `MetadataDatabase`'s `sources`
  table is truncated on open (only `rois` and `decode_rates` survive a
  restart), so every boot re-discovers from disk; a persisted catalog would
  let startup serve immediately and turn the background scan into pure
  revalidation, with `last_full_scan_finished_at` as the staleness signal.
