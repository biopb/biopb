# Discovery, the catalog and its freshness

Scope: `biopb-tensor-server` (`core/discovery.py`, `sources/source_manager.py`,
`sources/reconciler.py`), plus the client indexing-state hint in `biopb-mcp` and
the webapp.

The server binds and serves before it has scanned anything. Discovery then fills
the catalog from behind `SERVING`, and a separate signal says when that is
finished. This page states what discovery guarantees, in particular **which path
string is used where**, and how the first scan differs from every later one.

## 1. Model

```
 config / drop ──► root (resolved once)
                      │
        discover_sources(root) ── one walker ──► scratch DiscoveryState
                      │                              (claims found this pass)
                      ▼
        Reconciler: diff against the confirmed DiscoveryState
                      │  add / refresh / remove
                      ▼
   register_source (server) + sync_source_added (catalog row)
```

- **One walker.** A drop, a one-shot directory and the periodic rescan of a
  monitored root all call `discover_sources`. It keeps nothing between calls: each
  call stats the whole tree under its root that it does not prune. There is no
  scan snapshot.
- **A walk fills a scratch state; the Reconciler owns the confirmed one.** The
  confirmed state is what is registered and served. A walk never mutates it; only
  a commit does (`_commit_add_claim`, `_refresh_claim`, `_commit_remove_source`).
- **One lock.** Ticks, drops and `remove_source` all hold `_catalog_lock`, so the
  confirmed catalog has a single writer.

| Source kind | Found by | When | Removed by |
|---|---|---|---|
| Static (typed entry, a file, one remote source) | Seeded as a claim, no walk | Once, at construction | Never (config change) |
| Scan-once directory (`monitor = false`) | Walker, `_register_root` | First tick only, then again on a drop of it | A drop of the same root |
| Monitored directory | Walker, `_rescan_monitored_dirs` | Every tick | The reconcile diff (§5) |
| Dropped path (`add_local_source`) | Walker, `_register_root` | On request | A re-drop of the root, or `remove_source` for `dnd://` roots |
| Upstream tensor-server mirror | Upstream catalog query, not the walker | Adaptive cadence per upstream | The upstream re-list |

## 2. Where state lives

| Holder | State | Keyed by |
|---|---|---|
| `DiscoveryState` (scratch and confirmed) | `claims`, `path_to_source`, `source_to_paths`, `consumed_paths`, `visited_identities` | `source_id`; claim path strings; file identities |
| `Reconciler` | `_source_signatures`, `_missed_scans`, `_cloud_source_ids`, `_failed_sources`, `_path_to_source_id` | `source_id`; for the last, `claim.primary_path` |
| `SourceManager` | `_monitored_dirs`, `_cloud_roots`, `_dropped_cloud_roots`, `_monitored_aliases`, `_unavailable_roots`, the pending scan-once list | Resolved root `Path`s; for `_dropped_cloud_roots` a `dnd://<basename>` string |
| Adapter | `_source_url` (the raw claim path, or the library's own filename for hdf5 / nifti / bioio / dicom), `catalog_url` | Opens files with the raw path |
| Catalog (`sources` table) | `source_url` (display only), `tensors`, `metadata_json`, `indexed_at` | `source_id` |

Nothing is persisted by path. The `sources` table is truncated when the database
opens, so every boot rediscovers from disk.

## 3. Path invariants

A claim's `primary_path` and every `member_paths` entry is **the string the walk
produced**: the root as given to `discover_sources`, plus the names walked below
it. No adapter `claim()` normalizes the path it returns.

1. **A root is canonical, and becomes canonical once.** Config roots (`local_path`),
   drops and static seeds go through `resolve_local_path` (folds `file://`,
   symlinks, `..`, case, trailing separators). The monitored scan walks each stored
   root as given and never resolves it again, so its claims are spelled under the
   stored root even if that path later becomes a link (a migration that leaves one
   behind). `_monitored_dirs`, `_monitored_aliases` and `_cloud_roots` hold the same
   strings the walk uses.
2. **A claim path is never resolved to key or look it up.** All of
   `DiscoveryState.claims` / `path_to_source` / `source_to_paths` /
   `consumed_paths`, `_path_to_source_id` and the signature maps use the claim
   strings exactly as the walk spelled them.
3. **"Under a directory" is lexical.** A claim is under `D` when its string is
   `is_relative_to(D)` for a canonical `D`: a monitored root, a drop root, a
   declined directory, a cloud root. A claim is never resolved to answer this. A
   symlinked file leaf (`root/link.tif`) is under `root`, wherever it points; it
   would not be if resolved.
4. **"Same file" is answered by identity, not spelling.** `get_file_identity` is
   device and inode (resolved-path hash where the inode is synthetic), computed on
   the resolved target. The walk's `visited_identities` drops hardlinks and a
   file reached by two links. `generate_source_id` hashes the resolved path, so a
   link and its target are one `source_id` however the claim is spelled. Moving
   the data therefore gives it a new id: the old one goes by the two-miss rule and
   the new one is added behind it. A remote URL is hashed as written, minus
   trailing `/`.
5. **Cycle guards use real paths and decide entry only.** `_real_dir`
   (`realpath`), the identity set, `_leads_back_up` and `MAX_WALK_DEPTH` stop the
   walk from entering a directory. They never change how a path is spelled. A
   directory refused for any reason is added to `declined_dirs` in the walk's
   spelling, so lexical comparison against it is consistent (rule 3).
6. **I/O follows the claim path.** `open` and `stat` of a claim path follow links,
   so a link claim reads and signs its target. Quietness and change signatures stat
   the spelled path (`entry_is_quiet`, `build_entry_signature`).
7. **The catalog `source_url` is display, never an input.** It is
   `to_catalog_url(raw)` (forward slashes, `file://`) or an override: `dnd://<name>`
   for a drop, an alias root, `cache://`, `scratch://`, `grpc://<alias>/…` for a
   mirror. It is not fed to the filesystem or to `generate_source_id`, and rows are
   found and removed by `source_id`. The one lookup by url is `remove_source`,
   which matches the `dnd://` prefix.
8. **A remote claim is a URL.** `is_remote_url` claims are never resolved and are
   skipped by every path comparison.

Links are the one way a claim reaches outside its root. A symlinked file or dataset
(`.tif`, `.zarr`) inside a root is claimed under the link's spelling and read through
it. A symlinked plain directory is not entered. No check is made on where a followed
link points.

## 4. The walk

`walk_with_identity_tracking` yields `root/name` for every entry it does not
decline; `discover_sources` offers each to the adapters, first claim wins
(`AdapterRegistry.get_claims_for_path`). An adapter consumes the paths it takes
with `try_claim_path`, which fills `member_paths`; the walk does not descend below
a claimed directory.

An entry is declined, not yielded, when:

- it is hidden, a system or cloud-provider directory, or (outside a cloud root) an
  offline placeholder file (`should_skip_walk_entry`);
- the caller's `path_filter` rejects it. The monitored rescan passes
  `_should_claim`, the **stability gate**: an entry is claimable only once the later
  of its own mtime and ctime is older than `stability_window` (30 s by default; `0`
  disables the gate and the removal and rebuild shield). It applies to the entry's
  own stat, not to what is inside a directory;
- it is a directory that leads back up, or lies more than 64 levels below the root.

A declined **directory** goes into `WalkReport.declined_dirs`; a declined file does
not (that would make the set O(files)). A declined directory means "not looked at",
so what is registered below it is not taken for gone. Under a cloud root the
stability gate is bypassed and placeholders are admitted as unresolved claims.

One `DiscoveryState` is shared by all roots of a pass, so overlapping roots cannot
claim a subtree twice.

## 5. The reconcile diff (monitored roots)

After walking every monitored root into one scratch state, `_reconcile_discovered_state`
compares it with the confirmed claims that lie under a monitored root:

- **Added** (found, not confirmed): committed, unless the source is in failure backoff.
- **Changed** (found and confirmed, signatures differ): rebuilt in place with
  `replace=True` when quiet, so the source is never absent from the catalog and a
  failed rebuild keeps the old adapter. A source that fails to rebuild repeatedly is
  removed.
- **Removed** (confirmed, not found): only after `_MISSES_BEFORE_REMOVAL = 2`
  consecutive passes **and** the claim is quiet. A claim found again forfeits its
  count. Quiet is `entry_is_quiet` over the claim's own member paths, the same test the
  claim gate applies on the way in; a member that cannot be stat'd counts as quiet,
  since gone is what removal acts on.
- **Preserved** (declined): `_preserve_skipped_claims` carries forward confirmed
  claims under a declined directory, so a gate, a skip or a refused directory never
  removes what is registered below it. A monitored root that is not listable (unmounted
  drive, share down, deleted) is recorded as declined and keeps its sources until it is
  back.

**Cloud roots** are enumerated only on a full pass. On an incremental tick their
sources are left out of the candidate set (no diff, no removal, no miss counted), and
their signatures are identity-only so hydration and eviction do not flap a source.

**Full versus incremental.** A full pass runs when `full_rescan_interval` (default
3600 s) has elapsed since the last one, and on the first tick. Only a full pass walks
cloud roots and advances `last_full_scan_finished_at`. `full_rescan_interval = 0`
means no full pass ever runs: cloud roots are never walked, and the first tick is an
ordinary batched diff, not a streamed scan.

### Drops and scan-once roots

`_register_root` walks the root into a scratch state with **no stability gate**, so a file finished a moment before the drop is claimed; the user asked for it now. The gate applies to claiming only in the monitored rescan. Then `_remove_unclaimed_under`
removes what is gone **under that root only** (the periodic diff is whole-catalog, so it
cannot be reused on a subtree). A source is removed when it is under the root, is not
under a monitored root (the rescan does that, with its two-miss rule), is not under a
declined directory, is quiet, and is not claimed again by an adapter on a second look
(a drop has no later pass, so a transient decline must not cost a working source).
A single-file drop skips this scan. Then each claim is refreshed if known, else added.
A drop whose path lies inside an already-owned directory source is rejected
(`_find_containing_source` resolves the dropped path and looks each ancestor up in
`path_to_source`).

## 6. First scan versus re-scan

The first tick **is** the first scan. `start()` schedules it immediately, and
`_handle_rescan` runs the same steps every tick:

```
_scan_pending_roots        scan-once directories (first tick only)
_rescan_monitored_dirs     walk monitored roots, reconcile
_reconcile_due_upstreams   mirror upstreams that are due
complete_initial_scan      first tick only
```

| | First tick | Later ticks |
|---|---|---|
| Catalog on entry | Empty | Populated |
| Pass kind | Full (the interval has never elapsed) | Incremental, full once per `full_rescan_interval` |
| Registration | **Streamed**: `on_source_added` → `_stream_first_scan_add` commits each claim as the walk finds it, so the catalog grows during the walk | **Batched**: one `_reconcile_discovered_state` after the walk |
| Removal | None possible; every claim is a pure add | Two-miss rule, shielded and scoped as in §5 |
| Cloud roots | Walked | Skipped unless the pass is full |
| Scan-once roots | Scanned | Not touched |
| Upstreams | All due (countdown 0) | Adaptive: every tick while changing or failing, doubling toward `full_rescan_interval` while stable |
| Precache | Everything registered routes to the slow backlog | New sources prompt-enqueue |
| Freshness signal | Set by `complete_initial_scan` at the end | Advanced only by a completed full pass or upstream-only pass |

Streaming is safe only because the first scan is add-only. It is idempotent
against a retry: `_stream_first_scan_add` skips a claim already in the confirmed state,
because `_commit_add_claim` unregisters on a duplicate add and a retried scan would
otherwise delete what it had already streamed. The end-of-walk reconcile still runs and
is a no-op for streamed claims. The stability gate holds while streaming, so an unstable
entry is never claimed or streamed and is picked up by a later tick.

**`complete_initial_scan` runs at the end of the first tick**, whatever it scanned,
including nothing (a config of static sources only). It stamps `last_full_scan_finished_at`
(`_mark_catalog_complete`, which also advances the orphan clock), clears
`full_scan_in_progress`, flips `_initial_scan_done` and fires the completion hook, once.
It is last on purpose: the scan-once directories, the monitored walk and the upstream
pass have each had their first pass, so the upstream mirror is part of the startup set. An
unreachable upstream delays the flip by one failed attempt, not indefinitely. If the first
tick raises, `full_scan_in_progress` is cleared and the next tick retries the first scan.
`_initial_scan_done`, not the runtime-phase flag, is the precache gate: the scan runs after
`start()`, so the runtime flag is already true and would prompt-enqueue every startup
source.

A drop that arrives before the first tick finishes waits on `_catalog_lock`, heart-beating
to its caller, then runs as a live addition.

## 7. SERVING versus freshness

`SERVING` means the server is up and answering, not that the catalog is complete.
`mark_ready()` flips `health` from `STARTING` to `SERVING` once the `SourceManager` is
wired and started, before any scan.

Freshness is a separate, continuous signal on `health`:
`full_scan_in_progress: bool` (raised by the launcher before the manager starts) and
`last_full_scan_finished_at: float | null` (epoch seconds, `null` until the first scan
completes). The same fields serve boot and steady state: a later full pass (a forced full
rescan, or the pass of an upstream-only config) advances the timestamp exactly as the
first one did, so a client has no startup case to special-case. Incremental ticks touch
neither field.

The data plane tolerates a growing catalog: `register_source` / `unregister_source` and
the reads serialize on `_sources_lock`, and `list_flights` skips a source whose
descriptor is not built. `biopb-mcp`'s `_source_watch_loop` re-lists when `source_count`
changes, and the webapp polls the source list. The napari tensor-browser and the webapp
`SourceTree` show "Indexing... (N so far)" while `full_scan_in_progress` is set. A
consumer that reads `SERVING` as "catalog complete" will flash an empty catalog; gate
"no data" on `full_scan_in_progress` instead.

## 8. Limits and open items

- **`dnd://<basename>` is not unique.** Two drops with the same folder name share a
  `_dropped_cloud_roots` key and a `remove_source` prefix, so removing one removes both.
- **A retargeted configured symlink is not followed until restart.** Only its first
  target was stored as the root.
- **Per-root reconcile.** The removal diff covers all monitored roots at once, so
  steady-state staleness is bounded by the slowest root, not surfaced root by root.
- **The catalog is not persisted.** A persisted catalog would let startup serve at once and
  turn the first scan into revalidation, with `last_full_scan_finished_at` as its staleness
  signal.
