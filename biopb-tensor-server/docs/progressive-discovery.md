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
| Scan-once directory (`monitor = false`) | Walker, `_scan_root` | First tick only, then again on a drop of it | The same scan's removal (§5), at once |
| Monitored directory | Walker, `_scan_root` | Every tick | The same scan's removal (§5), after two misses |
| Dropped path (`add_local_source`) | Walker, `_register_root` | On request, once the first scan is done | A re-drop of the root (what vanished), or `remove_source` for a marked drop |
| Upstream tensor-server mirror | Upstream catalog query, not the walker | Adaptive cadence per upstream | The upstream re-list |

## 2. Where state lives

| Holder | State | Keyed by |
|---|---|---|
| `DiscoveryState` (scratch and confirmed) | `claims`, `_path_to_source`, `_source_to_paths`, `consumed_paths`, `visited_identities` | `source_id`; claim path strings; file identities |
| `Reconciler` | `_source_signatures`, `_missed_scans`, `_cloud_source_ids` | `source_id` |
| `Roots` (shared by `SourceManager` and `Reconciler`) | every known root: kind (monitored, scan-once, dropped, upstream), alias, cloud, `dnd://` label | Resolved root `Path`s; a drop's label |
| `SourceManager` | `_unavailable_roots` | Resolved root `Path`s |
| Adapter | `_source_url` (the raw claim path, or the library's own filename for nifti / bioio / dicom), `catalog_url` | Opens files with the raw path |
| Catalog (`sources` view) | `source_url` (display only), `tensors`, `metadata_json`, `indexed_at`; over `source_catalog` (a source with a claim: the claim, its claim-time signature and the adapter payload sit beside the row) and `sources_volatile` (the rest: mirrors, drops, uploads) | `source_id` |

Nothing is persisted by path, and nothing persisted is read back yet: both tables are
emptied when the database opens, so every boot rediscovers from disk. `source_catalog`
is written now so a later restart can restore from it.

## 3. Path invariants

A claim's `primary_path` and every `member_paths` entry is **the string the walk
produced**: the root as given to `discover_sources`, plus the names walked below
it. No adapter `claim()` normalizes the path it returns.

1. **A root is canonical, and becomes canonical once.** Config roots (`local_path`),
   drops and configured paths go through `resolve_local_path` (folds `file://`,
   symlinks, `..`, case, trailing separators). The monitored scan walks each stored
   root as given and never resolves it again, so its claims are spelled under the
   stored root even if that path later becomes a link (a migration that leaves one
   behind). `Roots` holds the same strings the walk uses.
2. **A claim path is never resolved to key or look it up.** All of
   `DiscoveryState.claims` / `_path_to_source` / `_source_to_paths` /
   `consumed_paths` and the signature maps use the claim strings exactly as the
   walk spelled them.
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
   `to_catalog_url(raw)` (forward slashes, `file://`) or an override: `dnd://<label>`
   for a drop from outside every known root, an alias root, `cache://`, `scratch://`, `grpc://<alias>/…` for a
   mirror. It is not fed to the filesystem or to `generate_source_id`, and rows are
   found and removed by `source_id`. The one lookup by url is `remove_source`,
   which matches a `dnd://<label>/` prefix.
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

## 5. One scan per root

A monitored root and a scan-once root are scanned the same way (`SourceManager._scan_root`),
one root at a time, each against its own claims:

1. Walk the root into a scratch state. A claim is committed the moment the walk finds it
   (`on_source_added` → `_stream_claim_add`), so the catalog grows within the walk.
2. Take the root's snapshot, `Reconciler.claims_under(root)`: the confirmed claims lying at
   or under the root, by the path each is spelled under (lexical, so a link belongs to the
   root it was found in).
3. `_reconcile_root(snapshot, discovered, recurring)` compares the two. A claim is never
   compared with another root's walk, so scanning one root cannot remove the claims of
   another.

Two things differ, both because a monitored root is walked again and a scan-once root is
not (`recurring`):

| | Monitored (`recurring`) | Scan-once |
|---|---|---|
| Stability gate on the walk and on a refresh or removal | Yes: an unstable entry is picked up by a later tick | No: nothing would pick it up |
| A claim the walk did not find | Removed after `_MISSES_BEFORE_REMOVAL = 2` walks running, once quiet | Removed at once, unless an adapter claims it again on a second look |
| Claim probes memoized across walks | Yes | No |

A cloud root is scanned only on a full pass; an incremental tick skips it and so leaves its
sources as they are. A root that cannot be listed is not scanned, so nothing under it is
removed. A directory the walk declined (the stability gate or the skip policy) is not
evidence that what is registered under it is gone: the root's claims under it are carried
into the comparison (`_preserve_skipped_claims`).

The scan assumes roots do not nest and that no file is reachable from two roots, which is
policy and not something the code can fully check (a link defeats a lexical test). Two
roots that reach the same file produce one `source_id` (it hashes the resolved path) spelled
two ways, and the claim belongs to the root whose walk committed it first. The stream hook
notices the second spelling (the id is already in state under a different path) and logs one
warning per source naming both paths.

For each root, the comparison decides:

- **Added** (found, not in the snapshot): committed by the walk, as it is found, and not
  again here. A claim whose registration fails is committed as a failed row, so it is known
  and the next walk does not try it again. One that did not commit when streamed (the
  catalog write failed) is found new by the next walk.
- **Changed** (found and confirmed, signatures differ): rebuilt in place with
  `replace=True` when quiet, so the source is never absent from the catalog and a
  failed rebuild keeps the old adapter. A rebuild that fails is not counted: only the
  walk removes a source.
- **Unchanged** and not registered: a source that waits for the pool (`pending`) is
  re-decided against its residency, and one that waits for a client (`needs_recall`)
  likewise, so a file that became resident is queued and one that stopped being
  resident waits for a client. A failed row is left alone.
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

### What each event writes to the catalog

A source with a claim under a monitored or scan-once root (cloud roots included) has its
row in `source_catalog` from the moment the walk finds it; a source with none (a mirror,
a drop, an upload) has it in `sources_volatile`. The row never changes table, and
registration fills it in.

| Event | Write |
|---|---|
| A new claim | Batched `INSERT ... ON CONFLICT DO NOTHING` while registration is deferred, else one statement: unresolved, `pending` if resident, else `needs_recall`; the claim, its signature, no payload |
| Known, signature unchanged, row resolved | None |
| Known, signature unchanged, row unresolved and not failed | `UPDATE` of the reason only: `pending` if resident, else `needs_recall` |
| Known, signature unchanged, row failed | None |
| Known, signature changed | Registers inline as a new claim would: one `UPDATE` of the resolved row on success; `needs_recall` if no longer resident. A failed rebuild leaves the old row and adapter |
| Gone | Delete |
| Registration succeeds | `UPDATE` of `metadata_json`, `tensors`, `payload`, `indexed_at`, resolved |
| Registration fails (a pending claim) | `UPDATE` to `failed` with `unresolved_error` |

"Resident" is `_claim_is_unresolved` being false: the adapter did not flag the claim, and
under a cloud root no member is a dehydrated placeholder (a metadata `stat`, nothing is
opened). A registration `UPDATE`s the pending row because that costs about two thirds of
the wall time and a quarter of the CPU of `INSERT OR REPLACE` on the indexed table. The
batched `DO NOTHING` also guards the race where a registration writes the real row before
the buffered pending one lands.

A failed row stays failed: nothing retries it on a timer, and the walk leaves it alone
while its signature is unchanged. A new signature for the same URL, a re-drop of its path
(`add_local_source`) or a `resolve` registers it again. There is no failure counter and no
removal for failing; only the walk removes a source.

The signature has known gaps: a cloud file's is `(dev, ino)` only, and a directory's is
its own stat, which does not move when a member is rewritten in place. A drop refreshes
every already-registered claim under the dropped path unconditionally, with no signature
compare, so a re-drop is the repair for both. It bypasses the stability window, inside a
cloud root it refreshes as cloud (a non-resident claim goes back to `needs_recall`
without being opened), and sources under a drop's own root stay volatile.

### Drops and scan-once roots

Where a drop lands decides what it may do. A **known root** is a monitored directory, a
`monitor = false` directory, or the root of an earlier marked drop; the test is lexical on
the resolved drop path (rule 3).

| The drop lands | Result | Display tree of new sources | `dnd://` mark |
|---|---|---|---|
| Inside a known root, the root itself included | Registers and refreshes as a rescan of that root would | The root's own: its alias if it has one (monitored or `monitor = false`); an earlier drop's label; otherwise the file url | An earlier drop's mark carries over to what the re-drop adds; a monitored or `monitor = false` root gets none |
| Outside every known root | Allowed if it overlaps nothing, otherwise refused | Its own root, labeled by the folder name (`exp`, then `exp (2)` for a second folder of that name) | Stamped |

Overlap means the walk found a source that is already registered, or an existing source lies
under the dropped path. It is refused before anything is removed or committed, and a cloud
consent that drop gave is taken back. Cloud mode cannot be switched on inside a known root:
it is set where a folder is first added, or in the config. The scheme is the removal key
(`remove_source("dnd://<label>")`), so a re-drop inside a marked drop stamps what it adds
with the same label and removing the drop takes it too. Removing a drop frees its label and
its cloud consent.

Drops are refused until the first scan has finished (a `ValueError`, which the server maps to
a clean error): both checks above need the whole catalog.

A drop is not a scan of a root: `_register_root` walks the dropped path into a scratch state
with **no stability gate** (the user asked for it now), then `_remove_unclaimed_under`
removes what is gone **under that path only**. A source is removed when it is under the path,
is not under a monitored root (the rescan does that, with its two-miss rule), is not under a
declined directory, is quiet, and is not claimed again by an adapter on a second look (a drop
has no later pass, so a transient decline must not cost a working source). A single-file
drop skips this scan. Then each claim is refreshed if known, whatever its signature, else
added: the signature never decides, which is what makes a re-drop the repair for the
signature gaps. (A configured `monitor = false` path is not a drop: it is scanned like a
monitored root, once.) A drop whose path lies inside an already-owned directory
source is rejected (`_find_containing_source` resolves the dropped path and looks each
ancestor up in `_path_to_source`).

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
| Registration | A new claim is **committed as the walk finds it** (`on_source_added` → `_stream_claim_add`, every tick), **registered afterwards** by a worker pool (below) while registration is deferred; with `registration_workers = 0` or once the first scan is over, registered as claimed | Same: new claims stream; the known ones are compared by `_reconcile_root` after each root's walk |
| Removal | None possible; every claim is a pure add | Two-miss rule, shielded and scoped as in §5 |
| Cloud roots | Walked | Skipped unless the pass is full |
| Scan-once roots | Scanned (the same scan, once) | Not touched |
| Upstreams | All due (countdown 0) | Adaptive: every tick while changing or failing, doubling toward `full_rescan_interval` while stable |
| Precache | Everything routes to the slow backlog, each source when its registration completes | New sources prompt-enqueue |
| Freshness signal | Set by `complete_initial_scan` at the end | Advanced only by a completed full pass or upstream-only pass |

### Registration after the walk

Registering a source opens and parses its file, which is most of a large site's start.
While `registration_workers` is above zero the first scan therefore only *claims*:
`_commit_pending_claim` commits each deferrable claim to the catalog only: a row built
from the claim (`is_resolved` false, `unresolved_reason` `pending`, no tensors) and no
adapter in the registry. Its claim and signatures are in the confirmed state, so the diff,
removal and refresh treat it like any other source. A `RegistrationWorker` pool then calls
`Reconciler.ensure_registered`, newest file first, which runs the ordinary registration and
puts the adapter in the registry.
The pool is held until the first scan is over (`complete_initial_scan` resumes it): walking
and registering contend, and a held pool lets every claim be committed first, so the whole
catalog exists, pending, before any file is opened. A rescan does not hold it again; it
registers what it finds inline.

- **`resolve` registers it at once; a read does not.** The server's read paths and the upload
  manager use `SourceRegistry.get_registered`, which on a registry miss asks the reconciler
  why: an unresolved error while the source waits ("open to resolve", the same as a cloud
  source), `SourceRegistrationError` for a failed one. Callers that only ask whether a source
  is registered keep `get`, which is `None` for a pending source. `resolve` calls the
  reconciler's `materialize`, single-flight per source, so a resolve racing the worker shares
  one registration; it registers the source (trying a failed one again: nothing else does)
  and returns the filled row.
- **Not deferred:** remote proxies (bulk-seeded), static sources, and everything claimed
  after the first scan. A cloud source is left unregistered for good: its row reads
  `needs_recall`, the pool never queues it, and only `resolve` downloads and registers it.
- **`unresolved_reason`** says why a row is not resolved: `needs_recall` (a cloud
  placeholder; resolving downloads it), `pending` (queued; resolving registers it, no
  download), `failed` (registration raised; `unresolved_error` holds the text,
  reads raise `SourceRegistrationError`). A failed row is not retried on a timer: it
  waits for a new signature from the walk, a re-drop of its path or a `resolve`.
- **While pending** a source can be refreshed (the rebuild is the registration) or removed,
  and a registration never registers a removed source back (a per-source lock orders the
  three). The upload attacher runs on the adapter once it is registered.
- **Precache.** One callback after every registration (`_notify_source_committed`) routes
  the source: one the first scan found goes to the backlog (`enqueue_backlog`, with its
  mtime), a later one is prompt-enqueued, a remote one is not enqueued as startup. The
  backlog tier waits for `registration_idle` (first scan over, nothing pending); the live
  tier is not held.
- **Cost.** `benchmarks/registration_cost_test.py` reports the per-source-type cost of a scan
  (`BIOPB_REG_COST_ROOT` points it at a site).

Streaming commits only claims that are new; a claim already known is left to the
end-of-walk reconcile, which compares its signature, and removals are only ever decided
after the walk. It is idempotent against a retry: `_stream_claim_add` skips a claim
already in the confirmed state, because `_commit_add_claim` unregisters on a duplicate
add and a retried walk would otherwise delete what it had already streamed. The
end-of-walk reconcile skips what was streamed (`_streamed_ids`, consumed once, so the
next rescan compares every claim). The stability gate holds while streaming, so an
unstable entry is never claimed or streamed and is picked up by a later tick.

While registration is deferred the pending rows of the claims a walk streams are written
in batches (`PendingRowWriter`), one statement for up to 500 rows, because one statement
a row costs about 4 ms and a walk claims tens of thousands.

**`complete_initial_scan` runs at the end of the first tick**, whatever it scanned,
including nothing (a config of single remote sources only). It stamps `last_full_scan_finished_at`
(`_mark_catalog_complete`, which also advances the orphan clock), clears
`full_scan_in_progress`, flips `_initial_scan_done` and fires the completion hook, once.
It is last on purpose: the scan-once directories, the monitored walk and the upstream
pass have each had their first pass, so the upstream mirror is part of the startup set. An
unreachable upstream delays the flip by one failed attempt, not indefinitely. If the first
tick raises, `full_scan_in_progress` is cleared and the next tick retries the first scan.
`_initial_scan_done`, not the runtime-phase flag, is the precache gate: the scan runs after
`start()`, so the runtime flag is already true and would prompt-enqueue every startup
source.

A drop that arrives before the first tick finishes is refused, not queued. Once it has
finished, a drop waits on `_catalog_lock` if a rescan is running (heart-beating to its
caller) and runs as a live addition. Where it lands (known root or not, which label) is
decided after the lock is taken, so a removal or an earlier drop that went first is seen.

## 7. SERVING versus freshness

`SERVING` means the server is up and answering, not that the catalog is complete.
`mark_ready()` flips `health` from `STARTING` to `SERVING` once the `SourceManager` is
wired and started, before any scan.

Freshness is a separate, continuous signal on `health`:
`full_scan_in_progress: bool` (raised by the launcher before the manager starts) and
`last_full_scan_finished_at: float | null` (epoch seconds, `null` until the first scan
completes). A third, `registration_pending: int`, counts the claimed sources still
waiting to be registered: the scan can be finished while the rows it wrote are still
filling in, and `0` means the catalog is whole. The same fields serve boot and steady state: a later full pass (a forced full
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

- **Marked drops are remembered in memory only.** After a restart an earlier drop's sources
  are gone with the catalog, so its folder is an ordinary outside drop again.
- **A retargeted configured symlink is not followed until restart.** Only its first
  target was stored as the root.
- **Per-root reconcile.** The removal diff covers all monitored roots at once, so
  steady-state staleness is bounded by the slowest root, not surfaced root by root.
- **The catalog is not persisted.** A persisted catalog would let startup serve at once and
  turn the first scan into revalidation, with `last_full_scan_finished_at` as its staleness
  signal.
