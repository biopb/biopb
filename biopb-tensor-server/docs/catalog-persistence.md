# Discovery, the catalog and its persistence

Scope: `biopb-tensor-server` (`core/discovery.py`, `sources/source_manager.py`,
`sources/reconciler.py`, `serving/metadata_db.py`, the adapters' payload API), plus the
client indexing-state hint in `biopb-mcp` and the webapp.

The server binds and serves before it has scanned anything. Discovery fills the catalog
from behind `SERVING`; a separate signal says when that is finished. With `catalog.restore`
on, the last run's catalog is read back first, so a restart lists every source at once and
registers none that is unchanged. This page states what discovery and the catalog
guarantee: which path string is used where, how a scan decides, what is persisted and how
it comes back.

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
   register_source (server) + catalog row (source_catalog)
```

- **One walker.** A drop, a one-shot directory and the periodic rescan of a monitored root
  all call `discover_sources`. It keeps nothing between calls: each call stats the whole
  tree under its root that it does not prune.
- **A walk fills a scratch state; the Reconciler owns the confirmed one.** The confirmed
  state is what is registered and served. A walk never mutates it; only a commit does
  (`_commit_add_claim`, `_refresh_claim`, `_commit_remove_source`).
- **One lock.** Ticks, drops and `remove_source` all hold `_catalog_lock`, so the confirmed
  catalog has a single writer.
- **A restart is a rescan with restored state** (§8): the claims and rows persisted last
  run are put back, then each root is walked as an ordinary rescan.

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
| `Reconciler` | `_source_signatures`, `_missed_scans`, `_pending`, `_pending_failed`, `_recall`, `_restored` | `source_id` |
| `Roots` (shared by `SourceManager` and `Reconciler`) | every known root: kind (monitored, scan-once, dropped, upstream), alias, cloud, `dnd://` label, `root_id`, `root_url` | Resolved root `Path`s; a drop's label |
| `SourceManager` | `_unavailable_roots` | Resolved root `Path`s |
| Adapter | `_source_url` (the raw claim path, or the library's own filename for nifti / bioio / dicom), `catalog_url` | Opens files with the raw path |
| Catalog | `catalog_roots`, `source_catalog`, the `sources` view (§7) | `root_id`; `source_id` |

## 3. Path invariants

A claim's `primary_path` and every `member_paths` entry is **the string the walk
produced**: the root as given to `discover_sources`, plus the names walked below it. No
adapter `claim()` normalizes the path it returns.

1. **A root is canonical, and becomes canonical once.** Config roots (`local_path`), drops
   and configured paths go through `resolve_local_path` (folds `file://`, symlinks, `..`,
   case, trailing separators). The monitored scan walks each stored root as given and never
   resolves it again, so its claims are spelled under the stored root even if that path
   later becomes a link. `Roots` holds the same strings the walk uses.
2. **A claim path is never resolved to key or look it up.** `DiscoveryState.claims` /
   `_path_to_source` / `_source_to_paths` / `consumed_paths` and the signature maps use the
   claim strings exactly as the walk spelled them.
3. **"Under a directory" is lexical.** A claim is under `D` when its string is
   `is_relative_to(D)` for a canonical `D`: a monitored root, a drop root, a declined
   directory, a cloud root. A symlinked file leaf (`root/link.tif`) is under `root`,
   wherever it points; it would not be if resolved.
4. **"Same file" is answered by identity, not spelling.** `get_file_identity` is device and
   inode (resolved-path hash where the inode is synthetic), computed on the resolved
   target. The walk's `visited_identities` drops hardlinks and a file reached by two links.
   `generate_source_id` hashes the resolved path, so a link and its target are one
   `source_id` however the claim is spelled. Moving the data gives it a new id: the old one
   goes by the removal rule and the new one is added behind it. A remote URL is hashed as
   written, minus trailing `/`.
5. **Cycle guards use real paths and decide entry only.** `_real_dir` (`realpath`), the
   identity set, `_leads_back_up` and `MAX_WALK_DEPTH` stop the walk from entering a
   directory. They never change how a path is spelled. A directory refused for any reason
   is added to `declined_dirs` in the walk's spelling, so lexical comparison against it is
   consistent (rule 3).
6. **I/O follows the claim path.** `open` and `stat` of a claim path follow links, so a
   link claim reads and signs its target. Quietness and change signatures stat the spelled
   path (`entry_is_quiet`, `build_entry_signature`).
7. **The catalog `source_url` is display, never an input.** For a row with a claim it is
   computed from its root (§7); for a row with none it is `dnd://<label>` for a drop from
   outside every known root, `cache://`, `scratch://`, or `grpc://<alias>/…` for a mirror.
   It is not fed to the filesystem or to `generate_source_id`, and rows are found and
   removed by `source_id`. The one lookup by url is `remove_source`, which matches a
   `dnd://<label>/` prefix.
8. **A remote claim is a URL.** `is_remote_url` claims are never resolved and are skipped by
   every path comparison.

Links are the one way a claim reaches outside its root. A symlinked file or dataset
(`.tif`, `.zarr`) inside a root is claimed under the link's spelling and read through it. A
symlinked plain directory is not entered. No check is made on where a followed link points.

**One root per source.** A source belongs to the root whose walk committed it (persisted as
`root_id`). Roots are not to nest and no file is to be reachable from two roots: policy, not
something the code can fully check (a link defeats a lexical test). Two roots that reach the
same file produce one `source_id` spelled two ways; the stream hook notices the second
spelling and logs one warning per source naming both paths. A drop never re-spells a claim
another path already holds (`Reconciler.is_held_under_another_path`), which keeps the
invariant and the source's table membership.

## 4. The walk

`walk_with_identity_tracking` yields `root/name` for every entry it does not decline;
`discover_sources` offers each to the adapters, first claim wins
(`AdapterRegistry.get_claims_for_path`). An adapter consumes the paths it takes with
`try_claim_path`, which fills `member_paths`; the walk does not descend below a claimed
directory.

An entry is declined, not yielded, when:

- it is hidden, a system or cloud-provider directory, or (outside a cloud root) an offline
  placeholder file (`should_skip_walk_entry`);
- the caller's `path_filter` rejects it. The monitored rescan passes `_should_claim`, the
  **stability gate**: an entry is claimable only once the later of its own mtime and ctime
  is older than `stability_window` (30 s by default; `0` disables the gate). It applies to
  the entry's own stat, not to what is inside a directory;
- it is a directory that leads back up, or lies more than 64 levels below the root.

A declined **directory** goes into `WalkReport.declined_dirs`; a declined file does not
(that would make the set O(files)). A declined directory means "not looked at", so what is
registered below it is not taken for gone. Under a cloud root the stability gate is
bypassed and placeholders are admitted as unresolved claims.

One `DiscoveryState` is shared by all roots of a pass, so overlapping roots cannot claim a
subtree twice.

## 5. One scan per root

A monitored root and a scan-once root are scanned the same way (`SourceManager._scan_root`),
one root at a time, each against its own claims:

1. Take the root's snapshot, `Reconciler.claims_under(root)`: the confirmed claims lying at
   or under the root, by the path each is spelled under (lexical, so a link belongs to the
   root it was found in). It is taken before the walk.
2. Walk the root into a scratch state. A claim is committed the moment the walk finds it
   (`on_source_added` → `_stream_claim_add`), so the catalog grows within the walk. What the
   walk commits is new, so it is not in the snapshot and is not stat'ed again.
3. `_reconcile_root(snapshot, discovered, recurring)` compares the two. A claim is never
   compared with another root's walk, so scanning one root cannot remove the claims of
   another.
4. A walk that ran to the end is confirmed (`_confirm_root`, §8).

A monitored root is walked again and a scan-once root is not (`recurring`), which is the
whole difference:

| | Monitored (`recurring`) | Scan-once, drop |
|---|---|---|
| Stability gate on the walk and on a refresh or removal | Yes: an unstable entry is picked up by a later tick | No: nothing would pick it up |
| A claim the walk did not find | Removed after `_MISSES_BEFORE_REMOVAL = 2` walks running, once quiet | Removed at once, unless an adapter claims it again on a second look |
| Claim probes memoized across walks | Yes | No |

Removal is one rule, `Reconciler._remove_absent(snapshot, discovered_ids, recurring=)`. A
drop removes by the scan-once rule, ungated: a file written a moment ago is read. Only a
monitored root can wait for quiet.

A claim that registers as it is found (a non-deferred add, or a refresh) is stat'ed once,
before the parse; that signature is the one persisted beside the row and the one state
keeps, so a file that changes during the parse reads as changed on the next scan.

A cloud root is scanned only on a full pass; an incremental tick skips it and so leaves its
sources as they are. A root that cannot be listed is not scanned, so nothing under it is
removed (a monitored one is recorded as declined and keeps its sources until it is back).

For each root, the comparison decides:

- **Added** (found, not in the snapshot): committed by the walk, as it is found, and not
  again here. A claim whose registration fails is committed as a failed row, so it is known
  and the next walk does not try it again. One that did not commit when streamed (the
  catalog write failed) is found new by the next walk.
- **Changed** (found and confirmed, signatures differ): rebuilt in place with
  `replace=True` when quiet, so the source is never absent from the catalog and a failed
  rebuild keeps the old adapter. A rebuild that fails is not counted: only the walk removes
  a source.
- **Unchanged** and not registered: a source that waits for the pool (`pending`) or a client
  (`needs_recall`) is re-decided against its residency, so a file that became resident is
  queued and one that stopped being resident waits for a client. A failed row is left alone.
- **Removed** (confirmed, not found): per the table above. Quiet is `entry_is_quiet` over
  the claim's own member paths; a member that cannot be stat'ed counts as quiet, since gone
  is what removal acts on.
- **Preserved** (declined): `_preserve_skipped_claims` carries forward confirmed claims under
  a declined directory, so a gate, a skip or a refused directory never removes what is
  registered below it.

**Cloud roots** are enumerated only on a full pass. On an incremental tick their sources are
left out of the candidate set (no diff, no removal, no miss counted), and their signatures
are identity-only so hydration and eviction do not flap a source.

**Full versus incremental.** A full pass runs when `full_rescan_interval` (default 3600 s)
has elapsed since the last one, and on the first tick. Only a full pass walks cloud roots
and advances `last_full_scan_finished_at`. `full_rescan_interval = 0` means no full pass
ever runs: cloud roots are never walked.

### What each event writes to the catalog

Every source has one row in `source_catalog`, keyed by `source_id`. A source with a claim
under a monitored or scan-once root (cloud roots included) has it, with its claim columns,
from the moment the walk finds it; a source with none (a mirror, a drop, an upload) has a row
without them. Registration fills the row in, and a write that carries no claim (an upload
re-listing its source) updates the public columns and leaves the claim as it was.

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
| A restored source rebuilt from its own row (§8) | `UPDATE` of `tensors` if the uploaded fields and label sets now attached differ from the row's; its root's walk confirms it |

"Resident" is `_claim_is_unresolved` being false: the adapter did not flag the claim, and
under a cloud root no member is a dehydrated placeholder (a metadata `stat`, nothing is
opened). A registration `UPDATE`s the pending row because that costs about two thirds of the
wall time and a quarter of the CPU of `INSERT OR REPLACE` on the indexed table. The batched
`DO NOTHING` also guards the race where a registration writes the real row before the
buffered pending one lands.

A failed row stays failed: nothing retries it on a timer, and the walk leaves it alone while
its signature is unchanged. A new signature for the same URL, a re-drop of its path or a
`resolve` registers it again. There is no failure counter and no removal for failing.

The signature has known gaps: a cloud file's is `(dev, ino)` only, and a directory's is its
own stat, which does not move when a member is rewritten in place. A drop refreshes every
already-registered claim under the dropped path unconditionally, with no signature compare,
so a re-drop is the repair for both.

### Drops

Where a drop lands decides what it may do. A **known root** is a monitored directory, a
`monitor = false` directory, or the root of an earlier marked drop; the test is lexical on
the resolved drop path (rule 3).

| The drop lands | Result | Display tree of new sources | `dnd://` mark |
|---|---|---|---|
| Inside a known root, the root itself included | Registers and refreshes as a rescan of that root would | The root's own: its alias if it has one; an earlier drop's label; otherwise the file url | An earlier drop's mark carries over to what the re-drop adds; a monitored or `monitor = false` root gets none |
| Outside every known root | Allowed if it overlaps nothing, otherwise refused | Its own root, labeled by the folder name (`exp`, then `exp (2)`) | Stamped |

Overlap means the walk found a source that is already registered, or an existing source lies
under the dropped path. It is refused before anything is removed or committed, and a cloud
consent that drop gave is taken back. Cloud mode cannot be switched on inside a known root:
it is set where a folder is first added, or in the config. The scheme is the removal key
(`remove_source("dnd://<label>")`), so a re-drop inside a marked drop stamps what it adds
with the same label and removing the drop takes it too. Removing a drop frees its label and
its cloud consent.

Drops are refused until the first scan has finished (a `ValueError`, which the server maps
to a clean error): both checks above need the whole catalog.

A drop is not a scan of a root: `_register_root` walks the dropped path into a scratch state
with no stability gate, then `_remove_unclaimed_under` removes what is gone **under that
path only**: the claims under it that are not under a monitored root (the rescan does that)
and not under a declined directory. A single-file drop skips this. Then each claim is
refreshed if known, whatever its signature, else added. A drop whose path lies inside an
already-owned directory source is rejected (`_find_containing_source`). A drop's sources
have no claim and are **session-only**: they are not persisted and not restored, and the marks
are remembered in memory only.

## 6. First scan versus re-scan

The first tick **is** the first scan. `start()` restores the catalog (§8) and schedules the
tick immediately; `_handle_rescan` runs the same steps every tick:

```
_scan_pending_roots        scan-once directories (first tick only)
_rescan_monitored_dirs     walk monitored roots, reconcile
_reconcile_due_upstreams   mirror upstreams that are due
complete_initial_scan      first tick only
```

| | First tick | Later ticks |
|---|---|---|
| Catalog on entry | Empty, or the restored rows | Populated |
| Pass kind | Full (the interval has never elapsed) | Incremental, full once per `full_rescan_interval` |
| Registration | A new claim is committed as the walk finds it, registered afterwards by a worker pool while registration is deferred; with `registration_workers = 0` or once the first scan is over, registered as claimed | Same: new claims stream; the known ones are compared by `_reconcile_root` after each root's walk |
| Removal | Only against restored claims; otherwise every claim is a pure add | As in §5 |
| Cloud roots | Walked | Skipped unless the pass is full |
| Scan-once roots | Scanned (the same scan, once) | Not touched |
| Upstreams | All due (countdown 0) | Adaptive: every tick while changing or failing, doubling toward `full_rescan_interval` while stable |
| Precache | Everything routes to the slow backlog, each source when its registration completes | New sources prompt-enqueue |
| Freshness signal | Set by `complete_initial_scan` at the end | Advanced only by a completed full pass or upstream-only pass |

### Registration after the walk

Registering a source opens and parses its file, which is most of a large site's start. While
`registration_workers` is above zero the first scan therefore only *claims*:
`_commit_pending_claim` commits each deferrable claim to the catalog only: a row built from
the claim (`is_resolved` false, `unresolved_reason` `pending`, no tensors) and no adapter in
the registry. Its claim and signatures are in the confirmed state, so the diff, removal and
refresh treat it like any other source. A `RegistrationWorker` pool then calls
`Reconciler.ensure_registered`, newest file first, which runs the ordinary registration and
puts the adapter in the registry. The pool is held until the first scan is over
(`complete_initial_scan` resumes it): walking and registering contend, and a held pool lets
every claim be committed first. A rescan does not hold it again; it registers what it finds
inline.

- **`resolve` registers it at once; a read does not.** The server's read paths and the upload
  manager use `SourceRegistry.get_registered`, which on a registry miss asks the reconciler
  (`check_registered`): an unresolved error while the source waits ("open to resolve", the
  same as a cloud source), `SourceRegistrationError` for a failed one, and for a restored
  source (§8) its registration. Callers that only ask whether a source is registered keep
  `get`, which is `None` for a pending source. `resolve` calls the reconciler's
  `materialize`, single-flight per source, so a resolve racing the worker shares one
  registration; it registers the source (trying a failed one again: nothing else does) and
  returns the filled row.
- **Not deferred:** remote proxies (bulk-seeded), static sources, and everything claimed
  after the first scan. A cloud source is left unregistered for good: its row reads
  `needs_recall`, the pool never queues it, and only `resolve` downloads and registers it.
- **`unresolved_reason`** says why a row is not resolved: `needs_recall` (a cloud
  placeholder; resolving downloads it), `pending` (queued; resolving registers it, no
  download), `failed` (registration raised; `unresolved_error` holds the text, reads raise
  `SourceRegistrationError`).
- **While pending** a source can be refreshed (the rebuild is the registration) or removed,
  and a registration never registers a removed source back (a per-source lock orders the
  three). The upload attacher runs on the adapter once it is registered.
- **Precache.** One callback after every registration (`_notify_source_committed`, a
  hydration included) routes the source: one the first scan found goes to the backlog
  (`enqueue_backlog`, with its mtime), a later one is prompt-enqueued, a remote one is not
  enqueued as startup. The backlog tier waits for `registration_idle` (first scan over,
  nothing pending); the live tier is not held.
- **Cost.** `benchmarks/registration_cost_test.py` reports the per-source-type cost of a scan
  (`BIOPB_REG_COST_ROOT` points it at a site).

Streaming commits only claims that are new; a claim already known is left to the end-of-walk
reconcile, which compares its signature, and removals are only ever decided after the walk.
It is idempotent against a retry: `_stream_claim_add` skips a claim already in the confirmed
state, because `_commit_add_claim` unregisters on a duplicate add. The end-of-walk reconcile
never sees what was streamed, because the root's snapshot was taken before the walk. The
stability gate holds while streaming, so an unstable entry is picked up by a later tick.

While registration is deferred the pending rows of the claims a walk streams are written in
batches (`PendingRowWriter`), one statement for up to 500 rows, because one statement a row
costs about 4 ms and a walk claims tens of thousands.

**`complete_initial_scan` runs at the end of the first tick**, whatever it scanned, including
nothing (a config of single remote sources only). It stamps `last_full_scan_finished_at`,
clears `full_scan_in_progress`, flips `_initial_scan_done` and fires the completion hook,
once. It is last on purpose: the scan-once directories, the monitored walk and the upstream
pass have each had their first pass, so the upstream mirror is part of the startup set. An
unreachable upstream delays the flip by one failed attempt, not indefinitely. If the first
tick raises, `full_scan_in_progress` is cleared and the next tick retries the first scan.
`_initial_scan_done`, not the runtime-phase flag, is the precache gate: the scan runs after
`start()`, so the runtime flag is already true and would prompt-enqueue every startup source.

A drop that arrives before the first tick finishes is refused, not queued. Once it has
finished, a drop waits on `_catalog_lock` if a rescan is running (heart-beating to its
caller) and runs as a live addition; where it lands is decided after the lock is taken.

## 7. The catalog store

The catalog is a DuckDB database (`serving/metadata_db.py`). Persisted state is kept apart
from the rest, and one view publishes both:

- **`catalog_roots(root_id, root_url, epoch, last_scanned)`**: one row per configured
  monitored or scan-once root, merged from config when the manager is built (`sync_roots`
  keeps `epoch` and `last_scanned` and deletes the rows of a root no longer in config).
  `root_id` is a hash of the resolved root path; `root_url` is the root's alias, or
  `to_catalog_url` of its path. A cache of config, not state: config stays the one source of
  truth.
- **`source_catalog`**: one row per source. The public row columns; for a source with a
  claim under a configured root (cloud roots included), `root_id` and `rel` (the
  forward-slashed path under the root, `.` for the root itself) in place of `source_url`, the
  private claim (`primary_path`, `member_paths`, `extra_config`, `source_type`), the
  claim-time signature, the adapter `payload`, and the `epoch` of its last write and
  `last_seen`. A row is resolved, pending, `needs_recall` or failed. A source with no claim to
  re-derive (a mirror, a remote proxy, an upload, a drop) has `root_id` NULL, a literal
  `source_url` and no claim columns; those rows are deleted at every open, and a restore reads
  only the rows with a root.
- **`sources`**: a view over it exposing the published columns only. A claimed row's
  `source_url` is `root_url`, or `root_url || '/' || rel`, which is what `Roots.display_url`
  gives the claim, so an alias edit is one `catalog_roots` row and no source row can carry a
  stale url. A row of a root that is no longer configured is not listed.
- **`source_confirmation(source_id, confirmed)`**: a second view, hidden from queries like the
  tables, saying whether each source was verified this run (§8). It is not part of `sources`
  because that is a published schema.

The private columns are hidden because `sources` is queryable by everyone with read access
and a claim can carry credential profile names, aliases and paths. `source_id` is the primary
key, so a source cannot be listed twice whichever way it is written: a write with a claim
fills the claim columns and clears a literal url, one without leaves them alone.

A source's one root is the column `root_id`: a root's claim snapshot is `WHERE root_id = ?`,
and `rel` is computed where the row is written, so path normalization stays out of SQL.

The view is persistent, created at open (a `TEMP` view belongs to the connection that made
it, and the read path takes a fresh cursor per call). At 100k rows a `source_id` lookup
through the view takes about 1 ms and a `source_url` lookup about 10 ms, since the url is
computed from the root.

**The row.**

- **`metadata_json` is stored once**, in the row (about 90% of the file: ~2 GB at 100k
  sources). There is no second copy.
- **A failed registration's error** is text in `unresolved_error`, never `metadata_json`.
- **The signature drops `st_dev`**: `(st_ino, size, mtime_ns, ctime_ns)` per member, bound to
  the path, because `st_dev` is stable within a boot only (NFS, external disks and overlay
  mounts can renumber). In-memory signatures keep it, and a restored one is compared on the
  other fields (`_same_signature`).
- **The signature is the one taken when the claim was made**, never a fresh stat after the
  parse, so a file that changes during a long parse reads as changed on the next scan.

**Versioning.** `SOURCE_CATALOG_FORMAT` (in `catalog_meta`) drops `source_catalog` and
`catalog_roots` at open on a mismatch or a missing key: a rebuild. There is no per-row or
per-adapter version. An additive payload change needs none (a payload missing a key is a
miss and the claim is parsed); bump it only when a field changes meaning. Downgrading needs
one manual step: an older build runs `DROP TABLE IF EXISTS sources` at open, which fails on
the view, so `DROP VIEW sources` on the file lets it open. The file also holds user ROI
annotations, so deleting it is not the recovery.

## 8. Restore

`catalog.restore` (off by default; otherwise the persisted tables are cleared at open)
turns on a restore in `SourceManager.start()`, before the first tick.

**What is restored.** A source with a claim under a configured root of kind monitored,
scan-once or cloud. Mirrors and drops are not (a drop's roots live in memory, so its rows
could never be confirmed). `Reconciler.restore(rows)` builds each row's claim and signature
into the same maps a live claim uses, so nothing new waits on it:

| Row | In memory | Registered by |
|---|---|---|
| resolved, local | `_pending`, `_restored` | the pool, or a read (`check_registered`) |
| `pending` | `_pending` | the pool |
| `needs_recall` (cloud) | `_pending`, `_recall` | a client's resolve |
| `failed` | `_pending`, `_pending_failed` (the row's error) | a resolve, a drop, or a new signature |

The row stays as it is until a registration rewrites it, so the listing is complete before
any scan. Restored sources are queued to the pool as found, without a stat each. A restored
resolved source with no adapter is registered on a read, where a fresh pending source raises
"resolve it"; a failed hydration makes the row `failed` like any registration. A row that
cannot be read (a payload or claim that does not decode) is deleted and logged, and the walk
finds its file again as a new claim. A restore that raises leaves a fresh catalog: the rows
are dropped, never a half-restored state.

**Ownership is re-derived from the current config.** Roots come from config, not from the
tables:

- A claim under no current root, or a multi-file source with a member outside the roots, is
  deleted. A removed root's sources disappear at once. Only roots of kind monitored,
  scan-once or cloud restore.
- Each claim is attributed to the innermost current root (`Roots.containing`); a row is
  rewritten only when its `root_id` or `rel` differs. An alias change needs no pass: it is
  one `catalog_roots` row and the view follows.
- A root whose cloudness changed makes its rows read as changed: the persisted signature
  form depends on cloudness, so the two never compare equal and the claim is refreshed.
- A row older than 30 days by both `last_seen` and its root's `last_scanned`
  (`_RESTORE_MAX_AGE`) is dropped, so an unreachable root does not leave its sources listed
  forever.

**Hydration from the payload.** Registering a restored source rebuilds it from its row
with no check of the files: the walk is what compares them with the persisted signature and
refreshes a claim that changed, as it does for any registered source. A restored source is
therefore as stale as a registered one between rescans, from restart until the first walk
reaches its root (a file deleted meanwhile raises when read; one rewritten meanwhile is read
through the old layout). The chunk cache is kept honest by the version: a rebuilt file
source takes its `content_version` from the persisted signature (`mtime_ns:size` it was
parsed at), as a registered one holds, so chunks read through a stale layout are cached
under a version the walk's refresh replaces, and an unchanged file keeps its cache across
the restart. A directory source takes the stat at build. A source with no payload is
parsed. A source rebuilt from its own row is neither rewritten (a row can hold megabytes of
metadata) nor confirmed: only its root's walk confirms it. Its registration does attach the
uploaded fields and label sets on disk, which the row, written before they changed, cannot
know, so it compares the row's tensor ids (as a set) with what the adapter lists and writes
the `tensors` column alone when they differ. A failed write is logged and costs only the
listing.

**The contract of a payload.** Every file adapter has one: the OME-TIFF family (OME-TIFF,
TIFF, LSM), nd2, czi, the BioIO family (Zeiss, Leica, Nikon, Olympus, Bioformats, aics),
LIF, DeltaVision, MRC, EMD, QPTIFF, DICOM, NIfTI, zarr, OME-Zarr, NDTiff, TIFF sequence and
the legacy micromanager layout. Only the remote proxy has none. `catalog_payload()` (source
level only) returns JSON, plain Python types, of everything the adapter takes from the file
before it reads pixels that the row does not hold: tensor structure, the transfer grid, the
native pyramid, the physical scale, `has_rois` and the `@ome` mask descriptors. A payload is
O(members) for a directory source. `create_from_payload(source, payload, metadata, creds)`
rebuilds an adapter that serves the same listing, descriptors, grid, pyramid, scale,
metadata (the row's, passed in), content version and pixels as `create_from_config`,
**without opening the file**: it may stat it, and opens it lazily on the first read.
Embedded ROIs and masks are answered from the payload flags and parsed only on request. It
returns `None`, or raises, when it cannot, and the source is parsed.
`tests/payload_equivalence.py::assert_hydrates_equivalently` is the check every adapter
passes: it stores through `sync_source_added`, rebuilds from `read_hydration`, forbids the
file readers during the rebuild, and compares the snapshots including a full read.

**Cloud.** A cloud row is never restored as resolved: it is loaded unresolved
(`needs_recall`, empty `tensors`, no stat), and the walk settles it, `pending` if resident,
else `needs_recall`. The cloud signature is identity-only, blind to in-place sync updates,
so a resolved row could serve a stale shape; and a read of a restored source must not
trigger a download. The cost is one registration per resident cloud source per restart.

**Confirmation: restored is not confirmed.** `catalog_meta.run_epoch` increments at every
open. A walk that ran to the end sets its root's `epoch` and `last_scanned`
(`_confirm_root`: one write per root); a root that could not be listed, or whose walk
raised, is not confirmed. A row's `epoch` is the run that wrote it, and it is *confirmed*
when the greater of its own and its root's equals `run_epoch`. `mark_sources_seen` and
`put_rois` observe only confirmed sources, so a listed-but-unverified source does not stop
`roi_prune` from firing.

**Removal and growth.** The key is `source_id`, so a changed file overwrites its row; growth
is orphans.

1. A claim removed during uptime deletes its row.
2. After a root's successful walk, `sweep_root` deletes that root's rows no claim holds
   (`Reconciler.has_claim`, asked per row), so a row of a source another root reaches is
   not dropped. Only rows whose `root_id` is the one walked are swept.
3. A root not scanned successfully is not swept; its rows age out by the 30-day cap.

## 9. SERVING versus freshness

`SERVING` means the server is up and answering, not that the catalog is complete.
`mark_ready()` flips `health` from `STARTING` to `SERVING` once the `SourceManager` is wired
and started, before any scan.

Freshness is a separate, continuous signal on `health`: `full_scan_in_progress: bool` (raised
by the launcher before the manager starts) and `last_full_scan_finished_at: float | null`
(epoch seconds, `null` until the first scan completes). A third, `registration_pending: int`,
counts the claimed sources still waiting to be registered: the scan can be finished while the
rows it wrote are still filling in, and `0` means the catalog is whole. The same fields serve
boot and steady state: a later full pass advances the timestamp exactly as the first one did,
so a client has no startup case to special-case. Incremental ticks touch neither field.

The data plane tolerates a growing catalog: `register_source` / `unregister_source` and the
reads serialize on `_sources_lock`, and `list_flights` skips a source whose descriptor is not
built. `biopb-mcp`'s `_source_watch_loop` re-lists when `source_count` changes, and the
webapp polls the source list. The napari tensor-browser and the webapp `SourceTree` show
"Indexing... (N so far)" while `full_scan_in_progress` is set. A consumer that reads
`SERVING` as "catalog complete" will flash an empty catalog; gate "no data" on
`full_scan_in_progress` instead. With a restored catalog the listing is complete at once but
not yet verified: `confirmed` is not in the published schema, so "verifying" is the same
`full_scan_in_progress` signal.

## 10. Limits and open items

- **Drops are session-only.** Their sources and `dnd://` marks are gone after a restart, so
  the folder is an ordinary outside drop again.
- **A directory source's signature is the directory's own**, which does not move when a
  member is rewritten in place. A restart no longer heals that; a re-drop does.
- **A retargeted configured symlink is not followed until restart.** Only its first target
  was stored as the root.
- **Per-root reconcile.** The removal diff covers all monitored roots at once, so
  steady-state staleness is bounded by the slowest root.
- **`confirmed` is not exposed to clients.** The SPA and SDK read it only through the
  freshness signal; adding it to `sources` is a protocol change.
- **A corrupt catalog is not loud.** When the store will not open the server serves from
  memory, which silently turns restore off.
- **What a hydration still reads.** `read_hydration` decodes the row's payload and its
  `metadata_json`, which for a micromanager dataset is several MB; that decode is most of
  what is left of its hydration (about 2 s at the 95th percentile on `/labs`). Passing the
  metadata as a lazy mapping would skip it for the adapters that never ask; it changes what
  every `create_from_payload` receives.
- **Measured on `/labs/Yu/Ji` (NFS, one run, the page cache dropped between the two),
  registration from the claim against from the row**, median (95th percentile), ms: tiff 11
  (60) / 1.7 (2.1), OME-TIFF 72 (3300) / 1.8 (2.3), LSM 138 (193) / 4.3 (5.2), czi 76 (180) /
  2.2 (3.5), nifti 34 (88) / 2.0 (2.7), mrc 8.6 (33) / 1.8 (2.7), micromanager 236 (21600) /
  8 (2300), tiff sequence 470 (20500) / 2.2 (24). Restoring 1016 sources took 79 ms and the
  verifying walk 0.46 s. `.zvi` (Bioformats) fails to open in that environment either way.
