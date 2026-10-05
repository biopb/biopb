# Catalog persistence

**Proposed, not implemented.** Tracked in biopb/biopb#1251. When a stage lands, move
what became true into [progressive-discovery.md](progressive-discovery.md) and delete
its section here.

Scope: `biopb-tensor-server` only (catalog store, reconciler, source registry). Companion
to [progressive-discovery.md](progressive-discovery.md), which describes how the catalog
fills today.

## Goal

A restart should show the catalog instantly and register nothing it already knows.
Today `sources` is dropped at open and rebuilt by the first scan, and every source is
parsed again. The catalogs here are static (they change little day to day), so an
instant catalog that may be stale beats one that fills progressively and is always true.

## Model: resume, not rebuild

The reconciler's claims and signatures live in memory and start empty, and so does
`sources`; the two agree by construction. That is a startup property only. Rescans,
drops and one-shot roots are incremental: diff signatures, refresh what changed, remove
what is gone at the end of a root's walk, add what is new.

The design persists the catalog and the claims, and treats a restart as a large rescan
with restored state:

1. Restorable rows and their claims are read back at start. The catalog is complete
   before any scan; nothing is registered.
2. Each root is then walked as an ordinary rescan. Unchanged signature: nothing. Changed:
   refresh. New: added as found. Gone: removed when the root's walk completes.
3. A read of a restored source finds no adapter in `SourceRegistry`. `get_registered`
   asks the reconciler, which builds the adapter from the row (`from_row`: no parse) and
   runs the upload attacher, then authorizes. This sits beside the pending check.

A restored row may be stale; the epoch below says so.

## Tables and the `sources` view

- **`source_catalog`**: persistent, restorable sources only. Public row columns plus the
  private claim (`member_paths`, `extra_config`, `source_type`), the claim-time
  signature, the adapter payload, the epoch of its last write and `last_seen`.
- **A volatile table**: today's `sources`, renamed, dropped and rebuilt at open. It holds
  what has no restore path: mirrors (catalog rows only, no claim), cloud-root sources,
  drops, and types without `from_row`.
- **`sources`**: a view over both, exposing the published columns plus `confirmed_epoch`
  and `confirmed` (below). The private columns are not in it. Volatile rows report the
  current epoch, since they were seen this run.

The writer routes by restorability. A source that moves between tables (its root turned
cloud, its adapter type changed) is deleted from the other table in the same locked
write, because a union view cannot enforce a unique `source_id`.

**The view must not be a persistent object named `sources`.** An older binary runs
`DROP TABLE IF EXISTS sources`, which fails on a view and would stop it opening the file.
Make it a `TEMP` view, recreated at each open, so the file holds only the two tables.
The physical tables use names `sources` never had, so an older binary ignores them.

## What is restored

A source is restorable when it is local, comes from a configured scanned root, is not a
mirror, is not under a cloud root, and its adapter implements `from_row`. Rollout is per
adapter type: OME-TIFF, nd2 and czi first, each shipping independently. Everything else
behaves as today.

- **Mirrors** are bulk-seeded from `catalog_seed` and need the upstream `indexed_at`.
- **Drops** (`dnd://`) are not restored: roots live in memory, so a drop is not re-found
  after a restart and its rows could never be confirmed.
- **Cloud**: see below.
- **Directory-backed sources** (zarr, ome-zarr, ndtiff, TIFF sequences) are restorable
  once their adapter implements `from_row`, with a known limitation. A directory
  source's signature is the directory's own, and that mtime does not move when a member
  is rewritten in place. A restart used to heal that; a restored directory source is
  refreshed only by an explicit re-drop (which rebuilds every known claim
  unconditionally) or by a change in the directory's own stat. The rebuild flag below is
  the escape hatch.

## Config changes between restarts

Roots come from config; persisting them would be a second source of truth. What is
persisted is each claim's paths, and ownership is re-derived against the current config.

- A claim not under any current root is deleted, not restored. A removed root's sources
  disappear at once.
- Only roots of kind monitored or scan-once restore. A root that is now cloud, dropped or
  upstream makes its rows unrestorable; the persisted signature form depends on cloudness,
  so it cannot be trusted either.
- A multi-file source needs every member under current roots, else it re-registers.
- Each claim is attributed to the innermost current root (`Roots.containing`), and the
  display `source_url` is recomputed from the current roots (`Roots.display_url`): an
  alias change must show. This is an in-memory pass; a row is rewritten only when its
  attribution or URL differs.
- The restart never merges overlapping roots. It restores a claim set that was already
  consistent, attributes each claim to one root and walks each root as a rescan, so the
  runtime refusal of a partly overlapping root does not arise.

## The epoch: restored is not confirmed

`catalog_meta` holds a run counter incremented once at open; no row is rewritten at
start. Each root has a `root_epoch(root_url, epoch)` entry set when its walk completes
successfully (entries for roots no longer in config are deleted at open). A row's
`confirmed_epoch` is the greater of its own write epoch and its root's epoch, and
`confirmed` is that equal to the run counter. A restored row whose root has not finished
reads unconfirmed. One write per root, not one per source; a batched per-source confirm
would write every row once per start, so add it only if per-root progress is not enough.

## Removal and growth

The key is `source_id`, so a changed file overwrites its row. Growth is orphans.

1. A claim removed during uptime deletes its row (the existing removal path).
2. After a root's successful walk, delete that root's rows the walk did not see. Take the
   claim set under the reconciler lock in one pass; protect anything written this run.
   The sweep must follow the same ownership rule as the walk: delete only rows whose
   innermost root is the one walked.
3. A root that was not scanned successfully (offline, unmounted, aborted) is not swept.
   Its rows age out by `last_seen` (say 30 days), or they stay as ghosts forever.

Registration threads and the sweep write through the same connection and lock as
`sync_source_added`; a row is written only by its own source's registration.

## The row and its payload

- **Payload** (JSON): what a read needs: base `chunk_shape`s, the pyramid, scene layout,
  the axis permutation from `normalize_adapter`, the claim, a `has_rois` flag and the
  descriptor (`array_id`, `dim_labels`, `shape`) of each embedded `@ome` mask label
  tensor. Base descriptors only, not the public `tensors` list: that includes attached
  fields and label sets, which the attacher re-derives at registration or hydration.
- **`metadata_json` is stored once**, in the row (about 90% of the file: ~2 GB at 100k
  sources). There is no second copy and no join.
- **A failed registration's error** is text in `unresolved_error`, never `metadata_json`.
- **The signature drops `st_dev`**: `(st_ino, size, mtime_ns, ctime_ns)` per member, bound
  to the path. `st_dev` is stable within a boot only (NFS, external disks and overlay
  mounts can renumber), so keeping it would make every row look changed after a restart.
  In-memory signatures keep it.
- **The signature is the one taken when the claim was made**, never a fresh stat after
  the parse. Otherwise a file that changes during a long parse is stamped with its new
  identity beside old metadata, and the stale row looks current forever. Test: change a
  file between claim and write, expect a refresh.
- The private columns are hidden because `sources` is queryable by everyone with read
  access and the claim can carry credential profile names, aliases and paths.

## Versioning and rollback

- **One table-level `CACHE_FORMAT` integer in `catalog_meta`.** At open the check drops
  `source_catalog` on a mismatch or a missing key, before the view is created; the table
  and the key are written in one transaction. The result is today's behaviour: a rebuild.
  No per-row or per-adapter version. Additive payload changes need none (`from_row`
  treats a missing required key as a miss); bump only when a field changes meaning.
- A committed golden payload per adapter fails the test when the written shape or
  semantics change without a bump (it guards the constant; comparing with a fresh parse
  would not, since both sides come from the current code).
- **Rollback is free** under the `TEMP` view rule above.
- **A rebuild flag** (the same drop) covers a bad parser upgrade and the directory limitation.
- **A corrupt catalog must be loud.** The `_open_catalog` fallback silently serves from
  memory when the store will not open, which would silently turn restore off. Log once at
  warning level.

## Embedded ROIs and masks (OME-TIFF)

A hydrated adapter has no metadata dict, so it never runs the two registration-time
readers of it, and two things `sync_source_added` produces today would vanish: imported
`@ome` ROIs and the `<image>/labels/@ome` label tensors (rasterized from `<Mask>` bitmaps
by `OmeTiffAdapter.get_embedded_labels`, and listed in the catalog's `tensors`). Both are
read lazily from the file.

- **Masks.** The payload carries each label tensor's descriptor (derivable from the scene
  descriptors; the dtype is fixed `<u4`), so the catalog lists it without a parse.
  `RasterizedMaskAdapter` takes a loader instead of eager bitmaps and parses on first
  read, which also stops each adapter holding its decoded bitmaps. The label value is the
  1-based index of the mask's ROI in the image's `roi_refs` order, deterministic per file.
- **ROIs.** The payload carries only `has_rois`. The `roi` read path for a flagged source
  imports on first request (parse, then today's delete-then-insert), idempotent and
  serialized per source. `rois` is not in `ALLOWED_TABLES`, so no SQL surface sees a
  partly filled table.
- **The open-time `@ome` wipe stays.** Imported rows keep the lifecycle of their source
  row, so each start re-derives them lazily. Rollback stays free, and no ROI-count guard,
  orphan GC, non-destructive CLI open, or coupling to `CACHE_FORMAT` is needed.
- `release_registration_cache` and `_mask_payloads_transferred` assume a registration
  parsed the XML; they must treat the never-parsed state as released.
- Not covered: OME-Zarr's `get_embedded_labels` (reads a zarr group, cheaper). Unchecked:
  whether nd2 and czi carry embedded masks or ROIs.

## Cloud

Cloud-root sources are out of the first rollout, and stay in the volatile table.

- The cloud signature is identity-only `(dev, ino)` so it survives hydration. It is also
  blind to in-place sync updates (a sync client rewriting a hydrated file usually keeps
  the inode), and some cloud filesystems report a zero inode: a restored row cannot be
  validated.
- Cloud content is the least static part of a catalog, and a restart is what heals that
  staleness today.
- A source restored as resolved would let a read trigger a download if its members were
  evicted meanwhile, the unintended hydration the resolve guard exists to prevent.
- There is nothing to save: an unresolved source is built from the claim plus a stat.

Restart behaviour is unchanged: the walk lists placeholders; a source whose members are
all resident registers normally, any dehydrated member gives `needs_recall`.

**Open option, for instant cloud listings:** restore cloud claims as `needs_recall` rows
only (empty `tensors`, no metadata), whatever they were persisted as. Safe, but a restored
`needs_recall` claim whose files are resident would stay unresolved after each restart,
because the identity signature does not change and the reconcile does nothing; it needs
the refresh to also handle the resident-to-hydrated flip (today's `_refresh_recall_claim`
handles the other direction, unchecked here). The benefit is latency only, since the walk
still runs. Worth doing only if a cloud-root walk is slow enough to notice; unmeasured.

## Interactions to handle

- **`mark_sources_seen` / `_observe_source`** treat any row as seen and would stop
  `roi_prune` firing; they must ignore unconfirmed rows.
- **Precache** warms at registration and `_notify_source_committed` fires there; hydration
  must notify, so startup warming still happens.
- **Uploads:** the boot sweep removes stores but does not call `_forget_rois` (an existing
  gap), and the reaper walks `registry.snapshot()`, so hydrated sources must be in it.
- **`scanning` / "Indexing…"** and `full_scan_in_progress` mean "verifying", not
  "empty", once the catalog is restored; the SPA hint and any client that treats a listed
  id as present must read `confirmed`.
- **Failure tracker, `_missed_scans`, `_cloud_source_ids`** are in memory; they are
  rebuilt from the restored claims, and empty is fine.
- **Single registration point:** the lookup belongs in the shared registration path, not
  only the background worker; a re-drop rebuilds known claims unconditionally.

## Stages

1. `source_catalog` with `CACHE_FORMAT`, written through at each registration with the
   claim, claim-time signature and payload, deleted on live removal; the volatile table
   and the `TEMP` view. Never read. Golden-payload tests per adapter; row verification
   against a fresh parse (embedded-label descriptors, `has_rois`, attached tensors); a
   persisted-signature test that perturbs `st_dev` and expects no change; a test that
   changes a file between claim and write.
2. Restore and hydrate for OME-TIFF, nd2, czi behind a setting: the restore rules, the
   epoch and per-root confirmation, the post-walk sweep and `last_seen` cap, lazy `@ome`
   ROIs and masks, hydration notifies precache, the observation hooks ignore unconfirmed
   rows, the loud corrupt-catalog log, the rebuild flag.
3. More adapter types, directory-backed ones included, with the documented limitation;
   default on after a release cycle with the setting opt-in.
4. Clients: the SPA dims unconfirmed rows and reads "verifying"; the SDK exposes
   `confirmed`.

## To verify before stage 2

- A `TEMP` view must be visible through the pooled read cursors (`_get_cursor`); DuckDB
  temp objects may be connection-scoped. If not, use a persistent view under another name
  with `sources` created temp over it, or give up free rollback.
- The check on `ALLOWED_TABLES` must pass the view and keep both physical tables hidden.
- Whether the indexes (`idx_source_url`, the primary key) are used through the union.
- The cost of `INSERT OR REPLACE` and one-column updates at 100k rows with large
  `metadata_json`, on a real catalog.
- Whether a root's walk skips nested roots' subtrees; the sweep scope must match.
- Whether nd2 and czi carry embedded masks or ROIs.
