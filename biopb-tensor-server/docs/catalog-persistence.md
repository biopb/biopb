# Catalog persistence

**Stage 1 is implemented (writes only); the rest is proposed.** Tracked in
biopb/biopb#1251. When a stage lands, move what became true into
[progressive-discovery.md](progressive-discovery.md) and delete its section here.

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
   asks the reconciler, which builds the adapter and runs the upload attacher, then
   authorizes. With a payload the adapter is built from it (`from_payload`: no parse);
   without one it is built from the claim, the registration path a pending source takes,
   so the parse happens on this first read. This sits beside the pending check.

A restored row may be stale; the epoch below says so.

## Tables and the `sources` view

- **`source_catalog`**: persistent, restorable sources only. Public row columns plus the
  private claim (`member_paths`, `extra_config`, `source_type`), the claim-time
  signature, the adapter payload, the epoch of its last write and `last_seen`.
- **A volatile table**: today's `sources`, renamed, dropped and rebuilt at open. It holds
  what has no restore path: mirrors (catalog rows only, no claim), cloud-root sources,
  drops and uploads. Until stage 1.5 it also holds every row that is not yet resolved.
- **`sources`**: a view over both, exposing the published columns plus `confirmed_epoch`
  and `confirmed` (below). The private columns are not in it. Volatile rows report the
  current epoch, since they were seen this run.

The writer routes by restorability. A source that moves between tables (its root turned
cloud, its adapter type changed) is deleted from the other table in the same locked
write, because a union view cannot enforce a unique `source_id`.

**The view is persistent, created at open** (dropped first; a physical `sources` table
from an older build is dropped too). It could be a `TEMP` view, which would let an older
build open the file, but a temp view belongs to the connection that made it and the read
path takes a fresh cursor per call, so each would pay about 190 µs (against 4 µs for a
bare cursor) and every reader would have to go through `_get_cursor`. Not worth keeping
downgrades free: see Versioning and rollback.

Measured at 100k rows: a primary-key lookup through the union is a sequential scan of
about 0.3 ms and a `source_url` lookup about 1.1 ms; the indexes are not used through the
union.

## What is restored

A source is restorable when it is resolved, local, comes from a configured scanned root,
is not a mirror and is not under a cloud root. Nothing is opt-in per adapter: the claim
is enough to rebuild any of them, and a payload is an optimization that skips the parse
(OME-TIFF, nd2 and czi have one; others store NULL and are parsed on first read). From
stage 1.5 a pending or failed row is persisted too, with its claim and signature, and
restore re-registers it from the claim (see stage 1.5 for failed rows).

- **Mirrors** are bulk-seeded from `catalog_seed` and need the upstream `indexed_at`.
- **Drops** (`dnd://`) are not restored: roots live in memory, so a drop is not re-found
  after a restart and its rows could never be confirmed.
- **Cloud**: see below.
- **Directory-backed sources** (zarr, ome-zarr, ndtiff, TIFF sequences) are restorable
  like the rest, with a known limitation. A directory
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
  a `has_rois` flag and the
  descriptor (`array_id`, `dim_labels`, `shape`) of each embedded `@ome` mask label
  tensor. Base descriptors only, not the public `tensors` list: that includes attached
  fields and label sets, which the attacher re-derives at registration or hydration.
  The claim is its own columns; the axis permutation is recomputed by
  `normalize_adapter` from the native axes.
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
  No per-row or per-adapter version. Additive payload changes need none (`from_payload`
  treats a missing required key as a miss, and falls back to the claim); bump only when a field changes meaning.
- A committed golden payload per adapter fails the test when the written shape or
  semantics change without a bump (it guards the constant; comparing with a fresh parse
  would not, since both sides come from the current code).
- **Downgrade needs one manual step.** An older build runs `DROP TABLE IF EXISTS
  sources` at open, which fails on the view. `DROP VIEW sources` on the file lets it open
  (the new build recreates the view at its next open). The file also holds user ROI
  annotations, so deleting it is not the recovery. The two physical tables are ignored by
  an older build. Document the step in the release notes.
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
  row, so each start re-derives them lazily. Downgrade stays one step, and no ROI-count guard,
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
- **A hydration that fails** (the file was removed or no longer parses) must not leave a
  resolved row behind: it takes the same path as a failed registration (the row becomes
  `failed` with the error), so a client does not read a listed source that cannot open.
- **Single registration point:** the lookup belongs in the shared registration path, not
  only the background worker; a re-drop rebuilds known claims unconditionally.

## Stages

Stage 1 as implemented: `source_catalog` and `sources_volatile` with the persistent view,
`SOURCE_CATALOG_FORMAT`, routing by claim record (a resolved source with one is
persisted; its payload is NULL when the adapter has none), claim-time signature without
`st_dev`, deletion from both on removal. Payloads exist for OME-TIFF, nd2 and czi, and
no others:

- **OME-TIFF**: the scene descriptors (with their transfer grid), `has_rois`, and the
  `@ome` mask label tensors' field, parent and extent, derived from the metadata without
  decoding a bitmap. Built before `release_registration_cache`, which drops the bitmaps.
- **nd2, czi**: the probed layout (`to_payload` / `from_payload` on the layout, so the
  round trip is tested now), without the metadata the row already holds. An nd2 payload
  carries one entry per frame (`frame_indices`), so a long timelapse is the large case;
  measure it before stage 2 and pack it if it matters.

Each payload is tested by rebuilding an adapter from the JSON and requiring the same
tensors, grid and scale as the parsed one. Rows are cleared at open, so nothing is
restored yet, and the epoch and `last_seen` sweep are stage 2.

1. `source_catalog` with `CACHE_FORMAT`, written through at each registration with the
   claim, claim-time signature and payload, deleted on live removal; the volatile table
   and the view. Never read. Golden-payload tests per adapter; row verification
   against a fresh parse (embedded-label descriptors, `has_rois`, attached tensors); a
   persisted-signature test that perturbs `st_dev` and expects no change; a test that
   changes a file between claim and write.
2. Unify the scan (stage 1.5, below): the walk writes every claim to `source_catalog`.
3. Restore and hydrate every persisted row, behind a setting: the restore rules, the
   epoch and per-root confirmation, the post-walk sweep and `last_seen` cap, lazy `@ome`
   ROIs and masks, hydration notifies precache, the observation hooks ignore unconfirmed
   rows, the loud corrupt-catalog log, the rebuild flag.
4. More payloads for adapters whose parse is slow, as measured; default on after a
   release cycle with the setting opt-in.
5. Clients: the SPA dims unconfirmed rows and reads "verifying"; the SDK exposes
   `confirmed`.

## Stage 1.5: one scan, claims in `source_catalog`

Today the first scan is a separate mode: with no prior state every claim is new, so the
walk streams them out as it finds them, batches their rows and skips the end-of-walk diff
for them. Once the catalog is restored, the first walk has prior state and is an ordinary
rescan. The special mode goes; what made it fast becomes the "new claim" case of the one
scan.

**The walk writes claims to `source_catalog`.** A claim that has a catalog record (local,
under a monitored or scan-once root, not cloud, not a mirror or remote) is written to
`source_catalog` when the walk finds it, not to the volatile table. Only what has no
claim stays volatile: mirrors, cloud placeholders, tensor-server and remote sources,
drops. A row then never changes table; registration fills it in.

| event | write to `source_catalog` |
|---|---|
| new claim | batched `INSERT ... ON CONFLICT DO NOTHING`: `is_resolved` false, reason `pending`, claim columns and signature, `payload` NULL |
| claim changed | `UPDATE` back to pending with the new claim and signature (an overwrite, so not the `DO NOTHING` path; rare, per row) |
| claim gone | delete the row |
| unchanged | none |
| registered | `UPDATE` of `metadata_json`, `tensors`, `payload`, `indexed_at`, `is_resolved` true, reason NULL |
| registration failed | `UPDATE` to reason `failed` with `unresolved_error` |

The batched writer serves every walk, not only the first. Its `DO NOTHING` is the
guard against a registration that wrote the real row before the buffered pending row
landed; it says nothing about whether the claim is current. That decision (unchanged,
changed, new) is made from the signature before anything is buffered, so a restored row
whose file changed is overwritten rather than skipped.

**Scan cases.** Each discovered claim is one of: in state and unchanged (no write), in
state and changed (refresh), new (batched insert). A restored claim the walk did not see
is dropped by the post-walk sweep, not by a diff of everything.

**Measured** (35k pending rows, 500 per statement, one transaction, local disk, fresh
store): into `sources_volatile` 1.95 s; into `source_catalog` 3.35 s (0.096 ms a row), the
same with the `source_url` index dropped, so the index is not the cost; re-inserting the
same rows, all conflicting, 2.87 s. Registration onto an existing row: `UPDATE` about
3.0 ms a row, `INSERT OR REPLACE` about 4.5 ms with four times the CPU. Registering by
`UPDATE` is cheaper than today's write.

**Open:**

- A failed row now persists with its signature. Restore either retries it or leaves it
  failed until the file changes; leaving it locks the failure in, so retry is the likelier
  choice.
- Contention between the batched writer and registration, and the HPC filesystem, are not
  measured.
- Rows are cleared at open until stage 2, so stage 1.5 changes what a running server
  writes and nothing a restart reads.

## To verify before stage 2

- Done in stage 1: the `ALLOWED_TABLES` check passes the view and keeps both physical
  tables hidden (tested); the indexes are not used through the union (measured, above).
  If the lookups matter, a keyed read can go to `source_catalog` and `sources_volatile`
  directly.
- The cost of `INSERT OR REPLACE` and one-column updates at 100k rows with large
  `metadata_json`, on a real catalog (measured at 35k rows with a synthetic 3.3 KB
  `metadata_json`: see stage 1.5).
- Whether a root's walk skips nested roots' subtrees; the sweep scope must match.
- Whether nd2 and czi carry embedded masks or ROIs.
