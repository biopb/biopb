# Catalog persistence

**Stages 1 and 1.5 are implemented, and so are steps 4a-4c of the restore (behind
`catalog.restore`, off by default); the rest is proposed.** Tracked in biopb/biopb#1251. When a stage lands, move what became true into
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

- **`catalog_roots(root_id, root_url)`**: one row per configured monitored or scan-once
  root, rewritten from config when the manager is built (the restore stage adds `epoch`). `root_id` is a hash of the resolved root path; `root_url` is
  the root's alias, or `to_catalog_url` of its path when it has none. A cache of config,
  not state: config stays the one source of truth.
- **`source_catalog`**: every source that has a claim under a configured root, cloud roots
  included. Public row columns, with `root_id` and `rel` (the forward-slashed path under
  the root, `.` for the root itself) in place of `source_url`, plus the private claim
  (`primary_path`, `member_paths`, `extra_config`, `source_type`), the claim-time
  signature, the adapter payload, the epoch of its last write and `last_seen`. A row is
  resolved, pending, `needs_recall` or failed.
- **A volatile table**: today's `sources`, renamed, dropped and rebuilt at open. It holds
  what has no claim to re-derive: mirrors (catalog rows only, no claim), tensor-server and
  remote sources, uploads, drops.
- **`sources`**: a view over both, exposing the published columns only; the private
  columns are not in it.
- **`source_confirmation(source_id, confirmed_epoch, confirmed)`**: whether each source was
  verified this run (below). A separate view, hidden from queries like the tables, because
  `sources` is a published schema; adding `confirmed` to it is a protocol change, made
  with the clients that read it. Volatile rows are confirmed, being seen this run. For a persisted row the view computes
  `source_url` from its root: `root_url`, or `root_url || '/' || rel`. That equals what
  `Roots.display_url` gives the claim today, so an alias edit is one `catalog_roots` row
  and no source row can carry a stale url. A volatile row keeps a literal `source_url`:
  a drop's `dnd://` label, a mirror or a remote url has no persisted root.

A source belongs to one root, held as `root_id`: the one-root-per-source rule is a column,
a root's claim snapshot is `WHERE root_id = ?`, and a claim found again under another root
(a link, an overlap) keeps the first, as the scan does. `rel` is computed where the row is
written, so path normalization stays out of SQL. The adapter's in-memory `_catalog_url`
comes from the same `display_url` function at construction; it is the one copy the table
does not own.

The writer routes by whether the source has a claim. A source that moves between tables
(it gained or lost its claim) is deleted from the other table in the same locked write,
because a union view cannot enforce a unique `source_id`. That happens rarely, since the
row is written when the claim is made.

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

A source is restorable when it has a claim under a configured root of kind monitored,
scan-once or cloud, and is not a mirror. Nothing is opt-in per adapter: the claim is
enough to rebuild any of them, and a payload is an optimization that skips the parse
(OME-TIFF, nd2 and czi have one; others store NULL and are parsed on first read). A
pending or failed row is persisted too, with its claim and signature
([progressive-discovery.md](progressive-discovery.md) lists what writes each). What
restore does with a row:

- **Resolved, local:** hydrated from its payload, or from its claim when it has none.
- **Pending:** registered from its claim, like any pending claim.
- **Failed:** restored as failed, and not retried until the walker finds a new signature
  for the same URL, the user re-drops its path or a client resolves it. It has to be put
  back in the reconciler's `_pending` and `_pending_failed` maps with its persisted
  `unresolved_error`, or a read of it would find neither an adapter nor the reason.
- **Cloud:** loaded as unresolved `needs_recall`, never resolved: see Cloud.

- **Mirrors** are bulk-seeded from `catalog_seed` and need the upstream `indexed_at`.
- **Drops** (`dnd://`) are not restored: roots live in memory, so a drop is not re-found
  after a restart and its rows could never be confirmed.
- **Directory-backed sources** (zarr, ome-zarr, ndtiff, TIFF sequences) are restorable
  like the rest, with a known limitation. A directory source's signature is the
  directory's own, and that mtime does not move when a member is rewritten in place. A
  restart used to heal that; a restored directory source is refreshed by a change in the
  directory's own stat or by a re-drop, which refreshes every known claim under the path
  unconditionally. The rebuild flag below is the other escape hatch.

## Config changes between restarts

Roots come from config; persisting them would be a second source of truth. What is
persisted is each claim's paths, and ownership is re-derived against the current config.

- A claim not under any current root is deleted, not restored. A removed root's sources
  disappear at once.
- Only roots of kind monitored, scan-once or cloud restore. A root that is now dropped or
  upstream makes its rows unrestorable. A root whose cloudness changed makes its rows
  read as changed: the persisted signature form depends on cloudness (one element,
  `st_ino`, for cloud; three for a directory and four for a file otherwise), so the two
  never compare equal and the claim is refreshed.
- A multi-file source needs every member under current roots, else it re-registers.
- Each claim is attributed to the innermost current root (`Roots.containing`); a row is
  rewritten only when its `root_id` or `rel` differs. The display url needs no pass: an
  alias change is in `catalog_roots`, rewritten at open, and the view follows. A row whose
  `root_id` is in no current root is the removed-root case above.
- The restart never merges overlapping roots. It restores a claim set that was already
  consistent, attributes each claim to one root and walks each root as a rescan, so the
  runtime refusal of a partly overlapping root does not arise.

## The epoch: restored is not confirmed

`catalog_meta` holds a run counter incremented at every open (`run_epoch`); no row is
rewritten at start. Each root's `catalog_roots.epoch` and `last_scanned` are set when its
walk completes (`SourceManager._confirm_root`); a root that cannot be listed, or whose walk
raised, is not confirmed. `sync_roots` keeps them across restarts and deletes the rows of
a root no longer in config. A row's `epoch` is the run that wrote it, and its
`confirmed_epoch` is the greater of that and its root's; `confirmed` is that equal to the
run counter. A restored row whose root has not finished reads unconfirmed.

Two things use it today: `mark_sources_seen` and `put_rois` observe only confirmed
sources, so a listed-but-unverified source does not stop `roi_prune` from firing; and a
restore drops a row whose `last_seen` and root's `last_scanned` are both more than 30
days old (`_RESTORE_MAX_AGE`), so a root that is never reachable does not leave its
sources listed for ever. One write per root, not one per source; a batched per-source confirm
would write every row once per start, so add it only if per-root progress is not enough.

## Removal and growth

The key is `source_id`, so a changed file overwrites its row. Growth is orphans.

1. A claim removed during uptime deletes its row (the existing removal path).
2. After a root's successful walk, delete that root's rows the walk did not see. Take the
   claim set under the reconciler lock in one pass; protect anything written this run.
   The sweep follows the same ownership rule as the walk: delete only rows whose
   `root_id` is the one walked.
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

Cloud-root claims are persisted like any other (the table predicate is "has a claim"), but
a cloud row is **never restored as resolved**. Restore loads it unresolved with reason
`needs_recall` and empty `tensors`, with no stat; the walk then settles it: a known
claim with the same signature and an unresolved row becomes `pending` if resident, else
stays `needs_recall`, and registers like any pending claim.

Why a resolved cloud row is not trusted:

- The cloud signature is identity-only `(dev, ino)` so it survives hydration. It is also
  blind to in-place sync updates (a sync client rewriting a hydrated file usually keeps
  the inode), and some cloud filesystems report a zero inode: a restored row cannot be
  validated, so a resolved one could serve a stale shape and tensors that nothing
  refreshes.
- A source restored as resolved would let a read trigger a download if its members were
  evicted meanwhile, the unintended hydration the resolve guard exists to prevent.
- There is nothing to save: an unresolved source is built from the claim plus a stat, so
  the payload is NULL.

The cost is one registration per resident cloud source per restart, which is today's
behaviour. The gain is that cloud sources list at once, as unresolved rows. The walk's
rule for an unresolved row with an unchanged signature also covers the resident flip
that `_refresh_recall_claim` does not (it handles only resident to dehydrated). The
signature gap itself is healed by a re-drop, as for directories.

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
- **`_missed_scans`** is in memory; it is
  rebuilt from the restored claims, and empty is fine.
- **A hydration that fails** (the file was removed or no longer parses) must not leave a
  resolved row behind: it takes the same path as a failed registration (the row becomes
  `failed` with the error), so a client does not read a listed source that cannot open.
- **Single registration point:** the lookup belongs in the shared registration path, not
  only the background worker; a re-drop rebuilds known claims unconditionally.

## Stages

Stages 1, 1.5 and the roots table (item 3 below) as implemented: `source_catalog` and `sources_volatile` with the
persistent view, `SOURCE_CATALOG_FORMAT`, routing by whether the source has a claim
record (a pending, failed or resolved row alike; the payload is NULL when the adapter has
none, and for a cloud claim), claim-time signature without `st_dev`, deletion from both on
removal, one scan for the first walk and every later one (new claims stream and their
pending rows are batched), the failure tracker removed. Payloads exist for OME-TIFF, nd2
and czi, and no others:

- **OME-TIFF**: the scene descriptors (with their transfer grid), `has_rois`, and the
  `@ome` mask label tensors' field, parent and extent, derived from the metadata without
  decoding a bitmap. Built before `release_registration_cache`, which drops the bitmaps.
- **nd2, czi**: the probed layout (`to_payload` / `from_payload` on the layout, so the
  round trip is tested now), without the metadata the row already holds. An nd2 payload
  carries one entry per frame (`frame_indices`), so a long timelapse is the large case;
  measure it before stage 2 and pack it if it matters.

Each payload is tested by rebuilding an adapter from the JSON and requiring the same
tensors, grid and scale as the parsed one. Rows are cleared at open unless
`catalog.restore` is on (step 4).

1. `source_catalog` with `CACHE_FORMAT`, written through at each registration with the
   claim, claim-time signature and payload, deleted on live removal; the volatile table
   and the view. Never read. Golden-payload tests per adapter; row verification
   against a fresh parse (embedded-label descriptors, `has_rois`, attached tensors); a
   persisted-signature test that perturbs `st_dev` and expects no change; a test that
   changes a file between claim and write.
2. Unify the scan: the walk writes every claim to `source_catalog`, registration updates
   the row, the failure tracker goes. Done.
3. Roots in the table: `catalog_roots`, `root_id` and `rel` in place of `source_url`, the
   view computing it, `SOURCE_CATALOG_FORMAT` bumped. Done; the root snapshot as a keyed
   query waits for the restore, which reads the rows.
   Tests: an alias edited between two opens shows in the view with no source row
   written; each id appears once; `rel` and `root_url` equal `Roots.display_url` for
   plain, aliased and scan-once roots, and for a file that is the root itself.
4. Restore and hydrate every persisted row, behind `catalog.restore` (off by default),
   in steps:
   - **4a. Restore from the claim.** At `start()`, before the first tick, each row
     becomes a claim and a pending source: the catalog is complete at once, the pool (or
     a client's read) registers it by parsing, and the first walk is an ordinary rescan
     against the restored claims. The config-change rules, failed and cloud rows, and the
     `check_registered` hook that hydrates a restored source on a read.
   - **4b. Hydrate from the payload:** nd2 and czi build their adapter from the row's
     payload (`create_from_payload`), skipping the parse, when the files' signature is
     still the one persisted; otherwise, and for every other type, the claim is parsed.
     OME-TIFF still parses: its serve path needs the reduced OME-XML for the physical
     scale and the cached descriptors seeded, which is an adapter change of its own.
   - **4c. Confirmation:** the run counter, `catalog_roots.epoch`, the
     `source_confirmation` view, the post-walk sweep and the age cap, the observation hooks
     ignoring unconfirmed rows.
   - **4d. The rest of Interactions:** lazy `@ome` ROIs and masks, precache on hydration,
     the loud corrupt-catalog log.

### 4a: how a restore works

A restored row is held in the reconciler's existing maps, so nothing new waits on it:

| Row | In memory | Hydrated by |
|---|---|---|
| resolved, local | `_pending`, and `_restored` | the pool, or a read (`check_registered`) |
| `pending` | `_pending` | the pool |
| `needs_recall` (cloud) | `_pending`, `_recall` | a client's resolve |
| `failed` | `_pending`, `_pending_failed` (the row's error) | a resolve, a drop, or a new signature |

- **The row stays as it is** until its registration rewrites it, so the listing is
  complete from the start. `_restored` marks a resolved row with no adapter: `check_registered`
  registers it on a read (where a fresh pending source raises "resolve it"), and a failed
  hydration makes the row `failed` like any registration.
- **Signatures:** the persisted form has no `st_dev`, so a restored signature is held with
  `None` there and compared on the other fields only (`_same_signature`); the first walk then
  keeps an unchanged claim and refreshes a changed one.
- **Ownership** is re-derived at restore (see Config changes): a claim under no persisted
  root, or with a member outside the roots, is deleted; a row whose `root_id` or `rel` differs
  is rewritten. The display url is recomputed from the roots, never read from the row.
- **A row that cannot be read** (a payload or claim that does not decode) is deleted and
  logged, and its file is found again by the walk as a new claim.
- **Off:** the tables are cleared at open, as before.

5. More payloads for adapters whose parse is slow, as measured; default on after a
   release cycle with the setting opt-in.
6. Clients: the SPA dims unconfirmed rows and reads "verifying"; the SDK exposes
   `confirmed`.

## To verify

- Done in stage 1: the `ALLOWED_TABLES` check passes the view and keeps both physical
  tables hidden (tested); the indexes are not used through the union (measured, above).
  If the lookups matter, a keyed read can go to `source_catalog` and `sources_volatile`
  directly.
- The cost of `INSERT OR REPLACE` and one-column updates at 100k rows with large
  `metadata_json`, on a real catalog (measured at 35k rows with a synthetic 3.3 KB
  `metadata_json`: an `UPDATE` about 3.1 ms a row against 4.5 ms for `INSERT OR REPLACE`).
- Roots are not to nest, and no file is to be reachable from two roots; that is policy, since a
  link defeats a lexical check. Each root's scan (and so its sweep) is scoped to the claims
  spelled under it. A file reached from two roots has one `source_id` and belongs to the
  root whose walk committed it, so a per-root sweep can drop what the other root still
  reaches: run it after every root has been walked, or only on ids no root found.
- Whether nd2 and czi carry embedded masks or ROIs.
- **OME-TIFF hydration from its payload** is the remaining slow case: measured on
  `~/data`, a restored nd2, tiff or ome-zarr hydrates in about 25 ms, an OME-TIFF in about
  500 ms because it still parses. It needs the reduced OME-XML (for the physical scale) and
  the cached descriptors seeded, and the embedded ROIs and masks imported lazily on
  first request, not at registration.
