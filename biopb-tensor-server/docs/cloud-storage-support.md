# Cloud storage support in the tensor server

**Experimental.** The cloud/synced-folder source model -- the `SourceConfig.cloud` config surface and the unresolved/resolve mechanism -- may still change without notice.

Scope: `biopb-tensor-server` (server + Python SDK), the Java SDK, and the web
SPA. Related: the discovery placeholder guard in `discovery.py`, the metadata
DB (`metadata_db.py`), the pre-cache worker (`precache.py`),
[progressive-discovery.md](progressive-discovery.md), [remote-tensor-cache.md](remote-tensor-cache.md).

## Why

Users archive microscopy data to cloud storage (OneDrive, Dropbox, iCloud) to
free local disk, then still want to browse and analyze it through biopb. The
"synced folder with Files-On-Demand" model — data appears as local paths but
content is *dehydrated* until accessed — is the case this targets.

**On a dehydrated cloud file the only thing knowable for free is its URL.**
Reading any byte triggers a whole-file recall — slow, refills the disk the
user just freed, and blocks indefinitely offline. So shape, dtype, dim
labels, field count, and even whether a path is one source or many are all
unknowable without paying the full hydration cost. Discovery's default
posture is therefore "enumerate safely (`stat`/`iterdir` are recall-free),
then skip every dehydrated placeholder" — cloud data is normally never
catalogued. Opting a root in replaces the skip with "record the URL without
paying for content," and defers the one legitimate hydrate to the moment a
human actually asks for the pixels.

## is_resolved vs residency

Two descriptor bits a naive API would conflate into one ("is `shape`
present?"):

| Bit | Meaning | Gates | Lives in |
|---|---|---|---|
| `is_resolved` | descriptor known (shape/dtype/fields) — immutable once true | serving; the resolution boundary | a `sources` column |
| residency | content local & cheap to read right now — volatile | pre-cache warming; leave-no-trace | the `is_resident` action |

`is_resolved` is monotonic — false to true once, never back — so a stored
copy can only lag in the direction that costs nothing. Residency goes both
ways: a synced-folder provider re-dehydrates under storage pressure, in the
same running process, with no event to hang a refresh on. So there is no
residency column and no descriptor field; `do_action("is_resident", [...])`
calls straight through to `adapter.is_resident()` on every invocation — a
live lookup, not a row. Unresolved sources (empty `tensors`) stay filterable
on purpose via `WHERE NOT is_resolved`, instead of being silently dropped by
a `WHERE tensors[1].dtype = …` predicate.

## Shipped architecture

**Per-root opt-in (`SourceConfig.cloud`).** `{ "url": "…/OneDrive/…", "cloud":
true }` flips one configured subtree from "skip" to
"register-URL-unresolved" without weakening the default placeholder guard
elsewhere. Under a cloud root the walk passes `admit_nonresident` so
dehydrated entries reach `claim()` (hidden/system-dir prunes still apply),
and every `claim()` recognizes a source from name + `stat` + `exists` +
layout only — the read-triggering readers (OME-Zarr/Zarr `.zattrs`,
MicroManager `metadata.txt`, single DICOM header, OME-TIFF embedded-XML
sniff) are guarded by `ClaimContext.is_resident()` and, when non-resident,
emit a provisional `unresolved=True` claim that defers the content read to
resolve.

**The unresolved adapter (`adapters/unresolved.py`).**
`Reconciler._claim_is_unresolved` registers a cloud source behind an
`UnresolvedSourceAdapter` — a catalog row with empty `tensors` and
`is_resolved=false`. It is split into two surfaces:

- a **catalog surface** (`list_tensor_descriptors` / `get_metadata` /
  `is_resident`) that never resolves, keeping the metadata-DB sync and the
  precache worker cheap (precache loops the empty tensor list and skips
  before any serving call — an unresolved source is never
  background-warmed);
- a **serve surface** (`get_tensor_adapter`) that raises
  `SourceUnresolvedError` rather than hydrating, so `GetFlightInfo`/`DoGet`
  and the SDK probes stay recall-free and steer callers to the dedicated
  resolve trigger.

**The streaming resolve action (`do_action("resolve")`, `_handle_resolve`).**
A single dedicated action is the sole resolution trigger — not a side effect
of `GetFlightInfo`, which would smuggle a minutes-long hydrate into a
descriptor RPC and trip proxy idle-read timeouts. It streams: empty-body
heartbeat Results keep the connection warm under proxy timeouts, then one
terminal Result carries the source's now-concrete catalog row (every tensor,
one call). Resolution re-runs the real claim + `create_from_config` on the
now-resident path (the recorded `source_type` was a recall-free guess; the
authoritative one comes from the hydrated content), caches the real adapter,
fires `on_resolved` (the metadata-DB backfill — an upsert, so the NULL-shape
row is overwritten in place), and delegates thereafter. It runs once under a
lock on a daemon thread, so a client disconnect mid-resolve doesn't abort it
and a retry coalesces. Failure is classified: a transient recall/IO error
raises `SourceResolveRetriableError` (UNAVAILABLE, "retry"); a permanent one
raises bare `SourceUnresolvedError` (FlightInternal, don't retry forever).

**Every client is a consenting front door.** The Python SDK's read methods
short-circuit on a zero-tensor descriptor, so "first read resolves" never
fires implicitly. `TensorFlightClient.resolve(source_id, *, on_progress,
should_cancel)` is the explicit trigger — a thin blocking façade over the
resolve stream that returns the full multi-field descriptor; an
unresolved-source error is directive, pointing at it. The napari browser
gets the human twin: double-click / "Resolve…" → modal download warning →
consented, blocking resolve on a worker thread (consuming the heartbeats for
progress/cancel) → repopulate. The Java `TensorFlightClient` mirrors this —
`resolve()` and the same directive error on every read entry point. The web
SPA has the same shape through its HTTP sidecar (`POST
/api/sources/{id}/resolve` + `/resolve/status` + `/resolve/cancel`, backed
by a pollable job): an unresolved row in `SourceTree` renders as a plain div
with a "Resolve…" control (not a disabled button — interactive content can't
nest inside one) instead of being hidden, and a finished resolve reloads the
catalog listing. `warm()` (`do_action("warm")` / `POST
/api/sources/{id}/warm`) separately recalls a resolved multi-file source's
member files server-side, for hydrate-ahead.

**Format choice matters more on cloud than local.** Pyramidal, per-chunk-object
stores (OME-Zarr) are the supported cloud path: separable metadata
(`.zattrs`) claims cheaply, and a native chunk pyramid gives a cheap overview
and cheap sub-region reads. Monolithic formats (OME-TIFF, CZI, ND2, …) can be
listed and resolved-on-demand but get no cheap thumbnail and no cheap
sub-region read — the recommendation is to transcode to OME-Zarr at archive
time.

## Gotchas

- **Multi-file content-membership formats degrade on cloud, permanently.**
  Multi-file OME-TIFF (member set in the OME-XML) and DICOM series (grouped
  by per-slice `SeriesInstanceUID`) need a content read to know their
  members, and a directory can hold several such datasets, so the dir isn't
  the boundary. They are gated on `ClaimContext.cloud_root` (recorded per
  entry at scan, carried onto the `UnresolvedSourceAdapter` so it holds at
  both scan *and* resolve — residency can't gate resolve, where the file is
  resident) and under cloud return `None`, so each `.tif`/`.dcm` becomes its
  own single-file source. No later reconstruction.
- **Resolve of a multi-file source leaks bulk recall onto the read path.**
  For a monolith-per-file fallback the actual whole-object recall happens
  lazily on the subsequent `do_get` reads, not during resolve; `warm` exists
  to pull that recall server-side up front. For zarr/OME-Zarr, resolve reads
  only the metadata and per-chunk reads stay fine-grained.
- **No UI auto-warms after a resolve.** Both the napari widget
  (`_AUTO_WARM_AFTER_RESOLVE`) and the SPA store (`AUTO_WARM_AFTER_RESOLVE`)
  gate the post-resolve hydrate-ahead off by default; the chunk cache serves
  segments by mmap, so warming a source larger than RAM walks the page-cache
  LRU and evicts every *other* source's segments, and `warm` guarantees disk
  residency, not page-cache warmth. Both still expose a manual warm trigger
  (napari's "Hydrate all files…", the SPA's `WarmTray`). Flip the flags back
  once `warm` has a retention policy.
- **Cloud subtrees are walked only on a `force_full` rescan.**
  `TreeScanner._scan_tree_state` skips a cloud subtree on an incremental
  rescan (carrying cached claims forward) and re-walks it only on the
  periodic `force_full` pass. When walked, the stability window is bypassed —
  a placeholder's mtime is untrustworthy, so it could never age into
  eligibility, and archived dehydrated data is never mid-write anyway.
- **`cloud` controls gating only, not monitoring.** A `monitor=false` cloud
  root is still scanned once at startup via the static-expand path, which
  threads the same `admit_nonresident` + `cloud_root` behavior from
  `source.cloud`.
- **Shape-presence doesn't protect pre-cache.** An unresolved source
  auto-skips (empty shape), but once resolved-and-persisted it returns with a
  concrete shape, so a naive backlog would re-warm it on restart. Residency
  has to be its own bit, checked live: `Reconciler.should_warm()` asks the
  registered adapter's `is_resident()` at warm time rather than trusting the
  claim's shape.

## Not done / future

- **Metadata-DB durability across restart.** A general file-backed catalog
  (`catalog.store_path`) exists, but the resolve backfill
  (`_on_source_resolved`) is process-lifetime only — a resolved cloud source
  reverts to unresolved and re-hydrates on restart.
- **Read-tolerance mode (`EXACT` vs `BEST_EFFORT`).** No `TensorReadOptions`
  read mode exists yet. Viz wants "return a coarse proxy or `PENDING`, never
  block"; compute wants "hydrate and wait, or fail loudly, never silently
  substitute" — the caller would need to declare its regime explicitly.
- **Coarse-proxy eviction-stickiness.** Cloud-derived cache entries (the
  overview especially) aren't yet stickier than local ones under the global
  LRU cache, though a cloud miss is far more expensive to redo than a local
  one.
- **Leave-no-trace TTL dehydration.** No mechanism releases files biopb
  itself hydrated; a deliberate local pin and a biopb-triggered recall are
  currently indistinguishable once resident.
- **Object-store / fsspec remotes (S3, GCS, OneDrive-Graph) via this model.**
  Several adapters already read remote files directly via fsspec
  (`bioio.py`, `zarr.py`, `dicom.py`, `nifti.py`, `ndtiff.py`), but the
  unresolved → resolve flow doesn't extend to them: `discover_remote_source`
  still refuses to recurse into a remote directory (too slow on large
  buckets today), and there's no OneDrive-Graph support. See
  [remote-tensor-cache.md](remote-tensor-cache.md) for the separate caching-proxy (`tensor-server`
  source-type) path.
- **Never-warm data** (synced down already-cold, never resident on this
  server): a persistent descriptor cache fixes the *restart* cycle but not
  first contact with data this server never saw warm.
