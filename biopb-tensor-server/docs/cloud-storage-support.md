# Cloud storage support in the tensor server

> **Status: implemented — experimental.** Cloud/synced-folder data is registered
> as *unresolved* sources (`adapters/unresolved.py`) and lazily hydrated via a
> streaming `do_action("resolve")` (`serving/server.py`), gated by the
> per-source `SourceConfig.cloud` flag, with the `TensorFlightClient.resolve()`
> SDK method + napari integration. The `cloud` flag and its lazy-resolve
> behavior may still change without notice. **Later phases remain design** —
> metadata-DB durability across restart, and the read-tolerance mode; both are
> under *Not done / future* below.

Scope: `biopb-tensor-server` (server + Python SDK), with knock-on effects for the
Java SDK and a carve-out for the TS/web client. Related: the discovery
placeholder guard in `discovery.py`, the metadata DB (`metadata_db.py`), the
pre-cache worker (`precache.py`), `progressive-discovery.md`,
`remote-tensor-cache.md`.

## Why

Users archive microscopy data to cloud storage (OneDrive, Dropbox, iCloud) to
free local disk, then still want to browse and analyze it through biopb. The
"synced folder with Files-On-Demand" model — data appears as local paths but
content is *dehydrated* until accessed — is the case this targets.

The guiding insight every design choice follows from:

> **On a dehydrated cloud file the only thing knowable for free is its URL.**
> Reading *any* byte triggers a whole-file recall — slow, refills the disk the
> user just freed, and blocks indefinitely offline. So shape, dtype, dim labels,
> field count, and even whether a path is one source or many are all unknowable
> without paying the full hydration cost. Everything else follows from refusing
> to pay that cost until a human asks for the pixels.

Three server assumptions break on a synced folder: (1) `stat`/`iterdir` are
recall-free so *traversal is safe*, but opening a file for even four header bytes
recalls the **whole** object — hydration granularity is the whole file, decoupled
from bytes logically read; (2) mtime is unreliable (why live-watch is already off
for these trees); (3) reads on an offline device don't error quickly — they hang.
The default posture is therefore "enumerate safely, then **skip** every
dehydrated placeholder" (`_is_offline_placeholder` in `discovery.py`), so cloud
data is normally never catalogued. Supporting cloud means *replacing* the skip
with "record the URL without paying for content," opted in per-root.

Eager download at scan time is rejected because it re-consumes the disk the user
freed and can block forever offline; lazy resolve makes the **first real read**
the one legitimate hydrate — the moment a human asked for the pixels.

## The model: unresolved sources + lazy resolution

A cloud source is a **URL-only, *unresolved* catalog entry** — an explicit
`UNKNOWN` descriptor (empty `tensors`, NULL `dtype`/`shape_summary`), not a
missing row. Resolution (learning shape/dtype/fields) is **lazy and
user-triggered**, deferred to the first consented access.

```
register URL (type ≈ guess, shape UNKNOWN)        ← scan time, free
        │  user opens the source (foreground, consented hydrate)
        ▼
resolve: hydrate → read real metadata → fill descriptor → backfill catalog row
        ▼
normal source thereafter (shape known, cached)
```

Two descriptor bits the naive API conflates ("is `shape` present?"), genuinely
orthogonal:

| Bit | Meaning | Gates | Lives in |
|---|---|---|---|
| `is_resolved` | descriptor known (shape/dtype/fields) — immutable once true | serving; the resolution boundary | a `sources` column |
| residency | content local & cheap to read **right now** — **volatile** | pre-cache warming; leave-no-trace | the `is_resident` action |

The difference in the last column is the whole point, and it took a wrong turn
to find (biopb/biopb#1035). `is_resolved` is monotonic — false to true once,
never back — so a stored copy can only lag in the direction that costs nothing.
Residency goes both ways: OneDrive re-dehydrates under storage pressure, in the
same running process, with no event for a refresh to hang on. A column could
therefore only ever record where the bytes were when someone last looked, which
is not the question anyone asks. So there is no residency column and no
descriptor field; `do_action("is_resident", [...])` calls straight through to
`adapter.is_resident()` on every invocation — a live lookup like `chunk_locate`,
not a row. Unresolved sources (NULL dtype) stay filterable on purpose via
`WHERE NOT is_resolved`, instead of being silently dropped by a `WHERE dtype=…`
predicate.

## Shipped architecture

**Per-root opt-in (`SourceConfig.cloud`).** The same blanket guard that keeps the
server from stalling on a Windows profile also blocks a *deliberately configured*
cloud root — it's the feature pointed the wrong way. `{ "url": "…/OneDrive/…",
"cloud": true }` flips one configured subtree from "skip" to
"register-URL-unresolved" without weakening the default guard elsewhere. Under a
cloud root the walk passes `admit_nonresident` so dehydrated entries reach
`claim()` (hidden/system-dir prunes still apply), and every `claim()` recognizes
a source from name + `stat` + `exists` + layout only — the read-triggering
readers (OME-Zarr/Zarr `.zattrs`, MicroManager `metadata.txt`, single DICOM
header, OME-TIFF embedded-XML sniff) are guarded by `ClaimContext.is_resident()`
and, when non-resident, emit a provisional `unresolved=True` claim that defers the
content read to resolve.

**The unresolved adapter (`adapters/unresolved.py`).**
`SourceManager._claim_is_unresolved` registers a cloud source behind an
`UnresolvedSourceAdapter` — a catalog row with empty `tensors` and
`is_resolved=false`. It is deliberately split into two surfaces:

- a **catalog surface** (`list_tensor_descriptors` / `get_metadata` /
  `is_resident`) that **never resolves**, keeping the metadata-DB sync and the
  precache worker cheap (precache loops the empty tensor list and
  skips before any serving call — an unresolved source is thus never
  background-warmed);
- a **serve surface** (`get_tensor_adapter`) that raises `SourceUnresolvedError`,
  so `GetFlightInfo`/`DoGet` and the SDK probes stay recall-free and steer callers
  to the dedicated resolve trigger.

**The streaming resolve action (`do_action("resolve")`, `_handle_resolve`).** A
single dedicated action is the **sole** resolution trigger (chosen over the
earlier "resolve-on-serve via `GetFlightInfo`", which smuggled a minutes-long
hydrate into a descriptor RPC, tripped proxy idle-read timeouts, and truncated
multi-field sources at the list cap). It **streams**: empty-body heartbeat
Results keep the connection warm under proxy timeouts, then one terminal Result
carries the source's now-concrete catalog row (every tensor, one call). Resolution
re-runs the real claim + `create_from_config` on the now-resident path (the
recorded `source_type` was a recall-free guess; the authoritative one comes from
the hydrated content), caches the real adapter, fires `on_resolved` (the
metadata-DB backfill — `sync_source_added` is an upsert, so the NULL-shape row is
overwritten in place), and delegates thereafter. It runs once under a lock on a
daemon thread, so a client disconnect mid-resolve doesn't abort it and a retry
coalesces. Failure is classified: a transient recall/IO error →
`SourceResolveRetriableError` → UNAVAILABLE ("retry"); a permanent one → bare
`SourceUnresolvedError` → FlightInternal (don't retry forever).

**Client `resolve()` + napari (the consenting front door).** The Python SDK's
read methods short-circuit on a zero-tensor descriptor, so "first read resolves"
never fires through them. `TensorFlightClient.resolve(source_id, *, on_progress,
should_cancel)` is the explicit trigger — a thin blocking façade over the resolve
stream that returns the full multi-field descriptor; an unresolved-source error
is **directive**, pointing at it, and a `guide://client` stanza teaches the agent.
The napari browser gets the human twin: double-click / "Resolve…" → modal
**download warning** → consented, blocking resolve on a worker thread (which can
consume the heartbeats for progress/cancel) → repopulate. `warm()` (the
`do_action("warm")` hydrate-ahead) recalls a resolved multi-file source's member
files server-side after resolve.

**Polyglot ring.** Python server owns resolution; the Python client tolerates
empty `tensors` and propagates the directive error. Java SDK mirrors the *client*
guards only (tracked, not yet shipped — see future). TS/web **refuses cloud
data** server-side at the sidecar (`/api/sources` filters unresolved sources,
`/api/slice` 409s) → zero TS changes; trade-off is cloud sources are invisible,
not greyed, in the web UI.

## The format thesis (§9)

The same conclusion recurs and hardens at every layer — separable metadata
(zarr `.zattrs`) claims cheaply while monolithic headers don't; only a native
per-chunk-object pyramid yields a cheap cloud overview; "read one chunk" only
exists for per-chunk-object stores, a monolith hydrates whole:

> **Pyramidal / chunked-object stores (OME-Zarr) are the supported cloud path.**
> Monolithic formats (OME-TIFF, CZI, ND2, …) can be *listed* and
> *resolved-on-demand* (one accepted full hydrate) but get **no cheap thumbnail
> and no cheap sub-region read**. The strong recommendation is that cloud
> ingestion **transcode to OME-Zarr at archive time**, so the metadata and coarse
> pyramid exist as separable, pinnable, cheaply-warmable objects.

(`biopb-tensor-server/ARCHITECTURE.md`'s "Cloud / synced-folder sources" section
cites this §9 for why multi-file monoliths degrade rather than reconstruct.)

## Gotchas

- **Multi-file content-membership formats degrade on cloud, permanently.**
  Multi-file OME-TIFF (member set in the OME-XML) and DICOM **series** (grouped by
  per-slice `SeriesInstanceUID`) need a content read to know their members, and a
  directory can hold several such datasets, so the dir isn't the boundary.
  Deferring the grouping would force a catalog reconciliation at resolve
  (forbidden — a claim is immutable once made). They are gated on
  `ClaimContext.cloud_root` (recorded per entry at scan, carried onto the
  `UnresolvedSourceAdapter` so it holds at both scan *and* resolve — residency
  can't gate resolve, where the file is resident) and under cloud return `None`,
  so each `.tif`/`.dcm` becomes its own single-file source. No later
  reconstruction.
- **Resolve of a multi-file source leaks bulk recall onto the read path.** For a
  monolith-per-file fallback the actual whole-object recall happens lazily on the
  subsequent `do_get` reads, not during resolve. `warm` (hydrate-ahead) exists to
  pull that recall server-side up front. For zarr/ome-zarr, resolve reads only the
  metadata and per-chunk reads stay fine-grained (the §9 payoff).
- **No UI auto-warms after a resolve any more.** Both front ends used to start a
  hydrate-ahead as soon as a resolve landed (biopb/biopb#202); both now gate it
  off (`_AUTO_WARM_AFTER_RESOLVE` in the napari widget, `AUTO_WARM_AFTER_RESOLVE`
  in the SPA store). The chunk cache serves its segments by mmap, so warming a
  source larger than RAM walks the whole page-cache LRU and evicts the segments
  serving every *other* source — for bytes `warm` never uses, since its read only
  exists to make the sync client write to disk. It does not keep its own coarse
  levels either (`warm` guarantees *disk* residency, not page-cache warmth), and
  nothing portable would: Windows has no `posix_fadvise`, and Windows is where
  synced-folder sources actually live (biopb/biopb#1043). The napari context
  menu's "Hydrate all files…" still warms on request; the SPA has no other
  trigger, so it does not warm at all. Reverse once `warm` has a retention policy.
- **Cloud subtrees are walked only on a `force_full` rescan.**
  `TreeScanner._scan_tree_state` skips a cloud subtree on an incremental rescan
  (carrying cached claims forward) and re-walks it only on the periodic
  `force_full` pass. When walked, `_should_scan_resolved` **bypasses the stability
  window** — a placeholder's mtime is untrustworthy, so it could never age into
  eligibility, and archived dehydrated data is never mid-write anyway.
- **`cloud` controls gating only, not monitoring.** `cli.py` routes on `monitor`
  alone; a `monitor=false` cloud root is scanned once at startup via the
  static-expand path, which threads `admit_nonresident` + `cloud_root` from
  `source.cloud` so the one-shot scan applies the same placeholder-admit and
  multi-file ban.
- **Shape-presence no longer protects pre-cache.** An unresolved source
  auto-skips (empty shape). But once resolved-and-persisted it returns with a
  concrete shape, so a naive backlog would re-warm it on restart — which is why
  residency has to be its own bit, checked live. `Reconciler.should_warm()` now
  does exactly that: it asks the registered adapter's `is_resident()` at warm
  time. It used to ask the claim's `member_paths` instead, a check that could
  not see inside a directory-claimed store (biopb/biopb#1035).

## Not done / future

Phases below remain design; none are built yet.

- **Metadata-DB durability across restart.** Phase-2 resolution is **in-memory
  only** — a resolved source reverts to unresolved on restart. The fix is
  `metadata_db` made file-backed and keyed to the data roots (it is `:memory:`
  today), with resolve-once write-through, plus a staleness policy that — given
  unreliable cloud mtime — is trust-until-manual-invalidation or size+ctime. Ties
  into the persistent-DuckDB work in `progressive-discovery.md`.
- **Read-tolerance mode (`EXACT` vs `BEST_EFFORT`).** Visualization and compute
  want opposite failure modes on a cold/offline chunk — viz wants "return a coarse
  proxy or `PENDING`, never block, never fail-hard"; compute wants "hydrate and
  wait, or fail loudly, **never silently substitute**." The correctness invariant
  is that compute must never silently receive a downsampled substitute, so the
  caller must *declare* its regime. Planned as a read mode on `TensorReadOptions`,
  **defaulted by caller** (compute plane → `EXACT`; viewer / `/api/slice` →
  `BEST_EFFORT`), best-effort returning data **+ status** (substitute? pending?).
  Open: one flag vs. two verbs (`get_tensor` exact / `get_tensor_view`
  best-effort).
- **Coarse-proxy eviction-stickiness.** Cloud-derived cache entries (especially
  the overview) should be *stickier* than local ones under global LRU — a cloud
  miss is expensive (re-hydrate or fail offline) versus a cheap local re-read, and
  it's what makes best-effort viz fast. Not the cache smaller; the cloud entries
  stickier.
- **Leave-no-trace TTL dehydration.** Optional, contract-neutral convenience:
  release files biopb itself hydrated (detect via the layer-1 placeholder stat;
  never touch files resident before biopb, so a user's deliberate pins survive),
  platform-specific (`attrib +U`, `brctl evict`), best-effort, conservative TTL.
- **Java SDK parity (#111).** The Java `TensorFlightClient` still lacks
  `resolve()` / the directive error and the empty-tensors guards; it must
  implement the resolve action to resolve at all. JS/TS needs none (it refuses
  cloud data server-side).
- **Object-store / fsspec remotes (S3, GCS, OneDrive-Graph).** The same
  unresolved → resolve → redirect-pixel-access-to-a-local-copy model, but with a
  *cheaper* side: range GETs let a remote be **born resolved** (header-only, no
  pixel hydration), and materialization is object-granular for chunked stores. See
  `remote-tensor-cache.md` for the caching-proxy (`tensor-server` source-type)
  path. Remaining remote-specific work: credentials, re-enabling remote-recursion
  in `discover_remote_source` (disabled today — too slow on large buckets), and
  weak OneDrive-Graph fsspec support.
- **Never-warm data** (synced down already-cold, never resident here): the
  persistent descriptor cache fixes the *restart* cycle but not first contact with
  data this server never saw warm — filled by a sidecar manifest written at
  ingest, or the object-store path.
