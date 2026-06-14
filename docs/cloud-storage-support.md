# Cloud storage support in the tensor server (proposal)

> **Status: proposal.** A design sketch, not a build plan — captures the
> approach and the decisions it rests on so they can be reviewed before any code.

Scope: `biopb-tensor-server` (server + Python SDK), with knock-on effects for the
Java SDK and a (minimal) carve-out for the TS/web client.
Related: `biopb/biopb#8` (read-grid decoupling), the discovery placeholder guard
already in `discovery.py`, the metadata DB (`metadata_db.py`), the pre-cache
worker (`precache.py`).

Users want to archive microscopy data to cloud storage (OneDrive, Dropbox,
iCloud, …) to free local disk, then still browse and analyze it through biopb.
The "synced folder with Files-On-Demand" model — data appears as local paths but
content is *dehydrated* until accessed — is the case this document targets. The
object-store model (S3/GCS, HTTP via fsspec) turns out to be the **same** problem
with a cheaper resolution path — see the generalization at the end of §2.

The guiding insight, which every section below is a consequence of:

> **On a dehydrated cloud file the only thing we can know for free is its URL.**
> Reading *any* byte of content triggers a whole-file recall — which is slow,
> consumes the local disk the user just freed, and blocks indefinitely when the
> device is offline. So shape, dtype, dimension labels, the number of fields, and
> even whether a path is one source or many are all **unknowable without paying
> the full hydration cost.** Everything else follows from refusing to pay that
> cost until a human asks for the pixels.

---

## 1. What cloud storage breaks

Three of the server's working assumptions stop holding on a synced folder:

1. **`stat` is cheap, content reads are not — and hydration is whole-file.**
   `iterdir()`/`stat()` read placeholder metadata without a recall, so *traversal
   is safe*. But opening a file for even four header bytes recalls the **entire**
   object. `stop_before_pixels`-style "I only read the header" cleverness
   (`dicom.py`) and `tifffile` reading only the IFD (`aicsimageio.py:913`) buy
   nothing: the hydration granularity is the whole file, decoupled from how many
   bytes the reader logically consumes.

2. **mtime is unreliable.** The signal the periodic watcher and stability window
   key on is untrustworthy on cloud filesystems — the same reason live-watch is
   already off for these trees.

3. **Reads can block forever.** A dehydrated file on an offline device does not
   error quickly; it hangs. The existing `_is_offline_placeholder` guard
   (`discovery.py:144`) exists precisely to keep an adapter from opening one.

The current defensive posture is therefore "enumerate safely, then **skip** every
dehydrated file" (`discovery.py:184-220`), plus a blanket prune of directories
named `OneDrive*` (`discovery.py:130-141`). Net effect today: cloud data is never
registered — permanently 0 sources. Supporting cloud means *replacing* the skip
with something that records the URL without paying for content.

---

## 2. The model: unresolved sources + lazy resolution

A cloud source is a **URL-only, *unresolved* catalog entry.** Its descriptor
carries an explicit `UNKNOWN` (empty `tensors`), not a missing row. Resolution —
learning shape/dtype/fields — is **lazy and user-triggered**, deferred to the
first real data access, which is the one moment a hydrate is legitimate (the user
asked for the pixels). The first read *is* the resolution event: opening the file
hydrates it and, as a side effect, finally yields the descriptor, which is
**backfilled and persisted** so resolution is paid once per source, not once per
restart.

```
register URL (type ≈ guess, shape UNKNOWN)        ← scan time, free
        │
        │  user opens the source  (foreground, consented hydrate)
        ▼
resolve: hydrate → read real metadata → fill descriptor → persist
        │
        ▼
normal source thereafter (shape known, cached)
```

This rests on two descriptor bits that the current API conflates into one
("is `shape` present?"), and which are genuinely orthogonal:

| Bit | Meaning | Gates |
|---|---|---|
| `resolved` | descriptor known (shape/dtype/fields) — immutable once true | serving (the resolution boundary, §4); TS refusal (§7) |
| `data_resident` | content local & cheap to read **right now** — **volatile** | pre-cache warming (§5); leave-no-trace (§6) |

The volatility of `data_resident` matters: OneDrive re-dehydrates files under
storage pressure, so a *persisted* `data_resident=true` goes stale. The
descriptor field is therefore an **advisory, point-in-time display hint** only;
the authoritative gate is a **fresh** `stat`-based check — `adapter.is_resident()`
reusing `discovery.py:_is_offline_placeholder` (recall-free) — evaluated at the
moment of use. The same stat signal anchors all three layers below; that is the
unifying mechanism of this whole design.

### fsspec remotes are cloud too

The same model is the right way to handle fsspec remotes (S3, GCS, HTTP). A
remote URL is also "data waiting to be hydrated": **allow query, but when pixels
are needed, materialize a local copy and redirect all pixel access to it** —
identical policy, identical `resolved`/`data_resident` bits, identical tolerance
modes (§7) and sticky cache (§6). Practically, a remote whose pixels are needed
is **promoted to a local source**: its bytes are materialized into the local
cache/scratch, and the local copy serves thereafter; the origin URL is retained
only to re-materialize if the copy is dropped. This **retires** the per-adapter
`is_remote` streaming branches (`FSStore` / `RemoteStore` / `RemoteNdTiffFileIO`
scattered through `create_from_config`) in favor of one path — a simplification,
not an addition.

Two things differ from the synced-folder case, both refinements rather than
exceptions:

- **Resolution stays cheap for remotes.** Unlike an OS placeholder (whole-file
  recall on any read), an fsspec remote supports *range* GETs, so the descriptor
  can be resolved by reading just the header (zarr `.zattrs`, the TIFF IFD).
  Remotes can therefore be **born resolved** (shape known, no pixel hydration),
  whereas synced-folder cold data is born unresolved. "Allow query" is genuinely
  cheap for remotes and costly for synced folders — but *pixel* access follows
  the same redirect-to-local policy for both.
- **Materialization granularity follows the format** (§9 again): chunked-object
  stores materialize **per chunk on demand**, keeping the local copy fine-grained
  and preserving the larger-than-disk promise; monoliths materialize
  **whole-file** (no choice). Voluntarily whole-copying a 500 GB chunked remote
  to read one plane would re-introduce the disk-footprint problem layer 3 avoids,
  so the local copy is object-granular wherever the format allows.

The `is_resident()` predicate thus has **two implementations under one
interface** — placeholder `stat` for synced folders, local-copy presence for
remotes — and the policy above them is shared.

---

## 3. Layer 1 — scan / claim

Claim is *content-probing by construction* — adapters read `.zattrs`,
`zarr.json`, OME-XML, `NDTiff.index` to recognize a format. Classified by **bytes
hydrated** (Σ full size of every file `claim()` opens), not bytes logically read:

| Group | Behavior | Examples |
|---|---|---|
| **1 — no content read** | name + `stat` + sibling `exists()` only; safe even offline | `ZarrAdapter` common path (`zarr.py:48`), `NdTiffAdapter`, `Hdf5Adapter`, all extension-only AICS adapters, `NiftiAdapter` |
| **2 — reads a small *separable* sidecar** | cheap online (hydrates only the sidecar); blocks offline | `OmeZarrAdapter` `.zattrs` (`ome_zarr.py:92`), micromanager `metadata.txt` (`tiff.py:469`), `.companion.ome` |
| **3 — opens the large pixel container** | full hydrate; must not run on a placeholder | OME-TIFF embedded-XML sniff (`aicsimageio.py:913`); **DICOM series**, which `dcmread`s *every* slice (`dicom.py:404`) — the worst case |

Two structural facts fall out:

- **The claim phase is already almost entirely Group 1/2.** The large hydration
  surface lives in **registration** (`create_from_config`), where monolithic
  formats open the container to build a descriptor. So deferring registration
  (§2) neutralizes most of it.
- **The blanket guard is the same mechanism as the feature, pointed the wrong
  way.** The name-prune and placeholder-skip that keep the server from stalling
  on a Windows profile also block a *deliberately configured* cloud root. So a
  **per-source opt-in** (`cloud = true`) is required to flip the policy for one
  configured subtree from "skip" to "register-URL-unresolved," without weakening
  the default safety guard elsewhere.

**Decision:** for a `cloud` root, claim degenerates to "record the URL + a
name-only `source_type` guess + mark unresolved." No content open. The one
claim-time Group-3 reader (OME-TIFF sniff) must defer its sniff to resolution.

---

## 4. The robustness prerequisite (audited)

Before any cloud code, the gating question is: **can the stack carry an `UNKNOWN`
descriptor?** A consumer audit of the Python core says **yes, as a contained
change**, because the two halves want different treatment and both are localized:

- **Cataloging/listing already tolerates empty `tensors`.** `metadata_db`
  guards `if source_desc.tensors:` (`metadata_db.py:357`) and leaves
  `shape_summary`/`dtype` NULL; the HTTP `/api/sources` path iterates `tensors`
  and an empty loop degrades. So an unresolved source can already be registered,
  listed, and queried.
- **Serving structurally needs a concrete shape, and every crash site is
  concentrated at one boundary:** `GetFlightInfo → get_tensor_descriptor() →
  _get_read_plan()` and the chunk planner (`base.py:365` dtype parse, `chunk.py:290`
  ndim, `chunk.py:434` pyramid guard).

That boundary is also the natural **lazy-resolution hook**: an unresolved cloud
adapter's `get_tensor_descriptor()` performs the consented hydrate-and-read on
first call, caches, and returns concrete — so every downstream site receives a
real shape, unchanged. The entire cloud-resolution mechanism collapses into one
lazy method.

Two corrections to the naive reading:

- Do **not** "prevent registration of unknown sources" (a tempting audit
  conclusion). That assumes resolve-at-registration — exactly the hydration we
  are avoiding. *Allow* unknown registration; move resolution to first-access.
- The ~5 crash sites should fail **legibly** for the offline/declined case — a
  clean `SourceUnresolvedError("shape unknown — open to resolve")` — instead of
  raw `numpy` errors. That is "make the boundary fail clean," not "make empty
  descriptors work" (which is meaningless — you cannot build a dask array from
  `shape=[]`).

---

## 5. Layer 2 — pre-cache

Pre-cache (`precache.py`) is **pure interactive-responsiveness optimization**: it
warms the file cache so a scientist's first view is already decoded. Its policy is
**exactly inverted for cloud**:

- it *skips* native-pyramid sources (`has_native_pyramid()`, `precache.py:287`) —
  the formats whose coarse level is a small, separately-addressable, *cheap-on-
  cloud* set of chunks;
- it *warms* computed/monolithic sources by reading full-resolution chunks and
  downsampling — i.e. **full hydration** on cloud;
- and its **backlog tier** seeds every existing local source at startup
  (`seed_backlog`, `precache.py:120-145`) and drains it on a background daemon —
  a silent full-archive re-download engine if cloud data is registered as local
  paths.

**Decision: keep background pre-cache *out* of cloud entirely** (option 2 of two
considered). Cloud users accept that interactive response is not fast. This is
chosen over inverting the policy because it is a **pure subtraction** — one skip
gate keyed on the live `is_resident()` check — so the dangerous "warm a cold
source" operation *does not exist* in the cloud path and cannot be mis-built. No
warm-on-resolution code is needed either: the user's own on-demand reads populate
the `ArrowFileBackend` naturally.

- **Given up:** cold first-views (accepted) and no *offline* thumbnail (the only
  real loss).
- **Forward-compatible:** option 2 is a strict subset of "background-warm the
  native coarse level," which can be added later as a pure enhancement.
- **Natural protection, and its backfire:** an unresolved source auto-skips
  (empty shape → `build_pyramid_plan` throws → caught, `precache.py:335-342`). But
  once a source is resolved-and-persisted it returns with a concrete shape, so the
  backlog would re-warm it on the next restart — which is *why* `data_resident`
  must be a separate, live-checked bit (§2): shape-presence no longer protects.

---

## 6. Layer 3 — real chunk read

This is the **one layer where hydration is legitimate**: the user asked for
pixels. The question is managing it honestly, not avoiding it. Two structural
tensions:

1. **Read amplification becomes catastrophic.** `biopb/biopb#8` (chunk size
   conflated with access granularity) is a ~23× annoyance on local disk; on a
   synced folder, scrubbing one Z-plane out of a monolithic file hydrates the
   **entire object**, because OS placeholder hydration is whole-file. The "read
   one chunk" abstraction does not exist at the cloud boundary for monolithic
   formats — another argument for per-chunk-object pyramidal stores (§9).

2. **Disk footprint and cache eviction.** biopb's own cache *is* bounded
   (`[cache] max_bytes`, global LRU eviction on every write — `precache.py:27-29`),
   so it does **not** rebuild the archive. The real disk re-consumption is the
   **OS hydration** footprint, *outside* biopb, whose only lever is granularity
   (tension 1). The genuine cache-specific issue is the *inverse* of "too big": a
   cloud cache **miss** is expensive (re-hydrate, or fail offline) versus a cheap
   local re-read, and global LRU lets a burst of local full-res traffic evict the
   hard-won cloud coarse proxy. So **cloud-derived cache entries (especially the
   overview) should be *stickier* than local ones** — not the cache smaller.

**Decision: delegate the disk-footprint cost to the user.** They asked for the
data; they own the cost. Default = do nothing. An optional, **contract-neutral**
future convenience is a **"leave-no-trace" TTL dehydration** mode: scoped
*strictly* to files biopb itself hydrated (detect via the layer-1 placeholder
stat — record pre-read residency, never touch files resident before biopb, so a
user's deliberate pins are never undone), optionally extracting the coarse proxy
into biopb's cache before releasing the source. Opt-in, best-effort,
platform-specific (`attrib +U` on Windows, `brctl evict` on macOS, drvfs murky),
partly redundant with OneDrive's own Storage Sense, and conservative on the TTL
(span a working session or thrash). Not core.

---

## 7. The two use patterns: a tolerance axis the API must expose

Visualization and compute want **opposite failure modes** — a third dimension
orthogonal to `scale_hint` that the read API currently collapses (`scale_hint` =
*which* resolution; this = *failure tolerance*):

| | Visualization | Compute |
|---|---|---|
| Wants | fast, non-blocking | accurate, complete |
| Cold/cloud/offline chunk | return coarse proxy or `PENDING`; async-fill; **never block, never fail-hard** | hydrate + wait, or **fail loudly** offline; **never substitute** |
| Silent downsample substitute | acceptable (progressive, like map tiles) | **forbidden** |

The dangerous failure is the cross-wire: a best-effort fallback leaking into
compute hands a measurement a silently-downsampled array. So the **correctness
invariant is: compute must never silently receive a substitute.** This is the
hard edge of the recurring "fail legibly, never silently-wrong" theme — inverted
between the two callers, which is exactly why the caller must *declare* its
regime; the data plane cannot infer it.

**Composition with the rest of the design:**

- Resolution stays a **foreground, consented** user action, so background viz
  never triggers a synchronous hydrate. Steady state: viz = best-effort, compute
  = exact, both against an already-resolved source.
- The coarse-proxy **eviction-stickiness** (§6) is precisely what makes
  best-effort *fast* — the substitute the viewer gets on a cold miss *is* the
  cached proxy.

**API shape:** a read mode on `TensorReadOptions` — `EXACT` vs `BEST_EFFORT` (or
a `deadline` + `allow_substitute` pair) — **defaulted by caller** so neither side
thinks about it: the compute plane (`ProcessImage`/`ObjectDetection` pulling lazy
inputs) defaults `EXACT`; the viewer and `/api/slice` default `BEST_EFFORT`.
Best-effort needs a richer return (data **+ status**: substitute? pending?);
exact returns data-or-error. This axis is latent even for local cold decodes;
cloud merely forces it to be explicit.

**Open choice:** one `EXACT|BEST_EFFORT` flag (less surface area) versus two
distinct verbs (`get_tensor` exact / `get_tensor_view` best-effort), which make
the fail-loud-vs-substitute invariant impossible to invoke by accident. Decide
when the code is written.

---

## 8. Polyglot ring

The descriptor's `UNKNOWN` state reaches three consumers; effort is matched to
each client's role:

| Client | Role | Treatment |
|---|---|---|
| **Python server** | owns resolution | the real work: lazy `get_tensor_descriptor()` + clean `SourceUnresolvedError` at the boundary (§4) |
| **Python client** | analysis SDK | small: tolerate empty `tensors` in listing; propagate the error |
| **Java SDK** | analysis SDK (mirrors Python) | mirror the Python *client* guards (`SerializedTensor` fetch, descriptor parse); **not** the server machinery — resolution is server-side and never duplicated in a client |
| **TS / web** | lightweight viewer | **refuse cloud data**, enforced server-side at the FastAPI sidecar (`/api/sources` filters out unresolved sources; `/api/slice` returns 409) → zero TS changes, zero crash risk |

Trade-off of the sidecar filter: cloud sources are *invisible* (not greyed) in
the web UI. Visible-but-disabled is a small opt-in TS change for later.

---

## 9. The format thesis

The same conclusion recurs at every layer, hardening as it goes:

- **L1:** separable metadata (zarr `.zattrs`) claims cheaply; monolithic headers
  do not.
- **L2:** only a native, per-chunk-object pyramid yields a *cheap cloud
  overview*; a computed pyramid needs full-res reads.
- **L3:** "read one chunk" only exists for per-chunk-object stores; a monolith
  hydrates whole.

> **Pyramidal / chunked-object stores (OME-Zarr) are the supported cloud path.**
> Monolithic formats (OME-TIFF, CZI, ND2, …) can be *listed* and
> *resolved-on-demand* (one accepted full hydrate), but get **no cheap thumbnail
> and no cheap sub-region read**. The strong recommendation is that cloud
> ingestion **transcode to OME-Zarr at archive time**, so the metadata and the
> coarse pyramid exist as separable, pinnable, cheaply-warmable objects.

---

## 10. Deferred / out of scope

- **Never-warm data** (synced down already-cold, never resident here): the
  persistent-descriptor cache resolves the *restart* cycle but not first contact
  with data this server never saw warm. That gap is filled by either a **sidecar
  manifest** written at ingest, or the **object-store path** below — both
  separate work.
- **Object-store specifics.** Per the §2 generalization, fsspec remotes
  (OneDrive via Graph, S3, GCS) follow the *same* unresolved → resolve →
  redirect-to-local pixel policy — not a separate model. What remains
  remote-specific is the *cheaper* side (resolve via range GET, born resolved)
  and the operational work it needs: credentials, re-enabling remote-recursion in
  `discover_remote_source` (today disabled — "too slow on large buckets"), and
  weaker OneDrive-Graph fsspec support. A cheaper resolution path into the same
  model, not an alternative to it.
- **Leave-no-trace TTL dehydration** (§6): optional, contract-neutral.
- **Metadata DB durability**: this design needs `metadata_db` made file-backed
  and keyed to the data roots (it is `:memory:` today — `metadata_db.py:159`),
  with a staleness policy that, given unreliable cloud mtime, is
  trust-until-manual-invalidation or size+ctime.

---

## 11. Build order

1. **Descriptor contract** — `resolved` (already implicit) + `data_resident`
   advisory field; `SourceUnresolvedError`; legible failure at the
   `_get_read_plan` boundary. *Pure robustness, no cloud code, unblocks the rest.*
   **(done — cloud phase 1.)**
2. **Unresolved adapter + lazy resolution** hook; per-source `cloud = true`
   opt-in; recall-free `claim()` (residency-guarded defer in each reader adapter)
   + OME-TIFF sniff deferral. Resolution = re-run claim + `create_from_config` on
   the hydrated path on first `GetFlightInfo` (resolve-on-serve); the in-memory
   metadata-DB row is backfilled via the `sync_source_added` upsert. **(done —
   cloud phase 2; see `biopb-tensor-server/ARCHITECTURE.md` "Cloud / synced-folder
   sources" and `tests/cloud_phase2_test.py`. Multi-file monolith member grouping
   degrades on cloud, as designed.)**
3. **Persistent metadata DB** keyed to roots; resolve-once write-through.
4. **Pre-cache skip gate** on `is_resident()`; exclude cloud from the backlog.
5. **Read tolerance mode** (`EXACT|BEST_EFFORT`) + best-effort status return;
   caller defaults; sidecar refusal filter for TS.
6. *(later, optional)* native-coarse background warming; leave-no-trace TTL;
   object-store path.
