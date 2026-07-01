# Remote Tensor Server as a Source Type — Local Caching Proxy (+ Proxy-First Client)

**Status:** Implemented — **experimental** (the config surface and the on-disk
segment-cache keys for proxied sources may change without notice; `biopb/biopb#178`, PR #248)
**Component:** `biopb-tensor-server`, `biopb-mcp` (client §§7–8)
**Related:** `biopb/biopb#178` (shared segment cache for remote),
`biopb/biopb#34` (config validation), `biopb/biopb#212` (startup-scan progress),
the claim-based discovery framework, `CacheManager` / `ArrowFileBackend`, the
localhost `chunk_locate` mmap fast path (`#9`), `docs/progressive-discovery.md`,
`docs/cloud-storage-support.md`.

---

## Goal

Add a new **source type** to the tensor server: *another biopb tensor server*.
A config entry whose `url` is `grpc://upstream-host:8815` makes the local server
a **caching proxy** in front of a remote tensor server — it mirrors the remote
catalog, serves reads from a local persistent segment cache, and fetches from
upstream only on a miss.

This is the realization of `#178`'s recommended **Option A (local caching proxy)**,
framed as a per-source adapter rather than a whole-server mode (the two coincide —
see [§9](#9-relationship-to-178)). It also covers the **client-side corollary** of
adopting a proxy: a proxy-first napari client ([§7](#7-client-corollary--a-proxy-first-napari-client))
and a GUI to configure the local server ([§8](#8-local-tensor-server-config-gui)).

Two user-facing capabilities fall out:

1. **A persistent, shared, segment-file cache for remote data** (`#178` goal 2) —
   on POSIX localhost, the MCP dask workers read the proxy's segments through the
   existing `chunk_locate` mmap fast path, so there is **one** cache per machine
   in the OS page cache instead of a per-worker in-RAM `cachey` slice.
2. **No client cache for local reads** (`#178` goal 1) — workers point at
   `grpc://localhost:<proxy>`, so the existing localhost rules zero out the
   per-worker cache automatically.

---

## The load-bearing finding — the segment cache already wraps every adapter

The proxy needs almost no new caching code, because caching is not an adapter
concern today. `TensorAdapter.resolve_chunk_data` (`base.py:476`) — a **base-class
method shared by every adapter** — already wraps the adapter's `get_data(bounds)`
in the cache:

```python
def resolve_chunk_data(self, chunk_id, cache_manager=None):
    array_id, bounds = decode_chunk_id(chunk_id)
    should_cache = cache_manager is not None and (
        is_scaled_chunk(chunk_id) or isinstance(cache_manager.backend, ArrowFileBackend))
    def compute_fn():
        return _as_record_batch(self.get_data(bounds)), ...
    if should_cache:
        entry = cache_manager.get_or_acquire(chunk_id, compute_fn, metadata={"array_id": array_id})
        ...
```

`do_get` calls `adapter.resolve_chunk_data(chunk_id, CacheManager.get_instance())`
(`server.py:1202`). So **any adapter whose `get_data` pulls bytes from an upstream
automatically inherits the `ArrowFileBackend` segment cache, eviction, WAL
recovery, and the `chunk_locate` mmap handoff** — unchanged.

Consequence: `#178`'s **Phase 1 (extract the cache down into `biopb` root to break
the circular dependency)** is **not required** for this approach. That phase
existed because `#178` first imagined the shared cache living in the *client*
(`biopb.tensor.client`, the `cachey` cache). With a proxy, the cache stays where
it already is — server-side — and the proxy is just another server process. The
only new code is a passthrough adapter and the config/expansion plumbing to reach
it.

---

## 1. Config surface

A remote tensor server is declared like any other source. Its `url` is a Flight
endpoint; `type` is the new `"tensor-server"` (auto-detected from the `grpc://`
scheme, so it can usually be omitted). **Any number** of `tensor-server` entries
may be configured, alongside ordinary local/cloud sources, in one proxy:

```json
{
  "sources": [
    { "url": "grpc://lab-store.internal:8815", "alias": "lab",  "credentials_profile": "lab-store" },
    { "url": "grpc://archive.internal:8815",   "alias": "arc",  "credentials_profile": "archive"   },
    { "url": "/data/scratch/" }
  ],
  "credentials": {
    "profiles": [
      { "name": "lab-store", "storage_type": "biopb-tensor", "token": "…bearer…" },
      { "name": "archive",   "storage_type": "biopb-tensor", "token": "…bearer…" }
    ]
  }
}
```

This proxy fronts two upstreams **and** a local directory, with one segment cache
in front of all of them. The upstreams' catalogs are namespaced by `alias` so
their `source_id`s cannot collide with each other or with the local sources (see
[§4](#4-identifier-policy)).

- **`url = grpc://host:port`** — mirror *every* source on the upstream (the
  network analogue of `url = "/data/"` directory discovery).
- **`url = grpc://host:port/<upstream_source_id>`** — mirror a single upstream
  source (the path component is the upstream `source_id`, which is slash-free by
  the `array_id` spec, so the first `/` after the authority cleanly separates
  endpoint from source). This is also the shape each *expanded* concrete source
  carries (see [§3](#3-catalog-mirroring--expansion)).
- **`alias`** (optional, slash-free) — the namespace prefix applied to this
  upstream's mirrored `source_id`s. Optional for a lone upstream with no name
  clashes; **recommended** when a collision is possible (multiple upstreams, or a
  local source sharing an id) — a collider is dropped (first wins) and warned, not
  fatal, so set an `alias` to keep the shadowed source.
- **`grpc+tls://`** is recognized identically (TLS upstream).

### Scheme / type plumbing (small, mechanical)

Three existing functions learn about `grpc`:

- `discovery.is_remote_url` — add `grpc`, `grpc+tls` to `remote_schemes`
  (today: `{s3, gs, gcs, http, https, ftp, az, azure}`). Without this a
  `grpc://` url is treated as a local path and `Path("grpc://…").resolve()`
  mangles it.
- `config.detect_source_type` — return `"tensor-server"` for a `grpc*` scheme,
  so the `type` need not be spelled out.
- `config.discover_sources` **Case 0 (remote)** — today a remote url with no
  `type` raises *"Remote URL requires explicit type"*. A `grpc*` url resolves to
  `tensor-server` instead of erroring, and routes to the new expansion in
  [§3](#3-catalog-mirroring--expansion).

> **Implemented (scheme/type plumbing only).** The recognition layer is in:
> `discovery.is_remote_url` accepts `grpc://` / `grpc+tls://` / `grpcs://`
> (one source of truth — `config._is_remote_url` is its alias);
> `detect_source_type` maps a `grpc*` scheme to `"tensor-server"` (before the
> remote-bail); `SourceConfig.type`'s `Literal` gained `"tensor-server"` and the
> dataclass gained the optional slash-free `alias` field (parsed from the config
> dict, validated in `__post_init__`, surfaced in the JSON-schema emitter +
> unknown-key warner); and `discover_sources` Case 0 auto-detects the type for a
> bare `grpc://` source (other remote schemes still require an explicit `type`).
> A configured `grpc://` source is now *classified* as `tensor-server` and
> returned as-is — it is **not yet served**: `RemoteTensorAdapter`
> ([§2](#2-the-adapter--a-three-layer-passthrough-that-understands-nothing)) and
> the catalog expansion ([§3](#3-catalog-mirroring--expansion)) are the next
> slice, so until then such a source errors cleanly at adapter creation
> (*"Unknown source type: tensor-server"*). Tests:
> `tests/config_discovery_test.py::TestTensorServerSourceType`.

`SourceConfig.type`'s `Literal` gains `"tensor-server"` and the dataclass gains an
optional `alias` field (the namespace prefix, [§4](#4-identifier-policy)). Upstream
auth rides on the existing `credentials_profile` field via a new
`storage_type="biopb-tensor"` profile carrying `token` (and optional
`tls`/`endpoint` overrides) — this keeps auth in one place rather than adding a
bespoke `token` field to `SourceConfig`, and gives **each upstream its own
profile/token** (the multi-upstream case). (A `BIOPB_UPSTREAM_TENSOR_TOKEN` env var
is a reasonable single-upstream convenience fallback.)

---

## 2. The adapter — a three-layer passthrough that understands nothing

`RemoteTensorAdapter(BackendAdapter)` (new, `adapters/remote_tensor.py`). **One
instance per mirrored upstream source**, each bound to its
`(upstream_endpoint, upstream_source_id, alias)`. It holds a
`TensorFlightClient(location, cache_bytes=0, token)` to its upstream and is, by
design, **format-agnostic and chunking-agnostic** — it understands only one thing
beyond pass-through: the **`array_id` rewrite** that maps its local (namespaced)
identifiers to the upstream's and back ([§4](#4-identifier-policy)).

Routing across many upstreams + local sources needs **no new dispatch**: the
server already routes `do_get` by `decode_chunk_id(chunk_id)[0].split("/")[0]` →
local `source_id` → the adapter registered under it (`server.py:1188`,
`_get_adapter_for_chunk`). Because every mirrored source registers under a unique
local `source_id`, the right `RemoteTensorAdapter` (hence the right upstream) is
selected automatically; each instance then rewrites to its own upstream.

**Source layer** (`SourceAdapter`):
- `list_tensor_descriptors()` → mirror the upstream source's tensor descriptors
  (from `client.get_descriptor` / `list_sources`), **with `array_id` rewritten**
  local-ward (`<source_id>` → `<alias>__<source_id>`, preserving any `/field`).
- `get_source_descriptor()` → mirror the upstream `DataSourceDescriptor` under the
  local namespaced `source_id`. `source_url` is a **display-friendly** proxied
  form — `grpc://<alias>:<upstream_source_id>` (e.g. `grpc://lab:experiment1`), or
  `grpc://<host:port>:<id>` when no alias — set on `_source_url` at construction.
  It keeps the `grpc://` scheme so it fits the URL pattern and passes
  `to_catalog_url` through unchanged, while being far more legible in a catalog
  listing than the bare `grpc://host:port` endpoint (identical for every source of
  an upstream). The real endpoint stays on `_upstream_location` for dialing.
- `is_resident()` → `True` once the upstream descriptor is fetched (the proxy
  treats a reachable upstream source as resident; an *unreachable* one is a
  natural fit for the `UnresolvedSourceAdapter` deferral pattern — see
  [§3](#3-catalog-mirroring--expansion)).
- `get_native_pyramid_levels()` / `has_native_pyramid()` → mirror upstream, so
  the proxy advertises the **same** pyramid the upstream does.
- `get_tensor_adapter(tensor_id)` → a tensor-layer view bound to the upstream
  `array_id`.

**Tensor layer** (`TensorAdapter`):
- `get_tensor_descriptor()` → mirror upstream (shape, `chunk_shape`, dtype,
  dim_labels) under the **local** `array_id`.
- `get_read_plan(request_desc)` → **delegate to the upstream**: rewrite the
  request's `array_id` upstream-ward, call upstream `GetFlightInfo`, then translate
  its `FlightInfo.endpoints` into a `TensorReadPlan` — **rewriting each endpoint's
  `chunk_id` local-ward** on the way out. The proxy never re-derives the chunk
  grid, so its `chunk_id`s (modulo the `array_id` field) are **byte-identical** to
  the upstream's even across version skew (different `MAX_ARROW_BATCH_BYTES`,
  splitting tweaks, native vs computed pyramid). The simpler alternative — mirror
  `chunk_shape` and let the inherited `_get_read_plan` re-plan locally — works only
  while proxy and upstream run the same planning code; prefer delegation. (Cache
  the upstream plan per `request_desc` to avoid an extra RPC on every
  `GetFlightInfo`.)

**Chunk layer** — override `resolve_chunk_data(chunk_id, cache_manager)`. The
miss handler rewrites the local `chunk_id` to the upstream's, forwards it to the
upstream `do_get`, and caches the returned `RecordBatch` **under the local key**:

```python
def resolve_chunk_data(self, chunk_id, cache_manager=None):       # chunk_id is local
    versioned = self._with_content_version(chunk_id)              # §5; keyed on the LOCAL id
    def compute_fn():
        rb = self._upstream_do_get(self._to_upstream(chunk_id))   # TensorTicket(upstream chunk_id)
        return rb, _nbytes(rb)
    entry = cache_manager.get_or_acquire(versioned, compute_fn, metadata={"array_id": self.local_array_id})
    ...
```

`_to_upstream` / `_to_local` are a **pure byte splice** on the `chunk_id`'s
length-prefixed `array_id` field — replace `[len][array_id]` and keep every byte
after it (`ndim`, bounds, and any scale suffix) untouched. So the proxy never has
to understand bounds or scale encoding; it only swaps two known strings. This
preserves the "understands nothing" property even with namespacing.

Forwarding the *scaled* `chunk_id` (rather than letting the inherited
`get_data`+`downsample_block` path run) means the **upstream** does the
downsampling and only the small downsampled chunk crosses the WAN — the whole
point of a proxy. `get_data(bounds)` is still implemented (build the upstream
`chunk_id` for `bounds`, `do_get`) as a fallback and to satisfy the abstract
method.

That is the entire data path. The proxy decodes no formats, computes no pyramids,
and re-implements no chunking. The segment cache — already wired into `do_get` —
turns it into the persistent shared cache of `#178` goal 2, **shared across all
upstreams and local sources** of this proxy.

> **Implemented (adapter + data path).** `adapters/remote_tensor.py`
> `RemoteTensorAdapter(SourceAdapter, TensorAdapter)`, registered
> `register_with_type("tensor-server", …)` in `adapters/__init__.py`. It holds a
> lazy `TensorFlightClient(upstream, cache_bytes=0, token)`, mirrors the upstream
> source/tensor descriptors with `array_id` rewritten local-ward, overrides
> `is_resident()` → `True` (a `grpc://` url is a remote scheme, so the base would
> wrongly call it non-resident and trip unresolved-source handling), and overrides
> `resolve_chunk_data` to forward the rewritten `chunk_id` to the upstream
> `do_get` and cache the returned `RecordBatch` under the **local** key. The
> `array_id` byte-splice is `chunk.rewrite_chunk_id_array_id` (pure splice on the
> length-prefixed field; bounds/scale tail untouched). `get_read_plan` is the
> **inherited uniform-grid planner** for this slice — correct because a scaled
> `chunk_id` forwarded to the upstream is downsampled there regardless of which
> levels it advertised; mirroring the upstream's *advertised* pyramid (so native
> OME-Zarr levels are reused rather than recomputed) is the read-plan-delegation
> follow-up in [§10](#10-open-questions--follow-ups). `create_from_config` handles
> the single-source `grpc://host:port/<upstream_source_id>` url form; the bare-host
> "mirror everything" expansion + alias-namespaced registration is the §3 slice.
> Tests: `tests/remote_tensor_adapter_test.py` (byte-splice unit + end-to-end
> proxy of an in-process upstream: mirrored catalog, byte-identical pixels, scaled
> read, and the inherited segment cache — upstream hit exactly once on a re-read,
> entry `locate_entry`-able for the mmap fast path).

---

## 3. Catalog mirroring & expansion

A `tensor-server` source **expands** like a directory. In `config.discover_sources`,
a new branch (reached from Case 0 for a `grpc*` url):

1. Connect to the upstream with `TensorFlightClient`.
2. `list_sources()` (or `get_descriptor` for the single-`source_id` form). For a
   large upstream catalog, prefer a `query_sources` page or the upstream's
   `list_flights` so this does not block startup — and **background it** the same
   way `docs/progressive-discovery.md` backgrounds the local walk (the upstream
   list is the network equivalent of a slow directory walk).
3. Yield one concrete `SourceConfig` per upstream source, each with
   `type="tensor-server"`, `url="grpc://host:port/<upstream_source_id>"`, the
   inherited `credentials_profile`, and the entry's `alias`. A small `extra_config`
   carries the upstream endpoint + `source_id` + alias so the adapter can build its
   id-rewrite without re-parsing.

Each concrete source registers a `RemoteTensorAdapter` under its **namespaced
local `source_id`** (`<alias>__<source_id>`). From there the normal server
machinery (`list_flights`, metadata-DB sync, `GetFlightInfo`, `do_get`, precache)
treats it like any other source — and because multiple upstreams + local sources
all land in the one `source_id`-keyed registry, they coexist with no special
multiplexing layer.

**Collision check at registration.** `register_source` rejects a duplicate local
`source_id`. The expansion runs this check across *all* configured entries
(every upstream's mirrored ids + the local sources) and, on a clash, **drops the
colliding entry (keeping the first) and logs a warning** naming the offending id
and the fix (*"set an `alias` on `grpc://…`"*) — it does **not** abort, so one
bad source can't take down the whole catalog. With distinct aliases the
namespaces are disjoint by construction, so this only fires on a genuine
misconfiguration (two upstreams sharing an alias,
or a local source literally named `<alias>__…`).

**Refresh (follow-up).** The upstream catalog can change. v1 mirrors **once at
startup**. v2 reuses the rescan loop: a `monitor=true` on a `tensor-server` source
triggers a periodic **upstream re-list** + reconcile (add/remove proxied sources)
instead of a filesystem walk — the `SourceManager` diff model applies unchanged
once the "scan" is a remote list. (`monitor` for fs watching does not apply to a
remote url today; this generalizes it.)

> **Implemented (monitor → upstream re-list, adaptive cadence).**
> `create_source_manager` collects the **bare-host** `tensor-server` sources with
> `monitor=true` into `SourceManager._monitored_upstreams` (single-source
> `grpc://host/<id>` entries are excluded — nothing to re-list). `_handle_rescan`
> runs `_reconcile_due_upstreams` every rescan tick (default 30 s) with an
> **adaptive per-upstream cadence counted in ticks** (no wall-clock interval — the
> rescan tick is the unit): a re-list runs every tick while an upstream is
> *changing* or *failing*, and its spacing **doubles per unchanged re-list** (a
> stable upstream is skipped for more ticks, capped at `_UPSTREAM_RELIST_MAX_TICKS`
> = 120 ≈ 1 h, so it isn't queried every 30 s forever). Any change *or* failure
> resets it to every-tick, so a new source / a recovered upstream is mirrored
> within ~one tick rather than after the full backoff (the original
> force-full-only gate made recovery take up to 1 h). `_reconcile_one_upstream`
> returns whether the mirrored set moved (drives the backoff), diffing the
> alias-namespaced desired id set against the currently-mirrored claims and
> applying adds/removes through the **same** `_commit_add_claim` /
> `_commit_remove_source` primitives as the filesystem reconcile. Best-effort per
> upstream: an unreachable one keeps its mirrored sources in place and retries
> next tick. For an *upstream-only* config (no monitored dirs) each pass also
> advances the progressive-discovery freshness signals; `full_scan_in_progress` /
> the first-scan gate fire once, on boot. The watcher ticks on a timer independent
> of its directory set, so the loop runs even with zero monitored dirs. Tests:
> `test_monitored_upstream_relist_adds_and_removes`,
> `test_failed_upstream_retried_on_fast_incremental_cadence`,
> `test_stable_upstream_backs_off_then_resets_on_change`,
> `test_create_source_manager_captures_bare_host_monitored_upstream`.

**Unreachable upstream.** A proxy "resolve" is a *cheap reconnect*, not a cloud
download, so the consented refuse-on-serve model of `UnresolvedSourceAdapter` is
the wrong fit — recovery should be **transparent**. Instead the proxy splits its
own surfaces:

- **Catalog surface degrades to a placeholder.** `list_tensor_descriptors()` /
  `get_metadata()` catch an unreachable upstream and return empty (`[]` / `{}`),
  and `is_resident()` tracks reachability (`_reachable`, updated by those calls),
  so `get_source_descriptor()` yields a catalog row with **empty tensors /
  `data_resident=false`** rather than raising. The source therefore registers and
  stays in the catalog while the upstream is down — registration's metadata-DB
  sync no longer fails-and-rolls-back, so the source is not silently dropped.
- **Serve surface stays live.** `get_tensor_descriptor` / `get_data` /
  `resolve_chunk_data` still raise (→ retryable `UNAVAILABLE`) on a miss, and
  `ListFlights`/`GetFlightInfo` are live, so the **real tensors reappear the
  moment the upstream is back** — no explicit `resolve` action, no consent step
  (a failed catalog call also drops the dead client so the next call reconnects).
  Already-cached chunks keep serving from the segment cache through an outage.

> **Implemented (unreachable-upstream policy).** The catalog/serve split above is
> in `RemoteTensorAdapter` (`_reachable`, `_mark_unreachable`,
> `is_resident()→_reachable`, `list_tensor_descriptors`/`get_metadata` →
> empty-on-unreachable; serve methods unchanged). Two URL forms:
> **single-source** `grpc://host/<id>` down at boot → registers as an empty
> placeholder (no rollback) and recovers transparently; **bare-host**
> `grpc://host` down at boot → its expansion is skipped (no per-source ids are
> knowable until the upstream answers), and recovery requires `monitor=true` (the
> re-list populates it on the next force-full pass once reachable). And
> `create_source_manager` no longer hard-fails to start when the *only* source is
> an unreachable monitored upstream — its guard now counts `monitored_upstreams`,
> so the server boots empty and the re-list fills it in. Tests:
> `remote_tensor_adapter_test.py::TestUnreachableUpstream` (catalog placeholder,
> serve-still-raises, transparent recovery via port reuse) +
> `test_unreachable_sole_monitored_upstream_does_not_block_startup`.

> **Implemented (catalog correctness — two fixes).**
> *(1) Metadata source.* `list_flights`/`list_sources` is lean (empty
> `metadata_json`); the only live RPC that fills it (`GetFlightInfo(with_metadata)`)
> returns it *wrapped* (`{"type","dim_label","metadata"}`), which `get_metadata()`
> would have had to unwrap and which it can't even get from `list_sources`. So
> `get_metadata()` reads the **upstream's DuckDB `sources.metadata_json` column**
> via `query_sources` — that column stores `json.dumps(get_metadata())` verbatim
> (the *raw* dict, no envelope), exactly this method's contract; the local server
> re-wraps it once on serialize. `_localize_descriptor` clears `metadata_json` +
> `pyramid` so mirrored descriptors stay lean (the local server fills both itself).
> *(2) Truncation-safe enumeration.* `list_sources()` is **capped**
> (`max_list_flights_results`), so building a catalog from it silently drops
> sources past the cap — and reconciling against a truncated list would
> *spuriously remove* them. The new `list_upstream_source_ids(client) →
> (ids, complete)` prefers the complete `query_sources("SELECT source_id FROM
> sources")` and only falls back to the capped `list_sources()` (flagged
> `complete=False`) when the upstream has no metadata DB. The bare-host expansion
> and the monitor re-list both use it; the re-list applies **removals only when
> `complete`**. And `list_tensor_descriptors` no longer scans the whole catalog to
> find one source (capped + O(N²)) — it fetches that source's descriptor directly
> via `get_descriptor` (a multi-tensor upstream source is advertised by its default
> field; reading other fields still works). Tests: `test_get_metadata_*`,
> `test_metadata_flows_through_proxy_single_wrapped`,
> `test_expansion_complete_despite_list_flights_truncation`.
> *(3) Physical scale.* `physical_scale` / `physical_unit` are, like the pyramid,
> *cleared and refilled* by the server on every `GetFlightInfo` from a dedicated
> `get_physical_scale()` (the compact #31 summary) — they are **not** carried
> implicitly by `get_tensor_descriptor`, so the base default of `None` silently
> dropped the upstream's physical pixel sizes. `RemoteTensorAdapter` now overrides
> `get_physical_scale()` to read them from the upstream's `get_descriptor`
> response (its server fills them identically). Tests:
> `test_get_physical_scale_mirrors_upstream`, `test_physical_scale_flows_through_proxy`.

> **Implemented (expansion + namespacing + collision check; mirror-once).**
> `config.discover_sources` gained a `credentials_config` param and a
> `tensor-server` branch (`_discover_tensor_server`): the single-source
> `grpc://host:port/<id>` form registers under the alias-namespaced local id
> (`_namespaced_source_id` → `<alias>__<id>`, verbatim when no alias); the
> bare-host `grpc://host:port` form connects (`TensorFlightClient(cache_bytes=0,
> token)`), `list_sources()`, and yields one concrete single-source `SourceConfig`
> per upstream source. `resolve_all_sources` threads `config.credentials` through
> and runs `_resolve_tensor_server_id_collisions` over the flattened set — a
> source_id clash involving a proxy **drops the collider (first wins) and warns**
> rather than aborting, so one bad source can't take down the catalog (non-proxy
> clashes keep historical last-wins). Per-upstream tokens reach the adapter because
> the static-seed `SourceClaim.extra_config` now carries `credentials_profile` and
> `_register_source_claim` rebuilds it onto the `SourceConfig` (it was previously
> dropped). Expansion is **synchronous at startup** under the serve path's
> `tolerant=True`, so an unreachable upstream is warned-and-skipped rather than
> aborting — the `UnresolvedSourceAdapter` deferral above and backgrounding the
> upstream list are still follow-ups. The `monitor=true → upstream re-list`
> reconcile below is **not** yet wired (a `tensor-server` source's `monitor` flag
> is currently inert — `_is_monitored_claim` rejects remote urls). Tests:
> `config_discovery_test.py::TestTensorServerSourceType` (single-source
> namespacing, collision, distinct-alias) and
> `remote_tensor_adapter_test.py::TestBareHostExpansion` + the credentials-profile
> token test.

---

## 4. Identifier policy

The proxy must serve **multiple upstreams and local sources at once** under one
flat, `source_id`-keyed catalog (that is how the server's registry and `do_get`
routing work). So local `source_id`s must be **globally unique within the proxy**,
which means an upstream's ids are **namespaced** by its `alias`:

```
local source_id   =  <alias>__<upstream_source_id>          (slash-free ✓)
local array_id     =  <alias>__<upstream_source_id>[/<field>]
```

`__` is a cosmetic, slash-free separator. Routing does **not** parse it back out:
each `RemoteTensorAdapter` instance stores its `(alias, upstream_source_id)`
explicitly and rewrites via that, so the separator only needs to be readable, not
unambiguous. The `array_id` spec (`proto/biopb/tensor/descriptor.proto`) holds —
the prefix is slash-free, so the **first** `/` still marks the source boundary and
`source_id = array_id.split("/", 1)[0]` still recovers `<alias>__<upstream_source_id>`.

**The cost of namespacing is one `array_id` rewrite**, paid as the pure byte
splice in [§2](#2-the-adapter--a-three-layer-passthrough-that-understands-nothing):
local ids on every surface the client sees (`list_flights`, `GetFlightInfo`
endpoints, descriptors), upstream ids on every call the proxy makes
(`GetFlightInfo`, `do_get`). Because the splice preserves all bytes after the
`array_id` field, `chunk_id`s remain otherwise identical to the upstream's, so the
cache and the localhost mmap fast path are unaffected.

**Transparency note.** Because ids are namespaced, a client addresses a proxied
source as `lab__experiment1`, not `experiment1` — it is *not* id-transparent to the
upstream. That is inherent to multiplexing (the client must say *which* store it
means), and it is the deliberate trade for supporting many upstreams + local data
behind one cache. A lone upstream may set no `alias`, in which case ids pass
through verbatim and transparency is recovered — but the moment a second upstream
or a colliding local id appears, an `alias` is required ([§3](#3-catalog-mirroring--expansion)
collision check).

**Why a flat namespace and not per-upstream sub-catalogs.** Keeping one
`source_id` space lets the entire existing stack — registry dispatch, metadata-DB
`sources` table, `list_flights`, precache, the `do_get` routing in `server.py:1188`
— work unchanged. A nested "endpoint → catalog" model would touch every one of
those. The flat alias prefix buys multi-upstream for the price of a string rewrite.

---

## 5. Cache staleness / versioning (`#178`'s open question)

A persistent segment cache extends *"`chunk_id` is immutable"* **across sessions**.
If an upstream re-registers the same `source_id`/`array_id` with **new content**,
a stale cached chunk is served. The cache key is the **local** `chunk_id` =
`<alias>__<source_id>` + bounds (+ scale). The alias prefix already makes the key
unambiguous *across* upstreams and local sources (no cross-source collision — the
`#45` lineage), but it carries no *content* identity for one source over time.

**Recommendation: fold an upstream content-version into the cache key.**
`_with_content_version(chunk_id)` namespaces the key (or the segment subdir) by a
per-source version token taken, in order of preference:

1. An explicit `content_version` / `etag` on the upstream `DataSourceDescriptor` —
   the clean long-term fix; **propose adding this field** so the proxy has an
   authoritative signal. (This also helps the lazy-input compute plane and any
   future client revalidation.)
2. Else the upstream catalog's existing **`indexed_at`** timestamp (already a
   column on `sources`: `source_id, source_url, source_type, dtype, indexed_at,
   …`) — a coarse but real freshness signal; a re-register bumps it.
3. Else fall back to a **session-scoped** cache namespace (no cross-session
   persistence for that source) — correct but forfeits `#178` goal 2 for sources
   that expose no version.

On a version change the old namespace is simply unreferenced and ages out under
normal eviction; no special invalidation RPC is needed.

---

## 6. The MCP consumer (`#178` Phase 3)

This is what makes the proxy worth building. `biopb-mcp`'s bootstrap, when
`mcp.tensor.server_url` is **remote** (now generalizable to a *list* of upstreams,
plus any local data dirs the scientist configures), launches/owns a **per-user
local proxy** (cache dir under `$XDG`/home, `max_bytes` a whole-user budget) whose
sources are those upstreams + locals, and points the dask workers at
`grpc://localhost:<proxy>`. Then:

- The localhost rules already zero the per-worker `cachey` cache
  (`_resolve_cache_bytes` → 0, the worker-init plugin) — `#178` goal 1.
- POSIX workers read proxy segments through the `chunk_locate` mmap fast path —
  one shared OS-page-cache copy, no per-worker duplication — `#178` goal 2.
- `cache_budget // n_workers` (the per-worker split that this design retires) is
  replaced by "workers cache nothing; the proxy owns the cache."

Multiplexing is what makes this genuinely useful for the bench: a scientist's
local scratch directory and one or more shared lab stores all appear in **one**
catalog behind **one** cache and **one** set of localhost-optimized workers — the
agent's `client` sees a single unified namespace (`lab__…`, `arc__…`, plus local
ids) rather than juggling several connections.

The "whole-server caching proxy" of `#178` Option A is therefore **not a separate
server mode** — it is just a proxy whose sources happen to be `tensor-server`
entries. Per-source adapter and whole-server proxy are the same mechanism at
different cardinalities, and "front several stores + local data at once" is simply
more entries in `sources`.

---

## 7. Client corollary — a *proxy-first* napari client

Once a remote store is reachable as a **cached, unified, persistent** proxied
source, letting the napari client *also* dial arbitrary remotes directly is a
strict downgrade (no cache, no unification, a second connection to babysit) and it
muddies the model. So the client moves to a **proxy-first** posture: the tensor
browser talks to **one** server — the local one — and remote data is reached by
configuring it as a proxied source ([§1](#1-config-surface)), edited through the
config GUI ([§8](#8-local-tensor-server-config-gui)).

### 7.1 Remove the direct-remote connect UI from the tensor browser

The browser today carries an inline "connect to any server" form
(`biopb-mcp/.../tensor_browser/_widget.py`): a **Server URL** `QLineEdit` (`:667`),
a **Token** `QLineEdit` + show/hide (`:676`), and a **Connect** button whose
handler retargets the connection to whatever the user typed
(`_on_connect_clicked`, `:753` → `self._conn.url = self._server_input.text()`).

**Remove** the URL field, the token field, and the Connect button. **Keep**
everything else: the **Refresh** button, the source tree, the server-side
`query_sources` path for large catalogs, the background source-watcher
(auto-refresh on `source_count` change), and `TensorConnection.auto_connect()`'s
**auto-start-a-local-server** fallback (`_connection.py:533`).

### 7.2 Endpoint resolution stays — only the *editing* goes away

The endpoint is still **resolved** (not hard-coded): `TensorConnection.resolve_from_config`
(`_connection.py:189`) keeps its `BIOPB_TENSOR_URL` env → `tensor_browser.server_url`
config → `grpc://localhost:8815` default chain. This is load-bearing: `biopb-mcp`'s
bootstrap points the client at its **managed local proxy** ([§6](#6-the-mcp-consumer-178-phase-3)),
which may not be on the default port. What changes is that `server_url` becomes
**config/env only** — no in-widget text field, and `persist_url()` /
`_on_connect_clicked` (the write-back of a user-typed URL) is dropped. A power user
or the MCP bootstrap still sets it; the casual user never sees a URL box.

### 7.3 Redirect the lost workflow to "add a proxied source"

Removing direct-connect must not dead-end the "I have data on a remote server" use
case. The browser gains (or links to) an **"Add remote tensor server…"** affordance
that appends a `tensor-server` entry ([§1](#1-config-surface)) to the local
server's config and applies it — i.e. it hands off to the config GUI below. The net
UX is *better*: the remote becomes a cached, named, persistent member of the one
catalog instead of a transient ad-hoc connection.

---

## 8. Local-tensor-server config GUI

To pay back the user-friendliness lost in [§7](#7-client-corollary--a-proxy-first-napari-client),
add a GUI that **reconfigures the local tensor server** — most importantly its
`sources` (including proxied remotes), but also cache/pyramid/server knobs. Ties
to `biopb/biopb#34` (config validation) and `biopb/biopb#212` (startup-scan
progress).

### 8.1 It edits the *tensor-server* config — a different file than biopb-mcp's

Be precise about which config: the tool edits the **tensor server's canonical
config**, `~/.config/biopb/biopb.json` (`biopb._config_location`,
`CANONICAL_CONFIG_NAME = "biopb.json"`, JSON-canonical per `#34`). This is **not**
the biopb-mcp client config (`~/.config/biopb-mcp/config.json`, which holds
`tensor_browser.server_url`, `mcp.*`, dask budget). The two stay separate; the GUI
targets the server file, where `sources` live.

### 8.2 Prerequisite — a config *writer*, and the model shared in `biopb` root

Two gaps block a GUI today:

- **No writer exists.** The server-config path is read-only: `load_config` /
  `parse_config` parse JSON/TOML → dataclasses, and nothing serializes back. Add
  `save_config(data, path)` that writes canonical `biopb.json`
  **atomically** (temp-file + `os.replace`, mirroring biopb-mcp's
  `_atomic_write_json`) and, critically, **round-trips on the raw dict, not the
  dataclass** — edit the loaded dict by key and re-serialize so advanced or
  future keys the GUI doesn't surface are preserved, not silently dropped.
  (`dataclasses.asdict()` would clobber unknown keys.) Writing JSON also migrates
  a legacy `biopb.toml` forward (`#34`).
- **The model lives in the wrong package for a client GUI.** The config
  dataclasses + `_CONSTRAINTS` validation live in `biopb-tensor-server`, but
  `biopb-mcp` depends on `biopb` (root), **not** on `biopb-tensor-server`. Follow
  the existing precedent — `_config_location` was already moved **down into
  `biopb` root** "so the CLI and biopb-mcp share one definition" — and move the
  **pure-data** layer down too: the `ServerConfig`/`SourceConfig`/`CacheConfig`/…
  dataclasses, the `_CONSTRAINTS` table, `_validate_config`, `parse_config` /
  `load_config` / the new `save_config`. The **adapter-dependent expansion**
  (`discover_sources`, `detect_source_type`, `resolve_all_sources`) **stays** in
  `biopb-tensor-server`, importing the dataclasses from root. Then server, CLI,
  and the GUI all consume one model with no new heavy dependency.

### 8.3 Validation reuses `#34`'s single source of truth

`#34` already added the declarative `_CONSTRAINTS` table (ranges/enums per
dataclass field) and notes it is meant to feed *"the planned JSON Schema emitter."*
The GUI is that emitter's first consumer: add `emit_json_schema()` derived from
`_CONSTRAINTS`, and drive the form's per-field validation (and inline error text)
from it. One table → server-side `__post_init__` validation **and** GUI form
validation, so they can't disagree. (When `_STRICT_VALIDATION` flips to raise at
the end of the TOML window, the GUI's pre-write check becomes the friendly
front-stop for the same rule.)

### 8.4 The widget

A napari **dock widget** in `biopb-mcp`, reusing the established patterns
(`image_processing/_widget_base.py`: server-URL validation, persistent combos,
progress bar, run/cancel). Surfaces:

- **Sources editor** (the headline): add / remove / edit entries — local dirs/files
  **and** `tensor-server` proxied remotes (url, `alias`, `credentials_profile`/token
  — [§1](#1-config-surface)). This is the concrete home of [§7.3](#73-redirect-the-lost-workflow-to-add-a-proxied-source)'s
  "Add remote tensor server…".
- **Server / cache / pyramid knobs** (advanced, collapsible), each validated via
  [§8.3](#83-validation-reuses-34s-single-source-of-truth).

### 8.5 Applying changes — write, restart, and *show the scan*

The server reads config **once at startup**; there is no hot-reload, SIGHUP, or
`reconfigure` action. So **Save & Apply** = write `biopb.json` → **restart via the
existing managed lever** (`biopb server restart`, already in the umbrella CLI) →
reconnect. A live `do_action("reconfigure")` for incremental source add/remove is a
clean follow-up but is **not** required for v1.

The restart then triggers the **startup discovery scan**, which on a large/cloud
root can run for minutes with no output — `#212` measured **142 s of silence** and
it "looks hung." The GUI must not reproduce that blind wait: after restart it
**polls the `health` action and shows progress** using the freshness fields from
`docs/progressive-discovery.md` — `full_scan_in_progress`,
`last_full_scan_finished_at`, and the climbing `source_count` — turning the
restart into a visible "scanning… N sources" state. `#212` (server-side progress
logging) and progressive-discovery's backgrounded scan + freshness `health` fields
are the **server-side signal**; this GUI is their **consumer**. (Backgrounding the
scan also means the server reaches `SERVING` immediately, so the GUI can reconnect
and watch the catalog *populate* rather than waiting for a complete scan.)

### 8.6 Frontend choice — open decision

A second viable surface already exists: the tensor server ships a **React web app**
(FastAPI sidecar, `:8814`). The same biopb-root load/validate/`save_config`
backend could be exposed as `GET`/`PUT /api/config` and edited there — works
without napari, natural for a headless server admin. Recommended split: **napari
dock widget as the v1 surface** (it is what the napari user lost), with the
editing backend in `biopb` root so a web `/api/config` page can be a second
frontend later. See [§10](#10-open-questions--follow-ups).

> **Resolved → web app.** This decision has since been taken the *other* way: the
> **web-app surface is v1** (`GET`/`PUT /api/config` + `GET /api/admin/status` +
> `POST /api/admin/restart` on the existing `:8814` sidecar), with napari
> integration reduced to a single "open admin" browser action rather than a Qt
> form. The detailed design — routes, the detached-`biopb server restart`
> self-restart, `save_config`, the same-origin auth guard, and the post-restart
> scan-progress UX — lives in **`docs/tensor-server-admin-endpoint.md`**.

---

## 9. Relationship to `#178`

| `#178` phase | This design |
|---|---|
| Phase 0 — finish local (no client cache) | Unchanged; already ~done. Falls out for the proxy. |
| **Phase 1 — extract cache to `biopb` root** | **Not needed.** Cache stays server-side; the proxy is a server, so there is no circular dependency to resolve. |
| Phase 2 — upstream adapter | **This document** — `RemoteTensorAdapter` + config/expansion plumbing. |
| Phase 3 — wire MCP | [§6](#6-the-mcp-consumer-178-phase-3). |
| Phase 4 — retire client `cachey` | Optional, after the proxy covers remote (`#178`). |

Net: dropping Phase 1 makes this materially smaller than the original plan — the
new surface is one adapter (with an `array_id` byte-splice), three scheme/type
touch-ups, an expansion branch, and the `alias` field + collision check. Multi-
upstream and local+proxy add no new dispatch layer — they fall out of the flat
`source_id`-keyed registry ([§4](#4-identifier-policy)).

---

## 10. Open questions / follow-ups

- **Content version on the wire.** Add `content_version`/`etag` to
  `DataSourceDescriptor`? (Cleanest staleness fix — [§5](#5-cache-staleness--versioning-178s-open-question).)
  Until then, `indexed_at` is the fallback.
- **Catalog refresh cadence.** Ship v1 mirror-once, or build the `monitor=true →
  upstream re-list` reconcile immediately? ([§3](#3-catalog-mirroring--expansion).)
- **Writes / upload passthrough.** Read-only in v1. Forwarding `do_put`
  (source creation, `upload_array`, and the **lazy-input compute plane's
  write-back** of result sources) to the upstream is a follow-up; until then a
  proxied source is read-only and results land on whichever non-proxy tensor
  server the compute plane is pointed at.
- **Per-user eviction budget.** The proxy `file_max_total_bytes` should default
  to a whole-user budget, not a per-worker slice (`#178`).
- **Share one upstream connection per endpoint** (`biopb/biopb#249`). Each
  `RemoteTensorAdapter` builds its own `TensorFlightClient`, so a bare-host
  upstream mirrored into N sources opens N connections to the same
  `(endpoint, token)` (plus the expansion + re-list throwaways). Pool them by
  `(normalized_location, token)`. Connection-efficiency only — the data path is
  unaffected.
- **Auth shape.** Confirm `storage_type="biopb-tensor"` credential profile vs a
  direct `SourceConfig.token` field vs env var. (Profile recommended for
  consistency with the existing remote-credential machinery.)
- **Chained proxies / loops.** A proxy whose upstream is itself a proxy works —
  each hop applies its own `array_id` rewrite, so aliases just stack
  (`a__b__source`). Guard against a misconfigured cycle (`A→B→A`) and decide
  whether stacked prefixes need a depth cap for sanity.
- **Alias ergonomics.** `<alias>__<source_id>` is functional but verbose in the
  agent's `client` namespace. Worth deciding if the default (no-alias, single
  upstream) should auto-derive a short alias from the host when a second upstream
  appears, rather than hard-failing the collision check.
- **Config-GUI frontend.** napari dock widget (recommended v1), the existing React
  web app via `/api/config`, or both off one biopb-root backend?
  ([§8.6](#86-frontend-choice--open-decision).)
- **Apply mechanism.** v1 = write + `biopb server restart`. Is a live
  `do_action("reconfigure")` for incremental source add/remove worth building
  sooner, to avoid a full restart + rescan on every source edit?
  ([§8.5](#85-applying-changes--write-restart-and-show-the-scan).)
- **Config-model relocation.** Moving the server-config dataclasses + `#34`
  validation down into `biopb` root ([§8.2](#82-prerequisite--a-config-writer-and-the-model-shared-in-biopb-root))
  is a real refactor — confirm it is in scope as the GUI's prerequisite, and that
  the adapter-dependent expansion staying in `biopb-tensor-server` is the right cut.
- **`server_url` after the connect UI goes.** Keep `tensor_browser.server_url` as a
  config/env-only override ([§7.2](#72-endpoint-resolution-stays--only-the-editing-goes-away)),
  or retire it entirely once the MCP bootstrap is the only writer?

---

## 11. Where to look first

- **Cache wrapping (the free lunch):** `biopb_tensor_server/base.py:476`
  (`resolve_chunk_data`), `server.py:1202` (`do_get`).
- **Config / type / expansion:** `biopb_tensor_server/config.py` —
  `SourceConfig` (`:315`), `detect_source_type` (`:863`), `discover_sources`
  (`:1138`, Case 0 remote at `:1166`), `resolve_all_sources` (`:1294`);
  `discovery.is_remote_url`.
- **Adapter interfaces to implement:** `base.py` — `SourceAdapter` (`:128`),
  `TensorAdapter` (`:370`), `BackendAdapter` (`:551`); the lazy-proxy precedent
  `adapters/unresolved.py`.
- **Upstream client API:** `biopb/.../biopb/tensor/client.py` —
  `TensorFlightClient` (`list_sources`, `get_descriptor`, `get_tensor`), the
  low-level `do_get` in `_fetch_chunk_distributed`.
- **Chunk id / forwarding:** `biopb_tensor_server/chunk.py`
  (`encode_chunk_id` / `decode_chunk_id`, `MAX_ARROW_BATCH_BYTES`).
- **Localhost fast path the workers reuse:** the `chunk_locate` action
  (`server.py:1220`) and the client mmap path (`#9`).
- **Tensor browser connect UI to remove ([§7](#7-client-corollary--a-proxy-first-napari-client)):**
  `biopb-mcp/src/biopb_mcp/tensor_browser/_widget.py` — server/token fields
  (`:667`, `:676`), `_on_connect_clicked` (`:753`); `_connection.py` —
  `resolve_from_config` (`:189`), `auto_connect` (`:533`), `persist_url`.
- **Config model / validation / location ([§8](#8-local-tensor-server-config-gui)):**
  `biopb_tensor_server/config.py` — `_CONSTRAINTS` (`:236`), `_validate_config`
  (`:288`), `load_config` / `parse_config` (no writer yet); `biopb/_config_location.py`
  (`CANONICAL_CONFIG_NAME="biopb.json"`, `find_config`).
- **Reusable Qt + atomic-write patterns:** biopb-mcp's
  `image_processing/_widget_base.py` (validated form widgets) and
  `_config.py` (`_atomic_write_json`, the `_Config` singleton).
- **Restart lever + startup-scan freshness:** `biopb server restart` in
  `biopb/cli.py`; the `health` action freshness fields in
  `docs/progressive-discovery.md` (and `#212`).
