# Remote tensor server as a source type — local caching proxy

**Status:** implemented — **experimental** (the config surface and the on-disk
segment-cache keys for proxied sources may change without notice).
**Component:** `biopb-tensor-server` (adapter + config/reconcile); `biopb-mcp` consumes the proxy.
**Related:** `progressive-discovery.md`, `cloud-storage-support.md`, `tensor-server-admin-endpoint.md`.

## Why

Add a new **source type** — *another biopb tensor server* — so a config entry whose
`url` is `grpc://upstream-host:8815` turns the local server into a **caching proxy**:
it mirrors the remote catalog, serves reads from its local persistent segment cache,
and fetches upstream only on a miss. This is `biopb/biopb#178`'s **Option A (local
caching proxy)**, framed as a per-source adapter rather than a whole-server mode —
the two coincide, because a "whole-server proxy" is just a proxy whose sources
happen to be `tensor-server` entries. Two user-facing wins fall out:

1. **A persistent, shared segment-file cache for remote data.** On POSIX localhost,
   MCP dask workers read the proxy's segments through the existing `chunk_locate`
   mmap fast path — **one** cache per machine in the OS page cache instead of a
   per-worker in-RAM `cachey` slice.
2. **No client cache for local reads.** Workers point at `grpc://localhost:<proxy>`,
   so the existing localhost rules zero out the per-worker cache automatically.

### The load-bearing finding — the segment cache already wraps every adapter

The proxy needs almost no new caching code. `TensorAdapter.resolve_chunk_data`
(`core/adapter_base.py`) is a base-class method shared by every adapter that already wraps
`get_data(bounds)` in the cache: it caches when the chunk is scaled or the backend
is an `ArrowFileBackend`, keying on the `chunk_id` via `get_or_acquire`. `do_get`
(`serving/server.py`) calls `adapter.resolve_chunk_data(chunk_id,
CacheManager.get_instance())`. So **any adapter whose `get_data` pulls from an
upstream automatically inherits the segment cache, eviction, WAL recovery, and the
`chunk_locate` mmap handoff — unchanged.** Consequently `#178`'s Phase 1 (extract
the cache down into `biopb` root to break a circular dependency) is **not needed**:
that phase assumed the shared cache would live in the *client*; with a proxy the
cache stays server-side and the proxy is just another server process.

## Config surface

A remote tensor server is declared like any other source. `type` is the new
`"tensor-server"`, auto-detected from the `grpc://` scheme (so usually omitted);
`grpc+tls://` / `grpcs://` are recognized identically. **Any number** of upstreams
may be configured alongside ordinary local/cloud sources in one proxy, all behind
one segment cache.

```json
{
  "sources": [
    { "url": "grpc://lab-store.internal:8815", "alias": "lab", "credentials_profile": "lab-store" },
    { "url": "grpc://archive.internal:8815",   "alias": "arc", "credentials_profile": "archive"   },
    { "url": "/data/scratch/" }
  ],
  "credentials": { "profiles": [
    { "name": "lab-store", "storage_type": "biopb-tensor", "token": "…bearer…" },
    { "name": "archive",   "storage_type": "biopb-tensor", "token": "…bearer…" }
  ]}
}
```

- **`url = grpc://host:port`** — mirror *every* source on the upstream (the network
  analogue of `url = "/data/"` directory discovery).
- **`url = grpc://host:port/<upstream_source_id>`** — mirror a single upstream
  source (the path is the upstream `source_id`, slash-free by the `array_id` spec,
  so the first `/` after the authority cleanly splits endpoint from source). This is
  also the shape each *expanded* concrete source carries.
- **`alias`** (optional, slash-free) — namespace prefix on this upstream's mirrored
  `source_id`s. Optional for a lone upstream; **required** once a collision is
  possible. Upstream auth rides the existing `credentials_profile` field via a
  `storage_type="biopb-tensor"` profile carrying `token` — one token per upstream,
  no bespoke `SourceConfig.token`.

**Scheme/type plumbing.** `core.remote.is_remote_url` accepts the `grpc*`
schemes (else `Path("grpc://…").resolve()` mangles the url);
`config.detect_source_type` maps `grpc*` → `"tensor-server"` before the remote-bail;
`config.discover_sources` Case 0 auto-detects the type for a bare `grpc://` source
instead of erroring "Remote URL requires explicit type" (other remote schemes still
require an explicit `type`). `SourceConfig.type`'s `Literal` gained
`"tensor-server"`; the dataclass gained the optional slash-free `alias`.

## The adapter — a passthrough that understands nothing

`RemoteTensorAdapter(SourceAdapter, TensorAdapter)` (`adapters/remote_tensor.py`),
registered `register(RemoteTensorAdapter, "tensor-server")`. **One
instance per mirrored upstream source**, bound to its `(upstream_endpoint,
upstream_source_id, alias)`, holding a lazy `TensorFlightClient(location,
cache_bytes=0, token)`. It is format- and chunking-agnostic; the only thing it
understands beyond passthrough is the **`array_id` rewrite** mapping its local
(namespaced) ids to the upstream's and back.

**No new dispatch.** The server already routes `do_get` by
`decode_chunk_id(chunk_id)[0].split("/")[0]` → local `source_id` → the registered
adapter (`_get_adapter_for_chunk` in `serving/server.py`). Every mirrored source
registers under a unique local `source_id`, so the right `RemoteTensorAdapter` (and
thus the right upstream) is selected automatically; each instance rewrites to its
own upstream. Multi-upstream + local sources coexist in the one flat
`source_id`-keyed registry with **no multiplexing layer**.

- **Source layer** — `list_tensor_descriptors` / `get_source_descriptor` /
  `get_native_pyramid_levels` mirror the upstream with `array_id` rewritten
  local-ward. `source_url` is a display-friendly `grpc://<alias>:<upstream_id>`
  (`_source_url`); the real endpoint stays on `_upstream_location` for dialing.
  `is_resident()` tracks reachability (`_reachable`) — overridden `True` because a
  `grpc://` url is a remote scheme the base would wrongly call non-resident.
- **Tensor layer** — `get_tensor_descriptor` mirrors upstream under the local
  `array_id`. `get_physical_scale()` is overridden to read `physical_scale` /
  `physical_unit` from the upstream `get_descriptor` (the server clears+refills
  these per `GetFlightInfo`, so the base default of `None` would silently drop them).
- **Chunk layer** — `resolve_chunk_data(chunk_id)` (chunk_id is a **proxy
  envelope**, biopb/biopb#178 W1): the miss handler peels it
  (`chunk.peel_proxy_envelope`) and forwards the opaque **inner** — the upstream's
  chunk_id, carried VERBATIM, never decoded or rewritten — to the upstream `do_get`
  (`_upstream_record_batch`), then caches the returned `RecordBatch` under the
  envelope's own canonical key (`cache_key_for_chunk_id(chunk_id)`). Forwarding the
  *scaled* inner means the **upstream** downsamples and only the small chunk crosses
  the WAN.

For this slice `get_read_plan` is the **inherited uniform-grid planner** — correct
because a scaled chunk_id forwarded upstream is downsampled there regardless of what
levels it advertised. Delegating `GetFlightInfo` to reuse the upstream's *advertised
native* pyramid (so on-disk OME-Zarr levels are reused rather than recomputed) is a
follow-up.

## Identifier policy

The proxy serves multiple upstreams + local sources under one flat,
`source_id`-keyed catalog, so local ids must be globally unique within the proxy —
an upstream's ids are namespaced by `alias`:

```
local source_id = <alias>__<upstream_source_id>          (slash-free ✓)
local array_id  = <alias>__<upstream_source_id>[/<field>]
```

`__` is a cosmetic separator; routing never parses it back out (each adapter stores
its `(alias, upstream_source_id)` explicitly). The `array_id` spec still holds — the
prefix is slash-free, so the first `/` marks the source boundary and
`source_id = array_id.split("/", 1)[0]` recovers `<alias>__<upstream_source_id>`.
The cost of namespacing is exactly one `array_id` rewrite (the byte splice); because
the splice preserves every byte after the `array_id` field, `chunk_id`s stay
otherwise identical to the upstream's, so the cache and the mmap fast path are
unaffected. Flat namespace, not per-upstream sub-catalogs, is deliberate: it lets
the entire existing stack (registry dispatch, metadata-DB `sources`, `list_flights`,
precache, `do_get` routing) work unchanged.

**Transparency trade.** A client addresses a proxied source as `lab__experiment1`,
not `experiment1` — not id-transparent to the upstream. A lone upstream may set no
`alias` (ids pass through verbatim, transparency recovered), but a second upstream
or a colliding local id makes an `alias` required.

## Catalog mirroring, expansion & refresh

A `tensor-server` source **expands like a directory**. `config.discover_sources`'s
`tensor-server` branch (`_discover_tensor_server`): the single-source form registers
under the namespaced local id (`_namespaced_source_id` → `<alias>__<id>`, verbatim
when no alias); the bare-host form connects, enumerates upstream ids, and yields one
concrete single-source `SourceConfig` per upstream source. Each registers a
`RemoteTensorAdapter` under its namespaced local `source_id`; from there normal
server machinery treats it like any other source. Per-upstream tokens reach the
adapter via `SourceClaim.extra_config` carrying `credentials_profile`.

**Truncation-safe enumeration.** `list_upstream_source_ids(client) → (ids,
complete)` prefers the complete `query_sources("SELECT source_id FROM sources")` and
only falls back to the capped `list_sources()` (flagged `complete=False`) when the
upstream has no metadata DB — because reconciling against a *truncated* list would
spuriously remove sources past the cap. The re-list applies **removals only when
`complete`**.

**Metadata source.** `get_metadata()` reads the upstream's DuckDB
`sources.metadata_json` column (the raw dict, no envelope — the method's contract),
not the wrapped `GetFlightInfo(with_metadata)` payload. `_localize_descriptor`
clears `metadata_json` + `pyramid` so mirrored descriptors stay lean (the local
server refills both).

**Collision check.** `_resolve_tensor_server_id_collisions` runs across the
flattened set; a clash involving a proxy **drops the collider (first wins) and
warns** rather than aborting, so one bad source can't take down the catalog.

**Refresh via `monitor=true`.** For a bare-host upstream, `monitor=true`
generalizes the filesystem rescan into a periodic **upstream re-list** in
`sources/reconciler.py`: `_reconcile_due_upstreams` runs each rescan tick (default
30 s) with an **adaptive per-upstream cadence counted in ticks** — every tick while
an upstream is changing or failing, spacing **doubling per unchanged re-list** up to
`_UPSTREAM_RELIST_MAX_TICKS` (120 ≈ 1 h). Any change *or* failure resets to
every-tick, so a new/recovered upstream is mirrored within ~one tick.
`_reconcile_one_upstream` diffs the alias-namespaced desired set against
currently-mirrored claims and applies adds/removes through the **same**
`_commit_add_claim` / `_commit_remove_source` primitives as the filesystem
reconcile. Best-effort: an unreachable upstream keeps its mirrored sources and
retries.

**Unreachable upstream.** A proxy "resolve" is a cheap reconnect, not a cloud
download, so recovery is **transparent** (no `UnresolvedSourceAdapter` consent
step). The adapter splits its surfaces: the **catalog surface degrades to a
placeholder** — `list_tensor_descriptors` / `get_metadata` catch the failure and
return empty, `is_resident()` follows `_reachable`, so `get_source_descriptor`
yields a row with empty tensors / `data_resident=false` and registration's
metadata-DB sync doesn't fail-and-roll-back. The **serve surface stays live** —
`get_tensor_descriptor` / `get_data` / `resolve_chunk_data` still raise (→ retryable
`UNAVAILABLE`) on a miss, and a failed catalog call drops the dead client so the
next call reconnects, so real tensors reappear the moment the upstream is back.
Already-cached chunks keep serving through an outage.

## The MCP consumer

This is what makes the proxy worth building. When `biopb-mcp`'s
`mcp.tensor.server_url` is remote (generalizable to a *list* of upstreams + local
dirs), bootstrap launches/owns a **per-user local proxy** (whole-user cache budget)
whose sources are those upstreams + locals, and points the dask workers at
`grpc://localhost:<proxy>`. The localhost rules zero the per-worker `cachey` cache;
POSIX workers read proxy segments through the `chunk_locate` mmap fast path (one
shared page-cache copy); `cache_budget // n_workers` is retired for "workers cache
nothing; the proxy owns the cache." Multiplexing is the payoff: a scientist's local
scratch dir and several shared lab stores appear in **one** catalog behind **one**
cache and **one** set of localhost-optimized workers, so the agent's `client` sees a
single unified namespace (`lab__…`, `arc__…`, plus local ids).

## Client corollary — a proxy-first napari client

Once a remote store is a cached, unified, persistent proxied source, letting the
napari client *also* dial arbitrary remotes directly is a strict downgrade (no
cache, no unification, a second connection). So the client moved **proxy-first**:
the tensor browser talks to **one** server (the local one), and remote data is
reached by adding a `tensor-server` source. The inline "connect to any server" form
(Server URL / Token fields + Connect button) is removed; **kept** are Refresh, the
source tree, the server-side `query_sources` path, the background source-watcher,
and `TensorConnection.auto_connect()`'s auto-start-a-local-server fallback. The
endpoint is still *resolved* (`TensorConnection.resolve_from_config`:
`BIOPB_TENSOR_URL` → `tensor_browser.server_url` config → `grpc://localhost:8815`) —
load-bearing because bootstrap points the client at its managed proxy on a
non-default port — only in-widget *editing* goes away. The lost "I have data on a
remote server" workflow redirects to an "add a proxied source" affordance.

## Local-tensor-server config editing

Reconfiguring the local server's `sources` (including proxied remotes) plus
cache/pyramid/server knobs is exposed through the **web-app admin surface**, not a
napari Qt form: `GET`/`PUT /api/config` + `GET /api/admin/status` +
`POST /api/admin/restart` on the `:8814` sidecar, with napari reduced to a single
"open admin" browser action. The tool edits the tensor server's canonical config
`~/.config/biopb/biopb.json` — **not** the biopb-mcp client config
(`~/.config/biopb/mcp-config.json`). Because the server reads config once at startup
(no hot-reload), applying = write → `biopb server restart` → reconnect, then **poll
`health` and show scan progress** (`full_scan_in_progress`,
`last_full_scan_finished_at`, climbing `source_count`) so the post-restart discovery
scan isn't a blind wait. Full route/restart/auth design lives in
**tensor-server-admin-endpoint.md**; the freshness fields it consumes are described
in **progressive-discovery.md**.

## Gotchas

- **Cache staleness is unsolved.** A persistent segment cache extends "`chunk_id` is
  immutable" *across sessions*: if an upstream re-registers the same
  `source_id`/`array_id` with new content, a stale cached chunk is served. The alias
  prefix disambiguates *across* sources but carries no *content* identity over time.
  The current key is `cache_key_for_chunk_id(chunk_id)` with **no content version**
  folded in — the recommended fix (namespace the key by an upstream
  `content_version`/`etag`, or fall back to the `sources.indexed_at` timestamp) is
  **not yet built**. Until an upstream exposes a version signal, re-registered
  content can be masked by the cache.
- **Namespacing is not id-transparent.** Any multi-upstream / colliding-local config
  *requires* an `alias`; a missing one drops the collider (first wins) with a warn.
- **Chained proxies stack aliases** (`a__b__source`) and work, but a misconfigured
  cycle (`A→B→A`) has no depth cap yet.
- **One connection per mirrored source.** Each `RemoteTensorAdapter` builds its own
  `TensorFlightClient`, so a bare-host upstream mirrored into N sources opens N
  connections to the same `(endpoint, token)` (plus expansion/re-list throwaways).
  Connection-efficiency only; the data path is unaffected (`biopb/biopb#249` —
  pool by `(normalized_location, token)`).
- **Bare-host down at boot needs `monitor=true` to recover.** A single-source
  `grpc://host/<id>` down at boot registers as an empty placeholder and recovers
  transparently, but a bare-host `grpc://host` can't be expanded (no per-source ids
  are knowable until the upstream answers), so its recovery depends on the re-list.
  `create_source_manager` no longer hard-fails when the *only* source is an
  unreachable monitored upstream — the server boots empty and the re-list fills it.

## Not done / future

- **Read-only.** No `do_put` passthrough — source creation, `upload_array`, and the
  lazy-input compute plane's write-back all land on whichever non-proxy server the
  compute plane is pointed at. Forwarding writes to the upstream is a follow-up.
- **Content version on the wire** — add `content_version`/`etag` to
  `DataSourceDescriptor` as the clean staleness fix (see Gotchas).
- **Read-plan delegation** — mirror the upstream's *advertised native* pyramid via
  delegated `GetFlightInfo` so on-disk OME-Zarr levels are reused, not recomputed.
- **Per-user eviction budget** — the proxy's `file_max_total_bytes` should default
  to a whole-user budget, not a per-worker slice.
- **Live `reconfigure`** — v1 apply is write + `biopb server restart`; a
  `do_action("reconfigure")` for incremental source add/remove (no full rescan)
  is a possible follow-up.
- **Alias ergonomics** — `<alias>__<source_id>` is verbose in the agent's `client`
  namespace; auto-deriving a short host-based alias on first collision is open.
