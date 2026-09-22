# biopb-tensor-server Architecture

## Overview

`biopb-tensor-server` provides two server components:

1. **TensorFlightServer** — Arrow Flight / gRPC server for chunked array access.
2. **FastAPI HTTP Server** — Browser-accessible HTTP API. It wraps the Python
   `TensorFlightClient` and re-exposes it as HTTP/JSON (+ binary slices). See
   **[docs/http-server.md](docs/http-server.md)**.

```
Client (Python or TypeScript)
    │
    ├── Arrow Flight / gRPC  (default :8815)  ─────► TensorFlightServer
    │                                                        │
    └── HTTP/JSON + binary   (default :8814)  ─────► FastAPI Server
                                                             │
                                                   TensorFlightClient
                                                             │
                                                    TensorFlightServer
                                                             │
                                              ┌──────────────────────────┐
                                              │  TensorAdapter           │
                                              │  (Zarr / OME-Zarr /      │
                                              │   OME-TIFF / HDF5 / CZI) │
                                              └──────────────────────────┘
```

```python
server = TensorFlightServer("grpc://0.0.0.0:8815")
server.register_source("my-zarr", ZarrAdapter(arr, "t0", ["z", "y", "x"]))
server.mark_ready()  # health reports SERVING (else STARTING forever)
server.serve()  # blocking
```

The `biopb-tensor-server` CLI launcher is the authoritative entry point. Code
that drives `TensorFlightServer` directly (as above) is responsible for calling
`mark_ready()` itself once it is ready to serve.

Datasets are keyed by `source_id`. Each source maps to one adapter (decoding
logic), which may expose multiple tensors (e.g., multi-field) from one source.


### Package layout

The `biopb_tensor_server` package is organized into layered subpackages:

- **`core/`** — foundational primitives and contracts: adapter ABCs, the
  `claim()` discovery protocol, `config` (the schema and its file I/O, nothing
  that reads a disk or a network), the `axes` vocabulary + its `normalize` seam,
  and the live `source_registry`.
- **`serving/`** — the runtime: `server` (Arrow Flight), `http_server` (FastAPI
  sidecar), `upload_manager`, `precache`, `renderer`, plus what those servers
  own directly: `metadata_db` (the DuckDB store whose `sources` / `rois` /
  `decode_rates` tables back the surfaces they expose), `tls` (the listener's
  self-signed leaf) and `activity` (in-flight read tracking). Builds on `core`.
- **`sources/`** — source lifecycle: `resolve` (config entries -> concrete
  sources), `source_manager` + `tree_scanner` + `watcher` (scan orchestration)
  and `reconciler` (the confirmed-catalog single writer). Builds on `core` and
  `adapters`; it names `serving`'s server and `metadata_db` only in type
  annotations, never importing them at runtime.
- **`adapters/`**, **`cache/`** — storage-format adapters and the virtual-chunk
  cache. A bare module name here is an adapter; an underscored one is shared
  machinery: `_scale` and `_ome_rois` (format metadata in, common representation
  out), `_handle_reaper`, and `_writable` — the mixin two of the adapters
  inherit for progress, completion and disposal. `cache/segment_index` and
  `cache/segment_store` are the Arrow segment format itself, so an uploaded
  `cache://` member writes what the chunk cache writes: the codec and the boot
  index are shared, the budget and the eviction are not.
- Top level — the entry points: `cli`, `__main__`, and `logging_config`, which
  configures the `biopb_tensor_server` logger hierarchy for them.

The dependency order is `core <- cache <- adapters <- {sources, serving} <-
cli`, with no edge back up. Executing a config entry — walking a directory
through the adapters' `claim()` protocol, listing an upstream catalog — needs
the adapter registry, so it lives in `sources/resolve`, above the adapters,
rather than in `core/config` reaching down into them.

---

## TensorFlightServer

- **Module:** `biopb_tensor_server.serving.server`
- **Class:** `TensorFlightServer(flight.FlightServerBase)`
- **Default location:** `grpc://0.0.0.0:8815`

`TensorFlightServer` is a thin Flight protocol handler; its mutable state lives
in three collaborators it composes:

| Collaborator | Class | Owns |
|---|---|---|
| `server.sources` | `SourceRegistry` |  The `source_id → SourceAdapter` map and adapter-lifecycle |
| `server.activity` | `ActivityTracker` |  In-flight activity tracking. Fed by every heavy read — `do_get`, `warm`, and `chunk_locate` |
| `server.uploads` | `UploadManager` | The writable-server upload boundary. `register_source` mints a container under `write_dir`; `add_tensor` puts `<scheme>://<source_id>/<field>` in one, the scheme naming the store format (`zarr://` an OME-Zarr image group, `cache://` the chunk batches as uploaded in segments of their own) and nothing else. Nothing here creates a source, so the catalog row kept in step is always the parent's. Progress and discard live on the adapter (`adapters._writable.WritableSource`), so a discarded upload is a tombstone its source still holds, not a second record. Its `reap` sweep (`upload_ttl`) discards uploads that went quiet, detaches aged tombstones, and takes a source left empty for that long. A store on disk is the server's own under `write_dir`, so discard removes it, and one still marked pending at startup is a crashed upload deleted before discovery runs — while a READY one is re-adopted, which is what makes a finished upload outlive the process |

### Flight protocol (v2)

Three flights. Every verb dispatches on a proto **oneof** -- never a sentinel
id or a byte-prefix sniff -- and the arm names the flight:

| Flight | Data | GetFlightInfo | DoGet ticket | DoPut command |
|---|---|---|---|---|
| `catalog` | public: the DuckDB tables (`sources`, `decode_rates`) | path descriptor (`for_path("sources")`) -> the table's schema + a ticket that reads it | `TensorTicket.catalog_query` -- runs the SQL, truncation flags on the stream's schema metadata | -- |
| `data` | private: pixels | `FlightRequest.tensor_read` -> chunk endpoints; fills `pyramid` / `metadata_json` on request | `TensorTicket.chunk_id` (opaque, server-minted) | `PutCommand.chunk` (writable servers) |
| `roi` | private: annotations | -- | `TensorTicket.roi_read` -> ROI rows (`biopb.image._roi_rows`), `sets` + `truncated` in schema metadata | `PutCommand.roi_put` / `roi_delete`, reply in the put's app_metadata |

`ListFlights` advertises the catalog only: one flight per table by path, with
its real Arrow schema and a `SELECT * FROM <table>` ticket, so a stock Flight
client can browse without a biopb proto. A source crosses the wire only as its
catalog row — `resolve` returns the row it just wrote, and each SDK hands rows
to the caller rather than decoding them into a structure of its own choosing. A source's pixels and annotations are
addressed, not listed.

**Two token tiers** (`_authorize`): the catalog tier requires the server-wide
token when one is configured; the private tiers require a source's capability
token when the adapter carries one, else the server-wide token. A private
source may still be catalogued -- the token gates reading, not knowing.

Custom `do_action` verbs: `health` (reports `protocol`), `register_source`,
`add_tensor`, `set_upload_status`, `chunk_locate`, `cache_stats`, `resolve`, `warm`,
`add_source`, `remove_source` (below), and `roi_prune`.

#### Server-advertised pyramid (`TensorDescriptor.pyramid`)

`GetFlightInfo` fills `pyramid` with an ordered list of `PyramidLevel`
(`scale_hint`, `reduction_method`, logical `shape`, `native`).

Two sources of pyramid specs (`TensorFlightServer._advertised_pyramid`):

- **Native** — data formats that ship a real on-disk pyramid override
  `TensorAdapter.get_native_pyramid_levels()` (`OmeZarrAdapter` and
  `QptiffAdapter`) to return one `native=True`, `reduction_method="precompute"`
  level per on-disk resolution.
- **Computed** — everything else gets `chunk.build_pyramid_plan(...)`: full
  resolution plus at most two levels — a 2-D target and a 3-D one — from the
  authoritative `[pyramid]` config knobs. The precache worker warms every level
  of this same plan but the first, each over a centred window bounded by
  `[precache] warm_budget_bytes`. On by default.

---

## Adapter interface

Two role ABCs in `core/adapter_base.py`, and they **nest**: `TensorAdapter`
subclasses `SourceAdapter` (biopb/biopb#380). The role *scopes* stay disjoint
where they are declared — `adapter_base.py` asserts that at import time
(`_SOURCE_SCOPED_API` / `_TENSOR_SCOPED_API`), so a tensor-scoped method can
never be written onto `SourceAdapter`.

Every concrete format adapter subclasses `TensorAdapter` and fills both roles in
one object. The lone source-only adapter is `UnresolvedSourceAdapter`, which has
no tensors until it resolves.

A source can also answer for **label sets** it did not produce (biopb/biopb#1059):
`SourceAdapter.label_sets` merges what the format reads from its own file
(`get_embedded_labels`, an OME-Zarr's NGFF `labels/` group) with what was
attached to it (finished sidecars under `write_dir/labels/<source_id>/`, by a
registration hook), each checked to span the image it binds to. The serve path resolves tensors through `resolve_tensor` /
`resolve_chunk_adapter`, which try a `.../@labels/<name>` field against the sets
before delegating to the format; `catalog_tensors` lists sets after the image
tensors. See **[docs/label-tensors.md](docs/label-tensors.md)**.

| Method | Returns |
|--------|---------|
| `list_tensor_descriptors()` | `list[TensorDescriptor]` — the source's tensors, as structural catalog entries |
| `get_tensor_descriptor()` | `TensorDescriptor` proto — the full serving descriptor of one *bound* tensor |
| `get_data(bounds)` | `np.ndarray` — decodes only the requested sub-region |
| `get_native_pyramid_levels()` | `list[PyramidLevel]` or `None` — native pyramid levels |

### Catalog entry vs serving descriptor (biopb/biopb#812)

The two descriptor methods answer different questions, and the split runs all the
way to the wire:

- **Structural** — `array_id`, `dim_labels`, `shape`, `dtype`. Stable per tensor
  and derivable from the container's index without opening one, so a *source*
  answers for all its tensors at once. This is what `list_tensor_descriptors()`
  returns and what the DuckDB `sources.tensors` STRUCT stores — the row is the
  only representation of a source that crosses the wire.
- **Serving** — above all the transfer `chunk_shape`, plus `pyramid` and
  `physical_scale`. These depend on the tensor actually being selected: the bound
  scene's own chunk layout, its labels, its native pyramid level, the request's
  scale. Only the adapter `get_tensor_adapter(array_id)` returns can answer them,
  and `GetFlightInfo` — which binds first — is where they reach a client.

`adapter_base.catalog_entry()` is the projection, and `catalog_tensors()`
re-applies it as the row is written, so no adapter can publish a read plan into
the catalog. A client that needs a grid describes the tensor; an empty
`chunk_shape` is not a fallback to plan on.

### Canonical axis order (biopb/biopb#596)

Adapters read whatever axis order their upstream reader emits. The server
normalizes that at the adapter seam, so the wire carries a guarantee instead of
each consumer re-deriving "which axis is Y/X/Z/S" with its own vocabulary:

> **Z, Y, X and S appear last, in that relative order**; every other axis — T, C,
> and any unrecognized label — keeps its relative order ahead of them.

Relative order, not index: `[z, dimq, y, x]` normalizes to `[dimq, z, y, x]` —
`dimq` moved relative to nothing, but a trailing axis moved out from in front of
it. And only Z/Y/X/S count as trailing; T and C classify through the same
vocabulary but have no canonical place, so they ride with the unlabeled.

The rule is `core/axes.py::canonical_permutation`; `core/normalize.py` is the
seam that applies it, and `SourceRegistry.register` — the single registration
chokepoint — is where it attaches. An already-canonical adapter is returned
**unchanged** (same object, same cost), which is nearly all of them: `bioio`
fixes `TCZYXS` upstream, and OME-TIFF / QPTIFF / TIFF-sequence / ndtiff / DICOM
are compliant by construction. `nifti` (which emits X before Y) is the one
family whose behavior actually changes.

| | |
|---|---|
| **Not in scope** | Unlabeled stores (`zarr`, `hdf5`) emit `dimN`, so nothing is reordered and nothing is relabeled — promoting a positional *guess* to a wire *assertion* would be wrong for e.g. an unlabeled `[y, x, c]`. Axis semantics come from the format; registration cannot relabel them. |
| **Fail-safe** | Ambiguity degrades to identity rather than moving pixels on a guess: rank mismatch, a duplicated canonical axis, or an `S` label that fails `samples_axis`' size-3/4 gate. Same posture the render path took toward adapter-supplied labels. |
| **chunk_ids** | Untouched — minted by the wrapped adapter and opaque here, so versioned / scaled / precompute-level ids all pass through. What is permuted is the client-visible geometry (descriptor + endpoint `bounds`) and the pixels. |
| **Cache** | The transpose happens *before* the cache store, so a segment holds what the client is served and the localhost mmap fast path stays valid. `CACHE_FILE_FORMAT_VERSION` was bumped to `2` for that (same layout, reordered content); an older client declines the fast path and reads the same normalized chunk over `do_get`. |
| **Plans** | `plan_flight_info` / `get_read_plan` are delegated and their answer permuted, not re-derived — which is what keeps the native-pyramid `precompute` routing working underneath. |

An order this server does not own is **refused, not permuted** — permuting works
only where the server owns the whole read path, and two seams don't. Both report
through the shared `core/axes.py::noncanonical_order`:

| | |
|---|---|
| **Writes** | `add_tensor` rejects a non-canonical declared order up front, so a writable source never disagrees with what `put_chunk` wrote — `physical_scale` and `chunk_shape` arrive aligned to the uploader's labels. |
| **Remote proxy** | Its upstream owns the order in the same sense: that server mints the chunk_ids, plans the reads (#295) and sizes the grid. So the proxy opts out of wrapping (`_normalizable_axes = False`) and refuses a non-canonical upstream at `plan_flight_info` / `get_read_plan`. The source stays catalogued and listed; only reads fail, with an error naming the order. Costs upstream-first upgrade ordering across a federation, and buys a check that holds nothing stateful — a re-seed or an upstream upgrade is picked up on the next open, where a frozen permutation would have silently mis-served it. |

### Adapter file-handle policy (biopb/biopb#71)

On Windows an opened/pinned file cannot be deleted, moved, or renamed, and on
POSIX an unlinked multi-GB volume frees no disk space. The default is therefore
**hold nothing between reads**; a persistent handle is opt-in and must be
justified by open cost.

| Open cost | Policy | Adapters |
|---|---|---|
| O(1) and/or fast (< 1 ms) | **reopen per read**, no handle, no `close()` needed | `hdf5`, `mrc`, TIFF sequences, `bioio`, `dicom`, local `zarr` |
| O(N) and/or unbounded | persistent handle + `close()`, and TTL reaper (`handle_reaper_ttl`) | `ome-tiff`, native plain TIFF/LSM, `qptiff`, `ndtiff` |

---

## Chunk caching / transcoding

`CacheManager` sits between `DoGet` and the adapter, backed by a persisted file
cache transcoding chunk data to Flight IPC format (`cache/file_backend.py`). It
serves a localhost client over an mmap fast path, bypassing the round trip
through a socket. See
**[../docs/localhost-fast-path.md](../docs/localhost-fast-path.md)**.

---

## Discovery and Monitoring

### Discovery protocol (`core.discovery`)

Adapters **claim** filesystem paths they recognize (a `claim()` classmethod
each). `AdapterRegistry.get_claims_for_path` returns claims in **registration
order** (`adapters/__init__.py::get_default_registry`), highest-specificity
first. Notably,

- *OME-TIFF before TIFF-sequence* — OmeTiffAdapter *file*-claims a set of
  `.ome.tif` files while TiffSequenceAdapter claims a dir.
- *OME-Zarr before plain Zarr* — both can claim a `.zarr`, so the specific one
  must win.

A `SourceClaim` (`__slots__`) carries `source_type` / `primary_path` /
`source_id` / `extra_config` / `is_remote`; `DiscoveryState`
holds the `source_id <-> path` maps and the `on_source_added` /
`on_source_removed` callbacks the `SourceManager` wires.

**Progressive discovery (biopb/biopb#212).** The CLI launcher reaches `SERVING`
ASAP and runs the monitored bootstrap scan in the background; the catalog grows
*within* that scan as each source is claimed (see Directory Monitoring below).
See **[docs/progressive-discovery.md](docs/progressive-discovery.md)**.

### Directory monitoring (`sources.watcher`, `sources.source_manager`)

`PeriodicRescanWatcher` emits a `RESCAN` on a fixed interval; per rescan the
`SourceManager` delegates the filesystem-signature walk to `TreeScanner` (a fs
walker gated on the stability window, returning an immutable `ScanSnapshot`), runs
discovery on the snapshot's paths, and diffs the result against the confirmed
catalog.

**Moves** within a monitored dir preserve `source_id`; a move out is a delete,
a move in a create.

### Cloud / synced-folder sources (`cloud = true`)

On a cloud-synced folder (OneDrive/Dropbox/iCloud "Files-On-Demand") content is
*dehydrated* until read. Discovery **skips offline placeholders**, unless the
`cloud = true` switch is set in source configuration, in which case the
dehydrated sources are registered as `unresolved` and are resolved on demand
(pulling the full file content down from the cloud). See
**[docs/cloud-storage-support.md](docs/cloud-storage-support.md)**.

### Runtime source registration (`add_source`)

The `add_source` Flight action registers an existing path on the **server's**
filesystem at runtime — the wire entrypoint behind the tensor-browser's
drag-and-drop. It routes the dropped path through the same claim → adapter →
catalog pipeline the watcher uses, so a folder may register several sources.

### ROI annotations (the `roi` flight)

User-drawn 2-D ROIs live in a `rois` table in the same DuckDB catalog as
`sources`, one row per ROI, anchored on the unversioned `array_id`. They are a
sibling table, not a field inside a source row: `sources.metadata_json` is
adapter-produced and rewritten on every re-registration. The table is private
data -- read over DoGet and written over DoPut, authorized per source like
pixels -- and is deliberately not on the SQL surface (biopb/biopb#1010).
Orphans (annotations whose source is gone) are reported and pruned by the
`roi_prune` action.

See **[docs/roi-annotations.md](docs/roi-annotations.md)**.

---

## CLI Launcher / lifecycle

**Command:** `biopb-tensor-server launch`

```
biopb-tensor-server launch --config biopb.json [--host 127.0.0.1] [--port 8815] [--writable] [--web-port 8816] [--web-host 127.0.0.1] [--cors ORIGIN]

# for grpc only (no web server) — same flight options + token handling as launch
biopb-tensor-server serve --config biopb.json [--host 127.0.0.1] [--port 8815] [--writable] [--tls] [--san NAME]

# generate / rotate the self-signed TLS cert and print its fingerprint
biopb-tensor-server cert init [--force] [--san NAME]
```

### Startup sequence

`serve` and `launch` share the prologue and the flight-server bring-up
(`_setup_flight_server`); they differ only in what blocks at the end. The
network bind is **CLI-only** — `host`/`port`/`tls`/`tls_cert`/`tls_key` were
retired from `[server]` config (biopb/biopb#604), so *what to serve* is config
and *where to expose it* is the launch command.

**Prologue (both commands)**

1. Load `biopb.json`; set up logging (CLI > env > config).
2. Decide whether a token is enforced from the effective flight bind: a loopback
   bind runs tokenless; a public bind (`0.0.0.0`/`::`/a real IP) **requires** a
   token. `BIOPB_TENSOR_ALLOW_NO_TOKEN` overrides. Resolve it `--token` flag →
   `BIOPB_TENSOR_TOKEN` env var → auto-generate (`secrets.token_urlsafe(32)`, on
   a public bind only), and print it once. `launch` layers its own check on top:
   it refuses a public `--web-host` when the resolved token is `None`.
3. Resolve TLS material (`_resolve_tls_material`): a BYO `--tls-cert`/`--tls-key`
   pair read off disk, else — under `--tls` — the self-signed state-dir cert
   (auto-minted on first use), else plaintext.

**Flight server (`_setup_flight_server`)**

4. Initialize the chunk cache. The server refuses to start when the cache dir
   cannot be mmapped safely (network mount, cloud-synced folder) or isn't
   writable — the on-disk cache is required infrastructure, not optional.
5. Resolve config sources into *static* and *monitored* sets, and build the
   metadata DB (mandatory — it backs `query_sources`). An empty catalog is a
   valid state and boots: sources can still arrive via `add_source`, DoPut, or a
   monitored dir that fills later. The cache's measured per-tensor decode
   throughput is attached to the catalog here too, as a `decode_rates` table —
   it belongs to the config, not to the cache directory the operator may clear.
6. Construct `TensorFlightServer` (token, writable, TLS material) — built, not
   yet serving.
7. Build the watcher + `SourceManager`; wire the runtime `add_source` /
   `remove_source` handlers and the precache worker's commit hook **before**
   starting either, then start watcher, manager, and worker.
8. `mark_ready()` — health reports `SERVING` on a **possibly-still-populating**
   catalog; the bootstrap scan runs in the background (progressive discovery).
   Catalog freshness is carried by the health action's `full_scan_in_progress` /
   `last_full_scan_finished_at`, not by `SERVING`.

**Blocking call**

9. `serve` calls `flight_server.serve()` on the main thread.
   `launch` puts it in a **daemon thread** and blocks in `run_http_server(...)`
   (uvicorn) instead. The sidecar dials the flight plane over loopback —
   resolving a wildcard bind to the matching loopback family, `grpcs://` plus
   the served cert as `tls_ca_pem` under TLS. CORS defaults to the loopback
   variants of the sidecar's own address; `--cors` adds a browser app served
   elsewhere.

Both install the SIGTERM handler and the control deathwatch before blocking, so
the shutdown below runs on a supervised stop instead of being signal-killed.

Token validation rules: 16–128 characters, regex `[A-Za-z0-9_\-]+`.

### Shutdown sequence (`_graceful_shutdown`)

Runs from `launch`'s `finally`, once the blocking uvicorn call returns. A
`restart` force-kills the process after a bounded graceful window.

1. Stop the precache worker — no new warm writes.
2. Release the file-cache process lock **immediately**. Cheap
   and upstream-independent, so after this even a mid-teardown SIGKILL leaves no
   stale lock for the next boot to crash-recover. Segment writers/mmaps are left
   **open** — closing them here would race the in-flight `do_get` reads step 3
   has not drained yet — and releasing early is safe because index rebuild
   tolerates a torn tail.
3. Drain the Flight server, **bounded**. `FlightServerBase.shutdown()` takes no
   timeout and can block forever on a stream gated by a dead upstream, so it
   runs in a daemon thread joined with a short bound (3 s); on timeout, proceed
   and the OS reclaims the sockets.
4. Close the cache fully — writers and mmaps — but **only on a clean drain**,
   for proper finalization (which matters on Windows). Skipped if step 3 timed
   out, since a stuck `do_get` may still touch an mmap; the essential work
   already happened in step 2.
5. Stop the source manager (short 1 s join) and then the watcher — the watcher's
   own teardown sets its shutdown event, waits for a clean subprocess exit, then
   escalates `join` → `terminate` → `kill`. Both come last because neither
   touches the chunk cache and the lock is already gone, so a long join buys
   nothing and only risks spending the kill budget on a blocked upstream
   re-list.

The *trigger* for that shutdown differs by platform and owner — `SIGTERM` on
POSIX, a sentinel file on Windows where no catchable signal exists, plus a
lifetime binding for when the control dies uncatchably. All three belong to the
supervisor: see
[`../biopb-control/ARCHITECTURE.md`](../biopb-control/ARCHITECTURE.md).

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `BIOPB_TENSOR_TOKEN` | Pre-set server token for remote mode (else auto-generated). Doubles as the *client*-side token below. |
| `BIOPB_TENSOR_ALLOW_NO_TOKEN` | Truthy (`1`/`true`/`yes`/`on`) forces **tokenless** operation even on a public bind — the deliberate insecure escape hatch (trusted networks only). Only takes effect when no token is supplied; auto-generation and the public-sidecar refusal both become a loud warning instead. Off by default, so the fail-closed guarantee is unchanged unless explicitly set. |
| `BIOPB_UPSTREAM_TENSOR_TOKEN` | Bearer token for **one** upstream tensor server (`tensor-server` sources) — a single-upstream convenience. A source's credentials profile overrides it, and is the only way to give several upstreams different tokens or any TLS trust. |
| `BIOPB_LOG_LEVEL` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`; anything else is ignored and the CLI/default level wins. |
| `BIOPB_DATA_PLANE_SUPERVISED` | Set **by the control** on the child it spawns. The sidecar reports it and refuses self-restart, so a supervised restart is control-routed instead of racing the supervisor. |
| `BIOPB_OMETIFF_PARALLEL_READ` | Opt in (`=1`) to lock-free OME-TIFF chunk reads — concurrent tile decodes run in parallel instead of serializing under `_io_lock` (biopb/biopb#473). **Default off**. |
| `BIOPB_CLAIM_GENERIC_IMAGES` | Seeds the initial default for claiming generic raster/video during discovery (**off**, biopb/biopb#40). Only matters on discovery paths that never load a `ServerConfig`; a loaded config's `claim_generic_images` overrides it at startup. |
| `BIOPB_DISCOVERY_SKIP_OFFLINE` | `0` disables skipping suspected cloud placeholders during discovery (**on** by default) — an escape hatch for a filesystem that reports zero allocated blocks spuriously. |

Object-storage sources additionally honor the standard vendor credentials
(`AWS_*`, `AZURE_STORAGE_*`, `GOOGLE_APPLICATION_CREDENTIALS`) when a source
names no credentials profile.

---

## Security Model

- The FastAPI sidecar validates `Authorization: Bearer <token>` on every request
  via `HTTPBearer`.
- The Arrow Flight server validates the same token via
  `BearerAuthMiddlewareFactory`.
- **Local mode** (the default loopback `--host`) enforces no token — the 90%
  single-machine case. **Remote mode** (a public `--host`, which `biopb control
  start --grpc-bind` selects) requires a token, auto-generated if none is
  supplied.
- **The transport is plaintext unless TLS is asked for.** `serve`/`launch`
  default `--tls` off, so a local deployment is a token (or no token) over
  cleartext loopback. TLS is opt-in — the self-signed state-dir cert (`cert
  init`, TOFU-pinned by clients) or a BYO `--tls-cert`/`--tls-key` — and it
  needs the `[tls]` extra. Via `biopb control start` the **bind picks the
  default**: a public flight bind turns TLS on, loopback leaves it off, and an
  explicit `--no-tls` on a public bind is allowed but warns that the token and
  every pixel cross the network in the clear. The sidecar has no TLS of its own;
  it stays on loopback behind the control.
- **The HTTP sidecar bind (`--web-host`) is fail-closed too.** It has its own
  bind address, independent of `--host`, and re-exposes the whole data API. So
  `launch` **refuses to start** if the sidecar would bind a public address
  (`--web-host 0.0.0.0`/a real IP) while no token is enforced — the
  loopback-`--host` case, where the token resolves to `None`. "Public +
  unauthenticated" is unrepresentable on *either* listener, not just the flight
  server (`_resolve_launch_token`).
- **The one deliberate escape hatch is `BIOPB_TENSOR_ALLOW_NO_TOKEN`**
  (`_allow_no_token_from_env`). Truthy, it forces tokenless operation even on a
  public bind — auto-generation and the public-sidecar refusal both degrade to a
  loud warning. It only takes effect when no token is otherwise supplied, and is
  **off by default**, so the fail-closed guarantee above holds unless an
  operator explicitly opts out for a trusted network (the
  host-loopback-published Docker case, where the in-container bind is `0.0.0.0`
  but the ports are published to `127.0.0.1`). This is *not* the old auto
  dev-bypass (removed in #447) — it is explicit, per-deployment, and
  self-announcing.
- Error messages are redacted before logging/storage (filesystem paths and
  potential tokens replaced with `[REDACTED]`).

---

## A remote biopb-tensor-server as a source

A server that mirrors a remote server as a `tensor-server` source is a *client*
of that plane, and the local filesystem token handoff cannot reach across hosts
— so the upstream's credentials are explicit config, bound **per source** rather
than globally. The carrier is the existing credentials profile (`storage_type:
"biopb-tensor"`), which holds the bearer `token` and, for a `grpcs://` upstream,
optional TLS trust (`tls_fingerprint` / `tls_ca_file`; unset means TOFU).

One invariant to preserve: **the client pool keys on the credentials, not just
the endpoint** — two sources naming one `host:port` with different tokens or
anchors get different connections, and the SDK's TOFU memo is keyed the same
way. Otherwise whichever dialed first would silently decide what the other
authenticates as and trusts.

Full config surface and behavior:
**[docs/remote-tensor-cache.md](docs/remote-tensor-cache.md)**.

---

## Container shape (Flight-only by default)

By default, the published image is a **pure gRPC data-plane endpoint** without
HTTP endpoints. The http sidecar is still installed in the image and can be re-
enabled with `BIOPB_ENABLE_HTTP_SIDECAR=1`.

Browsing containerized data is a *downstream* concern: a machine running the
full stack adds `grpc://`/`grpcs://<host>:8815` as a remote source, and its
browser talks only to its own loopback control.

Deployment, TLS, cert persistence, and worked examples:
**[containerize.md](containerize.md)**.

---
