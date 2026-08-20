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

- **`core/`** — foundational primitives and contracts: adapter ABCs, `config`,
  the `axes` vocabulary + its `normalize` seam, and the low-level
  `source_registry` / `metadata_db` stores.
- **`serving/`** — the runtime: `server` (Arrow Flight), `http_server` (FastAPI
  sidecar), `upload_manager`, `precache`, `renderer`. Builds on `core`.
- **`sources/`** — source lifecycle: `source_manager` + `tree_scanner` +
  `watcher` (scan orchestration) and `reconciler` (the confirmed-catalog single
  writer). Builds on `core` and `serving`.
- **`adapters/`**, **`cache/`** — storage-format adapters and the virtual-chunk
  cache.

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
| `server.uploads` | `UploadManager` | The writable-server DoPut path: source creation (`cache:`/`ome_zarr:`), polymorphic chunk writes, and upload-progress state machine |

### Flight methods

| Method | Description |
|--------|-------------|
| `ListFlights` | Returns one `FlightInfo` per registered source, embedding a serialised `DataSourceDescriptor` proto. Lean: leaves `TensorDescriptor.pyramid` and `metadata_json` empty |
| `GetFlightInfo` | Returns query ticket for real data (pixel or metadata). Pixel data request respects `TensorReadOptions` and optionally fills `TensorDescriptor.pyramid` or `metadata_json` when asked |
| `DoGet` | Fetches data by ticket, either a single pixel chunk or metadata query results; returns a `RecordBatch` stream |

Custom `do_action` verbs extend these: `health`, `create_source`,
`upload_status`, `chunk_locate`, `cache_stats`, `resolve`, `warm`, `add_source`,
and `remove_source` (below).

#### Server-advertised pyramid (`TensorDescriptor.pyramid`)

`GetFlightInfo` fills `pyramid` with an ordered list of `PyramidLevel`
(`scale_hint`, `reduction_method`, logical `shape`, `native`).

Two sources of pyramid specs (`TensorFlightServer._advertised_pyramid`):

- **Native** — data formats that ship a real on-disk pyramid override
  `TensorAdapter.get_native_pyramid_levels()` (`OmeZarrAdapter` and
  `QptiffAdapter`) to return one `native=True`, `reduction_method="precompute"`
  level per on-disk resolution.
- **Computed** — everything else gets `chunk.build_pyramid_plan(...)`, a full
  pyramid (level 0 → coarsest) generated from the authoritative `[pyramid]`
  config knobs (`threshold` / `downscale_factor` / `pixel_budget_cubic_root`).
  The precache worker warms the *coarsest* of this same plan.

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

| Method | Returns |
|--------|---------|
| `list_tensor_descriptors()` | `list[TensorDescriptor]` — the source's tensors |
| `get_source_descriptor()` | `DataSourceDescriptor` proto |
| `get_tensor_descriptor()` | `TensorDescriptor` proto |
| `get_data(bounds)` | `np.ndarray` — decodes only the requested sub-region |
| `get_native_pyramid_levels()` | `list[PyramidLevel]` or `None` — native pyramid levels |

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
| **Not in scope** | Unlabeled stores (`zarr`, `hdf5`) emit `dimN`, so nothing is reordered and nothing is relabeled — promoting the consumers' positional *guess* to a wire *assertion* would be wrong for e.g. an unlabeled `[y, x, c]`. Give such a source semantics with `dim_labels` in its config. |
| **Fail-safe** | Ambiguity degrades to identity rather than moving pixels on a guess: rank mismatch, a duplicated canonical axis, or an `S` label that fails `samples_axis`' size-3/4 gate. Same posture `serving/renderer.py` already takes toward adapter-supplied labels. |
| **chunk_ids** | Untouched — minted by the wrapped adapter and opaque here, so versioned / scaled / precompute-level ids all pass through. What is permuted is the client-visible geometry (descriptor + endpoint `bounds`) and the pixels. |
| **Cache** | The transpose happens *before* the cache store, so a segment holds what the client is served and the localhost mmap fast path stays valid. `CACHE_FILE_FORMAT_VERSION` was bumped to `2` for that (same layout, reordered content); an older client declines the fast path and reads the same normalized chunk over `do_get`. |
| **Plans** | `plan_flight_info` / `get_read_plan` are delegated and their answer permuted, not re-derived — which is what keeps the native-pyramid `precompute` routing working underneath. |

An order this server does not own is **refused, not permuted** — permuting works
only where the server owns the whole read path, and two seams don't. Both report
through the shared `core/axes.py::noncanonical_order`:

| | |
|---|---|
| **Writes** | `create_source` rejects a non-canonical declared order up front, so a writable source never disagrees with what `put_chunk` wrote — `physical_scale` and `chunk_shape` arrive aligned to the uploader's labels. |
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

`CacheManager` provides a pluggable cache layer between `DoGet` and the adapter.
Two implementations:

- In-process LRU memory cache (`OrderedDict`-based, in
  `cache/memory_backend.py`).
- Persisted file cache transcoding chunk data to Flight IPC format
  (`cache/file_backend.py`).

The file cache is _strongly_ preferred: it serves a localhost client over an
mmap fast path, bypassing the round trip through a socket. See
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
`source_id` / `dim_labels` / `extra_config` / `is_remote`; `DiscoveryState`
holds the `source_id <-> path` maps and the `on_source_added` /
`on_source_removed` callbacks the `SourceManager` wires.

**Progressive discovery (biopb/biopb#212).** The CLI launcher reaches `SERVING`
ASAP and runs the monitored bootstrap scan in the background; the catalog grows
*within* that scan as each source is claimed (see Directory Monitoring below).
See **[docs/progressive-discovery.md](docs/progressive-discovery.md)**.

### Directory monitoring (`sources.watcher`, `sources.source_manager`)

`PeriodicRescanWatcher` emits a `RESCAN` on a fixed interval; per rescan the
`SourceManager` delegates the filesystem-signature walk to `TreeScanner` (a fs
walker gated on stability window, returning an immutable `ScanSnapshot`), runs
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

4. Initialize the chunk cache. A `file` backend degrades to memory when the
   cache dir cannot be mmapped safely (network mount, cloud-synced folder) or
   isn't writable — the localhost fast path goes with it, but the server serves.
5. Resolve config sources into *static* and *monitored* sets, and build the
   metadata DB (mandatory — it backs `query_sources`). An empty catalog is a
   valid state and boots: sources can still arrive via `add_source`, DoPut, or a
   monitored dir that fills later.
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
2. Release the file-cache process lock and clear the WAL **immediately**. Cheap
   and upstream-independent, so after this even a mid-teardown SIGKILL leaves no
   stale lock for the next boot to crash-recover. Segment writers/mmaps are left
   **open** — closing them here would race the in-flight `do_get` reads step 3
   has not drained yet — and an early WAL clear is safe because index rebuild
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
