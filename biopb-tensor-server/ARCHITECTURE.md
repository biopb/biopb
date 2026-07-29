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

- **`core/`** — foundational primitives and contracts: `adapter_base` (adapter
  ABCs), `config` / `config_schema`, `discovery`, `chunk`, `downsample`,
  `errors`, `remote`, `activity`, `logging_config`, and the low-level
  `source_registry` / `metadata_db` stores. Depends only on itself plus
  `adapters` / `cache`.
- **`serving/`** — the runtime: `server` (Arrow Flight), `http_server` (FastAPI
  sidecar), `upload_manager`, `precache`, `renderer`. Builds on `core`.
- **`sources/`** — source lifecycle: `source_manager` + `tree_scanner` +
  `watcher` (scan orchestration) and `reconciler` (the confirmed-catalog single
  writer). Builds on `core` and `serving`.
- **`adapters/`**, **`cache/`** — storage-format adapters and the virtual-chunk
  cache.

`cli`, `__main__`, `__init__` (the public-API re-exports), and `_version` stay
at the package root.

---

## TensorFlightServer

**Module:** `biopb_tensor_server.serving.server` **Class:**
`TensorFlightServer(flight.FlightServerBase)` **Default location:**
`grpc://0.0.0.0:8815`

`TensorFlightServer` is a thin Flight protocol handler; its mutable state lives
in three collaborators it composes (biopb/biopb#278 item A):

| Collaborator | Module | Owns |
|---|---|---|
| `server.sources` (`SourceRegistry`) | `source_registry.py` | the `source_id → SourceAdapter` map, the registration chokepoint (slash-free id validation), and adapter-lifecycle cleanup (close on unregister/shutdown) |
| `server.activity` (`ActivityTracker`) | `activity.py` | in-flight heavy-read counters + last-active stamp (the precache idle signal) and the warm-in-progress guard set. Fed by every heavy read — `do_get`, `warm`, **and `chunk_locate`**: the localhost fast path *replaces* `do_get`, so leaving it untracked made the server look idle for the whole of a localhost read (biopb/biopb#548) |
| `server.uploads` (`UploadManager`) | `upload_manager.py` | the writable-server DoPut path: source creation (`cache:`/`ome_zarr:`), polymorphic chunk writes, and the per-source upload-progress state machine |

### Flight methods

| Method | Description |
|--------|-------------|
| `ListFlights` | Returns one `FlightInfo` per registered source, embedding a serialised `DataSourceDescriptor` proto. Lean: leaves `TensorDescriptor.pyramid` and `metadata_json` empty |
| `GetFlightInfo` | Returns chunk endpoints or metadata query ticket. The former respects `TensorReadOptions` in FlightCmd and optionally fills `TensorDescriptor.pyramid` or `metadata_json` when requested |
| `DoGet` | Fetches a single chunk identified by a `TensorTicket` or metadata query results; returns a `RecordBatch` stream |

Custom `do_action` verbs extend these: `health`, `create_source`,
`upload_status`, `chunk_locate`, `cache_stats`, `resolve`, `warm`, `add_source`,
and `remove_source` (below).

#### Server-advertised pyramid (`TensorDescriptor.pyramid`)

`GetFlightInfo` fills `pyramid` with an ordered list of `PyramidLevel`
(`scale_hint`, `reduction_method`, logical `shape`, `native`); level 0 is full
resolution.

Two sources of levels (`TensorFlightServer._advertised_pyramid`):

- **Native** — adapters that ship a real on-disk pyramid override
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

All adapters implement:

| Method | Returns |
|--------|---------|
| `list_tensor_descriptors()` | `list[TensorDescriptor]` — the source's tensors |
| `get_source_descriptor()` | `DataSourceDescriptor` proto |
| `get_tensor_descriptor()` | `TensorDescriptor` proto |
| `get_data(bounds)` | `np.ndarray` — decodes only the requested sub-region |
| `get_native_pyramid_levels()` | `list[PyramidLevel]` or `None` — native on-disk levels (default `None`; `OmeZarrAdapter` overrides) |

### Adapter file-handle policy (biopb/biopb#71)

On Windows an opened/pinned file cannot be deleted, moved, or renamed, and on
POSIX an unlinked multi-GB volume frees no disk space. The default is therefore
**hold nothing between reads**; a persistent handle is opt-in and must be
justified by open cost.

| Open cost | Policy | Adapters |
|---|---|---|
| O(1) in file size (~0.05–0.1 ms, <0.3% of a 64 MB chunk read) | **reopen per read**, no handle, no `close()` needed | `hdf5`, `mrc`, `tiff`, `bioio`, `dicom`, local `zarr` |
| O(IFD count) or O(file count) — unbounded, never amortises | persistent handle + `close()`, and a shared idle reaper closes the handle between reads so the pin is *bounded*, not lifetime-long (TTL from `[server] handle_reaper_ttl`, default 150 s) | `ome-tiff`, `qptiff`, `ndtiff` |

---

## Chunk caching / transcoding

`CacheManager` provides a pluggable cache layer between `DoGet` and the adapter.
Two implementations:

- In-process LRU memory cache (`OrderedDict`-based, in
  `cache/memory_backend.py`).
- Persisted file cache transcoding chunk data to Flight IPC format
  (`cache/file_backend.py`).

The *client* half of this — the mmap fast path, its pinned-segment accounting,
and the two-tier chunk cache it feeds — is in
**[../docs/localhost-fast-path.md](../docs/localhost-fast-path.md)**.

---

## Discovery and Monitoring

### Discovery protocol (`core.discovery`)

Adapters **claim** filesystem paths they recognize (a `claim()` classmethod
each). `AdapterRegistry.get_claims_for_path` returns claims in **registration
order** (`adapters/__init__.py::get_default_registry`), highest-specificity
first. Notably,

- *OME-TIFF before TIFF-sequence* — OmeTiffAdapter *file*-claims an `.ome.tif`
  (consuming multi-file siblings via the OME-XML file list) while
  TiffSequenceAdapter *dir*-claims plain stacks and **excludes** OME-named
  files.
- *OME-Zarr before plain Zarr* — both can claim a `.zarr`, so the specific one
  must win.

A `SourceClaim` (`__slots__`) carries `source_type` / `primary_path` /
`source_id` / `dim_labels` / `extra_config` / `is_remote`; `DiscoveryState`
holds the `source_id <-> path` maps and the `on_source_added` /
`on_source_removed` callbacks the `SourceManager` wires.

**Progressive discovery (biopb/biopb#212).** `mark_ready()` / `SERVING` means
"up and serving the **possibly-still-populating** catalog," *not* "the data
folder scan finished." The CLI launcher reaches `SERVING` immediately and runs
the monitored bootstrap scan in the background (the watcher fires its first
rescan at once); the catalog grows *within* that scan as each source is claimed
(see Directory Monitoring below). See
**[docs/progressive-discovery.md](docs/progressive-discovery.md)**.

### Directory monitoring (`sources.watcher`, `sources.source_manager`)

`PeriodicRescanWatcher` emits a `RESCAN` on a fixed interval; per rescan the
`SourceManager` delegates the filesystem-signature walk to `TreeScanner` (a pure
producer that skips subtrees until they pass the stability window, returning an
immutable `ScanSnapshot`), runs discovery on the snapshot's stable paths, and
diffs the result against the confirmed catalog. Server mutations are
lock-serialized on the main process; reconciliation is snapshot-diff, not
per-file events. Only local directories can be monitored (`{ "url": ".../",
"monitor": true }`).

**Moves** within a monitored dir preserve `source_id`; a move out is a delete, a
move in a create.

### Cloud / synced-folder sources (`cloud = true`)

On a synced folder (OneDrive/Dropbox/iCloud "Files-On-Demand") content is
*dehydrated* until read, and reading one byte recalls the **whole** file — so
discovery **skips offline placeholders** by default. `cloud = true` opts one
root into the **phase-2** model:

Full model — the residency/recall rules, the resolve state machine, and the
"transcode monoliths to OME-Zarr at archive time" guidance — in
**[docs/cloud-storage-support.md](docs/cloud-storage-support.md)**.

### Runtime source registration (`add_source`)

The `add_source` Flight action registers an existing path on the **server's**
filesystem as a served source at runtime, without editing config or restarting —
the wire entrypoint behind the napari tensor-browser's drag-and-drop. It routes
the dropped path through the same claim → adapter → catalog pipeline the
directory watcher uses (`SourceManager.add_local_source`), so a dropped file or
dataset-dir registers one source and a plain folder is walked recursively and
may register several. The `TensorFlightServer` holds no `SourceManager`
reference, so the launcher injects the entrypoint via
`set_add_source_handler(...)`.

---

## CLI Launcher

**Command:** `biopb-tensor-server launch`

```
biopb-tensor-server launch --config biopb.json [--host 127.0.0.1] [--port 8815] [--writable] [--web-port 8816] [--web-host 127.0.0.1] [--cors ORIGIN]

# for grpc only (no web server) — same flight options + token handling as launch
biopb-tensor-server serve --config biopb.json [--host 127.0.0.1] [--port 8815] [--writable] [--tls] [--san NAME]

# generate / rotate the self-signed TLS cert and print its fingerprint
biopb-tensor-server cert init [--force] [--san NAME]
```

`serve` and `launch` share the Flight-server flags
(`--host`/`--port`/`--writable` / `--token`/`--log-level`/`--log-file`) and the
same fail-closed token resolution (`_resolve_flight_token`). `launch` adds the
HTTP sidecar (`--web-host`/`--web-port`/`--cors`) and layers the sidecar
fail-closed check on top (`_resolve_launch_token`).

### Startup sequence (`launch`):

1. Decide whether a token is enforced from the effective flight bind (`--host`
   `--host`, loopback by default): a loopback bind runs tokenless (**local
   mode**); a public bind (`0.0.0.0`/`::`/a real IP) **requires** a token
   (**remote mode**).
2. Resolve token: `--token` flag → `BIOPB_TENSOR_TOKEN` env var →
   `secrets.token_urlsafe(32)` auto-generated (public flight bind only; local
   mode uses no token). No interactive prompt. `launch` then refuses a public
   `--web-host` when the resolved token is `None`.
3. Print the one-time access token (remote mode only).
4. Load `biopb.json` config; instantiate adapters and register sources.
5. Resolve TLS material CLI-over-config (`_merge_tls_options` →
   `_resolve_tls_material`): a BYO cert/key pair read off disk, else the
   self-signed state-dir cert, else plaintext.
6. Start `TensorFlightServer` in a **daemon thread**, serving TLS when step 5
   produced a cert.
7. Build CORS origins: loopback variants of the sidecar's own address by default
   (no web app is bundled here), plus any explicit `--cors` origins for a
   browser app served elsewhere.
8. Call `run_http_server(...)` — **blocking** uvicorn call. Under TLS it gets a
   `grpcs://` flight location plus the served cert as `tls_ca_pem`. The sidecar
   is API-only; it serves no static assets (the control plane serves the browser
   UI).

Token validation rules: 16–128 characters, regex `[A-Za-z0-9_\-]+`.

### Shutdown sequence (`_graceful_shutdown`)

Runs from `launch`'s `finally`, once the blocking uvicorn call returns. **The
order is load-bearing** (biopb/biopb#300): a `restart` force-kills the process
after a bounded graceful window, so the cheap upstream-independent work must
happen *before* anything that can block on an unresponsive upstream. Each step
is isolated — a failure in one still lets the rest run.

1. Stop the precache worker — no new warm writes.
2. Release the file-cache process lock and clear the WAL, **immediately**. Cheap
   and upstream-independent, so after this even a mid-teardown SIGKILL leaves no
   stale lock for the next boot to crash-recover. Segment writers/mmaps are left
   **open** — closing them here would race the in-flight `do_get` reads step 3
   has not drained yet — and an early WAL clear is safe because index rebuild
   tolerates a torn tail.
3. Drain the Flight server, **bounded**. `FlightServerBase.shutdown()` takes no
   timeout and can block forever on a stream gated by a dead upstream, so it
   runs in a daemon thread joined with a short bound (3 s); on timeout, proceed
   — the process is exiting and the OS reclaims the sockets. Never
   `flight_server.wait()`.
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

## Per-upstream credentials (mounting a remote plane)

A server that mirrors a remote plane as a `tensor-server` source is a *client*
of that plane, and the local filesystem token handoff cannot reach across hosts
— so the upstream's credentials are explicit config, bound **per source** rather
than globally, since one server may mount several upstreams owned by different
groups. The carrier is the existing credentials profile (`storage_type:
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

The published image is a **pure gRPC data-plane endpoint**: `entrypoint.sh` runs
`biopb-tensor-server serve`, so the container has **one** listener (Flight 8815)
and no HTTP surface at all — no browser origin, no CORS, no unlock page
(biopb/biopb#604 item 3). The sidecar is still installed and returns with
`BIOPB_ENABLE_HTTP_SIDECAR=1`; only the *default* changed.

Browsing containerized data is a *downstream* concern: a machine running the
full stack adds `grpc://`/`grpcs://<host>:8815` as a remote source, and its
browser talks only to its own loopback control. That is what makes remote TLS
cheap — no browser ever has to trust the container's cert.

Both opt-in extras are installed in the image. `[tls]` is opt-in on PyPI only
because of a missing Intel-macOS `cryptography` wheel (biopb/biopb#355) — a
source-install problem this linux image never has.

Deployment, TLS, cert persistence, and worked examples:
**[containerize.md](containerize.md)**.

---
