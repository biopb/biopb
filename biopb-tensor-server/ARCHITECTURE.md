# biopb-tensor-server Architecture

## Overview

`biopb-tensor-server` provides two complementary server components:

1. **TensorFlightServer** — Arrow Flight / gRPC server for chunked array access (port 8815).
2. **FastAPI HTTP Server** — Browser-accessible HTTP API for the data plane (port 8814). **API-only** — the browser UI is served by the control plane (the single web origin); see the top-level `web/` workspace and `web/README.md`.

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

The FastAPI server exposes the data-plane HTTP API. It wraps the Python `TensorFlightClient` and re-exposes its operations as plain HTTP so that browsers can use it without a gRPC-Web proxy. It does **not** serve the browser UI — the control plane owns that, as the single web origin, and reverse-proxies this sidecar under `/data_plane/*` (see `web/README.md`).

### Package layout

The `biopb_tensor_server` package is organized into layered subpackages:

- **`core/`** — foundational primitives and contracts: `base` (adapter ABCs), `config` / `config_schema`, `discovery`, `chunk`, `downsample`, `errors`, `remote`, `activity`, `logging_config`, and the low-level `source_registry` / `metadata_db` stores. Depends only on itself plus `adapters` / `cache`.
- **`serving/`** — the runtime: `server` (Arrow Flight), `http_server` (FastAPI sidecar), `upload_manager`, `precache`, `renderer`. Builds on `core`.
- **`sources/`** — source lifecycle: `source_manager` + `tree_scanner` + `watcher` (scan orchestration) and `reconciler` (the confirmed-catalog single writer). Builds on `core` and `serving`.
- **`adapters/`**, **`cache/`** — storage-format adapters and the virtual-chunk cache.

`cli`, `__main__`, `__init__` (the public-API re-exports), and `_version` stay at the package root.

---

## TensorFlightServer

**Module:** `biopb_tensor_server.serving.server`
**Class:** `TensorFlightServer(flight.FlightServerBase)`
**Default location:** `grpc://0.0.0.0:8815`

`TensorFlightServer` is a thin Flight protocol handler; its mutable state lives
in three collaborators it composes (biopb/biopb#278 item A):

| Collaborator | Module | Owns |
|---|---|---|
| `server.sources` (`SourceRegistry`) | `source_registry.py` | the `source_id → SourceAdapter` map, the registration chokepoint (slash-free id validation), and adapter-lifecycle cleanup (close on unregister/shutdown) |
| `server.activity` (`ActivityTracker`) | `activity.py` | in-flight heavy-read counters + last-active stamp (the precache idle signal) and the warm-in-progress guard set |
| `server.uploads` (`UploadManager`) | `upload_manager.py` | the writable-server DoPut path: source creation (`cache:`/`ome_zarr:`), polymorphic chunk writes, and the per-source upload-progress state machine |

`register_source` / `unregister_source` / `flight_idle_for` / `mark_ready`
remain on the server as thin delegators, so the CLI, source manager, and
precache worker drive it through the same public surface as before.

### Registration

```python
server = TensorFlightServer("grpc://0.0.0.0:8815")
server.register_source("my-zarr", ZarrAdapter(arr, "t0", ["z", "y", "x"]))
server.mark_ready()  # health reports SERVING (else STARTING forever)
server.serve()  # blocking
```

The `biopb-tensor-server` CLI launcher is the authoritative entry point. Code
that drives `TensorFlightServer` directly (as above) is responsible for calling
`mark_ready()` itself once it is ready to serve.

**Progressive discovery (biopb/biopb#212).** `mark_ready()` / `SERVING` means
"up and serving the **possibly-still-populating** catalog," *not* "the data
folder scan finished." The CLI launcher reaches `SERVING` immediately and runs
the monitored bootstrap scan in the background (the watcher fires its first
rescan at once); the catalog grows *within* that scan as each source is claimed
(see Directory Monitoring below). Catalog *freshness* is therefore a separate
signal carried by two `health` fields, not by `SERVING`:

- `full_scan_in_progress` (bool) — a full catalog rescan is running right now.
- `last_full_scan_finished_at` (epoch seconds, or `null` until the first full
  scan succeeds) — when the catalog was last fully reconciled. A periodic full
  rescan advances it, so boot and steady state share one mechanism.

A client that needs a complete catalog waits on these fields, not on `SERVING`;
a client that just needs "is the port up" still uses `SERVING`. (A static-only
config has nothing to scan, so the launcher stamps `last_full_scan_finished_at`
directly and `full_scan_in_progress` stays `false`.)

Sources are keyed by `source_id`. Each source maps to one adapter, which may
expose multiple tensors (e.g., multi-field).

### Flight methods

| Method | Description |
|--------|-------------|
| `ListFlights` | Returns one `FlightInfo` per registered source, embedding a serialised `DataSourceDescriptor` proto. Lean: leaves `TensorDescriptor.pyramid` and `metadata_json` empty |
| `GetFlightInfo` | Returns chunk endpoints for a specific tensor, respecting `SliceHint` and `TensorReadOptions` in the descriptor. Also fills `TensorDescriptor.pyramid` — the **server-advertised** resolution levels (see below) — and `metadata_json` when requested |
| `DoGet` | Fetches a single chunk identified by a `TensorTicket`; reads from the adapter and returns a `RecordBatch` stream |

Custom `do_action` verbs extend these: `health`, `create_source`,
`upload_status`, `chunk_locate`, `cache_stats`, `resolve`, `warm`,
`add_source`, and `remove_source` (below).

#### Runtime source registration (`add_source`)

The `add_source` Flight action registers an existing path on the **server's**
filesystem as a served source at runtime, without editing config or restarting —
the wire entrypoint behind the napari tensor-browser's drag-and-drop. It routes
the dropped path through the same claim → adapter → catalog pipeline the
directory watcher uses (`SourceManager.add_local_source`), so a dropped file or
dataset-dir registers one source and a plain folder is walked recursively and may
register several. The `TensorFlightServer` holds no `SourceManager` reference, so
the launcher injects the entrypoint via `set_add_source_handler(...)`.

- **Streaming.** A directory walk has no known size up front, so the action
  streams `AddSourceStreamMessage` (zero or more `AddSourceProgress` heartbeats —
  a running *count* of sources registered, not a percentage — then one terminal
  `AddSourceResult` carrying `added` / `already_present` / `failed(path, reason)`).
  The client can cancel by closing the stream; the walk stops but everything
  already registered stays (non-destructive).
- **Single-writer safety.** `add_local_source` runs inline on the Flight handler
  thread but under `SourceManager._catalog_lock`, which the periodic rescan also
  holds — so the two never mutate the confirmed catalog at once. Discovery runs
  into a scratch `DiscoveryState`; only committed claims touch the confirmed
  catalog.
- **Dedup & containment.** Re-dropping the exact same path is an upsert reported
  as `already_present` (deterministic `source_id`). Dropping a path **inside** an
  existing source is rejected (`_find_containing_source`, "already part of …") —
  the exact-member dedup in `DiscoveryState.add_claim` does not catch nesting
  because dir sources record only the directory as a member. Dropping a **parent**
  of existing sources re-discovers them (same id → `already_present`) and adds new
  siblings. The server does **not** gate a large directory walk — the
  large-folder footgun-stopper lives **client-side**: the tensor browser counts
  a dropped folder's entries (drag-drop is localhost-only, so the client shares
  this filesystem) and, above a coarse threshold, confirms with the user before
  sending the add. A direct SDK caller passing a path is explicit intent, so its
  walk is never gated.
- **Locality.** Runtime add is local-path only (a remote URL raises); the client
  gate additionally enables the drop UI only against a localhost server, since a
  dropped path is a client-side filesystem path.
- **Security.** The action is token-gated by the Flight auth middleware and, being
  a catalog mutation that exposes any server-readable path, is guarded by
  `TensorFlightServer._allow_runtime_source_add` (defaults **on**; a hardened
  read-only deployment can turn it off). It is *not* gated on write mode
  (`_writable`) — a normal read-only server still registers dropped local files.

#### Server-advertised pyramid (`TensorDescriptor.pyramid`)

The server decides the resolution pyramid, rather than the client computing one
from the tensor shape. `GetFlightInfo` fills `pyramid` with an ordered list of
`PyramidLevel` (`scale_hint`, `reduction_method`, logical `shape`, `native`);
level 0 is full resolution. The client reads each advertised level via the normal
`scale_hint` path. Two sources of levels (`TensorFlightServer._advertised_pyramid`):

- **Native** — adapters that ship a real on-disk pyramid override
  `TensorAdapter.get_native_pyramid_levels()` (`OmeZarrAdapter` and `QptiffAdapter`)
  to return one `native=True`, `reduction_method="precompute"` level per on-disk
  resolution, so the client requests the precomputed level directly. Each level's
  `scale_hint` is the value `_find_level_for_scale` matches on, so it round-trips.
  `QptiffAdapter` encodes each level's chunks with `array_id = source_id/{level}`,
  so `DoGet` dispatches back through `get_level_adapter` (the same seam OME-Zarr
  uses).
- **Computed** — everything else gets `chunk.build_pyramid_plan(...)`, a full
  pyramid (level 0 → coarsest) generated from the authoritative `[pyramid]` config
  knobs (`threshold` / `downscale_factor` / `pixel_budget_cubic_root`). The
  precache worker warms the *coarsest* of this same plan, so the warmed scale and
  the advertised scale can never drift.

### Adapter interface

Two role ABCs in `core/base.py`, and they **nest**: `TensorAdapter` subclasses
`SourceAdapter`, so a tensor adapter is a source that can also serve pixels
(biopb/biopb#380). Every concrete format adapter subclasses `TensorAdapter` and
fills both roles in one object — `get_tensor_adapter()` returns `self` for a
single-tensor format, and a clone of the same class (or a plain `ZarrAdapter`,
for OME-Zarr / QPTIFF levels and HCS fields) for a multi-tensor one. The lone
source-only adapter is `UnresolvedSourceAdapter`, which has no tensors until it
resolves. The role *scopes* stay disjoint where they are declared — `base.py`
asserts that at import time (`_SOURCE_SCOPED_API` / `_TENSOR_SCOPED_API`), so a
tensor-scoped method can never be written onto `SourceAdapter`.

All adapters implement:

| Method | Returns |
|--------|---------|
| `list_tensor_descriptors()` | `list[TensorDescriptor]` — the source's tensors |
| `get_source_descriptor()` | `DataSourceDescriptor` proto |
| `get_tensor_descriptor()` | `TensorDescriptor` proto |
| `get_data(bounds)` | `np.ndarray` — decodes only the requested sub-region |
| `get_native_pyramid_levels()` | `list[PyramidLevel]` or `None` — native on-disk levels (default `None`; `OmeZarrAdapter` overrides) |

Concrete adapters:

| Adapter | Format |
|---------|--------|
| `ZarrAdapter` | Zarr v2 arrays |
| `OmeZarrAdapter` | OME-Zarr with precomputed pyramid routing |
| `OmeTiffAdapter` | OME-TIFF (single- and multi-file), pure-tifffile — no aicsimageio |
| `QptiffAdapter` | Akoya PhenoImager QPTIFF (claimed by the `.qptiff` extension; a `.tif`-named QPTIFF needs an explicit `type: qptiff`) — pyramidal multiplex whole-slide via tifffile, serving the native on-disk pyramid as `precompute` levels (2nd native-pyramid adapter after OME-Zarr). Module: `adapters/qptiff.py` |
| `TiffSequenceAdapter` | Plain TIFF stacks (directory of non-OME `.tif`) |
| `Hdf5Adapter` | HDF5 chunked datasets |
| `MrcAdapter` | MRC electron-microscopy volumes (`.mrc/.mrcs/.rec/.st/.map`) — header parsed by rosettasciio, reads served from an own per-read `np.memmap`. Module: `adapters/mrc.py` |
| `EmdAdapter` | EMD electron-microscopy datasets (`.emd`, NCEM + Velox) via rosettasciio; multi-signal → multi-tensor, native HDF5 chunk grid. Module: `adapters/emd.py` |
| `AicsImageIoAdapter` (+ `Zeiss`/`Leica`/`Nikon`/`Dv`/`Olympus`/`Bioformats` subclasses) | Vendor formats (CZI, LIF, ND2, DV, …) and remote/non-OME `.tif` via bioio (successor to aicsimageio; per-format `bioio-*` plugins). Module: `adapters/bioio.py` |

### Adapter file-handle policy (biopb/biopb#71)

A source stays catalogued for as long as the server runs, so an adapter that
opens its file at registration pins it *continuously* — not just until shutdown.
That is user-visible: on Windows the pinned file cannot be deleted, moved, or
renamed (and deletion is what would have released it, so nothing ever does), and
on POSIX an unlinked multi-GB volume frees no disk space. The default is
therefore **hold nothing between reads**; a persistent handle is opt-in and must
be justified by open cost.

| Open cost | Policy | Adapters |
|---|---|---|
| O(1) in file size (~0.05–0.1 ms, <0.3% of a 64 MB chunk read) | **reopen per read**, no handle, no `close()` needed | `hdf5`, `mrc`, `tiff`, `bioio`, `dicom`, local `zarr` |
| O(IFD count) or O(file count) — unbounded, never amortises | persistent handle + `close()`; `ome-tiff` additionally reaps an idle store (`BIOPB_TIFF_STORE_TTL`) | `ome-tiff`, `qptiff`, `ndtiff` |

`close()` is **declared on `SourceAdapter`** with a concrete no-op default (and
classified in `_SOURCE_SCOPED_API`, so adding it had to be a deliberate interface
decision) — the same shape as `CacheBackend.release_process_lock`, and for the
same reason `put_chunk` is declared rather than sniffed: an optional capability
the registry drives on *every* adapter belongs in the interface. `SourceRegistry`
calls it directly on `unregister` / `close_all`. Second-row adapters override it
(plus a `__del__` backstop, refs nulled before the underlying close, safe to call
twice). `UnresolvedSourceAdapter` forwards it to the adapter it resolved to —
that forward was the omitted seventh of seven delegated methods, and a duck-typed
hook could not see the omission.

### Chunk caching

`CacheManager` provides a pluggable cache layer between `DoGet` and the
adapter. The default backend is an in-process LRU memory cache
(`OrderedDict`-based, in `cache/memory_backend.py`).
An optional `ArrowFileBackend` persists decoded chunks to disk.

**Sidecar boot index (biopb/biopb#300).** Each sealed segment `seg_NNNN.arrow`
gets a `seg_NNNN.idx` sidecar written at seal time (natural rotation and
graceful close) recording every entry's key -> byte range. Boot restores the
index from these small files instead of faulting the whole on-disk cache
(previously a full body walk — tens of GB, ~52-78 s on a caching-proxy cache).
Because a sealed segment is immutable, the sidecar needs no manifest or
generation counter: a boot trusts one iff its recorded `.arrow` size matches the
file on disk, and otherwise falls back to the body walk (which backfills a fresh
sidecar, so the first boot after upgrading an old cache pays the walk once). The
sidecar is purely additive — an older server ignores `.idx` (it globs `.arrow`)
— and its tiny bytes are not counted toward the eviction budget.

**Byte ranges are recorded at write time (biopb/biopb#541).** `complete_entry`
brackets each appended message with the sink cursor, so the localhost
`chunk_locate` fast path finds every entry already indexed. The one special case
is a segment's first append, which also flushes the writer's buffered schema
message: its start is recovered by reading that message's length off the file.
Together with the sidecar above, nothing on a normal path leaves an entry
unindexed — the lazy `_fill_byte_offsets_for_segment` walk survives only as a
fallback, because it costs O(entries in the segment) per call (measured ~5 ms at
145 MB with 0.87 MB chunks; it scales with entry *count*, so a 128 KB-chunk
source pays ~12 ms at the same 128 MB) and used to be paid on **every miss**.

---

## FastAPI HTTP Server

An **API-only** FastAPI sidecar (`biopb_tensor_server.serving.http_server`, factory
`create_app(...)`, **port 8814**) co-located with the Flight server: it wraps the
Python `TensorFlightClient` and re-exposes it as HTTP/JSON (+ binary slices) so
browsers reach the data plane without a gRPC-Web proxy. It serves **no** static
assets — the control plane owns the browser UI and reverse-proxies this sidecar
under `/data_plane/*`.

Auth mirrors the Flight server: `Authorization: Bearer <token>` / `X-Biopb-Token`,
timing-safe compared; a `None` token is **local mode** (loopback, no enforcement),
a token is **remote mode** (public `server.host`). The `TensorFlightClient` opens
lazily on the first authenticated request; a thread-safe `_DiagnosticsState` tracks
latency / errors / cache hit-rate / per-session rate-limit, with every error string
`_redact()`ed (filesystem paths and token-like strings -> `[REDACTED]`).

See **[docs/http-server.md](docs/http-server.md)** for the full endpoint table, the
`/api/slice` request/response contract, the diagnostics fields, and CORS defaults.

---

## CLI Launcher

**Command:** `biopb-tensor-server launch`

```
biopb-tensor-server launch --config biopb.json [--web-port 8814] [--web-host 127.0.0.1] [--open] [--web-url URL] [--cors ORIGIN]

# for grpc only (no web server)
biopb-tensor-server serve ...
```

Startup sequence:

1. Decide whether a token is enforced from the config's `server.host`: a
   loopback `server.host` runs tokenless (**local mode**); a public
   `server.host` (`0.0.0.0`/`::`/a real IP) **requires** a token (**remote
   mode**).
2. Resolve token: `--token` flag → `BIOPB_TENSOR_TOKEN` env var →
   `secrets.token_urlsafe(32)` auto-generated (public `server.host` only; local
   mode uses no token). No interactive prompt.
3. Print the one-time access token (remote mode only).
4. Load `biopb.json` config; instantiate adapters and register sources.
5. Start `TensorFlightServer` in a **daemon thread**.
6. Derive CORS origins from `--web-url` (default `http://localhost:5173`) or
   explicit `--cors` flags; optionally schedule `webbrowser.open(--web-url)`.
7. Call `run_http_server(...)` — **blocking** uvicorn call. The sidecar is
   API-only; it serves no static assets (the control plane serves the browser UI).

Token validation rules: 16–128 characters, regex `[A-Za-z0-9_\-]+`.

### Windows graceful shutdown of the supervised data plane

The control plane owns the data-plane process: it spawns `launch` and stops it on
teardown (`DataPlaneSupervisor._terminate`). On Windows that child runs with
`CREATE_NO_WINDOW` and `os.kill` is an uncatchable
`TerminateProcess`, so there is no way to deliver a catchable stop for a graceful
exit — and Win32 named events proved brittle across sessions/elevation. So
graceful shutdown is coordinated through a **sentinel file**: `run_http_server`
calls `_install_windows_shutdown_listener(server)`, which starts a daemon thread
that polls for `~/.local/state/biopb/tensor-server.stop`. When the supervisor
writes that file, the thread sets `server.should_exit = True` *and*
`server.force_exit = True` (so an open browser/keep-alive connection can't stall
shutdown). uvicorn returns from `run()`, so `launch`'s `finally →
_graceful_shutdown` runs and the file-cache process lock is released. The
supervisor then hard-kills (`TerminateProcess`) as a backstop if the child hasn't
exited within its timeout; on POSIX it sends `SIGTERM` instead. See `biopb/biopb#22`.

The sentinel name is **fixed, not PID-keyed**: on Windows the process the
supervisor records can differ from the one actually running `launch()`/uvicorn
(Store-Python/uv launcher shims), so a PID in the name would make writer and
watcher disagree. A leftover sentinel from a prior run is cleared once at
listener startup. Because the control is the **sole owner** of the plane, the
supervisor is the only writer of this sentinel — the former standalone `biopb
server` daemon that also wrote it has been retired.

A directly-launched `biopb-tensor-server launch`/`serve` (not under the control)
is **self-managed** — you stop it with Ctrl+C / your own process control. It still
installs the same watcher on the same fixed sentinel, so running a
control-supervised plane and a direct `launch` side by side on Windows is
unsupported (they would share the one sentinel). On POSIX there is no such
coupling — the supervisor signals one PID.

### Dying with the control (the uncatchable-death backstop)

The sentinel (Windows) and `SIGTERM` (POSIX) above are the *graceful* stop — they
run while the control is alive to ask for it. A separate bind covers the control
dying **uncatchably** (SIGKILL, OOM, crash, a session logout), where no graceful
signal is ever sent: the plane must still die, or it orphans onto the gRPC port
and the next control start refuses it as a *conflict* (a wedged restart / install).
So the supervisor ties the plane's lifetime to its own (Pattern O, shared with
biopb-mcp; the primitives live in `biopb._lifecycle`):

- **POSIX** — the child inherits a **parent-death pipe** and runs in its own
  session (`start_new_session`). `launch`/`serve` call `deathwatch.install()`,
  which blocks a thread on that pipe; when the control process dies the pipe EOFs
  and the plane group-kills itself (only its own session, so the reap is
  contained). A standalone launch inherits no pipe, so `install()` is a no-op.
- **Windows** — the child is assigned to a **kill-on-close Job Object** the
  control holds the only handle to (`_assign_to_job`); when the control exits for
  any reason the OS empties the job, reaping the plane and everything it spawned.

The bind is orthogonal to the sentinel/SIGTERM path: a graceful `control stop`
still runs the plane's orderly `_graceful_shutdown` (releasing the cache lock); the
bind only fires when the control is gone before it could.

---

## Discovery & Directory Monitoring

### Discovery protocol (`core.discovery`)

Adapters **claim** filesystem paths they recognize (a `claim()` classmethod each).
`AdapterRegistry.get_claims_for_path` returns claims in **registration order** and
callers take `claims[0]`, so **order = priority**
(`adapters/__init__.py::get_default_registry`), highest-specificity first:

1. `OmeTiffAdapter` — local OME-TIFF w/ embedded OME-XML (single- + multi-file),
   pure-tifffile · `QptiffAdapter` before the bioio group so it owns `.qptiff`
   (suffix-only; a `.tif`-named QPTIFF needs an explicit `type: qptiff`, #135)
2. `ZeissAdapter` / `LeicaAdapter` / `NikonAdapter` / `DvAdapter` /
   `OlympusAdapter` / `BioformatsAdapter` / `AicsImageIoAdapter` — vendor formats +
   the generic bioio fallback (also picks up remote / non-OME `.tif`)
3. `OmeZarrAdapter` (+ HCS)  ->  4. `ZarrAdapter`
5. `NdTiffAdapter` / `MicroManagerLegacyAdapter` / `TiffSequenceAdapter`
6. `DicomSeriesAdapter` / `DicomAdapter` / `NiftiAdapter`
7. `Hdf5Adapter` (explicit `hdf5` only) · 8. `RemoteTensorAdapter` (explicit
   `tensor-server`, never claims a path)

Optional bioio/ndtiff/dicom/nifti adapters register only when their dependency is
importable, so a slimmer install collapses the list without reordering the rest.

**Two orderings are load-bearing:**

- *OME-TIFF before TIFF-sequence* — OmeTiffAdapter *file*-claims an `.ome.tif`
  (consuming multi-file siblings via the OME-XML file list) while
  TiffSequenceAdapter *dir*-claims plain stacks and **excludes** OME-named files,
  so an `.ome.tif` becomes its own source rather than being welded into a sequence.
  OmeTiffAdapter declines a non-OME / remote `.tif`, which then falls through to
  bioio or the sequence adapter.
- *OME-Zarr before plain Zarr* — both can claim a `.zarr`, so the specific one must
  win. They stay disjoint once resident (OmeZarr declines a non-multiscales store,
  Zarr declines a real OME-Zarr, and OmeZarr declines early when a top-level
  `.zarray`/`zarr.json` exists), so the resident re-claim lands `claims[0]` on the
  right type even after a blind provisional guess (e.g. at cloud resolve).

A `SourceClaim` (`__slots__`) carries `source_type` / `primary_path` / `source_id`
/ `dim_labels` / `extra_config` / `is_remote`; `DiscoveryState` holds the
`source_id <-> path` maps and the `on_source_added` / `on_source_removed` callbacks
the `SourceManager` wires.

### Directory monitoring (`sources.watcher`, `sources.source_manager`)

`PeriodicRescanWatcher` emits a `RESCAN` on a fixed interval; per rescan the
`SourceManager` delegates the filesystem-signature walk to `TreeScanner` (a pure
producer that skips subtrees until they pass the stability window, returning an
immutable `ScanSnapshot`), runs discovery on the snapshot's stable paths, and diffs
the result against the confirmed catalog. Server mutations are lock-serialized on
the main process; reconciliation is snapshot-diff, not per-file events. Only local
directories can be monitored (`{ "url": ".../", "monitor": true }`).

**Startup is progressive (#212):** the launcher reaches `SERVING` immediately and
the first full scan runs in the background, **streaming** each source into the
catalog as it is claimed (so `health.source_count` grows during the scan);
`full_scan_in_progress` / `last_full_scan_finished_at` carry catalog freshness, not
`SERVING`. Full treatment in
**[docs/progressive-discovery.md](docs/progressive-discovery.md)**.

**Moves** within a monitored dir preserve `source_id`; a move out is a delete, a
move in a create. **Shutdown:** `source_manager.stop()` then `watcher.stop()` (->
`shutdown_event.set()` -> clean subprocess exit -> `join(5)` -> `terminate` ->
`kill`).

### Cloud / synced-folder sources (`cloud = true`)

On a synced folder (OneDrive/Dropbox/iCloud "Files-On-Demand") content is
*dehydrated* until read, and reading one byte recalls the **whole** file — so
discovery **skips offline placeholders** by default. `cloud = true` opts one root
into the **phase-2** model:

- **admit** placeholders (not skip), keeping every `claim()` **recall-free** —
  single-source formats (Zarr/OME-Zarr, MicroManager, single DICOM) defer as a
  provisional `unresolved=True` claim behind `UnresolvedSourceAdapter`, while
  content-membership formats (multi-file OME-TIFF, DICOM **series**) cannot be
  deferred and degrade to **N single-file sources**;
- **resolve on first serve** — the first `GetFlightInfo` re-claims the now-resident
  path and backfills the catalog (the recorded type was a recall-free guess);
- cloud subtrees are walked only on a `force_full` rescan, with the stability window
  + open-for-append probe **bypassed** (the probe would recall the file), and
  precache never touches an unresolved source.

Full model — the residency/recall rules, the resolve state machine, and the
"transcode monoliths to OME-Zarr at archive time" guidance — in
**[docs/cloud-storage-support.md](docs/cloud-storage-support.md)**.

---

## Configuration (`biopb.json`)

```json
{
  "server": { "host": "0.0.0.0", "port": 8815 },
  "cache": { "max_bytes": 2000000000 },
  "pyramid": {
    "threshold": 4096,
    "downscale_factor": 4,
    "pixel_budget_cubic_root": 512,
    "reduction_method": "area"
  },
  "sources": [
    { "url": "/data/" },
    {
      "source_id": "my-zarr",
      "type": "zarr",
      "url": "/data/experiment.zarr",
      "dim_labels": ["z", "y", "x"]
    },
    {
      "source_id": "ome",
      "type": "ome-zarr",
      "url": "/data/multiscale.zarr"
    }
  ]
}
```

Notes: `cache.max_bytes` is the in-process limit (2 GB above); `[pyramid]` is the
authoritative resolution-level definition (`threshold` = max X/Y extent of the
coarsest level, `downscale_factor` = per-level step, `pixel_budget_cubic_root`
= coarsest-level voxel budget cubed, `reduction_method` = on-the-fly
downsampling for computed levels); the bare `/data/` source triggers recursive
discovery.

---

## Browser front end

The TypeScript data-plane SDK (`@biopb/tensor-flight-client`) and the React SPA
(`@biopb/web`) live in the top-level `web/` workspace and are **not** part of the
tensor server — the sidecar is API-only, and the control plane serves the SPA. See
`../web/README.md` for the workspace layout, routes, and build, and
`../web/ARCHITECTURE.md` for the front-end internals (the HTTP client, lazy
`TensorArray`, axis mapping, `computeScaleHint`, the token/store model, and the
slice-render data flow).

---

## Test Suite

### Server tests

**Location:** `biopb-tensor-server/tests/`
**Runner:** pytest

| File | Scope | Count |
|------|-------|-------|
| `adapter_unit_test.py` | ZarrAdapter, OmeZarrAdapter, config parsing | ~20 |
| `adapter_integration_test.py` | Full server → client → dask compute per adapter | ~15 |
| `cache_test.py` | CacheManager, memory backend, file backend | ~10 |
| `multifield_test.py` | Multi-field / multi-position dataset handling | ~8 |
| `tensor_extended_test.py` | Scale routing, runtime downsampling | ~10 |
| `http_server_test.py` | FastAPI sidecar: auth, health, sources, slice, diagnostics, redaction, rate limit, integration | 37 |

`http_server_test.py` uses FastAPI `TestClient` (backed by `httpx`) with a
`unittest.mock.MagicMock` replacing `TensorFlightClient` for unit tests, and
a real `TensorFlightServer` + `ZarrAdapter` for the `TestIntegration` class.

---

## Environment Variables

| Variable | Where consumed | Purpose |
|----------|---------------|---------|
| `BIOPB_TENSOR_ENDPOINT` | TensorFlightClient (Python) | Arrow Flight server location (default `grpc://localhost:8815`) |
| `BIOPB_TENSOR_TOKEN` | `biopb-tensor-server launch` (server) | Pre-set server token for remote mode (else auto-generated) |
| `BIOPB_BIND_LOCALHOST` | Docker/Singularity entrypoint | Bind both HTTP and gRPC to loopback → local mode / no token (Singularity/HPC only; ignored in Docker) |
| `BIOPB_OMETIFF_PARALLEL_READ` | `OmeTiffAdapter.get_data` | Opt in (`=1`) to lock-free OME-TIFF chunk reads — concurrent tile decodes run in parallel instead of serializing under `_io_lock` (biopb/biopb#473). **Default off**: reads decode under the lock, as before. |
| `BIOPB_TIFF_STORE_TTL` | OME-TIFF store reaper | Seconds an idle OME-TIFF `aszarr` store handle is kept warm before the reaper closes it (default `300`). |

---

## Security Model

- Token is stored in `sessionStorage` (clears on tab close, never persisted to disk).
- The FastAPI sidecar validates `Authorization: Bearer <token>` on every request via `HTTPBearer`.
- The Arrow Flight server validates the same token via `BearerAuthMiddlewareFactory`.
- **Local mode** (loopback `server.host`) enforces no token — the 90% single-machine case. **Remote mode** (public `server.host`) requires a token, auto-generated if none is supplied.
- **The HTTP sidecar bind (`--web-host`) is fail-closed too.** It has its own bind address, independent of `server.host`, and re-exposes the whole data API. So `launch` **refuses to start** if the sidecar would bind a public address (`--web-host 0.0.0.0`/a real IP) while no token is enforced — the loopback-`server.host` case, where the token resolves to `None`. "Public + unauthenticated" is unrepresentable on *either* listener, not just the flight server (`_resolve_launch_token`).
- For Docker local mode with localhost-only access, use `-p 127.0.0.1:8814:8814 -p 127.0.0.1:8815:8815`.
- For Singularity/HPC local mode with localhost-only binding, use `BIOPB_BIND_LOCALHOST=true`.
- Error messages are redacted before logging/storage (filesystem paths and potential tokens replaced with `[REDACTED]`).

---

## Versioning

The tensor server has its **own version line**, keyed to the per-package tag
`server-v*`. Its Docker image is cut on that tag by `tensor-server-ci`, on its own
cadence — distinct from the SDK line (`v*`, for `biopb` + `biopb-image-base`) and
the product bundle line (`release-v*`, for mcp/control/web + the GitHub release).
See `../docs/release-model.md`.

```
git tags (server-vX.Y.Z)  →  setuptools_scm  →  biopb_tensor_server/_version.py
```

Version is derived via `setuptools_scm` with `tag_regex = "^server-v..."` (and a
matching `git describe --match 'server-v*'`). The web JS packages instead track the
product `release-v*` tag (`web/scripts/sync-version.js`), not this one.

**Docker image** (own cadence): `git tag server-v0.11.0 && git push --tags`.
`tensor-server-ci`'s `publish` job then builds and pushes
`biopb-tensor-server:0.11.0` (+ `:latest` for a clean X.Y.Z) to ghcr.io + Docker
Hub. The **wheel** still ships in the `release-v*` GitHub bundle (versioned off
this `server-v*` line), and the SDK (incl. image-base) releases on `v*`.
