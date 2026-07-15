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
                                              │  BackendAdapter          │
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

Sources are keyed by `source_id`. Each source maps to one `BackendAdapter`
which may expose multiple tensors (e.g., multi-field).

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

### BackendAdapter interface

All adapters implement `BackendAdapter`:

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
| `MrcAdapter` | MRC electron-microscopy volumes (`.mrc/.mrcs/.rec/.st/.map`) — header parsed by rosettasciio, reads served from an own `np.memmap`. Module: `adapters/mrc.py` |
| `EmdAdapter` | EMD electron-microscopy datasets (`.emd`, NCEM + Velox) via rosettasciio; multi-signal → multi-tensor, native HDF5 chunk grid. Module: `adapters/emd.py` |
| `AicsImageIoAdapter` (+ `Zeiss`/`Leica`/`Nikon`/`Dv`/`Olympus`/`Bioformats` subclasses) | Vendor formats (CZI, LIF, ND2, DV, …) and remote/non-OME `.tif` via bioio (successor to aicsimageio; per-format `bioio-*` plugins). Module: `adapters/bioio.py` |

### Chunk caching

`CacheManager` provides a pluggable cache layer between `DoGet` and the
adapter. The default backend is an in-process LRU memory cache
(`OrderedDict`-based, in `cache/memory_backend.py`).
An optional `ArrowFileBackend` persists decoded chunks to disk.

---

## FastAPI HTTP Server

**Module:** `biopb_tensor_server.serving.http_server`
**Factory:** `create_app(flight_location, token, cache_bytes, cors_origins, config_path, web_host, web_port, supervised) → FastAPI`
**Default port:** `8814`

### Lifecycle

The app holds two pieces of shared mutable state created at factory time:

- **`_client_holder`** — lazily initialised `TensorFlightClient`; the first
  authenticated request that reaches any protected endpoint triggers the gRPC
  connection to `flight_location`.
- **`_DiagnosticsState`** — thread-safe container for latency samples, error
  events, cache counters, and per-session rate-limit state.

### Authentication

Two equivalent header schemes are accepted on every protected endpoint:

```
Authorization: Bearer <token>
X-Biopb-Token: <token>
```

`secrets.compare_digest` is used for timing-safe comparison. The auth check
compares against `expected = self.token`; a `None` token means no enforcement —
this is **local mode**, where every listener binds loopback and no token exists.
A token is present (and enforced) only in **remote mode**, when the config's
`server.host` is a public address. There is no separate dev flag.

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/livez` | ✗ | Liveness probe — `{"status":"ok","timestamp":"…"}` |
| `GET` | `/readyz` | ✗ | Readiness — adds `ready`, `dev_mode`, `service`, `version` |
| `GET` | `/healthz` | ✗ | Alias for `/readyz` |
| `GET` | `/api/diagnostics` | ✓ | Diagnostics snapshot; rate-limited 1 req/s per session |
| `GET` | `/api/sources` | ✓ | JSON array of `DataSourceDescriptor` objects |
| `GET` | `/api/sources/{id}` | ✓ | Single descriptor |
| `GET` | `/api/sources/{id}/metadata` | ✓ | Parsed `metadata_json` field |
| `POST` | `/api/sources/query` | ✓ | Server-side DuckDB SQL over the catalog |
| `GET` | `/api/sources/{id}/ticket/{ticket_hex}` | ✓ | Resolve a Flight ticket to bytes |
| `POST` | `/api/slice` | ✓ | Binary tensor sub-region |
| `POST` | `/api/render` | ✓ | Server-rendered RGB image of a slice |
| `GET` | `/api/config` | ✓ | Current config (secrets redacted) |
| `PUT` | `/api/config` | ✓ | Update config (same-origin guarded) |
| `GET` | `/api/admin/status` | ✓ | Server/catalog status for the admin page |
| `GET` | `/api/admin/browse` | ✓ | Filesystem browse for the data-folder picker |
| `POST` | `/api/admin/restart` | ✓ | Restart the server (same-origin guarded) |

> **Route ordering:** `/api/sources/{id}/metadata` is registered *before* the
> greedy `{id:path}` catch-all to avoid Starlette first-match shadowing.

### Slice endpoint

**Request body** (`SliceRequest` Pydantic model):

```json
{
  "source_id":        "my-zarr",
  "tensor_id":        "0",
  "slice_start":      [0, 0, 0],
  "slice_stop":       [1, 512, 512],
  "scale_hint":       [1, 2, 2],
  "reduction_method": "area",
  "pixel_budget":     1000000
}
```

**Response:**
- `Content-Type: application/octet-stream` — C-contiguous `numpy.tobytes()`
- `X-Shape: 1,512,512`
- `X-Dtype: uint16`
- `X-Dim-Labels: z,y,x`

`scale_hint` and `reduction_method` are forwarded verbatim to
`TensorFlightClient.get_tensor(...)`, which resolves the appropriate
precomputed pyramid level (if available) or applies runtime downsampling.

### Diagnostics

`_DiagnosticsState` tracks:

| Field | Implementation |
|-------|---------------|
| `latency_p50_ms` / `latency_p95_ms` | `_LatencyTracker` — rolling deque of 200 samples, thread-safe interpolated percentile |
| `last_error_code` / `last_error_message` | Ring buffer of 20 `_ErrorEvent` objects |
| `cache_hit_rate` | Pulled from `TensorFlightClient.cache_info()` on each diagnostics request |
| `connection_state` | `"disconnected"` → `"connected"` or `"error"` |
| Rate limiting | Per-session 1 req/s window, keyed by raw token header value |

All error messages are passed through `_redact()` before storage:
- Filesystem paths matching `/...` or `C:\...` → `[REDACTED]`
- Strings of ≥ 16 URL-safe characters (potential tokens) → `[REDACTED]`

### CORS

Default allowed origins: `http://localhost:5173`, `http://127.0.0.1:5173`,
`http://[::1]:5173` (Vite dev server port). Overridable via the `cors_origins`
argument to `create_app`, or via `--cors` / `--web-url` on the CLI launcher.

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

### Windows daemon shutdown (`biopb server stop`)

When run as a background daemon (`biopb server start`), `launch` is spawned on
Windows with `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`. It therefore has no
console that `biopb server stop` (running in a different console) can deliver a
console control event to — and Win32 named events proved brittle across
sessions/elevation. So graceful shutdown is coordinated through a **sentinel
file** instead: `run_http_server` calls `_install_windows_shutdown_listener(server)`,
which starts a daemon thread that polls for `~/.local/share/biopb/tensor-server.stop`
every 0.2s. When `stop` writes that file, the thread sets `server.should_exit =
True` *and* `server.force_exit = True` (so an open browser/keep-alive connection
can't stall shutdown). uvicorn returns from `run()`, so `launch`'s `finally →
_graceful_shutdown` runs and the file-cache process lock is released.

The sentinel name is **fixed, not PID-keyed**: on Windows the process `start`
launches (and records in the PID file) can differ from the one actually running
`launch()`/uvicorn (Store-Python/uv launcher shims), so a PID in the name would
make `stop` and the daemon disagree. A leftover sentinel from a prior run is
ignored via an mtime guard. `stop` falls back to `TerminateProcess` (via
`os.kill`) if the sentinel can't be written or the daemon doesn't exit within
`--timeout`. On POSIX, `stop` sends `SIGTERM` as before. See `biopb/biopb#22`.

This assumes the **single-server model**: the fixed sentinel name (and the
singular PID file) are unambiguous because the `biopb` CLI runs at most one
managed daemon. That single-instance guarantee comes from `biopb server start`
(the PID-file check) — **not** from `launch` itself. Running
`biopb-tensor-server launch`/`serve` directly bypasses it: such a process is
**self-managed** — `biopb server stop` does not track it (no PID file, so it
reports "no server running"), and you stop it with Ctrl+C / your own process
control. Note that a directly-launched `launch` still installs the watcher on
the same fixed sentinel, so running a managed daemon and a direct `launch` side
by side on Windows is unsupported (a `biopb server stop` would also stop the
direct one). On POSIX there is no such coupling — `stop` signals one PID.

---

## Discovery & Directory Monitoring

### Discovery Protocol

**Module:** `biopb_tensor_server.core.discovery`

The discovery system uses a **claim-based protocol** where adapters "claim" filesystem paths they recognize. This enables:

1. Extensible format detection — new adapters register and participate
2. Cross-platform file identity tracking (symlink/hardlink safe via inode)
3. Live filesystem monitoring support

#### SourceClaim

Lightweight claim object using `__slots__` for memory efficiency when scanning large directories:

```python
class SourceClaim:
    __slots__ = ('source_type', 'primary_path',
                 'source_id', 'dim_labels', 'extra_config', 'is_remote')

    source_type: str      # "zarr", "ome-tiff", "hdf5", etc.
    primary_path: str     # Main entry point (str for URL support)
    source_id: str        # Auto-generated from URL hash
    dim_labels: List[str] # Optional dimension labels
    extra_config: dict    # Adapter-specific config (e.g., HDF5 dataset)
    is_remote: bool       # Flag for remote sources
```

Multi-file claims: Paths are tracked in `DiscoveryState.consumed_paths` via `try_claim_path()` callback during claim().

#### AdapterRegistry

Adapters register with the registry and implement the `claim()` classmethod:

```python
class AdapterRegistry:
    def register_with_type(source_type: str, adapter_cls: Type[BackendAdapter])
    def get_claims_for_path(path: Path, visited: Set[str]) -> List[SourceClaim]
    def get_adapter_for_type(source_type: str) -> Type[BackendAdapter]
```

Registration order (by priority/specificity, highest first — callers take
`claims[0]`; see `adapters/__init__.py::get_default_registry`):
1. `OmeTiffAdapter` — local OME-TIFF with embedded OME-XML (single- and
   multi-file); pure-tifffile, no aicsimageio dependency
1b. `QptiffAdapter` — before the bioio group so it owns `.qptiff` (bioio drops the
   QPTIFF pyramid). Claim is suffix-only; a QPTIFF saved as `.tif` is not sniffed
   (that read is unsafe under cloud and wasteful without caching — biopb/biopb#135)
   and falls through to bioio unless given an explicit `type: qptiff`
2. `ZeissAdapter` / `LeicaAdapter` / `NikonAdapter` / `DvAdapter` /
   `OlympusAdapter` / `BioformatsAdapter` / `AicsImageIoAdapter` — vendor formats
   and the generic bioio fallback (which also picks up a remote or non-OME
   `.tif` that OmeTiffAdapter declines)
3. `OmeZarrAdapter` (+ HCS) — OME-Zarr multiscales
4. `ZarrAdapter` — generic Zarr
5. `NdTiffAdapter` / `MicroManagerLegacyAdapter` / `TiffSequenceAdapter` —
   Micro-Manager NDTiff, legacy MM datasets, plain TIFF stacks
6. `DicomSeriesAdapter` / `DicomAdapter` / `NiftiAdapter` — medical imaging
7. `Hdf5Adapter` — HDF5 (explicit `hdf5` type only)
8. `RemoteTensorAdapter` — caching passthrough proxy (explicit `tensor-server`
   type only; never claims a filesystem path)

The optional bioio/ndtiff/dicom/nifti adapters register only when their
dependency is importable, so a slimmer install collapses the list without
changing the relative order of what remains.

**OME-TIFF before TIFF-sequence is load-bearing.** OmeTiffAdapter makes a
*file*-level claim on a local `.tif`/`.tiff` that carries embedded OME-XML
(consuming multi-file siblings via the master's OME-XML file list);
TiffSequenceAdapter makes a *directory*-level claim on plain TIFF stacks and
**excludes** OME-named files (`exclude_ome`, plus a `.companion.ome` guard).
Registering OmeTiffAdapter first — together with that exclude — guarantees an
`.ome.tif` becomes its own OME-TIFF source rather than being welded into a plain
sequence. OmeTiffAdapter declines a non-OME `.tif` (and any remote/cloud path),
so those fall through to the generic bioio adapter or the sequence adapter
with no regression.

**OME-Zarr before plain Zarr is load-bearing.** A `.zarr` dir can be claimed by
both; `get_claims_for_path` returns claims in registration order and callers take
`claims[0]`, so the specific adapter must win. The two stay disjoint once the store
is resident: `OmeZarrAdapter` declines a non-multiscales store, and `ZarrAdapter`
declines a real OME-Zarr — so the resident re-claim (e.g. at cloud resolve) lands
`claims[0]` on the right type even though the provisional guess was made blind.
`OmeZarrAdapter` also declines early when a top-level `.zarray`/`zarr.json` exists
(a bare array is never an OME multiscales group), so `ZarrAdapter`'s read-free,
definite claim wins for bare arrays — including deferred ones under cloud.

#### DiscoveryState

Bidirectional mappings for efficient source add/remove operations:

```python
class DiscoveryState:
    claims: Dict[str, SourceClaim]           # source_id → claim
    path_to_source: Dict[Path, str]          # primary_path → source_id
    consumed_paths: Set[Path]                # all claimed paths
    visited_identities: Set[str]             # inode-based dedup

    on_source_added: Callable[[SourceClaim], None]   # Callback
    on_source_removed: Callable[[str], None]         # Callback
```

### Directory Monitoring

**Modules:** `biopb_tensor_server.sources.watcher`, `biopb_tensor_server.sources.source_manager`

Periodic monitoring for configured directories. On each rescan interval, the catalog reconciles against a fresh stable snapshot of the monitored trees.

**Startup is progressive (biopb/biopb#212).** The bootstrap scan is **not** run
synchronously before the server serves: the launcher reaches `SERVING`
immediately, `PeriodicRescanWatcher` fires its first rescan at once
(`initial_immediate=True`), and the scan runs on the SourceManager's event-loop
thread. The **first** full scan also *streams* its additions — each source is
registered the moment the walk claims it (`SourceManager._stream_first_scan_add`
wired as the discovery state's `on_source_added`) rather than batching at
end-of-walk — so the catalog (and `health.source_count`) grows during the scan.
This is safe only for the first scan (empty + force-full ⇒ no removals to diff);
steady-state rescans keep the snapshot-diff reconcile. `_handle_rescan` pushes
`full_scan_in_progress` around a force-full pass and advances
`last_full_scan_finished_at` on success; the first success also flips the
internal `_initial_scan_done` gate (startup sources warm via the precache
*backlog*; live additions thereafter prompt-enqueue).

#### Architecture

```
Main Process
┌─────────────────────┐
│ TensorFlightServer  │
│ SourceManager       │
│ PeriodicRescanWatcher
│         │
│         ▼
│  get_events()
│         │
│         ▼
│  _process_event()
│         │
│         ▼
│  _handle_rescan()
│         │           │─────────────►│   shutdown_event    │
│         ▼           │ (shutdown)   │                     │
│  register_source()  │              │                     │
│  unregister_source()│              │                     │
└─────────────────────┘              └─────────────────────┘
```

Key design decisions:
- **Timer-driven monitoring**: The watcher emits periodic rescan requests only
- **Separation of concerns**: The watcher handles cadence while `SourceManager` handles stability checks and diffs
- **Thread-safe updates**: Server mutations are serialized via locks in the main process
- **Stable snapshots**: Catalog reconciliation runs against a scan-local discovered snapshot, not per-file events

#### Watcher Interface

```python
class DirectoryWatcher(ABC):
    def start(self, directories: Set[Path]) -> None
    def stop(self) -> None
    def get_events(self, timeout: float) -> List[WatcherEvent]
    def is_running(self) -> bool
```

Implementations:
- `PeriodicRescanWatcher` — emits `RESCAN` events at a fixed cadence for monitored directories

#### Event Types

```python
class WatcherEventType(Enum):
  RESCAN = "rescan"     # Trigger one reconciliation pass
```

#### Rescan Cadence

The watcher emits a `RESCAN` event on a fixed interval. `SourceManager` then:
- delegates the filesystem signature walk to `TreeScanner` (`tree_scanner.py`), a pure
  producer that refreshes cached directory/file signatures and skips unstable subtrees
  until they satisfy the stability window, returning an immutable `ScanSnapshot`
- performs discovery on the snapshot's stable paths only
- diffs the discovered snapshot against the confirmed catalog state

#### Move Handling

| Scenario | Behavior |
|----------|----------|
| Move within monitored dir | Preserve `source_id`, update paths |
| Move out of monitored dir | Treat as delete |
| Move into monitored dir | Treat as create (new `source_id`) |

#### SourceManager

Coordinates watcher, discovery, and server catalog updates:

```python
class SourceManager:
    def start() → None    # Start event processing thread
    def stop() → None     # Stop thread and clean up

    # Callbacks (set on DiscoveryState)
    def _on_source_added(claim)   # Create adapter, register with server
    def _on_source_removed(source_id)  # Unregister from server
```

#### Configuration

Enable monitoring per source:

```json
{
  "sources": [
    { "url": "/data/acquisition/", "monitor": true }
  ]
}
```

Only local directories can be monitored (remote URLs not supported).

#### Cloud / synced-folder sources (`cloud = true`, cloud-storage phase 2)

```json
{
  "sources": [
    { "url": "/home/u/OneDrive/microscopy/", "cloud": true }
  ]
}
```

(`cloud = true` admits dehydrated (offline-placeholder) data as *unresolved* sources.)

On a synced folder (OneDrive/Dropbox/iCloud "Files-On-Demand"), data appears as
local paths but content is *dehydrated* until accessed — and reading any byte
recalls the **whole** file (slow, refills the disk, blocks offline). The default
discovery guard therefore **skips** every offline placeholder
(`_is_offline_placeholder`, `discovery.py`), so cloud data is normally never
catalogued. `cloud = true` opts one configured root into the phase-2 model:

- **Admit, don't skip.** Under a cloud root the walk passes `admit_nonresident`
  to `should_skip_walk_entry`, so dehydrated entries reach `claim()` instead of
  being pruned. The hidden/system-dir prunes still apply.
- **Recall-free claim.** Every `claim()` recognizes a source from name + `stat` +
  `exists` + directory layout only. Two mechanisms keep it read-free under cloud,
  by *why* a format would otherwise read content:
  - *Single-source format recognition* (OME-Zarr/plain-Zarr `.zattrs`,
    MicroManager `metadata.txt`, single DICOM header): the read is guarded by
    `ClaimContext.is_resident()` and, when non-resident, the adapter emits a
    **provisional, `unresolved=True`** claim that defers it. (`is_resident()` is
    recall-free — the same stat signal — and treats directories as resident.)
    Resolution refines the *same* source in place, so there is nothing to
    reconcile.
  - *Content-derived membership* (multi-file OME-TIFF via the OME-XML file list,
    DICOM **series** via per-slice `SeriesInstanceUID`): these cannot be deferred
    safely — a directory can hold several such datasets, so the dir is not the
    boundary and the deferred member set could diverge at resolve. They are gated
    on the new **`ClaimContext.cloud_root`** flag (at scan the rescan walk records
    each entry's cloud-ness once — `TreeScanner._scan_tree_state` already knows it per
    monitored root — into a per-path map the claim phase reads via
    `discover_sources_from_entries(cloud_by_path=…)`; at resolve it comes from
    `UnresolvedSourceAdapter`, so it holds at *both* scan and resolve — residency
    cannot gate resolve, where the file is resident): under cloud
    `OmeTiffAdapter`/`DicomSeriesAdapter` **return `None`**, so each `.tif`/`.dcm`
    falls back to its own single-file source.
  - The content-free extension-only adapters need no change; a `cloud_phase2_test`
    guard pins that they (and the deferring readers) stay read-free.
- **Dir-claiming policy.** The five genuine one-dir-one-dataset formats (zarr,
  ome-zarr, ndtiff, micromanager, tiff-sequence) record **the directory** as the
  sole claim member, not an enumerated per-file glob. A claimed dir already prunes
  its whole subtree, so interior files are never independently walked; recording
  `member_paths = {dir}` makes the bookkeeping match that prune and removes the
  glob-vs-content membership divergence (so resolution never needs to reconcile a
  member set). Change-detection then keys on the dir signature (add/remove of an
  interior file bumps dir mtime).
- **Register unresolved.** `SourceManager._claim_is_unresolved` registers such a
  source behind an `UnresolvedSourceAdapter` (`adapters/unresolved.py`) — a
  catalog row with empty `tensors` / `data_resident = false`. This is also how a
  content-free *file* source (NIfTI, CZI, …) under a cloud root is deferred: its
  `claim()` never reads, but a non-resident content file (member-path placeholder
  stat, `is_file`-guarded so a directory's `st_blocks == 0` is never a false hit)
  flags it. The check is cloud-gated, so a normal local source is never marked
  unresolved.
- **Monitoring is governed by `monitor`, identically to non-cloud roots.** `cloud`
  only controls *gating* (admit placeholders, defer unresolved, set
  `ClaimContext.cloud_root`); it no longer forces a root onto the monitored
  pipeline — `cli.py` routes on `monitor` alone. A `monitor = true` cloud root is
  live-monitored; a `monitor = false` cloud root is scanned **once at startup**
  through the static-expand path, which is cloud-gated too: `config.discover_sources`
  threads `admit_nonresident` **and** `cloud_root` from `source.cloud`, so the
  one-shot scan admits placeholders and applies the multi-file OME-TIFF / DICOM
  ban exactly like the monitored path (the expanded configs also keep `cloud`, so
  they defer as unresolved).
- **Cloud subtrees are walked only on a `force_full` rescan (the monitored path).**
  `TreeScanner._scan_tree_state` **skips a cloud subtree entirely** on an incremental
  (non-`force_full`) rescan — carrying its cached claims forward untouched — and
  re-walks it only on the periodic `force_full` pass (`full_rescan_interval`,
  default 1h). The first rescan is `force_full` (last-full = −∞), so a cloud root
  is still catalogued at startup, and a brand-new cloud dataset surfaces on the
  next `force_full`. When a cloud subtree *is* walked (a `force_full` rescan, or
  the one-shot static scan), it is fully walked with no stability gate:
  `_should_scan_resolved` **bypasses the stability window and the open-for-append
  probe** for cloud entries. The probe (`_can_open_for_append`) opens the file — a
  whole-file recall on a placeholder — so skipping it is load-bearing, not an
  optimization; the mtime/age gate is skipped because archived dehydrated data is
  inherently stable and cloud mtime is untrustworthy (§1.2).
- **Pre-cache stays out of cloud (natural protection).** The background pre-cache
  worker loops `list_tensor_descriptors()` and consults `has_native_pyramid()` —
  both empty/False on the unresolved proxy — so `_process_source` returns before
  it ever reaches the resolving `get_tensor_adapter`. An unresolved source is thus
  never warmed/hydrated in the background. (The explicit `is_resident()` skip gate
  for the *post-resolution* re-warm backfire is phase 4; it cannot manifest in
  phase 2, whose resolution is in-memory and consented.)
- **Resolve on serve.** The proxy splits into a *catalog* surface
  (`list_tensor_descriptors`/`get_source_descriptor`/`has_native_pyramid`) that
  never resolves — keeping ListFlights, the metadata-DB sync, and the precache
  worker cheap (precache loops the empty tensor list and skips before any serving
  call) — and a *serve* surface (`get_tensor_adapter`) that **is** the consented
  resolution hook. The first `GetFlightInfo` hydrates: it re-runs the real claim
  + `create_from_config` on the now-resident path (the recorded `source_type` was
  a recall-free guess; the authoritative one comes from the hydrated content),
  caches the real adapter, fires `on_resolved` (the metadata-DB backfill —
  `sync_source_added` is an upsert, so the NULL-shape row is overwritten in
  place), and delegates thereafter. Resolution runs once under a lock; failure is
  classified: a transient recall/IO failure raises `SourceResolveRetriableError`
  (→ `FlightUnavailableError`/UNAVAILABLE, "retry"), a permanent one raises bare
  `SourceUnresolvedError` (→ `FlightInternalError` from the resolve action, so the
  client does not retry forever). The re-claim `except` is narrowed to **not**
  swallow an `OSError` into the claim-time guess — a recall blip can no longer be
  laundered into a wrong type.

**Known limitation (accepted).** Multi-file content-membership formats — multi-file
OME-TIFF (member set lives in the OME-XML) and DICOM **series** (grouped by the
per-slice `SeriesInstanceUID`) — are **not reconstructed on cloud**. Their
membership is intrinsically a content read and a directory can hold several such
datasets, so the dir is not the dataset boundary; deferring the grouping would
force a catalog reconciliation at resolve (forbidden — a claim is immutable once
made). Under a cloud root they therefore degrade to **N independent single-file
sources** (each `.tif`/`.dcm` its own source), permanently — there is no later
reconstruction. This matches §9 of `docs/cloud-storage-support.md` (pyramidal/
chunked OME-Zarr is the supported cloud path; **transcode monoliths to OME-Zarr
at archive time**). Phase-2 resolution is in-memory only; surviving a restart
(file-backed metadata DB) is phase 3.

#### Shutdown Sequence

```python
# CLI stop sequence:
source_manager.stop()   # First: stop event processing thread
watcher.stop()          # Then: signal subprocess shutdown
    # → shutdown_event.set()
    # → subprocess exits cleanly
    # → process.join(5)
    # → process.terminate() if needed
    # → process.kill() as last resort
```

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
