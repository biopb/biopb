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
| `server.activity` (`ActivityTracker`) | `activity.py` | in-flight heavy-read counters + last-active stamp (the precache idle signal) and the warm-in-progress guard set. Fed by every heavy read — `do_get`, `warm`, **and `chunk_locate`**: the localhost fast path *replaces* `do_get`, so leaving it untracked made the server look idle for the whole of a localhost read (biopb/biopb#548) |
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

Two role ABCs in `core/adapter_base.py`, and they **nest**: `TensorAdapter` subclasses
`SourceAdapter`, so a tensor adapter is a source that can also serve pixels
(biopb/biopb#380). Every concrete format adapter subclasses `TensorAdapter` and
fills both roles in one object — `get_tensor_adapter()` returns `self` for a
single-tensor format, and a clone of the same class (or a plain `ZarrAdapter`,
for OME-Zarr / QPTIFF levels and HCS fields) for a multi-tensor one. The lone
source-only adapter is `UnresolvedSourceAdapter`, which has no tensors until it
resolves. The role *scopes* stay disjoint where they are declared — `adapter_base.py`
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
| O(IFD count) or O(file count) — unbounded, never amortises | persistent handle + `close()`, and a shared idle reaper closes the handle between reads so the pin is *bounded*, not lifetime-long (TTL from `[server] handle_reaper_ttl`, default 150 s) | `ome-tiff`, `qptiff`, `ndtiff` |

The reaper is one small opt-in utility, `adapters/_handle_reaper.py` (`IdleHandleReaper`): the second-row adapters register the handle on open and expose the `ReapableHandle` contract (`_io_lock`, `_active_reads`, `_persistent_last_access`, `_release_persistent_handle`); a per-pool daemon thread closes any handle idle past its TTL, fenced against an in-flight read, and the next read reopens transparently. `close()` (teardown) and the reaper (steady state) share the one release hook. (`qptiff` keeps a persistent handle + `close()` but is not yet wired to the reaper — its multi-level store pool would register per level.)

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

**Local-disk gate (biopb/biopb#571).** The file backend mmaps its segments (for
its own segment reads/boot index and the localhost client fast path) and assumes
local-POSIX semantics — an unlinked-but-mapped inode survives to last close, a
mapped page never vanishes. A **network** (`nfs`/`cifs`/…) or **cloud
Files-On-Demand** (`OneDrive`/`iCloud`/`Dropbox`) `cache_dir` breaks that (mmap
SIGBUS/ESTALE on an evicted segment; a recall stall on a dehydrated one). So the
launcher classifies the configured `file_cache_dir` **once at startup**
(`core/fs_detect.py` — Linux `/proc/self/mountinfo`, Windows `GetDriveTypeW`/UNC,
macOS `statfs`, plus cloud-root path heuristics; all metadata-only, never
raising, and demoting **only on a positive signal**) and falls back to the
**memory backend** when the dir isn't plain local disk — which also disables the
client fast path for free (a memory backend never locates a chunk).

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
Between this and the sidecar above, index entries are created with their range
already known — the two constructors are the write path and the boot restore,
and there is no third. So `locate_entry` derives nothing: it is a dict lookup
under `_lock`. The single way an entry can still lack a range is a failed
schema-length read on a segment's *first* append; that entry reports unavailable
and the client transfers it over `do_get`, later entries in the segment are
unaffected (they bracket straight off the sink cursor), and a restart repairs it
from the segment body. Degrading to the socket is the designed floor of this
whole path, so there is nothing to recover beyond it.

The lazy `_fill_byte_offsets_for_segment` walk it replaced is **deleted**, not
kept as a fallback, for two reasons. It cost O(entries in the segment) per call
(measured ~5 ms at 145 MB with 0.87 MB chunks; it scales with entry *count*, so
a 128 KB-chunk source pays ~12 ms at the same 128 MB) and was paid on **every
cache miss**. And it was the only place the read path took `_write_lock` — so a
write stalled on a full filesystem blocked locates behind it, the coupling that
makes a wedged cache non-self-healing. Torn-tail tolerance (a partial write's
slack) still matters, but on the walk that actually meets it: the boot-time
`_scan_segment_records`.

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
biopb-tensor-server launch --config biopb.json [--host 0.0.0.0] [--port 8815] [--writable] [--web-port 8816] [--web-host 127.0.0.1] [--cors ORIGIN]

# for grpc only (no web server) — same flight options + token handling as launch
biopb-tensor-server serve --config biopb.json [--host 0.0.0.0] [--port 8815] [--writable] [--tls] [--san NAME]

# generate / rotate the self-signed TLS cert and print its fingerprint
biopb-tensor-server cert init [--force] [--san NAME]
```

`serve` and `launch` share the Flight-server flags (`--host`/`--port`/`--writable`
override the config bind; `--token`/`--log-level`/`--log-file`) and the same
fail-closed token resolution (`_resolve_flight_token`). `launch` adds the HTTP
sidecar (`--web-host`/`--web-port`/`--cors`) and layers the sidecar fail-closed
check on top (`_resolve_launch_token`).

**TLS (`serve --tls`, biopb/biopb#604).** The encryption story is deliberately
CA-free: `--tls` serves the flight plane over `grpc+tls://` using a **self-signed**
cert (its own trust anchor), auto-generated into the state tree
(`state/biopb/tls/`) on first use — so a headless server stands up TLS with no CA
to manage. Clients connect with `grpcs://` and **pin the cert on first connect**
(TOFU, `biopb.tensor._tls`); the SDK stores the pin in `state/biopb/tls-known-hosts.json`
and refuses a later mismatched cert. `cert init` pre-seeds or rotates the cert and
prints the fingerprint a client will pin. `--tls-cert`/`--tls-key` serve a BYO cert
instead. The cert machinery lives beside the token machinery (`core/tls.py`),
*not* in the control plane, so case 2 (no control installed) works.

**Pinning is the trust anchor; the SANs still have to match.** gRPC keeps
hostname verification on, so the name a client dials must appear in the cert's
SANs — pinning only replaces the *CA*, not the name check. `collect_san_hosts()`
enumerates what the host can see about itself (`localhost`, hostname/FQDN,
loopback, the primary outbound IP, `getaddrinfo` results), which misses a name
that lives elsewhere: a NAT/VPN address, a CNAME, a reverse-proxy hostname. That
case pins fine and *then* fails every handshake, so it gets explicit handling at
both ends — `--san NAME` (repeatable, on `cert init` and `serve --tls`; applies
only when the cert is generated, hence `cert init --force --san …` to widen an
existing one), and a client-side probe that logs the exact name and the
`cert init --force --san` fix instead of leaving an opaque TLS error. The probe is
diagnostic only: any outcome other than a definite hostname mismatch is silent, so
it can never break a working connection.

TOFU resolution is **memoized per process** keyed by `host:port` *and* the
configured trust material (a caller may supply an explicit CA or fingerprint —
see *Per-upstream credentials* below — and one process fronting two upstreams at
the same address must not serve one's anchor to the other)
(`_tls.clear_pin_cache()` drops it). The call sites evaluate it eagerly on paths
that usually reuse an already-open pooled connection, so without the memo every
`GetFlightInfo` would open a throwaway TLS handshake just to re-derive a value it
already had — and a momentary failure of *that* side handshake would fail a call
the healthy pooled connection could have served. The consequence is that a cert
rotation is detected at process start, not mid-run; that matches the ceremony a
mismatch requires anyway (confirm, clear the pin, reconnect).

`cryptography` (the cert generator) is an **opt-in `[tls]` extra**, deliberately
kept out of the default install closure: it drags a Rust/OpenSSL build surface
with no recent Intel-macOS wheel that broke `curl install.sh | bash` there
(biopb/biopb#355, which *dropped* the transitive dep). So TLS *serving* opts in
(`pip install 'biopb-tensor-server[tls]'`; `serve --tls`/`cert init` raise an
actionable error if it is absent — cleanly, not as a traceback), while the SDK
client's TOFU pinning (`biopb.tensor._tls`) is stdlib-`ssl` only and needs nothing
extra. A **BYO cert** (`--tls-cert`/`--tls-key`) is read straight off disk and
needs no `cryptography` at all — the escape hatch when the extra isn't installed.

**Switching an installed (non-`[tls]`) deployment to remote TLS:**
`pip install 'biopb-tensor-server[tls]'`, then `serve --tls --host 0.0.0.0`
(the public bind auto-generates a token; print it once). Clients connect
`grpcs://<host>:8815` with that token and TOFU-pin the cert. Skip the install and
bring a cert via `--tls-cert`/`--tls-key` instead.

**Config-driven TLS.** `--tls`/`--tls-cert`/`--tls-key` are on **both** `serve`
and `launch`, and each mirrors a `server.tls` / `server.tls_cert` /
`server.tls_key` config field. The config field is what makes TLS reachable at
all from the control: the supervisor spawns `launch -c config.json` and passes no
TLS arguments, so a config that could not express TLS would be a plane the
control could never serve over TLS. The flag is tri-state (`--tls/--no-tls`) and
overrides the config in **both** directions; omitting both defers to the config.
`--no-tls` additionally drops any cert/key pair — a cert on its own *means*
"serve TLS" (`--tls-cert` never needed `--tls`), so `_resolve_tls_material`
honors the pair before it consults the flag, and an explicit off has to be
applied in `_merge_tls_options` or it would be silently ignored.

Serving TLS from `launch` means the co-located sidecar has to reach a TLS Flight
plane. It does: the flight location becomes `grpcs://<loopback>` and the served
cert PEM is handed to the sidecar's `TensorFlightClient` as `tls_ca_pem` — an
explicit anchor read off local disk, *not* a TOFU pin. That distinction matters
operationally: pinning would record the cert in the shared pin store and then
break the sidecar the moment an operator rotated it with `cert init --force`.
The auto-generated cert always carries `localhost`/`127.0.0.1`/`::1` SANs
(`core.tls._host_identity`), so the loopback dial passes gRPC's name check; a BYO
cert minted only for a public name does not, and must be re-minted to include one.

Local clients of a TLS plane trust it the same way the sidecar does — the cert
off local disk, not a TOFU pin (`biopb_mcp._connection._local_ca`), failing loudly
if it is unreadable rather than degrading to pinning.

**Still open (case 1 / control).** The admin UI has no TLS toggle, and a missing
`[tls]` extra surfaces as a buried `tensor-server.log` crash rather than an
actionable message in the browser. That needs a `find_spec("cryptography")`
preflight ahead of the restart (biopb/biopb#604). Token rotation is deliberately
out of scope: the token is resolved once at `biopb control start`, so rotating it
is a control-level operation, not something the plane's admin UI can offer.

Startup sequence (`launch`):

1. Decide whether a token is enforced from the effective flight bind (`--host`
   override, else config `server.host`): a loopback bind runs tokenless (**local
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
7. Build CORS origins: loopback variants of the sidecar's own address by
   default (no web app is bundled here), plus any explicit `--cors` origins for
   a browser app served elsewhere.
8. Call `run_http_server(...)` — **blocking** uvicorn call. Under TLS it gets a
   `grpcs://` flight location plus the served cert as `tls_ca_pem`. The sidecar is
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
| `BIOPB_TENSOR_CACHE_LIMIT` | TensorFlightClient (Python) | Default client-side chunk-cache budget when the caller passes no `cache_bytes`; a size string with common units (`2GiB`, `512MB`) or a bare byte count (parsed via `dask.utils.parse_bytes`), `0` disables the cache. Unset/unparseable → 1 GB. A constructor `cache_bytes` overrides it. |
| `BIOPB_TENSOR_TOKEN` | `biopb-tensor-server launch` (server) | Pre-set server token for remote mode (else auto-generated) |
| `BIOPB_UPSTREAM_TENSOR_TOKEN` | `tensor-server` source dialing (`resolve_upstream_credentials`) | Bearer token for **one** upstream tensor server — a single-upstream convenience. A source's credentials profile overrides it, and is the only way to give several upstreams different tokens (or any TLS trust). |
| `BIOPB_TENSOR_ALLOW_NO_TOKEN` | `serve`/`launch` token resolution (`_allow_no_token_from_env`) | Truthy (`1`/`true`/`yes`/`on`) forces **tokenless** operation even on a public bind — the deliberate insecure escape hatch (trusted networks only). Only takes effect when no token is supplied; auto-generation and the public-sidecar refusal both become a loud warning instead. Off by default, so the fail-closed guarantee is unchanged unless explicitly set. |
| `BIOPB_BIND_LOCALHOST` | Docker/Singularity entrypoint | Bind to loopback → local mode / no token (Singularity/HPC only; ignored in Docker) |
| `BIOPB_ENABLE_HTTP_SIDECAR` | Docker/Singularity entrypoint | Truthy → run `launch` (Flight + the HTTP sidecar) instead of the default Flight-only `serve`. See *Container shape* below. |
| `BIOPB_TENSOR_TLS`, `BIOPB_TLS_CERT`/`BIOPB_TLS_KEY` | Docker/Singularity entrypoint | Serve Flight over TLS with the self-signed cert (`--tls`), or with a mounted BYO cert. Forwarded to `serve` *or* `launch`; with the sidecar opted in, a BYO cert needs a loopback SAN. |
| `BIOPB_OMETIFF_PARALLEL_READ` | `OmeTiffAdapter.get_data` | Opt in (`=1`) to lock-free OME-TIFF chunk reads — concurrent tile decodes run in parallel instead of serializing under `_io_lock` (biopb/biopb#473). **Default off**: reads decode under the lock, as before. |

The idle-handle reaper TTL is a **config** knob, not an env var: `[server] handle_reaper_ttl` (seconds, default `150`, `<= 0` disables). It applies to every opt-in adapter (OME-TIFF, NDTiff), set once at startup via `set_handle_reaper_ttl`.

---

## Security Model

- Token is stored in `sessionStorage` (clears on tab close, never persisted to disk).
- The FastAPI sidecar validates `Authorization: Bearer <token>` on every request via `HTTPBearer`.
- The Arrow Flight server validates the same token via `BearerAuthMiddlewareFactory`.
- **Local mode** (loopback `server.host`) enforces no token — the 90% single-machine case. **Remote mode** (public `server.host`) requires a token, auto-generated if none is supplied.
- **The HTTP sidecar bind (`--web-host`) is fail-closed too.** It has its own bind address, independent of `server.host`, and re-exposes the whole data API. So `launch` **refuses to start** if the sidecar would bind a public address (`--web-host 0.0.0.0`/a real IP) while no token is enforced — the loopback-`server.host` case, where the token resolves to `None`. "Public + unauthenticated" is unrepresentable on *either* listener, not just the flight server (`_resolve_launch_token`).
- **The one deliberate escape hatch is `BIOPB_TENSOR_ALLOW_NO_TOKEN`** (`_allow_no_token_from_env`). Truthy, it forces tokenless operation even on a public bind — auto-generation and the public-sidecar refusal both degrade to a loud warning. It only takes effect when no token is otherwise supplied, and is **off by default**, so the fail-closed guarantee above holds unless an operator explicitly opts out for a trusted network (the host-loopback-published Docker case, where the in-container bind is `0.0.0.0` but the ports are published to `127.0.0.1`). This is *not* the old auto dev-bypass (removed in #447) — it is explicit, per-deployment, and self-announcing.
- For Docker local mode with localhost-only access, use `-p 127.0.0.1:8815:8815`.
- For Singularity/HPC local mode with localhost-only binding, use `BIOPB_BIND_LOCALHOST=true`.
- Error messages are redacted before logging/storage (filesystem paths and potential tokens replaced with `[REDACTED]`).

---

## Per-upstream credentials (mounting a remote plane)

A downstream server that mirrors a remote plane as a `tensor-server` source is a
*client* of that plane, and cross-host the #470 filesystem token handoff does not
reach it: the token lives in the upstream host's state dir. So the upstream's
credentials are explicit config — and because one downstream may mount several
upstreams belonging to different groups, the binding is **per source**, not one
global setting (biopb/biopb#604 item 4).

The carrier is the existing credentials profile, with `storage_type:
"biopb-tensor"` selecting a tensor-server upstream rather than object storage; a
source names one through `credentials_profile`. Three keys apply:

| Key | Meaning |
|---|---|
| `token` | Bearer token for the upstream. Beats the single-upstream `BIOPB_UPSTREAM_TENSOR_TOKEN` env fallback. |
| `tls_fingerprint` | Expected SHA-256 of the upstream's cert (colon-grouped or bare hex, as `cert init` prints it). Verified on every connect. |
| `tls_ca_file` | Path to a PEM to trust — a private CA, or the upstream's own leaf. |

`resolve_upstream_credentials()` (in `adapters/remote_tensor.py`) produces one
frozen `UpstreamCredentials` from the source + config, and **all three** dial
sites use it: the adapter's pooled client, the reconciler's bulk catalog fetch,
and the bare-host expansion in `core/config.py`. The latter two dial the upstream
directly, outside the adapter pool, so leaving either on the old token-only path
would have TOFU-pinned a `grpcs://` upstream whose CA was configured.

Design points worth keeping:

- **TLS trust is optional, and unset means TOFU.** The zero-config default already
  works; configuring an anchor buys the one thing TOFU cannot — rejecting an
  impostor that is in the path at *first* contact, where there is no prior use to
  trust. `tls_fingerprint` is the light form (paste what the server printed);
  `tls_ca_file` is for a real private CA.
- **An unreadable `tls_ca_file` raises.** Degrading to TOFU would silently undo the
  stronger trust the operator explicitly configured, on something as ordinary as a
  typo'd path.
- **Both keys set → the CA wins, with a warning.** Never leave an operator
  believing a fingerprint is enforced when it isn't.
- **The client pool keys on the credentials, not just the endpoint.** Two sources
  naming one `host:port` with different tokens or anchors get different
  connections; otherwise whichever dialed first would silently decide what the
  other authenticates as and trusts. The SDK's TOFU memo is keyed the same way,
  for the same reason.
- **The env var stays token-only.** `BIOPB_UPSTREAM_TENSOR_TOKEN` remains a
  single-upstream convenience; TLS trust never grew an env twin, since anything
  worth overriding about it is inherently per-upstream.

---

## Container shape (Flight-only by default)

The published image is a **pure gRPC data-plane endpoint**: `entrypoint.sh` runs
`biopb-tensor-server serve`, so the container has **one** listener (Flight 8815)
and no HTTP surface at all — no browser origin, no CORS, no unlock page
(biopb/biopb#604 item 3). Browsing containerized data is a *downstream* concern:
a machine running the full stack adds `grpc://`/`grpcs://<host>:8815` as a remote
source, and its browser talks only to its own loopback control. That is what
makes remote TLS cheap — no browser ever has to trust the container's cert.

The FastAPI sidecar is still installed (the `web` extra) and returns with
`BIOPB_ENABLE_HTTP_SIDECAR=1`, which switches the entrypoint back to `launch`
plus `--web-host`/`--web-port`/`--cors`. Only the *default* changed.

Two consequences worth knowing:

- **TLS and the sidecar compose.** The sidecar's internal `TensorFlightClient`
  dials `grpcs://` over loopback with the served cert as its trust anchor, so the
  entrypoint forwards the TLS flags to `launch` as readily as to `serve`. (They
  used to be mutually exclusive, refused at exit 2.) A *BYO* cert must carry a
  loopback SAN — gRPC still name-checks the dial — which the auto-generated cert
  always does.
- **The self-signed cert lives in the container's state dir**, so it is ephemeral
  unless a volume is mounted at `/root/.local/state`. A recreated container mints
  a new cert, which every TOFU-pinned client then refuses — correct behavior, but
  it must be designed around. A BYO cert (`BIOPB_TLS_CERT`/`BIOPB_TLS_KEY`) is the
  other stable option.

The `[tls]` extra *is* installed in the image: the reason it is opt-in on PyPI
(no recent Intel-macOS `cryptography` wheel, biopb/biopb#355) is a source-install
problem this linux image never has.

`entrypoint.sh` is covered by `tests/entrypoint_test.py`, which runs the real
script with a stub CLI on `PATH` and asserts the argv it execs.

---

## Versioning

The tensor server tracks the **product line**, keyed to the tag `release-v*` (the
same line as `biopb-mcp`, `biopb-control`, and the `web/` bundle). Its wheel ships
in the `release-v*` GitHub bundle and its Docker image is cut on the same tag by
`tensor-server-ci` — distinct only from the SDK line (`v*`, for `biopb` +
`biopb-image-base`). See `../docs/release-model.md`.

```
git tags (release-vX.Y.Z)  →  setuptools_scm  →  biopb_tensor_server/_version.py
```

Version is derived via `setuptools_scm` with `tag_regex = "^release-v..."` (and a
matching `git describe --match 'release-v*'`). The web JS packages track the same
product `release-v*` tag (`web/scripts/sync-version.js`).

**Docker image**: `git tag release-v0.11.0 && git push --tags` (the same tag that
cuts the GitHub bundle). `tensor-server-ci`'s `publish` job then builds and pushes
`biopb-tensor-server:0.11.0` + `:latest` to ghcr.io + Docker Hub. Only a **final**
`release-vX.Y.Z` publishes — an rc tag (`release-v0.12.0rc1`) runs the tests and
the image build but pushes nothing. The SDK (incl. image-base) releases
separately on `v*`.
