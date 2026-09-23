# The tensor-server HTTP sidecar (FastAPI)

Endpoint/contract reference for the **API-only** HTTP sidecar. It wraps the
Python `TensorFlightClient` and re-exposes it as HTTP/JSON (+ binary slices)
so browsers reach the data plane without a gRPC-Web proxy — see
[ARCHITECTURE.md](../ARCHITECTURE.md) (§ FastAPI HTTP Server) for how it fits the data plane.
It serves **no** static assets: the control plane owns the browser UI and
reverse-proxies this sidecar under `/data_plane/*`.

**Module:** `biopb_tensor_server.serving.http_server` ·
**Factory:** `create_app(flight_location, token, cache_bytes, cors_origins, config_path, supervised) → FastAPI` ·
**Port:** `8814` under the control plane (which passes `--web-port 8814`); a
bare `biopb-tensor-server launch` defaults to `8816`.

## Lifecycle

Two pieces of shared mutable state live on `app.state.sidecar` (a
`_SidecarContext`, created at factory time):

- **The Flight client** connects lazily — the first authenticated request to
  reach any protected endpoint opens the gRPC connection to `flight_location`.
- **Diagnostics state** is a thread-safe container for latency samples, error
  events, cache counters, and per-session rate-limit state.

## Authentication

Two equivalent header schemes are accepted on every protected endpoint:

```
Authorization: Bearer <token>
X-Biopb-Token: <token>
```

The check is timing-safe (`secrets.compare_digest`, via the shared
`biopb._web_auth` policy the control plane also uses) and compares against
the token the sidecar was launched with. A `None` token means no enforcement;
a token present is enforced on every protected endpoint. There is no separate
dev flag.

**Enforcement is independent of the network mode.** `--host` decides the
*bind* (loopback = local, public = remote via `biopb control start
--remote`), not whether a token exists. Remote mode always requires one
(auto-generated if not supplied — a public listener is never left
unauthenticated). Local mode is tokenless *by default*, but a token passed
via `--token` / `BIOPB_TENSOR_TOKEN` is honored and enforced on the loopback
listeners too — defense-in-depth on a shared machine. "No token" is the local
**default**, not a property of local mode.

**`/api/admin/status`'s `local` flag is a proxy for the bind, not the bind
itself**: it reports `true` exactly when `token is None`. A loopback
deployment running behind an optional token therefore reports `local: false`,
which hides the `/api/admin/browse` file chooser it would otherwise qualify
for. It fails closed — the data endpoints are unaffected — but it's the thing
to remember when a local, token-protected box can't browse.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/livez` | ✗ | Liveness probe — `{"status":"ok","timestamp":"…"}`. Never contacts the backend |
| `GET` | `/readyz` | ✗ | Readiness — **200 when Flight reports `SERVING`, 503 otherwise**. Adds `ready`, `backend_health`, `backend_error`, `source_count`, `dev_mode`, `service`, `version` |
| `GET` | `/healthz` | ✗ | Alias for `/readyz` |
| `GET` | `/api/diagnostics` | ✓ | Diagnostics snapshot; rate-limited 1 req/s per session |
| `GET` | `/api/sources` | ✓ | JSON array of `DataSourceDescriptor` objects |
| `GET` | `/api/sources/{id}` | ✓ | Single descriptor, by targeted lookup — not capped like the listing |
| `GET` | `/api/sources/{id}/metadata` | ✓ | Parsed `metadata_json` field |
| `POST` | `/api/sources/query` | ✓ | Server-side DuckDB SQL over the catalog |
| `GET` | `/api/sources/{id}/ticket/{ticket_hex}` | ✓ | Resolve a Flight ticket to bytes |
| `POST` | `/api/sources/{id}/resolve` | ✓ | Begin resolving an unresolved source; joins one already running (same-origin guarded) |
| `GET` | `/api/sources/{id}/resolve/status` | ✓ | Poll that resolve; 404 if none was started |
| `POST` | `/api/sources/{id}/resolve/cancel` | ✓ | Ask it to stop (same-origin guarded) |
| `POST` | `/api/sources/{id}/warm` | ✓ | Hydrate-ahead a resolved source's member files (same-origin guarded) |
| `GET` | `/api/sources/{id}/warm/status` | ✓ | Poll that warm; 404 if none was started |
| `POST` | `/api/sources/{id}/warm/cancel` | ✓ | Ask it to stop (same-origin guarded) |
| `GET` | `/api/tile_info/{array_id}` | ✓ | Tile grid, pyramid levels, selectable axes and the 3-D volume plan |
| `GET` | `/api/tile/{array_id}` | ✓ | One tile, cacheable (raw bytes) |
| `POST` | `/api/slice` | ✓ | Binary tensor sub-region; `scale_policy` delegates the scale |
| `GET` | `/api/rois/{array_id}` | ✓ | A tensor's whole ROI annotation set (`?set=` for one layer) |
| `POST` | `/api/rois/{array_id}` | ✓ | Create/update annotations (same-origin guarded) |
| `DELETE` | `/api/rois/{array_id}` | ✓ | Delete by `?ids=a,b`, else the whole set / one `?set=` (same-origin guarded) |
| `GET` | `/api/config` | ✓ | Current config (secrets redacted) |
| `PUT` | `/api/config` | ✓ | Update config (same-origin guarded) |
| `GET` | `/api/admin/status` | ✓ | Server/catalog status for the admin page |
| `GET` | `/api/admin/browse` | ✓ | Filesystem browse for the data-folder picker (local only — see the auth note above) |

> **Route ordering:** `/api/sources/{id}/metadata`, `/ticket/{ticket_hex}` and
> the `/resolve` + `/warm` sub-paths are registered *before* the greedy
> `{source_id:path}` catch-all to avoid Starlette first-match shadowing.
>
> **`/readyz` connects.** It opens the Flight connection if none exists yet,
> so it answers from the backend rather than from whatever traffic happened
> to arrive first, and it is safe for a supervisor to gate on. `backend_health`
> is `null` exactly when the backend was not reached, and `backend_error`
> then says why (`connect failed: …` vs `health check failed: …`).

### Sources

**Source listings are structural.** Each `tensors[]` entry on `/api/sources`
carries `array_id` / `dim_labels` / `shape` / `dtype`; `chunk_shape` is `[]`
there and is **not** a usable grid. The transfer grid belongs to the tensor
the server binds to serve a read, so ask `/api/tile_info/{array_id}` for it.

**`/api/sources/{id}` is a single-row lookup** (a catalog query keyed on
`source_id`), so it is not bounded by the listing's row cap — a source past
that cap has a descriptor here but no entry on `/api/sources`. The catalog is
public: a token-protected source is listed too (the token gates its pixels),
but a `cache:` upload has no catalog row at all — reach one through
`/api/tile_info/{array_id}`.

### ROI annotations

`/api/rois/*` carries canonical proto3 JSON of `biopb.image.RoiAnnotation` in
both directions. The version token is stripped from `array_id` before the
store sees it — from the path *and* from each annotation in a POST body,
since responses carry versioned ids and a read-edit-write round trip hands
one back — and spliced onto the response. Annotations anchor on the
unversioned id so they outlive an in-place edit. `501` means the server does
not offer annotations (disabled, or no metadata DB); `422` means the request
was rejected (geometry the store does not accept, mismatched `array_id`,
per-tensor cap). Design in [roi-annotations.md](roi-annotations.md).

### Resolve and warm

Both hydrate cloud / synced-folder data and both can run for minutes, which is
longer than a request should be held open. They are **jobs**: `POST` to start,
`GET .../status` to poll, `POST .../cancel` to stop. The recall itself lives
on the server and outlives the HTTP request either way.

**Polling, not SSE.** Every other route here is request/response, and a
status object the client re-reads survives the two things a stream does not: a
reload mid-resolve, and a second tab watching the same source.

**Keyed by `(kind, source_id)`, not a generated job id.** That key *is* the
idempotency callers need — a double-click, a retry, or a second tab joins the
recall already running rather than starting a second one against the same
bytes. `POST` answers `202` either way, with `started` saying which happened.

| Field | Meaning |
| --- | --- |
| `state` | `running`, `done`, `error`, `cancelled` |
| `progress` | Kind-specific. Resolve: `elapsed_seconds`, `target_name`, `target_bytes`. Warm: `files_total`/`files_done`, `bytes_total`/`bytes_done`, `current_name`, `elapsed_seconds` |
| `error` | Reason, on `state == "error"` only |
| `cancel_requested` | Set the moment a cancel is asked for — before `state` turns, which only happens once the worker unwinds |

**Cancelling a finished job is a no-op, not an error.** The click races the
last heartbeat often enough that erroring would show a failure for doing
nothing wrong.

**Warm needs no client-side list of multi-file source types.** A source with
nothing to warm finishes immediately with `files_total == 0` — the server
decides structurally (is the source a directory), so no client has to keep a
copy of that list and keep it in step.

Finished outcomes stay readable for five minutes, then are reaped on the next
start, so a slow poller still learns *why* a job ended.

### Tile endpoints

```
GET /api/tile_info/{array_id}
GET /api/tile/{array_id}
```

Full wire contract (response shape, query parameters, versioning, caching,
validation) and the client-side rendering design both live in
[http-server-tile-endpoints.md](http-server-tile-endpoints.md). In short: **addressed by `array_id` alone**,
resolved by **one targeted `GetFlightInfo`** -- the catalog is never listed to
look up an id the request already carries -- and a bare `source_id` on a
multi-tensor source resolves to whatever the Flight server binds as its
default tensor rather than 404ing; the response carries the qualified
`array_id` it resolved to, which the sidecar reports and the client threads
through every later tile request.

### Slice endpoint

**Request body** (`SliceRequest` Pydantic model):

```json
{
  "array_id":         "my-zarr",
  "slice_start":      [0, 0, 0],
  "slice_stop":       [1, 512, 512],
  "scale_hint":       [1, 2, 2],
  "scale_policy":     null,
  "reduction_method": "area",
  "pixel_budget":     1000000
}
```

`array_id` is required and is the whole address — `my-zarr` for a
single-tensor source, `my-zarr/well_A1` for a multi-tensor one. It resolves
through the same lookup the tile routes use, so one id cannot name two
tensors depending on which route asked, and the read is issued for the
array_id that lookup came back with.

A bare within-source field (`well_A1`) is **not** accepted; send the
qualified form. There is **no `"0"` sentinel** either: a single-tensor
source's `array_id` *is* its `source_id`, so `"0"` addresses a field
literally named `0`. A body carrying a separate `(source_id, tensor_id)` pair
and no `array_id` is a **422**.

**Response:**
- `Content-Type: application/octet-stream` — C-contiguous `numpy.tobytes()`
- `X-Shape: 1,512,512`
- `X-Dtype: uint16`
- `X-Dim-Labels: z,y,x`
- `X-Scale-Hint: 1,2,2` — the per-axis scale actually read at

`scale_hint` and `reduction_method` are forwarded verbatim to
`TensorFlightClient.get_tensor(...)`, which resolves the appropriate
precomputed pyramid level (if available) or applies runtime downsampling.

#### `scale_policy` — letting the server choose the scale

`scale_policy: "volume"` reads at the one scale the server keeps a whole 3-D
volume warm at: the Flight ladder's coarsest level, which is what the precache
worker warms and what napari's 3-D mode uploads as a single texture (see
[precache-policy.md](precache-policy.md)). `/api/tile_info`'s `volume` block says what it will
resolve to for a given tensor, and a tensor with no volume is a **422**
carrying that block's own `reason`.

It exists because a client cannot compute that level without reimplementing
the server's pyramid planner, and a guess one rung away misses every warmed
chunk — `chunk_id` is `array_id + bounds + scale_hint + reduction_method`, so
a neighbouring scale shares no cache entry and pays a cold read of the
source.

`scale_policy` and `scale_hint` are **mutually exclusive** (422): one read
has one scale, and letting the two disagree would make which one applies a
silent policy. `X-Scale-Hint` is echoed either way, but it is load-bearing
here — the caller did not choose, so the header is the only statement of
what it got.

### Config and admin

```
GET  /api/config
PUT  /api/config
GET  /api/admin/status
GET  /api/admin/browse
```

Full contract (schema-driven validation, credential redaction, restart
ownership, the same-origin guard, the admin page itself) lives in
[http-server-admin-endpoints.md](http-server-admin-endpoints.md). In short: `GET`/`PUT /api/config` round-trip
the raw `biopb.json` dict, secrets redacted on the way out and restored on
the way in; the server reads config once at startup, so applying an edit
means a restart, which this sidecar does not perform itself -- restart is
owned by the control plane.

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

## Cancellation

The read routes (`/api/tile`, `/api/slice`) check
`request.is_disconnected()` before starting a read and return **499** when the
caller has already hung up, counted as `cancelled_reads` in `/api/diagnostics`.
This reclaims *queued* work only — neither the Flight read nor the dask graph
is interruptible, so a client that leaves mid-compute is not noticed.

Both read via `run_in_threadpool`. That is load-bearing for cancellation, not
just for latency: a read blocking the event loop denies it the turn it needs
to observe other callers' disconnects, which silently defeats the check for
every request queued behind it (measurements in [http-server-tile-endpoints.md](http-server-tile-endpoints.md)).
This matters most for `/api/slice`: a `scale_policy` read is a whole 3-D
block, so the queue behind it can be seconds long.

## CORS

Both `create_app(cors_origins=None)` and the CLI launcher default to the
loopback variants of the sidecar's own bind: `http://localhost:8814`,
`http://127.0.0.1:8814`, `http://[::1]:8814` (substituting the actual
`--web-host:--web-port`). That covers the control front reaching the data API
over loopback. No web app is bundled with this package, so there is no
frontend origin in the default set.

A browser app served on another origin must be allowed explicitly: the
`cors_origins` argument to `create_app`, or `--cors` (repeatable) on the CLI
launcher.
