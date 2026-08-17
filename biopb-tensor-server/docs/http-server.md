# The tensor-server HTTP sidecar (FastAPI)

Endpoint/contract reference for the **API-only** HTTP sidecar. For how it fits the
data plane, see `../ARCHITECTURE.md` (§ FastAPI HTTP Server): it wraps the Python
`TensorFlightClient` and re-exposes it as HTTP/JSON (+ binary slices) so browsers
reach the data plane without a gRPC-Web proxy. It serves **no** static assets —
the control plane owns the browser UI and reverse-proxies this sidecar under
`/data_plane/*`.

**Module:** `biopb_tensor_server.serving.http_server` ·
**Factory:** `create_app(flight_location, token, cache_bytes, cors_origins, config_path, supervised) → FastAPI` ·
**Port:** `8814` under the control plane (which passes `--web-port 8814`); a bare
`biopb-tensor-server launch` defaults to `8816`.

## Lifecycle

The app holds two pieces of shared mutable state created at factory time:

- **`_client_holder`** — lazily initialised `TensorFlightClient`; the first
  authenticated request that reaches any protected endpoint triggers the gRPC
  connection to `flight_location`.
- **`_DiagnosticsState`** — thread-safe container for latency samples, error
  events, cache counters, and per-session rate-limit state.

## Authentication

Two equivalent header schemes are accepted on every protected endpoint:

```
Authorization: Bearer <token>
X-Biopb-Token: <token>
```

`secrets.compare_digest` is used for timing-safe comparison. The auth check
compares against `expected = self.token`: a `None` token means no enforcement, a
token present means it is enforced on every protected endpoint. There is no
separate dev flag.

**Enforcement is independent of the network mode.** The `--host` flag decides
the *bind* (loopback = local, public = remote; `biopb control start --remote`
selects the public one), not whether a token exists. Remote mode **requires** one (auto-generated if not supplied — a public
listener is never left open). Local mode is tokenless *by default*, but a token
passed via `--token` / `BIOPB_TENSOR_TOKEN` is honored and enforced on the
loopback listeners too (`_resolve_launch_token` takes a supplied token before it
ever looks at the bind) — defense-in-depth on a shared machine. So "no token"
is the local **default**, not a property of local mode.

> **Caveat (biopb/biopb#470):** `/api/admin/status` reports `local` as
> "the token is `None`", which a token-gated *loopback* deployment fails — it
> reports `local: false`, and `/api/admin/browse` then 404s. Fails closed; the
> data endpoints are unaffected.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/livez` | ✗ | Liveness probe — `{"status":"ok","timestamp":"…"}`. Never contacts the backend |
| `GET` | `/readyz` | ✗ | Readiness — **200 when Flight reports `SERVING`, 503 otherwise**. Adds `ready`, `backend_health`, `backend_error`, `source_count`, `dev_mode`, `service`, `version` |
| `GET` | `/healthz` | ✗ | Alias for `/readyz` |

`/readyz` opens the Flight connection if none exists yet, so it answers from the
backend rather than from whatever traffic happened to arrive first, and it is
safe for a supervisor to gate on. `backend_health` is `null` exactly when the
backend was not reached, and `backend_error` then says why (`connect failed: …`
vs `health check failed: …`) — the two used to be indistinguishable, and both
looked the same as "nobody has asked yet" (biopb/biopb#755).
| `GET` | `/api/diagnostics` | ✓ | Diagnostics snapshot; rate-limited 1 req/s per session |
| `GET` | `/api/sources` | ✓ | JSON array of `DataSourceDescriptor` objects |
| `GET` | `/api/sources/{id}` | ✓ | Single descriptor |
| `GET` | `/api/sources/{id}/metadata` | ✓ | Parsed `metadata_json` field |
| `POST` | `/api/sources/query` | ✓ | Server-side DuckDB SQL over the catalog |
| `GET` | `/api/sources/{id}/ticket/{ticket_hex}` | ✓ | Resolve a Flight ticket to bytes |
| `GET` | `/api/tile_info/{id}` | ✓ | Tile grid, pyramid levels and selectable axes for one tensor |
| `GET` | `/api/tile/{id}` | ✓ | One tile, cacheable (`fmt=raw\|png\|jpeg`) |
| `POST` | `/api/slice` | ✓ | Binary tensor sub-region |
| `POST` | `/api/render` | ✓ | Server-rendered RGB image of a slice |
| `GET` | `/api/config` | ✓ | Current config (secrets redacted) |
| `PUT` | `/api/config` | ✓ | Update config (same-origin guarded) |
| `GET` | `/api/admin/status` | ✓ | Server/catalog status for the admin page |
| `GET` | `/api/admin/browse` | ✓ | Filesystem browse for the data-folder picker (local only — see the auth caveat) |
| `WS` | `/ws/render` | ✓ | Streaming render: JSON `{action:"render", params}` in, `render_start` metadata + binary image out, repeatable |

> **Route ordering:** `/api/sources/{id}/metadata` and `/ticket/{ticket_hex}` are
> registered *before* the greedy `{source_id:path}` catch-all to avoid Starlette
> first-match shadowing.

`/ws/render` takes its token from the `Authorization` / `X-Biopb-Token` header
**or a `token` query parameter**, since browsers cannot set custom headers on a
WebSocket handshake; an unauthorized socket is closed with code `4001`. It holds
no session state — each render is an independent request/response.

## Tile endpoints

Design rationale in `remote-viewer-tiles.md`; this is the contract.

`GET /api/tile_info/{source_id}?tensor_id=…` reports everything needed to address
the tensor as a tile grid — shaped to drop into a Viv `PixelSource[]`:

```json
{
  "array_id": "src/Image:0", "dim_labels": ["T","C","Z","Y","X"],
  "shape": [1,3,16,512,512], "chunk_shape": [1,1,1,512,512], "dtype": "<u2",
  "tile_size": 512,
  "plane": {"y": 3, "x": 4, "s": null},
  "selectable": {"t": 0, "c": 1, "z": 2},
  "levels": [{"level":0,"scale":1,"height":512,"width":512,"cols":1,"rows":1}]
}
```

**`level` 0 is full resolution** (Viv's `PixelSource[]` index convention, not the
map-tile one where z grows with detail); each level halves. `tile_size` is derived
from `chunk_shape` so a tile *nests* inside a stored chunk rather than straddling
one — clients must not assume a constant.

`GET /api/tile/{source_id}` takes `level`, `col`, `row`, optional `tensor_id`,
the selection `t` / `z` / `c` (default 0), and `fmt` (`raw` | `png` | `jpeg`, plus
`lo` / `hi` / `color` / `use_min_max` for the rendered formats).

| | |
|---|---|
| `fmt=raw` | `application/octet-stream`, the tensor's own dtype, C-contiguous. `X-Shape` / `X-Dtype` / `X-Dim-Labels` as on `/api/slice` |
| `fmt=png\|jpeg` | image bytes, plus `X-Image-Width` / `-Height` and `X-Percentile-Lo-Value` / `-Hi-Value` |
| always | `ETag`, `Cache-Control: private, max-age=3600`, `Vary: Authorization`, `X-Tile-Size` / `-Level` / `-Col` / `-Row` |

`If-None-Match` revalidates to **304 without touching the backend**. The ETag
covers render settings only for the rendered formats, so adjusting contrast does
not fragment the raw-tile cache (raw contrast is a client-side shader concern).

A tile outside the level's grid is **404**; a selection index outside its axis is
**422**.

> Caching is `max-age`, not `immutable`, because a tile's bytes are only stable
> while its `array_id` is, and re-indexing currently reuses the id. Tighten both
> together once the version lives in the array_id namespace.

> **`private`, never `public`.** The URL carries no token — auth is a header, so
> rotation does not bust the cache — and RFC 9111 §3.5 lets a *shared* cache reuse
> a response to an authenticated request for a *different* request when the
> response says `public` (or `s-maxage`, or `must-revalidate`). With no token in
> the cache key that other request can be an unauthenticated one, so an nginx
> `proxy_cache`, CDN, or corporate proxy in front of a `--remote` deployment would
> serve tiles with the token checked exactly once, for someone else. `private`
> keeps the per-user browser cache that the tiled design actually needs and
> withholds the shared-cache reuse that bearer auth cannot make safe. CDN caching
> would need a different grant in the cache key — signed URLs — not a header
> change.

## Cancellation

The read routes (`/api/tile`, `/api/slice`, `/api/render`) check
`request.is_disconnected()` before starting a read and return **499** when the
caller has already hung up, counted as `cancelled_reads` in `/api/diagnostics`.
This reclaims *queued* work only — neither the Flight read nor the dask graph is
interruptible, so a client that leaves mid-compute is not noticed.

`/api/tile` reads via `run_in_threadpool`. That is load-bearing for cancellation,
not just for latency: a read blocking the event loop denies it the turn it needs
to observe other callers' disconnects, which silently defeats the check for every
request queued behind it (`remote-viewer-tiles.md` has the measurements).
`/api/slice` and `/api/render` still compute on the loop, so their check only
fires when something else has yielded.

## Slice endpoint

**Request body** (`SliceRequest` Pydantic model):

```json
{
  "source_id":        "my-zarr",
  "tensor_id":        "my-zarr",
  "slice_start":      [0, 0, 0],
  "slice_stop":       [1, 512, 512],
  "scale_hint":       [1, 2, 2],
  "reduction_method": "area",
  "pixel_budget":     1000000
}
```

`source_id` and `tensor_id` are both required. `tensor_id` is normalized to the
**`array_id`** — the sole tensor identity (see the policy at the top of
`proto/biopb/tensor/descriptor.proto`) — by `_request_array_id`, which accepts
all three shapes a caller may send: the qualified `array_id` verbatim
(`my-zarr` for a single-tensor source, `my-zarr/well_A1` for a multi-tensor one),
a bare within-source field (`well_A1` → `my-zarr/well_A1`), or a value equal to
`source_id`. There is **no `"0"` sentinel**: a single-tensor source's `array_id`
*is* its `source_id`, so sending `"0"` addresses a field literally named `0`.

**Response:**
- `Content-Type: application/octet-stream` — C-contiguous `numpy.tobytes()`
- `X-Shape: 1,512,512`
- `X-Dtype: uint16`
- `X-Dim-Labels: z,y,x`

`scale_hint` and `reduction_method` are forwarded verbatim to
`TensorFlightClient.get_tensor(...)`, which resolves the appropriate
precomputed pyramid level (if available) or applies runtime downsampling.

## Diagnostics

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

## CORS

Both `create_app(cors_origins=None)` and the CLI launcher default to the loopback
variants of the sidecar's own bind: `http://localhost:8814`,
`http://127.0.0.1:8814`, `http://[::1]:8814` (substituting the actual
`--web-host:--web-port`). That covers the control front reaching the data API and
`/ws/render` over loopback. No web app is bundled with this package, so there is
no frontend origin in the default set.

A browser app served on another origin must be allowed explicitly: the
`cors_origins` argument to `create_app`, or `--cors` (repeatable) on the CLI
launcher.
