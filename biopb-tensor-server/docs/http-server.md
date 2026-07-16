# The tensor-server HTTP sidecar (FastAPI)

Endpoint/contract reference for the **API-only** HTTP sidecar. For how it fits the
data plane, see `../ARCHITECTURE.md` (§ FastAPI HTTP Server): it wraps the Python
`TensorFlightClient` and re-exposes it as HTTP/JSON (+ binary slices) so browsers
reach the data plane without a gRPC-Web proxy. It serves **no** static assets —
the control plane owns the browser UI and reverse-proxies this sidecar under
`/data_plane/*`.

**Module:** `biopb_tensor_server.serving.http_server` ·
**Factory:** `create_app(flight_location, token, cache_bytes, cors_origins, config_path, web_host, web_port, supervised) → FastAPI` ·
**Default port:** `8814`

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
compares against `expected = self.token`; a `None` token means no enforcement —
this is **local mode**, where every listener binds loopback and no token exists.
A token is present (and enforced) only in **remote mode**, when the config's
`server.host` is a public address. There is no separate dev flag.

## Endpoints

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

> **Route ordering:** `/api/sources/{id}/metadata` is registered *before* the
> greedy `{id:path}` catch-all to avoid Starlette first-match shadowing.

## Slice endpoint

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

Default allowed origins: `http://localhost:5173`, `http://127.0.0.1:5173`,
`http://[::1]:5173` (Vite dev server port). Overridable via the `cors_origins`
argument to `create_app`, or via `--cors` / `--web-url` on the CLI launcher.
