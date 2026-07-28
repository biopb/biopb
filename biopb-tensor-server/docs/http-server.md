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

**Enforcement is independent of the network mode.** The config's `server.host`
decides the *bind* (loopback = local, public = remote), not whether a token
exists. Remote mode **requires** one (auto-generated if not supplied — a public
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
| `GET` | `/api/admin/browse` | ✓ | Filesystem browse for the data-folder picker (local only — see the auth caveat) |
| `WS` | `/ws/render` | ✓ | Streaming render: JSON `{action:"render", params}` in, `render_start` metadata + binary image out, repeatable |

> **Route ordering:** `/api/sources/{id}/metadata` and `/ticket/{ticket_hex}` are
> registered *before* the greedy `{source_id:path}` catch-all to avoid Starlette
> first-match shadowing.

`/ws/render` takes its token from the `Authorization` / `X-Biopb-Token` header
**or a `token` query parameter**, since browsers cannot set custom headers on a
WebSocket handshake; an unauthorized socket is closed with code `4001`. It holds
no session state — each render is an independent request/response.

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
