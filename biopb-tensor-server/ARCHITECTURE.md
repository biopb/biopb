# biopb-tensor-server Architecture

## Overview

`biopb-tensor-server` provides two complementary server components:

1. **TensorFlightServer** — Arrow Flight / gRPC server for chunked array access.
2. **HTTP Sidecar** — FastAPI proxy that translates Flight calls into
   browser-accessible HTTP endpoints.

```
Client (Python or TypeScript)
    │
    ├── Arrow Flight / gRPC  (default :8815)  ─────► TensorFlightServer
    │                                                        │
    └── HTTP/JSON + binary   (default :8816)  ─────► FastAPI Sidecar
                                                             │
                                                   TensorFlightClient
                                                             │
                                                    TensorFlightServer
                                                             │
                                              ┌──────────────────────────┐
                                              │  BackendAdapter           │
                                              │  (Zarr / OME-Zarr /      │
                                              │   OME-TIFF / HDF5 / CZI) │
                                              └──────────────────────────┘
```

The sidecar exists because browsers cannot speak gRPC directly. It wraps the
existing Python `TensorFlightClient` and re-exposes its operations as plain
HTTP so that the Next.js web app (and any other HTTP client) can use it
without a gRPC-Web proxy.

---

## 1. TensorFlightServer

**Module:** `biopb_tensor_server.server`  
**Class:** `TensorFlightServer(flight.FlightServerBase)`  
**Default location:** `grpc://0.0.0.0:8815`

### Registration

```python
server = TensorFlightServer("grpc://0.0.0.0:8815")
server.register_source("my-zarr", ZarrAdapter(arr, "t0", ["z", "y", "x"]))
server.serve()  # blocking
```

Sources are keyed by `source_id`. Each source maps to one `BackendAdapter`
which may expose multiple tensors (e.g., multi-field).

### Flight methods

| Method | Description |
|--------|-------------|
| `ListFlights` | Returns one `FlightInfo` per registered source, embedding a serialised `DataSourceDescriptor` proto |
| `GetFlightInfo` | Returns chunk endpoints for a specific tensor, respecting `SliceHint` and `TensorReadOptions` in the descriptor |
| `DoGet` | Fetches a single chunk identified by a `TensorTicket`; reads from the adapter and returns a `RecordBatch` stream |

### BackendAdapter interface

All adapters implement `BackendAdapter`:

| Method | Returns |
|--------|---------|
| `get_tensor_descriptor()` | `TensorDescriptor` proto |
| `get_chunk_endpoints(ticket)` | List of `ChunkBounds` |
| `read_chunk(bounds)` | `np.ndarray` |

Concrete adapters:

| Adapter | Format |
|---------|--------|
| `ZarrAdapter` | Zarr v2 arrays |
| `OmeZarrAdapter` | OME-Zarr with precomputed pyramid routing |
| `OmeTiffAdapter` | Single-file OME-TIFF |
| `MultiFileOmeTiffAdapter` | Multi-file OME-TIFF / Micro-Manager datasets |
| `Hdf5Adapter` | HDF5 chunked datasets |
| `AicsAdapter` | Vendor formats (CZI, LIF, ND2, DV) via aicsimageio |

### Chunk caching

`CacheManager` provides a pluggable cache layer between `DoGet` and the
adapter. The default backend is an in-process LRU memory cache (`cachey`).
An optional `ArrowFileBackend` persists decoded chunks to disk.

---

## 2. FastAPI HTTP Sidecar

**Module:** `biopb_tensor_server.http_server`  
**Factory:** `create_app(flight_location, token, dev_mode, cache_bytes, cors_origins) → FastAPI`  
**Default port:** `8816`

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

`secrets.compare_digest` is used for timing-safe comparison.  
`dev_mode=True` skips the check entirely (enforced to localhost-only by the
CLI launcher).

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
| `POST` | `/api/slice` | ✓ | Binary tensor sub-region |

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

Default allowed origins: `http://localhost:3000`, `http://127.0.0.1:3000`,
`http://[::1]:3000`. Overridable via the `cors_origins` argument to
`create_app`.

---

## 3. CLI Launcher

**Command:** `biopb-tensor launch`

```
biopb-tensor launch --config biopb-tensor.toml [--web-port 8816] [--web-host 127.0.0.1] [--dev] [--open]
```

Startup sequence:

1. Resolve `dev_mode` (flag or `BIOPB_WEB_DEV_BYPASS` env var). Force off if
   `--web-host` is not a loopback address.
2. Resolve token: `--token` flag → `BIOPB_TENSOR_TOKEN` env var → interactive
   prompt (3 attempts) → `secrets.token_urlsafe(32)` auto-generated.
3. Print the one-time access URL: `http://localhost:3000?token=<value>`.
4. Load `biopb-tensor.toml` config; instantiate adapters and register sources.
5. Start `TensorFlightServer` in a **daemon thread**.
6. Optionally schedule `webbrowser.open(url)` after 1.5 s.
7. Call `run_http_server(...)` — **blocking** uvicorn call.

Token validation rules: 16–128 characters, regex `[A-Za-z0-9_\-]+`.

---

## 4. Configuration (`biopb-tensor.toml`)

```toml
[server]
host = "0.0.0.0"
port = 8815

[cache]
max_bytes = 2_000_000_000   # 2 GB in-process

[[sources]]
source_id  = "my-zarr"
type       = "zarr"
url        = "/data/experiment.zarr"
dim_labels = ["z", "y", "x"]

[[sources]]
source_id  = "ome"
type       = "ome-zarr"
url        = "/data/multiscale.zarr"
```

---

## 5. Test Suite

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

## 6. Environment Variables (server-side)

| Variable | Default | Purpose |
|----------|---------|---------|
| `BIOPB_TENSOR_ENDPOINT` | `grpc://localhost:8815` | Arrow Flight server location used by TensorFlightClient |
| `BIOPB_TENSOR_TOKEN` | — | Pre-set sidecar token (skips CLI prompt) |
| `BIOPB_WEB_DEV_BYPASS` | `0` | Enable dev-mode token bypass (localhost only) |
