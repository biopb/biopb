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
                                              │  BackendAdapter          │
                                              │  (Zarr / OME-Zarr /      │
                                              │   OME-TIFF / HDF5 / CZI) │
                                              └──────────────────────────┘
```

The sidecar exists because browsers cannot speak gRPC directly. It wraps the
existing Python `TensorFlightClient` and re-exposes its operations as plain
HTTP so that the Vite web app (and any other HTTP client) can use it
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

Default allowed origins: `http://localhost:5173`, `http://127.0.0.1:5173`,
`http://[::1]:5173` (Vite dev server port). Overridable via the `cors_origins`
argument to `create_app`, or via `--cors` / `--web-url` on the CLI launcher.

---

## 3. CLI Launcher

**Command:** `biopb-tensor launch`

```
biopb-tensor launch --config biopb-tensor.toml [--web-port 8816] [--web-host 127.0.0.1] [--dev] [--open] [--web-url URL] [--cors ORIGIN]

# for grpc only (no sidecar)
biopb-tensor serve ...
```

Startup sequence:

1. Resolve `dev_mode` (flag or `BIOPB_WEB_DEV_BYPASS` env var). Force off if
   `--web-host` is not a loopback address.
2. Resolve token: `--token` flag → `BIOPB_TENSOR_TOKEN` env var → interactive
   prompt (3 attempts) → `secrets.token_urlsafe(32)` auto-generated.
3. Print the one-time access token.
4. Load `biopb-tensor.toml` config; instantiate adapters and register sources.
5. Start `TensorFlightServer` in a **daemon thread**.
6. Derive CORS origins from `--web-url` (default `http://localhost:5173`) or
   explicit `--cors` flags; optionally schedule `webbrowser.open(--web-url)`.
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
url        = "/data/" # triggers recursive discovery

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

## 5. Client Packages (TypeScript)

The browser-facing side of biopb-tensor-server is split into two pnpm workspace
packages:

| Package | Purpose |
|---------|---------|
| `@biopb/tensor-flight-client` | HTTP client + lazy array API for the FastAPI sidecar |
| `@biopb/web` | Vite + React static web application |

Both packages live under `packages/` and are built together with
`pnpm -r run build`.

---

## 6. @biopb/tensor-flight-client

**Output:** ESM (`dist/index.js`, `dist/index.d.ts`)

### TensorHttpClient

Low-level HTTP wrapper around the sidecar's REST API.

```ts
const client = new TensorHttpClient("http://localhost:8816", token);
```

Internals:
- Injects `Authorization: Bearer <token>` on every request when a token is
  present.
- Per-call timeouts: **3 s** for metadata/listing, **8 s** for binary chunks.
- All non-OK responses throw `TensorApiError(status, message, detail)`.

| Method | Endpoint |
|--------|----------|
| `livez()` | `GET /livez` |
| `readyz()` | `GET /readyz` |
| `listSources()` | `GET /api/sources` |
| `getSource(id)` | `GET /api/sources/{id}` |
| `getSourceMetadata(id)` | `GET /api/sources/{id}/metadata` |
| `slice(req)` | `POST /api/slice` |
| `diagnostics()` | `GET /api/diagnostics` |

`slice()` parses the binary response headers (`X-Shape`, `X-Dtype`,
`X-Dim-Labels`) and returns a `TypedNdArray`:

```ts
interface TypedNdArray {
  buffer:    ArrayBuffer;   // C-contiguous numpy bytes
  shape:     number[];
  dtype:     string;        // e.g. "uint16", "float32"
  dimLabels: string[];      // e.g. ["z", "y", "x"]
}
```

### TensorArray

Lazy accessor for a single tensor within a data source.

```ts
const ta = new TensorArray(client, sourceId, descriptor);
const data = await ta.compute({ z: 3, scaleHint: [1, 2, 2], reductionMethod: "area" });
```

On `.compute(options)`:
1. Builds per-axis `[start, stop)` ranges from `SliceOptions` (scalar → single
   index, `[start, stop]` → range, `undefined` → full extent).
2. Clamps ranges to `[0, shape[axis])`.
3. Assembles and sends a `SliceRequest`.

#### Axis mapping

`buildAxisMap(dimLabels)` derives an `AxisMap` (`t | z | c | y | x → index`)
from explicit labels with a positional heuristic fallback for unknown labels:

| Axis | Recognized labels |
|------|------------------|
| `t`  | t, time, frame, frames |
| `z`  | z, depth, plane, planes, slice |
| `c`  | c, channel, channels, band, bands |
| `y`  | y, height, row, rows |
| `x`  | x, width, col, cols, column, columns |

Fallback (when a label is not in any set): last unassigned dim → X,
second-last → Y, third-last → Z, etc.

`isAxisMapAmbiguous(dimLabels)` returns `true` when any label triggered the
fallback; the web app surfaces a warning in that case.

### computeScaleHint

```ts
computeScaleHint(tensorShape, axisMap, viewportW, viewportH, pixelBudget?, prevFactors?)
  → ScaleVector { factors: number[], snapped: boolean }
```

Selects power-of-two downsampling factors for the Y/X axes:

1. Compute `rawScale = max(dataH/viewportH, dataW/viewportW)`.
2. Apply pixel-budget ceiling: `budgetFactor = sqrt(dataH×dataW / budget)`.
3. `targetScale = max(rawScale, budgetFactor, 1)`.
4. Snap to nearest power of two.
5. Apply 20 % hysteresis: if the new factor is within ±20 % of `prevFactors`,
   keep the previous factor to avoid oscillation at scale boundaries.

All non-spatial axes remain at `1`.

### TensorFlightClient

Higher-level facade over `TensorHttpClient` that caches the source list and
returns `LazyTensorArray` instances (wrapping `TensorArray`).

---

## 7. @biopb/web

**Framework:** Vite + React + React Router v6
**State management:** Zustand
**Rendering:** Pixi.js v8 (WebGL)

`@biopb/web` is a static Vite + React frontend for the BioPB TensorFlight viewer.

Key responsibilities:
- serve the browser UI (pure static files — no Node.js at runtime)
- gate access via a bearer token stored in `sessionStorage`
- initialize the client-side TensorFlight HTTP client
- call the FastAPI sidecar (`VITE_TENSOR_API`, default `http://localhost:8816`) directly from the browser

### Build

```sh
pnpm run build   # tsc + vite build → dist/
pnpm run dev     # vite dev server (HMR)
```

Output is `dist/` — plain HTML/CSS/JS, ready for nginx or any static file server.

From repo root:
```bash
pnpm --filter @biopb/web dev     # Vite dev server on :5173
pnpm --filter @biopb/web build   # tsc + vite build → dist/
```

### nginx deployment

Because React Router uses the HTML5 History API, nginx must fall back to `index.html` for all routes:

```nginx
location / {
    root /path/to/dist;
    try_files $uri $uri/ /index.html;
}
```

The sidecar (`VITE_TENSOR_API`) must be reachable from the browser — configure CORS origins (default to localhost:5173) in the tensor server accordingly. I.e, for nginx deployment:

```
biopb-tensor launch config.toml --web-url https://yourdomain.com --token mytoken...
```

### Auth flow

Token is stored in `sessionStorage` under key `biopb_token`.

1. On load, `ClientBootstrap` reads `sessionStorage.getItem("biopb_token")`.
2. If absent → redirect to `/unlock`.
3. `/unlock` page: user pastes token → `sessionStorage.setItem("biopb_token", token)` → navigate to `/`.
4. `ClientBootstrap` calls `initClient(apiBase, token)` → `TensorFlightClient` sends `Authorization: Bearer <token>` on every HTTP request to the sidecar.
5. The Arrow Flight server validates the same token via `BearerAuthMiddlewareFactory`; the FastAPI sidecar validates it via `HTTPBearer`.
6. "Lock" button → `sessionStorage.removeItem("biopb_token")` → redirect to `/unlock`.

Token is stored in `sessionStorage` (clears on tab close, not persisted across sessions).

### Zustand store

```ts
{
  client:          TensorHttpClient | null,
  connectionState: "idle" | "connecting" | "connected" | "error",
  sources:         DataSourceDescriptor[],
  activeSourceId:  string | null,
  activeTensorId:  string | null,
  slice: {
    t: number,
    z: number,
    c: number,
    scaleFactors:    number[] | null,
    reductionMethod: string,
  },
}
```

Actions: `initClient`, `loadSources`, `selectSource`, `setSlice`, `clearSession`.

### Component tree

```
main.tsx  (BrowserRouter + Routes)
├── /          → HomePage
│   └── ClientBootstrap          — reads sessionStorage token, initialises store
│       ├── SourceTree           — hierarchical source browser (sidebar)
│       ├── ImageViewer          — Pixi.js canvas
│       ├── SliceControls        — T/Z sliders, channel select
│       └── MetaPanel            — OME metadata accordion
└── /unlock    → UnlockPage      — token entry form
```

### File map

- `packages/web/package.json`
- `packages/web/vite.config.ts`
- `packages/web/src/main.tsx`
- `packages/web/src/ClientBootstrap.tsx`
- `packages/web/src/store.ts`
- `packages/web/src/pages/HomePage.tsx`
- `packages/web/src/pages/UnlockPage.tsx`
- `packages/web/src/components/`

---

## 8. Data Flow — Viewing a Slice

```
User moves Z slider
  → setSlice({ z: 5 }) [Zustand]
  → SliceControls re-renders (controlled input)
  → ImageViewer effect fires (deps: activeSourceId, activeTensorId, slice)
      → TensorArray.compute({ z: 5, scaleHint, reductionMethod })
          → TensorHttpClient.slice(SliceRequest)
              POST /api/slice  →  FastAPI sidecar  →  Flight server
              ← octet-stream + X-Shape / X-Dtype / X-Dim-Labels
          ← TypedNdArray { buffer, shape, dtype, dimLabels }
      → toGrayscaleRgba(buffer, shape, dtype)   [uint16 → Uint8ClampedArray RGBA]
      → new ImageData(rgba, w, h)
      → HTMLCanvasElement.putImageData(…)
      → Pixi.js Texture.from(canvas) → Sprite → stage.addChild
```

A request counter guards against race conditions: if a new request starts
before the previous one resolves, the stale response is discarded.

---

## 9. Test Suite

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

### Client tests

**Location:** `packages/tensor-flight-client/`
**Runner:** vitest

| Package | Tests |
|---------|-------|
| `@biopb/tensor-flight-client` | 45 — `buildAxisMap`, `isAxisMapAmbiguous`, `computeScaleHint`, `TensorArray.compute`, `TensorHttpClient` (all methods with mocked `fetch`) |

---

## 10. Environment Variables

| Variable | Where consumed | Purpose |
|----------|---------------|---------|
| `BIOPB_TENSOR_ENDPOINT` | TensorFlightClient (Python) | Arrow Flight server location (default `grpc://localhost:8815`) |
| `BIOPB_TENSOR_TOKEN` | `biopb-tensor launch` (server) | Pre-set sidecar token (skips CLI prompt) |
| `BIOPB_WEB_DEV_BYPASS` | `biopb-tensor launch` (server) | Enable dev-mode token bypass (localhost only) |
| `VITE_TENSOR_API` | `ClientBootstrap` (build-time) | Base URL of the FastAPI sidecar (default `http://localhost:8816`) |

---

## 11. Security Model

- Token is stored in `sessionStorage` (clears on tab close, never persisted to disk).
- The FastAPI sidecar validates `Authorization: Bearer <token>` on every request via `HTTPBearer`.
- The Arrow Flight server validates the same token via `BearerAuthMiddlewareFactory`.
- Dev mode (`biopb-tensor launch --dev`) disables token enforcement on the backend; the frontend still shows the unlock page but any string is accepted.
- Error messages are redacted before logging/storage (filesystem paths and potential tokens replaced with `[REDACTED]`).