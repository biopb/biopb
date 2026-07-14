# biopb `web/` — front-end architecture

Internals of the browser front end. For the workspace layout, the surfaces/routes,
and the dev/build/test commands, see `README.md`; this doc covers the TypeScript
data-plane SDK (`@biopb/tensor-flight-client`), how a slice reaches the screen, and
the SPA's token/store model.

## `@biopb/tensor-flight-client` — the data-plane SDK

ESM output (`dist/index.js` + `.d.ts`). The browser's HTTP client + lazy array API
for the tensor server's FastAPI data API — reached **same-origin** through the
control plane at `/data_plane/*` (`VITE_TENSOR_API=/data_plane`).

### TensorHttpClient

Low-level wrapper over the REST API. `new TensorHttpClient(apiBase, token)`.

- Injects `Authorization: Bearer <token>` on every request when a token is present.
- Per-call timeouts: **3 s** for metadata/listing, **8 s** for binary chunks.
- Non-OK responses throw `TensorApiError(status, message, detail)`.

| Method | Endpoint |
|--------|----------|
| `livez()` / `readyz()` | `GET /livez` / `GET /readyz` |
| `listSources()` | `GET /api/sources` |
| `getSource(id)` / `getSourceMetadata(id)` | `GET /api/sources/{id}[/metadata]` |
| `slice(req)` | `POST /api/slice` |
| `diagnostics()` | `GET /api/diagnostics` |

`slice()` parses the binary response headers (`X-Shape`, `X-Dtype`, `X-Dim-Labels`)
into a `TypedNdArray { buffer, shape, dtype, dimLabels }` (C-contiguous numpy bytes).

### TensorArray — lazy per-tensor accessor

`new TensorArray(client, sourceId, descriptor)`. On `.compute(options)`: build
per-axis `[start, stop)` ranges from `SliceOptions` (scalar → single index,
`[start, stop]` → range, `undefined` → full extent), clamp to `[0, shape[axis])`,
and send a `SliceRequest`.

### Axis mapping

`buildAxisMap(dimLabels)` derives an `AxisMap` (`t | z | c | y | x → index`) from
explicit labels, with a positional fallback for unrecognized labels:

| Axis | Recognized labels |
|------|------------------|
| `t`  | t, time, frame, frames |
| `z`  | z, depth, plane, planes, slice |
| `c`  | c, channel, channels, band, bands |
| `y`  | y, height, row, rows |
| `x`  | x, width, col, cols, column, columns |

Fallback: last unassigned dim → X, second-last → Y, third-last → Z, …
`isAxisMapAmbiguous(dimLabels)` returns `true` when any label used the fallback;
the app surfaces a warning in that case.

### computeScaleHint

```ts
computeScaleHint(tensorShape, axisMap, viewportW, viewportH, pixelBudget = 1_000_000, viewportZoom = 1)
  → ScaleVector { factors: number[], snapped: boolean }
```

A **pure** function — no `prevFactors`, no hysteresis band (that render-loop
gesture-stability concern lives entirely in the web layer now). It selects a
power-of-two downsample factor for the Y/X axes:

1. Pixel-budget ceiling: `maxScale = max(1, sqrt(dataH·dataW / pixelBudget))`.
2. `effectiveTargetScale = max(1 / viewportZoom, 1)` — zoomed in (`viewportZoom > 1`)
   means a smaller scale, i.e. more detail.
3. Snap that scale to the nearest power of two; non-spatial axes stay at `1`.

`snapped` is `true` when snapping changed the factor.

### TensorFlightClient

Higher-level facade over `TensorHttpClient`: caches the source list and returns
`LazyTensorArray` instances (wrapping `TensorArray`). Also exports
`validateConfig()`, the client-side mirror of the server's `PUT /api/config`
validation used by the admin page.

## Rendering a slice — the data flow

```
User moves Z slider
  → setSlice({ z: 5 }) [Zustand]
  → SliceControls re-renders (controlled input)
  → viewer effect fires (deps: activeSourceId, activeTensorId, slice)
      → TensorArray.compute({ z: 5, scaleHint, reductionMethod })
          → TensorHttpClient.slice(SliceRequest)
              POST /api/slice → control /data_plane proxy → FastAPI sidecar → Flight server
              ← octet-stream + X-Shape / X-Dtype / X-Dim-Labels
          ← TypedNdArray { buffer, shape, dtype, dimLabels }
      → toGrayscaleRgba(buffer, shape, dtype)   [e.g. uint16 → Uint8ClampedArray RGBA]
      → ImageData → canvas → Pixi.js Texture → Sprite → stage
```

A **request counter** guards against races: if a newer request starts before the
previous one resolves, the stale response is discarded.

## `@biopb/web` — SPA internals

Vite + React + React Router v6, Zustand state, Pixi.js v8 (WebGL). The routes, the
surfaces table, and the build/serve model are in `README.md`; the architectural
notes that aren't there:

- **Token / auth flow.** The token lives in `sessionStorage` under `biopb_token`
  (clears on tab close, never persisted to disk). On load `ClientBootstrap` reads
  it (`auth.ts`); absent → redirect to `/unlock`, which stores the pasted token and
  initializes the client. Every HTTP request then carries `Authorization: Bearer`;
  the Flight server (`BearerAuthMiddlewareFactory`) and the sidecar (`HTTPBearer`)
  validate the *same* token. In local mode (loopback, no token) the unlock step is
  skipped.
- **Store (Zustand, `store.ts`).** Holds the `TensorHttpClient` + connection state,
  the source list (plus a `scanning` flag seeded from `/readyz` `backend_health` so
  the UI can distinguish "indexing" from "empty" during progressive discovery — see
  `../biopb-tensor-server/docs/progressive-discovery.md`), the active source/tensor,
  and the slice selection (`t`/`z`/`c`, `scaleFactors`, `reductionMethod`). Actions:
  `initClient` / `loadSources` / `selectSource` / `setSlice` / `clearSession`.
- **Pages** (`packages/app/src/pages/`): `DashboardPage`, the dataviewer
  (`HomePage` / `ViewerLayout`), `AdminPage`, `McpAdminPage`, `UnlockPage`,
  `ObservePage` — wired to routes in `main.tsx` (see `README.md` for the
  route → surface map).

## Client tests

`packages/tensor-flight-client/` (vitest): `buildAxisMap`, `isAxisMapAmbiguous`,
`computeScaleHint`, `TensorArray.compute`, `TensorHttpClient` (all methods with a
mocked `fetch`), and `validateConfig`.
