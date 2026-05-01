# tensor-site Client Architecture

## Overview

The browser-facing side of tensor-site is split into two pnpm workspace
packages:

| Package | Purpose |
|---------|---------|
| `@biopb/tensor-flight-client` | HTTP client + lazy array API for the FastAPI sidecar |
| `@biopb/web` | Next.js 14 web application |

Both packages live under `packages/` and are built together with
`pnpm -r run build`.

---

## @biopb/tensor-flight-client

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

## @biopb/web

**Framework:** Next.js 14 App Router  
**State management:** Zustand  
**Rendering:** Pixi.js v8 (WebGL)

### Token gate

`packages/web/middleware.ts` runs on every incoming request:

1. Public paths (`/livez`, `/unlock`, `/_next/*`, `/favicon.ico`) bypass the gate.
2. If a `?token=` query parameter is present, set a `biopb_token` cookie
   (HttpOnly, SameSite=Strict) and redirect to the same path without the param.
3. Validate the `biopb_token` cookie against `BIOPB_WEB_TOKEN` using
   `crypto.timingSafeEqual`.
4. `NEXT_PUBLIC_DEV_MODE=1` + `BIOPB_WEB_DEV_BYPASS=1` bypasses validation.
5. Unauthenticated requests are redirected to `/unlock`.

### /api/token route

`GET /api/token` returns `{ token, apiBase, devMode }`.  
`apiBase` comes from `BIOPB_TENSOR_API` env var (default `http://localhost:8816`).  
Consumed by `ClientBootstrap` on first render.

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
layout.tsx
└── ClientBootstrap          — fetches /api/token, initialises store
    └── page.tsx             — app shell
        ├── Topbar           — title + connection state badge
        ├── SourceTree       — hierarchical source browser (sidebar)
        └── main panel
            ├── ImageViewer  — Pixi.js canvas
            ├── SliceControls — T/Z sliders, channel select
            └── MetaPanel    — OME-NGFF metadata accordion
```

| Component | Key behaviour |
|-----------|--------------|
| `ClientBootstrap` | Fetches `/api/token` on mount; calls `initClient()` then `loadSources()`; renders a dev-mode banner |
| `SourceTree` | Builds a tree from `source_url` path segments; search filter; tensor count badge; calls `selectSource` on click |
| `SliceControls` | Reads `descriptor.shape` + `axisMap` to render per-axis sliders; calls `setSlice()` on change |
| `MetaPanel` | Calls `getSourceMetadata()` on mount; collapsible JSON pretty-print |
| `ImageViewer` | Pixi.js v8 `Application`; pan/zoom via pointer events; `toGrayscaleRgba` uint16→RGBA; `Texture.from(canvas)`; fit-to-viewport on first load; aborts superseded requests |

### Layout

```
┌──────────────────────────────────────────┐
│  Topbar: title + connection state        │
├──────────┬───────────────────────────────┤
│          │                               │
│ Sidebar  │   ImageViewer (Pixi.js WebGL) │
│ Source   │                               │
│ Tree     ├───────────────────────────────┤
│          │   SliceControls               │
│          ├───────────────────────────────┤
│          │   MetaPanel                   │
└──────────┴───────────────────────────────┘
```

---

## Data Flow — Viewing a Slice

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

## Test Coverage

| Package | Runner | Tests |
|---------|--------|-------|
| `@biopb/tensor-flight-client` | vitest | 45 — `buildAxisMap`, `isAxisMapAmbiguous`, `computeScaleHint`, `TensorArray.compute`, `TensorHttpClient` (all methods with mocked `fetch`) |

---

## Environment Variables (client / web)

| Variable | Where consumed | Purpose |
|----------|---------------|---------|
| `BIOPB_WEB_TOKEN` | Next.js middleware | Token the browser cookie is validated against |
| `BIOPB_TENSOR_API` | `/api/token` route | Base URL of the FastAPI sidecar |
| `NEXT_PUBLIC_DEV_MODE` | `ClientBootstrap` | Show dev-mode banner in the UI |
| `BIOPB_WEB_DEV_BYPASS` | Middleware + CLI | Bypass token validation (localhost only) |
| `BIOPB_WORKSPACE_ROOT` | Next.js (future) | Workspace root for relative path display |
