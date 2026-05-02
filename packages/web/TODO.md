# web TODO

## ~~Issue 3~~ — ImageViewer not re-keyed on source/tensor change ✓ FIXED

---

## Issue 8 — Token echoed to browser JS

`/api/token` returns `BIOPB_WEB_TOKEN` to the browser so it can authenticate direct calls
to the FastAPI sidecar.  This is an intentional design trade-off for the current architecture
where the browser contacts the sidecar directly.

**Acceptable only under these conditions:**
- The sidecar port (default 8816) is firewalled / not publicly reachable.
- The Next.js app is deployed behind nginx (or equivalent) with TLS.
- XSS surface is minimal (no user-controlled content rendered as HTML).

**If any condition breaks** (e.g. sidecar port exposed, no TLS), the token must not be sent
to the browser. The alternative is to proxy all `/api/slice` and `/api/sources` calls through
Next.js API routes, keeping the token server-side. This adds ~2–8 ms latency per call and
requires Next.js to stream binary responses, but eliminates the credential-in-JS exposure.

---

## ~~Issue 9~~ — Wheel zoom anchors on top-left, not cursor ✓ FIXED

---

## ~~Issue 5~~ — Double scan / min-max normalisation ✓ FIXED

Replaced min-max with 1%–99% percentile normalization via `computePercentileCutoffs()`.
Systematic sampling (≤65536 values) keeps the pre-pass O(1) in tile size and eliminates
the full-array linear scan; `Float32Array.sort()` without a comparator uses the engine's
fast numeric sort path.

---

## Issue 6 — `coords` array mutation in inner render loop

`toGrayscaleRgba()` allocates `coords = new Array<number>(shape.length).fill(0)` once and
mutates `coords[yIdx]` / `coords[xIdx]` per pixel to compute the flat buffer index.  For the
common 2D/3D case (slice already requested as a 2D YX plane) this is overly general.

A simpler direct computation for the 2D case:
```typescript
const yStride = strides[yIdx] as number;
const xStride = strides[xIdx] as number;
// compute base offset from all non-y/x dims once
let base = 0;
for (let d = 0; d < shape.length; d++) {
  if (d !== yIdx && d !== xIdx) base += 0 * (strides[d] as number); // all non-yx already 0
}
for (let y = 0; y < height; y++) {
  for (let x = 0; x < width; x++) {
    const flat = base + y * yStride + x * xStride;
    ...
  }
}
```
Low priority — only worthwhile if profiling shows this is a bottleneck.
