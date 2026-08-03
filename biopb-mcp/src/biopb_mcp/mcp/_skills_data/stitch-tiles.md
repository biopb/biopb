---
id: stitch-tiles
title: Register a grid of overlapping tiles into one mosaic
description: Turn a grid of overlapping tiles into a single continuous image, placed by the specimen rather than by the stage's nominal coordinates.
tags: [stitching, mosaic, registration, preprocessing]
version: 1.0.0
checklist: [viewer, tensor, dask, pkg:biopb-mcp>=0.13.0]
---

# Register a grid of overlapping tiles into one mosaic

## When to use

A grid of **overlapping** tiles that has to become one continuous image — to
measure across a seam, or to segment without double-counting every boundary.

The stage's own coordinates are not good enough for either: measured against
tiles cut at known positions, nominal placement leaves an rms of **10 px** and
individual tiles **16–18 px** out. Registering on the overlap brings that to
**0.0–1.0 px**, and often to exactly right.

## When NOT to use

- **The tiles do not overlap, or overlap by less than about 10%.** Measured on
  the same fixture, 12% overlap still registers to under 1 px, 10% fragments the
  grid into 4–5 unregisterable pieces and 8% into 8. Place at nominal, say the
  seams are unregistered, and do not report a computed alignment.
- **The sample deforms between tiles.** Serial sections, folded tissue — anything
  where a tile is not a rigid translation of what it shares. This estimates two
  numbers per pair; elastic registration is a different job.
- **The frames are a time series, not a grid.** A field of view that moved
  between frames is [[drift-correction]]: there the frames are the same region at
  different times, here different regions at the same time.
- **The tiles are already a mosaic.** Vendor software often stitches on export;
  re-registering a mosaic cut back into tiles finds the seams the vendor chose.
- **Absolute stage coordinates are the measurement.** Registration deliberately
  overrides them; if where the stage *was* is the result, keep it.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `TILES` | `(N, Y, X)` | One channel, one resolution level, stacked in a stated acquisition order. Read `guide://data` first — pyramid level and laziness both bite here |
| `GRID` | (rows, cols) | From stage positions in the metadata if they are there; otherwise **ask**. `n_tiles == rows * cols` is a check, not a derivation — 24 tiles is 4×6 or 6×4 |
| `TILE_ORDER` | — | Row-major, or **snake** (alternate rows reversed). Snake read as row-major mirrors every other row. Step 5 does catch it — measured, the accepted pairs fell into 15 pieces instead of 1 — but as "registration failed", so ask rather than diagnose it backwards |
| `OVERLAP` | % of tile width | The nominal value from the acquisition, 10–20% typically. It sets where to look, not where the tile lands |
| `PAD` | px | How far past the nominal overlap to search — the stage's positional error, `32` if nobody knows. Too small misses the true offset; too large costs only time |
| `NCC_MIN` | 0–1 | `0.5`. Pairs below it are rejected. **Not a knob** — see step 4 |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status` — `guide://kernel` covers what to do about a gap. Everything
   below is numpy, `scipy.sparse.csgraph` and
   `skimage.registration.phase_cross_correlation`, all core, so there is no
   package to install and no degraded path to fall back to.

2. **Establish the grid, and budget the blend** *(confirm-input, blocking)*.
   `GRID`, `TILE_ORDER` and `OVERLAP` are not derivable from pixels when the
   metadata carries no stage positions, and all three are silently wrong-able.
   Ask for them together, in the same question as the memory answer below.

   Size the blend in the same breath, since an out-of-memory kill at step 7
   throws the registration away too. The blend holds an accumulator *and* a
   weight canvas, both float32, so budget **8 bytes per output pixel** — not the
   2 a `uint16` suggests — over a tile stack that stays resident. Past half of
   `memory_available` the escape is a **coarser pyramid level**, which is the
   user's call. Registration is never the problem: 180 pairs of 2048 px tiles
   take about 3 s.

3. **Correct the illumination first, with [[flatfield]]** — but not for the
   reason usually given. Registration barely notices shading: under a vignette
   whose corners sit at 15% of the centre it went 0.4 px to 0.7 px, with the same
   pairs accepted. What shading destroys is **step 4's score**, where a correct
   pair falls from 0.94 to 0.68 and a wrong one rises from 0.10 to 0.33 — most of
   the margin step 5's gate runs on. The seams themselves are photometric anyway:
   a brightness step no placement can remove.

4. **Estimate each neighbouring pair.** Correlate the strips where the two tiles
   are *expected* to overlap, not the whole tiles, and score the result on the
   region they then actually share:

   ```python
   def ncc(a, b, dy, dx):
       """Pearson correlation over the region two tiles share at (dy, dx)."""
       H, W = a.shape
       dy, dx = int(round(dy)), int(round(dx))
       pa = a[max(0, dy):min(H, H + dy), max(0, dx):min(W, W + dx)]
       pb = b[max(0, -dy):min(H, H - dy), max(0, -dx):min(W, W - dx)]
       if pa.size < 256:
           return -1.0            # too little shared area to mean anything
       pa, pb = pa - pa.mean(), pb - pb.mean()
       return float((pa * pb).sum() / np.sqrt((pa**2).sum() * (pb**2).sum()))

   def pair_offset(a, b, axis, overlap_px, pad=32):
       """(dy, dx) from tile `a` to its right (axis=1) or lower (axis=0) neighbour."""
       from skimage.registration import phase_cross_correlation

       H, W = a.shape
       s = min(overlap_px + pad, W if axis == 1 else H)
       if axis == 1:
           ref, mov, base = a[:, W - s:], b[:, :s], np.array([0.0, W - s])
       else:
           ref, mov, base = a[H - s:, :], b[:s, :], np.array([H - s, 0.0])
       shift, _, _ = phase_cross_correlation(ref, mov, normalization=None)
       d = np.asarray(shift, float) + base
       return d, ncc(a, b, *d)
   ```

   **`normalization=None` is not optional.** skimage's default is `"phase"`,
   which whitens every frequency bin and buries the peak on ordinary microscopy
   content: it takes the share of pairs correctly registered from **92% to 68%**,
   and on whole tiles from 90% to 18%. Cropping to the expected strip is what
   keeps that peak unambiguous, where a full-tile correlation wraps.

   **Score the pair separately, and never reuse `phase_cross_correlation`'s
   second return value as that score** — it describes the correlation, not the
   alignment, and does not separate a registered pair from an unregistered one.
   The correlation on the overlap separates them completely: correct pairs score
   **≥ 0.93** and wrong ones **≤ 0.13**. So `NCC_MIN` is not a knob — anything
   from 0.2 to 0.8 gives the same answer, and a value that changes the result is
   telling you about step 3.

5. **Gate on the graph, not on the count** *(blocking)*. Whether this mosaic is
   trustworthy is one number, it is available before anything is composed, and it
   is not the fraction of pairs accepted:

   ```python
   from scipy.sparse import coo_matrix
   from scipy.sparse.csgraph import connected_components

   i = np.array([e[0] for e in edges])   # edges: (i, j, score, offset), accepted only
   j = np.array([e[1] for e in edges])
   n_parts, part = connected_components(
       coo_matrix((np.ones(len(edges)), (i, j)), shape=(n, n)), directed=False)
   ```

   **`n_parts == 1` and every tile is registered; above 1 and the mosaic is
   not.** Across the whole sweep — overlap from 8% to 25%, heavy noise, blank
   tiles, uncorrected shading — one component gave 0.0–1.1 px every time and more
   than one gave 3.7–17.9 px every time, with no exception either way. The
   accepted *fraction* does not separate them: 55% accepted failed, 75% passed.

   So if `n_parts > 1`, **stop and report it**, naming which tiles are in which
   piece. Blending the stray pieces at nominal anyway produces a mosaic no better
   than not registering, while looking like it was.

6. **Compose the accepted pairs into one layout.** Do **not** accumulate offsets
   along the acquisition path, which is the obvious thing to write and is worse
   than not registering at all — one bad pair shifts everything downstream of it,
   for an rms of 11.9 px against nominal placement's 10.3. Take a maximum
   spanning tree over the scores instead, and walk it:

   ```python
   from scipy.sparse import coo_matrix
   from scipy.sparse.csgraph import minimum_spanning_tree

   def compose(n, edges):   # edges: (i, j, score, offset) for accepted pairs only
       i, j, w, d = (np.array([e[k] for e in edges]) for k in range(4))
       tree = minimum_spanning_tree(coo_matrix((-w, (i, j)), shape=(n, n))).tocoo()
       adj, where = {k: [] for k in range(n)}, {ab: k for k, ab in enumerate(zip(i, j))}
       for a, b in zip(tree.row, tree.col):
           k = where[(a, b)]
           adj[a].append((b, d[k]))
           adj[b].append((a, -d[k]))
       pos, seen, stack = np.zeros((n, 2)), {0}, [0]
       while stack:
           u = stack.pop()
           for v, step in adj[u]:
               if v not in seen:
                   seen.add(v)
                   pos[v] = pos[u] + step
                   stack.append(v)
       return pos - pos.min(0)
   ```

   This is the global step from **MIST**, and it is the whole of it — the
   published method's own composition rule, not an approximation of it. It
   assumes step 5 passed: a tile the tree cannot reach keeps the position it was
   initialised with, silently.

   **The usual argument against the tree is backwards, so do not be talked out
   of it.** Least squares over every accepted pair is the other obvious route,
   preferred on the grounds that a tree accumulates error along its path — which
   is not what happens. The tree composes offsets that are individually *exact*,
   so nothing accumulates; least squares averages them into fractional positions
   that then round the wrong way. At 20% overlap the tree puts **100%** of tiles
   on the correct pixel and least squares 52–92%. Regularising it toward the
   nominal grid, the other common addition, only pulls the answer back toward the
   stage error being corrected for.

7. **Blend, and publish what was applied.** Weight each pixel by its distance to
   its own tile's edge, as a **separable product** — an `np.minimum` of the two
   distances collapses at tile corners, for a 9.9% step there against 0.3%:

   ```python
   dy = np.minimum(np.arange(H), H - 1 - np.arange(H)) + 1.0
   dx = np.minimum(np.arange(W), W - 1 - np.arange(W)) + 1.0
   w = (dy[:, None] * dx[None, :]).astype(np.float32)

   acc = np.zeros((Hc, Wc), np.float32)
   wsum = np.zeros((Hc, Wc), np.float32)
   for i, (y, x) in enumerate(pos.astype(int)):
       acc[y:y + H, x:x + W] += TILES[i] * w
       wsum[y:y + H, x:x + W] += w
   mosaic = np.divide(acc, wsum, out=acc, where=wsum > 0)   # in place: no third canvas
   ```

   Then upload it in the input's dtype and print what reproduces it. Pixel
   spacing does not ride along by default (`guide://data`), and a mosaic that
   lost it measures in pixels while looking exactly the same.

   ```python
   print({"grid": GRID, "tile_order": TILE_ORDER, "overlap_pct": OVERLAP,
          "accepted": len(edges) / n_pairs, "components": n_parts,
          "max_dev_from_nominal_px": float(np.abs(pos - nominal).max()),
          "method": "strip phase correlation + NCC + maximum spanning tree"})
   ```

## Guardrails

- **A mosaic too large to hold is too large to `add_image`.** Upload it and
  `add_tensor` the returned id, so the viewer reads it back lazily.
- **Blending is cosmetic and nothing more.** Feathering does not rescue a bad
  placement and does not hide one: against tiles put 6 px wrong, every blend
  choice changed the mosaic by the same ~14%. Step 5 is the check; the blend is
  not.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Step 5 reports many components on tiles that plainly overlap | Snake acquisition read as row-major, so the "neighbours" being correlated are not adjacent | Reverse the odd rows when building the stack, then re-run from step 4 |
| Mosaic is a diagonal staircase, or tiles pile into one corner | `rows`/`cols` transposed | Step 2 — `n_tiles == rows * cols` does not catch a square grid |
| Nearly every pair rejected, on tiles that plainly overlap | `normalization` left at skimage's `"phase"` default | Pass `normalization=None` (step 4) |
| Pairs rejected along one axis only | `PAD` smaller than the stage's error on that axis | Raise `PAD`; it costs time, not accuracy |
| `n_parts > 1`, and raising `PAD` does not help | Genuinely too little overlap, or blank tiles with nothing to correlate | Below ~10% overlap this is not recoverable — report nominal placement as unregistered |
| Offsets are believable but the mosaic is blurred at every seam | Registered on uncorrected tiles, so shading dragged the correlation | Step 3 — correct first, then re-register |
| Seams show as brightness steps, with the pixels aligned | Illumination was never corrected; placement is fine | [[flatfield]] on the tiles, then re-blend — no need to re-register |
| One tile is far off and everything after it follows | Offsets accumulated along the acquisition path instead of composed | Step 6 |

## Next steps

- Segment the mosaic as one image — the usual point of stitching, and what
  removes the duplicate-objects-at-seams problem entirely. Score that with
  [[segmentation-qc-metrics]], and report sizes from it with
  [[calibrated-measurements]], carrying the spacing across (step 7).
- The settings printed at step 7 are the fixed configuration for the rest of a
  plate or slide set. The illumination fit that preceded them is not — that is
  per channel and per session.
