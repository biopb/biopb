---
id: flatfield-and-stitch-tiles
title: Correct illumination and stitch a tile grid into one mosaic
description: Turn a grid of overlapping tiles into a single seamless image, correcting uneven illumination before registering them.
tags: [stitching, illumination, preprocessing]
version: 1.0.0
requires: [viewer, tensor, dask, pkg:biopb-mcp>=0.13.0, pkg:basicpy, pkg:m2stitch]
---

# Correct illumination and stitch a tile grid into one mosaic

## When to use

A grid of overlapping tiles needs to become one continuous image — measuring
across a seam, or segmenting without double-counting at every boundary.

**Correct the illumination first, then register.** Vignetting darkens exactly the
tile edges the overlap correlation reads, so uncorrected tiles align on the
shading gradient rather than on the specimen.

## When NOT to use

- **A single field.** With no tiles there is nothing to register. Illumination
  correction alone still applies — fit the same estimator on whatever image
  collection is available, and skip everything from step 5 on.
- **Tiles that do not overlap.** Nothing to correlate; place them at the nominal
  grid and say the seams are unregistered. Do not report a computed alignment.
- **Already-stitched or vendor-corrected data.** Vendors often apply shading
  correction on export. A second flat-field fit on corrected tiles inverts the
  vignette instead of removing it. Check for an existing correction, and for a
  flat-field estimate that comes back essentially flat (step 4).
- **Serial sections or deforming samples.** Rigid translation cannot compose
  sections that differ in shape; this is elastic registration, a different job.
- **Quantification the correction would invalidate.** Flat-field division changes
  intensity values. That is *required* for comparing intensities across the field,
  but if absolute values from a calibrated detector are the measurement, correct a
  copy and keep the raw tiles.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `TILES` | image stack | The tiles as one `(N, H, W)` stack per channel — one source with a tile axis, or many sources gathered in acquisition order (step 2) |
| `GRID` | (rows, cols) | From the source metadata if it carries stage positions; otherwise from the tile count and the user (confirm-input). `n_tiles == rows*cols` is a check, not a derivation — 24 tiles is 4×6 or 6×4 |
| `TILE_ORDER` | — | Row-major or **snake** (boustrophedon, alternate rows reversed). Snake read as row-major mirrors every other row — obvious in the mosaic, invisible in the numbers |
| `OVERLAP` | % of tile width | Nominal value from the acquisition, typically 10–20%. Used as the search prior and to sanity-check the result, not as the final placement |
| `FF_CHANNEL` | — | One structural channel with signal across the whole field. Fit the flat-field per channel; never reuse one channel's field on another |
| `NCC_THRESHOLD` | 0–1 | `0.5` default. Pairs below it are rejected and fall back to the nominal offset. The **fraction rejected** is the quality signal to report |
| `GET_DARKFIELD` | bool | `True` when the camera has a measurable offset (most sCMOS). Adds the additive term; without it an offset is absorbed into the multiplicative field and over-corrects dark regions |

Read `guide://data` before pulling pixels off them — pyramid level and laziness
both bite here.

Flat-field estimation is a fit over the tile *collection*, so it needs enough
tiles with varied content — roughly 20+. With fewer, or with the same structure in
every tile, the estimator cannot separate specimen from illumination and will fit
the specimen. Say so rather than reporting a field you do not trust.

## Steps

1. **Check the requirements** *(blocking)*. Resolve `requires:` against
   `server_status`, and `import basicpy` / `import m2stitch` for the two `pkg:`
   tokens — `guide://kernel` covers what to do about a gap.

   Both are optional, and on a one-off run the degraded path usually beats waiting
   for an install — but name it, and carry the name to step 8, because a mosaic
   does not say on its face how it was made:

   - **No `basicpy`** → `FLATFIELD_METHOD = "smoothed-median"`, the smoothed
     per-pixel median across tiles. Valid only where specimen coverage is roughly
     uniform; otherwise it fits the specimen.
   - **No `m2stitch`** → `PLACEMENT = "nominal-grid"`, tiles at `OVERLAP`. Wrong
     by the stage's positional error, so seams stay visible.

2. **Inventory the tiles and establish the grid** *(confirm-input, blocking)*.
   Resolve `TILES` — one source with a tile axis, or many gathered into a stack,
   all from the same resolution level and assembled in a stated order rather than
   whatever the catalog listed. State `GRID`, `TILE_ORDER`, and `OVERLAP` and get
   them confirmed — with no stage positions in the metadata none of the three is
   derivable, and all three are silently wrong-able. Confirm `FF_CHANNEL` in the
   same question.

   Then size the job before spending it. The canvas follows from the grid; the
   footprint does not — the blend needs two float32 canvases, 8 bytes per output
   pixel and not the 2 a `uint16` suggests, over a stack materialized twice, for
   the fit and for what `transform` returns:

   ```python
   N, H, W = TILES.shape
   rows, cols = GRID
   Hc = rows * H - (rows - 1) * int(H * OVERLAP / 100)
   Wc = cols * W - (cols - 1) * int(W * OVERLAP / 100)
   canvas_gb = 2 * Hc * Wc * 4 / 1e9      # acc + wsum
   stack_gb  = 2 * N * H * W * 4 / 1e9    # the fit's stack, and transform's output
   print(f"canvas ~{Hc}x{Wc}  ->  canvas {canvas_gb:.1f} GB + stack {stack_gb:.1f} GB")
   ```

   Above **half** of `memory_available` from `server_status` — the rest is the
   viewer and numpy's temporaries — **say plainly that this will likely run the
   kernel out of memory**, and say *which term* is the problem, because the two
   have different escapes and only one of them costs anything:

   - **Stack-dominated** → fit on a subsample (step 3). The flatfield is one
     tile-shaped answer for the whole collection, so it does not need every tile
     — but it does need enough varied ones, and `transform` still runs on all of
     them, so this halves the stack term rather than removing it.
   - **Canvas-dominated** → resize the tiles: a coarser pyramid level, or a
     resize. Half resolution is a quarter of the canvas. Subsampling the fit
     cannot help here, and the user may well prefer a smaller mosaic to nothing.

   Ask in the same question as the grid, because the fit and the registration are
   the expensive part and an OOM at step 7 throws both away.

3. **Fit the illumination model on the tile stack.** Signatures differ across
   versions — `inspect_object("BaSiC.fit")` before relying on this call.

   `BaSiC` fits in memory on a `(N, H, W)` float stack of one channel, so this is
   where the tiles are materialized and the step-2 budget starts being spent. Fit
   on a representative subsample if the stack is large; apply to all of it.

   ```python
   from basicpy import BaSiC
   basic = BaSiC(get_darkfield=GET_DARKFIELD)
   basic.fit(TILES)
   flat, dark = basic.flatfield, basic.darkfield
   ```

4. **Visual check** *(non-blocking)* — the flat-field itself, before it is
   applied. A valid field is **smooth and low-contrast**: report
   `flat.max()/flat.min()` (expect roughly 1.1–2.0), and layer `flat` to the
   viewer. Two failures are visible here and nowhere later:

   - Ratio near 1.0 → nothing to correct, or the data was already corrected.
     Skip the correction rather than applying a no-op that adds noise.
   - Specimen structure visible in the field → the fit absorbed the sample. Fit
     on more or more varied tiles, or fall back to the smoothed median.

   Then `corrected = basic.transform(TILES)`, and report the drop in tile-to-tile
   median spread — the number that says the correction worked.

5. **Register the overlaps on the corrected tiles.** `rows` and `cols` are each
   tile's **grid indices**, one entry per tile, in the same order as the stack —
   this is where a snake acquisition must already have been unwound:

   ```python
   import m2stitch
   result_df, _ = m2stitch.stitch_images(
       corrected.astype(np.uint16), rows, cols, ncc_threshold=NCC_THRESHOLD,
       row_col_transpose=False)   # the default is True: it swaps them for you
   ys = (result_df["y_pos"] - result_df["y_pos"].min()).to_numpy()
   xs = (result_df["x_pos"] - result_df["x_pos"].min()).to_numpy()
   ```

6. **Validate-and-gate** *(blocking)* before the full-resolution blend — the
   expensive, hard-to-walk-back step. Report `ACCEPTED_FRAC` and the largest
   deviation between a computed offset and the nominal one. A deviation of many
   tile-widths is a failed registration rather than a remarkable stage, and a
   mosaic built from 40% rejected pairs is nominal placement wearing a computed
   alignment's clothes.

   Then recompute the canvas from the offsets actually found —
   `Hc, Wc = int(ys.max()) + H, int(xs.max()) + W`, which is what step 7
   allocates. It should land within a few pixels of the step-2 estimate; much
   larger means registration scattered the tiles, and blending it allocates the
   canvas that failure implies rather than the mosaic that was budgeted.

7. **Feather-blend into the mosaic.** Averaging the overlap — the obvious thing
   once a stitcher has given you positions — leaves a visible ridge wherever the
   tiles disagree. Weight each pixel by its distance to its own tile's edge
   instead:

   ```python
   H, W = corrected.shape[-2:]
   wy = np.minimum(np.arange(H), H - 1 - np.arange(H)) + 1.0
   wx = np.minimum(np.arange(W), W - 1 - np.arange(W)) + 1.0
   w  = np.minimum(wy[:, None], wx[None, :]).astype(np.float32)

   acc, wsum = np.zeros((Hc, Wc), np.float32), np.zeros((Hc, Wc), np.float32)
   for i, (y, x) in enumerate(zip(ys, xs)):
       acc[y:y + H, x:x + W]  += corrected[i] * w
       wsum[y:y + H, x:x + W] += w
   mosaic = np.divide(acc, wsum, out=acc, where=wsum > 0)   # in place: no third canvas
   ```

   Past `promote_after` this becomes a job — report progress per tile and name
   `interrupt_kernel` as the cancel path.

8. **Publish the mosaic and the settings.** Upload it in the input's dtype and
   print the dict that reproduces it. Uploading is `guide://data`'s subject; what
   matters here is that pixel size does not ride along by default, so pass it, or
   the mosaic comes back a calibrated image's uncalibrated copy.

   ```python
   print({"grid": GRID, "tile_order": TILE_ORDER, "overlap_pct": OVERLAP,
          "ff_channel": FF_CHANNEL, "get_darkfield": GET_DARKFIELD,
          "ncc_threshold": NCC_THRESHOLD, "pairs_accepted": ACCEPTED_FRAC,
          "flatfield": FLATFIELD_METHOD, "placement": PLACEMENT})
   ```

   Either way, check the spacing on the resulting layer before handing it on: a
   mosaic that lost it measures in pixels while looking exactly the same.

## Guardrails

- **Size the job at step 2, and stop there if it will not fit.** There is no
  out-of-core path here — fit, registration and blend each want the whole stack
  at once — so the answer to a mosaic that will not fit is **smaller tiles**,
  not a cleverer blend. Warn, and let the user pick the resolution.
- **A mosaic too large to hold is too large to `add_image`.** Upload it and
  `add_tensor` the returned id, so the viewer reads it back lazily.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Peak memory far above the canvas size that was estimated | Counted one canvas at the input dtype; the blend holds two float32 ones plus the fit and transform stacks | Budget 8 bytes per output pixel and add both stacks (step 2) |
| `MemoryError` during `fit` or `transform`, before any blending | The float32 tile stack, not the canvas | Fit on a subsample; if `transform` itself will not fit, the tiles are too large for this workflow at full resolution |
| Every other row of tiles mirrored | Snake acquisition read as row-major | Reverse odd rows when building the stack, before step 5 |
| Mosaic is a diagonal staircase, or tiles pile at one corner | `rows`/`cols` transposed — `m2stitch` does this itself unless `row_col_transpose=False`, and its default is `True` | Pass it explicitly (step 5); the default is documented to flip in a future release, so relying on it breaks either way |
| Seams still visible as brightness steps | Blended without flat-field correction, or correction applied after registration | Correct first (step 3), then register |
| Dark blotches at tile centres, bright rims | Flat-field fitted the specimen — too few or too-similar tiles | Check the field for structure (step 4); use the smoothed median instead |
| Over-corrected dark regions, crushed blacks | Camera offset absorbed into the multiplicative field | Refit with `get_darkfield=True` |
| Most pairs rejected; placement equals nominal | Overlap too small to correlate, or wrong nominal overlap | Confirm the real overlap; below ~5% registration is not recoverable |
| Kernel dies mid-run, no traceback | The OS killed it — the job was larger than what step 2 measured, or was never measured | Re-run from step 2 on resized tiles |

## Next steps

- Segment the mosaic as one image; that is the point of stitching, and it removes
  the duplicate-objects-at-seams problem entirely.
- Score any segmentation of it with [[segmentation-qc-metrics]], and report sizes
  from it with [[calibrated-measurements]] — after carrying the spacing across
  (step 8).
- The settings dict from step 8 is the fixed configuration for the rest of a plate
  or slide set; the illumination fit is per channel and per acquisition session,
  and should not be reused across either.
