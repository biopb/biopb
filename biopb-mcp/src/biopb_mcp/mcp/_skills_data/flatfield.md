---
id: flatfield
title: Correct uneven illumination across a set of tiles or fields
description: Remove vignetting and shading from a collection of images taken on the same optics, so intensities are comparable within a frame and between frames.
tags: [illumination, correction, preprocessing, intensity]
version: 1.1.0
checklist: [viewer, tensor, dask, pkg:biopb-mcp>=0.13.0]
---

# Correct uneven illumination across a set of tiles or fields

## When to use

Several images acquired through the **same optical path** — a tile scan, a
multi-position run, a plate — where the centre is brighter than the edges, or
one corner is consistently dim. Anything comparing intensities across a frame or
between frames needs this first: tile mosaics with visible seams, per-object
intensity, thresholds applied over a whole field, ratio images.

The correction is estimated from the **collection**, not from one image. A
single field cannot separate "the illumination falls off here" from "there is
less specimen here"; many fields can, because the specimen moves between them
and the illumination does not.

## When NOT to use

- **The data is already flat-field corrected.** Vendor software often corrects
  silently on export. Fitting a second field on corrected data *inverts* the
  vignette — the corners come out brighter than the centre. Ask before assuming,
  and check the metadata; this is the most common way to make an image worse.
- **The frames are not the same optical path.** Different objective, different
  zoom, a different illumination setting, or a re-alignment mid-run means more
  than one field. Split the collection and fit each group.
- **There are only a handful of frames and the specimen barely moves.** With a
  near-stationary specimen the estimator cannot tell the field from the content,
  and it will bake the specimen into the field. See `N_TILES` below.
- **The uneven brightness is the measurement.** A gradient that is genuinely in
  the sample — a diffusion front, an illumination-response experiment — is data,
  not an artefact.
- **A single image, with nothing else from that session.** There is no
  collection to fit against. Say so rather than smoothing the image and calling
  it a field.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `TILES` | `(N, Y, X)` | Every frame from one optical configuration, one channel, one resolution level, stacked. Read `guide://data` first — pyramid level and laziness both bite here |
| `DARKFIELD` | counts | The camera offset present in every pixel. **Ask** (step 2) — it is a property of the camera, not of the pixels, and getting it wrong costs more than every other choice combined |
| `KEEP` | — | How many low-order DCT coefficients per axis describe the field. `16` for a normal vignette; the optimum is broad (see step 4) so this is not a knob to tune |
| `N_TILES` | — | How many frames to fit on. Use all of them if they fit in memory; **8 is enough** and 3 is the floor. More tiles help only through specimen averaging, which saturates fast |

Illumination is estimated **per channel**. Channels have their own optical path
and their own vignette, and a shared field manufactures a chromatic gradient
that was not there.

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status` — `guide://kernel` covers what to do about a gap. Everything
   below is numpy and `scipy.fft`, both core, so there is no package to install
   and no degraded path to fall back to.

2. **Ask what is already in the pixels** *(confirm-input, blocking)*. Two facts,
   neither derivable from the data, both able to invalidate the whole result:

   - **Has this already been flat-field corrected?** If yes, stop — see *When
     NOT to use*.
   - **What is the camera offset?** The vendor's number, or a dark frame from
     the same session. This is the single most consequential input, and on a dim
     acquisition it dominates everything else in this skill: measured against a
     known field, the correct offset gives **0.5%** field error, assuming zero
     gives **4.0%**, and taking a low quantile of the stack as an estimate gives
     **10%**. Every choice in step 4 moves the answer by a fraction of that.

   If nobody knows it, the honest fallback is a **bound, not an estimate**: the
   offset must be below the darkest pixel in the stack, and the darkest pixel is
   the offset *plus* whatever background the specimen has — so a low quantile
   overshoots by exactly the amount you cannot see, which is why it is the worst
   of the three numbers above rather than a compromise between them. Use it only
   as a ceiling, and say that you did.

3. **Budget the memory before fitting** *(blocking, if the stack is large)*. The
   fit works in log space at float64, so the peak is about **20 bytes per input
   pixel** — the float32 stack, a float64 log copy of it, and the median's own
   copy — not the 2 bytes a `uint16` stack suggests. Check that against
   `memory_available` from `server_status`.

   Two escapes, in order of preference. **Fit on fewer tiles**: 8 frames give
   0.7% where 24 give 0.5%, so a subset is nearly free. **Fit on a decimated
   stack** and resize the field to full resolution: legitimate here precisely
   because the field is low-order, which is the same assumption `KEEP` encodes.

4. **Estimate the field.** Median polish in log space, then a low-order DCT
   projection:

   ```python
   def estimate_flatfield(tiles, darkfield, keep=16, iters=8):
       """(N, Y, X) -> the illumination field (Y, X), normalised to mean 1."""
       from scipy.fft import dctn, idctn

       n, height, width = tiles.shape
       logs = np.log(np.maximum(np.asarray(tiles, float) - darkfield, 1e-3))
       logs = logs.reshape(n, -1)
       field = np.zeros(logs.shape[1])
       for _ in range(iters):
           logs -= np.median(logs, axis=1, keepdims=True)   # per-tile brightness
           column = np.median(logs, axis=0)                 # the field
           logs -= column
           field += column
       coefficients = dctn(field.reshape(height, width), norm="ortho")
       coefficients[keep:, :] = 0.0
       coefficients[:, keep:] = 0.0
       flat = np.exp(idctn(coefficients, norm="ortho"))
       return flat / flat.mean()
   ```

   Both halves earn their place. `log I = log a + log F + log S` is additive, so
   the per-tile brightness and the field are the row and column effects of a
   two-way table, and alternating medians is their resistant fit — that is
   Tukey's median polish, and it is why exposure drift or bleaching across the
   run changes nothing: 0.5%, unchanged from tiles of identical brightness to a
   12x spread between the brightest and the dimmest.

   **The DCT truncation is there to be a parameter you can be wrong about.** A
   vignette is genuinely low-order, so the basis is well matched and the cutoff
   is broad: `KEEP` from 8 to 64 moves the answer between 0.8% and 0.5%, and it
   only fails when starved, at 4. Blurring the same median with a Gaussian is
   the usual alternative and has no such tolerance — across widths of 15 to 80
   pixels its answer moves between 0.8% and 7.2%, and nothing in the data tells
   you which width you are in. That is the reason to prefer it, not a claim that
   a well-chosen Gaussian is unusable.

   **Say what this is.** It is a low-order smooth-field estimator with a
   resistant two-way fit. It is **not BaSiC**, which solves a constrained L1
   low-rank-plus-sparse problem and estimates the darkfield jointly; do not
   report this under that name.

5. **Look at the field before applying it** *(blocking)*. This is the gate, and
   it exists because the obvious quality metric does not work:

   ```python
   print(f"range {flat.max() / flat.min():.2f}, "
         f"centre/corner {flat[flat.shape[0] // 2, flat.shape[1] // 2] / flat[0, 0]:.2f}")
   ```

   Put `flat` on the viewer and look at it. It should be **smooth, monotonic
   outward from a single bright region, and free of specimen structure**. Then
   check the range: ordinary vignetting spans **1.2–2.5x** across the frame. A
   range above ~4 means the darkfield was over-subtracted — a quantile used as
   an estimate lands here — and a range near 1.0 means there was nothing to
   correct. Both are step 2 coming back.

   **The range is a coarse guard, not a proof.** It catches over-subtraction,
   which blows up, and it catches a field that is not there. It will *not* catch
   a modest under-subtraction: a run that assumed a zero offset where the true
   one was 200 counts reports a range of 1.4 against a true 2.0, comfortably
   inside the expected band while the field is eight times further off than it
   needed to be. That is the reason step 2 asks instead of guessing — nothing
   downstream recovers it.

   **Residual spread across tiles is not this check either**, though it is the
   obvious thing to reach for. It is dominated by real variation between frames
   — exposure differences, specimen density — so it barely moves with the
   quantity you are trying to judge: measured on one collection, a good field
   and one four times further off both leave a residual spread of 11.0%.
   Recognisable structure in `flat` is what actually discriminates.

6. **Apply it, and publish what was applied.**

   ```python
   corrected = (TILES - DARKFIELD) / flat
   ```

   Divide, do not subtract: illumination is multiplicative gain, and the offset
   is what is subtracted. Apply the **same** `flat` to every frame in the group —
   re-fitting per tile removes the very differences between tiles that a mosaic
   needs preserved. Then upload the corrected stack if it is worth keeping and
   print the settings that reproduce it; `guide://data` covers upload and pixel
   spacing, which does not ride along by default.

   ```python
   print({"darkfield": DARKFIELD, "keep": KEEP, "n_tiles": len(TILES),
          "field_range": float(flat.max() / flat.min()),
          "method": "log median-polish + DCT truncation"})
   ```

## Guardrails

- **Correct before registering, never after.** Not because registration fails on
  uncorrected tiles — measured, it barely moves — but because the score a
  stitcher accepts pairs on loses most of its margin, and because a seam that is
  a brightness step cannot be fixed by moving a tile. See [[stitch-tiles]].
- **The corrected data is no longer raw counts.** Division rescales intensities,
  so photon statistics and any calibration expressed in counts belong to the raw
  stack. Keep it.
- **A dust speck or a scratch is not in this basis.** Low-order DCT smooths a
  sharp defect in the light path away, so it stays in the corrected image. Say so
  rather than claiming it was removed.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Corners come out *brighter* than the centre | The data was already flat-field corrected, so a second vignette was fitted and divided out | Step 2. Discard and use the original |
| `flat` has a range above ~4, and the corners blow up | `DARKFIELD` was over-subtracted — often a low quantile used as an estimate rather than a ceiling | Lower it; it must be below the darkest pixel by the specimen's background |
| Correction is visible but weak, and the range is under ~1.2 | `DARKFIELD` assumed 0 when the camera has an offset | Step 2 — this is worth more than any other change |
| `flat` shows recognisable specimen structure | Too few frames, or a specimen that barely moves between them | Fit on more frames, or on frames from the same optics with different fields of view |
| `flat` is nearly featureless and the images are unchanged | `KEEP` starved (4 or less), or there is genuinely no vignette | Raise `KEEP` to 16; if it stays flat, report that no correction was needed |
| Tiles that individually look right, but the mosaic still shows a checkerboard | A field was fitted per tile, cancelling the real differences between them | One `flat` per channel per optical configuration |
| One channel looks corrected and another looks worse | A single field applied to every channel | Estimate per channel |
| The fit runs out of memory on a large tile set | The float64 log stack is ~4x the uint16 input | Step 3 — fit on 8 tiles, or on a decimated stack |

## Next steps

- Register the corrected tiles into a mosaic with [[stitch-tiles]]. Order
  matters, though not for the reason usually given: registration barely notices
  shading, but the correlation score it accepts pairs on loses most of its margin
  without this step. A seam that survives the correction is a registration
  problem, not an illumination one.
- Report intensities and sizes from the corrected stack with
  [[calibrated-measurements]], carrying the pixel spacing across (step 6).
- For a time series where the field of view moved rather than the illumination
  being uneven, [[drift-correction]] is the other workflow, and it is independent
  of this one.
