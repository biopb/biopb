---
id: align-stack-by-features
title: Align serial sections that shifted and turned between slices
description: Register a stack of serial sections that were each placed independently, so the same structure sits at the same coordinates through the whole stack.
tags: [registration, alignment, preprocessing]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0]
---

# Align serial sections that shifted and turned between slices

## When to use

A stack whose slices were each **placed independently** — serial sections on
slides, array tomography, block-face imaging, a re-mounted specimen. The
misplacement is tens of pixels of translation *and* degrees of rotation, and the
content changes from one slice to the next because they are different sections
of the specimen, not repeated looks at the same one.

Both of those break the intensity-correlation route. Matching features and
fitting a rigid model to the ones that survive is what tolerates content that
turns over.

## When NOT to use

- **A time-lapse whose field of view drifted.** That is [[drift-correction]]:
  same region seen repeatedly, small displacements, and an intensity method that
  is both faster and more precise there. Here the content itself changes, which
  is the case that method has no answer for.
- **A grid of overlapping tiles.** [[stitch-tiles]] — those are different regions
  at the same time, and they have a known nominal layout to exploit.
- **The sections deform rather than move.** Folds, tears, stretch from cutting.
  A rigid model cannot express it, and widening the model to affine does not fix
  it — it *absorbs* the deformation into the fit and reports success.
- **Consecutive sections share too little structure.** Feature matching needs
  something common to match. Step 6's before-alignment number is the test: on a
  fixture whose content turned over almost completely between sections,
  neighbour NCC read 0.19 unaligned and the alignment came out 329 px wrong.
  Cut thinner sections or accept that the stack is not registrable.
- **Where each section physically sat is the measurement.** Alignment overrides
  it by construction.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `STACK` | `(Z, Y, X)` | One channel, one resolution level, in cutting order. Read `guide://data` first — pyramid level and laziness both bite here |
| `REF_CHANNEL` | — | One **structural** channel present in every section. The transforms are estimated once on it and applied to all channels |
| `MIN_INLIERS` | count | The gate on the direct fit (step 4). It must sit far above the model's `min_samples`, which is 3 — see step 4 for why 3 is the number that matters. `20` is a fine starting point and the value is not delicate |
| `DETECTOR` | — | `SIFT`. `ORB` is about 2x faster (0.14 s vs 0.30 s per 512² section) and cost one section of direct reach on the same fixture — take it only when the stack is large enough for that to matter |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status` — `guide://kernel` covers what to do about a gap. Everything
   below is `skimage.feature`, `skimage.measure` and `skimage.transform`, all
   core, so there is nothing to install and no degraded path to describe.

2. **Confirm what moved, and on which channel** *(confirm-input, blocking)*.
   Show the first two or three sections and ask whether the sections were *placed*
   differently — which this corrects — or are *deformed* relative to each other,
   which it cannot. Confirm `REF_CHANNEL` in the same question. Neither fact is in
   the pixels, and a deformed stack fitted rigidly returns a confident answer.

   The first section is the reference; the whole stack lands in its coordinate
   frame.

3. **Describe every section once.** Both fits below reuse these, so detecting per
   pair would do the work twice.

   ```python
   import numpy as np
   from skimage.feature import SIFT, match_descriptors
   from skimage.measure import ransac
   from skimage.transform import EuclideanTransform, warp

   def describe(section):
       d = SIFT()
       d.detect_and_extract(np.asarray(section, np.float32))
       return d.keypoints, d.descriptors

   def estimate(a, b):
       """Rigid map carrying section b onto section a, and its inlier count."""
       (ka, da), (kb, db) = a, b
       m = match_descriptors(da, db, cross_check=True)
       if len(m) < 4:
           return None, 0
       src, dst = kb[m[:, 1]][:, ::-1], ka[m[:, 0]][:, ::-1]   # (row,col)->(x,y)
       model, inliers = ransac((src, dst), EuclideanTransform, min_samples=3,
                               residual_threshold=2, max_trials=2000)
       return (model, int(inliers.sum())) if model else (None, 0)   # see below
   ```

   **`[:, ::-1]` is the whole ballgame.** Keypoints come back as `(row, col)` and
   every transform in `skimage.transform` is `(x, y)`. Feeding them straight
   through fits a mirror-conjugated transform that is **internally consistent**,
   so nothing complains: measured, it returned the same inlier counts to within
   one (294/236/177/115/73/40/28 either way) and placed sections **95.7 px**
   wrong. It is not only a rotation problem — on a translation-only stack the
   swap still landed **33.9 px** out, because it exchanges the two translation
   components.

   **Truth-test the model, do not compare it to `None`.** From `scikit-image`
   0.26 a failed `ransac` returns a falsy `FailedEstimation` object rather than
   `None`, so an `is None` check passes it through and the error surfaces much
   later as `FailedEstimationAccessError: FailedEstimation is not callable`.

4. **Fit direct-to-reference, and fall back to the neighbour when the inliers
   run out.** This is the decision the whole skill exists for:

   ```python
   feats = [describe(s) for s in STACK]
   pos, log = [EuclideanTransform()], [("reference", 0)]
   last = 0                      # the most recent section actually placed
   for k in range(1, len(STACK)):
       model, n = estimate(feats[0], feats[k])
       if model is not None and n >= MIN_INLIERS:
           pos.append(model); last = k; log.append(("direct", n)); continue
       model, n = estimate(feats[last], feats[k])      # not blindly k-1
       if model is None:
           pos.append(None); log.append(("failed", n)); continue
       # compose onto the LAST PLACED POSITION, never onto the reference
       pos.append(EuclideanTransform(matrix=pos[last].params @ model.params))
       last = k
       log.append(("chained", n))
   ```

   **Direct is preferred because chaining accumulates.** Composed
   neighbour-to-neighbour transforms drift with depth — measured on one stack,
   0.065, 0.173, 0.383 and 1.050 px of end-to-end error at 4, 8, 16 and 32
   sections, against a direct fit's 0.042, 0.140 and 0.313 over the same stacks
   where it survived.

   **The fallback exists because the direct fit dies silently.** Inliers against
   the reference decay as the sections stop being the same tissue: 95, 27, 6, 3,
   0 over one 8-section stack, while the *consecutive* pairs held 65–104
   throughout. What makes it silent is that RANSAC always reports a consistent
   model once it is down to `min_samples` points, because three points fit a
   3-parameter model exactly. Fits with 1–4 inliers came back a **median of 371
   px** out (worst 499); fits with 20 or more landed at 0.15–0.52 px. Ungated,
   that one bad fit put a section 182 px out of a stack that otherwise aligned
   to 0.4 px.

   **The gate value is not what needs tuning — having a gate is.** Anything from
   5 to 200 produced the same aligned stack on an 8-section fixture; only
   removing it changed the answer.

5. **Apply the transforms to every channel.** `warp` takes an *output → input*
   map, and `pos[k]` runs the other way:

   ```python
   aligned = np.stack([
       s if p is None else
       warp(s, EuclideanTransform(matrix=np.linalg.inv(p.params)),
            order=1, preserve_range=True)
       for s, p in zip(STACK, pos)]).astype(STACK.dtype)
   ```

   Reuse `pos` across channels; do not re-estimate per channel, which would
   shift the channels relative to each other. Interpolation resamples
   intensities, so keep the raw stack if the measurement is photon counts.

6. **Check the neighbour correlation, not the reference correlation**
   *(visual check)*. Report the route log and the correlation of each section
   with the one below it, before and after:

   ```python
   def ncc(a, b):
       a, b = a - a.mean(), b - b.mean()
       return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

   pairs = lambda st: [ncc(st[k - 1], st[k]) for k in range(1, len(st))]
   before, after = pairs(STACK), pairs(aligned)
   ```

   **Correlation against the *reference* is the wrong number**, because content
   turnover lowers it for a legitimate reason — a correctly aligned far section
   scored 0.535 against the reference on the same stack where every neighbour
   pair scored 0.95 or better. Neighbour correlation is what separates aligned
   from not: 0.90–0.99 aligned against 0.10–0.54 unaligned across the stacks this
   was measured on, and it is **the only signal that catches a swapped coordinate
   order** (0.34–0.48), which leaves the inlier counts untouched.

   Put `aligned` on the viewer and screenshot an orthogonal (XZ) slice too — a
   stack that is aligned looks continuous down the cut and a chained section that
   slipped shows as a step.

7. **Hand back the aligned stack and how each section got its place.** Upload it
   if it is worth keeping — `guide://data` covers upload, and pixel spacing does
   not ride along by default.

   ```python
   print({"ref_channel": REF_CHANNEL, "min_inliers": MIN_INLIERS,
          "model": "euclidean", "detector": "SIFT",
          "route": [t for t, _ in log], "inliers": [n for _, n in log],
          "neighbour_ncc_before": [round(v, 3) for v in before],
          "neighbour_ncc_after": [round(v, 3) for v in after]})
   ```

   The per-section route is not a diagnostic detail — a `chained` section is
   placed relative to its neighbour rather than measured against the reference,
   so anything read across the stack depends on which sections those were.

## Guardrails

- **Do not widen the transform model to buy robustness.** On rigid truth,
  similarity and affine cost accuracy rather than gaining it (median error 0.150
  px rigid, 0.172 similarity, 0.240 affine). The real cost is worse than the
  number: the extra freedom is exactly what a deforming section needs to be
  fitted and declared fine.
- **Do not tune the matcher.** `cross_check` and `max_ratio` make no difference
  once RANSAC is doing the rejection — all four combinations of them landed
  between 0.32 and 0.49 px on the same stack. The gate in step 4 is the check;
  the matcher is not.
- **A `failed` section is not aligned, and nothing downstream says so.** It is
  passed through at its original placement by the step 5 fence. Report it, and
  drop it rather than measure through it.

## Failure modes

Every row below was hit while building this; `scikit-image` 0.26.

| Symptom | Cause | Fix |
|---|---|---|
| Sections land tens of px out while the inlier counts look healthy, and NCC against the reference is the only thing that looks odd | Keypoints passed to the transform in `(row, col)`; every `skimage.transform` is `(x, y)`. The fit is self-consistent in the swapped frame, so the inlier count is unchanged | `[:, ::-1]` on both point sets (step 3). Measured: 95.7 px with rotation, still 33.9 px on a translation-only stack |
| `FailedEstimationAccessError: FailedEstimation is not callable` | A failed `ransac` returns a falsy object, not `None`, from 0.26 — an `is None` guard lets it through | Truth-test the returned model (step 3) |
| One section is wildly out and the rest are fine; its log entry says `direct` with a single-digit inlier count | RANSAC reports a consistent model at `min_samples` points because 3 points fit 3 parameters exactly | Gate on `MIN_INLIERS` (step 4). Measured: 1–4 inliers gave a median 371 px error, 182 px in the stack this was found in |
| Every section past the first fallback is hundreds of px out, in a stack whose fits all succeeded | The fallback transform composed onto the *reference* instead of onto the last placed section's position — it maps section k onto its neighbour, not onto the reference | `pos[last].params @ model.params` (step 4). Measured: 0.7–2.5 px re-anchored against 212–358 px |
| `AttributeError: 'NoneType' object has no attribute 'params'` partway down a stack | A section that failed both fits is `None` in `pos`, and the next section composed onto it | Fall back to the last *placed* section, not to `k-1` (step 4). Composing onto a failed neighbour is silently wrong even where it does not raise |
| The aligned stack is worse than the input, with no error anywhere | `warp` given `pos[k]` directly; it wants the output → input map | Pass `inv(pos[k].params)` (step 5). Measured: reference-to-last NCC −0.060 applied forwards, 0.535 inverted, −0.024 unaligned |

## Next steps

- A stack aligned here is what gets measured next, and the Z spacing to carry
  into that is the **section thickness**, which alignment neither knows nor
  changes.
