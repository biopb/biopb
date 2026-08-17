---
id: detect-filaments
title: Detect filaments and measure their width
description: Trace the centrelines of filamentous structures in a fluorescence image and report how wide the filaments are.
tags: [morphology, filaments, ridge-detection]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0]
---

# Detect filaments and measure their width

The numbers below come from synthetic 2D fields — 512 x 512, twelve curved
filaments 0.30-0.90 µm wide at 0.1 µm/px, peak SNR 3-30 — except where a
*real actin field* is named, which is a dense one and fails differently. Treat
them as the shape of the effect, not as targets for your data.

## When to use

An image holds curvilinear structures — cytoskeleton, neurites, collagen,
vessels, microtubules — and the user wants their centrelines, their length, or
their width, rather than a foreground mask.

## When NOT to use

- **The filaments are the same brightness everywhere and well separated.** A
  plain threshold on the raw image scored F1 0.978 on that field against the
  ridge filter's 0.986. The whole apparatus below buys 0.008 there — and 0.752
  against 0.982 once brightness spans a decade, which is what it is for.
- **The brightness varies because the *illumination* does.** Correct that
  first — [[flatfield]] — because a shading field also shifts the ridge
  response, and the low cut in step 4 then has to absorb both. On a field with
  both, this procedure recovered 11 of 12 filaments where it recovered 12 of 12
  without the gradient.
- **The question is topology** — branch points, loops, degree at a node. This
  produces centrelines, not a graph.
- **The structures are blobs, not ridges.** A ridge filter responds to a bright
  line held between two darker sides. Puncta, vesicles and nuclei are not that,
  and the response on them is an artifact of their edges.
- **Individual filaments must be told apart in a dense mesh.** Where filaments
  cross constantly the skeleton merges into one component — measured at the
  recommended cut, twelve filaments came back as **one** object. Centrelines
  survive that; identity does not, and on a genuinely dense field neither does
  the width (step 7).

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `IMAGE` | `(Y, X)` | One plane. `guide://data` for getting it off a layer or the tensor server. For a stack, run this per plane — the width step below measures across a ridge in the image plane |
| `PX_UM` | µm/px | From the acquisition; **ask**. It is not in the pixels, and every width you report is a pixel count times this number |
| `WIDTH_LO`, `WIDTH_HI` | µm | The range of filament widths to look for, bracketing what the user expects. Sato's `sigmas` are half-widths in pixels: `np.arange(WIDTH_LO, WIDTH_HI, PX_UM) / 2 / PX_UM`. Too narrow a range misses the filaments outside it; too wide only costs time |
| `LOW_FRAC` | — | How far below the seed threshold a filament may be and still be grown into. **0.25**, and it is a plateau rather than a lucky value: 0.25, 0.20 and 0.15 all recovered 11-12 of 12 filaments at precision ≥ 0.94, while 0.35 lost one more on a field with an illumination gradient, 0.10 dropped precision to 0.65 and 0.05 to 0.09 |
| `MIN_LEN_PX` | px | Shortest run of centreline worth keeping — shorter than the shortest filament the user would count. 20 px is 2 µm at 0.1 µm/px |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status`; `guide://kernel` covers a gap.

2. **Confirm the inputs** *(blocking)*. Two facts and one judgement:

   - the pixel size in µm;
   - the range of widths to look for, and whether the user means the **full
     width at half maximum** or a Gaussian `2σ`. For a Gaussian cross-section
     those differ by 17.75%, which is larger than everything else in this
     procedure put together. Report which one you measured;
   - whether brightness varies across the field because the filaments differ or
     because the illumination does. If it is the illumination, see *When NOT to
     use*.

3. **Compute a multiscale ridge response.**

   ```python
   import numpy as np
   from skimage.filters import sato

   sigmas = np.arange(WIDTH_LO, WIDTH_HI + 1e-9, PX_UM) / 2.0 / PX_UM
   resp = sato(np.asarray(IMAGE, float), sigmas=sigmas, black_ridges=False)
   ```

   `black_ridges` defaults to **`True`** — dark ridges on a bright background,
   which is a brightfield convention. Fluorescence filaments are bright, and
   left at the default the response is a picture of the background.

4. **Threshold with hysteresis, not with one cut.** This is the step, and the
   one an unaided run gets wrong.

   ```python
   from skimage.filters import apply_hysteresis_threshold, threshold_otsu

   hi = threshold_otsu(resp)
   mask = apply_hysteresis_threshold(resp, LOW_FRAC * hi, hi)
   ```

   One global cut keeps the brightest filaments and drops the rest, and **the
   run's own numbers do not say so**: over three seeds at a 10x brightness
   range, a single Otsu cut found 6-7 of 12 filaments *at precision 0.99-1.00*.
   Everything it detected was a real filament; half the field simply was not in
   the output. Hysteresis at `LOW_FRAC = 0.25` found 11-12 of 12 on the same
   fields at precision 0.98-0.99. The two agree up to a 3x range (dimmest peak
   SNR ≥ 10); the gap opens at 5x and is complete by 8x.

5. **Skeletonise, and prune 8-connected.**

   ```python
   from scipy import ndimage as ndi
   from skimage.morphology import skeletonize

   def centreline(mask, min_len_px):
       skel = skeletonize(mask)
       lab, _ = ndi.label(skel, structure=np.ones((3, 3)))   # 8-connected
       sizes = np.bincount(lab.ravel())
       keep = sizes >= min_len_px
       keep[0] = False                                       # background
       return keep[lab]
   ```

   **A one-pixel skeleton running diagonally is 8-connected.** Label or prune it
   4-connected and it is a string of isolated pixels, every one of them below
   any length you would set — so the diagonal filaments vanish and the
   axis-aligned ones survive, which looks exactly like a threshold that kept
   only the bright ones. Measured on true centrelines: 4-connected pruning at
   `min_size=3` kept **21%** of them. `ndi.label` defaults to 4-connectivity,
   and so does `skimage.morphology.remove_small_objects` (`connectivity=1`).

6. **Ask whether the cut ate anything** *(visual check)*. Recall needs a truth
   you do not have — but the *length-vs-cut curve* does not, and it answers the
   same question. Rebuild the centrelines at several low fractions and compare
   total length against the single cut:

   ```python
   for frac in (1.0, 0.5, 0.35, 0.25, 0.15):
       m = resp > hi if frac == 1.0 else apply_hysteresis_threshold(resp, frac * hi, hi)
       skel = centreline(m, MIN_LEN_PX)
       print(frac, int(skel.sum()), f"{m.mean():.1%}",
             ndi.label(skel, structure=np.ones((3, 3)))[1])
   ```

   Read the length **and** the coverage: the curve has two shapes and only one
   of them means what it looks like.

   - **It flattens.** One cut was enough, or hysteresis has now found
     everything. Measured **1.01x** at `frac=0.25` on a uniform field and at a
     3x range; **1.57-1.86x** over three seeds at 10x, reached by 0.25 and flat
     to 0.15, at ~13% coverage.
   - **It climbs while the mask fills the field.** The low cut is absorbing
     background, not finding filaments — a hazy or dense field responds to a
     ridge filter everywhere, so the seed threshold has nothing to be bimodal
     about. Measured on a real actin field: 1.00, 2.15, 2.87, 3.84, **4.89x**,
     coverage 12% to 62%, components 11 to 1. Stop at the last cut whose
     coverage is plausible for the structure, and say that you did.

   Then put `mask` on the viewer over `IMAGE` and look at the dim corners
   specifically, because that is where the missing filaments are.

7. **Measure the width on the image, never on the mask.** The distance
   transform of a thresholded mask measures **where the threshold landed**, not
   how wide the filament is. Measured per filament against the truth, `2 x` the
   mask's distance transform was −20% to **+119%** out, and its rank
   correlation with the true width was **−0.13** — it does not even order the
   filaments correctly. The transverse profile of the *image* was −11% to +3%,
   rank correlation **+0.97**.

   ```python
   def transverse_fwhm_um(image, skel, px_um, half=12, step=4):
       """Median FWHM across the ridge, in µm, measured on the image."""
       img = np.asarray(image, float)
       gy, gx = np.gradient(ndi.gaussian_filter(img, 1.0))
       t = np.linspace(-half, half, 8 * half + 1)
       ys, xs = np.nonzero(skel)
       out = []
       for y, x in zip(ys[::step], xs[::step]):
           if not (half < y < img.shape[0] - half and half < x < img.shape[1] - half):
               continue
           sl = (slice(y - 3, y + 4), slice(x - 3, x + 4))
           # the ridge normal: dominant eigenvector of the local structure tensor
           ang = 0.5 * np.arctan2(
               2 * float((gx[sl] * gy[sl]).sum()),
               float((gx[sl] ** 2).sum() - (gy[sl] ** 2).sum()),
           )
           p = ndi.map_coordinates(
               img, [y + t * np.sin(ang), x + t * np.cos(ang)], order=1, mode="nearest"
           )
           base = float(np.median(np.r_[p[:6], p[-6:]]))   # the profile's own tails
           peak = p.max() - base
           if peak <= 0:
               continue
           above = np.flatnonzero(p - base >= 0.5 * peak)
           if above[0] == 0 or above[-1] == len(p) - 1:
               continue          # clipped by a neighbour or by the edge
           lo, hi_ = above[0], above[-1]
           # sub-sample both half-max crossings, or the sample grid biases low
           a = lo - (p[lo] - base - 0.5 * peak) / (p[lo] - p[lo - 1] + 1e-12)
           b = hi_ + (p[hi_] - base - 0.5 * peak) / (p[hi_] - p[hi_ + 1] + 1e-12)
           out.append((b - a) * (t[1] - t[0]))
       return float(np.median(out)) * px_um if out else float("nan")
   ```

   Taking the background from each profile's own tails is what keeps an
   illumination gradient out of the width: the same field with and without a
   gradient gave 0.62 and 0.61 µm.

   **Then measure it again at a different `half`, and check it did not move.**
   The tails are a background only if the profile reaches one. Where it does
   not — a dense network, a hazy field — the half-maximum is set by the window
   and the number you get *is* the window: on a real actin field, `half` of 8,
   12, 20 and 30 px returned 1.17, 1.63, 2.35 and **3.56** µm, tracking it
   almost linearly, where the sparse fields moved only 0.60 to 0.69. Doubling
   `half` and getting more than ~20% is the signal, and it needs no truth.

8. **Report the numbers with the picture.** Centrelines as a Labels layer over
   the image, plus total length in µm (`skel.sum() * PX_UM`), the number of
   connected components, the median width and its spread. Say which width
   definition step 2 settled on.

   **The width is no evidence that the detection was complete.** On the field
   where one cut found 7 of 12 filaments, the width it reported was 0.63 µm
   against a truth of 0.67 — because it was right about the six it found. Only
   step 6's curve speaks to the other half.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| The response is strongest on the background, or nearly zero everywhere | `sato`'s `black_ridges=True` default looks for dark ridges | `black_ridges=False` for fluorescence |
| Precision looks perfect, and half the filaments are missing | One global cut on the response. Measured: 6-7 of 12 found at precision 0.99-1.00 | Hysteresis (step 4), then the curve in step 6 |
| Diagonal filaments are gone; horizontal and vertical ones survive | A skeleton is 8-connected, and `ndi.label` / `remove_small_objects` are 4-connected by default. Measured: 21% of true centrelines survive | `structure=np.ones((3, 3))`, or `connectivity=2` |
| Mean width is plausible but per-filament widths are not | Width taken from the mask's distance transform: −20% to +119% per filament, rank correlation −0.13 with the truth | Measure the transverse profile of the image (step 7) |
| Widths came out ~18% off and nothing else is wrong | FWHM against Gaussian `2σ` | Not an error — a definition. Step 2 |
| Everything joins into one filled blob, precision collapses | `LOW_FRAC` below the plateau: 0.10 scored precision 0.65 on a field with a gradient, 0.05 scored 0.09 | Keep `LOW_FRAC` in 0.25-0.15 |
| Twelve filaments come back as one connected component | They cross. The skeleton is one object and the count is meaningless | Expected — see *When NOT to use*. Length is still right |
| The length-vs-cut curve never flattens, and the mask fills the field | A hazy or dense field responds to a ridge filter everywhere, so there is no background for the low cut to stop at. Measured on real actin: 4.89x at `frac=0.15`, coverage 62%, components 11 → 1 | Stop at the last cut whose coverage is plausible, and report that the field is denser than one seed threshold can resolve |
| The width comes out near or above `WIDTH_HI`, and looks nothing like the picture | The transverse profile never returns to background, so the half-maximum is set by the window. Measured on real actin: 1.17, 1.63, 2.35, 3.56 µm at `half` 8, 12, 20, 30 px | The two-window check in step 7. There is no width to report from that field with this method |

## Next steps

- Widths and lengths per filament, rather than pooled, need the centrelines
  split at their crossings first — see the component caveat above.
- Comparing a detection against a hand-traced ground truth is the
  `segmentation_qc` kernel plugin's job; dilate both centrelines to a few pixels
  first, since one-pixel skeletons rarely overlap exactly.
