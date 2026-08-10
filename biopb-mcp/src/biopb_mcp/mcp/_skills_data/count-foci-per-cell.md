---
id: count-foci-per-cell
title: Count punctate spots inside each segmented cell
description: Count the foci, puncta or FISH spots inside every segmented cell and report a per-cell table, including the cells that have none.
tags: [detection, quantification, spots, single-cell]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0]
---

# Count punctate spots inside each segmented cell

## When to use

A punctate channel and a parent segmentation, and the question is *how many per
parent*: DNA-damage foci, FISH spots, PML bodies, transcription sites, puncta per
cell. The spots are near-diffraction-limited — they have a position and a
brightness but no useful shape — and the answer is a count, not an area.

## When NOT to use

- **The objects are big enough to have a shape.** If you would describe them by
  area or diameter rather than by number, label them and measure them in
  physical units instead.
- **You have no parent segmentation.** This procedure assigns spots to labels it
  is given; it does not produce them. Segment first.
- **The question is intensity, not number.** Mean or integrated intensity per
  cell is `regionprops` on the raw channel and needs none of this.
- **The data is a z-stack and you are counting on a maximum projection.** One
  focus spanning three planes projects to one spot, but two foci at the same
  (y, x) in different planes project to one spot as well. Detect in the volume —
  the steps below carry over with 3-D filters and a 3-D footprint — or count on
  a single plane and say so.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `SPOT_SIGMA` | px | From the physical spot size and the pixel size, both step 2's to ask: `SPOT_SIGMA = (diameter_um / pixel_um) / 2.355`. A 0.3 µm spot at 0.1 µm/px is 1.3 px. Under-estimating is safe; over-estimating is not — see step 3 |
| `K` | — | The MAD multiplier for the detection threshold. `5`. Anything from 3 to 8 gave the same answer to within 1% on the fields this was measured on, so it is not the parameter to reach for when the counts look wrong |
| `MAX_AREA` | px² | `3 * pi * (1.177 * SPOT_SIGMA)**2` — three times the area a diffraction-limited spot covers above half its own height. The multiplier absorbs noise and pixelation; 2 to 8 all landed within 2% of the truth, and 1 rejected every real spot |
| `MIN_DISTANCE` | px | `2 * SPOT_SIGMA`, rounded up. Two spots closer than that are one peak whatever you do. Values from 1 to 5 were indistinguishable where the foci were 10 px apart |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status` — `guide://kernel` covers what to do about a gap. Everything
   below is `skimage.feature`, `skimage.filters` and `numpy`, all core, so there
   is nothing to install and no degraded path to describe.

2. **Ask what a focus is** *(confirm-input, blocking)*. One question, three
   facts, none of them in the pixels:

   - **How big is a focus, and what is the pixel size?** Together they give
     `SPOT_SIGMA`, and every filter below is set from it.
   - **Which bright things are *not* foci?** Aggregated stain, nucleoli,
     autofluorescent granules — wider than a focus, and overlapping it in
     brightness, so no detector separates them unaided. This is the fact that
     decides the counts: on fields carrying about twenty of them across
     twenty-five cells, not rejecting them ran the count **53% over** the truth.
   - **Should cells with no foci be reported?** Almost always yes — the
     foci-negative fraction is usually the readout — and it changes what you
     build in step 6.

   Put the raw channel on the viewer before asking and say what you can see, so
   the person is answering about their own picture.

3. **Subtract the background at the spot scale.** A band-pass keeps what is the
   size of a focus and removes both the cell's diffuse pool and the pixel noise:

   ```python
   import numpy as np
   from skimage.feature import peak_local_max
   from skimage.filters import difference_of_gaussians

   residual = difference_of_gaussians(FOCI.astype(np.float32),
                                      SPOT_SIGMA, 2 * SPOT_SIGMA)
   ```

   **The scale is the whole decision, and it is wrong in both directions.** A
   background estimate at the *object* scale — a rolling ball or a white tophat
   sized to the nucleus — leaves the nucleus in the residual and reproduces the
   raw-channel failure exactly: on 60 px nuclei, a tophat radius of 25 px or
   50 px recovered no foci at all, and one of 15 px lost 15% of them. Smoothing
   past the spot size destroys them from the other end: where the true spot was
   1.3 px, filtering at 2.0 px cut the count to 65% of the truth and 3.0 px to
   nothing.

   **A background subtraction that does not also smooth is not enough.** A white
   tophat alone leaves per-pixel noise, and every noisy pixel is a local maximum:
   thresholded that way, fields of ~53 foci returned about **1,100 detections**
   each, against 70-85 for the same residual smoothed. Smooth it at `SPOT_SIGMA`
   or use the band-pass above, which does both in one call.

4. **Threshold the residual, once, for the whole field.**

   ```python
   inside = LABELS > 0
   r = residual[inside]
   threshold = np.median(r) + K * 1.4826 * np.median(np.abs(r - np.median(r)))
   peaks = peak_local_max(residual, min_distance=MIN_DISTANCE,
                          threshold_abs=threshold, labels=inside)
   ```

   **On the raw channel this same rule silently returns nothing**, and that is
   the failure this skill exists for. Each cell carries its own diffuse pool, so
   the dimmest cell's foci sit below the brightest cell's background: measured
   over seven fields, `median + 5·MAD` on the raw channel found **0 to 5 of the
   50-56 foci** in each, and nearly everything it did find was one of the
   aggregates from step 2 — the wrong population, and the only one bright enough
   to clear a threshold set that way. Per-parent thresholds on the raw channel
   are not the fix either — they recover 71% of the total and still score MAE
   1.39, against 0.02 for the residual.

   One threshold for the field is enough *because* step 3 removed the
   between-cell variation that would call for more: deriving it per parent
   instead changed nothing, MAE 0.02 either way.

5. **Reject anything wider than a spot.**

   ```python
   MAX_AREA = 3 * np.pi * (1.177 * SPOT_SIGMA) ** 2
   window = int(np.ceil(6 * SPOT_SIGMA))

   def too_wide(y, x):
       patch = residual[max(0, y - window):y + window + 1,
                        max(0, x - window):x + window + 1]
       return (patch >= residual[y, x] / 2).sum() > MAX_AREA

   kept = np.array([p for p in peaks if not too_wide(*p)])
   n_rejected, peaks = len(peaks) - len(kept), kept
   ```

   This cut assumes the foci really are diffraction-limited and all one size,
   which is step 2's first question and not something to take on trust: where
   the biology makes genuinely large foci — resolved repair bodies, clustered
   sites — it removes them, and the reject count in step 7 is what shows it
   happening.

   **Width, not brightness.** Aggregate is brighter than a focus *on average*
   and overlaps it completely: on one field the band-passed response ran 72-149
   at a focus and 46-114 at an aggregate, so a brightness cut placed anywhere
   trades one population for the other. Measured over seven fields, an amplitude
   cut scored exactly what applying no filter at all scored — MAE 1.11 either
   way, against 0.02 for the width filter, on about 21 aggregates per field of
   53 foci.

6. **Assign each spot to its parent, and keep the parents with none.**

   ```python
   owner = LABELS[tuple(peaks.T)] if len(peaks) else np.zeros(0, int)
   counts = np.bincount(owner[owner > 0], minlength=int(LABELS.max()) + 1)[1:]
   ```

   Both halves of that line are load-bearing. `owner > 0` drops what landed
   outside every parent — dirt on the coverslip put about 12 such peaks in each
   field here — and `minlength` is what keeps the zero-count cells in the table.
   Grouping by the labels that *appear* silently drops them, and they are the
   population the experiment is usually about: on one field it moved the mean
   from 2.00 to 2.50 foci per cell, **25% high**, and reported a foci-negative
   fraction of 0 where the truth was 0.20.

7. **Show the spots and report the distribution** *(visual check)*. Add `peaks`
   to the viewer as a Points layer over the raw channel and screenshot one
   crop — the eye is very good at "that is a focus" and very bad at "that is
   0.42 above threshold". Never the screenshot alone: report the total, the mean
   per cell, the foci-negative fraction, and **how many candidates step 5
   rejected**. That last number is the one that says whether the confirm-input
   answer was used, and a run where it is 0 on a sample that has aggregates has
   not applied it.

   The count histogram is the other check: foci counts are small integers, so a
   distribution running to 20+ in some cells and 0 in the rest usually means the
   bright cells are being read as clumps rather than counted.

8. **Hand back the per-cell table and what produced it.**

   ```python
   print({"cells": int(LABELS.max()), "total": int(counts.sum()),
          "mean_per_cell": round(float(counts.mean()), 2),
          "foci_negative_fraction": round(float((counts == 0).mean()), 2),
          "spot_sigma_px": SPOT_SIGMA, "k": K, "rejected_too_wide": n_rejected})
   ```

   Report `counts` in label order and say so — the caller has to be able to line
   it up with its own labels, and a table sorted by count is not that.

## Guardrails

- **Do not tune `MIN_DISTANCE` to bring an over-count down.** It is the wrong
  knob: from 1 to 5 px it changed nothing here, and by the time it does the
  answer is set by an arbitrary spacing rule rather than by what is in the image.
  Over-counting is step 5's job and under-counting is step 3's.
- **Do not set the width cut at the diffraction limit itself.** A real spot
  measures wider than an ideal one; a cut at `1 *` the ideal area rejected every
  true focus (0 of 53) while `2` to `8 *` all landed within 2% of the truth.
- **Counts are not comparable across acquisitions unless the spot size is.** The
  whole procedure is anchored to `SPOT_SIGMA`; a second dataset at a different
  magnification needs step 2 asked again, not the same numbers reused.

## Failure modes

Every row below was hit while measuring this; `scikit-image` 0.26, on fields of
25 nuclei carrying about 53 foci.

| Symptom | Cause | Fix |
|---|---|---|
| Nothing is detected, or only a handful of the very brightest blobs | The threshold was applied to the raw channel, where the between-cell background spread exceeds a focus's own amplitude | Threshold the residual (steps 3-4). Measured: 0-5 of 50-56 foci found, and nearly every detection was an aggregate |
| Most cells report zero and a handful report 15-20 | One intensity threshold over the whole field: it clears only inside the brightest cells, and there it breaks their diffuse pool into many pieces | Steps 3-4, and the histogram check in step 7. Measured: 85 counted against a truth of 52, 16 of 25 cells reported empty against a true 6, one cell credited with 18 |
| Hundreds of detections per field, scattered evenly inside the cells | The residual was background-subtracted but not smoothed at the spot scale, so single noisy pixels are local maxima | Band-pass, or smooth the tophat at `SPOT_SIGMA` (step 3). Measured: ~1,100 detections for 53 foci, against 70-85 smoothed |
| Almost nothing detected, and the residual still looks like the nuclei | Background estimated at the object scale — a rolling ball or tophat sized to the cell | Size it to the spot (step 3). Measured on 60 px nuclei: radius 15 px → 85% of the foci, radius 25 px and 50 px → none |
| Counts run about half again too high | Aggregated stain counted as foci. It overlaps real foci in brightness, so no intensity cut excludes it | Reject on width (step 5). Measured: MAE 1.11 against 0.02, ~21 aggregates counted per field; an amplitude cut scored 1.11, the same as no filter |
| Every candidate is rejected as too wide | The width cut was set at the diffraction-limited area itself | Use 2-3x it (step 5). Measured: 1x → 0 of 53 spots kept, 2x to 8x all within 2% |
| The mean per cell looks high and no cell has zero foci | The table was built from the labels that appeared, so foci-negative cells fell out | `bincount(..., minlength=)` (step 6). Measured: mean 2.00 → 2.50, foci-negative fraction 0.20 → 0 |
| The total exceeds the sum of the per-cell counts | Peaks outside every parent were counted in the total | Drop label 0 (step 6). Measured: about 12 such peaks per field |

## Next steps

- Foci per unit nuclear area or volume needs the parents measured in physical
  units — and the spacing that takes is the same pixel size step 2 already
  obtained.
- If the parent labels came from a segmentation nobody has checked, the counts
  inherit its splits and merges directly: one merged pair of nuclei reads as one
  cell with twice the foci. [[segmentation-qc-metrics]] is how that gets scored.
