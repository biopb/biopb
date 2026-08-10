---
id: measure-smlm-resolution
title: Localize an SMLM stack and measure a resolution you can quote
description: Run single molecule localization on an SMLM/STORM/PALM blinking stack and measure the reconstruction's resolution by Fourier ring correlation, apart from localization precision and label density.
tags: [smlm, storm, palm, super-resolution, localization, resolution, quantification]
version: 1.0.0
checklist: [viewer, tensor, dask, plugin:image_resolution, pkg:biopb-mcp>=0.13.0]
---

# Localize an SMLM stack and measure a resolution you can quote

Numbers below are measured and marked. **PAINT** is a real 40 000-frame
acquisition from the catalog (364x500, ~100 spots/frame, ~4 million fits).
**Synthetic** is 2000 frames of 9 981 blinks with exact positions and photon
counts. Treat both as the shape of the effect, not as targets for your data.

Detection, Gaussian fitting, the ADU-to-photon conversion and the CRLB formula
are **deliberately not spelled out here**: cold runs reproduce all of them
correctly and unprompted, and a step that only restates what the model already
does dilutes the parts that matter. What follows is the residue that cold runs
get wrong.

## When to use

A long stack of sparse blinking frames — STORM, PALM, DNA-PAINT — where the
deliverable is a super-resolution image, a resolution figure, or both. Also use
it when a render already exists and the user wants to know what resolution they
may claim for it.

## When NOT to use

- **Reporting localization precision as the resolution.** Median CRLB answers
  "how well was each molecule found"; resolution answers "what can be told
  apart", and label density and drift both sit between them. Quoting precision
  as resolution is the substitution to refuse — say which one you are giving.
- **Running decorrelation analysis on a localization render.** It is built for
  images whose noise is spatially white, and a rendered point cloud is the
  opposite. On a sparse-versus-dense pair it reports the **sparser render as the
  finer one** — the ranking inverts. `decorrelation_resolution` is for the raw
  camera frames and for diffraction-limited or fluctuation-based images; use FRC
  on localizations, always.
- **Emitters are dense enough to overlap.** Then it is a fluctuation problem
  (SRRF/SOFI), not a localization one, and fitting single Gaussians to merged
  spots produces confident nonsense.
- **The structure fills only a little of the field.** The density floor below is
  areal and will read as pessimistic on filaments; see step 5.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `PIXEL_NM` | nm | Camera pixel size. **Ask the user** — an acquisition fact, not in the pixels |
| `GAIN`, `OFFSET` | ADU/photon, ADU | **Ask the user.** Both are needed before any uncertainty is computed. Verify rather than trust: on a shot-noise-limited background the pixel std should equal `sqrt(mean photons)`. Synthetic: background ~50 photons/px, noise std 7.4, `sqrt(50)` = 7.07 — gain confirmed without truth |
| `RENDER_NM` | nm | FRC render bin. Sets a hard floor of **twice itself** on the answer, so pick it 5-10x finer than the resolution you expect. 5 nm is safe for a 20-40 nm result |
| `BLOCK_FRAMES` | frames | Split block length, step 5. A few times the typical molecular on-time — never 1. 500 is a reasonable start; on-time is measurable from the list |
| `N_EMITTERS` | count | Distinct **molecules**, not localizations — step 5 |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status`; `## Kernel plugins` answers for `plugin:image_resolution`,
   and `guide://kernel` covers a gap. Without the plugin, FRC is perhaps thirty
   lines to write by hand and the *split* below is still the part that decides
   the answer — write the split correctly and a hand-rolled FRC is fine. Without
   `dask` everything still works serially; step 3 says what that costs.

2. **Confirm the inputs** *(blocking)*. Pixel size, gain and offset, and whether
   the user wants a **display** render, a **resolution number**, or both — step 4
   builds them differently and one image cannot serve both.

3. **Localize.** Detect liberally, fit, convert ADU to photons before any
   uncertainty formula, filter on photons and precision. One decision here is
   not obvious and is worth the line, because cold runs get it wrong 3 times out
   of 3: passing `bounds=` to `curve_fit` silently selects the TRF solver.

   ```python
   curve_fit(model, coords, window.ravel(), p0=seed, method="lm", maxfev=200)
   ```

   PAINT, per spot: `method="lm"` **1 399 us** against TRF's **5 327 us** — 4.7x,
   which over ~4 million fits is 93 minutes against 5.9 hours on one core. It is
   not free: LM drops **5.4%** of the localizations TRF keeps and the sanity gate
   accepts. Where both survive they agree to a median of **0.00 nm** (p95 0.03),
   so the speed costs recall, not accuracy — and in a density-limited
   reconstruction 5.4% fewer localizations is worth roughly 2.7% of resolution.
   **Use bounds on a small precious dataset; use LM once the fit count reaches
   the millions.** Frames are independent, so fan the blocks across `dask`;
   parallel-LM on the full PAINT stack is ~17 minutes against ~6 hours serial,
   at which point the job is I/O bound on the fetch.

   To stop a runaway fit use **`interrupt_kernel`, never `restart_kernel`** — a
   restart destroys the viewer layers and any ROI the user drew by hand.

4. **Render, and know which render you are making** *(visual check)*. These are
   two different images and using one for the other is a silent error:

   - **For display** — splat each localization as a Gaussian of *its own* CRLB.
     Weighting by precision is what makes the picture honest to look at.
   - **For FRC** — plain 2-D histogram counts, unweighted. A precision-weighted
     render convolves every localization with its own uncertainty, which is a
     filter applied equally to both halves and therefore **invisible to FRC**:
     the number would describe your renderer instead of your data.

   `frc_from_localizations` builds its own counts render, so hand it the list,
   not your display image. Report the localization count and median CRLB with
   the picture, never a screenshot alone.

5. **Measure the resolution — and the split is the measurement.** A localization
   list is not a bag of independent samples: one fluorophore blinks many times
   and is localized once per blink. Any split that can put two blinks of the same
   molecule into opposite halves correlates those halves by something that is not
   structure, and FRC then reports a resolution better than the truth, silently.
   Four cold arms out of four split randomly and **reported a sparser
   reconstruction as 2x better where the truth is 3x worse**.

   ```python
   res = image_resolution.frc_from_localizations(
       x_nm, y_nm, frames,               # frames is what makes a safe split possible
       render_pixel_size=RENDER_NM,
       split="blocks", block_frames=BLOCK_FRAMES,
       n_emitters=N_EMITTERS,            # molecules, not localizations
   )
   print(res.summary())                  # carries the criterion and every warning
   ```

   `split="blocks"` is the default and the one to quote. Its cost is that slow
   drift now differs between halves and is charged against the resolution, which
   is the honest direction to err. `split="random"` warns and is for comparison
   only.

   **`n_emitters` is molecules, not localizations.** One fluorophore localized
   twelve times samples the structure once, so an un-merged count understates the
   density floor by the square root of the mean blink count — **3.5x at twelve
   blinks**, the difference between a floor that catches an overclaim and one
   that waves it through. Merge blinks that persist across consecutive frames
   within roughly one localization precision, and pass the merged count.

   Read `res.nyquist_limited` and the warnings before quoting anything. If FRC
   comes back finer than the density floor, the label density is the limit and no
   amount of precision moves it (Shroff 2008). On filaments or any sparse
   structure the areal floor is pessimistic — the spacing that limits a filament
   is the spacing *along* it — so treat that warning as "check this", not a
   verdict.

6. **Hand off with both numbers and the criterion.** Give the FRC resolution
   *with its threshold name* (`1/7` is the SMLM convention and is not comparable
   to a `half-bit` figure), the median and 10-90% CRLB, the localization and
   emitter counts, and the split used. A resolution quoted without its criterion
   and its split cannot be compared to anything.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| FRC on a thinned reconstruction comes back *better* than on the full one | Random split — repeat blinks of one molecule land in both halves and correlate them. Reported 2x better where truth was 3x worse | `split="blocks"`, `block_frames` a few times the on-time |
| Fitting will not finish; ~5 ms per spot | `bounds=` selected the TRF solver | `method="lm"`: PAINT 5 327 → 1 399 us/spot, at a cost of 5.4% of localizations |
| Reported precision is about half what it should be | ADU fed into a CRLB formula that wants photons; the error scales as `sqrt(gain)` | Synthetic, gain 4: calibration 2.25 against 1.13 once converted — same code, same detections, only the uncertainty column moved |
| A sparser render measures finer than a denser one | Decorrelation analysis run on a localization render | FRC for localizations; decorrelation only on raw frames or a diffraction-limited image |
| FRC beats the density floor and nothing warns | `n_emitters` left at the localization count | Merge blinks first; at twelve blinks per molecule the floor is 3.5x off |
| The render is a diffuse carpet with no structure | Often expected on a short or sparse acquisition — but uncorrected drift is the usual culprit | Diagnose by cross-correlating temporal blocks; on sparse data `phase_cross_correlation` returns absurd shifts, so treat those as "no estimate", not as drift |

## Next steps

- Drift correction before re-measuring: [[drift-correction]] works on the raw
  stack, but an SMLM list is better corrected in the localization domain by
  cross-correlating temporal blocks and least-squares solving all pairs, which
  stops error accumulating along the acquisition.
- Per-structure measurements on the render are made in physical units — and
  the render's pixel size, not the camera's, is the spacing that applies.
