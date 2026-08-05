---
id: ratiometric-fret
title: Compute a corrected FRET ratio that is comparable between conditions
description: Turn the donor, acceptor and FRET channels of a biosensor experiment into a bleedthrough-corrected, masked ratio that is comparable between conditions.
tags: [fret, ratiometric, biosensor, correction, quantification]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0]
---

# Compute a corrected FRET ratio that is comparable between conditions

Numbers below come from two places and are marked. **TIRF** is a real
dual-camera molecular-tension-sensor acquisition — donor and acceptor imaged
simultaneously on two cameras, 1024x1024, with the original author's processed
output available to compare against. **Synthetic** is a three-cube field built
with known coefficients. Treat both as the shape of the effect, not as targets
for your data.

## When to use

Two or three channels of one field where the answer is the **ratio between
them** rather than the brightness of any one: a FRET biosensor, a molecular
tension sensor, a ratiometric dye. The user wants a number they can compare
between cells, fields or conditions.

## When NOT to use

- **The readout is FRET efficiency, not a ratio.** A corrected ratio is in
  instrument units. `E` needs a G factor from a calibrated standard, acceptor
  photobleaching, or a lifetime measurement — producing a ratio and calling it
  efficiency is the substitution to refuse. Say which one you are delivering.
- **Correcting the ratio image.** Background subtraction, [[flatfield]],
  [[deconvolve-widefield]] and denoising belong on the **channels**, before the
  division. A ratio has no background to subtract, and a filter applied after
  the division mixes both channels' errors irreversibly.
- **The channels were acquired sequentially and the sample moves.** Anything
  that changed between the two exposures is indistinguishable from a change in
  the ratio.
- **One channel's intensity is the readout.** Then it is an intensity
  measurement — [[calibrated-measurements]] — and dividing discards the signal.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `DONOR`, `FRET`, `ACCEPTOR` | (Y, X) or (T, Y, X) | The three cubes: donor ex/donor em, donor ex/acceptor em, acceptor ex/acceptor em. Which layer is which is a **question for the user**, step 2. A two-channel ratiometric sensor has no `ACCEPTOR` and drops the direct-excitation term |
| `BT` | fraction | Donor bleedthrough: median of `FRET / DONOR` inside the cell on a **donor-only** control acquired at the same filters, exposure and excitation power. Not derivable from the experiment itself — step 2 |
| `DE` | fraction | Acceptor direct excitation: median of `FRET / ACCEPTOR` on an **acceptor-only** control, same settings |
| `GAIN` | — | Acceptor-arm over donor-arm sensitivity. An optical-system constant, not a property of one field: ask for it, carry it from a calibrated position, or set it to 1 and state that the ratio is in instrument units. Because `GAIN` and the filter set together set the scale, **a corrected ratio has no expected range** — "0.5 to 2.0 is normal" is not a check, and treating it as one rejects correct numbers |
| `MASK` | — | Where there is enough donor signal to divide by. Derive it from the **background-subtracted donor channel**, never from the ratio |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status`; `guide://kernel` covers a gap.

2. **Confirm the inputs** *(blocking)*. This is the step the result turns on,
   and none of these four facts is in the pixels:

   - **Which layer is which channel.** Donor and FRET the wrong way round
     inverts every conclusion and produces an image that looks entirely normal.
   - **Whether single-label controls exist**, and whether they were acquired at
     the same filters, exposure and excitation power. If they do not, **stop and
     say so**: splitting the FRET channel into sensitized emission and
     leak-through is not identifiable from a doubly-labelled field. You can
     deliver a raw ratio for display, not a fold-change.

     Two substitutes suggest themselves and **neither works**. A plausible
     literature default is not the conservative choice — the coefficients enter
     as an additive term, so a wrong one moves every fold-change rather than
     scaling it: subtracting a correct donor leak while leaving direct
     excitation out still left 70-89% of level error and 20-33% of contrast
     error. Nor does a fold-change survive on the grounds that the settings were
     constant across conditions; an additive offset compresses ratios toward
     each other, and synthetic, a true 3.05x came back as **1.64x**. Fitting the
     coefficients on the experiment itself is the other one — `F ~ a*D + b*A`
     recovered `a` = 0.31-0.42 against a true 0.35, over-subtracted, drove the
     resting population through zero and lost the fold-change entirely.
   - **Whether the two channels come from two detectors** — an image splitter or
     a dual camera. Step 3 applies only if they do.
   - **Display or quantification.** A ratio renormalised to look good keeps every
     fold-change and is comparable to nothing: synthetic, 165% off in absolute
     level with the contrast intact.

3. **Register the channels, if they came from two detectors.** Fit an **affine**
   (translation, rotation, scale, shear) on the temporal means, warping one
   channel onto the other and leaving the other fixed. TIRF: the displacement
   between the two cameras ran from 0.1 px at one corner of the field to 8.8 px
   at another — 0.47° of rotation and 0.92% of scale — and no single shift can
   absorb that.

   **Phase correlation is not the check.** It only sees translation, so it
   reports itself finished while a field-varying sub-pixel error remains. Read
   the residual off the ratio instead — a misregistered edge makes the ratio high
   on one side and low on the other, and the size of that dipole is the error:

   ```python
   import numpy as np

   def dipole_residual(donor, acceptor, mask):
       """Registration error in px, read off the ratio: r/median ~ 1 + d.grad(ln I)."""
       lo, hi = np.clip(donor, 1, None), np.clip(acceptor, 1, None)
       gy, gx = np.gradient(np.log((lo + hi) / 2))
       edges = np.hypot(gy, gx)
       keep = mask & (edges > np.percentile(edges[mask], 50))
       ratio = (hi / lo)[keep]
       design = np.stack([gy[keep], gx[keep], np.ones(keep.sum())], axis=1)
       coefficients, *_ = np.linalg.lstsq(design, ratio / np.median(ratio), rcond=None)
       return float(np.hypot(coefficients[0], coefficients[1]))
   ```

   Run it **per block** (256 px is a reasonable size) and report the spread
   across blocks, not one whole-field number: a field-varying error is what
   translation leaves behind, so the spread is what moves. TIRF, after a
   translation-only fit, phase correlation reported a residual of **0.071 px**
   while this read 0.381 px whole-field and **1.10 ± 0.20 px** per block; after
   the affine, 0.086 px and **0.52 ± 0.03 px**. The cost of stopping at the
   translation was a median ratio **23.8% high** at the structures being
   measured, against an affine that reproduced the author's own output to 0.2%.

4. **Subtract a background from each channel, per frame.** Before anything is
   divided, and never afterwards. The per-frame **median** is a reasonable
   estimate when the sample occupies a minority of the frame; check that it does
   rather than assume it.

   Use the median rather than a mean or a low percentile, and step 3 is the
   reason: a warp fills the vacated border with exact zeros, which are counts no
   detector produced. TIRF, with 0.78% of the frame zeroed by the warp, the mean
   moved by 0.78% and the 1st percentile by 1.03% while the median moved
   **0.05%**. If some estimator other than a median is unavoidable, exclude the
   zeros — but only on the channel that was actually warped.

   ```python
   def background(frame, warped=False):
       return float(np.median(frame[frame > 0] if warped else frame))
   ```

5. **Mask, from the background-subtracted donor channel.** A floor on the
   denominator is not the lever it looks like — TIRF, floors from 0 to 10 counts
   changed nothing inside the mask, because the mask had already removed every
   pixel a floor would have caught. It is there to keep the arithmetic finite.

6. **The corrected ratio.** Everything above was to make these three lines mean
   something:

   ```python
   sensitized = FRET - BT * DONOR - DE * ACCEPTOR   # drop the DE term if 2-channel
   ratio = np.zeros(DONOR.shape, float)
   ratio[MASK] = sensitized[MASK] / DONOR[MASK] / GAIN
   ```

7. **Validate, before calling any number final** *(blocking)*. Two checks, and
   both are ones an unaided run leaves out.

   - **Run the correction on the control slides themselves.** A donor-only and an
     acceptor-only control must both come out at a ratio of zero. This is close
     to tautological when the coefficients were medians over the same pixels —
     which is the point: it fails exactly when they were not, on a control that
     was saturated, differently exposed, or measured over a region including
     background.
   - **Check the ratio against donor brightness, and read the *shape*, not the
     correlation.** Compare the median ratio in the dimmest and brightest deciles
     of the donor channel inside the mask, and compare both against the ratio of
     the two channels' backgrounds. Background left in pulls dim pixels toward
     that background ratio: TIRF, backgrounds of 435 and 478 counts (ratio 1.10),
     and the uncorrected ratio climbed 1.90 → 3.28 from the dimmest to the
     brightest decile, against 5.47 → 4.30 once subtracted. **The trend
     reverses**, so a run that only looked at the direction of the effect got the
     biology backwards. Do not require the correlation to vanish: on the same
     field the correctly corrected ratio was rank-correlated with donor intensity
     at **-0.41**, because real biology varies with expression level.
8. **Hand off with the picture and the numbers.** Put the ratio on the viewer
   over the donor channel with a fixed contrast range, and give the median and
   inter-quartile range **per condition, inside the mask**, the two coefficients
   used, and the registration residual. Never a statistic over the whole ratio
   image: outside the cell it is noise over noise, TIRF p99 of 14.6 against an
   in-cell median of 6.29, and the whole-frame mean came out at 1.79 — off by
   3.5x with nothing visibly wrong.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Every object shows a high/low dipole across it, and the ratio at the structures being measured is ~20% high | Sub-pixel, field-varying misregistration between two detectors | Affine, not translation. TIRF: 1.10 ± 0.20 → 0.52 ± 0.03 px per block, and the measured ratio from 23.8% high to 0.2% off the author's own output |
| Phase correlation says the alignment is finished | It only sees translation. TIRF: 0.071 px reported while the ratio dipole read 0.381 px whole-field | `dipole_residual`, per block, on the spread |
| The ratio rises with donor brightness and the dimmest pixels all read the same low value | Background left in the channels; dim pixels converge on the ratio of the two backgrounds, 1.10 on this data | Subtract per channel per frame, before dividing |
| The background on the registered channel is low by about the fraction of the frame the warp vacated, and the other channel's is fine | The warp writes exact zeros into the vacated border, and a mean or a low percentile counts them as detector counts | A median absorbs it: 0.78% of the frame zeroed moved the mean 0.78% and the median 0.05%. Otherwise exclude the zeros, on that channel only |
| The fold-change between conditions is roughly half what was expected | A missing or guessed coefficient adds a constant to every ratio, which compresses fold-changes rather than scaling them. Synthetic: a true 3.05x reported as 1.64x | Subtract `BT * DONOR`, and `DE * ACCEPTOR` too — a correct donor leak with direct excitation left out still cost 20-33% of the contrast. Constant acquisition settings do not rescue this |
| The ratio image looks right and the reported number is 3-4x off | The statistic was taken over the frame rather than inside the mask. TIRF: whole-frame mean 1.79 against an in-cell median 6.29 | Report inside the mask |
| Coefficients fitted on the experimental field, and the resting population came out at or below zero | Not identifiable: the sensitized-emission term correlates with the donor channel, so the fit absorbs it and over-subtracts | Controls, or say the fold-change cannot be delivered |
| Ratios do not agree with a previous experiment's although the fold-change does | The ratio was renormalised for display. Synthetic: 165% off in level with contrast intact | Keep the unnormalised ratio for quantification and normalise only a display copy |

## Next steps

- Per-object or per-cell ratios, in physical units, are
  [[calibrated-measurements]] over a segmentation of the donor channel — segment
  on a channel, never on the ratio.
- A time series wants [[drift-correction]] applied to the channels before the
  ratio, and the same warp applied to both so the pair stays registered.
