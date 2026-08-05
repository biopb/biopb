---
id: deconvolve-widefield
title: Deconvolve a widefield z-stack
description: Restore a 3D widefield fluorescence stack blurred along z, and check that the restoration actually restored something.
tags: [restoration, deconvolution, preprocessing]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0]
---

# Deconvolve a widefield z-stack

Every number below was measured on a synthetic widefield stack —
`(40, 160, 160)`, NA 1.4 oil, 520 nm emission, immersion RI 1.515, voxels
0.20 x 0.065 x 0.065 µm, ~900 peak photons, and a generating PSF carrying
0.40 waves RMS of aberration. Treat them as the shape of the effect, not as
targets for your data.

## When to use

A 3D widefield fluorescence stack is smeared along z and hazy with
out-of-focus light, and the user wants it restored before *looking* at it or
measuring shape, size or resolution on it.

## When NOT to use

- **Before quantifying intensity.** This is the one that costs people results.
  Richardson-Lucy makes photometry *worse*, and not because your PSF is bad:
  brightness correlation against truth fell 0.820 (raw) to 0.670 after RL, and
  deconvolving with the exact generating PSF scored 0.667 — identical. It is
  intrinsic to the method. Measure on the raw stack. If something must be
  restored first, a regularised linear filter (Wiener) held 0.949-0.961 while
  restoring far less.
- **Confocal, light-sheet or already-deconvolved data.** The PSF is a different
  shape and the out-of-focus term this is built around is largely gone.
- **The stack is undersampled along z.** Nothing recovers detail finer than the
  z-step; a 1 µm step through a 0.7 µm axial PSF has already thrown it away.
- **Tiled or drift-affected input.** Restore per position after
  [[stitch-tiles]] / [[drift-correction]], not across a seam.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `IMAGE` | `(Z, Y, X)` | The stack, in acquisition order. `guide://data` for getting it off a layer or the tensor server |
| `BEADS` | `(Z, Y, X)` or `None` | A sub-resolution bead stack from the **same mount, depth and optics**. This is the single biggest lever and it is a question for the user, not something to infer — ask in step 2 |
| `SPACING` | µm | `(dz, dy, dx)`. From the acquisition; ask. The z-step is a separate number from the lateral pixel, never the same one |
| `NA`, `LAMBDA_NM`, `RI` | — | Objective NA, emission wavelength, immersion index. Used **only** for the no-beads fallback and for sizing windows |
| `PSF_SHAPE` | voxels | Odd per axis, and large enough to hold the visible hourglass: roughly `4 x 0.61 λ / NA` laterally and `4 x RI λ / NA²` axially, converted through `SPACING`. It must fit inside the stack with room for a bead's whole crop |
| `ITERS` | — | **Not a bias/variance knob on well-sampled data.** RL was monotone here to 640 iterations and then plateaued, at 900, 120 *and* 30 peak photons — it never turned over. Start at 40, and stop when the axial FWHM in step 6 stops moving. Cost: photometry decays as resolution improves (r 0.741 at 10 iterations, 0.618 at 160), which is the real trade |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status`; `guide://kernel` covers a gap.

2. **Confirm the inputs** *(blocking)*. Three things, and only the first is
   about pixels:

   - the voxel spacing, `(dz, dy, dx)` in µm;
   - **whether a bead stack exists** — see step 4 for what it is worth;
   - **whether anything will be quantified from the result.** If yes, say what
     *When NOT to use* says and let the user decide before you spend the run.

3. **Subtract the camera pedestal.** RL models Poisson photons; a DC offset is
   not signal, and left in it biases every ratio in the update toward 1.
   Measured: skipping this cost more than half the restoration — reconstruction
   fidelity 0.413 with it, 0.165 without, and photometry 0.670 against 0.421.

   ```python
   import numpy as np
   img = np.asarray(IMAGE, float)
   img = np.clip(img - np.percentile(img, 0.5), 0, None)
   ```

4. **Build the PSF.** Measure it from the beads if they exist; the theoretical
   model is the fallback, not the default.

   Why it matters this much: a real high-NA objective looking into an aqueous
   mount has spherical aberration that **is not in NA/λ/RI** — nothing derivable
   from the metadata can produce it. The measured PSF matched the true one at
   NCC 0.945; the best theoretical model of the same optics managed 0.545. That
   propagated straight through to the result (step 6's table).

   ```python
   from scipy import ndimage as ndi
   from skimage.feature import peak_local_max

   vol = np.clip(np.asarray(BEADS, float) - np.percentile(BEADS, 0.5), 0, None)
   half = [s // 2 for s in PSF_SHAPE]
   peaks = peak_local_max(
       ndi.gaussian_filter(vol, 1.0), min_distance=20, num_peaks=40,
       exclude_border=False,          # see the failure table
   )
   subs = [
       vol[tuple(slice(c - h, c + h + 1) for c, h in zip(p, half, strict=True))]
       for p in peaks
       if all(c - h >= 0 and c + h + 1 <= n
              for c, h, n in zip(p, half, vol.shape, strict=True))
   ]
   psf = np.clip(np.mean(subs, axis=0), 0, None)
   psf[psf < 0.01 * psf.max()] = 0.0   # the tails are noise, and RL amplifies them
   psf /= psf.sum()
   ```

   Crop each bead around **its own** brightest voxel and average. Do not
   register the crops onto one chosen bead: that inherits that bead's sub-voxel
   offset and translates the whole reconstruction. Measured, this plain version
   beat a sub-voxel-aligned one, 0.961 against 0.935.

   With no beads, build a diffraction-limited PSF from `NA`, `LAMBDA_NM`, `RI`
   and `SPACING` — any standard model — and **tell the user what it costs**: on
   this stack that route reached axial FWHM 1.48 µm where the measured PSF
   reached 0.52 µm, against an input of 1.94 µm. It is a third of the available
   restoration.

5. **Run Richardson-Lucy.**

   ```python
   from skimage.restoration import richardson_lucy
   scale = img.max()
   out = richardson_lucy(img / scale, psf, num_iter=ITERS, clip=False) * scale
   ```

   `clip=True` is the **default** and clamps the output to `[-1, 1]`: on raw ADU
   data it returns a flat 1.0. Either normalise as above or pass `clip=False`;
   doing neither destroys the result silently.

6. **Prove the restoration restored something.** This is the step, and it is
   the one an unaided run leaves out.

   ```python
   def axial_fwhm(vol, zyx, dz):
       """FWHM along z through a point-like object, in µm, measured locally."""
       z, y, x = zyx
       lo, hi = max(0, z - 9), min(vol.shape[0], z + 10)
       p = np.asarray(vol[lo:hi, y, x], float)
       p = p - p.min()
       half, pk = p.max() / 2.0, int(np.argmax(p))
       a = b = pk
       while a > 0 and p[a] >= half:
           a -= 1
       while b < len(p) - 1 and p[b] >= half:
           b += 1
       return (b - a) * dz
   ```

   Pick a few isolated point-like objects, measure before and after, and
   **report both numbers**. Three rules about reading them:

   - **A rising peak is not divergence.** RL concentrates a point spread over
     ~1200 voxels back toward one, so peak intensity climbs by orders of
     magnitude *because it is working*: measured, 409 to 305 662 ADU over 1 to
     240 iterations, while the fit residual fell monotonically 0.670 to 0.272
     and the axial FWHM reached the true value exactly. Judge by the residual or
     by FWHM, never by the peak.
   - **Never validate on the bead stack you built the PSF from.** It is
     circular and it cannot fail: that check reported 1.85 → 0.47 µm with the
     right PSF and still 1.85 → **1.05 µm** with a PSF from an entirely
     different microscope, while on real data those two differ threefold.
   - **A filter that did nothing looks clean.** No artifacts, no ringing,
     conserved flux, and an axial FWHM that has barely moved. If before and
     after are within a few percent, the run failed — say so rather than
     shipping it.

   For reference, what the routes reach on this stack:

   | route | axial FWHM | fidelity |
   |---|---|---|
   | raw input | 1.94 µm | 0.082 |
   | theoretical PSF | 1.48 µm | 0.211 |
   | PSF measured from beads | 0.52 µm | 0.413 |
   | exact generating PSF (unreachable) | 0.44 µm | 0.587 |

7. **Put it back on the viewer** next to the input, same contrast limits, so
   the user can see the difference rather than take it on trust.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Output is flat 1.0, or the dynamic range collapsed | `richardson_lucy`'s `clip=True` default clamps to `[-1, 1]` | Normalise by `img.max()` first, or pass `clip=False` |
| "RL is diverging" — peak grows 100x+ | It is converging. Peak went 409 → 305 662 while the residual fell 0.670 → 0.272 | Judge by residual or axial FWHM, not peak |
| No beads found; PSF is the fallback although a bead stack was given | `peak_local_max`'s `exclude_border` defaults to `min_distance`, which in a 40-plane stack leaves one legal z and finds **nothing** | `exclude_border=False` |
| Beads are found, but every one is rejected and the average is empty | A `PSF_SHAPE` crop 31 planes deep cannot fit around a bead lying near the top or bottom of a 40-plane stack | Shrink `PSF_SHAPE` along z, or reacquire with the beads nearer mid-stack; a clipped bead must stay rejected, since it biases the average asymmetrically |
| Result looks clean but axial FWHM barely moved | Over-regularised, or too few iterations. A Wiener balance 1000x too high scored 1.92 µm where 1e-5 reached 0.86 µm | Step 6, then raise `ITERS` or lower the regularisation |
| Restoration is mediocre and the PSF was theoretical | Real aberration is absent from NA/λ/RI: 0.545 PSF fidelity against 0.945 measured | Get a bead stack; say what the fallback costs |
| Intensities no longer proportional to the raw stack | Intrinsic to RL, not a bug — the oracle PSF degrades photometry identically | Quantify on the raw stack; see *When NOT to use* |
