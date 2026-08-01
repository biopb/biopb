---
id: drift-correction
title: Correct stage drift in a time series before measuring
description: Register a time-lapse whose field of view has drifted, so the same object stays at the same coordinates across frames.
tags: [registration, time-series, preprocessing]
version: 1.0.0
requires: [viewer, tensor, dask, pkg:biopb-mcp>=0.13.0, pkg:pystackreg~=0.2.8]
---

# Correct stage drift in a time series before measuring

## When to use

A time-lapse where the field of view has moved between frames — thermal drift,
a bumped stage, an imperfect return from a multi-position loop. Anything that
reads the same pixel across time needs this first: intensity traces, tracking,
kymographs, ratio images.

## When NOT to use

- **The objects are moving, the field is not.** Cells crawling in a still field
  is tracking, not registration. Registering it would cancel the motion you are
  trying to measure. If both are happening, correct the drift and *then* track.
- **The drift is deformation, not rigid motion.** A sample that swells, contracts
  or shears cannot be composed by translation and rotation. That is elastic
  registration, a different job — say so rather than fitting a rigid model to it.
- **Absolute stage coordinates are the measurement.** Registration rewrites where
  things are. If the position itself is the result, keep the raw movie and
  register a copy.
- **A single frame, or frames of unrelated fields.** Nothing to register against.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `MOVIE` | `(T, Y, X)` | One channel of the time series, one resolution level, in acquisition order. Read `guide://data` first — pyramid level and laziness both bite here |
| `REF_CHANNEL` | — | **One structural** channel that persists across the movie. Not a mean projection over channels — that mixes in the very channel whose intensity is the measurement. Never a sparse or blinking one |
| `MODE` | — | `TRANSLATION` unless rotation is visible. `RIGID_BODY` adds rotation, costs ~1.5x, and is the only option that handles a rotated stage return |
| `REFERENCE` | — | **`"previous"`.** See step 3 — this is not a preference |
| `MAX_SHIFT` | px | The largest believable drift, from the acquisition. Used to *reject* a result, not to constrain the fit |

Drift correction is estimated on one channel and **applied to all of them**. A
per-channel fit would shift channels relative to each other and manufacture a
chromatic error that was not there.

## Steps

1. **Check the requirements** *(blocking)*. Resolve `requires:` against
   `server_status`, and `import pystackreg` for the `pkg:` token —
   `guide://kernel` covers what to do about a gap.

   Without `pystackreg`, the degraded path is `skimage.registration.
   phase_cross_correlation`, which is translation-only and less precise but needs
   nothing extra. It is a real fallback, not a lesser one — see step 4 — but say
   which was used, because a corrected movie does not show its provenance.

2. **Confirm what is drifting** *(confirm-input, blocking)*. Show the user the
   first and last frames, and ask whether the *field* moved or the *objects* did.
   These look identical in a single frame and the correction for one destroys the
   other. Confirm `REF_CHANNEL` in the same question.

3. **Register against the previous frame, not the first.**

   ```python
   from pystackreg import StackReg
   sr = StackReg(getattr(StackReg, MODE))
   tmats = sr.register_stack(MOVIE, reference="previous")
   ```

   `reference="previous"` is load-bearing, not a tuning choice. StackReg is a
   local pyramid optimiser with a limited capture range: against frame 0 the
   displacement grows without bound over a long movie, and once it exceeds that
   range the fit does not fail loudly — it returns a confident, wrong transform.
   Measured on synthetic movies with known drift, `reference="first"` lost lock on
   2 of 4 runs at 1.7 px/frame and **4 of 4** at 4 px/frame, returning errors of
   20+ px; `reference="previous"` held at ~0.01–0.05 px throughout. Each
   consecutive pair is a small displacement, and pystackreg composes them.

4. **Check the trajectory before applying anything** *(blocking)*. Extract the
   per-frame offsets and look at them as a series, not as a summary:

   ```python
   dy = np.array([m[1, 2] for m in tmats])
   dx = np.array([m[0, 2] for m in tmats])
   step = np.hypot(np.diff(dy), np.diff(dx))
   print(f"total {np.hypot(dy[-1]-dy[0], dx[-1]-dx[0]):.1f} px, "
         f"largest single-frame step {step.max():.1f} px")
   ```

   Real drift is **smooth and slow**. Report the total excursion and the largest
   single-frame step, and stop if either exceeds `MAX_SHIFT` — a jump of tens of
   pixels in one frame is a lost registration, not a fast stage. Plot the two
   series and put them in front of the user; a staircase or a sudden reversal is
   obvious in the plot and invisible in a mean.

   **"Frame-to-frame correlation went up" is not this check.** It rises whenever
   the frames are more aligned on average, so it stays comfortable while one
   frame sits 30 px out — which is exactly the failure that matters, because
   `reference="previous"` carries it into every frame after it.

5. **Apply the transforms to every channel.**

   ```python
   corrected = sr.transform_stack(MOVIE, tmats=tmats)
   ```

   Reuse `tmats` across channels — do not re-register per channel. Interpolation
   resamples intensities, so if the measurement is photon counts, note that the
   corrected movie is for localisation and the raw one for photometry.

6. **Handle the edges honestly.** Registration slides data off one side and
   invents nothing on the other, so the borders accumulate frames where some
   channels have no data. Either crop to the common valid region —
   `int(np.ceil(np.abs(dy).max()))` and the same in x — or state that the margins
   are extrapolated. A measurement over the full frame silently includes them.

7. **Publish the corrected movie and the settings.** Upload it if it is worth
   keeping, and print the dict that reproduces it — `guide://data` covers upload
   and pixel size, which does not ride along by default.

   ```python
   print({"mode": MODE, "reference": "previous", "ref_channel": REF_CHANNEL,
          "total_drift_px": float(np.hypot(dy[-1]-dy[0], dx[-1]-dx[0])),
          "max_step_px": float(step.max()), "method": METHOD})
   ```

## Guardrails

- **Never register on the measurement channel.** A channel that bleaches, blinks
  or responds to a stimulus changes its own content over time, and an intensity
  registration will chase that change instead of the stage.
- **`reference="previous"` accumulates.** It composes per-frame transforms, so a
  single bad frame propagates into every frame after it. That is why step 4 looks
  at the per-frame steps and not just the total.
- **If the fallback is used, the answer is translation-only.** Do not report a
  rotation that was never estimated.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Recovered drift is ~0 for every frame while the movie visibly moves | `phase_cross_correlation` with its default `normalization="phase"` on a smooth, low-contrast image — it whitens frequency bins holding only numerical noise and buries the true peak | Pass `normalization=None`. Verified: a true (1.19, −0.68) px shift returns (0.0, −0.05) by default and (1.10, −0.55) with it |
| One frame's offset jumps by tens of px, then everything after is shifted | StackReg lost lock, and `reference="previous"` propagated it | Step 4 catches it; re-register that pair with `RIGID_BODY`, or exclude the frame |
| Steady, believable drift is reported but the movie is not stabilised | Registered on a channel whose content changes, so the fit tracked biology | Re-register on a structural channel |
| Objects are still moving after correction | They were moving in the sample; the field never drifted | Track instead; see next steps |
| Corrected movie is stable but intensities changed | Interpolation resampled the pixels | Expected — measure photometry on the raw movie |
| Registration is stable but measurements drift at the borders | The invalid margin was included | Crop to the common valid region (step 6) |
| `RIGID_BODY` rotation looks large and implausible | A near-symmetric field has no rotational information | Use `TRANSLATION`; report that rotation was not estimable |

## Next steps

- Measure intensity traces on the corrected movie — the same pixel is now the
  same place across frames, which is what a trace assumes.
- Report sizes and positions from it with [[calibrated-measurements]], after
  carrying the pixel spacing across (step 7).
- If objects move *within* the stabilised field, that is tracking, and it starts
  from this movie rather than the raw one.
