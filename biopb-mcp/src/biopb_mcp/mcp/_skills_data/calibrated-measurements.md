---
id: calibrated-measurements
title: Measure labeled objects in physical units, not pixels
description: Report object areas, volumes, and diameters in microns instead of pixels, using the image's real voxel spacing.
tags: [measurement, quantification]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0]
---

# Measure labeled objects in physical units, not pixels

## When to use

Any time a number derived from a Labels layer is going to be reported, compared
across datasets, or filtered against a biological threshold — area, volume,
diameter, length, or a size cutoff. The default `regionprops` output is in
**pixels and voxels**, which is silently wrong the moment voxels are not cubic or
the objective is not 1 µm/px, and anisotropic Z is the norm in confocal stacks.

Use it as the last step of any segmentation workflow, and before any statement of
the form "the objects are N units across".

## When NOT to use

- **Ratios and dimensionless shape descriptors.** `solidity`, `extent`, and
  `intensity_mean` are invariant under spacing (verified) — passing spacing does
  not change them and does not need justifying.
- **Pixel-space work.** Structuring-element radii, `peak_local_max` footprints,
  and crop bounds are array indices. Converting those to microns and back is a
  source of off-by-one errors; keep them in pixels and convert only the reported
  output.
- **When the spacing is genuinely unknown.** If the server has no physical sizes
  and the user does not know them, report pixels and **label them as pixels**. An
  invented spacing is worse than an honest `area_px2`, because it survives into
  the figure.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `IMAGE` | image | either an array from `client` or a layer on `viewer` |
| `LABELS` | int image | array from `client`, a layer on `viewer`, or a temporary name in the kernel |
| `PROPERTIES` | list | e.g. `["label", "area", "centroid", "equivalent_diameter_area", "intensity_mean"]` |
| `UNIT` | — | The unit reported beside the scale, usually `um`. Never assume; a source in nm and one in µm differ by 10⁹ in volume |

`IMAGE` is where the metadata (physical scale and units) lives. `LABELS` may be a temporary result
generated from a previous step. Read `guide://data` before pulling pixels and metadata off data sources.

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status` — `guide://kernel` covers what to do about a gap.

   `IMAGE` comes from either `viewer` or `client`. We need at least one of the two.

2. **Resolve the arrays and the spacing, and show the spacing before measuring.**
   A wrong spacing is invisible in the output table but scales every number in it.

   **Physical scale**: off a layer it is `layer.scale`; off the tensor server it
   is `client.get_physical_scale(array_id)`.

   - **Take it from `IMAGE`, not from `LABELS`.** A Labels layer the agent
     added itself defaults to all-ones scale, and a segmentation uploaded to
     `cache:` carries no physical size at all; both must inherit the image's.
   - **`LABELS` and `IMAGE` must come from the same resolution level.**
     Level 2 labels scored against a level 0 image are wrong by the downsample
     factor, and nothing in the output says so.
   - **Match the spacing to `LABELS`' axes by name, not by position.** The server
     guarantees a canonical `[..., Z, Y, X]` order, which is what makes the two
     arrays comparable at all — but a scale vector is still per-axis, not
     positional. A `[C, Y, X]` image has no Z, and for interleaved colour
     `layer.scale` is one element shorter than the array. Read the axis labels
     (`guide://data`) and pass one entry per axis `LABELS` actually has;
     `spacing` on the wrong axis changes every number and no shape.

3. **Confirm-input** *(blocking)* — only if the spacing is all ones or absent.
   Ask for µm/px in X/Y and Z if 3D, state that measurements are otherwise
   reported in pixels, and proceed either way. Do not guess from the objective or
   the file name.

4. **Measure with `spacing=`.** The only difference from an uncalibrated call:

   ```python
   import pandas as pd
   from skimage.measure import regionprops_table

   df = pd.DataFrame(regionprops_table(
       lab_arr, intensity_image=img_arr, properties=PROPERTIES, spacing=spacing))
   ```

   These arrays are lazy, so this call is what materializes them in full — crop
   first, or promote it to a job, before running it on a whole volume.

   In 3D, `area` **is the volume** — there is no `volume` property. Rename on the
   way out so the unit is carried by the column name, which is what survives into
   a spreadsheet:

   ```python
   d = lab_arr.ndim
   df = df.rename(columns={
       "area": f"{'volume' if d == 3 else 'area'}_{UNIT}{d}",
       "equivalent_diameter_area": f"equiv_diameter_{UNIT}",
   })
   ```

5. **Visual check** *(non-blocking)*. Report the object count, the median and
   IQR of the size column, and the fraction touching the border, then say whether
   the median is plausible for the stated object type. This is the step that
   catches a spacing that is off by 10³: a nucleus of 500 µm³ is wrong in a way
   that 500 voxels is not.

6. **Hand back the table and the exact spacing used.** End with the parameter
   dict, so the run is reproducible and a later batch pass does not re-derive it:

   ```python
   # layer names, or array_ids when the run came off the tensor server
   print({"labels": LABELS_REF, "image": IMAGE_REF,
          "spacing": spacing, "unit": UNIT, "properties": PROPERTIES,
          "n_objects": int(len(df))})
   ```

## Guardrails

- **A lazy array handed to `regionprops_table` materializes it.** Measure a
  computed crop or a single plane while iterating, and promote the full-volume
  pass to a job.
- **Centroids come back in physical units when `spacing` is given** — 7.5 µm, not
  row 1. Re-deriving array indices from them silently indexes the wrong voxel;
  measure a second time without spacing if indices are needed.

## Failure modes

`regionprops` behaviour as of `scikit-image` 0.26.

| Symptom | Cause | Fix |
|---|---|---|
| 3D volumes too small by exactly the Z:XY ratio | `spacing` omitted, or given as XY only | Pass the full per-axis vector; in 3D `area` is the volume |
| `NotImplementedError: perimeter supports isotropic spacings only` | `perimeter` / `perimeter_crofton` reject anisotropic spacing | Measure perimeter on an isotropic plane, or resample to isotropic first |
| `eccentricity` / `axis_major_length` shift when spacing is added | Correct, not a bug — anisotropic voxels genuinely change the fitted ellipse | Report the calibrated value; note the spacing alongside it |
| `ValueError` on the length of `spacing`, or Z-sized numbers on a channel axis | `layer.scale` read positionally — a non-spatial axis counted as one, or an interleaved-colour layer whose scale is one element short (napari does not count the samples axis) | Match entries to `LABELS`' axis labels, not to positions (step 2) |
