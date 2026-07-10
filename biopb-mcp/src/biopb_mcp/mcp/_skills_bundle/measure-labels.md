---
id: measure-labels
title: Measure a labels layer into a table
description: Compute per-object region properties from a napari labels layer and return a tidy pandas table.
tags: [measurement, workflow]
version: 1.0.0
requires: [viewer]
---

# Measure a labels layer into a table

**When to use.** The user has a labels layer (e.g. from [[segment-nuclei]]) and
wants quantitative per-object measurements — area/volume, intensity, centroid — as
a table they can inspect or export.

## Steps

1. **Confirm inputs with the user:** which labels layer, and which intensity image
   layer to measure against (if any).

   ```python
   labels = viewer.layers["<labels>"].data
   image = viewer.layers["<intensity>"].data     # optional
   ```

2. **Compute region properties.** Use scikit-image `regionprops_table`, matching the
   dimensionality (2D vs 3D) of the data:

   ```python
   import pandas as pd
   from skimage.measure import regionprops_table

   props = ("label", "area", "centroid", "mean_intensity")
   table = pd.DataFrame(
       regionprops_table(
           np.asarray(labels), intensity_image=np.asarray(image), properties=props
       )
   )
   ```

   `regionprops` is eager, so `.compute()`/`np.asarray` the labels first; for very
   large volumes, run this as a job (see `guide://kernel`) rather than inline.

3. **Show a preview** — the first rows and the object count — and put the full
   table where the user can retrieve it (e.g. assign it to a namespace variable and
   tell them the name).

4. **Offer next steps:** filtering by size/intensity, or exporting to CSV **only if
   the user asks** for a file.

## Guardrails

- Match the property list and dimensionality to what the user actually needs; do
  not dump every available property by default.
- Avoid writing files unless explicitly requested.
- Report the object count and any objects dropped (e.g. touching the border) so the
  numbers are trustworthy.
