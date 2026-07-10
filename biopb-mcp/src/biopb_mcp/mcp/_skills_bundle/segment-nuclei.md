---
id: segment-nuclei
title: Segment nuclei with a segmentation op
description: Run a server-side segmentation op over the active image layer and load the instance labels for validation.
tags: [segmentation, ops]
version: 1.0.0
requires: [viewer, "ops:segmentation"]
---

# Segment nuclei with a segmentation op

**When to use.** The user has a 2D or 3D fluorescence image loaded and wants
instance labels for nuclei (or cells). This uses a server-side image-processing op,
not local code.

## Steps

1. **Identify the input.** Confirm with the user which image layer and channel to
   segment. Do not assume the active layer is the intended one.

   ```python
   layer = viewer.layers["<name>"]     # confirmed with the user
   ```

2. **Discover the op.** Segmentation ops are exposed in the `ops` namespace; list
   what the connected servers actually provide (see `guide://ops`) rather than
   hard-coding an op name.

3. **Run it lazily**, keeping the result as a dask array until the user validates:

   ```python
   labels = ops["segmentation"](layer.data, min_score=0.4)   # params per the op's signature
   ```

   For long-running segmentation, submit it as a job and poll (see `guide://kernel`
   for the job pattern and cancellation) instead of blocking a tool call.

4. **Load the labels** so the user can check them against the raw image:

   ```python
   viewer.add_labels(labels, name=f"{layer.name}-nuclei")
   ```

5. **Report** the label count and ask the user to confirm the segmentation looks
   right before any downstream measurement.

## Guardrails

- Prefer lazy dask; `.compute()` only the final result the user asked for.
- Put intermediate results on `viewer` at each step for validation.
- Tune parameters (score threshold, size hint, NMS) with the user — they know the
  data; do not silently pick values.
- Follow up with [[measure-labels]] once the labels are accepted.
