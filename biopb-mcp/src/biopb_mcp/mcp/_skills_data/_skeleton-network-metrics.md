---
id: skeleton-network-metrics
title: Measure the length and branching of a segmented filament network
description: Measure how long a filament network is and how branched, in physical units — mitochondria, vessels, neurites or cytoskeleton, from a mask you already have.
tags: [morphology, quantification, network, skeleton, 3d]
version: 1.0.0
checklist: [viewer, pkg:biopb-mcp>=0.13.0, pkg:skan~=0.13.1, pkg:networkx~=3.6.1]
---

# Measure the length and branching of a segmented filament network

## When to use

A binary mask of something thin and connected — a mitochondrial network, a
vascular bed, neurites, an actin or septin network — and the question is how
much of it there is and how it is wired: total length, how many branches, how
many junctions, how many closed loops. The classic Fiji answer is
AnalyzeSkeleton, and this is the same measurement with the decisions written
down.

## When NOT to use

- **The objects are compact rather than thin.** Area, volume, eccentricity and
  the rest are `regionprops`, and in physical units they are
  [[calibrated-measurements]]. A skeleton of a blob is a meaningless star.
- **There is no segmentation yet.** This measures a mask; it does not make one,
  and every number below inherits the mask's mistakes directly — see the last
  guardrail.
- **The volume does not fit in memory.** Skeletonization is global: a thinning
  that has not seen the whole object thins it wrongly, and there is no correct
  per-chunk version. Crop to the cell or the field first.
- **The question is only how many separate pieces there are**, as in
  fragmentation scoring. That is `scipy.ndimage.label` on the mask and needs
  none of this — though step 7 reports it too, since a graph gives it away.

## Parameters

Everything below is in microns, and the first line of the table is the one the
whole measurement rests on.

| Name | Unit | How to derive it |
|---|---|---|
| `SPACING` | µm/voxel | **From the operator, per axis, and never assumed cubic.** A confocal z step is routinely 3-8x the lateral pixel. Measured on a 5:1 stack: reading it as cubic voxels reports **79.8%** of the true length, silently |
| `DIAMETER` | µm | `2 * median(distance_transform_edt(mask, sampling=SPACING)[skeleton])` — the median local radius along the skeleton, doubled. Take the median, not the max: one lump on a ragged mask doubled the answer here (1.0 µm to 2.0 µm) |
| `MIN_BRANCH` | µm | The shortest side branch the biology really makes — a *confirm-input* question. Default to `2 * DIAMETER` when nobody can say. Anything from 1 to 5 µm gave the identical answer on the network this was measured on, so it is not a knob to tune |
| `MERGE` | µm | How close two branch points have to be to be one junction. `DIAMETER`: a junction is as wide as the filaments crossing in it. 0.5 to 3 µm were indistinguishable here |
| `CHORD_STEP` | µm | `max(2 * DIAMETER, 2 * max(SPACING))` — how far apart to take the points a length is summed over (step 6). Values from 0.8 to 3 µm all landed within 7% of the truth, against 27% for summing voxel steps |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status`; `guide://kernel` covers what to do about a gap. `skan` is
   the branch-graph library everything below is written against and it installs
   into a biopb environment without moving anything else.

   **The degraded path is a different measurement, not a worse one.** Without
   `skan` you can still count skeleton voxels and voxels with three or more
   neighbours, and on the network measured here that returned **76%** of the
   true length, 25 branches for 11 and 29 junctions for 5. Report those as
   *relative* numbers between images acquired identically, never as microns or
   as counts, and say in the hand-off which you did.

2. **Ask about the sample** *(confirm-input, blocking)*. One question, three
   facts, none of them in a boolean mask. Put the mask on the viewer before
   asking, and say what you can see.

   - **What is the voxel size, in each axis separately?** This is the fact the
     answer scales with. If nobody knows, stop and say the length cannot be
     given in microns — a number in voxels is honest and a wrong number in
     microns is not.
   - **How short can a real side branch be?** Sets `MIN_BRANCH`. A thresholded
     mask is lumpy, and every lump becomes a branch a micron or two long that
     looks exactly like a small real one.
   - **Is the network expected to be one piece or several?** Fragmentation is
     usually the readout, and it decides whether a second component in step 7
     is a finding or a segmentation error.

3. **Skeletonize the mask on the grid it was acquired on.**

   ```python
   import numpy as np
   from scipy import ndimage as ndi
   from skimage.morphology import skeletonize

   skeleton = skeletonize(mask.astype(bool))
   radius = ndi.distance_transform_edt(mask, sampling=SPACING)
   DIAMETER = float(2 * np.median(radius[skeleton]))
   ```

   **Resampling the volume to isotropic first is not the fix**, though it is the
   natural move for an anisotropic stack. It treats a symptom: it does pull a
   step-summed length down — 168.5 µm on the acquired grid against 154.0 µm
   resampled, on a network whose true length is 132.5 µm — and it leaves the
   error at 16%, because the bias is in how the length is accumulated (step 6),
   not in the grid. Once the length is taken over chords, resampling moves the
   answer by up to 7% and never toward the truth. Skeletonize on the grid the
   stack was acquired on, and handle the anisotropy in the measurement.

4. **Turn the skeleton into a graph, with the spacing.**

   ```python
   from skan import Skeleton, summarize

   sk = Skeleton(skeleton, spacing=SPACING)
   ```

   `spacing` is not cosmetic and it is not the same as scaling the answer at the
   end: it is what makes a step between planes count for 0.5 µm and a step
   within one count for 0.1 µm.

5. **Prune spurs, repeatedly, before counting anything.**

   ```python
   for _ in range(10):
       branches = summarize(sk, separator="_")
       spurs = branches.index[(branches.branch_type == 1)
                              & (branches.branch_distance < MIN_BRANCH)]
       if not len(spurs) or len(spurs) == len(branches):
           break
       sk = sk.prune_paths(np.asarray(spurs))
   branches = summarize(sk, separator="_")
   ```

   `branch_type == 1` is junction-to-endpoint — a dead end. Never prune on
   length alone: a short branch between two junctions is a real short
   connection, and dropping it changes the topology.

   **The loop matters.** Pruning a spur can expose the one behind it, and one
   pass leaves those. Measured on a mask with ordinary threshold roughness,
   pruning took the branch count from **23 to 11** and the junction count from
   **10 to 5**, against a truth of 11 and 5.

6. **Take the length over chords, not over voxel steps.**

   ```python
   def branch_length(path):
       p = sk.path_coordinates(path) * np.asarray(SPACING)
       keep = [0]
       for i in range(1, len(p)):
           if np.linalg.norm(p[i] - p[keep[-1]]) >= CHORD_STEP:
               keep.append(i)
       if keep[-1] != len(p) - 1:
           keep.append(len(p) - 1)
       return float(np.linalg.norm(np.diff(p[keep], axis=0), axis=1).sum())
   ```

   **This is the step that is wrong by default.** `skan`'s `branch_distance`
   column — and every hand-written "sum the distance between consecutive
   skeleton voxels" — measures the staircase the filament was digitised into
   rather than the filament. It always overestimates, and on anisotropic voxels
   it overestimates badly, because one step between planes buys 0.5 µm of
   measured length for a rise the filament made over several microns. Measured
   on single straight filaments of known length: **108% to 136%** of the truth
   at 5:1 spacing, and still 105% to 118% on cubic voxels. On the network:
   **127%** by steps, **96%** by chords.

7. **Merge the branch points that are one junction, then read the numbers off.**

   ```python
   import networkx as nx

   g = nx.MultiGraph()
   for i in range(len(branches)):
       g.add_edge(int(branches.node_id_src[i]), int(branches.node_id_dst[i]),
                  length=branch_length(i))
   thick = [(u, v) for u, v, d in g.edges(data=True)
            if d["length"] < MERGE and g.degree(u) >= 3 and g.degree(v) >= 3]
   for u, v in thick:
       if u in g and v in g and u != v:
           nx.contracted_nodes(g, u, v, self_loops=False, copy=False)
   g.remove_edges_from(list(nx.selfloop_edges(g)))

   total_length_um = sum(d["length"] for _, _, d in g.edges(data=True))
   n_branches = g.number_of_edges()
   n_junctions = sum(1 for _, k in g.degree() if k >= 3)
   n_endpoints = sum(1 for _, k in g.degree() if k == 1)
   n_pieces = nx.number_connected_components(g)
   n_loops = n_branches - g.number_of_nodes() + n_pieces
   ```

   **Where two filaments cross, the skeleton does not make one branch point.**
   It makes two or three, a fraction of a micron apart, joined by stubs — a
   junction as wide as the filaments in it. Counting them as they come reported
   **8 junctions for 5**, and three branches that exist only between them.
   Merging on a *physical* distance is what makes the count independent of how
   finely the stack was sampled.

8. **Look at it, and report the numbers together** *(visual check)*. Put the
   skeleton on the viewer over the mask and screenshot one crop. Never the
   screenshot alone — report `total_length_um`, `n_branches`, `n_junctions`,
   `n_pieces`, **how many spurs step 5 dropped**, and `DIAMETER`.

   Those last two say whether the procedure ran as intended. A spur count of 0
   on a thresholded mask means `MIN_BRANCH` never bit; a `DIAMETER` far from what
   the operator described means the mask is not the object they think it is, and
   every length above is then measuring something else.

9. **Hand back the numbers and the spacing they were computed with.** Lengths in
   microns are comparable between acquisitions and voxel counts are not, which is
   usually the entire reason for doing this — so the spacing travels with the
   result, or the next stack gets compared against these wrongly.

## Guardrails

- **Do not tune `MIN_BRANCH` until the branch count looks right.** It is set by
  what the sample does, asked for in step 2; from 1 to 5 µm it changed nothing
  here, and by the time it does, the answer is set by an arbitrary cutoff rather
  than by the biology.
- **Do not report a length in microns from a spacing nobody confirmed.** It is
  the one error in this procedure that is invisible in every check: the picture
  looks right, the counts are right, and the number is off by the axial ratio.
- **Every number here inherits the segmentation.** A mask that bridges two
  filaments that only touch invents a junction and merges two branches into one;
  a mask that breaks a filament invents two free ends. If nobody has checked it,
  [[segmentation-qc-metrics]] is how that gets scored, and the branch and
  junction counts are the first things to distrust — length is the most robust
  of the three.

## Failure modes

Every row was hit while measuring this, on `scikit-image` 0.26 and `skan` 0.13,
on a 5:1 anisotropic stack of a 132 µm network with 11 branches and 5 junctions.

| Symptom | Cause | Fix |
|---|---|---|
| Length is 20-35% higher than a hand measurement, and looks worse on stacks with a coarse z step | Length summed between consecutive skeleton voxels — `branch_distance`, or an equivalent loop | Sum over chords (step 6). Measured: 127% by steps, 96% by chords; 108-136% on single filaments |
| Length is about 80% of the expected value on an anisotropic stack | The z step was treated as equal to the pixel size | `spacing=` in step 4, from step 2's answer. Measured: 79.8% |
| Twice as many branches as the network appears to have, and many very short ones | Spurs from a ragged mask never pruned | Step 5, in a loop. Measured: 23 branches for 11, 10 junctions for 5 |
| A few more junctions than there are crossings, joined by sub-micron branches | Each crossing gives several adjacent branch points | Merge on a physical distance (step 7). Measured: 8 for 5 |
| `DIAMETER` comes out about twice what the operator describes | Taken as the maximum of the distance transform, which one lump on the mask sets | Median along the skeleton (step 3). Measured: 2.0 µm against a true 1.0 µm |
| Resampling to isotropic first, and the length is still too high | The bias is in how the length is accumulated, not in the grid, so resampling only reduces it | Step 6. Measured: 168.5 µm on the acquired grid, 154.0 µm resampled, against a truth of 132.5 |

## Next steps

- Per-branch lengths are already in `branches`; a length histogram separates a
  network that lost total length from one that fragmented at the same length.
- Counting anything *inside* the network — puncta on the filaments, say — needs
  the mask as a parent, and lengths per parent in physical units is
  [[calibrated-measurements]].
