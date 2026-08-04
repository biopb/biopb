---
id: track-objects
title: Track segmented objects through a time-lapse
description: Follow each segmented object across the frames of a time-lapse, so cells keep one identity over time and their lineage survives division.
tags: [tracking, time-series, quantification]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0, pkg:laptrack~=0.17.1]
---

# Track segmented objects through a time-lapse

## When to use

Objects are already segmented frame by frame — a `(T, Y, X)` label image, or a
table of per-frame centroids — and the label ids mean nothing across frames.
This assigns each object one identity for the whole movie, closes the frames
where it was missed, and keeps lineage where cells divide.

## When NOT to use

- **The field of view moved.** Stage drift adds itself to every cell's
  displacement, so a cutoff sized for the biology links the wrong pairs.
  [[drift-correction]] first, then track the corrected movie.
- **The segmentation is the problem.** Touching cells merged into one label, or
  one cell split into two, arrive here as an object appearing and disappearing.
  Linking cannot repair that, it can only carry it — fix segmentation first.
- **Counting, not following.** Objects per frame, area over time, intensity in a
  fixed region: none of these need identity, and tracking adds a large failure
  surface for nothing.
- **Nothing distinguishes the objects.** If a typical step is comparable to the
  spacing between neighbours and the objects are alike in size and shape, then
  neither where they are nor what they look like identifies them. Acquire
  faster; no linker recovers this.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `LABELS` | `(T, Y, X)` | The segmented series, one label image per frame, in acquisition order. Ids need not agree across frames — that is what this produces. 3D is `(T, Z, Y, X)` and changes only the coordinate columns in step 3. `guide://data` for getting it out of a layer or off the tensor server |
| `METRIC` | — | `laptrack` takes any `cdist` metric **or a callable**, so this is a real choice, and it sets the units of `CUTOFF` and the columns in step 3. **Centroid distance** (`sqeuclidean`, the default) — every number in this skill was measured on it. With full masks, prefer **`1 - gIoU`** as a callable (step 4): it follows the mask, so growth and shape change stop reading as motion, and neighbours ambiguous by position are unambiguous by their footprints. Use *generalized* IoU, not IoU — plain IoU is 0 for every pair that does not overlap, so it cannot rank them and needs objects that move less than their own size; gIoU keeps falling as they separate and is applicable wherever centroid distance is. On anisotropic voxels put the coordinates in µm — one pixel cutoff cannot mean one speed limit along both z and y |
| `PIXEL_UM` | µm/px | From the acquisition. Ask (step 2) — no pixel carries it. With a z-step, ask for that too: it is a second number, not the same one |
| `INTERVAL_S` | s | Seconds between frames, likewise from the acquisition |
| `CUTOFF` | depends on `METRIC` | **Centroid:** `MAX_STEP_PX = max_speed_um_per_min * (INTERVAL_S / 60) / PIXEL_UM`, the largest step one object can plausibly take between two frames, from what the user says their cells do — cross-checked against the data in step 6. **Do not take the library default**, which is 15 px whatever your magnification and frame rate are. **gIoU:** a number in `[0, 2)` — 0 for identical masks, **1.0 for two that exactly touch**, and rising toward 2 as they separate (for equal boxes: 1.33 one width apart, 1.5 two widths). So a cutoff at or below 1.0 silently means *"must overlap"*, which throws away what gIoU is for; put it above 1.0 by as much as the largest separation you would accept, and check it in step 6 like any other |
| `MAX_GAP` | frames | One more than the longest run of frames an object may be missing. An object absent from a *single* frame reappears at a frame difference of **2**, so the smallest useful value is 2 |
| `DIVIDES` | — | Whether these objects divide, and whether the user wants lineage. From the user, not from the images |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status`; `guide://kernel` covers what to do about a gap. Read the
   `pkg:laptrack` version with `importlib.metadata.version("laptrack")` and not
   `laptrack.__version__` — the module attribute reads `0.17.0` inside the
   `0.17.1` distribution, so the import-time check reports a correctly installed
   package as too old.

   Without `laptrack`, the degraded path is a global assignment per frame pair
   (`scipy.optimize.linear_sum_assignment` over the pairwise distances, `MAX_STEP_PX`
   as the admissible cost) — step 4 without the second round. It is a real
   fallback: on the movie described in step 6 it recovers **87.0%** of the true
   links against the full method's **90.1%**. What it does not do is close gaps
   or handle divisions, so it fragments a track wherever one detection is
   missing — 190 tracks for 65 cells rather than 161, and no lineage at all.
   Say which was used.

2. **Confirm the scale and the biology** *(confirm-input, blocking)*. Four facts,
   one question: the pixel size, the frame interval, the fastest these objects
   plausibly move (in µm/min — a number the user has, from their own biology),
   and whether they divide. The first two set every physical number this produces;
   the third sets `MAX_STEP_PX`; the fourth decides whether splitting is on.

3. **Build the detection table.** One row per `(frame, label)`, and **keep the
   label column** — step 5 maps the answer back through it.

   ```python
   from skimage.measure import regionprops
   rows = [(t, r.label, *r.centroid)
           for t, frame in enumerate(np.asarray(LABELS))
           for r in regionprops(frame)]
   det = pd.DataFrame(rows, columns=["frame", "label", "y", "x"])
   det = det.sort_values("frame").reset_index(drop=True)
   ```

   Add `"z"` before `"y"` for a 3D series, and scale the coordinates into µm if
   the voxel is anisotropic (see `METRIC`). For **gIoU** they are not positions
   at all — what identifies a mask is `(frame, label)`, so add
   `det["frame_f"] = det["frame"].astype(float)` and pass `["frame_f", "label"]`.
   A copy, because `frame_col` is consumed separately from `coordinate_cols`.

4. **Link, on the metric you chose.** Centroid:

   ```python
   from laptrack import LapTrack
   lt = LapTrack(
       cutoff=MAX_STEP_PX ** 2,
       gap_closing_cutoff=(MAX_GAP * MAX_STEP_PX) ** 2,
       gap_closing_max_frame_count=MAX_GAP,
       splitting_cutoff=(2 * MAX_STEP_PX) ** 2 if DIVIDES else False,
   )
   track_df, split_df, merge_df = lt.predict_dataframe(
       det, coordinate_cols=["y", "x"], frame_col="frame")
   ```

   **The default metric is `sqeuclidean`, so every cutoff there is a squared
   distance.** Writing `cutoff=MAX_STEP_PX` means `MAX_STEP_PX` px², a cutoff
   √MAX_STEP_PX px wide — nothing raises, and at a 15 px prior it recovers
   **45.3%** of the true links instead of 90.1%, in 847 tracks instead of 161,
   with a mean speed **45% too low**. If squaring is a thing you would rather not
   have to remember, pass `metric="euclidean"` and give the cutoffs as plain
   distances; measured on the same movie the two forms agree to within 0.2
   points. **There are four metric fields** — `metric`, `gap_closing_metric`,
   `splitting_metric`, `merging_metric` — and they are independent, so setting
   only the first leaves the other rounds on squared distances.

   For **gIoU**, the metric is a callable over the two coordinate rows, and the
   cutoffs stop being distances:

   ```python
   lt = LapTrack(
       metric=giou_distance, cutoff=CUTOFF,
       gap_closing_metric=giou_distance, gap_closing_cutoff=CUTOFF,
       gap_closing_max_frame_count=MAX_GAP,
       splitting_metric=giou_distance,
       splitting_cutoff=CUTOFF if DIVIDES else False,
   )
   track_df, split_df, merge_df = lt.predict_dataframe(
       det, coordinate_cols=["frame_f", "label"], frame_col="frame")
   ```

   `giou_distance(u, v)` returns, for the two `(frame, label)` masks it is
   handed,

   ```
   1 - gIoU  where  gIoU = |A n B| / |A u B| - (|C| - |A u B|) / |C|
   ```

   with `C` the smallest box enclosing both masks — the second term is what
   keeps falling once the masks come apart, where `|A n B|` has already hit 0.

   **Precompute it and let the callable be a lookup**: `cdist` calls the metric
   once per candidate pair, so mask arithmetic inside it is what makes this
   slow. Per frame pair, one `np.bincount(a[both] * (b.max() + 1) + b[both])`
   gives every intersection at once, and `regionprops` bboxes broadcast into
   every `|C|` — build the whole matrix, since unlike IoU the non-overlapping
   entries are the informative ones and cannot be left out.

   The three cutoffs are one number in three places, scaled by the time each
   one spans (centroid) or left alone (gIoU, which is already scale-free).
   **`splitting_cutoff` and `merging_cutoff` are `False` by default**, which is
   the part that does not follow from the geometry: left alone, a division ends
   one track and starts two tracks belonging to nobody, and nothing says so.

5. **Map the result back by your own keys, never by position.**

   ```python
   ids = ["frame", "label", "track_id", "tree_id"]
   det = det.merge(track_df[ids], on=["frame", "label"])
   ```

   `predict_dataframe` returns the rows **sorted by frame under a fresh
   0..N-1 index**, carrying your own columns along. On a detection table that
   was already frame-sorted the order happens to survive; on one that was not,
   assigning `det["track_id"] = track_df["track_id"].values` silently scrambles
   every identity — measured at 0.2% of true links recovered, with 99.3% of the
   links it does make joining different objects. The join costs nothing and does
   not care.

   **`track_id` and `tree_id` are different answers.** `track_id` breaks at a
   division and is what you want for motion — a speed, a displacement, a
   direction. `tree_id` holds the whole lineage and is what you want for
   counting cells or following a family. On the movie below the same run gives
   161 `track_id`s and 67 `tree_id`s for 65 founder cells: count the wrong
   column and you report 2.5× the cells that were there.

6. **Validate before reporting anything** *(blocking)*. Three numbers, none of
   which needs a ground truth. Written for the centroid route; on gIoU the same
   three questions are asked of the realised distances — how many sit at the
   cutoff, what the worst accepted one is, and the track count:

   ```python
   by = det.groupby("track_id")
   dy, dx = by["y"].diff().to_numpy(), by["x"].diff().to_numpy()
   dt = by["frame"].diff().to_numpy()
   step = np.hypot(dy, dx)[np.isfinite(dt)] / dt[np.isfinite(dt)]
   print(f"{(step > 0.9 * MAX_STEP_PX).mean():.1%} of links at the cutoff, "
         f"largest {step.max() * PIXEL_UM / (INTERVAL_S / 60):.1f} um/min, "
         f"{det.track_id.nunique()} tracks for "
         f"{det.groupby('frame').size().median():.0f} objects per frame")
   ```

   - **Links piled up against the cutoff mean it is too small.** Measured on the
     movie below: 39% / 20% / 6.8% / 1.0% of links within 10% of the cutoff at
     cutoffs of 5 / 8 / 11 / 15 px, against a reported speed 30% / 12% / 5% / 3%
     below truth. Under a few percent is healthy.
   - **That check is one-sided and cannot see a cutoff that is too large** — at
     3× the right value only 0.3% of links sit near it, and the speed is 22%
     *over*. What catches that is the largest realised step in µm/min: if it
     exceeds what the user said in step 2, the linker is jumping between
     neighbours. This is why step 2 asks for a speed rather than a cutoff.
   - **Track count against objects per frame.** Comfortably more tracks than
     objects means fragmentation; fewer means identities are being merged.

   **Regime for every number quoted here**: centroid linking, on 5 seeds of a
   synthetic 24-frame movie — ~1,660 detections, 65 founder cells in 5 colonies
   with 50 divisions, 7% of detections dropped, median step 5.3 px against a
   median nearest-neighbour distance of 17 px, at 0.5 µm/px and 90 s. Where the
   step grows toward the spacing, centroid linking degrades — at 2.9× this
   movie's step it holds 73.8% of links, and no cutoff recovers the rest. That
   is the boundary the `METRIC` row is about: past it, position has stopped
   identifying the object, and the mask is what still does.

7. **Report the tracks and the settings.** `viewer.add_tracks` wants one row per
   detection as `[track_id, t, y, x]` — that column order, id first, which is
   not the order of the table you have been carrying (`inspect_object` for the
   rest of its arguments). Print the dict that reproduces the run beside it: the
   three cutoffs, `MAX_GAP`, `PIXEL_UM`, `INTERVAL_S`, and which id column each
   reported number was counted from.

## Guardrails

- **A speed read off tracks is biased low by whatever the cutoff rejected**, and
  the bias is invisible in the tracks themselves: the fast steps are the ones
  that failed to link. Report it with the fraction of detections that ended up
  linked, never alone.
- **Do not tune the cutoff until the tracks look right.** Continuity is a
  weak signal: over 11–22 px, a 2× range, link accuracy moves by 0.4 points
  while the speed the tracks report moves by 4; over 5–45 px accuracy is still
  65–90% while the speed runs from 30% under to 22% over. Tracks that look
  continuous are satisfied by a cutoff that is measurably wrong in either
  direction, which is why step 6 checks the two ends instead.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Hundreds of one- and two-frame tracks, and a mean speed far below anything the user recognises | A cutoff given as a distance while the metric is `sqeuclidean` | Square it, or set `metric="euclidean"`. Measured at a 15 px prior: 45.3% of links recovered against 90.1%, 847 tracks against 161, speed 45% low |
| Track ids look shuffled; consecutive detections of one object land in different tracks | `track_df` assigned back positionally onto a detection table that was not frame-sorted | Merge on `["frame", "label"]` (step 5). Measured: 0.2% of links recovered, 99.3% of them joining different objects |
| Tracks break wherever an object is missing from a single frame | `gap_closing_max_frame_count=1` — that is a frame *difference*, and a single missed frame is a difference of 2 | Use `MAX_GAP >= 2`. Measured: 83.4% of links against 90.1%, and 273 tracks for 65 cells |
| More tracks than there were ever cells, and the number grows through the movie | Counting `track_id`, which ends at every division, where lineage was wanted | Count `tree_id`, and enable `splitting_cutoff`. Measured: 161 / 97 / 67 lineages for 65 founders with `track_id` / splitting off / the full method |
| Speed rises whenever the cutoff is raised, and the tracks still look clean | A cutoff above what the biology allows, linking each object to a neighbour | Measured +22% at 3× the derived value, with only 0.3% of links near the cutoff — the truncation check cannot see this one; step 6's largest-step-in-µm/min can |

## Next steps

- Per-object size, shape or intensity along each track: measure them per frame
  with [[calibrated-measurements]] and join on `(frame, label)`, the same key
  step 5 uses.
- `split_df` and `merge_df` carry `parent_track_id` / `child_track_id`, which is
  the lineage tree if the user wants generation times rather than counts.
