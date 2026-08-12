---
id: pixel-classifier-segmentation
title: Segment by training a pixel classifier on scribbles
description: Train a classifier on a few hand-drawn scribbles and label every pixel of a field, with a quality number that is not the training accuracy.
tags: [segmentation, classification, annotation, quantification]
version: 1.0.0
checklist: [viewer, tensor, pkg:biopb-mcp>=0.13.0]
---

# Segment by training a pixel classifier on scribbles

**Every number below is synthetic** — a 640 x 640 three-class field, two
acquisitions of it, and about 4000 scribbled pixels. The two large classes have
the *same mean and the same standard deviation* and differ only in correlation
length, so no threshold separates them; the third is a 4 px bright rim, 2.6% of
the field. The second acquisition has its own illumination gradient, a 1.35x
exposure and a different camera offset. No real acquisition has been through
this yet — treat the numbers as the shape of the effect, not as targets.

## When to use

A field holds a few kinds of region that a person can point at but a threshold
cannot separate — texture, granularity, a rim — and the user is willing to draw
scribbles on one field. The output is a class per pixel.

## When NOT to use

- **The classes separate on brightness.** Then it is one call — an auto-threshold
  — and a forest over eleven features is a slower way to draw the same line.
- **The user needs objects, not classes.** This labels pixels; touching cells
  come back as one component. Counting, per-object measurements and identity all
  need an instance step afterwards, measured in physical units and scored
  against a hand-drawn truth with the `segmentation_qc` kernel plugin.
- **The structures are what a published model was trained on** — nuclei, whole
  cells in a standard stain. A pretrained segmenter behind the algorithm plane
  will beat scribbles and needs no training data.
- **The brightness varies because the *illumination* does.** Correct it first —
  [[flatfield]] — or the forest learns the shading as if it were a class.
- **One class is the readout and it is an intensity.** Then it is a calibrated
  measurement, and a hard class boundary throws away the signal.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `FIELDS` | `(Y, X)` each | The field the scribbles are on, plus every other field to be labelled. `guide://data` for getting them off a layer or the tensor server |
| `SCRIBBLES` | `(Y, X)` int | `0` where unlabelled, `1..K` for the classes. A napari Labels layer the user painted. What each number *means* is step 2 |
| `SIGMA_MIN`, `SIGMA_MAX` | px | **Leave them at the library defaults, 0.5 and 16.** Tying the top scale to the object size is the instinct and it is a pessimisation: `sigma_max=32`, chosen from a cell diameter of ~130 px, scored macro IoU **0.672** against **0.708** at the default on the same scribbles. Raise it only if a class is defined at a scale larger than 16 px, and re-measure when you do |
| `NORM` | — | The robust centre and spread of **each field on its own** — `np.percentile(field, [16, 50, 84])`. Never statistics carried from the scribbled field; step 3 is what that costs |
| `MEDIAN_PX` | px | Width of the label-smoothing filter, step 5. Raise it while the component count is still falling toward what the picture shows and the thin class's implied width (step 7) has not moved. Here 3, 7 and 9 px all improved every class and 9 px was still improving; the rim it might have eaten is 4 px wide, so the ceiling is a measurement, not a rule |
| `HOLDOUT` | — | Whole connected **strokes**, never a fraction of pixels. Step 6 |

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status`; `guide://kernel` covers a gap. `scikit-learn` ships with the
   server and is importable in the kernel.

2. **Confirm the inputs** *(blocking)*. Three facts, none of them in the pixels:

   - **The class list, and what each class *is*.** Not the label numbers — those
     are in the array — but where the user would put the boundary. A rim class
     especially: is the halo part of the cell or its own thing?
   - **Whether every field came off the same acquisition as the scribbled one.**
     A field that is brighter can be a brighter field or a longer exposure, and
     **the pixels do not say which**. Ask. It decides step 3, and getting it
     wrong is the largest failure here by a factor of two.
   - **Whether the user wants classes or objects.** See *When NOT to use*.

3. **Normalise each field on its own statistics, before features.** Three lines,
   and the single largest effect measured here:

   ```python
   def normalised(field):
       x = np.asarray(field, np.float32)
       lo, mid, hi = np.percentile(x, [16, 50, 84])
       return (x - mid) / max(hi - lo, 1e-6)
   ```

   On raw counts, the second acquisition scored macro IoU **0.466** where the
   same pipeline on per-field normalised input scored **0.709** — and on the
   scribbled field the two are indistinguishable (0.708 vs 0.707), so nothing in
   the run that trained it can see the difference.

   **Scaling the *features* instead does nothing at all, and that is the trap.**
   `StandardScaler` fitted on the training pixels is the textbook move, and
   against a forest it is a no-op — trees split on per-feature thresholds, so a
   monotone per-feature rescaling only relabels them. Measured as a 2x2 over
   both factors, adding the scaler moves the second field's macro IoU from
   0.7093 to 0.7100 with the normalisation underneath, and from 0.4664 to
   0.4664 without it. So it is not a weaker defence, it is not one — and a run
   that applied it has every reason to believe scale is handled. Normalise the
   **image**. A cold run that scaled features and left the image alone came out
   at macro IoU 0.325 on the second field, with the cell-interior class at an
   IoU of **0.012** and **24.5%** of the frame painted as a 4 px rim.

   You can check the premise before training and it costs nothing: compare each
   field's median and spread. 902 and 168 against 1348 and 305 is not two views
   of one sample.

4. **Features and a forest.**

   ```python
   from skimage.feature import multiscale_basic_features
   from sklearn.ensemble import RandomForestClassifier

   feat = {k: multiscale_basic_features(normalised(f)) for k, f in FIELDS.items()}
   ys, xs = np.nonzero(SCRIBBLES)
   model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
   model.fit(feat[SCRIBBLED][ys, xs], SCRIBBLES[ys, xs])
   ```

   `class_weight="balanced"` is worth trying and worth not agonising over: it
   moved macro IoU 0.708 → 0.704 on the scribbled field and 0.709 → 0.715 on the
   second — down on one, up on the other, which is noise. The thin class is not
   saved by reweighting.

5. **Then a spatial pass, because the classifier has none.** Every pixel is
   decided on its own, so the raw argmax is speckled: **644** connected bodies
   on a field holding **7 cells**, and four cold runs produced **2128 to
   12688**. A median filter on the *labels* removes it and improves every class
   at the same time — this is not a cosmetic step:

   ```python
   labels = ndi.median_filter(predicted, size=MEDIAN_PX)
   ```

   | | macro IoU | medium | interior | the 4 px rim | bodies |
   |---|---|---|---|---|---|
   | argmax | 0.708 | 0.885 | 0.694 | 0.543 | 644 |
   | median 3x3 | 0.718 | 0.893 | 0.714 | 0.548 | 154 |
   | median 7x7 | 0.735 | 0.906 | 0.746 | 0.551 | 42 |
   | median 9x9 | 0.741 | 0.911 | 0.757 | 0.554 | 28 |

   **Smooth the labels, not the probabilities.** Blurring the class
   probabilities and taking the argmax buys the same on the large classes and
   takes it out of the thin one: at `sigma=3` the interior reaches 0.760 while
   the rim falls **0.543 → 0.456**. A vote between labels cannot do that,
   because a rim pixel's neighbours along the rim are rim.

6. **Get a quality number, and know how optimistic it still is** *(visual
   check)*. Hold out whole **strokes** — refit without them, score on them:

   ```python
   from scipy import ndimage as ndi

   groups = np.zeros_like(SCRIBBLES, int)
   offset = 0
   for k in np.unique(SCRIBBLES[SCRIBBLES > 0]):          # strokes, per class
       lab, n = ndi.label(SCRIBBLES == k)
       groups[lab > 0] = lab[lab > 0] + offset
       offset += n
   ```

   Then rotate folds over `np.unique(groups[ys, xs])`. **Every cheaper split is
   worthless**, and by how much was measured against the truth on the same run:

   | scored on | macro IoU | the 4 px rim |
   |---|---|---|
   | the pixels it was fitted on | 1.000 | 1.000 |
   | a random 30% of scribble *pixels* | 0.999 | 1.000 |
   | held-out 80 px blocks | 0.908 | 0.968 |
   | held-out whole strokes | 0.887 | 0.972 |
   | **the truth** | **0.708** | **0.543** |

   A random pixel split tells you nothing a forest cannot already tell itself —
   neighbouring pixels in one stroke are one sample seen many times. And the
   stroke-wise number is *still* 0.18 high, worst on the thin class by
   **1.8x**, because a person scribbles down the middle of a region and the
   errors are at its edges. **Report it as a ceiling, not as an accuracy.**

7. **Run the three checks the output can make against itself** *(visual check)*.
   None needs a truth, all three were available to four cold runs and none ran
   any of them:

   - **Count the components** — fill holes, label, count — and compare with what
     is visible. Seven cells per field here; the cold runs reported **2128 to
     12688**. That is the check step 5 answers, and it is worth running again
     after the filter rather than assuming it worked.
   - **Divide a thin class's area by the perimeter it encloses.** That is its
     implied width, and the picture has one: 3.94 px in truth, on both fields,
     against **17.9 px** for the run that collapsed.
   - **Compare the forest's own confidence between fields.** This is the one that
     separates "the sample changed" from "my classifier broke", and it does it
     cleanly:

     ```python
     p = np.sort(model.predict_proba(f.reshape(-1, f.shape[-1])), axis=1)
     margin = (p[:, -1] - p[:, -2]).mean()      # top two, not the top one
     unsure = float((p[:, -1] < 0.6).mean())
     ```

     The gap between the top **two** classes is the statistic that moves. The
     top probability alone barely does: 0.845 → 0.840 sound against 0.845 →
     0.694 broken, where the margin halves.

     | | mean margin | unsure |
     |---|---|---|
     | broken on the second field | 0.705 → **0.484** | 0.118 → **0.276** |
     | sound on the second field | 0.705 → 0.691 | 0.118 → 0.103 |

     A sound model's confidence barely moves, and its unsure fraction goes
     *down*. A class balance that shifts while confidence holds is the sample; a
     class balance that shifts while confidence falls is the model.

8. **Validate before any number is final** *(blocking)*. Say which of step 7's
   checks you ran and what they returned, alongside the step 6 ceiling. If the
   run is being extended to more fields, do step 7 on **each** — a model that
   holds on the second field is not thereby good on the tenth.

9. **Hand off with the picture and the number.** The labels as a Labels layer
   over the field, the per-class pixel fraction per field, the stroke-holdout
   score **called a ceiling**, and the confidence comparison. Never the training
   accuracy: four cold runs on this field reported **98.1% to 99.5%** against a
   true macro IoU of **0.60 to 0.64**, and all four named the 4 px rim as their
   best class — "100% precision, 100% recall", "F1 = 1.0" — where its true IoU
   was **0.44 to 0.59**.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Accuracy is 98-99% and the labels look wrong | It was computed on the pixels the forest was fitted on. Measured: 1.000 against a true macro IoU of 0.708 | Stroke-wise holdout, step 6, reported as a ceiling |
| A cross-validated score is still ~99% | The split was over scribble *pixels*, which are not independent. Measured 0.999 against a truth of 0.708 | Split on whole strokes, never on pixels |
| The thin class scores best of all and looks worst | Holdout pixels for a rim sit in the middle of the rim, not at its edges. Measured: stroke holdout 0.972 against a truth of 0.543 | Believe the geometry check in step 7, not the score |
| The second field's class balance is nothing like the first | Either the sample changed or the classifier broke, and the balance alone does not say which. Four cold runs read it as biology; one of them was wrong by 2x | The confidence comparison in step 7 |
| One class nearly vanishes on a later field and a thin class floods it | The image was never normalised per field, so the forest learned this exposure. Measured on a cold run: interior IoU 0.012, and 24.5% of the frame labelled as a 4 px rim | Normalise each field on its own statistics, step 3 |
| Accuracy is fine on the training field and poor everywhere else | Same cause. Measured: macro IoU 0.466 on raw counts against 0.709 per-field normalised, with the *training* field identical either way (0.708 vs 0.707), so nothing in the run that trained it can see the difference | Step 3 |
| A scaler was fitted on the training pixels and the second field broke anyway | Feature scaling is a no-op against a forest: measured identical to four decimals with and without it. It is not a weak defence, it is not one | Normalise the image, not the features. Step 3 |
| The output is right in outline and speckled everywhere | Per-pixel classification has no spatial term. Measured: 644 components from the raw argmax, and 2128-12688 from four cold runs, on a field with 7 cells | A median filter on the labels, step 5 — 644 → 28 bodies at 9 px, and macro IoU 0.708 → 0.741. Count again after |
| Smoothing cleaned up the large classes and ate the thin one | The blur was applied to the class probabilities, not the labels. Measured at `sigma=3`: interior 0.694 → 0.760, the 4 px rim 0.543 → **0.456** | Median-filter the labels instead: the same rim went 0.543 → 0.554 at 9 px |
| Raising the top feature scale to the object size made it worse | `sigma_max` is a filter bank, not an object-size prior. Measured: macro IoU 0.672 at 32 against 0.708 at the default 16 | Leave the defaults; step 4 |

## Next steps

- Objects, counts and per-object measurements need an instance step on top of
  the class map, reported in physical units and scored against a hand-drawn
  truth with the `segmentation_qc` kernel plugin.
- More scribbles are worth more than more trees. The ceiling here is the
  annotation: with every true pixel of the field as training data the same
  features reach macro IoU **1.000** on it, and **0.818** on the second
  acquisition — so 0.708 from ~4000 scribbled pixels is a labelling limit, and
  the residual 0.18 on the second field is what a single field can never teach.
