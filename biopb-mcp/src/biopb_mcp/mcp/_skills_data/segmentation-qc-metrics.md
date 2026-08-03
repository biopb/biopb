---
id: segmentation-qc-metrics
title: Score an instance segmentation against ground truth
description: Compare a segmentation to ground truth and report F1 at matched IoU, plus how many objects were split or merged.
tags: [segmentation, qc, evaluation]
version: 1.0.0
checklist: [viewer, tensor, plugin:segmentation_qc, pkg:biopb-mcp>=0.13.0]
---

# Score an instance segmentation against ground truth

## When to use

Two Labels layers of the same field need to be compared: a prediction against a
hand-annotated truth, or two segmentation runs against each other — choosing
between models (`cellpose`, `lacss`, a threshold pipeline), tuning one model's
parameters, or answering "is this segmentation good enough to measure from".

The metric that matters for instances is **F1 at a matched IoU threshold**, with
splits and merges reported separately. Instance identity is the thing being
scored, and it is invisible to any pixel-level metric.

## When NOT to use

- **Semantic masks.** For a foreground/background mask with no object identity,
  pixel IoU or Dice is the correct metric and this is the wrong tool.
- **No ground truth.** With no annotation there is nothing to match against.
  Judge plausibility instead (count, size distribution, border fraction) and say
  plainly that it is not a measured accuracy.
- **A pixel-accuracy question.** "What fraction of pixels are right" is a
  different question; do not answer it with F1, and do not answer an instance
  question with Dice.
- **Grossly misaligned layers.** If prediction and truth are offset or on
  different grids, every IoU is near zero and the metrics are meaningless. Fix
  the alignment first — a global-shift report is not a segmentation result.

## Parameters

| Name | Unit | How to derive it |
|---|---|---|
| `GT` | int image | The **annotated** labels. Which one that is, is not derivable from the data — ask (step 2) |
| `PRED` | int image | The labels being scored |
| `IOU_THRESHOLD` | 0–1 | `0.5` is the standard operating point and the one to quote. Below 0.5 one-to-one matching is no longer guaranteed unique |
| `THRESHOLDS` | 0–1 each | `(0.5, 0.6, 0.7, 0.8, 0.9)` for the sweep. F1 falling off steeply across it means boundaries are poor even when detection is good |
| `EXCLUDE_BORDER` | bool | Objects clipped by the field edge are partial in *both* layers. Exclude them, or include them in both — never one side only |

`GT` and `PRED` are each an array from `client`, a layer on `viewer`, or a
temporary name in the kernel. Read `guide://data` before pulling pixels off data
sources.

Why the threshold sweep and not one number: F1@0.5 answers "were the objects
found", F1@0.8 answers "are the boundaries right". A model can score 0.95 and
0.35 on the same field, and only the pair tells you whether to trust a
measurement made from it.

## Steps

1. **Check the requirements** *(blocking)*. Resolve `checklist:` against
   `server_status` — `guide://kernel` covers what to do about a gap.

   `GT` and `PRED` come from either `viewer` or `client`. We need at least one of
   the two. `plugin:segmentation_qc` is a kernel plugin, so `## Kernel plugins`
   is what answers for it, and there is no degraded path. It is bound in the
   namespace as the module `segmentation_qc`; its functions are called through
   it.

   Do not reimplement the plugin:segmentation_qc unless directly requested by
   the user: greedy nearest-IoU matching disagrees with optimal assignment on
   exactly the crowded fields where the score matters.

2. **Confirm which one is ground truth** *(confirm-input, blocking)*. Name both
   back to the user and state which you will treat as `GT`. This is
   not derivable from the data, and getting it backwards swaps precision with
   recall and **inverts the split/merge diagnosis** — the one output that says
   what to fix. Also confirm the border policy here.

3. **Score at the operating point.** Read the signature once first — the
   docstring is the documentation, and the parameters are not guessable:

   ```python
   inspect_object("segmentation_qc")  # every callable, its signature, what each field means
   ```

   Resolve `GT` and `PRED` the way `guide://data` describes. Both must come from
   the **same pyramid level**: matched against a resampled version of itself, a
   perfect segmentation scores like a bad one, and nothing in the numbers says
   which happened.

   ```python
   m = segmentation_qc.match_labels(GT, PRED, iou_threshold=IOU_THRESHOLD,
                                    exclude_border=EXCLUDE_BORDER)
   print(f"F1={m.f1:.3f}  precision={m.precision:.3f}  recall={m.recall:.3f}")
   print(f"TP={m.tp} FP={m.fp} FN={m.fn}  mean IoU over matches={m.mean_iou:.3f}")
   print(f"splits={m.splits}  merges={m.merges}")
   ```

   Read the three numbers as a diagnosis, and say which it is:

   | Pattern | Reading |
   |---|---|
   | High recall, low precision | Over-segmenting — spurious objects, or one truth object broken up |
   | High precision, low recall | Missing objects — threshold too strict, or dim objects lost |
   | Both high, `mean_iou` near the threshold | Objects found, boundaries loose |
   | `splits` high | One truth object → several predictions: seeds too aggressive |
   | `merges` high | Several truth objects → one prediction: touching objects not separated |

4. **Sweep the threshold.**

   ```python
   sweep = segmentation_qc.f1_at_thresholds(GT, PRED, thresholds=THRESHOLDS,
                                            exclude_border=EXCLUDE_BORDER)
   print(sweep.to_string(index=False))
   ```

5. **Visual check** *(non-blocking)*. Put the disagreement on screen, not just
   the score — the numbers say how much is wrong, the overlay says what kind.
   Layer truth and prediction together with `blending="additive"`, screenshot one
   stated slice or crop, and report F1@0.5, F1@0.8, and the split/merge counts
   beside it. Never the screenshot alone: two label layers look similar at a
   glance at almost any F1.

6. **Validate-and-gate** *(blocking)* before scaling out. Scoring one field is
   cheap; scoring a whole catalog, or re-running a GPU segmentation with new
   parameters, is not. Show the single-field result, name the parameter change you
   would make from it, and get agreement before spending the compute.

7. **Report the numbers with the settings that produced them**, and **do not
   invent quality bands.** F1 > 0.9 is not "production-ready", and a 5% gap
   between two models is not "decisive" — where the bar sits depends on the
   object type, the annotation quality, and what the measurement downstream
   needs, none of which are in the score. Give the numbers, say what they
   diagnose, and let the user set the threshold. A bare F1 is not comparable
   across runs either:

   ```python
   # layer names, or array_ids when the run came off the tensor server
   print({"gt": GT_REF, "pred": PRED_REF, "iou_threshold": IOU_THRESHOLD,
          "exclude_border": EXCLUDE_BORDER, "n_gt": m.n_gt, "n_pred": m.n_pred,
          "f1": round(m.f1, 4), "splits": m.splits, "merges": m.merges})
   ```

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| A verdict like "excellent, production-ready", or one model winning by "a decisive margin" | Quality bands invented to bridge the gap between a number and a decision | Quote F1@0.5, F1@0.8 and the split/merge counts; *good enough* is the user's call, not the scorer's (step 7). Expect the pull: 3 of 3 models asked this cold invented a band |
| Precision and recall swapped versus expectation, while F1 looks right | `GT` and `PRED` passed in the wrong order. F1 is symmetric under the swap, so the headline number does not move and the split/merge diagnosis inverts silently | Step 2 exists to prevent this; re-run with them swapped and see if it resolves. Measured: precision and recall trade 0.80 ↔ 0.94 while F1 stays exact |
| Metrics undefined / `nan` | One layer has no objects after border exclusion | Report "no objects to match" — not a score of 0, which reads as a bad model rather than an empty field. This is what the plugin returns by design, so the `nan` is the answer rather than a bug |

## Next steps

- To report object sizes in physical units alongside the score, use
  [[calibrated-measurements]] — the QC numbers say whether the labels are
  trustworthy, the calibrated table is what gets quoted.
