"""Instance-segmentation QC: IoU matching, F1 at threshold, splits and merges.

Scores a predicted label image against a ground-truth one the way the
cell-segmentation literature does, which is **not** how a pixel metric does it.
Dice and pixel IoU answer "which pixels are foreground"; an instance metric has to
first decide *which predicted object is which truth object*, and only then count.
Two segmentations with identical pixel Dice can differ completely in F1 once a
merge across touching nuclei is charged as one false positive plus two false
negatives.

Delivered as a kernel plugin rather than as a snippet in the skill body because
the matching is the part that is easy to get subtly wrong and cheap to unit-test:

- **Objects are matched one-to-one by maximum total IoU** (``linear_sum_assignment``),
  not greedily nearest. Greedy and optimal agree above IoU 0.5 -- where a match is
  provably unique -- and disagree exactly on the crowded fields where the score
  matters. Above ``_DENSE_CAP`` candidate cells the dense assignment is skipped for
  a descending-IoU greedy pass (exact for a threshold above 0.5) with a warning.
- **IoU is computed once and swept**, so ``f1_at_thresholds`` costs one pass over
  the pixels, not one per threshold. Overlaps are accumulated sparsely (only
  label pairs that actually touch), so a 5000-object mosaic never materializes a
  5000x5000 matrix unless the assignment step needs it.
- **Splits and merges are reported separately from F1.** F1 says how much is
  wrong; the split/merge counts say *what kind* is wrong, which is the part that
  tells you which knob to turn. They are counted by coverage of the truth object,
  so a neighbour clipped by a few pixels is not called a merge.
- **Empty inputs give ``nan``, not zero.** A field with no objects after border
  exclusion has an undefined score; reporting 0.0 reads as a failed model rather
  than an empty field.

``exclude_border`` drops objects touching any face of the array, and always from
**both** images -- a truth object clipped by the field edge is partial in the
prediction too, and excluding one side only is the most common way these numbers
end up not comparable to a published figure.

Three public callables, reached through the module the agent gets bound
(``segmentation_qc``): ``match_labels`` (one operating point),
``f1_at_thresholds`` (the sweep, as a DataFrame), and the ``SegQCResult`` record
they return.
"""

# Private aliases keep the module's own surface to its public API, so
# `inspect_object("segmentation_qc")` shows the agent the three callables rather
# than every pandas/scipy handle this file imported. Style, not protection: as a
# kernel plugin this module is bound under one name (#664).
import warnings as _warnings
from dataclasses import dataclass as _dataclass, field as _dc_field

import numpy as np
import pandas as _pd
from scipy.optimize import linear_sum_assignment as _lsa

__all__ = ["match_labels", "f1_at_thresholds", "SegQCResult", "DEFAULT_THRESHOLDS"]

# The operating points worth quoting together: 0.5 answers "were the objects
# found", 0.9 answers "are the boundaries right". A model can score well on one
# and badly on the other, and only the pair says whether to measure from it.
DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

# Above this many (n_gt * n_pred) cells, skip the dense assignment matrix
# (~32 MB at 2000x2000 float64) and use the greedy pass instead.
_DENSE_CAP = 4_000_000


@_dataclass
class SegQCResult:
    """One operating point's scores. ``pairs`` holds ``(gt_id, pred_id, iou)``."""

    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    pq: float
    mean_iou: float
    splits: int
    merges: int
    iou_threshold: float
    exclude_border: bool
    pairs: list = _dc_field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        """Flat dict of the scalar fields, for logging or a results table."""
        return {k: v for k, v in self.__dict__.items() if k != "pairs"}


def _as_labels(lab, name):
    """Validate an integer label array (0 = background)."""
    lab = np.asarray(lab)
    if lab.size == 0:
        raise ValueError(f"{name} is empty")
    if lab.dtype == bool or not np.issubdtype(lab.dtype, np.integer):
        raise ValueError(
            f"{name} must be an integer label image (0 = background), got "
            f"{lab.dtype}. A boolean mask has no object identity — label it first."
        )
    if lab.min() < 0:
        raise ValueError(f"{name} has negative labels")
    return lab


def _border_ids(lab):
    """Labels touching any face of the array."""
    ids = set()
    for axis in range(lab.ndim):
        for index in (0, -1):
            ids.update(np.unique(np.take(lab, index, axis=axis)).tolist())
    ids.discard(0)
    return ids


def _drop_ids(lab, ids):
    if not ids:
        return lab
    out = lab.copy()
    out[np.isin(out, np.fromiter(ids, dtype=lab.dtype, count=len(ids)))] = 0
    return out


def _index_of(lab):
    """Compact ``0..n-1`` indices for the nonzero labels, and a lookup table."""
    ids = np.unique(lab)
    ids = ids[ids != 0]
    size = int(ids.max()) + 1 if ids.size else 1
    lut = np.zeros(size, dtype=np.int64)
    lut[ids] = np.arange(ids.size, dtype=np.int64)
    return ids, lut


@_dataclass
class _Overlaps:
    """Sparse per-pair overlap: only label pairs that actually intersect."""

    gt_ids: np.ndarray
    pred_ids: np.ndarray
    gi: np.ndarray  # gt index per intersecting pair
    pi: np.ndarray  # pred index per intersecting pair
    inter: np.ndarray  # intersection area per pair
    iou: np.ndarray  # IoU per pair
    gt_area: np.ndarray
    pred_area: np.ndarray


def _overlaps(gt, pred):
    """Sparse IoU over every intersecting (gt, pred) label pair — one pixel pass."""
    gt_ids, gt_lut = _index_of(gt)
    pred_ids, pred_lut = _index_of(pred)
    n_gt, n_pred = gt_ids.size, pred_ids.size

    gt_area = np.bincount(gt_lut[gt[gt > 0]], minlength=n_gt).astype(np.int64)
    pred_area = np.bincount(pred_lut[pred[pred > 0]], minlength=n_pred).astype(np.int64)

    empty_i = np.zeros(0, dtype=np.int64)
    if n_gt == 0 or n_pred == 0:
        return _Overlaps(
            gt_ids, pred_ids, empty_i, empty_i, empty_i, np.zeros(0), gt_area, pred_area
        )

    both = (gt > 0) & (pred > 0)
    if not both.any():
        return _Overlaps(
            gt_ids, pred_ids, empty_i, empty_i, empty_i, np.zeros(0), gt_area, pred_area
        )

    # Pair each overlapping pixel's (gt, pred) into one integer, then count
    # occurrences: O(overlapping pixels), and memory in *touching pairs* rather
    # than in n_gt * n_pred.
    codes = gt_lut[gt[both]].astype(np.int64) * n_pred + pred_lut[pred[both]]
    uniq, inter = np.unique(codes, return_counts=True)
    gi, pi = np.divmod(uniq, n_pred)
    inter = inter.astype(np.int64)
    union = gt_area[gi] + pred_area[pi] - inter
    return _Overlaps(gt_ids, pred_ids, gi, pi, inter, inter / union, gt_area, pred_area)


def _match(ov, threshold):
    """One-to-one matches with IoU >= *threshold*, as (gi, pi, iou) arrays."""
    n_gt, n_pred = ov.gt_ids.size, ov.pred_ids.size
    keep = ov.iou >= threshold
    gi, pi, iou = ov.gi[keep], ov.pi[keep], ov.iou[keep]
    if gi.size == 0:
        return gi, pi, iou

    # Above IoU 0.5 a pair is already a unique match (a prediction cannot cover
    # more than half of two disjoint truth objects), so the filtered set *is* the
    # matching and the assignment step would only confirm it.
    if threshold > 0.5:
        return gi, pi, iou

    if n_gt * n_pred <= _DENSE_CAP:
        cost = np.zeros((n_gt, n_pred), dtype=np.float64)
        cost[gi, pi] = iou
        rows, cols = _lsa(cost, maximize=True)
        sel = cost[rows, cols] >= threshold
        return rows[sel], cols[sel], cost[rows[sel], cols[sel]]

    _warnings.warn(
        f"{n_gt}x{n_pred} objects exceeds the dense-assignment cap "
        f"({_DENSE_CAP}); matching greedily by descending IoU. Exact above "
        "IoU 0.5, approximate at this threshold.",
        RuntimeWarning,
        stacklevel=3,
    )
    order = np.argsort(-iou)
    used_g, used_p, take = set(), set(), []
    for k in order:
        g, p = int(gi[k]), int(pi[k])
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        take.append(k)
    take = np.asarray(take, dtype=np.int64)
    return gi[take], pi[take], iou[take]


def _splits_merges(ov, fragment_fraction):
    """Truth objects broken into pieces, and predictions that fuse several.

    Counted by *coverage of the truth object* (``intersection / gt_area``), so a
    neighbour clipped by a few pixels is not charged as a merge.
    """
    if ov.gi.size == 0:
        return 0, 0
    substantial = (ov.inter / ov.gt_area[ov.gi]) >= fragment_fraction
    gi, pi = ov.gi[substantial], ov.pi[substantial]
    if gi.size == 0:
        return 0, 0
    per_gt = np.bincount(gi, minlength=ov.gt_ids.size)
    per_pred = np.bincount(pi, minlength=ov.pred_ids.size)
    return int((per_gt >= 2).sum()), int((per_pred >= 2).sum())


def _score(ov, threshold, exclude_border, fragment_fraction, with_pairs):
    n_gt, n_pred = int(ov.gt_ids.size), int(ov.pred_ids.size)
    gi, pi, iou = _match(ov, threshold)
    tp = int(gi.size)
    fp, fn = n_pred - tp, n_gt - tp

    nan = float("nan")
    denom = n_gt + n_pred
    pq_denom = tp + 0.5 * fp + 0.5 * fn
    splits, merges = _splits_merges(ov, fragment_fraction)

    return SegQCResult(
        n_gt=n_gt,
        n_pred=n_pred,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=(tp / n_pred) if n_pred else nan,
        recall=(tp / n_gt) if n_gt else nan,
        f1=(2 * tp / denom) if denom else nan,
        pq=(float(iou.sum()) / pq_denom) if pq_denom else nan,
        mean_iou=float(iou.mean()) if tp else nan,
        splits=splits,
        merges=merges,
        iou_threshold=float(threshold),
        exclude_border=bool(exclude_border),
        pairs=(
            [
                (int(ov.gt_ids[g]), int(ov.pred_ids[p]), float(v))
                for g, p, v in zip(gi, pi, iou, strict=True)
            ]
            if with_pairs
            else []
        ),
    )


def _prepare(gt, pred, exclude_border):
    gt = _as_labels(gt, "gt")
    pred = _as_labels(pred, "pred")
    if gt.shape != pred.shape:
        raise ValueError(
            f"gt and pred must have the same shape, got {gt.shape} and "
            f"{pred.shape}. Different grids score near zero for the wrong reason."
        )
    if exclude_border:
        gt = _drop_ids(gt, _border_ids(gt))
        pred = _drop_ids(pred, _border_ids(pred))
    return gt, pred


def match_labels(
    gt,
    pred,
    iou_threshold: float = 0.5,
    exclude_border: bool = False,
    fragment_fraction: float = 0.1,
) -> SegQCResult:
    """Score an instance segmentation against ground truth at one IoU threshold.

    Objects are matched one-to-one by maximum total IoU, then counted: a match at
    or above *iou_threshold* is a true positive, an unmatched prediction a false
    positive, an unmatched truth object a false negative.

    Args:
        gt: Ground-truth label image (integer, 0 = background).
        pred: Predicted label image, same shape as *gt*.
        iou_threshold: Match acceptance threshold. 0.5 is the standard operating
            point and the one to quote.
        exclude_border: Drop objects touching any face of the array, from **both**
            images. Use it when truth objects are clipped by the field edge.
        fragment_fraction: Minimum coverage of a truth object for an overlap to
            count toward the split/merge tallies.

    Returns:
        :class:`SegQCResult`. Read ``precision``/``recall`` as a diagnosis (low
        precision = over-segmenting, low recall = missing objects) and
        ``splits``/``merges`` as which of the two it is. Rates are ``nan``, not
        0.0, when their denominator is empty.
    """
    gt, pred = _prepare(gt, pred, exclude_border)
    ov = _overlaps(gt, pred)
    return _score(ov, iou_threshold, exclude_border, fragment_fraction, True)


def f1_at_thresholds(
    gt,
    pred,
    thresholds=DEFAULT_THRESHOLDS,
    exclude_border: bool = False,
    fragment_fraction: float = 0.1,
):
    """Sweep :func:`match_labels` over IoU thresholds, as a DataFrame.

    F1 holding up across the sweep means the boundaries are good, not just the
    detections; a steep fall from 0.5 to 0.8 means objects were found but their
    outlines are loose — which is what decides whether measurements taken from
    them are usable. IoU is computed once and reused, so this costs one pass over
    the pixels regardless of how many thresholds are given.

    Args:
        gt: Ground-truth label image (integer, 0 = background).
        pred: Predicted label image, same shape as *gt*.
        thresholds: IoU thresholds to score at.
        exclude_border: As in :func:`match_labels`.
        fragment_fraction: As in :func:`match_labels`.

    Returns:
        ``pandas.DataFrame``, one row per threshold, ordered as given.
    """
    gt, pred = _prepare(gt, pred, exclude_border)
    ov = _overlaps(gt, pred)
    rows = [
        _score(ov, t, exclude_border, fragment_fraction, False).to_dict()
        for t in thresholds
    ]
    columns = [
        "iou_threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "pq",
        "mean_iou",
    ]
    return _pd.DataFrame(rows, columns=columns)
