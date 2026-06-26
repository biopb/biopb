"""Unit tests for the chunk-stitching core (biopb_image_base.stitch).

Container-free and GPU-free: a synthetic ``compute_chunk`` plays the role of the
model by returning, for each tile, the foreground pixels and their *exact*
destinations (= their ground-truth cell center). This isolates the cross-chunk
ID-linking logic from model noise. The key cases are cells that straddle core
borders (center on either side) and 4-core corners.

Run with: pytest tests/test_stitch.py
"""

import numpy as np
from biopb_image_base import stitch

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _square(gt, label, cy, cx, half):
    """Stamp a (2*half+1)^2 square cell labelled ``label`` centered at (cy,cx)."""
    gt[cy - half : cy + half + 1, cx - half : cx + half + 1] = label


def _centers_from_gt(gt):
    """Map each GT label to its (rounded) centroid -- the destination it flows to."""
    centers = {}
    for lab in np.unique(gt):
        if lab == 0:
            continue
        ys, xs = np.nonzero(gt == lab)
        centers[lab] = (int(round(ys.mean())), int(round(xs.mean())))
    return centers


def _make_compute_chunk(gt, centers, jitter=0):
    """Synthetic model: per tile, return foreground inds + their destinations.

    With jitter>0, a few destinations are perturbed by +-1 px to emulate noise.
    """

    def compute_chunk(tile_start, tile_stop):
        sl = tuple(slice(tile_start[i], tile_stop[i]) for i in range(2))
        sub = gt[sl]
        inds = np.nonzero(sub > 0)
        if len(inds[0]) == 0:
            return inds, None
        labs = sub[inds]
        p = np.empty((2, labs.size), dtype="float32")
        for j, lab in enumerate(labs):
            cy, cx = centers[lab]
            p[0, j] = cy - tile_start[0]
            p[1, j] = cx - tile_start[1]
        if jitter:
            # Perturb a small fraction of points by +-1 px.
            rng = np.arange(labs.size)
            mask = rng % 7 == 0
            p[0, mask] += (rng[mask] % 3) - 1
            p[1, mask] += ((rng[mask] // 3) % 3) - 1
        return inds, p

    return compute_chunk


def _run(
    gt, *, core_shape=(40, 40), margin=12, jitter=0, cluster_kwargs=None, observer=None
):
    centers = _centers_from_gt(gt)
    compute_chunk = _make_compute_chunk(gt, centers, jitter=jitter)
    out = np.zeros(gt.shape, dtype="int32")

    def write_core(core_start, core_stop, labels):
        sl = tuple(slice(core_start[i], core_stop[i]) for i in range(2))
        out[sl] = labels

    n_ids = stitch.stitch_lazy_segmentation(
        gt.shape,
        core_shape,
        margin,
        compute_chunk,
        write_core,
        cluster_kwargs=cluster_kwargs or {"min_count": 4},
        observer=observer,
    )
    return out, n_ids


def _same_partition(a, b):
    """True if label images ``a`` and ``b`` are equal up to a relabeling.

    Checks background agreement and a consistent bijection between nonzero
    labels everywhere.
    """
    if (a == 0).astype(int).sum() != (b == 0).astype(int).sum():
        return False
    if not np.array_equal(a == 0, b == 0):
        return False
    fa, fb = a[a > 0], b[a > 0]
    a2b, b2a = {}, {}
    for la, lb in zip(fa.tolist(), fb.tolist()):
        if a2b.setdefault(la, lb) != lb:
            return False
        if b2a.setdefault(lb, la) != la:
            return False
    return True


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def test_cell_inside_single_core():
    gt = np.zeros((80, 80), dtype="int32")
    _square(gt, 1, 20, 20, 4)  # well inside core (0,0)
    out, n_ids = _run(gt)
    assert n_ids == 1
    assert _same_partition(out, gt)


def test_cell_straddles_border_center_in_earlier_core():
    gt = np.zeros((80, 80), dtype="int32")
    # Core boundary at x=40. Center x=37 (earlier/left core); body 33..41.
    _square(gt, 1, 20, 37, 4)
    out, n_ids = _run(gt)
    assert n_ids == 1, "straddling cell must be one instance"
    assert _same_partition(out, gt)


def test_cell_straddles_border_center_in_later_core():
    gt = np.zeros((80, 80), dtype="int32")
    # Center x=43 (later/right core); body 39..47. This is the case naive
    # center-readback would split; injection must still merge it.
    _square(gt, 1, 20, 43, 4)
    out, n_ids = _run(gt)
    assert n_ids == 1
    assert _same_partition(out, gt)


def test_cell_at_four_core_corner():
    gt = np.zeros((80, 80), dtype="int32")
    # Corner of cores (0,0)/(0,1)/(1,0)/(1,1) at (40,40); center (40,40).
    _square(gt, 1, 40, 40, 4)
    out, n_ids = _run(gt)
    assert n_ids == 1, "corner cell must be a single instance across 4 cores"
    assert _same_partition(out, gt)


def test_noisy_destinations_one_id_per_cell():
    gt = np.zeros((80, 80), dtype="int32")
    _square(gt, 1, 20, 43, 5)  # straddling, noisy
    _square(gt, 2, 60, 20, 5)  # interior, noisy
    out, n_ids = _run(gt, jitter=1, cluster_kwargs={"min_count": 4, "match_tol": 5.0})
    assert n_ids == 2, "noise must not split or spuriously create instances"
    # Under noise we don't require a pixel-exact partition, but each cell must be
    # covered by a single dominant ID (no split), and the two cells differ.
    dominant = {}
    for lab in (1, 2):
        cell_px = (gt == lab).sum()
        vals = out[gt == lab]
        vals = vals[vals > 0]
        ids, counts = np.unique(vals, return_counts=True)
        # Exactly one ID may cover a significant fraction of the cell (no split).
        significant = ids[counts > 0.1 * cell_px]
        assert len(significant) == 1, f"cell {lab} split across IDs {significant}"
        dominant[lab] = significant[0]
    assert dominant[1] != dominant[2], "distinct cells must keep distinct IDs"


def test_id_monotonicity_and_count():
    gt = np.zeros((160, 160), dtype="int32")
    label = 0
    placed = 0
    for cy in range(20, 160, 30):
        for cx in range(20, 160, 30):
            label += 1
            _square(gt, label, cy, cx, 5)
            placed += 1
    out, n_ids = _run(gt)
    assert n_ids == placed
    # No reused / skipped IDs in the assembled output.
    assert set(np.unique(out)) - {0} == set(range(1, placed + 1))


def test_assembled_output_matches_reference_partition():
    gt = np.zeros((200, 160), dtype="int32")
    label = 0
    # Place cells on/near core borders (x=40,80,120) and interiors, spaced far
    # enough apart (>= ~15px) that distinct cells never merge.
    for cy in range(20, 200, 33):
        for cx in (20, 40, 78, 120, 140):
            label += 1
            _square(gt, label, cy, cx, 4)
    out, n_ids = _run(gt)
    assert _same_partition(out, gt)
    assert n_ids == label


def test_pruning_keeps_known_set_bounded():
    # Tall image, many rows of cells -> pruning must keep the known set small.
    gt = np.zeros((400, 80), dtype="int32")
    label = 0
    for cy in range(20, 400, 25):
        for cx in (20, 60):
            label += 1
            _square(gt, label, cy, cx, 4)
    sizes = []
    out, n_ids = _run(gt, observer=lambda s: sizes.append(s["known_size"]))
    assert _same_partition(out, gt)  # nothing pruned too early
    # The image is 400x80 (HxW); with 40x40 cores that's 10 core-rows x 2
    # core-columns (32 cells total). The known set should stay far smaller
    # than that, bounded by a frontier band of about one core-row.
    assert max(sizes) <= 12, f"known set grew too large: {max(sizes)}"


def test_prune_known_helper():
    known = [((5, 0), 1), ((30, 0), 2), ((60, 0), 3)]
    # Frontier outer-axis at 40, margin 12 -> keep_threshold 28.
    kept = stitch._prune_known(known, keep_threshold=28)
    assert [k[1] for k in kept] == [2, 3]
