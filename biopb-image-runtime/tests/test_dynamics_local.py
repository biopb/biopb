"""Unit tests for the GPU-free dynamics fork (biopb_image_base.dynamics_local).

Fast, container-free: exercises flow integration, destination clustering, the
injected-seed ID-inheritance mechanism, and small-instance removal on synthetic
fields. Run with: pytest tests/test_dynamics_local.py
"""

import numpy as np
from biopb_image_base import dynamics_local


def test_follow_flows_converges_to_center():
    """Pixels under a flow pointing at a center end up near that center."""
    H = W = 30
    cy, cx = 15, 15
    yy, xx = np.mgrid[0:H, 0:W]
    # Unit flow toward the center on each axis.
    dy = np.sign(cy - yy).astype("float32")
    dx = np.sign(cx - xx).astype("float32")
    dP = np.stack([dy, dx])

    inds = np.nonzero(np.ones((H, W), bool))
    p = dynamics_local.follow_flows(dP, inds, niter=200)

    # Every pixel should converge to within a couple pixels of the center.
    assert np.abs(p[0] - cy).max() <= 2
    assert np.abs(p[1] - cx).max() <= 2


def test_cluster_two_distinct_destinations():
    """Two well-separated destination clusters -> two labels."""
    shape0 = (40, 40)
    # 20 pixels flowing to (10,10), 20 flowing to (30,30).
    p = np.array([[10] * 20 + [30] * 20, [10] * 20 + [30] * 20], dtype="float32")
    # foreground indices: arbitrary distinct pixels (identity not used by cluster
    # beyond placement), put them at the destinations for simplicity.
    inds = (p[0].astype(int), p[1].astype(int))

    labels, info = dynamics_local.cluster_destinations(p, inds, shape0, min_count=4)

    assert len(info) == 2
    # Two non-background labels present.
    assert len(set(np.unique(labels)) - {0}) == 2
    # No injected seeds -> both are "new".
    assert all(meta["inherited"] is None for meta in info.values())


def test_injected_seed_inherits_id():
    """A cluster landing on an injected seed reports the injected id."""
    shape0 = (40, 40)
    p = np.array([[20] * 25, [20] * 25], dtype="float32")
    inds = (p[0].astype(int), p[1].astype(int))

    labels, info = dynamics_local.cluster_destinations(
        p, inds, shape0, injected_seeds=[((20, 20), 777)], min_count=4
    )

    assert len(info) == 1
    (meta,) = info.values()
    assert meta["inherited"] == 777


def test_noisy_destinations_single_cluster():
    """Destinations spread by inference noise still resolve to one cell."""
    shape0 = (40, 40)
    cy, cx = 20, 20
    # Most pixels at the center, a few jittered by 1px.
    ys = [cy] * 30 + [cy + 1, cy - 1, cy, cy]
    xs = [cx] * 30 + [cx, cx, cx + 1, cx - 1]
    p = np.array([ys, xs], dtype="float32")
    inds = (p[0].astype(int), p[1].astype(int))

    labels, info = dynamics_local.cluster_destinations(p, inds, shape0, min_count=4)

    assert len(info) == 1
    assert len(set(np.unique(labels)) - {0}) == 1


def test_remove_small_instances():
    labels = np.zeros((10, 10), dtype="int32")
    labels[0:1, 0:3] = 1  # 3 px
    labels[5:9, 5:9] = 2  # 16 px
    out = dynamics_local.remove_small_instances(labels, min_area=10)
    assert 1 not in set(np.unique(out))
    assert 2 in set(np.unique(out))


def test_empty_input_returns_background():
    labels, info = dynamics_local.cluster_destinations(None, (np.array([]),), (20, 20))
    assert labels.max() == 0
    assert info == {}
