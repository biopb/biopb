"""Unit tests for the instance-segmentation QC plugin (biopb_mcp.plugins.segmentation_qc).

Pins the behaviour a user would notice being wrong: counts and rates on
hand-checkable cases, IoU computed against the definition rather than against
itself, one-to-one matching that a greedy pass cannot fake, splits/merges charged
to the right side, border exclusion applied symmetrically, and empty inputs
scoring ``nan`` instead of 0.0. Also checks the delivery path -- the plugin seeds
into the kernel dir and loads via the startup-file path with a clean namespace
surface. No kernel/display needed.
"""

import numpy as np
import pytest

from biopb_mcp.plugins import segmentation_qc as qc


def _boxes(shape, boxes, start=1):
    """Label image from ``{label: (slice_y, slice_x)}``-style box specs."""
    lab = np.zeros(shape, dtype=np.int32)
    for i, (ys, xs) in enumerate(boxes, start=start):
        lab[ys[0] : ys[1], xs[0] : xs[1]] = i
    return lab


class TestIoUAndCounts:
    def test_identical_labels_score_perfectly(self):
        gt = _boxes((40, 40), [((2, 10), (2, 10)), ((20, 30), (20, 30))])
        r = qc.match_labels(gt, gt.copy())
        assert (r.tp, r.fp, r.fn) == (2, 0, 0)
        assert r.precision == r.recall == r.f1 == pytest.approx(1.0)
        assert r.mean_iou == pytest.approx(1.0)
        assert r.pq == pytest.approx(1.0)

    def test_relabelling_does_not_change_the_score(self):
        # Label *values* carry no meaning; only the partition does.
        gt = _boxes((40, 40), [((2, 10), (2, 10)), ((20, 30), (20, 30))])
        pred = gt.copy()
        pred[gt == 1], pred[gt == 2] = 77, 5
        assert qc.match_labels(gt, pred).f1 == pytest.approx(1.0)

    def test_iou_matches_the_definition(self):
        # 10x10 truth, prediction shifted 5 px: intersection 50, union 150.
        gt = _boxes((40, 40), [((10, 20), (10, 20))])
        pred = _boxes((40, 40), [((10, 20), (15, 25))])
        r = qc.match_labels(gt, pred, iou_threshold=0.3)
        assert r.mean_iou == pytest.approx(50 / 150)
        assert r.tp == 1
        # ... and the same pair is rejected by a threshold above that IoU.
        assert qc.match_labels(gt, pred, iou_threshold=0.5).tp == 0

    def test_unmatched_objects_split_into_fp_and_fn(self):
        gt = _boxes((40, 40), [((2, 10), (2, 10)), ((20, 30), (20, 30))])
        pred = _boxes((40, 40), [((2, 10), (2, 10)), ((2, 6), (30, 34))])
        r = qc.match_labels(gt, pred)
        assert (r.tp, r.fp, r.fn) == (1, 1, 1)
        assert r.precision == pytest.approx(0.5)
        assert r.recall == pytest.approx(0.5)
        assert r.f1 == pytest.approx(0.5)

    def test_pairs_report_original_label_ids(self):
        gt = _boxes((30, 30), [((5, 15), (5, 15))])
        pred = np.where(gt > 0, np.int32(42), np.int32(0))
        ((gt_id, pred_id, iou),) = qc.match_labels(gt, pred).pairs
        assert (gt_id, pred_id) == (1, 42)
        assert iou == pytest.approx(1.0)


class TestMatchingIsOneToOne:
    def test_one_prediction_cannot_claim_two_truth_objects(self):
        # A single prediction spanning both truth objects is one TP + one FN,
        # never two TPs.
        gt = _boxes((20, 40), [((2, 18), (2, 18)), ((2, 18), (22, 38))])
        pred = _boxes((20, 40), [((2, 18), (2, 38))])
        r = qc.match_labels(gt, pred, iou_threshold=0.3)
        assert r.tp <= 1
        assert r.tp + r.fn == r.n_gt == 2

    def test_optimal_assignment_finds_a_match_greedy_strands(self, monkeypatch):
        # The case the assignment step exists for. Constructed IoUs:
        #   gt1-pA 0.500   gt1-pB 0.300
        #   gt2-pA 0.235   gt2-pB 0
        # Greedy takes the heaviest pair (gt1-pA) first, which consumes pA and
        # strands gt2 -> 1 TP. Optimal maximizes the total (0.300 + 0.235 =
        # 0.535 > 0.500) and matches both -> 2 TP. Only reachable below IoU 0.5,
        # where a pair is no longer provably unique.
        gt = _boxes((10, 20), [((0, 10), (0, 10)), ((0, 10), (10, 20))])
        pred = np.zeros((10, 20), dtype=np.int32)
        pred[0:7, 0:10] = 1  # 70 px of gt1
        pred[0:4, 10:20] = 1  # + 40 px of gt2  -> pA area 110
        pred[7:10, 0:10] = 2  # 30 px of gt1    -> pB area 30

        exact = qc.match_labels(gt, pred, iou_threshold=0.2)
        assert exact.tp == 2
        assert sorted(exact.pairs) == [
            (1, 2, pytest.approx(0.3)),
            (2, 1, pytest.approx(40 / 170)),
        ]

        monkeypatch.setattr(qc, "_DENSE_CAP", 0)  # force the greedy branch
        with pytest.warns(RuntimeWarning, match="dense-assignment cap"):
            assert qc.match_labels(gt, pred, iou_threshold=0.2).tp == 1

    def test_greedy_fallback_agrees_with_assignment_above_half(self, monkeypatch):
        rng = np.random.default_rng(0)
        gt = np.zeros((80, 80), dtype=np.int32)
        pred = np.zeros((80, 80), dtype=np.int32)
        for i in range(1, 13):
            y, x = int(rng.integers(0, 68)), int(rng.integers(0, 68))
            gt[y : y + 10, x : x + 10] = i
            dy, dx = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            pred[y + dy : y + dy + 10, x + dx : x + dx + 10] = i
        exact = qc.match_labels(gt, pred, iou_threshold=0.55)
        monkeypatch.setattr(qc, "_DENSE_CAP", 0)  # force the greedy branch
        with pytest.warns(RuntimeWarning, match="dense-assignment cap"):
            greedy = qc.match_labels(gt, pred, iou_threshold=0.5)
        assert greedy.tp == exact.tp
        assert sorted(greedy.pairs) == sorted(exact.pairs)


class TestSplitsAndMerges:
    def test_split_truth_object_is_charged_as_a_split(self):
        gt = _boxes((20, 20), [((2, 18), (2, 18))])
        pred = np.zeros((20, 20), dtype=np.int32)
        pred[2:9, 2:18] = 1
        pred[10:18, 2:18] = 2
        r = qc.match_labels(gt, pred)
        assert (r.splits, r.merges) == (1, 0)

    def test_merged_predictions_are_charged_as_a_merge(self):
        gt = _boxes((20, 40), [((2, 18), (2, 18)), ((2, 18), (22, 38))])
        pred = _boxes((20, 40), [((2, 18), (2, 38))])
        r = qc.match_labels(gt, pred)
        assert (r.splits, r.merges) == (0, 1)

    def test_a_clipped_neighbour_is_not_a_merge(self):
        # A prediction overlapping a second truth object by a sliver is not a
        # merge; without the coverage floor every touching object would be one.
        gt = _boxes((20, 40), [((2, 18), (2, 18)), ((2, 18), (19, 38))])
        pred = _boxes((20, 40), [((2, 18), (2, 19))])  # 1 px into gt 2
        assert qc.match_labels(gt, pred).merges == 0


class TestBorderExclusion:
    def test_border_objects_drop_from_both_images(self):
        gt = _boxes((30, 30), [((0, 8), (0, 8)), ((12, 22), (12, 22))])
        pred = gt.copy()
        r = qc.match_labels(gt, pred, exclude_border=True)
        assert (r.n_gt, r.n_pred) == (1, 1)
        assert r.f1 == pytest.approx(1.0)
        assert r.exclude_border is True

    def test_excluding_border_does_not_invent_false_negatives(self):
        # The interior object matches; the border object leaves on both sides, so
        # it must not surface as an unmatched truth object.
        gt = _boxes((30, 30), [((0, 8), (0, 8)), ((12, 22), (12, 22))])
        r = qc.match_labels(gt, gt.copy(), exclude_border=True)
        assert (r.tp, r.fp, r.fn) == (1, 0, 0)

    def test_3d_border_faces_are_all_checked(self):
        lab = np.zeros((6, 20, 20), dtype=np.int32)
        lab[0, 2:6, 2:6] = 1  # touches the z=0 face only
        lab[2:4, 10:14, 10:14] = 2  # interior
        assert qc.match_labels(lab, lab.copy(), exclude_border=True).n_gt == 1


class TestDegenerateInput:
    def test_no_objects_anywhere_scores_nan_not_zero(self):
        empty = np.zeros((20, 20), dtype=np.int32)
        r = qc.match_labels(empty, empty.copy())
        assert (r.n_gt, r.n_pred, r.tp) == (0, 0, 0)
        for rate in (r.precision, r.recall, r.f1, r.pq, r.mean_iou):
            assert np.isnan(rate)

    def test_predictions_without_truth_score_zero_precision_nan_recall(self):
        gt = np.zeros((20, 20), dtype=np.int32)
        pred = _boxes((20, 20), [((2, 8), (2, 8))])
        r = qc.match_labels(gt, pred)
        assert r.precision == pytest.approx(0.0)
        assert np.isnan(r.recall)
        assert r.fp == 1

    def test_disjoint_segmentations_score_zero_not_nan(self):
        gt = _boxes((20, 40), [((2, 8), (2, 8))])
        pred = _boxes((20, 40), [((2, 8), (30, 36))])
        r = qc.match_labels(gt, pred)
        assert r.f1 == pytest.approx(0.0)
        assert np.isnan(r.mean_iou)  # no matches to average

    def test_boolean_mask_is_rejected_with_a_usable_message(self):
        mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="integer label image"):
            qc.match_labels(mask, mask.copy())

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            qc.match_labels(np.zeros((10, 10), np.int32), np.zeros((10, 11), np.int32))


class TestThresholdSweep:
    def test_sweep_is_monotone_and_ordered_as_given(self):
        rng = np.random.default_rng(1)
        gt = np.zeros((80, 80), dtype=np.int32)
        pred = np.zeros((80, 80), dtype=np.int32)
        for i in range(1, 10):
            y, x = int(rng.integers(0, 66)), int(rng.integers(0, 66))
            gt[y : y + 12, x : x + 12] = i
            pred[y + 1 : y + 13, x + 1 : x + 13] = i
        df = qc.f1_at_thresholds(gt, pred)
        assert list(df["iou_threshold"]) == list(qc.DEFAULT_THRESHOLDS)
        # Raising the bar can only lose matches.
        assert df["tp"].is_monotonic_decreasing

    def test_sweep_agrees_with_the_single_operating_point(self):
        gt = _boxes((40, 40), [((5, 20), (5, 20)), ((25, 35), (25, 35))])
        pred = _boxes((40, 40), [((6, 21), (6, 21)), ((25, 35), (25, 35))])
        df = qc.f1_at_thresholds(gt, pred, thresholds=(0.5, 0.8))
        for t in (0.5, 0.8):
            row = df[df["iou_threshold"] == t].iloc[0]
            one = qc.match_labels(gt, pred, iou_threshold=t)
            assert row["tp"] == one.tp
            assert row["f1"] == pytest.approx(one.f1, nan_ok=True)

    def test_result_to_dict_drops_pairs(self):
        gt = _boxes((20, 20), [((2, 10), (2, 10))])
        d = qc.match_labels(gt, gt.copy()).to_dict()
        assert "pairs" not in d
        assert d["f1"] == pytest.approx(1.0)


class TestSeeding:
    """The delivery path: the installer seeds the plugin into the kernel dir."""

    def test_seed_includes_the_qc_plugin(self, tmp_path):
        from biopb_mcp.plugins._seed import SEED_FILES, seed_kernel_plugins

        assert "segmentation_qc.py" in SEED_FILES
        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)
        assert (dest / "segmentation_qc.py").exists()

    def test_seeded_file_loads_with_a_clean_namespace_surface(self, tmp_path):
        # The production path: the loader imports the seeded file and binds it
        # under its stem, so it contributes one name — its public API is reached
        # through the module, and the reserved np handle is left intact.
        from biopb_mcp.mcp import _bootstrap
        from biopb_mcp.plugins._seed import seed_kernel_plugins

        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)
        # Other seeded plugins have their own surface tests; drop them so this
        # assertion stays an exact set for *this* file rather than a superset check.
        for other in dest.glob("*.py"):
            if other.name not in ("__init__.py", "segmentation_qc.py"):
                other.unlink()

        class IP:
            def __init__(self):
                self.user_ns = {"viewer": 1, "client": 1, "np": np, "da": 1, "ops": {}}

        ip = IP()
        _bootstrap._load_plugin_files(ip, dest)
        builtins_ = {"viewer", "client", "np", "da", "ops"}
        contributed = {
            n for n in ip.user_ns if not n.startswith("_") and n not in builtins_
        }
        assert contributed == {"segmentation_qc"}
        plug = ip.user_ns["segmentation_qc"]
        assert {"match_labels", "f1_at_thresholds", "SegQCResult"} <= set(dir(plug))
        assert ip.user_ns["np"] is np  # reserved handle untouched

    def test_seeded_plugin_is_callable_from_the_namespace(self, tmp_path):
        from biopb_mcp.mcp import _bootstrap
        from biopb_mcp.plugins._seed import seed_kernel_plugins

        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)

        class IP:
            def __init__(self):
                self.user_ns = {"viewer": 1, "client": 1, "np": np, "da": 1, "ops": {}}

        ip = IP()
        _bootstrap._load_plugin_files(ip, dest)
        gt = _boxes((20, 20), [((2, 10), (2, 10))])
        qc = ip.user_ns["segmentation_qc"]
        assert qc.match_labels(gt, gt.copy()).f1 == pytest.approx(1.0)
