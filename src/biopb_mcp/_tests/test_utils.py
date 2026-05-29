"""Tests for _utils.py box operations."""

import numpy as np
import pytest

from biopb_mcp._utils import _box_intersection, _filter_boxes


class TestBoxIntersection:
    """Tests for _box_intersection function."""

    def test_no_intersection(self):
        """Non-overlapping boxes have zero intersection."""
        boxes_a = np.array([[0, 0, 1, 1]])  # box at (0,0) to (1,1)
        boxes_b = np.array([[2, 2, 3, 3]])  # box at (2,2) to (3,3)

        result = _box_intersection(boxes_a, boxes_b)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0

    def test_full_overlap(self):
        """Identical boxes have full intersection."""
        boxes_a = np.array([[0, 0, 2, 2]])
        boxes_b = np.array([[0, 0, 2, 2]])

        result = _box_intersection(boxes_a, boxes_b)
        assert result[0, 0] == 4  # 2x2 area

    def test_partial_overlap(self):
        """Partially overlapping boxes."""
        boxes_a = np.array([[0, 0, 2, 2]])  # (0,0) to (2,2)
        boxes_b = np.array([[1, 1, 3, 3]])  # (1,1) to (3,3)

        result = _box_intersection(boxes_a, boxes_b)
        assert result[0, 0] == 1  # 1x1 overlap area

    def test_multiple_boxes(self):
        """Multiple boxes pairwise intersection."""
        boxes_a = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
        boxes_b = np.array([[0, 0, 2, 2], [5, 5, 6, 6]])

        result = _box_intersection(boxes_a, boxes_b)
        assert result.shape == (2, 2)
        # box_a[0] vs box_b[0]: overlap 4
        assert result[0, 0] == 4
        # box_a[0] vs box_b[1]: no overlap
        assert result[0, 1] == 0
        # box_a[1] vs box_b[0]: overlap 1
        assert result[1, 0] == 1
        # box_a[1] vs box_b[1]: no overlap
        assert result[1, 1] == 0

    def test_3d_boxes(self):
        """3D boxes with 6 coordinates."""
        boxes_a = np.array([[0, 0, 0, 2, 2, 2]])  # z,y,x min; z,y,x max
        boxes_b = np.array([[1, 1, 1, 3, 3, 3]])

        result = _box_intersection(boxes_a, boxes_b)
        assert result[0, 0] == 1  # 1x1x1 overlap volume

    def test_touching_boxes(self):
        """Boxes that touch at edge have zero intersection."""
        boxes_a = np.array([[0, 0, 1, 1]])
        boxes_b = np.array([[1, 0, 2, 1]])  # touches at x=1

        result = _box_intersection(boxes_a, boxes_b)
        assert result[0, 0] == 0

    def test_invalid_shape_raises(self):
        """Boxes with non-2d shape raise assertion."""
        boxes_a = np.array([[0, 0, 1]])  # 3 coords, not divisible by 2
        boxes_b = np.array([[0, 0, 1, 1]])

        with pytest.raises(AssertionError):
            _box_intersection(boxes_a, boxes_b)


class TestFilterBoxes:
    """Tests for _filter_boxes function."""

    def test_single_box_kept(self):
        """Single box is always kept."""
        boxes = np.array([[0, 0, 10, 10]])
        result = _filter_boxes(boxes)
        assert len(result) == 1
        assert result[0] == True

    def test_non_overlapping_kept(self):
        """Non-overlapping boxes are all kept."""
        boxes = np.array([[0, 0, 1, 1], [5, 5, 6, 6], [10, 10, 11, 11]])
        result = _filter_boxes(boxes)
        assert np.all(result)

    def test_enclosed_box_removed(self):
        """Box mostly enclosed by another is removed."""
        # Large box enclosing a small one
        boxes = np.array(
            [
                [0, 0, 10, 10],  # large box
                [2, 2, 3, 3],  # small box inside (intersection ratio = 1.0)
            ]
        )
        result = _filter_boxes(boxes, threshold=0.75)
        # Small box (last) should be removed since it's fully enclosed
        assert result[1] == False
        assert result[0] == True

    def test_threshold_adjustment(self):
        """Different threshold values affect filtering."""
        boxes = np.array(
            [
                [0, 0, 4, 4],  # 16 area
                [1, 1, 3, 3],  # 4 area, intersection = 4, ratio = 4/4 = 1.0
            ]
        )

        # With threshold 0.5, small box should be removed
        result_low = _filter_boxes(boxes, threshold=0.5)
        assert not np.all(result_low)

        # With threshold 1.0 (strict), both kept since ratio < 1.0
        result_high = _filter_boxes(boxes, threshold=1.0)
        assert np.all(result_high)

    def test_partial_overlap_kept(self):
        """Partially overlapping boxes may be kept depending on threshold."""
        boxes = np.array(
            [
                [0, 0, 2, 2],  # 4 area
                [1, 1, 3, 3],  # 4 area, intersection = 1, ratio = 1/4 = 0.25
            ]
        )

        result = _filter_boxes(boxes, threshold=0.75)
        assert np.all(result)  # 0.25 < 0.75, both kept

    def test_empty_boxes(self):
        """Empty array returns empty result."""
        boxes = np.array([]).reshape(0, 4)
        result = _filter_boxes(boxes)
        assert len(result) == 0
