import numpy as np
import pytest


def test_roi_to_mask_2d():
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    template = np.zeros([8, 8], dtype="uint8")

    roi = proto.ROI(point=proto.Point(y=3, x=4))
    mask = roi_to_mask(roi, template)

    assert np.count_nonzero(mask) == 1
    assert mask[3, 4] > 0

    roi = proto.ROI(
        rectangle=proto.Rectangle(
            top_left=proto.Point(y=1, x=1),
            bottom_right=proto.Point(y=3, x=2),
        )
    )
    mask = roi_to_mask(roi, template)
    assert np.count_nonzero(mask) == 2

    pts = [[3, 6], [0, 4], [5, 2]]
    roi = proto.ROI(
        polygon=proto.Polygon(
            points=[proto.Point(x=p[1], y=p[0]) for p in pts],
        )
    )
    mask = roi_to_mask(roi, template)
    assert np.all(
        mask
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype="uint8",
        )
    )


def test_mask_to_roi_2d():
    from biopb.image.utils import mask_to_roi, roi_to_mask

    mask = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype="uint8",
    )

    roi = mask_to_roi(mask)

    assert roi.mask.rectangle.top_left.y == 0
    assert roi.mask.rectangle.top_left.x == 2
    assert roi.mask.rectangle.bottom_right.y == 6
    assert roi.mask.rectangle.bottom_right.x == 7

    mask_new = roi_to_mask(roi, mask)

    assert np.all(mask_new == mask)


# ============================================================================
# 3D ROI Tests
# ============================================================================


def test_roi_to_mask_3d_point():
    """Test 3D ROI to mask conversion with point."""
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    template = np.zeros([4, 8, 8], dtype="uint8")  # Z, Y, X

    roi = proto.ROI(point=proto.Point(z=2, y=3, x=4))
    mask = roi_to_mask(roi, template)

    assert np.count_nonzero(mask) == 1
    assert mask[2, 3, 4] > 0


def test_roi_to_mask_3d_rectangle():
    """Test 3D ROI to mask conversion with rectangle."""
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    template = np.zeros([4, 8, 8], dtype="uint8")

    roi = proto.ROI(
        rectangle=proto.Rectangle(
            top_left=proto.Point(z=1, y=2, x=3),
            bottom_right=proto.Point(z=3, y=4, x=5),
        )
    )
    mask = roi_to_mask(roi, template)

    # Rectangle spans z=1:3, y=2:4, x=3:5
    expected_count = (3 - 1) * (4 - 2) * (5 - 3)  # 2 * 2 * 2 = 8
    assert np.count_nonzero(mask) == expected_count

    # Verify specific locations
    assert mask[1, 2, 3] > 0
    assert mask[2, 3, 4] > 0
    assert mask[0, 2, 3] == 0  # Outside z range
    assert mask[1, 5, 3] == 0  # Outside y range


def test_roi_to_mask_3d_mask_roi():
    """Test 3D ROI to mask conversion with mask type ROI."""
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    template = np.zeros([2, 4, 4], dtype="uint8")

    # Create a small binary mask and pack it
    small_mask = np.array(
        [
            [[1, 0], [0, 1]],  # z=0
            [[0, 1], [1, 0]],  # z=1
        ],
        dtype="uint8",
    )

    packed_data = np.packbits(small_mask.flatten(), bitorder="big")

    roi = proto.ROI(
        mask=proto.Mask(
            rectangle=proto.Rectangle(
                top_left=proto.Point(z=0, y=0, x=0),
                bottom_right=proto.Point(z=2, y=2, x=2),
            ),
            bin_data=proto.BinData(
                data=packed_data.tobytes(),
                endianness=0,  # big endian
            ),
        )
    )

    mask = roi_to_mask(roi, template)

    # Verify the unpacked mask matches the original pattern
    # Note: positions are relative to the rectangle
    assert mask[0, 0, 0] > 0  # z=0, y=0, x=0 -> 1
    assert mask[0, 0, 1] == 0  # z=0, y=0, x=1 -> 0
    assert mask[1, 1, 0] > 0  # z=1, y=1, x=0 -> 1


def test_mask_to_roi_3d():
    """Test 3D mask to ROI conversion."""
    from biopb.image.utils import mask_to_roi, roi_to_mask

    mask = np.zeros([4, 6, 6], dtype="uint8")
    # Create a simple pattern
    mask[1:3, 2:4, 2:4] = 1

    roi = mask_to_roi(mask)

    # Verify rectangle bounds
    assert roi.mask.rectangle.top_left.z == 1
    assert roi.mask.rectangle.top_left.y == 2
    assert roi.mask.rectangle.top_left.x == 2
    assert roi.mask.rectangle.bottom_right.z == 3
    assert roi.mask.rectangle.bottom_right.y == 4
    assert roi.mask.rectangle.bottom_right.x == 4

    # Round-trip
    mask_new = roi_to_mask(roi, mask)
    assert np.all(mask_new == mask)


def test_roi_to_mask_invalid_dimension():
    """Test that invalid mask dimension raises ValueError."""
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    # 1D array - invalid
    template_1d = np.zeros([10], dtype="uint8")
    roi = proto.ROI(point=proto.Point(x=5))

    with pytest.raises(ValueError, match="Illegal mask dimension"):
        roi_to_mask(roi, template_1d)

    # 4D array - invalid
    template_4d = np.zeros([2, 2, 2, 2], dtype="uint8")

    with pytest.raises(ValueError, match="Illegal mask dimension"):
        roi_to_mask(roi, template_4d)


def test_roi_to_mask_polygon_3d_raises():
    """Test that 3D polygon ROI raises NotImplementedError."""
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    template = np.zeros([4, 8, 8], dtype="uint8")

    pts = [[1, 3, 5], [1, 6, 2], [1, 2, 7]]  # z=1 for all
    roi = proto.ROI(
        polygon=proto.Polygon(
            points=[proto.Point(z=p[0], y=p[1], x=p[2]) for p in pts],
        )
    )

    with pytest.raises(NotImplementedError, match="3D polygon not supported"):
        roi_to_mask(roi, template)


def test_roi_to_mask_unsupported_type():
    """Test that unsupported ROI type raises NotImplementedError."""
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    template = np.zeros([8, 8], dtype="uint8")

    # Create an empty ROI (no shape set)
    roi = proto.ROI()

    with pytest.raises(NotImplementedError, match="ROI type"):
        roi_to_mask(roi, template)


def test_roi_to_mask_cv2_import_error():
    """Test handling of cv2 import error for polygon ROI."""

    # This test verifies that roi_to_mask properly handles missing cv2
    # by raising ImportError with a helpful message

    # If cv2 is already installed, we can't easily test the ImportError path
    # So we just verify the ImportError message format in the source
    # or skip if cv2 is available

    try:
        import cv2  # noqa: F401  # availability probe

        # cv2 is available, skip this test
        pytest.skip("cv2 is available, cannot test ImportError handling")
    except ImportError:
        # cv2 is not available, verify the ImportError message
        import biopb.image as proto
        import numpy as np
        from biopb.image.utils import roi_to_mask

        template = np.zeros([8, 8], dtype="uint8")

        pts = [[3, 6], [0, 4], [5, 2]]
        roi = proto.ROI(
            polygon=proto.Polygon(
                points=[proto.Point(x=p[1], y=p[0]) for p in pts],
            )
        )

        with pytest.raises(ImportError, match="cv2"):
            roi_to_mask(roi, template)


def test_roi_to_mask_polygon_2d_various_shapes():
    """Test polygon ROI with various 2D shapes."""
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask

    template = np.zeros([20, 20], dtype="uint8")

    # Triangle
    roi = proto.ROI(
        polygon=proto.Polygon(
            points=[
                proto.Point(x=10, y=5),
                proto.Point(x=5, y=15),
                proto.Point(x=15, y=15),
            ]
        )
    )
    mask = roi_to_mask(roi, template)

    # Triangle should have some non-zero pixels
    assert np.count_nonzero(mask) > 0

    # Center of triangle should be filled
    assert mask[10, 10] > 0

    # Outside triangle should be empty
    assert mask[0, 0] == 0
    assert mask[18, 18] == 0


def test_mask_to_roi_bitorder_little_endian():
    """Test mask_to_roi with little endian bitorder."""
    from biopb.image.utils import mask_to_roi, roi_to_mask

    mask = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype="uint8",
    )

    roi = mask_to_roi(mask, bitorder="little")

    # Verify round-trip with little endian
    mask_new = roi_to_mask(roi, mask)
    assert np.all(mask_new == mask)
