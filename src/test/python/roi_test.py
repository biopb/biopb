import pytest

def test_roi_to_mask_2d():
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask
    import numpy as np
    template = np.zeros([8,8], dtype='uint8')

    roi = proto.ROI(
        point=proto.Point(y=3, x=4)
    )
    mask = roi_to_mask(roi, template)

    assert np.count_nonzero(mask) == 1
    assert mask[3, 4] > 0


    roi = proto.ROI( rectangle = proto.Rectangle(
        top_left=proto.Point(y=1, x=1),
        bottom_right=proto.Point(y=3, x=2),
    ))
    mask = roi_to_mask(roi, template)
    assert np.count_nonzero(mask) == 2

    pts = [[3, 6], [0, 4], [5, 2]]
    roi = proto.ROI( polygon = proto.Polygon(
        points = [proto.Point(x=p[1], y=p[0]) for p in pts],
    ))
    mask = roi_to_mask(roi, template)
    assert np.all(mask == np.array([
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0.],
        [0., 0., 0., 1., 1., 1., 0., 0.],
        [0., 0., 0., 1., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ], dtype='uint8'))


def test_mask_to_roi_2d():
    import biopb.image as proto
    from biopb.image.utils import roi_to_mask, mask_to_roi
    import numpy as np

    mask = np.array([
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 0., 0.],
        [0., 0., 0., 1., 1., 1., 0., 0.],
        [0., 0., 0., 1., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ], dtype='uint8')
    
    roi = mask_to_roi(mask)

    assert roi.mask.rectangle.top_left.y == 0
    assert roi.mask.rectangle.top_left.x == 2
    assert roi.mask.rectangle.bottom_right.y == 6
    assert roi.mask.rectangle.bottom_right.x == 7

    mask_new = roi_to_mask(roi, mask)

    assert np.all(mask_new == mask)

