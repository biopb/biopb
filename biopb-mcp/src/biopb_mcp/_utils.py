import numpy as np


def _box_intersection(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes_a: [..., N, 2d]
      boxes_b: [..., M, 2d]

    Returns:
      float array with shape [..., N, M] representing pairwise intersections.
    """
    from numpy import maximum, minimum

    ndim = boxes_a.shape[-1] // 2
    assert ndim * 2 == boxes_a.shape[-1]
    assert ndim * 2 == boxes_b.shape[-1]

    min_vals_1 = boxes_a[..., None, :ndim]  # [..., N, 1, d]
    max_vals_1 = boxes_a[..., None, ndim:]
    min_vals_2 = boxes_b[..., None, :, :ndim]  # [..., 1, M, d]
    max_vals_2 = boxes_b[..., None, :, ndim:]

    min_max = minimum(max_vals_1, max_vals_2)  # [..., N, M, d]
    max_min = maximum(min_vals_1, min_vals_2)

    intersects = maximum(0, min_max - max_min)  # [..., N, M, d]

    return intersects.prod(axis=-1)


def _filter_boxes(boxes: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    """Filter boxes based on overlap. Remove boxes mostly enclosed by another.

    Args:
      boxes: [N, 4/6]
      threshold: overlap threshold

    Returns:
      boolean array with shape [N]
    """
    areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    intersections = _box_intersection(boxes, boxes)  # [..., N, N]

    its_area_ratio = intersections / (areas[..., None] + 1e-6)
    np.fill_diagonal(its_area_ratio, 0)

    # scan from the lowest score
    bm = np.ones([its_area_ratio.shape[-1]], dtype=bool)
    for i in range(its_area_ratio.shape[-1] - 1, -1, -1):
        if np.any(its_area_ratio[i] > threshold):
            bm[i] = False
            its_area_ratio[..., i] = 0

    return bm
