from typing import Tuple

import cv2
import numpy as np

from .._utils import _filter_boxes


def _render_meshes(
    response, label: np.ndarray, post_process: bool = False
) -> np.ndarray:
    """Render 3D mesh detections into a label array.

    Args:
        response: DetectionResponse protobuf with mesh detections
        label: 3D numpy array to render into
        post_process: whether to filter overlapping boxes

    Returns:
        label array with rendered meshes
    """
    try:
        from vedo import Mesh
    except ImportError:
        raise ImportError(
            "vedo is required for 3D mesh rendering. Please install with `pip install biopb-mcp[3d]`"
        ) from None

    if post_process:
        bboxes = []
        for det in response.detections:
            x = [v.x for v in det.roi.mesh.verts]
            y = [v.y for v in det.roi.mesh.verts]
            z = [v.z for v in det.roi.mesh.verts]
            bboxes.append([min(z), min(y), min(x), max(z), max(y), max(x)])

        bm = _filter_boxes(np.array(bboxes))

    else:
        bm = [True] * len(response.detections)

    meshes = []
    for det, selected in zip(response.detections, bm, strict=False):
        if selected:
            verts, cells = [], []
            for vert in det.roi.mesh.verts:
                verts.append(
                    [
                        vert.z,
                        vert.y,
                        vert.x,
                    ]
                )
            for face in det.roi.mesh.faces:
                cells.append([face.p1, face.p2, face.p3])
            meshes.append(Mesh([verts, cells]))

    color = 1
    for mesh in meshes[::-1]:
        origin = np.floor(mesh.bounds()[::2]).astype(int)
        origin = np.maximum(origin, 0)

        max_size = np.array(label.shape) - origin

        vol = mesh.binarize(
            values=(color, 0),
            spacing=[1, 1, 1],
            origin=origin + 0.5,
        )

        vol_d = vol.tonumpy()[: max_size[0], : max_size[1], : max_size[2]]
        size = tuple(vol_d.shape)

        region = label[
            origin[0] : origin[0] + size[0],
            origin[1] : origin[1] + size[1],
            origin[2] : origin[2] + size[2],
        ]
        region[...] = np.maximum(region, vol_d)

        color = color + 1

    return label


def _render_polygons(
    response, label: np.ndarray, *, post_process: bool = False
) -> np.ndarray:
    """Render 2D polygon detections into a label array.

    Args:
        response: DetectionResponse protobuf with polygon detections
        label: 2D numpy array to render into
        post_process: whether to filter overlapping boxes

    Returns:
        label array with rendered polygons
    """
    if post_process:
        # get bboxes
        bboxes = []
        for det in response.detections:
            if det.roi.HasField("polygon"):
                x = [p.x for p in det.roi.polygon.points]
                y = [p.y for p in det.roi.polygon.points]
                bboxes.append([min(y), min(x), max(y), max(x)])

        bm = _filter_boxes(np.array(bboxes))

        detections = [
            det
            for det, selected in zip(response.detections, bm, strict=False)
            if selected
        ]

    else:
        detections = response.detections

    c = len(detections)
    for det in reversed(detections):
        polygon = [[p.x, p.y] for p in det.roi.polygon.points]
        polygon = np.round(np.array(polygon)).astype(int)

        cv2.fillPoly(label, [polygon], c)
        c = c - 1

    return label


def _generate_label(
    response, label: np.ndarray, *, post_process: bool = False
) -> np.ndarray:
    """Generate a label array from detection response.

    Args:
        response: DetectionResponse protobuf
        label: numpy array template to render into
        post_process: whether to apply box filtering

    Returns:
        label array with rendered detections
    """
    if label.ndim == 2:
        _render_polygons(response, label, post_process=post_process)

    elif label.ndim == 3:
        _render_meshes(response, label, post_process=post_process)

    else:
        raise ValueError(f"supplied label template is not 2d or 3d: {label.shape}")

    return label


def _adjust_response_offset(response, grid: Tuple[slice, ...]):
    """Adjust detection coordinates by grid offset.

    Args:
        response: DetectionResponse protobuf
        grid: tuple of slices defining the patch position

    Returns:
        response with adjusted coordinates
    """
    for det in response.detections:
        for p in det.roi.polygon.points:
            p.x += grid[1].start
            p.y += grid[0].start
        for v in det.roi.mesh.verts:
            v.x += grid[2].start
            v.y += grid[1].start
            v.z += grid[0].start

    return response
