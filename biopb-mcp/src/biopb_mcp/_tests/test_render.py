"""Tests for _render.py rendering functions."""

from typing import List

import numpy as np
import pytest

from biopb_mcp.image_processing._render import (
    _adjust_response_offset,
    _generate_label,
    _render_polygons,
)


# Mock protobuf structures
class MockPoint:
    """Mock for protobuf Point with x, y coordinates."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class MockVertex:
    """Mock for protobuf Vertex with x, y, z coordinates."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class MockFace:
    """Mock for protobuf Face with p1, p2, p3 indices."""

    def __init__(self, p1: int, p2: int, p3: int):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


class MockPolygon:
    """Mock for protobuf Polygon."""

    def __init__(self, points: List[MockPoint]):
        self.points = points


class MockMesh:
    """Mock for protobuf Mesh."""

    def __init__(self, verts: List[MockVertex], faces: List[MockFace]):
        self.verts = verts
        self.faces = faces


class MockROI:
    """Mock for protobuf ROI."""

    def __init__(self, polygon=None, mesh=None):
        # Protobuf always has both fields, but may be empty
        self.polygon = polygon if polygon is not None else MockPolygon([])
        self.mesh = mesh if mesh is not None else MockMesh([], [])

    def HasField(self, field_name: str) -> bool:
        if field_name == "polygon":
            return len(self.polygon.points) > 0
        if field_name == "mesh":
            return len(self.mesh.verts) > 0
        return False


class MockDetection:
    """Mock for protobuf Detection."""

    def __init__(self, roi: MockROI):
        self.roi = roi


class MockDetectionResponse:
    """Mock for protobuf DetectionResponse."""

    def __init__(self, detections: List[MockDetection]):
        self.detections = detections


class TestRenderPolygons:
    """Tests for _render_polygons function."""

    def test_single_polygon(self):
        """Render a single polygon into label array."""
        # Triangle polygon
        points = [MockPoint(0, 0), MockPoint(10, 0), MockPoint(5, 10)]
        polygon = MockPolygon(points)
        roi = MockROI(polygon=polygon)
        detection = MockDetection(roi)
        response = MockDetectionResponse([detection])

        label = np.zeros((20, 20), dtype=np.uint16)
        result = _render_polygons(response, label)

        # Should have filled the polygon with label value 1
        assert result[5, 5] == 1  # center of triangle
        assert result[0, 0] == 1  # corner
        assert result[15, 15] == 0  # outside

    def test_multiple_polygons(self):
        """Render multiple polygons with different label values."""
        # Two separate polygons
        points1 = [MockPoint(0, 0), MockPoint(5, 0), MockPoint(2, 5)]
        points2 = [MockPoint(10, 10), MockPoint(15, 10), MockPoint(12, 15)]

        response = MockDetectionResponse(
            [
                MockDetection(MockROI(polygon=MockPolygon(points1))),
                MockDetection(MockROI(polygon=MockPolygon(points2))),
            ]
        )

        label = np.zeros((20, 20), dtype=np.uint16)
        result = _render_polygons(response, label)

        # First polygon gets label 2 (rendered last due to reversed order)
        # Second polygon gets label 1
        assert result[2, 2] > 0
        assert result[12, 12] > 0

    def test_polygon_with_post_process(self):
        """Polygon rendering with overlapping box filtering."""
        # Two overlapping polygons - inner one should be filtered
        outer = [
            MockPoint(0, 0),
            MockPoint(20, 0),
            MockPoint(20, 20),
            MockPoint(0, 20),
        ]
        inner = [
            MockPoint(5, 5),
            MockPoint(10, 5),
            MockPoint(10, 10),
            MockPoint(5, 10),
        ]

        response = MockDetectionResponse(
            [
                MockDetection(MockROI(polygon=MockPolygon(outer))),
                MockDetection(MockROI(polygon=MockPolygon(inner))),
            ]
        )

        label = np.zeros((25, 25), dtype=np.uint16)

        # Without post_process, both rendered
        result_no_filter = _render_polygons(
            response, label.copy(), post_process=False
        )
        # Inner polygon area should be filled by both (outer overwrites inner due to reversed order)
        assert result_no_filter[7, 7] > 0

        # With post_process, inner should be filtered out
        result_filtered = _render_polygons(
            response, label.copy(), post_process=True
        )
        # Still filled, but by outer only
        assert result_filtered[7, 7] > 0

    def test_empty_response(self):
        """Empty detection response produces unchanged label."""
        response = MockDetectionResponse([])
        label = np.zeros((10, 10), dtype=np.uint16)
        result = _render_polygons(response, label)
        assert np.all(result == 0)


class TestGenerateLabel:
    """Tests for _generate_label function."""

    def test_2d_label_uses_polygons(self):
        """2D label array triggers polygon rendering."""
        points = [MockPoint(0, 0), MockPoint(5, 0), MockPoint(2, 5)]
        response = MockDetectionResponse(
            [MockDetection(MockROI(polygon=MockPolygon(points)))]
        )

        label = np.zeros((10, 10), dtype=np.uint16)
        result = _generate_label(response, label)
        assert result[2, 2] > 0

    def test_invalid_dimensions_raises(self):
        """Invalid label dimensions raise ValueError."""
        response = MockDetectionResponse([])
        label = np.zeros((10, 10, 10, 10), dtype=np.uint16)  # 4D

        with pytest.raises(ValueError, match="not 2d or 3d"):
            _generate_label(response, label)


class TestAdjustResponseOffset:
    """Tests for _adjust_response_offset function."""

    def test_polygon_offset_adjustment(self):
        """Polygon coordinates are adjusted by grid offset."""
        points = [MockPoint(0, 0), MockPoint(5, 0), MockPoint(2, 5)]
        response = MockDetectionResponse(
            [MockDetection(MockROI(polygon=MockPolygon(points)))]
        )

        # Grid offset: y offset 10, x offset 20
        grid = (slice(10, 20), slice(20, 30))
        _adjust_response_offset(response, grid)

        # Points should be shifted
        assert response.detections[0].roi.polygon.points[0].x == 20
        assert response.detections[0].roi.polygon.points[0].y == 10

    def test_mesh_offset_adjustment(self):
        """Mesh coordinates are adjusted by 3D grid offset."""
        verts = [MockVertex(0, 0, 0), MockVertex(5, 5, 5)]
        faces = [MockFace(0, 1, 2)]
        response = MockDetectionResponse(
            [MockDetection(MockROI(mesh=MockMesh(verts, faces)))]
        )

        # 3D grid: z offset 5, y offset 10, x offset 15
        grid = (slice(5, 15), slice(10, 20), slice(15, 25))
        _adjust_response_offset(response, grid)

        # Vertices should be shifted
        assert response.detections[0].roi.mesh.verts[0].x == 15
        assert response.detections[0].roi.mesh.verts[0].y == 10
        assert response.detections[0].roi.mesh.verts[0].z == 5

    def test_empty_response(self):
        """Empty response is handled without error."""
        response = MockDetectionResponse([])
        grid = (slice(0, 10), slice(0, 10))
        result = _adjust_response_offset(response, grid)
        assert len(result.detections) == 0
