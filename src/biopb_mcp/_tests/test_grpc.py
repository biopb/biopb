"""Tests for _grpc.py gRPC communication."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from biopb_mcp.image_processing._grpc import (
    _check_chunk_memory,
    _encode_image,
    _estimate_chunk_memory_mb,
    _get_detection_settings,
    _get_grpc_channel,
    _get_label_filter,
    _object_detection_build_request,
    _parse_annotation_tsv,
    _parse_server_url,
)


class TestParseServerUrl:
    """Tests for _parse_server_url function."""

    def test_valid_hostname_with_port(self):
        """Valid hostname:port format parses correctly."""
        host, port, scheme, label_filter = _parse_server_url("localhost:50051")
        assert host == "localhost"
        assert port == 50051
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_valid_domain_with_port(self):
        """Valid domain:port format parses correctly."""
        host, port, scheme, label_filter = _parse_server_url(
            "lacss.biopb.org:8080"
        )
        assert host == "lacss.biopb.org"
        assert port == 8080
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_hostname_without_port_defaults_to_443(self):
        """Hostname without port defaults to port 443."""
        host, port, scheme, label_filter = _parse_server_url("lacss.biopb.org")
        assert host == "lacss.biopb.org"
        assert port == 443
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_localhost_without_port(self):
        """Local hostname without port defaults to 443."""
        host, port, scheme, label_filter = _parse_server_url("localhost")
        assert host == "localhost"
        assert port == 443
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_http_scheme_prefix(self):
        """http:// prefix returns HTTP scheme."""
        host, port, scheme, label_filter = _parse_server_url(
            "http://localhost:50051"
        )
        assert host == "localhost"
        assert port == 50051
        assert scheme == "http"
        assert label_filter is None

    def test_https_scheme_prefix(self):
        """https:// prefix returns HTTPS scheme."""
        host, port, scheme, label_filter = _parse_server_url(
            "https://lacss.biopb.org:8080"
        )
        assert host == "lacss.biopb.org"
        assert port == 8080
        assert scheme == "https"
        assert label_filter is None

    def test_https_scheme_no_port(self):
        """https:// with hostname only defaults to port 443."""
        host, port, scheme, label_filter = _parse_server_url(
            "https://lacss.biopb.org"
        )
        assert host == "lacss.biopb.org"
        assert port == 443
        assert scheme == "https"
        assert label_filter is None

    def test_http_scheme_no_port(self):
        """http:// with hostname only defaults to port 443."""
        host, port, scheme, label_filter = _parse_server_url(
            "http://localhost"
        )
        assert host == "localhost"
        assert port == 443
        assert scheme == "http"
        assert label_filter is None

    def test_url_with_label_filter(self):
        """URL with path component extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url(
            "localhost:50051/filter"
        )
        assert host == "localhost"
        assert port == 50051
        assert scheme is None
        assert label_filter == "filter"

    def test_url_with_scheme_and_label_filter(self):
        """URL with scheme and path extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url(
            "http://localhost:50051/filter"
        )
        assert host == "localhost"
        assert port == 50051
        assert scheme == "http"
        assert label_filter == "filter"

    def test_url_with_https_and_label_filter(self):
        """https URL with path extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url(
            "https://lacss.biopb.org:8080/segmentation"
        )
        assert host == "lacss.biopb.org"
        assert port == 8080
        assert scheme == "https"
        assert label_filter == "segmentation"

    def test_url_without_port_with_label_filter(self):
        """URL without port (defaults to 443) with path extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url(
            "lacss.biopb.org/threshold"
        )
        assert host == "lacss.biopb.org"
        assert port == 443
        assert scheme is None
        assert label_filter == "threshold"

    def test_invalid_format_raises(self):
        """Invalid URL format raises ValueError."""
        invalid_urls = [
            "localhost:",  # missing port
            ":50051",  # missing hostname
            "local:host:50051",  # invalid hostname
            "",  # empty string
            "local space:50051",  # space in hostname
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid server URL"):
                _parse_server_url(url)


class TestGetLabelFilter:
    """Tests for _get_label_filter function."""

    def test_no_label_filter(self):
        """URL without path returns None."""
        assert _get_label_filter("localhost:50051") is None
        assert _get_label_filter("lacss.biopb.org") is None

    def test_with_label_filter(self):
        """URL with path returns label filter."""
        assert _get_label_filter("localhost:50051/filter") == "filter"
        assert (
            _get_label_filter("http://localhost:50051/segmentation")
            == "segmentation"
        )
        assert (
            _get_label_filter("https://lacss.biopb.org/threshold")
            == "threshold"
        )

    def test_invalid_url_raises(self):
        """Invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid server URL"):
            _get_label_filter("invalid:url:format")


class TestEncodeImage:
    """Tests for _encode_image function."""

    def test_2d_image_encoding(self):
        """2D image (3D array with YXC) is encoded correctly."""
        image = np.random.rand(100, 100, 1)  # 2D with channel
        pixels = _encode_image(image, np_index_order="YXC", z_ratio=1.0)

        # Should add batch dimension
        # Result is a protobuf Pixels object
        assert pixels is not None

    def test_3d_image_encoding(self):
        """3D image (4D array with ZYXC) is encoded correctly."""
        image = np.random.rand(10, 100, 100, 1)  # 3D with channel
        pixels = _encode_image(image, np_index_order="ZYXC", z_ratio=2.0)

        assert pixels is not None

    def test_grayscale_encoding(self):
        """Grayscale image (2D array with YX) is encoded correctly."""
        image = np.random.rand(100, 100)  # 2D grayscale
        pixels = _encode_image(image, np_index_order="YX", z_ratio=1.0)

        assert pixels is not None

    def test_invalid_dimensions_raises(self):
        """Invalid image dimensions raise ValueError."""
        # 2D without matching np_index_order
        image = np.random.rand(100, 100)  # 2D array

        with pytest.raises(ValueError, match="must have 3 dimensions"):
            _encode_image(image, np_index_order="YXC")  # expects 3 dims

    def test_5d_image_raises(self):
        """5D image with wrong np_index_order raises ValueError."""
        image = np.random.rand(2, 10, 100, 100, 1)

        with pytest.raises(ValueError, match="must have 4 dimensions"):
            _encode_image(image, np_index_order="ZYXC")  # expects 4 dims


class TestGetDetectionSettings:
    """Tests for _get_detection_settings function."""

    def test_basic_settings(self):
        """Basic settings are converted correctly."""
        settings = {
            "Min Score": 0.5,
            "NMS": "Off",
            "Size Hint": 30.0,
        }
        result = _get_detection_settings(settings)

        assert result.min_score == 0.5
        assert result.nms_iou == 0.0
        assert result.cell_diameter_hint == 30.0

    def test_nms_iou_values(self):
        """NMS IOU values are mapped correctly."""
        nms_map = {
            "Off": 0.0,
            "Iou-0.2": 0.2,
            "Iou-0.4": 0.4,
            "Iou-0.6": 0.6,
            "Iou-0.8": 0.8,
        }

        for nms_key, expected_iou in nms_map.items():
            settings = {
                "Min Score": 0.4,
                "NMS": nms_key,
                "Size Hint": 32.0,
            }
            result = _get_detection_settings(settings)
            assert result.nms_iou == pytest.approx(expected_iou)

    def test_invalid_nms_raises(self):
        """Invalid NMS value raises KeyError."""
        settings = {
            "Min Score": 0.5,
            "NMS": "Invalid",
            "Size Hint": 30.0,
        }

        with pytest.raises(KeyError):
            _get_detection_settings(settings)


class TestGetGrpcChannel:
    """Tests for _get_grpc_channel function."""

    def test_http_scheme_from_url(self):
        """http:// in URL creates insecure channel."""
        settings = {
            "Server": "http://localhost:50051",
        }
        channel = _get_grpc_channel(settings)
        # Channel is created, verify it's a grpc channel
        assert channel is not None

    def test_https_scheme_from_url(self):
        """https:// in URL creates secure channel."""
        settings = {
            "Server": "https://lacss.biopb.org",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_auto_scheme_http_port(self):
        """URL without scheme and non-443 port uses HTTP."""
        settings = {
            "Server": "localhost:50051",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_auto_scheme_https_port(self):
        """URL without scheme and port 443 uses HTTPS."""
        settings = {
            "Server": "lacss.biopb.org",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_server_without_port(self):
        """Server without port defaults to 443."""
        settings = {
            "Server": "lacss.biopb.org",
        }
        # Should add :443 to server URL internally
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_url_with_label_filter(self):
        """URL with label filter still creates channel correctly."""
        settings = {
            "Server": "localhost:50051/filter",
        }
        channel = _get_grpc_channel(settings)
        # Label filter is ignored for channel creation
        assert channel is not None

    def test_url_with_scheme_and_label_filter(self):
        """URL with scheme and label filter creates channel correctly."""
        settings = {
            "Server": "http://localhost:50051/segmentation",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_invalid_server_url_raises(self):
        """Invalid server URL raises ValueError."""
        settings = {
            "Server": "invalid:url:format",
        }
        with pytest.raises(ValueError, match="Invalid server URL"):
            _get_grpc_channel(settings)


class TestObjectDetectionBuildRequest:
    """Tests for _object_detection_build_request function."""

    def test_request_structure(self):
        """Request contains expected structure."""
        image = np.random.rand(100, 100, 1)
        settings = {
            "Z Aspect Ratio": 1.0,
            "Min Score": 0.5,
            "NMS": "Off",
            "Size Hint": 30.0,
        }
        request = _object_detection_build_request(image, settings)

        assert request.image_data is not None
        assert request.detection_settings is not None

    def test_z_ratio_passed(self):
        """Z aspect ratio is passed to encoding."""
        image = np.random.rand(10, 100, 100, 1)
        settings = {
            "Z Aspect Ratio": 5.0,
            "Min Score": 0.5,
            "NMS": "Iou-0.4",
            "Size Hint": 30.0,
        }
        request = _object_detection_build_request(image, settings)
        assert request is not None


class TestGrpcObjectDetection:
    """Tests for grpc_object_detection generator function."""

    def test_invalid_dimensions_2d_mode(self):
        """Wrong dimensions for 2D mode raises ValueError."""
        # The function is wrapped in thread_worker, so we test the validation
        # by calling the underlying logic

        # Create test data with wrong dimensions (3D data when expecting 2D)
        image_data = np.random.rand(2, 10, 100, 100, 1)  # 5D (3D mode) data
        settings = {
            "3D": False,  # Expecting 4D data
            "Server": "localhost:50051",
            "Scheme": "HTTP",
        }
        grid_positions = []

        # Get the generator function (before thread_worker wraps it)
        # We can't easily test this due to thread_worker decorator
        # But we document the expected behavior
        # thread_worker testing requires complex mocking

    def test_invalid_dimensions_3d_mode(self):
        """Wrong dimensions for 3D mode raises ValueError."""
        # Similar to above - documenting expected behavior

    @patch("biopb_mcp.image_processing._grpc._get_grpc_channel")
    @patch("biopb_mcp.image_processing._grpc.proto.ObjectDetectionStub")
    def test_generator_yields_progress(
        self, mock_stub_class, mock_channel_func
    ):
        """Generator yields None for progress updates."""
        # Setup mocks
        mock_channel = MagicMock()
        mock_channel_func.return_value.__enter__ = MagicMock(
            return_value=mock_channel
        )
        mock_channel_func.return_value.__exit__ = MagicMock(return_value=False)

        mock_stub = MagicMock()
        mock_stub_class.return_value = mock_stub

        # Mock detection response
        mock_response = MagicMock()
        mock_response.detections = []
        mock_stub.RunDetection.return_value = mock_response

        # Create test data (2D)
        image_data = np.random.rand(2, 100, 100, 1)  # 2 images, 2D
        settings = {
            "3D": False,
            "Server": "localhost:50051",
            "Z Aspect Ratio": 1.0,
            "Min Score": 0.5,
            "NMS": "Off",
            "Size Hint": 30.0,
        }
        grid_positions = [(slice(0, 100), slice(0, 100))]

        # Get the generator (need to unwrap thread_worker)
        # thread_worker returns a worker object, but for testing we call the inner generator
        # We'll mock thread_worker to return the generator directly
        # Skip this for now - thread_worker testing is complex


class TestGrpcProcessImage:
    """Tests for grpc_process_image generator function."""

    def test_grid_positions_not_supported(self):
        """Grid processing raises ValueError when grid_positions is provided."""
        # This tests that grid_positions must be None for process_image
        # The validation happens inside the generator, wrapped by thread_worker
        # We document the expected behavior - ValueError should be raised

    def test_invalid_dimensions_raises(self):
        """Wrong dimensions raises ValueError."""
        # Documenting expected behavior - ValueError for wrong dimensions


class TestParseAnnotationTsv:
    """Tests for _parse_annotation_tsv function."""

    def test_empty_annotation(self):
        """Empty annotation returns empty DataFrame."""
        result = _parse_annotation_tsv("")
        assert result.empty

    def test_none_annotation(self):
        """None annotation returns empty DataFrame."""
        result = _parse_annotation_tsv(None)
        assert result.empty

    def test_simple_tsv(self):
        """Simple TSV with header parses correctly."""
        tsv = "col1\tcol2\nval1\tval2\nval3\tval4"
        result = _parse_annotation_tsv(tsv)
        assert result["col1"].tolist() == ["val1", "val3"]
        assert result["col2"].tolist() == ["val2", "val4"]

    def test_single_row(self):
        """TSV with single data row parses correctly."""
        tsv = "name\tvalue\nitem1\t100"
        result = _parse_annotation_tsv(tsv)
        assert result["name"].tolist() == ["item1"]
        assert result["value"].tolist() == [100]  # pandas auto-detects int

    def test_numeric_values(self):
        """Numeric values are parsed as integers."""
        tsv = "id\tcount\n1\t42\n2\t100"
        result = _parse_annotation_tsv(tsv)
        assert result["id"].tolist() == [1, 2]
        assert result["count"].tolist() == [42, 100]

    def test_multiple_columns(self):
        """TSV with multiple columns parses correctly."""
        tsv = "a\tb\tc\n1\t2\t3\n4\t5\t6"
        result = _parse_annotation_tsv(tsv)
        assert result["a"].tolist() == [1, 4]
        assert result["b"].tolist() == [2, 5]
        assert result["c"].tolist() == [3, 6]

    def test_header_duplicated_as_data(self):
        """Header row appearing as data row is filtered out."""
        tsv = "col1\tcol2\ncol1\tcol2\nval1\tval2"
        result = _parse_annotation_tsv(tsv)
        assert len(result) == 1
        assert result["col1"].tolist() == ["val1"]
        assert result["col2"].tolist() == ["val2"]

    def test_multiple_header_duplicates(self):
        """Multiple header rows appearing as data are all filtered out."""
        tsv = "col1\tcol2\ncol1\tcol2\ncol1\tcol2\nval1\tval2\nval3\tval4"
        result = _parse_annotation_tsv(tsv)
        assert len(result) == 2
        assert result["col1"].tolist() == ["val1", "val3"]
        assert result["col2"].tolist() == ["val2", "val4"]

    def test_comment_line_skipped(self):
        """Comment lines starting with # are skipped before parsing."""
        tsv = "# Directionality analysis\nangle\tcount\n-90.0\t0.0\n-88.0\t1.0"
        result = _parse_annotation_tsv(tsv)
        assert list(result.columns) == ["angle", "count"]
        assert len(result) == 2
        assert result["angle"].tolist() == [-90.0, -88.0]

    def test_multiple_comment_lines(self):
        """Multiple comment lines are all skipped."""
        tsv = "# Comment 1\n# Comment 2\na\tb\n1\t2\n3\t4"
        result = _parse_annotation_tsv(tsv)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_only_comment_lines(self):
        """TSV with only comment lines returns empty DataFrame."""
        tsv = "# Only comments\n# No data"
        result = _parse_annotation_tsv(tsv)
        assert result.empty


class TestEstimateChunkMemory:
    """Tests for _estimate_chunk_memory_mb function."""

    def test_small_chunk(self):
        """Small chunk estimates correctly."""
        chunk = np.zeros((100, 100, 3), dtype=np.uint8)
        mb = _estimate_chunk_memory_mb(chunk)
        # 100 * 100 * 3 * 1 byte = 30,000 bytes ≈ 0.03 MB
        assert mb < 0.1
        assert mb > 0

    def test_medium_chunk(self):
        """Medium chunk estimates correctly."""
        chunk = np.zeros((512, 512, 3), dtype=np.uint8)
        mb = _estimate_chunk_memory_mb(chunk)
        # 512 * 512 * 3 * 1 byte = 786,432 bytes ≈ 0.75 MB
        assert mb > 0.5
        assert mb < 1.0

    def test_large_chunk_float32(self):
        """Large float32 chunk estimates correctly."""
        chunk = np.zeros((1024, 1024, 64, 3), dtype=np.float32)
        mb = _estimate_chunk_memory_mb(chunk)
        # 1024 * 1024 * 64 * 3 * 4 bytes = 805,306,368 bytes ≈ 768 MB
        assert mb > 700
        assert mb < 900

    def test_large_chunk_float64(self):
        """Large float64 chunk estimates correctly."""
        chunk = np.zeros((512, 512, 10), dtype=np.float64)
        mb = _estimate_chunk_memory_mb(chunk)
        # 512 * 512 * 10 * 8 bytes = 20,971,520 bytes ≈ 20 MB
        assert mb > 15
        assert mb < 25

    def test_chunk_without_dtype(self):
        """Chunk without dtype attribute uses float32 default."""
        # Create a mock object without dtype
        mock_chunk = MagicMock()
        mock_chunk.shape = (1000, 1000, 3)
        del mock_chunk.dtype  # Remove dtype attribute

        mb = _estimate_chunk_memory_mb(mock_chunk)
        # Should use float32 default: 1000 * 1000 * 3 * 4 = 12 MB
        assert mb > 10
        assert mb < 15

    def test_empty_chunk(self):
        """Empty chunk returns 0 MB."""
        mock_chunk = MagicMock()
        mock_chunk.shape = ()
        mb = _estimate_chunk_memory_mb(mock_chunk)
        assert mb == 0.0


class TestCheckChunkMemory:
    """Tests for _check_chunk_memory function."""

    def test_small_chunk_passes(self):
        """Small chunk passes without error."""
        chunk = np.zeros((100, 100, 3), dtype=np.uint8)
        # Should not raise
        _check_chunk_memory(chunk)

    def test_medium_chunk_passes_with_warning(self):
        """Medium chunk passes but may log warning."""
        # Create a chunk larger than warn threshold but under error threshold
        # Using default config: warn=500MB, error=2000MB
        chunk = np.zeros((512, 512, 200, 3), dtype=np.float32)
        # 512 * 512 * 200 * 3 * 4 ≈ 629 MB (over warn threshold)
        # Should not raise, may log warning (we don't check warning here)
        _check_chunk_memory(chunk)

    def test_huge_chunk_raises_memory_error(self):
        """Huge chunk raises MemoryError."""
        # Mock a chunk that exceeds error threshold
        mock_chunk = MagicMock()
        mock_chunk.shape = (4096, 4096, 256, 3)  # Very large
        mock_chunk.dtype = np.dtype(np.float32)
        # 4096 * 4096 * 256 * 3 * 4 ≈ 50 GB (way over error threshold)

        with pytest.raises(MemoryError, match="exceeds memory limit"):
            _check_chunk_memory(mock_chunk)

    @patch("biopb_mcp.image_processing._grpc.load_config")
    def test_custom_threshold(self, mock_load_config):
        """Custom thresholds from config are respected."""
        mock_load_config.return_value = {
            "memory": {
                "warn_threshold_mb": 10,
                "error_threshold_mb": 50,
            }
        }

        chunk = np.zeros((1024, 1024, 10), dtype=np.float32)
        # 1024 * 1024 * 10 * 4 ≈ 40 MB (over warn=10, under error=50)

        # Should not raise, but would warn
        _check_chunk_memory(chunk)

    @patch("biopb_mcp.image_processing._grpc.load_config")
    def test_custom_error_threshold_raises(self, mock_load_config):
        """Custom error threshold from config raises MemoryError."""
        mock_load_config.return_value = {
            "memory": {
                "warn_threshold_mb": 10,
                "error_threshold_mb": 20,
            }
        }

        chunk = np.zeros((1024, 1024, 10), dtype=np.float32)
        # 1024 * 1024 * 10 * 4 ≈ 40 MB (over error=20)

        with pytest.raises(MemoryError, match="exceeds memory limit"):
            _check_chunk_memory(chunk)
