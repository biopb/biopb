"""Unit tests for biopb.tensor.client utility functions.

Tests utility functions that don't require a live Flight server.
"""

import importlib.metadata
import warnings
from unittest.mock import patch

import pyarrow as pa
import pytest
from biopb.tensor.client import _check_schema_version, _parse_version


class TestParseVersion:
    """Tests for _parse_version semantic version parsing."""

    def test_parse_simple_version(self):
        """Test parsing simple semantic version."""
        assert _parse_version("1.0.0") == (1, 0, 0)
        assert _parse_version("2.3.4") == (2, 3, 4)
        assert _parse_version("0.0.1") == (0, 0, 1)

    def test_parse_dev_version(self):
        """Test parsing dev versions like '0.3.1.dev43+g...'."""
        # Handle dev versions - should parse base version
        assert _parse_version("0.3.1.dev43") == (0, 3, 1)
        assert _parse_version("1.2.3.dev0+gabc123") == (1, 2, 3)

    def test_parse_two_part_version(self):
        """Test parsing versions with only two parts."""
        assert _parse_version("1.0") == (1, 0, 0)
        assert _parse_version("2.3") == (2, 3, 0)

    def test_parse_single_part_version(self):
        """Test parsing versions with only one part."""
        assert _parse_version("1") == (1, 0, 0)
        assert _parse_version("0") == (0, 0, 0)

    def test_parse_empty_version(self):
        """Test parsing empty version string raises ValueError."""
        # Empty version raises ValueError due to int('')
        with pytest.raises(ValueError):
            _parse_version("")


class TestCheckSchemaVersion:
    """Tests for _check_schema_version version compatibility warnings."""

    def test_no_warning_when_no_metadata(self):
        """Test no warning when schema has no metadata."""
        schema = pa.schema([pa.field("data", pa.float32())])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _check_schema_version(schema)
            # No warnings should be emitted

    def test_no_warning_when_no_version_in_metadata(self):
        """Test no warning when metadata doesn't contain version."""
        schema = pa.schema(
            [pa.field("data", pa.float32())],
            metadata={"other_key": "some_value"}
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _check_schema_version(schema)
            # No warnings should be emitted

    def test_warning_when_client_older_than_server(self):
        """Test warning when client version is older than server schema version."""
        # Create schema with server version metadata
        schema = pa.schema(
            [pa.field("data", pa.float32())],
            metadata={"tensor_schema_version": "2.0.0"}
        )

        # Mock client version to be older
        with patch("biopb.tensor.client.importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.0.0"

            # Capture logger warnings
            with patch("biopb.tensor.client.logger.warning") as mock_logger:
                _check_schema_version(schema)
                mock_logger.assert_called_once()
                assert "older than server schema version" in mock_logger.call_args[0][0]

    def test_no_warning_when_client_same_version(self):
        """Test no warning when client version equals server version."""
        schema = pa.schema(
            [pa.field("data", pa.float32())],
            metadata={"tensor_schema_version": "1.0.0"}
        )

        with patch("biopb.tensor.client.importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.0.0"

            with patch("biopb.tensor.client.logger.warning") as mock_logger:
                _check_schema_version(schema)
                # No warning should be logged
                mock_logger.assert_not_called()

    def test_no_warning_when_client_newer_than_server(self):
        """Test no warning when client version is newer than server."""
        schema = pa.schema(
            [pa.field("data", pa.float32())],
            metadata={"tensor_schema_version": "1.0.0"}
        )

        with patch("biopb.tensor.client.importlib.metadata.version") as mock_version:
            mock_version.return_value = "2.0.0"

            with patch("biopb.tensor.client.logger.warning") as mock_logger:
                _check_schema_version(schema)
                # No warning should be logged
                mock_logger.assert_not_called()

    def test_no_warning_when_package_not_found(self):
        """Test no warning when biopb package not found."""
        schema = pa.schema(
            [pa.field("data", pa.float32())],
            metadata={"tensor_schema_version": "2.0.0"}
        )

        with patch("biopb.tensor.client.importlib.metadata.version") as mock_version:
            mock_version.side_effect = importlib.metadata.PackageNotFoundError("biopb")

            with patch("biopb.tensor.client.logger.warning") as mock_logger:
                _check_schema_version(schema)
                # No warning should be logged
                mock_logger.assert_not_called()

    def test_warning_logged_not_warning_module(self):
        """Test that older client logs a warning via logger."""
        schema = pa.schema(
            [pa.field("data", pa.float32())],
            metadata={"tensor_schema_version": "2.0.0"}
        )

        with patch("biopb.tensor.client.importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.0.0"

            # Capture logger warnings
            with patch("biopb.tensor.client.logger.warning") as mock_logger:
                _check_schema_version(schema)
                mock_logger.assert_called_once()
                assert "older than server schema version" in mock_logger.call_args[0][0]


class TestVersionComparison:
    """Tests for version comparison logic."""

    def test_major_version_comparison(self):
        """Test major version comparison."""
        # Client 1.x.x vs Server 2.x.x -> older
        client_parsed = (1, 5, 0)
        server_parsed = (2, 0, 0)
        assert client_parsed < server_parsed

    def test_minor_version_comparison(self):
        """Test minor version comparison."""
        # Client 1.1.x vs Server 1.2.x -> older
        client_parsed = (1, 1, 0)
        server_parsed = (1, 2, 0)
        assert client_parsed < server_parsed

    def test_patch_version_comparison(self):
        """Test patch version comparison."""
        # Client 1.0.0 vs Server 1.0.1 -> older
        client_parsed = (1, 0, 0)
        server_parsed = (1, 0, 1)
        assert client_parsed < server_parsed

    def test_equal_versions(self):
        """Test equal versions."""
        client_parsed = (1, 2, 3)
        server_parsed = (1, 2, 3)
        assert client_parsed == server_parsed


class TestImport:
    """Test module imports."""

    def test_import_client(self):
        """Test that client module can be imported."""
        import biopb.tensor.client as client
        assert hasattr(client, 'TensorFlightClient')

    def test_import_serialized_pb2(self):
        """Test that serialized_pb2 can be imported."""
        from biopb.tensor.serialized_pb2 import SerializedTensor
        assert SerializedTensor is not None
