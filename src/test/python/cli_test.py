"""Unit tests for biopb.tensor.cli diagnostic commands.

Uses typer.testing.CliRunner with mocked TensorFlightClient to avoid
requiring a live server.
"""

from typing import Dict
from unittest.mock import MagicMock, patch
import json
import os
import tempfile

import numpy as np
import pytest
from typer.testing import CliRunner

from biopb.tensor.cli import app, _parse_slice_hint
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor


runner = CliRunner()


def _build_mock_client() -> MagicMock:
    """Build a mock TensorFlightClient for testing."""
    mock_client = MagicMock()

    # Create mock source and tensor descriptors
    tensor_desc_1 = TensorDescriptor(
        array_id="pos_0",
        shape=[512, 512],
        dtype="uint8",
    )
    tensor_desc_2 = TensorDescriptor(
        array_id="pos_1",
        shape=[512, 512],
        dtype="uint16",
    )

    source_desc = DataSourceDescriptor(
        source_id="my-source",
        tensors=[tensor_desc_1, tensor_desc_2],
    )

    # Mock list_sources
    mock_client.list_sources.return_value = {
        "my-source": source_desc,
    }

    # Mock get_source_metadata
    mock_client.get_source_metadata.return_value = {
        "multiscales": [
            {"name": "0", "datasets": [{"path": "0"}]},
        ],
        "axes": [
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
    }

    # Mock cache_info
    mock_client.cache_info.return_value = {
        "size_bytes": 1_000_000,
        "max_bytes": 100_000_000,
        "item_count": 42,
        "hits": 100,
        "misses": 50,
    }

    # Mock get_tensor to return a dask-like mock
    mock_arr = MagicMock()
    mock_arr.shape = (512, 512)
    mock_arr.dtype = "uint8"
    mock_arr.size = 512 * 512
    mock_arr.min.return_value.compute.return_value = 10
    mock_arr.max.return_value.compute.return_value = 200
    mock_arr.mean.return_value.compute.return_value = 100.5
    mock_client.get_tensor.return_value = mock_arr

    return mock_client


class TestQueryCommand:
    """Tests for the 'query' command."""

    def test_query_lists_sources(self):
        """Test that query lists available sources."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 0
            assert "my-source" in result.stdout
            assert "pos_0" in result.stdout
            assert "pos_1" in result.stdout
            assert "[512, 512]" in result.stdout
            mock_client.close.assert_called_once()

    def test_query_with_custom_server(self):
        """Test that query respects --server option."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query", "--server", "grpc://custom:9000"])

            assert result.exit_code == 0
            mock_fc_class.assert_called_once_with(
                location="grpc://custom:9000",
                cache_bytes=100_000_000,
                token=None,
            )

    def test_query_shows_cache_info(self):
        """Test that query displays cache statistics."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 0
            assert "hits=100" in result.stdout
            assert "misses=50" in result.stdout

    def test_query_handles_empty_sources(self):
        """Test that query handles empty source list gracefully."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_client.list_sources.return_value = {}
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 0
            assert "No sources found" in result.stderr

    def test_query_connection_error(self):
        """Test that query handles connection errors."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.side_effect = ConnectionError("Connection refused")

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 1
            assert "Cannot connect" in result.stderr


class TestMetadataCommand:
    """Tests for the 'metadata' command."""

    def test_metadata_lists_tensors(self):
        """Test that metadata lists tensors in a source."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "my-source"])

            assert result.exit_code == 0
            assert "pos_0" in result.stdout
            assert "pos_1" in result.stdout
            mock_client.close.assert_called_once()

    def test_metadata_shows_ome_metadata(self):
        """Test that metadata displays source-level OME metadata."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "my-source"])

            assert result.exit_code == 0
            assert "multiscales" in result.stdout
            assert "axes" in result.stdout

    def test_metadata_with_specific_tensor(self):
        """Test that --tensor option shows detailed descriptor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "my-source", "--tensor", "pos_0"])

            assert result.exit_code == 0
            assert "Tensor Descriptor: pos_0" in result.stdout
            assert "uint8" in result.stdout

    def test_metadata_source_not_found(self):
        """Test that metadata handles missing source."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "nonexistent"])

            assert result.exit_code == 1
            assert "Source not found" in result.stderr

    def test_metadata_tensor_not_found(self):
        """Test that --tensor handles missing tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(
                app, ["metadata", "my-source", "--tensor", "nonexistent"]
            )

            assert result.exit_code == 1
            assert "Tensor not found" in result.stderr


class TestStatsCommand:
    """Tests for the 'stats' command."""

    def test_stats_computes_values(self):
        """Test that stats computes min, max, mean."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            with patch("biopb.tensor.cli.dask.compute") as mock_compute:
                mock_client = _build_mock_client()
                mock_fc_class.return_value = mock_client
                mock_compute.return_value = (10, 200, 100.5)

                result = runner.invoke(app, ["stats", "my-source/pos_0"])

                assert result.exit_code == 0
                assert "min" in result.stdout
                assert "max" in result.stdout
                assert "mean" in result.stdout
                assert "10" in result.stdout
                assert "200" in result.stdout
                assert "100.5" in result.stdout
                mock_client.close.assert_called_once()

    def test_stats_with_slice(self):
        """Test that stats respects --slice option."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(
                app,
                ["stats", "my-source/pos_0", "--slice", "0:100,0:100"],
            )

            assert result.exit_code == 0
            # Verify get_tensor was called with the slice
            call_args = mock_client.get_tensor.call_args
            assert call_args[1]["slice_hint"] == (slice(0, 100), slice(0, 100))

    def test_stats_missing_tensor(self):
        """Test that stats handles missing tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_client.get_tensor.side_effect = ValueError("Tensor not found")
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["stats", "my-source/nonexistent"])

            assert result.exit_code == 1
            assert "Failed to compute statistics" in result.stderr

    def test_stats_displays_shape_and_dtype(self):
        """Test that stats shows tensor shape and dtype."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["stats", "my-source/pos_0"])

            assert result.exit_code == 0
            assert "[512, 512]" in result.stdout
            assert "uint8" in result.stdout


class TestParseSliceHint:
    """Tests for _parse_slice_hint helper function."""

    def test_parse_valid_slice(self):
        """Test parsing valid slice specifications."""
        result = _parse_slice_hint("0:100,50:150")
        assert result == (slice(0, 100), slice(50, 150))

    def test_parse_slice_with_missing_start(self):
        """Test parsing slice with missing start."""
        result = _parse_slice_hint(":100,50:")
        assert result == (slice(None, 100), slice(50, None))

    def test_parse_empty_slice(self):
        """Test that empty string returns None."""
        result = _parse_slice_hint("")
        assert result is None

    def test_parse_none_slice(self):
        """Test that None returns None."""
        result = _parse_slice_hint(None)
        assert result is None

    def test_parse_slice_with_spaces(self):
        """Test that spaces are handled."""
        result = _parse_slice_hint("  0:100 , 50:150  ")
        assert result == (slice(0, 100), slice(50, 150))

    def test_parse_invalid_format(self):
        """Test that invalid format raises BadParameter."""
        from typer import BadParameter

        with pytest.raises(BadParameter):
            _parse_slice_hint("0-100")

    def test_parse_single_dimension(self):
        """Test parsing single-dimension slice."""
        result = _parse_slice_hint("10:20")
        assert result == (slice(10, 20),)


class TestArrayIdFirstAddressing:
    """The CLI must address tensors by array_id alone (identity policy), not the
    deprecated ``(source_id, tensor_id)`` pair. See biopb/biopb#75."""

    def test_get_passes_array_id_as_single_argument(self):
        """`get` forwards the raw array_id positionally to get_tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            # The pickle output path serializes the returned array, so hand back
            # a real (picklable) array instead of the MagicMock default.
            mock_client.get_tensor.return_value = np.zeros((4, 4), dtype="uint8")
            mock_fc_class.return_value = mock_client

            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "out.pkl")
                result = runner.invoke(app, ["get", "my-source/pos_0", "-o", out])

            assert result.exit_code == 0, result.stderr
            call_args = mock_client.get_tensor.call_args
            # array_id passed as the single first positional argument; no
            # deprecated source_id=/tensor_id= keywords.
            assert call_args.args[0] == "my-source/pos_0"
            assert "source_id" not in call_args.kwargs
            assert "tensor_id" not in call_args.kwargs

    def test_stats_passes_array_id_as_single_argument(self):
        """`stats` forwards the raw array_id positionally to get_tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["stats", "my-source/pos_0"])

            assert result.exit_code == 0, result.stderr
            call_args = mock_client.get_tensor.call_args
            assert call_args.args[0] == "my-source/pos_0"
            assert "source_id" not in call_args.kwargs
            assert "tensor_id" not in call_args.kwargs


class TestCliIntegration:
    """Integration-level CLI tests."""

    def test_help_messages(self):
        """Test that all commands have helpful help text."""
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "Data sources" in result.stdout or "sources" in result.stdout

        result = runner.invoke(app, ["metadata", "--help"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0

    def test_app_version(self):
        """Test that the app has a name and help text."""
        assert app.info.name == "tensor"
        assert app.info.help is not None
