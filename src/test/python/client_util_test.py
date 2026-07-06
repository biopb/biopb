"""Unit tests for biopb.tensor.client utility functions.

Tests utility functions that don't require a live Flight server.
"""

import pyarrow as pa
import pytest
from biopb.tensor._wire_version import (
    TENSOR_WIRE_PROTOCOL_VERSION,
    WIRE_PROTOCOL_METADATA_KEY,
)
from biopb.tensor.client import _check_wire_protocol, _parse_version


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


def _schema_with_protocol(version):
    """A chunk schema stamped with the given wire-protocol version (None = unstamped)."""
    md = {} if version is None else {WIRE_PROTOCOL_METADATA_KEY: str(version)}
    return pa.schema([pa.field("data", pa.binary())], metadata=md)


class TestCheckWireProtocol:
    """The hard chunk wire-protocol guard (biopb/biopb#293).

    A version mismatch means the client would misread every chunk, so the guard
    raises at GetFlightInfo instead of warning and proceeding.
    """

    def test_matching_version_passes(self):
        _check_wire_protocol(_schema_with_protocol(TENSOR_WIRE_PROTOCOL_VERSION))

    def test_older_server_raises(self):
        with pytest.raises(RuntimeError, match="wire protocol"):
            _check_wire_protocol(
                _schema_with_protocol(TENSOR_WIRE_PROTOCOL_VERSION - 1)
            )

    def test_newer_server_raises(self):
        with pytest.raises(RuntimeError, match="wire protocol"):
            _check_wire_protocol(
                _schema_with_protocol(TENSOR_WIRE_PROTOCOL_VERSION + 1)
            )

    def test_unstamped_schema_is_v1_and_raises(self):
        # A pre-#293 server sends no protocol tag; it speaks the v1 typed schema
        # this client can't read, so reject rather than fail cryptically.
        with pytest.raises(RuntimeError, match="wire protocol"):
            _check_wire_protocol(_schema_with_protocol(None))

    def test_no_metadata_at_all_raises(self):
        with pytest.raises(RuntimeError, match="wire protocol"):
            _check_wire_protocol(pa.schema([pa.field("data", pa.binary())]))

    def test_malformed_version_raises(self):
        with pytest.raises(RuntimeError, match="wire protocol"):
            _check_wire_protocol(_schema_with_protocol("not-an-int"))

    def test_error_names_the_stale_side(self):
        with pytest.raises(RuntimeError, match="upgrade the server"):
            _check_wire_protocol(
                _schema_with_protocol(TENSOR_WIRE_PROTOCOL_VERSION - 1)
            )
        with pytest.raises(RuntimeError, match="upgrade the client"):
            _check_wire_protocol(
                _schema_with_protocol(TENSOR_WIRE_PROTOCOL_VERSION + 1)
            )


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

        assert hasattr(client, "TensorFlightClient")

    def test_import_serialized_pb2(self):
        """Test that serialized_pb2 can be imported."""
        from biopb.tensor.serialized_pb2 import SerializedTensor

        assert SerializedTensor is not None
