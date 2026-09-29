"""Unit tests for biopb.tensor.client utility functions.

Tests utility functions that don't require a live Flight server.
"""

import pyarrow as pa
import pytest
from biopb.tensor._session import _check_wire_protocol
from biopb.tensor._wire_version import (
    TENSOR_WIRE_PROTOCOL_VERSION,
    WIRE_PROTOCOL_METADATA_KEY,
)


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
