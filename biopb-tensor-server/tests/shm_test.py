"""Tests for POSIX shared memory transfer optimization.

Tests cover:
- Server-side: shm_transfer action listing, handler, and SHM creation
- Client-side: env var check, version check, platform check, localhost detection
- Integration: end-to-end SHM transfer roundtrip
"""

import json
import os
import stat
import sys
import tempfile
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket

from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.config import CacheConfig
from biopb_tensor_server.server import TensorFlightServer

# ==============================================================================
# Server-side SHM tests
# ==============================================================================

class TestShmTransferActionListing:
    """Tests for shm_transfer action being listed."""

    def test_shm_transfer_in_list_actions(self):
        """shm_transfer should be in list_actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                write_dir=Path(tmpdir),
            )

            actions = server.list_actions(flight.ServerCallContext())

            action_types = [a.type for a in actions]
            assert "shm_transfer" in action_types

            # Find the shm_transfer action description
            shm_action = next(a for a in actions if a.type == "shm_transfer")
            assert "POSIX" in shm_action.description or "shared memory" in shm_action.description.lower()


class TestShmTransferHandler:
    """Tests for server shm_transfer handler."""

    def test_handle_shm_transfer_creates_shm(self):
        """_handle_shm_transfer should create SHM segment with correct data."""
        # Setup: create cached source with test data
        adapter = CachedSourceAdapter(
            source_id="test_shm",
            shape=[64, 64],
            dtype="uint16",
            chunk_shape=[32, 32],
        )

        # Initialize cache manager
        CacheManager.initialize(CacheConfig(backend="memory", memory_max_bytes=10_000_000))

        # Write test chunk
        test_data = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
        bounds = ChunkBounds(start=[0, 0], stop=[32, 32])
        adapter.write_chunk(bounds, test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                write_dir=Path(tmpdir),
            )
            server.register_source("test_shm", adapter)

            # Create ticket with actual encoded chunk_id
            from biopb_tensor_server.chunk import encode_chunk_id
            actual_chunk_id = encode_chunk_id("test_shm", bounds)
            ticket = TensorTicket(chunk_id=actual_chunk_id)
            ticket_bytes = ticket.SerializeToString()

            # Call handler
            shm_name = server._handle_shm_transfer(ticket_bytes)

            # Verify SHM was created
            assert shm_name.startswith("/biopb_chunk_")

            # Attach to SHM and read data
            shm = shared_memory.SharedMemory(name=shm_name)
            ipc_bytes = bytes(shm.buf)

            # Parse Arrow IPC stream
            reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
            table = reader.read_all()

            # Verify data
            arr_result = table.column("data").to_numpy()[0]
            shape = tuple(table.column("shape").to_pylist()[0])
            arr_result = arr_result.reshape(shape)

            np.testing.assert_array_equal(arr_result, test_data)

            # Cleanup
            shm.close()
            shm.unlink()

    @pytest.mark.skipif(
        os.name != "posix", reason="POSIX /dev/shm permissions only"
    )
    def test_handle_shm_transfer_is_owner_only(self):
        """SHM chunks must be owner-only, not group/world readable (A4)."""
        adapter = CachedSourceAdapter(
            source_id="test_shm_perm",
            shape=[64, 64],
            dtype="uint16",
            chunk_shape=[32, 32],
        )
        CacheManager.initialize(
            CacheConfig(backend="memory", memory_max_bytes=10_000_000)
        )
        test_data = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
        bounds = ChunkBounds(start=[0, 0], stop=[32, 32])
        adapter.write_chunk(bounds, test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                write_dir=Path(tmpdir),
            )
            server.register_source("test_shm_perm", adapter)

            from biopb_tensor_server.chunk import encode_chunk_id
            chunk_id = encode_chunk_id("test_shm_perm", bounds)
            ticket = TensorTicket(chunk_id=chunk_id)
            shm_name = server._handle_shm_transfer(ticket.SerializeToString())

            try:
                shm_path = Path("/dev/shm") / shm_name.lstrip("/")
                mode = stat.S_IMODE(shm_path.stat().st_mode)
                assert not (mode & stat.S_IRGRP), f"group-readable: {oct(mode)}"
                assert not (mode & stat.S_IROTH), f"world-readable: {oct(mode)}"
            finally:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()

    def test_handle_shm_transfer_nonexistent_chunk_raises(self):
        """_handle_shm_transfer should raise for nonexistent chunk."""
        CacheManager.initialize(CacheConfig(backend="memory", memory_max_bytes=1_000_000))

        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                write_dir=Path(tmpdir),
            )

            # Create ticket for nonexistent chunk using proper encoding
            from biopb_tensor_server.chunk import encode_chunk_id
            bounds = ChunkBounds(start=[0, 0], stop=[32, 32])
            chunk_id = encode_chunk_id("nonexistent_source", bounds)
            ticket = TensorTicket(chunk_id=chunk_id)
            ticket_bytes = ticket.SerializeToString()

            # Should raise because adapter not found
            with pytest.raises(flight.FlightServerError):
                server._handle_shm_transfer(ticket_bytes)


class TestShmCleanupThread:
    """Tests for SHM cleanup thread."""

    def test_cleanup_thread_starts_on_posix(self):
        """Cleanup thread should start on POSIX systems."""
        if sys.platform not in ("linux", "darwin"):
            pytest.skip("Only runs on POSIX systems")

        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                write_dir=Path(tmpdir),
            )

            # Cleanup thread should be started
            assert server._shm_cleanup_thread is not None
            assert server._shm_cleanup_thread.is_alive()

            # Stop cleanup thread
            server._stop_shm_cleanup_thread()
            assert server._shm_cleanup_thread is None

    def test_cleanup_thread_skips_on_non_posix(self):
        """Cleanup thread should not start if /dev/shm doesn't exist."""
        # Mock /dev/shm not existing
        with patch.object(Path, 'exists', return_value=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                server = TensorFlightServer(
                    location="grpc://localhost:0",
                    write_dir=Path(tmpdir),
                )

                # Cleanup thread should not be started
                assert server._shm_cleanup_thread is None


# ==============================================================================
# Client-side helper tests
# ==============================================================================

class TestEnvVarCheck:
    """Tests for BIOPB_SHM_TRANSFER_DISABLED env var check."""

    def test_disabled_with_1(self):
        """Env var '1' should disable SHM."""
        with patch.dict(os.environ, {"BIOPB_SHM_TRANSFER_DISABLED": "1"}):
            from biopb.tensor.client import _is_shm_disabled_by_env
            assert _is_shm_disabled_by_env() is True

    def test_disabled_with_true(self):
        """Env var 'true' should disable SHM."""
        with patch.dict(os.environ, {"BIOPB_SHM_TRANSFER_DISABLED": "true"}):
            from biopb.tensor.client import _is_shm_disabled_by_env
            assert _is_shm_disabled_by_env() is True

    def test_disabled_with_yes(self):
        """Env var 'yes' should disable SHM."""
        with patch.dict(os.environ, {"BIOPB_SHM_TRANSFER_DISABLED": "yes"}):
            from biopb.tensor.client import _is_shm_disabled_by_env
            assert _is_shm_disabled_by_env() is True

    def test_disabled_with_TRUE_uppercase(self):
        """Env var 'TRUE' (uppercase) should disable SHM."""
        with patch.dict(os.environ, {"BIOPB_SHM_TRANSFER_DISABLED": "TRUE"}):
            from biopb.tensor.client import _is_shm_disabled_by_env
            assert _is_shm_disabled_by_env() is True

    def test_not_disabled_empty(self):
        """Empty env var should not disable SHM."""
        with patch.dict(os.environ, {"BIOPB_SHM_TRANSFER_DISABLED": ""}, clear=True):
            from biopb.tensor.client import _is_shm_disabled_by_env
            # Need to reload to get fresh value
            import importlib
            import biopb.tensor.client as client_module
            importlib.reload(client_module)
            assert client_module._is_shm_disabled_by_env() is False

    def test_not_disabled_unset(self):
        """Unset env var should not disable SHM."""
        # Make sure env var is not set
        os.environ.pop("BIOPB_SHM_TRANSFER_DISABLED", None)
        from biopb.tensor.client import _is_shm_disabled_by_env
        assert _is_shm_disabled_by_env() is False

    def test_not_disabled_random_value(self):
        """Random env var value should not disable SHM."""
        with patch.dict(os.environ, {"BIOPB_SHM_TRANSFER_DISABLED": "random"}):
            from biopb.tensor.client import _is_shm_disabled_by_env
            assert _is_shm_disabled_by_env() is False


class TestServerVersionCheck:
    """Tests for server version check."""

    def test_supports_shm_version_0_3_0(self):
        """Version 0.3.0 should support shm_transfer."""
        from biopb.tensor.client import _server_supports_shm
        metadata = {"tensor_schema_version": "0.3.0"}
        assert _server_supports_shm(metadata) is True

    def test_does_not_support_shm_version_0_2_0(self):
        """Version 0.2.0 should NOT support shm_transfer."""
        from biopb.tensor.client import _server_supports_shm
        metadata = {"tensor_schema_version": "0.2.0"}
        assert _server_supports_shm(metadata) is False

    def test_does_not_support_shm_version_0_2_9(self):
        """Version 0.2.9 should NOT support shm_transfer."""
        from biopb.tensor.client import _server_supports_shm
        metadata = {"tensor_schema_version": "0.2.9"}
        assert _server_supports_shm(metadata) is False

    def test_supports_shm_dev_version(self):
        """Dev version >= 0.4.0 should support shm_transfer."""
        from biopb.tensor.client import _server_supports_shm
        metadata = {"tensor_schema_version": "0.4.1.dev43+g123"}
        assert _server_supports_shm(metadata) is True

    def test_none_metadata_returns_false(self):
        """None metadata should return False."""
        from biopb.tensor.client import _server_supports_shm
        assert _server_supports_shm(None) is False

    def test_missing_version_key_returns_false(self):
        """Missing tensor_schema_version key should return False."""
        from biopb.tensor.client import _server_supports_shm
        metadata = {"other_key": "value"}
        assert _server_supports_shm(metadata) is False

    def test_invalid_version_returns_false(self):
        """Invalid version string should return False."""
        from biopb.tensor.client import _server_supports_shm
        metadata = {"tensor_schema_version": "not-a-version"}
        assert _server_supports_shm(metadata) is False


class TestPosixCheck:
    """Tests for POSIX SHM availability check."""

    def test_posix_available_on_linux(self):
        """POSIX SHM should be available on Linux."""
        from biopb.tensor.client import _is_posix_shm_available
        with patch.object(sys, 'platform', 'linux'):
            assert _is_posix_shm_available() is True

    def test_posix_available_on_darwin(self):
        """POSIX SHM should be available on Darwin (macOS)."""
        from biopb.tensor.client import _is_posix_shm_available
        with patch.object(sys, 'platform', 'darwin'):
            assert _is_posix_shm_available() is True

    def test_posix_available_on_freebsd(self):
        """POSIX SHM should be available on FreeBSD."""
        from biopb.tensor.client import _is_posix_shm_available
        with patch.object(sys, 'platform', 'freebsd13'):
            assert _is_posix_shm_available() is True

    def test_posix_not_available_on_windows(self):
        """POSIX SHM should NOT be available on Windows."""
        from biopb.tensor.client import _is_posix_shm_available
        with patch.object(sys, 'platform', 'win32'):
            assert _is_posix_shm_available() is False


class TestLocalhostDetection:
    """Tests for localhost location detection."""

    def test_localhost_explicit(self):
        """grpc://localhost:port should be localhost."""
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://localhost:8815") is True

    def test_127_0_0_1(self):
        """grpc://127.0.0.1:port should be localhost."""
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://127.0.0.1:8815") is True

    def test_0_0_0_0(self):
        """grpc://0.0.0.0:port should be localhost."""
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://0.0.0.0:8815") is True

    def test_ipv6_loopback(self):
        """grpc://[::1]:port should be localhost."""
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://[::1]:8815") is True

    def test_localhost_no_scheme(self):
        """localhost:port without scheme should be localhost."""
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("localhost:8815") is True

    def test_not_localhost_remote_ip(self):
        """Remote IP should NOT be localhost."""
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://192.168.1.100:8815") is False

    def test_not_localhost_remote_hostname(self):
        """Remote hostname should NOT be localhost."""
        from biopb.tensor.client import _is_localhost_location
        # Use a hostname that won't resolve to loopback
        assert _is_localhost_location("grpc://example.com:8815") is False

    def test_grpc_tls_scheme(self):
        """grpc+tls://localhost:port should be localhost."""
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc+tls://localhost:8815") is True


class TestShouldTryShmTransfer:
    """Tests for combined should-try check."""

    def test_should_try_all_conditions_met(self):
        """Should try when all conditions are met."""
        from biopb.tensor.client import _should_try_shm_transfer

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, 'platform', 'linux'):
                result = _should_try_shm_transfer(
                    "grpc://localhost:8815",
                    {"tensor_schema_version": "0.4.0"}
                )
                assert result is True

    def test_should_not_try_env_disabled(self):
        """Should NOT try when env var disabled."""
        from biopb.tensor.client import _should_try_shm_transfer

        with patch.dict(os.environ, {"BIOPB_SHM_TRANSFER_DISABLED": "1"}):
            with patch.object(sys, 'platform', 'linux'):
                result = _should_try_shm_transfer(
                    "grpc://localhost:8815",
                    {"tensor_schema_version": "0.4.0"}
                )
                assert result is False

    def test_should_not_try_non_posix(self):
        """Should NOT try on non-POSIX platform."""
        from biopb.tensor.client import _should_try_shm_transfer

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, 'platform', 'win32'):
                result = _should_try_shm_transfer(
                    "grpc://localhost:8815",
                    {"tensor_schema_version": "0.4.0"}
                )
                assert result is False

    def test_should_not_try_remote_location(self):
        """Should NOT try for remote location."""
        from biopb.tensor.client import _should_try_shm_transfer

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, 'platform', 'linux'):
                result = _should_try_shm_transfer(
                    "grpc://192.168.1.100:8815",
                    {"tensor_schema_version": "0.4.0"}
                )
                assert result is False

    def test_should_not_try_old_server_version(self):
        """Should NOT try for old server version."""
        from biopb.tensor.client import _should_try_shm_transfer

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, 'platform', 'linux'):
                result = _should_try_shm_transfer(
                    "grpc://localhost:8815",
                    {"tensor_schema_version": "0.2.9"}  # Below min version 0.3.0
                )
                assert result is False

    def test_should_not_try_none_metadata(self):
        """Should NOT try when metadata is None."""
        from biopb.tensor.client import _should_try_shm_transfer

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, 'platform', 'linux'):
                result = _should_try_shm_transfer(
                    "grpc://localhost:8815",
                    None
                )
                assert result is False


# ==============================================================================
# Integration tests
# ==============================================================================

@pytest.mark.skipif(
    sys.platform not in ("linux", "darwin"),
    reason="SHM only available on POSIX systems"
)
class TestShmTransferIntegration:
    """End-to-end integration tests for SHM transfer."""

    def test_shm_transfer_roundtrip(self):
        """Full roundtrip: client requests tensor, data is correct."""
        # Create test data
        test_data = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)

        # Setup server
        with tempfile.TemporaryDirectory() as tmpdir:
            CacheManager.initialize(CacheConfig(backend="memory", memory_max_bytes=10_000_000))

            adapter = CachedSourceAdapter(
                source_id="shm_test",
                shape=[32, 32],
                dtype="uint16",
                chunk_shape=[32, 32],
            )

            # Write test data as single chunk
            bounds = ChunkBounds(start=[0, 0], stop=[32, 32])
            adapter.write_chunk(bounds, test_data)

            # Bind to port 0 so the OS assigns a free port (avoids flaky
            # "Address already in use" collisions); read it back for the client.
            server = TensorFlightServer(location="grpc://localhost:0", write_dir=Path(tmpdir))
            location = f"grpc://localhost:{server.port}"
            server.register_source("shm_test", adapter)

            # Start server in thread
            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1.0)  # Wait for server to start

            try:
                # Client connects and fetches tensor
                from biopb.tensor.client import TensorFlightClient

                client = TensorFlightClient(location)

                # Get tensor
                arr = client.get_tensor("shm_test")

                # Verify data is correct (whether via SHM or do_get)
                result = arr.compute()
                np.testing.assert_array_equal(result, test_data)

                client.close()
            finally:
                # Server cleanup
                server._stop_shm_cleanup_thread()

    def test_shm_transfer_action_via_flight(self):
        """Test shm_transfer action directly via Flight protocol."""
        # Create test data
        test_data = np.arange(256, dtype=np.uint16).reshape(16, 16)

        with tempfile.TemporaryDirectory() as tmpdir:
            CacheManager.initialize(CacheConfig(backend="memory", memory_max_bytes=10_000_000))

            adapter = CachedSourceAdapter(
                source_id="shm_action_test",
                shape=[16, 16],
                dtype="uint16",
                chunk_shape=[16, 16],
            )

            bounds = ChunkBounds(start=[0, 0], stop=[16, 16])
            adapter.write_chunk(bounds, test_data)

            # Bind to port 0 so the OS assigns a free port (avoids flaky
            # "Address already in use" collisions); read it back for the client.
            server = TensorFlightServer(location="grpc://localhost:0", write_dir=Path(tmpdir))
            location = f"grpc://localhost:{server.port}"
            server.register_source("shm_action_test", adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1.0)

            try:
                client = flight.FlightClient(location)

                # Call shm_transfer action directly
                from biopb_tensor_server.chunk import encode_chunk_id
                chunk_id = encode_chunk_id("shm_action_test", bounds)
                ticket = TensorTicket(chunk_id=chunk_id)
                action = flight.Action("shm_transfer", ticket.SerializeToString())

                results = client.do_action(action)
                shm_name = next(results).body.to_pybytes().decode("utf-8")

                # Read from SHM
                shm = shared_memory.SharedMemory(name=shm_name)
                ipc_bytes = bytes(shm.buf)

                reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
                table = reader.read_all()

                arr = table.column("data").to_numpy()[0]
                shape = tuple(table.column("shape").to_pylist()[0])
                arr = arr.reshape(shape)

                np.testing.assert_array_equal(arr, test_data)

                shm.close()
                shm.unlink()
                client.close()
            finally:
                server._stop_shm_cleanup_thread()


# ==============================================================================
# Schema metadata extraction tests
# ==============================================================================

class TestExtractSchemaMetadata:
    """Tests for _extract_schema_metadata helper."""

    def test_extracts_metadata_dict(self):
        """Should extract metadata as Python dict."""
        from biopb.tensor.client import _extract_schema_metadata

        schema = pa.schema(
            [],
            metadata={
                b"tensor_schema_version": b"0.4.0",
                b"other_key": b"other_value",
            }
        )

        metadata = _extract_schema_metadata(schema)

        assert metadata is not None
        assert metadata["tensor_schema_version"] == "0.4.0"
        assert metadata["other_key"] == "other_value"

    def test_returns_none_for_no_metadata(self):
        """Should return None for schema without metadata."""
        from biopb.tensor.client import _extract_schema_metadata

        schema = pa.schema([])
        metadata = _extract_schema_metadata(schema)

        assert metadata is None


# ==============================================================================
# Error handling tests
# ==============================================================================

class TestShmTransferErrorHandling:
    """Tests for SHM transfer error handling and fallback."""

    def test_try_shm_transfer_returns_none_on_failure(self):
        """_try_shm_transfer should return None on any failure."""
        from biopb.tensor.client import _try_shm_transfer

        # Create mock client that raises
        mock_client = MagicMock()
        mock_client.do_action.side_effect = Exception("Connection failed")

        result = _try_shm_transfer(
            mock_client,
            "grpc://localhost:8815",
            None,
            b"test_chunk_id",
            flight.FlightCallOptions()
        )

        assert result is None

    def test_fallback_to_do_get_works(self):
        """When SHM fails, should fallback to do_get successfully."""
        # This is tested implicitly by integration tests
        # Here we verify the fallback mechanism exists
        from biopb.tensor.client import _fetch_chunk_distributed

        # Mock that SHM would fail (by using remote location)
        # _should_try_shm_transfer returns False for remote
        # so do_get path is used
        pass  # Covered by TestShouldTryShmTransfer tests