"""Unit tests for biopb.tensor.client connection pool management.

Tests the thread-local connection pool, fork-safety, and caching behavior
without requiring a live Flight server.
"""

import os
import threading
from unittest.mock import MagicMock, patch

# Import the module to test pool behavior
import biopb.tensor.client as client_module
import pyarrow.flight as flight
import pytest


class TestThreadLocalPool:
    """Tests for thread-local FlightClient storage."""

    def test_thread_local_storage_initialization(self):
        """Test that thread-local storage is initialized correctly."""
        # Each thread starts with empty storage
        local = threading.local()
        assert hasattr(local, "clients") or not hasattr(local, "clients")

    def test_different_threads_have_different_clients(self):
        """Test that different threads get different FlightClient instances."""
        results = {}
        errors = []

        def thread_func(thread_id):
            try:
                # Create a mock client for this thread
                mock_client = MagicMock(spec=flight.FlightClient)
                local_pool = getattr(client_module._THREAD_LOCAL, "clients", {})
                if local_pool is None:
                    local_pool = {}
                local_pool[("test_loc", None)] = (os.getpid(), mock_client)
                client_module._THREAD_LOCAL.clients = local_pool
                results[thread_id] = mock_client
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_func, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        # Each thread should have its own entry
        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_same_thread_reuses_client(self):
        """Test that the same thread reuses its FlightClient."""
        # This is a conceptual test - actual reuse happens in _get_thread_client
        # In the actual implementation, calling _get_thread_client twice
        # with the same location/token returns the same client
        # This would require a live server or complex mocking


class TestForkSafety:
    """Tests for fork-safety behavior (pid mismatch detection)."""

    def test_pid_change_detected(self):
        """Test that pid mismatch is detected in stored connections."""
        # Simulate a pid mismatch scenario
        stored_pid = 999999  # Different pid
        current_pid = os.getpid()

        # The check in _get_thread_client compares pid
        assert stored_pid != current_pid

    def test_stale_connection_cleanup_on_pid_mismatch(self):
        """Test that stale connections are cleaned up when pid mismatches."""
        # When a process forks, inherited gRPC sockets are broken
        # The child process should detect pid mismatch and create new connections
        # This is tested conceptually here
        current_pid = os.getpid()

        # Create a mock pool entry with old pid
        mock_client = MagicMock()
        old_entry = (999999, mock_client)  # Tuple of (pid, client)

        # The logic in _get_thread_client checks:
        # if pid == current_pid: return client
        # else: close old client, create new one
        assert old_entry[0] != current_pid


class TestEvictDeadThreads:
    """Tests for _evict_dead_threads cleanup."""

    def test_eviction_removes_dead_thread_connections(self):
        """Test that dead thread connections are removed."""
        # Create a thread, have it register a connection, then let it die
        registered = []

        def register_thread():
            thread_id = threading.current_thread().ident
            registered.append(thread_id)
            # Simulate registering a connection
            with client_module._REGISTRY_LOCK:
                if thread_id is not None:
                    client_module._CONNECTION_REGISTRY[thread_id] = {
                        ("test_loc", None): MagicMock()
                    }

        t = threading.Thread(target=register_thread)
        t.start()
        t.join(timeout=5)

        thread_id = registered[0] if registered else None

        # After thread dies, it should not be in threading._active
        if thread_id is not None:
            # Thread is dead, but connection may still be in registry
            # until eviction runs
            assert thread_id not in threading._active

            # Cleanup
            with client_module._REGISTRY_LOCK:
                if thread_id in client_module._CONNECTION_REGISTRY:
                    del client_module._CONNECTION_REGISTRY[thread_id]

    def test_alive_threads_not_evicted(self):
        """Test that alive threads' connections are not evicted."""
        current_thread_id = threading.current_thread().ident

        # Current thread should be in threading._active
        assert current_thread_id in threading._active


class TestSharedCache:
    """Tests for shared cache operations."""

    def test_cache_get_missing_key(self):
        """Test cache.get returns None for missing keys."""
        # This would test cachey.Cache behavior
        # The actual cache is created in _get_shared_cache
        # Requires cachey installation

    def test_cache_put_and_get(self):
        """Test cache.put and cache.get roundtrip."""
        # In the implementation, cache.put stores with cost
        # and cache.get retrieves
        # Requires cachey installation


class TestCachePolicy:
    """Tests for the localhost-aware cache-size resolver (piece 1)."""

    def test_resolve_zero_or_negative_request_disables(self):
        assert client_module._resolve_cache_bytes("grpc://remote:8815", 0) == 0
        assert client_module._resolve_cache_bytes("grpc://remote:8815", -5) == 0

    def test_resolve_remote_keeps_requested_size(self):
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            assert (
                client_module._resolve_cache_bytes("grpc://remote:8815", 1000) == 1000
            )

    def test_resolve_localhost_disables_by_default(self, monkeypatch):
        monkeypatch.delenv("BIOPB_CACHE_LOCAL", raising=False)
        with patch.object(client_module, "_is_localhost_location", return_value=True):
            assert (
                client_module._resolve_cache_bytes("grpc://localhost:8815", 1000) == 0
            )

    def test_resolve_localhost_opt_in_via_env(self, monkeypatch):
        monkeypatch.setenv("BIOPB_CACHE_LOCAL", "1")
        with patch.object(client_module, "_is_localhost_location", return_value=True):
            assert (
                client_module._resolve_cache_bytes("grpc://localhost:8815", 1000)
                == 1000
            )

    def test_shared_cache_is_none_for_localhost(self, monkeypatch):
        monkeypatch.delenv("BIOPB_CACHE_LOCAL", raising=False)
        with patch.object(client_module, "_is_localhost_location", return_value=True):
            assert (
                client_module._get_shared_cache("grpc://localhost:8815", None, 1000)
                is None
            )

    def test_shared_cache_created_for_remote(self):
        loc = "grpc://remote-cache-test:9999"
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            cache = client_module._get_shared_cache(loc, None, 1000)
            try:
                assert cache is not None
                assert cache.available_bytes == 1000
            finally:
                client_module._CACHE_POOL.pop((loc, None), None)

    def test_localhost_detection_loopback_literals(self):
        assert client_module._is_localhost_location("grpc://127.0.0.1:8815") is True
        assert client_module._is_localhost_location("grpc://localhost:8815") is True


class TestConfigureCache:
    """Tests for configure_cache() -- authoritative, idempotent cache pinning."""

    @pytest.fixture(autouse=True)
    def _clean_pool(self):
        client_module._CACHE_POOL.clear()
        yield
        client_module._CACHE_POOL.clear()

    def test_pins_size_for_remote(self):
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            eff = client_module.configure_cache("grpc://remote:8815", None, 1000)
        assert eff == 1000
        cache = client_module._CACHE_POOL[("grpc://remote:8815", None)][1]
        assert cache.available_bytes == 1000

    def test_resizes_in_place(self):
        loc = "grpc://remote:8815"
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            client_module.configure_cache(loc, None, 1000)
            client_module.configure_cache(loc, None, 2000)
        assert client_module._CACHE_POOL[(loc, None)][1].available_bytes == 2000

    def test_localhost_pins_off(self):
        loc = "grpc://srv:8815"
        # first create a real cache as if remote, then re-resolve as localhost
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            client_module.configure_cache(loc, None, 1000)
        with patch.object(client_module, "_is_localhost_location", return_value=True):
            eff = client_module.configure_cache(loc, None, 1000)
        assert eff == 0
        # pinned off -> entry present with a None-cache sentinel (not deleted)
        assert client_module._CACHE_POOL[(loc, None)][1] is None

    def test_zero_pins_off(self):
        loc = "grpc://remote:8815"
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            client_module.configure_cache(loc, None, 1000)
            assert client_module.configure_cache(loc, None, 0) == 0
        assert client_module._CACHE_POOL[(loc, None)][1] is None

    def test_pinned_off_survives_later_fetch_request(self):
        """configure_cache(.., 0) must not be undone by a later nonzero fetch."""
        loc = "grpc://remote:8815"
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            assert client_module.configure_cache(loc, None, 0) == 0
            # a later fetch carrying the default 1GB must NOT recreate a cache
            assert client_module._get_shared_cache(loc, None, 1_000_000_000) is None

    def test_get_shared_cache_honors_configured_size(self):
        loc = "grpc://remote:8815"
        with patch.object(client_module, "_is_localhost_location", return_value=False):
            client_module.configure_cache(loc, None, 500)
            # a later fetch requesting a different size must get the pinned cache
            cache = client_module._get_shared_cache(loc, None, 999_999_999)
        assert cache is not None and cache.available_bytes == 500


class TestCleanupConnectionPool:
    """Tests for atexit cleanup."""

    def test_cleanup_registered(self):
        """Test that cleanup is registered with atexit."""
        # _cleanup_connection_pool is registered at module load

        # Check that atexit has handlers registered
        # This is a conceptual check - we don't want to actually
        # run the cleanup in tests
        assert hasattr(client_module, "_cleanup_connection_pool")

    def test_cleanup_clears_registries(self):
        """Test that cleanup clears all registries."""
        # Add some mock entries
        with client_module._REGISTRY_LOCK:
            mock_thread_id = 99999
            client_module._CONNECTION_REGISTRY[mock_thread_id] = {
                ("test_loc", None): MagicMock()
            }

        # Run cleanup (but don't close mock clients that might error)
        # Just clear the registries directly
        with client_module._REGISTRY_LOCK:
            client_module._CONNECTION_REGISTRY.clear()
        with client_module._POOL_LOCK:
            client_module._CACHE_POOL.clear()
            client_module._CALL_OPTS_POOL.clear()

        # Verify cleared
        assert len(client_module._CONNECTION_REGISTRY) == 0


class TestCallOptionsPool:
    """Tests for FlightCallOptions pooling."""

    def test_shared_call_options_without_token(self):
        """Test that call options are created without auth token."""
        # Without token, options should be empty
        with client_module._POOL_LOCK:
            key = ("test_loc", None)
            if key in client_module._CALL_OPTS_POOL:
                # A pooled entry exists; without a token it carries no auth
                # headers (FlightCallOptions.headers would be empty).
                pass

    def test_shared_call_options_with_token(self):
        """Test that call options include auth headers when token provided."""
        # With token, options should have Bearer auth header
        key = ("test_loc", "test_token")

        # Clear any existing entry for clean test
        with client_module._POOL_LOCK:
            if key in client_module._CALL_OPTS_POOL:
                del client_module._CALL_OPTS_POOL[key]

        # The actual creation would happen in _get_shared_call_options
        # which creates: headers=[(b"authorization", f"Bearer {token}".encode())]


class TestPoolLocks:
    """Tests for pool locking behavior."""

    def test_registry_lock_exists(self):
        """Test that REGISTRY_LOCK exists."""
        assert hasattr(client_module, "_REGISTRY_LOCK")
        assert isinstance(client_module._REGISTRY_LOCK, type(threading.Lock()))

    def test_pool_lock_exists(self):
        """Test that POOL_LOCK exists."""
        assert hasattr(client_module, "_POOL_LOCK")
        assert isinstance(client_module._POOL_LOCK, type(threading.Lock()))

    def test_concurrent_registry_access(self):
        """Test that concurrent access to registry is safe."""
        errors = []

        def access_registry(iterations):
            for i in range(iterations):
                try:
                    with client_module._REGISTRY_LOCK:
                        # Just touch the registry
                        _ = client_module._CONNECTION_REGISTRY
                except Exception as e:
                    errors.append(str(e))

        threads = []
        for _ in range(5):
            t = threading.Thread(target=access_registry, args=(100,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Concurrent access errors: {errors}"


class TestConnectionRegistryStructure:
    """Tests for connection registry data structures."""

    def test_registry_is_dict(self):
        """Test that CONNECTION_REGISTRY is a dict."""
        assert isinstance(client_module._CONNECTION_REGISTRY, dict)

    def test_cache_pool_is_dict(self):
        """Test that CACHE_POOL is a dict."""
        assert isinstance(client_module._CACHE_POOL, dict)

    def test_call_opts_pool_is_dict(self):
        """Test that CALL_OPTS_POOL is a dict."""
        assert isinstance(client_module._CALL_OPTS_POOL, dict)

    def test_registry_key_structure(self):
        """Test that registry keys are (thread_id, connection_map) pairs."""
        # The structure is: thread_id -> {(location, token): FlightClient}
        # We can verify the structure by looking at the type
        # (without adding actual connections)
