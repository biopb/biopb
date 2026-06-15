"""Tests for the localhost cache-file mmap fast path (biopb/biopb#9).

Replaces the retired /dev/shm shm_transfer path. Covers:
- Backend: ArrowFileBackend.locate_entry byte offsets, round-trip, restart,
  eviction, and the memory-backend "unavailable" case.
- Server: chunk_locate action listing + JSON contract.
- Client: localhost detection, schema-metadata extraction, _should_try_cachefile
  gating, and _array_from_unified_batch decode.
- Integration: file-backed server <-> client round-trip equals do_get, and the
  do_get fallback when the fast path is disabled or unavailable.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest

from biopb.tensor.ticket_pb2 import TensorTicket

from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.file_backend import (
    ArrowFileBackend,
    ArrowFileConfig,
    ChunkLocation,
    _cast_from_unified_schema,
)
from biopb_tensor_server.config import CacheConfig
from biopb_tensor_server.server import TensorFlightServer


def _zarr_available() -> bool:
    try:
        import zarr  # noqa: F401
        return True
    except ImportError:
        return False


# ==============================================================================
# Helpers
# ==============================================================================

def _make_typed_batch(arr: np.ndarray) -> pa.RecordBatch:
    """Build a batch in the [data: list<dtype>, shape, dtype] schema the cache
    expects from a compute_fn (mirrors BackendAdapter.resolve_chunk_data)."""
    data = pa.array([arr.ravel().tolist()], type=pa.list_(pa.from_numpy_dtype(arr.dtype)))
    shape = pa.array([list(arr.shape)], type=pa.list_(pa.int64()))
    dtype = pa.array([str(arr.dtype)], type=pa.string())
    return pa.RecordBatch.from_arrays([data, shape, dtype], ["data", "shape", "dtype"])


def _read_via_location(loc: ChunkLocation) -> np.ndarray:
    """Read a chunk back from its on-disk location the way the client does."""
    mm = pa.memory_map(loc.segment_path, "r")
    try:
        schema = pa.ipc.open_stream(mm).schema
        mm.seek(loc.byte_offset)
        batch = pa.ipc.read_record_batch(pa.ipc.read_message(mm), schema)
        typed = _cast_from_unified_schema(batch)
        data = np.asarray(typed.column("data").to_numpy(zero_copy_only=False)[0])
        return data.reshape(tuple(typed.column("shape").to_pylist()[0]))
    finally:
        mm.close()


@pytest.fixture
def file_backend():
    d = tempfile.mkdtemp()
    be = ArrowFileBackend(ArrowFileConfig(
        cache_dir=Path(d),
        max_segment_bytes=8 * 1024 * 1024,
        max_total_bytes=256 * 1024 * 1024,
    ))
    try:
        yield be
    finally:
        be.close()
        shutil.rmtree(d, ignore_errors=True)


# ==============================================================================
# Backend: locate_entry
# ==============================================================================

class TestLocateEntry:
    def test_locate_roundtrip_multiple_chunks(self, file_backend):
        """Each chunk in a shared segment locates to its exact message bytes."""
        arrs = {}
        for i in range(4):
            a = ((np.arange(1500, dtype=np.uint16) + i * 13) % 509).astype(np.uint16)
            arrs[i] = a
            key = f"chunk-{i}".encode()
            file_backend.get_or_acquire(key, (lambda a=a: (_make_typed_batch(a), a.nbytes)))
            file_backend.release(key)

        for i in range(4):
            loc = file_backend.locate_entry(f"chunk-{i}".encode())
            assert isinstance(loc, ChunkLocation)
            # First entry sits after the schema message, never at offset 0.
            assert loc.byte_offset > 0 and loc.byte_length > 0
            assert np.array_equal(_read_via_location(loc), arrs[i])

    def test_locate_preserves_dtype_and_shape(self, file_backend):
        a = np.random.RandomState(0).rand(40, 50).astype(np.float32)
        file_backend.get_or_acquire(b"f32", (lambda: (_make_typed_batch(a), a.nbytes)))
        file_backend.release(b"f32")
        got = _read_via_location(file_backend.locate_entry(b"f32"))
        assert got.dtype == np.float32 and got.shape == (40, 50)
        assert np.array_equal(got, a)

    def test_unknown_key_returns_none(self, file_backend):
        assert file_backend.locate_entry(b"does-not-exist") is None

    def test_locate_survives_restart(self):
        """Offsets are recoverable after a restart (rebuild from segments)."""
        d = tempfile.mkdtemp()
        cfg = dict(max_segment_bytes=8 * 1024 * 1024, max_total_bytes=256 * 1024 * 1024)
        be = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **cfg))
        a = (np.arange(2000, dtype=np.uint16) % 500).astype(np.uint16)
        be.get_or_acquire(b"k0", (lambda: (_make_typed_batch(a), a.nbytes)))
        be.release(b"k0")
        be.close()

        be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **cfg))
        try:
            loc = be2.locate_entry(b"k0")
            assert loc is not None and loc.byte_offset > 0
            assert np.array_equal(_read_via_location(loc), a)
        finally:
            be2.close()
            shutil.rmtree(d, ignore_errors=True)

    def test_generation_id_is_segment_inode(self, file_backend):
        a = (np.arange(100, dtype=np.uint16)).astype(np.uint16)
        file_backend.get_or_acquire(b"g", (lambda: (_make_typed_batch(a), a.nbytes)))
        file_backend.release(b"g")
        loc = file_backend.locate_entry(b"g")
        assert loc.generation_id == os.stat(loc.segment_path).st_ino

    def test_torn_trailing_message_does_not_crash_locate(self):
        """A torn trailing message must not crash the offset walk.

        Regression for the Windows-only CI failure ``OSError: Expected to be
        able to read N bytes for message body`` raised from
        ``_fill_byte_offsets_for_segment``: a prior partial/failed write can
        leave slack at the segment tail, and the lazy offset walk must treat it
        as end-of-region rather than propagate. Reproduced cross-platform by
        appending a truncated copy of a real message to the segment file.
        """
        d = tempfile.mkdtemp()
        cfg = dict(max_segment_bytes=64 * 1024 * 1024,
                   max_total_bytes=256 * 1024 * 1024)
        be = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **cfg))
        try:
            arrs = {}
            for i in range(3):
                a = ((np.arange(1500, dtype=np.uint16) + i) % 509).astype(np.uint16)
                arrs[i] = a
                key = f"good-{i}".encode()
                be.get_or_acquire(key, (lambda a=a: (_make_typed_batch(a), a.nbytes)))
                be.release(key)

            # Capture a real framed message and append a truncated copy: header
            # intact (advertises a full body) but the body cut short, exactly
            # what a partial write leaves behind.
            loc0 = be.locate_entry(b"good-0")
            seg_path = Path(loc0.segment_path)
            raw = seg_path.read_bytes()
            full = raw[loc0.byte_offset:loc0.byte_offset + loc0.byte_length]
            with open(seg_path, "ab") as f:
                f.write(full[: len(full) // 2])

            # Drop cached offsets so locate re-walks the now-torn segment.
            for info in be._metadata.values():
                info.byte_offset = 0
                info.byte_length = 0

            # Must not raise; every good entry ahead of the torn tail resolves.
            for i in range(3):
                loc = be.locate_entry(f"good-{i}".encode())
                assert loc is not None, f"good-{i} should still locate"
                assert np.array_equal(_read_via_location(loc), arrs[i])
        finally:
            be.close()
            shutil.rmtree(d, ignore_errors=True)


class TestLocateViaManager:
    def test_memory_backend_returns_none(self):
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        try:
            mgr = CacheManager.get_instance()
            assert mgr.locate_entry(b"anything") is None
        finally:
            CacheManager.reset()


# ==============================================================================
# Server: chunk_locate action
# ==============================================================================

class TestChunkLocateAction:
    def test_chunk_locate_listed(self):
        server = TensorFlightServer("grpc://localhost:0")
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(0.8)
        try:
            client = flight.FlightClient(f"grpc://localhost:{server.port}")
            actions = {a.type for a in client.list_actions()}
            assert "chunk_locate" in actions
            assert "shm_transfer" not in actions
        finally:
            server.shutdown()


# ==============================================================================
# Client helpers
# ==============================================================================

class TestLocalhostDetection:
    """Localhost detection (still used to gate the cache-file fast path)."""

    def test_localhost_explicit(self):
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://localhost:8815") is True

    def test_127_0_0_1(self):
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://127.0.0.1:8815") is True

    def test_ipv6_loopback(self):
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://[::1]:8815") is True

    def test_grpc_tls_scheme(self):
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc+tls://localhost:8815") is True

    def test_not_localhost_remote_ip(self):
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://192.168.1.100:8815") is False

    def test_not_localhost_remote_hostname(self):
        from biopb.tensor.client import _is_localhost_location
        assert _is_localhost_location("grpc://example.com:8815") is False


class TestExtractSchemaMetadata:
    def test_extracts_metadata_dict(self):
        from biopb.tensor.client import _extract_schema_metadata
        schema = pa.schema([], metadata={
            b"tensor_schema_version": b"0.4.0",
            b"other_key": b"other_value",
        })
        metadata = _extract_schema_metadata(schema)
        assert metadata is not None
        assert metadata["tensor_schema_version"] == "0.4.0"
        assert metadata["other_key"] == "other_value"

    def test_returns_none_for_no_metadata(self):
        from biopb.tensor.client import _extract_schema_metadata
        assert _extract_schema_metadata(pa.schema([])) is None


class TestShouldTryCachefile:
    def setup_method(self):
        import biopb.tensor.client as c
        c._cachefile_support.clear()

    def test_enabled_on_posix_localhost(self):
        from biopb.tensor.client import _should_try_cachefile
        if os.name != "posix":
            pytest.skip("POSIX-only path")
        assert _should_try_cachefile("grpc://localhost:8815") is True

    def test_disabled_by_env(self):
        from biopb.tensor.client import _should_try_cachefile
        with patch.dict(os.environ, {"BIOPB_CACHEFILE_TRANSFER_DISABLED": "1"}):
            assert _should_try_cachefile("grpc://localhost:8815") is False

    def test_disabled_for_remote(self):
        from biopb.tensor.client import _should_try_cachefile
        assert _should_try_cachefile("grpc://192.168.1.100:8815") is False

    def test_disabled_on_non_posix(self):
        import biopb.tensor.client as c
        with patch.object(os, "name", "nt"):
            assert c._should_try_cachefile("grpc://localhost:8815") is False

    def test_skips_after_unsupported_memoized(self):
        import biopb.tensor.client as c
        c._set_cachefile_supported("grpc://localhost:8815", False)
        assert c._should_try_cachefile("grpc://localhost:8815") is False


class TestArrayFromUnifiedBatch:
    def test_decode_matches_source(self):
        from biopb.tensor.client import _array_from_unified_batch
        from biopb_tensor_server.cache.file_backend import _cast_to_unified_schema
        a = (np.arange(120, dtype=np.uint16).reshape(8, 15) % 97).astype(np.uint16)
        unified = _cast_to_unified_schema(_make_typed_batch(a))
        got = _array_from_unified_batch(unified)
        assert got.dtype == np.uint16 and got.shape == (8, 15)
        assert np.array_equal(got, a)


# ==============================================================================
# Integration: file-backed server <-> client
# ==============================================================================

@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestCachefileIntegration:
    def _serve_zarr(self, tmp, backend_cfg):
        import zarr
        from biopb_tensor_server import ZarrAdapter

        CacheManager.reset()
        CacheManager.initialize(backend_cfg)
        zpath = str(Path(tmp) / "data.zarr")
        src = (np.arange(96 * 96, dtype=np.uint16).reshape(96, 96) % 997).astype(np.uint16)
        z = zarr.open_array(zpath, mode="w", shape=src.shape, chunks=(48, 48), dtype=src.dtype)
        z[:] = src
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("z", ZarrAdapter(zarr.open_array(zpath, mode="r"), "z", ["y", "x"]))
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(0.8)
        return server, src

    def test_cachefile_path_matches_source_and_is_exercised(self):
        import biopb.tensor.client as cmod
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            cmod._cachefile_support.clear()
            client = TensorFlightClient(loc, cache_bytes=0)
            got = client.get_tensor("z", "z").compute(scheduler="threads")
            assert np.array_equal(got, src)
            if os.name == "posix":
                # The fast path was actually used (chunk_locate succeeded).
                assert cmod._cachefile_support.get(loc) is True
            else:
                # Windows: the cache-file path is gated off (biopb/biopb#5),
                # so do_get is used and the probe cache stays untouched.
                assert cmod._cachefile_support.get(loc) is None
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_do_get_fallback_when_disabled(self):
        import biopb.tensor.client as cmod
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            cmod._cachefile_support.clear()
            with patch.dict(os.environ, {"BIOPB_CACHEFILE_TRANSFER_DISABLED": "1"}):
                client = TensorFlightClient(loc, cache_bytes=0)
                got = client.get_tensor("z", "z").compute(scheduler="threads")
                assert np.array_equal(got, src)
                # Never probed the fast path.
                assert cmod._cachefile_support.get(loc) is None
                client.close()
        finally:
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_memory_backend_falls_back_but_data_correct(self):
        import biopb.tensor.client as cmod
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        server, src = self._serve_zarr(tmp, CacheConfig(backend="memory"))
        loc = f"grpc://localhost:{server.port}"
        try:
            cmod._cachefile_support.clear()
            client = TensorFlightClient(loc, cache_bytes=0)
            got = client.get_tensor("z", "z").compute(scheduler="threads")
            assert np.array_equal(got, src)
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.skipif(os.name != "posix", reason="cache-file fast path is POSIX-only")
    def test_newer_segment_format_falls_back(self):
        """A server segment format newer than the client understands declines
        the fast path (and is memoized off), but data is still correct via do_get."""
        import biopb.tensor.client as cmod
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            cmod._cachefile_support.clear()
            # Pretend this client can't parse the server's (>=1) segment format.
            with patch.object(cmod, "_CACHEFILE_SUPPORTED_FORMAT", 0):
                client = TensorFlightClient(loc, cache_bytes=0)
                got = client.get_tensor("z", "z").compute(scheduler="threads")
                assert np.array_equal(got, src)  # correct via do_get fallback
                assert cmod._cachefile_support.get(loc) is False  # declined + memoized
                client.close()
        finally:
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)
