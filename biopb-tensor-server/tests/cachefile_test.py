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

import dataclasses
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
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.file_backend import (
    ArrowFileBackend,
    ArrowFileConfig,
    ChunkLocation,
)
from biopb_tensor_server.core.base import pack_chunk_batch, unpack_chunk_array
from biopb_tensor_server.core.config import CacheConfig
from biopb_tensor_server.serving.server import TensorFlightServer


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
    """Build a chunk batch in the unified binary [data: binary, shape, dtype]
    schema a compute_fn now produces (mirrors TensorAdapter.resolve_chunk_data)."""
    return pack_chunk_batch(arr)


def _read_via_location(loc: ChunkLocation) -> np.ndarray:
    """Read a chunk back from its on-disk location the way the client does."""
    mm = pa.memory_map(loc.segment_path, "r")
    try:
        schema = pa.ipc.open_stream(mm).schema
        mm.seek(loc.byte_offset)
        batch = pa.ipc.read_record_batch(pa.ipc.read_message(mm), schema)
        return unpack_chunk_array(batch)
    finally:
        mm.close()


@pytest.fixture
def file_backend():
    d = tempfile.mkdtemp()
    be = ArrowFileBackend(
        ArrowFileConfig(
            cache_dir=Path(d),
            max_segment_bytes=8 * 1024 * 1024,
            max_total_bytes=256 * 1024 * 1024,
        )
    )
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
            file_backend.get_or_acquire(
                key, (lambda a=a: (_make_typed_batch(a), a.nbytes))
            )
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

    def test_offsets_captured_at_write_time(self, file_backend):
        """The write path indexes every entry, including a segment's first.

        Regression for biopb/biopb#541: ranges used to be derived lazily, so
        every cache *miss* left an unindexed entry and the next locate re-walked
        the whole active segment -- a cost paid only by the localhost fast path
        that walk was meant to accelerate. With the walk gone, an unindexed
        entry no longer locates at all, so `locate_entry` returning None is the
        regression signal.
        """
        for i in range(3):
            a = ((np.arange(1200, dtype=np.uint16) + i * 7) % 401).astype(np.uint16)
            key = f"w{i}".encode()
            file_backend.get_or_acquire(
                key, (lambda a=a: (_make_typed_batch(a), a.nbytes))
            )
            file_backend.release(key)
            info = file_backend._metadata[key]
            assert info.byte_offset > 0 and info.byte_length > 0, (
                f"entry {i} left unindexed by the write path"
            )
            assert file_backend.locate_entry(key) is not None

    def test_underivable_schema_length_degrades_to_do_get(self):
        """If the schema length can't be read, only that one entry loses mmap.

        With no lazy walk behind it, `_schema_message_length` returning None is
        the sole way an entry can end up without a byte range. The blast radius
        must stay at the segment's *first* entry (the only append that shares
        its bracket with the schema message): it reports unavailable so the
        client falls back to do_get, while later entries locate normally, the
        data is still readable, and a restart repairs it from the segment body.
        """
        d = tempfile.mkdtemp()
        cfg = {
            "max_segment_bytes": 64 * 1024 * 1024,
            "max_total_bytes": 256 * 1024 * 1024,
        }
        be = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **cfg))
        arrs = {}
        try:
            with patch(
                "biopb_tensor_server.cache.file_backend._schema_message_length",
                return_value=None,
            ):
                for i in range(3):
                    a = ((np.arange(1100, dtype=np.uint16) + i) % 397).astype(np.uint16)
                    key = f"d{i}".encode()
                    arrs[key] = a
                    be.get_or_acquire(
                        key, (lambda a=a: (_make_typed_batch(a), a.nbytes))
                    )
                    be.release(key)

            # Only the first entry is un-ranged, and it degrades to "unavailable"
            # (-> do_get) rather than to a bogus location.
            assert be.locate_entry(b"d0") is None
            for i in (1, 2):
                loc = be.locate_entry(f"d{i}".encode())
                assert loc is not None
                assert np.array_equal(_read_via_location(loc), arrs[f"d{i}".encode()])

            # The chunk itself is intact -- it is the do_get path that serves it.
            entry = be.get_or_acquire(
                b"d0", (lambda: pytest.fail("d0 should still be cached"))
            )
            assert np.array_equal(unpack_chunk_array(entry.data), arrs[b"d0"])
            be.release(b"d0")
        finally:
            be.close()

        # Self-heals: the sidecar/boot walk derives ranges from the segment body,
        # independent of the write-time bracket.
        be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **cfg))
        try:
            loc = be2.locate_entry(b"d0")
            assert loc is not None, "restart should restore the range"
            assert np.array_equal(_read_via_location(loc), arrs[b"d0"])
        finally:
            be2.close()
            shutil.rmtree(d, ignore_errors=True)

    def test_locate_never_takes_the_write_lock(self, file_backend):
        """A stalled write must not block a locate.

        The lazy-fill branch was the one place the read path took
        ``_write_lock``, so a write blocked on a full filesystem also blocked
        every locate behind it. Nothing is derived at locate time now, so a held
        write lock is irrelevant to it (biopb/biopb#541).
        """
        a = (np.arange(600, dtype=np.uint16) % 251).astype(np.uint16)
        file_backend.get_or_acquire(b"L", (lambda: (_make_typed_batch(a), a.nbytes)))
        file_backend.release(b"L")

        located = []
        with file_backend._write_lock:  # stand in for a write stalled on I/O
            t = threading.Thread(
                target=lambda: located.append(file_backend.locate_entry(b"L"))
            )
            t.start()
            t.join(timeout=5)
            assert not t.is_alive(), "locate blocked on the write lock"
        assert located and located[0] is not None

    def test_write_time_offsets_match_segment_walk(self, file_backend):
        """Write-time ranges are byte-identical to what a boot walk derives.

        The sidecar / boot index (`_scan_segment_records`) and the write path
        must agree, or a chunk would locate to one range now and another after a
        restart.
        """
        arrs = {}
        for i in range(4):
            a = ((np.arange(900, dtype=np.uint16) + i * 31) % 307).astype(np.uint16)
            arrs[f"m{i}".encode()] = a
            key = f"m{i}".encode()
            file_backend.get_or_acquire(
                key, (lambda a=a: (_make_typed_batch(a), a.nbytes))
            )
            file_backend.release(key)

        segment_id = file_backend._metadata[b"m0"].segment_id
        seg_file = Path(file_backend.locate_entry(b"m0").segment_path)
        walked = {
            key: (byte_offset, byte_length)
            for key, byte_offset, byte_length, _, _ in file_backend._scan_segment_records(
                seg_file
            )
        }
        for key in arrs:
            info = file_backend._metadata[key]
            assert info.segment_id == segment_id
            assert walked[key] == (info.byte_offset, info.byte_length)

    def test_offsets_captured_across_segment_rotation(self):
        """Every segment's first entry is indexed too, not just later appends.

        The first append in a segment also flushes the writer's buffered schema
        message, so its bracket covers both; the write path has to subtract the
        schema length rather than leave the entry unindexed.
        """
        d = tempfile.mkdtemp()
        be = ArrowFileBackend(
            ArrowFileConfig(
                cache_dir=Path(d),
                max_segment_bytes=64 * 1024,  # rotate every few chunks
                max_total_bytes=256 * 1024 * 1024,
            )
        )
        try:
            arrs = {}
            for i in range(12):
                a = ((np.arange(8000, dtype=np.uint16) + i) % 509).astype(np.uint16)
                key = f"r{i}".encode()
                arrs[key] = a
                be.get_or_acquire(key, (lambda a=a: (_make_typed_batch(a), a.nbytes)))
                be.release(key)

            segments = {be._metadata[k].segment_id for k in arrs}
            assert len(segments) > 1, "test needs at least one rotation"

            for key, a in arrs.items():
                loc = be.locate_entry(key)
                assert loc is not None and loc.byte_offset > 0
                assert np.array_equal(_read_via_location(loc), a)
        finally:
            be.close()
            shutil.rmtree(d, ignore_errors=True)

    def test_locate_survives_restart(self):
        """Offsets are recoverable after a restart (rebuild from segments)."""
        d = tempfile.mkdtemp()
        cfg = {
            "max_segment_bytes": 8 * 1024 * 1024,
            "max_total_bytes": 256 * 1024 * 1024,
        }
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

    def test_locate_hit_counts_in_top_level_stats(self, file_backend):
        """A locate/mmap fast-path hit bumps `stats().hits`, same as do_get.

        Regression for biopb/biopb#514: on the single-machine deployment a
        cached chunk is served via `locate_entry` (mmap handoff), which credited
        only the per-pool counter and left top-level `self._hits` at 0 -- so the
        reported hit-rate (surfaced by the sidecar) trended to ~0 even at a high
        true hit rate, and `stats().hits` disagreed with the per-pool counters.
        A fast-path hit and a do_get hit must move `stats().hits` identically.
        """
        a = (np.arange(1000, dtype=np.uint16) % 400).astype(np.uint16)
        # Populate: the initial acquire is a miss (creates the pending entry).
        file_backend.get_or_acquire(b"k", (lambda: (_make_typed_batch(a), a.nbytes)))
        file_backend.release(b"k")
        assert file_backend.stats().hits == 0

        # A locate/mmap fast-path hit must be counted at the top level.
        assert file_backend.locate_entry(b"k") is not None
        assert file_backend.stats().hits == 1
        # ...and stay internally consistent with the per-pool counters.
        s = file_backend.stats()
        assert sum(p.hits for p in s.pool_stats.values()) == s.hits

        # A do_get hit (get_or_acquire on the ready entry) moves it identically.
        before = file_backend.stats().hits
        file_backend.get_or_acquire(b"k", (lambda: (_make_typed_batch(a), a.nbytes)))
        file_backend.release(b"k")
        assert file_backend.stats().hits == before + 1

    def test_generation_id_is_segment_inode(self, file_backend):
        a = (np.arange(100, dtype=np.uint16)).astype(np.uint16)
        file_backend.get_or_acquire(b"g", (lambda: (_make_typed_batch(a), a.nbytes)))
        file_backend.release(b"g")
        loc = file_backend.locate_entry(b"g")
        assert loc.generation_id == os.stat(loc.segment_path).st_ino

    def test_torn_trailing_message_does_not_crash_boot_walk(self):
        """A torn trailing message must not crash the boot index walk.

        Regression for the Windows-only CI failure ``OSError: Expected to be
        able to read N bytes for message body``: a partial/failed write leaves
        slack at the segment tail, and the walk must treat it as
        end-of-readable-region rather than propagate. That slack outlives the
        process that wrote it, so the walk that has to survive it is the one at
        boot (``_scan_segment_records``) -- which is also the only walk left
        since ranges became write-time (biopb/biopb#541). Reproduced
        cross-platform by appending a truncated copy of a real message, which
        also invalidates the ``.idx`` sidecar (recorded size != file size) and
        so forces the body walk rather than the sidecar fast path.
        """
        d = tempfile.mkdtemp()
        cfg = {
            "max_segment_bytes": 64 * 1024 * 1024,
            "max_total_bytes": 256 * 1024 * 1024,
        }
        be = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **cfg))
        arrs = {}
        try:
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
            full = raw[loc0.byte_offset : loc0.byte_offset + loc0.byte_length]
        finally:
            be.close()
        with open(seg_path, "ab") as f:
            f.write(full[: len(full) // 2])

        # Boot on the torn segment: must not raise, and every good entry ahead
        # of the torn tail must still index and read back.
        be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **cfg))
        try:
            for i in range(3):
                loc = be2.locate_entry(f"good-{i}".encode())
                assert loc is not None, f"good-{i} should still locate"
                assert np.array_equal(_read_via_location(loc), arrs[i])
        finally:
            be2.close()
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

        schema = pa.schema(
            [],
            metadata={
                b"tensor_schema_version": b"0.4.0",
                b"other_key": b"other_value",
            },
        )
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

        a = (np.arange(120, dtype=np.uint16).reshape(8, 15) % 97).astype(np.uint16)
        unified = _make_typed_batch(a)  # already the unified binary schema
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
        src = (np.arange(96 * 96, dtype=np.uint16).reshape(96, 96) % 997).astype(
            np.uint16
        )
        z = zarr.open_array(
            zpath, mode="w", shape=src.shape, chunks=(48, 48), dtype=src.dtype
        )
        z[:] = src
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source(
            "z", ZarrAdapter(zarr.open_array(zpath, mode="r"), "z", ["y", "x"])
        )
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
            got = client.get_tensor("z").compute(scheduler="threads")
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
                got = client.get_tensor("z").compute(scheduler="threads")
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
            got = client.get_tensor("z").compute(scheduler="threads")
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
            # The constant is read by _try_cachefile_transfer in biopb.tensor._pool
            # (issue #278 item C), so patch it there, not on the client re-export.
            with patch("biopb.tensor._pool._CACHEFILE_SUPPORTED_FORMAT", 0):
                client = TensorFlightClient(loc, cache_bytes=0)
                got = client.get_tensor("z").compute(scheduler="threads")
                assert np.array_equal(got, src)  # correct via do_get fallback
                assert cmod._cachefile_support.get(loc) is False  # declined + memoized
                client.close()
        finally:
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# Backend: direct-seek segment reads
# ==============================================================================


class TestDirectSeekRead:
    """A do_get cache hit decodes the one message at the entry's recorded byte
    range instead of walking the segment to reach it."""

    def _seal_segment(self, backend, n=12):
        """Write n entries and seal their segment so it is mmap-readable."""
        arrs = {}
        for i in range(n):
            a = ((np.arange(600, dtype=np.uint16) + i * 7) % 401).astype(np.uint16)
            arrs[i] = a
            key = f"seek-{i}".encode()
            backend.get_or_acquire(key, (lambda a=a: (_make_typed_batch(a), a.nbytes)))
            backend.release(key)
        segment_id = backend._metadata[f"seek-{n - 1}".encode()].segment_id
        backend._close_segment(segment_id)
        return arrs, segment_id

    def test_seek_and_walk_agree_for_every_entry(self, file_backend):
        """The seek path and the sequential walk return identical batches.

        Pins the fallback as a true equivalent, not an approximation: reading
        entry i by byte range must equal reading it by stream position.
        """
        arrs, segment_id = self._seal_segment(file_backend)
        mmap = file_backend._segment_mmaps[segment_id]

        for i in range(len(arrs)):
            info = file_backend._metadata[f"seek-{i}".encode()]
            assert info.byte_offset > 0 and info.byte_length > 0

            seeked = file_backend._read_batch_at(segment_id, mmap, info)
            # Same entry with its range stripped -> forced down the walk.
            walked = file_backend._read_batch_at(
                segment_id,
                mmap,
                dataclasses.replace(info, byte_offset=0, byte_length=0),
            )
            assert seeked.equals(walked)
            assert np.array_equal(unpack_chunk_array(seeked), arrs[i])

    def test_entry_without_range_still_reads(self, file_backend):
        """An entry whose byte range was never derived falls back to the walk
        rather than failing -- do_get is the designed floor of this path."""
        arrs, segment_id = self._seal_segment(file_backend, n=4)
        key = b"seek-2"
        info = file_backend._metadata[key]
        file_backend._metadata[key] = dataclasses.replace(
            info, byte_offset=0, byte_length=0
        )
        # Drop the in-memory mirror so the read must come off the segment.
        file_backend._entries.pop(key, None)

        entry = file_backend.get_or_acquire(key, (lambda: (None, 0)))
        try:
            assert np.array_equal(unpack_chunk_array(entry.data), arrs[2])
        finally:
            file_backend.release(key)

    def test_schema_cache_does_not_outlive_its_mapping(self, file_backend):
        """Dropping a segment's mmap drops its cached schema, so a segment id
        reused after clear() can never decode against a stale schema."""
        _arrs, segment_id = self._seal_segment(file_backend, n=4)
        mmap = file_backend._segment_mmaps[segment_id]
        assert file_backend._segment_schema(segment_id, mmap) is not None
        assert segment_id in file_backend._segment_schemas

        file_backend._forget_segment_mmap(segment_id)
        assert segment_id not in file_backend._segment_schemas
        assert segment_id not in file_backend._segment_mmaps

    def test_clear_drops_all_cached_schemas(self, file_backend):
        _arrs, segment_id = self._seal_segment(file_backend, n=4)
        mmap = file_backend._segment_mmaps[segment_id]
        file_backend._segment_schema(segment_id, mmap)
        assert file_backend._segment_schemas

        file_backend.clear()
        assert not file_backend._segment_schemas
        assert not file_backend._segment_mmaps
