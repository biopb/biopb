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
import hashlib
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
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.file_backend import (
    CACHE_FILE_FORMAT_VERSION,
    FORMAT_VERSION_MARKER,
    ArrowFileBackend,
    ArrowFileConfig,
    ChunkLocation,
)
from biopb_tensor_server.core.adapter_base import pack_chunk_batch, unpack_chunk_array
from biopb_tensor_server.core.chunk import encode_chunk_id
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

    def test_segment_files_are_never_truncated_in_place(self):
        """Load-bearing invariant for the localhost mmap fast path (Option C,
        biopb/biopb#571): a client hands out a zero-copy *view* onto a segment
        mapping, so a segment inode must never be truncated or recreated in place
        while it could still be mapped -- that would SIGBUS the remote reader far
        from the read that produced the array.

        The guard is strictly-monotonic segment ids (``_next_segment_id``, boot
        max+1, only incremented): the sole truncating open,
        ``pa.OSFile(path, "wb")``, must target a fresh path every time, even
        across rotation and eviction. Spy on that open and assert no segment path
        is ever opened for write twice.
        """
        import re

        d = tempfile.mkdtemp()
        cfg = ArrowFileConfig(
            cache_dir=Path(d),
            max_segment_bytes=64 * 1024,  # tiny -> frequent rotation
            max_total_bytes=256 * 1024,  # tiny -> frequent eviction
        )
        seg_re = re.compile(r"seg_\d+\.arrow$")
        opened_for_write: list[str] = []
        real_osfile = pa.OSFile

        def spy_osfile(path, mode="rb", *a, **k):
            if "w" in mode and seg_re.search(str(path)):
                opened_for_write.append(str(path))
            return real_osfile(path, mode, *a, **k)

        be = ArrowFileBackend(cfg)
        try:
            with patch.object(pa, "OSFile", spy_osfile):
                for i in range(200):
                    a = ((np.arange(4000, dtype=np.uint16) + i) % 509).astype(np.uint16)
                    key = f"k-{i}".encode()
                    be.get_or_acquire(
                        key, (lambda a=a: (_make_typed_batch(a), a.nbytes))
                    )
                    be.release(key)
        finally:
            be.close()
            shutil.rmtree(d, ignore_errors=True)

        # The workload must actually rotate (else the test proves nothing)...
        assert len(opened_for_write) > 1, "test did not rotate segments"
        # ...and no segment path was ever truncated/recreated in place.
        assert len(opened_for_write) == len(set(opened_for_write)), (
            "a segment path was opened 'wb' twice -- a mapped inode could be "
            "truncated under a remote reader (Option C invariant broken)"
        )


class TestFormatVersionEnforcement:
    """CACHE_FILE_FORMAT_VERSION is enforced with an on-disk marker + wipe.

    The version marks the segment message layout / cache-key encoding contract.
    Segments written under an incompatible version must be dropped at boot rather
    than indexed and served (mis-decoded / stale). Covers: the marker is stamped
    on init; a matching marker preserves the cache across restart; and a missing
    (pre-enforcement), mismatched, or torn marker wipes the segments + WAL.
    """

    CFG = {"max_segment_bytes": 8 * 1024 * 1024, "max_total_bytes": 256 * 1024 * 1024}

    def _seed(self, d, key=b"k0"):
        """Write one entry into a fresh backend at ``d`` and close it."""
        be = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))
        a = (np.arange(2000, dtype=np.uint16) % 500).astype(np.uint16)
        be.get_or_acquire(key, (lambda: (_make_typed_batch(a), a.nbytes)))
        be.release(key)
        be.close()
        return a

    def test_marker_written_on_init(self):
        d = tempfile.mkdtemp()
        try:
            be = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))
            be.close()
            marker = Path(d) / FORMAT_VERSION_MARKER
            assert marker.exists()
            assert int(marker.read_text().strip()) == CACHE_FILE_FORMAT_VERSION
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_matching_marker_preserves_cache(self):
        """A restart at the same version keeps the segments (no wipe)."""
        d = tempfile.mkdtemp()
        try:
            a = self._seed(d)
            be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))
            try:
                loc = be2.locate_entry(b"k0")
                assert loc is not None
                assert np.array_equal(_read_via_location(loc), a)
            finally:
                be2.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_missing_marker_wipes_preexisting_cache(self):
        """A pre-enforcement dir (segments, no marker) is wiped on next boot."""
        d = tempfile.mkdtemp()
        try:
            self._seed(d)
            marker = Path(d) / FORMAT_VERSION_MARKER
            marker.unlink()  # simulate a cache written before enforcement shipped
            assert list((Path(d) / "segments").glob("seg_*.arrow"))

            be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))
            try:
                assert be2.locate_entry(b"k0") is None  # wiped
                assert not list((Path(d) / "segments").glob("seg_*.arrow"))
                # marker re-stamped at the current version
                assert int(marker.read_text().strip()) == CACHE_FILE_FORMAT_VERSION
            finally:
                be2.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_mismatched_marker_wipes_cache(self):
        d = tempfile.mkdtemp()
        try:
            self._seed(d)
            marker = Path(d) / FORMAT_VERSION_MARKER
            marker.write_text(f"{CACHE_FILE_FORMAT_VERSION + 1}\n")

            be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))
            try:
                assert be2.locate_entry(b"k0") is None
                assert int(marker.read_text().strip()) == CACHE_FILE_FORMAT_VERSION
            finally:
                be2.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_torn_marker_wipes_cache(self):
        """An unparseable marker (torn write) is treated as a mismatch."""
        d = tempfile.mkdtemp()
        try:
            self._seed(d)
            (Path(d) / FORMAT_VERSION_MARKER).write_text("not-an-int")

            be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))
            try:
                assert be2.locate_entry(b"k0") is None
            finally:
                be2.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_partial_wipe_fails_closed(self):
        """If the wipe leaves a segment behind, init raises instead of serving it.

        ``rmtree(ignore_errors=True)`` can silently leave files (NFS unlink
        error, held handle). A surviving ``seg_*.arrow`` would be re-indexed as
        if current, so construction must fail rather than serve incompatible
        bytes. Simulated by neutering ``rmtree`` so the segments survive.
        """
        d = tempfile.mkdtemp()
        try:
            self._seed(d)
            (Path(d) / FORMAT_VERSION_MARKER).write_text(
                f"{CACHE_FILE_FORMAT_VERSION + 1}\n"
            )
            with patch("biopb_tensor_server.cache.file_backend.shutil.rmtree"):
                with pytest.raises(RuntimeError, match="survived the wipe"):
                    ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))

            # The aborted init must not leave the process lock held: a retry
            # (now with rmtree working) acquires cleanly and wipes.
            be2 = ArrowFileBackend(ArrowFileConfig(cache_dir=Path(d), **self.CFG))
            try:
                assert be2.locate_entry(b"k0") is None
            finally:
                be2.close()
        finally:
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

    def test_locate_counts_as_flight_activity(self):
        """A locate is in-flight *while it runs*, so precache parks (#548).

        The fast path replaces do_get, so if the handler is untracked the server
        looks idle for the whole of a localhost read and the precache worker
        warms straight through it.
        """
        server = TensorFlightServer("grpc://localhost:0")
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        try:
            observed = []

            class _Probe:
                source_id = "z"

                def resolve_chunk_data(self, chunk_id, cache_manager):
                    # Called on a cold miss -- the heaviest work in the handler.
                    observed.append(server.flight_idle_for(0.0))
                    raise ValueError("no data")

            chunk_id = encode_chunk_id("z", ChunkBounds(start=[0, 0], stop=[8, 8]))
            with patch.object(server, "_get_adapter_for_chunk", return_value=_Probe()):
                with pytest.raises(flight.FlightError):
                    server._handle_chunk_locate(chunk_id)

            assert observed == [False], "server reported idle mid-locate"
            # ...and the stamp advances on the way out, so the debounce window
            # starts from the end of the read rather than from process start.
            assert server.activity._last_active > 0.0
        finally:
            CacheManager.reset()
            server.shutdown()

    def test_locate_releases_the_activity_slot_on_error(self):
        """A failing locate must not leak an in-flight count (precache would
        then never run again)."""
        server = TensorFlightServer("grpc://localhost:0")
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        try:
            # No source registered -> the adapter lookup raises straight out of
            # the tracked block.
            ghost = encode_chunk_id("ghost", ChunkBounds(start=[0, 0], stop=[8, 8]))
            with pytest.raises(flight.FlightError):
                server._handle_chunk_locate(ghost)
            assert server.activity._inflight == 0
            assert server.flight_idle_for(0.0) is True
        finally:
            CacheManager.reset()
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

    def test_enabled_on_localhost(self):
        from biopb.tensor.client import _should_try_cachefile

        # Enabled on every platform now, Windows included (biopb/biopb#582).
        assert _should_try_cachefile("grpc://localhost:8815") is True

    def test_disabled_by_env(self):
        from biopb.tensor.client import _should_try_cachefile

        with patch.dict(os.environ, {"BIOPB_CACHEFILE_TRANSFER_DISABLED": "1"}):
            assert _should_try_cachefile("grpc://localhost:8815") is False

    def test_disabled_for_remote(self):
        from biopb.tensor.client import _should_try_cachefile

        assert _should_try_cachefile("grpc://192.168.1.100:8815") is False

    def test_not_gated_by_os_name(self):
        """The fast path is no longer gated on os.name (biopb/biopb#582): a
        Windows host still takes the localhost path, so the copy on read the
        POSIX gate used to protect is not needed."""
        import biopb.tensor.client as c

        with patch.object(os, "name", "nt"):
            assert c._should_try_cachefile("grpc://localhost:8815") is True

    def test_skips_after_unsupported_memoized(self):
        import biopb.tensor.client as c

        c._set_cachefile_supported("grpc://localhost:8815", False)
        assert c._should_try_cachefile("grpc://localhost:8815") is False


class TestClientViewSurvivesServerUnlink:
    """The cross-process guarantee the #582 gate removal rests on: a client that
    holds a fast-path mmap *view* (having closed its MemoryMappedFile handle, as
    _try_cachefile_transfer does) does not block the server unlinking that
    segment, and its data stays valid across the delete.

    On Windows this is the delete-on-last-close / delete-pending behaviour: an
    open handle would block the unlink (WinError 32), but a view does not -- the
    name is removed at once, the pages stay valid, and the freed name is even
    reusable. A pyarrow that pinned the handle through the view (or a platform
    that blocked the unlink) would fail here rather than in production.
    """

    def _write_segment(self, path: Path) -> tuple[np.ndarray, str]:
        arr = (np.arange(4 * 1024 * 1024, dtype=np.int64) & 0xFF).astype(np.uint8)
        batch = _make_typed_batch(arr.reshape(2048, 2048))
        with pa.OSFile(str(path), "wb") as sink:
            w = pa.RecordBatchStreamWriter(sink, batch.schema)
            w.write_batch(batch)
            w.close()
        return arr, hashlib.sha256(arr.tobytes()).hexdigest()

    def _client_view(self, path: Path) -> np.ndarray:
        """Mirror _try_cachefile_transfer: map, decode zero-copy, close mm, keep view."""
        import biopb.tensor._pool as pool

        mm = pa.memory_map(str(path), "r")
        try:
            batch = pa.ipc.open_stream(mm).read_next_batch()
            # The real fast path's decoder: arr is a frombuffer *view* aliasing
            # the mapping, not a copy (Option C, biopb/biopb#571).
            arr, _buf = pool._decode_unified_batch(batch)
        finally:
            mm.close()  # drop the handle; the view lives on via `arr`
        assert not arr.flags.owndata, "test must hold a real mmap view, not a copy"
        return arr

    def test_view_survives_unlink_and_name_is_reusable(self):
        d = Path(tempfile.mkdtemp())
        try:
            seg = d / "seg_0001.arrow"
            _src, want = self._write_segment(seg)
            arr = self._client_view(seg)  # client holds a view, handle closed

            # Server evicts: unlink must succeed even with the client view live.
            os.unlink(seg)
            assert not seg.exists()  # name gone at once

            # The client's view still reads the right bytes after the delete.
            assert hashlib.sha256(arr.tobytes()).hexdigest() == want

            # Windows-specific: the freed name is reusable (no ACCESS_DENIED
            # delete-pending block) -- the server can lay down a new segment here.
            # Trivially true on POSIX; the real assertion is on Windows.
            with open(seg, "wb") as f:
                f.write(b"new-segment")
            assert seg.exists()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestArrayFromUnifiedBatch:
    def test_decode_matches_source(self):
        from biopb.tensor.client import _array_from_unified_batch

        a = (np.arange(120, dtype=np.uint16).reshape(8, 15) % 97).astype(np.uint16)
        unified = _make_typed_batch(a)  # already the unified binary schema
        got = _array_from_unified_batch(unified)
        assert got.dtype == np.uint16 and got.shape == (8, 15)
        assert np.array_equal(got, a)

    def test_view_decode_matches_source(self):
        from biopb.tensor.client import _array_from_unified_batch

        a = (np.arange(120, dtype=np.uint16).reshape(8, 15) % 97).astype(np.uint16)
        got = _array_from_unified_batch(_make_typed_batch(a), copy=False)
        assert got.dtype == np.uint16 and got.shape == (8, 15)
        assert np.array_equal(got, a)

    def test_both_paths_return_readonly(self):
        # The mutability contract is uniform across the do_get view path
        # (copy=False) and the mmap fast path (copy=True) -- biopb/biopb#571.
        from biopb.tensor.client import _array_from_unified_batch

        a = (np.arange(120, dtype=np.uint16).reshape(8, 15) % 97).astype(np.uint16)
        for copy in (True, False):
            got = _array_from_unified_batch(_make_typed_batch(a), copy=copy)
            assert not got.flags.writeable
            with pytest.raises(ValueError):
                got[0, 0] = 1

    def test_copy_owns_its_bytes(self):
        # copy=True must not alias the batch's Arrow buffer: the mmap fast path
        # closes its segment mapping the instant the helper returns.
        from biopb.tensor.client import _array_from_unified_batch

        a = (np.arange(120, dtype=np.uint16).reshape(8, 15) % 97).astype(np.uint16)
        got = _array_from_unified_batch(_make_typed_batch(a), copy=True)
        assert got.flags.owndata

    def test_view_shares_and_survives_source(self):
        # copy=False is a zero-copy view kept alive through the buffer protocol:
        # it must stay valid after every intermediate Arrow ref is dropped.
        import gc

        from biopb.tensor.client import _array_from_unified_batch

        a = (np.arange(4000, dtype=np.uint32) % 251).astype(np.uint32).reshape(40, 100)
        batch = _make_typed_batch(a)
        got = _array_from_unified_batch(batch, copy=False)
        assert not got.flags.owndata  # a view, not an owned copy
        del batch
        gc.collect()
        assert np.array_equal(got, a)


class TestPinnedSegmentAccounting:
    """Client-side pinned-segment accounting (disk-leak workaround, #571).

    A fast-path view keeps its segment's mmap alive, pinning the server's disk
    against reclamation. The client caps the total mapped-segment size and copies
    the chunk out once over budget, so the leak is bounded.
    """

    def setup_method(self):
        import biopb.tensor._pool as pool

        pool._pinned_total = 0
        pool._pinned_segments.clear()

    teardown_method = setup_method

    def test_refcounted_by_inode(self):
        import biopb.tensor._pool as pool

        class _Anchor:
            pass

        a1, a2, a3 = _Anchor(), _Anchor(), _Anchor()
        # Two chunks from one segment (same inode) pin its disk exactly once.
        pool._register_segment_pin(100, 64_000_000, a1)
        pool._register_segment_pin(100, 64_000_000, a2)
        pool._register_segment_pin(200, 32_000_000, a3)
        assert pool._pinned_total == 96_000_000
        assert pool._pinned_segments[100].refs == 2
        assert pool._pinned_segments[200].refs == 1

    def test_last_holder_unpins_segment(self):
        import gc

        import biopb.tensor._pool as pool

        class _Anchor:
            pass

        a1, a2 = _Anchor(), _Anchor()
        pool._register_segment_pin(100, 64_000_000, a1)
        pool._register_segment_pin(100, 64_000_000, a2)

        del a1
        gc.collect()
        # Still one holder of inode 100 -> segment stays pinned.
        assert pool._pinned_segments[100].refs == 1
        assert pool._pinned_total == 64_000_000

        del a2
        gc.collect()
        # Last holder gone -> the finalizer un-pins and reclaims the account.
        assert 100 not in pool._pinned_segments
        assert pool._pinned_total == 0

    def test_gate_default_and_env_override(self):
        import biopb.tensor._pool as pool
        from biopb.tensor.client import _pin_budget_exhausted, _pin_limit_bytes

        # Nothing pinned -> under the (16 GiB) default budget.
        assert _pin_budget_exhausted() is False

        with patch.dict(os.environ, {"BIOPB_CACHEFILE_PIN_LIMIT_BYTES": "1000"}):
            assert _pin_limit_bytes() == 1000
            pool._pinned_total = 999
            assert _pin_budget_exhausted() is False
            pool._pinned_total = 1000  # at the limit -> exhausted (>=)
            assert _pin_budget_exhausted() is True

        # 0 disables the view handoff outright (always copy).
        pool._pinned_total = 0
        with patch.dict(os.environ, {"BIOPB_CACHEFILE_PIN_LIMIT_BYTES": "0"}):
            assert _pin_budget_exhausted() is True

        # Unparseable / negative fall back to the default.
        with patch.dict(os.environ, {"BIOPB_CACHEFILE_PIN_LIMIT_BYTES": "garbage"}):
            assert _pin_limit_bytes() == pool._PIN_LIMIT_DEFAULT
        with patch.dict(os.environ, {"BIOPB_CACHEFILE_PIN_LIMIT_BYTES": "-1"}):
            assert _pin_limit_bytes() == pool._PIN_LIMIT_DEFAULT


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
            # The fast path was actually used (chunk_locate succeeded), on every
            # platform now including Windows (biopb/biopb#582).
            assert cmod._cachefile_support.get(loc) is True
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

    def _one_endpoint(self, client):
        """A real (chunk_id, start, stop) triple for source "z"'s first chunk."""
        ctx = client._get_tensor_context("z")
        chunk_id, bounds = ctx.endpoints[0]
        return chunk_id, tuple(bounds.start), tuple(bounds.stop)

    def test_fetched_block_is_read_only_fast_path(self):
        """End-to-end read contract: a chunk pulled through the real fetch leaf
        via the localhost mmap fast path is read-only, and mutating it raises."""
        import biopb.tensor.client as cmod
        from biopb.tensor.client import TensorFlightClient, _fetch_chunk_distributed

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            cmod._cachefile_support.clear()
            client = TensorFlightClient(loc, cache_bytes=0)
            # Warm the server's file cache so chunk_locate can find the chunk.
            client.get_tensor("z").compute(scheduler="threads")
            chunk_id, start, stop = self._one_endpoint(client)

            block = _fetch_chunk_distributed(loc, None, chunk_id, start, stop, 0)
            assert cmod._cachefile_support.get(loc) is True  # fast path was used
            assert block.shape == (48, 48)
            assert not block.flags.writeable
            with pytest.raises(ValueError):
                block[0, 0] = 0
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fast_path_hands_out_mmap_view_that_outlives_close(self):
        """Option C (biopb/biopb#571): the fast path returns a zero-copy *view*
        onto the segment mapping, not an owned copy. The leaf closes its
        MemoryMappedFile handle before returning, so this also proves Arrow keeps
        the mapping alive for the array's lifetime -- the block still decodes the
        correct bytes after the handle is gone and every intermediate ref is
        dropped and gc'd."""
        import biopb.tensor.client as cmod
        from biopb.tensor.client import TensorFlightClient, _fetch_chunk_distributed

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            cmod._cachefile_support.clear()
            client = TensorFlightClient(loc, cache_bytes=0)
            client.get_tensor("z").compute(scheduler="threads")  # warm cache
            chunk_id, start, stop = self._one_endpoint(client)

            block = _fetch_chunk_distributed(loc, None, chunk_id, start, stop, 0)
            assert cmod._cachefile_support.get(loc) is True
            # A view, not an owned copy: this is the whole point of Option C.
            assert not block.flags.owndata
            # The mmap handle was already closed inside the leaf; force a gc to
            # drop any stray decode intermediates, then read through the mapping.
            import gc

            gc.collect()
            expected = src[start[0] : stop[0], start[1] : stop[1]]
            assert np.array_equal(block, expected)
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fast_path_view_pins_segment_and_unpins_on_gc(self):
        """A fast-path view charges its segment against the pinned-disk budget
        and releases the charge when the block (and its backing buffer) is gc'd
        -- the accounting that lets the client bound the server disk-leak (#571)."""
        import gc

        import biopb.tensor._pool as pool
        from biopb.tensor.client import TensorFlightClient, _fetch_chunk_distributed

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            pool._cachefile_support.clear()
            client = TensorFlightClient(loc, cache_bytes=0)
            client.get_tensor("z").compute(scheduler="threads")  # warm cache
            chunk_id, start, stop = self._one_endpoint(client)

            # Reset accounting after the warm compute so we measure this one fetch.
            gc.collect()
            pool._pinned_total = 0
            pool._pinned_segments.clear()

            block = _fetch_chunk_distributed(loc, None, chunk_id, start, stop, 0)
            assert pool._cachefile_support.get(loc) is True  # fast path used
            assert not block.flags.owndata  # a view, not a copy
            # The segment is now pinned for as long as the block references it.
            assert pool._pinned_total > 0
            assert len(pool._pinned_segments) == 1

            del block
            gc.collect()
            # Block gone -> the finalizer un-pins the segment; budget reclaimed.
            assert pool._pinned_total == 0
            assert not pool._pinned_segments
            client.close()
        finally:
            pool._pinned_total = 0
            pool._pinned_segments.clear()
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fast_path_copies_when_over_pin_budget(self):
        """Over the pinned-segment budget the fast path copies the chunk out and
        releases the mapping instead of handing out another view -- so it pins no
        further server disk (the disk-leak bound, #571). It still reads off the
        warm mmap (fast path used), it just owns its bytes and is still
        read-only."""
        import biopb.tensor._pool as pool
        from biopb.tensor.client import TensorFlightClient, _fetch_chunk_distributed

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            pool._cachefile_support.clear()
            pool._pinned_total = 0
            pool._pinned_segments.clear()
            client = TensorFlightClient(loc, cache_bytes=0)
            client.get_tensor("z").compute(scheduler="threads")  # warm cache
            chunk_id, start, stop = self._one_endpoint(client)

            # limit 0 -> always over budget -> always copy.
            with patch.dict(os.environ, {"BIOPB_CACHEFILE_PIN_LIMIT_BYTES": "0"}):
                pool._pinned_total = 0
                pool._pinned_segments.clear()
                block = _fetch_chunk_distributed(loc, None, chunk_id, start, stop, 0)

            assert pool._cachefile_support.get(loc) is True  # still the mmap path
            assert block.flags.owndata  # copied out, does not alias the mapping
            assert not block.flags.writeable  # uniform read-only contract
            # No segment was pinned -- the whole point over budget.
            assert pool._pinned_total == 0
            assert not pool._pinned_segments
            expected = src[start[0] : stop[0], start[1] : stop[1]]
            assert np.array_equal(block, expected)
            client.close()
        finally:
            pool._pinned_total = 0
            pool._pinned_segments.clear()
            server.shutdown()
            CacheManager.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fetched_block_is_read_only_do_get(self):
        """End-to-end read contract on the do_get path (fast path disabled): the
        zero-copy view is read-only on every platform, and still decodes right."""
        import biopb.tensor.client as cmod
        from biopb.tensor.client import TensorFlightClient, _fetch_chunk_distributed

        tmp = tempfile.mkdtemp()
        cfg = CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
        server, src = self._serve_zarr(tmp, cfg)
        loc = f"grpc://localhost:{server.port}"
        try:
            cmod._cachefile_support.clear()
            with patch.dict(os.environ, {"BIOPB_CACHEFILE_TRANSFER_DISABLED": "1"}):
                client = TensorFlightClient(loc, cache_bytes=0)
                chunk_id, start, stop = self._one_endpoint(client)

                block = _fetch_chunk_distributed(loc, None, chunk_id, start, stop, 0)
                # Fast path was never probed -- this is the do_get view path.
                assert cmod._cachefile_support.get(loc) is None
                assert not block.flags.writeable
                # The view holds the correct chunk bytes for its bounds.
                expected = src[start[0] : stop[0], start[1] : stop[1]]
                assert np.array_equal(block, expected)
                with pytest.raises(ValueError):
                    block[0, 0] = 0
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


class TestZeroCopySurvivesUnlink:
    """A cache hit hands out a batch that aliases the segment mmap, and that
    batch stays valid after the segment file is unlinked while the caller still
    holds it -- the POSIX delete-on-last-close guarantee that #572 dropped the
    Windows per-hit copy to rely on.

    This is the guardrail for that decision. The copy was removed on the basis
    that (a) every unlink path closes the server's own pa.memory_map before
    deleting, and (b) MemoryMappedFile.close() releases the OS handle while the
    mapped view stays alive for outstanding buffers -- true on pyarrow >= 14,
    the pinned floor. On a pyarrow where the batch instead pinned the file
    handle, the unlink below would raise WinError 32 on Windows; on any platform
    a decode-time copy would show up as an allocation. Both are asserted, so a
    pyarrow bump that regresses either fails here rather than in production.
    """

    def _seal_segment_with(self, backend, n=6):
        """Write n ~64 KiB chunks into one segment and seal it mmap-readable."""
        arrs = {}
        for i in range(n):
            a = ((np.arange(32768, dtype=np.uint16) + i * 11) % 509).astype(np.uint16)
            key = f"z{i}".encode()
            arrs[key] = a
            backend.get_or_acquire(key, (lambda a=a: (_make_typed_batch(a), a.nbytes)))
            backend.release(key)
        segment_id = backend._metadata[b"z0"].segment_id
        backend._close_segment(segment_id)
        # Drop any in-RAM mirror so the read must come off the mmap, not a copy.
        for key in arrs:
            backend._entries.pop(key, None)
        return arrs, segment_id

    def test_read_is_zero_copy(self, file_backend):
        """Decoding a ~64 KiB chunk off the segment allocates ~nothing."""
        arrs, _ = self._seal_segment_with(file_backend)
        # pa.total_allocated_bytes() is the process-wide default-pool counter, so
        # this before/after delta only isolates this call's allocation while the
        # suite runs serially. Keep tensor-server tests single-process (no
        # pytest-xdist ``-n``) or a concurrent test's allocation would leak in
        # and make this flaky.
        before = pa.total_allocated_bytes()
        batch = file_backend._read_batch_from_segment(b"z0")
        allocated = pa.total_allocated_bytes() - before
        assert batch is not None
        # A full copy would allocate the whole ~64 KiB payload; a zero-copy
        # alias allocates only tiny bookkeeping, well under a quarter of it.
        assert allocated < arrs[b"z0"].nbytes // 4
        assert np.array_equal(unpack_chunk_array(batch), arrs[b"z0"])

    def test_batch_survives_segment_eviction(self, file_backend):
        """Evicting a segment out from under a live batch unlinks the file and
        leaves the batch's bytes intact -- the core #572 guarantee."""
        arrs, segment_id = self._seal_segment_with(file_backend)
        batch = file_backend._read_batch_from_segment(b"z0")
        assert batch is not None

        seg_path = file_backend._segment_path(segment_id)
        assert seg_path.exists()

        # Eviction closes the server mmap, then unlinks. Must not raise (a
        # regressed pyarrow would raise WinError 32 on Windows) ...
        with file_backend._lock:
            file_backend._do_evict_segment(segment_id)
        assert not seg_path.exists()  # ... and must actually remove the file.

        # The batch the caller is still holding reads correctly after the delete.
        assert np.array_equal(unpack_chunk_array(batch), arrs[b"z0"])

    def test_batch_survives_clear(self, file_backend):
        """clear() unlinks every segment; a batch held across it stays valid."""
        arrs, _ = self._seal_segment_with(file_backend)
        batch = file_backend._read_batch_from_segment(b"z1")
        assert batch is not None

        file_backend.clear()  # unlinks all segments while `batch` is live

        assert np.array_equal(unpack_chunk_array(batch), arrs[b"z1"])
