"""Evicting an OME-TIFF source while it is read and while the reaper sweeps (#1284).

Eviction drops a registered adapter and rebuilds it from its catalog row on the
next use. These drive that churn against concurrent reads and against the idle
reaper, and check what must hold throughout: every read returns the right bytes
or a clear error, no thread hangs, and the process holds no more descriptors
afterwards than before. Set ``BIOPB_TEST_OMETIFF`` to a large file to make the
reopen window long enough to race reliably.
"""

import faulthandler
import gc
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import OmeTiffAdapter, ome_tiff as ome_tiff_module
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import SourceClaim
from biopb_tensor_server.serving.metadata_db import CatalogRecord, MetadataDatabase
from biopb_tensor_server.sources.source_registry import close_adapter

READERS = 4
SECONDS = float(os.environ.get("BIOPB_TEST_EVICTION_SECONDS", "1.5"))
JOIN_TIMEOUT = 30.0


def _descriptors():
    return len(os.listdir("/proc/self/fd"))


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="counts /proc/self/fd"
)


@pytest.fixture
def ome_path(tmp_path) -> Path:
    given = os.environ.get("BIOPB_TEST_OMETIFF")
    if given:
        return Path(given)
    path = tmp_path / "evict.ome.tif"
    data = np.random.default_rng(0).integers(0, 60000, (6, 256, 256), dtype=np.uint16)
    tifffile.imwrite(
        path, data, ome=True, metadata={"axes": "ZYX"}, tile=(64, 64), compression=None
    )
    return path


class _Churn:
    """A source that can be evicted and rebuilt from its row, and read from."""

    def __init__(self, path: Path):
        self.path = str(path)
        self.config = SourceConfig(url=self.path, source_id="s0")
        self.lock = threading.Lock()
        adapter = OmeTiffAdapter.create_from_config(self.config)
        self.array_id = adapter.list_tensor_descriptors()[0].array_id
        self.db = MetadataDatabase()
        self.db.sync_roots([("r", "file:///")])
        claim = SourceClaim(
            adapter.source_type, self.path, "src", member_paths=[self.path]
        )
        self.db.sync_source_added(
            "src", adapter, CatalogRecord(claim, {self.path: (1, 2, 3, 4)}, "r", "x")
        )
        self.current = adapter
        self.reference = self.read(self.scene(adapter))
        self.errors = []
        self.reads = 0
        self.evictions = 0

    def scene(self, source=None):
        return (source or self.current).get_tensor_adapter(self.array_id)

    def read(self, scene):
        shape = list(scene.get_tensor_descriptor().shape)
        stop = [min(s, 64) for s in shape]
        return np.asarray(
            scene.get_data(ChunkBounds(start=[0] * len(shape), stop=stop))
        )

    def rebuild(self):
        payload, metadata = self.db.read_hydration("src")
        return OmeTiffAdapter.create_from_payload(self.config, payload, metadata, None)

    def evict(self, close: bool):
        """Replace the registered adapter by a rebuilt one, then drop the old."""
        fresh = self.rebuild()
        with self.lock:
            old, self.current = self.current, fresh
        if close:
            close_adapter(old)
        self.evictions += 1

    def reader(self, stop: threading.Event):
        while not stop.is_set():
            with self.lock:
                source = self.current
            try:
                out = self.read(self.scene(source))
                if not np.array_equal(out, self.reference):
                    self.errors.append("wrong bytes")
                self.reads += 1
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}")


def _run(churn: _Churn, background, seconds=SECONDS):
    """Readers plus *background* threads for *seconds*; fail on a hang."""
    stop = threading.Event()
    threads = [
        threading.Thread(target=churn.reader, args=(stop,), name=f"reader{i}")
        for i in range(READERS)
    ] + [
        threading.Thread(target=fn, args=(stop,), name=fn.__name__) for fn in background
    ]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    deadline = time.monotonic() + JOIN_TIMEOUT
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    hung = [t.name for t in threads if t.is_alive()]
    if hung:
        faulthandler.dump_traceback(all_threads=True)
    assert not hung, f"threads did not finish: {hung}"


def _assert_clean(churn: _Churn, before: int):
    assert not churn.errors, sorted(set(churn.errors))[:5]
    assert churn.reads > 0
    gc.collect()
    assert _descriptors() <= before + 2, "descriptors leaked after eviction churn"


@pytest.fixture
def churn(ome_path):
    gc.collect()
    before = _descriptors()
    c = _Churn(ome_path)
    yield c, before
    close_adapter(c.current)
    c.db.close()


def _evictor(c: _Churn, close: bool):
    def evict_loop(stop):
        while not stop.is_set():
            c.evict(close)

    return evict_loop


class TestEvictionAgainstReads:
    def test_closing_the_old_adapter_under_reads(self, churn):
        c, before = churn
        _run(c, [_evictor(c, close=True)])
        assert c.evictions > 0
        _assert_clean(c, before)

    def test_dropping_the_old_adapter_under_reads(self, churn):
        c, before = churn
        _run(c, [_evictor(c, close=False)])
        assert c.evictions > 0
        _assert_clean(c, before)

    def test_closing_under_lock_free_reads(self, churn, monkeypatch):
        monkeypatch.setenv("BIOPB_OMETIFF_PARALLEL_READ", "1")
        c, before = churn
        _run(c, [_evictor(c, close=True)])
        assert c.evictions > 0
        _assert_clean(c, before)


class TestEvictionAgainstTheReaper:
    @pytest.fixture(autouse=True)
    def _tight_reaper(self):
        reaper = ome_tiff_module._store_reaper
        ttl, cap = reaper._pool_ttl, reaper._max_handles
        reaper.set_ttl(0.0005)
        reaper._max_handles = 1
        yield
        reaper.set_ttl(ttl)
        reaper._max_handles = cap

    def _sweeper(self, stop):
        reaper = ome_tiff_module._store_reaper
        while not stop.is_set():
            reaper._sweep()
            reaper._close_over_cap()

    def test_sweeps_and_a_one_handle_cap_under_reads(self, churn):
        c, before = churn
        _run(c, [self._sweeper])
        _assert_clean(c, before)

    def test_sweeps_while_the_adapter_is_closed_and_rebuilt(self, churn):
        c, before = churn
        _run(c, [self._sweeper, _evictor(c, close=True)])
        assert c.evictions > 0
        _assert_clean(c, before)

    def test_sweeps_while_the_adapter_is_dropped_and_rebuilt(self, churn):
        c, before = churn
        _run(c, [self._sweeper, _evictor(c, close=False)])
        assert c.evictions > 0
        _assert_clean(c, before)


class TestFinalizersRacingTheReaper:
    """A dropped adapter's ``__del__`` releases its handle and takes the reaper's
    (non-reentrant) lock. When the cycle collector runs it inside a block that
    holds that lock, the thread would wait on itself."""

    def test_adapters_collected_by_the_cycle_collector_under_registration(self, churn):
        c, before = churn
        thresholds = gc.get_threshold()
        gc.set_threshold(1, 1, 1)  # collect at almost every allocation

        def make_cyclic_garbage(stop):
            while not stop.is_set():
                source = c.rebuild()
                source._cycle = source  # only the cycle collector can free it
                c.read(c.scene(source))  # opens a handle and registers it

        def sweep(stop):
            reaper = ome_tiff_module._store_reaper
            while not stop.is_set():
                reaper._sweep()
                reaper._close_over_cap()

        try:
            _run(c, [make_cyclic_garbage, make_cyclic_garbage, sweep])
        finally:
            gc.set_threshold(*thresholds)
        _assert_clean(c, before)


class TestHydrationUnderEviction:
    def test_concurrent_hydration_reads_never_miss_a_row_that_exists(self, churn):
        """Eviction makes hydration reads concurrent. On the shared write
        connection a small share of them came back empty."""
        c, _ = churn
        misses = []

        def read_rows():
            for _ in range(300):
                if c.db.read_hydration("src") is None:
                    misses.append(1)

        threads = [threading.Thread(target=read_rows) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(JOIN_TIMEOUT)
        assert not misses
