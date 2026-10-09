"""Evictable sources: the registry lets go of an idle adapter and the next read
rebuilds it from the source's row."""

import gc
import sys
import time

import pytest
from biopb_tensor_server.sources.source_registry import SourceRegistry

from tests import deferred_registration_test as drt, restore_test as rt


class _Adapter:
    source_id = "s"

    def close(self):
        pass


class _Slotted:
    __slots__ = ()

    def close(self):
        pass


def _descriptors():
    """Open descriptors of this process, where /proc says (None elsewhere)."""
    if not sys.platform.startswith("linux"):
        return None
    import os

    return len(os.listdir("/proc/self/fd"))


def _release(registry, idle=0.0):
    return registry.release_idle(idle, now=1e12)


class TestRegistry:
    def test_an_idle_evictable_adapter_goes_with_its_last_reader(self):
        registry = SourceRegistry()
        registry.register("a", _Adapter(), evictable=True)

        assert _release(registry) == 1
        gc.collect()

        assert registry.get("a") is None
        assert "a" in registry and len(registry) == 1 and list(registry) == ["a"]
        assert registry.values() == [] and registry.snapshot() == []

    def test_a_reader_keeps_the_adapter_and_a_read_takes_it_back(self):
        registry = SourceRegistry()
        held = registry.register("a", _Adapter(), evictable=True)
        _release(registry)

        assert registry.get("a") is held
        assert registry.get_registered("a") is held  # a read: held strongly again
        assert _release(registry, idle=1e13) == 0
        assert registry.get("a") is held

    def test_get_does_not_count_as_a_read(self):
        registry = SourceRegistry()
        registry.register("a", _Adapter(), evictable=True)
        registry._sources["a"].last_access = 0.0

        registry.get("a")

        assert registry._sources["a"].last_access == 0.0

    def test_only_an_evictable_source_is_released(self):
        registry = SourceRegistry()
        registry.register("pinned", _Adapter())
        registry.register("slotted", _Slotted(), evictable=True)  # no weakref

        assert _release(registry) == 0
        assert registry.get("pinned") is not None
        assert registry.get("slotted") is not None

    def test_a_recent_read_is_not_idle(self):
        registry = SourceRegistry()
        registry.register("a", _Adapter(), evictable=True)

        assert registry.release_idle(60.0) == 0

    def test_unregister_forgets_an_evicted_source(self):
        registry = SourceRegistry()
        registry.register("a", _Adapter(), evictable=True)
        _release(registry)
        gc.collect()

        registry.unregister("a")

        assert "a" not in registry and len(registry) == 0

    def test_reregistering_pins_an_evicted_source_again(self):
        registry = SourceRegistry()
        registry.register("a", _Adapter(), evictable=True)
        _release(registry)
        registry.register("a", _Adapter())

        assert _release(registry) == 0
        assert registry.get("a") is not None


class TestRebuildAfterRelease:
    def _run(self, tmp_path):
        rt._first_run(tmp_path)
        run = rt._Run(tmp_path)
        run.restore()
        sid = sorted(run.rows())[0]
        first = run.server.sources.get_registered(sid)
        return run, sid, first

    def _let_go(self, run, sid):
        assert run.server.sources.release_idle(0.0, now=1e12) >= 1
        gc.collect()
        assert run.server.sources.get(sid) is None

    def test_a_released_source_is_rebuilt_by_the_next_read(self, tmp_path):
        run, sid, first = self._run(tmp_path)
        del first
        self._let_go(run, sid)

        assert sid in run.server.sources
        assert run.server.sources.get_registered(sid) is not None
        assert not run.reconciler.is_pending(sid)
        run.stop()

    def test_the_source_count_does_not_move_while_it_is_let_go(self, tmp_path):
        run, sid, first = self._run(tmp_path)
        total = len(run.server.sources) + run.reconciler.unregistered_count()
        del first
        self._let_go(run, sid)

        assert len(run.server.sources) + run.reconciler.unregistered_count() == total
        run.stop()

    def test_a_dropped_source_keeps_its_url_while_let_go(self, tmp_path):
        run = rt._Run(tmp_path)
        run.manager.complete_initial_scan()
        path = drt._make_zarr(tmp_path, "dropped.zarr")
        for _ in run.manager.add_local_source(str(path)):
            pass
        (sid,) = run.rows()
        url = run.reconciler.catalog_url_of(sid)
        assert url.startswith("dnd://")
        self._let_go(run, sid)

        assert run.reconciler.catalog_url_of(sid) == url
        assert run.server.sources.get_registered(sid).catalog_url == url
        run.stop()

    def test_a_removed_source_is_not_rebuilt(self, tmp_path):
        run, sid, first = self._run(tmp_path)
        del first
        self._let_go(run, sid)

        run.reconciler._commit_remove_source(sid)

        assert run.server.sources.get_registered(sid) is None
        assert sid not in run.server.sources
        run.stop()

    def test_a_rescan_tick_lets_idle_adapters_go(self, tmp_path):
        run, sid, first = self._run(tmp_path)
        del first
        run.manager._adapter_idle_seconds = 0.01
        time.sleep(0.1)  # the monotonic clock is coarse on Windows

        run.manager._release_idle_adapters()
        gc.collect()

        assert run.server.sources.get(sid) is None
        run.stop()

    def test_an_idle_time_of_zero_keeps_every_adapter(self, tmp_path):
        run, sid, first = self._run(tmp_path)

        run.manager._release_idle_adapters()

        assert run.server.sources.get(sid) is first
        run.stop()


class TestFormatsAreCollected:
    """Nothing else holds a read adapter: after a release it is collected, its
    handle goes with it, and the next read rebuilds it and returns the same bytes."""

    @pytest.mark.parametrize("name", ["plain.tif", "scene.ome.tif"])
    def test_a_tiff_family_source_after_a_read(self, tmp_path, name):
        import numpy as np
        import tifffile
        from biopb.tensor.ticket_pb2 import ChunkBounds

        def read(sid, run):
            scene = run.server.sources.get_registered(sid).get_tensor_adapter(
                run.server.sources.get_registered(sid).list_tensors()[0].array_id
            )
            shape = list(scene.get_tensor_descriptor().shape)
            stop = [min(s, 8) for s in shape]
            return np.asarray(
                scene.get_data(ChunkBounds(start=[0] * len(shape), stop=stop))
            )

        run = rt._Run(tmp_path)
        data = np.arange(4 * 64 * 64, dtype=np.uint16).reshape(4, 64, 64)
        kwargs = {"ome": True} if name.endswith("ome.tif") else {}
        tifffile.imwrite(
            run.monitored / name,
            data,
            tile=(16, 16),
            photometric="minisblack",
            **kwargs,
        )
        run.manager._handle_rescan()
        (sid,) = run.rows()
        run.stop()

        run = rt._Run(tmp_path)
        run.restore()
        before = _descriptors()
        first = read(sid, run)
        assert run.server.sources.release_idle(0.0, now=1e12) == 1
        gc.collect()

        assert run.server.sources.get(sid) is None
        if before is not None:
            assert _descriptors() <= before + 2  # the pooled handle
        assert np.array_equal(read(sid, run), first)
        run.stop()
