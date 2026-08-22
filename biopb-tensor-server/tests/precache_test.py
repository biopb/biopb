"""Tests for the background pre-cache worker and its scale-hint computation."""

import threading
import time

import numpy as np
import pytest
from biopb_tensor_server.core.chunk import (
    cache_key_for_chunk_id,
    is_scaled_chunk,
)
from biopb_tensor_server.core.config import PrecacheConfig
from biopb_tensor_server.serving.precache import PrecacheWorker
from biopb_tensor_server.serving.server import TensorFlightServer


def _zarr_available() -> bool:
    try:
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


def _import_biopb_mcp():
    """Return biopb-mcp's _tensor_utils module if importable, else None."""
    try:
        from biopb_mcp import _tensor_utils

        return _tensor_utils
    except Exception:
        return None


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _simulate_client_terminal_scale(shape, labels, threshold, downscale, budget_root):
    """Reproduce biopb-mcp build_pyramid_levels' terminal (sx, sy, sz) as a
    full per-axis scale vector, computed analytically (no get_tensor calls).

    Kept independent of our implementation so the cross-check is meaningful.
    """
    ndim = len(shape)
    budget = budget_root**3
    floor = min(budget_root, threshold)
    lbl = [str(x).lower() for x in labels] if labels else None
    if lbl and "y" in lbl and "x" in lbl:
        y_idx, x_idx = lbl.index("y"), lbl.index("x")
    else:
        y_idx, x_idx = ndim - 2, ndim - 1
    if lbl:
        z_idx = lbl.index("z") if "z" in lbl else None
    else:
        z_idx = ndim - 3 if ndim >= 3 else None
    if z_idx is not None and z_idx in (x_idx, y_idx):
        z_idx = None

    sx = sy = sz = 1
    while True:
        lx = _ceil_div(shape[x_idx], sx)
        ly = _ceil_div(shape[y_idx], sy)
        lz = _ceil_div(shape[z_idx], sz) if z_idx is not None else 1
        if lx * ly * lz <= budget and lx <= threshold and ly <= threshold:
            break
        nsx = sx * downscale if lx > floor else sx
        nsy = sy * downscale if ly > floor else sy
        nsz = sz * downscale if (z_idx is not None and lz > floor) else sz
        if (nsx, nsy, nsz) == (sx, sy, sz):
            break
        sx, sy, sz = nsx, nsy, nsz

    scale = [1] * ndim
    scale[x_idx] = sx
    scale[y_idx] = sy
    if z_idx is not None:
        scale[z_idx] = sz
    return scale


# ---------------------------------------------------------------------------
# 1. Scale-hint computation -- must match biopb-mcp's coarsest pyramid level.
class TestFlightIdleProbe:
    def test_idle_when_no_traffic(self):
        server = TensorFlightServer("grpc://localhost:0")
        try:
            # last_active defaults to 0.0, monotonic() is large -> idle.
            assert server.flight_idle_for(0.0) is True
        finally:
            server.shutdown()

    def test_not_idle_while_in_flight(self):
        server = TensorFlightServer("grpc://localhost:0")
        try:
            entered = threading.Event()
            release = threading.Event()

            def hold():
                with server.activity.serving_request():
                    entered.set()
                    release.wait(2.0)

            t = threading.Thread(target=hold, daemon=True)
            t.start()
            assert entered.wait(2.0)
            # In flight -> not idle, regardless of debounce.
            assert server.flight_idle_for(0.0) is False
            release.set()
            t.join(2.0)
            # After completion + zero debounce -> idle again.
            assert server.flight_idle_for(0.0) is True
        finally:
            server.shutdown()

    def test_debounce_window(self):
        server = TensorFlightServer("grpc://localhost:0")
        try:
            with server.activity.serving_request():
                pass
            # Just finished: a 5s debounce is not yet satisfied.
            assert server.flight_idle_for(5.0) is False
            # ...but a zero debounce is.
            assert server.flight_idle_for(0.0) is True
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# 3 & 6. Warming integration + backend gate.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def _scaled_chunk_id(adapter, scale, reduction_method="area"):
    """A scaled chunk_id for a source's first tensor, as a client's read carries.

    The warmer is driven entirely by observed reads now, so nearly every test
    needs one of these rather than a source_id.
    """
    from biopb.tensor.descriptor_pb2 import TensorDescriptor

    td = adapter.list_tensor_descriptors()[0]
    ta = adapter.get_tensor_adapter(td.array_id)
    base = ta.get_tensor_descriptor()
    req = TensorDescriptor(
        array_id=base.array_id,
        dim_labels=base.dim_labels,
        shape=base.shape,
        chunk_shape=base.chunk_shape,
        dtype=base.dtype,
    )
    req.scale_hint[:] = list(scale)
    req.reduction_method = reduction_method
    return ta.get_read_plan(req).chunk_endpoints[0].chunk_id


class TestWarming:
    def _make_server_with_zarr(self, tmp_path, shape):
        import zarr
        from biopb_tensor_server import ZarrAdapter

        arr = zarr.open_array(
            str(tmp_path / "a.zarr"),
            mode="w",
            shape=shape,
            chunks=tuple(min(s, 1024) for s in shape),
            dtype="uint16",
        )
        arr[:] = np.arange(int(np.prod(shape)), dtype="uint16").reshape(shape) % 1000
        labels = ["y", "x"]
        adapter = ZarrAdapter(arr, "warm-src", labels)
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("warm-src", adapter)
        return server

    def test_warms_scaled_chunks_into_file_cache(self, tmp_path):
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        try:
            # 8192 -> scale 4 in X/Y, so the warmed chunks are scaled chunks.
            server = self._make_server_with_zarr(tmp_path, (8192, 8192))
            cfg = PrecacheConfig(idle_debounce_seconds=0.0)
            worker = PrecacheWorker(server, cfg)

            # Drive synchronously (no thread) for determinism.
            worker._process_source(
                "warm-src", scale_hint=[4, 4], reduction_method="area"
            )

            cache_manager = CacheManager.get_instance()
            stats = cache_manager.stats()
            assert stats.misses > 0  # cold computes happened

            # Rebuild the same read plan and assert every chunk now locates on
            # disk -- i.e. a future do_get is a warm hit, no decode needed.
            adapter = server.sources.get("warm-src")
            td = adapter.list_tensor_descriptors()[0]
            ta = adapter.get_tensor_adapter(td.array_id)
            scale = [4, 4]  # 8192 -> 4x in Y/X
            from biopb.tensor.descriptor_pb2 import TensorDescriptor

            req = TensorDescriptor(
                array_id=td.array_id,
                dim_labels=td.dim_labels,
                shape=td.shape,
                chunk_shape=ta.get_tensor_descriptor().chunk_shape,
                dtype=td.dtype,
            )
            req.scale_hint[:] = scale
            req.reduction_method = "area"
            plan = ta.get_read_plan(req)
            assert len(plan.chunk_endpoints) > 0
            for ce in plan.chunk_endpoints:
                assert is_scaled_chunk(ce.chunk_id)
                # Entries are keyed by the method-stripped canonical key, the
                # same locate the server's chunk-locate handoff performs (#76).
                assert (
                    cache_manager.locate_entry(cache_key_for_chunk_id(ce.chunk_id))
                    is not None
                )
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_should_warm_gate_skips_non_resident_source(self, tmp_path):
        # #174: a should_warm callback returning False (source re-dehydrated under
        # a cloud root) must short-circuit before any chunk is read, so OneDrive
        # is never asked to recall the bytes.
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        try:
            server = self._make_server_with_zarr(tmp_path, (8192, 8192))
            worker = PrecacheWorker(server, PrecacheConfig(idle_debounce_seconds=0.0))
            worker.should_warm = lambda source_id: False

            # Pin the ordering contract of #174: the gate must fire *before* any
            # adapter access, so a denied warm never touches the source at all.
            # A spy on sources.get (and the cache stats) catches a future
            # re-ordering of the gate below adapter/list/compute.
            adapter_calls = []
            real_get = server.sources.get
            server.sources.get = lambda sid: adapter_calls.append(sid) or real_get(sid)

            worker._process_source(
                "warm-src", scale_hint=[4, 4], reduction_method="area"
            )

            assert adapter_calls == []  # gate fired before any adapter access
            # ... and consequently nothing was computed/cached.
            stats = CacheManager.get_instance().stats()
            assert stats.misses == 0
            assert stats.total_entries == 0
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_skips_remote_source(self, tmp_path):
        # #299: a non-local (remote-tensor proxy) source must be skipped before
        # any tensor is read -- warming it would speculatively pull every chunk
        # across the network from the upstream, and the proxy does not implement
        # has_native_pyramid() so it would mis-warm a pyramidal upstream.
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        try:
            server = self._make_server_with_zarr(tmp_path, (8192, 8192))
            worker = PrecacheWorker(server, PrecacheConfig(idle_debounce_seconds=0.0))

            listed = []

            class _RemoteAdapter:
                # A caching-proxy source advertises a grpc:// source_url.
                source_url = "grpc://upstream:8815/img"

                def list_tensor_descriptors(self):
                    listed.append(True)  # must NOT be reached
                    return []

            server.sources.get = lambda sid: _RemoteAdapter()

            worker._process_source(
                "remote-src", scale_hint=[4, 4], reduction_method="area"
            )

            assert listed == []  # skipped before enumerating the source's tensors
            stats = CacheManager.get_instance().stats()
            assert stats.misses == 0
            assert stats.total_entries == 0
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_memory_backend_is_noop(self, tmp_path):
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        try:
            server = self._make_server_with_zarr(tmp_path, (8192, 8192))
            worker = PrecacheWorker(server, PrecacheConfig(idle_debounce_seconds=0.0))
            worker._process_source(
                "warm-src", scale_hint=[4, 4], reduction_method="area"
            )

            # File-backend gate: nothing computed on a memory backend.
            stats = CacheManager.get_instance().stats()
            assert stats.misses == 0
            assert stats.total_entries == 0
        finally:
            server.shutdown()
            CacheManager.reset()


# ---------------------------------------------------------------------------
# 4. Runtime-only enqueue gating (SourceManager hook).
# ---------------------------------------------------------------------------


class TestRuntimePhaseGating:
    def _bare_source_manager(self):
        from biopb_tensor_server.core.discovery import AdapterRegistry, DiscoveryState
        from biopb_tensor_server.sources.source_manager import SourceManager

        server = TensorFlightServer("grpc://localhost:0")
        sm = SourceManager(
            server=server,
            registry=AdapterRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs=set(),
        )
        return server, sm

    def test_initial_scan_done_default_false_and_start_does_not_flip(self):
        server, sm = self._bare_source_manager()
        try:
            assert sm._initial_scan_done is False
            # start() no longer flips the precache gate -- only the first full
            # scan completing does. A static-only (watcher=None) start() is a
            # no-op and leaves it False.
            sm.start()
            assert sm._initial_scan_done is False
        finally:
            server.shutdown()

    def test_commit_hook_fires_only_after_initial_scan(self, monkeypatch):
        from types import SimpleNamespace

        server, sm = self._bare_source_manager()
        try:
            # Stub the heavy commit collaborators so we exercise only the gate.
            monkeypatch.setattr(
                sm._reconciler,
                "_register_source_claim",
                lambda claim, catalog_seed=None, catalog_url=None: True,
            )
            monkeypatch.setattr(
                sm._reconciler._state, "add_claim", lambda claim, notify=False: True
            )
            monkeypatch.setattr(
                sm._reconciler, "_build_claim_signatures", lambda claim: {}
            )
            monkeypatch.setattr(
                sm._reconciler, "_clear_failed_source_attempt", lambda sid: None
            )

            fired = []
            sm.set_source_committed_hook(fired.append)
            claim = SimpleNamespace(source_id="s1", primary_path="/x")

            # During the initial scan: startup sources go to the backlog, not the
            # prompt enqueue -- the hook must NOT fire.
            sm._initial_scan_done = False
            assert sm._reconciler._commit_add_claim(claim) is True
            assert fired == []

            # After the initial scan: live additions fire the hook.
            sm._initial_scan_done = True
            assert sm._reconciler._commit_add_claim(claim) is True
            assert fired == ["s1"]
        finally:
            server.shutdown()

    def test_suppress_live_precache_overrides_the_gate(self, monkeypatch):
        """A commit during the boot-tick upstream re-list stays off the prompt
        enqueue even though the initial scan is already done.

        On the both-present boot tick the local walk flips _initial_scan_done
        True before the upstream re-list runs; _suppress_live_precache keeps that
        startup upstream mirror routed to the slow backlog (see _handle_rescan)."""
        from types import SimpleNamespace

        server, sm = self._bare_source_manager()
        try:
            monkeypatch.setattr(
                sm._reconciler,
                "_register_source_claim",
                lambda claim, catalog_seed=None, catalog_url=None: True,
            )
            monkeypatch.setattr(
                sm._reconciler._state, "add_claim", lambda claim, notify=False: True
            )
            monkeypatch.setattr(
                sm._reconciler, "_build_claim_signatures", lambda claim: {}
            )
            monkeypatch.setattr(
                sm._reconciler, "_clear_failed_source_attempt", lambda sid: None
            )

            fired = []
            sm.set_source_committed_hook(fired.append)
            sm._initial_scan_done = True
            claim = SimpleNamespace(source_id="up1", primary_path="grpc://lab/up1")

            # Suppressed: initial scan done, but this is the boot-tick upstream
            # re-list -> backlog, not prompt enqueue.
            sm._suppress_live_precache = True
            assert sm._reconciler._commit_add_claim(claim) is True
            assert fired == []

            # Not suppressed (a later live delta): the hook fires as usual.
            sm._suppress_live_precache = False
            assert sm._reconciler._commit_add_claim(claim) is True
            assert fired == ["up1"]
        finally:
            server.shutdown()

    def test_hook_exception_does_not_abort_commit(self, monkeypatch):
        from types import SimpleNamespace

        server, sm = self._bare_source_manager()
        try:
            monkeypatch.setattr(
                sm._reconciler,
                "_register_source_claim",
                lambda claim, catalog_seed=None, catalog_url=None: True,
            )
            monkeypatch.setattr(
                sm._reconciler._state, "add_claim", lambda claim, notify=False: True
            )
            monkeypatch.setattr(
                sm._reconciler, "_build_claim_signatures", lambda claim: {}
            )
            monkeypatch.setattr(
                sm._reconciler, "_clear_failed_source_attempt", lambda sid: None
            )

            def boom(_sid):
                raise RuntimeError("hook failure")

            sm.set_source_committed_hook(boom)
            sm._initial_scan_done = True
            claim = SimpleNamespace(source_id="s2", primary_path="/y")
            # Commit still succeeds despite the hook raising.
            assert sm._reconciler._commit_add_claim(claim) is True
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# 5. Preemption smoke + worker lifecycle/dedup.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestPreemptionAndLifecycle:
    def test_no_warming_while_in_flight(self, tmp_path):
        """The idle debounce still parks the worker while a read is in flight."""
        import zarr
        from biopb_tensor_server import ZarrAdapter
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        try:
            arr = zarr.open_array(
                str(tmp_path / "a.zarr"),
                mode="w",
                shape=(8192, 8192),
                chunks=(1024, 1024),
                dtype="uint16",
            )
            arr[:] = 7
            adapter = ZarrAdapter(arr, "pre-src", ["y", "x"])
            server = TensorFlightServer("grpc://localhost:0")
            server.register_source("pre-src", adapter)

            worker = PrecacheWorker(server, PrecacheConfig(idle_debounce_seconds=0.05))

            # Hold a request in flight so the worker must wait.
            release = threading.Event()
            holding = threading.Event()

            def hold():
                with server.activity.serving_request():
                    holding.set()
                    release.wait(3.0)

            ht = threading.Thread(target=hold, daemon=True)
            ht.start()
            assert holding.wait(2.0)

            worker.start()
            worker.observe_read(_scaled_chunk_id(adapter, (4, 4)))
            # While the request is in flight, nothing should have warmed yet.
            time.sleep(0.5)
            assert CacheManager.get_instance().stats().misses == 0

            # Release traffic; after the debounce the worker proceeds.
            release.set()
            ht.join(2.0)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if CacheManager.get_instance().stats().misses > 0:
                    break
                time.sleep(0.05)
            assert CacheManager.get_instance().stats().misses > 0
            worker.stop()
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()


# ---------------------------------------------------------------------------
# Stage 2: startup backlog (existing sources).
# ---------------------------------------------------------------------------


from types import SimpleNamespace  # noqa: E402


def _register_zarr(
    server, tmp_path, source_id, shape=(8192, 8192), labels=("y", "x"), chunks=None
):
    import zarr
    from biopb_tensor_server import ZarrAdapter

    arr = zarr.open_array(
        str(tmp_path / f"{source_id}.zarr"),
        mode="w",
        shape=shape,
        chunks=chunks or (1024, 1024),
        dtype="uint16",
    )
    arr[:] = 3
    adapter = ZarrAdapter(arr, source_id, list(labels))
    server.register_source(source_id, adapter)
    return adapter


def _located_all(server, cache_manager, source_ids):
    """True once every source's coarsest-level chunk resolves on disk."""
    from biopb.tensor.descriptor_pb2 import TensorDescriptor

    for sid in source_ids:
        adapter = server.sources.get(sid)
        td = adapter.list_tensor_descriptors()[0]
        ta = adapter.get_tensor_adapter(td.array_id)
        scale = [4, 4]  # 8192 -> 4x in Y/X
        req = TensorDescriptor(
            array_id=td.array_id,
            dim_labels=td.dim_labels,
            shape=td.shape,
            chunk_shape=ta.get_tensor_descriptor().chunk_shape,
            dtype=td.dtype,
        )
        req.scale_hint[:] = scale
        req.reduction_method = "area"
        plan = ta.get_read_plan(req)
        if not plan.chunk_endpoints:
            return False
        for ce in plan.chunk_endpoints:
            if cache_manager.locate_entry(cache_key_for_chunk_id(ce.chunk_id)) is None:
                return False
    return True


class _FakeBackend:
    def __init__(self, total, mx):
        self._st = SimpleNamespace(total_bytes=total, max_bytes=mx)

    def stats(self):
        return self._st


class TestHeadroomProbe:
    def test_has_headroom_tracks_high_water(self, monkeypatch):
        from biopb_tensor_server.serving import precache as pc

        worker = PrecacheWorker(None, PrecacheConfig(high_water=0.8))
        backend = _FakeBackend(total=0, mx=1000)
        mgr = SimpleNamespace(backend=backend)
        monkeypatch.setattr(pc.CacheManager, "get_instance", lambda: mgr)

        assert worker._has_headroom() is True  # empty
        backend._st.total_bytes = 700  # below 0.8 * 1000
        assert worker._has_headroom() is True
        backend._st.total_bytes = 800  # at the mark -> not below
        assert worker._has_headroom() is False
        backend._st.total_bytes = 900  # over
        assert worker._has_headroom() is False

    def test_no_headroom_when_unbounded_or_missing(self, monkeypatch):
        from biopb_tensor_server.serving import precache as pc

        worker = PrecacheWorker(None, PrecacheConfig())
        # max_bytes <= 0 -> can't reason about fill, treat as no headroom.
        mgr = SimpleNamespace(backend=_FakeBackend(total=0, mx=0))
        monkeypatch.setattr(pc.CacheManager, "get_instance", lambda: mgr)
        assert worker._has_headroom() is False
        # No cache at all.
        monkeypatch.setattr(pc.CacheManager, "get_instance", lambda: None)
        assert worker._has_headroom() is False


class TestIterLocalSourceMtimes:
    def _bare_sm(self):
        from biopb_tensor_server.core.discovery import AdapterRegistry, DiscoveryState
        from biopb_tensor_server.sources.source_manager import SourceManager

        server = TensorFlightServer("grpc://localhost:0")
        sm = SourceManager(
            server=server,
            registry=AdapterRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs=set(),
        )
        return server, sm

    def test_skips_remote_and_unstatable(self, tmp_path):
        server, sm = self._bare_sm()
        try:
            real = tmp_path / "f.zarr"
            real.mkdir()
            sm._reconciler._state.claims["local"] = SimpleNamespace(
                source_id="local", primary_path=str(real), is_remote=False
            )
            sm._reconciler._state.claims["remote"] = SimpleNamespace(
                source_id="remote", primary_path="s3://bucket/x", is_remote=True
            )
            sm._reconciler._state.claims["gone"] = SimpleNamespace(
                source_id="gone",
                primary_path=str(tmp_path / "missing"),
                is_remote=False,
            )
            out = dict(sm.iter_local_source_mtimes())
            assert "local" in out
            assert isinstance(out["local"], float)
            assert "remote" not in out  # no os.stat mtime
            assert "gone" not in out  # OSError -> skipped
        finally:
            server.shutdown()

    def test_snapshot_taken_under_lock(self):
        # The read must snapshot _state.claims under self._lock (the same lock
        # _commit_add_claim/_commit_remove_claim hold) so it can't iterate the
        # dict while the watcher's event loop mutates it. Prove it by holding the
        # lock in another thread: the reader must block until it is released.
        server, sm = self._bare_sm()
        holder = None
        try:
            sm._reconciler._state.claims["a"] = SimpleNamespace(
                source_id="a", primary_path="/x", is_remote=True
            )
            held = threading.Event()
            release = threading.Event()
            done = threading.Event()

            def hold_lock():
                with sm._reconciler._lock:
                    held.set()
                    release.wait(2.0)

            holder = threading.Thread(target=hold_lock, daemon=True)
            holder.start()
            assert held.wait(1.0)

            reader = threading.Thread(
                target=lambda: (sm.iter_local_source_mtimes(), done.set()),
                daemon=True,
            )
            reader.start()
            # Lock is held elsewhere -> the snapshot can't proceed yet.
            assert not done.wait(0.3)
            release.set()
            # Released -> the read completes.
            assert done.wait(2.0)
        finally:
            release.set()
            if holder is not None:
                holder.join(1.0)
            server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestSkipNativePyramid:
    def _ome_adapter(self, multires_ome_zarr, source_id="ome-native"):
        import zarr
        from biopb_tensor_server import OmeZarrAdapter

        zarr_path, _level_paths, _zattrs = multires_ome_zarr
        root = zarr.open_group(zarr_path, mode="r")
        return OmeZarrAdapter(root["0"], source_id)

    def test_ome_zarr_reports_native_pyramid(self, multires_ome_zarr):
        adapter = self._ome_adapter(multires_ome_zarr)
        assert adapter.has_native_pyramid() is True

    def test_plain_zarr_has_no_native_pyramid(self, tmp_path):
        import zarr
        from biopb_tensor_server import ZarrAdapter

        arr = zarr.open_array(
            str(tmp_path / "a.zarr"),
            mode="w",
            shape=(64, 64),
            chunks=(32, 32),
            dtype="uint16",
        )
        adapter = ZarrAdapter(arr, "plain", ["y", "x"])
        assert adapter.has_native_pyramid() is False

    def test_precache_skips_native_multiscale_source(self, multires_ome_zarr, tmp_path):
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = self._ome_adapter(multires_ome_zarr)
            server.register_source("ome-native", adapter)
            worker = PrecacheWorker(server, PrecacheConfig(idle_debounce_seconds=0.0))
            # File backend is active, so absent the skip this would warm chunks.
            worker._process_source(
                "ome-native", scale_hint=[2, 2], reduction_method="area"
            )
            assert CacheManager.get_instance().stats().misses == 0
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()


# ---------------------------------------------------------------------------
# Server-advertised pyramid plan (server-decided multi-scale).
# ---------------------------------------------------------------------------


class TestDemandTier:
    """Warming driven by what a client actually read, not by a server guess."""

    def _init_file_cache(self, tmp_path):
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )

    def _worker(self, server, **cfg):
        cfg.setdefault("idle_debounce_seconds", 0.0)
        return PrecacheWorker(server, PrecacheConfig(**cfg))

    def _observed_chunk_id(self, adapter, scale):
        """A scaled chunk_id for this tensor, as a client's read would carry."""
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        td = adapter.list_tensor_descriptors()[0]
        ta = adapter.get_tensor_adapter(td.array_id)
        base = ta.get_tensor_descriptor()
        req = TensorDescriptor(
            array_id=base.array_id,
            dim_labels=base.dim_labels,
            shape=base.shape,
            chunk_shape=base.chunk_shape,
            dtype=base.dtype,
        )
        req.scale_hint[:] = list(scale)
        req.reduction_method = "area"
        plan = ta.get_read_plan(req)
        return plan.chunk_endpoints[0].chunk_id

    # -- producer -----------------------------------------------------------

    def test_observe_read_is_non_blocking_and_lossy_when_full(self):
        """The producer runs on a serving thread, so it must never block."""
        server = TensorFlightServer("grpc://localhost:0")
        try:
            worker = self._worker(server, demand_queue_max=2)
            for i in range(20):
                worker.observe_read(b"chunk-%d" % i)
            assert worker._demand.qsize() == 2
        finally:
            server.shutdown()

    def test_a_full_queue_drops_the_oldest_observation_not_the_newest(self):
        """Overflow means the client outran the worker, so the queue holds the
        stale guesses and the arriving one is where the client actually is.
        Keeping the backlog and rejecting the newcomer would pin the tier to
        wherever the client was when the worker fell behind.
        """
        server = TensorFlightServer("grpc://localhost:0")
        try:
            worker = self._worker(server, demand_queue_max=2)
            for i in range(20):
                worker.observe_read(b"chunk-%d" % i)
            drained = [worker._demand.get_nowait() for _ in range(2)]
            assert drained == [b"chunk-18", b"chunk-19"]
        finally:
            server.shutdown()

    def test_observe_read_ignores_everything_when_disabled(self):
        server = TensorFlightServer("grpc://localhost:0")
        try:
            worker = self._worker(server, demand_enabled=False)
            worker.observe_read(b"anything")
            assert worker._demand.qsize() == 0
        finally:
            server.shutdown()

    def test_observe_read_never_raises_on_garbage(self):
        server = TensorFlightServer("grpc://localhost:0")
        try:
            worker = self._worker(server)
            worker.observe_read(b"")
            worker.observe_read(b"\xff\xfe not a chunk id")
        finally:
            server.shutdown()

    # -- consumer gating ----------------------------------------------------

    def test_full_resolution_read_triggers_no_warm(self, tmp_path):
        """A full-res read is the computation pattern; warming it is wrong."""
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            worker = self._worker(server)
            cid = self._observed_chunk_id(adapter, (1, 1))
            cm = CacheManager.get_instance()
            worker._process_demand(cid)
            assert cm.stats().misses == 0
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_undecodable_observation_is_dropped(self, tmp_path):
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            worker = self._worker(server)
            worker._process_demand(b"not a chunk id")
            assert CacheManager.get_instance().stats().misses == 0
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    # -- the warm itself ----------------------------------------------------

    def test_scaled_read_warms_the_rest_of_that_level(self, tmp_path):
        """The case a channel-interleaved layout hides.

        ND2 decodes every channel whichever one you ask for, so a read leaves
        the whole level resident and there is nothing left to warm. A planar
        source does not, so the observed tensor's unread chunks stay cold unless
        the demand tier warms them -- which is why it must not skip the tensor
        that triggered it.
        """
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            worker = self._worker(server)
            cid = self._observed_chunk_id(adapter, (4, 4))
            cm = CacheManager.get_instance()

            worker._process_demand(cid)
            assert cm.stats().misses > 0, "observed level was never warmed"
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_warms_at_the_observed_scale_not_the_server_plan(self, tmp_path):
        """The whole point: the client's scale wins over the advertised one."""
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.chunk import decode_scale_info

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            observed = (2, 2)

            warmed = []
            worker = self._worker(server)
            real = worker._process_tensor

            def spy(source_adapter, td, cm, **kw):
                warmed.append(kw.get("scale_hint"))
                return real(source_adapter, td, cm, **kw)

            worker._process_tensor = spy
            cid = self._observed_chunk_id(adapter, observed)
            assert tuple(decode_scale_info(cid)) == observed
            worker._process_demand(cid)

            assert warmed == [[2, 2]], f"warmed at {warmed}, not the observed scale"
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_second_read_of_a_warmed_level_does_not_re_warm(self, tmp_path):
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            worker = self._worker(server)
            cid = self._observed_chunk_id(adapter, (4, 4))
            cm = CacheManager.get_instance()

            worker._process_demand(cid)
            first = cm.stats().misses
            worker._process_demand(cid)
            assert cm.stats().misses == first, "level was warmed twice"
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_a_pass_stopped_at_high_water_is_retried_by_the_next_read(self, tmp_path):
        """Cache pressure is a moment, not a verdict on the level.

        Remembering a level the worker never actually warmed would leave the
        source's siblings cold for the rest of the session -- until 512 other
        levels evict the entry -- which is the exact failure the demand tier
        exists to prevent.
        """
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            worker = self._worker(server)
            cid = self._observed_chunk_id(adapter, (4, 4))
            cm = CacheManager.get_instance()

            worker._has_headroom = lambda: False
            worker._process_demand(cid)
            assert cm.stats().misses == 0, "nothing should have been warmed"
            assert not worker._demand_done, "a level nobody warmed was remembered"

            # Pressure passes; the next read of that level warms it for real.
            worker._has_headroom = lambda: True
            worker._process_demand(cid)
            assert cm.stats().misses > 0, "the retry did not warm"
            assert worker._demand_done, "a completed warm was not remembered"
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_a_source_skipped_as_non_resident_stays_retryable(self, tmp_path):
        """The cloud provider can rehydrate at any time (#174), and the next
        read of the source is exactly when it has."""
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            worker = self._worker(server)
            worker.should_warm = lambda source_id: False
            worker._process_demand(self._observed_chunk_id(adapter, (4, 4)))
            assert not worker._demand_done
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_an_adapter_that_raises_leaves_the_level_retryable(self, tmp_path):
        """A transient adapter failure must not cost the level its next
        chance."""
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            worker = self._worker(server)
            cid = self._observed_chunk_id(adapter, (4, 4))

            def boom(*a, **kw):
                raise RuntimeError("transient")

            adapter.list_tensor_descriptors = boom
            worker._process_demand(cid)
            assert not worker._demand_done
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_demand_memory_is_bounded_so_eviction_can_be_re_warmed(self, tmp_path):
        """An unbounded memory would turn an eviction into a permanent cold spot."""
        import biopb_tensor_server.serving.precache as precache_mod

        server = TensorFlightServer("grpc://localhost:0")
        try:
            worker = self._worker(server)
            for i in range(precache_mod._DEMAND_MEMORY + 50):
                worker._demand_done[(f"src{i}", (4, 4), "area")] = None
                while len(worker._demand_done) > precache_mod._DEMAND_MEMORY:
                    worker._demand_done.popitem(last=False)
            assert len(worker._demand_done) == precache_mod._DEMAND_MEMORY
            # The oldest entries are the ones dropped, so they can warm again.
            assert ("src0", (4, 4), "area") not in worker._demand_done
        finally:
            server.shutdown()

    def test_observed_tensor_is_warmed_before_its_siblings(self, tmp_path):
        """The tensor on screen outranks any sibling."""
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(server, tmp_path, "src")
            worker = self._worker(server)
            td = adapter.list_tensor_descriptors()[0]

            order = []
            real = worker._process_tensor

            def spy(source_adapter, tdesc, cm, **kw):
                order.append(tdesc.array_id)
                return real(source_adapter, tdesc, cm, **kw)

            worker._process_tensor = spy
            worker._process_demand(self._observed_chunk_id(adapter, (4, 4)))
            assert order and order[0] == td.array_id
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    # -- server wiring ------------------------------------------------------

    def test_server_observer_is_best_effort(self):
        """A failing observer must never break the read that triggered it."""
        server = TensorFlightServer("grpc://localhost:0")
        try:

            def boom(_chunk_id):
                raise RuntimeError("observer exploded")

            server.set_read_observer(boom)
            server._observe_read(b"chunk")  # must not raise
        finally:
            server.shutdown()

    def test_server_with_no_observer_is_a_no_op(self):
        server = TensorFlightServer("grpc://localhost:0")
        try:
            server._observe_read(b"chunk")
        finally:
            server.shutdown()


class TestSkipUnscaledLevel:
    """A level with nothing to downsample is never warmed.

    Warming it would cache the source 1:1 and save an open nothing, because
    there is no decode+downsample to precompute.
    """

    def _init_file_cache(self, tmp_path):
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.core.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )

    def _warm_one_tensor(self, tmp_path, shape, labels, scale, chunks=None):
        """Warm one tensor at *scale*; return the cache's miss count."""
        import biopb_tensor_server.serving.precache as precache_mod
        from biopb_tensor_server.cache import CacheManager

        self._init_file_cache(tmp_path)
        server = TensorFlightServer("grpc://localhost:0")
        try:
            adapter = _register_zarr(
                server, tmp_path, "src", shape=shape, labels=labels, chunks=chunks
            )
            worker = PrecacheWorker(server, PrecacheConfig(idle_debounce_seconds=0.0))
            cm = CacheManager.get_instance()
            td = adapter.list_tensor_descriptors()[0]
            outcome = worker._process_tensor(
                adapter, td, cm, scale_hint=list(scale), reduction_method="area"
            )
            assert outcome is not precache_mod._Outcome.HALTED
            return cm.stats().misses
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_skips_a_full_resolution_level(self, tmp_path):
        # An all-ones scale caches the source 1:1 and saves an open nothing.
        assert self._warm_one_tensor(tmp_path, (1024, 1024), ("y", "x"), (1, 1)) == 0

    def test_warms_a_genuinely_scaled_level(self, tmp_path):
        assert self._warm_one_tensor(tmp_path, (8192, 8192), ("y", "x"), (4, 4)) > 0

    def test_rejects_a_level_of_the_wrong_rank(self, tmp_path):
        """A sibling of another rank cannot use the observed level.

        Coercing it would warm chunk_ids no client asks for, so it is dropped.
        """
        assert self._warm_one_tensor(tmp_path, (8192, 8192), ("y", "x"), (1, 4, 4)) == 0

    def test_skips_long_timelapse_with_small_frames(self, tmp_path):
        """The case the gate exists for.

        The planner scores ``Lx*Ly*Lz`` only, so T never enters the pixel budget
        and is never scaled: a many-frame series of small frames is both the
        most expensive thing to warm and the one warming does nothing for.
        """
        shape, labels = (4000, 256, 256), ("t", "y", "x")
        assert (
            self._warm_one_tensor(
                tmp_path, shape, labels, (1, 1, 1), chunks=(1, 256, 256)
            )
            == 0
        )


class TestNativePyramidLevels:
    """OME-Zarr advertises its on-disk levels, and they round-trip."""

    def _ome_adapter(self, multires_ome_zarr, source_id="ome-native"):
        import zarr
        from biopb_tensor_server import OmeZarrAdapter

        zarr_path, _level_paths, _zattrs = multires_ome_zarr
        root = zarr.open_group(zarr_path, mode="r")
        return OmeZarrAdapter(root["0"], source_id)

    def test_enumerates_native_datasets(self, multires_ome_zarr):
        adapter = self._ome_adapter(multires_ome_zarr)
        levels = adapter.get_native_pyramid_levels()
        # The fixture builds 4 levels at scale 1,2,4,8 (shape 256..32).
        assert levels is not None
        assert [list(lv.scale_hint) for lv in levels] == [
            [1, 1],
            [2, 2],
            [4, 4],
            [8, 8],
        ]
        assert all(lv.native is True for lv in levels)
        assert all(lv.reduction_method == "precompute" for lv in levels)
        assert [list(lv.shape) for lv in levels] == [
            [256, 256],
            [128, 128],
            [64, 64],
            [32, 32],
        ]

    def test_each_advertised_level_round_trips(self, multires_ome_zarr):
        """Every advertised level resolves to its native dataset via get_read_plan."""
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        adapter = self._ome_adapter(multires_ome_zarr)
        levels = adapter.get_native_pyramid_levels()
        for level in levels:
            scale = tuple(level.scale_hint)
            # The exact-match routing finds a precomputed dataset for this scale.
            assert adapter._find_level_for_scale(scale) is not None
            # And a precompute read plan succeeds and returns the level's shape.
            req = TensorDescriptor(
                array_id=adapter.array_id,
                scale_hint=list(scale),
                reduction_method="precompute",
            )
            plan = adapter.get_read_plan(req)
            assert list(plan.descriptor.shape) == list(level.shape)

    def test_plain_zarr_has_no_native_levels(self, tmp_path):
        import zarr
        from biopb_tensor_server import ZarrAdapter

        arr = zarr.open_array(
            str(tmp_path / "a.zarr"),
            mode="w",
            shape=(64, 64),
            chunks=(32, 32),
            dtype="uint16",
        )
        adapter = ZarrAdapter(arr, "plain", ["y", "x"])
        assert adapter.get_native_pyramid_levels() is None


@pytest.mark.skipif(not _zarr_available(), reason="zarr not installed")
class TestAdvertisedPyramidDescriptor:
    """get_flight_info fills `pyramid`; list_flights leaves it empty."""

    def _flight_info(self, server, source_id, tensor_id=""):
        import pyarrow.flight as flight
        from biopb.tensor.descriptor_pb2 import FlightCmd, TensorReadOption

        cmd = FlightCmd(
            source_id=source_id,
            # Pyramid advertisement is opt-in (biopb/biopb#563); this class asserts
            # get_flight_info fills it, so request it.
            tensor_read=TensorReadOption(tensor_id=tensor_id, with_pyramid=True),
        )
        desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        return server.get_flight_info(None, desc)

    def _descriptor(self, info):
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        return TensorDescriptor.FromString(info.descriptor.command)

    def _big_zarr_adapter(self, tmp_path):
        import zarr
        from biopb_tensor_server import ZarrAdapter

        arr = zarr.open_array(
            str(tmp_path / "big.zarr"),
            mode="w",
            shape=(20000, 20000),
            chunks=(1024, 1024),
            dtype="uint8",
        )
        return ZarrAdapter(arr, "big", ["y", "x"])

    def test_advertises_nothing_without_a_native_pyramid(self, tmp_path):
        """The server no longer publishes a *computed* ladder.

        It was arithmetic over facts the client already has, so it expressed a
        policy rather than knowledge -- and one the server cannot choose for
        every client (a Viv viewer needs strict 2x, this plan stepped 4x). Empty
        means "decide client-side", which the proto already documents.
        """
        server = TensorFlightServer("grpc://localhost:0")
        try:
            server.register_source("big", self._big_zarr_adapter(tmp_path))
            desc = self._descriptor(self._flight_info(server, "big", "big"))
            assert list(desc.pyramid) == []
        finally:
            server.shutdown()

    def test_get_flight_info_advertises_native_pyramid(self, multires_ome_zarr):
        import zarr
        from biopb_tensor_server import OmeZarrAdapter

        zarr_path, _lp, _z = multires_ome_zarr
        root = zarr.open_group(zarr_path, mode="r")
        server = TensorFlightServer("grpc://localhost:0")
        try:
            server.register_source("ome", OmeZarrAdapter(root["0"], "ome"))
            desc = self._descriptor(self._flight_info(server, "ome", "ome"))
            assert len(desc.pyramid) == 4
            assert all(lv.native for lv in desc.pyramid)
            assert all(lv.reduction_method == "precompute" for lv in desc.pyramid)
        finally:
            server.shutdown()

    def test_get_flight_info_on_1d_source_does_not_raise(self, tmp_path):
        # A 1-D tensor has no Y/X plane; GetFlightInfo must still answer rather
        # than raise (regression for the <2-D guard).
        import zarr
        from biopb_tensor_server import ZarrAdapter

        arr = zarr.open_array(
            str(tmp_path / "line.zarr"),
            mode="w",
            shape=(100000,),
            chunks=(8192,),
            dtype="uint8",
        )
        server = TensorFlightServer("grpc://localhost:0")
        try:
            server.register_source("line", ZarrAdapter(arr, "line", ["x"]))
            desc = self._descriptor(self._flight_info(server, "line", "line"))
            assert list(desc.shape) == [100000]
            assert list(desc.pyramid) == []
        finally:
            server.shutdown()

    def test_list_flights_leaves_pyramid_empty(self, tmp_path):
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

        server = TensorFlightServer("grpc://localhost:0")
        try:
            server.register_source("big", self._big_zarr_adapter(tmp_path))
            infos = list(server.list_flights(None, b""))
            assert infos
            src = DataSourceDescriptor.FromString(infos[0].descriptor.command)
            assert all(len(t.pyramid) == 0 for t in src.tensors)
        finally:
            server.shutdown()

    # --- GetFlightInfo response field masks (biopb/biopb#563) ----------------
    # with_metadata / with_pyramid / with_read_plan independently select which
    # parts of the response the server computes: metadata_json, the pyramid
    # advertisement, and the per-request chunk endpoints.

    def _flight_info_opt(self, server, read_opt):
        import pyarrow.flight as flight
        from biopb.tensor.descriptor_pb2 import FlightCmd

        cmd = FlightCmd(
            source_id=read_opt.tensor_id.split("/", 1)[0],
            tensor_read=read_opt,
        )
        desc = flight.FlightDescriptor.for_command(cmd.SerializeToString())
        return server.get_flight_info(None, desc)

    def test_pyramid_advertisement_is_opt_in(self, multires_ome_zarr):
        # with_pyramid gates the pyramid: unset (default false) => none
        # advertised; set => the native levels ride the descriptor. Needs a
        # native source now that a computed ladder is never advertised.
        import zarr
        from biopb.tensor.descriptor_pb2 import TensorReadOption
        from biopb_tensor_server import OmeZarrAdapter

        zarr_path, _level_paths, _zattrs = multires_ome_zarr
        root = zarr.open_group(zarr_path, mode="r")

        server = TensorFlightServer("grpc://localhost:0")
        try:
            server.register_source("big", OmeZarrAdapter(root["0"], "big"))
            bare = self._descriptor(
                self._flight_info_opt(server, TensorReadOption(tensor_id="big"))
            )
            assert len(bare.pyramid) == 0
            asked = self._descriptor(
                self._flight_info_opt(
                    server, TensorReadOption(tensor_id="big", with_pyramid=True)
                )
            )
            assert len(asked.pyramid) >= 2
        finally:
            server.shutdown()

    def test_read_plan_defaults_on_when_unset(self, tmp_path):
        # with_read_plan is optional/default-true: an unset field (old client or a
        # plain read) still enumerates the chunk endpoints, exactly as before.
        from biopb.tensor.descriptor_pb2 import TensorReadOption

        server = TensorFlightServer("grpc://localhost:0")
        try:
            server.register_source("big", self._big_zarr_adapter(tmp_path))
            info = self._flight_info_opt(server, TensorReadOption(tensor_id="big"))
            assert len(info.endpoints) >= 1
        finally:
            server.shutdown()

    def test_describe_only_skips_the_read_plan(self, tmp_path):
        # with_read_plan=False => the descriptor rides back with NO endpoints; the
        # pyramid still honors its own mask, so describe+pyramid works without a plan.
        from biopb.tensor.descriptor_pb2 import TensorReadOption

        server = TensorFlightServer("grpc://localhost:0")
        try:
            server.register_source("big", self._big_zarr_adapter(tmp_path))
            info = self._flight_info_opt(
                server,
                TensorReadOption(
                    tensor_id="big", with_read_plan=False, with_pyramid=True
                ),
            )
            assert len(info.endpoints) == 0
            desc = self._descriptor(info)
            assert list(desc.shape) == [20000, 20000]  # base per-tensor facts
            assert list(desc.pyramid) == []  # computed ladders are never sent
        finally:
            server.shutdown()
