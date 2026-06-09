"""Tests for the background pre-cache worker and its scale-hint computation."""

import threading
import time

import numpy as np
import pytest

from biopb_tensor_server.chunk import (
    compute_precache_scale_hint,
    is_scaled_chunk,
)
from biopb_tensor_server.config import PrecacheConfig
from biopb_tensor_server.precache import PrecacheWorker
from biopb_tensor_server.server import TensorFlightServer


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
# ---------------------------------------------------------------------------


class TestComputePrecacheScaleHint:
    @pytest.mark.parametrize(
        "shape,labels,expected",
        [
            # Small 2-D: below threshold, no downsampling.
            ([2048, 2048], ["y", "x"], [1, 1]),
            # Large 2-D: X/Y land in (1024, 4096].
            ([20000, 20000], ["y", "x"], [16, 16]),
            # Deep volume: the 512**3 voxel budget dominates, not the 4096 X/Y
            # rule -- this is the case a naive "shrink X/Y to <4096" gets wrong.
            ([1000, 8000, 8000], ["z", "y", "x"], [4, 16, 16]),
            # Thin-z stack: z stays at the floor (never over-shrunk).
            ([8, 20000, 20000], ["z", "y", "x"], [1, 16, 16]),
            # Non-spatial axes (t, c) stay at 1.
            ([10, 3, 20000, 20000], ["t", "c", "y", "x"], [1, 1, 16, 16]),
            # Missing labels: positional [..., Z, Y, X] fallback.
            ([1000, 8000, 8000], None, [4, 16, 16]),
            # 2-D positional fallback (last two axes = Y, X).
            ([20000, 20000], None, [16, 16]),
        ],
    )
    def test_scale_hint_values(self, shape, labels, expected):
        assert compute_precache_scale_hint(shape, labels) == expected

    def test_no_z_label_means_no_z_scale(self):
        # [t, y, x] with labels: the leading axis is time, not z, so it stays 1.
        scale = compute_precache_scale_hint([1000, 20000, 20000], ["t", "y", "x"])
        assert scale[0] == 1
        assert scale[1:] == [16, 16]

    def test_custom_knobs(self):
        # Tighter threshold downsamples further.
        scale = compute_precache_scale_hint(
            [20000, 20000], ["y", "x"], threshold=1024
        )
        ly = (20000 + scale[0] - 1) // scale[0]
        assert ly <= 1024

    @pytest.mark.skipif(
        True if _import_biopb_mcp() is None else False,
        reason="biopb-mcp not importable for cross-check",
    )
    def test_matches_biopb_mcp_loop(self):
        """Cross-check against biopb-mcp's actual pyramid loop, if importable."""
        tu = _import_biopb_mcp()
        get_setting = tu.get_setting
        cfg = tu.CONFIG.as_dict()
        threshold = get_setting(cfg, "pyramid.threshold")
        downscale = get_setting(cfg, "pyramid.downscale_factor")
        budget_root = get_setting(cfg, "pyramid.pixel_budget_cubic_root")

        for shape, labels in [
            ([20000, 20000], ["y", "x"]),
            ([1000, 8000, 8000], ["z", "y", "x"]),
            ([8, 20000, 20000], ["z", "y", "x"]),
        ]:
            ours = compute_precache_scale_hint(
                shape,
                labels,
                threshold=threshold,
                downscale_factor=downscale,
                pixel_budget_cubic_root=budget_root,
            )
            theirs = _simulate_client_terminal_scale(
                shape, labels, threshold, downscale, budget_root
            )
            assert ours == theirs, f"mismatch for {shape}: {ours} != {theirs}"


# ---------------------------------------------------------------------------
# 2. Flight-activity idle probe.
# ---------------------------------------------------------------------------


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
                with server._serving_request():
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
            with server._serving_request():
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
        from biopb_tensor_server.config import CacheConfig

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
            worker._process_source("warm-src")

            cache_manager = CacheManager.get_instance()
            stats = cache_manager.stats()
            assert stats.misses > 0  # cold computes happened

            # Rebuild the same read plan and assert every chunk now locates on
            # disk -- i.e. a future do_get is a warm hit, no decode needed.
            adapter = server._get_source_adapter("warm-src")
            td = adapter.list_tensor_descriptors()[0]
            ta = adapter.get_tensor_adapter(td.array_id)
            scale = compute_precache_scale_hint(
                list(td.shape), list(td.dim_labels)
            )
            assert scale == [4, 4]
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
                assert cache_manager.locate_entry(ce.chunk_id) is not None
        finally:
            server.shutdown()
            CacheManager.get_instance().close()
            CacheManager.reset()

    def test_memory_backend_is_noop(self, tmp_path):
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.config import CacheConfig

        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        try:
            server = self._make_server_with_zarr(tmp_path, (8192, 8192))
            worker = PrecacheWorker(server, PrecacheConfig(idle_debounce_seconds=0.0))
            worker._process_source("warm-src")

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
        from biopb_tensor_server.discovery import AdapterRegistry, DiscoveryState
        from biopb_tensor_server.source_manager import SourceManager

        server = TensorFlightServer("grpc://localhost:0")
        sm = SourceManager(
            server=server,
            registry=AdapterRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs=set(),
        )
        return server, sm

    def test_runtime_phase_default_false_then_start_flips(self):
        server, sm = self._bare_source_manager()
        try:
            assert sm._runtime_phase is False
            # Static-only (watcher=None) start() is a no-op and does NOT flip.
            sm.start()
            assert sm._runtime_phase is False
        finally:
            server.shutdown()

    def test_commit_hook_fires_only_in_runtime_phase(self, monkeypatch):
        from types import SimpleNamespace

        server, sm = self._bare_source_manager()
        try:
            # Stub the heavy commit collaborators so we exercise only the gate.
            monkeypatch.setattr(sm, "_register_source_claim", lambda claim: True)
            monkeypatch.setattr(sm._state, "add_claim", lambda claim, notify=False: True)
            monkeypatch.setattr(sm, "_build_claim_signatures", lambda claim: {})
            monkeypatch.setattr(sm, "_clear_failed_source_attempt", lambda sid: None)

            fired = []
            sm._on_source_committed = fired.append
            claim = SimpleNamespace(source_id="s1", primary_path="/x")

            # Startup phase: hook must NOT fire.
            sm._runtime_phase = False
            assert sm._commit_add_claim(claim) is True
            assert fired == []

            # Runtime phase: hook fires with the source_id.
            sm._runtime_phase = True
            assert sm._commit_add_claim(claim) is True
            assert fired == ["s1"]
        finally:
            server.shutdown()

    def test_hook_exception_does_not_abort_commit(self, monkeypatch):
        from types import SimpleNamespace

        server, sm = self._bare_source_manager()
        try:
            monkeypatch.setattr(sm, "_register_source_claim", lambda claim: True)
            monkeypatch.setattr(sm._state, "add_claim", lambda claim, notify=False: True)
            monkeypatch.setattr(sm, "_build_claim_signatures", lambda claim: {})
            monkeypatch.setattr(sm, "_clear_failed_source_attempt", lambda sid: None)

            def boom(_sid):
                raise RuntimeError("hook failure")

            sm._on_source_committed = boom
            sm._runtime_phase = True
            claim = SimpleNamespace(source_id="s2", primary_path="/y")
            # Commit still succeeds despite the hook raising.
            assert sm._commit_add_claim(claim) is True
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# 5. Preemption smoke + worker lifecycle/dedup.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestPreemptionAndLifecycle:
    def test_no_warming_while_in_flight(self, tmp_path):
        import zarr

        from biopb_tensor_server import ZarrAdapter
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.config import CacheConfig

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
                with server._serving_request():
                    holding.set()
                    release.wait(3.0)

            ht = threading.Thread(target=hold, daemon=True)
            ht.start()
            assert holding.wait(2.0)

            worker.start()
            worker.enqueue("pre-src")
            # While the request is in flight, the worker should not have warmed
            # anything yet.
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

    def test_enqueue_dedup(self):
        server = TensorFlightServer("grpc://localhost:0")
        try:
            worker = PrecacheWorker(server, PrecacheConfig())
            worker.enqueue("a")
            worker.enqueue("a")
            worker.enqueue("b")
            # 'a' deduped while still queued.
            assert worker._queue.qsize() == 2
        finally:
            server.shutdown()
