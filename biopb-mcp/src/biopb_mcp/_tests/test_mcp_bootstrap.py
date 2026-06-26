"""Unit tests for bootstrap helpers that don't need a kernel/display.

Currently: _register_cache_plugin -- the cluster-wide chunk-cache budget split
across dask workers, installed via biopb's worker-init plugin.
"""

from unittest.mock import MagicMock, patch

import biopb.tensor.client as tclient

from biopb_mcp.mcp import _bootstrap


def _fake_dask_client(n_workers):
    dc = MagicMock()
    dc.scheduler_info.return_value = {
        "workers": {f"w{i}": {} for i in range(n_workers)}
    }
    return dc


class TestRegisterCachePlugin:
    def test_splits_budget_by_planned_workers(self):
        dc = _fake_dask_client(5)  # live count (should be IGNORED)
        with patch.object(tclient, "make_cache_plugin") as mk:
            mk.return_value = MagicMock(name="plugin")
            _bootstrap._register_cache_plugin(
                dc,
                "grpc://remote:8815",
                "tok",
                {"mcp": {"dask": {"cache_budget": "1G"}}},
                planned_workers=12,
            )
        # 1G // 12 planned workers, NOT // 5 live workers
        loc, tok, per_worker = mk.call_args.args
        assert loc == "grpc://remote:8815" and tok == "tok"
        assert per_worker == 1_000_000_000 // 12
        dc.register_plugin.assert_called_once_with(mk.return_value)

    def test_falls_back_to_live_count_without_planned(self):
        dc = _fake_dask_client(4)
        with patch.object(tclient, "make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _bootstrap._register_cache_plugin(
                dc,
                "grpc://remote:8815",
                None,
                {"mcp": {"dask": {"cache_budget": "1G"}}},
            )
        assert mk.call_args.args[2] == 1_000_000_000 // 4

    def test_accepts_int_budget(self):
        dc = _fake_dask_client(2)
        with patch.object(tclient, "make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _bootstrap._register_cache_plugin(
                dc,
                "grpc://remote:8815",
                None,
                {"mcp": {"dask": {"cache_budget": 800_000_000}}},
                planned_workers=2,
            )
        assert mk.call_args.args[2] == 400_000_000

    def test_localhost_pins_workers_off(self):
        # On localhost a worker's per-process cache is redundant: workers consume
        # whole chunks and share the server's mmap/page-cache path, so the plugin
        # pins workers to 0 regardless of the configured budget (the main-process
        # viewer keeps its cache via BIOPB_CACHE_LOCAL, not this plugin).
        dc = _fake_dask_client(8)
        with patch.object(tclient, "make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _bootstrap._register_cache_plugin(
                dc,
                "grpc://localhost:8815",
                None,
                {"mcp": {"dask": {"cache_budget": "4G"}}},
                planned_workers=8,
            )
        assert mk.call_args.args[2] == 0
        dc.register_plugin.assert_called_once_with(mk.return_value)

    def test_noop_without_dask_client(self):
        # must not raise when there is no distributed client
        _bootstrap._register_cache_plugin(None, "grpc://x:1", None, {})

    def test_noop_when_plugin_unavailable(self):
        dc = _fake_dask_client(3)
        with patch.object(tclient, "make_cache_plugin", return_value=None) as mk:
            _bootstrap._register_cache_plugin(
                dc, "grpc://remote:8815", None, {}, planned_workers=3
            )
        mk.assert_called_once()
        dc.register_plugin.assert_not_called()


class TestHeadlessViewer:
    """The sentinel bound to ``viewer`` when the kernel runs without a display."""

    def test_attribute_access_raises_descriptive_error(self):
        v = _bootstrap._HeadlessViewer()
        import pytest

        with pytest.raises(RuntimeError) as exc:
            v.add_image(None)
        assert "headless" in str(exc.value).lower()
        assert "no display" in str(exc.value).lower()

    def test_is_falsy_for_if_viewer_guards(self):
        # Kernel snippets guard the viewer section with `if viewer:`.
        assert not _bootstrap._HeadlessViewer()

    def test_repr_is_self_describing(self):
        assert "headless" in repr(_bootstrap._HeadlessViewer()).lower()
