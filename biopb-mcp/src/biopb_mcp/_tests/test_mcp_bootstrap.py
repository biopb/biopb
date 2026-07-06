"""Unit tests for bootstrap helpers that don't need a kernel/display.

Currently: _register_cache_plugin -- the cluster-wide chunk-cache budget split
across dask workers, installed via biopb's worker-init plugin; and
_make_token_report_hook -- the connect-time token report to the launcher (#86).
"""

import os
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
        with patch.object(_bootstrap, "_make_cache_plugin") as mk:
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
        with patch.object(_bootstrap, "_make_cache_plugin") as mk:
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
        with patch.object(_bootstrap, "_make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _bootstrap._register_cache_plugin(
                dc,
                "grpc://remote:8815",
                None,
                {"mcp": {"dask": {"cache_budget": 800_000_000}}},
                planned_workers=2,
            )
        assert mk.call_args.args[2] == 400_000_000

    def test_localhost_is_not_special_cased(self):
        # The plugin no longer special-cases localhost: it splits the budget the
        # same as for a remote URL. The localhost no-cache rule is applied
        # authoritatively per worker by the tensor client
        # (_resolve_cache_bytes), which clamps this budget to 0 on localhost.
        dc = _fake_dask_client(8)
        with patch.object(_bootstrap, "_make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _bootstrap._register_cache_plugin(
                dc,
                "grpc://localhost:8815",
                None,
                {"mcp": {"dask": {"cache_budget": "4G"}}},
                planned_workers=8,
            )
        assert mk.call_args.args[2] == 4_000_000_000 // 8
        dc.register_plugin.assert_called_once_with(mk.return_value)

    def test_noop_without_dask_client(self):
        # must not raise when there is no distributed client
        _bootstrap._register_cache_plugin(None, "grpc://x:1", None, {})

    def test_noop_when_plugin_unavailable(self):
        dc = _fake_dask_client(3)
        with patch.object(_bootstrap, "_make_cache_plugin", return_value=None) as mk:
            _bootstrap._register_cache_plugin(
                dc, "grpc://remote:8815", None, {}, planned_workers=3
            )
        mk.assert_called_once()
        dc.register_plugin.assert_not_called()


class TestMakeCachePlugin:
    """The dask WorkerPlugin factory, moved out of the tensor SDK into MCP."""

    def test_returns_none_or_named_plugin(self):
        plugin = _bootstrap._make_cache_plugin("grpc://remote:8815", None, 1000)
        try:
            import distributed  # noqa: F401
        except Exception:
            assert plugin is None  # graceful no-op without distributed
            return
        assert plugin is not None
        assert plugin.name == "biopb-cache-config"

    def test_setup_pins_cache_via_sdk_configure_cache(self):
        import pytest

        pytest.importorskip("distributed")
        tclient._CACHE_POOL.clear()
        loc = "grpc://remote:8815"
        plugin = _bootstrap._make_cache_plugin(loc, None, 777)
        try:
            plugin.setup(worker=None)  # what dask calls on each worker
            assert tclient._CACHE_POOL[(loc, None)][1].available_bytes == 777
        finally:
            tclient._CACHE_POOL.clear()


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


class TestTokenReportHook:
    """The on_connect-side hook that reports (url, token) to the launcher (#86)."""

    def test_none_without_report_fd(self, monkeypatch):
        monkeypatch.delenv("BIOPB_TOKEN_REPORT_FD", raising=False)
        assert _bootstrap._make_token_report_hook() is None

    def test_none_when_fd_not_an_int(self, monkeypatch):
        monkeypatch.setenv("BIOPB_TOKEN_REPORT_FD", "not-a-number")
        assert _bootstrap._make_token_report_hook() is None

    def test_writes_url_tab_token_line(self, monkeypatch):
        r, w = os.pipe()
        monkeypatch.setenv("BIOPB_TOKEN_REPORT_FD", str(w))
        try:
            hook = _bootstrap._make_token_report_hook()
            assert hook is not None
            hook("grpc://srv:8815", "secret-tok")
            assert os.read(r, 4096) == b"grpc://srv:8815\tsecret-tok\n"
        finally:
            os.close(r)
            os.close(w)

    def test_none_token_serializes_as_empty_field(self, monkeypatch):
        r, w = os.pipe()
        monkeypatch.setenv("BIOPB_TOKEN_REPORT_FD", str(w))
        try:
            hook = _bootstrap._make_token_report_hook()
            hook("grpc://srv:8815", None)
            # Empty token field => the launcher clears its remembered token.
            assert os.read(r, 4096) == b"grpc://srv:8815\t\n"
        finally:
            os.close(r)
            os.close(w)

    def test_io_error_is_swallowed(self, monkeypatch):
        r, w = os.pipe()
        monkeypatch.setenv("BIOPB_TOKEN_REPORT_FD", str(w))
        hook = _bootstrap._make_token_report_hook()
        os.close(r)
        os.close(w)  # writing to a closed pipe raises -> must be swallowed
        hook("grpc://srv:8815", "tok")  # no exception
