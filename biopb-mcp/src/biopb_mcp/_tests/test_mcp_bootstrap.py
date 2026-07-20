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
                {"dask": {"cache_budget": "1G"}},
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
                {"dask": {"cache_budget": "1G"}},
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
                {"dask": {"cache_budget": 800_000_000}},
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
                {"dask": {"cache_budget": "4G"}},
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


class _FakeIP:
    """A stand-in kernel with just the user namespace the loader touches."""

    def __init__(self, ns):
        self.user_ns = ns


def _seeded_ns():
    """A namespace shaped like the post-step-7 kernel (the load-bearing handles)."""
    return {
        "viewer": "REAL_VIEWER",
        "client": None,
        "np": "NP",
        "da": "DA",
        "ops": {},
        "run_on_main": lambda f: f(),
        "_conn": object(),
        "_jobs": object(),
        "_dask_client": None,
        "_dask_attach_done": False,
        "_viewer_window_alive": lambda: True,
        "_resync_view": lambda: None,
    }


class TestLoadStartupFiles:
    """`~/.config/biopb/kernel/*.py`, exec'd in the namespace (#92)."""

    def test_top_level_defs_land_and_see_live_handles(self, tmp_path):
        # A plugin function referencing `viewer`/`client` as globals resolves them
        # against the kernel namespace at call time -- including a `client`
        # refreshed after load (the per-job refresh), exactly like execute_code.
        (tmp_path / "a.py").write_text(
            '"""Lab tools."""\ndef my_tool():\n    return (viewer, client)\n',
            encoding="utf-8",
        )
        ns = _seeded_ns()
        _bootstrap._load_startup_files(_FakeIP(ns), tmp_path)
        assert "my_tool" in ns
        ns["client"] = "CONNECTED"  # simulate the per-job client refresh
        assert ns["my_tool"]() == ("REAL_VIEWER", "CONNECTED")

    def test_failing_file_is_fail_open_and_next_still_loads(self, tmp_path):
        (tmp_path / "a_boom.py").write_text(
            'raise RuntimeError("boom at import")\n', encoding="utf-8"
        )
        (tmp_path / "b_ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        ns = _seeded_ns()
        _bootstrap._load_startup_files(_FakeIP(ns), tmp_path)
        assert "ok" in ns  # the boom did not abort the sweep

    def test_reserved_name_overwrite_is_restored(self, tmp_path):
        (tmp_path / "c.py").write_text(
            'viewer = "HIJACKED"\nclient = "HIJACKED"\n', encoding="utf-8"
        )
        ns = _seeded_ns()
        _bootstrap._load_startup_files(_FakeIP(ns), tmp_path)
        assert ns["viewer"] == "REAL_VIEWER" and ns["client"] is None

    def test_underscore_files_skipped_and_missing_dir_is_noop(self, tmp_path):
        (tmp_path / "_priv.py").write_text("secret = 1\n", encoding="utf-8")
        ns = _seeded_ns()
        _bootstrap._load_startup_files(_FakeIP(ns), tmp_path)
        assert "secret" not in ns
        # A non-existent dir must not raise.
        _bootstrap._load_startup_files(_FakeIP(_seeded_ns()), tmp_path / "nope")


class TestPublicNamesAndMerge:
    def test_public_names_honors_all_and_drops_modules(self):
        import numpy

        assert _bootstrap._public_names(
            {"__all__": ["keep"], "keep": 1, "skip": 2}
        ) == {"keep": 1}
        assert _bootstrap._public_names({"pub": 1, "_priv": 2, "mod": numpy}) == {
            "pub": 1
        }

    def test_merge_skips_reserved_names(self):
        ns = _seeded_ns()
        _bootstrap._merge_names(
            _FakeIP(ns), {"newname": 42, "viewer": "NO"}, source="ep:test"
        )
        assert ns["newname"] == 42 and ns["viewer"] == "REAL_VIEWER"


class TestLoadEntryPointPlugins:
    """The `biopb_mcp.namespace` entry-point dispatch (register / module / dict)."""

    def _run(self, monkeypatch, eps):
        import importlib.metadata as md

        monkeypatch.setattr(md, "entry_points", lambda group=None: eps)
        ns = _seeded_ns()
        _bootstrap._load_entry_point_plugins(_FakeIP(ns))
        return ns

    def _ep(self, name, obj):
        class _EP:
            def load(self_inner):
                return obj

        ep = _EP()
        ep.name = name
        return ep

    def test_register_hook_reads_handles_and_is_guarded(self, monkeypatch):
        def register(namespace):
            assert namespace["viewer"] == "REAL_VIEWER"  # read-through snapshot
            namespace["reg_tool"] = "R"
            namespace["viewer"] = "HIJACK"  # guarded on merge

        ns = self._run(monkeypatch, [self._ep("reg", register)])
        assert ns["reg_tool"] == "R" and ns["viewer"] == "REAL_VIEWER"

    def test_module_and_mapping_filter_public_names(self, monkeypatch):
        import types

        mod = types.ModuleType("m")
        mod.__all__ = ["mod_tool"]
        mod.mod_tool = "M"
        mod.hidden = "H"
        ns = self._run(
            monkeypatch,
            [self._ep("mod", mod), self._ep("map", {"map_tool": "MP", "_skip": "no"})],
        )
        assert ns["mod_tool"] == "M" and "hidden" not in ns
        assert ns["map_tool"] == "MP" and "_skip" not in ns

    def test_junk_and_import_failure_are_fail_open(self, monkeypatch):
        class _Boom:
            def load(self_inner):
                raise RuntimeError("import boom")

        boom = _Boom()
        boom.name = "boom"
        ns = self._run(monkeypatch, [self._ep("junk", 12345), boom])
        # Neither a non-register/module/mapping nor an import failure adds anything
        # or raises.
        assert "junk" not in ns


class TestLoadNamespacePluginsGate:
    def test_disabled_by_config_skips_everything(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(
            _bootstrap, "_load_startup_files", lambda *a: called.append("f")
        )
        monkeypatch.setattr(
            _bootstrap, "_load_entry_point_plugins", lambda *a: called.append("e")
        )
        _bootstrap._load_namespace_plugins(
            _FakeIP(_seeded_ns()), {"services": {"namespace_enabled": False}}
        )
        assert called == []
