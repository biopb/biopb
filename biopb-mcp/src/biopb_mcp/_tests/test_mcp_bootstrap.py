"""Unit tests for bootstrap helpers that don't need a kernel/display.

Currently: _register_cache_plugin -- the cluster-wide chunk-cache budget split
across dask workers, installed via biopb's worker-init plugin.
"""

from unittest.mock import MagicMock, patch

import biopb.tensor.client as tclient
import pytest

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
        # The plugin splits the budget the same for localhost and remote URLs --
        # it never special-cases the host. (_resolve_cache_bytes no longer clamps
        # localhost to 0 either; localhost caches copies like any other host.)
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
            assert tclient._CACHE_POOL[(loc, None)].available_bytes == 777
        finally:
            tclient._CACHE_POOL.clear()


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


class TestLoadPluginFiles:
    """`~/.config/biopb/kernel/*.py`, imported and bound by module name (#664)."""

    def test_a_plugin_contributes_exactly_one_name(self, tmp_path):
        # The whole point of the module binding: a file's helpers, its imports and
        # its dunders stay on the module. Under the old exec loader every one of
        # these landed in the agent's namespace.
        (tmp_path / "a.py").write_text(
            '"""Lab tools."""\n'
            "import math\n"
            "__all__ = ['my_tool']\n"
            "HELPER_CONST = 3\n"
            "def _helper():\n    return 1\n"
            "def my_tool():\n    return math.floor(2.5) + _helper()\n",
            encoding="utf-8",
        )
        ns = _seeded_ns()
        before = set(ns)
        _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
        assert set(ns) - before == {"a"}
        assert ns["a"].my_tool() == 3
        assert ns["a"].HELPER_CONST == 3  # reachable, just not in the namespace

    def test_a_plugin_cannot_shadow_a_reserved_handle(self, tmp_path):
        # Now a single check on one name, and the file that loses it is not bound
        # at all -- there is no partial contribution to clean up.
        (tmp_path / "viewer.py").write_text("def hijack():\n    return 1\n", "utf-8")
        ns = _seeded_ns()
        loaded = _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
        assert ns["viewer"] == "REAL_VIEWER"
        assert loaded == []

    def test_a_plugin_named_after_a_real_package_does_not_shadow_it(self, tmp_path):
        # Registered under the `biopb_kernel_plugins.` prefix, never the bare stem:
        # a `json.py` in the kernel dir must not become `sys.modules["json"]` for
        # everything imported afterwards.
        import json
        import sys

        (tmp_path / "json.py").write_text(
            "def dumps(x):\n    return 'HIJACKED'\n", "utf-8"
        )
        ns = _seeded_ns()
        _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
        assert ns["json"].dumps({}) == "HIJACKED"  # the plugin, under its own name
        assert sys.modules["json"] is json  # the real module, untouched
        assert json.dumps({"a": 1}) == '{"a": 1}'
        sys.modules.pop("biopb_kernel_plugins.json", None)

    def test_import_reaches_a_loaded_plugin(self, tmp_path):
        # `import <stem>` is what anyone writes, so it has to work. A bound name
        # that raises ModuleNotFoundError on import reads as broken, and telling
        # people about the difference is a weaker fix than not having one: a
        # benchmarked agent saw `files: image_resolution` in server_status,
        # wrote the import, and went hunting the filesystem for the file.
        import importlib
        import sys

        (tmp_path / "tool_for_import.py").write_text(
            "def answer():\n    return 7\n", "utf-8"
        )
        ns = _seeded_ns()
        try:
            loaded = _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
            assert loaded == ["tool_for_import"]
            imported = importlib.import_module("tool_for_import")
            assert imported.answer() == 7
            # The same object, not a second execution of the file.
            assert imported is ns["tool_for_import"]
        finally:
            sys.modules.pop("tool_for_import", None)
            sys.modules.pop("biopb_kernel_plugins.tool_for_import", None)
            _bootstrap._PLUGIN_IMPORT_HOOK.unregister("tool_for_import")

    def test_import_still_prefers_a_real_package_over_a_plugin(self, tmp_path):
        # The guarantee the module prefix was introduced for, now that `import`
        # can reach a plugin at all. The hook is *appended* to sys.meta_path, so
        # every standard finder runs first and an installed package always wins;
        # `sys.modules[stem] = mod` would not be equivalent, because imports
        # short-circuit on sys.modules before any finder is consulted.
        import colorsys
        import importlib
        import sys

        (tmp_path / "colorsys.py").write_text(
            "def rgb_to_hls(*a):\n    return 'HIJACKED'\n", "utf-8"
        )
        ns = _seeded_ns()
        saved = sys.modules.pop("colorsys", None)
        try:
            _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
            assert ns["colorsys"].rgb_to_hls() == "HIJACKED"  # bound, under its name
            # Nothing is in sys.modules for it, so this is a real resolution and
            # not a cache hit: the stdlib module has to win on the meta_path.
            assert "colorsys" not in sys.modules
            assert importlib.import_module("colorsys").rgb_to_hls(0, 0, 0) == (0, 0, 0)
        finally:
            sys.modules.pop("biopb_kernel_plugins.colorsys", None)
            _bootstrap._PLUGIN_IMPORT_HOOK.unregister("colorsys")
            sys.modules["colorsys"] = saved if saved is not None else colorsys

    def test_the_import_hook_is_installed_last(self, tmp_path):
        """Position is the whole guarantee, so it is asserted rather than
        assumed. Prepending would make every plugin able to shadow a package."""
        import sys

        (tmp_path / "positional.py").write_text("X = 1\n", "utf-8")
        try:
            _bootstrap._load_plugin_files(_FakeIP(_seeded_ns()), tmp_path)
            assert sys.meta_path[-1] is _bootstrap._PLUGIN_IMPORT_HOOK
            # And installed once, however many plugins or reloads there are.
            _bootstrap._load_plugin_files(_FakeIP(_seeded_ns()), tmp_path)
            assert sys.meta_path.count(_bootstrap._PLUGIN_IMPORT_HOOK) == 1
        finally:
            sys.modules.pop("biopb_kernel_plugins.positional", None)
            _bootstrap._PLUGIN_IMPORT_HOOK.unregister("positional")

    def test_a_plugin_that_lost_its_name_is_not_importable_either(self, tmp_path):
        """A reserved-name collision contributes nothing, and that has to include
        the import route — otherwise the name the namespace refused is still
        reachable one `import` away."""
        import importlib
        import sys

        (tmp_path / "viewer.py").write_text("def hijack():\n    return 1\n", "utf-8")
        try:
            assert _bootstrap._load_plugin_files(_FakeIP(_seeded_ns()), tmp_path) == []
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module("viewer")
        finally:
            sys.modules.pop("biopb_kernel_plugins.viewer", None)

    def test_plugin_functions_still_run_on_a_dask_worker(self, tmp_path):
        # A module's functions pickle *by reference* by default -- a few bytes
        # naming a module no worker can import, since the kernel plugin dir is on
        # no other process's sys.path. The loader registers each plugin for
        # by-value pickling to keep `da.map_blocks(plug.fn)` working; without it
        # this fails at compute time, inside the worker, far from the load.
        import subprocess
        import sys

        import cloudpickle

        (tmp_path / "shipped.py").write_text(
            "def double(x):\n    return x * 2\n", "utf-8"
        )
        ns = _seeded_ns()
        _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
        blob = cloudpickle.dumps(ns["shipped"].double)
        sys.modules.pop("biopb_kernel_plugins.shipped", None)

        # A fresh interpreter, as a dask worker process would be: no plugin dir on
        # sys.path, no loader run, nothing shared but the bytes.
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pickle,sys; fn=pickle.loads(sys.stdin.buffer.read());"
                " print(fn(21))",
            ],
            input=blob,
            capture_output=True,
            cwd=tmp_path.parent,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        assert proc.stdout.strip() == b"42"

    def test_failing_file_is_fail_open_and_next_still_loads(self, tmp_path):
        (tmp_path / "a_boom.py").write_text(
            'raise RuntimeError("boom at import")\n', encoding="utf-8"
        )
        (tmp_path / "b_ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        ns = _seeded_ns()
        _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
        assert "b_ok" in ns  # the boom did not abort the sweep
        assert "a_boom" not in ns

    def test_a_half_executed_file_leaves_no_module_registered(self, tmp_path):
        # The file raised partway through import; nothing may be left in
        # sys.modules for a later import to find in that state.
        import sys

        (tmp_path / "halfway.py").write_text(
            "def ok():\n    return 1\nraise RuntimeError('boom')\n", encoding="utf-8"
        )
        _bootstrap._load_plugin_files(_FakeIP(_seeded_ns()), tmp_path)
        assert "biopb_kernel_plugins.halfway" not in sys.modules

    def test_underscore_files_skipped_and_missing_dir_is_noop(self, tmp_path):
        (tmp_path / "_priv.py").write_text("secret = 1\n", encoding="utf-8")
        ns = _seeded_ns()
        _bootstrap._load_plugin_files(_FakeIP(ns), tmp_path)
        assert "_priv" not in ns and "secret" not in ns
        # A non-existent dir must not raise.
        _bootstrap._load_plugin_files(_FakeIP(_seeded_ns()), tmp_path / "nope")

    def test_only_the_files_that_survived_are_reported_as_loaded(self, tmp_path):
        # The record `server_status` reports a skill's `plugin:<name>` from. It
        # cannot be a directory listing: this loader is fail-open, so a file that
        # raised is on disk and *not* in the namespace -- the distinction the
        # record exists to get right.
        (tmp_path / "good.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        (tmp_path / "bad.py").write_text(
            'raise RuntimeError("boom")\n', encoding="utf-8"
        )
        (tmp_path / "_priv.py").write_text("x = 1\n", encoding="utf-8")
        loaded = _bootstrap._load_plugin_files(_FakeIP(_seeded_ns()), tmp_path)
        assert loaded == ["good"]


class TestPluginRecordReachesServerStatus:
    """The loader -> `_requires` record -> `server_status` handoff.

    What a skill's `plugin:<name>` is resolved against: not the kernel dir (this
    loader is fail-open) and not `dir()` (a file contributes its function names,
    not its own name).
    """

    def test_a_file_that_failed_to_load_is_absent_from_the_report(
        self, tmp_path, monkeypatch
    ):
        from biopb_mcp.mcp import _requires

        (tmp_path / "good.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        (tmp_path / "bad.py").write_text(
            'raise RuntimeError("boom")\n', encoding="utf-8"
        )
        monkeypatch.setattr("biopb._locations.mcp_plugin_dir", lambda: tmp_path)
        _bootstrap._load_namespace_plugins(_FakeIP(_seeded_ns()), {})

        report = "\n".join(_requires.plugin_status_lines())
        assert "good" in report
        assert "bad" not in report
        assert "session log" in report  # where the load failure is explained

    def test_the_disabled_switch_is_recorded_not_inferred(self, tmp_path, monkeypatch):
        from biopb_mcp.mcp import _requires

        (tmp_path / "good.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        monkeypatch.setattr("biopb._locations.mcp_plugin_dir", lambda: tmp_path)
        config = {"services": {"namespace_enabled": False}}
        _bootstrap._load_namespace_plugins(_FakeIP(_seeded_ns()), config)

        # Same empty record as "nothing seeded", but a different reason: seeding a
        # file would not help while the switch is off.
        (line,) = _requires.plugin_status_lines()
        assert "services.namespace_enabled" in line
        assert "seed" not in line

    def test_files_and_entry_points_are_reported_apart(self):
        from biopb_mcp.mcp import _requires

        # A `plugin:<name>` matches either, but only a *file* has the "on disk yet
        # missing here" story, so the two are not merged into one list.
        _requires.record_loaded_plugins(["rolling_ball"], ["labshop_tools"])
        report = "\n".join(_requires.plugin_status_lines())
        assert "files: rolling_ball" in report
        assert "packages: labshop_tools" in report

    def test_no_plugins_points_at_the_seeder(self):
        from biopb_mcp.mcp import _requires

        report = "\n".join(_requires.plugin_status_lines())
        assert "biopb-mcp-seed-plugins" in report

    def test_both_lines_print_even_when_empty(self):
        from biopb_mcp.mcp import _requires

        # An omitted line would make "my plugin isn't listed" ambiguous: absent
        # because it didn't load, or because that half of the report was skipped?
        # The agent resolving `plugin:<name>` has to be able to tell.
        _requires.record_loaded_plugins(["rolling_ball"])
        report = "\n".join(_requires.plugin_status_lines())
        assert "files: rolling_ball" in report
        assert "packages: (none" in report
        assert "biopb_mcp.namespace" in report  # what a "package" plugin even is


class TestInstallTargetIsDecidedByTheEnv:
    """`## Versions` has to answer "install where, how, and does it last?".

    The agent can read `sys.executable`; what it cannot see is that biopb's own
    deployment is a uv tool env the next upgrade rebuilds.
    """

    def _lines(self, tmp_path, *, receipt, has_pip):
        from biopb_mcp.mcp import _requires

        if receipt:
            (tmp_path / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
        return "\n".join(
            _requires.versions_status_lines(
                prefix=tmp_path,
                executable="/env/bin/python",
                has_pip=has_pip,
                version="9.9.9",
            )
        )

    def test_a_plain_env_gets_pip_and_no_warning(self, tmp_path):
        report = self._lines(tmp_path, receipt=False, has_pip=True)
        assert "biopb-mcp: 9.9.9" in report
        assert "/env/bin/python -m pip install <pkg>" in report
        assert "uv-managed" not in report  # the user's env; theirs to keep

    def test_no_pip_falls_back_to_uv(self, tmp_path):
        report = self._lines(tmp_path, receipt=False, has_pip=False)
        assert "uv pip install --python /env/bin/python" in report

    def test_a_uv_tool_env_warns_even_though_it_has_pip(self, tmp_path):
        # The real deployment carries pip transitively, so a pip probe would take
        # the `-m pip` branch and say nothing about the rebuild -- the receipt is
        # what identifies the env, not the absence of pip.
        report = self._lines(tmp_path, receipt=True, has_pip=True)
        assert "uv pip install --python /env/bin/python" in report
        assert "-m pip install" not in report
        assert "extra-packages.txt" in report  # the durable half of the fix


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

    def test_module_and_mapping_each_bind_one_name(self, monkeypatch):
        # Same contract as a plugin file (#664): the entry-point name is the whole
        # contribution, and a mapping is wrapped so it is reached the same way.
        import types

        mod = types.ModuleType("m")
        mod.mod_tool = "M"
        mod.hidden = "H"
        ns = _seeded_ns()
        before = set(ns)
        import importlib.metadata as md

        monkeypatch.setattr(
            md,
            "entry_points",
            lambda group=None: [
                self._ep("mod", mod),
                self._ep("map", {"map_tool": "MP", "_skip": "no"}),
            ],
        )
        _bootstrap._load_entry_point_plugins(_FakeIP(ns))
        assert set(ns) - before == {"mod", "map"}
        assert ns["mod"].mod_tool == "M" and ns["mod"].hidden == "H"
        assert ns["map"].map_tool == "MP" and not hasattr(ns["map"], "_skip")

    def test_a_module_entry_point_cannot_shadow_a_reserved_handle(self, monkeypatch):
        import types

        ns = self._run(monkeypatch, [self._ep("viewer", types.ModuleType("m"))])
        assert ns["viewer"] == "REAL_VIEWER"

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
            _bootstrap, "_load_plugin_files", lambda *a: called.append("f")
        )
        monkeypatch.setattr(
            _bootstrap, "_load_entry_point_plugins", lambda *a: called.append("e")
        )
        _bootstrap._load_namespace_plugins(
            _FakeIP(_seeded_ns()), {"services": {"namespace_enabled": False}}
        )
        assert called == []
