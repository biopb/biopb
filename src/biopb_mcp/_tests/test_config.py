"""Tests for _config.py configuration management."""

import json

import numpy as np
import pytest

from biopb_mcp._config import (
    CONFIG,
    DEFAULT_CONFIG,
    get_config_path,
    get_default_config,
    get_grid_params,
    get_setting,
    load_config,
    save_config,
)


@pytest.fixture
def mock_config_dir(monkeypatch, tmp_path):
    """Redirect the home-relative config dir (~/.config/biopb-mcp) to a tmp path."""
    import pathlib

    monkeypatch.setattr(
        pathlib.Path, "home", classmethod(lambda cls: tmp_path)
    )
    return tmp_path / ".config" / "biopb-mcp"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_returns_default_when_no_file(self, mock_config_dir):
        """Returns default config when file doesn't exist."""
        config = load_config()
        assert config == get_default_config()

    def test_loads_existing_config(self, mock_config_dir):
        """Loads and merges existing config file."""
        # Create a config file with some custom values
        custom_config = {
            "widget": {
                "server_url": "custom.server.org",
                "detection": {"min_score": 0.5},
            },
        }
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Custom values should override defaults
        assert config["widget"]["server_url"] == "custom.server.org"
        assert config["widget"]["detection"]["min_score"] == 0.5

        # Deep merge: sibling leaves under the overridden section survive.
        defaults = get_default_config()
        assert config["widget"]["grid"] == defaults["widget"]["grid"]
        assert (
            config["widget"]["detection"]["nms"]
            == defaults["widget"]["detection"]["nms"]
        )

    def test_deep_merge_preserves_sibling_leaves(self, mock_config_dir):
        """A partial nested override touches only its own leaf."""
        custom_config = {"mcp": {"dask": {"num_workers": 4}}}
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        assert config["mcp"]["dask"]["num_workers"] == 4
        # Sibling dask defaults and other mcp sub-sections are intact.
        assert config["mcp"]["dask"]["scheduler"] == "distributed"
        assert config["mcp"]["transport"]["kind"] == "stdio"

    def test_handles_malformed_json(self, mock_config_dir):
        """Returns default config for malformed JSON."""
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            f.write("{ invalid json }")

        config = load_config()
        assert config == get_default_config()

    def test_handles_missing_keys(self, mock_config_dir):
        """Merges with defaults for missing top-level keys."""
        custom_config = {"widget": {"server_url": "test.org"}}
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Should have all expected keys
        assert "widget" in config
        assert "tensor_browser" in config
        assert "timeout" in config
        assert "grpc" in config
        assert "mcp" in config

    def test_migrates_installer_flat_process_image_servers(
        self, mock_config_dir
    ):
        """The installer's flat mcp.process_image_servers is relocated."""
        # The shape the biopb installer writes (pre-nesting).
        installer_config = {
            "mcp": {"process_image_servers": ["grpcs://cellpose:443"]}
        }
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(installer_config, f)

        config = load_config()

        assert config["mcp"]["services"]["process_image_servers"] == [
            "grpcs://cellpose:443"
        ]
        # The legacy flat key is removed, not left as a dead duplicate.
        assert "process_image_servers" not in config["mcp"]

    def test_nested_process_image_servers_wins_over_legacy(
        self, mock_config_dir
    ):
        """An explicit nested value takes precedence over the legacy key."""
        both = {
            "mcp": {
                "process_image_servers": ["grpcs://old:443"],
                "services": {"process_image_servers": ["grpcs://new:443"]},
            }
        }
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(both, f)

        config = load_config()

        assert config["mcp"]["services"]["process_image_servers"] == [
            "grpcs://new:443"
        ]
        assert "process_image_servers" not in config["mcp"]

    def test_mcp_has_server_start_timeout(self):
        """The mcp section exposes the autostart boot-wait budget (issue #12)."""
        defaults = get_default_config()
        assert "server_start_timeout" in defaults["mcp"]
        assert defaults["mcp"]["server_start_timeout"] == 60.0


class TestSaveConfig:
    """Tests for save_config function."""

    def test_creates_config_file(self, mock_config_dir):
        """Creates config file in correct location."""
        config = get_default_config()
        config["widget"]["server_url"] = "saved.server.org"

        save_config(config)

        config_path = mock_config_dir / "config.json"
        assert config_path.exists()

    def test_saves_valid_json(self, mock_config_dir):
        """Saves valid JSON that can be loaded."""
        config = get_default_config()
        config["widget"]["detection"]["min_score"] = 0.6

        save_config(config)

        loaded = load_config()
        assert loaded["widget"]["detection"]["min_score"] == 0.6

    def test_preserves_all_values(self, mock_config_dir):
        """Preserves all config values when saving."""
        config = get_default_config()
        config["widget"]["grid"]["2d_size"] = [2048, 2048]
        config["timeout"]["detection_2d"] = 30

        save_config(config)

        config_path = mock_config_dir / "config.json"
        with config_path.open("r") as f:
            saved = json.load(f)

        assert saved["widget"]["grid"]["2d_size"] == [2048, 2048]
        assert saved["timeout"]["detection_2d"] == 30


class TestGetGridParams:
    """Tests for get_grid_params function."""

    def test_2d_grid_params(self):
        """Returns correct 2D grid parameters."""
        defaults = get_default_config()
        grid_size, stride = get_grid_params(False, defaults)

        assert grid_size.shape == (2,)
        assert stride.shape == (2,)
        assert np.array_equal(grid_size, np.array([4096, 4096]))
        assert np.array_equal(stride, np.array([4000, 4000]))

    def test_3d_grid_params(self):
        """Returns correct 3D grid parameters."""
        defaults = get_default_config()
        grid_size, stride = get_grid_params(True, defaults)

        assert grid_size.shape == (3,)
        assert stride.shape == (3,)
        assert np.array_equal(grid_size, np.array([64, 512, 512]))
        assert np.array_equal(stride, np.array([48, 480, 480]))

    def test_custom_grid_params(self):
        """Returns custom grid parameters from config."""
        config = get_default_config()
        config["widget"]["grid"]["2d_size"] = [2048, 2048]
        config["widget"]["grid"]["2d_stride"] = [2000, 2000]
        config["widget"]["grid"]["3d_size"] = [32, 256, 256]
        config["widget"]["grid"]["3d_stride"] = [24, 240, 240]

        grid_2d, stride_2d = get_grid_params(False, config)
        assert np.array_equal(grid_2d, np.array([2048, 2048]))
        assert np.array_equal(stride_2d, np.array([2000, 2000]))

        grid_3d, stride_3d = get_grid_params(True, config)
        assert np.array_equal(grid_3d, np.array([32, 256, 256]))
        assert np.array_equal(stride_3d, np.array([24, 240, 240]))

    def test_returns_int_dtype(self):
        """Returns arrays with int dtype."""
        defaults = get_default_config()
        grid_size, stride = get_grid_params(False, defaults)
        assert grid_size.dtype == np.int64 or grid_size.dtype == np.int32
        assert stride.dtype == np.int64 or stride.dtype == np.int32

    def test_handles_missing_grid_config(self):
        """Returns defaults when grid config is missing."""
        config = {}  # Empty config
        grid = get_default_config()["widget"]["grid"]
        grid_size, stride = get_grid_params(False, config)

        assert np.array_equal(grid_size, np.array(grid["2d_size"]))
        assert np.array_equal(stride, np.array(grid["2d_stride"]))


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG structure."""

    def test_has_all_required_keys(self):
        """DEFAULT_CONFIG contains all expected top-level keys."""
        required_keys = [
            "widget",
            "tensor_browser",
            "timeout",
            "grpc",
            "memory",
            "mcp",
        ]
        for key in required_keys:
            assert key in DEFAULT_CONFIG

    def test_widget_config_complete(self):
        """Widget config groups the demo-widget settings."""
        widget = DEFAULT_CONFIG["widget"]
        # server.url + the orphan "3D" bool folded into the namespace.
        assert widget["server_url"] == "localhost:50051"
        assert widget["is_3d"] is False
        for key in ("min_score", "size_hint", "nms", "z_aspect_ratio"):
            assert key in widget["detection"]
        for key in ("2d_size", "2d_stride", "3d_size", "3d_stride"):
            assert key in widget["grid"]

    def test_timeout_config_complete(self):
        """Timeout config has all timeout fields."""
        required = [
            "health_check",
            "get_op_names",
            "detection_2d",
            "detection_3d",
        ]
        for key in required:
            assert key in DEFAULT_CONFIG["timeout"]

    def test_no_dead_tensor_disable_shm_key(self):
        """tensor_disable_shm was removed (issue #10, no-op after biopb#9)."""
        assert "tensor_disable_shm" not in DEFAULT_CONFIG["mcp"]
        assert "tensor_disable_shm" not in DEFAULT_CONFIG["mcp"]["tensor"]
        assert DEFAULT_CONFIG["mcp"]["tensor"] == {"cache_local": True}

    def test_mcp_sub_sections_present(self):
        """The mcp section is grouped into its concern sub-sections."""
        mcp = DEFAULT_CONFIG["mcp"]
        for section in (
            "transport",
            "kernel",
            "dask",
            "tensor",
            "viewer",
            "services",
        ):
            assert section in mcp
        assert mcp["transport"]["kind"] == "stdio"
        assert mcp["transport"]["port"] == 8765

    def test_mcp_dask_defaults(self):
        """MCP dask defaults to a kernel-local distributed cluster."""
        dask = DEFAULT_CONFIG["mcp"]["dask"]
        # "distributed" + empty address -> auto-spun LocalCluster, the only
        # mode where cancel_job can stop an in-flight compute().
        assert dask["scheduler"] == "distributed"
        assert dask["address"] == ""
        # LocalCluster sizing knobs are present so they can be tuned.
        for key in (
            "num_workers",
            "threads_per_worker",
            "memory_limit",
            "dashboard_address",
        ):
            assert key in dask
        # Dashboard binds loopback only, matching the server security model.
        assert dask["dashboard_address"].startswith("127.0.0.1")


class TestGetSetting:
    """Tests for the dotted-path accessor."""

    def test_reads_present_value(self):
        config = {"mcp": {"transport": {"port": 9999}}}
        assert get_setting(config, "mcp.transport.port") == 9999

    def test_missing_falls_back_to_default_config(self):
        # Empty config -> the value comes from DEFAULT_CONFIG.
        assert get_setting({}, "mcp.transport.port") == 8765
        assert get_setting({}, "widget.server_url") == "localhost:50051"
        assert get_setting({}, "mcp.dask.scheduler") == "distributed"

    def test_partial_path_falls_back(self):
        # A section present but the leaf missing still resolves the default.
        config = {"mcp": {"dask": {"num_workers": 4}}}
        assert get_setting(config, "mcp.dask.scheduler") == "distributed"
        assert get_setting(config, "mcp.dask.num_workers") == 4

    def test_explicit_default_wins_over_default_config(self):
        assert get_setting({}, "mcp.transport.port", default=42) == 42

    def test_mutable_default_is_isolated_copy(self):
        """Mutating a returned mutable default must not touch DEFAULT_CONFIG."""
        servers = get_setting({}, "mcp.services.process_image_servers")
        servers.append("grpc://x:1")
        assert DEFAULT_CONFIG["mcp"]["services"]["process_image_servers"] == []

    def test_unknown_path_without_default_raises(self):
        with pytest.raises(KeyError):
            get_setting({}, "mcp.nope.nada")


class TestConfigSingleton:
    """Tests for the process-wide CONFIG singleton (issue #31).

    The autouse `_isolate_config` fixture (conftest.py) points Path.home at a
    tmp dir and resets CONFIG before/after each test, so these run hermetically.
    """

    def test_lazy_loads_once_until_reload(self, monkeypatch):
        """Disk is read once and memoized until reload()."""
        import biopb_mcp._config as cfg

        calls = {"n": 0}
        real = cfg._read_and_merge_from_disk

        def _counting():
            calls["n"] += 1
            return real()

        monkeypatch.setattr(cfg, "_read_and_merge_from_disk", _counting)

        CONFIG.reload()
        CONFIG.get("pyramid.threshold")
        CONFIG.get("mcp.transport.port")
        assert calls["n"] == 1  # second get hits the cache

        CONFIG.reload()
        CONFIG.get("pyramid.threshold")
        assert calls["n"] == 2  # reload forces a fresh read

    def test_get_falls_back_to_default_config(self):
        """With no file, get() resolves from DEFAULT_CONFIG."""
        assert CONFIG.get("mcp.transport.port") == 8765
        assert CONFIG.get("widget.server_url") == "localhost:50051"

    def test_set_persist_updates_cache_and_file(self):
        """set() with persist=True updates the cache AND the file."""
        CONFIG.set("tensor_browser.server_url", "grpc://set:1")

        assert CONFIG.get("tensor_browser.server_url") == "grpc://set:1"
        with get_config_path().open() as f:
            on_disk = json.load(f)
        assert on_disk["tensor_browser"]["server_url"] == "grpc://set:1"

    def test_set_no_persist_then_save_writes_once(self):
        """persist=False defers the write; save() flushes the batch."""
        CONFIG.set("widget.is_3d", True, persist=False)
        CONFIG.set("widget.server_url", "deferred:1", persist=False)
        # Nothing written yet.
        assert not get_config_path().exists()

        CONFIG.save()
        with get_config_path().open() as f:
            on_disk = json.load(f)
        assert on_disk["widget"]["is_3d"] is True
        assert on_disk["widget"]["server_url"] == "deferred:1"

    def test_set_creates_missing_intermediate_section(self):
        """A dotted path through a missing section is created."""
        CONFIG.set("brandnew.section.leaf", 5, persist=False)
        assert CONFIG.get("brandnew.section.leaf") == 5

    def test_reload_picks_up_external_edit(self):
        """After reload(), an externally-edited file is observed."""
        # Prime the cache with defaults.
        assert (
            CONFIG.get("tensor_browser.server_url") == "grpc://localhost:8815"
        )

        # Edit the file out-of-band; the cache is stale until reload().
        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"tensor_browser": {"server_url": "grpc://ext:9"}}, f)
        assert (
            CONFIG.get("tensor_browser.server_url") == "grpc://localhost:8815"
        )

        CONFIG.reload()
        assert CONFIG.get("tensor_browser.server_url") == "grpc://ext:9"

    def test_load_config_returns_live_singleton(self):
        """The load_config() shim returns the cached singleton dict."""
        assert load_config() is CONFIG.as_dict()

    def test_save_config_shim_refreshes_cache(self):
        """save_config() persists and the next read reflects it."""
        config = get_default_config()
        config["widget"]["detection"]["min_score"] = 0.9
        save_config(config)

        assert CONFIG.get("widget.detection.min_score") == 0.9

    def test_get_serialized_against_concurrent_reload(self, monkeypatch):
        """get() holds the lock across _ensure_loaded + the dotted-path walk.

        A reader paused *between* loading the cache and walking it must not let a
        concurrent reload() null the cache out from under it (which would make
        get() silently return the DEFAULT_CONFIG value instead of the on-disk
        one). We force exactly that interleaving deterministically: a slow
        _ensure_loaded blocks the reader at the critical point, then a reload
        runs in another thread. With the lock held across the whole get(), the
        reload blocks until the reader finishes; without it, the reload nulls
        the cache mid-read.
        """
        import threading

        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"tensor_browser": {"server_url": "grpc://disk:1"}}, f)
        CONFIG.reload()

        real_ensure = CONFIG._ensure_loaded
        at_critical_point = threading.Event()
        proceed = threading.Event()

        def slow_ensure():
            real_ensure()  # cache is now populated from disk
            at_critical_point.set()
            proceed.wait(2)  # pause between load and the dotted-path walk

        monkeypatch.setattr(CONFIG, "_ensure_loaded", slow_ensure)

        result = {}

        def getter():
            result["v"] = CONFIG.get("tensor_browser.server_url")

        g = threading.Thread(target=getter)
        g.start()
        assert at_critical_point.wait(2)

        # Reader is parked mid-get(). Fire a reload from another thread: it must
        # not take effect until the reader releases the lock.
        r = threading.Thread(target=CONFIG.reload)
        r.start()
        r.join(0.3)
        proceed.set()
        g.join(2)
        r.join(2)

        assert result["v"] == "grpc://disk:1"
