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

CONFIG_NAME = "mcp-config.json"


@pytest.fixture
def mock_config_dir(monkeypatch, tmp_path):
    """Redirect the home-relative config dir (~/.config/biopb) to a tmp path."""
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path / ".config" / "biopb"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_returns_default_when_no_file(self, mock_config_dir):
        """Returns default config when file doesn't exist."""
        config = load_config()
        assert config == get_default_config()

    def test_loads_existing_config(self, mock_config_dir):
        """Loads and merges existing config file."""
        custom_config = {
            "widget": {"server_url": "custom.server.org"},
            "detection": {"min_score": 0.5},
        }
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Custom values should override defaults
        assert config["widget"]["server_url"] == "custom.server.org"
        assert config["detection"]["min_score"] == 0.5

        # Deep merge: sibling leaves under the overridden section survive.
        defaults = get_default_config()
        assert config["grid"] == defaults["grid"]
        assert config["detection"]["nms"] == defaults["detection"]["nms"]

    def test_deep_merge_preserves_sibling_leaves(self, mock_config_dir):
        """A partial nested override touches only its own leaf."""
        custom_config = {"dask": {"num_workers": 4}}
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        assert config["dask"]["num_workers"] == 4
        # Sibling dask defaults and other sections are intact.
        assert config["dask"]["scheduler"] == "distributed"
        assert config["transport"]["kind"] == "stdio"

    def test_handles_malformed_json(self, mock_config_dir):
        """Returns default config for malformed JSON."""
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            f.write("{ invalid json }")

        config = load_config()
        assert config == get_default_config()

    def test_handles_missing_keys(self, mock_config_dir):
        """Merges with defaults for missing top-level keys."""
        custom_config = {"widget": {"server_url": "test.org"}}
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Should have all expected keys
        for key in ("widget", "tensor_browser", "timeout", "grpc", "transport"):
            assert key in config

    def test_has_server_start_timeout(self):
        """The transport section exposes the autostart boot-wait budget (#12)."""
        defaults = get_default_config()
        assert defaults["transport"]["server_start_timeout"] == 60.0


class TestSaveConfig:
    """Tests for save_config function."""

    def test_creates_config_file(self, mock_config_dir):
        """Creates config file in correct location."""
        config = get_default_config()
        config["widget"]["server_url"] = "saved.server.org"

        save_config(config)

        assert (mock_config_dir / CONFIG_NAME).exists()

    def test_saves_valid_json(self, mock_config_dir):
        """Saves valid JSON that can be loaded."""
        config = get_default_config()
        config["detection"]["min_score"] = 0.6

        save_config(config)

        loaded = load_config()
        assert loaded["detection"]["min_score"] == 0.6

    def test_preserves_all_values(self, mock_config_dir):
        """Preserves all config values when saving."""
        config = get_default_config()
        config["grid"]["size_2d"] = [2048, 2048]
        config["timeout"]["detection_2d"] = 30

        save_config(config)

        with (mock_config_dir / CONFIG_NAME).open("r") as f:
            saved = json.load(f)

        assert saved["grid"]["size_2d"] == [2048, 2048]
        assert saved["timeout"]["detection_2d"] == 30


class TestGetGridParams:
    """Tests for get_grid_params function."""

    def test_2d_grid_params(self):
        defaults = get_default_config()
        grid_size, stride = get_grid_params(False, defaults)

        assert grid_size.shape == (2,)
        assert stride.shape == (2,)
        assert np.array_equal(grid_size, np.array([4096, 4096]))
        assert np.array_equal(stride, np.array([4000, 4000]))

    def test_3d_grid_params(self):
        defaults = get_default_config()
        grid_size, stride = get_grid_params(True, defaults)

        assert grid_size.shape == (3,)
        assert stride.shape == (3,)
        assert np.array_equal(grid_size, np.array([64, 512, 512]))
        assert np.array_equal(stride, np.array([48, 480, 480]))

    def test_custom_grid_params(self):
        config = get_default_config()
        config["grid"]["size_2d"] = [2048, 2048]
        config["grid"]["stride_2d"] = [2000, 2000]
        config["grid"]["size_3d"] = [32, 256, 256]
        config["grid"]["stride_3d"] = [24, 240, 240]

        grid_2d, stride_2d = get_grid_params(False, config)
        assert np.array_equal(grid_2d, np.array([2048, 2048]))
        assert np.array_equal(stride_2d, np.array([2000, 2000]))

        grid_3d, stride_3d = get_grid_params(True, config)
        assert np.array_equal(grid_3d, np.array([32, 256, 256]))
        assert np.array_equal(stride_3d, np.array([24, 240, 240]))

    def test_returns_int_dtype(self):
        defaults = get_default_config()
        grid_size, stride = get_grid_params(False, defaults)
        assert grid_size.dtype in (np.int64, np.int32)
        assert stride.dtype in (np.int64, np.int32)

    def test_handles_missing_grid_config(self):
        """Returns defaults when grid config is missing."""
        grid = get_default_config()["grid"]
        grid_size, stride = get_grid_params(False, {})  # empty config

        assert np.array_equal(grid_size, np.array(grid["size_2d"]))
        assert np.array_equal(stride, np.array(grid["stride_2d"]))


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG structure."""

    def test_has_all_required_keys(self):
        """DEFAULT_CONFIG contains all expected top-level (flat) sections."""
        required_keys = [
            "widget",
            "detection",
            "grid",
            "tensor_browser",
            "pyramid",
            "timeout",
            "grpc",
            "memory",
            "transport",
            "kernel",
            "dask",
            "tensor",
            "viewer",
            "services",
            "observe",
            "update",
        ]
        for key in required_keys:
            assert key in DEFAULT_CONFIG

    def test_widget_and_detection_and_grid(self):
        """The demo-widget settings live in flat widget/detection/grid sections."""
        assert DEFAULT_CONFIG["widget"]["server_url"] == "localhost:50051"
        assert DEFAULT_CONFIG["widget"]["is_3d"] is False
        for key in ("min_score", "size_hint", "nms", "z_aspect_ratio"):
            assert key in DEFAULT_CONFIG["detection"]
        for key in ("size_2d", "stride_2d", "size_3d", "stride_3d"):
            assert key in DEFAULT_CONFIG["grid"]

    def test_timeout_config_complete(self):
        for key in ("health_check", "get_op_names", "detection_2d", "detection_3d"):
            assert key in DEFAULT_CONFIG["timeout"]

    def test_skills_are_flat_scalars(self):
        """The former services.skills object is flattened to scalar leaves."""
        services = DEFAULT_CONFIG["services"]
        assert services["skills_enabled"] is True
        assert services["skills_catalog_url"].startswith("https://")
        assert services["skills_cache_ttl"] == 3600
        # No nested object survives.
        assert "skills" not in services

    def test_dask_defaults(self):
        """MCP dask defaults to a session-child-owned distributed cluster."""
        dask = DEFAULT_CONFIG["dask"]
        assert dask["scheduler"] == "distributed"
        assert dask["address"] == ""
        assert "owner" not in dask  # the escape hatch was removed
        for key in (
            "num_workers",
            "threads_per_worker",
            "memory_limit",
            "dashboard_address",
        ):
            assert key in dask
        assert dask["dashboard_address"].startswith("127.0.0.1")

    def test_platform_dependent_bringup_defaults(self):
        """startup_timeout / num_workers defaults track the platform.

        Windows has no fork(), so dask LocalCluster workers cold-spawn; the
        defaults widen the startup budget and cap the worker count there, while
        POSIX keeps the lean values. Asserted against the module's own platform
        constants so the test is correct on whichever OS runs it.
        """
        import os

        from biopb_mcp import _config

        kernel = DEFAULT_CONFIG["kernel"]
        dask = DEFAULT_CONFIG["dask"]
        if os.name == "nt":
            assert _config._IS_WINDOWS is True
            assert kernel["startup_timeout"] == 120.0
            assert dask["num_workers"] == 4
        else:
            assert _config._IS_WINDOWS is False
            assert kernel["startup_timeout"] == 60.0
            assert dask["num_workers"] == 0
        assert kernel["startup_timeout"] == _config._DEFAULT_STARTUP_TIMEOUT
        assert dask["num_workers"] == _config._DEFAULT_DASK_NUM_WORKERS

    def test_tensor_health_poll_defaults(self):
        """The background source watcher's backoff bounds (issue #44)."""
        tensor = DEFAULT_CONFIG["tensor"]
        assert tensor["health_poll_min_interval"] == 2.0
        assert tensor["health_poll_max_interval"] == 60.0
        assert tensor["health_poll_min_interval"] < tensor["health_poll_max_interval"]


class TestGetSetting:
    """Tests for the dotted-path accessor."""

    def test_reads_present_value(self):
        config = {"transport": {"port": 9999}}
        assert get_setting(config, "transport.port") == 9999

    def test_missing_falls_back_to_default_config(self):
        assert get_setting({}, "transport.port") == 8765
        assert get_setting({}, "widget.server_url") == "localhost:50051"
        assert get_setting({}, "dask.scheduler") == "distributed"

    def test_partial_path_falls_back(self):
        config = {"dask": {"num_workers": 4}}
        assert get_setting(config, "dask.scheduler") == "distributed"
        assert get_setting(config, "dask.num_workers") == 4

    def test_explicit_default_wins_over_default_config(self):
        assert get_setting({}, "transport.port", default=42) == 42

    def test_mutable_default_is_isolated_copy(self):
        """Mutating a returned mutable default must not touch DEFAULT_CONFIG."""
        servers = get_setting({}, "services.process_image_servers")
        servers.append("grpc://x:1")
        assert DEFAULT_CONFIG["services"]["process_image_servers"] == []

    def test_unknown_path_without_default_raises(self):
        with pytest.raises(KeyError):
            get_setting({}, "nope.nada")


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
        CONFIG.get("transport.port")
        assert calls["n"] == 1  # second get hits the cache

        CONFIG.reload()
        CONFIG.get("pyramid.threshold")
        assert calls["n"] == 2  # reload forces a fresh read

    def test_get_falls_back_to_default_config(self):
        assert CONFIG.get("transport.port") == 8765
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
        assert not get_config_path().exists()

        CONFIG.save()
        with get_config_path().open() as f:
            on_disk = json.load(f)
        assert on_disk["widget"]["is_3d"] is True
        assert on_disk["widget"]["server_url"] == "deferred:1"

    def test_set_creates_missing_intermediate_section(self):
        CONFIG.set("brandnew.section.leaf", 5, persist=False)
        assert CONFIG.get("brandnew.section.leaf") == 5

    def test_reload_picks_up_external_edit(self):
        assert CONFIG.get("tensor_browser.server_url") == "grpc://localhost:8815"

        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"tensor_browser": {"server_url": "grpc://ext:9"}}, f)
        assert CONFIG.get("tensor_browser.server_url") == "grpc://localhost:8815"

        CONFIG.reload()
        assert CONFIG.get("tensor_browser.server_url") == "grpc://ext:9"

    def test_load_config_returns_live_singleton(self):
        assert load_config() is CONFIG.as_dict()

    def test_save_config_shim_refreshes_cache(self):
        config = get_default_config()
        config["detection"]["min_score"] = 0.9
        save_config(config)

        assert CONFIG.get("detection.min_score") == 0.9

    def test_get_serialized_against_concurrent_reload(self, monkeypatch):
        """get() holds the lock across _ensure_loaded + the dotted-path walk.

        A reader paused *between* loading the cache and walking it must not let a
        concurrent reload() null the cache out from under it. We force exactly
        that interleaving deterministically.
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

        r = threading.Thread(target=CONFIG.reload)
        r.start()
        r.join(0.3)
        proceed.set()
        g.join(2)
        r.join(2)

        assert result["v"] == "grpc://disk:1"


def _write_and_load(mock_config_dir, raw: dict) -> dict:
    """Write *raw* as the config file under the mocked home and load it fresh."""
    config_path = mock_config_dir / CONFIG_NAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(raw, f)
    CONFIG.reload()
    return load_config()


class TestValidation:
    """Out-of-range / bad-enum leaves are warned and reset to defaults (#182)."""

    def test_pyramid_out_of_range_clamped_to_default(self, mock_config_dir):
        """The three shared pyramid knobs are the same bug as the tensor server:
        a bad value would silently break build_pyramid_levels. Each resets."""
        defaults = get_default_config()
        config = _write_and_load(
            mock_config_dir,
            {
                "pyramid": {
                    "downscale_factor": 1,  # >= 2 required (==1 -> no pyramid)
                    "pixel_budget_cubic_root": 0,  # >= 1 (0 -> infinite loop)
                    "threshold": -5,  # >= 1
                }
            },
        )
        for key in ("downscale_factor", "pixel_budget_cubic_root", "threshold"):
            assert get_setting(config, f"pyramid.{key}") == defaults["pyramid"][key]

    def test_bad_enum_clamped_to_default(self, mock_config_dir):
        config = _write_and_load(
            mock_config_dir,
            {
                "transport": {"kind": "websocket"},
                "dask": {"scheduler": "bogus"},
            },
        )
        assert get_setting(config, "transport.kind") == "stdio"
        assert get_setting(config, "dask.scheduler") == "distributed"

    def test_port_out_of_range_clamped(self, mock_config_dir):
        config = _write_and_load(mock_config_dir, {"transport": {"port": 99999}})
        assert get_setting(config, "transport.port") == 8765

    def test_session_log_keep_default_is_five(self):
        assert get_setting(DEFAULT_CONFIG, "transport.session_log_keep") == 5

    def test_session_log_keep_below_one_reset_to_default(self, mock_config_dir):
        # Range(min=1): must always keep at least the current session's log.
        config = _write_and_load(
            mock_config_dir, {"transport": {"session_log_keep": 0}}
        )
        assert get_setting(config, "transport.session_log_keep") == 5

    def test_string_number_reset_to_default(self, mock_config_dir):
        """The no-coercion wrinkle: a JSON string where a number is expected fails
        the Range check and is replaced by the numeric default."""
        config = _write_and_load(
            mock_config_dir, {"pyramid": {"downscale_factor": "4"}}
        )
        assert get_setting(config, "pyramid.downscale_factor") == 4
        assert isinstance(get_setting(config, "pyramid.downscale_factor"), int)

    def test_zero_is_valid_where_it_disables(self, mock_config_dir):
        """0 is a documented sentinel for several knobs (disables the watcher /
        lets dask pick), so it must pass validation, not be clamped."""
        config = _write_and_load(
            mock_config_dir,
            {
                "dask": {"num_workers": 0},
                "kernel": {"watchdog_interval": 0},
                "tensor": {"health_poll_min_interval": 0},
            },
        )
        assert get_setting(config, "dask.num_workers") == 0
        assert get_setting(config, "kernel.watchdog_interval") == 0
        assert get_setting(config, "tensor.health_poll_min_interval") == 0

    def test_valid_values_untouched(self, mock_config_dir):
        config = _write_and_load(
            mock_config_dir,
            {
                "pyramid": {"downscale_factor": 2},
                "transport": {"kind": "http", "port": 9000},
            },
        )
        assert get_setting(config, "pyramid.downscale_factor") == 2
        assert get_setting(config, "transport.kind") == "http"
        assert get_setting(config, "transport.port") == 9000

    def test_health_poll_inverted_reset_to_defaults(self, mock_config_dir):
        """min > max would invert the exponential backoff; both reset."""
        defaults = get_default_config()
        config = _write_and_load(
            mock_config_dir,
            {
                "tensor": {
                    "health_poll_min_interval": 90.0,
                    "health_poll_max_interval": 10.0,
                }
            },
        )
        assert (
            get_setting(config, "tensor.health_poll_min_interval")
            == defaults["tensor"]["health_poll_min_interval"]
        )
        assert (
            get_setting(config, "tensor.health_poll_max_interval")
            == defaults["tensor"]["health_poll_max_interval"]
        )

    def test_warns_naming_key_value_and_range(self, mock_config_dir, caplog):
        with caplog.at_level("WARNING"):
            _write_and_load(mock_config_dir, {"pyramid": {"downscale_factor": 1}})
        msg = "\n".join(caplog.messages)
        assert "pyramid.downscale_factor" in msg
        assert "1" in msg
        assert ">= 2" in msg

    def test_malformed_file_still_falls_back_to_defaults(self, mock_config_dir):
        """A non-dict-JSON file is unaffected by validation (still defaults)."""
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("not json {{{")
        CONFIG.reload()
        assert load_config() == get_default_config()

    def test_shipped_defaults_pass_the_whole_check(self):
        """The clamp target must itself be valid, cross-field rules included --
        otherwise a bad leaf would be replaced by an equally invalid default and
        the config would never converge."""
        from biopb_mcp._config import config_problems

        assert config_problems(DEFAULT_CONFIG) == []

    def test_shipped_defaults_satisfy_every_constraint(self):
        """DEFAULT_CONFIG must itself pass validation -- otherwise a bad leaf
        would clamp to a default that is *also* invalid. Iterates the class-keyed
        _CONSTRAINTS through the section->class map, pinning the pyramid defaults
        against the shared constraint rows too."""
        from biopb_mcp._config import (
            _CONSTRAINTS,
            _MISSING,
            _SECTION_CLASSES,
            _walk_path,
        )

        for section, cls in _SECTION_CLASSES.items():
            for field_name, constraint in _CONSTRAINTS.get(cls.__name__, {}).items():
                value = _walk_path(DEFAULT_CONFIG, (section, field_name))
                assert value is not _MISSING, f"{section}.{field_name} missing"
                assert constraint.ok(value), (
                    f"default {section}.{field_name}={value!r} violates "
                    f"{constraint.describe()}"
                )
