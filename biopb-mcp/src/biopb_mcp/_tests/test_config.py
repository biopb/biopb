"""Tests for _config.py configuration management."""

import json

import pytest

from biopb_mcp._config import (
    CONFIG,
    DEFAULT_CONFIG,
    get_config_path,
    get_default_config,
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
            "kernel": {"name": "custom-kernel"},
            "timeout": {"health_check": 0.5},
        }
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Custom values should override defaults
        assert config["kernel"]["name"] == "custom-kernel"
        assert config["timeout"]["health_check"] == 0.5

        # Deep merge: sibling leaves under the overridden section survive.
        defaults = get_default_config()
        assert config["memory"] == defaults["memory"]
        assert config["timeout"]["get_op_names"] == defaults["timeout"]["get_op_names"]

    def test_deep_merge_preserves_sibling_leaves(self, mock_config_dir):
        """A partial nested override touches only its own leaf."""
        custom_config = {"kernel": {"promote_after": 30.0}}
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        assert config["kernel"]["promote_after"] == 30.0
        # Sibling kernel defaults and other sections are intact.
        assert config["kernel"]["name"] == "python3"
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
        custom_config = {"kernel": {"name": "test"}}
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Should have all expected keys
        for key in ("kernel", "timeout", "grpc", "transport"):
            assert key in config


class TestSaveConfig:
    """Tests for save_config function."""

    def test_creates_config_file(self, mock_config_dir):
        """Creates config file in correct location."""
        config = get_default_config()
        config["kernel"]["name"] = "saved"

        save_config(config)

        assert (mock_config_dir / CONFIG_NAME).exists()

    def test_saves_valid_json(self, mock_config_dir):
        """Saves valid JSON that can be loaded."""
        config = get_default_config()
        config["timeout"]["health_check"] = 0.6

        save_config(config)

        loaded = load_config()
        assert loaded["timeout"]["health_check"] == 0.6

    def test_preserves_all_values(self, mock_config_dir):
        """Preserves all config values when saving."""
        config = get_default_config()
        config["memory"]["warn_threshold_mb"] = 100
        config["timeout"]["detection_2d"] = 30

        save_config(config)

        with (mock_config_dir / CONFIG_NAME).open("r") as f:
            saved = json.load(f)

        assert saved["memory"]["warn_threshold_mb"] == 100
        assert saved["timeout"]["detection_2d"] == 30


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG structure."""

    def test_has_all_required_keys(self):
        """DEFAULT_CONFIG contains all expected top-level (flat) sections."""
        required_keys = [
            "timeout",
            "grpc",
            "memory",
            "transport",
            "kernel",
            "viewer",
            "services",
            "observe",
            "update",
        ]
        for key in required_keys:
            assert key in DEFAULT_CONFIG

    def test_timeout_config_complete(self):
        for key in ("health_check", "get_op_names", "detection_2d", "detection_3d"):
            assert key in DEFAULT_CONFIG["timeout"]

    def test_docs_are_flat_scalars(self):
        """The services.docs settings are scalar leaves, not a nested object."""
        services = DEFAULT_CONFIG["services"]
        # Docs ship on: they are package data, so the default install always has
        # something to answer with and there is nothing to fetch.
        assert services["docs_local_dir"] == ""
        # Docs are package data, not a fetch.
        assert "skills_catalog_url" not in services
        assert "skills_cache_ttl" not in services
        # Nothing of the skills catalog survives, nested or flat.
        assert not [k for k in services if k.startswith("skills")]

    def test_platform_dependent_bringup_defaults(self):
        """The startup budget tracks the platform: Windows has no fork(), so
        the kernel starts cold. Asserted against the module's own platform
        constants so the test is correct on whichever OS runs it.
        """
        import os

        from biopb_mcp import _config

        kernel = DEFAULT_CONFIG["kernel"]
        if os.name == "nt":
            assert _config._IS_WINDOWS is True
            assert kernel["startup_timeout"] == 120.0
        else:
            assert _config._IS_WINDOWS is False
            assert kernel["startup_timeout"] == 60.0
        assert kernel["startup_timeout"] == _config._DEFAULT_STARTUP_TIMEOUT


class TestGetSetting:
    """Tests for the dotted-path accessor."""

    def test_reads_present_value(self):
        config = {"transport": {"port": 9999}}
        assert get_setting(config, "transport.port") == 9999

    def test_missing_falls_back_to_default_config(self):
        assert get_setting({}, "transport.port") == 8765
        assert get_setting({}, "kernel.name") == "python3"

    def test_partial_path_falls_back(self):
        config = {"kernel": {"promote_after": 30.0}}
        assert get_setting(config, "kernel.name") == "python3"
        assert get_setting(config, "kernel.promote_after") == 30.0

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
        CONFIG.get("kernel.name")
        CONFIG.get("transport.port")
        assert calls["n"] == 1  # second get hits the cache

        CONFIG.reload()
        CONFIG.get("kernel.name")
        assert calls["n"] == 2  # reload forces a fresh read

    def test_get_falls_back_to_default_config(self):
        assert CONFIG.get("transport.port") == 8765
        assert CONFIG.get("kernel.name") == "python3"

    def test_set_persist_updates_cache_and_file(self):
        """set() with persist=True updates the cache AND the file."""
        CONFIG.set("kernel.name", "set-1")

        assert CONFIG.get("kernel.name") == "set-1"
        with get_config_path().open() as f:
            on_disk = json.load(f)
        assert on_disk["kernel"]["name"] == "set-1"

    def test_set_no_persist_then_save_writes_once(self):
        """persist=False defers the write; save() flushes the batch."""
        CONFIG.set("viewer.async_slicing", False, persist=False)
        CONFIG.set("kernel.name", "deferred", persist=False)
        assert not get_config_path().exists()

        CONFIG.save()
        with get_config_path().open() as f:
            on_disk = json.load(f)
        assert on_disk["viewer"]["async_slicing"] is False
        assert on_disk["kernel"]["name"] == "deferred"

    def test_set_creates_missing_intermediate_section(self):
        CONFIG.set("brandnew.section.leaf", 5, persist=False)
        assert CONFIG.get("brandnew.section.leaf") == 5

    def test_reload_picks_up_external_edit(self):
        assert CONFIG.get("kernel.name") == "python3"

        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"kernel": {"name": "ext"}}, f)
        assert CONFIG.get("kernel.name") == "python3"

        CONFIG.reload()
        assert CONFIG.get("kernel.name") == "ext"

    def test_load_config_returns_live_singleton(self):
        assert load_config() is CONFIG.as_dict()

    def test_save_config_shim_refreshes_cache(self):
        config = get_default_config()
        config["timeout"]["health_check"] = 0.9
        save_config(config)

        assert CONFIG.get("timeout.health_check") == 0.9

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
            json.dump({"kernel": {"name": "disk"}}, f)
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
            result["v"] = CONFIG.get("kernel.name")

        g = threading.Thread(target=getter)
        g.start()
        assert at_critical_point.wait(2)

        r = threading.Thread(target=CONFIG.reload)
        r.start()
        r.join(0.3)
        proceed.set()
        g.join(2)
        r.join(2)

        assert result["v"] == "disk"


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

    def test_out_of_range_clamped_to_default(self, mock_config_dir):
        defaults = get_default_config()
        config = _write_and_load(
            mock_config_dir,
            {"grpc": {"max_concurrent_calls": 0}, "timeout": {"health_check": -1}},
        )
        assert (
            config["grpc"]["max_concurrent_calls"]
            == defaults["grpc"]["max_concurrent_calls"]
        )
        assert config["timeout"]["health_check"] == defaults["timeout"]["health_check"]

    def test_bad_enum_clamped_to_default(self, mock_config_dir):
        config = _write_and_load(
            mock_config_dir,
            {
                "transport": {"kind": "websocket"},
            },
        )
        assert get_setting(config, "transport.kind") == "stdio"

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
        the Range check and is replaced by the numeric default.

        The written value is deliberately *not* the default, so a pass means the
        default won rather than the string having been coerced to its own value.
        """
        from biopb_mcp._config import GrpcConfig

        default = GrpcConfig().max_concurrent_calls
        config = _write_and_load(
            mock_config_dir, {"grpc": {"max_concurrent_calls": "8"}}
        )
        assert get_setting(config, "grpc.max_concurrent_calls") == default
        assert isinstance(get_setting(config, "grpc.max_concurrent_calls"), int)

    def test_zero_is_valid_where_it_disables(self, mock_config_dir):
        """0 is a documented sentinel (disables the watchdog), so it must pass
        validation, not be clamped."""
        config = _write_and_load(mock_config_dir, {"kernel": {"watchdog_interval": 0}})
        assert get_setting(config, "kernel.watchdog_interval") == 0

    def test_valid_values_untouched(self, mock_config_dir):
        config = _write_and_load(
            mock_config_dir,
            {
                "grpc": {"max_concurrent_calls": 2},
                "transport": {"kind": "http", "port": 9000},
            },
        )
        assert get_setting(config, "grpc.max_concurrent_calls") == 2
        assert get_setting(config, "transport.kind") == "http"
        assert get_setting(config, "transport.port") == 9000

    def test_warns_naming_key_value_and_range(self, mock_config_dir, caplog):
        with caplog.at_level("WARNING"):
            _write_and_load(mock_config_dir, {"grpc": {"max_concurrent_calls": 0}})
        msg = "\n".join(caplog.messages)
        assert "grpc.max_concurrent_calls" in msg
        assert "0" in msg
        assert ">= 1" in msg

    def test_malformed_file_still_falls_back_to_defaults(self, mock_config_dir):
        """A non-dict-JSON file is unaffected by validation (still defaults)."""
        config_path = mock_config_dir / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("not json {{{")
        CONFIG.reload()
        assert load_config() == get_default_config()

    def test_shipped_defaults_pass_the_whole_check(self):
        """The clamp target must itself be valid -- otherwise a bad leaf would be replaced by an equally invalid default and
        the config would never converge."""
        from biopb_mcp._config import config_problems

        assert config_problems(DEFAULT_CONFIG) == []

    def test_shipped_defaults_satisfy_every_constraint(self):
        """DEFAULT_CONFIG must itself pass validation -- otherwise a bad leaf
        would clamp to a default that is *also* invalid. Iterates the class-keyed
        _CONSTRAINTS through the section->class map."""
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
