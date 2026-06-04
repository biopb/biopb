"""Tests for _config.py configuration management."""

import json

import numpy as np
import pytest

from biopb_mcp._config import (
    DEFAULT_CONFIG,
    get_default_config,
    get_grid_params,
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
            "server": {"url": "custom.server.org"},
            "detection": {"min_score": 0.5},
        }
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Custom values should override defaults
        assert config["server"]["url"] == "custom.server.org"
        assert config["detection"]["min_score"] == 0.5

        # Missing keys should use defaults
        defaults = get_default_config()
        assert config["grid"] == defaults["grid"]

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
        custom_config = {"server": {"url": "test.org"}}
        config_path = mock_config_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(custom_config, f)

        config = load_config()

        # Should have all expected keys
        assert "server" in config
        assert "detection" in config
        assert "grid" in config
        assert "timeout" in config
        assert "grpc" in config


class TestSaveConfig:
    """Tests for save_config function."""

    def test_creates_config_file(self, mock_config_dir):
        """Creates config file in correct location."""
        config = get_default_config()
        config["server"]["url"] = "saved.server.org"

        save_config(config)

        config_path = mock_config_dir / "config.json"
        assert config_path.exists()

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
        config["grid"]["2d_size"] = [2048, 2048]
        config["timeout"]["detection_2d"] = 30

        save_config(config)

        config_path = mock_config_dir / "config.json"
        with config_path.open("r") as f:
            saved = json.load(f)

        assert saved["grid"]["2d_size"] == [2048, 2048]
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
        config["grid"]["2d_size"] = [2048, 2048]
        config["grid"]["2d_stride"] = [2000, 2000]
        config["grid"]["3d_size"] = [32, 256, 256]
        config["grid"]["3d_stride"] = [24, 240, 240]

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
        defaults = get_default_config()
        grid_size, stride = get_grid_params(False, config)

        assert np.array_equal(grid_size, np.array(defaults["grid"]["2d_size"]))
        assert np.array_equal(stride, np.array(defaults["grid"]["2d_stride"]))


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG structure."""

    def test_has_all_required_keys(self):
        """DEFAULT_CONFIG contains all expected top-level keys."""
        required_keys = ["server", "detection", "grid", "timeout", "grpc"]
        for key in required_keys:
            assert key in DEFAULT_CONFIG

    def test_server_config_complete(self):
        """Server config has url."""
        assert "url" in DEFAULT_CONFIG["server"]

    def test_detection_config_complete(self):
        """Detection config has all required fields."""
        required = ["min_score", "size_hint", "nms", "z_aspect_ratio"]
        for key in required:
            assert key in DEFAULT_CONFIG["detection"]

    def test_grid_config_complete(self):
        """Grid config has both 2D and 3D params."""
        assert "2d_size" in DEFAULT_CONFIG["grid"]
        assert "2d_stride" in DEFAULT_CONFIG["grid"]
        assert "3d_size" in DEFAULT_CONFIG["grid"]
        assert "3d_stride" in DEFAULT_CONFIG["grid"]

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

    def test_mcp_dask_defaults(self):
        """MCP dask defaults to a kernel-local distributed cluster."""
        mcp = DEFAULT_CONFIG["mcp"]
        # "distributed" + empty address -> auto-spun LocalCluster, the only
        # mode where cancel_job can stop an in-flight compute().
        assert mcp["dask_scheduler"] == "distributed"
        assert mcp["dask_distributed_address"] == ""
        # LocalCluster sizing knobs are present so they can be tuned.
        for key in (
            "dask_num_workers",
            "dask_threads_per_worker",
            "dask_memory_limit",
            "dask_dashboard_address",
        ):
            assert key in mcp
        # Dashboard binds loopback only, matching the server security model.
        assert mcp["dask_dashboard_address"].startswith("127.0.0.1")
