"""Unit tests for the XDG-aware path resolution in :mod:`biopb._config_location`.

Pins the on-disk contract every biopb component (and both installers) must agree
on: the three XDG base trees, the derived log/session/pid/sentinel paths, the
env-override rules, and the log rotator. See the module docstring for the policy
(config -> config tree; logs + registry -> state tree; assets -> data tree).
"""

from __future__ import annotations

import pytest
from biopb import _config_location as L


@pytest.fixture(autouse=True)
def _clean_xdg(tmp_path, monkeypatch):
    """Isolate HOME and drop inherited XDG_* so a test starts from the defaults."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(L.SESSIONS_DIR_ENV, raising=False)


class TestBaseTrees:
    def test_defaults_are_home_based(self, tmp_path):
        assert L.config_dir() == tmp_path / ".config" / "biopb"
        assert L.state_dir() == tmp_path / ".local" / "state" / "biopb"
        assert L.data_dir() == tmp_path / ".local" / "share" / "biopb"

    def test_xdg_env_honored_when_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xs"))
        assert L.state_dir() == tmp_path / "xs" / "biopb"
        assert (
            L.tensor_server_log()
            == tmp_path / "xs" / "biopb" / "logs" / "tensor-server.log"
        )

    def test_relative_xdg_value_is_ignored(self, tmp_path, monkeypatch):
        # The XDG spec says a non-absolute base dir is invalid -> fall back to the default.
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative/nope")
        assert L.config_dir() == tmp_path / ".config" / "biopb"


class TestDerivedPaths:
    def test_logs_live_in_state_tree(self, tmp_path):
        st = tmp_path / ".local" / "state" / "biopb"
        assert L.tensor_server_log() == st / "logs" / "tensor-server.log"
        assert L.control_log() == st / "logs" / "control.log"
        assert L.mcp_server_log() == st / "mcp" / "mcp-server.log"

    def test_registry_and_control_files_in_state_tree(self, tmp_path):
        st = tmp_path / ".local" / "state" / "biopb"
        assert L.sessions_dir() == st / "sessions"
        assert L.control_pid_file() == st / "control.pid"
        assert L.control_stop_sentinel() == st / "control.stop"
        assert L.tensor_stop_sentinel() == st / "tensor-server.stop"

    def test_assets_stay_in_data_tree(self, tmp_path):
        data = tmp_path / ".local" / "share" / "biopb"
        assert L.webapp_dir() == data / "webapp"
        assert L.samples_dir() == data / "samples"

    def test_config_files_in_config_tree(self, tmp_path):
        assert L.mcp_config_path() == tmp_path / ".config" / "biopb" / "mcp-config.json"

    def test_dir_accessors_create_on_access(self):
        # log_dir / sessions_dir / mcp_log_dir mkdir; file accessors do not.
        assert L.log_dir().is_dir()
        assert L.sessions_dir().is_dir()
        assert L.mcp_log_dir().is_dir()


class TestSessionsOverride:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(L.SESSIONS_DIR_ENV, str(tmp_path / "custom"))
        assert L.sessions_dir() == tmp_path / "custom"


class TestRotateLog:
    def test_noop_below_threshold(self, tmp_path):
        f = tmp_path / "x.log"
        f.write_text("small")
        L.rotate_log(f, max_bytes=1024)
        assert f.exists() and not (tmp_path / "x.log.1").exists()

    def test_rotates_over_threshold(self, tmp_path):
        f = tmp_path / "x.log"
        f.write_bytes(b"a" * 2048)
        L.rotate_log(f, max_bytes=1024)
        assert not f.exists()
        assert (tmp_path / "x.log.1").read_bytes() == b"a" * 2048

    def test_shifts_existing_backups(self, tmp_path):
        f = tmp_path / "x.log"
        f.write_bytes(b"a" * 2048)
        (tmp_path / "x.log.1").write_text("old1")
        L.rotate_log(f, max_bytes=1024, backup_count=3)
        assert (tmp_path / "x.log.2").read_text() == "old1"
        assert (tmp_path / "x.log.1").read_bytes() == b"a" * 2048
