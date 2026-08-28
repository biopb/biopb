"""Unit tests for the XDG-aware path resolution in :mod:`biopb._locations`.

Pins the on-disk contract every biopb component (and both installers) must agree
on: the three XDG base trees, the derived log/session/pid/sentinel paths, the
env-override rules, and the log rotator. See the module docstring for the policy
(config -> config tree; logs + registry -> state tree; assets -> data tree).
"""

from __future__ import annotations

import logging
import pathlib
import sys

import pytest
from biopb import _locations as L


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    """Isolate the home dir and drop inherited tree vars so tests start from the defaults.

    Isolate via ``Path.home`` rather than ``$HOME``: on Windows ``Path.home()``
    reads ``USERPROFILE``/``HOMEDRIVE``+``HOMEPATH``, not ``HOME``, so a
    ``setenv("HOME")`` would not redirect it and the machine's real home would
    leak into the default-path assertions below.

    ``XDG_*`` no longer changes any path, but CI sets ``XDG_CONFIG_HOME``, and
    leaving it set would make every test here emit the legacy-rename warning.
    Cleared so the assertions run quiet.
    """
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    for var in (
        "BIOPB_CONFIG_HOME",
        "BIOPB_STATE_HOME",
        "BIOPB_DATA_HOME",
        "BIOPB_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(L.SESSIONS_DIR_ENV, raising=False)
    L._LEGACY_XDG_WARNED.clear()


class TestBaseTrees:
    def test_defaults_are_home_based(self, tmp_path):
        assert L.config_dir() == tmp_path / ".config" / "biopb"
        assert L.state_dir() == tmp_path / ".local" / "state" / "biopb"
        assert L.data_dir() == tmp_path / ".local" / "share" / "biopb"

    def test_cache_tree_is_the_one_that_diverges_on_windows(self, tmp_path):
        """Config/state/data hold kilobytes and share one layout everywhere. The
        cache tree is sized for tens of GB, which is exactly what Windows keeps
        out of roaming profiles via %LOCALAPPDATA% -- so only this one splits."""
        if sys.platform == "win32":
            assert L.cache_dir() == tmp_path / "AppData" / "Local" / "biopb" / "Cache"
        else:
            assert L.cache_dir() == tmp_path / ".cache" / "biopb"

    def test_cache_env_honored_when_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_CACHE_HOME", str(tmp_path / "xc"))
        assert L.cache_dir() == tmp_path / "xc" / "biopb"

    def test_biopb_env_honored_when_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "xs"))
        assert L.state_dir() == tmp_path / "xs" / "biopb"
        assert (
            L.tensor_server_log()
            == tmp_path / "xs" / "biopb" / "logs" / "tensor-server.log"
        )

    @pytest.mark.parametrize(
        "var,accessor",
        [
            ("BIOPB_CONFIG_HOME", "config_dir"),
            ("BIOPB_STATE_HOME", "state_dir"),
            ("BIOPB_DATA_HOME", "data_dir"),
            ("BIOPB_CACHE_HOME", "cache_dir"),
        ],
    )
    def test_relative_value_is_rejected(self, monkeypatch, var, accessor):
        # A relative value resolves against each process's cwd, so the installer,
        # the control and the shim would each land somewhere different. Refused
        # loudly rather than silently defaulting -- the value was set on purpose.
        monkeypatch.setenv(var, "relative/nope")
        with pytest.raises(ValueError, match="must be an absolute path"):
            getattr(L, accessor)()

    def test_relative_sessions_override_is_rejected(self, monkeypatch):
        # The registry is the one dir a shim and a control MUST agree on, and
        # they never share a working directory.
        monkeypatch.setenv(L.SESSIONS_DIR_ENV, "relative/nope")
        with pytest.raises(ValueError, match="must be an absolute path"):
            L.sessions_dir()

    def test_absolute_sessions_override_is_honored(self, tmp_path, monkeypatch):
        monkeypatch.setenv(L.SESSIONS_DIR_ENV, str(tmp_path / "reg"))
        assert L.sessions_dir() == tmp_path / "reg"


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


class TestLegacyXdgIsNotRead:
    """biopb owns its own env namespace; ``XDG_*`` must not move any tree (#790).

    The bug: an MCP client (opencode desktop) sets ``XDG_STATE_HOME`` to its own
    working dir, the biopb-mcp shim inherits it, and a control started from a
    terminal does not -- so the two disagree about where the session registry
    lives and the control never sees the session.
    """

    @pytest.mark.parametrize(
        "xdg_var,accessor,rel",
        [
            ("XDG_CONFIG_HOME", "config_dir", (".config",)),
            ("XDG_STATE_HOME", "state_dir", (".local", "state")),
            ("XDG_DATA_HOME", "data_dir", (".local", "share")),
        ],
    )
    def test_xdg_is_ignored(self, tmp_path, monkeypatch, xdg_var, accessor, rel):
        monkeypatch.setenv(xdg_var, str(tmp_path / "someone-elses-cwd"))
        assert getattr(L, accessor)() == tmp_path.joinpath(*rel) / "biopb"

    def test_biopb_var_wins_over_a_stray_xdg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "someone-elses-cwd"))
        monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "mine"))
        assert L.state_dir() == tmp_path / "mine" / "biopb"

    def test_stray_xdg_warns_once_naming_the_replacement(
        self, tmp_path, monkeypatch, caplog
    ):
        # A deployment that relocated via XDG silently moves back to the default,
        # so the rename has to be announced -- but only once per process, since
        # state_dir() is called on nearly every path lookup.
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "someone-elses-cwd"))
        with caplog.at_level(logging.WARNING, logger=L.__name__):
            L.state_dir()
            L.state_dir()
        msgs = [r.getMessage() for r in caplog.records]
        assert len(msgs) == 1
        assert "XDG_STATE_HOME" in msgs[0] and "BIOPB_STATE_HOME" in msgs[0]

    def test_no_warning_when_no_xdg_is_set(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=L.__name__):
            L.state_dir()
        assert caplog.records == []
