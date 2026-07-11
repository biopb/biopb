"""Unit tests for the shared MCP-session filesystem registry.

The registry (:mod:`biopb._config_sessions`) is the seam between the biopb-mcp
shim (writer) and the control plane (reader) — neither imports the other, so its
on-disk contract is pinned here in the core ``biopb`` package. See
``biopb-mcp/docs/mcp-dedaemonization-migration.md`` §6.1.
"""

import json
import os

import pytest
from biopb import _config_sessions as reg


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    """Point the registry at a per-test dir so nothing touches the real home."""
    monkeypatch.setenv("BIOPB_SESSIONS_DIR", str(tmp_path / "sessions"))


def _live_pid():
    return os.getpid()  # this process is certainly alive


class TestRegisterRead:
    def test_register_then_read_roundtrips_fields(self):
        reg.register("s1", port=54321, pid=_live_pid(), mcp_url="http://x/mcp")
        rec = reg.read_session("s1")
        assert rec["session_id"] == "s1"
        assert rec["port"] == 54321
        assert rec["pid"] == _live_pid()
        assert rec["mcp_url"] == "http://x/mcp"
        assert rec["host"] == "127.0.0.1"
        assert isinstance(rec["started_at"], (int, float))

    def test_extra_fields_are_stored(self):
        reg.register("s1", port=1, pid=_live_pid(), base_path="/session/s1/")
        assert reg.read_session("s1")["base_path"] == "/session/s1/"

    def test_read_missing_is_none(self):
        assert reg.read_session("nope") is None

    def test_register_is_atomic_overwrite(self):
        reg.register("s1", port=1, pid=_live_pid())
        reg.register("s1", port=2, pid=_live_pid())  # re-register same id
        assert reg.read_session("s1")["port"] == 2
        assert len(list(reg.sessions_dir().glob("*.json"))) == 1  # no temp litter

    def test_env_override_directs_writes(self, tmp_path):
        reg.register("s1", port=1, pid=_live_pid())
        assert (tmp_path / "sessions" / "s1.json").exists()


class TestUnregister:
    def test_unregister_removes_record(self):
        reg.register("s1", port=1, pid=_live_pid())
        reg.unregister("s1")
        assert reg.read_session("s1") is None

    def test_unregister_missing_is_noop(self):
        reg.unregister("never")  # must not raise


class TestListSessions:
    def test_lists_all_live_newest_first(self):
        reg.register("20260101-000000-1", port=1, pid=_live_pid())
        reg.register("20260101-000100-2", port=2, pid=_live_pid())
        ids = [r["session_id"] for r in reg.list_sessions()]
        assert ids == ["20260101-000100-2", "20260101-000000-1"]  # newest first

    def test_prunes_and_unlinks_dead_pid_records(self):
        reg.register("dead", port=1, pid=_dead_pid())
        reg.register("live", port=2, pid=_live_pid())
        ids = [r["session_id"] for r in reg.list_sessions()]
        assert ids == ["live"]
        # The stale record is removed from disk, not merely filtered — self-heal.
        assert not (reg.sessions_dir() / "dead.json").exists()

    def test_prune_disabled_keeps_dead_records(self):
        reg.register("dead", port=1, pid=_dead_pid())
        ids = [r["session_id"] for r in reg.list_sessions(prune=False)]
        assert ids == ["dead"]
        assert (reg.sessions_dir() / "dead.json").exists()

    def test_missing_pid_is_kept_fail_open(self):
        # A record with no usable pid can't be disproven → kept even with prune.
        p = reg.sessions_dir() / "nopid.json"
        p.write_text(json.dumps({"session_id": "nopid", "port": 9}))
        assert [r["session_id"] for r in reg.list_sessions()] == ["nopid"]

    def test_corrupt_record_is_skipped_not_fatal(self):
        (reg.sessions_dir() / "bad.json").write_text("{ not json")
        reg.register("good", port=1, pid=_live_pid())
        assert [r["session_id"] for r in reg.list_sessions()] == ["good"]

    def test_empty_registry_is_empty_list(self):
        assert reg.list_sessions() == []


class TestPidAlive:
    def test_live_pid(self):
        assert reg._pid_alive(os.getpid()) is True

    def test_dead_pid(self):
        assert reg._pid_alive(_dead_pid()) is False

    def test_nonsense_pid_fails_open(self):
        assert reg._pid_alive(0) is True
        assert reg._pid_alive(-1) is True
        assert reg._pid_alive("abc") is True
        assert reg._pid_alive(None) is True


def _dead_pid():
    """A pid that is (almost certainly) not alive: spawn a child, reap it, reuse
    its pid before the OS recycles it."""
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid
