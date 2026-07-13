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
        # An identity token for the pid is recorded (None where unavailable).
        assert "create_time" in rec

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

    def test_same_second_sessions_ordered_by_started_at(self):
        # Two sessions in the same one-second bucket with un-padded pids: a
        # filename reverse-sort puts "-9" after "-42" lexically and mis-orders
        # them, so ordering must key on started_at, not the name (biopb/biopb#421).
        pid, ct = _live_pid(), reg.process_create_time(_live_pid())
        older = {
            "session_id": "20260101-000000-9",
            "port": 1,
            "pid": pid,
            "create_time": ct,
            "started_at": 100.0,
        }
        newer = {
            "session_id": "20260101-000000-42",
            "port": 2,
            "pid": pid,
            "create_time": ct,
            "started_at": 200.0,
        }
        (reg.sessions_dir() / "20260101-000000-9.json").write_text(json.dumps(older))
        (reg.sessions_dir() / "20260101-000000-42.json").write_text(json.dumps(newer))
        ids = [r["session_id"] for r in reg.list_sessions()]
        assert ids == ["20260101-000000-42", "20260101-000000-9"]  # newest (200) 1st

    def test_record_missing_started_at_sorts_last(self):
        pid, ct = _live_pid(), reg.process_create_time(_live_pid())
        (reg.sessions_dir() / "a.json").write_text(
            json.dumps(
                {
                    "session_id": "a",
                    "port": 1,
                    "pid": pid,
                    "create_time": ct,
                    "started_at": 50.0,
                }
            )
        )
        (reg.sessions_dir() / "b.json").write_text(
            json.dumps(  # no started_at
                {"session_id": "b", "port": 2, "pid": pid, "create_time": ct}
            )
        )
        assert [r["session_id"] for r in reg.list_sessions()] == ["a", "b"]

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


class TestRecordIsLive:
    def test_live_pid_with_matching_token(self):
        rec = {"pid": os.getpid(), "create_time": reg.process_create_time(os.getpid())}
        assert reg._record_is_live(rec) is True

    def test_dead_pid(self):
        assert reg._record_is_live({"pid": _dead_pid(), "create_time": None}) is False

    def test_nonsense_pid_fails_open(self):
        for pid in (0, -1, "abc", None):
            assert reg._record_is_live({"pid": pid}) is True

    def test_missing_token_degrades_to_liveness(self):
        # No recorded create_time (e.g. registered on a platform without one) ->
        # liveness-only; a live pid is kept.
        assert reg._record_is_live({"pid": os.getpid()}) is True

    def test_pid_reuse_detected_via_create_time(self):
        # An alive pid whose recorded identity token no longer matches is a
        # recycled pid naming a different process -> treated as gone (biopb#138).
        real = reg.process_create_time(os.getpid())
        if real is None:
            pytest.skip("platform has no create-time token (e.g. macOS)")
        assert (
            reg._record_is_live({"pid": os.getpid(), "create_time": real + 1}) is False
        )
        assert reg._record_is_live({"pid": os.getpid(), "create_time": real}) is True


def test_resolve_prunes_a_pid_reused_ghost():
    # End to end: a record for a live pid but a stale identity token must not
    # resolve, and the ghost is unlinked -- so the control never routes a session
    # to an unrelated process that inherited the recycled pid.
    real = reg.process_create_time(os.getpid())
    if real is None:
        pytest.skip("platform has no create-time token (e.g. macOS)")
    (reg.sessions_dir() / "ghost.json").write_text(
        json.dumps({"session_id": "ghost", "pid": os.getpid(), "create_time": real + 1})
    )
    assert reg.resolve("ghost") is None
    assert not (reg.sessions_dir() / "ghost.json").exists()


def _dead_pid():
    """A pid that is (almost certainly) not alive: spawn a child, reap it, reuse
    its pid before the OS recycles it."""
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid
