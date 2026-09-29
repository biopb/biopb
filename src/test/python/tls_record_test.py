"""The record a local plane publishes about what it serves (biopb/biopb#916).

Its whole job is to survive being wrong: every failure to read one means "fall
back to the minted certificate", never "fail", because the alternative is a
client that cannot dial a plane over a missing advisory file.
"""

import json
import os

import pytest
from biopb import _tls_record
from biopb._locations import tls_served_certs


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))


def test_publish_then_lookup_roundtrips():
    _tls_record.publish(8815, "abc123")
    assert _tls_record.lookup(8815) == "abc123"


def test_entries_are_per_port():
    """Two planes can share a state tree: the uid-scoped cache lock that would
    otherwise refuse the second is optional, and absent for the memory backend."""
    _tls_record.publish(8815, "aaa")
    _tls_record.publish(9815, "bbb")
    assert (_tls_record.lookup(8815), _tls_record.lookup(9815)) == ("aaa", "bbb")


def test_republishing_replaces_the_entry():
    """A plane that re-mints its cert must not leave the old claim standing."""
    _tls_record.publish(8815, "old")
    _tls_record.publish(8815, "new")
    assert _tls_record.lookup(8815) == "new"


def test_lookup_of_an_absent_record_is_none_not_an_error():
    assert _tls_record.lookup(8815) is None
    _tls_record.publish(9999, "x")
    assert _tls_record.lookup(8815) is None


def test_a_corrupt_record_reads_as_absent():
    """Falling back is right; raising would break a dial over an advisory file."""
    path = tls_served_certs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert _tls_record.lookup(8815) is None

    path.write_text('{"8815": "a bare string"}', encoding="utf-8")
    assert _tls_record.lookup(8815) is None

    path.write_text('{"8815": {"pid": 7}}', encoding="utf-8")
    assert _tls_record.lookup(8815) is None


def test_retract_removes_only_its_own_port():
    _tls_record.publish(8815, "aaa")
    _tls_record.publish(9815, "bbb")
    _tls_record.retract(8815)
    assert _tls_record.lookup(8815) is None
    assert _tls_record.lookup(9815) == "bbb"


def test_retracting_nothing_is_harmless():
    _tls_record.retract(8815)  # no file at all
    _tls_record.publish(9815, "bbb")
    _tls_record.retract(8815)  # file, no entry
    assert _tls_record.lookup(9815) == "bbb"


def test_the_entry_records_who_wrote_it():
    """A stale entry shows up as a refused connection; the pid and timestamp are
    what let an operator work out which plane left it."""
    import os

    _tls_record.publish(8815, "abc")
    entry = json.loads(tls_served_certs().read_text())["8815"]
    assert entry["pid"] == os.getpid()
    assert entry["updated_at"] > 0


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="a read-only directory stops neither root nor Windows",
)
def test_publish_never_raises_when_the_state_tree_is_unwritable(tmp_path, monkeypatch):
    """Advisory: a plane that cannot publish still serves, and its clients fall
    back to the minted cert. Failing startup over a hint would trade a working
    deployment for a broken one."""
    readonly = tmp_path / "readonly"
    readonly.mkdir(mode=0o500)
    monkeypatch.setenv("BIOPB_STATE_HOME", str(readonly / "state"))
    try:
        _tls_record.publish(8815, "abc")  # must not raise
        assert _tls_record.lookup(8815) is None  # and nothing was recorded
    finally:
        readonly.chmod(0o700)
