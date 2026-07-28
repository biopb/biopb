"""Unit tests for the local data-plane credential handoff (:mod:`biopb._credentials`).

Pins the contract the control (writer) and biopb-mcp (reader) share: where the
credential lives, that a write is atomic and owner-restricted, and that the
per-platform ``0600`` equivalent actually applies — POSIX mode bits, and on
Windows a DACL granting only the current user with inheritance disabled
(biopb/biopb#470).
"""

from __future__ import annotations

import pathlib
import stat
import subprocess
import sys

import pytest
from biopb import _credentials as C


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect the state tree to a temp dir (via ``Path.home``, matching
    ``config_paths_test``) so a test never touches the machine's real credential."""
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    for var in ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)


class TestLocation:
    def test_lives_in_state_tree(self, tmp_path):
        assert (
            C.credential_file()
            == tmp_path / ".local" / "state" / "biopb" / "tensor-server.token"
        )

    def test_xdg_state_home_honored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xs"))
        assert C.credential_file() == tmp_path / "xs" / "biopb" / "tensor-server.token"


class TestRoundtrip:
    def test_write_then_read(self):
        C.write_credential("a-secret-token-value")
        assert C.read_credential() == "a-secret-token-value"

    def test_read_absent_is_none(self):
        assert C.read_credential() is None

    def test_read_empty_is_none(self):
        # A control that somehow wrote an empty token reads back as "no token",
        # not as the empty string (which would be sent as a bogus credential).
        path = C.credential_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("   \n", encoding="utf-8")
        assert C.read_credential() is None

    def test_overwrite(self):
        C.write_credential("first")
        C.write_credential("second")
        assert C.read_credential() == "second"

    def test_remove(self):
        C.write_credential("gone-soon")
        C.remove_credential()
        assert not C.credential_file().exists()
        assert C.read_credential() is None

    def test_remove_absent_is_noop(self):
        C.remove_credential()  # must not raise when there is nothing to remove

    def test_write_leaves_no_temp_file(self):
        C.write_credential("tok")
        leftovers = list(C.credential_file().parent.glob(".tensor-server.token-*"))
        assert leftovers == []


class TestHardening:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    def test_posix_mode_is_0600(self):
        path = C.write_credential("tok")
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows DACL")
    def test_windows_dacl_is_owner_only(self):
        path = C.write_credential("tok")
        out = subprocess.run(
            ["icacls", str(path)], capture_output=True, text=True, check=True
        ).stdout
        me = subprocess.run(
            ["whoami"], capture_output=True, text=True, check=True
        ).stdout.strip()
        # Drop the printed file path first: it is itself under C:\Users\..., which
        # would false-match the "no BUILTIN\Users ACE" check below.
        aces = out.replace(str(path), "").lower()
        # The current user is granted access...
        assert me.lower() in aces
        # ...and no ACE is inherited (a protected DACL): inherited ACEs render with
        # an "(I)" flag, so their absence proves inheritance was broken — the
        # faithful analogue of 0600 rather than a file that merely rode the profile
        # dir's inherited ACL.
        assert "(i)" not in aces
        # No broad principals (Users / Everyone / Authenticated Users) survive.
        assert "everyone" not in aces
        assert "\\users" not in aces
        assert "authenticated users" not in aces
