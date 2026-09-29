"""The shim's launch of the biopb control (``start_control_detached``)."""

from unittest.mock import MagicMock

import pytest


class TestStartControlDetached:
    def test_spawns_detached_no_data_plane_and_returns_true(self, monkeypatch):
        import os

        from biopb_mcp import _control_client

        monkeypatch.setattr(_control_client, "_biopb_executable", lambda: "/opt/biopb")
        captured = {}

        def _fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return MagicMock()

        monkeypatch.setattr(_control_client.subprocess, "Popen", _fake_popen)

        assert _control_client.start_control_detached() is True
        # Starts the control WITHOUT the data plane (on-demand) and never blocks.
        assert captured["argv"] == ["/opt/biopb", "control", "start", "--no-data-plane"]
        # stdio is fully detached from this process.
        assert captured["kwargs"]["stdin"] == _control_client.subprocess.DEVNULL
        assert captured["kwargs"]["stdout"] == _control_client.subprocess.DEVNULL
        assert captured["kwargs"]["stderr"] == _control_client.subprocess.DEVNULL
        # Detached from the shim's process group so it outlives a disconnect.
        if os.name == "nt":
            assert "creationflags" in captured["kwargs"]
        else:
            assert captured["kwargs"]["start_new_session"] is True

    def test_returns_false_and_does_not_spawn_when_cli_missing(self, monkeypatch):
        from biopb_mcp import _control_client

        monkeypatch.setattr(_control_client, "_biopb_executable", lambda: None)

        def _must_not_spawn(*_a, **_k):
            raise AssertionError("Popen must not be called when the CLI is absent")

        monkeypatch.setattr(_control_client.subprocess, "Popen", _must_not_spawn)
        assert _control_client.start_control_detached() is False

    def test_returns_false_on_spawn_oserror(self, monkeypatch):
        from biopb_mcp import _control_client

        monkeypatch.setattr(_control_client, "_biopb_executable", lambda: "/opt/biopb")

        def _boom(*_a, **_k):
            raise OSError("no exec")

        monkeypatch.setattr(_control_client.subprocess, "Popen", _boom)
        assert _control_client.start_control_detached() is False


class TestBiopbExecutable:
    def test_prefers_sibling_of_interpreter(self, tmp_path, monkeypatch):
        import os

        from biopb_mcp import _control_client

        bindir = tmp_path / "bin"
        bindir.mkdir()
        name = "biopb.exe" if os.name == "nt" else "biopb"
        sibling = bindir / name
        sibling.write_text("")
        monkeypatch.setattr(_control_client.sys, "executable", str(bindir / "python"))
        assert _control_client._biopb_executable() == str(sibling)

    def test_prefers_sibling_when_interpreter_is_symlinked(self, tmp_path, monkeypatch):
        # A real venv's `python` is a symlink to the base interpreter, while the
        # `biopb` console script lives in the venv bin/ next to that symlink. The
        # sibling lookup must resolve against the symlink's own dir, NOT the base
        # it points to -- otherwise it misses biopb in exactly the symlinked-venv +
        # no-PATH case this is for. (No PATH fallback here: which() returns None.)
        import os
        import shutil

        from biopb_mcp import _control_client

        base_bin = tmp_path / "base"
        base_bin.mkdir()
        (base_bin / "python3.10").write_text("")  # the real base interpreter
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        name = "biopb.exe" if os.name == "nt" else "biopb"
        sibling = venv_bin / name
        sibling.write_text("")
        try:
            os.symlink(base_bin / "python3.10", venv_bin / "python")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform/privilege")

        monkeypatch.setattr(_control_client.sys, "executable", str(venv_bin / "python"))
        monkeypatch.setattr(shutil, "which", lambda n: None)
        assert _control_client._biopb_executable() == str(sibling)

    def test_falls_back_to_path_when_no_sibling(self, tmp_path, monkeypatch):
        import shutil

        from biopb_mcp import _control_client

        # An interpreter dir with no biopb sibling -> resolution falls to PATH.
        monkeypatch.setattr(_control_client.sys, "executable", str(tmp_path / "python"))
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/biopb")
        assert _control_client._biopb_executable() == "/usr/bin/biopb"
