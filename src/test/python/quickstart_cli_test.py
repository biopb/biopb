"""Unit tests for `biopb quick-start` -- the opt-in Windows Defender exclusion
(issue #384).

OS calls (PowerShell, elevation) are mocked so the tests are deterministic on any
platform; the platform gate is driven by patching `cli._is_windows` rather than
the global `os.name` (which pathlib reads to choose WindowsPath/PosixPath, so
mutating it would break every `Path(...)` in the process on POSIX).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import biopb.cli as cli
import pytest
from typer.testing import CliRunner

runner = CliRunner()


def _win(monkeypatch):
    """Pretend we're on Windows for the command body's platform gate."""
    monkeypatch.setattr(cli, "_is_windows", lambda: True)


class TestQuickStartDispatch:
    def test_enable_adds_exclusion(self, monkeypatch):
        _win(monkeypatch)
        with patch.object(cli, "_defender_exclusion") as excl:
            res = runner.invoke(cli.app, ["quick-start", "--enable"])
        assert res.exit_code == 0
        excl.assert_called_once()
        assert excl.call_args.kwargs == {"add": True}

    def test_disable_removes_exclusion(self, monkeypatch):
        _win(monkeypatch)
        with patch.object(cli, "_defender_exclusion") as excl:
            res = runner.invoke(cli.app, ["quick-start", "--disable"])
        assert res.exit_code == 0
        excl.assert_called_once()
        assert excl.call_args.kwargs == {"add": False}

    def test_no_flag_shows_status(self, monkeypatch):
        _win(monkeypatch)
        with (
            patch.object(cli, "_defender_status") as status,
            patch.object(cli, "_defender_exclusion") as excl,
        ):
            res = runner.invoke(cli.app, ["quick-start"])
        assert res.exit_code == 0
        status.assert_called_once()
        excl.assert_not_called()

    def test_non_windows_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(cli, "_is_windows", lambda: False)
        with patch.object(cli, "_defender_exclusion") as excl:
            res = runner.invoke(cli.app, ["quick-start", "--enable"])
        assert res.exit_code == 0
        assert "Windows-only" in res.output
        excl.assert_not_called()

    def test_hidden_matches_platform(self):
        """Registered everywhere, but hidden from help off Windows."""
        qs = next(c for c in cli.app.registered_commands if c.name == "quick-start")
        # hidden is fixed at import from the real platform: shown on Windows,
        # hidden elsewhere. Assert the relationship, not a hardcoded value.
        assert qs.hidden == (not cli._is_windows())


class TestDefenderExclusion:
    _VENV = Path(r"C:\Users\lab\.local\share\uv\tools\biopb")

    def test_add_snippet_is_wellformed(self):
        captured = {}
        with patch.object(
            cli, "_run_elevated_ps", side_effect=lambda s: captured.update(s=s) or 0
        ):
            cli._defender_exclusion(self._VENV, add=True)
        snip = captured["s"]
        assert "Add-MpPreference -ExclusionPath $p" in snip
        assert "$p = 'C:\\Users\\lab\\.local\\share\\uv\\tools\\biopb'" in snip
        # Literal PowerShell braces survive (no f-string brace-escape leakage).
        assert "if ($present) { exit 0 } else { exit 3 }" in snip
        assert "{{" not in snip and "}}" not in snip

    def test_remove_snippet_inverts_the_check(self):
        captured = {}
        with patch.object(
            cli, "_run_elevated_ps", side_effect=lambda s: captured.update(s=s) or 0
        ):
            cli._defender_exclusion(self._VENV, add=False)
        snip = captured["s"]
        assert "Remove-MpPreference -ExclusionPath $p" in snip
        assert "if (-not $present) { exit 0 } else { exit 3 }" in snip

    @pytest.mark.parametrize(
        "rc,exit_code,needle",
        [
            (0, 0, "added"),
            # add=True below -> the cmdlet named in the message is Add-MpPreference.
            (2, 1, "Add-MpPreference call failed"),
            (3, 1, "blocked by Tamper Protection"),
            (1, 1, "elevation was declined"),
        ],
    )
    def test_result_code_mapping(self, rc, exit_code, needle, capsys):
        with patch.object(cli, "_run_elevated_ps", return_value=rc):
            if exit_code == 0:
                cli._defender_exclusion(self._VENV, add=True)  # no raise
            else:
                with pytest.raises(cli.typer.Exit) as ei:
                    cli._defender_exclusion(self._VENV, add=True)
                assert ei.value.exit_code == exit_code
        # Also assert the user-facing message, not just the exit code. Collapse
        # Rich's soft-wrapping so a needle isn't split across wrapped lines.
        out = " ".join(capsys.readouterr().out.split())
        assert needle in out

    def test_single_quote_in_path_is_escaped(self):
        captured = {}
        weird = Path(r"C:\Users\o'brien\biopb")
        with patch.object(
            cli, "_run_elevated_ps", side_effect=lambda s: captured.update(s=s) or 0
        ):
            cli._defender_exclusion(weird, add=True)
        # ' -> '' so the PowerShell single-quoted string stays intact.
        assert "$p = 'C:\\Users\\o''brien\\biopb'" in captured["s"]


class TestRunElevatedPs:
    def test_builds_runas_launcher_and_cleans_temp(self):
        with (
            patch.object(cli.subprocess, "run") as run,
            patch.object(cli.os, "unlink") as unlink,
        ):
            run.return_value = MagicMock(returncode=0)
            rc = cli._run_elevated_ps("Write-Host hi")
        assert rc == 0
        argv = run.call_args.args[0]
        assert argv[0] == "powershell"
        launcher = argv[-1]
        assert "Start-Process powershell -Verb RunAs" in launcher
        assert "-Wait -PassThru" in launcher
        assert "'-File'," in launcher
        unlink.assert_called_once()  # temp .ps1 removed

    def test_returns_child_exit_code(self):
        with (
            patch.object(cli.subprocess, "run") as run,
            patch.object(cli.os, "unlink"),
        ):
            run.return_value = MagicMock(returncode=3)
            assert cli._run_elevated_ps("x") == 3


class TestDefenderStatus:
    _VENV = Path(r"C:\Users\lab\biopb")

    @pytest.mark.parametrize(
        "stdout,needle",
        [
            ("ON", "is enabled"),
            ("OFF", "not set"),
            ("UNKNOWN", "Could not read"),
        ],
    )
    def test_status_messages(self, stdout, needle, capsys):
        with patch.object(cli.subprocess, "run") as run:
            run.return_value = MagicMock(stdout=stdout + "\n")
            cli._defender_status(self._VENV)
        out = capsys.readouterr().out
        assert needle in out

    def test_status_swallows_subprocess_error(self, capsys):
        with patch.object(cli.subprocess, "run", side_effect=OSError("boom")):
            cli._defender_status(self._VENV)  # no raise
        assert "Could not read" in capsys.readouterr().out
