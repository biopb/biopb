"""Unit tests for `biopb uninstall` -- the hand-off to the saved uninstaller.

The command only locates the script the install saved, confirms, and passes
control to it; the teardown itself lives in install.sh / the Windows engine. The
exec and subprocess calls are mocked so nothing is run or removed.
"""

from unittest.mock import patch

import biopb.cli as cli
import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """A data tree holding a saved POSIX uninstaller."""
    monkeypatch.setenv("BIOPB_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_is_windows", lambda: False)
    script = cli._saved_uninstaller()
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n")
    return script


def test_execs_saved_uninstaller(saved):
    with patch.object(cli.os, "execv") as execv:
        res = runner.invoke(cli.app, ["uninstall", "--yes"])
    assert res.exit_code == 0
    argv = execv.call_args.args[1]
    assert argv[1:] == [str(saved)]


def test_purge_is_forwarded(saved):
    with patch.object(cli.os, "execv") as execv:
        runner.invoke(cli.app, ["uninstall", "--purge", "-y"])
    assert execv.call_args.args[1][1:] == [str(saved), "--purge"]


def test_declining_the_prompt_runs_nothing(saved):
    with patch.object(cli.os, "execv") as execv:
        res = runner.invoke(cli.app, ["uninstall"], input="n\n")
    assert res.exit_code == 1
    execv.assert_not_called()


def test_missing_uninstaller_points_at_the_release_installer(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOPB_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_is_windows", lambda: False)
    with patch.object(cli.os, "execv") as execv:
        res = runner.invoke(cli.app, ["uninstall", "--yes"])
    assert res.exit_code == 1
    assert "--uninstall" in res.output
    execv.assert_not_called()


@pytest.mark.parametrize("purge,flag", [(False, "-KeepData"), (True, "-Purge")])
def test_windows_answers_the_data_question(tmp_path, monkeypatch, purge, flag):
    monkeypatch.setenv("BIOPB_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_is_windows", lambda: True)
    script = cli._saved_uninstaller()
    script.parent.mkdir(parents=True)
    script.write_text("@echo off\r\n")
    with patch.object(cli.subprocess, "Popen") as popen:
        res = runner.invoke(
            cli.app, ["uninstall", "--yes", *(["--purge"] if purge else [])]
        )
    assert res.exit_code == 0
    assert popen.call_args.args[0] == [str(script), flag]
