"""Hold both system-Python probes to one table of interpreter shapes.

Step 2 of the install asks whatever `python` is on PATH for its version, and what
this module exists to pin down is that the ASKING -- not the answer, and not the
Python install that follows -- is what fails on a fresh machine. It failed two
ways, both of which reached users as "the installer fails at the Python stage":

  * A Microsoft Store *App Execution Alias*. A Windows box with no Python still
    has a 0-byte python.exe on PATH that prints to stderr and exits 9009. Under
    the engine's EAP='Stop', Windows PowerShell 5.1 turns a redirected native
    command's stderr line into a TERMINATING error, so the install died quoting
    "Python was not found..." instead of falling through to a uv-managed Python.
  * An interpreter that greets on stdout (conda, a sitecustomize banner), whose
    banner line was cast with [int] -- another terminating error, same place.

Neither reproduces under pwsh 7, which is the only PowerShell CI ran until the
Windows leg in install-scripts.yaml.

The rule is implemented twice -- biopb-engine.ps1::Get-SystemPythonVersion and
install.sh::_system_python_version -- so, like the extras parsers before it
(#648, #653), it gets ONE table and both halves are held to it. The bash half
never had the terminating-stderr problem, but it had the stdout-banner one, and
two implementations of one rule with only one under test is how they drift.

Cases live in python-probe-contract.json, never inline here. They are written
against stub interpreters rather than real ones because the thing under test is
the SHAPE of what an interpreter emits -- stream, exit code, extra lines -- and a
stub is the only way to hold a test to shapes the CI runner does not happen to
have installed.

The bar every case shares: the probe must ANSWER -- a version, or nothing -- and
never throw. A throw here is an aborted install.
"""

from __future__ import annotations

import json
import os
import stat

import pytest
from conftest import (
    TEST_DIR,
    bash,
    ps_literal,
    pwsh,
    requires_posix,
    requires_pwsh,
    sh,
)

IS_WINDOWS = os.name == "nt"

CONTRACT = json.loads(
    (TEST_DIR / "python-probe-contract.json").read_text(encoding="utf-8")
)
CONTRACT_CASES = CONTRACT["cases"]
CASE_IDS = [c["id"] for c in CONTRACT_CASES]

# What a probe returns for "I cannot read this interpreter". Both halves signal it
# natively as null/empty; the drivers below normalise to this so one table can
# describe both.
NONE = "NONE"


def _cmd_escape(text: str) -> str:
    """Escape a line for `echo` in a .cmd file.

    cmd treats & < > | ( ) as syntax even inside echo, so an unescaped one either
    truncates the line or makes the stub fail to parse -- which would look like a
    probe failure and pass a case for the wrong reason. The Store message in the
    contract contains a `>`, which is exactly why this exists.
    """
    for ch in "^&<>|()":
        text = text.replace(ch, "^" + ch)
    return text


def write_stub(tmp_path, name, *, stdout=(), stderr=(), exit_code=0):
    """Write a fake interpreter emitting fixed lines, then exiting `exit_code`.

    Two dialects because the probes run on two platforms: a .cmd batch file for
    Windows PowerShell, a /bin/sh script everywhere else. Both are invoked the way
    the installers invoke a real interpreter -- with `-c <program>` -- which the
    stub accepts and ignores.
    """
    if IS_WINDOWS:
        path = tmp_path / f"{name}.cmd"
        lines = ["@echo off"]
        # `echo.` for an empty line: a bare `echo` with nothing after it prints
        # "ECHO is on." instead, which would make the blank-line case assert
        # against a stub that never produced a blank line.
        lines += [
            (f"echo {_cmd_escape(v)} 1>&2" if v else "echo. 1>&2") for v in stderr
        ]
        lines += [(f"echo {_cmd_escape(v)}" if v else "echo.") for v in stdout]
        lines.append(f"exit /b {exit_code}")
        # write_bytes, not write_text: text mode on Windows would translate these
        # CRLFs into CRCRLF, and the stub would stop being a valid batch file.
        # cmd needs the CRLFs -- it mis-parses a .cmd with bare LF endings.
        path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    else:
        path = tmp_path / name
        lines = ["#!/bin/sh"]
        # printf rather than echo: echo mangles a leading -n or a backslash in
        # some shells, and the stub has to emit what it was handed.
        #
        # The format is a RAW string so the two characters `\` and `n` reach the
        # generated script. Written as a plain "\n" it is a real newline, which
        # lands INSIDE the quoted format -- still valid sh, and still prints the
        # right thing, so nothing fails and the next reader inherits a puzzle.
        fmt = r"printf '%s\n' "
        lines += [fmt + f'"{v}" >&2' for v in stderr]
        lines += [fmt + f'"{v}"' for v in stdout]
        lines.append(f"exit {exit_code}")
        path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _probe_pwsh(exe) -> str:
    """biopb-engine.ps1's Get-SystemPythonVersion, as "MAJOR.MINOR" or NONE.

    Runs through conftest.pwsh, which dot-sources the engine -- so this executes
    under the engine's own EAP='Stop', the setting both bugs needed. check=True
    means a terminating error inside the probe fails the case as a nonzero exit,
    which is exactly the production symptom.
    """
    out = pwsh(
        f"$v = Get-SystemPythonVersion -PythonExe {ps_literal(exe)}\n"
        f'if ($null -eq $v) {{ "{NONE}" }} else {{ "$($v.Major).$($v.Minor)" }}\n'
    )
    return out.stdout.strip()


def _probe_bash(exe) -> str:
    """install.sh's _system_python_version, in the same shape.

    It answers "MAJOR MINOR" or nothing at all; the reshaping to MAJOR.MINOR here
    is only so one table can describe both halves. Empty output is the bash way of
    saying $null, and `set -e` is in force (conftest.bash), so a helper that exits
    nonzero on an unreadable interpreter fails the case rather than returning.
    """
    out = bash(
        f"v=$(_system_python_version {sh(exe)})\n"
        f'if [ -z "$v" ]; then echo {NONE}; else echo "${{v%% *}}.${{v##* }}"; fi\n'
    )
    return out.stdout.strip()


PROBE_PARAMS = [
    # install.sh is the POSIX installer and is never run on Windows, so its half
    # of the contract is held on the Linux leg -- same call as test_install_sh.py.
    pytest.param(_probe_bash, id="install.sh", marks=requires_posix),
    pytest.param(_probe_pwsh, id="biopb-engine.ps1", marks=requires_pwsh),
]


@pytest.mark.parametrize("probe", PROBE_PARAMS)
@pytest.mark.parametrize("case", CONTRACT_CASES, ids=CASE_IDS)
def test_contract(probe, case, tmp_path):
    stub = write_stub(
        tmp_path,
        "python",
        stdout=case["stdout"],
        stderr=case["stderr"],
        exit_code=case["exit_code"],
    )
    expected = case["expected"] if case["expected"] is not None else NONE
    assert probe(stub) == expected, case["why"]


@pytest.mark.parametrize("probe", PROBE_PARAMS)
def test_missing_interpreter_is_not_a_version(probe, tmp_path):
    """A path that is not there at all: still an answer, still not a throw.

    Not a contract case because there is no stub to describe -- the interpreter is
    the one thing the table cannot express by listing what it printed.
    """
    assert probe(tmp_path / "does-not-exist") == NONE
