"""Hold install.sh's system-Python probe to a table of interpreter shapes.

Step 2 of the POSIX install asks whatever `python3` is on PATH for its version,
and what this module exists to pin down is that the ASKING -- not the answer, and
not the Python install that follows -- is what fails on a real machine. An
interpreter that greets on stdout (conda, a sitecustomize banner) puts a
non-numeric line ahead of the answer, and handing that word to `[ "$MAJOR" -eq 3 ]`
is an "integer expression expected" error rather than a fallback: the machine is
told its Python is "too old" and the real version is never looked at.

The Windows half of this contract is gone with the probe it held: biopb-engine.ps1
no longer looks at a system interpreter at all (see its step 2). The shapes below
that only Windows can present -- the Microsoft Store alias stub above all -- stay
anyway, because they describe what the bash probe must make of an interpreter that
declines to answer, and Git Bash reaches them.

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
from conftest import TEST_DIR, bash, requires_posix, sh

IS_WINDOWS = os.name == "nt"

CONTRACT = json.loads(
    (TEST_DIR / "python-probe-contract.json").read_text(encoding="utf-8")
)
CONTRACT_CASES = CONTRACT["cases"]
CASE_IDS = [c["id"] for c in CONTRACT_CASES]

# What the probe returns for "I cannot read this interpreter". It signals that
# natively as empty output; the driver below normalises to this so the table can
# state it.
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

    Two dialects, kept from when the PowerShell engine had a probe of its own: a
    .cmd batch file on Windows, a /bin/sh script everywhere else. Both are invoked
    the way the installer invokes a real interpreter -- with `-c <program>` --
    which the stub accepts and ignores.
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


def _probe_bash(exe) -> str:
    """install.sh's _system_python_version, as "MAJOR.MINOR" or NONE.

    It answers "MAJOR MINOR" or nothing at all; the reshaping here is only so the
    table can state one expectation per case. Empty output is the bash way of
    saying "I cannot read this", and `set -e` is in force (conftest.bash), so a
    helper that exits nonzero on an unreadable interpreter fails the case rather
    than returning.
    """
    out = bash(
        f"v=$(_system_python_version {sh(exe)})\n"
        f'if [ -z "$v" ]; then echo {NONE}; else echo "${{v%% *}}.${{v##* }}"; fi\n'
    )
    return out.stdout.strip()


# install.sh is the POSIX installer and is never run on Windows, so the contract
# is held on the Linux leg -- same call as test_install_sh.py.
PROBE_PARAMS = [pytest.param(_probe_bash, id="install.sh", marks=requires_posix)]


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
