"""Hold the system-Python probe to the interpreters real machines actually have.

Step 2 of the Windows install asks whatever `python` is on PATH for its version,
and what this module exists to pin down is that the ASKING -- not the answer, and
not the Python install that follows -- is what fails on a fresh machine. Two ways,
both of which surfaced to users as "the installer fails at the Python stage":

  * A Microsoft Store *App Execution Alias*. A Windows box with no Python still
    has a 0-byte python.exe on PATH that prints to stderr and exits 9009. Under
    the engine's EAP='Stop', Windows PowerShell 5.1 turns a redirected native
    command's stderr line into a TERMINATING error, so the install died quoting
    "Python was not found..." instead of falling through to a uv-managed Python.
  * An interpreter that greets on stdout (conda, a sitecustomize banner), whose
    banner line was cast with [int] -- another terminating error, same place.

Neither reproduces under pwsh 7, which is the only PowerShell CI ran until the
Windows leg in install-scripts.yaml. So these cases are written against stub
interpreters rather than real ones: the point is the SHAPE of what an interpreter
emits (stream, exit code, extra lines), and a stub is the only way to hold a test
to shapes the CI runner does not happen to have installed.

The bar every case shares: the probe must RETURN something -- a version or $null
-- and never throw. A throw here is an aborted install.
"""

from __future__ import annotations

import os
import stat

import pytest
from conftest import ps_literal, pwsh, requires_pwsh

IS_WINDOWS = os.name == "nt"

# The real message, verbatim, because its `>` is the part that matters: it is
# what forced the cmd-escaping below, and a stub that quietly dropped it would
# stop resembling the machine this test exists for.
STORE_ALIAS_MESSAGE = (
    "Python was not found; run without arguments to install from the "
    "Microsoft Store, or disable this shortcut from Settings > Manage App "
    "Execution Aliases."
)


def _cmd_escape(text: str) -> str:
    """Escape a line for `echo` in a .cmd file.

    cmd treats & < > | ( ) as syntax even inside echo, so an unescaped one either
    truncates the line or makes the stub fail to parse -- which would look like a
    probe failure and pass this test for the wrong reason.
    """
    for ch in "^&<>|()":
        text = text.replace(ch, "^" + ch)
    return text


def write_stub(tmp_path, name, *, stdout=(), stderr=(), exit_code=0):
    """Write a fake interpreter emitting fixed lines, then exiting `exit_code`.

    Two dialects because the probe runs on two platforms: a .cmd batch file for
    Windows PowerShell, a /bin/sh script for pwsh on Linux. Both are invoked by
    the engine exactly as a real interpreter is (`& $PythonExe -c ...`), so the
    argument is accepted and ignored.
    """
    if IS_WINDOWS:
        path = tmp_path / f"{name}.cmd"
        lines = ["@echo off"]
        lines += [f"echo {_cmd_escape(line)} 1>&2" for line in stderr]
        lines += [f"echo {_cmd_escape(line)}" for line in stdout]
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
        lines += [fmt + f'"{line}" >&2' for line in stderr]
        lines += [fmt + f'"{line}"' for line in stdout]
        lines.append(f"exit {exit_code}")
        path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def probe(exe) -> str:
    """Get-SystemPythonVersion's answer as "MAJOR.MINOR", or "NONE" for $null.

    Runs through conftest.pwsh, which dot-sources the engine -- so this executes
    under the engine's own EAP='Stop', the setting both bugs needed. check=True
    means a terminating error inside the probe fails the test as a nonzero exit,
    which is exactly the production symptom.
    """
    out = pwsh(
        f"$v = Get-SystemPythonVersion -PythonExe {ps_literal(exe)}\n"
        'if ($null -eq $v) { "NONE" } else { "$($v.Major).$($v.Minor)" }\n'
    )
    return out.stdout.strip()


@requires_pwsh
def test_clean_interpreter_is_read(tmp_path):
    """The ordinary case: version on stdout, nothing on stderr, exit 0."""
    stub = write_stub(tmp_path, "clean", stdout=["3 11"])
    assert probe(stub) == "3.11"


@requires_pwsh
def test_store_alias_stub_is_not_a_python(tmp_path):
    """The fresh-Windows case: stderr message, no stdout, exit 9009.

    $null, not a throw -- the caller's whole fallback to a uv-managed Python
    hangs on this returning.
    """
    stub = write_stub(tmp_path, "store", stderr=[STORE_ALIAS_MESSAGE], exit_code=9009)
    assert probe(stub) == "NONE"


@requires_pwsh
def test_stderr_noise_does_not_reject_a_good_interpreter(tmp_path):
    """A usable Python that warns on stderr stays usable.

    The regression that mattered most: this interpreter is perfectly fine, and
    the installer refused to use it -- or rather, died on it -- over a DLL or
    deprecation line it printed on the way past.
    """
    stub = write_stub(
        tmp_path,
        "noisy",
        stdout=["3 11"],
        stderr=["[warn] some benign startup notice"],
    )
    assert probe(stub) == "3.11"


@requires_pwsh
def test_stdout_banner_does_not_shadow_the_version(tmp_path):
    """conda and friends greet on stdout before answering.

    The version is the LAST line, not the first; reading the first cast a banner
    with [int] and took the install down with it.
    """
    stub = write_stub(
        tmp_path, "banner", stdout=["conda env active - banner line", "3 11"]
    )
    assert probe(stub) == "3.11"


@requires_pwsh
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"stdout": ["not a version"]}, id="unparseable-output"),
        pytest.param({"stdout": [], "stderr": []}, id="says-nothing"),
        pytest.param(
            {"stdout": ["3 11"], "exit_code": 1}, id="right-answer-wrong-exit"
        ),
        pytest.param({"stdout": ["3"]}, id="major-only"),
        pytest.param({"stdout": ["3.11"]}, id="dotted-not-space-separated"),
    ],
)
def test_unusable_interpreters_are_none(tmp_path, kwargs):
    """Anything the probe cannot vouch for is $null, so the caller falls back.

    `right-answer-wrong-exit` is the one worth naming: a nonzero exit means the
    interpreter did not answer the question, whatever it happened to print.
    """
    stub = write_stub(tmp_path, "unusable", **kwargs)
    assert probe(stub) == "NONE"


@requires_pwsh
def test_missing_interpreter_is_none(tmp_path):
    """A path that is not there at all: still an answer, still not a throw."""
    assert probe(tmp_path / "does-not-exist") == "NONE"
