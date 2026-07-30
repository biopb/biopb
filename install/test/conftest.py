"""Plumbing for the installer unit tests: run a shell helper, get its answer back.

`install/` is 4300 lines of bash and PowerShell that no Python imports, so these
tests drive it the only way it can be driven -- by subprocess. pytest is the
runner because it is already the repo's, not because the code under test is
Python: a second pinned toolchain (bats, Pester) would buy nothing here and cost
a pin, a CI step and a second reporting format. See biopb/biopb#653.

The bash side rests on install.sh's BIOPB_INSTALL_LIB guard, which suppresses the
trailing `main "$@"` so sourcing the file defines its functions instead of
installing biopb over the test runner's home directory. The PowerShell side needs
no such guard -- biopb-engine.ps1 already distinguishes dot-sourced from run.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
INSTALL_DIR = TEST_DIR.parent
INSTALL_SH = INSTALL_DIR / "install.sh"
ENGINE_PS1 = INSTALL_DIR / "biopb-engine.ps1"

# The shared input->requirements table both extras parsers are held to (#653
# item 2). Loaded here so a malformed fixture fails collection loudly rather than
# silently parametrizing zero cases.
CONTRACT = json.loads((TEST_DIR / "extras-contract.json").read_text(encoding="utf-8"))
CONTRACT_CASES = CONTRACT["cases"]

# PowerShell 7 on Linux is `pwsh`; Windows PowerShell is `powershell`. Absent on a
# bare dev box, present on ubuntu-latest -- so the PowerShell half of the contract
# skips locally and runs in CI rather than being unrunnable in either.
PWSH = shutil.which("pwsh") or shutil.which("powershell")
requires_pwsh = pytest.mark.skipif(
    PWSH is None, reason="no PowerShell interpreter on PATH (pwsh / powershell)"
)

# The helpers under test call out to `tr`, `sed`, `grep`, `basename` and python3,
# so a test PATH cannot be empty -- but it also must not be the developer's, or a
# `claude` binary on the real machine would decide what _detect_agents returns.
# This is the middle: the system tools, none of the agents.
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# Resolved once against the REAL environment. subprocess looks the executable up
# in the PATH it is handed, and some tests hand it a PATH containing nothing but
# stub agents -- under which "bash" itself is not findable.
BASH = shutil.which("bash") or "/bin/bash"


def sh(value) -> str:
    """Quote `value` for the shell.

    Not repr(): Python escapes a backslash where the shell would not, so a regex
    like `biopb_mcp-.*\\.whl` arrives at grep with a literal backslash in it and
    silently matches nothing.
    """
    return shlex.quote(str(value))


def bash(
    body: str,
    *,
    env: dict[str, str] | None = None,
    path: str = SYSTEM_PATH,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Source install.sh, then run `body` with its functions in scope.

    `set -euo pipefail` is applied after sourcing rather than before, mirroring
    the real script: it sets those inside install_biopb, not at file scope, so a
    helper that is only ever reached from there is tested under the options it
    actually runs under.
    """
    script = f". {sh(INSTALL_SH)}\nset -euo pipefail\n{body}"
    full_env = {
        "BIOPB_INSTALL_LIB": "1",
        "PATH": path,
        # Every helper that reads $HOME gets a caller-supplied one; this default
        # is a deliberately nonexistent path so a test that forgets to set it
        # fails on the assertion rather than on the developer's real dotfiles.
        "HOME": "/nonexistent-biopb-test-home",
        **(env or {}),
    }
    return subprocess.run(
        [BASH, "-c", script],
        capture_output=True,
        text=True,
        check=check,
        env=full_env,
        timeout=60,
    )


def pwsh(
    body: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Dot-source biopb-engine.ps1, then run `body` with its functions in scope."""
    assert PWSH is not None, "guard with @requires_pwsh"
    script = f". {sh(ENGINE_PS1)}\n{body}"
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ.get("PATH", SYSTEM_PATH), **(env or {})},
        timeout=120,
    )


def write_extras(config_dir: Path, content: str) -> Path:
    """Drop an extra-packages.txt with EXACTLY `content` in it.

    Written as bytes, not text: the contract's CRLF and no-trailing-newline cases
    are about the literal bytes on disk, and Python's text mode would be free to
    translate both away before either parser ever saw them.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / "extra-packages.txt"
    target.write_bytes(content.encode("utf-8"))
    return target


@pytest.fixture
def stub_bin(tmp_path):
    """Make named commands resolvable on PATH, and nothing else.

    Returns (make, path): call make("claude") to plant an executable stub, then
    pass path= to bash(). The helpers this serves -- _detect_agents,
    _agent_launch_cmd -- probe with `command -v`, a bash builtin, so the stub dir
    can be the entire PATH. That is the point: no system binary can leak in and
    make the answer depend on the machine.
    """
    stubs = tmp_path / "stub-bin"
    stubs.mkdir()

    def make(name: str, body: str = "") -> Path:
        exe = stubs / name
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)
        return exe

    return make, str(stubs)
