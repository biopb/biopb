"""bootstrap.ps1: pick a release, fetch its installer, run it.

The network is two stub functions shadowing Invoke-RestMethod (the release
listing) and Invoke-WebRequest (the installer download, which logs its URL).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from conftest import PWSH, _pwsh_base_env, requires_pwsh, sh

BOOTSTRAP = Path(__file__).resolve().parent.parent / "bootstrap.ps1"

PRELUDE = r"""
$script:urls = @()
function Invoke-RestMethod {
    param($Uri)
    $script:urls += $Uri
    @(
        [pscustomobject]@{ tag_name = 'release-v0.16.0rc1' },
        [pscustomobject]@{ tag_name = 'v0.11.1' },
        [pscustomobject]@{ tag_name = 'release-v0.15.2' },
        [pscustomobject]@{ tag_name = 'release-v0.15.1' }
    )
}
function Invoke-WebRequest {
    param($Uri, [switch]$UseBasicParsing)
    $script:urls += $Uri
    if ($env:STUB_INSTALLER -eq 'missing') { throw '404' }
    $body = if ($env:STUB_INSTALLER -eq 'broken') { 'if (' }
            else { 'Write-Output "ran mark=[$env:BIOPB_MARK]"' }
    [pscustomobject]@{ Content = $body }
}
"""


def run(body: str, env: dict[str, str] | None = None):
    assert PWSH is not None
    script = f'$env:BIOPB_INSTALL_LIB = "1"\n. {sh(BOOTSTRAP)}\n{PRELUDE}\n{body}'
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        env={**_pwsh_base_env(), **(env or {})},
        timeout=120,
    )


def tag(env=None) -> str:
    proc = run("Resolve-BiopbTag", env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@requires_pwsh
def test_latest_stable_skips_candidates_and_the_sdk_line():
    assert tag() == "release-v0.15.2"


@requires_pwsh
def test_rc_channel_takes_the_candidate():
    assert tag({"BIOPB_INSTALL_RC": "1"}) == "release-v0.16.0rc1"


@requires_pwsh
def test_rc_zero_is_the_stable_channel():
    assert tag({"BIOPB_INSTALL_RC": "0"}) == "release-v0.15.2"


@requires_pwsh
def test_an_exact_version_in_any_spelling():
    for given in ("0.15.1", "v0.15.1", "release-v0.15.1"):
        assert tag({"BIOPB_INSTALL_VERSION": given}) == "release-v0.15.1"


@requires_pwsh
def test_runs_the_release_installer_with_the_environment():
    proc = run(
        "Invoke-BiopbBootstrap; $script:urls -join ' '",
        {"BIOPB_INSTALL_VERSION": "0.15.1", "BIOPB_MARK": "x"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "ran mark=[x]" in proc.stdout
    assert "releases/download/release-v0.15.1/install.ps1" in proc.stdout


@requires_pwsh
def test_a_release_without_an_installer_fails_clearly():
    proc = run("Invoke-BiopbBootstrap", {"STUB_INSTALLER": "missing"})
    assert proc.returncode != 0
    assert "Could not download" in proc.stderr + proc.stdout


@requires_pwsh
def test_a_truncated_installer_is_not_run():
    proc = run("Invoke-BiopbBootstrap", {"STUB_INSTALLER": "broken"})
    assert proc.returncode != 0
    assert "not a valid script" in proc.stderr + proc.stdout


@requires_pwsh
def test_a_tag_that_is_not_a_plain_name_is_refused():
    proc = run("Invoke-BiopbBootstrap", {"BIOPB_INSTALL_VERSION": "0.1/../x?y"})
    assert proc.returncode != 0
    assert "Unexpected release tag" in proc.stderr + proc.stdout
