"""Uninstall removes what install wrote, and nothing of the user's."""

from __future__ import annotations

import json
import subprocess

from conftest import (
    SYSTEM_PATH,
    bash,
    ps_literal,
    pwsh,
    requires_posix,
    requires_pwsh,
    sh,
)


@requires_posix
def test_remove_desktop_shortcut_removes_every_launcher(tmp_path):
    launchers = [
        tmp_path / "Desktop" / "biopb Dashboard.command",
        tmp_path / "Desktop" / "biopb-dashboard.desktop",
        tmp_path / ".local" / "share" / "applications" / "biopb-dashboard.desktop",
    ]
    for f in launchers:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("biopb dashboard\n")
    other = tmp_path / "Desktop" / "notes.txt"
    other.write_text("mine")
    bash("_remove_desktop_shortcut", env={"HOME": str(tmp_path)})
    assert not any(f.exists() for f in launchers)
    assert other.read_text() == "mine"


@requires_pwsh
def test_remove_mcp_clients_edits_the_opencode_file_opencode_reads(tmp_path):
    """opencode prefers opencode.jsonc; registration writes it, so removal must."""
    oc = tmp_path / ".config" / "opencode"
    oc.mkdir(parents=True)
    jsonc = oc / "opencode.jsonc"
    jsonc.write_text(json.dumps({"mcp": {"biopb": {}, "other": {}}}))
    plain = oc / "opencode.json"
    plain.write_text(json.dumps({"mcp": {"biopb": {}}}))
    pwsh(
        f"Remove-McpClients -BiopbHome {ps_literal(tmp_path)}",
        env={"APPDATA": str(tmp_path / "appdata")},
    )
    assert json.loads(jsonc.read_text())["mcp"] == {"other": {}}
    assert "biopb" in json.loads(plain.read_text())["mcp"], (
        "the unread file is not touched"
    )


def _run_uninstall_ps1(tmp_path, *args):
    """Run uninstall.ps1 over a fake install in tmp_path; returns (webapp, samples, config)."""
    from conftest import INSTALL_DIR, PWSH, _pwsh_base_env

    share = tmp_path / ".local" / "share" / "biopb"
    (share / "webapp").mkdir(parents=True)
    (share / "samples").mkdir()
    (share / "uninstall").mkdir()
    config = tmp_path / ".config" / "biopb" / "biopb.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    env = {**_pwsh_base_env(), "USERPROFILE": str(tmp_path), "HOME": str(tmp_path)}
    # No uv/claude/codex/biopb reachable, and every tool-dir root the engine
    # force-stops processes under is inside tmp_path.
    env.pop("UV_TOOL_DIR", None)
    env["LOCALAPPDATA"] = env["APPDATA"] = str(tmp_path / "appdata")
    env["TEMP"] = str(tmp_path / "temp")
    env["PATH"] = str(tmp_path / "empty-bin")
    script = str(INSTALL_DIR / "uninstall.ps1")
    result = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-File", script, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return share / "webapp", share / "samples", config


@requires_pwsh
def test_uninstall_ps1_removes_the_install_and_keeps_data(tmp_path):
    """Unattended with no answer given: keep data, never prompt."""
    webapp, samples, config = _run_uninstall_ps1(tmp_path)
    assert not webapp.exists()
    assert samples.exists() and config.exists()
    assert not (tmp_path / ".local" / "share" / "biopb" / "uninstall").exists()


@requires_pwsh
def test_uninstall_ps1_purge_removes_data(tmp_path):
    webapp, samples, config = _run_uninstall_ps1(tmp_path, "-Purge")
    assert not webapp.exists() and not samples.exists() and not config.exists()


# --- the saved uninstaller ---------------------------------------------------

RELEASE_JSON = json.dumps(
    {
        "assets": [
            {
                "browser_download_url": "https://example.test/dl/release-v9.9.9/install.sh"
            },
        ]
    }
)


@requires_posix
def test_save_uninstaller_keeps_the_release_installer(tmp_path, stub_bin):
    make, stubs = stub_bin
    # curl -fsSL <url> -o <dest>: the "downloaded" installer echoes its arguments.
    make("curl", 'printf \'#!/usr/bin/env bash\\necho "$@"\\n\' > "$4"')
    dest = tmp_path / "uninstall"
    bash(
        f"_save_uninstaller {sh(dest)}",
        env={"RELEASE_JSON": RELEASE_JSON},
        path=f"{stubs}:{SYSTEM_PATH}",
    )
    ran = subprocess.run(
        [str(dest / "uninstall.sh"), "--purge"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ran.stdout.strip() == "--uninstall --purge"


@requires_posix
def test_save_uninstaller_without_the_asset_is_a_note(tmp_path, stub_bin):
    make, stubs = stub_bin
    make("curl", "exit 22")
    dest = tmp_path / "uninstall"
    result = bash(
        f"_save_uninstaller {sh(dest)}",
        env={"RELEASE_JSON": RELEASE_JSON},
        path=f"{stubs}:{SYSTEM_PATH}",
    )
    assert "Could not save the uninstaller" in result.stdout
    assert not (dest / "uninstall.sh").exists()


@requires_pwsh
def test_save_uninstaller_ps1_without_the_asset_leaves_nothing(tmp_path):
    dest = tmp_path / "uninstall"
    result = pwsh(
        "Save-Uninstaller -Release ([pscustomobject]@{ tag_name = 'release-v9.9.9'; assets = @() })"
        f" -Dir {ps_literal(dest)}"
    )
    assert "Could not save the uninstaller" in result.stdout
    assert not dest.exists()
