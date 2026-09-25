"""Uninstall removes what install wrote, and nothing of the user's."""

from __future__ import annotations

import json

from conftest import bash, ps_literal, pwsh, requires_posix, requires_pwsh


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
