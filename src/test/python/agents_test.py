"""Unit tests for ``biopb._agents`` — registering biopb-mcp with agent clients.

Covers the three things the module does per client: a subprocess-free status read
(not_installed / installed / registered + drift), an atomic JSON merge/delete that
preserves the user's other config, and the Claude Code path that shells out to the
``claude`` CLI. Everything is exercised against a monkeypatched ``$HOME`` (and
``$APPDATA``), and the ``biopb-mcp`` command is pinned so entries and drift are
deterministic; the ``claude`` CLI is mocked (no real binary needed).
"""

import json
from pathlib import Path

import pytest
from biopb import _agents

_CMD = "/opt/biopb/bin/biopb-mcp"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate every home-relative config location and pin the mcp command."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Claude Desktop reads %APPDATA% on Windows; point it inside the tmp home so a
    # real Claude Desktop install on the test machine can't leak in.
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setattr(_agents, "_mcp_executable", lambda: _CMD)
    return tmp_path


def _no_binaries(monkeypatch):
    """Make every ``shutil.which`` probe miss (no claude/opencode on PATH)."""
    monkeypatch.setattr(_agents.shutil, "which", lambda name: None)


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def test_catalog_is_the_installer_set_minus_hermes():
    ids = [s.id for s in _agents.supported()]
    assert ids == ["claude-code", "claude-desktop", "cursor", "opencode"]
    assert "hermes" not in ids


def test_unknown_client_raises():
    with pytest.raises(_agents.AgentError):
        _agents.status("nope")
    with pytest.raises(_agents.AgentError):
        _agents.register("nope")


# --------------------------------------------------------------------------- #
# Status (subprocess-free)
# --------------------------------------------------------------------------- #


def test_cursor_state_transitions(home):
    # No ~/.cursor -> not installed.
    assert _agents.status("cursor")["state"] == "not_installed"
    # Dir exists, no biopb entry -> installed.
    (home / ".cursor").mkdir()
    s = _agents.status("cursor")
    assert s["state"] == "installed" and s["drifted"] is False
    # Entry present -> registered.
    _agents.register("cursor")
    s = _agents.status("cursor")
    assert s["state"] == "registered" and s["drifted"] is False


def test_registered_is_drift_when_command_differs(home):
    cfg = home / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "biopb": {
                        "command": "/old/biopb-mcp",
                        "args": ["--transport", "stdio"],
                    }
                }
            }
        )
    )
    s = _agents.status("cursor")
    assert s["state"] == "registered"
    assert s["drifted"] is True  # stored command != freshly resolved _CMD


def test_status_ignores_malformed_config(home):
    # A malformed config reads as "not registered" for display (installed here,
    # since the dir exists) rather than raising.
    cfg = home / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text("{ not json")
    assert _agents.status("cursor")["state"] == "installed"


def test_statuses_covers_all_clients_not_installed(home, monkeypatch):
    _no_binaries(monkeypatch)
    rows = _agents.statuses()
    assert [r["id"] for r in rows] == [
        "claude-code",
        "claude-desktop",
        "cursor",
        "opencode",
    ]
    assert all(r["state"] == "not_installed" for r in rows)


# --------------------------------------------------------------------------- #
# JSON clients: register / unregister
# --------------------------------------------------------------------------- #


def test_register_cursor_writes_stdio_entry(home):
    (home / ".cursor").mkdir()
    s = _agents.register("cursor")
    assert s["state"] == "registered"
    data = json.loads((home / ".cursor" / "mcp.json").read_text())
    assert data["mcpServers"]["biopb"] == {
        "command": _CMD,
        "args": ["--transport", "stdio"],
    }


def test_register_preserves_other_config(home):
    cfg = home / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "misc": 7}))
    _agents.register("cursor")
    data = json.loads(cfg.read_text())
    assert data["mcpServers"]["other"] == {"command": "x"}  # sibling untouched
    assert data["misc"] == 7  # unrelated top-level key untouched
    assert "biopb" in data["mcpServers"]


def test_register_leaves_no_temp_file(home):
    (home / ".cursor").mkdir()
    _agents.register("cursor")
    names = sorted(p.name for p in (home / ".cursor").iterdir())
    assert names == ["mcp.json"]  # atomic write cleaned up its temp


def test_register_refuses_to_clobber_unreadable_config(home):
    cfg = home / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text("{ not json")
    with pytest.raises(_agents.AgentError):
        _agents.register("cursor")
    assert cfg.read_text() == "{ not json"  # left exactly as-is


def test_unregister_removes_entry_keeps_siblings(home):
    cfg = home / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {"mcpServers": {"biopb": {"command": _CMD}, "other": {"command": "x"}}}
        )
    )
    s = _agents.unregister("cursor")
    assert s["state"] == "installed"
    data = json.loads(cfg.read_text())
    assert "biopb" not in data["mcpServers"]
    assert data["mcpServers"]["other"] == {"command": "x"}


def test_unregister_is_idempotent_when_absent(home):
    (home / ".cursor").mkdir()  # installed, but nothing registered
    s = _agents.unregister("cursor")  # must not raise
    assert s["state"] == "installed"


def test_opencode_uses_its_own_entry_shape(home):
    cfg = home / ".config" / "opencode" / "opencode.json"
    cfg.parent.mkdir(parents=True)
    _agents.register("opencode")
    data = json.loads(cfg.read_text())
    assert data["mcp"]["biopb"] == {
        "type": "local",
        "command": [_CMD, "--transport", "stdio"],
        "enabled": True,
    }
    assert _agents.status("opencode")["state"] == "registered"


def test_claude_desktop_config_paths(home, monkeypatch):
    spec = _agents._spec("claude-desktop")
    monkeypatch.setattr(_agents.sys, "platform", "linux")
    assert (
        _agents._config_path(spec)
        == home / ".config" / "Claude" / "claude_desktop_config.json"
    )
    monkeypatch.setattr(_agents.sys, "platform", "darwin")
    assert (
        _agents._config_path(spec)
        == home
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json"
    )
    monkeypatch.setattr(_agents.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(home / "Roaming"))
    assert (
        _agents._config_path(spec)
        == home / "Roaming" / "Claude" / "claude_desktop_config.json"
    )


# --------------------------------------------------------------------------- #
# Claude Code: managed through the `claude` CLI
# --------------------------------------------------------------------------- #


def _claude_on_path(monkeypatch):
    monkeypatch.setattr(
        _agents.shutil,
        "which",
        lambda name: "/usr/bin/claude" if name == "claude" else None,
    )


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_claude_code_status_reads_claude_json(home, monkeypatch):
    # Not on PATH, no entry -> not installed.
    _no_binaries(monkeypatch)
    assert _agents.status("claude-code")["state"] == "not_installed"
    # On PATH, no entry -> installed.
    _claude_on_path(monkeypatch)
    assert _agents.status("claude-code")["state"] == "installed"
    # Entry in ~/.claude.json -> registered.
    (home / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "biopb": {"command": _CMD, "args": ["--transport", "stdio"]}
                }
            }
        )
    )
    s = _agents.status("claude-code")
    assert s["state"] == "registered" and s["drifted"] is False


def test_register_claude_removes_then_adds_via_cli(home, monkeypatch):
    _claude_on_path(monkeypatch)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _Result(returncode=0)

    monkeypatch.setattr(_agents.subprocess, "run", fake_run)
    _agents.register("claude-code")
    # Idempotent: remove first, then add (matches the installer).
    assert calls[0][1:4] == ["mcp", "remove", "biopb"]
    assert calls[1][1:5] == ["mcp", "add", "--scope", "user"]
    # The resolved command + stdio transport are passed through to `add`.
    assert _CMD in calls[1]
    assert "--transport" in calls[1] and "stdio" in calls[1]


def test_register_claude_raises_when_add_fails(home, monkeypatch):
    _claude_on_path(monkeypatch)
    results = iter([_Result(0), _Result(1, stderr="boom")])  # remove ok, add fails
    monkeypatch.setattr(_agents.subprocess, "run", lambda argv, **kw: next(results))
    with pytest.raises(_agents.AgentError):
        _agents.register("claude-code")


def test_register_claude_raises_when_cli_missing(home, monkeypatch):
    _no_binaries(monkeypatch)  # no claude on PATH
    with pytest.raises(_agents.AgentError):
        _agents.register("claude-code")


def test_register_never_calls_claude_get_or_list(home, monkeypatch):
    # Status must stay subprocess-free and register must never probe with a
    # connection test (`claude mcp get`/`list` would spawn biopb-mcp).
    _claude_on_path(monkeypatch)
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv[1:])
        return _Result(0)

    monkeypatch.setattr(_agents.subprocess, "run", fake_run)
    _agents.status("claude-code")  # no subprocess at all
    _agents.register("claude-code")
    verbs = [a[1] for a in seen]  # the mcp subcommand
    assert "get" not in verbs and "list" not in verbs
