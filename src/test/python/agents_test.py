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
    # Codex reads $CODEX_HOME ahead of ~/.codex; drop it so a developer who has
    # one set can't have a real Codex install leak into these assertions.
    monkeypatch.delenv("CODEX_HOME", raising=False)
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
    assert ids == ["claude-code", "claude-desktop", "codex-cli", "cursor", "opencode"]
    assert "hermes" not in ids


def test_every_spec_axis_has_an_implementation():
    """Each dispatch axis must resolve for every shipped client.

    The dispatch tables have no default branch on purpose, so this is what keeps
    a new client (or a typo in one) from reaching a user: an unimplemented axis
    fails here rather than in someone's config file.
    """
    for spec in _agents.supported():
        assert spec.manager in _agents._WRITERS, spec.id
        assert spec.config_format in _agents._READERS, spec.id
        assert spec.entry_style in _agents._SHAPES, spec.id


def _inject(monkeypatch, **overrides):
    """A catalog entry with one axis set to something unimplemented."""
    base = {
        "id": "ghost",
        "name": "Ghost Client",
        "manager": "json",
        "config_format": "json",
        "parent_key": "mcpServers",
        "entry_style": "stdio",
    }
    spec = _agents.AgentSpec(**{**base, **overrides})
    monkeypatch.setitem(_agents._SPECS_BY_ID, "ghost", spec)
    return spec


def test_unimplemented_manager_raises_instead_of_writing_json(home, monkeypatch):
    """The regression this dispatch exists for.

    A manager with no writer used to fall through to the JSON writer, which
    happily emitted a JSON document at whatever path the client used — a
    ``cordis.yml`` or a ``config.toml``. It must raise and touch nothing.
    """
    _inject(monkeypatch, manager="yaml")
    cfg = home / "cordis.yml"
    monkeypatch.setattr(_agents, "_config_path", lambda spec: cfg)
    with pytest.raises(_agents.AgentError):
        _agents.register("ghost")
    assert not cfg.exists()


def test_unimplemented_manager_raises_on_unregister(home, monkeypatch):
    _inject(monkeypatch, manager="yaml")
    monkeypatch.setattr(_agents, "_config_path", lambda spec: home / "cordis.yml")
    with pytest.raises(_agents.AgentError):
        _agents.unregister("ghost")


def test_unimplemented_config_format_raises_rather_than_guessing(home, monkeypatch):
    """A format with no reader must not be parsed as JSON on the status path."""
    _inject(monkeypatch, config_format="yaml")
    cfg = home / "cordis.yml"
    cfg.write_text("mcpServers:\n  biopb:\n    command: x\n")
    monkeypatch.setattr(_agents, "_config_path", lambda spec: cfg)
    with pytest.raises(_agents.AgentError):
        _agents.status("ghost")


def test_unimplemented_entry_style_raises_rather_than_writing_stdio(home, monkeypatch):
    _inject(monkeypatch, entry_style="cordis")
    cfg = home / "config.json"
    monkeypatch.setattr(_agents, "_config_path", lambda spec: cfg)
    with pytest.raises(_agents.AgentError):
        _agents.register("ghost")
    assert not cfg.exists()


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
        "codex-cli",
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


# --------------------------------------------------------------------------- #
# opencode .json / .jsonc drift hardening (biopb/biopb#536)
# --------------------------------------------------------------------------- #


def _opencode_dir(home):
    d = home / ".config" / "opencode"
    d.mkdir(parents=True)
    return d


def test_opencode_targets_existing_jsonc_over_json(home):
    # A user who keeps opencode.jsonc: we must target THAT file, not a shadow
    # opencode.json opencode may ignore.
    d = _opencode_dir(home)
    (d / "opencode.jsonc").write_text('{\n  "theme": "dark"\n}\n')
    _agents.register("opencode")
    data = json.loads((d / "opencode.jsonc").read_text())
    assert data["mcp"]["biopb"]["type"] == "local"
    assert data["theme"] == "dark"  # sibling preserved
    assert not (d / "opencode.json").exists()  # no shadow file written
    assert _agents.status("opencode")["state"] == "registered"


def test_opencode_fresh_install_creates_json(home):
    # Neither file exists -> canonical opencode.json is the create target.
    _opencode_dir(home)
    _agents.register("opencode")
    path = _agents._config_path(_agents._spec("opencode"))
    assert path.name == "opencode.json"
    assert path.exists()


def test_opencode_register_refuses_commented_jsonc(home):
    d = _opencode_dir(home)
    original = '{\n  // my agent config\n  "theme": "dark",\n}\n'
    (d / "opencode.jsonc").write_text(original)
    with pytest.raises(_agents.AgentError) as exc:
        _agents.register("opencode")
    # Fails safe: the commented file is left byte-for-byte, no shadow .json.
    assert (d / "opencode.jsonc").read_text() == original
    assert not (d / "opencode.json").exists()
    # And the message hands the user a ready-to-paste entry.
    assert '"biopb"' in str(exc.value)


def test_opencode_status_detects_entry_in_commented_jsonc(home):
    # A commented jsonc that already registers biopb must read as "registered",
    # not silently fall back to "installed".
    d = _opencode_dir(home)
    (d / "opencode.jsonc").write_text(
        "{\n"
        "  // comment tolerated by the status reader\n"
        '  "mcp": {\n'
        '    "biopb": {"type": "local", "command": ["'
        + _CMD
        + '", "--transport", "stdio"], "enabled": true},\n'
        "  },\n"
        "}\n"
    )
    assert _agents.status("opencode")["state"] == "registered"


def test_opencode_register_merges_clean_jsonc_in_place(home):
    # A .jsonc with no comments is strict JSON: safe to merge and rewrite (nothing
    # lost), so registration proceeds rather than refusing.
    d = _opencode_dir(home)
    (d / "opencode.jsonc").write_text('{"theme": "dark"}')
    s = _agents.register("opencode")
    assert s["state"] == "registered"
    data = json.loads((d / "opencode.jsonc").read_text())
    assert data["theme"] == "dark"
    assert data["mcp"]["biopb"]["enabled"] is True


def test_opencode_unregister_commented_jsonc_with_entry_raises(home):
    d = _opencode_dir(home)
    (d / "opencode.jsonc").write_text(
        "{\n"
        "  // keep me\n"
        '  "mcp": {"biopb": {"type": "local", "command": ["x"], "enabled": true}},\n'
        "}\n"
    )
    with pytest.raises(_agents.AgentError):
        _agents.unregister("opencode")


def test_opencode_unregister_commented_jsonc_without_entry_is_silent(home):
    d = _opencode_dir(home)
    (d / "opencode.jsonc").write_text('{\n  // no biopb here\n  "theme": "dark",\n}\n')
    # Nothing to remove -> stays idempotent, no raise.
    _agents.unregister("opencode")


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


# --------------------------------------------------------------------------- #
# Codex CLI (TOML config, read-only; writes go through the `codex` CLI)
# --------------------------------------------------------------------------- #


def _codex_on_path(monkeypatch):
    monkeypatch.setattr(
        _agents.shutil,
        "which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )


def _write_codex_config(home, command=_CMD, extra=""):
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        extra + f'[mcp_servers.biopb]\ncommand = "{command}"\n'
        'args = ["--transport", "stdio"]\n'
    )
    return cfg


def test_codex_state_transitions(home, monkeypatch):
    # No ~/.codex and no binary -> not installed.
    _no_binaries(monkeypatch)
    assert _agents.status("codex-cli")["state"] == "not_installed"
    # On PATH, no entry -> installed.
    _codex_on_path(monkeypatch)
    assert _agents.status("codex-cli")["state"] == "installed"
    # The biopb table -> registered, and the command matches so no drift.
    _write_codex_config(home)
    s = _agents.status("codex-cli")
    assert s["state"] == "registered" and s["drifted"] is False


def test_codex_home_dir_alone_counts_as_installed(home, monkeypatch):
    """A Codex install we can't see on PATH still shows up by its home dir."""
    _no_binaries(monkeypatch)
    (home / ".codex").mkdir()
    assert _agents.status("codex-cli")["state"] == "installed"


def test_codex_honors_codex_home_env(home, monkeypatch):
    """$CODEX_HOME relocates the whole Codex home, config.toml included."""
    elsewhere = home / "relocated"
    elsewhere.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(elsewhere))
    assert _agents.status("codex-cli")["config_path"] == str(elsewhere / "config.toml")
    (elsewhere / "config.toml").write_text(f'[mcp_servers.biopb]\ncommand = "{_CMD}"\n')
    assert _agents.status("codex-cli")["state"] == "registered"


def test_codex_registered_is_drift_when_command_differs(home, monkeypatch):
    _codex_on_path(monkeypatch)
    _write_codex_config(home, command="/somewhere/else/biopb-mcp")
    s = _agents.status("codex-cli")
    assert s["state"] == "registered" and s["drifted"] is True


def test_codex_status_ignores_malformed_toml(home, monkeypatch):
    _codex_on_path(monkeypatch)
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text("[mcp_servers.biopb\ncommand = ")
    assert _agents.status("codex-cli")["state"] == "installed"


def test_codex_status_ignores_a_sibling_server(home, monkeypatch):
    """Another MCP server in the file is not biopb."""
    _codex_on_path(monkeypatch)
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('[mcp_servers.other]\ncommand = "npx"\n')
    assert _agents.status("codex-cli")["state"] == "installed"


def test_codex_register_adds_via_cli(home, monkeypatch):
    _codex_on_path(monkeypatch)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _Result(returncode=0)

    monkeypatch.setattr(_agents.subprocess, "run", fake_run)
    _agents.register("codex-cli")
    # One call: `codex mcp add` overwrites in place, so no remove-then-add.
    assert len(calls) == 1
    assert calls[0][1:5] == ["mcp", "add", "biopb", "--"]
    assert _CMD in calls[0]
    assert "--transport" in calls[0] and "stdio" in calls[0]


def test_codex_unregister_removes_via_cli(home, monkeypatch):
    _codex_on_path(monkeypatch)
    calls = []
    monkeypatch.setattr(
        _agents.subprocess,
        "run",
        lambda argv, **kw: (calls.append(argv), _Result(0))[1],
    )
    _agents.unregister("codex-cli")
    assert calls[0][1:4] == ["mcp", "remove", "biopb"]


def test_codex_register_raises_when_add_fails(home, monkeypatch):
    _codex_on_path(monkeypatch)
    monkeypatch.setattr(
        _agents.subprocess, "run", lambda argv, **kw: _Result(1, stderr="boom")
    )
    with pytest.raises(_agents.AgentError):
        _agents.register("codex-cli")


def test_codex_register_raises_when_cli_missing(home, monkeypatch):
    _no_binaries(monkeypatch)
    with pytest.raises(_agents.AgentError):
        _agents.register("codex-cli")


def test_codex_never_calls_mcp_get_or_list(home, monkeypatch):
    # Same rule as Claude Code: status stays subprocess-free, and neither write
    # probes with `mcp get`/`list` (a live connection test spawning biopb-mcp).
    _codex_on_path(monkeypatch)
    seen = []
    monkeypatch.setattr(
        _agents.subprocess,
        "run",
        lambda argv, **kw: (seen.append(argv[1:]), _Result(0))[1],
    )
    _agents.status("codex-cli")  # no subprocess at all
    _agents.register("codex-cli")
    _agents.unregister("codex-cli")
    verbs = [a[1] for a in seen]
    assert "get" not in verbs and "list" not in verbs


# --- the 3.10 fallback (no tomllib) ---------------------------------------- #


@pytest.fixture
def no_tomllib(monkeypatch):
    """Force the hand-rolled scanner that stands in for tomllib on 3.10."""
    monkeypatch.setattr(_agents, "tomllib", None)


def test_scanner_reads_the_biopb_command(home, monkeypatch, no_tomllib):
    _codex_on_path(monkeypatch)
    _write_codex_config(home)
    s = _agents.status("codex-cli")
    assert s["state"] == "registered" and s["drifted"] is False


def test_scanner_stops_at_the_next_table(home, monkeypatch, no_tomllib):
    """A command in a LATER table must not be read as biopb's."""
    _codex_on_path(monkeypatch)
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(
        "[mcp_servers.biopb]\n"
        'args = ["--transport", "stdio"]\n'
        "\n[mcp_servers.other]\n"
        f'command = "{_CMD}"\n'
    )
    # biopb's own table has no command -> unreadable, so not registered.
    assert _agents.status("codex-cli")["state"] == "installed"


def test_scanner_skips_a_preceding_table(home, monkeypatch, no_tomllib):
    _codex_on_path(monkeypatch)
    _write_codex_config(
        home, extra='model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "npx"\n\n'
    )
    assert _agents.status("codex-cli")["state"] == "registered"


def test_scanner_reads_a_literal_string(home, monkeypatch, no_tomllib):
    """Windows paths land in a literal string, where backslashes are verbatim."""
    _codex_on_path(monkeypatch)
    monkeypatch.setattr(_agents, "_mcp_executable", lambda: r"C:\biopb\biopb-mcp.exe")
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text("[mcp_servers.biopb]\ncommand = 'C:\\biopb\\biopb-mcp.exe'\n")
    s = _agents.status("codex-cli")
    assert s["state"] == "registered" and s["drifted"] is False


def test_scanner_gives_up_on_an_unquoted_value(home, monkeypatch, no_tomllib):
    _codex_on_path(monkeypatch)
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text("[mcp_servers.biopb]\ncommand = biopb-mcp\n")
    assert _agents.status("codex-cli")["state"] == "installed"
