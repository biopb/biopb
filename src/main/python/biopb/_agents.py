"""Register biopb-mcp with local AI agent clients — shared, stdlib-only.

An MCP client (Claude Code, Claude Desktop, Cursor, opencode, …) spawns
``biopb-mcp`` over stdio; wiring biopb into a client means writing a small MCP
server entry into that client's config. The installer already does this once at
install time (``install/install.sh`` + ``install/biopb-engine.ps1``); this module
is the same knowledge as an importable Python API so the control-plane dashboard
and the ``biopb agents`` CLI can *also* do it after install — the user installs,
say, Claude Code later and registers it from the dashboard with one click.

It is the single source of truth for the catalog going forward: the two installer
scripts are meant to delegate here (their hand-kept-in-sync copies collapse into
one). Kept **stdlib-only** — like ``_endpoints`` / ``_sessions`` — so
importing it never drags in a heavy stack, and so both the lean control plane and
the core CLI can call it.

Three things it does per client:

- **status** — a subprocess-free read (``not_installed`` / ``installed`` /
  ``registered``, plus ``drifted``). Deliberately never spawns anything: it is
  polled by the dashboard, and (for Claude Code) ``claude mcp get``/``list`` run a
  *live connection test* that would launch ``biopb-mcp`` on every refresh. So
  status is always a plain config-file read.
- **register** — write the biopb entry. The calm JSON configs (Claude Desktop,
  Cursor, opencode) get an atomic read-merge-replace that preserves every other
  key. Claude Code goes through its ``claude`` CLI (``mcp add --scope user``):
  ``~/.claude.json`` is a busy file Claude Code rewrites constantly, so we let it
  serialize its own writes rather than race it with our merge.
- **unregister** — the inverse; idempotent (removing an absent entry is fine).

The registered command is the **absolute path** to ``biopb-mcp`` (resolved beside
this interpreter, then PATH), because GUI clients launch it without inheriting a
shell PATH (the same reason ``_control_client._biopb_executable`` resolves
absolutely). That absolute path is also the drift signal: if biopb is reinstalled
elsewhere, the stored command no longer matches the freshly resolved one, and the
client's status comes back ``registered`` with ``drifted=True`` so the UI can
offer a Re-register.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The invocation every client registers: `biopb-mcp --transport stdio`. The
# command itself is resolved per call (_mcp_command) so a reinstall that moves
# biopb-mcp is reflected as drift rather than baked in here.
_MCP_ARGS = ("--transport", "stdio")


class AgentError(Exception):
    """A register/unregister could not be completed (bad config, CLI missing,
    unwritable file). Carries a human-facing message the CLI/API surfaces."""


@dataclass(frozen=True)
class AgentSpec:
    """Static per-client knowledge.

    ``manager`` selects how register/unregister act: ``"json"`` edits a config
    file directly (atomic merge/delete under ``parent_key``), ``"claude-cli"``
    shells out to the ``claude`` CLI. ``entry_style`` selects the MCP entry shape
    for JSON clients: ``"stdio"`` is the canonical ``{command, args}`` form every
    ``mcpServers`` client accepts; ``"opencode"`` is opencode's
    ``{type: "local", command: [...]}`` form. ``parent_key`` is the top-level key
    the biopb entry lives under (``mcpServers`` for most, ``mcp`` for opencode).
    """

    id: str
    name: str
    manager: str  # "json" | "claude-cli"
    parent_key: str  # "mcpServers" | "mcp"
    entry_style: str  # "stdio" | "opencode"


# The catalog — consistent with the installer (install/install.sh,
# install/biopb-engine.ps1). Hermes is intentionally omitted: the installer only
# ever prints a manual YAML snippet for it (it will not edit YAML), so it can
# never reach `registered` through a button — not worth a dead row.
_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec("claude-code", "Claude Code", "claude-cli", "mcpServers", "stdio"),
    AgentSpec("claude-desktop", "Claude Desktop", "json", "mcpServers", "stdio"),
    AgentSpec("cursor", "Cursor", "json", "mcpServers", "stdio"),
    AgentSpec("opencode", "opencode", "json", "mcp", "opencode"),
)

_SPECS_BY_ID = {s.id: s for s in _SPECS}


def _spec(spec_id: str) -> AgentSpec:
    try:
        return _SPECS_BY_ID[spec_id]
    except KeyError:
        raise AgentError(f"unknown agent client {spec_id!r}")


# --------------------------------------------------------------------------- #
# Resolving the biopb-mcp command to register
# --------------------------------------------------------------------------- #


def _mcp_executable() -> Optional[str]:
    """Absolute path to the ``biopb-mcp`` console script, or ``None`` if not found.

    Prefer the script installed beside this interpreter (the venv / uv-tool
    ``Scripts``/``bin`` dir where ``biopb-mcp`` lands), so we register the same
    environment that shipped biopb even when PATH is not inherited; fall back to
    PATH. Mirrors ``biopb_mcp._control_client._biopb_executable`` — do NOT
    ``resolve()`` ``sys.executable`` first, or a symlinked venv python would lead
    the sibling lookup out of the venv bin dir.
    """
    name = "biopb-mcp.exe" if os.name == "nt" else "biopb-mcp"
    sibling = Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    return shutil.which("biopb-mcp")


def _mcp_command() -> str:
    """The command to register. Falls back to the bare name when the console
    script cannot be located, so a client still gets a working entry if PATH
    resolves ``biopb-mcp`` at launch — the sibling/PATH resolution above only
    fails when neither is present, which is also when the bare name is the best
    we can offer."""
    return _mcp_executable() or "biopb-mcp"


# --------------------------------------------------------------------------- #
# Per-client config location + install detection
# --------------------------------------------------------------------------- #


def _config_path(spec: AgentSpec) -> Optional[Path]:
    """The config file biopb's entry lives in, or ``None`` on a platform where the
    client has no known location. Resolved at call time (not cached) so a test
    that repoints ``Path.home()`` / ``$APPDATA`` gets an isolated location."""
    home = Path.home()
    if spec.id == "claude-code":
        # Claude Code stores user-scope MCP servers in ~/.claude.json under the
        # top-level `mcpServers` key. We only READ this for status; register/
        # unregister go through the `claude` CLI (see _run_claude).
        return home / ".claude.json"
    if spec.id == "claude-desktop":
        if sys.platform == "win32":
            base = os.environ.get("APPDATA")
            root = Path(base) if base else home / "AppData" / "Roaming"
            return root / "Claude" / "claude_desktop_config.json"
        if sys.platform == "darwin":
            return (
                home
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json"
            )
        return home / ".config" / "Claude" / "claude_desktop_config.json"
    if spec.id == "cursor":
        return home / ".cursor" / "mcp.json"
    if spec.id == "opencode":
        return _opencode_config_path()
    return None


def _opencode_config_path() -> Path:
    """The opencode global config biopb should target (biopb/biopb#536).

    opencode reads either ``opencode.jsonc`` or ``opencode.json``. Prefer an
    existing ``.jsonc`` so we edit the file opencode actually honors instead of
    writing a shadow ``.json`` it may ignore; otherwise fall back to ``.json`` —
    the canonical file we create on a fresh install. Resolved at call time so a
    test that repoints ``Path.home()`` is isolated."""
    base = Path.home() / ".config" / "opencode"
    jsonc = base / "opencode.jsonc"
    if jsonc.exists():
        return jsonc
    return base / "opencode.json"


def _is_installed(spec: AgentSpec) -> bool:
    """Whether the client appears present — the same signals the installer uses.

    Deliberately cheap and subprocess-free: a binary on PATH or a well-known
    config directory. A false negative (a portable/flatpak install we can't see)
    just shows ``not_installed``; register still works as an escape hatch.
    """
    if spec.id == "claude-code":
        return shutil.which("claude") is not None
    if spec.id == "opencode":
        if shutil.which("opencode") is not None:
            return True
        return (Path.home() / ".config" / "opencode").is_dir()
    # Claude Desktop / Cursor: the app owns a config directory; its presence is
    # the install signal (the config file itself may not exist until first use).
    path = _config_path(spec)
    return path is not None and path.parent.is_dir()


# --------------------------------------------------------------------------- #
# Reading status (subprocess-free)
# --------------------------------------------------------------------------- #


def _load_json_object(path: Path) -> dict:
    """Parse ``path`` as a JSON object. ``{}`` if it does not exist; raises
    :class:`AgentError` if it exists but is unreadable or not an object — so a
    write never clobbers a config we could not understand."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentError(f"could not read {path}: {exc}")
    if not isinstance(data, dict):
        raise AgentError(f"{path} is not a JSON object")
    return data


def _strip_jsonc(text: str) -> str:
    """Best-effort JSONC → JSON so :func:`json.loads` can read an opencode
    ``.jsonc``: drop ``//`` and ``/* */`` comments (outside string literals) and
    trailing commas (biopb/biopb#536).

    A **read-only** transform used only for status detection — it is intentionally
    never used to rewrite a file, because it is lossy (it discards the comments).
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:  # keep an escaped char verbatim
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _load_json_tolerant(path: Path) -> Optional[dict]:
    """Read ``path`` as a JSON object, tolerating ``.jsonc`` comments / trailing
    commas. ``None`` (never raises) on any problem — the status-read path, where a
    config we cannot parse simply reads as "not registered"."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    candidates = (text, _strip_jsonc(text)) if path.suffix == ".jsonc" else (text,)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        return data if isinstance(data, dict) else None
    return None


def _jsonc_unmergeable(path: Path) -> bool:
    """True when ``path`` is a ``.jsonc`` our strict-JSON writer must not edit:
    it parses only after comment/trailing-comma stripping, so rewriting it would
    silently drop the user's comments (biopb/biopb#536). A ``.jsonc`` that is
    already strict JSON (nothing to lose) returns False and is merged in place;
    so does a ``.json`` or a missing file."""
    if path.suffix != ".jsonc" or not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if not text.strip():
        return False
    try:
        json.loads(text)
        return False
    except ValueError:
        return True


def _manual_edit_message(spec: AgentSpec, path: Path, *, removing: bool) -> str:
    """The AgentError text shown when biopb can't safely edit a commented
    ``.jsonc`` — a clear manual instruction instead of clobbering comments or
    writing a shadow config (biopb/biopb#536)."""
    if removing:
        return (
            f"{path} has comments, so biopb won't rewrite it (that would drop "
            f'them). Remove the "biopb" key under "{spec.parent_key}" by hand.'
        )
    snippet = json.dumps({spec.parent_key: {"biopb": _mcp_entry(spec)}}, indent=2)
    return (
        f"{path} has comments, so biopb won't rewrite it (that would drop them). "
        f"Add this entry under the top level by hand:\n{snippet}"
    )


def _read_entry(spec: AgentSpec, path: Optional[Path]) -> Optional[dict]:
    """The biopb MCP entry currently in the client's config, or ``None``.

    A read-only, exception-tolerant probe used for status (unlike
    :func:`_load_json_object` it never raises — a malformed config simply reads as
    "not registered" for display, and the write path reports the parse error).
    Reads ``.jsonc`` tolerantly so a commented opencode config is still detected as
    registered rather than silently reading as "installed" (biopb/biopb#536).
    """
    if path is None or not path.exists():
        return None
    data = _load_json_tolerant(path)
    if not isinstance(data, dict):
        return None
    parent = data.get(spec.parent_key)
    if isinstance(parent, dict):
        entry = parent.get("biopb")
        if isinstance(entry, dict):
            return entry
    return None


def _entry_command(spec: AgentSpec, entry: dict) -> Optional[str]:
    """Extract the executable path a registered entry points at, for drift.

    ``stdio`` entries store ``command`` as a string; ``opencode`` stores
    ``command`` as a list whose first element is the executable. ``None`` when the
    entry has no recognizable command (treated as drift so a malformed prior entry
    prompts a Re-register)."""
    command = entry.get("command")
    if spec.entry_style == "opencode":
        if isinstance(command, list) and command:
            return command[0] if isinstance(command[0], str) else None
        return None
    return command if isinstance(command, str) else None


def status(spec_id: str) -> dict:
    """One client's status: ``{id, name, state, drifted, config_path}``.

    ``state`` is ``registered`` if the biopb entry is present (regardless of
    detection — the entry is ground truth), else ``installed`` if the client is
    detected, else ``not_installed``. ``drifted`` is set only when ``registered``
    and the stored command no longer matches the freshly resolved ``biopb-mcp``
    path (a moved/reinstalled biopb), so the UI can offer a Re-register.
    """
    spec = _spec(spec_id)
    path = _config_path(spec)
    entry = _read_entry(spec, path)
    if entry is not None:
        state = "registered"
        drifted = _entry_command(spec, entry) != _mcp_command()
    elif _is_installed(spec):
        state, drifted = "installed", False
    else:
        state, drifted = "not_installed", False
    return {
        "id": spec.id,
        "name": spec.name,
        "state": state,
        "drifted": drifted,
        "config_path": str(path) if path is not None else None,
    }


def supported() -> list[AgentSpec]:
    """The static client catalog."""
    return list(_SPECS)


def statuses() -> list[dict]:
    """Status for every supported client, in catalog order."""
    return [status(s.id) for s in _SPECS]


# --------------------------------------------------------------------------- #
# Writing: register / unregister
# --------------------------------------------------------------------------- #


def _mcp_entry(spec: AgentSpec) -> dict:
    """The MCP server entry to write for ``spec``, in its client's shape."""
    command = _mcp_command()
    if spec.entry_style == "opencode":
        # opencode: {type:"local", command:[exe, ...args], enabled:true}
        return {
            "type": "local",
            "command": [command, *_MCP_ARGS],
            "enabled": True,
        }
    # Canonical mcpServers stdio form: bare command+args, no "type" (a stray
    # "type" trips stricter validators — matches the installer's choice).
    return {"command": command, "args": list(_MCP_ARGS)}


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace`` in the
    same dir), so a client reading concurrently never sees a half-written config.
    Same idiom as ``_sessions``/``cli._write_pid_file``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _run_claude(args: list[str], *, required: bool) -> tuple[int, str]:
    """Run ``claude <args>`` windowless, returning ``(returncode, output)``.

    ``required`` distinguishes the two callers: the ``mcp add`` that must succeed
    (``required=True`` → a missing ``claude`` is an :class:`AgentError`) from the
    best-effort ``mcp remove`` we run before an add to stay idempotent
    (``required=False`` → tolerate a non-zero code, i.e. "wasn't registered").
    We never call ``claude mcp get``/``list`` — those run a live connection test
    that would spawn ``biopb-mcp``.
    """
    exe = shutil.which("claude")
    if exe is None:
        if required:
            raise AgentError("the `claude` CLI is not on PATH")
        return 1, ""
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            [exe, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentError(f"`claude {' '.join(args)}` failed: {exc}")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _register_json(spec: AgentSpec) -> None:
    path = _config_path(spec)
    if path is None:
        raise AgentError(f"{spec.name} has no known config location on this platform")
    if _jsonc_unmergeable(path):
        raise AgentError(_manual_edit_message(spec, path, removing=False))
    data = _load_json_object(path)
    parent = data.get(spec.parent_key)
    if not isinstance(parent, dict):
        parent = {}
    parent["biopb"] = _mcp_entry(spec)
    data[spec.parent_key] = parent
    _write_json_atomic(path, data)


def _unregister_json(spec: AgentSpec) -> None:
    path = _config_path(spec)
    if path is None or not path.exists():
        return  # nothing registered
    if _jsonc_unmergeable(path):
        # Can't safely rewrite a commented .jsonc. If biopb is actually present,
        # surface a manual-removal instruction rather than silently leaving a
        # stale entry (biopb/biopb#536); otherwise there is nothing to do.
        if _read_entry(spec, path) is not None:
            raise AgentError(_manual_edit_message(spec, path, removing=True))
        return
    data = _load_json_object(path)
    parent = data.get(spec.parent_key)
    if isinstance(parent, dict) and "biopb" in parent:
        del parent["biopb"]
        _write_json_atomic(path, data)


def _register_claude() -> None:
    # Idempotent: drop any existing entry, then add (matches the installer). The
    # remove is best-effort (a not-yet-registered client returns non-zero), the
    # add must succeed.
    _run_claude(["mcp", "remove", "biopb", "-s", "user"], required=False)
    code, out = _run_claude(
        ["mcp", "add", "--scope", "user", "biopb", "--", _mcp_command(), *_MCP_ARGS],
        required=True,
    )
    if code != 0:
        raise AgentError(f"`claude mcp add` failed: {out.strip()}")


def _unregister_claude() -> None:
    _run_claude(["mcp", "remove", "biopb", "-s", "user"], required=True)


def register(spec_id: str) -> dict:
    """Register biopb with the client and return its fresh status.

    Works regardless of detection (the "register anyway" escape hatch for a client
    we could not auto-detect); a genuinely absent client surfaces as an
    :class:`AgentError` (e.g. Claude Code with no ``claude`` on PATH). Raises
    :class:`AgentError` on any failure — the caller renders it.
    """
    spec = _spec(spec_id)
    if spec.manager == "claude-cli":
        _register_claude()
    else:
        _register_json(spec)
    logger.info("registered biopb with %s", spec.name)
    return status(spec_id)


def unregister(spec_id: str) -> dict:
    """Remove biopb from the client and return its fresh status. Idempotent."""
    spec = _spec(spec_id)
    if spec.manager == "claude-cli":
        _unregister_claude()
    else:
        _unregister_json(spec)
    logger.info("unregistered biopb from %s", spec.name)
    return status(spec_id)
