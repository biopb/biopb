"""Register biopb-shim with local AI agent clients -- shared, stdlib-only.

Wiring biopb into an MCP client (Claude Code, Claude Desktop, Cursor, opencode,
Codex CLI) means writing a small server entry into its config. This is the
catalog and API behind the dashboard and ``biopb agents``; stdlib-only so the
lean control plane and the core CLI can both import it.

Each client is a :class:`ClientBackend`; adding one is a class plus a line in
``_CLIENTS``. The format and entry-shape tables (``_READERS``, ``_SHAPES``) have
no default branch and the write path is abstract, so an omission is a
``TypeError``/``AgentError`` rather than a config written in the wrong format.

Per client:

- **status** -- a subprocess-free config read (``not_installed`` / ``installed``
  / ``registered``, plus ``drifted``). Never ``claude mcp get``/``list``: they
  run a live connection test that would launch ``biopb-shim`` on every poll.
- **register** -- JSON configs (Claude Desktop, Cursor, opencode) get an atomic
  read-merge-replace. Claude Code and Codex go through their own CLIs: Claude
  Code rewrites ``~/.claude.json`` constantly, and only Codex's editor keeps
  its TOML comments and sibling servers intact (we have no TOML writer).
- **unregister** -- the inverse; idempotent.

The registered command is the absolute path to ``biopb-shim`` (GUI clients
launch it without a shell PATH). If biopb moves, the stored command no longer
matches the resolved one and status reports ``drifted=True``.
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

try:  # tomllib is 3.11+; on 3.10 (our floor) _scan_toml_entry stands in
    import tomllib
except ImportError:  # pragma: no cover - only reached on 3.10
    tomllib = None  # type: ignore[assignment]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base args for the registered `biopb-shim`; the command is resolved per call
# (_mcp_command) so a move shows up as drift.
_MCP_ARGS = ()


class AgentError(Exception):
    """A register/unregister could not be completed (bad config, CLI missing,
    unwritable file). Carries a human-facing message the CLI/API surfaces."""


# --------------------------------------------------------------------------- #
# Resolving the biopb-shim command to register
# --------------------------------------------------------------------------- #


def console_script(name: str) -> Optional[str]:
    """Absolute path to the console script *name*, or ``None`` if not found.

    Prefers the script beside this interpreter, then PATH. Do NOT ``resolve()``
    ``sys.executable``: a venv's ``python`` symlinks to the base interpreter,
    which would lead the lookup out of the venv bin dir.
    """
    sibling = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    if sibling.exists():
        return str(sibling)
    return shutil.which(name)


def _mcp_executable() -> Optional[str]:
    """Absolute path to the ``biopb-shim`` console script, or ``None``."""
    return console_script("biopb-shim")


def _mcp_command() -> str:
    """The command to register; the bare name if the script cannot be located."""
    return _mcp_executable() or "biopb-shim"


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
    trailing commas.

    Read-only, for status detection: it is lossy (drops comments), so never use
    it to rewrite a file.
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
    commas. ``None`` (never raises) on any problem, which reads as "not
    registered"."""
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


def _read_toml_entry(path: Path, parent_key: str) -> Optional[dict]:
    """The biopb table from Codex's ``config.toml``, shaped like a JSON stdio
    entry (``{command, args}``) so :func:`_entry_command` and :func:`status` need
    no TOML-specific branch.

    Read-only: writes go through ``codex mcp add``/``remove``, since re-emitting
    a parsed dict would drop the user's comments. ``None`` (never raises) on any
    problem.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if tomllib is None:  # pragma: no cover - only on 3.10; forced in tests
        return _scan_toml_entry(text, parent_key)
    try:
        data = tomllib.loads(text)
    except ValueError:
        return None
    parent = data.get(parent_key)
    if isinstance(parent, dict):
        entry = parent.get("biopb")
        if isinstance(entry, dict):
            return entry
    return None


def _scan_toml_entry(text: str, parent_key: str) -> Optional[dict]:
    """``_read_toml_entry`` for Python 3.10, which has no ``tomllib``.

    Pulls just ``command`` and ``args`` out of the ``[<parent_key>.biopb]``
    table. Deliberately narrow: it reads the shape ``codex mcp add`` writes and
    gives up on anything else ("not registered").
    """
    header = re.compile(r"^\s*\[\s*" + re.escape(parent_key) + r"\s*\.\s*biopb\s*\]")
    in_table = False
    found: dict = {}
    for line in text.splitlines():
        if line.lstrip().startswith("["):
            if in_table:
                break  # the next table ends ours
            in_table = bool(header.match(line))
            continue
        if not in_table:
            continue
        key, sep, raw = line.partition("=")
        key = key.strip()
        if sep and key == "command":
            value = _toml_string(raw.strip())
            if value is None:
                return None
            found["command"] = value
        elif sep and key == "args":
            try:
                args = json.loads(raw.strip())  # basic strings are JSON's
            except ValueError:
                continue  # unreadable args read as drift, not as "unregistered"
            if isinstance(args, list):
                found["args"] = args
    return found if "command" in found else None


def _toml_string(raw: str) -> Optional[str]:
    """A single-line TOML basic (``"..."``, escapes decoded) or literal
    (``'...'``, verbatim) string, or ``None`` if ``raw`` is neither. A trailing
    comment makes the value unparseable (fails safe) rather than guessing where a
    ``#`` inside a path ends."""
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        try:
            return json.loads(raw)  # TOML basic escapes are a subset of JSON's
        except ValueError:
            return None
    return None


def _read_json_entry(path: Path, parent_key: str) -> Optional[dict]:
    """The biopb entry from a JSON (or ``.jsonc``) config, or ``None``."""
    data = _load_json_tolerant(path)
    if not isinstance(data, dict):
        return None
    parent = data.get(parent_key)
    if isinstance(parent, dict):
        entry = parent.get("biopb")
        if isinstance(entry, dict):
            return entry
    return None


#: config_format -> reader. No default: an unimplemented format raises instead
#: of being parsed as JSON.
_READERS = {
    "json": _read_json_entry,
    "toml": _read_toml_entry,
}


# --------------------------------------------------------------------------- #
# Entry shapes
# --------------------------------------------------------------------------- #
# Each style pairs a builder (what to write) with extractors (read back for
# drift); they must stay inverses of each other.


def _stdio_entry(command: str, args) -> dict:
    # No "type": a stray one trips stricter validators.
    return {"command": command, "args": list(args)}


def _stdio_command(entry: dict) -> Optional[str]:
    command = entry.get("command")
    return command if isinstance(command, str) else None


def _stdio_args(entry: dict) -> Optional[list]:
    args = entry.get("args")
    if isinstance(args, list) and all(isinstance(a, str) for a in args):
        return args
    return None


def _opencode_entry(command: str, args) -> dict:
    return {"type": "local", "command": [command, *args], "enabled": True}


def _opencode_command(entry: dict) -> Optional[str]:
    command = entry.get("command")
    if isinstance(command, list) and command:
        return command[0] if isinstance(command[0], str) else None
    return None


def _opencode_args(entry: dict) -> Optional[list]:
    command = entry.get("command")
    if (
        isinstance(command, list)
        and command
        and all(isinstance(a, str) for a in command)
    ):
        return command[1:]
    return None


#: entry_style -> (builder, command extractor, args extractor). No default,
#: like _READERS.
_SHAPES = {
    "stdio": (_stdio_entry, _stdio_command, _stdio_args),
    "opencode": (_opencode_entry, _opencode_command, _opencode_args),
}


def _dispatch(table: dict, key: str, client: ClientBackend, axis: str):
    """Look ``key`` up in ``table``, or raise :class:`AgentError` naming the
    client and the axis."""
    try:
        return table[key]
    except KeyError:
        raise AgentError(f"{client.name}: unsupported {axis} {key!r}")


def _jsonc_unmergeable(path: Path) -> bool:
    """True when ``path`` is a ``.jsonc`` our strict-JSON writer must not edit:
    it parses only after comment/trailing-comma stripping, so rewriting would
    drop the user's comments. Strict-JSON ``.jsonc``, ``.json`` and a missing
    file return False."""
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


# --------------------------------------------------------------------------- #
# Writing helpers
# --------------------------------------------------------------------------- #


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace`` in the
    same dir), so a concurrent reader never sees a half-written config."""
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


def _run_client_cli(
    exe_name: str, args: list[str], *, required: bool
) -> tuple[int, str]:
    """Run ``<exe_name> <args>`` windowless, returning ``(returncode, output)``.

    ``required=True`` makes a missing binary an :class:`AgentError`; ``False``
    is best-effort (a missing binary returns ``(1, "")``).
    """
    exe = shutil.which(exe_name)
    if exe is None:
        if required:
            raise AgentError(f"the `{exe_name}` CLI is not on PATH")
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
        raise AgentError(f"`{exe_name} {' '.join(args)}` failed: {exc}")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# --------------------------------------------------------------------------- #
# Client backends
# --------------------------------------------------------------------------- #
# One object per client: config location, install signal, read/write path.
# The write path is abstract, so an incomplete backend fails at import.


class ClientBackend(ABC):
    """One MCP client biopb can register itself with."""

    #: stable identifier -- the CLI argument and the /api/agents path segment
    id: str
    #: human label for the dashboard and the CLI table
    name: str
    #: the container biopb's entry lives under, named in this format's terms: a
    #: JSON object key (``mcpServers``, ``mcp``) or a TOML table (``mcp_servers``)
    parent_key: str
    #: a key of :data:`_READERS` -- how to parse the config for status
    config_format: str = "json"
    #: a key of :data:`_SHAPES` -- the entry's shape, written and read back out
    entry_style: str = "stdio"
    #: what the client launches ``biopb-shim`` with; a change shows as drift
    mcp_args: tuple = _MCP_ARGS

    @abstractmethod
    def config_path(self) -> Optional[Path]:
        """The config file biopb's entry lives in, or ``None`` on a platform
        where this client has no known location. Resolved at call time (not
        cached) so tests can repoint ``Path.home()`` / ``$APPDATA``."""

    @abstractmethod
    def is_installed(self) -> bool:
        """Whether the client appears present; cheap and subprocess-free (a
        binary on PATH or a config directory, per base class). A false negative
        only shows ``not_installed``; register still works for a
        :class:`JsonConfigClient`."""

    @abstractmethod
    def register(self) -> None:
        """Write biopb's entry into this client's config."""

    @abstractmethod
    def unregister(self) -> None:
        """Remove it. Idempotent -- removing an absent entry is fine."""

    # -- shared, driven by config_format / entry_style ---------------------- #

    def read_entry(self) -> Optional[dict]:
        """The biopb entry currently in this client's config, or ``None``.

        A malformed user config reads as "not registered" (the write path
        reports the parse error); a ``config_format`` with no reader raises.
        """
        path = self.config_path()
        if path is None or not path.exists():
            return None
        read = _dispatch(_READERS, self.config_format, self, "config format")
        return read(path, self.parent_key)

    def entry(self) -> dict:
        """The MCP server entry to write, in this client's shape."""
        build, _, _ = _dispatch(_SHAPES, self.entry_style, self, "entry style")
        return build(_mcp_command(), self.mcp_args)

    def entry_command(self, entry: dict) -> Optional[str]:
        """The executable a registered entry points at, for drift. ``None`` (read
        as drift) when unrecognizable."""
        _, extract, _ = _dispatch(_SHAPES, self.entry_style, self, "entry style")
        return extract(entry)

    def entry_args(self, entry: dict) -> Optional[list]:
        """The arguments a registered entry launches with, for drift. ``None``
        (read as drift) when unreadable."""
        _, _, extract = _dispatch(_SHAPES, self.entry_style, self, "entry style")
        return extract(entry)


class JsonConfigClient(ClientBackend):
    """A client whose config is a calm JSON object we edit ourselves.

    Register/unregister are an atomic read-merge-replace preserving other keys.
    ``is_installed`` is the config *directory*; the file may not exist until
    first use.
    """

    def is_installed(self) -> bool:
        path = self.config_path()
        return path is not None and path.parent.is_dir()

    def register(self) -> None:
        path = self.config_path()
        if path is None:
            raise AgentError(
                f"{self.name} has no known config location on this platform"
            )
        if _jsonc_unmergeable(path):
            raise AgentError(self._manual_edit_message(path, removing=False))
        data = _load_json_object(path)
        parent = data.get(self.parent_key)
        if not isinstance(parent, dict):
            parent = {}
        parent["biopb"] = self.entry()
        data[self.parent_key] = parent
        _write_json_atomic(path, data)

    def unregister(self) -> None:
        path = self.config_path()
        if path is None or not path.exists():
            return  # nothing registered
        if _jsonc_unmergeable(path):
            # Can't safely rewrite a commented .jsonc: ask for manual removal
            # if biopb is present.
            if self.read_entry() is not None:
                raise AgentError(self._manual_edit_message(path, removing=True))
            return
        data = _load_json_object(path)
        parent = data.get(self.parent_key)
        if isinstance(parent, dict) and "biopb" in parent:
            del parent["biopb"]
            _write_json_atomic(path, data)

    def _manual_edit_message(self, path: Path, *, removing: bool) -> str:
        """The AgentError text shown when biopb can't safely edit a commented
        ``.jsonc``."""
        if removing:
            return (
                f"{path} has comments, so biopb won't rewrite it (that would drop "
                f'them). Remove the "biopb" key under "{self.parent_key}" by hand.'
            )
        snippet = json.dumps({self.parent_key: {"biopb": self.entry()}}, indent=2)
        return (
            f"{path} has comments, so biopb won't rewrite it (that would drop "
            f"them). Add this entry under the top level by hand:\n{snippet}"
        )


class CliManagedClient(ClientBackend):
    """A client that ships its own CLI for managing MCP servers.

    Writes shell out (see the module docstring); status is still a plain config
    read.
    """

    #: the binary to shell out to
    exe: str

    def is_installed(self) -> bool:
        """On PATH or not detected. No config-directory fallback: the binary is
        the write path, and a leftover directory (``~/.codex``) would offer a
        Register button that cannot work."""
        return shutil.which(self.exe) is not None


class ClaudeCode(CliManagedClient):
    id = "claude-code"
    name = "Claude Code"
    exe = "claude"
    # User-scope servers live in ~/.claude.json; read-only here.
    parent_key = "mcpServers"

    def config_path(self) -> Optional[Path]:
        return Path.home() / ".claude.json"

    def register(self) -> None:
        # Idempotent: best-effort remove, then add.
        _run_client_cli(
            self.exe, ["mcp", "remove", "biopb", "-s", "user"], required=False
        )
        code, out = _run_client_cli(
            self.exe,
            [
                "mcp",
                "add",
                "--scope",
                "user",
                "biopb",
                "--",
                _mcp_command(),
                *self.mcp_args,
            ],
            required=True,
        )
        if code != 0:
            raise AgentError(f"`claude mcp add` failed: {out.strip()}")

    def unregister(self) -> None:
        _run_client_cli(
            self.exe, ["mcp", "remove", "biopb", "-s", "user"], required=True
        )


class ClaudeDesktop(JsonConfigClient):
    id = "claude-desktop"
    name = "Claude Desktop"
    parent_key = "mcpServers"

    def config_path(self) -> Optional[Path]:
        home = Path.home()
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


class CodexCli(CliManagedClient):
    id = "codex-cli"
    name = "Codex CLI"
    exe = "codex"
    parent_key = "mcp_servers"
    config_format = "toml"
    # Codex does not refresh its tool list within a turn, so `--session auto`
    # binds the newest free session (else a new one) before the handshake. Codex
    # cannot choose its session.
    mcp_args = (*_MCP_ARGS, "--session", "auto")

    def config_path(self) -> Optional[Path]:
        # $CODEX_HOME relocates the Codex home; read at call time for tests.
        base = os.environ.get("CODEX_HOME")
        return (Path(base) if base else Path.home() / ".codex") / "config.toml"

    def register(self) -> None:
        # `codex mcp add` overwrites an existing entry, so no remove first.
        code, out = _run_client_cli(
            self.exe,
            ["mcp", "add", "biopb", "--", _mcp_command(), *self.mcp_args],
            required=True,
        )
        if code != 0:
            raise AgentError(f"`codex mcp add` failed: {out.strip()}")

    def unregister(self) -> None:
        # codex exits 0 removing an absent server, so this is idempotent.
        code, out = _run_client_cli(self.exe, ["mcp", "remove", "biopb"], required=True)
        if code != 0:
            raise AgentError(f"`codex mcp remove` failed: {out.strip()}")


class Cursor(JsonConfigClient):
    id = "cursor"
    name = "Cursor"
    parent_key = "mcpServers"

    def config_path(self) -> Optional[Path]:
        return Path.home() / ".cursor" / "mcp.json"


class Opencode(JsonConfigClient):
    id = "opencode"
    name = "opencode"
    parent_key = "mcp"
    entry_style = "opencode"

    def config_path(self) -> Optional[Path]:
        """An existing ``opencode.jsonc`` (so no shadow ``.json`` is written
        beside it), else ``opencode.json``."""
        base = Path.home() / ".config" / "opencode"
        jsonc = base / "opencode.jsonc"
        return jsonc if jsonc.exists() else base / "opencode.json"

    def is_installed(self) -> bool:
        return shutil.which("opencode") is not None or super().is_installed()


# --------------------------------------------------------------------------- #
# The catalog
# --------------------------------------------------------------------------- #
# Hermes is omitted: its YAML config is never edited, so it could not reach
# `registered` through a button.
_CLIENTS: tuple[ClientBackend, ...] = (
    ClaudeCode(),
    ClaudeDesktop(),
    CodexCli(),
    Cursor(),
    Opencode(),
)

_CLIENTS_BY_ID = {c.id: c for c in _CLIENTS}


def _client(client_id: str) -> ClientBackend:
    try:
        return _CLIENTS_BY_ID[client_id]
    except KeyError:
        raise AgentError(f"unknown agent client {client_id!r}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def supported() -> list[ClientBackend]:
    """The static client catalog."""
    return list(_CLIENTS)


def status(client_id: str) -> dict:
    """One client's status: ``{id, name, state, drifted, config_path}``.

    ``state`` is ``registered`` if the biopb entry is present (regardless of
    detection), else ``installed`` if detected, else ``not_installed``.
    ``drifted`` is set only when ``registered`` and the stored command or
    arguments no longer match what would be registered now.
    """
    client = _client(client_id)
    path = client.config_path()
    entry = client.read_entry()
    if entry is not None:
        state = "registered"
        drifted = client.entry_command(entry) != _mcp_command() or client.entry_args(
            entry
        ) != list(client.mcp_args)
    elif client.is_installed():
        state, drifted = "installed", False
    else:
        state, drifted = "not_installed", False
    return {
        "id": client.id,
        "name": client.name,
        "state": state,
        "drifted": drifted,
        "config_path": str(path) if path is not None else None,
    }


def statuses() -> list[dict]:
    """Status for every supported client, in catalog order."""
    return [status(c.id) for c in _CLIENTS]


def register(client_id: str) -> dict:
    """Register biopb with the client and return its fresh status.

    Works regardless of detection; an absent client raises :class:`AgentError`
    (e.g. no ``claude`` on PATH).
    """
    client = _client(client_id)
    client.register()
    logger.info("registered biopb with %s", client.name)
    return status(client_id)


def unregister(client_id: str) -> dict:
    """Remove biopb from the client and return its fresh status. Idempotent."""
    client = _client(client_id)
    client.unregister()
    logger.info("unregistered biopb from %s", client.name)
    return status(client_id)
