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

Each client is a :class:`ClientBackend` — one object owning its config location,
its install signal, and its read/write path. Adding a client is a class plus a
line in ``_CLIENTS``, with nothing to remember elsewhere. Two rules keep a
half-implemented one from reaching a user: the format and entry-shape tables
(``_READERS``, ``_SHAPES``) have no default branch, and the write path is
abstract, so an omission is a ``TypeError`` at import rather than a config
written in a format it is not.

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
  serialize its own writes rather than race it with our merge. Codex CLI goes
  through ``codex mcp add`` for a different reason: its config is TOML, and only
  Codex's own editor keeps the user's comments and sibling servers intact — we
  have no TOML writer and want none (see :func:`_read_toml_entry`).
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

try:  # tomllib is 3.11+; on 3.10 (our floor) _scan_toml_entry stands in
    import tomllib
except ImportError:  # pragma: no cover - only reached on 3.10
    tomllib = None  # type: ignore[assignment]

from abc import ABC, abstractmethod
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


def _read_toml_entry(path: Path, parent_key: str) -> Optional[dict]:
    """The biopb table from Codex's ``config.toml``, shaped like a JSON stdio
    entry (``{command, args}``) so :func:`_entry_command` and :func:`status` need
    no TOML-specific branch.

    Read-only on purpose. Writes go through ``codex mcp add``/``remove``, which
    edit the table surgically and leave the user's comments and sibling servers
    alone; a re-emit from a parsed dict would drop the comments, the same trap
    ``.jsonc`` set for opencode (biopb/biopb#536). ``None`` (never raises) on any
    problem, like :func:`_load_json_tolerant`: a config we cannot parse simply
    reads as "not registered".
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

    Pulls just ``command`` — the one value status and drift need — out of the
    ``[<parent_key>.biopb]`` table, stopping at the next table header.
    Deliberately narrow: it reads the shape ``codex mcp add`` writes (one
    quoted string per line) and gives up on anything else, which reads as "not
    registered" like every other config we cannot parse.
    """
    header = re.compile(r"^\s*\[\s*" + re.escape(parent_key) + r"\s*\.\s*biopb\s*\]")
    in_table = False
    for line in text.splitlines():
        if line.lstrip().startswith("["):
            if in_table:
                break  # the next table ends ours
            in_table = bool(header.match(line))
            continue
        if not in_table:
            continue
        key, sep, raw = line.partition("=")
        if sep and key.strip() == "command":
            value = _toml_string(raw.strip())
            return None if value is None else {"command": value}
    return None


def _toml_string(raw: str) -> Optional[str]:
    """A single-line TOML basic (``"..."``, escapes decoded) or literal
    (``'...'``, verbatim) string, or ``None`` if ``raw`` is neither. A trailing
    comment is not stripped — it makes the value unparseable, which fails safe
    to "not registered" rather than guessing where a ``#`` inside a path ends."""
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


#: config_format -> reader. Every read goes through this; there is deliberately
#: no default, so a client whose format we have not implemented raises instead
#: of being silently parsed as JSON.
_READERS = {
    "json": _read_json_entry,
    "toml": _read_toml_entry,
}


# --------------------------------------------------------------------------- #
# Entry shapes
# --------------------------------------------------------------------------- #
# Each style pairs a builder (what to write) with an extractor (how to read the
# executable back out for drift). They must stay inverses of each other, which
# is why they are defined as a pair rather than in the read and write halves.


def _stdio_entry(command: str) -> dict:
    # Canonical mcpServers stdio form: bare command+args, no "type" (a stray
    # "type" trips stricter validators — matches the installer's choice).
    return {"command": command, "args": list(_MCP_ARGS)}


def _stdio_command(entry: dict) -> Optional[str]:
    command = entry.get("command")
    return command if isinstance(command, str) else None


def _opencode_entry(command: str) -> dict:
    return {"type": "local", "command": [command, *_MCP_ARGS], "enabled": True}


def _opencode_command(entry: dict) -> Optional[str]:
    command = entry.get("command")
    if isinstance(command, list) and command:
        return command[0] if isinstance(command[0], str) else None
    return None


#: entry_style -> (builder, extractor). No default branch, for the same reason
#: as _READERS: an unknown style must not silently get the stdio shape.
_SHAPES = {
    "stdio": (_stdio_entry, _stdio_command),
    "opencode": (_opencode_entry, _opencode_command),
}


def _dispatch(table: dict, key: str, client: ClientBackend, axis: str):
    """Look ``key`` up in ``table``, or raise :class:`AgentError` naming the
    client and the axis. The single place a missing implementation is caught —
    every dispatch in this module goes through it rather than an ``else``."""
    try:
        return table[key]
    except KeyError:
        raise AgentError(f"{client.name}: unsupported {axis} {key!r}")


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


# --------------------------------------------------------------------------- #
# Writing helpers
# --------------------------------------------------------------------------- #


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


def _run_client_cli(
    exe_name: str, args: list[str], *, required: bool
) -> tuple[int, str]:
    """Run ``<exe_name> <args>`` windowless, returning ``(returncode, output)``.

    Shared by the two CLI-managed clients (``claude``, ``codex``). ``required``
    distinguishes a call that must succeed (``True`` → a missing binary is an
    :class:`AgentError`) from a best-effort one, like the ``mcp remove`` Claude
    Code runs before an add to stay idempotent (``False`` → tolerate a non-zero
    code, i.e. "wasn't registered"). We never call either client's ``mcp
    get``/``list`` — those run a live connection test that would spawn
    ``biopb-mcp``.
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
# One object per client, holding everything biopb knows about it: where its
# config lives, whether it looks installed, and how to read and write biopb's
# entry. Adding a client is a new class plus a line in _CLIENTS -- there is no
# second place to remember, which is what the per-client `if spec.id == ...`
# chains this replaced kept getting wrong (a client added to some of them and
# missed in others took the fall-through, i.e. JSON).
#
# The write path is abstract, so a backend that omits it cannot be instantiated
# at all: the failure is a TypeError at import, not a config file written in a
# format it is not.


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

    @abstractmethod
    def config_path(self) -> Optional[Path]:
        """The config file biopb's entry lives in, or ``None`` on a platform
        where this client has no known location. Resolved at call time (not
        cached) so a test that repoints ``Path.home()`` / ``$APPDATA`` gets an
        isolated location."""

    @abstractmethod
    def is_installed(self) -> bool:
        """Whether the client appears present -- the same signals the installer
        uses. Deliberately cheap and subprocess-free: a binary on PATH or a
        well-known config directory, per the policy each base sets.

        A false negative (a portable/flatpak install we can't see) just shows
        ``not_installed``. For a :class:`JsonConfigClient` register is still an
        escape hatch that works anyway; for a :class:`CliManagedClient` it is
        not, which is why that base detects the binary and nothing else."""

    @abstractmethod
    def register(self) -> None:
        """Write biopb's entry into this client's config."""

    @abstractmethod
    def unregister(self) -> None:
        """Remove it. Idempotent -- removing an absent entry is fine."""

    # -- shared, driven by config_format / entry_style ---------------------- #

    def read_entry(self) -> Optional[dict]:
        """The biopb entry currently in this client's config, or ``None``.

        Tolerant of the *user's* data: a malformed config simply reads as "not
        registered" for display, and the write path reports the parse error.
        Raises on a malformed *catalog* -- a ``config_format`` with no reader --
        because that is our bug, and the alternative is parsing the file as
        something it is not.
        """
        path = self.config_path()
        if path is None or not path.exists():
            return None
        read = _dispatch(_READERS, self.config_format, self, "config format")
        return read(path, self.parent_key)

    def entry(self) -> dict:
        """The MCP server entry to write, in this client's shape."""
        build, _ = _dispatch(_SHAPES, self.entry_style, self, "entry style")
        return build(_mcp_command())

    def entry_command(self, entry: dict) -> Optional[str]:
        """The executable a registered entry points at, for drift. ``None`` when
        the entry has no recognizable command (treated as drift, so a malformed
        prior entry prompts a Re-register)."""
        _, extract = _dispatch(_SHAPES, self.entry_style, self, "entry style")
        return extract(entry)


class JsonConfigClient(ClientBackend):
    """A client whose config is a calm JSON object we edit ourselves.

    Register/unregister are an atomic read-merge-replace that preserves every
    other key in the file. ``is_installed`` is the app's config *directory*:
    these are apps that own one, and the config file itself may not exist until
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
            # Can't safely rewrite a commented .jsonc. If biopb is actually
            # present, surface a manual-removal instruction rather than silently
            # leaving a stale entry (biopb/biopb#536); else there is nothing to do.
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
        ``.jsonc`` -- a clear manual instruction instead of clobbering comments
        or writing a shadow config (biopb/biopb#536)."""
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

    We shell out rather than edit the file, for two different reasons: Claude
    Code rewrites ``~/.claude.json`` constantly and we would race its writes,
    and Codex's config is TOML whose comments and sibling servers only its own
    editor keeps intact. Status is still a plain config read -- never
    ``mcp get``/``list``, which run a live connection test that would spawn
    ``biopb-mcp`` on every dashboard poll.
    """

    #: the binary to shell out to
    exe: str

    def is_installed(self) -> bool:
        """On PATH or not detected — deliberately no config-directory fallback.

        For these clients the binary *is* the write path, so PATH presence is a
        precondition rather than a hint: without it ``register`` can only raise.
        A leftover config directory (``~/.codex`` keeps auth, history and logs
        long after the binary is uninstalled) would otherwise report
        ``installed`` forever and offer a Register button that cannot work.
        A client that is registered still reads as ``registered`` either way —
        :func:`status` takes the entry as ground truth before asking us.
        """
        return shutil.which(self.exe) is not None


class ClaudeCode(CliManagedClient):
    id = "claude-code"
    name = "Claude Code"
    exe = "claude"
    # Claude Code stores user-scope MCP servers in ~/.claude.json under the
    # top-level `mcpServers` key. We only READ that for status.
    parent_key = "mcpServers"

    def config_path(self) -> Optional[Path]:
        return Path.home() / ".claude.json"

    def register(self) -> None:
        # Idempotent: drop any existing entry, then add (matches the installer).
        # The remove is best-effort (a not-yet-registered client returns
        # non-zero); the add must succeed.
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
                *_MCP_ARGS,
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

    def config_path(self) -> Optional[Path]:
        # $CODEX_HOME relocates the whole Codex home (config.toml included);
        # read at call time so a test can point it at a tmp dir.
        base = os.environ.get("CODEX_HOME")
        return (Path(base) if base else Path.home() / ".codex") / "config.toml"

    def register(self) -> None:
        # `codex mcp add` overwrites an existing server of the same name and
        # exits 0, so unlike Claude Code no remove-then-add dance is needed.
        code, out = _run_client_cli(
            self.exe,
            ["mcp", "add", "biopb", "--", _mcp_command(), *_MCP_ARGS],
            required=True,
        )
        if code != 0:
            raise AgentError(f"`codex mcp add` failed: {out.strip()}")

    def unregister(self) -> None:
        # Removing an absent server is not an error for codex (it exits 0), so
        # this is idempotent without tolerating a failure that is real.
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
        """The opencode global config biopb should target (biopb/biopb#536).

        opencode reads either ``opencode.jsonc`` or ``opencode.json``. Prefer an
        existing ``.jsonc`` so we edit the file opencode actually honors instead
        of writing a shadow ``.json`` it may ignore; otherwise fall back to
        ``.json`` -- the canonical file we create on a fresh install."""
        base = Path.home() / ".config" / "opencode"
        jsonc = base / "opencode.jsonc"
        return jsonc if jsonc.exists() else base / "opencode.json"

    def is_installed(self) -> bool:
        return shutil.which("opencode") is not None or super().is_installed()


# --------------------------------------------------------------------------- #
# The catalog
# --------------------------------------------------------------------------- #
# Consistent with the installer (install/install.sh, install/biopb-engine.ps1).
# Hermes is intentionally omitted: the installer only ever prints a manual YAML
# snippet for it (it will not edit YAML), so it can never reach `registered`
# through a button -- not worth a dead row.
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
    detection -- the entry is ground truth), else ``installed`` if the client is
    detected, else ``not_installed``. ``drifted`` is set only when ``registered``
    and the stored command no longer matches the freshly resolved ``biopb-mcp``
    path (a moved/reinstalled biopb), so the UI can offer a Re-register.
    """
    client = _client(client_id)
    path = client.config_path()
    entry = client.read_entry()
    if entry is not None:
        state = "registered"
        drifted = client.entry_command(entry) != _mcp_command()
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

    Works regardless of detection (the "register anyway" escape hatch for a
    client we could not auto-detect); a genuinely absent client surfaces as an
    :class:`AgentError` (e.g. Claude Code with no ``claude`` on PATH).
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
