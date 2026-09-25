"""Configuration management for biopb-mcp.

The config is defined as **flat dataclasses** (one per section) with each field's
help in ``field(metadata={"help": ...})`` and its validation rule in the
class-keyed :data:`_CONSTRAINTS` table -- the same machinery the tensor server
uses, so both project a JSON Schema from one source of truth (biopb/biopb#34) and
a schema-driven admin editor can render either. ``DEFAULT_CONFIG`` is just
``asdict(McpConfig())``, so the dataclass defaults are the only place a default
literal lives.

Config lives at ``~/.config/biopb/mcp-config.json`` -- co-located with the
tensor server's ``biopb.json`` and the installer's client-definition ``mcp.json``
(a *distinct* file: that one registers biopb-mcp with MCP clients; this one is
biopb-mcp's own runtime settings). Logs -- runtime state, not config -- live in
the shared biopb XDG *state* tree (``~/.local/state/biopb/mcp``), resolved via
:mod:`biopb._locations` (no more separate top-level ``biopb-mcp`` dir).

Sections are flat (no ``mcp.``/``widget.`` wrapper): ``transport`` / ``kernel`` /
``viewer`` / ``services`` / ``observe`` / ``update`` are
the MCP-server knobs; ``timeout`` / ``grpc`` / ``memory`` are compute-plane knobs
read by ``ops``. The napari widgets keep their own settings (biopb-napari-widget).

There is deliberately **no data-plane endpoint here** (biopb/biopb#628): the
control plane owns the data plane and is asked for its address at connect time,
so a configured URL could only be a second, staler answer -- and was how this
machine's credential reached endpoints the control never named (#626). A section the
schema does not know is carried by the merge and read by nothing.

Read settings with :func:`get_setting`, which falls back to ``DEFAULT_CONFIG`` so
call sites never duplicate a default literal.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Shared with the tensor server: the constraint primitives (so a knob is judged
# by the same rules in both packages; biopb/biopb#182, #34) and the config-file
# location.
from biopb import _locations
from biopb._config_constraints import Enum, Range
from biopb._config_io import atomic_write_json
from biopb._config_validate import MISSING, Problem, check_sections, warn_and_clamp
from biopb._locations import mcp_config_path

logger = logging.getLogger(__name__)

# Windows widens the kernel's startup budget: it has no fork(), so the kernel
# and everything it imports start cold. A user config still overrides it.
_IS_WINDOWS = os.name == "nt"
_DEFAULT_STARTUP_TIMEOUT = 120.0 if _IS_WINDOWS else 60.0


def _h(default, help_text, **kw):
    """A dataclass field carrying user-facing help (-> schema description)."""
    return field(default=default, metadata={"help": help_text}, **kw)


def _hlist(default_list, help_text):
    """A list-valued field with help (defaults are copied per instance)."""
    return field(
        default_factory=lambda: list(default_list), metadata={"help": help_text}
    )


# --- Section dataclasses ------------------------------------------------------
# Each maps 1:1 to a top-level section in mcp-config.json; every field's `help`
# becomes its schema `description` (the single source of truth). List fields are
# added to the schema by the composer in _config_schema.py.


@dataclass
class TimeoutConfig:
    """Per-call gRPC timeouts (seconds) for the compute plane."""

    health_check: float = _h(5.0, "Timeout for a server health check.")
    get_op_names: float = _h(10.0, "Timeout for listing a server's op names.")
    detection_2d: int = _h(15, "Timeout for a 2D detection call.")
    detection_3d: int = _h(300, "Timeout for a 3D detection call.")
    process_image: int = _h(300, "Timeout for a ProcessImage call.")


@dataclass
class GrpcConfig:
    """gRPC channel limits for the compute plane."""

    max_message_size_mb: int = _h(512, "Max gRPC message size (MB) for send/receive.")
    max_concurrent_calls: int = _h(4, "Max concurrent gRPC calls to a server.")


@dataclass
class MemoryConfig:
    """Chunk-size guardrails for eager transfers."""

    warn_threshold_mb: int = _h(
        500, "Log a warning when a single chunk exceeds this size (MB)."
    )
    error_threshold_mb: int = _h(
        2000, "Raise MemoryError when a single chunk exceeds this size (MB)."
    )


@dataclass
class TransportConfig:
    """The MCP server's front-end transport and its network guards."""

    kind: str = _h(
        "stdio",
        'Front-end transport: "http" (loopback streamable-http on `port`) or '
        '"stdio" (the client spawns biopb-mcp; the shim owns a private http '
        "session child on a dynamic port and bridges stdin/stdout to it).",
    )
    port: int = _h(
        8765,
        "Fixed loopback port for the http server. Applies only to a directly-"
        "launched `--transport http` server (the stdio shim and `biopb mcp view` "
        "use dynamic ports).",
    )
    kernel_log: str = _h(
        "",
        "Force the stdio bridge's session child to log to ONE fixed file instead "
        "of the default per-session file. Empty -> each session gets its own log.",
    )
    session_log_keep: int = _h(
        5,
        "How many per-session shim logs to keep (newest by mtime); older ones are "
        "pruned on each new session. Ignored when kernel_log forces a shared file.",
    )
    allowed_origins: List[str] = _hlist(
        [],
        "Extra Origin header values appended to the loopback allowlist guarding "
        "against DNS-rebinding / cross-origin browser requests. http transport only.",
    )
    allowed_hosts: List[str] = _hlist(
        [],
        "Extra Host header values appended to the loopback allowlist. Set only when "
        "fronting the server with a reverse proxy. http transport only.",
    )


@dataclass
class KernelConfig:
    """The child Jupyter kernel that runs agent code (separate process)."""

    name: str = _h("python3", "Jupyter kernel name to launch.")
    startup_timeout: float = _h(
        _DEFAULT_STARTUP_TIMEOUT,
        "Seconds to wait for kernel bring-up. 60 on POSIX; 120 on Windows, where "
        "a cold interpreter start makes bring-up legitimately slower.",
    )
    execute_timeout: float = _h(
        120.0,
        "Bounds only the quick in-band kernel snippets (screenshot / status / "
        "inspect / job submit+poll), not long jobs -- execute_code runs agent code "
        "in a background thread that may run indefinitely.",
    )
    promote_after: float = _h(
        10.0,
        "Seconds execute_code waits before promoting a job: completes within this "
        "window -> inline result; otherwise a job handle is returned and it runs on.",
    )
    parent_death_pipe: bool = _h(
        True,
        "Kernel inherits a pipe read-end and group-kills itself when the launcher "
        "process dies (POSIX only; orphan hardening, #13).",
    )
    watchdog_interval: float = _h(
        5.0,
        "Seconds between kernel liveness polls; on an unexpected death the host "
        "reaps the orphaned process group and respawns. 0 disables the watchdog.",
    )
    watchdog_max_respawns: int = _h(
        3, "Max respawns within the window before the host is marked dead."
    )
    watchdog_respawn_window: float = _h(
        60.0, "Sliding window (seconds) over which respawns are counted."
    )


@dataclass
class ViewerConfig:
    """napari viewer slice-read behavior in the kernel."""

    compute_scheduler: str = _h(
        "threads",
        "Scheduler for the viewer's serial plane reads. Pinning to a single-process "
        'scheduler ("threads"/"synchronous") uses the one shared client cache (~100% '
        'hit on revisit, no worker scatter; #8); "" computes on the global default.',
    )
    async_slicing: bool = _h(
        True,
        "Fetch napari slices off the Qt main thread (async slicing), so a cold "
        "tile read doesn't freeze the viewer. take_screenshot force-syncs before "
        "capturing. False keeps fully-synchronous slicing.",
    )


@dataclass
class ServicesConfig:
    """Compute-plane servers and the knowledge store wired into the kernel."""

    process_image_servers: List[str] = _hlist(
        [],
        "biopb.image ProcessImage servicer URLs (grpc:// or grpcs://). Each is "
        "queried via GetOpNames and exposed as callables in the kernel's `ops` dict.",
    )
    docs_local_dir: str = _h(
        "",
        "Directory of the agent's own docs (*.md), written by write_doc and "
        "shadowing a shipped doc of the same id; empty -> ~/.config/biopb/docs. "
        "Holds the index the agent edits, and is re-read on every access so a "
        "hand edit is live without a restart.",
    )
    namespace_enabled: bool = _h(
        True,
        "Load user 'bring your own tool' plugins into the agent kernel namespace at "
        "start: *.py files in ~/.config/biopb/kernel/ and installed "
        "biopb_mcp.namespace packages (biopb/biopb-mcp#92). Off -> a clean "
        "built-in-only namespace.",
    )


@dataclass
class ObserveConfig:
    """Minimal loopback web UI for watching execute_code job history (http only)."""

    enabled: bool = _h(
        True,
        "Enable the observe UI. http transport only (it shares the MCP port and "
        "event loop, adding no network surface); silently skipped under stdio.",
    )
    max_output_chars: int = _h(
        20000,
        "Detail-view stdout cap (chars); the tail is kept with a truncation marker "
        "and the full length is reported alongside.",
    )
    poll_interval_ms: int = _h(
        3000,
        "How often (ms) the observe page polls the job list/status. Deliberately "
        "slow: each poll is a kernel round-trip competing with agent calls.",
    )
    chat_enabled: bool = _h(
        True,
        "Offer the built-in chat client: a pane on the observe page that drives "
        "this session's kernel through a model. Lives here because it needs the "
        "page: chat routes served without one have nothing to reach them. On by "
        "default, but inert until the model and key in `chat` are set, so it "
        "costs nothing to leave on. This can only narrow the surface -- the "
        "control refuses to proxy chat at all unless it is loopback-bound.",
    )


@dataclass
class ChatConfig:
    """Which agent drives the chat pane, and how to reach it.

    Two engines. ``builtin`` is the in-process loop (``mcp/_chat.py``) talking to
    an OpenAI-compatible endpoint: ``model`` / ``base_url`` / ``api_key_env`` /
    ``request_timeout`` describe it. ``acp`` hands the pane to a coding harness
    the user already runs, over the Agent Client Protocol: the ``acp_*`` settings
    describe that one. Nothing is shared between the two but the pane.

    The on/off switch is **not** here: it is ``observe.chat_enabled``, because
    what it turns on is a pane on the observe page. This
    section is only *which* agent that pane talks to, so there is one place to
    enable a surface and one place to point it somewhere.

    The provider key is deliberately not here either. This file is served whole
    by the control's ``GET /api/mcp_config`` so the admin page can edit it, and a
    key in it would be rendered in a browser; it lives in an owner-only
    credential file instead (``biopb._credentials``, name ``chat-provider.token``)
    for the same reasons that module was written. What is here is configuration
    a person may reasonably want to change and no one needs to keep secret.
    """

    engine: str = _h(
        "builtin",
        "Which agent drives the pane: 'builtin' (the in-process loop, needs a "
        "model and a provider key) or 'acp' (a coding harness you already have, "
        "which brings its own model and its own subscription).",
    )
    model: str = _h(
        "",
        "Model id to send, e.g. 'gpt-4o' or 'deepseek-v4'. Empty means chat is "
        "unconfigured, which reports as such rather than guessing a default that "
        "would bill the user for a model they did not choose.",
    )
    base_url: str = _h(
        "https://api.openai.com/v1",
        "OpenAI-compatible API root. Any gateway speaking that shape works; the "
        "route ('/chat/completions' or '/responses', per chat.api) is appended, "
        "so this is the root and not the endpoint itself.",
    )
    api: str = _h(
        "completions",
        "Which API shape chat.model speaks: 'completions' (POST "
        "{base_url}/chat/completions) or 'responses' (POST {base_url}/responses). "
        "One gateway can serve both and disagree per model, and GET /models does "
        "not say which, so it is configured rather than probed -- a probe costs a "
        "billed call and reads a wrong-route 500 as an outage.",
    )
    api_key_env: str = _h(
        "BIOPB_CHAT_API_KEY",
        "Environment variable consulted *before* the credential file. For CI and "
        "development; the file is the supported path, because an env var leaks "
        "through /proc, `ps e`, and every inherited child.",
    )
    request_timeout: float = _h(
        120.0,
        "Seconds to wait for one model reply. Covers a slow first token on a "
        "long conversation, not the tool calls it triggers.",
    )
    extra_headers: list = _hlist(
        [],
        "Extra headers to send with every request, as 'Name: value'. For a "
        "gateway that requires something the OpenAI API does not define -- a "
        "session id, an attribution tag. '{session}' becomes this "
        "conversation's id. Known gateways are configured automatically; these "
        "merge over that by name, and a name with an empty value removes one. "
        "These values are returned by /api/mcp_config and must not contain "
        "secrets; keep API keys and authorization tokens in the credential "
        "file instead.",
    )
    vision: str = _h(
        "auto",
        "Whether screenshots are sent to the model: 'auto' sends them until the "
        "provider refuses one and then stops, 'on' always sends, 'off' never "
        "sends and does not offer take_screenshot. A model without vision does "
        "not merely fail the screenshot -- the image is stored and re-sent, so "
        "every later turn fails too, which is what 'auto' recovers from.",
    )
    acp_agent: str = _h(
        "opencode",
        "Which ACP harness to run when engine is 'acp'. Only 'opencode' is "
        "supported: it is the one that ships an ACP mode natively and honours "
        "the MCP server handed to it in the session handshake.",
    )
    acp_command: str = _h(
        "",
        "Absolute path to the harness binary, overriding the usual lookup. For "
        "an install PATH does not reach; empty means resolve 'opencode' the "
        "normal way.",
    )
    acp_model: str = _h(
        "",
        "Model the harness should use, in its own spelling (opencode: "
        "'openai/gpt-5.5'). Empty takes whatever the harness defaults to — "
        "which is a model you did not choose, on a provider that may not even "
        "be reachable. Ignored by a harness that exposes no model setting.",
    )
    acp_permission: str = _h(
        "ask",
        "What to do when the harness asks permission to run something: 'ask' "
        "puts the request in the pane, 'allow' answers yes for you. A harness "
        "brings its own file and shell tools, so 'allow' is unattended access "
        "to this machine, not just to the viewer.",
    )


@dataclass
class UpdateConfig:
    """Kernel-start auto-updater (#87): offers to re-run the installer on a newer release."""

    enabled: bool = _h(
        True, "Enable the background, fail-open update check at kernel start."
    )
    repo: str = _h(
        "biopb/biopb",
        "owner/name of the deployment repo whose release-v* line is the product "
        "(overridable for forks/testing).",
    )
    channel: str = _h(
        "stable",
        '"stable" -> newest clean release-vX.Y.Z (prereleases skipped); '
        '"prerelease" -> also consider release candidates.',
    )
    skipped_version: str = _h(
        "",
        'A version the user chose to skip ("Skip vX.Y.Z"); suppresses the prompt for exactly that version.',
    )
    timeout: float = _h(
        5.0,
        "Per-request network timeout (s) for the GitHub API fetch; short so the check never delays viewer start.",
    )


def _section(cls, title=None, summary=None):
    """A top-level section. *title* and *summary* are its settings-page nav
    label and panel prose (schema ``title`` / ``description``); a section with
    no title is left off the page, still editable as raw JSON."""
    return field(default_factory=cls, metadata={"title": title, "summary": summary})


@dataclass
class McpConfig:
    """The whole biopb-mcp config: one field per top-level section, in the
    settings page's nav order."""

    services: ServicesConfig = _section(
        ServicesConfig,
        "Services",
        "ProcessImage algorithm servers wired into the kernel as `ops`, and the "
        "knowledge store.",
    )
    timeout: TimeoutConfig = _section(
        TimeoutConfig, "Timeouts", "Per-call gRPC timeouts for the compute plane."
    )
    grpc: GrpcConfig = _section(
        GrpcConfig, "gRPC", "gRPC channel limits for the compute plane."
    )
    memory: MemoryConfig = _section(
        MemoryConfig, "Memory", "Chunk-size guardrails for eager transfers."
    )
    transport: TransportConfig = _section(
        TransportConfig,
        "Transport",
        "The MCP server's front-end transport (stdio / http) and its network guards.",
    )
    kernel: KernelConfig = _section(
        KernelConfig,
        "Kernel",
        "The child Jupyter kernel that runs agent code: bring-up, timeouts, and "
        "the orphan watchdog.",
    )
    viewer: ViewerConfig = _section(
        ViewerConfig, "Viewer", "How the napari viewer fetches image slices."
    )
    observe: ObserveConfig = _section(
        ObserveConfig,
        "Observe",
        "The loopback web UI for watching execute_code job history (http "
        "transport only).",
    )
    chat: ChatConfig = _section(
        ChatConfig,
        "Chat",
        "Which model the built-in chat pane talks to. The on/off switch is on "
        "the Observe page (chat_enabled); the provider key is not here, by "
        "design — this file is served to the browser, so the key lives in an "
        "owner-only credential file.",
    )
    update: UpdateConfig = _section(
        UpdateConfig,
        "Updates",
        "The kernel-start auto-updater that offers to re-run the installer on a "
        "newer release.",
    )


# Section name -> its dataclass, derived from McpConfig so the two never drift.
# Used by validation and by the schema composer (_config_schema.py).
_SECTION_CLASSES = {
    f.name: type(getattr(McpConfig(), f.name)) for f in dataclasses.fields(McpConfig)
}


# Per-class validation rules (biopb/biopb#182). Keyed by class name like the
# tensor server's table, judged by the same shared Range/Enum primitives.
_CONSTRAINTS = {
    "TimeoutConfig": {
        "health_check": Range(exclusive_min=0),
        "get_op_names": Range(exclusive_min=0),
        "detection_2d": Range(exclusive_min=0),
        "detection_3d": Range(exclusive_min=0),
        "process_image": Range(exclusive_min=0),
    },
    "GrpcConfig": {
        "max_message_size_mb": Range(exclusive_min=0),
        "max_concurrent_calls": Range(min=1),
    },
    "MemoryConfig": {
        "warn_threshold_mb": Range(exclusive_min=0),
        "error_threshold_mb": Range(exclusive_min=0),
    },
    "ChatConfig": {
        "request_timeout": Range(exclusive_min=0),
        "engine": Enum({"builtin", "acp"}),
        "api": Enum({"completions", "responses"}),
        "acp_agent": Enum({"opencode"}),
        "acp_permission": Enum({"ask", "allow"}),
        "vision": Enum({"auto", "on", "off"}),
    },
    "TransportConfig": {
        "kind": Enum({"http", "stdio"}),
        "port": Range(min=1, max=65535),
        "session_log_keep": Range(min=1),  # keep at least the current
    },
    "KernelConfig": {
        "startup_timeout": Range(exclusive_min=0),
        "execute_timeout": Range(exclusive_min=0),
        "promote_after": Range(exclusive_min=0),
        "watchdog_interval": Range(min=0),  # 0 disables the watchdog
        "watchdog_max_respawns": Range(min=0),
        "watchdog_respawn_window": Range(min=0),
    },
}


# The default config as a plain nested dict (the runtime form). asdict() is the
# single place defaults come from -- there is no separate default literal.
DEFAULT_CONFIG = dataclasses.asdict(McpConfig())


def get_default_config() -> dict:
    """Return a deep copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


# The shared "key is not present" sentinel, not a second local one: `_walk_path`
# is handed to the shared clamping policy as its default lookup, which tests the
# result against this exact object -- two sentinels would make "no default" look
# like a value and write the sentinel into the config.
_MISSING = MISSING


def get_setting(config: dict, path: str, default=_MISSING):
    """Read an absolute dotted *path* from *config*, else ``DEFAULT_CONFIG``.

    ``get_setting(config, "kernel.promote_after")`` walks *config* by the dotted path;
    on a miss at any level it falls back to *default* if given, else to the value
    at the same path in ``DEFAULT_CONFIG``. Mutable defaults are deep-copied so
    callers cannot alias the shared ``DEFAULT_CONFIG``. Centralizing the fallback
    keeps each default declared exactly once -- call sites never restate a literal.

    Raises:
        KeyError: the path is absent from both *config* and ``DEFAULT_CONFIG`` and
            no *default* was given (a programmer error, not a config issue).
    """
    node = config
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            break
    else:
        return node

    if default is not _MISSING:
        return default

    node = DEFAULT_CONFIG
    for key in path.split("."):
        node = node[key]
    return copy.deepcopy(node)


def get_config_path() -> Path:
    """Path to the config file (``~/.config/biopb/mcp-config.json``).

    Delegates to :func:`biopb._locations.mcp_config_path` so biopb-mcp and
    the core-``biopb`` readers (control plane / ``_algorithms``) share one location.
    """
    return mcp_config_path()


def get_log_dir() -> Path:
    """Log directory (``~/.local/state/biopb/mcp`` on all platforms).

    Logs are persistent runtime state, not user-editable config, so they live in
    the shared biopb *state* tree (``$BIOPB_STATE_HOME``), beside the tensor
    server's ``logs/`` and the session registry. Delegates to
    :func:`biopb._locations.mcp_log_dir`, which creates it on access.
    """
    return _locations.mcp_log_dir()


def get_workflow_dir() -> Path:
    """Directory for spooled workflow notebooks (``<log dir>/workflows``).

    Every verification that passes is written here, because the child cannot
    know which passing run has the *right* numbers -- only that every cell ran.
    State rather than data: this is a spool the session fills on its own, and
    the copy a person keeps is the one they download or ask an agent to place.
    Retention (:data:`biopb_mcp.mcp._scratch._SPOOL_KEEP`) prunes it to the
    newest N, as the per-session logs beside it are pruned.
    """
    d = get_log_dir() / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_session_log_dir() -> Path:
    """Directory for per-session stdio-shim logs (``<log dir>/sessions``).

    Each shim-owned session writes its own logfile here rather than the shared
    ``mcp-server.log``, so concurrent sessions never interleave; retention
    (``transport.session_log_keep``) prunes it to the newest N. Composed from the
    local :func:`get_log_dir` so it tracks any override of that seam.
    """
    d = get_log_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_daemon_log_file(config: Optional[dict] = None) -> Path:
    """Path of the canonical combined stdout/stderr log for a direct
    ``--transport http`` server whose output is redirected to a file.

    A *single* canonical location so any such launcher and every reader agree on
    one file. Honors ``transport.kernel_log`` if set, else the shared
    ``mcp-server.log``. ``config`` is loaded (cached singleton) when not
    supplied; the shim passes the dict it already holds.
    """
    if config is None:
        config = load_config()
    override = get_setting(config, "transport.kernel_log")
    if override:
        return Path(override)
    return _locations.mcp_server_log()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* in place, returning *base*.

    Nested dicts are merged key-by-key so a partial user section (e.g. only
    ``{"kernel": {"promote_after": 30}}``) overrides just that leaf and leaves its
    sibling defaults intact. Non-dict values (and dict-vs-non-dict mismatches)
    replace wholesale.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# --- Declarative config validation (biopb/biopb#182) --------------------------
#
# The merged dict is otherwise trusted: a valid-JSON but out-of-range leaf passes
# straight through and only blows up on a hot path. Validate here, driven by the
# dataclass _CONSTRAINTS + section->class map, so a value is judged by the same
# rule the schema advertises. Severity per #182: the deep-merge already degrades
# gracefully, so a bad leaf is a logged WARNING and is reset to its known-good
# DEFAULT_CONFIG value (which also neutralizes the no-coercion wrinkle -- a string
# where a number is expected fails its Range check and is replaced by the default).


def _walk_path(node: dict, keys):
    """Follow dotted-path *keys* into *node*; return ``_MISSING`` on any miss."""
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def config_problems(config: dict) -> List[Problem]:
    """Every constraint violation in *config* (empty when valid).

    The shared checker, applied to biopb-mcp's sections -- the same call the
    tensor server's load path and the control's admin endpoints make, so a knob
    is judged identically wherever it is met (biopb/biopb#34, #182).
    """
    return check_sections(
        ((section, config.get(section, {})) for section in _SECTION_CLASSES),
        _CONSTRAINTS,
        class_names={s: cls.__name__ for s, cls in _SECTION_CLASSES.items()},
    )


def _validate_and_clamp(config: dict) -> dict:
    """Warn on each out-of-range leaf and reset it to its default, in place.

    The load-path policy (see :mod:`biopb._config_validate`): a bad value must
    not reach the runtime, but must not take the session down either -- a raise
    here is a dead MCP client and no viewer. A leaf absent from the merged dict
    is skipped (nothing to check). Returns *config*.
    """
    warn_and_clamp(
        config_problems(config),
        lambda path: _walk_path(DEFAULT_CONFIG, path),
        lambda path, value: config[path[0]].__setitem__(path[1], copy.deepcopy(value)),
        logger,
    )
    return config


# Keys renamed by the knowledge-store redesign (biopb-mcp/docs/knowledge.md §1),
# read for one release so an existing config file is not silently ignored. The
# new key wins where both are present.
_RENAMED_KEYS = {
    "services": {
        "skills_local_dir": "docs_local_dir",
    },
}


def _apply_renames(config: dict) -> dict:
    """Carry a retired key's value onto its replacement, in place."""
    for section, mapping in _RENAMED_KEYS.items():
        values = config.get(section)
        if not isinstance(values, dict):
            continue
        for old, new in mapping.items():
            if old in values and new not in values:
                values[new] = values.pop(old)
                logger.warning(
                    "config: %s.%s is now %s.%s; reading the old key this release",
                    section,
                    old,
                    section,
                    new,
                )
    return config


def _read_and_merge_from_disk() -> dict:
    """Read the config file and merge onto defaults.

    Returns a fully-merged config dict with every expected key present. A missing,
    malformed, or unreadable file falls back to ``get_default_config()`` (logged).
    This is the single code path from disk into memory; the ``CONFIG`` singleton
    and :func:`load_config` both route through it.
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.debug("Config file not found, using defaults")
        return get_default_config()

    try:
        with config_path.open("r") as f:
            config = json.load(f)

        # Deep-merge with defaults so partial user sections override only their own
        # leaves and every expected key still resolves.
        merged = _deep_merge(get_default_config(), _apply_renames(config))
        # Reject out-of-range / bad-enum leaves (warn + reset) before any hot path
        # reads them (biopb/biopb#182).
        _validate_and_clamp(merged)

        logger.debug("Loaded config from %s", config_path)
        return merged

    except json.JSONDecodeError as e:
        logger.warning("Config file malformed, using defaults: %s", e)
        return get_default_config()
    except Exception as e:
        logger.warning("Failed to load config, using defaults: %s", e)
        return get_default_config()


class _Config:
    """Process-wide config singleton: lazy cached read, write-through to disk.

    One instance (:data:`CONFIG`) is the single runtime source of truth. The
    first access reads disk once (via :func:`_read_and_merge_from_disk`) and
    caches the merged dict for the process lifetime; :meth:`reload` invalidates
    it. Every read routes through :meth:`get` and every write through :meth:`set`.

    Thread-safe: the MCP kernel runs agent code in background threads that may
    read config, so lazy init / set / reload take an ``RLock``.
    """

    def __init__(self) -> None:
        self._data: Optional[dict] = None
        self._lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._data is None:
                self._data = _read_and_merge_from_disk()

    def get(self, path: str, default=_MISSING):
        """Read a dotted *path*, falling back to ``DEFAULT_CONFIG``.

        Same contract as :func:`get_setting` (the ambient form of it). Holds the
        lock across the dotted-path walk so a read never observes a half-applied
        multi-key write or a momentarily-``None`` cache mid-reload.
        """
        with self._lock:
            self._ensure_loaded()
            return get_setting(self._data, path, default)

    def set(self, path: str, value, *, persist: bool = True) -> None:
        """Set a dotted *path* in the cache; write through to disk by default.

        Walks/creates intermediate sections, sets the leaf, and (when *persist*)
        atomically rewrites the file so cache and disk stay in lockstep. Pass
        ``persist=False`` to batch several sets, then call :meth:`save` once.
        """
        self._ensure_loaded()
        with self._lock:
            node = self._data
            keys = path.split(".")
            for key in keys[:-1]:
                child = node.get(key)
                if not isinstance(child, dict):
                    child = {}
                    node[key] = child
                node = child
            node[keys[-1]] = value
            if persist:
                self._save()

    def save(self) -> None:
        """Persist the current cached config to disk (single atomic write)."""
        self._ensure_loaded()
        with self._lock:
            self._save()

    def reload(self) -> None:
        """Invalidate the cache; the next read re-reads disk."""
        with self._lock:
            self._data = None

    def as_dict(self) -> dict:
        """Return the *live* cached merged dict (not a copy).

        Escape hatch for code that threads the raw dict (the MCP bootstrap).
        Callers must treat it as read-only; all mutation goes through :meth:`set`.
        Not safe to hold across a concurrent :meth:`set` -- use it for
        startup/single-threaded threading and prefer :meth:`get` otherwise.
        """
        self._ensure_loaded()
        return self._data

    def _save(self) -> None:
        atomic_write_json(get_config_path(), self._data, raise_on_error=False)


# The one config instance in the process. All reads/writes route through it.
CONFIG = _Config()


def load_config() -> dict:
    """Return the process config dict (cached singleton).

    Back-compat shim over :data:`CONFIG`: returns the live merged dict so callers
    that thread a dict (e.g. the MCP bootstrap) keep working. The result is the
    shared cached instance -- treat it read-only and write via ``CONFIG.set``.
    """
    return CONFIG.as_dict()


def save_config(config: dict) -> None:
    """Persist *config* to disk and refresh the singleton cache.

    Back-compat shim: atomically writes the given dict, then invalidates the cache
    so the next read re-merges it from disk. Prefer ``CONFIG.set(path, value)``
    for targeted writes.
    """
    atomic_write_json(get_config_path(), config, raise_on_error=False)
    CONFIG.reload()
