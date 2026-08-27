"""Single source of truth for where every biopb file lives — shared, stdlib-only.

Two concerns, one module because they answer the same question ("what path does
this file have?") and every consumer needs both:

1. **The config file** — *where* the tensor-server config lives and *which*
   format wins when both exist (JSON is the only format read; a leftover legacy
   TOML is detected purely to point at the migration command). Imported by
   ``biopb-tensor-server`` (``config.find_config``) and the umbrella ``biopb`` CLI.
2. **The runtime trees** — the XDG base dirs and every log / session-registry /
   pid / stop-sentinel / asset path derived from them. These used to be
   open-coded as literal strings across five packages (the core CLI, biopb-mcp,
   biopb-control, biopb-tensor-server, and both installers), which drifted
   (``logs`` vs ``log``; a stray top-level ``biopb-mcp`` tree) and forced
   hand-synced duplicates (the same ``tensor-server.stop`` literal in the
   supervisor *and* the tensor server's shutdown listener). Centralizing them
   here means a reader and a writer cannot disagree.

**Base directories** (the same layout on every platform, matching the
installer's ``~/.config``-everywhere convention rather than per-OS native dirs).
Each is relocated by its own ``BIOPB_*`` variable, which must be an ABSOLUTE
path; biopb does **not** read the ``XDG_*`` variables (see the note above
``_tree``):

- config  -> ``$BIOPB_CONFIG_HOME`` (default ``~/.config``)      ``biopb.json`` etc.
- state   -> ``$BIOPB_STATE_HOME``  (default ``~/.local/state``) logs, sessions, pids
- data    -> ``$BIOPB_DATA_HOME``   (default ``~/.local/share``) webapp, samples

Logs and the session registry are XDG **state** (per-machine, regenerable), not
**data** (portable assets) — so they sit in the state tree, beside the pid and
sentinel files, while the browser bundle and sample images stay in data.

Deliberately stdlib-only (``os`` + ``pathlib`` + ``logging``) so importing it is
cheap on every CLI invocation and so both ``biopb-control`` and
``biopb-tensor-server`` (which already depend on core ``biopb``) can bind to it
without a new dependency edge; it drags in none of the heavy adapter/discovery
machinery ``biopb_tensor_server.core.config`` does. Paths are resolved **at call
time**, never cached in a module constant, so a test that repoints
``Path.home()`` / a ``BIOPB_*`` env var gets an isolated tree for free.

JSON is the *only* on-disk config format: the config is machine-generated (the
installer / the admin endpoint write it), and once nobody hand-edits it, TOML's
hand-editing ergonomics stop paying for its one wart — no stdlib *writer*. JSON
has a stdlib writer on both ends, unifies the format with biopb-mcp's
``mcp-config.json``, and pairs with JSON Schema for validation. The TOML read
path was dropped once the deprecation window closed (biopb/biopb#34); a leftover
``biopb.toml`` is still *recognized* — by the installers, which convert it, and
by :func:`find_config`, which names ``biopb-tensor-server migrate-config`` — so an old
install fails with the fix rather than with a phantom missing file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Env override for just the session-registry dir (predates this module). Kept so
# a test / an unusual deployment can repoint the registry without moving the rest
# of the state tree. BIOPB_STATE_HOME moves everything; this moves only sessions.
SESSIONS_DIR_ENV = "BIOPB_SESSIONS_DIR"


# --- base trees ---------------------------------------------------------- #
#
# biopb owns its own env namespace (``BIOPB_*_HOME``) and does NOT read the
# ``XDG_*`` variables.
#
# It used to read them, and that was a bug (biopb/biopb#790). The XDG variables
# are a freedesktop convention, but biopb honored them on every platform --
# including Windows, where nothing owns them and any process may set them for
# its own purposes. An MCP client that sets ``XDG_STATE_HOME`` to its own
# working directory (opencode desktop does) has that value inherited by the
# biopb-mcp shim it spawns, while a control plane started from a terminal keeps
# the default. The two then disagree about where the state tree is, and the
# session registry -- whose whole contract is that the shim writes what the
# control reads (see ``biopb._sessions``) -- silently splits in half.
#
# The other consumers of the state tree hid the same skew behind fallbacks: the
# control endpoint record degrades to the default port, and the credential file
# degrades to the tokenless path. Sessions were simply the one with no fallback.
#
# A ``BIOPB_*`` variable that merely takes *precedence* over ``XDG_*`` would not
# fix this: the bug happens when the biopb variable is unset, which is the normal
# case. So the XDG read is gone, not reordered. The DEFAULTS are unchanged
# (``~/.config``, ``~/.local/state``, ``~/.local/share``), so an install that
# never set an XDG variable sees no difference.

_TREE_ENV_CONFIG = "BIOPB_CONFIG_HOME"
_TREE_ENV_STATE = "BIOPB_STATE_HOME"
_TREE_ENV_DATA = "BIOPB_DATA_HOME"

# One warning per (biopb var) per process: relocating via XDG used to work, so a
# stale deployment must be told its tree moved back to the default rather than
# silently losing sight of its logs / certs / pids.
_LEGACY_XDG_WARNED: set = set()


def _warn_legacy_xdg(biopb_var: str, xdg_var: str) -> None:
    if biopb_var in _LEGACY_XDG_WARNED:
        return
    _LEGACY_XDG_WARNED.add(biopb_var)
    logger.warning(
        "%s is set but biopb no longer reads it; using the default tree. Set %s "
        "instead to relocate this tree (biopb/biopb#790).",
        xdg_var,
        biopb_var,
    )


def _require_absolute(env_var: str, raw: str) -> None:
    """Refuse a relative path in a location variable.

    A relative value resolves against the **current working directory**, which
    differs between the processes that must agree on these paths: the biopb-mcp
    shim inherits its client's cwd, a control started from a terminal has that
    terminal's, and the installer has whatever the user ran it from. So the same
    variable would name a different directory in each -- the failure mode
    biopb/biopb#790 already produced once, reintroduced through the override.

    Loud rather than ignored-with-a-default: the value was set deliberately, and
    silently relocating the tree somewhere else is exactly the drift this guards.
    """
    if not os.path.isabs(raw):
        raise ValueError(
            f"{env_var} must be an absolute path (got {raw!r}). A relative value "
            f"resolves against each process's working directory, so the installer, "
            f"the control plane, and the biopb-mcp shim would disagree about where "
            f"this tree lives."
        )


def _tree(env_var: str, legacy_xdg_var: str, default_rel: str) -> Path:
    """The ``biopb`` subdir of a base dir.

    Honors *env_var* when set, which must be an **absolute** path (see
    :func:`_require_absolute`); otherwise falls back to ``~/<default_rel>``.
    ``Path.home()`` is read at call time for test isolation.

    *legacy_xdg_var* is only ever *detected*, never read for its value -- see the
    note above.
    """
    raw = os.environ.get(env_var)
    if raw:
        _require_absolute(env_var, raw)
    elif os.environ.get(legacy_xdg_var):
        _warn_legacy_xdg(env_var, legacy_xdg_var)
    return (Path(raw) if raw else Path.home() / default_rel) / "biopb"


def config_dir() -> Path:
    """Config tree (``~/.config/biopb``): ``biopb.json``, ``mcp-config.json``, …"""
    return _tree(_TREE_ENV_CONFIG, "XDG_CONFIG_HOME", ".config")


def state_dir() -> Path:
    """State tree (``~/.local/state/biopb``): logs, session registry, pid, sentinels."""
    return _tree(_TREE_ENV_STATE, "XDG_STATE_HOME", ".local/state")


def data_dir() -> Path:
    """Data tree (``~/.local/share/biopb``): portable assets (webapp bundle, samples)."""
    return _tree(_TREE_ENV_DATA, "XDG_DATA_HOME", ".local/share")


# --- config file (location + format) ------------------------------------- #

# The config tree, resolved at import for the typer Option default; honors
# ``$XDG_CONFIG_HOME``. ``config_dir()`` is the call-time source.
DEFAULT_CONFIG_DIR = config_dir()
CANONICAL_CONFIG_NAME = "biopb.json"
LEGACY_CONFIG_NAME = "biopb.toml"

# biopb-mcp's own settings file, co-located in the same dir. Distinct from the
# installer's client-definition ``mcp.json`` (which registers biopb-mcp with MCP
# clients). Defined here so the three consumers that touch it -- biopb-mcp
# (its config module) and the lean control plane + ``biopb._algorithms`` (which
# read it WITHOUT importing biopb_mcp, invariant I2) -- agree on one location
# and cannot drift. See biopb/biopb#34.
MCP_CONFIG_NAME = "mcp-config.json"


def mcp_config_path() -> Path:
    """The biopb-mcp settings file (``~/.config/biopb/mcp-config.json``).

    Computed at call time (not the import-time ``DEFAULT_CONFIG_DIR`` constant)
    so a test that repoints ``Path.home()`` / ``$XDG_CONFIG_HOME`` gets an
    isolated location.
    """
    return config_dir() / MCP_CONFIG_NAME


def mcp_plugin_dir() -> Path:
    """User kernel-plugin dir (``~/.config/biopb/kernel``).

    ``*.py`` files here are loaded into the biopb-mcp agent kernel's namespace at
    bootstrap -- the low-friction "bring your own tool" path (biopb/biopb-mcp#92),
    beside the installed ``biopb_mcp.namespace`` entry-point packages. Config-tree
    (user-authored), co-located with ``mcp-config.json``. Resolved at call time for
    test isolation and **not created on access**: absence is the normal no-plugins
    case and the loader / the dashboard inspector simply find nothing, so a bare
    read must not materialize an empty dir.
    """
    return config_dir() / "kernel"


def mcp_skill_dir() -> Path:
    """User skills dir (``~/.config/biopb/skills``).

    ``*.md`` files here are merged into the agent's skills catalog beside the
    curated ones, which ship inside biopb-mcp -- the personal tier of the same
    "drop a file in a config dir" path as :func:`mcp_plugin_dir`, and the only
    way a skill reaches a machine outside a release. Config-tree
    (user-authored), resolved at call time for test isolation and **not created
    on access**: absence is the normal no-local-skills case, and a bare read
    must not materialize an empty dir.
    """
    return config_dir() / "skills"


def find_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Path:
    """Resolve the config file in *config_dir*: ``biopb.json``, else a legacy
    ``biopb.toml`` that must be migrated.

    Returns the first of ``biopb.json`` / ``biopb.toml`` that exists. When
    neither exists, returns the canonical JSON path so callers seed / print the
    forward-looking name. Callers that need a guaranteed-existing file should
    still check ``.exists()`` on the result.

    A legacy TOML is **no longer readable** (biopb/biopb#34) but is still
    returned when it is the only config present, and both cases log a warning
    naming ``biopb-tensor-server migrate-config``. Handing the real file back — rather
    than the canonical name that does not exist — is what lets the caller fail
    with "this config needs migrating" instead of "no config at all", which
    every downstream default (a defaulted bind address, a seeded fresh config)
    would otherwise quietly paper over.
    """
    json_path = config_dir / CANONICAL_CONFIG_NAME
    toml_path = config_dir / LEGACY_CONFIG_NAME
    if json_path.exists():
        if toml_path.exists():
            logger.warning(
                "Both %s and %s exist in %s; using %s and ignoring the legacy "
                "%s. Run `biopb-tensor-server migrate-config` to retire it. "
                "See biopb/biopb#34.",
                CANONICAL_CONFIG_NAME,
                LEGACY_CONFIG_NAME,
                config_dir,
                CANONICAL_CONFIG_NAME,
                LEGACY_CONFIG_NAME,
            )
        return json_path
    if toml_path.exists():
        logger.warning(
            "%s in %s is the legacy TOML config format, which is no longer "
            "read; %s is the only supported format. Run "
            "`biopb-tensor-server migrate-config` to convert it. See biopb/biopb#34.",
            LEGACY_CONFIG_NAME,
            config_dir,
            CANONICAL_CONFIG_NAME,
        )
        return toml_path
    return json_path


# --- logs (daemon: control + supervised tensor server) ------------------- #


def log_dir() -> Path:
    """Directory for the durable daemon logs; created on access."""
    d = state_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tensor_server_log() -> Path:
    """The data plane's stdout/stderr log (the supervisor's redirect target)."""
    return log_dir() / "tensor-server.log"


def control_log() -> Path:
    """The control plane's own supervision / control-API log."""
    return log_dir() / "control.log"


# --- logs (biopb-mcp sessions) ------------------------------------------- #


def mcp_log_dir() -> Path:
    """biopb-mcp's log subtree (``state/biopb/mcp``); created on access.

    Replaces the former separate top-level ``~/.local/share/biopb-mcp/log`` tree.
    """
    d = state_dir() / "mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mcp_server_log() -> Path:
    """Canonical combined log for a direct ``--transport http`` MCP launch."""
    return mcp_log_dir() / "mcp-server.log"


def mcp_viewer_log() -> Path:
    """Combined stdout/stderr for a viewer session the **control** launched.

    Control-launched viewers are the one session kind whose output has no other
    home: a shim-owned child logs to the shim's per-session file and a
    ``biopb mcp view`` started by hand writes to that terminal, but a viewer
    spawned from the dashboard has neither. Lives here, in the core SDK, because
    the control may not import biopb-mcp (control ARCHITECTURE.md, I2) and so
    cannot ask it where its logs go.
    """
    return mcp_log_dir() / "viewer.log"


# --- session registry / pids / sentinels --------------------------------- #


def sessions_dir() -> Path:
    """The live-session registry dir; created on access.

    ``BIOPB_SESSIONS_DIR`` overrides the location (used by tests and unusual
    deployments); otherwise ``state/biopb/sessions``. The override must be an
    absolute path -- this registry is the one directory a shim and a control
    *must* agree on, and they do not share a working directory
    (:func:`_require_absolute`).
    """
    raw = os.environ.get(SESSIONS_DIR_ENV)
    if raw:
        _require_absolute(SESSIONS_DIR_ENV, raw)
    d = Path(raw) if raw else state_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tls_known_hosts() -> Path:
    """TOFU pin store for the tensor Flight client (``state/biopb/tls-known-hosts.json``).

    Maps a ``host:port`` to the server certificate pinned on first connect (the
    SSH ``known_hosts`` model, biopb/biopb#604). Machine-local, regenerable trust
    state — hence the state tree, beside the pids/sentinels — not user-authored
    config. Resolved at call time for test isolation; not created on access (an
    absent file is the normal "nothing pinned yet" case).
    """
    return state_dir() / "tls-known-hosts.json"


def tls_server_cert() -> Path:
    """The tensor server's TLS certificate (``state/biopb/tls/server-cert.pem``).

    Auto-generated self-signed cert served when the flight plane runs with
    ``--tls`` (biopb/biopb#604). Public material — world-readable is fine — kept
    in the state tree beside its key. Resolved at call time for test isolation.
    """
    return state_dir() / "tls" / "server-cert.pem"


def tls_server_key() -> Path:
    """The tensor server's TLS private key (``state/biopb/tls/server-key.pem``).

    The secret half of :func:`tls_server_cert`; written owner-only (``0600`` on
    POSIX). Resolved at call time for test isolation.
    """
    return state_dir() / "tls" / "server-key.pem"


def control_pid_file() -> Path:
    """The control plane's pid file."""
    return state_dir() / "control.pid"


def control_runtime_file() -> Path:
    """Where a *serving* control publishes its endpoint (``state/biopb/control.json``).

    The discovery half of what the pid file used to imply. ``control.pid`` is a
    **lifecycle** record -- ``control start`` writes it about the daemon it
    spawned so ``control stop`` can signal it later -- and a foreground
    ``control run`` deliberately has none (its terminal or service manager owns
    the process). But once the control's port became derivable from
    ``--base-port`` rather than fixed at 8813, *both* forms need to publish
    **where** they listen, or a client has no way to find a control that moved.

    So the endpoint is written by whoever actually bound the socket
    (``biopb_control._run``), on the path both commands share, beside the
    ``tensor-server.token`` credential and on the same publish-on-serve /
    retract-on-clean-stop lifetime. Not a secret -- the port is not a
    credential -- so unlike the credential file it carries no owner-only perms
    and is written unconditionally, including for a tokenless local plane.
    """
    return state_dir() / "control.json"


def control_stop_sentinel() -> Path:
    """The control plane's Windows stop-sentinel (watched by ``biopb_control._run``)."""
    return state_dir() / "control.stop"


def tensor_stop_sentinel() -> Path:
    """The data plane's Windows stop-sentinel.

    Written by ``DataPlaneSupervisor`` and watched by the tensor server's
    ``_install_windows_shutdown_listener`` — the single definition both bind to
    (they previously duplicated the literal and relied on a "keep in sync" note).
    """
    return state_dir() / "tensor-server.stop"


# --- portable assets (data tree) ----------------------------------------- #


def webapp_dir() -> Path:
    """The installed browser bundle (``data/biopb/webapp``)."""
    return data_dir() / "webapp"


def samples_dir() -> Path:
    """The sample-image data folder the installer seeds (``data/biopb/samples``)."""
    return data_dir() / "samples"


# --- rotation ------------------------------------------------------------ #


def rotate_log(
    log_file: Path, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5
) -> None:
    """Rotate *log_file* if it exceeds *max_bytes*, keeping up to *backup_count*
    backups (``.1`` … ``.N``).

    A size-triggered manual rotation applied at process (re)start: the core CLI
    calls it for ``control.log`` at ``control start`` and the supervisor for
    ``tensor-server.log`` at each (re)spawn, so their stdout-redirect logs (which
    have no in-process ``RotatingFileHandler``) don't grow unbounded.
    """
    if not log_file.exists() or log_file.stat().st_size < max_bytes:
        return
    for i in range(backup_count - 1, 0, -1):
        src = log_file.parent / f"{log_file.name}.{i}"
        dst = log_file.parent / f"{log_file.name}.{i + 1}"
        if src.exists():
            src.rename(dst)
    log_file.rename(log_file.parent / f"{log_file.name}.1")
