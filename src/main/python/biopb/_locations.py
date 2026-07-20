"""Single source of truth for where every biopb file lives — shared, stdlib-only.

Two concerns, one module because they answer the same question ("what path does
this file have?") and every consumer needs both:

1. **The config file** — *where* the tensor-server config lives and *which*
   format wins when both exist (JSON is canonical; a legacy TOML is read through
   a deprecation window). Imported by ``biopb-tensor-server``
   (``config.find_config``), the umbrella ``biopb`` CLI, and ``biopb-mcp``.
2. **The runtime trees** — the XDG base dirs and every log / session-registry /
   pid / stop-sentinel / asset path derived from them. These used to be
   open-coded as literal strings across five packages (the core CLI, biopb-mcp,
   biopb-control, biopb-tensor-server, and both installers), which drifted
   (``logs`` vs ``log``; a stray top-level ``biopb-mcp`` tree) and forced
   hand-synced duplicates (the same ``tensor-server.stop`` literal in the
   supervisor *and* the tensor server's shutdown listener). Centralizing them
   here means a reader and a writer cannot disagree.

**XDG base directories** (honored on every platform, matching the installer's
``~/.config``-everywhere convention rather than per-OS native dirs):

- config  -> ``$XDG_CONFIG_HOME`` (default ``~/.config``)      ``biopb.json`` etc.
- state   -> ``$XDG_STATE_HOME``  (default ``~/.local/state``) logs, sessions, pids
- data    -> ``$XDG_DATA_HOME``   (default ``~/.local/share``) webapp, samples

Logs and the session registry are XDG **state** (per-machine, regenerable), not
**data** (portable assets) — so they sit in the state tree, beside the pid and
sentinel files, while the browser bundle and sample images stay in data.

Deliberately stdlib-only (``os`` + ``pathlib`` + ``logging``) so importing it is
cheap on every CLI invocation and so both ``biopb-control`` and
``biopb-tensor-server`` (which already depend on core ``biopb``) can bind to it
without a new dependency edge; it drags in none of the heavy adapter/discovery
machinery ``biopb_tensor_server.core.config`` does. Paths are resolved **at call
time**, never cached in a module constant, so a test that repoints
``Path.home()`` / an ``XDG_*`` env var gets an isolated tree for free.

JSON is the *canonical* on-disk config format going forward: the config is
machine-generated (the installer / a future generator write it), and once nobody
hand-edits it, TOML's hand-editing ergonomics stop paying for its one wart — no
stdlib *writer*. JSON has a stdlib writer on both ends, unifies the format with
biopb-mcp's ``mcp-config.json``, and pairs with JSON Schema for validation. TOML
stays *readable* through a deprecation window so no existing ``biopb.toml`` breaks
on upgrade. See biopb/biopb#34.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Env override for just the session-registry dir (predates this module). Kept so
# a test / an unusual deployment can repoint the registry without moving the rest
# of the state tree. XDG_STATE_HOME moves everything; this moves only sessions.
SESSIONS_DIR_ENV = "BIOPB_SESSIONS_DIR"


# --- XDG base trees ------------------------------------------------------ #


def _tree(env_var: str, default_rel: str) -> Path:
    """The ``biopb`` subdir of an XDG base dir.

    Honors *env_var* when it holds an **absolute** path (the XDG spec says a
    relative value is invalid and must be ignored); otherwise falls back to
    ``~/<default_rel>``. ``Path.home()`` is read at call time for test isolation.
    """
    raw = os.environ.get(env_var)
    root = Path(raw) if raw and os.path.isabs(raw) else Path.home() / default_rel
    return root / "biopb"


def config_dir() -> Path:
    """Config tree (``~/.config/biopb``): ``biopb.json``, ``mcp-config.json``, …"""
    return _tree("XDG_CONFIG_HOME", ".config")


def state_dir() -> Path:
    """State tree (``~/.local/state/biopb``): logs, session registry, pid, sentinels."""
    return _tree("XDG_STATE_HOME", ".local/state")


def data_dir() -> Path:
    """Data tree (``~/.local/share/biopb``): portable assets (webapp bundle, samples)."""
    return _tree("XDG_DATA_HOME", ".local/share")


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


def find_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Path:
    """Resolve the config file in *config_dir*, preferring JSON over TOML.

    Returns the first of ``biopb.json`` / ``biopb.toml`` that exists. When
    neither exists, returns the canonical JSON path so callers seed / print the
    forward-looking name. Callers that need a guaranteed-existing file should
    still check ``.exists()`` on the result.

    When *both* files exist the legacy TOML is silently shadowed, so this logs a
    warning naming the file that is being ignored.
    """
    json_path = config_dir / CANONICAL_CONFIG_NAME
    toml_path = config_dir / LEGACY_CONFIG_NAME
    if json_path.exists():
        if toml_path.exists():
            logger.warning(
                "Both %s and %s exist in %s; using %s and ignoring the legacy "
                "%s. Remove the TOML file to silence this. See biopb/biopb#34.",
                CANONICAL_CONFIG_NAME,
                LEGACY_CONFIG_NAME,
                config_dir,
                CANONICAL_CONFIG_NAME,
                LEGACY_CONFIG_NAME,
            )
        return json_path
    if toml_path.exists():
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


# --- session registry / pids / sentinels --------------------------------- #


def sessions_dir() -> Path:
    """The live-session registry dir; created on access.

    ``BIOPB_SESSIONS_DIR`` overrides the location (used by tests and unusual
    deployments); otherwise ``state/biopb/sessions``.
    """
    raw = os.environ.get(SESSIONS_DIR_ENV)
    d = Path(raw) if raw else state_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def control_pid_file() -> Path:
    """The control plane's pid file."""
    return state_dir() / "control.pid"


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
