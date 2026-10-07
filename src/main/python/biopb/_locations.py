"""Single source of truth for where every biopb file lives (stdlib-only).

Covers the config file (``biopb.json``, JSON only) and the runtime trees (logs,
session registry, pids, stop sentinels, assets). Paths are resolved at call time,
never cached, so tests can repoint ``Path.home()`` / ``BIOPB_*`` variables.

Base directories use the same layout on every platform, each relocated by an
ABSOLUTE ``BIOPB_*`` variable; the ``XDG_*`` variables are not read:

- config  -> ``$BIOPB_CONFIG_HOME`` (default ``~/.config``)
- state   -> ``$BIOPB_STATE_HOME``  (default ``~/.local/state``) logs, sessions, pids
- data    -> ``$BIOPB_DATA_HOME``   (default ``~/.local/share``) webapp, samples
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Env override for just the session-registry dir (BIOPB_STATE_HOME moves everything).
SESSIONS_DIR_ENV = "BIOPB_SESSIONS_DIR"

# Env var naming the file a session's own stdout/stderr went to; set by whoever
# redirected it (shim or control), read by the session. Defined here because the
# processes involved may not import each other.
MCP_SESSION_LOG_ENV = "BIOPB_MCP_SESSION_LOG"

# One-shot token identifying this launch of a viewer session, set by the control
# on the child and echoed into the child's registry record. Matching on pid is
# unreliable: on Windows a venv python.exe is often a trampoline whose pid differs
# from the real interpreter's.
MCP_LAUNCH_TOKEN_ENV = "BIOPB_MCP_LAUNCH_TOKEN"

# The registry-record field the token is echoed into (biopb._sessions.register).
LAUNCH_TOKEN_FIELD = "launch_token"


# --- base trees ---------------------------------------------------------- #
#
# biopb owns its own env namespace (``BIOPB_*_HOME``) and does NOT read ``XDG_*``:
# processes that inherit different XDG values would disagree about the state tree
# and split the session registry. A mere precedence order would not fix that, so
# the XDG read is absent entirely.

_TREE_ENV_CONFIG = "BIOPB_CONFIG_HOME"
_TREE_ENV_STATE = "BIOPB_STATE_HOME"
_TREE_ENV_DATA = "BIOPB_DATA_HOME"
_TREE_ENV_CACHE = "BIOPB_CACHE_HOME"

# One warning per biopb var per process about an ignored XDG variable.
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

    It would resolve against each process's own working directory, so processes
    that must agree on a path would not.
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

    Honors *env_var* when set (must be absolute, see :func:`_require_absolute`);
    otherwise ``~/<default_rel>``. *legacy_xdg_var* is only detected, to warn.
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


def cache_dir() -> Path:
    """Cache tree: regenerable bytes, safe to delete.

    ``~/.cache/biopb``, and ``%LOCALAPPDATA%\\biopb\\Cache`` on Windows (the one
    platform-divergent tree: it can hold tens of gigabytes and must stay out of
    roaming profiles). ``AppData/Local`` is derived from :func:`Path.home`, not
    from ``%LOCALAPPDATA%``.
    """
    if sys.platform == "win32":
        raw = os.environ.get(_TREE_ENV_CACHE)
        if raw:
            _require_absolute(_TREE_ENV_CACHE, raw)
            return Path(raw) / "biopb"
        return Path.home() / "AppData" / "Local" / "biopb" / "Cache"
    return _tree(_TREE_ENV_CACHE, "XDG_CACHE_HOME", ".cache")


# --- config file (location + format) ------------------------------------- #

# Resolved at import for the typer Option default; ``config_dir()`` is the call-time source.
DEFAULT_CONFIG_DIR = config_dir()
CANONICAL_CONFIG_NAME = "biopb.json"

# biopb-mcp's own settings file, in the config dir. Defined here so consumers that
# may not import biopb_mcp agree on its location.
MCP_CONFIG_NAME = "mcp-config.json"


def mcp_config_path() -> Path:
    """The biopb-mcp settings file (``~/.config/biopb/mcp-config.json``)."""
    return config_dir() / MCP_CONFIG_NAME


def mcp_docs_dir() -> Path:
    """The agent's own docs (``~/.config/biopb/docs``).

    The local tier of biopb-mcp's knowledge store (``*.md`` written by
    ``write_doc``, plus its index); shadows a shipped doc of the same id. Not
    created on access.
    """
    return config_dir() / "docs"


def algorithms_dir() -> Path:
    """The algorithm registry (``~/.config/biopb/algorithms``).

    One entry per file, named by its stem: ``<name>.py`` is a server file the
    control runs under uv, ``<name>.json`` (``{"url": ...}``) a server someone
    else runs. Not created on access.
    """
    return config_dir() / "algorithms"


def algorithms_state_dir() -> Path:
    """What the control keeps per algorithm entry: each script's cached op
    list, and its log. Created on access."""
    d = state_dir() / "algorithms"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Path:
    """Resolve the config file in *config_dir*: ``biopb.json``.

    Returns the canonical path whether or not it exists.
    """
    return config_dir / CANONICAL_CONFIG_NAME


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
    """biopb-mcp's log subtree (``state/biopb/mcp``); created on access."""
    d = state_dir() / "mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mcp_server_log() -> Path:
    """Canonical combined log for a direct ``--transport http`` MCP launch."""
    return mcp_log_dir() / "mcp-server.log"


def mcp_viewer_log_dir() -> Path:
    """Where a viewer session the **control** launched writes its output.

    One file per launch, so concurrent viewers do not interleave. Lives here
    because the control may not import biopb-mcp. Retention is the caller's.
    """
    d = mcp_log_dir() / "viewers"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- session registry / pids / sentinels --------------------------------- #


def sessions_dir() -> Path:
    """The live-session registry dir; created on access.

    ``BIOPB_SESSIONS_DIR`` overrides the location (must be absolute);
    otherwise ``state/biopb/sessions``.
    """
    raw = os.environ.get(SESSIONS_DIR_ENV)
    if raw:
        _require_absolute(SESSIONS_DIR_ENV, raw)
    d = Path(raw) if raw else state_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tls_served_certs() -> Path:
    """What each local flight plane serves (``state/biopb/tls-served.json``).

    Written by a plane that serves TLS, read by same-machine clients so they
    verify the certificate actually served. Distinct from :func:`tls_server_cert`
    (the pair the plane mints; an operator's own ``--tls-cert`` never lands
    there). Keyed by port; never created on access.
    """
    return state_dir() / "tls-served.json"


def tls_known_hosts() -> Path:
    """TOFU pin store for the tensor Flight client (``state/biopb/tls-known-hosts.json``).

    Maps ``host:port`` to the certificate pinned on first connect (SSH
    ``known_hosts`` model). Not created on access.
    """
    return state_dir() / "tls-known-hosts.json"


def tls_server_cert() -> Path:
    """The tensor server's TLS certificate (``state/biopb/tls/server-cert.pem``).

    Auto-generated self-signed cert served under ``--tls``; public material.
    """
    return state_dir() / "tls" / "server-cert.pem"


def tls_server_key() -> Path:
    """The tensor server's TLS private key (``state/biopb/tls/server-key.pem``).

    The secret half of :func:`tls_server_cert`; written owner-only.
    """
    return state_dir() / "tls" / "server-key.pem"


def control_pid_file() -> Path:
    """The control plane's pid file."""
    return state_dir() / "control.pid"


def control_runtime_file() -> Path:
    """Where a *serving* control publishes its endpoint (``state/biopb/control.json``).

    Written by whoever bound the socket, for both ``control start`` and a
    foreground ``control run`` (which has no pid file), and retracted on clean
    stop. Not a secret, so no owner-only perms.
    """
    return state_dir() / "control.json"


def control_stop_sentinel() -> Path:
    """The control plane's Windows stop-sentinel (watched by ``biopb_control._run``)."""
    return state_dir() / "control.stop"


def tensor_stop_sentinel() -> Path:
    """The data plane's Windows stop-sentinel.

    Written by ``DataPlaneSupervisor``, watched by the tensor server's
    ``_install_windows_shutdown_listener``.
    """
    return state_dir() / "tensor-server.stop"


def tensor_catalog_path(config_path: Path) -> Path:
    """The tensor server's on-disk DuckDB catalog for *config_path*.

    Named by a digest of the resolved config path, so servers started from
    different ``biopb.json`` files do not share (and lock-collide on) one file.
    In the state tree, not the cache tree: its ``rois`` rows are hand-drawn and
    not regenerable.
    """
    digest = hashlib.sha256(
        str(Path(config_path).expanduser().resolve()).encode("utf-8")
    ).hexdigest()[:16]
    return state_dir() / "catalogs" / f"{digest}.duckdb"


# --- portable assets (data tree) ----------------------------------------- #


def webapp_dir() -> Path:
    """The installed browser bundle (``data/biopb/webapp``)."""
    return data_dir() / "webapp"


def samples_dir() -> Path:
    """The sample-image data folder the installer seeds (``data/biopb/samples``)."""
    return data_dir() / "samples"


# --- rotation ------------------------------------------------------------ #


LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5


def rotate_log(
    log_file: Path, max_bytes: int = LOG_MAX_BYTES, backup_count: int = LOG_BACKUP_COUNT
) -> None:
    """Rotate *log_file* if it exceeds *max_bytes*, keeping up to *backup_count*
    backups (``.1`` … ``.N``).

    """
    if not log_file.exists() or log_file.stat().st_size < max_bytes:
        return
    for i in range(backup_count - 1, 0, -1):
        src = log_file.parent / f"{log_file.name}.{i}"
        dst = log_file.parent / f"{log_file.name}.{i + 1}"
        if src.exists():
            os.replace(src, dst)  # rename() will not replace the oldest on Windows
    os.replace(log_file, log_file.parent / f"{log_file.name}.1")
