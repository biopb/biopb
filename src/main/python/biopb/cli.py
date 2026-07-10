"""Top-level CLI for BioPB."""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from ._config_location import DEFAULT_CONFIG_DIR, find_config
from ._proc import (
    is_process_running as _is_process_running,
    process_create_time as _process_create_time,
)

console = Console()

app = typer.Typer(
    name="biopb",
    help="BioPB: open protobuf/gRPC protocols for biomedical image processing",
)


def _add_optional_typer(name: str, import_path: str, help: str) -> None:
    """Register a subcommand whose imports may fail.

    The tensor/image subcommands pull in optional dependencies (installed via
    biopb[tensor]) that may be absent or broken (e.g. a transient
    numcodecs/zarr ImportError). When that happens we still want the rest of
    the CLI (version, server management) to work, so we register a stub that
    surfaces the error only when the subcommand is actually invoked.
    """
    import importlib

    try:
        module = importlib.import_module(import_path)
        app.add_typer(module.app, name=name, help=help)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any import error
        error = exc

        # Register a catch-all command so that any `biopb <name> ...` invocation
        # surfaces the import error instead of a confusing crash or usage error.
        @app.command(
            name=name,
            help=f"{help} (unavailable - optional dependencies missing)",
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        )
        def _unavailable(args: List[str] = typer.Argument(None)) -> None:
            console.print(
                f"[red]The '{name}' commands are unavailable:[/red] {error}\n"
                r"[yellow]Install optional dependencies with: pip install 'biopb\[tensor]'[/yellow]"
            )
            raise typer.Exit(1)


# TensorFlight client diagnostics
_add_optional_typer("tensor", "biopb.tensor.cli", "TensorFlight client diagnostics")

# ProcessImage client operations
_add_optional_typer("image", "biopb.image.cli", "ProcessImage client operations")

# Tensor server daemon management
server_app = typer.Typer(
    name="server",
    help="Biopb server daemon management (start/stop/restart/status)",
)
app.add_typer(server_app, name="server")

# Daemon management constants
PID_FILE = Path.home() / ".local" / "share" / "biopb" / "tensor-server.pid"
LOG_DIR = Path.home() / ".local" / "share" / "biopb" / "logs"
DEFAULT_WEBAPP = Path.home() / ".local" / "share" / "biopb" / "webapp"

# Default config path, preferring JSON over legacy TOML and warning when both
# exist. Shared with biopb-tensor-server and biopb-mcp via the (dependency-light)
# core module, so resolving this typer Option default does not import the heavy
# server config module (biopb/biopb#34).
DEFAULT_CONFIG = find_config()

# biopb-mcp daemon management. The MCP server is a separate, optional process
# (the `biopb-mcp` package) managed independently of the tensor server, so it
# gets its own PID/log under biopb-mcp's XDG data dir (matching
# biopb_mcp._config.get_log_dir() -> ~/.local/share/biopb-mcp/log). These
# commands manage the *standalone* `biopb mcp start` daemon: it writes its own
# PID file (biopb_mcp._config.get_pid_file()) so `status`/`stop` find it, and
# logs to ONE canonical file (biopb_mcp._config.get_daemon_log_file() ->
# mcp-server.log). Keep both literals in sync with get_pid_file()/get_log_dir().
#
# Since de-daemonization (biopb-mcp/docs/mcp-dedaemonization-migration.md,
# Layer 1) the stdio shim no longer spawns a shared daemon through here: it
# spawns and OWNS an ephemeral session child that writes NO PID file (the shim
# reaps it directly), so `biopb mcp status` tracks only the standalone daemon.
# That child still writes the same canonical daemon log (fd inherited from the
# shim), so `mcp logs` shows whichever session last ran. Repurposing `biopb mcp`
# around the shared serve-core is deferred to Layer 3.
MCP_PID_FILE = Path.home() / ".local" / "share" / "biopb-mcp" / "mcp-server.pid"
MCP_LOG_DIR = Path.home() / ".local" / "share" / "biopb-mcp" / "log"
MCP_DEFAULT_PORT = 8765  # biopb_mcp default mcp.transport.port (loopback /mcp)


# The installer records the release-v* deployment version it pulled the wheels
# from in this marker file -- a clean PEP 440 string (e.g. "0.6.7"), the
# auto-updater's baseline. This is the *deployment* version and is distinct from
# any single package's version: one release bundles the mutually-paired
# biopb / biopb-tensor-server / biopb-mcp triple. Kept in sync with
# CONFIG_DIR/release.version in install/install.sh.
_RELEASE_VERSION_FILE = DEFAULT_CONFIG_DIR / "release.version"

# The three wheels the installer bundles in one release-v* deployment (see
# install/install.sh). `biopb version` reports each separately so a version skew
# within the installed set is visible; any may be absent, hence "not installed".
_RELEASE_PACKAGES = ("biopb", "biopb-tensor-server", "biopb-mcp")


def _read_release_version() -> str:
    """The installed deployment version from the installer's marker file, or
    'unknown' when it is absent (a dev checkout or non-installer setup that never
    wrote CONFIG_DIR/release.version) or unreadable. Best-effort like
    ``_package_version`` -- reading a version must never crash ``biopb version``,
    so a missing/permission-denied/corrupt (non-UTF-8) marker degrades to
    'unknown' rather than propagating."""
    try:
        # Explicit utf-8 (the installer writes a plain ASCII/utf-8 version), so
        # decoding is deterministic across platforms rather than dependent on the
        # reader's locale (cp1252 on Windows would decode a corrupt marker to
        # garbage instead of failing to 'unknown').
        return _RELEASE_VERSION_FILE.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"
    except Exception:  # noqa: BLE001 - marker read is best-effort (e.g. decode errors)
        return "unknown"


def _package_version(dist_name: str) -> str:
    """Installed version of distribution `dist_name`, or 'not installed'.

    Reads distribution metadata (like biopb.__init__ does for its own version)
    instead of importing the package, so `biopb version` never drags in the
    packages' heavy optional stacks just to print a number, and still reports a
    version when a package is installed but its runtime imports are broken.
    """
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        return "not installed"
    except Exception:  # noqa: BLE001 - metadata read is best-effort
        return "unknown"


@app.command()
def version():
    """Show the installed release version and each bundled package's version."""
    rows = [("release", _read_release_version())]
    rows += [(name, _package_version(name)) for name in _RELEASE_PACKAGES]

    # Left-align the labels so the versions line up in a readable column.
    width = max(len(name) for name, _ in rows) + 1  # +1 for the trailing ':'
    for name, ver in rows:
        console.print(f"{name + ':':<{width}} {ver}")


def _ensure_dirs():
    """Ensure required directories exist."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid_record(pid_file: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read (pid, identity_token) from `pid_file`.

    The file holds one or two whitespace-separated integers: the PID and, since
    the PID-identity fix, a process create-time token (see _process_create_time)
    that distinguishes our daemon from an unrelated process that later inherited
    a reused PID. Tolerates the legacy bare-PID format (token None -> identity
    unverifiable, callers fall back to a liveness-only check) so a pre-upgrade
    file still reads. Returns (None, None) if missing or unparseable.
    """
    if not pid_file.exists():
        return None, None
    try:
        parts = pid_file.read_text().split()
        pid = int(parts[0])
    except (OSError, ValueError, IndexError):
        return None, None
    token: Optional[int] = None
    if len(parts) > 1:
        try:
            token = int(parts[1])
        except ValueError:
            token = None
    return pid, token


def _write_pid_file(pid_file: Path, pid: int, token: Optional[int] = None):
    """Write `pid` (and, when known, its create-time `token`) to `pid_file`.

    The two-line `pid\\ntoken` form is read back by _read_pid_record; a None token
    falls back to the legacy bare-PID form (callers then verify by liveness only).
    """
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{pid}\n{token}" if token is not None else str(pid))


def _remove_pid_file(pid_file: Path):
    """Remove `pid_file` if present."""
    if pid_file.exists():
        pid_file.unlink()


def _write_pid(pid: int):
    """Write the tensor-server daemon PID (+ its create-time identity token)."""
    _ensure_dirs()
    _write_pid_file(PID_FILE, pid, _process_create_time(pid))


def _remove_pid():
    """Remove the tensor-server daemon PID file."""
    _remove_pid_file(PID_FILE)


def _is_our_daemon(pid: Optional[int], token: Optional[int]) -> bool:
    """Whether `pid` is alive AND is the daemon we recorded -- not a reused PID.

    Returns False only when the PID can be PROVEN to be someone else (alive but a
    different creation time), so `stop`/`restart` never force-kill, and `status`
    never trusts, an unrelated process. When identity can't be established -- a
    legacy bare-PID file, or a platform/moment with no create-time -- it falls
    back to liveness, matching the pre-fix behavior rather than risk a false
    "stopped" (which would strand a running daemon).
    """
    if not pid or not _is_process_running(pid):
        return False
    if token is None:
        return True
    current = _process_create_time(pid)
    if current is None:
        return True
    return current == token


# Diagnostic from the most recent _win_request_shutdown() failure, surfaced by
# _stop_daemon() so a Windows user can see why graceful stop fell back to force-kill.
_LAST_WIN_SHUTDOWN_DIAG: Optional[str] = None


def _win_shutdown_sentinel() -> Path:
    """Path of the shutdown sentinel file (Windows graceful stop).

    Must match shutdown_sentinel_path() in biopb_tensor_server.serving.http_server (kept
    as a literal here to avoid importing heavy server deps just to stop). A single
    fixed name - NOT keyed by PID: on Windows the daemon's os.getpid() can differ
    from the PID `start` recorded (Store-Python/uv launcher shims), so a pid in
    the name would make stop and the daemon disagree. One daemon, one sentinel.
    """
    return PID_FILE.parent / "tensor-server.stop"


def _win_request_shutdown(sentinel: Path) -> bool:
    """Windows: ask a daemon to shut down gracefully by dropping its stop
    sentinel. Returns True if the request was delivered (not whether the
    process has exited yet).

    The daemons are windowless background processes in their own process groups,
    so they have no console to receive a CTRL_BREAK, and Win32 named events are
    brittle across sessions/elevation. We instead drop a sentinel *file* the
    daemon polls for (tensor server: _install_windows_shutdown_listener in
    biopb_tensor_server.serving.http_server; MCP daemon: _install_shutdown_sentinel_watcher
    in biopb_mcp.mcp.__main__); a file under the user's biopb dir is unambiguous
    regardless of how either process was launched.
    """
    global _LAST_WIN_SHUTDOWN_DIAG
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("stop")
        return True
    except OSError as e:
        _LAST_WIN_SHUTDOWN_DIAG = f"could not write shutdown sentinel: {e}"
        return False


def _win_remove_sentinel(sentinel: Path) -> None:
    """Remove a shutdown sentinel (best effort), so it doesn't linger after a
    force-kill where the daemon never consumed it."""
    try:
        sentinel.unlink()
    except OSError:
        pass


def _request_graceful_stop(pid: int, sentinel: Path) -> bool:
    """Ask a daemon to shut down gracefully. Returns whether the request was
    delivered (not whether the process has exited yet). `sentinel` is that
    daemon's Windows stop-sentinel path (unused on POSIX, which signals)."""
    if sys.platform == "win32":
        return _win_request_shutdown(sentinel)
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def _stop_daemon(
    pid: int,
    timeout: int,
    token: Optional[int] = None,
    *,
    sentinel: Path,
    remove_pid: Callable[[], None],
) -> bool:
    """Stop a running daemon: request graceful shutdown, wait up to `timeout`
    seconds, then force-kill. Clears the PID file via `remove_pid`. Returns True
    if it exited gracefully, False if it had to be force-killed. Assumes `pid` is
    running.

    The single stop path for both SDK daemons (tensor server and biopb-mcp); the
    two differ only in `sentinel` (their Windows stop-sentinel path) and
    `remove_pid` (which PID file to clear).

    On POSIX, graceful shutdown is a SIGTERM the daemon's handler catches. On
    Windows os.kill is TerminateProcess -- immediate and uncatchable, so a
    handler never runs and (for the MCP daemon) the napari kernel is not reaped
    (issue #323); both daemons therefore watch for a sentinel *file* instead
    (tensor server: http_server._install_windows_shutdown_listener; MCP:
    mcp.__main__._install_shutdown_sentinel_watcher). See _request_graceful_stop.

    `token` is the recorded create-time identity (see _process_create_time):
    delivery, the wait loop, and the force-kill are all gated on it so that if
    the daemon exits and its PID is reused mid-stop, we neither signal, keep
    waiting on, nor TerminateProcess the innocent new owner.
    """
    if _is_our_daemon(pid, token):
        delivered = _request_graceful_stop(pid, sentinel)
        if not delivered and sys.platform == "win32" and _LAST_WIN_SHUTDOWN_DIAG:
            console.print(
                f"[yellow]Graceful stop unavailable ({_LAST_WIN_SHUTDOWN_DIAG}); "
                f"force killing.[/yellow]"
            )

    graceful = False
    for _ in range(timeout):
        if not _is_our_daemon(pid, token):
            graceful = True
            break
        time.sleep(1)

    if not graceful:
        # Force kill. signal.SIGKILL is POSIX-only; on Windows fall back to
        # SIGTERM, which os.kill maps to an unconditional TerminateProcess. Re-verify
        # identity first: a reused PID must never take this unconditional kill.
        if _is_our_daemon(pid, token):
            try:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
            time.sleep(0.5)

    remove_pid()
    if sys.platform == "win32":
        # tidy up if the daemon never consumed it
        _win_remove_sentinel(sentinel)
    return graceful


def _get_log_file() -> Path:
    """Get log file path."""
    return LOG_DIR / "tensor-server.log"


# Severity ranks for the `--level` filter, matching the daemon's log format
# `[asctime] LEVEL name: message` (DEFAULT_LOG_FORMAT in
# biopb_tensor_server.core.logging_config).
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _line_level(line: str) -> Optional[str]:
    """Return the log level of a formatted line, or None if it has none.

    Lines look like `[2026-06-12 10:00:00] WARNING biopb_tensor_server.x: msg`.
    Returns None for anything off-format - the `--- Started at ... ---` banners
    the daemon writes, blank lines, and multi-line traceback continuations.
    """
    if not line.startswith("["):
        return None
    try:
        after_ts = line.split("] ", 1)[1]
    except IndexError:
        return None
    token = after_ts.split(" ", 1)[0]
    return token if token in _LOG_LEVELS else None


def _mcp_line_level(line: str) -> Optional[str]:
    """Best-effort log level of a biopb-mcp daemon log line, or None.

    biopb-mcp logs through logging.basicConfig (`LEVEL:name:message`) and uvicorn
    emits `LEVEL:    ...`: both lead with the level name, delimited by `:` or
    whitespace. Native Qt/dask/gRPC stdout the daemon also captures has no level
    token and is treated as a continuation (carried by _filter_lines). This is
    best-effort - lines in other shapes (e.g. dask's `... - LEVEL - ...`) are
    left unclassified - so it pairs with carry-forward, never hard filtering.
    """
    head = line.split(":", 1)[0].split(" ", 1)[0].strip()
    return head if head in _LOG_LEVELS else None


def _filter_lines(lines, min_level: Optional[str], level_of=_line_level):
    """Keep lines at or above `min_level`. With min_level None, keep all.

    `level_of` extracts a line's level (the tensor-server format by default;
    `_mcp_line_level` for the MCP daemon). Off-format lines (no parseable level)
    inherit the previous line's keep/drop decision, so a kept WARNING record
    carries its traceback continuation lines along and a dropped INFO record
    takes its continuations with it. The initial decision (before any leveled
    line) is keep.
    """
    if min_level is None:
        return list(lines)
    threshold = _LOG_LEVELS[min_level]
    kept = []
    keeping = True
    for line in lines:
        lvl = level_of(line)
        if lvl is not None:
            keeping = _LOG_LEVELS[lvl] >= threshold
        if keeping:
            kept.append(line)
    return kept


def _validate_level(level: Optional[str]) -> Optional[str]:
    """Normalize a `--level` value to upper-case, or exit(1) if unrecognized."""
    if level is None:
        return None
    norm = level.upper()
    if norm not in _LOG_LEVELS:
        console.print(
            f"[red]Invalid --level '{level}'.[/red] "
            f"Choose one of: {', '.join(_LOG_LEVELS)}"
        )
        raise typer.Exit(1)
    return norm


def _tail_and_follow(
    log_file: Path,
    follow: bool,
    lines: int,
    min_level: Optional[str],
    level_of=_line_level,
):
    """Print the last `lines` lines of `log_file` (0 = all) filtered by
    `min_level`, then optionally stream appended lines until interrupted.

    Shared by `server logs` and `mcp logs`; `level_of` selects the per-daemon
    level parser. A missing file is reported (not an error) and exits 0. Follow
    reopens the file when it is rotated or truncated out from under us.
    """
    if not log_file.exists():
        console.print(
            f"[yellow]No log file at {log_file} - has the server ever started?[/yellow]"
        )
        raise typer.Exit(0)

    # Daemon logs rotate at 10 MB (see _rotate_log / RotatingFileHandler), so the
    # current file is small enough to read whole and slice - no seek-based tail.
    existing = log_file.read_text(errors="replace").splitlines()
    tail = existing if lines <= 0 else existing[-lines:]
    for line in _filter_lines(tail, min_level, level_of):
        print(line)

    if not follow:
        raise typer.Exit(0)

    # Follow: poll for appended lines, reopening if the file is rotated or
    # truncated out from under us (a `restart` rotates it mid-follow). Track the
    # inode + size so a replaced or shrunk file restarts from the top.
    try:
        f = open(log_file, errors="replace")
    except OSError:
        raise typer.Exit(0)
    try:
        f.seek(0, os.SEEK_END)
        last_ino = os.fstat(f.fileno()).st_ino
        carry = ""  # buffer a partial final line until its newline arrives
        while True:
            chunk = f.read()
            if chunk:
                carry += chunk
                parts = carry.split("\n")
                carry = parts.pop()  # trailing partial (or "" if chunk ended on \n)
                for line in _filter_lines(parts, min_level, level_of):
                    print(line, flush=True)
                continue
            # No new data: check whether the file was rotated/truncated.
            try:
                st = os.stat(log_file)
            except OSError:
                st = None
            if st is not None and (st.st_ino != last_ino or st.st_size < f.tell()):
                f.close()
                f = open(log_file, errors="replace")
                last_ino = os.fstat(f.fileno()).st_ino
                carry = ""
                continue
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    raise typer.Exit(0)


def _rotate_log(
    log_file: Path, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5
):
    """Rotate log file if it exceeds max_bytes, keeping up to backup_count backups."""
    if not log_file.exists() or log_file.stat().st_size < max_bytes:
        return
    for i in range(backup_count - 1, 0, -1):
        src = log_file.parent / f"{log_file.name}.{i}"
        dst = log_file.parent / f"{log_file.name}.{i + 1}"
        if src.exists():
            src.rename(dst)
    log_file.rename(log_file.parent / f"{log_file.name}.1")


def _detach_kwargs() -> dict:
    """Popen kwargs that detach a spawned daemon from the launching console and
    process group, so it survives this command returning and isn't killed by
    the terminal's Ctrl+C or close (SIGHUP).

    POSIX: start_new_session (setsid) gives the daemon its own session/process
    group. Windows: CREATE_NO_WINDOW runs it without a console *window* (so none
    pops); CREATE_NEW_PROCESS_GROUP makes it a group leader the terminal's
    Ctrl+C does not reach (start_new_session is a silent no-op on Windows).

    Note this detaches the daemon *within* the login session; it does NOT make
    it persistent across logout, and is not meant to. Windows hard-kills session
    processes on logout, and modern systemd-logind kills the user scope on
    logout regardless of setsid (unless `loginctl enable-linger` is set). A
    daemon that must outlive the session — e.g. a shared/remote tensor server —
    must be made persistent at the container/service/wrapper level (a systemd
    unit, `enable-linger`, or a long-lived container), not here.
    """
    if sys.platform == "win32":
        return {
            "creationflags": (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        }
    return {"start_new_session": True}


@server_app.command("start")
def start(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to config file (JSON or TOML)",
    ),
    static_dir: Optional[Path] = typer.Option(
        DEFAULT_WEBAPP,
        "--static-dir",
        help="Directory containing static webapp files",
    ),
    web_port: int = typer.Option(
        8814,
        "--web-port",
        help="HTTP server port",
    ),
    web_host: str = typer.Option(
        "127.0.0.1",
        "--web-host",
        help="HTTP server bind address",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level: DEBUG, INFO, WARNING, ERROR",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Access token (auto-generated if not provided)",
    ),
):
    """Start TensorFlight server as a background daemon."""
    _ensure_dirs()

    # Check if already running (identity-checked, so a reused stale PID does not
    # masquerade as a live server and silently block a real start).
    existing_pid, existing_token = _read_pid_record(PID_FILE)
    if _is_our_daemon(existing_pid, existing_token):
        console.print(
            f"[yellow]TensorFlight server already running (PID {existing_pid})[/yellow]"
        )
        raise typer.Exit(0)

    # Clean up stale PID file
    if existing_pid:
        console.print(
            f"[yellow]Removing stale PID file (process {existing_pid} not running)[/yellow]"
        )
        _remove_pid()

    # Refuse to start on top of an already-bound gRPC port. The stale-but-dead
    # PID was handled above; this catches the orphan case the PID file cannot
    # see -- a data server still serving after its manager exited, its PID file
    # deleted, or a second login session -- which would otherwise double-bind,
    # fail silently in the log, and leave a dead process behind the PID file we
    # are about to write.
    probe_host, probe_port = _resolve_grpc_hostport(config)
    if _port_listening(probe_host, probe_port):
        console.print(
            f"[red]gRPC port {probe_host}:{probe_port} is already in use.[/red]"
        )
        console.print(
            "It is held by a process biopb is not tracking (no matching PID "
            "file -- an orphaned daemon, or another login session), so "
            "[bold]biopb server stop[/bold] cannot reach it. Identify and stop "
            f"the owner (`netstat -ano | findstr {probe_port}` on Windows, "
            f"`lsof -i :{probe_port}` on macOS/Linux), then retry."
        )
        raise typer.Exit(1)

    # Token: skip only when both web and gRPC are bound to localhost
    _LOCALHOST_ADDRS = {"127.0.0.1", "localhost", "::1"}
    if not token:
        token = os.environ.get("BIOPB_TENSOR_TOKEN")

    grpc_host = "0.0.0.0"  # safe default: assume public if config unreadable
    if config.exists():
        try:
            from biopb_tensor_server.core.config import (
                load_config as _load_server_config,
            )

            grpc_host = _load_server_config(config).host
        except Exception:
            pass

    local_only = (
        not token and web_host in _LOCALHOST_ADDRS and grpc_host in _LOCALHOST_ADDRS
    )
    if not token and not local_only:
        import secrets as _secrets

        token = _secrets.token_urlsafe(32)
        console.print(f"[bold green]Generated access token:[/bold green] {token}")

    log_file = _get_log_file()
    _rotate_log(log_file)

    # Build subprocess environment
    env = os.environ.copy()
    if local_only:
        # No token: tell launch to skip token enforcement without prompting
        env["BIOPB_WEB_DEV_BYPASS"] = "1"
    elif token:
        env["BIOPB_TENSOR_TOKEN"] = token

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "biopb_tensor_server.cli",
        "launch",
        "--config",
        str(config),
        "--web-port",
        str(web_port),
        "--web-host",
        str(web_host),
        "--log-level",
        str(log_level),
    ]
    if static_dir and static_dir.exists():
        cmd.extend(["--static-dir", str(static_dir)])

    # Start subprocess
    console.print("[green]Starting TensorFlight server...[/green]")
    console.print(f"  Config: {config}")

    # Detach the daemon from the launching console/process group so it survives
    # this command returning. On Windows it then has no console for graceful
    # `stop` to signal, so stop uses a named shutdown sentinel instead (see
    # _win_request_shutdown / http_server._install_windows_shutdown_listener).
    detach_kwargs = _detach_kwargs()

    with open(log_file, "a") as log:
        log.write(f"\n--- Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            env=env,
            **detach_kwargs,
        )

    _write_pid(process.pid)

    # Wait for the daemon to actually bind its gRPC port -- a readiness check,
    # not just "is the child alive". A bind collision or early crash surfaces as
    # the port never coming up (or the process exiting), which we report here
    # instead of a false "started".
    if not _await_listening(process.pid, probe_host, probe_port, 15.0):
        if _is_process_running(process.pid):
            console.print(
                f"[red]TensorFlight server started but is not listening on "
                f"{probe_host}:{probe_port} after 15s.[/red]"
            )
            console.print(
                "It may still be coming up; check [bold]biopb server status -w 30[/bold], "
                "or [bold]biopb server stop[/bold] if it is wedged."
            )
        else:
            console.print("[red]Failed to start TensorFlight server[/red]")
            _remove_pid()
        console.print(f"Check logs: {log_file}")
        raise typer.Exit(1)

    console.print(f"[green]TensorFlight server started (PID {process.pid})[/green]")
    url = f"http://{web_host}:{web_port}/" + (f"?token={token}" if token else "")
    console.print(f"  HTTP: {url}")
    console.print("  gRPC: grpc://127.0.0.1:8815")
    console.print(f"  Logs: {log_file}")


@server_app.command("stop")
def stop(
    timeout: int = typer.Option(
        10,
        "--timeout",
        "-t",
        help="Seconds to wait for graceful shutdown",
    ),
):
    """Stop TensorFlight server daemon."""
    pid, token = _read_pid_record(PID_FILE)

    if not pid:
        console.print("[yellow]No TensorFlight server running[/yellow]")
        raise typer.Exit(0)

    if not _is_our_daemon(pid, token):
        console.print(
            f"[yellow]Process {pid} not running, cleaning up PID file[/yellow]"
        )
        _remove_pid()
        raise typer.Exit(0)

    console.print(f"[green]Stopping TensorFlight server (PID {pid})...[/green]")

    if _stop_daemon(
        pid, timeout, token, sentinel=_win_shutdown_sentinel(), remove_pid=_remove_pid
    ):
        console.print("[green]TensorFlight server stopped[/green]")
    else:
        console.print(f"[yellow]Did not stop within {timeout}s; force killed[/yellow]")
    raise typer.Exit(0)


@server_app.command("restart")
def restart(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to config file (JSON or TOML)",
    ),
    static_dir: Optional[Path] = typer.Option(
        DEFAULT_WEBAPP,
        "--static-dir",
        help="Directory containing static webapp files",
    ),
    web_port: int = typer.Option(
        8814,
        "--web-port",
        help="HTTP server port",
    ),
    web_host: str = typer.Option(
        "127.0.0.1",
        "--web-host",
        help="HTTP server bind address",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level: DEBUG, INFO, WARNING, ERROR",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Access token (auto-generated if not provided)",
    ),
    timeout: int = typer.Option(
        10,
        "--timeout",
        "-t",
        help="Seconds to wait for graceful shutdown",
    ),
):
    """Restart TensorFlight server daemon."""
    # Stop first. Use id_token (the PID-record identity token) so we don't clobber
    # the `token` access-token parameter, which must be passed through to start().
    pid, id_token = _read_pid_record(PID_FILE)
    if _is_our_daemon(pid, id_token):
        console.print(f"[green]Stopping TensorFlight server (PID {pid})...[/green]")
        _stop_daemon(
            pid,
            timeout,
            id_token,
            sentinel=_win_shutdown_sentinel(),
            remove_pid=_remove_pid,
        )
        time.sleep(1)

    # Start with same options
    start(
        config=config,
        static_dir=static_dir,
        web_port=web_port,
        web_host=web_host,
        log_level=log_level,
        token=token,
    )


def _resolve_grpc_hostport(config: Path) -> Tuple[str, int]:
    """Loopback-reachable gRPC host/port from the config (default
    127.0.0.1:8815). A server bound to 0.0.0.0/:: is reached over loopback, so
    the returned host is always something connect()-able locally."""
    host, port = "127.0.0.1", 8815
    if config and config.exists():
        try:
            from biopb_tensor_server.core.config import (
                load_config as _load_server_config,
            )

            cfg = _load_server_config(config)
            host = cfg.host or host
            port = int(cfg.port or port)
        except Exception:
            pass
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return host, port


def _resolve_grpc_endpoint(config: Path) -> Tuple[str, Optional[str]]:
    """Best-effort gRPC endpoint + token for a running server's health query.

    Reads host/port from the config (defaults 127.0.0.1:8815); a server
    bound to 0.0.0.0/:: is reached over loopback. The token comes from
    BIOPB_TENSOR_TOKEN if set -- localhost-only daemons run without one.
    """
    host, port = _resolve_grpc_hostport(config)
    token = os.environ.get("BIOPB_TENSOR_TOKEN") or None
    return f"grpc://{host}:{port}", token


def _query_server(
    location: str, token: Optional[str], call: Callable[[Any], dict]
) -> Optional[dict]:
    """Open a short-lived TensorFlightClient to *location*, return ``call(client)``.

    Returns None if the client import fails or the server is unreachable; the
    client is always closed. The shared body behind the status/cache-stats probes.
    """
    try:
        from biopb.tensor.client import TensorFlightClient
    except Exception:
        return None
    client = None
    try:
        client = TensorFlightClient(location, cache_bytes=0, token=token)
        return call(client)
    except Exception:
        return None
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _query_health(location: str, token: Optional[str]) -> Optional[dict]:
    """Return the server's Flight health dict, or None if unreachable."""
    return _query_server(location, token, lambda c: c.health_check())


def _query_cache_stats(location: str, token: Optional[str]) -> Optional[dict]:
    """Return the server's cache-stats dict, or None if unreachable / no cache."""
    return _query_server(location, token, lambda c: c.cache_stats())


@dataclass
class Probe:
    """A daemon's liveness/health snapshot. `listening` says the daemon is up;
    `health` is a richer status dict a daemon may expose (None if it exposes none,
    or if the query failed -- probing never raises, so callers render either
    daemon uniformly instead of guarding every query)."""

    listening: bool
    health: Optional[dict] = None


def _probe_daemon(
    host: str, port: int, health_fn: Optional[Callable[[], Optional[dict]]] = None
) -> Probe:
    """One uniform liveness/health snapshot for either SDK daemon (never raises).

    Readiness and health are the same question at two fidelities, unified here. A
    daemon that exposes a health RPC passes `health_fn`: its answer both fills
    `health` and *defines* liveness (it answered -> it is up). A daemon with only
    a bound port passes none, and a cheap TCP connect to (host, port) defines
    liveness. Either way the caller gets a Probe it can render or poll without a
    try/except -- a failed RPC comes back health=None, a closed port listening=False.
    """
    if health_fn is not None:
        health = health_fn()
        return Probe(listening=health is not None, health=health)
    return Probe(listening=_port_listening(host, port))


def _emit_daemon_status(
    *,
    title: str,
    pid: Optional[int],
    running: bool,
    stale: bool,
    pid_file: Path,
    log_file: Path,
    json_output: bool,
    json_fields: dict,
    table_rows: List[Tuple[str, str]],
) -> None:
    """Render one daemon's status (JSON or table); the command exits 0.

    The running/stale/stopped verdict and the common PID / PID-file / Log-file
    rows are identical for both daemons; `json_fields` and `table_rows` carry the
    per-daemon extras (Flight health for the tensor server, the HTTP endpoint for
    biopb-mcp). `table_rows` are inserted between the PID row and the trailing
    PID-file / Log-file rows, preserving each command's original row order.

    The JSON and not-running paths short-circuit via `typer.Exit(0)`; the
    running-table path returns normally, which typer likewise maps to exit 0.
    """
    if json_output:
        print(
            json.dumps(
                {
                    "running": running,
                    "pid": pid if running else None,
                    "status": "running"
                    if running
                    else ("stale" if stale else "stopped"),
                    **json_fields,
                }
            )
        )
        raise typer.Exit(0)

    table = Table(title=title)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    if not running:
        table.add_row("Status", "Not running (stale PID)" if stale else "Not running")
        if stale:
            table.add_row("PID file", str(pid_file) + " (stale)")
        console.print(table)
        raise typer.Exit(0)

    table.add_row("Status", "Running")
    table.add_row("PID", str(pid))
    for label, value in table_rows:
        table.add_row(label, value)
    table.add_row("PID file", str(pid_file))
    table.add_row("Log file", str(log_file))
    console.print(table)


@server_app.command("status")
def status(
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Path to config file (JSON or TOML)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
    wait: float = typer.Option(
        0.0,
        "--wait",
        "-w",
        help="Seconds to poll for the server to reach SERVING (0 = check once)",
    ),
):
    """Check TensorFlight server daemon status and live health."""
    pid, id_token = _read_pid_record(PID_FILE)
    running = _is_our_daemon(pid, id_token)
    stale = bool(pid and not running)

    # When running, ask the daemon for its Flight health (status + source_count),
    # polling up to --wait for it to finish its initial scan and report SERVING.
    health: Optional[dict] = None
    if running:
        location, token = _resolve_grpc_endpoint(config)
        host, port = _resolve_grpc_hostport(config)
        deadline = time.monotonic() + max(0.0, wait)
        last_report = None
        while True:
            health = _probe_daemon(
                host, port, lambda: _query_health(location, token)
            ).health
            st = health.get("status") if health else None
            n = health.get("source_count") if health else None
            # While waiting, log human-facing progress to stderr so stdout stays
            # clean for --json. The terminal SERVING line is left to the table /
            # JSON (or the caller) so we don't duplicate the final verdict.
            if wait > 0 and st != "SERVING":
                report = (
                    "waiting for the data server to come up..."
                    if st is None
                    else f"data server starting - {n} source(s) found so far..."
                )
                if report != last_report:
                    print(report, file=sys.stderr, flush=True)
                    last_report = report
            if (health and st == "SERVING") or time.monotonic() >= deadline:
                break
            time.sleep(0.5)

    health_status = health.get("status") if health else None
    source_count = health.get("source_count") if health else None

    table_rows: List[Tuple[str, str]] = []
    if running:
        if health:
            table_rows.append(("Health", str(health_status)))
            if source_count is not None:
                table_rows.append(("Sources", str(source_count)))
            if health.get("uptime_seconds") is not None:
                table_rows.append(("Uptime", f"{health.get('uptime_seconds')}s"))
            if health.get("writable") is not None:
                table_rows.append(("Writable", str(health.get("writable"))))
        else:
            table_rows.append(("Health", "unreachable"))
        table_rows.append(("Config", str(config)))

    _emit_daemon_status(
        title="TensorFlight Server Status",
        pid=pid,
        running=running,
        stale=stale,
        pid_file=PID_FILE,
        log_file=_get_log_file(),
        json_output=json_output,
        json_fields={
            "health": health_status,
            "source_count": source_count,
            "writable": health.get("writable") if health else None,
            "uptime_seconds": health.get("uptime_seconds") if health else None,
        },
        table_rows=table_rows,
    )


@server_app.command("logs")
def logs(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Stream new log lines as they are written"
    ),
    lines: int = typer.Option(
        200, "--lines", "-n", help="Number of lines from the end to show (0 = all)"
    ),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        help="Minimum level to show: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
    path: bool = typer.Option(False, "--path", help="Print the log file path and exit"),
):
    """Show the TensorFlight server daemon log."""
    log_file = _get_log_file()
    if path:
        print(log_file)
        raise typer.Exit(0)
    _tail_and_follow(log_file, follow, lines, _validate_level(level), _line_level)


def _fmt_mb(n_bytes: int) -> str:
    """Format a byte count as MB."""
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def _hit_rate(hits: int, misses: int) -> str:
    """Hit rate as a percentage string (guards divide-by-zero)."""
    total = hits + misses
    return f"{(hits / total * 100):.1f}%" if total else "n/a"


def _render_cache_stats(stats: dict) -> None:
    """Render a CacheStats dict (from TensorFlightClient.cache_stats) as tables."""
    g = stats.get
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    hits, misses = g("hits", 0), g("misses", 0)
    table.add_row("Hits", str(hits))
    table.add_row("Misses", str(misses))
    table.add_row("Hit rate", _hit_rate(hits, misses))
    table.add_row("Evictions", str(g("evictions", 0)))
    table.add_row("Pending waits", str(g("pending_waits", 0)))
    table.add_row("Oversized skips", str(g("oversized_skips", 0)))
    table.add_row("Ref-held evictions skipped", str(g("ref_held_evictions_skipped", 0)))
    table.add_row("Entries", str(g("total_entries", 0)))
    table.add_row("Size", _fmt_mb(g("total_bytes", 0)))
    if g("max_entries", 0):
        table.add_row("Max entries", str(g("max_entries")))
    if g("max_bytes", 0):
        table.add_row("Max size", _fmt_mb(g("max_bytes")))
    console.print(table)

    pool_stats = stats.get("pool_stats") or {}
    if pool_stats:
        ptable = Table(title="Per-pool Statistics")
        for col in ("Pool", "Hits", "Misses", "Hit rate", "Segments", "Size"):
            ptable.add_column(
                col,
                style="cyan" if col == "Pool" else "green",
                justify="left" if col == "Pool" else "right",
            )
        for name, p in sorted(pool_stats.items()):
            ptable.add_row(
                name,
                str(p.get("hits", 0)),
                str(p.get("misses", 0)),
                _hit_rate(p.get("hits", 0), p.get("misses", 0)),
                str(p.get("segments", 0)),
                _fmt_mb(p.get("bytes", 0)),
            )
        console.print(ptable)


@server_app.command("cache-stats")
def cache_stats(
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Path to config file (JSON or TOML)"
    ),
    token: Optional[str] = typer.Option(
        None, "--token", help="Access token (or set BIOPB_TENSOR_TOKEN)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
):
    """Show cache hit/miss diagnostics from the running server."""
    pid, id_token = _read_pid_record(PID_FILE)
    if not _is_our_daemon(pid, id_token):
        console.print("[yellow]TensorFlight server is not running.[/yellow]")
        raise typer.Exit(1)

    location, env_token = _resolve_grpc_endpoint(config)
    stats = _query_cache_stats(location, token or env_token)

    if stats is None:
        console.print(
            "[red]Could not retrieve cache stats[/red] "
            "(server unreachable or cache not initialized)."
        )
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(stats))
        raise typer.Exit(0)

    _render_cache_stats(stats)


@server_app.command("migrate-config")
def migrate_config(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file (or dir) to migrate; defaults to ~/.config/biopb",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Report what would happen; write nothing"
    ),
):
    """Migrate a legacy ``biopb.toml`` to the canonical ``biopb.json``.

    JSON is the canonical on-disk format (biopb/biopb#34); TOML stays readable
    through a deprecation window. This converts a legacy TOML config in place --
    reading the raw table (so advanced/unknown keys survive) and writing the
    sibling ``biopb.json`` (plus its schema sidecar), then backing the old TOML
    up to ``biopb.toml.bak``. Settings are preserved verbatim, so a running
    server need not be restarted.
    """
    from ._config_location import (
        CANONICAL_CONFIG_NAME,
        DEFAULT_CONFIG_DIR,
        LEGACY_CONFIG_NAME,
    )

    # Resolve the config directory. --config may point at a file (use its parent)
    # or a directory; with nothing given, use the standard location.
    if config is None:
        config_dir = DEFAULT_CONFIG_DIR
    elif config.is_dir():
        config_dir = config
    else:
        config_dir = config.parent

    toml_path = config_dir / LEGACY_CONFIG_NAME
    json_path = config_dir / CANONICAL_CONFIG_NAME

    if not toml_path.exists():
        if json_path.exists():
            console.print(
                f"[green]Already canonical:[/green] {json_path} is JSON; "
                "nothing to migrate."
            )
        else:
            console.print(
                f"[yellow]No legacy config found[/yellow] at {toml_path} "
                "(and no JSON either); nothing to migrate."
            )
        raise typer.Exit(0)

    # A legacy TOML exists. If a JSON also exists it already shadows the TOML
    # (find_config prefers JSON), so we must NOT overwrite it from the TOML --
    # just retire the stale TOML to clear the both-files shadow warning.
    if json_path.exists():
        backup = toml_path.with_name(toml_path.name + ".bak")
        console.print(
            f"[yellow]Both configs present:[/yellow] {json_path} is already "
            f"canonical and in use; the legacy {toml_path.name} is ignored."
        )
        if dry_run:
            console.print(f"  [dim](dry run)[/dim] would back it up to {backup.name}")
            raise typer.Exit(0)
        toml_path.replace(backup)
        console.print(f"  Retired the legacy TOML -> {backup.name}")
        raise typer.Exit(0)

    # The migration case: TOML only. Read the raw table and write canonical JSON.
    try:
        from biopb_tensor_server.core.config import _read_config_file, save_config
    except Exception as exc:  # noqa: BLE001 - optional dependency
        console.print(
            "[red]Config migration is unavailable:[/red] "
            f"{exc}\n"
            "[yellow]Re-run the BioPB installer to fix.[/yellow]"
        )
        raise typer.Exit(1)

    try:
        data = _read_config_file(toml_path)
    except Exception as exc:  # noqa: BLE001 - surface a parse error cleanly
        console.print(f"[red]Could not read {toml_path}:[/red] {exc}")
        raise typer.Exit(1)

    if dry_run:
        backup = toml_path.with_name(toml_path.name + ".bak")
        console.print(f"[cyan](dry run)[/cyan] would migrate {toml_path}")
        console.print(f"  write  {json_path} (+ schema sidecar)")
        console.print(f"  backup {toml_path.name} -> {backup.name}")
        raise typer.Exit(0)

    try:
        written = save_config(data, toml_path)
    except Exception as exc:  # noqa: BLE001 - write must surface, not crash
        console.print(f"[red]Failed to write {json_path}:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"[green]Migrated[/green] {toml_path} -> {written} "
        f"(old file backed up to {toml_path.name}.bak)."
    )


# ---------------------------------------------------------------------------
# biopb-mcp daemon management (`biopb mcp ...`)
#
# Manages the biopb-mcp MCP server (HTTP/streamable-http transport) as a
# background daemon, mirroring the tensor-server daemon commands. The biopb-mcp
# package is an optional dependency: every subcommand first calls
# _require_biopb_mcp(), which surfaces a clear install hint (rather than a raw
# ImportError) when it is absent. The server itself runs in a child process
# (`python -m biopb_mcp.mcp --transport http`), so this CLI never imports the
# heavy MCP/napari stack - only biopb_mcp._config, lazily, for the default port.
# ---------------------------------------------------------------------------

mcp_app = typer.Typer(
    name="mcp",
    help="biopb-mcp MCP server daemon management (start/stop/restart/status/logs)",
)


def _require_biopb_mcp() -> None:
    """Exit(1) with an install hint if the biopb-mcp package is not importable.

    Checks the import *spec* (not a real import) so the heavy MCP/napari stack is
    never loaded into this CLI process just to gate a command.
    """
    import importlib.util

    if importlib.util.find_spec("biopb_mcp") is None:
        console.print(
            "[red]The 'mcp' commands require the biopb-mcp package, which is "
            "not installed.[/red]\n"
            r"[yellow]Install it with: pip install 'biopb-mcp\[mcp]'[/yellow]"
        )
        raise typer.Exit(1)


def _write_mcp_pid(pid: int):
    """Write the MCP daemon PID (+ its create-time identity token).

    The biopb-mcp daemon also writes this file itself (biopb_mcp.mcp.__main__),
    with the same pid+token format; whoever writes last is authoritative and both
    are read by _read_pid_record.
    """
    _write_pid_file(MCP_PID_FILE, pid, _process_create_time(pid))


def _remove_mcp_pid():
    """Remove the MCP daemon PID file."""
    _remove_pid_file(MCP_PID_FILE)


def _ensure_mcp_dirs():
    """Ensure the MCP PID and log directories exist."""
    MCP_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_mcp_log_file() -> Path:
    """Canonical path the MCP daemon's stdout/stderr is written to.

    Unified with the stdio shim via biopb_mcp._config.get_daemon_log_file(), so
    `mcp start` (which redirects the daemon here) writes the SAME file the shim
    does -- and `mcp logs`/`status` therefore read the right one regardless of
    launcher. Falls back to the in-package default (the same mcp-server.log
    name) if biopb_mcp is somehow unavailable.
    """
    try:
        from biopb_mcp._config import get_daemon_log_file

        return get_daemon_log_file()
    except Exception:
        return MCP_LOG_DIR / "mcp-server.log"


def _resolve_mcp_log_for_read() -> Path:
    """The daemon log to display: the canonical file, or the legacy
    `kernel.log` a pre-unification shim-spawned daemon wrote, if only that one
    exists.

    Keeps a pre-unification log readable instead of falsely reporting "never
    started"; when neither exists, returns the canonical path so the
    not-found message names where the daemon will log.
    """
    canonical = _get_mcp_log_file()
    if canonical.exists():
        return canonical
    legacy = MCP_LOG_DIR / "kernel.log"
    if legacy.exists():
        return legacy
    return canonical


def _mcp_default_port() -> int:
    """Configured MCP transport port (biopb_mcp config), or the 8765 default.

    Reads mcp.transport.port from biopb-mcp's config so `mcp start`/`status`
    honor a user-set port; any failure (package absent, malformed config) falls
    back to the documented default.
    """
    try:
        from biopb_mcp._config import get_setting, load_config

        return int(get_setting(load_config(), "mcp.transport.port"))
    except Exception:
        return MCP_DEFAULT_PORT


def _mcp_url(port: int) -> str:
    """Loopback streamable-http endpoint the MCP daemon serves."""
    return f"http://127.0.0.1:{port}/mcp"


def _port_listening(host: str, port: int, timeout: float = 0.3) -> bool:
    """Whether a TCP connection to (host, port) succeeds - a cheap liveness probe
    for the daemon's HTTP listener (it binds before serving)."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _await_listening(pid: int, host: str, port: int, timeout: float) -> bool:
    """Block until (host, port) accepts a connection, returning True. Returns
    False if the process dies first or `timeout` elapses without the port coming
    up -- a readiness check (did the daemon actually bind?), strictly stronger
    than "is the child process still alive". Callers re-check liveness to tell a
    crash apart from a slow/wedged bind."""
    deadline = time.monotonic() + timeout
    while True:
        if not _is_process_running(pid):
            return False
        if _probe_daemon(host, port).listening:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _mcp_shutdown_sentinel() -> Path:
    """Path of the MCP daemon's shutdown sentinel (Windows graceful stop).

    Must match _shutdown_sentinel_path() in biopb_mcp.mcp.__main__ (kept as a
    literal here, like MCP_PID_FILE itself, to avoid importing biopb_mcp just
    to stop). A single fixed name, not pid-keyed, for the same shim-PID-mismatch
    reason as _win_shutdown_sentinel().
    """
    return MCP_PID_FILE.parent / "mcp-server.stop"


@mcp_app.command("start")
def mcp_start(
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="HTTP transport port (default: biopb-mcp config, else 8765)",
    ),
):
    """Start the biopb-mcp MCP server (HTTP transport) as a background daemon."""
    _require_biopb_mcp()
    _ensure_mcp_dirs()

    existing_pid, existing_token = _read_pid_record(MCP_PID_FILE)
    if _is_our_daemon(existing_pid, existing_token):
        console.print(
            f"[yellow]biopb-mcp server already running (PID {existing_pid})[/yellow]"
        )
        raise typer.Exit(0)
    if existing_pid:
        console.print(
            f"[yellow]Removing stale PID file (process {existing_pid} not running)[/yellow]"
        )
        _remove_mcp_pid()

    resolved_port = port if port is not None else _mcp_default_port()

    # Refuse to start on top of an already-bound port -- the orphan case the PID
    # file cannot see (daemon still serving with a missing PID file, or another
    # session). Otherwise the new process double-binds, fails silently in the
    # log, and leaves a dead process behind the PID file we are about to write.
    if _port_listening("127.0.0.1", resolved_port):
        console.print(f"[red]Port 127.0.0.1:{resolved_port} is already in use.[/red]")
        console.print(
            "It is held by a process biopb is not tracking (no matching PID "
            "file -- an orphaned daemon, or another login session), so "
            "[bold]biopb mcp stop[/bold] cannot reach it. Identify and stop the "
            f"owner (`netstat -ano | findstr {resolved_port}` on Windows, "
            f"`lsof -i :{resolved_port}` on macOS/Linux), then retry."
        )
        raise typer.Exit(1)

    log_file = _get_mcp_log_file()
    _rotate_log(log_file)

    cmd = [
        sys.executable,
        "-m",
        "biopb_mcp.mcp",
        "--transport",
        "http",
        "--port",
        str(resolved_port),
    ]

    console.print("[green]Starting biopb-mcp server...[/green]")
    with open(log_file, "a") as log:
        log.write(f"\n--- Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            env=os.environ.copy(),
            **_detach_kwargs(),
        )

    _write_mcp_pid(process.pid)

    # Wait for the daemon to actually bind its HTTP port -- a readiness check,
    # not just "is the child alive". A bind collision or early crash surfaces as
    # the port never coming up (or the process exiting).
    if not _await_listening(process.pid, "127.0.0.1", resolved_port, 15.0):
        if _is_process_running(process.pid):
            console.print(
                f"[red]biopb-mcp server started but is not listening on "
                f"127.0.0.1:{resolved_port} after 15s.[/red]"
            )
            console.print(
                "Check the logs; run [bold]biopb mcp stop[/bold] if it is wedged."
            )
        else:
            console.print("[red]Failed to start biopb-mcp server[/red]")
            _remove_mcp_pid()
        console.print(f"Check logs: {log_file}")
        raise typer.Exit(1)

    console.print(f"[green]biopb-mcp server started (PID {process.pid})[/green]")
    console.print(f"  MCP:  {_mcp_url(resolved_port)}")
    console.print(f"  Logs: {log_file}")


@mcp_app.command("stop")
def mcp_stop(
    timeout: int = typer.Option(
        10, "--timeout", "-t", help="Seconds to wait for graceful shutdown"
    ),
):
    """Stop the biopb-mcp server daemon."""
    _require_biopb_mcp()
    pid, token = _read_pid_record(MCP_PID_FILE)

    if not pid:
        console.print("[yellow]No biopb-mcp server running[/yellow]")
        raise typer.Exit(0)
    if not _is_our_daemon(pid, token):
        console.print(
            f"[yellow]Process {pid} not running, cleaning up PID file[/yellow]"
        )
        _remove_mcp_pid()
        raise typer.Exit(0)

    console.print(f"[green]Stopping biopb-mcp server (PID {pid})...[/green]")
    if _stop_daemon(
        pid,
        timeout,
        token,
        sentinel=_mcp_shutdown_sentinel(),
        remove_pid=_remove_mcp_pid,
    ):
        console.print("[green]biopb-mcp server stopped[/green]")
    else:
        console.print(f"[yellow]Did not stop within {timeout}s; force killed[/yellow]")
    raise typer.Exit(0)


@mcp_app.command("restart")
def mcp_restart(
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="HTTP transport port (default: config, else 8765)"
    ),
    timeout: int = typer.Option(
        10, "--timeout", "-t", help="Seconds to wait for graceful shutdown"
    ),
):
    """Restart the biopb-mcp server daemon."""
    _require_biopb_mcp()
    pid, token = _read_pid_record(MCP_PID_FILE)
    if _is_our_daemon(pid, token):
        console.print(f"[green]Stopping biopb-mcp server (PID {pid})...[/green]")
        _stop_daemon(
            pid,
            timeout,
            token,
            sentinel=_mcp_shutdown_sentinel(),
            remove_pid=_remove_mcp_pid,
        )
        time.sleep(1)
    mcp_start(port=port)


@mcp_app.command("status")
def mcp_status(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
):
    """Check biopb-mcp server daemon status and HTTP liveness."""
    _require_biopb_mcp()
    pid, token = _read_pid_record(MCP_PID_FILE)
    running = _is_our_daemon(pid, token)
    stale = bool(pid and not running)

    port = _mcp_default_port()
    listening = _probe_daemon("127.0.0.1", port).listening if running else False

    _emit_daemon_status(
        title="biopb-mcp Server Status",
        pid=pid,
        running=running,
        stale=stale,
        pid_file=MCP_PID_FILE,
        log_file=_resolve_mcp_log_for_read(),
        json_output=json_output,
        json_fields={
            "port": port,
            "listening": listening,
            "url": _mcp_url(port) if running else None,
        },
        table_rows=[
            ("Port", str(port)),
            ("MCP", _mcp_url(port)),
            # `running` (process alive) but not listening is ambiguous: still
            # starting, or the /mcp transport died under it (biopb/biopb#383).
            # Don't assert "not bound yet" -- point at the recovery either way.
            (
                "Listening",
                "yes"
                if listening
                else "no (starting up, or transport died -- try 'biopb mcp restart')",
            ),
        ],
    )


@mcp_app.command("logs")
def mcp_logs(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Stream new log lines as they are written"
    ),
    lines: int = typer.Option(
        200, "--lines", "-n", help="Number of lines from the end to show (0 = all)"
    ),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        help="Minimum level to show: DEBUG, INFO, WARNING, ERROR, CRITICAL "
        "(best-effort; the MCP daemon log mixes formats)",
    ),
    path: bool = typer.Option(False, "--path", help="Print the log file path and exit"),
):
    """Show the biopb-mcp server daemon log."""
    _require_biopb_mcp()
    log_file = _resolve_mcp_log_for_read()
    if path:
        print(log_file)
        raise typer.Exit(0)
    _tail_and_follow(log_file, follow, lines, _validate_level(level), _mcp_line_level)


@mcp_app.command("view")
def mcp_view(
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="MCP port for an optional agent to attach (default: dynamic, "
        "OS-assigned — printed on startup).",
    ),
):
    """Open the napari viewer in the foreground (agentless).

    Runs a biopb-mcp session in *this* terminal: the napari window opens
    immediately and the process blocks until Ctrl-C. Unlike `biopb mcp start`
    (a detached background daemon managed by stop/status), this is a foreground,
    user-owned viewer that writes no PID file. It still serves /mcp on the chosen
    (default dynamic) port, so an AI agent may optionally attach to the same live
    session.

    Implemented by running `biopb-mcp --view` as a foreground child that shares
    this terminal's stdio and process group, so Ctrl-C reaches it directly (its
    own SIGINT handler reaps the kernel/viewer). This CLI stays free of the heavy
    napari/Qt import — it only launches and waits.
    """
    _require_biopb_mcp()
    resolved_port = 0 if port is None else port
    cmd = [
        sys.executable,
        "-m",
        "biopb_mcp.mcp",
        "--view",
        "--port",
        str(resolved_port),
    ]
    console.print("[green]Opening biopb-mcp viewer (Ctrl-C to stop)...[/green]")
    # Foreground: NO _detach_kwargs — inherit this terminal's stdio and stay in
    # its process group so Ctrl-C (SIGINT / CTRL_C_EVENT) reaches the launcher.
    try:
        process = subprocess.Popen(cmd, env=os.environ.copy())
    except OSError as exc:
        console.print(f"[red]Could not launch the viewer:[/red] {exc}")
        raise typer.Exit(1)
    try:
        raise typer.Exit(process.wait())
    except KeyboardInterrupt:
        # Ctrl-C already reached the child via the shared group; give it a moment
        # to tear the kernel/viewer down, then force-reap if it overruns.
        try:
            process.wait(timeout=20)
        except Exception:
            process.kill()
        raise typer.Exit(0)


app.add_typer(mcp_app, name="mcp")


# ---------------------------------------------------------------------------
# quick-start: Windows Defender exclusion for the biopb install (issue #384)
# ---------------------------------------------------------------------------
# Windows Defender real-time scanning of biopb's DLLs / .pyd / .pyc on every
# launch is the single largest first-start tax on Windows (see #384). Excluding
# the install trees from scanning removes it -- both the uv tool env (deps) and
# the base Python it runs on (stdlib .pyd + pythonXY.dll), which are separate
# directories for a uv tool venv. This is the *privileged* half of #384 -- it
# needs admin -- so it lives here as an opt-in command, separate from the
# admin-free bytecode precompile the installer already does for everyone.


def _is_windows() -> bool:
    """Whether we're on Windows.

    A function (not an inline `os.name == "nt"`) so tests can simulate Windows
    without monkeypatching the global `os.name` -- which `pathlib` reads to pick
    WindowsPath vs PosixPath, so mutating it breaks every `Path(...)` in the
    process (a WindowsPath can't be instantiated on POSIX before Python 3.13).
    """
    return os.name == "nt"


def _defender_targets() -> List[str]:
    """The install trees to exclude -- every tree this interpreter reads at startup.

    A uv tool venv is a venv pointing at a *separate* base Python, so two distinct
    trees get read + Defender-scanned on every launch (verified against a real
    installer-based install, #384):

    * ``sys.prefix``      -- the tool env's ``Lib\\site-packages`` (numpy / PyQt6 /
      grpcio / scipy / ... -- the heavy deps; ``Qt6\\bin`` alone is ~100 MB of DLLs).
      It's ~half ``.pyd`` and ~half ``.dll``, so we exclude the *directory*, not
      ``*.pyd``.
    * ``sys.base_prefix`` -- the base interpreter + stdlib ``.pyd`` (``_ssl``,
      ``_socket``, ``_ctypes``, ...) + ``pythonXY.dll``, loaded on every start.

    They differ for a uv tool venv and coincide for a plain (non-venv) install, so
    we dedup. Both come from the *running interpreter* -- never hardcode the uv
    path, because the base Python can live outside ``%LOCALAPPDATA%\\uv`` entirely
    (the installer's ``--python`` may pick a pre-existing interpreter). Sorted so
    the elevated snippet and the status read agree on order.
    """
    return sorted({str(Path(p).resolve()) for p in (sys.prefix, sys.base_prefix)})


def _ps_string_array(paths: List[str]) -> str:
    """A PowerShell array literal of single-quote-safe path strings (`'a', 'b'`)."""
    return ", ".join("'" + p.replace("'", "''") + "'" for p in paths)


def _run_elevated_ps(inner: str) -> int:
    """Run a PowerShell snippet elevated (one UAC prompt); return its exit code.

    Writes the snippet to a temp .ps1 and launches it via
    `Start-Process -Verb RunAs -Wait -PassThru`, propagating the elevated
    process's exit code. A nonzero code also covers the launch itself failing --
    most commonly the user declining the UAC prompt (Start-Process then throws,
    so the outer shell exits nonzero).
    """
    import tempfile

    # utf-8-sig, not plain utf-8: Windows PowerShell 5.1 reads a BOM-less script
    # as the ANSI code page, so a non-ASCII install path (e.g. an accented
    # username in sys.prefix) would be misread -- and since the same misread $p
    # feeds both Add-MpPreference and the -contains verify, the script would
    # exit 0 while excluding the wrong path. The BOM pins UTF-8 decoding.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".ps1", delete=False, encoding="utf-8-sig"
    ) as f:
        f.write(inner)
        script = f.name
    ps_script = script.replace("'", "''")  # single-quote-safe for the launcher
    try:
        launcher = (
            "$p = Start-Process powershell -Verb RunAs -Wait -PassThru "
            "-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',"
            f"'-File','{ps_script}'; exit $p.ExitCode"
        )
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", launcher]
        ).returncode
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def _defender_exclusion(targets: List[str], *, add: bool) -> None:
    """Add/remove Defender exclusions for every tree in `targets` in ONE elevated
    session (a single UAC prompt), then VERIFY.

    Admin is necessary but not sufficient: Tamper Protection (consumer) or
    Intune/GPO (managed) can silently no-op the write even when elevated. So the
    elevated snippet re-reads Get-MpPreference and confirms *every* path reached
    the intended state (exit 0 = all took, 3 = at least one blocked) rather than
    assuming success from a clean return.
    """
    verb = "Add" if add else "Remove"
    # Verify each path reached its intended end-state; fail (exit 3) if any didn't.
    fail = "-not ($excl -contains $p)" if add else "($excl -contains $p)"
    ps_array = _ps_string_array(targets)
    # Placeholder substitution (not an f-string) so PowerShell's literal { } blocks
    # don't collide with brace escaping. Paths are substituted LAST so a path can
    # never be re-interpreted as one of the other placeholders.
    inner = (
        "$ErrorActionPreference = 'Stop'\n"
        "$paths = @(__PATHS__)\n"
        "foreach ($p in $paths) {\n"
        "  try { __VERB__-MpPreference -ExclusionPath $p }\n"
        '  catch { Write-Host "FAILED: $($_.Exception.Message)"; exit 2 }\n'
        "}\n"
        "$excl = (Get-MpPreference).ExclusionPath\n"
        "foreach ($p in $paths) {\n"
        '  if (__FAIL__) { Write-Host "MISSING: $p"; exit 3 }\n'
        "}\n"
        "exit 0\n"
    )
    inner = (
        inner.replace("__VERB__", verb)
        .replace("__FAIL__", fail)
        .replace("__PATHS__", ps_array)
    )

    rc = _run_elevated_ps(inner)
    joined = "\n".join(f"  {p}" for p in targets)
    if rc == 0:
        console.print(
            f"[green]Defender exclusion {'added' if add else 'removed'}:[/green]\n{joined}"
        )
        if add:
            console.print("  biopb should now start faster on this machine.")
        return
    if rc == 2:
        console.print(
            f"[red]Could not {verb.lower()} the Defender exclusion[/red] "
            f"(the {verb}-MpPreference call failed)."
        )
    elif rc == 3:
        console.print(
            "[yellow]Defender exclusion did not take[/yellow] -- blocked by Tamper "
            "Protection or your organization's policy. This is expected on managed "
            "machines; the bytecode precompile still helps."
        )
    else:
        console.print(
            "[red]Could not change the Defender exclusion[/red] "
            "(elevation was declined or failed)."
        )
    raise typer.Exit(1)


def _defender_status(targets: List[str]) -> None:
    """Print whether the biopb trees are currently Defender exclusions
    (best-effort, no admin).

    Get-MpPreference is usually readable by a normal user; when it isn't we say
    'unknown' rather than guess. With more than one tree the state can also be
    PARTIAL (some excluded, some not -- e.g. after upgrading from the earlier
    single-path version that excluded only sys.prefix).
    """
    ps_array = _ps_string_array(targets)
    inner = (
        "$ErrorActionPreference='Stop'\n"
        "try {\n"
        "  $excl = (Get-MpPreference).ExclusionPath\n"
        "  $paths = @(__PATHS__)\n"
        "  $on = 0\n"
        "  foreach ($p in $paths) { if ($excl -contains $p) { $on++ } }\n"
        "  if ($on -eq $paths.Count) { Write-Host 'ON' }\n"
        "  elseif ($on -eq 0) { Write-Host 'OFF' }\n"
        "  else { Write-Host 'PARTIAL' }\n"
        "} catch { Write-Host 'UNKNOWN' }\n"
    ).replace("__PATHS__", ps_array)
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", inner],
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        out = "UNKNOWN"

    joined = "\n".join(f"  {p}" for p in targets)
    if out == "ON":
        console.print("[green]Defender exclusion is enabled[/green] for:")
        console.print(joined)
        console.print("  Remove it with: biopb quick-start --disable")
    elif out == "OFF":
        console.print("Defender exclusion is [yellow]not set[/yellow] for:")
        console.print(joined)
        console.print(
            "  Enable it for a faster startup (needs admin): biopb quick-start --enable"
        )
    elif out == "PARTIAL":
        console.print(
            "[yellow]Defender exclusion is only partially set[/yellow] "
            "-- some biopb trees are excluded, some aren't:"
        )
        console.print(joined)
        console.print("  Complete it (needs admin): biopb quick-start --enable")
    else:
        console.print(
            "[yellow]Could not read the Defender exclusion state[/yellow] "
            "(Get-MpPreference unavailable). Enable with: biopb quick-start --enable"
        )


@app.command("quick-start", hidden=not _is_windows())
def quick_start(
    enabled: Optional[bool] = typer.Option(
        None,
        "--enable/--disable",
        help="Enable (add) or disable (remove) the Defender exclusion; "
        "omit to show the current status.",
    ),
):
    """Speed up biopb startup on Windows via a Defender exclusion (issue #384).

    Windows Defender rescans biopb's DLLs / .pyd / .pyc on every launch, which
    dominates the first-start wait. This adds (or removes, with --disable) Defender
    exclusions for the biopb install trees -- both the tool env (heavy deps) and
    the base Python it runs on (stdlib .pyd + pythonXY.dll) -- so those files
    aren't rescanned. It needs admin -- one UAC prompt -- and is fully reversible.
    Windows only.
    """
    if not _is_windows():
        console.print(
            "[yellow]quick-start is Windows-only[/yellow] -- Defender exclusions "
            "don't apply on this platform (nothing to do)."
        )
        raise typer.Exit(0)

    targets = _defender_targets()
    if enabled is None:
        _defender_status(targets)
        return
    _defender_exclusion(targets, add=enabled)


if __name__ == "__main__":
    app()
