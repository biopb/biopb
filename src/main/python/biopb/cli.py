"""Top-level CLI for BioPB."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional, Tuple

import typer

console = Console()

app = typer.Typer(
    name="biopb",
    help="BioPB: open protobuf/gRPC protocols for biomedical image processing",
)
console = Console()


def _add_optional_typer(name: str, import_path: str, help: str) -> None:
    """Register a subcommand whose imports may fail.

    The tensor/image subcommands pull in optional dependencies (installed via
    biopb[tensor]) that may be absent or broken (e.g. a transient
    numcodecs/zarr ImportError). When that happens we still want the rest of
    the CLI (version, server management) to work, so we register a stub that
    surfaces the error only when the subcommand is actually invoked.
    """
    import importlib
    from typing import List

    try:
        module = importlib.import_module(import_path)
        app.add_typer(getattr(module, "app"), name=name, help=help)
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
DEFAULT_CONFIG = Path.home() / ".config" / "biopb" / "biopb.toml"
DEFAULT_WEBAPP = Path.home() / ".local" / "share" / "biopb" / "webapp"

@app.command()
def version():
    """Show version information."""
    try:
        from biopb import __version__ as biopb_version
    except Exception:
        biopb_version = "unknown"

    console.print(f"biopb: {biopb_version}")

def _ensure_dirs():
    """Ensure required directories exist."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid() -> Optional[int]:
    """Read PID from file, return None if missing or invalid."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        return pid
    except (ValueError, IOError):
        return None


def _write_pid(pid: int):
    """Write PID to file."""
    _ensure_dirs()
    PID_FILE.write_text(str(pid))


def _remove_pid():
    """Remove PID file."""
    if PID_FILE.exists():
        PID_FILE.unlink()


def _is_process_running(pid: int) -> bool:
    """Check if process with PID is running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # On Windows os.kill(pid, 0) is NOT a liveness probe: signal value 0 is
        # signal.CTRL_C_EVENT, so os.kill would call GenerateConsoleCtrlEvent and
        # deliver a real Ctrl+C to the console process group (killing the daemon
        # we just started, mid-import). Query the process handle instead.
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # Signal 0 just checks if process exists
        return True
    except OSError:
        return False


# Diagnostic from the most recent _win_request_shutdown() failure, surfaced by
# _graceful_stop() so a Windows user can see why graceful stop fell back to force-kill.
_LAST_WIN_SHUTDOWN_DIAG: Optional[str] = None


def _win_shutdown_sentinel() -> Path:
    """Path of the shutdown sentinel file (Windows graceful stop).

    Must match shutdown_sentinel_path() in biopb_tensor_server.http_server (kept
    as a literal here to avoid importing heavy server deps just to stop). A single
    fixed name - NOT keyed by PID: on Windows the daemon's os.getpid() can differ
    from the PID `start` recorded (Store-Python/uv launcher shims), so a pid in
    the name would make stop and the daemon disagree. One daemon, one sentinel.
    """
    return PID_FILE.parent / "tensor-server.stop"


def _win_request_shutdown() -> bool:
    """Windows: ask the daemon to shut down gracefully. Returns True if the
    request was delivered (not whether the process has exited yet).

    The daemon is a windowless background process in its own process group, so it
    has no console to receive a CTRL_BREAK, and Win32 named events are brittle
    across sessions/elevation. We instead drop a sentinel *file* the daemon polls
    for (see _install_windows_shutdown_listener in biopb_tensor_server.http_server);
    a file under the user's biopb dir is unambiguous regardless of how either
    process was launched.
    """
    global _LAST_WIN_SHUTDOWN_DIAG
    try:
        _ensure_dirs()
        _win_shutdown_sentinel().write_text("stop")
        return True
    except OSError as e:
        _LAST_WIN_SHUTDOWN_DIAG = f"could not write shutdown sentinel: {e}"
        return False


def _win_remove_sentinel() -> None:
    """Remove the shutdown sentinel (best effort), so it doesn't linger after a
    force-kill where the daemon never consumed it."""
    try:
        _win_shutdown_sentinel().unlink()
    except OSError:
        pass


def _request_graceful_stop(pid: int) -> bool:
    """Ask the daemon to shut down gracefully. Returns whether the request was
    delivered (not whether the process has exited yet)."""
    if sys.platform == "win32":
        return _win_request_shutdown()
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def _graceful_stop(pid: int, timeout: int) -> bool:
    """Stop a running daemon: request graceful shutdown, wait up to `timeout`
    seconds, then force-kill. Removes the PID file. Returns True if it exited
    gracefully, False if it had to be force-killed. Assumes `pid` is running.
    """
    delivered = _request_graceful_stop(pid)
    if not delivered and sys.platform == "win32" and _LAST_WIN_SHUTDOWN_DIAG:
        console.print(
            f"[yellow]Graceful stop unavailable ({_LAST_WIN_SHUTDOWN_DIAG}); "
            f"force killing.[/yellow]"
        )

    graceful = False
    for _ in range(timeout):
        if not _is_process_running(pid):
            graceful = True
            break
        time.sleep(1)

    if not graceful:
        # Force kill. signal.SIGKILL is POSIX-only; on Windows fall back to
        # SIGTERM, which os.kill maps to an unconditional TerminateProcess.
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass
        time.sleep(0.5)

    _remove_pid()
    if sys.platform == "win32":
        _win_remove_sentinel()  # tidy up if the daemon never consumed it
    return graceful


def _get_log_file() -> Path:
    """Get log file path."""
    return LOG_DIR / "tensor-server.log"


# Severity ranks for the `--level` filter, matching the daemon's log format
# `[asctime] LEVEL name: message` (DEFAULT_LOG_FORMAT in
# biopb_tensor_server.logging_config).
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


def _filter_lines(lines, min_level: Optional[str]):
    """Keep lines at or above `min_level`. With min_level None, keep all.

    Off-format lines (no parseable level) inherit the previous line's keep/drop
    decision, so a kept WARNING record carries its traceback continuation lines
    along and a dropped INFO record takes its continuations with it. The initial
    decision (before any leveled line) is keep.
    """
    if min_level is None:
        return list(lines)
    threshold = _LOG_LEVELS[min_level]
    kept = []
    keeping = True
    for line in lines:
        lvl = _line_level(line)
        if lvl is not None:
            keeping = _LOG_LEVELS[lvl] >= threshold
        if keeping:
            kept.append(line)
    return kept


def _rotate_log(log_file: Path, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    """Rotate log file if it exceeds max_bytes, keeping up to backup_count backups."""
    if not log_file.exists() or log_file.stat().st_size < max_bytes:
        return
    for i in range(backup_count - 1, 0, -1):
        src = log_file.parent / f"{log_file.name}.{i}"
        dst = log_file.parent / f"{log_file.name}.{i + 1}"
        if src.exists():
            src.rename(dst)
    log_file.rename(log_file.parent / f"{log_file.name}.1")


@server_app.command("start")
def start(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to TOML config file",
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

    # Check if already running
    existing_pid = _read_pid()
    if existing_pid and _is_process_running(existing_pid):
        console.print(f"[yellow]TensorFlight server already running (PID {existing_pid})[/yellow]")
        raise typer.Exit(0)

    # Clean up stale PID file
    if existing_pid:
        console.print(f"[yellow]Removing stale PID file (process {existing_pid} not running)[/yellow]")
        _remove_pid()

    # Token: skip only when both web and gRPC are bound to localhost
    _LOCALHOST_ADDRS = {"127.0.0.1", "localhost", "::1"}
    if not token:
        token = os.environ.get("BIOPB_TENSOR_TOKEN")

    grpc_host = "0.0.0.0"  # safe default: assume public if config unreadable
    if config.exists():
        try:
            from biopb_tensor_server.config import load_config as _load_server_config
            grpc_host = _load_server_config(config).host
        except Exception:
            pass

    local_only = not token and web_host in _LOCALHOST_ADDRS and grpc_host in _LOCALHOST_ADDRS
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
        "--config", str(config),
        "--web-port", str(web_port),
        "--web-host", str(web_host),
        "--log-level", str(log_level),
    ]
    if static_dir and static_dir.exists():
        cmd.extend(["--static-dir", str(static_dir)])

    # Start subprocess
    console.print(f"[green]Starting TensorFlight server...[/green]")
    console.print(f"  Config: {config}")

    # Detach the daemon from the launching console/process group so it survives
    # this command returning and ignores console control events (Ctrl+C, close).
    detach_kwargs: dict = {}
    if sys.platform == "win32":
        # start_new_session (setsid) is POSIX-only and a silent no-op on Windows.
        # Use creationflags instead. CREATE_NO_WINDOW runs the daemon without a
        # console *window*, so no visible console pops up (unlike DETACHED_PROCESS,
        # where a later console-API call could AllocConsole a *visible* window).
        # CREATE_NEW_PROCESS_GROUP makes it a group leader so the launching
        # terminal's Ctrl+C doesn't propagate to it. Because it has no console the
        # terminal can interact with, graceful `stop` can't use a console control
        # event - it uses a named Win32 event instead (see _win_set_shutdown_event
        # and biopb_tensor_server.http_server._install_windows_shutdown_listener).
        detach_kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        detach_kwargs["start_new_session"] = True  # setsid(): own process group

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

    # Brief wait to check if it started successfully
    time.sleep(0.5)
    if not _is_process_running(process.pid):
        console.print("[red]Failed to start TensorFlight server[/red]")
        console.print(f"Check logs: {log_file}")
        _remove_pid()
        raise typer.Exit(1)

    console.print(f"[green]TensorFlight server started (PID {process.pid})[/green]")
    url = f"http://{web_host}:{web_port}/" + (f"?token={token}" if token else "")
    console.print(f"  HTTP: {url}")
    console.print(f"  gRPC: grpc://127.0.0.1:8815")
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
    pid = _read_pid()

    if not pid:
        console.print("[yellow]No TensorFlight server running[/yellow]")
        raise typer.Exit(0)

    if not _is_process_running(pid):
        console.print(f"[yellow]Process {pid} not running, cleaning up PID file[/yellow]")
        _remove_pid()
        raise typer.Exit(0)

    console.print(f"[green]Stopping TensorFlight server (PID {pid})...[/green]")

    if _graceful_stop(pid, timeout):
        console.print("[green]TensorFlight server stopped[/green]")
    else:
        console.print(
            f"[yellow]Did not stop within {timeout}s; force killed[/yellow]"
        )
    raise typer.Exit(0)


@server_app.command("restart")
def restart(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to TOML config file",
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
    # Stop first
    pid = _read_pid()
    if pid and _is_process_running(pid):
        console.print(f"[green]Stopping TensorFlight server (PID {pid})...[/green]")
        _graceful_stop(pid, timeout)
        time.sleep(1)

    # Start with same options
    start(config=config, static_dir=static_dir, web_port=web_port, web_host=web_host, log_level=log_level, token=token)


def _resolve_grpc_endpoint(config: Path) -> Tuple[str, Optional[str]]:
    """Best-effort gRPC endpoint + token for a running server's health query.

    Reads host/port from the TOML config (defaults 127.0.0.1:8815); a server
    bound to 0.0.0.0/:: is reached over loopback. The token comes from
    BIOPB_TENSOR_TOKEN if set -- localhost-only daemons run without one.
    """
    host, port = "127.0.0.1", 8815
    if config and config.exists():
        try:
            from biopb_tensor_server.config import load_config as _load_server_config
            cfg = _load_server_config(config)
            host = cfg.host or host
            port = int(cfg.port or port)
        except Exception:
            pass
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    token = os.environ.get("BIOPB_TENSOR_TOKEN") or None
    return f"grpc://{host}:{port}", token


def _query_health(location: str, token: Optional[str]) -> Optional[dict]:
    """Return the server's Flight health dict, or None if unreachable."""
    try:
        from biopb.tensor.client import TensorFlightClient
    except Exception:
        return None
    client = None
    try:
        client = TensorFlightClient(location, cache_bytes=0, token=token)
        return client.health_check()
    except Exception:
        return None
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


@server_app.command("status")
def status(
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Path to TOML config file"
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
    pid = _read_pid()
    running = bool(pid and _is_process_running(pid))
    stale = bool(pid and not running)

    # When running, ask the daemon for its Flight health (status + source_count),
    # polling up to --wait for it to finish its initial scan and report SERVING.
    health: Optional[dict] = None
    if running:
        location, token = _resolve_grpc_endpoint(config)
        deadline = time.monotonic() + max(0.0, wait)
        last_report = None
        while True:
            health = _query_health(location, token)
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

    if json_output:
        print(
            json.dumps(
                {
                    "running": running,
                    "pid": pid if running else None,
                    "status": "running"
                    if running
                    else ("stale" if stale else "stopped"),
                    "health": health_status,
                    "source_count": source_count,
                    "writable": health.get("writable") if health else None,
                    "uptime_seconds": health.get("uptime_seconds") if health else None,
                }
            )
        )
        raise typer.Exit(0)

    table = Table(title="TensorFlight Server Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    if not running:
        table.add_row("Status", "Not running (stale PID)" if stale else "Not running")
        if stale:
            table.add_row("PID file", str(PID_FILE) + " (stale)")
        console.print(table)
        raise typer.Exit(0)

    table.add_row("Status", "Running")
    table.add_row("PID", str(pid))
    if health:
        table.add_row("Health", str(health_status))
        if source_count is not None:
            table.add_row("Sources", str(source_count))
        if health.get("uptime_seconds") is not None:
            table.add_row("Uptime", f"{health.get('uptime_seconds')}s")
        if health.get("writable") is not None:
            table.add_row("Writable", str(health.get("writable")))
    else:
        table.add_row("Health", "unreachable")
    table.add_row("Config", str(config))
    table.add_row("PID file", str(PID_FILE))
    table.add_row("Log file", str(_get_log_file()))
    console.print(table)


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
    path: bool = typer.Option(
        False, "--path", help="Print the log file path and exit"
    ),
):
    """Show the TensorFlight server daemon log."""
    log_file = _get_log_file()

    if path:
        print(log_file)
        raise typer.Exit(0)

    min_level: Optional[str] = None
    if level is not None:
        min_level = level.upper()
        if min_level not in _LOG_LEVELS:
            console.print(
                f"[red]Invalid --level '{level}'.[/red] "
                f"Choose one of: {', '.join(_LOG_LEVELS)}"
            )
            raise typer.Exit(1)

    if not log_file.exists():
        console.print(
            f"[yellow]No log file at {log_file} - has the server ever started?[/yellow]"
        )
        raise typer.Exit(0)

    # Daemon logs rotate at 10 MB (see _rotate_log / RotatingFileHandler), so the
    # current file is small enough to read whole and slice - no seek-based tail.
    existing = log_file.read_text(errors="replace").splitlines()
    tail = existing if lines <= 0 else existing[-lines:]
    for line in _filter_lines(tail, min_level):
        print(line)

    if not follow:
        raise typer.Exit(0)

    # Follow: poll for appended lines, reopening if the file is rotated or
    # truncated out from under us (a `restart` rotates it mid-follow). Track the
    # inode + size so a replaced or shrunk file restarts from the top.
    try:
        f = open(log_file, "r", errors="replace")
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
                for line in _filter_lines(parts, min_level):
                    print(line, flush=True)
                continue
            # No new data: check whether the file was rotated/truncated.
            try:
                st = os.stat(log_file)
            except OSError:
                st = None
            if st is not None and (
                st.st_ino != last_ino or st.st_size < f.tell()
            ):
                f.close()
                f = open(log_file, "r", errors="replace")
                last_ino = os.fstat(f.fileno()).st_ino
                carry = ""
                continue
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    raise typer.Exit(0)


if __name__ == "__main__":
    app()