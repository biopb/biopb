"""Top-level CLI for BioPB."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional

import typer

from biopb.tensor.cli import app as tensor_app
from biopb.image.cli import app as image_app

console = Console()

app = typer.Typer(
    name="biopb",
    help="BioPB: open protobuf/gRPC protocols for biomedical image processing",
)
console = Console()

# TensorFlight client diagnostics
app.add_typer(tensor_app, name="tensor", help="TensorFlight client diagnostics")

# ProcessImage client operations
app.add_typer(image_app, name="image", help="ProcessImage client operations")

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
    try:
        os.kill(pid, 0)  # Signal 0 just checks if process exists
        return True
    except OSError:
        return False


def _get_log_file() -> Path:
    """Get log file path."""
    return LOG_DIR / "tensor-server.log"


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
    import secrets

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

    # Generate token if not provided
    if not token:
        token = os.environ.get("BIOPB_TENSOR_TOKEN")
    if not token:
        token = secrets.token_urlsafe(32)
        console.print(f"[bold green]Generated access token:[/bold green] {token}")

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
        "--token", str(token),
    ]
    if static_dir and static_dir.exists():
        cmd.extend(["--static-dir", str(static_dir)])

    log_file = _get_log_file()

    # Start subprocess
    console.print(f"[green]Starting TensorFlight server...[/green]")
    console.print(f"  Config: {config}")

    with open(log_file, "a") as log:
        log.write(f"\n--- Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True,  # Detach from current process group
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
    console.print(f"  HTTP: http://{web_host}:{web_port}/?token={str(token)}")
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

    # Send SIGTERM for graceful shutdown
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        console.print(f"[red]Failed to send SIGTERM: {e}[/red]")
        raise typer.Exit(1)

    # Wait for process to terminate
    for i in range(timeout):
        if not _is_process_running(pid):
            console.print("[green]TensorFlight server stopped[/green]")
            _remove_pid()
            raise typer.Exit(0)
        time.sleep(1)

    # Force kill if still running
    console.print(f"[yellow]Process still running after {timeout}s, sending SIGKILL[/yellow]")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    time.sleep(0.5)
    _remove_pid()
    console.print("[green]TensorFlight server stopped (forced)[/green]")


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
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

        for i in range(timeout):
            if not _is_process_running(pid):
                break
            time.sleep(1)

        if _is_process_running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        _remove_pid()
        time.sleep(1)

    # Start with same options
    start(config=config, static_dir=static_dir, web_port=web_port, web_host=web_host, log_level=log_level, token=token)


@server_app.command("status")
def status():
    """Check TensorFlight server daemon status."""
    pid = _read_pid()

    table = Table(title="TensorFlight Server Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    if not pid:
        table.add_row("Status", "Not running")
        console.print(table)
        raise typer.Exit(0)

    if _is_process_running(pid):
        table.add_row("Status", "Running")
        table.add_row("PID", str(pid))
        table.add_row("PID file", str(PID_FILE))
        table.add_row("Log file", str(_get_log_file()))
    else:
        table.add_row("Status", "Not running (stale PID)")
        table.add_row("PID file", str(PID_FILE) + " (stale)")

    console.print(table)


if __name__ == "__main__":
    app()