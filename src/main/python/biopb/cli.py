"""Top-level CLI for BioPB."""

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from . import _agents, _endpoints, _locations, _web_auth
from ._endpoints import (
    flight_port_for as _flight_port,
    sidecar_port_for as _sidecar_port,
)
from ._lifecycle.daemon import (
    detach_kwargs as _detach_kwargs,
    is_our_daemon as _is_our_daemon,
    read_pid_record as _read_pid_record,
    remove_pid_file as _remove_pid_file,
    stop_daemon as _stop_daemon,
    write_pid_file as _write_pid_file,
)
from ._lifecycle.file_lock import LockTimeout, file_lock
from ._lifecycle.proc import (
    is_process_running as _is_process_running,
    process_create_time as _process_create_time,
)
from ._locations import DEFAULT_CONFIG_DIR, find_config

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
_add_optional_typer(
    "tensor",
    "biopb.tensor.cli",
    "Query a TensorFlight data plane (sources, tensors, stats, cache).",
)

# ProcessImage client operations
_add_optional_typer("image", "biopb.image.cli", "Call ProcessImage algorithm servers.")

# The `biopb server` group is gone (biopb/biopb#615). Its lifecycle commands went
# first, when the control plane took over the data-plane process; the two that
# outlived them were not a group: `cache-stats` is a Flight query, so it moved to
# `biopb tensor cache-stats` beside the other queries, and `migrate-config` needs
# biopb-tensor-server to do anything at all, so it moved to
# `biopb-tensor-server migrate-config` and left the SDK.

# Daemon management constants. On-disk locations come from the shared
# `_locations` module (XDG-aware): the installed webapp bundle is a portable
# asset (data tree); logs / pid / sentinels are per-machine state (state tree).
DEFAULT_WEBAPP = _locations.webapp_dir()

# Default config path, preferring JSON over legacy TOML and warning when both
# exist. Shared with biopb-tensor-server and biopb-mcp via the (dependency-light)
# core module, so resolving this typer Option default does not import the heavy
# server config module (biopb/biopb#34).
DEFAULT_CONFIG = find_config()

# biopb-control (control plane) management. The control plane is a separate, lean package
# (`biopb-control`) started as `python -m biopb_control run` by `biopb control start`;
# the lifecycle plumbing (pidfile / detach / stop-sentinel) lives here, reused
# from the tensor-server / mcp daemons, so the package itself stays a pure
# supervisor. It supervises the tensor server, which keeps writing the canonical
# tensor-server.log (the state-tree logs dir) that the control's log endpoint
# tails; the control plane's own supervision/control-API log is control.log.
CONTROL_PID_FILE = _locations.control_pid_file()


# The installer records the release-v* deployment version it pulled the wheels
# from in this marker file -- a clean PEP 440 string (e.g. "0.11.0"), the
# auto-updater's baseline. This is the *product* version: one release-v* tag
# versions the mutually-paired biopb-tensor-server / biopb-mcp / biopb-control /
# web set together, so the marker represents them all. (The biopb SDK ships on
# its own v* line, so its wheel version differs.) Kept in sync with
# CONFIG_DIR/release.version in install/install.sh.
_RELEASE_VERSION_FILE = DEFAULT_CONFIG_DIR / "release.version"


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


@app.command(help="Show the product deployment and biopb SDK versions.")
def version():
    """Show the two version lines: the product deployment and the biopb SDK."""
    rows = [
        # The product line (release-v*): biopb-tensor-server / mcp / control / web
        # all share this version, so the installer's deployment marker stands in
        # for the whole set -- no need to list each wheel separately.
        ("release", _read_release_version()),
        # The SDK line (v*): biopb ships to PyPI/Maven on its own tag, so its
        # version is independent of the product bundle it is also packaged into.
        ("biopb", _package_version("biopb")),
    ]

    # Left-align the labels so the versions line up in a readable column.
    width = max(len(name) for name, _ in rows) + 1  # +1 for the trailing ':'
    for name, ver in rows:
        console.print(f"{name + ':':<{width}} {ver}")


def _ensure_dirs():
    """Ensure required directories exist."""
    CONTROL_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _locations.log_dir()  # creates the state-tree logs dir on access


def _get_log_file() -> Path:
    """Get log file path."""
    return _locations.tensor_server_log()


# The rotation helper lives in `_locations` so the supervisor shares one
# rotator; re-exported here under the old name for the existing call sites.
_rotate_log = _locations.rotate_log


# --- log tailing (`biopb control logs`) ---------------------------------- #

# Severity ranks for the `--level` filter.
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _tensor_line_level(line: str) -> Optional[str]:
    """Level of a data-plane log line, or None if it has none.

    tensor-server.log carries the server's own format (DEFAULT_LOG_FORMAT in
    biopb_tensor_server.core.logging_config): `[2026-06-12 10:00:00] WARNING
    biopb_tensor_server.x: msg`. Returns None for the supervisor's `--- control:
    starting data plane ---` banners, blank lines, native gRPC/Arrow stdout, and
    traceback continuations — all of which _filter_lines carries forward.
    """
    if not line.startswith("["):
        return None
    try:
        after_ts = line.split("] ", 1)[1]
    except IndexError:
        return None
    token = after_ts.split(" ", 1)[0]
    return token if token in _LOG_LEVELS else None


def _control_line_level(line: str) -> Optional[str]:
    """Level of a control-plane log line, or None if it has none.

    control.log interleaves two formats, both handled here: the control's
    basicConfig (`2026-06-12 10:00:00,123 INFO biopb_control._run: msg`, level in
    the third whitespace token) and uvicorn's (`INFO:     msg`, level first).
    Best-effort by design — anything unrecognized pairs with the carry-forward in
    _filter_lines rather than being hard-dropped.
    """
    head = line.split(":", 1)[0].split(" ", 1)[0].strip()
    if head in _LOG_LEVELS:
        return head
    parts = line.split(maxsplit=3)
    if len(parts) >= 3 and parts[2] in _LOG_LEVELS:
        return parts[2]
    return None


def _filter_lines(lines, min_level: Optional[str], level_of=_tensor_line_level):
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
    level_of=_tensor_line_level,
):
    """Print the last `lines` lines of `log_file` (0 = all) filtered by
    `min_level`, then optionally stream appended lines until interrupted.

    `level_of` selects the per-log level parser. A missing file is reported (not
    an error) and exits 0. Follow reopens the file when it is rotated or
    truncated out from under us.
    """
    if not log_file.exists():
        console.print(
            f"[yellow]No log file at {log_file} — has it ever been started?[/yellow]"
        )
        raise typer.Exit(0)

    # Both logs rotate at 10 MB (_locations.rotate_log), so the current file is
    # small enough to read whole and slice - no seek-based tail.
    existing = log_file.read_text(errors="replace").splitlines()
    tail = existing if lines <= 0 else existing[-lines:]
    for line in _filter_lines(tail, min_level, level_of):
        print(line)

    if not follow:
        raise typer.Exit(0)

    # Flush the tail before blocking on new lines: piped/redirected stdout is
    # block-buffered, so without this `logs -f > file` (or `| grep`) shows nothing
    # until 4 KB accumulates -- and Ctrl-C before that loses the tail entirely.
    sys.stdout.flush()

    # Follow: poll for appended lines, reopening if the file is rotated or
    # truncated out from under us (a restart rotates it mid-follow). Track the
    # inode + size so a replaced or shrunk file restarts from the top.
    try:
        f = open(log_file, errors="replace")  # noqa: SIM115 - handle kept open across the follow loop, reopened on rotation
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
            try:
                st = os.stat(log_file)
            except OSError:
                st = None
            if st is not None and (st.st_ino != last_ino or st.st_size < f.tell()):
                f.close()
                f = open(log_file, errors="replace")  # noqa: SIM115 - reopened handle lives across the follow loop
                last_ino = os.fstat(f.fileno()).st_ino
                carry = ""
                continue
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    raise typer.Exit(0)


def _reject_legacy_toml(config: Path) -> None:
    """Refuse to start on a pre-#34 ``biopb.toml``, naming the migration command.

    The server no longer reads TOML (biopb/biopb#34), and every config probe on
    the start path is best-effort, so a legacy config would otherwise surface as
    a plane that starts on defaults and serves none of the user's data. Check it
    once, up front, where the user can act on it.
    """
    if config and config.suffix.lower() == ".toml" and config.exists():
        console.print(f"[red]Config {config} is in the legacy TOML format.[/red]")
        console.print(
            "JSON is the only supported format. Convert it with "
            "[bold]biopb-tensor-server migrate-config[/bold] (settings are "
            "preserved and the old file is backed up), then retry."
        )
        raise typer.Exit(1)


def _plane_bind(grpc_bind: str, base_port: int) -> Tuple[str, int]:
    """The flight plane's bind: the address from ``--grpc-bind``, port from the base.

    The address used to be read out of ``biopb.json`` (``server.host``), which
    made the deployment's exposure a *property of a file* the control snapshotted
    at startup -- so a config edit could silently disagree with the running plane
    (biopb/biopb#604). It is now the CLI's, and named for what it does: the flag
    that exposes the plane is the one you type an address into, and everything
    downstream (token required? TLS by default?) derives from that one address
    through :func:`biopb._web_auth.host_is_public_bind`.
    """
    return (grpc_bind, _flight_port(base_port))


def _probe_hostport(grpc_bind: str, base_port: int) -> Tuple[str, int]:
    """Loopback-reachable form of :func:`_plane_bind`, for health probes."""
    host, port = _plane_bind(grpc_bind, base_port)
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return host, port


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


# ---------------------------------------------------------------------------
# biopb-mcp (`biopb mcp view`)
#
# The shared background MCP daemon (`biopb mcp start/stop/restart/status/logs`)
# was retired with de-daemonization (biopb-mcp/docs/mcp-dedaemonization-
# migration.md): each MCP client's stdio shim now spawns and owns its own
# ephemeral session, and `biopb mcp view` covers the foreground/agentless case.
# `view` runs the server in a child process (`python -m biopb_mcp.mcp --view`),
# so this CLI never imports the heavy MCP/napari stack. The biopb-mcp package is
# an optional dependency: the subcommand first calls _require_biopb_mcp(), which
# surfaces a clear install hint (rather than a raw ImportError) when it is absent.
# ---------------------------------------------------------------------------

# Every command below passes an explicit one-line `help=`. Typer prefers it over
# the docstring, which keeps `--help` to a single sentence per command while the
# docstring stays where the rationale belongs -- read by maintainers, not printed
# at a user who asked what a command does.
mcp_app = typer.Typer(
    name="mcp",
    help="Run a foreground napari viewer session (biopb-mcp).",
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


@mcp_app.command(
    "view", help="Open the napari viewer in this terminal (Ctrl-C to stop)."
)
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
    immediately and the process blocks until Ctrl-C. A foreground, user-owned
    viewer that writes no PID file. It still serves /mcp on the chosen (default
    dynamic) port, so an AI agent may optionally attach to the same live session.

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
# biopb control: the control plane (supervises the durable planes)
# ---------------------------------------------------------------------------
# `biopb control` manages the lean control-plane process (the `biopb-control`
# package). Since the de-daemonization
# (biopb-mcp/ARCHITECTURE.md): the control plane becomes the
# durable root that supervises the tensor server, so `_connection` no longer
# shells out `biopb server start` -- it asks the control plane to ensure the data plane.
control_app = typer.Typer(
    name="control",
    help="Manage the control plane, which supervises the data plane.",
)


def _require_biopb_control() -> None:
    """Exit(1) with an install hint if the biopb-control package is absent.

    Checks the import *spec* (not a real import), matching _require_biopb_mcp, so
    gating a command never imports the package.
    """
    import importlib.util

    if importlib.util.find_spec("biopb_control") is None:
        console.print(
            "[red]The 'control' commands require the biopb-control package, which "
            "is not installed.[/red]\n"
            r"[yellow]Install it with: pip install biopb-control[/yellow]"
        )
        raise typer.Exit(1)


def _require_tls_extra() -> None:
    """Exit(2) with an install hint if ``--tls`` cannot possibly work.

    ``cryptography`` is an opt-in extra (biopb/biopb#355 -- it drags a
    Rust/OpenSSL build surface that breaks ``curl install.sh | bash`` on some
    platforms), so a default install cannot mint the self-signed certificate
    ``--tls`` needs.

    Without this check the failure lands in the *wrong place entirely*: the
    control starts fine and reports success, then its supervised plane exits 2
    on every spawn and crash-loops on backoff, with the one useful sentence
    buried in ``tensor-server.log``. The user sees a control that started and a
    data plane that never serves. Checking here fails the command the user
    actually typed, before anything is spawned.

    Same-interpreter check: the control is spawned as ``sys.executable -m
    biopb_control`` and spawns the plane the same way, so this process's import
    spec is the plane's (and ``sys.executable`` names the exact environment to
    install into -- which a generic ``pip install`` hint would not, under the
    ``uv tool`` layout the installer uses).
    """
    import importlib.util

    if importlib.util.find_spec("cryptography") is not None:
        return
    console.print(
        "[red]--tls needs the 'cryptography' package, which is not installed."
        "[/red]\nIt is an opt-in extra: it needs a Rust/OpenSSL build that the "
        "default install deliberately avoids (biopb/biopb#355).\n"
    )
    console.print(
        "[yellow]Install it into the environment that runs the data plane:[/yellow]"
    )
    # The one line the reader must copy verbatim, so it gets the same treatment
    # as a printed fingerprint: soft_wrap so a narrow terminal cannot break it
    # across lines, and markup off because Rich reads a bare `[tls]` as a style
    # tag and silently eats it.
    console.print(
        f"    {sys.executable} -m pip install 'biopb-tensor-server[tls]'",
        soft_wrap=True,
        markup=False,
        highlight=False,
    )
    console.print(
        "\nThen retry. Or start without [bold]--tls[/bold] — the data plane still "
        "serves, clients just dial grpc:// instead of grpcs://."
    )
    raise typer.Exit(2)


def _control_endpoint() -> Tuple[str, int]:
    """Where to *find* a control: env override -> published record -> 8813.

    The discovery form, for commands that talk to a control someone else started
    (``status`` / ``stop`` / ``logs`` / ``dashboard``). A control started with a
    non-default ``--base-port`` publishes its endpoint on serve, which is what
    lets these follow it (see ``biopb._locations.control_runtime_file``).

    Never used to decide a *bind* -- that is :func:`_control_bind_endpoint`.
    Binding to a discovered value would mean a crashed control's stale record
    dictates where the next one listens.
    """
    from ._endpoints import control_host, control_port

    return control_host(), control_port()


def _control_bind_endpoint(base_port: int) -> Tuple[str, int]:
    """Where a control we are *starting* should listen: base+3, env still wins.

    ``BIOPB_CONTROL_HOST`` / ``BIOPB_CONTROL_PORT`` keep top precedence so the
    pre-base-port escape hatch still works and a reader and a writer that both
    honor the env can never disagree; otherwise the port comes from the base and
    nowhere else -- notably *not* from the published record, which describes some
    other (possibly dead) control.
    """
    from ._endpoints import CONTROL_DEFAULT_HOST, control_port_for

    host = os.environ.get("BIOPB_CONTROL_HOST") or CONTROL_DEFAULT_HOST
    raw = os.environ.get("BIOPB_CONTROL_PORT")
    if raw:
        try:
            return host, int(raw)
        except ValueError:
            pass
    return host, control_port_for(base_port)


def _print_ui_tunnel_hint(control_port: int) -> None:
    """Print the SSH-tunnel recipe for reaching the browser UI off-box.

    ``--remote`` publishes the flight plane and nothing else: the control serves
    plaintext HTTP and has no TLS support, so a public bind would carry the
    data-plane token — which unlocks the data *and* admin API — in the clear
    (biopb/biopb#614). A tunnel gets encryption and authentication for free, adds
    no listener, and is the pattern Jupyter users already know. Print it here so
    it is discoverable at the moment the user needs it, rather than folklore.
    """
    import socket

    host = socket.gethostname() or "<host>"
    console.print("  Browser UI: loopback only. From another machine, tunnel it:")
    # soft_wrap so a narrow terminal cannot break the one line the reader has to
    # copy verbatim (same treatment as the pip hint in _require_tls_extra).
    console.print(
        f"    [bold]ssh -L {control_port}:localhost:{control_port} {host}[/bold]",
        soft_wrap=True,
        highlight=False,
    )
    console.print(f"    then open http://localhost:{control_port}")


def _control_log_file() -> Path:
    """The control plane's own supervision / control-API log (distinct from the data
    plane's tensor-server.log, which the supervised server keeps writing)."""
    return _locations.control_log()


def _write_control_pid(pid: int) -> None:
    _ensure_dirs()
    _write_pid_file(CONTROL_PID_FILE, pid, _process_create_time(pid))


def _remove_control_pid() -> None:
    _remove_pid_file(CONTROL_PID_FILE)


def _control_shutdown_sentinel() -> Path:
    """The control plane's Windows stop-sentinel path (watched by biopb_control._run).
    A single fixed name under the biopb state dir, like the other daemons'."""
    return _locations.control_stop_sentinel()


def _control_start_lock() -> Path:
    """Cross-process lock file serializing `biopb control start`.

    The launcher, the installer, and -- once the shim starts the control on demand
    -- racing agent sessions can all invoke `control start` at once. Holding this
    lock across the check-then-spawn below makes it atomic between processes:
    without it two starters can both see "no pidfile", both spawn a control, and
    the bind-loser's parent overwrite/remove the live winner's pidfile, orphaning a
    control that `control stop` can no longer reach. See biopb._lifecycle.file_lock.
    """
    return CONTROL_PID_FILE.parent / "control.start.lock"


def _resolve_grpc_bind(grpc_bind: Optional[str], remote: bool) -> str:
    """The flight bind from ``--grpc-bind``, honoring the deprecated ``--remote``.

    ``--remote`` named a *mode* back when it also published the browser UI. Since
    biopb/biopb#614 it sets one thing — the flight address — so the flag is now
    named for that. It survives as an alias because it is in install scripts,
    service units, and every doc; an explicit ``--grpc-bind`` wins over it, since
    naming an address is more specific than asking for "public".
    """
    if grpc_bind is None:
        if remote:
            console.print(
                "[yellow]--remote is deprecated[/yellow]; it now means exactly "
                "[bold]--grpc-bind 0.0.0.0[/bold]. Prefer the explicit form."
            )
            return "0.0.0.0"
        return "127.0.0.1"
    if remote:
        console.print(
            f"[yellow]--remote ignored[/yellow]: --grpc-bind {grpc_bind} is explicit."
        )
    return grpc_bind


def _resolve_tls(tls: Optional[bool], grpc_bind: str) -> bool:
    """Whether to serve the flight plane over TLS. **The bind decides the default.**

    A public flight bind defaults TLS *on*; loopback defaults it off. Tying it
    this way round — bind drives TLS, not TLS drives bind — keeps each flag's
    name matching its own effect, and makes the dangerous combination the one you
    have to ask for by name: ``--grpc-bind 0.0.0.0 --no-tls`` puts the access
    token on the wire in cleartext on every gRPC call, which is precisely the
    objection biopb/biopb#614 raised about the control, transplanted onto the
    data plane. It stays *possible* — a trusted intranet is a real deployment —
    but it is spelled out rather than defaulted into.

    ``--tls`` alone therefore still means "encrypted, loopback only", which is
    what exercising the TOFU pinning and SAN-verification paths (#606) needs.
    """
    if tls is not None:
        return tls
    return _web_auth.host_is_public_bind(grpc_bind)


def _warn_public_plaintext(grpc_bind: str, tls: bool) -> None:
    """Warn when the data plane is published without TLS (an explicit --no-tls)."""
    if not tls and _web_auth.host_is_public_bind(grpc_bind):
        console.print(
            f"[yellow]Warning:[/yellow] the data plane is bound publicly "
            f"({grpc_bind}) without TLS. The access token and every pixel cross "
            "the network in cleartext — keep this to a trusted intranet, or drop "
            "[bold]--no-tls[/bold] to serve grpcs://."
        )


def _resolve_mode(grpc_bind: str, token: Optional[str]) -> Optional[str]:
    """Resolve the data-plane token for the chosen flight bind.

    Token enforcement is **independent** of the network mode: a token may be
    supplied — via ``--token`` or ``BIOPB_TENSOR_TOKEN`` — with *either* bind, so
    a single-machine deployment can still gate its listeners for defense-in-depth
    on a shared host. What ``--grpc-bind`` controls is who can reach the plane.

    - **Loopback** (the default): a token is *optional*; when one is supplied it
      is enforced (the browser then gates behind the unlock page just the same).
    - **Public**: the flight server is reachable off-box, so a token is
      **required** — supplied, or else generated and printed.

    The control (the browser UI) stays on loopback with *either* bind; it is
    plaintext HTTP with no TLS support, so publishing it would put this very
    token on the wire in the clear (biopb/biopb#614). Reach it over an SSH tunnel.

    "Public but unauthenticated" is therefore unrepresentable through this
    command: the one address that decides exposure is the one this reads, through
    the same :func:`biopb._web_auth.host_is_public_bind` the tensor ``launch`` and
    the control's own bind guard use, so the three cannot drift.

    Returns the token to enforce (``None`` only when none is supplied on a
    loopback bind).
    """
    token = token or os.environ.get("BIOPB_TENSOR_TOKEN")
    if token:
        # Validate here with the shared rule the tensor `launch` applies, so the
        # two layers can't disagree: an invalid token this layer accepted would be
        # silently regenerated (remote) or ignored (local) downstream, leaving the
        # browser holding a token the data plane rejects.
        token = token.strip()
        if not _web_auth.valid_token(token):
            console.print(
                "[red]Invalid access token[/red]: must be 16-128 URL-safe "
                "characters ([A-Za-z0-9_-]). Fix --token / BIOPB_TENSOR_TOKEN, or "
                "omit it to run tokenless (loopback) / auto-generate one (public)."
            )
            raise typer.Exit(1)
        return token

    if _web_auth.host_is_public_bind(grpc_bind):
        import secrets as _secrets

        token = _secrets.token_urlsafe(32)
        console.print(f"[bold green]Generated access token:[/bold green] {token}")
        return token
    return None


def _query_control_health(host: str, port: int, timeout: float = 2.0) -> Optional[dict]:
    """GET the control API's /health, or None if unreachable."""
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=timeout
        ) as resp:
            return _json.loads(resp.read().decode())
    except Exception:
        return None


def _flight_location(grpc_bind: str, base_port: int, tls: bool) -> str:
    """The flight plane's dial string, e.g. ``grpcs://0.0.0.0:8815``.

    Printed at startup because ``--base-port`` makes the port a computation: an
    operator who firewalled "8815" by habit needs to see where it actually landed,
    and whether it is plaintext or TLS.
    """
    host, port = _plane_bind(grpc_bind, base_port)
    scheme = "grpcs" if tls else "grpc"
    authority = f"[{host}]" if ":" in host else host
    return f"{scheme}://{authority}:{port}"


def _guard_ports_free(base_port: int, grpc_bind: str, data_plane: bool) -> None:
    """Refuse to start into a port something already holds, naming which one.

    Shared by ``control start`` and ``control run`` -- the foreground command used
    to skip this entirely and crash-land in uvicorn's bind error instead, which is
    the same deployment failing with a worse message.

    All three listeners are checked. The sidecar was previously unguarded, which
    was survivable while it was pinned to 8814 but is not now that ``--base-port``
    can land it on anything: an unguarded collision surfaces as a control that
    starts clean and then crash-loops its plane in the background.
    """
    checks = [("Control-plane", *_control_bind_endpoint(base_port))]
    if data_plane:
        checks.append(("Data-plane gRPC", *_probe_hostport(grpc_bind, base_port)))
        checks.append(("Tensor HTTP sidecar", "127.0.0.1", _sidecar_port(base_port)))
    for label, host, port in checks:
        if not _port_listening(host, port):
            continue
        console.print(f"[red]{label} port {host}:{port} is already in use.[/red]")
        console.print(
            "It is held by a process biopb is not tracking (an orphaned plane, or "
            "another login session), so [bold]biopb control stop[/bold] cannot "
            f"reach it. Identify and stop the owner (`lsof -i :{port}` / "
            f"`netstat -ano | findstr {port}`), then retry -- or move the whole "
            "deployment with [bold]--base-port[/bold]."
        )
        raise typer.Exit(1)


def _control_run_argv(
    *,
    config: Path,
    static_dir: Optional[Path],
    web_host: str,
    base_port: int,
    log_level: str,
    data_plane: bool,
    grpc_bind: str,
    tls: bool = False,
) -> List[str]:
    """Build the `python -m biopb_control run ...` argv `control start` spawns.

    The core CLI resolves everything (binds, ports, token, log paths) and passes
    it explicitly, so biopb_control imports no server config (invariant I2) and
    knows nothing of the base-port convention -- it receives three already-derived
    ports. The supervised tensor server logs to tensor-server.log; the control
    plane's own output is redirected by the caller to control.log.

    The access token is **not** on this argv: a command line is world-readable
    (`ps aux`, Task Manager) on exactly the multi-user hosts a token is meant to
    protect (biopb/biopb#414). It travels only via ``BIOPB_TENSOR_TOKEN`` in the
    child env (set by the caller).

    No ``--remote`` either, and not because it is secret: ``--grpc-host`` below
    already carries the one fact it used to signal. The child re-derives "is this
    deployment public?" from that address with the shared predicate, so the two
    layers cannot disagree about it (biopb/biopb#614). The control's own listener
    stays on loopback regardless.
    """
    grpc_host, grpc_port = _plane_bind(grpc_bind, base_port)
    control_host, control_port = _control_bind_endpoint(base_port)
    web_port = _sidecar_port(base_port)
    argv = [
        sys.executable,
        "-m",
        "biopb_control",
        "run",
        "--config",
        str(config),
        "--grpc-host",
        grpc_host,
        "--grpc-port",
        str(grpc_port),
        "--web-host",
        web_host,
        "--web-port",
        str(web_port),
        "--log-level",
        str(log_level),
        "--server-log",
        str(_get_log_file()),
        "--control-host",
        control_host,
        "--control-port",
        str(control_port),
        "--win-sentinel",
        str(_control_shutdown_sentinel()),
    ]
    if static_dir and static_dir.exists():
        argv += ["--static-dir", str(static_dir)]
    if not data_plane:
        argv.append("--no-data-plane")
    if tls:
        argv.append("--tls")
    return argv


# --- shared `control start` / `control run` options ----------------------- #
#
# The two commands stand up the *same* deployment; only process ownership
# differs (daemon vs foreground). Their flags therefore have to agree, and they
# had drifted -- same option set, three different help texts, so `--help` told
# you different things depending on which you asked. One `typer.Option` object
# per flag, referenced by both, makes that unrepresentable instead of a review
# item. Anything genuinely command-specific stays declared inline.

_OPT_CONFIG = typer.Option(
    DEFAULT_CONFIG, "--config", "-c", help="Tensor-server config (biopb.json)"
)
_OPT_STATIC_DIR = typer.Option(
    DEFAULT_WEBAPP,
    "--static-dir",
    help="Web UI bundle the control serves at its root (the built web/ dist)",
)
_OPT_BASE_PORT = typer.Option(
    _endpoints.BASE_DEFAULT_PORT,
    "--base-port",
    help="Base port for the whole deployment. The three listeners are derived "
    "from it: control/browser UI = base+3, tensor HTTP sidecar = base+4, flight "
    "gRPC = base+5 (so the 8810 default gives 8813/8814/8815). Same convention "
    "as the container's BIOPB_BASE_PORT. Move it to run a second deployment "
    "alongside another user's — give that one its own XDG_STATE_HOME too.",
)
_OPT_LOG_LEVEL = typer.Option("INFO", "--log-level", "-l", help="Control log level")
_OPT_GRPC_BIND = typer.Option(
    None,
    "--grpc-bind",
    help="Address the flight (data-plane) server binds. Loopback (the default, "
    "127.0.0.1) keeps the deployment on this machine. A public address "
    "(0.0.0.0, or one interface's IP) serves it to other machines, and then an "
    "access token is REQUIRED — supplied via --token, else generated and "
    "printed — and TLS is on by default. This is the only listener that is ever "
    "published: the sidecar and the browser UI stay on loopback, reachable "
    "off-box through the ssh -L tunnel printed on start.",
)
_OPT_TLS = typer.Option(
    None,
    "--tls/--no-tls",
    help="Serve the flight port over TLS with a self-signed certificate "
    "(generated on first use); clients dial grpcs:// and pin it on first connect. "
    "Defaults to ON for a public --grpc-bind and off for loopback, so the "
    "default follows the exposure. --tls needs the 'tls' extra; read the "
    "fingerprint with `biopb-tensor-server cert init`. --no-tls on a public bind "
    "sends the token in cleartext — trusted networks only.",
)
_OPT_TOKEN = typer.Option(
    None,
    "--token",
    help="Access token (or set BIOPB_TENSOR_TOKEN). Enforced with either bind: "
    "required for a public --grpc-bind (auto-generated if omitted), optional on "
    "loopback as defense-in-depth on a shared machine. A loopback token gates "
    "the browser too; local clients read it from the credential file the control "
    "writes, so biopb-mcp needs no environment of its own (biopb/biopb#470).",
)
_OPT_DATA_PLANE = typer.Option(
    True,
    "--data-plane/--no-data-plane",
    help="Bring the data plane up on start (default). With --no-data-plane the "
    "control plane starts without it; a client brings it up on demand via the "
    "control API.",
)


@control_app.command(
    "start", help="Start the control plane (and its data plane) as a daemon."
)
def control_start(
    config: Path = _OPT_CONFIG,
    static_dir: Optional[Path] = _OPT_STATIC_DIR,
    base_port: int = _OPT_BASE_PORT,
    log_level: str = _OPT_LOG_LEVEL,
    grpc_bind: Optional[str] = _OPT_GRPC_BIND,
    tls: Optional[bool] = _OPT_TLS,
    token: Optional[str] = _OPT_TOKEN,
    data_plane: bool = _OPT_DATA_PLANE,
    remote: bool = typer.Option(
        False,
        "--remote",
        hidden=True,
        help="Deprecated alias for --grpc-bind 0.0.0.0.",
    ),
):
    """Start the biopb control plane as a background daemon.

    The control plane supervises the tensor (data) plane -- and by default brings it up
    on start, so `biopb control start` is the single command that stands up a local
    deployment. It is the *sole owner* of the plane: it always spawns and manages
    its own tensor server, restarts it on crash, and answers clients that ask it
    to ensure the plane is up. It does not adopt a server it did not start -- if
    the gRPC port is already in use, `control start` refuses (stop the stray server
    first), so `biopb control stop` is always a complete data-plane teardown.

    **Ports** come from one number, ``--base-port`` (default 8810): control =
    base+3, sidecar = base+4, flight = base+5 — the container's convention. A
    control that moved off 8813 publishes where it landed, so `stop` / `status` /
    `logs` and biopb-mcp follow it without being told.

    **Exposure** comes from one address, ``--grpc-bind`` (default 127.0.0.1).
    Loopback is the single-machine 90% case, tokenless unless you pass ``--token``
    (defense-in-depth on a shared machine). A public address serves the data plane
    to other machines, and then a token is *required* and TLS is on by default —
    the bind is read once, through the predicate the tensor `launch` and the
    control's own guard share, so "public but unauthenticated" is unrepresentable
    rather than something to validate against (biopb/biopb#604).

    Only the flight plane is ever published. The tensor HTTP sidecar stays on
    loopback (the control proxies it), and so does the control itself — the
    browser UI is plaintext HTTP with no TLS support, so publishing it would send
    the token that unlocks the whole data and admin API in the clear, which is
    exactly the client class the TLS work set out to remove (biopb/biopb#614). To
    open the UI from another machine, tunnel it: ``ssh -L 8813:localhost:8813
    <host>``, then browse http://localhost:8813.
    """
    _require_biopb_control()
    grpc_bind = _resolve_grpc_bind(grpc_bind, remote)
    tls = _resolve_tls(tls, grpc_bind)
    if tls:
        _require_tls_extra()
    _warn_public_plaintext(grpc_bind, tls)
    _ensure_dirs()
    _reject_legacy_toml(config)

    # Serialize concurrent starts so the check-then-spawn below is atomic across
    # processes (see _control_start_lock / biopb._lifecycle.file_lock). Held through the
    # readiness wait too, so a second starter that was blocked wakes to a fully
    # started control (pidfile written, port listening) and reports the idempotent
    # "already running" rather than racing a half-up one. The lock auto-releases if
    # a holder dies, so a crashed starter leaves nothing to clean up.
    try:
        with file_lock(_control_start_lock(), timeout=30.0):
            existing_pid, existing_token = _read_pid_record(CONTROL_PID_FILE)
            if _is_our_daemon(existing_pid, existing_token):
                console.print(
                    f"[yellow]biopb control already running (PID {existing_pid})[/yellow]"
                )
                raise typer.Exit(0)
            if existing_pid:
                console.print(
                    f"[yellow]Removing stale PID file (process {existing_pid} not running)[/yellow]"
                )
                _remove_control_pid()

            control_host, control_port = _control_bind_endpoint(base_port)
            _guard_ports_free(base_port, grpc_bind, data_plane)

            resolved_token = _resolve_mode(grpc_bind, token)
            argv = _control_run_argv(
                config=config,
                static_dir=static_dir,
                # The sidecar always binds loopback; the control proxies it. The
                # flight server is the only listener --grpc-bind can publish.
                web_host="127.0.0.1",
                base_port=base_port,
                log_level=log_level,
                data_plane=data_plane,
                grpc_bind=grpc_bind,
                tls=tls,
            )

            log_file = _control_log_file()
            _rotate_log(log_file)
            console.print("[green]Starting biopb control plane...[/green]")
            console.print(f"  Config: {config}")
            env = os.environ.copy()
            if resolved_token:
                # The token travels to the control child (and on to the tensor
                # server) via the env only, never the argv (biopb/biopb#414):
                # biopb_control reads it back off BIOPB_TENSOR_TOKEN.
                env["BIOPB_TENSOR_TOKEN"] = resolved_token
            with open(log_file, "a") as log:
                log.write(
                    f"\n--- Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )
                process = subprocess.Popen(
                    argv, stdout=log, stderr=log, env=env, **_detach_kwargs()
                )

            _write_control_pid(process.pid)

            if not _await_listening(process.pid, control_host, control_port, 15.0):
                if _is_process_running(process.pid):
                    console.print(
                        f"[red]Control plane started but its control API is not listening on "
                        f"{control_host}:{control_port} after 15s.[/red]"
                    )
                    console.print(f"Check the log: {log_file}")
                else:
                    console.print("[red]Failed to start biopb control plane[/red]")
                    _remove_control_pid()
                    console.print(f"Check the log: {log_file}")
                raise typer.Exit(1)

            console.print(
                f"[green]biopb control plane started (PID {process.pid})[/green]"
            )
            console.print(f"  Control: http://{control_host}:{control_port}")
            if data_plane:
                console.print(
                    f"  Data plane: starting on {_flight_location(grpc_bind, base_port, tls)}"
                )
            else:
                console.print("  Data plane: not started (--no-data-plane; on-demand)")
            console.print(f"  Logs: {log_file}")
            if _web_auth.host_is_public_bind(grpc_bind):
                _print_ui_tunnel_hint(control_port)
    except LockTimeout:
        console.print(
            "[red]Another 'biopb control start' is already in progress and did not "
            "finish within 30s.[/red] Retry shortly, or check 'biopb control status'."
        )
        raise typer.Exit(1)


def _live_foreground_control() -> Optional[Tuple[dict, int]]:
    """The published record of a live foreground control, as ``(record, pid)``.

    A foreground `biopb control run` writes no pid file -- its terminal or
    service manager owns it -- so this endpoint record is the only trace of it.
    Verified for *identity*, not merely liveness: a clean stop retracts the
    record, so the way to strand one is a crash, and a pid recycled since then
    would otherwise read as a control still serving. `_is_our_daemon` compares
    the recorded create-time token and refuses to vouch for a different process.

    Falls back to liveness when the record carries no usable token (written
    before the field existed, or a platform with no cheap create-time), matching
    the pid file's own degradation -- never a false "not running".
    """
    record = _endpoints.read_runtime_record()
    pid = record.get("pid")
    if not isinstance(pid, int):
        return None
    token = record.get("create_time")
    if not _is_our_daemon(pid, token if isinstance(token, int) else None):
        return None
    return record, pid


@control_app.command("stop", help="Stop the control plane and the data plane it owns.")
def control_stop(
    timeout: int = typer.Option(
        10, "--timeout", "-t", help="Seconds to wait for graceful shutdown"
    ),
):
    """Stop the biopb control plane and the data plane it owns.

    The control plane owns the data plane exclusively, so stopping it is a complete
    teardown: the supervised tensor server is shut down too. This is the single
    command an installer/upgrade uses to free the control-managed processes before
    replacing files.

    Only reaches a *daemonized* control (`biopb control start`). A foreground
    `biopb control run` belongs to its terminal or service manager, so this
    reports it and declines rather than signalling a process it does not own.
    """
    _require_biopb_control()
    pid, token = _read_pid_record(CONTROL_PID_FILE)
    if not pid:
        # `status` reports a foreground control as Running, so "nothing running"
        # here would flatly contradict it. Say which one is up and who owns it.
        live = _live_foreground_control()
        if live:
            record, record_pid = live
            console.print(
                f"[yellow]A foreground control plane is running (PID {record_pid}, "
                f"http://{record.get('host')}:{record.get('port')}).[/yellow]"
            )
            console.print(
                "It was started with [bold]biopb control run[/bold], so it has no "
                "PID file and this command does not own it. Stop it with Ctrl-C in "
                "its terminal, or through your service manager."
            )
            raise typer.Exit(1)
        console.print("[yellow]No biopb control plane running[/yellow]")
        raise typer.Exit(0)
    if not _is_our_daemon(pid, token):
        console.print(
            f"[yellow]Process {pid} not running, cleaning up PID file[/yellow]"
        )
        _remove_control_pid()
        raise typer.Exit(0)

    console.print(f"[green]Stopping biopb control plane (PID {pid})...[/green]")
    if _stop_daemon(
        pid,
        timeout,
        token,
        sentinel=_control_shutdown_sentinel(),
        remove_pid=_remove_control_pid,
        notify=lambda diag: console.print(
            f"[yellow]Graceful stop unavailable ({diag}); force killing.[/yellow]"
        ),
    ):
        console.print("[green]biopb control plane stopped[/green]")
    else:
        console.print(f"[yellow]Did not stop within {timeout}s; force killed[/yellow]")
    raise typer.Exit(0)


@control_app.command(
    "status", help="Show the control plane's status and the data plane it supervises."
)
def control_status(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
):
    """Show the control plane's status and the data plane it supervises."""
    _require_biopb_control()
    pid, token = _read_pid_record(CONTROL_PID_FILE)
    running = _is_our_daemon(pid, token)
    stale = bool(pid and not running)

    # A foreground `control run` writes no pid file -- the terminal or service
    # manager owns it -- so it used to report "not running" however healthy it
    # was. It does publish its endpoint, though, so fall back to that record and
    # report it honestly rather than denying it exists.
    foreground = False
    if not running:
        live = _live_foreground_control()
        if live:
            pid, running, stale, foreground = live[1], True, False, True

    control_host, control_port = _control_endpoint()
    health = _query_control_health(control_host, control_port) if running else None
    data_plane = (health or {}).get("data_plane") or {}
    dp_state = data_plane.get("state", "unknown")

    _emit_daemon_status(
        title="biopb Control Plane Status",
        pid=pid,
        running=running,
        stale=stale,
        pid_file=CONTROL_PID_FILE,
        log_file=_control_log_file(),
        json_output=json_output,
        json_fields={
            "control_url": f"http://{control_host}:{control_port}" if running else None,
            "control_api": bool(health) if running else False,
            "foreground": foreground,
            "data_plane": (data_plane or None) if running else None,
        },
        table_rows=[
            ("Control", f"http://{control_host}:{control_port}"),
            ("Control API", "responding" if health else "not responding"),
            (
                "Ownership",
                "foreground (Ctrl-C or your service manager stops it; "
                "'biopb control stop' does not)"
                if foreground
                else "daemon ('biopb control stop')",
            ),
            ("Data plane", dp_state),
            ("Data plane URL", data_plane.get("grpc_url", "-")),
            ("Restarts", str(data_plane.get("restarts", 0))),
        ],
    )


@control_app.command(
    "logs", help="Show the control plane's log, or the data plane's with --data-plane."
)
def control_logs(
    data_plane: bool = typer.Option(
        False,
        "--data-plane",
        help="Show the supervised tensor server's log instead of the control's own",
    ),
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
    """Show the control plane's log, or the data plane's with --data-plane.

    Two logs, because the control plane is two processes: the control writes its
    own supervision / control-API log (control.log, the default here), and the
    tensor server it supervises keeps writing the data-plane log
    (tensor-server.log) that the control redirects its child's output to.

    Reads the file straight off disk rather than through the control API, so it
    works on a stopped or wedged control -- which is when the log matters most.
    """
    log_file = (
        _get_log_file() if data_plane else _control_log_file()  # tensor-server.log
    )
    if path:
        print(log_file)
        raise typer.Exit(0)
    level_of = _tensor_line_level if data_plane else _control_line_level
    _tail_and_follow(log_file, follow, lines, _validate_level(level), level_of)


@control_app.command(
    "run", help="Run the control plane in the foreground (Ctrl-C to stop)."
)
def control_run(
    config: Path = _OPT_CONFIG,
    static_dir: Optional[Path] = _OPT_STATIC_DIR,
    base_port: int = _OPT_BASE_PORT,
    log_level: str = _OPT_LOG_LEVEL,
    grpc_bind: Optional[str] = _OPT_GRPC_BIND,
    tls: Optional[bool] = _OPT_TLS,
    token: Optional[str] = _OPT_TOKEN,
    data_plane: bool = _OPT_DATA_PLANE,
    remote: bool = typer.Option(
        False,
        "--remote",
        hidden=True,
        help="Deprecated alias for --grpc-bind 0.0.0.0.",
    ),
):
    """Run the control plane in the foreground (Ctrl-C to stop).

    The foreground counterpart of `biopb control start`, and the *same*
    deployment: identical flags, identical binds, identical port derivation from
    ``--base-port``. Only process ownership differs — no PID file, blocks this
    terminal, tears everything down on Ctrl-C. Useful for a systemd/launchd unit
    (let the service manager own the process) or for debugging supervision.

    It still publishes where it listens, so `status` / `logs` and biopb-mcp find
    a foreground control exactly as they find a daemonized one. `biopb control
    stop` does not reach it, by design: the pid file is the daemon's lifecycle
    record and this process belongs to your terminal or your service manager.
    See `biopb control start` for the bind / token / TLS model.
    """
    _require_biopb_control()
    grpc_bind = _resolve_grpc_bind(grpc_bind, remote)
    tls = _resolve_tls(tls, grpc_bind)
    if tls:
        _require_tls_extra()
    _warn_public_plaintext(grpc_bind, tls)
    _ensure_dirs()
    _reject_legacy_toml(config)
    from biopb_control import run_control
    from biopb_control._supervisor import DataPlaneSpec

    grpc_host, grpc_port = _plane_bind(grpc_bind, base_port)
    control_host, control_port = _control_bind_endpoint(base_port)
    # The same pre-flight `start` does. It used to be missing here, so a busy port
    # surfaced as uvicorn's bind traceback instead of a message naming the port.
    _guard_ports_free(base_port, grpc_bind, data_plane)
    resolved_token = _resolve_mode(grpc_bind, token)
    console.print(f"  Control: http://{control_host}:{control_port}")
    if data_plane:
        console.print(f"  Data plane: {_flight_location(grpc_bind, base_port, tls)}")
    # Only the flight plane is ever published; the control and the sidecar stay on
    # loopback either way (biopb/biopb#614), so point the user at the tunnel.
    if _web_auth.host_is_public_bind(grpc_bind):
        _print_ui_tunnel_hint(control_port)
    spec = DataPlaneSpec(
        config=config,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        tls=tls,
        web_host="127.0.0.1",
        web_port=_sidecar_port(base_port),
        static_dir=static_dir if (static_dir and static_dir.exists()) else None,
        log_level=log_level,
        server_log=_get_log_file(),
        token=resolved_token,
    )
    code = run_control(
        spec,
        control_host=control_host,
        control_port=control_port,
        data_plane=data_plane,
        win_sentinel=_control_shutdown_sentinel(),
        log_level=log_level,
    )
    raise typer.Exit(code)


app.add_typer(control_app, name="control")


@app.command(
    "dashboard", help="Open the biopb dashboard, starting the control plane if needed."
)
def dashboard(
    base_port: int = _OPT_BASE_PORT,
    grpc_bind: Optional[str] = _OPT_GRPC_BIND,
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Ensure the control plane is up but only print the dashboard URL "
        "instead of opening a browser.",
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        hidden=True,
        help="Deprecated alias for --grpc-bind 0.0.0.0.",
    ),
):
    """Open the biopb dashboard, starting the control plane first if needed.

    The one-command way in: it makes sure the control plane (which owns the data
    plane and serves the web UI) is running, then points your default web browser
    at the dashboard. Idempotent -- if the control plane is already up it just
    opens the page. This is what the desktop shortcut the installer creates runs.

    ``--base-port`` / ``--grpc-bind`` are forwarded to `biopb control start` and
    only matter when there is nothing running to open.
    """
    # Prefer a control that is already serving -- it publishes its endpoint, so
    # this finds one that `--base-port` moved. Fall back to where we *would* start
    # one, which is also what a first run resolves to.
    control_host, control_port = _control_endpoint()
    if not _port_listening(control_host, control_port):
        control_host, control_port = _control_bind_endpoint(base_port)
    url = f"http://{control_host}:{control_port}"

    if _port_listening(control_host, control_port):
        console.print(f"[green]biopb control plane already running[/green] ({url})")
    else:
        # Reuse `biopb control start`'s full start/port-guard/readiness logic (it
        # returns only once the control API is listening). It signals its outcome
        # by raising typer.Exit; a non-zero code means the plane never came up, so
        # bail out rather than open a browser at a dead URL.
        try:
            control_start(
                config=DEFAULT_CONFIG,
                static_dir=DEFAULT_WEBAPP,
                base_port=base_port,
                log_level="INFO",
                grpc_bind=grpc_bind,
                tls=None,
                token=None,
                data_plane=True,
                remote=remote,
            )
        except typer.Exit as started:
            if started.exit_code:
                raise

    if no_browser:
        console.print(f"Dashboard: {url}")
        raise typer.Exit(0)

    import webbrowser

    console.print(f"[green]Opening the dashboard:[/green] {url}")
    if not webbrowser.open(url):
        console.print(
            "[yellow]Could not open a browser automatically.[/yellow] "
            f"Open this URL manually: {url}"
        )
    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# biopb agents: register biopb-mcp with local AI agent clients
# ---------------------------------------------------------------------------
# The installer wires biopb into detected MCP clients once at install time; these
# commands do the same afterwards (install Claude Code later, register it now),
# over the shared, stdlib-only catalog in biopb._agents -- the single source of
# truth both this CLI and the control-plane dashboard call.
agents_app = typer.Typer(
    name="agents",
    help="Register biopb-mcp with local AI agent clients.",
)

# State -> rich style for the status column.
_AGENT_STATE_STYLE = {
    "registered": "green",
    "installed": "yellow",
    "not_installed": "dim",
}


def _agent_state_label(row: dict) -> str:
    """Human label for a status row (``drifted`` annotated so a stale entry that
    needs a Re-register is visible)."""
    state = row["state"]
    if state == "registered" and row.get("drifted"):
        return "registered (drifted)"
    return state.replace("_", " ")


def _known_agent_ids() -> List[str]:
    return [s.id for s in _agents.supported()]


def _resolve_agent_targets(
    client: Optional[str], all_: bool, *, states: Optional[set] = None
) -> List[str]:
    """The client ids a register/unregister should act on.

    An explicit ``client`` acts on exactly that one (validated). ``--all`` acts on
    every client whose current state is in ``states`` (e.g. skip ``not_installed``
    for register, target only ``registered`` for unregister), matching the
    installer's "only touch clients that are actually there" behavior. Exits 1 on
    a bad/missing selector.
    """
    ids = _known_agent_ids()
    if all_ and client:
        console.print("[red]Pass either a client id or --all, not both.[/red]")
        raise typer.Exit(1)
    if not all_ and not client:
        console.print(
            "[red]Specify a client id or --all.[/red] Known clients: " + ", ".join(ids)
        )
        raise typer.Exit(1)
    if client is not None:
        if client not in ids:
            console.print(
                f"[red]Unknown client {client!r}.[/red] Known clients: "
                + ", ".join(ids)
            )
            raise typer.Exit(1)
        return [client]
    # --all: filter by current state.
    targets = [
        row["id"]
        for row in _agents.statuses()
        if states is None or row["state"] in states
    ]
    return targets


@agents_app.command(
    "list", help="Show each supported client and whether biopb is registered."
)
def agents_list(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
):
    """Show each supported client and whether biopb is registered."""
    rows = _agents.statuses()
    if json_output:
        print(json.dumps({"agents": rows}))
        raise typer.Exit(0)
    table = Table(title="Agent clients")
    table.add_column("Client", style="cyan")
    table.add_column("Status")
    table.add_column("Config", style="dim")
    for row in rows:
        style = _AGENT_STATE_STYLE.get(row["state"], "white")
        table.add_row(
            row["name"],
            f"[{style}]{_agent_state_label(row)}[/{style}]",
            row.get("config_path") or "-",
        )
    console.print(table)


@agents_app.command(
    "register", help="Register biopb-mcp with a client (or all, with --all)."
)
def agents_register(
    client: Optional[str] = typer.Argument(
        None, help="Client id (e.g. claude-code); omit when using --all"
    ),
    all_: bool = typer.Option(
        False, "--all", help="Register with every detected client"
    ),
):
    """Register biopb-mcp with a client (or every detected client with --all)."""
    # For --all, skip clients that aren't even installed; an explicit id is
    # attempted regardless (a "register anyway" escape hatch).
    targets = _resolve_agent_targets(client, all_, states={"installed", "registered"})
    if not targets:
        console.print("[yellow]No agent clients detected to register.[/yellow]")
        raise typer.Exit(0)
    failures = 0
    for cid in targets:
        try:
            st = _agents.register(cid)
            console.print(f"[green]Registered[/green] {st['name']}")
        except _agents.AgentError as exc:
            failures += 1
            console.print(f"[red]{cid}: {exc}[/red]")
    console.print("[dim]Restart the client for the change to take effect.[/dim]")
    raise typer.Exit(1 if failures else 0)


@agents_app.command(
    "unregister", help="Remove biopb-mcp from a client (or all, with --all)."
)
def agents_unregister(
    client: Optional[str] = typer.Argument(
        None, help="Client id (e.g. claude-code); omit when using --all"
    ),
    all_: bool = typer.Option(
        False, "--all", help="Unregister from every currently registered client"
    ),
):
    """Remove biopb-mcp from a client (or every registered client with --all)."""
    targets = _resolve_agent_targets(client, all_, states={"registered"})
    if not targets:
        console.print("[yellow]biopb is not registered with any client.[/yellow]")
        raise typer.Exit(0)
    failures = 0
    for cid in targets:
        try:
            st = _agents.unregister(cid)
            console.print(f"[green]Unregistered[/green] {st['name']}")
        except _agents.AgentError as exc:
            failures += 1
            console.print(f"[red]{cid}: {exc}[/red]")
    raise typer.Exit(1 if failures else 0)


app.add_typer(agents_app, name="agents")


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


@app.command(
    "quick-start",
    hidden=not _is_windows(),
    help="Speed up biopb startup on Windows with a Defender exclusion.",
)
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
