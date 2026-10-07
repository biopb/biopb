"""Top-level CLI for BioPB."""

import json
import os
import shutil
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

from . import _agents, _locations, _tls_material, _web_auth
from ._control import _endpoints
from ._control._endpoints import (
    flight_port_for as _flight_port,
    sidecar_port_for as _sidecar_port,
)
from ._locations import find_config
from .lifecycle.daemon import (
    detach_kwargs as _detach_kwargs,
    is_our_daemon as _is_our_daemon,
    read_pid_record as _read_pid_record,
    remove_pid_file as _remove_pid_file,
    stop_daemon as _stop_daemon,
    write_pid_file as _write_pid_file,
)
from .lifecycle.file_lock import LockTimeout, file_lock
from .lifecycle.proc import (
    is_process_running as _is_process_running,
    process_create_time as _process_create_time,
)

console = Console()

app = typer.Typer(
    name="biopb",
    help="BioPB: open protobuf/gRPC protocols for biomedical image processing",
)


class _LazySubcommands(typer.core.TyperGroup):
    """A subcommand group whose module is imported on first use, not at startup.

    The tensor/image subcommands need heavy optional dependencies (biopb[tensor])
    that may be absent or broken. The module loads when the group is invoked, its
    help is rendered, or completion lists it; `biopb --help` lists it unloaded.
    An import failure makes any `biopb <name> ...` print the error and install
    hint and exit 1, leaving the rest of the CLI working.
    """

    import_path: str = ""  # set on the per-group subclass by `_add_optional_typer`

    def __init__(self, **attrs) -> None:
        self._loaded: Optional[dict] = None
        self._error: Optional[BaseException] = None
        super().__init__(**attrs)

    # A property is the one hook covering every read of `commands`; the
    # constructor's assignment of an empty dict is discarded.
    @property
    def commands(self) -> dict:
        self._ensure_loaded()
        return self._loaded or {}

    @commands.setter
    def commands(self, value) -> None:
        pass

    def _ensure_loaded(self) -> None:
        if self._loaded is not None or self._error is not None:
            return
        import importlib

        try:
            module = importlib.import_module(self.import_path)
            self._loaded = dict(typer.main.get_group(module.app).commands)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any import error
            self._error = exc
            self.help = f"{self.help} (unavailable - optional dependencies missing)"

    def invoke(self, ctx: typer.Context):
        self._ensure_loaded()
        if self._error is not None:
            console.print(
                f"[red]The '{self.name}' commands are unavailable:[/red] {self._error}\n"
                r"[yellow]Install optional dependencies with: pip install 'biopb\[tensor]'[/yellow]"
            )
            raise typer.Exit(1)
        return super().invoke(ctx)

    def format_help(self, ctx, formatter) -> None:
        self._ensure_loaded()  # so a failed import's note is in the help text
        return super().format_help(ctx, formatter)


def _add_optional_typer(name: str, import_path: str, help: str) -> None:
    """Register `import_path`'s Typer `app` as the `name` subcommand group, lazily."""
    cls = type(f"_Lazy_{name}", (_LazySubcommands,), {"import_path": import_path})
    app.add_typer(typer.Typer(), name=name, help=help, cls=cls)


# TensorFlight client diagnostics
_add_optional_typer(
    "tensor",
    "biopb.tensor.cli",
    "Query a TensorFlight data plane (sources, tensors, stats, cache).",
)

# Ops client operations
_add_optional_typer("image", "biopb.image.cli", "Call algorithm servers (Ops).")

# On-disk locations come from the shared XDG-aware `_locations` module.
DEFAULT_WEBAPP = _locations.webapp_dir()

# Resolved via the dependency-light core module so the typer Option default does
# not import the heavy server config module.
DEFAULT_CONFIG = find_config()

# The control plane (`biopb-control`) is started as `python -m biopb_control run`;
# pidfile / detach / stop-sentinel plumbing lives here. It supervises the tensor
# server (which writes tensor-server.log); its own log is control.log.
CONTROL_PID_FILE = _locations.control_pid_file()


# One release-v* tag versions these together; the first one installed answers.
_RELEASE_PACKAGES = ("biopb-control", "biopb-mcp", "biopb-tensor-server")


def _release_version() -> str:
    """The product deployment version, from whichever release-v* wheel is here.

    Read from per-environment distribution metadata, not the user-global
    ``CONFIG_DIR/release.version`` marker (which the auto-updater reads itself).
    """
    for name in _RELEASE_PACKAGES:
        found = _package_version(name)
        if found not in ("not installed", "unknown"):
            return found
    return "not installed"


def _package_version(dist_name: str) -> str:
    """Installed version of distribution `dist_name`, or 'not installed'.

    Reads metadata rather than importing: reports what is *installed* (an
    editable checkout's ``__version__`` can differ), avoids heavy imports, and
    works when a package's runtime imports are broken.
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
        ("release", _release_version()),
        # The SDK ships on its own v* tag, independent of the release.
        ("biopb", _package_version("biopb")),
    ]

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


_rotate_log = _locations.rotate_log


# --- log tailing (`biopb control logs`) ---------------------------------- #

# Severity ranks for the `--level` filter.
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _tensor_line_level(line: str) -> Optional[str]:
    """Level of a data-plane log line, or None if it has none.

    tensor-server.log carries the server's own format (DEFAULT_LOG_FORMAT in
    biopb_tensor_server.logging_config): `[2026-06-12 10:00:00] WARNING
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

    # Logs rotate at 10 MB, so the file is small enough to read whole.
    existing = log_file.read_text(errors="replace").splitlines()
    tail = existing if lines <= 0 else existing[-lines:]
    for line in _filter_lines(tail, min_level, level_of):
        print(line)

    if not follow:
        raise typer.Exit(0)

    # Piped stdout is block-buffered; flush so `logs -f | grep` shows the tail.
    sys.stdout.flush()

    # Poll for appended lines; a changed inode or shrunk size means rotation.
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


def _plane_bind(grpc_bind: str, base_port: int) -> Tuple[str, int]:
    """The flight plane's bind: the address from ``--grpc-bind``, port from the base.

    Everything downstream (token required? TLS by default?) derives from this
    address through :func:`biopb._web_auth.host_is_public_bind`.
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
    """A daemon's liveness snapshot. `health` is an optional richer status dict
    (None if none is exposed or the query failed; probing never raises)."""

    listening: bool
    health: Optional[dict] = None


def _probe_daemon(
    host: str, port: int, health_fn: Optional[Callable[[], Optional[dict]]] = None
) -> Probe:
    """One uniform liveness/health snapshot for either SDK daemon (never raises).

    With `health_fn`, its answer fills `health` and defines liveness; without
    one, a TCP connect to (host, port) does.
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

    `json_fields` and `table_rows` carry per-daemon extras; `table_rows` go
    between the PID row and the trailing PID-file / Log-file rows.
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
# `view` runs the server in a child process (`python -m biopb_mcp.mcp --view`),
# so this CLI never imports the heavy MCP/napari stack. biopb-mcp is optional;
# _require_biopb_mcp() gives an install hint when it is absent.
# ---------------------------------------------------------------------------

# Commands pass an explicit one-line `help=`, which Typer prefers over the
# (maintainer-facing) docstring.
mcp_app = typer.Typer(
    name="mcp",
    help="Run a foreground napari viewer session (biopb-mcp).",
)


def _require_biopb_mcp() -> None:
    """Exit(1) with an install hint if the biopb-mcp package is not importable.

    Checks the import spec only, so the MCP/napari stack is never loaded.
    """
    import importlib.util

    if importlib.util.find_spec("biopb_mcp") is None:
        console.print(
            "[red]The 'mcp' commands require the biopb-mcp package, which is "
            "not installed.[/red]\n"
            r"[yellow]Install it with: pip install 'biopb-mcp\[napari]'[/yellow]"
        )
        raise typer.Exit(1)


def _require_control_for_view() -> None:
    """Exit(1) unless a control plane answers — or ``$BIOPB_TENSOR_URL`` is set.

    The viewer's data comes from the control's plane; failing here avoids a
    napari load into an empty Tensor Browser. The env override names a plane
    directly, so it bypasses this check.

    "Answered" is :func:`_query_control_health`, shared with ``control status``.
    Its 2s budget is deliberate: ``/health`` takes the supervisor lock, so a busy
    control answers late, and a false negative here is a hard exit.
    """
    from . import ENV_TENSOR_URL

    if os.environ.get(ENV_TENSOR_URL, "").strip():
        return
    if _query_control_health(*_control_endpoint()) is not None:
        return
    console.print(
        "[red]No biopb control plane is running,[/red] so there is no data plane "
        "for the viewer to read.\n"
        "[yellow]Start it first: biopb control start[/yellow]"
    )
    raise typer.Exit(1)


def _port_listening(host: str, port: int, timeout: float = 0.3) -> bool:
    """Whether a TCP connection to (host, port) succeeds."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _await_listening(pid: int, host: str, port: int, timeout: float) -> bool:
    """Block until (host, port) accepts a connection (True), or the process dies
    or `timeout` elapses (False). Callers re-check liveness to tell a crash from
    a slow bind."""
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

    Runs `biopb-mcp --view` as a foreground child sharing this terminal's stdio
    and process group, so Ctrl-C reaches it. Writes no PID file; still serves
    /mcp so an agent may attach.

    Requires a running control plane, checked before the child's slow napari
    import; it is not started here because a person is at the terminal.
    """
    _require_biopb_mcp()
    _require_control_for_view()
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
    # Foreground: no _detach_kwargs, so Ctrl-C reaches the child.
    try:
        process = subprocess.Popen(cmd, env=os.environ.copy())
    except OSError as exc:
        console.print(f"[red]Could not launch the viewer:[/red] {exc}")
        raise typer.Exit(1)
    try:
        raise typer.Exit(process.wait())
    except KeyboardInterrupt:
        # Ctrl-C already reached the child; let it tear down, then force-reap.
        try:
            process.wait(timeout=20)
        except Exception:
            process.kill()
        raise typer.Exit(0)


app.add_typer(mcp_app, name="mcp")


# ---------------------------------------------------------------------------
# biopb control: the control plane (supervises the durable planes)
# ---------------------------------------------------------------------------
# `biopb control` manages the `biopb-control` process, the durable root that
# supervises the tensor server.
control_app = typer.Typer(
    name="control",
    help="Manage the control plane, which supervises the data plane.",
)


def _require_biopb_control() -> None:
    """Exit(1) with an install hint if the biopb-control package is absent.

    Checks the import spec only, like _require_biopb_mcp.
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

    ``cryptography`` is an opt-in extra, so a default install cannot mint the
    self-signed certificate ``--tls`` needs. Without this check the control
    starts cleanly while its supervised plane crash-loops, with the error only in
    ``tensor-server.log``.

    The control and plane are spawned via ``sys.executable``, so this process's
    import spec is the plane's, and ``sys.executable`` names the environment to
    install into.
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
    # Copied verbatim: soft_wrap avoids line breaks; markup off because Rich
    # would eat a bare `[tls]` as a style tag.
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

    For commands that talk to a control someone else started; a non-default
    ``--base-port`` is followed via ``biopb._locations.control_runtime_file``.
    Never used to decide a *bind* (see :func:`_control_bind_endpoint`): a crashed
    control's stale record must not dictate where the next one listens.
    """
    from ._control._endpoints import control_host, control_port

    return control_host(), control_port()


def _control_bind_endpoint(base_port: int) -> Tuple[str, int]:
    """Where a control we are *starting* should listen: base+3, env still wins.

    ``BIOPB_CONTROL_HOST`` / ``BIOPB_CONTROL_PORT`` take top precedence;
    otherwise the port comes from the base, never from the published record
    (which describes some other, possibly dead, control).
    """
    from ._control._endpoints import CONTROL_DEFAULT_HOST, control_port_for

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

    A public bind publishes the flight plane only: the control serves plaintext
    HTTP, so publishing it would expose the data-plane token (which unlocks the
    admin API) in the clear.
    """
    import socket

    host = socket.gethostname() or "<host>"
    console.print("  Browser UI: loopback only. From another machine, tunnel it:")
    # soft_wrap: the line is copied verbatim.
    console.print(
        f"    [bold]ssh -L {control_port}:localhost:{control_port} {host}[/bold]",
        soft_wrap=True,
        highlight=False,
    )
    console.print(f"    then open http://localhost:{control_port}")


def _control_log_file() -> Path:
    """The control plane's own log (distinct from the data plane's tensor-server.log)."""
    return _locations.control_log()


def _write_control_pid(pid: int) -> None:
    _ensure_dirs()
    _write_pid_file(CONTROL_PID_FILE, pid, _process_create_time(pid))


def _remove_control_pid() -> None:
    _remove_pid_file(CONTROL_PID_FILE)


def _control_shutdown_sentinel() -> Path:
    """The control plane's Windows stop-sentinel path (watched by biopb_control._run)."""
    return _locations.control_stop_sentinel()


def _control_start_lock() -> Path:
    """Cross-process lock file serializing `biopb control start`.

    Concurrent starters (launcher, installer, agent sessions) would otherwise
    both see "no pidfile" and the bind-loser could clobber the winner's pidfile,
    orphaning a control `control stop` cannot reach. See biopb.lifecycle.file_lock.
    """
    return CONTROL_PID_FILE.parent / "control.start.lock"


def _resolve_grpc_bind(grpc_bind: Optional[str], remote: bool) -> str:
    """The flight bind from ``--grpc-bind``, honoring the deprecated ``--remote``.

    ``--remote`` is a deprecated alias for ``--grpc-bind 0.0.0.0``; an explicit
    ``--grpc-bind`` wins.
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

    A public flight bind defaults TLS on; loopback defaults it off. The
    cleartext combination (``--grpc-bind 0.0.0.0 --no-tls``) must be asked for
    by name. ``--tls`` alone means "encrypted, loopback only".
    """
    if tls is not None:
        return tls
    return _web_auth.host_is_public_bind(grpc_bind)


def _resolve_tls_material(
    tls: bool, tls_cert: Optional[Path], tls_key: Optional[Path]
) -> bool:
    """Validate BYO TLS material and return whether the plane serves TLS.

    A supplied cert means TLS even without ``--tls``; the control advertises the
    plane's scheme from this result.

    Validated here for the same reason as :func:`_require_tls_extra` (otherwise
    the supervised child crash-loops under a clean-looking start). Each file is
    opened rather than stat'd, since an unreadable key passes ``is_file()``. The
    rule is shared via :mod:`biopb._tls_material`.
    """
    if (tls_cert is None) != (tls_key is None):
        console.print("[red]--tls-cert and --tls-key must be given together.[/red]")
        raise typer.Exit(2)
    for label, path in (("--tls-cert", tls_cert), ("--tls-key", tls_key)):
        if path is None:
            continue
        try:
            _tls_material.read_pem(path, label)
        except _tls_material.TlsMaterialError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2) from None
    return tls or tls_cert is not None


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

    A token (``--token`` or ``BIOPB_TENSOR_TOKEN``) is enforced with either bind.

    - **Loopback** (the default): optional.
    - **Public**: required -- supplied, else generated and printed.

    The control (browser UI) stays on loopback with either bind (plaintext HTTP).
    Exposure is decided by :func:`biopb._web_auth.host_is_public_bind`, shared with
    the tensor ``launch`` and the control's bind guard, so they cannot drift.

    Returns the token to enforce (``None`` only when none is supplied on a
    loopback bind).
    """
    token = token or os.environ.get("BIOPB_TENSOR_TOKEN")
    if token:
        # Same rule as the tensor `launch`; an invalid token accepted here would
        # be silently regenerated or ignored downstream.
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

    Printed at startup because ``--base-port`` makes the port a computation.
    """
    host, port = _plane_bind(grpc_bind, base_port)
    scheme = "grpcs" if tls else "grpc"
    authority = f"[{host}]" if ":" in host else host
    return f"{scheme}://{authority}:{port}"


def _guard_ports_free(base_port: int, grpc_bind: str, data_plane: bool) -> None:
    """Refuse to start into a port something already holds, naming which one.

    Checks all three listeners; an unguarded collision surfaces as a control
    that starts clean and then crash-loops its plane.
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


def _resolve_url_prefix(url_prefix: Optional[str]) -> Optional[str]:
    """Normalize ``--url-prefix`` / ``$BIOPB_URL_PREFIX``, or exit naming the fault.

    Rejected before anything is spawned: the prefix ends up in the served
    ``<base href>``. The rule lives in biopb_control.
    """
    from biopb_control._control import normalize_url_prefix

    try:
        return normalize_url_prefix(url_prefix)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)


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
    tls_cert: Optional[Path] = None,
    tls_key: Optional[Path] = None,
    san: Optional[List[str]] = None,
    url_prefix: Optional[str] = None,
    grpc_external_location: Optional[str] = None,
) -> List[str]:
    """Build the `python -m biopb_control run ...` argv `control start` spawns.

    Everything (binds, ports, log paths) is resolved here and passed explicitly,
    so biopb_control imports no server config and knows nothing of base ports.

    The access token is not on this argv (world-readable); it travels only via
    ``BIOPB_TENSOR_TOKEN`` in the child env, set by the caller. The child
    re-derives "public?" from ``--grpc-host``.
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
    if url_prefix:
        argv += ["--url-prefix", url_prefix]
    if grpc_external_location:
        argv += ["--grpc-external-location", grpc_external_location]
    if not data_plane:
        argv.append("--no-data-plane")
    if tls:
        argv.append("--tls")
    if tls_cert and tls_key:
        argv += ["--tls-cert", str(tls_cert), "--tls-key", str(tls_key)]
    for name in san or ():
        argv += ["--san", name]
    return argv


# --- shared `control start` / `control run` options ----------------------- #
#
# One `typer.Option` per flag, shared so the commands' flags and help agree.
# Command-specific flags stay declared inline.

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
    "alongside another user's — give that one its own BIOPB_STATE_HOME too.",
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
_OPT_TLS_CERT = typer.Option(
    None,
    "--tls-cert",
    help="PEM certificate chain the data plane serves, instead of the "
    "self-signed one it mints into the state tree. Implies --tls, and needs no "
    "'cryptography' extra. Use it to serve one long-lived certificate that "
    "outlives a single launch — one cert shared by every node a scheduler might "
    "pick — so clients pin once instead of per launch. It must carry a loopback "
    "SAN (localhost / 127.0.0.1) besides the names clients dial, or the "
    "co-located sidecar cannot reach the flight plane; and a client on this "
    "machine reads its anchor from the state tree, so put a copy of the cert "
    "(not the key) there too. Requires --tls-key.",
)
_OPT_TLS_KEY = typer.Option(
    None,
    "--tls-key",
    help="PEM private key paired with --tls-cert.",
)
_OPT_SAN = typer.Option(
    None,
    "--san",
    help="Extra hostname or IP to put in the certificate the data plane mints "
    "(repeatable). Needed when clients dial a name this host cannot discover "
    "itself — a NAT/VPN address, a CNAME, the scheduler's name for this node — "
    "because gRPC verifies the dialed name against the SANs even though trust "
    "comes from the client's pin. Applies only when the cert is generated: it is "
    "ignored once one exists (re-mint with `biopb-tensor-server cert init "
    "--force --san ...`) and by --tls-cert.",
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
_OPT_URL_PREFIX = typer.Option(
    None,
    "--url-prefix",
    envvar="BIOPB_URL_PREFIX",
    help="Path prefix a reverse proxy publishes the browser UI under, instead of "
    "the origin root — e.g. --url-prefix /node/$host/$port for an Open OnDemand "
    "interactive app, whose route passes the full path through and rewrites "
    "nothing. Requests under the prefix are stripped before routing and the SPA "
    "shell is rewritten to point back at it; unprefixed requests keep working. "
    "Explicit configuration only: the prefix is never read off a request header "
    "such as X-Forwarded-Prefix.",
)
_OPT_DATA_PLANE = typer.Option(
    True,
    "--data-plane/--no-data-plane",
    help="Bring the data plane up on start (default). With --no-data-plane the "
    "control plane starts without it; a client brings it up on demand via the "
    "control API.",
)
_OPT_GRPC_EXTERNAL_LOCATION = typer.Option(
    None,
    "--grpc-external-location",
    envvar="BIOPB_GRPC_EXTERNAL_LOCATION",
    help="The address a remote client should dial to reach the data plane, "
    "advertised via its `health` action (biopb/biopb#1158) -- e.g. "
    "'grpc://hostname:8815', or a scheduler-assigned FQDN on an HPC job. "
    "Required when --grpc-bind is a public address: nothing here can guess a "
    "reachable address for a wildcard bind. Pure passthrough -- the data "
    "plane is the single place this is validated and enforced.",
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
    tls_cert: Optional[Path] = _OPT_TLS_CERT,
    tls_key: Optional[Path] = _OPT_TLS_KEY,
    san: Optional[List[str]] = _OPT_SAN,
    token: Optional[str] = _OPT_TOKEN,
    data_plane: bool = _OPT_DATA_PLANE,
    url_prefix: Optional[str] = _OPT_URL_PREFIX,
    grpc_external_location: Optional[str] = _OPT_GRPC_EXTERNAL_LOCATION,
    remote: bool = typer.Option(
        False,
        "--remote",
        hidden=True,
        help="Deprecated alias for --grpc-bind 0.0.0.0.",
    ),
):
    """Start the biopb control plane as a background daemon.

    The control plane is the sole owner of the tensor (data) plane: it spawns it
    (by default on start), restarts it on crash, and never adopts a server it did
    not start -- if the gRPC port is in use, `control start` refuses, so
    `biopb control stop` is always a complete teardown.

    **Ports** come from ``--base-port`` (default 8810): control = base+3,
    sidecar = base+4, flight = base+5. A moved control publishes where it landed,
    so `stop` / `status` / `logs` and biopb-mcp follow it.

    **Exposure** comes from ``--grpc-bind`` (default 127.0.0.1), read through the
    predicate shared with the tensor `launch` and the control's guard. A public
    address requires a token and defaults TLS on, and also requires
    ``--grpc-external-location`` (a wildcard bind is not dialable), forwarded
    verbatim to the data plane, which enforces it.

    **Long-lived certificate.** ``--tls`` alone mints a self-signed cert that
    clients pin on first connect. ``--tls-cert`` / ``--tls-key`` serve an
    operator's own cert instead (no ``cryptography`` needed); ``--san`` names
    addresses a minted cert cannot discover. A supplied cert needs a
    ``localhost`` / ``127.0.0.1`` SAN (the sidecar dials over loopback) and a
    copy (never the key) at ``state/biopb/tls/server-cert.pem`` for local SDK
    clients.

    Only the flight plane is ever published; the sidecar and the control stay on
    loopback (plaintext HTTP). To reach the UI from another machine, tunnel it:
    ``ssh -L 8813:localhost:8813 <host>``.
    """
    _require_biopb_control()
    grpc_bind = _resolve_grpc_bind(grpc_bind, remote)
    url_prefix = _resolve_url_prefix(url_prefix)
    tls = _resolve_tls_material(_resolve_tls(tls, grpc_bind), tls_cert, tls_key)
    # A BYO cert needs no `cryptography`; only a minted one does.
    if tls and tls_cert is None:
        _require_tls_extra()
    _warn_public_plaintext(grpc_bind, tls)
    _ensure_dirs()

    # Serialize concurrent starts (see _control_start_lock); held through the
    # readiness wait so a blocked starter sees a fully started control.
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

            # A foreground control writes only the endpoint record, no pid file.
            live = _live_foreground_control()
            if live:
                record, live_pid = live
                console.print(
                    f"[yellow]biopb control already running (PID {live_pid}, foreground, "
                    f"{record.get('host')}:{record.get('port')})[/yellow]"
                )
                raise typer.Exit(0)

            control_host, control_port = _control_bind_endpoint(base_port)
            _guard_ports_free(base_port, grpc_bind, data_plane)

            resolved_token = _resolve_mode(grpc_bind, token)
            argv = _control_run_argv(
                config=config,
                static_dir=static_dir,
                # The sidecar always binds loopback; the control proxies it.
                web_host="127.0.0.1",
                base_port=base_port,
                log_level=log_level,
                data_plane=data_plane,
                grpc_bind=grpc_bind,
                tls=tls,
                tls_cert=tls_cert,
                tls_key=tls_key,
                san=san,
                url_prefix=url_prefix,
                grpc_external_location=grpc_external_location,
            )

            log_file = _control_log_file()
            _rotate_log(log_file)
            console.print("[green]Starting biopb control plane...[/green]")
            console.print(f"  Config: {config}")
            env = os.environ.copy()
            if resolved_token:
                # Env only, never the argv; biopb_control reads it back.
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
                if grpc_external_location:
                    console.print(f"  Data plane advertises: {grpc_external_location}")
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

    A foreground control writes no pid file, so its endpoint record is the only
    trace. Verified for identity, not just liveness: a crash can strand the
    record and the pid may be recycled, so `_is_our_daemon` compares the recorded
    create-time token. Falls back to liveness when the record has no usable token.
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

    Stopping the control also shuts down the supervised tensor server. Only
    reaches a daemonized control; a foreground one belongs to its terminal or
    service manager, so this reports it and declines.
    """
    _require_biopb_control()
    pid, token = _read_pid_record(CONTROL_PID_FILE)
    if not pid:
        # `status` reports a foreground control as Running; stay consistent.
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

    # A foreground control has no pid file; fall back to its endpoint record.
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

    Two logs: control.log (default) and the supervised tensor server's
    tensor-server.log. Read straight off disk, so it works on a stopped or
    wedged control.
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
    "run",
    help="Removed -- use `biopb control start`, or `biopb-control run` for a "
    "true foreground process.",
)
def control_run() -> None:
    """Removed; points to `biopb control start` or `biopb-control run`."""
    console.print(
        "[red]`biopb control run` has been removed (biopb/biopb#736).[/red]\n"
        "For the same deployment with the same defaults, use [bold]biopb "
        "control start[/bold].\n"
        "For a true foreground process, use [bold]biopb-control run[/bold] "
        "(or `python -m biopb_control run` if that script isn't on PATH) -- "
        "but note it fills in none of this command's defaults: `--config` and "
        "`--static-dir` must both be passed explicitly, and every other "
        "setting individually rather than derived from --base-port. See "
        "`biopb-control run --help`."
    )
    raise typer.Exit(2)


app.add_typer(control_app, name="control")


@app.command(
    "dashboard", help="Open the biopb dashboard, starting the control plane if needed."
)
def dashboard(
    base_port: int = _OPT_BASE_PORT,
    grpc_bind: Optional[str] = _OPT_GRPC_BIND,
    grpc_external_location: Optional[str] = _OPT_GRPC_EXTERNAL_LOCATION,
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

    Ensures the control plane is running, then opens the dashboard; idempotent.
    This is what the installer's desktop shortcut runs.

    ``--base-port`` / ``--grpc-bind`` / ``--grpc-external-location`` are
    forwarded to `biopb control start` and only matter when there is nothing
    running to open.
    """
    # Prefer a serving control (finds one `--base-port` moved), else where we
    # would start one.
    control_host, control_port = _control_endpoint()
    if not _port_listening(control_host, control_port):
        control_host, control_port = _control_bind_endpoint(base_port)
    url = f"http://{control_host}:{control_port}"

    if _port_listening(control_host, control_port):
        console.print(f"[green]biopb control plane already running[/green] ({url})")
    else:
        # control_start returns once the control API listens; a non-zero
        # typer.Exit means it never came up.
        #
        # Every parameter must be passed explicitly: called as a plain function,
        # typer defaults are not applied and arrive as `OptionInfo` sentinels.
        try:
            control_start(
                config=DEFAULT_CONFIG,
                static_dir=DEFAULT_WEBAPP,
                base_port=base_port,
                log_level="INFO",
                grpc_bind=grpc_bind,
                tls=None,
                tls_cert=None,
                tls_key=None,
                san=None,
                token=None,
                data_plane=True,
                remote=remote,
                url_prefix=None,
                grpc_external_location=grpc_external_location,
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
# Uses the stdlib-only catalog in biopb._agents, shared with the dashboard.
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
    """Human label for a status row, annotating a stale (``drifted``) entry."""
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

    An explicit ``client`` acts on that one (validated); ``--all`` acts on every
    client whose state is in ``states``. Exits 1 on a bad/missing selector.
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
    # --all skips uninstalled clients; an explicit id is attempted regardless.
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
# skip-windows-defender: Defender exclusion for the biopb install
# ---------------------------------------------------------------------------
# Defender scanning of biopb's DLLs / .pyd / .pyc is the largest first-start
# cost on Windows. Excluding the install trees needs admin, so it is an opt-in
# command, separate from the installer's admin-free bytecode precompile.


def _is_windows() -> bool:
    """Whether we're on Windows.

    A function so tests can simulate Windows without patching `os.name`, which
    `pathlib` reads.
    """
    return os.name == "nt"


def _defender_targets() -> List[str]:
    """The install trees to exclude -- every tree this interpreter reads at startup.

    A uv tool venv points at a separate base Python, so two trees are scanned:

    * ``sys.prefix``      -- the tool env's site-packages (heavy deps; mixed
      ``.pyd`` / ``.dll``, so the directory is excluded, not ``*.pyd``).
    * ``sys.base_prefix`` -- the base interpreter, stdlib ``.pyd`` and
      ``pythonXY.dll``.

    They coincide for a non-venv install, so dedup. Taken from the running
    interpreter, never a hardcoded uv path. Sorted for stable order.
    """
    return sorted({str(Path(p).resolve()) for p in (sys.prefix, sys.base_prefix)})


def _ps_string_array(paths: List[str]) -> str:
    """A PowerShell array literal of single-quote-safe path strings (`'a', 'b'`)."""
    return ", ".join("'" + p.replace("'", "''") + "'" for p in paths)


def _run_elevated_ps(inner: str) -> int:
    """Run a PowerShell snippet elevated (one UAC prompt); return its exit code.

    A nonzero code also covers the launch failing (e.g. UAC declined).
    """

    # utf-8-sig: PowerShell 5.1 reads a BOM-less script as the ANSI code page,
    # misreading non-ASCII install paths while still exiting 0.
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
    """Add/remove Defender exclusions for `targets` in one elevated session, then verify.

    Tamper Protection or Intune/GPO can silently no-op the write even when
    elevated, so the snippet re-reads Get-MpPreference (exit 0 = all took,
    3 = at least one blocked).
    """
    verb = "Add" if add else "Remove"
    fail = "-not ($excl -contains $p)" if add else "($excl -contains $p)"
    ps_array = _ps_string_array(targets)
    # Placeholders, not an f-string, to avoid escaping PowerShell braces; paths
    # are substituted last so they cannot be re-interpreted as placeholders.
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
    """Print whether the biopb trees are Defender exclusions (best-effort, no admin).

    Reports 'unknown' when Get-MpPreference is unreadable, and PARTIAL when only
    some trees are excluded.
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
        console.print("  Remove it with: biopb skip-windows-defender --disable")
    elif out == "OFF":
        console.print("Defender exclusion is [yellow]not set[/yellow] for:")
        console.print(joined)
        console.print(
            "  Enable it for a faster startup (needs admin): biopb skip-windows-defender --enable"
        )
    elif out == "PARTIAL":
        console.print(
            "[yellow]Defender exclusion is only partially set[/yellow] "
            "-- some biopb trees are excluded, some aren't:"
        )
        console.print(joined)
        console.print(
            "  Complete it (needs admin): biopb skip-windows-defender --enable"
        )
    else:
        console.print(
            "[yellow]Could not read the Defender exclusion state[/yellow] "
            "(Get-MpPreference unavailable). Enable with: biopb skip-windows-defender --enable"
        )


@app.command(
    "skip-windows-defender",
    hidden=not _is_windows(),
    help="Speed up biopb startup on Windows with a Defender exclusion.",
)
def skip_windows_defender(
    enabled: Optional[bool] = typer.Option(
        None,
        "--enable/--disable",
        help="Enable (add) or disable (remove) the Defender exclusion; "
        "omit to show the current status.",
    ),
):
    """Speed up biopb startup on Windows via a Defender exclusion (issue #384).

    Adds (or removes, with --disable) Defender exclusions for the tool env and
    base Python so biopb's files aren't rescanned. Needs admin (one UAC prompt);
    reversible. Windows only.
    """
    if not _is_windows():
        console.print(
            "[yellow]skip-windows-defender is Windows-only[/yellow] -- Defender exclusions "
            "don't apply on this platform (nothing to do)."
        )
        raise typer.Exit(0)

    targets = _defender_targets()
    if enabled is None:
        _defender_status(targets)
        return
    _defender_exclusion(targets, add=enabled)


def _saved_uninstaller() -> Path:
    """The uninstaller the installer saved for the release that installed biopb."""
    name = "uninstall.cmd" if _is_windows() else "uninstall.sh"
    return _locations.data_dir() / "uninstall" / name


@app.command(
    help="Remove biopb: stop services, unregister from agents, delete the install."
)
def uninstall(
    purge: bool = typer.Option(
        False,
        "--purge",
        help="Also delete biopb's config and cached/state data. Your image data "
        "is never touched.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
):
    """Uninstall biopb by running the uninstaller saved by the install.

    Hands over to the uninstaller saved by the install. It deletes the
    environment this process runs from, so it replaces the process (POSIX) or
    outlives it in its own console (Windows).
    """
    script = _saved_uninstaller()
    if not script.is_file():
        console.print(
            f"[red]No saved uninstaller at {script}.[/red] An install older than "
            "this has none: run its release's `install.sh --uninstall` (POSIX) or "
            "`install.ps1 -Uninstall` (Windows)."
        )
        raise typer.Exit(1)

    what = "biopb, including its config and cached data" if purge else "biopb"
    if not yes and not typer.confirm(f"Uninstall {what}?"):
        raise typer.Exit(1)

    if _is_windows():
        # uninstall.ps1 asks about config/cache unless told; answer for the user.
        args = [str(script), "-Purge" if purge else "-KeepData"]
        subprocess.Popen(  # noqa: S603 - our own saved script
            args, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        )
        console.print("Uninstaller started in a new window.")
        return

    bash = shutil.which("bash")
    if bash is None:
        console.print("[red]bash not found; cannot run the uninstaller.[/red]")
        raise typer.Exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(bash, [bash, str(script), *(["--purge"] if purge else [])])


if __name__ == "__main__":
    app()
