"""Top-level CLI for BioPB."""

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from . import _agents, _locations, _web_auth
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
_add_optional_typer("tensor", "biopb.tensor.cli", "TensorFlight client diagnostics")

# ProcessImage client operations
_add_optional_typer("image", "biopb.image.cli", "ProcessImage client operations")

# Tensor server daemon management
server_app = typer.Typer(
    name="server",
    help="Biopb tensor-server diagnostics (cache-stats, migrate-config). The "
    "control plane (`biopb control`) owns the data-plane process lifecycle; the "
    "former standalone-daemon commands (start/stop/restart/status/logs) are gone.",
)
app.add_typer(server_app, name="server")

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
# from in this marker file -- a clean PEP 440 string (e.g. "0.6.7"), the
# auto-updater's baseline. This is the *deployment* version and is distinct from
# any single package's version: one release bundles the mutually-paired
# biopb / biopb-tensor-server / biopb-mcp / biopb-control set. Kept in sync with
# CONFIG_DIR/release.version in install/install.sh.
_RELEASE_VERSION_FILE = DEFAULT_CONFIG_DIR / "release.version"

# The wheels the installer bundles in one release-v* deployment (see
# install/install.sh). `biopb version` reports each separately so a version skew
# within the installed set is visible; any may be absent, hence "not installed".
_RELEASE_PACKAGES = ("biopb", "biopb-tensor-server", "biopb-mcp", "biopb-control")


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
    """Show cache hit/miss diagnostics from the running server.

    Liveness is the Flight query itself: an unreachable server yields no stats
    (handled below), so there is no separate PID-file gate -- the control plane
    now owns the data-plane process and writes no ``tensor-server.pid``.
    """
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
    from ._locations import (
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

mcp_app = typer.Typer(
    name="mcp",
    help="biopb-mcp MCP server: `view` opens the foreground napari viewer.",
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
    help="Biopb control plane: supervise the data plane (start/stop/status/run)",
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


def _control_endpoint() -> Tuple[str, int]:
    """The control-API (host, port) from the shared core-SDK location."""
    from ._endpoints import control_host, control_port

    return control_host(), control_port()


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


_LOCALHOST_ADDRS = {"127.0.0.1", "localhost", "::1"}


def _read_flight_host(config: Path) -> str:
    """The flight (gRPC) server's configured bind address (``server.host``).

    Defaults to ``0.0.0.0`` when the config can't be read, so an unreadable
    config is treated as a *public* bind — the fail-closed direction.
    """
    grpc_host = "0.0.0.0"
    if config.exists():
        try:
            from biopb_tensor_server.core.config import (
                load_config as _load_server_config,
            )

            grpc_host = _load_server_config(config).host
        except Exception:
            pass
    return grpc_host


def _resolve_mode(config: Path, remote: bool, token: Optional[str]) -> Optional[str]:
    """Resolve the data-plane token for the chosen deployment mode.

    Token enforcement is **independent** of the network mode: a token may be
    supplied — via ``--token`` or ``BIOPB_TENSOR_TOKEN`` — in *either* mode, so a
    single-machine deployment can still gate its listeners for defense-in-depth
    on a shared host. What ``--remote`` controls is the *bind address*: local
    binds every listener to loopback, remote binds the control UI + flight server
    publicly.

    - **Local** (default): every listener binds loopback. A token is *optional*;
      when one is supplied it is enforced (the browser then gates behind the
      unlock page, same as remote). Fail-closed: a config that binds the flight
      server publicly (``server.host`` non-loopback) with **no** token is refused,
      because that would expose data on the network without auth.
    - **Remote** (``--remote``): the flight server + control bind publicly, so a
      token is **required** — supplied, or else generated and printed.

    Returns the token to enforce (``None`` only when none is supplied in local
    mode).
    """
    grpc_host = _read_flight_host(config)
    flight_public = grpc_host not in _LOCALHOST_ADDRS

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
                "characters ([A-Za-z0-9_-]). Fix --token / BIOPB_TENSOR_TOKEN, "
                "or omit it to run tokenless (local) / auto-generate one (--remote)."
            )
            raise typer.Exit(1)
        return token

    # No token supplied.
    if remote:
        import secrets as _secrets

        token = _secrets.token_urlsafe(32)
        console.print(f"[bold green]Generated access token:[/bold green] {token}")
        return token

    # Local mode, tokenless — must not expose data on the network without auth.
    if flight_public:
        console.print(
            f"[red]The flight server config binds publicly (server.host="
            f"{grpc_host}), but no token would be enforced.[/red] "
            "Pass [bold]--token[/bold] (or set BIOPB_TENSOR_TOKEN) to enforce one, "
            "start with [bold]--remote[/bold], or set server.host to a loopback "
            "address (127.0.0.1)."
        )
        raise typer.Exit(1)
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


def _control_run_argv(
    *,
    config: Path,
    static_dir: Optional[Path],
    web_host: str,
    web_port: int,
    log_level: str,
    data_plane: bool,
    remote: bool,
) -> List[str]:
    """Build the `python -m biopb_control run ...` argv `control start` spawns.

    The core CLI resolves everything (grpc endpoint, mode, endpoint, log paths)
    and passes it explicitly, so biopb_control imports no server config
    (invariant I2). The supervised tensor server logs to tensor-server.log; the
    control plane's own output is redirected by the caller to control.log.

    The access token is **not** on this argv: a command line is world-readable
    (`ps aux`, Task Manager) on exactly the multi-user hosts a token is meant to
    protect (biopb/biopb#414). It travels only via ``BIOPB_TENSOR_TOKEN`` in the
    child env (set by the caller). ``--remote`` — not a secret — is the explicit
    public-deployment signal; it also binds the control's own listener publicly
    (0.0.0.0) so the browser UI is reachable off-box, while the local (default)
    mode keeps every listener on loopback.
    """
    grpc_host, grpc_port = _resolve_grpc_hostport(config)
    control_host, control_port = _control_endpoint()
    # Remote mode: bind the control's HTTP listener on all interfaces so the
    # dashboard/dataviewer is reachable off-box (the token gates it). The starter
    # still probes readiness over loopback, so _control_endpoint()'s host is kept
    # only for that probe, not passed through here.
    if remote:
        control_host = "0.0.0.0"
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
    if remote:
        argv.append("--remote")
    return argv


@control_app.command("start")
def control_start(
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Tensor-server config (JSON or TOML)"
    ),
    static_dir: Optional[Path] = typer.Option(
        DEFAULT_WEBAPP,
        "--static-dir",
        help="Web UI bundle the control serves at its root (the built web/ dist)",
    ),
    web_port: int = typer.Option(8814, "--web-port", help="Tensor-server HTTP port"),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Control log level"
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Serve on the network behind a token: bind the control (browser UI) "
        "and the flight server publicly, and require an access token. Without it "
        "(the default) every listener binds loopback; a token is optional (pass "
        "--token to enforce one).",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Access token (or set BIOPB_TENSOR_TOKEN). Enforced in either mode: "
        "required for --remote (auto-generated if omitted), optional in local "
        "mode (loopback bind either way). A local token gates the browser too, "
        "and biopb-mcp then needs BIOPB_TENSOR_TOKEN in its own environment to "
        "reach the data plane (biopb/biopb#470).",
    ),
    data_plane: bool = typer.Option(
        True,
        "--data-plane/--no-data-plane",
        help="Bring the data plane up on start (default). With --no-data-plane "
        "the control plane starts without it; a client brings it up on demand via the "
        "control API.",
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

    Two deployment modes: **local** (default) binds every listener to loopback —
    the single-machine 90% case, tokenless unless you pass ``--token`` (an
    optional defense-in-depth gate on a shared machine); and **remote**
    (``--remote``) binds the control's browser UI and the flight server publicly
    behind a *required* token, for serving a tensor deployment to other machines.
    The tensor HTTP sidecar always stays on loopback (the control proxies it),
    and the flight server's bind comes from the config's ``server.host`` — local
    mode refuses to start if that is non-loopback and no token is enforced.
    """
    _require_biopb_control()
    _ensure_dirs()

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

            control_host, control_port = _control_endpoint()
            if _port_listening(control_host, control_port):
                console.print(
                    f"[red]Control-plane port {control_host}:{control_port} is already in use.[/red]"
                )
                console.print(
                    "It is held by a process biopb is not tracking (an orphaned control plane, or "
                    "another login session), so [bold]biopb control stop[/bold] cannot reach "
                    "it. Identify and stop the owner, then retry."
                )
                raise typer.Exit(1)

            # The control plane owns the data plane exclusively, so refuse to start into a gRPC
            # port a stray server already holds -- otherwise the supervised child would
            # crash-loop on EADDRINUSE. Skipped for --no-data-plane (the plane comes
            # up on demand, guarded there too).
            if data_plane:
                grpc_host, grpc_port = _resolve_grpc_hostport(config)
                if _port_listening(grpc_host, grpc_port):
                    console.print(
                        f"[red]Data-plane gRPC port {grpc_host}:{grpc_port} is already "
                        "in use.[/red]"
                    )
                    console.print(
                        "The control plane owns the data plane exclusively and will not adopt a "
                        "server it did not start. Stop the process holding that port "
                        f"(`lsof -i :{grpc_port}` / `netstat -ano | findstr {grpc_port}`), "
                        "then retry -- or start with [bold]--no-data-plane[/bold]."
                    )
                    raise typer.Exit(1)

            resolved_token = _resolve_mode(config, remote, token)
            argv = _control_run_argv(
                config=config,
                static_dir=static_dir,
                # The sidecar always binds loopback; the control proxies it. Only
                # the control (browser UI) and flight server go public in --remote.
                web_host="127.0.0.1",
                web_port=web_port,
                log_level=log_level,
                data_plane=data_plane,
                remote=remote,
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
                console.print("  Data plane: starting (see 'biopb control status')")
            else:
                console.print("  Data plane: not started (--no-data-plane; on-demand)")
            console.print(f"  Logs: {log_file}")
    except LockTimeout:
        console.print(
            "[red]Another 'biopb control start' is already in progress and did not "
            "finish within 30s.[/red] Retry shortly, or check 'biopb control status'."
        )
        raise typer.Exit(1)


@control_app.command("stop")
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
    """
    _require_biopb_control()
    pid, token = _read_pid_record(CONTROL_PID_FILE)
    if not pid:
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


@control_app.command("status")
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
            "data_plane": (data_plane or None) if running else None,
        },
        table_rows=[
            ("Control", f"http://{control_host}:{control_port}"),
            ("Control API", "responding" if health else "not responding"),
            ("Data plane", dp_state),
            ("Data plane URL", data_plane.get("grpc_url", "-")),
            ("Restarts", str(data_plane.get("restarts", 0))),
        ],
    )


@control_app.command("logs")
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


@control_app.command("run")
def control_run(
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Tensor-server config (JSON or TOML)"
    ),
    static_dir: Optional[Path] = typer.Option(
        DEFAULT_WEBAPP,
        "--static-dir",
        help="Web UI bundle the control serves at its root (the built web/ dist)",
    ),
    web_port: int = typer.Option(8814, "--web-port", help="Tensor-server HTTP port"),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Control log level"
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Serve on the network behind a token (bind control + flight server "
        "publicly). Without it every listener binds loopback; a token is optional "
        "(pass --token to enforce one).",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Access token (or set BIOPB_TENSOR_TOKEN). Enforced in either mode: "
        "required for --remote (auto-generated if omitted), optional in local "
        "mode (loopback bind either way). A local token gates the browser too, "
        "and biopb-mcp then needs BIOPB_TENSOR_TOKEN in its own environment to "
        "reach the data plane (biopb/biopb#470).",
    ),
    data_plane: bool = typer.Option(
        True,
        "--data-plane/--no-data-plane",
        help="Bring the data plane up (default), or start without it (on-demand).",
    ),
):
    """Run the control plane in the foreground (Ctrl-C to stop).

    The foreground counterpart of `biopb control start`: no PID file, blocks this
    terminal, tears everything down on Ctrl-C. Useful for a systemd/launchd unit
    (let the service manager own the process) or for debugging supervision. See
    `biopb control start` for the local/remote mode model.
    """
    _require_biopb_control()
    _ensure_dirs()
    from biopb_control import run_control
    from biopb_control._supervisor import DataPlaneSpec

    grpc_host, grpc_port = _resolve_grpc_hostport(config)
    control_host, control_port = _control_endpoint()
    resolved_token = _resolve_mode(config, remote, token)
    # Remote mode binds the control's own listener publicly so the browser UI is
    # reachable off-box; the token gates it. The sidecar always stays on loopback.
    if remote:
        control_host = "0.0.0.0"
    spec = DataPlaneSpec(
        config=config,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        web_host="127.0.0.1",
        web_port=web_port,
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


@app.command("dashboard")
def dashboard(
    remote: bool = typer.Option(
        False,
        "--remote",
        help="If the control plane isn't already running, start it in remote mode "
        "(bind publicly behind a token). See 'biopb control start --remote'.",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Ensure the control plane is up but only print the dashboard URL "
        "instead of opening a browser.",
    ),
):
    """Open the biopb dashboard, starting the control plane first if needed.

    The one-command way in: it makes sure the control plane (which owns the data
    plane and serves the web UI) is running, then points your default web browser
    at the dashboard. Idempotent -- if the control plane is already up it just
    opens the page. This is what the desktop shortcut the installer creates runs.
    """
    control_host, control_port = _control_endpoint()
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
                web_port=8814,
                log_level="INFO",
                remote=remote,
                token=None,
                data_plane=True,
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
    help="Register biopb-mcp with local AI agent clients "
    "(Claude Code, Claude Desktop, Cursor, opencode).",
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


@agents_app.command("list")
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


@agents_app.command("register")
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


@agents_app.command("unregister")
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
