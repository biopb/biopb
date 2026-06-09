"""CLI for TensorFlight server.

Commands:
    serve      Start the Flight server
    launch     Start the Flight server and HTTP sidecar for web app
    validate   Validate a config file
    list       List tensors in a config file
"""

import os
import secrets
import signal
import threading
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.downsample import configure_compute_backend
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.file_backend import ArrowFileBackend
from biopb_tensor_server.config import (
    CacheConfig,
    MetadataDbConfig,
    ServerConfig,
    SourceConfig,
    load_config,
    resolve_all_sources,
)
from biopb_tensor_server.discovery import discover_sources_async, is_remote_url
from biopb_tensor_server.http_server import run as run_http_server
from biopb_tensor_server.logging_config import get_log_level_from_env, setup_logging
from biopb_tensor_server.metadata_db import MetadataDatabase
from biopb_tensor_server.precache import PrecacheWorker
from biopb_tensor_server.server import TensorFlightServer
from biopb_tensor_server.source_manager import create_source_manager
from biopb_tensor_server.watcher import get_watcher

app = typer.Typer(
    name="biopb-tensor-server",
    help="BioPB Tensor: Arrow Flight server for multi-dimensional arrays",
)
console = Console()

diag_app = typer.Typer(help="Diagnostic commands for a running TensorFlight server")
app.add_typer(diag_app, name="diagnose")


def _install_sigterm_handler() -> None:
    """Make SIGTERM behave like Ctrl+C (KeyboardInterrupt).

    `biopb server stop`, `docker/singularity stop`, and SLURM all terminate the
    server with SIGTERM, which Python ignores (default disposition) for a
    blocking call like the Flight server's serve(). Translating it into a
    KeyboardInterrupt lets the same graceful-shutdown path run, so the file
    cache process lock is released instead of being left behind as a stale lock.

    Must be called from the main thread; no-op if that's not possible.
    """
    def _handler(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        # Not on the main thread (e.g. under some test runners) - skip.
        pass


def _graceful_shutdown(
    source_manager, watcher, flight_server, precache_worker=None
) -> None:
    """Best-effort orderly shutdown.

    Stops the precache worker and source discovery and the filesystem watcher,
    shuts down the Flight server, and closes the cache manager so the
    file-backend process lock is released. Each step is isolated so a failure in
    one still lets the cache lock be released (the important part for clean
    restarts). The precache worker is stopped first so no new warm work starts
    during teardown.
    """
    for label, action in (
        ("precache worker", lambda: precache_worker and precache_worker.stop()),
        ("source manager", lambda: source_manager and source_manager.stop()),
        ("watcher", lambda: watcher and watcher.stop()),
        ("flight server", lambda: flight_server and flight_server.shutdown()),
    ):
        try:
            action()
        except Exception as e:  # noqa: BLE001 - shutdown must not raise
            console.print(f"[yellow]Error stopping {label}: {e}[/yellow]")

    # Release the file cache process lock (no-op for the memory backend).
    manager = CacheManager.get_instance()
    if manager is not None:
        try:
            manager.close()
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Error closing cache: {e}[/yellow]")


def _setup_flight_server(
    server_config: ServerConfig,
    host: Optional[str] = None,
    port: Optional[int] = None,
    compute_backend: Optional[str] = None,
    gpu_min_input_mb: Optional[float] = None,
    gpu_min_linear_input_mb: Optional[float] = None,
    gpu_memory_safety_factor: Optional[int] = None,
    gpu_min_merged_chunks: Optional[int] = None,
    writable: Optional[bool] = None,
    token: Optional[str] = None,
) -> Tuple[TensorFlightServer, Optional[object], Optional[object]]:
    """Set up the Flight server with cache, sources, and monitoring.

    Args:
        server_config: Loaded server configuration
        host: Override host
        port: Override port
        compute_backend: Override compute backend policy
        gpu_* params: Override GPU policy parameters
        token: Access token for Flight server authentication

    Returns:
        Tuple of (flight_server, source_manager, watcher, precache_worker)

    Raises:
        typer.Exit: If no sources configured or no sources loaded successfully
    """
    # Apply overrides
    host = host or server_config.host
    port = port or server_config.port
    effective_writable = writable if writable is not None else server_config.writable
    write_dir = server_config.write_dir

    configure_compute_backend(
        force_backend=compute_backend or server_config.compute_backend,
        gpu_min_input_bytes=int(
            (
                gpu_min_input_mb
                if gpu_min_input_mb is not None
                else server_config.gpu_min_input_mb
            )
            * 1024
            * 1024
        ),
        gpu_min_linear_input_bytes=int(
            (
                gpu_min_linear_input_mb
                if gpu_min_linear_input_mb is not None
                else server_config.gpu_min_linear_input_mb
            )
            * 1024
            * 1024
        ),
        gpu_memory_safety_factor=gpu_memory_safety_factor
        or server_config.gpu_memory_safety_factor,
        gpu_min_merged_chunks=gpu_min_merged_chunks
        or server_config.gpu_min_merged_chunks,
    )

    # Initialize cache manager for virtual chunks
    cache_config = server_config.cache
    if cache_config.backend == "memory":
        CacheManager.initialize(cache_config)
        console.print(
            "[green]Virtual chunk cache initialized:[/green] "
            f"backend=memory, "
            f"max_entries={cache_config.memory_max_entries}, "
            f"max_bytes={cache_config.memory_max_bytes // (1024 * 1024)}MB"
        )
        console.print("[green]Raw chunk cache: OS page cache[/green]")
    elif cache_config.backend == "file":
        try:
            manager = CacheManager.initialize(cache_config)
        except OSError as e:
            # Cache dir not writable (e.g. read-only HPC scratch). Fall back to
            # the in-memory backend so the server still starts; the localhost
            # cache-file fast path (issue #9) is simply unavailable.
            console.print(
                f"[yellow]File cache unavailable at {cache_config.file_cache_dir} "
                f"({e}); falling back to in-memory cache.[/yellow]"
            )
            manager = CacheManager.initialize(CacheConfig(
                backend="memory",
                memory_max_entries=cache_config.memory_max_entries,
                memory_max_bytes=cache_config.memory_max_bytes,
            ))
        if isinstance(manager.backend, ArrowFileBackend):
            console.print(
                "[green]Virtual chunk cache initialized:[/green] "
                f"backend=file, "
                f"cache_dir={cache_config.file_cache_dir}, "
                f"max_segment_mb={cache_config.file_max_segment_bytes // (1024 * 1024)}, "
                f"max_total_gb={cache_config.file_max_total_bytes // (1024 * 1024 * 1024)}"
            )
            # Check for recovery status
            recovery_status = manager.backend.get_recovery_status()
            if recovery_status:
                console.print(
                    "[yellow]Cache recovery completed:[/yellow] "
                    f"recovered={recovery_status.recovered_entries} entries "
                    f"({recovery_status.recovered_bytes // (1024 * 1024)}MB), "
                    f"lost={recovery_status.lost_entries} entries"
                )
                if recovery_status.errors:
                    for err in recovery_status.errors[:3]:
                        console.print(f"[red]  Error: {err}[/red]")
        else:
            console.print(
                "[green]Virtual chunk cache initialized:[/green] backend=memory (fallback)"
            )
        console.print("[green]Raw chunk cache: OS page cache[/green]")
    else:
        console.print(
            f"[yellow]Warning: Unknown cache backend '{cache_config.backend}', using memory[/yellow]"
        )
        CacheManager.initialize(CacheConfig())

    # Resolve and separate sources
    sources = resolve_all_sources(server_config)
    monitored_sources = [s for s in server_config.sources if s.monitor]

    # Get monitored directory paths to filter out sources discovered from them
    # Sources under monitored dirs will be discovered via rescan, not static registration
    monitored_dirs = {
        ms.local_path
        for ms in monitored_sources
        if not ms.is_remote and ms.local_path
    }

    # Static sources: those NOT under monitored directories (or remote sources)
    # Remote sources are always static (no filesystem monitoring)
    static_sources = [
        s for s in sources
        if s.is_remote or (s.local_path and not any(
            s.local_path.is_relative_to(md) for md in monitored_dirs
        ))
    ]

    if not static_sources and not monitored_sources:
        console.print("[yellow]Warning: No data sources configured[/yellow]")
        raise typer.Exit(1)

    console.print(
        f"[green]Loading {len(static_sources)} static data source(s)...[/green]"
    )
    if monitored_sources:
        console.print(
            f"[green]Monitoring {len(monitored_sources)} directory(s) for live updates[/green]"
        )

    console.print(
        "[green]Compute backend policy:[/green] "
        f"backend={compute_backend or server_config.compute_backend}, "
        f"gpu_min_input_mb={gpu_min_input_mb if gpu_min_input_mb is not None else server_config.gpu_min_input_mb}, "
        f"gpu_min_linear_input_mb={gpu_min_linear_input_mb if gpu_min_linear_input_mb is not None else server_config.gpu_min_linear_input_mb}, "
        f"gpu_memory_safety_factor={gpu_memory_safety_factor or server_config.gpu_memory_safety_factor}, "
        f"gpu_min_merged_chunks={gpu_min_merged_chunks or server_config.gpu_min_merged_chunks}"
    )

    # Create registry for adapters
    registry = get_default_registry()

    # Create metadata database if enabled
    metadata_db: Optional[MetadataDatabase] = None
    if server_config.metadata_db.enabled:
        metadata_db = MetadataDatabase(
            enabled=True,
            max_query_results=server_config.metadata_db.max_query_results,
            query_timeout_ms=server_config.metadata_db.query_timeout_ms,
        )
        console.print(
            "[green]Metadata database initialized:[/green] "
            f"max_query_results={server_config.metadata_db.max_query_results}, "
            f"max_list_flights_results={server_config.metadata_db.max_list_flights_results}, "
            f"query_timeout_ms={server_config.metadata_db.query_timeout_ms}"
        )

    # Create and start server with gRPC message size tuned for 64MB chunks
    location = f"grpc://{host}:{port}"
    # 80MB max message size (slightly above 64MB chunk threshold)
    server = TensorFlightServer(
        location,
        token=token,
        writable=effective_writable,
        write_dir=write_dir,
        metadata_db=metadata_db,
        max_list_flights_results=server_config.metadata_db.max_list_flights_results,
        grpc_max_message_size=80 * 1024 * 1024,
    )

    # Set up watcher for monitored sources (None for static-only configs)
    watcher = None
    source_manager = None
    monitored_dirs = set()
    if monitored_sources:
        try:
            monitored_dirs = {
                ms.local_path
                for ms in monitored_sources
                if not ms.is_remote and ms.local_path
            }
            if server_config.monitor_mode != "off":
                watcher = get_watcher(
                    watcher_type=server_config.monitor_mode,
                    directories=monitored_dirs,
                    poll_interval=server_config.rescan_interval,
                    debounce_window=1.5,
                )
        except Exception as e:
            console.print(f"[red]Failed to create watcher: {e}[/red]")

    # Register all sources (both static and monitored) through unified discovery
    source_manager = create_source_manager(
        server=server,
        registry=registry,
        watcher=watcher,
        monitored_sources=monitored_sources,
        static_sources=static_sources,
        credentials_config=server_config.credentials,
        stability_window=server_config.stability_window,
        probe_open_files=server_config.probe_open_files,
        full_rescan_interval=server_config.full_rescan_interval,
        stable_rescans_required=server_config.stable_rescans_required,
        aggressive_dir_pruning=server_config.aggressive_dir_pruning,
    )

    if source_manager is None:
        console.print("[red]No sources loaded successfully[/red]")
        raise typer.Exit(1)

    # Initial sync of metadata database (batch insert all discovered sources)
    if metadata_db is not None:
        metadata_db.initial_sync(server._sources)

    # Background precache worker: warm the file cache for sources added live.
    # Wire the commit hook BEFORE source_manager.start() so runtime additions
    # are captured; the initial scan was committed before start() and is
    # excluded. The worker itself no-ops on a memory backend.
    precache_worker = None
    if server_config.precache.enabled:
        precache_worker = PrecacheWorker(server, server_config.precache)
        source_manager._on_source_committed = precache_worker.enqueue

    if watcher and source_manager:
        try:
            watcher.start(monitored_dirs)
            source_manager.start()
            console.print(f"[green]Started monitoring: {list(monitored_dirs)}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start monitoring: {e}[/red]")
            watcher.stop()
            watcher = None

    if precache_worker is not None:
        precache_worker.start()

    # Initial scan/registration is complete: flip the health action from
    # STARTING to SERVING so clients waiting through startup can proceed.
    server.mark_ready()

    console.print(f"[green]Flight server ready at {location}[/green]")

    return server, source_manager, watcher, precache_worker


def _create_source_adapter(source: SourceConfig, registry=None):
    """Create a backend adapter from a source config.

    Uses the registry's get_adapter_for_type to find the adapter class,
    then calls create_from_config to instantiate it.

    Args:
        source: SourceConfig with type, url, source_id, dim_labels
        registry: Optional adapter registry (uses default if None)

    Returns:
        BackendAdapter instance

    Raises:
        ValueError: If source type is not registered
    """
    if registry is None:
        registry = get_default_registry()

    adapter_cls = registry.get_adapter_for_type(source.type)
    if adapter_cls is None:
        raise ValueError(f"Unknown source type: {source.type}")

    return adapter_cls.create_from_config(source)


@app.command()
def serve(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        help="Path to TOML config file",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (overrides config and env)",
    ),
    log_scope_biopb: bool = typer.Option(
        True,
        "--log-scope-biopb/--log-scope-all",
        help="Scope logging to biopb_tensor_server only (default) or affect all packages",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        "-h",
        help="Server host (overrides config)",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="Server port (overrides config)",
    ),
    compute_backend: Optional[str] = typer.Option(
        None,
        "--compute-backend",
        help="Compute backend policy: auto, cpu, or gpu",
    ),
    gpu_min_input_mb: Optional[float] = typer.Option(
        None,
        "--gpu-min-input-mb",
        help="Minimum input size in MB before GPU is considered for area-like methods",
    ),
    gpu_min_linear_input_mb: Optional[float] = typer.Option(
        None,
        "--gpu-min-linear-input-mb",
        help="Minimum input size in MB before GPU is considered for linear interpolation",
    ),
    gpu_memory_safety_factor: Optional[int] = typer.Option(
        None,
        "--gpu-memory-safety-factor",
        help="Required free-GPU-memory multiplier over estimated working set",
    ),
    gpu_min_merged_chunks: Optional[int] = typer.Option(
        None,
        "--gpu-min-merged-chunks",
        help="Minimum merged source chunk count before GPU is preferred",
    ),
    writable: bool = typer.Option(
        False,
        "--writable",
        help="Enable write mode for source creation and data upload",
    ),
):
    """Start the TensorFlight server.

    Example:
        biopb-tensor-server serve --config biopb-tensor.toml
        biopb-tensor-server serve -c config.toml --port 9000
        biopb-tensor-server serve -c config.toml --log-level DEBUG
    """
    server_config = load_config(config)

    # Setup logging with priority: CLI > env > config > default
    effective_log_level = (
        log_level or get_log_level_from_env() or server_config.log_level
    )
    setup_logging(effective_log_level, scope_to_biopb=log_scope_biopb)

    # Token from env var only (no auto-gen for serve - it's non-interactive)
    token = os.environ.get("BIOPB_TENSOR_TOKEN") or None

    server, source_manager, watcher, precache_worker = _setup_flight_server(
        server_config,
        host=host,
        port=port,
        compute_backend=compute_backend,
        gpu_min_input_mb=gpu_min_input_mb,
        gpu_min_linear_input_mb=gpu_min_linear_input_mb,
        gpu_memory_safety_factor=gpu_memory_safety_factor,
        gpu_min_merged_chunks=gpu_min_merged_chunks,
        writable=writable,
        token=token,
    )

    location = f"grpc://{host or server_config.host}:{port or server_config.port}"
    console.print(f"\n[green]Starting TensorFlight server at {location}[/green]")
    console.print("Press Ctrl+C to stop\n")

    # Treat SIGTERM (e.g. `biopb server stop`) like Ctrl+C so shutdown is clean.
    _install_sigterm_handler()

    try:
        server.serve()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        _graceful_shutdown(source_manager, watcher, server, precache_worker)


@app.command()
def validate(
    config: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to TOML config file",
    ),
):
    """Validate a config file.

    Example:
        biopb-tensor-server validate biopb-tensor.toml
    """
    try:
        server_config = load_config(config)
        sources = resolve_all_sources(server_config)

        console.print("[green]✓ Config valid[/green]")
        console.print(f"  Server: {server_config.host}:{server_config.port}")
        console.print(
            "  Compute: "
            f"backend={server_config.compute_backend}, "
            f"gpu_min_input_mb={server_config.gpu_min_input_mb}, "
            f"gpu_min_linear_input_mb={server_config.gpu_min_linear_input_mb}, "
            f"gpu_memory_safety_factor={server_config.gpu_memory_safety_factor}, "
            f"gpu_min_merged_chunks={server_config.gpu_min_merged_chunks}"
        )
        console.print(f"  Cache: backend={server_config.cache.backend}, ")
        if server_config.cache.backend == "memory":
            console.print(
                f"    max_entries={server_config.cache.memory_max_entries}, "
                f"max_bytes={server_config.cache.memory_max_bytes // (1024 * 1024)}MB"
            )
        elif server_config.cache.backend == "file":
            console.print(
                f"    cache_dir={server_config.cache.file_cache_dir}, "
                f"max_segment_mb={server_config.cache.file_max_segment_bytes // (1024 * 1024)}, "
                f"max_total_gb={server_config.cache.file_max_total_bytes // (1024 * 1024 * 1024)}"
            )
        console.print(f"  Sources: {len(sources)} data source(s)")

        for source in sources:
            console.print(f"    - {source.source_id} ({source.type}: {source.url})")

    except Exception as e:
        console.print(f"[red]✗ Config invalid: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tensors(
    config: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to TOML config file",
    ),
):
    """List data sources and tensors defined in a config file.

    Example:
        biopb-tensor-server list biopb-tensor.toml
    """
    try:
        server_config = load_config(config)
        sources = resolve_all_sources(server_config)

        table = Table(title="Data Sources and Tensors")
        table.add_column("Source ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Tensor ID", style="magenta")
        table.add_column("Path")
        table.add_column("Shape")

        for source in sources:
            try:
                adapter = _create_source_adapter(source)
                tensor_descs = adapter.list_tensor_descriptors()
                if len(tensor_descs) == 1:
                    desc = tensor_descs[0]
                    table.add_row(
                        source.source_id,
                        source.type,
                        desc.array_id,
                        str(source.url),
                        str(list(desc.shape)),
                    )
                else:
                    for desc in tensor_descs:
                        table.add_row(
                            source.source_id,
                            source.type,
                            desc.array_id,
                            str(source.url),
                            str(list(desc.shape)),
                        )
            except Exception as e:
                table.add_row(
                    source.source_id,
                    source.type,
                    "[red]Error[/red]",
                    str(source.url),
                    str(e),
                )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    try:
        from biopb_tensor_server import __version__ as tensor_version
    except Exception:
        tensor_version = "unknown"

    try:
        from biopb import __version__ as biopb_version
    except Exception:
        biopb_version = "unknown"

    console.print(f"biopb-tensor-server: {tensor_version}")
    console.print(f"biopb: {biopb_version}")


@app.command()
def launch(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        help="Path to TOML config file",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (overrides config and env)",
    ),
    log_scope_biopb: bool = typer.Option(
        True,
        "--log-scope-biopb/--log-scope-all",
        help="Scope logging to biopb_tensor_server only (default) or affect all packages",
    ),
    web_port: int = typer.Option(
        8816,
        "--web-port",
        help="HTTP sidecar port",
    ),
    web_host: str = typer.Option(
        "127.0.0.1",
        "--web-host",
        help="HTTP sidecar bind address",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Website access token (generated if blank)",
        hide_input=True,
    ),
    dev_mode: bool = typer.Option(
        False,
        "--dev",
        help="Enable dev mode (skips token check, localhost only)",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open browser to the web app after startup",
    ),
    web_url: str = typer.Option(
        "http://localhost:5173",
        "--web-url",
        help="Base URL of the running web app (used for --open and CORS)",
    ),
    cors_origins: Optional[List[str]] = typer.Option(
        None,
        "--cors",
        help="Extra CORS origin to allow (repeatable). Defaults to --web-url variants.",
    ),
    static_dir: Optional[Path] = typer.Option(
        None,
        "--static-dir",
        help="Directory containing static webapp files. If empty, serves API only.",
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Path to rotating log file (e.g. /var/log/biopb.log). Rotates at 10MB by default.",
    ),
):
    """Launch the full BioPB Tensor stack (Flight server + HTTP sidecar).

    This command starts:
      1. The Arrow Flight server (data access)
      2. The FastAPI HTTP sidecar (browser-friendly API)

    Run the web app separately with ``pnpm --filter=web dev``,
    or point --web-url to a production deployment.

    Example:
        biopb-tensor-server launch --config biopb-tensor.toml
        biopb-tensor-server launch -c config.toml --web-port 9000 --dev
        biopb-tensor-server launch -c config.toml --log-level DEBUG
    """
    import re as _re

    # --- Load server config and setup logging ---
    server_config = load_config(config)

    # Setup logging with priority: CLI > env > config > default
    effective_log_level = (
        log_level or get_log_level_from_env() or server_config.log_level
    )
    setup_logging(effective_log_level, scope_to_biopb=log_scope_biopb, log_file=log_file)

    # --- Determine dev mode ---
    env_dev = os.environ.get("BIOPB_WEB_DEV_BYPASS", "").lower() in ("1", "true", "yes")
    effective_dev_mode = dev_mode or env_dev

    # Enforce: bypass only on localhost
    if effective_dev_mode and web_host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            "[bold red]SECURITY WARNING:[/bold red] "
            "BIOPB_WEB_DEV_BYPASS ignored because host is non-local; "
            "website token enforcement remains enabled."
        )
        effective_dev_mode = False

    # --- Token management ---
    def _valid_token(t: str) -> bool:
        t = t.strip()
        return bool(t) and 16 <= len(t) <= 128 and _re.fullmatch(r"[A-Za-z0-9_\-]+", t)

    if effective_dev_mode:
        effective_token = None
        console.print(
            "[yellow]DEV MODE: Website token bypass is active (localhost only).[/yellow]"
        )
    else:
        env_token = os.environ.get("BIOPB_TENSOR_TOKEN", "")
        if token and _valid_token(token):
            effective_token = token.strip()
        elif env_token and _valid_token(env_token):
            effective_token = env_token.strip()
        else:
            # Prompt up to 3 times, then auto-generate
            effective_token = None
            for attempt in range(3):
                try:
                    entered = typer.prompt(
                        "Enter website access token (leave blank to auto-generate)",
                        default="",
                        hide_input=True,
                    )
                except Exception:
                    break
                entered = entered.strip()
                if not entered:
                    break
                if _valid_token(entered):
                    effective_token = entered
                    break
                console.print(
                    f"[red]Invalid token (attempt {attempt + 1}/3): "
                    "must be 16-128 URL-safe characters [A-Za-z0-9_-][/red]"
                )
            if effective_token is None:
                effective_token = secrets.token_urlsafe(32)
                console.print("[yellow]Auto-generated secure access token.[/yellow]")

        console.print(
            f"\n[bold green]Access URL (shown once — do not share):[/bold green]\n"
            f"  {web_url}/?token={effective_token}\n"
        )

    # --- Start Flight server ---
    flight_server, source_manager, watcher, precache_worker = _setup_flight_server(
        server_config, token=effective_token
    )

    flight_location = f"grpc://{server_config.host}:{server_config.port}"
    flight_thread = threading.Thread(target=flight_server.serve, daemon=True)
    flight_thread.start()

    if open_browser:
        url = web_url
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    # --- Build effective CORS origins ---
    if cors_origins:
        effective_cors = list(cors_origins)
    else:
        # Derive CORS origins, expanding loopback aliases for any local hostname
        from urllib.parse import urlparse as _urlparse

        _loopback_aliases: dict = {
            "localhost": ["127.0.0.1", "[::1]"],
            "127.0.0.1": ["localhost", "[::1]"],
            "::1": ["localhost", "127.0.0.1"],
            "[::1]": ["localhost", "127.0.0.1"],
        }

        def _expand_origin(url: str) -> list:
            parsed = _urlparse(url)
            p = parsed.port
            port_suffix = f":{p}" if p else ""
            scheme = parsed.scheme
            hostname = parsed.hostname or "localhost"
            origins = [f"{scheme}://{hostname}{port_suffix}"]
            for alias in _loopback_aliases.get(hostname, []):
                origins.append(f"{scheme}://{alias}{port_suffix}")
            return origins

        effective_cors = _expand_origin(web_url)

        # When serving the webapp from this same server (--static-dir), also
        # allow all loopback variants of the server's own address so users
        # can reach it via localhost or 127.0.0.1 interchangeably.
        if static_dir:
            for origin in _expand_origin(f"http://{web_host}:{web_port}"):
                if origin not in effective_cors:
                    effective_cors.append(origin)

    # --- Start HTTP sidecar (blocks) ---
    console.print(
        f"[green]Starting HTTP sidecar at http://{web_host}:{web_port}[/green]"
    )
    console.print("Press Ctrl+C to stop\n")
    # uvicorn installs its own SIGINT/SIGTERM handlers and returns normally on
    # shutdown (it does not re-raise), so cleanup must run in `finally` rather
    # than an except block - otherwise the cache lock is never released.
    try:
        run_http_server(
            flight_location=flight_location,
            token=effective_token,
            dev_mode=effective_dev_mode,
            host=web_host,
            port=web_port,
            cors_origins=effective_cors,
            static_dir=str(static_dir) if static_dir else None,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        _graceful_shutdown(source_manager, watcher, flight_server, precache_worker)

    try:
        from biopb_tensor_server import __version__

        console.print(f"TensorFlight server (using biopb-tensor-server {__version__})")
    except ImportError:
        console.print("TensorFlight server")


if __name__ == "__main__":
    app()
