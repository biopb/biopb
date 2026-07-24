"""CLI for TensorFlight server.

Commands:
    serve      Start the Flight server
    launch     Start the Flight server and HTTP sidecar for web app
    validate   Validate a config file
    list       List tensors in a config file
"""

import logging
import os
import secrets
import signal
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from biopb import _web_auth
from biopb._lifecycle import deathwatch as _deathwatch
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.table import Table

from biopb_tensor_server.adapters import AdapterRegistry, get_default_registry
from biopb_tensor_server.adapters._handle_reaper import set_handle_reaper_ttl
from biopb_tensor_server.adapters.bioio import set_claim_generic_images
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.file_backend import ArrowFileBackend
from biopb_tensor_server.core.config import (
    CacheConfig,
    ServerConfig,
    SourceConfig,
    _read_config_file,
    load_config,
    resolve_all_sources,
    validate_config_dict,
)
from biopb_tensor_server.core.fs_detect import unsafe_cache_dir_reason
from biopb_tensor_server.core.logging_config import (
    get_log_level_from_env,
    setup_logging,
)
from biopb_tensor_server.core.metadata_db import MetadataDatabase
from biopb_tensor_server.serving.http_server import run as run_http_server
from biopb_tensor_server.serving.precache import PrecacheWorker
from biopb_tensor_server.serving.server import TensorFlightServer
from biopb_tensor_server.sources.source_manager import create_source_manager
from biopb_tensor_server.sources.watcher import get_watcher

app = typer.Typer(
    name="biopb-tensor-server",
    help="BioPB Tensor: Arrow Flight server for multi-dimensional arrays",
)
console = Console()
logger = logging.getLogger(__name__)

diag_app = typer.Typer(help="Diagnostic commands for a running TensorFlight server")
app.add_typer(diag_app, name="diagnose")

# The bind address is the mode: a loopback bind is reachable same-machine only
# (local mode); anything else is network-reachable. The wildcard binds
# (``0.0.0.0`` / ``::`` / ``""``) and any real IP/hostname are public, so they are
# *not* in this set and are treated as public — fail-closed.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _host_is_public(host: str) -> bool:
    """True if ``host`` is a network-reachable bind address (not loopback)."""
    return host not in _LOOPBACK_HOSTS


def _resolve_flight_token(
    server_host: str,
    token: Optional[str],
    env_token: str,
) -> Optional[str]:
    """Resolve the token the Flight (gRPC) server enforces, fail-closed on a
    public bind.

    The flight bind (config ``server.host``, or a ``--host`` override) is the mode
    switch: a loopback bind is **local mode** (tokenless, same-machine only); any
    public bind is **remote mode** and MUST carry a token, so a public bind with
    none supplied auto-generates one rather than serving the data API open.

    Shared by ``serve`` and ``launch``; ``launch`` layers its sidecar fail-closed
    check on top of the returned token (see ``_resolve_launch_token``).

    Returns the effective token (``None`` = local mode).
    """
    if token and _web_auth.valid_token(token):
        return token.strip()
    if env_token and _web_auth.valid_token(env_token):
        return env_token.strip()
    if _host_is_public(server_host):
        generated = secrets.token_urlsafe(32)
        console.print(
            "[yellow]Auto-generated secure access token "
            f"(server.host={server_host} is a public bind).[/yellow]"
        )
        return generated
    # Loopback flight bind, no token supplied: local mode.
    return None


def _resolve_launch_token(
    server_host: str,
    web_host: str,
    token: Optional[str],
    env_token: str,
) -> Optional[str]:
    """Decide the token ``launch`` enforces, fail-closed on every public listener.

    Builds on :func:`_resolve_flight_token` (the flight bind is the mode switch),
    then adds the sidecar check: the HTTP sidecar has its own, independent bind
    (``--web-host``). Because it re-exposes the whole data API, a **public sidecar
    with no enforced token** is exactly the "public + unauthenticated" combination
    the model makes unrepresentable — so it is refused rather than served open.
    (This is the ``--web-host 0.0.0.0`` + loopback ``server.host`` footgun: the
    token would otherwise resolve to ``None`` and the data API would bind public
    and open.)

    Returns the effective token (``None`` = local mode). Raises ``typer.Exit(1)``
    if the sidecar would bind public without a token.
    """
    effective_token = _resolve_flight_token(server_host, token, env_token)

    if effective_token is None and _host_is_public(web_host):
        console.print(
            "[red]Refusing to bind the HTTP sidecar to a public address "
            f"(--web-host {web_host}) with no access token.[/red]\n"
            "The sidecar re-exposes the data API, so this would serve it "
            "unauthenticated to the network. Either bind it to loopback "
            "(--web-host 127.0.0.1), or make the flight server public "
            "(server.host) so a token is enforced across both listeners."
        )
        raise typer.Exit(1)

    return effective_token


def _install_sigterm_handler() -> None:
    """Make SIGTERM behave like Ctrl+C (KeyboardInterrupt).

    The control supervisor (POSIX), `docker/singularity stop`, and SLURM all
    terminate the server with SIGTERM, which Python ignores (default disposition) for a
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


# How long to wait for the Flight drain before proceeding without it. The
# control supervisor force-kills after a ~10s graceful window; keep this well
# under that so the bounded steps that follow (source-manager join) still fit.
_FLIGHT_DRAIN_TIMEOUT_S = 3.0


def _graceful_shutdown(
    source_manager, watcher, flight_server, precache_worker=None
) -> None:
    """Best-effort orderly shutdown -- release the cache lock first, never hang.

    Step ORDER is load-bearing for clean restarts (biopb/biopb#300). ``restart``
    force-kills the daemon after a bounded graceful window (``--timeout``, 10s by
    default), so the file-cache process lock -- whose release is a local, instant,
    upstream-independent operation -- must be dropped FIRST, before any step that
    can block on an unresponsive upstream. Otherwise a mid-teardown SIGKILL leaves
    a stale lock and the next boot pays a crash-recovery scan. On a caching proxy
    the two slow steps are both upstream-coupled -- the Flight drain (in-flight
    ``do_get`` streams gated on a possibly-dead upstream) and the source-manager
    join (a blocking re-list RPC to that upstream) -- so they are sequenced
    *after* the lock release and then individually bounded:

    1. Stop the precache worker -- no new warm writes.
    2. Release the process lock + clear the WAL IMMEDIATELY. Cheap and
       upstream-independent; leaves segment writers/mmaps OPEN (closing them here
       would race the in-flight ``do_get`` reads the drain has not finished).
       Clearing the WAL early is safe -- index rebuild tolerates a torn tail.
       After this, even a SIGKILL during the steps below finds the lock released.
    3. Drain the Flight server, BOUNDED. ``FlightServerBase.shutdown()`` takes no
       timeout and can block unbounded on an upstream-gated stream, so run it in a
       daemon thread and join with a short bound; on timeout, proceed (the process
       is exiting; the OS reclaims the sockets). Never call ``flight_server.wait()``.
    4. Full cache close ONLY on a clean drain -- closes writers/mmaps for proper
       finalization (matters on Windows). Skipped if the drain timed out: a stuck
       in-flight ``do_get`` may still touch an mmap, so closing it could segfault,
       and the essential work (lock + WAL) already happened in step 2. ``close()``'s
       own lock-release is then a harmless no-op (already released).
    5. Stop the source manager (short join) and watcher last -- neither touches the
       chunk cache and the lock is already gone, so a long join has no value; a
       short bound keeps a blocked upstream re-list RPC from eating the budget.

    Each step is isolated so a failure in one still lets the others run.
    """
    drain_ok = {"value": False}

    def _release_lock() -> None:
        # Cheap, upstream-independent: clear the WAL + drop the process lock while
        # leaving writers/mmaps open (no-op for the memory backend).
        manager = CacheManager.get_instance()
        if manager is not None:
            manager.release_process_lock()

    def _bounded_drain() -> None:
        # FlightServerBase.shutdown() blocks until in-flight RPCs finish and takes
        # no timeout, so on a caching proxy a do_get gated on a dead upstream can
        # block forever. Bound it: run in a daemon thread, join briefly, and
        # proceed on timeout. Do NOT call flight_server.wait().
        if flight_server is None:
            drain_ok["value"] = True
            return
        drain_thread = threading.Thread(
            target=flight_server.shutdown,
            name="flight-drain",
            daemon=True,
        )
        drain_thread.start()
        drain_thread.join(_FLIGHT_DRAIN_TIMEOUT_S)
        if drain_thread.is_alive():
            console.print(
                "[yellow]Flight drain did not finish within "
                f"{_FLIGHT_DRAIN_TIMEOUT_S:g}s (upstream unresponsive?); "
                "proceeding -- the process is exiting and the cache lock is "
                "already released.[/yellow]"
            )
        else:
            drain_ok["value"] = True

    def _close_cache_if_drained() -> None:
        # Full close (writers/mmaps) only after a clean drain. If the drain timed
        # out a stuck do_get could still touch an mmap, so closing mid-flight
        # risks a segfault; the lock + WAL were already handled in step 2.
        if not drain_ok["value"]:
            return
        manager = CacheManager.get_instance()
        if manager is not None:
            manager.close()

    for label, action in (
        ("precache worker", lambda: precache_worker and precache_worker.stop()),
        ("cache lock", _release_lock),
        ("flight server", _bounded_drain),
        ("cache", _close_cache_if_drained),
        # Short join: the lock is already released and the thread is a daemon, so a
        # long wait has no value and only risks burning the SIGKILL budget on a
        # blocked upstream re-list.
        (
            "source manager",
            lambda: source_manager and source_manager.stop(join_timeout=1),
        ),
        ("watcher", lambda: watcher and watcher.stop()),
    ):
        try:
            action()
        except Exception as e:  # noqa: BLE001 - shutdown must not raise
            console.print(f"[yellow]Error stopping {label}: {e}[/yellow]")


def _is_bare_host_upstream(source: SourceConfig) -> bool:
    """True for a bare-host ``grpc://host:port`` tensor-server upstream (no ``/<id>``).

    Only the bare-host "mirror everything" form has an upstream catalog to
    re-list; a single-source ``grpc://host:port/<id>`` names exactly one source
    and is registered directly. Mirrors the ``monitored_upstreams`` filter in
    ``source_manager.create_source_manager``.
    """
    if not source.is_remote:
        return False
    if not source.url.lower().startswith(("grpc://", "grpc+tls://", "grpcs://")):
        return False
    from biopb_tensor_server.adapters.remote_tensor import _split_grpc_url

    return _split_grpc_url(source.url)[1] is None


def _resolve_serve_sources(
    server_config: ServerConfig,
    registry: Optional[AdapterRegistry] = None,
) -> Tuple[List[SourceConfig], List[SourceConfig]]:
    """Partition configured sources for the serve path.

    Returns ``(static_sources, monitored_sources)``.

    Local ``monitor = true`` directories are NOT expanded here: they are
    (re)discovered by the bootstrap rescan, so expanding them at startup only
    walks the tree an extra time before the server binds and crashes on a
    not-yet-mounted directory (biopb/biopb#54). Remote monitor entries and
    non-monitored entries are expanded as before; a single-file ``monitor=true``
    entry is registered statically (with a warning) instead of being silently
    dropped, and a missing/broken static source is warned-and-skipped rather
    than aborting startup.
    """
    to_expand: List[SourceConfig] = []  # entries run through discover_sources
    monitored_sources: List[SourceConfig] = []

    for s in server_config.sources:
        # The ``monitor`` flag alone decides live monitoring -- identically for
        # cloud and non-cloud roots. ``cloud`` only controls *gating* (admit
        # dehydrated placeholders as unresolved sources); it no longer forces a
        # root onto the monitored pipeline. So a cloud root with monitor=false is
        # scanned once at startup via the static-expand path (cloud-gated there
        # too), exactly like any other monitor=false directory.
        if s.monitor and not s.is_remote:
            local_path = s.local_path
            # local_path cannot be None here -- both is_remote and local_path
            # derive from _is_remote_url(url), so `not is_remote` guarantees a
            # resolved path. Guard anyway: if that invariant ever broke, a None
            # path must NOT be registered as a monitored directory. Route it to
            # the expansion path, which validates it (and skips under tolerant).
            if local_path is None:
                to_expand.append(s)
                continue
            if local_path.is_file():
                # Files cannot be live-monitored: register as a static source
                # instead of silently dropping it.
                logger.warning(
                    "Cannot live-monitor a single file; registering it as a "
                    "static source: %s",
                    s.url,
                )
                to_expand.append(s)
                continue
            if not local_path.exists():
                # Not-yet-mounted dir: skip the (crashing) expansion. The watcher
                # and rescan pick it up when it appears (the runtime self-heals).
                logger.warning(
                    "Monitored path does not exist yet; will start monitoring "
                    "when it appears: %s",
                    s.url,
                )
            # A local `alias` sets a catalog tree-root, but that override is
            # display-only and non-durable: a monitored directory is re-discovered
            # under its native path on every rescan and re-merges into the shared
            # tree, so the alias root would flicker away on the first rescan. Ignore
            # it loudly rather than pretend it holds. (Honored fine for a static /
            # monitor=false root, and for a monitor=true single *file* -- which is
            # registered static, above -- neither of which is rescanned.)
            if s.alias:
                logger.warning(
                    "Ignoring 'alias' tree-root %r on monitored directory %s: a "
                    "monitored root re-merges into the shared path tree on rescan. "
                    "Drop 'monitor' to keep the alias as its own catalog root.",
                    s.alias,
                    s.url,
                )
            monitored_sources.append(s)  # directory (or not-yet-mounted dir)
            continue

        # A bare-host tensor-server upstream ("mirror everything") is ALWAYS routed
        # to the background seeded re-list, regardless of `monitor`. Inline
        # expansion registers every mirrored source through a blocking per-source
        # upstream get_descriptor RPC *before* mark_ready(), which both keeps the
        # server STARTING until it finishes (observed ~1h / 900s+ stuck registering
        # hundreds of hpc__* proxies -- each descriptor an expensive OME-TIFF open
        # on the upstream) and bypasses the bulk-seed fast path. The re-list instead
        # seeds the entire catalog in ONE upstream query_sources (no per-source RPC,
        # biopb/biopb#266) and runs in the background, so the server reaches SERVING
        # immediately and the mirror fills progressively -- exactly like a monitored
        # local directory. `monitor=false` on a bare-host upstream is not "static":
        # it just means the adaptive cadence reconciles once at the boot tick and
        # then backs off toward full_rescan_interval, rather than never mirroring
        # the upstream at all (biopb/biopb#178).
        if _is_bare_host_upstream(s):
            monitored_sources.append(s)
            continue
        # Remote monitor entries are also handed to create_source_manager.
        if s.monitor:
            monitored_sources.append(s)
        to_expand.append(s)

    # Expand only the non-monitored-dir entries. tolerant=True so one missing or
    # broken static source is warned-and-skipped rather than killing the server.
    sources = resolve_all_sources(
        server_config, registry, sources=to_expand, tolerant=True
    )

    # Static sources: those NOT under a monitored directory (or remote sources).
    # Still needed when a non-monitored entry's expansion lands under a
    # monitored root. Remote sources are always static (no filesystem monitoring).
    monitored_dirs = {
        ms.local_path for ms in monitored_sources if not ms.is_remote and ms.local_path
    }
    static_sources = [
        s
        for s in sources
        if s.is_remote
        or (
            s.local_path
            and not any(s.local_path.is_relative_to(md) for md in monitored_dirs)
        )
    ]
    return static_sources, monitored_sources


def _grpc_location(host: str, port: int) -> str:
    """Build a ``grpc://`` URL, bracketing an IPv6 literal in the authority.

    An IPv6 address contains ``:`` and must be wrapped in brackets to be a valid
    URL authority, e.g. ``grpc://[::1]:8815``; IPv4 addresses and hostnames pass
    through unchanged. Used for both the server bind location and the sidecar's
    connect target so neither emits a malformed URL for an IPv6 host.
    """
    authority = f"[{host}]" if isinstance(host, str) and ":" in host else host
    return f"grpc://{authority}:{port}"


def _setup_flight_server(
    server_config: ServerConfig,
    host: Optional[str] = None,
    port: Optional[int] = None,
    writable: Optional[bool] = None,
    token: Optional[str] = None,
) -> Tuple[
    TensorFlightServer, Optional[object], Optional[object], Optional[PrecacheWorker]
]:
    """Set up the Flight server with cache, sources, and monitoring.

    Args:
        server_config: Loaded server configuration
        host: Override host
        port: Override port
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

    # Apply the discovery-claim policy for generic raster/video (biopb/biopb#40).
    # Off by default so recursive scans don't register screenshots/icons/movies.
    set_claim_generic_images(server_config.claim_generic_images)

    # Apply the idle-handle reaper TTL to the opt-in adapters (OME-TIFF, NDTiff)
    # before any source registers, so it fully takes effect (biopb/biopb#71).
    set_handle_reaper_ttl(server_config.handle_reaper_ttl)

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

        def _memory_fallback() -> CacheConfig:
            return CacheConfig(
                backend="memory",
                memory_max_entries=cache_config.memory_max_entries,
                memory_max_bytes=cache_config.memory_max_bytes,
            )

        # The file cache mmaps its segments and assumes local-POSIX semantics
        # (unlinked-but-mapped inodes stay alive, mapped pages never vanish). A
        # network mount (NFS/CIFS) can SIGBUS/ESTALE a mapping to an evicted
        # segment, and a cloud Files-On-Demand folder recalls a dehydrated
        # segment on mmap read -- so classify the cache dir once and fall back to
        # memory rather than serve unsafe reads (biopb/biopb#571 follow-up). This
        # also disables the localhost client fast path for free: a memory backend
        # never locates a chunk, so clients use do_get.
        unsafe = unsafe_cache_dir_reason(cache_config.file_cache_dir)
        if unsafe:
            console.print(
                f"[yellow]File cache dir {cache_config.file_cache_dir} is on "
                f"{unsafe}; the mmap-based file cache is unsafe there, falling "
                f"back to in-memory cache.[/yellow]"
            )
            manager = CacheManager.initialize(_memory_fallback())
        else:
            try:
                manager = CacheManager.initialize(cache_config)
            except OSError as e:
                # Cache dir not writable (e.g. read-only HPC scratch). Fall back
                # to the in-memory backend so the server still starts; the
                # localhost cache-file fast path (issue #9) is simply unavailable.
                console.print(
                    f"[yellow]File cache unavailable at {cache_config.file_cache_dir} "
                    f"({e}); falling back to in-memory cache.[/yellow]"
                )
                manager = CacheManager.initialize(_memory_fallback())
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
                # (No per-segment error list here: recovery no longer scans
                # segment bodies -- biopb/biopb#300 -- so it surfaces no read
                # errors. Corrupt segments are detected, logged, and dropped by
                # _rebuild_index_from_segments' own logger.error instead.)
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

    # Resolve and separate sources (see _resolve_serve_sources)
    registry = get_default_registry()
    static_sources, monitored_sources = _resolve_serve_sources(server_config, registry)

    if not static_sources and not monitored_sources:
        # An empty catalog is a valid state -- start and serve it (health SERVING,
        # empty list_flights) rather than exiting. Sources can arrive after
        # startup: runtime add_source (napari drag-drop), DoPut uploads, or a
        # monitored dir that is currently empty but fills later. Refusing to boot
        # would also make the control-plane data-plane supervisor read a healthy
        # empty server as a crash -> backoff/restart loop (biopb/biopb#515).
        console.print(
            "[yellow]No data sources configured; serving an empty catalog "
            "(sources can be added at runtime).[/yellow]"
        )

    console.print(
        f"[green]Loading {len(static_sources)} static data source(s)...[/green]"
    )
    if monitored_sources:
        console.print(
            f"[green]Monitoring {len(monitored_sources)} directory(s) for live updates[/green]"
        )

    # The metadata database is mandatory (biopb/biopb#225): always constructed --
    # it is the canonical source-browsing surface (`client.query_sources`).
    metadata_db = MetadataDatabase(
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
    location = _grpc_location(host, port)
    # 80MB max message size (slightly above 64MB chunk threshold)
    server = TensorFlightServer(
        location,
        token=token,
        writable=effective_writable,
        write_dir=write_dir,
        metadata_db=metadata_db,
        max_list_flights_results=server_config.metadata_db.max_list_flights_results,
        grpc_max_message_size=80 * 1024 * 1024,
        pyramid_config=server_config.pyramid,
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
        metadata_db=metadata_db,
        credentials_config=server_config.credentials,
        stability_window=server_config.stability_window,
        probe_open_files=server_config.probe_open_files,
        full_rescan_interval=server_config.full_rescan_interval,
        stable_rescans_required=server_config.stable_rescans_required,
        aggressive_dir_pruning=server_config.aggressive_dir_pruning,
        # An empty (or all-invalid) source set is a valid runtime state: build an
        # empty manager and serve an empty catalog rather than refusing to boot
        # (biopb/biopb#515).
        allow_empty=True,
    )

    # With allow_empty=True an empty/all-invalid source set yields an empty manager
    # (served as an empty catalog), so a None here no longer means "no sources" --
    # it can only be a genuine construction failure. Guard it: the startup code
    # below dereferences source_manager unconditionally (unlike _graceful_shutdown,
    # which tolerates None), so fail cleanly rather than with an opaque
    # AttributeError. This exit is inside serve()/launch()'s try, so the finally
    # still releases the cache lock (biopb/biopb#515).
    if source_manager is None:
        console.print("[red]Failed to initialize the source manager[/red]")
        raise typer.Exit(1)

    # Wire the runtime add_source handler (tensor-browser drag-drop): the server
    # holds no SourceManager reference, so inject the entrypoint that routes a
    # dropped path into the same claim -> adapter -> catalog pipeline. Its
    # counterpart removes a dropped (dnd://) branch.
    server.set_add_source_handler(source_manager.add_local_source)
    server.set_remove_source_handler(source_manager.remove_dropped_root)

    # Note: the metadata DB is already populated at this point. Static sources
    # are seeded via SourceManager._commit_add_claim -> _register_source_claim
    # (which syncs each), and monitored sources stream in through the background
    # first scan -- both before this line. No separate initial_sync is needed.

    # Background precache worker: warm the file cache for sources added live.
    # Wire the commit hook BEFORE source_manager.start(). Under progressive
    # discovery the startup set is committed by the *background* scan (after
    # start); the manager gates it out of the prompt enqueue via
    # _initial_scan_done and seeds it into the backlog through the first-scan
    # callback below. The worker no-ops on a memory backend.
    precache_worker = None
    if server_config.precache.enabled:
        precache_worker = PrecacheWorker(
            server, server_config.precache, server_config.pyramid
        )
        source_manager.set_source_committed_hook(precache_worker.enqueue)
        # Residency gate (#174): let the worker re-check, at warm time, that a
        # cloud-root source's files are still resident before reading them, so a
        # backlog/live pass never recalls bytes OneDrive has re-dehydrated since
        # registration.
        precache_worker.should_warm = source_manager.should_warm

    # Seed the precache backlog with the startup catalog the moment the first
    # full scan establishes it (newest first; warmed when the server is idle).
    # Wired before start() so the background scan's completion finds it.
    def _seed_backlog_on_first_scan() -> None:
        if precache_worker is not None and server_config.precache.backlog_enabled:
            precache_worker.seed_backlog(source_manager.iter_local_source_mtimes())

    source_manager.set_initial_scan_complete_hook(_seed_backlog_on_first_scan)

    # Report "a full scan is running" from the first SERVING moment. The
    # background scan sets this itself on entry, but pre-setting here closes the
    # brief window between mark_ready() and the event loop picking up the first
    # rescan, so a client never sees "SERVING, not scanning, never scanned".
    if monitored_dirs:
        server.set_full_scan_in_progress(True)

    background_scan_running = False
    if watcher and source_manager:
        try:
            watcher.start(monitored_dirs)
            source_manager.start()
            background_scan_running = True
            console.print(f"[green]Started monitoring: {list(monitored_dirs)}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start monitoring: {e}[/red]")
            watcher.stop()
            watcher = None

    if precache_worker is not None:
        precache_worker.start()

    # Progressive discovery: reach SERVING immediately. The monitored bootstrap
    # scan runs in the background; the catalog populates live and the health
    # action carries its freshness (full_scan_in_progress /
    # last_full_scan_finished_at). A client needing a complete catalog waits on
    # those fields, not on SERVING.
    server.mark_ready()

    if not background_scan_running:
        # No event loop will drive the bootstrap scan. Two cases:
        #  - monitored dirs but the watcher failed to start: scan synchronously
        #    now so those sources are still registered (the pre-progressive
        #    behavior for watcher-less setups); run_initial_scan also stamps
        #    freshness, flips the startup gate, and seeds the backlog.
        #  - static-only config (no monitored dirs, nothing to scan): advance the
        #    completion protocol directly so it still reports a timestamp and seeds.
        if monitored_dirs:
            source_manager.run_initial_scan()
        else:
            source_manager.complete_initial_scan()

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
        SourceAdapter instance

    Raises:
        ValueError: If source type is not registered
    """
    if registry is None:
        registry = get_default_registry()

    adapter_cls = registry.get_adapter_for_type(source.type)
    if adapter_cls is None:
        raise ValueError(f"Unknown source type: {source.type}")

    return adapter_cls.create_from_config(source)


def _load_config_or_exit(config: Path) -> ServerConfig:
    """Load *config*, turning a bad file into a one-line refusal, not a traceback.

    Config errors are the user's to fix -- a bad knob, an unmigrated TOML, a
    JSON typo -- and this is where the server fails fast on them
    (biopb/biopb#34). A traceback buries that message, and under the control
    plane it lands in ``tensor-server.log`` where it reads as a crash rather
    than as "your config says downscale_factor=0".
    """
    try:
        return load_config(config)
    except (ValueError, FileNotFoundError) as e:
        # Escaped: a validation message names its section as "[pyramid]", which
        # rich would otherwise eat as a style tag and print as nothing.
        console.print(f"[red]✗ Config invalid: {_rich_escape(str(e))}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        help="Path to config file (biopb.json)",
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
    writable: bool = typer.Option(
        False,
        "--writable",
        help="Enable write mode for source creation and data upload",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Access token (required when the flight bind is non-loopback; "
        "auto-generated if blank on a public bind)",
        hide_input=True,
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Path to rotating log file (e.g. /var/log/biopb.log). Rotates at 10MB by default.",
    ),
):
    """Start the TensorFlight server.

    Example:
        biopb-tensor-server serve --config biopb.json
        biopb-tensor-server serve -c config.json --port 9000
        biopb-tensor-server serve -c config.json --log-level DEBUG
    """
    server_config = _load_config_or_exit(config)

    # Setup logging with priority: CLI > env > config > default
    effective_log_level = (
        log_level or get_log_level_from_env() or server_config.log_level
    )
    setup_logging(
        effective_log_level, scope_to_biopb=log_scope_biopb, log_file=log_file
    )

    # The flight bind is the mode switch; --host overrides config, so resolve the
    # token against the effective host. A public bind with no token auto-generates
    # one (fail-closed) rather than serving the data API open.
    effective_host = host or server_config.host
    effective_token = _resolve_flight_token(
        effective_host,
        token,
        os.environ.get("BIOPB_TENSOR_TOKEN", ""),
    )

    if effective_token is not None:
        console.print(
            "\n[bold green]Access token (shown once — do not share):[/bold green]"
        )
        # The gRPC client sends this as an `authorization: Bearer <token>` header
        # (there is no `?token=` URL form for the flight plane), so print the bare
        # token. soft_wrap keeps it on one line in a narrow/non-TTY log; markup
        # False so nothing in it is read as Rich markup.
        console.print(effective_token, soft_wrap=True, markup=False)
        console.print()
    else:
        console.print(
            "[yellow]Local mode: no access token (loopback-only bind).[/yellow]"
        )

    # Pre-bind so the `finally` runs graceful shutdown -- which releases the file
    # cache process lock -- on EVERY exit path, not just a clean serve() return.
    # _setup_flight_server acquires the lock during cache init and can still raise
    # afterwards (e.g. a bad static source); keeping it inside the try means such
    # an early exit no longer orphans the lock as a stale lock (biopb/biopb#515).
    server = source_manager = watcher = precache_worker = None
    try:
        server, source_manager, watcher, precache_worker = _setup_flight_server(
            server_config,
            host=host,
            port=port,
            writable=writable,
            token=effective_token,
        )

        location = _grpc_location(effective_host, port or server_config.port)
        console.print(f"\n[green]Starting TensorFlight server at {location}[/green]")
        console.print("Press Ctrl+C to stop\n")

        # Treat SIGTERM (e.g. the control supervisor's graceful stop) like Ctrl+C
        # so shutdown is clean.
        _install_sigterm_handler()
        # If launched under the control supervisor, self-terminate when it dies
        # uncatchably (no-op when run standalone; see biopb._lifecycle.deathwatch).
        _deathwatch.install()

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
        help="Path to config file (biopb.json)",
    ),
):
    """Validate a config file.

    Example:
        biopb-tensor-server validate biopb.json
    """
    # The strict end of the validation scheme (biopb/biopb#34): loading would
    # clamp a bad knob to its default and carry on -- right for a supervised
    # start, wrong here, where a human is explicitly asking whether the file is
    # good. Report every problem and fail. Kept out of the try below so the exit
    # isn't swallowed by its catch-all and re-reported as an error.
    try:
        problems = validate_config_dict(_read_config_file(config))
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]✗ Config invalid: {_rich_escape(str(e))}[/red]")
        raise typer.Exit(1)
    if problems:
        console.print("[red]✗ Config invalid[/red]")
        for problem in problems:
            where = ".".join(problem["path"]) or "config"
            console.print(f"  {where}: {_rich_escape(problem['message'])}")
        raise typer.Exit(1)

    try:
        server_config = load_config(config)
        sources = resolve_all_sources(server_config)

        console.print("[green]✓ Config valid[/green]")
        console.print(f"  Server: {server_config.host}:{server_config.port}")
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
        # Escaped: a validation message names its section as "[pyramid]", which
        # rich would otherwise eat as a style tag and print as nothing.
        console.print(f"[red]✗ Config invalid: {_rich_escape(str(e))}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tensors(
    config: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to config file (biopb.json)",
    ),
):
    """List data sources and tensors defined in a config file.

    Example:
        biopb-tensor-server list biopb.json
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
        console.print(f"[red]Error: {_rich_escape(str(e))}[/red]")
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


@app.command(name="config-schema")
def config_schema(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the schema to this file instead of stdout.",
    ),
):
    """Print the JSON Schema for the config file.

    The schema is generated from the server's own validation table, so its
    value bounds and enums match what the server enforces at startup. Save it
    and reference it from a config via "$schema" for editor autocomplete, or
    feed it to a JSON Schema validator for pre-flight checks (biopb/biopb#34).

    Example:
        biopb-tensor-server config-schema -o biopb-config.schema.json
    """
    import json

    from biopb_tensor_server.core.config_schema import build_config_schema

    text = json.dumps(build_config_schema(), indent=2) + "\n"
    if output is not None:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]✓ Wrote schema to {output}[/green]")
    else:
        # Raw stdout (not console.print) so the output is clean, pipeable JSON.
        typer.echo(text, nl=False)


@app.command()
def launch(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        help="Path to config file (biopb.json)",
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
        help="Flight server host (overrides config)",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="Flight server port (overrides config)",
    ),
    writable: bool = typer.Option(
        False,
        "--writable",
        help="Enable write mode for source creation and data upload",
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
        help="Access token (required when server.host is non-loopback; "
        "auto-generated if blank on a public bind)",
        hide_input=True,
    ),
    cors_origins: Optional[List[str]] = typer.Option(
        None,
        "--cors",
        help="CORS origin to allow (repeatable). Required to reach the sidecar "
        "from a browser app on another origin; defaults to loopback only.",
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

    The web app is not part of this package; run and serve it separately, and
    pass its origin with ``--cors`` so the browser can reach this sidecar.

    Example:
        biopb-tensor-server launch --config biopb.json
        biopb-tensor-server launch -c config.json --web-port 9000
        biopb-tensor-server launch -c config.json --host 0.0.0.0 --writable
        biopb-tensor-server launch -c config.json --log-level DEBUG
    """
    # --- Load server config and setup logging ---
    server_config = _load_config_or_exit(config)

    # Setup logging with priority: CLI > env > config > default
    effective_log_level = (
        log_level or get_log_level_from_env() or server_config.log_level
    )
    setup_logging(
        effective_log_level, scope_to_biopb=log_scope_biopb, log_file=log_file
    )

    # Treat SIGTERM (the control supervisor's graceful stop, `docker/slurm stop`)
    # like Ctrl+C so the post-uvicorn `finally` below actually runs. uvicorn does
    # handle SIGTERM for its own HTTP shutdown, but when its event loop closes it
    # reverts SIGTERM to the default (terminate) disposition -- so the process is
    # signal-killed (exit 143) before control reaches `_graceful_shutdown`, and
    # the file-cache process lock is left behind as a stale lock on every control
    # stop/restart (biopb/biopb#516; #512's lock-release-first reorder is moot on
    # this path until the finally actually runs). Owning the handler here routes
    # SIGTERM through `except KeyboardInterrupt`/`finally` instead. Harmless no-op
    # on Windows, which uses the sentinel-file path in
    # http_server._install_windows_shutdown_listener.
    _install_sigterm_handler()
    # If launched under the control supervisor, self-terminate when it dies
    # uncatchably so a crashed/killed control never orphans this plane into a
    # port-holding conflict (no-op standalone; see biopb._lifecycle.deathwatch).
    _deathwatch.install()

    # --- Token management ---
    # The flight bind (server.host, or a --host override) is the mode switch; the
    # sidecar's own bind (--web-host) must never be public-and-unauthenticated.
    # _resolve_launch_token decides the enforced token fail-closed (and refuses a
    # public sidecar with no token). There is no separate dev flag.
    effective_host = host or server_config.host
    effective_token = _resolve_launch_token(
        effective_host,
        web_host,
        token,
        os.environ.get("BIOPB_TENSOR_TOKEN", ""),
    )

    if effective_token is not None:
        # The web app lives outside this package, so we can only show the token
        # against the sidecar's own origin; a browser app appends it there (or
        # carries it however that app expects). Normalize a wildcard bind to a
        # dialable loopback host for display.
        _display_host = web_host
        if _display_host in ("0.0.0.0", ""):
            _display_host = "127.0.0.1"
        elif _display_host == "::":
            _display_host = "[::1]"
        console.print(
            "\n[bold green]Access token (shown once — do not share):[/bold green]"
        )
        # soft_wrap keeps the URL on one line so the token stays copy-pasteable
        # even in a narrow / non-TTY log (e.g. `docker logs`, where Rich would
        # otherwise hard-wrap to width 80 and split the token). markup=False so
        # nothing in the URL is interpreted as Rich markup.
        console.print(
            f"http://{_display_host}:{web_port}/?token={effective_token}",
            soft_wrap=True,
            markup=False,
        )
        console.print()
    else:
        console.print(
            "[yellow]Local mode: no access token (loopback-only bind).[/yellow]"
        )

    # --- Start Flight server + HTTP sidecar ---
    # Pre-bind so the `finally` runs graceful shutdown -- which releases the file
    # cache process lock -- on EVERY exit path after cache init, not just a clean
    # uvicorn return. _setup_flight_server acquires the lock during cache init and
    # can still raise afterwards (e.g. a bad static source, or a bind failure
    # starting the flight thread below), so keeping the whole startup body inside
    # the try means such an early exit no longer orphans the lock as a stale lock
    # (biopb/biopb#515). uvicorn also installs its own SIGINT/SIGTERM handlers and
    # returns normally on shutdown (it does not re-raise), so cleanup must run in
    # `finally` rather than an except block regardless.
    flight_server = source_manager = watcher = precache_worker = None
    try:
        flight_server, source_manager, watcher, precache_worker = _setup_flight_server(
            server_config,
            host=host,
            port=port,
            writable=writable,
            token=effective_token,
        )

        # The HTTP sidecar is co-located with the Flight server and reaches it over
        # the loopback interface. A wildcard bind address is a bind target, not a
        # valid connect target, so dial the matching loopback explicitly — and
        # match the address family: the IPv4 wildcard maps to 127.0.0.1, the IPv6
        # wildcard to ::1. (A `::`-bound server with IPV6_V6ONLY set — the default
        # on some hosts — would refuse an IPv4 127.0.0.1 connection.)
        _flight_connect_host = effective_host
        if _flight_connect_host in ("0.0.0.0", ""):
            _flight_connect_host = "127.0.0.1"
        elif _flight_connect_host == "::":
            _flight_connect_host = "::1"
        flight_location = _grpc_location(
            _flight_connect_host, port or server_config.port
        )
        flight_thread = threading.Thread(target=flight_server.serve, daemon=True)
        flight_thread.start()

        # --- Build effective CORS origins ---
        if cors_origins:
            effective_cors = list(cors_origins)
        else:
            # No web app is bundled here, so there is no frontend origin to derive
            # by default. The control front reaches this sidecar over loopback for
            # the data API + /ws/render, so allow all loopback variants of the
            # server's own address; a browser app on any other origin must be
            # allowed explicitly via --cors.
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

            effective_cors = _expand_origin(f"http://{web_host}:{web_port}")

        # --- Start HTTP sidecar (blocks) ---
        console.print(
            f"[green]Starting HTTP sidecar at http://{web_host}:{web_port}[/green]"
        )
        console.print("Press Ctrl+C to stop\n")
        run_http_server(
            flight_location=flight_location,
            token=effective_token,
            host=web_host,
            port=web_port,
            cors_origins=effective_cors,
            config_path=str(config),
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
