"""Standalone tensor data-access service.

Owns the :class:`~biopb.tensor.TensorFlightClient`, the source catalog, and the
server URL/token resolution + persistence. Both the napari ``TensorBrowserWidget``
and the MCP kernel *consume* this service rather than the widget owning the
client.

This module deliberately imports **no Qt and no napari** so it can be used (and
unit-tested) without a GUI — only ``biopb.tensor`` and the local config module.

Threading: a ``TensorConnection`` instance is mutated on the Qt main thread (the
widget's auto-connect tick and button handlers) and read in the same kernel
process during ``execute_code``. The one off-thread writer is the background
source watcher (:meth:`start_source_watch`, issue #44), which re-lists from its
own daemon thread. It only ever *rebinds* ``self.sources`` to a fresh dict
returned by ``list_sources()`` (never mutates the dict in place), so readers on
other threads see either the whole old catalog or the whole new one under the
GIL — no torn reads, hence no lock. ``connect()`` is still expected on one
thread at a time.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse

from biopb._config_location import find_config
from biopb.tensor import TensorFlightClient
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

from ._config import CONFIG, get_setting

logger = logging.getLogger(__name__)

# Catalogs larger than this switch to server-side SQL filtering.
SERVER_QUERY_THRESHOLD = 1000


# Default location of the biopb server's config (matches the `biopb server
# start` CLI default). Prefers JSON over legacy TOML and warns when both exist;
# resolution is shared with the tensor server and the umbrella CLI via the core
# `biopb` package (biopb/biopb#34).
DEFAULT_SERVER_CONFIG = find_config()

# Hosts considered "local" — auto-starting a server only makes sense for these.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ServerStarting(Exception):
    """Raised by :meth:`TensorConnection.connect` when the server's health
    action reports a non-``SERVING`` status (e.g. ``STARTING`` while it scans
    its data folder; see biopb#17).

    Distinct from a connection failure: the server is up and will become ready,
    so callers should keep waiting with feedback rather than fail fast (issue
    #12). The catalog is *not* committed while this is raised.
    """

    def __init__(self, status, health=None) -> None:
        self.status = status
        self.health = health
        super().__init__(f"tensor server starting (status={status})")


def _starting_message(health) -> str:
    """Friendly 'server is starting' message, enriched with progress from the
    health payload (``source_count`` / ``uptime_seconds``) when available."""
    msg = (
        "Tensor server is starting — scanning its data folder; this can take "
        "a while for large catalogs."
    )
    if isinstance(health, dict):
        bits = []
        count = health.get("source_count")
        if count is not None:
            bits.append(f"{count} sources registered so far")
        uptime = health.get("uptime_seconds")
        if uptime is not None:
            bits.append(f"up {int(uptime)}s")
        if bits:
            msg += " (" + ", ".join(bits) + ")"
    return msg


def biopb_cli_available() -> bool:
    """Return True if the ``biopb`` command-line tool is on PATH."""
    return shutil.which("biopb") is not None


def is_local_url(url: str) -> bool:
    """Return True if *url* points at the local machine."""
    try:
        host = urlparse(url).hostname
    except Exception:
        return False
    return host is None or host in _LOCAL_HOSTS


# Substrings (lowercased) that mark a connect failure as an authentication
# problem across the Flight/gRPC stacks, vs a plain "server is down".
_AUTH_ERROR_MARKERS = (
    "unauthenticat",
    "unauthoriz",
    "permission denied",
    "invalid token",
    "missing token",
)
_UNREACHABLE_MARKERS = (
    "unavailable",
    "refused",
    "failed to connect",
    "deadline",
    "timed out",
    "timeout",
)


def connect_error_message(exc: Exception, url: str, token: str | None) -> str:
    """A human-readable reason for a failed :meth:`TensorConnection.connect`.

    Read by the widget's status label and the MCP ``server_status`` tool, which
    used to show a *blank* error here (issue #86 secondary). The common cause of
    the silent failure was an auth problem — a token needed but missing or wrong
    (e.g. a GUI-entered token lost across a kernel restart) — so that case names
    the fix; an unreachable server gets a friendly hint; anything else echoes the
    underlying error so nothing is hidden.
    """
    text = f"{type(exc).__name__}: {exc}".strip()
    low = text.lower()
    if any(m in low for m in _AUTH_ERROR_MARKERS):
        if token:
            return (
                f"Authentication failed: the tensor server at {url} rejected the token."
            )
        return (
            f"Authentication required: the tensor server at {url} needs a token. "
            "Enter it in the Tensor Browser (or set BIOPB_TENSOR_TOKEN)."
        )
    if any(m in low for m in _UNREACHABLE_MARKERS):
        return f"Cannot reach the tensor server at {url} — is it running?"
    if text:
        return f"Could not connect to the tensor server at {url}: {text}"
    return f"Could not connect to the tensor server at {url}."


class TensorConnection:
    """Owns the tensor client, source catalog, and connection settings."""

    def __init__(self, config: dict | None = None) -> None:
        self.client: TensorFlightClient | None = None
        self.sources: Dict[str, DataSourceDescriptor] = {}
        self.use_server_query: bool = False

        # Last connect outcome, read by the widget status label and the MCP
        # server_status tool: "disconnected" | "starting" | "connected" |
        # "error". last_message carries the friendly "starting…" detail.
        self.last_status: str = "disconnected"
        self.last_message: str = ""

        # Most recent health dict observed (from the connect probe and the
        # source-watch poll), cached so a UI can distinguish "catalog still
        # indexing" from "genuinely empty" without an extra round-trip on its
        # paint thread. None until the first health probe; may lack the
        # progressive-discovery freshness fields on an older server.
        self.last_health: dict | None = None

        cfg = config if config is not None else CONFIG.as_dict()
        self.url, self.token = self.resolve_from_config(cfg)

        # Optional callback invoked after every successful connect with the
        # final (url, token). Lets a caller react once the connection params are
        # settled -- e.g. the MCP bootstrap registers the dask chunk-cache
        # plugin here, since the token is only known after connect. Kept as a
        # plain callable so this service stays GUI/dask-free.
        self.on_connect = None

        # Optional callback invoked (from the watcher's daemon thread) after the
        # background source watcher re-lists the catalog, with the fresh sources
        # dict. The widget wires this to a queued Qt signal to rebuild its tree;
        # the kernel needs no callback since the agent reads ``sources`` live.
        self.on_sources_changed = None

        # Background source-catalog watcher state (issue #44). Started lazily by
        # :meth:`start_source_watch`; a daemon thread, so it never blocks
        # interpreter shutdown. The Event both signals the thread to stop and
        # backs its interruptible sleep.
        self._watch_thread: threading.Thread | None = None
        self._watch_stop = threading.Event()
        self._watch_lock = threading.Lock()

    @staticmethod
    def resolve_from_config(config: dict) -> Tuple[str, str | None]:
        """Resolve tensor server URL and token.

        Fallback order: environment variables -> config file -> default.
        """
        url = os.environ.get("BIOPB_TENSOR_URL") or get_setting(
            config, "tensor_browser.server_url"
        )
        token = os.environ.get("BIOPB_TENSOR_TOKEN") or None
        return url, token

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    def connect(
        self, url: str, token: str | None = None
    ) -> Dict[str, DataSourceDescriptor]:
        """Connect to *url* and list available sources.

        Updates ``client``/``sources``/``url``/``token``/``use_server_query`` and
        persists the URL to config. On failure, resets ``client``/``sources`` and
        re-raises so the caller can drive its own error display.

        Before trusting the catalog, a best-effort health probe gates on the
        server being ready: the server binds its port *before* finishing its
        data-folder scan, so a mid-scan ``list_sources()`` can return a partial
        catalog that looks complete. If the server reports a non-``SERVING``
        status (biopb#17), this raises :class:`ServerStarting` so the caller can
        keep waiting with feedback instead of failing or trusting a half-built
        catalog. The probe is *advisory*: a server with no health action (older
        servers) or a transient probe error falls through to ``list_sources()``,
        which stays the authoritative connectivity test — so older servers
        behave exactly as before (issue #12).
        """
        try:
            client = TensorFlightClient(url, token=token)

            # Advisory probe: any failure falls through to list_sources(),
            # which stays the authoritative connectivity test.
            try:
                health = client.health_check()
                self.last_health = health if isinstance(health, dict) else None
                status = (
                    health.get("status", "SERVING")
                    if isinstance(health, dict)
                    else "SERVING"
                )
            except Exception:  # noqa: BLE001
                health, status = None, "SERVING"
            if status != "SERVING":
                self.client = None
                self.sources = {}
                self.use_server_query = False
                self.last_status = "starting"
                self.last_message = _starting_message(health)
                raise ServerStarting(status, health)

            sources = client.list_sources()
            self.client = client
            self.url = url
            self.token = token
            self.sources = sources
            self.use_server_query = len(sources) > SERVER_QUERY_THRESHOLD
            self.last_status = "connected"
            self.last_message = ""
            self.persist_url()
            if self.on_connect is not None:
                try:
                    self.on_connect(self.url, self.token)
                except Exception:  # noqa: BLE001 - hook is best-effort
                    logger.exception("on_connect hook failed")
            return sources
        except ServerStarting:
            raise
        except Exception as exc:
            self.client = None
            self.sources = {}
            self.use_server_query = False
            self.last_status = "error"
            # Populate the reason (issue #86 secondary): the disconnect used to
            # surface as a blank error, hiding the common "token required/wrong"
            # cause behind a kernel restart.
            self.last_message = connect_error_message(exc, url, token)
            raise

    def refresh(self) -> Dict[str, DataSourceDescriptor]:
        """Re-list sources from the connected server."""
        if self.client is None:
            raise RuntimeError("Not connected")
        sources = self.client.list_sources()
        self.sources = sources
        self.use_server_query = len(sources) > SERVER_QUERY_THRESHOLD
        return sources

    def mark_disconnected(self, message: str = "") -> None:
        """Drop the client so ``is_connected`` reflects a lost server.

        ``is_connected`` is only ``client is not None`` and nothing re-validates
        it after ``connect()``, so a server that dies mid-session leaves the flag
        (and the widget's status line) stale on "connected". Callers that observe
        a failure against a previously-connected server — e.g. a failed manual
        ``refresh()`` — call this to reset the state to disconnected, mirroring
        the reset ``connect()`` already does on its own failure path. This is a
        stopgap; a live health signal is the real fix (biopb/biopb#319).
        """
        self.client = None
        self.sources = {}
        self.use_server_query = False
        self.last_status = "error"
        self.last_message = message

    def resolve_source(
        self,
        source_id: str,
        *,
        on_progress=None,
        should_cancel=None,
    ) -> DataSourceDescriptor:
        """Resolve an unresolved (cloud / synced-folder) source, then refresh.

        Delegates to the SDK's :meth:`TensorFlightClient.resolve` — which asks the
        server to hydrate the source (for a dehydrated placeholder this **downloads
        the whole file**, so it is slow and blocking and must be called off the GUI
        thread) and returns the now-populated ``DataSourceDescriptor``. The local
        catalog snapshot (:attr:`sources`) is then refreshed so callers re-render
        from the resolved field list. Returns the resolved descriptor.

        ``on_progress`` (called with a ``ResolveProgress`` per server heartbeat)
        and ``should_cancel`` (polled per heartbeat; raising
        :class:`~biopb.tensor.ResolveCancelled` when it returns True) are forwarded
        verbatim so a GUI can show progress and offer a Cancel button.
        """
        if self.client is None:
            raise RuntimeError("Not connected")
        descriptor = self.client.resolve(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )
        # resolve() already re-listed server-side; mirror it into our snapshot so
        # the widget/agent see the full field set without a second round-trip.
        self.refresh()
        return descriptor

    def warm_source(
        self,
        source_id: str,
        *,
        on_progress=None,
        should_cancel=None,
    ):
        """Hydrate-ahead a resolved multi-file source, recalling its member files.

        Delegates to :meth:`TensorFlightClient.warm` — the server walks the
        source directory and reads every file to force the sync engine's recall,
        so later reads are warm and never stall. The recall is entirely
        server-side (no pixels cross the wire); this is slow and blocking, so call
        it off the GUI thread. Only meaningful for multi-file (directory) sources;
        a single-file source returns immediately.

        ``on_progress`` (called with a ``WarmProgress`` per message) and
        ``should_cancel`` (polled per message; raising
        :class:`~biopb.tensor.ResolveCancelled` when it returns True) are forwarded
        verbatim so a GUI can show a non-modal progress + Cancel affordance.
        Returns the terminal ``WarmProgress`` snapshot. No catalog refresh — warm
        changes residency, not the descriptor.
        """
        if self.client is None:
            raise RuntimeError("Not connected")
        return self.client.warm(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )

    def is_localhost(self) -> bool:
        """True if the connected server runs on this machine.

        The tensor-browser drag-drop gate: a dropped path is a *client-side*
        filesystem path, meaningful to the server only when they share a
        filesystem, so the drop affordance is enabled only for a localhost
        server. Cheap and GUI-free so the widget can call it on every drag.
        """
        return is_local_url(self.url)

    def add_source(
        self,
        path: str,
        *,
        on_progress=None,
        should_cancel=None,
    ):
        """Register a local path on the server as a source, then refresh.

        Delegates to the SDK's :meth:`TensorFlightClient.add_source` — the wire
        entrypoint behind the tensor-browser drag-drop. The server interprets
        ``path`` on *its own* filesystem (a localhost server shares ours), routes
        it through the discovery pipeline, and streams progress as each source
        registers; a dropped directory may add several. The local catalog
        snapshot (:attr:`sources`) is then refreshed so the browser shows the new
        rows. Slow for a large directory, so call off the GUI thread.

        ``on_progress`` (an ``AddSourceProgress`` per registered source) and
        ``should_cancel`` (polled per message; a cancel stops the walk but keeps
        what is already registered) are forwarded verbatim. Returns the terminal
        ``AddSourceResult`` (added / already_present / failed); a directory
        dropped above the server's large-scan threshold comes back as a
        ``failed`` entry.
        """
        if self.client is None:
            raise RuntimeError("Not connected")
        result = self.client.add_source(
            path,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        self.refresh()
        return result

    def remove_source(self, root_url: str):
        """Deregister a drag-dropped source branch on the server, then refresh.

        Delegates to the SDK's :meth:`TensorFlightClient.remove_source` — the
        wire entrypoint behind the tensor-browser's dropped-root [x] button.
        ``root_url`` is a ``dnd://`` branch root; the server removes every source
        at or under it (a dropped folder's sources go as a unit) and refuses
        anything that is not a drag-dropped (``dnd://``) source. The local catalog
        snapshot (:attr:`sources`) is then refreshed so the row disappears. Quick,
        but still call off the GUI thread (a rescan may briefly hold the catalog
        lock). Returns the ``RemoveSourceResult`` (removed / failed).
        """
        if self.client is None:
            raise RuntimeError("Not connected")
        result = self.client.remove_source(root_url)
        self.refresh()
        return result

    def start_source_watch(
        self,
        min_interval: float | None = None,
        max_interval: float | None = None,
    ) -> None:
        """Start the background source-catalog watcher (issue #44).

        Spawns a daemon thread that periodically health-checks the connected
        server and re-lists sources whenever its ``source_count`` changes — so a
        catalog cached while the server was still indexing (a partial scene
        list) self-heals without a manual :meth:`refresh`. The poll interval
        backs off exponentially from *min_interval* to *max_interval* while the
        count is stable and snaps back to *min_interval* on a change, so active
        indexing is tracked promptly and a settled server is polled cheaply.

        Idempotent: a second call while a watcher is alive is a no-op. The
        thread is a daemon (never blocks interpreter shutdown); use
        :meth:`stop_source_watch` to end it early. Intervals default to the
        ``mcp.tensor.health_poll_*`` config; ``min_interval <= 0`` disables the
        watcher.

        Deliberately thread-based (not a ``QTimer``): it must run in the kernel
        with no assumption of a Qt event loop (the kernel can be headless), and
        its blocking ``health_check`` belongs off any GUI thread.
        """
        if min_interval is None:
            min_interval = CONFIG.get("mcp.tensor.health_poll_min_interval")
        if max_interval is None:
            max_interval = CONFIG.get("mcp.tensor.health_poll_max_interval")
        if not min_interval or min_interval <= 0:
            logger.info("Source watch disabled (min_interval=%s)", min_interval)
            return
        max_interval = max(max_interval or min_interval, min_interval)

        with self._watch_lock:
            if self._watch_thread is not None and self._watch_thread.is_alive():
                return
            self._watch_stop.clear()
            thread = threading.Thread(
                target=self._source_watch_loop,
                args=(min_interval, max_interval),
                name="biopb-source-watch",
                daemon=True,
            )
            self._watch_thread = thread
            thread.start()
        logger.info("Source watch started (%.1fs..%.1fs)", min_interval, max_interval)

    def stop_source_watch(self) -> None:
        """Signal the background source watcher to stop (best-effort)."""
        self._watch_stop.set()

    def _source_watch_loop(self, min_interval: float, max_interval: float) -> None:
        """Poll loop for the source watcher; runs on its own daemon thread.

        ``last_count`` is the server ``source_count`` we last reconciled the
        cached catalog against. It is reset to ``None`` whenever we observe no
        client (disconnected or between connects); the next connected poll
        adopts ``len(self.sources)`` — the size of the catalog cached at connect
        — as the baseline, so a connect-mid-index partial (server already
        reports more sources than we listed) re-lists on the very first tick.

        Note (issue #44): ``source_count`` does not grow when an
        already-listed source gains *scenes* (1 -> 18 tensors), so that specific
        partial isn't caught by count alone; the common case — sources still
        being discovered — is. A finer server-side readiness signal would
        subsume this (tracked upstream).
        """
        interval = min_interval
        last_count: int | None = None
        while not self._watch_stop.is_set():
            if self._watch_stop.wait(interval):
                break

            client = self.client
            if client is None:
                # Not connected (yet, or lost). Re-baseline on reconnect and
                # poll at the floor so a fresh catalog is adopted promptly.
                last_count = None
                interval = min_interval
                continue

            try:
                health = client.health_check()
            except Exception:  # noqa: BLE001 - transient; back off and retry
                interval = min(interval * 2, max_interval)
                continue

            self.last_health = health if isinstance(health, dict) else None
            count = health.get("source_count") if isinstance(health, dict) else None
            if count is None:
                # Server's health carries no source_count (older server) — there
                # is nothing to watch; idle at the cap.
                interval = max_interval
                continue

            if last_count is None:
                # Reconcile against the catalog cached at connect time.
                last_count = len(self.sources or {})

            if count != last_count:
                self._relist_from_watch()
                last_count = count
                interval = min_interval
            else:
                interval = min(interval * 2, max_interval)

    def _relist_from_watch(self) -> None:
        """Re-list sources from the watcher thread and fire the change hook."""
        try:
            sources = self.refresh()
        except Exception:  # noqa: BLE001 - keep the watcher alive on a blip
            logger.exception("Source watch: re-list failed")
            return
        logger.info("Source watch: catalog changed, re-listed %d sources", len(sources))
        callback = self.on_sources_changed
        if callback is not None:
            try:
                callback(sources)
            except Exception:  # noqa: BLE001 - hook is best-effort
                logger.exception("Source watch: on_sources_changed hook failed")

    def query_sources(self, sql: str):
        """Server-side SQL filter passthrough."""
        if self.client is None:
            raise RuntimeError("Not connected")
        return self.client.query_sources(sql)

    def persist_url(self) -> None:
        """Save the current URL via the config singleton.

        A targeted ``CONFIG.set`` touches only ``tensor_browser.server_url``;
        keys this service does not own (e.g.
        ``mcp.services.process_image_servers``) are preserved in the cached
        merged config and re-persisted intact.
        """
        CONFIG.set("tensor_browser.server_url", self.url)

    def health(self):
        """Return the server health check result, or ``None`` if not connected."""
        return self.client.health_check() if self.client else None

    def scan_in_progress(self) -> bool:
        """Whether the last-observed health said a full catalog scan is running.

        Reads the cached :attr:`last_health` (no round-trip), so a UI can tell
        "still indexing" from "genuinely empty" on its paint thread. ``False``
        when unknown or on an older server lacking the freshness field — i.e.
        callers treat absence as "not scanning", preserving old behavior.
        """
        h = self.last_health
        return bool(h.get("full_scan_in_progress")) if isinstance(h, dict) else False

    def scan_source_count(self) -> int:
        """The ``source_count`` from the last-observed health (0 if unknown)."""
        h = self.last_health
        if isinstance(h, dict):
            try:
                return int(h.get("source_count") or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    def can_autostart_server(self) -> bool:
        """Whether a local biopb server could be auto-started for this URL.

        True only when the configured URL is local *and* the ``biopb`` CLI is
        installed. Used as a last-resort fallback when the initial connect
        fails — see :meth:`start_local_server`.
        """
        return is_local_url(self.url) and biopb_cli_available()

    def connect_when_booted(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 60.0,
        poll_interval: float = 0.5,
        max_interval: float = 5.0,
    ) -> Dict[str, DataSourceDescriptor]:
        """Connect to a server we just launched, waiting it through boot.

        Polls :meth:`connect` with capped exponential backoff until it returns
        sources (``SERVING``) or *timeout* elapses. Unlike the normal connect
        path, a connection failure is *tolerated* here (the freshly-launched
        daemon may not have bound its port yet) and a ``STARTING`` status is
        kept waiting on — both just retry. Raises ``RuntimeError`` on timeout.
        """
        deadline = time.monotonic() + timeout
        interval = poll_interval
        last_exc: Exception | None = None
        while True:
            try:
                return self.connect(url, token)
            except ServerStarting as exc:
                last_exc = exc
            # Just-launched server: tolerate connection failures and keep
            # waiting until the deadline.
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"biopb server did not become ready in {timeout:.0f}s"
                ) from last_exc
            time.sleep(interval)
            interval = min(interval * 2, max_interval)

    def server_start_timeout(self) -> float:
        """The configured ``mcp.server_start_timeout`` boot-wait budget (s)."""
        return CONFIG.get("mcp.server_start_timeout")

    def auto_connect(self) -> None:
        """Connect using the resolved URL, with an automatic fallback policy.

        The single connect policy shared by **both** faces: try the resolved
        ``(url, token)``, wait the server through a ``STARTING`` data-folder
        scan, and — with no prompt — auto-start a local biopb server as a last
        resort when the URL is local and the ``biopb`` CLI is on PATH.

        Lives here, on the GUI-independent service, rather than in the MCP
        bootstrap or the widget so the policy is unit-testable without Qt/napari
        and neither caller reimplements it. It is a *mechanism the caller
        drives*, not something the constructor does: ``connect`` blocks on
        network I/O and ``on_connect`` is wired only after construction, so
        self-connecting in ``__init__`` would both stall the constructor and skip
        the cache-plugin hook. Both callers run it **off their main thread** —
        the MCP kernel on a daemon thread (``execute_code`` refreshes ``client``
        from ``_conn.client`` per job, so a late connect is still picked up), the
        widget on a worker thread that signals the tree render back to the Qt
        main thread (``_start_connect``) — because a blocking connect on the
        kernel's Qt loop would wedge ``start_kernel`` (and a modal prompt used
        to, which is why the prompt is gone).

        Fully best-effort: every failure path is swallowed (``last_status`` and
        ``last_message`` already record the outcome for ``server_status`` and the
        widget's error label), so a propagated error never aborts the caller.
        """
        url, token = self.url, self.token
        try:
            self.connect(url, token)
            return
        except ServerStarting:
            # Server is up but still scanning its data folder: keep waiting with
            # capped backoff until it reports SERVING (or the budget elapses).
            try:
                self.connect_when_booted(
                    url, token, timeout=self.server_start_timeout()
                )
            except Exception:  # noqa: BLE001
                logger.exception("auto_connect: %s did not finish starting", url)
            return
        except Exception:  # noqa: BLE001
            logger.info("auto_connect: %s unreachable", url)

        # Unreachable and no dialog to offer autostart: start a local server
        # ourselves when the URL is local and the biopb CLI is available.
        if self.can_autostart_server():
            try:
                logger.info("auto_connect: auto-starting local biopb server")
                self.start_local_server()
            except Exception:  # noqa: BLE001
                logger.exception("auto_connect: local server autostart failed")

    def launch_local_server(
        self,
        config_path: str | None = None,
        startup_timeout: float | None = None,
    ) -> None:
        """Spawn ``biopb server start`` and return as soon as it is launched.

        Runs ``biopb server start`` (with ``--config`` when the file exists).
        The daemon detaches immediately, so this returns quickly *without*
        waiting for the server to become reachable — callers poll for readiness
        themselves (the GUI does so non-blockingly; see
        :meth:`connect_when_booted` for the blocking equivalent). Raises on a
        non-zero exit. *startup_timeout* (defaulting to
        ``mcp.server_start_timeout``) bounds only the launch subprocess.

        Token handling is left entirely to the CLI: a default local server
        needs none, and if the user's config enables token generation the CLI
        prints it to the console. Output is not captured so any such printed
        token reaches the console.
        """
        if not biopb_cli_available():
            raise RuntimeError("biopb CLI not found on PATH")

        if startup_timeout is None:
            startup_timeout = self.server_start_timeout()

        config = Path(config_path) if config_path else DEFAULT_SERVER_CONFIG
        cmd = ["biopb", "server", "start"]
        if config.exists():
            cmd += ["--config", str(config)]

        logger.info("Starting local biopb server: %s", " ".join(cmd))
        result = subprocess.run(cmd, timeout=startup_timeout)
        if result.returncode != 0:
            raise RuntimeError(
                "`biopb server start` failed (exit "
                f"{result.returncode}); see the console and server logs"
            )

    def start_local_server(
        self,
        config_path: str | None = None,
        startup_timeout: float | None = None,
    ) -> Dict[str, DataSourceDescriptor]:
        """Launch a local biopb server daemon and block until it is ready.

        Convenience for headless/programmatic use: :meth:`launch_local_server`
        followed by :meth:`connect_when_booted`. The GUI uses the two steps
        separately so it can poll without freezing. Returns the source catalog
        on success; raises on failure.
        """
        # launch_local_server resolves a None timeout itself; we only need a
        # concrete value for connect_when_booted's deadline arithmetic below.
        self.launch_local_server(
            config_path=config_path, startup_timeout=startup_timeout
        )
        # The daemon detaches immediately; poll until it is reachable AND past
        # its data-folder scan (SERVING), tolerating refused/STARTING meanwhile.
        return self.connect_when_booted(
            self.url,
            self.token,
            timeout=startup_timeout or self.server_start_timeout(),
        )
