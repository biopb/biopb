"""Standalone tensor data-access service.

Owns the :class:`~biopb.tensor.TensorFlightClient`, the source catalog, and the
server URL/token resolution + persistence. Both the napari ``TensorBrowserWidget``
and the MCP kernel *consume* this service rather than the widget owning the
client.

This module deliberately imports **no Qt and no napari** so it can be used (and
unit-tested) without a GUI — only ``biopb.tensor`` and the local config module.

Threading: a single ``TensorConnection`` instance is mutated on the Qt main
thread (the widget's auto-connect tick and button handlers) and read in the same
kernel process during ``execute_code``. There is no cross-thread access today, so
no locking is used. A future off-thread connect would need synchronization.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse

from biopb.tensor import TensorFlightClient
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

from ._config import load_config, save_config

logger = logging.getLogger(__name__)

# Default tensor server URL when neither env nor config provides one.
_DEFAULT_URL = "grpc://localhost:8815"

# Catalogs larger than this switch to server-side SQL filtering.
SERVER_QUERY_THRESHOLD = 1000

# Default location of the biopb server's TOML config (matches the
# `biopb server start` CLI default).
DEFAULT_SERVER_CONFIG = Path.home() / ".config" / "biopb" / "biopb.toml"

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

        cfg = config if config is not None else load_config()
        self.url, self.token = self.resolve_from_config(cfg)

        # Optional callback invoked after every successful connect with the
        # final (url, token). Lets a caller react once the connection params are
        # settled -- e.g. the MCP bootstrap registers the dask chunk-cache
        # plugin here, since the token is only known after connect. Kept as a
        # plain callable so this service stays GUI/dask-free.
        self.on_connect = None

    @staticmethod
    def resolve_from_config(config: dict) -> Tuple[str, str | None]:
        """Resolve tensor server URL and token.

        Fallback order: environment variables -> config file -> default.
        """
        url = (
            os.environ.get("BIOPB_TENSOR_URL")
            or config.get("tensor_browser", {}).get("server_url")
            or _DEFAULT_URL
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
        except Exception:
            self.client = None
            self.sources = {}
            self.use_server_query = False
            self.last_status = "error"
            self.last_message = ""
            raise

    def refresh(self) -> Dict[str, DataSourceDescriptor]:
        """Re-list sources from the connected server."""
        if self.client is None:
            raise RuntimeError("Not connected")
        sources = self.client.list_sources()
        self.sources = sources
        self.use_server_query = len(sources) > SERVER_QUERY_THRESHOLD
        return sources

    def query_sources(self, sql: str):
        """Server-side SQL filter passthrough."""
        if self.client is None:
            raise RuntimeError("Not connected")
        return self.client.query_sources(sql)

    def persist_url(self) -> None:
        """Save the current URL to config.

        Reloads from disk first so keys this service does not own (e.g.
        ``mcp.process_image_servers``) are not clobbered by a stale snapshot.
        """
        config = load_config()
        config.setdefault("tensor_browser", {})
        config["tensor_browser"]["server_url"] = self.url
        save_config(config)

    def health(self):
        """Return the server health check result, or ``None`` if not connected."""
        return self.client.health_check() if self.client else None

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
        return load_config().get("mcp", {}).get("server_start_timeout", 60.0)

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
        if startup_timeout is None:
            startup_timeout = self.server_start_timeout()
        self.launch_local_server(
            config_path=config_path, startup_timeout=startup_timeout
        )
        # The daemon detaches immediately; poll until it is reachable AND past
        # its data-folder scan (SERVING), tolerating refused/STARTING meanwhile.
        return self.connect_when_booted(
            self.url, self.token, timeout=startup_timeout
        )
