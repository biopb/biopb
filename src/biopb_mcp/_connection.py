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
from typing import Dict, Tuple

from biopb.tensor import TensorFlightClient
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

from ._config import load_config, save_config

logger = logging.getLogger(__name__)

# Default tensor server URL when neither env nor config provides one.
_DEFAULT_URL = "grpc://localhost:8815"

# Catalogs larger than this switch to server-side SQL filtering.
SERVER_QUERY_THRESHOLD = 1000


class TensorConnection:
    """Owns the tensor client, source catalog, and connection settings."""

    def __init__(self, config: dict | None = None) -> None:
        self.client: TensorFlightClient | None = None
        self.sources: Dict[str, DataSourceDescriptor] = {}
        self.use_server_query: bool = False

        cfg = config if config is not None else load_config()
        self.url, self.token = self.resolve_from_config(cfg)

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
        """
        try:
            self.client = TensorFlightClient(url, token=token)
            sources = self.client.list_sources()
            self.url = url
            self.token = token
            self.sources = sources
            self.use_server_query = len(sources) > SERVER_QUERY_THRESHOLD
            self.persist_url()
            return sources
        except Exception:
            self.client = None
            self.sources = {}
            self.use_server_query = False
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
