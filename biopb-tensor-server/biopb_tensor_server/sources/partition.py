"""Sort the configured ``[[sources]]`` into what the serve path does with each.

One classifier: :func:`partition_sources` is the only place that decides whether
an entry is registered as it is (static), kept live by the manager's rescan loop
(monitored, or an upstream), or walked once (scan-once), and it is the only one to warn about an
entry that is not what it asked for.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional

from biopb_tensor_server.adapters.remote_tensor import is_bare_host_upstream_url
from biopb_tensor_server.core.config import ServerConfig, SourceConfig
from biopb_tensor_server.core.discovery import AdapterRegistry
from biopb_tensor_server.serving.upload_manager import write_dir_under_root
from biopb_tensor_server.sources.resolve import resolve_all_sources

logger = logging.getLogger(__name__)


class Route(Enum):
    """Where one configured source goes on the serve path."""

    STATIC = "static"  # nothing to discover: expanded and registered as it is
    # Both owned by the manager's rescan loop: a watched directory is walked, a
    # bare-host tensor-server upstream is re-listed.
    MONITORED = "monitored"
    UPSTREAM = "upstream"
    SCAN_ONCE = "scan_once"  # a directory the first tick walks, then never again


def route_source(s: SourceConfig) -> Route:
    """Decide where a configured source goes; log why when that is not what the
    entry asked for.

    Every local directory is discovered by the manager after the server is SERVING,
    never expanded here: that would walk the tree an extra time before the server
    binds, and crash on a not-yet-mounted directory (biopb/biopb#54).
    """
    if s.is_remote:
        # A bare-host tensor-server upstream ("mirror everything") holds many
        # sources of its own, so it always goes to the manager's background
        # re-list. Every other remote (s3://, ...) names a single source.
        return Route.UPSTREAM if is_bare_host_upstream_url(s.url) else Route.STATIC

    path = s.local_path
    if path is None:  # unreachable: a local url always resolves to a path
        return Route.STATIC

    if path.is_file():
        if s.monitor:
            logger.warning(
                "Cannot live-monitor a single file; registering it as a static "
                "source: %s",
                s.url,
            )
        return Route.STATIC

    if s.monitor:
        if not path.exists():
            logger.warning(
                "Monitored path does not exist yet; will start monitoring when it "
                "appears: %s",
                s.url,
            )
        return Route.MONITORED

    # Not watched, but a directory still has to be discovered, once. A typed entry
    # (a zarr directory given as such) has nothing to discover, and a path that is
    # not there is left to the expansion, which warns and skips it.
    if not s.type and path.is_dir():
        return Route.SCAN_ONCE
    return Route.STATIC


class SourcePartition(NamedTuple):
    """The configured sources by route; ``static`` is already expanded."""

    static: List[SourceConfig]
    # The rest are discovered by the manager after SERVING.
    upstreams: List[SourceConfig]  # bare-host tensor servers, re-listed
    monitored: List[SourceConfig]  # watched local directories
    scan_once: List[SourceConfig]  # unwatched directories, walked once


def partition_sources(
    sources: List[SourceConfig],
    registry: Optional[AdapterRegistry] = None,
    *,
    credentials_config=None,
    write_dir: Optional[Path] = None,
) -> SourcePartition:
    """Partition configured sources for the serve path.

    Static sources are expanded here (a file, a typed entry, one remote source --
    nothing to walk); the directories and upstreams are left for the manager's
    rescan. See :func:`route_source`.
    """
    to_expand: List[SourceConfig] = []
    upstream_sources: List[SourceConfig] = []
    monitored_sources: List[SourceConfig] = []
    scan_once_sources: List[SourceConfig] = []

    for s in sources:
        route = route_source(s)
        if route is Route.STATIC:
            to_expand.append(s)
        elif route is Route.UPSTREAM:
            upstream_sources.append(s)
        elif route is Route.SCAN_ONCE:
            scan_once_sources.append(s)
        else:
            monitored_sources.append(s)

    # tolerant=True so one missing or broken static source is warned-and-skipped
    # rather than killing the server.
    expanded = resolve_all_sources(
        ServerConfig(sources=to_expand, credentials=credentials_config),
        registry,
        tolerant=True,
    )

    # A source an entry expands to may still land under a monitored root (a file
    # listed inside it); the rescan owns those. Remote sources are never under one.
    monitored_dirs = {ms.local_path for ms in monitored_sources if ms.local_path}
    static_sources = [
        s
        for s in expanded
        if s.is_remote
        or (
            s.local_path
            and not any(s.local_path.is_relative_to(md) for md in monitored_dirs)
        )
    ]

    # Upload stores are registered by the upload path, and the adapters decline
    # them if discovery reaches one, so a write_dir inside a scanned directory is
    # not catalogued twice -- but the walk still descends into every store and
    # stats its chunk files, and a store being written keeps its directory busy.
    scanned_dirs = (
        monitored_dirs
        | {s.local_path for s in scan_once_sources if s.local_path}
        | {
            s.local_path
            for s in static_sources
            if s.local_path and s.local_path.is_dir()
        }
    )
    inside = write_dir_under_root(write_dir, scanned_dirs)
    if inside is not None:
        logger.warning(
            "write_dir %s lies inside the source directory %s: its upload stores "
            "are walked whenever that directory is scanned (every rescan, if it "
            "is monitored). Keep write_dir outside every source directory.",
            write_dir,
            inside,
        )

    return SourcePartition(
        static_sources, upstream_sources, monitored_sources, scan_once_sources
    )
