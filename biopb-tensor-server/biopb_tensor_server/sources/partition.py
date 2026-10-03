"""Sort the configured ``[[sources]]`` into what the serve path does with each.

One classifier: :func:`partition_sources` is the only place that decides whether
an entry is kept live by the manager's rescan loop (monitored, or an upstream),
registered once by the manager's first tick (scan-once: a local directory, file or
typed dataset), or a single remote source registered as it is (static), and it is
the only one to warn about an entry that is not what it asked for.
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
from biopb_tensor_server.sources.roots import Root, RootKind, Roots

logger = logging.getLogger(__name__)


class Route(Enum):
    """Where one configured source goes on the serve path."""

    STATIC = (
        "static"  # one remote source (s3://, grpc://host/<id>): registered as it is
    )
    # Both owned by the manager's rescan loop: a watched directory is walked, a
    # bare-host tensor-server upstream is re-listed.
    MONITORED = "monitored"
    UPSTREAM = "upstream"
    # A local path the first tick registers, then never rescans: a directory is
    # walked, a file or typed dataset is claimed in place.
    SCAN_ONCE = "scan_once"


def route_source(s: SourceConfig) -> Route:
    """Decide where a configured source goes; log why when that is not what the
    entry asked for.

    Every local path is discovered by the manager after the server is SERVING, never
    expanded here: that would walk the tree an extra time before the server binds,
    and crash on a not-yet-mounted directory (biopb/biopb#54).
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
                "Cannot live-monitor a single file; registering it once instead: %s",
                s.url,
            )
        return Route.SCAN_ONCE

    if s.monitor:
        if not path.exists():
            logger.warning(
                "Monitored path does not exist yet; will start monitoring when it "
                "appears: %s",
                s.url,
            )
        return Route.MONITORED

    # Not watched, but still registered once. A path that is not there is left to
    # that pass, which warns and skips it.
    return Route.SCAN_ONCE


_ROOT_KIND = {
    Route.MONITORED: RootKind.MONITORED,
    Route.UPSTREAM: RootKind.UPSTREAM,
    Route.SCAN_ONCE: RootKind.SCAN_ONCE,
}


class SourcePartition(NamedTuple):
    """The configured sources by route.

    ``static`` is the single remote sources, already expanded. Everything the
    manager registers after SERVING (watched directories, scan-once paths,
    upstreams) is a root in ``roots``.
    """

    static: List[SourceConfig]
    roots: Roots


def partition_sources(
    sources: List[SourceConfig],
    registry: Optional[AdapterRegistry] = None,
    *,
    credentials_config=None,
    write_dir: Optional[Path] = None,
) -> SourcePartition:
    """Partition configured sources for the serve path.

    A single remote source is expanded here; every local path and upstream is left
    for the manager, as a root. See :func:`route_source`.
    """
    to_expand: List[SourceConfig] = []
    roots = Roots()

    for s in sources:
        route = route_source(s)
        if route is Route.STATIC:
            to_expand.append(s)
        else:
            roots.add(Root.from_config(s, _ROOT_KIND[route]))

    # tolerant=True so one missing or broken static source is warned-and-skipped
    # rather than killing the server.
    static_sources = resolve_all_sources(
        ServerConfig(sources=to_expand, credentials=credentials_config),
        registry,
        tolerant=True,
    )

    # A file or typed dataset listed inside a monitored directory is the rescan's:
    # registering it again here would claim it twice.
    monitored_dirs = {r.path for r in roots.of_kind(RootKind.MONITORED)}
    for root in roots.of_kind(RootKind.SCAN_ONCE):
        if (root.path.is_file() or root.source.type) and any(
            root.path.is_relative_to(md) for md in monitored_dirs
        ):
            roots.remove(root)

    # Upload stores are registered by the upload path, and the adapters decline
    # them if discovery reaches one, so a write_dir inside a scanned directory is
    # not catalogued twice -- but the walk still descends into every store and
    # stats its chunk files, and a store being written keeps its directory busy.
    scanned_dirs = monitored_dirs | {
        r.path for r in roots.of_kind(RootKind.SCAN_ONCE) if r.path.is_dir()
    }
    inside = write_dir_under_root(write_dir, scanned_dirs)
    if inside is not None:
        logger.warning(
            "write_dir %s lies inside the source directory %s: its upload stores "
            "are walked whenever that directory is scanned (every rescan, if it "
            "is monitored). Keep write_dir outside every source directory.",
            write_dir,
            inside,
        )

    return SourcePartition(static_sources, roots)
