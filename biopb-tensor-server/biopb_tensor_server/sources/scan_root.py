"""Claim discovery for one root: the walk behind a drop and a one-shot directory.

The periodic rescan walks its monitored roots with :class:`TreeScanner` and claims
off that snapshot (``discover_sources_from_entries``). A drag-dropped folder and a
configured ``monitor = false`` directory are the same operation on one root with
no history, so they take the same two steps rather than a walker of their own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    discover_sources_from_entries,
    get_file_identity,
)
from biopb_tensor_server.sources.tree_scanner import TreeScanner

logger = logging.getLogger(__name__)

# Nothing here reads the stability fields (no claim-phase gate runs), so the
# scanner's tuning is moot.
_ONE_SHOT_SCANNER = TreeScanner(stability_window=0.0, aggressive_dir_pruning=False)


def discover_under(
    root: Path,
    registry: AdapterRegistry,
    state: Optional[DiscoveryState] = None,
    *,
    cloud: bool = False,
) -> DiscoveryState:
    """Claim every dataset at or under ``root``.

    ``root`` may be a file, a dataset directory (a ``.zarr``), or a plain folder to
    recurse into. A root that is itself a dataset is claimed without walking: the
    interior of a claimed directory belongs to it, and stat-ing a 20k-chunk store
    entry by entry to learn that costs ~200 ms against ~0.2 ms.

    ``cloud`` marks ``root`` as a cloud/synced folder: dehydrated placeholders are
    admitted (the adapters claim them as unresolved sources) and the multi-file
    OME-TIFF / DICOM grouping is off, the same gating a monitored cloud root gets.
    """
    if state is None:
        state = DiscoveryState()

    root = Path(root)
    try:
        state.visited_identities.add(get_file_identity(root))
    except OSError:
        logger.debug("discover_under: cannot get identity for %s", root)
        return state

    claims = registry.get_claims_for_path(ClaimContext(root, cloud_root=cloud), state)
    if claims:
        state.add_claim(claims[0])
        logger.info(
            "discover_under: root %s claimed as %s", root, claims[0].source_type
        )
        return state
    if not root.is_dir():
        return state

    snapshot = _ONE_SHOT_SCANNER.scan_once(root, cloud=cloud)
    # The root was probed above, with no listing; its children are what is left.
    root_str = str(root.resolve())
    discover_sources_from_entries(
        (
            (path_str, entry.is_directory, entry.signature)
            for path_str, entry in snapshot.entry_states.items()
            if path_str != root_str
        ),
        registry,
        state=state,
        cloud_by_path=snapshot.cloud_by_path,
    )
    logger.debug("discover_under: %s -> %d sources", root, len(state.claims))
    return state
