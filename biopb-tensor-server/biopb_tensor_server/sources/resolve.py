"""Turn configured source entries into the concrete sources the server serves.

A config entry is a request -- "serve /data", "mirror grpc://lab:8815" -- not a
source. This is where it becomes a list of concrete :class:`SourceConfig`
objects: directories walked through the adapters' ``claim()`` protocol, grpc
endpoints expanded by listing the upstream catalog.

Layering: resolution sits *above* the adapters, which is why it is not in
``core.config``. The dataclasses there are imported by ``cache``, ``adapters``,
``serving`` and ``sources``, so ``core.config`` has to stay below all of them and
could only reach the adapter registry through a deferred import -- one that
looked safe only because ``adapters/__init__`` happens to omit ``cached_source``,
the one adapter module that reaches the cache. Resolution runs on the serve /
validate boot path only, so it lives here and imports the registry outright.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.adapters.remote_tensor import (
    _split_grpc_url,
    list_upstream_source_ids,
    mirrorable_upstream_id,
    resolve_upstream_credentials,
)
from biopb_tensor_server.core.config import (
    ServerConfig,
    SourceConfig,
    detect_source_type,
)
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    discover_sources as claim_based_discover,
    generate_source_id,
    get_file_identity,
)
from biopb_tensor_server.core.errors import UpstreamConfigError

logger = logging.getLogger(__name__)


def _namespaced_source_id(alias: Optional[str], upstream_source_id: str) -> str:
    """Local source_id for a mirrored upstream source.

    The proxy serves many upstreams + local sources from one flat, source_id-keyed
    catalog, so an upstream's ids are namespaced by the configured ``alias``:
    ``<alias>__<upstream_source_id>`` (slash-free -- ``__`` is a cosmetic
    separator, and the upstream id is slash-free by the array_id spec). A lone
    upstream with no alias keeps the verbatim id.

    Note what is deliberately absent: the endpoint. A local id carries no host,
    port or scheme, so moving an upstream changes only the source's ``url`` --
    source_id, array_ids and the route inside every chunk_id all survive, and with
    them the segment cache and the ROI rows keyed on source_id. ``host:port``
    would disambiguate two upstreams just as well, but would re-key the whole
    mirror on a move; the alias is the stable stand-in that does not.

    The contract: an alias is part of the identity, not a label. Renaming one
    re-keys every source mirrored from that upstream.
    """
    return f"{alias}__{upstream_source_id}" if alias else upstream_source_id


def _discover_tensor_server(
    source: SourceConfig, credentials_config: Optional[Any]
) -> List[SourceConfig]:
    """Expand a ``tensor-server`` source into one concrete source per upstream tensor.

    ``grpc://host:port/<id>`` mirrors a single upstream source; a bare
    ``grpc://host:port`` connects to the upstream and mirrors *every* source it
    lists (the network analogue of directory discovery). Each concrete source
    carries the single-source url form and an alias-namespaced ``source_id`` so
    the adapter (``RemoteTensorAdapter.create_from_config``) and the rest of the
    server machinery treat it like any other source.
    """
    # EXPERIMENTAL: the tensor-server remote-source proxy is not yet stable. Its
    # config surface (url forms, `alias`, monitor re-list) and the on-disk
    # segment-cache keys for proxied sources may change without notice in a future
    # release (biopb/biopb#178). Warned once per configured upstream at expansion.
    logger.warning(
        "Source %r uses the EXPERIMENTAL tensor-server remote proxy: its config "
        "surface (url forms, 'alias', monitor re-list) and the on-disk cache keys "
        "for proxied sources may change without notice in a future release.",
        source.url,
    )
    endpoint, upstream_source_id = _split_grpc_url(source.url)

    if upstream_source_id is not None:
        # Single-source form: register under the alias-namespaced local id.
        if not mirrorable_upstream_id(upstream_source_id):
            # Named outright rather than reached by enumeration, so it is
            # refused rather than silently skipped. Same reasons either way
            # (``mirrorable_upstream_id``).
            raise ValueError(
                f"{source.url}: an upstream's {upstream_source_id!r} source is "
                f"its temp store, not a source to mirror -- everything on it "
                f"has a deadline set by that server. Upload what you need to "
                f"keep onto a source of its own."
            )
        local_id = _namespaced_source_id(source.alias, upstream_source_id)
        return [replace(source, source_id=local_id)]

    # Bare-host form: mirror every source on the upstream. Enumerate via the
    # complete server-side catalog (not the capped list_sources -- see
    # list_upstream_source_ids).
    from biopb.tensor import TensorFlightClient

    credentials = resolve_upstream_credentials(source, credentials_config)
    client = TensorFlightClient(
        endpoint,
        cache_bytes=0,
        token=credentials.token,
        tls_ca_pem=credentials.tls_ca_pem,
        tls_fingerprint=credentials.tls_fingerprint,
    )
    try:
        upstream_ids = sorted(list_upstream_source_ids(client, endpoint))
    finally:
        # Never let a failing close() replace the upstream error propagating out
        # of the try: body (biopb/biopb#529).
        try:
            client.close()
        except Exception:
            logger.debug("error closing upstream client", exc_info=True)

    expanded = []
    for upstream_id in upstream_ids:
        local_id = _namespaced_source_id(source.alias, upstream_id)
        expanded.append(
            replace(
                source,
                url=f"{endpoint}/{upstream_id}",
                source_id=local_id,
                type="tensor-server",
            )
        )
    return expanded


def _resolve_tensor_server_id_collisions(
    sources: List[SourceConfig],
) -> List[SourceConfig]:
    """Drop -- don't abort on -- source_id collisions involving a tensor-server proxy.

    With distinct aliases the namespaces are disjoint by construction, so a clash
    is a misconfiguration (two upstreams sharing an alias, or a local source named
    like a proxied id). A single bad entry must not take down the whole catalog,
    so keep the first source for each id and skip later colliders, logging the fix
    (set a distinct alias). Non-proxy collisions are left untouched (the historical
    last-wins at registration).
    """
    seen: Dict[str, SourceConfig] = {}
    result: List[SourceConfig] = []
    for src in sources:
        prior = seen.get(src.source_id)
        if prior is not None and "tensor-server" in (src.type, prior.type):
            logger.warning(
                "Skipping source_id %r from %s (%s): it collides with %s (%s) "
                "already in the catalog. Set a distinct 'alias' on the conflicting "
                "tensor-server entry to namespace its mirrored sources.",
                src.source_id,
                src.url,
                src.type,
                prior.url,
                prior.type,
            )
            continue
        seen.setdefault(src.source_id, src)
        result.append(src)
    return result


def _reroot_catalog_url(label: str, root_path: str, primary_path: str) -> str:
    """Re-root ``primary_path`` under ``label``, preserving its position beneath
    ``root_path``. Shared core of the two re-rooting entry points -- drag-drop
    (``SourceManager._drop_catalog_url``, ``label`` = the dropped item's basename)
    and a configured ``alias`` (``_alias_catalog_url``, ``label`` = the alias).

    The tensor-browser (and web viewer) build their tree by splitting each
    source's ``source_url`` on ``/``, so ``label`` becomes the top-level root and
    the sub-structure beneath ``root_path`` is preserved under it:

        root /data/exp, primary /data/exp            -> "<label>"
        root /data/exp, primary /data/exp/sub/b.tif  -> "<label>/sub/b.tif"

    Display-only: it feeds the descriptor's ``source_url`` and never the
    ``source_id`` (which hashes the raw path), so a bare virtual path with no
    scheme is fine.
    """
    try:
        rel = os.path.relpath(str(primary_path), str(root_path)).replace("\\", "/")
    except ValueError:  # different drive on Windows, etc. -- can't relativize
        rel = "."
    if rel in (".", "") or rel.startswith("../"):
        # primary IS the root (single file / dataset dir), or (defensively) not
        # under it -- keep the whole thing as one root, never emit a "../" url.
        return label
    return f"{label}/{rel}"


def _alias_catalog_url(alias: str, root_path: str, primary_path: str) -> str:
    """Catalog ``source_url`` that re-roots a configured local source under ``alias``.

    The config-line analogue of ``SourceManager._drop_catalog_url``: the root
    label is the configured ``alias`` (rather than a dropped item's basename), and
    the sub-structure of a configured folder is preserved relative to it:

        alias "exp", configure /data/exp/            (root_path == primary_path)
            -> "exp"
        alias "exp", configure folder /data/exp/ with
            .../exp/a.tif, .../exp/sub/b.tif -> "exp/a.tif", "exp/sub/b.tif"

    Display-only (never touches ``source_id``). Applied on the static / one-shot
    expand path; a monitored directory's alias is dropped upstream (its rescan
    re-discovers native paths), so this is never fed a live-monitored source.
    """
    return _reroot_catalog_url(alias, root_path, primary_path)


def discover_sources(
    source: SourceConfig,
    registry: Optional[AdapterRegistry] = None,
    credentials_config: Optional[Any] = None,
) -> List[SourceConfig]:
    """Expand a source config to actual data sources.

    Directory scanning and file typing are done by the adapters' claim()
    protocol -- the single source of truth for format detection
    (biopb/biopb#277 item B). The only URL-derived typing left here is remote
    scheme routing (grpc -> tensor-server).

    Supports multiple modes:
    - Explicit source: type set -> returned as-is (single source). source_id is
      always present (__post_init__ auto-fills it), so type is the discriminator.
    - Remote URL: requires explicit 'type' in config (cannot auto-discover)
    - tensor-server (grpc://) URL: a caching proxy in front of an upstream biopb
      tensor server -- a bare ``grpc://host:port`` mirrors *every* upstream source
      (one concrete source each, alias-namespaced), and ``grpc://host:port/<id>``
      mirrors a single upstream source.
    - Local file with no type: claim-based detection (error if unclaimed)
    - Local directory with no type: claim-based discovery

    Args:
        source: Source configuration
        registry: Optional adapter registry (uses default if None)
        credentials_config: Optional CredentialsConfig, used to authenticate the
            upstream ``list_sources`` call when expanding a tensor-server source.

    Returns:
        List of concrete SourceConfig objects (one per data source, NOT expanded to tensors)

    Raises:
        ValueError: If remote URL lacks explicit 'type'
    """
    if registry is None:
        registry = get_default_registry()

    # Case 0: Remote URLs require an explicit (or auto-detectable) type.
    # A grpc(+tls) endpoint auto-detects to "tensor-server"; other remote
    # schemes (s3://, http://, ...) still require an explicit 'type'.
    if source.is_remote:
        resolved_type = source.type or detect_source_type(source.url)
        if resolved_type is None:
            raise ValueError(
                f"Remote URL requires explicit 'type' in config: {source.url}"
            )
        if source.type is None:
            source = replace(source, type=resolved_type)
        if resolved_type == "tensor-server":
            return _discover_tensor_server(source, credentials_config)
        # For other remote URLs, return the source as-is (no directory discovery)
        return [source]

    # Local filesystem handling
    local_path = source.local_path
    if local_path is None:
        raise ValueError(f"Could not resolve local path from: {source.url}")

    if not local_path.exists():
        raise ValueError(f"Path does not exist: {local_path}")

    # Case 1: explicit type -> return as-is. source_id is always set by
    # __post_init__, so it never discriminates here; type is the real gate.
    if source.type and source.source_id:
        return [source]

    # Case 2: File with no type - try claim-based detection
    if local_path.is_file():
        # Try claim-based detection first. cloud_root carries the multi-file ban
        # (OME-TIFF/DICOM-series -> single file) onto a directly-configured cloud
        # file, matching the monitored path.
        ctx = ClaimContext(local_path, cloud_root=source.cloud)
        state = DiscoveryState()
        try:
            identity = get_file_identity(local_path)
            state.visited_identities.add(identity)
        except OSError:
            pass

        claims = registry.get_claims_for_path(ctx, state)
        if claims:
            claim = claims[0]
            return [_claim_to_source_config(claim, source)]

        # No adapter recognized the file. There is no legacy fallback: format
        # detection lives only in the adapters (biopb/biopb#277 item B), so an
        # unclaimed file is a hard error rather than a guessed (often wrong) type.
        raise ValueError(
            f"Could not detect type for file: {local_path}. "
            f"Please specify 'type' explicitly in config."
        )

    # Case 3: Directory with no type - use claim-based discovery.
    # First check if the directory itself is a data source
    ctx = ClaimContext(local_path, cloud_root=source.cloud)
    state = DiscoveryState()
    try:
        identity = get_file_identity(local_path)
        state.visited_identities.add(identity)
    except OSError:
        pass

    claims = registry.get_claims_for_path(ctx, state)
    if claims:
        claim = claims[0]
        return [_claim_to_source_config(claim, source)]

    # Directory is not itself a data source - do recursive claim-based scan. Under
    # a cloud root, admit dehydrated placeholders so the one-shot startup scan of a
    # monitor=false cloud directory still catalogues offline data as unresolved
    # sources, and set cloud_root so the multi-file OME-TIFF / DICOM-series ban
    # applies -- the same gating the monitored rescan uses (cloud-storage phase 2).
    state = claim_based_discover(
        local_path,
        registry,
        admit_nonresident=source.cloud,
        cloud_root=source.cloud,
    )
    return [_claim_to_source_config(claim, source) for claim in state.get_all_claims()]


def _claim_to_source_config(
    claim: SourceClaim, original_source: SourceConfig
) -> SourceConfig:
    """Convert a SourceClaim to SourceConfig.

    Args:
        claim: SourceClaim from discovery
        original_source: Original SourceConfig for credentials_profile inheritance

    Returns:
        SourceConfig with claim information
    """
    source_id = claim.source_id or generate_source_id(
        str(claim.primary_path), claim.source_type
    )

    # Handle HDF5 special case - needs dataset path
    dataset = None
    if claim.source_type == "hdf5" and claim.extra_config.get("needs_dataset"):
        # HDF5 claims have needs_dataset flag in extra_config. This will fail at
        # adapter creation unless dataset is provided; for backward compatibility,
        # pass through any original dataset.
        dataset = original_source.dataset

    return SourceConfig(
        type=claim.source_type,
        url=str(claim.primary_path),
        source_id=source_id,
        dataset=dataset,
        credentials_profile=original_source.credentials_profile,  # preserve credentials_profile
        cloud=original_source.cloud,  # propagate cloud gating to expanded sources
    )


def resolve_all_sources(
    config: ServerConfig,
    registry: Optional[AdapterRegistry] = None,
    *,
    sources: Optional[List[SourceConfig]] = None,
    tolerant: bool = False,
) -> List[SourceConfig]:
    """Resolve all sources in config, expanding directories.

    Uses claim-based discovery for automatic source detection.

    Args:
        config: Server configuration
        registry: Optional adapter registry (uses default if None)
        sources: Optional explicit list of source entries to expand. When None,
            ``config.sources`` is used. The serve path passes a filtered subset
            (local ``monitor=true`` directories are discovered by the rescan
            instead, not expanded here) to avoid an extra pre-bind walk that also
            crashes on a not-yet-mounted directory (biopb/biopb#54).
        tolerant: When True, a source that fails to resolve (e.g. a missing
            static path) is logged and skipped instead of aborting the whole
            expansion. Used by the serve path so one bad entry cannot take down
            the server; ``validate``/``list_tensors`` keep the default (False) so
            a broken source is surfaced as a hard error.

    Returns:
        List of all concrete SourceConfig objects (one per data source)
    """
    if registry is None:
        registry = get_default_registry()

    source_list = sources if sources is not None else config.sources
    credentials_config = getattr(config, "credentials", None)

    all_sources = []
    hdf5_warnings = []

    for source in source_list:
        try:
            discovered = discover_sources(source, registry, credentials_config)
        except UpstreamConfigError as e:
            if not tolerant:
                raise
            # Still skip rather than abort the boot -- one bad entry must not take
            # the server down -- but not at the volume of a missing static path.
            # The operator asked for a stronger trust anchor and the server is
            # coming up without the source that needed it; that is a broken
            # deployment, not a transient upstream (biopb/biopb#608).
            logger.error(
                "NOT SERVING %s: its credentials configuration is broken (%s). "
                "This is a configuration error, not an unreachable upstream -- "
                "it will not resolve until the config is corrected.",
                source.url,
                e,
            )
            continue
        except Exception as e:
            if not tolerant:
                raise
            logger.warning(
                "Skipping source that could not be resolved: %s (%s)",
                source.url,
                e,
            )
            continue
        # A local source's `alias` re-roots it (and everything discovered under a
        # configured folder) into its own catalog tree root -- the config-line
        # analogue of a drag-dropped folder becoming its own root. Compute the
        # display source_url now, while both the configured root and each concrete
        # child are in hand. Skipped for remote entries: a tensor-server upstream's
        # alias means the source_id namespace (handled by the proxy adapter's own
        # display authority), not a tree root. A monitored local *directory* never
        # reaches here (it is discovered by the rescan, not expanded), so its alias
        # is correctly never applied -- see _resolve_serve_sources's warning.
        reroot = bool(source.alias) and not source.is_remote
        root_path = source.local_path if reroot else None
        for src in discovered:
            # Track HDF5 sources that need dataset config
            if src.type == "hdf5" and src.dataset is None:
                hdf5_warnings.append(src.url)
            if reroot and root_path is not None:
                src = replace(
                    src,
                    _catalog_url=_alias_catalog_url(
                        source.alias, str(root_path), src.url
                    ),
                )
            all_sources.append(src)

    # source_id collisions involving a tensor-server proxy are a misconfiguration
    # (two upstreams sharing an alias, or a local source named like a proxied id):
    # the catalog is one flat source_id space, so a clash would silently shadow a
    # source. Drop the colliding entry (keeping the first) and warn with the fix --
    # a single bad source must not abort the whole catalog. Non-proxy collisions
    # keep the historical last-wins behavior.
    all_sources = _resolve_tensor_server_id_collisions(all_sources)

    # Print warnings for HDF5 files that need explicit dataset
    if hdf5_warnings:
        print("Warning: HDF5 files require explicit 'dataset' path in config:")
        for h5_url in hdf5_warnings[:5]:
            print(f"  - {h5_url}")
        if len(hdf5_warnings) > 5:
            print(f"  ... and {len(hdf5_warnings) - 5} more")

    return all_sources
