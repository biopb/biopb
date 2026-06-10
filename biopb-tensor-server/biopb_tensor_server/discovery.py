"""Discovery module for tensor data sources.

Provides a claim-based discovery architecture where each adapter can
"claim" filesystem paths it recognizes. This enables:

1. Extensible format detection - new adapters register and participate
2. Cross-platform file identity tracking (symlink/hardlink safe)
3. Future filesystem monitoring compatibility (DiscoveryState)
4. Remote storage discovery via fsspec (S3, GCS, HTTP)

Key components:
- SourceClaim: Represents a claimed data source (str paths for URL support)
- AdapterRegistry: Registry of all adapter backends with remote claim support
- DiscoveryState: Persistent state for incremental discovery
- discover_sources(): Main discovery function (local + remote)
- discover_sources_async(): Background discovery with progress logging
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

if TYPE_CHECKING:
    from biopb_tensor_server.base import BackendAdapter
    from biopb_tensor_server.remote import RemoteStore

logger = logging.getLogger(__name__)


def get_file_identity(path: Path) -> str:
    """Get cross-platform stable file identity.

    Uses device + inode on Unix/NTFS, falls back to path hash on FAT32.

    Args:
        path: Path to get identity for (should be resolved symlink)

    Returns:
        Stable identity string for deduplication
    """
    try:
        real_path = path.resolve()
        stat = os.stat(real_path)

        # st_ino > 0 on Unix and Windows NTFS (sometimes)
        if stat.st_ino > 0:
            # Combine device + inode for uniqueness across mount points
            return f"{stat.st_dev}:{stat.st_ino}"
        else:
            # FAT32 or filesystem without stable inodes
            # Use path hash as fallback
            return _hash_path(real_path)
    except OSError:
        # Permission issue or broken symlink
        return _hash_path(path)


def _hash_path(path: Path) -> str:
    """Hash path for identity when inode unavailable."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


# Directory names that are never microscopy-data roots but are common, enormous,
# and — fatally on Windows/WSL — full of OneDrive "Files On-Demand" placeholders
# whose content recalls (and can hang) on read. Recursive discovery must never
# descend into them no matter how broad a root it is pointed at; a Windows user
# profile (e.g. /mnt/c/Users/<user>) otherwise stalls the server before it binds.
# Matched case-insensitively against the bare directory name.
_SKIP_DIR_NAMES = frozenset(
    {
        "appdata",
        "$recycle.bin",
        "$winreagent",
        "system volume information",
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
        "recovery",
        "node_modules",
    }
)

# Windows file-attribute bits marking content that is not resident on local disk
# (cloud placeholder / HSM stub). Defined numerically because the stdlib ``stat``
# module exposes only some of them. Reading such a file triggers an on-demand
# recall that can block indefinitely — OneDrive Files On-Demand over WSL ``drvfs``
# is the motivating case — so discovery skips it rather than let an adapter open
# it to sniff its format.
_FILE_ATTRIBUTE_OFFLINE = 0x00001000
_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
_OFFLINE_ATTR_MASK = (
    _FILE_ATTRIBUTE_OFFLINE
    | _FILE_ATTRIBUTE_RECALL_ON_OPEN
    | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)

# Skipping suspected cloud placeholders is best-effort and on by default; the
# POSIX signal (zero allocated blocks) is a heuristic, so allow an escape hatch in
# case a filesystem reports it spuriously and discovery wrongly skips real files.
_SKIP_OFFLINE = os.environ.get("BIOPB_DISCOVERY_SKIP_OFFLINE", "1") != "0"


def _is_skippable_system_dir(name: str) -> bool:
    """True for well-known system/cloud directory names discovery must not enter.

    Covers the fixed names in ``_SKIP_DIR_NAMES`` plus OneDrive roots, which are
    named ``OneDrive`` or ``OneDrive - <Org>``.
    """
    low = name.lower()
    if low in _SKIP_DIR_NAMES:
        return True
    return low == "onedrive" or low.startswith("onedrive -") or low.startswith(
        "onedrive-"
    )


def _is_offline_placeholder(path: Path) -> bool:
    """Best-effort: True when *path*'s content is not resident on local disk.

    ``os.stat`` reads metadata only and does **not** trigger a recall, so a
    placeholder can be detected and skipped before any adapter opens the file.
    Two signals, by platform:

    - Windows: the ``FILE_ATTRIBUTE_OFFLINE`` / ``RECALL_ON_*`` bits in
      ``st_file_attributes``.
    - POSIX/WSL: zero allocated blocks (``st_blocks == 0``) — the content is
      stubbed out. No logical-size floor: a placeholder is indistinguishable
      from a resident/sparse file by stat alone (verified — drvfs reports a
      OneDrive placeholder as a plain ``regular file`` with ``blocks == 0`` and
      no reparse tag or xattr), so we skip *every* zero-block file. Missing a
      benign one (resident tiny file, sparse file, empty file) is harmless;
      opening a real placeholder while offline would block indefinitely, which
      is the failure this guard exists to prevent.

    Never raises; returns ``False`` whenever the signal is unavailable, so a
    normal file is never wrongly skipped on that account.
    """
    if not _SKIP_OFFLINE:
        return False
    try:
        st = path.stat()
    except OSError:
        return False

    attrs = getattr(st, "st_file_attributes", 0)
    if attrs and (attrs & _OFFLINE_ATTR_MASK):
        return True

    st_blocks = getattr(st, "st_blocks", None)
    return st_blocks == 0


def should_skip_walk_entry(path: Path, is_dir: bool) -> bool:
    """Shared per-entry skip policy for discovery tree walks.

    Both traversals over the monitored trees route their skip decision through
    this one predicate — the claim walk (``walk_with_identity_tracking``) and the
    signature/stability scan (``SourceManager._scan_tree_state``) — so the policy
    cannot drift between them. That drift is exactly what left the signature scan
    descending into OneDrive placeholders the claim walk had already learned to
    prune.

    Decides on the entry's *name* and metadata only — never opens content, so it
    cannot itself trigger a cloud recall:

    - hidden entries (name starts with ``.``);
    - well-known system/cloud directories (``_is_skippable_system_dir``);
    - files whose content is not resident locally (``_is_offline_placeholder``).

    Checked against the supplied (pre-resolution) ``path`` name so a symlink or
    junction named e.g. ``OneDrive`` is pruned by its own name, not its target's.

    Loop protection (symlinks, Windows junctions, hardlinks, bind mounts) is
    **not** handled here — it needs per-walk identity state and is applied at
    each walk's recursion point via ``get_file_identity``.
    """
    name = path.name
    if name.startswith("."):
        return True
    if is_dir:
        return _is_skippable_system_dir(name)
    return _is_offline_placeholder(path)


class ClaimContext:
    """Unified path access for claim protocol.

    Wraps local Path or RemoteStore operations with identical interface,
    allowing claim() to work with both filesystem types uniformly.
    """

    def __init__(self, path: Path | str, store: Optional["RemoteStore"] = None):
        self._path = Path(path) if store is None else None
        self._store = store
        self._remote_path = str(path) if store is not None else None

    def is_dir(self) -> bool:
        """Check if path is directory."""
        if self._store:
            return self._store.isdir(self._remote_path)
        return self._path.is_dir()

    def is_file(self) -> bool:
        """Check if path is file."""
        if self._store:
            return self._store.isfile(self._remote_path)
        return self._path.is_file()

    def exists(self) -> bool:
        """Check if path exists."""
        if self._store:
            return self._store.exists(self._remote_path)
        return self._path.exists()

    def read_text(self, subpath: str = "") -> str:
        """Read file contents as text.

        Args:
            subpath: Relative path within this context (empty for current path)
        """
        if self._store:
            target = (
                (self._remote_path + "/" + subpath).lstrip("/")
                if subpath
                else self._remote_path
            )
            return self._store.read_text(target)
        target = self._path / subpath if subpath else self._path
        return target.read_text()

    def join(self, subpath: str) -> "ClaimContext":
        """Create context for subpath."""
        if self._store:
            new_path = (
                self._remote_path.rstrip("/") + "/" + subpath
                if self._remote_path
                else subpath
            )
            return ClaimContext(new_path, self._store)
        return ClaimContext(self._path / subpath)

    def glob(self, pattern: str) -> List["ClaimContext"]:
        """Find files matching pattern."""
        if self._store:
            matches = self._store.find(pattern, maxdepth=1)
            return [ClaimContext(m, self._store) for m in matches]
        return [ClaimContext(p) for p in self._path.glob(pattern)]

    @property
    def path_str(self) -> str:
        """Get path as string (for SourceClaim)."""
        if self._store:
            return self._store._join(self._remote_path)
        return str(self._path)

    @property
    def name(self) -> str:
        """Get filename/directory name."""
        if self._store:
            return self._remote_path.rstrip("/").split("/")[-1]
        return self._path.name

    @property
    def parent(self) -> "ClaimContext":
        """Get parent directory context."""
        if self._store:
            parent_path = (
                self._remote_path.rsplit("/", 1)[0] if "/" in self._remote_path else ""
            )
            return ClaimContext(parent_path, self._store)
        return ClaimContext(self._path.parent)

    @property
    def is_remote(self) -> bool:
        """Check if this is a remote context."""
        return self._store is not None

    @property
    def store(self) -> Optional["RemoteStore"]:
        """Get underlying RemoteStore if remote."""
        return self._store


def walk_with_identity_tracking(
    root: Path,
    visited_identities: Set[str],
    path_filter: Optional[Callable[[Path], bool]] = None,
) -> Iterator[Path]:
    """Walk filesystem with cross-platform identity tracking.

    Prevents infinite loops from symlink cycles and duplicate processing
    from hardlinks.

    Args:
        root: Root directory to walk
        visited_identities: Set of already-visited file identities

    Yields:
        Paths to files/directories (not yet claimed)
    """
    try:
        for path in root.iterdir():
            try:
                is_dir = path.is_dir()
            except OSError:
                continue  # Broken entry or permission issue

            # Shared skip policy: hidden entries, system/cloud directories
            # (AppData, OneDrive, Windows, …), and offline/placeholder files
            # whose content recalls on read. Pruned by name/metadata so the whole
            # subtree is skipped without a content open that could hang.
            if should_skip_walk_entry(path, is_dir):
                logger.debug("walk: skipping %s", path)
                continue

            if path_filter is not None and not path_filter(path):
                continue

            try:
                identity = get_file_identity(path)
            except OSError:
                continue  # Broken symlink or permission issue

            if identity in visited_identities:
                continue  # Already processed (cycle or hardlink duplicate)
            visited_identities.add(identity)

            yield path

            # Recurse into real directories (not symlinks pointing to dirs)
            if is_dir and not path.is_symlink():
                yield from walk_with_identity_tracking(
                    path,
                    visited_identities,
                    path_filter=path_filter,
                )
    except OSError:
        # Permission issue reading directory
        pass


class SourceClaim:
    """Represents a claimed data source.

    A claim describes what paths an adapter recognizes and wants to handle.
    Claims can be single-node (one file/dir) or multi-node (multiple files).

    Uses __slots__ for memory efficiency when scanning large directories.

    Attributes:
        source_type: Type identifier ("zarr", "ome-tiff", "hdf5", etc.)
        primary_path: Main entry point for the source (str to support URLs)
        source_id: Unique identifier (auto-generated if None)
        dim_labels: Optional dimension labels
        extra_config: Adapter-specific configuration (e.g., HDF5 dataset path)
        is_remote: Flag indicating if this is a remote source
    """

    __slots__ = (
        "source_type",
        "primary_path",
        "source_id",
        "dim_labels",
        "extra_config",
        "is_remote",
        "member_paths",
    )

    def __init__(
        self,
        source_type: str,
        primary_path: Path | str,
        source_id: Optional[str] = None,
        dim_labels: Optional[List[str]] = None,
        extra_config: Optional[dict] = None,
        is_remote: bool = False,
        member_paths: Optional[Set[str] | List[str]] = None,
    ):
        self.source_type = source_type
        self.primary_path = (
            str(primary_path) if isinstance(primary_path, Path) else primary_path
        )
        self.source_id = source_id
        self.dim_labels = dim_labels
        self.extra_config = extra_config if extra_config is not None else {}
        self.is_remote = is_remote
        normalized_member_paths = {self.primary_path}
        if member_paths is not None:
            normalized_member_paths.update(str(path) for path in member_paths)
        self.member_paths = normalized_member_paths

    def __repr__(self) -> str:
        return (
            f"SourceClaim(source_type={self.source_type!r}, "
            f"primary_path={self.primary_path!r}, "
            f"source_id={self.source_id!r}, "
            f"is_remote={self.is_remote!r})"
        )


class AdapterRegistry:
    """Registry of all adapter backends.

    Adapters register themselves and participate in discovery by
    implementing the claim() classmethod with ClaimContext and DiscoveryState.

    Usage:
        registry = AdapterRegistry()
        registry.register(ZarrAdapter)
        registry.register(OmeZarrAdapter)
        ctx = ClaimContext(path)  # or ClaimContext("", store) for remote
        claims = registry.get_claims_for_path(ctx, state)
    """

    def __init__(self):
        self._adapters: List[Type[BackendAdapter]] = []
        self._type_to_adapter: Dict[str, Type[BackendAdapter]] = {}

    def register(self, cls: Type[BackendAdapter]) -> None:
        """Register an adapter class.

        Args:
            cls: BackendAdapter subclass with claim(ctx, state) method
        """
        self._adapters.append(cls)

    def register_with_type(self, source_type: str, cls: Type[BackendAdapter]) -> None:
        """Register an adapter class with explicit source type mapping.

        Args:
            source_type: Source type string (e.g., "zarr", "ome-tiff")
            cls: BackendAdapter subclass
        """
        self._adapters.append(cls)
        self._type_to_adapter[source_type] = cls

    def get_claims_for_path(
        self, ctx: ClaimContext, state: DiscoveryState
    ) -> List[SourceClaim]:
        """Ask all adapters to claim this path.

        Each adapter's claim(ctx, state) method is called in registration order.
        First adapter to return a non-None claim wins.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            List of SourceClaim objects (may be empty if no adapter claims)
        """
        claims = []
        for adapter_cls in self._adapters:
            consumed_before = set(state.consumed_paths)
            try:
                claim = adapter_cls.claim(ctx, state)
                if claim is not None:
                    member_paths = set(state.consumed_paths) - consumed_before
                    member_paths.add(claim.primary_path)
                    claim.member_paths.update(member_paths)
                    claims.append(claim)
                    logger.debug(
                        f"Adapter {adapter_cls.__name__} claimed {ctx.path_str} as {claim.source_type}"
                    )
                    self._type_to_adapter[claim.source_type] = adapter_cls
            except Exception as e:
                logger.debug(
                    f"Adapter {adapter_cls.__name__} claim() raised exception: {e}"
                )
                continue
        return claims

    def get_adapter_for_type(self, source_type: str) -> Optional[Type[BackendAdapter]]:
        """Get adapter class for a source type.

        Args:
            source_type: Source type string

        Returns:
            BackendAdapter subclass or None if not registered
        """
        return self._type_to_adapter.get(source_type)


class DiscoveryState:
    """Persistent state for incremental discovery.

    Maintains bidirectional mappings for efficient source add/remove
    operations. Designed for future filesystem monitoring support.

    Attributes:
        claims: Forward mapping (source_id → SourceClaim)
        path_to_source: Reverse mapping (primary_path → source_id)
        consumed_paths: All paths consumed by any source (Set[str] for URLs)
        visited_identities: File identities already visited
        on_source_added: Callback for source addition events
        on_source_removed: Callback for source removal events
    """

    claims: Dict[str, SourceClaim]
    path_to_source: Dict[str, str]  # Changed from Dict[Path, str]
    source_to_paths: Dict[str, Set[str]]
    consumed_paths: Set[str]  # Changed from Set[Path]
    visited_identities: Set[str]
    on_source_added: Optional[Callable[[SourceClaim], None]]
    on_source_removed: Optional[Callable[[str], None]]

    def __init__(
        self,
        on_source_added: Optional[Callable[[SourceClaim], None]] = None,
        on_source_removed: Optional[Callable[[str], None]] = None,
    ):
        self.claims = {}
        self.path_to_source = {}
        self.source_to_paths = {}
        self.consumed_paths = set()
        self.visited_identities = set()
        self.on_source_added = on_source_added
        self.on_source_removed = on_source_removed

    def try_claim_path(self, path: str | Path, identity: Optional[str] = None) -> bool:
        """Check if path can be claimed and mark it as consumed.

        This is the callback for multi-file source discovery. Adapters call
        this for each path they want to claim. The method handles identity
        tracking and path consumption atomically.

        Args:
            path: Path to claim (local Path or remote URL string)
            identity: Optional pre-computed identity (computed if None)

        Returns:
            True if path is available and now claimed, False if already claimed/visited
        """
        path_str = str(path)
        if path_str in self.consumed_paths:
            return False

        if identity is None:
            try:
                identity = get_file_identity(Path(path_str))
            except OSError:
                identity = hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:16]

        if identity not in self.visited_identities:
            self.visited_identities.add(identity)

        self.consumed_paths.add(path_str)
        return True

    def add_claim(self, claim: SourceClaim, notify: bool = True) -> bool:
        """Add a claim with callback notification.

        Paths should already be consumed via try_claim_path() during discovery.

        Args:
            claim: SourceClaim to add
            notify: Whether to invoke on_source_added after storing the claim

        Returns:
            True if added, False if path already claimed
        """
        # Generate source_id if not provided
        source_id = claim.source_id or generate_source_id(
            str(claim.primary_path), claim.source_type
        )

        member_paths = set(claim.member_paths)
        member_paths.add(claim.primary_path)

        existing_owner = None
        for path in member_paths:
            owner = self.path_to_source.get(path)
            if owner is not None and owner != source_id:
                existing_owner = owner
                break

        if existing_owner is not None:
            return False

        # Update claim's source_id (important for callbacks)
        claim.source_id = source_id
        claim.member_paths = member_paths

        # Store claim
        self.claims[source_id] = claim
        self.source_to_paths[source_id] = member_paths
        for path in member_paths:
            self.path_to_source[path] = source_id
            self.consumed_paths.add(path)

        # Callback
        if notify and self.on_source_added:
            self.on_source_added(claim)

        return True

    def remove_claim(self, path: str, notify: bool = True) -> Optional[str]:
        """Remove claim by path (for file deletion events).

        Args:
            path: Primary path of the claim to remove (str to support URLs)
            notify: Whether to invoke on_source_removed after removing the claim

        Returns:
            source_id if removed, None if not found
        """
        source_id = self.path_to_source.get(path)
        if source_id is None:
            return None

        claim = self.claims.pop(source_id)
        member_paths = self.source_to_paths.pop(source_id, set(claim.member_paths))
        for member_path in member_paths:
            self.path_to_source.pop(member_path, None)
            self.consumed_paths.discard(member_path)

        # Callback
        if notify and self.on_source_removed:
            self.on_source_removed(source_id)

        return source_id

    def is_path_claimed(self, path: str) -> bool:
        """Check if a path is already part of a claim."""
        return path in self.consumed_paths

    def get_source_for_path(self, path: str) -> Optional[str]:
        """Get source_id that owns this path (reverse lookup)."""
        return self.path_to_source.get(path)

    def get_all_claims(self) -> List[SourceClaim]:
        """Get all claims as a list."""
        return list(self.claims.values())

    def get_paths_for_source(self, source_id: str) -> Set[str]:
        """Get all claimed member paths for a source."""
        return set(self.source_to_paths.get(source_id, set()))


def generate_source_id(url: str, source_type: str) -> str:
    """Generate deterministic unique source_id from URL.

    Uses SHA-256 hash of the URL to ensure uniqueness while remaining
    deterministic for the same URL.

    Args:
        url: URL or path to the data source
        source_type: Source type prefix (e.g., "zarr", "aics", "ome-zarr")

    Returns:
        Unique source_id like "zarr_a3f2b1c4d5e6"
    """
    if url is None or url == "":
        raise ValueError("Cannot generate source_id from empty URL")

    # For local paths, resolve to absolute path for consistency
    abs_path = str(Path(url).resolve())

    hash_hex = hashlib.sha256(abs_path.encode()).hexdigest()[:12]
    return f"{source_type}_{hash_hex}"


def discover_sources(
    root: Path,
    registry: AdapterRegistry,
    state: Optional[DiscoveryState] = None,
    dim_labels: Optional[List[str]] = None,
    path_filter: Optional[Callable[[Path], bool]] = None,
) -> DiscoveryState:
    """Recursive filesystem discovery with claim protocol.

    Walks the filesystem recursively, asking each registered adapter
    to claim paths it recognizes.

    Args:
        root: Root directory to scan
        registry: Adapter registry for claims
        state: Existing DiscoveryState to update (creates new if None)
        dim_labels: Optional dimension labels to apply to all claims

    Returns:
        DiscoveryState with all discovered sources
    """
    if state is None:
        state = DiscoveryState()

    logger.debug(f"discover_sources: scanning {root}")

    if path_filter is not None and not path_filter(root):
        logger.debug(f"discover_sources: skipping filtered root {root}")
        return state

    # Get identity for root itself
    try:
        root_identity = get_file_identity(root)
        state.visited_identities.add(root_identity)
    except OSError:
        logger.debug(f"discover_sources: cannot get identity for {root}")
        return state

    # Check if root itself is a data source (e.g., a .zarr directory)
    ctx = ClaimContext(root)
    claims = registry.get_claims_for_path(ctx, state)
    if claims:
        claim = claims[0]
        if claim.dim_labels is None and dim_labels is not None:
            claim.dim_labels = dim_labels
        state.add_claim(claim)
        logger.info(f"discover_sources: root {root} claimed as {claim.source_type}")
        return state  # Root claimed, no need to recurse

    # Walk filesystem
    paths_scanned = 0
    for path in walk_with_identity_tracking(
        root,
        state.visited_identities,
        path_filter=path_filter,
    ):
        paths_scanned += 1
        path_str = str(path)
        if state.is_path_claimed(path_str):
            continue

        ctx = ClaimContext(path)
        claims = registry.get_claims_for_path(ctx, state)
        if claims:
            claim = claims[0]
            if claim.dim_labels is None and dim_labels is not None:
                claim.dim_labels = dim_labels
            state.add_claim(claim)

    logger.debug(
        f"discover_sources: scanned {paths_scanned} paths, found {len(state.claims)} sources"
    )
    return state


def discover_remote_source(
    url: str,
    registry: AdapterRegistry,
    credentials_config: Optional[Any] = None,
    profile_name: Optional[str] = None,
    state: Optional[DiscoveryState] = None,
    dim_labels: Optional[List[str]] = None,
) -> DiscoveryState:
    """Discover a single remote source using fsspec.

    For remote URLs, we check if the URL itself is a data source.
    Unlike local discovery, we don't recursively scan remote directories
    by default (too slow on large buckets).

    Args:
        url: Remote URL (s3://..., gs://..., etc.)
        registry: Adapter registry for claims
        credentials_config: CredentialsConfig for authentication
        profile_name: Credential profile name to use
        state: Existing DiscoveryState to update (creates new if None)
        dim_labels: Optional dimension labels

    Returns:
        DiscoveryState with discovered remote source
    """
    from biopb_tensor_server.remote import RemoteStore

    if state is None:
        state = DiscoveryState()

    logger.debug(f"discover_remote_source: checking {url}")

    # Create RemoteStore for this URL
    store = RemoteStore.from_config(
        url=url,
        credentials_config=credentials_config,
        profile_name=profile_name,
    )

    # Get identity for remote path
    try:
        identity = store.get_identity("")
        if identity in state.visited_identities:
            logger.debug(f"discover_remote_source: {url} already visited")
            return state
        state.visited_identities.add(identity)
    except Exception as e:
        logger.debug(f"discover_remote_source: cannot get identity for {url}: {e}")

    # Check if root URL is a data source
    ctx = ClaimContext("", store)
    claims = registry.get_claims_for_path(ctx, state)
    if claims:
        claim = claims[0]
        if claim.dim_labels is None and dim_labels is not None:
            claim.dim_labels = dim_labels
        state.add_claim(claim)
        logger.info(f"discover_remote_source: {url} claimed as {claim.source_type}")

    return state


def discover_sources_async(
    sources: List[Any],
    registry: AdapterRegistry,
    server: Any,
    credentials_config: Optional[Any] = None,
    console: Optional[Any] = None,
    on_source_registered: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Background discovery with progress logging.

    Runs in a daemon thread:
    1. For each source in sources:
       2. Discover sources (local or remote)
       3. Log progress: "Found X sources from Y..."
       4. Register each discovered source with server

    Args:
        sources: List of SourceConfig objects
        registry: Adapter registry for claims
        server: TensorFlightServer instance for registration
        credentials_config: CredentialsConfig for authentication
        console: Rich Console for progress output (None for no output)
        on_source_registered: Optional callback(source_id, source_type) after registration

    Example output:
        Server started at grpc://0.0.0.0:8815
        Discovering sources from /data/local (1/3)
          ✓ plate-001 (ome-zarr)
          ✓ plate-002 (ome-zarr)
          Found 2 sources
        Discovering sources from s3://bucket/experiments (2/3)
          ✓ experiment-a (ome-zarr)
          Found 1 source
    """
    from biopb_tensor_server.config import SourceConfig

    total_sources = len(sources)
    state = DiscoveryState()

    for i, source in enumerate(sources):
        # Handle both SourceConfig and dict-like objects
        if isinstance(source, SourceConfig):
            url = source.url
            dim_labels = source.dim_labels
            is_remote = source.is_remote
            profile_name = source.credentials_profile
        else:
            url = source.get("url", "")
            dim_labels = source.get("dim_labels")
            is_remote = is_remote_url(url)
            profile_name = source.get("credentials_profile")

        if console:
            console.print(
                f"[dim]Discovering sources from {url} ({i + 1}/{total_sources})[/dim]"
            )

        discovered_count = 0

        if is_remote:
            # Remote discovery is disabled for now
            logger.warning(
                f"Remote source {url} is ignored - remote discovery is not yet enabled"
            )
        else:
            # Local discovery
            local_path = Path(url).resolve() if Path(url).exists() else None
            if local_path:
                state = discover_sources(
                    root=local_path,
                    registry=registry,
                    state=state,
                    dim_labels=dim_labels,
                )

        # Register discovered sources with server
        for claim in state.get_all_claims():
            if claim.source_id:
                # Create adapter for this source
                adapter_cls = registry.get_adapter_for_type(claim.source_type)
                if adapter_cls is None:
                    logger.error(f"No adapter for type: {claim.source_type}")
                    continue

                try:
                    # Build SourceConfig from claim
                    source_config = SourceConfig(
                        type=claim.source_type,
                        url=claim.primary_path,
                        source_id=claim.source_id,
                        dim_labels=claim.dim_labels,
                        credentials_profile=profile_name,
                        is_remote=claim.is_remote,
                    )

                    # Create adapter (unified create_from_config with optional credentials)
                    adapter = adapter_cls.create_from_config(
                        source_config, credentials_config
                    )

                    # Register with server
                    server.register_source(claim.source_id, adapter)
                    discovered_count += 1

                    if console:
                        console.print(
                            f"[green]  ✓ {claim.source_id}[/green] ({claim.source_type})"
                        )

                    if on_source_registered:
                        on_source_registered(claim.source_id, claim.source_type)

                except Exception as e:
                    logger.error(f"Failed to create adapter for {claim.source_id}: {e}")
                    if console:
                        console.print(f"[red]  ✗ {claim.source_id}[/red] ({e})")

        if console:
            console.print(f"[dim]  Found {discovered_count} sources[/dim]")

    if console:
        total_registered = len(state.claims)
        console.print(
            f"[green]Discovery complete: {total_registered} sources registered[/green]"
        )


def is_remote_url(url: str) -> bool:
    """Check if URL is a remote (non-local) URL.

    Args:
        url: URL string to check

    Returns:
        True if URL is remote (s3://, http://, etc.), False if local path
    """
    remote_prefixes = (
        "s3://",
        "gs://",
        "gcs://",
        "http://",
        "https://",
        "ftp://",
        "az://",
        "azure://",
    )
    return url.lower().startswith(remote_prefixes)
