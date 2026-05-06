"""Discovery module for tensor data sources.

Provides a claim-based discovery architecture where each adapter can
"claim" filesystem paths it recognizes. This enables:

1. Extensible format detection - new adapters register and participate
2. Cross-platform file identity tracking (symlink/hardlink safe)
3. Future filesystem monitoring compatibility (DiscoveryState)

Key components:
- SourceClaim: Represents a claimed data source
- AdapterRegistry: Registry of all adapter backends
- DiscoveryState: Persistent state for incremental discovery
- discover_sources(): Main discovery function
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Set, Type

if TYPE_CHECKING:
    from biopb_tensor_server.base import BackendAdapter


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
    return hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:16]


def walk_with_identity_tracking(
    root: Path,
    visited_identities: Set[str]
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
            # Skip hidden files/directories
            if path.name.startswith('.'):
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
            if path.is_dir() and not path.is_symlink():
                yield from walk_with_identity_tracking(path, visited_identities)
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
        primary_path: Main entry point for the source
        claimed_paths: All paths consumed by this claim (multi-node claims)
        source_id: Unique identifier (auto-generated if None)
        dim_labels: Optional dimension labels
        extra_config: Adapter-specific configuration (e.g., HDF5 dataset path)
    """

    __slots__ = (
        'source_type',
        'primary_path',
        'claimed_paths',
        'source_id',
        'dim_labels',
        'extra_config',
    )

    def __init__(
        self,
        source_type: str,
        primary_path: Path,
        claimed_paths: Set[Path],
        source_id: Optional[str] = None,
        dim_labels: Optional[List[str]] = None,
        extra_config: Optional[dict] = None,
    ):
        self.source_type = source_type
        self.primary_path = primary_path
        self.claimed_paths = claimed_paths
        self.source_id = source_id
        self.dim_labels = dim_labels
        self.extra_config = extra_config if extra_config is not None else {}

    def __repr__(self) -> str:
        return (
            f"SourceClaim(source_type={self.source_type!r}, "
            f"primary_path={self.primary_path!r}, "
            f"source_id={self.source_id!r})"
        )


class AdapterRegistry:
    """Registry of all adapter backends.

    Adapters register themselves and participate in discovery by
    implementing the claim() classmethod.

    Usage:
        registry = AdapterRegistry()
        registry.register(ZarrAdapter)
        registry.register(OmeZarrAdapter)
        claims = registry.get_claims_for_path(path, visited)
    """

    def __init__(self):
        self._adapters: List[Type[BackendAdapter]] = []
        self._type_to_adapter: Dict[str, Type[BackendAdapter]] = {}

    def register(self, cls: Type[BackendAdapter]) -> None:
        """Register an adapter class.

        Args:
            cls: BackendAdapter subclass with claim() method
        """
        self._adapters.append(cls)
        # Map source_type to adapter class (derived from claim returns)
        # We infer the type by checking a sample claim
        # This is done lazily when get_claims_for_path is called
        # For factory use, we can also set it explicitly via register_with_type

    def register_with_type(self, source_type: str, cls: Type[BackendAdapter]) -> None:
        """Register an adapter class with explicit source type mapping.

        Args:
            source_type: Source type string (e.g., "zarr", "ome-tiff")
            cls: BackendAdapter subclass
        """
        self._adapters.append(cls)
        self._type_to_adapter[source_type] = cls

    def get_claims_for_path(self, path: Path, visited_identities: Set[str]) -> List[SourceClaim]:
        """Ask all adapters to claim this path.

        Each adapter's claim() method is called in registration order.
        First adapter to return a non-None claim wins.

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities

        Returns:
            List of SourceClaim objects (may be empty if no adapter claims)
        """
        claims = []
        for adapter_cls in self._adapters:
            try:
                claim = adapter_cls.claim(path, visited_identities)
                if claim is not None:
                    claims.append(claim)
                    # Update type mapping for factory use
                    self._type_to_adapter[claim.source_type] = adapter_cls
            except Exception:
                # Adapter claim() should not raise, but handle gracefully
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
        consumed_paths: All paths consumed by any source
        visited_identities: File identities already visited
        on_source_added: Callback for source addition events
        on_source_removed: Callback for source removal events
    """

    claims: Dict[str, SourceClaim]
    path_to_source: Dict[Path, str]
    consumed_paths: Set[Path]
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
        self.consumed_paths = set()
        self.visited_identities = set()
        self.on_source_added = on_source_added
        self.on_source_removed = on_source_removed

    def add_claim(self, claim: SourceClaim) -> bool:
        """Add a claim with callback notification.

        Args:
            claim: SourceClaim to add

        Returns:
            True if added, False if path already claimed
        """
        if claim.primary_path in self.consumed_paths:
            return False

        # Generate source_id if not provided
        source_id = claim.source_id or generate_source_id(
            str(claim.primary_path), claim.source_type
        )

        # Update claim's source_id (important for callbacks)
        claim.source_id = source_id

        # Store claim
        self.claims[source_id] = claim
        self.path_to_source[claim.primary_path] = source_id
        self.consumed_paths.update(claim.claimed_paths)

        # Callback
        if self.on_source_added:
            self.on_source_added(claim)

        return True

    def remove_claim(self, path: Path) -> Optional[str]:
        """Remove claim by path (for file deletion events).

        Args:
            path: Primary path of the claim to remove

        Returns:
            source_id if removed, None if not found
        """
        source_id = self.path_to_source.get(path)
        if source_id is None:
            return None

        claim = self.claims.pop(source_id)
        self.path_to_source.pop(path)
        self.consumed_paths.difference_update(claim.claimed_paths)

        # Callback
        if self.on_source_removed:
            self.on_source_removed(source_id)

        return source_id

    def is_path_claimed(self, path: Path) -> bool:
        """Check if a path is already part of a claim."""
        return path in self.consumed_paths

    def get_source_for_path(self, path: Path) -> Optional[str]:
        """Get source_id that owns this path (reverse lookup)."""
        return self.path_to_source.get(path)

    def get_all_claims(self) -> List[SourceClaim]:
        """Get all claims as a list."""
        return list(self.claims.values())


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

    # Get identity for root itself
    try:
        root_identity = get_file_identity(root)
        state.visited_identities.add(root_identity)
    except OSError:
        return state

    # Check if root itself is a data source (e.g., a .zarr directory)
    claims = registry.get_claims_for_path(root, state.visited_identities)
    if claims:
        claim = claims[0]
        if claim.dim_labels is None and dim_labels is not None:
            claim.dim_labels = dim_labels
        state.add_claim(claim)
        return state  # Root claimed, no need to recurse

    # Walk filesystem
    for path in walk_with_identity_tracking(root, state.visited_identities):
        if state.is_path_claimed(path):
            continue

        claims = registry.get_claims_for_path(path, state.visited_identities)
        if claims:
            claim = claims[0]
            if claim.dim_labels is None and dim_labels is not None:
                claim.dim_labels = dim_labels
            # Add identities for claimed paths to visited set
            for claimed_path in claim.claimed_paths:
                try:
                    identity = get_file_identity(claimed_path)
                    state.visited_identities.add(identity)
                except OSError:
                    pass
            state.add_claim(claim)

    return state


def is_remote_url(url: str) -> bool:
    """Check if URL is a remote (non-local) URL.

    Args:
        url: URL string to check

    Returns:
        True if URL is remote (s3://, http://, etc.), False if local path
    """
    remote_prefixes = ('s3://', 'gs://', 'http://', 'https://', 'ftp://', 'file://')
    return url.lower().startswith(remote_prefixes)