"""TIFF sequence adapters for tensor storage.

Handles plain TIFF file sequences and legacy MicroManager datasets.
OME-TIFF files are handled by the aicsimageio adapter.
"""

import json
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.discovery import (
    ClaimContext,
    SourceClaim,
    _is_offline_placeholder,
)

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState


# Known OME / MicroManager naming patterns handled by other adapters.
_TIFF_EXCLUDE_PATTERNS = {
    "*.ome.tif",
    "*.ome.tiff",
    "img_*.tif",
    "img_*.tiff",
    "img_channel*.tif",
    "img_channel*.tiff",
}


def _filter_tiff_candidates(files: List[Path]) -> List[Path]:
    """Drop files matching known OME / MicroManager naming patterns."""
    return [f for f in files if not any(f.match(p) for p in _TIFF_EXCLUDE_PATTERNS)]


def _mask_and_digits(name: str) -> Tuple[str, List[int]]:
    """Split a filename into its structural mask and its numeric fields.

    Each run of digits is replaced by ``#`` to form a *mask* (so files that
    belong to the same sequence share a mask), and the integer value of each
    run is returned in order.

    ``"s1-0001_bf.tif"`` -> ``("s#-#_bf.tif", [1, 1])``
    """
    digits = [int(m.group()) for m in re.finditer(r"\d+", name)]
    mask = re.sub(r"\d+", "#", name)
    return mask, digits


def _group_tiff_sequence(files: List[Path]) -> Optional[List[Path]]:
    """Group plain-TIFF files into one ordered sequence by a single varying field.

    Files are bucketed by their digit-run mask; the dominant (largest) bucket is
    inspected for exactly one numeric field that varies across its members (all
    other numeric tokens, e.g. the ``s1`` in ``s1-0001_bf.tif``, must be
    constant). The bucket is sorted by that varying field.

    Returns the sorted file list, or ``None`` if no valid single-varying-field
    sequence of at least three files exists. Never returns an empty list.
    """
    candidates = _filter_tiff_candidates(files)
    if len(candidates) < 3:
        return None

    # Bucket files by structural mask. Files with no digit runs get their own
    # singleton masks and fall out below (they can never form a varying field).
    groups: Dict[str, List[Tuple[Path, List[int]]]] = {}
    for f in candidates:
        mask, digits = _mask_and_digits(f.name)
        groups.setdefault(mask, []).append((f, digits))

    # Dominant bucket: most files, tie-broken by mask string for determinism.
    best_mask = max(groups, key=lambda m: (len(groups[m]), m))
    members = groups[best_mask]
    if len(members) < 3:
        return None

    n_fields = len(members[0][1])  # all members share the mask -> same count
    if n_fields == 0:
        return None

    varying = [
        pos
        for pos in range(n_fields)
        if len({digits[pos] for _, digits in members}) > 1
    ]
    if len(varying) != 1:
        return None

    vpos = varying[0]
    members.sort(key=lambda fd: fd[1][vpos])
    return [f for f, _ in members]


# =============================================================================
# TiffSequenceAdapter - Plain TIFF sequences (no metadata)
# =============================================================================


class TiffSequenceAdapter(SourceAdapter, TensorAdapter):
    """Adapter for plain TIFF file sequences in a directory (no metadata).

    Handles datasets where multiple TIFF files form a single logical image:
    - ND000_aligned.tiff, ND001_aligned.tiff, ND002_aligned.tiff, ...

    This is a single-tensor source that tracks file list and index mapping.
    Uses TiffFile().aszarr() for true tile-level lazy reading.

    Unlike MultiFileOmeTiffAdapter, this does NOT:
    - Parse OME-XML or MicroManager metadata
    - Handle 5D/6D coordinate mapping
    - Discover related files via companion files

    Simple axis inference:
    - (num_files, Y, X) for single-page files -> 't' or 'z'
    - (num_files, pages, Y, X) for multi-page files -> 't', 'z' or 'c'
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories containing plain TIFF file sequences.

        A sequence is detected (see ``_group_tiff_sequence``) when, among 3+
        TIFF files sharing a filename template (excluding OME/MicroManager
        patterns), exactly one numeric field varies across the names; that field
        orders the sequence and constant numeric tokens are ignored.

        Claiming is intentionally metadata-free — no per-file reads, so discovery
        scans stay cheap. Dimension consistency across the files is validated
        lazily in ``__init__`` (where every file is opened anyway), not here.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if sequence detected
        """
        # Only support local directories for now (remote glob/stat is expensive)
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Gather all TIFF files
        all_tiffs = list(ctx.glob("*.tif")) + list(ctx.glob("*.tiff"))
        tiff_files = [t._path for t in all_tiffs]  # Extract underlying Path objects

        # Check for metadata files - if present, don't claim (let other adapters handle)
        metadata_patterns = [
            "metadata.txt",
            "_metadata.txt",
            "DisplaySettings.json",
            "*.companion.ome",
        ]
        for pattern in metadata_patterns:
            # Route through ctx.glob (not ctx._path.glob) so the snapshot's cached
            # child listing serves the match without re-reading the directory
            # (biopb/biopb#65).
            if ctx.glob(pattern):
                return None

        # Group into a single ordered sequence by the one varying numeric field.
        # Filename-template based: no per-file metadata reads, so discovery scans
        # stay cheap. Dimension consistency is verified lazily in __init__, where
        # every file is opened anyway.
        sorted_files = _group_tiff_sequence(tiff_files)
        if sorted_files is None:
            return None

        # Dir-claiming policy (biopb/biopb): the directory IS the dataset
        # boundary. ``_group_tiff_sequence`` above is still needed to *decide*
        # whether to claim, but its file list need not be recorded as members:
        # claiming the dir already prunes its whole subtree, so the interior
        # TIFFs are never independently walked. Recording the glob would only
        # duplicate that prune and pin a brittle membership.
        state.try_claim_path(ctx.path_str)

        # Cloud-storage phase 2 (biopb/biopb#173): ``__init__`` opens EVERY file
        # in the sequence to validate dimension consistency (see below). Under a
        # cloud root those members may be dehydrated placeholders, so building the
        # adapter eagerly would recall the whole sequence -- one synced-folder
        # round-trip per file (~hundreds of ms each on OneDrive Files-On-Demand),
        # serialized -- during the synchronous startup scan. That wedges health at
        # STARTING for minutes (a 749-file sequence measured at ~4 min) before
        # ``mark_ready()`` is ever reached, and the per-file content skip the cloud
        # policy applies elsewhere never fires because the sequence claim records
        # only the directory (``is_file()`` is False, so ``_claim_is_unresolved``
        # has no member to test). Claim the directory provisionally and defer
        # construction to first access, where the sequence is hydrated and opening
        # every file is expected -- the same strategy MicroManagerLegacyAdapter
        # already uses for its placeholder metadata.txt. Gated on ``cloud_root``
        # (not residency) so the deferral still holds at resolve, when the files
        # are resident -- consistent with OmeTiffAdapter and the cloud opt-in's
        # "don't open it eagerly" contract. An unresolved source also advertises no
        # tensors, so the precache worker skips it instead of warming (recalling)
        # every chunk in the background.
        return SourceClaim(
            source_type="tiff-sequence",
            primary_path=ctx.path_str,
            unresolved=ctx.cloud_root,
        )

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "TiffSequenceAdapter":
        """Create adapter instance from SourceConfig."""
        return cls(str(source.url), source.source_id or "", source.dim_labels)

    def __init__(
        self,
        directory: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize TIFF sequence adapter.

        Args:
            directory: Path to directory containing TIFF sequence
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (if None, inferred from file count)
        """
        import tifffile

        self.directory = Path(directory)
        self.source_id = source_id
        self._source_url = str(directory)
        self._source_type = "tiff-sequence"
        self._io_lock = threading.Lock()

        # Discover and order the sequence with the same helper claim() uses, so
        # the runtime file list always matches the claimed member set / order.
        all_tiffs = list(self.directory.glob("*.tif")) + list(
            self.directory.glob("*.tiff")
        )
        sorted_files = _group_tiff_sequence(all_tiffs)
        if sorted_files is None:
            raise ValueError(f"No TIFF sequence found in {directory}")
        self._tiff_files = sorted_files

        # Open first file to get shape and tile info
        with tifffile.TiffFile(str(self._tiff_files[0])) as tf:
            first_page = tf.pages[0]
            self._dtype = str(first_page.dtype)

            # Tile info
            if first_page.is_tiled:
                self.is_tiled = True
                self.tile_width = first_page.tilewidth
                self.tile_length = first_page.tilelength
                self._spatial_chunk = [self.tile_length, self.tile_width]
            else:
                self.is_tiled = False
                self._spatial_chunk = [first_page.shape[0], first_page.shape[1]]

            # Pages per file
            n_pages_per_file = len(tf.pages)

        spatial_shape = (first_page.shape[0], first_page.shape[1])

        # Build file index map: (file_path, n_pages). We open every file here
        # anyway, so verify the sequence is uniform at near-zero extra cost —
        # this is the real consistency check that replaces the old (size-based)
        # proxy in claim(). The descriptor's dtype / pages-per-file and
        # get_data()'s stacking all assume the first file is representative, so
        # spatial shape, dtype, and page count must match across the sequence.
        self._file_ifd_map = []
        for file_path in self._tiff_files:
            with tifffile.TiffFile(str(file_path)) as tf:
                page = tf.pages[0]
                n_pages = len(tf.pages)
                if (page.shape[0], page.shape[1]) != spatial_shape:
                    raise ValueError(
                        f"Inconsistent TIFF dimensions in {directory}: "
                        f"{file_path.name} is {page.shape[:2]}, expected "
                        f"{spatial_shape}"
                    )
                if str(page.dtype) != self._dtype:
                    raise ValueError(
                        f"Inconsistent TIFF dtype in {directory}: "
                        f"{file_path.name} is {page.dtype}, expected {self._dtype}"
                    )
                if n_pages != n_pages_per_file:
                    raise ValueError(
                        f"Inconsistent TIFF page count in {directory}: "
                        f"{file_path.name} has {n_pages} pages, expected "
                        f"{n_pages_per_file}"
                    )
                self._file_ifd_map.append((file_path, n_pages))

        # Determine shape and dim labels
        # Shape: (num_files, pages_per_file, Y, X) or (num_files, Y, X)
        n_files = len(self._tiff_files)

        # Get spatial shape from first page (not series, which may include page dimension)
        spatial_shape = [first_page.shape[0], first_page.shape[1]]

        if n_pages_per_file > 1:
            # Multi-page files: (num_files, pages, Y, X)
            self.full_shape = [n_files, n_pages_per_file] + spatial_shape
            self.chunk_shape = [1, 1] + self._spatial_chunk
            if dim_labels:
                self.dim_labels = dim_labels
            else:
                self.dim_labels = ["t", "z", "y", "x"]
        else:
            # Single-page files: (num_files, Y, X)
            self.full_shape = [n_files] + spatial_shape
            self.chunk_shape = [1] + self._spatial_chunk
            if dim_labels:
                self.dim_labels = dim_labels
            else:
                self.dim_labels = ["t", "y", "x"]

        # Total IFDs for coordinate mapping
        self._total_ifds = sum(n for _, n in self._file_ifd_map)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds using tile-level lazy access.

        Uses TiffFile().aszarr() for true tile-level lazy reading.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        import tifffile
        import zarr

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))

        with self._io_lock:
            pages_per_file = self._file_ifd_map[0][1] if self._file_ifd_map else 1
            original_ndim = len(slices)

            # Build slice tuple: (file_slice, [page_slice], y_slice, x_slice)
            if original_ndim == 4:
                file_slice, page_slice, y_slice, x_slice = slices
            elif original_ndim == 3:
                file_slice, y_slice, x_slice = slices
                page_slice = (
                    slice(0, pages_per_file) if pages_per_file > 1 else slice(0, 1)
                )
            else:
                # Handle other dimensionalities
                file_slice = slices[0]
                page_slice = (
                    slice(0, pages_per_file) if len(slices) > 3 else slice(0, 1)
                )
                y_slice = slices[-2] if len(slices) >= 2 else slice(None)
                x_slice = slices[-1] if len(slices) >= 1 else slice(None)

            # Determine which files and pages to read
            file_indices = range(
                file_slice.start or 0,
                min(
                    file_slice.stop or len(self._file_ifd_map), len(self._file_ifd_map)
                ),
            )
            page_indices = range(
                page_slice.start or 0,
                min(page_slice.stop or pages_per_file, pages_per_file),
            )

            n_files = (file_slice.stop or len(self._file_ifd_map)) - (
                file_slice.start or 0
            )
            n_pages = (page_slice.stop or pages_per_file) - (page_slice.start or 0)

            # Read data via aszarr() for tile-level access
            result_pages = []
            for file_idx in file_indices:
                file_path, n_pages_in_file = self._file_ifd_map[file_idx]
                with tifffile.TiffFile(str(file_path)) as tf:
                    zarr_arr = zarr.open_array(tf.series[0].aszarr(), mode="r")
                    zarr_ndim = len(zarr_arr.shape)
                    for page_idx in page_indices:
                        if page_idx < n_pages_in_file:
                            if zarr_ndim == 2:
                                # 2D array (Y, X) - single page
                                page_data = zarr_arr[y_slice, x_slice]
                            else:
                                # 3D+ array (pages, Y, X)
                                page_data = zarr_arr[page_idx, y_slice, x_slice]
                            result_pages.append(page_data)

            # Stack into result array
            if result_pages:
                result_4d = np.stack(result_pages, axis=0)
                h, w = result_4d.shape[-2:]
                result_4d = result_4d.reshape(n_files, n_pages, h, w)
            else:
                result_4d = np.array([])

            # Reshape to match original ndim
            if original_ndim == 3:
                if pages_per_file > 1:
                    result = result_4d.reshape(n_files * n_pages, h, w)
                else:
                    result = result_4d.squeeze(axis=1)
            elif original_ndim == 4:
                result = result_4d
            else:
                result = result_4d

            return result

    def get_metadata(self) -> dict:
        """Return empty dict (no metadata for plain TIFF sequences)."""
        return {}


# =============================================================================
# MicroManagerLegacyAdapter - Legacy MicroManager datasets with JSON metadata
# =============================================================================


class MicroManagerLegacyAdapter(SourceAdapter, TensorAdapter):
    """Adapter for legacy MicroManager datasets with JSON metadata.

    Handles MicroManager v1 v2 datasets with metadata.txt containing:
    - Summary: IntendedDimensions, AxisOrder, Channels, Slices, Frames, Positions
    - Coords-Default/<filename>: PositionIndex, TimeIndex, ChannelIndex, SliceIndex
    - Metadata-Default/<filename>: UUID, Width, Height

    This adapter supports full 5D/6D datasets: (position, time, channel, z, y, x).

    Uses TiffFile().aszarr() for true tile-level lazy reading.
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories with MicroManager metadata files.

        Handles both:
        - MicroManager v1: metadata.txt directly in directory
        - MicroManager v2: DisplaySettings.json at root, metadata.txt in subdirectory

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if MicroManager format
        """
        # Only support local directories for now
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Check for v1 metadata.txt directly in this directory
        v1_metadata = None
        if ctx.join("metadata.txt").exists():
            v1_metadata = ctx.join("metadata.txt")._path

        # Check for v2 DisplaySettings.json (need to find subdirectory with metadata.txt)
        v2_data_dir = None
        if v1_metadata is None and ctx.join("DisplaySettings.json").exists():
            # Look for subdirectories containing metadata.txt
            for subdir in ctx._path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    candidate_metadata = subdir / "metadata.txt"
                    if candidate_metadata.exists():
                        v2_data_dir = subdir
                        v1_metadata = candidate_metadata
                        break

        if v1_metadata is None:
            return None

        # Cloud-storage phase 2: a non-resident metadata.txt placeholder cannot be
        # read+parsed without a whole-file recall (or it blocks offline). Defer
        # format validation and the coordinate-map build: the directory was
        # recognized structurally (a metadata.txt plus img_* TIFFs, all recall-
        # free), so claim it provisionally and resolve on first access. The
        # authoritative Coords map is rebuilt then.
        if _is_offline_placeholder(v1_metadata):
            # Dir-claiming policy: claim the dir (+ metadata.txt marker) only.
            # The dir claim prunes its whole subtree, so the interior img_*
            # TIFFs are never independently walked -- no need to enumerate them
            # as members (which under cloud would also be a recall-free but
            # divergent best-effort glob vs the Coords map).
            state.try_claim_path(ctx.path_str)
            state.try_claim_path(str(v1_metadata))
            if v2_data_dir:
                state.try_claim_path(str(v2_data_dir))
            return SourceClaim(
                source_type="micromanager-legacy",
                primary_path=ctx.path_str,
                unresolved=True,
            )

        # Parse metadata to confirm it's MicroManager v1 format
        try:
            content = v1_metadata.read_text()
            data = json.loads(content)

            # Check for MicroManager v1 format markers
            has_coords = any(k.startswith("Coords-") for k in data.keys())
            has_summary = "Summary" in data

            if not (has_coords or has_summary):
                return None
        except (json.JSONDecodeError, OSError):
            return None

        # Discover TIFF files (in current dir or v2 data subdirectory)
        search_dir = v2_data_dir if v2_data_dir else ctx._path
        tiff_patterns = [
            "img_channel*.tif",
            "img_channel*.tiff",
            "img_pos*_channel*.tif",
            "img_pos*_channel*.tiff",
            "img_*.tif",
            "img_*.tiff",
        ]
        tiff_files = []
        seen = set()
        for pattern in tiff_patterns:
            for f in search_dir.glob(pattern):
                if f not in seen:
                    seen.add(f)
                    tiff_files.append(f)

        if not tiff_files:
            return None

        # Dir-claiming policy: the glob above is kept only to *confirm* this is a
        # real MicroManager dataset (TIFFs present); the dir claim prunes the
        # subtree, so the interior TIFFs need not be recorded as members.
        state.try_claim_path(ctx.path_str)
        state.try_claim_path(str(v1_metadata))
        if v2_data_dir:
            state.try_claim_path(str(v2_data_dir))

        return SourceClaim(
            source_type="micromanager-legacy",
            primary_path=ctx.path_str,
        )

    @staticmethod
    def _find_metadata_file(directory: Path) -> Optional[Path]:
        """Find MicroManager metadata file in directory.

        Handles both v1 (metadata.txt directly) and v2 (DisplaySettings.json at root,
        metadata.txt in subdirectory) formats.

        Args:
            directory: Path to directory to search

        Returns:
            Path to metadata.txt if found, None otherwise
        """
        # Check for v1 format: metadata.txt directly in directory
        v1_metadata = directory / "metadata.txt"
        if v1_metadata.exists():
            return v1_metadata

        # Check for v2 format: DisplaySettings.json at root, metadata.txt in subdirectory
        display_settings = directory / "DisplaySettings.json"
        if display_settings.exists():
            for subdir in directory.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    candidate_metadata = subdir / "metadata.txt"
                    if candidate_metadata.exists():
                        return candidate_metadata

        return None

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "MicroManagerLegacyAdapter":
        """Create adapter instance from SourceConfig."""
        return cls(str(source.url), source.source_id or "", source.dim_labels)

    def __init__(
        self,
        directory: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize MicroManager legacy adapter.

        Args:
            directory: Path to directory containing MicroManager dataset
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels (if None, inferred from metadata)
        """
        import json

        import tifffile

        self.directory = Path(directory)
        self.source_id = source_id
        self._source_url = str(directory)
        self._source_type = "micromanager-legacy"
        self._io_lock = threading.Lock()

        # Find and parse metadata file
        metadata_file = self._find_metadata_file(self.directory)
        if metadata_file is None:
            raise ValueError(f"No MicroManager metadata file found in {directory}")

        with open(metadata_file) as f:
            self._raw_metadata = json.load(f)

        # Parse Summary to get dimensions
        summary = self._raw_metadata.get("Summary", {})
        intended = summary.get("IntendedDimensions", {})

        # Get dimension sizes
        self._n_positions = intended.get("position", summary.get("Positions", 1))
        self._n_times = intended.get("time", summary.get("Frames", 1))
        self._n_channels = intended.get("channel", summary.get("Channels", 1))
        self._n_z = intended.get("z", summary.get("Slices", 1))

        # Build coordinate map: (p, t, c, z) -> filename
        self._coord_map: Dict[Tuple[int, int, int, int], Path] = {}
        self._file_list: List[Path] = []

        for key in self._raw_metadata.keys():
            if key.startswith("Coords-"):
                coords = self._raw_metadata[key]
                # Extract path from key (Coords-Default/<filename> or Coords-<filename>)
                coords_prefix_len = len("Coords-")
                filepath_in_key = key[coords_prefix_len:]

                # Get indices
                pos_idx = coords.get("PositionIndex", 0)
                time_idx = coords.get("TimeIndex", 0)
                chan_idx = coords.get("ChannelIndex", 0)
                slice_idx = coords.get("SliceIndex", 0)

                # Look for file relative to directory (may be in subdirectory for v2)
                file_path = self.directory / filepath_in_key
                if file_path.exists():
                    self._coord_map[(pos_idx, time_idx, chan_idx, slice_idx)] = (
                        file_path
                    )
                    self._file_list.append(file_path)

        if not self._file_list:
            raise ValueError(f"No MicroManager TIFF files found in {directory}")

        # Sort file list for consistent ordering
        self._file_list = sorted(set(self._file_list))

        # Open first file to get shape and dtype
        with tifffile.TiffFile(str(self._file_list[0])) as tf:
            first_page = tf.pages[0]
            self._dtype = str(first_page.dtype)
            self._height = first_page.shape[0]
            self._width = first_page.shape[1]

            # Tile info
            if first_page.is_tiled:
                self.is_tiled = True
                self.tile_width = first_page.tilewidth
                self.tile_length = first_page.tilelength
                self._spatial_chunk = [self.tile_length, self.tile_width]
            else:
                self.is_tiled = False
                self._spatial_chunk = [self._height, self._width]

        # Get axis order from metadata
        axis_order = summary.get("AxisOrder", ["position", "time", "channel", "z"])
        if isinstance(axis_order, str):
            axis_order = [a.strip() for a in axis_order.split(",")]
        self._axis_order = axis_order

        # Map axis names to their counts
        axis_counts = {
            "position": self._n_positions,
            "time": self._n_times,
            "channel": self._n_channels,
            "z": self._n_z,
        }

        # Build shape following axis_order from metadata
        shape_axes = [a.lower() for a in axis_order]
        initial_shape = [axis_counts.get(a.lower(), 1) for a in axis_order] + [
            self._height,
            self._width,
        ]

        # Remove singleton dimensions while preserving axis correspondence
        # Only remove from non-spatial dimensions (keep y, x)
        self.full_shape = []
        self._shape_axes = []
        for i, (size, axis) in enumerate(zip(initial_shape, shape_axes)):
            if i < len(shape_axes):  # Non-spatial axis
                if size > 1:  # Keep non-singleton dimensions
                    self.full_shape.append(size)
                    self._shape_axes.append(axis)
            else:  # This shouldn't happen, shape_axes is shorter
                pass

        # Always append spatial dimensions
        self.full_shape.extend([self._height, self._width])

        # Chunk shape
        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self._spatial_chunk

        # Dimension labels (from remaining axes after singleton removal)
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            axis_alias = {
                "position": "p",
                "pos": "p",
                "time": "t",
                "frame": "t",
                "channel": "c",
                "z": "z",
                "slice": "z",
            }
            self.dim_labels = []
            for axis in self._shape_axes:
                label = axis_alias.get(axis.lower(), axis.lower()[0])
                self.dim_labels.append(label)
            self.dim_labels.extend(["y", "x"])

        # Build index for efficient lookups
        self._build_file_index()

    def _build_file_index(self) -> None:
        """Build index mapping for efficient coordinate lookups."""
        # Create reverse lookup: global_index -> file_path
        # For single-position/time/z datasets, this simplifies access
        self._index_to_file: Dict[int, Path] = {}

        if len(self.full_shape) == 3:
            # (channels, y, x) - single index is channel
            for (pos, time, chan, z), file_path in self._coord_map.items():
                if pos == 0 and time == 0 and z == 0:
                    self._index_to_file[chan] = file_path
        elif len(self.full_shape) == 4:
            # (channels, z, y, x) or similar
            for (pos, time, chan, z), file_path in self._coord_map.items():
                if pos == 0 and time == 0:
                    # Index = chan * n_z + z (assuming channel-z order)
                    idx = chan * self._n_z + z
                    self._index_to_file[idx] = file_path

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds using tile-level lazy access.

        Uses TiffFile().aszarr() for true tile-level lazy reading.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds
        """
        import tifffile
        import zarr

        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))

        with self._io_lock:
            ndim = len(self.full_shape)

            # Extract spatial slices (last 2 axes)
            y_slice = slices[-2] if len(slices) >= 2 else slice(None)
            x_slice = slices[-1] if len(slices) >= 1 else slice(None)

            # Determine coordinate ranges based on shape
            if ndim == 3:
                # (channels, y, x)
                chan_slice = slices[0]
                pos_range = [0]
                time_range = [0]
                chan_range = range(
                    chan_slice.start or 0, chan_slice.stop or self._n_channels
                )
                z_range = [0]
            elif ndim == 4:
                # (channels, z, y, x) or (time, channels, y, x) depending on axis_order
                if "z" in self.dim_labels[:2]:
                    # (channels, z, y, x) format
                    chan_slice = slices[0]
                    z_slice = slices[1]
                    pos_range = [0]
                    time_range = [0]
                    chan_range = range(
                        chan_slice.start or 0, chan_slice.stop or self._n_channels
                    )
                    z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
                else:
                    # (time, channels, y, x) format
                    time_slice = slices[0]
                    chan_slice = slices[1]
                    pos_range = [0]
                    time_range = range(
                        time_slice.start or 0, time_slice.stop or self._n_times
                    )
                    chan_range = range(
                        chan_slice.start or 0, chan_slice.stop or self._n_channels
                    )
                    z_range = [0]
            elif ndim == 5:
                # (time, channels, z, y, x) or (position, channels, z, y, x)
                if "p" in self.dim_labels:
                    pos_slice = slices[0]
                    chan_slice = slices[1]
                    z_slice = slices[2]
                    pos_range = range(
                        pos_slice.start or 0, pos_slice.stop or self._n_positions
                    )
                    time_range = [0]
                    chan_range = range(
                        chan_slice.start or 0, chan_slice.stop or self._n_channels
                    )
                    z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
                else:
                    time_slice = slices[0]
                    chan_slice = slices[1]
                    z_slice = slices[2]
                    pos_range = [0]
                    time_range = range(
                        time_slice.start or 0, time_slice.stop or self._n_times
                    )
                    chan_range = range(
                        chan_slice.start or 0, chan_slice.stop or self._n_channels
                    )
                    z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
            elif ndim == 6:
                # (position, time, channels, z, y, x)
                pos_slice = slices[0]
                time_slice = slices[1]
                chan_slice = slices[2]
                z_slice = slices[3]
                pos_range = range(
                    pos_slice.start or 0, pos_slice.stop or self._n_positions
                )
                time_range = range(
                    time_slice.start or 0, time_slice.stop or self._n_times
                )
                chan_range = range(
                    chan_slice.start or 0, chan_slice.stop or self._n_channels
                )
                z_range = range(z_slice.start or 0, z_slice.stop or self._n_z)
            else:
                # Fallback - assume all dimensions except last 2 are 1
                pos_range = [0]
                time_range = [0]
                chan_range = [0]
                z_range = [0]

            # Collect data for each coordinate
            result_pages = []
            for pos_idx in pos_range:
                for time_idx in time_range:
                    for chan_idx in chan_range:
                        for slice_idx in z_range:
                            coord = (pos_idx, time_idx, chan_idx, slice_idx)
                            file_path = self._coord_map.get(coord)

                            if file_path is None:
                                # Create empty array for missing data
                                h = (
                                    y_slice.stop - (y_slice.start or 0)
                                    if y_slice.stop
                                    else self._height
                                )
                                w = (
                                    x_slice.stop - (x_slice.start or 0)
                                    if x_slice.stop
                                    else self._width
                                )
                                result_pages.append(np.zeros((h, w), dtype=self._dtype))
                                continue

                            with tifffile.TiffFile(str(file_path)) as tf:
                                zarr_arr = zarr.open_array(
                                    tf.series[0].aszarr(), mode="r"
                                )
                                zarr_ndim = len(zarr_arr.shape)
                                if zarr_ndim == 2:
                                    page_data = zarr_arr[y_slice, x_slice]
                                else:
                                    page_data = (
                                        zarr_arr[0, y_slice, x_slice]
                                        if zarr_ndim >= 3
                                        else zarr_arr[y_slice, x_slice]
                                    )
                                result_pages.append(page_data)

            # Build output array with proper shape
            if not result_pages:
                return np.array([])

            # Stack pages and reshape to match the expected shape
            n_pos = len(pos_range)
            n_time = len(time_range)
            n_chan = len(chan_range)
            n_z = len(z_range)

            h = result_pages[0].shape[0] if result_pages else 0
            w = result_pages[0].shape[1] if result_pages else 0

            # Stack all pages and reshape
            result = np.stack(result_pages, axis=0)

            # Reshape to (pos, time, chan, z, h, w) based on actual ranges
            if ndim == 3:
                result = result.reshape(n_chan, h, w)
            elif ndim == 4:
                result = result.reshape(n_chan * n_z, h, w)
                if "z" in self.dim_labels[:2]:
                    result = result.reshape(n_chan, n_z, h, w)
                else:
                    result = result.reshape(n_time, n_chan, h, w)
            elif ndim == 5:
                if "p" in self.dim_labels:
                    result = result.reshape(n_pos, n_chan, n_z, h, w)
                else:
                    result = result.reshape(n_time, n_chan, n_z, h, w)
            elif ndim == 6:
                result = result.reshape(n_pos, n_time, n_chan, n_z, h, w)

            return result

    def get_metadata(self) -> dict:
        """Return parsed MicroManager metadata."""
        return self._raw_metadata
