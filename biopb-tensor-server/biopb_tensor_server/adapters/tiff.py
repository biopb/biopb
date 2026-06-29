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


# OME-TIFF naming patterns owned by the file-level OmeTiffAdapter. Excluded only
# OFF cloud (see ``exclude_ome`` below): OmeTiffAdapter claims individual .ome.tif
# FILES, not the directory, so if this directory-claiming adapter grouped a folder
# of .ome.tif into a plain sequence it would prune the subtree before any OME-XML /
# multi-scene parsing ran. Under a cloud root OmeTiffAdapter is disabled
# (``if ctx.cloud_root: return None``) and every .ome.tif degrades to a per-file
# aics source anyway, so deferring to it there is pointless -- grouping the
# directory into one sequence is the better fallback.
#
# MicroManager img_* patterns are deliberately NOT excluded. This adapter is
# registered *below* MicroManagerLegacyAdapter (see adapters/__init__.py), and
# registration order is load-bearing priority, so a valid MicroManager dataset is
# already claimed -- and its subtree pruned -- before this adapter is ever probed.
# The only img_* directories that reach here are ones MM declined (no metadata.txt
# or a corrupt/truncated one from an aborted acquisition). Grouping their frames
# into one sequence is exactly the wanted fallback, instead of letting the walk
# descend and register every frame as its own per-file aics source.
_TIFF_EXCLUDE_PATTERNS = {
    "*.ome.tif",
    "*.ome.tiff",
}


def _filter_tiff_candidates(files: List[Path], exclude_ome: bool = True) -> List[Path]:
    """Drop files matching OME-TIFF naming patterns owned by OmeTiffAdapter.

    ``exclude_ome`` is ``False`` under a cloud root, where OmeTiffAdapter is
    disabled and grouping .ome.tif into one sequence beats per-file fallback.
    """
    if not exclude_ome:
        return list(files)
    return [f for f in files if not any(f.match(p) for p in _TIFF_EXCLUDE_PATTERNS)]


# Minimum number of TIFFs for a directory to be claimed as a stacked sequence.
# Below this a directory is left to per-file fallback. The stack-all policy
# (#215) no longer requires a single varying numeric field -- any several plain
# TIFFs are claimed and stacked, with per-file provenance exposed for the agent.
_MIN_TIFF_FILES = 3


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


def _natural_key(name: str) -> List[Tuple[int, Any]]:
    """Sort key ordering embedded numbers numerically (``img_2`` < ``img_10``).

    Splits ``name`` into alternating digit / non-digit chunks. Each chunk is
    wrapped as ``(0, int)`` or ``(1, str)`` so numbers sort before text at any
    position and int/str never compare directly (no ``TypeError`` on names of
    differing structure). Gives the stacked file axis a sensible *default* order;
    the authoritative interpretation of that axis is the agent's, via
    :meth:`TiffSequenceAdapter.get_metadata`.
    """
    return [
        (0, int(t)) if t.isdigit() else (1, t.lower())
        for t in re.findall(r"\d+|\D+", name)
    ]


# A claimed directory should look like a *coherent* set of related files, not an
# incidental grab-bag of unrelated TIFFs that merely share a directory (and, after
# shape-bucketing, a pixel size). Stack-all (#215) stopped parsing what the
# filename fields *mean* -- that is the agent's job -- but "is this a dataset at
# all?" is still the adapter's call, so we keep a cheap, filename-only coherence
# gate at claim time. Two signals, either sufficient:
#   (a) a majority share one digit-template (mask) -- catches numbered sequences
#       including short stems like ``a1/a2/a3`` whose common prefix is tiny;
#   (b) a majority share a non-trivial common stem (prefix) -- catches sets that
#       vary by a token rather than a number, e.g. ``sp_0001_{red,green,blue}``
#       or MetaMorph ``..._w1DIC_.. / .._w2GFP_..``.
# A bare, no-number, no-stem set (``red/green/blue.tif``) is indistinguishable
# from a grab-bag by filename alone, so it is left to per-file fallback -- a
# graceful miss, never a wrong tensor.
_MIN_STEM = 3  # chars; a shorter shared prefix is too weak to imply coherence


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _looks_like_tiff_sequence(names: List[str]) -> bool:
    """Filename-only coherence gate (see the comment above). Pure, no I/O."""
    n = len(names)
    if n < _MIN_TIFF_FILES:
        return False
    majority = n // 2 + 1

    # (a) a digit-template (mask) shared by a majority of the names.
    mask_counts: Dict[str, int] = {}
    for nm in names:
        mask, _ = _mask_and_digits(nm)
        mask_counts[mask] = mask_counts.get(mask, 0) + 1
    if max(mask_counts.values()) >= majority:
        return True

    # (b) a non-trivial common stem shared by a majority. A majority-shared prefix
    # is contiguous once the names are sorted, so the LCP of the first and last
    # entry of each majority-sized window covers every candidate prefix.
    ordered = sorted(nm.lower() for nm in names)
    for i in range(n - majority + 1):
        if _common_prefix_len(ordered[i], ordered[i + majority - 1]) >= _MIN_STEM:
            return True
    return False


def _group_tiff_sequence(
    files: List[Path], exclude_ome: bool = True
) -> Optional[List[Path]]:
    """Group plain-TIFF files into one ordered sequence by a single varying field.

    .. note::
       Retained for unit coverage and as a single-field ordering reference. Under
       the stack-all policy (#215) it is no longer the claim gate: ``claim`` now
       claims any directory with enough TIFFs and ``__init__`` stacks the
       dominant *shape* bucket, delegating axis semantics to the agent.

    Files are bucketed by their digit-run mask; the dominant (largest) bucket is
    inspected for exactly one numeric field that varies across its members (all
    other numeric tokens, e.g. the ``s1`` in ``s1-0001_bf.tif``, must be
    constant). The bucket is sorted by that varying field.

    ``exclude_ome`` is forwarded to :func:`_filter_tiff_candidates`. claim() sets
    it to ``not ctx.cloud_root`` (OME ownership arbitration vs OmeTiffAdapter);
    read passes ``False`` (the directory is already claimed, so all TIFFs are
    members). See the call sites for the rationale.

    Returns the sorted file list, or ``None`` if no valid single-varying-field
    sequence of at least three files exists. Never returns an empty list.
    """
    candidates = _filter_tiff_candidates(files, exclude_ome=exclude_ome)
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

    Stack-all policy (#215): every uniformly-shaped TIFF in the directory is
    stacked along an opaque file axis (label ``i``); the axis's semantic
    structure (channel / time / site / z -- e.g. MetaMorph ``_w/_s/_t`` or
    ``_red/_green/_blue``) is deliberately NOT inferred here. Instead the per-file
    names are exposed via ``get_metadata`` so a downstream agent can parse them
    and reshape / relabel. This is metadata-free, never silently drops files
    (odd-shaped siblings are listed too), and avoids guessing axes wrong.

    Shape:
    - (num_files, Y, X) for single-page files -> ['i', 'y', 'x']
    - (num_files, pages, Y, X) for multi-page files -> ['i', 'z', 'y', 'x']
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim directories holding several plain TIFFs (stack-all, #215).

        Claim when at least ``_MIN_TIFF_FILES`` TIFFs are present (excluding OME
        names owned by OmeTiffAdapter) AND their names cohere as a related set
        (see ``_looks_like_tiff_sequence``). The previous single-varying-numeric-
        field requirement is gone: directories whose filenames encode several axes
        (channel x site x time, ``_red/_green/_blue``, MetaMorph ``_w/_s/_t``) are
        now claimed too. ``__init__`` stacks the dominant *shape* group and
        exposes per-file names for the agent to interpret -- no filename-pattern
        inference, and no silent channel drop. The coherence gate only avoids
        welding an incidental grab-bag of unrelated TIFFs into one tensor; it does
        NOT parse what the fields mean.

        Claiming is intentionally metadata-free — no per-file reads, so discovery
        scans stay cheap. Stackability (shape/dtype/pages) is resolved lazily in
        ``__init__`` (where every file is opened anyway), not here.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with directory if enough TIFFs are present
        """
        # Only support local directories for now (remote glob/stat is expensive)
        if ctx.is_remote or not ctx.is_dir():
            return None

        # Gather all TIFF files. Match common extension cases (MetaMorph and
        # other instrument exports use uppercase ``.TIF``), deduping by path so
        # case-insensitive filesystems don't double-count. Route through ctx.glob
        # so the snapshot's cached child listing serves the match (biopb#65).
        seen: Dict[str, Path] = {}
        for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            for t in ctx.glob(pat):
                seen[str(t._path)] = t._path
        tiff_files = list(seen.values())

        # OME guards arbitrate ownership against the file-level OmeTiffAdapter, and
        # apply only OFF cloud. Under a cloud root OmeTiffAdapter is disabled (it
        # returns None for every .ome.tif / .companion.ome), so the files it would
        # normally own degrade to per-file aics sources; deferring to it there is
        # pointless and grouping the directory is the better fallback. This is a
        # claim-time-only decision: once the directory is claimed, __init__ reads
        # ALL TIFFs (exclude_ome=False) since OmeTiffAdapter is already locked out.
        exclude_ome = not ctx.cloud_root

        # OME companion guard: a *.companion.ome marks this directory as a
        # multi-file OME-TIFF set owned by the file-level OmeTiffAdapter, so don't
        # claim it out from under that adapter. MicroManager metadata.txt /
        # DisplaySettings.json are deliberately NOT guarded here -- MicroManager-
        # LegacyAdapter has higher priority and prunes any valid MM dataset before
        # this adapter runs, so a metadata file that survives to here belongs to a
        # dataset MM could not parse (e.g. a truncated metadata.txt from an aborted
        # acquisition); falling back to a plain sequence claim is preferable to
        # registering every frame as its own per-file source. Route through
        # ctx.glob (not ctx._path.glob) so the snapshot's cached child listing
        # serves the match without re-reading the directory (biopb/biopb#65).
        if exclude_ome and ctx.glob("*.companion.ome"):
            return None

        # Stack-all claim gate (#215): enough plain TIFFs present AND their names
        # cohere as a related set (not a grab-bag). No filename-template *parsing*
        # -- multi-field names are claimed too and sorted out by the agent at read
        # time -- only a coherence check. Metadata-free: filenames only, no reads.
        candidates = _filter_tiff_candidates(tiff_files, exclude_ome=exclude_ome)
        if len(candidates) < _MIN_TIFF_FILES:
            return None
        if not _looks_like_tiff_sequence([p.name for p in candidates]):
            return None

        # Dir-claiming policy (biopb/biopb): the directory IS the dataset
        # boundary. Claiming the dir prunes its whole subtree, so the interior
        # TIFFs are never independently walked and need not be recorded as
        # members -- that would only duplicate the prune and pin a brittle
        # membership.
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

        # Gather every TIFF in the claimed directory. Unlike claim(), read does
        # NOT exclude OME names: the OME exclusion is a claim-time ownership
        # decision; by now the directory is claimed and its subtree pruned, so
        # every TIFF here is a member. Natural-sort for a stable, numeric-aware
        # default order (``img_2`` before ``img_10``); what that order *means* is
        # the agent's to decide (see get_metadata), not ours.
        all_tiffs = sorted(
            (
                p
                for p in self.directory.iterdir()
                if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
            ),
            key=lambda p: _natural_key(p.name),
        )
        if not all_tiffs:
            raise ValueError(f"No TIFF files found in {directory}")

        # Stack-all policy (#215). A dense tensor requires its members to share
        # shape / dtype / page-count -- a *physical* constraint, not a naming one.
        # Bucket the directory by that stackability signature and take the largest
        # uniform group as the tensor. Files that do not fit (thumbnails, a max-
        # projection, an odd overview) are NOT stacked but their names are still
        # surfaced via get_metadata(), so the directory is represented losslessly.
        # We deliberately do not group by filename pattern or infer what the file
        # axis means (channel / time / site / z): that ambiguous, metadata-free
        # inference is delegated to the agent, which gets the per-file names
        # alongside the array. This replaces the old "single varying field or
        # raise" grouping -- a mismatched file now lands in a different bucket and
        # becomes a sibling instead of aborting the source.
        #
        # Exception policy: split "could not read the bytes" from "the bytes are
        # not a valid image". OSError (a missing file, or a cloud recall that
        # fails on a network blip) is a transport failure -> re-raise, so it
        # surfaces as a retryable error instead of silently shrinking the stack.
        # Any other exception means the file opened at the I/O level but is not a
        # stackable image (TiffFileError, or a corrupt header that makes
        # ``pages[0]`` raise IndexError / a parse error) -> demote to a sibling.
        # Tile info is captured here from the first-seen member of each bucket --
        # which, since all_tiffs is natural-sorted, is exactly that bucket's
        # members[0] -- so there is no second open of the representative (no extra
        # recall, no time-of-check/use window).
        buckets: Dict[Tuple[Any, ...], List[Path]] = {}
        tile_info: Dict[Tuple[Any, ...], Tuple[bool, int, int]] = {}
        unreadable: List[Path] = []
        for p in all_tiffs:
            try:
                with tifffile.TiffFile(str(p)) as tf:
                    page = tf.pages[0]
                    sig = (
                        (page.shape[0], page.shape[1]),
                        str(page.dtype),
                        len(tf.pages),
                    )
                    if sig not in buckets:  # first member of this bucket == members[0]
                        tile_info[sig] = (
                            bool(page.is_tiled),
                            page.tilewidth,
                            page.tilelength,
                        )
            except OSError:
                raise  # transport / recall failure -- retryable, do not swallow
            except Exception:
                unreadable.append(p)  # not a valid image -- demote to sibling
                continue
            buckets.setdefault(sig, []).append(p)

        if not buckets:
            raise ValueError(f"No readable TIFF files in {directory}")

        # Dominant bucket: most files; tie-broken toward the larger frame (so a
        # thumbnail bucket never wins a tie against the real images), then the
        # signature itself for determinism.
        best_sig, members = max(
            buckets.items(),
            key=lambda kv: (len(kv[1]), kv[0][0][0] * kv[0][0][1], kv[0]),
        )
        (spatial_h, spatial_w), self._dtype, n_pages_per_file = best_sig

        self._tiff_files = members  # members preserve all_tiffs' natural order
        stacked = set(members)
        self._unstacked_files = [p for p in all_tiffs if p not in stacked] + unreadable

        # Tile info captured for members[0] in the bucketing pass above.
        self.is_tiled, tile_w, tile_l = tile_info[best_sig]
        if self.is_tiled:
            self.tile_width = tile_w
            self.tile_length = tile_l
            self._spatial_chunk = [tile_l, tile_w]
        else:
            self._spatial_chunk = [spatial_h, spatial_w]

        # Members share best_sig by construction, so the per-file map needs no
        # re-validation: bucketing already subsumes the old dimension/dtype/page
        # consistency checks.
        self._file_ifd_map = [(p, n_pages_per_file) for p in members]

        n_files = len(members)
        spatial_shape = [spatial_h, spatial_w]
        if n_pages_per_file > 1:
            # Multi-page files: (num_files, pages, Y, X). File axis is opaque;
            # page axis keeps the conventional 'z' default.
            self.full_shape = [n_files, n_pages_per_file] + spatial_shape
            self.chunk_shape = [1, 1] + self._spatial_chunk
            self.dim_labels = dim_labels if dim_labels else ["i", "z", "y", "x"]
        else:
            # Single-page files: (num_files, Y, X). 'i' = opaque file/stack axis.
            self.full_shape = [n_files] + spatial_shape
            self.chunk_shape = [1] + self._spatial_chunk
            self.dim_labels = dim_labels if dim_labels else ["i", "y", "x"]

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
        """Expose per-file provenance so the agent can interpret the file axis.

        The file axis (label ``i``) is an opaque stack of every uniformly-shaped
        TIFF in the directory; its semantic structure (channel / time / site / z
        -- e.g. MetaMorph ``_w/_s/_t`` or ``_red/_green/_blue``) is intentionally
        NOT inferred here. ``files`` lists the stacked members index-aligned to
        axis 0, so a downstream agent can parse the names and reshape / relabel as
        needed. ``unstacked_files`` lists TIFFs in the directory that did not fit
        the dominant shape (thumbnails, projections, unreadable files) -- present
        for completeness so nothing is silently dropped.
        """
        md: Dict[str, Any] = {"files": [p.name for p in self._tiff_files]}
        if self._unstacked_files:
            md["unstacked_files"] = [p.name for p in self._unstacked_files]
        return md


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
